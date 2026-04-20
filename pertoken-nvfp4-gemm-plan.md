# Per-Token NVFP4 Grouped GEMM — Implementation Plan

## Goal

Add a **per-token NVFP4** forward-pass grouped GEMM to cuDNN Frontend, and expose it in TransformerEngine as an option alongside the existing MXFP8 fused grouped MLP path.

**Scope:** Forward pass only. NVFP4 (FP4 E2M1 data + FP8 E4M3 block scales + FP32 per-token global scale). Dense weight mode (matching TE's `single_grouped_weight=True` path initially).

---

## Background: Current vs. Target

### Current MXFP8 Path (what TE uses today)
```
Quantization: MXFP8 block scaling
  - Data dtype:  float8_e4m3fn
  - Scale dtype: float8_e8m0fnu
  - sf_vec_size: 32 (one scale per 32 elements in K)
  - alpha_tensor: ones, shape (num_groups,) — per-expert, unused
  - norm_const:   ones — unused
  - No per-token global scale
```

### Target NVFP4 Per-Token Path
```
Quantization: NVFP4 block scaling with per-token global scale
  - Data dtype:  float4_e2m1fn_x2
  - Scale dtype: float8_e4m3fn
  - sf_vec_size: 16 (one scale per 16 elements in K)
  - alpha_tensor: ones, shape (num_groups,) — per-expert, unchanged
  - norm_const:   1 / (fp8_max * fp4_max) = 1/2688
  - NEW: global_scale_tensor: FP32, shape (valid_m, 1, 1) — per-token global scale
```

### Key Difference
MXFP8 uses `float8_e8m0fnu` scales (pure exponent, no mantissa) with sf_vec_size=32.
NVFP4 uses `float8_e4m3fn` scales (has mantissa) with sf_vec_size=16, plus a **per-token FP32 global scale** that captures the dynamic range of each token's activation row.

---

## Design Decision: New `global_scale_tensor` Parameter

### Why not reuse `alpha_tensor`?
- `alpha_tensor` is per-expert `(expert_cnt,)`, loaded once per expert tile via `alpha[expert_idx]`
- Changing it to per-token would break every existing caller and waste memory for MXFP8 (broadcast ones to `(valid_m,)`)
- Semantically different: alpha is a per-expert scaling knob, global_scale is a quantization artifact

### Why not fold into `sfa_tensor`?
- Would require multiplying FP32 global scale into FP8 E4M3 block scales during quantization
- Loses FP32 precision — the whole point of a separate global scale is to preserve dynamic range

### Chosen approach: New `global_scale_tensor` parameter

Add a new optional parameter to both the kernel and the API wrapper:

```python
global_scale_tensor: Optional[torch.Tensor]  # shape: (valid_m, K // group_size, 1), dtype: float32
```

**Design for future generality:**
- **Per-token (NVFP4 today):** shape `(valid_m, 1, 1)` — one FP32 scale per token row, broadcast across K
- **Per-subchannel (future):** shape `(valid_m, K // group_size, 1)` — one FP32 scale per (token, channel_group) pair
- **Per-tensor (degenerate):** shape `(1, 1, 1)` — broadcast everywhere
- When `None` (default): no global scale applied, backward compatible with all existing callers

**Kernel implementation:** Follows the same pattern as `prob_tensor`:
- Shape `(valid_m, S, 1)` where S is the subchannel dimension (1 for per-token)
- Indexed per-token via `mPosition` (same as prob), per-subchannel via the N-subtile index
- Applied as FP32 multiply on accumulator, alongside `alpha_val`:
  `acc = acc * alpha_val * global_scale[mPosition, subchannel_idx]`

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│  TransformerEngine (forward_grouped_mlp.py)                   │
│                                                               │
│  Input (BF16) ──► NVFP4 Quantize ──► grouped_gemm_glu       │
│                   ├─ data (E2M1)      wrapper_sm100()        │
│                   ├─ block_scale       sf_vec_size=16         │
│                   │   (E4M3, per-16)   sf_dtype=e4m3fn       │
│                   └─ global_scale      global_scale_tensor=  │
│                       (FP32, per-row)   (valid_m, 1, 1)      │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  cuDNN Frontend Kernel                                        │
│                                                               │
│  acc = SFA * A * SFB * B            (block-scaled GEMM)      │
│  acc = acc * alpha[expert]          (per-expert, existing)   │
│  acc = acc * global_scale[token,s]  (per-token, NEW)         │
│  out = SwiGLU(acc) * prob[token]    (activation + gating)    │
└──────────────────────────────────────────────────────────────┘
```

---

## Phase 1: cuDNN Frontend Changes

### 1a. Kernel Modification — Add `global_scale_tensor`

| File | Change |
|------|--------|
| `python/cudnn/grouped_gemm/grouped_gemm_glu/moe_blockscaled_grouped_gemm_glu_bias.py` | **Modify kernel** to accept `global_scale` parameter. Load per-token via `mPosition` (same pattern as `prob`). Apply as FP32 multiply on accumulator between alpha and SwiGLU. When `None`/disabled, no-op (zero overhead). |

Kernel change (~20 lines):
```python
# In epilogue, after alpha multiply, before SwiGLU:
if cutlass.const_expr(self.enable_global_scale):
    gs_val = global_scale[mPosition, subchannel_idx, 0]
    for i in cutlass.range_constexpr(cute.size(tTR_rAcc_gate)):
        tTR_rAcc_gate[i] = tTR_rAcc_gate[i] * cutlass.Float32(gs_val)
        tTR_rAcc_up[i] = tTR_rAcc_up[i] * cutlass.Float32(gs_val)
```

### 1b. API Changes — Plumb `global_scale_tensor` Through

| File | Change |
|------|--------|
| `python/cudnn/grouped_gemm/grouped_gemm_glu/api.py` | **Modify** `GroupedGemmGluSm100.__init__` and `grouped_gemm_glu_wrapper_sm100` to accept optional `global_scale_tensor`. Add shape validation: `(valid_m, S, 1)` where S >= 1. Pass to kernel. |
| `python/cudnn/grouped_gemm/grouped_gemm_quant/api.py` | **Same change** for FC2 quant wrapper (`grouped_gemm_quant_wrapper_sm100`). |

### 1c. Verification — Existing FP4 Path

| File | Action |
|------|--------|
| `python/cudnn/grouped_gemm/moe_kernel_helpers.py` | **Verify** sf_vec_size=16 + FP4 + e4m3fn passes validation (line 538 only blocks FP8+16). |
| `python/cudnn/grouped_gemm/utils.py` | **Verify** `logical_shape_for_fp4` (line 230) handles shapes correctly. |

### 1d. New Tests

| File | Action |
|------|--------|
| `test/python/fe_api/test_grouped_gemm_glu_nvfp4.py` | **Create** — NVFP4 tests with: (1) FP4 E2M1 inputs, sf_vec_size=16, E4M3 scales, (2) non-trivial per-token `global_scale_tensor` values, (3) numerical accuracy vs BF16 reference. Test both `global_scale_tensor=None` (backward compat) and per-token shapes. |

---

## Phase 2: TransformerEngine Changes

### 2a. New Fused Op Class

| File | Change |
|------|--------|
| `transformer_engine/pytorch/ops/fused/forward_grouped_mlp.py` | **Add** `ForwardGroupedMLP_CuTeGEMMSwiGLU_NVFP4` (~200 lines), modeled after `_MXFP8` variant. Key differences: |

Differences from MXFP8 class:
1. Use `NVFP4Quantizer` instead of `MXFP8Quantizer` for input quantization
2. `sf_vec_size = 16` (not 32)
3. Scale dtype = `float8_e4m3fn` (not `float8_e8m0fnu`)
4. Extract per-token `global_scale` from NVFP4 quantizer output, pass as `global_scale_tensor`
5. Scale reshape: `(1, M//128, K//16//4, 32, 4, 4)` permuted to `(32, 4, M//128, 4, K//16//4, 1)`
6. Gate behind env var `NVTE_CUTEDSL_FUSED_GROUPED_MLP_NVFP4` or recipe-based dispatch

### 2b. NVFP4 Quantization Integration

| File | Change |
|------|--------|
| `transformer_engine/pytorch/tensor/nvfp4_tensor.py` | **Verify / Minor** — ensure `NVFP4Quantizer` exposes per-token global scale separately from block scales. May need a property or helper to extract `global_scale` as a `(M, 1)` FP32 tensor. |
| `transformer_engine/pytorch/tensor/utils.py` | **Verify** — `group_quantize` NVFP4 path produces cuDNN-compatible layout. |
| `transformer_engine/pytorch/csrc/extensions/cast.cpp` | **Verify** — `group_quantize_nvfp4_impl` output format. |

### 2c. Routing and Registration

| File | Change |
|------|--------|
| `transformer_engine/pytorch/ops/fused/__init__.py` | **Minor** — export new class. |
| `transformer_engine/pytorch/module/grouped_linear.py` | **Minor** — select NVFP4 fused op when quantization recipe is `NVFP4BlockScaling`. |
| `transformer_engine/pytorch/quantization.py` | **Verify** — `NVFP4BlockScalingRecipeState` compatibility. |

### 2d. Backward Pass (Deferred)

Not in scope. The backward pass needs `grouped_gemm_dglu_wrapper_sm100` and `grouped_gemm_quant_wrapper_sm100` with NVFP4 config + `global_scale_tensor`. The forward changes should not preclude adding this later — the new parameter is optional throughout.

---

## Phase 3: Testing & Validation

### cuDNN Frontend Tests

| Test | Description |
|------|-------------|
| `test_grouped_gemm_glu_nvfp4.py` | FP4 forward with sf_vec_size=16, E4M3 scales, per-token global_scale_tensor. Dense mode. Accuracy vs BF16 reference. |
| Existing tests | **Verify no regression** — all existing tests must pass with `global_scale_tensor=None`. |

### TransformerEngine Tests

| Test | Description |
|------|-------------|
| `test/pytorch/test_grouped_linear.py` | Add NVFP4 recipe variant. Forward pass correctness. |
| Integration | End-to-end MoE forward with NVFP4 recipe vs MXFP8 and BF16 baselines. |

---

## File Change Summary

### cuDNN Frontend (`cudnn-frontend/`)

| File | Change Type | Description |
|------|-------------|-------------|
| `grouped_gemm_glu/moe_blockscaled_grouped_gemm_glu_bias.py` | **Modify** | Add `global_scale` parameter to kernel. Per-token load + FP32 multiply on accumulator. ~20 lines. |
| `grouped_gemm_glu/api.py` | **Modify** | Add `global_scale_tensor` to `GroupedGemmGluSm100` and `grouped_gemm_glu_wrapper_sm100`. Shape validation. ~40 lines. |
| `grouped_gemm_quant/api.py` | **Modify** | Same `global_scale_tensor` plumbing for FC2 path. ~30 lines. |
| `test/python/fe_api/test_grouped_gemm_glu_nvfp4.py` | **New** | NVFP4-specific test cases. |

### TransformerEngine (`TransformerEngine/`)

| File | Change Type | Description |
|------|-------------|-------------|
| `transformer_engine/pytorch/ops/fused/forward_grouped_mlp.py` | **Major add** | New `ForwardGroupedMLP_CuTeGEMMSwiGLU_NVFP4` class. ~200 lines. |
| `transformer_engine/pytorch/ops/fused/__init__.py` | Minor | Export new class. |
| `transformer_engine/pytorch/tensor/nvfp4_tensor.py` | Verify / Minor | Expose per-token global scale. |
| `transformer_engine/pytorch/tensor/utils.py` | Verify / Minor | NVFP4 grouped quantize layout. |
| `transformer_engine/pytorch/module/grouped_linear.py` | Minor | Recipe-based fused op selection. |
| `transformer_engine/pytorch/quantization.py` | Verify | Recipe state compatibility. |
| `test/pytorch/test_grouped_linear.py` | Add cases | NVFP4 forward pass tests. |

---

## Execution Order

1. **Add `global_scale_tensor` to cuDNN kernel** (`moe_blockscaled_grouped_gemm_glu_bias.py`) — per-token load following `prob` pattern, FP32 multiply on accumulator
2. **Plumb through cuDNN API** (`grouped_gemm_glu/api.py`, `grouped_gemm_quant/api.py`) — optional parameter, `None` = no-op
3. **Write cuDNN test** (`test_grouped_gemm_glu_nvfp4.py`) — validate FP4+sf_vec_size=16+e4m3fn+global_scale end-to-end
4. **Run existing tests** — verify no regression with `global_scale_tensor=None`
5. **Investigate TE NVFP4 quantizer output** — how to extract global_scale + block_scales + data
6. **Write TE fused op** (`ForwardGroupedMLP_CuTeGEMMSwiGLU_NVFP4`)
7. **Wire up** recipe-based dispatch in TE's GroupedLinear
8. **End-to-end TE test**

---

## Open Questions

1. **NVFP4 scale layout in TE:** Does `NVFP4Quantizer` output block scales in the 6D `(32, 4, M//128, 4, K//16//4, 1)` layout cuDNN expects, or does it need reshaping?
2. **Global scale extraction:** How does TE's NVFP4 quantizer expose the per-token global scale? Is it a separate tensor or embedded in `scale_inv`?
3. **norm_const_tensor:** For NVFP4 the normalization constant should be `1/(fp8_max * fp4_max) = 1/2688`. Confirm this is correct and where to pass it.
4. **FC1→FC2 output dtype:** Currently MXFP8 produces `d_dtype=float8_e4m3fn` for FC2 input. Should NVFP4 path also produce FP8 output, or NVFP4 output with its own global_scale?
5. **Subchannel dimension:** For the initial per-token implementation, `global_scale_tensor` shape is `(valid_m, 1, 1)`. Future subchannel support would use `(valid_m, K//group_size, 1)`. Does the kernel need to handle variable subchannel sizes at compile time or can it be dynamic?
