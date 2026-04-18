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
  - Global scale: alpha_tensor = ones (no per-token scaling)
  - norm_const:   ones (no normalization)
```

### Target NVFP4 Per-Token Path
```
Quantization: NVFP4 block scaling with per-token global scale
  - Data dtype:  float4_e2m1fn_x2
  - Scale dtype: float8_e4m3fn
  - sf_vec_size: 16 (one scale per 16 elements in K)
  - Global scale: alpha_tensor = per-token FP32 scale (shape: (num_groups,) or per-token)
  - norm_const:   1 / (fp8_max * fp4_max) = 1 / (448 * 6) = 1/2688
```

### Key Difference
MXFP8 uses `float8_e8m0fnu` scales (pure exponent, no mantissa) with sf_vec_size=32.
NVFP4 uses `float8_e4m3fn` scales (has mantissa) with sf_vec_size=16, plus a **per-token FP32 global scale** that captures the dynamic range of each token's activation row. The cuDNN kernel's `alpha_tensor` parameter carries this per-token global scale.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│  TransformerEngine (forward_grouped_mlp.py)               │
│                                                           │
│  Input (BF16) ──► NVFP4 Quantize ──► grouped_gemm_glu   │
│                   (per-token scale)    wrapper_sm100()    │
│                                        sf_vec_size=16     │
│                                        sf_dtype=e4m3fn    │
│                                        alpha=per_token    │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│  cuDNN Frontend (grouped_gemm_glu/api.py)                 │
│                                                           │
│  Same kernel, different config:                           │
│  - ab_dtype = float4_e2m1fn_x2                           │
│  - sf_dtype = float8_e4m3fn (not e8m0fnu)                │
│  - sf_vec_size = 16                                      │
│  - alpha_tensor = per-token global scales                │
│  - norm_const_tensor = 1/2688                            │
└──────────────────────────────────────────────────────────┘
```

---

## Phase 1: cuDNN Frontend Changes

The cuDNN Frontend kernel **already supports** FP4 + sf_vec_size=16 + sf_dtype=float8_e4m3fn. The main work is verification and minor adjustments.

### Files to Verify / Modify

| File | What | Action |
|------|------|--------|
| `python/cudnn/grouped_gemm/grouped_gemm_glu/api.py` | Unified GLU wrapper | **Verify** FP4+sf_vec_size=16+e4m3fn path works with per-token alpha. Check `alpha_tensor` shape flexibility (currently per-group; need per-token or confirm per-group suffices). |
| `python/cudnn/grouped_gemm/grouped_gemm_glu/moe_blockscaled_grouped_gemm_glu_bias.py` | Kernel implementation | **Verify** how `alpha_tensor` is applied — is it per-expert or can it be per-token? If per-expert only, this is the main kernel change needed. |
| `python/cudnn/grouped_gemm/grouped_gemm_quant/api.py` | FC2 quant wrapper | **Verify** same NVFP4 config works for FC2 path. |
| `python/cudnn/grouped_gemm/moe_kernel_helpers.py` | Helper functions | **Verify** sf_vec_size=16 + FP4 + e4m3fn scale factor path. Line 538 has a check for FP8+sf_vec_size=16 but FP4+16 should pass. |
| `python/cudnn/grouped_gemm/utils.py` | NVFP4 tensor shape util | **Verify** `logical_shape_for_fp4` (line 230) handles per-token shapes. |

### Key Question: `alpha_tensor` Granularity

**Current behavior:** `alpha_tensor` shape is `(num_groups,)` — one scalar per expert.

**Per-token NVFP4 requires:** One global scale per token (row of A), shape `(valid_m, 1)` or equivalent.

**Investigation needed:** Read the kernel code to determine if `alpha_tensor` can be reshaped to per-token granularity, or if `norm_const_tensor` can absorb this. If neither works, the kernel needs a new parameter or the per-token scale must be folded into `sfa_tensor`.

**Likely approach:** The NVFP4 hierarchical scaling `x = x_e2m1 * block_scale_e4m3 * global_scale_f32` can be achieved by:
- `block_scale_e4m3` → goes into `sfa_tensor` (already per-block)
- `global_scale_f32` → per-token, could be folded into `sfa_tensor` during quantization on the TE side, OR passed via a reshaped `alpha_tensor`/`prob_tensor`

### New Test File

| File | Action |
|------|--------|
| `test/python/fe_api/test_grouped_gemm_glu_nvfp4.py` | **Create** — dedicated NVFP4 per-token tests using the GLU wrapper with sf_vec_size=16, sf_dtype=e4m3fn, FP4 inputs, and non-trivial alpha_tensor values. |

---

## Phase 2: TransformerEngine Changes

### New Fused Op Class

| File | What | Action |
|------|------|--------|
| `transformer_engine/pytorch/ops/fused/forward_grouped_mlp.py` | Fused MLP ops | **Add** new class `ForwardGroupedMLP_CuTeGEMMSwiGLU_NVFP4` modeled after `ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8`. Key differences: (1) NVFP4 quantization instead of MXFP8, (2) sf_vec_size=16, (3) e4m3fn scale dtype, (4) per-token global scale via alpha_tensor. |

### NVFP4 Quantization Integration

| File | What | Action |
|------|------|--------|
| `transformer_engine/pytorch/tensor/nvfp4_tensor.py` | NVFP4 tensor/quantizer | **Verify** `NVFP4Quantizer` produces the right scale layout. May need a helper to extract per-token global scale + per-block e4m3fn scales separately for the cuDNN kernel. |
| `transformer_engine/pytorch/tensor/utils.py` | Tensor utilities | **Possibly modify** `group_quantize` path to support NVFP4 grouped quantization for activations. |
| `transformer_engine/pytorch/csrc/extensions/cast.cpp` | C++ quantize extension | **Verify** `group_quantize_nvfp4_impl` exists and produces the right output format. |

### Scale Factor Reshaping

| File | What | Action |
|------|------|--------|
| `transformer_engine/pytorch/ops/fused/forward_grouped_mlp.py` | Scale reshape logic | **Add** NVFP4 scale reshaping in the new class. NVFP4 scales are `float8_e4m3fn` with sf_vec_size=16, so the reshape differs from MXFP8's `(1, M//128, K//128, 32, 4, 4)` layout. New layout: `(1, M//128, K//16//4, 32, 4, 4)` permuted to `(32, 4, M//128, 4, K//16//4, 1)`. Also extract per-token `global_scale` from `NVFP4Tensor.amax_rowwise` or equivalent. |

### Routing and Registration

| File | What | Action |
|------|------|--------|
| `transformer_engine/pytorch/ops/fused/forward_grouped_mlp.py` | Op registration | **Add** `ForwardGroupedMLP_CuTeGEMMSwiGLU_NVFP4` to the list of candidate fused ops. Gate behind a new env var `NVTE_CUTEDSL_FUSED_GROUPED_MLP_NVFP4` or reuse the existing one with recipe-based dispatch. |
| `transformer_engine/pytorch/ops/fused/__init__.py` | Exports | **Add** import for the new class. |
| `transformer_engine/pytorch/quantization.py` | Recipe integration | **Verify** `NVFP4BlockScalingRecipeState` (line 1281) can produce quantizers compatible with the fused grouped MLP path. |
| `transformer_engine/pytorch/module/grouped_linear.py` | GroupedLinear module | **Possibly modify** to select between MXFP8 and NVFP4 fused paths based on the quantization recipe. |

### Backward Pass (Not in Scope — Document for Future)

The backward pass needs `grouped_gemm_dglu_wrapper_sm100` and `grouped_gemm_quant_wrapper_sm100` with NVFP4 config. This is deferred but the forward changes should not preclude it.

---

## Phase 3: Testing & Validation

### cuDNN Frontend Tests

| Test | Description |
|------|-------------|
| `test_grouped_gemm_glu_nvfp4.py` | NVFP4 FP4 forward with sf_vec_size=16, e4m3fn scales, per-token alpha_tensor. Both dense and discrete modes. Numerical accuracy vs BF16 reference. |

### TransformerEngine Tests

| Test | Description |
|------|-------------|
| Unit test in `test/pytorch/test_grouped_linear.py` | Add NVFP4 recipe variant to existing grouped linear tests. Verify forward pass produces correct output. |
| Integration test | End-to-end MoE forward with NVFP4 quantization recipe, comparing against MXFP8 and BF16 baselines. |

---

## File Change Summary

### cuDNN Frontend (`cudnn-frontend/`)

| File | Change Type | Description |
|------|-------------|-------------|
| `python/cudnn/grouped_gemm/grouped_gemm_glu/api.py` | Verify / Minor | Confirm FP4+sf_vec_size=16+e4m3fn config. Check alpha_tensor shape handling for per-token. |
| `python/cudnn/grouped_gemm/grouped_gemm_glu/moe_blockscaled_grouped_gemm_glu_bias.py` | Verify / Possibly modify | Check how alpha is applied in the kernel. If per-expert only, need per-token support. |
| `python/cudnn/grouped_gemm/grouped_gemm_quant/api.py` | Verify | FC2 path with same NVFP4 config. |
| `python/cudnn/grouped_gemm/moe_kernel_helpers.py` | Verify | sf_vec_size=16 + FP4 validation. |
| `test/python/fe_api/test_grouped_gemm_glu_nvfp4.py` | **New** | NVFP4-specific test cases. |

### TransformerEngine (`TransformerEngine/`)

| File | Change Type | Description |
|------|-------------|-------------|
| `transformer_engine/pytorch/ops/fused/forward_grouped_mlp.py` | **Major add** | New `ForwardGroupedMLP_CuTeGEMMSwiGLU_NVFP4` class (~200 lines). |
| `transformer_engine/pytorch/ops/fused/__init__.py` | Minor | Export new class. |
| `transformer_engine/pytorch/ops/fused/backward_grouped_mlp.py` | No change | Backward not in scope. |
| `transformer_engine/pytorch/tensor/nvfp4_tensor.py` | Verify / Minor | May need helper to split global scale from block scales. |
| `transformer_engine/pytorch/tensor/utils.py` | Verify / Minor | Ensure NVFP4 grouped quantize produces cuDNN-compatible layout. |
| `transformer_engine/pytorch/module/grouped_linear.py` | Minor | Route to NVFP4 fused op when recipe is NVFP4. |
| `transformer_engine/pytorch/quantization.py` | Verify | NVFP4BlockScalingRecipeState compatibility. |
| `test/pytorch/test_grouped_linear.py` | Add cases | NVFP4 forward pass tests. |

---

## Execution Order

1. **Investigate** alpha_tensor granularity in cuDNN kernel (is it per-expert or can it be per-token?)
2. **Investigate** NVFP4 quantization output format in TE — how are global_scale and block_scales separated?
3. **Write cuDNN test** (`test_grouped_gemm_glu_nvfp4.py`) with hardcoded NVFP4 tensors to validate the kernel config works
4. **If kernel change needed:** modify cuDNN kernel to support per-token alpha
5. **Write TE fused op** class `ForwardGroupedMLP_CuTeGEMMSwiGLU_NVFP4`
6. **Wire up** recipe-based dispatch in TE's GroupedLinear
7. **End-to-end test** in TE

---

## Open Questions

1. **alpha_tensor shape:** Can the cuDNN kernel accept per-token (valid_m,) alpha instead of per-expert (num_groups,)? If not, can the per-token global scale be folded into sfa_tensor during quantization?
2. **NVFP4 scale layout:** Does TE's `NVFP4Quantizer` output scales in the 6D `(32, 4, M//128, 4, K//16//4, 1)` layout cuDNN expects, or does it need reshaping?
3. **norm_const_tensor:** For NVFP4 the normalization constant is `1/(fp8_max * fp4_max) = 1/2688`. Is this the right place to pass it, or should it be baked into the global scale?
4. **Output quantization:** The FC1→FC2 handoff currently produces MXFP8 output (d_dtype=float8_e4m3fn). Should FC1 output also be NVFP4 (d_dtype=float4_e2m1fn_x2), or stay FP8 for FC2 input?
