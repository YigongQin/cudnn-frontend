"""
Tests for Grouped GEMM GLU with NVFP4 per-token global_scale_tensor (SM100+)

Validates the new global_scale_tensor parameter in grouped_gemm_glu_wrapper_sm100.
Tests both:
  - global_scale_tensor=None (backward compat, should match existing behavior)
  - global_scale_tensor with per-token FP32 values (NVFP4 use case)
"""

import torch
import pytest
from test_utils import torch_fork_set_rng
from fe_api.test_grouped_gemm_swiglu_utils import (
    grouped_gemm_swiglu_init,
    allocate_grouped_gemm_input_tensors,
    check_ref_grouped_gemm_swiglu,
)


def _nvfp4_config():
    """Return a minimal NVFP4-style configuration dict."""
    return {
        "ab_dtype": torch.float4_e2m1fn_x2,
        "sf_dtype": torch.float8_e4m3fn,
        "sf_vec_size": 16,
        "c_dtype": torch.bfloat16,
        "d_dtype": torch.bfloat16,
        "cd_major": "n",
        "acc_dtype": torch.float32,
        "mma_tiler_mn": (256, 256),
        "cluster_shape_mn": (2, 1),
        "vector_f32": False,
        "discrete_col_sfd": False,
    }


# ---------------------------------------------------------------------------
#  Test: global_scale_tensor=None (backward compatibility)
# ---------------------------------------------------------------------------


@pytest.mark.L0
@torch_fork_set_rng(seed=42)
def test_global_scale_none_matches_baseline(request):
    """Passing global_scale_tensor=None should produce identical results
    to the existing code path (no global scale applied)."""
    try:
        from cudnn import grouped_gemm_glu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("cudnn optional dependencies not installed")

    nvfp4_cfg = _nvfp4_config()
    cfg = grouped_gemm_swiglu_init(
        request,
        ab_dtype=nvfp4_cfg["ab_dtype"],
        c_dtype=nvfp4_cfg["c_dtype"],
        d_dtype=nvfp4_cfg["d_dtype"],
        cd_major=nvfp4_cfg["cd_major"],
        acc_dtype=nvfp4_cfg["acc_dtype"],
        mma_tiler_mn=nvfp4_cfg["mma_tiler_mn"],
        cluster_shape_mn=nvfp4_cfg["cluster_shape_mn"],
        sf_vec_size=nvfp4_cfg["sf_vec_size"],
        sf_dtype=nvfp4_cfg["sf_dtype"],
        vector_f32=nvfp4_cfg["vector_f32"],
        discrete_col_sfd=nvfp4_cfg["discrete_col_sfd"],
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    inputs = allocate_grouped_gemm_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        l=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
    )

    try:
        outputs = grouped_gemm_glu_wrapper_sm100(
            a_tensor=inputs["a_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            b_tensor=inputs["b_tensor"],
            sfb_tensor=inputs["sfb_tensor"],
            norm_const_tensor=inputs.get("norm_const_tensor"),
            prob_tensor=inputs.get("prob_tensor"),
            global_scale_tensor=None,  # <-- backward compat
            acc_dtype=cfg["acc_dtype"],
            c_dtype=cfg["c_dtype"],
            d_dtype=cfg["d_dtype"],
            cd_major=cfg["cd_major"],
            mma_tiler_mn=cfg["mma_tiler_mn"],
            cluster_shape_mn=cfg["cluster_shape_mn"],
            sf_vec_size=cfg["sf_vec_size"],
            vector_f32=cfg["vector_f32"],
            m_aligned=cfg["m_aligned"],
            discrete_col_sfd=cfg["discrete_col_sfd"],
            act_func="swiglu",
            current_stream=stream,
        )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    check_ref_grouped_gemm_swiglu(inputs, outputs, cfg, skip_ref=cfg["skip_ref"])


# ---------------------------------------------------------------------------
#  Test: global_scale_tensor = ones (should match no-global-scale path)
# ---------------------------------------------------------------------------


@pytest.mark.L0
@torch_fork_set_rng(seed=42)
def test_global_scale_ones_matches_no_scale(request):
    """Passing global_scale_tensor=ones should produce the same result as None."""
    try:
        from cudnn import grouped_gemm_glu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("cudnn optional dependencies not installed")

    nvfp4_cfg = _nvfp4_config()
    cfg = grouped_gemm_swiglu_init(
        request,
        ab_dtype=nvfp4_cfg["ab_dtype"],
        c_dtype=nvfp4_cfg["c_dtype"],
        d_dtype=nvfp4_cfg["d_dtype"],
        cd_major=nvfp4_cfg["cd_major"],
        acc_dtype=nvfp4_cfg["acc_dtype"],
        mma_tiler_mn=nvfp4_cfg["mma_tiler_mn"],
        cluster_shape_mn=nvfp4_cfg["cluster_shape_mn"],
        sf_vec_size=nvfp4_cfg["sf_vec_size"],
        sf_dtype=nvfp4_cfg["sf_dtype"],
        vector_f32=nvfp4_cfg["vector_f32"],
        discrete_col_sfd=nvfp4_cfg["discrete_col_sfd"],
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    inputs = allocate_grouped_gemm_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        l=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
    )

    valid_m = inputs["valid_m"]

    def _run(gs_tensor):
        return grouped_gemm_glu_wrapper_sm100(
            a_tensor=inputs["a_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            b_tensor=inputs["b_tensor"],
            sfb_tensor=inputs["sfb_tensor"],
            norm_const_tensor=inputs.get("norm_const_tensor"),
            prob_tensor=inputs.get("prob_tensor"),
            global_scale_tensor=gs_tensor,
            acc_dtype=cfg["acc_dtype"],
            c_dtype=cfg["c_dtype"],
            d_dtype=cfg["d_dtype"],
            cd_major=cfg["cd_major"],
            mma_tiler_mn=cfg["mma_tiler_mn"],
            cluster_shape_mn=cfg["cluster_shape_mn"],
            sf_vec_size=cfg["sf_vec_size"],
            vector_f32=cfg["vector_f32"],
            m_aligned=cfg["m_aligned"],
            discrete_col_sfd=cfg["discrete_col_sfd"],
            act_func="swiglu",
            current_stream=stream,
        )

    try:
        out_none = _run(None)
        gs_ones = torch.ones((valid_m, 1, 1), dtype=torch.float32, device="cuda")
        out_ones = _run(gs_ones)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    # With global_scale=ones, D output should match global_scale=None exactly
    torch.testing.assert_close(
        out_ones["d_tensor"],
        out_none["d_tensor"],
        atol=0,
        rtol=0,
        msg="global_scale=ones should produce identical output to global_scale=None",
    )


# ---------------------------------------------------------------------------
#  Test: global_scale_tensor with non-trivial per-token values
# ---------------------------------------------------------------------------


@pytest.mark.L0
@torch_fork_set_rng(seed=42)
def test_global_scale_scales_output(request):
    """Verify that global_scale_tensor correctly scales the output.
    If we run with global_scale=2.0 for all tokens, the C tensor (pre-activation)
    should be ~2x the C tensor from global_scale=None."""
    try:
        from cudnn import grouped_gemm_glu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("cudnn optional dependencies not installed")

    nvfp4_cfg = _nvfp4_config()
    cfg = grouped_gemm_swiglu_init(
        request,
        ab_dtype=nvfp4_cfg["ab_dtype"],
        c_dtype=nvfp4_cfg["c_dtype"],
        d_dtype=nvfp4_cfg["d_dtype"],
        cd_major=nvfp4_cfg["cd_major"],
        acc_dtype=nvfp4_cfg["acc_dtype"],
        mma_tiler_mn=nvfp4_cfg["mma_tiler_mn"],
        cluster_shape_mn=nvfp4_cfg["cluster_shape_mn"],
        sf_vec_size=nvfp4_cfg["sf_vec_size"],
        sf_dtype=nvfp4_cfg["sf_dtype"],
        vector_f32=nvfp4_cfg["vector_f32"],
        discrete_col_sfd=nvfp4_cfg["discrete_col_sfd"],
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    inputs = allocate_grouped_gemm_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        l=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
    )

    # Force alpha=1 and prob=1 so the only scaling comes from global_scale
    valid_m = inputs["valid_m"]
    inputs["alpha_tensor"] = torch.ones(cfg["l"], dtype=torch.float32, device="cuda")
    inputs["prob_tensor"] = torch.ones(
        (inputs["tensor_m"], 1, 1), dtype=torch.float32, device="cuda"
    )

    scale_factor = 2.0

    def _run(gs_tensor):
        return grouped_gemm_glu_wrapper_sm100(
            a_tensor=inputs["a_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            b_tensor=inputs["b_tensor"],
            sfb_tensor=inputs["sfb_tensor"],
            norm_const_tensor=inputs.get("norm_const_tensor"),
            prob_tensor=inputs["prob_tensor"],
            global_scale_tensor=gs_tensor,
            acc_dtype=cfg["acc_dtype"],
            c_dtype=cfg["c_dtype"],
            d_dtype=cfg["d_dtype"],
            cd_major=cfg["cd_major"],
            mma_tiler_mn=cfg["mma_tiler_mn"],
            cluster_shape_mn=cfg["cluster_shape_mn"],
            sf_vec_size=cfg["sf_vec_size"],
            vector_f32=cfg["vector_f32"],
            m_aligned=cfg["m_aligned"],
            discrete_col_sfd=cfg["discrete_col_sfd"],
            act_func="swiglu",
            current_stream=stream,
        )

    try:
        out_base = _run(None)
        gs_scaled = torch.full(
            (valid_m, 1, 1), scale_factor, dtype=torch.float32, device="cuda"
        )
        out_scaled = _run(gs_scaled)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    # The C tensor stores the pre-activation accumulator (acc * alpha * global_scale).
    # With alpha=1 and global_scale=2, the C tensor should be 2x the baseline.
    c_base = out_base["c_tensor"].float()
    c_scaled = out_scaled["c_tensor"].float()

    # Mask out zero elements (padding) to avoid 0/0
    nonzero_mask = c_base.abs() > 1e-6
    if nonzero_mask.any():
        ratio = c_scaled[nonzero_mask] / c_base[nonzero_mask]
        torch.testing.assert_close(
            ratio,
            torch.full_like(ratio, scale_factor),
            atol=1e-2,
            rtol=1e-2,
            msg=f"C tensor should be {scale_factor}x with global_scale={scale_factor}",
        )
