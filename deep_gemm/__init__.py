import os
import subprocess
import torch

# SDK root: env PPU_SDK > env PPU_HOME > default
PPU_HOME = os.environ.get('PPU_SDK') or os.environ.get('PPU_HOME') or '/usr/local/PPU_SDK'

from . import jit
from . import deep_gemm_tuner
from . import deep_gemm_cpp

# Configs
from .deep_gemm_cpp import (
    set_num_sms,
    get_num_sms,
    set_tc_util,
    get_tc_util,
    set_ignore_compile_dims,
    set_block_size_multiple_of,
    set_pdl,
    get_pdl,
    set_compile_mode,
    get_compile_mode,
)

# Layout utilities
from .deep_gemm_cpp import (
    get_mn_major_tma_aligned_tensor,
    get_col_major_tensor,
    get_mk_alignment_for_contiguous_layout,
)

# acblasLt GEMMs
from .deep_gemm_cpp import (
    acblaslt_gemm_nt, acblaslt_gemm_nn,
    acblaslt_gemm_tn, acblaslt_gemm_tt,
)

# DeepGEMM Kernels
from .deep_gemm_cpp import (
    # BF16 GEMMs
    gemm_bf16_bf16_bf16_nt,
    gemm_bf16_bf16_bf16_nn,
    gemm_bf16_bf16_bf16_tn,
    gemm_bf16_bf16_bf16_tt,
    m_grouped_gemm_bf16_bf16_bf16_nt_contiguous,
    m_grouped_gemm_bf16_bf16_bf16_nt_masked,
    m_grouped_gemm_bf16_bf16_bf16_nt_nopad,
    m_grouped_gemm_bf16_bf16_bf16_nt_fused,
    # INT8 GEMMs
    gemm_int8_int8_bf16_nt,
    m_grouped_gemm_int8_int8_bf16_nt_contiguous,
    m_grouped_gemm_int8_int8_bf16_nt_masked,
    m_grouped_gemm_int8_int8_bf16_nt_nopad,
    m_grouped_gemm_int8_int8_bf16_nt_fused,
    # FP8 GEMMs
    gemm_fp8_fp8_bf16_nt,
    gemm_fp8_fp8_bf16_nn,
    gemm_fp8_fp8_bf16_tn,
    gemm_fp8_fp8_bf16_tt,
    m_grouped_gemm_fp8_fp8_bf16_nt_contiguous,
    m_grouped_gemm_fp8_fp8_bf16_nt_masked,
    m_grouped_gemm_fp8_fp8_bf16_nt_nopad,
    m_grouped_gemm_fp8_fp8_bf16_nt_fused,
    # FP4 GEMMs
    gemm_fp4_fp4_bf16_nt,
    m_grouped_gemm_fp4_fp4_bf16_nt_masked,
    m_grouped_gemm_fp4_fp4_bf16_nt_nopad,
    m_grouped_gemm_fp4_fp4_bf16_nt_fused,
    # MoE kernels
    moe_align_block_size,
    # FP4 scale / weight preprocessing
    preprocess_mxfp4_scales,
    preprocess_mxfp4_weight_for_act_and_quant_fusing,
    # W4A16 / W4FA16 GEMMs
    m_grouped_gemm_w4a16_nopad,
    m_grouped_gemm_w4a16_masked,
    m_grouped_gemm_w4a16_fused,
    # TF32 hyperconnection kernels
    tf32_hc_prenorm_gemm,
    # Einsum kernels
    fp8_einsum,
    int8_einsum,
    # Attention kernels (MQA logits)
    get_paged_mqa_logits_metadata,
    bf16_mqa_logits,
    fp8_mqa_logits,
    fp8_mqa_avg_logits,
    int8_mqa_logits,
    fp8_fp4_mqa_logits,
    # Attention kernels (paged MQA logits)
    bf16_paged_mqa_logits,
    fp8_paged_mqa_logits,
    int8_paged_mqa_logits,
    fp8_paged_mqa_avg_logits,
    fp8_fp4_paged_mqa_logits,
)

deep_gemm_cpp.init(
    os.path.dirname(os.path.abspath(__file__)), # Library root directory path
    PPU_HOME         # SDK root
)


# Unimplemented kernels are instead routed to a unified `unimplemented` function below.
# from .deep_gemm_cpp import (
#     # K-Grouped
#     k_grouped_fp8_gemm_nt_contiguous,
#     k_grouped_fp8_gemm_tn_contiguous,
#     k_grouped_bf16_gemm_tn_contiguous,

#     # a8w4 M-Grouped NN
#     m_grouped_fp8_fp4_gemm_nn_contiguous,

#     # M-grouped NN
#     m_grouped_fp8_gemm_nn_contiguous,
#     m_grouped_bf16_gemm_nn_contiguous,
# )

def unimplemented(*args, **kwargs):
    raise NotImplementedError(
        'This kernel is not yet ported to DeepGEMM (PPU build)')

# API names for kernels not yet ported to the PPU build -> unimplemented
k_grouped_fp8_gemm_nt_contiguous = unimplemented
k_grouped_fp8_gemm_tn_contiguous = unimplemented
k_grouped_bf16_gemm_tn_contiguous = unimplemented
m_grouped_fp8_gemm_nn_contiguous = unimplemented
fp8_gemm_nt_skip_head_mid = unimplemented

# Backward-compatible aliases
get_col_major_tma_aligned_tensor = get_mn_major_tma_aligned_tensor
get_m_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout

# ---------------------------------------------------------------------------
# The latest APIs compatibility aliases
# ---------------------------------------------------------------------------

# --- BF16 dense GEMM ---
bf16_gemm_nt = gemm_bf16_bf16_bf16_nt
bf16_gemm_nn = gemm_bf16_bf16_bf16_nn
bf16_gemm_tn = gemm_bf16_bf16_bf16_tn
bf16_gemm_tt = gemm_bf16_bf16_bf16_tt

# --- BF16 M-grouped GEMM ---
m_grouped_bf16_gemm_nt_contiguous = m_grouped_gemm_bf16_bf16_bf16_nt_contiguous
m_grouped_bf16_gemm_nt_masked = m_grouped_gemm_bf16_bf16_bf16_nt_masked
m_grouped_bf16_gemm_nt_nopad = m_grouped_gemm_bf16_bf16_bf16_nt_nopad
m_grouped_bf16_gemm_nt_fused = m_grouped_gemm_bf16_bf16_bf16_nt_fused

# --- INT8 dense GEMM ---
int8_gemm_nt = gemm_int8_int8_bf16_nt

# --- INT8 M-grouped GEMM ---
m_grouped_int8_gemm_nt_contiguous = m_grouped_gemm_int8_int8_bf16_nt_contiguous
m_grouped_int8_gemm_nt_masked = m_grouped_gemm_int8_int8_bf16_nt_masked
m_grouped_int8_gemm_nt_nopad = m_grouped_gemm_int8_int8_bf16_nt_nopad
m_grouped_int8_gemm_nt_fused = m_grouped_gemm_int8_int8_bf16_nt_fused

# --- FP8 dense GEMM ---
# The latest APIs return FP32 for results
fp8_gemm_nt = gemm_fp8_fp8_bf16_nt
fp8_gemm_nn = gemm_fp8_fp8_bf16_nn
fp8_gemm_tn = gemm_fp8_fp8_bf16_tn
fp8_gemm_tt = gemm_fp8_fp8_bf16_tt

# --- FP8 M-grouped GEMM
m_grouped_fp8_gemm_nt_contiguous = m_grouped_gemm_fp8_fp8_bf16_nt_contiguous
m_grouped_fp8_gemm_nt_masked = m_grouped_gemm_fp8_fp8_bf16_nt_masked
m_grouped_fp8_gemm_nt_nopad = m_grouped_gemm_fp8_fp8_bf16_nt_nopad
m_grouped_fp8_gemm_nt_fused = m_grouped_gemm_fp8_fp8_bf16_nt_fused

fp8_m_grouped_gemm_nt_masked = m_grouped_gemm_fp8_fp8_bf16_nt_masked

# --- FP4 dense GEMM ---
fp4_gemm_nt = gemm_fp4_fp4_bf16_nt

# --- FP4 M-grouped GEMM
m_grouped_fp4_gemm_nt_masked = m_grouped_gemm_fp4_fp4_bf16_nt_masked
m_grouped_fp4_gemm_nt_nopad = m_grouped_gemm_fp4_fp4_bf16_nt_nopad
m_grouped_fp4_gemm_nt_fused = m_grouped_gemm_fp4_fp4_bf16_nt_fused

# --- FP8 FP4 dense GEMM ---
fp8_fp4_gemm_nt = unimplemented
fp8_fp4_gemm_nn = unimplemented
fp8_fp4_gemm_tn = unimplemented
fp8_fp4_gemm_tt = unimplemented

# --- FP8 FP4 M-grouped GEMM ---
m_grouped_fp8_fp4_gemm_nt_contiguous = unimplemented
m_grouped_fp8_fp4_gemm_nn_contiguous = unimplemented
m_grouped_fp8_fp4_gemm_nt_masked = unimplemented

# Some utils
from . import testing
from . import utils
from .utils import *
from .testing import bench_kineto, calc_diff

# Legacy Triton kernels
try:
    from . import legacy
except Exception as e:
    print(f'Failed to load legacy DeepGEMM Triton kernels: {e}')
