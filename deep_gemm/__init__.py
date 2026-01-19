import torch

from . import jit
from . import deep_gemm_tuner
from .jit_kernels import (
    gemm_bf16_bf16_bf16_nt,
    gemm_int8_int8_bf16_nt,
    gemm_fp8_fp8_bf16_nt,
    m_grouped_gemm_fp8_fp8_bf16_nt_contiguous,
    m_grouped_gemm_fp8_fp8_bf16_nt_masked,
    m_grouped_gemm_fp8_fp8_bf16_nt_nopad,
    ceil_div,
    set_num_sms, get_num_sms,
    get_col_major_tma_aligned_tensor,
    get_col_major_tensor,
    get_m_alignment_for_contiguous_layout,
    m_grouped_gemm_int8_int8_bf16_nt_masked,
    m_grouped_gemm_int8_int8_bf16_nt_contiguous,
    m_grouped_gemm_int8_int8_bf16_nt_nopad,
    m_grouped_gemm_bf16_bf16_bf16_nt_masked,
    m_grouped_gemm_bf16_bf16_bf16_nt_contiguous,
    m_grouped_gemm_bf16_bf16_bf16_nt_nopad,
    # Attention kernels
    get_paged_mqa_logits_metadata,
    fp8_mqa_logits,
    bf16_mqa_logits,
    int8_mqa_logits,
    fp8_paged_mqa_logits,
    bf16_paged_mqa_logits,
    int8_paged_mqa_logits,
)

# Some alias for APIs
fp8_m_grouped_gemm_nt_masked = m_grouped_gemm_fp8_fp8_bf16_nt_masked
m_grouped_fp8_gemm_nt_contiguous = m_grouped_gemm_fp8_fp8_bf16_nt_contiguous
get_mn_major_tma_aligned_tensor = get_col_major_tma_aligned_tensor

from .utils import (
    bench,
    bench_kineto,
    calc_diff,
    transform_sf_into_required_layout,
)
from .jit import set_compile_mode, get_compile_mode
