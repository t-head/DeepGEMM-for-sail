import os
import subprocess
import torch

# SDK root: env PPU_SDK > env PPU_HOME > default
PPU_HOME = os.environ.get('PPU_SDK') or os.environ.get('PPU_HOME') or '/usr/local/PPU_SDK'

from . import jit
from . import deep_gemm_tuner
from . import deep_gemm_cpp

# Benchmarking / correctness utilities
from .utils import (
    bench,
    bench_kineto,
    calc_diff,
    transform_sf_into_required_layout,
)

# Configs
from .deep_gemm_cpp import (
    set_num_sms,
    get_num_sms,
    set_compile_mode,
    get_compile_mode,
)

# Layout utilities
from .deep_gemm_cpp import (
    get_col_major_tma_aligned_tensor,
    get_col_major_tensor,
    get_m_alignment_for_contiguous_layout,
)

# DeepGEMM Kernels
from .deep_gemm_cpp import (
    # BF16 GEMMs
    gemm_bf16_bf16_bf16_nt,
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

# Some aliases for APIs
fp8_gemm_nt = gemm_fp8_fp8_bf16_nt
fp8_m_grouped_gemm_nt_masked = m_grouped_gemm_fp8_fp8_bf16_nt_masked
m_grouped_fp8_gemm_nt_contiguous = m_grouped_gemm_fp8_fp8_bf16_nt_contiguous
get_mn_major_tma_aligned_tensor = get_col_major_tma_aligned_tensor
