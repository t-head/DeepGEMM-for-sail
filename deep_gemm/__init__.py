import os
import subprocess
import torch

# PPU SDK path: env PPU_SDK > env PPU_HOME > default
def _get_ppu_home():
    for env_key in ('PPU_SDK', 'PPU_HOME'):
        val = os.environ.get(env_key)
        if val:
            return val
    return '/usr/local/PPU_SDK'

PPU_HOME = _get_ppu_home()

from . import jit
from . import deep_gemm_tuner
from . import deep_gemm_cpp
from .jit_kernels import (
    gemm_fp4_fp4_bf16_nt,
    tf32_hc_prenorm_gemm,
    gemm_fp8_fp8_bf16_nt,
    gemm_int8_int8_bf16_nt,
    m_grouped_gemm_fp4_fp4_bf16_nt_masked,
    m_grouped_gemm_fp4_fp4_bf16_nt_nopad,
    m_grouped_gemm_w4a16_masked,
    m_grouped_gemm_w4a16_nopad,
    m_grouped_gemm_w4a16_fused,
    preprocess_mxfp4_scales,
    uint8_padding,
    # m_grouped_gemm_fp8_fp8_bf16_nt_contiguous,
    # m_grouped_gemm_fp8_fp8_bf16_nt_masked,
    # m_grouped_gemm_fp8_fp8_bf16_nt_nopad,
    ceil_div,
    set_num_sms, get_num_sms,
    get_col_major_tma_aligned_tensor,
    get_col_major_tensor,
    get_m_alignment_for_contiguous_layout,
    #fused kernel
    moe_align_block_size,
    m_grouped_gemm_bf16_bf16_bf16_nt_fused,
    m_grouped_gemm_fp8_fp8_bf16_nt_fused,
    m_grouped_gemm_int8_int8_bf16_nt_fused,
    # Attention kernels
    get_paged_mqa_logits_metadata,
    fp8_mqa_logits,
    bf16_mqa_logits,
    int8_mqa_logits,
    fp8_paged_mqa_logits,
    bf16_paged_mqa_logits,
    int8_paged_mqa_logits,
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    fp8_einsum,
    int8_einsum,
)

from .utils import (
    bench,
    bench_kineto,
    calc_diff,
    transform_sf_into_required_layout,
)

from .jit import set_compile_mode, get_compile_mode
# Import functions from the CPP module

from .deep_gemm_cpp import (
    gemm_bf16_bf16_bf16_nt,
    # gemm_fp8_fp8_bf16_nt,
    # gemm_int8_int8_bf16_nt,
    m_grouped_gemm_bf16_bf16_bf16_nt_masked,
    m_grouped_gemm_bf16_bf16_bf16_nt_contiguous,
    m_grouped_gemm_bf16_bf16_bf16_nt_nopad,
    m_grouped_gemm_fp8_fp8_bf16_nt_contiguous,
    m_grouped_gemm_fp8_fp8_bf16_nt_masked,
    m_grouped_gemm_fp8_fp8_bf16_nt_nopad,
    m_grouped_gemm_int8_int8_bf16_nt_masked,
    m_grouped_gemm_int8_int8_bf16_nt_contiguous,
    m_grouped_gemm_int8_int8_bf16_nt_nopad,
)

deep_gemm_cpp.init(
    os.path.dirname(os.path.abspath(__file__)), # Library root directory path
    PPU_HOME         # PPU SDK home
)

use_cpp_jit_for_python = os.environ.get('USE_CPP_JIT_FOR_PYTHON', '').lower()
should_init_deep_gemm_cpp = use_cpp_jit_for_python in ('1', 'true', 'yes', 'on')
if should_init_deep_gemm_cpp:
    # from .deep_gemm_cpp import (
    #     set_num_sms,
    #     get_num_sms,
    #     set_tc_util,
    #     get_tc_util,
    # )
    # DeepGEMM Kernels
    from .deep_gemm_cpp import (
        # FP8 GEMMs
        # gemm_fp8_fp8_bf16_nt,
        # fp8_gemm_nn,
        # fp8_gemm_tn, fp8_gemm_tt,
        # m_grouped_gemm_fp8_fp8_bf16_nt_contiguous,
        # m_grouped_gemm_fp8_fp8_bf16_nt_masked,
        # m_grouped_gemm_fp8_fp8_bf16_nt_nopad,
        # fp8_gemm_nt_skip_head_mid,
        gemm_fp8_fp8_bf16_nt,
        gemm_int8_int8_bf16_nt,
        m_grouped_gemm_fp4_fp4_bf16_nt_masked,
        m_grouped_gemm_fp4_fp4_bf16_nt_nopad,
        gemm_fp4_fp4_bf16_nt,
    )

# Some alias for APIs
fp8_gemm_nt = gemm_fp8_fp8_bf16_nt
fp8_m_grouped_gemm_nt_masked = m_grouped_gemm_fp8_fp8_bf16_nt_masked
m_grouped_fp8_gemm_nt_contiguous = m_grouped_gemm_fp8_fp8_bf16_nt_contiguous
get_mn_major_tma_aligned_tensor = get_col_major_tma_aligned_tensor
