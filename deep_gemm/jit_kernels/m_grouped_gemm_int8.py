import torch
from typing import Tuple

from .gemm_int8 import get_best_configs
from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, get_extra_info, is_ppu1v5_device
from .gemm import get_gemv_best_configs
import os

# C++ code templates
includes = ('"deep_gemm/int8_gemm.cuh"', )
includes_cutlass3 = ('"../deep_gemm/int8_gemm_cutlass3.cuh"', )
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto kNumStages = {NUM_STAGES};

// Make a templated grouped GEMM
using gemm_t = Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, kNumGroups, kNumStages, GemmType::{GEMM_TYPE}>;

// Launch kernel
gemm_t::run(out, grouped_layout,
            m, expected_m, lhs, lhs_scales, rhs, rhs_scales,
            stream, num_sms, smem_size);
"""


includes_gemv = ('"deep_gemm/gemvt.cuh"', )
template_gemv = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto ThreadPerN = {ThreadPerN};
constexpr auto NPerThread = {NPerThread};
constexpr auto NUM_UNROLL = {NUM_UNROLL};
constexpr auto SWZL_SIZE_M = {SWZL_SIZE_M};
constexpr auto USE_SMALL_K = {USE_SMALL_K};
constexpr auto BlockSize = {BlockSize};

// Make a templated grouped GEMM
using gemm_v = Gemvt<int8_t, __nv_bfloat16, N, K, kNumGroups, ThreadPerN, NPerThread, NUM_UNROLL, SWZL_SIZE_M, BlockSize, USE_SMALL_K>;

// Launch kernel
gemm_v::run(out, grouped_layout,
            m, lhs, rhs,
            stream,
            lhs_scales, rhs_scales);
"""

def m_grouped_gemm_int8_int8_bf16_nt_contiguous(lhs: Tuple[torch.Tensor, torch.Tensor],
                                                rhs: Tuple[torch.Tensor, torch.Tensor],
                                                out: torch.Tensor, m_indices: torch.Tensor, configs = None) -> None:
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    m_, n_ = out.shape
    m__ = m_indices.numel()

    # Type and shape checks
    assert m == m_ == m__ and k == k_ and n == n_
    assert lhs_scales.shape == (m, 1)
    assert rhs_scales.shape == (num_groups, n, 1)
    assert lhs.dtype == torch.int8 and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.int8 and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert m_indices.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and m_indices.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    # lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()
    if configs:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(m, n, k, num_groups, num_sms, is_grouped_contiguous=True)
    expected_m = ceil_div(m, num_groups)

    extra_info = get_extra_info()

    args = (lhs, lhs_scales, rhs, rhs_scales, out,
            m_indices, m, expected_m, num_groups,
            torch.cuda.current_stream(), num_sms, smem_config[0])
    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_int8_int8_bf16_nt',
        keys={'N': n, 'K': k,
              'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
              'WARP_M': warp_m, 'WARP_N': warp_n,
              'BLOCK_N_PADDING': smem_config[2],
              'NUM_GROUPS': num_groups, 'NUM_STAGES': num_stages,
              'GEMM_TYPE': 'GroupedContiguous'},
        space=(),
        includes=includes_cutlass3 if extra_info['use_cutlass3'] else includes,
        arg_defs=(('lhs', torch.int8), ('lhs_scales', torch.float),
                  ('rhs', torch.int8), ('rhs_scales', torch.float),
                  ('out', torch.bfloat16),
                  ('grouped_layout', torch.int32), ('m', int),
                  ('num_groups', int), ('expected_m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
        template=template,
        jit_include_dir='cutlass3' if extra_info['use_cutlass3'] else None,
        args=args
    )

    # Run the kernel
    runtime(*args)


def m_grouped_gemm_int8_int8_bf16_nt_masked(lhs: Tuple[torch.Tensor, torch.Tensor],
                                          rhs: Tuple[torch.Tensor, torch.Tensor],
                                          out: torch.Tensor, masked_m: torch.Tensor, expected_m: int, configs = None) -> None:
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    num_groups, m, k = lhs.shape
    num_groups_, n, k_ = rhs.shape
    num_groups__, m_, n_ = out.shape
    num_groups___ = masked_m.numel()

    # Type and shape checks
    assert num_groups == num_groups_ == num_groups__ == num_groups___
    assert m == m_ and n == n_ and k == k_
    assert expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0
    assert lhs_scales.shape == (num_groups, m, 1)
    assert rhs_scales.shape == (num_groups, n, 1)
    assert lhs.dtype == torch.int8 and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.int8 and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert masked_m.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and masked_m.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    # lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()
    if configs:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(expected_m, n, k, num_groups, num_sms, is_grouped_masked=True)

    extra_info = get_extra_info()

    # Extra checks for TMA store
    if num_groups > 1 and m > block_m:
        assert m % block_m == 0, f'For masked grouped GEMM, shape M should be multiple of the block M (current block M: {block_m})'

    template_updated = template
    if extra_info['use_multistage_on_N']:
        template_updated = template.replace("GemmType::{GEMM_TYPE}>", "GemmType::{GEMM_TYPE},1>")

    args = (lhs, lhs_scales, rhs, rhs_scales, out,
            masked_m, m, expected_m,
            torch.cuda.current_stream(), num_sms, smem_config[0])
    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_int8_int8_bf16_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
              'WARP_M': warp_m, 'WARP_N': warp_n,
              'BLOCK_N_PADDING': smem_config[2],
              'NUM_GROUPS': num_groups, 'NUM_STAGES': num_stages,
              'GEMM_TYPE': 'GroupedMasked'},
        space=(),
        includes=includes_cutlass3 if extra_info['use_cutlass3'] else includes,
        arg_defs=(('lhs', torch.int8), ('lhs_scales', torch.float),
                  ('rhs', torch.int8), ('rhs_scales', torch.float),
                  ('out', torch.bfloat16),
                  ('grouped_layout', torch.int32), ('m', int), ('expected_m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
        template=template_updated,
        jit_include_dir='cutlass3' if extra_info['use_cutlass3'] else None,
        args=args
    )

    # Run the kernel
    runtime(*args)


def m_grouped_gemm_int8_int8_bf16_nt_nopad(lhs: Tuple[torch.Tensor],
                                     rhs: Tuple[torch.Tensor],
                                     out: torch.Tensor, m_indices: torch.Tensor,
                                     m_rows: torch.Tensor = None, configs = None) -> None:
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    m_, n_ = out.shape
    m__ = m_indices.numel()

    # Type and shape checks
    assert m == m_ == m__ and k == k_ and n == n_
    assert lhs_scales.shape == (m, 1)
    assert rhs_scales.shape == (num_groups, n, 1)
    assert lhs.dtype == torch.int8 and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.int8 and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert m_indices.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and m_indices.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    # lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    expected_m = ceil_div(m, num_groups)

    # Auto-tuning with compilation
    global includes, template, includes_gemv, template_gemv
    num_sms = get_num_sms()
    use_gemv = False

    if expected_m <= 2 and k % 16 == 0 and (k % 128 == 0 or (n >= 1024 and k <= 32 * 8)):
        # use gemmv if avg m small
        # ThreadPerN = 8
        # NUM_UNROLL = 1
        # SWZL_SIZE_M = 1
        # NPerThread = 1
        BlockSize, ThreadPerN, NUM_UNROLL, SWZL_SIZE_M, NPerThread, USE_SMALL_K = get_gemv_best_configs(m, n, k, num_groups, num_sms, torch.int8)

        if ThreadPerN != -1:
            args = (lhs, rhs, out,
                m_indices, m,
                torch.cuda.current_stream(),
                lhs_scales, rhs_scales)

            runtime = jit_tuner.compile_and_tune(
                name='m_grouped_gemv_int8_int8_bf16_nt',
                keys={'N': n, 'K': k, 'NUM_GROUPS': num_groups,
                    'ThreadPerN':ThreadPerN, 'NUM_UNROLL':NUM_UNROLL,
                    'SWZL_SIZE_M':SWZL_SIZE_M, 'NPerThread':NPerThread,
                    'BlockSize':BlockSize, 'USE_SMALL_K':USE_SMALL_K},
                space=(),
                includes=includes_gemv,
                arg_defs=(('lhs', torch.int8),
                        ('rhs', torch.int8),
                        ('out', torch.bfloat16),
                        ('grouped_layout', torch.int32), ('m', int),
                        ('stream', torch.cuda.Stream),
                        ('lhs_scales', torch.float),
                        ('rhs_scales', torch.float)),
                template=template_gemv,
                args=args
            )
            use_gemv = True

    if use_gemv == False:
        if configs:
            num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
        else:
            num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(expected_m, n, k, num_groups, num_sms, is_grouped_contiguous=False)

        extra_info = get_extra_info()

        if m_rows is None:
            counts = torch.bincount(m_indices)
            min_n = min(counts.size(0), num_groups)
            experts_for_rows = torch.zeros(num_groups, dtype=torch.int32, device='cuda')
            if min_n > 0:
                experts_for_rows[:min_n] = counts[:min_n]
            m_rows = experts_for_rows
        args = (lhs, lhs_scales, rhs, rhs_scales, out,
            m_rows, m, expected_m, num_groups,
            torch.cuda.current_stream(), num_sms, smem_config[0])

        runtime = jit_tuner.compile_and_tune(
            name='m_grouped_gemm_int8_int8_bf16_nt',
            keys={'N': n, 'K': k,
                'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                'WARP_M': warp_m, 'WARP_N': warp_n,
                'BLOCK_N_PADDING': smem_config[2],
                'NUM_GROUPS': num_groups, 'NUM_STAGES': num_stages,
                'GEMM_TYPE': 'GroupedNoPad'},
            space=(),
            includes=includes_cutlass3 if extra_info['use_cutlass3'] else includes,
            arg_defs=(  ('lhs', torch.int8), ('lhs_scales', torch.float),
                        ('rhs', torch.int8), ('rhs_scales', torch.float),
                        ('out', torch.bfloat16),
                        ('grouped_layout', torch.int32), ('m', int),
                        ('num_groups', int), ('expected_m', int),
                        ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
            template=template,
            jit_include_dir='cutlass3' if extra_info['use_cutlass3'] else None,
            args=args
        )

    # Run the kernel
    runtime(*args)
