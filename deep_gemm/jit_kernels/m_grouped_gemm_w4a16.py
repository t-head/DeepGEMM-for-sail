import torch
from typing import Tuple

from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, is_ppu1v5_device, GemmType

# C++ code templates
includes = ('"../deep_gemm/w4a16_gemm_cutlass3.cuh"', )
w4a16_nopad_template = """
using namespace deep_gemm;
using ElementA = cutlass::bfloat16_t;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto WARP_K = {WARP_K};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kGroupSize = {GROUP_SIZE};
constexpr auto N_EXPAND = {N_EXPAND};

// Make a templated grouped GEMM
using gemm_t = W4A16Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, kNumGroups, kNumStages, GemmType::{GEMM_TYPE}, kGroupSize, N_EXPAND>;
gemm_t::run((const ElementA*) lhs, rhs, (const ElementA*) rhs_scales, (ElementA*) out,
            m, expected_m, stream, num_sms,
            m_rows, {block_m_info});
"""

w4a16_masked_template = """
using namespace deep_gemm;
using ElementA = cutlass::bfloat16_t;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto WARP_K = {WARP_K};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kGroupSize = {GROUP_SIZE};
constexpr auto N_EXPAND = {N_EXPAND};

// Make a templated grouped GEMM
using gemm_t = W4A16Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, kNumGroups, kNumStages, GemmType::{GEMM_TYPE}, kGroupSize, N_EXPAND>;
gemm_t::run((const ElementA*) lhs, rhs, (const ElementA*) rhs_scales, (ElementA*) out,
            m, expected_m, stream, num_sms,
            m_rows, {nullptr});
"""

w4a16_fused_template = """
using namespace deep_gemm;
using ElementA = cutlass::bfloat16_t;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto WARP_K = {WARP_K};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kGroupSize = {GROUP_SIZE};
constexpr auto N_EXPAND = {N_EXPAND};

// Make a templated grouped GEMM
using gemm_t = W4A16Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, kNumGroups, kNumStages, GemmType::{GEMM_TYPE}, kGroupSize, N_EXPAND>;
gemm_t::run((const ElementA*) lhs, rhs, (const ElementA*) rhs_scales, (ElementA*) out,
            m, expected_m, stream, num_sms,
            m_rows, {expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks});
"""

def w4a16_get_best_configs(gemm_type, expected_m, n, k, num_groups, num_sms, group_size=32):
    for block_m in [16, 32, 64, 128]:
        if expected_m / block_m < 0.9: break
    warp_m = block_m if block_m <= 64 else block_m // 2
    # Now warps_on_k is only tested on group_size == 32
    warps_on_k = (block_m == warp_m and group_size == 32)
    if warps_on_k and k >= 2048:
        if is_ppu1v5_device(): # warps_on_n = 4, warps_on_k = 4
            block_n, warp_n, block_k, warp_k = 256, 64, 128, 32
        else: # warps_on_n = 2, warps_on_k = 8
            block_n, warp_n, block_k, warp_k = 128, 64, 256, 32
    else:
        block_n, warp_n, block_k, warp_k = 256, 64, 64, 64
    # small tile for debug
    # block_m, warp_m = 16, 16
    # block_n, warp_n = 128, 64
    # block_k, warp_k = 64, 32
    num_stages = 2 if k <= 512 else 3
    configs = (num_sms, block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages)
    return configs


def m_grouped_gemm_w4a16_common(gemm_type: GemmType, expected_m: int,
                                lhs: torch.Tensor,
                                rhs_: Tuple[torch.Tensor, torch.Tensor],
                                out: torch.Tensor,
                                group_size: int, configs,
                                m_rows: torch.Tensor,
                                scheduler_extra):
    """
    W4A16 grouped GEMM without padding.

    Args:
        lhs: Activation tensor in BF16, shape nopad: (m, k), masked: (num_groups, m, k), fused: (num_token, k)
        rhs_: Tuple of (weight, scale)
            - weight: 4-bit weight stored in int32, shape (num_groups, k // 16, n * 2)
            - scale: Per-channel scale in BF16, shape (num_groups, k // group_size), n)
        out: Output tensor in BF16, shape nopad/fused: (m, n), masked: (num_groups, m, n)
        m_rows: Number of rows per group, shape (num_groups,)
        group_size: Group size for quantization (default 32)
        configs: Optional pre-configured kernel parameters
    """
    rhs, rhs_scales = rhs_
    if gemm_type == GemmType.GroupedMasked:
        _, m, k = lhs.shape
        _, m_, n_ = out.shape
    else:
        m, k = lhs.shape
        m_, n_ = out.shape
    num_groups, _, n2 = rhs.shape
    n = n2 // 2

    # Type and shape checks
    if m == 0: return
    assert k % group_size == 0, f"K must be a multiple of group_size, got k={k}, group_size={group_size}"
    assert n == n_
    assert rhs.shape == (num_groups, k // 16, n * 2), f"Weights shape {rhs.shape} != ({num_groups}, {k // 16}, {n * 2})"
    assert rhs_scales.shape == (num_groups, k // group_size, n), f"Scale shape {rhs_scales.shape} != ({num_groups}, {k // group_size}, {n})"
    assert n > 0 and k > 0 and n % 64 == 0 and k % 16 == 0
    assert lhs.dtype == torch.bfloat16
    assert rhs.dtype == torch.int32
    assert rhs_scales.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    assert lhs.is_contiguous() and rhs.is_contiguous() and out.is_contiguous()
    assert rhs_scales.is_contiguous()

    num_sms, block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages = configs
    assert warp_n == 64
    n_expand = 4 if (is_ppu1v5_device() and k <= 512 and n % (block_n * 4) == 0 and k % block_k == 0 and num_stages == 2 and block_k == warp_k) else 1
    gemm_type_name = gemm_type.name

    if gemm_type == GemmType.GroupedNoPad:
        block_m_info = scheduler_extra
        args = (lhs, rhs, rhs_scales, out, m, expected_m, torch.cuda.current_stream(), num_sms,
                m_rows, block_m_info)
        runtime = jit_tuner.compile_and_tune(
            name='m_grouped_gemm_w4a16',
            keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                'WARP_M': warp_m, 'WARP_N': warp_n, 'WARP_K': warp_k, 'NUM_GROUPS': num_groups,
                'NUM_STAGES': num_stages, 'GROUP_SIZE': group_size, 'GEMM_TYPE': gemm_type_name, 'N_EXPAND': n_expand},
            space=(),
            includes=includes,
            arg_defs=(('lhs', torch.bfloat16), ('rhs', torch.int32), ('rhs_scales', torch.bfloat16), ('out', torch.bfloat16),
                    ('m', int), ('expected_m', int), ('stream', torch.cuda.Stream), ('num_sms', int),
                    ('m_rows', torch.int32), ('block_m_info', torch.int32)),
            template=w4a16_nopad_template,
            args=args,
            jit_include_dir='actlize_v1.0.0'
        )
    elif gemm_type == GemmType.GroupedMasked:
        args = (lhs, rhs, rhs_scales, out, m, expected_m, torch.cuda.current_stream(), num_sms, m_rows)
        runtime = jit_tuner.compile_and_tune(
            name='m_grouped_gemm_w4a16',
            keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                'WARP_M': warp_m, 'WARP_N': warp_n, 'WARP_K': warp_k, 'NUM_GROUPS': num_groups,
                'NUM_STAGES': num_stages, 'GROUP_SIZE': group_size, 'GEMM_TYPE': gemm_type_name, 'N_EXPAND': n_expand},
            space=(),
            includes=includes,
            arg_defs=(('lhs', torch.bfloat16), ('rhs', torch.int32), ('rhs_scales', torch.bfloat16), ('out', torch.bfloat16),
                    ('m', int), ('expected_m', int), ('stream', torch.cuda.Stream), ('num_sms', int),
                    ('m_rows', torch.int32)),
            template=w4a16_masked_template,
            args=args,
            jit_include_dir='actlize_v1.0.0'
        )
    elif gemm_type == GemmType.GroupedFused:
        expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks = scheduler_extra
        args = (lhs, rhs, rhs_scales, out, m, expected_m, torch.cuda.current_stream(), num_sms,
                m_rows, expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks)
        runtime = jit_tuner.compile_and_tune(
            name='m_grouped_gemm_w4a16',
            keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                'WARP_M': warp_m, 'WARP_N': warp_n, 'WARP_K': warp_k, 'NUM_GROUPS': num_groups,
                'NUM_STAGES': num_stages, 'GROUP_SIZE': group_size, 'GEMM_TYPE': gemm_type_name, 'N_EXPAND': n_expand},
            space=(),
            includes=includes,
            arg_defs=(('lhs', torch.bfloat16), ('rhs', torch.int32), ('rhs_scales', torch.bfloat16), ('out', torch.bfloat16),
                    ('m', int), ('expected_m', int), ('stream', torch.cuda.Stream), ('num_sms', int),
                    ('m_rows', torch.int32),
                    ('expert_ids_and_cumsum', torch.int32), ('sorted_token_ids', torch.int32), ('aligned_num_m_blocks', torch.int32)),
            template=w4a16_fused_template,
            args=args,
            jit_include_dir='actlize_v1.0.0'
        )

    runtime(*args)
    return


def m_grouped_gemm_w4a16_fused(lhs: torch.Tensor,
                                rhs_: Tuple[torch.Tensor, torch.Tensor],
                                out: torch.Tensor,
                                m_rows: torch.Tensor,
                                expert_ids_and_cumsum: torch.Tensor,
                                sorted_token_ids: torch.Tensor,
                                aligned_num_m_blocks: torch.Tensor,
                                configs):
    num_token = lhs.shape[0]
    num_groups = rhs_[0].shape[0]
    m_sum = out.shape[0]
    assert m_sum % num_token == 0, f"out rows ({m_sum}) must be divisible by num_token ({num_token})"
    topk = m_sum // num_token
    assert num_groups >= topk
    expected_m = ceil_div(m_sum, num_groups)
    n = rhs_[1].shape[2]
    k = lhs.shape[1]
    group_size = lhs.shape[1] // rhs_[1].shape[1]
    m_grouped_gemm_w4a16_common(GemmType.GroupedFused, expected_m, lhs, rhs_, out, group_size, configs, m_rows, (expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks))


def m_grouped_gemm_w4a16_masked(lhs: torch.Tensor,
                                rhs_: Tuple[torch.Tensor, torch.Tensor],
                                out: torch.Tensor,
                                masked_m: torch.Tensor, expected_m: int, configs=None):
    num_groups, m_padded, k = lhs.shape
    n = rhs_[1].shape[2]
    group_size = k // rhs_[1].shape[1]
    if configs is None:
        configs = w4a16_get_best_configs(GemmType.GroupedMasked, expected_m, n, k, num_groups, get_num_sms(), group_size)
    m_grouped_gemm_w4a16_common(GemmType.GroupedMasked, expected_m, lhs, rhs_, out, group_size, configs, masked_m, None)


def m_grouped_gemm_w4a16_nopad(lhs: torch.Tensor,
                                rhs_: Tuple[torch.Tensor, torch.Tensor],
                                out: torch.Tensor,
                                m_indices: torch.Tensor, m_rows: torch.Tensor = None,
                                configs = None):
    num_groups = rhs_[0].shape[0]
    m = lhs.shape[0]
    group_size = lhs.shape[1] // rhs_[1].shape[1]
    expected_m = ceil_div(m, num_groups)
    n = rhs_[1].shape[2]
    k = lhs.shape[1]
    if configs is None:
        configs = w4a16_get_best_configs(GemmType.GroupedNoPad, expected_m, n, k, num_groups, get_num_sms(), group_size)
    block_m = configs[1]
    if m_rows is None:
        counts = torch.bincount(m_indices)
        min_n = min(counts.size(0), num_groups)
        experts_for_rows = torch.zeros(num_groups, dtype=torch.int32, device='cuda')
        if min_n > 0:
            experts_for_rows[:min_n] = counts[:min_n]
        m_rows = experts_for_rows
    block_m_info = torch.empty((num_groups + ceil_div(m + 1 - num_groups, block_m)) * 4, dtype=torch.int32, device=m_rows.device)
    m_grouped_gemm_w4a16_common(GemmType.GroupedNoPad, expected_m, lhs, rhs_, out, group_size, configs, m_rows, block_m_info)
