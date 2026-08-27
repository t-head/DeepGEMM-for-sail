from enum import Enum
from typing import Tuple

import torch

from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, is_ppu1v5_device, GemmType

class W4A16Type(Enum):
    int4 = 0
    mxfp4_e8m0 = 1
    mxfp4_bf16 = 2
    mxfp4_e8m0_mma = 3


def get_w4a16_type(rhs_: Tuple[torch.Tensor, torch.Tensor], fp4_use_bf16_scale: bool):
    if rhs_[0].dtype == torch.uint8:
        assert is_ppu1v5_device(), "w4fa16_mma is only supported on PPU1.5"
        return W4A16Type.mxfp4_e8m0_mma
    if fp4_use_bf16_scale:
        return W4A16Type.mxfp4_bf16
    if rhs_[1].dtype == torch.uint8:
        return W4A16Type.mxfp4_e8m0
    return W4A16Type.int4


# C++ code templates
includes = ('"../deep_gemm/w4a16_gemm_cutlass3.cuh"', )
w4a16_nopad_template = """
using namespace deep_gemm;
using ElementA = cutlass::bfloat16_t;
using ElementB = {ELEMENT_B};
using ElementScale = {ELEMENT_SCALE};

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

using gemm_t = W4A16Gemm<ElementB, ElementScale, N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, kNumGroups, kNumStages, GemmType::{GEMM_TYPE}, kGroupSize, N_EXPAND>;
gemm_t::run((const ElementA*) lhs, rhs, (const ElementScale*) rhs_scales, (ElementA*) out,
            m, expected_m, stream, num_sms,
            m_rows, {block_m_info});
"""

w4a16_masked_template = """
using namespace deep_gemm;
using ElementA = cutlass::bfloat16_t;
using ElementB = {ELEMENT_B};
using ElementScale = {ELEMENT_SCALE};

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

using gemm_t = W4A16Gemm<ElementB, ElementScale, N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, kNumGroups, kNumStages, GemmType::{GEMM_TYPE}, kGroupSize, N_EXPAND>;
gemm_t::run((const ElementA*) lhs, rhs, (const ElementScale*) rhs_scales, (ElementA*) out,
            m, expected_m, stream, num_sms,
            m_rows, {nullptr});
"""

w4a16_fused_template = """
using namespace deep_gemm;
using ElementA = cutlass::bfloat16_t;
using ElementB = {ELEMENT_B};
using ElementScale = {ELEMENT_SCALE};

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

using gemm_t = W4A16Gemm<ElementB, ElementScale, N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, kNumGroups, kNumStages, GemmType::{GEMM_TYPE}, kGroupSize, N_EXPAND>;
gemm_t::run((const ElementA*) lhs, rhs, (const ElementScale*) rhs_scales, (ElementA*) out,
            m, expected_m, stream, num_sms,
            m_rows, {expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks, topk});
"""

def w4a16_get_best_configs(expected_m, n, k, num_groups, num_sms, gemm_type, w4a16_type):
    is_ppu1v5 = is_ppu1v5_device()
    block_m_list = [16, 32, 64, 128]
    for block_m in block_m_list:
        if expected_m / block_m < 0.9: break
    warp_m = block_m if block_m <= 64 else 64
    if w4a16_type == W4A16Type.mxfp4_e8m0_mma:
        assert is_ppu1v5, "w4fa16_mma is only supported on PPU1.5"
        warps_on_k = (k >= 2048)
        tile_list = {
            (64, False): (512, 64, 128, 128, 2),
            (64, True): (256, 64, 128, 64, 3),
            (32, True): (256, 64, 128, 64, 3),
            # (16, True): (256, 64, 128, 64, 3),
        }
        block_n, warp_n, block_k, warp_k, num_stages = tile_list.get((block_m, warps_on_k), (256, 64, 128, 128, 2))
    else:
        warps_on_k = (block_m == warp_m and k >= 2048)
        if warps_on_k:
            if is_ppu1v5: # warps_on_n = 4, warps_on_k = 4
                block_n, warp_n, block_k, warp_k = 256, 64, 128, 32
            else: # warps_on_n = 2, warps_on_k = 8
                block_n, warp_n, block_k, warp_k = 128, 64, 256, 32
        else:
            block_n, warp_n, block_k, warp_k = 256, 64, 64, 64
        num_stages = 2 if k <= 512 else 3
    if is_ppu1v5 and k <= 512 and k % block_k == 0 and num_stages == 2 and block_k == warp_k:
        for n_expand in [4, 3, 2, 1]:
            if n % (block_n * n_expand) == 0: break
        if warp_m == 64 and gemm_type == GemmType.GroupedFused and w4a16_type == W4A16Type.mxfp4_e8m0_mma:
            n_expand = 1 # n_expand > 1 will cause vreg exceeds the 256 limit
    else:
        n_expand = 1
    configs = (num_sms, block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages, n_expand)
    # print(f'expected_m:{expected_m}, n:{n}, k:{k}, configs: {configs}')
    return configs


def m_grouped_gemm_w4a16_common(w4a16_type: W4A16Type, gemm_type: GemmType, expected_m: int,
                                lhs: torch.Tensor,
                                rhs_: Tuple[torch.Tensor, torch.Tensor],
                                out: torch.Tensor,
                                configs,
                                m_rows: torch.Tensor,
                                scheduler_extra):
    """
    W4A16 / W4FA16 grouped GEMM.

    Args:
        w4a16_type: Quantization layout and dequant path selector.
        lhs: Activation tensor in BF16, shape nopad: (m, k), masked: (num_groups, m, k), fused: (num_token, k)
        rhs_: Tuple of (weight, scale)
            - normal weight: 4-bit weight stored in int32, shape (num_groups, k // 16, n * 2)
            - mma weight: packed uint8 weight, shape (num_groups, n, k // 2)
            - normal scale: BF16 numerical scale or uint8 raw E8M0 exponent bytes, shape (num_groups, k // group_size, n)
            - mma scale: packed uint8 raw E8M0 exponent bytes, shape (num_groups, n // 64, k * 2)
        out: Output tensor in BF16, shape nopad/fused: (m, n), masked: (num_groups, m, n)
        m_rows: Number of rows per group, shape (num_groups,)
        configs: Optional pre-configured kernel parameters
    """
    rhs, rhs_scales = rhs_
    if w4a16_type == W4A16Type.mxfp4_e8m0_mma:
        assert is_ppu1v5_device(), "w4fa16_mma is only supported on PPU1.5"

    if gemm_type == GemmType.GroupedMasked:
        _, m, k = lhs.shape
        _, m_, n_ = out.shape
    else:
        m, k = lhs.shape
        m_, n_ = out.shape
    if w4a16_type == W4A16Type.mxfp4_e8m0_mma:
        num_groups, n, _ = rhs.shape
        n = n_
    else:
        num_groups, _, n2 = rhs.shape
        n = n2 // 2

    scale_elements_per_group = rhs_scales.shape[1] * rhs_scales.shape[2]
    assert k * n % scale_elements_per_group == 0
    group_size = k * n // scale_elements_per_group
    assert group_size == 32, f"W4A16 only supports group_size=32, got {group_size}"

    # Type and shape checks
    if m == 0: return
    assert k % group_size == 0, f"K must be a multiple of group_size, got k={k}, group_size={group_size}"
    assert n == n_
    if w4a16_type == W4A16Type.mxfp4_e8m0_mma:
        assert rhs.dtype == torch.uint8, f"w4fa16_mma weight dtype must be uint8, got {rhs.dtype}"
        assert rhs.shape == (num_groups, n, k // 2), \
            f"Weights shape {rhs.shape} != ({num_groups}, {n}, {k // 2})"
        assert rhs_scales.dtype == torch.uint8, \
            f"w4fa16_mma scale dtype must be uint8, got {rhs_scales.dtype}"
        assert rhs_scales.shape == (num_groups, n // 64, k * 2), \
            f"Scale shape {rhs_scales.shape} != ({num_groups}, {n // 64}, {k * 2})"
    else:
        assert rhs.shape == (num_groups, k // 16, n * 2), f"Weights shape {rhs.shape} != ({num_groups}, {k // 16}, {n * 2})"
        assert rhs_scales.shape == (num_groups, k // group_size, n), f"Scale shape {rhs_scales.shape} != ({num_groups}, {k // group_size}, {n})"
        assert rhs.dtype == torch.int32
    assert n > 0 and k > 0 and n % 64 == 0 and k % 16 == 0
    assert lhs.dtype == torch.bfloat16
    if w4a16_type == W4A16Type.mxfp4_e8m0:
        assert rhs_scales.dtype == torch.uint8, \
            f"W4FA16 E8M0 scale dtype must be uint8, got {rhs_scales.dtype}"
    elif w4a16_type != W4A16Type.mxfp4_e8m0_mma:
        assert rhs_scales.dtype == torch.bfloat16, \
            f"W4A16 BF16 scale dtype must be bfloat16, got {rhs_scales.dtype}"
    assert out.dtype == torch.bfloat16
    assert lhs.is_contiguous() and rhs.is_contiguous() and out.is_contiguous()
    assert rhs_scales.is_contiguous()
    scale_dtype = rhs_scales.dtype
    element_b = 'int4_t' if w4a16_type == W4A16Type.int4 else (
        'uint8_t' if w4a16_type == W4A16Type.mxfp4_e8m0_mma else 'cutlass::float4_t'
    )
    element_scale = 'uint8_t' if w4a16_type in (W4A16Type.mxfp4_e8m0, W4A16Type.mxfp4_e8m0_mma) else 'bfloat16_t'

    num_sms, block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages, n_expand = configs
    assert warp_n == 64
    if w4a16_type == W4A16Type.mxfp4_e8m0_mma:
        assert block_k >= 128 and (warp_k == 64 or warp_k == block_k == 128)
    gemm_type_name = gemm_type.name
    # w4fa16_mma not use "-sort-copy-before-coalesce"
    jit_name = 'm_grouped_gemm_w4fa16_mma' if w4a16_type == W4A16Type.mxfp4_e8m0_mma else 'm_grouped_gemm_w4a16'

    if gemm_type == GemmType.GroupedNoPad:
        block_m_info = scheduler_extra
        args = (lhs, rhs, rhs_scales, out, m, expected_m, torch.cuda.current_stream(), num_sms,
                m_rows, block_m_info)
        runtime = jit_tuner.compile_and_tune(
            name=jit_name,
            keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                'WARP_M': warp_m, 'WARP_N': warp_n, 'WARP_K': warp_k, 'NUM_GROUPS': num_groups,
                'NUM_STAGES': num_stages, 'GROUP_SIZE': group_size, 'GEMM_TYPE': gemm_type_name, 'N_EXPAND': n_expand,
                'ELEMENT_B': element_b, 'ELEMENT_SCALE': element_scale},
            space=(),
            includes=includes,
            arg_defs=(('lhs', torch.bfloat16), ('rhs', rhs.dtype), ('rhs_scales', scale_dtype), ('out', torch.bfloat16),
                    ('m', int), ('expected_m', int), ('stream', torch.cuda.Stream), ('num_sms', int),
                    ('m_rows', torch.int32), ('block_m_info', torch.int32)),
            template=w4a16_nopad_template,
            args=args,
            jit_include_dir='actlize_v1.0.0'
        )
    elif gemm_type == GemmType.GroupedMasked:
        args = (lhs, rhs, rhs_scales, out, m, expected_m, torch.cuda.current_stream(), num_sms, m_rows)
        runtime = jit_tuner.compile_and_tune(
            name=jit_name,
            keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                'WARP_M': warp_m, 'WARP_N': warp_n, 'WARP_K': warp_k, 'NUM_GROUPS': num_groups,
                'NUM_STAGES': num_stages, 'GROUP_SIZE': group_size, 'GEMM_TYPE': gemm_type_name, 'N_EXPAND': n_expand,
                'ELEMENT_B': element_b, 'ELEMENT_SCALE': element_scale},
            space=(),
            includes=includes,
            arg_defs=(('lhs', torch.bfloat16), ('rhs', rhs.dtype), ('rhs_scales', scale_dtype), ('out', torch.bfloat16),
                    ('m', int), ('expected_m', int), ('stream', torch.cuda.Stream), ('num_sms', int),
                    ('m_rows', torch.int32)),
            template=w4a16_masked_template,
            args=args,
            jit_include_dir='actlize_v1.0.0'
        )
    elif gemm_type == GemmType.GroupedFused:
        expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks, topk = scheduler_extra
        args = (lhs, rhs, rhs_scales, out, m, expected_m, torch.cuda.current_stream(), num_sms,
                m_rows, expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks, topk)
        runtime = jit_tuner.compile_and_tune(
            name=jit_name,
            keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                'WARP_M': warp_m, 'WARP_N': warp_n, 'WARP_K': warp_k, 'NUM_GROUPS': num_groups,
                'NUM_STAGES': num_stages, 'GROUP_SIZE': group_size, 'GEMM_TYPE': gemm_type_name, 'N_EXPAND': n_expand,
                'ELEMENT_B': element_b, 'ELEMENT_SCALE': element_scale},
            space=(),
            includes=includes,
            arg_defs=(('lhs', torch.bfloat16), ('rhs', rhs.dtype), ('rhs_scales', scale_dtype), ('out', torch.bfloat16),
                    ('m', int), ('expected_m', int), ('stream', torch.cuda.Stream), ('num_sms', int),
                    ('m_rows', torch.int32),
                    ('expert_ids_and_cumsum', torch.int32), ('sorted_token_ids', torch.int32), ('aligned_num_m_blocks', torch.int32),
                    ('topk', int)),
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
                                configs,
                                fp4_use_bf16_scale: bool = False):
    w4a16_type = get_w4a16_type(rhs_, fp4_use_bf16_scale)
    num_token = lhs.shape[0]
    num_groups = rhs_[0].shape[0]
    m_sum = out.shape[0]
    assert m_sum % num_token == 0, f"out rows ({m_sum}) must be divisible by num_token ({num_token})"
    topk = m_sum // num_token
    assert num_groups >= topk
    expected_m = ceil_div(m_sum, num_groups)
    m_grouped_gemm_w4a16_common(w4a16_type, GemmType.GroupedFused, expected_m, lhs, rhs_, out, configs, m_rows, (expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks, topk))


def m_grouped_gemm_w4a16_masked(lhs: torch.Tensor,
                                rhs_: Tuple[torch.Tensor, torch.Tensor],
                                out: torch.Tensor,
                                masked_m: torch.Tensor, expected_m: int, configs=None,
                                fp4_use_bf16_scale: bool = False):
    w4a16_type = get_w4a16_type(rhs_, fp4_use_bf16_scale)
    num_groups, m_padded, k = lhs.shape
    n = rhs_[0].shape[1] if w4a16_type == W4A16Type.mxfp4_e8m0_mma else rhs_[1].shape[2]
    if configs is None:
        configs = w4a16_get_best_configs(expected_m, n, k, num_groups, get_num_sms(), GemmType.GroupedMasked, w4a16_type)
    m_grouped_gemm_w4a16_common(w4a16_type, GemmType.GroupedMasked, expected_m, lhs, rhs_, out, configs, masked_m, None)


def m_grouped_gemm_w4a16_nopad(lhs: torch.Tensor,
                                rhs_: Tuple[torch.Tensor, torch.Tensor],
                                out: torch.Tensor,
                                m_indices: torch.Tensor, m_rows: torch.Tensor = None,
                                configs = None,
                                fp4_use_bf16_scale: bool = False):
    num_groups = rhs_[0].shape[0]
    m = lhs.shape[0]
    w4a16_type = get_w4a16_type(rhs_, fp4_use_bf16_scale)
    n = rhs_[0].shape[1] if w4a16_type == W4A16Type.mxfp4_e8m0_mma else rhs_[1].shape[2]
    expected_m = ceil_div(m, num_groups)
    k = lhs.shape[1]
    if configs is None:
        configs = w4a16_get_best_configs(expected_m, n, k, num_groups, get_num_sms(), GemmType.GroupedNoPad, w4a16_type)
    block_m = configs[1]
    if m_rows is None:
        counts = torch.bincount(m_indices)
        min_n = min(counts.size(0), num_groups)
        experts_for_rows = torch.zeros(num_groups, dtype=torch.int32, device='cuda')
        if min_n > 0:
            experts_for_rows[:min_n] = counts[:min_n]
        m_rows = experts_for_rows
    block_m_info = torch.empty((num_groups + ceil_div(m + 1 - num_groups, block_m)) * 4, dtype=torch.int32, device=m_rows.device)
    m_grouped_gemm_w4a16_common(w4a16_type, GemmType.GroupedNoPad, expected_m, lhs, rhs_, out, configs, m_rows, block_m_info)
