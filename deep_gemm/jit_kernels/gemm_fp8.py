import math
import torch
from functools import lru_cache
from typing import Tuple

from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, get_m_alignment_for_contiguous_layout,get_col_major_tma_aligned_tensor, GemmType
from .gemm_fp8_lut import get_best_configs_from_lut
from .gemm_int8 import gemm_a8w8_per_channel_nt

# C++ code templates
includes = ('"../deep_gemm/fp8_gemm.cuh"', )
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto BLOCK_N_PADDING = {BLOCK_N_PADDING};
constexpr auto kSwizzleDMode = {SWIZZLE_D_MODE};
constexpr auto kNumGroups = 1;
constexpr auto kNumStages = {NUM_STAGES};

// Make a templated GEMM
using gemm_t = Fp8Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BLOCK_N_PADDING, kSwizzleDMode, kNumGroups, kNumStages, GemmType::DenseGemm>;

// Launch kernel
gemm_t::run(out, lhs, rhs, lhs_scales,
            rhs_scales, nullptr, nullptr, m, 0,
            stream, num_sms, smem_size);
"""


def is_tma_multicast_legal(shape_dim: int, block_dim: int, num_tma_multicast: int, num_sms: int) -> bool:
    if num_tma_multicast == 1:
        return True
    return (shape_dim % (block_dim * num_tma_multicast) == 0) and num_sms % num_tma_multicast == 0


def get_swizzle_mode(block_n: int) -> int:
    # TODO: remove some candidates if slow
    elem_size = 2
    for mode_bytes in (128, 64, 32):
        if (block_n * elem_size) % mode_bytes == 0:
            return mode_bytes
    return 0


def get_block_n_padding_for_smem_d(block_n: int) -> int:
    # NOTES: padding is for solving bank conflicts, but wastes shared memory space
    elem_size, requirement = 2, (4, 8)
    bank_stride = (block_n * elem_size) // 4
    padding = (requirement[0] - bank_stride) % requirement[1]
    return (((padding + requirement[1]) if padding < 0 else padding) * 4) // elem_size


def get_smem_config(num_stages: int, k: int, block_m: int, block_n: int, block_k: int = 128) -> Tuple[int, int, int]:
    # Try swizzle first, as it does not waste shared memory
    swizzle_mode = 128
    block_n_padding = 0

    smem_d = block_m * (block_n + block_n_padding)
    smem_a_per_stage = block_m * block_k
    smem_b_per_stage = block_n * block_k

    smem_size_d = smem_d * 2
    smem_size_a = num_stages * smem_a_per_stage
    smem_size_b = num_stages * smem_b_per_stage
    smem_scales_a_per_stage = block_m * 4
    smem_scales_b = ceil_div(k, block_k) * 4

    smem_size = max(smem_size_d, smem_size_a + smem_size_b)
    smem_size += num_stages * smem_scales_a_per_stage
    smem_size += ceil_div(smem_scales_b * (1 if block_k % block_n == 0 else 2), 8) * 8

    # Swizzle and padding are not compatible
    assert int(swizzle_mode > 0) + int(block_n_padding > 0) <= 1

    return smem_size, swizzle_mode, block_n_padding

def get_num_occ(block_m: int, block_n: int, block_k: int, num_stages: int) -> Tuple[int]:
    if block_m is None:
        return 0

    # use static suppose.
    ppu_capacity = 262144
    bpp = 1
    smem_d = block_m * block_n
    smem_a_per_stage = block_m * block_k
    smem_b_per_stage = block_n * block_k

    smem_size_d = smem_d * 2
    smem_size_a = num_stages * smem_a_per_stage * bpp
    smem_size_b = num_stages * smem_b_per_stage * bpp

    smem_size = max(smem_size_d, smem_size_a + smem_size_b)

    smem_occ = int(262144 // smem_size)

    vreg_occ_lut = {
        (16, 256) : 3,
        (32, 128) : 3,
        (32, 256) : 2,
        (64, 64) : 2,
        (64, 128) : 2,
        (64, 256) : 2,
        (128, 128) : 2,
    }
    key = (block_m, block_n)
    if key in vreg_occ_lut.keys():
        vreg_occ = vreg_occ_lut[key]
    else:
        vreg_occ =1

    return min(smem_occ, vreg_occ)

def get_smem_occ(block_m: int, block_n: int, block_k: int, num_stages: int) -> Tuple[int]:
    if block_m is None:
        return 0

    # use static suppose.
    ppu_capacity = 262144
    bpp = 1
    smem_d = block_m * block_n
    smem_a_per_stage = block_m * block_k
    smem_b_per_stage = block_n * block_k

    smem_size_d = smem_d * 2
    smem_size_a = num_stages * smem_a_per_stage * bpp
    smem_size_b = num_stages * smem_b_per_stage * bpp

    smem_size = max(smem_size_d, smem_size_a + smem_size_b)

    return 262144 // smem_size

@lru_cache(maxsize=None)
def get_best_configs_dense(m: int, n: int, k: int, num_groups: int, num_sms: int) -> \
    Tuple[int, int, int, int, int, int, int, int, int, dict]:
    block_ms = (256, 192, 128, 64)
    block_ns = (256, 128, 64)


    fix_wave_saturate = lambda x: num_sms if x == 0 else x
    get_num_waves = lambda bm, bn: (ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms) if bm else None) # groups*m_block*n_block/sms = num waves
    get_last_wave_util = lambda bm, bn: fix_wave_saturate((ceil_div(m, bm) * ceil_div(n, bn) * num_groups) % num_sms)

    #block size wasted
    get_block_utils = lambda m, bm: (((m / bm ) / ((m + bm -1) // bm)) if m % bm != 0 else 1.0) if bm else 0

    # Decide block sizes by waves
    best_block_m, best_block_n = None, None
    for block_m in block_ms:
        # NOTES: the block sizes can not be too large, so at least one dim less than 128
        for block_n in filter(lambda bn: ((block_m <= 128 or bn <= 128) and (bn != n and n >= 1) and not (block_m == 128 and bn == 128)), block_ns):
            success = False
            num_waves, best_num_waves = get_num_waves(block_m, block_n), get_num_waves(best_block_m, best_block_n)
            num_utils = get_block_utils(m, block_m) * get_block_utils(n, block_n)
            best_num_utils = get_block_utils(m, best_block_m) * get_block_utils(n, best_block_n)
            num_occ, best_num_occ = get_smem_occ(block_m, block_n, 128, 2), get_smem_occ(best_block_m, best_block_n, 128, 2)
            # print(f"block_m:{block_m}, block_n:{block_n}, best_block_m:{best_block_m}, best_block_n:{best_block_n}")
            # print(f'num_occ:{num_occ}, best_num_occ:{best_num_occ}')
            # # # print(f'num_waves:{num_waves / num_occ}, best_num_waves:{best_num_waves / best_num_occ}')
            # print(f'm_util:{get_block_utils(m, block_m)}, n_util:{get_block_utils(n, block_n)}, num_utils:{num_utils}')
            # print(f'best_m_util:{get_block_utils(m, best_block_m)}, best_n_util:{get_block_utils(n, best_block_n)}, best_num_utils:{best_num_utils}')
            if best_block_m is None or best_block_n is None:
                success = True
            # elif (m < 512 or n < 512):
            #     # if single group block is small, balance wave, utils and occ
            #     occ_wave = num_waves / num_occ
            #     best_occ_wave = best_num_waves / best_num_occ
            #     ai_util = (block_m * block_n) / (block_m + block_n)
            #     best_ai_util = (best_block_m * best_block_n) / (best_block_m + best_block_n)

            #     valid_occ  = (num_occ / best_num_occ) >= 1
            #     valid_wave = (occ_wave / best_occ_wave) <= 1
            #     valid_util = (num_utils / best_num_utils) >= 1
            #     valid_ai   = (ai_util / best_ai_util) >= 1

            #     # print(f'num_waves:{num_waves / num_occ}, best_num_waves:{best_num_waves / best_num_occ}')
            #     # print(f'ai:{ai_util}, best_ai:{best_ai_util}')
            #     # print(f'util_ratio:{(num_utils / best_num_utils)}, wave_ratio:{occ_wave / best_occ_wave}, occ_ratio:{(num_occ / best_num_occ)}, ai_ratio:{ai_util / best_ai_util}')

            #     # print(f'valid_occ:{valid_occ}, valid_wave:{valid_wave}, valid_util:{valid_util}, valid_ai:{valid_ai}')
            #     success = (valid_wave + valid_util + valid_occ + valid_ai) >= 3

            elif num_waves < best_num_waves:
                success = True
            elif num_waves == best_num_waves:
                # Check last wave utilization
                util = get_last_wave_util(block_m, block_n)
                best_util = get_last_wave_util(best_block_m, best_block_n)
                success = util > best_util
                # print(f'best_block_m:{best_block_m}, best_block_n:{best_block_n}, num_waves:{num_waves}, best_num_waves:{best_num_waves}, util:{util}, best_util:{best_util}\n')
                if util == best_util:
                    # Case 1: same `block_m`, smaller `block_n` (wasted)
                    success |= block_m == best_block_m and block_n < best_block_n
                    # Case 2: same `block_n`, smaller `block_m` (wasted)
                    success |= block_n == best_block_n and block_m < best_block_m
                    # Case 3: different for both `block_m` and `block_n`, `block_n` larger is better
                    success |= block_m != best_block_m and block_n > best_block_n
                # print(f'success:{success}\n')
            best_block_m, best_block_n = (block_m, block_n) if success else (best_block_m, best_block_n)
    assert best_block_m is not None and best_block_n is not None
    if m >= 2048 and n >= 2048 and k >= 2048:
        (best_block_m, best_block_n, best_block_k, best_warp_m, best_warp_n, best_stages) = (192, 256, 128, 48, 64, 4)




    # Always pick the longest one
    # NOTES: for double B scales, the best number of stages may be reduced
    best_num_stages, best_smem_config, ppu_capacity = None, None, 262144
    # barrier impl only support 128
    block_k = 128
    stage_candidates = tuple(filter(lambda s: s <= k // block_k, (4, 3, 2)))
    # if not stage_candidates or (128 % best_block_n != 0 and 128 // math.gcd(128, best_block_n) <= 4) or best_block_m == 16 or best_block_m == 32:
    if not stage_candidates or (128 % best_block_n != 0 and 128 // math.gcd(128, best_block_n) <= 4):
        stage_candidates = (4, 3, 2)
    best_occ = 0
    for num_stages in stage_candidates:
        best_smem_config = get_smem_config(num_stages, k, best_block_m, best_block_n, block_k)
        # print(best_smem_config[0])
        if best_smem_config[0] < ppu_capacity:
            occ = ppu_capacity // best_smem_config[0]
            # if k < 256 or (best_block_m > 32 and best_block_n >= 64) and occ >= best_occ:
            #     # compute block and too small-k use higer occ rather than large stage
            #     best_num_stages = num_stages
            #     best_occ = occ
            # else:
            #     best_num_stages = num_stages
            #     break
            best_num_stages = num_stages
            break

    assert best_smem_config is not None
    assert best_num_stages is not None


    # Recompute the minimal number of SMs required
    # NOTES: less L2 cache usage and less GPU frequency drop
    num_waves = get_num_waves(best_block_m, best_block_n)
    num_min_sms = ceil_div(ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups, num_waves)
    assert num_min_sms <= num_sms
    
    warp_m = best_block_m // 4
    warp_n = best_block_n // 4

    if best_block_m == 64 and best_block_n == 256:
        warp_m = 32
        warp_n = 32

    return num_min_sms, best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages, best_smem_config

@lru_cache(maxsize=None)
def get_best_configs(m: int, n: int, k: int, num_groups: int, num_sms: int,
                     gemm_type: GemmType = GemmType.DenseGemm,
                     max_block_n: int = 256) -> \
        Tuple[int, int, int, int, Tuple[int, bool], Tuple[int, int, int]]:
    lut_result = get_best_configs_from_lut(m, n, k, num_groups, gemm_type)
    if lut_result:
        best_block_m, best_block_n, best_block_k, best_warp_m, best_warp_n, best_stages = lut_result
        best_smem_config = get_smem_config(best_stages, k, best_block_m, best_block_n, best_block_k)
        return num_sms, best_block_m, best_block_n, best_block_k, best_warp_m, best_warp_n, best_stages, best_smem_config
    elif gemm_type == GemmType.DenseGemm or gemm_type == GemmType.BatchGemm:
        return get_best_configs_dense(m, n, k, num_groups, num_sms)
    if gemm_type != GemmType.GroupedContiguous:
        block_ms = (256, 192, 128, 64, 32, 16) if k > 384 else (64, 32, 16)
    else:
        block_ms = (get_m_alignment_for_contiguous_layout(), )

    # block_ns = (256, 128, 64, 32)
    assert max_block_n > 0 and (max_block_n & (max_block_n - 1)) == 0
    block_ns = tuple(map(lambda x: 2**x, range(max_block_n.bit_length() - 1, 4, -1))) if k > 384  else tuple(map(lambda x: 2**x, range(max_block_n.bit_length() - 2, 4, -1)))

    fix_wave_saturate = lambda x: num_sms if x == 0 else x
    get_num_waves = lambda bm, bn: (ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms) if bm else None) # groups*m_block*n_block/sms = num waves
    get_last_wave_util = lambda bm, bn: fix_wave_saturate((ceil_div(m, bm) * ceil_div(n, bn) * num_groups) % num_sms)

    # block size wasted
    get_block_utils = lambda m, bm: (((m / bm ) / ((m + bm -1) // bm)) if m % bm != 0 else 1.0) if bm else 0

    # Decide block sizes by waves
    best_block_m, best_block_n = None, None
    best_num_occ = 1
    for block_m in block_ms:
        # NOTES: the block sizes can not be too large, so at least one dim less than 128
        for block_n in filter(lambda bn: ((block_m <= 128 or bn <= 128) and (bn != n and n >= 32) and not (block_m == 16 and bn <= 32)), block_ns):
            success = False
            num_waves, best_num_waves = get_num_waves(block_m, block_n), get_num_waves(best_block_m, best_block_n)
            num_utils = get_block_utils(m, block_m) * get_block_utils(n, block_n)
            best_num_utils = get_block_utils(m, best_block_m) * get_block_utils(n, best_block_n)
            num_occ, best_num_occ = get_smem_occ(block_m, block_n, 128, 2), get_smem_occ(best_block_m, best_block_n, 128, 2)
            # print(f'best_block_m:{best_block_m}, best_block_n:{best_block_n}, num_waves:{num_waves}, best_num_waves:{best_num_waves}\n')
            if best_block_m is None or best_block_n is None:
                success = True
            elif (m < 64 or  n < 512):
                # if single group block is small, balance wave, utils and occ
                occ_wave = num_waves / num_occ
                best_occ_wave = best_num_waves / best_num_occ
                ai_util = (block_m * block_n) / (block_m + block_n)
                best_ai_util = (best_block_m * best_block_n) / (best_block_m + best_block_n)

                valid_occ  = (num_occ / best_num_occ) >= 1
                valid_wave = (occ_wave / best_occ_wave) <= 1
                valid_util = (num_utils / best_num_utils) >= 1
                valid_ai   = (ai_util / best_ai_util) >= 1

                # print(f'num_waves:{num_waves / num_occ}, best_num_waves:{best_num_waves / best_num_occ}')
                # print(f'ai:{ai_util}, best_ai:{best_ai_util}')
                # print(f'util_ratio:{(num_utils / best_num_utils)}, wave_ratio:{occ_wave / best_occ_wave}, occ_ratio:{(num_occ / best_num_occ)}, ai_ratio:{ai_util / best_ai_util}')

                # print(f'valid_occ:{valid_occ}, valid_wave:{valid_wave}, valid_util:{valid_util}, valid_ai:{valid_ai}')
                success = (valid_wave + valid_util + valid_occ + valid_ai) >= 3
            elif num_waves < best_num_waves:
                success = True
            elif num_waves == best_num_waves:
                # Check last wave utilization
                util = get_last_wave_util(block_m, block_n)
                best_util = get_last_wave_util(best_block_m, best_block_n)
                success = util > best_util
                # print(f'best_block_m:{best_block_m}, best_block_n:{best_block_n}, num_waves:{num_waves}, best_num_waves:{best_num_waves}, util:{util}, best_util:{best_util}\n')
                if util == best_util:
                    # Case 1: same `block_m`, smaller `block_n` (wasted)
                    success |= block_m == best_block_m and block_n < best_block_n
                    # Case 2: same `block_n`, smaller `block_m` (wasted)
                    success |= block_n == best_block_n and block_m < best_block_m
                    # Case 3: different for both `block_m` and `block_n`, `block_n` larger is better
                    success |= block_m != best_block_m and block_n > best_block_n
                # print(f'success:{success}\n')
            best_block_m, best_block_n = (block_m, block_n) if success else (best_block_m, best_block_n)
    assert best_block_m is not None and best_block_n is not None

    if gemm_type != GemmType.GroupedContiguous:
        if m > 256 and n >= 256:
            # compute bound, force to best tile
            best_block_m = 192
            best_block_n = 256

        if m >= 128 and k <= 512:
            best_block_m = 64
            best_block_n = 128

        if (best_block_m - 10 <= m <= best_block_m) and best_block_m == 16:
            best_block_m = best_block_m * 2

        if (best_block_m - 10 <= m <= best_block_m) and (best_block_m == 32) and k > 384:
            best_block_m = best_block_m * 2

    # print(f'best_block_m:{best_block_m}, best_block_n:{best_block_n}')


    # Always pick the longest one
    # NOTES: for double B scales, the best number of stages may be reduced
    best_num_stages, best_smem_config, ppu_capacity = None, None, 262144

    block_k = 128
    stage_candidates = tuple(filter(lambda s: s <= k // block_k, (7, 6, 5, 4, 3, 2)))
    if not stage_candidates or (128 % best_block_n != 0 and 128 // math.gcd(128, best_block_n) <= 4) or best_block_m == 16 or best_block_m == 32:
        # Unrolling both stages and `num_former_iters` will cause large code size
        stage_candidates = (3, 2)
    if best_block_m >= 128 and best_block_n >= 128:
        stage_candidates = (4,)
    if (best_block_m == 128 or best_block_m == 64) and best_block_n == 128 and k > 384:
        stage_candidates = (3,)
    best_occ = 0
    for num_stages in stage_candidates:
        best_smem_config = get_smem_config(num_stages, k, best_block_m, best_block_n, block_k)
        if best_smem_config[0] < ppu_capacity:
            occ = ppu_capacity // best_smem_config[0]
            # print(f'occ:{occ}, best_occ:{best_occ}, stages:{num_stages}\n')
            if k < 512 or (best_block_m > 64 and best_block_n >= 64) and occ >= best_occ:
                # compute block and too small-k use higer occ rather than large stage
                best_num_stages = num_stages
                best_occ = occ
            else:
                best_num_stages = num_stages
                break
    assert best_smem_config is not None
    assert best_num_stages is not None


    # Recompute the minimal number of SMs required
    # NOTES: less L2 cache usage and less GPU frequency drop
    num_waves = get_num_waves(best_block_m, best_block_n)
    num_min_sms = ceil_div(ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups, num_waves)
    assert num_min_sms <= num_sms

    warp_m = best_block_m // 2
    warp_n = best_block_n // 2
    if best_block_m == 32 and best_block_n >= 64:
        warp_m = best_block_m // 2
        # best_block_n = 64 if best_block_n == 128 else best_block_n
        warp_n = best_block_n // 2 if best_block_n <= 128 else best_block_n // 4
    elif best_block_n == 32 and n <= 128 and best_block_m >=64:
        warp_m = best_block_m // 4
        warp_n = 32
    elif best_block_m == 192 and best_block_n == 128:
        best_block_n = 256
        warp_m = 48
        warp_n = 64
    elif best_block_m == 128 or best_block_m == 192 or best_block_m == 256 and best_block_n >= 32:
        warp_m = best_block_m // 4
        warp_n = best_block_n // 2 if best_block_n <= 128 else best_block_n // 4
    elif best_block_m == 16:
        warp_m = 16
        best_block_n = 64 if n < 512 else best_block_n
        warp_n = best_block_n // 4 if best_block_n <= 128 else best_block_n // 8
    elif best_block_n == 128 or best_block_n == 256:
        warp_m = best_block_m // 2 if best_block_m != 32 else best_block_m
        warp_n = best_block_n // 4
    # print(f'best_block_m:{best_block_m}, best_block_n:{best_block_n}')


    return num_min_sms, best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages, best_smem_config


def gemm_fp8_fp8_bf16_nt(lhs_: Tuple[torch.Tensor, torch.Tensor],
                         rhs_: Tuple[torch.Tensor, torch.Tensor],
                         out: torch.Tensor, configs = None) -> None:
    """
    Do a normal GEMM with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.
    LHS, RHS, RHS scaling factors, and output tensors must be in contiguous format.
    RHS and RHS scaling factors are required to be transposed.
    The LHS scaling tensor requires TMA-aligned transposed format, if your input does not match the requirement,
        this function will do a transposing with a set of slow PyTorch operations.

    Arguments:
        lhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[m, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[n, k]`.
             the second element is an FP32 128x128 scaling tensor for RHS of shape `[⌈n / 128⌉, ⌈k / 128⌉]`.
        out: the BF16 output tensor of shape `[m, n]`, representing the result.
        configs: The best configs from the framework auto tuning.
    """
    lhs, lhs_scales = lhs_
    rhs, rhs_scales = rhs_
    m, k = lhs.shape
    n, k_ = rhs.shape
    m_, n_ = out.shape

    if lhs_scales.shape == (m, 1) and rhs_scales.shape == (n, 1):
        return gemm_a8w8_per_channel_nt(lhs_, rhs_, out, configs)

    # assert k % 128 == 0

    # Type and shape checks
    assert m == m_ and n == n_ and k == k_
    assert n > 0 and k > 0
    assert lhs_scales.shape == (m, (k + 127) // 128)
    assert rhs_scales.shape == ((n + 127) // 128, (k + 127) // 128)
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert lhs.is_contiguous() and rhs.is_contiguous() and out.is_contiguous()

    # NOTES: `get_tma_aligned_lhs_scales` may launch a kernel if not processed by previous kernels
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)

    assert rhs_scales.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()
    if configs is not None:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(m, n, k, 1, num_sms)

    args = (lhs, lhs_scales, rhs, rhs_scales, out, m, torch.cuda.current_stream(), num_sms, smem_config[0])
    runtime = jit_tuner.compile_and_tune(
        name='gemm_fp8_fp8_bf16_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n,
              'BLOCK_K' : block_k, 'WARP_M' : warp_m, 'WARP_N' : warp_n,
              'SWIZZLE_D_MODE': smem_config[1],
              'BLOCK_N_PADDING': smem_config[2],
              'NUM_STAGES': num_stages},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.float8_e4m3fn), ('lhs_scales', torch.float),
                  ('rhs', torch.float8_e4m3fn), ('rhs_scales', torch.float),
                  ('out', torch.bfloat16), ('m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
        template=template,
        args=args,
        jit_include_dir='actlize_v1.0.0'
    )
    # Run the kernel
    runtime(*args)
