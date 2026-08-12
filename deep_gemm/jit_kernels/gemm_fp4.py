import math
import torch
import os
from typing import Tuple
import random
import warnings
from functools import lru_cache

from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, is_ppu1v5_device, GemmType
from ..deep_gemm_tuner.autotune_fp4 import lookup_best_config
# C++ code templates
includes = ('"../deep_gemm/fp4_gemm_cutlass3.cuh"', )
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto hasBias = {HAS_BIAS};

// Make a templated grouped GEMM
using gemm_t = Fp4Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, kNumGroups, kNumStages, GemmType::{GEMM_TYPE}, false, hasBias>;
gemm_t::run(lhs, lhs_scales, rhs, rhs_scales,
            bias, out, m, nullptr, nullptr, 0,
            stream, num_sms, smem_size);
"""
tile_config_normal = {
    #(block_m, block_n, block_k):(warp_m, warp_n, stages)
    (16,  64,  128) : (16, 32, 3),
    (16,  64,  256) : (16, 16, 4),
    (32,  128, 128) : (32, 64, 2),
    (32,  64,  128) : (32, 32, 2),
    (64,  64,  128) : (32, 64, 3),
    (64,  128, 128) : (32, 64, 2),
    (128, 128, 128) : (32, 64, 2),
    (128, 256, 64)  : (64, 64, 3)
}
tile_config_smallK = {
    (16,  64,  128) : (16, 16, 2),
    (32,  64,  128) : (16, 32, 2),
    (32, 128,  128) : (16, 64, 2),
    (64, 128,  128) : (32, 64, 2),
    (64, 256,  128) : (32, 64, 2)
}

def get_sf_per_stage_size(block_mn: int, block_k: int) -> Tuple[int, int]:
    bpp = 2                           # bpp=2 means sizeof(uint16)
    smem_sf_k = ceil_div(block_k, 32) # uint16
    base_smem_sf_size = block_mn * smem_sf_k * bpp

    if block_mn <= 64:
        invalid_sf_element_size = 0
        return (base_smem_sf_size, invalid_sf_element_size)
    else:
        aiu_num_on_m = ceil_div(block_mn, 64)
        sf_padding_size = 16 * aiu_num_on_m * bpp    # 16 means padding 16 rows for bankconflicts
        invalid_sf_element_size = 16 * bpp           # -16 means cute::cosize will only reserve spaces until the last valid elemnt

        return (base_smem_sf_size + sf_padding_size, invalid_sf_element_size)

@lru_cache(maxsize=None)
def get_smem_config_fp4(num_stages: int, block_m: int, block_n: int, warp_m: int, warp_n: int, block_k: int = 128, bpp: int = 1) -> Tuple[int, int, int]:
    # Try swizzle first, as it does not waste shared memory
    swizzle_mode = 128
    # block_n_padding = get_block_n_padding_for_smem_d(block_n) if swizzle_mode == 0 else 0
    block_n_padding = 0

    smem_d = block_m * (block_n + block_n_padding)
    smem_a_per_stage = block_m * block_k
    smem_b_per_stage = block_n * block_k
    # smem_barrier = num_stages * 8 * 2
    smem_sfa_per_stage, invalid_sfa_element_size = get_sf_per_stage_size(block_m, block_k)
    smem_sfb_per_stage, invalid_sfb_element_size = get_sf_per_stage_size(block_n, block_k)

    ### output dtype of fp4 is bf16 currently
    smem_size_d = smem_d * 2
    smem_size_a = num_stages * smem_a_per_stage * bpp
    smem_size_b = num_stages * smem_b_per_stage * bpp
    smem_size_sfa = num_stages * smem_sfa_per_stage * bpp - invalid_sfa_element_size
    smem_size_sfb = num_stages * smem_sfb_per_stage * bpp - invalid_sfb_element_size

    smem_size = max(smem_size_d, smem_size_a + smem_size_b + smem_size_sfa + smem_size_sfb)

    # Swizzle and padding are not compatible
    assert int(swizzle_mode > 0) + int(block_n_padding > 0) <= 1

    return smem_size, swizzle_mode, block_n_padding

def get_smem_occ(block_m: int, block_n: int, block_k: int, num_stages: int, warp_m: int, warp_n: int) -> Tuple[int]:
    if block_m is None:
        return 0

    # use static suppose.
    ppu_capacity = 262144
    smem_size = get_smem_config_fp4(num_stages=num_stages, block_m=block_m, block_n=block_n, block_k=block_k, warp_m=warp_m, warp_n=warp_n)[0]

    return ppu_capacity // smem_size

@lru_cache(maxsize=None)
def get_best_configs_dense_ppu1v5(m: int, n: int, k: int, num_groups: int, num_sms: int) -> \
        Tuple[int, int, int, int, int, int, int, int, int, dict]:

    #FIXME: block m can add 16, and blockM/N could be 512, and 48, 96 blockM.
    block_ms = (256, 128, 64, 32, 16)
    block_ns = (256, 128, 64, 32, 16)

    fix_wave_saturate = lambda x: num_sms if x == 0 else x
    get_num_waves = lambda bm, bn: (ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms) if bm else None)
    get_last_wave_util = lambda bm, bn: fix_wave_saturate((ceil_div(m, bm) * ceil_div(n, bn) * num_groups) % num_sms)
    get_block_utils = lambda m, bm: (((m / bm ) / ((m + bm -1) // bm)) if m % bm != 0 else 1.0) if bm else 0
    get_block_ai = lambda block_m, block_n: (block_m * block_n) / (block_m + block_n)

    @lru_cache(maxsize=None)
    def get_warp_mn_dense(m: int, n: int, k: int, block_m_: int, block_n_: int) -> Tuple[int, int, int, int]:
        if block_m_ is None or block_n_ is None:
            return (None, None, None, None)

        warp_m_ = block_m_ // 2
        warp_n_ = block_n_ // 2

        if block_m_ >= 128 and block_n_ == 256:
            warp_m_ = block_m_ // 4
            warp_n_ = block_n_ // 4
        elif block_m_ == 32 and block_n_ >= 64:
            warp_m_ = 32
            warp_n_ = block_n_ // 4
        elif block_n_ == 32 and n <= 128 and block_m_ >= 64:
            warp_m_ = block_m_ // 4
            warp_n_ = 32
        elif block_m_ == 128 or block_m_ == 256 or block_m_ == 192 and block_n_ >= 32:
            warp_m_ = block_m_ // 4
            warp_n_ = block_n_ // 2 if block_n_ != 32 else block_n_
        elif block_m_ == 16:
            warp_m_ = 16
            block_n_ = 64 if n < 512 else block_n_
            warp_n_ = 16
        elif block_n_ == 128 or block_n_ == 256:
            warp_m_ = block_m_ // 2 if block_m_ != 32 else block_m_
            warp_n_ = block_n_ // 4
        elif block_n_ == 16:
            warp_n_ = 16

        return (block_m_, block_n_, warp_m_, warp_n_)

    # Decide block sizes by waves
    best_block_m, best_block_n = None, None
    for block_m in block_ms:
        # NOTES:
        # for PPU1.0: the block sizes can not be too large, so at least one dim less than 128
        # for PPU1.5: the tile 256x256 is good for many compute bound case
        if (m >= 128 and k >= 2048) or m >= 256:
            block_ns_after_filter = filter(lambda bn: (bn != n and n >= 1), block_ns)
        else:
            block_ns_after_filter = \
                filter(lambda bn: ((block_m <= 128 or bn <= 128) and (bn != n and n >= 1) and not (block_m == 16 and bn == 32)), block_ns)
        for block_n in block_ns_after_filter:
            success = False
            num_waves, best_num_waves = get_num_waves(block_m, block_n), get_num_waves(best_block_m, best_block_n)
            num_utils = get_block_utils(m, block_m) * get_block_utils(n, block_n)
            best_num_utils = get_block_utils(m, best_block_m) * get_block_utils(n, best_block_n)

            tmp_block_m, tmp_block_n, tmp_warp_m, tmp_warp_n = get_warp_mn_dense(m, n, k, block_m, block_n)
            tmp_best_block_m, tmp_best_block_n, tmp_best_warp_m, tmp_best_warp_n = get_warp_mn_dense(m, n, k, best_block_m, best_block_n)
            num_occ, best_num_occ = get_smem_occ(tmp_block_m, tmp_block_n, 128, 2, warp_m=tmp_warp_m, warp_n=tmp_warp_n), get_smem_occ(tmp_best_block_m, tmp_best_block_n, 128, 2, warp_m=tmp_best_warp_m, warp_n=tmp_best_warp_n)

            # print(f"block_m:{block_m}, block_n:{block_n}, best_block_m:{best_block_m}, best_block_n:{best_block_n}")
            # print(f'num_occ:{num_occ}, best_num_occ:{best_num_occ}')
            # # print(f'num_waves:{num_waves / num_occ}, best_num_waves:{best_num_waves / best_num_occ}')
            # print(f'm_util:{get_block_utils(m, block_m)}, n_util:{get_block_utils(n, block_n)}, num_utils:{num_utils}')
            # print(f'best_m_util:{get_block_utils(m, best_block_m)}, best_n_util:{get_block_utils(n, best_block_n)}, best_num_utils:{best_num_utils}')

            if best_block_m is None or best_block_n is None:
                success = True
            elif (m <= 256 or n < 512):
                # if single group block is small, balance wave, utils and occ
                occ_wave = num_waves / num_occ
                best_occ_wave = best_num_waves / best_num_occ
                ai_util = get_block_ai(block_m, block_n)
                best_ai_util = get_block_ai(best_block_m, best_block_n)

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
                if util == best_util:
                    # Case 1: same `block_m`, smaller `block_n` (wasted)
                    success |= block_m == best_block_m and block_n < best_block_n
                    # Case 2: same `block_n`, smaller `block_m` (wasted)
                    success |= block_n == best_block_n and block_m < best_block_m
                    # Case 3: different for both `block_m` and `block_n`, `block_n` larger is better
                    success |= block_m != best_block_m and block_n > best_block_n

            # print(f'm:{m}, n:{n}, k:{k}, block_m:{block_m}, block_n:{block_n}, num_waves:{num_waves}, best_num_waves:{best_num_waves}, success:{success}\n')
            # print(f'\n\n\n')
            best_block_m, best_block_n = (block_m, block_n) if success else (best_block_m, best_block_n)

    #small m hbm bound or latency bound, wave is not usful, for better occ for 810e hbm bound, use smallest blockN for m16
    if (m < 20 and n < 512) :
        best_block_m = 16
        best_block_n = 64

    assert best_block_m is not None and best_block_n is not None

    # Always pick the longest one
    # NOTES: for double B scales, the best number of stages may be reduced
    best_num_stages, best_smem_config, ppu_capacity = None, None, 262144

    block_k = 128
    if k <= 256:
        block_k = 64
    if k >= 4096 and ((best_block_m <= 32 and best_block_n <= 64) or (best_block_m == 64 and best_block_n == 128)):
        block_k = 256

    # Recompute the minimal number of SMs required
    # NOTES: less L2 cache usage and less GPU frequency drop
    num_waves = get_num_waves(best_block_m, best_block_n)

    num_min_sms = num_sms

    assert num_min_sms <= num_sms

    best_block_m, best_block_n, warp_m, warp_n = get_warp_mn_dense(m, n, k, best_block_m, best_block_n)

    stage_candidates = tuple(filter(lambda s: s <= k // block_k, (8, 7, 6, 5, 4, 3, 2)))
    ### for those cases with super small k.
    if not stage_candidates: stage_candidates = (2, )

    # if not stage_candidates or (128 % best_block_n != 0 and 128 // math.gcd(128, best_block_n) <= 4):
    #     stage_candidates = (3, 2)
    # if (best_block_m >= 128 and best_block_n >= 128):
    #     stage_candidates = (4,)

    best_occ = 0
    for num_stages in stage_candidates:
        best_smem_config = get_smem_config_fp4(num_stages, best_block_m, best_block_n, warp_m, warp_n, block_k, 1)
        # print(f"num_stages:{num_stages}, best_smem_config:{best_smem_config}")
        if best_smem_config[0] <= ppu_capacity:
            # occ = ppu_capacity // best_smem_config[0]
            # if k < 512 or (best_block_m > 64 and best_block_n >= 64) and occ >= best_occ:
            #     # compute block use higer occ rather than large stage
            #     best_num_stages = num_stages
            #     best_occ = occ
            # else:
            #     # best_num_stages = num_stages
            #     break
            best_num_stages = num_stages
            break

    # best_num_stages = 2
    assert best_smem_config is not None
    assert best_num_stages is not None

    return min(num_min_sms, num_sms), best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages, best_smem_config

@lru_cache(maxsize=None)
def get_best_configs(total_m: int, m: int, n: int, k: int, num_groups: int, num_sms: int,
                     gemm_type: GemmType=GemmType.DenseGemm,
                     max_block_n: int = 256) -> \
        Tuple[int, int, int, int, Tuple[int, bool], Tuple[int, int, int]]:
    assert is_ppu1v5_device(), "mxfp4 is noly supported on PPU-ZW890"

    # todo: add lut logic
    # use_heuristic = os.environ.get('USE_HEUR', 'False').lower() == 'true'
    # if not use_heuristic:
    #   if is_grouped_masked:
    #       configs = lookup_best_config(m, n, k * 2, num_groups, is_grouped_masked)
    #   else:
    #       configs = lookup_best_config(total_m, n, k * 2, num_groups, False)
    #   if configs is not None:
    #       return configs
    if gemm_type == GemmType.DenseGemm:
        return get_best_configs_dense_ppu1v5(m, n, k, num_groups, num_sms)

    block_ms = (256, 128, 64, 32, 16) if k > 768 else (128, 64, 32, 16)
    # block_ns = (256, 128, 64, 32)
    assert max_block_n > 0 and (max_block_n & (max_block_n - 1)) == 0
    block_ns = tuple(map(lambda x: 2**x, range(max_block_n.bit_length() - 1, 4, -1))) if k >= 384 else tuple(map(lambda x: 2**x, range(max_block_n.bit_length() - 2, 4, -1)))

    fix_wave_saturate = lambda x: num_sms if x == 0 else x
    get_num_waves = lambda bm, bn: (ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms) if bm else None)
    get_last_wave_util = lambda bm, bn: fix_wave_saturate((ceil_div(m, bm) * ceil_div(n, bn) * num_groups) % num_sms)

    get_block_utils = lambda m, bm: (((m / bm ) / ((m + bm -1) // bm)) if m % bm != 0 else 1.0) if bm else 0
    get_block_ai = lambda block_m, block_n: (block_m * block_n) / (block_m + block_n)

    @lru_cache(maxsize=None)
    def get_warp_mn_grouped(m: int, n: int, k: int, block_m_: int, block_n_: int) -> Tuple[int, int, int, int]:
        if block_m_ is None or block_n_ is None:
            return (None, None, None, None)

        warp_m_ = block_m_ // 2
        warp_n_ = block_n_ // 2

        if block_m_ > 128 and block_n_ == 256:
            warp_m_ = block_m_ // 4
            warp_n_ = block_n_ // 4
        elif block_m_ == 64 and block_n_ >= 128:
            warp_m_ = 64 if k < 256 else 32
            warp_n_ = 64
        elif block_m_ == 32 and block_n_ >= 64:
            warp_m_ = 32
            warp_n_ = block_n_ // 2 if block_n_ <= 128 else block_n_ // 4
        elif block_n_ == 32 and n <= 128 and block_m_ >=64:
            warp_m_ = block_m_ // 4
            warp_n_ = 32
        elif block_m_ == 128 or block_m_ == 256 and block_n_ >= 32:
            warp_m_ = 64
            warp_n_ = block_n_ // 2 if block_n_ <= 128 else block_n_ // 4
        elif block_m_ == 16:
            warp_m_ = 16
            block_n_ = 64 if n < 512 else block_n_
            warp_n_ = 16
        elif block_n_ == 128 or block_n_ == 256:
            warp_m_ = block_m_ // 2 if block_m_ != 32 else block_m_
            warp_n_ = 64
        elif block_n_ == 16:
            warp_n_ = 16

        return (block_m_, block_n_, warp_m_, warp_n_)

    # Decide block sizes by waves
    best_block_m, best_block_n = None, None
    for block_m in block_ms:
        # NOTES:
        # for PPU1.0: the block sizes can not be too large, so at least one dim less than 128
        # for PPU1.5: the tile 256x256 is good for many compute bound case
        if (m >= 96 and k > 1024) or (m >= 256 and k >= 512):
            block_ns_after_filter = filter(lambda bn: (bn != n and n >= 32) and not (block_m == 16 and bn <= 32), block_ns)
        else:
            block_ns_after_filter = \
                filter(lambda bn: ((block_m <= 128 or bn <= 128) and (bn != n and n >= 32) and not (block_m == 16 and bn <= 32)), block_ns)
        for block_n in block_ns_after_filter:
            success = False
            num_waves, best_num_waves = get_num_waves(block_m, block_n), get_num_waves(best_block_m, best_block_n)
            num_utils = get_block_utils(m, block_m) * get_block_utils(n, block_n)
            best_num_utils = get_block_utils(m, best_block_m) * get_block_utils(n, best_block_n)

            tmp_block_m, tmp_block_n, tmp_warp_m, tmp_warp_n = get_warp_mn_grouped(m, n, k, block_m, block_n)
            tmp_best_block_m, tmp_best_block_n, tmp_best_warp_m, tmp_best_warp_n = get_warp_mn_grouped(m, n, k, best_block_m, best_block_n)
            num_occ, best_num_occ = get_smem_occ(tmp_block_m, tmp_block_n, 128, 2, warp_m=tmp_warp_m, warp_n=tmp_warp_n), get_smem_occ(tmp_best_block_m, tmp_best_block_n, 128, 2, warp_m=tmp_best_warp_m, warp_n=tmp_best_warp_n)

            # print(f"block_m:{block_m}, block_n:{block_n}, best_block_m:{best_block_m}, best_block_n:{best_block_n}")
            # print(f'num_occ:{num_occ}, best_num_occ:{best_num_occ}')
            # # print(f'num_waves:{num_waves / num_occ}, best_num_waves:{best_num_waves / best_num_occ}')
            # print(f'm_util:{get_block_utils(m, block_m)}, n_util:{get_block_utils(n, block_n)}, num_utils:{num_utils}')
            # print(f'best_m_util:{get_block_utils(m, best_block_m)}, best_n_util:{get_block_utils(n, best_block_n)}, best_num_utils:{best_num_utils}')

            if best_block_m is None or best_block_n is None:
                success = True
            elif (m < 512 or n < 512):
                # if single group block is small, balance wave, utils and occ
                occ_wave = num_waves
                best_occ_wave = best_num_waves
                ai_util = get_block_ai(block_m, block_n)
                best_ai_util = get_block_ai(best_block_m, best_block_n)

                valid_occ  = (num_occ / best_num_occ) >= 1
                valid_wave = (occ_wave / best_occ_wave) <= 1
                valid_util = (num_utils / best_num_utils) >= 1
                valid_ai   = (ai_util / best_ai_util) >= 1

                # print(f'num_waves:{num_waves / num_occ}, best_num_waves:{best_num_waves / best_num_occ}')
                # print(f'ai:{ai_util}, best_ai:{best_ai_util}')
                # print(f'util_ratio:{(num_utils / best_num_utils)}, wave_ratio:{occ_wave / best_occ_wave}, occ_ratio:{(num_occ / best_num_occ)}, ai_ratio:{ai_util / best_ai_util}')

                # print(f'valid_occ:{valid_occ}, valid_wave:{valid_wave}, valid_util:{valid_util}, valid_ai:{valid_ai}')
                # success = (valid_wave + valid_util + valid_occ + valid_ai) >= 3
                success = (valid_wave + valid_util + valid_ai) >= 2
            elif num_waves < best_num_waves:
                success = True
            elif num_waves == best_num_waves:
                # Check last wave utilization
                util = get_last_wave_util(block_m, block_n)
                best_util = get_last_wave_util(best_block_m, best_block_n)
                success = util > best_util
                if util == best_util:
                    # Case 1: same `block_m`, smaller `block_n` (wasted)
                    success |= block_m == best_block_m and block_n < best_block_n
                    # Case 2: same `block_n`, smaller `block_m` (wasted)
                    success |= block_n == best_block_n and block_m < best_block_m
                    # Case 3: different for both `block_m` and `block_n`, `block_n` larger is better
                    success |= block_m != best_block_m and block_n > best_block_n

            # print(f'm:{m}, n:{n}, k:{k}, block_m:{block_m}, block_n:{block_n}, num_waves:{num_waves}, best_num_waves:{best_num_waves}, success:{success}\n')
            # print(f'\n\n\n')
            best_block_m, best_block_n = (block_m, block_n) if success else (best_block_m, best_block_n)

    if (best_block_m - 10 <= m <= best_block_m) and best_block_m == 16:
        best_block_m = best_block_m * 2

    # qwen3-next & deepseek gemm2 need to fix some issues
    if (best_block_m - 10 <= m <= best_block_m) and (best_block_m == 32 or best_block_m == 64) and k >= 192:
        best_block_m = best_block_m * 2

    assert best_block_m is not None and best_block_n is not None

    # Always pick the longest one
    # NOTES: for double B scales, the best number of stages may be reduced
    best_num_stages, best_smem_config, ppu_capacity = None, None, 262144

    block_k = 64
    if (best_block_m <= 32 and best_block_n <= 128):
        if (best_block_n <= 64 or k <= 128):
            block_k = 128
    if (best_block_m == 256 and best_block_n == 256):
        block_k = 128
    if k <= 256:
        block_k = 128
        # for deepseek-pro tp8 gemm2
        if k == 192 and best_block_m == 128 and best_block_n == 128:
            block_k = 64

    if(m < 6 and n >= 512):
        if m < 2:
            best_block_m = 16
        else:
            best_block_m = 32
        best_block_n = 64
        block_k = 128
    # for deepseek-v4 pro tp16 gemm1 smallN
    if(m < 2 and n < 512):
        best_block_m = 16
        best_block_n = 64
        block_k = 256
    # for DeepSeek-V4 Pro EP
    if (n == 6144 and k == 3584) or (n == 7168 and k == 1536):
        if (m < 6):
            if (best_block_m == 32 and best_block_n == 64):
                best_block_m, best_block_n, block_k = 64, 64, 128
                if (k == 1536):
                    best_block_n = 128
        elif (m >= 6) and (best_block_m in [32, 64] and best_block_n == 256):
            best_block_m, best_block_n, block_k = 128, 256, 64
    tile_config = tile_config_normal
    if (k < 128) and not (best_block_n >= 128 and best_block_m >= 128):
        tile_config = tile_config_smallK

    warp_stage = tile_config.get((best_block_m, best_block_n, block_k))
    if warp_stage:
        warp_m, warp_n, num_stages = warp_stage
        best_smem_config = get_smem_config_fp4(num_stages, best_block_m, best_block_n, warp_m, warp_n, block_k, 1)
        return num_sms, best_block_m, best_block_n, block_k, warp_m, warp_n, num_stages, best_smem_config

    #todo: opt this logic
    if (m >= 520 and k <= 768):
        if best_block_m == 128 and best_block_n == 256:
            best_block_m, best_block_n = best_block_n, best_block_m
            stage_candidates = (3, )

    num_waves = get_num_waves(best_block_m, best_block_n)

    num_min_sms = num_sms

    assert num_min_sms <= num_sms

    best_block_m, best_block_n, warp_m, warp_n = get_warp_mn_grouped(m, n, k, best_block_m, best_block_n)

    stage_candidates = tuple(filter(lambda s: s <= k // block_k, (8, 7, 6, 5, 4, 3, 2)))

    ### for those cases with super small k.
    if not stage_candidates: stage_candidates = (2, )
    if best_block_m <= 32: stage_candidates = (3, 2)
    if best_block_m > 32: stage_candidates = (3,)
    if best_block_m >= 128 and best_block_n >= 128: stage_candidates = (3, 4)
    if best_block_m == 128 and best_block_n == 256:
        if k >= 768:
            stage_candidates = (4,)
        else:
            stage_candidates = (3,)
    # if not stage_candidates or (128 % best_block_n != 0 and 128 // math.gcd(128, best_block_n) <= 4) or best_block_m == 16 or best_block_m == 32:
    #     stage_candidates = (3, 2)
    # if best_block_m > 128 and best_block_n == 256:
    #     stage_candidates = (4,)

    best_occ = 0
    for num_stages in stage_candidates:
        best_smem_config = get_smem_config_fp4(num_stages, best_block_m, best_block_n, warp_m, warp_n, block_k, 1)
        # print(f"num_stages:{num_stages}, best_smem_config:{best_smem_config}")
        if best_smem_config[0] <= ppu_capacity:
            occ = ppu_capacity // best_smem_config[0]
            # if k < 512 or (best_block_m > 32 and best_block_n >= 64) and occ >= best_occ:
            #     # compute block use higer occ rather than large stage
            #     best_num_stages = num_stages
            #     best_occ = occ
            # else:
            #     best_num_stages = num_stages
            #     break
            best_num_stages = num_stages
            break

    assert best_smem_config is not None
    assert best_num_stages is not None

    # (best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages) = (16, 64, 256, 16, 16, 4)
    # print(best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages)

    return min(num_min_sms, num_sms), best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages, best_smem_config

def check_mxfp4_scales_layout(scale: torch.Tensor) -> bool:
    """
        check whether the layout of the MXFP4 scale is satisfied with the requirement.

        NOTE: Only the canonical packed layouts are accepted: an M/N-major scale as produced by
        `preprocess_mxfp4_scales`, or its degenerate form when a dim has extent 1. Tensors
        whose strides carry padding (for example a row slice of a larger buffer) are
        rejected on purpose, even though they may be M/N-major.
    """

    # A dim of extent 1 makes the transpose/permute round-trip in preprocess_mxfp4_scales
    # a no-op, so the canonical layout of a degenerate shape is the plain contiguous one.
    target_stride = None
    if scale.dim() == 2:
        MN, K = scale.shape
        target_stride = (1, MN) if (MN > 1 and K > 1) else (K, 1)
    elif scale.dim() == 3:
        G, MN, K = scale.shape
        target_stride = (MN * K, 1, MN) if (MN > 1 and K > 1) else (MN * K, K, 1)
    is_mn_major = (scale.stride() == target_stride)
    if scale.dtype == torch.uint16 and is_mn_major:
        return True

    return False

def uint8_padding(scale: torch.Tensor) -> torch.Tensor:
    """
        forward compatible interface for release_2v1. might be removed later.
    """
    if not torch.compiler.is_compiling():
        warnings.warn("uint8_padding is deprecated and will be remove in DeepGemm later!!! Please use preprocess_mxfp4_scales to preprocess SFA instead.", DeprecationWarning, stacklevel=2)

    return preprocess_mxfp4_scales(scale=scale)

def _post_preprocess_mxfp4_scales(scale: torch.Tensor) -> torch.Tensor:
    """
        Internal interface for forward compatibility interface for release_2v1. might be removed later.
    """
    if scale.dtype == torch.uint16 and scale.is_contiguous() and scale.dim() == 3:
        if not torch.compiler.is_compiling():
            warnings.warn("called preprocess_mxfp4_scales for a grouped tensor(dim=3) before torch.stack(), which might lower the preprocess performance. Please use preprocess_mxfp4_scales on the tensor after torch.stack() directly!", UserWarning, stacklevel=3)
        return scale.permute(0, 2, 1).contiguous().permute(0, 2, 1)

    return scale

def preprocess_mxfp4_scales(scale: torch.Tensor) -> torch.Tensor:
    ### make scale contiguous for SGLang warmup.
    if not scale.is_contiguous(): scale = scale.contiguous()
    if scale.dtype == torch.uint16: scale = scale.view(torch.uint8)
    assert scale.dtype == torch.uint8, f"The dtype of scale to be preprocessed in MXFP4 should be torch.uint8 but got {scale.dtype}."
    assert scale.dim() == 2 or scale.dim() == 3, f"The rank of scale to be preprocessed in MXFP4 should be 2(dense/groupedNoPad) or 3(groupedMasked)."
    if (scale.shape[-1] % 2):
        scale = torch.nn.functional.pad(scale, (0, 1))
    assert scale.shape[-1] % 2 == 0, f'The dim of contiguous must be even number for being viewed as b16.'

    if scale.dim() == 2:
        return scale.view(torch.uint16).t().contiguous().t()
    else:
        return scale.view(torch.uint16).permute(0, 2, 1).contiguous().permute(0, 2, 1)

def preprocess_mxfp4_weight_for_act_and_quant_fusing(weight: torch.Tensor, weight_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert weight.dim() == weight_scale.dim() == 3, "The dimensions of weight and weight_scale must be 3."
    assert weight.dtype == torch.uint8 and weight_scale.dtype == torch.uint8, "The dtype of weight and weight_scale must be uint8, preprocess_mxfp4_weight_for_act_and_quant_fusing should be called before preprocess_mxfp4_scales."
    assert weight.is_contiguous() and weight_scale.is_contiguous(), "weight and weight_scale must be contiguous. preprocess_mxfp4_weight_for_act_and_quant_fusing should be called before preprocess_mxfp4_scales."

    ### do interleaving to make up and gate be adjecent.
    num_groups, n, k = weight.shape
    assert n % 2 == 0, "N must be divideable by 2 for silu_and_mul."
    half_n = n // 2
    gate = weight[:, :half_n, :]
    up = weight[:, half_n:, :]
    weight_out = torch.stack([gate, up], dim=2).view(num_groups, n, k)

    num_groups_, sfn, sfk = weight_scale.shape
    assert num_groups == num_groups_ and n == sfn and sfk == ceil_div(k, 16) # weight is uint8
    gate_scale = weight_scale[:, :half_n, :]
    up_scale = weight_scale[:, half_n:, :]
    weight_scale_out = torch.stack([gate_scale, up_scale], dim=2).view(num_groups_, sfn, sfk)
    weight_scale_out = preprocess_mxfp4_scales(scale=weight_scale_out)
    return (weight_out, weight_scale_out)

def gemm_fp4_fp4_bf16_nt(lhs_: Tuple[torch.Tensor, torch.Tensor],
                         rhs_: Tuple[torch.Tensor, torch.Tensor],
                         bias: torch.Tensor, out: torch.Tensor, configs = None) -> None:
    lhs, lhs_scales = lhs_
    rhs, rhs_scales = rhs_
    m, k = lhs.shape
    n, k_ = rhs.shape
    m_, n_ = out.shape

    if (not check_mxfp4_scales_layout(scale=lhs_scales)):
        if not torch.compiler.is_compiling():
            warnings.warn("[DeepGemm] Called preprocess_mxfp4_scales for SFA inner DenseGemm interface.", UserWarning, stacklevel=3)
        lhs_scales = preprocess_mxfp4_scales(scale=lhs_scales)
    if (not check_mxfp4_scales_layout(scale=rhs_scales)):
        if not torch.compiler.is_compiling():
            warnings.warn(("[DeepGemm] Called preprocess_mxfp4_scales for SFB inner DenseGemm interface. "
                          "preprocess the weight scale might degrade the performance!"), UserWarning, stacklevel=3)
        ### forward compatibility for release_2v1
        rhs_scales = _post_preprocess_mxfp4_scales(scale=rhs_scales)
        if (not check_mxfp4_scales_layout(scale=rhs_scales)):
            rhs_scales = preprocess_mxfp4_scales(scale=rhs_scales)

    # Type and shape checks
    assert m == m_ and n == n_ and k == k_
    assert n > 0 and k > 0
    assert lhs.dtype == torch.uint8 and rhs.dtype == torch.uint8
    assert (bias is None) or (bias.dtype == torch.float32)
    assert out.dtype == torch.bfloat16
    assert lhs.is_contiguous() and rhs.is_contiguous() and out.is_contiguous()
    assert check_mxfp4_scales_layout(scale=lhs_scales) and check_mxfp4_scales_layout(scale=rhs_scales)
    has_bias = True
    if bias is None: bias = torch.empty(0, dtype=torch.float32, device=lhs.device); has_bias = False

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()

    # TODO: enable fp4 get_best_configs
    if configs is not None:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        # import ipdb; ipdb.set_trace()
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(m, m, n, k, 1, num_sms)
        # num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages = (num_sms, 256, 256, 128, 64, 64, 3)
        # smem_config = get_smem_config_fp4(num_stages, block_m, block_n, warp_m, warp_n, block_k)

    args = (lhs, lhs_scales, rhs, rhs_scales, bias, out, m, torch.cuda.current_stream(), num_sms, smem_config[0])
    runtime = jit_tuner.compile_and_tune(
        name='gemm_fp4_fp4_bf16_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
              'WARP_M': warp_m, 'WARP_N': warp_n, 'NUM_GROUPS': 1,
              'NUM_STAGES': num_stages,'GEMM_TYPE': 'DenseGemm', 'HAS_BIAS': has_bias},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.uint8), ('lhs_scales', torch.uint16),
                  ('rhs', torch.uint8), ('rhs_scales', torch.uint16),
                  ('bias', torch.float32), ('out', torch.bfloat16),
                  ('m', int), ('stream', torch.cuda.Stream),
                  ('num_sms', int), ('smem_size', int)),
        template=template,
        args=args,
        jit_include_dir='actlize_v1.0.0'
    )

    # Run the kernel
    runtime(*args)