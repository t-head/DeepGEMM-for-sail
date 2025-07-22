import math
import torch
import os
from functools import lru_cache
from typing import Tuple
import re

from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, get_m_alignment_for_contiguous_layout

# C++ code templates
includes = ('"deep_gemm/int8_gemm.cuh"', )
includes_cutlass3 = ('"../deep_gemm/int8_gemm_cutlass3.cuh"', )
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto kNumGroups = 1;
constexpr auto kNumStages = {NUM_STAGES};

// Make a templated grouped GEMM
using gemm_t = Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, kNumGroups, kNumStages, GemmType::Normal>;

// Launch kernel
gemm_t::run(out, nullptr,
            m, 0, lhs, lhs_scales, rhs, rhs_scales,
            stream, num_sms, smem_size);
"""

def get_smem_config(num_stages: int, k: int, block_m: int, block_n: int, block_k: int = 128, bpp: int = 2) -> Tuple[int, int, int]:
    # Try swizzle first, as it does not waste shared memory
    swizzle_mode = 128
    # block_n_padding = get_block_n_padding_for_smem_d(block_n) if swizzle_mode == 0 else 0
    block_n_padding = 0

    smem_d = block_m * (block_n + block_n_padding)
    smem_a_per_stage = block_m * block_k
    # smem_scales_a_per_stage = block_m * 4
    smem_b_per_stage = block_n * block_k
    # smem_scales_b = ceil_div(k, block_k) * 4
    # smem_barrier = num_stages * 8 * 2

    # smem_size = 0
    smem_size_d = smem_d * 2
    smem_size_a = num_stages * smem_a_per_stage * bpp
    smem_size_b = num_stages * smem_b_per_stage * bpp

    smem_size = max(smem_size_d, smem_size_a + smem_size_b)
    # smem_size += num_stages * smem_a_per_stage
    # smem_size += num_stages * smem_scales_a_per_stage
    # smem_size += num_stages * smem_b_per_stage
    # smem_size += ceil_div(smem_scales_b * (1 if block_k % block_n == 0 else 2), 8) * 8
    # smem_size += smem_barrier

    # Swizzle and padding are not compatible
    assert int(swizzle_mode > 0) + int(block_n_padding > 0) <= 1

    return smem_size, swizzle_mode, block_n_padding

@lru_cache(maxsize=None)
def get_best_configs(m: int, n: int, k: int, num_groups: int, num_sms: int,
                     is_grouped_contiguous: bool = False, is_grouped_masked: bool = False) -> \
        Tuple[int, int, int, int, Tuple[int, bool], Tuple[int, int, int]]:
 
    if not is_grouped_contiguous:
        block_ms = (256, 128, 64, 32, 16)
    else:
        block_ms = (get_m_alignment_for_contiguous_layout(), )

    block_ns = (256, 128, 64, 32)

    fix_wave_saturate = lambda x: num_sms if x == 0 else x
    get_num_waves = lambda bm, bn: (ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms) if bm else None)
    get_last_wave_util = lambda bm, bn: fix_wave_saturate((ceil_div(m, bm) * ceil_div(n, bn) * num_groups) % num_sms)

    # Decide block sizes by waves
    best_block_m, best_block_n = None, None
    for block_m in block_ms:
        # NOTES: the block sizes can not be too large, so at least one dim less than 128
        for block_n in filter(lambda bn: block_m <= 128 or bn <= 128, block_ns):
            success = False
            num_waves, best_num_waves = get_num_waves(block_m, block_n), get_num_waves(best_block_m, best_block_n)

            if best_block_m is None or best_block_n is None:
                success = True
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

    #small m hbm bound, wave is not usful, for better occ for 810e hbm bound, use smallest blockN for m16
    if (m <=24) :
        best_block_m = 16
        best_block_n = 64
    
    assert best_block_m is not None and best_block_n is not None
    
    # Always pick the longest one
    # NOTES: for double B scales, the best number of stages may be reduced
    best_num_stages, best_smem_config, ppu_capacity = None, None, 262144

    # print(f'best_block_m:{best_block_m}, best_block_n:{best_block_n}')

    block_k = 128
    if k <= 48 * 2:
        block_k = 64
    if k >= 4096 and (best_block_m == 32 and best_block_n == 32):
        block_k = 256
 
    stage_candidates = tuple(filter(lambda s: s <= k // block_k, (8, 7, 6, 5, 4, 3, 2)))

    if not stage_candidates or (128 % best_block_n != 0 and 128 // math.gcd(128, best_block_n) <= 4) or best_block_m == 16:
        # Unrolling both stages and `num_former_iters` will cause large code size
        stage_candidates = (3, 2)

    # print(f'stage_candidates:{stage_candidates}')

    for num_stages in stage_candidates:
        best_smem_config = get_smem_config(num_stages, k, best_block_m, best_block_n, block_k, 1)
        # print(f"num_stages:{num_stages}, best_smem_config:{best_smem_config}")
        if best_smem_config[0] < ppu_capacity:
            best_num_stages = num_stages
            break
    assert best_smem_config is not None
    assert best_num_stages is not None

    # Recompute the minimal number of SMs required
    # NOTES: less L2 cache usage and less GPU frequency drop
    num_waves = get_num_waves(best_block_m, best_block_n)
    num_min_sms = ceil_div(ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups, num_waves)
    assert num_min_sms <= num_sms

    # print(f'num_waves:{num_waves}, num_sms:{num_sms}\n')
    # print(f'm:{m}, n:{n}, best_block_m:{best_block_m}, best_block_n:{best_block_n}, num_groups:{num_groups}\n')
    # print(f'num_min_sms:{num_min_sms}\n')

    warp_m = best_block_m // 2
    warp_n = best_block_n // 2

    if best_block_m == 32 and m == 32 and best_block_n >= 64:
        warp_m = 32
        warp_n = best_block_n // 4
    elif best_block_n == 32 and n <= 128 and best_block_m >=64:
        warp_m = best_block_m // 4
        warp_n = 32
    elif best_block_m == 128 or best_block_m == 256 and best_block_n >= 32:
        warp_m = best_block_m // 4
        warp_n = best_block_n // 2 if best_block_n != 32 else best_block_n
    elif best_block_m == 16:
        warp_m = 16
        best_block_n = 64
        warp_n = best_block_n // 4 if best_block_n <= 128 else best_block_n // 8
    elif best_block_n == 128 or best_block_n == 256:
        warp_m = best_block_m // 2 if best_block_m != 32 else best_block_m
        warp_n = best_block_n // 4

    if num_groups == 1 and m <= 16:
        num_min_sms = 20
        if k >= 7168:
            # memory bound
            (best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages) = (16, 64, 256, 16, 16, 4)
        elif k <= 512:
            # latency bound
            (best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages) = (16, 128, 64, 16, 32, 2)
        else:
            (best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages) = (16, 128, 128, 16, 32, 4)

    # (best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages) = (16, 64, 256, 16, 16, 4)
    # print(best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages)

    extra_info = {}
    use_cutlass3 = False
    use_multistage_on_N = False
    if 'DG_USE_CUTLASS3' in os.environ:
        use_cutlass3 = int(os.getenv('DG_USE_CUTLASS3'))
    extra_info['use_cutlass3'] = use_cutlass3
    if 'DG_USE_MULTISTAGE_ON_N' in os.environ:
        use_multistage_on_N = int(os.getenv('DG_USE_MULTISTAGE_ON_N'))
    extra_info['use_multistage_on_N'] = use_multistage_on_N

    return num_min_sms, best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages, best_smem_config, extra_info

def generate_search_space():
    tile_list = [
        [16, 64, 64, 3],
        [16, 64, 128, 3],
        [16, 64, 256, 2],
        [16, 64, 256, 3],
        [16, 64, 512, 2],
        [16, 64, 512, 3],
        [16, 64, 64, 4],
        [16, 64, 128, 4],
        [16, 64, 256, 4],
        [16, 64, 512, 4],

        [16, 128, 128, 3],
        [16, 128, 256, 3],
        [16, 128, 128, 4],
        [16, 128, 256, 4],
        [16, 128, 128, 5],
        [16, 128, 256, 5],

        [16, 128, 64, 2],
        [16, 128, 64, 3],
        [16, 256, 64, 2],
        [16, 256, 64, 3],
    ]
    space = []
    for tile in tile_list:
        config = {'BLOCK_M': tile[0], 'BLOCK_N': tile[1], 'BLOCK_K': tile[2],
              'WARP_M': tile[0], 'WARP_N': tile[1] // 4,
              'NUM_STAGES': tile[3]}
        space.append(config)
    return space

def gemm_int8_int8_bf16_nt(lhs: Tuple[torch.Tensor, torch.Tensor],
                         rhs: Tuple[torch.Tensor, torch.Tensor],
                         out: torch.Tensor) -> None:
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    n, k_ = rhs.shape
    m_, n_ = out.shape

    # Type and shape checks
    assert m == m_ and n == n_ and k == k_
    assert n > 0 and k > 0
    assert lhs.dtype == torch.int8 and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.int8 and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert lhs.is_contiguous() and rhs.is_contiguous() and out.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    global includes, template

    num_sms = get_num_sms()
    num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config, extra_info = get_best_configs(m, n, k, 1, num_sms)

    args = (lhs, lhs_scales, rhs, rhs_scales, out,
            m, torch.cuda.current_stream(), num_sms, smem_config[0])

    template_updated = template
    if extra_info['use_multistage_on_N']:
        template_updated = template.replace("GemmType::Normal>", "GemmType::Normal,1>")

    runtime = jit_tuner.compile_and_tune(
        name='gemm_int8_int8_bf16_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
              'WARP_M': warp_m, 'WARP_N': warp_n,
              'NUM_STAGES': num_stages},
        space=(),
        # space=generate_search_space(),
        includes=includes_cutlass3 if extra_info['use_cutlass3'] else includes,
        arg_defs=(('lhs', torch.int8), ('lhs_scales', torch.float),
                  ('rhs', torch.int8), ('rhs_scales', torch.float),
                  ('out', torch.bfloat16), ('m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
        template=template_updated,
        jit_include_dir='cutlass3' if extra_info['use_cutlass3'] else None,
        args=args
    )

    # Run the kernel
    runtime(*args)
