#!/usr/bin/env python3
"""
Deep GEMM Tools
A collection of utility functions for GEMM operations.
"""
import torch
import math
import os
from functools import lru_cache
from typing import Tuple, List, Optional
try:
    from deep_gemm import (
        get_m_alignment_for_contiguous_layout,
        ceil_div,
        get_num_sms
    )
    from deep_gemm.jit_kernels.gemm_int8 import get_smem_config
    import deep_gemm.jit_kernels.m_grouped_gemm_int8
    DEEP_GEMM_AVAILABLE = True
except ImportError:
    DEEP_GEMM_AVAILABLE = False
    print("Warning: deep_gemm not available, some functions will be mocked")
    
@lru_cache(maxsize=None)
def get_best_configs(m: int, n: int, k: int, num_groups: int, num_sms: int,
                     block_ms_lists: Tuple[int], block_ns_lists: Tuple[int],
                     block_k: int) -> Tuple[int, int, int, int, int, int, int, Tuple]:
    """
    Calculate optimal GEMM configuration given matrix dimensions and hardware parameters

    Args:
    m, n, k: Matrix multiplication dimensions (M x K) * (K x N)
    num_groups: Number of groups
    num_sms: Number of SMs on GPU
    block_ms_lists: Optional block_m value list
    block_ns_lists: Optional block_n value list
    block_k: block_k value

    Returns:
    Tuple containing optimal configuration: (num_min_sms, best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages, best_smem_config)
    """
 
    block_ms = block_ms_lists
    block_ns = block_ns_lists
    fix_wave_saturate = lambda x: num_sms if x == 0 else x
    get_num_waves = lambda bm, bn: (ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms) if bm else None)
    get_last_wave_util = lambda bm, bn: fix_wave_saturate((ceil_div(m, bm) * ceil_div(n, bn) * num_groups) % num_sms)
    # Decide block sizes by waves
    best_block_m, best_block_n = None, None
    for block_m in block_ms:
        # NOTES: the block sizes can not be too large, so at least one dim less than 128
        for block_n in block_ns:
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
                if util == best_util:
                    # Case 1: same `block_m`, smaller `block_n` (wasted)
                    success |= block_m == best_block_m and block_n < best_block_n
                    # Case 2: same `block_n`, smaller `block_m` (wasted)
                    success |= block_n == best_block_n and block_m < best_block_m
                    # Case 3: different for both `block_m` and `block_n`, `block_n` larger is better
                    success |= block_m != best_block_m and block_n > best_block_n
            if success:
                best_block_m, best_block_n = (block_m, block_n)
    # Small m hbm bound, wave is not useful, for better occ for 810e hbm bound, use smallest blockN for m16
    if (m <= 16):
        best_block_m = 16
        best_block_n = 64
    
    assert best_block_m is not None and best_block_n is not None
    
    # Always pick the longest one
    # NOTES: for double B scales, the best number of stages may be reduced
    best_num_stages, best_smem_config, ppu_capacity = None, None, 262144
    
    stage_candidates = tuple(filter(lambda s: s <= k // block_k, (8, 7, 6, 5, 4, 3, 2)))
    if not stage_candidates or (128 % best_block_n != 0 and 128 // math.gcd(128, best_block_n) <= 4) or best_block_m == 16:
        # Unrolling both stages and `num_former_iters` will cause large code size
        stage_candidates = (4, 3, 2)
    for num_stages in stage_candidates:
        best_smem_config = get_smem_config(num_stages, k, best_block_m, best_block_n, block_k, 1)
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
    warp_m = best_block_m // 2
    warp_n = best_block_n // 2
    if best_block_m == 32 and m == 32 and best_block_n >= 64:
        warp_m = 32
        warp_n = best_block_n // 4
    elif best_block_n == 32 and n <= 128 and best_block_m >= 64:
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
    return num_min_sms, best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages, best_smem_config
def get_supported_configs(m: int, n: int, k: int, num_groups: int, num_sms: int,
                          is_grouped_contiguous: bool = False, is_grouped_masked: bool = False) -> List[Tuple]:
    """
    Get list of supported configurations

    Args:
    m, n, k: Matrix dimensions
    num_groups: Number of groups
    num_sms: Number of SMs
    is_grouped_contiguous: Whether it's grouped contiguous
    is_grouped_masked: Whether it's masked grouped

    Returns:
    List of configurations
    """
    if not is_grouped_contiguous:
        block_ms = (256, 128, 64, 32, 16)
    else:
        block_ms = (get_m_alignment_for_contiguous_layout(), )
    
    block_ns = (256, 128, 64, 32)
    
    block_ks = (256, 128, 64, 32)
    rst = []
    
    for bm in block_ms:
        for bn in block_ns:
            for bk in block_ks:
                try:
                    num_min_sms, best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages, best_smem_config = get_best_configs(m, n, k, num_groups, num_sms, (bm,), (bn,), bk)
                    rst.append((num_min_sms, best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages, best_smem_config))
                    # we need to try warp_n == 32 and stage == 3
                    if best_block_n == 256 and warp_n == 64:
                        warp_n = 32
                        best_num_stages = 3
                        best_smem_config = get_smem_config(best_num_stages, k, best_block_m, best_block_n, block_k, 1)
                        rst.append((num_min_sms, best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages, best_smem_config))
                except Exception as e:
                    print(f"Failed to find a valid config for {bm}x{bn}x{bk}")
    
    return rst