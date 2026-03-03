#!/usr/bin/env python3
"""
Autotune for Grouped GEMM Kernel.
Searches over tile configurations [block_m, block_n, warp_m, warp_n, block_k, stage]
and saves the best config as a LUT file for fast lookup.
"""

import itertools
import json
import os
from functools import lru_cache
from typing import List, Dict, Optional, Tuple, Any
import copy

import numpy as np
import torch
import deep_gemm
import traceback
from deep_gemm import uint8_padding, ppu_cutlass_mxfp4_scales_swizzle, get_num_sms
from deep_gemm.jit_kernels.gemm_fp4 import get_smem_config_fp4
import concurrent.futures

from triton_kernels.numerics_details.mxfp import downcast_to_mxfp, upcast_from_mxfp


CANDIDATE_Ms = sorted(set(
    list(range(1, 16))
    + [int(x) for x in np.logspace(np.log2(8), np.log2(32768), base=2, num=36)]
    + [16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768]
))

def get_nk_pairs(
    hidden_size: int,
    moe_intermediate_size: int,
    tp_size: int,
) -> List[Tuple[int, int]]:
    nk_pairs = [
        (int(moe_intermediate_size * 2 / tp_size), hidden_size),
        (hidden_size, int(moe_intermediate_size / tp_size)),
    ]
    return nk_pairs

def get_all_cases(
    dense_nk_pairs: Optional[List[Tuple[int, int]]] = None,
    hidden_size: Optional[int] = None,
    moe_intermediate_size: Optional[int] = None,
    tp_size: Optional[int] = None,
    gemm_type: str = "GroupedNoPad",
    topk_experts: int = 1,
) -> List[Dict]:
    cases = []
    if gemm_type == "DenseGemm":
        assert dense_nk_pairs is not None, "DenseGemm must provide dense_nk_pairs"
        for N, K in dense_nk_pairs:
            for M in CANDIDATE_Ms:
                cases.append({"M": M, "N": N, "K": K, "topk_experts": topk_experts})
    elif gemm_type == "GroupedNoPad" or gemm_type == "GroupedMasked":
        assert hidden_size is not None and moe_intermediate_size is not None and tp_size is not None
        nk_pairs = get_nk_pairs(hidden_size, moe_intermediate_size, tp_size)
        for M in CANDIDATE_Ms:
            for N, K in nk_pairs:
                cases.append({"M": M, "N": N, "K": K, "topk_experts": topk_experts})
    else:
        raise ValueError(f"Unsupported gemm_type: {gemm_type}")
    return cases

def config_to_str(config: List[int]) -> str:
    return (f"[block_m={config[1]}, block_n={config[2]}, block_k={config[3]}, "
            f"warp_m={config[4]}, warp_n={config[5]}, num_stages={config[6]}]")

def construct_fp4_dense(
    M: int,
    N: int,
    K: int,
):
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda').contiguous()
    B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda').contiguous()
    x = downcast_to_mxfp(A, torch.uint8, axis=1)
    y = downcast_to_mxfp(B, torch.uint8, axis=1)
    bias = torch.randn(1, N, dtype=torch.float32, device='cuda')
    out = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    ref_out = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    x_scale = uint8_padding(x[1])
    y_scale = ppu_cutlass_mxfp4_scales_swizzle(scale=y[1])
    x = x[0], x_scale
    y = y[0], y_scale
    return x, y, bias, out, ref_out


def construct_fp4_nopad(
    N: int,
    K: int,
    m_rows: torch.Tensor,
    num_experts: int,
    device: str = "cuda",
):
    total_tokens = int(m_rows.sum().item())
    dtype = torch.bfloat16
    x = torch.randn(total_tokens, K, dtype=dtype, device=device)
    y = torch.randn(num_experts, N, K, dtype=dtype, device=device)
    bias = torch.randn((num_experts, N), dtype=torch.float, device=device)
    out = torch.empty(total_tokens, N, dtype=dtype, device=device)
    ref_out = torch.empty(total_tokens, N, dtype=dtype, device=device)

    x_fp4 = downcast_to_mxfp(x.cuda(), torch.uint8, axis=1)
    x_fp4_scale = uint8_padding(x_fp4[1])
    y_fp4 = (torch.empty((num_experts, N, int(K / 2)), device='cuda', dtype=torch.uint8),
             torch.empty((num_experts, N, int(K / 32)),
                         device='cuda', dtype=torch.uint8),
             )
    y_scale = []
    for i in range(num_experts):
        y_fp4[0][i], y_fp4[1][i] = downcast_to_mxfp(
            y[i].cuda(), torch.uint8, axis=1)
        y_scale.append(ppu_cutlass_mxfp4_scales_swizzle(scale=y_fp4[1][i]))
    y_fp4_scale = torch.stack(y_scale, dim=0)

    return (x_fp4[0], x_fp4_scale), (y_fp4[0], y_fp4_scale), bias, out, ref_out

def construct_fp4_masked(num_groups: int, max_m: int, expected_m_per_group: int, k: int, n: int, distribution: str,
                             enable_sbo_overlap: bool = False, with_bias: bool = True):
    tensor_device = 'cuda'
    def find_next_power_of_2(m_list):
        if isinstance(m_list, torch.Tensor):
            max_val = m_list.amax().item()
        elif isinstance(m_list, list):
            max_val = max(m_list)
        else:
            raise ValueError("Invalid m_list")
        if max_val <= 0:
            return 1
        bit = (int(max_val) - 1).bit_length()
        return 1 << bit

    list_m = construct_uniform_m_list(distribution, num_groups, max_m, is_mask=True, em=expected_m_per_group)
    masked_m = torch.tensor(list_m, device=tensor_device, dtype=torch.int)

    max_m = max(max_m, find_next_power_of_2(list_m))
    assert masked_m.amax().item() <= max_m, f"max masked_m={masked_m.amax().item()}, allowed max_m={max_m}"

    x = torch.randn((num_groups, max_m, k), device=tensor_device, dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device=tensor_device, dtype=torch.bfloat16)
    out = torch.empty((num_groups, max_m, n), device=tensor_device, dtype=torch.bfloat16)
    bias = torch.randn((num_groups, n), device='cuda', dtype=torch.float) if with_bias else None

    x_fp4 = []
    x_fp4_scale = []
    y_fp4 = []
    y_fp4_scale = []
    x_ref = []
    y_ref = []
    for i in range(num_groups):
        a, a_scale = downcast_to_mxfp(x[i].cuda(), torch.uint8, axis=1)
        b, b_scale = downcast_to_mxfp(y[i].cuda(), torch.uint8, axis=1)
        a_dequant = upcast_from_mxfp(a, a_scale, torch.bfloat16, axis=-1)
        b_dequant = upcast_from_mxfp(b, b_scale, torch.bfloat16, axis=-1)
        x_fp4.append(a)
        x_fp4_scale.append(uint8_padding(a_scale))
        y_fp4.append(b)
        y_fp4_scale.append(ppu_cutlass_mxfp4_scales_swizzle(b_scale))
        x_ref.append(a_dequant)
        y_ref.append(b_dequant)

    x_fp4 = torch.stack(x_fp4, dim=0)
    x_fp4_scale = torch.stack(x_fp4_scale, dim=0)
    y_fp4 = torch.stack(y_fp4, dim=0)
    y_fp4_scale = torch.stack(y_fp4_scale, dim=0)
    x_ref = torch.stack(x_ref, dim=0)
    y_ref = torch.stack(y_ref, dim=0)

    ref_out = torch.einsum('gmk,gnk->gmn', x_ref, y_ref)
    ref_out = ref_out + bias.unsqueeze(1) if bias is not None else ref_out

    max_signal_size = num_groups * ceil_div(max_m, 64)
    signal = torch.zeros(max_signal_size, dtype=torch.int32, device=tensor_device) if enable_sbo_overlap else torch.empty(0).int()

    return (x_fp4.to('cuda'), x_fp4_scale.to('cuda')), (y_fp4.to('cuda'), y_fp4_scale.to('cuda')), bias, masked_m.to('cuda'), out.to('cuda'), ref_out.to('cuda').to(torch.bfloat16), signal.to('cuda'), max_m

def get_search_space(d: torch.dtype, gemm_type: str) -> list:
    """
    Returns search space according input gemm type

    Arguments:
        gemm_type: GroupedMasked, GroupedNoPad, DenseGemm

    Returns:
        The tile list:{block_m, block_n, warp_m, warp_n, stage}
    """
    assert gemm_type in ('GroupedMasked', 'GroupedNoPad', 'DenseGemm')

    block_k = 64 if d == torch.bfloat16 else 128
    tile_list = [
        # blockM = 16
        [16, 64, 16, 16, block_k, 2],
        [16, 64, 16, 16, block_k, 3],
        [16, 64, 16, 16, block_k, 4],
        [16, 64, 16, 16, block_k, 5],
        [16, 64, 16, 16, block_k * 2, 2],
        [16, 64, 16, 16, block_k * 2, 3],
        [16, 64, 16, 16, block_k * 2, 4],
        [16, 64, 16, 16, block_k * 2, 5],

        [16, 128, 16, 32, block_k, 2],
        [16, 128, 16, 32, block_k, 3],
        [16, 128, 16, 32, block_k, 4],
        [16, 128, 16, 32, block_k * 2, 2],
        [16, 128, 16, 32, block_k * 2, 3],
        [16, 128, 16, 32, block_k * 2, 4],
        [16, 128, 16, 16, block_k, 2],
        [16, 128, 16, 16, block_k, 3],
        [16, 128, 16, 16, block_k, 4],
        [16, 128, 16, 16, block_k * 2, 2],
        [16, 128, 16, 16, block_k * 2, 3],
        [16, 128, 16, 16, block_k * 2, 4],

        [16, 256, 16, 64, block_k, 2],
        [16, 256, 16, 64, block_k, 3],
        [16, 256, 16, 64, block_k * 2, 2],
        [16, 256, 16, 64, block_k * 2, 3],

        # blockM = 32
        [32, 64, 16, 32, block_k, 2],
        [32, 64, 16, 32, block_k, 3],
        [32, 64, 16, 32, block_k, 4],
        [32, 64, 16, 32, block_k, 5],
        [32, 64, 16, 32, block_k * 2, 2],
        [32, 64, 16, 32, block_k * 2, 3],
        [32, 64, 16, 32, block_k * 2, 4],
        [32, 64, 16, 16, block_k, 2],
        [32, 64, 16, 16, block_k, 3],
        [32, 64, 16, 16, block_k * 2, 2],
        [32, 64, 16, 16, block_k * 2, 3],

        [32, 128, 16, 64, block_k, 2],
        [32, 128, 16, 64, block_k, 3],
        [32, 128, 16, 64, block_k, 4],
        [32, 128, 16, 64, block_k * 2, 2],
        [32, 128, 16, 64, block_k * 2, 3],
        [32, 128, 16, 32, block_k, 2],
        [32, 128, 16, 32, block_k, 3],
        [32, 128, 16, 32, block_k * 2, 2],
        [32, 128, 16, 32, block_k * 2, 3],

        [32, 256, 16, 64, block_k, 2],
        [32, 256, 16, 64, block_k, 3],
        [32, 256, 16, 64, block_k * 2, 2],
        [32, 256, 16, 64, block_k * 2, 3],

        # blockM = 64
        [64, 64, 16, 16, block_k, 2],
        [64, 64, 16, 16, block_k, 3],
        [64, 64, 16, 16, block_k, 4],
        [64, 64, 16, 16, block_k * 2, 2],
        [64, 64, 16, 16, block_k * 2, 3],
        [64, 64, 16, 16, block_k * 2, 4],
        [64, 64, 16, 16, int(block_k / 2), 2],
        [64, 64, 16, 16, int(block_k / 2), 3],
        [64, 64, 16, 16, int(block_k / 2), 4],

        [64, 64, 32, 32, block_k, 2],
        [64, 64, 32, 32, block_k, 3],
        [64, 64, 32, 32, block_k, 4],
        [64, 64, 32, 32, block_k * 2, 2],
        [64, 64, 32, 32, block_k * 2, 3],
        [64, 64, 32, 32, block_k * 2, 4],
        [64, 64, 32, 32, int(block_k / 2), 2],
        [64, 64, 32, 32, int(block_k / 2), 3],
        [64, 64, 32, 32, int(block_k / 2), 4],

        [64, 128, 32, 64, block_k, 2],
        [64, 128, 32, 64, block_k, 3],
        [64, 128, 32, 32, block_k, 2],
        [64, 128, 32, 32, block_k, 3],
        [64, 128, 32, 64, block_k * 2, 2],
        [64, 128, 32, 64, block_k * 2, 3],
        [64, 128, 32, 32, block_k * 2, 2],
        [64, 128, 32, 32, block_k * 2, 3],
        [64, 128, 32, 64, int(block_k / 2), 2],
        [64, 128, 32, 64, int(block_k / 2), 3],
        [64, 128, 32, 32, int(block_k / 2), 2],
        [64, 128, 32, 32, int(block_k / 2), 3],

        [64, 256, 32, 64, block_k, 2],
        [64, 256, 32, 64, block_k, 3],
        [64, 256, 32, 64, block_k * 2, 2],
        [64, 256, 32, 64, block_k * 2, 3],
        [64, 256, 32, 64, int(block_k / 2), 2],
        [64, 256, 32, 64, int(block_k / 2), 3],
        [64, 256, 32, 64, int(block_k / 2), 2],
        [64, 256, 32, 64, int(block_k / 2), 4],

        # blockM = 128
        [128, 128, 64, 64, block_k, 2],
        [128, 128, 64, 64, block_k, 3],
        [128, 128, 64, 64, block_k, 4],
        [128, 128, 64, 64, block_k * 2, 2],
        [128, 128, 64, 64, block_k * 2, 3],
        [128, 128, 64, 64, block_k * 2, 4],
        [128, 128, 64, 64, int(block_k / 2), 2],
        [128, 128, 64, 64, int(block_k / 2), 3],
        [128, 128, 64, 64, int(block_k / 2), 4],
        [128, 256, 64, 64, block_k, 2],
        [128, 256, 64, 64, block_k, 3],
        [128, 256, 64, 64, block_k, 4],
        [128, 256, 64, 64, block_k * 2, 2],
        [128, 256, 64, 64, block_k * 2, 3],
        [128, 256, 64, 64, block_k * 2, 4],
        [128, 256, 64, 64, int(block_k / 2), 2],
        [128, 256, 64, 64, int(block_k / 2), 3],
        [128, 256, 64, 64, int(block_k / 2), 4],

        # blockM = 256
        [256, 64, 32, 64, block_k,      2],
        [256, 64, 32, 64, block_k,      3],
        [256, 64, 32, 64, block_k * 2,  2],
        [256, 64, 32, 64, block_k * 2,  3],
        [256, 128, 64, 64, block_k, 2],
        [256, 128, 64, 64, block_k, 3],
        [256, 128, 64, 64, block_k, 4],
        [256, 128, 64, 64, block_k * 2, 2],
        [256, 128, 64, 64, block_k * 2, 3],
        [256, 128, 64, 64, block_k * 2, 4],
        [256, 128, 64, 64, int(block_k / 2), 2],
        [256, 128, 64, 64, int(block_k / 2), 3],
        [256, 128, 64, 64, int(block_k / 2), 4],
        [256, 256, 64, 64, block_k,     2],
        [256, 256, 64, 64, block_k,     3],
        [256, 256, 64, 64, block_k,     4],
        [256, 256, 64, 64, block_k * 2, 2],
        [256, 256, 64, 64, block_k * 2, 3],
        [256, 256, 64, 64, block_k * 2, 4],
        [256, 256, 64, 64, int(block_k / 2), 2],
        [256, 256, 64, 64, int(block_k / 2), 3],
        [256, 256, 64, 64, int(block_k / 2), 4],
    ]

    if 'DenseGemm' in gemm_type:
        tile_list.extend([
            # blockM = 128
            [128, 128, 64, 64, block_k, 2],
            [128, 128, 64, 64, block_k, 3],
            [128, 128, 64, 64, block_k, 4],
            [128, 128, 64, 64, block_k * 2, 3],
            [128, 256, 64, 64, block_k, 2],
            [128, 256, 64, 64, block_k, 3],

            # blockM = 256
            [256, 64, 32, 64, block_k,      2],
            [256, 64, 32, 64, block_k,      3],
            [256, 64, 32, 64, block_k * 2,  2],
            [256, 64, 32, 64, block_k * 2,  3],
            [256, 128, 64, 64, block_k, 2],
            [256, 128, 64, 64, block_k, 3],
            [256, 128, 64, 64, block_k, 4],
            [256, 256, 64, 64, block_k,     4],

            # blockK = 64B
            [128, 128, 64, 64, int(block_k / 2), 2],
            [128, 256, 64, 64, int(block_k / 2), 2],
            [256, 128, 64, 64, int(block_k / 2), 2],
        ])

    tile_list_rtn = set()

    if d == torch.uint8:
        for tile in tile_list:
            if tile[2] % 16 == 0 and tile[3] % 16 == 0 and tile[0] % 16 == 0 and tile[1] % 16 == 0 and tile[4] % 32 == 0:
                tile_list_rtn.add(tuple(tile))
                if tile[4] == block_k:
                    tile_copy = copy.deepcopy(tile)
                    tile_copy[4] = int(block_k / 2)
                    if tile[4] != 0 and tile_copy[4] % 32 == 0:
                        tile_list_rtn.add(tuple(tile_copy))

    return tile_list_rtn

def get_full_search_space(k) -> list:

    block_ms = [256, 128, 64, 32, 16]
    block_ns = [256, 128, 64, 32, 16]
    block_ks = [256, 128, 64]

    base_warp_sizes = [16, 32, 64, 128, 256]

    configs = []

    for BLOCK_M in block_ms:
        for BLOCK_N in block_ns:
            for BLOCK_K in block_ks:
                valid_warp_ms = [w for w in base_warp_sizes if w <= BLOCK_M]
                valid_warp_ns = [w for w in base_warp_sizes if w <= BLOCK_N]
                for WARP_M in valid_warp_ms:
                    for WARP_N in valid_warp_ns:
                        stage_candidates = tuple(filter(lambda s: s <= k // BLOCK_K, (8, 7, 6, 5, 4, 3, 2)))
                            ### for those cases with super small k.
                        if not stage_candidates: stage_candidates = (2, )
                        for num_stages in stage_candidates:
                            best_smem_config = get_smem_config_fp4(num_stages, BLOCK_M, BLOCK_N, WARP_M, WARP_N, BLOCK_K)
                            ppu_capacity = 262144
                            if best_smem_config[0] <= ppu_capacity:
                                configs.append([BLOCK_M, BLOCK_N, WARP_M, WARP_N, BLOCK_K, num_stages])
    return configs

def save_lut(lut: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(lut, f, indent=2)
    print(f"[AutoTune] LUT saved to {filepath}")


def load_lut(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r") as f:
        lut = json.load(f)
    print(f"[AutoTune] LUT loaded from {filepath}, {len(lut)} entries")
    return lut

def precompile_kernels(search_space, M, N, K, num_experts, topk_experts, gemm_type, verbose=True):
    """Precompile all kernels in the search space to avoid JIT compilation during benchmarking."""
    if verbose:
        print("Precompiling kernels...")

    def calc_diff(x, y):
        x, y = x.double(), y.double()
        denominator = (x * x + y * y).sum()
        sim = 2 * (x * y).sum() / denominator
        return 1 - sim

    def compile_and_validate(config, gemm_type):
        num_sms = get_num_sms()
        smem_config = get_smem_config_fp4(2, 256, 256, 64, 64, 128)
        ref_config = (num_sms, 256, 256, 128, 64, 64, 2, smem_config)
        block_m, block_n, block_k, warp_m, warp_n, num_stages = config
        smem_config = get_smem_config_fp4(num_stages, block_m, block_n, warp_m, warp_n, block_k)
        config = (num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config)
        if gemm_type == "GroupedNoPad":
            m_rows = construct_uniform_m_list("uniform", num_experts, M*topk_experts)
            m_rows = torch.tensor(m_rows).to('cuda').to(torch.int32)

            cuda_m_rows = m_rows.to('cuda')
            x_fp4, y_fp4, bias, out, ref_out = construct_fp4_nopad(
                N, K, cuda_m_rows, num_experts)

            try:
                deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad(
                    x_fp4, y_fp4, bias, out, m_rows=cuda_m_rows, configs=config)
                deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad(
                    x_fp4, y_fp4, bias, ref_out, m_rows=cuda_m_rows, configs=ref_config)
                torch.cuda.synchronize()

                diff = calc_diff(out, ref_out)
                if diff >= 0.001:
                    if verbose:
                        print(
                            f"  [FAIL] {config_to_str(config)}: diff={diff:.5f} ≥ 0.001")
                    return None  # exclude this config

                if verbose:
                    print(f"  [OK] {config_to_str(config)}")
                return config
            except Exception as e:
                if verbose:
                    print(f"  [SKIP] {config_to_str(config)}: {e}")
                return None
        elif gemm_type == "GroupedMasked":
            expected_m_per_group = ceil_div(M*topk_experts, num_experts)
            x, y, bias, masked_m, out, ref_out, _, _ = construct_fp4_masked(num_experts, M, expected_m_per_group, K, N, "uniform", False, with_bias=False)

            try:
                deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(x, y, bias, out, masked_m, expected_m_per_group, configs=config)
                torch.cuda.synchronize()

                for j in range(num_experts):
                    diff = calc_diff(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()])
                    if (masked_m[j] != 0):
                        if diff >= 0.001:
                            if verbose:
                                print(
                                    f"  [FAIL] {config_to_str(config)}: diff={diff:.5f} ≥ 0.001")
                            return None  # exclude this config

                if verbose:
                    print(f"  [OK] {config_to_str(config)}")
                return config
            except Exception as e:
                if verbose:
                    print(f"  [SKIP] {config_to_str(config)}: {e}")
                return None
        elif gemm_type == "DenseGemm":
            x_fp4, y_fp4, bias, out, ref_out = construct_fp4_dense(
                M, N, K)
            try:
                deep_gemm.gemm_fp4_fp4_bf16_nt(
                    x_fp4, y_fp4, bias, out, config)
                deep_gemm.gemm_fp4_fp4_bf16_nt(
                    x_fp4, y_fp4, bias, ref_out, ref_config)
                torch.cuda.synchronize()

                diff = calc_diff(out, ref_out)
                if diff >= 0.001:
                    if verbose:
                        print(
                            f"  [FAIL] {config_to_str(config)}: diff={diff:.5f} ≥ 0.001")
                    return None  # exclude this config

                if verbose:
                    print(f"  [OK] {config_to_str(config)}")
                return config
            except Exception as e:
                if verbose:
                    print(f"  [SKIP] {config_to_str(config)}: {e}")
                return None

    valid_configs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(compile_and_validate, config, gemm_type)
                   for config in search_space]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                valid_configs.append(result)

    if verbose:
        print(
            f"Precompile done: {len(valid_configs)}/{len(search_space)} configs passed.")

    return valid_configs

def benchmark_kernel(
    config: List[int],
    M: int,
    N: int,
    K: int,
    num_experts: int,
    gemm_type: str,
    m_rows: torch.Tensor = None,
    topk_experts: int = 10,
    warmup: int = 5,
    repeat: int = 20,
) -> float:

    if gemm_type == "GroupedNoPad":
        total_tokens = int(m_rows.sum().item())
        x_fp4, y_fp4, bias, out, _ = construct_fp4_nopad(N, K, m_rows, num_experts)
        m_rows = m_rows.to('cuda')

        def run():
            deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad(
                x_fp4, y_fp4, bias, out, m_rows=m_rows, configs=config)

        try:
            for i in range(warmup):
                run()
            torch.cuda.synchronize()
        except Exception as e:
            raise RuntimeError(
                f"Warmup failed: config={config}, total_tokens={total_tokens}, "
                f"N={N}, K={K}, m_rows_sum={m_rows.sum().item()}, "
                f"m_rows_min={m_rows.min().item()}, m_rows_max={m_rows.max().item()}"
            ) from e

        start_events = [torch.cuda.Event(enable_timing=True)
                        for _ in range(repeat)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

        try:
            for i in range(repeat):
                start_events[i].record()
                run()
                end_events[i].record()
            torch.cuda.synchronize()
        except Exception as e:
            raise RuntimeError(
                f"Benchmark failed: config={config}, total_tokens={total_tokens}"
            ) from e

        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        return float(np.median(times))
    elif gemm_type == "GroupedMasked":
        expected_m_per_group = ceil_div(M*topk_experts, num_experts)
        x, y, bias, masked_m, out, _, _, _ = construct_fp4_masked(num_experts, M, expected_m_per_group, K, N, "uniform", False, with_bias=False)

        def run():
            deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(x, y, bias, out, masked_m, expected_m_per_group, configs=config)

        try:
            for i in range(warmup):
                run()
            torch.cuda.synchronize()
        except Exception as e:
            raise RuntimeError(
                f"Warmup failed: config={config}, M={M}, N={N}, K={K}"
            ) from e

        start_events = [torch.cuda.Event(enable_timing=True)
                        for _ in range(repeat)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

        try:
            for i in range(repeat):
                start_events[i].record()
                run()
                end_events[i].record()
            torch.cuda.synchronize()
        except Exception as e:
            raise RuntimeError(
                f"Benchmark failed: config={config}, M={M}, N={N}, K={K}"
            ) from e

        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        return float(np.median(times))
    elif gemm_type == "DenseGemm":
        x_fp4, y_fp4, bias, out, _ = construct_fp4_dense(M, N, K)

        def run():
            deep_gemm.gemm_fp4_fp4_bf16_nt(
                x_fp4, y_fp4, bias, out, configs=config)

        try:
            for i in range(warmup):
                run()
            torch.cuda.synchronize()
        except Exception as e:
            raise RuntimeError(
                f"Warmup failed: config={config}, M={M}, N={N}, K={K}, m_rows_sum={m_rows.sum().item()}"
            ) from e

        start_events = [torch.cuda.Event(enable_timing=True)
                        for _ in range(repeat)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

        try:
            for i in range(repeat):
                start_events[i].record()
                run()
                end_events[i].record()
            torch.cuda.synchronize()
        except Exception as e:
            raise RuntimeError(
                f"Benchmark failed: config={config}, M={M}, N={N}, K={K}"
            ) from e

        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        return float(np.median(times))

def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y

def construct_uniform_m_list(distribution, num_groups = int, m = int, is_mask=False, seed=0, em=0):
    import random

    group_m_list = list()
    if is_mask and em != 0:
        expected_m_per_group = em
    else:
        expected_m_per_group = ceil_div(m, num_groups)
    if distribution == None: # default value
        distribution = "uniform"
    if type(distribution) is list:
        return distribution
    elif distribution == "uniform":
        random.seed(seed)
        group_m_list = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
    else:
        print("ERROR: Unsupported distribution type, please check!")
        exit(1)
    if not is_mask:
        group_m_list = round_m_list_sum_to_m(group_m_list, m)
        assert sum(group_m_list) <= m, f"sum(m_list)={sum(group_m_list)} must <= m_sum={m}"
    group_m_list = sorted(group_m_list, reverse=True)
    return group_m_list

def round_m_list_sum_to_m(m_list, m_sum):
    current_sum = sum(m_list)
    step = int(abs(current_sum - m_sum) / len(m_list))
    _res = abs(current_sum - m_sum) % len(m_list)
    if current_sum > m_sum:
        m_list[:] = [x - step if x > step else x for x in m_list]
        idx, idy = 0, 0
        _res = abs(sum(m_list) - m_sum)
        while idx < _res:
            if m_list[idy] >= 1:
                m_list[idy] -= 1
                idx += 1
            idy = (idy+1) % len(m_list)
    elif current_sum < m_sum:
        m_list[:] = [x + step for x in m_list]
        m_list[:_res] = [x + 1 for x in m_list[:_res]]
    return m_list

def autotune_single(
    M: int,
    N: int,
    K: int,
    topk_experts: int,
    search_space: List[List[int]],
    num_experts: int,
    gemm_type: str,
    warmup: int = 10,
    repeat: int = 100,
    verbose: bool = True,
) -> Tuple[Optional[List[int]], float, List[dict]]:
    results = []
    if gemm_type == "GroupedNoPad":
        m_rows = construct_uniform_m_list("uniform", num_experts, M*topk_experts)
        m_rows = torch.tensor(m_rows).to('cuda').to(torch.int32)

        best_time = float("inf")
        best_config = None

        for idx, config in enumerate(search_space):
            times_across_samples = []

            try:
                t = benchmark_kernel(
                    config=config,
                    M=M, N=N, K=K,
                    m_rows=m_rows,
                    gemm_type=gemm_type,
                    num_experts=num_experts,
                    warmup=warmup,
                    repeat=repeat,
                )
                times_across_samples.append(t)
            except Exception as e:
                if verbose:
                    print(
                        f"    [SKIP] {config_to_str(config)} failed: {e}")
                    traceback.print_exc()
                times_across_samples.append(float("inf"))
                break

            avg_time = float(np.mean(times_across_samples))
            results.append({"config": config, "avg_time_ms": avg_time})

            if verbose:
                print(
                    f"    [{idx + 1}/{len(search_space)}] Testing: {config_to_str(config)} @ {avg_time:.4f} ms")

            if avg_time < best_time:
                best_time = avg_time
                best_config = config

            if verbose and (idx + 1) % 20 == 0:
                if best_config is not None:
                    print(f"    [{idx + 1}/{len(search_space)}] best so far: "
                        f"{config_to_str(best_config)} @ {best_time:.4f} ms")
                else:
                    print(
                        f"    [{idx + 1}/{len(search_space)}] no valid config yet")
    elif gemm_type == "GroupedMasked":
        best_time = float("inf")
        best_config = None

        for idx, config in enumerate(search_space):
            times_across_samples = []

            try:
                t = benchmark_kernel(
                    config=config,
                    M=M, N=N, K=K,
                    gemm_type=gemm_type,
                    num_experts=num_experts,
                    topk_experts=topk_experts,
                    warmup=warmup,
                    repeat=repeat,
                )
                times_across_samples.append(t)
            except Exception as e:
                if verbose:
                    print(
                        f"    [SKIP] {config_to_str(config)} failed: {e}")
                    traceback.print_exc()
                times_across_samples.append(float("inf"))
                break

            avg_time = float(np.mean(times_across_samples))
            results.append({"config": config, "avg_time_ms": avg_time})

            if verbose:
                print(
                    f"    [{idx + 1}/{len(search_space)}] Testing: {config_to_str(config)} @ {avg_time:.4f} ms")

            if avg_time < best_time:
                best_time = avg_time
                best_config = config

            if verbose and (idx + 1) % 20 == 0:
                if best_config is not None:
                    print(f"    [{idx + 1}/{len(search_space)}] best so far: "
                        f"{config_to_str(best_config)} @ {best_time:.4f} ms")
                else:
                    print(
                        f"    [{idx + 1}/{len(search_space)}] no valid config yet")
    elif gemm_type == "DenseGemm":
        best_time = float("inf")
        best_config = None

        for idx, config in enumerate(search_space):
            times_across_samples = []

            try:
                t = benchmark_kernel(
                    config=config,
                    M=M, N=N, K=K,
                    num_experts=1,
                    warmup=warmup,
                    repeat=repeat,
                    gemm_type=gemm_type,
                )
                times_across_samples.append(t)
            except Exception as e:
                if verbose:
                    print(
                        f"    [SKIP] {config_to_str(config)} failed: {e}")
                    traceback.print_exc()
                times_across_samples.append(float("inf"))
                break

            avg_time = float(np.mean(times_across_samples))
            results.append({"config": config, "avg_time_ms": avg_time})

            if verbose:
                print(
                    f"    [{idx + 1}/{len(search_space)}] Testing: {config_to_str(config)} @ {avg_time:.4f} ms")

            if avg_time < best_time:
                best_time = avg_time
                best_config = config

            if verbose and (idx + 1) % 20 == 0:
                if best_config is not None:
                    print(f"    [{idx + 1}/{len(search_space)}] best so far: "
                        f"{config_to_str(best_config)} @ {best_time:.4f} ms")
                else:
                    print(
                        f"    [{idx + 1}/{len(search_space)}] no valid config yet")

    if verbose:
        sorted_results = sorted(results, key=lambda x: x["avg_time_ms"])
        print(f"\n  {'='*60}")
        print(f"  Autotune results for M={M}, N={N}, K={K}, G={topk_experts}")
        print(f"  Total configs tested: {len(results)}")
        valid_count = sum(
            1 for r in results if r["avg_time_ms"] < float("inf"))
        failed_count = len(results) - valid_count
        print(f"  Valid: {valid_count}, Failed: {failed_count}")
        print(f"  {'─'*60}")
        print(f"  {'Rank':<6} {'Time(ms)':<12} {'Config'}")
        print(f"  {'─'*60}")

        for rank, r in enumerate(sorted_results[:10], 1):
            t = r["avg_time_ms"]
            t_str = f"{t:.4f}" if t < float("inf") else "FAILED"
            print(f"  {rank:<6} {t_str:<12} {config_to_str(r['config'])}")
        if len(sorted_results) > 10:
            print(f"  ... ({len(sorted_results) - 10} more configs omitted)")

        valid_results = [
            r for r in sorted_results if r["avg_time_ms"] < float("inf")]
        if len(valid_results) > 10:
            print(f"  {'─'*60}")
            print(f"  Worst 3 valid configs:")
            for r in valid_results[-3:]:
                print(
                    f"         {r['avg_time_ms']:.4f} ms   {config_to_str(r['config'])}")
        print(f"  {'─'*60}")
        if best_config is not None:
            print(
                f"  ★ Best: {config_to_str(best_config)} @ {best_time:.4f} ms")
        print(f"  {'='*60}\n")

    return best_config, best_time, results


def autotune_all(
    dense_nk_pairs: Optional[List[Tuple[int, int]]] = None,
    hidden_size: Optional[int] = None,
    moe_intermediate_size: Optional[int] = None,
    topk_experts: int = 1,
    num_experts: int = 1,  # 🔑 默认为 1，DenseGemm 固定用 1
    tp_size: Optional[int] = None,
    ep_size: Optional[int] = None,
    save_path: str = "",
    gemm_type: str = "GroupedNoPad",
    warmup: int = 10,
    repeat: int = 100,
    verbose: bool = True,
):
    cases = get_all_cases(
        dense_nk_pairs=dense_nk_pairs,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        tp_size=tp_size,
        gemm_type=gemm_type,
        topk_experts=topk_experts
    )
    print(f"[AutoTune] Total cases: {len(cases)}")
    print(f"[AutoTune] CANDIDATE_Ms: {len(CANDIDATE_Ms)} values, "
          f"range [{CANDIDATE_Ms[0]}, {CANDIDATE_Ms[-1]}]")

    if gemm_type == "DenseGemm":
        print(f"[AutoTune] Using DenseGemm with N,K pairs: {dense_nk_pairs}")

    lut = load_lut(save_path)

    grouped_cases: Dict[str, List[Dict]] = {}
    for case in cases:
        if gemm_type == "GroupedMasked":
            group_key = f"N{case['N']}_K{case['K']}_E{num_experts}_Masked"
        else:
            group_key = f"N{case['N']}_K{case['K']}_E{num_experts}"
        grouped_cases.setdefault(group_key, []).append(case)

    for group_key, case_list in grouped_cases.items():
        N = case_list[0]["N"]
        K = case_list[0]["K"]
        topk_experts = case_list[0]["topk_experts"]

        search_space = get_search_space(torch.uint8, gemm_type=gemm_type)
        config_list = []
        for tile in search_space:
            block_m, block_n, warp_m, warp_n, block_k, num_stages = tile
            config_list.append((block_m, block_n, block_k,
                            warp_m, warp_n, num_stages))

        print(f"\n{'='*70}")
        print(f"[AutoTune] Processing {group_key}: {len(case_list)} M values")
        print(f"  gemm_type={gemm_type}, N={N}, K={K}, topk_experts={topk_experts}")
        print(f"{'='*70}")

        valid_search_space = precompile_kernels(
            search_space=config_list.copy(),
            M=4, N=N, K=K,
            num_experts=num_experts,
            topk_experts=topk_experts,
            gemm_type=gemm_type,
            verbose=verbose,
        )

        if group_key not in lut:
            if gemm_type == "DenseGemm":
                lut[group_key] = {
                    "meta": {
                        "N": N,
                        "K": K,
                        "num_experts": 1,
                    },
                    "configs": {},
                }
            elif gemm_type == "GroupedNoPad" or gemm_type == "GroupedMasked":
                lut[group_key] = {
                    "meta": {
                        "N": N,
                        "K": K,
                        "num_experts": num_experts,
                        "topk_experts": topk_experts,
                        "tp_size": tp_size,
                        "ep_size": ep_size,
                    },
                    "configs": {},
                }

        for case_idx, case in enumerate(case_list):
            M = case["M"]
            if gemm_type == "GroupedMasked":
                m_key = str(M)
            else:
                total_tokens = M * topk_experts
                m_key = str(total_tokens)

            if m_key in lut[group_key]["configs"]:
                if verbose:
                    print(f"  [SKIP] M={m_key} already in LUT, "
                          f"config={lut[group_key]['configs'][m_key]}")
                continue

            print(f"\n  [{case_idx + 1}/{len(case_list)}] "
                  f"Tuning M={m_key}, N={N}, K={K}")

            best_config, best_time, _ = autotune_single(
                M=M, N=N, K=K,
                topk_experts=topk_experts,
                search_space=valid_search_space,
                num_experts=num_experts,
                warmup=warmup,
                repeat=repeat,
                gemm_type=gemm_type,
                verbose=verbose,
            )

            print(
                f"  Result: {config_to_str(best_config)} @ {best_time:.4f} ms ")

            lut[group_key]["configs"][m_key] = best_config

            save_lut(lut, save_path)

    print(f"\n[AutoTune] All done! LUT saved to {save_path}")


_LUT_CACHE = {}
_LUT_CACHE_DIR = None


@lru_cache(maxsize=1)
def _load_lut_from_folder(lut_dir: str) -> dict:
    merged_lut: Dict[str, Any] = {}

    if not os.path.isdir(lut_dir):
        print(f"[AutoTune] LUT directory not found: {lut_dir}")
        return merged_lut

    for fname in os.listdir(lut_dir):
        if fname.startswith("FP4_") and fname.endswith(".json"):
            fpath = os.path.join(lut_dir, fname)
            try:
                with open(fpath, "r") as f:
                    lut = json.load(f)
                for group_key, data in lut.items():
                    if group_key in merged_lut:
                        existing = merged_lut[group_key].setdefault(
                            "configs", {})
                        existing.update(data.get("configs", {}))
                    else:
                        merged_lut[group_key] = data
            except Exception as e:
                print(f"[AutoTune] Warning: failed to load {fpath}: {e}")

    return merged_lut


def lookup_best_config(
    M: int,
    N: int,
    K: int,
    num_experts: int,
    is_grouped_masked: bool = False,
) -> Optional[List[int]]:
    lut_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
    )
    if not os.path.exists(lut_dir):
        return None

    global _LUT_CACHE_DIR
    if _LUT_CACHE_DIR != lut_dir:
        _LUT_CACHE.clear()
        _LUT_CACHE_DIR = lut_dir

    lut = _load_lut_from_folder(lut_dir)
    if is_grouped_masked:
        group_key = f"N{N}_K{K}_E{num_experts}_Masked"
    else:
        group_key = f"N{N}_K{K}_E{num_experts}"
    if group_key not in lut:
        print(f"[AutoTune] No entry for {group_key} in LUT folder")
        return None

    configs = lut[group_key].get("configs", {})
    m_key = str(M)

    if m_key in configs:
        print(f"[AutoTune] lookup the lut configs: {configs[m_key]}")
        return configs[m_key]

    available_ms = sorted([int(k) for k in configs.keys()])
    if not available_ms:
        return None

    closest_m = min(available_ms, key=lambda x: abs(x - M))
    print(f"[AutoTune] Exact M={M} not found, using closest M={closest_m}")
    return configs[str(closest_m)]

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Autotune Grouped GEMM Kernel")

    # MoE-related args (optional for DenseGemm)
    parser.add_argument("--hidden_size", type=int, default=None,
                        help="Model hidden size (e.g. 7168); required for GroupedNoPad")
    parser.add_argument("--moe_intermediate_size", type=int, default=None,
                        help="MoE intermediate size (e.g. 18432); required for GroupedNoPad")
    parser.add_argument("--tp_sizes", type=lambda s: [int(x) for x in s.split(',')], default=1,
                        help="Comma-separated TP sizes (e.g., '4,8,16'); required for GroupedNoPad")
    parser.add_argument("--ep_sizes", type=lambda s: [int(x) for x in s.split(',')], default=1,
                        help="Comma-separated EP sizes (e.g., '16,32,64'); optional")
    parser.add_argument("--num_experts", type=int, default=1,
                        help="Number of experts (e.g. 512); used only for GroupedNoPad")
    parser.add_argument("--topk_experts", type=int, default=10,
                        help="Top-k experts per token (e.g. 10); used only for GroupedNoPad")

    # DenseGemm-specific args
    parser.add_argument("--N", type=lambda s: [int(x) for x in s.split(',')], default=None,
                        help="N dimensions (comma-separated, e.g. '4096,2048'); required for DenseGemm")
    parser.add_argument("--K", type=lambda s: [int(x) for x in s.split(',')], default=None,
                        help="K dimensions (comma-separated, e.g. '7168,4096'); required for DenseGemm")

    # Shared args
    parser.add_argument("--gemm_type", type=str, choices=["DenseGemm", "GroupedNoPad", "GroupedMasked"], required=True,
                        help='choose the gemm_type')
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save the unified LUT file (e.g. configs/grouped_gemm_lut.json)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args()

    tp_sizes = [args.tp_sizes] if isinstance(args.tp_sizes, int) else args.tp_sizes
    ep_sizes = [args.ep_sizes] if isinstance(args.ep_sizes, int) else args.ep_sizes

    if args.gemm_type == "GroupedNoPad" or args.gemm_type == "GroupedMasked":
        assert args.hidden_size is not None, "GroupedNoPad requires --hidden_size"
        assert args.moe_intermediate_size is not None, "GroupedNoPad requires --moe_intermediate_size"
        assert args.tp_sizes is not None, "GroupedNoPad requires --tp_sizes"
        assert args.ep_sizes is not None, "GroupedNoPad requires --ep_sizes"
        print(f"[AutoTune] Model config:")
        print(f"  hidden_size = {args.hidden_size}")
        print(f"  moe_intermediate_size = {args.moe_intermediate_size}")
        print(f"  topk_experts = {args.topk_experts}")
        print(f"  CANDIDATE_Ms = {len(CANDIDATE_Ms)} values")

        for tp_size in tp_sizes:
            for ep_size in ep_sizes:
                local_num_experts = int(args.num_experts / ep_size)
                autotune_all(
                    hidden_size=args.hidden_size,
                    moe_intermediate_size=args.moe_intermediate_size,
                    topk_experts=args.topk_experts,
                    num_experts=local_num_experts,
                    tp_size=tp_size,
                    ep_size=ep_size,
                    gemm_type=args.gemm_type,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    save_path=args.save_path,
                    verbose=args.verbose,
                )

    elif args.gemm_type == "DenseGemm":
        assert args.N is not None and args.K is not None, "DenseGemm requires --N and --K"
        assert len(args.N) == len(args.K), "Number of N and K values must match"
        dense_nk_pairs = list(zip(args.N, args.K))

        print(f"[AutoTune] DenseGemm config:")
        print(f"  N,K pairs = {dense_nk_pairs}")
        print(f"  CANDIDATE_Ms = {len(CANDIDATE_Ms)} values")

        autotune_all(
            dense_nk_pairs=dense_nk_pairs,
            gemm_type=args.gemm_type,
            save_path=args.save_path,
            warmup=args.warmup,
            repeat=args.repeat,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()