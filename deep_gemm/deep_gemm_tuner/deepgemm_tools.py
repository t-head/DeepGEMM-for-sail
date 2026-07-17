#!/usr/bin/env python3
"""
Deep GEMM Tools
A collection of utility functions for GEMM operations.
"""
import torch
import math
import os
import triton
import triton.language as tl
import numpy as np
from functools import lru_cache
from typing import Tuple, List, Optional, Any

from deep_gemm import calc_diff
from deep_gemm.jit_kernels import m_grouped_gemm_int8_int8_bf16_nt_masked, m_grouped_gemm_int8_int8_bf16_nt_nopad, gemm_int8_int8_bf16_nt, m_grouped_gemm_bf16_bf16_bf16_nt_nopad, get_num_sms
from deep_gemm.jit_kernels.gemm_int8 import get_smem_config
from .utils import get_deep_gemm_luts
DEEP_GEMM_AVAILABLE = True

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

from deep_gemm.jit_kernels.utils import get_search_space
def get_pre_assert_configs(m: int, n: int, k: int, num_groups: int, num_sms: int,
                          is_grouped_contiguous: bool = False, is_grouped_masked: bool = False, gemm_type: str = "dense", dtype=torch.int8) -> List[Tuple]:
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
    assert dtype in (torch.int8, torch.float8_e4m3fn, torch.bfloat16)
    assert_config = get_search_space(dtype, gemm_type)
    rst = []
    for config in assert_config:
        best_block_m, best_block_n, warp_m, warp_n,  block_k, best_num_stages = config
        # best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages = config['BLOCK_M'],  config['BLOCK_N'],  config['BLOCK_K'],  config['WARP_M'], config['WARP_N'], config['NUM_STAGES']
        smem_config = get_smem_config(best_num_stages, k, best_block_m, best_block_n, block_k, 1)
        rst.append((num_sms,  best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages, smem_config))

    return rst

@triton.jit
def _count_expert_num_tokens(
    topk_ids_ptr,
    expert_num_tokens_ptr,
    num_experts,
    topk_numel,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    curr_expert = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    topk_ids_ptrs = topk_ids_ptr + offsets

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for x in range(tl.cdiv(topk_numel, BLOCK_SIZE)):
        mask = offsets < (topk_numel - x * BLOCK_SIZE)
        expert_ids = tl.load(topk_ids_ptrs, mask=mask, other=-1)

        has_curr_expert = tl.where(expert_ids == curr_expert, 1, 0)
        acc = acc + has_curr_expert
        topk_ids_ptrs += BLOCK_SIZE

    if curr_expert < num_experts:
        tl.store(
            expert_num_tokens_ptr + curr_expert, round_up_triton(tl.sum(acc), BLOCK_E)
        )

def count_expert_num_tokens(
    topk_ids: torch.Tensor, num_local_experts: int, block_align: int
) -> torch.Tensor:
    """
    Count the number to tokens assigned to each expert.

    Parameters:
    - topk_ids (torch.Tensor): Tensor mapping each token to its
    list of experts.
    - num_local_experts (int): Number of experts in this rank.
    - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
    from the global expert space to the local expert space of the expert
    parallel shard.

    Returns:
    A tensor of size num_local_experts, where tensor[i] holds the number
    of tokens assigned to the ith expert.
    """
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"
    expert_num_tokens = torch.empty(
        (num_local_experts), device=topk_ids.device, dtype=torch.int32
    )

    grid = num_local_experts
    BLOCK_SIZE = min(topk_ids.numel(), 1024)
    BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE)

    _count_expert_num_tokens[(grid,)](
        topk_ids,
        expert_num_tokens,
        num_local_experts,
        topk_ids.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_E=block_align,
    )

    return expert_num_tokens

def fused_topk_torch_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor = None,
    scoring_func: str = "softmax",
):
    def scoring_func_impl(gating_output: torch.Tensor) -> torch.Tensor:
        if scoring_func == "softmax":
            return gating_output.softmax(dim=-1)
        elif scoring_func == "sigmoid":
            return gating_output.sigmoid()
        else:
            raise ValueError(f"Invalid scoring function: {scoring_func}")

    if correction_bias is not None:
        n_routed_experts = gating_output.shape[-1]
        scores = scoring_func_impl(gating_output)
        scores_for_choice = scores.view(
            -1, n_routed_experts
        ) + correction_bias.unsqueeze(0)
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
    else:
        assert (
            hidden_states.shape[0] == gating_output.shape[0]
        ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
        M, _ = hidden_states.shape
        topk_weights = torch.empty(
            M, topk, dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
        topk_weights = scoring_func_impl(gating_output.float())
        topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids

### Gamma Sample Interface ###
gamma_params =[
    {7: (1.45, 4.271)},
    {10: (1.284, 6.957)},
    {13: (1.711, 6.695)},
    {16: (1.283, 10.943)},
    {19: (1.853, 9.013)},
    {22: (1.595, 12.031)},
    {25: (1.788, 12.266)},
    {28: (1.542, 15.892)},
    {31: (1.404, 19.307)}
]

def gamma_sample(shape, scale, num_groups, num_samples):
    rate = 1.0 / scale
    concentration = torch.tensor([shape])
    rate_tensor = torch.tensor([rate])

    dist = torch.distributions.Gamma(concentration=concentration, rate=rate_tensor)
    samples = dist.sample((num_samples, num_groups)).to('cuda').int().squeeze()
    return samples

def grouped_masked_m_sample(expect_m, num_groups, num_samples=100):
    keys = []
    shapes = []
    scales = []

    for d in gamma_params:
        k = list(d.keys())[0]
        shape, scale = d[k]
        keys.append(k)
        shapes.append(shape)
        scales.append(scale)

    def interpolate_gamma_params(target_key: float) -> Optional[Tuple[float, float]]:
        """
        Interpolate to get (shape, scale) based on target_key

        Args:
            target_key (float): Input key (e.g. 15, 20, etc.)

        Returns:
            tuple: (interpolated_shape, interpolated_scale)
        """
        if target_key < min(keys) or target_key > max(keys):
            print(f"Warning: {target_key} is out of interpolation range [{min(keys)}, {max(keys)}], results may be inaccurate.")

        # Linear interpolation
        shape_interp = np.interp(target_key, keys, shapes)
        scale_interp = np.interp(target_key, keys, scales)

        return (shape_interp, scale_interp)

    shape, scale = interpolate_gamma_params(expect_m)
    return gamma_sample(shape, scale, num_groups, num_samples)

# sample from real dataset
def parse_and_figure(file_path, expect_m, total_cases):
    """Extract all values from 'masked_m' lists in the log file."""
    all_values = []
    flag=True
    flag_pattern = f"expected_m {expect_m}"
    pattern = r'masked_m\s*\[(.*?)\]'
    import re
    iter = 0
    with open(file_path, 'r') as file:
        for line in file:
            # Match the masked_m list pattern
            flag_match = re.search(flag_pattern, line)
            match = re.search(pattern, line)
            if match and flag_match:
                iter+=1
                if iter > 5000:
                    # Extract and convert values
                    values_str = match.group(1)
                    values = list(map(int, values_str.split(', ')))
                    all_values.append(values)

            if len(all_values) > total_cases:
                break
    return torch.tensor(all_values, dtype=torch.int32, device='cuda')

### End of Gamma Sample Interface ###

### Int8 Tools ###
def per_token_quant_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token quantization function for int8"""
    x = x.to(torch.float32)
    scale = x.amax(dim=-1, keepdim=True).clamp(min=1e-5) / 127
    x_q = (x.div(scale)).round().clamp(-128, 127).to(torch.int8)
    return x_q, scale

# int8 implementation
def gemm_nt_i8i8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    configs: Tuple = None,
):
    m, k = lhs[0].shape
    n, _ = rhs[0].shape
    num_groups = 1
    best_config = (
        configs
        if configs is not None
        else get_deep_gemm_luts(m, n, k, num_groups=num_groups)
    )
    gemm_int8_int8_bf16_nt(lhs, rhs, out, best_config)

def grouped_gemm_nt_i8i8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    configs=None,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
):
    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    best_config = (
        configs
        if configs is not None
        else get_deep_gemm_luts(expected_m, n, k, num_groups=num_groups)
    )

    return m_grouped_gemm_int8_int8_bf16_nt_masked(
        lhs,
        rhs,
        out,
        masked_m,
        expected_m,
        best_config,
        **(
            dict(
                enable_sbo_overlap=True,
                max_block_n=max_block_n,
                signal=overlap_args.signal,
            )
            if overlap_args is not None
            else {}
        ),
    )

def grouped_gemm_nt_i8i8bf16_nopad(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor = None,
    configs=None,
):
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    best_config = (
        configs
        if configs is not None
        else get_deep_gemm_luts(m, n, k, num_groups=num_groups, nopad=True)
    )

    m_grouped_gemm_int8_int8_bf16_nt_nopad(
        lhs, rhs, out, m_indices, m_rows, best_config
    )

def grouped_gemm_nt_bf16bf16bf16_nopad(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor = None,
    configs=None,
):
    m, k = lhs.shape
    num_groups, n, _ = rhs.shape
    best_config = (
        configs
        if configs is not None
        else get_deep_gemm_luts(m, n, k, num_groups=num_groups, nopad=True)
    )

    m_grouped_gemm_bf16_bf16_bf16_nt_nopad(
        lhs, rhs, out, m_indices, m_rows, best_config
    )


def grouped_gemm_nt_bf16bf16bf16_masked(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    configs=None,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
):
    num_groups, _, k = lhs.shape
    _, n, _ = rhs.shape
    best_config = (
        configs
        if configs is not None
        else tuner.get_deep_gemm_config(expected_m, n, k, num_groups=num_groups)
    )

    m_grouped_gemm_bf16_bf16_bf16_nt_masked(
        lhs,
        rhs,
        out,
        masked_m,
        expected_m,
        **(
            dict(
                enable_sbo_overlap=True,
                max_block_n=max_block_n,
                signal=overlap_args.signal,
            )
            if overlap_args is not None
            else {}
        ),
    )

def exec_tuning_iter(func, func_name, debug_mode=False):
    if debug_mode:
        return func()
    else:
        try:
            return func()
        except Exception as e:
            print(f"Error in {func_name}: {e}")
            return None

@triton.jit
def round_up_triton(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y

@triton.jit
def _fwd_kernel_ep_scatter_1(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)

    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        num_recv_tokens_per_expert + offset_cumsum,
        mask=offset_cumsum < num_experts,
        other=0,
    )
    cumsum = tl.cumsum(tokens_per_expert).to(tl.int32) - tokens_per_expert
    tl.store(expert_start_loc + offset_cumsum, cumsum, mask=offset_cumsum < num_experts)
    tl.debug_barrier()

    cur_expert_start = tl.load(expert_start_loc + cur_expert)
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)

    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)

    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E, num_stages=4):
        tl.store(
            m_indices_start_ptr + start_m + off_expert,
            cur_expert,
        )

@triton.jit
def _fwd_kernel_ep_scatter_2_optimal(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    with_scale: tl.constexpr,
    topk_num: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr,
    COPY_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    token_id = tl.program_id(0)

    expert_offsets = tl.arange(0, BLOCK_SIZE)
    expert_mask = expert_offsets < topk_num
    expert_loc = tl.load(
        recv_topk + token_id * recv_topk_stride0 + expert_offsets, mask=expert_mask
    )

    tt_mask = expert_mask & (expert_loc >= 0)
    if tt_mask.sum() == 0:
        return
    dest_token_index_int32 = tl.atomic_add(
        expert_start_loc + expert_loc, 1, mask=tt_mask
    )
    tl.store(
        output_index + token_id * output_index_stride0 + expert_offsets,
        dest_token_index_int32,
        mask=tt_mask,
        eviction_policy="evict_last",
    )
    tl.debug_barrier()
    dest_token_index = tl.load(
        output_index + token_id * output_index_stride0 + expert_offsets, mask=tt_mask
    ).to(tl.int64)

    value_offsets = tl.arange(0, COPY_SIZE)
    for _ in tl.range(0, triton.cdiv(HIDDEN_SIZE, COPY_SIZE), num_stages=num_stages):
        copy_mask = value_offsets < HIDDEN_SIZE
        to_copy = tl.load(
            recv_x + token_id * recv_x_stride0 + value_offsets, mask=copy_mask
        )
        output_offsets = (
            dest_token_index[:, None] * output_tensor_stride0 + value_offsets[None, :]
        )
        to_copy = to_copy[None, :].broadcast_to(BLOCK_SIZE, COPY_SIZE)
        tl.store(
            output_tensor + output_offsets,
            to_copy,
            mask=(copy_mask[None, :] & tt_mask[:, None]),
        )
        value_offsets += COPY_SIZE

    if with_scale:
        scale_offsets = tl.arange(0, COPY_SIZE)
        for _ in tl.range(
            0,
            triton.cdiv(SCALE_HIDDEN_SIZE, COPY_SIZE),
        ):
            copy_mask = scale_offsets < SCALE_HIDDEN_SIZE
            to_copy_scale = tl.load(
                recv_x_scale + token_id * recv_x_scale_stride0 + scale_offsets,
                mask=copy_mask,
            )
            output_scale_offsets = (
                dest_token_index[:, None] * output_tensor_scale_stride0
                + scale_offsets[None, :]
            )
            to_copy_scale = to_copy_scale[None, :].broadcast_to(BLOCK_SIZE, COPY_SIZE)
            tl.store(
                output_tensor_scale + output_scale_offsets,
                to_copy_scale,
                mask=(copy_mask[None, :] & tt_mask[:, None]),
            )
            scale_offsets += COPY_SIZE


@torch.no_grad()
def ep_scatter_sail(
    recv_x: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
    recv_x_scale: Optional[torch.Tensor],
    output_tensor_scale: Optional[torch.Tensor],
    BLOCK_E: int = 128,  # token num of per expert is aligned to BLOCK_E
    BLOCK_D: int = 128,
):

    num_warps = 8
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    if recv_x.dtype == torch.int8:
        # current not support int8 blockwise quant, change the BLOCK_D to hidden_size
        BLOCK_D = hidden_size
    # grid = (triton.cdiv(hidden_size, BLOCK_D), num_experts)
    grid = num_experts

    assert m_indices.shape[0] % BLOCK_E == 0

    _fwd_kernel_ep_scatter_1[(grid,)](
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=num_warps,
        BLOCK_E=BLOCK_E,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )

    grid = lambda meta: (recv_x.shape[0],)
    _fwd_kernel_ep_scatter_2_optimal[grid](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        0 if recv_x_scale is None else recv_x_scale.stride(0),
        0 if recv_x_scale is None else recv_x_scale.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor_scale,
        0 if output_tensor_scale is None else output_tensor_scale.stride(0),
        0 if output_tensor_scale is None else output_tensor_scale.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        with_scale=(recv_x_scale is not None),
        topk_num=recv_topk.shape[1],
        HIDDEN_SIZE=hidden_size,
        SCALE_HIDDEN_SIZE=hidden_size // BLOCK_D,
        BLOCK_SIZE=triton.next_power_of_2(recv_topk.shape[1]),
        COPY_SIZE=512,
        num_stages=3,
        num_warps=8,
    )
    return

def compute_aligned_M(
    M: int,
    num_topk: int,
    local_num_experts: int,
    alignment: int,
):
    def round_up(x: int, y: int) -> int:
        return ((x + y - 1) // y) * y
    # expert_num_tokens information is not available on the cpu.
    # compute the max required size.
    M_sum = (M * num_topk) + local_num_experts * (alignment - 1)
    M_sum = round_up(M_sum, alignment)
    return M_sum

def deepgemm_moe_permute(
    aq: torch.Tensor,
    aq_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    local_num_experts: int,
    aq_out: Optional[torch.Tensor] = None,
    block_align: int = 1,
    block_k: int = 1,
):
    assert aq.ndim == 2
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"
    H = aq.size(1)
    device = aq.device

    M_sum = compute_aligned_M(
        M=topk_ids.size(0),
        num_topk=topk_ids.size(1),
        local_num_experts=local_num_experts,
        alignment=block_align,
    )

    expert_start_loc = torch.empty(
        (local_num_experts), device=device, dtype=torch.int32
    )

    assert aq_out is None or aq_out.shape == (M_sum, H)
    if aq_out is None:
        aq_out = torch.empty((M_sum, H), device=device, dtype=aq.dtype)

    aq_scale_out = torch.empty(
        (M_sum, (H + block_k - 1) // block_k), device=device, dtype=torch.float32
    )

    expert_ids = torch.zeros((M_sum), device=device, dtype=torch.int32)
    inv_perm = torch.empty(topk_ids.shape, device=device, dtype=torch.int32)

    expert_num_tokens = count_expert_num_tokens(
        topk_ids, local_num_experts, block_align
    )

    ep_scatter_sail(
        recv_x=aq,
        recv_x_scale=aq_scale,
        recv_topk=topk_ids.to(torch.int32),
        num_recv_tokens_per_expert=expert_num_tokens,
        expert_start_loc=expert_start_loc,
        output_tensor=aq_out,
        output_tensor_scale=aq_scale_out,
        m_indices=expert_ids,
        output_index=inv_perm,
        BLOCK_E=block_align,
    )
    return aq_out, aq_scale_out, expert_ids, inv_perm, expert_num_tokens

### End of Int8 Tools ###


class PrettyTable:
    def __init__(self):
        self.table_row = []
        self.float_format = None
        self.field_names = None

    def add_row(self, item):
        if not isinstance(item, (list, tuple)):
            raise TypeError("add_row() expects a list or tuple")
        self.table_row.append(list(item))

    def __str__(self):
        if not self.table_row:
            return "(empty table)"

        max_cols = max(len(row) for row in self.table_row)
        
        rows_to_print = []
        if self.field_names:
            header = list(self.field_names)[:max_cols] + [''] * (max_cols - len(self.field_names))
            rows_to_print.append(header)

        for row in self.table_row:
            padded_row = list(row)[:max_cols] + [''] * (max_cols - len(row))
            rows_to_print.append(padded_row)

        col_widths = [0] * max_cols
        for row in rows_to_print:
            for i, cell in enumerate(row):
                if isinstance(cell, float) and self.float_format:
                    display_str = f"{cell:{self.float_format}}"
                else:
                    display_str = str(cell)
                col_widths[i] = max(col_widths[i], len(display_str))

        def make_separator():
            parts = ["+" + "-" * (w + 2) + "+" for w in col_widths]
            return "".join(parts)

        lines = []
        for i, row in enumerate(rows_to_print):
            formatted_cells = []
            for j, cell in enumerate(row):
                if isinstance(cell, float) and self.float_format:
                    s = f"{cell:{self.float_format}}"
                else:
                    s = str(cell)
                formatted_cells.append(f" {s:<{col_widths[j]}} ")
            line = "|" + "|".join(formatted_cells) + "|"
            lines.append(line)
            if i == 0 and self.field_names:
                lines.append(make_separator())

        return "\n".join(lines)

