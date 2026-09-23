import os
import csv
import subprocess
import re
import traceback
import functools
from deep_gemm import get_m_alignment_for_contiguous_layout
try:
    from deep_gemm import get_col_major_tma_aligned_tensor
    from deep_gemm import bench_kineto
except:
    from deep_gemm import get_mn_major_tma_aligned_tensor
    from deep_gemm.testing import bench_kineto

import deep_gemm
import random
import torch
import torch.nn.functional as F
from typing import Tuple, Callable
from enum import Enum
import ast
from math_utils import *
import numpy as np
from deep_gemm.jit_kernels.gemm import get_best_configs as bf16_get_best_configs
from deep_gemm.jit_kernels.gemm_fp8 import get_best_configs as fp8_blkwise_get_best_configs
from deep_gemm.jit_kernels.gemm_int8 import get_best_configs as perchannel_get_best_configs

global _acc_check, _benchmark, _ref_backend
global use_ppu, show_log
_acc_check, _benchmark, use_ppu = True, False, True
show_log = False if "show_log" not in os.environ.keys() else True
_ref_backend = "device"

class KernelType(Enum):
    Kernel1D1D = 0
    Kernel1D2D = 1
    KernelNoSF = 2

    def is_1d1d(self):
        return self.value == 0

    def is_1d2d(self):
        return self.value == 1

    def is_nosf(self):
        return self.value == 2

class MajorTypeAB(Enum):
    KMajor = 0
    MNMajor = 1

    def is_k_major(self):
        return self.value == 0

    def is_mn_major(self):
        return self.value == 1

def get_arch_major() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major

def get_arch_version() -> int:
    # Compute capability as a single comparable integer (major * 10 + minor):
    # PPU1.0 (ZW810E) reports 8.0 -> 80, PPU1.5 (ZW890) reports 8.9 -> 89.
    # Usage: `get_arch_version() >= 89` means "PPU1.5 (sm_89) or later".
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor

def test_filter(condition: Callable):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if condition():
                func(*args, **kwargs)
            else:
                print(f'{func.__name__}:')
                print(f' > Filtered by {condition}')
                print()
        return wrapper
    return decorator

def get_kernel_types(dtype: torch.dtype) -> KernelType:
    if dtype == torch.bfloat16:
        return KernelType.KernelNoSF

    return KernelType.Kernel1D2D if get_arch_major() == 9 else KernelType.Kernel1D1D

def get_ue8m0_usage(kernel_type: KernelType) -> bool:
    if get_arch_major() == 9:
        return False
    return kernel_type.is_1d1d()

def generate_normal(m: int, n: int, k: int,
                    major_a: MajorTypeAB, major_b: MajorTypeAB,
                    accumulate: bool, out_dtype: torch.dtype,
                    kernel_type: KernelType,
                    use_ue8m0: bool = False, use_bf16: bool = False):
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    d = torch.randn((m, n), device='cuda', dtype=out_dtype) * 32 if accumulate else \
        torch.empty((m, n), device='cuda', dtype=out_dtype)
    c = d if accumulate else None
    ref_d = (a.float() @ b.float().t() + (c if accumulate else 0)).to(out_dtype)

    if use_bf16:
        a = a if major_a.is_k_major() else a.T.contiguous().T
        b = b if major_b.is_k_major() else b.T.contiguous().T
        return a, b, c, d, ref_d

    a_fp8 = deep_gemm.per_token_cast_to_fp8(a, use_ue8m0=use_ue8m0)
    b_fp8 = deep_gemm.per_token_cast_to_fp8(b, use_ue8m0=use_ue8m0) if kernel_type.is_1d1d() and accumulate \
            else deep_gemm.per_block_cast_to_fp8(b, use_ue8m0=use_ue8m0)
    a_fp8 = a_fp8 if major_a.is_k_major() else (a_fp8[0].T.contiguous().T, a_fp8[1])
    b_fp8 = b_fp8 if major_b.is_k_major() else (b_fp8[0].T.contiguous().T, b_fp8[1])
    return a_fp8, b_fp8, c, d, ref_d

def generate_m_grouped_contiguous(num_groups: int, m: int, n: int, k: int, distribution: str, alignment: int,
                                  major_a: MajorTypeAB, major_b: MajorTypeAB,
                                  use_ue8m0: bool = False, use_bf16: bool = False):
    group_ms = construct_group_m_list(distribution, num_groups, m)
    m = sum([ceil_div(x, alignment) * alignment for x in group_ms])
    m_indices = torch.empty(m, device='cuda', dtype=torch.int32)

    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_d = torch.randn((m, n), device='cuda', dtype=torch.bfloat16)

    start = 0
    for i, group_m in enumerate(group_ms):
        actual_end = start + group_m
        aligned_end = start + ceil_div(group_m, alignment) * alignment
        m_indices[start:actual_end] = i
        m_indices[actual_end:aligned_end] = -1
        if _acc_check:
            ref_d[start:aligned_end] = a[start:aligned_end] @ b[i].t()
        start = aligned_end

    if _acc_check:
        ref_d= torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(ref_d), ref_d)

    if use_bf16:
        b = b if major_b.is_k_major() else b.mT.contiguous().mT
        return m, a, b, m_indices, d, ref_d

    assert major_a.is_k_major()
    a_fp8 = deep_gemm.per_token_cast_to_fp8(a, use_ue8m0=use_ue8m0)
    b_fp8 = (torch.empty_like(b, dtype=torch.float8_e4m3fn),
             torch.empty((num_groups, ceil_div(n, 128), ceil_div(k, 128)), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        b_fp8[0][i], b_fp8[1][i] = deep_gemm.per_block_cast_to_fp8(b[i], use_ue8m0=use_ue8m0)
    b_fp8 = b_fp8 if major_b.is_k_major() else (b_fp8[0].mT.contiguous().mT, b_fp8[1])
    return m, a_fp8, b_fp8, m_indices, d, ref_d

def generate_m_grouped_masked(num_groups: int, m: int, n: int, k: int, distribution: str, expected_m_per_group: int,
                              use_ue8m0: bool = False, use_bf16: bool = False):
    list_m =  construct_group_m_list(distribution, num_groups, m, is_mask=True, em=expected_m_per_group)
    max_m = find_next_power_of_2(list_m)
    masked_m = torch.tensor(list_m, device='cuda', dtype=torch.int)
    assert masked_m.amax().item() <= max_m

    a = torch.randn((num_groups, max_m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    d = torch.empty((num_groups, max_m, n), device='cuda', dtype=torch.bfloat16)
    ref_d = torch.einsum('gmk,gnk->gmn', a, b)

    if use_bf16:
        return a, b, masked_m, d, ref_d

    a_fp8 = (torch.empty_like(a, dtype=torch.float8_e4m3fn), torch.empty((num_groups, max_m, ceil_div(k, 128)), device='cuda', dtype=torch.float))
    b_fp8 = (torch.empty_like(b, dtype=torch.float8_e4m3fn), torch.empty((num_groups, ceil_div(n, 128), ceil_div(k, 128)), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        a_fp8[0][i], a_fp8[1][i] = deep_gemm.per_token_cast_to_fp8(a[i], use_ue8m0=use_ue8m0)
        b_fp8[0][i], b_fp8[1][i] = deep_gemm.per_block_cast_to_fp8(b[i], use_ue8m0=use_ue8m0)

    return a_fp8, b_fp8, masked_m, d, ref_d

class DGStatus(Enum):
    Pass = 0
    Fail = 1
    Skip = 2
def judge_device_type():
    device_name = torch.cuda.get_device_name()
    # cmodel device name return empty
    use_ppu_ = (device_name.lower().find("ppu") != -1) or (device_name.lower().find("zw") != -1) or device_name == ""
    if not any(k in device_name.lower() for k in ['ppu', 'zw','nvidia','']):
        raise ValueError("Unrecognized device name: {}!".format(device_name))
    global use_ppu
    use_ppu = use_ppu_
    return use_ppu_
use_ppu = judge_device_type()

def set_acc_check(value):
    global _acc_check
    _acc_check = value

def get_acc_check():
    return _acc_check

def set_benchmark(value):
    global _benchmark
    _benchmark = value

def get_benchmark():
    return _benchmark

def set_ref_backend(value):
    global _ref_backend
    _ref_backend = value

def get_ref_backend():
    return _ref_backend

def check_signal(num_local_expert, max_m, block_m, threshold, signal, masked_m):
    ceil_div = lambda a, b: (a + b - 1) // b

    expert_len = max_m // block_m
    for expert in range(num_local_expert):
        mask = masked_m[expert]
        start = expert * expert_len
        end = expert * expert_len + expert_len
        valid_len = ceil_div(mask, block_m)
        for i in range(start, end):
            if i < start + valid_len:
                assert signal[i] == threshold, f'{i=}, {signal[i]=}, {threshold=}'
            else:
                assert signal[i] == 0, f'{i=}, {signal[i]=}'

def construct(m: int, k: int, n: int, d: torch.dtype, quant_type: str = "block") -> \
        Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor], torch.Tensor]:
    tensor_device = 'cuda' if get_ref_backend() == "device" else 'cpu'
    x = torch.randn((m, k), device=tensor_device, dtype=torch.bfloat16)
    if d == torch.float32:
        # Match upstream prenorm coverage: sample the weights directly in FP32.
        y = torch.randn((n, k), device=tensor_device, dtype=torch.float32)
        out = torch.empty((m, n), device=tensor_device, dtype=torch.float32)
        out_s = torch.empty((m,), device=tensor_device, dtype=torch.float32)
        if _acc_check:
            ref_out = x.float() @ y.t()
            ref_s = x.float().square().sum(-1)
        else:
            ref_out = torch.empty_like(out)
            ref_s = torch.empty_like(out_s)
        return x.to('cuda'), y.to('cuda'), out.to('cuda'), out_s.to('cuda'), ref_out.to('cuda'), ref_s.to('cuda')

    y = torch.randn((n, k), device=tensor_device, dtype=torch.bfloat16)
    out = torch.empty((m, n), device=tensor_device, dtype=torch.bfloat16)

    if _acc_check:
        ref_out = x @ y.t()
    else:
        ref_out = torch.empty_like(out)

    if d == torch.bfloat16:
        return x.to('cuda'), y.to('cuda'), out.to('cuda'), ref_out.to('cuda')
    elif d == torch.int8:
        x_int8, y_int8 = per_token_cast_to_int8(x), per_token_cast_to_int8(y)
        return  (x_int8[0].to('cuda'), x_int8[1].to('cuda')), (y_int8[0].to('cuda'), y_int8[1].to('cuda')), out.to('cuda'), ref_out.to('cuda')
    elif d == torch.uint8:
        from deep_gemm import preprocess_mxfp4_scales
        from test_fp4_core import quantize_fp4_torch, dequantize_fp4_torch
        A = torch.randn(m, k, dtype=torch.bfloat16, device='cuda').contiguous()
        B = torch.randn(n, k, dtype=torch.bfloat16, device='cuda').contiguous()
        a, a_scale = quantize_fp4_torch(A)
        b, b_scale = quantize_fp4_torch(B)
        a_dequant = dequantize_fp4_torch(a, a_scale)
        b_dequant = dequantize_fp4_torch(b, b_scale)
        out = torch.zeros(m, n, dtype=torch.bfloat16, device='cuda')
        a_scale = preprocess_mxfp4_scales(scale=a_scale)
        b_scale = preprocess_mxfp4_scales(scale=b_scale)
        ref_out = torch.mm(a_dequant, b_dequant.T)
        ref_out = ref_out.to(torch.bfloat16)
        return (a, a_scale), (b, b_scale), out, ref_out
    elif d == torch.float8_e4m3fn:
        if quant_type == "channel":
            x_fp8, y_fp8 = per_custom_dims_cast_to_fp8(x, (0, ), False, True), per_custom_dims_cast_to_fp8(y, (0, ), False, True)
        else:
            x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
        # Transpose earlier so that the testing will not trigger transposing kernels
        if use_ppu:
            # x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
            x_fp8 = (x_fp8[0], x_fp8[1])
        else:
            x_fp8 = (x_fp8[0], get_mn_major_tma_aligned_tensor(x_fp8[1]))
        return (x_fp8[0].to('cuda'),x_fp8[1].to('cuda')), (y_fp8[0].to('cuda'), y_fp8[1].to('cuda')), out.to('cuda'), ref_out.to('cuda')
    else:
        print("ERROR: Unsupported dtype, please check!")
        exit(1)

def unbincount(counts):
    return torch.repeat_interleave(torch.arange(len(counts)),torch.tensor(counts))

def bincount(counts, min_lenth):
    return torch.bincount(torch.tensor(counts), weights=None, minlength=min_lenth)

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

def truncated_zipf(num_experts, tokens, a=1.1, random_seed=None):
    """
    return list[int]: length=num_experts, means selected time of each expert
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # 1. custom zifp: P(i) ∝ 1/i^a, allow a < 1
    v = torch.arange(1, num_experts + 1, dtype=torch.float32)
    p = 1.0 / (v ** a)
    p = p / p.sum()  # normalize

    # 2. add noise
    noise = torch.randn(tokens, num_experts) * 0.1
    logits = p.log().unsqueeze(0) + noise
    pertoken_p = torch.softmax(logits, dim=-1)  # [tokens, num_experts]

    # 3. sampling
    selected_experts = torch.multinomial(pertoken_p, num_samples=1).squeeze(-1)  # [tokens]
    count = np.bincount(selected_experts, minlength=num_experts)
    return count

def construct_group_m_list(distribution, num_groups = int, m = int, is_mask=False, seed=0, em=0):
    group_m_list = list()
    if is_mask and em != 0:
        expected_m_per_group = em
    else:
        expected_m_per_group = ceil_div(m, num_groups)
    if distribution == None: # default value
        distribution = "uniform"
    if type(distribution) is list:
        return distribution
    elif distribution.endswith("dump") or distribution.endswith("bin"):
        print(f"Read dump from {distribution}")
        group_m_list = read_numbers_from_file(distribution)
        if len(group_m_list) < num_groups:
            for i in range(len(group_m_list), num_groups):
                group_m_list.append(0)
    elif distribution == "uniform":
        random.seed(seed)
        group_m_list = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
    elif distribution == "normal" or distribution == "gaussian":
        random.seed(seed)
        # avg is expected_m_per_group, sigma is expected_m_per_group * 0.5
        group_m_list = [max(0, int(random.gauss(expected_m_per_group, expected_m_per_group * 0.5))) for _ in range(num_groups)]
    elif "zipf" in distribution:
        np.random.seed(seed)
        random.seed(seed)
        zipf_a = 1.1 if "." not in distribution else float(distribution.replace("zipf",""))
        group_m_list = truncated_zipf(num_groups, m, a=zipf_a)
        # dist = np.random.zipf(zipf_a, num_groups)
        # Add noise to zipf distribution to avoid many identical values
        # noise = [random.gauss(0, 1) for _ in range(num_groups)]
        # dist = dist + noise
        # scale = expected_m_per_group * num_groups / dist.sum()
        # group_m_list = [max(0, x) for x in np.round(dist * scale).astype(int)]
    else:
        print("ERROR: Unsupported distribution type, please check!")
        exit(1)
    if not is_mask:
        group_m_list = round_m_list_sum_to_m(group_m_list, m)
        assert sum(group_m_list) <= m, f"sum(m_list)={sum(group_m_list)} must <= m_sum={m}"
    group_m_list = sorted(group_m_list, reverse=True)
    if show_log:
        print(f"distribution:{group_m_list}")
    return group_m_list

def silu_and_mul_post_quant_torch(ref_out: torch.Tensor, swiglu_limit: float = 0.0):
    """weight consists [gate, up]"""
    from test_fp4_core import quantize_fp4_torch

    d2 = ref_out.shape[-1]
    assert d2 % 2 == 0, f"the last dim must be even, got {d2}"
    gate, up = ref_out.split(d2 // 2, dim=-1)

    if swiglu_limit is not None and swiglu_limit > 0.0:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)

    silu =  F.silu(gate) * up
    ref_out_quanted, ref_out_quanted_scale = quantize_fp4_torch(src_tensor=silu)
    return ref_out_quanted, ref_out_quanted_scale

def quantize_fp4_weight_torch(weight: torch.Tensor):
    ### for GroupedNoPad & GroupedFused
    from test_fp4_core import quantize_fp4_torch

    num_groups, n, k = weight.shape

    y_fp4 = torch.empty((num_groups, n, int(k / 2)), device='cuda', dtype=torch.uint8)
    y_fp4_scale_list = []
    for i in range(num_groups):
        b_packed, b_s = quantize_fp4_torch(weight[i])
        y_fp4[i] = b_packed
        y_fp4_scale_list.append(b_s)
    y_fp4_scale = torch.stack(y_fp4_scale_list, dim=0)

    return y_fp4, y_fp4_scale


def construct_non_permute_grouped(num_groups: int, num_token: int, k: int, n: int, topk:int, d: torch.dtype, quant_type: str, group_size = 32, nopad = False, **kwargs) -> \
        Tuple[int, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    tensor_device = 'cuda' if get_ref_backend() == "device" else 'cpu'

    topk_ids = torch.empty((num_token, topk), device=tensor_device, dtype=torch.int32)
    for i in range(num_token):
        topk_ids[i] = torch.randperm(num_groups)[:topk]

    x = torch.randn((num_token, k), device=tensor_device, dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device=tensor_device, dtype=torch.bfloat16)
    if d in ('w4a16', 'w4fa16', 'w4fa16_s16', 'w4fa16_mma'):
        assert quant_type == 'group'
        # use y_dequant instead of origin y
        y, y_quant, y_scale = quant_w4a16(y, group_size, d)

    out = torch.empty((num_token * topk, n), device=tensor_device, dtype=torch.bfloat16)
    ref_out = torch.empty((num_token * topk, n), device=tensor_device, dtype=torch.bfloat16)

    if _acc_check and not nopad:
        # Step 1: Compute GEMM in Token order
        output_token_order = torch.empty((num_token * topk, n), device=tensor_device, dtype=torch.bfloat16)
        for token_idx in range(num_token):
            for topk_idx in range(topk):
                expert_id = topk_ids[token_idx, topk_idx].item()
                flat_idx = token_idx * topk + topk_idx
                output_token_order[flat_idx] = x[token_idx] @ y[expert_id].T
        # Step 2: Collect (expert_id, src_idx) pairs
        entries = []
        for token_idx in range(num_token):
            for topk_idx in range(topk):
                expert_id = topk_ids[token_idx, topk_idx].item()
                src_idx = token_idx * topk + topk_idx
                entries.append((expert_id, src_idx))
        # Step 3: Sort by expert_id and permute output
        entries.sort(key=lambda e: e[0])
        for dst_idx, (expert_id, src_idx) in enumerate(entries):
            ref_out[dst_idx] = output_token_order[src_idx]

    '''
    #  check for nopad interface
    if _acc_check and nopad:
        # Step 1 Compute GEMM in Token order
        output_token_order = torch.empty((num_token * topk, n), device=tensor_device, dtype=torch.bfloat16)
        x_permuted = torch.randn((num_token * topk, k), device=tensor_device, dtype=torch.bfloat16)
        for token_idx in range(num_token):
            for topk_idx in range(topk):
                expert_id = topk_ids[token_idx, topk_idx].item()
                flat_idx = token_idx * topk + topk_idx
                output_token_order[flat_idx] = x[token_idx] @ y[expert_id].T
        # Step 2: Permute
        entries = []
        for token_idx in range(num_token):
            for topk_idx in range(topk):
                expert_id = topk_ids[token_idx, topk_idx].item()
                src_idx = token_idx * topk + topk_idx
                entries.append((expert_id, src_idx))

        entries.sort(key=lambda e: e[0])
        for dst_idx, (expert_id, src_idx) in enumerate(entries):
            ref_out[dst_idx] = output_token_order[src_idx]
            x_permuted[dst_idx] = x[src_idx // topk]

        topk_ids_permuted = torch.tensor(
            [e[0] for e in entries],
            device=tensor_device,
            dtype=topk_ids.dtype
        )
        topk_ids = topk_ids_permuted
        x = x_permuted
    '''

    if d == torch.bfloat16:
        assert quant_type == 'non_quantized', "BF16 only supports non-quantized inputs, got '{}'".format(quant_type)
        return x.to('cuda'), y.to('cuda'), topk_ids.to('cuda'), out.to('cuda'), ref_out.to('cuda')
    elif d == torch.int8:
        assert quant_type == 'channel', "Expected perchannel quantization for int8, got '{}'".format(quant_type)
        x_int8 = per_token_cast_to_int8(x)
        y_int8 = (torch.empty_like(y, dtype=torch.int8), torch.empty((num_groups, n, 1), device=tensor_device, dtype=torch.float))
        for i in range(num_groups):
            y_int8[0][i], y_int8[1][i] = per_token_cast_to_int8(y[i])
        return (x_int8[0].to('cuda'), x_int8[1].to('cuda')), (y_int8[0].to('cuda'), y_int8[1].to('cuda')), topk_ids.to('cuda'), out.to('cuda'), ref_out.to('cuda')
    elif d == torch.float8_e4m3fn:
        assert quant_type == 'channel' or quant_type == 'block', "Expected perchannel/blockwise quantization for fp8, got '{}'".format(quant_type)
        if quant_type == "channel":
            x_fp8 = per_custom_dims_cast_to_fp8(x, (0, ), False, True)
            y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, n, 1), device=tensor_device, dtype=torch.float))
            for i in range(num_groups):
                y_fp8[0][i], y_fp8[1][i] = per_custom_dims_cast_to_fp8(y[i], (0, ), False, True)
        else: # block wise
            x_fp8 = per_token_cast_to_fp8(x)
            y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, ceil_div(n, 128), k // 128), device=tensor_device, dtype=torch.float))
            for i in range(num_groups):
                y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])
        if use_ppu:
            # x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
            x_fp8 = (x_fp8[0], x_fp8[1])
        else:
            x_fp8 = (x_fp8[0], get_mn_major_tma_aligned_tensor(x_fp8[1]))
        return (x_fp8[0].to('cuda'),x_fp8[1].to('cuda')), (y_fp8[0].to('cuda'), y_fp8[1].to('cuda')), topk_ids.to('cuda'), out.to('cuda'), ref_out.to('cuda')
    elif d == torch.uint8:
        from deep_gemm import preprocess_mxfp4_scales, preprocess_mxfp4_weight_for_act_and_quant_fusing
        from test_fp4_core import quantize_fp4_torch, dequantize_fp4_torch

        epilogue_type = kwargs.get("epilogue_type", "Default")
        swiglu_limit = kwargs.get("swiglu_limit", 0.0)
        if epilogue_type in ["SiluAndMulPostQuantFp4"]:
            ### meta_data
            out_quanted = torch.empty((num_token * topk, (n // 4)), dtype=torch.uint8, device=tensor_device)
            out_quanted_scale = torch.empty((num_token * topk, ceil_div((n // 4), 32)), dtype=torch.uint16, device=tensor_device)
            ref_out = ref_out.to(torch.float)
            ref_out_quanted = None
            ref_out_quanted_scale = None

        x_fp4, x_fp4_scale = quantize_fp4_torch(x)
        x_fp4_scale = x_fp4_scale.view(torch.uint16) # x_fp4_scale is K-Major for GroupedFused
        y_fp4, y_fp4_scale = quantize_fp4_weight_torch(weight=y)
        y_fp4_, y_fp4_scale_ = y_fp4.clone(), y_fp4_scale.clone()
        if epilogue_type in ["SiluAndMulPostQuantFp4"]:
            y_fp4, y_fp4_scale = preprocess_mxfp4_weight_for_act_and_quant_fusing(weight=y_fp4, weight_scale=y_fp4_scale)
        else:
            y_fp4_scale = preprocess_mxfp4_scales(scale=y_fp4_scale)

        # Recompute ref_out from fp4-dequantized values for accurate comparison
        if _acc_check and not nopad:
            ref_out_list = []
            k_, sfk_ = x_fp4.shape[-1], x_fp4_scale.shape[-1]
            x_fp4_permuted = x_fp4.view(num_token, -1, k_).repeat(1, topk, 1).reshape(-1, k_)
            x_fp4_scale_permuted = x_fp4_scale.view(num_token, -1, sfk_).repeat(1, topk, 1).reshape(-1, sfk_).view(torch.uint8)
            topk_ids_ = topk_ids.view(-1)
            for i in range(num_groups):
                mask = (topk_ids_ == i)
                if mask.sum():
                    ref_out_group = dequantize_fp4_torch(x_fp4_permuted[mask], x_fp4_scale_permuted[mask]).to(torch.float) @ dequantize_fp4_torch(y_fp4_[i], y_fp4_scale_[i]).to(torch.float).transpose(0, 1)
                    ref_out_list.append(ref_out_group)
            ref_out = torch.concat(ref_out_list, dim=0)

            if epilogue_type in ["SiluAndMulPostQuantFp4"]:
                ### Reference
                ref_out_quanted, ref_out_quanted_scale = silu_and_mul_post_quant_torch(ref_out=ref_out, swiglu_limit=swiglu_limit)
                ref_out_quanted_scale = preprocess_mxfp4_scales(scale=ref_out_quanted_scale)

                return (x_fp4.to('cuda'), x_fp4_scale.to('cuda')), (y_fp4.to('cuda'), y_fp4_scale.to('cuda')), topk_ids.to('cuda'), (out_quanted.to('cuda'), out_quanted_scale.to('cuda')), (ref_out_quanted.to('cuda'), ref_out_quanted_scale.to('cuda'))
            else:
                ref_out = ref_out.to(torch.bfloat16)

        return (x_fp4.to('cuda'), x_fp4_scale.to('cuda')), (y_fp4.to('cuda'), y_fp4_scale.to('cuda')), topk_ids.to('cuda'), out.to('cuda'), ref_out.to('cuda')
    elif d in ('w4a16', 'w4fa16', 'w4fa16_s16', 'w4fa16_mma'):
        return x.to('cuda'), (y_quant.to('cuda'), y_scale.to('cuda')), topk_ids.to('cuda'), out.to('cuda'), ref_out.to('cuda')
    else:
        print("ERROR: Unsupported dtype, please check!")
        exit(1)

def construct_contiguous_grouped(num_groups: int, m: int, k: int, n: int, d, distribution: str, alignment: int, quant_type: str = "block", group_size = 32, **kwargs) -> \
        Tuple[int, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    tensor_device = 'cuda' if get_ref_backend() == "device" else 'cpu'
    group_ms = construct_group_m_list(distribution, num_groups, m)
    m = sum([ceil_div(x, alignment) * alignment for x in group_ms])
    m_indices = torch.empty(m, device=tensor_device, dtype=torch.int32)
    x = torch.randn((m, k), device=tensor_device, dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device=tensor_device, dtype=torch.bfloat16)
    if d in ('w4a16', 'w4fa16', 'w4fa16_s16', 'w4fa16_mma'):
        assert quant_type == 'group'
        # use y_dequant instead of origin y
        y, y_quant, y_scale = quant_w4a16(y, group_size, d)

    out = torch.empty((m, n), device=tensor_device, dtype=torch.bfloat16)
    ref_out = torch.randn((m, n), device=tensor_device, dtype=torch.bfloat16)

    start = 0
    for i, group_m in enumerate(group_ms):
        actual_end = start + group_m
        aligned_end = start + ceil_div(group_m, alignment) * alignment
        m_indices[start:actual_end] = i
        m_indices[actual_end:aligned_end] = -1
        if _acc_check:
            ref_out[start:aligned_end] = x[start:aligned_end] @ y[i].t()
        start = aligned_end
    if _acc_check:
        ref_out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(ref_out), ref_out)

    if d == torch.bfloat16:
        return m, x.to('cuda'), y.to('cuda'), m_indices.to('cuda'), out.to('cuda'), ref_out.to('cuda')
    elif d == torch.int8:
        x_int8 = per_token_cast_to_int8(x)
        y_int8 = (torch.empty_like(y, dtype=torch.int8), torch.empty((num_groups, n, 1), device=tensor_device, dtype=torch.float))
        for i in range(num_groups):
            y_int8[0][i], y_int8[1][i] = per_token_cast_to_int8(y[i])
        return m, (x_int8[0].to('cuda'), x_int8[1].to('cuda')), (y_int8[0].to('cuda'), y_int8[1].to('cuda')), m_indices.to('cuda'), out.to('cuda'), ref_out.to('cuda')
    elif d == torch.float8_e4m3fn:
        # assert m % 4 == 0, f'TMA alignment error: {m}'
        if quant_type == "channel":
            x_fp8 = per_custom_dims_cast_to_fp8(x, (0, ), False, True)
            y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, n, 1), device=tensor_device, dtype=torch.float))
            for i in range(num_groups):
                y_fp8[0][i], y_fp8[1][i] = per_custom_dims_cast_to_fp8(y[i], (0, ), False, True)
        else: # block wise
            x_fp8 = per_token_cast_to_fp8(x)
            y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, ceil_div(n, 128), k // 128), device=tensor_device, dtype=torch.float))
            for i in range(num_groups):
                y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])
        if use_ppu:
            # x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
            x_fp8 = (x_fp8[0], x_fp8[1])
        else:
            x_fp8 = (x_fp8[0], get_mn_major_tma_aligned_tensor(x_fp8[1]))
        return m, (x_fp8[0].to('cuda'),x_fp8[1].to('cuda')), (y_fp8[0].to('cuda'), y_fp8[1].to('cuda')), m_indices.to('cuda'), out.to('cuda'), ref_out.to('cuda')
    elif d == torch.uint8:
        from deep_gemm import preprocess_mxfp4_scales, preprocess_mxfp4_weight_for_act_and_quant_fusing
        from test_fp4_core import quantize_fp4_torch, dequantize_fp4_torch

        epilogue_type = kwargs.get("epilogue_type", "Default")
        swiglu_limit = kwargs.get("swiglu_limit", 0.0)
        if epilogue_type in ["SiluAndMulPostQuantFp4"]:
            ### meta_data
            out_quanted = torch.empty((m, (n // 4)), dtype=torch.uint8, device=tensor_device)
            out_quanted_scale = torch.empty((m, ceil_div((n // 4), 32)), dtype=torch.uint16, device=tensor_device)
            ref_out = ref_out.to(torch.float)
            ref_out_quanted = None
            ref_out_quanted_scale = None

        # Quantize x (whole tensor) and y (per group) to fp4
        x_fp4 = quantize_fp4_torch(x)
        x_fp4_scale = preprocess_mxfp4_scales(scale=x_fp4[1])
        y_fp4, y_fp4_scale = quantize_fp4_weight_torch(weight=y)
        if epilogue_type in ["SiluAndMulPostQuantFp4"]:
            y_fp4, y_fp4_scale = preprocess_mxfp4_weight_for_act_and_quant_fusing(weight=y_fp4, weight_scale=y_fp4_scale)
        else:
            y_fp4_scale = preprocess_mxfp4_scales(scale=y_fp4_scale)

        # Recompute ref_out from fp4-dequantized values for accurate comparison
        if _acc_check:
            start = 0
            for i, group_m in enumerate(group_ms):
                aligned_end = start + ceil_div(group_m, alignment) * alignment
                a_q, a_s = quantize_fp4_torch(x[start:aligned_end])
                b_q, b_s = quantize_fp4_torch(y[i])
                a_dq = dequantize_fp4_torch(a_q, a_s).to(torch.float)
                b_dq = dequantize_fp4_torch(b_q, b_s).to(torch.float)
                ref_out[start:aligned_end] = a_dq @ b_dq.t()
                start = aligned_end
            ref_out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(ref_out), ref_out)

            if epilogue_type in ["SiluAndMulPostQuantFp4"]:
                ### Reference
                ref_out_quanted, ref_out_quanted_scale = silu_and_mul_post_quant_torch(ref_out=ref_out, swiglu_limit=swiglu_limit)
                ref_out_quanted_scale = preprocess_mxfp4_scales(scale=ref_out_quanted_scale)

                return m, (x_fp4[0].to('cuda'), x_fp4_scale.to('cuda')), (y_fp4.to('cuda'), y_fp4_scale.to('cuda')), m_indices.to('cuda'), (out_quanted.to('cuda'), out_quanted_scale.to('cuda')), (ref_out_quanted.to('cuda'), ref_out_quanted_scale.to('cuda'))

        return m, (x_fp4[0].to('cuda'), x_fp4_scale.to('cuda')), (y_fp4.to('cuda'), y_fp4_scale.to('cuda')), m_indices.to('cuda'), out.to('cuda'), ref_out.to('cuda')
    elif d in ('w4a16', 'w4fa16', 'w4fa16_s16', 'w4fa16_mma'):
        return m, x.to('cuda'), (y_quant.to('cuda'), y_scale.to('cuda')), m_indices.to('cuda'), out.to('cuda'), ref_out.to('cuda')
    else:
        print("ERROR: Unsupported dtype, please check!")
        exit(1)

def construct_grouped_masked(num_groups: int, max_m: int, expected_m_per_group: int, k: int, n: int, d: torch.dtype, distribution: str,
                             enable_sbo_overlap: bool = False, quant_type: str = "block", group_size: int = 32, **kwargs):
    tensor_device = 'cuda' if get_ref_backend() == "device" else 'cpu'
    # Construct mask
    list_m =  construct_group_m_list(distribution, num_groups, max_m, is_mask=True, em=expected_m_per_group)
    masked_m = torch.tensor(list_m, device=tensor_device, dtype=torch.int)
    max_m = max(128, find_next_power_of_2(list_m))
    assert masked_m.amax().item() <= max_m, f"max masked_m={masked_m.amax().item()}, allowed max_m={max_m}"

    x = torch.randn((num_groups, max_m, k), device=tensor_device, dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device=tensor_device, dtype=torch.bfloat16)
    out = torch.empty((num_groups, max_m, n), device=tensor_device, dtype=torch.bfloat16)
    if d in ('w4a16', 'w4fa16', 'w4fa16_s16', 'w4fa16_mma'):
        assert quant_type == 'group'
        # use y_dequant instead of origin y
        y, y_quant, y_scale = quant_w4a16(y, group_size, d)

    if _acc_check:
        ref_out = torch.einsum('gmk,gnk->gmn', x, y)
    else:
        ref_out = torch.empty_like(out)

    max_signal_size = num_groups * ceil_div(max_m, 64)
    signal = torch.zeros(max_signal_size, dtype=torch.int32, device=tensor_device) if enable_sbo_overlap else torch.empty(0).int()

    if d == torch.bfloat16:
        return x.to('cuda'), y.to('cuda'), masked_m.to('cuda'), out.to('cuda'), ref_out.to('cuda'), signal.to('cuda'), max_m
    elif d == torch.int8:
        x_int8 = (torch.empty_like(x, dtype=torch.int8), torch.empty((num_groups, max_m, 1), device=tensor_device, dtype=torch.float))
        y_int8 = (torch.empty_like(y, dtype=torch.int8), torch.empty((num_groups, n, 1), device=tensor_device, dtype=torch.float))
        for i in range(num_groups):
            x_int8[0][i], x_int8[1][i] = per_token_cast_to_int8(x[i])
            y_int8[0][i], y_int8[1][i] = per_token_cast_to_int8(y[i])
        return (x_int8[0].to('cuda'), x_int8[1].to('cuda')), (y_int8[0].to('cuda'), y_int8[1].to('cuda')), masked_m.to('cuda'), out.to('cuda'), ref_out.to('cuda'), signal.to('cuda'), max_m
    elif d == torch.float8_e4m3fn:
        if quant_type == "channel":
            x_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty((num_groups, max_m, 1), device=tensor_device, dtype=torch.float))
            y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, n, 1), device=tensor_device, dtype=torch.float))
            for i in range(num_groups):
                x_fp8[0][i], x_fp8[1][i] = per_custom_dims_cast_to_fp8(x[i], (0, ), False, True)
                y_fp8[0][i], y_fp8[1][i] = per_custom_dims_cast_to_fp8(y[i], (0, ), False, True)
        else: # block wise
            x_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty((num_groups, max_m, k // 128), device=tensor_device, dtype=torch.float))
            y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, (n + 127) // 128, k // 128), device=tensor_device, dtype=torch.float))
            for i in range(num_groups):
                x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])
                y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])
        # Transpose earlier so that the testing will not trigger transposing kernels
        if use_ppu:
            # x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
            x_fp8 = (x_fp8[0], x_fp8[1])
        else:
            x_fp8 = (x_fp8[0], get_mn_major_tma_aligned_tensor(x_fp8[1]))
        return (x_fp8[0].to('cuda'),x_fp8[1].to('cuda')), (y_fp8[0].to('cuda'), y_fp8[1].to('cuda')), masked_m.to('cuda'), out.to('cuda'), ref_out.to('cuda'), signal.to('cuda'), max_m
    elif d == torch.uint8:
        from deep_gemm import preprocess_mxfp4_scales, preprocess_mxfp4_weight_for_act_and_quant_fusing
        from test_fp4_core import quantize_fp4_torch, dequantize_fp4_torch

        epilogue_type = kwargs.get("epilogue_type", "Default")
        swiglu_limit = kwargs.get("swiglu_limit", 0.0)
        if epilogue_type in ["SiluAndMulPostQuantFp4"]:
            ### meta_data
            out_quanted = torch.empty((num_groups, max_m, (n // 4)), dtype=torch.uint8, device=tensor_device)
            out_quanted_scale = torch.empty((num_groups, max_m, ceil_div((n // 4), 32)), dtype=torch.uint16, device=tensor_device)
            ref_out = ref_out.to(torch.float)
            ref_out_quanted = None
            ref_out_quanted_scale = None

        # Quantize x and y per group to fp4 and compute ref from dequantized values
        x_fp4_list, x_fp4_scale_list = [], []
        y_fp4_list, y_fp4_scale_list = [], []
        x_ref_list, y_ref_list = [], []
        for i in range(num_groups):
            a_q, a_s = quantize_fp4_torch(x[i])
            b_q, b_s = quantize_fp4_torch(y[i])
            x_fp4_list.append(a_q)
            x_fp4_scale_list.append(a_s)
            y_fp4_list.append(b_q)
            y_fp4_scale_list.append(b_s)
            x_ref_list.append(dequantize_fp4_torch(a_q, a_s).to(torch.float))
            y_ref_list.append(dequantize_fp4_torch(b_q, b_s).to(torch.float))
        x_fp4 = torch.stack(x_fp4_list, dim=0)
        x_fp4_scale = preprocess_mxfp4_scales(scale=torch.stack(x_fp4_scale_list, dim=0))
        y_fp4 = torch.stack(y_fp4_list, dim=0)
        y_fp4_scale = torch.stack(y_fp4_scale_list, dim=0)
        if epilogue_type in ["SiluAndMulPostQuantFp4"]:
            y_fp4, y_fp4_scale = preprocess_mxfp4_weight_for_act_and_quant_fusing(weight=y_fp4, weight_scale=y_fp4_scale)
        else:
            y_fp4_scale = preprocess_mxfp4_scales(scale=y_fp4_scale)

        if _acc_check:
            x_ref = torch.stack(x_ref_list, dim=0)
            y_ref = torch.stack(y_ref_list, dim=0)
            ref_out = torch.einsum('gmk,gnk->gmn', x_ref, y_ref)

            if epilogue_type in ["SiluAndMulPostQuantFp4"]:
                ### Reference
                ref_out_quanted, ref_out_quanted_scale = silu_and_mul_post_quant_torch(ref_out=ref_out, swiglu_limit=swiglu_limit)
                ref_out_quanted_scale = preprocess_mxfp4_scales(scale=ref_out_quanted_scale)

                return (x_fp4.to('cuda'), x_fp4_scale.to('cuda')), (y_fp4.to('cuda'), y_fp4_scale.to('cuda')), masked_m.to('cuda'), (out_quanted.to('cuda'), out_quanted_scale.to('cuda')), (ref_out_quanted.to('cuda'), ref_out_quanted_scale.to('cuda')), signal.to('cuda'), max_m

        return (x_fp4.to('cuda'), x_fp4_scale.to('cuda')), (y_fp4.to('cuda'), y_fp4_scale.to('cuda')), masked_m.to('cuda'), out.to('cuda'), ref_out.to('cuda').to(torch.bfloat16), signal.to('cuda'), max_m
    elif d in ('w4a16', 'w4fa16', 'w4fa16_s16', 'w4fa16_mma'):
        return x.to('cuda'), (y_quant.to('cuda'), y_scale.to('cuda')), masked_m.to('cuda'), out.to('cuda'), ref_out.to('cuda'), signal.to('cuda'), max_m
    else:
        print("ERROR: Unsupported dtype, please check!")
        exit(1)

def run_cmd(cmd: str, timeout=3600, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
    print(f"Run command: {cmd}, timeout: {timeout}")
    try:
        ret = subprocess.run(args=cmd, timeout=timeout, shell=True, stdout=stdout, stderr=stderr, encoding="utf-8")
        if stdout:
            for line in ret.stdout.splitlines() + ret.stderr.splitlines():
                print(line)
        if ret.returncode != 0:
            print(f"Run command failed!")
        else:
            print(f"Run command succeed!")
        return ret
    except Exception as e:
        print(traceback.format_exc())
        return None

def str_to_list(s, type_func=int):
    """Convert a comma-separated string to a list of a specified type."""
    return [type_func(i.strip()) for i in s.split(',')]

def split_list_into_groups(lst, num):
    group_size = len(lst) // num
    remainder = len(lst) % num
    start = 0
    groups = []
    for i in range(num):
        groups.append([])
    for i in range(len(lst)):
        group_idx = i % num
        groups[group_idx].append(lst[i])
    return groups

def worker(gpu_id, cases, output, device, mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Process {os.getpid()} is running on GPU {gpu_id}")
    run_cycle_on_device(cases, output, device, mode, gpu_id)

def read_detail_from_nculog(filename):
    detail_info = {'m': 0, 'n': 0, 'k': 0}
    keyword_pattern = r"(GemmGrouped-BF16|GemmGrouped-FP8|GemmGrouped-INT8|GemV-BF16|GemV-Small-BF16)"
    pattern_dict = {"group": r"group:(\d+)", "problem": r"problem:\[(\d+), (\d+), (\d+)\]", "expected_m":r"expected_m:(\d+)", "gemm_type": r"gemm_type:(\w+)",
    "ThreadblockShape": r"ThreadblockShape\[(\d+), (\d+), (\d+)\]" , "WarpShape": r"WarpShape\[(\d+), (\d+), (\d+)\]", "kNumStages":r"kNumStages:(\d+)",
    "num_sms": r"num_sms:(\d+)", "max_active_tb_num": r"max_active_tb_num:(\d+)", "threadblock_count": r"threadblock_count:(\d+)",
    "smem_size":r"smem_size:(\d+)", "vreg": r"vreg:(\d+)", "stack": r"stack:(\d+)",
    "BlockSize":r"BlockSize:(\d+)", "NPerThread":r"NPerThread:(\d+)", "ThreadPerN":r"ThreadPerN:(\d+)", "NPerBlock":r"NPerBlock:(\d+)", "SWZL_SIZE_M":r"SWZL_SIZE_M:(\d+)"}
    with open(filename, newline='') as log_file:
        lines = log_file.readlines()
        for idx, line in enumerate(lines):
            keyword_match = re.search(keyword_pattern, line)
            if keyword_match:
                target_lines = "".join(lines[idx+1:idx+10])
                for _info, _pattern in pattern_dict.items():
                    _info_match = re.search(_pattern, target_lines)
                    if _info_match:
                        if len(_info_match.groups()) == 1 :
                            detail_info[_info] = _info_match.groups()[0]
                        else:
                            detail_info[_info] = _info_match.groups()
                    else:
                        detail_info[_info] = " "
                break
    detail_info['m'] = detail_info['problem'][0]
    detail_info['n'] = detail_info['problem'][1]
    detail_info['k'] = detail_info['problem'][2]
    # print(detail_info)
    return detail_info


# devices = {
#     "name": ["cycle", "tensor cell efficiency", "waves"],
#      hopper tc: sm__pipe_tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma_cycles_active.avg.pct_of_peak_sustained_elapsed
#     "gpu":  ["gpu__time_duration.sum", "sm__cycles_elapsed.max", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", "launch__waves_per_multiprocessor"],
#     "ppu":  ["ppu__time_duration.sum","ce__cycles_elapsed.max", "cu__we_pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", "launch__waves_per_cu"],
# }

def read_cycle_from_nculog(filename):
    # kernel_pattern = r"(.*)deep_gemm(.*)"
    # kernel_pattern = r"(.*)kernel(.*)Device(.*)"
    kernel_pattern = r"(.*)Device\s+\d+"
    cycles_pattern = r"__cycles_elapsed.max"
    time_pattern = r"time_duration"
    global tc_pattern
    # tc_pattern = "we_pipe_tensor_cycles_active"
    tc_pattern = "_tensor_(.*)avg.pct_of_peak_sustained"
    hbm_pattern = "dram"
    kernel_list = []
    cycles_list = []
    tc_list = []
    hbm_list = []
    time_list = []

    with open(filename, newline='') as log_file:
        for line in log_file.read().split("\n"):
            if re.search(kernel_pattern, line):
                kernel_list.append(line.strip())
            if re.search(cycles_pattern, line):
                cycles_list.append(int(line.strip().split()[-1]))
            if re.search(tc_pattern, line):
                tc_list.append(float(line.strip().split()[-1]))
            if re.search(hbm_pattern, line):
                hbm_list.append(float(line.strip().split()[-1]))
            if re.search(time_pattern, line):
                fwd_time = float(line.strip().split()[-1])
                if "ns" in line: # ppu return ns, gpu return us
                    fwd_time = fwd_time/1000
                time_list.append(round(fwd_time,4))

    if (len(kernel_list) != len(cycles_list)) or (len(kernel_list) != len(tc_list)) or (len(kernel_list) != len(hbm_list)):
        print(f"assert len(kernel_list){len(kernel_list)} == len(cycles_list){len(cycles_list)} == len(tc_list){len(tc_list)} == len(hbm_list){len(hbm_list)} failed!!")
        return 0, 0, [], 0, 0
    # assert(len(kernel_list) == len(cycles_list))
    # assert(len(kernel_list) == len(tc_list))
    # assert(len(kernel_list) == len(hbm_list))

    op_cycles = dict()
    fwd_cycle_sum = 0
    fwd_tc_sum = 0
    fwd_hbm_sum = 0
    fwd_time = 0

    for i in range(len(kernel_list)):
        op = kernel_list[i]
        cycle = cycles_list[i]
        op_cycles[op] = cycle
        # NOTES: the C++ JIT names its kernels by dtype tag (e.g. `attention_mqa_logits_int8`), so
        # unlike the Python JIT -- whose symbol carries `cutlass::gemm::kernel::...` -- "gemm" alone
        # no longer identifies the kernel of interest. `metadata` is excluded so the tiny scheduling
        # kernel is reported in `op_cycles` but not summed, keeping the totals comparable with the
        # Python path (whose `smxx_*_metadata` matches neither term).
        if ("gemm" in op.lower() or "attention" in op.lower()) and "metadata" not in op.lower():
            fwd_cycle_sum += cycle
            fwd_tc_sum = tc_list[i]
            fwd_hbm_sum = hbm_list[i]
            fwd_time = time_list[i]
    # calculate statistics data
    if fwd_cycle_sum != 0:
        # fwd unit case
        return fwd_cycle_sum, fwd_tc_sum, op_cycles, fwd_hbm_sum, fwd_time
    else:
        print("Not valid CSV file!")
        return 0, 0, [], 0, 0
        #exit(-1)

def clean_casename(name):
    _need_replace = ['--', '=', 'format', 'Formatted', '[', ']', ":", "*", " ", ","]
    # for item in _need_replace:
    #     name = name.replace(item, "_")
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    while "__" in name:
        name = name.replace("__", "_")
    return name

def run_cycle_on_device(cases, output_file, dev="gpu", mode="metrics", gpu_id="0"):
    # Kernel filter for the PPU profiler. `attention_` is needed because the C++ JIT emits
    # `attention_mqa_logits_*` / `attention_paged_mqa_logits_*`, which match none of the other terms
    # (the Python JIT is covered by `device_kernel`). Without it the profiler silently reports 0.
    ppu_kernel_name = "--kernel-name 'regex:Kernel|device_kernel|batched_gemvt*|gemm*|attention_'"
    output_lines = list()
    headers = ["casename","time(us)","cycle","tc efficiency", "hbm efficiency", "dtype", "result", "cmd", "detail"]
    # new_row=["casename"]  metrics.get("name", [])  ["detail"]
    # output_lines.append(new_row)
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    total = len(cases)
    for idx, case in enumerate(cases):
        print(f'Profiling {idx + 1}/{total} on device{gpu_id}')
        print(f'case name:{case}')
        log_file = f"./logs/gpu{gpu_id}_{case.replace(' ','').replace(',', '_').replace(':', '_').replace('/','_').replace('.','_')[:100]}.log"
        cmd = "rm -f "+ log_file
        run_cmd(cmd)
        # gpu
        # metrics = devices.get(dev, [])
        # metrics_string = ', '.join(metrics) if metrics else ""
        current_file_path = os.path.abspath(__file__)
        # pattern = r"data_type:(bf16|int8|fp8|tf32|fp4)"
        # dtype = re.search(pattern, case).groups()[0]
        pattern = r"data_type:([a-zA-Z0-9_]+)"
        dtype_match = re.search(pattern, case)
        dtype = dtype_match.groups()[0] if dtype_match else "unknown"
        script = f"{os.path.dirname(current_file_path)}/run_deep_gemm.py"
        if mode == "full":
            output_name = clean_casename(case)[:50]
            cmd = "{} --set full {} -o {} python {} --format {} \
                2>&1 | tee {}".format("ncu" if dev == "gpu" else "acu",
                                      "--kernel-name 'regex:Kernel|device_kernel|batched_gemvt*|gemm*'"
                                      if dev == "gpu" else ppu_kernel_name,
                                      output_name, script, case, log_file)
            ret = run_cmd(cmd)
            continue
        else:
            if mode == "show_log":
                os.environ["show_log"] = "1"
            if dev == "gpu":
                arch = get_arch_major()
                if arch == 8:
                    metrics_string = "gpu__time_duration.sum,sm__cycles_elapsed.max,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed"
                elif arch == 9:
                    metrics_string = "gpu__time_duration.sum,sm__cycles_elapsed.max,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed"
                else:
                    metrics_string = "gpu__time_duration.sum,sm__cycles_elapsed.max,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed"
            else:
                # "ce__cycles_elapsed.max,cu__we_pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.pct_of_peak_sustained_elapsed"
                metrics_string = "ppu__time_duration.sum,ce__cycles_elapsed.max,cu__we_pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.pct_of_peak_sustained_elapsed"

            _acc = "--disable_acc"
            cmd = '{} --clock-control none {} --metrics="{}"  \
                --page=details python {} --format "{}" {} \
                2>&1 | tee {}'.format("ncu" if dev == "gpu" else "acu", '--kernel-name regex:gemm*' if dev == "gpu" else ppu_kernel_name, metrics_string, script, case, _acc, log_file)

        ret = run_cmd(cmd)
        result = "Fail"
        cycle, tc, detail, hbm, time = 0, 0, "", 0, 0
        if mode != "full" and ret != None:
            if ret.returncode == 0:
                cycle, tc, detail, hbm, time = read_cycle_from_nculog(log_file)
                if mode == "show_log":
                    other_metrics = read_detail_from_nculog(log_file)
                    print(other_metrics)

        if _acc_check:
            # check accuracy
            cmd = "python {} --format {}".format(script, case)
            ret = run_cmd(cmd)
            if ret.returncode == 0:
                result = "Pass"
            else:
                result = "Fail"
                print("ERROR: failed to pass accuracy test, please check!!")
        else:
            # only perf, check cycle found
            if cycle != 0:
                result = "Pass"
            else:
                result = "Fail"
                print("ERROR: failed to find cycle info, please check!!")
        row =  [f"'{case.replace(',','_')}_{dtype}'", str(time), str(cycle), str(tc), str(hbm), dtype, result, str(cmd), str(detail)]
        if mode == "show_log":
            if "problem" not in headers:
                headers.extend(other_metrics.keys())
            row.extend(other_metrics.values())
        output_lines.append(row)
        if not os.path.exists(f"{output_file}.csv"):
            with open(f"{output_file}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        with open(f"{output_file}.csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            print("write result succeed")

        if len(cases) == 1 and result == "Fail":
            exit(-1) # only one case, fail and exit

    output_file = output_file + '.csv'
    if len(cases) == 1:
        with open("local.log", "w") as f:
            writer = csv.writer(f)
            for row in output_lines:
                writer.writerow(row)
            print("write result to local.log succeed")

def read_numbers_from_file(file_path):
    numbers = []
    with open(file_path, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:
                try:
                    number = int(stripped_line)
                    numbers.append(number)
                except ValueError:
                    print(f"Warning: skip invalid: {stripped_line}")
    return numbers

def parse_deepgemm_string_re(s):
    # give default value, for fp8 we have block and channel
    result = {"distribution": "uniform", "enable_sbo_overlap": False}
    supported_keys = ["data_type", "groups", "m", "n", "k", "distribution", "em", "enable_sbo_overlap", "num_token", "topk", "group_size", "logits_dtype", "weights_dtype", "c4_compressed", "epilogue_type", "swiglu_limit", "sparse_block_kv", "num_max_sparse_blocks", "use_unaligned_ks"]
    supported_gemm_type = ["GroupedContiguous", "GroupedNoPad", "GroupedFused", "GroupedMasked", "Normal", "DenseGemm", "MqaLogits", "PagedMqaLogits", "MqaAvgLogits", "PagedMqaAvgLogits", "BatchGemm", "SparseMqaLogits", "PagedSparseMqaLogits", ]
    supported_indexer_epilogue_type = ["fp32", "bf16"]
    supported_quant_type = ["non_quantized", "block", "channel", "group"]
    import re
    dg_params = r"(GroupedContiguous|GroupedNoPad|GroupedFused|GroupedMasked|DenseGemm|Normal|MqaAvgLogits|PagedMqaAvgLogits|PagedSparseMqaLogits|SparseMqaLogits|MqaLogits|PagedMqaLogits|BatchGemm),(.+)"
    pattern = re.compile(dg_params)
    m = pattern.search(s.strip("."))
    if not m:
        print("Invalid input format string did not match deepgemm params:", dg_params)
        exit(1)
    grps = m.groups()
    if grps[0] in supported_gemm_type:
        result['gemm_type'] = grps[0]
    param_pattern = r"(\w+):(\[.*?\]|[^,]+)"
    match_string = re.findall(param_pattern, grps[1])
    if len(match_string) == 0:
        print("ERROR: wrong deepgemm format input, please check!!")
    for key, value in match_string:
        if value.startswith('[') and value.endswith(']'):
            result[key] = ast.literal_eval(value)
        elif key == "data_type":
            result[key] = convert_data_type_to_dtype(value)
        elif key == "quant_type":
            if value not in supported_quant_type:
                print("Invalid input quant_type")
                exit(1)
            result[key] = value
        elif key == "logits_dtype" or key == "weights_dtype":
            if value not in supported_indexer_epilogue_type:
                print(f"Invalid input indexer epilogue type")
                exit(1)
            result[key] = convert_data_type_to_dtype(value)
        else:
            try:
                result[key] = int(value)
            except ValueError:
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
                    pass
    if "quant_type" not in result:
        quant_type_defaults = {
            torch.bfloat16: 'non_quantized',
            torch.int8 : 'channel',
            torch.float8_e4m3fn: 'block',
            torch.uint8: 'group',
            'w4a16': 'group',
            'w4fa16': 'group',
            'w4fa16_s16': 'group',
            'w4fa16_mma': 'group'
        }
        result["quant_type"] = quant_type_defaults.get(result["data_type"], 'block')
    return result

def parse_dump_file(file):
    import re, math
    if ("GroupedMasked" in file or "Contiguous" in file or "GroupedNoPad" in file):
        pattern = r'groups(\d+)_m(\d+)_n(\d+)_k(\d+)_em(\d+)'
        match = re.search(pattern, file)

        if match:
            num_groups = int(match.group(1))
            m = int(match.group(2))
            n = int(match.group(3))
            k = int(match.group(4))
            expected_m_per_group = int(match.group(5))

            print(f"m: {m}")
            print(f"n: {n}")
            print(f"k: {k}")
            print(f"expected_m_per_group: {expected_m_per_group}")
        else:
            print("Pattern not found.")
    elif "DenseGemm" in file:
        # print("DenseGemm found int file, ", file)
        pattern = r'm(\d+)_n(\d+)_k(\d+)'
        match = re.search(pattern, file)
        if match:
            num_groups = 1
            expected_m_per_group = 1
            m = int(match.group(1))
            n = int(match.group(2))
            k = int(match.group(3))
            print(f"m: {m}")
            print(f"n: {n}")
            print(f"k: {k}")
        else:
            print("Pattern not found.")
    else:
        print("GemmType not supported.")
    return num_groups, m, n, k, expected_m_per_group

def convert_data_type_to_dtype(data_type):
    if data_type in [torch.bfloat16, torch.int8, torch.float8_e4m3fn]:
        return data_type
    elif data_type in ["bf16", "torch.bfloat16"]:
        return torch.bfloat16
    elif data_type in ["fp32", "tf32", "torch.float32"]:
        return torch.float32
    elif data_type in ["int8", "torch.int8"]:
        return torch.int8
    elif data_type in ["fp4"]:
        return torch.uint8
    elif data_type in ["fp8", "torch.float8_e4m3fn"]:
        return torch.float8_e4m3fn
    elif data_type in ["w4a16", "w4fa16", "w4fa16_s16", "w4fa16_mma"]:
        return data_type
    else:
        print(f"ERROR: Unsupported dtype: {data_type}, please check!")
        exit(1)

def test_gemm(args) -> None:
    print('Testing GEMM:')
    m, n, k, d = args['m'], args['n'], args['k'], args["data_type"]
    quant_type = args['quant_type'] if 'quant_type' in args else 'block'
    print(f"test_gemm->test_func: DenseGemm,m:{m},n:{n},k:{k},data_type:{d},quant_type:{quant_type}")
    if use_ppu:
        if d == torch.float32:
            x, y, out, out_s, ref_out, ref_s = construct(m, k, n, d, quant_type=quant_type)
        else:
            x, y, out, ref_out = construct(m, k, n, d, quant_type=quant_type)
        if d == torch.bfloat16:
            deep_gemm.gemm_bf16_bf16_bf16_nt(x, y, out)
        elif d == torch.float32:
            deep_gemm.tf32_hc_prenorm_gemm(x, y, out, out_s, num_splits=None)
        elif d == torch.int8:
            deep_gemm.gemm_int8_int8_bf16_nt(x, y, out)
        elif d == torch.float8_e4m3fn:
            deep_gemm.gemm_fp8_fp8_bf16_nt(x, y, out)
        elif d == torch.uint8:
            deep_gemm.gemm_fp4_fp4_bf16_nt(x, y, None, out)
        else:
            print("ERROR: Unsupported dtype, please check!")
            exit(1)
    else:
        kernel_type = get_kernel_types(d)
        out_type = torch.bfloat16
        accumulate = False
        if d == torch.bfloat16:
            x, y, c, out, ref_out = generate_normal(m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor, accumulate, out_type, kernel_type, False, use_bf16=True)
            deep_gemm.bf16_gemm_nt(x, y, out, c=c)
        elif d == torch.float8_e4m3fn:
            use_ue8m0 = get_ue8m0_usage(kernel_type)
            disable_ue8m0_cast = not use_ue8m0
            recipe = (1, 1, 128) if kernel_type.is_1d1d() and accumulate else None
            x, y, c, out, ref_out = generate_normal(m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor, accumulate, out_type, kernel_type, use_ue8m0=use_ue8m0)
            deep_gemm.fp8_gemm_nt(x, y, out, c=c, disable_ue8m0_cast=disable_ue8m0_cast, recipe=recipe)
        else:
            print("ERROR: Unsupported dtype, please check!")
            exit(1)
    if _acc_check:
        tf32_threshold = 1e-8 if d == torch.float32 else 0.001
        if d == torch.float32:
            diff_s = calc_diff(out_s, ref_s)
            if diff_s >= tf32_threshold:
                print("ref_out_s:", ref_s)
                print("out_s:", out_s)
            assert diff_s < tf32_threshold, f'{m=}, {k=}, {n=}, {diff_s:.10f}'
        diff = calc_diff(out, ref_out)
        if diff >= tf32_threshold:
            print("ref_out:", ref_out)
            print("out:", out)
        assert diff < tf32_threshold, f'{m=}, {k=}, {n=}, {diff:.10f}'
        print("Passed with acc_check\n")
    else:
        print("Passed without acc_check\n")

    if get_benchmark():
        # noinspection PyShadowingNames
        def test_func():
            if d == torch.bfloat16:
                deep_gemm.gemm_bf16_bf16_bf16_nt(x, y, out)
            elif d == torch.int8:
                deep_gemm.gemm_int8_int8_bf16_nt(x, y, out)
            elif d == torch.float8_e4m3fn:
                deep_gemm.gemm_fp8_fp8_bf16_nt(x, y, out)
            elif d == torch.uint8:
                deep_gemm.gemm_fp4_fp4_bf16_nt(x, y, None, out)
            else:
                print("ERROR: Unsupported dtype, please check!")
                exit(1)
        t = bench_kineto(test_func, 'gemm', suppress_kineto_output=True)
        print(f' > Performance (dtype={str(d)}, m={m:5}, n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
            f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
            f'{(m * k + k * n + m * n * 2) / 1e9 / t:4.0f} GB/s')
    return

def test_m_grouped_gemm_contiguous(args) -> None:
    print('Testing grouped contiguous GEMM:')
    num_groups, m, n, k, d, distribution = args['groups'], args['m'], args['n'], args['k'], args['data_type'], args['distribution']
    expected_m_per_group = ceil_div(m, num_groups) if "em" not in args.keys() else args["em"]
    quant_type = args['quant_type'] if 'quant_type' in args else 'block'
    if d == torch.uint8:
        print("ERROR: fp4 does not support GroupedContiguous, please use GroupedNoPad or GroupedMasked instead!")
        exit(1)
    if use_ppu:
        m, x, y, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, m, k, n, d, distribution, get_m_alignment_for_contiguous_layout(),quant_type=quant_type)
        print(f"test_m_grouped_gemm_contiguous->test_func: GroupedContiguous,groups:{num_groups},m:{m},n:{n},k:{k},data_type:{d},quant_type:{quant_type}")
        if d == torch.bfloat16:
            deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_contiguous(x, y, out, m_indices)
        elif d == torch.int8:
            deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_contiguous(x, y, out, m_indices)
        else:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x, y, out, m_indices)
    else:
        if d == torch.bfloat16:
            m, x, y, m_indices, out, ref_out = generate_m_grouped_contiguous(num_groups, m, n, k, distribution, get_m_alignment_for_contiguous_layout(), MajorTypeAB.KMajor, MajorTypeAB.KMajor, use_bf16=True)
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(x, y, out, m_indices)
        elif d == torch.float8_e4m3fn:
            kernel_type = get_kernel_types(d)
            use_ue8m0 = get_ue8m0_usage(kernel_type)
            disable_ue8m0_cast = not use_ue8m0
            m, x, y, m_indices, out, ref_out = generate_m_grouped_contiguous(num_groups, m, n, k, distribution, get_m_alignment_for_contiguous_layout(), MajorTypeAB.KMajor, MajorTypeAB.KMajor, use_ue8m0=use_ue8m0)
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(x, y, out, m_indices, disable_ue8m0_cast=disable_ue8m0_cast)
        else:
          print("ERROR: Unsupported dtype, please check!")
          exit(1)

    if _acc_check:
        out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(out), out)
        diff = calc_diff(out, ref_out)
        if diff >= 0.001:
            print("ref_out:", ref_out)
            print("out:", out)
            torch.testing.assert_close(out, ref_out, rtol=2e-1, atol=1)
        assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'
        print("Passed with acc_check\n")
    else:
        print("Passed without acc_check\n")

    if get_benchmark():
        # noinspection PyShadowingNames
        def test_func():
            if d == torch.bfloat16:
                deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_contiguous(x, y, out, m_indices)
            elif d == torch.int8:
                deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_contiguous(x, y, out, m_indices)
            elif d == torch.float8_e4m3fn:
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x, y, out, m_indices)
            else:
                print("ERROR: Unsupported dtype, please check!")
                exit(1)
        t = bench_kineto(test_func, 'gemm', suppress_kineto_output=True)
        valid_m = (m_indices != -1).sum().item()
        print(f' > Perf ((contiguous dtype={str(d)}, {num_groups=:2}, {ceil_div(m, num_groups)=:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
        f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
        f'{(valid_m * k + num_groups * k * n + m * n * 2) / 1e9 / t:4.0f} GB/s')
    return

def estimate_expected_m(max_m, group_size, m_indices):
    assert group_size != 0
    estimate = round_up(int(max_m // group_size), 16)
    max_m = round_up(min(max(m_indices.tolist()), max_m), 16)
    # clamp estimate
    estimate = max(estimate, 16)
    estimate = min(max_m, estimate)
    return estimate

def test_m_grouped_gemm_masked(args) -> None:
    print('Testing grouped masked GEMM:')

    num_groups, m, n, k, d, distribution = args["groups"], args['m'], args['n'], args['k'], args['data_type'], args['distribution']
    enable_sbo_overlap = args['enable_sbo_overlap'] if 'enable_sbo_overlap' in args else False
    epilogue_type = args.get("epilogue_type", "Default")
    swiglu_limit = args.get("swiglu_limit", 0.0)

    if isinstance(enable_sbo_overlap, str):
        enable_sbo_overlap = enable_sbo_overlap.lower() == 'true'

    quant_type = args['quant_type'] if 'quant_type' in args else 'block'
    group_size = args.get('group_size', 32)
    if use_ppu:
        expected_m_per_group = ceil_div(m, num_groups)
        x, y, masked_m, out, ref_out, signal, max_m = construct_grouped_masked(num_groups, m, expected_m_per_group, k, n, d, distribution, enable_sbo_overlap=enable_sbo_overlap, quant_type=quant_type, group_size=group_size, epilogue_type=epilogue_type, swiglu_limit=swiglu_limit)

        expected_m_per_group = estimate_expected_m(m, num_groups, masked_m) if "em" not in args.keys() else args["em"]
        print(f"test_m_grouped_gemm_masked->test_func: GroupedMasked,groups:{num_groups},m:{m},n:{n},k:{k},data_type:{d},em:{expected_m_per_group},max_m:{max_m},distribution:{masked_m},sbo_overlap:{enable_sbo_overlap}")

        if d == torch.bfloat16:
            result = deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_masked(x, y, out, masked_m, expected_m_per_group,
                                                                      enable_sbo_overlap=enable_sbo_overlap, signal=signal)
        elif d == torch.int8:
            result = deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_masked(x, y, out, masked_m, expected_m_per_group,
                                                                      enable_sbo_overlap=enable_sbo_overlap, signal=signal)
        elif d == torch.float8_e4m3fn:
            result = deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x, y, out, masked_m, expected_m_per_group,
                                                                    enable_sbo_overlap=enable_sbo_overlap, signal=signal)
        elif d == torch.uint8:
            if epilogue_type in ["SiluAndMulPostQuantFp4"]:
                out, out_scale = out
                ref_out, ref_out_scale = ref_out
                result = deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(x, y, None, out, masked_m, expected_m_per_group,
                                                                    enable_sbo_overlap=enable_sbo_overlap, signal=signal,
                                                                    out_scale=out_scale, swiglu_limit=swiglu_limit)
            else:
                # `Default` epilogue
                result = deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(x, y, None, out, masked_m, expected_m_per_group,
                                                                    enable_sbo_overlap=enable_sbo_overlap, signal=signal)
        elif d in ('w4a16', 'w4fa16', 'w4fa16_s16', 'w4fa16_mma'):
            deep_gemm.m_grouped_gemm_w4a16_masked(x, y, out, masked_m, expected_m_per_group, fp4_use_bf16_scale=(d == 'w4fa16_s16'))
        else:
            print("ERROR: Unsupported dtype, please check!")
            exit(1)
    else:
        expected_m_per_group = ceil_div(m, num_groups) if "em" not in args.keys() else args["em"]
        if d == torch.bfloat16:
            x, y, masked_m, out, ref_out = generate_m_grouped_masked(num_groups, m, n, k, distribution, expected_m_per_group, use_bf16=True)
            deep_gemm.m_grouped_bf16_gemm_nt_masked(x, y, out, masked_m, expected_m_per_group)
        elif d == torch.float8_e4m3fn:
            kernel_type = get_kernel_types(d)
            use_ue8m0 = get_ue8m0_usage(kernel_type)
            disable_ue8m0_cast = not use_ue8m0
            x, y, masked_m, out, ref_out = generate_m_grouped_masked(num_groups, m, n, k, distribution, expected_m_per_group, use_ue8m0=use_ue8m0)
            deep_gemm.m_grouped_fp8_gemm_nt_masked(x, y, out, masked_m, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast)
        else:
            print("ERROR: Unsupported dtype, please check!")
            exit(1)
    if _acc_check:
        if enable_sbo_overlap:
            block_m, threshold = result
            check_signal(num_groups, max_m, block_m, threshold, signal, masked_m)

        diff, diff_scale = None, None
        for j in range(num_groups):
            diff = calc_diff(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()])
            if (masked_m[j] != 0):
                if diff >= 0.001:
                    print(f"ref_out[{j}]:", ref_out[j, :masked_m[j].item()])
                    print(f"out[{j}]:", out[j, :masked_m[j].item()])
                    # torch.testing.assert_close(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()], rtol=5e-1, atol=2)
                assert diff < 0.001, f'{expected_m_per_group=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'
            if epilogue_type in ["SiluAndMulPostQuantFp4"]:
                diff_scale = calc_diff(out_scale[j, :masked_m[j].item()], ref_out_scale[j, :masked_m[j].item()])
                if (masked_m[j] != 0):
                    if diff_scale >= 0.00001:
                        print(f"ref_out_scale[{j}]:", ref_out_scale[j, :masked_m[j].item()])
                        print(f"out_scale[{j}]:", out_scale[j, :masked_m[j].item()])
                    assert diff_scale < 0.00001, f'{expected_m_per_group=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff_scale:.5f}'
        print(f"Passed with acc_check. {diff=}, {diff_scale=}\n")
    else:
        print("Passed without acc_check\n")
    if get_benchmark():
        # noinspection PyShadowingNames
        def test_func():
            if d == torch.bfloat16:
                result = deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_masked(x, y, out, masked_m, expected_m_per_group,
                                                                        enable_sbo_overlap=enable_sbo_overlap, signal=signal)
            elif d == torch.int8:
                result = deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_masked(x, y, out, masked_m, expected_m_per_group,
                                                                        enable_sbo_overlap=enable_sbo_overlap, signal=signal)
            elif d == torch.float8_e4m3fn:
                result = deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x, y, out, masked_m, expected_m_per_group,
                                                                        enable_sbo_overlap=enable_sbo_overlap, signal=signal)
            elif d == torch.uint8:
                result = deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(x, y, None, out, masked_m, expected_m_per_group,
                                                                        enable_sbo_overlap=enable_sbo_overlap, signal=signal)
            else:
                print("ERROR: Unsupported dtype, please check!")
                exit(1)
        # Test performance with fixed shapes
        # noinspection PyUnboundLocalVariable
        valid_m = masked_m.sum().item()
        t = bench_kineto(test_func, 'gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
            f'throughput: {2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS, '
            f'{(valid_m * k + num_groups * k * n + valid_m * n * 2) / 1e9 / t:4.0f} GB/s')
    return

def test_m_grouped_gemm_fused(args) -> None:
    print('Testing grouped fuse permute GEMM:')
    num_groups, n, k, d = args['groups'], args['n'], args['k'], args['data_type']
    num_token, topk = args['num_token'], args['topk']
    quant_type = args.get('quant_type')
    group_size = args.get('group_size', 32)
    epilogue_type = args.get("epilogue_type", "Default")
    swiglu_limit = args.get("swiglu_limit", 0.0)
    if use_ppu:
        x, y, topk_ids, out, ref_out = construct_non_permute_grouped(num_groups, num_token, k, n, topk, d, quant_type, group_size, epilogue_type=epilogue_type, swiglu_limit=swiglu_limit)
        # Extract main tensors for moe_align (may be tuples for quantized inputs)
        x_tensor = x[0] if isinstance(x, (tuple, list)) else x
        y_tensor = y[0] if isinstance(y, (tuple, list)) else y
        is_perchannel = quant_type == 'channel'
        configs, m_rows, expert_ids_and_offset, sorted_token_ids, aligned_num_m_blocks, _, _ = deep_gemm.moe_align_block_size(x_tensor, y_tensor, topk_ids, is_perchannel, enable_act_and_quant_fusing=(epilogue_type == "SiluAndMulPostQuantFp4"))

        if d == torch.bfloat16:
            deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_fused(x, y, out, m_rows, expert_ids_and_offset, sorted_token_ids, aligned_num_m_blocks, configs)
        elif d == torch.int8:
            deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_fused(x, y, out, m_rows, expert_ids_and_offset, sorted_token_ids, aligned_num_m_blocks, configs)
        elif d == torch.float8_e4m3fn:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_fused(x, y, out, m_rows, expert_ids_and_offset, sorted_token_ids, aligned_num_m_blocks, configs)
        elif d == torch.uint8:
            if epilogue_type in ["SiluAndMulPostQuantFp4"]:
                out, out_scale = out
                ref_out, ref_out_scale = ref_out
                deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_fused(x, y, out, m_rows, expert_ids_and_offset, sorted_token_ids, aligned_num_m_blocks, configs, out_scale=out_scale, swiglu_limit=swiglu_limit)
            else:
                deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_fused(x, y, out, m_rows, expert_ids_and_offset, sorted_token_ids, aligned_num_m_blocks, configs)
        elif d in ('w4a16', 'w4fa16', 'w4fa16_s16', 'w4fa16_mma'):
            deep_gemm.m_grouped_gemm_w4a16_fused(x, y, out, m_rows, expert_ids_and_offset, sorted_token_ids, aligned_num_m_blocks, configs, fp4_use_bf16_scale=(d == 'w4fa16_s16'))
        else:
            print("ERROR: Unsupported dtype, please check!")
            exit(1)
    else:
        print("ERROR: unsupport m_grouped_gemm_fused!")
        exit(1)
    if _acc_check:
        diff = calc_diff(out, ref_out)
        diff_scale = None
        if diff >= 0.0015 or torch.isnan(diff) or torch.isinf(diff):
            # torch.set_printoptions(threshold=10000000, linewidth=10000, precision=2, sci_mode=False)
            print("ref_out:", ref_out)
            print("out:", out)
            torch.testing.assert_close(out, ref_out, rtol=5e-1, atol=2)
        assert diff < 0.0015, f'{num_token=}, {k=}, {n=}, {diff:.5f}'
        if epilogue_type in ["SiluAndMulPostQuantFp4"]:
            diff_scale = calc_diff(out_scale, ref_out_scale)
            if diff_scale >= 0.00001:
                print("ref_out_scale:", ref_out_scale)
                print("out_scale:", out_scale)
            assert diff_scale < 0.00001, f'{num_token=}, {k=}, {n=}, {diff_scale:.5f}'
        print(f"Passed with acc_check, {diff=}, {diff_scale=}\n")
    else:
        print("Passed without acc_check\n")

def test_m_grouped_gemm_nopad(args) -> None:
    print('Testing grouped nopad GEMM:')
    num_groups, m, n, k, d, distribution = args['groups'], args['m'], args['n'], args['k'], args['data_type'], args['distribution']
    UT = f"test_m_grouped_gemm_nopad->test_func: GroupedNoPad,data_type:{d},groups:{num_groups},m:{m},n:{n},k:{k}"
    quant_type = args.get('quant_type', 'block')
    group_size = args.get('group_size', 32)
    if quant_type in args: UT += f",quant_type:{quant_type}"
    if group_size in args: UT += f",group_size:{group_size}"
    epilogue_type = args.get("epilogue_type", "Default")
    swiglu_limit = args.get("swiglu_limit", 0.0)
    print(UT)
    if use_ppu:
        m, x, y, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, m, k, n, d, distribution, 1, quant_type=quant_type, group_size=group_size, epilogue_type=epilogue_type, swiglu_limit=swiglu_limit)
        # topk = 8
        # num_token = m // topk
        # x, y, m_indices, out, ref_out = construct_non_permute_grouped(num_groups, num_token, k, n, topk, d, quant_type, group_size, True)
        if d == torch.bfloat16:
            deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad(x, y, out, m_indices)
        elif d == torch.int8:
            deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_nopad(x, y, out, m_indices)
        elif d == torch.float8_e4m3fn:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_nopad(x, y, out, m_indices)
        elif d == torch.uint8:
            if epilogue_type in ["SiluAndMulPostQuantFp4"]:
                out, out_scale = out
                ref_out, ref_out_scale = ref_out
                deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad(x, y, None, out, m_indices, out_scale=out_scale, swiglu_limit=swiglu_limit)
            else:
                deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad(x, y, None, out, m_indices)
        elif d in ('w4a16', 'w4fa16', 'w4fa16_s16', 'w4fa16_mma'):
            deep_gemm.m_grouped_gemm_w4a16_nopad(x, y, out, m_indices, fp4_use_bf16_scale=(d == 'w4fa16_s16'))
        else:
            print("ERROR: Unsupported dtype, please check!")
            exit(1)
    else:
        print("ERROR: unsupport m_grouped_gemm_nopad!")
        exit(1)

    if _acc_check:
        # out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(out), out)
        diff = calc_diff(out, ref_out)
        diff_scale = None
        if diff >= 0.0015:
            print("ref_out:", ref_out)
            print("out:", out)
            torch.testing.assert_close(out, ref_out, rtol=5e-1, atol=2)

        assert diff < 0.0015, f'{m=}, {k=}, {n=}, {diff:.5f}'
        if epilogue_type in ["SiluAndMulPostQuantFp4"]:
            diff_scale = calc_diff(out_scale, ref_out_scale)
            if diff_scale >= 0.00001:
                print("ref_out_scale:", ref_out_scale)
                print("out_scale:", out_scale)
            assert diff_scale < 0.00001, f'{m=}, {k=}, {n=}, {diff_scale:.5f}'
        print(f"Passed with acc_check, {diff=}, {diff_scale=}\n")
    else:
        print("Passed without acc_check\n")
    if get_benchmark():
        # noinspection PyShadowingNames
        def test_func():
            if d == torch.bfloat16:
                deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad(x, y, out, m_indices)
            elif d == torch.int8:
                deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_nopad(x, y, out, m_indices)
            elif d == torch.float8_e4m3fn:
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_nopad(x, y, out, m_indices)
            elif d == torch.uint8:
                deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad(x, y, None, out, m_indices)
            else:
                print("ERROR: Unsupported dtype, please check!")
                exit(1)
        t = bench_kineto(test_func, 'gemm', suppress_kineto_output=True)
        valid_m = (m_indices != -1).sum().item()
        print(f' > Perf ((contiguous dtype={str(d)}, {num_groups=:2}, {ceil_div(m, num_groups)=:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
        f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
        f'{(valid_m * k + num_groups * k * n + m * n * 2) / 1e9 / t:4.0f} GB/s')

    return

def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_fp8 = torch.empty((num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8)
    x_fp8[ :, : block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(dtype=torch.uint8)
    x_fp8[ :, block_size * head_dim :] = sf.view(num_blocks, block_size).view(dtype=torch.uint8)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)

def kv_cache_cast_to_int8(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 127.0
    x_scaled = (x * (1.0 / sf)).to(torch.int8)
    x_int8 = torch.empty((num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8)
    x_int8[:, : block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(dtype=torch.uint8)
    x_int8[:, block_size * head_dim:] = sf.view(num_blocks, block_size).view(dtype=torch.uint8)
    return x_int8.view(num_blocks, block_size, num_heads, head_dim + 4)

def kv_cache_cast_to_fp4(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1 and head_dim == 128
    x_scaled, sf = per_token_cast_to_fp4(x.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    x_cast_back = cast_back_from_fp4(x_scaled, sf, gran_k=32, use_packed_ue8m0=True).view(num_blocks, block_size, 1, head_dim)

    x_fp4 = torch.empty((num_blocks, block_size * (head_dim // 2 + 4)), device=x.device, dtype=torch.uint8)
    x_fp4[ :, : block_size * head_dim // 2] = x_scaled.view(num_blocks, block_size * head_dim // 2).view(torch.uint8)
    x_fp4[ :, block_size * head_dim // 2 :] = sf.view(num_blocks, block_size).view(torch.uint8)
    return x_fp4.view(num_blocks, block_size, num_heads, head_dim // 2 + 4), x_cast_back.to(x.dtype)

def generate_cp_test_data(seq_len, seq_len_kv):
    assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
    chunk_size = seq_len // 2
    cp_size = seq_len_kv // seq_len
    # Select an arbitrary CP rank
    cp_id = cp_size // 3
    ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
    ke = torch.zeros(seq_len, dtype=torch.int, device='cuda')
    for i in range(chunk_size):
        ke[i] = cp_id * chunk_size + i
        ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    return ks, ke

def ref_get_metadata(context_lens: torch.Tensor, block_kv: int, num_sms: int, metadata_extra: tuple = None):
    split_kv = block_kv * 4  # fallback: assume max num_math_warpgroups=4 to avoid crash
    tb_per_cu = 1
    if metadata_extra is not None:
        from deep_gemm.jit_kernels.utils import get_paged_mqa_logits_tile
        next_n, num_heads, head_dim, element_size = metadata_extra
        _, _, split_kv, _, _, tb_per_cu = get_paged_mqa_logits_tile(next_n, block_kv, num_heads, head_dim, element_size)
    num_splits = num_sms * tb_per_cu

    context_lens = context_lens[:, -1] # convert to 1D context lens, use last token's context_len
    batch_size = context_lens.size(0)
    blocksize = split_kv
    block_len = (context_lens + blocksize - 1) // blocksize
    block_start = torch.cumsum(block_len, dim=0)
    total_blocks = block_start[-1].item()
    block_count = total_blocks // num_splits
    extra_blocks = total_blocks - block_count * num_splits

    schedule_meta_data = torch.zeros((num_splits + 1, 2), dtype=torch.int32, device=context_lens.device)
    for i in range(num_splits + 1):
        seg_starts = i * block_count + min(i, extra_blocks)
        q_idx = 0
        while q_idx < batch_size and block_start[q_idx] <= seg_starts:
            q_idx += 1
        kv_idx = seg_starts if q_idx == 0 else seg_starts - block_start[q_idx - 1]
        schedule_meta_data[i, 0] = q_idx
        schedule_meta_data[i, 1] = kv_idx
    return schedule_meta_data

def ref_fp8_mqa_logits(q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor,
                       cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor, cost_only: bool = False,
                       avg: bool = False):
    seq_len_kv = kv.shape[0]

    if cost_only:
        start = cu_seqlen_ks.clamp(min=0, max=seq_len_kv)
        end   = cu_seqlen_ke.clamp(min=0, max=seq_len_kv)
        count_ones_per_row = (end - start).clamp(min=0)
        return count_ones_per_row.sum()

    k = kv
    q = q.float()
    k = k.float()

    mask_lo = torch.arange(0, seq_len_kv, device='cuda')[None, :] >= cu_seqlen_ks[:, None]
    mask_hi = torch.arange(0, seq_len_kv, device='cuda')[None, :] < cu_seqlen_ke[:, None]
    mask = mask_lo & mask_hi

    score = torch.einsum('mhd,nd->hmn', q, k)
    if avg:
        # Avg variant: unweighted head sum scaled by 1/sqrt(head_dim), weights unused
        logits = score.relu().sum(dim=0) * (q.size(-1) ** -0.5)
    else:
        logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float('-inf'))

    cost = mask.sum()
    return logits, cost

def ref_fp8_paged_mqa_logits(q: torch.Tensor, kv_cache: torch.Tensor,
                             weights: torch.Tensor, context_lens: torch.Tensor, block_tables: torch.Tensor,
                             max_model_len: int, avg: bool = False):
    batch_size, next_n, heads, dim = q.size()
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full([batch_size * next_n, max_model_len], float('-inf'), device=q.device, dtype=torch.float32)
    # 2D context_lens: use last token's context_len (maximum) as KV range, matching kernel behavior
    context_lens_max = context_lens[:, -1].tolist()
    for i in range(batch_size):
        context_len = context_lens_max[i]
        # All tokens see the same KV range (up to max context_len)
        # Per-token masking is done externally
        q_offsets = torch.full((next_n, ), context_len, device='cuda')
        if not avg:
            weight_slice = weights[i * next_n:(i + 1) * next_n, :].transpose(0, 1).contiguous()
        for block_rk in range(ceil_div(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(block_rk * block_size, (block_rk + 1) * block_size, device='cuda')
            mask = (k_offsets[None, :] < context_len) & (k_offsets[None, :] <= q_offsets[:, None])
            s = torch.where(mask[None, :, :], (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(logits.dtype), float('-inf'))
            if avg:
                # Avg variant: unweighted head sum scaled by 1/sqrt(head_dim), weights unused
                s = torch.relu(s).sum(dim=0) * (dim ** -0.5)
            else:
                s = torch.relu(s) * weight_slice[..., None]
                s = s.sum(dim=0)
            logits[i * next_n:(i + 1) * next_n, block_rk * block_size: (block_rk + 1) * block_size] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float('-inf'))
    return logits

def test_mqa_logits(args) -> None:
    print('Testing MQA Logits:')
    data_type = args['data_type']
    seq_len_q = args['seq_len_q']
    seq_len_kv = args['seq_len_kv']
    num_heads = args.get('num_heads', 64)
    head_dim = args.get('head_dim', 128)
    logits_dtype = args.get('logits_dtype', torch.float32)
    weights_dtype = args.get('weights_dtype', torch.float32)
    c4_compressed = bool(args.get('c4_compressed', 0))
    compressed_logits = bool(args.get('compressed_logits', 0))
    # Avg variant (gemm_type 'MqaAvgLogits'): no weights input, /sqrt(head_dim) inside kernel;
    # q_scale: 0 = none, 1 = per-Q-row vector, 2 = broadcast scalar
    is_avg = args.get('gemm_type') == 'MqaAvgLogits'
    q_scale_mode = args.get('q_scale', 0) if is_avg else 0
    assert q_scale_mode in (0, 1, 2), "q_scale must be 0 (none), 1 (per-row) or 2 (scalar)"

    print("test_mqa_logits->test_func: {},data_type:{},seq_len_q:{},seq_len_kv:{},num_heads:{},head_dim:{},c4_compressed:{},compressed_logits:{},weights_dtype:{},logits_dtype:{},q_scale:{}".format(args.get('gemm_type', 'MqaLogits'), data_type, seq_len_q, seq_len_kv, num_heads, head_dim, c4_compressed, compressed_logits, weights_dtype, logits_dtype, q_scale_mode))

    if is_avg:
        assert data_type == torch.float8_e4m3fn, "MQA Avg Logits supports fp8 only"
        assert logits_dtype in (torch.float32, torch.bfloat16), "MQA Avg Logits supports float32 or bf16 logits"

    q = torch.randn(seq_len_q, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv, head_dim, device='cuda', dtype=torch.bfloat16)
    weights = torch.randn(seq_len_q, num_heads, device='cuda', dtype=weights_dtype)

    ks = torch.zeros(seq_len_q, dtype=torch.int, device='cuda')
    if c4_compressed: # c4_compressed for deepseek v4
        ke = torch.arange(seq_len_q, dtype=torch.int, device='cuda') // 4 + (seq_len_kv - seq_len_q // 4)
    else:
        ke = torch.arange(seq_len_q, dtype=torch.int, device='cuda') + (seq_len_kv - seq_len_q)

    max_seqlen_k = (ke - ks).max().item() if compressed_logits else 0
    clean_logits_arg = not compressed_logits
    q_scale = None

    if data_type == torch.bfloat16:
        logits = deep_gemm.bf16_mqa_logits(q, kv, weights, ks, ke,
                                           clean_logits=clean_logits_arg, max_seqlen_k=max_seqlen_k,
                                           logits_dtype=logits_dtype)
    elif data_type == torch.float8_e4m3fn:
        q_fp8 = q.to(torch.float8_e4m3fn)
        kv_fp8 = per_custom_dims_cast_to_fp8(kv, (0, ), False)
        if is_avg:
            if q_scale_mode == 1:
                q_scale = torch.rand(seq_len_q, device='cuda', dtype=torch.float32) + 0.5
            elif q_scale_mode == 2:
                q_scale = torch.tensor(2.5, device='cuda', dtype=torch.float32)
            logits = deep_gemm.fp8_mqa_avg_logits(q_fp8, kv_fp8, ks, ke,
                                                  clean_logits=clean_logits_arg, max_seqlen_k=max_seqlen_k,
                                                  q_scale=q_scale, logits_dtype=logits_dtype)
        else:
            logits = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke,
                                              clean_logits=clean_logits_arg, max_seqlen_k=max_seqlen_k,
                                              logits_dtype=logits_dtype)
    elif data_type == torch.int8:
        q_int8, q_int8_scale = per_token_cast_to_int8(q.reshape(seq_len_q*num_heads, head_dim))
        q_int8 = q_int8.reshape(seq_len_q, num_heads, head_dim)
        q_int8_scale = q_int8_scale.reshape(seq_len_q, num_heads)
        weights_int8 = weights * q_int8_scale.to(weights.dtype)
        kv_int8 = per_token_cast_to_int8(kv)
        logits = deep_gemm.int8_mqa_logits(q_int8, kv_int8, weights_int8, ks, ke,
                                           clean_logits=clean_logits_arg, max_seqlen_k=max_seqlen_k,
                                           logits_dtype=logits_dtype)
    elif data_type == torch.uint8:  # FP4
        q_fp4 = per_token_cast_to_fp4(q.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        q_in = (q_fp4[0].view(seq_len_q, num_heads, head_dim // 2), q_fp4[1].view(seq_len_q, num_heads))
        q_dequant = cast_back_from_fp4(q_fp4[0], q_fp4[1], gran_k=32, use_packed_ue8m0=True).view(seq_len_q, num_heads, head_dim).to(torch.bfloat16)
        kv_fp4 = per_token_cast_to_fp4(kv, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        kv_in = (kv_fp4[0], kv_fp4[1].squeeze(-1))
        kv_dequant = cast_back_from_fp4(kv_fp4[0], kv_fp4[1], gran_k=32, use_packed_ue8m0=True).view(seq_len_kv, head_dim).to(torch.bfloat16)
        q, kv = q_dequant, kv_dequant
        logits = deep_gemm.fp8_fp4_mqa_logits(
            q=q_in, kv=kv_in, weights=weights,
            cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
            clean_logits=clean_logits_arg, max_seqlen_k=max_seqlen_k,
            logits_dtype=logits_dtype,
        )
    else:
        print("ERROR: Unsupported dtype for MQA Logits, please check!")
        exit(1)

    # Compute reference
    ref_logits = None
    if get_acc_check():
        assert get_ref_backend() == "device", "ref_backend only supports 'device' for MQA Logits"
        ref_logits, _ = ref_fp8_mqa_logits(q=q, kv=kv, weights=weights.float(), cu_seqlen_ks=ks, cu_seqlen_ke=ke, avg=is_avg)
        # q_scale is an extra multiplier applied inside the kernel (per-row vector or broadcast scalar)
        if is_avg and q_scale is not None:
            ref_logits = ref_logits * (q_scale if q_scale.numel() == 1 else q_scale.view(-1, 1))

    # Accuracy check
    if get_acc_check() and ref_logits is not None:
        # Unpack compressed logits to full [seq_len_q, seq_len_kv] for comparison
        if compressed_logits:
            full_logits = torch.full((seq_len_q, seq_len_kv), float('-inf'), device='cuda', dtype=logits.dtype)
            for i in range(seq_len_q):
                valid_len = (ke[i] - ks[i]).item()
                if valid_len > 0:
                    full_logits[i, ks[i]:ke[i]] = logits[i, :valid_len]
            logits = full_logits

        neginf_mask = (logits == float('-inf'))
        ref_neginf_mask = (ref_logits == float('-inf'))

        if not torch.equal(neginf_mask, ref_neginf_mask):
            print("ERROR: -inf mask mismatch!")
            exit(1)

        logits_masked = logits.masked_fill(neginf_mask, 0)
        ref_logits_masked = ref_logits.masked_fill(ref_neginf_mask, 0)

        from math_utils import calc_diff
        diff = calc_diff(logits_masked, ref_logits_masked)
        if torch.isnan(diff) or diff >= 1e-3:
            print(f"ERROR: Accuracy check failed, diff={diff}")
            exit(1)
        else:
            print("Accuracy check passed\n")
    return

def test_paged_mqa_logits(args) -> None:
    print('Testing Paged MQA Logits:')
    data_type = args['data_type']
    batch_size = args['batch_size']
    next_n = args['next_n']
    avg_context_len = args.get('avg_context_len', 8192)
    distribution = args.get('distribution', [])
    pre_distribution = args.get('pre_distribution', [])
    num_heads = args.get('num_heads', 64)
    head_dim = args.get('head_dim', 128)
    logits_dtype = args.get('logits_dtype', torch.float32)
    weights_dtype = args.get('weights_dtype', torch.float32)
    flatten_mtp = args.get('flatten_mtp', False)
    # Avg variant (gemm_type 'PagedMqaAvgLogits'): no weights input, /sqrt(head_dim) inside kernel
    # (aligned with PAI: the paged avg kernel takes no q_scale)
    is_avg = args.get('gemm_type') == 'PagedMqaAvgLogits'

    print("test_paged_mqa_logits->test_func: {},data_type:{},batch_size:{},next_n:{},avg_context_len:{},num_heads:{},head_dim:{},logits_dtype:{},weights_dtype:{}".format(args.get('gemm_type', 'PagedMqaLogits'), data_type, batch_size, next_n, avg_context_len, num_heads, head_dim, logits_dtype, weights_dtype))

    if is_avg:
        assert data_type == torch.float8_e4m3fn, "Paged MQA Avg Logits supports fp8 only"
        assert logits_dtype in (torch.float32, torch.bfloat16), "Paged MQA Avg Logits supports float32 or bf16 logits"

    max_model_len = 262144
    blocksize = 64

    q = torch.randn((batch_size, next_n, num_heads, head_dim), device='cuda', dtype=torch.bfloat16)
    weights = torch.randn((batch_size * next_n, num_heads), device='cuda', dtype=weights_dtype)

    # Generate context_lens (2D [batch_size, next_n] for DeepGEMM 2D context_lens support)
    if distribution and isinstance(distribution, list):
        assert(len(distribution) == batch_size)
        context_lens = torch.tensor(distribution[:batch_size], device='cuda', dtype=torch.int32)
    else:
        context_lens = torch.randint(int(0.7 * avg_context_len), int(1.3 * avg_context_len), (batch_size, )).cuda().to(torch.int32)

    max_block_len = (context_lens.max().item() + blocksize - 1) // blocksize
    num_blocks = int(((context_lens + blocksize - 1) // blocksize).sum().item() * 1.3)  # 30% slack
    kv_cache = torch.randn((num_blocks, blocksize, 1, head_dim), device='cuda', dtype=torch.bfloat16)

    block_tables = torch.full((batch_size, max_block_len), fill_value=0, device='cuda', dtype=torch.int32)
    block_idx_pool = torch.randperm(num_blocks, device='cuda', dtype=torch.int32)
    counter = 0
    for i in range(batch_size):
        nblk = ceil_div(context_lens[i].item(), blocksize)
        block_tables[i, :nblk] = block_idx_pool[counter:counter+nblk]
        counter += nblk

    # Generate 2D context_lens for main kernel
    if next_n == 1:
        context_lens = context_lens.unsqueeze(-1)  # [batch_size] -> [batch_size, 1]
    else:
        # Generate per-token context_lens: last column = actual context_len, others decrease by 1
        # e.g. context_lens=[4,5], next_n=3 -> [[2,3,4],[3,4,5]]
        offsets = torch.arange(next_n, device='cuda', dtype=torch.int32)
        context_lens_2d = (context_lens.unsqueeze(1) - (next_n - 1) + offsets).clamp(min=0).to(torch.int32)
        context_lens = context_lens_2d

    if flatten_mtp:
        # Flatten next_n into batch dimension, call subsequent APIs with next_n=1
        q = q.reshape(batch_size * next_n, 1, num_heads, head_dim)
        context_lens = context_lens.flatten().unsqueeze(-1)  # [batch_size * next_n, 1]
        block_tables = torch.repeat_interleave(block_tables, repeats=next_n, dim=0)
        batch_size = batch_size * next_n
        next_n = 1

    # Generate pre_context_lens for get_paged_mqa_logits_metadata (may differ from main kernel context_lens in MTP + CUDA graph)
    if pre_distribution and isinstance(pre_distribution, list):
        assert(next_n == 1 and not flatten_mtp)
        pre_context_lens = torch.tensor(pre_distribution, device='cuda', dtype=torch.int32).unsqueeze(-1)
    else:
        pre_context_lens = context_lens

    get_metadata_kernel = deep_gemm.get_paged_mqa_logits_metadata
    # get_metadata_kernel = ref_get_metadata

    if data_type == torch.bfloat16:
        metadata_extra = (next_n, num_heads, head_dim, q.element_size())
        schedule_metadata = get_metadata_kernel(pre_context_lens, blocksize, deep_gemm.get_num_sms(), metadata_extra=metadata_extra)
        logits = deep_gemm.bf16_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables, schedule_metadata, max_model_len, clean_logits=False, logits_dtype=logits_dtype)
    elif data_type == torch.float8_e4m3fn:
        q_fp8 = q.to(torch.float8_e4m3fn)
        kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache)
        metadata_extra = (next_n, num_heads, head_dim, q_fp8.element_size())
        schedule_metadata = get_metadata_kernel(pre_context_lens, blocksize, deep_gemm.get_num_sms(), metadata_extra=metadata_extra)
        if is_avg:
            logits = deep_gemm.fp8_paged_mqa_avg_logits(q_fp8, kv_cache_fp8, context_lens, block_tables, schedule_metadata, max_model_len, clean_logits=False, logits_dtype=logits_dtype)
        else:
            logits = deep_gemm.fp8_paged_mqa_logits(q_fp8, kv_cache_fp8, weights, context_lens, block_tables, schedule_metadata, max_model_len, clean_logits=False, logits_dtype=logits_dtype)
    elif data_type == torch.int8:
        q_int8, q_int8_scale = per_token_cast_to_int8(q.reshape(batch_size * next_n * num_heads, head_dim))
        q_int8 = q_int8.reshape(batch_size, next_n, num_heads, head_dim)
        q_int8_scale = q_int8_scale.reshape(batch_size * next_n, num_heads)
        weights_int8 = weights * q_int8_scale.to(weights.dtype)
        kv_cache_int8 = kv_cache_cast_to_int8(kv_cache)
        metadata_extra = (next_n, num_heads, head_dim, q_int8.element_size())
        schedule_metadata = get_metadata_kernel(pre_context_lens, blocksize, deep_gemm.get_num_sms(), metadata_extra=metadata_extra)
        logits = deep_gemm.int8_paged_mqa_logits(q_int8, kv_cache_int8, weights_int8, context_lens, block_tables, schedule_metadata, max_model_len, clean_logits=False, logits_dtype=logits_dtype)
    elif data_type == torch.uint8:  # FP4
        q_fp4 = per_token_cast_to_fp4(q.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        q_in = (q_fp4[0].view(batch_size, next_n, num_heads, head_dim // 2), q_fp4[1].view(batch_size, next_n, num_heads))
        q_dequant = cast_back_from_fp4(q_fp4[0], q_fp4[1], gran_k=32, use_packed_ue8m0=True).view(batch_size, next_n, num_heads, head_dim).to(torch.bfloat16)
        kv_cache_fp4, kv_cache_dequant = kv_cache_cast_to_fp4(kv_cache)
        q, kv_cache = q_dequant, kv_cache_dequant
        metadata_extra = (next_n, num_heads, head_dim // 2, q_fp4[0].element_size())
        schedule_metadata = get_metadata_kernel(pre_context_lens, blocksize, deep_gemm.get_num_sms(), metadata_extra=metadata_extra)
        logits = deep_gemm.fp8_fp4_paged_mqa_logits(
            q=q_in, fused_kv_cache=kv_cache_fp4, weights=weights,
            context_lens=context_lens, block_table=block_tables,
            schedule_meta=schedule_metadata, max_context_len=max_model_len,
            clean_logits=False, logits_dtype=logits_dtype,
        )
    else:
        print("ERROR: Unsupported dtype for Paged MQA Logits, please check!")
        exit(1)

    # Compute reference
    ref_logits = None
    if get_acc_check():
        assert get_ref_backend() == "device", "ref_backend only supports 'device' for Paged MQA Logits"
        ref_logits = ref_fp8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables, max_model_len, avg=is_avg)

    # Accuracy check
    if get_acc_check() and ref_logits is not None:
        positions = torch.arange(max_model_len, device='cuda').unsqueeze(0).expand(batch_size * next_n, -1)
        # 2D context_lens: each token has its own context_len for per-token masking
        ref_neginf_mask = ~(positions < context_lens.view(-1, 1))

        neginf_mask = (logits == float('-inf'))
        # assert torch.equal(neginf_mask, ref_neginf_mask)

        logits_masked = logits.masked_fill(ref_neginf_mask, 0)
        ref_logits_masked = ref_logits.masked_fill(ref_neginf_mask, 0)

        from math_utils import calc_diff
        diff = calc_diff(logits_masked, ref_logits_masked)
        threshold = 1.5e-3 if logits_dtype == torch.bfloat16 else 1e-3
        if torch.isnan(diff) or diff >= threshold:
            print(f"ERROR: Accuracy check failed, diff={diff}")
            exit(1)
        else:
            print("Accuracy check passed\n")
    return

def make_sparse_kv_block_indices(context_lens: torch.Tensor, context_starts: torch.Tensor,
                                 sparse_block_kv: int, num_max_sparse_blocks: int, seed: int = 0,
                                 request_indices: list = None) -> Tuple[torch.Tensor, list]:
    """Sparse KV block selection generator for the sparse MQA logits UT (DS indexer simulation).

    Semantics aligned with the open-source 26/09 `test_attention.py::make_sparse_kv_block_indices`:
    valid count = min(num_max_sparse_blocks, ceil_div(len, sparse_block_kv)), block range anchored
    to [ks // sparse_block_kv, ks // sparse_block_kv + num_available); full coverage selects all
    blocks in range (exercises the contiguous fast path); the first token of a request samples
    uniformly, later tokens inherit ~80% of the previous token's blocks (decode-step locality) and
    fill the rest randomly. The row tail is padded with the last block id and never read by kernels.
    Returns (indices int32 [num_q_tokens, num_max_sparse_blocks] on cuda, num_blocks_per_q list).
    """
    rng = random.Random(seed)
    if request_indices is None:
        request_indices = [0] * context_lens.size(0)
    indices, num_blocks_per_q = [], []
    previous_blocks, previous_request_idx = None, None
    for context_end, context_start, request_idx in zip(context_lens.tolist(), context_starts.tolist(),
                                                       request_indices):
        first_block = context_start // sparse_block_kv
        num_available = ceil_div(max(0, context_end - context_start), sparse_block_kv)
        block_end = first_block + num_available
        num_sparse_blocks = min(num_available, num_max_sparse_blocks)
        if num_sparse_blocks == num_available:
            blocks = list(range(first_block, block_end))
        elif request_idx != previous_request_idx:
            blocks = rng.sample(range(first_block, block_end), num_sparse_blocks)
        else:
            previous = [b for b in previous_blocks if first_block <= b < block_end]
            retained = rng.sample(previous, min(round(num_sparse_blocks * 0.8), len(previous)))
            retained_set = set(retained)
            replacements = set()
            while len(retained) + len(replacements) < num_sparse_blocks:
                block_idx = rng.randrange(first_block, block_end)
                if block_idx not in retained_set:
                    replacements.add(block_idx)
            blocks = retained + list(replacements)
        blocks.sort()
        indices.append(blocks + [blocks[-1] if blocks else 0] * (num_max_sparse_blocks - num_sparse_blocks))
        num_blocks_per_q.append(num_sparse_blocks)
        previous_blocks, previous_request_idx = blocks, request_idx
    return torch.tensor(indices, device='cuda', dtype=torch.int32), num_blocks_per_q

def gather_sparse_reference(full_logits: torch.Tensor, sparse_indices: torch.Tensor,
                            num_blocks_per_q: list, context_starts: torch.Tensor,
                            context_lens: torch.Tensor, sparse_block_kv: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather the dense logits (compressed layout: column j == kv token ks + j) at each token's
    selected block positions. Returns (gathered [num_q_tokens, num_max_sparse_blocks *
    sparse_block_kv], valid_mask); the kernels leave non-valid slots unwritten, so only valid_mask
    positions are comparable."""
    kv_offsets = torch.arange(sparse_block_kv, device='cuda', dtype=torch.int64)
    block_offsets = (context_starts % sparse_block_kv).to(torch.int64)
    token_indices = (sparse_indices.long().unsqueeze(-1) * sparse_block_kv
                     + block_offsets[:, None, None] + kv_offsets).flatten(1)
    num_valid_tokens = torch.tensor(num_blocks_per_q, device='cuda',
                                    dtype=torch.int64) * sparse_block_kv
    valid_mask = torch.arange(token_indices.size(1), device='cuda')[None, :] < num_valid_tokens[:, None]
    valid_mask &= token_indices >= context_starts.long()[:, None]
    valid_mask &= token_indices < context_lens.long()[:, None]
    full_token_indices = (token_indices - context_starts.long()[:, None]).clamp(
        min=0, max=full_logits.size(1) - 1)
    return full_logits.gather(1, full_token_indices), valid_mask

def ref_sparse_mqa_logits(q: torch.Tensor, kv_gathered: torch.Tensor, weights: torch.Tensor,
                          valid_mask: torch.Tensor) -> torch.Tensor:
    """FP32 torch reference for sparse MQA logits: logits = sum_h ReLU(q_h . k_v) * weights[h] at
    the gathered KV positions (mirrors `ref_fp8_mqa_logits` epilogue, restricted to the sparse
    selection). `kv_gathered` is [num_q_tokens, W, head_dim] (padded rows arbitrary)."""
    score = torch.einsum('mhd,mwd->hmw', q.float(), kv_gathered.float())
    logits = (score.relu() * weights.float().transpose(0, 1).unsqueeze(-1)).sum(dim=0)
    return logits * valid_mask  # zero out non-valid slots

# ------------------------------------------------------------ sparse metadata reference checks

# Layout mirrors deep_gemm/include/deep_gemm/impls/sparse_mqa_logits_layout.cuh
SPARSE_SPLIT_KV = 128
SPARSE_INVALID_SLOT = 0xFFFF
SPARSE_CONTIGUOUS_FLAG = 0x80000000
# KVBlockInfo slot fields are raw 16-bit offsets (no present bit; the present bit only exists in
# the metadata kernel's shared-memory merge packing, which is never written out)
SPARSE_SLOT_MASK = 0xFFFF

def decode_sparse_metadata(metadata: torch.Tensor, sparse_block_kv: int) -> dict:
    """Decode the metadata buffer produced by the sparse metadata kernel (contiguous layout)."""
    # Schedule table columns = resident CTAs of the sparse main kernel = num_cu x tb_per_cu
    # (tb_per_cu = 4 for the BLOCK_Q=2/BLOCK_KV=128 tile)
    num_slots = 4 * deep_gemm.get_num_sms()
    words = metadata.view(torch.int32).cpu().numpy()
    num_kv_splits, num_waves, use_unaligned_ks = int(words[0]), int(words[1]), int(words[2])
    blocks_per_split = SPARSE_SPLIT_KV // sparse_block_kv
    split_words = 4 + blocks_per_split * 2
    splits = []
    for split_idx in range(num_kv_splits):
        base = 4 + split_idx * split_words
        packed_num = int(words[base + 1]) & 0xFFFFFFFF  # int32 -> unsigned before flag mask
        blocks = []
        for block_idx in range(blocks_per_split):
            w0, w1 = int(words[base + 4 + block_idx * 2]), int(words[base + 5 + block_idx * 2])
            blocks.append((w0, w1 & SPARSE_SLOT_MASK, (w1 >> 16) & SPARSE_SLOT_MASK))
        splits.append(dict(q_token_base=int(words[base]), num_kv_blocks=packed_num & ~SPARSE_CONTIGUOUS_FLAG,
                           is_contiguous=bool(packed_num & SPARSE_CONTIGUOUS_FLAG),
                           q0_slot_base=int(words[base + 2]), q1_slot_base=int(words[base + 3]),
                           blocks=blocks))
    schedule_base = 4 + num_kv_splits * split_words
    schedule = [tuple(int(words[schedule_base + e * 4 + i]) for i in range(4))
                for e in range(num_waves * num_slots)]
    return dict(num_kv_splits=num_kv_splits, num_waves=num_waves, use_unaligned_ks=use_unaligned_ks,
                splits=splits, schedule=schedule)

def ref_sparse_metadata(sparse_indices: torch.Tensor, num_blocks_per_q: list, context_starts: torch.Tensor,
                        context_ends: torch.Tensor, num_kv_tokens: int, sparse_block_kv: int,
                        num_max_sparse_blocks: int, use_unaligned_ks: bool) -> dict:
    """Python reference for the contiguous sparse metadata kernel: per Q block merge the paired
    tokens' scaled index rows (dedup), chunk into KV splits, and reproduce slot bases / offsets /
    physical block ids / contiguous flags. Split *reservation order* is nondeterministic in the
    kernel (atomic), so callers compare via canonical sorting; the schedule is validated
    structurally rather than exactly."""
    blocks_per_split = SPARSE_SPLIT_KV // sparse_block_kv
    indices = sparse_indices.cpu().numpy()
    starts = context_starts.cpu().numpy().tolist()
    ends = context_ends.cpu().numpy().tolist()
    num_q_tokens = len(num_blocks_per_q)
    splits = []
    # Upstream semantics (sm100_sparse_mqa_logits_metadata.cuh): the merge runs in the *logical*
    # space (raw block ids when ks is grid-aligned; raw*sparse_block_kv + ks%sparse_block_kv when
    # unaligned), and physical = logical * (1 if unaligned else sparse_block_kv) — scaled exactly
    # once. The odd tail token forms a Q block with an empty q1 row.
    for q_token_base in range(0, num_q_tokens, 2):
        scaled, counts = [], []
        for q_offset in range(2):
            q_idx = q_token_base + q_offset
            if q_idx >= num_q_tokens:
                counts.append(0)
                scaled.append([])
                continue
            ks = min(starts[q_idx], num_kv_tokens)
            ke = max(ks, min(ends[q_idx], num_kv_tokens))
            num_kv_blocks = min(num_max_sparse_blocks, ceil_div(ke - ks, sparse_block_kv))
            counts.append(num_kv_blocks)
            scaled.append([int(indices[q_idx, slot]) * (sparse_block_kv if use_unaligned_ks else 1) +
                           (ks % sparse_block_kv if use_unaligned_ks else 0)
                           for slot in range(num_kv_blocks)])
        # Global two-way merge with dedup; per-Q slot counters mirror the kernel's merge exactly.
        # (The kernel's `consume_q1` rule reduces to `i1 += in_q1`: an in_q1 step always has input
        # left, so `remaining > in_q0` always holds.)
        merged = []
        i0, i1 = 0, 0
        while i0 < counts[0] or i1 < counts[1]:
            v0 = scaled[0][i0] if i0 < counts[0] else (1 << 32) - 1
            v1 = scaled[1][i1] if i1 < counts[1] else (1 << 32) - 1
            in_q0, in_q1 = v0 <= v1, v1 <= v0
            merged.append((v0 if in_q0 else v1, i0, i1, in_q0, in_q1))
            i0 += int(in_q0)
            i1 += int(in_q1)
        num_splits = ceil_div(len(merged), blocks_per_split)
        for split_offset in range(num_splits):
            blocks_in_split = merged[split_offset * blocks_per_split:(split_offset + 1) * blocks_per_split]
            q0_slot_base, q1_slot_base = blocks_in_split[0][1], blocks_in_split[0][2]
            is_contiguous = False
            if not use_unaligned_ks and len(blocks_in_split) == blocks_per_split:
                logicals = [b[0] for b in blocks_in_split]
                is_contiguous = logicals == list(range(logicals[0], logicals[0] + len(logicals)))
            block_records = []
            for logical, i0, i1, in_q0, in_q1 in blocks_in_split:
                physical = logical * (1 if use_unaligned_ks else sparse_block_kv)
                q0_off = (i0 - q0_slot_base) if in_q0 else SPARSE_INVALID_SLOT
                q1_off = (i1 - q1_slot_base) if in_q1 else SPARSE_INVALID_SLOT
                block_records.append((physical, q0_off, q1_off))
            block_records += [(0, SPARSE_INVALID_SLOT, SPARSE_INVALID_SLOT)] * (blocks_per_split - len(block_records))
            splits.append(dict(q_token_base=q_token_base, num_kv_blocks=len(blocks_in_split),
                               is_contiguous=is_contiguous, q0_slot_base=q0_slot_base,
                               q1_slot_base=q1_slot_base if q_token_base + 1 < num_q_tokens else SPARSE_INVALID_SLOT,
                               blocks=block_records))
    return dict(num_kv_splits=len(splits), splits=splits, use_unaligned_ks=int(use_unaligned_ks))

def verify_sparse_metadata(decoded: dict, ref: dict, total_q_tokens: int) -> None:
    """Compare the kernel's decoded metadata against the Python reference. Split reservation order
    is nondeterministic, so split lists are canonically sorted by (q_token_base, q0_slot_base,
    q1_slot_base); the schedule is validated structurally (exact tiling, slot evenness, Q-block cuts)."""
    num_slots = 4 * deep_gemm.get_num_sms()
    if decoded['num_kv_splits'] != ref['num_kv_splits']:
        print(f"ERROR: metadata num_kv_splits mismatch: {decoded['num_kv_splits']} vs {ref['num_kv_splits']}")
        exit(1)
    if decoded['use_unaligned_ks'] != ref['use_unaligned_ks']:
        print("ERROR: metadata use_unaligned_ks mismatch")
        exit(1)
    key = lambda s: (s['q_token_base'], s['q0_slot_base'], s['q1_slot_base'])
    got = sorted(decoded['splits'], key=key)
    want = sorted(ref['splits'], key=key)
    for g, w in zip(got, want):
        for field in ('q_token_base', 'num_kv_blocks', 'is_contiguous', 'q0_slot_base', 'q1_slot_base'):
            if g[field] != w[field]:
                print(f"ERROR: metadata split field {field} mismatch at {key(g)}: {g[field]} vs {w[field]}")
                exit(1)
        if g['blocks'] != w['blocks']:
            for i, (gb, wb) in enumerate(zip(g['blocks'], w['blocks'])):
                if gb != wb:
                    print(f"ERROR: metadata block {i} mismatch at split {key(g)}: {gb} vs {wb}")
                    exit(1)
            print("ERROR: metadata blocks mismatch (no differing block found?!)")
            exit(1)
    # Schedule: non-empty entries tile [0, num_kv_splits) in slot-major order, cut at Q-block bounds
    total = ref['num_kv_splits']
    num_waves = decoded['num_waves']
    assert num_waves >= 1 and len(decoded['schedule']) == num_waves * num_slots
    covered = []
    for wave in range(num_waves):
        for slot_idx in range(num_slots):
            begin, end, q_token_base, num_q_tokens = decoded['schedule'][wave * num_slots + slot_idx]
            if begin == 0 and end == 0:
                continue
            window_begin = ceil_div(total * slot_idx, num_slots)
            window_end = ceil_div(total * (slot_idx + 1), num_slots)
            covered.append((begin, end))
            if not (window_begin <= begin < end <= window_end):
                print(f"ERROR: schedule entry ({begin},{end}) escapes slot window [{window_begin},{window_end})")
                exit(1)
            if decoded['splits'][begin]['q_token_base'] != q_token_base or \
               decoded['splits'][end - 1]['q_token_base'] != q_token_base:
                print(f"ERROR: schedule entry ({begin},{end}) crosses Q-block boundary (q_token_base={q_token_base})")
                exit(1)
            if num_q_tokens != min(2, total_q_tokens - q_token_base):
                print(f"ERROR: schedule entry ({begin},{end}) num_q_tokens={num_q_tokens} inconsistent with pair at {q_token_base}")
                exit(1)
    covered.sort()
    expect_begin = 0
    for begin, end in covered:
        if begin != expect_begin or end <= begin:
            print(f"ERROR: schedule tiling broken at ({begin},{end}), expected begin {expect_begin}")
            exit(1)
        expect_begin = end
    if expect_begin != total:
        print(f"ERROR: schedule covers [0,{expect_begin}) but total splits = {total}")
        exit(1)

def ref_paged_sparse_metadata(sparse_indices: torch.Tensor, num_blocks_per_q: list,
                              context_lens: torch.Tensor, block_table: torch.Tensor,
                              request_ids: torch.Tensor, page_kv: int, sparse_block_kv: int,
                              num_max_sparse_blocks: int) -> dict:
    """Python reference for the paged sparse metadata kernel. Differences vs contiguous:
    KV window = [0, context_lens[q]) per token (logical values are raw block ids), Q blocks pair
    only tokens of the same request (from the request start), and physical =
    block_table[q_token_base][logical // blocks_per_page] * blocks_per_page + logical % blocks_per_page.
    Split reservation order is nondeterministic (atomic); callers compare via canonical sorting."""
    blocks_per_split = SPARSE_SPLIT_KV // sparse_block_kv
    blocks_per_page = page_kv // sparse_block_kv
    indices = sparse_indices.cpu().numpy()
    lens = context_lens.cpu().numpy().tolist()
    bt = block_table.cpu().numpy()
    req = request_ids.cpu().numpy().tolist()
    num_q_tokens = len(num_blocks_per_q)
    # Per-token pairing: base t pairs with t+1 iff t is request-aligned and t+1 is the same request
    paired = [False] * num_q_tokens
    q_token_bases = []
    for t in range(num_q_tokens):
        r_start = t
        while r_start > 0 and req[r_start - 1] == req[t]:
            r_start -= 1
        if (t - r_start) % 2 != 0:
            continue
        q_token_bases.append(t)
        if t + 1 < num_q_tokens and req[t + 1] == req[t]:
            paired[t] = True
    splits = []
    for q_token_base in q_token_bases:
        counts, rows = [], []
        for q_offset in range(2):
            q_idx = q_token_base + q_offset
            if q_offset == 1 and not paired[q_token_base]:
                counts.append(0)
                rows.append([])
                continue
            num_kv_blocks = min(num_max_sparse_blocks, ceil_div(lens[q_idx], sparse_block_kv))
            counts.append(num_kv_blocks)
            rows.append([int(indices[q_idx, slot]) for slot in range(num_kv_blocks)])
        merged = []
        i0, i1 = 0, 0
        while i0 < counts[0] or i1 < counts[1]:
            v0 = rows[0][i0] if i0 < counts[0] else (1 << 32) - 1
            v1 = rows[1][i1] if i1 < counts[1] else (1 << 32) - 1
            in_q0, in_q1 = v0 <= v1, v1 <= v0
            merged.append((v0 if in_q0 else v1, i0, i1, in_q0, in_q1))
            i0 += int(in_q0)
            i1 += int(in_q1)
        num_splits = ceil_div(len(merged), blocks_per_split)
        for split_offset in range(num_splits):
            blocks_in_split = merged[split_offset * blocks_per_split:(split_offset + 1) * blocks_per_split]
            q0_slot_base, q1_slot_base = blocks_in_split[0][1], blocks_in_split[0][2]
            block_records = []
            for logical, i0, i1, in_q0, in_q1 in blocks_in_split:
                logical_page_idx = logical // blocks_per_page
                physical = int(bt[q_token_base, logical_page_idx]) * blocks_per_page + logical % blocks_per_page
                q0_off = (i0 - q0_slot_base) if in_q0 else SPARSE_INVALID_SLOT
                q1_off = (i1 - q1_slot_base) if in_q1 else SPARSE_INVALID_SLOT
                block_records.append((physical, q0_off, q1_off))
            block_records += [(0, SPARSE_INVALID_SLOT, SPARSE_INVALID_SLOT)] * (blocks_per_split - len(block_records))
            splits.append(dict(q_token_base=q_token_base, num_kv_blocks=len(blocks_in_split),
                               is_contiguous=False, q0_slot_base=q0_slot_base,
                               q1_slot_base=q1_slot_base if paired[q_token_base] else SPARSE_INVALID_SLOT,
                               blocks=block_records))
    return dict(num_kv_splits=len(splits), splits=splits, use_unaligned_ks=0)

def verify_paged_sparse_metadata(decoded: dict, ref: dict, total_q_tokens: int,
                                 request_ids: torch.Tensor) -> None:
    """Paged schedule check: entries tile [0, num_kv_splits) exactly; each entry stays inside its
    Q block's split range with a pairing count matching the request-aligned pairs."""
    num_slots = 4 * deep_gemm.get_num_sms()
    if decoded['num_kv_splits'] != ref['num_kv_splits']:
        print(f"ERROR: metadata num_kv_splits mismatch: {decoded['num_kv_splits']} vs {ref['num_kv_splits']}")
        exit(1)
    key = lambda s: (s['q_token_base'], s['q0_slot_base'], s['q1_slot_base'])
    got = sorted(decoded['splits'], key=key)
    want = sorted(ref['splits'], key=key)
    for g, w in zip(got, want):
        for field in ('q_token_base', 'num_kv_blocks', 'is_contiguous', 'q0_slot_base', 'q1_slot_base'):
            if g[field] != w[field]:
                print(f"ERROR: metadata split field {field} mismatch at {key(g)}: {g[field]} vs {w[field]}")
                exit(1)
        if g['blocks'] != w['blocks']:
            for i, (gb, wb) in enumerate(zip(g['blocks'], w['blocks'])):
                if gb != wb:
                    print(f"ERROR: metadata block {i} mismatch at split {key(g)}: {gb} vs {wb}")
                    exit(1)
            print("ERROR: metadata blocks mismatch (no differing block found?!)")
            exit(1)
    # Paged schedule: entries (any slot order, balanced across waves) must tile the split space.
    # Schedule entries carry *kernel* split indices, so Q-block ranges come from the decoded
    # splits in kernel order (one base's range is contiguous — a CTA reserves it atomically).
    req = request_ids.cpu().numpy().tolist()
    base_ranges, base_pairing = {}, {}
    for idx, s in enumerate(decoded['splits']):
        b = s['q_token_base']
        if b not in base_ranges:
            base_ranges[b] = [idx, idx + 1]
            base_pairing[b] = 2 if (b + 1 < total_q_tokens and req[b + 1] == req[b]) else 1
        else:
            base_ranges[b][1] = idx + 1
    total = ref['num_kv_splits']
    num_waves = decoded['num_waves']
    assert num_waves >= 1 and len(decoded['schedule']) == num_waves * num_slots
    covered = []
    for wave in range(num_waves):
        for slot_idx in range(num_slots):
            begin, end, q_token_base, num_q_tokens = decoded['schedule'][wave * num_slots + slot_idx]
            if begin == 0 and end == 0:
                continue
            covered.append((begin, end))
            if q_token_base not in base_ranges:
                print(f"ERROR: schedule entry references unknown Q base {q_token_base}")
                exit(1)
            lo, hi = base_ranges[q_token_base]
            if not (lo <= begin < end <= hi):
                print(f"ERROR: schedule entry ({begin},{end}) escapes Q block range [{lo},{hi})")
                exit(1)
            if num_q_tokens != base_pairing[q_token_base]:
                print(f"ERROR: schedule entry at base {q_token_base} num_q_tokens={num_q_tokens} "
                      f"vs expected {base_pairing[q_token_base]}")
                exit(1)
    covered.sort()
    expect_begin = 0
    for begin, end in covered:
        if begin != expect_begin or end <= begin:
            print(f"ERROR: paged schedule tiling broken at ({begin},{end}), expected begin {expect_begin}")
            exit(1)
        expect_begin = end
    if expect_begin != total:
        print(f"ERROR: paged schedule covers [0,{expect_begin}) but total splits = {total}")
        exit(1)

    # Verify the stable wave ordering, not only coverage: equal-sized entries
    # retain Q order, and rotation (including the reversed two-wave tail) is exact.
    ordered = []
    for base, (begin, end) in sorted(base_ranges.items()):
        entry_count = ceil_div(end - begin, 8)
        size, larger = divmod(end - begin, entry_count)
        for i in range(entry_count):
            next_begin = begin + size + (i < larger)
            ordered.append((begin, next_begin, base, base_pairing[base]))
            begin = next_begin
    expected_waves = max(1, ceil_div(len(ordered), num_slots))
    assert num_waves == expected_waves, (num_waves, expected_waves)
    ordered += [(0, 0, 0, 0)] * (num_waves * num_slots - len(ordered))
    for wave in range(num_waves):
        entries = sorted(ordered[wave * num_slots:(wave + 1) * num_slots], key=lambda e: e[1] - e[0])
        for rank, entry in enumerate(entries):
            slot = (num_slots - 1 - rank if num_waves == 2 and wave == 1 else
                    (rank + num_slots - wave * num_slots // num_waves) % num_slots)
            actual = tuple(decoded['schedule'][wave * num_slots + slot])
            assert actual == entry, f"Paged wave ordering mismatch at wave {wave}, slot {slot}: {actual} vs {entry}"

def test_sparse_mqa_logits(args) -> None:
    print('Testing Sparse MQA Logits:')
    data_type = args['data_type']
    seq_len_q = args['seq_len_q']
    seq_len_kv = args['seq_len_kv']
    num_heads = args.get('num_heads', 32)
    head_dim = args.get('head_dim', 128)
    sparse_block_kv = args.get('sparse_block_kv', 16)
    num_max_sparse_blocks = args.get('num_max_sparse_blocks', 128)
    use_unaligned_ks = bool(args.get('use_unaligned_ks', 0))
    print("test_sparse_mqa_logits->test_func: {},data_type:{},seq_len_q:{},seq_len_kv:{},num_heads:{},head_dim:{},"
          "sparse_block_kv:{},num_max_sparse_blocks:{},use_unaligned_ks:{}".format(
              args.get('gemm_type', 'SparseMqaLogits'), data_type, seq_len_q, seq_len_kv, num_heads, head_dim,
              sparse_block_kv, num_max_sparse_blocks, use_unaligned_ks))

    assert data_type == torch.uint8, "Sparse MQA Logits supports MXFP4 only"
    assert num_heads == 32 and head_dim == 128, "num_heads/head_dim are sparse layout constants (32/128)"
    assert num_max_sparse_blocks % 4 == 0 and num_max_sparse_blocks <= 4096
    assert sparse_block_kv in (8, 16)

    if args.get('gemm_type', 'SparseMqaLogits') == 'PagedSparseMqaLogits':
        # Paged KV: ks = 0 per token, ke = context_lens[q]; the fused fp4
        # cache interleaves a trailing 4B SF per token row; block_table resolves logical pages.
        page_kv = args.get('page_kv', 64)
        assert page_kv % sparse_block_kv == 0
        rng = random.Random(seq_len_q * 1000003 + seq_len_kv + sparse_block_kv)
        num_requests = min(rng.randint(2, 4), seq_len_q)
        request_ends = sorted(rng.sample(range(1, seq_len_q), num_requests - 1)) + [seq_len_q]
        request_sizes = [end - begin for begin, end in zip([0] + request_ends, request_ends)]
        request_indices = [r for r, s in enumerate(request_sizes) for _ in range(s)]
        context_lens_list = [seq_len_kv + offset for offset in range(seq_len_q)]
        context_lens = torch.tensor(context_lens_list, device='cuda', dtype=torch.int32)
        request_ids = torch.tensor(request_indices, device='cuda', dtype=torch.int32)

        # Page table: per-request page lists drawn from a permuted pool; tokens of one request
        # share their row content (the kernel only reads the Q block base token's row)
        max_pages = ceil_div(max(context_lens_list), page_kv)
        total_pages = sum(ceil_div(max(context_lens_list[i] for i, rr in enumerate(request_indices) if rr == r),
                                   page_kv) for r in range(num_requests))
        pool = torch.randperm(total_pages, device='cuda', dtype=torch.int32)
        req_page_lists, counter = [], 0
        for r in range(num_requests):
            r_max_len = max(context_lens_list[i] for i, rr in enumerate(request_indices) if rr == r)
            nblk = ceil_div(r_max_len, page_kv)
            req_page_lists.append(pool[counter:counter + nblk].tolist())
            counter += nblk
        block_table = torch.tensor(
            [req_page_lists[request_indices[t]] + [0] * (max_pages - len(req_page_lists[request_indices[t]]))
             for t in range(seq_len_q)], device='cuda', dtype=torch.int32)

        # Fused fp4 cache: [num_pages, page_kv, 1, head_dim/2 + 4] uint8 (SF trails each row),
        # page stride padded to 512B (upstream test construction; the kernel asserts it)
        kv_x = torch.randn(total_pages, page_kv, 1, head_dim, device='cuda', dtype=torch.bfloat16)
        kv_raw = kv_cache_cast_to_fp4(kv_x)[0]
        page_stride_bytes = ceil_div(page_kv * (head_dim // 2 + 4), 512) * 512
        kv_storage = torch.zeros((total_pages, page_stride_bytes), device='cuda', dtype=torch.uint8)
        kv_cache_fp4 = kv_storage.as_strided((total_pages, page_kv, 1, head_dim // 2 + 4),
                                             (page_stride_bytes, head_dim // 2 + 4, head_dim // 2 + 4, 1))
        kv_cache_fp4.copy_(kv_raw)

        torch.manual_seed(0)
        q = torch.randn(seq_len_q, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
        weights = torch.randn(seq_len_q, num_heads, device='cuda', dtype=torch.bfloat16)
        q_fp4 = per_token_cast_to_fp4(q.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        q_in = (q_fp4[0].view(seq_len_q, num_heads, head_dim // 2), q_fp4[1].view(seq_len_q, num_heads))

        # ks = 0 for every token: absolute sparse block ids from the pool start
        sparse_indices, num_blocks_per_q = make_sparse_kv_block_indices(
            context_lens, torch.zeros_like(context_lens), sparse_block_kv, num_max_sparse_blocks,
            seed=seq_len_q + seq_len_kv + sparse_block_kv, request_indices=request_indices)

        # Ground truth (acc mode only — perf mode must not launch the dense kernel into the
        # cycle CSV): dense paged fp4 kernel, gathered at the selected positions
        gathered_ref, valid_mask = None, None
        if get_acc_check():
            q_in_4d = (q_fp4[0].view(seq_len_q, 1, num_heads, head_dim // 2),
                       q_fp4[1].view(seq_len_q, 1, num_heads))
            context_lens_2d = context_lens.unsqueeze(-1)
            schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
                context_lens_2d, page_kv, deep_gemm.get_num_sms(),
                metadata_extra=(1, num_heads, head_dim // 2, 1))
            full_logits = deep_gemm.fp8_fp4_paged_mqa_logits(
                q=q_in_4d, fused_kv_cache=kv_cache_fp4, weights=weights, context_lens=context_lens_2d,
                block_table=block_table, schedule_meta=schedule_meta,
                max_context_len=max(context_lens_list), clean_logits=False, logits_dtype=torch.bfloat16)
            gathered_ref, valid_mask = gather_sparse_reference(
                full_logits, sparse_indices, num_blocks_per_q, torch.zeros_like(context_lens), context_lens,
                sparse_block_kv)

        metadata = deep_gemm.get_paged_sparse_mqa_logits_metadata(
            context_lens=context_lens, block_table=block_table, indices=request_ids, page_kv=page_kv,
            sparse_kv_block_indices=sparse_indices, qk_dtype=q_in[0].dtype, sparse_block_kv=sparse_block_kv)
        if args.get('check_metadata', 0):
            decoded = decode_sparse_metadata(metadata, sparse_block_kv)
            ref_meta = ref_paged_sparse_metadata(
                sparse_indices, num_blocks_per_q, context_lens, block_table, request_ids, page_kv,
                sparse_block_kv, num_max_sparse_blocks)
            verify_paged_sparse_metadata(decoded, ref_meta, seq_len_q, request_ids)
            print("Metadata check passed")
        sparse_logits = deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
            q=q_in, kv_cache=kv_cache_fp4, weights=weights, metadata=metadata,
            num_max_sparse_blocks=num_max_sparse_blocks, sparse_block_kv=sparse_block_kv)
        if get_acc_check():
            torch.cuda.synchronize()
            # The API pads each output row to 512 elements; padding is not a sparse slot.
            logical_logits = sparse_logits[:, :num_max_sparse_blocks * sparse_block_kv]
            if not torch.equal(logical_logits[valid_mask], gathered_ref[valid_mask]):
                print("ERROR: paged sparse MQA logits mismatch against gathered dense paged reference")
                exit(1)
            print("Accuracy check passed\n")
        return

    # Requests with random split points (2~4, upstream style); ks constant per request, per-token
    # sliding ke = seq_len_kv + token offset (deterministic, so edge values like 0 / split_kv±1 are
    # expressible via the case string). ks is grid-aligned unless use_unaligned_ks; the KV pool
    # stacks requests back to back.
    rng = random.Random(seq_len_q * 1000003 + seq_len_kv + sparse_block_kv)
    num_requests = min(rng.randint(2, 4), seq_len_q)
    request_ends = sorted(rng.sample(range(1, seq_len_q), num_requests - 1)) + [seq_len_q]
    request_sizes = [end - begin for begin, end in zip([0] + request_ends, request_ends)]
    request_indices = [req for req, size in enumerate(request_sizes) for _ in range(size)]
    aligned_starts, unaligned_starts, lengths = [], [], []
    aligned_end = unaligned_end = 0
    for size in request_sizes:
        aligned_start = ceil_div(aligned_end, sparse_block_kv) * sparse_block_kv
        unaligned_start = ceil_div(unaligned_end, sparse_block_kv) * sparse_block_kv + rng.randrange(1, sparse_block_kv)
        aligned_starts.extend([aligned_start] * size)
        unaligned_starts.extend([unaligned_start] * size)
        lengths.extend([seq_len_kv + offset for offset in range(size)])
        aligned_end = aligned_start + lengths[-1]
        unaligned_end = unaligned_start + lengths[-1]
    assert all(s % sparse_block_kv == 0 for s in aligned_starts)
    assert all(s % sparse_block_kv != 0 for s in unaligned_starts)
    num_kv_tokens = max(1, unaligned_end if use_unaligned_ks else aligned_end)
    context_starts_list = unaligned_starts if use_unaligned_ks else aligned_starts
    context_starts = torch.tensor(context_starts_list, device='cuda', dtype=torch.int32)
    context_ends = torch.tensor([s + l for s, l in zip(context_starts_list, lengths)],
                                device='cuda', dtype=torch.int32)

    q = torch.randn(seq_len_q, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    kv = torch.randn(num_kv_tokens, head_dim, device='cuda', dtype=torch.bfloat16)
    weights = torch.randn(seq_len_q, num_heads, device='cuda', dtype=torch.bfloat16)
    q_fp4 = per_token_cast_to_fp4(q.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    q_in = (q_fp4[0].view(seq_len_q, num_heads, head_dim // 2), q_fp4[1].view(seq_len_q, num_heads))
    q_dequant = cast_back_from_fp4(q_fp4[0], q_fp4[1], gran_k=32, use_packed_ue8m0=True).view(
        seq_len_q, num_heads, head_dim).to(torch.bfloat16)
    kv_fp4 = per_token_cast_to_fp4(kv, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    kv_in = (kv_fp4[0], kv_fp4[1].squeeze(-1))
    kv_dequant = cast_back_from_fp4(kv_fp4[0], kv_fp4[1], gran_k=32, use_packed_ue8m0=True).view(
        num_kv_tokens, head_dim).to(torch.bfloat16)

    sparse_indices, num_blocks_per_q = make_sparse_kv_block_indices(
        context_ends, context_starts, sparse_block_kv, num_max_sparse_blocks,
        seed=seq_len_q + num_kv_tokens + sparse_block_kv, request_indices=request_indices)

    # Ground truth (acc mode only — perf mode must not launch the dense kernel into the cycle
    # CSV): dense MXFP4 kernel with bf16 logits (compressed relative columns), gathered at the
    # selected positions. The sparse kernel must match this bitwise (same block-scale MMA
    # grouping and same ReLU+weighted-sum epilogue).
    gathered_ref, valid_mask = None, None
    if get_acc_check():
        full_logits = deep_gemm.fp8_fp4_mqa_logits(
            q=q_in, kv=kv_in, weights=weights, cu_seq_len_k_start=context_starts, cu_seq_len_k_end=context_ends,
            clean_logits=False, max_seqlen_k=int(num_kv_tokens), logits_dtype=torch.bfloat16)
        gathered_ref, valid_mask = gather_sparse_reference(
            full_logits, sparse_indices, num_blocks_per_q, context_starts, context_ends, sparse_block_kv)

        # Reference harness self-check: FP32 torch recompute at the gathered positions vs the
        # gathered dense output. This validates generation/gather/mask logic.
        if not valid_mask.any():
            # All-empty selections (e.g. seq_len_kv:0 edge case): nothing to compare; the sparse
            # kernel flow below must still run (empty selection is a legal metadata case)
            print("Sparse reference harness check skipped (no valid positions)")
        elif seq_len_q * num_max_sparse_blocks * sparse_block_kv > (1 << 25):
            # FP32 torch recompute materializes q_width x head_dim gathered KV (~256B/position);
            # skip on V4.1-scale cases (candidate_topk_blocks=2048) to avoid tens-of-GB spikes —
            # the dense-gather bitwise compare below remains the correctness check
            print("Sparse reference harness check skipped (case too large for FP32 recompute)")
        else:
            kv_offsets = torch.arange(sparse_block_kv, device='cuda', dtype=torch.int64)
            block_offsets = (context_starts % sparse_block_kv).to(torch.int64)
            token_indices = (sparse_indices.long().unsqueeze(-1) * sparse_block_kv
                             + block_offsets[:, None, None] + kv_offsets).flatten(1)
            kv_gathered = kv_dequant[token_indices.clamp(min=0, max=num_kv_tokens - 1)]
            ref_logits = ref_sparse_mqa_logits(q_dequant, kv_gathered, weights, valid_mask)
            from math_utils import calc_diff
            diff = calc_diff(gathered_ref.masked_fill(~valid_mask, 0).float(), ref_logits.float())
            threshold = 1.5e-3
            if torch.isnan(diff) or diff >= threshold:
                print(f"ERROR: Sparse reference harness check failed, diff={diff}")
                exit(1)
            else:
                print("Sparse reference harness check passed")

    metadata = deep_gemm.get_sparse_mqa_logits_metadata(
        cu_seq_len_k_start=context_starts, cu_seq_len_k_end=context_ends, num_kv_tokens=int(num_kv_tokens),
        sparse_kv_block_indices=sparse_indices, qk_dtype=q_in[0].dtype,
        sparse_block_kv=sparse_block_kv, use_unaligned_ks=use_unaligned_ks)
    if args.get('check_metadata', 0):
        # Decode the kernel's metadata and verify against the Python reference
        decoded = decode_sparse_metadata(metadata, sparse_block_kv)
        ref_meta = ref_sparse_metadata(
            sparse_indices, num_blocks_per_q, context_starts, context_ends, int(num_kv_tokens),
            sparse_block_kv, num_max_sparse_blocks, use_unaligned_ks)
        verify_sparse_metadata(decoded, ref_meta, seq_len_q)
        num_contiguous = sum(s["is_contiguous"] for s in decoded["splits"])
        print(f"Contiguous splits: {num_contiguous}/{decoded['num_kv_splits']}")
        print("Metadata check passed")
    sparse_logits = deep_gemm.fp8_fp4_sparse_mqa_logits(
        q=q_in, kv=kv_in, weights=weights, metadata=metadata,
        num_max_sparse_blocks=num_max_sparse_blocks, sparse_block_kv=sparse_block_kv,
        use_unaligned_ks=use_unaligned_ks)

    # Bitwise comparison against the gathered dense output (single kernel call per UT convention)
    if get_acc_check():
        # The API pads each output row to 512 elements; padding is not a sparse slot.
        logical_logits = sparse_logits[:, :num_max_sparse_blocks * sparse_block_kv]
        if not torch.equal(logical_logits[valid_mask], gathered_ref[valid_mask]):
            print("ERROR: sparse MQA logits mismatch against gathered dense reference")
            exit(1)
        print("Accuracy check passed\n")
    return

def test_einsum(args) -> None:
    b, h, r, d, data_type, expr = args['b'], args['h'], args['r'], args['d'], args["data_type"], args["expr"]
    quant_type = args['quant_type'] if 'quant_type' in args else 'block'
    if expr in ('bhr,hdr->bhd', 'bhr,hdr->hbd'):
        # The two expressions take the same inputs and the same quantization; they differ only in
        # the output layout, which the C++ side picks via `transposed_out`. `bhd` gets the strided
        # [b, h, d] view, `hbd` the plain packed [h, b, d].
        tensor_device = 'cuda' if get_ref_backend() == "device" else 'cpu'
        x = torch.randn((b, h, r), device=tensor_device, dtype=torch.bfloat16)
        y = torch.randn((h, d, r), device=tensor_device, dtype=torch.bfloat16)
        out_shape = (b, h, d) if expr == 'bhr,hdr->bhd' else (h, b, d)
        out = torch.empty(out_shape, device='cuda', dtype=torch.bfloat16)
        if _acc_check:
            ref_out = torch.einsum(expr, x, y)
        else:
            ref_out = torch.empty_like(out)
        if data_type == torch.float8_e4m3fn:
            if quant_type == 'block':
                x_fp8 = per_token_cast_to_fp8(x.view(-1, r), use_ue8m0=False)
                x_fp8 = x_fp8[0].view(b, h, r), x_fp8[1].view(b, h, ceil_div(r, 128))
                y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn),
                           torch.empty((h, ceil_div(d, 128), ceil_div(r, 128)), device='cuda', dtype=torch.float))
                for i in range(h):
                    y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i], use_ue8m0=False)
            elif quant_type == 'channel':
                x_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty((b, h, 1), device=tensor_device, dtype=torch.float))
                y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((h, d, 1), device=tensor_device, dtype=torch.float))
                for i in range(b):
                    x_fp8[0][i], x_fp8[1][i] = per_custom_dims_cast_to_fp8(x[i], (0, ), False, True)
                for i in range(h):
                    y_fp8[0][i], y_fp8[1][i] = per_custom_dims_cast_to_fp8(y[i], (0, ), False, True)
            deep_gemm.fp8_einsum(expr, x_fp8, y_fp8, out)
        elif data_type == torch.int8 and quant_type == 'channel':
            x_fp8 = (torch.empty_like(x, dtype=torch.int8), torch.empty((b, h, 1), device=tensor_device, dtype=torch.float))
            y_fp8 = (torch.empty_like(y, dtype=torch.int8), torch.empty((h, d, 1), device=tensor_device, dtype=torch.float))
            for i in range(b):
                x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_int8(x[i])
            for i in range(h):
                y_fp8[0][i], y_fp8[1][i] = per_token_cast_to_int8(y[i])
            deep_gemm.int8_einsum(expr, x_fp8, y_fp8, out)
        else:
            print("ERROR: Unsupported dtype, please check!")
            exit(1)

    else:
        raise ValueError(f"unsupported expr expression: {expr}!")

    if _acc_check:
        diff = calc_diff(out, ref_out)
        if diff >= 0.001:
            print("ref_out:", ref_out)
            print("out:", out)
        assert diff < 0.001, f'{h=}, {b=}, {d=}, {r=}, {diff:.5f}'
        print("Passed with acc_check\n")
    else:
        print("Passed without acc_check\n")

    if get_benchmark():
        t = bench_kineto(lambda: deep_gemm.fp8_einsum(expr, x_fp8, y_fp8, out), 'gemm', suppress_kineto_output=True)
        # t_cublaslt = bench_kineto(lambda: deep_gemm.einsum(expr, x, y, z, use_cublaslt=True), 'nvjet', suppress_kineto_output=True)
        print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                f'{t * 1e6:4.0f} us | '
                f'{2 * b * h * r * d / t / 1e12:4.0f} TFLOPS | '
                f'{count_bytes((x_fp8, y_fp8, out)) / t / 1e9:4.0f} GB/s | ')
                # f'{t_cublaslt / t:4.2f} x')


def read_cmds_from_file(casefile):
    dg_cases = list()
    with open(casefile, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            if line.endswith("list"):
                _case = read_cmds_from_file(line)
                dg_cases.extend(_case)
            else:
                _case = parse_deepgemm_string_re(line)
                dg_cases.append(_case)
    return dg_cases
