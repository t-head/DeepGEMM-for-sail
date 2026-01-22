#!/usr/bin/env python3
"""
GEMM Auto-Tuning Script
A single-file script for auto-tuning GEMM operations to find optimal configurations
for given matrix dimensions. This script avoids re-tuning already tuned configurations
by loading and saving results to a JSON file.
"""
import torch
import json
import os
import argparse
import triton
import math
import numpy as np
from datetime import datetime
from functools import lru_cache
from typing import Tuple, Dict, Any, Optional, List
from tqdm import tqdm

try:
    from transformers import AutoConfig
except ImportError:
    AutoConfig = None

from deep_gemm.deep_gemm_tuner.deepgemm_tools import get_supported_configs, get_pre_assert_configs
from deep_gemm.deep_gemm_tuner.utils import CANDIDATE_Ms, get_deep_gemm_luts, get_device_name

# Try to import deep_gemm functions
try:
    # breakpoint()
    from deep_gemm.utils import calc_diff
    from deep_gemm.jit_kernels import m_grouped_gemm_int8_int8_bf16_nt_masked, gemm_int8_int8_bf16_nt, get_num_sms
    from deep_gemm.jit_kernels.gemm_int8 import get_smem_config
    DEEP_GEMM_AVAILABLE = True
except ImportError:
    DEEP_GEMM_AVAILABLE = False
    print("Warning: deep_gemm not available, some functions will be mocked")

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
    rate = 1.0 / scale  # rate 参数 = 1 / scale
    # 定义 Gamma 分布（PyTorch 使用 concentration 和 rate）
    concentration = torch.tensor([shape])  # 形状参数
    rate_tensor = torch.tensor([rate])     # 速率参数

    # 创建分布并采样
    dist = torch.distributions.Gamma(concentration=concentration, rate=rate_tensor)
    samples = dist.sample((num_samples, num_groups)).to('cuda').int().squeeze()  # 采样 1000 个样本
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
        根据 target_key 插值得到 (shape, scale)

        参数:
            target_key (float): 输入的 key（如 15, 20 等）

        返回:
            tuple: (interpolated_shape, interpolated_scale)
        """
        if target_key < min(keys) or target_key > max(keys):
            print(f"警告: {target_key} 超出插值范围 [{min(keys)}, {max(keys)}]，结果可能不准确。")

        # 线性插值
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

    get_deep_gemm_luts(lhs, rhs, out, best_config)


def run_normal_gemm_test(M: int, K: int, N: int, config: Optional[Tuple] = None, num_groups: int = 1) -> Optional[float]:
    """
    运行普通的GEMM测试

    参数:
    M, K, N: 矩阵维度
    config: 配置参数
    num_groups: 分组数量

    返回:
    执行时间（毫秒）
    """
    assert(DEEP_GEMM_AVAILABLE)

    # Create input tensors
    x = torch.randn((M, K), dtype=torch.float16, device="cuda") * 0.1

    # Create weight tensors
    weight_fp32 = (torch.rand((K, N), dtype=torch.float32, device="cuda") - 0.5) * 2
    weight = (weight_fp32 * 127).clamp(min=-128, max=127).to(torch.int8)
    weight = weight.t().contiguous().t()

    # Quantize input
    x_q, x_scale = per_token_quant_int8(x)

    # Create weight scale
    weight_scale = torch.rand(N, 1, device=weight.device) * 1e-2

    # No bias in these test cases
    bias = None
    try:
        with torch.inference_mode():
            ref_out = torch.empty([M, N], device=x.device, dtype=torch.bfloat16)
            # Compute reference result using acext kernel
            gemm_nt_i8i8bf16(
                (x_q, x_scale),
                (weight.t(), weight_scale),
                ref_out
            )

            f1 = lambda: gemm_nt_i8i8bf16(
                (x_q, x_scale),
                (weight.t(), weight_scale),
                ref_out
            )

            test_out = torch.empty([M, N], device=x.device, dtype=torch.bfloat16)
            gemm_nt_i8i8bf16(
                (x_q, x_scale),
                (weight.t(), weight_scale),
                test_out,
                config
            )
            f2 = lambda: gemm_nt_i8i8bf16(
                (x_q, x_scale),
                (weight.t(), weight_scale),
                test_out,
                config
            )
            for _ in range(10):
                f1()
                f2()

            # Check that results match within tolerance
            diff = torch.mean(torch.abs(test_out.to(torch.float32) - ref_out.to(torch.float32)))
            rel_diff = diff / torch.mean(torch.abs(ref_out.to(torch.float32)))
            if rel_diff >= 0.05:
                print(f"Relative difference too large: {rel_diff}. "
                    f"Shapes: x_q={x_q.shape}, weight={weight.shape}, "
                    f"x_scale={x_scale.shape}, weight_scale={weight_scale.shape}")
                return None

            use_time_deep_gemm = triton.testing.do_bench(f2)

            return 1000 * use_time_deep_gemm

    except Exception as e:
        print(f"Error in run_normal_gemm_test: {e}")
        return None



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
        else tuner.get_deep_gemm_luts(expected_m, n, k, num_groups=num_groups)
    )

    with configure_deep_gemm_num_sms(
        overlap_args.num_sms if overlap_args is not None else None
    ):
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

def run_grouped_gemm_test(M: int, K: int, N: int, config: Optional[Tuple] = None, num_groups: int = 1) -> Optional[float]:
    """
    运行分组GEMM测试

    参数:
    M, K, N: 矩阵维度
    config: 配置参数
    num_groups: 分组数量

    返回:
    执行时间（毫秒）
    """
    assert(DEEP_GEMM_AVAILABLE)

    # Create input tensors for grouped gemm, similar to normal gemm but with groups dimensionn
    x = torch.randn((num_groups, 4096, K), dtype=torch.float16, device="cuda") * 0.1

    # Create weight tensors for grouped gemm with shape [num_groups, N, K]
    # inplace op to save memory
    weight_fp32 = torch.rand((num_groups, N, K), dtype=torch.float32, device="cuda")
    weight_fp32.sub(0.5).mul(127*2).clamp(min=-128, max=127)
    weight = weight_fp32.to(torch.int8)

    # Quantize input
    x_q, x_scale = per_token_quant_int8(x)

    # Create weight scale with shape [num_groups, N, 1]
    weight_scale = torch.rand((num_groups, N, 1), device="cuda", dtype=torch.float32) * 1e-2

    # Create output tensor
    output_baseline = torch.empty([num_groups, 4096, N], device="cuda", dtype=torch.bfloat16)
    output_test = torch.empty((num_groups, 4096, N), device="cuda", dtype=torch.bfloat16)

    total_cases = 100

    # sample from random input gate
    def gen_expert_num_tokens(total_cases=100):
        ret = []
        from sglang.srt.layers.moe.topk import select_experts, TopKConfig
        from sglang.srt.layers.moe.fused_moe_triton.deepgemm_moe import count_expert_num_tokens
        num_tokens = (M-1)*256//(8)
        topk_config = TopKConfig(
            top_k=8,
            renormalize=True,
        )

        for _ in range(total_cases):
            x_dummy =  torch.randn((num_tokens, K), dtype=torch.float16, device="cuda")
            input_gating = torch.randn(num_tokens, 256, device='cuda', dtype=torch.float32)
            _, topk_ids, _ = select_experts(x_dummy, input_gating, topk_config)
            expert_num_tokens = count_expert_num_tokens(topk_ids, 16, 1)
            ret.append(expert_num_tokens)
        return ret

    # Create masked_m tensor and fill with M value
    if 7168 in (K, N): # deepseek-R1
        # gamma sample
        masked_m_all = grouped_masked_m_sample(expect_m=M, num_groups=num_groups, num_samples=total_cases)
    else:
        # standard distribution sample
        masked_m_all = gen_expert_num_tokens(total_cases)

    try:
        with torch.inference_mode():
            def run_with_masked_m(output, config=None):
                for i in range(total_cases):
                    grouped_gemm_nt_i8i8bf16_masked(
                        (x_q, x_scale),
                        (weight, weight_scale),
                        output,
                        masked_m_all[i],
                        M,
                        config
                    )
            f1 = lambda: run_with_masked_m(output_baseline)

            f2 = lambda: run_with_masked_m(output_test, config=config)

            # Warmup runs
            f1()
            f2()

            ref_out = output_baseline
            output = output_test

            # Check that results match within tolerance
            masked_m = masked_m_all[-1]
            for j in range(num_groups):
                diff = calc_diff(output[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()])
                if (masked_m[j] != 0):
                    if diff >= 0.001:
                        print(f"ref_out[{j}]:", ref_out[j, :masked_m[j].item()])
                        print(f"out[{j}]:", output[j, :masked_m[j].item()])
                        assert diff < 0.001, f'{M=}, {K=}, {N=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'
                        return None

            use_time_deep_gemm = triton.testing.do_bench(f2) / total_cases

            return 1000 * use_time_deep_gemm

    except Exception as e:
        print(f"Error in run_grouped_gemm_test: {e}")
        return None


def load_tuned_configs(filename: str = "best_gemm_configs.json") -> Dict[Tuple[int, int, int, int], Dict[str, Any]]:
    """
    从文件中加载已调优的配置，避免重复调优

    参数:
    filename: 配置文件名

    返回:
    以(M,K,N,num_groups)为键的配置字典
    """
    try:
        with open(filename, 'r') as f:
            configs_data = json.load(f)

        # Convert configs data to dictionary with (M,K,N,num_groups) as keys
        tuned_configs = {}
        for config in configs_data:
            key = (config["M"], config["K"], config["N"], config.get("num_groups", 1))
            # fix old version with num_groups arg not exists
            if "num_groups" not in config:
                config["num_groups"] = 1
            tuned_configs[key] = config

        print(f"Loaded {len(tuned_configs)} pre-tuned configurations from {filename}")
        return tuned_configs
    except FileNotFoundError:
        print(f"Configuration file {filename} not found. Starting fresh tuning.")
        return {}
    except Exception as e:
        print(f"Error loading configurations from {filename}: {e}. Starting fresh tuning.")
        return {}


def save_tuned_configs(configs: List[Dict[str, Any]], filename: str = "best_gemm_configs.json") -> None:
    """
    将调优后的配置保存到文件

    参数:
    configs: 配置列表
    filename: 保存的文件名
    """
    # Ensure config data is serializable
    serializable_configs = []
    for config in configs:
        if config is None:
            continue
        serializable_config = {}
        for key, value in config.items():
            if isinstance(value, torch.Tensor):
                serializable_config[key] = value.tolist()
            elif isinstance(value, (int, float, str, bool, type(None))):
                serializable_config[key] = value
            else:
                serializable_config[key] = str(value)
        serializable_configs.append(serializable_config)

    # Also save to the default file for future use
    with open(filename, 'w') as f:
        json.dump(serializable_configs, f, indent=2)

    print(f"Best configs also saved to {filename}")


def tune_gemm_config(
    m: int,
    k: int,
    n: int,
    num_groups: int,
    tuned_configs: Optional[Dict[Tuple[int, int, int, int], Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
    """
    为特定尺寸的GEMM操作调优配置

    参数:
    m, k, n: 矩阵维度
    num_groups: 分组数量
    tuned_configs: 已调优的配置字典

    返回:
    最佳配置或None（如果已存在或调优失败）
    """
    # Check if already tuned
    config_key = (m, k, n, num_groups)
    if config_key in tuned_configs:
        print(f"Configuration for M={m}, K={k}, N={n}, num_groups={num_groups} already tuned. Skipping...")
        return tuned_configs[config_key]

    print(f"Tuning configuration for M={m}, K={k}, N={n}, num_groups={num_groups}...")
    if num_groups == 1:
        baseline_time = run_normal_gemm_test(m, k, n)
        gemm_type = "dense"
    else:
        baseline_time = run_grouped_gemm_test(m, k, n, num_groups=num_groups)
        gemm_type = "masked"
    if baseline_time is None:
        print(f"Failed to get baseline time for M={m}, K={k}, N={n}, num_groups={num_groups}")
        return None
    
    configs = get_pre_assert_configs(m, n, k, num_groups, get_num_sms(), gemm_type)
    best_time = baseline_time
    best_config = None

    for config in tqdm(configs):
        num_min_sms, best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages, best_smem_config = config
        if num_groups == 1:
            time = run_normal_gemm_test(m, k, n, config)
        else:
            time = run_grouped_gemm_test(m, k, n, config, num_groups)
        if time is not None and time < best_time and (1 - (time / baseline_time)) > 0.01:
            best_time = time
            best_config = {
                "M": m,
                "K": k,
                "N": n,
                "num_groups": num_groups,
                "config": {
                    "num_min_sms": num_min_sms,
                    "best_block_m": best_block_m,
                    "best_block_n": best_block_n,
                    "block_k": block_k,
                    "warp_m": warp_m,
                    "warp_n": warp_n,
                    "best_num_stages": best_num_stages,
                    "best_smem_config": best_smem_config,
                },
                "time_ms": time,
                "baseline_time_ms": baseline_time,
            }
            acc = (1 - (time / baseline_time))
            best_config["acc"] = acc
            print(f"mnk: {m}x{n}x{k}, config: {config}, (groups:{num_groups}) - Time: {time:.3f}us - Acc: {acc:.2f}")

    print(f"{m}x{n}x{k} (groups:{num_groups}), config: {config}, - best_time: {best_time:.3f}us - baseline_time: {baseline_time:.2f}")

    return best_config

import ray
import logging

@ray.remote(num_gpus=1)
class BenchmarkWorker:

    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        torch.cuda.manual_seed_all(0)
        self.seed = seed

    def tune(
        self,
        m: int,
        k: int,
        n: int,
        num_groups: int,
        tuned_configs
    ) -> Dict[str, int]:
        best_config = tune_gemm_config(m, k, n, num_groups, tuned_configs)
        if best_config is None:
            print(f"Warning: No valid configuration found for M={m}, K={k}, N={n}, num_groups={num_groups}")
        return best_config


def get_test_cases(args):
    test_case_base = []
    if not AutoConfig:
        return []
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    # breakpoint()
    if config.architectures[0] in ["Qwen2MoeForCausalLM", "Qwen3MoeForCausalLM"]:
        n_routed_experts = config.num_experts
        n_ep_device = args.tp_size

        hidden_size = config.hidden_size
        head_dim = config.head_dim
        num_attention_heads = config.num_attention_heads // args.tp_size
        num_key_value_heads = max(config.num_key_value_heads // args.tp_size, 1)

        moe_intermediate_size = config.moe_intermediate_size
        shard_moe_intermediate_size = 2 * moe_intermediate_size

        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2*intermediate_size // args.tp_size

        # qkv_proj: hidden_size -> (num_attention_heads + 2 * num_key_value_heads)*head_dim
        test_case_base.append((hidden_size, (num_attention_heads + 2 * num_key_value_heads)*head_dim, 1))
        # o_proj: num_attention_heads * head_dim -> hidden_size
        test_case_base.append((num_attention_heads * head_dim, hidden_size, 1))

        # fused_moe_up_gate: hidden_size -> shard_moe_intermediate_size
        test_case_base.append((hidden_size, shard_moe_intermediate_size, 1))
        # fused_mlp_up_gate: hidden_size -> shard_intermediate_size
        test_case_base.append((hidden_size, shard_intermediate_size, 1))
        # moe_down: moe_intermediate_size -> hidden_size
        test_case_base.append((moe_intermediate_size, hidden_size, 1))
        # mlp_down: intermediate_size -> hidden_size
        test_case_base.append((intermediate_size, hidden_size, 1))

        for n_experts in [n_routed_experts//n_ep_device, n_routed_experts//n_ep_device + 1]:
            test_case_base.append((moe_intermediate_size, hidden_size, n_experts))
            test_case_base.append((hidden_size, shard_moe_intermediate_size, n_experts))
    elif config.architectures[0] in ["MixTBStarsForCausalLM"]:
        n_routed_experts = config.num_routed_experts
        n_ep_device = args.tp_size

        hidden_size = config.hidden_size
        head_dim = config.hidden_size // config.num_attention_heads
        num_attention_heads = config.num_attention_heads // args.tp_size
        num_key_value_heads = max(config.num_key_value_heads // args.tp_size, 1)

        moe_intermediate_size = config.intermediate_size
        shard_moe_intermediate_size = 2 * moe_intermediate_size

        # qkv_proj: hidden_size -> (num_attention_heads + 2 * num_key_value_heads)*head_dim
        test_case_base.append((hidden_size, (num_attention_heads + 2 * num_key_value_heads)*head_dim, 1))
        # o_proj: num_attention_heads * head_dim -> hidden_size
        test_case_base.append((num_attention_heads * head_dim, hidden_size, 1))

        # fused_moe_up_gate: hidden_size -> shard_moe_intermediate_size
        test_case_base.append((hidden_size, shard_moe_intermediate_size, 1))
        # moe_down: moe_intermediate_size -> hidden_size
        test_case_base.append((moe_intermediate_size, hidden_size, 1))

        for n_experts in [n_routed_experts//n_ep_device, n_routed_experts//n_ep_device + 1]:
            test_case_base.append((moe_intermediate_size, hidden_size, n_experts))
            test_case_base.append((hidden_size, shard_moe_intermediate_size, n_experts))
    elif config.architectures[0] in ["TBStars2_5_ForCausalLM"]:
        n_routed_experts = config.num_routed_experts
        n_ep_device = args.tp_size

        hidden_size = config.hidden_size
        head_dim = config.hidden_size // config.num_attention_heads
        num_attention_heads = config.num_attention_heads // args.tp_size
        num_key_value_heads = max(config.num_key_value_heads // args.tp_size, 1)

        moe_intermediate_size = config.moe_intermediate_size
        shard_moe_intermediate_size = 2 * moe_intermediate_size

        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2*intermediate_size // args.tp_size

        # qkv_proj: hidden_size -> (num_attention_heads + 2 * num_key_value_heads)*head_dim
        test_case_base.append((hidden_size, (num_attention_heads + 2 * num_key_value_heads)*head_dim, 1))
        # o_proj: num_attention_heads * head_dim -> hidden_size
        test_case_base.append((num_attention_heads * head_dim, hidden_size, 1))

        # fused_moe_up_gate: hidden_size -> shard_moe_intermediate_size
        test_case_base.append((hidden_size, shard_moe_intermediate_size, 1))
        # fused_mlp_up_gate: hidden_size -> shard_intermediate_size
        test_case_base.append((hidden_size, shard_intermediate_size, 1))
        # moe_down: moe_intermediate_size -> hidden_size
        test_case_base.append((moe_intermediate_size, hidden_size, 1))
        # mlp_down: intermediate_size -> hidden_size
        test_case_base.append((intermediate_size, hidden_size, 1))

        for n_experts in [n_routed_experts//n_ep_device, n_routed_experts//n_ep_device + 1]:
            test_case_base.append((moe_intermediate_size, hidden_size, n_experts))
            test_case_base.append((hidden_size, shard_moe_intermediate_size, n_experts))
    elif config.architectures[0] in ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"]:
        n_routed_experts = config.n_routed_experts
        n_ep_device = args.tp_size

        moe_intermediate_size = config.moe_intermediate_size
        shard_moe_intermediate_size = 2 * moe_intermediate_size

        hidden_size = config.hidden_size
        q_lora_rank = config.q_lora_rank
        qk_nope_head_dim = config.qk_nope_head_dim
        qk_rope_head_dim = config.qk_rope_head_dim
        kv_lora_rank = config.kv_lora_rank
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2*intermediate_size // args.tp_size
        num_attention_heads = config.num_attention_heads // args.tp_size
        v_head_dim = config.v_head_dim

        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        # fused_qkv_a_proj_with_mqa: hidden_size -> self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
        test_case_base.append((hidden_size, q_lora_rank+kv_lora_rank+qk_rope_head_dim, 1))
        # q_b_proj: q_lora_rank -> n_head * qk_head_dim
        test_case_base.append((q_lora_rank, num_attention_heads * qk_head_dim, 1))
        # kv_b_proj: kv_lora_rank -> n_head * (qk_nope_head_dim + v_head_dim)
        test_case_base.append((kv_lora_rank, num_attention_heads * (qk_nope_head_dim + v_head_dim), 1))
        # o_proj: num_attention_heads * v_head_dim ->hidden_size
        test_case_base.append((num_attention_heads * v_head_dim, hidden_size, 1))
        # fused_moe_up_gate: hidden_size -> shard_moe_intermediate_size
        test_case_base.append((hidden_size, shard_moe_intermediate_size, 1))
        # fused_mlp_up_gate: hidden_size -> shard_intermediate_size
        test_case_base.append((hidden_size, shard_intermediate_size, 1))
        # moe_down: moe_intermediate_size -> hidden_size
        test_case_base.append((moe_intermediate_size, hidden_size, 1))
        # mlp_down: intermediate_size -> hidden_size
        test_case_base.append((intermediate_size, hidden_size, 1))

        for n_experts in [n_routed_experts//n_ep_device, n_routed_experts//n_ep_device + 1]:
            test_case_base.append((moe_intermediate_size, hidden_size, n_experts))
            test_case_base.append((hidden_size, shard_moe_intermediate_size, n_experts))

    else:
        # Default: Mixtral
        n_routed_experts = config.num_local_experts
        n_ep_device = args.tp_size

        hidden_size = config.hidden_size
        head_dim = config.head_dim
        num_attention_heads = config.num_attention_heads // args.tp_size
        num_key_value_heads = max(config.num_key_value_heads // args.tp_size, 1)

        moe_intermediate_size = config.intermediate_size
        shard_moe_intermediate_size = 2 * moe_intermediate_size

        # qkv_proj: hidden_size -> (num_attention_heads + 2 * num_key_value_heads)*head_dim
        test_case_base.append((hidden_size, (num_attention_heads + 2 * num_key_value_heads)*head_dim, 1))
        # o_proj: num_attention_heads * head_dim -> hidden_size
        test_case_base.append((num_attention_heads * head_dim, hidden_size, 1))
        # fused_moe_up_gate: hidden_size -> shard_moe_intermediate_size
        test_case_base.append((hidden_size, shard_moe_intermediate_size, 1))
        # moe_down: moe_intermediate_size -> hidden_size
        test_case_base.append((moe_intermediate_size, hidden_size, 1))

        for n_experts in [n_routed_experts//n_ep_device, n_routed_experts//n_ep_device + 1]:
            test_case_base.append((moe_intermediate_size, hidden_size, n_experts))
            test_case_base.append((hidden_size, shard_moe_intermediate_size, n_experts))

    test_cases = []

    for bs in CANDIDATE_Ms:
        for tc in test_case_base:
            test_cases.append((bs,)+tc[:])

    return test_cases


def tuning_deepgemm_config_entrypoint(test_cases, tp, seed=0, model="anonymous", tuned_config=None, out_of_box=False):
    print(f"tuning cases are {test_cases}")

    # Load previously tuned configurations
    tuned_configs = load_tuned_configs(tuned_config)

    # Tune configurations for test cases
    best_configs = [x for x in tuned_configs.values()]
    best_configs = sorted(best_configs, key = lambda x : str(x))

    # init cluster
    ray.init()

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='ray_output.log', level=logging.INFO)

    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(seed) for _ in range(num_gpus)]
    def _distribute(method: str, inputs: List[Any]) -> List[Any]:
        outputs = []
        worker_idx = 0
        for input_args in inputs:
            worker = workers[worker_idx]
            worker_method = getattr(worker, method)
            output = worker_method.remote(*input_args)
            outputs.append(output)
            worker_idx = (worker_idx + 1) % num_gpus
        return ray.get(outputs)

    best_configs = _distribute(
        "tune",
        [tuple(test_case) + (tuned_configs,) for test_case in test_cases ]
    )

    # Save all configurations
    if best_configs:
        device_name = get_device_name().replace(" ", "_")
        save_path = ""
        if out_of_box:
            save_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "configs",
            )
            assert os.path.exists(save_dir), "Deepgemm int8 configs dir "+save_dir+" do not exist, please upgrade to the latest version."
            def get_timestamp():
                from datetime import datetime
                # 获取当前时间
                current_time = datetime.now()
                # 格式化时间为 "YY/MM/DD/HH/MM"
                formatted_time = current_time.strftime("%y-%m-%d-%H-%M")
                return formatted_time

            save_path = os.path.join(save_dir, get_timestamp())

        else:
            save_path = (model.split("/")[-1] if len(model.split("/")[-1])>0 else args.model.split("/")[-2])

        save_tuned_configs(best_configs,  save_path + "-tp" + str(tp) +",device_name="+device_name+"-deepgemm_configs.json")
        print(f"Tuning completed. Found {len(best_configs)} best configurations.")
    else:
        print("No configurations were tuned.")


def tuning_deepgemm_model_config(args):
    """Main function to run auto-tuning"""
    torch.set_default_device("cuda")

    print(args)
    tuning_deepgemm_config_entrypoint(get_test_cases(args), args.tp_size, args.seed, args.model, args.tuned_config, out_of_box=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tuned-config", type=str, default=""
    )
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tp-size", "--tp", type=int, default=2)
    args = parser.parse_args()

    tuning_deepgemm_model_config(args)