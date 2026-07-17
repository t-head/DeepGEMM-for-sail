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
from datetime import datetime
from functools import lru_cache
from typing import Tuple, Dict, Any, Optional, List
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

try:
    from transformers import AutoConfig
except ImportError:
    AutoConfig = None

from deep_gemm.deep_gemm_tuner.deepgemm_tools import get_supported_configs, get_pre_assert_configs, exec_tuning_iter
from deep_gemm.deep_gemm_tuner.deepgemm_tools import count_expert_num_tokens, fused_topk_torch_native, grouped_masked_m_sample, deepgemm_moe_permute
from deep_gemm.deep_gemm_tuner.deepgemm_tools import per_token_quant_int8, gemm_nt_i8i8bf16, grouped_gemm_nt_i8i8bf16_masked, grouped_gemm_nt_i8i8bf16_nopad
from deep_gemm.deep_gemm_tuner.deepgemm_tools import grouped_gemm_nt_bf16bf16bf16_nopad
from deep_gemm.deep_gemm_tuner.utils import CANDIDATE_Ms, get_deep_gemm_luts, get_device_name

# Try to import deep_gemm functions
from deep_gemm import calc_diff
from deep_gemm.jit_kernels import m_grouped_gemm_int8_int8_bf16_nt_masked, gemm_int8_int8_bf16_nt, get_num_sms
from deep_gemm.jit_kernels.gemm_int8 import get_smem_config
DEEP_GEMM_AVAILABLE = True

ENABLE_GAMMA_SAMPLE = os.environ.get("ENABLE_GAMMA_SAMPLE", None)

def run_grouped_gemm_nopad_test_bf16(M: int, K: int, N: int, config: Optional[Tuple] = None, num_groups: int = 1) -> Optional[float]:
    """
    Run grouped GEMM test

    Args:
    M, K, N: Matrix dimensions
    config: Configuration parameters
    num_groups: Number of groups

    Returns:
    Execution time (milliseconds)
    """
    assert(DEEP_GEMM_AVAILABLE)
    DEBUG_MODE = int(os.getenv("DEEPGEMM_TUNER_DEBUG_MODE", 0))

    # Create input tensors for grouped gemm, similar to normal gemm but with groups dimensionn
    topk = 8 # hard code
    actual_M = M // topk
    x = torch.randn((actual_M, K), dtype=torch.bfloat16, device="cuda") * 0.1

    # Create weight tensors for grouped gemm with shape [num_groups, N, K]
    weight_bf16 = (torch.randn((num_groups, N, K), dtype=torch.bfloat16, device="cuda") - 0.5) * 2

    output_baseline = torch.empty([M, N], device="cuda", dtype=torch.bfloat16)
    output_test = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

    # Create expert_ids tensor and fill with [0...num_group] value
    input_gating = torch.randn(actual_M, num_groups, dtype=torch.float32, device="cuda")
    _, topk_ids = fused_topk_torch_native(x, input_gating, topk=topk, renormalize=True)
    a, a_scale, expert_ids, inv_perm, num_recv_tokens_per_expert = deepgemm_moe_permute(
        aq=x,
        aq_scale=None,
        topk_ids=topk_ids,
        local_num_experts=num_groups,
        block_align=1,
        block_k=K
    )

    def grouped_gemm_nopad_bf16():
        with torch.inference_mode():
            f1 = lambda: grouped_gemm_nt_bf16bf16bf16_nopad(
                a, 
                weight_bf16, 
                output_baseline, 
                expert_ids
            )

            f2 = lambda: grouped_gemm_nt_bf16bf16bf16_nopad(
                a, 
                weight_bf16,
                output_test, 
                expert_ids,
                num_recv_tokens_per_expert, 
                config
            )

            # Warmup runs
            for _ in range(10):
                f1()
                f2()
            ref_out = output_baseline
            output = output_test

            # Check that results match within tolerance
            diff = torch.mean(torch.abs(output.to(torch.float32) - ref_out.to(torch.float32)))
            rel_diff = diff / torch.mean(torch.abs(ref_out.to(torch.float32)))
            if rel_diff >= 0.05:
                # breakpoint()
                print(f"Relative difference too large: {rel_diff}. "
                    f"Shapes: x_q={x_q.shape}, weight={weight.shape}, "
                    f"x_scale={x_scale.shape}, weight_scale={weight_scale.shape}")
                return None

            use_time_deep_gemm = triton.testing.do_bench(f2)

            return 1000 * use_time_deep_gemm

    return exec_tuning_iter(grouped_gemm_nopad_bf16, "run_grouped_gemm_nopad_bf16_test", DEBUG_MODE)


def run_grouped_gemm_nopad_test_int8(M: int, K: int, N: int, config: Optional[Tuple] = None, num_groups: int = 1) -> Optional[float]:
    """
    Run grouped GEMM test

    Args:
    M, K, N: Matrix dimensions
    config: Configuration parameters
    num_groups: Number of groups

    Returns:
    Execution time (milliseconds)
    """
    assert(DEEP_GEMM_AVAILABLE)
    DEBUG_MODE = int(os.getenv("DEEPGEMM_TUNER_DEBUG_MODE", 0))

    # Create input tensors for grouped gemm, similar to normal gemm but with groups dimensionn
    # hard code
    if num_groups == 512:
        topk = 10 # qwen3.5
    else:
        topk = 8
    actual_M = M // topk
    x = torch.randn((actual_M, K), dtype=torch.float16, device="cuda") * 0.1

    # Create weight tensors for grouped gemm with shape [num_groups, N, K]
    weight_fp32 = (torch.randn((num_groups, N, K), dtype=torch.float32, device="cuda") - 0.5) * 2
    weight = (weight_fp32 * 127).clamp(min=-128, max=127).to(torch.int8)

    # Quantize input
    x_q, x_scale = per_token_quant_int8(x)
    block_k = K
    block_align = 1 # for int8

    # Create weight scale with shape [num_groups, N, 1]
    weight_scale = torch.rand((num_groups, N, 1), device="cuda", dtype=torch.float32) * 1e-2

    # Create output tensor

    output_baseline = torch.empty([M, N], device="cuda", dtype=torch.bfloat16)
    output_test = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

    # Create expert_ids tensor and fill with [0...num_group] value
    input_gating = torch.randn(actual_M, num_groups, dtype=torch.float32, device="cuda")
    _, topk_ids = fused_topk_torch_native(x, input_gating, topk=topk, renormalize=True)
    a, a_scale, expert_ids, inv_perm, num_recv_tokens_per_expert = deepgemm_moe_permute(
        aq=x_q,
        aq_scale=x_scale,
        topk_ids=topk_ids,
        local_num_experts=num_groups,
        block_align=1,
        block_k=K
    )
    
    def grouped_gemm_nopad():
        with torch.inference_mode():
            f1 = lambda: grouped_gemm_nt_i8i8bf16_nopad(
                (a, a_scale),
                (weight, weight_scale),
                output_baseline,
                expert_ids
            )

            f2 = lambda: grouped_gemm_nt_i8i8bf16_nopad(
                (a, a_scale),
                (weight, weight_scale),
                output_test,
                expert_ids,
                num_recv_tokens_per_expert,
                config
            )

            # Warmup runs
            for _ in range(10):
                f1()
                f2()
            ref_out = output_baseline
            output = output_test

            # Check that results match within tolerance
            diff = torch.mean(torch.abs(output.to(torch.float32) - ref_out.to(torch.float32)))
            rel_diff = diff / torch.mean(torch.abs(ref_out.to(torch.float32)))
            if rel_diff >= 0.05:
                print(f"Relative difference too large: {rel_diff}. "
                    f"Shapes: x_q={x_q.shape}, weight={weight.shape}, "
                    f"x_scale={x_scale.shape}, weight_scale={weight_scale.shape}")
                return None

            use_time_deep_gemm = triton.testing.do_bench(f2)

            return 1000 * use_time_deep_gemm

    return exec_tuning_iter(grouped_gemm_nopad, "run_grouped_gemm_nopad_test_int8", DEBUG_MODE)


def run_normal_gemm_test_int8(M: int, K: int, N: int, config: Optional[Tuple] = None, num_groups: int = 1) -> Optional[float]:
    """
    Run normal GEMM test

    Args:
    M, K, N: Matrix dimensions
    config: Configuration parameters
    num_groups: Number of groups

    Returns:
    Execution time (milliseconds)
    """
    assert(DEEP_GEMM_AVAILABLE)
    DEBUG_MODE = int(os.getenv("DEEPGEMM_TUNER_DEBUG_MODE", 0))

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
    
    def dense_gemm_normal():
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
        
    return exec_tuning_iter(dense_gemm_normal, "run_normal_gemm_test_int8", DEBUG_MODE)


def run_grouped_gemm_test_masked_int8(M: int, K: int, N: int, config: Optional[Tuple] = None, num_groups: int = 1) -> Optional[float]:
    """
    Run grouped GEMM test

    Args:
    M, K, N: Matrix dimensions
    config: Configuration parameters
    num_groups: Number of groups

    Returns:
    Execution time (milliseconds)
    """
    assert(DEEP_GEMM_AVAILABLE)
    DEBUG_MODE = int(os.getenv("DEEPGEMM_TUNER_DEBUG_MODE", 0))
    # hard code
    topk=8

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
        num_tokens = (M-1)*256//(8)

        for _ in range(total_cases):
            x_dummy =  torch.randn((num_tokens, K), dtype=torch.float16, device="cuda")
            input_gating = torch.randn(num_tokens, 256, device='cuda', dtype=torch.float32)
            _, topk_ids = fused_topk_torch_native(x_dummy, input_gating, topk=topk, renormalize=True)
            expert_num_tokens = count_expert_num_tokens(topk_ids, 16, 1)
            ret.append(expert_num_tokens)
        return ret

    # Create masked_m tensor and fill with M value
    if ENABLE_GAMMA_SAMPLE: # deepseek-R1
        # gamma sample
        masked_m_all = grouped_masked_m_sample(expect_m=M, num_groups=num_groups, num_samples=total_cases)
    else:
        # standard distribution sample
        masked_m_all = gen_expert_num_tokens(total_cases)

    def group_gemm_masked():
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
    return exec_tuning_iter(group_gemm_masked, "run_grouped_gemm_test_masked_int8", DEBUG_MODE)


def load_tuned_configs(filename: str = "best_gemm_configs.json") -> Dict[Tuple[int, int, int, int], Dict[str, Any]]:
    """
    Load pre-tuned configurations from file to avoid re-tuning

    Args:
    filename: Configuration file name

    Returns:
    Configuration dictionary with (M,K,N,num_groups) as key
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
    Save tuned configurations to file

    Args:
    configs: List of configurations
    filename: File name to save to
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

def dispatch_tune_method(num_groups:int, nopad: bool, dtype: str):
    if dtype == "int8":
        if num_groups == 1:
            return run_normal_gemm_test_int8
        elif nopad:
            return run_grouped_gemm_nopad_test_int8
        else:
             return run_grouped_gemm_test_masked_int8
    elif dtype == "bf16":
        if nopad:
             return run_grouped_gemm_nopad_test_bf16
        else:
            assert False, f"unsupport dtype bf16 with nopad=False"
    else:
        assert False, f"unsupport dtype {dtype}"

def tune_gemm_config(
    m: int,
    k: int,
    n: int,
    num_groups: int,
    nopad: bool,
    dtype: str,
    tuned_configs: Optional[Dict[Tuple[int, int, int, int], Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
    """
    Tune configuration for GEMM operation with specific dimensions

    Args:
    m, k, n: Matrix dimensions
    num_groups: Number of groups
    tuned_configs: Pre-tuned configuration dictionary

    Returns:
    Best configuration or None (if already exists or tuning failed)
    """
    # Check if already tuned
    config_key = (m, k, n, num_groups, nopad)
    if config_key in tuned_configs:
        logger.info(f"Configuration for M={m}, K={k}, N={n}, num_groups={num_groups}, nopad={nopad} already tuned. Skipping...")
        return tuned_configs[config_key]
 
    logger.info(f"Tuning configuration for M={m}, K={k}, N={n}, num_groups={num_groups}, nopad={nopad}...")
    baseline_time = dispatch_tune_method(num_groups, nopad, dtype)(m, k, n, num_groups=num_groups)
    if baseline_time is None:
        logger.info(f"Failed to get baseline time for M={m}, K={k}, N={n}, num_groups={num_groups}")
        return None
    
    if dtype == "fp8":
        torch_dtype=torch.float8_e4m3fn
    elif dtype == "bf16":
        torch_dtype=torch.bfloat16
    else:
        torch_dtype=torch.int8

    configs = get_pre_assert_configs(m, n, k, num_groups, get_num_sms(), torch_dtype)
    best_time = baseline_time
    best_config = None

    for config in tqdm(configs):
        num_min_sms, best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages, best_smem_config = config
        time = dispatch_tune_method(num_groups, nopad, dtype)(m, k, n, config=config, num_groups=num_groups)
        if time is not None and time < best_time and (1 - (time / baseline_time)) > 0.01:
            best_time = time
            best_config = {
                "M": m,
                "K": k,
                "N": n,
                "num_groups": num_groups,
                "nopad": nopad,
                "dtype": dtype,
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
            logger.info(f"mnk: {m}x{n}x{k}, config: {config}, (groups:{num_groups}, nopad{nopad}) - Time: {time:.3f}us - Acc: {acc:.2f}")
    
    logger.info(f"{m}x{n}x{k} (groups:{num_groups}, nopad{nopad}), config: {config}, - best_time: {best_time:.3f}us - baseline_time: {baseline_time:.2f}")

    return best_config

import ray


@ray.remote(num_gpus=1)
class BenchmarkWorker:

    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        torch.cuda.manual_seed_all(0)
        self.seed = seed

    def tune(
        self, m: int, k: int, n: int, num_groups: int, nopad: bool, dtype: str, tuned_configs
    ) -> Dict[str, int]:
        best_config = tune_gemm_config(m, k, n, num_groups, nopad, dtype, tuned_configs)
        if best_config is None:
            logger.info(
                f"Warning: No valid configuration found for M={m}, K={k}, N={n}, num_groups={num_groups}, nopad={nopad}"
            )
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
    DEBUG_MODE = int(os.getenv("DEEPGEMM_TUNER_DEBUG_MODE", 0))

    # Load previously tuned configurations
    tuned_configs = load_tuned_configs(tuned_config)

    # Tune configurations for test cases
    best_configs = [x for x in tuned_configs.values()]
    best_configs = sorted(best_configs, key = lambda x : str(x))

    # init cluster

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='ray_output.log', level=logging.INFO)

    if DEBUG_MODE:
        best_configs = []
        for case in [tuple(test_case) + (tuned_configs,) for test_case in test_cases ]:
            best_config = tune_gemm_config(*case)
            if best_config is None:
                logger.info(
                    f"Warning: No valid configuration found for {case}"
                )
            else:
                logger.info(best_config)
                best_configs.append(best_config)
    else:
        ray.init()
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
        def get_timestamp():
            from datetime import datetime
            current_time = datetime.now()
            # "YY/MM/DD/HH/MM"
            formatted_time = current_time.strftime("%y-%m-%d-%H-%M")
            return formatted_time
        if out_of_box:
            save_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "configs",
            )
            assert os.path.exists(save_dir), "Deepgemm int8 configs dir "+save_dir+" do not exist, please upgrade to the latest version."

            save_path = os.path.join(save_dir, get_timestamp())

        else:
            save_path = get_timestamp()
        save_path = save_path + "-tp" + str(tp) +",device_name="+device_name+"-deepgemm_configs.json"
        save_tuned_configs(best_configs,  save_path)
        print(f"Tuning completed. Found {len(best_configs)} best configurations.")
    else:
        print("No configurations were tuned.")
    return save_path


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