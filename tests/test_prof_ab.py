"""Single-GPU A/B test: kernel WITHOUT profiler vs WITH profiler.
No P2P needed (world_size=1), isolates whether kstripe_profile_buf causes crash.

Usage: CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 python tests/test_prof_ab.py
"""
import os, sys, torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29542")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

torch.cuda.set_device(0)
dist.init_process_group(backend="nccl", init_method="env://")

import deep_gemm
from deep_gemm import preprocess_mxfp4_scales, mxfp4_quantize_to_sym_buffer, get_sym_buffer_size, dispatch_preprocess
from deep_gemm.jit_kernels.dispatch_fused_gemm import (
    dispatch_expert_preprocess, create_expert_preprocess_workspace,
    fused_dispatch_block_copy_gemm1_fp4, create_block_copy_buffers,
)
from deep_gemm.jit_kernels.gemm_fp4 import get_best_configs as get_best_configs_fp4
from deep_gemm.jit_kernels.utils import get_num_sms, ceil_div, GemmType

sys.path.insert(0, os.path.join(_repo_root, "tests"))
from test_fp4_core import quantize_fp4_torch

device = "cuda:0"
num_local_experts = 2
hidden = 2048
N = 512
max_tokens = 8
num_total_experts = 2
local_expert_start = 0
K_TILES_PER_FLAG = int(os.getenv("K_TILES_PER_FLAG", "4"))
NCB = int(os.getenv("NCB", "4"))

print(f"[Single-GPU A/B Test] No P2P, world_size=1")
print(f"  K_TILES_PER_FLAG={K_TILES_PER_FLAG}, NCB={NCB}")

# Weights
torch.manual_seed(200)
W = torch.randn(num_local_experts, N, hidden, dtype=torch.bfloat16, device=device) * 0.01
fp4_l, sc_l = [], []
for g in range(num_local_experts):
    d, s = quantize_fp4_torch(W[g])
    fp4_l.append(d)
    sc_l.append(s)
W_fp4 = torch.stack(fp4_l)
W_scale_u16 = preprocess_mxfp4_scales(scale=torch.stack(sc_l).clone())

# Sym buffer (single rank, so all data is local - no P2P)
symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
buf_size = get_sym_buffer_size(num_local_experts, num_total_experts, max_tokens, hidden)
buf = symm_mem.empty(buf_size, dtype=torch.int8, device=device)
handle = symm_mem.rendezvous(buf, group=dist.group.WORLD)
sym_buf = buf.view(torch.uint8)
sym_buf_addrs = torch.tensor(handle.buffer_ptrs, dtype=torch.int64, device=device)
sym_buf.zero_()
torch.cuda.synchronize()
print("  sym_buf allocated OK")

# Quantize
torch.manual_seed(77)
x = torch.randn(4, hidden, dtype=torch.bfloat16, device=device)
topk_ids = torch.zeros(4, 1, dtype=torch.int32, device=device)
topk_ids[0] = 0; topk_ids[1] = 0; topk_ids[2] = 1; topk_ids[3] = 1
mxfp4_quantize_to_sym_buffer(
    x, topk_ids, sym_buf,
    num_local_experts=num_total_experts,
    num_total_experts=num_total_experts,
    max_tokens_per_expert=max_tokens)
torch.cuda.synchronize()
print("  quant OK")

# Preprocess
_, _, _, _, shape_m = dispatch_preprocess(
    sym_buf_addrs, 0, 1, num_local_experts, num_total_experts,
    max_tokens, hidden, local_expert_start, block_m=64)
bc_configs = get_best_configs_fp4(
    shape_m, ceil_div(shape_m, num_local_experts), N, hidden // 2,
    num_local_experts, get_num_sms(), gemm_type=GemmType.GroupedNoPad)
block_m = bc_configs[1]
ws = create_expert_preprocess_workspace(num_local_experts, 1, max_tokens, block_m, device)
gl, ra, rs, sm, rc, blocks, shape_m = dispatch_expert_preprocess(
    sym_buf_addrs, 0, 1, num_local_experts, num_total_experts,
    max_tokens, hidden, local_expert_start, block_m, _workspace=ws)
print(f"  preprocess OK: shape_m={shape_m}, block_m={block_m}, blocks={blocks}")

# Buffers
bc_fp4, bc_flags = create_block_copy_buffers(num_local_experts, 1, max_tokens, hidden, block_m, device)
out = torch.zeros(shape_m, N, dtype=torch.bfloat16, device=device)
merged_sfa = torch.zeros(num_local_experts, dtype=torch.int64, device=device)

# ============ Test A: WITHOUT profiler ============
print("\n  [Test A] Running kernel WITHOUT kstripe_profile_buf...")
try:
    fused_dispatch_block_copy_gemm1_fp4(
        (W_fp4, W_scale_u16), out, gl, ra, rs, sm, rc,
        shape_m, max_tokens, 1,
        local_fp4_buf=bc_fp4, copy_ready_flags=bc_flags,
        num_copy_blocks=NCB, k_tiles_per_flag=K_TILES_PER_FLAG,
        configs=bc_configs, merged_sfa_addrs=merged_sfa)
    torch.cuda.synchronize()
    print("  [Test A] OK - kernel runs without profiler")
except Exception as e:
    print(f"  [Test A] FAILED: {e}")

# ============ Test B: WITH profiler ============
print("\n  [Test B] Running kernel WITH kstripe_profile_buf...")
try:
    prof = torch.zeros(3, dtype=torch.int64, device=device)
    fused_dispatch_block_copy_gemm1_fp4(
        (W_fp4, W_scale_u16), out, gl, ra, rs, sm, rc,
        shape_m, max_tokens, 1,
        local_fp4_buf=bc_fp4, copy_ready_flags=bc_flags,
        num_copy_blocks=NCB, k_tiles_per_flag=K_TILES_PER_FLAG,
        configs=bc_configs, merged_sfa_addrs=merged_sfa,
        kstripe_profile_buf=prof)
    torch.cuda.synchronize()
    print("  [Test B] OK - kernel runs with profiler")
    print(f"  prof[0] (wait cycles)  = {prof[0].item()}")
    print(f"  prof[1] (total cycles) = {prof[1].item()}")
    print(f"  prof[2] (num CTAs)     = {prof[2].item()}")
    if prof[1].item() > 0:
        print(f"  P2P exposure fraction  = {prof[0].item()/prof[1].item():.6f}")
except Exception as e:
    print(f"  [Test B] FAILED: {e}")

dist.destroy_process_group()
print("\n========== DONE ==========")
