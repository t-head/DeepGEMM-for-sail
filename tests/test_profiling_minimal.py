"""
Minimal P2P vs GEMM cycle profiling test.
Only tests kstripe_profile_buf without deep_ep dependency.

Usage:
    CUDA_VISIBLE_DEVICES=4,6 torchrun --nproc_per_node=2 --master_port=29535 \
        tests/test_profiling_minimal.py
"""

import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import deep_gemm
from deep_gemm import (
    preprocess_mxfp4_scales,
    mxfp4_quantize_to_sym_buffer,
    dispatch_preprocess,
    get_sym_buffer_size,
)
from deep_gemm.jit_kernels.dispatch_fused_gemm import (
    dispatch_expert_preprocess, create_expert_preprocess_workspace,
    fused_dispatch_block_copy_gemm1_fp4,
    create_block_copy_buffers,
)
from deep_gemm.jit_kernels.gemm_fp4 import get_best_configs as get_best_configs_fp4
from deep_gemm.jit_kernels.utils import get_num_sms, ceil_div, GemmType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fp4_core import quantize_fp4_torch


def quantize_grouped_fp4(tensor_3d):
    G = tensor_3d.shape[0]
    fp4_list, scale_raw_list = [], []
    for g in range(G):
        d, s = quantize_fp4_torch(tensor_3d[g])
        fp4_list.append(d)
        scale_raw_list.append(s)
    fp4 = torch.stack(fp4_list, dim=0)
    scale_raw = torch.stack(scale_raw_list, dim=0)
    scale_u16 = preprocess_mxfp4_scales(scale=scale_raw.clone())
    return fp4, scale_raw, scale_u16


def main():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    device = f'cuda:{local_rank}'

    if rank == 0:
        print(f"Minimal P2P vs GEMM Cycle Profiling Test")
        print(f"  world_size={world_size}, device={torch.cuda.get_device_name(local_rank)}")

    # Small config to minimize memory
    num_local_experts = 2
    hidden = 2048
    N = 512
    num_tokens = 4
    topk = 1
    max_tokens = 8
    num_total_experts = num_local_experts * world_size
    local_expert_start = rank * num_local_experts
    K_TILES_PER_FLAG = int(os.getenv('K_TILES_PER_FLAG', '4'))
    NCB = int(os.getenv('NCB', '4'))

    if rank == 0:
        print(f"  experts/rank={num_local_experts}, hidden={hidden}, N={N}")
        print(f"  K_TILES_PER_FLAG={K_TILES_PER_FLAG}, NCB={NCB}")

    # Generate data
    torch.manual_seed(77 + rank)
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    scores = torch.randn(num_tokens, num_total_experts, dtype=torch.float32, device=device)
    topk_ids = torch.topk(scores, topk, dim=-1, largest=True, sorted=False)[1].to(torch.int32)

    torch.manual_seed(200)
    W = torch.randn(num_local_experts, N, hidden, dtype=torch.bfloat16, device=device) * 0.01
    W_fp4, W_scale_raw, W_scale_u16 = quantize_grouped_fp4(W)

    if rank == 0:
        print(f"\n  Step 1: Allocating symmetric memory buffer...")

    # Symmetric memory
    buf_size = get_sym_buffer_size(num_local_experts, num_total_experts, max_tokens, hidden)
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    buf = symm_mem.empty(buf_size, dtype=torch.int8, device=device)
    handle = symm_mem.rendezvous(buf, group=dist.group.WORLD)
    sym_buf = buf.view(torch.uint8)
    sym_buf_addrs = torch.tensor(handle.buffer_ptrs, dtype=torch.int64, device=device)

    if rank == 0:
        print(f"  Step 1 OK: buf_size={buf_size} bytes")

    sym_buf.zero_()
    torch.cuda.synchronize()
    dist.barrier()

    # Quantize input to sym buffer
    if rank == 0:
        print(f"  Step 2: Quantizing to sym buffer...")

    mxfp4_quantize_to_sym_buffer(
        x, topk_ids, sym_buf,
        num_local_experts=num_total_experts,
        num_total_experts=num_total_experts,
        max_tokens_per_expert=max_tokens)
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        print(f"  Step 2 OK")

    # Preprocess
    if rank == 0:
        print(f"  Step 3: Expert preprocess...")

    block_m = 128  # safe default
    # Probe shape_m
    _, _, _, _, shape_m_probe = dispatch_preprocess(
        sym_buf_addrs, rank, world_size,
        num_local_experts, num_total_experts, max_tokens, hidden,
        local_expert_start, block_m=64)

    bc_configs = get_best_configs_fp4(
        shape_m_probe, ceil_div(shape_m_probe, num_local_experts),
        N, hidden // 2, num_local_experts, get_num_sms(),
        gemm_type=GemmType.GroupedNoPad)
    block_m = bc_configs[1]

    ws_expert = create_expert_preprocess_workspace(
        num_local_experts, world_size, max_tokens, block_m, device)
    gl, ra, rs, sm, rc, blocks, shape_m = dispatch_expert_preprocess(
        sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
        max_tokens, hidden, local_expert_start, block_m, _workspace=ws_expert)

    # Recompute if needed
    bc_configs2 = get_best_configs_fp4(
        shape_m, ceil_div(shape_m, num_local_experts),
        N, hidden // 2, num_local_experts, get_num_sms(),
        gemm_type=GemmType.GroupedNoPad)
    if bc_configs2[1] != block_m:
        block_m = bc_configs2[1]
        ws_expert = create_expert_preprocess_workspace(
            num_local_experts, world_size, max_tokens, block_m, device)
        gl, ra, rs, sm, rc, blocks, shape_m = dispatch_expert_preprocess(
            sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
            max_tokens, hidden, local_expert_start, block_m, _workspace=ws_expert)
    bc_configs = bc_configs2

    if rank == 0:
        print(f"  Step 3 OK: shape_m={shape_m}, block_m={block_m}, blocks={blocks}")

    # Build merged SFA (simple: use zeros placeholder for this minimal test)

    # Create block-copy buffers
    bc_fp4, bc_sfa, bc_flags = create_block_copy_buffers(
        num_local_experts, world_size, max_tokens, hidden, block_m, device)
    out_bc = torch.zeros(shape_m, N, dtype=torch.bfloat16, device=device)

    if rank == 0:
        print(f"\n  Step 4: Running block-copy kernel (warmup)...")

    # Warmup
    for _ in range(3):
        fused_dispatch_block_copy_gemm1_fp4(
            (W_fp4, W_scale_u16), out_bc, gl, ra, rs, sm, rc,
            shape_m, max_tokens, world_size,
            local_fp4_buf=bc_fp4, local_sfa_buf=bc_sfa, copy_ready_flags=bc_flags,
            num_copy_blocks=NCB, k_tiles_per_flag=K_TILES_PER_FLAG,
            configs=bc_configs,)
    torch.cuda.synchronize()

    if rank == 0:
        print(f"  Step 4 OK: kernel warmup completed")

    # Timing
    if rank == 0:
        print(f"\n  Step 5: Timing kernel...")
    NUM_ITERS = 20
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record()
    for _ in range(NUM_ITERS):
        fused_dispatch_block_copy_gemm1_fp4(
            (W_fp4, W_scale_u16), out_bc, gl, ra, rs, sm, rc,
            shape_m, max_tokens, world_size,
            local_fp4_buf=bc_fp4, local_sfa_buf=bc_sfa, copy_ready_flags=bc_flags,
            num_copy_blocks=NCB, k_tiles_per_flag=K_TILES_PER_FLAG,
            configs=bc_configs,)
    ev_e.record()
    torch.cuda.synchronize()
    kernel_ms = ev_s.elapsed_time(ev_e) / NUM_ITERS

    if rank == 0:
        print(f"  Step 5 OK: kernel_ms={kernel_ms:.3f} ms")

    # ---- P2P vs GEMM cycle profiling ----
    if rank == 0:
        print(f"\n  Step 6: P2P vs GEMM cycle profiling (kstripe_profile_buf)...")

    ks_prof_buf = torch.zeros(3, dtype=torch.int64, device=device)
    fused_dispatch_block_copy_gemm1_fp4(
        (W_fp4, W_scale_u16), out_bc, gl, ra, rs, sm, rc,
        shape_m, max_tokens, world_size,
        local_fp4_buf=bc_fp4, local_sfa_buf=bc_sfa, copy_ready_flags=bc_flags,
        num_copy_blocks=NCB, k_tiles_per_flag=K_TILES_PER_FLAG,
        configs=bc_configs,
        kstripe_profile_buf=ks_prof_buf)
    torch.cuda.synchronize()

    if rank == 0:
        total_wait = ks_prof_buf[0].item()
        total_main = ks_prof_buf[1].item()
        num_ctas = ks_prof_buf[2].item()
        print(f"  Step 6 OK")
        print(f"\n{'='*60}")
        print(f"P2P vs GEMM Cycle Profiling Results")
        print(f"{'='*60}")
        print(f"  K_TILES_PER_FLAG={K_TILES_PER_FLAG}, NCB={NCB}")
        print(f"  Kernel time: {kernel_ms:.3f} ms")
        print(f"  num_ctas (GEMM CTA invocations): {num_ctas}")
        if num_ctas > 0 and total_main > 0:
            exposure = total_wait / total_main
            avg_wait = total_wait / num_ctas
            avg_main = total_main / num_ctas
            avg_compute = avg_main - avg_wait
            print(f"  Total P2P wait cycles:    {total_wait:>14,}")
            print(f"  Total mainloop cycles:    {total_main:>14,}")
            print(f"  Avg P2P wait/CTA:         {avg_wait:>14,.0f} cycles")
            print(f"  Avg GEMM compute/CTA:     {avg_compute:>14,.0f} cycles")
            print(f"  P2P exposure fraction:    {exposure:.4f}")
            print(f"    (0 = P2P fully hidden by compute)")
            print(f"    (1 = GEMM entirely stalled on P2P)")
            if K_TILES_PER_FLAG == 0:
                print(f"  NOTE: K_TILES_PER_FLAG=0 -> single wait at entry, not per-stripe")
        else:
            print(f"  WARNING: No profiling data collected (total_main={total_main}, num_ctas={num_ctas})")
            if K_TILES_PER_FLAG == 0:
                print(f"  HINT: kstripe profiling requires K_TILES_PER_FLAG > 0")
        print(f"{'='*60}")

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("\nDone!")


if __name__ == '__main__':
    main()
