"""
Block-Copy Fused Dispatch GEMM1 Multi-GPU Test

Tests block-copy fused dispatch: dedicated copy blocks do P2P data movement
(remote → local HBM) while remaining blocks run standard GEMM from local HBM.

  Test 1: Correctness (random routing → compare vs CPU reference)
  Test 2: Performance (ncb sweep + non-fused comparison)

Usage:
    # 2-GPU correctness + perf
    FULL_CORRECTNESS=1 PERF_VERBOSE=1 CUDA_VISIBLE_DEVICES=1,2 torchrun \\
        --nproc_per_node=2 --master_port=29530 \\
        tests/test_block_copy_gemm1_multi_gpu.py --verbose

    # 4-GPU
    FULL_CORRECTNESS=1 PERF_VERBOSE=1 CUDA_VISIBLE_DEVICES=1,2,5,6 torchrun \\
        --nproc_per_node=4 --master_port=29531 \\
        tests/test_block_copy_gemm1_multi_gpu.py --verbose

Environment variables:
    TEST_CONFIG=small|prod  Test configuration (default: prod)
    FULL_CORRECTNESS=1      Enable per-expert Python reference check (default: 0)
    PERF_VERBOSE=1          Print detailed performance breakdown
    SKIP_CORRECTNESS=1      Skip Test 1, run only performance
"""

import os
import sys

# Ensure we import deep_gemm from THIS repo, not the pip-installed original
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

import itertools
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


# ============================================================
# Configuration
# ============================================================

class TestConfig:
    def __init__(self, name='prod'):
        if name == 'small':
            self.num_local_experts = 2
            self.hidden = 2048
            self.N = 512
            self.num_tokens = 4
            self.topk = 1
            self.max_tokens = 8
        elif name == 'prod':
            self.num_local_experts = 12
            self.hidden = 7168
            self.N = 6144
            self.num_tokens = 256
            self.topk = 6
            self.max_tokens = 256
        else:
            raise ValueError(f"Unknown config: {name}")
        self.name = name

    def num_total_experts(self, world_size):
        return self.num_local_experts * world_size

    def local_expert_start(self, rank):
        return rank * self.num_local_experts

    def __repr__(self):
        return (f"TestConfig(name={self.name}, experts/rank={self.num_local_experts}, "
                f"hidden={self.hidden}, N={self.N}, tokens/rank={self.num_tokens}, "
                f"topk={self.topk}, max_tokens={self.max_tokens})")


CONFIG = TestConfig(os.getenv('TEST_CONFIG', 'prod'))
FULL_CORRECTNESS = int(os.getenv('FULL_CORRECTNESS', '0'))
SKIP_CORRECTNESS = int(os.getenv('SKIP_CORRECTNESS', '0'))
SKIP_ISOLATION = int(os.getenv('SKIP_ISOLATION', '0'))
COL_MAJOR_SCALE = int(os.getenv('COL_MAJOR_SCALE', '1'))
PAD_ALIGN = 1  # no padding; matches dispatch_preprocess.cuh (removed 8-alignment)


# ============================================================
# Imports
# ============================================================

import deep_gemm
import deep_ep
from deep_gemm import (
    calc_diff, preprocess_mxfp4_scales,
    mxfp4_quantize_to_sym_buffer,
    dispatch_preprocess,
    get_sym_buffer_size,
)
from deep_gemm.jit_kernels.dispatch_fused_gemm import (
    create_preprocess_workspace,
    dispatch_expert_preprocess, create_expert_preprocess_workspace,
    fused_dispatch_block_copy_gemm1_fp4,
    create_block_copy_buffers,
)
from deep_gemm.jit_kernels.gemm_fp4 import get_best_configs as get_best_configs_fp4
from deep_gemm.jit_kernels.utils import get_num_sms, ceil_div, GemmType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fp4_core import quantize_fp4_torch, dequantize_fp4_torch


# ============================================================
# Helpers
# ============================================================

def generate_test_input(num_tokens, hidden, device='cuda'):
    """Generate test input with wide dynamic range to stress-test quantization."""
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    x[:, ::32] *= 50.0
    x[:, 1::32] *= 0.01
    if num_tokens >= 4:
        x[0] = torch.full((hidden,), 60.0, dtype=torch.bfloat16, device=device)
        x[0, ::2] *= -1
    return x


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


def get_gemm_configs(shape_m, num_groups, n, k):
    """Full GEMM config used by BOTH preprocess (block_m) and the fused kernel.

    Returning the whole tuple and passing it to fused_dispatch guarantees the
    preprocess block_m == kernel block_m (a mismatch silently zeroes 2nd+ blocks).
    FORCE_EXPECTED_M lets you steer block_m for testing (e.g. 129 -> block_m=256).
    """
    expected_m = ceil_div(shape_m, num_groups)
    force = int(os.getenv('FORCE_EXPECTED_M', '0'))
    if force > 0:
        expected_m = force
    num_sms = get_num_sms()
    return get_best_configs_fp4(
        shape_m, expected_m, n, k, num_groups, num_sms,
        gemm_type=GemmType.GroupedNoPad)


def get_gemm_block_m(shape_m, num_groups, n, k, num_ranks=0):
    return get_gemm_configs(shape_m, num_groups, n, k)[1]


def init_dist():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    return rank, local_rank, world_size, dist.group.WORLD


def alloc_sym_buffer(buf_size, device, group):
    buf = symm_mem.empty(buf_size, dtype=torch.int8, device=device)
    handle = symm_mem.rendezvous(buf, group=group)
    addrs = torch.tensor(handle.buffer_ptrs, dtype=torch.int64, device=device)
    return buf.view(torch.uint8), addrs, handle


def create_ep_buffer(group, num_local_experts, num_tokens, hidden, world_size, num_total_experts):
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        num_tokens, hidden, world_size, num_total_experts)
    return deep_ep.Buffer(
        group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
        num_qps_per_rank=num_local_experts, explicitly_destroy=True)


def _align_up(x, align):
    return (x + align - 1) & ~(align - 1) if align > 1 else x


def nonfused_dispatch(ep_buffer, x, topk_ids_i64, num_tokens, num_total_experts):
    """Run DeepEP low_latency_dispatch with optional column-major scale."""
    kwargs = dict(use_mxfp4=True, quant_size=32)
    if COL_MAJOR_SCALE:
        kwargs['mxfp4_scale_row_major'] = False
    (pf, ps), pc, eh, ee, ehk = ep_buffer.low_latency_dispatch(
        x, topk_ids_i64, num_tokens, num_total_experts, **kwargs)
    if COL_MAJOR_SCALE:
        lhs_sc = ps
    else:
        lhs_sc = preprocess_mxfp4_scales(ps.contiguous().view(torch.uint8))
    return pf, lhs_sc, pc, eh, ee, ehk


def get_expert_token_counts(all_sym_bufs, ge, world_size):
    """Return list of per-rank token counts for global expert ge."""
    return [all_sym_bufs[r][ge * 4: ge * 4 + 4].view(torch.int32).item()
            for r in range(world_size)]


def build_merged_sfa(all_sym_bufs, num_local_experts, local_expert_start,
                     num_total_experts, max_tokens, hidden, world_size, device,
                     pad_align=None):
    """Build merged SFA buffer from all-gathered symmetric buffers.

    The CuTe mainloop creates SFA tensors with LayoutLeft (column-major):
    shape (M, K_scale), stride (1, M) where M = padded_total per expert.
    So the SFA data must be packed column-major with stride = padded_total.

    Returns (merged_sfa, merged_sfa_addrs) tensors.
    """
    if pad_align is None:
        pad_align = PAD_ALIGN
    k_blocks = (hidden + 31) // 32
    k_scale_blocks = (k_blocks + 1) // 2
    scale_elems_per_expert_max = k_scale_blocks * max_tokens
    metadata_bytes = ((num_total_experts * 4 + 15) // 16) * 16
    fp4_per_expert = max_tokens * (hidden // 2)
    fp4_region_size = num_total_experts * fp4_per_expert

    expert_padded_totals = []
    for le in range(num_local_experts):
        ge = local_expert_start + le
        counts = get_expert_token_counts(all_sym_bufs, ge, world_size)
        padded = sum(_align_up(c, pad_align) for c in counts)
        expert_padded_totals.append(padded)

    total_sfa_elems = sum(pt * k_scale_blocks for pt in expert_padded_totals)
    merged_sfa = torch.zeros(total_sfa_elems, dtype=torch.uint16, device=device)
    merged_sfa_addrs = torch.zeros(num_local_experts, dtype=torch.int64, device=device)

    offset = 0
    for le in range(num_local_experts):
        ge = local_expert_start + le
        pt = expert_padded_totals[le]
        merged_sfa_addrs[le] = merged_sfa.data_ptr() + offset * 2

        if pt == 0:
            offset += pt * k_scale_blocks
            continue

        expert_sfa = merged_sfa[offset:offset + pt * k_scale_blocks].view(k_scale_blocks, pt)

        merged_token_pos = 0
        for r in range(world_size):
            count_r = all_sym_bufs[r][ge * 4: ge * 4 + 4].view(torch.int32).item()
            if count_r == 0:
                continue
            scale_off_r = metadata_bytes + fp4_region_size + ge * scale_elems_per_expert_max * 2
            src_scale = all_sym_bufs[r][scale_off_r:scale_off_r + scale_elems_per_expert_max * 2] \
                .view(torch.uint16).view(k_scale_blocks, max_tokens)
            expert_sfa[:, merged_token_pos:merged_token_pos + count_r] = src_scale[:, :count_r]
            merged_token_pos += _align_up(count_r, pad_align)

        offset += pt * k_scale_blocks

    return merged_sfa, merged_sfa_addrs


# ============================================================
# Test 1: Correctness (Random Routing)
# ============================================================

def test_correctness(rank, world_size, group, device):
    cfg = CONFIG
    num_total_experts = cfg.num_total_experts(world_size)
    num_local_experts = cfg.num_local_experts
    local_expert_start = cfg.local_expert_start(rank)
    hidden = cfg.hidden
    N = cfg.N
    num_tokens = cfg.num_tokens
    topk = cfg.topk
    max_tokens = cfg.max_tokens

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Test 1: Correctness (Block-Copy, Random Routing)")
        print(f"  Config: {cfg}")
        print(f"  world_size={world_size}")
        print(f"{'='*60}\n")

    # Random routing
    torch.manual_seed(77 + rank)
    x = generate_test_input(num_tokens, hidden, device)
    scores = torch.randn(num_tokens, num_total_experts, dtype=torch.float32, device=device)
    topk_ids = torch.topk(scores, topk, dim=-1, largest=True, sorted=False)[1].to(torch.int32)

    # Weights
    torch.manual_seed(200)
    W = torch.randn(num_local_experts, N, hidden, dtype=torch.bfloat16, device=device) * 0.01
    W_fp4, W_scale_raw, W_scale_u16 = quantize_grouped_fp4(W)

    # ---- Fused path: symmetric memory ----
    buf_size = get_sym_buffer_size(num_local_experts, num_total_experts, max_tokens, hidden)
    sym_buf, sym_buf_addrs, sym_handle = alloc_sym_buffer(buf_size, device, group)

    sym_buf.zero_()
    torch.cuda.synchronize()
    dist.barrier()

    mxfp4_quantize_to_sym_buffer(
        x, topk_ids, sym_buf,
        num_local_experts=num_total_experts,
        num_total_experts=num_total_experts,
        max_tokens_per_expert=max_tokens)
    torch.cuda.synchronize()
    dist.barrier()

    # Two-pass block_m selection: probe → expert preprocess → recompute if needed
    _, _, _, _, shape_m_probe = dispatch_preprocess(
        sym_buf_addrs, rank, world_size,
        num_local_experts, num_total_experts, max_tokens, hidden,
        local_expert_start, block_m=64)

    bc_configs = get_gemm_configs(shape_m_probe, num_local_experts, N, hidden // 2)
    block_m = bc_configs[1]

    ws_expert = create_expert_preprocess_workspace(
        num_local_experts, world_size, max_tokens, block_m, device)
    gl, ra, rs, sm, rc, blocks, shape_m = dispatch_expert_preprocess(
        sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
        max_tokens, hidden, local_expert_start, block_m, _workspace=ws_expert)

    # Recompute configs from actual shape_m; re-run preprocess if block_m changed
    bc_configs2 = get_gemm_configs(shape_m, num_local_experts, N, hidden // 2)
    if bc_configs2[1] != block_m:
        if rank == 0:
            print(f"  block_m changed: {block_m} -> {bc_configs2[1]} (shape_m_probe={shape_m_probe}, shape_m={shape_m})")
        bc_configs = bc_configs2
        block_m = bc_configs[1]
        ws_expert = create_expert_preprocess_workspace(
            num_local_experts, world_size, max_tokens, block_m, device)
        gl, ra, rs, sm, rc, blocks, shape_m = dispatch_expert_preprocess(
            sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
            max_tokens, hidden, local_expert_start, block_m, _workspace=ws_expert)
    else:
        bc_configs = bc_configs2
    if rank == 0:
        print(f"  block_m={block_m}, shape_m={shape_m}, blocks={blocks}")

    # All-gather sym bufs for reference + merged SFA construction
    all_sym_bufs = [torch.zeros_like(sym_buf) for _ in range(world_size)]
    dist.all_gather(all_sym_bufs, sym_buf)

    if rank == 0:
        for le in range(num_local_experts):
            ge = local_expert_start + le
            counts = get_expert_token_counts(all_sym_bufs, ge, world_size)
            padded_total = sum(_align_up(c, PAD_ALIGN) for c in counts)
            mblocks = (padded_total + block_m - 1) // block_m
            print(f"    Expert {ge}: counts={counts} total={sum(counts)} padded={padded_total} mblocks={mblocks}")
    dist.barrier()

    # Build merged SFA
    merged_sfa, merged_sfa_addrs = build_merged_sfa(
        all_sym_bufs, num_local_experts, local_expert_start,
        num_total_experts, max_tokens, hidden, world_size, device)

    # ---- Run block-copy GEMM ----
    k_half = hidden // 2
    NUM_COPY_BLOCKS = int(os.getenv('NCB', '8'))
    K_TILES_PER_FLAG = int(os.getenv('K_TILES_PER_FLAG', '0'))
    bc_fp4, bc_flags = create_block_copy_buffers(
        num_local_experts, world_size, max_tokens, hidden, block_m, device)
    out_bc = torch.zeros(shape_m, N, dtype=torch.bfloat16, device=device)

    if rank == 0:
        print(f"\n  Running block-copy GEMM (num_copy_blocks={NUM_COPY_BLOCKS}, k_tiles_per_flag={K_TILES_PER_FLAG})")
    fused_dispatch_block_copy_gemm1_fp4(
        (W_fp4, W_scale_u16), out_bc, gl, ra, rs, sm, rc,
        shape_m, max_tokens, world_size,
        local_fp4_buf=bc_fp4, copy_ready_flags=bc_flags,
        num_copy_blocks=NUM_COPY_BLOCKS, k_tiles_per_flag=K_TILES_PER_FLAG,
        configs=bc_configs,
        merged_sfa_addrs=merged_sfa_addrs)
    torch.cuda.synchronize()

    # ---- Non-fused baseline (DeepEP + standard GEMM) ----
    topk_ids_i64 = topk_ids.to(torch.int64)
    ep_buffer = create_ep_buffer(group, num_local_experts, num_tokens, hidden,
                                 world_size, num_total_experts)
    pf, lhs_sc, pc, eh, ee, ehk = nonfused_dispatch(
        ep_buffer, x, topk_ids_i64, num_tokens, num_total_experts)
    mm = max(int(pc.max().item()), 1)
    nf_block_m = int(os.getenv('NF_BLOCK_M', '128'))
    if nf_block_m > 0:
        nf_configs = get_best_configs_fp4(
            mm * num_local_experts, mm, N, hidden // 2,
            num_local_experts, get_num_sms(), gemm_type=GemmType.GroupedNoPad)
    else:
        nf_configs = None
    nf_out = torch.empty(num_local_experts, pf.shape[1], N,
                         dtype=torch.bfloat16, device=device)
    deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
        (pf.contiguous(), lhs_sc),
        (W_fp4, W_scale_u16),
        None, nf_out, pc.to(torch.int32), mm,
        configs=nf_configs)
    torch.cuda.synchronize()

    # ---- Compare block-copy vs non-fused + CPU reference ----
    metadata_bytes = ((num_total_experts * 4 + 15) // 16) * 16
    fp4_per_expert = max_tokens * (hidden // 2)
    fp4_region_size = num_total_experts * fp4_per_expert
    k_blocks = (hidden + 31) // 32
    k_scale_blocks = (k_blocks + 1) // 2
    scale_elems_per_expert = k_scale_blocks * max_tokens

    # Parse expert layout from grouped_layout
    gl_cpu = gl.cpu()
    expert_seg = {}
    for b in range(blocks):
        e = gl_cpu[4 + b * 4 + 0].item()
        cnt = gl_cpu[4 + b * 4 + 1].item()
        bm_off = gl_cpu[4 + b * 4 + 3].item()
        if e not in expert_seg:
            expert_seg[e] = (bm_off, cnt)

    all_passed = True

    if rank == 0:
        print(f"\n  {'Expert':<10s} {'M':>5s} {'BC vs NF':>10s} {'BC vs CPU':>12s} {'(elem)':>8s} {'Status':>8s}")

    for le in range(num_local_experts):
        ge = local_expert_start + le
        m_nf = int(pc[le].item())

        # Count fused tokens from sym buf metadata
        counts_per_rank = get_expert_token_counts(all_sym_bufs, ge, world_size)
        m_fused = sum(counts_per_rank)

        if m_fused == 0:
            if rank == 0:
                print(f"  Expert {ge:<4d}  {0:>5d}  {'—':>10s} {'—':>12s} {'—':>8s} {'SKIP':>8s}")
            continue

        # Token count consistency check
        if m_fused != m_nf:
            if rank == 0:
                print(f"  Expert {ge}: token count mismatch: fused={m_fused}, non-fused={m_nf}")
            all_passed = False
            continue

        # Non-fused output for this expert
        nf_e = nf_out[le, :m_nf].cpu()

        # Block-copy output for this expert
        if le in expert_seg:
            ebm, ecnt = expert_seg[le]
            bc_e = out_bc[ebm:ebm + ecnt].cpu()
        else:
            all_passed = False
            if rank == 0:
                print(f"  Expert {ge:<4d}  {m_fused:>5d}  {'—':>10s} {'—':>12s} {'—':>8s} {'NOSEG':>8s}")
            continue

        # Extract non-padding rows from block-copy output for sorted-norm comparison
        bc_real_rows = []
        pos = 0
        for r in range(world_size):
            cr = counts_per_rank[r]
            if cr > 0:
                bc_real_rows.append(bc_e[pos:pos + cr])
            pos += _align_up(cr, PAD_ALIGN)
        bc_real = torch.cat(bc_real_rows, dim=0) if bc_real_rows else torch.zeros(0, N)

        # Sorted-norm comparison (BC vs NF) — secondary check (token orders differ)
        bc_norms = bc_real.float().norm(dim=-1).sort().values
        nf_norms = nf_e.float().norm(dim=-1).sort().values
        min_len = min(len(bc_norms), len(nf_norms))
        diff_bc_nf = calc_diff(bc_norms[:min_len], nf_norms[:min_len]) if min_len > 0 else 0.0

        # CPU reference with same padding layout — primary element-wise check
        diff_bc_cpu_norm = float('nan')
        diff_bc_cpu_elem = float('nan')
        if FULL_CORRECTNESS and le in expert_seg:
            # Build A matrix with same padding layout as block-copy
            A_rows = []
            for r in range(world_size):
                count_r = counts_per_rank[r]
                if count_r == 0:
                    pad_rows = _align_up(0, PAD_ALIGN)
                    if pad_rows > 0:
                        A_rows.append(torch.zeros(pad_rows, hidden, dtype=torch.bfloat16))
                    continue
                fp4_off_r = metadata_bytes + ge * fp4_per_expert
                fp4_r = all_sym_bufs[r][fp4_off_r:fp4_off_r + count_r * (hidden // 2)].cpu()
                fp4_r = fp4_r.view(count_r, hidden // 2)
                scale_off_r = metadata_bytes + fp4_region_size + ge * scale_elems_per_expert * 2
                src_scale_raw = all_sym_bufs[r][scale_off_r:scale_off_r + scale_elems_per_expert * 2]
                src_scale = src_scale_raw.cpu().view(torch.uint16).view(k_scale_blocks, max_tokens)
                scale_per_token = src_scale[:, :count_r].T.contiguous()
                scale_u8 = scale_per_token.view(torch.uint8).view(count_r, k_blocks)
                A_rank = dequantize_fp4_torch(fp4_r, scale_u8)
                A_rows.append(A_rank)
                # Add padding rows
                pad_count = _align_up(count_r, PAD_ALIGN) - count_r
                if pad_count > 0:
                    A_rows.append(torch.zeros(pad_count, hidden, dtype=torch.bfloat16))

            A_padded = torch.cat(A_rows, dim=0)
            W_bf16 = dequantize_fp4_torch(W_fp4[le].cpu(), W_scale_raw[le].cpu())
            ref_out = (A_padded.float() @ W_bf16.float().T).bfloat16()

            # Element-wise comparison (with padding layout matched)
            cmp_len = min(len(ref_out), len(bc_e))
            diff_bc_cpu_elem = calc_diff(bc_e[:cmp_len].float(), ref_out[:cmp_len].float())

            # Sorted-norm comparison as secondary metric
            ref_norms = ref_out.float().norm(dim=-1).sort().values
            bc_norms_all = bc_e.float().norm(dim=-1).sort().values
            cmp_n = min(len(bc_norms_all), len(ref_norms))
            diff_bc_cpu_norm = calc_diff(bc_norms_all[:cmp_n], ref_norms[:cmp_n]) if cmp_n > 0 else 0.0

        ok = diff_bc_nf < 0.01
        if FULL_CORRECTNESS and diff_bc_cpu_elem == diff_bc_cpu_elem:
            ok = ok and diff_bc_cpu_elem < 0.002
        if not ok:
            all_passed = False

        if rank == 0:
            status = 'PASSED' if ok else 'FAILED'
            cpu_n_str = f"{diff_bc_cpu_norm:.6f}" if diff_bc_cpu_norm == diff_bc_cpu_norm else "—"
            cpu_e_str = f"{diff_bc_cpu_elem:.6f}" if diff_bc_cpu_elem == diff_bc_cpu_elem else "—"
            print(f"  Expert {ge:<4d}  {m_fused:>5d}  {diff_bc_nf:>10.6f} {cpu_n_str:>12s} {cpu_e_str:>8s} {status:>8s}")

    ep_buffer.destroy()

    # Multi-rank correctness aggregation
    pass_tensor = torch.tensor([1 if all_passed else 0], dtype=torch.int32, device=device)
    dist.all_reduce(pass_tensor, op=dist.ReduceOp.MIN)
    all_passed = pass_tensor.item() == 1

    if rank == 0:
        print(f"\n  Test 1: {'PASSED' if all_passed else 'FAILED'} (all ranks)")
    return all_passed


# ============================================================
# Test 2: Performance (NCB Sweep + Non-fused Comparison)
# ============================================================

def test_performance(rank, world_size, group, device):
    cfg = CONFIG
    num_total_experts = cfg.num_total_experts(world_size)
    num_local_experts = cfg.num_local_experts
    local_expert_start = cfg.local_expert_start(rank)
    hidden = cfg.hidden
    N = cfg.N
    num_tokens = cfg.num_tokens
    topk = cfg.topk
    max_tokens = cfg.max_tokens
    num_warmup = 3
    num_iters = 20
    verbose = os.environ.get('PERF_VERBOSE', '0') != '0'
    K_TILES_PER_FLAG = int(os.getenv('K_TILES_PER_FLAG', '0'))

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Test 2: Performance Benchmarking (Block-Copy)")
        print(f"  Config: {cfg}")
        print(f"  world_size={world_size}")
        print(f"{'='*60}\n")

    torch.manual_seed(99 + rank)
    x = generate_test_input(num_tokens, hidden, device)

    # Uniform token distribution: round-robin assignment so each expert
    # gets roughly num_tokens * topk / num_total_experts tokens per rank.
    # Deterministic across runs and uniform across all world_size values.
    token_idx = torch.arange(num_tokens, device=device, dtype=torch.int32)
    topk_ids = torch.stack(
        [(token_idx * topk + j) % num_total_experts for j in range(topk)], dim=1)
    topk_ids_i64 = topk_ids.to(torch.int64)

    torch.manual_seed(200)
    W = torch.randn(num_local_experts, N, hidden, dtype=torch.bfloat16, device=device) * 0.01
    W_fp4, W_scale_raw, W_scale_u16 = quantize_grouped_fp4(W)

    # ---- Fused path setup ----
    buf_size = get_sym_buffer_size(num_local_experts, num_total_experts, max_tokens, hidden)
    sym_buf, sym_buf_addrs, sym_handle = alloc_sym_buffer(buf_size, device, group)
    # One-time zero of the atomic-arrival slot region (last 128B); symm_mem.empty() is
    # uninitialized and warmup uses generation=0 (no push), so slots must start < any g.
    sym_buf[buf_size - 128:].zero_()
    dist.barrier(); torch.cuda.synchronize()
    metadata_size = ((num_total_experts * 4 + 15) // 16) * 16

    # Probe block_m
    mxfp4_quantize_to_sym_buffer(
        x, topk_ids, sym_buf,
        num_local_experts=num_total_experts,
        num_total_experts=num_total_experts,
        max_tokens_per_expert=max_tokens)
    torch.cuda.synchronize()
    dist.barrier()
    _, _, _, _, shape_m_probe = dispatch_preprocess(
        sym_buf_addrs, rank, world_size,
        num_local_experts, num_total_experts, max_tokens, hidden,
        local_expert_start, block_m=64)

    block_m = get_gemm_block_m(shape_m_probe, num_local_experts, N, hidden // 2, num_ranks=world_size)
    if rank == 0:
        print(f"  block_m={block_m}, shape_m_probe={shape_m_probe}")

    # Two-pass block_m: run expert preprocess to get actual shape_m, recompute if needed
    ws_expert = create_expert_preprocess_workspace(
        num_local_experts, world_size, max_tokens, block_m, device)
    gl_e, ra_e, rs_e, sm_e, rc_e, blocks_e, expert_shape_m = dispatch_expert_preprocess(
        sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
        max_tokens, hidden, local_expert_start, block_m, _workspace=ws_expert)
    block_m2 = get_gemm_block_m(expert_shape_m, num_local_experts, N, hidden // 2, num_ranks=world_size)
    if block_m2 != block_m:
        if rank == 0:
            print(f"  block_m changed: {block_m} -> {block_m2} (shape_m_probe={shape_m_probe}, shape_m={expert_shape_m})")
        block_m = block_m2
        ws_expert = create_expert_preprocess_workspace(
            num_local_experts, world_size, max_tokens, block_m, device)
        gl_e, ra_e, rs_e, sm_e, rc_e, blocks_e, expert_shape_m = dispatch_expert_preprocess(
            sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
            max_tokens, hidden, local_expert_start, block_m, _workspace=ws_expert)

    if rank == 0:
        k_half = hidden // 2
        # Block-copy fused config (GroupedNoPad)
        bc_expected_m = ceil_div(expert_shape_m, num_local_experts)
        bc_sms, bc_bm, bc_bn, bc_bk, bc_wm, bc_wn, bc_stages, bc_smem = get_best_configs_fp4(
            expert_shape_m, bc_expected_m, N, k_half, num_local_experts, get_num_sms(),
            gemm_type=GemmType.GroupedNoPad)
        print(f"  Block-copy fused config (GroupedNoPad): block_m={bc_bm}, block_n={bc_bn}, block_k={bc_bk}, "
              f"warp_m={bc_wm}, warp_n={bc_wn}, stages={bc_stages}, num_sms={bc_sms}, smem={bc_smem}")
        # Non-fused config (GroupedMasked)
        nf_sms, nf_bm, nf_bn, nf_bk, nf_wm, nf_wn, nf_stages, nf_smem = get_best_configs_fp4(
            max_tokens * num_local_experts, max_tokens, N, k_half, num_local_experts, get_num_sms(),
            gemm_type=GemmType.GroupedMasked)
        print(f"  Non-fused config (GroupedMasked):       block_m={nf_bm}, block_n={nf_bn}, block_k={nf_bk}, "
              f"warp_m={nf_wm}, warp_n={nf_wn}, stages={nf_stages}, num_sms={nf_sms}, smem={nf_smem}")

    # ---- Non-fused path setup ----
    ep_buffer = create_ep_buffer(group, num_local_experts, num_tokens, hidden,
                                 world_size, num_total_experts)

    # ---- Warmup ----
    for _ in range(num_warmup):
        mxfp4_quantize_to_sym_buffer(
            x, topk_ids, sym_buf,
            num_local_experts=num_total_experts,
            num_total_experts=num_total_experts,
            max_tokens_per_expert=max_tokens)
        torch.cuda.synchronize()
        dist.barrier()
        dispatch_expert_preprocess(
            sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
            max_tokens, hidden, local_expert_start, block_m, _workspace=ws_expert)

        # Non-fused warmup
        pf, lhs_sc, pc, _, _, _ = nonfused_dispatch(
            ep_buffer, x, topk_ids_i64, num_tokens, num_total_experts)
        mm = max(int(pc.max().item()), 1)
        nf_out = torch.empty(num_local_experts, pf.shape[1], N,
                             dtype=torch.bfloat16, device=device)
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
            (pf.contiguous(), lhs_sc),
            (W_fp4, W_scale_u16),
            None, nf_out, pc.to(torch.int32), mm)
        torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        # Print fused dispatch (block-copy) GEMM config
        bc_k_half = hidden // 2
        bc_expected_m = ceil_div(expert_shape_m, num_local_experts)
        bc_sms, bc_bm, bc_bn, bc_bk, bc_wm, bc_wn, bc_stages, bc_smem = get_best_configs_fp4(
            expert_shape_m, bc_expected_m, N, bc_k_half, num_local_experts, get_num_sms(),
            gemm_type=GemmType.GroupedNoPad)
        print(f"  expert_shape_m={expert_shape_m}, blocks={blocks_e}, block_m={block_m}")
        print(f"  Block-copy config: block_m={bc_bm}, block_n={bc_bn}, block_k={bc_bk}, "
              f"warp_m={bc_wm}, warp_n={bc_wn}, stages={bc_stages}, num_sms={bc_sms}, smem={bc_smem}")

    # ---- Build merged SFA for block-copy ----
    all_sym_bufs = [torch.zeros_like(sym_buf) for _ in range(world_size)]
    dist.all_gather(all_sym_bufs, sym_buf)
    dist.barrier()

    merged_sfa, merged_sfa_addrs = build_merged_sfa(
        all_sym_bufs, num_local_experts, local_expert_start,
        num_total_experts, max_tokens, hidden, world_size, device)

    # ---- Benchmark non-fused (pipeline throughput) ----
    fixed_expected_m = max(int(pc.max().item()), 1)
    nf_block_m = int(os.getenv('NF_BLOCK_M', '128'))
    if nf_block_m > 0:
        nf_configs = get_best_configs_fp4(
            fixed_expected_m * num_local_experts, fixed_expected_m, N, hidden // 2,
            num_local_experts, get_num_sms(), gemm_type=GemmType.GroupedNoPad)
    else:
        nf_configs = None
    if rank == 0:
        if nf_configs:
            nf_sms, nf_bm, nf_bn, nf_bk, nf_wm, nf_wn, nf_stages, nf_smem = nf_configs
        else:
            nf_sms, nf_bm, nf_bn, nf_bk, nf_wm, nf_wn, nf_stages, nf_smem = get_best_configs_fp4(
                fixed_expected_m * num_local_experts, fixed_expected_m, N, hidden // 2,
                num_local_experts, get_num_sms(), gemm_type=GemmType.GroupedMasked)
        print(f"\n  Non-fused config (expected_m={fixed_expected_m}): block_m={nf_bm}, block_n={nf_bn}, block_k={nf_bk}, "
              f"warp_m={nf_wm}, warp_n={nf_wn}, stages={nf_stages}, num_sms={nf_sms}, smem={nf_smem}")
    nf_pipe_out = torch.empty(num_local_experts, pf.shape[1], N,
                              dtype=torch.bfloat16, device=device)
    for _ in range(num_warmup):
        pf2, lhs2, pc2, _, _, _ = nonfused_dispatch(
            ep_buffer, x, topk_ids_i64, num_tokens, num_total_experts)
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
            (pf2.contiguous(), lhs2),
            (W_fp4, W_scale_u16),
            None, nf_pipe_out, pc2.to(torch.int32), fixed_expected_m,
            configs=nf_configs)
    torch.cuda.synchronize()

    ev_nfp = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    pipe_iters = num_iters * 2
    ev_nfp[0].record()
    for _ in range(pipe_iters):
        pf2, lhs2, pc2, _, _, _ = nonfused_dispatch(
            ep_buffer, x, topk_ids_i64, num_tokens, num_total_experts)
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
            (pf2.contiguous(), lhs2),
            (W_fp4, W_scale_u16),
            None, nf_pipe_out, pc2.to(torch.int32), fixed_expected_m,
            configs=nf_configs)
    ev_nfp[1].record()
    torch.cuda.synchronize()
    nf_pipeline_ms = ev_nfp[0].elapsed_time(ev_nfp[1]) / pipe_iters

    # ---- Non-fused GEMM-only (for reference) ----
    dist.barrier()
    pf_go, lhs_go, pc_go, _, _, _ = nonfused_dispatch(
        ep_buffer, x, topk_ids_i64, num_tokens, num_total_experts)
    nf_go_out = torch.empty(num_local_experts, pf_go.shape[1], N,
                            dtype=torch.bfloat16, device=device)
    for _ in range(num_warmup):
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
            (pf_go.contiguous(), lhs_go),
            (W_fp4, W_scale_u16),
            None, nf_go_out, pc_go.to(torch.int32), fixed_expected_m,
            configs=nf_configs)
    torch.cuda.synchronize()
    ev_nfgo = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev_nfgo[0].record()
    for _ in range(pipe_iters):
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
            (pf_go.contiguous(), lhs_go),
            (W_fp4, W_scale_u16),
            None, nf_go_out, pc_go.to(torch.int32), fixed_expected_m,
            configs=nf_configs)
    ev_nfgo[1].record()
    torch.cuda.synchronize()
    nf_gemm_only_ms = ev_nfgo[0].elapsed_time(ev_nfgo[1]) / pipe_iters

    # ---- Block-Copy NCB sweep ----
    dist.barrier()
    bc_fp4_perf, bc_flags_perf = create_block_copy_buffers(
        num_local_experts, world_size, max_tokens, hidden, block_m, device)
    out_bc = torch.zeros(expert_shape_m, N, dtype=torch.bfloat16, device=device)

    stream = torch.cuda.current_stream()
    bc_results = {}

    ncb_list = [int(x) for x in os.getenv('NCB_SWEEP', '4,8,12').split(',')]
    for ncb in ncb_list:
        # Warmup
        for _ in range(num_warmup):
            fused_dispatch_block_copy_gemm1_fp4(
                (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
                expert_shape_m, max_tokens, world_size,
                local_fp4_buf=bc_fp4_perf, copy_ready_flags=bc_flags_perf,
                num_copy_blocks=ncb, k_tiles_per_flag=K_TILES_PER_FLAG,
                merged_sfa_addrs=merged_sfa_addrs)
        torch.cuda.synchronize()

        # Benchmark: full pipeline (quant + preprocess + block-copy GEMM)
        bc_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        bc_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        gen_bc = 9000 + ncb * 100
        for i in range(num_iters):
            g = gen_bc + i + 1
            bc_start_events[i].record(stream)
            mxfp4_quantize_to_sym_buffer(x, topk_ids, sym_buf,
                num_local_experts=num_total_experts, num_total_experts=num_total_experts,
                max_tokens_per_expert=max_tokens, generation=g,
                sym_buf_addrs=sym_buf_addrs, rank_idx=rank, num_ranks=world_size)
            dispatch_expert_preprocess(
                sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
                max_tokens, hidden, local_expert_start, block_m,
                generation=g, sync=False, _workspace=ws_expert)
            fused_dispatch_block_copy_gemm1_fp4(
                (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
                expert_shape_m, max_tokens, world_size,
                local_fp4_buf=bc_fp4_perf, copy_ready_flags=bc_flags_perf,
                num_copy_blocks=ncb, k_tiles_per_flag=K_TILES_PER_FLAG,
                merged_sfa_addrs=merged_sfa_addrs)
            bc_end_events[i].record(stream)
        torch.cuda.synchronize()
        bc_t = sorted([s.elapsed_time(e) for s, e in zip(bc_start_events, bc_end_events)])
        bc_results[ncb] = bc_t[len(bc_t) // 2]  # median

    best_ncb = min(bc_results, key=bc_results.get)
    bc_pipeline_ms = bc_results[best_ncb]

    # ---- Pipeline overhead breakdown (quant vs preprocess vs flags) ----
    dist.barrier()
    n_breakdown = 40
    ev_q_s = [torch.cuda.Event(enable_timing=True) for _ in range(n_breakdown)]
    ev_q_e = [torch.cuda.Event(enable_timing=True) for _ in range(n_breakdown)]
    ev_p_s = [torch.cuda.Event(enable_timing=True) for _ in range(n_breakdown)]
    ev_p_e = [torch.cuda.Event(enable_timing=True) for _ in range(n_breakdown)]
    gen_bd = 50000
    for i in range(n_breakdown):
        g = gen_bd + i + 1
        ev_q_s[i].record(stream)
        mxfp4_quantize_to_sym_buffer(x, topk_ids, sym_buf,
            num_local_experts=num_total_experts, num_total_experts=num_total_experts,
            max_tokens_per_expert=max_tokens, generation=g,
            sym_buf_addrs=sym_buf_addrs, rank_idx=rank, num_ranks=world_size)
        ev_q_e[i].record(stream)
        ev_p_s[i].record(stream)
        dispatch_expert_preprocess(
            sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
            max_tokens, hidden, local_expert_start, block_m,
            generation=g, sync=False, _workspace=ws_expert)
        ev_p_e[i].record(stream)
    torch.cuda.synchronize()
    quant_times = sorted([ev_q_s[i].elapsed_time(ev_q_e[i]) for i in range(n_breakdown)])
    preproc_times = sorted([ev_p_s[i].elapsed_time(ev_p_e[i]) for i in range(n_breakdown)])
    quant_ms = quant_times[len(quant_times) // 2]
    preproc_ms = preproc_times[len(preproc_times) // 2]
    if rank == 0:
        print(f"\n  Pipeline Overhead Breakdown:")
        print(f"    quant + sym_buf_zero:   {quant_ms:.3f} ms")
        print(f"    expert_preprocess:      {preproc_ms:.3f} ms")
        print(f"    sum:                    {quant_ms + preproc_ms:.3f} ms")

    # ---- Pipeline without preprocess (upper bound of preprocess fusion savings) ----
    # Skip dispatch_expert_preprocess, use metadata from warmup/breakdown.
    # Routing is uniform → metadata unchanged between iterations.
    no_preproc_pipeline_ms = 0.0
    preproc_savings_ms = 0.0
    bc_kernel_only_ms = 0.0
    bc_kernel_std = 0.0
    local_kernel_ms = 0.0
    local_kernel_std = 0.0
    if not SKIP_ISOLATION:
        dist.barrier()
        gen_np = 60000
        for _ in range(num_warmup):
            g = gen_np
            mxfp4_quantize_to_sym_buffer(x, topk_ids, sym_buf,
                num_local_experts=num_total_experts, num_total_experts=num_total_experts,
                max_tokens_per_expert=max_tokens, generation=g,
                sym_buf_addrs=sym_buf_addrs, rank_idx=rank, num_ranks=world_size)
            dispatch_expert_preprocess(
                sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
                max_tokens, hidden, local_expert_start, block_m,
                generation=g, sync=False, _workspace=ws_expert)
            fused_dispatch_block_copy_gemm1_fp4(
                (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
                expert_shape_m, max_tokens, world_size,
                local_fp4_buf=bc_fp4_perf, copy_ready_flags=bc_flags_perf,
                num_copy_blocks=best_ncb, k_tiles_per_flag=K_TILES_PER_FLAG,
                merged_sfa_addrs=merged_sfa_addrs)
        torch.cuda.synchronize()

        np_start = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        np_end = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        for i in range(num_iters):
            g = gen_np + i + 1
            np_start[i].record(stream)
            mxfp4_quantize_to_sym_buffer(x, topk_ids, sym_buf,
                num_local_experts=num_total_experts, num_total_experts=num_total_experts,
                max_tokens_per_expert=max_tokens, generation=g,
                sym_buf_addrs=sym_buf_addrs, rank_idx=rank, num_ranks=world_size)
            fused_dispatch_block_copy_gemm1_fp4(
                (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
                expert_shape_m, max_tokens, world_size,
                local_fp4_buf=bc_fp4_perf, copy_ready_flags=bc_flags_perf,
                num_copy_blocks=best_ncb, k_tiles_per_flag=K_TILES_PER_FLAG,
                merged_sfa_addrs=merged_sfa_addrs)
            np_end[i].record(stream)
        torch.cuda.synchronize()
        np_times = sorted([np_start[i].elapsed_time(np_end[i]) for i in range(num_iters)])
        no_preproc_pipeline_ms = np_times[len(np_times) // 2]
        preproc_savings_ms = bc_pipeline_ms - no_preproc_pipeline_ms

        # ---- Block-Copy kernel-only: copy+GEMM without quant/preprocess ----
        dist.barrier()
        for _ in range(num_warmup):
            fused_dispatch_block_copy_gemm1_fp4(
                (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
                expert_shape_m, max_tokens, world_size,
                local_fp4_buf=bc_fp4_perf, copy_ready_flags=bc_flags_perf,
                num_copy_blocks=best_ncb, k_tiles_per_flag=K_TILES_PER_FLAG,
                merged_sfa_addrs=merged_sfa_addrs)
        torch.cuda.synchronize()
        bc_kernel_times = []
        for _ in range(pipe_iters):
            ev_s = torch.cuda.Event(enable_timing=True)
            ev_e = torch.cuda.Event(enable_timing=True)
            ev_s.record()
            fused_dispatch_block_copy_gemm1_fp4(
                (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
                expert_shape_m, max_tokens, world_size,
                local_fp4_buf=bc_fp4_perf, copy_ready_flags=bc_flags_perf,
                num_copy_blocks=best_ncb, k_tiles_per_flag=K_TILES_PER_FLAG,
                merged_sfa_addrs=merged_sfa_addrs)
            ev_e.record()
            bc_kernel_times.append((ev_s, ev_e))
        torch.cuda.synchronize()
        bc_kernel_ms_list = sorted([s.elapsed_time(e) for s, e in bc_kernel_times])
        bc_kernel_only_ms = bc_kernel_ms_list[len(bc_kernel_ms_list) // 2]
        bc_kernel_std = (sum((t - bc_kernel_only_ms)**2 for t in bc_kernel_ms_list) / len(bc_kernel_ms_list)) ** 0.5

        # ---- Local-to-local P2P isolation experiment ----
        dist.barrier()
        ra_local = ra_e.clone()
        sym_addrs_list = sym_buf_addrs.cpu().tolist()
        ra_list = ra_e.cpu().tolist()
        num_ra_entries = blocks_e * world_size
        for idx in range(num_ra_entries):
            addr = ra_list[idx]
            if addr == 0:
                continue
            r = idx % world_size
            offset = addr - sym_addrs_list[r]
            ra_local[idx] = all_sym_bufs[r].data_ptr() + offset

        for _ in range(num_warmup):
            fused_dispatch_block_copy_gemm1_fp4(
                (W_fp4, W_scale_u16), out_bc, gl_e, ra_local, rs_e, sm_e, rc_e,
                expert_shape_m, max_tokens, world_size,
                local_fp4_buf=bc_fp4_perf, copy_ready_flags=bc_flags_perf,
                num_copy_blocks=best_ncb, k_tiles_per_flag=K_TILES_PER_FLAG,
                merged_sfa_addrs=merged_sfa_addrs)
        torch.cuda.synchronize()

        local_kernel_times = []
        for _ in range(pipe_iters):
            ev_s = torch.cuda.Event(enable_timing=True)
            ev_e = torch.cuda.Event(enable_timing=True)
            ev_s.record()
            fused_dispatch_block_copy_gemm1_fp4(
                (W_fp4, W_scale_u16), out_bc, gl_e, ra_local, rs_e, sm_e, rc_e,
                expert_shape_m, max_tokens, world_size,
                local_fp4_buf=bc_fp4_perf, copy_ready_flags=bc_flags_perf,
                num_copy_blocks=best_ncb, k_tiles_per_flag=K_TILES_PER_FLAG,
                merged_sfa_addrs=merged_sfa_addrs)
            ev_e.record()
            local_kernel_times.append((ev_s, ev_e))
        torch.cuda.synchronize()
        local_ms_list = sorted([s.elapsed_time(e) for s, e in local_kernel_times])
        local_kernel_ms = local_ms_list[len(local_ms_list) // 2]
        local_kernel_std = (sum((t - local_kernel_ms)**2 for t in local_ms_list) / len(local_ms_list)) ** 0.5

    # ---- Print results ----
    if rank == 0:
        print(f"\n  Block-Copy NCB Sweep (median of {num_iters} iters):")
        for ncb, ms in sorted(bc_results.items()):
            marker = " <-- best" if ncb == best_ncb else ""
            bc_sp = nf_pipeline_ms / ms if ms > 0 else 0
            print(f"    ncb={ncb}: {ms:.3f} ms (vs non-fused: {bc_sp:.2f}x){marker}")

        bc_speedup = nf_pipeline_ms / bc_pipeline_ms if bc_pipeline_ms > 0 else 0
        bc_overhead_ms = bc_pipeline_ms - bc_kernel_only_ms
        nf_overhead_ms = nf_pipeline_ms - nf_gemm_only_ms

        p2p_delta_ms = bc_kernel_only_ms - local_kernel_ms
        p2p_pct = p2p_delta_ms / bc_kernel_only_ms * 100 if bc_kernel_only_ms > 0 else 0

        print(f"\n{'='*70}")
        print(f"Performance Summary (M={expert_shape_m}, N={N}, K={hidden}, {world_size} GPUs)")
        print(f"  block_m={block_m}, best ncb={best_ncb}")
        print(f"{'='*70}")
        bc_label = f"BC(ncb={best_ncb})"
        print(f"  {'(ms)':<28s} {bc_label:>10s} {'Local-only':>10s} {'Non-fused':>10s}")
        print(f"  {'-'*62}")
        print(f"  {'Pipeline (full):':<28s} {bc_pipeline_ms:>10.3f} {'—':>10s} {nf_pipeline_ms:>10.3f}")
        print(f"  {'Pipeline (no preprocess):':<28s} {no_preproc_pipeline_ms:>10.3f} {'—':>10s} {'—':>10s}")
        print(f"  {'Kernel-only (copy+GEMM):':<28s} {bc_kernel_only_ms:>10.3f} {local_kernel_ms:>10.3f} {nf_gemm_only_ms:>10.3f}")
        print(f"  {'Pipeline overhead:':<28s} {bc_overhead_ms:>10.3f} {'—':>10s} {nf_overhead_ms:>10.3f}")
        print(f"  {'Preprocess savings:':<28s} {preproc_savings_ms:>9.3f}ms {'—':>10s} {'—':>10s}")
        print(f"  {'-'*62}")
        print(f"  {'vs non-fused:':<28s} {bc_speedup:>9.2f}x {'—':>10s} {'1.00x':>10s}")
        print(f"{'='*70}")
        print(f"\n  P2P Isolation Analysis:")
        print(f"    Kernel (normal P2P):    {bc_kernel_only_ms:.3f} ms (std={bc_kernel_std:.3f})")
        print(f"    Kernel (all-local):     {local_kernel_ms:.3f} ms (std={local_kernel_std:.3f})")
        print(f"    P2P link overhead:      {p2p_delta_ms:+.3f} ms ({p2p_pct:+.1f}%)")
        if abs(p2p_delta_ms) < 0.010:
            print(f"    Conclusion: P2P link NOT the bottleneck (delta < 10μs)")
            print(f"                MC contention (copy writes + GEMM reads on same HBM) is primary")
        elif p2p_delta_ms > 0.020:
            print(f"    Conclusion: P2P link IS a significant bottleneck ({p2p_delta_ms:.3f} ms)")
            print(f"                NVLink bandwidth limits copy block throughput")
        else:
            print(f"    Conclusion: Marginal P2P impact — both MC contention and link contribute")

        if verbose:
            K_packed = hidden // 2
            a_bytes = expert_shape_m * K_packed
            b_bytes = N * K_packed
            a_scale_bytes = expert_shape_m * ((hidden + 31) // 32) * 2
            b_scale_bytes = N * ((hidden + 31) // 32) * 2
            c_bytes = expert_shape_m * N * 2
            total_bytes = a_bytes + b_bytes + a_scale_bytes + b_scale_bytes + c_bytes
            total_flops = 2 * expert_shape_m * N * hidden
            remote_fraction = (world_size - 1) / world_size
            p2p_bytes = (a_bytes + a_scale_bytes) * remote_fraction

            print(f"\n  Detailed Analysis:")
            print(f"    GEMM: M={expert_shape_m}, N={N}, K={hidden} (packed={K_packed})")
            print(f"    Memory: total={total_bytes / 1e6:.2f} MB")
            print(f"    P2P remote: {p2p_bytes / 1e6:.2f} MB "
                  f"({remote_fraction * 100:.0f}% remote, {world_size - 1}/{world_size} ranks)")
            print(f"    Compute: {total_flops / 1e9:.2f} GFLOP, AI={total_flops / total_bytes:.1f}")
            print(f"    Kernel timing: median={bc_kernel_only_ms:.3f} ms, std={bc_kernel_std:.3f} ms")
            print(f"    Local-only timing: median={local_kernel_ms:.3f} ms, std={local_kernel_std:.3f} ms")
            print(f"    Pipeline overhead = quant + preprocess: {bc_overhead_ms:.3f} ms")
            print(f"    Note: Kernel-only includes P2P copy time (copy_ready_flags zeroed each call)")
            print(f"    Note: Local-only replaces all rank_addr_a with local copies (no NVLink traffic)")

    ep_buffer.destroy()
    return True


# ============================================================
# Test 3: K-stripe clock64 profiling
# ============================================================

def test_kstripe_profile(rank, world_size, group, device):
    cfg = CONFIG
    num_total_experts = cfg.num_total_experts(world_size)
    num_local_experts = cfg.num_local_experts
    local_expert_start = cfg.local_expert_start(rank)
    hidden = cfg.hidden
    N = cfg.N
    num_tokens = cfg.num_tokens
    topk = cfg.topk
    max_tokens = cfg.max_tokens
    K_TILES_PER_FLAG = int(os.getenv('K_TILES_PER_FLAG', '4'))
    NUM_COPY_BLOCKS = int(os.getenv('NCB', '8'))

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Test 3: K-stripe clock64 profiling")
        print(f"  K_TILES_PER_FLAG={K_TILES_PER_FLAG}, NCB={NUM_COPY_BLOCKS}")
        print(f"{'='*60}\n")

    torch.manual_seed(99 + rank)
    x = generate_test_input(num_tokens, hidden, device)
    token_idx = torch.arange(num_tokens, device=device, dtype=torch.int32)
    topk_ids = torch.stack(
        [(token_idx * topk + j) % num_total_experts for j in range(topk)], dim=1)

    torch.manual_seed(200)
    W = torch.randn(num_local_experts, N, hidden, dtype=torch.bfloat16, device=device) * 0.01
    W_fp4, W_scale_raw, W_scale_u16 = quantize_grouped_fp4(W)

    buf_size = get_sym_buffer_size(num_local_experts, num_total_experts, max_tokens, hidden)
    sym_buf, sym_buf_addrs, sym_handle = alloc_sym_buffer(buf_size, device, group)

    block_m = get_gemm_block_m(num_tokens * topk, num_local_experts, N, hidden // 2, num_ranks=world_size)

    metadata_size = ((num_total_experts * 4 + 15) // 16) * 16

    sym_buf[:metadata_size].zero_()
    mxfp4_quantize_to_sym_buffer(x, topk_ids, sym_buf,
        num_local_experts=num_total_experts, num_total_experts=num_total_experts,
        max_tokens_per_expert=max_tokens)
    torch.cuda.synchronize()
    dist.barrier()

    ws_expert = create_expert_preprocess_workspace(
        num_local_experts, world_size, max_tokens, block_m, device)
    gl_e, ra_e, rs_e, sm_e, rc_e, total_mb, shape_m = dispatch_expert_preprocess(
        sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
        max_tokens, hidden, local_expert_start, block_m,
        sync=True, _workspace=ws_expert)

    all_sym_bufs = [torch.empty_like(sym_buf) for _ in range(world_size)]
    dist.all_gather(all_sym_bufs, sym_buf)
    merged_sfa, merged_sfa_addrs = build_merged_sfa(
        all_sym_bufs, num_local_experts, local_expert_start,
        num_total_experts, max_tokens, hidden, world_size, device)

    bc_fp4, bc_flags = create_block_copy_buffers(
        num_local_experts, world_size, max_tokens, hidden, block_m, device)
    out_bc = torch.zeros(shape_m, N, dtype=torch.bfloat16, device=device)

    max_mb = total_mb
    # Compute num_stripes for buffer layout
    block_k = 128  # from tuner config
    num_k_tiles = (hidden // 2) // block_k
    num_stripes = (num_k_tiles // K_TILES_PER_FLAG) if K_TILES_PER_FLAG > 0 else 1
    profile_buf_size = max_mb * (2 * num_stripes + 1)
    profile_buf = torch.zeros(profile_buf_size, dtype=torch.int64, device=device)

    if rank == 0:
        print(f"  num_k_tiles={num_k_tiles}, num_stripes={num_stripes}, max_mb={max_mb}")

    for _ in range(5):
        fused_dispatch_block_copy_gemm1_fp4(
            (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
            shape_m, max_tokens, world_size,
            local_fp4_buf=bc_fp4, copy_ready_flags=bc_flags,
            num_copy_blocks=NUM_COPY_BLOCKS, k_tiles_per_flag=K_TILES_PER_FLAG,
            merged_sfa_addrs=merged_sfa_addrs)
    torch.cuda.synchronize()

    NUM_ITERS = 20
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(NUM_ITERS):
        fused_dispatch_block_copy_gemm1_fp4(
            (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
            shape_m, max_tokens, world_size,
            local_fp4_buf=bc_fp4, copy_ready_flags=bc_flags,
            num_copy_blocks=NUM_COPY_BLOCKS, k_tiles_per_flag=K_TILES_PER_FLAG,
            merged_sfa_addrs=merged_sfa_addrs)
    end_evt.record()
    torch.cuda.synchronize()
    kernel_ms = start_evt.elapsed_time(end_evt) / NUM_ITERS

    profile_buf.zero_()
    fused_dispatch_block_copy_gemm1_fp4(
        (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
        shape_m, max_tokens, world_size,
        local_fp4_buf=bc_fp4, copy_ready_flags=bc_flags,
        num_copy_blocks=NUM_COPY_BLOCKS, k_tiles_per_flag=K_TILES_PER_FLAG,
        merged_sfa_addrs=merged_sfa_addrs,
        kstripe_profile_buf=profile_buf)
    torch.cuda.synchronize()

    prof = profile_buf.cpu().numpy()
    # New layout: [copy: max_mb * num_stripes] [gemm_wait: max_mb * num_stripes] [gemm_mma: max_mb]
    copy_section = prof[:max_mb * num_stripes].reshape(max_mb, num_stripes)
    wait_section = prof[max_mb * num_stripes : max_mb * num_stripes * 2].reshape(max_mb, num_stripes)
    mma_section = prof[max_mb * num_stripes * 2 : max_mb * num_stripes * 2 + max_mb]

    if rank == 0:
        mode = f"kstripe(KTPF={K_TILES_PER_FLAG})" if K_TILES_PER_FLAG > 0 else "mblock"
        print(f"\n  Mode: {mode}, total_m_blocks={max_mb}, num_stripes={num_stripes}")
        print(f"  Kernel time: {kernel_ms:.3f} ms (avg over {NUM_ITERS} iters)")

        # Per-stripe copy stats
        print(f"\n  Copy block (per stripe, clock cycles):")
        for s in range(num_stripes):
            stripe_data = copy_section[:, s]
            active = stripe_data[stripe_data > 0]
            if len(active) > 0:
                med = int(sorted(active)[len(active)//2])
                print(f"    stripe {s}: median={med:>7}, mean={active.mean():.0f}, "
                      f"min={active.min():.0f}, max={active.max():.0f} (n={len(active)})")

        # Per-stripe GEMM wait stats
        print(f"\n  GEMM wait (per stripe, clock cycles):")
        for s in range(num_stripes):
            stripe_data = wait_section[:, s]
            active = stripe_data[stripe_data > 0]
            if len(active) > 0:
                med = int(sorted(active)[len(active)//2])
                print(f"    stripe {s}: median={med:>7}, mean={active.mean():.0f}, "
                      f"min={active.min():.0f}, max={active.max():.0f} (n={len(active)})")
            else:
                print(f"    stripe {s}: (no wait, flag already ready)")

        # MMA total stats
        active_mma = mma_section[mma_section > 0]
        if len(active_mma) > 0:
            med = int(sorted(active_mma)[len(active_mma)//2])
            print(f"\n  GEMM MMA total (per M-block, clock cycles):")
            print(f"    median={med}, mean={active_mma.mean():.0f}, "
                  f"min={active_mma.min():.0f}, max={active_mma.max():.0f} (n={len(active_mma)})")

        # Summary: total copy per M-block (sum across stripes)
        copy_total_per_mb = copy_section.sum(axis=1)
        active_total = copy_total_per_mb[copy_total_per_mb > 0]
        if len(active_total) > 0:
            med = int(sorted(active_total)[len(active_total)//2])
            print(f"\n  Copy total per M-block (sum of stripes):")
            print(f"    median={med}, mean={active_total.mean():.0f}")

    return True


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    rank, local_rank, world_size, group = init_dist()
    device = f'cuda:{local_rank}'

    if rank == 0:
        print(f"Block-Copy Fused Dispatch GEMM1 Test")
        print(f"  world_size={world_size}, device={torch.cuda.get_device_name(local_rank)}")
        print(f"  Config: {CONFIG}")
        if FULL_CORRECTNESS:
            print(f"  FULL_CORRECTNESS=1 (per-expert CPU reference check)")

    assert world_size >= 2, f"Requires at least 2 GPUs, got {world_size}"

    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)

    results = {}

    if not SKIP_CORRECTNESS:
        try:
            results['test1_correctness'] = test_correctness(rank, world_size, group, device)
        except Exception as e:
            if rank == 0:
                print(f"Test 1 error: {e}")
                import traceback; traceback.print_exc()
            results['test1_correctness'] = False
        dist.barrier()
    elif rank == 0:
        print("Skipping Test 1 (correctness) — SKIP_CORRECTNESS=1")

    try:
        results['test2_performance'] = test_performance(rank, world_size, group, device)
    except Exception as e:
        if rank == 0:
            print(f"Test 2 error: {e}")
            import traceback; traceback.print_exc()
        results['test2_performance'] = False

    dist.barrier()

    if os.getenv('KSTRIPE_PROFILE', '0') == '1':
        try:
            results['test3_profile'] = test_kstripe_profile(rank, world_size, group, device)
        except Exception as e:
            if rank == 0:
                print(f"Test 3 error: {e}")
                import traceback; traceback.print_exc()
            results['test3_profile'] = False
        dist.barrier()
    elif rank == 0:
        print("Skipping Test 3 (profiling) — set KSTRIPE_PROFILE=1 to enable")

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        print(f"\n{'='*60}")
        print("Results:")
        for name, passed in results.items():
            print(f"  {name}: {'PASSED' if passed else 'FAILED'}")
        all_ok = all(results.values())
        print(f"\n{'All PASSED!' if all_ok else 'Some tests FAILED'}")
        print(f"{'='*60}")
