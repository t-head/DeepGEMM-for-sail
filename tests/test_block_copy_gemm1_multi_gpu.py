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
    (Fused dispatch GEMM1 is masked-only; the NoPad fused path was removed.)
"""

import os
import sys
import time
import inspect

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
SKIP_PERFORMANCE = int(os.getenv('SKIP_PERFORMANCE', '0'))
RUN_RANK_SKEW = int(os.getenv('RUN_RANK_SKEW', '0'))
SKIP_ISOLATION = int(os.getenv('SKIP_ISOLATION', '0'))
# The all-local P2P isolation experiment replaces every rank_addr_a with a LOCAL
# copy. That is fundamentally incompatible with DG_BULK_REMOTE, whose copy loop
# issues remote-only intrinsics for peer ranks (r != rank_idx) — running them on a
# local address is illegal (illegal memory access → crash). So when the kernel is
# built with DG_BULK_REMOTE, force-skip isolation regardless of the env value.
if int(os.getenv('DG_BULK_REMOTE', '0')) != 0 and not SKIP_ISOLATION:
    SKIP_ISOLATION = 1
    print("[test] DG_BULK_REMOTE set -> forcing SKIP_ISOLATION=1 "
          "(remote intrinsics can't run on the all-local isolation buffers)",
          flush=True)
COL_MAJOR_SCALE = int(os.getenv('COL_MAJOR_SCALE', '1'))
PAD_ALIGN = 1  # expert packing has no artificial token alignment
# Fused dispatch GEMM1 is masked-only; the NoPad fused path has been removed.
FUSED_GEMM_GROUPING = 'masked'
USE_MASKED_FUSED_GEMM = True


# ============================================================
# Imports
# ============================================================

import deep_gemm
import deep_ep
from deep_gemm import (
    calc_diff, preprocess_mxfp4_scales,
    get_sym_buffer_size,
)
from deep_gemm.jit_kernels.dispatch_fused_gemm import (
    _mxfp4_quantize_to_sym_buffer,
    BlockCopyDispatchContext,
    dispatch_expert_prepare, dispatch_expert_finalize,
    dispatch_expert_preprocess_merged,
    dispatch_expert_sfa_overlap,
    dispatch_sfa_preprocess,
    create_expert_preprocess_workspace,
    fused_dispatch_block_copy_gemm1_fp4,
    create_block_copy_buffers,
    get_sfa_staging_size, set_sfa_staging_addrs,
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


def distributed_max_samples(samples, device):
    """Return elementwise max timing samples across ranks for critical-path latency."""
    values = torch.tensor(samples, dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return values.cpu().tolist()


def distributed_max_scalar(value, device):
    return distributed_max_samples([value], device)[0]


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


def get_gemm_configs(shape_m, expected_m, num_groups, n, k, padded_m=None):
    """Full GEMM config used by BOTH preprocess (block_m) and the fused kernel.

    Returning the whole tuple and passing it to fused_dispatch guarantees the
    preprocess block_m == kernel block_m (a mismatch silently zeroes 2nd+ blocks).
    FORCE_EXPECTED_M lets you steer block_m for testing (e.g. 129 -> block_m=256).
    """
    assert padded_m is not None
    tuning_m = shape_m
    tuning_expected_m = expected_m
    gemm_type = GemmType.GroupedMasked
    force = int(os.getenv('FORCE_EXPECTED_M', '0'))
    if force > 0:
        tuning_expected_m = force
    num_sms = get_num_sms()
    return get_best_configs_fp4(
        tuning_m, tuning_expected_m, n, k, num_groups, num_sms,
        gemm_type=gemm_type)


def get_gemm_block_m(shape_m, expected_m, num_groups, n, k, padded_m=None):
    return get_gemm_configs(
        shape_m, expected_m, num_groups, n, k, padded_m=padded_m)[1]


def refresh_expert_preprocess(
    sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
    max_tokens, hidden, local_expert_start, block_m, _workspace, generation=0,
    sync=False, dbg_cyc=None, local_sfa_buf=None, copy_ready_flags=None,
):
    """Refresh fixed-config metadata without a host count readback.

    Merged prepare+finalize (single launch) is the DEFAULT for this fixed-block_m
    path: it saves one launch + the kernel-boundary gap, worth ~7us clean-min on
    the 2-GPU pipeline (bit-exact vs split, VALIDATE_MERGED=1). Set
    MERGED_PREPROCESS=0 to force the split prepare();finalize() sequence."""
    if (os.getenv("DG_SFA_OVERLAP", "0") != "0"
            and local_sfa_buf is not None and copy_ready_flags is not None):
        if dbg_cyc is not None:
            raise ValueError("SFA overlap path does not support dbg_cyc")
        return dispatch_expert_sfa_overlap(
            sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
            max_tokens, hidden, local_expert_start, block_m,
            local_sfa_buf, copy_ready_flags,
            generation=generation, sync=sync, _workspace=_workspace)
    if os.getenv("MERGED_PREPROCESS", "1") == "1":
        return dispatch_expert_preprocess_merged(
            sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
            max_tokens, hidden, local_expert_start, block_m, generation=generation,
            sync=sync, dbg_cyc=dbg_cyc, _workspace=_workspace)
    dispatch_expert_prepare(
        sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
        max_tokens, hidden, local_expert_start, generation=generation,
        sync=False, dbg_cyc=dbg_cyc, _workspace=_workspace)
    return dispatch_expert_finalize(
        sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
        max_tokens, hidden, local_expert_start, block_m, generation=generation,
        sync=sync, dbg_cyc=dbg_cyc, _workspace=_workspace)


def create_fused_output(shape_m, num_groups, padded_m, n, device):
    return torch.zeros((num_groups, padded_m, n), dtype=torch.bfloat16, device=device)


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


def data_region_size(num_total_experts, max_tokens, hidden):
    metadata_bytes = ((num_total_experts * 4 + 15) // 16) * 16
    fp4_region = num_total_experts * max_tokens * (hidden // 2)
    k_scale_blocks = (((hidden + 31) // 32) + 1) // 2
    scale_region = num_total_experts * k_scale_blocks * max_tokens * 2
    return metadata_bytes + fp4_region + scale_region


def tagged_count_bits(max_tokens):
    bits = 1
    mask = 1
    while mask < max_tokens and bits < 12:
        bits += 1
        mask = (1 << bits) - 1
    return bits


def unpack_generation_count(value, generation, max_tokens):
    if generation == 0 or max_tokens > 0xFFF:
        return value
    bits = tagged_count_bits(max_tokens)
    gen_mask = (1 << (32 - bits)) - 1
    if generation > gen_mask:
        return value
    count_mask = (1 << bits) - 1
    value &= 0xFFFFFFFF
    tag = generation << bits
    return (value & count_mask) if (value & ~count_mask) == tag else 0


def get_expert_token_counts(all_sym_bufs, ge, world_size, data_base=0, generation=0, max_tokens=0):
    """Return list of per-rank token counts for global expert ge."""
    return [
        unpack_generation_count(
            all_sym_bufs[r][data_base + ge * 4:data_base + ge * 4 + 4].view(torch.int32).item(),
            generation, max_tokens)
        for r in range(world_size)
    ]


def build_merged_sfa(all_sym_bufs, num_local_experts, local_expert_start,
                     num_total_experts, max_tokens, hidden, world_size, device,
                     data_base=0, generation=0, pad_align=None):
    """Build the pre-0f40bad host merged SFA buffer from all-gathered sym bufs.

    Diagnostic helper for DG_SFA_SOURCE=host: reassembles each local expert's A
    scales into a tightly-packed column-major [k_scale_blocks, padded_total]
    region (K-stride = padded_total), the exact layout the GEMM's host read path
    expects (ptr_scale_A = base, dSFA K-stride = M). Rank r's rows are placed at
    the cumulative (pad_align-aligned) token offset, matching rank_split_m under
    PAD_ALIGN=1 for the single-m-block-per-expert production config.

    Returns (merged_sfa, merged_sfa_addrs) — keep merged_sfa alive until the GEMM
    has run (merged_sfa_addrs holds raw data pointers into it).
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
        counts = get_expert_token_counts(
            all_sym_bufs, ge, world_size, data_base=data_base,
            generation=generation, max_tokens=max_tokens)
        expert_padded_totals.append(sum(_align_up(c, pad_align) for c in counts))

    total_sfa_elems = max(sum(pt * k_scale_blocks for pt in expert_padded_totals), 1)
    merged_sfa = torch.zeros(total_sfa_elems, dtype=torch.uint16, device=device)
    merged_sfa_addrs = torch.zeros(num_local_experts, dtype=torch.int64, device=device)

    offset = 0
    for le in range(num_local_experts):
        ge = local_expert_start + le
        pt = expert_padded_totals[le]
        merged_sfa_addrs[le] = merged_sfa.data_ptr() + offset * 2
        if pt == 0:
            continue
        expert_sfa = merged_sfa[offset:offset + pt * k_scale_blocks].view(k_scale_blocks, pt)
        counts = get_expert_token_counts(
            all_sym_bufs, ge, world_size, data_base=data_base,
            generation=generation, max_tokens=max_tokens)
        merged_token_pos = 0
        for r in range(world_size):
            count_r = counts[r]
            if count_r == 0:
                continue
            scale_off_r = (data_base + metadata_bytes + fp4_region_size +
                           ge * scale_elems_per_expert_max * 2)
            src_scale = all_sym_bufs[r][scale_off_r:scale_off_r + scale_elems_per_expert_max * 2] \
                .view(torch.uint16).view(k_scale_blocks, max_tokens)
            expert_sfa[:, merged_token_pos:merged_token_pos + count_r] = src_scale[:, :count_r]
            merged_token_pos += _align_up(count_r, pad_align)
        offset += pt * k_scale_blocks

    return merged_sfa, merged_sfa_addrs


def print_routing_table(topk_ids, num_local_experts, rank, world_size, num_tokens, topk):
    """Print where each rank's tokens are routed: (target_rank, local_expert)."""
    ids = topk_ids.cpu().tolist()  # [num_tokens, topk]
    # Aggregate: count how many token-slots go to each (target_rank, local_expert)
    from collections import defaultdict
    routing_counts = defaultdict(int)  # (target_rank, local_expert_idx) -> count
    for t in range(num_tokens):
        for k in range(topk):
            ge = ids[t][k]  # global expert id
            target_rank = ge // num_local_experts
            local_expert = ge % num_local_experts
            routing_counts[(target_rank, local_expert)] += 1

    print(f"  [Rank {rank}] Routing table (num_tokens={num_tokens}, topk={topk}, "
          f"total_slots={num_tokens * topk}):")
    # Print per target rank
    for tr in range(world_size):
        experts_on_rank = []
        for le in range(num_local_experts):
            cnt = routing_counts.get((tr, le), 0)
            if cnt > 0:
                experts_on_rank.append(f"expert{le}:{cnt}")
        if experts_on_rank:
            print(f"    -> rank {tr}: {', '.join(experts_on_rank)}")
    # Also print first few token details (up to 8 tokens)
    max_show = min(num_tokens, 8)
    print(f"    First {max_show} tokens detail:")
    for t in range(max_show):
        dests = []
        for k in range(topk):
            ge = ids[t][k]
            target_rank = ge // num_local_experts
            local_expert = ge % num_local_experts
            dests.append(f"rank{target_rank}/expert{local_expert}(ge={ge})")
        print(f"      token[{t}]: {', '.join(dests)}")
    if num_tokens > max_show:
        print(f"      ... ({num_tokens - max_show} more tokens)")


# ============================================================
# Test 1: Correctness (Random Routing)
# ============================================================


def test_public_api_contract():
    """Production API must not let callers select generation or parity."""
    run_params = inspect.signature(deep_gemm.BlockCopyDispatchContext.run).parameters
    forbidden_round_args = {'generation', 'parity'}
    leaked_round_args = forbidden_round_args.intersection(run_params)
    assert not leaked_round_args, f"public run() leaks {sorted(leaked_round_args)}"

    forbidden_exports = {
        'mxfp4_quantize_to_sym_buffer',
        'dispatch_preprocess',
        'dispatch_expert_preprocess',
        'create_preprocess_workspace',
    }
    leaked_exports = sorted(name for name in forbidden_exports if hasattr(deep_gemm, name))
    assert not leaked_exports, f"deprecated low-level exports remain: {leaked_exports}"
    assert deep_gemm.BlockCopyDispatchContext.generation.fset is None
    assert deep_gemm.BlockCopyDispatchContext.parity.fset is None
    return True


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
    num_rounds = int(os.getenv('TEST_ROUNDS', '3'))
    if RUN_RANK_SKEW and num_rounds < 3:
        raise ValueError("RUN_RANK_SKEW requires TEST_ROUNDS >= 3")
    num_copy_blocks = int(os.getenv('NCB', '8'))
    k_tiles_per_flag = int(os.getenv('K_TILES_PER_FLAG', '0'))

    if rank == 0:
        print(f"\n{'='*60}")
        print("Test 1: Context-managed multi-round correctness")
        print(f"  Config: {cfg}")
        print(f"  world_size={world_size}, rounds={num_rounds}")
        print(f"{'='*60}\n")

    torch.manual_seed(200)
    W = torch.randn(
        num_local_experts, N, hidden, dtype=torch.bfloat16, device=device) * 0.01
    W_fp4, W_scale_raw, W_scale_u16 = quantize_grouped_fp4(W)

    buf_size = get_sym_buffer_size(
        num_local_experts, num_total_experts, max_tokens, hidden)
    sym_buf, sym_buf_addrs, sym_handle = alloc_sym_buffer(buf_size, device, group)
    # DG_SFA_PUSH: separate symmetric staging buffer (see the perf-path setup for why it
    # must NOT be appended to sym_buf). Register once for this context test.
    sfa_staging_buf = None
    sfa_staging_addrs = None
    sfa_staging_handle = None
    if os.getenv('DG_SFA_PUSH', '0') != '0':
        staging_size = get_sfa_staging_size(num_local_experts, num_total_experts, max_tokens, hidden)
        sfa_staging_buf, sfa_staging_addrs, sfa_staging_handle = alloc_sym_buffer(staging_size, device, group)
        set_sfa_staging_addrs(sfa_staging_addrs, sfa_staging_buf.data_ptr())
    context = BlockCopyDispatchContext(
        sym_buf=sym_buf,
        sym_buf_addrs=sym_buf_addrs,
        rank_idx=rank,
        num_ranks=world_size,
        num_local_experts=num_local_experts,
        num_total_experts=num_total_experts,
        max_tokens_per_expert=max_tokens,
        local_expert_start=local_expert_start,
        group=group,
        sfa_staging_buf=sfa_staging_buf,
        sfa_staging_addrs=sfa_staging_addrs,
    )
    ep_buffer = create_ep_buffer(
        group, num_local_experts, num_tokens, hidden, world_size,
        num_total_experts)

    metadata_bytes = ((num_total_experts * 4 + 15) // 16) * 16
    fp4_per_expert = max_tokens * (hidden // 2)
    fp4_region_size = num_total_experts * fp4_per_expert
    k_blocks = (hidden + 31) // 32
    k_scale_blocks = (k_blocks + 1) // 2
    scale_elems_per_expert = k_scale_blocks * max_tokens
    region_size = data_region_size(num_total_experts, max_tokens, hidden)
    all_passed = True
    previous_x = None
    previous_topk_ids = None

    for round_idx in range(num_rounds):
        seed = 77 + round_idx * 1000 + rank
        torch.manual_seed(seed)
        x = generate_test_input(num_tokens, hidden, device)
        if int(os.getenv('UNIFORM_ROUTING', '0')):
            # Uniform round-robin routing (each expert ~128 tokens => block_m=128),
            # matching the perf section. round_idx offset keeps each round's routing
            # distinct (satisfies the no-reuse guard) while staying uniform. Lets us
            # validate configs whose smem only fits at block_m=128 (e.g. block_n=512).
            token_idx = torch.arange(num_tokens, device=device, dtype=torch.int32)
            topk_ids = torch.stack(
                [(token_idx * topk + j + round_idx) % num_total_experts
                 for j in range(topk)], dim=1).to(torch.int32)
        else:
            scores = torch.randn(
                num_tokens, num_total_experts, dtype=torch.float32, device=device)
            topk_ids = torch.topk(
                scores, topk, dim=-1, largest=True, sorted=False)[1].to(torch.int32)
        if previous_x is not None:
            if torch.equal(x, previous_x):
                raise AssertionError(f"round {round_idx + 1} reused the previous input")
            if torch.equal(topk_ids, previous_topk_ids):
                raise AssertionError(f"round {round_idx + 1} reused the previous routing")
        previous_x = x.clone()
        previous_topk_ids = topk_ids.clone()

        if RUN_RANK_SKEW and round_idx == 2:
            # Align before injecting skew, then delay exactly one producer. The
            # other ranks must block inside expert prepare on full generation 3
            # (parity 1), not accept generation 1 data or a stale parity flag.
            dist.barrier()
            skew_seconds = float(os.getenv('RANK_SKEW_SECONDS', '1.0'))
            if rank == 0:
                time.sleep(skew_seconds)
        if int(os.getenv('CONTEXT_WALL_TIMING', '0')):
            # CPU reference work from the previous round is rank-skewed. Align
            # before measuring submission so arrival polling is not charged for
            # unrelated host-side validation skew.
            dist.barrier()
            torch.cuda.synchronize()
        round_start = time.monotonic()
        if int(os.getenv('USE_CONTEXT_DEFAULTS', '0')):
            round_result = context.run(x, topk_ids, (W_fp4, W_scale_u16))
        else:
            round_result = context.run(
                x, topk_ids, (W_fp4, W_scale_u16),
                num_copy_blocks=num_copy_blocks,
                k_tiles_per_flag=k_tiles_per_flag)
        round_elapsed = time.monotonic() - round_start
        torch.cuda.synchronize()
        if int(os.getenv('CONTEXT_WALL_TIMING', '0')):
            elapsed_us = torch.tensor(
                [round_elapsed * 1e6], dtype=torch.float64, device=device)
            dist.all_reduce(elapsed_us, op=dist.ReduceOp.MAX)
            if rank == 0:
                print(f"  context.run host submission round {round_idx + 1}: "
                      f"{elapsed_us.item():.1f} us (max rank)")
        if RUN_RANK_SKEW and round_idx == 2:
            min_wait = float(os.getenv('RANK_SKEW_SECONDS', '1.0')) * 0.7
            elapsed_tensor = torch.tensor([round_elapsed], dtype=torch.float64, device=device)
            elapsed_by_rank = [torch.zeros_like(elapsed_tensor) for _ in range(world_size)]
            dist.all_gather(elapsed_by_rank, elapsed_tensor)
            elapsed_values = [value.item() for value in elapsed_by_rank]
            skew_ok = all(value >= min_wait for value in elapsed_values[1:])
            all_passed = all_passed and skew_ok
            if rank == 0:
                waits = ', '.join(
                    f"rank{r}={value:.3f}s" for r, value in enumerate(elapsed_values))
                print(f"  Rank-skew generation 3 waits: {waits} "
                      f"({'PASSED' if skew_ok else 'FAILED'})")

        expected_generation = round_idx + 1
        if (round_result.generation != expected_generation or
                round_result.parity != (expected_generation & 1)):
            raise AssertionError(
                f"internal round sequence mismatch: got "
                f"gen={round_result.generation}/parity={round_result.parity}, "
                f"expected {expected_generation}/{expected_generation & 1}")
        data_base = round_result.parity * region_size

        # This collective is deliberately after prepare + GEMM. It exists only
        # to build the CPU reference and cannot satisfy the arrival barrier.
        all_sym_bufs = [torch.zeros_like(sym_buf) for _ in range(world_size)]
        dist.all_gather(all_sym_bufs, sym_buf)

        pf, lhs_sc, pc, _, _, _ = nonfused_dispatch(
            ep_buffer, x, topk_ids.to(torch.int64), num_tokens,
            num_total_experts)
        nonfused_expected_m = max(int(pc.max().item()), 1)
        nf_configs = get_best_configs_fp4(
            nonfused_expected_m * num_local_experts, nonfused_expected_m,
            N, hidden // 2, num_local_experts, get_num_sms(),
            gemm_type=GemmType.GroupedNoPad)
        nf_out = torch.empty(
            num_local_experts, pf.shape[1], N,
            dtype=torch.bfloat16, device=device)
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
            (pf.contiguous(), lhs_sc), (W_fp4, W_scale_u16), None,
            nf_out, pc.to(torch.int32), nonfused_expected_m,
            configs=nf_configs)
        torch.cuda.synchronize()

        if round_result.expected_m != nonfused_expected_m:
            if rank == 0:
                print(
                    f"  Round {round_idx + 1}: expected_m mismatch: "
                    f"fused={round_result.expected_m}, "
                    f"non-fused={nonfused_expected_m}")
            all_passed = False

        if rank == 0:
            print(
                f"\n  Round {round_idx + 1}: internal "
                f"gen={round_result.generation}, parity={round_result.parity}, "
                f"shape_m={round_result.shape_m}, "
                f"expected_m={round_result.expected_m}, "
                f"block_m={round_result.block_m}")
            print(
                f"  {'Expert':<10s} {'M':>5s} {'BC vs NF':>10s} "
                f"{'BC vs CPU':>12s} {'Status':>8s}")

        expert_base_m = 0
        for le in range(num_local_experts):
            ge = local_expert_start + le
            counts_per_rank = get_expert_token_counts(
                all_sym_bufs, ge, world_size, data_base=data_base,
                generation=round_result.generation, max_tokens=max_tokens)
            m_fused = sum(counts_per_rank)
            m_nf = int(pc[le].item())

            if m_fused != m_nf:
                if rank == 0:
                    print(
                        f"  Expert {ge}: token count mismatch: "
                        f"fused={m_fused}, non-fused={m_nf}")
                all_passed = False
                expert_base_m += m_fused
                continue
            if m_fused == 0:
                expert_base_m += m_fused
                continue

            bc_e = round_result.out[le, :m_fused]
            nf_e = nf_out[le, :m_nf]
            bc_norms = bc_e.float().norm(dim=-1).sort().values
            nf_norms = nf_e.float().norm(dim=-1).sort().values
            diff_bc_nf = calc_diff(bc_norms, nf_norms)

            A_rows = []
            for src_rank, count_r in enumerate(counts_per_rank):
                if count_r == 0:
                    continue
                fp4_off = (
                    data_base + metadata_bytes + ge * fp4_per_expert)
                fp4_r = all_sym_bufs[src_rank][
                    fp4_off:fp4_off + count_r * (hidden // 2)
                ].cpu().view(count_r, hidden // 2)
                pushed_scale_ref = (
                    os.getenv('DG_SFA_PUSH', '0') != '0'
                    and os.getenv('DG_SFA_KEEP_LOCAL_SCALE', '0') == '0'
                    and os.getenv('DG_SFA_SOURCE', 'gpu') != 'host')
                if pushed_scale_ref:
                    # The optimized push path omits the redundant sym-buffer scale
                    # copy. Validate against this owner's local staging band instead:
                    # [parity][local_expert][source_rank][slot][ksb].
                    parity_bytes = (
                        num_total_experts * k_scale_blocks * max_tokens * 2)
                    band_bytes = (
                        (le * world_size + src_rank)
                        * scale_elems_per_expert * 2)
                    scale_off = (
                        (round_result.generation & 1) * parity_bytes + band_bytes)
                    src_scale_flat = sfa_staging_buf[
                        scale_off:scale_off + scale_elems_per_expert * 2
                    ].cpu().view(torch.uint16)
                    scale_pairs = src_scale_flat.view(
                        max_tokens, k_scale_blocks)[:count_r, :]
                else:
                    scale_off = (
                        data_base + metadata_bytes + fp4_region_size +
                        ge * scale_elems_per_expert * 2)
                    src_scale_flat = all_sym_bufs[src_rank][
                        scale_off:scale_off + scale_elems_per_expert * 2
                    ].cpu().view(torch.uint16)
                if not pushed_scale_ref and os.getenv('DG_SFA_ROWMAJOR_SRC', '0') != '0':
                    # Row-major source [max_tokens, k_scale_blocks]: each token's
                    # ksb scales contiguous -> first count_r rows are this expert's.
                    scale_pairs = src_scale_flat.view(
                        max_tokens, k_scale_blocks)[:count_r, :]  # [count_r, ksb]
                elif not pushed_scale_ref:
                    # Column-major source [k_scale_blocks, max_tokens].
                    scale_pairs = src_scale_flat.view(
                        k_scale_blocks, max_tokens)[:, :count_r].T  # [count_r, ksb]
                scale_u8 = scale_pairs.contiguous().view(
                    torch.uint8).view(count_r, k_blocks)
                A_rows.append(dequantize_fp4_torch(fp4_r, scale_u8))
            A = torch.cat(A_rows, dim=0)
            W_bf16 = dequantize_fp4_torch(
                W_fp4[le].cpu(), W_scale_raw[le].cpu())
            ref_out = (A.float() @ W_bf16.float().T).bfloat16()
            diff_bc_cpu = calc_diff(
                bc_e.cpu().float(), ref_out.float())

            ok = diff_bc_nf < 0.01
            if FULL_CORRECTNESS:
                ok = ok and diff_bc_cpu < 0.002
            all_passed = all_passed and ok
            if rank == 0:
                print(
                    f"  Expert {ge:<4d}  {m_fused:>5d}  "
                    f"{diff_bc_nf:>10.6f} {diff_bc_cpu:>12.6f} "
                    f"{'PASSED' if ok else 'FAILED':>8s}")
            expert_base_m += m_fused

    ep_buffer.destroy()
    pass_tensor = torch.tensor(
        [1 if all_passed else 0], dtype=torch.int32, device=device)
    dist.all_reduce(pass_tensor, op=dist.ReduceOp.MIN)
    all_passed = pass_tensor.item() == 1
    if rank == 0:
        print(
            f"\n  Test 1: {'PASSED' if all_passed else 'FAILED'} "
            f"({num_rounds} rounds, all ranks)")
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
    is_prod_8gpu = (
        world_size == 8 and num_local_experts == 12
        and hidden == 7168 and N == 6144)
    K_TILES_PER_FLAG = int(os.getenv(
        'K_TILES_PER_FLAG', '14' if is_prod_8gpu else '0'))

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Test 2: Fixed-routing kernel microbenchmark (Block-Copy)")
        print(f"  Config: {cfg}")
        print(f"  world_size={world_size}")
        print("  Includes GPU-side SFA copy; dynamic-routing E2E is not measured here")
        print(f"{'='*60}\n")

    torch.manual_seed(99 + rank)
    x = generate_test_input(num_tokens, hidden, device)

    # Uniform token distribution: round-robin assignment so each expert
    # gets roughly num_tokens * topk / num_total_experts tokens per rank.
    # Deterministic across runs and uniform across all world_size values.
    token_idx = torch.arange(num_tokens, device=device, dtype=torch.int32)
    topk_ids = torch.stack(
        [(token_idx * topk + j) % num_total_experts for j in range(topk)], dim=1)

    # Print routing information (默认关闭;set PRINT_ROUTING=1 打开)
    if int(os.getenv('PRINT_ROUTING', '0')):
        print_routing_table(topk_ids, num_local_experts, rank, world_size, num_tokens, topk)
    dist.barrier()

    topk_ids_i64 = topk_ids.to(torch.int64)

    torch.manual_seed(200)
    W = torch.randn(num_local_experts, N, hidden, dtype=torch.bfloat16, device=device) * 0.01
    W_fp4, W_scale_raw, W_scale_u16 = quantize_grouped_fp4(W)

    # ---- Fused path setup ----
    buf_size = get_sym_buffer_size(num_local_experts, num_total_experts, max_tokens, hidden)
    sym_buf, sym_buf_addrs, sym_handle = alloc_sym_buffer(buf_size, device, group)
    # DG_SFA_PUSH: separate symmetric staging buffer (NOT appended to sym_buf — growing
    # sym_buf slows its remote FP4 P2P ~3x on this platform). quant pushes SFA into peers'
    # staging; expert_preprocess points rank_addr_sfa at this rank's local staging; the
    # reshape repacks it locally. Register once so the ~10 quant/preprocess call sites are
    # unchanged.
    if os.getenv('DG_SFA_PUSH', '0') != '0':
        staging_size = get_sfa_staging_size(num_local_experts, num_total_experts, max_tokens, hidden)
        sfa_staging_buf, sfa_staging_addrs, sfa_staging_handle = alloc_sym_buffer(staging_size, device, group)
        set_sfa_staging_addrs(sfa_staging_addrs, sfa_staging_buf.data_ptr())
    # One-time zero of the atomic-arrival slot region (last 128B); symm_mem.empty() is
    # uninitialized and warmup uses generation=0 (no push), so slots must start < any g.
    sym_buf[buf_size - 128:].zero_()
    dist.barrier(); torch.cuda.synchronize()
    metadata_size = ((num_total_experts * 4 + 15) // 16) * 16

    # Count/shape prepare is independent of BLOCK_M and directly provides the
    # actual maximum expert M used by both fused and non-fused tuning.
    _mxfp4_quantize_to_sym_buffer(
        x, topk_ids, sym_buf,
        num_local_experts=num_total_experts,
        num_total_experts=num_total_experts,
        max_tokens_per_expert=max_tokens)
    torch.cuda.synchronize()
    dist.barrier()
    ws_expert = create_expert_preprocess_workspace(
        num_local_experts, world_size, max_tokens, device)
    expert_shape_m, expert_expected_m, _ = dispatch_expert_prepare(
        sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
        max_tokens, hidden, local_expert_start, _workspace=ws_expert)
    block_m = get_gemm_block_m(
        expert_shape_m, expert_expected_m, num_local_experts, N, hidden // 2,
        padded_m=max_tokens)
    gl_e, ra_e, rs_e, sm_e, rc_e, blocks_e, finalized_shape_m = dispatch_expert_finalize(
        sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
        max_tokens, hidden, local_expert_start, block_m, _workspace=ws_expert)
    assert finalized_shape_m == expert_shape_m

    # --- Validate merged preprocess kernel vs split (bit-identical metadata) ---
    if os.getenv("VALIDATE_MERGED", "0") == "1":
        ws2 = create_expert_preprocess_workspace(
            num_local_experts, world_size, max_tokens, device)
        gl2, ra2, rs2, sm2, rc2, blk2, sh2 = dispatch_expert_preprocess_merged(
            sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
            max_tokens, hidden, local_expert_start, block_m, generation=0,
            sync=True, _workspace=ws2)
        torch.cuda.synchronize()
        checks = {
            "grouped_layout": (gl_e, gl2),
            "rank_addr_a": (ra_e, ra2),
            "rank_addr_sfa": (rs_e, rs2),
            "rank_split_m": (sm_e, sm2),
            "rank_counts": (rc_e, rc2),
            "masked_m": (ws_expert.masked_m, ws2.masked_m),
        }
        allok = True
        for name, (a, b) in checks.items():
            eq = bool(torch.equal(a, b))
            allok = allok and eq
            if rank == 0:
                print(f"    [VALIDATE_MERGED] {name}: {(chr(0x2713) if eq else chr(0x2717))} "
                      f"(split_shape={tuple(a.shape)})")
        ok_scalar = (blk2 == blocks_e) and (sh2 == finalized_shape_m)
        if rank == 0:
            print(f"    [VALIDATE_MERGED] total_m_blocks={blk2}=={blocks_e}, shape_m={sh2}=={finalized_shape_m}: "
                  f"{(chr(0x2713) if ok_scalar else chr(0x2717))}")
            result = "ALL MATCH" if (allok and ok_scalar) else "MISMATCH!"
            print(f"    [VALIDATE_MERGED] RESULT: {result}")
    if rank == 0:
        print(f"  block_m={block_m}, shape_m={expert_shape_m}, expected_m={expert_expected_m}")

    if rank == 0:
        k_half = hidden // 2
        bc_sms, bc_bm, bc_bn, bc_bk, bc_wm, bc_wn, bc_stages, bc_smem = get_gemm_configs(
            expert_shape_m, expert_expected_m, num_local_experts, N, k_half,
            padded_m=max_tokens)
        print(f"  Block-copy fused config ({FUSED_GEMM_GROUPING}): block_m={bc_bm}, block_n={bc_bn}, block_k={bc_bk}, "
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
        _mxfp4_quantize_to_sym_buffer(
            x, topk_ids, sym_buf,
            num_local_experts=num_total_experts,
            num_total_experts=num_total_experts,
            max_tokens_per_expert=max_tokens)
        torch.cuda.synchronize()
        dist.barrier()
        refresh_expert_preprocess(
            sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
            max_tokens, hidden, local_expert_start, block_m, ws_expert, sync=True)

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
        bc_sms, bc_bm, bc_bn, bc_bk, bc_wm, bc_wn, bc_stages, bc_smem = get_gemm_configs(
            expert_shape_m, expert_expected_m, num_local_experts, N, bc_k_half,
            padded_m=max_tokens)
        print(f"  expert_shape_m={expert_shape_m}, blocks={blocks_e}, block_m={block_m}")
        print(f"  Block-copy config: block_m={bc_bm}, block_n={bc_bn}, block_k={bc_bk}, "
              f"warp_m={bc_wm}, warp_n={bc_wn}, stages={bc_stages}, num_sms={bc_sms}, smem={bc_smem}")

    # All-gather sym bufs (used by the local-to-local P2P isolation experiment
    # below). SFA is produced on-device by the kernel — no host build_merged_sfa.
    all_sym_bufs = [torch.zeros_like(sym_buf) for _ in range(world_size)]
    dist.all_gather(all_sym_bufs, sym_buf)
    dist.barrier()

    # ---- DG_SFA_SOURCE=host diagnostic: build the pre-0f40bad host merged_sfa ----
    # The GEMM read side is the suspected high-latency-machine regression (see
    # SFA_COPY_OPT_HANDOFF.md). With DG_SFA_SOURCE=host we rebuild the tightly-
    # packed host merged_sfa here (from the parity-0 warmup snapshot; x/topk are
    # fixed so the scale values are generation-independent) and feed it to the
    # block-copy GEMM, which then reads ptr_scale_A from it (K-stride = M) instead
    # of the GPU-side local_sfa_buf. gpu mode leaves merged_sfa_addrs=None so the
    # kernel path is byte-for-byte the current default.
    sfa_source = os.getenv('DG_SFA_SOURCE', 'gpu').lower()
    merged_sfa_addrs = None
    _merged_sfa_keep = None  # keep the backing tensor alive across kernel launches
    if sfa_source == 'host':
        _merged_sfa_keep, merged_sfa_addrs = build_merged_sfa(
            all_sym_bufs, num_local_experts, local_expert_start,
            num_total_experts, max_tokens, hidden, world_size, device)
        if rank == 0:
            print("  DG_SFA_SOURCE=host: block-copy GEMM reads host-built "
                  "merged_sfa (K-stride=M); NCB sweep + kernel-only reflect this.")

    # ---- Benchmark non-fused (pipeline throughput) ----
    fixed_expected_m = max(int(pc.max().item()), 1)
    # FORCE_EXPECTED_M must steer the non-fused path too. It already forces the
    # FUSED block_m (get_gemm_configs), so if the non-fused kept its own real
    # expected_m the "vs non-fused" ratio would compare fused@block_m=256 against
    # non-fused@block_m=128 — an unfair cross-block_m comparison. Apply it
    # symmetrically so both sides tune to the same expected_m -> same block_m.
    # (configs are passed explicitly and pc2 carries the real per-expert counts,
    # so steering expected_m only changes tile selection, not the work done.)
    _force_em = int(os.getenv('FORCE_EXPECTED_M', '0'))
    if _force_em > 0:
        fixed_expected_m = _force_em
    nf_block_m = int(os.getenv('NF_BLOCK_M', '128'))
    nf_block_n = int(os.getenv('NF_BLOCK_N', '0'))  # apple-to-apple: force NF block_n (e.g. 512)
    if nf_block_m > 0:
        nf_configs = get_best_configs_fp4(
            fixed_expected_m * num_local_experts, fixed_expected_m, N, hidden // 2,
            num_local_experts, get_num_sms(), gemm_type=GemmType.GroupedNoPad)
        if nf_block_n > 0:
            # Give NF the same block_n lever as fused (warp_n=64 keeps sfb iter<=4).
            from deep_gemm.jit_kernels.gemm_fp4 import get_smem_config_fp4
            _s, _bm, _bn, _bk, _wm, _wn, _st, _sm = nf_configs
            _sm = get_smem_config_fp4(num_stages=_st, block_m=_bm, block_n=nf_block_n,
                                      warp_m=_wm, warp_n=64, block_k=_bk)
            nf_configs = (_s, _bm, nf_block_n, _bk, _wm, 64, _st, _sm)
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
        # ---- MNK PARTITION CHECK: 确认融合 vs 非融合底层用同一套 MNK 切分 ----
        # 两条路径都走 GroupedMasked kernel；get_best_configs 只按 expected_m/n/k
        # /num_groups/num_sms 选 tile（gemm_type 非 Dense 时不影响选择）。这里把两
        # 边实际解析出的 (block_m,n,k / warp / stages / num_sms) 与 expected_m 并排
        # 打出来，逐次核对是否一致；不一致就说明 M 分布让 expected_m 分叉了。
        fused_expected_m = expert_expected_m
        fu_sms, fu_bm, fu_bn, fu_bk, fu_wm, fu_wn, fu_stages, fu_smem = get_gemm_configs(
            expert_shape_m, expert_expected_m, num_local_experts, N, hidden // 2,
            padded_m=max_tokens)
        pc_list = pc.tolist()
        fused_tile = (fu_bm, fu_bn, fu_bk, fu_wm, fu_wn, fu_stages, fu_sms)
        nf_tile = (nf_bm, nf_bn, nf_bk, nf_wm, nf_wn, nf_stages, nf_sms)
        print(f"  --- MNK partition check ---")
        print(f"    per-expert tokens: min={min(pc_list)}, max={max(pc_list)}, "
              f"avg={sum(pc_list)/len(pc_list):.1f}, sum={sum(pc_list)} (experts={num_local_experts})")
        print(f"    FUSED   (masked): expected_m={fused_expected_m}, "
              f"tile(bm,bn,bk,wm,wn,st,sms)={fused_tile}, padded_rows={max_tokens}")
        print(f"    NONFUSED(masked): expected_m={fixed_expected_m}, "
              f"tile(bm,bn,bk,wm,wn,st,sms)={nf_tile}, padded_rows={pf.shape[1]}")
        print(f"    -> TILE MATCH: {fused_tile == nf_tile}  "
              f"(N={N}, K/2={hidden//2}, num_groups={num_local_experts})")
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
    nf_pipeline_ms = distributed_max_scalar(
        ev_nfp[0].elapsed_time(ev_nfp[1]) / pipe_iters, device)

    # Minimal ASYS/HGTX capture mode.  All allocation, JIT compilation and warmup
    # stay outside the outer FULL_PIPELINES range; inside it we submit exactly two
    # steady-state workloads, labelled NONFUSED_FULL and FUSED_FULL.  This avoids
    # mixing correctness, isolation, GEMM-only and timing-breakdown kernels into a
    # report intended to compare the two production pipelines.
    if int(os.getenv('ASYS_FULL_ONLY', '0')):
        profile_iters = int(os.getenv('ASYS_FULL_ITERS', '10'))
        if profile_iters < 1:
            raise ValueError('ASYS_FULL_ITERS must be positive')
        sfa_overlap = os.getenv('DG_SFA_OVERLAP', '0') != '0'
        stream = torch.cuda.current_stream()
        bc_fp4_prof, bc_sfa_prof, bc_flags_prof = create_block_copy_buffers(
            num_local_experts, world_size, max_tokens, hidden, block_m, device)
        out_bc_prof = create_fused_output(
            expert_shape_m, num_local_experts, max_tokens, N, device)

        # Compile and warm the exact fused generation/parity path that is captured.
        gl_prof, ra_prof, rs_prof, sm_prof, rc_prof = gl_e, ra_e, rs_e, sm_e, rc_e
        for wi in range(num_warmup):
            g = 70000 + wi + 1
            _mxfp4_quantize_to_sym_buffer(
                x, topk_ids, sym_buf,
                num_local_experts=num_total_experts,
                num_total_experts=num_total_experts,
                max_tokens_per_expert=max_tokens, generation=g,
                sym_buf_addrs=sym_buf_addrs, rank_idx=rank,
                num_ranks=world_size)
            gl_prof, ra_prof, rs_prof, sm_prof, rc_prof, _, _ = \
                refresh_expert_preprocess(
                    sym_buf_addrs, rank, world_size, num_local_experts,
                    num_total_experts, max_tokens, hidden, local_expert_start,
                    block_m, generation=g, sync=False, _workspace=ws_expert,
                    local_sfa_buf=(bc_sfa_prof if sfa_overlap else None),
                    copy_ready_flags=(bc_flags_prof if sfa_overlap else None))
            fused_dispatch_block_copy_gemm1_fp4(
                (W_fp4, W_scale_u16), out_bc_prof,
                gl_prof, ra_prof, rs_prof, sm_prof, rc_prof,
                expert_shape_m, max_tokens, world_size,
                expected_m=expert_expected_m,
                local_fp4_buf=bc_fp4_prof, local_sfa_buf=bc_sfa_prof,
                copy_ready_flags=bc_flags_prof,
                num_copy_blocks=3, k_tiles_per_flag=K_TILES_PER_FLAG,
                merged_sfa_addrs=merged_sfa_addrs, rank_idx=rank,
                sfa_preprocessed=sfa_overlap)
        torch.cuda.synchronize()
        dist.barrier()

        nf_start = torch.cuda.Event(enable_timing=True)
        nf_end = torch.cuda.Event(enable_timing=True)
        fused_start = torch.cuda.Event(enable_timing=True)
        fused_end = torch.cuda.Event(enable_timing=True)

        # Rank 0 owns the capture trigger and human-readable child ranges.  The
        # boundary barriers keep all eight ranks inside the capture window; ASYS
        # is invoked without PCCL tracing, so they do not add another workload.
        if rank == 0:
            torch.cuda.nvtx.range_push('FULL_PIPELINES')
        dist.barrier()

        if rank == 0:
            torch.cuda.nvtx.range_push('NONFUSED_FULL')
        nf_start.record(stream)
        for _ in range(profile_iters):
            pf2, lhs2, pc2, _, _, _ = nonfused_dispatch(
                ep_buffer, x, topk_ids_i64, num_tokens, num_total_experts)
            deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
                (pf2.contiguous(), lhs2),
                (W_fp4, W_scale_u16), None, nf_pipe_out,
                pc2.to(torch.int32), fixed_expected_m, configs=nf_configs)
        nf_end.record(stream)
        torch.cuda.synchronize()
        if rank == 0:
            torch.cuda.nvtx.range_pop()

        if rank == 0:
            torch.cuda.nvtx.range_push('FUSED_FULL')
        fused_start.record(stream)
        for i in range(profile_iters):
            g = 71000 + i + 1
            _mxfp4_quantize_to_sym_buffer(
                x, topk_ids, sym_buf,
                num_local_experts=num_total_experts,
                num_total_experts=num_total_experts,
                max_tokens_per_expert=max_tokens, generation=g,
                sym_buf_addrs=sym_buf_addrs, rank_idx=rank,
                num_ranks=world_size)
            gl_prof, ra_prof, rs_prof, sm_prof, rc_prof, _, _ = \
                refresh_expert_preprocess(
                    sym_buf_addrs, rank, world_size, num_local_experts,
                    num_total_experts, max_tokens, hidden, local_expert_start,
                    block_m, generation=g, sync=False, _workspace=ws_expert,
                    local_sfa_buf=(bc_sfa_prof if sfa_overlap else None),
                    copy_ready_flags=(bc_flags_prof if sfa_overlap else None))
            fused_dispatch_block_copy_gemm1_fp4(
                (W_fp4, W_scale_u16), out_bc_prof,
                gl_prof, ra_prof, rs_prof, sm_prof, rc_prof,
                expert_shape_m, max_tokens, world_size,
                expected_m=expert_expected_m,
                local_fp4_buf=bc_fp4_prof, local_sfa_buf=bc_sfa_prof,
                copy_ready_flags=bc_flags_prof,
                num_copy_blocks=3, k_tiles_per_flag=K_TILES_PER_FLAG,
                merged_sfa_addrs=merged_sfa_addrs, rank_idx=rank,
                sfa_preprocessed=sfa_overlap)
        fused_end.record(stream)
        torch.cuda.synchronize()
        if rank == 0:
            torch.cuda.nvtx.range_pop()

        dist.barrier()
        if rank == 0:
            torch.cuda.nvtx.range_pop()

        nf_full_ms = distributed_max_scalar(
            nf_start.elapsed_time(nf_end) / profile_iters, device)
        fused_full_ms = distributed_max_scalar(
            fused_start.elapsed_time(fused_end) / profile_iters, device)
        if rank == 0:
            print('\n  ASYS full-pipeline-only capture complete:')
            print(f'    NONFUSED_FULL: {nf_full_ms * 1000:.1f} us/iter '
                  f'({profile_iters} iterations)')
            print(f'    FUSED_FULL:    {fused_full_ms * 1000:.1f} us/iter '
                  f'({profile_iters} iterations)')
        ep_buffer.destroy()
        return True

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
    nf_gemm_only_ms = distributed_max_scalar(
        ev_nfgo[0].elapsed_time(ev_nfgo[1]) / pipe_iters, device)

    # ---- Production GroupedNoPad GEMM-only (no copy) — isolate NoPad grouping vs
    #      custom-fused-kernel overhead. Synthetic inputs at prod shape (timing only). ----
    if int(os.getenv('NOPAD_GEMM_BENCH', '0')):
        m_np = expert_shape_m
        A_np = torch.randint(0, 256, (m_np, hidden // 2), dtype=torch.uint8, device=device)
        Asc_np = torch.randint(0, 256, (m_np, hidden // 32), dtype=torch.uint8, device=device)
        Asc_np = preprocess_mxfp4_scales(Asc_np)
        out_np = torch.empty(m_np, N, dtype=torch.bfloat16, device=device)
        per = max(m_np // num_local_experts, 1)
        m_idx = (torch.arange(m_np, device=device) // per).clamp(max=num_local_experts - 1).to(torch.int32)
        for _ in range(num_warmup):
            deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad(
                (A_np, Asc_np), (W_fp4, W_scale_u16), None, out_np, m_idx)
        torch.cuda.synchronize()
        ev_np = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ev_np[0].record()
        for _ in range(pipe_iters):
            deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad(
                (A_np, Asc_np), (W_fp4, W_scale_u16), None, out_np, m_idx)
        ev_np[1].record()
        torch.cuda.synchronize()
        nopad_gemm_ms = distributed_max_scalar(
            ev_np[0].elapsed_time(ev_np[1]) / pipe_iters, device)
        if rank == 0:
            print(f"  NoPad-GEMM-only (production, no copy, m={m_np}): {nopad_gemm_ms:.3f} ms  "
                  f"| masked-GEMM-only: {nf_gemm_only_ms:.3f} ms")

    # ---- Block-Copy NCB sweep ----
    dist.barrier()
    bc_fp4_perf, bc_sfa_perf, bc_flags_perf = create_block_copy_buffers(
        num_local_experts, world_size, max_tokens, hidden, block_m, device)
    out_bc = create_fused_output(
        expert_shape_m, num_local_experts, max_tokens, N, device)

    stream = torch.cuda.current_stream()
    bc_results = {}

    # ---- DG_SFA_SOURCE correctness self-check ----
    # The host read path must be bit-exact with the (independently CPU-verified in
    # Test 1) gpu read path on active rows. Metadata (gl_e/ra_e/...) and sym_buf
    # are still at the parity-0 warmup generation here — the same snapshot
    # merged_sfa was built from — so gpu and host read identical scales.
    if sfa_source == 'host':
        chk_counts = [
            sum(get_expert_token_counts(
                all_sym_bufs, local_expert_start + le, world_size,
                data_base=0, generation=0, max_tokens=max_tokens))
            for le in range(num_local_experts)]
        out_gpu = create_fused_output(expert_shape_m, num_local_experts, max_tokens, N, device)
        out_host = create_fused_output(expert_shape_m, num_local_experts, max_tokens, N, device)
        for out_t, host_flag in ((out_gpu, False), (out_host, True)):
            fused_dispatch_block_copy_gemm1_fp4(
                (W_fp4, W_scale_u16), out_t, gl_e, ra_e, rs_e, sm_e, rc_e,
                expert_shape_m, max_tokens, world_size,
                expected_m=expert_expected_m,
                local_fp4_buf=bc_fp4_perf, local_sfa_buf=bc_sfa_perf, copy_ready_flags=bc_flags_perf,
                num_copy_blocks=3, k_tiles_per_flag=K_TILES_PER_FLAG,
                merged_sfa_addrs=(merged_sfa_addrs if host_flag else None),
                sfa_source_host=host_flag)
        torch.cuda.synchronize()
        chk_ok = all(
            torch.equal(out_gpu[le, :c], out_host[le, :c])
            for le, c in enumerate(chk_counts) if c > 0)
        chk_t = torch.tensor([1 if chk_ok else 0], dtype=torch.int32, device=device)
        dist.all_reduce(chk_t, op=dist.ReduceOp.MIN)
        chk_ok = chk_t.item() == 1
        if rank == 0:
            print(f"  DG_SFA_SOURCE host-vs-gpu bit-exact self-check: "
                  f"{'PASSED' if chk_ok else 'FAILED'}")

    sfa_overlap = os.getenv('DG_SFA_OVERLAP', '0') != '0'
    if sfa_overlap and rank == 0:
        print("  DG_SFA_OVERLAP=1: dynamic-count metadata || SFA reshape")

    # The overlap kernel derives its SFA offsets from dynamic source-rank counts
    # without consuming rank_addr_sfa/rank_split_m/rank_counts.  Validate both a
    # genuinely non-uniform route and the timed fixed route against the accepted
    # metadata-driven reshape.  Each A/B reuses one exact staging generation:
    # quant's atomic token-row allocation need not be stable across launches.
    if sfa_overlap:
        route_gen = torch.Generator(device='cpu')
        route_gen.manual_seed(20260717 + rank)
        dynamic_topk = torch.topk(
            torch.rand(num_tokens, num_total_experts, generator=route_gen),
            topk, dim=-1, largest=True, sorted=False)[1].to(
                device=device, dtype=torch.int32)
        for route_name, route_topk, check_gen in (
                ("dynamic", dynamic_topk, 8201),
                ("fixed", topk_ids, 8301)):
            bc_sfa_perf.zero_()
            _mxfp4_quantize_to_sym_buffer(
                x, route_topk, sym_buf,
                num_local_experts=num_total_experts,
                num_total_experts=num_total_experts,
                max_tokens_per_expert=max_tokens, generation=check_gen,
                sym_buf_addrs=sym_buf_addrs, rank_idx=rank, num_ranks=world_size)
            refresh_expert_preprocess(
                sym_buf_addrs, rank, world_size, num_local_experts,
                num_total_experts, max_tokens, hidden, local_expert_start,
                block_m, generation=check_gen, sync=False,
                _workspace=ws_expert, local_sfa_buf=bc_sfa_perf,
                copy_ready_flags=bc_flags_perf)
            torch.cuda.synchronize()
            overlap_sfa = bc_sfa_perf.clone()

            gl_chk, _, rsfa_chk, split_chk, cnt_chk, _, _ = \
                refresh_expert_preprocess(
                    sym_buf_addrs, rank, world_size, num_local_experts,
                    num_total_experts, max_tokens, hidden, local_expert_start,
                    block_m, generation=check_gen, sync=False,
                    _workspace=ws_expert)
            bc_sfa_perf.zero_()
            dispatch_sfa_preprocess(
                gl_chk, rsfa_chk, split_chk, cnt_chk,
                bc_sfa_perf, bc_flags_perf,
                num_ranks=world_size, block_m=block_m,
                k_scale_blocks=ceil_div(hidden // 2, 32),
                max_tokens_per_expert=max_tokens,
                num_sms=get_num_sms(), num_threads=256)
            torch.cuda.synchronize()
            overlap_ok = torch.equal(overlap_sfa, bc_sfa_perf)
            overlap_ok_t = torch.tensor(
                [int(overlap_ok)], dtype=torch.int32, device=device)
            dist.all_reduce(overlap_ok_t, op=dist.ReduceOp.MIN)
            if overlap_ok_t.item() != 1:
                raise AssertionError(
                    f"{route_name} SFA overlap differs from generic reshape")
            if rank == 0:
                print(f"  {route_name} SFA overlap vs generic reshape: BIT-EXACT")

    default_ncb_sweep = '3' if is_prod_8gpu else '4,8,12'
    ncb_list = [int(x) for x in os.getenv('NCB_SWEEP', default_ncb_sweep).split(',')]
    for ncb in ncb_list:
        # Warmup
        for wi in range(num_warmup):
            if sfa_overlap:
                g = 8500 + ncb * 100 + wi + 1
                _mxfp4_quantize_to_sym_buffer(
                    x, topk_ids, sym_buf,
                    num_local_experts=num_total_experts,
                    num_total_experts=num_total_experts,
                    max_tokens_per_expert=max_tokens, generation=g,
                    sym_buf_addrs=sym_buf_addrs, rank_idx=rank, num_ranks=world_size)
                refresh_expert_preprocess(
                    sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
                    max_tokens, hidden, local_expert_start, block_m,
                    generation=g, sync=False, _workspace=ws_expert,
                    local_sfa_buf=bc_sfa_perf, copy_ready_flags=bc_flags_perf)
            fused_dispatch_block_copy_gemm1_fp4(
                (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
                expert_shape_m, max_tokens, world_size,
                expected_m=expert_expected_m,
                local_fp4_buf=bc_fp4_perf, local_sfa_buf=bc_sfa_perf, copy_ready_flags=bc_flags_perf,
                num_copy_blocks=ncb, k_tiles_per_flag=K_TILES_PER_FLAG,
                merged_sfa_addrs=merged_sfa_addrs, rank_idx=rank,
                sfa_preprocessed=sfa_overlap)
        torch.cuda.synchronize()

        # Benchmark: full pipeline (quant + preprocess + block-copy GEMM)
        bc_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        bc_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        gen_bc = 9000 + ncb * 100
        for i in range(num_iters):
            g = gen_bc + i + 1
            bc_start_events[i].record(stream)
            _mxfp4_quantize_to_sym_buffer(x, topk_ids, sym_buf,
                num_local_experts=num_total_experts, num_total_experts=num_total_experts,
                max_tokens_per_expert=max_tokens, generation=g,
                sym_buf_addrs=sym_buf_addrs, rank_idx=rank, num_ranks=world_size)
            refresh_expert_preprocess(
                sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
                max_tokens, hidden, local_expert_start, block_m,
                generation=g, sync=False, _workspace=ws_expert,
                local_sfa_buf=(bc_sfa_perf if sfa_overlap else None),
                copy_ready_flags=(bc_flags_perf if sfa_overlap else None))
            fused_dispatch_block_copy_gemm1_fp4(
                (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
                expert_shape_m, max_tokens, world_size,
                expected_m=expert_expected_m,
                local_fp4_buf=bc_fp4_perf, local_sfa_buf=bc_sfa_perf, copy_ready_flags=bc_flags_perf,
                num_copy_blocks=ncb, k_tiles_per_flag=K_TILES_PER_FLAG,
                merged_sfa_addrs=merged_sfa_addrs, rank_idx=rank,
                sfa_preprocessed=sfa_overlap)
            bc_end_events[i].record(stream)
        torch.cuda.synchronize()
        bc_t = sorted(distributed_max_samples(
            [s.elapsed_time(e) for s, e in zip(bc_start_events, bc_end_events)], device))
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
    _prof_len = 7 if os.getenv('DG_QUANT_PROFILE_P1SPLIT', '0') != '0' else 3
    _prof_clk = torch.zeros(_prof_len, dtype=torch.int64, device=device)
    for i in range(n_breakdown):
        g = gen_bd + i + 1
        ev_q_s[i].record(stream)
        _mxfp4_quantize_to_sym_buffer(x, topk_ids, sym_buf,
            num_local_experts=num_total_experts, num_total_experts=num_total_experts,
            max_tokens_per_expert=max_tokens, generation=g,
            sym_buf_addrs=sym_buf_addrs, rank_idx=rank, num_ranks=world_size,
            profile_clocks=_prof_clk)
        ev_q_e[i].record(stream)
        ev_p_s[i].record(stream)
        refresh_expert_preprocess(
            sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
            max_tokens, hidden, local_expert_start, block_m,
            generation=g, sync=False, _workspace=ws_expert)
        ev_p_e[i].record(stream)
    torch.cuda.synchronize()
    quant_times = sorted(distributed_max_samples(
        [ev_q_s[i].elapsed_time(ev_q_e[i]) for i in range(n_breakdown)], device))
    preproc_times = sorted(distributed_max_samples(
        [ev_p_s[i].elapsed_time(ev_p_e[i]) for i in range(n_breakdown)], device))
    quant_ms = quant_times[len(quant_times) // 2]
    preproc_ms = preproc_times[len(preproc_times) // 2]
    if rank == 0:
        _pc = _prof_clk.cpu().tolist()
        _p1, _p2, _tot = _pc[0], _pc[1], max(_pc[2], 1)
        print(f"  quant in-kernel clock64 (block0, one token): "
              f"P1 quantize={_p1:,} ({100*_p1/_tot:.0f}%) | "
              f"P2 scatter={_p2:,} ({100*_p2/_tot:.0f}%) | total={_pc[2]:,} cyc")
        if len(_pc) >= 7:
            print("  quant P1 split clock64 (thread32): "
                  f"setup={_pc[3]:,} | load+compute={_pc[4]:,} | "
                  f"first barrier={_pc[5]:,} | pack+barrier={_pc[6]:,} cyc")
    # ---- preprocess split: generation=0 skips the cross-rank arrival barrier
    #      (Phase 2 is gated on generation>0). So preproc(gen>0) - preproc(gen=0)
    #      isolates the barrier wait; preproc(gen=0) = remote count reads + compute
    #      + launch. Same runtime kernel (generation is a runtime arg, no recompile).
    # Generation-tagged count metadata is only unpacked on gen>0. Rebuild parity-0
    # metadata in the raw-count format before timing this no-barrier diagnostic path.
    _mxfp4_quantize_to_sym_buffer(
        x, topk_ids, sym_buf,
        num_local_experts=num_total_experts,
        num_total_experts=num_total_experts,
        max_tokens_per_expert=max_tokens)
    torch.cuda.synchronize()
    dist.barrier()
    ev_p0_s = [torch.cuda.Event(enable_timing=True) for _ in range(n_breakdown)]
    ev_p0_e = [torch.cuda.Event(enable_timing=True) for _ in range(n_breakdown)]
    for i in range(n_breakdown):
        ev_p0_s[i].record(stream)
        refresh_expert_preprocess(
            sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
            max_tokens, hidden, local_expert_start, block_m,
            generation=0, sync=False, _workspace=ws_expert)
        ev_p0_e[i].record(stream)
    torch.cuda.synchronize()
    preproc0_times = sorted(distributed_max_samples(
        [ev_p0_s[i].elapsed_time(ev_p0_e[i]) for i in range(n_breakdown)], device))
    preproc0_ms = preproc0_times[len(preproc0_times) // 2]

    # ---- DeepEP dispatch cost (non-fused baseline) — Q2 target reference ----
    # Fused pre-GEMM overhead (quant+preprocess) should be <= 1/2 of this.
    dist.barrier()
    ev_d_s = [torch.cuda.Event(enable_timing=True) for _ in range(n_breakdown)]
    ev_d_e = [torch.cuda.Event(enable_timing=True) for _ in range(n_breakdown)]
    for i in range(n_breakdown):
        ev_d_s[i].record(stream)
        nonfused_dispatch(ep_buffer, x, topk_ids_i64, num_tokens, num_total_experts)
        ev_d_e[i].record(stream)
    torch.cuda.synchronize()
    deepep_times = sorted(distributed_max_samples(
        [ev_d_s[i].elapsed_time(ev_d_e[i]) for i in range(n_breakdown)], device))
    deepep_ms = deepep_times[len(deepep_times) // 2]

    # ---- in-kernel clock64 breakdown of preprocess (gen>0): 8 phases ----
    dist.barrier()
    dbg_cyc = torch.zeros(8, dtype=torch.int64, device=device)
    for i in range(10):  # collect last (warm) invocation's cycles
        g = 70000 + i
        # MUST run quant first: it pushes the arrival flag the preprocess
        # barrier (gen>0) waits on — else preprocess deadlocks.
        _mxfp4_quantize_to_sym_buffer(x, topk_ids, sym_buf,
            num_local_experts=num_total_experts, num_total_experts=num_total_experts,
            max_tokens_per_expert=max_tokens, generation=g,
            sym_buf_addrs=sym_buf_addrs, rank_idx=rank, num_ranks=world_size)
        refresh_expert_preprocess(
            sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
            max_tokens, hidden, local_expert_start, block_m,
            generation=g, sync=False, _workspace=ws_expert, dbg_cyc=dbg_cyc)
    torch.cuda.synchronize()
    dbg_by_rank = [torch.zeros_like(dbg_cyc) for _ in range(world_size)]
    dist.all_gather(dbg_by_rank, dbg_cyc)
    dc_by_rank = torch.stack(dbg_by_rank).cpu().tolist()
    slowest_profile_rank = max(range(world_size), key=lambda r: sum(dc_by_rank[r]))
    dc = dc_by_rank[slowest_profile_rank]
    dc_tot = max(sum(dc), 1)

    if rank == 0:
        phase_names = (
            'prepare sym setup', 'arrival poll+barrier', 'remote count reads',
            'count reduce+shape', 'finalize sym setup', 'M-block counts',
            'expert prefix', 'greedy metadata pack')
        print(f"\n  preprocess in-kernel clock64 "
              f"(cycles, one gen>0 call, slowest rank={slowest_profile_rank}):")
        for name, cycles in zip(phase_names, dc):
            print(f"    {name:<22s}: {cycles:>9,} ({100*cycles/dc_tot:4.1f}%)")
        print(f"    {'profiled total':<22s}: {dc_tot:>9,} cyc")
        print(f"\n  Pipeline Overhead Breakdown:")
        print(f"    quant + sym_buf_zero:   {quant_ms:.3f} ms")
        print(f"    expert_preprocess:      {preproc_ms:.3f} ms")
        print(f"      - preproc gen=0 (no barrier: remote-read+compute+launch): {preproc0_ms:.3f} ms")
        print(f"      - arrival barrier wait (gen>0 - gen0):                    {max(preproc_ms - preproc0_ms, 0.0):.3f} ms")
        print(f"    sum:                    {quant_ms + preproc_ms:.3f} ms")
        print(f"    DeepEP dispatch (nonfused): {deepep_ms:.3f} ms")
        print(f"    Q2 target (<=1/2 DeepEP):   {deepep_ms/2:.3f} ms  "
              f"[{'MET' if (quant_ms+preproc_ms) <= deepep_ms/2 else 'NOT MET'}]")

    # ---- Pipeline without preprocess (upper bound of preprocess fusion savings) ----
    # Skip expert prepare/finalize, use metadata from warmup/breakdown.
    # Routing is uniform → metadata unchanged between iterations.
    no_preproc_pipeline_ms = 0.0
    no_preproc_quant_ms = 0.0
    no_preproc_consumer_ms = 0.0
    no_preproc_closure_us = 0.0
    preproc_savings_ms = 0.0
    bc_kernel_only_ms = 0.0
    bc_kernel_std = 0.0
    local_kernel_ms = 0.0
    local_kernel_std = 0.0
    if not SKIP_ISOLATION:
        dist.barrier()
        gen_np = 80000
        for i in range(num_warmup):
            g = gen_np + i + 1
            _mxfp4_quantize_to_sym_buffer(x, topk_ids, sym_buf,
                num_local_experts=num_total_experts, num_total_experts=num_total_experts,
                max_tokens_per_expert=max_tokens, generation=g,
                sym_buf_addrs=sym_buf_addrs, rank_idx=rank, num_ranks=world_size)
            refresh_expert_preprocess(
                sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
                max_tokens, hidden, local_expert_start, block_m,
                generation=g, sync=False, _workspace=ws_expert)
            fused_dispatch_block_copy_gemm1_fp4(
                (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
                expert_shape_m, max_tokens, world_size,
                expected_m=expert_expected_m,
                local_fp4_buf=bc_fp4_perf, local_sfa_buf=bc_sfa_perf, copy_ready_flags=bc_flags_perf,
                num_copy_blocks=best_ncb, k_tiles_per_flag=K_TILES_PER_FLAG,
                merged_sfa_addrs=merged_sfa_addrs, rank_idx=rank)
        torch.cuda.synchronize()

        np_start = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        np_quant_end = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        np_end = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        for i in range(num_iters):
            g = gen_np + num_warmup + i + 1
            np_start[i].record(stream)
            _mxfp4_quantize_to_sym_buffer(x, topk_ids, sym_buf,
                num_local_experts=num_total_experts, num_total_experts=num_total_experts,
                max_tokens_per_expert=max_tokens, generation=g,
                sym_buf_addrs=sym_buf_addrs, rank_idx=rank, num_ranks=world_size)
            np_quant_end[i].record(stream)
            fused_dispatch_block_copy_gemm1_fp4(
                (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
                expert_shape_m, max_tokens, world_size,
                expected_m=expert_expected_m,
                local_fp4_buf=bc_fp4_perf, local_sfa_buf=bc_sfa_perf, copy_ready_flags=bc_flags_perf,
                num_copy_blocks=best_ncb, k_tiles_per_flag=K_TILES_PER_FLAG,
                merged_sfa_addrs=merged_sfa_addrs, rank_idx=rank)
            np_end[i].record(stream)
        torch.cuda.synchronize()
        np_times = sorted(distributed_max_samples(
            [np_start[i].elapsed_time(np_end[i]) for i in range(num_iters)], device))
        no_preproc_pipeline_ms = np_times[len(np_times) // 2]
        preproc_savings_ms = bc_pipeline_ms - no_preproc_pipeline_ms

        # Same-iteration closed timing. Select the rank with the largest TOTAL
        # for each sample, then take that same rank's quant and consumer segments;
        # this avoids subtracting independently aggregated/median measurements.
        np_local_split = torch.tensor([
            [np_start[i].elapsed_time(np_quant_end[i]),
             np_quant_end[i].elapsed_time(np_end[i]),
             np_start[i].elapsed_time(np_end[i])]
            for i in range(num_iters)
        ], dtype=torch.float32, device=device)
        np_split_by_rank = [torch.empty_like(np_local_split) for _ in range(world_size)]
        dist.all_gather(np_split_by_rank, np_local_split)
        np_split_all = torch.stack(np_split_by_rank).cpu()
        np_critical_split = []
        for i in range(num_iters):
            critical_rank = int(torch.argmax(np_split_all[:, i, 2]).item())
            np_critical_split.append(np_split_all[critical_rank, i].tolist())
        np_quant_segments = sorted(v[0] for v in np_critical_split)
        np_consumer_segments = sorted(v[1] for v in np_critical_split)
        np_total_segments = sorted(v[2] for v in np_critical_split)
        np_closure = sorted(abs(v[2] - v[0] - v[1]) for v in np_critical_split)
        no_preproc_quant_ms = np_quant_segments[len(np_quant_segments) // 2]
        no_preproc_consumer_ms = np_consumer_segments[len(np_consumer_segments) // 2]
        no_preproc_pipeline_ms = np_total_segments[len(np_total_segments) // 2]
        no_preproc_closure_us = np_closure[len(np_closure) // 2] * 1000.0
        preproc_savings_ms = bc_pipeline_ms - no_preproc_pipeline_ms

        if rank == 0:
            print("\n  Same-iteration no-preprocess split (critical-total rank):")
            print(f"    fresh quant + SFA push: {no_preproc_quant_ms:.3f} ms")
            print(f"    consumer after quant:   {no_preproc_consumer_ms:.3f} ms")
            print(f"    closed total:           {no_preproc_pipeline_ms:.3f} ms")
            print(f"    closure residual:       {no_preproc_closure_us:.3f} us")

        # ---- Block-Copy kernel-only: copy+GEMM without quant/preprocess ----
        dist.barrier()
        for _ in range(num_warmup):
            fused_dispatch_block_copy_gemm1_fp4(
                (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
                expert_shape_m, max_tokens, world_size,
                expected_m=expert_expected_m,
                local_fp4_buf=bc_fp4_perf, local_sfa_buf=bc_sfa_perf, copy_ready_flags=bc_flags_perf,
                num_copy_blocks=best_ncb, k_tiles_per_flag=K_TILES_PER_FLAG,
                merged_sfa_addrs=merged_sfa_addrs, rank_idx=rank)
        torch.cuda.synchronize()
        bc_kernel_times = []
        for _ in range(pipe_iters):
            ev_s = torch.cuda.Event(enable_timing=True)
            ev_e = torch.cuda.Event(enable_timing=True)
            ev_s.record()
            fused_dispatch_block_copy_gemm1_fp4(
                (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
                expert_shape_m, max_tokens, world_size,
                expected_m=expert_expected_m,
                local_fp4_buf=bc_fp4_perf, local_sfa_buf=bc_sfa_perf, copy_ready_flags=bc_flags_perf,
                num_copy_blocks=best_ncb, k_tiles_per_flag=K_TILES_PER_FLAG,
                merged_sfa_addrs=merged_sfa_addrs, rank_idx=rank)
            ev_e.record()
            bc_kernel_times.append((ev_s, ev_e))
        torch.cuda.synchronize()
        bc_kernel_ms_list = sorted(distributed_max_samples(
            [s.elapsed_time(e) for s, e in bc_kernel_times], device))
        bc_kernel_only_ms = bc_kernel_ms_list[len(bc_kernel_ms_list) // 2]
        bc_kernel_std = (sum((t - bc_kernel_only_ms)**2 for t in bc_kernel_ms_list) / len(bc_kernel_ms_list)) ** 0.5

        # ---- P2P vs GEMM per-tile / per-wave profiling (kstripe_profile_buf) ----
        # Set KSTRIPE_PROFILE=1 to enable (default: disabled). Buffer records one
        # 4-int64 row per tile: [wait, mainloop, wave, gemm_cta]. We group by wave
        # to see whether only the first wave stalls on P2P (later waves fully hidden).
        if int(os.getenv('KSTRIPE_PROFILE', '0')):
            # Upper bound on tiles = num_m_blocks * num_n_blocks. Recompute the
            # block-copy config here so bc_bn is always in scope.
            bc_k_half_p = hidden // 2
            _, _bm_p, bc_bn_p, *_ = get_gemm_configs(
                expert_shape_m, expert_expected_m, num_local_experts, N, bc_k_half_p,
                padded_m=max_tokens)
            num_n_blocks_p = ceil_div(N, bc_bn_p)
            num_tiles_cap = (blocks_e + 4) * num_n_blocks_p  # +margin for m-block alignment
            ks_prof_buf = torch.zeros(num_tiles_cap * 4, dtype=torch.int64, device=device)
            # warm the exact call so the profiled iteration isn't cold
            for _ in range(5):
                fused_dispatch_block_copy_gemm1_fp4(
                    (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
                    expert_shape_m, max_tokens, world_size,
                    expected_m=expert_expected_m,
                    local_fp4_buf=bc_fp4_perf, local_sfa_buf=bc_sfa_perf, copy_ready_flags=bc_flags_perf,
                    num_copy_blocks=best_ncb, k_tiles_per_flag=K_TILES_PER_FLAG, rank_idx=rank)
            torch.cuda.synchronize()
            ks_ev0 = torch.cuda.Event(enable_timing=True)
            ks_ev1 = torch.cuda.Event(enable_timing=True)
            ks_ev0.record()
            fused_dispatch_block_copy_gemm1_fp4(
                (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
                expert_shape_m, max_tokens, world_size,
                expected_m=expert_expected_m,
                local_fp4_buf=bc_fp4_perf, local_sfa_buf=bc_sfa_perf, copy_ready_flags=bc_flags_perf,
                num_copy_blocks=best_ncb, k_tiles_per_flag=K_TILES_PER_FLAG,
                kstripe_profile_buf=ks_prof_buf, rank_idx=rank)
            ks_ev1.record()
            torch.cuda.synchronize()
            ks_wall_us = ks_ev0.elapsed_time(ks_ev1) * 1000.0

            if rank == 0:
                rec = ks_prof_buf.cpu().numpy().reshape(-1, 4)
                rec = rec[rec[:, 1] > 0]  # keep tiles that actually ran (mainloop>0)
                if len(rec) > 0:
                    wait_all = int(rec[:, 0].sum()); main_all = int(rec[:, 1].sum())
                    waves = rec[:, 2].astype(int)
                    n_waves = int(waves.max()) + 1
                    n_ctas = int(rec[:, 3].max()) + 1
                    print(f"\n  P2P vs GEMM Per-Wave Profiling (kstripe_profile_buf):")
                    print(f"    K_TILES_PER_FLAG={K_TILES_PER_FLAG}, tiles={len(rec)}, "
                          f"gemm_ctas={n_ctas}, waves={n_waves}")
                    print(f"    Overall exposure: {wait_all/main_all:.4f}  "
                          f"(wait={wait_all:,} / main={main_all:,})")
                    print(f"    {'wave':>4} {'tiles':>6} {'exposure':>9} "
                          f"{'avg_wait':>12} {'avg_compute':>12}")
                    for w in range(n_waves):
                        wr = rec[waves == w]
                        if len(wr) == 0:
                            continue
                        ww = int(wr[:, 0].sum()); wm = int(wr[:, 1].sum())
                        exp = ww / wm if wm > 0 else 0.0
                        avg_w = ww / len(wr)
                        # KTPF=0: wait 在 mainloop 之前的 tile 入口计时,与 mainloop
                        # 区间不重叠 → compute 即 mainloop 本身(不能再减 wait,否则高
                        # P2P 延迟机器上 wait>compute 会穿负)。KTPF>0: wait 的 spin 嵌
                        # 套在 mainloop 计时窗口内 → compute = mainloop - wait。
                        if K_TILES_PER_FLAG == 0:
                            avg_c = wm / len(wr)
                        else:
                            avg_c = (wm - ww) / len(wr)
                        print(f"    {w:>4} {len(wr):>6} {exp:>9.4f} "
                              f"{avg_w:>12,.0f} {avg_c:>12,.0f}")
                    if K_TILES_PER_FLAG == 0:
                        print(f"    NOTE: K_TILES_PER_FLAG=0 → wait once at tile entry, not per-stripe")
                    # --- Cumulative critical-path (per-CTA sum) vs wall-clock calibration ---
                    ctas = rec[:, 3].astype(int)
                    cum_wm = []   # per-CTA sum of (wait + mainloop)
                    cum_m = []    # per-CTA sum of mainloop only
                    for c in range(n_ctas):
                        cr = rec[ctas == c]
                        if len(cr) == 0:
                            continue
                        cum_wm.append(int((cr[:, 0] + cr[:, 1]).sum()))
                        cum_m.append(int(cr[:, 1].sum()))
                    crit_wm = max(cum_wm); crit_m = max(cum_m)
                    print(f"    Cumulative critical-path CTA: wait+main={crit_wm:,} cyc, "
                          f"main-only={crit_m:,} cyc  (of {n_ctas} GEMM CTAs)")
                    print(f"    Profiled wall (event): {ks_wall_us:,.1f} us  "
                          f"| exact_grid={os.getenv('FUSED_EXACT_GRID','0')}")
                    print(f"    Implied clock: {crit_wm/ks_wall_us/1e3:.3f} GHz (wait+main / wall)  "
                          f"→ out-of-mainloop = wall - main_time")
                else:
                    print(f"\n  P2P vs GEMM Per-Wave Profiling: no data (buffer empty)")

        # ---- Copy-only bandwidth sweep vs num_copy_blocks ----
        # Set COPY_BW_SWEEP=1. Launches ONLY the copy blocks (grid.x = ncb, no GEMM)
        # and host-times them, to find how many copy blocks are needed to saturate
        # the P2P copy bandwidth. Time drops ~1/ncb until bandwidth-bound, then flattens.
        if int(os.getenv('COPY_BW_SWEEP', '0')):
            dist.barrier()
            ncb_bw_list = [int(x) for x in os.getenv('COPY_NCB_LIST', '1,2,4,8,12,16,24').split(',')]
            bw_iters = 30
            # Approx bytes moved: total token rows * fp4 bytes/row (hidden/2).
            approx_bytes = expert_shape_m * (hidden // 2)
            results_bw = {}
            for ncb in ncb_bw_list:
                for _ in range(num_warmup):
                    fused_dispatch_block_copy_gemm1_fp4(
                        (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
                        expert_shape_m, max_tokens, world_size,
                        expected_m=expert_expected_m,
                        local_fp4_buf=bc_fp4_perf, local_sfa_buf=bc_sfa_perf, copy_ready_flags=bc_flags_perf,
                        num_copy_blocks=ncb, k_tiles_per_flag=0,
                        copy_only=True, rank_idx=rank)
                torch.cuda.synchronize()
                evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                       for _ in range(bw_iters)]
                for s, e in evs:
                    s.record(stream)
                    fused_dispatch_block_copy_gemm1_fp4(
                        (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
                        expert_shape_m, max_tokens, world_size,
                        expected_m=expert_expected_m,
                        local_fp4_buf=bc_fp4_perf, local_sfa_buf=bc_sfa_perf, copy_ready_flags=bc_flags_perf,
                        num_copy_blocks=ncb, k_tiles_per_flag=0,
                        copy_only=True, rank_idx=rank)
                    e.record(stream)
                torch.cuda.synchronize()
                ts = sorted([s.elapsed_time(e) for s, e in evs])
                results_bw[ncb] = ts[len(ts) // 2]
            if rank == 0:
                print(f"\n  Copy-only Bandwidth Sweep (copy blocks only, no GEMM):")
                print(f"    ~bytes/call={approx_bytes/1e6:.1f} MB (rows={expert_shape_m} x {hidden//2}B)")
                print(f"    {'ncb':>4} {'time_us':>9} {'GB/s':>8} {'vs_prev':>8}")
                prev = None
                for ncb in ncb_bw_list:
                    ms = results_bw[ncb]
                    gbps = approx_bytes / (ms * 1e-3) / 1e9
                    speedup = f"{prev/ms:.2f}x" if prev else "—"
                    print(f"    {ncb:>4} {ms*1e3:>9.1f} {gbps:>8.1f} {speedup:>8}")
                    prev = ms

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
                expected_m=expert_expected_m,
                local_fp4_buf=bc_fp4_perf, local_sfa_buf=bc_sfa_perf, copy_ready_flags=bc_flags_perf,
                num_copy_blocks=best_ncb, k_tiles_per_flag=K_TILES_PER_FLAG, rank_idx=rank,)
        torch.cuda.synchronize()

        local_kernel_times = []
        for _ in range(pipe_iters):
            ev_s = torch.cuda.Event(enable_timing=True)
            ev_e = torch.cuda.Event(enable_timing=True)
            ev_s.record()
            fused_dispatch_block_copy_gemm1_fp4(
                (W_fp4, W_scale_u16), out_bc, gl_e, ra_local, rs_e, sm_e, rc_e,
                expert_shape_m, max_tokens, world_size,
                expected_m=expert_expected_m,
                local_fp4_buf=bc_fp4_perf, local_sfa_buf=bc_sfa_perf, copy_ready_flags=bc_flags_perf,
                num_copy_blocks=best_ncb, k_tiles_per_flag=K_TILES_PER_FLAG, rank_idx=rank,)
            ev_e.record()
            local_kernel_times.append((ev_s, ev_e))
        torch.cuda.synchronize()
        local_ms_list = sorted(distributed_max_samples(
            [s.elapsed_time(e) for s, e in local_kernel_times], device))
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
        print(f"Performance Summary (max-rank, M={expert_shape_m}, N={N}, K={hidden}, {world_size} GPUs)")
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

    metadata_size = ((num_total_experts * 4 + 15) // 16) * 16

    sym_buf[:metadata_size].zero_()
    _mxfp4_quantize_to_sym_buffer(x, topk_ids, sym_buf,
        num_local_experts=num_total_experts, num_total_experts=num_total_experts,
        max_tokens_per_expert=max_tokens)
    torch.cuda.synchronize()
    dist.barrier()

    ws_expert = create_expert_preprocess_workspace(
        num_local_experts, world_size, max_tokens, device)
    shape_m, expert_expected_m, _ = dispatch_expert_prepare(
        sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
        max_tokens, hidden, local_expert_start, _workspace=ws_expert)
    block_m = get_gemm_block_m(
        shape_m, expert_expected_m, num_local_experts, N, hidden // 2,
        padded_m=max_tokens)
    gl_e, ra_e, rs_e, sm_e, rc_e, total_mb, finalized_shape_m = dispatch_expert_finalize(
        sym_buf_addrs, rank, world_size, num_local_experts, num_total_experts,
        max_tokens, hidden, local_expert_start, block_m,
        sync=True, _workspace=ws_expert)
    assert finalized_shape_m == shape_m

    # SFA produced on-device by the kernel (P0) — no host merged SFA needed.
    bc_fp4, bc_sfa, bc_flags = create_block_copy_buffers(
        num_local_experts, world_size, max_tokens, hidden, block_m, device)
    out_bc = create_fused_output(
        shape_m, num_local_experts, max_tokens, N, device)

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
            expected_m=expert_expected_m,
            local_fp4_buf=bc_fp4, local_sfa_buf=bc_sfa, copy_ready_flags=bc_flags,
            num_copy_blocks=NUM_COPY_BLOCKS, k_tiles_per_flag=K_TILES_PER_FLAG, rank_idx=rank,)
    torch.cuda.synchronize()

    NUM_ITERS = 20
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(NUM_ITERS):
        fused_dispatch_block_copy_gemm1_fp4(
            (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
            shape_m, max_tokens, world_size,
            expected_m=expert_expected_m,
            local_fp4_buf=bc_fp4, local_sfa_buf=bc_sfa, copy_ready_flags=bc_flags,
            num_copy_blocks=NUM_COPY_BLOCKS, k_tiles_per_flag=K_TILES_PER_FLAG, rank_idx=rank,)
    end_evt.record()
    torch.cuda.synchronize()
    kernel_ms = start_evt.elapsed_time(end_evt) / NUM_ITERS

    profile_buf.zero_()
    fused_dispatch_block_copy_gemm1_fp4(
        (W_fp4, W_scale_u16), out_bc, gl_e, ra_e, rs_e, sm_e, rc_e,
        shape_m, max_tokens, world_size,
        expected_m=expert_expected_m,
        local_fp4_buf=bc_fp4, local_sfa_buf=bc_sfa, copy_ready_flags=bc_flags,
        num_copy_blocks=NUM_COPY_BLOCKS, k_tiles_per_flag=K_TILES_PER_FLAG,
        kstripe_profile_buf=profile_buf, rank_idx=rank)
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
        print(f"  Fused GEMM grouping: {FUSED_GEMM_GROUPING}")
        if FULL_CORRECTNESS:
            print(f"  FULL_CORRECTNESS=1 (per-expert CPU reference check)")

    assert world_size >= 2, f"Requires at least 2 GPUs, got {world_size}"

    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)

    results = {}
    try:
        results['test0_api_contract'] = test_public_api_contract()
    except Exception as e:
        if rank == 0:
            print(f"Test 0 API contract error: {e}")
            import traceback; traceback.print_exc()
        results['test0_api_contract'] = False
    if rank == 0 and results['test0_api_contract']:
        print("Test 0: public API generation/parity contract PASSED")

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

    if not SKIP_PERFORMANCE:
        try:
            results['test2_performance'] = test_performance(rank, world_size, group, device)
        except Exception as e:
            if rank == 0:
                print(f"Test 2 error: {e}")
                import traceback; traceback.print_exc()
            results['test2_performance'] = False
    elif rank == 0:
        print("Skipping Test 2 (performance) — SKIP_PERFORMANCE=1")

    dist.barrier()

    # Test 3 (test_kstripe_profile) used the old per-mb/per-stripe buffer layout,
    # which is superseded by the per-tile/per-wave profiling now embedded in Test 2
    # (enabled with KSTRIPE_PROFILE=1). Opt in explicitly with LEGACY_TEST3=1.
    if os.getenv('KSTRIPE_PROFILE', '0') == '1' and os.getenv('LEGACY_TEST3', '0') == '1':
        try:
            results['test3_profile'] = test_kstripe_profile(rank, world_size, group, device)
        except Exception as e:
            if rank == 0:
                print(f"Test 3 error: {e}")
                import traceback; traceback.print_exc()
            results['test3_profile'] = False
        dist.barrier()
    elif rank == 0 and os.getenv('KSTRIPE_PROFILE', '0') != '1':
        print("Skipping profiling — set KSTRIPE_PROFILE=1 to enable per-wave profiling in Test 2")

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
