"""Distributed GEMM+RS correctness and optional matched PCCL benchmarking.

Run with torchrun --standalone --nproc-per-node=2 (or 1, 4, 8). All validation,
Graph replay, state reuse and timing for BF16 and both FP8 quantizations live here.
"""

import argparse
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm import get_workspace_for_gemm_reduce_scatter, fp8_gemm_nt_reduce_scatter


def get_packet_rows(local_m):
    return 4 if local_m < 8 else (8 if local_m < 16 else 16)


def packet_map(m, n, world, tile_n=0, tile_m=0, compact_min_m=0):
    """Recover logical rows/columns from the PPU0015 MMA packet layout."""
    lm = m // world
    rows = get_packet_rows(lm)
    aligned = lm % rows == 0 or lm < 3
    row = torch.arange(lm, device='cuda')[:, None]
    if world > 2 and not aligned and m <= tile_m and m >= compact_min_m:
        assert tile_n > 0, 'compact source check requires selected BN'
        col = torch.arange(n, device='cuda')[None, :]
        base = col // tile_n * tile_n
        width = torch.minimum(torch.full_like(base, tile_n), n - base)
        tile_elements = (lm * tile_n + 63) // 64 * 64
        return (base // tile_n * tile_elements + row * width + col % tile_n).reshape(-1)
    if not aligned:
        row = row + dist.get_rank() * lm % rows
    col = torch.arange(n, device='cuda')[None, :]
    atom = (row // rows) * (n // 16) + col // 16
    lane = (row % (4 if rows == 4 else 8)) * 4 + (col % 8) // 2
    if rows == 4:
        lane = lane + (col % 16) // 8 * 16
        word = 0
    elif rows == 8:
        word = (col % 16) // 8
    else:
        word = (row % 16) // 8 + (col % 16) // 8 * 2
    return (atom * rows * 16 + word * 64 + lane * 2 + col % 2).reshape(-1)


def bf16_packet_map(m, n, k, world, config):
    tile, wk = config
    _, bm, bn, bk, wm, _, stages, _ = tile
    compact_min = 16 if (world > 4 and m < wm and get_packet_rows(m // world) > 4
                         and wk == bk and k // bk >= 3 * stages) else wm
    return packet_map(m, n, world, bn, bm, compact_min)


def source_exact(buffer, partial, index, *, include_own=True, sources=None):
    """Check canonical rounded BF16 source packets in the active workspace phase."""
    if buffer.world == 1:
        return None  # TP1 runs the native GEMM without packet staging.
    m, n = partial.shape
    if sources is None:
        sources = [torch.empty_like(partial) for _ in range(buffer.world)]
        dist.all_gather(sources, partial)
    expected = torch.stack([p.chunk(buffer.world)[buffer.rank] for p in sources])
    phase = ((int(buffer._completed[0].item()) >> 32) - 1) & 1
    lm = m // buffer.world
    rows = get_packet_rows(lm)
    packet_rows = max(lm, 4) if lm % rows == 0 or lm < 3 else (lm + 2 * rows - 2) // rows * rows
    actual = buffer._slots[phase, :buffer.world * packet_rows * n].reshape(buffer.world, -1)[:, index]
    actual = actual.reshape_as(expected)
    if not include_own:
        peers = [source for source in range(buffer.world) if source != buffer.rank]
        actual, expected = actual[peers], expected[peers]
    return bool(torch.equal(actual, expected))



def source_sum_reference(partial, *, sources=None):
    """Independent FP64 reference from native rounded BF16 source partials."""
    world, rank = dist.get_world_size(), dist.get_rank()
    if sources is None:
        sources = [partial] if world == 1 else [torch.empty_like(partial) for _ in range(world)]
        if world > 1:
            dist.all_gather(sources, partial)
    values = torch.stack([p.chunk(world)[rank] for p in sources]).double()
    total, absolute = values.sum(0), values.abs().sum(0)
    additions = world - 1
    gamma32 = additions * 2.0**-24 / (1 - additions * 2.0**-24)
    gamma64 = additions * 2.0**-53 / (1 - additions * 2.0**-53)
    absolute_upper = absolute / (1 - gamma64)
    # The finite FP32 forward-error theorem excludes intermediate overflow.
    if not bool(torch.isfinite(absolute_upper).all()) or not bool(
        (absolute_upper * (1 + gamma32) <= torch.finfo(torch.float32).max).all()
    ):
        raise RuntimeError('source-sum fixture needs separate nonfinite/FP32-overflow validation')
    sum_error = (gamma32 + gamma64) * absolute_upper
    # Any FP32 summation order, followed by one RN BF16 conversion. The tiny
    # absolute term covers the final BF16 subnormal rounding interval.
    bound = sum_error + 2.0**-8 * (total.abs() + sum_error) + 2.0**-134
    return total, bound, sum_error


def source_sum_matches(result, reference):
    """Return an order-independent elementwise bound check and its worst ratio."""
    total, bound, sum_error = reference
    actual = result.double()
    finite = torch.isfinite(actual)
    # BF16 may legitimately overflow even when the FP32 source sum cannot.
    threshold = 2.0**128 - 2.0**119
    allowed_inf = (torch.isposinf(actual) & (total + sum_error >= threshold)) | (
        torch.isneginf(actual) & (total - sum_error <= -threshold)
    )
    error = (actual - total).abs()
    accepted = (finite & (error <= bound)) | allowed_inf
    ratio = torch.where(allowed_inf, torch.zeros_like(error), error / bound)
    return bool(accepted.all()), float(ratio.max())


def bf16_source_reference(a, b, out, selected):
    """Reuse native MMA, including the recorded K-warp accumulation order."""
    tile, warp_k = selected
    block_k = tile[3]
    if warp_k == block_k:
        deep_gemm.bf16_gemm_nt(a, b, out, configs=tile)
        return
    groups = (a.shape[1] + block_k - 1) // block_k
    warps = block_k // warp_k
    operands = [torch.nn.functional.pad(x, (0, groups * block_k - x.shape[1]))
                .view(x.shape[0], groups, warps, warp_k) for x in (a, b)]
    accumulator, partial = (torch.empty_like(out, dtype=torch.float32) for _ in range(2))
    config = (*tile[:3], warp_k, *tile[4:7], (0, 0, 0))
    for warp in range(warps):
        lhs, rhs = (x[:, :, warp, :].contiguous().view(x.shape[0], groups * warp_k) for x in operands)
        deep_gemm.bf16_gemm_nt(lhs, rhs, partial, configs=config)
        # Match WarpKReduce: warp zero adds the other FP32 partials in order.
        if warp == 0:
            accumulator.copy_(partial)
        else:
            accumulator.add_(partial)
    out.copy_(accumulator)


def check_partial_rounding(lhs, rhs, partial, output, buffer, fused, tile_n, tile_m, warp_m, *, index=None):
    """Cancellation distinguishes rounded own BF16 from an FP32 shortcut."""
    bf16 = isinstance(lhs, torch.Tensor)
    x, sx = (lhs, None) if bf16 else lhs
    w, sw = (rhs, None) if bf16 else rhs
    native = deep_gemm.bf16_gemm_nt if bf16 else deep_gemm.fp8_gemm_nt
    m, n = partial.shape
    if index is None:
        index = packet_map(m, n, buffer.world, tile_n, tile_m, warp_m)
    lm, rank, world = m // buffer.world, buffer.rank, buffer.world
    own = slice(rank * lm, (rank + 1) * lm)
    peer_owner = (rank - 1) % world
    peer = slice(peer_owner * lm, (peer_owner + 1) * lm)
    tags = []
    for sign in (1, -1):
        x.zero_()
        w.zero_()
        if not bf16:
            sx.fill_(1)
            sw.fill_(1)
        x[own, 0].fill_(sign)
        x[own, 1].fill_(sign / 16)
        x[peer, 0].fill_(-sign)
        w[:, 0].fill_(1)
        w[:, 1].fill_(1 / 32)
        native(lhs, rhs, partial)
        expected = torch.zeros_like(partial)
        expected[own].fill_(sign)
        expected[peer].fill_(-sign)
        if not torch.equal(partial, expected):
            raise RuntimeError('cancellation fixture did not produce native rounded partials')
        buffer._slots.fill_(float('nan'))
        output.fill_(123)
        torch.cuda.synchronize()
        dist.barrier()
        fused()
        torch.cuda.synchronize()
        if (bool(output.count_nonzero())
                or source_exact(buffer, partial, index, include_own=True) is False):
            raise RuntimeError('fusion changed native BF16 partial rounding')
        tags.append('own-rounding-positive' if sign > 0 else 'own-rounding-negative')
    return tags


def measure(function, divisor, args):
    for _ in range(args.warmup):
        function()
    torch.cuda.synchronize()
    dist.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        function()
    end.record()
    end.synchronize()
    elapsed = torch.tensor([start.elapsed_time(end) * 1000 / args.iters / divisor], device='cuda')
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed.item())


def make_configs(tile):
    return (deep_gemm.get_num_sms(), *tile, (0, 0, 0)) if tile is not None else None


def check_config_guards(report):
    from deep_gemm import _C as cpp

    world = dist.get_world_size()
    queries = [getattr(cpp, f'_gemm_comm_fp8_{q}_gemm_rs_config') for q in ('blockwise', 'channelwise')]
    num_sms = deep_gemm.get_num_sms()
    try:
        deep_gemm.set_num_sms(1)
        for query in queries:
            if query(8, 7168, 24576 // world, world)[0] != 1:
                raise RuntimeError('GEMM + RS ignored set_num_sms')
    finally:
        deep_gemm.set_num_sms(num_sms)
    invalid = [(query, (2**20, 7168, 24576 // world, world), {}) for query in queries]
    invalid += [(query, (0, 7168, 768, world), {}) for query in queries]
    if world > 1:
        invalid += [(query, (world + 1, 7168, 768, world), {}) for query in queries]
    invalid += [(queries[0], (8, 7168, 512, world), {}),
                (queries[1], (8, 7168, 48, world), {}),
                (queries[1], (48 * world, 272, 32, world), dict(configs=make_configs((64, 64, 32, 32, 32, 2))))]
    invalid += [
        (queries[0], (8, 7168, 24576 // world, world), dict(configs=make_configs(tile)))
        for tile in ((16, 192, 128, 16, 16, 1), (16, 64, 128, 16, 64, 3),
                     (16, 96, 128, 16, 16, 2), (16, 160, 128, 16, 16, 2),
                     (16, 224, 128, 16, 16, 2), (32, 192, 128, 16, 16, 2),
                     (32, 192, 128, 16, 64, 2), (1024, 1024, 128, 256, 256, 3))
    ]
    bf16_query = cpp._gemm_comm_bf16_gemm_rs_config
    invalid += [(bf16_query, (16, 4096, 12288 // world, world), dict(configs=make_configs(tile)))
                for tile in ((16, 64, 0, 16, 16, 2), (16, 64, 128, 0, 16, 2),
                             (0, 64, 128, 16, 16, 2), (16, 64, 24, 16, 16, 2),
                             (16, 64, 128, 16, 16, 0), (1024, 1024, 512, 16, 16, 4))]
    invalid.append((bf16_query, (16, 4096, 48, world), {}))
    bf16_query(16, 4096, 12288 // world, world)
    for lm in (1, 2, 3, 12, 17, 24, 1023, 1025):
        for query in (*queries, bf16_query):
            query(lm * world, 144, 768, world)
    for query, shape, kwargs in invalid:
        try:
            query(*shape, **kwargs)
        except RuntimeError:
            pass
        else:
            raise RuntimeError('invalid GEMM + RS configuration was accepted')
    if world == 1:
        try:
            queries[0](8, 7168, 24576, world, configs=make_configs((16, 192, 128, 16, 16, 2)))
        except RuntimeError:
            pass
        else:
            raise RuntimeError('TP1 accepted BN192 without weight-scale refinement')
    blockwise_bns = (32, 64, 128, 256) if world == 1 else (32, 64, 128, 192, 256)
    for bn in blockwise_bns:
        wn = 48 if bn == 192 else 16
        queries[0](8, 7168, 24576 // world, world, configs=make_configs((32, bn, 128, 16, wn, 2)))
    queries[1](8, 7168, 24576 // world, world, configs=make_configs((16, 96, 128, 16, 16, 2)))
    report['config_guards'] = dict(sm_budget_queries=len(queries), rejected_configs=len(invalid) + (world == 1),
                                  accepted_configs=len(blockwise_bns) + 1)


def check_workspace_guards(report):
    from unittest.mock import patch

    from deep_gemm import GemmReduceScatterWorkspace

    world, rank = dist.get_world_size(), dist.get_rank()
    rejected = 0
    if world > 1:
        # Include distinct capacities that round to the same packet allocation.
        for rows, cols in ((32 + rank, 16), (32, 16 * (rank + 1)), (1 + rank, 16)):
            with patch.object(GemmReduceScatterWorkspace, '_allocate', side_effect=AssertionError('allocated before validation')):
                try:
                    get_workspace_for_gemm_reduce_scatter(dist.group.WORLD, rows, cols)
                except ValueError as error:
                    if 'same workspace capacities' not in str(error):
                        raise
                    rejected += 1
                else:
                    raise RuntimeError('different workspace capacities were accepted')

    # All ranks must reject an invalid peer, including numeric values equal to integers.
    invalid = 0
    for rows, cols in ((32 if rank == 0 else 32.0, 16), (1 if rank == 0 else True, 16)):
        if world == 1:
            rows = float(rows)
        with patch.object(GemmReduceScatterWorkspace, '_allocate', side_effect=AssertionError('allocated before validation')):
            try:
                get_workspace_for_gemm_reduce_scatter(dist.group.WORLD, rows, cols)
            except ValueError as error:
                if 'positive integers' not in str(error):
                    raise
                invalid += 1
            else:
                raise RuntimeError('invalid peer workspace capacities were accepted')

    # TP1 must not import symmetric memory; multi-rank failures need a useful error.
    import builtins
    original_import = builtins.__import__

    def without_symmetric_memory(name, *args, **kwargs):
        if name == 'torch.distributed._symmetric_memory':
            raise ImportError('symmetric memory unavailable for this test')
        return original_import(name, *args, **kwargs)

    with patch('builtins.__import__', side_effect=without_symmetric_memory):
        try:
            get_workspace_for_gemm_reduce_scatter(dist.group.WORLD, 32, 16)
        except RuntimeError as error:
            if world == 1 or 'torch.distributed._symmetric_memory' not in str(error) or not isinstance(error.__cause__, ImportError):
                raise
        else:
            if world > 1:
                raise RuntimeError('missing symmetric memory dependency was accepted')
    report['workspace_guards'] = dict(capacity_mismatches=rejected, invalid_capacities=invalid, symmetric_memory_dependency=True)
    dist.barrier()


def check_buffer_reuse(report):
    """Alternate queued/generic kernels, input types and graphs on one allocation."""
    world, rank = dist.get_world_size(), dist.get_rank()
    buffer = get_workspace_for_gemm_reduce_scatter(dist.group.WORLD, 4096 + world, 4096)
    table = buffer._state.clone()
    allocation = buffer.buffer.data_ptr()
    calls = []

    def launch(lhs, rhs, out, *, workspace):
        if isinstance(lhs, torch.Tensor):
            deep_gemm.bf16_gemm_nt_reduce_scatter(lhs, rhs, out, workspace=workspace)
        else:
            fp8_gemm_nt_reduce_scatter(lhs, rhs, out, workspace=workspace)

    def native(lhs, rhs, partial):
        if isinstance(lhs, torch.Tensor):
            deep_gemm.bf16_gemm_nt(lhs, rhs, partial)
        else:
            deep_gemm.fp8_gemm_nt(lhs, rhs, partial)

    def fresh_reference(lhs, rhs, partial, zero=False):
        data = lhs if isinstance(lhs, torch.Tensor) else lhs[0]
        if zero:
            data.zero_()
        else:
            data.copy_(torch.randn(data.shape, device='cuda').to(data.dtype))
        native(lhs, rhs, partial)
        return source_sum_reference(partial)

    specs = [
        ('blockwise', 32 * world, 272, 768, 1),
        ('channelwise', world, 16, 64, 1),
        ('blockwise', 16 * world, 144, 640, 1),
        ('channelwise', 8 * world, 80, 96, 1),
        ('bf16', 32 * world, 272, 768, 1),
        ('bf16', 8 * world, 80, 96, 1),
        ('bf16', 16 * world, 144, 640, 1),
        ('channelwise', 4096, 4096, 12288 // world, 3),
        ('bf16', 4096, 4096, 12288 // world, 2),
        ('blockwise', 1024, 4096, 12288 // world, 16),
        ('channelwise', 3 * world, 48, 64, 1),
        ('blockwise', 5 * world, 144, 768, 1),
        ('bf16', 7 * world, 80, 256, 1),
    ]
    queued = 0
    for quantization, m, n, k, repeats in specs:
        generator = torch.Generator(device='cuda').manual_seed(233 + m + rank)
        dtype = torch.bfloat16 if quantization == 'bf16' else torch.float8_e4m3fn
        x = torch.randn((m, k), generator=generator, device='cuda').to(dtype)
        w = torch.randn((n, k), generator=generator, device='cuda').to(dtype)
        if quantization == 'bf16':
            lhs, rhs = x, w
        else:
            channelwise = quantization == 'channelwise'
            a_shape = (m, 1) if channelwise else (m, k // 128)
            b_shape = (n, 1) if channelwise else ((n + 127) // 128, k // 128)
            sx = torch.rand(a_shape, generator=generator, device='cuda') * 0.09 + 0.01
            sw = torch.rand(b_shape, generator=generator, device='cuda') * 0.09 + 0.01
            lhs, rhs = (x, sx), (w, sw)
        partial = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        out = torch.empty((m // world, n), dtype=partial.dtype, device='cuda')
        inline = False
        if quantization != 'bf16':
            config = buffer._get_config(m, n, k, quantization == 'channelwise', None)
        if world > 1 and quantization != 'bf16':
            entry = getattr(deep_gemm._C, f'_gemm_comm_fp8_{quantization}_gemm_rs')
            resources = entry(lhs, rhs, out, buffer._state, m, rank, world, config, False)
            inline = resources[3] > config[7][0]
            report.setdefault('kernel_local_bytes', []).append(dict(quantization=quantization, m=m, n=n, local_bytes=resources[1]))
            queued += inline
        native(lhs, rhs, partial)
        launch(lhs, rhs, out, workspace=buffer)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                launch(lhs, rhs, out, workspace=buffer)
        bf_config = deep_gemm._C._gemm_comm_bf16_gemm_rs_config(m, n, k, world, tile_policy=buffer.tile_policy) if quantization == 'bf16' else None
        selected = config if bf_config is None else bf_config[0]
        tile_m, tile_n = selected[1:3]
        warp_m = selected[4]
        index = bf16_packet_map(m, n, k, world, bf_config) if bf_config is not None else packet_map(m, n, world, tile_n, tile_m, warp_m)
        calls.append((lhs, rhs, out, graph, index, partial, repeats, inline))
        if quantization != 'bf16':
            for tile_policy in ('native', 'rs_aware'):
                buffer.tile_policy = tile_policy
                actual = buffer._get_config(m, n, k, quantization == 'channelwise', None)
                query = getattr(deep_gemm._C, f'_gemm_comm_fp8_{quantization}_gemm_rs_config')
                if actual != query(m, n, k, world, tile_policy=tile_policy):
                    raise RuntimeError('buffer reused a config from a different tile_policy or shape')
        buffer.tile_policy = 'rs_aware'
    if world in (2, 4) and queued == 0:
        raise RuntimeError('mixed workspace fixture did not exercise a queued kernel')
    for tile_policy in ('rs_aware', 'native', 'rs_aware'):
        buffer.tile_policy = tile_policy
        actual = buffer._get_config(8, 7168, 24576 // world, False, None)
        expected_config = deep_gemm._C._gemm_comm_fp8_blockwise_gemm_rs_config(
            8, 7168, 24576 // world, world, tile_policy=tile_policy
        )
        if actual != expected_config:
            raise RuntimeError('changing the buffer tile_policy returned a stale config')
    buffer.tile_policy = 'rs_aware'
    for repeat in range(5):
        for lhs, rhs, out, graph, index, partial, repeats, inline in calls[:: (-1 if repeat % 2 else 1)]:
            expected = fresh_reference(lhs, rhs, partial, zero=repeat == 1)
            buffer._slots.fill_(float('nan'))
            # Complete every rank's poison before incoming stores can begin.
            torch.cuda.synchronize()
            dist.barrier()
            before = int(buffer._completed[0]) >> 32
            target = torch.empty_like(out) if repeat == 4 else out
            if repeat % 2:
                graph.replay()
            else:
                launch(lhs, rhs, target, workspace=buffer)
            torch.cuda.synchronize()
            count = repeats if repeat % 2 else 1
            after = int(buffer._completed[0]) >> 32
            if world > 1 and after != before + count:
                raise RuntimeError('mixed captured calls corrupted completion epochs')
            packets_equal = source_exact(buffer, partial, index, include_own=True)
            sum_valid, sum_ratio = source_sum_matches(target, expected)
            if not sum_valid or packets_equal is False:
                detail = dict(
                    rank=rank,
                    round=repeat,
                    shape=tuple(partial.shape),
                    k=lhs.shape[1] if isinstance(lhs, torch.Tensor) else lhs[0].shape[1],
                    bf16=isinstance(lhs, torch.Tensor),
                    queued=inline,
                    packets_equal=packets_equal,
                    source_sum_error_ratio=sum_ratio,
                    nan=int(target.isnan().sum()),
                    max_abs=float((target.double() - expected[0]).abs().max()),
                    before=before,
                    after=after,
                )
                print('MIXED_REUSE_DIAGNOSTIC', detail, flush=True)
                raise RuntimeError(f'changing buffer shape or captured calls corrupted output/packets: {detail}')
    for direction in (1, -1):
        sequence = calls[::direction]
        mixed = torch.cuda.CUDAGraph()
        with torch.cuda.graph(mixed):
            for lhs, rhs, out, *_ in sequence:
                launch(lhs, rhs, out, workspace=buffer)
        expected = [fresh_reference(lhs, rhs, partial) for lhs, rhs, _, _, _, partial, _, _ in sequence]
        buffer._slots.fill_(float('nan'))
        # Complete every rank's poison before incoming stores can begin.
        torch.cuda.synchronize()
        dist.barrier()
        before = int(buffer._completed[0]) >> 32
        mixed.replay()
        torch.cuda.synchronize()
        if world > 1 and int(buffer._completed[0]) >> 32 != before + len(sequence):
            raise RuntimeError('mixed graph corrupted completion epochs')
        if not all(source_sum_matches(call[2], value)[0] for call, value in zip(sequence, expected)):
            raise RuntimeError('mixed graph exceeded the native-source reduction error bound')
        _, _, _, _, index, partial, _, inline = sequence[-1]
        if source_exact(buffer, partial, index, include_own=True) is False:
            raise RuntimeError('mixed graph corrupted final source packets')
        mixed.reset()
    if buffer.buffer.data_ptr() != allocation or not torch.equal(table, buffer._state):
        raise RuntimeError('changing shapes mutated buffer allocation or pointer metadata')
    for *_, graph, index, partial, repeats, inline in calls:
        graph.reset()
    lhs, rhs, out, *_ = calls[0]
    for too_small in ((world, 272), (4096 + world, 16)):
        small = get_workspace_for_gemm_reduce_scatter(dist.group.WORLD, *too_small)
        try:
            fp8_gemm_nt_reduce_scatter(lhs, rhs, out, workspace=small)
        except ValueError:
            pass
        else:
            raise RuntimeError('undersized communication buffer was accepted')
    report['buffer_reuse_checks'] = 7 * len(calls)
    report['buffer_reuse_mixed_graph_calls'] = len(calls)
    report['buffer_reuse_flag_capacity'] = buffer._chunk_capacity
    report['buffer_reuse_queued_cases'] = queued
    report['buffer_reuse_graph_repeats'] = sorted({entry[-2] for entry in calls})
    torch.cuda.synchronize()
    dist.barrier()

def check_public_configs(report):
    """Reject malformed configs before the TP1 native or multi-rank launch."""
    world = dist.get_world_size()
    from checkin.run_checkin import gemm_rs_config_check
    m, n, k, raw_config = gemm_rs_config_check(world)
    buffer = get_workspace_for_gemm_reduce_scatter(dist.group.WORLD, m, n)
    x = torch.randn((m, k), device='cuda').to(torch.float8_e4m3fn)
    w = torch.randn((n, k), device='cuda').to(torch.float8_e4m3fn)
    out = torch.empty((m // world, n), dtype=torch.bfloat16, device='cuda')
    rejected = 0
    for channelwise in (False, True):
        sa_shape = (m, 1) if channelwise else (m, k // 128)
        sb_shape = (n, 1) if channelwise else ((n + 127) // 128, k // 128)
        sa = torch.rand(sa_shape, device='cuda') * 0.09 + 0.01
        sb = torch.rand(sb_shape, device='cuda') * 0.09 + 0.01
        lhs, rhs = (x, sa), (w, sb)
        valid = (*raw_config, (0, 0, 0))
        invalid = [valid[:-1], (*valid[:7], (0, 0)),
                   (*valid[:4], 0, *valid[5:]),
                   (0, *valid[1:]), (deep_gemm.get_num_sms() + 1, *valid[1:]),
                   (*valid[:6], 0, valid[7])]
        if world == 1 and not channelwise:
            invalid.append((1, 16, 192, 128, 16, 16, 2, (0, 0, 0)))
        for config in invalid:
            out.fill_(float('nan'))
            try:
                fp8_gemm_nt_reduce_scatter(lhs, rhs, out, workspace=buffer, configs=config)
            except (ValueError, RuntimeError):
                rejected += 1
            else:
                raise RuntimeError('malformed public configs reached GEMM')
            if not bool(torch.isnan(out).all()):
                raise RuntimeError('rejected configs modified the caller output')
        if world > 1:
            misaligned = torch.empty(out.numel() + 1, dtype=out.dtype, device=out.device)[1:].view_as(out)
            try:
                fp8_gemm_nt_reduce_scatter(lhs, rhs, misaligned, workspace=buffer)
            except ValueError as error:
                if '16-byte aligned' not in str(error):
                    raise
                rejected += 1
            else:
                raise RuntimeError('misaligned public output was accepted')
            config = buffer._get_config(m, n, k, channelwise, valid)
            entry = getattr(deep_gemm._C, '_gemm_comm_fp8_' +
                            ('channelwise' if channelwise else 'blockwise') + '_gemm_rs')
            try:
                entry(lhs, rhs, misaligned, buffer._state, m, buffer.rank, world, config, False)
            except RuntimeError as error:
                if '16-byte aligned' not in str(error):
                    raise
                rejected += 1
            else:
                raise RuntimeError('misaligned private output was accepted')
        # A valid explicit tile still computes correctly after rejection paths.
        fp8_gemm_nt_reduce_scatter(lhs, rhs, out, workspace=buffer, configs=valid)
        a = x.float() * (sa if channelwise else sa.repeat_interleave(128, dim=1))
        b = w.float() * (sb if channelwise else sb.repeat_interleave(128, dim=0)[:n].repeat_interleave(128, dim=1))
        reference = a @ b.T
        expected = torch.empty_like(out, dtype=torch.float32)
        dist.reduce_scatter_tensor(expected, reference)
        error = float((out.float() - expected).norm() / expected.norm().clamp_min(1e-30))
        if not bool(torch.isfinite(out).all()) or error >= 0.015:
            raise RuntimeError('valid public configs changed quantization semantics')
    a, b = x.bfloat16(), w.bfloat16()
    for config in invalid[:6]:
        out.fill_(float('nan'))
        try:
            deep_gemm.bf16_gemm_nt_reduce_scatter(a, b, out, workspace=buffer, configs=config)
        except (ValueError, RuntimeError):
            rejected += 1
        else:
            raise RuntimeError('malformed BF16 public configs reached GEMM')
        if not bool(torch.isnan(out).all()):
            raise RuntimeError('rejected BF16 configs modified the caller output')
    if world > 1:
        try:
            deep_gemm.bf16_gemm_nt_reduce_scatter(a, b, misaligned, workspace=buffer)
        except ValueError as error:
            if '16-byte aligned' not in str(error):
                raise
            rejected += 1
        else:
            raise RuntimeError('misaligned BF16 output was accepted')
        config = deep_gemm._C._gemm_comm_bf16_gemm_rs_config(m, n, k, world, configs=valid)
        try:
            deep_gemm._C._gemm_comm_bf16_gemm_rs(
                a, b, misaligned, buffer._state, m, buffer.rank, world, config, False
            )
        except RuntimeError as error:
            if '16-byte aligned' not in str(error):
                raise
            rejected += 1
        else:
            raise RuntimeError('misaligned BF16 private output was accepted')
    deep_gemm.bf16_gemm_nt_reduce_scatter(a, b, out, workspace=buffer, configs=valid)
    report['public_config_checks'] = dict(rejected=rejected, valid=3)
    dist.barrier()


def run_case(quantization, m, n, k_global, args, report, tile=None):
    operation = fp8_gemm_nt_reduce_scatter

    rank, world = dist.get_rank(), dist.get_world_size()
    if rank == 0:
        print('RUN', quantization, 'TP', world, 'shape', (m, n, k_global // world), flush=True)
    k, lm = k_global // world, m // world
    x = torch.empty((m, k), dtype=torch.float8_e4m3fn, device='cuda')
    w = torch.empty((n, k), dtype=torch.float8_e4m3fn, device='cuda')
    if quantization == 'blockwise':
        sx = (
            torch.empty((m, k // 128), device='cuda') if m == 32
            else torch.empty((k // 128, m), device='cuda').T
        )
        sw = torch.empty(((n + 127) // 128, k // 128), device='cuda')
    else:
        sx = torch.empty((m, 1), device='cuda')
        sw = torch.empty((n, 1), device='cuda')
    native = deep_gemm.fp8_gemm_nt
    configs = make_configs(args.tile if tile is None else tile)
    buffer = get_workspace_for_gemm_reduce_scatter(dist.group.WORLD, m, n, tile_policy=args.tile_policy)
    rs_config = buffer._get_config(m, n, k, quantization == 'channelwise', configs)
    buffers = {'fused': buffer}
    lhs, rhs = (x, sx), (w, sw)
    partial = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    output = torch.empty((lm, n), dtype=torch.bfloat16, device='cuda')
    fused_output = torch.empty_like(output)
    index = packet_map(m, n, world, rs_config[2], rs_config[1], rs_config[4])

    def fresh(case):
        generator = torch.Generator(device='cuda').manual_seed(17003 * case + rank + 13)
        x.copy_(torch.randn(x.shape, generator=generator, device='cuda').to(x.dtype))
        w.copy_(torch.randn(w.shape, generator=generator, device='cuda').to(w.dtype))
        sx.copy_(torch.rand(sx.shape, generator=generator, device='cuda') * 0.09 + 0.01)
        sw.copy_(torch.rand(sw.shape, generator=generator, device='cuda') * 0.09 + 0.01)
        if case == 1:
            x.zero_()

    def dense():
        native(lhs, rhs, partial)
        return partial

    def collective():
        dist.reduce_scatter_tensor(output, partial)
        return output

    def baseline():
        dense()
        return collective()

    functions = {'baseline_rs': baseline, 'gemm': dense, 'pccl_rs': collective}
    results = {'baseline_rs': output, 'gemm': partial, 'pccl_rs': output}
    for name, buffer in buffers.items():
        def fused(buffer=buffer):
            operation(lhs, rhs, fused_output, workspace=buffer, configs=configs)
            return fused_output

        functions[name] = fused
        results[name] = fused_output
    staged_send = False
    if world > 1:
        resource_query = getattr(deep_gemm._C, f'_gemm_comm_fp8_{quantization}_gemm_rs')
        resources = resource_query(lhs, rhs, fused_output, buffer._state, m, rank, world, rs_config, False)
        staged_send = resources[3] > rs_config[7][0]
        report.setdefault('kernel_local_bytes', []).append(dict(quantization=quantization, m=m, n=n, local_bytes=resources[1]))
    if world > 1:
        tags = check_partial_rounding(lhs, rhs, partial, fused_output, buffer, functions['fused'], rs_config[2], rs_config[1], rs_config[4])
        report['partial_rounding_checks'] = report.get('partial_rounding_checks', 0) + len(tags)
    fresh(0)
    baseline()
    native_source_reference = source_sum_reference(partial)
    for name, function in functions.items():
        for _ in range(3):
            function()
        torch.cuda.synchronize()
        dist.barrier()
    # Distinct outputs share only communication state, including during capture.
    # Explicit configs override the tile_policy and their placeholder SMEM is recomputed.
    preserved = fused_output.clone()
    alternate = torch.empty_like(fused_output)
    table = buffer._state.clone()
    explicit = (*rs_config[:7], (0, 0, 0))
    buffer._slots.fill_(float('nan'))
    torch.cuda.synchronize()
    dist.barrier()
    if args.skew and rank == world - 1:
        time.sleep(0.2)
    operation(lhs, rhs, alternate, workspace=buffer, configs=explicit)
    torch.cuda.synchronize()
    if not torch.equal(preserved, fused_output) or not source_sum_matches(alternate, native_source_reference)[0]:
        raise RuntimeError(f'explicit configs or an alternate output changed prior results: '
                           f'rank={rank}, config={rs_config}, '
                           f'prior_changed={int((preserved != fused_output).sum())}, '
                           f'alternate_changed={int((alternate != preserved).sum())}, '
                           f'max_difference={float((alternate.float() - preserved.float()).abs().max())}')
    alternate_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(alternate_graph):
        operation(lhs, rhs, alternate, workspace=buffer, configs=configs)
    alternate.fill_(float('nan'))
    alternate_graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(preserved, fused_output) or not source_sum_matches(alternate, native_source_reference)[0]:
        raise RuntimeError('Graph replay wrote a different call output')
    # Change the actual resident grid, then alternate captured configurations.
    # The communication epoch must count calls independently of CTA count.
    limited = (1, *rs_config[1:7], (0, 0, 0))
    operation(lhs, rhs, alternate, workspace=buffer, configs=limited)
    limited_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(limited_graph):
        operation(lhs, rhs, alternate, workspace=buffer, configs=limited)
    for _ in range(3):
        limited_graph.replay()
        alternate_graph.replay()
        functions['fused']()
    torch.cuda.synchronize()
    if not source_sum_matches(fused_output, native_source_reference)[0] or not source_sum_matches(alternate, native_source_reference)[0]:
        raise RuntimeError('changing configs or Graph grids corrupted output')
    if not torch.equal(table, buffer._state):
        raise RuntimeError('per-call output changed the communication pointer table')
    for invalid_out in (torch.empty_like(fused_output, dtype=torch.float32), fused_output.T):
        try:
            operation(lhs, rhs, invalid_out, workspace=buffer, configs=configs)
        except ValueError:
            pass
        else:
            raise RuntimeError('invalid caller output reached the fused kernel')
    del alternate_graph, limited_graph, alternate, preserved, table
    report['caller_output_checks'] = report.get('caller_output_checks', 0) + 6
    # Both capture lengths exercise phase preservation and phase alternation.
    graphs = {}
    for name, function in functions.items():
        capture_lengths = sorted({2, 3, args.capture_repeats}) if args.benchmark else (2, 3)
        for repeats in capture_lengths:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(repeats):
                    function()
            graphs[name, repeats] = graph
            torch.cuda.synchronize()
            dist.barrier()
    for case in range(1, args.cases + 1):
        fresh(case)
        dense()
        native_source_reference = source_sum_reference(partial)
        if quantization == 'blockwise':
            a32 = x.float() * sx.repeat_interleave(128, dim=1)
            b32 = w.float() * sw.repeat_interleave(128, dim=0)[:n].repeat_interleave(128, dim=1)
            reference = a32 @ b32.T
            del a32, b32
        else:
            reference = (x.float() @ w.float().T) * sx * sw.T
        ref_rs = torch.empty((lm, n), dtype=torch.float32, device='cuda')
        dist.reduce_scatter_tensor(ref_rs, reference)
        if case == args.cases:
            for buffer in buffers.values():
                state = [
                    fused_output,
                    buffer._slots,
                    buffer._flags,
                    buffer._completed,
                ]
                addresses = [tensor.data_ptr() for tensor in state]
                buffer.reset_state()
                current = [
                    fused_output,
                    buffer._slots,
                    buffer._flags,
                    buffer._completed,
                ]
                if addresses != [tensor.data_ptr() for tensor in current]:
                    raise RuntimeError('reset changed captured buffer addresses')
                if any(bool(tensor.count_nonzero().item()) for tensor in current[2:]):
                    raise RuntimeError('reset left nonzero epochs or completion state')
        for name, function in functions.items():
            if name in ('gemm', 'pccl_rs'):
                continue
            buffer = buffers.get(name)
            lengths = {2, 3}
            if args.benchmark:
                lengths.add(args.capture_repeats)
            for timing, repeats in [('eager', 1), *(('graph', length) for length in sorted(lengths))]:
                if buffer is not None:
                    buffer._slots.fill_(float('nan'))
                    fused_output.fill_(float('nan'))
                    torch.cuda.synchronize()
                    dist.barrier()
                if args.skew and rank == case % world:
                    torch.cuda._sleep(20000)
                if timing == 'eager':
                    actual = function()
                else:
                    graphs[name, repeats].replay()
                    actual = results[name]
                torch.cuda.synchronize()
                expected = ref_rs
                error = float((actual.float() - expected).norm() / expected.norm().clamp_min(1e-30))
                exact = source_exact(buffer, partial, index, include_own=True) if buffer is not None else None
                sum_valid, sum_ratio = source_sum_matches(actual, native_source_reference) if buffer is not None else (True, None)
                valid = bool(torch.isfinite(actual).all()) and error < 0.015 and exact is not False and sum_valid
                entry = dict(rank=rank, relative_l2=error, source_exact=exact, source_sum_error_ratio=sum_ratio, passed=valid)
                ranks = [None] * world
                dist.all_gather_object(ranks, entry)
                report['validation'].append(
                    dict(
                        quantization=quantization,
                        m=m,
                        n=n,
                        k=k,
                        method=name,
                        case=case,
                        timing=timing,
                        repeats=repeats,
                        per_rank=ranks,
                    )
                )
                if not all(item['passed'] for item in ranks):
                    raise RuntimeError(str(report['validation'][-1]))
        del reference, ref_rs
    # Reject a malformed scale tensor before launching either quantization.
    for buffer in buffers.values():
        wrong_shape = (m, 1) if quantization == 'blockwise' else (m, 2)
        wrong = torch.empty(wrong_shape, dtype=torch.float32, device='cuda')
        try:
            operation((x, wrong), rhs, fused_output, workspace=buffer, configs=configs)
        except ValueError:
            pass
        else:
            raise RuntimeError('wrong quantization scales reached the kernel')
    if args.benchmark:
        fresh(0)
        dense()
        # Rotate the method order each round to balance clock/temperature drift.
        # All ranks use the same order; each sample records the slowest rank.
        names = list(functions)
        for timing in ('eager', 'graph'):
            samples = {name: [] for name in names}
            divisor = args.capture_repeats if timing == 'graph' else 1
            for round_index in range(args.rounds):
                offset = round_index % len(names)
                order = names[offset:] + names[:offset]
                if round_index % 2:
                    order.reverse()
                for name in order:
                    fn = graphs[name, args.capture_repeats].replay if timing == 'graph' else functions[name]
                    samples[name].append(measure(fn, divisor, args))
            for name in names:
                buffer = buffers.get(name)
                report['timings'].append(
                    dict(
                        quantization=quantization,
                        m=m,
                        n=n,
                        k=k,
                        method=name,
                        timing=timing,
                        max_rank_us=statistics.median(samples[name]),
                        round_max_rank_us=samples[name],
                        config=rs_config if buffer else None,
                        output_shape=list(results[name].shape),
                    )
                )
    if rank == 0:
        print('PASS', quantization, 'TP', world, 'shape', (m, n, k), flush=True)
        Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    # Release every captured pointer/communicator use before destroying buffers.
    graph = None
    function = None
    functions.clear()
    graphs.clear()
    torch.cuda.synchronize()
    dist.barrier()
    buffers.clear()
    gc.collect()


def run_bf16_case(item, *, timed=True, warmup=20, iters=100, rounds=7, capture_repeats=16, cases=3):
    """Strict BF16 source-rounding, graph/state and paired PCCL validation."""
    import time

    rank, world = dist.get_rank(), dist.get_world_size()
    m, n, k = item['m'], item.get('n', 4096), item.get('k_global', 12288) // world
    tile_policy = item.get('tile_policy', 'rs_aware')
    ws = get_workspace_for_gemm_reduce_scatter(dist.group.WORLD, m, n, tile_policy=tile_policy)
    other = get_workspace_for_gemm_reduce_scatter(dist.group.WORLD, m, n, tile_policy=tile_policy)
    a = torch.empty((m, k), dtype=torch.bfloat16, device='cuda')
    b = torch.empty((n, k), dtype=torch.bfloat16, device='cuda')
    native = torch.empty((m, n), dtype=a.dtype, device='cuda')
    out = torch.empty((m // world, n), dtype=a.dtype, device='cuda')
    ref, float_ref, alternate, public_out = (
        torch.empty_like(out),
        torch.empty_like(out, dtype=torch.float32),
        torch.empty_like(out),
        torch.empty_like(out),
    )
    raw = item.get('configs')
    raw = (*raw[:7], tuple(raw[7])) if raw else None
    config = deep_gemm._C._gemm_comm_bf16_gemm_rs_config(m, n, k, world, configs=raw, tile_policy=tile_policy)
    tile, wk = config
    entry = deep_gemm._C._gemm_comm_bf16_gemm_rs
    resources = entry(a, b, out, ws._state, m, rank, world, config, False)
    # Nonzero local allocation is a measured resource, not an execution failure.
    print('BF16_RESOURCES', rank, item, config, resources, flush=True)

    def call(dst=out, cfg=config):
        entry(a, b, dst, ws._state, m, rank, world, cfg, True)

    def native_default():
        deep_gemm.bf16_gemm_nt(a, b, native)

    def native_matched():
        deep_gemm.bf16_gemm_nt(a, b, native, configs=tile)

    def pccl_rs():
        dist.reduce_scatter_tensor(ref, native)

    def unfused():
        native_default()
        pccl_rs()

    def public():
        deep_gemm.bf16_gemm_nt_reduce_scatter(a, b, public_out, workspace=other, configs=raw)

    def fresh(seed, zero=False):
        g = torch.Generator(device='cuda').manual_seed(seed + rank)
        a.copy_(torch.randn(a.shape, device='cuda', generator=g).bfloat16())
        b.copy_(torch.randn(b.shape, device='cuda', generator=g).bfloat16())
        if zero:
            a.zero_()
        bf16_source_reference(a, b, native, config)
        pccl_rs()
        independent = a.float() @ b.float().t()
        dist.all_reduce(independent)
        float_ref.copy_(independent.chunk(world)[rank])

    checks, index, table = [], bf16_packet_map(m, n, k, world, config), ws._state.clone()

    def verify(tag, buffer=ws, result=out):
        if world == 1 and buffer is other:
            native_default()  # The public TP1 API retains native GEMV dispatch.
        torch.cuda.synchronize()
        parts = [torch.empty_like(native) for _ in range(world)]
        dist.all_gather(parts, native)
        reduction_reference = source_sum_reference(native, sources=parts)
        exact = source_exact(buffer, native, index, sources=parts) if world > 1 else torch.equal(result, native)
        sum_valid, sum_ratio = source_sum_matches(result, reduction_reference)
        if not exact or not sum_valid:
            print(
                'BF16_DIAGNOSTIC',
                rank,
                tag,
                dict(
                    source_exact=exact,
                    source_sum_within_bound=sum_valid,
                    source_sum_error_ratio=sum_ratio,
                    nan=int(result.isnan().sum()),
                    completed=int(buffer._completed[0]),
                ),
                flush=True,
            )
        assert exact and sum_valid, ('BF16 source/reduction bound', rank, item, tag, exact, sum_ratio)
        error = (
            float((result.float() - float_ref).norm() / float_ref.norm())
            if bool(float_ref.any())
            else float(result.abs().max())
        )
        assert error < 0.015, ('BF16 float reference', item, tag, error)
        checks.append(dict(tag=tag, source_exact=True, source_sum_error_ratio=sum_ratio, relative_error=error))
        print('CHECK_PASS', rank, 'bf16', m, tag, flush=True)

    for trial in range(cases):
        fresh(19013 * (trial + 1), zero=trial == 2)
        ws.reset_state()
        ws._slots.fill_(float('nan'))
        out.fill_(123.0)
        dist.barrier()
        if trial == 1 and rank == world - 1:
            time.sleep(0.2)
        call()
        verify('fresh-' + str(trial))
    fresh(24031)
    call()
    verify('output-precheck')
    preserved = out.clone()
    call(alternate)
    torch.cuda.synchronize()
    assert torch.equal(out, preserved), 'alternate call changed an untouched output'
    assert source_sum_matches(alternate, source_sum_reference(native))[0]
    del preserved
    limited = ((1, *tile[1:]), wk)
    limited_resources = entry(a, b, out, ws._state, m, rank, world, limited, False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(cfg=limited)
    for trial in range(3):
        fresh(23003 * (trial + 1), zero=trial == 0)
        ws._slots.fill_(float('nan'))
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        verify('varying-grid-limited-' + str(trial))
        call()
        verify('varying-grid-normal-' + str(trial))
    graph.reset()
    torch.cuda.synchronize()
    assert torch.equal(table, ws._state)
    public()
    verify('public-precheck', other, public_out)
    methods = dict(
        candidate=call,
        default_fused=public,
        native_default=native_default,
        native_matched=native_matched,
        pccl_rs=pccl_rs,
        unfused=unfused,
    )
    graphs, values = {}, {name: [] for name in methods}
    if timed:
        for name, method in methods.items():
            for _ in range(warmup):
                method()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(capture_repeats):
                    method()
            graphs[name] = g
            torch.cuda.synchronize()
        epoch = int(ws._completed[0]) >> 32
        for round_index in range(rounds):
            order = list(methods)
            shift = round_index % len(order)
            order = order[shift:] + order[:shift]
            if round_index % 2:
                order.reverse()
            for name in order:
                dist.barrier()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iters):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                elapsed = torch.tensor(start.elapsed_time(end) * 1000 / (iters * capture_repeats), device='cuda')
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                values[name].append(float(elapsed))
                if name == 'candidate':
                    expected = epoch + (round_index + 1) * iters * capture_repeats
                    assert int(ws._completed[0]) >> 32 == expected
    validation = []
    for repeats in (2, 3, 16):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(repeats):
                call()
        validation.append(g)
    public_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(public_graph):
        for _ in range(16):
            public()
    for trial in range(3):
        fresh(26003 * (trial + 1), zero=trial == 0)
        dist.barrier()
        for repeats, g in zip((2, 3, 16), validation):
            g.replay()
            verify('graph-' + str(repeats) + '-trial-' + str(trial))
        public_graph.replay()
        verify('public-graph-trial-' + str(trial), other, public_out)
    assert torch.equal(table, ws._state) and len(checks) == 20 + cases
    all_checks = [None] * world
    dist.all_gather_object(all_checks, checks)
    for g in [*graphs.values(), *validation, public_graph]:
        g.reset()
    torch.cuda.synchronize()
    dist.barrier()
    result = dict(
        **item,
        config=config,
        resources=resources,
        local_bytes=resources[1],
        limited_grid_resources=limited_resources,
        source_reference='native_warp_k' if wk != tile[3] else 'native_matched',
        checks=all_checks,
        timings={name: dict(max_rank_us=statistics.median(v), rounds=v) for name, v in values.items() if v},
    )
    print(
        'BF16_PASS',
        rank,
        m,
        json.dumps({name: round(d['max_rank_us'], 3) for name, d in result['timings'].items()}),
        flush=True,
    )
    return result


def replay_inputs(item):
    m, n, k, world = (int(item[key]) for key in ('m', 'n', 'k', 'tp'))
    a = torch.empty((m, k), dtype=item['data_type'], device='cuda')
    b = torch.empty((n, k), dtype=a.dtype, device='cuda')
    out = torch.empty((m // world, n), dtype=torch.bfloat16, device='cuda')
    if a.dtype == torch.bfloat16:
        return a, b, out
    channelwise = item.get('quant_type', 'block') == 'channel'
    sa = torch.empty((m, 1 if channelwise else k // 128), device='cuda')
    sb = torch.empty((n, 1) if channelwise else ((n + 127) // 128, k // 128), device='cuda')
    return (a, sa), (b, sb), out


def replay_config(item, tile_policy="rs_aware"):
    m, n, k, world = (int(item[key]) for key in ('m', 'n', 'k', 'tp'))
    bf16 = item['data_type'] == torch.bfloat16
    quantization = 'channelwise' if item.get('quant_type') == 'channel' else 'blockwise'
    prefix = 'bf16' if bf16 else f'fp8_{quantization}'
    raw = item.get('configs')
    configs = (*raw, (0, 0, 0)) if raw is not None else None
    selected = getattr(deep_gemm._C, f'_gemm_comm_{prefix}_gemm_rs_config')(m, n, k, world, configs=configs, tile_policy=tile_policy)
    if bf16 and 'warp_k' in item:
        selected = (selected[0], int(item['warp_k']))
    return prefix, selected


def precompile_case(item):
    """Compile one rank without a process group, peer allocation or collective launch."""
    if not (os.environ.get('HGGC_WARM_UP') or deep_gemm.get_compile_mode() == 1):
        raise ValueError('GEMM/RS precompilation requires compile-only mode')
    lhs, rhs, out = replay_inputs(item)
    prefix, selected = replay_config(item)
    # State has six metadata words; launch=False never dereferences them.
    state = torch.empty(6, dtype=torch.int64, device='cuda')
    entry = getattr(deep_gemm._C, f'_gemm_comm_{prefix}_gemm_rs')
    entry(lhs, rhs, out, state, item['m'], item.get('rank', 0), item['tp'], selected, launch=False)
    if item.get('rank', 0) == 0:
        partial = torch.empty((item['m'], item['n']), dtype=out.dtype, device='cuda')
        native = deep_gemm.bf16_gemm_nt if prefix == 'bf16' else deep_gemm.fp8_gemm_nt
        native(lhs, rhs, partial)  # Cancellation uses native automatic tiles.
        if prefix == 'bf16':
            bf16_source_reference(lhs, rhs, partial, selected)
        else:
            deep_gemm.fp8_gemm_nt(lhs, rhs, partial, configs=selected if selected[2] != 192 else None)
    return dict(passed=True, compile_only=True)


def replay_case(item, workspace=None, *, check_accuracy=True):
    """Replay a host-shape dump; peer pointers and epochs are rebuilt collectively."""
    tp = int(item['tp'])
    if 'LOCAL_RANK' not in os.environ and (os.environ.get('HGGC_WARM_UP') or deep_gemm.get_compile_mode() == 1):
        return precompile_case(item)
    if tp != int(os.environ.get('WORLD_SIZE', '1')) or 'LOCAL_RANK' not in os.environ:
        raise ValueError(f'GEMM/RS replay requires torchrun --nproc-per-node={tp}')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group('nccl', timeout=timedelta(seconds=180))
    graph = None
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if tp != world:
            raise ValueError('dump TP must match the replay process group')
        m, n, k = (int(item[key]) for key in ('m', 'n', 'k'))
        bf16 = item['data_type'] == torch.bfloat16
        channelwise = item.get('quant_type', 'block') == 'channel'
        if item['data_type'] not in (torch.bfloat16, torch.float8_e4m3fn):
            raise ValueError('GEMM/RS replay supports BF16 and FP8 E4M3')
        ws = workspace or get_workspace_for_gemm_reduce_scatter(dist.group.WORLD, m, n)
        lhs, rhs, out = replay_inputs(item)
        a, b = (lhs, rhs) if bf16 else (lhs[0], rhs[0])
        a.zero_()
        b.zero_()
        if not bf16:
            sa, sb = lhs[1], rhs[1]
            sa.fill_(1)
            sb.fill_(1)
        partial = torch.empty((m, n), dtype=out.dtype, device='cuda')
        raw = item.get('configs')
        configs = (*raw, (0, 0, 0)) if raw is not None else None
        _, selected = replay_config(item, ws.tile_policy)
        if bf16:
            tile, warp_k = selected
            index = bf16_packet_map(m, n, k, world, selected)
        else:
            tile = selected
            index = packet_map(m, n, world, tile[2], tile[1], tile[4])
        if bf16 and 'warp_k' in item:
            def call():
                deep_gemm._C._gemm_comm_bf16_gemm_rs(a, b, out, ws._state, m, rank, world, selected)
        else:
            operation = deep_gemm.bf16_gemm_nt_reduce_scatter if bf16 else fp8_gemm_nt_reduce_scatter
            def call():
                operation(lhs, rhs, out, workspace=ws, configs=configs)
        def native_reference():
            if bf16:
                if world == 1 and configs is None and 'warp_k' not in item:
                    deep_gemm.bf16_gemm_nt(a, b, partial)
                else:
                    bf16_source_reference(a, b, partial, selected)
            else:
                # Native blockwise excludes the fused BN192 scale adapter.
                deep_gemm.fp8_gemm_nt(lhs, rhs, partial, configs=tile if tile[2] != 192 else None)
        compile_only = bool(os.environ.get('HGGC_WARM_UP')) or deep_gemm.get_compile_mode() == 1
        call()  # Warm up outside capture, or compile the requested kernel only.
        if compile_only or not check_accuracy:
            return dict(passed=True, compile_only=compile_only)
        tags = check_partial_rounding(lhs, rhs, partial, out, ws, call, tile[2], tile[1], tile[4], index=index) if world > 1 else []
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(3):
                call()
        checks = len(tags)
        for trial in range(3):
            generator = torch.Generator(device='cuda').manual_seed(6011 * trial + rank)
            a.copy_(torch.randn(a.shape, device='cuda', generator=generator).to(a.dtype))
            b.copy_(torch.randn(b.shape, device='cuda', generator=generator).to(b.dtype))
            if not bf16:
                sa.copy_(torch.rand(sa.shape, device='cuda', generator=generator) * 0.09 + 0.01)
                sb.copy_(torch.rand(sb.shape, device='cuda', generator=generator) * 0.09 + 0.01)
            if trial > 0:
                a.zero_()
            if trial == 1:
                a[:, 0] = (torch.arange(m, device='cuda') % 8 + 1).to(a.dtype)
                b[:, 0] = (torch.arange(n, device='cuda') % 16 - 8).to(b.dtype)
            native_reference()
            reference = source_sum_reference(partial)
            for invoke in (call, graph.replay):
                invoke()
                torch.cuda.synchronize()
                exact = source_exact(ws, partial, index) if world > 1 else torch.equal(out, partial)
                valid, ratio = source_sum_matches(out, reference)
                assert exact and valid, (item, rank, trial, exact, ratio)
                checks += 1
        print('GEMM_RS_REPLAY_PASS', rank, item, checks, flush=True)
        return dict(passed=True, checks=checks)
    finally:
        try:
            if graph is not None:
                graph.reset()
            # --disable_acc still executes asynchronously; retire it before cleanup.
            torch.cuda.synchronize()
        finally:
            if owns_group:
                dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quantization', choices=('blockwise', 'channelwise', 'bf16', 'both'), default='both')
    parser.add_argument('--tile-policy', choices=('rs_aware', 'native'), default='rs_aware')
    parser.add_argument('--tile', nargs=6, type=int, metavar=('BM', 'BN', 'BK', 'WM', 'WN', 'STAGES'))
    parser.add_argument('--extended', action='store_true', help='also cover general shapes, small K and larger/non-power-of-two M')
    parser.add_argument('--caselist', type=Path, help='replay a GEMM/RS caselist in one process group')
    parser.add_argument('--smoke', action='store_true', help='small checkin set for all dtypes, tails, Graphs and workspace reuse')
    parser.add_argument('--ms', default='8,16,32,64,128,256,512,1024,2048,4096')
    parser.add_argument('--n', type=int, default=7168)
    parser.add_argument('--k', type=int, default=24576, help='K before TP splitting')
    parser.add_argument('--cases', type=int, default=3)
    parser.add_argument('--skew', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=30)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--capture-repeats', type=int, default=16, help='calls per captured Graph in benchmark timing')
    parser.add_argument('--output', default='tmp/gemm_comm/reduce-scatter.json')
    args = parser.parse_args()
    if args.cases < 3:
        parser.error('at least three changing-input cases are required')
    if args.benchmark and min(args.warmup, args.iters, args.rounds, args.capture_repeats) < 1:
        parser.error('benchmark counts must be positive')
    if args.caselist and (args.smoke or args.benchmark):
        parser.error('--caselist cannot be combined with --smoke or --benchmark')
    world = int(os.environ['WORLD_SIZE'])
    if world not in (1, 2, 4, 8) or args.k <= 0 or args.k % world:
        parser.error('TP must be 1, 2, 4 or 8 and the original K must be divisible by TP')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=timedelta(seconds=180))
    if args.caselist:
        from utils import read_cmds_from_file
        items = read_cmds_from_file(str(args.caselist))
        if not items or any(item['gemm_type'] != 'GemmReduceScatter' or item['tp'] != world for item in items):
            raise ValueError('caselist must contain GEMM/RS cases with matching TP')
        ws = get_workspace_for_gemm_reduce_scatter(
            dist.group.WORLD, max(item['m'] for item in items), max(item['n'] for item in items),
            tile_policy=args.tile_policy)
        for item in items:
            replay_case(item, ws)
        torch.cuda.synchronize()
        dist.destroy_process_group()
        return
    if args.smoke:
        check_public_configs({})
        ws = get_workspace_for_gemm_reduce_scatter(dist.group.WORLD, 1024, 4096)
        sms = deep_gemm.get_num_sms()
        from checkin.run_checkin import gemm_rs_smoke_cases
        from utils import parse_deepgemm_string_re
        for spec in gemm_rs_smoke_cases(world, sms):
            replay_case(parse_deepgemm_string_re(spec), ws)
        dist.barrier()
        dist.destroy_process_group()
        return
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / 'deep_gemm/gemm_comm.py',
        root / 'deep_gemm/include/deep_gemm/scheduler/scheduler_cutlass3.cuh',
        *root.joinpath('deep_gemm/include/deep_gemm/gemm_comm').glob('*.cuh'),
        root / 'deep_gemm/include/deep_gemm/impls/fp8_gemm.cuh',
        root / 'csrc/jit_kernels/heuristics/common_fp8.hpp',
        root / 'csrc/jit_kernels/heuristics/common_gemm_comm.hpp',
        root / 'csrc/jit_kernels/impls/int8_gemm.hpp',
        root / 'csrc/jit/compiler.hpp',
        root / 'deep_gemm/__init__.py',
        root / 'deep_gemm/include/deep_gemm/impls/int8_densegemm_cutlass3.cuh',
        root / 'deep_gemm/include/deep_gemm/scheduler/densegemm_scheduler_cutlass3.cuh',
        root / 'csrc/apis/gemm_comm.hpp',
        root / 'csrc/jit_kernels/impls/fp8_gemm.hpp',
        root / 'csrc/jit_kernels/impls/bf16_gemm_comm.hpp',
        root / 'csrc/jit_kernels/impls/fp8_blockwise_gemm_comm.hpp',
        root / 'csrc/jit_kernels/impls/fp8_channelwise_gemm_comm.hpp',
        root / 'csrc/jit_kernels/impls/gemm_comm.hpp',
        Path(__file__).resolve(),
    ]
    report = dict(
        args=vars(args),
        world=dist.get_world_size(),
        torch=torch.__version__,
        device=str(torch.cuda.get_device_properties(torch.cuda.current_device())),
        source_hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        validation=[],
        timings=[],
    )
    check_config_guards(report)
    check_workspace_guards(report)
    check_buffer_reuse(report)
    check_public_configs(report)
    if dist.get_rank() == 0:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    quantizations = ('blockwise', 'channelwise') if args.quantization == 'both' else (args.quantization,)
    for quantization in quantizations:
        shapes = [(m, args.n, args.k, None) for m in map(int, args.ms.split(','))]
        if args.extended:
            # Local K is listed here so all TP sizes exercise the same K tails.
            common = ((1, 16, 768), (2, 48, 640), (4, 80, 1024), (8, 144, 1536),
                      (16, 7200, 3072), (48, 272, 768), (80, 1008, 1024),
                      (512, 1024, 1536), (1024, 4096, 1024))
            channel = ((48, 272, 32), (80, 1008, 64), (96, 4112, 96),
                       (112, 7168, 128), (128, 7200, 256), (144, 8192, 384),
                       (16, 35200, 512), (48, 5136, 544))
            extra = common + channel if quantization == 'channelwise' else common
            shapes += [(lm * world, n, k * world, None) for lm, n, k in extra]
            if quantization == 'bf16':
                shapes += [(13 * world, 7168, 1536 * world, None),
                           (13 * world, 272, 768 * world, (128, 80, 128, 16, 80, 2)),
                           (384, 272, 128 * world, (384, 256, 64, 96, 64, 2))]
            else:
                shapes.append((13 * world, 272, 768 * world, (128, 32, 128, 16, 16, 2)))
        for m, n, k_global, tile in dict.fromkeys(shapes):
            if quantization == 'bf16':
                report['validation'].append(
                    run_bf16_case(
                        dict(quantization='bf16', m=m, n=n, k_global=k_global,
                             tile_policy=args.tile_policy, configs=make_configs(args.tile if tile is None else tile)),
                        timed=args.benchmark, warmup=args.warmup, iters=args.iters,
                        rounds=args.rounds, capture_repeats=args.capture_repeats, cases=args.cases,
                    )
                )
                if dist.get_rank() == 0:
                    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
            else:
                run_case(quantization, m, n, k_global, args, report, tile=tile)
        if args.extended and quantization == 'channelwise':
            # Multiple M/N tiles, plus N/K tails, with BN64.
            for lm, n, k in ((3, 48, 128), (32, 128, 128), (48, 144, 96)):
                run_case(quantization, lm * world, n, k * world, args, report,
                         tile=(16, 64, 128, 16, 16, 2))
            if world in (2, 4):
                # A small N can still produce a full queued CTA grid with an explicit tile.
                run_case(quantization, 1280, 64, 2048 * world, args, report,
                         tile=(32, 64, 256, 32, 32, 8))
    dist.barrier()
    torch.cuda.synchronize()
    if dist.get_rank() == 0:
        print('COMPLETE', args.output, flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
