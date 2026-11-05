"""Distributed GEMM+RS correctness and optional matched PCCL benchmarking.

Run with torchrun --standalone --nproc-per-node=2 (or 1, 4, 8). All validation,
Graph replay, state reuse and timing for both quantizations live here.
"""

import argparse
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm import get_workspace_for_gemm_reduce_scatter, fp8_gemm_nt_reduce_scatter


def packet_map(m, n, world):
    """Recover logical rows/columns from the PPU0015 MMA packet layout."""
    lm = m // world
    rows = min(16, max(4, lm))
    row = torch.arange(lm, device='cuda')[:, None]
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


def source_exact(buffer, partial, index):
    if buffer.world == 1:
        return None  # TP1 runs the native GEMM without packet staging.
    m, n = partial.shape
    sources = [torch.empty_like(partial) for _ in range(buffer.world)]
    dist.all_gather(sources, partial)
    expected = torch.stack([p.chunk(buffer.world)[buffer.rank] for p in sources])
    phase = ((int(buffer._completed[0].item()) >> 32) - 1) & 1
    actual = buffer._slots[phase, :max(m, 4 * buffer.world) * n].reshape(buffer.world, -1)[:, index]
    return bool(torch.equal(actual.reshape_as(expected), expected))


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
            if query(8, 7168, 24576 // world, world)[0][0] != 1:
                raise RuntimeError('GEMM + RS ignored set_num_sms')
    finally:
        deep_gemm.set_num_sms(num_sms)
    invalid = [(query, (2**20, 7168, 24576 // world, world), {}) for query in queries]
    invalid += [(query, (3 * world, 7168, 768, world), {}) for query in queries]
    invalid += [(queries[0], (8, 7168, 512, world), {}),
                (queries[1], (8, 7168, 48, world), {}),
                (queries[1], (48 * world, 272, 32, world), dict(configs=make_configs((64, 64, 32, 32, 32, 2))))]
    invalid += [
        (queries[0], (8, 7168, 24576 // world, world), dict(configs=make_configs(tile)))
        for tile in ((16, 192, 128, 16, 16, 1), (16, 64, 128, 16, 64, 3),
                     (16, 96, 128, 16, 16, 2), (16, 160, 128, 16, 16, 2),
                     (16, 224, 128, 16, 16, 2), (1024, 1024, 128, 256, 256, 3))
    ]
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
        queries[0](8, 7168, 24576 // world, world, configs=make_configs((32, bn, 128, 16, 16, 2)))
    queries[1](8, 7168, 24576 // world, world, configs=make_configs((16, 96, 128, 16, 16, 2)))
    report['config_guards'] = dict(sm_budget_queries=len(queries), rejected_configs=len(invalid) + (world == 1),
                                  accepted_configs=len(blockwise_bns) + 1)


def check_buffer_reuse(report):
    """Alternate shapes, quantizations, presets and captured calls on one buffer."""
    world, rank = dist.get_world_size(), dist.get_rank()
    buffer = get_workspace_for_gemm_reduce_scatter(dist.group.WORLD, 33 * world, 272)
    table = buffer._state.clone()
    allocation = buffer.buffer.data_ptr()
    calls = []
    # Decreasing and increasing M/N/K exercise fixed phase strides. The last
    # capacity row is deliberately unused, including padding for small local M.
    for m, n, k, channelwise in ((32 * world, 272, 768, False),
                                 (world, 16, 64, True),
                                 (16 * world, 144, 640, False),
                                 (8 * world, 80, 96, True)):
        generator = torch.Generator(device='cuda').manual_seed(233 + m + rank)
        x = torch.randn((m, k), generator=generator, device='cuda').to(torch.float8_e4m3fn)
        w = torch.randn((n, k), generator=generator, device='cuda').to(torch.float8_e4m3fn)
        a_shape = (m, 1) if channelwise else (m, k // 128)
        b_shape = (n, 1) if channelwise else ((n + 127) // 128, k // 128)
        sx = torch.rand(a_shape, generator=generator, device='cuda') * 0.09 + 0.01
        sw = torch.rand(b_shape, generator=generator, device='cuda') * 0.09 + 0.01
        lhs, rhs = (x, sx), (w, sw)
        partial = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        expected = torch.empty((m // world, n), dtype=torch.bfloat16, device='cuda')
        out = torch.empty_like(expected)
        deep_gemm.fp8_gemm_nt(lhs, rhs, partial)
        dist.reduce_scatter_tensor(expected, partial)
        fp8_gemm_nt_reduce_scatter(lhs, rhs, out, workspace=buffer)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fp8_gemm_nt_reduce_scatter(lhs, rhs, out, workspace=buffer)
        calls.append((lhs, rhs, out, expected, graph, packet_map(m, n, world), partial))
        for preset in ('native', 'decode'):
            buffer.preset = preset
            actual = buffer._get_config(m, n, k, channelwise, None)
            quantization = 'channelwise' if channelwise else 'blockwise'
            query = getattr(deep_gemm._C, f'_gemm_comm_fp8_{quantization}_gemm_rs_config')
            if actual != query(m, n, k, world, preset=preset):
                raise RuntimeError('buffer reused a config from a different preset or shape')
    # A tuned shape makes decode/native differ on TP2/TP4, detecting stale
    # cached configurations after changing the public preset attribute.
    for preset in ('decode', 'native', 'decode'):
        buffer.preset = preset
        actual = buffer._get_config(8, 7168, 24576 // world, False, None)
        expected_config = deep_gemm._C._gemm_comm_fp8_blockwise_gemm_rs_config(
            8, 7168, 24576 // world, world, preset=preset)
        if actual != expected_config:
            raise RuntimeError('changing the buffer preset returned a stale config')
    for repeat in range(5):
        for lhs, rhs, out, expected, graph, index, partial in calls[::(-1 if repeat % 2 else 1)]:
            if repeat % 2:
                graph.replay()
            else:
                fp8_gemm_nt_reduce_scatter(lhs, rhs, out, workspace=buffer)
            torch.cuda.synchronize()
            error = float((out.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30))
            if not bool(torch.isfinite(out).all()) or error >= 0.015 or source_exact(buffer, partial, index) is False:
                raise RuntimeError('changing buffer shape or captured calls corrupted output')
    if buffer.buffer.data_ptr() != allocation or not torch.equal(table, buffer._state):
        raise RuntimeError('changing shapes mutated buffer allocation or pointer metadata')
    lhs, rhs, out, *_ = calls[0]
    for too_small in ((world, 272), (33 * world, 16)):
        small = get_workspace_for_gemm_reduce_scatter(dist.group.WORLD, *too_small)
        try:
            fp8_gemm_nt_reduce_scatter(lhs, rhs, out, workspace=small)
        except ValueError:
            pass
        else:
            raise RuntimeError('undersized communication buffer was accepted')
    report['buffer_reuse_checks'] = 20
    torch.cuda.synchronize()
    dist.barrier()


def check_public_configs(report):
    """Reject malformed configs before the TP1 native or multi-rank launch."""
    world = dist.get_world_size()
    m, n, k = 32 * world, 272, 768
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
        valid = (1, 16, 64, 128, 16, 16, 2, (0, 0, 0))
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
    report['public_config_checks'] = dict(rejected=rejected, valid=2)
    dist.barrier()


def run_case(quantization, m, n, k_global, args, report, tile=None):
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
    buffer = get_workspace_for_gemm_reduce_scatter(dist.group.WORLD, m, n, preset=args.preset)
    rs_config = buffer._get_config(m, n, k, quantization == 'channelwise', configs)
    buffers = {'fused': buffer}
    lhs, rhs = (x, sx), (w, sw)
    partial = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    output = torch.empty((lm, n), dtype=torch.bfloat16, device='cuda')
    fused_output = torch.empty_like(output)
    ar_output = torch.empty_like(partial) if args.allreduce else None
    index = packet_map(m, n, world)

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

    def baseline_ar():
        native(lhs, rhs, ar_output)
        dist.all_reduce(ar_output)
        return ar_output

    functions = {'baseline_rs': baseline, 'gemm': dense, 'pccl_rs': collective}
    results = {'baseline_rs': output, 'gemm': partial, 'pccl_rs': output}
    if args.allreduce:
        functions['baseline_ar'] = baseline_ar
        results['baseline_ar'] = ar_output
    for name, buffer in buffers.items():
        def fused(buffer=buffer):
            fp8_gemm_nt_reduce_scatter(lhs, rhs, fused_output, workspace=buffer, configs=configs)
            return fused_output

        functions[name] = fused
        results[name] = fused_output
    fresh(0)
    baseline()
    for name, function in functions.items():
        for _ in range(3):
            function()
        torch.cuda.synchronize()
        dist.barrier()
    # Distinct outputs share only communication state, including during capture.
    # Explicit configs override the preset and their placeholder SMEM is recomputed.
    preserved = fused_output.clone()
    alternate = torch.empty_like(fused_output)
    table = buffer._state.clone()
    explicit = (*rs_config[0][:7], (0, 0, 0))
    fp8_gemm_nt_reduce_scatter(lhs, rhs, alternate, workspace=buffer, configs=explicit)
    torch.cuda.synchronize()
    if not torch.equal(preserved, fused_output) or not torch.equal(alternate, preserved):
        raise RuntimeError('explicit configs or an alternate output changed prior results')
    alternate_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(alternate_graph):
        fp8_gemm_nt_reduce_scatter(lhs, rhs, alternate, workspace=buffer, configs=configs)
    alternate.fill_(float('nan'))
    alternate_graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(preserved, fused_output) or not torch.equal(alternate, preserved):
        raise RuntimeError('Graph replay wrote a different call output')
    # Change the actual resident grid, then alternate captured configurations.
    # The communication epoch must count calls independently of CTA count.
    limited = (1, *rs_config[0][1:7], (0, 0, 0))
    fp8_gemm_nt_reduce_scatter(lhs, rhs, alternate, workspace=buffer, configs=limited)
    limited_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(limited_graph):
        fp8_gemm_nt_reduce_scatter(lhs, rhs, alternate, workspace=buffer, configs=limited)
    for _ in range(3):
        limited_graph.replay()
        alternate_graph.replay()
        functions['fused']()
    torch.cuda.synchronize()
    if not torch.equal(preserved, fused_output) or not torch.equal(alternate, preserved):
        raise RuntimeError('changing configs or Graph grids corrupted output')
    if not torch.equal(table, buffer._state):
        raise RuntimeError('per-call output changed the communication pointer table')
    for invalid_out in (torch.empty_like(fused_output, dtype=torch.float32), fused_output.T):
        try:
            fp8_gemm_nt_reduce_scatter(lhs, rhs, invalid_out, workspace=buffer, configs=configs)
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
        if quantization == 'blockwise':
            a32 = x.float() * sx.repeat_interleave(128, dim=1)
            b32 = w.float() * sw.repeat_interleave(128, dim=0)[:n].repeat_interleave(128, dim=1)
            reference = a32 @ b32.T
            del a32, b32
        else:
            reference = (x.float() @ w.float().T) * sx * sw.T
        ref_rs = torch.empty((lm, n), dtype=torch.float32, device='cuda')
        dist.reduce_scatter_tensor(ref_rs, reference)
        ref_ar = reference.clone() if args.allreduce else None
        if ref_ar is not None:
            dist.all_reduce(ref_ar)
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
                expected = ref_ar if name == 'baseline_ar' else ref_rs
                error = float((actual.float() - expected).norm() / expected.norm().clamp_min(1e-30))
                exact = source_exact(buffer, partial, index) if buffer is not None else None
                valid = bool(torch.isfinite(actual).all()) and error < 0.015 and exact is not False
                entry = dict(rank=rank, relative_l2=error, source_exact=exact, passed=valid)
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
        del reference, ref_rs, ref_ar
    # Reject a malformed scale tensor before launching either quantization.
    for buffer in buffers.values():
        wrong_shape = (m, 1) if quantization == 'blockwise' else (m, 2)
        wrong = torch.empty(wrong_shape, dtype=torch.float32, device='cuda')
        try:
            fp8_gemm_nt_reduce_scatter((x, wrong), rhs, fused_output, workspace=buffer, configs=configs)
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
                        config=rs_config[0] if buffer else None,
                        specialize=rs_config[1] if buffer else None,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quantization', choices=('blockwise', 'channelwise', 'both'), default='both')
    parser.add_argument('--preset', '--blockwise-preset', choices=('decode', 'native'), default='decode')
    parser.add_argument('--tile', nargs=6, type=int, metavar=('BM', 'BN', 'BK', 'WM', 'WN', 'STAGES'))
    parser.add_argument('--extended', action='store_true', help='also cover general shapes, small K and larger/non-power-of-two M')
    parser.add_argument('--ms', default='8,16,32,64,128,256,512,1024,2048,4096')
    parser.add_argument('--n', type=int, default=7168)
    parser.add_argument('--k', type=int, default=24576, help='K before TP splitting')
    parser.add_argument('--cases', type=int, default=3)
    parser.add_argument('--skew', action='store_true')
    parser.add_argument('--allreduce', action='store_true', help='also validate and time full GEMM+AllReduce')
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
    world = int(os.environ['WORLD_SIZE'])
    if world not in (1, 2, 4, 8) or args.k <= 0 or args.k % world:
        parser.error('TP must be 1, 2, 4 or 8 and the original K must be divisible by TP')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=timedelta(seconds=180))
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / 'deep_gemm/gemm_comm.py',
        *root.joinpath('deep_gemm/include/deep_gemm/gemm_comm').glob('*.cuh'),
        root / 'deep_gemm/include/deep_gemm/impls/fp8_gemm.cuh',
        root / 'deep_gemm/include/deep_gemm/impls/direct_sfb_mainloop.cuh',
        root / 'csrc/jit_kernels/heuristics/common_fp8.hpp',
        root / 'csrc/jit_kernels/heuristics/common_gemm_comm.hpp',
        root / 'csrc/jit_kernels/impls/int8_gemm.hpp',
        root / 'csrc/jit/compiler.hpp',
        root / 'deep_gemm/__init__.py',
        root / 'deep_gemm/include/deep_gemm/impls/int8_gemm_cutlass3.cuh',
        root / 'csrc/apis/gemm_comm.hpp',
        root / 'csrc/jit_kernels/impls/fp8_gemm.hpp',
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
    check_buffer_reuse(report)
    check_public_configs(report)
    if dist.get_rank() == 0:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    quantizations = ('blockwise', 'channelwise') if args.quantization == 'both' else (args.quantization,)
    for quantization in quantizations:
        shapes = [(m, args.n, args.k) for m in map(int, args.ms.split(','))]
        if args.extended:
            # Local K is listed here so all TP sizes exercise the same K tails.
            common = ((1, 16, 768), (2, 48, 640), (4, 80, 1024), (8, 144, 1536),
                      (16, 7200, 3072), (48, 272, 768), (80, 1008, 1024),
                      (512, 1024, 1536), (1024, 4096, 1024))
            channel = ((48, 272, 32), (80, 1008, 64), (96, 4112, 96),
                       (112, 7168, 128), (128, 7200, 256), (144, 8192, 384),
                       (16, 35200, 512), (48, 5136, 544))
            extra = common + channel if quantization == 'channelwise' else common
            shapes += [(lm * world, n, k * world) for lm, n, k in extra]
        for m, n, k_global in dict.fromkeys(shapes):
            run_case(quantization, m, n, k_global, args, report)
        if args.extended and quantization == 'channelwise':
            # Multiple M/N tiles, plus N/K tails, with BN64.
            for lm, n, k in ((32, 128, 128), (48, 144, 96)):
                run_case(quantization, lm * world, n, k * world, args, report,
                         tile=(16, 64, 128, 16, 16, 2))
    dist.barrier()
    torch.cuda.synchronize()
    if dist.get_rank() == 0:
        print('COMPLETE', args.output, flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
