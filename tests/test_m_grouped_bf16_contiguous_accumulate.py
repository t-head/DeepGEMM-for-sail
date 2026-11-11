"""Grouped BF16 NT correctness and performance tests.

Run this script without arguments for the correctness regression, or use pytest.
Pass --benchmark to measure the nine compact shapes in three independent processes.
Each expert receives a multiple of 256 rows, with no padding between experts.
Use --round 0, 1, or 2 to measure one round and --output PATH to save the JSON.
Either --round or --output by itself selects the benchmark.
"""
import argparse
import ctypes
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time
from unittest import SkipTest

import deep_gemm
import torch


ACCUMULATION_MODES = ("alias", "alias", "overwrite", "separate", "none", "alias")


def padded_group_lengths(lengths, alignment):
    return [((count + alignment - 1) // alignment) * alignment for count in lengths]


def make_operands(lengths, padded, n, k, device, generator=None):
    # Dyadic BF16 operands keep the per-expert dot products exact in FP32.
    def values(shape):
        return torch.randint(-4, 5, shape, dtype=torch.int32, device=device,
                             generator=generator).bfloat16().div_(8)

    m = sum(padded)
    return values((m, k)), values((len(lengths), n, k)), values((m, n))


def group_layout_and_reference(a, b, lengths, padded, *, check, benchmark_inputs=False):
    # The kernel consumes padded row IDs. Keep the FP32 reference compact so
    # its rows correspond to precisely the valid rows selected from D.
    indices = torch.full((sum(padded),), -1, dtype=torch.int32, device=a.device)
    product = torch.empty((sum(lengths), b.size(1)), dtype=torch.float32, device=a.device) if check else None
    offset, logical = 0, 0
    for expert, (count, stride) in enumerate(zip(lengths, padded)):
        rows = slice(offset, offset + count)
        indices[rows] = expert
        if benchmark_inputs:
            a[offset + count:offset + stride].zero_()
        if count and check:
            out = product[logical:logical + count]
            out.copy_(a[rows].float() @ b[expert].float().T)
            if benchmark_inputs and expert in (0, len(lengths) - 1):
                ir, ic = [0, count - 1], [0, b.size(1) - 1]
                anchor = a[rows][ir].float().cpu() @ b[expert][ic].float().cpu().T
                torch.testing.assert_close(out[ir][:, ic].cpu(), anchor, rtol=0, atol=0)
        offset += stride
        logical += count
    return indices, product


def exercise_grouped_accumulation(lengths, n, k, modes, *, configs=None, c_offset=0,
                                  check=True, reference_device="cpu", benchmark=False, use_psum_layout=False):
    padded = lengths if use_psum_layout else padded_group_lengths(
        lengths, deep_gemm.get_m_alignment_for_contiguous_layout())
    m = sum(padded)
    rng = torch.Generator(device=reference_device).manual_seed(103)
    a, b, initial = make_operands(lengths, padded, n, k, reference_device, rng)
    if m and k >= 2:
        # AB=1+1/256 and C=1/256 distinguish FP32 addition from an early BF16 round.
        first_expert = next(i for i, count in enumerate(lengths) if count)
        a[0].zero_()
        a[0, 0], a[0, 1] = 1, 1 / 256
        b[first_expert, 0].zero_()
        b[first_expert, 0, :2] = 1
        initial[0, 0] = 1 / 256

    layout, product = group_layout_and_reference(a, b, lengths, padded, check=check)
    valid = layout >= 0
    if use_psum_layout:
        layout = torch.tensor(lengths, dtype=torch.int32, device=reference_device).cumsum(0, dtype=torch.int32)
    a, b, indices = a.cuda(), b.cuda(), layout.cuda()

    sentinel = -117
    c_store = torch.full((m * n + 32 + c_offset,), sentinel, device="cuda", dtype=torch.bfloat16)
    c = c_store[16 + c_offset:-16].view(m, n)
    c.copy_(initial)
    d_store = torch.full((m * n + 32,), sentinel, device="cuda", dtype=torch.bfloat16)
    d = d_store[16:-16].view(m, n)
    d.copy_(initial)
    initial_valid = initial[valid] if check else None
    expected = initial_valid
    op = deep_gemm.m_grouped_bf16_gemm_nt_contiguous
    for mode in modes:
        kwargs = {"configs": configs} if configs is not None else {}
        if mode != "overwrite":
            kwargs["c"] = {"alias": d, "separate": c, "none": None}[mode]
        if use_psum_layout:
            kwargs["use_psum_layout"] = True
            if mode != "none":
                kwargs["ensure_zero_padding"] = False
        if mode in ("overwrite", "separate"):
            run = partial(op, lhs=a, rhs=b, out=d, m_indices=indices, **kwargs)
        else:
            run = partial(op, a, b, d, indices, **kwargs)
        run()
        if check:
            addend = expected if mode == "alias" else initial_valid
            expected = ((product + addend.float()) if mode in ("alias", "separate") else product).bfloat16()
            torch.testing.assert_close(d.to(reference_device)[valid], expected, atol=0, rtol=0)
        if benchmark:
            seconds = deep_gemm.bench_kineto(run, 'gemm', suppress_kineto_output=True)
            print(f"Grouped BF16 {mode}: {seconds * 1e6:.2f} us, {2*sum(lengths)*n*k/seconds/1e12:.2f} TFLOPS")
    if check:
        torch.testing.assert_close(c.to(reference_device), initial, atol=0, rtol=0)
        for storage, first in ((d_store, 16), (c_store, 16 + c_offset)):
            assert (storage[:first] == sentinel).all() and (storage[-16:] == sentinel).all()


def invoke_invalid_compact_ends(m, ends):
    # A device assertion can invalidate the CUDA context. Run one failure per child.
    a = torch.ones((m, 72), device="cuda", dtype=torch.bfloat16)
    b = torch.ones((len(ends), 136, 72), device="cuda", dtype=torch.bfloat16)
    d = torch.empty((m, 136), device="cuda", dtype=torch.bfloat16)
    valid = torch.full((len(ends),), m, dtype=torch.int32, device="cuda")
    op = deep_gemm.m_grouped_bf16_gemm_nt_contiguous
    op(a, b, d, valid, use_psum_layout=True)
    torch.cuda.synchronize()
    print("valid compact layout passed", flush=True)
    op(a, b, d, torch.tensor(ends, dtype=torch.int32, device="cuda"), use_psum_layout=True)
    torch.cuda.synchronize()


def assert_rejected(a, b, d, layout, **kwargs):
    original = d.clone()
    try:
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, layout, **kwargs)
    except RuntimeError:
        torch.testing.assert_close(d, original, rtol=0, atol=0)
    else:
        raise AssertionError("Invalid grouped BF16 arguments were accepted")


def check_compact_grouped_layout():
    # Leading, middle and trailing empty experts preserve their original IDs.
    # The second case has M == G, so the layout mode cannot be inferred by size.
    for lengths in ((17, 129, 0, 65), (0, 1, 3, 0), (0, 0, 0, 0),
                    (0,) * 127 + (1, 0, 321, 0)):
        exercise_grouped_accumulation(lengths, 136, 72, ACCUMULATION_MODES, use_psum_layout=True)

    a = torch.ones((5, 72), device="cuda", dtype=torch.bfloat16)
    b = torch.ones((3, 136, 72), device="cuda", dtype=torch.bfloat16)
    d = torch.full((5, 136), -117, device="cuda", dtype=torch.bfloat16)
    ends = torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32)
    for bad in (ends.long(), ends.view(1, 3), ends[:2], torch.cat((ends.new_zeros(1), ends)),
                ends.repeat_interleave(2)[::2], ends.cpu()):
        assert_rejected(a, b, d, bad, use_psum_layout=True)

    # Check the error after synchronization in both synchronous and asynchronous runtimes.
    child = ("import json, runpy, sys; "
             "runpy.run_path(sys.argv[1])['invoke_invalid_compact_ends'](*json.loads(sys.argv[2]))")
    for m, bad_ends in ((5, [-1, 1, 5]), (5, [3, 2, 5]), (5, [0, 1, 4]), (0, [0, 0, 1])):
        proc = subprocess.run([sys.executable, "-c", child, str(Path(__file__).resolve()),
                               json.dumps([m, bad_ends])], capture_output=True, text=True)
        output = proc.stdout + proc.stderr
        rejected = "cu_seqlens must" in output or "device-side assert" in output
        assert proc.returncode != 0 and "valid compact layout passed" in proc.stdout and rejected, (
            f"Invalid ends {bad_ends} with M={m}: exit {proc.returncode}\n{output}")


def check_ppu_interface():
    a = torch.ones((128, 64), dtype=torch.bfloat16, device="cuda")
    b = torch.ones((1, 128, 64), dtype=torch.bfloat16, device="cuda")
    d = torch.full((128, 128), 3, dtype=torch.bfloat16, device="cuda")
    indices = torch.zeros(128, dtype=torch.int32, device="cuda")
    op = deep_gemm.m_grouped_bf16_gemm_nt_contiguous
    op(lhs=a, rhs=b, out=d, m_indices=indices, c=d)
    assert (d == 67).all()
    # Preserve the original positional configs slot; C is appended after it.
    config = (1, 128, 128, 64, 32, 32, 3, (0, 0, 0))
    op(a, b, d, indices, "nk", False, True, None, config, d)
    assert (d == 131).all()
    for kwargs in ({"a": a, "b": b, "d": d, "grouped_layout": indices},
                   {"lhs": a, "rhs": b, "out": d, "m_indices": indices, "epilogue": None}):
        original = d.clone()
        try:
            op(**kwargs)
        except TypeError:
            torch.testing.assert_close(d, original, atol=0, rtol=0)
        else:
            raise AssertionError("Unexpected compatibility overload accepted")


def run_grouped_accumulation_regression(check=True):
    if not torch.cuda.is_available() or "M890" not in torch.cuda.get_device_name():
        raise SkipTest("The explicit grouped kernel configurations require M890")
    # One persistent CTA visits multiple output tiles. Different K counts rotate
    # the final live A/B slot. Check valid rows with one empty expert and padding.
    tile = (1, 128, 128, 64, 32, 32, 3, (0, 0, 0))
    cases = [(256, k, tile, 0) for k in (32, 64, 128, 192, 200, 256)]
    mixed_tile = (1, 128, 256, 64, 32, 64, 2, (0, 0, 0))
    # Cover an unused prologue slot and both possible final slots with mixed shared/global C reads.
    cases += [(256, k, mixed_tile, 0) for k in (32, 192, 256)]
    cases += [
        (256, 128, (1, 128, 128, 64, 32, 32, 4, (0, 0, 0)), 0),  # prologue and drain both supply C
        (264, 320, tile, 0),  # partial last N tile
        (104, 192, (1, 64, 32, 64, 32, 16, 3, (0, 0, 0)), 0),  # narrow vectorized C copy
        (256, 256, None, 1),  # misaligned independent C
        (384, 384, None, 0),  # automatic tile selection
    ]
    for n, k, config, c_offset in cases:
        exercise_grouped_accumulation(
            (119, 0, 247, 113), n, k, ACCUMULATION_MODES,
            configs=config, c_offset=c_offset, check=check)
    # A whole tile of -1 IDs precedes a valid expert. Both epilogues must skip
    # this tile before deriving B or D addresses, then process the next tile.
    a = torch.ones((256, 64), device="cuda", dtype=torch.bfloat16)
    b = torch.ones((1, 256, 64), device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros(256, device="cuda", dtype=torch.int32)
    indices[:128] = -1
    for config in (tile, mixed_tile):
        d = torch.full((256, 256), 5, device="cuda", dtype=torch.bfloat16)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, indices, c=d, configs=config)
        if check:
            assert (d[:128] == 5).all() and (d[128:] == 69).all()

    # Partial N reaches the direct-read part of the final C tile. Exercise
    # unused prologue slots, both final A/B slots and compact expert boundaries.
    for compact in (False, True):
        for k in (32, 128, 192):
            exercise_grouped_accumulation(
                (0, 3, 139, 1, 0), 456, k, ACCUMULATION_MODES,
                configs=mixed_tile, check=check, use_psum_layout=compact)

    if check:
        check_ppu_interface()
        check_compact_grouped_layout()
        # With a zero addend the BF16 output must match the ordinary MMA path,
        # also for real-valued inputs whose sums are sensitive to K reduction order.
        rng = torch.Generator().manual_seed(4711)
        a = torch.randn((256, 520), generator=rng).bfloat16().cuda()
        b = torch.randn((2, 264, 520), generator=rng).bfloat16().cuda()
        indices = torch.arange(2, device="cuda", dtype=torch.int32).repeat_interleave(128)
        overwrite = torch.empty((256, 264), device="cuda", dtype=torch.bfloat16)
        alias = torch.zeros_like(overwrite)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, overwrite, indices, configs=tile)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, alias, indices, c=alias, configs=tile)
        torch.testing.assert_close(alias, overwrite, atol=0, rtol=0)

        # DLPack can expose a partially overlapping address range through a new StorageImpl.
        a = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16)
        b = torch.ones_like(a).unsqueeze(0)
        indices = torch.zeros(128, device="cuda", dtype=torch.int32)
        storage = torch.arange(a.numel() + 8, device="cuda", dtype=torch.int32).bfloat16()
        d = storage[:-8].view_as(a)
        for bad_c in (d.float(), d[:1], torch.from_dlpack(storage)[8:].view_as(a)):
            assert_rejected(a, b, d, indices, c=bad_c)


def test_m_grouped_bf16_contiguous_accumulate():
    run_grouped_accumulation_regression()


WARMUP = 5
CALLS = 20
MODES = ('no_c', 'c_eq_d')
# (local experts, total valid token rows, output columns, reduction dimension)
SHAPES = ((12, 39936, 3072, 4096),
          (14, 8192, 3072, 3584), (14, 8192, 3584, 3072),
          (56, 8192, 3072, 3584), (56, 8192, 3584, 3072),
          (14, 8192, 1024, 1024), (56, 8192, 1024, 1024),
          (56, 8192, 400, 896), (56, 8192, 896, 400))
PEAK_BF16_TFLOPS = 415
HBM_BANDWIDTH_GBPS = 2600


def capture_first_warmup(op, stream):
    libc = ctypes.CDLL(None)
    sys.stdout.flush()
    libc.fflush(None)
    previous = os.environ.get('show_log')
    saved = os.dup(1)
    with tempfile.TemporaryFile() as buf:
        try:
            os.dup2(buf.fileno(), 1)
            os.environ['show_log'] = '1'
            op()
            stream.synchronize()
        except BaseException:
            libc.fflush(None)
            buf.seek(0)
            print(buf.read().decode(errors='replace'), file=sys.stderr, flush=True)
            raise
        finally:
            libc.fflush(None)
            os.dup2(saved, 1)
            os.close(saved)
            if previous is None:
                os.environ.pop('show_log', None)
            else:
                os.environ['show_log'] = previous
        buf.seek(0)
        return buf.read().decode(errors='replace')


def benchmark_case(round_idx, index, e, total_rows, n, k, replica, align):
    if total_rows % align:
        raise ValueError(f'Compact benchmark requires total_rows divisible by {align}, got {total_rows}')
    quotient, remainder = divmod(total_rows // align, e)
    lengths = [align * (quotient + (g < remainder)) for g in range(e)]
    seed = 20261110 + round_idx * 1000 + 2 * index + replica - 1
    torch.manual_seed(seed)

    a, b, initial = make_operands(lengths, lengths, n, k, 'cuda')
    d = torch.empty_like(initial)
    _, product = group_layout_and_reference(a, b, lengths, lengths, check=True, benchmark_inputs=True)
    ends = torch.tensor(lengths, device=a.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    product_bf16 = product.bfloat16()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    order = MODES if (round_idx + index + replica - 1) % 2 == 0 else MODES[::-1]
    measured = {}
    with torch.cuda.stream(stream):
        for mode in order:
            alias = mode == 'c_eq_d'
            kwargs = {'c': d} if alias else {}

            def op():
                deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, ends, use_psum_layout=True, **kwargs)

            d.copy_(initial)
            log = capture_first_warmup(op, stream)
            match = re.search(r'WithC:(\d),\s*StageC:(\d)', log)
            if match is None or int(match[1]) != int(alias):
                raise RuntimeError(f'Invalid {mode} dispatch: {log}')
            expected = (product + initial.float()).bfloat16() if alias else product_bf16
            torch.testing.assert_close(d, expected, rtol=0, atol=0)
            for _ in range(WARMUP - 1):
                op()
            stream.synchronize()

            # Capture only records commands. The single replay below executes all 20 calls.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(CALLS):
                    op()
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            d.copy_(initial)
            start.record(stream)
            end.record(stream)
            end.synchronize()  # initialize timing objects and finish the out-of-window reset
            t0 = time.time_ns()
            start.record(stream)
            graph.replay()
            end.record(stream)
            end.synchronize()
            t1 = time.time_ns()
            ms = start.elapsed_time(end)
            assert ms > 0
            if alias:
                expected = initial
                for _ in range(CALLS):
                    expected = (product + expected.float()).bfloat16()
            torch.testing.assert_close(d, expected, rtol=0, atol=0)
            measured[mode] = dict(avg_us=1000 * ms / CALLS, elapsed_ms=ms, window_ns=[t0, t1],
                                  stage_c=bool(int(match[2])), dispatch_log=log, correctness_passed=True)
            del graph
    stream.synchronize()
    return dict(e=e, sum_t=total_rows, n=n, k=k, replica=replica, seed=seed, m_physical=total_rows, layout="compact",
                alignment=align, lengths=lengths, padded_lengths=lengths, mode_order=order, modes=measured)


def run_round(round_idx):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    alignment = 256
    pkg = Path(deep_gemm.__file__).resolve().parent
    rev = subprocess.run(['git', '-C', str(pkg.parent), 'rev-parse', 'HEAD'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    result = dict(round=round_idx, warmup=WARMUP, calls=CALLS, pid=os.getpid(),
                  physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES', os.environ.get('HGGC_VISIBLE_DEVICES', 'runtime-default')),
                  device=str(torch.cuda.get_device_properties(0)),
                  package=str(pkg), git_head=rev.stdout.strip() if rev.returncode == 0 else None,
                  native_sha256=hashlib.sha256(Path(deep_gemm._C.__file__).read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=[])
    # Separate duplicate square measurements in the run. Rotate/reverse the case order between rounds.
    schedule = [(i, *shape, 1) for i, shape in enumerate(SHAPES)]
    schedule += [(i, *shape, 2) for i, shape in enumerate(SHAPES) if shape[2] == shape[3]]
    if round_idx == 1:
        schedule.reverse()
    elif round_idx == 2:
        schedule = schedule[3:] + schedule[:3]
    for i, e, total_rows, n, k, replica in schedule:
        print(f'Round {round_idx} E={e} sumT={total_rows} N={n} K={k} replica={replica}', flush=True)
        record = benchmark_case(round_idx, i, e, total_rows, n, k, replica, alignment)
        result['cases'].append(record)
        print('PASS', {m: round(v['avg_us'], 2) for m, v in record['modes'].items()}, flush=True)
    return result


def summarize(rounds):
    if sorted(r['round'] for r in rounds) != [0, 1, 2]:
        raise ValueError('Expected three independent rounds')
    rows = []
    for e, total_rows, n, k in SHAPES:
        flops = 2 * total_rows * n * k
        shape = (e, total_rows, n, k)
        cases_by_round = [[c for c in r['cases'] if (c['e'], c['sum_t'], c['n'], c['k']) == shape]
                          for r in rounds]
        active_counts = {sum(length > 0 for length in c['lengths']) for cases in cases_by_round for c in cases}
        if len(active_counts) != 1:
            raise ValueError(f'Expected the same number of nonempty experts in all measurements for {shape}')
        active_e, = active_counts
        for mode in MODES:
            replica_tflops, stages = [], set()
            for replica in range(1, 3 if n == k else 2):
                latencies = []
                for cases in cases_by_round:
                    matches = [c for c in cases if c['replica'] == replica]
                    if len(matches) != 1:
                        raise ValueError(f'Missing or duplicated measurement for {shape}, replica {replica}')
                    timing = matches[0]['modes'][mode]
                    latencies.append(timing['avg_us'])
                    stages.add(int(timing['stage_c']))
                replica_tflops.append(flops / (statistics.median(latencies) * 1e6))
            tf = statistics.mean(replica_tflops)
            # For a square shape, report the latency equivalent to the prescribed mean throughput.
            latency = flops / (tf * 1e6)
            beta = int(mode == 'c_eq_d')
            data_bytes = 2 * (total_rows * k + active_e * n * k + (1 + beta) * total_rows * n)
            gbps = data_bytes / (latency * 1000)
            rows.append(dict(e=e, active_e=active_e, sum_t=total_rows, n=n, k=k, mode=mode, beta=beta,
                             latency_us=latency, tflops=tf, mfu_pct=100 * tf / PEAK_BF16_TFLOPS,
                             effective_gbps=gbps, mbu_pct=100 * gbps / HBM_BANDWIDTH_GBPS,
                             stage_c=sorted(stages), replica_tflops=replica_tflops))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', action='store_true', help='Run the performance benchmark')
    parser.add_argument('--round', type=int, choices=range(3), help='Run only the selected independent round')
    parser.add_argument('--output', type=Path, help='Write raw timings and, for the full run, summary metrics')
    args = parser.parse_args()
    if not args.benchmark and args.round is None and args.output is None:
        test_m_grouped_bf16_contiguous_accumulate()
        print("Grouped BF16 accumulation: PASS", flush=True)
        return
    if args.round is not None:
        result = run_round(args.round)
    else:
        rounds = []
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(3):
                path = Path(tmp) / f'round-{i}.json'
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--benchmark',
                                '--round', str(i), '--output', str(path)], check=True)
                rounds.append(json.loads(path.read_text()))
        rows = summarize(rounds)
        result = dict(rounds=rounds, summary=rows, peak_bf16_tflops=PEAK_BF16_TFLOPS, hbm_bandwidth_gbps=HBM_BANDWIDTH_GBPS)
        print(' E  active E   sumT     N     K    mode      us   TFLOPS    MFU(%)    GB/s    MBU(%)  StageC')
        for r in rows:
            print(f"{r['e']:2d} {r['active_e']:9d} {r['sum_t']:6d} {r['n']:5d} {r['k']:5d} {r['mode']:>7s} "
                  f"{r['latency_us']:8.2f} {r['tflops']:8.2f} {r['mfu_pct']:8.2f} {r['effective_gbps']:8.2f} "
                  f"{r['mbu_pct']:8.2f}  {','.join(map(str,r['stage_c']))}")
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
