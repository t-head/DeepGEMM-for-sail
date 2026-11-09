#!/usr/bin/env python3
"""BF16 grouped GEMM + C correctness test and two fixed accumulation benchmarks.

    python tests/test_m_grouped_bf16_add_c.py
    CUDA_VISIBLE_DEVICES=6 python tests/test_m_grouped_bf16_add_c.py --bench --output results.json

Both benchmarks use E=14, sumT=65536: up/gate N=3072 K=3584; down N=3584 K=3072.
Groups have 128-aligned lengths: 8*4736 + 6*4608 = 65536, without padding.
Compare the automatic no-C and C=D paths. CUDA Graph timing uses warm caches
and repeated buffers; excludes JIT, reference checks and D resets. TFLOPS is
2*sumT*N*K/time (no extra E factor or C-add FLOPs). StageC and RegC report the actual dispatch.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
import torch
import deep_gemm

CASES = (("up_gate_acc", 3072, 3584), ("down_acc", 3584, 3072))
COUNTS = [4736] * 8 + [4608] * 6
MODES = ("no_c", "c_eq_d")


def inputs(counts, n, k, seed):
    torch.manual_seed(seed)
    # Integer/8 values give exact FP32 products/partial sums at these K values.
    def values(shape):
        return torch.randint(-4, 5, shape, device="cuda", dtype=torch.int32).to(torch.bfloat16) / 8
    a, b = values((sum(counts), k)), values((len(counts), n, k))
    initial = values((sum(counts), n))
    indices = torch.repeat_interleave(torch.arange(len(counts), dtype=torch.int32), torch.tensor(counts)).cuda()
    return a, b, initial, indices


def reference(a, b, counts):
    product = torch.empty((a.shape[0], b.shape[1]), dtype=torch.float32, device="cuda")
    start = 0
    columns = [0, b.shape[1] // 2, b.shape[1] - 1]
    for group, count in enumerate(counts):
        product[start:start + count] = a[start:start + count].float() @ b[group].float().T
        # Independent CPU anchors include the first, middle and last row of every expert.
        rows = [start, start + count // 2, start + count - 1]
        anchor = a[rows].float().cpu() @ b[group, columns].float().cpu().T
        check(product[rows][:, columns].cpu(), anchor)
        start += count
    return product


def check(actual, expected):
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_m_grouped_bf16_add_c():
    """Mode/cache switching, exact in-place aliasing, rounding and invalid C."""
    if not torch.cuda.is_available():
        raise RuntimeError("A PPU exposed through torch.cuda is required")
    fn = deep_gemm.m_grouped_bf16_gemm_nt_contiguous
    counts = [128, 256, 128]
    a, b, c, indices = inputs(counts, 256, 384, 20261106)
    product, d = reference(a, b, counts), torch.empty_like(c)
    original_c = c.clone()
    for mode in ("no_c", "separate_c", "c_eq_d", "none", "separate_c"):
        d.copy_(c)
        expected = c.clone()
        for _ in range(2):
            if mode == "no_c":
                fn(a, b, d, indices)
            else:
                fn(a, b, d, indices, c={"none": None, "separate_c": c, "c_eq_d": d}[mode])
            expected = (product + (expected.float() if mode == "c_eq_d" else c.float())
                        if mode in ("c_eq_d", "separate_c") else product).to(torch.bfloat16)
            check(d, expected)
            check(c, original_c)
    before = d.clone()
    try:
        fn(a, b, d, indices, c=c.float())
    except RuntimeError:
        pass
    else:
        raise AssertionError("FP32 C must be rejected")
    check(d, before)
    # AB=1+1/256, C=1/256: adding after BF16 rounding would incorrectly return 1.
    a = torch.zeros((128, 128), dtype=torch.bfloat16, device="cuda")
    a[:, 0], a[:, 1] = 1, 1 / 256
    b = torch.ones((1, 128, 128), dtype=torch.bfloat16, device="cuda")
    d = torch.full_like(a, 1 / 256)
    fn(a, b, d, torch.zeros(128, dtype=torch.int32, device="cuda"), c=d)
    check(d, torch.full_like(d, 1 + 1 / 128))
    check_c_access_edges()
    torch.cuda.synchronize()


def check_c_access_edges():
    """Vector C loads with N tails and unaligned scalar C/D, including C=D."""
    counts = [128, 256, 128]
    config = (torch.cuda.get_device_properties(0).multi_processor_count,
              128, 256, 64, 64, 64, 2, (0, 0, 0))
    for n, offset in ((256, 0), (264, 0), (264, 1)):
        a, b, initial, indices = inputs(counts, n, 192, 20261107)
        product = reference(a, b, counts)
        start = 16 + offset  # One BF16 offset forces the scalar fallback.
        def guarded():
            storage = torch.full((start + initial.numel() + 16,), 17,
                                 dtype=torch.bfloat16, device="cuda")
            view = storage[start:-16].view_as(initial)
            view.copy_(initial)
            assert view.is_contiguous() and view.data_ptr() % 16 == 2 * offset
            return storage, view
        for in_place in (False, True):
            c_storage, c = guarded()
            d_storage, d = guarded()
            def op():
                deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                    a, b, d, indices, configs=config, c=d if in_place else c)
            info = dispatch(op)
            assert info["with_c"], info
            assert not info["stage_c"], info
            if offset or "M890P" not in torch.cuda.get_device_name():
                assert not info["reg_c"], info
            elif n == config[2]:
                assert info["reg_c"], info
            assert info["tile"] == [128, 256, 64], info
            expected = initial
            for call in range(3):
                if call:
                    op()
                expected = (product + (expected.float() if in_place else initial.float())).to(torch.bfloat16)
                check(d, expected)
                check(c, initial)
            graph = graph_for(op, d, initial, 3)
            check(d, expected)
            for _ in range(2):
                d.copy_(initial)
                graph.replay()
                torch.cuda.synchronize()
                check(d, expected)
                check(c, initial)
            for storage in (c_storage, d_storage):
                check(storage[:start], torch.full_like(storage[:start], 17))
                check(storage[-16:], torch.full_like(storage[-16:], 17))


def dispatch(op):
    """Capture an actual untimed launch, including C accumulation and tile selection."""
    libc = ctypes.CDLL(None)
    sys.stdout.flush()
    libc.fflush(None)
    previous, saved = os.environ.get("show_log"), os.dup(1)
    with tempfile.TemporaryFile() as log:
        try:
            os.dup2(log.fileno(), 1)
            os.environ["show_log"] = "1"
            op()
            torch.cuda.synchronize()
        finally:
            libc.fflush(None)
            os.dup2(saved, 1)
            os.close(saved)
            if previous is None:
                os.environ.pop("show_log", None)
            else:
                os.environ["show_log"] = previous
        log.seek(0)
        raw = log.read().decode(errors="replace")
    flags = re.search(r"WithC:(\d+), StageC:(\d+), RegC:(\d+)", raw)
    tile = re.search(r"ThreadblockShape\[(\d+), (\d+), (\d+)\]", raw)
    if flags is None or tile is None:
        raise RuntimeError(f"Missing actual dispatch information:\n{raw}")
    return dict(with_c=bool(int(flags[1])), stage_c=bool(int(flags[2])), reg_c=bool(int(flags[3])),
                tile=list(map(int, tile.groups())), raw_log=raw)


def graph_for(op, d, initial, calls):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        d.copy_(initial)
        for _ in range(5):
            op()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            op()
    torch.cuda.synchronize()
    for _ in range(2):
        d.copy_(initial)
        graph.replay()
    torch.cuda.synchronize()
    return graph


def benchmark(name, n, k, index, args, source_id):
    a, b, initial, indices = inputs(COUNTS, n, k, 20261106 + index)
    product = reference(a, b, COUNTS)
    outputs = {mode: torch.empty_like(initial) for mode in MODES}
    graphs, configs = {}, {}
    for mode in MODES:
        d = outputs[mode]
        kwargs = {"c": d} if mode == "c_eq_d" else {}
        def op(d=d, kwargs=kwargs):
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, indices, **kwargs)
        d.copy_(initial)
        configs[mode] = dispatch(op)
        assert configs[mode]["with_c"] == (mode == "c_eq_d")
        check(d, (product + initial.float() if mode == "c_eq_d" else product).to(torch.bfloat16))
        graphs[mode] = graph_for(op, d, initial, args.calls)
    samples = {mode: [] for mode in MODES}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(); end.record(); end.synchronize()
    for batch in range(args.batches):
        # Alternate mode order, with the D reset outside the measured interval.
        for mode in (MODES[::-1] if (batch + index) % 2 else MODES):
            outputs[mode].copy_(initial)
            start.record()
            graphs[mode].replay()
            end.record()
            end.synchronize()
            samples[mode].append(start.elapsed_time(end) * 1000 / args.calls)
    rows = []
    for mode in MODES:
        expected = product.to(torch.bfloat16)
        if mode == "c_eq_d":
            expected = initial
            for _ in range(args.calls):
                expected = (product + expected.float()).to(torch.bfloat16)
        check(outputs[mode], expected)  # Full output after the final graph replay.
        us = statistics.median(samples[mode])
        row = dict(case=name, gpu=torch.cuda.get_device_name(), dtype="bf16",
                   shape=dict(e=14, sum_t=sum(COUNTS), n=n, k=k, group_m=COUNTS),
                   mode=mode, metric="latency_us", value=us, source_id=source_id,
                   tflops=2 * sum(COUNTS) * n * k / (us * 1e6),
                   stage_c=configs[mode]["stage_c"], reg_c=configs[mode]["reg_c"],
                   dispatch=configs[mode], samples_us=samples[mode],
                   cv_percent=100 * statistics.pstdev(samples[mode]) / statistics.mean(samples[mode]),
                   validation="full-output bit-exact before/after graph; CPU anchors per expert")
        rows.append(row)
        print(f"{name:12s} {mode:6s} E={len(COUNTS)} sumT={sum(COUNTS)} N={n} K={k} "
              f"{us:.3f} us  {row['tflops']:.3f} TFLOPS  StageC={int(row['stage_c'])} "
              f"RegC={int(row['reg_c'])} CV={row['cv_percent']:.2f}%", flush=True)
    print(f"  C=D latency overhead: {(rows[1]['value']/rows[0]['value']-1)*100:+.2f}%", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", action="store_true", help="run both requested shapes after correctness checks")
    parser.add_argument("--calls", type=int, default=20, help="GEMMs per CUDA Graph")
    parser.add_argument("--batches", type=int, default=11, help="paired timing batches; median is reported")
    parser.add_argument("--output", type=Path, help="write benchmark metadata and raw samples as JSON")
    args = parser.parse_args()
    if args.calls <= 0 or args.batches < 3:
        parser.error("--calls must be positive and --batches must be at least 3")
    if args.output and (not args.bench or args.output.exists()):
        parser.error("--output requires --bench and a new output filename")
    torch.set_num_threads(1)
    test_m_grouped_bf16_add_c()
    print("Correctness: PASS", flush=True)
    if args.bench:
        digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        native = Path(deep_gemm._C.__file__).resolve()
        data = dict(metadata=dict(source_id=f"{revision}+test-{digest[:12]}", script_sha256=digest,
                                  native=str(native), native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                                  torch_version=torch.__version__, physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                  calls=args.calls, batches=args.batches, warmup=5, seed=20261106,
                                  timing="warm-cache CUDA Graph; alternating mode order; resets excluded",
                                  flops="2*sumT*N*K; no extra E factor or C addition"), results=[])
        for index, (name, n, k) in enumerate(CASES):
            data["results"].extend(benchmark(name, n, k, index, args, data["metadata"]["source_id"]))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
