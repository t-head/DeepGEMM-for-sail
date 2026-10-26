r"""PPU W4A4 MegaMoE complete-call benchmark and optional DeepEP baselines.

Prefill, including a DeepEP normal comparison (eager timing):
    torchrun --standalone --nproc-per-node=8 tests/bench_mega_moe.py \
        --case deepseek_v4_pro --num-tokens 1024 2048 4096 --routing random \
        --reference-tokens 8 --baseline normal

Decode, including a DeepEP low-latency comparison (graph timing):
    torchrun --standalone --nproc-per-node=8 tests/bench_mega_moe.py \
        --case deepseek_v4_pro --num-tokens 32 64 128 --routing random \
        --reference-tokens 8 --baseline low_latency

Both pipelines start at BF16 input and include input preparation, quantization,
communication, expert computation, combine and required state recycling. Setup,
weight transforms, JIT and numerical references are outside timing. DeepEP and
Triton are imported only for a requested baseline; missing dependencies skip
that comparison on every rank. Installed but broken backends fail explicitly.
"""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import sys

import torch
import torch.distributed as dist

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deep_gemm
from deep_gemm.testing import calc_diff
from test_mega_moe import (
    CASES, MAX_DIFF, _check_mega_moe,
    _make_routes, _make_weights, _process_group, _reference_mega_moe,
    _relative_l2, _transform_weights,
)


def _import_baseline():
    # The JIT functions below use module globals so Triton can resolve their
    # annotations and dependencies, while remaining optional for MegaMoE-only runs.
    global triton, tl, _quantize_input, _permute, _weighted_gather
    error = None
    buffer_type = None
    try:
        import triton
        import triton.language as tl
        from deep_ep import Buffer
        buffer_type = Buffer
    except ModuleNotFoundError as exc:
        missing = exc.name in ("triton", "triton.language", "deep_ep")
        error = ("missing" if missing else "error", str(exc))
    except Exception as exc:
        error = ("error", f"{type(exc).__name__}: {exc}")

    # A dependency missing on one rank must not leave its peers entering DeepEP.
    statuses = [None] * dist.get_world_size()
    dist.all_gather_object(statuses, error)
    failures = [f"rank {rank}: {item[1]}" for rank, item in enumerate(statuses)
                if item is not None and item[0] == "error"]
    if failures:
        raise RuntimeError("DeepEP baseline import failed: " + "; ".join(failures))
    missing = [f"rank {rank}: {item[1]}" for rank, item in enumerate(statuses)
               if item is not None]
    if missing:
        return None, "; ".join(missing)

    @triton.jit
    def _quantize_input(x, values, scales, H: tl.constexpr, B: tl.constexpr):
        row, tile = tl.program_id(0), tl.program_id(1)
        columns = tile * B + tl.arange(0, B)
        data = tl.load(x + row * H + columns, columns < H, 0).to(tl.float32)
        data = data.reshape(B // 32, 32)
        amax = tl.maximum(tl.max(tl.abs(data), 1), 1e-10)
        bits = ((amax / 6.0).to(tl.int32, bitcast=True) + 0x7fffff) & 0x7f800000
        normalized = data / bits.to(tl.float32, bitcast=True)[:, None]
        magnitude = tl.abs(normalized)
        code = ((magnitude > .25).to(tl.int32) + (magnitude >= .75).to(tl.int32)
                + (magnitude > 1.25).to(tl.int32) + (magnitude >= 1.75).to(tl.int32)
                + (magnitude > 2.5).to(tl.int32) + (magnitude >= 3.5).to(tl.int32)
                + (magnitude > 5.0).to(tl.int32))
        code |= (normalized < 0).to(tl.int32) << 3
        packed = tl.sum(code.reshape(B // 2, 2) << (4 * tl.arange(0, 2)[None, :]), 1)
        q_columns = tile * (B // 2) + tl.arange(0, B // 2)
        s_columns = tile * (B // 32) + tl.arange(0, B // 32)
        tl.store(values + row * (H // 2) + q_columns, packed, q_columns < H // 2)
        tl.store(scales + row * (H // 32) + s_columns, bits >> 23, s_columns < H // 32)
    @triton.jit
    def _permute(values, scales, ids, cursors, grouped, grouped_scales, group_ids, row_map,
                 H: tl.constexpr, K: tl.constexpr, E: tl.constexpr, SLOTS: tl.constexpr,
                 B: tl.constexpr):
        token = tl.program_id(0)
        slot = tl.arange(0, SLOTS)
        expert = tl.load(ids + token * K + slot, slot < K, -1)
        valid = (slot < K) & (expert >= 0) & (expert < E)
        # Widen before row-stride multiplication; grouped FP4 can exceed 2 GiB.
        destination = tl.atomic_add(cursors + expert, 1, mask=valid).to(tl.int64)
        tl.store(row_map + token * K + slot, destination, valid)
        tl.store(group_ids + destination, expert, valid)
        for begin in range(tl.cdiv(H // 2, B)):
            column = begin * B + tl.arange(0, B)
            packed = tl.load(values + token * (H // 2) + column, column < H // 2, 0)
            tl.store(grouped + destination[:, None] * (H // 2) + column[None, :],
                     packed[None, :], valid[:, None] & (column[None, :] < H // 2))
        for begin in range(tl.cdiv(H // 32, B)):
            column = begin * B + tl.arange(0, B)
            scale = tl.load(scales + token * (H // 32) + column, column < H // 32, 0)
            tl.store(grouped_scales + destination[:, None] * (H // 32) + column[None, :],
                     scale[None, :], valid[:, None] & (column[None, :] < H // 32))
    @triton.jit
    def _weighted_gather(expert_output, ids, weights, row_map, output,
                         H: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
        token = tl.program_id(0)
        column = tl.program_id(1) * B + tl.arange(0, B)
        result = tl.full((B,), 0, tl.float32)
        for slot in range(K):
            expert = tl.load(ids + token * K + slot)
            weight = tl.load(weights + token * K + slot)
            active = (expert >= 0) & (weight != 0)
            row = tl.load(row_map + token * K + slot, active, 0)
            value = tl.load(expert_output + row * H + column, active & (column < H), 0).to(tl.float32)
            result += weight * value
        tl.store(output + token * H + column, result, column < H)

    return buffer_type, None


class _DeepEPBaseline:
    def __init__(self, buffer_type, case, group, l1_weights, l2_weights, kind):
        if kind not in ("normal", "low_latency"):
            raise ValueError(f"unsupported DeepEP baseline: {kind}")
        if case.activation not in ("swiglu", "step", "situ"):
            raise ValueError(f"unsupported DeepEP activation: {case.activation}")
        self.case, self.kind = case, kind
        self.world, self.local_experts = group.size(), case.experts // group.size()
        self.warming_up = False
        self.fused_activation = case.activation in ("swiglu", "step")
        mode = deep_gemm.PreprocessMode.MXFP4
        self.l1_weights = deep_gemm.preprocess_weights_and_scales(
            mode, self.fused_activation, weight=l1_weights[0], weight_scale=l1_weights[1])
        self.l2_weights = deep_gemm.preprocess_weights_and_scales(
            mode, False, weight=l2_weights[0], weight_scale=l2_weights[1])
        if kind == "normal":
            self.dispatch_config = buffer_type.get_intranode_dispatch_config(
                self.world, case.hidden // 2, "quant")
            self.combine_config = buffer_type.get_intranode_combine_config(self.world, case.hidden)
            configs = self.dispatch_config, self.combine_config
            self.buffer = buffer_type(
                group,
                num_nvl_bytes=max(c.get_nvl_buffer_size_hint(case.hidden * 2, self.world) for c in configs),
                num_rdma_bytes=max(c.get_rdma_buffer_size_hint(case.hidden * 2, self.world) for c in configs),
                allow_mnnvl=True, explicitly_destroy=True)
        else:
            self.capacity = max(1, max(case.tokens))
            self.buffer = buffer_type(
                group, num_rdma_bytes=buffer_type.get_low_latency_rdma_size_hint(
                    self.capacity, case.hidden, self.world, case.experts),
                low_latency_mode=True, num_qps_per_rank=self.local_experts,
                allow_mnnvl=True, allow_nvlink_for_low_latency_mode=True, explicitly_destroy=True)
            rows = self.world * self.capacity
            shape = self.local_experts, rows
            self.down = torch.empty(*shape, case.hidden, device="cuda", dtype=torch.bfloat16)
            if self.fused_activation:
                self.activated = torch.empty(*shape, case.intermediate // 2, device="cuda", dtype=torch.uint8)
                self.scales = torch.empty(self.local_experts, case.intermediate // 64, rows,
                                          device="cuda", dtype=torch.uint16).transpose(1, 2)
            else:
                self.raw = torch.empty(*shape, 2 * case.intermediate, device="cuda", dtype=torch.bfloat16)

    def _warmup_barrier(self):
        # Finish first-use local JIT before entering a peer-dependent operation.
        # These barriers run only during setup, outside correctness replay/timing.
        if self.warming_up:
            torch.cuda.synchronize()
            dist.barrier()

    def warmup(self, x, ids, weights):
        self.warming_up = True
        try:
            return self(x, ids, weights)
        finally:
            self.warming_up = False

    def __call__(self, x, ids, weights):
        if self.kind == "normal":
            return self._normal(x, ids, weights)
        return self._low_latency(x, ids, weights)

    def _gemm1(self, lhs, counts, group_ids, raw, activated, scales, expected_m):
        gemm = (deep_gemm.m_grouped_fp4_gemm_nt_masked if self.kind == "low_latency"
                else deep_gemm.m_grouped_fp4_gemm_nt_nopad)
        args = (counts, expected_m) if self.kind == "low_latency" else (group_ids, counts)
        if self.fused_activation:
            gemm(lhs, self.l1_weights, None, activated, *args, out_scale=scales,
                 swiglu_limit=self.case.clamp or None,
                 activation_type="silu_and_mul_step" if self.case.activation == "step" else "silu_and_mul")
            return activated, scales
        # Ordinary grouped FP4 GEMM has no fused SiTU epilogue. Include the
        # standalone SiTU + MXFP4 quantization kernel in the complete baseline.
        gemm(lhs, self.l1_weights, None, raw, *args)
        if self.kind == "low_latency":
            return deep_gemm.activation_and_mul_masked(
                deep_gemm.ActivationMode.SITU, deep_gemm.QuantMode.MXFP4,
                raw, counts, expected_m=expected_m, beta=4., linear_beta=25.)
        return deep_gemm.activation_and_mul(
            deep_gemm.ActivationMode.SITU, deep_gemm.QuantMode.MXFP4,
            raw, beta=4., linear_beta=25.)

    def _normal(self, x, ids, weights):
        c = self.case
        quantized = torch.empty(x.shape[0], c.hidden // 2, device=x.device, dtype=torch.uint8)
        scales = torch.empty(x.shape[0], c.hidden // 32, device=x.device, dtype=torch.uint8)
        if x.shape[0]:
            _quantize_input[(x.shape[0], triton.cdiv(c.hidden, 512))](x, quantized, scales, c.hidden, 512)
        per_rank, per_rdma, per_expert, in_rank, _ = self.buffer.get_dispatch_layout(ids, c.experts)
        self._warmup_barrier()
        (recv, recv_scales), recv_ids, recv_weights, counts, handle, _ = self.buffer.dispatch(
            (quantized, scales), topk_idx=ids, topk_weights=weights,
            num_tokens_per_rank=per_rank, num_tokens_per_rdma_rank=per_rdma,
            num_tokens_per_expert=per_expert, is_token_in_rank=in_rank,
            expert_alignment=1, config=self.dispatch_config)
        rows = sum(counts)
        counts = torch.tensor(counts, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cursors = counts.cumsum(0, dtype=torch.int32) - counts
        grouped = torch.empty(rows, c.hidden // 2, device=x.device, dtype=torch.uint8)
        grouped_scales = torch.empty(rows, c.hidden // 32, device=x.device, dtype=torch.uint8)
        group_ids = torch.empty(rows, device=x.device, dtype=torch.int32)
        row_map = torch.empty_like(recv_ids)
        if recv.shape[0]:
            _permute[(recv.shape[0],)](
                recv, recv_scales, recv_ids, cursors, grouped, grouped_scales, group_ids, row_map,
                c.hidden, c.topk, self.local_experts, triton.next_power_of_2(c.topk), 128)
        down = torch.empty(rows, c.hidden, device=x.device, dtype=torch.bfloat16)
        if rows:
            _, grouped_scales = deep_gemm.preprocess_weights_and_scales(
                deep_gemm.PreprocessMode.MXFP4, False, weight_scale=grouped_scales)
            lhs = grouped, grouped_scales
            raw = None if self.fused_activation else torch.empty(
                rows, 2 * c.intermediate, device=x.device, dtype=torch.bfloat16)
            activated = torch.empty(rows, c.intermediate // 2, device=x.device, dtype=torch.uint8)
            act_scales = torch.empty(rows, c.intermediate // 64, device=x.device, dtype=torch.uint16)
            act = self._gemm1(lhs, counts, group_ids, raw, activated, act_scales, None)
            deep_gemm.m_grouped_fp4_gemm_nt_nopad(act, self.l2_weights, None, down, group_ids, counts)
        gathered = torch.empty(recv.shape[0], c.hidden, device=x.device, dtype=torch.bfloat16)
        if recv.shape[0]:
            _weighted_gather[(recv.shape[0], triton.cdiv(c.hidden, 512))](
                down, recv_ids, recv_weights, row_map, gathered, c.hidden, c.topk, 512)
        self._warmup_barrier()
        return self.buffer.combine(gathered, handle, config=self.combine_config)[0]

    def _low_latency(self, x, ids, weights):
        c = self.case
        self._warmup_barrier()
        lhs, counts, handle, _, _ = self.buffer.low_latency_dispatch(
            x, ids, self.capacity, c.experts, use_mxfp4=True, quant_size=32,
            mxfp4_scale_row_major=False)
        expected_m = max(1, math.ceil(x.shape[0] * c.topk / self.local_experts))
        act = self._gemm1(lhs, counts, None, getattr(self, "raw", None),
                          getattr(self, "activated", None), getattr(self, "scales", None), expected_m)
        deep_gemm.m_grouped_fp4_gemm_nt_masked(act, self.l2_weights, None, self.down, counts, expected_m)
        self._warmup_barrier()
        return self.buffer.low_latency_combine(self.down, ids, weights, handle)[0]

    def destroy(self):
        self.buffer.destroy()


def _benchmark_pipeline(pipeline, *, warmup=10, iterations=100, repeats=5,
                       mode="graph", expected=None):
    """Time the entire pipeline, using identical call counts on every rank.

    Each sample is the slowest rank's mean latency over `iterations` forwards.
    JIT, reference checks and host barriers are outside the timed region. Graph
    mode captures the whole batch, avoiding gaps between Python replay calls.
    """
    if warmup < 0 or iterations < 1 or repeats < 1 or mode not in ("graph", "eager"):
        raise ValueError("invalid benchmark mode or iteration counts")
    for _ in range(warmup):
        pipeline()
    torch.cuda.synchronize()
    dist.barrier()
    graph = None
    if mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(iterations):
                result = pipeline()
        graph.replay()
        torch.cuda.synchronize()
        dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        dist.barrier()
        start.record()
        if graph is not None:
            graph.replay()
        else:
            for _ in range(iterations):
                result = pipeline()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
        if expected is not None:
            torch.testing.assert_close(result, expected, rtol=0, atol=0)

    times = torch.tensor(samples, dtype=torch.float64, device="cuda")
    dist.all_reduce(times, op=dist.ReduceOp.MAX)
    samples = times.tolist()
    return dict(mode=mode, warmup=warmup, iterations=iterations, repeats=repeats,
                max_rank_us=dict(median=statistics.median(samples),
                                 minimum=min(samples), maximum=max(samples)),
                samples_max_rank_us=samples)


def _check_baseline(actual, expected):
    sample = actual[:expected.shape[0]]
    metrics = torch.tensor([float(calc_diff(sample, expected)), _relative_l2(sample, expected)],
                           device=actual.device)
    finite = torch.tensor(int(actual.isfinite().all()), device=actual.device)
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    assert finite.item(), "nonfinite DeepEP baseline output"
    assert metrics[0].item() < MAX_DIFF, ("DeepEP baseline diff", metrics.tolist())
    return dict(status="passed", max_pipeline_diff=metrics[0].item(),
                max_relative_l2=metrics[1].item())


@torch.inference_mode()
def _benchmark_case(case, group, args, buffer_type, skip_reason):
    rank, world = dist.get_rank(), dist.get_world_size()
    rng = torch.Generator(device="cuda").manual_seed(args.seed + rank)
    # Match the correctness runner's generation order, including workspace capacity.
    source = torch.randn(case.capacity, case.hidden, device="cuda", dtype=torch.bfloat16,
                         generator=rng) * .5
    w1_raw = _make_weights(case.experts // world, 2 * case.intermediate, case.hidden, .5, rng)
    w2_raw = _make_weights(case.experts // world, case.hidden, case.intermediate, .125, rng)
    w1, w2, replica_options = _transform_weights(w1_raw, w2_raw, case.activation, args.replicas)
    baseline = None
    workspace = deep_gemm.SymmBuffer(
        group, case.experts, case.capacity, case.topk, case.hidden, case.intermediate,
        activation=case.activation)
    try:
        if buffer_type is not None:
            baseline = _DeepEPBaseline(buffer_type, case, group, w1_raw, w2_raw, args.baseline)
        for requested in case.tokens:
            empty_rank = args.edge_cases and world > 1 and requested == 1 and rank == world - 1
            tokens = 0 if empty_rank else requested
            ids = _make_routes(case, case.capacity, rng)
            weights = torch.rand(case.capacity, case.topk, device="cuda", generator=rng).softmax(-1)
            ids[tokens:], weights[tokens:] = -1, 0
            if args.edge_cases and tokens > 1:
                ids[0, 0] = -1
                weights[1, 0] = 0
            x, routed_ids, routed_weights = source[:tokens], ids[:tokens], weights[:tokens]
            y = torch.empty_like(x)

            def run_mega():
                deep_gemm.mega_moe_pre_dispatch(x, routed_ids, routed_weights, workspace)
                deep_gemm.fp4_fp4_mega_moe(
                    y, w1, w2, workspace, activation=case.activation,
                    activation_clamp=case.clamp, fast_math=bool(args.fast_math), **replica_options)
                return y

            def reference(checked):
                return _reference_mega_moe(case, source, ids, weights, w1_raw, w2_raw,
                                          tokens, checked)

            report = _check_mega_moe(
                case, x, requested, run_mega, reference,
                workspace=workspace,
                check=not args.no_check, reference_tokens=args.reference_tokens,
                replicas=args.replicas, edge_cases=args.edge_cases, verbose=False,
                routing_weights=routed_weights, routing_ids=routed_ids,
                high_precision_reference=lambda checked: _reference_mega_moe(
                    case, source, ids, weights, w1_raw, w2_raw, tokens, checked,
                    quantization_aware=False))
            mega_expected = run_mega().clone() if not args.no_check else None
            baseline_expected = None
            baseline_report = dict(kind=args.baseline, status="not_requested")
            if skip_reason is not None:
                baseline_report.update(status="skipped", reason=skip_reason)
            if baseline is not None:
                baseline.warmup(x, routed_ids, routed_weights)
                baseline_expected = baseline(x, routed_ids, routed_weights).clone()
                baseline_report.update(
                    status="completed", activation=case.activation,
                    activation_impl="gemm1_epilogue" if baseline.fused_activation else "separate_situ_and_mul",
                    correctness=dict(status="not_checked"))
                if args.baseline == "low_latency":
                    baseline_report["num_max_tokens_per_rank"] = baseline.capacity
                if not args.no_check:
                    checked = min(requested, args.reference_tokens) if args.reference_tokens else requested
                    baseline_report["correctness"] = _check_baseline(baseline_expected, reference(checked))

            timing_options = dict(warmup=args.warmup, iterations=args.iterations,
                                  repeats=args.repeats, mode=args.mode)
            timing = _benchmark_pipeline(run_mega, expected=mega_expected, **timing_options)
            timing["scope"] = "bf16_input_to_bf16_output"
            timing["tokens_per_second"] = sum(report["tokens_per_rank"]) * 1e6 / timing["max_rank_us"]["median"]
            report.update(benchmark=timing, seed=args.seed, fast_math=bool(args.fast_math))
            if baseline is not None:
                def run_baseline():
                    return baseline(x, routed_ids, routed_weights)

                baseline_timing = _benchmark_pipeline(
                    run_baseline, expected=baseline_expected if not args.no_check else None,
                    **timing_options)
                baseline_timing["scope"] = timing["scope"]
                baseline_report["benchmark"] = baseline_timing
                baseline_report["speedup"] = (baseline_timing["max_rank_us"]["median"]
                                                / timing["max_rank_us"]["median"])
            report["baseline"] = baseline_report
            if rank == 0:
                print(json.dumps(report), flush=True)
    finally:
        if baseline is not None:
            baseline.destroy()
        workspace.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=[c.name for c in CASES], default=CASES[0].name)
    parser.add_argument("--num-tokens", "--tokens", dest="tokens", type=int, nargs="+")
    parser.add_argument("--num-max-tokens-per-rank", "--capacity", dest="capacity", type=int)
    parser.add_argument("--routing", choices=("round_robin", "random", "skewed"), default="round_robin")
    parser.add_argument("--fast-math", type=int, choices=(0, 1), default=1)
    parser.add_argument("--reference-tokens", type=int, default=0,
                        help="check first N tokens/rank; 0 checks all")
    parser.add_argument("--no-check", action="store_true", help="skip numerical reference checks")
    parser.add_argument("--baseline", choices=("none", "normal", "low_latency"), default="none")
    parser.add_argument("--mode", choices=("graph", "eager"),
                        help="default: eager for normal baseline, graph otherwise")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--replicas", action="store_true", help="test separately allocated replica weights")
    parser.add_argument("--edge-cases", action="store_true",
                        help="also check empty ranks and masked routes before timing")
    args = parser.parse_args(argv)
    case = next(c for c in CASES if c.name == args.case)
    case = replace(case, tokens=tuple(args.tokens) if args.tokens is not None else case.tokens,
                   capacity=args.capacity if args.capacity is not None else case.capacity, routing=args.routing)
    if case.capacity < 1 or any(n < 0 or n > case.capacity for n in case.tokens):
        parser.error("tokens must be nonnegative and no larger than a positive capacity")
    if args.reference_tokens < 0 or args.warmup < 0 or args.iterations < 1 or args.repeats < 1:
        parser.error("reference-tokens/warmup must be nonnegative; iterations/repeats must be positive")
    args.mode = args.mode or ("eager" if args.baseline == "normal" else "graph")
    if args.baseline == "normal" and args.mode != "eager":
        parser.error("DeepEP normal requires eager timing because dispatch returns CPU expert counts; "
                     "both paths must use the same timing mode")
    with _process_group() as group:
        world = dist.get_world_size()
        if world not in case.worlds or case.experts % world:
            raise ValueError(f"{case.name} requires EP{case.worlds}; launched EP{world}")
        if args.replicas and case.experts // world < 2:
            raise ValueError("--replicas requires at least two local experts")
        buffer_type, skip_reason = _import_baseline() if args.baseline != "none" else (None, None)
        if skip_reason is not None and dist.get_rank() == 0:
            print("DeepEP baseline skipped: " + skip_reason, file=sys.stderr, flush=True)
        _benchmark_case(case, group, args, buffer_type, skip_reason)


if __name__ == "__main__":
    main()
