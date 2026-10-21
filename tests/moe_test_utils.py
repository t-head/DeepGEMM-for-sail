"""Shared runtime helpers and independent numerical references for MoE tests."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import timedelta
import argparse
import json
import os
import statistics
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from deep_gemm.jit_kernels.utils import is_ppu1v5_device


# Per-rank input tokens for production model correctness/performance cases.
MODEL_TOKENS = (1024, 2048, 4096)


def compile_kernels(*calls):
    """Compile and load native kernels serially without launching them."""
    import deep_gemm as dg

    previous = dg.get_compile_mode()
    dg.set_compile_mode(1)
    try:
        for call in calls:
            call()
    finally:
        dg.set_compile_mode(previous)


@dataclass(frozen=True)
class MoECase:
    name: str
    hidden: int
    intermediate: int
    experts: int
    topk: int
    capacity: int
    tokens: tuple[int, ...] = (1, 33)
    activation: str = "swiglu"
    clamp: float = 0.0
    worlds: tuple[int, ...] = (1, 2, 4, 8)
    input_scale: float = 1.0
    reference_tokens: int = 8
    precision: str = "w4a4"
    extra_replicas_per_rank: int = 0
    routing: str = "round_robin"

    def require_world(self):
        world = dist.get_world_size()
        if world not in self.worlds:
            pytest.skip(f"{self.name} requires EP{self.worlds}; launched EP{world}")
        assert self.experts % world == 0


@contextmanager
def moe_process_group(request=None):
    """Shared process-group lifecycle for pytest and directly launched runners."""
    if not torch.cuda.is_available():
        raise RuntimeError("requires a PPU runtime")
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
    if not is_ppu1v5_device():
        message = "MegaMoE requires PPU1.5"
        if request is not None:
            pytest.skip(message)
        raise RuntimeError(message)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    owned = not dist.is_initialized()
    store = None
    if owned:
        kwargs = dict(backend="nccl", timeout=timedelta(seconds=600),
                      device_id=torch.device("cuda", torch.cuda.current_device()))
        if "RANK" not in os.environ:
            store = tempfile.TemporaryDirectory(prefix="deepgemm-moe-test-")
            kwargs.update(init_method=f"file://{store.name}/store", rank=0, world_size=1)
        dist.init_process_group(**kwargs)
    failed = False
    try:
        yield dist.group.WORLD
    except BaseException:
        failed = True
        raise
    finally:
        # A failed rank must exit so torchrun can stop peers waiting in their
        # next collective. Graceful NCCL shutdown can otherwise wait for them.
        distributed_failure = dist.get_world_size() > 1 and (
            failed or (request is not None and request.session.testsfailed))
        if owned and not distributed_failure:
            dist.destroy_process_group()
        if store is not None:
            store.cleanup()
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


@pytest.fixture(scope="session")
def moe_group(request):
    if not torch.cuda.is_available():
        pytest.skip("requires a PPU runtime")
    with moe_process_group(request) as group:
        yield group


def gather(tensor):
    tensors = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors, tensor)
    return torch.stack(tensors)


def activation(projection, kind="swiglu", clamp=0.0):
    """Independent Torch activation; the caller supplies projection precision."""
    gate, up = projection.float().chunk(2, -1)
    if kind == "situ":
        return 4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * 25 * torch.tanh(up / 25)
    if clamp > 0:
        up = up.clamp(-clamp, clamp)
        if kind != "step":
            gate = gate.clamp(max=clamp)
    if kind == "swiglu_oai":
        return gate * torch.sigmoid(1.702 * gate) * (up + 1)
    gate = torch.nn.functional.silu(gate)
    if kind == "step" and clamp > 0:
        gate = gate.clamp(max=clamp)
    return gate * up


def quantize_fp8_rows(tensor):
    values = tensor.float()
    # Use a device FP32 divisor: Torch replaces division by a CPU scalar with
    # reciprocal multiplication, which can cross an E4M3 rounding midpoint.
    # Preserve the kernel's FP32 scale -> reciprocal -> multiply order.
    fp8_max = values.new_tensor(448.0)
    scales = values.abs().amax(-1, keepdim=True).clamp_min(1e-10) / fp8_max
    q = (values * scales.reciprocal()).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scales


def quantize_fp8_blocks(tensor):
    # Keep the independent blockwise oracle on CPU: device Torch's approximate
    # reciprocal can move an E4M3 rounding midpoint to the adjacent FP8 value.
    # This reference runs outside graph capture and the benchmark region.
    shape = tensor.shape
    values = tensor.cpu().float().reshape(*shape[:-1], shape[-1] // 128, 128)
    scales = values.abs().amax(-1).clamp_min(1e-10) / 448
    q = (values / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.reshape(shape).to(tensor.device), scales.to(tensor.device)


def split_expert_weights(weights):
    """Independent allocations expose an ignored or incorrectly offset replica pointer."""
    return tuple(t[:-1].clone() for t in weights), tuple(t[-1:].clone() for t in weights)


def make_routes(case, tokens, rng):
    """Distinct top-k experts for balanced, random or 80%-hot routing."""
    token = torch.arange(tokens, device="cuda")[:, None]
    slot = torch.arange(case.topk, device="cuda")[None, :]
    if case.routing == "round_robin":
        return ((token * case.topk + slot + dist.get_rank() * 17) % case.experts).contiguous()
    if case.routing not in ("random", "skewed"):
        raise ValueError(f"unsupported routing distribution: {case.routing}")
    ids = torch.rand(tokens, case.experts, device="cuda", generator=rng).topk(case.topk, dim=1).indices
    if case.routing == "skewed":
        ids = torch.where(token % 5 != 0, slot.expand_as(ids), ids)
    return ids.contiguous()


def make_inputs(case, requested, rng, edge_cases):
    rank, world = dist.get_rank(), dist.get_world_size()
    count = 0 if edge_cases and world > 1 and requested == 1 and rank == world - 1 else requested
    source = torch.randn(requested, case.hidden, device="cuda", dtype=torch.bfloat16,
                         generator=rng) * (0.5 * case.input_scale)
    ids = make_routes(case, requested, rng)
    weights = torch.rand(requested, case.topk, device="cuda", generator=rng).softmax(-1)
    ids[count:], weights[count:] = -1, 0
    if edge_cases and count > 1:
        # Invalid IDs must mask even a nonzero routing weight.
        ids[0, 0] = -1
        weights[1, 0] = 0
    return source, ids, weights, count


def check_workspace_options(case, group):
    """All failures must occur before collective symmetric allocation."""
    import deep_gemm as dg

    options = dict(num_experts=case.experts, num_max_tokens_per_rank=case.capacity,
                   num_topk=case.topk, hidden=case.hidden, precision=case.precision,
                   intermediate_hidden=case.intermediate)
    invalid = [({"precision": "unsupported"}, "precision must be"),
               ({"intermediate_hidden": 127}, "intermediate_hidden"),
               ({"hidden": case.hidden + 1}, "hidden must be divisible"),
               ({"num_topk": 17}, "Invalid MegaMoE"),
               ({"num_max_tokens_per_rank": 0}, "Invalid MegaMoE")]
    if case.precision.startswith("fp8_"):
        invalid += [({"block_m": 16}, "requires block_m="),
                    ({"block_m": 192}, "requires block_m="),
                    ({"num_experts": 129 * group.size()}, "at most 128 local physical experts")]
        if case.precision == "fp8_blockwise":
            invalid.append(({"block_m": 256}, "blockwise dispatch requires block_m=128"))
    elif case.precision == "w4a16":
        invalid += [({"num_experts": 128 * group.size()}, "fewer than 128 local physical experts")]
    else:
        invalid += [({"block_m": 16}, "set BLOCK_M through overrides")]
    for changed, message in invalid:
        with pytest.raises((ValueError, RuntimeError), match=message):
            dg.get_symm_buffer_for_mega_moe(group, **{**options, **changed})


def check_complete_entry(entry, output, w1, w2, buffer, source, ids, weights, **options):
    """Reject malformed calls before advancing the collective compute protocol."""
    import deep_gemm as dg

    with pytest.raises(RuntimeError, match="call mega_moe_pre_dispatch"):
        entry(output, w1, w2, buffer, **options)
    dg.mega_moe_pre_dispatch(source, ids, weights, buffer)
    with pytest.raises(RuntimeError, match="unsupported precision/activation"):
        entry(output, w1, w2, buffer, **{**options, "activation": "unsupported"})
    unsupported_activation = {"w4a4": "swiglu_oai", "w4a16": "swiglu",
                              "fp8_channelwise": "situ", "fp8_blockwise": "situ"}[buffer.precision]
    with pytest.raises(RuntimeError, match="unsupported precision/activation"):
        entry(output, w1, w2, buffer, **{**options, "activation": unsupported_activation})
    for changed, message in (
        ({"activation_limit": float("nan")}, "activation_limit"),
        ({"activation_alpha": float("nan")}, "activation_alpha"),
        ({"situ_beta": float("nan")}, "SiTU parameters|situ_beta"),
        ({"situ_linear_beta": 0.0}, "SiTU parameters|situ_beta"),
    ):
        with pytest.raises(RuntimeError, match=message):
            entry(output, w1, w2, buffer, **{**options, **changed})
    with pytest.raises(RuntimeError, match="tensor device, dtype or layout"):
        entry(output.float(), w1, w2, buffer, **options)
    bad_w2 = (w2[0][:, :-1].contiguous(), w2[1])
    with pytest.raises(RuntimeError, match="tensor shape"):
        entry(output, w1, bad_w2, buffer, **options)
    other = dg.fp4_fp4_mega_moe if buffer.precision != "w4a4" else dg.bf16_fp4_mega_moe
    with pytest.raises(RuntimeError, match="does not match workspace precision"):
        other(output, w1, w2, buffer)


def run_forward(case, source, requested, pipeline, reference, *, combine, check=True,
                reference_tokens=0, max_diff=.0005, max_relative_l2=None,
                replicas=False, edge_cases=False, benchmark=None, verbose=True,
                routing_weights=None, routing_ids=None, workspace=None):
    """One complete forward for correctness, graph replay and optional end-to-end timing."""
    rank, world, count = dist.get_rank(), dist.get_world_size(), source.shape[0]
    compile_kernels(pipeline)
    block_info = workspace._gemm2_block_info if workspace is not None else None
    if block_info is not None:
        # Scheduler metadata must be regenerated, including after route changes
        # and replay with empty experts; old contents must never be consumed.
        block_info.fill_(-1)
    combine.fill_(float("nan"))
    torch.cuda.synchronize()
    dist.barrier()
    actual = pipeline().clone()
    counts = gather(torch.tensor(count, device="cuda")).tolist()
    report = dict(case=case.name, precision=case.precision, ep=world, tokens_per_rank=counts,
                  hidden=case.hidden, intermediate=case.intermediate, experts=case.experts,
                  capacity=case.capacity, topk=case.topk, activation=case.activation,
                  scatter_dtype="bf16", routing=case.routing, split_weights=replicas,
                  edge_cases=edge_cases, correctness=dict(status="not_checked"))
    if check:
        checked = min(requested, reference_tokens) if reference_tokens else requested
        replay_metrics = []
        def check_reference(result):
            expected = reference(checked)
            assert result.isfinite().all(), (rank, "nonfinite pipeline output")
            sample = result[:min(count, checked)]
            metrics = torch.tensor([float(calc_diff(sample, expected)), relative_l2(sample, expected)],
                                   device="cuda")
            dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
            assert metrics[0].item() < max_diff, (case.name, "pipeline diff", metrics.tolist())
            if max_relative_l2 is not None:
                assert metrics[1].item() < max_relative_l2, (case.name, "pipeline relative L2", metrics.tolist())
            replay_metrics.append(metrics.tolist())
            return metrics

        metrics = check_reference(actual)
        report["correctness"] = dict(status="passed", max_pipeline_diff=metrics[0].item(),
                                     max_relative_l2=metrics[1].item(),
                                     checked_tokens_per_rank=[min(n, checked) for n in counts])
        check_graph_replay(pipeline, actual, input_tensor=source, routing_weights=routing_weights,
                           routing_ids=routing_ids, check_reference=check_reference, combine=combine)
        report["correctness"]["max_replay_pipeline_diff"] = max(m[0] for m in replay_metrics[1:])
        report["correctness"]["max_replay_relative_l2"] = max(m[1] for m in replay_metrics[1:])
        report["correctness"]["changed_routes_reference_checked"] = True
        if routing_weights is not None:
            report["correctness"]["routing_weight_replay_checked"] = True
        if workspace is not None:
            assert workspace._gemm2_block_info is block_info, "GEMM2 replaced workspace scratch"
            report["correctness"]["workspace_reuse_checked"] = True
    if benchmark is not None:
        timing = benchmark_pipeline(pipeline, expected=actual if check else None, **benchmark)
        timing["scope"] = "pre_dispatch_to_combine"
        timing["tokens_per_second"] = sum(counts) * 1e6 / timing["max_rank_us"]["median"]
        report["benchmark"] = timing
    if verbose and rank == 0:
        print(json.dumps(report), flush=True)
    dist.barrier()
    return report


def run_cli(cases, runner, description):
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=[c.name for c in cases], default=cases[0].name)
    parser.add_argument("--tokens", type=int, nargs="+", help="tokens per rank; defaults to the case table")
    parser.add_argument("--capacity", type=int, help="workspace token capacity")
    parser.add_argument("--routing", choices=("round_robin", "random", "skewed"), default="round_robin")
    parser.add_argument("--benchmark", action="store_true", help="also time the complete forward")
    parser.add_argument("--no-check", action="store_true", help="performance only; requires --benchmark")
    parser.add_argument("--reference-tokens", type=int, default=0, help="check first N tokens/rank; 0 checks all")
    parser.add_argument("--mode", choices=("graph", "eager"), default="graph")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--replicas", action="store_true", help="use separate master/replica weight buffers")
    parser.add_argument("--edge-cases", action="store_true", help="include empty ranks and masked/zero-weight routes")
    args = parser.parse_args()
    case = next(c for c in cases if c.name == args.case)
    case = replace(case, tokens=tuple(args.tokens) if args.tokens is not None else case.tokens,
                   capacity=args.capacity if args.capacity is not None else case.capacity, routing=args.routing)
    if case.capacity < 1 or any(n < 0 or n > case.capacity for n in case.tokens):
        parser.error("tokens must be nonnegative and no larger than a positive capacity")
    if args.reference_tokens < 0 or args.warmup < 0 or args.iterations < 1 or args.repeats < 1:
        parser.error("reference-tokens/warmup must be nonnegative; iterations/repeats must be positive")
    if args.no_check and not args.benchmark:
        parser.error("--no-check requires --benchmark")
    with moe_process_group() as group:
        world = dist.get_world_size()
        if world not in case.worlds or case.experts % world:
            raise ValueError(f"{case.name} requires EP{case.worlds}; launched EP{world}")
        if args.replicas and case.experts // world < 2:
            raise ValueError("--replicas requires at least two local experts")
        timing = dict(warmup=args.warmup, iterations=args.iterations,
                      repeats=args.repeats, mode=args.mode) if args.benchmark else None
        runner(case, group, check=not args.no_check, reference_tokens=args.reference_tokens,
               edge_cases=args.edge_cases, splits=(args.replicas,), benchmark=timing)


def relative_l2(actual, expected):
    return ((actual.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1e-8)).item()


def check_graph_replay(pipeline, expected, *, input_tensor=None, routing_weights=None, routing_ids=None,
                       check_reference=None, combine=None):
    """Recycle state with changing data; check eager and replay against a Torch oracle."""
    for _ in range(3):
        pipeline()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = pipeline()

    def compare():
        if combine is not None:
            combine.fill_(float("nan"))
            torch.cuda.synchronize()
            dist.barrier()
        eager = pipeline().clone()
        torch.cuda.synchronize()
        dist.barrier()
        if combine is not None:
            combine.fill_(float("nan"))
            torch.cuda.synchronize()
            dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        assert result.isfinite().all(), "recycled routes read an unproduced slot"
        torch.testing.assert_close(result, eager, rtol=0, atol=0)
        if check_reference is not None:
            check_reference(result)
        dist.barrier()

    if input_tensor is not None:
        saved = input_tensor.clone()
        for factor in (0.0, -1.0, 1.0):
            input_tensor.copy_(saved * factor)
            compare()
    else:
        compare()
    if routing_ids is not None:
        saved_ids = routing_ids.clone()
        hot_routes = torch.arange(saved_ids.shape[1], device=saved_ids.device).expand_as(saved_ids)
        for changed in (hot_routes, torch.full_like(saved_ids, -1), saved_ids):
            routing_ids.copy_(changed)
            compare()
        # Return to original routing before changing weights.
    if routing_weights is not None:
        saved_weights = routing_weights.clone()
        slots = torch.arange(saved_weights.shape[1], device=saved_weights.device)
        # Include tiny, nonuniform and zero routing weights. Do not renormalize:
        # callers may supply weights whose sum is not one.
        factors = torch.where(slots == 0, 0.0, 10.0 ** (-slots.float()))
        for changed in (saved_weights * factors, torch.zeros_like(saved_weights), saved_weights):
            routing_weights.copy_(changed)
            compare()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


def benchmark_pipeline(pipeline, *, warmup=10, iterations=100, repeats=5,
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


def calc_diff(x, y):
    x, y = x.double(), y.double()
    return (x - y).square().sum() / (x.square() + y.square()).sum().clamp_min(1e-30)

def quant_w4a16_mxfp4(y, groupsize=32):
    """Quantize [E, N, K] weights to canonical K-paired E2M1 values and UE8M0 scales."""
    e, n, k = y.shape
    tile = 16
    assert groupsize == 32
    assert k % groupsize == 0
    assert k % tile == 0 and n % 64 == 0
    assert y.dtype in [torch.half, torch.bfloat16]

    w = y.transpose(1, 2).contiguous().float()
    w_grouped = w.view(e, k // groupsize, groupsize, n)
    scale = w_grouped.abs().amax(dim=2) / 6.0
    scale_bits = scale.contiguous().view(torch.int32)
    scale_bits = (scale_bits + 0x007FFFFF) & 0x7F800000
    scale_fp32 = scale_bits.view(torch.float32)
    inv_scale = torch.where(scale_fp32 == 0, torch.zeros_like(scale_fp32), scale_fp32.reciprocal())
    normalized = w_grouped * inv_scale.unsqueeze(2)

    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=y.device, dtype=normalized.dtype)
    magnitude = torch.bucketize(normalized.abs().clamp_max(6.0), boundaries).to(torch.uint8)
    sign = ((normalized < 0) & (magnitude != 0)).to(torch.uint8) << 3
    codes = (magnitude | sign).view(e, k, n)

    e2m1_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=y.device, dtype=torch.float32)
    refs = (e2m1_lut[codes.long()].view(e, k // groupsize, groupsize, n)
            * scale_fp32.unsqueeze(2)).view(e, k, n).transpose(1, 2).to(y.dtype)

    row_codes = codes.transpose(1, 2).contiguous()
    packed = row_codes[..., 0::2] | (row_codes[..., 1::2] << 4)
    raw_scale = (scale_bits >> 23).to(torch.uint8).transpose(1, 2).contiguous()
    return refs, packed, raw_scale

def quantize_fp4_rne(src_tensor):
    src = src_tensor.float()
    *batch, n, k = src.shape
    grouped = src.reshape(*batch,n,k//32,32)
    amax = grouped.abs().amax(-1)
    bits = ((amax / 6.0).contiguous().view(torch.int32) + 0x007fffff) & 0x7f800000
    scale = bits.view(torch.float32)
    normalized = (grouped / scale.unsqueeze(-1)).nan_to_num().reshape_as(src)
    v = normalized.abs()
    thresholds = torch.tensor([0.25,0.75,1.25,1.75,2.5,3.5,5.0],device=src.device)
    code = (v.unsqueeze(-1) > thresholds).sum(-1)
    code += ((v == 0.75) | (v == 1.75) | (v == 3.5)).long()
    code = (code | ((normalized < 0).long() << 3)).to(torch.uint8)
    code = code.reshape(*batch,n,k//2,2)
    return (code[...,0] | (code[...,1]<<4)).contiguous(), (bits>>23).to(torch.uint8).contiguous()

def dequantize_fp4_torch(quant_tensor, scale, target_dtype=torch.float32):
    assert quant_tensor.dtype == torch.uint8 and scale.dtype == torch.uint8
    device = quant_tensor.device
    *batch_dims, N, packed_K = quant_tensor.shape
    K = packed_K * 2
    quant_int = quant_tensor.to(torch.int32)
    evens = quant_int & 0xF
    odds = (quant_int >> 4) & 0xF
    vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    lookup_table = torch.tensor(vals + [-v for v in vals], dtype=torch.float32, device=device)
    fp32_tensor = torch.stack([lookup_table[evens], lookup_table[odds]], dim=-1).view(*batch_dims, N, K)
    dq_scale = (scale.to(torch.int32) << 23).view(torch.float32)
    padded_K = dq_scale.shape[-1] * 32
    padded_tensor = F.pad(fp32_tensor, (0, padded_K - K))
    padded_tensor = padded_tensor.view(*batch_dims, N, dq_scale.shape[-1], 32)
    out_padded = (padded_tensor * dq_scale.unsqueeze(-1)).view(*batch_dims, N, padded_K)
    return out_padded[..., :K].to(target_dtype)
