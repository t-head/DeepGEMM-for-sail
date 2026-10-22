"""Shared runtime helpers and independent numerical references for MoE tests."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import timedelta
import argparse
import json
import os
import statistics
import tempfile

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F


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
    scatter_tile: tuple[int, int, int, int, int, int] | None = None
    precision: str = "w4a4"
    extra_replicas_per_rank: int = 0

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


class CombineBarrier:
    """One persistent generation counter per rank and one local completion flag.

    Construct collectively outside graph capture. Reuse on one stream, with
    the same number of mega_moe_combine calls on every rank, including empty ones.
    """

    def __init__(self, group):
        import torch.distributed._symmetric_memory as symm_mem

        world = dist.get_world_size(group)
        self.flag = symm_mem.empty(1, dtype=torch.int32, device="cuda")
        self.handle = symm_mem.rendezvous(self.flag, group=group)
        self.pointers = torch.tensor([
            self.handle.get_buffer(rank, (1,), torch.int32).data_ptr()
            for rank in range(world)
        ], dtype=torch.int64, device="cuda")
        self.done = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.flag.zero_()
        torch.cuda.synchronize()
        dist.barrier(group=group)


def gather(tensor):
    tensors = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors, tensor)
    return torch.stack(tensors)


def interleave_gate_up(tensor):
    experts, n, *rest = tensor.shape
    return torch.stack((tensor[:, :n // 2].reshape(experts, -1, 8, *rest),
                        tensor[:, n // 2:].reshape(experts, -1, 8, *rest)), 2).reshape_as(tensor).contiguous()


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


def make_inputs(case, requested, rng, edge_cases):
    rank, world = dist.get_rank(), dist.get_world_size()
    count = 0 if edge_cases and world > 1 and requested == 1 and rank == world - 1 else requested
    source = torch.randn(requested, case.hidden, device="cuda", dtype=torch.bfloat16,
                         generator=rng) * (0.5 * case.input_scale)
    token = torch.arange(requested, device="cuda")[:, None]
    slot = torch.arange(case.topk, device="cuda")[None, :]
    ids = ((token * case.topk + slot + rank * 17) % case.experts).contiguous()
    weights = torch.rand(requested, case.topk, device="cuda", generator=rng).softmax(-1)
    ids[count:], weights[count:] = -1, 0
    if edge_cases and count > 1:
        ids[0, 0], weights[0, 0] = -1, 0
        weights[1, 0] = 0
    return source, ids, weights, count


def run_forward(case, source, requested, pipeline, reference, *, combine, check=True,
                reference_tokens=0, max_diff=.0005, max_relative_l2=None,
                replicas=False, edge_cases=False, benchmark=None, verbose=True,
                routing_weights=None):
    """One complete forward for correctness, graph replay and optional end-to-end timing."""
    rank, world, count = dist.get_rank(), dist.get_world_size(), source.shape[0]
    compile_kernels(pipeline)
    combine.fill_(float("nan"))
    torch.cuda.synchronize()
    dist.barrier()
    actual = pipeline().clone()
    counts = gather(torch.tensor(count, device="cuda")).tolist()
    report = dict(case=case.name, precision=case.precision, ep=world, tokens_per_rank=counts,
                  hidden=case.hidden, intermediate=case.intermediate, experts=case.experts,
                  capacity=case.capacity, topk=case.topk, activation=case.activation,
                  scatter_dtype="bf16", routing="round_robin", split_weights=replicas,
                  edge_cases=edge_cases, correctness=dict(status="not_checked"))
    if check:
        checked = min(requested, reference_tokens) if reference_tokens else requested
        expected = reference(checked)
        assert actual.isfinite().all(), (rank, "nonfinite pipeline output")
        sample = actual[:min(count, checked)]
        metrics = torch.tensor([float(calc_diff(sample, expected)), relative_l2(sample, expected)],
                               device="cuda")
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        assert metrics[0].item() < max_diff, (case.name, "pipeline diff", metrics.tolist())
        if max_relative_l2 is not None:
            assert metrics[1].item() < max_relative_l2, (case.name, "pipeline relative L2", metrics.tolist())
        report["correctness"] = dict(status="passed", max_pipeline_diff=metrics[0].item(),
                                     max_relative_l2=metrics[1].item(),
                                     checked_tokens_per_rank=[min(n, checked) for n in counts])
        check_graph_replay(pipeline, actual, input_tensor=source, routing_weights=routing_weights)
        if routing_weights is not None:
            report["correctness"]["routing_weight_replay_checked"] = True
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
                   capacity=args.capacity if args.capacity is not None else case.capacity)
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


def check_graph_replay(pipeline, expected, *, input_tensor=None, routing_weights=None):
    """Check replay generations, optionally changing inputs to expose stale reads."""
    for _ in range(3):
        pipeline()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = pipeline()
    saved = input_tensor.clone() if input_tensor is not None else None
    for factor in (0.0, -1.0, 1.0):
        reference = expected
        if saved is not None:
            input_tensor.copy_(saved * factor)
            reference = pipeline().clone()
            torch.cuda.synchronize()
            dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        dist.barrier()
    # The last replay uses the original inputs, so downstream timing and
    # correctness checks still see the caller's original workspace contents.
    if routing_weights is not None:
        saved_weights = routing_weights.clone()
        routing_weights.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.count_nonzero(result).item() == 0, "graph replay used stale routing weights"
        dist.barrier()
        routing_weights.copy_(saved_weights)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        dist.barrier()


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


def clear_dispatch_counts(buffer, experts):
    """Preserve generation flags; only recycle per-invocation route counts."""
    offset = buffer._layout.get("off_ws", 0) + 32
    count_bytes = (2 * experts + experts // dist.get_world_size()) * 8
    buffer.buffer[offset:offset + count_bytes].zero_()


def calc_diff(x, y):
    x, y = x.double(), y.double()
    return (x - y).square().sum() / (x.square() + y.square()).sum().clamp_min(1e-30)

def _get_weight_perm():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm

_perm = _get_weight_perm()

def quant_w4a16_mxfp4(y, groupsize=32):
    """Quantize [E, N, K] weights to raw W4A16-packed E2M1 values and UE8M0 scales."""
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

    all_qs = []
    for expert in range(e):
        packed_order = codes[expert]
        packed_order = packed_order.view(k // tile, tile, n // tile, tile)
        packed_order = packed_order.permute(0, 2, 1, 3).reshape(k // tile, n * tile)
        packed_order = packed_order.reshape(-1, _perm.numel())[:, _perm].reshape(packed_order.shape)
        packed_order = packed_order.to(torch.int32)

        packed = torch.zeros(
            (packed_order.shape[0], packed_order.shape[1] // 8),
            dtype=torch.int32, device=y.device)
        for nibble in range(8):
            packed |= packed_order[:, nibble::8] << (4 * nibble)
        all_qs.append(packed)

    raw_scale = (scale_bits >> 23).to(torch.uint8)
    return refs, torch.stack(all_qs), raw_scale

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
