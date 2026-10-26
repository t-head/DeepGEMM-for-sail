r"""PPU MXFP4 MegaMoE public-entry correctness and regression tests.

Single rank:
    python tests/test_mega_moe.py --case smoke_swiglu --num-tokens 33

Distributed correctness:
    torchrun --standalone --nproc-per-node=8 tests/test_mega_moe.py \
        --case deepseek_v4_pro --num-tokens 32 1024 4096 --routing random \
        --reference-tokens 8

Distributed regression suite:
    torchrun --standalone --nproc-per-node=2 -m pytest -q -s -k smoke \
        tests/test_mega_moe.py

Performance and optional DeepEP comparisons live in tests/bench_mega_moe.py.
The legacy --benchmark CLI forwards there.
"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

# Direct execution exercises this checkout, even if another is installed.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deep_gemm
from deep_gemm.testing import calc_diff
from utils import is_ppu1v5_device


# Cases and input/reference helpers are shared with bench_mega_moe.py.
MODEL_TOKENS = (1024, 2048, 4096)
MAX_DIFF = .003


@dataclass(frozen=True)
class MegaMoECase:
    name: str
    hidden: int
    intermediate: int
    experts: int
    topk: int
    capacity: int
    tokens: tuple[int, ...] = (1, 33)
    activation: str = "swiglu"
    clamp: float | None = None
    worlds: tuple[int, ...] = (1, 2, 4, 8)
    reference_tokens: int = 8
    precision: str = "w4a4"
    routing: str = "round_robin"

    def require_world(self):
        world = dist.get_world_size()
        if world not in self.worlds:
            pytest.skip(f"{self.name} requires EP{self.worlds}; launched EP{world}")
        assert self.experts % world == 0


# FP4 dispatch copies scales in 16-byte chunks; smoke H is at least 512.
CASES = (
    MegaMoECase("smoke_swiglu", 512, 256, 8, 2, 129, tokens=(1, 0, 33, 129, 8), clamp=2.0),
    MegaMoECase("smoke_step", 512, 256, 8, 2, 129, tokens=(1, 0, 33, 129, 8), activation="step", clamp=2.0),
    MegaMoECase("smoke_situ", 512, 256, 8, 2, 129, tokens=(1, 0, 33, 129, 8), activation="situ"),
    MegaMoECase("deepseek_v4_pro", 7168, 3072, 384, 6, 8192, tokens=MODEL_TOKENS, clamp=10.0, worlds=(8,)),
    MegaMoECase("deepseek_v4_flash", 4096, 2048, 256, 6, 4096, tokens=MODEL_TOKENS, clamp=10.0, worlds=(4,)),
    MegaMoECase("glm52", 6144, 2048, 256, 8, 4096, tokens=MODEL_TOKENS, worlds=(8,)),
    MegaMoECase("kimi_k3", 3584, 3072, 896, 16, 4096, tokens=MODEL_TOKENS, activation="situ", worlds=(8, 16)),
    MegaMoECase("qwen38_2_4t", 8192, 2048, 512, 10, 4096, tokens=MODEL_TOKENS, worlds=(8, 16)),
    MegaMoECase("mimo_v26_pro", 6144, 2048, 384, 8, 8192, tokens=(*MODEL_TOKENS, 8192), worlds=(8,)),
)


# Input generation and independent mathematical references.


def _quantize_mxfp4(src_tensor):
    src = src_tensor.float()
    *batch, n, k = src.shape
    grouped = src.reshape(*batch, n, k // 32, 32)
    amax = grouped.abs().amax(-1).clamp_min(1e-10)
    bits = ((amax / 6.0).contiguous().view(torch.int32) + 0x007fffff) & 0x7f800000
    scale = bits.view(torch.float32)
    normalized = (grouped / scale.unsqueeze(-1)).nan_to_num().reshape_as(src)
    v = normalized.abs()
    thresholds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=src.device)
    code = (v.unsqueeze(-1) > thresholds).sum(-1)
    code += ((v == 0.75) | (v == 1.75) | (v == 3.5)).long()
    code = (code | ((normalized < 0).long() << 3)).to(torch.uint8)
    code = code.reshape(*batch, n, k // 2, 2)
    return (code[..., 0] | (code[..., 1] << 4)).contiguous(), (bits >> 23).to(torch.uint8).contiguous()


def _dequantize_mxfp4(quant_tensor, scale, target_dtype=torch.float32):
    assert quant_tensor.dtype == torch.uint8 and scale.dtype == torch.uint8
    device = quant_tensor.device
    *batch_dims, n, packed_k = quant_tensor.shape
    k = packed_k * 2
    quant_int = quant_tensor.to(torch.int32)
    evens = quant_int & 0xF
    odds = (quant_int >> 4) & 0xF
    vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    lookup_table = torch.tensor(vals + [-v for v in vals], dtype=torch.float32, device=device)
    fp32_tensor = torch.stack([lookup_table[evens], lookup_table[odds]], dim=-1).view(*batch_dims, n, k)
    dq_scale = (scale.to(torch.int32) << 23).view(torch.float32)
    padded_k = dq_scale.shape[-1] * 32
    padded_tensor = F.pad(fp32_tensor, (0, padded_k - k))
    padded_tensor = padded_tensor.view(*batch_dims, n, dq_scale.shape[-1], 32)
    out_padded = (padded_tensor * dq_scale.unsqueeze(-1)).view(*batch_dims, n, padded_k)
    return out_padded[..., :k].to(target_dtype)


def _make_weights(experts, n, k, amplitude, rng):
    # Bound temporary quantization memory for model-sized expert matrices.
    packed, scales = [], []
    for _ in range(experts):
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=rng) * amplitude
        q, s = _quantize_mxfp4(weight)
        packed.append(q)
        scales.append(s)
    return torch.stack(packed), torch.stack(scales)


def _make_routes(case, tokens, rng):
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


def _replicate_physical_expert(weights):
    """Prepare a duplicate physical expert at load time, in the same weight pair."""
    for tensor in weights:
        tensor[-1].copy_(tensor[0])


def _transform_weights(l1_raw, l2_raw, activation, replicas=False):
    if not replicas:
        l1, l2 = deep_gemm.transform_weights_for_mega_moe(l1_raw, l2_raw, activation=activation)
        return l1, l2, {}
    _replicate_physical_expert(l1_raw)
    _replicate_physical_expert(l2_raw)
    # Allocate master and replica banks independently, retaining the full raw
    # bank for the reference. Forward must not concatenate or copy these weights.
    l1, l2 = deep_gemm.transform_weights_for_mega_moe(
        tuple(t[:-1].clone() for t in l1_raw), tuple(t[:-1].clone() for t in l2_raw), activation=activation)
    r1, r2 = deep_gemm.transform_weights_for_mega_moe(
        tuple(t[-1:].clone() for t in l1_raw), tuple(t[-1:].clone() for t in l2_raw), activation=activation)
    return l1, l2, dict(l1_weights_replicas=r1, l2_weights_replicas=r2)


def _all_gather(tensor):
    tensors = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors, tensor)
    return torch.stack(tensors)


def _reference_activation(projection, kind="swiglu", clamp=None, *, round_bf16=True):
    """Use the kernel's BF16 boundaries, or retain FP32 for accuracy diagnostics."""
    projection = projection.bfloat16() if round_bf16 else projection.float()
    if kind == "situ":
        gate, up = projection.float().chunk(2, -1)
        output = (4 * torch.tanh(gate / 4) * torch.sigmoid(gate)
                  * 25 * torch.tanh(up / 25))
        return output.bfloat16() if round_bf16 else output
    gate, up = projection.chunk(2, -1)
    if clamp is not None:
        up = up.clamp(-clamp, clamp)
        if kind != "step":
            gate = gate.clamp(max=clamp)
    gate = torch.nn.functional.silu(gate.float())
    if kind == "step" and clamp is not None:
        gate = gate.clamp(max=clamp)
    output = gate * up.float()
    return output.bfloat16() if round_bf16 else output


def _reference_mega_moe(case, source, ids, weights, l1_weights_raw, l2_weights_raw,
                        num_tokens, checked, *, quantization_aware=True):
    """Independent MoE with MXFP4 weights and optional activation quantization.

    The FP32 diagnostic retains the supplied MXFP4 weights: it measures activation
    quantization/rounding error, not accuracy against an unquantized model.
    """
    c = case
    rank_idx, num_ranks = dist.get_rank(), dist.get_world_size()
    num_experts_per_rank = case.experts // num_ranks
    if quantization_aware:
        q, scales = _quantize_mxfp4(source[:checked])
        x = _all_gather(_dequantize_mxfp4(q, scales))
    else:
        x = _all_gather(source[:checked].float())
    all_ids, all_weights = _all_gather(ids[:checked].contiguous()), _all_gather(weights[:checked].contiguous())
    expected = torch.zeros(num_ranks, checked, c.hidden, device="cuda")
    (l1_weights, l1_weights_sf), (l2_weights, l2_weights_sf) = (l1_weights_raw, l2_weights_raw)
    for expert in range(num_experts_per_rank):
        src, tok, slot = torch.where(all_ids == rank_idx * num_experts_per_rank + expert)
        if not src.numel():
            continue
        projection = x[src, tok] @ _dequantize_mxfp4(l1_weights[expert], l1_weights_sf[expert]).t()
        activated = _reference_activation(projection, c.activation, c.clamp,
                                          round_bf16=quantization_aware)
        route_weights = all_weights[src, tok, slot, None]
        if quantization_aware:
            q, scales = _quantize_mxfp4(activated)
            activated = _dequantize_mxfp4(q, scales)
        y = activated @ _dequantize_mxfp4(l2_weights[expert], l2_weights_sf[expert]).t()
        if quantization_aware:
            y = y.bfloat16()
        y = y.float() * route_weights
        expected.view(-1, c.hidden).index_add_(0, src * checked + tok, y)
    dist.all_reduce(expected)
    result = expected[rank_idx, :min(num_tokens, checked)]
    return result.bfloat16() if quantization_aware else result


def _relative_l2(actual, expected):
    return ((actual.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1e-8)).item()


# Distributed lifecycle and public-entry regression checks.


@contextmanager
def _process_group(request=None):
    """Process-group lifecycle for pytest and standalone runs."""
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
def mega_moe_group(request):
    if not torch.cuda.is_available():
        pytest.skip("requires a PPU runtime")
    with _process_group(request) as group:
        yield group


def _compile_kernels(*calls):
    """Compile and load native kernels serially without launching them."""

    previous = deep_gemm.get_compile_mode()
    deep_gemm.set_compile_mode(1)
    try:
        for call in calls:
            call()
    finally:
        deep_gemm.set_compile_mode(previous)


def _check_workspace_options(case, group):
    """All failures must occur before collective symmetric allocation."""

    # Exercise the public constructor's row-index guard without allocating
    # the oversized workspace or requiring a 32-rank process group.
    oversized_group = SimpleNamespace(size=lambda: 32)
    with patch("torch.distributed._symmetric_memory.empty",
               side_effect=AssertionError("overflow must fail before allocation")) as allocate:
        for factory in (deep_gemm.get_symm_buffer_for_mega_moe, deep_gemm.SymmBuffer):
            with pytest.raises(RuntimeError, match="signed 32-bit row index range"):
                factory(oversized_group, num_experts=1024, num_max_tokens_per_rank=1 << 23,
                        num_topk=16, hidden=512, intermediate_hidden=256)
        allocate.assert_not_called()

    with patch("torch.distributed._symmetric_memory.empty",
               side_effect=AssertionError("expert limit must fail before allocation")) as allocate:
        for factory in (deep_gemm.get_symm_buffer_for_mega_moe, deep_gemm.SymmBuffer):
            with pytest.raises(RuntimeError, match="at most 1024 local physical experts"):
                factory(SimpleNamespace(size=lambda: 1), num_experts=1025,
                        num_max_tokens_per_rank=1, num_topk=2, hidden=512, intermediate_hidden=256)
        allocate.assert_not_called()

    options = dict(num_experts=case.experts, num_max_tokens_per_rank=case.capacity,
                   num_topk=case.topk, hidden=case.hidden, activation=case.activation,
                   intermediate_hidden=case.intermediate)
    invalid = [({"mma_type": "unsupported"}, "requires mma_type"),
               ({"activation": "unsupported"}, "unsupported precision/activation"),
               ({"intermediate_hidden": 127}, "intermediate_hidden"),
               ({"hidden": case.hidden + 1}, "hidden must be divisible"),
               ({"num_topk": 17}, "Invalid MegaMoE"),
               ({"num_max_tokens_per_rank": 0}, "Invalid MegaMoE")]
    for mma_type in ("fp8xfp4", "fp8xfp8", "bf16xbf16"):
        invalid.append(({"mma_type": mma_type}, "requires mma_type"))
    invalid.append(({"num_shared_experts": 1}, "does not support shared experts"))
    for legacy in ({"gemm1_block_n": 256}, {"gemm1_block_k": 128}, {"config_override": {}},
                   {"gemm1_config": (64, 256, 128, 32, 64, 3)},
                   {"gemm2_config": (64, 256, 128, 32, 64, 3)}):
        for factory in (deep_gemm.get_symm_buffer_for_mega_moe, deep_gemm.SymmBuffer):
            with pytest.raises(TypeError, match=next(iter(legacy))):
                factory(group, **options, **legacy)
    for changed, message in invalid:
        for allocate in (deep_gemm.get_symm_buffer_for_mega_moe, deep_gemm.SymmBuffer):
            with pytest.raises((ValueError, RuntimeError), match=message):
                allocate(group, **{**options, **changed})


def _check_public_entry(entry, y, l1_weights, l2_weights, sym_buffer, source, ids, weights, **options):
    """Reject malformed calls before advancing the collective compute protocol."""

    deep_gemm.mega_moe_pre_dispatch(source, ids, weights, sym_buffer)
    with pytest.raises((ValueError, RuntimeError), match="unsupported precision/activation"):
        entry(y, l1_weights, l2_weights, sym_buffer, **{**options, "activation": "unsupported"})
    unsupported_activation = "swiglu_oai"
    with pytest.raises((ValueError, RuntimeError), match="unsupported precision/activation"):
        entry(y, l1_weights, l2_weights, sym_buffer, **{**options, "activation": unsupported_activation})
    for changed, message in (
        ({"recipe": (128, 128, 128)}, "recipe=\\(1, 1, 32\\)"),
        ({"recipe": (1, 1, 16)}, "recipe=\\(1, 1, 32\\)"),
        ({"activation_clamp": float("nan")}, "activation_clamp"),
        ({"activation_clamp": -1.0}, "activation_clamp"),
        ({"activation_clamp": -float("inf")}, "activation_clamp"),
        ({"situ_beta": float("nan")}, "SiTU parameters|situ_beta"),
        ({"situ_linear_beta": 0.0}, "SiTU parameters|situ_beta"),
    ):
        with pytest.raises(RuntimeError, match=message):
            entry(y, l1_weights, l2_weights, sym_buffer, **{**options, **changed})
    num_experts_per_rank = sym_buffer.num_experts // sym_buffer.num_ranks
    invalid_stats = [torch.zeros(num_experts_per_rank, device=y.device),
                     torch.zeros(num_experts_per_rank, dtype=torch.int32)]
    if num_experts_per_rank > 1:
        invalid_stats.append(torch.zeros(2 * num_experts_per_rank, device=y.device, dtype=torch.int32)[::2])
    for stats in invalid_stats:
        with pytest.raises(RuntimeError, match="tensor device, dtype or layout"):
            entry(y, l1_weights, l2_weights, sym_buffer,
                  **{**options, "cumulative_local_expert_recv_stats": stats})
    with pytest.raises(RuntimeError, match="one INT32 element per local physical expert"):
        entry(y, l1_weights, l2_weights, sym_buffer,
              **{**options, "cumulative_local_expert_recv_stats": torch.zeros(
                  num_experts_per_rank + 1, device=y.device, dtype=torch.int32)})
    with pytest.raises(RuntimeError, match="tensor device, dtype or layout"):
        entry(y.float(), l1_weights, l2_weights, sym_buffer, **options)
    bad_w2 = (l2_weights[0][:, :-1].contiguous(), l2_weights[1])
    with pytest.raises(RuntimeError, match="tensor shape"):
        entry(y, l1_weights, bad_w2, sym_buffer, **options)
    # Preserve shape and N stride while breaking the packed expert/scale strides.
    def invalid_scale_views(scale):
        yield scale[:, :, :1].expand_as(scale)
        yield scale.transpose(1, 2).repeat(1, 2, 1).transpose(1, 2)[:, :, ::2]
        if scale.size(0) > 1:
            yield scale.transpose(1, 2).repeat(2, 1, 1).transpose(1, 2)[::2]

    for scale in invalid_scale_views(l2_weights[1]):
        with pytest.raises(RuntimeError, match="invalid transformed W4A4 GEMM2 scales"):
            entry(y, l1_weights, (l2_weights[0], scale), sym_buffer, **options)
    for name in ("l1_weights_replicas", "l2_weights_replicas"):
        with pytest.raises(RuntimeError, match="must be provided together"):
            entry(y, l1_weights, l2_weights, sym_buffer, **options, **{name: l1_weights})
    with pytest.raises(RuntimeError, match="at least one replica slot"):
        entry(y, l1_weights, l2_weights, sym_buffer, **options,
              l1_weights_replicas=l1_weights, l2_weights_replicas=l2_weights)
    if num_experts_per_rank > 1:
        masters = tuple(t[:-1] for t in l1_weights), tuple(t[:-1] for t in l2_weights)
        replicas = tuple(t[-1:] for t in l1_weights), tuple(t[-1:] for t in l2_weights)
        with pytest.raises(RuntimeError, match="all local physical experts"):
            entry(y, *masters, sym_buffer, **options)
        with pytest.raises(RuntimeError, match="tensor shape"):
            entry(y, *masters, sym_buffer, **options,
                  l1_weights_replicas=l1_weights, l2_weights_replicas=replicas[1])
        with pytest.raises(RuntimeError, match="invalid transformed W4A4 GEMM2 scales"):
            entry(y, *masters, sym_buffer, **options, l1_weights_replicas=replicas[0],
                  l2_weights_replicas=(replicas[1][0], next(invalid_scale_views(replicas[1][1]))))


def _check_input_views(entry, y, l1_weights, l2_weights, sym_buffer, source, ids, weights, **options):
    """Public input views and positional calls match BF16 input preparation."""
    deep_gemm.mega_moe_pre_dispatch(source, ids, weights, sym_buffer)
    # A caller may fill public views directly and choose tokens through y.
    # Preserve exact staged bytes so this checks the API contract independently
    # of Torch versus kernel quantization at FP4 rounding boundaries.
    count = source.size(0)
    views = (sym_buffer.x, sym_buffer.x_sf, sym_buffer.topk_idx, sym_buffer.topk_weights)
    staged = [view[:count].clone() for view in views]
    entry(y, l1_weights, l2_weights, sym_buffer, **options)
    expected = y.clone()
    # The common argument order matches SM90; PPU extensions remain keyword-only.
    entry(y, l1_weights, l2_weights, sym_buffer, None, (1, 1, 32),
          options["activation"], options["activation_clamp"], options["fast_math"])
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    if options.get("fast_math", True):
        default_options = {name: value for name, value in options.items() if name != "fast_math"}
        entry(y, l1_weights, l2_weights, sym_buffer, **default_options)
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
    for view, value in zip(views, staged):
        view[:count].copy_(value)
    entry(y[:0], l1_weights, l2_weights, sym_buffer, **options)
    entry(y, l1_weights, l2_weights, sym_buffer, **options)
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    _check_offset_views(entry, l1_weights, l2_weights, sym_buffer, source, ids, weights, **options)
    # Direct input writes also support -1 with nonzero routing weights.
    for view, value in zip(views, staged):
        view[:count].copy_(value)
    sym_buffer.topk_weights[:count].copy_(weights)

    def direct():
        entry(y, l1_weights, l2_weights, sym_buffer, **options)
        return y

    torch.testing.assert_close(direct(), expected, rtol=0, atol=0)
    _check_graph_replay(direct, expected, routing_ids=sym_buffer.topk_idx[:count],
                       routing_weights=sym_buffer.topk_weights[:count],
                       num_experts=sym_buffer.num_experts)


def _check_recv_stats(pipeline, y, routing_ids, routing_weights, num_experts):
    """Cumulative receive statistics survive reuse and change with replayed routes."""
    rank, world = dist.get_rank(), dist.get_world_size()
    local_experts = num_experts // world
    stats = torch.full((local_experts,), 7, dtype=torch.int32, device=y.device)
    expected_stats = stats.clone()
    saved_ids, saved_weights = routing_ids.clone(), routing_weights.clone()

    def received():
        ids = routing_ids.flatten()
        ids = ids[(ids >= 0) & (ids < num_experts)]
        counts = torch.zeros(num_experts, dtype=torch.int32, device=y.device)
        counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
        dist.all_reduce(counts)
        return counts[rank * local_experts:(rank + 1) * local_experts]

    try:
        _compile_kernels(lambda: pipeline(stats))
        torch.testing.assert_close(stats, expected_stats, rtol=0, atol=0)
        baseline = pipeline().clone()
        for _ in range(3):
            pipeline(stats)
            expected_stats += received()
            torch.testing.assert_close(stats, expected_stats, rtol=0, atol=0)
            torch.testing.assert_close(y, baseline, rtol=0, atol=0)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pipeline(stats)
        # Capture records work; only actual launches may increment the counters.
        torch.testing.assert_close(stats, expected_stats, rtol=0, atol=0)
        hot = (torch.arange(routing_ids.size(1), device=y.device) +
               ((rank + 1) % world) * local_experts) % num_experts
        for ids, zero_weights in ((hot.expand_as(routing_ids), False),
                                  (saved_ids, True),
                                  (torch.full_like(saved_ids, -1), False),
                                  (saved_ids, False)):
            routing_ids.copy_(ids)
            routing_weights.copy_(0 * saved_weights if zero_weights else saved_weights)
            baseline = pipeline().clone()
            count = received()
            for _ in range(2):
                graph.replay()
                expected_stats += count
                torch.testing.assert_close(stats, expected_stats, rtol=0, atol=0)
                torch.testing.assert_close(y, baseline, rtol=0, atol=0)
    finally:
        routing_ids.copy_(saved_ids)
        routing_weights.copy_(saved_weights)


def _check_offset_views(entry, l1_weights, l2_weights, sym_buffer, source, ids, weights, **options):
    """Offset views must match aligned complete calls without touching their guards."""

    def forward(input_tensor, y):
        deep_gemm.mega_moe_pre_dispatch(input_tensor, ids, weights, sym_buffer)
        entry(y, l1_weights, l2_weights, sym_buffer, **options)
        return y

    aligned_input = source.clone()
    aligned_output = torch.empty_like(source)
    _compile_kernels(lambda: forward(aligned_input, aligned_output))
    torch.cuda.synchronize()
    dist.barrier()
    expected = forward(aligned_input, aligned_output).clone()

    def make_view(offset):
        storage = torch.full((source.numel() + offset + 8,), 13.0,
                             dtype=source.dtype, device=source.device)
        view = storage[offset:offset + source.numel()].view_as(source)
        assert view.is_contiguous() and view.storage_offset() == offset
        return storage, view

    def check_guards(storage, offset):
        assert torch.all(storage[:offset] == 13)
        assert torch.all(storage[offset + source.numel():] == 13)

    # Exercise input-only, output-only and joint offsets, including every BF16
    # alignment modulo 16 bytes. The empty peer still participates in compute.
    for input_offset, output_offset in ((1, 0), (0, 1), *((i, i) for i in range(1, 8))):
        input_storage, offset_input = make_view(input_offset)
        output_storage, offset_output = make_view(output_offset)
        offset_input.copy_(source)
        actual = forward(offset_input, offset_output)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(offset_input, source, rtol=0, atol=0)
        check_guards(input_storage, input_offset)
        check_guards(output_storage, output_offset)

    # Replay from offset allocations too; changed inputs and routes must not
    # consume data left by an eager call or by an earlier workspace generation.
    _check_graph_replay(lambda: forward(offset_input, offset_output), expected,
                       input_tensor=offset_input, routing_ids=ids, routing_weights=weights,
                       num_experts=sym_buffer.num_experts)
    check_guards(input_storage, input_offset)
    check_guards(output_storage, output_offset)


def _check_graph_replay(pipeline, expected, *, input_tensor=None, routing_weights=None, routing_ids=None,
                       check_reference=None, num_experts=None):
    """Recycle state through public calls with changing inputs and routes."""
    for _ in range(3):
        pipeline()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = pipeline()

    rank, world = dist.get_rank(), dist.get_world_size()
    check_peers = num_experts is not None and world > 1
    generation = 0

    def compare(expect_zero=False):
        nonlocal generation
        result.fill_(float("nan"))
        torch.cuda.synchronize()
        dist.barrier()
        # Stagger a rotating rank after poisoning the public output. The
        # complete entry must synchronize its own scatter and combine stages.
        if check_peers and rank == generation % world:
            time.sleep(.003)
        # Replay must produce its own intermediates before eager can refresh them.
        graph.replay()
        replayed = result.clone()
        torch.cuda.synchronize()
        dist.barrier()
        result.fill_(float("nan"))
        torch.cuda.synchronize()
        dist.barrier()
        if check_peers and rank == generation % world:
            time.sleep(.003)
        eager = pipeline().clone()
        torch.cuda.synchronize()
        assert replayed.isfinite().all(), "recycled routes read an unproduced slot"
        torch.testing.assert_close(replayed, eager, rtol=0, atol=0)
        if expect_zero:
            assert torch.count_nonzero(replayed) == 0, "inactive routes must produce exact zeros"
        if check_reference is not None:
            check_reference(replayed)
        dist.barrier()
        generation += 1

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
        if check_peers:
            experts_per_rank = num_experts // world
            # Send to the next rank, then concentrate work on the last rank.
            # The last rank has no input in the one-token edge case but must
            # still compute remote experts. Wrap slots for small expert counts.
            for destination in ((rank + 1) % world, world - 1):
                routing_ids.copy_((destination * experts_per_rank + hot_routes) % num_experts)
                compare()
            rng = torch.Generator(device=saved_ids.device).manual_seed(20261022 + rank)
            routing_ids.copy_(torch.rand(saved_ids.shape[0], num_experts,
                                        device=saved_ids.device, generator=rng)
                              .topk(saved_ids.shape[1], dim=1).indices)
            compare()
        for changed, expect_zero in ((hot_routes, False), (torch.full_like(saved_ids, -1), True),
                                     (saved_ids, False)):
            routing_ids.copy_(changed)
            compare(expect_zero=expect_zero)
        # Return to original routing before changing weights.
    if routing_weights is not None:
        saved_weights = routing_weights.clone()
        slots = torch.arange(saved_weights.shape[1], device=saved_weights.device)
        # Include tiny, nonuniform and zero routing weights. Do not renormalize:
        # callers may supply weights whose sum is not one.
        factors = torch.where(slots == 0, 0.0, 10.0 ** (-slots.float()))
        for changed, expect_zero in ((saved_weights * factors, False),
                                     (torch.zeros_like(saved_weights), True), (saved_weights, False)):
            routing_weights.copy_(changed)
            compare(expect_zero=expect_zero)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


# Complete-call numerical and state-reuse checks.


def _check_mega_moe(case, source, requested, pipeline, reference, *, check=True,
                reference_tokens=0, max_diff=MAX_DIFF, max_relative_l2=None,
                replicas=False, edge_cases=False, verbose=True,
                routing_weights=None, routing_ids=None, workspace=None,
                high_precision_reference=None):
    """Check a complete forward, then reuse its workspace with graph replay."""
    rank, world, count = dist.get_rank(), dist.get_world_size(), source.shape[0]
    _compile_kernels(pipeline)
    storage = workspace.buffer if workspace is not None else None
    torch.cuda.synchronize()
    dist.barrier()
    actual = pipeline().clone()
    counts = _all_gather(torch.tensor(count, device="cuda")).tolist()
    report = dict(case=case.name, precision=case.precision, ep=world, tokens_per_rank=counts,
                  hidden=case.hidden, intermediate=case.intermediate, experts=case.experts,
                  capacity=case.capacity, topk=case.topk, activation=case.activation,
                  scatter_dtype="bf16", routing=case.routing, replicated_physical_experts=replicas,
                  edge_cases=edge_cases, correctness=dict(status="not_checked"))
    if check:
        checked = min(requested, reference_tokens) if reference_tokens else requested
        replay_metrics = []
        def check_reference(result):
            expected = reference(checked)
            assert result.isfinite().all(), (rank, "nonfinite pipeline output")
            sample = result[:min(count, checked)]
            metrics = torch.tensor([float(calc_diff(sample, expected)), _relative_l2(sample, expected)],
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
                                     reference="quantization_aware", diff_limit=max_diff,
                                     checked_tokens_per_rank=[min(n, checked) for n in counts])
        if high_precision_reference is not None:
            expected = high_precision_reference(checked)
            sample = actual[:min(count, checked)]
            errors = torch.tensor([float(calc_diff(sample, expected)), _relative_l2(sample, expected)],
                                  device="cuda")
            dist.all_reduce(errors, op=dist.ReduceOp.MAX)
            # Diagnostic only: MXFP4 activation quantization is intentionally
            # absent here, so the kernel agreement threshold does not apply.
            report["activation_quantization_error"] = dict(
                reference="fp32_activations_with_mxfp4_weights",
                max_diff=errors[0].item(), max_relative_l2=errors[1].item())
        if edge_cases:
            _check_graph_replay(pipeline, actual, input_tensor=source, routing_weights=routing_weights,
                               routing_ids=routing_ids, check_reference=check_reference,
                               num_experts=case.experts)
            report["correctness"]["changed_routes_reference_checked"] = True
            if world > 1:
                report["correctness"]["cross_rank_visibility_checked"] = True
            if routing_weights is not None:
                report["correctness"]["routing_weight_replay_checked"] = True
        else:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replayed = pipeline()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(replayed, actual, rtol=0, atol=0)
            check_reference(replayed)
        report["correctness"]["max_replay_pipeline_diff"] = max(m[0] for m in replay_metrics[1:])
        report["correctness"]["max_replay_relative_l2"] = max(m[1] for m in replay_metrics[1:])
        if workspace is not None:
            assert workspace.buffer is storage, "complete call replaced workspace storage"
            report["correctness"]["workspace_reuse_checked"] = True
    if verbose and rank == 0:
        print(json.dumps(report), flush=True)
    dist.barrier()
    return report


# End-to-end correctness driver: setup -> public complete call -> reference.


@pytest.fixture
def mega_moe_smoke(mega_moe_group):
    """One small public pipeline shared by API and state-reuse regressions."""
    case = replace(CASES[0], tokens=(33,))
    case.require_world()
    rank, world = dist.get_rank(), dist.get_world_size()
    rng = torch.Generator(device="cuda").manual_seed(20261016 + rank)
    source = torch.randn(case.capacity, case.hidden, device="cuda", dtype=torch.bfloat16, generator=rng) * .5
    ids = _make_routes(case, case.capacity, rng)
    weights = torch.rand(case.capacity, case.topk, device="cuda", generator=rng).softmax(-1)
    count = 0 if world > 1 and rank == world - 1 else 33
    ids[count:], weights[count:] = -1, 0
    if count:
        ids[0, 0] = -1
    w1_raw = _make_weights(case.experts // world, 2 * case.intermediate, case.hidden, .5, rng)
    w2_raw = _make_weights(case.experts // world, case.hidden, case.intermediate, .125, rng)
    if case.experts // world > 1:
        # Include externally replicated slots in the public state/statistics tests.
        _replicate_physical_expert(w1_raw)
        _replicate_physical_expert(w2_raw)
    w1, w2 = deep_gemm.transform_weights_for_mega_moe(w1_raw, w2_raw)
    buffer = deep_gemm.SymmBuffer(mega_moe_group, case.experts, case.capacity, case.topk,
                                 case.hidden, case.intermediate)
    x, routed_ids, routed_weights = source[:count], ids[:count], weights[:count]
    y = torch.empty_like(x)
    options = dict(activation=case.activation, activation_clamp=case.clamp, fast_math=True)

    def pipeline(stats=None):
        deep_gemm.mega_moe_pre_dispatch(x, routed_ids, routed_weights, buffer)
        deep_gemm.fp4_fp4_mega_moe(y, w1, w2, buffer, stats, **options)
        return y

    def reference(checked, clamp=case.clamp):
        return _reference_mega_moe(replace(case, clamp=clamp), source, ids, weights,
                                  w1_raw, w2_raw, count, checked)

    try:
        yield SimpleNamespace(case=case, buffer=buffer, x=x, ids=routed_ids, weights=routed_weights,
                              y=y, w1=w1, w2=w2, w1_raw=w1_raw, w2_raw=w2_raw,
                              options=options, pipeline=pipeline, reference=reference)
    finally:
        buffer.destroy()


def test_mega_moe_smoke_api(mega_moe_smoke, mega_moe_group):
    sample = mega_moe_smoke
    _check_workspace_options(sample.case, mega_moe_group)
    transform = deep_gemm.transform_weights_for_mega_moe
    with pytest.raises(TypeError, match="weight, raw UE8M0 scale"):
        transform(sample.w1_raw[0], sample.w2_raw)
    with pytest.raises(ValueError, match="rank-3"):
        transform((sample.w1_raw[0][0], sample.w1_raw[1]), sample.w2_raw)
    with pytest.raises(ValueError, match="raw UE8M0"):
        transform((sample.w1_raw[0], sample.w1_raw[1].float()), sample.w2_raw)
    _check_public_entry(deep_gemm.fp4_fp4_mega_moe, sample.y, sample.w1, sample.w2, sample.buffer,
                        sample.x, sample.ids, sample.weights, **sample.options)
    # A valid call must still work after rejected arguments.
    _check_mega_moe(sample.case, sample.x, 33, sample.pipeline, sample.reference,
                    workspace=sample.buffer, verbose=False)


def test_mega_moe_smoke_stateful(mega_moe_smoke):
    sample = mega_moe_smoke
    if dist.get_world_size() < 2:
        pytest.skip("state-reuse regression requires at least two ranks")
    _compile_kernels(sample.pipeline)
    _check_input_views(deep_gemm.fp4_fp4_mega_moe, sample.y, sample.w1, sample.w2, sample.buffer,
                       sample.x, sample.ids, sample.weights, **sample.options)
    _check_recv_stats(sample.pipeline, sample.y, sample.ids, sample.weights, sample.case.experts)
    _check_mega_moe(sample.case, sample.x, 33, sample.pipeline, sample.reference,
                    workspace=sample.buffer, edge_cases=True,
                    routing_ids=sample.ids, routing_weights=sample.weights, verbose=False)


def test_mega_moe_smoke_workspace(mega_moe_smoke, mega_moe_group):
    """One allocation/registration; base aliases preserve bytes and protocol state."""
    import torch.distributed._symmetric_memory as symm_mem
    sample, group = mega_moe_smoke, mega_moe_group
    base, case = sample.buffer, sample.case
    expected = sample.pipeline().clone()
    saved = base.buffer.clone()
    torch.cuda.synchronize()
    arguments = (group, case.experts, case.capacity, case.topk, case.hidden, case.intermediate)
    with patch.object(symm_mem, "empty", side_effect=AssertionError("base allocated storage")), \
         patch.object(symm_mem, "rendezvous", side_effect=AssertionError("base registered again")), \
         patch.object(torch, "empty", side_effect=AssertionError("base allocated storage")):
        alias = deep_gemm.SymmBuffer(*arguments, 0, "fp4xfp4", "swiglu", base)
    assert alias.buffer is base.buffer and alias.handle is base.handle
    torch.testing.assert_close(alias.buffer, saved, rtol=0, atol=0)
    for view in (alias.x, alias.x_sf, alias.topk_idx, alias.topk_weights):
        assert view.untyped_storage().data_ptr() == base.buffer.data_ptr()
        assert view.storage_offset() * view.element_size() + view.nbytes <= base.buffer.nbytes
    with pytest.raises(ValueError, match="compatible workspace layout"):
        deep_gemm.SymmBuffer(group, case.experts, case.capacity + 1, case.topk,
                            case.hidden, case.intermediate, base=base)
    with pytest.raises(ValueError, match="different process group"):
        deep_gemm.SymmBuffer(SimpleNamespace(size=group.size), *arguments[1:], base=base)
    # The positional constructor/factory order follows upstream. EP1 must not
    # enter the symmetric allocator; EP>1 registers exactly this one allocation.
    if group.size() == 1:
        with patch.object(symm_mem, "empty", side_effect=AssertionError("EP1 symmetric allocation")), \
             patch.object(symm_mem, "rendezvous", side_effect=AssertionError("EP1 rendezvous")), \
             patch.object(torch, "empty", wraps=torch.empty) as allocate:
            with pytest.warns(DeprecationWarning, match="use_fp8_dispatch"):
                fresh = deep_gemm.get_symm_buffer_for_mega_moe(*arguments, 0, False, "fp4xfp4", "swiglu")
        assert allocate.call_count == 1
    else:
        with patch.object(symm_mem, "empty", wraps=symm_mem.empty) as allocate, \
             patch.object(symm_mem, "rendezvous", wraps=symm_mem.rendezvous) as register:
            with pytest.warns(DeprecationWarning, match="use_fp8_dispatch"):
                fresh = deep_gemm.get_symm_buffer_for_mega_moe(*arguments, 0, False, "fp4xfp4", "swiglu")
        assert allocate.call_count == register.call_count == 1
    fresh.destroy()
    with pytest.raises(ValueError, match="invalid SymmBuffer"):
        deep_gemm.SymmBuffer(*arguments, base=fresh)
    with pytest.raises(ValueError, match="incompatible with mma_type"):
        deep_gemm.get_symm_buffer_for_mega_moe(*arguments, use_fp8_dispatch=True)
    # Destroying one Python owner must not invalidate another live alias.
    base.destroy()
    deep_gemm.mega_moe_pre_dispatch(sample.x, sample.ids, sample.weights, alias)
    deep_gemm.fp4_fp4_mega_moe(sample.y, sample.w1, sample.w2, alias, **sample.options)
    torch.testing.assert_close(sample.y, expected, rtol=0, atol=0)
    alias.destroy()


def test_mega_moe_smoke_clamp(mega_moe_smoke):
    """Zero clamps to zero; None and +inf have identical no-clamp semantics."""
    sample = mega_moe_smoke
    outputs = {}
    for limit in (None, float("inf"), 0.0, 2.0):
        options = {**sample.options, "activation_clamp": limit}
        def pipeline():
            deep_gemm.mega_moe_pre_dispatch(sample.x, sample.ids, sample.weights, sample.buffer)
            deep_gemm.fp4_fp4_mega_moe(sample.y, sample.w1, sample.w2, sample.buffer, **options)
            return sample.y
        _compile_kernels(pipeline)
        actual = pipeline().clone()
        expected = sample.reference(33, clamp=limit)
        assert actual.isfinite().all()
        assert float(calc_diff(actual, expected)) < MAX_DIFF
        outputs[limit] = actual
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pipeline()
        graph.replay()
        torch.testing.assert_close(sample.y, actual, rtol=0, atol=0)
    torch.testing.assert_close(outputs[None], outputs[float("inf")], rtol=0, atol=0)
    assert torch.count_nonzero(outputs[0.0]) == 0


def test_mega_moe_smoke_nan_slots(mega_moe_smoke):
    """Produce NaN slots through a public call, then mask them on reuse/replay."""
    sample = mega_moe_smoke
    saved_ids, saved_weights = sample.ids.clone(), sample.weights.clone()
    saved_scale = sample.w2[1].clone()
    # UE8M0 code 255 is NaN. This fault injection uses valid-shaped weights;
    # no private workspace offsets or test-only library entry points are needed.
    try:
        for invalid_ids in (True, False):
            sample.w2[1].fill_(0xffff)
            sample.ids.copy_(torch.arange(sample.case.topk, device="cuda").expand_as(sample.ids))
            sample.weights.fill_(1)
            assert sample.pipeline().isnan().all()
            sample.w2[1].copy_(saved_scale)
            sample.ids.copy_(torch.full_like(saved_ids, -1) if invalid_ids else saved_ids)
            sample.weights.zero_()
            result = sample.pipeline()
            assert torch.count_nonzero(result) == 0
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                sample.pipeline()
            graph.replay()
            assert torch.count_nonzero(result) == 0
    finally:
        sample.ids.copy_(saved_ids)
        sample.weights.copy_(saved_weights)
        sample.w2[1].copy_(saved_scale)


@torch.inference_mode()
def _run_mega_moe(case, group, *, check=True, reference_tokens=0, edge_cases=False,
                  replicas=False, verbose=True, fast_math=True):
    """Run the public complete operator against an independent mathematical reference."""
    rank_idx, num_ranks = dist.get_rank(), dist.get_world_size()
    num_experts_per_rank = case.experts // num_ranks
    rng = torch.Generator(device="cuda").manual_seed(20261016 + rank_idx)
    x_bf16 = torch.randn(case.capacity, case.hidden, device="cuda", dtype=torch.bfloat16, generator=rng) * .5
    l1_weights_raw = _make_weights(num_experts_per_rank, 2 * case.intermediate, case.hidden, .5, rng)
    l2_weights_raw = _make_weights(num_experts_per_rank, case.hidden, case.intermediate, .125, rng)
    l1_weights, l2_weights, replica_options = _transform_weights(
        l1_weights_raw, l2_weights_raw, case.activation, replicas)
    sym_buffer = deep_gemm.SymmBuffer(group, case.experts, case.capacity, case.topk,
                                     case.hidden, case.intermediate, activation=case.activation)
    reports = []
    try:
        for requested in case.tokens:
            empty_rank = edge_cases and num_ranks > 1 and requested == 1 and rank_idx == num_ranks - 1
            num_tokens = 0 if empty_rank else requested
            topk_idx = _make_routes(case, case.capacity, rng)
            topk_weights = torch.rand(case.capacity, case.topk, device="cuda", generator=rng).softmax(-1)
            topk_idx[num_tokens:], topk_weights[num_tokens:] = -1, 0
            if edge_cases and num_tokens > 1:
                topk_idx[0, 0] = -1
                topk_weights[1, 0] = 0
            x = x_bf16[:num_tokens]
            routed_idx, routed_weights = topk_idx[:num_tokens], topk_weights[:num_tokens]
            y = torch.empty_like(x)

            def run_fused(cumulative_local_expert_recv_stats=None):
                deep_gemm.mega_moe_pre_dispatch(x, routed_idx, routed_weights, sym_buffer)
                deep_gemm.fp4_fp4_mega_moe(
                    y, l1_weights, l2_weights, sym_buffer,
                    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                    activation=case.activation, activation_clamp=case.clamp, fast_math=fast_math,
                    **replica_options)
                return y

            report = _check_mega_moe(
                case, x, requested, run_fused,
                lambda checked: _reference_mega_moe(case, x_bf16, topk_idx, topk_weights,
                    l1_weights_raw, l2_weights_raw, num_tokens, checked),
                workspace=sym_buffer, check=check, reference_tokens=reference_tokens,
                replicas=replicas, edge_cases=edge_cases, verbose=False,
                routing_weights=routed_weights, routing_ids=routed_idx,
                high_precision_reference=lambda checked: _reference_mega_moe(
                    case, x_bf16, topk_idx, topk_weights, l1_weights_raw, l2_weights_raw,
                    num_tokens, checked, quantization_aware=False))
            report["fast_math"] = fast_math
            if verbose and rank_idx == 0:
                print(json.dumps(report), flush=True)
            reports.append(report)
    finally:
        sym_buffer.destroy()
    return reports


@pytest.mark.parametrize("case,fast_math", [
    pytest.param(case, fast_math, id=case.name + ("" if fast_math else "_precise"))
    for case in CASES
    for fast_math in ((True, False) if case.name.startswith("smoke") else (True,))
])
def test_mega_moe(case, fast_math, mega_moe_group):
    case.require_world()
    _run_mega_moe(case, mega_moe_group, verbose=False,
                  reference_tokens=0 if case.name.startswith("smoke") else case.reference_tokens,
                  fast_math=fast_math)


@pytest.mark.parametrize("case", CASES[:3], ids=lambda case: case.name)
def test_mega_moe_physical_experts(case, mega_moe_group):
    case.require_world()
    if case.experts // dist.get_world_size() < 2:
        pytest.skip("requires multiple local physical experts")
    _run_mega_moe(replace(case, routing="random"), mega_moe_group,
                  replicas=True, edge_cases=True, verbose=False)


def test_mega_moe_smoke_grouped_workspace(mega_moe_group):
    """Exercise the grouped scheduler's buffer-owned scratch at 128+ experts."""
    if dist.get_world_size() > 2:
        pytest.skip("grouped scratch boundary is covered by EP1/EP2")
    case = replace(CASES[0], experts=256, tokens=(1, 33))
    _run_mega_moe(case, mega_moe_group, verbose=False)


# Standalone torchrun entry; pytest uses the same driver and case table.


def main():
    if "--benchmark" in sys.argv[1:]:
        from bench_mega_moe import main as benchmark_main
        benchmark_main([arg for arg in sys.argv[1:] if arg != "--benchmark"])
        return
    cases = CASES
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=[c.name for c in cases], default=cases[0].name)
    parser.add_argument("--num-tokens", "--tokens", dest="tokens", type=int, nargs="+",
                        help="tokens per rank; defaults to the case table")
    parser.add_argument("--num-max-tokens-per-rank", "--capacity", dest="capacity", type=int,
                        help="workspace token capacity")
    parser.add_argument("--routing", choices=("round_robin", "random", "skewed"), default="round_robin")
    parser.add_argument("--fast-math", type=int, choices=(0, 1), default=1,
                        help="approximate GEMM1 activation math (default: 1); 0 uses expf/division/tanhf")
    parser.add_argument("--reference-tokens", type=int, default=0, help="check first N tokens/rank; 0 checks all")
    parser.add_argument("--replicas", action="store_true", help="test separately allocated replica weights")
    parser.add_argument("--edge-cases", action="store_true",
                        help="include empty ranks, masked routes and staggered cross-rank replay")
    args = parser.parse_args()
    case = next(c for c in cases if c.name == args.case)
    case = replace(case, tokens=tuple(args.tokens) if args.tokens is not None else case.tokens,
                   capacity=args.capacity if args.capacity is not None else case.capacity, routing=args.routing)
    if case.capacity < 1 or any(n < 0 or n > case.capacity for n in case.tokens):
        parser.error("tokens must be nonnegative and no larger than a positive capacity")
    if args.reference_tokens < 0:
        parser.error("reference-tokens must be nonnegative")
    with _process_group() as group:
        world = dist.get_world_size()
        if world not in case.worlds or case.experts % world:
            raise ValueError(f"{case.name} requires EP{case.worlds}; launched EP{world}")
        if args.replicas and case.experts // world < 2:
            raise ValueError("--replicas requires at least two local experts")
        _run_mega_moe(case, group, reference_tokens=args.reference_tokens,
                      edge_cases=args.edge_cases, replicas=args.replicas,
                      fast_math=bool(args.fast_math))


if __name__ == "__main__":
    main()
