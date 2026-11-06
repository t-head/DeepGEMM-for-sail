"""Complete FP8, FP8 x FP4 and BF16 x FP4 MegaMoE through public entry points.

python -m pytest -q -s tests/test_mega_moe_precisions.py -k smoke
torchrun --standalone --nproc-per-node=2 -m pytest -q -s tests/test_mega_moe_precisions.py -k smoke
"""

from dataclasses import replace
from pathlib import Path
import argparse
import sys
from types import SimpleNamespace
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deep_gemm as dg
from deep_gemm.testing import calc_diff
from quantization import quant_w4
from test_mega_moe import (
    MegaMoECase, MODEL_TOKENS, mega_moe_group,
    _process_group, _all_gather, _relative_l2, _make_routes,
)

CASES = (
    MegaMoECase("smoke_channelwise", 512, 256, 8, 2, 129, tokens=(1, 0, 33, 129, 8),
               precision="fp8_channelwise", routing="random", reference_tokens=129),
    MegaMoECase("smoke_blockwise", 512, 256, 8, 2, 129, tokens=(1, 0, 33, 129, 8),
               precision="fp8_blockwise", routing="random", reference_tokens=129),
    MegaMoECase("smoke_w4a16", 256, 128, 8, 2, 129, tokens=(1, 0, 33, 129, 8),
               activation="situ", precision="w4a16", routing="random", reference_tokens=129),
    MegaMoECase(
        "minimax_m3",
        6144,
        3072,
        128,
        4,
        4096,
        tokens=MODEL_TOKENS,
        activation="swiglu_oai",
        clamp=7.0,
        worlds=(8,),
        precision="fp8_channelwise",
        routing="random",
    ),
    MegaMoECase(
        "glm52", 6144, 2048, 256, 8, 4096, tokens=(32, *MODEL_TOKENS), worlds=(8,),
        precision="fp8_channelwise", routing="random",
    ),
    MegaMoECase(
        "glm53_flash", 4096, 2048, 288, 8, 4096, tokens=MODEL_TOKENS, worlds=(4,),
        precision="fp8_channelwise", routing="random",
    ),
    MegaMoECase(
        "qwen38_flash", 2560, 640, 512, 10, 4096, tokens=MODEL_TOKENS, worlds=(4,),
        precision="fp8_channelwise", routing="random",
    ),
    MegaMoECase(
        "minimax_m3_blockwise",
        7168,
        5120,
        384,
        4,
        4096,
        tokens=(64, *MODEL_TOKENS),
        activation="swiglu_oai",
        clamp=7.0,
        worlds=(8, 16, 32, 64),
        precision="fp8_blockwise",
        routing="random",
    ),
    MegaMoECase(
        "kimi_k3",
        3584,
        3072,
        896,
        16,
        4096,
        tokens=MODEL_TOKENS,
        activation="situ",
        worlds=(8, 16),
        precision="w4a16",
        routing="random",
    ),
    MegaMoECase(
        "kimi_k3_decode",
        3584,
        3072,
        896,
        16,
        128,
        tokens=(4, 8, 16, 32, 64, 128),
        activation="situ",
        worlds=(32,),
        precision="w4a16",
        routing="random",
    ),
    MegaMoECase("smoke_w4a8", 512, 256, 16, 2, 129, tokens=(1, 0, 33, 129, 8),
               precision="w4a8", routing="random", reference_tokens=129),
    MegaMoECase("smoke_w4a8_situ", 512, 512, 16, 2, 129, tokens=(1, 0, 33, 129, 8),
               activation="situ", precision="w4a8", routing="random", reference_tokens=129),
    MegaMoECase("deepseek_v4_pro_w4a8", 7168, 3072, 384, 6, 4096, tokens=(32, *MODEL_TOKENS),
               worlds=(8,), precision="w4a8", routing="random", reference_tokens=8),
    MegaMoECase("kimi_k3_w4a8", 3584, 3072, 896, 16, 4096,
               tokens=(32, 64, 128, *MODEL_TOKENS), activation="situ", worlds=(8, 16, 32),
               precision="w4a8", routing="random", reference_tokens=8),
)


SMOKE_CASES = tuple(c for c in CASES if c.name.startswith("smoke_"))


def check_output(actual, expected, max_diff=0.001):
    # Use grouped FP8's normalized squared-error metric and 0.001 threshold;
    # W4A8 retains its 0.0015 threshold. Relative L2 is a different metric
    # and is diagnostic only.
    difference = float(calc_diff(actual, expected)) if actual.numel() else 0.0
    assert torch.isfinite(actual).all() and difference < max_diff, (difference, _relative_l2(actual, expected))


def quantize_rows(x, blockwise=False):
    """Device reference: FP32 scale reciprocal followed by multiplication.

    Direct division can round on the opposite side of an FP8 midpoint. Keep
    the normalization operation order explicit, matching the device contract.
    """
    original = x.shape
    values = x.float().reshape(*original[:-1], -1, 128) if blockwise else x.float().unsqueeze(-2)
    scale = values.abs().amax(-1).clamp_min(1e-10) / 448.0
    q = (values * scale.reciprocal().unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(original)
    return q, scale


def dequantize(q, s, blockwise=False):
    if blockwise:
        s = s.repeat_interleave(128, -1)
        if s.shape[-2] != q.shape[-2]:
            s = s.repeat_interleave(128, -2)
    return q.float() * s


def make_weights(e, n, k, precision, rng):
    if precision in ("w4a16", "w4a8"):
        source = torch.randn((e, n, k), device="cuda", dtype=torch.bfloat16, generator=rng) * 0.05
        ref, w, s = quant_w4(source, d="w4fa16_mma")
        return (w, s), ref
    # Quantized weights are fixed before transforming their storage layout.
    w = torch.randint(-8, 9, (e, n, k), device="cuda", generator=rng).to(torch.float8_e4m3fn)
    shape = (e, n // 128, k // 128) if precision == "fp8_blockwise" else (e, n, 1)
    s = (torch.rand(shape, device="cuda", generator=rng) + 0.25) / 64
    return (w, s), dequantize(w, s, precision == "fp8_blockwise")


def reference(c, source, ids, weights, w1, w2, rank, tokens, *, situ_beta=4.0, situ_linear_beta=25.0):
    blockwise = c.precision == "fp8_blockwise"
    if c.precision != "w4a16":
        q, s = quantize_rows(source, blockwise)
        source = dequantize(q, s, blockwise)
    result = torch.zeros((source.size(0), source.size(1), c.hidden), device="cuda", dtype=torch.float32)
    local_experts = c.experts // dist.get_world_size()
    for local_expert in range(local_experts):
        expert = rank * local_experts + local_expert
        src, row, slot = (ids == expert).nonzero(as_tuple=True)
        if not row.numel():
            continue
        gate, up = (source[src, row].float() @ w1[local_expert].float().T).bfloat16().float().chunk(2, -1)
        if c.activation == "situ":
            activated = (situ_beta * torch.tanh(gate / situ_beta) * torch.sigmoid(gate)
                         * situ_linear_beta * torch.tanh(up / situ_linear_beta))
        else:
            if c.clamp is not None:
                gate = gate.clamp(max=c.clamp)
                up = up.clamp(-c.clamp, c.clamp)
            activated = (
                (gate * torch.sigmoid(1.702 * gate)) * (up + 1) if c.activation == "swiglu_oai" else F.silu(gate) * up
            )
        activated = activated.bfloat16()
        if c.precision != "w4a16":
            q, s = quantize_rows(activated, blockwise)
            activated = dequantize(q, s, blockwise)
        output = (activated.float() @ w2[local_expert].float().T).bfloat16().float() * weights[src, row, slot, None]
        result.view(-1, c.hidden).index_add_(0, src * source.size(1) + row, output)
    dist.all_reduce(result)
    return result[rank, :tokens].bfloat16()


def make_pipeline(c, group, rng, *, replicas=False, fast_math=True):
    c.require_world()
    world = group.size()
    if replicas and c.experts // world < 2:
        pytest.skip("separate master and replica banks require at least two local slots")
    e = c.experts // world
    raw1, ref1 = make_weights(e, 2 * c.intermediate, c.hidden, c.precision, rng)
    raw2, ref2 = make_weights(e, c.hidden, c.intermediate, c.precision, rng)
    if c.precision == "w4a8":
        w1, w2 = dg.transform_weights_for_w4a8_mega_moe(raw1, raw2)
        buffer = dg.W4A8SymmBuffer(group, c.experts, c.capacity, c.topk, c.hidden, c.intermediate,
                                  activation=c.activation)
        entry = dg.fp8_fp4_mega_moe
    elif c.precision == "w4a16":
        w1, w2 = dg.transform_weights_for_w4a16_mega_moe(raw1, raw2)
        buffer = dg.W4A16SymmBuffer(group, c.experts, c.capacity, c.topk, c.hidden, c.intermediate)
        entry = dg.bf16_fp4_mega_moe
    else:
        quantization = c.precision.removeprefix("fp8_")
        w1, w2 = dg.transform_weights_for_fp8_mega_moe(raw1, raw2, quantization)
        buffer = dg.FP8SymmBuffer(
            group,
            c.experts,
            c.capacity,
            c.topk,
            c.hidden,
            c.intermediate,
            quantization=quantization,
            activation=c.activation,
        )
        entry = dg.fp8_fp8_mega_moe
    options = dict(activation=c.activation, activation_clamp=c.clamp, fast_math=fast_math)
    if replicas:
        assert e >= 2
        if c.precision == "w4a8":
            # Separate allocations may need different exponent rebasing.
            # Shift replica raw scales before independently preparing each bank.
            for raw, ref, shift in ((raw1, ref1, -16), (raw2, ref2, 16)):
                raw[1][-1:] = (raw[1][-1:].int() + shift).to(torch.uint8)
                ref[-1:] *= 2.0 ** shift
            master1, master2 = dg.transform_weights_for_w4a8_mega_moe(
                tuple(t[:-1].contiguous() for t in raw1), tuple(t[:-1].contiguous() for t in raw2))
            replica1, replica2 = dg.transform_weights_for_w4a8_mega_moe(
                tuple(t[-1:].contiguous() for t in raw1), tuple(t[-1:].contiguous() for t in raw2))
            assert master1[2] != replica1[2] or master2[2] != replica2[2]
            w1, w2 = master1, master2
        else:
            replica1 = tuple(t[-1:].contiguous() for t in w1)
            replica2 = tuple(t[-1:].contiguous() for t in w2)
            w1 = tuple(t[:-1].contiguous() for t in w1)
            w2 = tuple(t[:-1].contiguous() for t in w2)
        options.update(l1_weights_replicas=replica1, l2_weights_replicas=replica2)

    def forward(source, ids, weights, y, stats=None):
        dg.mega_moe_pre_dispatch(source, ids, weights, buffer)
        entry(y, w1, w2, buffer, stats, **options)
        return y

    return SimpleNamespace(buffer=buffer, raw1=raw1, raw2=raw2, ref1=ref1, ref2=ref2,
                           l1_weights=w1, l2_weights=w2, options=options, entry=entry, forward=forward)


def run_case(c, group, *, replicas=False, fast_math=True, reference_tokens=None):
    rank, world = group.rank(), group.size()
    max_diff = 0.0015 if c.precision == "w4a8" else 0.001
    rng = torch.Generator(device="cuda").manual_seed(20261030 + rank)
    pipeline = make_pipeline(c, group, rng, replicas=replicas, fast_math=fast_math)
    buffer, refs1, refs2 = pipeline.buffer, pipeline.ref1, pipeline.ref2
    e = c.experts // world
    source = torch.randn((c.capacity, c.hidden), device="cuda", dtype=torch.bfloat16, generator=rng) * 0.1
    stats = torch.zeros(e, device="cuda", dtype=torch.int32)
    try:
        for iteration, requested in enumerate(c.tokens):
            tokens = 0 if world > 1 and rank == world - 1 and iteration == 1 else requested
            ids = torch.full((c.capacity, c.topk), -1, device="cuda", dtype=torch.int64)
            routes = _make_routes(c, tokens, rng)
            ids[:tokens] = routes
            weights = torch.rand((c.capacity, c.topk), device="cuda", generator=rng)
            weights[:, 0] = 0
            if iteration % 2:
                ids[:tokens, 0] = -1
            y = torch.empty((tokens, c.hidden), device="cuda", dtype=torch.bfloat16)

            def forward():
                return pipeline.forward(source[:tokens], ids[:tokens], weights[:tokens], y, stats)

            before = stats.clone()
            forward()
            torch.cuda.synchronize()
            all_ids = _all_gather(ids)
            reference_limit = c.reference_tokens if reference_tokens is None else reference_tokens
            checked = min(c.capacity, reference_limit) if reference_limit else c.capacity
            expected = reference(
                c,
                _all_gather(source[:checked]),
                all_ids[:, :checked],
                _all_gather(weights[:checked]),
                refs1,
                refs2,
                rank,
                min(tokens, checked),
            )
            check_output(y[:checked], expected, max_diff)
            assert torch.isfinite(y).all()
            counts = torch.bincount(all_ids[all_ids >= 0], minlength=c.experts)[rank * e : (rank + 1) * e]
            torch.testing.assert_close(stats - before, counts.int(), rtol=0, atol=0)
            # Capture and replay the public call, then change routes in-place.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward()
            graph.replay()
            torch.cuda.synchronize()
            check_output(y[:checked], expected, max_diff)
            ids[:tokens] = torch.where(ids[:tokens] < 0, ids[:tokens], (ids[:tokens] + 1) % c.experts)
            graph.replay()
            torch.cuda.synchronize()
            changed_ids = _all_gather(ids)
            expected = reference(
                c,
                _all_gather(source[:checked]),
                changed_ids[:, :checked],
                _all_gather(weights[:checked]),
                refs1,
                refs2,
                rank,
                min(tokens, checked),
            )
            check_output(y[:checked], expected, max_diff)
            replayed = y.clone()
            forward()
            torch.cuda.synchronize()
            torch.testing.assert_close(y, replayed, rtol=0, atol=0)
            weights.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.count_nonzero(y) == 0
    finally:
        buffer.destroy()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_mega_moe_precision(case, mega_moe_group):
    run_case(case, mega_moe_group)


@pytest.mark.parametrize("case", SMOKE_CASES, ids=lambda c: c.name)
def test_mega_moe_precision_replicas(case, mega_moe_group):
    run_case(replace(case, tokens=(1, 33)), mega_moe_group, replicas=True, fast_math=False)


def test_mega_moe_channelwise_quantization_reuse(mega_moe_group):
    # I=512 exercises cached vector quantization. Dense and sparse calls share
    # one allocation while the row lookup and selected G1 tile can change.
    run_case(replace(CASES[0], intermediate=512, capacity=4096,
                     tokens=(4096, 33, 0, 129, 8), reference_tokens=32), mega_moe_group)


def test_mega_moe_channelwise_worklist_reuse(mega_moe_group):
    # Reuse G1's valid-tile list across token counts that select different G2
    # tiles without changing the G1 tile. I=640 also covers quantization tails.
    case = replace(CASES[0], experts=128 * mega_moe_group.size(), topk=10,
                   intermediate=640, tokens=(1, 33, 0, 129, 8), worlds=(mega_moe_group.size(),))
    for replicas in (False, True):
        run_case(case, mega_moe_group, replicas=replicas)


def test_mega_moe_w4a16_complete_tiles(mega_moe_group):
    # H=768 cannot use a 512-column scatter tile. The narrow GEMM1 must also
    # select complete gate/up tiles rather than inherit ordinary GEMM tails.
    run_case(replace(CASES[2], hidden=768, tokens=(96, 33, 0, 8)), mega_moe_group)


def test_mega_moe_w4a8_complete_tiles(mega_moe_group):
    # Widths of one and three full scatter vectors, plus the block-info
    # scheduler at its 128-local-expert transition. Reuse the public workspace.
    base = next(c for c in SMOKE_CASES if c.name == "smoke_w4a8")
    for hidden, intermediate in ((256, 128), (768, 384)):
        run_case(replace(base, hidden=hidden, intermediate=intermediate,
                         experts=128 * mega_moe_group.size(), worlds=(mega_moe_group.size(),),
                         tokens=(1, 33, 0, 129, 8)), mega_moe_group)


def test_mega_moe_w4a8_activation_reuse(mega_moe_group):
    # Keep geometry and weights fixed so only source-specialized activation
    # parameters distinguish the generated kernels within this process.
    case = next(c for c in SMOKE_CASES if c.name == "smoke_w4a8_situ")
    rng = torch.Generator(device="cuda").manual_seed(20261106 + mega_moe_group.rank())
    pipeline = make_pipeline(case, mega_moe_group, rng, fast_math=False)
    swiglu_buffer = dg.W4A8SymmBuffer(mega_moe_group, case.experts, case.capacity, case.topk,
                                     case.hidden, case.intermediate, activation="swiglu")
    source = torch.randn(33, case.hidden, device="cuda", dtype=torch.bfloat16, generator=rng) * .5
    ids = _make_routes(case, source.size(0), rng)
    weights = torch.rand(33, case.topk, device="cuda", generator=rng).softmax(-1)
    y = torch.empty_like(source)
    try:
        all_source, all_ids, all_weights = _all_gather(source), _all_gather(ids), _all_gather(weights)
        for activation, beta, linear_beta in (("situ", 4., 25.), ("situ", .25, 25.),
                                               ("situ", .25, .25), ("swiglu", 4., 25.),
                                               ("situ", 4., 25.)):
            buffer = swiglu_buffer if activation == "swiglu" else pipeline.buffer

            def forward():
                dg.mega_moe_pre_dispatch(source, ids, weights, buffer)
                pipeline.entry(y, pipeline.l1_weights, pipeline.l2_weights, buffer,
                               activation=activation, fast_math=False,
                               situ_beta=beta, situ_linear_beta=linear_beta)

            forward()
            expected = reference(replace(case, activation=activation), all_source, all_ids, all_weights,
                                 pipeline.ref1, pipeline.ref2, mega_moe_group.rank(), source.size(0),
                                 situ_beta=beta, situ_linear_beta=linear_beta)
            check_output(y, expected, .0015)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward()
            graph.replay()
            torch.cuda.synchronize()
            check_output(y, expected, .0015)
    finally:
        swiglu_buffer.destroy()
        pipeline.buffer.destroy()


@pytest.mark.parametrize("case", tuple(c for c in SMOKE_CASES if c.precision != "w4a16"), ids=lambda c: c.name)
@pytest.mark.parametrize("clamp", (0.0, 0.125))
@pytest.mark.parametrize("fast_math", (True, False))
def test_mega_moe_fp8_oai(case, clamp, fast_math, mega_moe_group):
    run_case(replace(case, tokens=(1, 33), activation="swiglu_oai", clamp=clamp),
             mega_moe_group, fast_math=fast_math)


@pytest.mark.parametrize("case", SMOKE_CASES, ids=lambda c: c.name)
def test_mega_moe_precision_options(case, mega_moe_group):
    factory = (dg.W4A16SymmBuffer if case.precision == "w4a16" else
               dg.W4A8SymmBuffer if case.precision == "w4a8" else dg.FP8SymmBuffer)
    with pytest.raises(ValueError, match="unsupported activation"):
        factory(
            mega_moe_group,
            case.experts,
            case.capacity,
            case.topk,
            case.hidden,
            case.intermediate,
            activation="swiglu" if case.precision == "w4a16" else "step",
        )
    with pytest.raises(RuntimeError, match="local expert count"):
        factory(mega_moe_group, 256 * mega_moe_group.size(), case.capacity, case.topk, case.hidden, case.intermediate)


@pytest.mark.parametrize("case", SMOKE_CASES, ids=lambda c: c.name)
def test_mega_moe_precision_api(case, mega_moe_group):
    rng = torch.Generator(device="cuda").manual_seed(20261101 + mega_moe_group.rank())
    pipeline = make_pipeline(case, mega_moe_group, rng)
    source = torch.randn(33, case.hidden, device="cuda", dtype=torch.bfloat16, generator=rng) * .1
    ids = _make_routes(case, 1, rng)
    weights = torch.ones(1, case.topk, device="cuda")
    y = torch.empty_like(source[:1])
    try:
        expected = pipeline.forward(source[:1], ids, weights, y).clone()
        for invalid in ((), (None, None), (pipeline.l1_weights[0].flatten(), pipeline.l1_weights[1])):
            with pytest.raises(ValueError, match="l1_weights must match"):
                pipeline.entry(y, invalid, pipeline.l2_weights, pipeline.buffer, **pipeline.options)

        # The explicit prefix count has identical semantics for every precision:
        # hidden states and routing inputs only need to contain that many rows.
        dg.mega_moe_pre_dispatch(source, ids, weights, pipeline.buffer, num_tokens=1)
        pipeline.entry(y, pipeline.l1_weights, pipeline.l2_weights, pipeline.buffer, **pipeline.options)
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="num_tokens"):
            dg.mega_moe_pre_dispatch(source, ids, weights, pipeline.buffer, num_tokens=2)
        torch.testing.assert_close(pipeline.forward(source[:1], ids, weights, y), expected, rtol=0, atol=0)

        raw_weight, raw_scale = pipeline.raw1
        odd_weight = torch.cat((raw_weight, raw_weight[:, :1]), dim=1)
        odd_scale = (raw_scale if case.precision == "fp8_blockwise"
                     else torch.cat((raw_scale, raw_scale[:, :1]), dim=1))
        transform = (dg.transform_weights_for_w4a8_mega_moe if case.precision == "w4a8" else
                     dg.transform_weights_for_w4a16_mega_moe if case.precision == "w4a16"
                     else dg.transform_weights_for_fp8_mega_moe)
        options = {} if case.precision in ("w4a16", "w4a8") else dict(quantization=case.precision.removeprefix("fp8_"))
        for invalid in ((), (None, None)):
            with pytest.raises(TypeError):
                transform(invalid, pipeline.raw2, **options)
        with pytest.raises(ValueError, match="rank-3"):
            transform((raw_weight.flatten(), raw_scale), pipeline.raw2, **options)
        with pytest.raises(ValueError, match="matching"):
            transform((odd_weight, odd_scale), pipeline.raw2, **options)
    finally:
        pipeline.buffer.destroy()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--benchmark" in argv:
        from bench_mega_moe import main as benchmark_main
        benchmark_main([arg for arg in argv if arg != "--benchmark"], cases=CASES)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[c.name for c in CASES], default=CASES[0].name)
    parser.add_argument("--num-tokens", "--tokens", dest="tokens", type=int, nargs="+")
    parser.add_argument("--num-max-tokens-per-rank", "--capacity", dest="capacity", type=int)
    parser.add_argument("--routing", choices=("round_robin", "random", "skewed"), default="round_robin")
    parser.add_argument("--fast-math", type=int, choices=(0, 1), default=1)
    parser.add_argument("--reference-tokens", type=int, default=0, help="check first N tokens/rank; 0 checks all")
    parser.add_argument("--replicas", action="store_true")
    args = parser.parse_args(argv)
    case = next(c for c in CASES if c.name == args.case)
    case = replace(case, tokens=tuple(args.tokens) if args.tokens is not None else case.tokens,
                   capacity=args.capacity if args.capacity is not None else case.capacity, routing=args.routing)
    if case.capacity < 1 or any(n < 0 or n > case.capacity for n in case.tokens):
        parser.error("tokens must be nonnegative and no larger than a positive capacity")
    if args.reference_tokens < 0:
        parser.error("reference-tokens must be nonnegative")
    with _process_group() as group:
        world = group.size()
        if world not in case.worlds or case.experts % world:
            raise ValueError(f"{case.name} requires EP{case.worlds}; launched EP{world}")
        if args.replicas and case.experts // world < 2:
            raise ValueError("--replicas requires at least two local experts")
        run_case(case, group, replicas=args.replicas, fast_math=bool(args.fast_math),
                 reference_tokens=args.reference_tokens)


if __name__ == "__main__":
    main()
