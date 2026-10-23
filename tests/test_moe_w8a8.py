r"""FP8 end-to-end correctness and benchmark, with BF16 scatter/combine.

    python tests/test_moe_w8a8.py --case smoke_blockwise --tokens 17 --benchmark
    torchrun --standalone --nproc-per-node=4 tests/test_moe_w8a8.py \
        --case glm53_flash --tokens 1024 2048 4096 --benchmark
"""

from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch
import torch.distributed as dist

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deep_gemm as dg
from moe_test_utils import (
    check_complete_entry, check_workspace_options, MODEL_TOKENS, MoECase, activation, gather,
    make_inputs, moe_group, quantize_fp8_blocks, quantize_fp8_rows,
    run_cli, run_forward, split_expert_weights,
)

CHANNELWISE_CASES = tuple(replace(c, precision="fp8_channelwise") for c in (
    MoECase("smoke_channelwise", 512, 256, 16, 2, 129, tokens=(1, 0, 33, 129, 8), worlds=(1, 2, 4, 8)),
    MoECase("minimax_m3", 6144, 3072, 128, 4, 4096, tokens=MODEL_TOKENS,
            activation="swiglu_oai", clamp=7.0, worlds=(8,)),
    MoECase("qwen38_flash", 2560, 640, 512, 10, 8192, tokens=MODEL_TOKENS,
            worlds=(4,), reference_tokens=64),
    MoECase("glm52", 6144, 2048, 256, 8, 4096, tokens=(32, *MODEL_TOKENS), worlds=(8,)),
    MoECase("glm53_flash", 4096, 2048, 288, 8, 4096, tokens=MODEL_TOKENS,
            clamp=10.0, input_scale=16, worlds=(4,)),
))
BLOCKWISE_CASES = (
    MoECase("smoke_blockwise", 256, 128, 8, 2, 129, tokens=(1, 0, 33, 129, 8), precision="fp8_blockwise"),
    MoECase("minimax_m3_blockwise", 7168, 5120, 384, 4, 4096, tokens=(64, *MODEL_TOKENS),
            activation="swiglu_oai", clamp=7.0, worlds=(8, 16, 32, 64), precision="fp8_blockwise"),
)
CASES = CHANNELWISE_CASES + BLOCKWISE_CASES


def make_weights(experts, n, k, rng, blockwise, amplitude):
    if blockwise:
        weights = torch.randint(-16, 17, (experts, n, k), device="cuda", generator=rng).to(torch.float8_e4m3fn)
        scales = (torch.rand(experts, n // 128, k // 128, device="cuda", generator=rng) + .25) / (k**.5 * 4)
        return weights, scales
    weights, scales = [], []
    for _ in range(experts):
        source = torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=rng) * amplitude
        source *= 2.0 ** (torch.arange(n, device="cuda")[:, None] % 3 - 1)
        q, s = quantize_fp8_rows(source)
        weights.append(q)
        scales.append(s)
    return torch.stack(weights), torch.stack(scales)


def dequantize(weights, scales, blockwise):
    if blockwise:
        scales = scales.repeat_interleave(128, -1)
        if scales.shape[-2] != weights.shape[-2]:
            scales = scales.repeat_interleave(128, -2)
    return weights.float() * scales


@torch.inference_mode()
def run_case(case, group, *, check=True, reference_tokens=0, edge_cases=False,
             splits=(False,), benchmark=None, verbose=True, block_m=None):
    c = case
    rank, world = dist.get_rank(), dist.get_world_size()
    if check and case.name.startswith("smoke"):
        check_workspace_options(case, group)
    epr, blockwise = c.experts // world, c.precision == "fp8_blockwise"
    quantize = quantize_fp8_blocks if blockwise else quantize_fp8_rows
    rng = torch.Generator(device="cuda").manual_seed(20260909 + rank)
    w1, s1 = make_weights(epr, 2 * c.intermediate, c.hidden, rng, blockwise, .125)
    w2, s2 = make_weights(epr, c.hidden, c.intermediate, rng, blockwise, .015625)
    packed_g1, packed_g2 = dg.transform_weights_for_mega_moe((w1, s1), (w2, s2), precision=c.precision)
    buf = dg.get_symm_buffer_for_mega_moe(
        group, c.experts, c.capacity, c.topk, c.hidden,
        precision=c.precision, intermediate_hidden=c.intermediate, block_m=block_m)
    reports = []
    try:
        for split in splits:
            g1, r1 = split_expert_weights(packed_g1) if split else (packed_g1, None)
            g2, r2 = split_expert_weights(packed_g2) if split else (packed_g2, None)
            for token_index, requested in enumerate(c.tokens):
                source, ids, weights, count = make_inputs(c, requested, rng, edge_cases)
                if (check and edge_cases and not blockwise and world > 1
                        and not c.name.startswith("smoke") and rank == world - 1):
                    # Empty peers use fallback tuning alongside the other ranks,
                    # without changing the symmetric allocation.
                    count = 0
                    ids.fill_(-1)
                    weights.zero_()
                output = torch.empty(count, c.hidden, dtype=torch.bfloat16, device="cuda")

                def forward():
                    dg.mega_moe_pre_dispatch(source[:count], ids[:count], weights[:count], buf)
                    dg.fp8_fp8_mega_moe(
                        output, g1, g2, buf, activation=c.activation, activation_limit=c.clamp or None,
                        w1_replicas=r1, w2_replicas=r2)
                    return output

                if check and c.name.startswith("smoke") and not split and token_index == 0:
                    check_complete_entry(
                        dg.fp8_fp8_mega_moe, output, g1, g2, buf, source[:count], ids[:count], weights[:count],
                        activation=c.activation, activation_limit=c.clamp or None)

                def reference(checked):
                    q, s = quantize(source[:checked])
                    x = gather(dequantize(q, s, blockwise))
                    all_ids = gather(ids[:checked].contiguous())
                    all_weights = gather(weights[:checked].contiguous())
                    expected = torch.zeros(world * checked, c.hidden, device="cuda")
                    for expert in range(epr):
                        src, tok, slot = torch.where(all_ids == rank * epr + expert)
                        if not src.numel():
                            continue
                        projection = (x[src, tok] @ dequantize(w1[expert], s1[expert], blockwise).t()).bfloat16()
                        a = activation(projection, c.activation, c.clamp).bfloat16()
                        q, s = quantize(a)
                        output = (dequantize(q, s, blockwise)
                                  @ dequantize(w2[expert], s2[expert], blockwise).t()).bfloat16()
                        expected.index_add_(0, src * checked + tok,
                                            output.float() * all_weights[src, tok, slot, None])
                    dist.all_reduce(expected)
                    return expected.view(world, checked, c.hidden)[rank, :min(count, checked)].bfloat16()

                # Match the FP8 grouped contiguous/masked GEMM error threshold.
                reports.append(run_forward(
                    c, source[:count], requested, forward, reference, combine=buf._combine, routing_weights=weights[:count], routing_ids=ids[:count],
                    workspace=buf, check=check, reference_tokens=reference_tokens, max_diff=.001, replicas=split,
                    edge_cases=edge_cases, benchmark=benchmark, verbose=verbose))
    finally:
        buf.destroy()
    return reports


@pytest.mark.parametrize("case, block_m",
                         [pytest.param(c, None, id=c.name) for c in CASES] +
                         [pytest.param(c, bm, id=f"{c.name}_bm{bm}")
                          for c in CHANNELWISE_CASES if c.name.startswith("smoke")
                          for bm in (128, 256)])
def test_pipeline(case, block_m, moe_group):
    case.require_world()
    if case.precision == "fp8_channelwise":
        # Return to the first shape after changing token counts and G1 tiles.
        case = replace(case, tokens=(*case.tokens, case.tokens[0]))
    splits = (False, True) if case.experts // dist.get_world_size() > 1 else (False,)
    run_case(case, moe_group, edge_cases=True, splits=splits, verbose=False,
             reference_tokens=0 if case.name.startswith("smoke") else case.reference_tokens,
             block_m=block_m)


if __name__ == "__main__":
    run_cli(CASES, run_case, __doc__)
