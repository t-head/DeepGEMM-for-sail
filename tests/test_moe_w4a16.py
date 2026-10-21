r"""W4A16 end-to-end correctness and benchmark: BF16 dispatch -> SiTU -> scatter/combine.

    python tests/test_moe_w4a16.py --case smoke --tokens 1 0 33 --replicas
    # Launch with EP8, EP16 or EP32 for Kimi K3:
    torchrun ... tests/test_moe_w4a16.py --case kimi_k3 --tokens 1024 --benchmark
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
    make_inputs, moe_group, quant_w4a16_mxfp4, run_cli, run_forward,
    split_expert_weights,
)

CASES = (
    MoECase("smoke", 256, 128, 32, 2, 64, tokens=(1, 0, 33, 8), activation="situ",
            worlds=(1, 2, 4, 8, 16, 32), precision="w4a16"),
    MoECase("kimi_k3", 3584, 3072, 896, 16, 4096, tokens=(4, 8, 16, 32, 64, 128, *MODEL_TOKENS),
            activation="situ", worlds=(8, 16, 32), precision="w4a16", extra_replicas_per_rank=1),
)


def make_weight(experts, n, k, rng):
    references, packed, scales = [], [], []
    for _ in range(experts):
        source = torch.randn(1, n, k, device="cuda", dtype=torch.bfloat16, generator=rng) * .125
        reference, q, s = quant_w4a16_mxfp4(source)
        references.append(reference[0])
        packed.append(q[0])
        scales.append(s[0])
    return torch.stack(references), (torch.stack(packed), torch.stack(scales))


@torch.inference_mode()
def run_case(case, group, *, check=True, reference_tokens=0, edge_cases=False,
             splits=(False,), benchmark=None, verbose=True):
    rank, world = dist.get_rank(), dist.get_world_size()
    if check and case.name.startswith("smoke"):
        check_workspace_options(case, group)
    reports = []
    for split in splits:
        c = replace(case, experts=case.experts + world * case.extra_replicas_per_rank) if split else case
        epr = c.experts // world
        rng = torch.Generator(device="cuda").manual_seed(20260826 + rank)
        w13_ref, w13 = make_weight(epr, 2 * c.intermediate, c.hidden, rng)
        w2_ref, w2 = make_weight(epr, c.hidden, c.intermediate, rng)
        w13, w2 = dg.transform_weights_for_mega_moe(w13, w2, precision="w4a16")
        g1, r1 = split_expert_weights(w13) if split else (w13, None)
        g2, r2 = split_expert_weights(w2) if split else (w2, None)
        # Model cases use their common input count as the allocation bucket.
        # Smoke cases reuse one workspace across changing token counts.
        batches = ({None: c.tokens} if c.name.startswith("smoke") else
                   {requested or None: (requested,) for requested in c.tokens})
        for bucket, counts in batches.items():
            buf = dg.get_symm_buffer_for_mega_moe(
                group, c.experts, c.capacity, c.topk, c.hidden,
                precision="w4a16", intermediate_hidden=c.intermediate,
                num_tokens_per_rank=bucket)
            try:
                for requested in counts:
                    source, ids, weights, count = make_inputs(c, requested, rng, edge_cases)
                    # Keep the collective allocation bucket when this rank is
                    # empty: rank-local tuning must not change the protocol.
                    if edge_cases and bucket is not None and world > 1 and rank == world - 1:
                        count = 0
                        ids.fill_(-1)
                        weights.zero_()
                    output = torch.empty(count, c.hidden, dtype=torch.bfloat16, device="cuda")

                    def forward():
                        dg.mega_moe_pre_dispatch(source[:count], ids[:count], weights[:count], buf)
                        dg.bf16_fp4_mega_moe(output, g1, g2, buf, w1_replicas=r1, w2_replicas=r2)
                        return output

                    if check and c.name.startswith("smoke") and not split and requested == c.tokens[0]:
                        check_complete_entry(
                            dg.bf16_fp4_mega_moe, output, g1, g2, buf, source[:count], ids[:count], weights[:count],
                            activation=c.activation, activation_limit=c.clamp or None)

                    def reference(checked):
                        x = gather(source[:checked].contiguous())
                        all_ids = gather(ids[:checked].contiguous())
                        all_weights = gather(weights[:checked].contiguous())
                        expected = torch.zeros(world * checked, c.hidden, device="cuda")
                        for expert in range(epr):
                            src, tok, slot = torch.where(all_ids == rank * epr + expert)
                            if not src.numel():
                                continue
                            projection = (x[src, tok].float() @ w13_ref[expert].float().t()).bfloat16()
                            a = activation(projection, "situ").bfloat16()
                            output = (a.float() @ w2_ref[expert].float().t()).bfloat16()
                            expected.index_add_(0, src * checked + tok,
                                                output.float() * all_weights[src, tok, slot, None])
                        dist.all_reduce(expected)
                        return expected.view(world, checked, c.hidden)[rank, :min(count, checked)].bfloat16()

                    reports.append(run_forward(
                        c, source[:count], requested, forward, reference, combine=buf._combine, routing_weights=weights[:count], routing_ids=ids[:count],
                        workspace=buf, check=check, reference_tokens=reference_tokens, max_relative_l2=.005,
                        replicas=split, edge_cases=edge_cases, benchmark=benchmark, verbose=verbose))
            finally:
                buf.destroy()
    return reports


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_pipeline(case, moe_group):
    case.require_world()
    run_case(case, moe_group, edge_cases=True, splits=(False, True), verbose=False,
             reference_tokens=0 if case.name == "smoke" else case.reference_tokens)


if __name__ == "__main__":
    run_cli(CASES, run_case, __doc__)
