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
import torch.distributed._symmetric_memory as symm_mem

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deep_gemm as dg
from moe_test_utils import (
    MODEL_TOKENS, CombineBarrier, MoECase, activation, gather,
    interleave_gate_up, make_inputs, moe_group, quant_w4a16_mxfp4, run_cli, run_forward,
    split_expert_weights,
)

CASES = (
    MoECase("smoke", 256, 128, 32, 2, 64, tokens=(1, 0, 33, 8), activation="situ",
            worlds=(1, 2, 4, 8, 16, 32), precision="w4a16"),
    MoECase("kimi_k3", 3584, 3072, 896, 16, 4096, tokens=(4, 8, 16, 32, 64, 128, *MODEL_TOKENS),
            activation="situ", worlds=(8, 16, 32), precision="w4a16", extra_replicas_per_rank=1),
)


def make_weight(experts, n, k, rng, gate_up=False):
    references, packed, scales = [], [], []
    for _ in range(experts):
        source = torch.randn(1, n, k, device="cuda", dtype=torch.bfloat16, generator=rng) * .125
        reference, q, s = quant_w4a16_mxfp4(source)
        references.append(reference[0])
        if gate_up:
            _, q, s = quant_w4a16_mxfp4(interleave_gate_up(source))
        q, s = dg.preprocess_w4a16_mxfp4((q, s))
        packed.append(q[0])
        scales.append(s[0])
    return torch.stack(references), (torch.stack(packed), torch.stack(scales))


@torch.inference_mode()
def run_case(case, group, *, check=True, reference_tokens=0, edge_cases=False,
             splits=(False,), benchmark=None, verbose=True):
    rank, world = dist.get_rank(), dist.get_world_size()
    reports = []
    for split in splits:
        c = replace(case, experts=case.experts + world * case.extra_replicas_per_rank) if split else case
        epr = c.experts // world
        rng = torch.Generator(device="cuda").manual_seed(20260826 + rank)
        w13_ref, w13 = make_weight(epr, 2 * c.intermediate, c.hidden, rng, gate_up=True)
        w2_ref, w2 = make_weight(epr, c.hidden, c.intermediate, rng)
        g1, r1 = split_expert_weights(w13) if split else (w13, None)
        g2, r2 = split_expert_weights(w2) if split else (w2, None)
        # Allocation is collective: select its profile from the common requested
        # count, before make_inputs creates any rank-local empty/tail cases.
        batches = {}
        for requested in c.tokens:
            allocation = dg.get_mega_moe_tuning_config(
                world, c.experts, c.topk, c.hidden, c.intermediate, requested,
                precision="w4a16", activation="situ")
            bucket = requested if allocation.buffer_block_m is not None else None
            batches.setdefault(bucket, []).append(requested)
        for bucket, counts in batches.items():
            buf = dg.get_symm_buffer_for_mega_moe(
                group, c.experts, c.capacity, c.topk, c.hidden,
                precision="w4a16", intermediate_hidden=2 * c.intermediate,
                num_tokens_per_rank=bucket)
            activated = torch.empty(buf.num_pool, c.intermediate, dtype=torch.bfloat16, device="cuda")
            rows, valid, _, _, _, meta = buf.get_gemm2_metadata()
            combine = symm_mem.empty(c.topk * c.capacity * c.hidden, dtype=torch.bfloat16, device="cuda")
            handle = symm_mem.rendezvous(combine, group=group)
            pointers = [handle.get_buffer(r, (combine.numel(),), torch.bfloat16).data_ptr() for r in range(world)]
            peer_offsets = torch.tensor([p - pointers[rank] for p in pointers], dtype=torch.int64, device="cuda")
            barrier = CombineBarrier(group)
            try:
                for requested in counts:
                    source, ids, weights, count = make_inputs(c, requested, rng, edge_cases)
                    # Keep the collective allocation bucket when this rank is
                    # empty: rank-local tuning must not change the protocol.
                    if edge_cases and bucket is not None and world > 1 and rank == world - 1:
                        count = 0
                        ids.fill_(-1)
                        weights.zero_()
                    tuning = dg.get_mega_moe_tuning_config(
                        world, c.experts, c.topk, c.hidden, c.intermediate, count,
                        precision="w4a16", activation="situ")

                    def forward():
                        dg.mega_moe_pre_dispatch(
                            source[:count], ids[:count], weights[:count], buf, count, precision="w4a16")
                        dg.mega_moe_dispatch_gemm1(
                            buf, g1, activated, rank, count, precision="w4a16", activation="situ",
                            rhs_replicas=r1, tuning=tuning)
                        dg.mega_moe_gemm2_scatter(
                            activated, g2, meta, combine, peer_offsets, rows, valid, rank, c.capacity, c.topk,
                            precision="w4a16", rhs_replicas=r2, tuning=tuning)
                        return dg.mega_moe_combine(
                            combine, weights[:count], count, c.hidden, c.topk, c.capacity,
                            barrier.pointers, rank, world, barrier.done, tuning=tuning)

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
                        c, source[:count], requested, forward, reference, combine=combine,
                        check=check, reference_tokens=reference_tokens, max_relative_l2=.005,
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
