r"""W4A4 routed MoE example, correctness test and end-to-end benchmark.

Inputs: BF16 hidden states plus router-produced top-k IDs and weights.
Flow: pre-dispatch -> dispatch/GEMM1/activation/FP4 -> GEMM2/BF16 scatter -> combine.

Run from the built checkout (use torchrun for multiple ranks):
    python tests/test_moe_w4a4.py --case smoke_swiglu --tokens 33
    torchrun --standalone --nproc-per-node=8 tests/test_moe_w4a4.py \
        --case deepseek_v4_pro --tokens 1024 --benchmark

W4A4MoE.forward() is shared by the CLI and pytest. Setup, independent Torch
references and timing are separate from this complete kernel call sequence.
"""

from pathlib import Path
import sys

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# Direct execution should exercise this checkout, even if another is installed.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deep_gemm as dg
from moe_test_utils import (
    MODEL_TOKENS, CombineBarrier, MoECase, activation, run_forward, run_cli,
    clear_dispatch_counts, compile_kernels, dequantize_fp4_torch, gather, interleave_gate_up,
    moe_group, quantize_fp4_rne, split_expert_weights,
)

# FP4 dispatch copies scales in 16-byte chunks, so smoke H is at least 512.
CASES = (
    MoECase("smoke_swiglu", 512, 256, 8, 2, 64, tokens=(1, 0, 33, 8), clamp=2.0),
    MoECase("smoke_step", 512, 256, 8, 2, 64, tokens=(1, 0, 33, 8), activation="step", clamp=2.0),
    MoECase("smoke_situ", 512, 256, 8, 2, 64, tokens=(1, 0, 33, 8), activation="situ"),
    MoECase("deepseek_v4_pro", 7168, 3072, 384, 6, 8192, tokens=MODEL_TOKENS, clamp=10.0, worlds=(8,)),
    MoECase("deepseek_v4_flash", 4096, 2048, 256, 6, 4096, tokens=MODEL_TOKENS, clamp=10.0, worlds=(4,)),
    MoECase("glm52", 6144, 2048, 256, 8, 4096, tokens=MODEL_TOKENS, worlds=(8,)),
    MoECase("kimi_k3", 3584, 3072, 896, 16, 4096, tokens=MODEL_TOKENS, activation="situ", worlds=(8, 16)),
    MoECase("qwen38_2_4t", 8192, 2048, 512, 10, 4096, tokens=MODEL_TOKENS, worlds=(8, 16)),
    MoECase("mimo_v26_pro", 6144, 2048, 384, 8, 8192, tokens=(*MODEL_TOKENS, 8192), worlds=(8,)),
)


def _pack_gemm1_scales(scales, block_n=256):
    # Each N tile is K-major; a whole-expert transpose mixes tile scales.
    chunks = []
    for expert in scales:
        processed = dg.preprocess_mxfp4_scales(expert)
        chunks.extend(tile.t().contiguous().flatten() for tile in processed.split(block_n))
    return torch.cat(chunks).view(scales.shape[0], -1)


class W4A4MoE:
    """Reusable example workspace for one routed MoE layer.

    Pass local expert weights as uint8 (packed FP4 [E, N, K/2], raw E8M0
    scales [E, N, K/32]) tuples; GEMM1 rows are [gate; up]. Call set_inputs() before
    compile()/forward(), reusing the workspace across token counts. All ranks,
    including empty ranks, must invoke forward() the same number of times.
    """

    def __init__(self, case, group, gemm1_weights, gemm2_weights):
        self.case = case
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.epr = case.experts // self.world
        self.reference_weights = gemm1_weights, gemm2_weights
        self.w1, self.s1 = (interleave_gate_up(t) for t in gemm1_weights)
        self.s1_packed = _pack_gemm1_scales(self.s1)
        self.w2, s2 = gemm2_weights
        self.s2_packed = dg.preprocess_mxfp4_scales(s2)
        self.buf = dg.get_symm_buffer_for_mega_moe(
            group, case.experts, case.capacity, case.topk, case.hidden,
            precision="w4a4", intermediate_hidden=2 * case.intermediate)
        self.aq = torch.empty(self.buf.num_pool, case.intermediate // 2, dtype=torch.uint8, device="cuda")
        self.asc = torch.empty(case.intermediate // 64, self.buf.num_pool, dtype=torch.uint16, device="cuda").t()
        (self.rows, self.valid, self.offsets, self.src_rows, self.src_offsets,
         self.token_meta) = self.buf.get_gemm2_metadata()
        self.combine = symm_mem.empty(case.topk * case.capacity * case.hidden, dtype=torch.bfloat16, device="cuda")
        self.combine_handle = symm_mem.rendezvous(self.combine, group=group)
        pointers = [self.combine_handle.get_buffer(r, (self.combine.numel(),), torch.bfloat16).data_ptr()
                    for r in range(self.world)]
        self.peer_offsets = torch.tensor([p - pointers[self.rank] for p in pointers], dtype=torch.int64)
        self.barrier = CombineBarrier(group)
        self.combine.zero_()
        clear_dispatch_counts(self.buf, case.experts)
        torch.cuda.synchronize()
        dist.barrier()

    def set_inputs(self, hidden_states, topk_ids, topk_weights, *, split_weights=False):
        """Bind inputs and select layouts before graph capture or timing."""
        c = self.case
        count = hidden_states.shape[0]
        if count > c.capacity or hidden_states.shape != (count, c.hidden):
            raise ValueError("hidden_states must be [tokens <= capacity, hidden]")
        if topk_ids.shape != (count, c.topk) or topk_weights.shape != topk_ids.shape:
            raise ValueError("routing IDs and weights must be [tokens, topk]")
        if split_weights and self.epr < 2:
            raise ValueError("split weights require at least two local experts")
        self.source, self.ids, self.weights = hidden_states, topk_ids, topk_weights
        self.count = count
        self.tuning = dg.get_mega_moe_tuning_config(
            self.world, c.experts, c.topk, c.hidden, c.intermediate, count,
            precision="w4a4", activation=c.activation)
        # Pre-dispatch refreshes and masks this view on every replay. Do not copy
        # routing weights here: their values may change after graph capture.
        self.reduce_weights = self.buf.input_topk_weights[:count]
        if split_weights and not hasattr(self, "split_gemm1"):
            self.split_gemm1 = split_expert_weights((self.w1, self.s1_packed))
            self.split_gemm2 = split_expert_weights((self.w2, self.s2_packed))
        self.gemm1, self.gemm1_replicas = (self.split_gemm1 if split_weights
                                          else ((self.w1, self.s1_packed), None))
        self.gemm2, self.gemm2_replicas = (self.split_gemm2 if split_weights
                                          else ((self.w2, self.s2_packed), None))

    def pre_dispatch(self):
        dg.mega_moe_pre_dispatch(
            self.source, self.ids, self.weights, self.buf, self.count, precision="w4a4")

    def dispatch_gemm1(self):
        self.gemm1_block_m = dg.mega_moe_dispatch_gemm1(
            self.buf, self.gemm1, self.aq, self.rank, self.count, precision="w4a4",
            activation=self.case.activation, activation_limit=self.case.clamp or None,
            output_quantization="mxfp4", out_scales=self.asc, rhs_replicas=self.gemm1_replicas,
            tuning=self.tuning)

    def gemm2_scatter(self):
        c = self.case
        dg.mega_moe_gemm2_scatter(
            (self.aq, self.asc), self.gemm2, self.token_meta, self.combine,
            self.peer_offsets, self.rows, self.valid, self.rank, c.capacity, c.topk,
            precision="w4a4", row_offsets=self.src_offsets,
            rhs_replicas=self.gemm2_replicas, tuning=self.tuning)

    def combine_output(self):
        c = self.case
        return dg.mega_moe_combine(
            self.combine, self.reduce_weights, self.count, c.hidden,
            c.topk, c.capacity, self.barrier.pointers, self.rank, self.world,
            self.barrier.done, tuning=self.tuning)

    def forward(self):
        """Complete BF16-input -> BF16-output routed MoE, with device synchronization."""
        self.pre_dispatch()
        self.dispatch_gemm1()
        self.gemm2_scatter()
        clear_dispatch_counts(self.buf, self.case.experts)
        # Every rank participates, even when its returned output is empty.
        return self.combine_output()

    def compile(self):
        """Precompile stages before entering cooperative device barriers."""
        compile_kernels(self.pre_dispatch, self.dispatch_gemm1, self.gemm2_scatter, self.combine_output)
        torch.cuda.synchronize()
        dist.barrier()

    def close(self):
        self.buf.destroy()


def make_weights(experts, n, k, amplitude, rng):
    # Bound temporary quantization memory for model-sized expert matrices.
    packed, scales = [], []
    for _ in range(experts):
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=rng) * amplitude
        q, s = quantize_fp4_rne(weight)
        packed.append(q)
        scales.append(s)
    return torch.stack(packed), torch.stack(scales)


def reference_activation(projection, kind="swiglu", clamp=0.0):
    """Round gate/up, their clamps and the activation output to BF16."""
    projection = projection.bfloat16()
    if kind == "situ":
        return activation(projection, kind).bfloat16()
    gate, up = projection.chunk(2, -1)
    if clamp > 0:
        up = up.clamp(-clamp, clamp)
        if kind != "step":
            gate = gate.clamp(max=clamp)
    gate = torch.nn.functional.silu(gate.float())
    if kind == "step" and clamp > 0:
        gate = gate.clamp(max=clamp)
    return (gate * up.float()).bfloat16()


def reference_output(moe, source, ids, weights, checked):
    """Compute the entire routed MoE from source inputs, without kernel intermediates."""
    c = moe.case
    q, scales = quantize_fp4_rne(source[:checked])
    x = gather(dequantize_fp4_torch(q, scales))
    all_ids, all_weights = gather(ids[:checked].contiguous()), gather(weights[:checked].contiguous())
    expected = torch.zeros(moe.world, checked, c.hidden, device="cuda")
    (w1, s1), (w2, s2) = moe.reference_weights
    for expert in range(moe.epr):
        src, tok, slot = torch.where(all_ids == moe.rank * moe.epr + expert)
        if not src.numel():
            continue
        projection = x[src, tok] @ dequantize_fp4_torch(w1[expert], s1[expert]).t()
        activated = reference_activation(projection, c.activation, c.clamp)
        route_weights = all_weights[src, tok, slot, None]
        q, scales = quantize_fp4_rne(activated)
        output = (dequantize_fp4_torch(q, scales)
                  @ dequantize_fp4_torch(w2[expert], s2[expert]).t()).bfloat16()
        output = output.float() * route_weights
        expected.view(-1, c.hidden).index_add_(0, src * checked + tok, output)
    dist.all_reduce(expected)
    return expected[moe.rank, :min(moe.count, checked)].bfloat16()


@torch.inference_mode()
def run_case(case, group, *, check=True, reference_tokens=0, edge_cases=False,
             splits=(False,), benchmark=None, verbose=True):
    """Shared CLI/pytest runner; references and data preparation are not timed."""
    rank, world = dist.get_rank(), dist.get_world_size()
    rng = torch.Generator(device="cuda").manual_seed(20261016 + rank)
    source = torch.randn(case.capacity, case.hidden, device="cuda", dtype=torch.bfloat16, generator=rng) * .5
    w1 = make_weights(case.experts // world, 2 * case.intermediate, case.hidden, .5, rng)
    w2 = make_weights(case.experts // world, case.hidden, case.intermediate, .125, rng)
    moe = W4A4MoE(case, group, w1, w2)
    reports = []
    token = torch.arange(case.capacity, device="cuda")[:, None]
    slot = torch.arange(case.topk, device="cuda")[None, :]
    try:
        for requested in case.tokens:
            count = 0 if edge_cases and world > 1 and requested == 1 and rank == world - 1 else requested
            ids = ((token * case.topk + slot + rank * 17) % case.experts).contiguous()
            weights = torch.rand(case.capacity, case.topk, device="cuda", generator=rng).softmax(-1)
            ids[count:], weights[count:] = -1, 0
            if edge_cases and count > 1:
                # Invalid IDs must mask even a nonzero router weight.
                ids[0, 0] = -1
                weights[1, 0] = 0
            for split in splits:
                moe.set_inputs(source[:count], ids[:count], weights[:count], split_weights=split)
                report = run_forward(
                    case, source[:count], requested, moe.forward,
                    lambda checked: reference_output(moe, source, ids, weights, checked),
                    combine=moe.combine, check=check, reference_tokens=reference_tokens,
                    max_diff=.003, replicas=split, edge_cases=edge_cases,
                    benchmark=benchmark, verbose=verbose,
                    routing_weights=moe.weights)
                reports.append(report)
    finally:
        moe.close()
    return reports


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_pipeline(case, moe_group):
    case.require_world()
    splits = (False, True) if case.experts // dist.get_world_size() > 1 else (False,)
    run_case(case, moe_group, edge_cases=True, splits=splits, verbose=False,
             reference_tokens=0 if case.name.startswith("smoke") else case.reference_tokens)


if __name__ == "__main__":
    run_cli(CASES, run_case, __doc__)
