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

import argparse
from dataclasses import replace
import json
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
    MODEL_TOKENS, CombineBarrier, MoECase, activation, benchmark_pipeline, calc_diff, check_graph_replay,
    clear_dispatch_counts, compile_kernels, dequantize_fp4_torch, gather, interleave_gate_up,
    moe_group, moe_process_group, quantize_fp4_rne, relative_l2,
)

# FP4 dispatch copies scales in 16-byte chunks, so smoke H is at least 512.
CASES = (
    MoECase("smoke_swiglu", 512, 256, 8, 2, 64, tokens=(1, 0, 33, 8), clamp=2.0),
    MoECase("smoke_step", 512, 256, 8, 2, 64, tokens=(1, 0, 33, 8), activation="step", clamp=2.0),
    MoECase("smoke_situ", 512, 256, 8, 2, 64, tokens=(1, 0, 33, 8), activation="situ"),
    MoECase("deepseek_v4_pro", 7168, 3072, 384, 6, 8192, tokens=MODEL_TOKENS, clamp=10.0, worlds=(8,)),
    MoECase("deepseek_v4_flash", 4096, 2048, 256, 6, 4096, tokens=MODEL_TOKENS, clamp=10.0, worlds=(4,)),
    MoECase("glm52", 6144, 2048, 256, 8, 4096, tokens=MODEL_TOKENS, worlds=(8,)),
    MoECase("kimi_k3", 3584, 3072, 896, 16, 4096, tokens=MODEL_TOKENS, activation="situ", worlds=(16,)),
    MoECase("qwen38_2_4t", 8192, 2048, 512, 10, 4096, tokens=MODEL_TOKENS, worlds=(16,)),
)


def _pack_gemm1_scales(scales, block_n=256):
    # Each N tile is K-major; a whole-expert transpose mixes tile scales.
    chunks = []
    for expert in scales:
        processed = dg.preprocess_mxfp4_scales(expert)
        chunks.extend(tile.t().contiguous().flatten() for tile in processed.split(block_n))
    return torch.cat(chunks).view(scales.shape[0], -1)


def _unpack_output_scales(scales):
    words = scales.to(torch.int32)
    return torch.stack((words & 255, words >> 8), -1).flatten(-2).to(torch.uint8)


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
        self.w1, self.s1 = map(interleave_gate_up, gemm1_weights)
        self.s1_packed = _pack_gemm1_scales(self.s1)
        self.w2, s2 = gemm2_weights
        self.s2_packed = dg.preprocess_mxfp4_scales(s2)
        self.buf = dg.get_symm_buffer_for_mega_moe_dispatch_gemm1(
            group, case.experts, case.capacity, case.topk, case.hidden,
            2 * case.intermediate, gemm1_block_n=256)
        self.aq = torch.empty(self.buf.num_pool, case.intermediate // 2, dtype=torch.uint8, device="cuda")
        self.asc = torch.empty(case.intermediate // 64, self.buf.num_pool, dtype=torch.uint16, device="cuda").t()
        self.rows, self.valid, self.offsets, self.src_rows, self.src_offsets = [
            torch.empty(self.epr, dtype=torch.int32, device="cuda") for _ in range(5)]
        self.packed_meta = torch.empty(self.buf.num_pool, dtype=torch.int64, device="cuda")
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
        # GEMM1 applies routing weights before FP4 quantization. Reduce only
        # sums valid slots; empty ranks still participate in mega_moe_combine.
        self.reduce_weights = (topk_ids >= 0).float()
        self.expected_m = max(1, (count * c.topk + self.epr - 1) // self.epr)
        self.config = dg.get_m_grouped_gemm_fp4_moe_scatter_config(
            self.buf.num_pool, c.hidden, c.intermediate // 2, self.epr,
            prefer_low_latency=True, use_fp8_scatter=False, expected_m=self.expected_m)
        masters = self.epr - int(split_weights)
        self.gemm1 = self.w1[:masters], self.s1_packed[:masters]
        self.gemm1_replicas = (self.w1[masters:], self.s1_packed[masters:]) if split_weights else None
        self.gemm2 = self.w2[:masters], self.s2_packed[:masters]
        self.gemm2_replicas = (self.w2[masters:], self.s2_packed[masters:]) if split_weights else None

    def pre_dispatch(self):
        dg.mega_moe_pre_dispatch(
            self.source, self.ids, self.weights, self.buf, self.count, precision="w4a4")

    def dispatch_gemm1(self):
        self.gemm1_block_m = dg.mega_moe_dispatch_gemm1(
            self.buf, self.gemm1, self.aq, self.rank, self.count, precision="w4a4",
            activation=self.case.activation, activation_limit=self.case.clamp or None,
            output_quantization="mxfp4", out_scales=self.asc, rhs_replicas=self.gemm1_replicas)

    def prepare_rows(self):
        dg.prepare_moe_gemm2_rows(
            self.buf.recv_count_sum, self.rows, self.valid, self.offsets, self.config[1],
            token_meta=self.buf.token_src_metadata, packed_src_info=self.packed_meta,
            src_m_rows=self.src_rows, src_m_offsets=self.src_offsets, src_block_m=self.gemm1_block_m)

    def gemm2_scatter(self):
        c = self.case
        dg.mega_moe_gemm2_scatter(
            (self.aq, self.asc), self.gemm2, self.packed_meta, self.combine,
            self.peer_offsets, self.rows, self.valid, self.rank, c.capacity, c.topk,
            precision="w4a4", configs=self.config, row_offsets=self.src_offsets,
            expected_m=self.expected_m, rhs_replicas=self.gemm2_replicas)

    def combine_output(self):
        c = self.case
        return dg.mega_moe_combine(
            self.combine, self.reduce_weights, self.count, c.hidden,
            c.topk, c.capacity, self.barrier.pointers, self.rank, self.world,
            self.barrier.done)

    def forward(self):
        """Complete BF16-input -> BF16-output routed MoE, with device synchronization."""
        self.pre_dispatch()
        self.dispatch_gemm1()
        self.prepare_rows()
        self.gemm2_scatter()
        clear_dispatch_counts(self.buf, self.case.experts)
        # Every rank participates, even when its returned output is empty.
        return self.combine_output()

    def compile(self):
        """Precompile independent stages concurrently before cooperative device barriers."""
        compile_kernels(self.pre_dispatch, self.dispatch_gemm1, self.gemm2_scatter, self.combine_output)
        compile_kernels(self.prepare_rows)
        torch.cuda.synchronize()
        dist.barrier()

    def close(self):
        self.buf.destroy()


@pytest.mark.parametrize("mode,scale_block_n", (
    ("shift", 0), ("unit", 0),
    ("static", 64), ("static", 128), ("static", 256),
    ("dynamic", 64), ("dynamic", 128), ("dynamic", 256),
))
def test_fp8_combine_pipeline(mode, scale_block_n, moe_group, monkeypatch):
    """Full W4A4 pipeline, followed by its optional FP8 scatter/combine contract."""
    rank, world = dist.get_rank(), dist.get_world_size()
    if world not in (1, 2, 4, 8):
        pytest.skip("FP8 combine smoke covers EP1/2/4/8")
    monkeypatch.setenv("DG_MOE_COMBINE_TOKEN_MAJOR", "1")
    monkeypatch.setenv("DG_MOE_SCATTER_FP8_STATIC_SCALE_ONE", str(int(mode == "unit")))
    monkeypatch.setenv("DG_MOE_SCATTER_FP8_SIDE_SCALE_DYNAMIC", str(int(mode == "dynamic")))
    monkeypatch.setenv("DG_MOE_SCATTER_FP8_EXP_SHIFT", str(int(mode == "shift")))
    # Explicit scale tensors define the contract even if an old environment override disagrees.
    monkeypatch.setenv("DG_MOE_SCATTER_FP8_SIDE_SCALE", "0" if scale_block_n else "1")
    c = MoECase("fp8_combine", 512, 256, 16, 2, 64, tokens=(33,), clamp=2.0)
    rng = torch.Generator(device="cuda").manual_seed(731 + rank)
    w1 = make_weights(c.experts // world, 2 * c.intermediate, c.hidden, .25, rng)
    w2 = make_weights(c.experts // world, c.hidden, c.intermediate, .125, rng)
    moe = W4A4MoE(c, moe_group, w1, w2)
    try:
        x = torch.randn(33, c.hidden, dtype=torch.bfloat16, device="cuda", generator=rng)
        ids = ((torch.arange(33, device="cuda")[:, None] * 2
                + torch.arange(2, device="cuda")[None, :] + rank) % c.experts).contiguous()
        weights = torch.full((33, 2), .5, device="cuda")
        moe.set_inputs(x, ids, weights)
        moe.compile()
        baseline = moe.forward().clone()
        check_correctness(moe, baseline, x, ids, weights, 33)
        fp8 = symm_mem.empty(moe.combine.numel(), dtype=torch.uint8, device="cuda")
        fp8_handle = symm_mem.rendezvous(fp8, group=moe_group)
        pointers = [fp8_handle.get_buffer(r, (fp8.numel(),), torch.uint8).data_ptr() for r in range(world)]
        offsets = torch.tensor([p - pointers[rank] for p in pointers], dtype=torch.int64)
        group = scale_block_n or 256
        tiles = c.hidden // group
        scales = symm_mem.empty(c.topk * c.capacity * tiles, dtype=torch.int64, device="cuda")
        scale_handle = symm_mem.rendezvous(scales, group=moe_group)
        pointers = [scale_handle.get_buffer(r, (scales.numel(),), torch.int64).data_ptr() for r in range(world)]
        scale_offsets = torch.tensor([p - pointers[rank] for p in pointers], dtype=torch.int64)
        extra = dict(combine_scales=scales, scale_offsets=scale_offsets,
                     scale_block_n=scale_block_n) if scale_block_n else {}
        reduce_extra = dict(combine_scales=scales, scale_block_n=scale_block_n) if scale_block_n else {}

        def scatter(config=moe.config, **options):
            dg.mega_moe_gemm2_scatter(
                (moe.aq, moe.asc), moe.gemm2, moe.packed_meta, fp8.view(torch.float8_e4m3fn),
                offsets, moe.rows, moe.valid, rank, c.capacity, c.topk, precision="w4a4",
                configs=config, output_quantization="fp8", row_offsets=moe.src_offsets,
                expected_m=moe.expected_m, **options)

        def pipeline():
            scatter(**extra)
            return dg.mega_moe_combine(
                fp8.view(torch.float8_e4m3fn), moe.reduce_weights, 33, c.hidden, c.topk, c.capacity,
                moe.barrier.pointers, rank, world, moe.barrier.done, **reduce_extra)

        for bad_group in (32, 96, 512):
            with pytest.raises(RuntimeError, match="scale_block_n"):
                scatter(combine_scales=scales, scale_offsets=scale_offsets, scale_block_n=bad_group)
        for bad_n in (64, 128):
            config = list(moe.config)
            config[2] = bad_n
            with pytest.raises(RuntimeError, match="complete 256-column tiles"):
                scatter(config=tuple(config))
        config = list(moe.config)
        config[2] = 512
        with pytest.raises(RuntimeError, match="at most four groups"):
            scatter(config=tuple(config), combine_scales=scales, scale_offsets=scale_offsets, scale_block_n=64)

        compile_kernels(pipeline)
        fp8.fill_(255)
        scales.fill_(-1)
        torch.cuda.synchronize()
        dist.barrier()
        actual = pipeline()
        # Adjacent combine must wait for remote payloads and side scales.
        torch.cuda.synchronize()
        dequant = fp8.view(torch.float8_e4m3fn).view(c.topk, c.capacity, c.hidden)[:, :33].float()
        if scale_block_n:
            raw = scales.view(c.topk, c.capacity, tiles)[:, :33] & 65535
            assert bool((raw != 65535).all()), "scatter did not publish every side scale"
            scale = raw.to(torch.int16).view(torch.bfloat16).float()
            dequant *= scale.repeat_interleave(scale_block_n, -1)
        elif mode == "shift":
            dequant *= 2
        expected = dequant.sum(0).bfloat16()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert relative_l2(actual, baseline) < .06
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = pipeline()
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(replayed, expected, rtol=0, atol=0)
            dist.barrier()
    finally:
        moe.close()


def make_weights(experts, n, k, amplitude, rng):
    # Bound temporary quantization memory for model-sized expert matrices.
    packed, scales = [], []
    for _ in range(experts):
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=rng) * amplitude
        q, s = quantize_fp4_rne(weight)
        packed.append(q)
        scales.append(s)
    return torch.stack(packed), torch.stack(scales)


def check_correctness(moe, result, source, ids, weights, checked):
    """Independent full-chain oracle plus a separate GEMM2/scatter/reduce check."""
    c, buf = moe.case, moe.buf
    x, xs = quantize_fp4_rne(source)
    xd, all_ids, all_weights = gather(dequantize_fp4_torch(x, xs)), gather(ids), gather(weights)
    torch.testing.assert_close(buf.input_token[:moe.count], x[:moe.count], rtol=0, atol=0)
    torch.testing.assert_close(buf.input_sf[:moe.count], xs[:moe.count], rtol=0, atol=0)
    assert buf._gemm1_output is None
    assert result.isfinite().all()
    reference = torch.zeros(moe.world, checked, c.hidden, device="cuda")
    stage_reference = torch.zeros_like(reference)
    (w1, s1), (w2, s2) = moe.reference_weights
    max_gemm1_diff = 0.0
    for expert in range(moe.epr):
        routes = (all_ids == moe.rank * moe.epr + expert).nonzero()
        nr = len(routes)
        assert int(moe.valid[expert]) == nr
        begin, dst = int(moe.src_offsets[expert]), int(moe.offsets[expert])
        meta = buf.token_src_metadata[begin:begin + nr].long()
        assert sorted(map(tuple, meta.tolist())) == sorted(map(tuple, routes.tolist()))
        src, tok, slot = meta.unbind(-1)
        torch.testing.assert_close(moe.packed_meta[dst:dst + nr],
                                   (src << 32) | (slot << 24) | tok, rtol=0, atol=0)
        selected = (tok < checked).nonzero().flatten()
        if not len(selected):
            continue
        src, tok, slot = (v[selected] for v in (src, tok, slot))
        projection = (xd[src, tok] @ dequantize_fp4_torch(w1[expert], s1[expert]).t()).bfloat16()
        activated = (activation(projection, c.activation, c.clamp)
                     * all_weights[src, tok, slot, None]).bfloat16()
        q, s = quantize_fp4_rne(activated)
        expected_a = dequantize_fp4_torch(q, s)
        actual_a = dequantize_fp4_torch(moe.aq[begin + selected],
                                       _unpack_output_scales(moe.asc[begin:begin + nr])[selected])
        diff = float(calc_diff(actual_a, expected_a))
        assert diff < .003, (moe.rank, expert, "GEMM1", diff)
        max_gemm1_diff = max(max_gemm1_diff, diff)
        b = dequantize_fp4_torch(w2[expert], s2[expert])
        destinations = src * checked + tok
        reference.view(-1, c.hidden).index_add_(0, destinations, (expected_a @ b.t()).bfloat16().float())
        stage_reference.view(-1, c.hidden).index_add_(0, destinations, (actual_a @ b.t()).bfloat16().float())
    dist.all_reduce(reference)
    dist.all_reduce(stage_reference)
    local = min(moe.count, checked)
    stage_error = relative_l2(result[:local], stage_reference[moe.rank, :local].bfloat16())
    pipeline_diff = float(calc_diff(result[:local], reference[moe.rank, :local].bfloat16()))
    assert stage_error < .01, (moe.rank, "GEMM2/scatter/combine", stage_error)
    assert pipeline_diff < .003, (moe.rank, "pipeline", pipeline_diff)
    errors = torch.tensor([max_gemm1_diff, stage_error, pipeline_diff], device="cuda")
    dist.all_reduce(errors, op=dist.ReduceOp.MAX)
    return dict(status="passed", checked_tokens_per_rank=min(moe.count, checked),
                max_gemm1_diff=errors[0].item(), max_scatter_reduce_relative_l2=errors[1].item(),
                max_pipeline_diff=errors[2].item())


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
                ids[0, 0], weights[0, 0] = -1, 0
                weights[1, 0] = 0
            counts = gather(torch.tensor(count, device="cuda")).tolist()
            for split in splits:
                moe.set_inputs(source[:count], ids[:count], weights[:count], split_weights=split)
                moe.compile()
                if edge_cases and case.name.startswith("smoke"):
                    moe.combine.fill_(float("nan"))
                    torch.cuda.synchronize()
                    dist.barrier()
                result = moe.forward()
                torch.cuda.synchronize()
                report = dict(case=case.name, ep=world, tokens_per_rank=counts,
                              capacity=case.capacity, hidden=case.hidden, intermediate=case.intermediate,
                              experts=case.experts, topk=case.topk, activation=case.activation,
                              scatter_dtype="bf16", routing="round_robin", split_weights=split,
                              edge_cases=edge_cases,
                              correctness=dict(status="not_checked"))
                expected = result.clone() if check else None
                if check:
                    checked = min(requested, reference_tokens) if reference_tokens else requested
                    report["correctness"] = check_correctness(moe, result, source, ids, weights, checked)
                    report["correctness"]["checked_tokens_per_rank"] = [min(n, checked) for n in counts]
                    torch.testing.assert_close(moe.forward(), expected, rtol=0, atol=0)
                    if requested == case.tokens[-1]:
                        check_graph_replay(moe.forward, expected, input_tensor=moe.source)
                if benchmark is not None:
                    timing = benchmark_pipeline(moe.forward, expected=expected, **benchmark)
                    timing["scope"] = "pre_dispatch_to_combine"
                    timing["tokens_per_second"] = sum(counts) * 1e6 / timing["max_rank_us"]["median"]
                    report["benchmark"] = timing
                if verbose and rank == 0:
                    print(json.dumps(report), flush=True)
                reports.append(report)
                dist.barrier()
    finally:
        moe.close()
    return reports


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_pipeline(case, moe_group):
    case.require_world()
    splits = (False, True) if case.experts // dist.get_world_size() > 1 else (False,)
    run_case(case, moe_group, edge_cases=True, splits=splits, verbose=False,
             reference_tokens=0 if case.name.startswith("smoke") else case.reference_tokens)


@pytest.mark.parametrize("activation, quantization", (
    (None, "mxfp4"),
    ("swiglu", "fp8"),
    ("swiglu_oai", None),
))
def test_api_rejects_unsupported_options(activation, quantization):
    # Reject unsupported combinations before touching a workspace or launching a collective.
    empty = torch.empty(0)
    with pytest.raises(RuntimeError, match="unsupported precision/activation/output_quantization"):
        dg.mega_moe_dispatch_gemm1(
            None, (empty, empty), empty, 0, 0, precision="w4a4",
            activation=activation, output_quantization=quantization)


@pytest.mark.parametrize("groups", (0, 1, 17, 256, 257))
@pytest.mark.parametrize("dst_bm, src_bm", ((64, 64), (32, 128), (128, 32)))
def test_prepare_rows(groups, dst_bm, src_bm):
    # Independent CPU layout and route oracle, including the parallel-scan boundary.
    counts = torch.tensor(([0, 1, 33, 129, 256] * ((groups + 4) // 5))[:groups], dtype=torch.int64)
    padded = ((counts + dst_bm - 1) // dst_bm * dst_bm).int()
    source_rows = ((counts + src_bm - 1) // src_bm * src_bm).int()
    offsets = (padded.cumsum(0) - padded).int()
    source_offsets = (source_rows.cumsum(0) - source_rows).int()
    meta = torch.full((int(source_rows.sum()), 3), -1, dtype=torch.int32)
    expected = torch.full((int(padded.sum()),), -7, dtype=torch.int64)
    for expert, count in enumerate(counts.tolist()):
        row = torch.arange(count)
        routes = torch.stack((torch.full_like(row, expert % 8), row + 7, row % 16), dim=1)
        src, dst = int(source_offsets[expert]), int(offsets[expert])
        meta[src:src + count] = routes.int()
        expected[dst:dst + count] = (routes[:, 0] << 32) | (routes[:, 2] << 24) | routes[:, 1]
    # Receive-count control bits must not become part of the row count.
    device_counts = (counts | (3 << 32)).cuda()
    rows, valid, starts, src_rows, src_starts = [
        torch.full((groups,), -1, dtype=torch.int32, device="cuda") for _ in range(5)]
    dg.prepare_moe_gemm2_rows(device_counts, rows, valid, starts, dst_bm)
    for actual, reference in ((rows, padded), (valid, counts.int()), (starts, offsets)):
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
    packed = torch.full_like(expected, -7, device="cuda")
    kwargs = dict(token_meta=meta.cuda(), packed_src_info=packed, src_m_rows=src_rows,
                  src_m_offsets=src_starts)
    if src_bm != dst_bm:
        kwargs["src_block_m"] = src_bm
    dg.prepare_moe_gemm2_rows(device_counts, rows, valid, starts, dst_bm, **kwargs)
    for actual, reference in ((rows, padded), (valid, counts.int()), (starts, offsets),
                              (src_rows, source_rows), (src_starts, source_offsets), (packed, expected)):
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
    # Replays must pick up new counts; no host read or synchronization is part of the API.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dg.prepare_moe_gemm2_rows(device_counts, rows, valid, starts, dst_bm, **kwargs)
    device_counts.zero_()
    graph.replay()
    for actual in (rows, valid, starts, src_rows, src_starts):
        assert not actual.count_nonzero().item()


def test_prepare_rows_rejects_incomplete_routes():
    counts = torch.zeros(2, dtype=torch.int64, device="cuda")
    rows = [torch.empty(2, dtype=torch.int32, device="cuda") for _ in range(3)]
    with pytest.raises(RuntimeError, match="route packing requires"):
        dg.prepare_moe_gemm2_rows(counts, *rows, 64, src_block_m=32)
    with pytest.raises(RuntimeError, match="route packing requires"):
        dg.prepare_moe_gemm2_rows(counts, *rows, 64,
                                 token_meta=torch.empty(0, 3, dtype=torch.int32, device="cuda"))
    with pytest.raises(RuntimeError):
        dg.prepare_moe_gemm2_rows(counts, *rows, 0)
    with pytest.raises(RuntimeError):
        dg.prepare_moe_gemm2_rows(counts, rows[0][:1], *rows[1:], 64)


def test_dispatch_config():
    args = dict(num_ranks=8, num_experts=384, num_topk=6, hidden=7168,
                num_max_tokens_per_rank=4096, precision="w4a4", intermediate_hidden=6144)
    for tokens in (0, 1, 33, 1024, 2048, 4096):
        config = dg.get_mega_moe_dispatch_config(**args, num_tokens=tokens)
        assert config["TOKEN_BYTES"] == 3584 and config["SF_BYTES"] == 224
        assert config["NUM_EXPERTS_PER_RANK"] == 48 and config["GEMM1_N"] == 6144
    with pytest.raises(RuntimeError):
        dg.get_mega_moe_dispatch_config(**args, num_tokens=4097)
    with pytest.raises(RuntimeError):
        dg.get_mega_moe_dispatch_config(**args, block_m=64)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=[c.name for c in CASES], default="smoke_swiglu")
    parser.add_argument("--tokens", type=int, nargs="+", help="tokens per rank; defaults to the case's token list")
    parser.add_argument("--capacity", type=int, help="workspace token capacity; defaults to the model table")
    parser.add_argument("--benchmark", action="store_true", help="also time the complete routed MoE forward")
    parser.add_argument("--no-check", action="store_true", help="skip numerical references (requires --benchmark)")
    parser.add_argument("--reference-tokens", type=int, default=0, help="check only the first N tokens/rank; 0 checks all")
    parser.add_argument("--mode", choices=("graph", "eager"), default="graph", help="benchmark execution mode")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100, help="forwards per timing sample, equal on every rank")
    parser.add_argument("--repeats", type=int, default=5, help="number of timing samples")
    parser.add_argument("--replicas", action="store_true", help="split the last local expert into replica weight buffers")
    parser.add_argument("--edge-cases", action="store_true", help="include empty last rank for tokens=1 and masked/zero-weight routes")
    args = parser.parse_args()
    case = next(c for c in CASES if c.name == args.case)
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
        benchmark = dict(warmup=args.warmup, iterations=args.iterations,
                         repeats=args.repeats, mode=args.mode) if args.benchmark else None
        run_case(case, group, check=not args.no_check, reference_tokens=args.reference_tokens,
                 edge_cases=args.edge_cases, splits=(args.replicas,), benchmark=benchmark)


if __name__ == "__main__":
    main()
