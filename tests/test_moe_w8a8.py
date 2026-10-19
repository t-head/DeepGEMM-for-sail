"""W8A8 routed MoE: channelwise and blockwise quantization, model cases below."""

import json
import math

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import deep_gemm as dg
from deep_gemm import interleave_fp8_blockwise_mega_moe_weights
from deep_gemm import m_grouped_gemm_fp8_fp8_bf16_nt_nopad as grouped_gemm
from moe_test_utils import (
    compile_kernels,
    MODEL_TOKENS, CombineBarrier, MoECase, activation, calc_diff, check_graph_replay, clear_dispatch_counts, gather,
    interleave_gate_up as interleave, moe_group, quantize_fp8_rows as quantize, relative_l2,
)

CHANNELWISE_CASES = (
    MoECase("smoke", 512, 256, 16, 2, 64, tokens=(1, 33, 64), worlds=(4, 8),
            scatter_tile=(32, 256, 128, 16, 64, 2)),
    MoECase("minimax_m3", 6144, 3072, 128, 4, 4096, tokens=MODEL_TOKENS,
            activation="swigluoai", clamp=7.0, worlds=(8,)),
    MoECase("qwen38_flash", 2560, 640, 512, 10, 8192, tokens=MODEL_TOKENS, worlds=(4,), reference_tokens=64),
    MoECase("glm52", 6144, 2048, 256, 8, 4096, tokens=MODEL_TOKENS, worlds=(8,)),
    MoECase("glm53_flash", 4096, 2048, 288, 8, 4096, tokens=MODEL_TOKENS,
            clamp=10.0, input_scale=16, worlds=(4,)),
)
BLOCKWISE_CASES = (
    MoECase("smoke", 256, 128, 8, 2, 17, tokens=(17, 0, 17, 17), worlds=(1, 2, 4, 8)),
)


@pytest.mark.parametrize("case", CHANNELWISE_CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("routing", ("biased", "balanced"))
def test_channelwise_pipeline(case, routing, moe_group):
    case.require_world()
    rank, world, group = dist.get_rank(), dist.get_world_size(), moe_group
    experts, topk, hidden, intermediate, cap = (
        case.experts, case.topk, case.hidden, case.intermediate, case.capacity)
    epr = experts // world
    alpha = 1.702 if case.activation == "swigluoai" else None
    # The generic tuner selects BN128 for the tiny case, while this scatter
    # epilogue requires a full BN256 transaction. Production models use defaults.
    scatter_config = ((dg.deep_gemm_cpp.get_num_sms(), *case.scatter_tile, (0, 0, 0))
                      if case.scatter_tile else None)
    rng = torch.Generator(device="cuda").manual_seed(20260909 + rank)
    weights = []
    for n, k, amplitude in [
        (2 * intermediate, hidden, 0.125),
        (hidden, intermediate, 0.015625),
    ]:
        packed = torch.empty(epr, n, k, dtype=torch.float8_e4m3fn, device="cuda")
        scales = torch.empty(epr, n, 1, device="cuda")
        for e in range(epr):
            source = (
                torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=rng)
                * amplitude
            )
            source *= torch.pow(
                2.0, torch.arange(n, device="cuda").float()[:, None] % 3 - 1
            )
            q, s = quantize(source)
            packed[e].copy_(q)
            scales[e].copy_(s)
        weights.append((packed, scales))
    del source, q, s
    (w13, s13), (w2, s2) = weights
    w13i, s13i = interleave(w13), interleave(s13)
    # Exercise the buffer defaults through the complete fused pipeline.
    buf = dg.get_symm_buffer_for_mega_moe(
        group, experts, cap, topk, hidden, precision="fp8_channelwise"
    )
    off = buf._layout["off_ws"]
    buf.buffer[off : off + buf._layout["ws_bytes"]].zero_()
    activated = torch.empty(
        buf.num_pool, intermediate, device="cuda", dtype=torch.bfloat16
    )
    aq = torch.empty_like(activated, dtype=torch.float8_e4m3fn)
    asc = torch.empty(buf.num_pool, 1, device="cuda")
    rows, valid, offsets, src_rows, src_offsets, meta = buf.get_gemm2_metadata()
    indices = torch.empty(buf.num_pool, dtype=torch.int32, device="cuda")
    elements = topk * cap * hidden
    combine = symm_mem.empty(elements, dtype=torch.bfloat16, device="cuda")
    combine_handle = symm_mem.rendezvous(combine, group=group)
    barrier = CombineBarrier(group)
    pointers = [
        combine_handle.get_buffer(r, (elements,), torch.bfloat16).data_ptr()
        for r in range(world)
    ]
    deltas = torch.tensor(
        [v - pointers[rank] for v in pointers], device="cuda", dtype=torch.int64
    )
    combine.zero_()
    torch.cuda.synchronize()
    dist.barrier()


    for requested in case.tokens:
        assert 0 < requested <= cap
        # A zero-token peer still participates in dispatch and mega_moe_combine.
        nt = (
            0
            if requested in (1, 33) and rank == world - 1
            else requested
        )
        x = (
            torch.randn(
                requested, hidden, dtype=torch.bfloat16, device="cuda", generator=rng
            )
            * (0.5 * case.input_scale)
        )
        if requested > 1:
            x[0].zero_()  # Zero rows exercise the quantizer scale floor.
        logits = torch.randn(requested, experts, device="cuda", generator=rng)
        scores = logits.sigmoid()
        bias = torch.linspace(-0.1, 0.1, experts, device="cuda")
        ids = (scores + bias).topk(topk, dim=1).indices.contiguous()
        if routing == "balanced":
            # Equal expert counts, distinct top-k experts, all peers.
            ids = (
                (
                    torch.arange(requested, device="cuda")[:, None] * topk
                    + torch.arange(topk, device="cuda")[None, :]
                    + rank * 17
                )
                % experts
            ).contiguous()
        route_weights = scores.gather(1, ids)
        route_weights = (
            route_weights / route_weights.sum(1, keepdim=True)
        ).contiguous()
        if nt == 0:
            ids.fill_(-1)
            route_weights.zero_()

        tuning = dg.get_mega_moe_tuning_config(
            world, experts, topk, hidden, intermediate, nt, precision="fp8_channelwise",
            activation="swiglu_oai" if alpha is not None else "swiglu")

        def pipeline():
            dg.mega_moe_pre_dispatch(
                x[:nt], ids[:nt], route_weights[:nt], buf, nt, precision="fp8_channelwise")
            dg.mega_moe_dispatch_gemm1(
                buf, (w13i, s13i), aq, rank, nt, precision="fp8_channelwise", output_quantization="fp8",
                out_scales=asc, bf16_scratch=activated, activation_limit=case.clamp or None,
                activation="swiglu_oai" if alpha is not None else "swiglu", activation_alpha=alpha, tuning=tuning)
            dg.mega_moe_gemm2_scatter(
                (aq, asc), (w2, s2), meta, combine, deltas, rows, valid, rank, cap, topk,
                precision="fp8_channelwise", m_indices=indices, configs=scatter_config, kernel_type="Default",
                validate_metadata=False, tuning=tuning)
            clear_dispatch_counts(buf, experts)
            return dg.mega_moe_combine(
                combine, route_weights[:nt], nt, hidden, topk, cap,
                barrier.pointers, rank, world, barrier.done, tuning=tuning)

        actual = pipeline().clone()
        torch.cuda.synchronize()
        assert torch.isfinite(actual).all()
        if nt:
            assert actual.abs().max() > 0
        # Repeated calls must overwrite the routed rows and reset counts correctly.
        repeated = pipeline()
        torch.cuda.synchronize()
        assert torch.equal(actual, repeated), (rank, requested, "repeat mismatch")

        all_routes = gather(ids)
        expected_valid = torch.bincount(all_routes[all_routes >= 0], minlength=experts)[
            rank * epr:(rank + 1) * epr].int()
        expected_rows = (expected_valid + 255) // 256 * 256
        expected_offsets = (expected_rows.cumsum(0) - expected_rows).int()
        for actual_rows, expected in ((valid, expected_valid), (rows, expected_rows),
                                      (src_rows, expected_rows), (offsets, expected_offsets),
                                      (src_offsets, expected_offsets)):
            torch.testing.assert_close(actual_rows, expected, rtol=0, atol=0)
        for expert, count in enumerate(expected_valid.tolist()):
            start = int(expected_offsets[expert])
            routes = (all_routes == rank * epr + expert).nonzero()
            expected = (routes[:, 0] << 32) | (routes[:, 2] << 24) | routes[:, 1]
            torch.testing.assert_close(meta[start:start + count].sort().values,
                                       expected.sort().values, rtol=0, atol=0)

        # Check the pre-quantizer independently. Near FP8 halfway values,
        # division and reciprocal-multiply need not choose identical bytes.
        if nt:
            expected_scale = x[:nt].float().abs().amax(-1, keepdim=True) / 448.0
            assert torch.allclose(
                buf.input_scale[:nt].reshape_as(expected_scale),
                expected_scale,
                rtol=1e-6,
                atol=0,
            )
            dequant = buf.input_token[:nt].float() * buf.input_scale[:nt].view(-1, 1)
            assert float(calc_diff(dequant, x[:nt])) < 0.001

        sample_count = min(requested, case.reference_tokens)
        # The GEMM reference starts at the actual FP8 input boundary; it does
        # not reuse any GEMM, activation or output-quantization implementation.
        input_sample = torch.zeros(sample_count, hidden, device="cuda")
        if nt:
            input_sample.copy_(
                buf.input_token[:sample_count].float()
                * buf.input_scale[:sample_count].view(-1, 1)
            )
        all_x = [torch.empty_like(input_sample) for _ in range(world)]
        all_ids = [torch.empty_like(ids[:sample_count]) for _ in range(world)]
        dist.all_gather(all_x, input_sample)
        dist.all_gather(all_ids, ids[:sample_count].contiguous())
        all_x, all_ids = torch.stack(all_x), torch.stack(all_ids)
        xd = all_x
        stage_g1_diff = 0.0
        quant_diff = 0.0
        g2_contributions = torch.zeros(
            world, topk, sample_count, hidden, dtype=torch.bfloat16, device="cuda"
        )
        row_offset = 0
        row_counts, valid_counts = rows.cpu().tolist(), valid.cpu().tolist()
        for e, (padded, count) in enumerate(zip(row_counts, valid_counts)):
            if count:
                pool_rows = torch.arange(row_offset, row_offset + count, device="cuda")
                md = buf.token_src_metadata[pool_rows].long()
                pool_rows = pool_rows[md[:, 1] < sample_count]
                md = md[md[:, 1] < sample_count]
                if pool_rows.numel():
                    assert torch.all(
                        all_ids[md[:, 0], md[:, 1], md[:, 2]] == rank * epr + e
                    )
                    projection = xd[md[:, 0], md[:, 1]] @ (w13[e].float() * s13[e]).T
                    a = activation(projection, case.activation, case.clamp).bfloat16()
                    stage_g1_diff = max(
                        stage_g1_diff, float(calc_diff(activated[pool_rows], a))
                    )
                    actual_a = aq[pool_rows].float() * asc[pool_rows]
                    quant_diff = max(
                        quant_diff, float(calc_diff(actual_a, activated[pool_rows]))
                    )
                    y = (actual_a @ (w2[e].float() * s2[e]).T).bfloat16()
                    g2_contributions[md[:, 0], md[:, 2], md[:, 1]] = y
            row_offset += padded
        dist.all_reduce(g2_contributions)
        g2_expected = (
            (
                g2_contributions[rank].permute(1, 0, 2).float()
                * route_weights[:sample_count, :, None]
            )
            .sum(1)
            .bfloat16()
        )
        g2_diff = (
            float(
                calc_diff(actual[:sample_count], g2_expected[: min(nt, sample_count)])
            )
            if nt
            else 0.0
        )
        stage_value = torch.tensor(
            [stage_g1_diff, quant_diff, g2_diff], device="cuda", dtype=torch.float64
        )
        dist.all_reduce(stage_value, op=dist.ReduceOp.MAX)
        contributions = torch.zeros(
            world, topk, sample_count, hidden, dtype=torch.bfloat16, device="cuda"
        )
        for e in range(epr):
            source, token, slot = torch.where(all_ids == rank * epr + e)
            if not source.numel():
                continue
            projection = xd[source, token] @ (w13[e].float() * s13[e]).T
            a = activation(projection, case.activation, case.clamp).bfloat16()
            q, scale = quantize(a)
            y = ((q.float() * scale) @ (w2[e].float() * s2[e]).T).bfloat16()
            contributions[source, slot, token] = y
        dist.all_reduce(contributions)
        expected = (
            (
                contributions[rank].permute(1, 0, 2).float()
                * route_weights[:sample_count, :, None]
            )
            .sum(1)
            .bfloat16()
        )
        difference = (
            float(calc_diff(actual[:sample_count], expected[: min(nt, sample_count)]))
            if nt
            else 0.0
        )
        metric = torch.tensor([difference], device="cuda", dtype=torch.float64)
        dist.all_reduce(metric, op=dist.ReduceOp.MAX)
        assert stage_value[0].item() < 1e-4, (requested, "GEMM1", stage_value.tolist())
        assert stage_value[1].item() < 0.001, (
            requested,
            "FP8 quantization",
            stage_value.tolist(),
        )
        assert stage_value[2].item() < 1e-4, (
            requested,
            "GEMM2/scatter",
            stage_value.tolist(),
        )
        # The full reference re-quantizes an independently accumulated GEMM1.
        # Small BF16 differences can cross FP8 bins. Keep stage checks tighter.
        assert metric.item() < 5e-4, (requested, metric.item())
        check_graph_replay(pipeline, actual, input_tensor=x)
    dist.barrier()
    buf.destroy()


@pytest.mark.parametrize("padded", (False, True), ids=("default_ep2", "overlap_padded_ep2"))
def test_channelwise_scatter_combine(padded, moe_group, monkeypatch):
    monkeypatch.setenv("DG_MOE_COMBINE_TOKEN_MAJOR", "1")
    """Regress the former world2 IPC barriers, including padding and guard rows."""
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        pytest.skip("cross-rank scatter/combine regression requires EP2")
    hidden, k, topk, capacity = 512, 2048, 8, 80
    groups, physical = (4, 128) if padded else (1, 257)
    counts = (127, 65, 63, 1) if padded else (257,)
    m, total_valid = groups * physical, sum(counts)
    rows = torch.arange(m, device="cuda")
    channels = torch.arange(hidden, device="cuda")
    group_ids = rows // physical
    valid_rows = torch.cat([torch.arange(count, device="cuda") + g * physical
                            for g, count in enumerate(counts)])
    logical = torch.arange(total_valid, device="cuda")
    palette = torch.tensor((-2., -1., -.5, 0., .5, 1., 2.), device="cuda")
    lhs_palette = torch.tensor((.25, .5, 1., 2.), device="cuda")
    rhs_palette = torch.tensor((.5, 1., 2., 4.), device="cuda")

    def data_for(source_rank):
        values = palette[(channels[None, :] + group_ids[:, None] * 5
                          + (rows % physical)[:, None] * 3 + source_rank * 2) % 7]
        lhs_scale = lhs_palette[(rows + group_ids + source_rank) % 4].view(m, 1)
        rhs_scale = rhs_palette[(channels[None, :] // 16
                                + torch.arange(groups, device="cuda")[:, None] + source_rank) % 4]
        return values, lhs_scale, rhs_scale.unsqueeze(-1).contiguous()

    values, lhs_scale, rhs_scale = data_for(rank)
    selected_k = (rank * 509 + rows * 7) % k
    lhs = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device="cuda")
    lhs[rows, selected_k] = 1.
    original_lhs = lhs.clone()
    rhs = torch.zeros(groups, hidden, k, device="cuda")
    for g in range(groups):
        group_rows = rows[g * physical:(g + 1) * physical]
        rhs[g, :, selected_k[group_rows]] = values[group_rows].T
    rhs = rhs.to(torch.float8_e4m3fn)
    reference_rows = []
    for source_rank in range(world):
        value, ls, rs = data_for(source_rank)
        reference_rows.append((value * ls * rs.squeeze(-1)[group_ids]).bfloat16())

    sentinel, prefix, suffix = -37., 256, 544
    elements = topk * capacity * hidden
    raw = symm_mem.empty(prefix + elements + suffix, dtype=torch.bfloat16, device="cuda")
    handle = symm_mem.rendezvous(raw, group=moe_group)
    combine = raw[prefix:prefix + elements]
    peer_ptrs = [handle.get_buffer(r, raw.shape, raw.dtype).data_ptr() for r in range(world)]
    offsets = torch.tensor([p - raw.data_ptr() for p in peer_ptrs], dtype=torch.int64, device="cuda")
    barrier = CombineBarrier(moe_group)
    metadata = torch.zeros(m, dtype=torch.int64, device="cuda")
    m_rows = torch.full((groups,), physical, dtype=torch.int32, device="cuda")
    valid = torch.tensor(counts, dtype=torch.int32, device="cuda")
    indices = group_ids.int()
    weights = torch.zeros(capacity, topk, device="cuda")
    tile = (64, 256, 128, 32, 64, 3) if padded else (256, 256, 128, 64, 64, 4)
    config = (dg.deep_gemm_cpp.get_num_sms(), *tile, (0, 0, 0))

    def pipeline():
        dg.mega_moe_gemm2_scatter(
            (lhs, lhs_scale), (rhs, rhs_scale), metadata, combine, offsets, m_rows, valid, rank, capacity,
            topk, precision="fp8_channelwise", m_indices=indices, configs=config,
            kernel_type="OverlapPrologue" if padded else "Default", validate_metadata=False)
        reduced = dg.mega_moe_combine(combine, weights, capacity, hidden, topk, capacity,
                                    barrier.pointers, rank, world, barrier.done)
        # Enqueue a second consumer before any host synchronization. It checks
        # every routed row, including locations that have zero reduce weight.
        return reduced, combine.clone()

    compile_kernels(pipeline)
    torch.cuda.synchronize()
    dist.barrier()
    graph = None
    for iteration in range(8):
        factor = (1., -1., 0., 2.)[iteration % 4]
        lhs.copy_((original_lhs.float() * factor).to(lhs.dtype))
        expected = torch.full((topk, capacity, hidden), sentinel, dtype=torch.bfloat16, device="cuda")
        metadata.zero_()
        weights.zero_()
        for source_rank in range(world):
            remote = ((logical + source_rank + iteration) & 1).bool()
            destination = torch.where(remote, 1 - source_rank, source_rank)
            key = ((logical + source_rank * total_valid) * 3 + iteration * 17) % (2 * total_valid)
            key = key + int(padded)  # Padding metadata zero must not touch rank0/token0/slot0.
            slot, token = key % topk, key // topk
            if source_rank == rank:
                metadata[valid_rows] = (destination << 32) | (slot << 24) | token
            selected = destination == rank
            expected[slot[selected], token[selected]] = reference_rows[source_rank][valid_rows[selected]] * factor
            weights[token[selected], slot[selected]] = (slot[selected].float() + 1) / topk
        expected.masked_fill_(expected == 0, 0.)  # Canonical GEMM zero for the bitwise oracle.
        raw.fill_(sentinel)
        # Complete destination resets on both ranks before either producer
        # writes remotely; there is no host barrier between scatter and reduce.
        torch.cuda.synchronize()
        dist.barrier()
        if iteration < 4:
            reduced, observed = pipeline()
        else:
            if graph is None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    reduced, observed = pipeline()
            graph.replay()
        torch.cuda.synchronize()
        expected_reduce = (expected.permute(1, 0, 2).float() * weights[..., None]).sum(1).bfloat16()
        ok = (torch.equal(observed.view(torch.uint16), expected.flatten().view(torch.uint16))
              and torch.equal(reduced.view(torch.uint16), expected_reduce.view(torch.uint16))
              and torch.equal(raw[:prefix], torch.full_like(raw[:prefix], sentinel))
              and torch.equal(raw[-suffix:], torch.full_like(raw[-suffix:], sentinel))
              and int(barrier.flag.item()) == iteration + 1)
        status = torch.tensor(int(ok), dtype=torch.int32, device="cuda")
        dist.all_reduce(status, op=dist.ReduceOp.MIN)
        assert status.item(), (rank, padded, iteration, "scatter/combine or guard mismatch")


def quantize_blockwise(x):
    # Independent CPU oracle: device Torch division's approximate reciprocal
    # can move an E4M3 rounding midpoint. Activations have one FP32 scale/128 K.
    blocks = x.cpu().float().reshape(x.shape[0], -1, 128)
    scales = blocks.abs().amax(-1).clamp_min(1e-10) / 448.0
    q = (blocks / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.reshape(x.shape).to(x.device), scales.to(x.device)


@pytest.mark.parametrize("case", BLOCKWISE_CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("routing", ("balanced", "hot"))
@torch.inference_mode()
def test_blockwise_pipeline(case, routing, moe_group, monkeypatch):
    monkeypatch.setenv("DG_MOE_COMBINE_TOKEN_MAJOR", "1")
    case.require_world()
    limit = case.clamp or None
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(1729 + rank)
    h, n, e, cap, topk = (case.hidden, 2 * case.intermediate, case.experts,
                          case.capacity, case.topk)
    epr = e // world
    buf = dg.get_symm_buffer_for_mega_moe(
        dist.group.WORLD, e, cap, topk, h, precision="fp8_blockwise",
    )
    x = torch.empty((cap, h), dtype=torch.bfloat16, device="cuda")
    ids = torch.empty((cap, topk), dtype=torch.int64, device="cuda")
    weights = torch.empty((cap, topk), device="cuda")
    w = torch.empty((epr, n, h), dtype=torch.float8_e4m3fn, device="cuda")
    # Deliberately distinct, non-power-of-two scales for every 128x128 block.
    sw = (torch.rand((epr, n // 128, h // 128), device="cuda") + 0.25) / (h**0.5 * 4)
    for expert in range(epr):
        w[expert].copy_(torch.randint(-16, 17, (n, h), device="cuda").float())
    out = torch.empty((buf.num_pool, n), dtype=torch.bfloat16, device="cuda")
    packed_w, packed_sw = interleave_fp8_blockwise_mega_moe_weights(w, sw)
    assert torch.equal(packed_w.view(torch.uint8).reshape(epr, n // (2 * 8), 2, 8, h)
                       .transpose(1, 2).reshape(epr, n, h), w.view(torch.uint8))
    aq = torch.empty((buf.num_pool, n // 2), dtype=torch.float8_e4m3fn, device="cuda")
    sq = torch.empty((n // 256, buf.num_pool), device="cuda").t()
    rows, valid, offsets, _, _, packed_meta = buf.get_gemm2_metadata()
    indices = torch.empty(buf.num_pool, dtype=torch.int32, device="cuda")

    # GEMM2/combine check uses actual GEMM1 FP8 outputs as independent inputs,
    # isolating scatter/reduce correctness from the already checked GEMM1 error.
    w2 = torch.randint(-16, 17, (epr, h, n // 2), device="cuda").to(torch.float8_e4m3fn)
    sw2 = (torch.rand((epr, h // 128, n // 256), device="cuda") + .25) / ((n // 2)**.5 * 4)
    combine = symm_mem.empty(topk * cap * h, dtype=torch.bfloat16, device="cuda")
    ch = symm_mem.rendezvous(combine, group=dist.group.WORLD)
    cp = [ch.get_buffer(i, (combine.numel(),), torch.bfloat16).data_ptr() for i in range(world)]
    peer_offsets = torch.tensor([p-cp[rank] for p in cp], dtype=torch.int64, device="cuda")
    combine.zero_()
    barrier = CombineBarrier(moe_group)
    torch.cuda.synchronize()
    dist.barrier()

    def scatter():
        dg.mega_moe_gemm2_scatter(
            (aq, sq), (w2, sw2), packed_meta, combine, peer_offsets, rows, valid, rank, cap, topk,
            precision="fp8_blockwise", row_offsets=offsets)

    def check_scatter(count, all_weights):
        reference = torch.zeros((world * cap, h), dtype=torch.float32, device="cuda")
        for expert in range(epr):
            nr = int(valid[expert])
            begin = int(offsets[expert])
            if not nr:
                continue
            sel = slice(begin, begin + nr)
            a2 = aq[sel].float() * sq[sel].repeat_interleave(128, -1)
            b2 = w2[expert].float() * sw2[expert].repeat_interleave(128, 0).repeat_interleave(128, 1)
            out2 = (a2 @ b2.t()).bfloat16().float()
            src, tok, slot = buf.token_src_metadata[sel].long().unbind(-1)
            reference.index_add_(0, src * cap + tok, out2 * all_weights[src, tok, slot, None])
        dist.all_reduce(reference)
        scatter()
        reduced = dg.mega_moe_combine(combine, weights[:count], count, h, topk, cap,
                                    barrier.pointers, rank, world, barrier.done)
        expected = reference.view(world, cap, h)[rank, :count].bfloat16()
        assert reduced.isfinite().all()
        error = relative_l2(reduced, expected)
        assert error < .01, (rank, "GEMM2_scatter_reduce", error)
        print(json.dumps(dict(rank=rank, tokens=count, scatter_reduce_relative_l2=error, scratch_bytes=0)), flush=True)

    def launch(fused, count):
        dg.mega_moe_pre_dispatch(
            x[:count], ids[:count], weights[:count], buf, count, precision="fp8_blockwise")
        if fused == "epilogue":
            dg.mega_moe_dispatch_gemm1(
                buf, (packed_w, packed_sw), aq, rank, count, precision="fp8_blockwise", activation="swiglu",
                output_quantization="fp8", out_scales=sq, activation_limit=limit)
        elif fused:
            dg.mega_moe_dispatch_gemm1(
                buf, (w, sw), out, rank, count, precision="fp8_blockwise", activation=None)
        else:
            dg.deep_gemm_cpp._fp8_blockwise_mega_moe_dispatch(buf, rank, count)
            grouped_gemm((buf.pool_token, buf.pool_scale), (w, sw), out,
                         indices, m_rows=rows)

    def reset_counts():
        # No GEMM2 consumer in this dispatch/GEMM1 test. Recycle send/receive
        # counters, never the cooperative barrier signals in the first 32 bytes.
        buf.buffer[32:32 + (e * 2 + epr) * 8].zero_()
        torch.cuda.synchronize()
        dist.barrier()

    # Calls compile automatically. Ensure every rank finishes its JIT work
    # before any rank enters the first cooperative device dispatch barrier.
    compile_kernels(lambda: launch(True, cap), lambda: launch(False, cap),
                    lambda: launch("epilogue", cap), scatter)
    torch.cuda.synchronize()
    dist.barrier()

    stride = next(s for s in range(53, 0, -1) if math.gcd(s, e) == 1)
    token = torch.arange(cap, device="cuda")[:, None] + rank * cap
    slot = torch.arange(topk, device="cuda")[None, :]
    clamp_coverage = torch.zeros(3, dtype=torch.int64, device="cuda")
    for iteration, requested in enumerate(case.tokens):
        assert 0 <= requested <= cap
        counts = [requested] * world
        if case.name == "smoke" and iteration == 0:
            counts = [min(v, requested) for v in ([0, 1, 7] + [requested] * world)[:world]]
        count = counts[rank]
        x.normal_(std=(0.3 + 0.1 * iteration) * case.input_scale)
        route = slot * stride + iteration * 17
        if routing == "balanced":
            route = route + token * 31
        ids.copy_((route % e).expand(cap, topk))
        weights.copy_(torch.rand_like(weights).softmax(-1))
        ids[count:], weights[count:] = -1, 0
        if case.name == "smoke" and count and iteration == 2:
            ids[count - 1], weights[count - 1] = -1, 0
        if case.name == "smoke" and count and iteration == 3:
            weights[0, 0] = 0  # Zero weight must still preserve the route.
        qref, sref = quantize_blockwise(x)
        all_ids, all_weights = gather(ids), gather(weights)
        all_q, all_s = gather(qref.view(torch.uint8)), gather(sref)
        for fused in (True, False, "epilogue"):
            reset_counts()
            out.fill_(float("nan"))
            aq.view(torch.uint8).fill_(127)
            sq.fill_(float("nan"))
            launch(fused, count)
            torch.cuda.synchronize()
            torch.testing.assert_close(buf.input_scale[:count], sref[:count], rtol=1e-6, atol=0)
            assert torch.equal(buf.input_token[:count].view(torch.uint8), qref[:count].view(torch.uint8))
            next_offset, max_error, covered = 0, 0.0, 0
            for expert in range(epr):
                routes = (all_ids == rank * epr + expert).nonzero()
                nr = len(routes)
                assert valid[expert].item() == nr
                assert rows[expert].item() == (nr + 127) // 128 * 128
                assert offsets[expert].item() == next_offset
                begin = next_offset
                next_offset += (nr + 127) // 128 * 128
                if not nr:
                    continue
                covered += 1
                selection = slice(begin, begin + nr)
                meta = buf.token_src_metadata[selection].long()
                assert sorted(map(tuple, meta.tolist())) == sorted(map(tuple, routes.tolist()))
                src, tok, slt = meta.unbind(-1)
                assert torch.equal(buf.pool_token[selection].view(torch.uint8), all_q[src, tok])
                torch.testing.assert_close(buf.pool_scale[selection], all_s[src, tok], rtol=1e-6, atol=0)
                torch.testing.assert_close(buf.pool_topk_weights[selection], all_weights[src, tok, slt], rtol=0, atol=0)
                a = all_q[src, tok].view(torch.float8_e4m3fn).float() * all_s[src, tok].repeat_interleave(128, -1)
                b = w[expert].float() * sw[expert].repeat_interleave(128, 0).repeat_interleave(128, 1)
                reference = (a @ b.t()).bfloat16()
                if fused == "epilogue":
                    gate, up = reference.float().chunk(2, dim=-1)
                    if limit is not None:
                        limit = limit
                        clamp_coverage += torch.stack([
                            (gate > limit).sum(), (up > limit).sum(), (up < -limit).sum(),
                        ])
                        gate = gate.clamp(max=limit)
                        up = up.clamp(min=-limit, max=limit)
                    activated = (gate / (1 + torch.exp(-gate)) * up).bfloat16()
                    q, sf = quantize_blockwise(activated)
                    actual = aq[selection].float() * sq[selection].repeat_interleave(128, -1)
                    expected = q.float() * sf.repeat_interleave(128, -1)
                    assert actual.isfinite().all()
                    error = relative_l2(actual, expected)
                    torch.testing.assert_close(packed_meta[selection], (src << 32) | (slt << 24) | tok, rtol=0, atol=0)
                    padded = slice(begin + nr, next_offset)
                    assert torch.count_nonzero(aq[padded].view(torch.uint8)).item() == 0
                    torch.testing.assert_close(sq[padded], torch.full_like(sq[padded], 1e-10 / 448), rtol=1e-6, atol=0)
                else:
                    assert out[selection].isfinite().all()
                    error = relative_l2(out[selection], reference)
                assert error < (0.025 if fused == "epilogue" else 0.01), (rank, expert, fused, error)
                max_error = max(max_error, error)
            if routing == "balanced" and e == 512 and world == 4 and requested >= 64:
                assert covered > 64  # Full EPR128 scheduler, not only the first 64.
            if fused == "epilogue":
                assert torch.all(aq[next_offset:].view(torch.uint8) == 127)
                assert sq[next_offset:].isnan().all()
            if fused == "epilogue":
                check_scatter(count, all_weights)
            print(json.dumps(dict(rank=rank, iteration=iteration, fused=fused,
                                  tokens=count, covered_experts=covered,
                                  swiglu_limit=limit,
                                  max_g1_relative_l2=max_error)), flush=True)
            dist.barrier()

    if limit is not None:
        dist.all_reduce(clamp_coverage)
        assert torch.all(clamp_coverage > 0), "test inputs did not exercise every clamp bound"
        if rank == 0:
            print(json.dumps(dict(clamp_coverage=clamp_coverage.tolist(),
                                  swiglu_limit=limit)), flush=True)

    dist.barrier()
    buf.destroy()


def test_channelwise_replica_gemm1(moe_group):
    world_size = dist.get_world_size()
    if world_size != 8:
        pytest.skip("32-master + 1-replica specialization requires EP8")
    group = moe_group
    local_rank = dist.get_rank()
    masters_per_rank = 32
    replicas_per_rank = 1
    physical_per_rank = masters_per_rank + replicas_per_rank
    hidden = 128
    gemm1_n = 256
    physical_experts = physical_per_rank * world_size
    device = torch.device("cuda", local_rank)

    buffer = dg.get_symm_buffer_for_mega_moe(
        group,
        physical_experts,
        1,
        1,
        hidden,
        precision="fp8_channelwise",
        block_m=256,
    )
    buffer.input_token.fill_(1.0)
    buffer.input_scale.fill_(1.0)
    buffer.input_topk_idx.fill_(
        local_rank * physical_per_rank + masters_per_rank
    )
    buffer.input_topk_weights.fill_(1.0)

    master_weight = torch.zeros(
        (masters_per_rank, gemm1_n, hidden),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    master_scale = torch.ones(
        (masters_per_rank, gemm1_n, 1),
        dtype=torch.float32,
        device=device,
    )
    replica_weight = torch.full(
        (replicas_per_rank, gemm1_n, hidden),
        1.0 / hidden,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    replica_scale = torch.ones(
        (replicas_per_rank, gemm1_n, 1),
        dtype=torch.float32,
        device=device,
    )
    output = torch.full(
        (buffer.num_pool, gemm1_n // 2),
        -1.0,
        dtype=torch.bfloat16,
        device=device,
    )

    dg.mega_moe_dispatch_gemm1(
        buffer, (master_weight, master_scale), output, local_rank, 1, precision="fp8_channelwise",
        rhs_replicas=(replica_weight, replica_scale))
    torch.cuda.synchronize()
    expected = torch.nn.functional.silu(torch.tensor(1.0)).item()
    torch.testing.assert_close(
        output[0].float(),
        torch.full((gemm1_n // 2,), expected, device=device),
        rtol=0.02,
        atol=0.02,
    )
    replica_recv_status = int(buffer.recv_count_sum[masters_per_rank].item())
    assert (replica_recv_status & 0xFFFFFFFF) == 1
    metadata = buffer.get_gemm2_metadata()
    expected_valid = torch.zeros(physical_per_rank, dtype=torch.int32, device=device)
    expected_valid[masters_per_rank] = 1
    torch.testing.assert_close(metadata.valid_m_rows, expected_valid, rtol=0, atol=0)
    for rows in (metadata.m_rows, metadata.src_m_rows):
        torch.testing.assert_close(rows, expected_valid * 256, rtol=0, atol=0)
    assert not metadata.m_offsets.count_nonzero().item()
    assert not metadata.src_m_offsets.count_nonzero().item()
    assert metadata.packed_src_info[0].item() == (local_rank << 32)
    dist.barrier(group)
    buffer.destroy()



# Production selector boundary cases; full model pipelines live in W8A8 tests.
@pytest.mark.parametrize("rows_per_group,valid_rows_per_group", (
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (129, 129),
    (256, 129),
))
@pytest.mark.usefixtures("moe_group")
def test_glm52_scatter_selector(
    rows_per_group: int,
    valid_rows_per_group: int,
) -> None:
    from deep_gemm.deep_gemm_cpp import (
        _select_glm_channelwise_scatter_config,
    )
    from deep_gemm.deep_gemm_cpp import get_num_sms

    device = torch.device("cuda", torch.cuda.current_device())
    fp8 = torch.float8_e4m3fn
    num_groups = 32
    if rows_per_group < 8:
        raise ValueError("rows per group must be at least eight")
    if not 0 < valid_rows_per_group <= rows_per_group:
        raise ValueError(
            "valid rows per group must be in (0, rows_per_group]"
        )
    rows = num_groups * rows_per_group
    physical_expected_m = (
        rows + num_groups - 1
    ) // num_groups
    topk = 8
    max_tokens = rows // topk
    hidden = 6144
    k = 2048

    row_ids = torch.arange(rows, dtype=torch.int64, device=device)
    group_ids = torch.div(
        row_ids, rows_per_group, rounding_mode="floor"
    )
    rows_in_group = row_ids % rows_per_group
    selected_k = (
        group_ids * 29 + (rows_in_group % 8) * 257
    ) % k

    lhs_float = torch.zeros(
        (rows, k), dtype=torch.float32, device=device
    )
    lhs_float[row_ids, selected_k] = 1.0
    lhs = lhs_float.to(fp8).contiguous()
    del lhs_float

    # FP8 zero has the all-zero bit pattern.  Avoid materializing a
    # [32, 6144, 2048] float32 tensor solely to initialize the RHS.
    rhs = torch.zeros(
        (num_groups, hidden, k), dtype=fp8, device=device
    )
    channels = torch.arange(hidden, dtype=torch.int64, device=device)
    fp8_values = torch.tensor(
        (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0),
        dtype=torch.float32,
        device=device,
    ).to(fp8)
    for group in range(num_groups):
        for local_slot in range(8):
            row = group * rows_per_group + local_slot
            pattern = (
                channels + group * 3 + local_slot
            ) % fp8_values.numel()
            rhs[group, :, selected_k[row]].copy_(
                fp8_values[pattern]
            )
    rhs = rhs.contiguous()

    lhs_scale_palette = torch.tensor(
        (0.25, 0.5, 1.0, 2.0),
        dtype=torch.float32,
        device=device,
    )
    lhs_scales = lhs_scale_palette[
        (row_ids + group_ids) % lhs_scale_palette.numel()
    ].reshape(rows, 1).contiguous()

    rhs_scale_palette = torch.tensor(
        (0.5, 1.0, 2.0, 4.0),
        dtype=torch.float32,
        device=device,
    )
    rhs_scale_indices = (
        torch.div(channels, 16, rounding_mode="floor")[None, :]
        + torch.arange(
            num_groups, dtype=torch.int64, device=device
        )[:, None]
    ) % rhs_scale_palette.numel()
    rhs_scales = rhs_scale_palette[
        rhs_scale_indices
    ].unsqueeze(-1).contiguous()

    selected_rhs = rhs[group_ids, :, selected_k]
    assert selected_rhs.shape == (rows, hidden)
    expected_rows = (
        selected_rhs.float()
        * lhs_scales
        * rhs_scales[group_ids, :, 0]
    ).to(torch.bfloat16).contiguous()
    del selected_rhs

    guard_elements = 64
    m_indices = group_ids.to(torch.int32).contiguous()
    m_rows = torch.full(
        (num_groups,), rows_per_group,
        dtype=torch.int32, device=device
    )
    valid_m_rows = (
        m_rows
        if valid_rows_per_group == rows_per_group
        else torch.full(
            (num_groups,), valid_rows_per_group,
            dtype=torch.int32, device=device
        )
    )
    valid_mask = rows_in_group < valid_rows_per_group
    token = torch.div(row_ids, topk, rounding_mode="floor")
    slot = row_ids % topk
    packed_token_meta = (slot << 24) | token
    token_meta = torch.where(
        valid_mask,
        packed_token_meta,
        torch.zeros_like(packed_token_meta),
    ).contiguous()

    combine_sentinel = torch.tensor(
        -37.0, dtype=torch.bfloat16, device=device
    )
    combine_raw = torch.full(
        (
            guard_elements
            + topk * max_tokens * hidden
            + guard_elements,
        ),
        -37.0,
        dtype=torch.bfloat16,
        device=device,
    )
    combine = combine_raw.narrow(
        0, guard_elements, topk * max_tokens * hidden
    ).view(topk, max_tokens, hidden)
    sym_offsets = torch.zeros(
        1, dtype=torch.int64, device=device
    )

    expected = torch.full(
        (topk, max_tokens, hidden),
        combine_sentinel,
        dtype=torch.bfloat16,
        device=device,
    )
    expected[slot[valid_mask], token[valid_mask]] = (
        expected_rows[valid_mask]
    )

    selected = _select_glm_channelwise_scatter_config(
        physical_expected_m, hidden, k, num_groups, get_num_sms())
    assert selected is not None
    dg.mega_moe_gemm2_scatter(
        (lhs, lhs_scales), (rhs, rhs_scales), token_meta, combine.reshape(-1), sym_offsets,
        m_rows, valid_m_rows, 0, max_tokens, topk, precision="fp8_channelwise", m_indices=m_indices)
    torch.cuda.synchronize()
    torch.testing.assert_close(combine, expected, rtol=0, atol=0)
    assert (combine_raw[:guard_elements] == combine_sentinel).all()
    assert (combine_raw[-guard_elements:] == combine_sentinel).all()


@pytest.mark.parametrize("precision, activation, quantization", (
    ("fp8_channelwise", "situ", None),
    ("fp8_channelwise", "step", "fp8"),
    ("fp8_blockwise", "swiglu", None),
    ("fp8_blockwise", None, "fp8"),
))
def test_api_rejects_unsupported_options(precision, activation, quantization):
    # Reject unsupported combinations before touching a workspace or launching a collective.
    empty = torch.empty(0)
    with pytest.raises(RuntimeError, match="unsupported precision/activation/output_quantization"):
        dg.mega_moe_dispatch_gemm1(
            None, (empty, empty), empty, 0, 0, precision=precision,
            activation=activation, output_quantization=quantization)


@pytest.mark.parametrize("case", CHANNELWISE_CASES[1:], ids=lambda c: c.name)
def test_tuning_profiles(case, monkeypatch):
    monkeypatch.delenv("DG_MOE_USE_TUNED_CONFIGS", raising=False)
    if torch.cuda.get_device_name() != "ZW-M890P" or dg.get_num_sms() != 39:
        pytest.skip("measured profiles target ZW-M890P with 39 SMs")
    args = dict(num_ranks=case.worlds[0], num_experts=case.experts, num_topk=case.topk,
                hidden=case.hidden, intermediate_hidden=case.intermediate,
                precision="fp8_channelwise", activation="swiglu_oai" if case.name == "minimax_m3" else "swiglu")
    for tokens in MODEL_TOKENS:
        assert dg.get_mega_moe_tuning_config(**args, num_tokens=tokens).profile == case.name
    for tokens in (0, 1, 1023, 1025, 2047, 2049, 4095, 4097):
        assert not dg.get_mega_moe_tuning_config(**args, num_tokens=tokens).profile
    for change in (dict(precision="fp8_blockwise"), dict(num_ranks=case.worlds[0] // 2)):
        assert not dg.get_mega_moe_tuning_config(**(args | change), num_tokens=1024).profile
    if case.name in ("glm53_flash", "qwen38_flash"):
        assert tuple(dg.get_mega_moe_tuning_config(**args, num_tokens=1024).gemm2_tile) == (128, 256, 128, 32, 64, 3)
        assert dg.get_mega_moe_tuning_config(**args, num_tokens=2048).gemm2_tile is None
    if case.name == "minimax_m3":
        assert not dg.get_mega_moe_tuning_config(**(args | dict(activation="swiglu")), num_tokens=1024).profile
    monkeypatch.setenv("DG_MOE_USE_TUNED_CONFIGS", "0")
    assert not dg.get_mega_moe_tuning_config(**args, num_tokens=1024).profile


@pytest.mark.parametrize("precision", ("fp8_channelwise", "fp8_blockwise"))
@pytest.mark.parametrize("options", ({"intermediate_hidden": 512}, {"config_override": {}}))
def test_buffer_options_rejected(precision, options, monkeypatch):
    from types import SimpleNamespace

    def no_allocation(*args, **kwargs):
        pytest.fail("invalid buffer options must fail before symmetric allocation")
    monkeypatch.setattr(symm_mem, "empty", no_allocation)
    with pytest.raises(RuntimeError, match="does not select GEMM1 tiles|only supported for w4a4"):
        dg.get_symm_buffer_for_mega_moe(
            SimpleNamespace(size=lambda: 1), 8, 64, 2, 512, precision=precision, **options)


@pytest.mark.parametrize("precision, block_m, scale_bytes, num_threads", (
    ("fp8_channelwise", 256, 4, 512), ("fp8_blockwise", 128, 128, 256),
))
def test_buffer_config(precision, block_m, scale_bytes, num_threads, monkeypatch):
    from types import SimpleNamespace
    from deep_gemm import mega_moe

    monkeypatch.setattr(mega_moe, "Fp8MegaMoEDispatchSymmBuffer", lambda group, config: config)
    args = dict(num_experts=128, num_topk=8, hidden=4096,
                num_max_tokens_per_rank=4096, precision=precision)
    config = dg.get_symm_buffer_for_mega_moe(SimpleNamespace(size=lambda: 4), **args)
    assert config["TOKEN_BYTES"] == 4096 and config["SF_BYTES"] == scale_bytes
    assert config["NUM_EXPERTS_PER_RANK"] == 32 and config["BLOCK_M"] == block_m
    assert config["NUM_THREADS"] == num_threads
    assert not any(key.startswith("GEMM1_") for key in config)
