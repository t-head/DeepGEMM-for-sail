"""W4A16 public routed MoE pipeline: SiTU, replica weights and EP8/EP16."""

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import deep_gemm
from moe_test_utils import (
    MODEL_TOKENS, CombineBarrier, MoECase, activation, calc_diff, check_graph_replay,
    clear_dispatch_counts, compile_kernels, gather, interleave_gate_up, moe_group, quant_w4a16_mxfp4, relative_l2,
)

PIPELINE_CASES = (
    MoECase("smoke", 256, 128, 4, 2, 8, tokens=(4,), worlds=(1,)),
    MoECase("situ_large_shape", 3584, 3072, 4, 2, 8, tokens=(4,), worlds=(1,)),
)


def _quantize_weight(weight: torch.Tensor):
    weight_ref, packed, scales = quant_w4a16_mxfp4(weight)
    return weight_ref, deep_gemm.preprocess_w4a16_mxfp4((packed, scales))


def _make_weight(n: int, k: int, generator: torch.Generator, num_experts: int):
    weight = torch.randn(
        num_experts,
        n,
        k,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ) * 0.125
    weight_ref, processed = _quantize_weight(weight)
    return weight, weight_ref, processed


def _deinterleave_gate_up(x: torch.Tensor) -> torch.Tensor:
    """Convert physical G=8 gate/up pairs back to logical halves."""
    if x.dim() == 2:
        physical = x.view(x.shape[0], -1, 2, 8)
        gate = physical[:, :, 0].reshape(x.shape[0], -1)
        up = physical[:, :, 1].reshape(x.shape[0], -1)
        return torch.cat((gate, up), dim=1).contiguous()
    assert x.dim() == 3
    physical = x.view(x.shape[0], -1, 2, 8, x.shape[2])
    gate = physical[:, :, 0].reshape(x.shape[0], -1, x.shape[2])
    up = physical[:, :, 1].reshape(x.shape[0], -1, x.shape[2])
    return torch.cat((gate, up), dim=1).contiguous()


@pytest.mark.parametrize("case", PIPELINE_CASES, ids=lambda c: c.name)
def test_pipeline(case, moe_group, monkeypatch):
    # An obsolete layout override must not change the shared scatter/combine contract.
    monkeypatch.setenv("DG_MOE_COMBINE_TOKEN_MAJOR", "1")
    case.require_world()
    num_experts, topk, cap = case.experts, case.topk, case.capacity
    num_tokens, = case.tokens
    hidden_size, intermediate = case.hidden, case.intermediate
    generator = torch.Generator(device="cuda").manual_seed(20260824)
    hidden = (torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16,
                          device="cuda", generator=generator)).contiguous()
    topk_ids = torch.tensor([[0, 1], [2, 3], [1, 2], [3, 0]],
                            dtype=torch.int64, device="cuda")
    topk_weights = torch.tensor([[0.75, 0.25], [0.6, 0.4], [0.3, 0.7], [0.55, 0.45]],
                                dtype=torch.float32, device="cuda")
    w13_source, w13_ref, _ = _make_weight(2 * intermediate, hidden_size, generator, num_experts)
    w13_physical_ref, w13 = _quantize_weight(interleave_gate_up(w13_source))
    _, w2_ref, w2 = _make_weight(hidden_size, intermediate, generator, num_experts)
    torch.testing.assert_close(_deinterleave_gate_up(w13_physical_ref), w13_ref, rtol=0.0, atol=0.0)

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        moe_group, num_experts=num_experts, num_max_tokens_per_rank=cap,
        num_topk=topk, hidden=hidden_size, precision="w4a16",
        intermediate_hidden=2 * intermediate, block_m=64)
    m_rows, valid_m_rows, m_offsets, src_m_rows, src_m_offsets, packed_src_info = buffer.get_gemm2_metadata()
    activated_capacity = torch.empty(buffer.num_pool, intermediate,
                                     dtype=torch.bfloat16, device="cuda")
    deep_gemm.mega_moe_pre_dispatch(
        hidden, topk_ids, topk_weights, buffer, num_tokens, precision="w4a16")
    block_m = deep_gemm.mega_moe_dispatch_gemm1(
        buffer, (w13[0], w13[1]), activated_capacity, 0, num_tokens, precision="w4a16",
        activation="situ")
    active_rows = int(m_rows.sum().item())
    assert valid_m_rows.cpu().tolist() == [2, 2, 2, 2]
    assert active_rows == num_experts * block_m
    activated = activated_capacity[:active_rows]

    # Fused dispatch/GEMM1 feeds GEMM2 directly. Split master/replica weights
    # exercise the GEMM2 expert boundary using the same activation tensor.
    combine = torch.zeros(topk * cap * hidden_size, dtype=torch.bfloat16, device="cuda")
    replica_combine = torch.zeros_like(combine)
    barrier = CombineBarrier(moe_group)
    sym_offsets = torch.zeros(1, dtype=torch.int64, device="cuda")
    w2_master = tuple(t[:3].contiguous() for t in w2)
    w2_replicas = tuple(t[3:].contiguous() for t in w2)

    def gemm2_combine(weights, destination, replicas=None):
        deep_gemm.mega_moe_gemm2_scatter(
            activated, weights, packed_src_info, destination, sym_offsets, m_rows, valid_m_rows, 0, cap, topk,
            precision="w4a16", rhs_replicas=replicas)
        # Keep the consumer adjacent to scatter: a reference assertion or a
        # host synchronize here would hide a missing device completion fence.
        return deep_gemm.mega_moe_combine(
            destination, topk_weights, num_tokens, hidden_size, topk, cap,
            barrier.pointers, 0, 1, barrier.done)

    # A bulk BF16 warp store writes 256 columns. Reject unsafe overrides before JIT/launch.
    for block_n in (64, 128):
        with pytest.raises(RuntimeError, match="complete 256-column tiles"):
            deep_gemm.mega_moe_gemm2_scatter(
                activated, w2, packed_src_info, combine, sym_offsets, m_rows, valid_m_rows,
                0, cap, topk, precision="w4a16",
                configs=(deep_gemm.get_num_sms(), 64, block_n, 128, 32, 64, 128, 2))

    actual = gemm2_combine(w2, combine)
    replica_actual = gemm2_combine(w2_master, replica_combine, w2_replicas)

    # Ordinary GEMM + Torch activation is only the reference for fused GEMM1;
    # no standalone GEMM/SiTU extension is needed by the execution path.
    empty_indices = torch.empty(0, dtype=torch.int32, device="cuda")
    g1_physical = torch.empty(active_rows, 2 * intermediate,
                              dtype=torch.bfloat16, device="cuda")
    deep_gemm.m_grouped_gemm_w4a16_nopad(
        buffer.dispatched_token_bf16[:active_rows], w13, g1_physical, empty_indices, m_rows)
    reference_activated = activation(_deinterleave_gate_up(g1_physical), "situ").bfloat16()
    # GEMM2 consumes only routed rows; padding has no defined activation value.
    for start, count in zip(m_offsets.tolist(), valid_m_rows.tolist()):
        torch.testing.assert_close(activated[start:start + count], reference_activated[start:start + count],
                                   rtol=0.025, atol=0.025)

    # Compare GEMM2/scatter to the existing ordinary GEMM and explicit scatter.
    g2 = torch.empty(active_rows, hidden_size, dtype=torch.bfloat16, device="cuda")
    deep_gemm.m_grouped_gemm_w4a16_nopad(activated, w2, g2, empty_indices, m_rows)
    reference_combine = torch.zeros(topk, cap, hidden_size, dtype=torch.bfloat16, device="cuda")
    for expert in range(num_experts):
        start, rows = int(m_offsets[expert]), int(valid_m_rows[expert])
        meta = buffer.token_src_metadata[start:start + rows].long()
        assert bool((meta[:, 0] == 0).all())
        reference_combine[meta[:, 2], meta[:, 1]] = g2[start:start + rows]
    torch.testing.assert_close(combine.view(topk, cap, hidden_size)[:, :num_tokens],
                               reference_combine[:, :num_tokens], rtol=0, atol=0)
    # Match the kernel's FP32 fused multiply-add in slot order before BF16 rounding.
    reference_actual = torch.zeros(num_tokens, hidden_size, device="cuda")
    for slot in range(topk):
        reference_actual.addcmul_(reference_combine[slot, :num_tokens].float(),
                                 topk_weights[:, slot, None])
    reference_actual = reference_actual.bfloat16()
    torch.testing.assert_close(replica_actual, actual, rtol=0.0, atol=0.0)
    torch.testing.assert_close(reference_actual, actual, rtol=0.0, atol=0.0)

    # Independent full-token reference using dequantized weights.
    expected = torch.zeros_like(actual)
    for token in range(num_tokens):
        for slot in range(topk):
            expert = int(topk_ids[token, slot])
            g1_ref = hidden[token].float() @ w13_ref[expert].float().T
            act_ref = activation(g1_ref.bfloat16().view(1, -1), "situ").bfloat16()[0]
            g2_ref = act_ref.float() @ w2_ref[expert].float().T
            expected[token] += (g2_ref.bfloat16() * topk_weights[token, slot]).bfloat16()
    diff = calc_diff(actual, expected)
    error = actual.float() - expected.float()
    rmse = error.square().mean().sqrt().item()
    ref_rms = expected.float().square().mean().sqrt().item()
    rel_rmse = rmse / max(ref_rms, 1e-12)
    max_abs = error.abs().max().item()
    assert diff < 0.005, f"W4A16 MegaMoE diff={diff:.6f}"
    if case.name == "situ_large_shape":
        assert rel_rmse < 0.005, f"SiTU relative RMSE={rel_rmse:.6f}"
        assert max_abs / ref_rms < 0.03, f"SiTU normalized max error={max_abs / ref_rms:.6f}"
    else:
        torch.testing.assert_close(actual, expected, rtol=0.08, atol=0.08)

    # Exercise the former standalone IPC-barrier boundary under graph replay.
    # Poison destinations before each replay so an earlier eager result cannot
    # satisfy the check when the new scatter has not completed its bulk stores.
    saved_activated = activated.clone()
    golden = torch.stack((actual, replica_actual)).clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = torch.stack((gemm2_combine(w2, combine),
                                gemm2_combine(w2_master, replica_combine, w2_replicas)))
    for factor in (0.0, -1.0, 1.0):
        activated.copy_(saved_activated * factor)
        combine.fill_(float("nan"))
        replica_combine.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        # GEMM2 and weighted combine are linear; these factors have exact BF16
        # representations and do not introduce an independent rounding error.
        torch.testing.assert_close(replayed, golden * factor, rtol=0, atol=0)
    print(f"[PASS] fused BF16 dispatch/W4A16 G1/SiTU -> G2/scatter -> combine, "
          f"shape={case.name}, diff={diff:.8g}, rel_rmse={rel_rmse:.8g}")


@pytest.mark.parametrize("model,num_tokens", (
    pytest.param("smoke", 4, id="smoke"),
    *(pytest.param("kimi_k3", tokens, id=f"kimi_k3_compute_{tokens}")
      for tokens in MODEL_TOKENS),
))
def test_distributed_pipeline(model, num_tokens, moe_group):
    """One public pipeline; Torch checks routing, landed bytes and both GEMMs."""
    rank, world = dist.get_rank(), dist.get_world_size()
    kimi = model == "kimi_k3"
    if world not in ((16,) if kimi else (8, 16)):
        pytest.skip("distributed smoke requires EP8/16; Kimi requires EP16")
    topk, cap = (16, max(MODEL_TOKENS)) if kimi else (2, 4)
    hidden_size, intermediate = (3584, 3072) if kimi else (256, 128)
    num_experts = 896 if kimi else world
    experts_per_rank = num_experts // world
    token = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    columns = torch.arange(hidden_size, device="cuda", dtype=torch.float32)
    hidden = torch.sin(columns[None, :] * 0.017 + rank * 0.31
                       + token[:, None] * 0.2).to(torch.bfloat16)
    if kimi:
        # Every source sends one slot to every rank; tokens cycle across all
        # 56 local experts, including the last expert and partially filled tiles.
        topk_ids = (torch.arange(world, device="cuda") * experts_per_rank)[None, :]
        topk_ids = (topk_ids + ((rank * num_tokens + token) % experts_per_rank)[:, None]).contiguous()
        topk_weights = torch.full((num_tokens, topk), 1.0 / topk, device="cuda")
    else:
        topk_ids = torch.stack(((rank + token + 1) % world,
                                (rank + token + 7) % world), dim=1).contiguous()
        topk_weights = torch.tensor([[0.75, 0.25], [0.6, 0.4], [0.3, 0.7], [0.55, 0.45]],
                                    dtype=torch.float32, device="cuda")

    generator = torch.Generator(device="cuda").manual_seed(20260826 + rank)
    # GEMM1 weights are stored in physical G=8 gate/up order.
    _, w13_ref, w13 = _make_weight(2 * intermediate, hidden_size, generator, experts_per_rank)
    _, w2_ref, w2 = _make_weight(hidden_size, intermediate, generator, experts_per_rank)
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        moe_group, num_experts=num_experts, num_max_tokens_per_rank=cap,
        num_topk=topk, hidden=hidden_size, precision="w4a16", intermediate_hidden=2 * intermediate)
    activated = torch.empty(buffer.num_pool, intermediate, dtype=torch.bfloat16, device="cuda")
    m_rows, valid_m_rows, m_offsets, src_m_rows, src_m_offsets, packed_src_info = buffer.get_gemm2_metadata()
    combine = symm_mem.empty(topk * cap * hidden_size, dtype=torch.bfloat16, device="cuda")
    handle = symm_mem.rendezvous(combine, group=moe_group)
    pointers = [handle.get_buffer(r, (combine.numel(),), torch.bfloat16).data_ptr() for r in range(world)]
    offsets = torch.tensor([p - pointers[rank] for p in pointers], dtype=torch.int64, device="cuda")
    barrier = CombineBarrier(moe_group)

    tuning = deep_gemm.get_mega_moe_tuning_config(
        world, num_experts, topk, hidden_size, intermediate, num_tokens,
        precision="w4a16", activation="situ")

    def pipeline():
        deep_gemm.mega_moe_pre_dispatch(
            hidden, topk_ids, topk_weights, buffer, num_tokens, precision="w4a16")
        deep_gemm.mega_moe_dispatch_gemm1(
            buffer, w13, activated, rank, num_tokens, precision="w4a16", activation="situ", tuning=tuning)
        deep_gemm.mega_moe_gemm2_scatter(
            activated, w2, packed_src_info, combine, offsets, m_rows, valid_m_rows,
            rank, cap, topk, precision="w4a16", tuning=tuning)
        clear_dispatch_counts(buffer, num_experts)
        return deep_gemm.mega_moe_combine(
            combine, topk_weights, num_tokens, hidden_size, topk, cap,
            barrier.pointers, rank, world, barrier.done, tuning=tuning)

    try:
        # All ranks finish JIT before cooperative device barriers can execute.
        compile_kernels(pipeline)
        combine.fill_(float("nan"))
        clear_dispatch_counts(buffer, num_experts)
        torch.cuda.synchronize()
        dist.barrier()
        actual = pipeline()

        # The oracle starts from source inputs/routes, independently of kernel
        # metadata. Metadata must describe exactly the same set of routes.
        all_hidden, all_ids, all_weights = gather(hidden), gather(topk_ids), gather(topk_weights)
        reference = torch.zeros(world * num_tokens, hidden_size, device="cuda")
        assert int(valid_m_rows.sum()) == (world * num_tokens if kimi else num_tokens * topk)
        if kimi and num_tokens >= 1024:
            assert bool((valid_m_rows > 0).all())
        for expert in range(experts_per_rank):
            routes = (all_ids == rank * experts_per_rank + expert).nonzero()
            rows = len(routes)
            assert int(valid_m_rows[expert]) == rows
            if not rows:
                continue
            start = int(m_offsets[expert])
            meta = buffer.token_src_metadata[start:start + rows].long()
            src, tok, slot = meta.unbind(-1)
            actual_keys = (src * num_tokens + tok) * topk + slot
            expected_keys = (routes[:, 0] * num_tokens + routes[:, 1]) * topk + routes[:, 2]
            torch.testing.assert_close(actual_keys.sort().values, expected_keys.sort().values, rtol=0, atol=0)
            landed = all_hidden[src, tok]
            torch.testing.assert_close(buffer.dispatched_token_bf16[start:start + rows], landed, rtol=0, atol=0)
            torch.testing.assert_close(packed_src_info[start:start + rows],
                                       (src << 32) | (slot << 24) | tok, rtol=0, atol=0)
            projection = (landed.float() @ w13_ref[expert].float().T).bfloat16()
            expected_activated = activation(_deinterleave_gate_up(projection), "situ").bfloat16()
            torch.testing.assert_close(activated[start:start + rows], expected_activated, rtol=0.025, atol=0.025)
            output = (expected_activated.float() @ w2_ref[expert].float().T).bfloat16()
            reference.index_add_(0, src * num_tokens + tok,
                                 output.float() * all_weights[src, tok, slot, None])
        dist.all_reduce(reference)
        expected = reference.view(world, num_tokens, hidden_size)[rank].bfloat16()
        assert actual.isfinite().all()
        error = relative_l2(actual, expected)
        assert error < 0.005, (rank, error)

        # Reuse the same buffer and generation counters. Poisoned receive
        # buffers and changed inputs catch stale scatter data during graph replay.
        def poisoned_pipeline():
            combine.fill_(float("nan"))
            return pipeline()
        check_graph_replay(poisoned_pipeline, actual, input_tensor=hidden)
        if rank == 0:
            print(f"[PASS] EP{world} W4A16 public pipeline, model={model}, "
                  f"tokens/rank={num_tokens}, relative_l2={error:.8g}")
    finally:
        buffer.destroy()


@pytest.mark.parametrize("activation, quantization", (
    (None, None),
    ("swiglu", None),
    ("step", None),
    ("swiglu_oai", None),
    ("invalid", None),
    ("situ", "mxfp4"),
    ("situ", "fp8"),
))
def test_api_rejects_unsupported_options(activation, quantization):
    # Reject unsupported combinations before touching a workspace or launching a collective.
    empty = torch.empty(0)
    with pytest.raises(RuntimeError, match="unsupported precision/activation/output_quantization"):
        deep_gemm.mega_moe_dispatch_gemm1(
            None, (empty, empty), empty, 0, 0, precision="w4a16",
            activation=activation, output_quantization=quantization)


def test_tuning_profiles(monkeypatch):
    monkeypatch.delenv("DG_MOE_USE_TUNED_CONFIGS", raising=False)
    if torch.cuda.get_device_name() != "ZW-M890P" or deep_gemm.get_num_sms() != 39:
        pytest.skip("measured profiles target ZW-M890P with 39 SMs")
    args = dict(num_ranks=16, num_experts=896, num_topk=16, hidden=3584,
                intermediate_hidden=3072, precision="w4a16", activation="situ")
    for tokens, block_m in ((1024, 128), (2048, 256), (4096, 256)):
        config = deep_gemm.get_mega_moe_tuning_config(**args, num_tokens=tokens)
        assert config.profile == "kimi_k3" and config.gemm1_block_m == block_m
    for tokens in (0, 1, 1023, 1025, 2047, 2049, 4095, 4097):
        assert not deep_gemm.get_mega_moe_tuning_config(**args, num_tokens=tokens).profile
    for change in (dict(num_ranks=8), dict(activation="swiglu")):
        assert not deep_gemm.get_mega_moe_tuning_config(**(args | change), num_tokens=1024).profile
    monkeypatch.setenv("DG_MOE_USE_TUNED_CONFIGS", "0")
    assert not deep_gemm.get_mega_moe_tuning_config(**args, num_tokens=1024).profile


@pytest.mark.parametrize("options", ({}, {"intermediate_hidden": 512, "gemm1_block_n": 128}))
def test_buffer_options_rejected(options, monkeypatch):
    from types import SimpleNamespace

    def no_allocation(*args, **kwargs):
        pytest.fail("invalid buffer options must fail before symmetric allocation")
    monkeypatch.setattr(symm_mem, "empty", no_allocation)
    with pytest.raises(RuntimeError, match="requires intermediate_hidden|only supported for w4a4"):
        deep_gemm.get_symm_buffer_for_mega_moe(
            SimpleNamespace(size=lambda: 1), 8, 64, 2, 512, precision="w4a16", **options)


def test_buffer_config(monkeypatch):
    from types import SimpleNamespace
    from deep_gemm import mega_moe

    monkeypatch.setattr(mega_moe, "MegaMoEDispatchGemm1SymmBuffer", lambda group, config, **kwargs: config)
    args = dict(num_experts=896, num_topk=16, hidden=3584,
                precision="w4a16", intermediate_hidden=6144)
    for capacity in (1, 33, 1024, 2048, 4096):
        config = deep_gemm.get_symm_buffer_for_mega_moe(
            SimpleNamespace(size=lambda: 16), **args, num_max_tokens_per_rank=capacity)
        assert config["TOKEN_BYTES"] == 7168 and config["SF_BYTES"] == 0
        assert config["NUM_EXPERTS_PER_RANK"] == 56 and config["BLOCK_M"] == 64
        assert config["NMTPR"] == capacity
