"""Routed MoE acceptance tests; see docs/mega_moe_training.md for launch commands."""

import argparse
import hashlib
import json
import os
from benchmark_mega_moe_training import (
    ROOT,
    configure,
    destroy_training_process_group,
    load_recipes,
    materialize_lut,
    recipe_arguments,
    select_recipe,
)


def load_test_recipes():
    book = load_recipes()
    for recipe in book["recipes"]:
        materialize_lut(recipe, book)
    return book


def test_prepare(options):
    import torch
    import deep_gemm as dg

    torch.cuda.set_device(0)
    # The selected vector activation must be compiled before any peer waits.
    # A generic mode0 binary does not cover the production mode4 specialization.
    from pathlib import Path

    cache = Path(os.environ.get("DG_JIT_CACHE_DIR") or Path.home() / ".deep_gemm") / "cache"
    dg._C._mega_moe_training_precompile(0, [27, 28], 4)
    before = {p.name for p in cache.glob("kernel.*") if p.is_dir()}
    for operation in (27, 28):
        assert any(f"common_{operation}_mn_scales_vec8." in name for name in before)
    gate = torch.zeros((128, 256), device="cuda", dtype=torch.bfloat16)
    hidden = torch.zeros((128, 128), device="cuda", dtype=torch.bfloat16)
    active = torch.zeros((), device="cuda", dtype=torch.int32)
    status = torch.zeros_like(active)
    for backward in (False, True):
        width = 256 if backward else 128
        values = torch.empty((128, width), device="cuda", dtype=torch.float8_e4m3fn)
        scales = torch.empty((width // 128, 128), device="cuda")
        tensors = [gate, hidden] + ([gate] if backward else [])
        dg._C._mega_moe_training_common_launch(
            28 if backward else 27,
            tensors + [values, scales, active, status],
            [128, 128, 1, 4],
        )
    torch.cuda.synchronize()
    assert not status.item()
    assert before == {p.name for p in cache.glob("kernel.*") if p.is_dir()}
    print(json.dumps(dict(case="activation_precompile_contract", passed=True)), flush=True)


def test_recipes():
    book = load_recipes()
    assert len(book["recipes"]) == 24
    identities = set()
    for recipe in book["recipes"]:
        args = recipe_arguments(recipe)
        assert recipe["id"] not in identities
        identities.add(recipe["id"])
        assert args.local_experts * recipe["ep"] == 512 and args.topk == 10
        assert args.fresh_weight_gradients and args.scope == "step"
        assert (args.warmup, args.iterations, args.repeats) == (5, 20, 5)
        assert args.hidden == (4096 if recipe["model"] == "Qwen3.5-397B-A17B" else 8192)
        assert args.intermediate == (1024 if recipe["model"] == "Qwen3.5-397B-A17B" else 2048)
        assert args.tokens == (4096 if recipe["model"] == "Qwen3.5-397B-A17B" else 16384)
        assert (
            recipe["required_deepep_scaleout_groups"] == 1
            if recipe["backend"] == "deepep"
            else True
        )
        environment = configure(recipe, book, write_lut=False)
        assert environment["DG_MOE_FP8_WGRAD"] == ("1" if recipe["precision"] == "fp8" else "0")
        for name, contents in book["lut_profiles"][recipe["lut_profile"]].items():
            assert hashlib.sha256(contents.encode()).hexdigest() == recipe["lut_sha256"][name]
    print(json.dumps(dict(case="moe_recipe_contracts", recipes=24, passed=True)), flush=True)


def test_capacity(options):
    import json

    import torch

    from deep_gemm.training.deepep import plan_compact_route_capacity

    for experts in (8, 16, 256):
        ids = torch.full((4, 2), -1, dtype=torch.int64)
        plan = plan_compact_route_capacity(ids, 0, experts, 2, alignment=128)
        assert plan["capacity"] == (256 if experts == 256 else 128)
        assert plan["capacity"] >= experts and plan["capacity"] % 128 == 0
        assert plan["representative_max_padded_rows"] == 0
        print(
            json.dumps(
                dict(
                    case="compact_capacity_cpu_contract",
                    experts=experts,
                    empty=True,
                    capacity=plan["capacity"],
                    passed=True,
                )
            ),
            flush=True,
        )

    plan = plan_compact_route_capacity(
        torch.full((1, 1), -1, dtype=torch.int64), 0, 16, 1, alignment=1
    )

    assert plan["capacity"] == plan["worst_case_rows"] == 16

    print(
        json.dumps(
            dict(
                case="compact_capacity_cpu_contract",
                experts=16,
                alignment=1,
                empty=True,
                capacity=plan["capacity"],
                passed=True,
            )
        ),
        flush=True,
    )

    ids = torch.tensor([[0, 1], [2, -1], [1, 2], [0, 2]], dtype=torch.int64)

    for experts in (8, 16):
        plan = plan_compact_route_capacity(ids, 3, experts, 2, alignment=128, headroom=1.25)
        assert plan["representative_max_padded_rows"] == 384 and plan["capacity"] == 512
        print(
            json.dumps(
                dict(
                    case="compact_capacity_cpu_contract",
                    experts=experts,
                    empty=False,
                    capacity=plan["capacity"],
                    passed=True,
                )
            ),
            flush=True,
        )

    print(json.dumps(dict(case="compact_capacity_cpu_finished", passed=True)), flush=True)


def test_bf16_dw(options):
    import torch
    import deep_gemm as dg

    torch.cuda.set_device(0)
    torch.manual_seed(718)
    groups, m, n, capacity = 3, 128, 128, 512
    a = (torch.randn(capacity, m, device="cuda") * 0.1).bfloat16()
    b = (torch.randn(capacity, n, device="cuda") * 0.1).bfloat16()
    initial = torch.randn(groups, m, n, device="cuda") * 0.01
    ks = [128, 0, 256]
    lengths = torch.tensor(ks, device="cuda", dtype=torch.int32)
    aa, bb = a.cpu().double(), b.cpu().double()

    def reference(values):
        result, start = [], 0
        for k in values:
            result.append(aa[start : start + k].T @ bb[start : start + k])
            start += k
        return torch.stack(result)

    def check(name, output, expected, case="bf16_dw_api"):
        torch.testing.assert_close(output.cpu().double(), expected, rtol=2e-4, atol=2e-5)
        print(json.dumps(dict(case=case, name=name, passed=True)), flush=True)

    product = reference(ks)
    initial_cpu = initial.cpu().double()
    for mode in ("host", "device"):
        for separate in (False, True):
            out = torch.full_like(initial, float("nan")) if separate else initial.clone()
            c = initial if separate else out
            args = (
                (a[: sum(ks)], b[: sum(ks)], out, ks, lengths)
                if mode == "host"
                else (a, b, out, None, lengths)
            )
            dg.k_grouped_bf16_gemm_tn_contiguous(*args, c=c)
            check(f"{mode}_{'separate' if separate else 'inplace'}", out, initial_cpu + product)
            torch.testing.assert_close(initial.cpu().double(), initial_cpu, rtol=0, atol=0)

    out = torch.full_like(initial, float("nan"))
    dg.k_grouped_bf16_gemm_tn_contiguous(a, b, out, None, lengths, max_group_k=256)
    check("device_overwrite", out, product)
    flags = torch.tensor([1, 0, 1], device="cuda", dtype=torch.int32)
    out.copy_(initial)
    dg.k_grouped_bf16_gemm_tn_contiguous(a, b, out, None, lengths, c=out, first_flags=flags)
    expected = initial_cpu.clone()
    expected[0].zero_()
    expected[2].zero_()
    check("device_first_nonempty", out, expected + product)

    zero_lengths = torch.zeros_like(lengths)
    dg.k_grouped_bf16_gemm_tn_contiguous(a[:0], b[:0], out, [0] * groups, zero_lengths, c=initial)
    check("host_zero_k", out, initial_cpu)

    dg.k_grouped_bf16_gemm_tn_contiguous(a, b, out, None, lengths)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dg.k_grouped_bf16_gemm_tn_contiguous(a, b, out, None, lengths)
    for values in (ks, [0, 256, 128], [0, 0, 0]):
        lengths.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
        graph.replay()
        check("device_mutable_graph", out, reference(values))
    graph.reset()
    lengths.copy_(torch.tensor(ks, device="cuda", dtype=torch.int32))

    # Preserve upstream's padded/unequal capacities and device-only inclusive
    # prefixes alongside the training device-length extension.
    for name, left, right in (("padded_unequal", a, b[:416]), ("unequal_reverse", a[:416], b)):
        dg.k_grouped_bf16_gemm_tn_contiguous(left, right, out, ks, lengths, c=initial)
        check(name, out, initial_cpu + product, "bf16_dw_upstream")
    prefixes = lengths.cumsum(0, dtype=torch.int32)
    for host in (ks, None):
        dg.k_grouped_bf16_gemm_tn_contiguous(
            a, b[:416], out, host, prefixes, c=initial, use_psum_layout=True
        )
        check(
            "host_psum" if host else "device_psum", out, initial_cpu + product, "bf16_dw_upstream"
        )
    prefix_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(prefix_graph):
        dg.k_grouped_bf16_gemm_tn_contiguous(
            a, b[:416], out, None, prefixes, c=initial, use_psum_layout=True
        )
    for values in (ks, [0, 256, 128], [0, 0, 0]):
        prefixes.copy_(
            torch.tensor(values, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
        )
        prefix_graph.replay()
        check("mutable_device_psum", out, initial_cpu + reference(values), "bf16_dw_upstream")
    prefix_graph.reset()
    dg.k_grouped_bf16_gemm_tn_contiguous(a[:0], b[:128], out, [0] * groups, zero_lengths, c=initial)
    check("zero_total_unequal", out, initial_cpu, "bf16_dw_upstream")
    for name, host, metadata, extra in (
        ("host_sum_exceeds_capacity", [128, 128, 256], lengths, {}),
        ("prefix_training_flags", None, prefixes, dict(use_psum_layout=True, first_flags=flags)),
    ):
        out.fill_(37)
        try:
            dg.k_grouped_bf16_gemm_tn_contiguous(a, b[:416], out, host, metadata, c=out, **extra)
        except RuntimeError:
            assert torch.all(out == 37).item()
        else:
            raise AssertionError(name)
        print(
            json.dumps(dict(case="bf16_dw_upstream_rejected", name=name, passed=True)), flush=True
        )

    segments_a, segments_b = [a[:256], a[256:384]], [b[:256], b[256:384]]
    offsets = [
        torch.tensor(v, device="cuda", dtype=torch.int32)
        for v in ([0, 128, 128, 256], [0, 0, 0, 128])
    ]
    for accumulate in (False, True):
        out.copy_(initial if accumulate else torch.full_like(initial, float("nan")))
        dg.k_grouped_bf16_gemm_tn_segmented(
            segments_a, segments_b, out, offsets, 256, accumulate=accumulate
        )
        check(f"segmented_{accumulate}", out, product + (initial_cpu if accumulate else 0))

    for name, host_ks, kwargs in (
        ("bad_device_bound", None, dict(c=initial, max_group_k=65)),
        ("host_device_option", ks, dict(c=initial, max_group_k=256)),
        ("flags_without_accumulator", None, dict(first_flags=flags)),
    ):
        out.fill_(37)
        try:
            dg.k_grouped_bf16_gemm_tn_contiguous(a[:384], b[:384], out, host_ks, lengths, **kwargs)
        except RuntimeError:
            assert torch.all(out == 37).item()
        else:
            raise AssertionError(f"Invalid BF16 DW metadata accepted: {name}")
        print(json.dumps(dict(case="bf16_dw_rejected", name=name, passed=True)), flush=True)
    storage = torch.full((initial.numel() + 1,), 37.0, device="cuda")
    overlap_out = storage[:-1].view_as(initial)
    try:
        dg.k_grouped_bf16_gemm_tn_contiguous(
            a, b, overlap_out, None, lengths, c=storage[1:].view_as(initial)
        )
    except RuntimeError:
        assert torch.all(storage == 37).item()
    else:
        raise AssertionError("Overlapping C/D accepted")
    print(
        json.dumps(dict(case="bf16_dw_rejected", name="partial_c_alias", passed=True)), flush=True
    )
    print(json.dumps(dict(case="bf16_dw_finished", passed=True)), flush=True)


def test_dw_alias(options):
    import json

    import torch

    import deep_gemm as dg

    torch.cuda.set_device(0)

    groups, m, n, capacity = 2, 128, 128, 512

    output = torch.full((groups, m, n), 17.0, device="cuda", dtype=torch.float32)

    a = torch.zeros((m, capacity), device="cuda", dtype=torch.float8_e4m3fn)

    b = torch.zeros((n, capacity), device="cuda", dtype=torch.float8_e4m3fn)

    sa = torch.ones((groups, m), device="cuda")

    sb = torch.ones((groups, n), device="cuda")

    lengths = torch.zeros(groups, device="cuda", dtype=torch.int32)

    offsets = torch.zeros(groups + 1, device="cuda", dtype=torch.int32)

    status = torch.zeros(1, device="cuda", dtype=torch.int32)

    flags = torch.zeros(groups, device="cuda", dtype=torch.int32)

    original = dict(
        a=a, b=b, sa=sa, sb=sb, lengths=lengths, offsets=offsets, status=status, flags=flags
    )

    before = output.clone()

    for name in original:
        values = dict(original)
        target = values[name]
        values[name] = output.view(target.dtype).flatten()[: target.numel()].view(target.shape)
        try:
            dg._C._mega_moe_training_fp8_wgrad_nt(
                (values["a"], values["sa"]),
                (values["b"], values["sb"]),
                output,
                values["lengths"],
                values["offsets"],
                values["status"],
                compile_only=True,
                first_flags=values["flags"],
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("FP8 DW accepted an output alias of " + name)
        torch.cuda.synchronize()
        assert torch.equal(output, before), name
        print(
            json.dumps(
                dict(
                    case="fp8_dw_output_alias_rejected",
                    operand=name,
                    output_unchanged=True,
                    passed=True,
                )
            ),
            flush=True,
        )

    print(json.dumps(dict(case="fp8_dw_output_alias_finished", passed=True)), flush=True)


def test_dw_longk(options):
    import json, os

    import torch

    import deep_gemm as dg

    os.environ["DG_JIT_FP8_DW_MMA_FLAGS"] = "1"

    torch.cuda.set_device(0)

    torch.set_num_threads(2)

    torch.manual_seed(629)

    device = "cuda"

    m = n = 256

    counts = [8192, 0, 6272]

    groups = len(counts)

    total = sum(counts)

    starts = [0]

    for count in counts:
        starts.append(starts[-1] + count)

    qa = torch.zeros(m, total, device=device, dtype=torch.float8_e4m3fn)

    qb = torch.zeros(n, total, device=device, dtype=torch.float8_e4m3fn)

    sa = torch.ones(total // 128, m // 64, device=device)

    sb = torch.ones(n // 128, total // 128, device=device)

    products = []

    bounds = []

    for group, count in enumerate(counts):
        if not count:
            products.append(torch.zeros(m, n, dtype=torch.float64))
            bounds.append(torch.zeros(m, n, dtype=torch.float64))
            continue
        start = starts[group]
        blocks = count // 128
        aa = torch.randint(-8, 9, (m, count), device=device).to(torch.float8_e4m3fn)
        bb = torch.randint(-8, 9, (n, count), device=device).to(torch.float8_e4m3fn)
        ascale = torch.exp2(torch.randint(-6, -2, (blocks, m // 64), device=device).float())
        bscale = torch.exp2(torch.randint(-6, -2, (n // 128, blocks), device=device).float())
        qa.flatten()[start * m : (start + count) * m].copy_(aa.flatten())
        qb.flatten()[start * n : (start + count) * n].copy_(bb.flatten())
        sa[start // 128 : (start + count) // 128].copy_(ascale)
        sb.flatten()[start // 128 * (n // 128) : (start + count) // 128 * (n // 128)].copy_(
            bscale.flatten()
        )
        av = (
            aa.cpu().float().double().reshape(m, blocks, 128)
            * ascale.cpu().double().t().repeat_interleave(64, dim=0)[..., None]
        ).reshape(m, count)
        bv = (
            bb.cpu().float().double().reshape(n, blocks, 128)
            * bscale.cpu().double().repeat_interleave(128, dim=0)[..., None]
        ).reshape(n, count)
        products.append(av @ bv.t())
        bounds.append(av.abs() @ bv.abs().t())

    product = torch.stack(products)

    absolute = torch.stack(bounds)

    lengths = torch.tensor(counts, device=device, dtype=torch.int32)

    offsets = torch.tensor(starts, device=device, dtype=torch.int32)

    status = torch.zeros(1, device=device, dtype=torch.int32)

    storage = torch.full((groups + 2, m, n), float("nan"), device=device)

    output = storage[:groups]

    output.copy_(torch.randn_like(output) * 0.01 + 0.125)

    initial = output.cpu().double().clone()

    guard = storage[groups:].view(torch.int32).clone()

    for repeat in (1, 2):
        dg._C._mega_moe_training_fp8_wgrad_nt(
            (qa, sa),
            (qb, sb),
            output,
            lengths,
            offsets,
            status,
            config=[192, 256, 128, 48, 64, 4],
            promotion_k=128,
            block_scales=True,
            native_blockwise=True,
            metadata_mode=1,
            scale_m=64,
        )
        got = output.cpu().double()
        want = initial + repeat * product
        rms = float((got - want).norm() / want.norm().clamp_min(1e-300))
        scaled = float(
            ((got - want).abs() / (initial.abs() + repeat * absolute).clamp_min(1e-300)).max()
        )
        assert int(status) == 0 and bool(torch.isfinite(got).all())
        assert rms < 0.003 and scaled < 0.001, (rms, scaled)
        assert torch.equal(got[1], initial[1]) and torch.equal(
            storage[groups:].view(torch.int32), guard
        )
        print(
            json.dumps(
                dict(
                    case="longk_fp8_dw_beta1_fp64",
                    repeat=repeat,
                    lengths=counts,
                    shape=[groups, m, n],
                    relative_rms=rms,
                    error_over_sum_absolute_products=scaled,
                    every_element_compared=True,
                    zero_group_and_tail_unchanged=True,
                    passed=True,
                )
            ),
            flush=True,
        )

    print(json.dumps(dict(case="longk_fp8_dw_finished", passed=True)), flush=True)


def test_cold(options):
    from datetime import timedelta
    from pathlib import Path
    import gc
    import json
    import os
    import torch
    import torch.distributed as dist
    from deep_gemm.training import CompactDeepEPMoEPlan, DeepEPMoEPlan, MegaMoEPlan, to_g8_weight

    rank = int(os.environ["RANK"])
    size = int(os.environ["WORLD_SIZE"])
    assert size == 8
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    root = ROOT
    book = load_test_recipes()
    cache = Path(os.environ["DG_JIT_CACHE_DIR"]) / "cache"
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=600))
    plan = None
    try:
        for compute in (options.compute,):
            recipe = next(
                r
                for r in book["recipes"]
                if (r["ep"], r["model"], r["precision"], r["backend"])
                == (32, "Qwen3.5-397B-A17B", compute, "deepep")
            )
            for key in list(os.environ):
                if key.startswith(("DG_MOE_", "DG_BF16_WGRAD_")) or key in (
                    "DG_JIT_FP8_TRAINING_MMA_FLAGS",
                    "DG_JIT_BF16_TRAINING_MMA_REUSE",
                    "DG_JIT_FP8_DW_MMA_FLAGS",
                    "DG_JIT_TRAINING_FP8_MAX_REGS",
                ):
                    os.environ.pop(key)
            os.environ.update({**book["common_environment"], **recipe["environment"]})
            os.environ.update(
                PCCL_MNNVL_ENABLE="1",
                NCCL_MNNVL_ENABLE="1",
                DEEPEP_NUM_MAX_ICN_PEERS=str(size),
                SAIL_DEEPGEMM_LUT_PATH=str(root / recipe["lut_directory"]),
            )
            for kind in ("windowed", "compact", "megamoe"):
                selected = select_recipe(
                    book,
                    32,
                    "Qwen3.5-397B-A17B",
                    compute,
                    "megamoe" if kind == "megamoe" else "deepep",
                )
                configure(selected, book)
                os.environ["DEEPEP_NUM_MAX_ICN_PEERS"] = str(size)
                torch.manual_seed(661 + rank)
                # The selected gradient producer needs two dHidden N tiles.
                h, i, e, k, tokens = 512, (512 if kind == "megamoe" else 256), 4, 2, 129
                x = (torch.randn(tokens, h, device=device) * 0.1).bfloat16()
                w1 = (torch.randn(e, 2 * i, h, device=device) * 0.02).bfloat16()
                w2 = (torch.randn(e, h, i, device=device) * 0.02).bfloat16()
                ids = torch.randn(tokens, size * e, device=device).topk(k, -1).indices.contiguous()
                scores = torch.ones(tokens, k, device=device) / k
                valid = torch.tensor(tokens, device=device, dtype=torch.int32)
                options = dict(
                    chunk_tokens=128,
                    ring_rows=256,
                    block_m=128,
                    group=dist.group.WORLD,
                    device=device,
                    compute=compute,
                    comm_sms=8,
                    elastic_hybrid=True,
                    check_errors=False,
                    gate_up_layout="split",
                    wgrad_overlap="none",
                    overlap_expert_dy=True,
                    reuse_forward_context=True,
                )
                cls = DeepEPMoEPlan if kind == "windowed" else CompactDeepEPMoEPlan
                if kind == "compact":
                    options["compact_rows"] = 2560
                elif kind == "megamoe":
                    cls = MegaMoEPlan
                    for name in ("comm_sms", "elastic_hybrid"):
                        options.pop(name)
                    options.update(
                        overlap_expert_dy=False,
                        expert_local_grad=True,
                        reuse_packed_dy=True,
                        fuse_dispatch=compute == "bf16",
                    )
                    if compute == "bf16":
                        w1 = to_g8_weight(w1)
                        options.update(
                            gate_up_layout="g8",
                            block_m=256,
                            gemm_configs={
                                r: (256, 64, 64, 64, 3) for r in ("fc1", "fc2", "dhidden", "dx")
                            },
                            dispatch_producer=True,
                            producer_input_aiu=True,
                            producer_local_aiu=True,
                            input_copy_ilp=8,
                            fuse_activation_backward=True,
                            fuse_fc1_activation=True,
                            fuse_dhidden_scores=True,
                            transpose_free_dgrad=True,
                        )
                plan = cls(h, i, e, k, tokens, **options)
                plan.prepare(w1, w2)
                dist.barrier()
                before = {p.name for p in cache.glob("kernel.*") if p.is_dir()}
                output = plan.forward(x, ids, scores, w1, w2, valid)
                gradients = plan.backward(torch.ones_like(x), x, ids, scores, w1, w2, valid)
                torch.cuda.synchronize(device)
                plan.check()
                dist.barrier()
                after = {p.name for p in cache.glob("kernel.*") if p.is_dir()}
                added = sorted(after - before)
                assert not added, (compute, kind, added)
                assert torch.isfinite(output).all()
                assert all(torch.isfinite(value).all() for value in gradients)
                print(
                    json.dumps(
                        dict(
                            case=(
                                "megamoe_cold_prepare_no_late_jit"
                                if kind == "megamoe"
                                else "deepep_cold_prepare_no_late_jit"
                            ),
                            rank=rank,
                            compute=compute,
                            layout=kind,
                            new_modules=added,
                            passed=True,
                        )
                    ),
                    flush=True,
                )
                del output, gradients
                if hasattr(plan, "close"):
                    plan.close()
                plan = None
                gc.collect()
                torch.cuda.empty_cache()
                dist.barrier()
        print(
            json.dumps(dict(case="deepep_cold_prepare_finished", rank=rank, passed=True)),
            flush=True,
        )
    finally:
        if plan is not None and hasattr(plan, "close"):
            plan.close()
        destroy_training_process_group()


def test_compact_small(options):
    from contextlib import contextmanager

    from datetime import timedelta

    import gc, json, os

    import torch

    import torch.distributed as dist

    from deep_gemm.training import DeepEPMoEPlan, CompactDeepEPMoEPlan

    from deep_gemm.training import to_g8_weight

    rank = int(os.environ.get("RANK", "0"))

    size = int(os.environ.get("WORLD_SIZE", "1"))

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))

    torch.cuda.set_device(device)

    torch.set_num_threads(2)

    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=300))

    execution_stream = torch.cuda.Stream(device=device)

    execution_stream.wait_stream(torch.cuda.current_stream(device))

    torch.cuda.set_stream(execution_stream)

    source = ROOT

    book = load_test_recipes()

    def record(case, **fields):
        print(
            json.dumps(dict(case=case, rank=rank, ep_size=size, passed=True, **fields)), flush=True
        )

    def compare(actual, expected, tolerance, detail):
        errors = {}
        for name, x, y in zip(("y", "dx", "dw1", "dw2", "dscores"), actual, expected):
            x = x.float().cpu()
            y = y.float().cpu()
            error = float((x - y).norm() / y.norm().clamp_min(1e-12))
            errors[name] = error
        good = all(torch.isfinite(torch.tensor(x)) and x < tolerance for x in errors.values())
        results = [None] * size
        dist.all_gather_object(results, dict(rank=rank, good=bool(good), errors=errors))
        assert all(x["good"] for x in results), (detail, results)
        return errors

    @contextmanager
    def work_audit(plan):
        original_gemm, original_wgrad = plan._gemm, plan._wgrad
        rows = []

        def gemm(a, weight, output, name, prequantized=False):
            rows.append((name, plan.lengths.clone()))
            return original_gemm(a, weight, output, name, prequantized=prequantized)

        def wgrad(left, right, output, asynchronous=False):
            rows.append(("DW_" + str(tuple(output.shape[1:])), plan.lengths.clone()))
            return original_wgrad(left, right, output, asynchronous=asynchronous)

        plan._gemm, plan._wgrad = gemm, wgrad
        try:
            yield rows
        finally:
            plan._gemm, plan._wgrad = original_gemm, original_wgrad

    plans = []

    graphs = []

    try:
        for compute, layout in [("bf16", "split"), ("bf16", "g8"), ("fp8", "split")]:
            for key in list(os.environ):
                if key.startswith(("DG_MOE_", "DG_BF16_WGRAD_")) or key in (
                    "DG_JIT_FP8_TRAINING_MMA_FLAGS",
                    "DG_JIT_BF16_TRAINING_MMA_REUSE",
                    "DG_JIT_FP8_DW_MMA_FLAGS",
                    "DG_JIT_TRAINING_FP8_MAX_REGS",
                ):
                    os.environ.pop(key)
            recipe = next(
                r
                for r in book["recipes"]
                if r["ep"] == 32
                and r["model"] == "Qwen3.5-397B-A17B"
                and r["precision"] == compute
                and r["backend"] == "deepep"
            )
            os.environ.update({**book["common_environment"], **recipe["environment"]})
            os.environ.update(
                DEEPEP_NUM_MAX_ICN_PEERS=str(size),
                PCCL_MNNVL_ENABLE="1",
                NCCL_MNNVL_ENABLE="1",
                SAIL_DEEPGEMM_LUT_PATH=str(source / recipe["lut_directory"]),
            )
            torch.manual_seed(62300 + rank)
            h, i, e, k, t, c = 512, 256, 4, 2, 193, 128
            x = (torch.randn(t, h, device=device) * 0.1).bfloat16()
            dy = (torch.randn_like(x) * 0.1).bfloat16()
            ids = torch.randn(t, e * size, device=device).topk(k, -1).indices.contiguous()
            original_ids = ids.clone()
            scores = torch.randn(t, k, device=device).softmax(-1)
            w1 = (torch.randn(e, 2 * i, h, device=device) * 0.02).bfloat16()
            w2 = (torch.randn(e, h, i, device=device) * 0.02).bfloat16()
            if layout == "g8":
                w1 = to_g8_weight(w1)
            valid = torch.tensor(t, device=device, dtype=torch.int32)
            options = dict(
                chunk_tokens=c,
                device=device,
                group=dist.group.WORLD,
                compute=compute,
                check_errors=False,
                comm_sms=8,
                elastic_hybrid=True,
                ring_rows=256,
                block_m=128,
                gate_up_layout=layout,
                wgrad_overlap="none",
                overlap_expert_dy=True,
                reuse_forward_context=True,
            )
            old = DeepEPMoEPlan(h, i, e, k, t, **options)
            compact = CompactDeepEPMoEPlan(
                h, i, e, k, t, compact_rows=old.unbounded_row_capacity, **options
            )
            plans = [old, compact]
            assert old.deep_ep_rdma_ranks == compact.deep_ep_rdma_ranks == 1
            for plan in plans:
                plan.prepare(w1, w2)

            def step(plan, accumulators=None):
                plan.reset_errors()
                y = plan.forward(x, ids, scores, w1, w2, valid)
                return (y, *plan.backward(dy, x, ids, scores, w1, w2, valid, accumulators))

            reference = step(old)
            actual = step(compact)
            compare(
                actual, reference, 0.05 if compute == "fp8" else 0.005, "eager " + compute + layout
            )
            del reference, actual
            outputs = []
            graphs = []
            for plan in plans:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=execution_stream):
                    out = step(plan)
                graphs.append(graph)
                outputs.append(out)
            for scenario in ["original", "empty", "partial", "shifted", "skewed", "live_weights"]:
                ids.copy_(original_ids)
                valid.fill_(t)
                if scenario == "empty":
                    valid.zero_()
                elif scenario == "partial":
                    valid.fill_(65)
                elif scenario == "shifted":
                    ids.copy_((original_ids + 3) % (e * size))
                elif scenario == "skewed":
                    ids.copy_(torch.tensor([0, 1], device=device, dtype=torch.int64).expand_as(ids))
                elif scenario == "live_weights":
                    x.mul_(0.875)
                    dy.mul_(1.125)
                    w1.mul_(1.0625)
                    w2.mul_(0.9375)
                for graph in graphs:
                    graph.replay()
                for plan in plans:
                    plan.check()
                errors = compare(
                    outputs[1],
                    outputs[0],
                    0.05 if compute == "fp8" else 0.005,
                    (compute, layout, scenario),
                )
                record(
                    "compact_full_replay",
                    compute=compute,
                    layout=layout,
                    scenario=scenario,
                    relative_rms=errors,
                    every_element_compared=True,
                    fresh_f32_dw=True,
                )
            ids.copy_(torch.tensor([0, 1], device=device, dtype=torch.int64).expand_as(ids))
            valid.fill_(t)
            audits = []
            for plan in plans:
                with work_audit(plan) as rows:
                    values = step(plan)
                summary = {}
                for role, lengths in rows:
                    summary[role] = summary.get(role, 0) + int(lengths.sum())
                audits.append(summary)
                del values
            agreed = audits[0] == audits[1]
            flags = [None] * size
            dist.all_gather_object(flags, dict(agreed=agreed, rows=audits))
            assert all(r["agreed"] for r in flags), flags
            record(
                "compact_full_work_audit",
                compute=compute,
                layout=layout,
                windowed_rows=audits[0],
                compact_rows=audits[1],
                all_gemm_and_dw_rows_equal=True,
                reference_window_rows=compact.reference_window_rows,
            )
            ids.copy_(original_ids)
            values = [step(plan) for plan in plans]
            fresh = [v[2].detach().cpu().clone() for v in values]
            fresh2 = [v[3].detach().cpu().clone() for v in values]
            del values
            external = []
            for plan in plans:
                buffers = (
                    torch.full_like(w1, 0.125, dtype=torch.float32),
                    torch.full_like(w2, 0.25, dtype=torch.float32),
                )
                external.append(step(plan, buffers))
            errors = compare(
                external[1],
                external[0],
                0.05 if compute == "fp8" else 0.005,
                (compute, layout, "external"),
            )
            for index in range(2):
                for value, expected, bias in [
                    (external[index][2], fresh[index], 0.125),
                    (external[index][3], fresh2[index], 0.25),
                ]:
                    error = float(
                        (value.cpu() - (expected + bias)).norm()
                        / (expected + bias).norm().clamp_min(1e-12)
                    )
                    assert error < 0.001, error
            record(
                "compact_full_external_accumulation",
                compute=compute,
                layout=layout,
                relative_rms=errors,
                preserves_initial_f32_values=True,
            )
            del external, fresh, fresh2, outputs
            for graph in graphs:
                graph.reset()
            graphs = []
            torch.cuda.synchronize(device)
            for plan in plans:
                plan.close()
            plans = []
            del old, compact
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
        record("compact_full_finished")
    finally:
        for graph in graphs:
            graph.reset()
        for plan in plans:
            plan.close()
        destroy_training_process_group()


def test_compact_large(options):
    from contextlib import contextmanager

    from datetime import timedelta

    import gc, json, os

    import torch

    import torch.distributed as dist

    import deep_gemm as dg

    from deep_gemm.training import DeepEPMoEPlan, CompactDeepEPMoEPlan

    from deep_gemm.training.deepep import plan_compact_route_capacity

    from deep_gemm.training import to_g8_weight

    a = options

    rank = int(os.environ["RANK"])

    size = int(os.environ["WORLD_SIZE"])

    assert size in (32, 64)

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))

    torch.cuda.set_device(device)

    torch.set_num_threads(2)

    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=900))

    stream = torch.cuda.Stream(device=device)

    stream.wait_stream(torch.cuda.current_stream(device))

    torch.cuda.set_stream(stream)

    source = ROOT

    book = load_test_recipes()

    recipe = next(
        r
        for r in book["recipes"]
        if (r["ep"], r["model"], r["precision"], r["backend"])
        == (size, a.model, a.compute, "deepep")
    )

    os.environ.update(
        {
            **book["common_environment"],
            **recipe["environment"],
            "SAIL_DEEPGEMM_LUT_PATH": str(source / recipe["lut_directory"]),
        }
    )

    argv = recipe["benchmark_arguments"]

    def option(key, default=None, cast=int):
        return cast(argv[argv.index(key) + 1]) if key in argv else default

    h, i, t, c, ring, e, k = (
        option("--hidden"),
        option("--intermediate"),
        option("--tokens"),
        option("--chunk-tokens"),
        option("--ring-rows"),
        512 // size,
        10,
    )

    layout = option("--gate-up-layout", "split", str)

    def record(case, **fields):
        print(
            json.dumps(
                dict(
                    case=case,
                    rank=rank,
                    ep_size=size,
                    model=a.model,
                    compute=a.compute,
                    passed=True,
                    **fields,
                )
            ),
            flush=True,
        )

    def relative_cpu(actual, expected):
        # Bound conversion temporaries while visiting every element, including F32 DW.
        x, y = actual.reshape(-1), expected.reshape(-1)
        error = denom = 0.0
        finite = True
        for start in range(0, x.numel(), 1 << 20):
            xx = x[start : start + (1 << 20)].double()
            yy = y[start : start + (1 << 20)].double()
            finite = finite and bool(torch.isfinite(xx).all()) and bool(torch.isfinite(yy).all())
            error += float((xx - yy).square().sum())
            denom += float(yy.square().sum())
        return (error / max(denom, 1e-24)) ** 0.5 if finite else float("inf")

    @contextmanager
    def audit(plan):
        original_gemm, original_wgrad = plan._gemm, plan._wgrad
        rows = []

        def gemm(x, w, y, name, prequantized=False):
            rows.append((name, plan.lengths.clone()))
            return original_gemm(x, w, y, name, prequantized=prequantized)

        def wgrad(x, y, z, asynchronous=False):
            rows.append(("DW_" + str(tuple(z.shape[1:])), plan.lengths.clone()))
            return original_wgrad(x, y, z, asynchronous=asynchronous)

        plan._gemm, plan._wgrad = gemm, wgrad
        try:
            yield rows
        finally:
            plan._gemm, plan._wgrad = original_gemm, original_wgrad

    torch.manual_seed(98231 + rank)

    x = (torch.randn(t, h, device=device) * 0.1).bfloat16()

    w1 = (torch.randn(e, 2 * i, h, device=device) * 0.02).bfloat16()

    w2 = (torch.randn(e, h, i, device=device) * 0.02).bfloat16()

    if layout == "g8":
        w1 = to_g8_weight(w1)

    ids = torch.randn(t, e * size, device=device).topk(k, -1).indices.contiguous()

    original_ids = ids.clone()

    scores = torch.randn(t, k, device=device).softmax(-1)

    dy = (torch.randn_like(x) * 0.1).bfloat16()

    valid = torch.full((), t, device=device, dtype=torch.int32)

    capacity = plan_compact_route_capacity(
        ids, valid, e, c, group=dist.group.WORLD, alignment=128, headroom=1.25
    )

    original_weights = (w1.clone(), w2.clone())

    reference = []

    reference_work = None

    plan = graph = None

    try:
        for kind in ("windowed", "compact"):
            w1.copy_(original_weights[0])
            w2.copy_(original_weights[1])
            ids.copy_(original_ids)
            valid.fill_(t)
            options = dict(
                chunk_tokens=c,
                device=device,
                group=dist.group.WORLD,
                compute=a.compute,
                check_errors=False,
                comm_sms=option("--deepep-comm-sms", 38),
                elastic_hybrid=True,
                ring_rows=ring,
                block_m=128,
                gate_up_layout=layout,
                wgrad_overlap="none",
                overlap_expert_dy="--overlap-expert-dy" in argv,
                fast_weight_transpose=True,
                fuse_activation_quant=True,
            )
            cls = DeepEPMoEPlan if kind == "windowed" else CompactDeepEPMoEPlan
            if kind == "compact":
                options["compact_rows"] = capacity["capacity"]
            plan = cls(h, i, e, k, t, **options)
            assert plan.deep_ep_rdma_ranks == 1
            plan.prepare(w1, w2)

            def step(accumulators=None):
                plan.reset_errors()
                y = plan.forward(x, ids, scores, w1, w2, valid)
                return (y, *plan.backward(dy, x, ids, scores, w1, w2, valid, accumulators))

            if kind == "compact":
                expected = {
                    role: dg._C._mega_moe_training_nopad_config(
                        device.index, a.compute, min(ring, plan.unbounded_row_capacity), n, kk, e
                    )
                    for role, (n, kk) in dict(
                        fc1=(2 * i, h), fc2=(h, i), dhidden=(i, h), dx=(h, 2 * i)
                    ).items()
                }
                assert plan.resolved_gemm_configs == expected
            with audit(plan) as work:
                values = step()
            plan.check()
            summary = {}
            for role, lengths in work:
                summary[role] = summary.get(role, 0) + int(lengths.sum())
            if kind == "windowed":
                reference_work = summary
            else:
                assert summary == reference_work, (summary, reference_work)
            del values, work
            record(
                "compact_large_work",
                layout=kind,
                gemm_and_dw_rows=summary,
                reference_rows=ring,
                reserved_rows=plan.rows,
                capacity_plan=capacity,
                configs=getattr(plan, "resolved_gemm_configs", None),
                hidden=h,
                intermediate=i,
                tokens=t,
                local_experts=e,
                topk=k,
            )
            assert not plan._forward_context_available and not plan._using_forward_context
            for index in range(len(plan.forward_context_receives)):
                plan.forward_context_receives[index] = None
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = step()
            for index, scenario in enumerate(("original", "live_weights_partial_routes")):
                if index:
                    w1.mul_(1.0625)
                    w2.mul_(0.9375)
                    valid.fill_(t - 17)
                    ids[-17:].fill_(-1)
                graph.replay()
                plan.check()
                actual = [v.detach().cpu().clone() for v in outputs]
                if kind == "windowed":
                    reference.append(actual)
                else:
                    errors = {
                        name: relative_cpu(xx, yy)
                        for name, xx, yy in zip(
                            ("y", "dx", "dw1", "dw2", "dscores"), actual, reference[index]
                        )
                    }
                    tolerance = 0.05 if a.compute == "fp8" else 0.005
                    checks = [None] * size
                    dist.all_gather_object(
                        checks,
                        dict(
                            rank=rank,
                            errors=errors,
                            good=all(v < tolerance for v in errors.values()),
                        ),
                    )
                    assert all(r["good"] for r in checks), checks
                    record(
                        "compact_large_replay",
                        scenario=scenario,
                        relative_rms=errors,
                        every_element_compared=True,
                        fresh_f32_dw=True,
                        live_weight_prepare=True,
                        hidden=h,
                        intermediate=i,
                        tokens=t,
                        local_experts=e,
                        topk=k,
                    )
                del actual
            assert not plan._forward_context_available and not plan._using_forward_context
            before = torch.cuda.memory_allocated(device)
            del outputs
            graph.reset()
            graph = None
            for slot in range(len(plan.forward_context_receives)):
                plan.forward_context_receives[slot] = None
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            record(
                "compact_large_release_graph_before_external",
                allocated_before=before,
                allocated_after=torch.cuda.memory_allocated(device),
            )
            # Preserve caller-provided F32 accumulation values on the same live graph inputs.
            external = (
                torch.full_like(w1, 0.125, dtype=torch.float32),
                torch.full_like(w2, 0.25, dtype=torch.float32),
            )
            expected = [value.clone() for value in external]
            targets = {value.data_ptr(): index for index, value in enumerate(external)}
            original_wgrad = plan._wgrad

            def check_same_operands(
                left,
                right,
                output,
                asynchronous=False,
                *,
                targets=targets,
                original_wgrad=original_wgrad,
                expected=expected,
            ):
                assert not asynchronous
                index = targets[output.data_ptr()]
                contribution = torch.zeros_like(output)
                # Reuse this exact step/window's BF16 operands and route lengths.
                # A separate dispatch can reorder FP8 K128 quantization blocks.
                original_wgrad(left, right, contribution, asynchronous=False)
                expected[index].add_(contribution)
                return original_wgrad(left, right, output, asynchronous=False)

            plan._wgrad = check_same_operands
            try:
                extra = step(external)
            finally:
                plan._wgrad = original_wgrad
            plan.check()
            errors = [
                relative_cpu(value.detach().cpu(), reference.detach().cpu())
                for value, reference in zip(extra[2:4], expected)
            ]
            assert max(errors) < 0.001, errors
            assert all(
                value.data_ptr() == buffer.data_ptr() for value, buffer in zip(extra[2:4], external)
            )
            record(
                "compact_large_external",
                layout=kind,
                relative_rms=errors,
                all_elements_compared=True,
                preserves_initial_f32_values=True,
                reference="initial_F32_plus_zero_beta_DW_contribution_using_same_step_window_operands",
                validation_only_extra_dw_calls=True,
                relative_rms_limit=0.001,
            )
            # Drop oracle bound-method aliases so the previous plan/workspace is
            # collected before constructing the next sequential layout.
            del extra, external, expected, targets, original_wgrad, check_same_operands
            torch.cuda.synchronize(device)
            plan.close()
            plan = None
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
        record("compact_large_finished")
    finally:
        if graph is not None:
            graph.reset()
        if plan is not None:
            plan.close()
        destroy_training_process_group()


def test_overflow(options):
    from datetime import timedelta

    import gc, json, os

    import torch

    import torch.distributed as dist

    from deep_gemm.training import CompactDeepEPMoEPlan

    rank = int(os.environ["RANK"])

    size = int(os.environ["WORLD_SIZE"])

    assert size == 8

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))

    torch.cuda.set_device(device)

    torch.set_num_threads(2)

    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=600))

    stream = torch.cuda.Stream(device=device)

    stream.wait_stream(torch.cuda.current_stream(device))

    torch.cuda.set_stream(stream)

    source = ROOT

    book = load_test_recipes()

    plan = graph = None

    def record(case, **fields):
        print(
            json.dumps(dict(case=case, rank=rank, ep_size=size, passed=True, **fields)), flush=True
        )

    try:
        for compute in ("bf16", "fp8"):
            for key in list(os.environ):
                if key.startswith(("DG_MOE_", "DG_BF16_WGRAD_")) or key in (
                    "DG_JIT_FP8_TRAINING_MMA_FLAGS",
                    "DG_JIT_BF16_TRAINING_MMA_REUSE",
                    "DG_JIT_FP8_DW_MMA_FLAGS",
                    "DG_JIT_TRAINING_FP8_MAX_REGS",
                ):
                    os.environ.pop(key)
            recipe = next(
                r
                for r in book["recipes"]
                if (r["ep"], r["model"], r["precision"], r["backend"])
                == (32, "Qwen3.5-397B-A17B", compute, "deepep")
            )
            os.environ.update({**book["common_environment"], **recipe["environment"]})
            os.environ.update(
                DEEPEP_NUM_MAX_ICN_PEERS=str(size),
                PCCL_MNNVL_ENABLE="1",
                NCCL_MNNVL_ENABLE="1",
                SAIL_DEEPGEMM_LUT_PATH=str(source / recipe["lut_directory"]),
            )
            torch.manual_seed(639 + rank)
            h, i, e, k, t = 512, 256, 4, 2, 129
            x = (torch.randn(t, h, device=device) * 0.1).bfloat16()
            dy = torch.randn_like(x)
            w1 = (torch.randn(e, 2 * i, h, device=device) * 0.02).bfloat16()
            w2 = (torch.randn(e, h, i, device=device) * 0.02).bfloat16()
            ids = torch.full((t, k), -1, device=device, dtype=torch.int64)
            ids[0, 0] = rank * e
            scores = torch.ones(t, k, device=device) / k
            valid = torch.tensor(1, device=device, dtype=torch.int32)
            plan = CompactDeepEPMoEPlan(
                h,
                i,
                e,
                k,
                t,
                compact_rows=128,
                chunk_tokens=128,
                ring_rows=256,
                block_m=128,
                group=dist.group.WORLD,
                device=device,
                compute=compute,
                comm_sms=8,
                elastic_hybrid=True,
                check_errors=False,
                gate_up_layout="split",
                wgrad_overlap="none",
                overlap_expert_dy=True,
                reuse_forward_context=True,
            )
            assert plan.deep_ep_rdma_ranks == 1
            plan.prepare(w1, w2)

            def step():
                plan.reset_errors()
                y = plan.forward(x, ids, scores, w1, w2, valid)
                return (y, *plan.backward(dy, x, ids, scores, w1, w2, valid))

            outputs = step()
            plan.check()
            del outputs
            for index in range(len(plan.forward_context_receives)):
                plan.forward_context_receives[index] = None
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = step()
            for scenario in ("fit", "overflow", "recovered_fit"):
                if scenario == "overflow":
                    # Overfill every rank's own two experts; a global hot expert
                    # would overflow only its owner, leaving other statuses zero.
                    ids.copy_(
                        torch.tensor(
                            [rank * e, rank * e + 1], device=device, dtype=torch.int64
                        ).expand_as(ids)
                    )
                    valid.fill_(t)
                else:
                    ids.fill_(-1)
                    ids[0, 0] = rank * e
                    valid.fill_(1)
                graph.replay()
                torch.cuda.synchronize(device)
                status = int(plan.status)
                if scenario == "overflow":
                    assert status & 2, status
                    try:
                        plan.check()
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError("Overflow was accepted as a valid training step")
                else:
                    plan.check()
                    assert status == 0, status
                record(
                    "compact_graph_overflow_recovery",
                    compute=compute,
                    scenario=scenario,
                    status=status,
                    reservation_rows=128,
                    same_graph_reused=True,
                    overflowing_outputs_rejected=scenario == "overflow",
                )
            del outputs
            graph.reset()
            graph = None
            plan.close()
            plan = None
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
        record("compact_graph_overflow_recovery_finished")
    finally:
        if graph is not None:
            graph.reset()
        if plan is not None:
            plan.close()
        destroy_training_process_group()


def test_transport_bf16(options):
    import gc, hashlib, json, os, tempfile

    from pathlib import Path

    import torch

    import torch.distributed as dist

    import deep_gemm as dg

    from deep_gemm.training import MegaMoEPlan

    from deep_gemm.training import to_g8_weight

    os.environ.update(
        DG_MOE_REUSE_FORWARD_CONTEXT="1",
        DG_MOE_BF16_GRAD_PRODUCER="1",
        DG_MOE_SKIP_MASKED_ROUTE_ZERO="1",
        DG_MOE_WGRAD_FIRST_NONEMPTY="1",
        DG_MOE_WGRAD_FIRST_OVERWRITE="0",
        DG_MOE_WGRAD_AGGREGATE_WINDOWS="1",
        DG_JIT_BF16_TRAINING_MMA_REUSE="0",
    )

    rank = int(os.environ.get("RANK", "0"))

    size = int(os.environ.get("WORLD_SIZE", "1"))

    local = int(os.environ.get("LOCAL_RANK", "0"))

    torch.cuda.set_device(local)

    torch.set_num_threads(2)

    torch.manual_seed(55700 + rank)

    if "RANK" in os.environ:
        dist.init_process_group("nccl", init_method="env://")
    else:
        fd, path = tempfile.mkstemp(prefix="training_bf16_transport-")
        os.close(fd)
        os.unlink(path)
        dist.init_process_group("nccl", init_method="file://" + path, rank=0, world_size=1)

    print(
        json.dumps(
            dict(
                case="training_bf16_transport_library",
                rank=rank,
                extension_sha256=hashlib.sha256(Path(dg._C.__file__).read_bytes()).hexdigest(),
                passed=True,
            )
        ),
        flush=True,
    )

    book = load_test_recipes()

    if options.model == "toy":
        template = select_recipe(book, 32, "Qwen3.5-397B-A17B", "bf16", backend="megamoe")
        RECIPES = [
            dict(env=template["environment"], h=512, i=512, t=512, bm=128, ring=32768),
            dict(env=template["environment"], h=1024, i=1024, t=513, bm=256, ring=32768),
        ]
    else:
        assert size in (32, 64), "Actual model shape requires EP32 or EP64"
        recipe = select_recipe(book, size, options.model, "bf16", backend="megamoe")
        bargs = recipe["benchmark_arguments"]

        def argument(name):
            return int(bargs[bargs.index(name) + 1])

        RECIPES = [
            dict(
                env=recipe["environment"],
                h=argument("--hidden"),
                i=argument("--intermediate"),
                t=argument("--tokens"),
                bm=argument("--block-m"),
                ring=argument("--ring-rows"),
            )
        ]

    selected_gemm = (256, 64, 64, 64, 3)

    if options.model != "toy":
        position = bargs.index("--gemm-config") + 1
        selected_gemm = tuple(int(value) for value in bargs[position : position + 5])

    for recipe in RECIPES:
        os.environ.update(recipe["env"])
        h, i, t, bm, ring = (recipe[k] for k in ("h", "i", "t", "bm", "ring"))
        # Component-major NCP batch stores require a complete 256-column warp tile.
        os.environ["DG_MOE_BF16_RETURN_NCP_MODE"] = "2"
        try:
            MegaMoEPlan(
                384,
                512,
                4,
                2,
                512,
                chunk_tokens=512,
                compute="bf16",
                group=dist.group.WORLD,
                device=torch.device("cuda", local),
                ring_rows=32768,
                block_m=128,
                gate_up_layout="g8",
                gemm_configs={r: selected_gemm for r in ("fc1", "fc2", "dhidden", "dx")},
                expert_local_grad=True,
                reuse_packed_dy=True,
                fuse_activation_backward=True,
                fuse_fc1_activation=True,
                fuse_dhidden_scores=True,
                transpose_free_dgrad=True,
                dispatch_producer=True,
                producer_input_aiu=True,
                producer_local_aiu=True,
                input_copy_ilp=8,
            )
        except ValueError as error:
            assert "H%256=0" in str(error), str(error)
        else:
            raise AssertionError("unaligned NCP return shape was accepted")
        print(
            json.dumps(
                dict(
                    case="training_bf16_transport_reject_unaligned_ncp",
                    rank=rank,
                    hidden=384,
                    passed=True,
                )
            ),
            flush=True,
        )

        e, k = (4, 2) if options.model == "toy" else (512 // size, 10)
        x = (torch.randn(t, h, device="cuda") * 0.1).bfloat16()
        dy = (torch.randn(t, h, device="cuda") * 0.1).bfloat16()
        w1 = to_g8_weight((torch.randn(e, 2 * i, h, device="cuda") * 0.02).bfloat16())
        w2 = (torch.randn(e, h, i, device="cuda") * 0.02).bfloat16()
        base_ids = torch.randn(t, e * size, device="cuda").topk(k, -1).indices.contiguous()
        ids = base_ids.clone()
        scores = torch.softmax(torch.randn(t, k, device="cuda"), -1)
        valid = torch.tensor([t], device="cuda", dtype=torch.int32)
        plans = []
        os.environ["DG_MOE_BF16_RETURN_NCP_MODE"] = "0"
        p = MegaMoEPlan(
            h,
            i,
            e,
            k,
            t,
            chunk_tokens=t,
            compute="bf16",
            group=dist.group.WORLD,
            device=torch.device("cuda", local),
            ring_rows=ring,
            block_m=bm,
            gate_up_layout="g8",
            gemm_configs={r: selected_gemm for r in ("fc1", "fc2", "dhidden", "dx")},
            expert_local_grad=True,
            reuse_packed_dy=True,
            fuse_activation_backward=True,
            fuse_fc1_activation=True,
            fuse_dhidden_scores=True,
            transpose_free_dgrad=True,
            dispatch_producer=True,
            producer_input_aiu=True,
            producer_local_aiu=True,
            input_copy_ilp=8,
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        # Exercise the private C++ guard directly without launching an invalid kernel.
        # The public plan already enforces H%256; the private API must enforce it too.
        p.bf16_return_ncp_mode = 2
        bad_weight = torch.empty((e, 384, i), device="cuda", dtype=torch.bfloat16)
        bad_output = torch.empty((p.rows, 384), device="cuda", dtype=torch.bfloat16)
        try:
            p._peer_gemm(
                p.hidden,
                bad_weight,
                bad_output,
                p.identity_rows,
                write=True,
                destination="local_values",
                role="fc2",
                compile_only=True,
            )
        except RuntimeError as error:
            assert "n%256==0" in str(error), str(error)
        else:
            raise AssertionError("private native API accepted an incomplete NCP output tile")
        p.bf16_return_ncp_mode = 0
        del bad_weight, bad_output
        print(
            json.dumps(
                dict(
                    case="training_bf16_transport_reject_unaligned_native_output",
                    rank=rank,
                    output_columns=384,
                    passed=True,
                )
            ),
            flush=True,
        )
        for plan_index, mode in enumerate((0, 0, 2)):
            p.bf16_return_ncp_mode = mode
            p._prepared = False
            with torch.cuda.stream(stream):
                p.prepare(w1, w2)
            assert p.bf16_return_ncp_mode == mode
            with torch.cuda.stream(stream):
                for _ in range(2):
                    p.forward(x, ids, scores, w1, w2, valid)
                    p.backward(dy, x, ids, scores, w1, w2, valid)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    y = p.forward(x, ids, scores, w1, w2, valid)
                    route_values = p.local_values.clone()
                    outputs = (y, *p.backward(dy, x, ids, scores, w1, w2, valid))
            torch.cuda.current_stream().wait_stream(stream)
            plans.append((mode, p, graph, outputs, stream, route_values))
        for change in range(3):
            x.add_(0.003)
            dy.mul_(0.97)
            w1.mul_(1.001)
            w2.mul_(0.999)
            scores.mul_(0.91)
            if change == 1:
                ids.copy_(base_ids)
                ids[-13:] = -1
                valid.fill_(t - 13)
            if change == 2:
                ids.fill_(-1)
                valid.zero_()
            for mode, p, graph, outputs, stream, route_values in plans:
                graph.replay()
            torch.cuda.synchronize()
            control = plans[0][3]
            for mode, p, graph, outputs, stream, route_values in plans:
                p.check()
                exact = [torch.equal(a, b) for a, b in zip(outputs, control)]
                finite = [bool(torch.isfinite(v).all()) for v in outputs]
                assert all(finite), (rank, h, mode, change, finite)
                errors = [
                    float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-20))
                    for a, b in zip(outputs, control)
                ]
                maxima = [
                    float((a.float() - b.float()).abs().max()) for a, b in zip(outputs, control)
                ]
                active = (ids.reshape(-1) >= 0) & (
                    torch.arange(t, device="cuda")[:, None].expand(t, k).reshape(-1) < valid.item()
                )
                route_exact = torch.equal(
                    route_values.reshape(t * k, h)[active], plans[0][5].reshape(t * k, h)[active]
                )
                assert exact[0] and exact[1] and route_exact and max(errors[2:]) < 2e-6, (
                    rank,
                    h,
                    mode,
                    change,
                    exact,
                    route_exact,
                    errors,
                )
                if change == 2:
                    assert all(torch.count_nonzero(v) == 0 for v in outputs)
                assert outputs[2].dtype == outputs[3].dtype == torch.float32
                print(
                    json.dumps(
                        dict(
                            case="training_bf16_transport_large_replay",
                            rank=rank,
                            ep_size=size,
                            hidden=h,
                            intermediate=i,
                            tokens=t,
                            local_experts=e,
                            topk=k,
                            block_m=bm,
                            mode=mode,
                            iteration=change,
                            acceptance_gate=True,
                            f32_relative_rms_limit=2e-6,
                            active_route_y_bitwise=route_exact,
                            exact_by_output=exact,
                            relative_errors=errors,
                            max_absolute_errors=maxima,
                            fresh_f32_dw=True,
                            passed=True,
                        )
                    ),
                    flush=True,
                )
        ids.copy_(base_ids)
        valid.fill_(t)
        for mode, p, graph, outputs, stream, route_values in plans:
            p.bf16_return_ncp_mode = mode
            graph.replay()
            torch.cuda.synchronize()
            fresh = [v.clone() for v in outputs[2:4]]
            initial = [torch.randn_like(w, dtype=torch.float32) * 0.01 for w in (w1, w2)]
            acc = [v.clone() for v in initial]
            with torch.cuda.stream(stream):
                stream.wait_stream(torch.cuda.default_stream())
                p.forward(x, ids, scores, w1, w2, valid)
                result = p.backward(dy, x, ids, scores, w1, w2, valid, accumulators=tuple(acc))
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            p.check()
            errors = [
                float((a - b - c).norm() / (b + c).norm().clamp_min(1e-20))
                for a, b, c in zip(result[1:3], initial, fresh)
            ]
            assert max(errors) < 2e-6 and all(
                a.data_ptr() == b.data_ptr() for a, b in zip(result[1:3], acc)
            )
            print(
                json.dumps(
                    dict(
                        case="training_bf16_transport_large_external",
                        rank=rank,
                        mode=mode,
                        block_m=bm,
                        relative_errors=errors,
                        passed=True,
                    )
                ),
                flush=True,
            )
        del plans, p, graph, outputs, control, x, dy, w1, w2, ids, base_ids, scores, valid
        gc.collect()
        torch.cuda.empty_cache()

    destroy_training_process_group()

    print(
        json.dumps(dict(case="training_bf16_transport_large_finished", rank=rank, passed=True)),
        flush=True,
    )


def test_transport_fp8(options):
    import gc, hashlib, json, os

    from pathlib import Path

    import torch

    import torch.distributed as dist

    from deep_gemm.training import MegaMoEPlan

    import deep_gemm as dg

    args = options

    config = load_test_recipes()

    world = int(os.environ["WORLD_SIZE"])

    selected = next(
        r
        for r in config["recipes"]
        if (r["ep"], r["model"], r["precision"], r["backend"])
        == (
            world if args.model != "toy" else 32,
            args.model if args.model != "toy" else "Qwen3.5-397B-A17B",
            "fp8",
            "megamoe",
        )
    )

    argv = selected["benchmark_arguments"]

    RECIPES = [
        dict(
            env=selected["environment"],
            h=int(argv[argv.index("--hidden") + 1]),
            i=int(argv[argv.index("--intermediate") + 1]),
            tokens=int(argv[argv.index("--tokens") + 1]),
            ring=int(argv[argv.index("--ring-rows") + 1]),
        )
    ]

    rank = int(os.environ["RANK"])

    size = int(os.environ["WORLD_SIZE"])

    local_rank = int(os.environ["LOCAL_RANK"])

    assert size in (32, 64) or args.model == "toy"

    torch.cuda.set_device(local_rank)

    torch.set_num_threads(2)

    torch.manual_seed(13539 + rank)

    dist.init_process_group("nccl", init_method="env://")

    device = torch.device("cuda", local_rank)

    print(
        json.dumps(
            dict(
                case="fp8_training_large_library",
                rank=rank,
                ep_size=size,
                extension_sha256=hashlib.sha256(Path(dg._C.__file__).read_bytes()).hexdigest(),
                passed=True,
            )
        ),
        flush=True,
    )

    for recipe in RECIPES:
        os.environ.update(recipe["env"])
        h, i, t, ring = (recipe[k] for k in ("h", "i", "tokens", "ring"))
        if args.model == "toy":
            h, i, t, ring = 512, 256, 193, 32768
        e, k = (512 // size, 10) if args.model != "toy" else (4, 2)
        x = (torch.randn(t, h, device=device) * 0.1).bfloat16()
        dy = (torch.randn_like(x.float()) * 0.1).bfloat16()
        w1 = (torch.randn(e, 2 * i, h, device=device) * 0.02).bfloat16()
        w2 = (torch.randn(e, h, i, device=device) * 0.02).bfloat16()
        ids = torch.randn(t, e * size, device=device).topk(k, -1).indices.contiguous()
        base_ids = ids.clone()
        scores = torch.softmax(torch.randn(t, k, device=device), -1)
        valid = torch.full((), t, device=device, dtype=torch.int32)
        plan = MegaMoEPlan(
            h,
            i,
            e,
            k,
            t,
            chunk_tokens=t,
            compute="fp8",
            group=dist.group.WORLD,
            device=device,
            ring_rows=ring,
            fuse_dispatch=False,
            fuse_return=True,
            reuse_packed_dy=True,
            expert_local_grad=True,
            block_m=128,
            gate_up_layout="split",
            wgrad_overlap="none",
            check_errors=True,
        )
        assert plan.fp8_return_store_bytes == 16 and plan.fp8_return_remote_mode == 2
        selected_fusion = dict(
            forward_tile_source=getattr(plan, "fp8_forward_tile_source", 0),
            forward_ilp=getattr(plan, "fp8_inline_bulk_ilp", 4),
            backward_ilp=getattr(plan, "fp8_raw_bulk_ilp", 4),
            copy_threads=plan.fp8_raw_gradient_tile_threads,
            source_grid=plan.fp8_raw_gradient_source_grid,
            gemm_ctas=plan.fp8_raw_gradient_gemm_ctas,
        )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream(device))
        graphs = []
        print(
            json.dumps(
                dict(
                    case="fp8_training_large_prepare",
                    rank=rank,
                    hidden=h,
                    intermediate=i,
                    tokens=t,
                    memory_bytes=torch.cuda.memory_allocated(),
                    passed=True,
                )
            ),
            flush=True,
        )
        with torch.cuda.stream(stream):
            for mode in ((8, 0, 0), (7, 1, 0), (7, 1, 1)):
                plan.fp8_raw_gradient_source, qp, plan.fp8_raw_gradient_source_bulk = mode
                selected_path = bool(mode[2])
                plan.fp8_forward_tile_source = (
                    selected_fusion["forward_tile_source"] if selected_path else 0
                )
                plan.fp8_inline_bulk_ilp = selected_fusion["forward_ilp"] if selected_path else 4
                plan.fp8_raw_bulk_ilp = selected_fusion["backward_ilp"] if selected_path else 4
                plan.fp8_raw_gradient_tile_threads = (
                    selected_fusion["copy_threads"] if selected_path else 128
                )
                plan.fp8_raw_gradient_source_grid = (
                    selected_fusion["source_grid"] if selected_path else 0
                )
                plan.fp8_raw_gradient_gemm_ctas = (
                    selected_fusion["gemm_ctas"] if selected_path else 0
                )

                plan.fp8_raw_gradient_quant_pipeline = bool(qp)
                plan.fp8_raw_gradient_source_prefetch = False
                plan.fp8_inline_source_bulk = 1
                plan._prepared = False
                plan.prepare(w1, w2)
                for _ in range(2):
                    y = plan.forward(x, ids, scores, w1, w2, valid)
                    b = plan.backward(dy, x, ids, scores, w1, w2, valid)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    y = plan.forward(x, ids, scores, w1, w2, valid)
                    outputs = (y, *plan.backward(dy, x, ids, scores, w1, w2, valid))
                graphs.append((mode, graph, outputs))
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize()
        for change in range(3):
            x.add_(0.001)
            dy.mul_(0.97)
            w1.mul_(1.0625)
            w2.mul_(0.9375)
            scores.mul_(0.93)
            if change == 1:
                ids.copy_(base_ids)
                ids[-113:] = -1
                valid.fill_(t - 113)
            if change == 2:
                ids.fill_(-1)
                valid.zero_()
            for mode, graph, outputs in graphs:
                graph.replay()
            torch.cuda.synchronize()
            control = graphs[0][2]
            for mode, graph, outputs in graphs:
                plan.check()
                exact = [torch.equal(a, b) for a, b in zip(outputs, control)]
                finite = [bool(torch.isfinite(a).all()) for a in outputs]
                assert all(exact) and all(finite), (rank, h, mode, change, exact, finite)
                if change == 2:
                    assert all((torch.count_nonzero(a) == 0 for a in outputs))
                assert outputs[2].dtype == outputs[3].dtype == torch.float32
                print(
                    json.dumps(
                        dict(
                            case="fp8_training_large_replay",
                            rank=rank,
                            ep_size=size,
                            hidden=h,
                            intermediate=i,
                            tokens=t,
                            local_experts=e,
                            topk=k,
                            iteration=change,
                            return_remote_mode=2,
                            input_bulk_modes=(1, mode[2]),
                            raw_gradient_recipe=mode,
                            all_five_outputs_bitwise=True,
                            fresh_f32_dw=True,
                            mutable_inputs=["x", "dy", "w1", "w2", "scores", "ids", "valid"],
                            passed=True,
                        )
                    ),
                    flush=True,
                )
        del graphs, graph, outputs, control, plan, x, dy, w1, w2, ids, base_ids, scores, valid, y, b
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    destroy_training_process_group()

    print(
        json.dumps(dict(case="fp8_training_large_finished", rank=rank, ep_size=size, passed=True)),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "recipes",
            "capacity",
            "prepare",
            "dw",
            "cold",
            "compact",
            "transport",
            "overflow",
        ),
    )
    parser.add_argument(
        "--model", choices=("toy", "Qwen3.5-397B-A17B", "Qwen3.8-2.4T-A95B"), default="toy"
    )
    parser.add_argument("--precision", dest="compute", choices=("bf16", "fp8"), default="bf16")
    options = parser.parse_args()
    if options.mode not in ("recipes", "capacity"):
        book = load_recipes()
        ep = int(os.environ.get("WORLD_SIZE", "1"))
        recipe = select_recipe(
            book,
            ep if ep in (32, 64) else 32,
            options.model if options.model != "toy" else "Qwen3.5-397B-A17B",
            options.compute,
            "deepep" if options.mode in ("cold", "compact", "overflow") else "megamoe",
        )
        configure(recipe, book)
    if options.mode == "recipes":
        test_recipes()
    elif options.mode == "capacity":
        test_capacity(options)
    elif options.mode == "prepare":
        test_prepare(options)
    elif options.mode == "dw":
        test_bf16_dw(options)
        test_dw_alias(options)
        test_dw_longk(options)
    elif options.mode == "compact":
        (test_compact_small if options.model == "toy" else test_compact_large)(options)
    elif options.mode == "transport":
        (test_transport_bf16 if options.compute == "bf16" else test_transport_fp8)(options)
    elif options.mode == "cold":
        test_cold(options)
    else:
        test_overflow(options)
    print(
        json.dumps(dict(case="mega_moe_training_tests_finished", mode=options.mode, passed=True)),
        flush=True,
    )


if __name__ == "__main__":
    main()
