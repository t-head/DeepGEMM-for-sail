"""Reproduce the selected routed MoE forward/backward recipes on PPU0015."""

import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def load_recipes():
    book = json.loads((ROOT / "docs/ppu_mega_moe_training_recipes.json").read_text())
    expanded = []
    defaults = book["argument_defaults"]
    for raw in book["recipes"]:
        recipe = {
            **book["recipe_defaults"],
            **book["variant_defaults"][raw["backend"] + "/" + raw["precision"]],
            **raw,
        }
        arguments = list(defaults["common"])
        for scope, key in (
            ("models", "model"),
            ("eps", "ep"),
            ("backends", "backend"),
            ("precisions", "precision"),
        ):
            arguments.extend(defaults[scope][str(recipe[key])])
        recipe["benchmark_arguments"] = arguments + raw["benchmark_arguments"]
        recipe["environment"] = {
            **book["common_environment"],
            **book["variant_defaults"][raw["backend"] + "/" + raw["precision"]]["environment"],
            **raw["environment"],
        }
        recipe["lut_directory"] = "build/mega_moe_training_luts/" + recipe["lut_profile"]
        recipe["lut_sha256"] = book["lut_checksums"][recipe["lut_profile"]]
        expanded.append(recipe)
    book["recipes"] = expanded
    return book


def select_recipe(book, ep, model, precision, backend="megamoe", layout="windowed"):
    return next(
        r
        for r in book["recipes"]
        if (r["ep"], r["model"], r["precision"], r["backend"], r.get("layout", "windowed"))
        == (ep, model, precision, backend, layout)
    )


def materialize_lut(recipe, book):
    directory = ROOT / recipe["lut_directory"]
    directory.mkdir(parents=True, exist_ok=True)
    for name, contents in book["lut_profiles"][recipe["lut_profile"]].items():
        payload = contents.encode()
        if hashlib.sha256(payload).hexdigest() != recipe["lut_sha256"][name]:
            raise ValueError("DW LUT checksum mismatch: " + name)
        with tempfile.NamedTemporaryFile(
            dir=directory, prefix="." + name + ".", delete=False
        ) as stream:
            stream.write(payload)
            temporary = Path(stream.name)
        try:
            temporary.replace(directory / name)
        finally:
            temporary.unlink(missing_ok=True)


def configure(recipe, book=None, *, write_lut=True):
    """Restore the complete measured environment; never inherit stale tuning flags."""
    book = load_recipes() if book is None else book
    lut = ROOT / recipe["lut_directory"]
    for name, contents in book["lut_profiles"][recipe["lut_profile"]].items():
        if hashlib.sha256(contents.encode()).hexdigest() != recipe["lut_sha256"][name]:
            raise ValueError("DW LUT checksum mismatch: " + name)
    if write_lut:
        materialize_lut(recipe, book)
    environment = {**recipe["environment"], "SAIL_DEEPGEMM_LUT_PATH": str(lut)}
    if write_lut:
        for key in list(os.environ):
            if key.startswith(("DG_MOE_", "DG_BF16_WGRAD_")) or key in (
                "DG_JIT_FP8_TRAINING_MMA_FLAGS",
                "DG_JIT_BF16_TRAINING_MMA_REUSE",
                "DG_JIT_FP8_DW_MMA_FLAGS",
                "DG_JIT_TRAINING_FP8_MAX_REGS",
            ):
                os.environ.pop(key)
        os.environ.update(environment)
    return environment


def recipe_arguments(recipe):
    """Parse only knobs used by the 24 selected recipes."""
    parser = argparse.ArgumentParser(add_help=False)
    for name in (
        "tokens",
        "hidden",
        "intermediate",
        "local-experts",
        "topk",
        "chunk-tokens",
        "ring-rows",
        "block-m",
        "input-copy-ilp",
        "dgrad-compute-cus",
        "deepep-comm-sms",
        "warmup",
        "iterations",
        "repeats",
    ):
        parser.add_argument("--" + name, type=int)
    for name in ("backend", "compute", "gate-up-layout", "wgrad-overlap", "topology", "scope"):
        parser.add_argument("--" + name)
    parser.add_argument("--deepep-layout", default="windowed")
    parser.add_argument("--gemm-config", type=int, nargs=5)
    for name in (
        "deepep-scaleup",
        "fresh-weight-gradients",
        "expert-local-grad",
        "dispatch-producer",
        "fuse-activation-backward",
        "fuse-dhidden-scores",
        "fuse-fc1-activation",
        "no-fuse-dispatch",
        "overlap-expert-dy",
        "producer-input-aiu",
        "producer-local-aiu",
        "reuse-packed-dy",
        "transpose-free-dgrad",
    ):
        parser.add_argument("--" + name, action="store_true")
    return parser.parse_args(recipe["benchmark_arguments"])


def make_plan(recipe, args, ids, valid, group, device):
    import deep_gemm as dg
    from deep_gemm.training import CompactDeepEPMoEPlan, DeepEPMoEPlan, MegaMoEPlan
    from deep_gemm.training.deepep import plan_compact_route_capacity

    options = dict(
        chunk_tokens=args.chunk_tokens,
        group=group,
        device=device,
        compute=args.compute,
        check_errors=False,
        fast_weight_transpose=True,
        block_m=args.block_m,
        gate_up_layout=args.gate_up_layout,
        ring_rows=args.ring_rows,
        wgrad_overlap=args.wgrad_overlap,
        overlap_expert_dy=args.overlap_expert_dy,
    )
    if args.gemm_config:
        options["gemm_configs"] = {
            role: args.gemm_config for role in ("fc1", "fc2", "dhidden", "dx")
        }
    capacity = None
    if recipe["backend"] == "megamoe":
        cls = MegaMoEPlan
        options.update(fuse_dispatch=not args.no_fuse_dispatch)
        for name in (
            "dispatch_producer",
            "expert_local_grad",
            "fuse_activation_backward",
            "fuse_dhidden_scores",
            "fuse_fc1_activation",
            "input_copy_ilp",
            "producer_input_aiu",
            "producer_local_aiu",
            "reuse_packed_dy",
            "transpose_free_dgrad",
        ):
            value = getattr(args, name)
            if value is not None:
                options[name] = value
    else:
        cls = DeepEPMoEPlan
        options.update(
            comm_sms=args.deepep_comm_sms, fuse_activation_quant=True, elastic_hybrid=True
        )
        if args.deepep_layout == "compact":
            cls = CompactDeepEPMoEPlan
            capacity = plan_compact_route_capacity(
                ids,
                valid,
                args.local_experts,
                args.chunk_tokens,
                group=group,
                alignment=math.lcm(args.block_m, dg.get_m_alignment_for_contiguous_layout()),
                headroom=1.25,
            )
            options["compact_rows"] = capacity["capacity"]
    plan = cls(
        args.hidden, args.intermediate, args.local_experts, args.topk, args.tokens, **options
    )
    expected = recipe.get("required_deepep_scaleout_groups")
    if expected is not None and plan.deep_ep_rdma_ranks != expected:
        plan.close()
        raise RuntimeError(
            f"Expected {expected} DeepEP scale-out group, observed "
            f"{plan.deep_ep_rdma_ranks}; check physical supernode and MNNVL"
        )
    return plan, capacity


def destroy_training_process_group():
    """Keep the rendezvous store alive until every rank finishes teardown."""
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        return
    rank, size = dist.get_rank(), dist.get_world_size()
    store = dist.distributed_c10d._get_default_store()
    torch.cuda.synchronize()
    ready = [f"deep_gemm_training_device_ready_{peer}" for peer in range(size)]
    store.set(ready[rank], "done")
    store.wait(ready, timedelta(seconds=300))
    dist.destroy_process_group()
    keys = [f"deep_gemm_training_shutdown_{peer}" for peer in range(size)]
    store.set(keys[rank], "done")
    store.wait(keys, timedelta(seconds=300))
    acknowledgements = [f"deep_gemm_training_shutdown_ack_{peer}" for peer in range(1, size)]
    if rank:
        store.set(acknowledgements[rank - 1], "done")
    elif acknowledgements:
        store.wait(acknowledgements, timedelta(seconds=300))


def benchmark(recipe, args, profile=False):
    import torch
    import torch.distributed as dist
    import deep_gemm as dg
    from deep_gemm.training import to_g8_weight

    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=600))
    group = dist.group.WORLD
    torch.manual_seed(98231 + rank)
    h, i, e, k, t = args.hidden, args.intermediate, args.local_experts, args.topk, args.tokens
    x = (torch.randn(t, h, device=device) * 0.1).bfloat16()
    w1 = (torch.randn(e, 2 * i, h, device=device) * 0.02).bfloat16()
    w2 = (torch.randn(e, h, i, device=device) * 0.02).bfloat16()
    if args.gate_up_layout == "g8":
        w1 = to_g8_weight(w1)
    ids = torch.randn(t, e * size, device=device).topk(k, -1).indices.contiguous()
    scores = torch.randn(t, k, device=device).softmax(-1)
    dy = (torch.randn_like(x) * 0.1).bfloat16()
    valid = torch.full((), t, device=device, dtype=torch.int32)
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream())
    plan = graph = None
    try:
        with torch.cuda.stream(stream):
            plan, capacity = make_plan(recipe, args, ids, valid, group, device)
            plan.prepare(w1, w2)

            def step():
                plan.reset_errors()
                y = plan.forward(x, ids, scores, w1, w2, valid)
                return (y, *plan.backward(dy, x, ids, scores, w1, w2, valid))

            for _ in range(args.warmup):
                outputs = step()
            torch.cuda.synchronize(device)
            plan.check()
            reference = [
                value.detach().flatten()[:: max(1, value.numel() // 8192)].float().cpu().clone()
                for value in outputs
            ]
            del outputs
            if hasattr(plan, "forward_context_receives"):
                assert not plan._forward_context_available and not plan._using_forward_context
                for index in range(len(plan.forward_context_receives)):
                    plan.forward_context_receives[index] = None
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = step()
            for _ in range(args.warmup):
                graph.replay()
            torch.cuda.synchronize(device)
            plan.check()
            elapsed = []
            if profile:
                torch.cuda.profiler.start()
            for _ in range(args.repeats):
                dist.barrier(group=group)
                torch.cuda.synchronize(device)
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record(stream)
                for _ in range(args.iterations):
                    graph.replay()
                end.record(stream)
                end.synchronize()
                timing = torch.tensor(
                    begin.elapsed_time(end) / args.iterations, device=device, dtype=torch.float64
                )
                dist.all_reduce(timing, op=dist.ReduceOp.MAX, group=group)
                elapsed.append(float(timing))
            if profile:
                torch.cuda.profiler.stop()
            plan.check()
            errors = []
            for value, expected in zip(outputs, reference):
                actual = value.detach().flatten()[:: max(1, value.numel() // 8192)].float().cpu()
                error = float((actual - expected).norm() / expected.norm().clamp_min(1e-12))
                errors.append(error)
            good = torch.tensor(
                int(all(math.isfinite(v) and v < 0.05 for v in errors)), device=device
            )
            dist.all_reduce(good, op=dist.ReduceOp.MIN, group=group)
            if not int(good):
                raise RuntimeError(f"Graph replay sample mismatch on rank {rank}: {errors}")
            if rank == 0:
                print(
                    json.dumps(
                        dict(
                            case="mega_moe_training_benchmark",
                            recipe=recipe["id"],
                            ep_size=size,
                            shape=dict(tokens=t, hidden=h, intermediate=i, local_experts=e, topk=k),
                            precision=args.compute,
                            backend=recipe["backend"],
                            layout=recipe.get("layout"),
                            median_ms=statistics.median(elapsed),
                            samples_ms=elapsed,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            repeats=args.repeats,
                            graph=True,
                            instrumented=profile,
                            measurement=recipe["measurement"],
                            dw=recipe["dw"],
                            communication=recipe["communication"],
                            torch=torch.__version__,
                            device=torch.cuda.get_device_name(device),
                            extension_sha256=hashlib.sha256(
                                Path(dg._C.__file__).read_bytes()
                            ).hexdigest(),
                            lut_sha256=recipe["lut_sha256"],
                            compact_capacity_plan=capacity,
                            resolved_gemm_configs=getattr(plan, "resolved_gemm_configs", None),
                            deep_ep_rdma_ranks=getattr(plan, "deep_ep_rdma_ranks", None),
                            sampled_replay_relative_rms=errors,
                        )
                    ),
                    flush=True,
                )
    finally:
        if graph is not None:
            graph.reset()
        if plan is not None and hasattr(plan, "close"):
            plan.close()
        destroy_training_process_group()


def prepare_deepep_compat(root):
    import importlib.util
    import shutil

    original = Path(importlib.util.find_spec("deep_ep").origin).parent
    parent = root / "build/campaign/deepep-native-scope"
    destination = parent / "deep_ep"
    if not destination.exists():
        shutil.copytree(original, destination)
    records = []
    for name in ("handle.cuh", "comm.cuh"):
        source = original / "include/deep_ep/common" / name
        target = destination / "include/deep_ep/common" / name
        contents = source.read_text()
        if name == "handle.cuh":
            contents = contents.replace("cuda::thread_scope_thread", "hggc::thread_scope_thread")
            contents = contents.replace("cuda::thread_scope_device", "hggc::thread_scope_device")
        else:
            # Native half aliases/guards must precede cooperative_groups' CUDA
            # half headers, including the standalone teardown barrier kernel.
            contents = contents.replace(
                "#include <cooperative_groups.h>\n#include <pccl.h>",
                "#include <pccl.h>\n#include <cooperative_groups.h>",
            )
        target.write_text(contents)
        records.append(
            dict(
                path=str(target.relative_to(parent)),
                original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                patched_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            )
        )
    bridge = root / "build/campaign/compat/deepep/pccl.h"
    bridge.parent.mkdir(parents=True, exist_ok=True)
    bridge.write_text("""#pragma once
    // Native PCCL declarations before CUDA-compatible JIT headers.
    #include <hggc_runtime_api.h>
    #include <hggc_fp16.h>
    #include <hggc_bf16.h>
    #define __CUDA_FP16_H__
    #define __CUDA_FP16_HPP__
    #define __CUDA_BF16_H__
    #define __CUDA_BF16_HPP__
    using __nv_bfloat16 = __ppu_bfloat16;
    using __nv_bfloat162 = __ppu_bfloat162;
    using __nv_bfloat16_raw = __ppu_bfloat16_raw;
    using __nv_bfloat162_raw = __ppu_bfloat162_raw;
    using nv_bfloat16 = __ppu_bfloat16;
    using nv_bfloat162 = __ppu_bfloat162;
    #include_next <pccl.h>
    """)
    result = dict(
        original_package=str(original),
        shadow_package=str(destination),
        patches=records,
        installed_package_modified=False,
        pccl_bridge=str(bridge),
        pccl_bridge_sha256=hashlib.sha256(bridge.read_bytes()).hexdigest(),
    )
    (parent / "compat-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep", type=int, choices=(32, 64), default=32)
    parser.add_argument("--model", choices=("Qwen3.5-397B-A17B", "Qwen3.8-2.4T-A95B"))
    parser.add_argument("--precision", choices=("bf16", "fp8"))
    parser.add_argument("--backend", choices=("megamoe", "deepep"), default="megamoe")
    parser.add_argument("--deepep-layout", choices=("windowed", "compact"), default="windowed")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prepare-deepep-compat",
        action="store_true",
        help="Create an SDK 2.2 DeepEP header shadow under build/; no installed files change",
    )
    parser.add_argument("--profile", action="store_true")
    for name in ("warmup", "iterations", "repeats"):
        parser.add_argument("--" + name, type=int)
    cli = parser.parse_args()
    if cli.prepare_deepep_compat:
        prepare_deepep_compat(ROOT)
        return
    if cli.model is None or cli.precision is None:
        parser.error("--model and --precision are required")
    if cli.backend != "deepep" and cli.deepep_layout != "windowed":
        parser.error("--deepep-layout compact requires --backend deepep")
    book = load_recipes()
    recipe = select_recipe(book, cli.ep, cli.model, cli.precision, cli.backend, cli.deepep_layout)
    args = recipe_arguments(recipe)
    for name in ("warmup", "iterations", "repeats"):
        value = getattr(cli, name)
        if value is not None:
            if value < 1:
                parser.error(name + " must be positive")
            setattr(args, name, value)
    environment = configure(recipe, book, write_lut=False)
    resolved = dict(
        recipe=recipe["id"],
        expected_world_size=cli.ep,
        environment=environment,
        arguments=vars(args),
        lut_sha256=recipe["lut_sha256"],
    )
    if cli.dry_run:
        print(json.dumps(resolved, indent=2))
        return
    if int(os.environ.get("WORLD_SIZE", "1")) != cli.ep:
        parser.error(
            f"Launch with torchrun on {cli.ep // 8} eight-device hosts in one physical supernode"
        )
    configure(recipe, book)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(resolved), flush=True)
    benchmark(recipe, args, cli.profile)


if __name__ == "__main__":
    main()
