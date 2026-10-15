"""Megamoe fused-kernel benchmark with DeepEP reference pipelines.

Usage:
  python tests/test_megamoe_ppu.py [--num-tokens N] ...

``--run-baseline`` follows vLLM's DeepEP high-throughput pipeline:
dispatch, expert-layout permutation, two grouped GEMMs with SwiGLU and
requantization, routed-weight reduction, and combine.

``--run-low-latency-baseline`` follows vLLM's DeepEP low-latency pipeline:
low-latency dispatch, two masked grouped GEMMs with SwiGLU and requantization,
and weighted low-latency combine. W4A4 uses PPU-specific DeepEP and MegaMoE
extensions while preserving the same communication and reduction structure.

The baselines reproduce vLLM's dataflow, not its full service overlap: shared
experts are absent, expert ids stay local, and the PPU SwiGLU+quant producers
are fused kernels.
"""

import argparse
import math
import os
import random
import sys
from typing import Tuple
import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

from megamoe.bench.ref.dist import (
    dist_print,
    init_dist,
)
from megamoe.bench import bench_kineto, bench_kineto_ranges
from megamoe.bench.engine import (
    _Benchmark,
    _iter_jit_builders,
    _start_jit_prewarm,
)
from megamoe.bench.report import (
    HT_BASELINE_STAGE_NAMES,
    LL_BASELINE_STAGE_NAMES,
    bench_cuda_events,
    dist_print_time_average,
    fmt_pipeline_line,
    make_pipeline_row_fields,
    safe_div,
)
import megamoe


def _run_jit_prewarm_only(args: argparse.Namespace) -> None:
    """Compile this configuration's JIT modules and exit without launching anything.

    ``--num-bench-tests 0`` and ``--compile-only`` share this path: one run compiles
    exactly one kernel configuration, so parallel warming is one instance per
    configuration.
    """
    random.seed(0)
    compile_tokens = (max(0, args.num_max_tokens_per_rank -
                          random.randint(0, args.num_max_removed_tokens)) if args.num_tokens == 0 else args.num_tokens)
    # Total ranks = local processes x WORLD_SIZE (node count): compile-only must produce the
    # same JIT signature a real multi-node run would ask for.
    num_ranks = args.num_processes * int(os.environ.get('WORLD_SIZE', '1'))
    aligned_max_tokens = megamoe.get_megamoe_workspace_layout(num_ranks, args.num_experts, args.num_max_tokens_per_rank, args.num_topk,
                                                                args.hidden, args.intermediate_hidden).aligned_max_tokens_per_rank
    outcomes = {}
    threads = _start_jit_prewarm(
        _iter_jit_builders(num_ranks, args.num_experts, aligned_max_tokens, args.num_topk, args.hidden, args.intermediate_hidden,
                           compile_tokens, args.activation_clamp if math.isfinite(args.activation_clamp) else None,
                           args.num_max_tokens_per_rank), outcomes)
    for t in threads:
        t.join()
    for name, result in outcomes.items():
        print(f'[jit-prewarm] {name:16s} {result}', flush=True)


def _time_baseline_events(run_fn, buffer, args) -> float:
    """CUDA-event timing for one baseline, or 0.0 when that baseline was not built."""
    if buffer is None:
        return 0.0
    return bench_cuda_events(run_fn, num_warmup=args.num_warmup, num_tests=args.num_repeat, l2_flush_gb=args.l2_flush_gb, align_ranks=True)


def _time_baseline_stages(run_fn, stage_names, prefix: str, args, group, rank_idx: int, buffer):
    """Kineto range timing for one baseline's stage table, or None when it was not built."""
    if buffer is None:
        return None
    return bench_kineto_ranges(
        lambda: run_fn(profile_stages=True),
        tuple(f"{prefix}.{n}" for n in stage_names),
        num_warmup=args.num_warmup,
        num_tests=args.num_repeat,
        l2_flush_gb=args.l2_flush_gb,
        barrier=lambda: dist.barrier(group=group),
        trace_path=(f"{args.dump_profile_traces}/megamoe_{prefix}_stages_rank{rank_idx}.json" if args.dump_profile_traces else None))


def _time_benchmark_points(bench, args, rank_idx: int, group, run_fused_enabled: bool):
    """Time the fused kernel and every available baseline; returns (fused, ht, ll, ht stages, ll stages)."""

    def wait_dist_barrier():
        torch.cuda.synchronize()
        dist.barrier()

    time_fused = (bench_kineto(
        bench.run_fused,
        "ppu890_fp4_fp4_megamoe",
        num_tests=args.num_bench_tests,
        l2_flush_gb=args.l2_flush_gb,
        barrier=wait_dist_barrier,
        trace_path=(f"{args.dump_profile_traces}/megamoe_ppu_rank{rank_idx}.json" if args.dump_profile_traces else None),
    ) if run_fused_enabled else 0.0)

    time_ht = _time_baseline_events(bench.run_ht_baseline, bench.ht_buffer, args)
    time_ll = _time_baseline_events(bench.run_ll_baseline, bench.ll_buffer, args)
    ht_stages = _time_baseline_stages(bench.run_ht_baseline, HT_BASELINE_STAGE_NAMES, "ht_baseline", args, group, rank_idx, bench.ht_buffer)
    ll_stages = _time_baseline_stages(bench.run_ll_baseline, LL_BASELINE_STAGE_NAMES, "ll_baseline", args, group, rank_idx, bench.ll_buffer)
    return time_fused, time_ht, time_ll, ht_stages, ll_stages


def _check_output_diff_before_warmup(bench, args, run_fused_enabled: bool) -> None:
    """Run the fused kernel once, before the timed loop warms up, and diff that launch.

    The timed loop's warmup launches reuse -- and leave dirty -- the workspace, so the end-of-run
    check only ever sees the last launch. This run is the kernel's first, the only one that sees the
    workspace's all-zero invariant; checking it here also surfaces a wrong kernel before the timing
    work rather than after it.

    ``DG_BENCH_COLD_REPEATS`` (default 1) samples that cold launch more than once by re-zeroing
    the workspace before each launch: every launch after the first sees the previous launch's data,
    which hides a stale read whenever the payload is launch-invariant, so a single sample can miss a
    race that a handful of samples catch.
    """
    if not (args.check_output_diff and run_fused_enabled):
        return
    y_ref = bench.get_torch_reference()
    cold_repeats = max(1, int(os.environ.get('DG_BENCH_COLD_REPEATS', '1')))
    for cold_idx in range(cold_repeats):
        if cold_idx:
            bench.sym_buffer.zero_()
            torch.cuda.synchronize()
            # Every rank's workspace must be clean before ANY rank launches: a peer's zeroing pass
            # would otherwise erase the remote stores a faster rank has already issued into it.
            dist.barrier()
        bench.run_fused()
        label = ("fused first launch vs torch reference, global"
                 if cold_repeats == 1 else f"cold launch {cold_idx} vs torch reference, global")
        bench.compute_output_diff(bench.y_fused, y_ref, label=label)
        if cold_repeats > 1:
            # Where this launch's residual sits, as a cluster shape: a full-token cluster means one
            # dispatched slot's payload was consumed stale -- a token's local contributions all copy the
            # x of one slot per destination rank, so a stale slot shows up as that whole token -- while a
            # per-column one points at a single store lane.
            abs_diff = (bench.y_fused.float() - y_ref.float()).abs()
            per_token_diff = abs_diff.mean(dim=1)
            bad_cells = int((abs_diff > 1.0e3).sum())
            if bad_cells:
                print('[cold %d] rank=%d bad_tokens=%d bad_cells=%d' % (cold_idx, bench.rank_idx, int(
                    (per_token_diff > 1.0).sum()), bad_cells),
                      flush=True)
            del abs_diff, per_token_diff


def _check_output_diffs(bench, args, y_fused: torch.Tensor, run_fused_enabled: bool) -> None:
    """Compare the fused output and each baseline against the torch reference when asked to."""
    if args.check_output_diff and run_fused_enabled:
        y_ref = bench.get_torch_reference()
        bench.compute_output_diff(y_fused, y_ref)

    def _get_baseline_target():
        return y_fused if run_fused_enabled else bench.get_torch_reference()

    if bench.ht_buffer and args.check_output_diff:
        ht_out = bench.run_ht_baseline()
        bench.compute_baseline_diff("high-throughput baseline", _get_baseline_target(), ht_out)
    if bench.ll_buffer and args.check_output_diff:
        ll_out = bench.run_ll_baseline()
        bench.compute_baseline_diff("low-latency baseline", _get_baseline_target(), ll_out)


def _get_baseline_metrics(time_s: float, fused_flops: float, fused_hbm_bytes: float, icn_bytes: float) -> Tuple[float, float, float]:
    """A baseline row's compute/HBM/ICN rates: the fused workload's totals over the baseline's time."""
    if time_s <= 0:
        return 0.0, 0.0, 0.0
    return (safe_div(fused_flops / 1e12, time_s), safe_div(fused_hbm_bytes / 1e9, time_s), safe_div(icn_bytes / 1e9, time_s))


def _emit_benchmark_rows(bench, time_fused: float, time_ht: float, time_ll: float, run_fused_enabled: bool, run_ht_enabled: bool,
                         run_ll_enabled: bool, rank_idx: int, num_ranks: int) -> None:
    """Print the work stats, the performance header and one row per pipeline."""
    recv_m, active_experts, tokens_per_expert_min, tokens_per_expert_max, tokens_per_expert_avg = bench._compute_work_stats()
    fused_flops, fused_hbm_bytes, icn_bytes = bench._compute_work_flops(recv_m, active_experts)

    bench.print_perf_header()

    if run_fused_enabled:
        bench.print_fused_row(time_fused, recv_m, active_experts, fused_flops, fused_hbm_bytes, icn_bytes, tokens_per_expert_min,
                              tokens_per_expert_max, tokens_per_expert_avg)
    else:
        dist_print(" > [fused]     (skipped; --baseline-only)", once_in_node=True)
    dist.barrier()

    def _print_baseline_row(name, label, time_s, compute_tflops, hbm_gbs, icn_gbs):
        speedup = safe_div(time_s, time_fused) if run_fused_enabled else None
        row_fields = make_pipeline_row_fields(rank_idx=rank_idx,
                                              num_ranks=num_ranks,
                                              recv_m=recv_m,
                                              active_experts=active_experts,
                                              tokens_per_expert_min=tokens_per_expert_min,
                                              tokens_per_expert_max=tokens_per_expert_max,
                                              tokens_per_expert_avg=tokens_per_expert_avg)
        bench._dist_print_by_rank(
            fmt_pipeline_line(name=name,
                              time_s=time_s,
                              compute_tflops=compute_tflops,
                              hbm_gbs=hbm_gbs,
                              icn_gbs=icn_gbs,
                              speedup=speedup,
                              **row_fields))
        dist_print_time_average(time_s, label, num_ranks)

    if bench.ht_buffer is not None:
        ht_tflops, ht_hbm_gbs, ht_icn_gbs = _get_baseline_metrics(time_ht, fused_flops, fused_hbm_bytes, icn_bytes)
        _print_baseline_row("[ht_base]", "ht_base", time_ht, ht_tflops, ht_hbm_gbs, ht_icn_gbs)
    elif run_ht_enabled:
        dist_print(" > [ht_base] (deep_ep unavailable)", once_in_node=True)
    else:
        dist_print(" > [ht_base] (disabled; pass --run-baseline to compare)", once_in_node=True)

    if bench.ll_buffer is not None:
        ll_tflops, ll_hbm_gbs, ll_icn_gbs = _get_baseline_metrics(time_ll, fused_flops, fused_hbm_bytes, icn_bytes)
        _print_baseline_row("[ll_base]", "ll_base", time_ll, ll_tflops, ll_hbm_gbs, ll_icn_gbs)
    elif run_ll_enabled:
        dist_print(" > [ll_base] (deep_ep unavailable)", once_in_node=True)


def run_benchmark_process(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    bench = _Benchmark(group, rank_idx, num_ranks, args)

    # --- Early-exit paths ---
    if args.num_bench_tests == 0:
        # Compile-only (equivalent to --compile-only): warm this configuration's JIT
        # modules without building inputs -- no workspace is mapped and no kernel runs.
        _run_jit_prewarm_only(args)
        dist.barrier()
        dist.destroy_process_group()
        return

    # --- Standard benchmark ---
    y_fused = bench.build_inputs()
    bench.prewarm_jit()
    bench.print_config()

    run_fused_enabled = not bool(args.baseline_only)
    run_ht_enabled = bool(args.run_baseline)
    run_ll_enabled = bool(args.run_low_latency_baseline)

    # --- Baseline buffers ---
    if _is_deep_ep_available() and run_ht_enabled:
        bench.build_ht_baseline()
    if _is_deep_ep_available() and run_ll_enabled:
        bench.build_ll_baseline()

    bench.join_prewarm()

    # --- Correctness of the first launch, before the timing loop's warmup dirties the workspace ---
    _check_output_diff_before_warmup(bench, args, run_fused_enabled)

    # --- Benchmark execution ---
    time_fused, time_ht, time_ll, ht_stages, ll_stages = _time_benchmark_points(bench, args, rank_idx, group, run_fused_enabled)

    # --- Correctness ---
    _check_output_diffs(bench, args, y_fused, run_fused_enabled)

    # --- Work statistics, performance header and rows ---
    _emit_benchmark_rows(bench, time_fused, time_ht, time_ll, run_fused_enabled, run_ht_enabled, run_ll_enabled, rank_idx, num_ranks)

    # --- Stage timings ---
    bench.print_ht_stage_report(ht_stages, time_ht)
    bench.print_ll_stage_report(ll_stages, time_ll)

    # --- Cleanup ---
    dist.barrier()
    bench.destroy()


def _is_deep_ep_available() -> bool:
    try:
        import deep_ep  # noqa: F401
        return True
    except Exception:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPU890 Megamoe fused kernel and DeepEP reference pipelines")
    parser.add_argument("--compile-only",
                        action="store_true",
                        help="Equivalent to --num-bench-tests 0: compile this configuration and exit without launching anything")
    parser.add_argument("--num-processes", type=int, default=8, help="Number of spawned processes, one per GPU")
    parser.add_argument("--local-rank-idx",
                        type=int,
                        default=None,
                        help="Local rank for single-process mode, useful for external launchers/profilers")
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=8192)
    parser.add_argument("--num-tokens", type=int, default=0, help="Actual per-rank token count; 0 means num-max-tokens-per-rank")
    parser.add_argument("--num-max-removed-tokens", type=int, default=0, help="Max random token removals per rank when num-tokens is 0")
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--intermediate-hidden", type=int, default=2048, help="Intermediate (down-projection) dimension")
    parser.add_argument("--activation-clamp",
                        type=float,
                        default=10.0,
                        help="Clamp threshold for gate/up before SwiGLU; pass inf to disable")
    parser.add_argument("--num-experts", type=int, default=512)
    parser.add_argument("--num-topk", type=int, default=10)
    parser.add_argument("--masked-ratio",
                        type=float,
                        default=0.0,
                        help="Randomly mask some topk expert selections to test sparse routing edges")
    parser.add_argument("--num-bench-tests",
                        "--num-tests",
                        dest="num_bench_tests",
                        type=int,
                        default=30,
                        help="Number of bench_kineto iterations for the fused kernel")
    parser.add_argument("--num-warmup", type=int, default=5, help="baseline cuda events warmup")
    parser.add_argument("--num-repeat", type=int, default=20, help="Baseline CUDA event timing iterations")
    parser.add_argument("--l2-flush-gb",
                        type=float,
                        default=8.0,
                        help="Temporary write size used to flush L2 before baseline timing; 0 disables it")
    parser.add_argument("--run-baseline", action="store_true", help="Enable the vLLM-style DeepEP high-throughput baseline")
    parser.add_argument("--run-low-latency-baseline",
                        action="store_true",
                        help="Enable the vLLM-style DeepEP low-latency baseline (MXFP4 dispatch + FP4 masked GEMM)")
    parser.add_argument("--baseline-only",
                        action="store_true",
                        help="Skip the fused kernel entirely and measure only the enabled baseline(s)")
    parser.add_argument("--check-output-diff", type=int, default=0, help="If nonzero, run the torch reference and print output differences")
    parser.add_argument("--dump-profile-traces",
                        type=str,
                        default="",
                        help="If nonempty, write one fused Chrome trace per rank to this directory")
    args = parser.parse_args()

    if args.baseline_only and not (args.run_baseline or args.run_low_latency_baseline):
        parser.error("--baseline-only needs at least one baseline")

    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    if args.compile_only:
        _run_jit_prewarm_only(args)
        sys.exit(0)

    if args.local_rank_idx is not None:
        run_benchmark_process(args.local_rank_idx, args.num_processes, args)
    else:
        torch.multiprocessing.spawn(run_benchmark_process, args=(args.num_processes, args), nprocs=args.num_processes)
