"""Timing, metric modelling, and report formatting for the Megamoe tests."""

from contextlib import nullcontext
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from .ref.dist import dist_print
from .runner import _get_bench_align_env, _get_l2_flush_bytes, _resolve_l2_flush_gb

LL_BASELINE_STAGE_NAMES = (
    "dispatch",
    "l1_gemm",
    "swiglu_quant",
    "l2_gemm",
    "combine",
)
HT_BASELINE_STAGE_NAMES = (
    "dispatch",
    "permute",
    "l1_gemm",
    "swiglu_quant",
    "l2_gemm",
    "unpermute",
    "combine",
)


def kineto_range(enabled: bool, name: str):
    return torch.profiler.record_function(name) if enabled else nullcontext()


def safe_div(numerator, denominator):
    """Return NaN when the divisor is zero."""
    return float("nan") if denominator == 0 else numerator / denominator


def bench_cuda_events(
    fn,
    num_warmup: int = 5,
    num_tests: int = 20,
    l2_flush_gb: float = 8.0,
    align_ranks: bool = False,
) -> float:
    """Return median runtime, optionally aligning distributed launches."""
    should_align, spin_cycles = _get_bench_align_env(align_ranks)
    align_tensor = torch.ones(1, dtype=torch.float, device="cuda") if should_align else None
    # The same switch the fused helper reads: with it set, the flushes must go for both sides of the
    # comparison or the baseline keeps paying a cost the fused row no longer pays.
    l2_flush_gb = _resolve_l2_flush_gb(l2_flush_gb)
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    times_ms = []
    for _ in range(num_tests):
        flush_bytes = _get_l2_flush_bytes(l2_flush_gb)
        if flush_bytes:
            torch.empty(flush_bytes // 4, dtype=torch.int, device="cuda").zero_()
        if should_align:
            if spin_cycles > 0:
                torch.cuda._sleep(spin_cycles)
            dist.all_reduce(align_tensor)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        fn()
        end_event.record()
        end_event.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))
    times_ms.sort()
    return times_ms[len(times_ms) // 2] / 1e3


def _ceil_to_multiple(value: torch.Tensor, multiple: int) -> torch.Tensor:
    return (value + multiple - 1) // multiple * multiple


def estimate_masked_issued_m(masked_m: Optional[torch.Tensor], block_m: int) -> int:
    if masked_m is None or block_m <= 0:
        return 0
    return int(_ceil_to_multiple(masked_m.to(torch.int64), block_m).sum().item())


def fmt_issued_m(l1_m: int, l2_m: int) -> str:
    return str(l1_m) if l1_m == l2_m else f"{l1_m}/{l2_m}"


def _get_quantized_row_bytes(width: int) -> int:
    """Modeled bytes of one quantized activation row: FP4 pairs plus 2 B of scales per 64 K elements."""
    return width // 2 + (width // 64) * 2


def _get_packed_activation_row_bytes(x_quantized: Tuple[torch.Tensor, torch.Tensor]) -> int:
    """The same row, measured from the quantized activation and its scale tensors."""
    return x_quantized[0].size(1) * x_quantized[0].element_size() + x_quantized[1].size(1) * x_quantized[1].element_size()


def _get_weight_bytes_per_expert(weights: Tuple[torch.Tensor, torch.Tensor]) -> int:
    """Bytes one expert's two grouped weight tensors occupy (packed FP4 and its scales)."""
    return weights[0][0].numel() * weights[0].element_size() + weights[1][0].numel() * weights[1].element_size()


def estimate_l1_l2_gemm_work(
    l1_m: int,
    l2_m: int,
    active_experts: int,
    *,
    hidden: int,
    intermediate_hidden: int,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[int, int]:
    """Return logical FLOPs and modeled HBM bytes for the two expert GEMMs."""
    l1_n, l1_k = intermediate_hidden * 2, hidden
    l2_n, l2_k = hidden, intermediate_hidden

    l1_flops = 2 * l1_m * l1_n * l1_k
    l2_flops = 2 * l2_m * l2_n * l2_k

    l1_weight_bytes_per_expert = _get_weight_bytes_per_expert(l1_weights)
    l2_weight_bytes_per_expert = _get_weight_bytes_per_expert(l2_weights)

    l1_input_bytes_per_row = _get_quantized_row_bytes(hidden)
    l2_input_bytes_per_row = _get_quantized_row_bytes(intermediate_hidden)

    l1_hbm_bytes = (l1_m * l1_input_bytes_per_row + active_experts * l1_weight_bytes_per_expert + l1_m * l1_n * 2)
    l2_hbm_bytes = (l2_m * l2_input_bytes_per_row + active_experts * l2_weight_bytes_per_expert + l2_m * l2_n * 2)
    return l1_flops + l2_flops, l1_hbm_bytes + l2_hbm_bytes


def estimate_pipeline_work(
    useful_m: int,
    active_experts: int,
    *,
    hidden: int,
    intermediate_hidden: int,
    x_quantized: Tuple[torch.Tensor, torch.Tensor],
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[int, int]:
    """Return useful FLOPs and modeled HBM bytes for the full MoE compute path.

    The model counts routed rows, the weights/scales of locally active experts,
    quantized L1 input, L1 output plus its L2 scales (write and read), and BF16
    L2 output. Communication is reported separately as ICN traffic.
    """
    flops = 2 * useful_m * hidden * intermediate_hidden * 3
    weight_bytes = 0
    for weights in (l1_weights, l2_weights):
        weight_bytes += active_experts * _get_weight_bytes_per_expert(weights)

    l1_input_bytes_per_row = _get_packed_activation_row_bytes(x_quantized)
    l2_input_bytes_per_row = _get_quantized_row_bytes(intermediate_hidden)
    hbm_bytes = (weight_bytes + useful_m * l1_input_bytes_per_row + useful_m * l2_input_bytes_per_row * 2 + useful_m * hidden * 2)
    return flops, hbm_bytes


def estimate_routed_icn_bytes(
    useful_m: int,
    *,
    hidden: int,
    x_quantized: Tuple[torch.Tensor, torch.Tensor],
) -> int:
    """Model dispatch and combine bytes for useful routed rows."""
    input_bytes_per_row = _get_packed_activation_row_bytes(x_quantized)
    return useful_m * (input_bytes_per_row + 4 + hidden * 2)


def fmt_pipeline_header() -> str:
    return (f" > {'kind':<10} {'EP':^5} "
            f"{'recv_M':>7} "
            f"{'exp':>3} "
            f"{'tok/E(min/max/avg)':>18} | "
            f"{'comp_TF':>7} "
            f"{'HBM_GBs':>7} "
            f"{'ICN_GBs':>7} "
            f"{'time_us':>8} "
            f"{'speedup':>12}")


def make_pipeline_row_fields(*, rank_idx: int, num_ranks: int, recv_m: int, active_experts: int, tokens_per_expert_min: int,
                             tokens_per_expert_max: int, tokens_per_expert_avg: float) -> Dict[str, float | int]:
    """The per-point fields fmt_pipeline_line expects, so every row is built from one key set."""
    return dict(rank_idx=rank_idx,
                num_ranks=num_ranks,
                recv_m=recv_m,
                active_experts=active_experts,
                tokens_per_expert_min=tokens_per_expert_min,
                tokens_per_expert_max=tokens_per_expert_max,
                tokens_per_expert_avg=tokens_per_expert_avg)


def dist_print_time_average(time_s: float, label: str, num_ranks: int) -> None:
    """All-reduce one point's time and print its per-rank mean as a single line per node."""
    time_tensor = torch.tensor([time_s], dtype=torch.float64, device="cuda")
    dist.all_reduce(time_tensor, op=dist.ReduceOp.SUM)
    # Already all-reduced, so every rank holds the same string: one line per node, not per rank.
    dist_print(f" > [{label}-avg] time_us={time_tensor.item() / num_ranks * 1e6:.0f}", once_in_node=True)


def fmt_pipeline_line(
    *,
    name: str,
    rank_idx: int,
    num_ranks: int,
    recv_m: int,
    active_experts: int,
    tokens_per_expert_min: int,
    tokens_per_expert_max: int,
    tokens_per_expert_avg: float,
    time_s: float,
    compute_tflops: float,
    hbm_gbs: float,
    icn_gbs: float,
    speedup: Optional[float] = None,
) -> str:
    speedup_text = (f"{speedup:5.2f}x {'fused' if speedup > 1 else 'base'}" if speedup is not None else "-")
    speedup_text = f"{speedup_text:>12}"
    tokens_per_expert = (f"{tokens_per_expert_min:d}/{tokens_per_expert_max:d}/"
                         f"{tokens_per_expert_avg:.1f}")
    return (f" > {name:<10} {rank_idx:>2d}/{num_ranks:<2d} "
            f"{recv_m:7d} {active_experts:3d} {tokens_per_expert:>18} | "
            f"{compute_tflops:7.0f} {hbm_gbs:7.0f} {icn_gbs:7.0f} "
            f"{time_s * 1e6:8.0f} {speedup_text}")


def make_stage_summary(
    *,
    name: str,
    stage_times: Optional[Dict[str, float]],
    stage_names: Sequence[str],
    range_prefix: str,
    event_time_s: float,
    useful_m: int,
    issued_l1_m: int,
    issued_l2_m: int,
    active_experts: int,
    rank_idx: int,
    num_ranks: int,
    gemm_work_kwargs: Dict,
) -> Optional[Dict[str, float | int | str]]:
    """Build the stage-timing and GEMM-model row shared by both baselines."""
    if stage_times is None:
        return None
    stage_times_by_name = {stage: stage_times[f"{range_prefix}.{stage}"] for stage in stage_names}
    comm_time = stage_times_by_name["dispatch"] + stage_times_by_name["combine"]
    gemm_time = stage_times_by_name["l1_gemm"] + stage_times_by_name["l2_gemm"]
    layout_time = stage_times_by_name.get("permute", 0.0) + stage_times_by_name.get("unpermute", 0.0)
    attributed = comm_time + gemm_time + layout_time + stage_times_by_name["swiglu_quant"]
    useful_flops, useful_hbm_bytes = estimate_l1_l2_gemm_work(useful_m, useful_m, active_experts, **gemm_work_kwargs)
    issued_flops, issued_hbm_bytes = estimate_l1_l2_gemm_work(issued_l1_m, issued_l2_m, active_experts, **gemm_work_kwargs)
    return dict(
        name=name,
        rank=rank_idx,
        world=num_ranks,
        useful_m=useful_m,
        issued_m=fmt_issued_m(issued_l1_m, issued_l2_m),
        padding_ratio=safe_div(issued_flops, useful_flops),
        active_experts=active_experts,
        event_us=event_time_s * 1e6,
        kineto_sum_us=attributed * 1e6,
        dispatch_us=stage_times_by_name["dispatch"] * 1e6,
        permute_us=stage_times_by_name.get("permute", 0.0) * 1e6,
        l1_us=stage_times_by_name["l1_gemm"] * 1e6,
        swiglu_us=stage_times_by_name["swiglu_quant"] * 1e6,
        l2_us=stage_times_by_name["l2_gemm"] * 1e6,
        unpermute_us=stage_times_by_name.get("unpermute", 0.0) * 1e6,
        combine_us=stage_times_by_name["combine"] * 1e6,
        comm_us=comm_time * 1e6,
        gemm_us=gemm_time * 1e6,
        layout_us=layout_time * 1e6,
        useful_tflops=safe_div(useful_flops / 1e12, gemm_time),
        issued_tflops=safe_div(issued_flops / 1e12, gemm_time),
        useful_hbm_gbs=safe_div(useful_hbm_bytes / 1e9, gemm_time),
        issued_hbm_gbs=safe_div(issued_hbm_bytes / 1e9, gemm_time),
    )


def fmt_stage_timing_header(with_layout: bool) -> str:
    middle_columns = [f"{'dispatch':>8}"]
    if with_layout:
        middle_columns.append(f"{'permute':>8}")
    middle_columns += [f"{'L1':>8}", f"{'swiglu':>8}", f"{'L2':>8}"]
    if with_layout:
        middle_columns.append(f"{'unperm':>8}")
    middle_columns.append(f"{'combine':>8}")
    tail_columns = [f"{'comm':>8}", f"{'gemm':>8}"]
    if with_layout:
        tail_columns.append(f"{'layout':>8}")
    return (f" > {'kind':<10} {'EP':^5} {'event':>8} {'sum_us':>8} | " + " ".join(middle_columns) + " | " + " ".join(tail_columns))


def fmt_stage_timing_line(summary: Dict[str, float | int | str], with_layout: bool) -> str:
    middle_columns = [f"{summary['dispatch_us']:8.1f}"]
    if with_layout:
        middle_columns.append(f"{summary['permute_us']:8.1f}")
    middle_columns += [f"{summary['l1_us']:8.1f}", f"{summary['swiglu_us']:8.1f}", f"{summary['l2_us']:8.1f}"]
    if with_layout:
        middle_columns.append(f"{summary['unpermute_us']:8.1f}")
    middle_columns.append(f"{summary['combine_us']:8.1f}")
    tail_columns = [f"{summary['comm_us']:8.1f}", f"{summary['gemm_us']:8.1f}"]
    if with_layout:
        tail_columns.append(f"{summary['layout_us']:8.1f}")
    return (f" > {summary['name']:<10} {summary['rank']:>2d}/{summary['world']:<2d} "
            f"{summary['event_us']:8.1f} {summary['kineto_sum_us']:8.1f} | " + " ".join(middle_columns) + " | " + " ".join(tail_columns))


def fmt_gemm_model_header() -> str:
    return (f" > {'kind':<10} {'EP':^5} "
            f"{'useful_M':>8} "
            f"{'issued_M':>9} "
            f"{'pad':>5} "
            f"{'exp':>3} | "
            f"{'useful':>6} "
            f"{'TFLOPS':>9} "
            f"{'HBM':>10} | "
            f"{'issued':>6} "
            f"{'TFLOPS':>9} "
            f"{'HBM':>10}")


def fmt_gemm_model_line(summary: Dict[str, float | int | str]) -> str:
    return (f" > {summary['name']:<10} {summary['rank']:>2d}/{summary['world']:<2d} "
            f"{summary['useful_m']:8d} {summary['issued_m']:>9} "
            f"{summary['padding_ratio']:5.2f} {summary['active_experts']:3d} | "
            f"{'':>6} "
            f"{summary['useful_tflops']:9.1f} "
            f"{summary['useful_hbm_gbs']:10.1f} | "
            f"{'':>6} "
            f"{summary['issued_tflops']:9.1f} "
            f"{summary['issued_hbm_gbs']:10.1f}")
