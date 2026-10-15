"""Benchmark timing harness: kineto profiling and calc_diff."""

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


class EmptySuppress:

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def _resolve_l2_flush_gb(l2_flush_gb: float, caller_enables: bool = True) -> float:
    """The flush size to use: zero when the caller disabled it or DG_BENCH_NO_L2_FLUSH is set.

    The switch is the one all three timing helpers read: with it set, the flushes must go for both
    sides of a comparison or the baseline keeps paying a cost the fused row no longer pays.
    """
    if not caller_enables:
        return 0.0
    if int(os.environ.get('DG_BENCH_NO_L2_FLUSH', 0)):
        return 0.0
    return l2_flush_gb


def _get_l2_flush_bytes(l2_flush_gb: float) -> int:
    """The flush payload in bytes, capped to what the device can spare, or 0 when there is no flush."""
    if l2_flush_gb <= 0:
        return 0
    free_bytes, _ = torch.cuda.mem_get_info()
    flush_bytes = min(int(l2_flush_gb * 1e9), int(free_bytes * 0.5))
    return flush_bytes if flush_bytes >= 4 else 0


def _get_bench_align_env(caller_aligns: bool = True) -> Tuple[bool, int]:
    """(align the ranks, spin cycles) as every timing helper reads them from the environment."""
    should_align = bool(int(os.environ.get('DG_BENCH_ALIGN_RANKS', 1))) and dist.is_available() and dist.is_initialized() \
        and caller_aligns
    spin_cycles = int(float(os.environ.get('DG_BENCH_SPIN_CYCLES', 2e6)))
    return should_align, spin_cycles


def _get_device_time_total(event) -> Optional[float]:
    """The event's total device time, under whichever key this torch build exposes."""
    device_time_total = getattr(event, 'device_time_total', None)
    if device_time_total is None:
        device_time_total = getattr(event, 'cuda_time_total', None)
    return device_time_total


def _get_event_avg_ms(event, name: str, kind: str) -> float:
    """The event's average device time in milliseconds, validated against the profiler's count."""
    device_time_total = _get_device_time_total(event)
    if device_time_total is None or event.count <= 0:
        raise RuntimeError(f'Kineto {kind} {name!r} has no CUDA timing')
    return float(device_time_total) / event.count / 1e6


def _get_kineto_kernel_ms(averages: Dict, kernel_names: Tuple[str, ...], num_tests: int) -> List[float]:
    """Per-kernel average milliseconds from a Kineto event table.

    Reads the event list rather than the rendered table: the table truncates long kernel names and
    its time column is a formatted string whose unit has to be parsed back.
    """
    kernel_times = []
    for name in kernel_names:
        matches = [event for key, event in averages.items() if name in key]
        assert len(matches) == 1, f'Errors of the kernel {name} in the profiling table'
        event = matches[0]
        time_ms = _get_event_avg_ms(event, name, 'kernel')
        if event.count % num_tests != 0:
            # A run whose iterations did not all launch the kernel would otherwise be averaged in
            # silently.
            raise RuntimeError(f'Kineto kernel {name!r} ran {event.count} times, not a multiple of the '
                               f'{num_tests} timed iterations')
        kernel_times.append(time_ms)
    return kernel_times


def bench_kineto(fn,
                 kernel_names,
                 num_tests: int = 30,
                 trace_path: str = None,
                 flush_l2: bool = True,
                 l2_flush_gb: float = 8.0,
                 barrier: Optional[Callable] = None):
    if num_tests <= 0:
        raise ValueError("num_tests must be positive")
    using_nsys = os.environ.get('DG_NSYS_PROFILING', '0') != '0'

    # Same flush sizing and switch as bench_kineto_ranges and report.bench_cuda_events: cap the
    # request to what the device can spare. The bytes are computed once here, unlike the other two,
    # because this loop runs its test pass twice and compares the two.
    flush_bytes = _get_l2_flush_bytes(_resolve_l2_flush_gb(l2_flush_gb, flush_l2))

    should_align, spin_cycles = _get_bench_align_env()
    align_tensor = torch.ones(1, dtype=torch.float, device='cuda') if should_align else None

    fn()

    with EmptySuppress():
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1) if not using_nsys else None
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                                          schedule=schedule) if not using_nsys else EmptySuppress()
        with profiler:
            for _ in range(2):
                for _ in range(num_tests):
                    if flush_bytes:
                        torch.empty(flush_bytes // 4, dtype=torch.int, device='cuda').zero_()
                    if barrier is not None:
                        barrier()
                        if spin_cycles > 0:
                            torch.cuda._sleep(spin_cycles)
                    if should_align:
                        dist.all_reduce(align_tensor)
                    fn()

                if not using_nsys:
                    torch.cuda.synchronize()
                    profiler.step()

    if using_nsys:
        # Kineto is not capturing under nsys; its own report carries the timing. Return NaN so
        # downstream rows render "nan" instead of a plausible number.
        return float('nan')

    assert isinstance(kernel_names, (str, tuple))
    is_tupled = isinstance(kernel_names, tuple)
    kernel_names = (kernel_names, ) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])

    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    averages = {event.key: event for event in profiler.key_averages()}
    kernel_times = _get_kineto_kernel_ms(averages, kernel_names, num_tests)
    return tuple(kernel_times) if is_tupled else kernel_times[0]


def bench_kineto_ranges(fn: Callable,
                        range_names: Sequence[str],
                        num_warmup: int = 5,
                        num_tests: int = 20,
                        l2_flush_gb: float = 8.0,
                        barrier: Optional[Callable] = None,
                        trace_path: Optional[str] = None) -> Dict[str, float]:
    """Return average CUDA time attributed to named record_function ranges."""
    if num_warmup < 0 or num_tests <= 0:
        raise ValueError("num_warmup must be non-negative and num_tests positive")
    names = tuple(range_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("range_names must be non-empty and unique")

    fn()
    torch.cuda.synchronize()

    schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    should_align, spin_cycles = _get_bench_align_env()
    # The same switch the other two helpers read, so a run with flushes disabled does not leave the
    # stage table measuring a flushed loop while the totals measure an unflushed one.
    l2_flush_gb = _resolve_l2_flush_gb(l2_flush_gb)
    align_tensor = torch.ones(1, dtype=torch.float, device='cuda') if should_align else None
    with torch.profiler.profile(activities=activities, schedule=schedule) as profiler:
        for num_iterations in (num_warmup, num_tests):
            for _ in range(num_iterations):
                flush_bytes = _get_l2_flush_bytes(l2_flush_gb)
                if flush_bytes:
                    torch.empty(flush_bytes // 4, dtype=torch.int, device='cuda').zero_()
                if barrier is not None:
                    barrier()
                if should_align:
                    if spin_cycles > 0:
                        torch.cuda._sleep(spin_cycles)
                    dist.all_reduce(align_tensor)
                fn()
            torch.cuda.synchronize()
            profiler.step()

    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    averages = {event.key: event for event in profiler.key_averages()}
    missing = [name for name in names if name not in averages]
    if missing:
        raise RuntimeError(f"Kineto did not capture ranges: {missing}")

    result = {}
    for name in names:
        result[name] = _get_event_avg_ms(averages[name], name, 'range')
    return result


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    similarity = 2 * (x * y).sum() / denominator
    return 1 - similarity
