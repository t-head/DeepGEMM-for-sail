"""Timing and diff helpers shared by the benchmark entry point and the tests."""

from .runner import bench_kineto, bench_kineto_ranges, calc_diff

__all__ = ("bench_kineto", "bench_kineto_ranges", "calc_diff")
