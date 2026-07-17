import random
import torch
from typing import Tuple
import os

import deep_gemm
from deep_gemm import bench_kineto, get_m_alignment_for_contiguous_layout
from utils import calc_diff, construct, construct_contiguous_grouped, construct_grouped_masked
from utils import test_gemm, test_m_grouped_gemm_contiguous, test_m_grouped_gemm_masked, test_m_grouped_gemm_nopad
from utils import set_acc_check, get_acc_check, set_benchmark, get_benchmark

def test_gemm_loop(d: torch.dtype) -> None:
    for m in (64, 128, 4096):
        for k, n in [(576, 7168), (7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
            args = {"m":m, "n":n, "k":k, "data_type":d}
            test_gemm(args)
    print("Passed\n")

def test_m_grouped_gemm_contiguous_loop(d: torch.dtype) -> None:
    for num_groups, expected_m_per_group, k, n in ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168),
                                                    (8, 4096, 7168, 4096), (8, 4096, 2048, 7168),
                                                    (32, 256, 7168, 4096), (32, 256, 2048, 7168)):
        args = {"groups":num_groups,"m":num_groups*expected_m_per_group, "n":n, "k":k, "data_type":d, "distribution": "uniform"}
        test_m_grouped_gemm_contiguous(args)
    print("Passed\n")


def test_m_grouped_gemm_masked_loop(d: torch.dtype) -> None:
    # Test correctness
    # num_groups, expected_m_per_group, k, n = 4, 2, 128, 64
    for num_groups, expected_m_per_group in ((1, 1024), (2, 512), (4, 256)):
        for k, n in ((7168, 4096), (2048, 7168), ):
            for enable_sbo_overlap in (False, True):
                args = {"groups":num_groups,"m":num_groups*expected_m_per_group, "n":n, "k":k, "data_type":d, "distribution": "uniform", "max_m": 2048,
                        "enable_sbo_overlap":enable_sbo_overlap}
                test_m_grouped_gemm_masked(args)
    print('passed\n')


def test_m_grouped_gemm_nopad_loop(d: torch.dtype) -> None:
    for num_groups, expected_m_per_group in ((256, 1), (256, 4), (256, 16), (256, 32), (128, 8), (128, 64), (128, 1024)):
        for k, n in ((7168, 4096), (2048, 7168), (256, 768), (512, 128)):
    # num_groups, expected_m_per_group, k, n = 256, 1, 7168, 4096
            args = {"groups":num_groups,"m":num_groups*expected_m_per_group, "n":n, "k":k, "data_type":d, "distribution": "uniform"}
            test_m_grouped_gemm_nopad(args)

    print("Passed\n")


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    set_acc_check(1)

    import argparse

    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('--func', default=None, nargs="*", choices=["DenseGemm","GroupedContiguous", "GroupedMasked", "GroupedNoPad"], required=False, help='target test func')
    parser.add_argument('--benchmark', default=False, action="store_true", required=False, help='specify if run benchmark')

    args = parser.parse_args()
    set_benchmark(args.benchmark)
    if args.func is not None:
        for item in args.func:
            if "GroupedContiguous" in item:
                test_m_grouped_gemm_contiguous_loop(torch.int8)
                test_m_grouped_gemm_contiguous_loop(torch.bfloat16)
            elif "GroupedMasked" in item:
                test_m_grouped_gemm_masked_loop(torch.int8)
                test_m_grouped_gemm_masked_loop(torch.bfloat16)
            elif "GroupedNoPad" in item:
                test_m_grouped_gemm_nopad_loop(torch.int8)
                test_m_grouped_gemm_nopad_loop(torch.bfloat16)
            elif "DenseGemm" in item:
                test_gemm_loop(torch.int8)
                test_gemm_loop(torch.bfloat16)
            else:
                print("invalid test function\n")
    else:
        test_gemm_loop(torch.int8)
        test_m_grouped_gemm_contiguous_loop(torch.int8)
        test_m_grouped_gemm_masked_loop(torch.int8)
        test_m_grouped_gemm_nopad_loop(torch.int8)

        test_gemm_loop(torch.bfloat16)
        test_m_grouped_gemm_contiguous_loop(torch.bfloat16)
        test_m_grouped_gemm_masked_loop(torch.bfloat16)
        test_m_grouped_gemm_nopad_loop(torch.bfloat16)


