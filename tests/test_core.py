import random
import torch
from typing import Tuple
import os

import deep_gemm
from deep_gemm import bench_kineto, get_m_alignment_for_contiguous_layout
from utils import calc_diff, construct, construct_contiguous_grouped, construct_grouped_masked
from utils import test_gemm, test_m_grouped_gemm_contiguous, test_m_grouped_gemm_masked, test_m_grouped_gemm_nopad
from utils import set_acc_check, get_acc_check, check_signal
from utils import judge_device_type
use_ppu = judge_device_type()
def test_gemm_loop(d: torch.dtype) -> None:
    for m in (64, 128, 4096):
        for k, n in [(576, 7168), (7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
            args = {"m":m, "n":n, "k":k, "data_type":d}
            test_gemm(args)

            if benchmark:
                # Construct new tensors only once to avoid L2 cache acceleration (creating them puts them in L2)
                x, y, out, ref_out = construct(m, k, n, d)

                # noinspection PyShadowingNames
                def test_func():
                    if d == torch.bfloat16:
                        deep_gemm.gemm_bf16_bf16_bf16_nt(x, y, out)
                    else:
                        deep_gemm.gemm_int8_int8_bf16_nt(x, y, out)

                t = bench_kineto(test_func, 'gemm', suppress_kineto_output=True)
                print(f' > Performance (dtype={str(d)}, m={m:5}, n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                    f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
                    f'{(m * k + k * n + m * n * 2) / 1e9 / t:4.0f} GB/s')
    print("Passed\n")

def test_m_grouped_gemm_contiguous_loop(d: torch.dtype) -> None:
    for num_groups, expected_m_per_group, k, n in ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168),
                                                    (8, 4096, 7168, 4096), (8, 4096, 2048, 7168),
                                                    (32, 256, 7168, 4096), (32, 256, 2048, 7168)):
        args = {"groups":num_groups,"m":num_groups*expected_m_per_group, "n":n, "k":k, "data_type":d, "distribution": "uniform"}
        test_m_grouped_gemm_contiguous(args)

    if benchmark:
        # noinspection PyShadowingNames
        def test_func():
            if (d == torch.bfloat16):
                deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_contiguous(x, y, out, m_indices)
            else:
                deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_contiguous(x, y, out, m_indices)

        t = bench_kineto(test_func, 'gemm', suppress_kineto_output=True)
        valid_m = (m_indices != -1).sum().item()
        print(f' > Perf ((contiguous dtype={str(d)}, {num_groups=:2}, {expected_m_per_group=:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
        f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
        f'{(valid_m * k + num_groups * k * n + m * n * 2) / 1e9 / t:4.0f} GB/s')

    print("Passed\n")


def test_m_grouped_gemm_masked_loop(d: torch.dtype) -> None:
    # Test correctness
    # num_groups, expected_m_per_group, k, n = 4, 2, 128, 64
    for num_groups, expected_m_per_group in ((1, 1024), (2, 512), (4, 256)):
        for k, n in ((7168, 4096), (2048, 7168), ):
            for enable_sbo_overlap in (False, True):
                for i in range(10):
                    args = {"groups":num_groups,"m":num_groups*expected_m_per_group, "n":n, "k":k, "data_type":d, "distribution": "uniform", "max_m": 2048,
                            "enable_sbo_overlap":enable_sbo_overlap}
                    test_m_grouped_gemm_masked(args)

    if benchmark:
        # noinspection PyShadowingNames
        def test_func():
            if (d == torch.bfloat16):
                deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_masked(x, y, out, masked_m, expected_m_per_group)
            else:
                deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_masked(x, y, out, masked_m, expected_m_per_group)

        x, y, masked_m, out, ref_out = construct_grouped_masked(num_groups, max_m, k, n, d, expected_m_per_group)

        # Test performance with fixed shapes
        # noinspection PyUnboundLocalVariable
        valid_m = masked_m.sum().item()
        t = bench_kineto(test_func, 'gemm', suppress_kineto_output=True)

        print(f' > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
            f'throughput: {2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS, '
            f'{(valid_m * k + num_groups * k * n + valid_m * n * 2) / 1e9 / t:4.0f} GB/s')
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
    global benchmark
    benchmark = 0

    import argparse

    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('--func', default=None, type=str, choices=["DenseGemm","GroupedContiguous", "GroupedMasked", "GroupedNoPad"], required=False, help='target test func')

    args = parser.parse_args()
    if args.func is not None:
        if "GroupedContiguous" in args.func:
            #test_m_grouped_gemm_contiguous_loop(torch.int8)
            test_m_grouped_gemm_contiguous_loop(torch.bfloat16)
        elif "GroupedMasked" in args.func:
            test_m_grouped_gemm_masked_loop(torch.int8)
            test_m_grouped_gemm_masked_loop(torch.bfloat16)
        elif "GroupedNoPad" in args.func:
            test_m_grouped_gemm_nopad_loop(torch.int8)
            test_m_grouped_gemm_nopad_loop(torch.bfloat16)
        elif "DenseGemm" in args.func:
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


