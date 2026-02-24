import random
import torch
from typing import Tuple

import deep_gemm
from bench import *
from utils import test_mqa_logits, test_paged_mqa_logits, set_acc_check
from deep_gemm.jit_kernels.utils import is_ppu1v5_device

# from generators import get_arch_major, generate_normal, get_ue8m0_usage, get_kernel_types, MajorTypeAB


def apply_skip_head_mid(d: torch.Tensor, head_splits: Tuple[int, int, int]):
    left, mid, right = head_splits
    m, n = d.shape
    assert n % (left + right) == 0
    num_heads = n // (left + right)

    # Split and insert padding tensor
    d = d.view(m, num_heads, -1)
    d_left = d[:, :, :left]
    d_right = d[:, :, -right:]

    d_mid = torch.zeros((m, num_heads, mid), dtype=d.dtype, device=d.device)
    return torch.cat([d_left, d_mid, d_right], dim=2).view(m, -1)


def test_gemm_skip_head_mid() -> None:
    print('Testing GEMM skip head mid:')
    head_splits = (128, 64, 128)

    major_a, major_b = MajorTypeAB.KMajor,  MajorTypeAB.KMajor
    out_dtype, accumulate = torch.bfloat16, False

    for kernel_type in get_kernel_types(dtype=torch.float8_e4m3fn):
        for m in (128, 4096):
            for n, k in [(32768, 512), (8192, 512)]:
                kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
                use_ue8m0 = get_ue8m0_usage(kernel_type)
                disable_ue8m0_cast = not use_ue8m0

                a, b, _, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_ue8m0=use_ue8m0)
                d = apply_skip_head_mid(d, head_splits)
                ref_d = apply_skip_head_mid(ref_d, head_splits)

                deep_gemm.fp8_gemm_nt_skip_head_mid(a, b, d, head_splits, disable_ue8m0_cast=disable_ue8m0_cast)
                diff = calc_diff(d, ref_d)
                assert diff < 0.001, f'{m=}, {n=}, {k=}, {kernel_opt}, {diff:.5f}'

                t = bench_kineto(lambda: deep_gemm.fp8_gemm_nt_skip_head_mid(a, b, d, head_splits, disable_ue8m0_cast=disable_ue8m0_cast),
                                'fp8_gemm', suppress_kineto_output=True)
                print(f' > Perf (m={m:5}, n={n:5}, k={k:5}, {kernel_opt}): '
                    f'{t * 1e6:4.0f} us | '
                    f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
                    f'{(count_bytes(a, b, d)) / 1e9 / t:4.0f} GB/s')
    print()


def test_mqa_logits_loop():
    print('Testing MQA Logits:')
    qk_dtype_list = [torch.bfloat16, torch.int8]
    if is_ppu1v5_device():
        qk_dtype_list.append(torch.float8_e4m3fn)
    num_heads, head_dim = 64, 128
    for qk_dtype in qk_dtype_list:
        for seq_len in (2048, 4096):
            # deepseek v3.2 (64, 128), glm5 (32, 128)
            for num_heads, head_dim in [(32, 128), (64, 128)]:
                for seq_len_kv in (4096, 8192, 16384, 32768, 65536, 131072):
                    do_check = (seq_len_kv < 32768)
                    # Call test_mqa_logits with the parameters
                    args = {
                        'data_type': qk_dtype,
                        'seq_len_q': seq_len,
                        'seq_len_kv': seq_len_kv,
                        'num_heads': num_heads,
                        'head_dim': head_dim
                    }
                    set_acc_check(do_check)
                    test_mqa_logits(args)

    print("Passed\n")


def test_paged_mqa_logits_loop():
    print('Testing Paged MQA Logits:')
    qk_dtype_list = [torch.bfloat16, torch.int8]
    if is_ppu1v5_device():
        qk_dtype_list.append(torch.float8_e4m3fn)
    for qk_dtype in qk_dtype_list:
        for batch_size, next_n in [(1, 1), (64, 1), (64, 2), (128, 1)]:
            # deepseek v3.2 (64, 128), glm5 (32, 128)
            for num_heads, head_dim in [(32, 128), (64, 128)]:
                if next_n == 2 and num_heads == 32: continue
                for avg_kv in (8192, 32768):
                    do_check = (avg_kv < 32768)
                    # Call test_paged_mqa_logits with the parameters
                    args = {
                        'data_type': qk_dtype,
                        'batch_size': batch_size,
                        'next_n': next_n,
                        'avg_context_len': avg_kv,
                        'num_heads': num_heads,
                        'head_dim': head_dim
                    }
                    set_acc_check(do_check)
                    test_paged_mqa_logits(args)

    # context_len = 0
    args = {
        'data_type': torch.int8,
        'batch_size': 4,
        'next_n': 1,
        'num_heads': 64,
        'head_dim': 128,
        'distribution': [20, 10, 0, 0]
    }
    set_acc_check(True)
    test_paged_mqa_logits(args)
    print("Passed\n")


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    # test_gemm_skip_head_mid()

    test_mqa_logits_loop()
    test_paged_mqa_logits_loop()
