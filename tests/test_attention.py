import random
import torch
from typing import Tuple

import deep_gemm
from bench import *
from utils import test_mqa_logits, test_paged_mqa_logits, set_acc_check
from deep_gemm.jit_kernels.utils import is_ppu1v5_device


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


def test_ks_ke():
    # Single INT8 case: seq_len_q=1 with ks != 0, manually constructed (not via test_mqa_logits).
    # Verifies the kernel correctly offsets into KV when the valid window does not start at 0.
    # Only q[0, 0, 0]=1 and k[4, 0]=5 are non-zero, so logits[0, 4] = relu(1*5) * 1 * 1 = 5.
    print('Testing INT8 MQA Logits (seq_len_q=1, ks!=0):')
    sq, nh, hd = 1, 32, 128
    skv = 6
    q = torch.zeros((sq, nh, hd), device='cuda', dtype=torch.int8)
    k = torch.zeros((skv, hd), device='cuda', dtype=torch.int8)
    k_scale = torch.ones(skv, device='cuda', dtype=torch.float32)
    weights = torch.ones((sq, nh), device='cuda', dtype=torch.float32)
    q[0, 0, 0] = 1
    k[4, 0] = 5
    # ks != 0: only KV token 4 is within [ks, ke)
    ks = torch.tensor([4], device='cuda', dtype=torch.int32)
    ke = torch.tensor([5], device='cuda', dtype=torch.int32)
    # Reference: logits = (relu(q·k) * weights).sum(heads) * k_scale
    scores = torch.matmul(q.float(), k.float().T)            # [1, 32, 6]
    ref = (scores.relu() * weights[:, :, None]).sum(dim=1)  # [1, 6]
    ref = ref * k_scale[None, :]
    actual = deep_gemm.int8_mqa_logits(q, (k, k_scale), weights, ks, ke, clean_logits=False)
    torch.testing.assert_close(actual[0, 4].float(), ref[0, 4].float(), rtol=1e-3, atol=1e-3)
    print(f'  INT8 ks!=0 case: expected[0,4]={ref[0, 4].item():.4f}, actual[0,4]={actual[0, 4].item():.4f}')
    print("Passed\n")


def test_mqa_logits_loop():
    print('Testing MQA Logits:')
    qk_dtype_list = [torch.int8]
    if is_ppu1v5_device():
        qk_dtype_list.extend([torch.float8_e4m3fn, torch.uint8]) # fp8, fp4
    num_heads, head_dim = 64, 128
    for qk_dtype in qk_dtype_list:
        for seq_len in (2048, 4096):
            # deepseek (64, 128), glm5 (32, 128)
            for num_heads, head_dim in [(32, 128), (64, 128)]:
                if qk_dtype == torch.uint8 and (num_heads != 64 or head_dim != 128): continue
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
                    if qk_dtype in [torch.uint8, torch.float8_e4m3fn, torch.int8] and do_check:
                        args['logits_dtype'] = torch.bfloat16
                        test_mqa_logits(args)
    for args in [
        # bf16 test
        {
            'data_type': torch.bfloat16,
            'seq_len_q': 4096,
            'seq_len_kv': 8191,
            'num_heads': 64,
            'head_dim': 128,
        },
    ]:
        set_acc_check(True)
        test_mqa_logits(args)
    print("Passed\n")


def test_paged_mqa_logits_loop():
    print('Testing Paged MQA Logits:')
    qk_dtype_list = [torch.int8]
    if is_ppu1v5_device():
        qk_dtype_list.extend([torch.float8_e4m3fn, torch.uint8]) # fp8, fp4
    for qk_dtype in qk_dtype_list:
        for batch_size, next_n in [(1, 1), (64, 1), (64, 2), (128, 1)]:
            # deepseek (64, 128), glm5 (32, 128)
            for num_heads, head_dim in [(32, 128), (64, 128)]:
                if next_n == 2 and num_heads == 32: continue
                if qk_dtype == torch.uint8 and (num_heads != 64 or head_dim != 128): continue
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
                    if qk_dtype in [torch.uint8, torch.float8_e4m3fn, torch.int8] and do_check:
                        args['logits_dtype'] = torch.bfloat16
                        test_paged_mqa_logits(args)

    for args in [
        # context_len = 0
        {
            'data_type': torch.int8,
            'batch_size': 4,
            'next_n': 1,
            'num_heads': 64,
            'head_dim': 128,
            'distribution': [20, 10, 0, 0]
        },
        # mtp: context_len = 0 in middle
        {
            'data_type': torch.int8,
            'batch_size': 16,
            'next_n': 1,
            'num_heads': 32,
            'head_dim': 128,
            'distribution': [4090,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1]
        },
        # batch_size > 1024
        {
            'data_type': torch.bfloat16,
            'batch_size': 1119,
            'next_n': 1,
            'num_heads': 64,
            'head_dim': 128,
            'avg_context_len': 1087
        },
    ]:
        set_acc_check(True)
        test_paged_mqa_logits(args)
    print("Passed\n")


def test_nvtx():
    print('Testing mvtx dump:')
    import torch.cuda.nvtx as nvtx
    nvtx.range_push("paged_mqa_logits")
    qk_dtype_list = [torch.int8]
    for qk_dtype in qk_dtype_list:
        for batch_size, next_n in [(1, 1), (64, 1)]:
            for num_heads, head_dim in [(32, 128), (64, 128)]:
                if next_n == 2 and num_heads == 32: continue
                for avg_kv in [8192]:
                    # Call test_paged_mqa_logits with the parameters
                    args = {
                        'data_type': qk_dtype,
                        'batch_size': batch_size,
                        'next_n': next_n,
                        'avg_context_len': avg_kv,
                        'num_heads': num_heads,
                        'head_dim': head_dim
                    }
                    set_acc_check(False)
                    test_paged_mqa_logits(args)
    nvtx.range_pop()
    print("Passed\n")


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    # test_gemm_skip_head_mid()
    # test_nvtx()

    test_ks_ke()
    test_mqa_logits_loop()
    test_paged_mqa_logits_loop()
