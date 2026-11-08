import json

import torch

import deep_gemm
from test_fp4_core import dequantize_fp4_torch, quantize_fp4_torch


def make_case(group_ms, n, k, seed, canonical_scales):
    torch.manual_seed(seed)
    alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    assert alignment == 256
    padded_ms = [(m + alignment - 1) // alignment * alignment for m in group_ms]
    m = sum(padded_ms)
    indices = torch.full((m,), -1, dtype=torch.int32, device='cuda')
    activations = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
    weight_base = torch.randn((n, k), dtype=torch.bfloat16, device='cuda')
    weights = torch.stack([weight_base * (2.0 ** (group % 5 - 2)) for group in range(len(group_ms))])
    if m:
        a, sfa = quantize_fp4_torch(activations)
        dequant_a = dequantize_fp4_torch(a, sfa, torch.float32)
    else:
        a = torch.empty((0, k // 2), dtype=torch.uint8, device='cuda')
        sfa = torch.empty((0, (k + 31) // 32), dtype=torch.uint8, device='cuda')
        dequant_a = torch.empty((0, k), dtype=torch.float32, device='cuda')
    b, sfb = quantize_fp4_torch(weights)
    dequant_b = dequantize_fp4_torch(b, sfb, torch.float32)
    reference = torch.zeros((m, n), dtype=torch.bfloat16, device='cuda')
    start = 0
    for group, (valid_m, padded_m) in enumerate(zip(group_ms, padded_ms)):
        indices[start:start + valid_m] = group
        reference[start:start + valid_m] = dequant_a[start:start + valid_m] @ dequant_b[group].t()
        start += padded_m
    if canonical_scales and m:
        _, sfa = deep_gemm.preprocess_weights_and_scales(
            deep_gemm.PreprocessMode.MXFP4, False, weight_scale=sfa)
        b, sfb = deep_gemm.preprocess_weights_and_scales(
            deep_gemm.PreprocessMode.MXFP4, False, b, sfb)
    storage = torch.full((m + 2, n), 1232.0, dtype=torch.bfloat16, device='cuda')
    return (a, sfa), (b, sfb), storage[1:-1], indices, reference, storage


def run_case(name, group_ms, n, k, seed):
    a, b, out, indices, reference, storage = make_case(group_ms, n, k, seed, seed % 2 == 1)
    deep_gemm.m_grouped_fp4_gemm_nt_contiguous(a, b, out, indices)
    torch.cuda.synchronize()
    assert torch.all(storage[0] == 1232.0) and torch.all(storage[-1] == 1232.0), 'Output guard overwritten'
    valid = indices >= 0
    diff = float(deep_gemm.calc_diff(out[valid], reference[valid])) if sum(group_ms) else 0.0
    assert diff < 0.001, (name, seed, diff)
    result = {'name': name, 'seed': seed, 'groups': len(group_ms), 'group_ms': group_ms,
              'logical_m': sum(group_ms), 'physical_m': out.shape[0], 'n': n, 'k': k, 'diff': diff}
    print('PASSED ' + json.dumps(result), flush=True)
    return result


def test_invalid_arguments():
    a, b, out, indices, _, _ = make_case([17, 0, 257, 0], 128, 128, 0, True)
    errors = [
        ('indices_dtype', lambda: deep_gemm.m_grouped_fp4_gemm_nt_contiguous(a, b, out, indices.long())),
        ('output_dtype', lambda: deep_gemm.m_grouped_fp4_gemm_nt_contiguous(a, b, out.float(), indices)),
        ('scale_shape', lambda: deep_gemm.m_grouped_fp4_gemm_nt_contiguous(a, (b[0], b[1][:, :-1]), out, indices)),
        ('misaligned_block_m', lambda: deep_gemm.m_grouped_fp4_gemm_nt_contiguous(
            a, b, out, indices, configs=(39, 512, 128, 128, 64, 64, 3, (0, 0, 0)))),
    ]
    for name, call in errors:
        try:
            call()
        except (RuntimeError, ValueError):
            print('REJECTED ' + name, flush=True)
        else:
            raise AssertionError('Invalid argument accepted: ' + name)
    return len(errors)


def main():
    sparse_groups = [0] * 128
    sparse_groups[0], sparse_groups[63], sparse_groups[127] = 17, 257, 129
    cases = [
        ('aligned', [256] * 4, 256, 512),
        ('uneven_empty_experts', [17, 0, 257, 0], 256, 128),
        ('all_empty', [0] * 4, 128, 128),
        ('small_k', [1, 0, 33, 257], 128, 64),
        ('odd_scale_count', [17, 129, 0, 257], 128, 96),
        ('n_tail', [257, 511, 0, 129], 384, 512),
        ('odd_n', [17, 0, 257, 0], 129, 128),
        ('sparse_expert_ids', sparse_groups, 128, 256),
    ]
    results = []
    for seed in [0, 1]:
        for name, group_ms, n, k in cases:
            results.append(run_case(name, group_ms, n, k, seed))
    rejected = test_invalid_arguments()
    assert deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_contiguous is deep_gemm.m_grouped_fp4_gemm_nt_contiguous
    print('FP4_CONTIGUOUS_TESTS_OK ' + json.dumps({'passed': len(results), 'rejected': rejected}), flush=True)


if __name__ == '__main__':
    main()
