import argparse
import json

import torch
import deep_gemm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--default-config', action='store_true')
    parser.add_argument('--large', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(20261102)
    api = deep_gemm.m_grouped_bf16_gemm_nt_contiguous
    alignment = deep_gemm.get_mk_alignment_for_contiguous_layout() if args.default_config else 256
    assert alignment == 256, alignment
    cases = [
        ([256, 256, 256, 256], 512, 512),
        ([1, 127, 128, 129, 255, 256, 257, 511], 512, 512),
        ([0, 129, 0, 257], 512, 512),
        ([129, 257, 513], 384, 576),
        ([129, 257, 513], 512, 64),
        ([1, 129, 257], 128, 4096),
    ]
    if args.large:
        cases.append(([8192, 8192, 8192, 8192], 6144, 7168))
    print('ALIGNMENT', alignment, 'DEFAULT_CONFIG', args.default_config, flush=True)
    results = []
    for case_id, (lengths, n, k) in enumerate(cases, 1):
        allocated = [(length + alignment - 1) // alignment * alignment for length in lengths]
        m = sum(allocated)
        x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
        y = torch.randn(len(lengths), n, k, device='cuda', dtype=torch.bfloat16)
        indices = torch.full((m,), -1, dtype=torch.int32, device='cuda')
        reference = torch.zeros(m, n, dtype=torch.bfloat16, device='cuda')
        start = 0
        for expert, (length, storage) in enumerate(zip(lengths, allocated)):
            if length:
                indices[start:start + length] = expert
                reference[start:start + length] = x[start:start + length] @ y[expert].t()
            x[start + length:start + storage] = 0
            start += storage
        out = torch.full((m, n), float('nan'), dtype=torch.bfloat16, device='cuda')
        configs = None if args.default_config else (39, 256, 256, 64, 64, 64, 4, (262144, 128, 0))
        api(x, y, out, indices, configs=configs)
        torch.cuda.synchronize()
        valid = indices >= 0
        actual = out[valid].float()
        expected = reference[valid].float()
        denominator = (actual.square() + expected.square()).sum()
        diff = ((actual - expected).square().sum() / denominator).item()
        max_error = (actual - expected).abs().max().item()
        finite = bool(torch.isfinite(out).all().item())
        padding_zero = bool((out[~valid] == 0).all().item())
        result = {'case': case_id, 'lengths': lengths, 'm': m, 'n': n, 'k': k,
                  'diff': diff, 'max_abs_error': max_error, 'all_finite': finite,
                  'zero_input_padding_zero_output': padding_zero,
                  'passed': finite and diff < 0.001 and padding_zero}
        results.append(result)
        print('CASE_RESULT', json.dumps(result), flush=True)
    print('REGRESSION_RESULTS', json.dumps(results), flush=True)
    assert all(result['passed'] for result in results)


if __name__ == '__main__':
    main()
