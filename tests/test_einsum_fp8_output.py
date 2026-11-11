import torch
import deep_gemm


def quantize_output(x):
    # Upstream output: BF16 rounding, 32-column groups, UE8M0 scales.
    rows = x.to(torch.bfloat16).float().flatten(1).reshape(x.size(0), -1, 32)
    scale = torch.exp2(torch.ceil(torch.log2(rows.abs().amax(-1).clamp_min(1e-4) / 448)))
    quantized = (rows / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).view_as(x)
    exponents = (torch.log2(scale).to(torch.int32) + 127).reshape(x.size(0), -1, 4)
    packed = sum(exponents[:, :, i] << (8 * i) for i in range(4))
    return quantized, packed


def make_inputs(b, h, d, r, channelwise):
    a = torch.randn((b, h, r), device='cuda').to(torch.float8_e4m3fn)
    rhs = torch.randn((h, d, r), device='cuda').to(torch.float8_e4m3fn)
    sa = torch.rand((b, h, 1 if channelwise else (r + 127) // 128), device='cuda') + 0.1
    sb = torch.rand((h, d if channelwise else d // 128, 1 if channelwise else (r + 127) // 128), device='cuda') + 0.1
    return (a, sa), (rhs, sb)


def output_pair(b, h, d):
    out = torch.empty((b, h, d), device='cuda', dtype=torch.float8_e4m3fn)
    aligned_b = deep_gemm.get_tma_aligned_size(b, 4)
    sf = torch.empty((h * d // 128, aligned_b), device='cuda', dtype=torch.int32).t()[:b]
    return out, sf


def test_fp8_output():
    torch.manual_seed(0)
    for channelwise in (False, True):
        for b, h, d, r in ((1, 3, 128, 128), (4, 8, 256, 256), (32, 8, 1024, 4096),
                            (129, 3, 384, 256), (32, 3, 1024, 256), (256, 8, 1024, 4096), (4096, 8, 1024, 4096)):
            a, rhs = make_inputs(b, h, d, r, channelwise)
            baseline = torch.empty((b, h, d), device='cuda', dtype=torch.bfloat16)
            deep_gemm.fp8_einsum('bhr,hdr->bhd', a, rhs, baseline)
            expected, expected_sf = quantize_output(baseline)
            out, sf = output_pair(b, h, d)
            for _ in range(3):
                out.fill_(float('nan'))
                sf.fill_(-1)
                deep_gemm.fp8_einsum('bhr,hdr->bhd', a, rhs, (out, sf), epilogue=None)
                torch.testing.assert_close(sf, expected_sf, rtol=0, atol=0)
                torch.testing.assert_close(out.float(), expected.float(), rtol=0, atol=0)
            print(f'PASS channelwise={channelwise} B,H,D,R={b,h,d,r}', flush=True)
    # Check exponent floor, zero values, and packed exponents above 127.
    a, rhs = make_inputs(1, 3, 128, 128, True)
    baseline = torch.empty((1, 3, 128), device='cuda', dtype=torch.bfloat16)
    out, sf = output_pair(1, 3, 128)
    for value, scale in ((0.0, 1.0), (1.0, 1e-10), (448.0, 1.0)):
        a[0].fill_(value)
        rhs[0].fill_(value)
        a[1].fill_(scale)
        rhs[1].fill_(scale)
        deep_gemm.fp8_einsum('bhr,hdr->bhd', a, rhs, baseline)
        expected, expected_sf = quantize_output(baseline)
        deep_gemm.fp8_einsum('bhr,hdr->bhd', a, rhs, (out, sf), epilogue=None)
        torch.testing.assert_close(out.float(), expected.float(), rtol=0, atol=0)
        torch.testing.assert_close(sf, expected_sf, rtol=0, atol=0)
    print('PASS scale floor/zero/large exponent', flush=True)
    # Quantized zero-K follows upstream early_return: zero D and zero SFD.
    a, rhs = make_inputs(3, 2, 128, 0, True)
    out, sf = output_pair(3, 2, 128)
    deep_gemm.fp8_einsum('bhr,hdr->bhd', a, rhs, (out, sf), epilogue=None)
    assert not out.float().count_nonzero() and not sf.count_nonzero()
    # An output pair is only supported for BHD, with packed MN-major scales.
    a, rhs = make_inputs(3, 2, 128, 128, True)
    out, sf = output_pair(3, 2, 128)
    misaligned = torch.empty((3 * (2 * 128 + 1),), device='cuda', dtype=out.dtype).as_strided(
        out.shape, (2 * 128 + 1, 128, 1))
    for expr, output, c in (('bhr,hdr->bhd', (misaligned, sf), None),
                            ('bhr,hdr->hbd', (out.transpose(0, 1), sf), None),
                            ('bhr,hdr->bhd', (out, sf.to(torch.float32)), None),
                            ('bhr,hdr->bhd', (out, sf), out.to(torch.bfloat16))):
        try:
            deep_gemm.fp8_einsum(expr, a, rhs, output, c)
        except RuntimeError:
            pass
        else:
            raise AssertionError('invalid quantized output was accepted')
    torch.cuda.synchronize()
    print('FP8 OUTPUT ALL PASS', flush=True)


if __name__ == '__main__':
    test_fp8_output()
