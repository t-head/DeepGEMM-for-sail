"""Unsupported inputs must fail on the host before modifying output tensors."""
import pytest
import torch
import deep_gemm as dg


def dense_quantized(kind, m=16, n=128, k=256):
    dtype = torch.int8 if kind == 'int8' else torch.float8_e4m3fn
    a = torch.ones((m, k), device='cuda').to(dtype)
    b = torch.ones((n, k), device='cuda').to(dtype)
    block = kind == 'fp8-block'
    sa = torch.ones((m, (k + 127) // 128 if block else 1), device='cuda')
    sb = torch.ones(((n + 127) // 128, (k + 127) // 128) if block else (n, 1), device='cuda')
    out = torch.full((m, n), -123, dtype=torch.bfloat16, device='cuda')
    return (dg.int8_gemm_nt if kind == 'int8' else dg.fp8_gemm_nt), (a, sa), (b, sb), out


def reject(call, message, out):
    before = out.clone()
    with pytest.raises(RuntimeError, match=message):
        call()
    torch.testing.assert_close(out, before, rtol=0, atol=0)


def test_bug07_grouped_bf16_gemv_stride():
    a = torch.ones((1, 70), dtype=torch.bfloat16, device='cuda')
    b = torch.ones((8, 128, 70), dtype=torch.bfloat16, device='cuda')
    out = torch.full((1, 128), -123, dtype=torch.bfloat16, device='cuda')
    ids = torch.zeros(1, dtype=torch.int32, device='cuda')
    reject(lambda: dg.m_grouped_bf16_gemm_nt_nopad(a, b, out, ids),
           'grouped BF16 GEMV requires', out)


@pytest.mark.parametrize('kind', ['int8', 'fp8-channel', 'fp8-block'])
def test_bug08_quantized_dense_odd_n(kind):
    api, a, b, out = dense_quantized(kind, n=65)
    reject(lambda: api(a, b, out), 'epilogue requires even N', out)


def test_bug08_fused_bf16_odd_n():
    a = torch.ones((1, 128), dtype=torch.bfloat16, device='cuda')
    b = torch.ones((2, 65, 128), dtype=torch.bfloat16, device='cuda')
    out = torch.full((1, 65), -123, dtype=torch.bfloat16, device='cuda')
    meta = torch.zeros(4, dtype=torch.int32, device='cuda')
    reject(lambda: dg.m_grouped_bf16_gemm_nt_fused(a, b, out, meta, meta, meta, meta,
                                                  (1, 16, 64, 128, 16, 32, 2)),
           'epilogue requires even N', out)


@pytest.mark.parametrize('k', [80, 96])
def test_bug10_fused_bf16_k_tail(k):
    a = torch.ones((1, k), dtype=torch.bfloat16, device='cuda')
    b = torch.ones((2, 64, k), dtype=torch.bfloat16, device='cuda')
    out = torch.full((1, 64), -123, dtype=torch.bfloat16, device='cuda')
    meta = torch.zeros(4, dtype=torch.int32, device='cuda')
    reject(lambda: dg.m_grouped_bf16_gemm_nt_fused(a, b, out, meta, meta, meta, meta,
                                                  (1, 16, 64, 64, 16, 16, 2)),
           'A-copy kernel does not mask K tails', out)


@pytest.mark.parametrize('dtype', [torch.int8, torch.float8_e4m3fn])
def test_bug11_grouped_a_scale_view(dtype):
    a = torch.ones((32, 128), device='cuda').to(dtype)
    b = torch.ones((2, 128, 128), device='cuda').to(dtype)
    sa = torch.ones((32, 2), device='cuda')[:, :1]
    sb = torch.ones((2, 128, 1), device='cuda')
    out = torch.full((32, 128), -123, dtype=torch.bfloat16, device='cuda')
    ids = torch.arange(32, dtype=torch.int32, device='cuda') // 16
    api = dg.m_grouped_int8_gemm_nt_nopad if dtype == torch.int8 else dg.m_grouped_fp8_gemm_nt_nopad
    reject(lambda: api((a, sa), (b, sb), out, ids), 'contiguous lhs_scales', out)


@pytest.mark.parametrize('expr', ['bhr,hdr->bhd', 'bhr,hdr->hbd'])
def test_bug11_einsum_b_scale_view(expr):
    a = torch.ones((16, 2, 256), device='cuda').to(torch.float8_e4m3fn)
    b = torch.ones((2, 256, 256), device='cuda').to(torch.float8_e4m3fn)
    sa = torch.ones((16, 2, 2), device='cuda')
    sb = torch.ones((2, 2, 2, 2), device='cuda')[..., 0]
    out = torch.full((16, 2, 256) if expr.endswith('bhd') else (2, 16, 256),
                     -123, dtype=torch.bfloat16, device='cuda')
    reject(lambda: dg.fp8_einsum(expr, (a, sa), (b, sb), out), 'contiguous rhs_scales', out)


def w4fa16_call(n, stages, expand):
    a = torch.ones((16, 128), dtype=torch.bfloat16, device='cuda')
    b = torch.full((1, n, 64), 0x22, dtype=torch.uint8, device='cuda')
    sb = torch.full((1, n // 64, 256), 127, dtype=torch.uint8, device='cuda')
    out = torch.full((16, n), -123, dtype=torch.bfloat16, device='cuda')
    ids = torch.zeros(16, dtype=torch.int32, device='cuda')
    rows = torch.full((1,), 16, dtype=torch.int32, device='cuda')
    return (lambda: dg.m_grouped_gemm_w4a16_nopad(a, (b, sb), out, ids, rows,
                                                 (1, 16, 128, 128, 16, 64, 128, stages, expand))), out


def test_bug12_w4fa16_expanded_pipeline():
    call, out = w4fa16_call(256, 3, 2)
    reject(call, 'N expansion requires num_stages=2', out)


@pytest.mark.parametrize('m', [8, 8193])
def test_bug14_route_view(m):
    a = torch.empty((m, 128), dtype=torch.bfloat16, device='cuda')
    b = torch.empty((4, 128, 128), dtype=torch.bfloat16, device='cuda')
    ids = torch.arange(4, dtype=torch.int32, device='cuda').repeat(m, 1)[:, ::2]
    with pytest.raises(RuntimeError, match='contiguous topk_ids'):
        dg.moe_align_block_size(a, b, ids)
    _, rows, *_ = dg.moe_align_block_size(a, b, ids.contiguous())
    torch.testing.assert_close(rows, torch.tensor([m, 0, m, 0], dtype=torch.int32, device='cuda'))


def fused_quantized(kind, k=256):
    dtype = torch.int8 if kind == 'int8' else torch.float8_e4m3fn
    a = torch.ones((1, k), device='cuda').to(dtype)
    b = torch.ones((2, 128, k), device='cuda').to(dtype)
    block = kind == 'fp8-block'
    sa = torch.ones((1, k // 128 if block else 1), device='cuda')
    sb = torch.ones((2, 1, k // 128) if block else (2, 128, 1), device='cuda')
    out = torch.full((1, 128), -123, dtype=torch.bfloat16, device='cuda')
    meta = torch.zeros(4, dtype=torch.int32, device='cuda')
    api = dg.m_grouped_int8_gemm_nt_fused if kind == 'int8' else dg.m_grouped_fp8_gemm_nt_fused
    return lambda cfg: api((a, sa), (b, sb), out, meta, meta, meta, meta, cfg), out


@pytest.mark.parametrize('kind', ['int8', 'fp8-channel'])
@pytest.mark.parametrize('k', [192, 272])
def test_bug15_fused_scale_k_tail(kind, k):
    call, out = fused_quantized(kind, k)
    reject(lambda: call((1, 16, 64, 128, 16, 32, 2)), 'scale staging does not support K tails', out)


def test_bug16_w4fa16_expanded_n_tail():
    call, out = w4fa16_call(384, 2, 2)
    reject(call, 'N divisible by block_n', out)


@pytest.mark.parametrize('kind', ['bf16', 'int8', 'fp8-channel', 'fp8-block'])
def test_bug18_contiguous_psum(kind):
    dtype = {'bf16': torch.bfloat16, 'int8': torch.int8}.get(kind, torch.float8_e4m3fn)
    a = torch.ones((256, 256), device='cuda').to(dtype)
    b = torch.ones((2, 128, 256), device='cuda').to(dtype)
    out = torch.full((256, 128), -123, dtype=torch.bfloat16, device='cuda')
    ids = torch.tensor([128, 256], dtype=torch.int32, device='cuda')
    if kind == 'bf16':
        api = dg.m_grouped_bf16_gemm_nt_contiguous
    else:
        block = kind == 'fp8-block'
        a = a, torch.ones((256, 2 if block else 1), device='cuda')
        b = b, torch.ones((2, 1, 2) if block else (2, 128, 1), device='cuda')
        api = dg.m_grouped_int8_gemm_nt_contiguous if kind == 'int8' else dg.m_grouped_fp8_gemm_nt_contiguous
    if kind == 'bf16':
        api(a, b, out, ids, use_psum_layout=True)
        torch.testing.assert_close(out, torch.full_like(out, 256), rtol=0, atol=0)
    else:
        reject(lambda: api(a, b, out, ids, use_psum_layout=True), 'psum layout is unsupported', out)


def test_bug19_fp4_sbo():
    a = torch.full((2, 16, 64), 0x22, dtype=torch.uint8, device='cuda')
    b = torch.full((2, 128, 64), 0x22, dtype=torch.uint8, device='cuda')
    sa = torch.full((2, 16, 4), 127, dtype=torch.uint8, device='cuda')
    sb = torch.full((2, 128, 4), 127, dtype=torch.uint8, device='cuda')
    out = torch.full((2, 16, 128), -123, dtype=torch.bfloat16, device='cuda')
    rows = torch.full((2,), 16, dtype=torch.int32, device='cuda')
    reject(lambda: dg.m_grouped_fp4_gemm_nt_masked((a, sa), (b, sb), None, out, rows, 16,
                                                  enable_sbo_overlap=True),
           'FP4 masked SBO overlap is unsupported', out)


def test_bug20_fp8_expanded_sbo():
    a = torch.ones((2, 16, 256), device='cuda').to(torch.float8_e4m3fn)
    b = torch.ones((2, 4096, 256), device='cuda').to(torch.float8_e4m3fn)
    sa = torch.ones((2, 16, 2), device='cuda')
    sb = torch.ones((2, 32, 2), device='cuda')
    out = torch.full((2, 16, 4096), -123, dtype=torch.bfloat16, device='cuda')
    rows = torch.full((2,), 16, dtype=torch.int32, device='cuda')
    signal = torch.zeros(2, dtype=torch.int32, device='cuda')
    reject(lambda: dg.m_grouped_fp8_gemm_nt_masked((a, sa), (b, sb), out, rows, 16,
                    enable_sbo_overlap=True, signal=signal,
                    configs=(1, 16, 128, 128, 16, 32, 2, (0, 0, 0))),
           'FP8 masked SBO with N expansion is unsupported', out)
    assert torch.count_nonzero(signal).item() == 0


@pytest.mark.parametrize('dtype', [torch.int8, torch.float8_e4m3fn])
def test_bug22_dynamic_masked_n_tail(monkeypatch, dtype):
    monkeypatch.setenv('DG_USE_MOE_DYNAMIC_TILE', '1')
    a = torch.ones((2, 64, 4096), device='cuda').to(dtype)
    b = torch.ones((2, 384, 4096), device='cuda').to(dtype)
    sa = torch.ones((2, 64, 1), device='cuda')
    sb = torch.ones((2, 384, 1), device='cuda')
    rows = torch.tensor([48, 64], dtype=torch.int32, device='cuda')
    out = torch.full((2, 64, 384), -123, dtype=torch.bfloat16, device='cuda')
    api = dg.m_grouped_int8_gemm_nt_masked if dtype == torch.int8 else dg.m_grouped_fp8_gemm_nt_masked
    reject(lambda: api((a, sa), (b, sb), out, rows, 32), 'dynamic A8W8 masked requires N divisible', out)


@pytest.mark.parametrize('width', [4, 5])
def test_bug23_fp4_scale_offset(width):
    sf = torch.full((16 * width + 1,), 127, dtype=torch.uint8, device='cuda')[1:].view(16, width)
    if width == 4:
        with pytest.raises(RuntimeError, match='scale storage offset must be aligned to 2 bytes'):
            dg.preprocess_mxfp4_scales(sf)
    else:
        actual = dg.preprocess_mxfp4_scales(sf)
        torch.testing.assert_close(actual, dg.preprocess_mxfp4_scales(sf.clone()), rtol=0, atol=0)
    assert dg.preprocess_mxfp4_scales(sf.clone()).dtype == torch.uint16


def test_bug24_fp4_bias_view():
    a = torch.full((16, 64), 0x22, dtype=torch.uint8, device='cuda')
    b = torch.full((128, 64), 0x22, dtype=torch.uint8, device='cuda')
    sa = torch.full((16, 4), 127, dtype=torch.uint8, device='cuda')
    sb = torch.full((128, 4), 127, dtype=torch.uint8, device='cuda')
    bias = torch.ones((128, 2), dtype=torch.float32, device='cuda')[:, 0]
    out = torch.full((16, 128), -123, dtype=torch.bfloat16, device='cuda')
    reject(lambda: dg.fp4_gemm_nt((a, sa), (b, sb), bias, out), 'FP4 bias must be contiguous', out)
