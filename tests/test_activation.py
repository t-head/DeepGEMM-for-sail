"""SiLU/SiTU activation and quantization, including PPU1.5 scale layout."""
from functools import partial
import torch
import torch.nn.functional as F
import deep_gemm
from deep_gemm.testing.numeric import calc_diff


def activation_reference(x, limit=0.):
    gate, up = x.float().chunk(2, dim=-1)
    if limit > 0:
        limit = torch.tensor(limit, dtype=torch.bfloat16).float().item()
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    return F.silu(gate) * up


def check(actual, expected, name):
    assert torch.isfinite(actual).all(), name
    diff = calc_diff(actual, expected).item()
    assert diff < .0015, f'{name}: {diff=}'
    print(f'PASS {name}: diff={diff:.8g}')


def test_fp8_channel_activation():
    from quantization import quant_activation
    for n, limit in ((17, 0.), (128, 0.), (384, 2.)):
        x = (torch.randn(2, 3, 2*n, device='cuda') * 4).bfloat16()
        x[0, 0].zero_()
        expected_q, expected_s = quant_activation(activation_reference(x, limit))
        aq, scale = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.FP8_CHANNEL, x, swiglu_limit=limit)
        assert aq.shape == (2, 3, n) and scale.shape == (2, 3, 1)
        assert aq.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
        torch.testing.assert_close(scale, expected_s, rtol=2e-6, atol=1e-12)
        check(aq.float()*scale, expected_q.float()*expected_s, f'activation N={n} limit={limit}')
        folded_q, folded_s = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.FP8_CHANNEL, x, swiglu_limit=limit, weight_inv_scale=0.03125)
        assert torch.equal(aq.view(torch.uint8), folded_q.view(torch.uint8))
        torch.testing.assert_close(folded_s, scale * .03125, rtol=0, atol=0)
    aq, scale = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.FP8_CHANNEL, torch.empty(0, 256, device='cuda', dtype=torch.bfloat16))
    assert aq.shape == (0, 128) and scale.shape == (0, 1)


def test_bf16_activation():
    for n, limit in ((17, 0.), (128, 0.), (384, 2.)):
        x = (torch.randn(2, 3, 2*n, device='cuda') * 4).bfloat16()
        x[0, 0].zero_()
        actual = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.BF16, x, swiglu_limit=limit)
        assert actual.shape == (2, 3, n) and actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, activation_reference(x, limit).bfloat16(), rtol=.008, atol=1e-6)
        check(actual, activation_reference(x, limit).bfloat16(), f'BF16 activation N={n} limit={limit}')
    actual = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.BF16, torch.empty(0, 256, device='cuda', dtype=torch.bfloat16))
    assert actual.shape == (0, 128)


def reference(input, fp4, limit, value=None):
    if value is None:
        gate, up = input.float().chunk(2, -1)
        if limit > 0:
            limit = torch.tensor(limit, dtype=torch.bfloat16).float().item()
            gate = gate.clamp(max=limit)
            up = up.clamp(-limit, limit)
        value = F.silu(gate) * up
    n = value.shape[-1]
    padded = F.pad(value, (0, (-n) % 32)).reshape(*value.shape[:-1], -1, 32)
    maximum = padded.abs().amax(-1).clamp_min(1e-10 if fp4 else 1e-30)
    scale = maximum * (1 / (6 if fp4 else 448))
    bits = scale.contiguous().view(torch.int32)
    if fp4:
        exponent = (((bits + 0x7fffff) & 0x7f800000) >> 23).byte()
    else:
        exponent = ((bits >> 23) + ((bits & 0x7fffff) != 0).int()).clamp(1, 254).byte()
    scale = torch.exp2(exponent.float() - 127)
    normalized = padded / scale[..., None]
    if fp4:
        magnitude = normalized.abs()
        codes = torch.zeros_like(normalized, dtype=torch.uint8)
        for i, boundary in enumerate((.25, .75, 1.25, 1.75, 2.5, 3.5, 5.)):
            codes += ((magnitude > boundary) | ((magnitude == boundary) & bool(i % 2))).byte()
        codes |= torch.signbit(normalized).byte() << 3
        codes = codes.flatten(-2)[..., :n]
        output = codes[..., ::2] | (codes[..., 1::2] << 4)
    else:
        output = normalized.flatten(-2)[..., :n].clamp(-448, 448).to(torch.float8_e4m3fn)
    return output, exponent


def decode(data, scales, fp4):
    if fp4:
        codes = torch.stack((data & 15, data >> 4), -1).flatten(-2).long()
        table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=data.device)
        values = table[codes]
    else:
        values = data.float()
    return values * torch.exp2(scales.float()-127).repeat_interleave(32, -1)[..., :values.shape[-1]]


def test_mx_activation():
    for n in (2, 32, 48, 64, 96, 384, 1024):
        for batched in (False, True):
            shape = (2, 17, 2*n) if batched else (17, 2*n)
            input = (torch.randn(shape, device='cuda') * 4).bfloat16()
            input.reshape(-1, 2*n)[0].zero_()
            for limit in (0., 2.1):
                for fp4 in (True, False):
                    if fp4:
                        data, scales = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.MXFP4, input, swiglu_limit=limit, reorder_scale=False)
                    else:
                        data, scales = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.MXFP8, input, swiglu_limit=limit)
                    ref_data, ref_scales = reference(input, fp4, limit)
                    assert data.shape == (*shape[:-1], n//2 if fp4 else n)
                    assert data.dtype == (torch.uint8 if fp4 else torch.float8_e4m3fn)
                    assert scales.dtype == torch.uint8 and scales.is_contiguous()
                    assert torch.equal(scales, ref_scales), f'scale mismatch {n=} {limit=} {fp4=}'
                    actual, expected = decode(data, scales, fp4), decode(ref_data, ref_scales, fp4)
                    diff = calc_diff(actual, expected).item()
                    assert torch.isfinite(actual).all() and diff < .0015, (n, limit, fp4, diff)
                    if fp4:
                        packed, reordered = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.MXFP4, input, swiglu_limit=limit)
                        assert torch.equal(packed, data)
                        _, layout_ref = deep_gemm.preprocess_weights_and_scales(
                            deep_gemm.PreprocessMode.MXFP4, False, weight_scale=scales)
                        assert reordered.shape == layout_ref.shape
                        assert all(size <= 1 or actual == expected for size, actual, expected in
                                   zip(reordered.shape, reordered.stride(), layout_ref.stride()))
                        assert torch.equal(reordered, layout_ref), f'scale layout mismatch {n=}'
                    print(f'PASS MX activation {n=} {batched=} {limit=} {fp4=}: diff={diff:.8g}')
    for shape in ((0, 256), (2, 0, 96), (0, 17, 64)):
        input = torch.empty(shape, device='cuda', dtype=torch.bfloat16)
        for api in (partial(deep_gemm.silu_and_mul, deep_gemm.ActivationQuantMode.MXFP4), partial(deep_gemm.silu_and_mul, deep_gemm.ActivationQuantMode.MXFP8)):
            data, scales = api(input)
            assert data.numel() == scales.numel() == 0
    print('PASS MX activation empty tensors')

    # Odd MXFP8 row strides require byte stores and a single-element final pair.
    input = torch.randn(2, 7, 34, device='cuda', dtype=torch.bfloat16)
    data, scales = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.MXFP8, input)
    ref_data, ref_scales = reference(input, False, 0.)
    assert torch.equal(scales, ref_scales)
    assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))
    print('PASS MXFP8 odd N pair-store fallback')
    # Positive limits rounding to BF16 zero must still apply clipping.
    input = torch.randn(3, 64, device='cuda', dtype=torch.bfloat16)
    packed, scales = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.MXFP4, input, swiglu_limit=1e-42, reorder_scale=False)
    assert ((packed & 0x77) == 0).all()
    assert torch.equal(scales, reference(input, True, 1e-42)[1])
    print('PASS MXFP4 positive limit rounding to BF16 zero')


def test_fp4_gemm_consumption():
    # Both scale layouts must be directly accepted by the PPU1.5 FP4 GEMM API.
    from deep_gemm.utils.math import per_token_cast_to_fp4, cast_back_from_fp4
    input = torch.randn(17, 256, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(64, 128, device='cuda', dtype=torch.bfloat16)
    packed, scale = per_token_cast_to_fp4(weight, use_ue8m0=True, gran_k=32)
    raw_scale = ((scale.float().view(torch.int32) >> 23) & 255).byte()
    rhs = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.MXFP4, False, packed.view(torch.uint8), raw_scale)
    ref_weight = cast_back_from_fp4(packed, scale, gran_k=32).float()
    outputs = []
    for reorder in (False, True):
        lhs = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.MXFP4, input, reorder_scale=reorder)
        out = torch.empty(17, 64, device='cuda', dtype=torch.bfloat16)
        deep_gemm.gemm_fp4_fp4_bf16_nt(lhs, rhs, None, out)
        raw = reference(input, True, 0.)[1]
        expected = (decode(lhs[0], raw, True) @ ref_weight.T).bfloat16()
        diff = calc_diff(out, expected).item()
        assert diff < .0015, diff
        outputs.append(out)
        print(f'PASS FP4 GEMM consumption {reorder=}: diff={diff:.8g}')
    assert torch.equal(*outputs)


def test_masked_activation():
    counts = torch.tensor([0, 1, 7, 17], device='cuda', dtype=torch.int32)
    for n in (2, 48, 128, 384, 1024):
        x = (torch.randn(4, 17, 2*n, device='cuda') * 4).bfloat16()
        x[2, 0].zero_()
        for limit in (None, 2.1):
            variants = (
                (partial(deep_gemm.silu_and_mul, deep_gemm.ActivationQuantMode.BF16),
                 partial(deep_gemm.silu_and_mul, deep_gemm.ActivationQuantMode.BF16), {}),
                (partial(deep_gemm.silu_and_mul, deep_gemm.ActivationQuantMode.FP8_CHANNEL),
                 partial(deep_gemm.silu_and_mul, deep_gemm.ActivationQuantMode.FP8_CHANNEL), {'weight_inv_scale': .03125}),
                (partial(deep_gemm.silu_and_mul, deep_gemm.ActivationQuantMode.MXFP4),
                 partial(deep_gemm.silu_and_mul, deep_gemm.ActivationQuantMode.MXFP4), {'reorder_scale': False}),
                (partial(deep_gemm.silu_and_mul, deep_gemm.ActivationQuantMode.MXFP4),
                 partial(deep_gemm.silu_and_mul, deep_gemm.ActivationQuantMode.MXFP4), {}),
                (partial(deep_gemm.silu_and_mul, deep_gemm.ActivationQuantMode.MXFP8),
                 partial(deep_gemm.silu_and_mul, deep_gemm.ActivationQuantMode.MXFP8), {}),
            )
            for dense, masked, kwargs in variants:
                expected = dense(x, swiglu_limit=limit, **kwargs)
                # Hint much smaller than valid counts: loops must still cover all rows.
                actual = masked(x, masked_m=counts, swiglu_limit=limit, expected_m=1, **kwargs)
                actual = actual if isinstance(actual, tuple) else (actual,)
                expected = expected if isinstance(expected, tuple) else (expected,)
                if len(actual) == 2 and actual[1].dtype == torch.uint16:
                    assert actual[1].shape == (4, 17, (n+63)//64)
                for e, valid in enumerate((0, 1, 7, 17)):
                    for output, ref in zip(actual, expected):
                        output, ref = output[e, :valid].contiguous(), ref[e, :valid].contiguous()
                        if output.dtype == torch.float8_e4m3fn:
                            output, ref = output.view(torch.uint8), ref.view(torch.uint8)
                        assert torch.equal(output, ref), (n, limit, getattr(masked, "func", masked).__name__)
                # Omitted expected_m and zero hint both preserve valid output rows.
                if n == 48 and limit is None:
                    for hint in (None, 0):
                        other = masked(x, masked_m=counts, expected_m=hint, **kwargs)
                        other = other[0] if isinstance(other, tuple) else other
                        assert torch.equal(other[3].view(torch.uint8), actual[0][3].view(torch.uint8))
                print(f'PASS masked activation N={n} limit={limit} api={getattr(masked, "func", masked).__name__} kwargs={kwargs}')
    # No host count readback is needed for all-empty experts or empty capacities.
    for shape in ((4, 17, 256), (4, 0, 96), (0, 17, 256)):
        x = torch.zeros(shape, device='cuda', dtype=torch.bfloat16)
        empty = torch.zeros(shape[0], device='cuda', dtype=torch.int32)
        for masked in (partial(deep_gemm.silu_and_mul, deep_gemm.ActivationQuantMode.BF16),
                       partial(deep_gemm.silu_and_mul, deep_gemm.ActivationQuantMode.FP8_CHANNEL),
                       partial(deep_gemm.silu_and_mul, deep_gemm.ActivationQuantMode.MXFP4),
                       partial(deep_gemm.silu_and_mul, deep_gemm.ActivationQuantMode.MXFP8)):
            result = masked(x, masked_m=empty, expected_m=0)
            output = result[0] if isinstance(result, tuple) else result
            assert output.shape[:2] == shape[:2]
            if isinstance(result, tuple) and result[1].dtype == torch.uint16:
                assert result[1].shape == (shape[0], shape[1], (shape[-1]//2+63)//64)
    print('PASS masked all-empty and zero-capacity inputs')
    for old in ('silu_and_mul_quant_fp8', 'silu_and_mul_quant_mxfp4',
                'silu_and_mul_quant_mxfp4_ppu15', 'silu_and_mul_quant_mxfp8'):
        assert not hasattr(deep_gemm, old), old
    print('PASS removed old public activation names')


def test_masked_fp4_gemm_consumption():
    from deep_gemm.utils.math import per_token_cast_to_fp4, cast_back_from_fp4
    e, t, n = 3, 17, 128
    x = torch.randn(e, t, 2*n, device='cuda', dtype=torch.bfloat16)
    counts = torch.tensor([0, 7, t], device='cuda', dtype=torch.int32)
    weight = torch.randn(e, 64, n, device='cuda', dtype=torch.bfloat16)
    packed, scales = per_token_cast_to_fp4(weight.flatten(0,1), use_ue8m0=True, gran_k=32)
    raw = ((scales.float().view(torch.int32) >> 23) & 255).byte().reshape(e,64,n//32)
    rhs = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.MXFP4, False, packed.view(torch.uint8).reshape(e,64,n//2), raw)
    ref_weight = cast_back_from_fp4(packed, scales, gran_k=32).reshape(e,64,n).float()
    for reorder in (False, True):
        data, scales = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.MXFP4, x, masked_m=counts, expected_m=1, reorder_scale=reorder)
        lhs = (data, scales)
        out = torch.empty(e,t,64,device='cuda',dtype=torch.bfloat16)
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(lhs, rhs, None, out, counts, 8)
        for expert, valid in enumerate((0,7,t)):
            if not valid: continue
            reference_scale = reference(x[expert,:valid], True, 0.)[1]
            expected = (decode(data[expert,:valid], reference_scale, True) @ ref_weight[expert].T).bfloat16()
            diff = calc_diff(out[expert,:valid], expected).item()
            assert diff < .0015, diff
        print(f'PASS masked FP4 GEMM consumption {reorder=}')


def situ_reference(input, beta, linear_beta):
    gate, up = input.float().chunk(2, -1)
    gated = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return gated * up


def test_situ_activation():
    from quantization import quant_activation
    counts = torch.tensor([0, 1, 7, 17], device='cuda', dtype=torch.int32)
    for n in (2, 48, 128, 384):
        x = (torch.randn(4, 17, 2*n, device='cuda') * 8).bfloat16()
        x[2, 0].zero_()
        x[3, 0, :n] = 100
        x[3, 0, n:] = -100
        for beta, linear_beta in ((4., 25.), (1., None)):
            value = situ_reference(x, beta, linear_beta)
            for suffix, kwargs in (('', {}), ('_post_per_token_quant_fp8', {'weight_inv_scale': .03125}),
                                   ('_post_quant_mxfp4', {'reorder_scale': False}),
                                   ('_post_quant_mxfp4', {}), ('_post_quant_mxfp8', {})):
                quant = {'': deep_gemm.ActivationQuantMode.BF16, '_post_per_token_quant_fp8': deep_gemm.ActivationQuantMode.FP8_CHANNEL, '_post_quant_mxfp4': deep_gemm.ActivationQuantMode.MXFP4, '_post_quant_mxfp8': deep_gemm.ActivationQuantMode.MXFP8}[suffix]
                dense = partial(deep_gemm.situ_and_mul, quant)
                masked = dense
                actual = dense(x, beta=beta, linear_beta=linear_beta, **kwargs)
                if suffix == '':
                    expected = value.bfloat16()
                    diff = calc_diff(actual, expected).item()
                    assert diff < .0015, diff
                elif suffix == '_post_per_token_quant_fp8':
                    q, scales = quant_activation(value, .03125)
                    torch.testing.assert_close(actual[1], scales, rtol=2e-5, atol=1e-12)
                    diff = calc_diff(actual[0].float()*actual[1], q.float()*scales).item()
                    assert diff < .0015, diff
                else:
                    fp4 = suffix == '_post_quant_mxfp4'
                    ref_data, raw = reference(x, fp4, 1., value=value)
                    data, scales = actual
                    if fp4 and kwargs.get('reorder_scale', True):
                        assert scales.shape == (4, 17, (n+63)//64)
                        # Materialize packed M-major pairs to decode logical group32 scales.
                        raw_actual = scales.to(torch.int32)
                        raw_actual = torch.stack((raw_actual & 255, raw_actual >> 8), -1).flatten(-2)[..., :(n+31)//32].byte()
                    else:
                        raw_actual = scales
                    assert torch.equal(raw_actual, raw), (n, beta, suffix)
                    diff = calc_diff(decode(data, raw_actual, fp4), decode(ref_data, raw, fp4)).item()
                    assert diff < .0015, diff
                valid_output = masked(x, masked_m=counts, beta=beta, linear_beta=linear_beta, expected_m=1, **kwargs)
                valid_output = valid_output if isinstance(valid_output, tuple) else (valid_output,)
                all_output = actual if isinstance(actual, tuple) else (actual,)
                for e, valid in enumerate((0, 1, 7, 17)):
                    for output, ref in zip(valid_output, all_output):
                        output, ref = output[e, :valid], ref[e, :valid]
                        if output.dtype == torch.float8_e4m3fn:
                            output, ref = output.view(torch.uint8), ref.view(torch.uint8)
                        assert torch.equal(output, ref), (n, beta, suffix)
                # Exercise the rank-2 dense wrapper as well.
                flat = dense(x.flatten(0,1), beta=beta, linear_beta=linear_beta, **kwargs)
                flat = flat if isinstance(flat, tuple) else (flat,)
                for output, ref in zip(flat, all_output):
                    output, ref = output.reshape(ref.shape), ref
                    if output.dtype == torch.float8_e4m3fn:
                        output, ref = output.view(torch.uint8), ref.view(torch.uint8)
                    assert torch.equal(output, ref)
                print(f'PASS SiTU N={n} beta={beta} linear_beta={linear_beta} api={getattr(dense, "func", dense).__name__} kwargs={kwargs}')
    for shape in ((0, 256), (4, 0, 256), (0, 17, 256)):
        x = torch.empty(shape, device='cuda', dtype=torch.bfloat16)
        if len(shape) == 2:
            result = deep_gemm.situ_and_mul(deep_gemm.ActivationQuantMode.BF16, x, beta=4.0, linear_beta=25.0)
        else:
            result = deep_gemm.situ_and_mul(deep_gemm.ActivationQuantMode.BF16, x, masked_m=torch.zeros(shape[0], device='cuda', dtype=torch.int32), beta=4.0)
        assert result.shape == (*shape[:-1], 128)
    print('PASS SiTU empty inputs')



def test_mxfp4_pipeline():
    # Exercise the 256-column boundary, partial groups/bytes, and ping-pong reuse.
    counts = torch.tensor([0, 3, 9], device='cuda', dtype=torch.int32)
    for n in (10, 34, 254, 256, 258, 512, 2048):
        x = (torch.randn(3, 9, 2*n, device='cuda') * 4).bfloat16()
        for situ in (False, True):
            dense = partial(deep_gemm.situ_and_mul, deep_gemm.ActivationQuantMode.MXFP4) if situ else partial(deep_gemm.silu_and_mul, deep_gemm.ActivationQuantMode.MXFP4)
            masked = partial(deep_gemm.situ_and_mul, deep_gemm.ActivationQuantMode.MXFP4) if situ else partial(deep_gemm.silu_and_mul, deep_gemm.ActivationQuantMode.MXFP4)
            params = {'beta': 4., 'linear_beta': 25.} if situ else {'swiglu_limit': 2.1}
            data, raw = dense(x, reorder_scale=False, **params)
            if situ:
                value = situ_reference(x, 4., 25.)
                ref_data, ref_raw = reference(x, True, 1., value=value)
            else:
                ref_data, ref_raw = reference(x, True, 2.1)
            assert torch.equal(raw, ref_raw), (n, situ, 'scale')
            check(decode(data, raw, True), decode(ref_data, ref_raw, True), f'pipeline {n=} {situ=}')
            packed, reordered = dense(x, **params)
            assert torch.equal(data, packed)
            _, layout_ref = deep_gemm.preprocess_weights_and_scales(
                deep_gemm.PreprocessMode.MXFP4, False, weight_scale=raw)
            assert torch.equal(reordered, layout_ref)
            for reorder in (False, True):
                aq, sf = masked(x, masked_m=counts, expected_m=1, reorder_scale=reorder, **params)
                expected_sf = reordered if reorder else raw
                for e, valid in enumerate((0, 3, 9)):
                    assert torch.equal(aq[e, :valid], data[e, :valid]), (n, situ, reorder, 'codes')
                    assert torch.equal(sf[e, :valid], expected_sf[e, :valid]), (n, situ, reorder, 'masked scales')
        if n == 256:
            # Contiguous storage with a two-byte offset must select scalar loads.
            storage = torch.empty(x.numel()+1, device='cuda', dtype=torch.bfloat16)
            view = storage[1:].view_as(x)
            view.copy_(x)
            assert view.is_contiguous() and view.data_ptr() % 16 != 0
            for limit in (None, 2.1):
                aligned = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.MXFP4, x, swiglu_limit=limit)
                unaligned = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.MXFP4, view, swiglu_limit=limit)
                assert all(torch.equal(a,b) for a,b in zip(aligned,unaligned))
        # Natural scales also support more than one leading batch dimension.
        rank4 = x.reshape(1,3,9,2*n)
        actual = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.MXFP4, rank4, reorder_scale=False)
        expected = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.MXFP4, x, reorder_scale=False)
        assert all(torch.equal(a.reshape(b.shape), b) for a,b in zip(actual,expected))
    print('PASS MXFP4 pipeline tails, scale layouts, unaligned inputs and masked reuse')


def test_mxfp8_pipeline():
    counts = torch.tensor([0, 3, 9], device='cuda', dtype=torch.int32)
    for n in (17, 255, 256, 257, 768):
        x = (torch.randn(3, 9, 2*n, device='cuda') * 4).bfloat16()
        for situ, limit in ((False, None), (False, 2.1), (True, None)):
            fn = deep_gemm.situ_and_mul if situ else deep_gemm.silu_and_mul
            params = {'beta': 4., 'linear_beta': 25.} if situ else {'swiglu_limit': limit}
            data, scales = fn(deep_gemm.ActivationQuantMode.MXFP8, x, **params)
            value = situ_reference(x, 4., 25.) if situ else activation_reference(x, limit or 0.)
            ref_data, ref_scales = reference(x, False, 0., value=value)
            assert torch.equal(scales, ref_scales)
            check(decode(data, scales, False), decode(ref_data, ref_scales, False),
                  f'MXFP8 pipeline {n=} {situ=} {limit=}')
            aq, sf = fn(deep_gemm.ActivationQuantMode.MXFP8, x, masked_m=counts, expected_m=1, **params)
            for e, valid in enumerate((0, 3, 9)):
                assert torch.equal(aq[e, :valid].view(torch.uint8), data[e, :valid].view(torch.uint8))
                assert torch.equal(sf[e, :valid], scales[e, :valid])
            if n == 256:
                storage = torch.empty(x.numel()+1, device='cuda', dtype=torch.bfloat16)
                view = storage[1:].view_as(x)
                view.copy_(x)
                uq, us = fn(deep_gemm.ActivationQuantMode.MXFP8, view, **params)
                assert torch.equal(uq.view(torch.uint8), data.view(torch.uint8))
                assert torch.equal(us, scales)
    print('PASS MXFP8 pipeline tails, odd row strides, unaligned inputs and masked reuse')


def test_fp8_channel_vec8():
    from quantization import quant_activation
    counts = torch.tensor([0, 3, 9], device='cuda', dtype=torch.int32)
    for n in (17, 255, 257, 768, 1280, 4096, 8448):
        # Offset by one BF16 element to exercise the unaligned vector-load fallback.
        storage = (torch.randn(3 * 9 * 2 * n + 1, device='cuda') * 4).bfloat16()
        x = storage[1:].view(3, 9, 2 * n)
        x[1, 0].zero_()
        x[2, 0, n - 1] = 20
        x[2, 0, 2 * n - 1] = 32
        for situ in (False, True):
            fn = deep_gemm.situ_and_mul if situ else deep_gemm.silu_and_mul
            params = {'beta': 4., 'linear_beta': 25.} if situ else {'swiglu_limit': 2.1}
            value = situ_reference(x, 4., 25.) if situ else activation_reference(x, 2.1)
            ref_q, ref_s = quant_activation(value, .03125)
            q, scale = fn(deep_gemm.ActivationQuantMode.FP8_CHANNEL, x, weight_inv_scale=0.03125, **params)
            mq, ms = fn(deep_gemm.ActivationQuantMode.FP8_CHANNEL, x, weight_inv_scale=0.03125, masked_m=counts, expected_m=1, **params)
            torch.testing.assert_close(scale, ref_s, rtol=2e-6, atol=1e-12)
            check(q.float() * scale, ref_q.float() * ref_s, f'channel vec8 N={n} situ={situ}')
            for e, valid in enumerate((0, 3, 9)):
                assert torch.equal(q[e, :valid].view(torch.uint8), mq[e, :valid].view(torch.uint8))
                assert torch.equal(scale[e, :valid], ms[e, :valid])
    print('PASS channel vector tails, unaligned input, wide rows and masked loops')


def test_fp8_block_activation():
    mode = deep_gemm.ActivationQuantMode.FP8_BLOCK
    counts = torch.tensor([0, 3, 9], device='cuda', dtype=torch.int32)
    for n in (128, 256, 384, 768, 2048):
        storage = (torch.randn(3 * 9 * 2 * n + 1, device='cuda') * 4).bfloat16()
        x = storage[1:].view(3, 9, 2 * n)
        x[1, 0].zero_()
        # Different group amplitudes expose scale or group/row permutations.
        for g in range(n // 128):
            x[..., n + g * 128:n + (g + 1) * 128] *= 2. ** (g % 5 - 2)
        for situ in (False, True):
            fn = deep_gemm.situ_and_mul if situ else deep_gemm.silu_and_mul
            for limit in ((0.,) if situ else (0., 1.003, 2.1)):
                params = {'beta': 4., 'linear_beta': 25.} if situ else {'swiglu_limit': limit}
                value = situ_reference(x, 4., 25.) if situ else activation_reference(x, limit)
                grouped = value.reshape(3, 9, n // 128, 128)
                amax = grouped.abs().amax(-1).clamp_min(1e-10)
                ref_s = amax * (1. / 448.)
                ref_q = (grouped * (448. / amax).unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(3, 9, n)
                q, scale = fn(mode, x, **params)
                aligned_q, aligned_scale = fn(mode, x.clone(), **params)
                assert torch.equal(q.view(torch.uint8), aligned_q.view(torch.uint8))
                assert torch.equal(scale, aligned_scale)
                bf16 = fn(deep_gemm.ActivationQuantMode.BF16, x, **params)
                assert torch.equal(bf16, fn(deep_gemm.ActivationQuantMode.BF16, x.clone(), **params))
                check(bf16.float(), value, f'BF16 vec8 N={n} situ={situ} limit={limit}')
                assert q.shape == (3, 9, n) and scale.shape == (3, 9, n // 128)
                assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
                assert scale.is_contiguous()
                torch.testing.assert_close(scale, ref_s, rtol=2e-6, atol=1e-12)
                actual = q.float().reshape_as(grouped) * scale.unsqueeze(-1)
                expected = ref_q.float().reshape_as(grouped) * ref_s.unsqueeze(-1)
                check(actual, expected, f'FP8 block N={n} situ={situ} limit={limit}')
                mq, ms = fn(mode, x, masked_m=counts, expected_m=1, **params)
                assert ms.shape == scale.shape and ms.stride() == (n // 128 * 12, 1, 12)
                for e, valid in enumerate((0, 3, 9)):
                    assert torch.equal(q[e, :valid].view(torch.uint8), mq[e, :valid].view(torch.uint8))
                    assert torch.equal(scale[e, :valid], ms[e, :valid])
                    assert torch.count_nonzero(ms[e, valid:]).item() == 0
                dq, ds = fn(mode, x[2], **params)
                assert torch.equal(dq.view(torch.uint8), q[2].view(torch.uint8))
                assert torch.equal(ds, scale[2])
    for shape in ((3, 0, 256), (0, 9, 256)):
        x = torch.empty(shape, dtype=torch.bfloat16, device='cuda')
        for masked in (False, True):
            opts = {'masked_m': torch.zeros(shape[0], dtype=torch.int32, device='cuda')} if masked else {}
            q, scale = deep_gemm.silu_and_mul(mode, x, **opts)
            assert q.shape == (*shape[:-1], 128) and scale.shape == (*shape[:-1], 1)
    for n, opts in ((129, {}), (128, {'weight_inv_scale': .5}), (128, {'reorder_scale': False})):
        try:
            deep_gemm.silu_and_mul(mode, torch.zeros(2, 2 * n, device='cuda', dtype=torch.bfloat16), **opts)
        except RuntimeError:
            pass
        else:
            raise AssertionError((n, opts))
    print('PASS FP8 block scale layout, masked loops, empty shapes and rejection paths')


def test_unified_api_contract():
    x = torch.zeros(2, 128, device='cuda', dtype=torch.bfloat16)
    for api, base in ((deep_gemm.silu_and_mul, {}), (deep_gemm.situ_and_mul, {'beta': 4.})):
        for mode, bad in ((deep_gemm.ActivationQuantMode.BF16, {'expected_m': 1}),
                          (deep_gemm.ActivationQuantMode.BF16, {'weight_inv_scale': 2.}),
                          (deep_gemm.ActivationQuantMode.MXFP8, {'weight_inv_scale': 2.}),
                          (deep_gemm.ActivationQuantMode.FP8_CHANNEL, {'reorder_scale': False}),
                          (deep_gemm.ActivationQuantMode.BF16, {'reorder_scale': True}),
                          ('fp8', {}), (None, {})):
            try:
                api(mode, x, **base, **bad)
            except (RuntimeError, TypeError):
                pass
            else:
                raise AssertionError(f'accepted invalid activation options: {mode}, {bad}')
        for args, kwargs in (((x,), base), ((x,), {**base, 'quant': deep_gemm.ActivationQuantMode.MXFP8}),
                             ((), {'input': x, **base})):
            try:
                api(*args, **kwargs)
            except TypeError:
                pass
            else:
                raise AssertionError('mode must be explicitly supplied')
        expected = api(deep_gemm.ActivationQuantMode.BF16, x, **base)
        actual = api(mode=deep_gemm.ActivationQuantMode.BF16, input=x, **base)
        assert torch.equal(actual, expected)
    for prefix in ('silu_and_mul', 'situ_and_mul'):
        for suffix in ('_masked', '_post_per_token_quant_fp8', '_post_quant_mxfp4',
                       '_post_quant_mxfp8', '_masked_post_per_token_quant_fp8',
                       '_masked_post_quant_mxfp4', '_masked_post_quant_mxfp8'):
            assert not hasattr(deep_gemm, prefix + suffix)
            assert not hasattr(deep_gemm.deep_gemm_cpp, prefix + suffix)
    print('PASS unified activation API validation and removed bindings')


def test_bf16_clamp_boundary():
    # Only the clamp boundary rounds to BF16: 1.003 becomes 1.0.
    x = torch.full((1, 256), 2., device='cuda', dtype=torch.bfloat16)
    x[0, 128] = .1279296875
    data, scales = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.MXFP4, x, swiglu_limit=1.003, reorder_scale=False)
    expected, expected_scales = reference(x, True, 1.003)
    assert torch.equal(data, expected) and torch.equal(scales, expected_scales)
    assert (data[0, 0] & 15).item() == 1
    for mode in (deep_gemm.ActivationQuantMode.BF16, deep_gemm.ActivationQuantMode.FP8_CHANNEL,
                 deep_gemm.ActivationQuantMode.MXFP4, deep_gemm.ActivationQuantMode.MXFP8):
        actual = deep_gemm.silu_and_mul(mode, x, swiglu_limit=1.003)
        rounded = deep_gemm.silu_and_mul(mode, x, swiglu_limit=1.0)
        if mode == deep_gemm.ActivationQuantMode.BF16:
            assert torch.equal(actual, rounded)
        else:
            for a, b in zip(actual, rounded):
                if a.dtype == torch.float8_e4m3fn:
                    a, b = a.view(torch.uint8), b.view(torch.uint8)
                assert torch.equal(a, b)
    print('PASS BF16 clamp boundary with FP32 activation intermediates')


def test_mxfp4_fp32_rounding():
    # SiLU(1.5)*0.611328125 is ~0.749709: FP32 rounds to FP4 0.5,
    # whereas BF16 intermediate rounding hits the 0.75 tie and produces 1.0.
    x = torch.empty(1, 256, device='cuda', dtype=torch.bfloat16)
    x[:, :128] = 32
    x[:, 128:] = .1875  # Other elements are exactly 6, fixing E8M0 scale to 1.
    x[0, 0] = 1.5
    x[0, 128] = .611328125
    data, scales = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.MXFP4, x, reorder_scale=False)
    assert (scales == 127).all()
    assert (data[0, 0] & 15).item() == 1
    ref_data, ref_scales = reference(x, True, 0.)
    assert torch.equal(data, ref_data) and torch.equal(scales, ref_scales)
    print('PASS MXFP4 FP32 intermediates near the 0.75 rounding boundary')


def test_scale_arithmetic():
    # Large positive gates make sigmoid exactly 1, isolating scale arithmetic.
    amplitudes = torch.tensor([0., 2. ** -60, 2. ** -16, 1., 3., 2. ** 50], device='cuda')
    pattern = torch.tensor([0., .25, .5, 1., -.25, -.5, -1., 1.], device='cuda').repeat(16)
    x = torch.empty(len(amplitudes), 256, device='cuda', dtype=torch.bfloat16)
    x[:, :128] = 32
    x[:, 128:] = amplitudes[:, None] * pattern
    value = x[:, 128:].float() * 32
    amax = value.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    expected_scale = amax * (1. / 448.)
    expected_q = (value * (448. / amax)).clamp(-448, 448).to(torch.float8_e4m3fn)
    q, scale = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.FP8_CHANNEL, x)
    assert torch.equal(scale, expected_scale)
    assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
    folded_q, folded_scale = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.FP8_CHANNEL, x, weight_inv_scale=0.125)
    assert torch.equal(q.view(torch.uint8), folded_q.view(torch.uint8))
    assert torch.equal(folded_scale, expected_scale * .125)
    # Overflowing BF16 products produce E8M0 exponent 255 rather than clamping
    # to 254. This checks the scale encoding, not accuracy of nonfinite data.
    overflow = torch.full((1, 256), 2. ** 70, device='cuda', dtype=torch.bfloat16)
    _, scales = deep_gemm.silu_and_mul(deep_gemm.ActivationQuantMode.MXFP4, overflow, reorder_scale=False)
    assert (scales == 255).all()
    print('PASS FP8 scale/inverse arithmetic, zero/tiny rows, and unclamped MXFP4 exponent')


def test_launch_parameter_reuse():
    # Reuse JIT specializations with changing pointers and scalar/mask parameters.
    # This catches stale fields or mismatched host/device argument layouts.
    from quantization import quant_activation
    counts = torch.tensor([0, 2, 5], device='cuda', dtype=torch.int32)
    for step in range(4):
        x = (torch.randn(3, 5, 512, device='cuda') * (step + 1)).bfloat16()
        inverse = 2. ** (step - 2)
        for situ in (False, True):
            params = {'beta': float(step + 1), 'linear_beta': float(step + 3)} if situ else {'swiglu_limit': float(step)}
            fn = deep_gemm.situ_and_mul if situ else deep_gemm.silu_and_mul
            value = situ_reference(x, params['beta'], params['linear_beta']) if situ else activation_reference(x, params['swiglu_limit'])
            expected_q, expected_scale = quant_activation(value, inverse)
            for masked in (False, True):
                mask = {'masked_m': counts, 'expected_m': step} if masked else {}
                q, scale = fn(deep_gemm.ActivationQuantMode.FP8_CHANNEL, x, weight_inv_scale=inverse, **params, **mask)
                for e, valid in enumerate((0, 2, 5) if masked else (5, 5, 5)):
                    if valid:
                        check(q[e, :valid].float() * scale[e, :valid],
                              expected_q[e, :valid].float() * expected_scale[e, :valid],
                              f'launch parameters step={step} situ={situ} masked={masked}')
    print('PASS activation parameter reuse across JIT launches')


if __name__ == '__main__':
    torch.manual_seed(13)
    test_unified_api_contract()
    test_scale_arithmetic()
    test_mxfp4_fp32_rounding()
    test_bf16_clamp_boundary()
    test_launch_parameter_reuse()
    test_bf16_activation()
    test_fp8_channel_activation()
    test_fp8_channel_vec8()
    test_fp8_block_activation()
    test_situ_activation()
    test_mx_activation()
    test_mxfp4_pipeline()
    test_mxfp8_pipeline()
    test_fp4_gemm_consumption()
    test_masked_activation()
    test_masked_fp4_gemm_consumption()
