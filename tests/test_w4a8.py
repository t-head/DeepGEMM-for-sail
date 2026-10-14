"""PPU1.5 W4FA8 functional tests. Run: python tests/test_w4a8.py"""
import random

import torch
import deep_gemm
from utils import is_ppu1v5_device
from deep_gemm.testing.numeric import calc_diff
from quantization import decode_weight, quant_activation
import utils


def test_grouped_gemm():
    cases = [
        'GroupedNoPad,data_type:w4fa8_mma,groups:4,m:35,n:64,k:128,quant_type:group,group_size:32',
        'GroupedNoPad,data_type:w4fa8_mma,groups:128,m:7,n:192,k:256,quant_type:group,group_size:32',
        'GroupedMasked,data_type:w4fa8_mma,groups:4,m:67,n:192,k:256,distribution:[0,1,33,67],quant_type:group,group_size:32',
        'GroupedFused,data_type:w4fa8_mma,groups:8,num_token:17,topk:3,n:192,k:256,quant_type:group,group_size:32',
        'GroupedFused,data_type:w4fa8_mma,groups:128,num_token:4,topk:2,n:64,k:2048,quant_type:group,group_size:32',
    ]
    runners = {
        'GroupedNoPad': utils.test_m_grouped_gemm_nopad,
        'GroupedMasked': utils.test_m_grouped_gemm_masked,
        'GroupedFused': utils.test_m_grouped_gemm_fused,
    }
    utils.set_acc_check(True)
    utils.set_benchmark(False)
    for case in cases:
        print(case)
        args = utils.parse_deepgemm_string_re(case)
        runners[args['gemm_type']](args)


def test_nopad_expert_limit():
    # The count-preprocessing kernel has storage for at most 32 warps.
    for groups in (1024, 1025):
        packed = torch.zeros(groups, 64, 64, device='cuda', dtype=torch.uint8)
        raw = torch.full((groups, 64, 4), 127, device='cuda', dtype=torch.uint8)
        rhs = deep_gemm.preprocess_weights_and_scales(
            deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
        lhs = (torch.zeros(1, 128, device='cuda', dtype=torch.float8_e4m3fn),
               torch.ones(1, 1, device='cuda'))
        rows = torch.zeros(groups, device='cuda', dtype=torch.int32)
        rows[-1] = 1
        indices = torch.full((1,), groups - 1, device='cuda', dtype=torch.int32)
        out = torch.full((1, 64), float('nan'), device='cuda', dtype=torch.bfloat16)
        if groups == 1024:
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(lhs, rhs[:2], out, indices, rows)
            assert torch.equal(out, torch.zeros_like(out))
        else:
            try:
                deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(lhs, rhs[:2], out, indices, rows)
            except RuntimeError as error:
                assert 'at most 1024 experts' in str(error)
            else:
                raise AssertionError('accepted 1025 nopad experts')
    print('PASS nopad 1024 experts and rejection of 1025 experts')


def test_nopad_n_expand():
    # Short-K pipeline boundaries, three-way expansion, N tails and long-K fallback.
    for groups, n, k, expand in [(4, 512, 128, 2), (4, 768, 128, 3),
                                  (128, 1024, 128, 4), (4, 768, 512, 3),
                                  (4, 1024, 512, 4), (4, 320, 128, 1),
                                  (4, 1024, 768, 1)]:
        counts = [0, 1, 17, 65] + [0] * (groups - 4)
        rows = torch.tensor(counts, device='cuda', dtype=torch.int32)
        indices = torch.repeat_interleave(torch.arange(groups, device='cuda', dtype=torch.int32), rows.long())
        m = sum(counts)
        lhs = quant_activation(torch.randn(m, k, device='cuda', dtype=torch.bfloat16))
        packed = torch.randint(256, (groups, n, k // 2), device='cuda', dtype=torch.uint8)
        raw = torch.randint(125, 130, (groups, n, k // 32), device='cuda', dtype=torch.uint8)
        rhs = deep_gemm.preprocess_weights_and_scales(
            deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
        gemm_lhs = (lhs[0], lhs[1] * rhs[2])
        bm = 16
        while bm < (m + groups - 1) // groups and bm < 64:
            bm *= 2
        config = (deep_gemm.get_num_sms(), bm, 256, 128, bm, 64, 128, 2, 1)
        baseline = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
        out = torch.empty_like(baseline)
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(gemm_lhs, rhs[:2], baseline, indices, rows, config)
        offset = 0
        for expert, count in enumerate(counts):
            if count:
                a = lhs[0][offset:offset+count].float() * lhs[1][offset:offset+count]
                ref = (a @ decode_weight(packed[expert], raw[expert]).T).bfloat16()
                assert torch.isfinite(baseline[offset:offset+count]).all()
                assert calc_diff(baseline[offset:offset+count], ref) < .0015
            offset += count
        for selected in (None, config[:-1] + (expand,)):
            out.fill_(float('nan'))
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(gemm_lhs, rhs[:2], out, indices, rows, selected)
            assert torch.equal(out, baseline), (groups, n, k, expand)
        # Explicit expansion remains valid above the automatic short-K threshold.
        if k == 768:
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(gemm_lhs, rhs[:2], out, indices, rows, config[:-1] + (4,))
            assert torch.equal(out, baseline)
        invalid = [config[:-1] + (5,), config[:7] + (3, 2), config[:6] + (64, 2, 2)]
        if n == 320:
            invalid.append(config[:-1] + (2,))
        for bad in invalid:
            try:
                deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(gemm_lhs, rhs[:2], out, indices, rows, bad)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f'accepted invalid N_EXPAND config {bad}')
        print(f'PASS N_EXPAND {groups=} {n=} {k=} {expand=}')


def test_nopad_resident_a_and_scales():
    # N/K-varying weights and all legal scales expose reuse of the wrong shared
    # stage or output-row scale across successive N tiles and scheduler items.
    groups, n = 4, 768
    counts = [0, 1, 65, 197]
    m = sum(counts)
    rows = torch.tensor(counts, device='cuda', dtype=torch.int32)
    indices = torch.repeat_interleave(torch.arange(groups, device='cuda', dtype=torch.int32), rows.long())
    for k in (128, 256, 384, 512):
        codes = ((torch.arange(n, device='cuda')[:, None]*3 + torch.arange(k, device='cuda')[None, :]*5) % 16).byte()
        packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).unsqueeze(0).expand(groups, n, k//2).contiguous()
        raw = (114 + (torch.arange(groups, device='cuda')[:, None, None] +
                      torch.arange(n, device='cuda')[None, :, None] +
                      torch.arange(k//32, device='cuda')[None, None, :]) % 27).byte().contiguous()
        rhs = deep_gemm.preprocess_weights_and_scales(
            deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
        col = (torch.arange(m, device='cuda')*7) % k
        a = torch.zeros(m, k, device='cuda')
        a[torch.arange(m, device='cuda'), col] = 1
        scales = torch.linspace(.5, 1.5, m, device='cuda').reshape(-1, 1)
        lhs = (a.to(torch.float8_e4m3fn), scales)
        gemm_lhs = (lhs[0], lhs[1] * rhs[2])
        configs = [None]
        if k == 256:
            configs.append((deep_gemm.get_num_sms(), 128, 256, 128, 64, 64, 128, 2, 3))
        for config in configs:
            out = torch.full((m, n), float('nan'), device='cuda', dtype=torch.bfloat16)
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(gemm_lhs, rhs[:2], out, indices, rows, config)
            start = 0
            for expert, count in enumerate(counts):
                if count:
                    expected = (decode_weight(packed[expert], raw[expert])[:, col[start:start+count]].T *
                                scales[start:start+count]).bfloat16()
                    assert torch.equal(out[start:start+count], expected), (k, expert, config)
                start += count
        print(f'PASS resident A/scale basis {k=}')


def test_fused_nopad_equivalence():
    case = 'GroupedFused,data_type:w4fa8_mma,groups:8,num_token:17,topk:3,n:192,k:256,quant_type:group,group_size:32'
    print(f'Fused/nopad equivalence: {case}')
    args = utils.parse_deepgemm_string_re(case)
    lhs, rhs, routes, out, _ = utils.construct_non_permute_grouped(
        args['groups'], args['num_token'], args['k'], args['n'], args['topk'],
        args['data_type'], args['quant_type'], args['group_size'])
    config, rows, experts, tokens, aligned, _, _ = deep_gemm.moe_align_block_size(lhs[0], rhs[0], routes)
    deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused(lhs, rhs, out, rows, experts, tokens, aligned, config)
    flat_experts = routes.flatten()
    order = flat_experts.argsort(stable=True)
    source = torch.arange(routes.size(0), device=routes.device).repeat_interleave(routes.size(1))[order]
    nopad_lhs = (lhs[0].float()[source].to(lhs[0].dtype), lhs[1][source].contiguous())
    nopad_out = torch.empty_like(out)
    deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(nopad_lhs, rhs, nopad_out, flat_experts[order], rows)
    diff = calc_diff(out, nopad_out)
    assert torch.isfinite(out).all() and diff < .0015, f'fused/nopad {diff=}'
    print('PASS fused/nopad equivalence')


def test_boundaries():
    packed = torch.zeros(1, 64, 64, device='cuda', dtype=torch.uint8)
    scale = torch.full((1, 64, 4), 127, device='cuda', dtype=torch.uint8)
    for low, high in ((0, 127), (127, 255), (113, 141), (1, 254)):
        scale.fill_(high)
        scale[0, 0, 0] = low
        try:
            deep_gemm.preprocess_weights_and_scales(deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, None, scale)
        except RuntimeError as error:
            assert f'[{low},{high}]' in str(error) and '[114,140]' in str(error)
        else:
            raise AssertionError(f'accepted invalid scale range [{low},{high}]')
    for original, adjusted, inverse in ((113, 114, .5), (114, 114, 1.),
                                        (140, 140, 1.), (141, 140, 2.),
                                        (254, 140, float(2 ** 114))):
        scale.fill_(original)
        result = deep_gemm.preprocess_weights_and_scales(deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, scale)
        assert result[1].shape == (1, 1, 256)
        assert (result[1] == adjusted).all() and result[2] == inverse
    scale.fill_(127)
    rhs = deep_gemm.preprocess_weights_and_scales(deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, scale)
    a = (torch.empty(0, 128, device='cuda', dtype=torch.float8_e4m3fn), torch.empty(0, 1, device='cuda'))
    deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(a, rhs[:2], torch.empty(0, 64, device='cuda', dtype=torch.bfloat16),
                                      torch.empty(0, device='cuda', dtype=torch.int32))
    # Catch dtype/scale shape mistakes at the public boundary, before any launch.
    a = (torch.zeros(1, 128, device='cuda', dtype=torch.float8_e4m3fn), torch.ones(1, 2, device='cuda'))
    try:
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(a, rhs[:2], torch.empty(1, 64, device='cuda', dtype=torch.bfloat16),
                                          torch.zeros(1, device='cuda', dtype=torch.int32))
    except RuntimeError:
        pass
    else:
        raise AssertionError('accepted invalid activation scale shape')
    # Scale-only preparation derives alignment from [E,N,K/32] before reducing exponents.
    for bad in (None, scale.float(), scale[0], scale[:, :63], scale[..., :3],
                scale[:0], scale[:, :0], scale[..., :0]):
        try:
            deep_gemm.preprocess_weights_and_scales(
                deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, weight_scale=bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError('accepted invalid scale-only W4A8 input')
    for mode, fuse in ((deep_gemm.PreprocessMode.MXFP4, True),
                       (deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, True),
                       (deep_gemm.PreprocessMode.W4A16_INT4, False),
                       (deep_gemm.PreprocessMode.W4A16_MXFP4, False),
                       (deep_gemm.PreprocessMode.W4A16_MXFP4_MMA, False)):
        try:
            deep_gemm.preprocess_weights_and_scales(mode, fuse, weight_scale=scale)
        except RuntimeError as error:
            assert 'requires weights' in str(error)
        else:
            raise AssertionError('accepted missing required weights')
    print('PASS preprocessing boundaries, empty input, invalid scale shape')


def test_tensor_scale_shift():
    # Folding the inverse must not change the activation codes.
    x = torch.randn(7, 128, device='cuda', dtype=torch.bfloat16)
    aq, a_scale = quant_activation(x)
    for inverse in (.03125, 32.):
        folded_aq, folded_scale = quant_activation(x, inverse)
        assert torch.equal(aq.view(torch.uint8), folded_aq.view(torch.uint8))
        torch.testing.assert_close(folded_scale, a_scale * inverse, rtol=0, atol=0)
    # The offset is shared across experts; compare the logical N/K scale order.
    e, n, k = 2, 128, 256
    packed = torch.randint(256, (e, n, k // 2), device='cuda', dtype=torch.uint8)
    for low, high, shift in ((109, 124, 5), (111, 127, 3),
                             (119, 145, -5), (114, 140, 0)):
        raw = torch.full((e, n, k // 32), high, device='cuda', dtype=torch.uint8)
        raw[0, 0, 0] = low
        weight, adjusted, inverse = deep_gemm.preprocess_weights_and_scales(
            deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
        assert torch.equal(weight, packed)
        no_weight, scale_only, inverse_only = deep_gemm.preprocess_weights_and_scales(
            deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, None, raw)
        assert no_weight is None and inverse_only == inverse
        assert torch.equal(scale_only, adjusted) and scale_only.stride() == adjusted.stride()
        no_weight, scale_only, inverse_only = deep_gemm.preprocess_weights_and_scales(
            deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, weight_scale=raw)
        assert no_weight is None and inverse_only == inverse and torch.equal(scale_only, adjusted)
        assert inverse == float(2. ** -shift)
        assert adjusted.min().item() == low + shift
        assert adjusted.max().item() == high + shift
        expected = torch.empty(e, n // 64, k * 2, dtype=torch.uint8)
        host = raw.cpu()
        for expert in range(e):
            for tile in range(n // 64):
                for kg in range(k // 64):
                    for i in range(8):
                        for j in range(8):
                            for half in range(2):
                                index = kg * 128 + i * 16 + j * 2 + half
                                expected[expert, tile, index] = host[
                                    expert, tile * 64 + i + 8 * j, kg * 2 + half] + shift
        assert torch.equal(adjusted.cpu(), expected)
        decoded = decode_weight(packed, raw)
        adjusted_natural = (raw.short() + shift).byte()
        torch.testing.assert_close(decode_weight(packed, adjusted_natural) * inverse, decoded, rtol=0, atol=0)
    print('PASS batch-wide exponent shift, inverse scale, and layout')


def test_preprocessing_layout():
    # Explicit coordinate mapping, independent of the implementation's reshape/index_select.
    e, n, k = 2, 128, 256
    packed = torch.randint(256, (e, n, k//2), device='cuda', dtype=torch.uint8)
    raw = (torch.arange(e*n*k//32, device='cuda').reshape(e, n, k//32) % 27 + 114).byte()
    weight, actual, inverse = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
    assert inverse == 1.0
    expected = torch.empty(e, n//64, k*2, dtype=torch.uint8)
    host = raw.cpu()
    for expert in range(e):
        for tile in range(n//64):
            for kg in range(k//64):
                for i in range(8):
                    for j in range(8):
                        for half in range(2):
                            expected[expert, tile, kg*128+i*16+j*2+half] = host[expert, tile*64+i+8*j, kg*2+half]
    assert torch.equal(weight, packed)
    assert torch.equal(actual.cpu(), expected)
    # W4FA16 MMA consumes the same byte layout, without W4FA8's scale window.
    weight16, scale16 = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A16_MXFP4_MMA, False, packed, raw)
    assert torch.equal(weight16, packed)
    assert torch.equal(scale16, actual)
    raw16 = raw.clone()
    raw16[0, 0, 0] = 113
    raw16[1, 1, 1] = 141
    _, scale16 = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A16_MXFP4_MMA, False, packed, raw16)
    assert scale16[0, 0, 0].item() == 113
    assert scale16[1, 0, 17].item() == 141
    packed64 = packed[..., :32].contiguous()
    raw64 = raw[..., :2].contiguous()
    weight64, scale64 = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A16_MXFP4_MMA, False, packed64, raw64)
    assert torch.equal(weight64, packed64)
    assert torch.equal(scale64.cpu(), expected[..., :128])
    # Existing MXFP4 activation-fusion mode retains its interleaving and scale layout.
    weight, scale = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.MXFP4, True, packed, raw)
    interleaved_weight = torch.stack((packed[:, :n//2], packed[:, n//2:]), 2).flatten(1, 2)
    interleaved_scale = torch.stack((raw[:, :n//2], raw[:, n//2:]), 2).flatten(1, 2)
    assert torch.equal(weight, interleaved_weight)
    assert torch.equal(scale, deep_gemm.preprocess_mxfp4_scales(interleaved_scale))
    legacy_weight, legacy_scale = deep_gemm.preprocess_mxfp4_weight_for_act_and_quant_fusing(packed, raw)
    assert torch.equal(legacy_weight, weight) and torch.equal(legacy_scale, scale)
    print('PASS unified preprocessing layouts and existing MXFP4 mode')


def test_basis():
    # Each row selects one K coordinate. All weight codes and alternating group scales
    # expose layout swaps even when aggregate random-GEMM error is small.
    n, k = 64, 128
    codes = ((torch.arange(n, device='cuda')[:, None]*3 + torch.arange(k, device='cuda')[None, :]*5) % 16).byte()
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).unsqueeze(0).contiguous()
    aq = torch.eye(k, device='cuda').to(torch.float8_e4m3fn)
    scales = torch.linspace(.5, 1.5, k, device='cuda').reshape(k, 1)
    out = torch.empty(k, n, device='cuda', dtype=torch.bfloat16)
    indices = torch.zeros(k, device='cuda', dtype=torch.int32)
    base = torch.tensor([114, 127, 130, 140], device='cuda', dtype=torch.int16)
    for offset in (-5, 0, 5):
        raw = (base + offset).byte().expand(1, n, 4).contiguous()
        rhs = deep_gemm.preprocess_weights_and_scales(deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad((aq, scales * rhs[2]), rhs[:2], out, indices)
        expected = (decode_weight(packed[0], raw[0]).T * scales).bfloat16()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    raw = base.byte().expand(1, n, 4).contiguous()
    rhs16 = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A16_MXFP4_MMA, False, packed, raw)
    out16 = torch.empty_like(out)
    deep_gemm.m_grouped_gemm_w4a16_nopad(
        torch.eye(k, device='cuda', dtype=torch.bfloat16), rhs16, out16, indices)
    torch.testing.assert_close(out16, decode_weight(packed[0], raw[0]).T.bfloat16(), rtol=0, atol=0)
    print('PASS one-hot K basis, scale boundaries and nonuniform row scales')


def test_fused_k32_basis():
    # Exercise both K64 weight slices, N tail, and multiple compact M64 blocks.
    # Vary scales along N as well as K so a misplaced scale tile cannot pass.
    n, k = 320, 256
    codes = ((torch.arange(n, device='cuda')[:, None]*3 +
              torch.arange(k, device='cuda')[None, :]*5) % 16).byte()
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).unsqueeze(0).contiguous()
    raw = ((torch.arange(n, device='cuda')[:, None] +
            7*torch.arange(k//32, device='cuda')[None, :]) % 27 + 109).byte().unsqueeze(0).contiguous()
    rhs = deep_gemm.preprocess_weights_and_scales(deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
    assert rhs[2] == .03125
    config = (deep_gemm.get_num_sms(), 64, 256, 128, 64, 64, 128, 2, 1)
    for m in (117, 128, 131):
        aq = torch.eye(k, device='cuda')[:m].to(torch.float8_e4m3fn)
        scales = torch.linspace(.5, 1.5, m, device='cuda').reshape(m, 1)
        routes = torch.zeros(m, 1, device='cuda', dtype=torch.int32)
        _, rows, experts, tokens, aligned, _, _ = deep_gemm.moe_align_block_size(
            aq, rhs[0], routes, config=config[:7])
        out = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
            (aq, scales * rhs[2]), rhs[:2], out, rows, experts, tokens, aligned, config)
        expected = (decode_weight(packed[0], raw[0]).T[:m]*scales).bfloat16()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    print('PASS fused K32 basis, shifted full scale window, M64 block boundaries and N tail')


def test_masked_shifted_scale():
    groups, n, k, padding_m = 2, 64, 128, 128
    packed = torch.randint(256, (groups, n, k // 2), device='cuda', dtype=torch.uint8)
    raw = torch.full((groups, n, k // 32), 124, device='cuda', dtype=torch.uint8)
    raw[0, 0, 0] = 109
    rhs = deep_gemm.preprocess_weights_and_scales(deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
    assert rhs[2] == .03125
    aq = torch.eye(k, device='cuda')[:padding_m].to(torch.float8_e4m3fn).unsqueeze(0).expand(groups, -1, -1).contiguous()
    row_scales = torch.ones(groups, padding_m, 1, device='cuda', dtype=torch.float32)
    rows = torch.tensor([3, 5], device='cuda', dtype=torch.int32)
    out = torch.empty(groups, padding_m, n, device='cuda', dtype=torch.bfloat16)
    deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_masked((aq, row_scales * rhs[2]), rhs[:2], out, rows, 4)
    for expert, count in enumerate((3, 5)):
        expected = decode_weight(packed[expert], raw[expert]).T[:count].bfloat16()
        torch.testing.assert_close(out[expert, :count], expected, rtol=0, atol=0)
    print('PASS masked shifted scale and per-expert rows')


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)
    utils.set_acc_check(True)
    utils.set_benchmark(False)

    if is_ppu1v5_device():
        test_grouped_gemm()
        test_nopad_expert_limit()
        test_nopad_n_expand()
        test_nopad_resident_a_and_scales()
        test_fused_nopad_equivalence()
        test_boundaries()
        test_tensor_scale_shift()
        test_preprocessing_layout()
        test_basis()
        test_fused_k32_basis()
        test_masked_shifted_scale()
    else:
        print('Skipping W4FA8: requires PPU1.5')
