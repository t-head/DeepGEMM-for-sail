"""PPU1.5 NVFP4 W4A8 accuracy tests and offline scale absorption example.

Run: python tests/test_w4a8_nvfp4.py
Or:  python -m pytest tests/test_w4a8_nvfp4.py -q -s
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

import pytest
import torch

# Direct execution exercises this checkout, even if another is installed.
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deep_gemm


# M, N, K: tails for small batches and repeated K stages for larger batches.
GEMM_CASES = (
    (1, 1024, 64),   # One K tile must disable N expansion for the two-stage pipeline.
    (1, 1024, 128),  # Two K tiles retain N expansion.
    (1, 256, 192),   # Three K64 tiles recycle native K16 scale slots.
    (17, 256, 192),  # The same reuse with FP8 blockwise scales.
    (1, 192, 160),
    (17, 192, 160),
    (128, 256, 1024),
    (257, 256, 1024),
    (4096, 256, 1024),
)

# EP, N, K: model dimensions for the 384-expert, top-k=4 NVFP4 MoE workload.
MODEL_GEMM_CASES = (
    (32, 10240, 7168),  # Gate/up.
    (32, 7168, 5120),   # Down.
    (64, 10240, 7168),
    (64, 7168, 5120),
)


def _is_supported_device():
    if not torch.cuda.is_available():
        return False
    # tests.utils queries the device at import time; defer it until one is available.
    from utils import is_ppu1v5_device
    return is_ppu1v5_device()


@pytest.fixture(scope='module', autouse=True)
def _require_ppu1v5():
    if not _is_supported_device():
        pytest.skip('NVFP4 W4A8 requires PPU1.5')


@contextmanager
def _fp32_reference():
    # Both pytest and direct calls need references without TF32 rounding.
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def _decode_nvfp4(codes, scales):
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                       device=codes.device)
    return lut[codes.long()] * scales.float().repeat_interleave(16, -1)


def _cast_fp8(values):
    # Match the existing kernel's finite saturation, including out-of-range input.
    return values.clamp(-448, 448).to(torch.float8_e4m3fn)


def _expand_block_scale(scale, n, k):
    return scale.repeat_interleave(128, -2).repeat_interleave(128, -1)[..., :n, :k]


def _relative_l2(actual, expected):
    return ((actual.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1e-20)).item()


def _check_nvfp4(out, ref, name):
    expected = ref.to(torch.bfloat16)
    diff = (out.float() - expected.float()).square().sum()
    norm = expected.float().square().sum().clamp_min(1e-10)
    error = (diff / norm).item()
    assert error < 2e-05, f'{name}: squared relative L2 error={error}'
    torch.testing.assert_close(out, expected, rtol=0.02, atol=0.25, msg=name)


def _prepare_nvfp4(codes, native_scale, execution):
    mode = (deep_gemm.PreprocessMode.W4A8_NVFP4 if execution == 'fused'
            else deep_gemm.PreprocessMode.W4A8_NVFP4_FP8)
    packed = codes[..., ::2] | (codes[..., 1::2] << 4)
    return deep_gemm.preprocess_weights_and_scales(mode, False, packed, native_scale)


def _reference_gemm(a, weight, kind, counts):
    if kind == 'dense':
        return a @ weight[0].t()
    outputs = []
    offset = 0
    for expert, count in enumerate(counts):
        if count:
            outputs.append(a[offset:offset + count] @ weight[expert].t())
        offset += count
    return torch.cat(outputs) if outputs else a.new_empty((0, weight.shape[-2]))


def _absorb_fp8_scale(codes, native_scale):
    """Return normalized K16 E4M3 scales and restoring 128x128 FP32 scales.

    Global scale is kept separate and passed unchanged to both GEMM paths.
    Power-of-two normalization preserves native E4M3 scales except at their
    representable range boundaries. Small scales can still underflow; the
    comparison measures that loss against the original NVFP4 values.
    """
    e, n, k = codes.shape
    values = _decode_nvfp4(codes, native_scale)
    amax = values.abs().reshape(e, n // 128, 128, k // 128, 128).amax(dim=(2, 4))
    scale_amax = native_scale.float().reshape(e, n // 128, 128, k // 128, 8).amax(dim=(2, 4))
    # Also keep the normalized K16 scales themselves within E4M3 range.
    bound = torch.maximum(amax, scale_amax)
    rhs_scales = torch.exp2(torch.ceil(torch.log2((bound / 448).clamp_min(1e-30))))
    rhs_scales = torch.where(bound == 0, torch.ones_like(rhs_scales), rhs_scales)
    divisor = rhs_scales.repeat_interleave(128, -2).repeat_interleave(8, -1)
    normalized = (native_scale.float() / divisor).to(torch.float8_e4m3fn)
    assert torch.isfinite(normalized.float()).all()
    assert _decode_nvfp4(codes, normalized).abs().max() <= 448
    return normalized, rhs_scales


@_fp32_reference()
def run_nvfp4_gemm(args, *, check_accuracy=True, benchmark=False, ref_backend='device', regressions=False):
    """Shared NVFP4 case runner for the standalone tests and common GEMM entries."""
    from utils import construct_group_m_list, get_masked_padding_m

    torch.manual_seed(42)
    kinds = {'DenseGemm': 'dense', 'Normal': 'dense', 'GroupedNoPad': 'nopad',
             'GroupedMasked': 'masked', 'GroupedFused': 'fused'}
    if args['gemm_type'] not in kinds:
        raise ValueError('NVFP4 supports dense, nopad, masked and fused-routing GEMMs')
    kind = kinds[args['gemm_type']]
    execution = args.get('execution', 'fused')
    if execution not in ('fused', 'dequantize'):
        raise ValueError("execution must be 'fused' or 'dequantize'")
    quant_type = args.get('quant_type', 'block')
    if quant_type not in ('block', 'channel') or args.get('group_size', 16) != 16:
        raise ValueError('NVFP4 requires K16 native scales and block/channel outer scales')
    if (args.get('with_bias') or args.get('epilogue_type', 'Default') != 'Default'
            or args.get('accumulate', 'none') != 'none'):
        raise ValueError('NVFP4 supports the default BF16 epilogue without bias or accumulation')
    if args.get('enable_sbo_overlap'):
        raise ValueError('NVFP4 does not support enable_sbo_overlap')
    n, k = args['n'], args['k']
    m = args['num_token'] if kind == 'fused' else args['m']
    e = 1 if kind == 'dense' else args['groups']
    topk = args.get('topk', 2) if kind == 'fused' else 1
    if m < 0 or e <= 0 or n <= 0 or n % 64 or k <= 0 or k % 32:
        raise ValueError('NVFP4 requires M >= 0, groups > 0, N % 64 == 0 and K % 32 == 0')
    if not 1 <= topk <= e:
        raise ValueError('topk must be between 1 and groups')
    if execution == 'dequantize' and kind == 'fused' and (n % 128 or k % 128):
        raise ValueError('NVFP4 decode-to-FP8 fused routing requires N/K % 128 == 0')
    name = f'NVFP4 {execution}: {kind}, M={m}, N={n}, K={k}'
    a_scale_group = args.get('lhs_scale_group_size', 0 if quant_type == 'channel' else 128)
    b_scale_block = 0 if quant_type == 'channel' else 128
    if a_scale_group not in (0, 128):
        raise ValueError('The NVFP4 case runner supports per-token or K128 activation scales')
    if execution == 'dequantize' and b_scale_block != 128:
        raise ValueError('NVFP4 decode-to-FP8 execution requires quant_type:block')
    if kind in ('nopad', 'masked'):
        counts = [int(x) for x in construct_group_m_list(
            args.get('distribution', 'uniform'), e, m, is_mask=kind == 'masked',
            em=args.get('em', (m + e - 1) // e))]
        if len(counts) != e or any(x < 0 for x in counts):
            raise ValueError('distribution must contain one nonnegative row count per expert')
        if kind == 'nopad':
            if sum(counts) != m:
                raise ValueError('GroupedNoPad distribution must sum to m')
        else:
            m = get_masked_padding_m(counts, args.get('padding_m', 0))
    else:
        counts = [m] * e
    m_rows = torch.tensor(counts, device='cuda', dtype=torch.int32)
    # Compile-only workers need full GEMM shapes but only one preprocessed expert.
    warm_up = bool(os.environ.get('HGGC_WARM_UP'))
    check_accuracy = check_accuracy and not warm_up
    weight_experts = 1 if warm_up else e
    codes = torch.randint(16, (weight_experts, n, k), device='cuda', dtype=torch.uint8)
    native_scale = (torch.rand(weight_experts, n, k // 16, device='cuda') + .25).to(torch.float8_e4m3fn)
    if regressions and k == 192:
        # Make reading an adjacent K64 tile's scales visibly incorrect.
        factors = torch.tensor([.25, 4., 1.], device='cuda').repeat_interleave(4)
        native_scale = (native_scale.float() * factors).to(torch.float8_e4m3fn)
    rhs = _prepare_nvfp4(codes, native_scale, execution)
    if warm_up:
        rhs = tuple(t.expand(e, *t.shape[1:]).contiguous() for t in rhs)

    a_shape = (e, m, k) if kind == 'masked' else (m, k)
    a = torch.randn(a_shape, device='cuda').to(torch.float8_e4m3fn)
    a_scale = torch.rand((*a_shape[:-1], (k + 127) // 128 if a_scale_group else 1), device='cuda') + .25
    b_scale_shape = (e, (n + 127) // 128, (k + 127) // 128) if b_scale_block else (e, n)
    b_scale = torch.rand(b_scale_shape, device='cuda') + .25
    if regressions:
        b_scale[:, 0] = 0  # Include zero outer weight scales.
    global_scale = torch.linspace(.5, 1.25, e, device='cuda')
    out_shape = (*a_shape[:-1], n) if kind != 'fused' else (m * topk, n)
    out = torch.full(out_shape, 123., device='cuda', dtype=torch.bfloat16)
    workspace = (torch.empty((e, n, k), device='cuda', dtype=torch.float8_e4m3fn)
                 if execution == 'dequantize' else None)
    workspace_arg = workspace[0] if kind == 'dense' and workspace is not None else workspace
    kwargs = dict(
        lhs_scale_group_size=a_scale_group, rhs_scales=b_scale[0] if kind == 'dense' else b_scale,
        rhs_scale_block_size=b_scale_block,
        weight_global_scale=global_scale[0] if kind == 'dense' else global_scale,
        execution=execution, fp8_workspace=workspace_arg,
    )
    configs = None
    if kind == 'nopad':
        m_indices = torch.repeat_interleave(torch.arange(e, device='cuda', dtype=torch.int32), m_rows)
    elif kind == 'fused':
        # Standalone regressions keep expert zero empty; common cases use all experts.
        first_expert = 1 if regressions else 0
        routes = ((torch.arange(m, device='cuda')[:, None]
                   + torch.arange(topk, device='cuda')[None, :]) % (e - first_expert) + first_expert).int()
        routing_weight = workspace if workspace is not None else rhs[0]
        routing_options = (dict(lhs_scale_group_size=a_scale_group, rhs_scale_block_size=b_scale_block)
                           if execution == 'fused' else {})
        configs, aligned_rows, experts, tokens, blocks, _, _ = deep_gemm.moe_align_block_size(
            a, routing_weight, routes, False, **routing_options)
        if regressions and m == 17:
            # Explicit overrides use the same public entry as automatic selection.
            override = (max(1, configs[0] // 2), *configs[1:])
            selected = deep_gemm.moe_align_block_size(
                a, routing_weight, routes, False, config=override, **routing_options)
            assert selected[0] == override
            configs, aligned_rows, experts, tokens, blocks, _, _ = selected
    if regressions and execution == 'fused' and kind in ('nopad', 'masked'):
        # Exercise rolling operands and both shared-decode M tiles, including
        # partial M fragments and persistent CTAs moving between experts.
        tiles = {128: (64, 128, 128, 64, 32, 128, 3),
                 257: (128, 128, 128, 64, 32, 128, 3),
                 4096: (256, 64, 128, 64, 32, 128, 3)}
        if m in tiles:
            configs = (deep_gemm.get_num_sms(), *tiles[m])

    def run():
        lhs = (a, a_scale)
        if kind == 'dense':
            deep_gemm.gemm_fp8_nvfp4_bf16_nt(lhs, tuple(x[0] for x in rhs), out, **kwargs)
        elif kind == 'nopad':
            deep_gemm.m_grouped_gemm_fp8_nvfp4_bf16_nt_nopad(
                lhs, rhs, out, m_indices, m_rows=m_rows, configs=configs, **kwargs)
        elif kind == 'masked':
            deep_gemm.m_grouped_gemm_fp8_nvfp4_bf16_nt_masked(
                lhs, rhs, out, m_rows, args.get('em', m), configs=configs, **kwargs)
        else:
            deep_gemm.m_grouped_gemm_fp8_nvfp4_bf16_nt_fused(
                lhs, rhs, out, aligned_rows, experts, tokens, blocks, configs, **kwargs)

    def check():
        device = 'cpu' if ref_backend == 'host' else 'cuda'
        a_ref = a.to(device).float() * a_scale.to(device).repeat_interleave(a_scale_group or k, -1)[..., :k]
        b_ref = _decode_nvfp4(codes.to(device), native_scale.to(device))
        if b_scale_block:
            # Blockwise execution rounds native NVFP4 values to FP8.
            b_ref = _cast_fp8(b_ref).float()
            b_ref *= _expand_block_scale(b_scale.to(device), n, k)
        else:
            b_ref *= b_scale.to(device)[..., None]
        b_ref *= global_scale.to(device)[:, None, None]
        result = out.to(device)
        if kind in ('dense', 'nopad'):
            _check_nvfp4(result, _reference_gemm(a_ref, b_ref, kind, counts), name)
        elif kind == 'masked':
            for expert, count in enumerate(counts):
                if count:
                    _check_nvfp4(result[expert, :count], a_ref[expert, :count] @ b_ref[expert].t(),
                                 f'{name}, expert={expert}')
                assert torch.all(result[expert, count:] == 123.), f'{name}, expert={expert}: padding overwritten'
        else:
            token_ids = torch.arange(m, device=device).repeat_interleave(topk)
            route_ids = routes.to(device).flatten()
            ref = torch.cat([a_ref[token_ids[route_ids == expert]] @ b_ref[expert].t()
                             for expert in range(e)])
            _check_nvfp4(result, ref, name)
        if workspace is not None:
            expected = _cast_fp8(_decode_nvfp4(codes, native_scale))
            torch.testing.assert_close(workspace.float(), expected.float(), rtol=0, atol=0,
                                       msg=f'{name}: FP8 workspace')

    print(name, flush=True)
    if regressions and execution == 'fused' and kind == 'dense' and m == 17:
        # Binary K-warp reduction requires power-of-two group counts.
        invalid = (deep_gemm.get_num_sms(), 16, 64, 96, 16, 32, 32, 2)
        with pytest.raises(RuntimeError, match='power of two'):
            deep_gemm.gemm_fp8_nvfp4_bf16_nt(
                (a, a_scale), tuple(x[0] for x in rhs), out, configs=invalid, **kwargs)
    if regressions and kind == 'fused' and m == 17:
        # Metadata made for one M tile cannot be consumed using a different tile.
        invalid = list(configs)
        invalid[1] *= 2
        invalid[4] = invalid[1]
        with pytest.raises(RuntimeError, match='metadata block_m'):
            deep_gemm.m_grouped_gemm_fp8_nvfp4_bf16_nt_fused(
                (a, a_scale), rhs, out, aligned_rows, experts, tokens, blocks, tuple(invalid), **kwargs)
        if execution == 'fused':
            with pytest.raises(RuntimeError, match='Activation scale group'):
                deep_gemm.moe_align_block_size(a, rhs[0], routes, lhs_scale_group_size=-1)
            with pytest.raises(RuntimeError, match='rhs_scale_block_size'):
                deep_gemm.moe_align_block_size(a, rhs[0], routes, rhs_scale_block_size=64)
        else:
            with pytest.raises(RuntimeError, match='require NVFP4 prepared weights'):
                deep_gemm.moe_align_block_size(a, workspace, routes, lhs_scale_group_size=128)
    run()
    if check_accuracy:
        check()
    print('Passed with acc_check\n' if check_accuracy else 'Passed without acc_check\n', flush=True)
    if benchmark and not warm_up:
        from utils import bench_kineto
        # Include the decoder in decode-plus-FP8 timing, outside preparation/allocation.
        names = (('gemm', 'dequantize_nvfp4_fp8') if execution == 'dequantize'
                 else ('w4a8_nvfp4',))
        times = bench_kineto(run, names, suppress_kineto_output=True, with_multiple_kernels=True)
        assert all(t > 0 for t in times), f'NVFP4 profiling did not find every kernel: {names}, {times}'
        t = sum(times)
        rows = sum(counts) if kind in ('nopad', 'masked') else m * topk
        print(f' > Performance (NVFP4 {execution}, {kind}, groups={e}, rows={rows}, n={n}, k={k}): '
              f'{t * 1e6:.1f} us | throughput: {2 * rows * n * k / t / 1e12:.2f} TFLOPS')
    if regressions and check_accuracy and kind == 'nopad' and (m == 17 or (m == 1 and k == 192)):
        # Replay verifies that weights and scales refresh in both execution modes.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        codes = torch.randint(16, codes.shape, device='cuda', dtype=torch.uint8)
        native_scale = (native_scale.float() * .75).to(native_scale.dtype)
        updated = _prepare_nvfp4(codes, native_scale, execution)
        for dst, src in zip(rhs, updated):
            dst.copy_(src)
        a_scale.mul_(.75)
        b_scale.add_(.125)
        global_scale.mul_(.875)
        for _ in range(16 if k == 192 else 1):
            graph.replay()
            check()


def _test_nvfp4_preprocessing_memory(execution):
    mode = (deep_gemm.PreprocessMode.W4A8_NVFP4 if execution == 'fused'
            else deep_gemm.PreprocessMode.W4A8_NVFP4_FP8)
    # Bound temporaries across experts and within one model-size expert.
    for e, n, k in ((8, 2048, 2048), (1, 10240, 7168)):
        packed = torch.randint(256, (e, n, k // 2), device='cuda', dtype=torch.uint8)
        scales = torch.ones((e, n, k // 16), device='cuda').to(torch.float8_e4m3fn)
        torch.cuda.synchronize()
        start = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        rhs = deep_gemm.preprocess_weights_and_scales(mode, False, packed, scales)
        torch.cuda.synchronize()
        temporary = torch.cuda.max_memory_allocated() - start - sum(t.nbytes for t in rhs)
        assert temporary < 128 * 1024**2, f'{execution}: preprocessing temporary memory={temporary} bytes'
        del rhs


def _test_nvfp4_gemm(execution):
    _test_nvfp4_preprocessing_memory(execution)
    for m, n, k in GEMM_CASES:
        for kind in ('dense', 'nopad', 'masked', 'fused'):
            if execution == 'dequantize' and kind == 'fused':
                n_case, k_case = 256, 256  # Mainline FP8 routing requires N/K % 128 == 0.
            else:
                n_case, k_case = n, k
            types = {'dense': 'DenseGemm', 'nopad': 'GroupedNoPad',
                     'masked': 'GroupedMasked', 'fused': 'GroupedFused'}
            counts = [0, m // 4, m // 3, m - m // 4 - m // 3]
            if kind == 'masked':
                counts[-1] = m
            args = dict(gemm_type=types[kind], groups=4, m=m, num_token=m, topk=2,
                        n=n_case, k=k_case, distribution=counts, padding_m=m,
                        execution=execution, lhs_scale_group_size=0 if m == 1 else 128,
                        quant_type='channel' if execution == 'fused' and m == 1 else 'block')
            run_nvfp4_gemm(args, regressions=True)
    for ep, n, k in MODEL_GEMM_CASES:
        groups = 384 // ep
        for batch in (1, 128, 4096):
            m = 4 * batch
            counts = [m // groups + int(i < m % groups) for i in range(groups)]
            args = dict(gemm_type='GroupedNoPad', groups=groups, m=m, n=n, k=k,
                        distribution=counts, execution=execution, quant_type='block')
            print(f'Model-size NVFP4: EP={ep}, tokens/rank={batch}, local rows={m}', flush=True)
            run_nvfp4_gemm(args)
    print('Passed\n')


def _test_nvfp4_offline_scale_profile(profile):
    e, n, k = 3, 256, 256
    torch.manual_seed(42)
    codes = torch.randint(16, (e, n, k), device='cuda', dtype=torch.uint8)
    samples = torch.rand((e, n, k // 16), device='cuda')
    if profile == 'small':
        scales = samples + .25
    elif profile == 'large':
        scales = 128 + 320 * samples
    else:
        scales = torch.exp2(-9 + (9 + 8.807354922) * samples)
    native_scale = scales.to(torch.float8_e4m3fn)
    if profile != 'small':
        native_scale[..., 0] = 448
        codes[..., 0] = 7  # q=6 exposes the direct cast's range limit.
    global_scale = torch.exp2(torch.linspace(-12, -6, e, device='cuda'))
    original = _decode_nvfp4(codes, native_scale)
    target = original * global_scale[:, None, None]
    folded, restore = _absorb_fp8_scale(codes, native_scale)
    methods = [('direct_cast', native_scale, torch.ones_like(restore)),
               ('offline_absorb', folded, restore)]
    weight_errors = {}
    for method, kernel_scales, rhs_scales in methods:
        pre_cast = _decode_nvfp4(codes, kernel_scales)
        # E2M1 x E4M3 fits BF16 exactly before either FP8 cast.
        torch.testing.assert_close(pre_cast, pre_cast.bfloat16().float(), rtol=0, atol=0)
        w8 = _cast_fp8(pre_cast)
        restored = w8.float() * _expand_block_scale(rhs_scales, n, k) * global_scale[:, None, None]
        weight_errors[method] = _relative_l2(restored, target)
        saturated = int((pre_cast.abs() > 448).sum().item())
        if method == 'offline_absorb':
            assert saturated == 0 and weight_errors[method] < .05, (profile, method, weight_errors[method])
            if profile == 'small':
                torch.testing.assert_close(restored, direct_restored, rtol=0, atol=0)
        else:
            direct_restored = restored
        print(json.dumps({
            'profile': profile, 'method': method,
            'weight_vs_nvfp4_rel_l2': weight_errors[method],
            'saturated_values': saturated,
            'nonzero_to_zero': int(((w8.float() == 0) & (original != 0)).sum().item()),
        }), flush=True)
        prepared = {execution: _prepare_nvfp4(codes, kernel_scales, execution)
                    for execution in ('fused', 'dequantize')}
        for batch in (1, 128, 4096):
            torch.manual_seed(100 + batch)
            a = torch.randn((batch, k), device='cuda').to(torch.float8_e4m3fn)
            a_scale = torch.rand((batch, k // 128), device='cuda') + .25
            a_ref = a.float() * a_scale.repeat_interleave(128, -1)
            counts = [0, batch // 3, batch - batch // 3]
            m_rows = torch.tensor(counts, device='cuda', dtype=torch.int32)
            m_indices = torch.repeat_interleave(torch.arange(e, device='cuda', dtype=torch.int32), m_rows)
            for kind in ('dense', 'nopad'):
                reference = _reference_gemm(a_ref, target, kind, counts)
                fp8_reference = _reference_gemm(a_ref, restored, kind, counts)
                record = {'profile': profile, 'method': method, 'batch': batch, 'kind': kind,
                          'fp8_reference_vs_nvfp4_rel_l2': _relative_l2(fp8_reference, reference)}
                outputs = []
                for execution in ('fused', 'dequantize'):
                    rhs = prepared[execution]
                    workspace = (torch.empty((e, n, k), device='cuda', dtype=torch.float8_e4m3fn)
                                 if execution == 'dequantize' else None)
                    workspace_arg = workspace[0] if kind == 'dense' and workspace is not None else workspace
                    kwargs = dict(
                        lhs_scale_group_size=128, rhs_scale_block_size=128,
                        rhs_scales=rhs_scales[0] if kind == 'dense' else rhs_scales,
                        weight_global_scale=global_scale[0] if kind == 'dense' else global_scale,
                        execution=execution, fp8_workspace=workspace_arg,
                    )
                    out = torch.empty((batch, n), device='cuda', dtype=torch.bfloat16)
                    if kind == 'dense':
                        deep_gemm.gemm_fp8_nvfp4_bf16_nt(
                            (a, a_scale), tuple(x[0] for x in rhs), out, **kwargs)
                    else:
                        deep_gemm.m_grouped_gemm_fp8_nvfp4_bf16_nt_nopad(
                            (a, a_scale), rhs, out, m_indices, m_rows=m_rows, **kwargs)
                    kernel_error = _relative_l2(out, fp8_reference)
                    output_error = _relative_l2(out, reference)
                    assert kernel_error < .006, (profile, method, batch, kind, execution, kernel_error)
                    if method == 'offline_absorb':
                        assert output_error < .06, (profile, batch, kind, execution, output_error)
                    record[execution + '_vs_fp8_reference_rel_l2'] = kernel_error
                    record[execution + '_vs_nvfp4_rel_l2'] = output_error
                    if workspace is not None:
                        decoded = workspace[0] if kind == 'dense' else workspace
                        expected = w8[0] if kind == 'dense' else w8
                        torch.testing.assert_close(decoded.float(), expected.float(), rtol=0, atol=0)
                    outputs.append(out)
                record['fused_vs_dequantize_rel_l2'] = _relative_l2(outputs[0], outputs[1])
                assert record['fused_vs_dequantize_rel_l2'] < .006
                print(json.dumps(record), flush=True)
    if profile != 'small':
        assert weight_errors['offline_absorb'] < weight_errors['direct_cast'], (profile, weight_errors)


def test_nvfp4_decode() -> None:
    """Fused dense/grouped boundaries and model-size EP32/EP64 nopad GEMMs."""
    with _fp32_reference():
        _test_nvfp4_gemm('fused')


def test_nvfp4_prefill() -> None:
    """Per-call decode plus FP8 for boundaries and model-size EP32/EP64 GEMMs."""
    with _fp32_reference():
        _test_nvfp4_gemm('dequantize')


def test_nvfp4_offline_scale() -> None:
    """Compare offline absorption and direct casts against the same NVFP4 weights.

    Printed relative L2 errors are norm(actual - reference) / norm(reference).
    The NVFP4 reference uses the original K16/global scales in FP32; the FP8
    reference includes each method's conversion and outer scales.
    """
    with _fp32_reference():
        for profile in ('small', 'large', 'wide'):
            _test_nvfp4_offline_scale_profile(profile)


if __name__ == '__main__':
    if _is_supported_device():
        print(f'Library path: {deep_gemm.__path__}\n')
        test_nvfp4_decode()
        test_nvfp4_prefill()
        test_nvfp4_offline_scale()
    else:
        print('Skipping NVFP4: requires PPU1.5')
