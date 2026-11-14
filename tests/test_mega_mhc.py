"""V4.1 Flash shifted Mega-mHC versus the official PyTorch reference."""
import argparse
import json
import os
import torch
import deep_gemm
from deep_gemm import bench_kineto
from deep_gemm.testing import assert_bitwise_equal, calc_diff


def _sinkhorn_reference(value: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    value = value.softmax(dim=-1) + eps
    value = value / (value.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        value = value / (value.sum(dim=-1, keepdim=True) + eps)
        value = value / (value.sum(dim=-2, keepdim=True) + eps)
    return value


def fp8_reference(x, per_token=False):
    values = x.view(x.size(0), -1, x.size(-1) if per_token else 32)
    clamped = values.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = clamped / 448.0
    if per_token:
        fp8 = (values.float() * (448.0 / clamped).unsqueeze(2)).to(torch.float8_e4m3fn).view_as(x)
        return fp8, sf
    bits = sf.view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0).int()
    sf = (exponent.clamp(1, 254) << 23).view(torch.float32)
    fp8 = (values * sf.reciprocal().unsqueeze(2)).to(torch.float8_e4m3fn).view_as(x)
    return fp8, (sf.view(torch.int32) >> 23).to(torch.uint8).view(torch.int32)


@torch.no_grad()
def mhc_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    mix_scales: torch.Tensor,
    mix_bases: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    hc_mult: int,
    hc_norm_eps: float,
    hc_pre_eps: float,
    hc_post_scale: float,
    sinkhorn_eps: float,
    num_sinkhorn_iters: int,
    rmsnorm_eps: float,
    rmsnorm_scale: float,
    shifted_prev_mix: torch.Tensor,
    sf_layout: str = 'bf16',
) -> dict[str, torch.Tensor]:
    new_residual = x.float().unsqueeze(1) * post_mix
    new_residual += torch.einsum(
        'tij,tih->tjh', comb_res_mix, residual.float())
    new_residual = new_residual.bfloat16()
    norm_input = (
        new_residual.float() * shifted_prev_mix).sum(dim=1).bfloat16()

    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        mixes = new_residual.flatten(1).float() @ fn.mT
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
    hc_sqr_sum = new_residual.float().square().sum((1, 2))
    mixes *= torch.rsqrt(
        hc_sqr_sum / (hc_mult * x.size(1)) + hc_norm_eps).unsqueeze(1)
    scales = torch.cat((
        mix_scales[0].expand(hc_mult),
        mix_scales[1].expand(hc_mult),
        mix_scales[2].expand(hc_mult * hc_mult),
    ))
    mixes = mixes * scales + mix_bases
    new_prev_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(2) + hc_pre_eps
    new_post_mix = (
        mixes[:, hc_mult:2 * hc_mult].sigmoid() * hc_post_scale
    ).unsqueeze(2)
    new_comb_res_mix = _sinkhorn_reference(
        mixes[:, 2 * hc_mult:].view(-1, hc_mult, hc_mult),
        num_sinkhorn_iters,
        sinkhorn_eps,
    )
    norm_input_float = norm_input.float()
    y_bf16 = (
        norm_input_float
        * torch.rsqrt(norm_input_float.square().mean(1) + rmsnorm_eps).unsqueeze(1)
        * (rmsnorm_weight.float() * rmsnorm_scale)
    ).bfloat16()
    result = {
        'new_residual': new_residual,
        'new_prev_mix': new_prev_mix,
        'new_post_mix': new_post_mix,
        'new_comb_res_mix': new_comb_res_mix,
        'y_bf16': y_bf16,
    }
    if sf_layout != 'bf16':
        result['y_fp8'], sf = fp8_reference(y_bf16, per_token=sf_layout == 'token')
        if sf_layout == 'token':
            result['y_token_sf'] = sf
        elif sf_layout == 'col':
            result['y_gemm_sf'] = sf
        else:
            result.update(y_routed_sf=sf, y_shared_sf=sf)
    return result


def make_inputs(num_tokens: int, hidden: int, seed: int):
    hc_mult = 4
    num_hc_outputs = hc_mult * (hc_mult + 2)
    generator = torch.Generator(device='cuda').manual_seed(seed)

    def randn(shape, dtype):
        return torch.randn(shape, dtype=dtype, device='cuda', generator=generator)

    inputs = dict(
        x=randn((num_tokens, hidden), torch.bfloat16),
        residual=randn((num_tokens, hc_mult, hidden), torch.bfloat16),
        post_mix=randn((num_tokens, hc_mult, 1), torch.float).sigmoid(),
        comb_res_mix=torch.rand((num_tokens, hc_mult, hc_mult), device='cuda', generator=generator),
        shifted_prev_mix=randn((num_tokens, hc_mult, 1), torch.float).sigmoid(),
        fn=randn((num_hc_outputs, hc_mult * hidden), torch.float).mul_(1e-2),
        mix_scales=randn((3,), torch.float).mul_(0.1),
        mix_bases=randn((num_hc_outputs,), torch.float).mul_(0.1),
        rmsnorm_weight=randn((hidden,), torch.float).mul_(0.1).add_(1).bfloat16(),
        hc_mult=hc_mult,
        hc_norm_eps=2e-5,
        hc_pre_eps=3e-4,
        hc_post_scale=1.25,
        sinkhorn_eps=2e-6,
        rmsnorm_eps=7e-6,
        rmsnorm_scale=1.25,
    )
    for _ in range(3):
        inputs['comb_res_mix'] /= inputs['comb_res_mix'].sum(-1, keepdim=True)
        inputs['comb_res_mix'] /= inputs['comb_res_mix'].sum(-2, keepdim=True)
    duplicates = tuple(i for i in (63, 64, 511, 512, num_tokens - 1) if 0 < i < num_tokens)
    for name in ('x', 'residual', 'post_mix', 'comb_res_mix', 'shifted_prev_mix'):
        if duplicates:
            inputs[name][list(set(duplicates))] = inputs[name][0].clone()
    return inputs, duplicates


def make_case(tokens=64, hidden=5120, seed=2026, num_sinkhorn_iters=20,
              sf_layout='bf16', shared_sf_block_m=224, store_bf16=True):
    if sf_layout == 'bf16' and not store_bf16:
        raise ValueError('MegaMhc BF16 mode requires store_bf16=1')
    inputs, duplicates = make_inputs(tokens, hidden, seed)
    inputs['num_sinkhorn_iters'] = num_sinkhorn_iters
    outputs = dict(
        new_residual=torch.empty_like(inputs['residual']),
        new_prev_mix=torch.empty_like(inputs['shifted_prev_mix']),
        new_post_mix=torch.empty_like(inputs['post_mix']),
        new_comb_res_mix=torch.empty_like(inputs['comb_res_mix']),
    )
    if store_bf16:
        outputs['y_bf16'] = torch.empty_like(inputs['x'])
    if sf_layout != 'bf16':
        outputs['y_fp8'] = torch.empty((tokens, hidden), dtype=torch.float8_e4m3fn, device='cuda')
        if sf_layout == 'token':
            outputs['y_token_sf'] = torch.empty((tokens, 1), dtype=torch.float32, device='cuda')
        elif sf_layout == 'col':
            outputs['y_gemm_sf'] = torch.empty_strided(
                (tokens, hidden // 128), (1, (tokens + 3) // 4 * 4), dtype=torch.int32, device='cuda')
        else:
            rows = (tokens + shared_sf_block_m - 1) // shared_sf_block_m * ((shared_sf_block_m + 127) // 128 * 128)
            outputs['y_routed_sf'] = torch.empty((tokens, hidden // 128), dtype=torch.int32, device='cuda')
            outputs['y_shared_sf'] = torch.empty_strided(
                (rows, hidden // 128), (1, rows), dtype=torch.int32, device='cuda')[:tokens]
            inputs['shared_sf_block_m'] = shared_sf_block_m
    return inputs, outputs, duplicates


def logical_outputs(outputs, shared_sf_block_m=0):
    result = dict(outputs)
    if 'y_shared_sf' in result:
        sf = result['y_shared_sf']
        tokens = sf.size(0)
        token = torch.arange(tokens, device=sf.device)
        index = token % shared_sf_block_m
        rows = token // shared_sf_block_m * ((shared_sf_block_m + 127) // 128 * 128)
        rows += index // 128 * 128 + index % 32 * 4 + index % 128 // 32
        result['y_shared_sf'] = sf.as_strided((sf.stride(1), sf.size(1)), sf.stride())[rows]
    return result


def check_correctness(actual, pytorch_ref, case):
    result = {}
    for name, value in actual.items():
        target = pytorch_ref[name]
        if value.dtype == torch.int32:
            value, target = (v.contiguous().view(torch.uint8) for v in (value, target))
        diff = float(calc_diff(value, target))
        limit = 2e-4 if name == 'y_fp8' else 5e-5
        err = (value.double() - target.double()).abs()
        result[name] = dict(
            calc_diff=diff, limit=limit,
            max_absolute_error=err.max().item(),
            max_relative_error=(err / target.double().abs().clamp_min(1e-8)).max().item(),
        )
        assert diff < limit, f'{case}, {name=}, {diff=}, {limit=}'
    return result


def benchmark_case(inputs, outputs):
    if int(os.environ.get('DG_USE_PPU_TOOLS', 0)):
        print(' > Benchmark skipped (DG_USE_PPU_TOOLS=1)', flush=True)
        return
    kernel_names = 'ppu_mega_mhc'
    if torch.cuda.get_device_capability() == (8, 0):
        # Match the column separator so the main phase excludes pre/finish.
        kernel_names = ('ppu_mega_mhc_tc01_pre ', 'ppu_mega_mhc_tc01 ',
                        'ppu_mega_mhc_tc01_finish ')
    t = bench_kineto(lambda: deep_gemm.mega_mhc(**inputs, **outputs),
                     kernel_names, suppress_kineto_output=True)
    if isinstance(t, tuple):
        assert all(phase_t > 0 for phase_t in t), f'Missing Mega-mHC phase timing: {t}'
        t = sum(t)
    assert t > 0, 'No Mega-mHC kernel timing found'
    tokens, hidden = inputs['x'].shape
    output = 'fp8' if 'y_fp8' in outputs else 'bf16'
    print(f' > Performance (mode=shifted, output={output}, tokens={tokens}, hidden={hidden}, '
          f'store_bf16={int("y_bf16" in outputs)}, '
          f'sinkhorn={inputs["num_sinkhorn_iters"]}): '
          f'{t * 1e6:.2f} us', flush=True)


def check_interface():
    inputs, outputs, _ = make_case()
    checks = []

    def rejects(name, supplied, exception=RuntimeError):
        try:
            deep_gemm.mega_mhc(**supplied)
        except exception:
            checks.append(name)
        else:
            raise AssertionError(name + ' was accepted')

    arguments = dict(inputs, **outputs)
    rejects('required_output', {k: v for k, v in arguments.items() if k != 'y_bf16'})
    rejects('output_dtype', dict(arguments, y_bf16=outputs['y_bf16'].float()))
    rejects('output_alias', dict(arguments, y_bf16=inputs['x']))
    rejects('overlapping_outputs', dict(arguments, new_prev_mix=outputs['new_post_mix']))
    rejects('noncontiguous_output', dict(arguments, y_bf16=torch.empty(
        (5120, 64), dtype=torch.bfloat16, device='cuda').t()))
    rejects('invalid_iterations', dict(arguments, num_sinkhorn_iters=0))
    rejects('invalid_epsilon', dict(arguments, rmsnorm_eps=0.0))
    rejects('nonfinite_scale', dict(arguments, rmsnorm_scale=float('nan')))
    if torch.cuda.get_device_capability() == (8, 0):
        for store_bf16 in (False, True):
            fp8_inputs, fp8_outputs, _ = make_case(sf_layout='col', store_bf16=store_bf16)
            rejects(f'fp8_output_store_bf16_{store_bf16}', dict(fp8_inputs, **fp8_outputs))
    empty_inputs, empty_outputs, _ = make_case(0)
    deep_gemm.mega_mhc(**empty_inputs, **empty_outputs)
    checks.append('empty_input')
    return checks


@torch.no_grad()
def check_graph():
    cases = []
    for tokens in (7, 37):
        inputs, outputs, _ = make_case(tokens)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            deep_gemm.mega_mhc(**inputs, **outputs)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            deep_gemm.mega_mhc(**inputs, **outputs)
        # Grow eager workspace after capture; the graph must retain its addresses.
        larger_inputs, larger_outputs, _ = make_case(257)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            deep_gemm.mega_mhc(**larger_inputs, **larger_outputs)
        stream.synchronize()
        cases.append((tokens, inputs, outputs, stream, graph))
    for seed in (2027, 2028):
        expected = []
        for tokens, inputs, outputs, stream, graph in cases:
            updated, _, _ = make_case(tokens, seed=seed)
            for name, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    value.copy_(updated[name])
            expected.append(mhc_reference(**inputs))
            for value in outputs.values():
                value.zero_()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
        torch.cuda.synchronize()
        for (_, _, outputs, _, _), reference in zip(cases, expected):
            for name, value in outputs.items():
                assert torch.isfinite(value).all(), name
                assert calc_diff(value, reference[name]) < 5e-5, (seed, name)


@torch.no_grad()
def check_layout(tokens):
    inputs, outputs, _ = make_case(tokens)
    hidden = inputs['x'].size(1)
    token = torch.arange(tokens, device='cuda')
    route = torch.arange(4, device='cuda')
    inputs['x'].zero_()
    inputs['x'][token, (token * 131 + 31) % hidden] = 0.5
    inputs['residual'].zero_()
    inputs['residual'][token[:, None], route, (token[:, None] * 67 + route * 129) % hidden] = (
        (route + 1).bfloat16() / 4)
    inputs['post_mix'].copy_(((route + 1).float() / 8).view(1, 4, 1))
    inputs['shifted_prev_mix'].copy_(((4 - route).float() / 8).view(1, 4, 1))
    inputs['comb_res_mix'].copy_(torch.arange(16, device='cuda').view(1, 4, 4).float() / 16)
    n = torch.arange(24, device='cuda')[:, None]
    k = torch.arange(4 * hidden, device='cuda')[None, :]
    # Exact TF32 weights isolate row, route and K-lane mapping from conversion error.
    inputs['fn'].copy_((((n + 1) * (k % 17 + 1) + k // 64 % 13) % 31 - 15).float() / 256)
    inputs['mix_scales'].fill_(0.125)
    inputs['mix_bases'].zero_()
    inputs['rmsnorm_weight'].fill_(1)
    expected = mhc_reference(**inputs)
    deep_gemm.mega_mhc(**inputs, **outputs)
    metrics = check_correctness(outputs, expected, ('layout', tokens))
    assert_bitwise_equal(outputs['new_residual'], expected['new_residual'], 'structured residual')
    for name in ('new_prev_mix', 'new_post_mix', 'new_comb_res_mix'):
        torch.testing.assert_close(outputs[name], expected[name], rtol=2e-5, atol=2e-6)
    return metrics


@torch.no_grad()
def check(tokens, hidden=5120, repeats=30, num_sinkhorn_iters=20, seed=2026,
          benchmark=False, check_accuracy=True, sf_layout='bf16', shared_sf_block_m=224,
          store_bf16=True):
    inputs, outputs, duplicates = make_case(
        tokens, hidden, seed, num_sinkhorn_iters, sf_layout, shared_sf_block_m, store_bf16)
    reference_inputs = {k: v for k, v in inputs.items() if k != 'shared_sf_block_m'}
    expected = mhc_reference(**reference_inputs, sf_layout=sf_layout) if check_accuracy else None
    deep_gemm.mega_mhc(**inputs, **outputs)
    torch.cuda.synchronize()
    if not check_accuracy:
        if os.environ.get('HGGC_WARM_UP') and sf_layout != 'bf16' and store_bf16:
            fp8_only = {k: v for k, v in outputs.items() if k != 'y_bf16'}
            deep_gemm.mega_mhc(**inputs, **fp8_only)
        print('Passed without acc_check', flush=True)
        if benchmark:
            benchmark_case(inputs, outputs)
        return
    actual = logical_outputs(outputs, shared_sf_block_m)
    result = check_correctness(actual, expected, ('shifted', sf_layout, tokens, hidden))
    if sf_layout == 'token' and store_bf16:
        fp8, sf = fp8_reference(outputs['y_bf16'], per_token=True)
        assert_bitwise_equal(outputs['y_fp8'], fp8, 'Per-token FP8 rounding')
        assert_bitwise_equal(outputs['y_token_sf'], sf, 'Per-token scale')
    saved = {name: value.clone() for name, value in actual.items()}
    for _ in range(repeats):
        deep_gemm.mega_mhc(**inputs, **outputs)
        for name, value in logical_outputs(outputs, shared_sf_block_m).items():
            assert_bitwise_equal(value, saved[name], f'Deterministic invocation: {name}')
    for name, value in logical_outputs(outputs, shared_sf_block_m).items():
        for token_idx in duplicates:
            assert_bitwise_equal(value[token_idx], value[0], f'{name}, token={token_idx}')
    if sf_layout != 'bf16' and store_bf16:
        fp8_only = {k: v for k, v in outputs.items() if k != 'y_bf16'}
        deep_gemm.mega_mhc(**inputs, **fp8_only)
        for name, value in logical_outputs(fp8_only, shared_sf_block_m).items():
            assert_bitwise_equal(value, saved[name], f'FP8-only: {name}')
    print(json.dumps(dict(
        tokens=tokens, hidden=hidden, shifted=True,
        num_sinkhorn_iters=num_sinkhorn_iters, sf_layout=sf_layout,
        store_bf16=bool(store_bf16), outputs=result,
    )), flush=True)
    if benchmark:
        benchmark_case(inputs, outputs)
    return result


def test_mega_mhc(benchmark=False, sf_layout=None, store_bf16=True):
    if sf_layout in (None, 'bf16'):
        check_interface()
        check_graph()
        check_layout(65)
    layouts = ('bf16',) if torch.cuda.get_device_capability() == (8, 0) else ('bf16', 'col', 'extra', 'token')
    for layout in ((sf_layout,) if sf_layout else layouts):
        for tokens in (1, 64, 65, 153, 200, 512, 520, 529, 624, 1016, 1024, 1025, 1248,
                       3834, 4095, 4096, 5334, 8192, 8193, 9984, 10321, 32768):
            check(tokens, benchmark=benchmark, sf_layout=layout, store_bf16=store_bf16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens', type=int, help='Run one token count instead of the full suite')
    parser.add_argument('--hidden', type=int, choices=(5120,), default=5120)
    parser.add_argument('--sinkhorn-iters', type=int, default=20)
    parser.add_argument('--sf-layout', choices=('bf16', 'col', 'extra', 'token'))
    parser.add_argument('--store-bf16', type=int, choices=(0, 1), default=1)
    parser.add_argument('--benchmark', action='store_true', help='Run the shared kernel benchmark after correctness checks')
    args = parser.parse_args()
    if args.tokens is None and args.sinkhorn_iters != 20:
        parser.error('Case options require --tokens')
    if not args.store_bf16 and args.sf_layout in (None, 'bf16'):
        parser.error('--store-bf16=0 requires an FP8 --sf-layout')
    print('device', torch.cuda.get_device_name(), 'visible', torch.cuda.device_count(), flush=True)
    if args.tokens is None:
        test_mega_mhc(benchmark=args.benchmark, sf_layout=args.sf_layout, store_bf16=args.store_bf16)
    else:
        check(args.tokens, args.hidden,
              num_sinkhorn_iters=args.sinkhorn_iters, benchmark=args.benchmark,
              sf_layout=args.sf_layout or 'bf16', store_bf16=args.store_bf16)
