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
    t = bench_kineto(lambda: deep_gemm.mega_mhc(**inputs, **outputs),
                     'ppu_mega_mhc', suppress_kineto_output=True)
    assert t > 0, 'No Mega-mHC kernel timing found'
    tokens, hidden = inputs['x'].shape
    output = 'fp8' if 'y_fp8' in outputs else 'bf16'
    print(f' > Performance (mode=shifted, output={output}, tokens={tokens}, hidden={hidden}, '
          f'store_bf16={int("y_bf16" in outputs)}, '
          f'sinkhorn={inputs["num_sinkhorn_iters"]}): '
          f'{t * 1e6:.2f} us', flush=True)


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
    for layout in ((sf_layout,) if sf_layout else ('bf16', 'col', 'extra', 'token')):
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
