"""Mega-mHC versus the official PyTorch reference: V4.1 Flash shifted (H5120)
and GLM non-shifted (H4096)."""
import argparse
import json
import os
import torch
import deep_gemm
from deep_gemm import bench_kineto
from deep_gemm.testing import assert_bitwise_equal, calc_diff


# Source: third-party/tilelang_ops/ref_mhc.py


def _sinkhorn_reference(value: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    value = value.softmax(dim=-1) + eps
    value = value / (value.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        value = value / (value.sum(dim=-1, keepdim=True) + eps)
        value = value / (value.sum(dim=-2, keepdim=True) + eps)
    return value


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
    return {
        'new_residual': new_residual,
        'new_prev_mix': new_prev_mix,
        'new_post_mix': new_post_mix,
        'new_comb_res_mix': new_comb_res_mix,
        'y_bf16': y_bf16,
    }


@torch.no_grad()
def mhc_reference_non_shift(
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
) -> dict[str, torch.Tensor]:
    """GLM H4096 pre-norm path: FP32 sum-of-squares, BF16-rounded operand."""
    new_residual = x.float().unsqueeze(1) * post_mix
    new_residual += torch.einsum(
        'tij,tih->tjh', comb_res_mix, residual.float())
    new_residual = new_residual.bfloat16()

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

    # The sum of squares comes from the FP32 weighted sum while the operand
    # being normalized is its BF16 rounding, exactly as
    # mhc_pre_big_fuse_with_norm_tilelang does. Rounding the sum of squares too
    # scales y by ~1.5e-3 on near-constant inputs, so this asymmetry is
    # intentional and must not be "fixed".
    mixed = (new_residual.float() * new_prev_mix).sum(dim=1)
    y_bf16 = (
        mixed.bfloat16().float()
        * torch.rsqrt(mixed.square().mean(1) + rmsnorm_eps).unsqueeze(1)
        * (rmsnorm_weight.float() * rmsnorm_scale)
    ).bfloat16()
    return {
        'new_residual': new_residual,
        'new_prev_mix': new_prev_mix,
        'new_post_mix': new_post_mix,
        'new_comb_res_mix': new_comb_res_mix,
        'y_bf16': y_bf16,
    }


# Source: tests/test_mega_mhc.py


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
        num_sinkhorn_iters=10,
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


def make_case(tokens=64, hidden=5120, seed=2026, num_sinkhorn_iters=20):
    inputs, duplicates = make_inputs(tokens, hidden, seed)
    inputs['num_sinkhorn_iters'] = num_sinkhorn_iters
    outputs = dict(
        new_residual=torch.empty_like(inputs['residual']),
        new_prev_mix=torch.empty_like(inputs['shifted_prev_mix']),
        new_post_mix=torch.empty_like(inputs['post_mix']),
        new_comb_res_mix=torch.empty_like(inputs['comb_res_mix']),
        y_bf16=torch.empty_like(inputs['x']),
    )
    return inputs, outputs, duplicates


def check_correctness(actual, pytorch_ref, case):
    result = {}
    for name, value in actual.items():
        target = pytorch_ref[name]
        diff = float(calc_diff(value, target))
        limit = 5e-5
        err = (value.double() - target.double()).abs()
        result[name] = dict(
            calc_diff=diff, limit=limit,
            max_absolute_error=err.max().item(),
            max_relative_error=(err / target.double().abs().clamp_min(1e-8)).max().item(),
        )
        assert diff < limit, f'{case}, {name=}, {diff=}, {limit=}'
    return result


def benchmark_case(inputs, outputs, kernel='ppu_mega_mhc'):
    if int(os.environ.get('DG_USE_PPU_TOOLS', 0)):
        print(' > Benchmark skipped (DG_USE_PPU_TOOLS=1)', flush=True)
        return
    t = bench_kineto(lambda: deep_gemm.mega_mhc(**inputs, **outputs),
                     kernel, suppress_kineto_output=True)
    assert t > 0, 'No Mega-mHC kernel timing found'
    tokens, hidden = inputs['x'].shape
    mode = 'shifted' if inputs.get('shifted_prev_mix') is not None else 'non_shift'
    print(f' > Performance (mode={mode}, output=bf16, tokens={tokens}, hidden={hidden}, '
          f'sinkhorn={inputs["num_sinkhorn_iters"]}): '
          f'{t * 1e6:.2f} us', flush=True)


@torch.no_grad()
def check(tokens, hidden=5120, repeats=30, num_sinkhorn_iters=20, seed=2026,
          benchmark=False, check_accuracy=True):
    inputs, outputs, duplicates = make_case(tokens, hidden, seed, num_sinkhorn_iters)
    expected = mhc_reference(**inputs) if check_accuracy else None
    deep_gemm.mega_mhc(**inputs, **outputs)
    torch.cuda.synchronize()
    if not check_accuracy:
        print('Passed without acc_check', flush=True)
        if benchmark:
            benchmark_case(inputs, outputs)
        return
    result = check_correctness(outputs, expected, ('shifted', 'bf16', tokens, hidden))
    saved = {name: value.clone() for name, value in outputs.items()}
    for _ in range(repeats):
        deep_gemm.mega_mhc(**inputs, **outputs)
        for name, value in outputs.items():
            assert_bitwise_equal(value, saved[name], f'Deterministic invocation: {name}')
    for name, value in outputs.items():
        for token_idx in duplicates:
            assert_bitwise_equal(value[token_idx], value[0], f'{name}, token={token_idx}')
    print(json.dumps(dict(
        tokens=tokens, hidden=hidden, shifted=True,
        num_sinkhorn_iters=num_sinkhorn_iters, outputs=result,
    )), flush=True)
    if benchmark:
        benchmark_case(inputs, outputs)
    return result


def make_non_shift_inputs(num_tokens, hidden=4096, seed=2026, constant=False):
    inputs, duplicates = make_inputs(num_tokens, hidden, seed)
    inputs.pop('shifted_prev_mix')
    inputs['hc_post_scale'] = 2.0
    inputs['rmsnorm_scale'] = 1.0
    inputs['num_sinkhorn_iters'] = 20
    if constant:
        # Near-constant rows keep the weighted sum off the BF16 grid, which is
        # where a BF16 sum-of-squares would visibly disagree with the reference.
        gen = torch.Generator(device='cuda').manual_seed(seed + 1)
        hc_mult = inputs['hc_mult']
        inputs['x'] = torch.full((num_tokens, hidden), 1.0037, dtype=torch.bfloat16, device='cuda')
        inputs['residual'] = torch.full(
            (num_tokens, hc_mult, hidden), 1.0037, dtype=torch.bfloat16, device='cuda')
        inputs['post_mix'] = (
            torch.rand((num_tokens, hc_mult, 1), device='cuda', generator=gen).sigmoid() * 2)
        comb = torch.rand((num_tokens, hc_mult, hc_mult), device='cuda', generator=gen)
        for _ in range(3):
            comb /= comb.sum(-1, keepdim=True)
            comb /= comb.sum(-2, keepdim=True)
        inputs['comb_res_mix'] = comb
        duplicates = ()
    return inputs, duplicates


@torch.no_grad()
def check_non_shift(tokens, hidden=4096, repeats=5, seed=2026, constant=False,
                    benchmark=False):
    inputs, duplicates = make_non_shift_inputs(tokens, hidden, seed, constant)
    outputs = dict(
        new_residual=torch.empty_like(inputs['residual']),
        new_prev_mix=torch.empty((tokens, inputs['hc_mult'], 1), device='cuda', dtype=torch.float),
        new_post_mix=torch.empty_like(inputs['post_mix']),
        new_comb_res_mix=torch.empty_like(inputs['comb_res_mix']),
        y_bf16=torch.empty_like(inputs['x']),
    )
    expected = mhc_reference_non_shift(**inputs)
    deep_gemm.mega_mhc(**inputs, shifted_prev_mix=None, **outputs)
    torch.cuda.synchronize()
    case = ('non_shift', 'bf16', tokens, hidden, 'constant' if constant else 'random')
    result = check_correctness(outputs, expected, case)

    # Without a caller-provided pre buffer the kernel uses internal scratch and
    # must produce the same result.
    scratch = dict(
        new_residual=torch.empty_like(inputs['residual']),
        new_prev_mix=None,
        new_post_mix=torch.empty_like(inputs['post_mix']),
        new_comb_res_mix=torch.empty_like(inputs['comb_res_mix']),
        y_bf16=torch.empty_like(inputs['x']),
    )
    deep_gemm.mega_mhc(**inputs, shifted_prev_mix=None, **scratch)
    for name in ('new_residual', 'new_post_mix', 'new_comb_res_mix', 'y_bf16'):
        assert_bitwise_equal(scratch[name], outputs[name], f'Scratch pre buffer: {name}, {case}')

    if constant:
        # calc_diff alone cannot separate the two sum-of-squares semantics: a
        # whole-row scale (1+delta) only costs about delta**2/2 (~2e-6 here)
        # against the 5e-5 limit, and fewer than one BF16 ulp changes. Pin the
        # semantics against an explicit FP32-denominator prediction instead.
        mixed = (outputs['new_residual'].float() * outputs['new_prev_mix']).sum(1)
        mixed_bf16 = mixed.bfloat16().float()
        weight = inputs['rmsnorm_weight'].float() * inputs['rmsnorm_scale']
        inv_fp32 = torch.rsqrt(
            mixed.square().mean(1) + inputs['rmsnorm_eps']).unsqueeze(1)
        inv_bf16 = torch.rsqrt(
            mixed_bf16.square().mean(1) + inputs['rmsnorm_eps']).unsqueeze(1)
        fp32_denom = (mixed_bf16 * inv_fp32 * weight).bfloat16()
        bf16_denom = (mixed_bf16 * inv_bf16 * weight).bfloat16()
        fp32_diff = float(calc_diff(outputs['y_bf16'], fp32_denom))
        bf16_diff = float(calc_diff(outputs['y_bf16'], bf16_denom))
        assert fp32_diff < 1e-7, f'FP32 sum-of-squares, {case}, {fp32_diff=}'
        assert bf16_diff > 100 * fp32_diff and bf16_diff > 1e-7, (
            f'Constant case no longer discriminates the sum-of-squares semantics, '
            f'{case}, {fp32_diff=}, {bf16_diff=}')

    saved = {name: value.clone() for name, value in outputs.items()}
    for _ in range(repeats):
        deep_gemm.mega_mhc(**inputs, shifted_prev_mix=None, **outputs)
        for name, value in outputs.items():
            assert_bitwise_equal(value, saved[name], f'Deterministic invocation: {name}')
    for name, value in outputs.items():
        for token_idx in duplicates:
            assert_bitwise_equal(value[token_idx], value[0], f'{name}, token={token_idx}')
    print(json.dumps(dict(
        tokens=tokens, hidden=hidden, shifted=False, constant=constant, outputs=result,
    )), flush=True)
    if benchmark:
        benchmark_case(inputs, outputs, kernel='ppu_mega_mhc_nonshift')
    return result


def test_mega_mhc_non_shift(benchmark=False, constant=True):
    for tokens in (1, 64, 65, 511, 512, 1024, 1025, 4096, 8192):
        check_non_shift(tokens, benchmark=benchmark)
    if constant:
        check_non_shift(1024, constant=True)


def test_mega_mhc(benchmark=False):
    for tokens in (1, 64, 65, 153, 200, 512, 520, 529, 624, 1016, 1024, 1025, 1248,
                   3834, 4095, 4096, 5334, 8192, 8193, 9984, 10321, 32768):
        check(tokens, benchmark=benchmark)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens', type=int, help='Run one token count instead of the full suite')
    parser.add_argument('--hidden', type=int, choices=(4096, 5120), default=5120)
    parser.add_argument('--sinkhorn-iters', type=int, default=20)
    parser.add_argument('--constant', action='store_true',
                        help='Also run the near-constant-rows case (H4096 only)')
    parser.add_argument('--benchmark', action='store_true', help='Run the shared kernel benchmark after correctness checks')
    args = parser.parse_args()
    if args.tokens is None and args.sinkhorn_iters != 20:
        parser.error('Case options require --tokens')
    if args.hidden == 4096 and args.sinkhorn_iters != 20:
        parser.error('H4096 non-shift requires 20 Sinkhorn iterations')
    print('device', torch.cuda.get_device_name(), 'visible', torch.cuda.device_count(), flush=True)
    if args.tokens is None:
        if args.hidden == 4096:
            test_mega_mhc_non_shift(benchmark=args.benchmark)
        else:
            test_mega_mhc(benchmark=args.benchmark)
    elif args.hidden == 4096:
        check_non_shift(args.tokens, constant=args.constant, benchmark=args.benchmark)
    else:
        check(args.tokens, args.hidden,
              num_sinkhorn_iters=args.sinkhorn_iters, benchmark=args.benchmark)
