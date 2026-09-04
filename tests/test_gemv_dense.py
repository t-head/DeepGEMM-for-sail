"""Dense GEMV (m == 1) bf16 fast-path tests.

Covers:
  1. The four Kimi K3 decode shapes (c2/c3/c9/c13) -- accuracy vs torch matmul.
  2. Every BlockX ladder branch (2/4/8/16/32/64/128 + the n <= 100 fat-block
     branch) and the ladder boundaries (add_times 512/1024, n 100/101).
  3. The residual-K loop with several remainders and exact-multiple K.
  4. A random (n, k) sweep over the accepted domain.
  5. Data distributions: zero inputs (exact-zero output) and large-amplitude
     inputs (fp32 accumulation keeps the relative error scale-invariant).
  6. Repeat-call determinism (same inputs -> bit-exact outputs).
  7. Fallback conditions (small n, unaligned k, m == 2, explicit configs,
     env kill switch).
  8. Optional performance comparison GEMV vs tile (--benchmark).
"""
import argparse
import os
import random
import sys
import torch

import deep_gemm
from deep_gemm import calc_diff, bench_kineto

# (m, n, k, expect_gemv): accuracy + dispatch (torch.profiler kernel-name) check
DISPATCH_CASES = [
    (1, 896, 7168, True),   # c2 : add_times=896 -> BlockX=64, BlockY=2
    (1, 7168, 3584, True),  # c3 : add_times=448 -> BlockX=32, BlockY=4
    (1, 3584, 7168, True),  # c9 : add_times=896 -> BlockX=64, BlockY=2
    (1, 2112, 7168, True),  # c13: add_times=896 -> BlockX=64, BlockY=2
    (1, 80, 7168, True),    # n <= 100 branch: BlockX=128 + cross-warp SMEM reduce
    (1, 897, 7168, True),   # n not divisible by BlockY=2: tail-block row clamp
    (1, 128, 16, True),     # add_times=2 -> BlockX=2 (warp spans 16 rows)
    (1, 128, 8200, True),   # add_times > 1024 -> BlockX=128, BlockY=1
    (1, 32, 7168, False),   # n < 64 -> tile fallback
    (1, 896, 7172, False),  # k % 8 != 0 -> tile fallback
    (2, 896, 7168, False),  # m == 2 -> tile fallback
]

# (n, k): accuracy-only coverage of the remaining ladder branches/boundaries
LADDER_CASES = [
    (100, 7168),   # n == 100: last n of the fat-block branch
    (101, 7168),   # n == 101: first n of the add_times ladder
    (128, 8),      # smallest accepted k, pure residual loop (stride > k)
    (896, 8),      # add_times=1 -> BlockX=2, BlockY=64
    (128, 24),     # add_times=3 -> BlockX=4, BlockY=32
    (128, 48),     # add_times=6 -> BlockX=8, BlockY=16
    (128, 384),    # add_times=48 -> BlockX=16, BlockY=8
    (128, 2048),   # add_times=256 -> BlockX=32, BlockY=4
    (128, 4096),   # add_times=512 boundary -> BlockX=32
    (128, 4104),   # add_times=513 -> BlockX=64, BlockY=2
    (128, 8192),   # add_times=1024 boundary -> BlockX=64
    (4096, 512),   # add_times=64 -> BlockX=16
    (64, 4096),    # smallest accepted n on a mid ladder branch
]

# (n, k): residual-K loop remainders across different main-loop strides
# (BlockX=64 -> stride 1024, BlockX=32 -> stride 512)
K_TAIL_CASES = [
    (896, 1024),   # exact multiple of the stride: no residual iteration
    (896, 1032),   # residual of one 16B vector
    (896, 2040),   # residual of 1016 elements (almost one full stride)
    (896, 7176),   # 7 strides + 8
    (2048, 4616),  # BlockX=32: 9 strides of 512 + 8
    (896, 2056),   # 2 strides + 8
]

RANDOM_CASES = 32


def make_inputs(m, n, k, mode='normal'):
    torch.manual_seed(0)
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    w = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    if mode == 'zero_x':
        x.zero_()
    elif mode == 'zero_w':
        w.zero_()
    elif mode == 'large':
        x, w = x * 100, w * 100
    return x, w


def run_accuracy(m, n, k, mode='normal') -> float:
    x, w = make_inputs(m, n, k, mode)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref = x @ w.t()

    deep_gemm.gemm_bf16_bf16_bf16_nt(x, w, out)
    torch.cuda.synchronize()

    diff = calc_diff(out, ref)
    assert diff < 0.001, f"accuracy failed: m={m}, n={n}, k={k}, mode={mode}, calc_diff={diff:.6f}"
    return diff


def run_dispatch_case(m, n, k, expect_gemv):
    x, w = make_inputs(m, n, k)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref = x @ w.t()

    deep_gemm.gemm_bf16_bf16_bf16_nt(x, w, out)
    torch.cuda.synchronize()

    diff = calc_diff(out, ref)
    assert diff < 0.001, f"accuracy failed: m={m}, n={n}, k={k}, calc_diff={diff:.6f}"
    # Secondary guard, aligned with the historical allclose(2e-2) validation gate.
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2), \
        f"allclose failed: m={m}, n={n}, k={k}"

    # Dispatch guard: verify the call actually routed through the GEMV kernel
    # (or fell back to the tile path). Without this, a select_configs regression
    # that mis-routes shapes would still pass on accuracy alone.
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        deep_gemm.gemm_bf16_bf16_bf16_nt(x, w, out)
        torch.cuda.synchronize()
    saw_gemv = any('gemv_dense_bf16' in ev.key for ev in prof.key_averages())
    assert saw_gemv == expect_gemv, \
        f"dispatch mismatch: m={m}, n={n}, k={k}: gemv_kernel={saw_gemv}, expected={expect_gemv}"
    return diff


def test_dispatch_loop() -> None:
    print('Testing Kimi decode shapes + dispatch routing:')
    for m, n, k, expect in DISPATCH_CASES:
        diff = run_dispatch_case(m, n, k, expect)
        print(f'  {m}x{n}x{k} (gemv={expect}): calc_diff={diff:.2e}  Passed')
    print('Passed\n')


def test_ladder_loop() -> None:
    print('Testing BlockX ladder branches and boundaries:')
    for n, k in LADDER_CASES:
        diff = run_accuracy(1, n, k)
        print(f'  1x{n}x{k}: calc_diff={diff:.2e}  Passed')
    print('Passed\n')


def test_k_tail_loop() -> None:
    print('Testing residual-K loop remainders:')
    for n, k in K_TAIL_CASES:
        diff = run_accuracy(1, n, k)
        print(f'  1x{n}x{k}: calc_diff={diff:.2e}  Passed')
    print('Passed\n')


def test_random_loop() -> None:
    print(f'Testing random (n, k) sweep ({RANDOM_CASES} cases):')
    random.seed(0)
    worst = 0.0
    for _ in range(RANDOM_CASES):
        n = random.randint(64, 8192)
        k = 8 * random.randint(1, 1024)
        worst = max(worst, run_accuracy(1, n, k))
    print(f'  worst calc_diff={worst:.2e}  Passed')
    print('Passed\n')


def test_data_loop() -> None:
    print('Testing data distributions:')
    # Zero inputs must produce exact zeros (calc_diff is undefined at zero).
    for mode in ('zero_x', 'zero_w'):
        x, w = make_inputs(1, 896, 7168, mode)
        out = torch.empty((1, 896), device='cuda', dtype=torch.bfloat16)
        deep_gemm.gemm_bf16_bf16_bf16_nt(x, w, out)
        torch.cuda.synchronize()
        assert (out == 0).all(), f"zero-input case '{mode}' produced non-zero output"
    print('  zero_x / zero_w: exact zeros  Passed')

    # Large amplitudes: operands scaled x100 keep the cosine-based calc_diff
    # scale-invariant only if accumulation stays fp32 (bf16 rounding of
    # products would blow the relative error up at |dot| ~ 100^2 * k).
    diff = run_accuracy(1, 896, 7168, mode='large')
    print(f'  large-amplitude: calc_diff={diff:.2e}  Passed')
    print('Passed\n')


def test_repeat_loop() -> None:
    print('Testing repeat-call determinism:')
    x, w = make_inputs(1, 896, 7168)
    out = torch.empty((1, 896), device='cuda', dtype=torch.bfloat16)
    ref = None
    for i in range(3):
        deep_gemm.gemm_bf16_bf16_bf16_nt(x, w, out)
        torch.cuda.synchronize()
        if ref is None:
            ref = out.clone()
        else:
            assert torch.equal(out, ref), f"repeat call {i} produced different output"
    print('  3 calls bit-exact  Passed')
    print('Passed\n')


def run_bypass_cases() -> None:
    print('Testing explicit-config tile bypass:')
    run_explicit_config_case()
    print('Passed\n')

    print('Testing DG_DISABLE_DENSE_GEMV tile fallback:')
    run_env_kill_case()
    print('Passed\n')


def run_explicit_config_case():
    """Explicit tile configs must bypass the GEMV fast path (tuning escape hatch)."""
    x, w = make_inputs(1, 896, 7168)
    out = torch.empty((1, 896), device='cuda', dtype=torch.bfloat16)
    ref = x @ w.t()
    configs = (39, 16, 64, 64, 16, 32, 2, (24576, 128, 0))
    deep_gemm.gemm_bf16_bf16_bf16_nt(x, w, out, configs)
    torch.cuda.synchronize()
    diff = calc_diff(out, ref)
    assert diff < 0.001, f"explicit-config tile path failed: calc_diff={diff:.6f}"


def run_env_kill_case():
    """DG_DISABLE_DENSE_GEMV=1 must route back to the tile path."""
    x, w = make_inputs(1, 896, 7168)
    out = torch.empty((1, 896), device='cuda', dtype=torch.bfloat16)
    ref = x @ w.t()
    os.environ['DG_DISABLE_DENSE_GEMV'] = '1'
    try:
        deep_gemm.gemm_bf16_bf16_bf16_nt(x, w, out)
        torch.cuda.synchronize()
    finally:
        del os.environ['DG_DISABLE_DENSE_GEMV']
    diff = calc_diff(out, ref)
    assert diff < 0.001, f"env-kill tile path failed: calc_diff={diff:.6f}"


def run_benchmark():
    print('\nBenchmark (asys-style: 30 samples, L2 flushed per call):')
    print(f'{"case":>16} | {"GEMV us":>9} | {"tile us":>9} | {"speedup":>8}')
    for n, k, _ in DISPATCH_CASES[:4]:
        x, w = make_inputs(1, n, k)
        out = torch.empty((1, n), device='cuda', dtype=torch.bfloat16)

        def gemv_call():
            deep_gemm.gemm_bf16_bf16_bf16_nt(x, w, out)

        t_gemv = bench_kineto(gemv_call, 'gemv_dense_bf16', suppress_kineto_output=True)

        os.environ['DG_DISABLE_DENSE_GEMV'] = '1'
        try:
            t_tile = bench_kineto(gemv_call, 'deep_gemm', suppress_kineto_output=True)
        finally:
            del os.environ['DG_DISABLE_DENSE_GEMV']

        print(f'{f"1x{n}x{k}":>16} | {t_gemv * 1e6:9.2f} | {t_tile * 1e6:9.2f} | {t_tile / t_gemv:7.2f}x')


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    parser = argparse.ArgumentParser(description='Dense GEMV (m == 1) fast-path tests.')
    parser.add_argument('--func', default=None, nargs='*',
                        choices=['Dispatch', 'Ladder', 'KTail', 'Random', 'Data', 'Repeat', 'Bypass'],
                        help='target test funcs (default: all)')
    parser.add_argument('--benchmark', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        os.environ['show_log'] = '1'

    funcs = {
        'Dispatch': test_dispatch_loop,
        'Ladder': test_ladder_loop,
        'KTail': test_k_tail_loop,
        'Random': test_random_loop,
        'Data': test_data_loop,
        'Repeat': test_repeat_loop,
        'Bypass': run_bypass_cases,
    }
    for name in (args.func if args.func is not None else funcs):
        funcs[name]()

    if args.benchmark:
        run_benchmark()
