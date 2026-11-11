"""Exercise the public INT8 MegaMoE API against an independent Torch reference.

Run on PPU SM80 with a power-of-two number of idle devices, for example:
    env -u DG_USE_MOE_DYNAMIC_TILE -u HGGC_DEVICE_GRAPH \
        torchrun --standalone --nproc_per_node=8 tests/test_mega_moe_int8.py --graph

Use --hidden 4096 --intermediate 2048 --topk 6 --experts 16 for a larger shape.
Exercise the opt-in EP16 scatter workaround on SM80 with:
    env -u HGGC_DEVICE_GRAPH -u DG_USE_MOE_DYNAMIC_TILE \
        DG_MEGAMOE_SM80_EP16_SCATTER_WAIT=1 \
        torchrun --standalone --nproc_per_node=16 tests/test_mega_moe_int8.py \
        --tokens 8192 --hidden 6144 --intermediate 2048 --experts 256 --topk 8 --continuous-iters 2048
No private test extension or Python JIT is required.
"""

import argparse
import inspect
import json
import os
import subprocess
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

import deep_gemm as dg
from mega_moe_test_utils import check_bf16_offset_views


def test_import_without_symmetric_memory():
    # A fresh interpreter also exercises the top-level MegaMoE exports.
    probe = """\
import sys
import torch.distributed

sys.modules['torch.distributed._symmetric_memory'] = None
from deep_gemm import get_symm_buffer_for_int8_mega_moe, int8_gemm_nt
"""
    subprocess.run([sys.executable, '-c', probe], check=True)


def activation_reference(gate, up, activation='swiglu', activation_clamp=None,
                         situ_beta=4.0, situ_linear_beta=25.0, activation_alpha=None):
    gate, up = gate.float(), up.float()
    if activation == 'situ':
        if situ_linear_beta is not None:
            up = situ_linear_beta * torch.tanh(up / situ_linear_beta)
        value = torch.sigmoid(gate) * (situ_beta * torch.tanh(gate / situ_beta)) * up
    else:
        if activation_clamp is not None:
            up = up.clamp(-activation_clamp, activation_clamp)
            if activation != 'swiglu_step':
                gate = gate.clamp_max(activation_clamp)
        if activation == 'swiglu_oai':
            alpha = 1.702 if activation_alpha is None else activation_alpha
            value = gate * torch.sigmoid(alpha * gate) * (up + 1)
        else:
            value = torch.nn.functional.silu(gate)
            if activation == 'swiglu_step' and activation_clamp is not None:
                value = value.clamp_max(activation_clamp)
            if activation == 'swiglu':
                value = value.bfloat16().float()
            value = value * up
    return value.bfloat16().float()


def reference(x, ids, routes, w1, s1, w2, s2, capacity, activation):
    # Use standard Torch math as an independent oracle for either math policy.
    activation = {key: value for key, value in activation.items() if key != 'fast_math'}
    amax = x.float().abs().amax(1).clamp_min(1e-6)
    scale = amax * (1.0 / 127.0)
    # Match the channelwise dispatch contract: nearest-even INT8 conversion.
    q = (x.float() * (127.0 / amax[:, None])).round().clamp(-127, 127)
    slots = torch.zeros((*ids.shape, x.shape[1]), device=x.device)
    for expert in range(len(w1)):
        token, slot = (ids == expert).nonzero(as_tuple=True)
        if token.numel() == 0:
            continue
        gateup = ((q[token] @ w1[expert].float().T) * scale[token, None]
                  * s1[expert]).bfloat16()
        activated = activation_reference(*gateup.chunk(2, dim=1), **activation)
        a_scale = activated.abs().amax(1).clamp_min(1e-10) / 127
        a_q = (activated / a_scale[:, None]).clamp(-128, 127).to(torch.int8)
        down = ((a_q.float() @ w2[expert].float().T) * a_scale[:, None]
                * s2[expert]).bfloat16().float()
        slots[token, slot] = down
    result = torch.zeros((capacity, x.shape[1]), dtype=torch.bfloat16, device=x.device)
    result[:len(x)] = (slots * routes[..., None]).sum(1).bfloat16()
    return result


def assert_output(actual, expected):
    assert torch.isfinite(actual).all(), 'nonfinite output'
    error = ((actual.float() - expected.float()).square().sum()
             / expected.float().square().sum().clamp_min(1e-30)).sqrt().item()
    # Native sigmoid and BF16 rounding can move INT8 values by one unit.
    assert error < 0.015, f'relative L2 error: {error}'
    zero_rows = expected.abs().amax(1) == 0
    assert torch.count_nonzero(actual[zero_rows]) == 0, 'stale data in an empty output row'
    return error


def expect_error(fn):
    try:
        fn()
    except (RuntimeError, ValueError, TypeError):
        return
    raise AssertionError('invalid input was accepted')


def check_fast_math_policy(group):
    # One route per rank; GEMM1 produces gate=1 and up=1/4 in every channel.
    # This FP32 clamp is near a BF16 rounding boundary for SiLU on SM80.
    # Inspect Stage1's scale before GEMM2 can hide a one-ULP difference.
    hidden, intermediate = 2048, 256
    buf = dg.get_symm_buffer_for_int8_mega_moe(group, group.size(), 16, 1, hidden, intermediate)
    w1 = torch.zeros((1, 2 * intermediate, hidden), device='cuda', dtype=torch.int8)
    w1[:, :, 0] = 1
    s1 = torch.ones((1, 2 * intermediate), device='cuda')
    s1[:, intermediate:] = 0.25
    w2 = torch.zeros((1, hidden, intermediate), device='cuda', dtype=torch.int8)
    weights = dg.transform_weights_for_int8_mega_moe((w1, s1), (w2, torch.ones((1, hidden), device='cuda')))
    x = torch.zeros((1, hidden), dtype=torch.bfloat16, device='cuda')
    x[:, 0] = 1
    ids = torch.full((1, 1), group.rank(), dtype=torch.int64, device='cuda')
    routes = torch.ones((1, 1), device='cuda')
    y = torch.empty((16, hidden), dtype=x.dtype, device=x.device)
    limit = float.fromhex('0x1.003408p-1')
    scales = []
    for fast in (True, False, True):
        dg.int8_mega_moe_pre_dispatch(x, ids, routes, buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights, 1)
        dg.int8_int8_mega_moe(y, *weights, buf, activation_clamp=limit, fast_math=fast)
        scales.append(buf.l2_acts_sf[0].item())
    gated = torch.nn.functional.silu(torch.tensor(limit, dtype=torch.float64)).bfloat16().item()
    expected = torch.tensor(gated * 0.25 / 127)
    torch.testing.assert_close(torch.tensor(scales[1]), expected, rtol=1e-6, atol=0)
    assert scales[0] == scales[2], 'switching back must reuse the fast policy'
    if group.rank() == 0:
        print(json.dumps(dict(case='fast_math/rounding_boundary', stage1_scales=scales)), flush=True)
    buf.destroy()


def run(args):
    fp4_params = {name: param for name, param in inspect.signature(dg.fp4_fp4_mega_moe).parameters.items()
                  if name not in ('l1_weights_replicas', 'l2_weights_replicas')}
    int8_params = inspect.signature(dg.int8_int8_mega_moe).parameters
    assert list(int8_params) == list(fp4_params), 'public parameter names/order must match MXFP4 without replica banks'
    for name, param in int8_params.items():
        assert param.kind == fp4_params[name].kind, name
        if name != 'recipe':
            assert param.default == fp4_params[name].default, name
    assert 'DG_USE_MOE_DYNAMIC_TILE' not in os.environ, 'UT must remove DG_USE_MOE_DYNAMIC_TILE'
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    group = dist.group.WORLD
    rank, ranks = group.rank(), group.size()
    experts = args.experts or max(8, ranks * 2)
    assert experts % ranks == 0 and args.topk <= experts
    hidden, intermediate = args.hidden, args.intermediate
    requested_capacity = max(1, max(args.tokens))
    capacity = (requested_capacity + 15) // 16 * 16
    # Every rank independently constructs the same raw expert weights. Only its
    # local shard is passed to the implementation; reference routing is unfused.
    torch.manual_seed(1234)
    w1 = torch.randint(-8, 9, (experts, 2 * intermediate, hidden), dtype=torch.int8, device='cuda')
    w2 = torch.randint(-8, 9, (experts, hidden, intermediate), dtype=torch.int8, device='cuda')
    s1 = torch.rand(experts, 2 * intermediate, device='cuda') * 0.02 + 0.01
    s2 = torch.rand(experts, hidden, device='cuda') * 0.02 + 0.01
    local = slice(rank * experts // ranks, (rank + 1) * experts // ranks)
    weights = dg.transform_weights_for_int8_mega_moe((w1[local], s1[local]), (w2[local], s2[local]))
    buffer = dg.get_symm_buffer_for_int8_mega_moe(group, experts, requested_capacity, args.topk, hidden, intermediate)
    assert buffer.num_max_tokens_per_rank == capacity, 'capacity must round to 16 tokens'
    compact_rows = ranks * capacity * min(args.topk, experts // ranks)
    assert buffer.l1_acts.shape == (compact_rows, hidden)
    assert buffer.l2_acts.shape == (compact_rows, intermediate)
    # A smaller alias inherits the allocation's fixed communication offsets.
    alias = dg.mega.Int8SymmBuffer(group, experts, 16, args.topk, hidden, intermediate, base=buffer)
    assert buffer.topk_idx.dtype == alias.topk_idx.dtype == torch.int64
    torch.manual_seed(931 + rank)
    y = torch.empty((capacity, hidden), dtype=torch.bfloat16, device='cuda')

    def execute(x, ids, routes, activation, workspace=buffer, output=y, stats=None):
        dg.int8_mega_moe_pre_dispatch(x, ids, routes, workspace.x, workspace.x_sf,
                                    workspace.topk_idx, workspace.topk_weights, len(x))
        dg.int8_int8_mega_moe(output[:workspace.num_max_tokens_per_rank], *weights, workspace,
                            cumulative_local_expert_recv_stats=stats, **activation)

    stats = torch.full((experts // ranks,), 7, dtype=torch.int32, device='cuda')
    expected_stats = stats.clone()

    def check_stats(ids):
        valid = ids[(ids >= 0) & (ids < experts)].long()
        counts = torch.bincount(valid, minlength=experts).int()
        dist.all_reduce(counts)
        expected_stats.add_(counts[local])
        assert torch.equal(stats, expected_stats), (stats, expected_stats)

    def inputs(tokens, mode):
        x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device='cuda')
        ids = torch.rand(tokens, experts, device='cuda').argsort(1)[:, :args.topk]
        ids = ids.to(getattr(torch, args.index_dtype)).contiguous()
        routes = torch.rand(tokens, args.topk, device='cuda')
        if mode == 'hot':
            ids.copy_(torch.arange(args.topk, device='cuda').expand_as(ids))
        elif mode == 'empty':
            ids.fill_(-1)
        elif mode == 'empty_rank' and rank == 0:
            ids.fill_(-1)
        elif mode == 'invalid' and tokens:
            ids[0, 0] = experts + 1
            ids[-1, -1] = -1
        if tokens > 1:
            x[0].zero_()
            routes[1].zero_()
        return x, ids, routes

    def check(label, x, ids, routes, activation, workspace=buffer, output=y):
        torch.cuda.synchronize()
        expected = reference(x, ids, routes, w1, s1, w2, s2,
                             min(workspace.num_max_tokens_per_rank, len(output)), activation)
        error = assert_output(output[:len(expected)], expected)
        errors = torch.tensor(error, device='cuda')
        dist.all_reduce(errors, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(json.dumps(dict(case=label, ep=ranks, tokens=len(x), hidden=hidden,
                                  intermediate=intermediate, topk=args.topk, index_dtype=str(ids.dtype),
                                  capacity=workspace.num_max_tokens_per_rank,
                                  max_relative_l2=errors.item())), flush=True)

    for tokens in args.tokens:
        for mode in ('random', 'hot', 'invalid', 'empty', 'empty_rank'):
            x, ids, routes = inputs(tokens, mode)
            execute(x, ids, routes, {})
            check(mode, x, ids, routes, {})

    # Exercise column splitting, partial token groups and the 128-row policy
    # boundary independently of the symmetric allocation's rounded capacity.
    for output_rows in (1, 31, 32, 33, 64, 127, 128, 129):
        if output_rows > capacity:
            continue
        mode = 'invalid' if output_rows % 2 else 'empty_rank'
        x, ids, routes = inputs(min(17, output_rows), mode)
        visible_output = y[:output_rows]
        execute(x, ids, routes, {}, output=visible_output)
        check(f'reduce/output_rows={output_rows}', x, ids, routes, {}, output=visible_output)
        if output_rows == 1:
            # Cover offset stores at column-part boundaries, including guards.
            def forward_reduce_offset(source, output):
                execute(source, ids, routes, {}, output=output)
                return output
            check_bf16_offset_views(x, visible_output.clone(), forward_reduce_offset)

    if args.sync_modes:
        original = os.environ.get('DG_MEGAMOE_SPLIT_SYNC')
        try:
            for split_sync, mode in ((0, 'hot'), (1, 'empty_rank'), (0, 'random')):
                os.environ['DG_MEGAMOE_SPLIT_SYNC'] = str(split_sync)
                x, ids, routes = inputs(min(31, capacity), mode)
                if rank == 0:
                    time.sleep(0.05)
                visible_output = y[:len(x)]
                execute(x, ids, routes, {}, output=visible_output, stats=stats)
                check(f'split_sync/{split_sync}/{mode}', x, ids, routes, {}, output=visible_output)
                check_stats(ids)
        finally:
            if original is None:
                os.environ.pop('DG_MEGAMOE_SPLIT_SYNC', None)
            else:
                os.environ['DG_MEGAMOE_SPLIT_SYNC'] = original

    # The public INT8 output uses capacity rows. Exercise both paths with the same shape.
    x, ids, routes = inputs(capacity, 'random')
    def forward_offset(source, output):
        execute(source, ids, routes, {}, output=output)
        return output
    aligned = forward_offset(x, y).clone()
    check_bf16_offset_views(x, aligned, forward_offset)

    if args.continuous_iters:
        batches = [inputs(max(args.tokens), 'random') for _ in range(min(2, args.continuous_iters))]
        outputs = [y, torch.empty_like(y)] if len(batches) == 2 else [y]
        for i in range(args.continuous_iters):
            index = i % len(batches)
            execute(*batches[index], {}, output=outputs[index])
        for index, (batch, output) in enumerate(zip(batches, outputs)):
            check(f'continuous_random/{index}', *batch, {}, output=output)

    if ranks > 1:
        x, ids, routes = inputs(0 if rank == 0 else min(33, capacity), 'random')
        execute(x, ids, routes, {})
        check('zero_token_rank', x, ids, routes, {})

    # Accumulate from all ranks, preserving the caller's initial values. Hot
    # routing includes zero-weight routes; aliases and empty batches count once.
    for index, mode in enumerate(('hot', 'invalid', 'empty', 'empty_rank', 'random')):
        workspace = alias if index % 2 else buffer
        tokens = min(31, workspace.num_max_tokens_per_rank) if mode != 'random' else 0
        x, ids, routes = inputs(tokens, mode)
        execute(x, ids, routes, {}, workspace, stats=stats)
        check('stats/' + mode, x, ids, routes, {}, workspace)
        check_stats(ids)

    # Both input widths publish normalized int64 IDs, including masked and padded slots.
    x, ids, routes = inputs(min(15, capacity), 'random')
    for dtype in (torch.int32, torch.int64):
        typed_ids = ids.to(dtype)
        execute(x, typed_ids, routes, {}, stats=stats)
        check('route_dtype/' + str(dtype), x, typed_ids, routes, {})
        torch.testing.assert_close(buffer.topk_idx[:len(x)], typed_ids.long())
        assert (buffer.topk_idx[len(x):] == -1).all()
        check_stats(typed_ids)
    wide_ids = ids.long()
    for index, value in enumerate((2**32, 2**32 + 1, -(2**32) + 1, 2**63 - 1, -(2**63))):
        wide_ids[-1 - index, 0] = value
    execute(x, wide_ids, routes, {}, stats=stats)
    check('route_dtype/int64_out_of_range', x, wide_ids, routes, {})
    normalized = wide_ids.masked_fill((wide_ids < 0) | (wide_ids >= experts), -1)
    torch.testing.assert_close(buffer.topk_idx[:len(x)], normalized)
    check_stats(wide_ids)

    # Vary finite clamp values in one process to catch incomplete JIT/plan keys.
    activations = [dict(activation='swiglu', activation_clamp=v) for v in (0., 1., 7., None, float('inf'))]
    activations += [dict(activation='swiglu_step', activation_clamp=1.),
                    dict(activation='swiglu_oai', activation_clamp=1., activation_alpha=1.3),
                    dict(activation='situ'),
                    dict(activation='situ', situ_beta=4., situ_linear_beta=25.),
                    dict(activation='situ', situ_beta=1.5),
                    dict(activation='situ', situ_linear_beta=None),
                    dict(activation='situ', situ_beta=1.5, situ_linear_beta=2.)]
    for activation in activations if args.activations else [{}]:
        x, ids, routes = inputs(min(31, alias.num_max_tokens_per_rank), 'random')
        # Switch policies on the same buffer to exercise the cached split plan.
        for fast_math in (True, False):
            config = dict(activation, fast_math=fast_math)
            execute(x, ids, routes, config, alias)
            check(str(config) + '/alias', x, ids, routes, config, alias)

    x, ids, routes = inputs(min(33, capacity), 'random')
    expect_error(lambda: dg.int8_mega_moe_pre_dispatch(
        x, ids, routes, buffer.x, alias.x_sf, buffer.topk_idx, buffer.topk_weights, len(x)))
    expect_error(lambda: dg.int8_mega_moe_pre_dispatch(
        x, ids, routes, buffer.x, buffer.x_sf, buffer.topk_idx, buffer.topk_weights, len(x) + 1))
    expect_error(lambda: dg.int8_mega_moe_pre_dispatch(
        x, ids.float(), routes, buffer.x, buffer.x_sf, buffer.topk_idx, buffer.topk_weights, len(x)))
    expect_error(lambda: dg.get_symm_buffer_for_int8_mega_moe(
        group, experts, capacity, 33, hidden, intermediate))
    expect_error(lambda: dg.get_symm_buffer_for_int8_mega_moe(
        group, experts, capacity, args.topk, 512, intermediate))
    expect_error(lambda: dg.int8_int8_mega_moe(y, *weights, buffer, expected_m=1))
    expect_error(lambda: dg.int8_int8_mega_moe(y, *weights, buffer, activation_clamp=-1.))
    expect_error(lambda: dg.int8_int8_mega_moe(y, *weights, buffer, situ_beta=1.5))
    expect_error(lambda: dg.int8_int8_mega_moe(y, *weights, buffer, situ_linear_beta=2.))
    expect_error(lambda: dg.int8_int8_mega_moe(y, *weights, buffer,
                                             cumulative_local_expert_recv_stats=stats.long()))
    expect_error(lambda: dg.int8_int8_mega_moe(y, *weights, buffer,
                                             cumulative_local_expert_recv_stats=stats[:0]))

    if args.graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graphs = []
        graph_activations = [dict(activation_clamp=1., fast_math=mode) for mode in (True, False)]
        with torch.cuda.stream(stream):
            for activation in graph_activations:
                execute(x, ids, routes, activation, stats=stats)
                check_stats(ids)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    execute(x, ids, routes, activation, stats=stats)
                graphs.append(graph)
        torch.cuda.current_stream().wait_stream(stream)
        assert torch.equal(stats, expected_stats), 'Graph capture must not count as execution'
        for iteration in range(8):
            for dst, src in zip((x, ids, routes), inputs(len(x), ('random', 'hot', 'empty', 'empty_rank')[iteration % 4])):
                dst.copy_(src)
            which = iteration % len(graphs)
            graphs[which].replay()
            check(f'graph/{iteration}', x, ids, routes, graph_activations[which])
            check_stats(ids)
        del graphs, graph
    if args.activations:
        check_fast_math_policy(group)
    torch.cuda.synchronize()
    alias.destroy()
    buffer.destroy()
    dist.destroy_process_group()
    if rank == 0:
        print('PASS: public INT8 MegaMoE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=int, nargs='+', default=[0, 1, 31, 32, 33, 65, 129])
    parser.add_argument('--hidden', type=int, default=2048)
    parser.add_argument('--intermediate', type=int, default=256)
    parser.add_argument('--experts', type=int)
    parser.add_argument('--topk', type=int, default=4)
    parser.add_argument('--index-dtype', choices=('int32', 'int64'), default='int32')
    parser.add_argument('--continuous-iters', type=int, default=0,
                        help='Enqueue this many forwards at the largest token count before checking the output')
    parser.add_argument('--activations', action='store_true')
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--sync-modes', action='store_true',
                        help='Switch fused/split rank waits on one buffer with a delayed rank')
    args = parser.parse_args()
    if args.continuous_iters < 0:
        parser.error('--continuous-iters must be nonnegative')
    run(args)
