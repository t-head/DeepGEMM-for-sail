"""Exercise the public INT8 MegaMoE API against an independent Torch reference.

Run on PPU SM80 with a power-of-two number of idle devices, for example:
    env -u DG_USE_MOE_DYNAMIC_TILE HGGC_DEVICE_GRAPH=0 \
        torchrun --standalone --nproc_per_node=8 tests/test_mega_moe_int8.py --graph

Use --hidden 4096 --intermediate 2048 --topk 6 --experts 16 for a larger shape.
No private test extension or Python JIT is required.
"""

import argparse
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist

import deep_gemm as dg


def activation_reference(gate, up, activation='swiglu', activation_clamp=None,
                         situ_beta=None, situ_linear_beta=None, activation_alpha=None):
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


def run(args):
    assert 'DG_USE_MOE_DYNAMIC_TILE' not in os.environ, 'UT must remove DG_USE_MOE_DYNAMIC_TILE'
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    group = dist.group.WORLD
    rank, ranks = group.rank(), group.size()
    experts = args.experts or max(8, ranks * 2)
    assert experts % ranks == 0 and args.topk <= experts
    hidden, intermediate = args.hidden, args.intermediate
    capacity = (max(args.tokens) + 31) // 32 * 32
    capacity = max(32, capacity)
    # Every rank independently constructs the same raw expert weights. Only its
    # local shard is passed to the implementation; reference routing is unfused.
    torch.manual_seed(1234)
    w1 = torch.randint(-8, 9, (experts, 2 * intermediate, hidden), dtype=torch.int8, device='cuda')
    w2 = torch.randint(-8, 9, (experts, hidden, intermediate), dtype=torch.int8, device='cuda')
    s1 = torch.rand(experts, 2 * intermediate, device='cuda') * 0.02 + 0.01
    s2 = torch.rand(experts, hidden, device='cuda') * 0.02 + 0.01
    local = slice(rank * experts // ranks, (rank + 1) * experts // ranks)
    weights = dg.transform_weights_for_int8_mega_moe((w1[local], s1[local]), (w2[local], s2[local]))
    buffer = dg.get_symm_buffer_for_int8_mega_moe(group, experts, capacity, args.topk, hidden, intermediate)
    # A smaller alias inherits the allocation's fixed communication offsets.
    alias = dg.mega.Int8SymmBuffer(group, experts, 32, args.topk, hidden, intermediate, base=buffer)
    torch.manual_seed(931 + rank)
    y = torch.empty((capacity, hidden), dtype=torch.bfloat16, device='cuda')

    def execute(x, ids, routes, activation, workspace=buffer):
        dg.int8_mega_moe_pre_dispatch(x, ids, routes, workspace.x, workspace.x_sf,
                                    workspace.topk_idx, workspace.topk_weights, len(x))
        dg.int8_int8_mega_moe(y[:workspace.num_max_tokens_per_rank], *weights, workspace,
                            expected_m=args.expected_m, **activation)

    def inputs(tokens, mode):
        x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device='cuda')
        ids = torch.rand(tokens, experts, device='cuda').argsort(1)[:, :args.topk].int().contiguous()
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

    def check(label, x, ids, routes, activation, workspace=buffer):
        torch.cuda.synchronize()
        expected = reference(x, ids, routes, w1, s1, w2, s2,
                             workspace.num_max_tokens_per_rank, activation)
        error = assert_output(y[:len(expected)], expected)
        errors = torch.tensor(error, device='cuda')
        dist.all_reduce(errors, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(json.dumps(dict(case=label, ep=ranks, tokens=len(x), hidden=hidden,
                                  intermediate=intermediate, max_relative_l2=errors.item())), flush=True)

    for tokens in args.tokens:
        for mode in ('random', 'hot', 'invalid', 'empty', 'empty_rank'):
            x, ids, routes = inputs(tokens, mode)
            execute(x, ids, routes, {})
            check(mode, x, ids, routes, {})

    if ranks > 1:
        x, ids, routes = inputs(0 if rank == 0 else min(33, capacity), 'random')
        execute(x, ids, routes, {})
        check('zero_token_rank', x, ids, routes, {})

    # Vary finite clamp values in one process to catch incomplete JIT/plan keys.
    activations = [dict(activation='swiglu', activation_clamp=v) for v in (0., 1., 7., None, float('inf'))]
    activations += [dict(activation='swiglu_step', activation_clamp=1.),
                    dict(activation='swiglu_oai', activation_clamp=1., activation_alpha=1.3),
                    dict(activation='situ', situ_beta=1.5),
                    dict(activation='situ', situ_beta=1.5, situ_linear_beta=2.)]
    for activation in activations if args.activations else [{}]:
        x, ids, routes = inputs(min(31, capacity), 'random')
        execute(x, ids, routes, activation, alias)
        check(str(activation) + '/alias', x, ids, routes, activation, alias)

    x, ids, routes = inputs(min(33, capacity), 'random')
    expect_error(lambda: dg.int8_mega_moe_pre_dispatch(
        x, ids, routes, buffer.x, alias.x_sf, buffer.topk_idx, buffer.topk_weights, len(x)))
    expect_error(lambda: dg.int8_mega_moe_pre_dispatch(
        x, ids, routes, buffer.x, buffer.x_sf, buffer.topk_idx, buffer.topk_weights, len(x) + 1))
    expect_error(lambda: dg.int8_mega_moe_pre_dispatch(
        x, ids.long(), routes, buffer.x, buffer.x_sf, buffer.topk_idx, buffer.topk_weights, len(x)))
    expect_error(lambda: dg.int8_int8_mega_moe(y, *weights, buffer, activation_clamp=-1.))
    expect_error(lambda: dg.int8_int8_mega_moe(y, *weights, buffer, fast_math=False))

    if args.graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graphs = []
        graph_activations = [dict(activation_clamp=1.), dict(activation_clamp=7.)]
        with torch.cuda.stream(stream):
            for activation in graph_activations:
                execute(x, ids, routes, activation)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    execute(x, ids, routes, activation)
                graphs.append(graph)
        torch.cuda.current_stream().wait_stream(stream)
        for iteration in range(8):
            for dst, src in zip((x, ids, routes), inputs(len(x), ('random', 'hot', 'empty', 'empty_rank')[iteration % 4])):
                dst.copy_(src)
            which = iteration % len(graphs)
            graphs[which].replay()
            check(f'graph/{iteration}', x, ids, routes, graph_activations[which])
        del graphs, graph
    torch.cuda.synchronize()
    alias.destroy()
    buffer.destroy()
    dist.destroy_process_group()
    if rank == 0:
        print('PASS: public INT8 MegaMoE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=int, nargs='+', default=[0, 1, 31, 32, 33, 65])
    parser.add_argument('--hidden', type=int, default=2048)
    parser.add_argument('--intermediate', type=int, default=256)
    parser.add_argument('--experts', type=int)
    parser.add_argument('--topk', type=int, default=4)
    parser.add_argument('--expected-m', type=int, help='Override the GEMM scheduling hint')
    parser.add_argument('--activations', action='store_true')
    parser.add_argument('--graph', action='store_true')
    run(parser.parse_args())
