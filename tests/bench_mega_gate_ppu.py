"""PPU GEMM + Top-k comparison. All timings are microseconds; JIT is warmed up."""
import argparse
import json
import os
import statistics
import torch
import torch.nn.functional as F
import deep_gemm
from deep_gemm.testing import bench_kineto
from test_mega_gate import make_inputs, make_outputs, run_mega_gate, check_topk_weights


def graph_bench(fn, repeats=40, samples=7):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            fn()
    for _ in range(3):
        graph.replay()
    times = []
    for _ in range(samples):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / repeats)
    return dict(median_us=statistics.median(times), min_us=min(times), max_us=max(times))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--tokens', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=5120)
    parser.add_argument('--experts', type=int, default=384)
    parser.add_argument('--topk', type=int, default=6, choices=range(1, 33))
    parser.add_argument('--split-k', type=int, default=2, choices=(1, 2, 4, 8))
    parser.add_argument('--topk-threads', type=int, default=128, choices=(32, 64, 128, 256, 512))
    args = parser.parse_args()
    torch.manual_seed(0)
    os.environ['DG_MEGA_GATE_SPLIT_K'] = str(args.split_k)
    os.environ['DG_MEGA_GATE_TOPK_THREADS'] = str(args.topk_threads)
    inputs = make_inputs(args.tokens, args.hidden, args.experts, with_physical_map=False)
    outputs = make_outputs(inputs, num_topk=args.topk)
    logits = torch.empty((args.tokens, args.experts), dtype=torch.float32, device='cuda')
    reference = {}
    native_outputs = make_outputs(inputs, num_topk=args.topk)

    def gemm():
        deep_gemm.acblaslt_gemm_nt(inputs['x'], inputs['weight'], logits)

    def topk():
        scores = F.softplus(logits).sqrt()
        idx = (scores + inputs['bias']).topk(args.topk, dim=-1).indices
        selected = scores.gather(1, idx)
        reference['topk_idx'] = idx
        reference['topk_weights'] = selected / (selected.sum(-1, keepdim=True) + 1e-20) * 1.5
        reference['unmapped_topk_idx'] = None

    def unfused():
        gemm()
        topk()

    def native_topk():
        deep_gemm.deep_gemm_cpp._mega_gate_topk(
            logits, inputs['bias'], native_outputs['topk_idx'], native_outputs['topk_weights'], args.topk, 1.5)

    def native_unfused():
        gemm()
        native_topk()

    def fused():
        run_mega_gate(inputs, outputs, num_topk=args.topk)

    unfused()
    fused()
    diff = check_topk_weights(inputs, outputs, logits, num_topk=args.topk)
    check_topk_weights(inputs, reference, logits, num_topk=args.topk)
    assert torch.equal(outputs['topk_idx'], reference['topk_idx'])
    native_unfused()
    check_topk_weights(inputs, native_outputs, logits, num_topk=args.topk)
    assert torch.equal(outputs['topk_idx'], native_outputs['topk_idx'])
    result = dict(device=torch.cuda.get_device_name(), shape=[args.tokens, args.hidden, args.experts], split_k=args.split_k,
                  topk=args.topk, topk_threads=args.topk_threads, bias=True, physical_map=False, calc_diff=diff,
                  timings={})
    for name, fn in [('acblaslt_gemm', gemm), ('torch_topk_pipeline', topk),
                     ('acblaslt_plus_torch_topk', unfused),
                     ('native_fused_topk', native_topk), ('acblaslt_plus_fused_topk', native_unfused),
                     ('mega_gate', fused)]:
        result['timings'][name] = graph_bench(fn)
        print(name, result['timings'][name], flush=True)
    if args.profile:
        for flush in (False, True):
            result['timings'][f'mega_gate_kernel_flush_l2_{flush}'] = bench_kineto(
                fused, 'ppu_bf16_mega_gate', num_tests=10,
                suppress_kineto_output=True, flush_l2=flush) * 1e6
        # Same 8 GB/32-bit write volume as bench_kineto; a different dtype lets
        # the profiler distinguish cache flushing from the int counter reset.
        flush_cache = torch.empty(int(8e9 // 4), dtype=torch.float32, device='cuda')
        for name, fn, names in [
            ('mega_gate', fused, ('ppu_bf16_mega_gate', 'FillFunctor<int>')),
            ('acblaslt_plus_fused_topk', native_unfused,
             ('gemm_ktype', 'reduction_dtype', 'ppu_mega_gate_topk')),
        ]:
            for flush in (False, True):
                def measured():
                    if flush:
                        flush_cache.zero_()
                    fn()
                times = bench_kineto(measured, names, num_tests=20,
                                    suppress_kineto_output=True, flush_l2=False)
                assert times[0] > 0 and times[-1] > 0, (names, times)
                result['timings'][f'{name}_kernel_sum_flush_l2_{flush}'] = sum(times) * 1e6
                result['timings'][f'{name}_kernel_parts_flush_l2_{flush}'] = dict(zip(names, [t * 1e6 for t in times]))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                native_unfused()
                fused()
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=30,
                                       max_name_column_width=120), flush=True)
        prof.export_chrome_trace(args.output + '.trace.json')
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
