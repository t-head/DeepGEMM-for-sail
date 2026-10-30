"""Representative native W4A8 API timings and GPU component traces.

Run this same file against both built packages with PYTHONPATH. Weights use all
E2M1 codes and nonuniform N scales, allowing an independent analytical reference.
The functional tests additionally cover random packed weights and K-group scales.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
import deep_gemm as dg


def problems():
    cases = []
    for groups, n, k, rs in ((28, 6144, 3584, (1, 16, 32, 64, 512, 4096)),
                             (56, 6144, 3584, (4096,)),
                             (13, 320, 768, (1,)), (31, 512, 1024, (23,)),
                             (64, 1024, 512, (49,)), (97, 512, 768, (137,)),
                             (10, 2048, 2048, (1024,)), (17, 1536, 1024, (1024,))):
        for r in rs:
            cases.append(dict(name=f'E{groups}_R{r}_N{n}_K{k}', groups=groups, n=n, k=k, r=r, api='nopad'))
    for pattern in ('balanced', 'random'):
        for api in ('nopad', 'fused'):
            cases.append(dict(name=f'K3_{pattern}_{api}', groups=896, n=768, k=3584,
                              tokens=8192, topk=16, pattern=pattern, api=api))
    cases.append(dict(name='E97_T2048_Top8_Fused', groups=97, n=512, k=768,
                      tokens=2048, topk=8, pattern='balanced', api='fused'))
    cases.append(dict(name='E97_T1733_Top8_Fused', groups=97, n=512, k=768,
                      tokens=1733, topk=8, pattern='balanced', api='fused'))
    return cases


def checksum(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def construct(c, seed):
    torch.manual_seed(seed)
    e, n, k = c['groups'], c['n'], c['k']
    code = (torch.arange(e, device='cuda')[:, None] + torch.arange(n, device='cuda')[None, :]) % 16
    packed = (code.to(torch.uint8) * 17)[:, :, None].expand(e, n, k // 2).contiguous()
    delta = torch.arange(n, device='cuda') % 3 - 1
    raw = (127 + delta).to(torch.uint8)[None, :, None].expand(e, n, k // 32).contiguous()
    rhs = dg.preprocess_weights_and_scales(dg.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw)
    values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], device='cuda')
    expected_weights = values[code] * torch.exp2(delta.float())[None, :]
    if 'r' in c:
        counts = [c['r']] * e
        counts[0] += counts[-1]
        counts[-1] = 0
        m = sum(counts)
        a = torch.randn(m, k, device='cuda').to(torch.float8_e4m3fn)
        scale = torch.rand(m, 1, device='cuda') * .5 + .5
        lhs = a, scale * rhs[2]
        sums = a.float().sum(1) * scale[:, 0]
        rows = torch.tensor(counts, device='cuda', dtype=torch.int32)
        indices = torch.repeat_interleave(torch.arange(e, device='cuda', dtype=torch.int32), rows.long())
    else:
        tokens, topk = c['tokens'], c['topk']
        routes = ((torch.arange(tokens * topk).reshape(tokens, topk) % e).int() if c['pattern'] == 'balanced'
                  else torch.rand(tokens, e).topk(topk, dim=1, sorted=False).indices.int())
        flat = routes.flatten()
        order = flat.argsort(stable=True)
        counts = torch.bincount(flat.long(), minlength=e).tolist()
        source = (order // topk).cuda()
        m = tokens * topk
        a = torch.randn(tokens, k, device='cuda').to(torch.float8_e4m3fn)
        scale = torch.rand(tokens, 1, device='cuda') * .5 + .5
        lhs = a, scale * rhs[2]
        sums = (a.float().sum(1) * scale[:, 0])[source]
        if c['api'] == 'fused':
            config, rows, experts, sorted_tokens, aligned, _, _ = dg.moe_align_block_size(a, rhs[0], routes.cuda())
        else:
            lhs = a.float()[source].to(a.dtype).contiguous(), lhs[1][source].contiguous()
            rows = torch.tensor(counts, device='cuda', dtype=torch.int32)
            indices = flat[order].cuda()
    if c['api'] == 'fused':
        def invoke(out):
            dg.m_grouped_gemm_fp8_fp4_bf16_nt_fused(lhs, rhs[:2], out, rows, experts, sorted_tokens, aligned, config)
    else:
        def invoke(out):
            dg.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(lhs, rhs[:2], out, indices, m_rows=rows)
    outputs = [torch.full((m, n), float('nan'), dtype=torch.bfloat16, device='cuda') for _ in range(2)]
    checksums = dict(a=checksum(lhs[0]), a_scale=checksum(lhs[1]), weights=checksum(rhs[0]),
                     scales=checksum(rhs[1]), counts=counts)
    def check(output):
        dot = norm = 0.
        begin = 0
        for expert, count in enumerate(counts):
            for start in range(begin, begin + count, 1024):
                stop = min(begin + count, start + 1024)
                ref = (sums[start:stop, None] * expected_weights[expert]).bfloat16()
                x, y = output[start:stop].double(), ref.double()
                dot += float((x * y).sum())
                norm += float((x * x + y * y).sum())
            begin += count
        difference = 1 - 2 * dot / norm if norm else 0.
        assert torch.isfinite(output).all() and difference < .003, difference
        return difference
    return invoke, outputs, checksums, check, lhs


def timing(invoke, output):
    for _ in range(12):
        invoke(output)
    torch.cuda.synchronize()
    samples, host = [], []
    for _ in range(15):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        t = time.perf_counter()
        for _ in range(20):
            invoke(output)
        host.append((time.perf_counter() - t) * 1e6 / 20)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 20)
    return dict(samples_us=samples, median_us=statistics.median(samples),
                host_enqueue_samples_us=host, host_enqueue_median_us=statistics.median(host))


def profile(invoke, output, path):
    # Components are measured in the profiler's own timeline, independently of
    # the unprofiled event measurements. Decode and compute can overlap.
    for _ in range(3):
        invoke(output)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            with torch.profiler.record_function('w4a8_measured_api'):
                invoke(output)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--case', action='append', dest='cases')
    parser.add_argument('--baseline-dir', type=Path)
    parser.add_argument('--save-baseline', action='store_true')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.baseline_dir:
        args.baseline_dir.mkdir(parents=True, exist_ok=True)
    records = []
    properties = torch.cuda.get_device_properties(0)
    result = dict(complete=False, label=args.label, package=dg.__file__, torch_version=str(torch.__version__),
                  device=dict(name=properties.name, compute_units=properties.multi_processor_count),
                  environment={key: os.environ.get(key) for key in ('DG_JIT_CACHE_DIR', 'DG_JIT_USE_PCH', 'CUDA_VISIBLE_DEVICES')},
                  configured_sms=dg.get_num_sms(),
                  method=dict(warmups=12, rounds=15, inner=20, ordinary_output_slots=2, graph_output_slots=1), records=records)
    for index, case in enumerate(problems()):
        if args.cases and case['name'] not in args.cases:
            continue
        invoke, outputs, checksums, check, lhs = construct(case, 2026103000 + index)
        invoke(outputs[0])
        oracle = check(outputs[0])
        if args.baseline_dir:
            baseline_file = args.baseline_dir / (case['name'] + '.pt')
            if args.save_baseline:
                torch.save(dict(output=outputs[0].cpu(), checksums=checksums), baseline_file)
            else:
                base = torch.load(baseline_file, weights_only=True)
                assert checksums == base['checksums'], case['name']
                torch.testing.assert_close(outputs[0].cpu(), base['output'], atol=.125, rtol=.02)
                del base
        ordinary = [timing(invoke, out) for out in outputs]
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                invoke(outputs[0])
        capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            invoke(outputs[0])
        graph_result = timing(lambda _: graph.replay(), outputs[0])
        original = outputs[0].clone()
        lhs[1].neg_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(outputs[0], -original)
        lhs[1].neg_()
        record = dict(**case, checksum=checksums, independent_difference=oracle, ordinary=ordinary,
                      ordinary_us=statistics.mean(x['median_us'] for x in ordinary),
                      host_enqueue_us=statistics.mean(x['host_enqueue_median_us'] for x in ordinary), graph=graph_result,
                      graph_us=graph_result['median_us'], changed_input_graph_replay=True)
        if args.profile:
            trace = args.output.parent / (args.label + '-' + case['name'] + '.trace.json')
            profile(invoke, outputs[0], trace)
            record['trace'] = trace.name
        records.append(record)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({key: record[key] for key in ('name','ordinary_us','graph_us','host_enqueue_us','independent_difference')}), flush=True)
        del invoke, outputs, check, lhs, graph, original, capture_stream
        gc.collect()
        torch.cuda.empty_cache()
    result['complete'] = True
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
