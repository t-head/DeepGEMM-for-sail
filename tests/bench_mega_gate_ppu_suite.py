"""Compare every performance shape with acBLASLt + fused Top-k."""
import argparse
import json
import os
from pathlib import Path
import torch
import deep_gemm
from bench_mega_gate_ppu import graph_bench
from test_mega_gate import enumerate_cases, make_inputs, make_outputs, run_mega_gate, check_topk_weights


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--skip-unaligned-experts', action='store_true')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    os.environ['DG_MEGA_GATE_SPLIT_K'] = '0'
    cases = list(dict.fromkeys(
        [(t, h, e, 6) for t, h, e in enumerate_cases()] +
        [(16, 5120, 384, k) for k in (1, 2, 4, 8, 9, 16, 32)] +
        [(16, 4096, 128, 8), (17, 7168, 256, 2), (128, 4096, 512, 16)]))
    if args.quick:
        cases = [(16, 5120, 384, k) for k in (1, 6, 16, 32)] + [
            (128, 4096, 512, 16), (2048, 5120, 384, 6),
            (32768, 7168, 512, 6), (8192, 4352, 264, 6)]
    rows = []
    for t, h, e, k in cases:
        if args.skip_unaligned_experts and e % 128:
            continue
        torch.manual_seed(0)
        inp = make_inputs(t, h, e, with_physical_map=False)
        out = make_outputs(inp, num_topk=k)
        ref = make_outputs(inp, num_topk=k)
        logits = torch.empty((t, e), device='cuda', dtype=torch.float32)
        def gemm():
            deep_gemm.acblaslt_gemm_nt(inp['x'], inp['weight'], logits)
        def unfused():
            gemm()
            deep_gemm.deep_gemm_cpp._mega_gate_topk(logits, inp['bias'], ref['topk_idx'], ref['topk_weights'], k, 1.5)
        def fused():
            run_mega_gate(inp, out, num_topk=k)
        gemm(); fused()
        diff = check_topk_weights(inp, out, logits, num_topk=k)
        candidates = []
        for threads in (64, 128, 256):
            os.environ['DG_MEGA_GATE_TOPK_THREADS'] = str(threads)
            unfused()
            check_topk_weights(inp, ref, logits, num_topk=k)
            candidates.append(dict(threads=threads, **graph_bench(unfused, repeats=20, samples=5)))
        best = min(candidates, key=lambda r: r['median_us'])
        timing = graph_bench(fused, repeats=20, samples=7)
        row = dict(shape=[t, h, e], topk=k, split_k=2 if t < 512 and h >= 2048 else 1,
                   calc_diff=diff, gemm=graph_bench(gemm, repeats=20, samples=5),
                   baseline=best, baseline_candidates=candidates, mega_gate=timing,
                   speedup=best['median_us']/timing['median_us'])
        if hasattr(deep_gemm.deep_gemm_cpp, '_mega_gate_config'):
            row['config'] = deep_gemm.deep_gemm_cpp._mega_gate_config(t, h, e, k)
            row['split_k'] = row['config'][1]
        rows.append(row)
        Path(args.output).write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
    print(f'PASS correctness: {len(rows)} cases; faster: {sum(r["speedup"] > 1 for r in rows)}', flush=True)


if __name__ == '__main__':
    main()
