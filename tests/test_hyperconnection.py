"""PPU prenorm correctness with upstream input dtypes, reference and tolerance."""

import argparse
import json
from pathlib import Path

import torch
import deep_gemm

from utils import calc_diff, construct, set_acc_check, set_ref_backend


def test_hc_prenorm_gemm(ms=(13, 137, 4096, 8192), ks=(28672, 7680, 7168)):
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_acc_check(True)
    set_ref_backend("device")
    rows = []
    try:
        for m in ms:
            for k in ks:
                a, b, d, s, ref_d, ref_s = construct(m, k, 24, torch.float32)
                assert a.dtype == torch.bfloat16 and b.dtype == torch.float32
                # Catch regression to BF16-generated weights even if the kernel passes.
                assert torch.any(b != b.bfloat16().float()), "B contains only BF16 values"
                # PPU reduces split-K internally; its explicit output has one slice.
                for num_splits in (None, 1):
                    # A missing write must not inherit an earlier call's correct value.
                    d.fill_(float("nan"))
                    s.fill_(float("nan"))
                    output_d = d if num_splits is None else d.unsqueeze(0)
                    output_s = s if num_splits is None else s.unsqueeze(0)
                    deep_gemm.tf32_hc_prenorm_gemm(
                        a, b, output_d, output_s, num_splits=num_splits
                    )
                    diff_d = float(calc_diff(d, ref_d))
                    diff_s = float(calc_diff(s, ref_s))
                    row = dict(m=m, n=24, k=k, num_splits=num_splits,
                               diff_d=diff_d, diff_s=diff_s)
                    print(json.dumps(row), flush=True)
                    assert torch.isfinite(d).all() and torch.isfinite(s).all(), row
                    assert diff_d < 1e-8 and diff_s < 1e-8, row
                    rows.append(row)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[13, 137, 4096, 8192])
    parser.add_argument("--k", type=int, nargs="+", default=[28672, 7680, 7168])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(0)
    rows = test_hc_prenorm_gemm(args.m, args.k)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(rows=rows), indent=2))
    print(f"ALL {len(rows)} CASES PASSED", flush=True)
