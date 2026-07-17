#!/usr/bin/env python3
import json
import ast

def lut_to_py_file(lut: dict, lut_name: str, out_path: str):
    keys   = sorted(lut)
    m_w = max(len(str(k[0])) for k in keys)
    n_w = max(len(str(k[1])) for k in keys)
    k_w = max(len(str(k[2])) for k in keys)
    blk_m_w = max(len(str(v[0])) for v in lut.values())
    blk_n_w = max(len(str(v[1])) for v in lut.values())
    blk_k_w = max(len(str(v[2])) for v in lut.values())
    wm_w    = max(len(str(v[3])) for v in lut.values())
    wn_w    = max(len(str(v[4])) for v in lut.values())
    stg_w   = max(len(str(v[5])) for v in lut.values())

    lines = [f'# -*- coding: utf-8 -*-',
             f'# Codegen By Python',
             f'{lut_name} = {{',
             f'    # (M,  N,  K) : (blkM, blkN, blkK, warpM, warpN, stages)']

    for (m, n, k), (bm, bn, bk, wm, wn, stg) in [(k, lut[k]) for k in keys]:
        lines.append(
            f'    ({m:>{m_w}}, {n:>{n_w}}, {k:>{k_w}}): '
            f'({bm:>{blk_m_w}}, {bn:>{blk_n_w}}, {bk:>{blk_k_w}}, '
            f'{wm:>{wm_w}}, {wn:>{wn_w}}, {stg:>{stg_w}}),'
        )

    lines += ['}']
    pathlib.Path(out_path).write_text('\n'.join(lines))

def load_json_lut(src_path: str):
    with open(src_path) as f:
        records = json.load(f)
    lut_result = {}
    for rec in records:
        key = (rec["M"], rec["N"], rec["K"])
        cfg = ast.literal_eval(rec["config"])
        lut_result[key] = (
            cfg['best_block_m'],
            cfg['best_block_n'],
            cfg['block_k'],
            cfg['warp_m'],
            cfg['warp_n'],
            cfg['best_num_stages']
        )
    return lut_result


if __name__ == '__main__':
    import sys, pathlib
    if len(sys.argv) != 4:
        sys.exit("Usage: python build_lookup_table.py source.json lut_name out_file.py")
    src   = sys.argv[1]
    name  = sys.argv[2]
    out   = sys.argv[3] if len(sys.argv) > 3 else 'codegen_gemm_lut.py'

    lut = load_json_lut(src)

    lut_to_py_file(lut, name, out)

