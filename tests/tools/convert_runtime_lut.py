#!/usr/bin/env python3
"""Experimental converter from legacy tuner JSON to C++ runtime text LUTs."""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path


KINDS = ("int8", "bf16", "fp8_blockwise", "fp8_channelwise", "fp4")
GEMM_TYPES = ("densegemm", "groupednopad", "groupedmasked")
CONFIG_NAMES = (
    "best_block_m", "best_block_n", "block_k", "warp_m", "warp_n",
    "best_num_stages",
)
FP4_GROUP = re.compile(r"N([0-9]+)_K([0-9]+)_E([0-9]+)(_Masked)?")


def integer(value, label):
    if type(value) is not int or not 0 < value < (1 << 31):
        raise ValueError(f"{label}: expected a positive signed 32-bit integer")
    return value


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def checked_tile(values, label):
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        raise ValueError(f"{label}: expected six tile parameters")
    bm, bn, bk, wm, wn, stages = (
        integer(value, f"{label} tile[{index}]")
        for index, value in enumerate(values)
    )
    if bm % wm or bn % wn:
        raise ValueError(f"{label}: warp dimensions must divide block dimensions")
    return bm, bn, bk, wm, wn, stages


def legacy_tile(config, label):
    if isinstance(config, str):
        config = ast.literal_eval(config)
    if not isinstance(config, dict):
        raise ValueError(f"{label}: config must be a dictionary or its literal string")
    return checked_tile([config[name] for name in CONFIG_NAMES], label)


def legacy_kind(record, fp8_quant_type):
    dtype = record.get("dtype", "int8")
    if dtype in ("int8", "bf16", "fp4"):
        return dtype
    if dtype not in ("fp8", "fp8_blockwise", "fp8_channelwise"):
        raise ValueError(f"unsupported dtype: {dtype!r}")
    quant = record.get("quant_type")
    if quant is not None:
        quant = {"block": "blockwise", "channel": "channelwise"}.get(quant, quant)
        if quant not in ("blockwise", "channelwise"):
            raise ValueError(f"unsupported FP8 quant_type: {quant!r}")
    if dtype != "fp8":
        declared = dtype[len("fp8_"):]
        if quant and quant != declared:
            raise ValueError("FP8 dtype conflicts with quant_type")
        quant = declared
    if quant and fp8_quant_type and quant != fp8_quant_type:
        raise ValueError("--fp8-quant-type conflicts with quant_type")
    quant = quant or fp8_quant_type
    if quant is None:
        raise ValueError("FP8 JSON requires --fp8-quant-type blockwise|channelwise")
    return "fp8_" + quant


def select_gemm(groups, inferred, override, declared=None):
    if declared is not None and declared not in GEMM_TYPES:
        raise ValueError(f"unsupported gemm_type: {declared!r}")
    if override and declared and override != declared:
        raise ValueError("--gemm-type conflicts with the declared GEMM type")
    gemm = override or declared or inferred
    if gemm == "densegemm" and groups != 1:
        raise ValueError("Dense GEMM requires exactly one group")
    return gemm


def add_entry(tables, kind, gemm, m, n, k, groups, tile, label):
    m, n, k, groups = (
        integer(value, f"{label} {name}")
        for name, value in zip(("M", "N", "K", "groups"), (m, n, k, groups))
    )
    if gemm == "densegemm":
        shape = (integer(ceil_div(m, 16) * 16, f"{label} aligned M"), n, k)
    else:
        em = ceil_div(m, groups) if gemm == "groupednopad" else m
        shape = (groups, em, n, k)
    table = tables.setdefault((kind, gemm), {})
    if shape in table and table[shape] != tile:
        raise ValueError(f"{label}: conflicting tiles map to the same LUT key {shape}")
    table[shape] = tile


def convert_json(data, gemm_type=None, fp8_quant_type=None, fp4_masked_m=None):
    tables = {}
    if isinstance(data, list):
        for index, record in enumerate(data):
            label = f"record {index + 1}"
            if not isinstance(record, dict):
                raise ValueError(f"{label}: expected an object")
            groups = integer(record.get("num_groups", 1), f"{label} num_groups")
            nopad = record.get("nopad", False)
            if type(nopad) is not bool:
                raise ValueError(f"{label}: nopad must be a boolean")
            inferred = "groupednopad" if nopad else (
                "densegemm" if groups == 1 else "groupedmasked"
            )
            gemm = select_gemm(groups, inferred, gemm_type, record.get("gemm_type"))
            add_entry(tables, legacy_kind(record, fp8_quant_type), gemm,
                      record["M"], record["N"], record["K"], groups,
                      legacy_tile(record["config"], label), label)
    elif isinstance(data, dict):
        for key, group in data.items():
            match = FP4_GROUP.fullmatch(key)
            if match is None or not isinstance(group, dict):
                raise ValueError(f"invalid FP4 group: {key!r}")
            meta, configs = group["meta"], group["configs"]
            if not isinstance(meta, dict) or not isinstance(configs, dict) or not configs:
                raise ValueError(f"{key}: expected meta and nonempty configs objects")
            n, k, groups = (integer(meta[name], f"{key} {name}")
                            for name in ("N", "K", "num_experts"))
            if (n, k, groups) != tuple(map(int, match.groups()[:3])):
                raise ValueError(f"{key}: group name conflicts with meta dimensions")
            masked = bool(match[4])
            inferred = "groupedmasked" if masked else (
                "densegemm" if groups == 1 else "groupednopad"
            )
            gemm = select_gemm(groups, inferred, gemm_type,
                               "groupedmasked" if masked else None)
            if gemm == "groupedmasked" and fp4_masked_m is None:
                raise ValueError("FP4 Masked JSON requires --fp4-masked-m source-m|expected-m")
            for m_key, config in configs.items():
                if not m_key.isdecimal() or str(int(m_key)) != m_key:
                    raise ValueError(f"{key}: M keys must be canonical positive integer strings")
                m = integer(int(m_key), f"{key} M")
                if gemm == "groupedmasked" and fp4_masked_m == "source-m":
                    topk = integer(meta["topk_experts"], f"{key} topk_experts")
                    m = ceil_div(m * topk, groups)
                if not isinstance(config, (list, tuple)) or len(config) != 8:
                    raise ValueError(f"{key} M={m_key}: expected an eight-element FP4 config")
                tile = checked_tile(config[1:7], f"{key} M={m_key}")
                add_entry(tables, "fp4", gemm, m, n, k, groups, tile, key)
    else:
        raise ValueError("JSON must be a legacy record list or an FP4 group object")
    if not tables:
        raise ValueError("JSON has no tuning records")
    return tables


def serialize_lut(table, gemm):
    names = "M, N, K" if gemm == "densegemm" else "GROUPS, EM_upper, N, K"
    lines = ["# {{" + names + "}, {BM, BN, BK, WM, WN, STAGES}}"]
    for shape, tile in sorted(table.items()):
        lines.append("{{" + ", ".join(map(str, shape)) + "}, {" +
                     ", ".join(map(str, tile)) + "}},")
    return "\n".join(lines) + "\n"


def write_tables(tables, args):
    outputs = [(args.output_dir / f"{args.device_name}_{kind}_{gemm}.lut",
                serialize_lut(table, gemm))
               for (kind, gemm), table in sorted(tables.items())]
    for path, _ in outputs:
        if path.resolve() == args.input.resolve() or (
            path.exists() and os.path.samefile(path, args.input)
        ):
            raise ValueError(f"refusing to overwrite input file: {path}")
        if path.exists() and (not args.force or not path.is_file()):
            raise ValueError(f"output already exists: {path}; use --force to overwrite a file")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, text in outputs:
        with path.open("w" if args.force else "x", encoding="utf-8") as stream:
            stream.write(text)
        print(f"Wrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="legacy tuner JSON file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-name", required=True)
    parser.add_argument("--fp8-quant-type", choices=("blockwise", "channelwise"))
    parser.add_argument("--gemm-type", choices=GEMM_TYPES)
    parser.add_argument("--fp4-masked-m", choices=("source-m", "expected-m"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.device_name.strip() or args.device_name in (".", "..") or any(
            char in args.device_name for char in ("/", "\\", "\x00", "\n", "\r")
        ):
            raise ValueError("--device-name must be a nonempty filename component")
        data = json.loads(args.input.read_text(encoding="utf-8"))
        tables = convert_json(data, args.gemm_type, args.fp8_quant_type, args.fp4_masked_m)
        write_tables(tables, args)
        return 0
    except (OSError, ValueError, TypeError, KeyError, SyntaxError, RecursionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
