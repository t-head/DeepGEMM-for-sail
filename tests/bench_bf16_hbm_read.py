"""Reproduce cold-cache BF16 NT kernel timing on M890 and NVIDIA cuBLAS.

See docs/performance/bf16_hbm_read.md. No Torch dependency for CSV aggregation.
"""
import argparse
import csv
import ctypes
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone

SHAPES = [(m, n, k) for n, k in (
    (49152, 7168), (7168, 12288), (7168, 8192), (12288, 1536), (2112, 7168)
) for m in (1, 2, 4, 8, 16, 32, 64)]
METRICS = {"ppu": "ppu__time_duration.sum", "cublas": "gpu__time_duration.sum"}
FORMULA = "2*(M*K+N*K); BF16 A/B reads only, excluding output writes"
HERE = Path(__file__).resolve()


def digest(path):
    """Return the SHA-256 hex digest of a file, read in 1 MiB blocks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    """Write JSON atomically, so an interrupted run leaves no partial file."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def command_output(command):
    """Return stripped stdout of a successful command, else None.

    Optional provenance tools are allowed to be absent: a missing executable,
    a launch failure and a timeout all resolve to None instead of aborting the
    surrounding capture.
    """
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def hgcc_executable():
    """Resolve the compiler the runtime uses, not merely one found on PATH.

    Mirrors HGCCCompiler: DG_JIT_HGCC_COMPILER, then DG_HGCC_COMPILER, then the
    SDK root's bin/hgcc, then PATH. Recording a different executable than the one
    that built the measured kernels would mislabel cross-build comparisons.
    """
    for name in ("DG_JIT_HGCC_COMPILER", "DG_HGCC_COMPILER"):
        if os.environ.get(name):
            return os.environ[name]
    sdk = os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME") or "/usr/local/PPU_SDK"
    candidate = Path(sdk) / "bin" / "hgcc"
    return str(candidate) if candidate.is_file() else shutil.which("hgcc")


def header_digest(root):
    """Digest every file under a JIT header root, including relative paths."""
    if not root.is_dir():
        return None
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            h.update(str(path.relative_to(root)).encode())
            h.update(bytes.fromhex(digest(path)))
    return h.hexdigest()


def positive(value):
    """argparse type for an integer greater than zero."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def shape(value):
    """argparse type for a single M,N,K shape."""
    try:
        result = tuple(positive(x) for x in value.split(","))
    except (ValueError, argparse.ArgumentTypeError) as e:
        raise argparse.ArgumentTypeError("expected positive M,N,K") from e
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected M,N,K")
    return result


def selected_shapes(args):
    """Return the requested shapes, defaulting to the report's full set."""
    shapes = args.shape or SHAPES
    if len(set(shapes)) != len(shapes):
        raise ValueError("duplicate shapes")
    return shapes


def loaded_cublas(torch):
    """Fingerprint the cuBLAS libraries this process already loaded.

    Resolve libraries already mapped into this process, rather than a possibly
    different toolkit found through LD_LIBRARY_PATH or find_library().
    """
    libraries = {}
    for name in ("libcublas", "libcublasLt"):
        paths = {line.split(maxsplit=5)[-1] for line in
                 Path("/proc/self/maps").read_text().splitlines()
                 if len(line.split(maxsplit=5)) == 6 and
                 Path(line.split(maxsplit=5)[-1]).name.startswith(name + ".so")}
        if len(paths) > 1 or (name == "libcublas" and not paths):
            raise RuntimeError(f"ambiguous/missing loaded {name}: {paths}")
        if not paths:
            continue
        path = paths.pop()
        lib = ctypes.CDLL(path)
        if name == "libcublas":
            version = ctypes.c_int()
            get_version = lib.cublasGetVersion_v2
            get_version.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
            get_version.restype = ctypes.c_int
            status = get_version(torch.cuda.current_blas_handle(), ctypes.byref(version))
            if status:
                raise RuntimeError(f"cublasGetVersion failed: {status}")
            version = version.value
        else:
            lib.cublasLtGetVersion.restype = ctypes.c_size_t
            version = lib.cublasLtGetVersion()
        libraries[name] = dict(path=path, version_integer=version, sha256=digest(path))
    return libraries


def case(args, dims, seed, capture=False):
    """Run one shape, validate its output, and return its metadata record."""
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        raise RuntimeError("requires the appropriate PPU/CUDA PyTorch environment")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(0)  # Physical selection remains CUDA_VISIBLE_DEVICES.
    prop = torch.cuda.get_device_properties(0)
    if args.backend == "ppu" and (prop.major, prop.minor) != (8, 9):
        raise RuntimeError("this reproduction entry targets M890")
    torch.manual_seed(seed)
    m, n, k = dims
    with torch.inference_mode():
        if args.input_mode == "normal":
            # Match the archived measured report (including its GPU RNG).
            a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
            b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
            ah, bh = a.cpu(), b.cpu()
        else:
            ah = torch.randint(-2, 3, (m, k), dtype=torch.int16).to(torch.bfloat16)
            bh = torch.randint(-2, 3, (n, k), dtype=torch.int16).to(torch.bfloat16)
            a, b = ah.cuda(), bh.cuda()
        reference = F.linear(ah.float(), bh.float())
        errors = []
        def validate(actual):
            """Check one output against the FP32 reference, appending its error.

            Inputs are never regenerated, so every call validates the same
            reference. Integer inputs require exact BF16 equality; random inputs
            use the normalized squared error bound.
            """
            value = actual.cpu().float()
            error = ((value-reference).square().sum() /
                     (value.square().sum()+reference.square().sum()).clamp_min(1e-30)).item()
            errors.append(error)
            passed = (torch.equal(value, reference.to(torch.bfloat16).float())
                      if args.input_mode == "integer" else error < 1e-5)
            if not torch.isfinite(value).all() or not passed:
                raise AssertionError(f"full-output reference mismatch: {dims}, error={error}")
        storage = None
        if args.backend == "ppu":
            import deep_gemm as dg
            storage = torch.full((128 + m*n + 128,), -123.,
                                 device="cuda", dtype=torch.bfloat16)
            out = storage[128:-128].view(m, n)
            def call():
                """Invoke the operator once and return its output tensor."""
                dg.bf16_gemm_nt(a, b, out)
                return out
            module_file = Path(dg.__file__).resolve()
            hgcc = hgcc_executable()
            operator = dict(entry="deep_gemm.bf16_gemm_nt", file=str(module_file),
                            version=getattr(dg, "__version__", None),
                            git_commit=command_output(
                                ["git", "-C", str(module_file.parent), "rev-parse", "HEAD"]),
                            binaries={p.name: digest(p) for p in
                                      module_file.parent.glob("*.so")},
                            jit_headers_sha256=header_digest(
                                module_file.parent / "include/deep_gemm"),
                            compiler_version=command_output([hgcc, "--version"]) if hgcc else None,
                            num_sms=dg.get_num_sms())
        else:
            call = lambda: F.linear(a, b)
            operator = dict(entry="torch.nn.functional.linear")
        actual = call()
        torch.cuda.synchronize()
        if actual.dtype != torch.bfloat16 or tuple(actual.shape) != (m, n):
            raise AssertionError("wrong output dtype/shape")
        validate(actual)
        if args.backend == "cublas":
            operator["libraries"] = loaded_cublas(torch)
        # Finish reference transfers and library hashing before warming, so
        # this CPU work cannot insert a long idle interval before profiling.
        for _ in range(args.warmup):
            actual = call()
        torch.cuda.synchronize()
        if capture:
            if storage is not None:
                actual.fill_(float("nan"))
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            try:
                actual = call()  # Exactly one complete API, with all its kernels.
                torch.cuda.synchronize()
            finally:
                torch.cuda.profiler.stop()
            validate(actual)
        if storage is not None:
            if not ((storage[:128] == -123).all() and (storage[-128:] == -123).all()):
                raise AssertionError("output guard overwritten")
    return dict(shape=list(dims), seed=seed, backend=args.backend, correctness="passed",
                input_mode=args.input_mode, max_normalized_squared_error=max(errors),
                harness_sha256=digest(HERE), capture_requested=capture,
                environment=dict(device=prop.name, compute_capability=[prop.major, prop.minor],
                    num_sms=prop.multi_processor_count, total_memory=prop.total_memory,
                    torch=str(torch.__version__), cuda=torch.version.cuda,
                    python=platform.python_version(), operator=operator,
                    matmul_flags={key: getattr(torch.backends.cuda.matmul, key, None)
                        for key in ("allow_tf32", "allow_bf16_reduced_precision_reduction")},
                    flags={key: os.environ.get(key) for key in (
                        "CUDA_VISIBLE_DEVICES", "PPU_HOME", "PPU_SDK", "DG_JIT_USE_HGRTC",
                        "DG_BF16_ADAPTIVE", "DG_BF16_MEMORY_TUNING",
                        "SAIL_DEEPGEMM_LUT_PATH", "DG_DISABLE_DENSE_GEMV",
                        "DG_DENSE_GEMV_M2_MAX_N", "DG_DENSE_GEMV_MIN_N",
                        "DG_DENSE_GEMV_M2_NPT", "DG_TILE_CANDIDATES",
                        "DG_TILE_COST_RATIO", "DG_M512_OVH_K")}))


def read_kernels(path, backend):
    """Read ACU wide CSV (native nanoseconds), or NCU long/wide CSV with units."""
    metric = METRICS[backend]
    factors = {"ns": .001, "nsecond": .001, "nanosecond": .001,
               "us": 1., "usecond": 1., "microsecond": 1.,
               "ms": 1000., "msecond": 1000., "millisecond": 1000.,
               "s": 1e6, "second": 1e6}
    text = Path(path).read_text(encoding="utf-8-sig")
    if "==ERROR==" in text:
        raise ValueError(f"profiler reported an error: {path}")
    header, unit, kernels, seen = None, None, [], set()
    for row in csv.reader(io.StringIO(text)):
        if "ID" in row and "Kernel Name" in row:
            header = row
            unit = "ns" if backend == "ppu" else None
            continue
        if not header or not row:
            continue
        values = dict(zip(header, row))
        if not values.get("ID", "").isdigit():
            if metric in values and values[metric] in factors:
                unit = values[metric]
            continue
        if "Metric Name" in header:
            if values.get("Metric Name") != metric:
                continue
            raw, row_unit = values.get("Metric Value"), values.get("Metric Unit")
        else:
            raw, row_unit = values.get(metric), unit
        if raw is None or row_unit not in factors:
            raise ValueError(f"missing {metric} value/unit: {path}")
        duration = float(raw.replace(",", "")) * factors[row_unit]
        identity = (values.get("Process ID"), values["ID"])
        if identity in seen or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"duplicate launch or invalid duration: {path}")
        seen.add(identity)
        kernels.append(dict(id=values["ID"], name=values["Kernel Name"], us=duration))
    if not kernels:
        raise ValueError(f"no valid kernel durations in {path}")
    return kernels


def summarize(folder):
    """Reduce one capture folder to medians, bandwidth and the shape peak."""
    folder = Path(folder)
    plan = json.loads((folder / "plan.json").read_text())
    if plan["schema_version"] != 1 or plan["status"] != "complete":
        raise ValueError("capture is incomplete; no performance summary is accepted")
    grouped, environment = {}, None
    for entry in plan["captures"]:
        info = json.loads((folder / (entry["stem"] + ".json")).read_text())
        if (info["shape"] != entry["shape"] or info["seed"] != entry["seed"] or
            info["backend"] != plan["backend"] or info["correctness"] != "passed" or
            not info["capture_requested"] or info["harness_sha256"] != plan["harness_sha256"] or
            info.get("input_mode", "integer") != plan.get("input_mode", "integer")):
            raise ValueError("sample metadata does not match the capture plan")
        if environment is None:
            environment = info["environment"]
        if environment != info["environment"]:
            raise ValueError("device/software environment changed during collection")
        kernels = read_kernels(folder / (entry["stem"] + ".csv"), plan["backend"])
        grouped.setdefault(tuple(entry["shape"]), []).append(
            dict(seed=entry["seed"], sample=entry["sample"], kernels=kernels,
                 latency_us=sum(k["us"] for k in kernels)))
    results = []
    for dims in map(tuple, plan["shapes"]):
        samples = grouped[dims]
        if len(samples) != len(plan["seeds"]) * plan["samples"]:
            raise ValueError(f"incomplete samples: {dims}")
        times = sorted(s["latency_us"] for s in samples)
        m, n, k = dims
        median = statistics.median(times)
        read_bytes = 2 * (m*k + n*k)
        bandwidth = read_bytes / median / 1e6
        results.append(dict(M=m, N=n, K=k, samples=len(times), read_bytes=read_bytes,
            latency_median_us=median, latency_min_us=times[0],
            latency_p90_us=times[math.ceil(.9*len(times))-1],
            logical_read_tb_s=bandwidth, hbm_tb_s=plan["hbm_tb_s"],
            logical_read_util_pct=100*bandwidth/plan["hbm_tb_s"], raw_samples=samples))
    payload = dict(schema_version=1, backend=plan["backend"], formula=FORMULA,
        input_mode=plan.get("input_mode", "integer"),
        harness_sha256=plan["harness_sha256"], environment=environment, results=results,
        peak_of_shape_medians=max(results, key=lambda r: r["logical_read_util_pct"]))
    write_json(folder / "summary.json", payload)
    fields = [k for k in results[0] if k != "raw_samples"]
    with (folder / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Summarized {len(results)} shapes: {folder / 'summary.csv'}")
    return payload


def collect(args):
    """Capture one profiler process per sample and record the full plan."""
    if not math.isfinite(args.hbm_tb_s) or args.hbm_tb_s <= 0:
        raise ValueError("--hbm-tb-s must be finite and positive")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("duplicate seeds")
    profiler = "acu" if args.backend == "ppu" else "ncu"
    executable = shutil.which(profiler)
    if not executable:
        raise RuntimeError(f"{profiler} is not in PATH")
    folder = Path(args.output).resolve()
    shapes = selected_shapes(args)
    folder.mkdir(parents=True, exist_ok=False)
    captures = [dict(stem=f"case-{ci:02d}-round-{ri}-sample-{si}", shape=list(dims),
                     seed=seed+ci, sample=si)
                for ri, seed in enumerate(args.seeds) for ci, dims in enumerate(shapes)
                for si in range(args.samples)]
    plan = dict(schema_version=1, status="running", backend=args.backend,
        hbm_tb_s=args.hbm_tb_s, shapes=shapes, seeds=args.seeds, samples=args.samples,
        warmup=args.warmup, harness_sha256=digest(HERE), captures=captures,
        input_mode=args.input_mode,
        profiler=executable, profiler_version=command_output([executable, "--version"]),
        created_at=datetime.now(timezone.utc).isoformat(), formula=FORMULA)
    write_json(folder / "plan.json", plan)
    monitor = shutil.which("ppu-smi" if args.backend == "ppu" else "nvidia-smi")
    if monitor:
        (folder / "device-before.txt").write_text(command_output([monitor]) or "unavailable\n")
    for index, entry in enumerate(captures):
        stem = folder / entry["stem"]
        command = [executable, "--profile-from-start", "off", "--cache-control", "all",
                   "--clock-control", "none", "--metrics", METRICS[args.backend], "--csv"]
        if profiler == "acu":
            command += ["--csv-file", str(stem) + ".csv"]
        else:
            command += ["--page", "raw", "--print-units", "base",
                        "--log-file", str(stem) + ".csv"]
        command += [sys.executable, str(HERE), "_sample", "--backend", args.backend,
                    "--shape", ",".join(map(str, entry["shape"])),
                    "--seed", str(entry["seed"]), "--warmup", str(args.warmup),
                    "--input-mode", args.input_mode,
                    "--output", str(stem) + ".json"]
        write_json(Path(str(stem) + ".command.json"), command)
        print(f"[{index+1}/{len(captures)}] {entry['stem']} {entry['shape']}", flush=True)
        with Path(str(stem) + ".log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=args.timeout, env=dict(os.environ, LC_ALL="C"))
        read_kernels(Path(str(stem) + ".csv"), args.backend)  # Fail at first bad capture.
    plan["status"] = "complete"
    if monitor:
        (folder / "device-after.txt").write_text(command_output([monitor]) or "unavailable\n")
    write_json(folder / "plan.json", plan)
    summarize(folder)


def compare(args):
    """Match two summaries and emit per-shape latency and bandwidth rows."""
    ppu, gpu = [json.loads(Path(path).read_text()) for path in (args.ppu, args.cublas)]
    if (ppu["backend"], gpu["backend"]) != ("ppu", "cublas"):
        raise ValueError("expected PPU summary followed by cuBLAS summary")
    if ppu["formula"] != gpu["formula"] or ppu["harness_sha256"] != gpu["harness_sha256"]:
        raise ValueError("summaries use different harnesses or traffic definitions")
    if ppu.get("input_mode", "integer") != gpu.get("input_mode", "integer"):
        raise ValueError("summaries use different input distributions")
    key = lambda r: (r["M"], r["N"], r["K"])
    other = {key(r): r for r in gpu["results"]}
    if set(map(key, ppu["results"])) != set(other):
        raise ValueError("summary shape sets differ")
    rows = []
    for p in ppu["results"]:
        g = other[key(p)]
        rows.append(dict(M=p["M"], N=p["N"], K=p["K"],
            ppu_us=p["latency_median_us"], cublas_us=g["latency_median_us"],
            speedup=g["latency_median_us"]/p["latency_median_us"],
            ppu_read_tb_s=p["logical_read_tb_s"], cublas_read_tb_s=g["logical_read_tb_s"],
            ppu_read_util_pct=p["logical_read_util_pct"],
            cublas_read_util_pct=g["logical_read_util_pct"]))
    with Path(args.output).open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Compared {len(rows)} shapes: {args.output}")


def main():
    """Parse the subcommand and dispatch to it."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "collect", "_sample"):
        p = commands.add_parser(name)
        p.add_argument("--backend", choices=METRICS, required=True)
        p.add_argument("--shape", type=shape, action="append", help="M,N,K; default: report's 35 shapes")
        p.add_argument("--warmup", type=positive, default=64)
        p.add_argument("--input-mode", choices=("normal", "integer"), default="normal",
                       help="normal matches measured report; integer enables exact checks")
        p.add_argument("--output", required=True, help="new run directory, or sample JSON")
        if name == "collect":
            p.add_argument("--hbm-tb-s", type=float, required=True)
            p.add_argument("--seeds", type=int, nargs="+", default=[2701, 2702, 2703])
            p.add_argument("--samples", type=positive, default=3, help="per shape and seed")
            p.add_argument("--timeout", type=positive, default=180, help="seconds per profiler process")
        else:
            p.add_argument("--seed", type=int, default=2701)
    p = commands.add_parser("summarize")
    p.add_argument("folder")
    p = commands.add_parser("compare")
    p.add_argument("ppu")
    p.add_argument("cublas")
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args)
    elif args.command == "summarize":
        summarize(args.folder)
    elif args.command == "compare":
        compare(args)
    elif args.command == "_sample":
        dims, = selected_shapes(args)
        write_json(args.output, case(args, dims, args.seed, capture=True))
    else:
        folder = Path(args.output)
        shapes = selected_shapes(args)
        folder.mkdir(parents=True, exist_ok=False)
        for index, dims in enumerate(shapes):
            write_json(folder / f"case-{index:02d}.json", case(args, dims, args.seed+index))
            print(f"PASS {dims}", flush=True)


if __name__ == "__main__":
    main()
