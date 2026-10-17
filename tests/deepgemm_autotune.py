#!/usr/bin/env python3
"""Tune Dense/NoPad/Masked NT GEMMs through C++ JIT + ACU and export runtime LUTs.

Example:
  python tests/deepgemm_autotune.py \
      --format 'DenseGemm,data_type:int8,m:232,n:160,k:4096' --output-dir ./lut
M is rounded up to 16, matching the runtime Dense LUT key.
Grouped LUT keys use GROUPS, EM_upper, N, K; EM is not aligned.
Grouped inputs reuse repository tests/utils.py.
FP4 uses the default, bias-free epilogue; shape K is logical elements, tile BK is packed bytes.
Preferred input: --format 'GroupedNoPad,data_type:fp8,quant_type:block,groups:16,m:256,n:768,k:4096'
or --caselist with one such format (or command containing --format) per line.
Uses an installed DeepGEMM package; no model config or old tuner JSON needed.
Existing LUT keys are skipped; new entries are appended and flushed per case.
ACU profiles only the tested GEMM; raw profiling logs are saved under output-dir/profiles.
Independent precompile records explicit static-assert rejections in unsupported_tiles.json
and excludes them from precision/ACU testing; unclassified or device errors fail the case.
"""
import argparse
import csv
import fcntl
import io
import json
import math
import os
import re
import shutil
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def lut_entry(shape, tile):
    return "{{" + ", ".join(map(str, shape)) + "}, {" + ", ".join(map(str, tile)) + "}},\n"


def read_lut_keys(text, path, gemm_type="densegemm"):
    keys = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"\{\{\s*([^{}]+)\},\s*\{([^{}]+)\}\}\s*,?", line)
        try:
            if not match:
                raise ValueError("expected {{M,N,K}, {BM,BN,BK,WM,WN,STAGES}}")
            shape = tuple(map(int, match[1].split(",")))
            tile = tuple(map(int, match[2].split(",")))
            expected = 3 if gemm_type == "densegemm" else 4
            if len(shape) != expected or len(tile) != 6 or min(shape + tile) <= 0:
                raise ValueError(f"expected {expected} shape and 6 tile positive integers")
            if (gemm_type == "densegemm" and shape[0] % 16) or tile[3] > tile[0] or tile[4] > tile[1]:
                raise ValueError("Dense M must align to 16; warp must not exceed block")
            if tile[0] % tile[3] or tile[1] % tile[4]:
                raise ValueError("warp dimensions must divide block dimensions")
        except ValueError as e:
            raise ValueError(f"{path}:{line_number}: invalid existing LUT entry: {e}") from e
        keys.add(shape)
    return keys


def clear_tuning_env():
    # Clear before importing DeepGEMM: its runtime loader caches the first path.
    os.environ.pop("SAIL_DEEPGEMM_LUT_PATH", None)
    os.environ.pop("SAIL_DEEPGEMM_TUNER_CUSTOM_CONFIG_DIR", None)
    os.environ.pop("show_log", None)  # Kernel logging must not distort timings.
    os.environ.pop("HGGC_WARM_UP", None)


def test_utils():
    tests = Path(__file__).resolve().parent
    if not (tests / "utils.py").is_file():
        raise RuntimeError("Format/grouped tuning requires this script inside the DeepGEMM repository")
    if str(tests) not in sys.path:
        sys.path.insert(0, str(tests))
    import utils
    return utils


def extract_format(line):
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    if '--format' not in line:
        return line
    tokens = shlex.split(line)
    for i, token in enumerate(tokens):
        if token.startswith('--format='):
            return token.split('=', 1)[1]
        if token == '--format' and i + 1 < len(tokens):
            return tokens[i + 1]
    raise ValueError("command has no --format value")


def make_grouped_inputs(shape, gemm_type, torch, case):
    # Reuse the tile_scan fixtures instead of duplicating MoE quantization/layouts.
    utils = test_utils()
    groups, em, n, k = shape
    d = case['data_type']
    quant = case['quant_type']
    distribution = case['distribution']
    m = case['m']
    if gemm_type == "groupednopad":
        actual_m, a, b, indices, out, ref = utils.construct_contiguous_grouped(
            groups, m, k, n, d, distribution, 1, quant_type=quant)
        if (actual_m + groups - 1) // groups != em:
            raise ValueError("constructed NoPad EM differs from LUT key")
        rows = torch.bincount(indices.long(), minlength=groups).to(torch.int32)
        return a, b, out, ref, (indices, rows)
    a, b, masked_m, out, ref, _, _ = utils.construct_grouped_masked(
        groups, m, em, k, n, d, distribution, quant_type=quant)
    return a, b, out, ref, (masked_m, em)


def parse_case(case_format):
    import torch
    case = test_utils().parse_deepgemm_string_re(case_format)
    dtype = {torch.float8_e4m3fn: 'fp8', torch.int8: 'int8', torch.bfloat16: 'bf16', torch.uint8: 'fp4'}[case['data_type']]
    if dtype == 'fp8' and case['quant_type'] not in ('block', 'channel'):
        raise ValueError('FP8 tuning supports block/channel quantization only')
    if case.get('enable_sbo_overlap') or case.get('epilogue_type', 'Default') != 'Default':
        raise ValueError('this entry supports the default epilogue without SBO overlap only')
    gemm = {'DenseGemm': 'densegemm', 'Normal': 'densegemm',
            'GroupedNoPad': 'groupednopad', 'GroupedMasked': 'groupedmasked'}[case['gemm_type']]
    quant = 'channelwise' if case['quant_type'] == 'channel' else 'blockwise'
    m, n, k = case['m'], case['n'], case['k']
    if gemm == 'densegemm':
        shape = ((m + 15) // 16 * 16, n, k)
    else:
        groups = case['groups']
        em = case['em'] if gemm == 'groupedmasked' else (m + groups - 1) // groups
        shape = (groups, em, n, k)
    if min(shape) <= 0 or m <= 0:
        raise ValueError('dimensions must be positive')
    # Keep the original case M for tensors; shape is the runtime LUT key.
    return case, shape, (dtype, quant, gemm)


def case_inputs(args, shape, torch, case):
    utils = test_utils()
    if args.gemm_type == 'densegemm':
        return (*utils.construct(case['m'], case['k'], case['n'],
                                 case['data_type'], quant_type=case['quant_type']), ())
    return make_grouped_inputs(shape, args.gemm_type, torch, case)


def tile_config(args, shape, tile):
    import torch
    import deep_gemm
    from deep_gemm.jit_kernels.gemm_int8 import get_smem_config
    from deep_gemm.jit_kernels.gemm_fp8 import get_smem_config as get_fp8_smem_config
    bm, bn, bk, wm, wn, stages = tile
    k = shape[-1]
    if args.dtype == "fp4":
        from deep_gemm.jit_kernels.gemm_fp4 import get_smem_config_fp4
        # BK is packed bytes; the LUT shape K remains logical elements.
        smem = get_smem_config_fp4(stages, bm, bn, wm, wn, bk, shape[-2], False, False)
    elif args.dtype == "fp8" and args.quant_type == "blockwise":
        smem = get_fp8_smem_config(stages, k, bm, bn, bk)
    else:
        smem = get_smem_config(stages, k, bm, bn, bk, 2 if args.dtype == "bf16" else 1)
    sms = deep_gemm.get_num_sms() or torch.cuda.get_device_properties(args.device).multi_processor_count
    return (sms, bm, bn, bk, wm, wn, stages, smem)


def filter_smem_tiles(args, shape, candidates, smem_limit):
    supported = []
    for index, tile in enumerate(candidates):
        required_smem = tile_config(args, shape, tile)[-1][0]
        if required_smem > smem_limit:
            print(f'UNSUPPORTED_SMEM index={index} tile={tile}: '
                  f'required={required_smem} > limit={smem_limit}', flush=True)
            continue
        supported.append(tile)
    if not supported:
        raise RuntimeError('No candidates fit the device shared-memory limit')
    return supported


def kernel_name(args):
    name = f"gemm_{args.dtype}_{args.dtype}_bf16_nt"
    if args.gemm_type != "densegemm":
        name = "m_grouped_" + name + ("_nopad" if args.gemm_type == "groupednopad" else "_masked")
    return name


def max_grouped_diff(diffs):
    maximum = 0.
    for diff in diffs:
        if not math.isfinite(diff):
            return diff  # Preserve NaN/Inf so the caller rejects this tile.
        maximum = max(maximum, diff)
    return maximum


def profile_worker(args):
    clear_tuning_env()
    if args.compile_worker is not None:
        os.environ['HGGC_WARM_UP'] = '1'  # C++ JIT builds kernels without launching them.
    import torch
    import deep_gemm
    torch.cuda.set_device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    case, shape, (args.dtype, args.quant_type, args.gemm_type) = parse_case(args.format)
    tiles = list(enumerate(json.loads(Path(args.profile_tiles).read_text())))
    if args.compile_worker is not None:
        index, workers = args.compile_worker
        tiles = tiles[index::workers]
    a, b, out, reference, extra = case_inputs(args, shape, torch, case)
    if args.measure_memory:
        import resource
        gpu = torch.cuda.max_memory_reserved()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        print(f'WORKER_MEMORY gpu={gpu} host={rss}', flush=True)
        return
    kernel = getattr(deep_gemm, kernel_name(args))
    if args.compile_worker is not None:
        import ctypes
        flush_native = ctypes.CDLL(None).fflush
    call_args = (a, b, None, out, *extra) if args.dtype == "fp4" else (a, b, out, *extra)
    # Like run_tile_scan: one ACU/Python process and one fixture per case.
    for index, tile in tiles:
        config = tile_config(args, shape, tuple(tile)) if tile is not None else None
        if args.compile_worker is not None:
            print(f'COMPILE_TILE_START index={index}', flush=True)
            try:
                kernel(*call_args, configs=config)
            except RuntimeError as e:
                if 'HGCC compilation failed' not in str(e):
                    raise  # Device errors and allocation failures are not unsupported tiles.
                # Native compiler diagnostics are already in this worker's log.
                # The parent must classify them before accepting any result.
                flush_native(None)  # Flush C/C++ diagnostics before Python's result marker.
                print(f'\nCOMPILE_TILE_FAILED index={index}', flush=True)
            else:
                flush_native(None)
                print(f'COMPILE_TILE_DONE index={index} tile={tile}', flush=True)
            continue
        out.fill_(float('nan'))  # A previous tile must not mask an incomplete write.
        if tile is None:
            kernel(*call_args)  # Baseline: original config selection without an external LUT.
        else:
            kernel(*call_args, configs=config)
        torch.cuda.synchronize()
        if args.gemm_type == "groupedmasked":
            diff = max_grouped_diff(
                deep_gemm.calc_diff(out[g, :int(count)], reference[g, :int(count)])
                for g, count in enumerate(extra[0].tolist()) if count)
        else:
            diff = deep_gemm.calc_diff(out, reference)
        if not math.isfinite(diff) or diff > args.tolerance:
            print(f'PROFILE_TILE_SKIP index={index} diff={diff}', flush=True)
            continue
        print(f'PROFILE_TILE_PASS index={index} diff={diff}', flush=True)
    print('COMPILE_WORKER_PASS' if args.compile_worker is not None else 'PROFILE_WORKER_PASS', flush=True)


def parse_acu_samples(text):
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if line.startswith('"ID",') and '"ppu__time_duration.sum"' in line), None)
    if start is None:
        raise ValueError("ACU CSV header/ppu__time_duration.sum missing")
    rows = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    values = []
    for row in rows:
        if not row.get("ID", "").isdigit():
            continue
        name = row.get("Kernel Name", "").lower()
        if "gemm" not in name or "metadata" in name:
            raise ValueError(f"unexpected profiled kernel: {name}")
        # ACU's PPU duration metric is in ns; keep the tuning score in seconds.
        value = float(row["ppu__time_duration.sum"]) * 1e-9
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid ACU duration: {value}")
        values.append(value)
    return values


def parse_case_profiles(text, tiles):
    records = re.findall(r'^PROFILE_TILE_(PASS|SKIP) index=(\d+) diff=([^\s]+)$', text, re.MULTILINE)
    if [int(index) for _, index, _ in records] != list(range(len(tiles))):
        raise ValueError('profile tile records are missing, duplicated or out of order')
    passed = [int(index) for status, index, _ in records if status == 'PASS']
    samples = parse_acu_samples(text)
    if len(samples) != len(tiles):
        raise ValueError(f'expected {len(tiles)} GEMM samples, got {len(samples)}')
    # Failed-precision tiles were also profiled; consume their samples without selecting them.
    return [(samples[index], tuple(tiles[index]) if tiles[index] is not None else None) for index in passed]


def worker_command(args, tile_file, case_format):
    return [sys.executable,
           str(Path(__file__).resolve()), "--profile-worker",
           '--profile-tiles', str(tile_file), '--format', case_format,
           "--device", str(args.device),
           "--seed", str(args.seed), "--tolerance", str(args.tolerance)]


def parse_compile_results(text, expected_indices):
    sections = re.split(r'^COMPILE_TILE_START index=(\d+)\s*$', text, flags=re.MULTILINE)
    indices = [int(index) for index in sections[1::2]]
    if indices != list(expected_indices):
        raise ValueError('compile tile records are missing, duplicated or out of order')
    rejected = {}
    for index, chunk in zip(indices, sections[2::2]):
        records = re.findall(r'^COMPILE_TILE_(DONE|FAILED) index=(\d+)\b', chunk, re.MULTILINE)
        if len(records) != 1 or int(records[0][1]) != index:
            raise ValueError(f'incomplete compile result for tile {index}')
        if records[0][0] == 'FAILED':
            constraints = re.findall(r'^.*error:.*(?:static_assert failed|static assertion failed).*$', chunk, re.MULTILINE)
            fatal = re.search(r'out of memory|bad_alloc|cannot allocate memory|killed|illegal memory access|device-side assert|HG_ERROR|'
                              r'internal compiler error|segmentation fault|bus error|no space left on device|'
                              r'too many open files|resource temporarily unavailable', chunk, re.IGNORECASE)
            if not constraints or fatal:
                raise ValueError(f'unclassified/fatal compiler failure for tile {index}')
            rejected[index] = constraints[0].strip()
    return rejected


def precompile_tiles(args, shape, candidates, log_dir, case_format):
    tiles = candidates
    tile_file = log_dir / 'compile_tiles.json'
    tile_file.write_text(json.dumps(tiles))
    workers = min(args.compile_workers, len(tiles)) if args.compile_workers != 'auto' else auto_compile_workers(
        args, shape, tile_file, len(tiles), log_dir, case_format)
    print(f'PRECOMPILE {shape}: workers={workers}, candidates={len(candidates)}', flush=True)
    processes = []
    try:
        # Fresh interpreters, like spawn: never fork a process with an initialized GPU context.
        for index in range(workers):
            log = log_dir / f'compile_worker_{index}.log'
            cmd = worker_command(args, tile_file, case_format) + ['--compile-worker', str(index), str(workers)]
            with log.open('w') as stream:
                process = subprocess.Popen(cmd, env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                                           stdout=stream, stderr=subprocess.STDOUT)
            processes.append((process, log))
        failures, rejected = [], {}
        for index, (process, log) in enumerate(processes):
            code = process.wait()
            text = log.read_text(errors='replace')
            if code or 'COMPILE_WORKER_PASS' not in text:
                failures.append(f'{log} (exit={code})')
                continue
            try:
                rejected.update(parse_compile_results(text, range(index, len(tiles), workers)))
            except ValueError as e:
                failures.append(f'{log}: {e}')
        if failures:
            raise RuntimeError('Precompile failed: ' + ', '.join(failures))
    finally:
        for process, _ in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
    skipped = [{'index': index, 'tile': tiles[index], 'reason': reason}
               for index, reason in sorted(rejected.items())]
    (log_dir / 'unsupported_tiles.json').write_text(json.dumps(skipped, indent=2))
    for record in skipped:
        print(f'UNSUPPORTED_TILE index={record["index"]} tile={record["tile"]}: {record["reason"]}', flush=True)
    supported = [tile for index, tile in enumerate(candidates) if index not in rejected]
    if not supported:
        raise RuntimeError(f'No supported compile candidates; see {log_dir}')
    print(f'PRECOMPILE_DONE {shape}: supported={len(supported)}/{len(candidates)}, '
          f'unsupported={len(skipped)}; starting serial precision/profiling', flush=True)
    return supported


def memory_worker_limit(gpu_free, host_free, gpu_peak, host_peak, cpus, candidates, maximum):
    # ponytail: sampled allocation plus fixed headroom, not a guarantee against concurrent users.
    gib = 1 << 30
    gpu_budget, host_budget = int(gpu_free * .8), int(host_free * .8)
    per_gpu, per_host = gpu_peak + gib, host_peak + 2 * gib
    if gpu_budget < per_gpu or host_budget < per_host:
        raise RuntimeError('Insufficient free memory for one compile worker with safety margin')
    return min(candidates, maximum, max(1, cpus // 2), gpu_budget // per_gpu, host_budget // per_host)


def auto_compile_workers(args, shape, tile_file, candidates, log_dir, case_format):
    import torch
    log = log_dir / 'memory_probe.log'
    cmd = worker_command(args, tile_file, case_format) + ['--compile-worker', '0', '1', '--measure-memory']
    with log.open('w') as stream:
        result = subprocess.run(cmd, env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                                stdout=stream, stderr=subprocess.STDOUT)
    match = re.search(r'^WORKER_MEMORY gpu=(\d+) host=(\d+)$', log.read_text(errors='replace'), re.MULTILINE)
    if result.returncode or not match:
        raise RuntimeError(f'Compile memory probe failed; see {log}')
    gpu_peak, host_peak = map(int, match.groups())
    gpu_free, _ = torch.cuda.mem_get_info(args.device)
    meminfo = Path('/proc/meminfo').read_text()
    host_free = int(re.search(r'^MemAvailable:\s+(\d+)', meminfo, re.MULTILINE)[1]) * 1024
    cpus = len(os.sched_getaffinity(0))
    workers = memory_worker_limit(gpu_free, host_free, gpu_peak, host_peak, cpus, candidates, 32)
    print(f'AUTO_COMPILE workers={workers}: peak_gpu={gpu_peak / (1 << 20):.1f}MiB, '
          f'peak_host={host_peak / (1 << 20):.1f}MiB, free_gpu={gpu_free / (1 << 30):.1f}GiB; '
          f'20% reserve + 1GiB GPU/2GiB host headroom per worker, cap=32', flush=True)
    return workers


def parse_compile_workers(value):
    return value if value == 'auto' else int(value)


class ProfileTimeoutError(RuntimeError):
    pass


def run_profile_command(cmd, stream, timeout):
    process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        # Also clean up on Ctrl+C; kill descendants even if ACU exits first.
        def send_group(sig):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        send_group(signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        finally:
            send_group(signal.SIGKILL)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f'WARNING: process {process.pid} did not exit after SIGKILL; check device state', flush=True)
        raise


def profile_case(args, shape, candidates, log_dir, case_format):
    tiles = candidates
    tile_file = log_dir / 'profile_tiles.json'
    tile_file.write_text(json.dumps(tiles))
    log = log_dir / 'acu.log'
    cmd = [args.acu, "--clock-control", "none", "--cache-control", "all",
           "--kernel-name", "regex:^(fp8|int8|bf16|fp4)_.*gemm.*$", "--metrics", "ppu__time_duration.sum",
           "--csv", "--page", "raw", "--check-exit-code", "yes"] + worker_command(args, tile_file, case_format)
    print(f'PROFILE_CASE {shape}: candidates={len(tiles)}; log={log}', flush=True)
    started = time.monotonic()
    with log.open("w") as stream:
        try:
            code = run_profile_command(cmd, stream, args.profile_timeout)
        except subprocess.TimeoutExpired as e:
            raise ProfileTimeoutError(
                f'ACU timeout: shape={shape}, limit={args.profile_timeout}s; '
                f'stopping tuning without writing this case; check device state; see {log}') from e
    text = log.read_text(errors="replace")
    if code != 0 or "PROFILE_WORKER_PASS" not in text:
        raise RuntimeError(f"ACU/precision check failed (exit={code}); see {log}")
    results = parse_case_profiles(text, tiles)
    print(f'PROFILE_CASE_DONE {shape}: valid={len(results)}/{len(tiles)}, wall={time.monotonic() - started:.3f}s', flush=True)
    return results


def tune(args):
    clear_tuning_env()
    if not shutil.which(args.acu):
        raise RuntimeError(f"ACU executable not found: {args.acu}")
    import torch
    from deep_gemm.jit_kernels.utils import get_search_space
    torch.cuda.set_device(args.device)
    specs = args.case_specs
    device_props = torch.cuda.get_device_properties(args.device)
    smem_limit = max(device_props.shared_memory_per_block,
                     device_props.shared_memory_per_block_optin)
    cases = list(specs)
    for shape in cases:
        n, k = shape[-2:]
        if n % 16 or k % 32:
            raise ValueError("Cases require N divisible by 16 and K divisible by 32")
    torch_dtype = {"fp8": torch.float8_e4m3fn, "int8": torch.int8, "bf16": torch.bfloat16, "fp4": torch.uint8}[args.dtype]
    # Channelwise FP8 shares the INT8 tile/kernel path, not blockwise FP8 candidates.
    search_dtype = torch.int8 if args.dtype == "fp8" and args.quant_type == "channelwise" else torch_dtype
    search_type = {"densegemm": "dense", "groupednopad": "nopad", "groupedmasked": "masked"}[args.gemm_type]
    if args.dtype == 'fp4':
        from tile_config_check_fp4 import get_search_space_doublecheck
        # The FP4 helper has dense and grouped-base (nopad) candidate sets.
        fp4_type = 'dense' if search_type == 'dense' else 'nopad'
        search_space = sorted(get_search_space_doublecheck(torch_dtype, fp4_type))
    else:
        search_space = get_search_space(search_dtype, search_type)
    # Normalize search-space tuples to LUT order once: BM, BN, BK, WM, WN, STAGES.
    candidates = list(dict.fromkeys(
        (bm, bn, bk, wm, wn, stages)
        for bm, bn, wm, wn, bk, stages in search_space))
    if args.max_candidates:
        candidates = candidates[:args.max_candidates]
    device_name = torch.cuda.get_device_name(args.device)
    kind = f"fp8_{args.quant_type}" if args.dtype == "fp8" else args.dtype
    output = Path(args.output_dir) / f"{device_name}_{kind}_{args.gemm_type}.lut"
    output.parent.mkdir(parents=True, exist_ok=True)
    failures = saved = skipped = not_improved = 0
    with output.open("a+") as table:
        try:
            fcntl.flock(table, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f"{output}: another tuner is writing this LUT") from e
        table.seek(0)
        content = table.read()
        existing = read_lut_keys(content, output, args.gemm_type)
        needs_newline = bool(content) and not content.endswith("\n")
        if not content:
            key = "M, N, K" if args.gemm_type == "densegemm" else "GROUPS, EM_upper, N, K"
            table.write("# {{" + key + "}, {BM, BN, BK, WM, WN, STAGES}}\n")
            table.flush()
        for shape in cases:
            if shape in existing:
                skipped += 1
                print(f"EXISTS {shape}: skip tuning", flush=True)
                continue
            profiles = output.parent / "profiles"
            profiles.mkdir(exist_ok=True)
            log_dir = Path(tempfile.mkdtemp(prefix=f"{kind}_{args.gemm_type}_" + "_".join(map(str, shape)) + "_", dir=profiles))
            log_dir.chmod(0o755)  # Keep logs readable on the host when tuning inside Docker.
            try:
                resource_candidates = filter_smem_tiles(args, shape, candidates, smem_limit)
                case_candidates = resource_candidates
                if args.compile_workers:
                    case_candidates = precompile_tiles(args, shape, resource_candidates, log_dir, specs[shape])
                profile_tiles = [None, *case_candidates] if args.require_improvement else case_candidates
                valid = profile_case(args, shape, profile_tiles, log_dir, specs[shape])
                baseline = None
                if args.require_improvement:
                    baseline = next((elapsed for elapsed, tile in valid if tile is None), None)
                    if baseline is None:
                        raise RuntimeError('Default config failed precision; cannot compare performance')
                    print(f'BASELINE {shape}: default config, {baseline * 1e6:.3f} us', flush=True)
                    valid = [(elapsed, tile) for elapsed, tile in valid if tile is not None]
            except ProfileTimeoutError:
                raise  # Do not continue with other cases after a possible GPU hang.
            except Exception as e:
                # A device/context error invalidates the case; never write a partial result.
                failures += 1
                print(f'FAIL {shape}: {e}', flush=True)
                continue
            for elapsed, tile in valid:
                print(f'ACU {shape} tile={tile}: {elapsed * 1e6:.3f} us', flush=True)
            if not valid:
                failures += 1
                print(f"FAIL {shape}: no valid candidate", flush=True)
                continue
            elapsed, tile = min(valid, key=lambda item: item[0])
            if baseline is not None and (elapsed >= baseline or elapsed > baseline * (1 - args.min_improvement)):
                not_improved += 1
                print(f'NO_IMPROVEMENT {shape}: best={elapsed * 1e6:.3f} us, '
                      f'default={baseline * 1e6:.3f} us; required reduction={args.min_improvement:.1%}; '
                      f'not adding to LUT', flush=True)
                continue
            if needs_newline:
                table.write("\n")
                needs_newline = False
            table.write(lut_entry(shape, tile))
            table.flush()
            existing.add(shape)
            saved += 1
            print(f"BEST {shape}: tile={tile}, {elapsed * 1e6:.3f} us; valid={len(valid)}/{len(case_candidates)}, "
                  f"smem_unsupported={len(candidates) - len(resource_candidates)}, "
                  f"compile_unsupported={len(resource_candidates) - len(case_candidates)}", flush=True)
    print(f"Saved {saved} new, skipped {skipped} existing, not improved {not_improved}, "
          f"failed {failures}; LUT: {output}", flush=True)
    return 1 if failures else 0


def tune_formats(args):
    clear_tuning_env()
    lines = [args.format] if args.format else Path(args.caselist).read_text().splitlines()
    batches = {}
    for line_number, line in enumerate(lines, 1):
        text = extract_format(line)
        if text is None:
            continue
        try:
            _, shape, kind = parse_case(text)
        except (KeyError, ValueError, ZeroDivisionError) as e:
            raise ValueError(f'{args.caselist or "--format"}:{line_number}: unsupported/invalid case: {text}') from e
        # A LUT stores one tile per key, so first case wins even if distributions differ.
        batches.setdefault(kind, {}).setdefault(shape, text)
    if not batches:
        raise ValueError('no cases')
    failures = 0
    for (dtype, quant, gemm), specs in batches.items():
        batch = argparse.Namespace(**vars(args))
        batch.dtype, batch.quant_type, batch.gemm_type = dtype, quant, gemm
        batch.case_specs = specs
        failures |= tune(batch)
    return failures


def self_test():
    from unittest.mock import patch
    with tempfile.TemporaryFile(mode='w+') as log:
        assert run_profile_command([sys.executable, '-c', 'print("ok")'], log, 10) == 0
        assert run_profile_command([sys.executable, '-c', 'raise SystemExit(3)'], log, 10) == 3
        try:
            run_profile_command([sys.executable, '-c', 'import time; time.sleep(60)'], log, .2)
        except subprocess.TimeoutExpired:
            pass
        else:
            raise AssertionError('profile timeout was ignored')
    for gemm in ('densegemm', 'groupednopad', 'groupedmasked'):
        shape = (16, 768, 4096) if gemm == 'densegemm' else (8, 16, 768, 4096)
        good = lut_entry(shape, (64, 64, 128, 32, 32, 3))
        assert read_lut_keys(good.rstrip() + ' # comment\r\n', 'test.lut', gemm) == {shape}
        for bad in (lut_entry(shape, (64, 64, 128, 48, 32, 3)),
                    lut_entry(shape, (64, 64, 128, 32, 48, 3)),
                    lut_entry(shape, (64, 64, 128, 0, 32, 3)),
                    good.rstrip() + ' // comment\n'):
            try:
                read_lut_keys(bad, 'test.lut', gemm)
            except ValueError:
                pass
            else:
                raise AssertionError('incompatible runtime LUT accepted')
    assert max_grouped_diff(iter(())) == 0.
    assert max_grouped_diff(iter((0.0001, 0.0003, 0.0002))) == 0.0003
    for invalid in (float('nan'), float('inf'), float('-inf')):
        for index in range(3):
            diffs = [0.0001, 0.0002, 0.0003]
            diffs[index] = invalid
            assert not math.isfinite(max_grouped_diff(iter(diffs)))
    candidates = [(16, 64, 128, 16, 16, 2), (128, 128, 256, 64, 64, 4)]
    with patch(__name__ + '.tile_config', side_effect=[((262144, 128, 0),), ((278976, 128, 0),)]):
        assert filter_smem_tiles(None, (32, 768, 4096), candidates, 262144) == candidates[:1]
    with patch(__name__ + '.tile_config', return_value=((278976, 128, 0),)):
        try:
            filter_smem_tiles(None, (32, 768, 4096), candidates, 262144)
        except RuntimeError:
            pass
        else:
            raise AssertionError('all-oversized candidates accepted')
    done = 'COMPILE_TILE_START index=0\nCOMPILE_TILE_DONE index=0\n'
    failed = 'COMPILE_TILE_START index=1\nerror: static_assert failed: unsupported shape\nCOMPILE_TILE_FAILED index=1\n'
    assert parse_compile_results(done + failed, [0, 1]) == {1: 'error: static_assert failed: unsupported shape'}
    for bad in (failed.replace('static_assert failed', 'unknown type'), failed.replace('unsupported shape', 'out of memory'),
                done + done, done.replace('DONE index=0', 'DONE index=1'), done.replace('COMPILE_TILE_DONE index=0', '')):
        try:
            parse_compile_results(bad, [0] if bad.startswith(done.splitlines()[0]) else [1])
        except ValueError:
            pass
        else:
            raise AssertionError('unsafe/incomplete compile results accepted')
    gib = 1 << 30
    assert memory_worker_limit(100 * gib, 200 * gib, gib, 2 * gib, 192, 112, 32) == 32
    assert memory_worker_limit(100 * gib, 200 * gib, gib, 2 * gib, 8, 112, 64) == 4
    assert memory_worker_limit(100 * gib, 200 * gib, gib, 2 * gib, 192, 2, 64) == 2
    try:
        memory_worker_limit(gib, 200 * gib, gib, gib, 192, 112, 64)
    except RuntimeError:
        pass
    else:
        raise AssertionError('insufficient worker memory accepted')
    assert extract_format('python tile_scan_with_test.py --format="GroupedNoPad,data_type:fp8,groups:16,m:64,n:768,k:4096"') == 'GroupedNoPad,data_type:fp8,groups:16,m:64,n:768,k:4096'
    assert extract_format(' --format DenseGemm,m:16,n:768,k:4096') == 'DenseGemm,m:16,n:768,k:4096'
    assert extract_format('# comment') is None
    with tempfile.TemporaryDirectory() as directory:
        compile_args = argparse.Namespace(compile_workers=2)
        case_format = 'DenseGemm,data_type:int8,m:16,n:768,k:4096'
        candidates = [(16, 64, 128, 16, 16, 2), (16, 64, 128, 16, 16, 3)]
        fake_worker = "import sys; i=int(sys.argv[-2]); print(f'COMPILE_TILE_START index={i}'); print(f'COMPILE_TILE_DONE index={i}'); print('COMPILE_WORKER_PASS')"
        with patch(__name__ + '.worker_command', return_value=[sys.executable, '-c', fake_worker]):
            assert precompile_tiles(compile_args, (16, 768, 4096), candidates, Path(directory), case_format) == candidates
            assert json.loads((Path(directory) / 'compile_tiles.json').read_text()) == list(map(list, candidates))
        rejecting_worker = "import sys; i=int(sys.argv[-2]); print(f'COMPILE_TILE_START index={i}'); print('error: static_assert failed: unsupported shape' if i else ''); print(f'COMPILE_TILE_{\"FAILED\" if i else \"DONE\"} index={i}'); print('COMPILE_WORKER_PASS')"
        with patch(__name__ + '.worker_command', return_value=[sys.executable, '-c', rejecting_worker]):
            assert precompile_tiles(compile_args, (16, 768, 4096), candidates, Path(directory), case_format) == candidates[:1]
            assert json.loads((Path(directory) / 'unsupported_tiles.json').read_text())[0]['index'] == 1
        with patch(__name__ + '.worker_command', return_value=[sys.executable, '-c', 'raise SystemExit(3)']):
            try:
                precompile_tiles(compile_args, (16, 768, 4096), candidates, Path(directory), case_format)
            except RuntimeError as e:
                assert 'exit=3' in str(e)
            else:
                raise AssertionError('failed precompile worker accepted')
        path = Path(directory) / "test.lut"
        for gemm_type in ("groupednopad", "groupedmasked"):
            entry = lut_entry((16, 4, 768, 4096), (32, 64, 128, 16, 32, 3))
            assert read_lut_keys(entry, path, gemm_type) == {(16, 4, 768, 4096)}
        assert lut_entry((16, 768, 4096), (32, 64, 128, 16, 32, 3)) == "{{16, 768, 4096}, {32, 64, 128, 16, 32, 3}},\n"
        old = lut_entry((16, 768, 4096), (32, 64, 128, 16, 32, 3))
        assert read_lut_keys("# header\r\n\r\n" + old, path) == {(16, 768, 4096)}
        table_path = Path(directory) / "test.lut"
        table_path.write_text(old)
        cases = [(16, 768, 4096), (32, 768, 4096)]
        for _ in range(2):
            with table_path.open("a+") as table:
                fcntl.flock(table, fcntl.LOCK_EX | fcntl.LOCK_NB)
                table.seek(0)
                keys = read_lut_keys(table.read(), table_path)
                for shape in cases:
                    if shape not in keys:
                        table.write(lut_entry(shape, (32, 64, 128, 16, 32, 3)))
                        keys.add(shape)
        assert table_path.read_text() == old + lut_entry((32, 768, 4096), (32, 64, 128, 16, 32, 3))
        try:
            read_lut_keys(old + "broken row\n", table_path)
        except ValueError as e:
            assert f"{table_path}:2" in str(e)
        else:
            raise AssertionError("malformed LUT accepted")
        csv_text = 'profiler output\n"ID","Kernel Name","ppu__time_duration.sum"\n"0","fp8_deep_gemm",20000\n"1","fp8_deep_gemm",22000\n"2","fp8_deep_gemm",21000\n'
        tiles = [(16, 64, 128, 16, 16, 2), (16, 64, 128, 16, 16, 3), (32, 64, 128, 16, 16, 3)]
        markers = 'PROFILE_TILE_PASS index=0 diff=0\nPROFILE_TILE_SKIP index=1 diff=nan\nPROFILE_TILE_PASS index=2 diff=0\n'
        report = markers + csv_text
        single = parse_case_profiles(report, tiles)
        assert [tile for _, tile in single] == [tiles[0], tiles[2]]
        assert math.isclose(single[0][0], 20e-6) and math.isclose(single[1][0], 21e-6)
        assert parse_case_profiles(report, [None, *tiles[1:]])[0] == (single[0][0], None)
        for bad in (report.replace('index=2', 'index=0'), report.replace('index=1', 'index=4'),
                    report.replace('"2","fp8_deep_gemm",21000\n', ''), report + '"3","fp8_deep_gemm",21000\n'):
            try:
                parse_case_profiles(bad, tiles)
            except ValueError:
                pass
            else:
                raise AssertionError('misaligned batch profiling results accepted')
        for bad in ('missing metrics', csv_text.replace('fp8_deep_gemm', 'reference_matmul'),
                    csv_text.replace('20000', 'nan')):
            try:
                parse_acu_samples(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid ACU report accepted")
    print("self-test PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--format', help='same case format as tests/run_tile_scan.py; datatype is taken from the case')
    inputs.add_argument('--caselist', help='one case format or command containing --format per line')
    parser.add_argument("--output-dir", default="./lut")
    parser.add_argument("--device", type=int, default=0, help="index within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=0.001, help="maximum calc_diff; grouped uses existing tile_scan reference")
    parser.add_argument("--acu", default="acu", help="ACU executable name or path")
    parser.add_argument('--profile-timeout', type=float, default=600,
                        help='whole-case ACU profiling timeout in seconds (default: 600); timeout stops tuning')
    parser.add_argument('--compile-workers', type=parse_compile_workers, default='auto', help='auto: memory/CPU-based concurrency; integer: manual workers; 0 disables precompile')
    parser.add_argument("--profile-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument('--profile-tiles', help=argparse.SUPPRESS)
    parser.add_argument('--compile-worker', nargs=2, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--measure-memory', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument("--max-candidates", type=int, default=0, help="smoke test only: truncate search; 0 = full search")
    parser.add_argument('--require-improvement', action='store_true', help='only append tiles faster than the default config in the same ACU run')
    parser.add_argument('--min-improvement', type=float, default=0.05, help='minimum time reduction relative to the default config (default: 0.05, or 5%%); positive values enable baseline comparison')
    parser.add_argument("--self-test", action="store_true", help="check caselist/LUT format without GPU dependencies")
    args = parser.parse_args()
    if not math.isfinite(args.min_improvement) or not 0 <= args.min_improvement < 1:
        parser.error('min-improvement must be finite and in [0, 1)')
    args.require_improvement |= args.min_improvement > 0
    if not math.isfinite(args.profile_timeout) or args.profile_timeout <= 0:
        parser.error('profile-timeout must be finite and positive')
    if args.self_test:
        self_test()
        return 0
    if (args.compile_workers != 'auto' and not 0 <= args.compile_workers <= 32) or args.device < 0 or args.max_candidates < 0 or not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("compile-workers must be auto or 0..32; device/max-candidates nonnegative; tolerance positive")
    if args.profile_worker:
        if not args.profile_tiles or not args.format:
            parser.error('profile worker requires --profile-tiles and --format')
        if args.compile_worker is not None and not (0 <= args.compile_worker[0] < args.compile_worker[1]):
            parser.error('compile worker requires 0 <= index < workers')
        profile_worker(args)
        return 0
    if not (args.format or args.caselist):
        parser.error("--format or --caselist is required")
    return tune_formats(args)


if __name__ == "__main__":
    raise SystemExit(main())
