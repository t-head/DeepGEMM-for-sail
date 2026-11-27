"""Compile a functional caselist in isolated workers, then check accuracy serially.

This entry point deliberately has no Torch/PPU dependency: --list and scheduling
unit tests can run on a CPU host. Each worker owns one reusable child process.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import re
import select
import shlex
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITES = {'ZW-810E': 'checkin_810e.list', 'ZW-M890P': 'checkin_890p.list'}
SCRIPT_SUITES = {
    'ZW-810E': [
        '-m pytest -q -s --disable-warnings -x tests/test_cold_graph_capture.py -k bf16',
    ],
    'ZW-M890P': [
        '-m pytest -q -s --disable-warnings -x tests/test_cold_graph_capture.py',
    ],
    'ZW-M890P-MegaMoE': [
        '-m pytest -q -s --disable-warnings -x -k smoke tests/test_mega_moe.py',
        '-m pytest -q -s --disable-warnings -x '
        'tests/test_mega_gate.py::test_mega_gate_scoring_modes '
        'tests/test_mega_gate.py::test_mega_gate_sigmoid_infinite_logits '
        'tests/test_mega_gate.py::test_mega_gate_sigmoid_routing_controls '
        'tests/test_mega_gate.py::test_mega_gate_sigmoid_partial_scores '
        'tests/test_mega_gate.py::test_mega_gate_api_contract',
    ],
    'ZW-M890P-GemmRS': [
        f'-m torch.distributed.run --standalone --nproc-per-node={tp} tests/test_gemm_comm.py --smoke'
        for tp in (2, 4, 8)
    ],
}
_worker_context = threading.local()


def gemm_rs_config_check(world):
    return 32 * world, 272, 768, (1, 16, 64, 128, 16, 16, 2)


def gemm_rs_smoke_cases(world, num_sms=None, *, include_config_checks=False):
    if num_sms is None:
        from deep_gemm import get_num_sms
        num_sms = get_num_sms()
    cases = []
    for dtype, quant, m, n, k, tile in (
        ('fp8', 'block', world, 48, 640, None),
        ('fp8', 'channel', 2 * world, 80, 96, None),
        ('bf16', 'non_quantized', 3 * world, 144, 128, None),
        ('fp8', 'block', 13 * world, 272, 768,
         (32, 128, 128, 16, 32, 2) if world == 1 else (32, 192, 128, 16, 48, 2)),
        ('fp8', 'channel', 1024, 4096, 768, (64, 64, 128, 32, 32, 3)),
        ('bf16', 'non_quantized', 1024, 4096, 768, (64, 64, 128, 32, 32, 3)),
    ):
        spec = (f'GemmReduceScatter,data_type:{dtype},quant_type:{quant},m:{m},n:{n},k:{k},'
                f'tp:{world},rank:0')
        if tile:
            spec += ',configs:[' + ','.join(map(str, (num_sms, *tile))) + ']'
        cases.append(spec)
    if include_config_checks:
        m, n, k, config = gemm_rs_config_check(world)
        for dtype, quant in (('fp8', 'block'), ('fp8', 'channel'), ('bf16', 'non_quantized')):
            cases.append(f'GemmReduceScatter,data_type:{dtype},quant_type:{quant},m:{m},n:{n},k:{k},'
                         f'tp:{world},rank:0,configs:[' + ','.join(map(str, config)) + ']')
    return cases


def gemm_rs_world(spec):
    if not spec.startswith('GemmReduceScatter,'):
        return None
    match = re.search(r'(?:^|,)tp:(1|2|4|8)(?:,|$)', spec)
    if match is None:
        raise ValueError('GEMM/RS cases require tp:1, 2, 4 or 8')
    return int(match[1])


def compile_cases(cases):
    expanded = []
    for spec in cases:
        world = gemm_rs_world(spec)
        if world is None:
            expanded.append(spec)
            continue
        spec = re.sub(r',rank:\d+(?=,|$)', '', spec)
        expanded.extend(f'{spec},rank:{rank}' for rank in range(world))
    # Repeated model aliases and rank dumps use the same compiler/cache entry.
    return list(dict.fromkeys(expanded))


def _process_tree(pid):
    rows = subprocess.check_output(['ps', '-e', '-o', 'pid=,ppid=,pgid=,stat='], text=True)
    processes = {int(p): (int(parent), int(group), state) for p, parent, group, state in
                 (line.split() for line in rows.splitlines())}
    descendants = {pid}
    while True:
        children = {p for p, (parent, _, _) in processes.items() if parent in descendants}
        if children <= descendants:
            return {p: processes[p] for p in descendants if p in processes}
        descendants.update(children)


def _stop_process(process):
    # Elastic ranks have their own sessions; retain their groups before the
    # launcher exits and they are reparented. Never signal our own process group.
    descendants = _process_tree(process.pid)
    groups = {process.pid, *(group for _, group, _ in descendants.values())} - {os.getpgrp()}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for group in groups:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass
            except PermissionError:
                if process.poll() is None:
                    raise
        if sig == signal.SIGTERM:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            descendants.update(_process_tree(process.pid))
            groups.update(group for _, group, _ in descendants.values() if group != os.getpgrp())
    process.wait()
    deadline = time.monotonic() + 5
    while descendants:
        # The original descendants may have been adopted after launcher exit.
        rows = subprocess.check_output(['ps', '-e', '-o', 'pid=,stat='], text=True)
        alive = {int(p) for p, state in (line.split() for line in rows.splitlines()) if not state.startswith('Z')}
        descendants = {p: info for p, info in descendants.items() if p in alive}
        if not descendants:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f'worker processes did not exit: {sorted(descendants)}')
        time.sleep(0.05)


@dataclass
class Result:
    index: int
    spec: str
    phase: str
    elapsed: float
    error: str
    log: Path


class Worker:
    """One process per compiler worker, or one for the serial accuracy phase."""

    def __init__(self, phase, log_dir):
        self.phase = phase
        self.log_dir = Path(log_dir)
        self.process = None
        self.response_fd = None

    def start(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.setdefault('OMP_NUM_THREADS', '1')
        if self.phase == 'compile':
            env['HGGC_WARM_UP'] = '1'
        else:
            env.pop('HGGC_WARM_UP', None)
        response_fd, child_fd = os.pipe()
        try:
            with (self.log_dir / f'worker-{self.phase}-{threading.get_ident()}.log').open('a') as log:
                self.process = subprocess.Popen(
                    [sys.executable, str(REPO_ROOT / 'tests/checkin/checkin_worker.py'),
                     '--result-fd', str(child_fd)], cwd=REPO_ROOT, env=env,
                    stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                    pass_fds=(child_fd,), start_new_session=True)
            self.response_fd = response_fd
        except BaseException:
            os.close(response_fd)
            raise
        finally:
            os.close(child_fd)

    def close(self, kill=False):
        process = self.process
        if process is not None:
            if not kill:
                try:
                    process.stdin.close()
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    kill = True
            if kill:
                _stop_process(process)
                try:
                    process.stdin.close()
                except OSError:
                    pass
            self.process = None
        if self.response_fd is not None:
            os.close(self.response_fd)
            self.response_fd = None

    def run(self, index, spec, phase, log_dir, timeout):
        assert phase == self.phase
        log = Path(log_dir) / f'{phase}-{index:04d}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        log.touch()
        start = time.monotonic()
        error = ''
        try:
            if self.process is None:
                self.start()
            request = dict(index=index, spec=spec, log=str(log.resolve()))
            self.process.stdin.write((json.dumps(request) + '\n').encode())
            self.process.stdin.flush()
            response = b''
            while not response.endswith(b'\n'):
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0 or not select.select([self.response_fd], [], [], remaining)[0]:
                    raise TimeoutError(f'Timeout after {timeout}s')
                chunk = os.read(self.response_fd, 4096)
                if not chunk:
                    raise RuntimeError('Worker exited before reporting a result; see worker log')
                response += chunk
            result = json.loads(response)
            if result['index'] != index:
                raise RuntimeError('Worker returned the wrong case index')
            error = result['error']
        except (OSError, ValueError, KeyError, RuntimeError, TimeoutError) as exc:
            error = str(exc)
            self.close(kill=True)
            with log.open('a') as output:
                output.write(f'\nWorker error: {error}\n')
        return Result(index, spec, phase, time.monotonic() - start, error, log)


def load_cases(path, stack=()):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f'Recursive caselist include: {path}')
    cases = []
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.endswith('.list'):
            cases.extend(load_cases(path.parent / line, (*stack, path)))
        elif re.fullmatch(r'\w+,\w+:.+', line):
            cases.append(line)
        else:
            raise ValueError(f'{path}:{number}: expected a DeepGEMM case specification')
    if not cases:
        raise ValueError(f'Empty caselist: {path}')
    return cases


def run_case(index, spec, phase, log_dir, timeout, base_env=None):
    worker = getattr(_worker_context, 'worker', None)
    if worker is not None and base_env is None and phase != 'script':
        return worker.run(index, spec, phase, log_dir, timeout)
    env = dict(os.environ if base_env is None else base_env)
    env.setdefault('OMP_NUM_THREADS', '1')
    if phase == 'compile':
        env['HGGC_WARM_UP'] = '1'
    else:
        # A nonempty string, including "0", enables warm-up in both runtimes.
        env.pop('HGGC_WARM_UP', None)
    command = ([sys.executable, *shlex.split(spec)] if phase == 'script' else
               [sys.executable, str(REPO_ROOT / 'tests/run_deep_gemm.py'),
                '--format', spec, '--require_supported'])
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f'{phase}-{index:04d}.log'
    start = time.monotonic()
    error = ''
    with log.open('w') as output:
        try:
            process = subprocess.Popen(command, cwd=REPO_ROOT, env=env,
                                       stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _stop_process(process)
                error = f'Timeout after {timeout}s'
            else:
                if code:
                    error = f'Exit code {code}'
        except OSError as exc:
            error = str(exc)
    return Result(index, spec, phase, time.monotonic() - start, error, log)


def execute(cases, workers, log_dir, timeout, phase='all', runner=run_case):
    results = []
    if phase in ('all', 'compile'):
        phase_start = time.monotonic()
        compiler_workers = []

        def install_worker():
            worker = Worker('compile', log_dir)
            compiler_workers.append(worker)
            _worker_context.worker = worker

        try:
            with ThreadPoolExecutor(max_workers=workers, initializer=install_worker) as pool:
                planned = compile_cases(cases)
                pending = [pool.submit(runner, i, spec, 'compile', log_dir, timeout)
                           for i, spec in enumerate(planned, 1)]
                for future in as_completed(pending):
                    result = future.result()
                    results.append(result)
                    print(f'Compile [{result.index}/{len(planned)}]: {result.error or "OK"}', flush=True)
        finally:
            for worker in compiler_workers:
                worker.close()
        print(f'Compile wall time: {time.monotonic() - phase_start:.3f}s', flush=True)
        if any(result.error for result in results):
            print('Compilation incomplete; accuracy phase was not started.', flush=True)
            return results
    if phase in ('all', 'run'):
        phase_start = time.monotonic()
        collectives = {}
        previous = getattr(_worker_context, 'worker', None)
        worker = Worker('run', log_dir)
        _worker_context.worker = worker
        try:
            for i, spec in enumerate(cases, 1):
                world = gemm_rs_world(spec)
                if world is not None:
                    collectives.setdefault(world, []).append(spec)
                    continue
                result = runner(i, spec, 'run', log_dir, timeout)
                results.append(result)
                print(f'Accuracy [{i}/{len(cases)}]: {result.error or "OK"}', flush=True)
        finally:
            worker.close()
            _worker_context.worker = previous
        # One process group per TP, sharing the warmed cache across all cases.
        for world, specs in collectives.items():
            path = Path(log_dir) / f'gemm-rs-tp{world}.list'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('\n'.join(specs) + '\n')
            command = (f'-m torch.distributed.run --standalone --nproc-per-node={world} '
                       f'tests/test_gemm_comm.py --caselist {shlex.quote(str(path.resolve()))}')
            result = runner(world, command, 'script', log_dir, timeout)
            results.append(result)
            print(f'GEMM/RS TP{world}: {result.error or "OK"}', flush=True)
            if result.error:
                break
        print(f'Run wall time: {time.monotonic() - phase_start:.3f}s', flush=True)
    return results


def execute_scripts(specs, log_dir, timeout):
    results = []
    for i, spec in enumerate(specs, 1):
        result = run_case(i, spec, 'script', log_dir, timeout)
        results.append(result)
        print(f'Script [{i}/{len(specs)}]: {result.error or "OK"}', flush=True)
        if result.error:
            print(result.log.read_text(errors='replace')[-12000:], flush=True)
    return results


def write_report(results, path):
    failed = sum(bool(result.error) for result in results)
    suite = ET.Element('testsuite', name='DeepGEMM checkin', tests=str(len(results)),
                       failures=str(failed), errors='0', skipped='0',
                       time=f'{sum(result.elapsed for result in results):.6f}')
    for result in sorted(results, key=lambda r: ({'compile': 0, 'run': 1, 'script': 2}[r.phase], r.index)):
        case = ET.SubElement(suite, 'testcase', name=f'{result.phase} [{result.index}]: {result.spec}',
                             classname=f'DeepGEMM.{result.phase}', time=f'{result.elapsed:.6f}')
        if result.error:
            failure = ET.SubElement(case, 'failure', message=result.error)
            # Keep XML bounded; the full compiler/UT output is retained in the log.
            with result.log.open('rb') as log:
                log.seek(0, 2)
                log.seek(max(0, log.tell() - 12000))
                failure.text = log.read().decode('utf-8', errors='replace')
            error_log = failure.text if failure.text.strip() else result.error
            ET.SubElement(case, 'system-err').text = error_log
        ET.SubElement(case, 'system-out').text = f'Full log: {result.log}'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # XML 1.0 cannot represent control characters emitted by some compilers.
    xml = ET.tostring(suite, encoding='unicode')
    xml = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', xml)
    path.write_text('<?xml version="1.0" encoding="utf-8"?>\n' + xml, encoding='utf-8')


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--suite', choices=sorted(SUITES.keys() | SCRIPT_SUITES.keys()))
    source.add_argument('--caselist', type=Path)
    parser.add_argument('--compile-workers', type=positive_int, default=16)
    parser.add_argument('--timeout', type=positive_int, default=1200, help='Per-case, per-phase timeout in seconds')
    parser.add_argument('--phase', choices=('all', 'compile', 'run'), default='all')
    parser.add_argument('--output-xml', type=Path, default=Path('test-results.xml'))
    parser.add_argument('--log-dir', type=Path, default=Path('checkin-logs'))
    parser.add_argument('--list', action='store_true', help='Print the selected cases without importing Torch')
    args = parser.parse_args(argv)
    if args.suite in SCRIPT_SUITES and args.suite not in SUITES:
        gemm_rs = args.suite == 'ZW-M890P-GemmRS'
        if args.phase == 'compile' and not gemm_rs:
            parser.error('Script suites run compilation and correctness together; use --phase all or run')
        cases = SCRIPT_SUITES[args.suite]
        if args.list:
            print('\n'.join(f'{i}: {sys.executable} {spec}' for i, spec in enumerate(cases, 1)))
            return 0
        results = []
        if gemm_rs and args.phase in ('all', 'compile'):
            warmup = [spec for tp in (2, 4, 8) for spec in gemm_rs_smoke_cases(tp, include_config_checks=True)]
            results = execute(warmup, args.compile_workers, args.log_dir.resolve(), args.timeout, 'compile')
        if args.phase != 'compile' and not any(result.error for result in results):
            results.extend(execute_scripts(cases, args.log_dir.resolve(), args.timeout))
        write_report(results, args.output_xml)
        return int(any(result.error for result in results))
    path = REPO_ROOT / 'tests/checkin' / SUITES[args.suite] if args.suite else args.caselist
    try:
        cases = load_cases(path)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.list:
        for i, spec in enumerate(cases, 1):
            print(f'{i}: {spec}')
        return 0
    results = execute(cases, args.compile_workers, args.log_dir.resolve(), args.timeout, args.phase)
    if args.phase != 'compile' and not any(result.error for result in results):
        results.extend(execute_scripts(SCRIPT_SUITES.get(args.suite, ()),
                                       args.log_dir.resolve(), args.timeout))
    write_report(results, args.output_xml)
    failed = sum(bool(result.error) for result in results)
    print(f'Phase={args.phase}, cases={len(cases)}, phase results={len(results)}, failures={failed}')
    return int(bool(failed))


if __name__ == '__main__':
    sys.exit(main())
