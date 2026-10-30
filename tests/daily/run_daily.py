"""Run the functional caselist, then the board's full shape and tile sweeps.

Daily-only logic (sweep lists, per-command timeouts and failure detection)
lives here; shared scheduling and reporting comes from run_checkin.
"""
import argparse
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'checkin'))
from run_checkin import REPO_ROOT, SUITES, Result, execute, load_cases, positive_int, write_report

DAILY_SUITES = {'ZW-810E': 'daily_810e.list', 'ZW-M890P': 'daily_890p.list'}

# Ceilings are sized for the first run of a fresh CI workspace, where every
# JIT kernel must be compiled by hgcc on a cold kernel cache. Measured on
# ZW810E (PR #15, commit 35fb2c0): test_core.py alone needs ~2580s single
# process; tile_scan_with_test.py spends most of its wall time compiling
# per-tile kernels, which also exceeds the 1200s default on a cold cache.
# Keep these comfortably above the observed times so slow-but-passing runs
# are not killed.
SCRIPT_TIMEOUT_OVERRIDES = {
    'tests/test_core.py': 3600,
    'tests/tile_scan_with_test.py': 3600,
    'tests/test_fp4_core.py': 3000,
    'tests/test_fp8_core.py': 2100,
}
DEFAULT_SCRIPT_TIMEOUT = 1200
FAILURE_KEYWORD_RE = re.compile(
    r'\bfail(?:ed|ure)?\b|Traceback \(most recent call last\):|\bAssertionError\b',
    re.IGNORECASE)


def load_script_checks(suite):
    """Load daily test commands from the board-specific .list file."""
    filename = DAILY_SUITES.get(suite)
    if not filename:
        return []
    path = Path(__file__).resolve().parent / filename
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith('#')]


def script_timeout(spec):
    """Return a case-specific timeout, falling back to the default limit."""
    for case_path, timeout_seconds in SCRIPT_TIMEOUT_OVERRIDES.items():
        if case_path in spec:
            return timeout_seconds
    return DEFAULT_SCRIPT_TIMEOUT


def run_script_command(index, spec, total, log_dir, base_env=None):
    """Run one daily script command with a per-case timeout and keyword check.

    Mirrors the former pr-deepgemm-ut.py runner: shell execution in a new
    process group, SIGKILL the group on timeout, and treat a failure keyword
    in the output as a failed case.
    """
    log = Path(log_dir) / f'script-{index:04d}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    log.touch()
    start = time.monotonic()
    error = ''
    timeout = script_timeout(spec)
    env = dict(os.environ if base_env is None else base_env)
    try:
        with log.open('w') as output:
            process = subprocess.Popen(
                spec, shell=True, cwd=REPO_ROOT, env=env,
                stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                error = f'Timeout after {timeout}s'
        if not error and code:
            error = f'Exit code {code}'
        if not error and FAILURE_KEYWORD_RE.search(log.read_text(errors='replace')):
            error = 'Failure keyword detected in output'
    except OSError as exc:
        error = str(exc)
    print(f'Script [{index}/{total}] {spec}: {error or "OK"}', flush=True)
    return Result(index, spec, 'script', time.monotonic() - start, error, log)


def run_script_checks(suite, log_dir):
    """Run the board's daily list directly with timeout and keyword checks.

    The per-command execution previously delegated to pr-deepgemm-ut.py is
    inlined here; commands come from daily_<board>.list in this directory.
    """
    commands = load_script_checks(suite)
    if not commands:
        return []
    log_dir = Path(log_dir).resolve()
    env = dict(os.environ)
    env.pop('HGGC_WARM_UP', None)
    return [run_script_command(index, spec, len(commands), log_dir, env)
            for index, spec in enumerate(commands, 1)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--suite', choices=SUITES)
    source.add_argument('--caselist', type=Path)
    parser.add_argument('--skip-checkin', action='store_true',
                        help='Skip the default checkin cases and run only the daily scripts for --suite')
    parser.add_argument('--compile-workers', type=positive_int, default=16)
    parser.add_argument('--timeout', type=positive_int, default=1200, help='Per-case, per-phase timeout in seconds')
    parser.add_argument('--phase', choices=('all', 'compile', 'run'), default='all')
    parser.add_argument('--output-xml', type=Path, default=Path('test-results.xml'))
    parser.add_argument('--log-dir', type=Path, default=Path('checkin-logs'))
    parser.add_argument('--list', action='store_true', help='Print selected functional cases without importing Torch; empty with --skip-checkin')
    parser.add_argument('--list-daily-scripts', action='store_true', help='Print the full shape/tile sweeps without running them')
    args = parser.parse_args(argv)
    if args.skip_checkin:
        if not args.suite:
            parser.error('--skip-checkin requires --suite and cannot be used with --caselist')
        if args.phase == 'compile':
            parser.error('--skip-checkin cannot be used with --phase compile; daily scripts do not support compile-only mode')
    if args.list_daily_scripts:
        if not args.suite:
            parser.error('--list-daily-scripts requires --suite')
        print('\n'.join(load_script_checks(args.suite)))
        return 0
    cases = []
    if not args.skip_checkin:
        path = REPO_ROOT / 'tests/checkin' / SUITES[args.suite] if args.suite else args.caselist
        try:
            cases = load_cases(path)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    if args.list:
        for i, spec in enumerate(cases, 1):
            print(f'{i}: {spec}')
        return 0
    results = []
    if not args.skip_checkin:
        results = execute(cases, args.compile_workers, args.log_dir.resolve(), args.timeout, args.phase)
    if (args.suite and args.phase != 'compile'
            and not any(result.phase == 'compile' and result.error for result in results)):
        results.extend(run_script_checks(args.suite, args.log_dir))
    write_report(results, args.output_xml)
    failed = sum(bool(result.error) for result in results)
    print(f'Phase={args.phase}, cases={len(cases)}, phase results={len(results)}, failures={failed}')
    return int(bool(failed))


if __name__ == '__main__':
    sys.exit(main())
