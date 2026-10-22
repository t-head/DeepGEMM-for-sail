"""Private protocol for run_checkin.py; each process serves just one phase."""
import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import random
import sys
import traceback

# Shared operator tests and their helpers remain in tests/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_deep_gemm as entry


def run(request):
    # Reset per-case controls and RNG, matching the former fresh-process entry.
    torch = entry.torch
    warm_up = bool(os.environ.get('HGGC_WARM_UP'))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)
    entry.set_benchmark(False)
    entry.set_ref_backend('device')
    entry.set_acc_check(not warm_up)
    case = entry.parse_deepgemm_string_re(request['spec'])
    supported, reason = entry.device_version_supported(case)
    if not supported:
        raise RuntimeError(f'Unsupported case: {reason}')
    print(f'Library path: {entry.deep_gemm.__path__}', flush=True)
    print(f'case info: {case}; warm_up={warm_up}', flush=True)
    entry.call_test_func(case['gemm_type'], case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-fd', type=int, required=True)
    args = parser.parse_args()
    for line in sys.stdin:
        request = json.loads(line)
        error = ''
        # Redirect native compiler/runtime output too; the result uses a separate
        # inherited pipe, so arbitrary UT output cannot corrupt the protocol.
        sys.stdout.flush()
        sys.stderr.flush()
        saved = os.dup(1), os.dup(2)
        try:
            with open(request['log'], 'w', buffering=1) as output:
                os.dup2(output.fileno(), 1)
                os.dup2(output.fileno(), 2)
                try:
                    entry.torch.cuda.reset_peak_memory_stats()
                    run(request)
                    entry.torch.cuda.synchronize()
                except (Exception, SystemExit) as exc:
                    error = f'{type(exc).__name__}: {exc}'[:1000]
                    traceback.print_exc()
                finally:
                    print('Device memory peak: allocated={:.3f} GiB, reserved={:.3f} GiB'.format(
                        entry.torch.cuda.max_memory_allocated() / 2**30,
                        entry.torch.cuda.max_memory_reserved() / 2**30), flush=True)
                    gc.collect()
                    entry.torch.cuda.empty_cache()
                    sys.stdout.flush()
                    sys.stderr.flush()
                    ctypes.CDLL(None).fflush(None)
        finally:
            for target, original in enumerate(saved, 1):
                os.dup2(original, target)
                os.close(original)
        os.write(args.result_fd, (json.dumps(dict(index=request['index'], error=error)) + '\n').encode())


if __name__ == '__main__':
    main()
