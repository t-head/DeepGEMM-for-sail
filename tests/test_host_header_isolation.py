"""Host header-isolation regression. Pure source scan: no PPU device, no SDK, no build.

Run with: python tests/test_host_header_isolation.py -v

`acblasLt.h` pulls the SDK's native `hggc_fp16.h`, which defines `__half` and
`__half_raw` behind its own `__FP16_DEFINED__` guard. The SDK's CUDA compatibility
header `cuda_fp16.h` defines the same names but never sets that guard, so the yield
never triggers and a translation unit holding both definitions fails to compile with
~300 redefinition errors. `ATen/cuda/CUDAContext.h` is what drags `cuda_fp16.h` in;
the narrower `c10/cuda/*` headers and `cuda_runtime_api.h` do not.

DeepGEMM therefore keeps the host sources off ATen's CUDA headers. The single
exception is `at::cuda::getCurrentCUDABlasLtHandle`, which lives behind
`PYTORCH_SUPPORTS_GET_ACBLASLT_HANDLE` (off unless a build sets
`DG_ENABLE_PYTORCH_ACBLASLT_HANDLE=1`).

These checks are what the full build would otherwise be needed to catch, so they
exist to fail fast in environments that cannot build, and to keep the reason for the
restriction attached to the code.
"""
import re
import unittest
from pathlib import Path

CSRC = Path(__file__).resolve().parent.parent / 'csrc'
SOURCE_SUFFIXES = ('.hpp', '.cpp', '.h', '.cuh', '.cu')

# The one region allowed to use ATen's CUDA headers.
GUARD = 'PYTORCH_SUPPORTS_GET_ACBLASLT_HANDLE'

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]ATen/cuda/')
USAGE_RE = re.compile(r'\bat::cuda::')
# Both the `#if`/`#elif` that opens the allowed region and the `#if not ...`
# diagnostic that closes off the unsupported runtime mode mention the guard.
IF_RE = re.compile(r'^\s*#\s*(if|ifdef|ifndef|elif)\b')
ENDIF_RE = re.compile(r'^\s*#\s*endif\b')


def strip_comments(text: str) -> str:
    """Blank out comments while preserving line numbering."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text.startswith('//', i):
            end = text.find('\n', i)
            end = n if end < 0 else end
            out.append(' ' * (end - i))
            i = end
        elif text.startswith('/*', i):
            end = text.find('*/', i + 2)
            end = n if end < 0 else end + 2
            out.append(''.join(c if c == '\n' else ' ' for c in text[i:end]))
            i = end
        elif text[i] in '"\'':
            quote, j = text[i], i + 1
            while j < n and text[j] != quote:
                j += 2 if text[j] == '\\' else 1
            j = min(j + 1, n)
            out.append(text[i:j])
            i = j
        else:
            out.append(text[i])
            i += 1
    return ''.join(out)


def offending_lines(path: Path):
    """ATen/CUDA includes and uses outside any conditional mentioning the guard."""
    lines = strip_comments(path.read_text(encoding='utf-8', errors='replace')).splitlines()
    # Depth of the #if nest, and the depths whose condition named the guard.
    depth, guarded_depths, found = 0, set(), []
    for number, line in enumerate(lines, start=1):
        if IF_RE.match(line):
            if not line.lstrip().startswith('#elif') and not line.lstrip().startswith('# elif'):
                depth += 1
            guarded_depths.discard(depth)
            if GUARD in line:
                guarded_depths.add(depth)
        elif ENDIF_RE.match(line):
            guarded_depths.discard(depth)
            depth = max(depth - 1, 0)
        elif not guarded_depths and (INCLUDE_RE.match(line) or USAGE_RE.search(line)):
            found.append((number, line.strip()))
    return found


class TestHostHeaderIsolation(unittest.TestCase):
    def setUp(self):
        self.sources = sorted(p for p in CSRC.rglob('*') if p.suffix in SOURCE_SUFFIXES)
        self.assertTrue(self.sources, f'no host sources found under {CSRC}')

    def test_no_unguarded_aten_cuda(self):
        """ATen/CUDA headers and `at::cuda::` stay behind the acBLASLt guard."""
        offenders = {
            path.relative_to(CSRC).as_posix(): hits
            for path in self.sources
            if (hits := offending_lines(path))
        }
        self.assertEqual(
            offenders, {},
            'ATen/CUDA used outside `#if ' + GUARD + '`. These headers define `__half` '
            'a second time in any translation unit that also includes `acblasLt.h`. Use '
            '`c10::cuda::getCurrentCUDAStream()`, `c10::cuda::current_device()` or '
            '`device_runtime->get_prop()` instead:\n' + '\n'.join(
                f'  {name}:{number}: {line}'
                for name, hits in sorted(offenders.items()) for number, line in hits
            ))

    def test_guard_defaults_to_off(self):
        """The one allowed ATen/CUDA user is opt-in, so default builds stay isolated."""
        text = (CSRC / 'jit' / 'device_runtime.hpp').read_text(encoding='utf-8')
        for needed in ('#define DG_ENABLE_PYTORCH_ACBLASLT_HANDLE 0', f'#define {GUARD}'):
            self.assertTrue(needed in text,
                            f'csrc/jit/device_runtime.hpp must contain `{needed}`, so that a '
                            'default build keeps ATen/CUDA out of the translation unit')

    def test_scan_detects_a_violation(self):
        """The scan would actually fail: it flags an unguarded use, and not a guarded one."""
        bad = CSRC / 'jit' / 'device_runtime.hpp'
        self.assertEqual(offending_lines(bad), [], 'fixture file must itself be clean')

        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            unguarded = Path(directory) / 'unguarded.hpp'
            unguarded.write_text('#include <ATen/cuda/CUDAContext.h>\n'
                                 'auto s = at::cuda::getCurrentCUDAStream();\n')
            self.assertEqual(len(offending_lines(unguarded)), 2)

            guarded = Path(directory) / 'guarded.hpp'
            guarded.write_text(f'#if {GUARD}\n'
                               '#include <ATen/cuda/CUDAContext.h>\n'
                               '#endif\n'
                               f'#if {GUARD}\n'
                               'auto s = at::cuda::getCurrentCUDABlasLtHandle();\n'
                               '#endif\n')
            self.assertEqual(offending_lines(guarded), [])

            commented = Path(directory) / 'commented.hpp'
            commented.write_text('// mentions at::cuda::getCurrentCUDAStream in prose\n'
                                 '/* and <ATen/cuda/CUDAContext.h> in a block */\n')
            self.assertEqual(offending_lines(commented), [])


if __name__ == '__main__':
    unittest.main()
