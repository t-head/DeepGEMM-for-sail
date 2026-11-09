"""Exercise external INT8 LUT fallback through the SM80 MegaMoE public API.

Run on one idle PPU SM80 device with the matching library build:
    env -u DG_USE_MOE_DYNAMIC_TILE -u HGGC_DEVICE_GRAPH \
        CUDA_VISIBLE_DEVICES=0 python tests/test_mega_moe_int8_lut.py -v
"""

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


def run_worker():
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    import deep_gemm as dg
    from test_mega_moe_int8 import assert_output, reference

    assert 'DG_USE_MOE_DYNAMIC_TILE' not in os.environ
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    with tempfile.TemporaryDirectory() as directory:
        dist.init_process_group('nccl', init_method=(Path(directory) / 'rendezvous').as_uri(),
                                rank=0, world_size=1, timeout=timedelta(minutes=5))
        hidden, intermediate = 2048, 256
        buffer = dg.get_symm_buffer_for_int8_mega_moe(dist.group.WORLD, 1, 16, 1, hidden, intermediate)
        try:
            torch.manual_seed(1234)
            w1 = torch.randint(-8, 9, (1, 2 * intermediate, hidden), dtype=torch.int8, device='cuda')
            w2 = torch.randint(-8, 9, (1, hidden, intermediate), dtype=torch.int8, device='cuda')
            s1 = torch.rand(1, 2 * intermediate, device='cuda') * 0.02 + 0.01
            s2 = torch.rand(1, hidden, device='cuda') * 0.02 + 0.01
            weights = dg.transform_weights_for_int8_mega_moe((w1, s1), (w2, s2))
            x = torch.randn(1, hidden, dtype=torch.bfloat16, device='cuda')
            ids = torch.zeros((1, 1), dtype=torch.int64, device='cuda')
            routes = torch.ones((1, 1), device='cuda')
            y = torch.empty_like(x)
            expected = reference(x, ids, routes, w1, s1, w2, s2, 1, {})

            def forward():
                dg.int8_mega_moe_pre_dispatch(x, ids, routes, buffer.x, buffer.x_sf,
                                            buffer.topk_idx, buffer.topk_weights, 1)
                # One output row, expert and route give the default expected_m=2.
                dg.int8_int8_mega_moe(y, *weights, buffer)

            forward()
            torch.cuda.synchronize()
            assert_output(y, expected)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                forward()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                forward()
            for _ in range(3):
                y.fill_(float('nan'))
                graph.replay()
                torch.cuda.synchronize()
                assert_output(y, expected)
        finally:
            buffer.destroy()
            dist.destroy_process_group()


class MegaMoEInt8LutTests(unittest.TestCase):
    def test_scatter_fallback_and_valid_override(self):
        import torch

        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 0):
            self.skipTest('requires PPU SM80')
        device = torch.cuda.get_device_name(0)
        if 'PPU' not in device and 'ZW810' not in device:
            self.skipTest('requires the PPU runtime')
        cases = {
            'default': None,
            'odd_rows': (16, 256, 64, 16, 16, 2),
            'narrow_vector': (16, 64, 64, 16, 32, 2),
            'wide_vector': (16, 512, 64, 16, 64, 2),
            'valid': (16, 256, 64, 16, 32, 2),
        }
        selected = {}
        with tempfile.TemporaryDirectory() as directory:
            lut = Path(directory) / f'{device}_int8_groupedmasked.lut'
            for name, tile in cases.items():
                with self.subTest(case=name):
                    lut.write_text('' if tile is None else '1 2 2048 256 ' + ' '.join(map(str, tile)) + '\n')
                    env = dict(os.environ, SAIL_DEEPGEMM_LUT_PATH=directory, show_log='1')
                    env.pop('DG_USE_MOE_DYNAMIC_TILE', None)
                    env.pop('HGGC_DEVICE_GRAPH', None)
                    # The runtime caches LUT files: each case needs a fresh interpreter.
                    process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                             env=env, capture_output=True, text=True, timeout=600)
                    output = process.stdout + process.stderr
                    self.assertEqual(process.returncode, 0, output)
                    matches = re.findall(
                        r'\[megamoe_int8_scatter\][^\n]*\n'
                        r'tile:\[(\d+),(\d+),(\d+)\] warp:\[(\d+),(\d+)\] stages:(\d+)', output)
                    self.assertTrue(matches, output)
                    selected[name] = tuple(map(int, matches[0]))
                    self.assertTrue(all(tuple(map(int, match)) == selected[name] for match in matches), output)
                    if name == 'valid':
                        self.assertEqual(selected[name], tile)
                    elif name != 'default':
                        self.assertEqual(selected[name], selected['default'])


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        run_worker()
    else:
        unittest.main()
