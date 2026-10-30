"""Device regression checks for compile-only input layouts (requires Torch/PPU)."""
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import utils


class WarmupInputsTests(unittest.TestCase):
    def test_w4_layout_matches_real_quantization_without_reference_allocation(self):
        y = torch.randn((2, 64, 128), dtype=torch.bfloat16, device='cuda')
        for mode in ('w4a16', 'w4fa16', 'w4fa16_mma', 'w4fa8_mma'):
            with self.subTest(mode=mode), mock.patch.dict(os.environ, HGGC_WARM_UP=''):
                real = utils.prepare_w4_weights(y, 32, mode)
                # Runtime treats every nonempty value, including "0", as true.
                for flag in ('1', '0'):
                    with mock.patch.dict(os.environ, HGGC_WARM_UP=flag), \
                            mock.patch.object(utils, 'quant_w4', side_effect=AssertionError('quantized during warmup')):
                        warm = utils.prepare_w4_weights(y, 32, mode)
                    self.assertIs(warm[0], y)
                    for actual, expected in zip(warm[1:3], real[1:3]):
                        self.assertEqual((actual.shape, actual.stride(), actual.dtype, actual.device),
                                         (expected.shape, expected.stride(), expected.dtype, expected.device))
                    self.assertEqual(warm[3], 1.0)

    def test_disable_accuracy_alone_keeps_real_quantization(self):
        y = torch.randn((1, 64, 128), dtype=torch.bfloat16, device='cuda')
        previous = utils.get_acc_check()
        try:
            utils.set_acc_check(False)
            with mock.patch.dict(os.environ, HGGC_WARM_UP=''), \
                    mock.patch.object(utils, 'quant_w4', wraps=utils.quant_w4) as quantize:
                result = utils.prepare_w4_weights(y, 32, 'w4fa16')
                quantize.assert_called_once()
                self.assertIsNot(result[0], y)
        finally:
            utils.set_acc_check(previous)


if __name__ == '__main__':
    unittest.main()
