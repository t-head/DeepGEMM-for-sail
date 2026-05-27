import random
import torch
from typing import Tuple
import os
import deep_gemm
from utils import test_fp8_einsum

def test_fp8_bhr_hdr_bhd(quant_type: str = "block"):
    print('Testing FP8 "bhr, hdr -> bhd":')
    for h, r, d in [(8, 4096, 1024)]:
        for b in (4, 32, 128, 4096, 8192):
            args = {"h":h,"b":b, "r":r, "d":d, "data_type":torch.float8_e4m3fn, "quant_type": quant_type, "expr": 'bhr,hdr->bhd'}
            test_fp8_einsum(args)
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_fp8_bhr_hdr_bhd()
    test_fp8_bhr_hdr_bhd("channel")

