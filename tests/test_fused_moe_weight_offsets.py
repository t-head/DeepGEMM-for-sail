"""Fused MoE must reject overflowing uint32 expert offsets on the host."""
import pytest
import torch

import deep_gemm as dg


@pytest.mark.parametrize('kind', ['bf16', 'int8', 'fp8-channel', 'fp8-block', 'fp4'])
@pytest.mark.parametrize('shape,overflow', [
    ((1, 65536, 65536), True),   # N*K overflows, even with one expert.
    ((129, 4096, 8192), True),   # Expert 128 starts exactly at 2^32.
    ((128, 4096, 8192), False),  # Entire tensor has 2^32 elements; bases fit.
    ((65, 4096, 8192), False),   # Expert 64 starts at 2^31, valid for uint32.
])
def test_fused_weight_offset_bounds(kind, shape, overflow):
    groups, n, k = shape
    dtype = {'bf16': torch.bfloat16, 'int8': torch.int8, 'fp4': torch.uint8}.get(
        kind, torch.float8_e4m3fn)
    # The public API requires accelerator tensors. Reserve uninitialized weights
    # without reading/writing them; empty output avoids launching accepted shapes.
    required_bytes = groups * n * k * torch.empty((), dtype=dtype).element_size()
    free_bytes, _ = torch.cuda.mem_get_info()
    available_bytes = free_bytes + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    if available_bytes < required_bytes + (1 << 30):
        pytest.skip('weight-offset boundary test needs up to 9 GiB of free device memory')
    a = torch.empty((1, k), dtype=dtype, device='cuda')
    b = torch.empty(shape, dtype=dtype, device='cuda')
    out = torch.empty((0, n), dtype=torch.bfloat16, device='cuda')
    metadata = torch.empty((0,), dtype=torch.int32, device='cuda')
    if kind == 'bf16':
        api = dg.m_grouped_bf16_gemm_nt_fused
    else:
        if kind == 'fp4':
            sa = torch.empty((1, (k + 31) // 32), dtype=torch.uint16, device='cuda')
            sb = torch.empty((groups, (k + 31) // 32, n),
                             dtype=torch.uint16, device='cuda').transpose(1, 2)
            api = dg.m_grouped_fp4_gemm_nt_fused
        elif kind == 'fp8-block':
            sa = torch.empty((1, k // 128), device='cuda')
            sb = torch.empty((groups, n // 128, k // 128), device='cuda')
            api = dg.m_grouped_fp8_gemm_nt_fused
        else:
            sa = torch.empty((1, 1), device='cuda')
            sb = torch.empty((groups, n, 1), device='cuda')
            api = (dg.m_grouped_int8_gemm_nt_fused if kind == 'int8'
                   else dg.m_grouped_fp8_gemm_nt_fused)
        a, b = (a, sa), (b, sb)

    def launch():
        api(a, b, out, metadata, metadata, metadata, metadata,
            (1, 16, 64, 128, 16, 32, 2))

    if overflow:
        with pytest.raises(RuntimeError, match='kernel uint32 weight-offset overflow'):
            launch()
    else:
        launch()


def test_bf16_fused_weight_offset_smoke():
    a = torch.ones((1, 128), dtype=torch.bfloat16, device='cuda')
    b = torch.ones((2, 64, 128), dtype=torch.bfloat16, device='cuda')
    b[1].fill_(2)
    ids = torch.ones((1, 1), dtype=torch.int32, device='cuda')
    cfg, rows, meta, tokens, blocks, _, _ = dg.moe_align_block_size(
        a, b, ids, config=(1, 16, 64, 128, 16, 32, 2))
    out = torch.full((1, 64), -1, dtype=torch.bfloat16, device='cuda')
    dg.m_grouped_bf16_gemm_nt_fused(a, b, out, rows, meta, tokens, blocks, cfg)
    torch.testing.assert_close(out, torch.full_like(out, 256), rtol=0, atol=0)
