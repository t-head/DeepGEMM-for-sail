"""BF16 x FP32 dense GEMM correctness and API contract tests.

The FP32 weight is decomposed into two BF16 matrices (high + low * scale,
scale = 1/256), the GEMM runs on the BF16 tensor cores with FP32 accumulation
and the result must match the FP32 reference within the reference tolerances
(rtol=0.08, atol=0.01).

API under test:
    deep_gemm.gemm_bf16xfp32(x, w_high, w_low, scale, use_fp32_output,
                             use_splitk, split_flag)          # compatibility wrapper
    deep_gemm.bf16xfp32_gemm_nt(x, w_high, w_low, scale,
                                use_fp32_output, configs=None) # native API
    deep_gemm.get_gemm_bf16xfp32_workspace(n)                 # legacy workspace

Documented contract (see csrc/apis/gemm.hpp):
    x          : BF16, row-major [m, k]
    w_high/low : BF16, row-major [n, k]   (NT layout)
    output     : FP32 (use_fp32_output=True) or BF16, row-major [m, n]
    split_flag : accepted only by the compatibility wrapper; the workspace
                 stays untouched and is not used for partial results.
"""
import os

import pytest
import torch

import deep_gemm

# Same shape sweep as the reference test (sm90 skip is replaced by the PPU
# device availability check below).
M_CASES = [1, 6, 16, 32, 48, 64, 96, 144, 208, 304, 416, 624, 832, 1024, 2048, 4096, 12303, 32768]
N_CASES = [192, 512, 1024, 2048]


def _make_case(m, n, k, device="cuda"):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    x = torch.randn((m, k), dtype=torch.float, device=device).to(torch.bfloat16)
    w = torch.randn((n, k), dtype=torch.float, device=device)
    scale = 1 / 256
    w_high = w.to(torch.bfloat16)
    w_low = ((w - w_high.float()) / scale).to(torch.bfloat16)
    gt = torch.matmul(x.float(), w.t())
    return x, w_high, w_low, scale, gt


@pytest.mark.parametrize("m", M_CASES)
@pytest.mark.parametrize("n", N_CASES)
@pytest.mark.parametrize("k", [4096])
@pytest.mark.parametrize("use_fp32_output", [True, False])
@pytest.mark.parametrize("use_split_flag", [True, False])
def test_gemm_bf16xfp32(m, n, k, use_fp32_output, use_split_flag):
    if not torch.cuda.is_available():
        pytest.skip("no PPU device available")
    x, w_high, w_low, scale, gt = _make_case(m, n, k)

    split_flag = None
    if use_split_flag:
        split_flag = deep_gemm.get_gemm_bf16xfp32_workspace(n)

    my = deep_gemm.gemm_bf16xfp32(x, w_high, w_low, scale,
                                  use_fp32_output, use_split_flag, split_flag)
    assert my.shape == (m, n)
    assert my.dtype == (torch.float32 if use_fp32_output else torch.bfloat16)

    # The kernel never performs a K-split in this build: the split-flag
    # workspace must be untouched (all zeros), exactly what the reference
    # test asserts.
    if use_split_flag:
        assert (split_flag == 0).all()

    torch.testing.assert_close(my.float(), gt, rtol=0.08, atol=0.01)


@pytest.mark.parametrize("k", [512, 1024, 2048, 4096, 7168])
@pytest.mark.parametrize("use_fp32_output", [True, False])
def test_gemm_bf16xfp32_k_sweep(k, use_fp32_output):
    """A few extra K depths beyond the reference test (all even N)."""
    if not torch.cuda.is_available():
        pytest.skip("no PPU device available")
    m, n = 304, 512
    x, w_high, w_low, scale, gt = _make_case(m, n, k)
    my = deep_gemm.bf16xfp32_gemm_nt(x, w_high, w_low, scale, use_fp32_output)
    torch.testing.assert_close(my.float(), gt, rtol=0.08, atol=0.01)


@pytest.fixture
def router_ppu1v5(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("no PPU device available")
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("router fusion regression requires PPU1.5")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    # Exercise automatic defaults without inherited tuning overrides.
    for name in tuple(os.environ):
        if name.startswith("DG_BF16XFP32_"):
            monkeypatch.delenv(name)


def _check_bf16xfp32_router_case(m, use_fp32_output, api):
    n, k = 384, 7168
    x, w_high, w_low, scale, gt = _make_case(m, n, k)
    workspace = None
    if api == "default":
        actual = deep_gemm.bf16xfp32_gemm_nt(
            x, w_high, w_low, scale, use_fp32_output)
    elif api == "explicit":
        configs = (39, 64, 64, 64, 16, 16, 2, (0, 0, 0))
        actual = deep_gemm.bf16xfp32_gemm_nt(
            x, w_high, w_low, scale, use_fp32_output, configs)
    elif api == "compatible":
        workspace = deep_gemm.get_gemm_bf16xfp32_workspace(n, max_tokens=m)
        assert workspace.dtype == torch.int32
        assert workspace.shape == ((m + 15) // 16, (n + 63) // 64)
        assert (workspace == 0).all()
        actual = deep_gemm.gemm_bf16xfp32(
            x, w_high, w_low, scale, use_fp32_output, False, workspace)
    else:
        raise ValueError(f"unknown router API: {api}")
    assert actual.shape == (m, n)
    assert actual.dtype == (torch.float32 if use_fp32_output else torch.bfloat16)
    assert actual.is_contiguous()
    assert torch.allclose(gt, actual.float(), rtol=0.08, atol=0.01), (m, api, use_fp32_output)
    if workspace is not None:
        assert (workspace == 0).all()


@pytest.mark.usefixtures("router_ppu1v5")
@pytest.mark.parametrize("m", [8, 16], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("use_fp32_output", [True, False], ids=["fp32", "bf16"])
@pytest.mark.parametrize("api", ["default", "explicit", "compatible"])
def test_gemm_bf16xfp32_router_decode(m, use_fp32_output, api):
    """Cover the exact small-M router shapes, including packed FP32 defaults."""
    _check_bf16xfp32_router_case(m, use_fp32_output, api)


@pytest.mark.usefixtures("router_ppu1v5")
@pytest.mark.parametrize("m", [128, 256, 512, 1024, 2048, 4096, 8192, 16384],
                         ids=lambda m: f"m{m}")
@pytest.mark.parametrize("use_fp32_output", [True, False], ids=["fp32", "bf16"])
@pytest.mark.parametrize("api", ["default", "explicit", "compatible"])
def test_gemm_bf16xfp32_router_prefill(m, use_fp32_output, api):
    """Cover all eight prefill shapes without turning UT into a timing test."""
    _check_bf16xfp32_router_case(m, use_fp32_output, api)


def test_gemm_bf16xfp32_workspace_shape():
    """Workspace contract: [ceil(max_tokens / 16), ceil(n / 64)] int32 zeros."""
    if not torch.cuda.is_available():
        pytest.skip("no PPU device available")
    ws = deep_gemm.get_gemm_bf16xfp32_workspace(1000, max_tokens=100)
    assert ws.dtype == torch.int32
    assert ws.shape == (7, 16)  # ceil(100/16)=7, ceil(1000/64)=16
    assert (ws == 0).all()


def test_gemm_bf16xfp32_rejects_invalid_inputs():
    if not torch.cuda.is_available():
        pytest.skip("no PPU device available")
    x, w_high, w_low, scale, _ = _make_case(16, 64, 128)
    with pytest.raises(Exception):
        # odd N is rejected (packed BF16 stores need even N)
        w_h = w_high[:63]
        w_l = w_low[:63]
        deep_gemm.bf16xfp32_gemm_nt(x, w_h, w_l, scale)
    with pytest.raises(Exception):
        # FP32 activations are rejected: the kernel is BF16 x (BF16, BF16)
        deep_gemm.bf16xfp32_gemm_nt(x.float(), w_high, w_low, scale)
    with pytest.raises(Exception):
        # mismatched high/low shapes are rejected
        deep_gemm.bf16xfp32_gemm_nt(x, w_high, w_low[:32], scale)
