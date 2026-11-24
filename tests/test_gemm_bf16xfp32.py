"""BF16 x FP32 dense GEMM correctness and API contract tests.

The FP32 weight is decomposed into two BF16 matrices (high + low * scale,
scale = 1/256), the GEMM runs on the BF16 tensor cores with FP32 accumulation
and both FP32 references must satisfy the ordinary BF16 dense calc_diff < 0.001:
computation uses X @ high.T + scale * (X @ low.T); approximation quality uses
X @ original_weight.T. BF16 references round only after the final FP32 merge.
Known-answer regressions additionally require zero wrong elements and zero
modified guard bytes. PPU1.0 code is retained, but execution is disabled.

API under test:
    deep_gemm.gemm_bf16xfp32(x, w_high, w_low, scale,
                             use_fp32_output=False)            # convenience wrapper
    deep_gemm.bf16xfp32_gemm_nt(x, w_high, w_low, scale,
                                use_fp32_output, configs=None) # native API

Documented contract (see csrc/apis/gemm.hpp):
    x          : BF16, row-major [m, k]
    w_high/low : BF16, row-major [n, k]   (NT layout)
    output     : FP32 (use_fp32_output=True) or BF16, row-major [m, n]
"""
import inspect
import os

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import deep_gemm
from deep_gemm.testing import calc_diff

# Same shape sweep as the reference test (sm90 skip is replaced by the PPU
# device availability check below).
M_CASES = [1, 6, 16, 32, 48, 64, 96, 144, 208, 304, 416, 624, 832, 1024, 2048, 4096, 12303, 32768]
N_CASES = [192, 512, 1024, 2048]


def test_gemm_bf16xfp32_signature(monkeypatch):
    """The convenience wrapper accepts only the five documented arguments."""
    signature = inspect.signature(deep_gemm.gemm_bf16xfp32)
    assert tuple(signature.parameters) == (
        "x", "w_high", "w_low", "scale", "use_fp32_output")
    assert signature.parameters["use_fp32_output"].default is False
    monkeypatch.setattr(deep_gemm, "bf16xfp32_gemm_nt", lambda *args: args)
    inputs = (object(), object(), object(), 1 / 256)
    assert deep_gemm.gemm_bf16xfp32(*inputs) == (*inputs, False)
    assert deep_gemm.gemm_bf16xfp32(*inputs, use_fp32_output=True) == (*inputs, True)
    with pytest.raises(TypeError):
        deep_gemm.gemm_bf16xfp32(*inputs, False, False)
    with pytest.raises(TypeError):
        deep_gemm.gemm_bf16xfp32(*inputs, False, False, object())
    with pytest.raises(TypeError):
        deep_gemm.gemm_bf16xfp32(*inputs, workspace=object())


@pytest.fixture(autouse=True)
def _disable_tf32_golden():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.fixture
def ppu1v5():
    if not torch.cuda.is_available():
        pytest.skip("no PPU device available")
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("BF16xFP32 execution requires PPU1.5; PPU1.0 is disabled")


def _assert_accuracy(actual, correctness, quality):
    # Aggregate error is not a per-element tolerance; exact/guard tests below
    # cover the sparse-error and out-of-bounds blind spots of calc_diff.
    assert actual.dtype in (torch.float32, torch.bfloat16)
    assert torch.isfinite(actual).all(), "non-finite output"
    for label, reference in (("computation", correctness), ("quality", quality)):
        assert reference.dtype == torch.float32
        assert actual.shape == reference.shape
        assert torch.isfinite(reference).all(), f"non-finite {label} reference"
        reference = reference.to(actual.dtype)
        diff = float(calc_diff(actual, reference))
        assert diff < 0.001, (label, diff)


def _assert_exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    wrong_elements = int((actual != expected).sum())
    assert wrong_elements == 0, ("wrong output elements", wrong_elements)


def _assert_guard(storage, before, start, count):
    after = torch.cat((storage[:start], storage[start + count:])).view(torch.uint8)
    modified_bytes = int((after != before).sum())
    assert modified_bytes == 0, ("modified output guard bytes", modified_bytes)


class _GuardedOutput(TorchDispatchMode):
    """Intercept the native output allocation without changing the public API."""
    def __init__(self, shape, dtype, device):
        super().__init__()
        self.shape, self.dtype, self.device = shape, dtype, device
        # Preserve the allocator's 256-byte alignment for both output dtypes.
        self.start = 128
        self.count = shape[0] * shape[1]
        self.storage = torch.full((self.count + 256,), -123, dtype=dtype, device=device)
        self.output = self.storage[self.start:self.start + self.count].view(shape)
        self.before = torch.cat((self.storage[:self.start], self.storage[self.start + self.count:])).view(torch.uint8).clone()
        self.captured = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if (not self.captured and func == torch.ops.aten.empty.memory_format
                and tuple(args[0]) == self.shape and kwargs.get("dtype") == self.dtype
                and kwargs.get("device") == self.device):
            self.captured = True
            return self.output
        return func(*args, **kwargs)

    def check(self, actual):
        assert self.captured, "native output allocation was not guarded"
        assert actual.data_ptr() == self.output.data_ptr()
        _assert_guard(self.storage, self.before, self.start, self.count)


def _call_api(api, x, high, low, scale, use_fp32_output):
    if api == "wrapper":
        return deep_gemm.gemm_bf16xfp32(x, high, low, scale, use_fp32_output)
    if api == "native":
        return deep_gemm._C.bf16xfp32_gemm_nt(x, high, low, scale, use_fp32_output)
    if api == "explicit":
        configs = (39, 64, 64, 64, 16, 16, 2, (0, 0, 0))
        return deep_gemm.bf16xfp32_gemm_nt(x, high, low, scale, use_fp32_output, configs)
    assert api == "default"
    return deep_gemm.bf16xfp32_gemm_nt(x, high, low, scale, use_fp32_output)


def _make_case(m, n, k, device="cuda"):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    x = torch.randn((m, k), dtype=torch.float, device=device).to(torch.bfloat16)
    w = torch.randn((n, k), dtype=torch.float, device=device)
    scale = 1 / 256
    w_high = w.to(torch.bfloat16)
    w_low = ((w - w_high.float()) / scale).to(torch.bfloat16)
    correctness = torch.matmul(x.float(), w_high.float().t())
    correctness = correctness + scale * torch.matmul(x.float(), w_low.float().t())
    quality = torch.matmul(x.float(), w.t())
    return x, w_high, w_low, scale, (correctness, quality)


@pytest.mark.usefixtures("ppu1v5")
@pytest.mark.parametrize("m", M_CASES)
@pytest.mark.parametrize("n", N_CASES)
@pytest.mark.parametrize("k", [4096])
@pytest.mark.parametrize("use_fp32_output", [True, False])
def test_gemm_bf16xfp32(m, n, k, use_fp32_output):
    if not torch.cuda.is_available():
        pytest.skip("no PPU device available")
    x, w_high, w_low, scale, gt = _make_case(m, n, k)

    my = deep_gemm.gemm_bf16xfp32(x, w_high, w_low, scale, use_fp32_output)
    assert my.shape == (m, n)
    assert my.dtype == (torch.float32 if use_fp32_output else torch.bfloat16)

    _assert_accuracy(my, *gt)


@pytest.mark.usefixtures("ppu1v5")
@pytest.mark.parametrize("k", [512, 1024, 2048, 4096, 7168])
@pytest.mark.parametrize("use_fp32_output", [True, False])
def test_gemm_bf16xfp32_k_sweep(k, use_fp32_output):
    """A few extra K depths beyond the reference test (all even N)."""
    if not torch.cuda.is_available():
        pytest.skip("no PPU device available")
    m, n = 304, 512
    x, w_high, w_low, scale, gt = _make_case(m, n, k)
    my = deep_gemm.bf16xfp32_gemm_nt(x, w_high, w_low, scale, use_fp32_output)
    _assert_accuracy(my, *gt)


@pytest.fixture
def router_ppu1v5(monkeypatch, ppu1v5):
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
    if api == "default":
        actual = deep_gemm.bf16xfp32_gemm_nt(
            x, w_high, w_low, scale, use_fp32_output)
    elif api == "explicit":
        configs = (39, 64, 64, 64, 16, 16, 2, (0, 0, 0))
        actual = deep_gemm.bf16xfp32_gemm_nt(
            x, w_high, w_low, scale, use_fp32_output, configs)
    elif api == "wrapper":
        actual = deep_gemm.gemm_bf16xfp32(
            x, w_high, w_low, scale, use_fp32_output)
    else:
        raise ValueError(f"unknown router API: {api}")
    assert actual.shape == (m, n)
    assert actual.dtype == (torch.float32 if use_fp32_output else torch.bfloat16)
    assert actual.is_contiguous()
    _assert_accuracy(actual, *gt)


@pytest.mark.usefixtures("router_ppu1v5")
@pytest.mark.parametrize("m", [8, 16], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("use_fp32_output", [True, False], ids=["fp32", "bf16"])
@pytest.mark.parametrize("api", ["default", "explicit", "wrapper"])
def test_gemm_bf16xfp32_router_decode(m, use_fp32_output, api):
    """Cover the exact small-M router shapes, including packed FP32 defaults."""
    _check_bf16xfp32_router_case(m, use_fp32_output, api)


@pytest.mark.usefixtures("router_ppu1v5")
@pytest.mark.parametrize("m", [128, 256, 512, 1024, 2048, 4096, 8192, 16384],
                         ids=lambda m: f"m{m}")
@pytest.mark.parametrize("use_fp32_output", [True, False], ids=["fp32", "bf16"])
@pytest.mark.parametrize("api", ["default", "explicit", "wrapper"])
def test_gemm_bf16xfp32_router_prefill(m, use_fp32_output, api):
    """Cover all eight prefill shapes without turning UT into a timing test."""
    _check_bf16xfp32_router_case(m, use_fp32_output, api)


@pytest.mark.usefixtures("ppu1v5")
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


@pytest.mark.parametrize("api", ["default", "explicit", "wrapper", "native"])
@pytest.mark.parametrize("use_fp32_output", [True, False])
@pytest.mark.parametrize("shape", [(5, 14, 13), (0, 14, 13), (5, 0, 13), (5, 14, 0)])
def test_gemm_bf16xfp32_ppu10_disabled(api, use_fp32_output, shape):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("PPU1.0 disablement regression requires PPU1.0")
    m, n, k = shape
    x = torch.zeros((m, k), dtype=torch.bfloat16, device="cuda")
    high = torch.zeros((n, k), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="BF16xFP32 GEMM is disabled on PPU1.0"):
        _call_api(api, x, high, high, 1 / 256, use_fp32_output)


@pytest.mark.usefixtures("router_ppu1v5")
@pytest.mark.parametrize("shape", [(5, 14, 13), (9, 130, 65), (8, 384, 7168), (128, 384, 7168)])
@pytest.mark.parametrize("use_fp32_output", [True, False])
@pytest.mark.parametrize("api", ["default", "explicit", "wrapper"])
def test_gemm_bf16xfp32_exact_and_guard(shape, use_fp32_output, api):
    # Two exact K terms distinguish a final FP32 merge from early BF16 rounding.
    # Spatially varying answers catch shifted stores; the last term catches K tails.
    m, n, k = shape
    x = torch.zeros((m, k), dtype=torch.bfloat16, device="cuda")
    high = torch.zeros((n, k), dtype=torch.bfloat16, device="cuda")
    low = torch.zeros_like(high)
    rows = 1 + torch.arange(m, device=x.device) % 4
    cols = 1 + torch.arange(n, device=x.device) % 3
    x[:, 0] = rows
    x[:, -1] = rows
    high[:, 0] = cols
    high[:, -1] = cols / 256
    low[:, -1] = cols * 2
    dtype = torch.float32 if use_fp32_output else torch.bfloat16
    # Independent known answer: (1 + 1/256 + 2/256) * row * column.
    expected = (rows[:, None].float() * cols[None, :] * (259 / 256)).to(dtype)
    guard = _GuardedOutput((m, n), dtype, x.device)
    with guard:
        actual = _call_api(api, x, high, low, 1 / 256, use_fp32_output)
    torch.cuda.synchronize()
    guard.check(actual)
    _assert_exact(actual, expected)


def test_gemm_bf16xfp32_accuracy_standard_selfcheck():
    ones = torch.ones((3, 4), dtype=torch.float32)
    _assert_accuracy(ones, ones, ones)
    zeros = torch.zeros_like(ones)
    _assert_accuracy(zeros, zeros, zeros)
    for correctness, quality in ((ones * 2, ones), (ones, ones * 2)):
        with pytest.raises(AssertionError):
            _assert_accuracy(ones, correctness, quality)
    # BF16 output is compared with the final rounded FP32 reference.
    reference = ones * (259 / 256)
    _assert_accuracy(reference.bfloat16(), reference, reference)
    for value in (float("nan"), float("inf")):
        with pytest.raises(AssertionError):
            _assert_accuracy(ones * value, ones, ones)
        with pytest.raises(AssertionError):
            _assert_accuracy(ones, ones * value, ones)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gemm_bf16xfp32_exact_guard_selfcheck(dtype):
    expected = torch.full((9, 130), 259 / 256, dtype=dtype)
    _assert_exact(expected, expected)
    missing_low = torch.full_like(expected, 257 / 256)
    early_round = torch.full_like(expected, 258 / 256)
    missed_tail = torch.ones_like(expected)
    shifted = expected.clone()
    shifted[0, 0] = 0
    assert float(calc_diff(shifted, expected)) < 0.001
    for wrong in (missing_low, early_round, missed_tail, shifted):
        with pytest.raises(AssertionError):
            _assert_exact(wrong, expected)
    storage = torch.full((16,), -123, dtype=dtype)
    before = torch.cat((storage[:4], storage[12:])).view(torch.uint8).clone()
    storage[4:12] = 1
    _assert_guard(storage, before, 4, 8)
    for index in (0, 15):
        storage[index] = 0
        with pytest.raises(AssertionError):
            _assert_guard(storage, before, 4, 8)
        storage[index] = -123
