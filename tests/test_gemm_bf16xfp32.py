"""BF16 x FP32 dense GEMM correctness and API contract tests.

The FP32 weight is decomposed into two BF16 matrices (high + low * scale,
scale = 1/256), the GEMM runs on the BF16 matrix units with FP32 accumulation
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
import math
import os
from pathlib import Path
import re
import subprocess
import sys

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


def run_bf16xfp32_case(args, check_accuracy=True, ref_backend="device"):
    """Replay operator parameters with fresh dual-B inputs, not saved tensor data."""
    m, n, k = (args[name] for name in ("m", "n", "k"))
    scale = float(args["scale"])
    fp32 = args["use_fp32_output"]
    if min(m, n, k) < 0 or n % 2 or not math.isfinite(scale) or scale <= 0 or fp32 not in (0, 1):
        raise ValueError("Invalid BF16xFP32 replay dimensions, scale or output type")
    if args.get("with_bias") or args.get("accumulate", "none") != "none":
        raise ValueError("BF16xFP32 replay does not support bias or C accumulation")
    config = args.get("configs")
    if config is not None:
        if not isinstance(config, (list, tuple)) or len(config) != 7 or any(type(v) is not int for v in config):
            raise ValueError("BF16xFP32 configs require SMS/BM/BN/BK/WM/WN/stages")
        config = tuple(config) + ((0, 0, 0),)
    compiling = bool(os.environ.get("HGGC_WARM_UP")) or deep_gemm.get_compile_mode() == 1
    check = check_accuracy and not compiling
    expected_error = args.get("expect_error")
    x = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    high = torch.empty((n, k), dtype=torch.bfloat16, device="cuda")
    low = torch.empty_like(high)
    references = None
    if not compiling and not expected_error:
        x.normal_()
        weight = torch.randn((n, k), dtype=torch.float32, device=x.device)
        high.copy_(weight)
        low.copy_((weight - high.float()) / scale)
        if check:
            lhs, hi, lo, original = x.float(), high.float(), low.float(), weight
            if ref_backend == "host":
                lhs, hi, lo, original = (t.cpu() for t in (lhs, hi, lo, original))
            elif ref_backend != "device":
                raise ValueError("Unknown reference backend")
            backends = (torch.backends.cuda.matmul, torch.backends.cudnn)
            previous = tuple(b.allow_tf32 for b in backends)
            try:
                for backend in backends:
                    backend.allow_tf32 = False
                references = (lhs @ hi.t() + scale * (lhs @ lo.t()), lhs @ original.t())
                references = tuple(t.to(x.device) for t in references)
            finally:
                for backend, setting in zip(backends, previous):
                    backend.allow_tf32 = setting
    previous_sms = deep_gemm.get_num_sms()
    dtype = torch.float32 if fp32 else torch.bfloat16
    guard = _GuardedOutput((m, n), dtype, x.device)
    try:
        if "num_sms" in args:
            deep_gemm.set_num_sms(args["num_sms"])
        try:
            with guard:
                actual = deep_gemm.bf16xfp32_gemm_nt(x, high, low, scale, bool(fp32), config)
        except RuntimeError as error:
            if not expected_error or expected_error not in str(error):
                raise
            assert not guard.captured, "rejected call allocated an output"
            print("Passed expected BF16xFP32 API rejection")
            return
        if expected_error:
            raise AssertionError("BF16xFP32 accepted unsupported arguments")
        torch.cuda.synchronize()
        if m and n and k:
            guard.check(actual)
            if compiling:
                _assert_exact(actual, torch.full_like(actual, -123))
        if check:
            if m and n:
                _assert_accuracy(actual, *references)
            else:
                for reference in references:
                    _assert_exact(actual, reference.to(dtype))
        assert actual.shape == (m, n) and actual.dtype == dtype
        print("Passed BF16xFP32 replay with acc_check" if check else "Passed BF16xFP32 replay without acc_check")
        return actual
    finally:
        if "num_sms" in args:
            deep_gemm.set_num_sms(previous_sms)


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
@pytest.mark.parametrize("shape", [(5, 14, 13), (9, 130, 65), (8, 384, 7168), (128, 384, 7168),
                                 (17, 194, 129), (63, 258, 257), (129, 386, 513),
                                 (1023, 384, 65), (1025, 384, 65), (2049, 386, 129)]
                         + [(m, 384, 7168) for m in (16, 256, 512, 1024, 2048, 4096, 8192, 16384)])
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


@pytest.mark.usefixtures("router_ppu1v5")
def test_gemm_bf16xfp32_tile_cache_is_not_tensor_cache():
    previous_sms = deep_gemm.get_num_sms()
    try:
        # Alternate every selection-key component, then return to the first key
        # with new tensors. Shape/dtype/SM caching must not retain tensor data.
        cases = [(17, 194, 65, 39, True), (9, 130, 129, 20, False),
                 (63, 258, 257, 39, False), (17, 194, 65, 20, True),
                 (17, 194, 65, 39, False), (17, 194, 65, 39, True)]
        for index, (m, n, k, sms, fp32) in enumerate(cases):
            deep_gemm.set_num_sms(sms)
            x, high, low, scale, gt = _make_case(m, n, k)
            x = x * (2 ** index)
            gt = tuple(reference * (2 ** index) for reference in gt)
            actual = _call_api("default", x, high, low, scale, fp32)
            _assert_accuracy(actual, *gt)
    finally:
        deep_gemm.set_num_sms(previous_sms)


@pytest.mark.usefixtures("router_ppu1v5")
@pytest.mark.parametrize(
    "m, n, k, use_fp32_output, tile, warp, stages, grid, smem, pipeline, static_router, static_stages, split_k, cta_k, k_order", [
    (8,384,7168,False,(16,16,128),(16,16,128),4,48,65536,"packed_dual_n",1,0,2,0,0),
    (16,384,7168,False,(16,16,128),(16,16,128),4,48,65536,"packed_dual_n",1,0,2,0,0),
    (128,384,7168,False,(48,32,128),(16,16,128),5,36,143360,"double_buffer",1,0,1,0,0),
    (256,384,7168,False,(96,32,128),(48,16,64),6,36,245760,"double_buffer",1,0,1,1,0),
    (512,384,7168,False,(96,64,64),(48,16,64),5,36,143360,"mma_s2r_overlap",1,0,1,0,0),
    (1024,384,7168,False,(192,64,64),(48,32,64),4,36,163840,"mma_s2r_overlap",1,0,1,0,0),
    (2048,384,7168,False,(256,96,64),(32,48,64),4,32,229376,"overlap_prologue",1,0,1,0,0),
    (4096,384,7168,False,(320,128,64),(80,32,64),3,39,221184,"double_buffer",0,0,1,0,0),
    (8192,384,7168,False,(320,128,64),(80,32,64),3,39,221184,"overlap_valid",1,0,1,0,1),
    (16384,384,7168,False,(256,128,64),(64,32,64),4,39,262144,"overlap_valid",1,1,1,0,6),
    (32768,384,7168,False,(256,128,64),(64,16,64),4,39,262144,"double_buffer",0,0,1,0,0),
    (8,384,7168,True,(16,16,128),(16,16,128),4,48,65536,"packed_dual_n",1,0,2,0,0),
    (16,384,7168,True,(16,16,128),(16,16,128),4,48,65536,"packed_dual_n",1,0,2,0,0),
    (128,384,7168,True,(64,32,128),(16,16,128),4,24,131072,"early_g2s",1,0,1,0,0),
    (256,384,7168,True,(96,32,64),(32,16,64),7,36,143360,"early_g2s",1,0,1,0,0),
    (512,384,7168,True,(128,64,64),(32,16,64),4,24,131072,"double_buffer",1,0,1,0,0),
    (1024,384,7168,True,(192,64,64),(32,16,64),4,36,163840,"double_buffer",0,0,1,0,0),
    (2048,384,7168,True,(128,192,64),(32,32,64),4,32,262144,"double_buffer",0,0,1,0,0),
    (4096,384,7168,True,(320,128,64),(80,32,64),2,39,147456,"valid_epilogue",1,0,1,0,0),
    (8192,384,7168,True,(320,128,64),(80,32,64),3,39,221184,"double_buffer",0,0,1,0,0),
    (16384,384,7168,True,(256,128,64),(64,32,64),4,39,262144,"overlap_valid",1,0,1,0,0),
    (32768,384,7168,True,(256,128,64),(64,16,64),4,39,262144,"double_buffer",0,0,1,0,0),
    (512,2048,4096,False,(128,256,64),(64,32,64),3,32,245760,"double_buffer",0,0,1,0,0),
    (512,2048,4096,True,(128,256,64),(64,32,64),3,32,245760,"double_buffer",0,0,1,0,0),
])
@pytest.mark.parametrize("shortcut_disabled", [0, 1])
def test_gemm_bf16xfp32_exact_lut_and_explicit_priority(
        m, n, k, use_fp32_output, tile, warp, stages, grid, smem, pipeline,
        static_router, static_stages, split_k, cta_k, k_order, shortcut_disabled, monkeypatch, capfd):
    import ctypes
    if (int(os.environ.get("DG_JIT_USE_HGRTC", "0")) and not use_fp32_output
            and n == 384 and m != 32768):
        pytest.skip("packed-store LUT configurations require HGRTC validation")
    previous_sms = deep_gemm.get_num_sms()
    monkeypatch.setenv("show_log", "1")
    monkeypatch.setenv("DG_JIT_DISABLE_SHORTCUT_CACHE", str(shortcut_disabled))
    try:
        deep_gemm.set_num_sms(39)
        x, high, low, scale, gt = _make_case(m, n, k)
        for multiplier in (1, 2):
            actual = _call_api("default", x * multiplier, high.clone(), low.clone(), scale, use_fp32_output)
            _assert_accuracy(actual, *(r * multiplier for r in gt))
            torch.cuda.synchronize()
            ctypes.CDLL(None).fflush(None)
            log = capfd.readouterr().out
            assert f"ThreadblockShape{list(tile)}, WarpShape{list(warp)}, num_stages:{stages}\n" in log
            assert f"threadblock_count:{grid}\n" in log and f"SMSIZE:{smem}," in log
            assert f"dual_b_exact_lut:1, cta_k:{cta_k}, k_order:{k_order}\n" in log
            assert f"dual_b_pipeline:{pipeline}\n" in log
            assert f"dual_b_static_router:{static_router}, static_stages:{static_stages}\n" in log
            assert f"dual_b_decode_split_k:{split_k}\n" in log
        actual = _call_api("explicit", x, high, low, scale, use_fp32_output)
        _assert_accuracy(actual, *gt)
        torch.cuda.synchronize()
        ctypes.CDLL(None).fflush(None)
        log = capfd.readouterr().out
        assert "ThreadblockShape[64, 64, 64], WarpShape[16, 16, 64], num_stages:2\n" in log
        assert "dual_b_exact_lut:0, cta_k:0, k_order:0\n" in log
        assert "dual_b_static_router:0, static_stages:0\n" in log
        assert "dual_b_decode_split_k:1\n" in log and "dual_b_pipeline:double_buffer\n" in log
    finally:
        deep_gemm.set_num_sms(previous_sms)


@pytest.mark.usefixtures("router_ppu1v5")
@pytest.mark.parametrize("change", ["m", "n", "k", "sms", "sms64", "scale", "near_scale", "fp32", "configs"])
def test_gemm_bf16xfp32_exact_lut_boundaries(change, monkeypatch, capfd):
    import ctypes
    import math
    previous_sms = deep_gemm.get_num_sms()
    monkeypatch.setenv("show_log", "1")
    monkeypatch.setenv("DG_JIT_DISABLE_SHORTCUT_CACHE", "0")
    m, n, k = 8, 384, 7168
    if change == "m": m += 1
    if change == "n": n += 2
    if change == "k": k += 64
    scale = math.nextafter(1 / 256, 1) if change == "near_scale" else (1 / 128 if change == "scale" else 1 / 256)
    try:
        requested_sms = 20 if change == "sms" else 64 if change == "sms64" else 39
        if requested_sms > torch.cuda.get_device_properties(0).multi_processor_count:
            # Reject budgets above the physical SM count before LUT selection.
            with pytest.raises(RuntimeError, match="new_num_sms"):
                deep_gemm.set_num_sms(requested_sms)
            return
        deep_gemm.set_num_sms(requested_sms)
        x, high, low, _, _ = _make_case(m, n, k)
        computation = x.float() @ high.float().t() + scale * (x.float() @ low.float().t())
        quality = x.float() @ (high.float() + scale * low.float()).t()
        actual = _call_api("explicit" if change == "configs" else "default",
                           x, high, low, scale, change == "fp32")
        _assert_accuracy(actual, computation, quality)
        torch.cuda.synchronize()
        ctypes.CDLL(None).fflush(None)
        expected_lut = change == "fp32" or (
            change in ("scale", "near_scale") and not int(os.environ.get("DG_JIT_USE_HGRTC", "0")))
        log = capfd.readouterr().out
        assert f"dual_b_exact_lut:{int(expected_lut)}," in log
        if not expected_lut:
            assert "dual_b_static_router:0, static_stages:0\n" in log
            assert "dual_b_decode_split_k:1\n" in log and "dual_b_pipeline:double_buffer\n" in log
    finally:
        deep_gemm.set_num_sms(previous_sms)


@pytest.mark.usefixtures("router_ppu1v5")
@pytest.mark.parametrize("mode", ["lut", "whitebox", "explicit"])
@pytest.mark.parametrize("use_fp32_output", [False, True])
def test_gemm_bf16xfp32_removed_overrides_are_ignored(mode, use_fp32_output, monkeypatch, capfd):
    import ctypes
    removed = (
        "PACKED_DUAL_N", "STATIC_ROUTER", "DECODE_SPLIT_K", "FULL_K_PREFETCH", "EARLY_G2S",
        "MMA_S2R_OVERLAP", "BOUNDED_KLOOP", "OVERLAP_PROLOGUE", "VALID_EPILOGUE", "STATIC_STAGES",
        "STREAMED_LOW", "STANDARD_S2", "STREAMED_PAIRS", "LAUNCH_BOUNDS2", "PAIRS_M_FIRST",
        "KLOOP_UNROLL", "KLOOP_CHUNK",
    )
    for name in removed:
        monkeypatch.delenv(f"DG_BF16XFP32_{name}", raising=False)
    monkeypatch.setenv("show_log", "1")
    previous_sms = deep_gemm.get_num_sms()
    api = "explicit" if mode == "explicit" else "default"

    def config_log():
        ctypes.CDLL(None).fflush(None)
        lines = tuple(line for line in capfd.readouterr().out.splitlines()
                      if line.startswith(("num_sms:", "ThreadblockShape[", "SMSIZE:", "dual_b_")))
        assert lines
        return lines

    try:
        deep_gemm.set_num_sms(20 if mode == "whitebox" else 39)
        x, high, low, scale, gt = _make_case(8, 384, 7168)
        baseline = _call_api(api, x, high, low, scale, use_fp32_output)
        _assert_accuracy(baseline, *gt)
        torch.cuda.synchronize()
        baseline_log = config_log()
        for value in ("1", "invalid"):
            for name in removed:
                monkeypatch.setenv(f"DG_BF16XFP32_{name}", value)
            actual = _call_api(api, x, high, low, scale, use_fp32_output)
            _assert_exact(actual, baseline)
            torch.cuda.synchronize()
            assert config_log() == baseline_log
    finally:
        deep_gemm.set_num_sms(previous_sms)


@pytest.mark.parametrize("api", ["default", "explicit", "wrapper", "native"])
@pytest.mark.parametrize("fp32", [False, True])
@pytest.mark.parametrize("operand", [0, 1, 2])
@pytest.mark.parametrize("empty", [False, True])
def test_gemm_bf16xfp32_rejects_cpu_operand(api, fp32, operand, empty):
    if not torch.cuda.is_available():
        pytest.skip("no PPU device available")
    x = torch.empty((0 if empty else 5, 13), dtype=torch.bfloat16, device="cuda")
    high = torch.empty((14, 13), dtype=torch.bfloat16, device="cuda")
    inputs = [x, high, high.clone()]
    inputs[operand] = inputs[operand].cpu()
    with pytest.raises(RuntimeError, match="BF16xFP32 inputs must be CUDA tensors"):
        _call_api(api, *inputs, 1 / 256, fp32)


@pytest.mark.parametrize("api", ["default", "explicit", "wrapper", "native"])
@pytest.mark.parametrize("fp32", [False, True])
@pytest.mark.parametrize("operand", [0, 1, 2])
def test_gemm_bf16xfp32_rejects_mixed_devices(api, fp32, operand):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible PPU devices")
    current = torch.cuda.current_device()
    other = (current + 1) % torch.cuda.device_count()
    x = torch.empty((5, 13), dtype=torch.bfloat16, device=current)
    high = torch.empty((14, 13), dtype=torch.bfloat16, device=current)
    inputs = [x, high, high.clone()]
    inputs[operand] = inputs[operand].to(other)
    with pytest.raises(RuntimeError, match="BF16xFP32 inputs must be on the same device"):
        _call_api(api, *inputs, 1 / 256, fp32)
    assert torch.cuda.current_device() == current


@pytest.mark.parametrize("api", ["default", "explicit", "wrapper", "native"])
@pytest.mark.parametrize("fp32", [False, True])
def test_gemm_bf16xfp32_rejects_noncurrent_device(api, fp32):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible PPU devices")
    current = torch.cuda.current_device()
    other = (current + 1) % torch.cuda.device_count()
    x = torch.empty((0, 13), dtype=torch.bfloat16, device=other)
    high = torch.empty((14, 13), dtype=torch.bfloat16, device=other)
    with pytest.raises(RuntimeError, match="BF16xFP32 inputs must be on the current device"):
        _call_api(api, x, high, high, 1 / 256, fp32)
    assert torch.cuda.current_device() == current


@pytest.mark.parametrize("scale", [1 / 256, 1 / 128, math.nextafter(1 / 256, 1)])
@pytest.mark.parametrize("fp32", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_gemm_bf16xfp32_replay_parser(scale, fp32, explicit):
    from utils import parse_deepgemm_string_re
    spec = f"Normal,data_type:bf16xfp32,m:8,n:384,k:7168,scale:{scale:.17g},use_fp32_output:{int(fp32)}"
    if explicit:
        spec += ",configs:[20,64,64,64,16,16,2]"
    parsed = parse_deepgemm_string_re(spec)
    assert parsed["scale"] == scale and parsed["use_fp32_output"] == fp32
    assert parsed["data_type"] == "bf16xfp32" and parsed["quant_type"] == "non_quantized"
    if explicit:
        assert parsed["configs"] == [20, 64, 64, 64, 16, 16, 2]
    assert parse_deepgemm_string_re("DenseGemm,data_type:bf16_bf16_fp32,m:16,n:64,k:128")["data_type"] == "bf16_bf16_fp32"


@pytest.mark.parametrize("suffix", ["", ",scale:0.00390625", ",use_fp32_output:1", ",scale:0,use_fp32_output:1",
                                    ",scale:nan,use_fp32_output:1", ",scale:inf,use_fp32_output:1",
                                    ",scale:0.00390625,use_fp32_output:2"])
def test_gemm_bf16xfp32_replay_requires_metadata(suffix):
    from utils import parse_deepgemm_string_re
    with pytest.raises(ValueError, match="BF16xFP32 replay"):
        parse_deepgemm_string_re("Normal,data_type:bf16xfp32,m:16,n:64,k:128" + suffix)


@pytest.mark.usefixtures("ppu1v5")
@pytest.mark.parametrize("shape", [(0, 14, 13), (5, 0, 13), (5, 14, 0), (5, 14, 13)])
@pytest.mark.parametrize("fp32", [False, True])
def test_gemm_bf16xfp32_replay_empty_and_tail(shape, fp32):
    m, n, k = shape
    actual = run_bf16xfp32_case(dict(m=m, n=n, k=k, scale=1 / 128, use_fp32_output=int(fp32)))
    assert actual.shape == (m, n)
    if not k:
        _assert_exact(actual, torch.zeros_like(actual))


@pytest.mark.usefixtures("ppu1v5")
@pytest.mark.parametrize("scale", [1 / 256, 1 / 128, math.nextafter(1 / 256, 1)])
@pytest.mark.parametrize("fp32", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_gemm_bf16xfp32_dump_replay(scale, fp32, explicit, tmp_path):
    from utils import parse_deepgemm_string_re
    spec = f"Normal,data_type:bf16xfp32,m:8,n:384,k:7168,scale:{scale:.17g},use_fp32_output:{int(fp32)}"
    if explicit:
        spec += ",configs:[20,64,64,64,16,16,2]"
    entry = Path(__file__).with_name("run_deep_gemm.py")
    env = dict(os.environ, PPU_LIB_SHOW_PARAMS="1")
    env.pop("HGGC_WARM_UP", None)
    env["DG_JIT_COMPILE_ONLY"] = "0"
    def run(command):
        result = subprocess.run([sys.executable, str(entry), *command], cwd=tmp_path, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        assert result.returncode == 0, result.stdout.decode(errors="replace")
        return result.stdout.decode(errors="replace")
    output = run(["--format", spec, "--require_supported"])
    records = re.findall(r"\[DeepGemm\] --format=((?:Normal|DenseGemm),data_type:bf16xfp32[^\n]+)", output)
    assert len(records) == 1, output
    parsed = parse_deepgemm_string_re(records[0])
    assert parsed["scale"] == scale and parsed["use_fp32_output"] == fp32
    if explicit:
        assert parsed["configs"] == [20, 64, 64, 64, 16, 16, 2]
    else:
        assert "configs" not in parsed
    assert "num_sms" in parsed
    caselist = tmp_path / "replay.list"
    caselist.write_text(records[0] + "\n")
    assert "Passed BF16xFP32 replay with acc_check" in run(["--caselist", str(caselist), "--require_supported"])


def test_gemm_bf16xfp32_automatic_registration():
    from checkin.run_checkin import SCRIPT_SUITES, load_cases
    from daily.run_daily import load_script_checks
    root = Path(__file__).parent
    for board, filename in (("ZW-M890P", "890p"), ("ZW-810E", "810e")):
        checkin = [spec for spec in SCRIPT_SUITES[board] if "test_gemm_bf16xfp32.py" in spec]
        daily = [spec for spec in load_script_checks(board) if "test_gemm_bf16xfp32.py" in spec]
        assert len(checkin) == len(daily) == 1
        assert checkin[0].split()[:3] == ["-P", "-m", "pytest"]
        assert daily[0].split()[:4] == ["python", "-P", "-m", "pytest"]
        cases = [case for case in load_cases(root / "checkin" / f"checkin_{filename}.list") if "data_type:bf16xfp32," in case]
        assert cases and any("use_fp32_output:0" in case for case in cases) and any("use_fp32_output:1" in case for case in cases)
        assert all(("expect_error:BF16xFP32 GEMM is disabled on PPU1.0" in case) == (filename == "810e") for case in cases)


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
