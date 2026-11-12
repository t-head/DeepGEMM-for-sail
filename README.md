# DeepGEMM

## Overview

This repository is a PPU-oriented fork of [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), a unified high-performance tensor cell kernel library for modern large language model workloads. It brings together low-precision GEMMs (FP8, FP4, BF16), fused MoE (with implicit permute) kernels, MQA scoring for the lightning indexer, HyperConnection (HC), and related primitives into a single cohesive codebase adapted for T-Head PPU hardware.

DeepGEMM for PPU focuses on the supported core computation primitives used by dense models, MoE models, and related LLM workloads. The fork updates the build and runtime path for the PPU SDK environment, replaces the original device-specific assumptions with PPU-oriented execution requirements, and extends the supported layouts and data types for PPU use cases.

Supported PPU platforms include Zhenwu 610 / 610E / 810 / 810E / M890. Runtime compilation is handled through the PPU toolchain, with HGCC / HGRTC support exposed through the JIT configuration options.

## Core Features and Optimizations

The PPU-oriented fork provides the following core capabilities and optimizations:

- **PPU runtime and JIT integration**: The runtime compilation path is adapted to the PPU SDK environment. Kernel compilation, compiler selection, debug output, and cache behavior are controlled through PPU-oriented JIT environment variables such as `DG_JIT_USE_HGRTC`.
- **Supported precision modes**: The PPU version supports INT8, FP4, FP8, and BF16 across non-grouped GEMM and grouped GEMM layouts, covering both dense GEMM and MoE grouped GEMM scenarios.
- **Standalone activation on M890**: SiLU and SiTU with BF16, per-row/per-block FP8, MXFP4, or MXFP8 output, including masked expert batches and explicit output-mode selection.
- **W4A8 MoE on M890**: E4M3 per-row activations and group-32 MXFP4 weights, with FP32 accumulation and BF16 output for nopad, masked, and fused routing layouts.
- **NVFP4 W4A8 on M890**: E4M3 activations and group-16 NVFP4 weights for dense, nopad, masked, and fused routing GEMMs, with fused decoding or per-call decode-to-FP8 execution.
- **A8W4 INT8 MoE**: INT8 per-token activations and portable packed signed-INT4 per-channel weights, with INT32 accumulation and BF16 output for nopad and masked grouped layouts on Zhenwu 610 / 610E / 810 / 810E (PPU001).
- **Additional nopad grouped GEMM layout**: In addition to contiguous and masked grouped GEMM layouts, this fork introduces a nopad layout. It uses an index tensor to describe per-expert token counts, avoiding mandatory M-block padding for variable-token MoE workloads and reducing unnecessary data movement.
- **AIU-oriented alignment requirements**: The PPU version replaces the original alignment assumptions with AIU-aligned scaling factors and AIU-aligned utility tensors, matching the data movement requirements described by this fork.
- **Warp interleave execution design**: The PPU kernels adopt a warp interleave design to overlap data movement, MMA instructions, and promotion operations during GEMM execution.
- **PPU-oriented scheduling and block-size choices**: The fork keeps the unified scheduler, rasterization strategy, fully JIT design, and unaligned block-size support, while enabling PPU-oriented larger block sizes up to `256x256`.

## Release Notes

### 1.1.0+v1.1.0

- **Breaking indexer API changes**: updates weighted and AVG entry points,
  dtype and output contracts, and paged `metadata_extra`. Downstream callers
  must follow the [indexer migration guide](docs/api/indexer.md).
- **Paged metadata correctness fix**: fixes the shared-memory prefix-scan race
  in both main and tail scans without increasing the barrier count.

See [CHANGELOG.md](CHANGELOG.md#110v110) for complete release notes.

### 1.1.0+v1.0.0

Highlights:

- **W4A4 MegaMoE** on PPU1.5: a complete MXFP4 routed MoE pipeline with BF16 output, reusable workspaces, and load-time weight preparation. See the [calling contract](#mega-moe).
- **More low-precision MoE kernels**: W4A8 on M890 and INT8 A8W4 on Zhenwu 610 / 610E / 810 / 810E, with unified weight preprocessing.
- **Activation and quantization** on M890: standalone SiLU, SiTU, and SiLU STEP with BF16, FP8, MXFP4, or MXFP8 output, plus STEP activation in fused FP4 epilogues.
- **Mega-mHC and sparse MQA** on M890: fused shifted mHC for DeepSeek V4.1 Flash and MXFP4 sparse attention-indexer logits for prefill and paged decode. Regular and AVG paged MQA on M890 support page size 32 or any positive multiple of 64, with no fixed 128-token cap.
- **External GEMM tuning tables**: device- and precision-specific configurations through `SAIL_DEEPGEMM_LUT_PATH`, with fallback to built-in selection.

See [CHANGELOG.md](CHANGELOG.md#110v100) for complete release notes and API migration details.

### 1.1.0+v0.1.0 — Initial Public Release

This is the first formal public release of DeepGEMM for PPU. The `1.1.0+v0.1.0` identifier follows the current DeepGEMM-for-sail release naming convention and is maintained independently of the upstream DeepGEMM project.

Highlights:

- Native PPU runtime and JIT compilation through HGCC and HGRTC.
- BF16, INT8, FP8, and MXFP4 GEMM kernels across supported dense and grouped layouts, with W4A16 support for no-pad, masked, and fused MoE workloads.
- Fused MoE with implicit permutation, MQA and paged MQA scoring, a TF32 HyperConnection prenorm GEMM, and FP8/INT8 einsum.
- Support for Zhenwu 610 / 610E / 810 / 810E / M890 platforms.

See [CHANGELOG.md](CHANGELOG.md) for the complete release notes.

### Package versioning

- Release: `1.1.0+v1.1.0.ppu2.2.0` combines the placeholder version `1.1.0`,
  the release version `v1.1.0`, and the PPU SDK version `2.2.0`.
  In [setup.py](setup.py), these first two components correspond to
  `DEEPGEMM_VERSION` and `RELEASE_VERSION`, respectively.
- `1.1.0` is expected to remain unchanged. Future releases and API changes
  are tracked by the `v1.1.0` component in `RELEASE_VERSION`.
  The middle number in `v1.1.0` is incremented for API changes or additions
  when they are formally integrated into downstream frameworks, rather than
  for each individual API change or addition.
- Development: `1.1.0.devN+v1.1.0.ppu2.2.0.g<sha>` additionally includes
  the commit count and Git commit hash.

## Quick Start

### Requirements

- Zhenwu 610 / 610E / 810 / 810E / M890
- Python 3.8 or above
- PPU SDK 2.2 or above
- PyTorch 2.1 or above
- ACTLIZE for PPU: v1.0.0

### Development

Before contributing, read the [DeepGEMM development guidelines](docs/development.md).

```bash
# Submodule must be cloned recursively
git clone --recursive git@github.com:t-head/DeepGEMM-for-sail.git
python -m pip install setuptools setuptools-scm
# Make symbolic links for third-party (ACTLIZE) include directories and build
bash develop.sh

# Test all GEMM implements (normal, contiguous-grouped and masked-grouped)
python tests/test_core.py
```

### Installation

```bash
python -m pip install setuptools setuptools-scm
bash install.sh
```

Then, import `deep_gemm` in your Python project, and enjoy!

## Interfaces

#### Notices

This library focuses on high-performance tensor cell kernels and related LLM computation primitives. GEMM-family interfaces only support the NT format (non-transposed LHS and transposed RHS), and expect contiguous inputs with scaling factors in the layout required by the corresponding quantization recipe (for example, the blockwise FP8 recipe expects a transposed LHS scaling factor). If the layout does not match, the interfaces will fix it up internally with a set of slow PyTorch operations. For transposition or other quantization casting operations, please implement or fuse them into prior kernels independently. While the library provides some simple PyTorch utility functions, these may result in slower performance, but our primary focus is on optimizing the core kernels themselves.

#### Data Precisions

| Algorithm | INT8 | FP4 | FP8 | BF16 | W4A16 | MXFP4 W4A8 (M890) | NVFP4 W4A8 (M890) | A8W4 INT8 (PPU001) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Non-grouped** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| **Contiguous** | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **No-pad** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Masked** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Fused** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

#### Normal dense GEMMs (non-grouped)

To perform a basic non-grouped FP8 GEMM, call the `deep_gemm.gemm_fp8_fp8_bf16_nt` function. For more details, please refer to the function documentation.

BF16 dense GEMM: `deep_gemm.bf16_gemm_nt(a, b, out)` computes `out = a @ b.T`
with contiguous BF16 inputs. On 810E and 890P, `out` supports BF16 or FP32.
890P supports odd and even N. On 810E, N must be even except for N=1
without an explicit configuration, which requires K divisible by 8 and
16-byte-aligned inputs.

#### Grouped GEMMs (contiguous layout)

Unlike traditional grouped GEMMs in ACTLIZE, DeepGEMM groups only the M-axis, while N and K must remain fixed. This design is tailored for scenarios where experts in an MoE model share the same shape.

For training forward passes or inference prefilling, where each expert may process a varying number of tokens, we concatenate these tokens into a single tensor, referred to as the "contiguous" layout. Note that each expert segment must be aligned to the GEMM M block size (`get_m_alignment_for_contiguous_layout()`).

For more information, please refer to the `m_grouped_gemm_fp8_fp8_bf16_nt_contiguous` function documentation.

#### Grouped GEMMs (nopad layout)

We provide a new Grouped GEMM mode, namely the nopad layout. When the number of tokens processed by each expert varies, by providing an index tensor that stores the token count for different experts, there is no need to align to the GEMM M block size, reducing unnecessary data movement.

For more information, please refer to the `m_grouped_gemm_fp8_fp8_bf16_nt_nopad` function documentation.

#### Grouped GEMMs (masked layout)

During the inference decoding phase, when HGGC Graph is enabled and the CPU is unaware of the number of tokens each expert receives, we support masked grouped GEMMs. By providing a mask tensor, the kernel computes only the valid portions.

Use `m_grouped_gemm_fp8_fp8_bf16_nt_masked` for this purpose and consult the relevant documentation. An example usage is to use the output of low-latency kernels from [DeepEP](https://github.com/deepseek-ai/DeepEP) as input.

For the FP4 masked GEMM, the `silu_and_mul` activation and the MXFP4 quantization that follows it can optionally be fused into the epilogue, so that MoE gemm1 writes the packed FP4 values and their E8M0 scales directly instead of a BF16 tensor that a standalone kernel has to read back. The fusion is **disabled by default**; pass an `out_scale` tensor to `deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked` to enable it, and interleave the gemm1 weight once with `deep_gemm.preprocess_weights_and_scales(mode=deep_gemm.PreprocessMode.MXFP4, fuse_activation=True)` beforehand. Refer to the `m_grouped_gemm_fp4_fp4_bf16_nt_masked` documentation for the tensor layouts, the `swiglu_limit` argument and the current limitations. The fused epilogue is also available on `m_grouped_gemm_fp4_fp4_bf16_nt_nopad` and `m_grouped_gemm_fp4_fp4_bf16_nt_fused`; use `activation_type="silu_and_mul_step"` for STEP-series models, which clamp `silu(gate)` from above instead of clamping the gate before silu.

#### K-grouped GEMMs (contiguous layout)

The grouped GEMMs above group tokens along the M axis while N and K stay fixed. MoE training backward weight gradients instead reduce over each expert's own token count, so the groups vary along the K axis: every group's `a_g [K_g, M]` and `b_g [K_g, N]` slices are concatenated into `a [sumK, M]` and `b [sumK, N]`, and the kernel computes `d[g] = c[g] + a_g^T @ b_g` per group.

`deep_gemm.k_grouped_bf16_gemm_tn_contiguous` (M890) implements the BF16 wgrad path with FP32 output. The LHS is the physically contiguous `[sumK, M]` tensor (typically a `.t()` view that is itself contiguous), and `d` is the `[G, M, N]` FP32 `main_grad` accumulator. `c` is required and normally the same tensor as `d`, giving in-place beta=1 accumulation for consecutive micro-batches; a separate FP32 `c` is copied into `d` first. Each group's length must be a non-negative multiple of 128 (per-group physical K alignment, with padded rows zeroed), passed as a host-side `ks_cpu` list together with the equivalent int32 `grouped_layout` tensor on device. Groups with `K_g = 0` keep the existing output, and zero total K returns without computation. `use_psum_layout` must be `False`, custom epilogues are not supported, and `compiled_dims` only accepts `"mn"`. 

For more information, please refer to the `k_grouped_bf16_gemm_tn_contiguous` function documentation.

#### Fused MoE (with implicit permute)

Unlike the standalone grouped GEMM layouts above (which expect tokens to already be routed and grouped per expert), the fused MoE path takes raw MoE routing results (`topk_ids`) and fuses the token permutation, per-expert block alignment, and grouped GEMM into a single flow. The "with implicit permute" naming distinguishes it from a plain grouped MoE GEMM: the permutation from token order to expert-grouped order is computed and applied inside these kernels, so no separate gather/scatter step is required.

First call `deep_gemm.moe_align_block_size` with the LHS/RHS inputs and `topk_ids` to compute the block-aligned scheduling metadata (`m_rows`, `expert_ids_and_cumsum`, `sorted_token_ids`, `aligned_num_m_blocks`) together with the inverse permutation (`inv_perm`), the per-token expert indices (`m_indices`), and the auto-selected GEMM config. Then feed this metadata into one of the fused grouped GEMM kernels according to the input precision:

- `deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_fused`: fused MoE grouped GEMM with BF16 LHS/RHS and BF16 output.
- `deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_fused`: fused MoE grouped GEMM with FP8 LHS/RHS (with scales) and BF16 output; supports both per-channel and blockwise quantization.
- `deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_fused`: fused MoE grouped GEMM with INT8 LHS/RHS (with per-channel scales) and BF16 output.
- `deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_fused`: fused MoE grouped GEMM with FP4 (mxfp4) LHS/RHS (with scales) and BF16 output. Note that this path expects the **LHS scale (SFA) in uint16 K-major layout**, that is a plain contiguous `(num_token, ceil_div(k // 32, 2))` uint16 tensor, rather than the uint16 M-major layout produced by `preprocess_mxfp4_scales` that the masked/nopad/dense FP4 interfaces expect; the RHS scale (SFB) stays MN-major. To build it, pad the `ceil_div(k, 32)` uint8 E8M0 scales to an even count along K and then view them as uint16; passing a `preprocess_mxfp4_scales` output as SFA here will fail the contiguity assertion. When `out_scale` is provided to `m_grouped_gemm_fp4_fp4_bf16_nt_fused`, also pass `enable_act_and_quant_fusing=True` to `moe_align_block_size`, so that its auto-selected config satisfies the fused epilogue requirements.
- `deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused`: fused routing gather with per-row E4M3 activations, group-32 MXFP4 weights, and BF16 output on M890.
- `deep_gemm.m_grouped_gemm_w4a16_fused`: fused MoE grouped GEMM with 4-bit group-quantized weights and BF16 activation (W4A16), producing BF16 output.

For more information, please refer to the `moe_align_block_size` and `m_grouped_gemm_*_nt_fused` function documentation.

#### W4A8 dense and MoE GEMMs (M890)

The W4A8 implementation accepts per-row E4M3 activations and group-32
MXFP4 E2M1 weights.
The kernel dequantizes weights to E5M2 using FP4 MMA, performs E4M3 x E5M2
MMA, accumulates in FP32, and writes BF16. It supports these layouts:

| API | Activation / row-scale shape | Output shape |
| :--- | :--- | :--- |
| `fp8_fp4_gemm_nt` (alias `gemm_fp8_fp4_bf16_nt`) | `[M,K]` / `[M,1]`, dense | `[M,N]` |
| `m_grouped_gemm_fp8_fp4_bf16_nt_nopad` | `[M,K]` / `[M,1]`, grouped by expert | `[M,N]` |
| `m_grouped_gemm_fp8_fp4_bf16_nt_masked` | `[E,T,K]` / `[E,T,1]` | `[E,T,N]`, valid rows only |
| `m_grouped_gemm_fp8_fp4_bf16_nt_fused` | `[num_token,K]` / `[num_token,1]`, original token order | `[num_token*topk,N]`, grouped by expert |

Activation scales are FP32. Dense weights are naturally packed uint8 `[N,K/2]`
with raw uint8 E8M0 scales `[N,K/32]`. Grouped weights use `[E,N,K/2]`
(even K elements in the low nibble), with raw uint8 E8M0 scales
`[E,N,K/32]`. `N % 64 == 0` and `K % 128 == 0` are required; arbitrary
N/K padding is not performed. GEMM operands and metadata must be contiguous
on the same PPU device; routing/count metadata uses int32.

Weight preprocessing is performed once during model initialization. The
function returns a tuple containing the processed weights, rearranged
scales, and a scalar `weight_inv_scale`.
It applies a common power-of-two shift to bring weight-scale exponents into
`[114,140]` when necessary. Exponent codes 0 and 255, and exponent ranges
spanning more than 26, are rejected. The returned inverse must be applied
exactly once per GEMM. Either fold it into the activation row scales or pass
it as the optional scalar `inv_scale` to any W4FA8 GEMM API.
`inv_scale` must be finite and positive; its default value of `1.0` skips
the extra multiplication. The E4M3 codes remain unchanged:

```python
import deep_gemm as dg

# packed_weight and raw_weight_scale are checkpoint tensors described above.
_, weight_scale, weight_inv_scale = dg.preprocess_weights_and_scales(
    dg.PreprocessMode.W4A8_MXFP4_MMA, False, weight_scale=raw_weight_scale)
lhs = (a_e4m3, row_scale)
rhs = (packed_weight, weight_scale)
config, rows, experts, tokens, aligned, inv_perm, m_indices = dg.moe_align_block_size(
    a_e4m3, packed_weight, topk_ids)
dg.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
    lhs, rhs, out, rows, experts, tokens, aligned, config,
    inv_scale=weight_inv_scale)
```

For dense NT GEMM, preprocessing preserves the two-dimensional weight layout:

```python
rhs = dg.preprocess_weights_and_scales(
    dg.PreprocessMode.W4A8_MXFP4_MMA, False, packed_weight, raw_weight_scale)
dg.fp8_fp4_gemm_nt((a_e4m3, row_scale), rhs[:2], out, inv_scale=rhs[2])
```

The fused path gathers activations and their scales with the same token
index. Its output already uses the grouped, unpadded row order consumed by
nopad. The fused interface integrates routing gather with GEMM execution. Weight
preprocessing uses `fuse_activation=False` to preserve concatenated
gate/up rows. Activation
and top-k combine remain separate operations.

Nopad accepts per-row expert `m_indices` and optional per-expert `m_rows`;
rows must already be grouped by expert. Masked accepts device-side
`masked_m` counts and an `expected_m` scheduling hint. Computation is
restricted to the valid rows specified by `masked_m`. Dense/nopad/masked select configurations internally unless
an explicit config tuple is supplied. The fused config comes from
`moe_align_block_size`. The existing nine-element config format is unchanged:
`(num_sms, block_m, block_n, block_k, warp_m, warp_n, warp_k, stages, n_expand)`.
Native tile selection and streamed execution policies are handled internally
by the C++ runtime. For explicit streamed Dense/NoPad/Fused execution, use
`(dg.get_num_sms(), 256, 256, 128, 64, 64, 128, 4, 1)`.
Streamed execution supports one or more experts, including dense GEMM, on PPU1.5.
Subject to device, layout and workspace limits, automatic streamed execution
requires at least 512 rows per expert on average when `K >= 2048` and
`ceil(N / 256) * (K / 128) >= 128`, or 1024 rows otherwise.
It also requires enough M/N tiles in an expert chunk to fill one device wave;
smaller workloads retain native execution. Each decoded-weight buffer is bounded
by 384 MiB; a single-part plan allocates one buffer, and multi-part plans use two.
When a single expert requires N slices to fit that budget, automatic streamed
execution requires at least 1024 rows to amortize the additional decoding launches.
Set `show_log=1` to print the selected tile, resources and streamed execution plan.
Before graph capture, warm up the selected configuration and unit/non-unit
`inv_scale` variant on the capture stream.
Run `python tests/test_w4a8.py` for correctness coverage.

For standard MXFP4 GEMM weights, use
`preprocess_weights_and_scales(PreprocessMode.MXFP4, False, weight, weight_scale)`.
This preserves packed weights and prepares uint16 M/N-major scales; setting
`fuse_activation=True` additionally interleaves gate/up weights and scales.
The legacy `preprocess_mxfp4_scales(scale)` and
`preprocess_mxfp4_weight_for_act_and_quant_fusing(weight, weight_scale)`
remain available as deprecated compatibility aliases. The scale-only alias
also supports activation and reference scale conversion. Fused GEMM activation
input scales retain their separate K-major layout requirement.
For `MXFP4` and `W4A8_MXFP4_MMA` with `fuse_activation=False`,
`weight` defaults to `None`. Pass `weight_scale=...` to prepare scales alone;
the first result is then `None`. W4A8 also returns `weight_inv_scale` and derives
E/N/K alignment from the raw `[E,N,K/32]` scales, or `[N,K/32]` for dense GEMM. Supplying a weight tensor
preserves the existing return format; W4A8 also checks its shape against the scales. Fusion and other
weight modes still require weights; all supported modes require scales.

**W4A16 migration from 2.0.0:**
`m_grouped_gemm_w4a16_{nopad,masked,fused}` no longer accept
`fp4_use_bf16_scale`, including an explicit `False`. The MXFP4 weight path
with BF16 scales has also been removed. For MXFP4 weights with E8M0 scales,
use `PreprocessMode.W4A16_MXFP4` or `W4A16_MXFP4_MMA`; for INT4 weights
with BF16 scales, use `PreprocessMode.W4A16_INT4`. These modes require
`fuse_activation=False`. Existing MXFP4-with-BF16-scale data must be
converted to a supported quantization format and layout before use;
removing the keyword alone is insufficient. See the
[preprocessing implementation](csrc/jit_kernels/heuristics/common_preprocess.hpp)
for the required weight and scale layouts.

#### NVFP4 W4A8 GEMMs (M890)

`gemm_fp8_nvfp4_bf16_nt` and
`m_grouped_gemm_fp8_nvfp4_bf16_nt_{nopad,masked,fused}` accept E4M3
activations and NVFP4 weights with native K16 E4M3 scales, and write BF16.
They support optional FP32 global and per-channel or 128x128 outer weight scales.

For every layout, callers explicitly select `execution='fused'` (the default,
decode weights inside GEMM) or `execution='dequantize'` (decode each call into
a caller-owned FP8 workspace, then run FP8 GEMM). The `_fused` API suffix
selects routing gather independently of this choice.

See the [NVFP4 W4A8 API guide](docs/api/nvfp4_w4a8.md) for operand layouts,
scale normalization, preprocessing, execution constraints and calling examples,
and the [W4A8 test guide](tests/README.MD#w4a8-correctness-and-performance-tests)
for accuracy and performance reproduction.

#### A8W4 INT8 MoE GEMMs

The A8W4 INT8 path accepts INT8 activations with FP32 per-token scales and portable packed signed-INT4 weights with FP32 per-channel scales, and writes BF16. It runs on Zhenwu 610 / 610E / 810 / 810E (PPU001) and supports the grouped nopad and masked expert layouts:

| API | Activation / row-scale shape | Weight / weight-scale shape | Output shape |
| :--- | :--- | :--- | :--- |
| `m_grouped_gemm_int8_int4_bf16_nt_nopad` | `[M,K]` / `[M,1]`, grouped by expert | `[E,N,K/2]` / `[E,N,1]` | `[M,N]` |
| `m_grouped_gemm_int8_int4_bf16_nt_masked` | `[E,T,K]` / `[E,T,1]` | `[E,N,K/2]` / `[E,N,1]` | `[E,T,N]`, valid rows only |

The kernel converts the packed INT4 weights to INT8 in-register, runs INT8 x INT8 MMA with INT32 accumulation, and scales each result by the per-token activation scale and the per-channel weight scale (both FP32) before writing BF16. Activations are `torch.int8`; weights use portable packed signed-INT4 `torch.uint8` storage `[E,N,K/2]` (two INT4 values per byte); all scales are FP32. `K % 32 == 0` and `N % 8 == 0` are required, and all operands and metadata must be contiguous on the same PPU device.

Weight preprocessing runs once during model initialization. The checkpoint stores the portable packed signed-INT4 weight; call `preprocess_a8w4_weight` after loading to apply the device-specific permutation before passing it as the GEMM `b` operand. The FP32 per-channel weight scales `[E,N,1]` are used as-is and are not preprocessed:

```python
import deep_gemm as dg

# packed_int4_weight is the checkpoint tensor [E, N, K/2] (torch.uint8).
weight = dg.preprocess_a8w4_weight(packed_int4_weight)
lhs = (a_int8, a_scale)       # masked: [E,T,K] int8 / [E,T,1] fp32
rhs = (weight, weight_scale)  # [E,N,K/2] uint8 / [E,N,1] fp32
# Masked returns (block_m, ceil_div(n, block_n)) and computes only the masked_m valid rows.
block_m, n_blocks = dg.m_grouped_gemm_int8_int4_bf16_nt_masked(
    lhs, rhs, out, masked_m, expected_m)
```

Masked/nopad follow the standard grouped GEMM argument semantics described above (`masked_m`/`expected_m`, `m_indices`/`m_rows`, internal `configs` selection); masked additionally supports optional SBO overlap (`enable_sbo_overlap` + `signal`). For correctness coverage, run `tests/run_deep_gemm.py` with `data_type:w4a8_int8`, or the `w4a8_int8` entries in `tests/deepgemm.caselist`.

#### Standalone SiLU / SiTU activation (M890)

The interfaces are `activation_and_mul(activation_mode, quant_mode, input, *, ...)`
and `activation_and_mul_masked(activation_mode, quant_mode, input, masked_m, *, ...)`.
Both mode arguments are required: `ActivationMode` selects `SILU`, `SITU`, or
`SILU_STEP`; `QuantMode` selects the output type. The old `silu_and_mul`,
`situ_and_mul`, and `ActivationQuantMode` bindings have been removed.
Input is contiguous BF16 `[...,2*N]` in concatenated `[gate,up]` order.

| Mode | Output | Scales / constraints |
| :--- | :--- | :--- |
| `BF16` | BF16 tensor | No scale; any positive N |
| `FP8_CHANNEL` | E4M3 tensor | FP32 per-row `[...,1]` |
| `FP8_BLOCK` | E4M3 tensor | FP32 group-128 `[...,N/128]`; N%128=0 |
| `MXFP4` | Packed E2M1 uint8 `[...,N/2]` | Group-32 E8M0; even N |
| `MXFP8` | E4M3 tensor | Natural uint8 group-32 E8M0 `[...,ceil(N/32)]` |

BF16 mode returns one tensor; all quantized modes return `(values, scales)`.
MXFP4 defaults to `reorder_scale=True`, producing uint16 M-major scales
with logical `[M,ceil(N/64)]` or `[E,T,ceil(N/64)]` shape; this view may be
non-contiguous and should be passed directly to the matching GEMM.
The default reordered MXFP4 mode requires rank-2 or rank-3 input.
`reorder_scale=False` instead returns natural uint8
`[...,ceil(N/32)]` scales. This option is only accepted for MXFP4.
FP8-block dense scales are contiguous; masked scales use an M-major
`[E,T,N/128]` view backed by `[E,N/128,align(T,4)]` storage, with invalid
and padding scale rows zeroed.

```python
import deep_gemm as dg

A, Q = dg.ActivationMode, dg.QuantMode
bf16 = dg.activation_and_mul(A.SILU, Q.BF16, gate_up)
fp8, block_scale = dg.activation_and_mul(
    A.SILU_STEP, Q.FP8_BLOCK, gate_up, swiglu_limit=2.1, route_ids=route_ids)
masked_bf16 = dg.activation_and_mul_masked(
    A.SILU_STEP, Q.BF16, expert_gate_up, masked_m, expected_m=expected_m)
# SiTU with gate/up softcaps; down_inv is the GEMM2 weight_inv_scale.
q, row_scale = dg.activation_and_mul(
    A.SITU, Q.FP8_CHANNEL, gate_up, beta=4.0, linear_beta=25.0,
    weight_inv_scale=down_inv)
# row_scale already includes down_inv and is passed directly to W4A8 GEMM2.
```

For masked inputs use BF16 `[E,T,2*N]` and contiguous device int32
`masked_m` of shape `[E]` through `activation_and_mul_masked`, with an
optional `expected_m` hint. The hint only chooses the number of row slots;
the device loops over all valid rows, including when the hint is smaller
than the actual count. Output values are written only in valid rows.

The dense entry point accepts optional contiguous device int32 `route_ids`
with one entry per input row (any shape). Negative IDs skip the row; skipped
values/scales are unspecified. IDs are checked on device on every call or graph replay.

SiLU computes `gate * sigmoid(gate) * up`. Optional positive
`swiglu_limit` clamps gate from above and up on both sides. Clamp boundaries
round to BF16. `SILU_STEP` instead clamps `SiLU(gate)` from above before
multiplying by the clamped up. None or zero disables clipping in either mode.
Activation, multiplication, amax, scale computation and
normalization use FP32 with native PPU sigmoid. SiTU uses
`beta*tanh(gate/beta)*sigmoid(gate)` for the gate and optionally
`linear_beta*tanh(up/linear_beta)` for up; beta and a supplied linear_beta
must be positive finite values.

`weight_inv_scale` is only accepted for FP8-channel and multiplies the
returned scale, leaving the quantized codes unchanged. The kernels process
eight consecutive elements per thread and use native packed conversions
and vector stores. Fallback paths support partial tiles and unaligned input. MXFP4/MXFP8
share a single-warp double-buffered load pipeline; FP8-channel retains a
whole-row amax reduction. Run `python tests/test_activation.py` for the
standalone correctness suite.

#### DSA and AVG indexer logits

Non-paged and paged DSA indexers support legacy INT8/FP8 and MXFP4 inputs.
Legacy FP8 uses FP32 weights/logits;
INT8 accepts BF16 or FP32 weights and returns the same logits dtype. New
`fp8_fp4_*mqa_logits` weighted entry points support MXFP4 only, with BF16
weights/logits. Weighted paged calls require `next_n=1` and support page
size 32 or any positive multiple of 64 on M890, including 256/512/1024.
The same page-size rule applies to M890 AVG. 810E rejects page32 due to
an actlize TSM addressing issue; use a positive multiple of 64.

AVG indexers use no weights, default to BF16 output and retain paged
`next_n=1–6`. Non-paged AVG requires a positive `max_seqlen_k`. The paged
`fp8_fp4` AVG entry point accepts MXFP4 only; ordinary FP8 uses the separate
FP8 entry point. All AVG entry points omit `clean_logits`.

Logits outside valid windows are unspecified and require caller-side masking
or valid candidate bounds for top-k. For paged scheduling on both PPU0010/PPU0015, callers must pass
`metadata_extra=(N,H,D,qk_dtype)` using logical D. On M890, `torch.int8`
means MXFP4; on 810E it means INT8. True INT8 indexer APIs are 810E-only
and reject calls on M890. For non-paged
calls, positive `max_seqlen_k` selects compressed relative columns and can
reduce memory usage.

See the [indexer API guide](docs/api/indexer.md) for all signatures, dtype
and layout contracts, metadata generation, top-k coordinate mapping and
recommended 810E/M890 integration settings.

#### Sparse MQA logits (Zhenwu M890)

Sparse MQA computes indexer logits only for selected KV blocks, for both
prefill and paged decode. The current implementation supports **MXFP4
only**, despite the `fp8_fp4` API prefix, with **32 heads** and
**head dimension 128**. Build metadata with the matching helper before
calling the logits kernel:

| Layout | Metadata API | Logits API | Packed Q / Q scale shapes |
| :--- | :--- | :--- | :--- |
| Non-paged | `get_sparse_mqa_logits_metadata` | `fp8_fp4_sparse_mqa_logits` | `[T,32,64]` / `[T,32]` |
| Paged | `get_paged_sparse_mqa_logits_metadata` | `fp8_fp4_paged_sparse_mqa_logits` | `[T,1,32,64]` / `[T,1,32]` |

Here `T` is the number of query tokens. Q stores packed MXFP4 codes in
contiguous int8 tensors, and Q scales pack the E8M0 bytes into contiguous
int32 tensors. Paged decode requires `next_n=1`. Per-token head weights
are contiguous BF16 `[T,32]`.

`sparse_block_kv` is **8 or 16**, and `num_max_sparse_blocks` must be a
positive multiple of 4, at most 4096. The selected KV block indices are
contiguous int32 `[T,num_max_sparse_blocks]`, with a strictly increasing
valid prefix in each row. Paged sparse MQA requires a positive `page_kv`
that is a multiple of `sparse_block_kv`; its fused uint8 KV cache has shape
`[num_pages,page_kv,1,68]` and a 512-byte-aligned page stride.

The output is BF16 with shape
`[T,num_max_sparse_blocks*sparse_block_kv]`, with an aligned physical row stride.
Logical columns are compacted by selected block slot:
`column = slot*sparse_block_kv + token_in_block`, rather than absolute KV
token position. Allocation padding is not exposed as logical columns.
See the [indexer API guide](docs/api/indexer.md) for metadata and layout
requirements and [attention tests](tests/test_attention.py) for supported
cases.

#### Mega Gate

`deep_gemm.bf16_mega_gate` fuses the BF16 routing GEMM and MoE routing.
Given BF16 tokens `x [T, H]` and expert weights `weight [E, H]`, it
computes `x @ weight.T` with FP32 accumulation, applies `sqrt(softplus)`, and
selects the top-K experts using the scores plus optional bias. The selected
unbiased scores are normalized and scaled to produce routing weights; expert
indices can optionally be mapped from logical to physical IDs.

The outputs are INT64 indices and FP32 weights, each shaped `[T, K]`. When
shared experts are included as routed experts, each output is `[T, K + S]`,
where `S` is the number of shared experts. See the [API definition](csrc/apis/mega_gate.hpp)
for arguments and constraints and [the tests](tests/test_mega_gate.py) for examples.

#### Mega MoE

MegaMoE combines expert-parallel dispatch, two expert GEMMs, gated activation
and reduction on PPU. Allocate the symmetric buffer and transform the local
weights once, then run pre-dispatch and forward for each batch. The following
example uses MXFP4 (W4A4). Both `mma_type="fp4xfp4"` and
`mma_type="mxfp4xmxfp4"` select this backend. Workspaces store the canonical
name `fp4xfp4`; either spelling can be used for pre-dispatch, including with
a workspace created using the other spelling.

```python
import torch
import deep_gemm

buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,
    mma_type="fp4xfp4")
l1_weights, l2_weights = deep_gemm.transform_weights_for_mega_moe(
    (w13_fp4, w13_scales), (w2_fp4, w2_scales))

# Pass BF16 inputs, int32 or int64 routing IDs, and float32 routing weights.
output = torch.empty_like(hidden_states)
deep_gemm.mega_moe_pre_dispatch(
    hidden_states, topk_ids, topk_weights,
    buffer.x, buffer.x_sf, buffer.topk_idx, buffer.topk_weights,
    num_tokens=hidden_states.shape[0], group_size=32, mma_type="fp4xfp4")
deep_gemm.fp4_fp4_mega_moe(output, l1_weights, l2_weights, buffer)
```

For UltraEP, pass `l1_weights_replicas` and `l2_weights_replicas` together.

See [test_mega_moe.py](tests/test_mega_moe.py) for a complete example.

##### FP8, W4A8 and W4A16 MegaMoE

These backends follow the same complete-call and workspace ownership contract.
`intermediate_hidden` always means the activated width I. Routing weights are
supplied once through `topk_weights` and applied once in the final BF16 combine.
Shared experts remain outside the routed pipeline; `num_shared_experts` must be 0.

All precisions use `SymmBuffer` or `get_symm_buffer_for_mega_moe` for workspace
allocation. Select the mode below; FP8 additionally distinguishes scale granularity
with `quantization` (channelwise by default). These backends require PPU1.5.

| Backend | Compute | Workspace mode | Activation | EP limit |
|---|---|---|---|---|
| FP8 channelwise | `fp8_fp8_mega_moe` | `mma_type="fp8xfp8", quantization="channelwise"` | SwiGLU, SwiGLU-OAI | 64 |
| FP8 blockwise | `fp8_fp8_mega_moe` | `mma_type="fp8xfp8", quantization="blockwise"` | SwiGLU, SwiGLU-OAI | 64 |
| E4M3 x MXFP4 | `fp8_fp4_mega_moe` | `mma_type="fp8xfp4"` | SwiGLU, SwiGLU-OAI, SiTU | 64 |
| BF16 x MXFP4 | `bf16_fp4_mega_moe` | `mma_type="bf16xfp4", activation="situ"` | SiTU | 64 |

H is divisible by 256 and I by 128. FP8 and W4A8 support at most 128 physical experts
per rank; W4A16 supports at most 127. Capacity is positive and below 2^24;
top-k is 1–16 and no larger than the expert count. `num_experts` counts global
physical expert slots, including replicas, and is divisible by EP. Valid expert
IDs are in `[0, num_experts)` and distinct within each token; use `-1` for masked
slots, which may repeat. All ranks, including empty ranks, call in the same
order. Reuse each workspace serially on one stream; graph
replay may update inputs and routes in place without changing shapes. The
activation is selected by each complete compute call, and the same workspace
can be reused across activations supported by its precision. The library
recycles dispatch counts and applies routing weights exactly once,
skipping zero-weight or invalid slots before reading combine storage. All routed
backends share the 64-rank peer-address capacity of `SymmBuffer`; see the tests
for model-specific EP coverage.
Separate master/replica pairs use the same keyword arguments and physical slot
ordering as W4A4. Transform each bank separately before use.

The examples below are independent: initialize `group` and select the local
device first. Each uses contiguous BF16 `x` of shape `[T,H]`, routing IDs and
FP32 weights of shape `[T,topk]`, and a BF16 output `y` of shape `[T,H]`.
Pass exactly T input rows to pre-dispatch and use the same T for compute.
Each raw L1/L2 argument is a `(weight, scale)` pair in the layouts below.

```python
import torch
import deep_gemm as dg

# Raw FP8 weights: [E,2I,H] and [E,H,I], with the scales described below.
l1, l2 = dg.transform_weights_for_fp8_mega_moe(raw_l1, raw_l2, quantization="channelwise")
workspace = dg.get_symm_buffer_for_mega_moe(
    group, num_experts=256, num_max_tokens_per_rank=4096,
    num_topk=8, hidden=6144, intermediate_hidden=2048,
    mma_type="fp8xfp8", quantization="channelwise", activation="swiglu")
y = torch.empty_like(x)
dg.mega_moe_pre_dispatch(
    x, topk_ids, topk_weights,
    workspace.x, workspace.x_sf, workspace.topk_idx, workspace.topk_weights,
    num_tokens=x.shape[0], mma_type=workspace.mma_type)
dg.fp8_fp8_mega_moe(y, l1, l2, workspace, activation="swiglu")
```

```python
# Raw W4A16 weights are packed MXFP4 with raw group-32 UE8M0 scales.
l1, l2 = dg.transform_weights_for_w4a16_mega_moe(raw_l1_w4, raw_l2_w4)
workspace = dg.get_symm_buffer_for_mega_moe(
    group, num_experts=896, num_max_tokens_per_rank=4096,
    num_topk=16, hidden=3584, intermediate_hidden=3072,
    mma_type="bf16xfp4", activation="situ")
y = torch.empty_like(x)
dg.mega_moe_pre_dispatch(
    x, topk_ids, topk_weights,
    workspace.x, workspace.x_sf, workspace.topk_idx, workspace.topk_weights,
    num_tokens=x.shape[0], mma_type=workspace.mma_type)
dg.bf16_fp4_mega_moe(y, l1, l2, workspace, activation="situ")
```

```python
# W4A8 uses the common workspace entry point; I remains the activated width.
l1, l2 = dg.transform_weights_for_w4a8_mega_moe(raw_l1_w4, raw_l2_w4)
workspace = dg.SymmBuffer(
    group, num_experts=896, num_max_tokens_per_rank=4096,
    num_topk=16, hidden=3584, intermediate_hidden=3072,
    mma_type="fp8xfp4", activation="situ")
y = torch.empty_like(x)
dg.mega_moe_pre_dispatch(
    x, topk_ids, topk_weights,
    workspace.x, workspace.x_sf, workspace.topk_idx, workspace.topk_weights,
    num_tokens=x.shape[0], mma_type=workspace.mma_type)
dg.fp8_fp4_mega_moe(y, l1, l2, workspace, activation="situ")
```

FP8 weights are contiguous E4M3 `[E,N,K]`. Channelwise FP32 scales are
`[E,N,1]`; blockwise scales are `[E,N/128,K/128]`. The transform interleaves
GEMM1 gate/up rows in groups of eight or 128 respectively, including their
scales, without requantization. In these weight shapes, E is the number of
local experts in the bank, N the output width, and K the input width. All raw
weights and scales must be contiguous and on the same device.

W4A16 and W4A8 take uint8-packed MXFP4 weights `[E,2I,H/2]` and `[E,H,I/2]`,
with raw uint8 UE8M0 group-32 scales `[E,2I,H/32]` and `[E,H,I/32]`.
GEMM1 raw rows contain all gate rows followed by all up rows. The W4A16
transform interleaves gate/up rows in groups of eight and packs its scale layout.
W4A8 uses the same raw MXFP4 inputs, transformed with
`transform_weights_for_w4a8_mega_moe`. Each returned bank is
`(packed_weight, packed_scale, weight_inv_scale)`: the scalar compensates
load-time exponent rebasing and is independent of routing weights and input
scales. Prepare separately allocated master/replica banks independently; their
inverse scales may differ. W4A8 scale exponents follow the
[ordinary W4A8 preprocessing requirements](#w4a8-dense-and-moe-gemms-m890).
FP8, W4A8 and W4A16 `base` aliases require the same mode, quantization, group,
device and workspace dimensions; aliases execute serially on the same stream.
W4A8 input and activated output use rowwise E4M3 quantization, as for FP8
channelwise, with the chosen activation fused into GEMM1.
Transformed weights are specific to their backend and must not be interchanged.

FP8 `x` is E4M3 `[capacity,H]`, with FP32 `x_sf` `[capacity,1]` for channelwise
or `[capacity,H/128]` for blockwise. W4A16 `x` is BF16 `[capacity,H]`, and
`x_sf` is an empty `[capacity,0]` view. `mega_moe_pre_dispatch` fills these
inputs and routes. FP8 also quantizes activated GEMM1 output to E4M3; W4A16
keeps BF16 activated output. SiTU is fused into GEMM1.
These backends write BF16 GEMM2 scatter results and launch a separate combine
kernel inside the complete C++ compute call. Combine handles the generation
barrier, weighted reduction and dispatch-counter recycling.

An optional `cumulative_local_expert_recv_stats` is a contiguous device int32
vector with one element per local physical expert, initialized by the caller.
Compute adds valid routed assignments, including zero-weight assignments;
it does not reset these cumulative statistics. Keep weights, output, workspace
and statistics alive until execution and any captured graph replays finish.

The common compute arguments retain their names and meaning. Precision-specific
entry points omit the MXFP4 `recipe`: FP8 granularities are selected by
the workspace, W4A8 weight groups are fixed at 32, and W4A16 does not quantize
activations. SwiGLU-OAI uses
keyword-only `activation_alpha=None` (effective default `1.702`); an explicit
alpha is accepted only for OAI. W4A16 and W4A8 SiTU retain `situ_beta=4` and
`situ_linear_beta=25`; `situ_linear_beta=None` disables the up softcap, while
`situ_beta` must be positive and finite. All pre-dispatch paths accept int32
or int64 routing IDs and store int64 IDs in the workspace.
FP8 and W4A8 SwiGLU/SwiGLU-OAI round clamp boundaries to BF16. For these
MegaMoE paths, `activation_clamp=None` or `+inf` disables clipping; zero is
an active clamp boundary. SiTU (W4A16 and W4A8) rejects any explicit
`activation_clamp`. `fast_math` selects activation math only.

#### HyperConnection (HC) prenorm GEMM

We provide a HyperConnection (HC) prenorm GEMM that fuses the prenorm square-sum reduction into a TF32 GEMM. Call `deep_gemm.tf32_hc_prenorm_gemm` with a BF16 input `a`, a TF32 weight `b`, and it produces the TF32 output `d` together with the per-row square sum `sqr_sum` used by the subsequent normalization. The K-split configuration is selected automatically based on the K dimension.

For more information, please refer to the `tf32_hc_prenorm_gemm` function documentation.

#### Mega-mHC for DeepSeek V4.1 Flash (810E and M890)

`deep_gemm.mega_mhc` implements the fused shifted mHC computation used by
**DeepSeek V4.1 Flash** on **Zhenwu 810E and M890**. The current interface
supports `hc_mult=4`, hidden size `5120`, and BF16 output.
M890 additionally supports FP8 E4M3 output.
It computes the updated residual, mix coefficients, and RMSNorm output into
caller-provided output buffers.

On M890, provide at least one of `y_bf16` or `y_fp8`. FP8 quantization uses the
BF16-rounded RMSNorm output and requires one of these scale formats:

- **Grouped UE8M0**: one power-of-two scale per 32 values, with four UE8M0
  scales packed into each INT32 element of a `[T, H / 128]` tensor. Use
  `y_gemm_sf` for aligned column-major scales, or `y_routed_sf` together with
  `y_shared_sf` for contiguous routed scales and padded, interleaved shared
  scales. The latter also requires `shared_sf_block_m > 0`.
- **Per-token FP32 (M890 only)**: one FP32 scale per token in a contiguous `[T, 1]`
  `y_token_sf` tensor. `y_token_sf` is an optional extension to the upstream
  API and cannot be combined with the grouped scale outputs. The scale is
  used for dequantization: `y ≈ y_fp8 * y_token_sf`.

810E uses three kernels for residual/RMSNorm, projection, and coefficients.

See the [API definition](csrc/apis/mega_mhc.hpp) for argument and layout requirements,
and [the reference and correctness tests](tests/test_mega_mhc.py) for a
complete invocation.

#### Einsum (batched GEMM with fused permute)

We provide einsum-style batched GEMM interfaces that fuse a `(1,0,2)` permute on the LHS into the GEMM flow, currently supporting the `bhr,hdr->bhd` expression:

- `deep_gemm.fp8_einsum`: FP8 LHS/RHS with FP32 scales (per-channel or blockwise) and BF16 output.
- `deep_gemm.int8_einsum`: INT8 LHS/RHS with FP32 per-channel scales and BF16 output.

Both take `(tensor, scales)` tuples as `a`/`b`, write to the BF16 output `d`, and optionally accumulate onto `c`.

For more information, please refer to the `fp8_einsum` and `int8_einsum` function documentation.

#### Interfaces exposed but not yet implemented

For API compatibility with the upstream DeepGEMM, some symbols are exported by `deep_gemm` but are **not yet ported to the PPU build**. Calling them raises `NotImplementedError`:

- K-axis grouped GEMMs: `k_grouped_fp8_gemm_nt_contiguous`, `k_grouped_fp8_gemm_tn_contiguous`
- M-grouped NN contiguous: `m_grouped_fp8_gemm_nn_contiguous`
- FP8xFP4 dense GEMMs: `fp8_fp4_gemm_nt`, `fp8_fp4_gemm_nn`, `fp8_fp4_gemm_tn`, `fp8_fp4_gemm_tt`
- FP8xFP4 M-grouped GEMMs: `m_grouped_fp8_fp4_gemm_nt_contiguous`, `m_grouped_fp8_fp4_gemm_nn_contiguous`, `m_grouped_fp8_fp4_gemm_nt_masked`
- `fp8_gemm_nt_skip_head_mid`

In addition, the dense GEMM kernels only implement the NT layout. The `nn` / `tn` / `tt` variants (`bf16_gemm_{nn,tn,tt}`, `fp8_gemm_{nn,tn,tt}`) are exposed for naming completeness but raise an error, because non-NT layouts are not supported for performance reasons. Use the `acblaslt_gemm_{nt, nn, tn, tt}` fallback if a non-NT layout is required.

#### Utilities

The library provides some utility functions besides the above kernels:

- `deep_gemm.set_num_sms` / `get_num_sms`: set/get the maximum CU count to use
- `deep_gemm.set_tc_util` / `get_tc_util`: set/get an approximated tensor cell utilization ratio
- `deep_gemm.set_pdl` / `get_pdl`: enable/disable Programmatic Dependent Launch (PDL)
- `deep_gemm.set_compile_mode` / `get_compile_mode`: set/get the JIT compile mode
- `deep_gemm.set_ignore_compile_dims`: configure dimensions to ignore during JIT compilation
- `deep_gemm.set_block_size_multiple_of`: constrain block sizes to be multiples of a given value
- `deep_gemm.get_mk_alignment_for_contiguous_layout`: get the group-level M/K alignment requirement for grouped contiguous layout
- `deep_gemm.get_tma_aligned_size`: get the required AIU alignment size
- `deep_gemm.get_mn_major_tma_aligned_tensor`: get an MN-major AIU-aligned tensor
- `deep_gemm.get_col_major_tensor`: get a column-major tensor
- `deep_gemm.transform_sf_into_required_layout`: transform scaling factors into the layout required by the target recipe

The library also provides some environment variables, which may be useful:

- **General**
  - `DG_JIT_DEBUG`: `0` or `1`, print more JIT debugging information, `0` by default
- **JIT cache related**
  - `DG_JIT_CACHE_DIR`: string, the cache directory to store compiled kernels, `$HOME/.deep_gemm` by default
  - `DG_JIT_DISABLE_SHORTCUT_CACHE`: `0` or `1`, bypass the native PPU1.5 dense BF16 (including FP32 output and GEMV), FP8, INT8 and FP4 compiled-variant lookup and regenerate the JIT source on every call, `0` by default. `DG_JIT_DEBUG=1` also bypasses this lookup. Tensor pointers and launch arguments are always refreshed.
- **HGCC/HGRTC selections**
  - `DG_JIT_USE_HGRTC`: `0` or `1`, use HGRTC instead of HGCC, faster compilation but maybe have lower performance for some cases, `0` by default
  - `DG_JIT_HGCC_COMPILER`: string, specified compiler path; will find in `PPU_SDK` or `PPU_HOME` env by default
- **Compiler options**
  - `DG_JIT_CPP_STANDARD`: integer, native HGCC C++ standard version, `17` by default (replaces `DG_CPP_STANDARD`); HGRTC uses C++17
  - `DG_JIT_DISABLE_PCH`: `0` or `1`, disable precompiled headers during JIT compilation, `0` by default
  - `DG_JIT_PTXAS_VERBOSE`: `0` or `1`, show detailed PTXAS compiler output, `0` by default
  - `DG_JIT_WITH_LINEINFO`: `0` or `1`, embed source line info for profiling tools, `0` by default
  - `DG_JIT_PRINT_LOAD_TIME`: `0` or `1`, print the kernel load time, `0` by default
  - `DG_JIT_DUMP_ASM` / `DG_JIT_DUMP_SASS`: `0` or `1`, save `kernel.sass` after a fresh compilation; cache hits skip export, `0` by default
- **Runtime**
  - `DG_USE_PYTORCH_ACBLASLT_HANDLE`: `0` or `1`, use the PyTorch-managed acBLASLt handle instead of the built-in one, `0` by default
  - `SAIL_DEEPGEMM_LUT_PATH`: directory containing external GEMM tuning tables; unset by default. See [External GEMM tuning tables](#external-gemm-tuning-tables).
- **Testing**
  - `DG_USE_PPU_TOOLS`: `0` or `1`, skip the built-in Kineto profiling to stay compatible with external tools (Asight Systems / Asight Compute / Compute Sanitizer), `0` by default

For additional examples and details, please refer to [the test code](tests/test_core.py) or review the corresponding Python documentation.

#### External GEMM tuning tables

Set `SAIL_DEEPGEMM_LUT_PATH` before the first GEMM call. Tables are loaded
on demand and cached for the process lifetime. Files use the name
`<device_name>_<precision>_<gemm_type>.lut`, where `device_name` is the
runtime device property name, including any spaces. Supported precision
names are `bf16`, `int8`, `fp8_blockwise`, `fp8_channelwise`, and `fp4`;
supported GEMM types are `densegemm`, `groupednopad`, and `groupedmasked`.

Each non-comment row contains positive integers in the following order
(whitespace-separated; `#` starts a comment):

```text
# densegemm: 9 integers
M N K BLOCK_M BLOCK_N BLOCK_K WARP_M WARP_N STAGES
# groupednopad / groupedmasked: 10 integers
NUM_GROUPS M N K BLOCK_M BLOCK_N BLOCK_K WARP_M WARP_N STAGES
```

Dense lookup rounds M up to a multiple of 16 and requires an exact match.
Grouped lookup matches N, K, and group count, then selects the smallest
table M greater than or equal to the requested M. Missing or malformed
files, or queries with no matching entry, fall back to built-in
configuration selection. These tables supply configurations; they do not
run an automatic tuning search. See the
[LUT loader](csrc/jit_kernels/heuristics/runtime_lut.hpp) for validation rules.

## Optimizations

We indicate the techniques excluded from ACTLIZE with 🐳.

#### Persistent warp-specialization

Following the ACTLIZE design, the kernels in DeepGEMM for PPU adopt warp interleave design, enabling overlapping data movement, MMA instructions, and promotion operations.

#### Common detail optimizations

- Larger block sizes (up to 256x256 🐳)

#### A unified and optimized block scheduler

- [One scheduler](deep_gemm/include/deep_gemm/scheduler.cuh) for all non-grouped and grouped kernels
- [Rasterization](https://github.com/NVIDIA/cutlass/blob/eefa171318b79cbe2e78514d4cce5cd0fe919d0c/media/docs/efficient_gemm.md#threadblock-rasterization) to enhance L2 cache reuse

#### Fully JIT design 🐳

DeepGEMM uses a C++ [Just-In-Time](csrc/jit) (JIT) implementation. Installation builds the C++ extension; kernels are compiled at runtime. This approach offers several advantages:

- GEMM shapes, block sizes, and the number of pipeline stages are treated as compile-time constants
  - Saving registers
  - Compilers may do more optimizations
- Automatic selection of block sizes, number of warpgroups, and optimal pipeline stages
  - But without auto-tuning, the optimal one is deterministically selected
- Full unrolling of the MMA pipelines, providing compilers with more optimization opportunities
  - Very important for small shapes
  - Refer to `launch_k_iterations` in [the kernel file](deep_gemm/include/deep_gemm/fp8_gemm.cuh) for details

Overall, JIT significantly improves performance for small shapes, similar to the approach of the [Triton](https://github.com/triton-lang/triton/) compiler.

#### Unaligned block sizes 🐳

For certain shapes, block sizes aligned to powers of 2 can lead to underutilized SMs. For instance, with `M=256, N=7168`, a typical block size assignment of `BLOCK_M=128, BLOCK_N=128` results in only `(256 / 128) * (7168 / 128) = 112` out of 132 SMs being utilized. To address this, we support unaligned block sizes like 112, enabling `(256 / 128) * (7168 / 112) = 128` SMs to work in such scenarios. Implementing this technique alongside fine-grained scaling requires careful optimization but ultimately delivers performance gains.

## Acknowledgement

DeepGEMM is inspired by the [CUTLASS](https://github.com/nvidia/cutlass) project. Thanks and respect to the developers!

## License

This code repository is released under [the MIT License](LICENSE).

## Citation

```bibtex
@misc{deepgemm2025,
      title={DeepGEMM: clean and efficient FP8 GEMM kernels with fine-grained scaling},
      author={Chenggang Zhao and Liang Zhao and Jiashi Li and Zhean Xu},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/DeepGEMM}},
}
```
