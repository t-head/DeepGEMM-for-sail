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
- **A8W4 INT8 MoE**: INT8 per-token activations and portable packed signed-INT4 per-channel weights, with INT32 accumulation and BF16 output for nopad and masked grouped layouts on Zhenwu 610 / 610E / 810 / 810E (PPU001).
- **Additional nopad grouped GEMM layout**: In addition to contiguous and masked grouped GEMM layouts, this fork introduces a nopad layout. It uses an index tensor to describe per-expert token counts, avoiding mandatory M-block padding for variable-token MoE workloads and reducing unnecessary data movement.
- **AIU-oriented alignment requirements**: The PPU version replaces the original alignment assumptions with AIU-aligned scaling factors and AIU-aligned utility tensors, matching the data movement requirements described by this fork.
- **Warp interleave execution design**: The PPU kernels adopt a warp interleave design to overlap data movement, MMA instructions, and promotion operations during GEMM execution.
- **PPU-oriented scheduling and block-size choices**: The fork keeps the unified scheduler, rasterization strategy, fully JIT design, and unaligned block-size support, while enabling PPU-oriented larger block sizes up to `256x256`.

## Release Notes

### 1.1.0+v0.1.0 — Initial Public Release

This is the first formal public release of DeepGEMM for PPU. The `1.1.0+v0.1.0` identifier follows the current DeepGEMM-for-sail release naming convention and is maintained independently of the upstream DeepGEMM project.

Highlights:

- Native PPU runtime and JIT compilation through HGCC and HGRTC.
- BF16, INT8, FP8, and MXFP4 GEMM kernels across supported dense and grouped layouts, with W4A16 support for no-pad, masked, and fused MoE workloads.
- Fused MoE with implicit permutation, MQA and paged MQA scoring, a TF32 HyperConnection prenorm GEMM, and FP8/INT8 einsum.
- Support for Zhenwu 610 / 610E / 810 / 810E / M890 platforms.

See [CHANGELOG.md](CHANGELOG.md) for the complete release notes.

## Quick Start

### Requirements

- Zhenwu 610 / 610E / 810 / 810E / M890
- Python 3.8 or above
- PPU SDK 2.2 or above
- PyTorch 2.1 or above
- ACTLIZE for PPU: v1.0.0

### Development

```bash
# Submodule must be cloned recursively
git clone --recursive git@github.com:t-head/DeepGEMM-for-sail.git
# Make symbolic links for third-party (ACTLIZE) include directories and build
bash develop.sh

# Test all GEMM implements (normal, contiguous-grouped and masked-grouped)
python tests/test_core.py
```

### Installation

```bash
bash install.sh
```

Then, import `deep_gemm` in your Python project, and enjoy!

## Interfaces

#### Notices

This library focuses on high-performance tensor cell kernels and related LLM computation primitives. GEMM-family interfaces only support the NT format (non-transposed LHS and transposed RHS), and expect contiguous inputs with scaling factors in the layout required by the corresponding quantization recipe (for example, the blockwise FP8 recipe expects a transposed LHS scaling factor). If the layout does not match, the interfaces will fix it up internally with a set of slow PyTorch operations. For transposition or other quantization casting operations, please implement or fuse them into prior kernels independently. While the library provides some simple PyTorch utility functions, these may result in slower performance, but our primary focus is on optimizing the core kernels themselves.

#### Data Precisions

| Algorithm | INT8 | FP4 | FP8 | BF16 | W4A16 | W4A8 (M890) | A8W4 INT8 (PPU001) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Non-grouped** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Contiguous** | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **No-pad** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Masked** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Fused** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

#### Normal dense GEMMs (non-grouped)

To perform a basic non-grouped FP8 GEMM, call the `deep_gemm.gemm_fp8_fp8_bf16_nt` function. For more details, please refer to the function documentation.

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

#### W4A8 MoE GEMMs (M890)

The W4A8 implementation accepts per-row E4M3 activations and group-32
MXFP4 E2M1 weights.
The kernel dequantizes weights to E5M2 using FP4 MMA, performs E4M3 x E5M2
MMA, accumulates in FP32, and writes BF16. It supports these expert layouts:

| API | Activation / row-scale shape | Output shape |
| :--- | :--- | :--- |
| `m_grouped_gemm_fp8_fp4_bf16_nt_nopad` | `[M,K]` / `[M,1]`, grouped by expert | `[M,N]` |
| `m_grouped_gemm_fp8_fp4_bf16_nt_masked` | `[E,T,K]` / `[E,T,1]` | `[E,T,N]`, valid rows only |
| `m_grouped_gemm_fp8_fp4_bf16_nt_fused` | `[num_token,K]` / `[num_token,1]`, original token order | `[num_token*topk,N]`, grouped by expert |

Activation scales are FP32. Weights are naturally packed uint8 `[E,N,K/2]`
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
to the activation row scales exactly once per GEMM; the E4M3 codes remain
unchanged:

```python
import deep_gemm as dg

# packed_weight and raw_weight_scale are checkpoint tensors described above.
_, weight_scale, weight_inv_scale = dg.preprocess_weights_and_scales(
    dg.PreprocessMode.W4A8_MXFP4_MMA, False, weight_scale=raw_weight_scale)
lhs = (a_e4m3, row_scale * weight_inv_scale)
rhs = (packed_weight, weight_scale)
config, rows, experts, tokens, aligned, inv_perm, m_indices = dg.moe_align_block_size(
    a_e4m3, packed_weight, topk_ids)
dg.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
    lhs, rhs, out, rows, experts, tokens, aligned, config)
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
restricted to the valid rows specified by `masked_m`. Nopad/masked select configurations internally unless
an explicit config tuple is supplied. The fused config comes from
`moe_align_block_size`. Configuration selection remains internal to the
C++ runtime.
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
E/N/K alignment from the raw `[E,N,K/32]` scales. Supplying a weight tensor
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

The activation interfaces are `silu_and_mul(mode, input, *, ...)` and
`situ_and_mul(mode, input, *, beta, linear_beta=None, ...)`. The first
`mode` argument is **required** and must be an `ActivationQuantMode`;
omitting it, passing None, or using the old `quant=` keyword is unsupported.
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

M = dg.ActivationQuantMode
bf16 = dg.silu_and_mul(M.BF16, gate_up)
fp8, block_scale = dg.silu_and_mul(M.FP8_BLOCK, gate_up, swiglu_limit=2.1)
# SiTU with gate/up softcaps; down_inv is the GEMM2 weight_inv_scale.
q, row_scale = dg.situ_and_mul(
    M.FP8_CHANNEL, gate_up, beta=4.0, linear_beta=25.0,
    weight_inv_scale=down_inv)
# row_scale already includes down_inv and is passed directly to W4A8 GEMM2.
```

For masked inputs use BF16 `[E,T,2*N]` and contiguous device int32
`masked_m` of shape `[E]`. Both APIs accept `masked_m=masked_m` and an
optional `expected_m` hint. The hint only chooses the number of row slots;
the device loops over all valid rows, including when the hint is smaller
than the actual count. Output values are written only in valid rows.

SiLU computes `gate * sigmoid(gate) * up`. Optional positive
`swiglu_limit` clamps gate from above and up on both sides. Clamp boundaries
round to BF16; activation, multiplication, amax, scale computation and
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

#### MQA logits

We provide Multi-Query Attention (MQA) logits kernels that compute the query-key logits used by the LLM attention indexer. Both a non-paged variant (for contiguous KV) and a paged variant (for a paged KV cache with a block table) are supported, across BF16, INT8, FP8, and FP4 precisions.

- Non-paged: `deep_gemm.bf16_mqa_logits`, `deep_gemm.int8_mqa_logits`, `deep_gemm.fp8_mqa_logits`, and the unified `deep_gemm.fp8_fp4_mqa_logits`. These take the query, key-value pair (with scales for the quantized variants), per-token weights, and the `cu_seq_len_k_start` / `cu_seq_len_k_end` range tensors, and return the masked logits.
- Paged: `deep_gemm.bf16_paged_mqa_logits`, `deep_gemm.int8_paged_mqa_logits`, `deep_gemm.fp8_paged_mqa_logits`, and the unified `deep_gemm.fp8_fp4_paged_mqa_logits`. These operate on a fused KV cache with a `block_table` and schedule metadata for variable-length sequences. Call `deep_gemm.get_paged_mqa_logits_metadata` first to compute the scheduling metadata.

Regular paged MQA supports `page_kv` values of **32, 64, and 128**.
Paged logits require external masking of positions outside each context;
`clean_logits=True` is unsupported. `fp8_paged_mqa_logits` now defaults to
`clean_logits=False`, matching this requirement.

For more information, please refer to the corresponding `*_mqa_logits` and `*_paged_mqa_logits` function documentation.

#### Sparse MQA logits (Zhenwu M890P)

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
`[T,round_up(num_max_sparse_blocks*sparse_block_kv,512)]`.
Logical columns are compacted by selected block slot:
`column = slot*sparse_block_kv + token_in_block`, rather than absolute KV
token position. Padding columns are not additional KV positions.
See the [attention API](csrc/apis/attention.hpp) for metadata and layout
requirements and [attention tests](tests/test_attention.py) for supported
cases.

#### HyperConnection (HC) prenorm GEMM

We provide a HyperConnection (HC) prenorm GEMM that fuses the prenorm square-sum reduction into a TF32 GEMM. Call `deep_gemm.tf32_hc_prenorm_gemm` with a BF16 input `a`, a TF32 weight `b`, and it produces the TF32 output `d` together with the per-row square sum `sqr_sum` used by the subsequent normalization. The K-split configuration is selected automatically based on the K dimension.

For more information, please refer to the `tf32_hc_prenorm_gemm` function documentation.

#### Mega-mHC for DeepSeek V4.1 Flash (M890P)

`deep_gemm.mega_mhc` implements the fused shifted mHC computation used by
**DeepSeek V4.1 Flash** on **Zhenwu M890P**. The current interface
supports `hc_mult=4`, hidden size `5120`, and BF16 output. It computes the
updated residual, mix coefficients, and RMSNorm output into caller-provided
output buffers.

The `y_bf16` output is required. FP8 output and its scale-output parameters
are not supported by the current implementation. See the
[API definition](csrc/apis/mega_mhc.hpp) for argument and layout requirements,
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

- K-axis grouped GEMMs: `k_grouped_fp8_gemm_nt_contiguous`, `k_grouped_fp8_gemm_tn_contiguous`, `k_grouped_bf16_gemm_tn_contiguous`
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
  - `DG_JIT_DISABLE_CACHE`: `0` or `1`, disable the use of cache directory, `0` by default
- **HGCC/HGRTC selections**
  - `DG_JIT_USE_HGRTC`: `0` or `1`, use HGRTC instead of HGCC, faster compilation but maybe have lower performance for some cases, `0` by default
  - `DG_JIT_HGCC_COMPILER`: string, specified compiler path; will find in `PPU_SDK` or `PPU_HOME` env by default
- **Compiler options**
  - `DG_CPP_STANDARD`: integer, C++ standard version, `17` by default
  - `DG_JIT_DISABLE_PCH`: `0` or `1`, disable precompiled headers during JIT compilation, `0` by default
  - `DG_JIT_PTXAS_VERBOSE`: `0` or `1`, show detailed PTXAS compiler output, `0` by default
  - `DG_JIT_WITH_LINEINFO`: `0` or `1`, embed source line info for profiling tools, `0` by default
  - `DG_JIT_PRINT_LOAD_TIME`: `0` or `1`, print the kernel load time, `0` by default
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

DeepGEMM employs a fully [Just-In-Time](deep_gemm/jit) (JIT) design, with no compilation required at installation. All kernels are compiled at runtime using a lightweight JIT implementation. This approach offers several advantages:

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
