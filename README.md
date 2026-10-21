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

| Algorithm | INT8 | FP4 | FP8 | BF16 | W4A16 | W4A8 (M890) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Non-grouped** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Contiguous** | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **No-pad** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Masked** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Fused** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

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

For more information, please refer to the corresponding `*_mqa_logits` and `*_paged_mqa_logits` function documentation.

#### HyperConnection (HC) prenorm GEMM

We provide a HyperConnection (HC) prenorm GEMM that fuses the prenorm square-sum reduction into a TF32 GEMM. Call `deep_gemm.tf32_hc_prenorm_gemm` with a BF16 input `a`, a TF32 weight `b`, and it produces the TF32 output `d` together with the per-row square sum `sqr_sum` used by the subsequent normalization. The K-split configuration is selected automatically based on the K dimension.

For more information, please refer to the `tf32_hc_prenorm_gemm` function documentation.

#### Einsum (batched GEMM with fused permute)

We provide einsum-style batched GEMM interfaces that fuse a `(1,0,2)` permute on the LHS into the GEMM flow, currently supporting the `bhr,hdr->bhd` expression:

- `deep_gemm.fp8_einsum`: FP8 LHS/RHS with FP32 scales (per-channel or blockwise) and BF16 output.
- `deep_gemm.int8_einsum`: INT8 LHS/RHS with FP32 per-channel scales and BF16 output.

Both take `(tensor, scales)` tuples as `a`/`b`, write to the BF16 output `d`, and optionally accumulate onto `c`.

For more information, please refer to the `fp8_einsum` and `int8_einsum` function documentation.

#### MegaMoE

MegaMoE supports W4A4, W4A16 and W8A8 routed MoE pipelines on PPU1.5.
Allocate a workspace with `get_symm_buffer_for_mega_moe`, transform both expert
weights once with `transform_weights_for_mega_moe`, then call
`mega_moe_pre_dispatch` followed by one complete output-first entry point:
`fp4_fp4_mega_moe`, `bf16_fp4_mega_moe` or `fp8_fp8_mega_moe`.
These entries own dispatch/GEMM1, GEMM2/scatter, counter recycling and combine.
`intermediate_hidden` always means the activated width I, not 2I.
The calling contract below covers supported layouts and workspace reuse.
See the [precision-specific end-to-end tests](tests/README.MD#routed-moe-correctness-and-performance-tests).

##### Reference and scope

The calling convention follows upstream DeepGEMM commit
[`66081d4c9c7d7c44f13fea402e5b622aa0f409c2`](https://github.com/deepseek-ai/DeepGEMM/blob/66081d4c9c7d7c44f13fea402e5b622aa0f409c2/deep_gemm/mega/__init__.py):
a reusable symmetric workspace, a common load-time weight transformation, and
an output-first complete MoE call. This identifies the API reference, not the
historical source of the PPU kernels. The PPU implementation keeps its existing
multi-kernel pipeline. No cross-architecture dispatcher is introduced.

The PPU interface differs from upstream as follows:

- PPU uses `get_symm_buffer_for_mega_moe` rather than upstream `SymmBuffer`.
- The complete entries are named by activation/input and weight precision:
  `fp4_fp4_mega_moe`, `bf16_fp4_mega_moe`, and `fp8_fp8_mega_moe`.
- Pre-dispatch accepts BF16 hidden states and performs required input
  quantization/staging. It stays a separate explicit call, as in the existing
  PPU pipeline. Router and shared experts remain outside the library.
- Complete calls use BF16 scatter/combine.
- Optional `w1_replicas` and `w2_replicas` hold separately allocated physical
  replica experts. Expert placement and logical-to-physical routing remain the
  caller's responsibility. Both stages must use the same physical order.

##### Supported complete entries

| Entry | Workspace precision | GEMM1 activation | Output |
| --- | --- | --- | --- |
| `fp4_fp4_mega_moe` | `w4a4` | SwiGLU, Step, SiTU | BF16 |
| `bf16_fp4_mega_moe` | `w4a16` | SiTU only | BF16 |
| `fp8_fp8_mega_moe` | `fp8_channelwise` | SwiGLU, SwiGLU-OAI | BF16 |
| `fp8_fp8_mega_moe` | `fp8_blockwise` | SwiGLU, SwiGLU-OAI | BF16 |

Use `activation="swiglu"`, `"step"`, `"situ"` or `"swiglu_oai"`.
`activation_limit` optionally sets a finite positive clamp for supported SwiGLU
variants; SiTU uses `situ_beta=4` and `situ_linear_beta=25`.
OAI defaults to `activation_alpha=1.702`. Unsupported combinations raise before
dispatch/GEMM1. Workspace precision determines output quantization between the
two GEMMs. The default tuning selector uses actual local token count, not
capacity. Configuration is selected and cached internally.

##### Allocation and weight preparation

`hidden` is H; `intermediate_hidden` is I for **every** precision. GEMM1 logical
weights have 2I rows ordered `[gate; up]`; GEMM2 has H rows and I columns.
I must be a positive multiple of 128. H must be a positive multiple of 512 for
W4A4, or 256 otherwise. Local physical expert counts must agree across ranks.
The existing precision-specific EP restrictions still apply.

Pass quantized `(weight, scale)` pairs to `transform_weights_for_mega_moe`:

| Precision | Weight storage | Raw scales |
| --- | --- | --- |
| W4A4/W4A16 | uint8 `[E, N, K/2]`, adjacent K nibbles, low nibble first | uint8 UE8M0 `[E, N, K/32]` |
| FP8 channelwise | E4M3 `[E, N, K]` | FP32 `[E, N, 1]` |
| FP8 blockwise | E4M3 `[E, N, K]` | FP32 `[E, N/128, K/128]` |

The transformation does not quantize weights. It rearranges G1 into G8
(gate/up groups of eight), or G128 for blockwise FP8, and packs scales for the
backend. Do not pass previously transformed tensors through it again.
Transform at model load time, outside graph capture and timing. The returned
pairs are passed directly to the complete compute entry.

```python
import torch
import deep_gemm as dg

# group is initialized; the current device is selected on every rank.
# w1_raw/w2_raw are quantized pairs in the table's logical layouts.
w1, w2 = dg.transform_weights_for_mega_moe(w1_raw, w2_raw, precision="w4a4")
buffer = dg.get_symm_buffer_for_mega_moe(
    group, num_experts=384, num_max_tokens_per_rank=4096,
    num_topk=6, hidden=7168, intermediate_hidden=3072, precision="w4a4")
y = torch.empty_like(x)  # x: BF16 [local_tokens, 7168]

def forward():
    # IDs: contiguous int64; original weights: contiguous FP32 [local_tokens, 6].
    dg.mega_moe_pre_dispatch(x, topk_ids, topk_weights, buffer)
    dg.fp4_fp4_mega_moe(y, w1, w2, buffer,
                      activation="swiglu", activation_limit=10.0)
    return y

# Compile every stage on all ranks before launching cooperative kernels.
previous_mode = dg.get_compile_mode()
dg.set_compile_mode(1)  # compile/load only
try:
    forward()
finally:
    dg.set_compile_mode(previous_mode)
group.barrier().wait()
forward()  # warm up before graph capture
# Reuse the same workspace for subsequent calls, including new routing values.
# After all work/graphs using it have completed:
torch.cuda.synchronize()
buffer.destroy()
```

##### State and routing contract

Allocation is collective, outside capture. The workspace owns activation and
scale scratch, route metadata, GEMM2 scheduler scratch, combine storage, dispatch
counts and barrier state. Scheduler scratch is sized for the workspace capacity
and reused when token counts, routes or tuning tiles change; no caller-managed
reset is needed. Count reset, cross-rank generation barrier and stream-ordered completion
reset are internal. Callers never clear counters or construct peer pointers.

All ranks, including ranks with zero tokens, issue the same number of calls in
the same order. Use one stream per workspace and keep its storage alive while
any graph or kernel references it. Concurrent invocations require separate
workspaces. `y` is contiguous BF16 `[local_tokens, H]` on the workspace device.
`local_tokens` is inferred by pre-dispatch and must not exceed capacity.
For W4A16 decode, `num_tokens_per_rank` can choose a common allocation bucket;
this must be identical across ranks even if one rank's actual input is empty.

Provide the original top-k weights only to pre-dispatch. Every precision
applies them exactly once in combine, after GEMM2 BF16 conversion. W4A4
activation quantization therefore operates on unweighted values. These are
routing weights, not the quantization scales used by GEMM. Invalid routes
(`ID=-1`) are masked internally; zero-weight combine slots are never read.
Graph replay re-reads current input IDs and weights and safely recycles state.
The caller supplies distinct valid expert IDs within each token's top-k list.

##### Implementation and tuning

FP8 channelwise, FP8 blockwise and W4A16 share a rank-local G1 work queue.
Pull CTAs join computation after publishing all their input blocks; consumers
still wait only for the input blocks required by their current GEMM tile.
Channelwise prefetches one next tile using the existing MMA/epilogue barriers;
W4A16 claims one tile at a time; blockwise retains its measured two-tile batch.
Measured profiles select the queue policy internally. On ZW-M890P with 39 CUs,
these queue profiles cover GLM52 channelwise EP8 at 32/1024 tokens per rank and
GLM53 Flash channelwise EP4 at 1024 tokens per rank. Other counts retain the existing
schedule. No queue-mode environment variable is required.

GLM53 Flash channelwise EP4 at 1024 tokens per rank also selects BM128 for
dispatch/GEMM1, with WM32 and 512 threads. Other channelwise profiles retain
BM256. The symmetric workspace layout is independent of this rank-local tile;
the same workspace can switch tiles as input counts change, including empty
ranks. Omit `block_m` to use these profiles; an explicit 128 or 256 overrides
the table. GEMM2 consumes the row metadata produced by each invocation.

Kimi K3 W4A16 EP8 at 1024/2048/4096 tokens per rank uses separate pull dispatch
and GEMM1 launches, with 64-row transport padding and a BM128 SiTU GEMM1 tile.
The two launches use independent CTA geometry; SiTU remains fused into GEMM1.

FP8 channelwise always quantizes the activated BF16 G1 output in a
stream-ordered standalone kernel before G2. Quantization walks only valid
expert rows from the G1 metadata and preserves their pool offsets. Its launch
uses 128 threads per CTA and four CTAs per configured SM, independently of the
G1 tile. Widths with partial vector iterations use a predicated tail. There is
no quantization-mode environment variable or model-specific placement policy.

W4A16 supports SiTU with small-token dispatch and compute tiles for Kimi K3
EP32 decode. Blockwise FP8 supports SwiGLU and SwiGLU-OAI, native 128-by-128
weight scales, BM128/BM192 compute tiles and up to 64 peer ranks.
K3 decode profiles include the buffer tile, separate dispatch/GEMM1 launch
geometry, GEMM2 tile and combine configuration in the same tuning table.
Blockwise gate/up weights use G128 interleaving; regenerate cached G8 weights
and expanded scales from their logical representation with `transform_weights_for_mega_moe`.
Symmetric allocations use Torch directly; blockwise expert landing pools
use local allocations. The complete entry points reuse the staged kernels and fused GEMM2 metadata preparation.

The source follows the native extension layout:

| Path | Responsibility |
| --- | --- |
| `deep_gemm/__init__.py` | Public compute APIs imported directly from `.deep_gemm_cpp` |
| `csrc/python_api.cpp` | `PYBIND11_MODULE` entry, registering the API groups |
| `csrc/apis/mega_moe.hpp` | Public function definitions, input validation/normalization and Python binding adapters |
| `csrc/jit_kernels/impls/mega_moe*.hpp` | Typed `_impl` entrypoints, workspace/scheduler/argument assembly, source generation, JIT compilation and launch |
| `csrc/jit_kernels/heuristics/mega_moe.hpp` | C++ configuration types and tile, shared-memory and scheduling policies |
| `deep_gemm/mega_moe.py` | Symmetric buffer ownership and process-group lifecycle |
| `deep_gemm/include/deep_gemm/` | Device kernels and shared layouts |

`deep_gemm_cpp` is the compiled extension module (`deep_gemm_cpp*.so`), built
from `csrc/python_api.cpp` and the headers it includes; it is not a Python file.
Standalone grouped GEMMs continue to use `csrc/apis/gemm.hpp`.
MegaMoE follows the same API/implementation split: internal calls use tensors,
scalars and typed C++ configurations. Python configuration dictionaries are
converted at the API boundary; internal launch code does not depend on Python
objects or positional Python tuples.
Configuration queries and load-time weight preprocessing remain helpers;
dispatch/GEMM1 produces GEMM2 row and route metadata in the same kernel.
The precision/activation-specific
compute exports and their Python forwarding modules have been removed.

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
- **Testing**
  - `DG_USE_PPU_TOOLS`: `0` or `1`, skip the built-in Kineto profiling to stay compatible with external tools (Asight Systems / Asight Compute / Compute Sanitizer), `0` by default

For additional examples and details, please refer to [the test code](tests/test_core.py) or review the corresponding Python documentation.

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
