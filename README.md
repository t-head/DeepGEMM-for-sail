# DeepGEMM

## Overview

This repository is a PPU-oriented fork of [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), a unified high-performance tensor cell kernel library for modern large language model workloads. It brings together low-precision GEMMs (FP8, FP4, BF16), fused MoE (with implicit permute) kernels, MQA scoring for the lightning indexer, HyperConnection (HC), and related primitives into a single cohesive codebase adapted for T-Head PPU hardware.

DeepGEMM for PPU focuses on the supported core computation primitives used by dense models, MoE models, and related LLM workloads. The fork updates the build and runtime path for the PPU SDK environment, replaces the original device-specific assumptions with PPU-oriented execution requirements, and extends the supported layouts and data types for PPU use cases.

Supported PPU platforms include ZW 610 / 610E / 810 / 810E / M890. Runtime compilation is handled through the PPU toolchain, with HGCC / HGRTC support exposed through the JIT configuration options.

## Core Features and Optimizations

The PPU-oriented fork provides the following core capabilities and optimizations:

- **PPU runtime and JIT integration**: The runtime compilation path is adapted to the PPU SDK environment. Kernel compilation, compiler selection, debug output, and cache behavior are controlled through PPU-oriented JIT environment variables such as `DG_JIT_USE_HGRTC`.
- **Supported precision modes**: The PPU version supports INT8, FP4, FP8, and BF16 across non-grouped GEMM and grouped GEMM layouts, covering both dense GEMM and MoE grouped GEMM scenarios.
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
- Support for ZW 610 / 610E / 810 / 810E / M890 platforms.

See [CHANGELOG.md](CHANGELOG.md) for the complete release notes.

## Quick Start

### Requirements

- ZW 610 / 610E / 810 / 810E / M890
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

# Test JIT compilation
python tests/test_jit.py

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

For the FP4 masked GEMM, the `silu_and_mul` activation and the MXFP4 quantization that follows it can optionally be fused into the epilogue, so that MoE gemm1 writes the packed FP4 values and their E8M0 scales directly instead of a BF16 tensor that a standalone kernel has to read back. The fusion is **disabled by default**; pass an `out_scale` tensor to `deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked` to enable it, and interleave the gemm1 weight once with `deep_gemm.preprocess_weights_and_scales(mode=deep_gemm.PreprocessMode.MXFP4, fuse_activation=True)` beforehand. Refer to the `m_grouped_gemm_fp4_fp4_bf16_nt_masked` documentation for the tensor layouts, the `swiglu_limit` argument and the current limitations.

#### Fused MoE (with implicit permute)

Unlike the standalone grouped GEMM layouts above (which expect tokens to already be routed and grouped per expert), the fused MoE path takes raw MoE routing results (`topk_ids`) and fuses the token permutation, per-expert block alignment, and grouped GEMM into a single flow. The "with implicit permute" naming distinguishes it from a plain grouped MoE GEMM: the permutation from token order to expert-grouped order is computed and applied inside these kernels, so no separate gather/scatter step is required.

First call `deep_gemm.moe_align_block_size` with the LHS/RHS inputs and `topk_ids` to compute the block-aligned scheduling metadata (`m_rows`, `expert_ids_and_cumsum`, `sorted_token_ids`, `aligned_num_m_blocks`) together with the inverse permutation (`inv_perm`), the per-token expert indices (`m_indices`), and the auto-selected GEMM config. Then feed this metadata into one of the fused grouped GEMM kernels according to the input precision:

- `deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_fused`: fused MoE grouped GEMM with BF16 LHS/RHS and BF16 output.
- `deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_fused`: fused MoE grouped GEMM with FP8 LHS/RHS (with scales) and BF16 output; supports both per-channel and blockwise quantization.
- `deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_fused`: fused MoE grouped GEMM with INT8 LHS/RHS (with per-channel scales) and BF16 output.
- `deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_fused`: fused MoE grouped GEMM with FP4 (mxfp4) LHS/RHS (with scales) and BF16 output. Note that this path expects the **LHS scale (SFA) in uint16 K-major layout**, that is a plain contiguous `(num_token, ceil_div(k // 32, 2))` uint16 tensor, rather than the uint16 M-major layout produced by `preprocess_mxfp4_scales` that the masked/nopad/dense FP4 interfaces expect; the RHS scale (SFB) stays MN-major. To build it, pad the `ceil_div(k, 32)` uint8 E8M0 scales to an even count along K and then view them as uint16; passing a `preprocess_mxfp4_scales` output as SFA here will fail the contiguity assertion.
- `deep_gemm.m_grouped_gemm_w4a8_fused`: fused routing gather with per-row E4M3 activations, group-32 MXFP4 weights, and BF16 output on M890.
- `deep_gemm.m_grouped_gemm_w4a16_fused`: fused MoE grouped GEMM with 4-bit group-quantized weights and BF16 activation (W4A16), producing BF16 output.

For more information, please refer to the `moe_align_block_size` and `m_grouped_gemm_*_nt_fused` function documentation.

#### W4A8 MoE GEMMs (M890)

The W4A8 implementation accepts per-row E4M3 activations and group-32
MXFP4 E2M1 weights.
The kernel dequantizes weights to E5M2 using FP4 MMA, performs E4M3 x E5M2
MMA, accumulates in FP32, and writes BF16. It supports these expert layouts:

| API | Activation / row-scale shape | Output shape |
| :--- | :--- | :--- |
| `m_grouped_gemm_w4a8_nopad` | `[M,K]` / `[M,1]`, grouped by expert | `[M,N]` |
| `m_grouped_gemm_w4a8_masked` | `[E,T,K]` / `[E,T,1]` | `[E,T,N]`, valid rows only |
| `m_grouped_gemm_w4a8_fused` | `[num_token,K]` / `[num_token,1]`, original token order | `[num_token*topk,N]`, grouped by expert |

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
weight, weight_scale, weight_inv_scale = dg.preprocess_weights_and_scales(
    dg.PreprocessMode.W4A8_MXFP4_MMA, False, packed_weight, raw_weight_scale)
lhs = (a_e4m3, row_scale * weight_inv_scale)
rhs = (weight, weight_scale)
config, rows, experts, tokens, aligned, inv_perm, m_indices = dg.moe_align_block_size(
    a_e4m3, weight, topk_ids)
dg.m_grouped_gemm_w4a8_fused(
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
