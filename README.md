# DeepGEMM

## Overview

This repository is a PPU-oriented fork of [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), a unified high-performance tensor core kernel library for modern large language model workloads. It brings together low-precision GEMMs (FP8, FP4, BF16), fused MoE (with implicit permute) kernels, MQA scoring for the lightning indexer, HyperConnection (HC), and related primitives into a single cohesive codebase adapted for T-Head PPU hardware.

DeepGEMM for PPU focuses on the supported core computation primitives used by dense models, MoE models, and related LLM workloads. The fork updates the build and runtime path for the PPU SDK environment, replaces the original device-specific assumptions with PPU-oriented execution requirements, and extends the supported layouts and data types for PPU use cases.

Supported PPU platforms include ZW 610 / 610E / 810 / 810E / M890. Runtime compilation is handled through the PPU toolchain, with HGCC / HGRTC support exposed through the JIT configuration options.

## Core Features and Optimizations

The PPU-oriented fork provides the following core capabilities and optimizations:

- **PPU runtime and JIT integration**: The runtime compilation path is adapted to the PPU SDK environment. Kernel compilation, compiler selection, debug output, and cache behavior are controlled through PPU-oriented JIT environment variables such as `DG_JIT_USE_HGRTC`.
- **Supported precision modes**: The PPU version supports INT8, FP4, FP8, and BF16 across non-grouped GEMM and grouped GEMM layouts, covering both dense GEMM and MoE grouped GEMM scenarios.
- **Additional nopad grouped GEMM layout**: In addition to contiguous and masked grouped GEMM layouts, this fork introduces a nopad layout. It uses an index tensor to describe per-expert token counts, avoiding mandatory M-block padding for variable-token MoE workloads and reducing unnecessary data movement.
- **AIU-oriented alignment requirements**: The PPU version replaces the original alignment assumptions with AIU-aligned scaling factors and AIU-aligned utility tensors, matching the data movement requirements described by this fork.
- **Warp interleave execution design**: The PPU kernels adopt a warp interleave design to overlap data movement, MMA instructions, and promotion operations during GEMM execution.
- **PPU-oriented scheduling and block-size choices**: The fork keeps the unified scheduler, rasterization strategy, fully JIT design, and unaligned block-size support, while enabling PPU-oriented larger block sizes up to `256x256`.

## Quick Start

### Requirements

- ZW 610 / 610E / 810 / 810E / M890
- Python 3.8 or above
- PPU SDK 12.3 or above
- PyTorch 2.1 or above
- ACTLIZE for PPU: v1.0.0

### Development

```bash
# Submodule must be cloned recursively
git clone --recursive git@github.com:t-head/DeepGEMM-for-sail.git
# Make symbolic links for third-party (ACTLIZE) include directories
python setup.py develop

# Test JIT compilation
python tests/test_jit.py

# Test all GEMM implements (normal, contiguous-grouped and masked-grouped)
python tests/test_core.py
```

### Installation

```bash
python setup.py install
```

Then, import `deep_gemm` in your Python project, and enjoy!

## Interfaces

#### Notices

This library focuses on high-performance tensor core kernels and related LLM computation primitives. GEMM-family interfaces only support the NT format (non-transposed LHS and transposed RHS), and expect contiguous inputs with scaling factors in the layout required by the corresponding quantization recipe (for example, the blockwise FP8 recipe expects a transposed LHS scaling factor). If the layout does not match, the interfaces will fix it up internally with a set of slow PyTorch operations. For transposition or other quantization casting operations, please implement or fuse them into prior kernels independently. While the library provides some simple PyTorch utility functions, these may result in slower performance, but our primary focus is on optimizing the core kernels themselves.

#### Data Precisions

| Algorithm | INT8 | FP4 | FP8 | BF16 |
| :--- | :---: | :---: | :---: | :---: |
| **Non-grouped** | ✅ | ✅ | ✅ | ✅ |
| **Contiguous** | ✅ | ✅ | ✅ | ✅ |
| **No-pad** | ✅ | ✅ | ✅ | ✅ |
| **Masked** | ✅ | ✅ | ✅ | ✅ |

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

#### Fused MoE (with implicit permute)

Unlike the standalone grouped GEMM layouts above (which expect tokens to already be routed and grouped per expert), the fused MoE path takes raw MoE routing results (`topk_ids`) and fuses the token permutation, per-expert block alignment, and grouped GEMM into a single flow. The "with implicitt ermute" naming distinguishes it from a plain grouped MoE GEMM: the permutation from token order to expert-grouped order is computed and applied inside these kernels, so no separate gather/scatter step is required.

First call `deep_gemm.moe_align_block_size` with the input tokens and `topk_ids` to compute the block-aligned scheduling metadata (`m_rows`, `expert_ids_and_cumsum`, `sorted_token_ids`, `aligned_num_m_blocks`) together with the inverse permutation (`inv_perm`) and the auto-selected GEMM config. Then feed this metadata into one of the fused grouped GEMM kernels according to the input precision:

- `deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_fused`: fused MoE grouped GEMM with BF16 LHS/RHS and BF16 output.
- `deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_fused`: fused MoE grouped GEMM with FP8 LHS/RHS (with scales) and BF16 output; supports both per-channel and blockwise quantization.
- `deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_fused`: fused MoE grouped GEMM with INT8 LHS/RHS (with per-channel scales) and BF16 output.

For more information, please refer to the `moe_align_block_size` and `m_grouped_gemm_*_nt_fused` function documentation.

#### MQA logits

We provide Multi-Query Attention (MQA) logits kernels that compute the query-key logits used by the LLM attention indexer. Both a non-paged variant (for contiguous KV) and a paged variant (for a paged KV cache with a block table) are supported, across BF16, INT8, FP8, and FP4 precisions.

- Non-paged: `deep_gemm.bf16_mqa_logits`, `deep_gemm.int8_mqa_logits`, `deep_gemm.fp8_mqa_logits`, and the unified `deep_gemm.fp8_fp4_mqa_logits`. These take the query, key-value pair (with scales for the quantized variants), per-token weights, and the `cu_seq_len_k_start` / `cu_seq_len_k_end` range tensors, and return the masked logits.
- Paged: `deep_gemm.bf16_paged_mqa_logits`, `deep_gemm.int8_paged_mqa_logits`, `deep_gemm.fp8_paged_mqa_logits`, and the unified `deep_gemm.fp8_fp4_paged_mqa_logits`. These operate on a fused KV cache with a `block_table` and schedule metadata for variable-length sequences. Call `deep_gemm.get_paged_mqa_logits_metadata` first to compute the scheduling metadata.

For more information, please refer to the corresponding `*_mqa_logits` and `*_paged_mqa_logits` function documentation.

#### HyperConnection (HC) prenorm GEMM

We provide a HyperConnection (HC) prenorm GEMM that fuses the prenorm square-sum reduction into a TF32 GEMM. Call `deep_gemm.tf32_hc_prenorm_gemm` with a BF16 input `a`, a TF32 weight `b`, and it produces the TF32 output `d` together with the per-row square sum `sqr_sum` used by the subsequent normalization. The K-split configuration is selected automatically based on the K dimension.

For more information, please refer to the `tf32_hc_prenorm_gemm` function documentation.

#### Utilities

The library provides some utility functions besides the above kernels:

- `deep_gemm.set_num_sms`: set the maximum SM count to use
- `deep_gemm.get_num_sms`: get the current SM maximum count
- `deep_gemm.get_m_alignment_for_contiguous_layout`: get the group-level alignment requirement for grouped contiguous layout
- `deep_gemm.get_tma_aligned_size`: get the required AIU alignment size
- `deep_gemm.get_col_major_tma_aligned_tensor`: get a column-major AIU-aligned tensor

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
  - `DG_JIT_PTXAS_VERBOSE`: `0` or `1`, show detailed PTXAS compiler output, `0` by default
  - `DG_JIT_PRINT_COMPILER_COMMAND`: `0` or `1`, print HGCC compilation command, `0` by default
- **Testing**
  - `DG_NSYS_PROFILING`: `0` or `1`, Asight-system compatible testing, `0` by default

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
