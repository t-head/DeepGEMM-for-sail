# Changelog

This document records user-visible changes for formal DeepGEMM-for-sail releases, with the latest release listed first.

DeepGEMM-for-sail maintains its release identifiers independently of the upstream [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) project.

## [1.1.0+v0.1.0](https://github.com/t-head/DeepGEMM-for-sail/tree/v1.1.0_v0.1.0_release) — Initial Public Release

This is the first formal public release of DeepGEMM for PPU. The version identifier follows the repository's current release naming convention.

### Platform Support

- Supports ZW 610 / 610E / 810 / 810E / M890 platforms.

### Kernels and Operators

- Provides BF16, INT8, and FP8 dense and grouped NT GEMMs in contiguous, masked, and no-pad layouts.
- Provides MXFP4 dense, grouped masked, and grouped no-pad NT GEMMs.
- Provides W4A16 grouped masked, no-pad, and fused MoE kernels for INT4 and MXFP4 weights.
- Provides implicit-permute fused MoE GEMMs for BF16, INT8, FP8, MXFP4, and W4A16 workloads, with `moe_align_block_size` generating routing and scheduling metadata.
- Adds optional fused `silu_and_mul` plus MXFP4 post-quantization epilogues for FP4 MoE GEMM1 paths.
- Provides non-paged and paged MQA-logits operators for BF16, INT8, FP8, and FP4 inputs.
- Provides TF32 HyperConnection prenorm GEMM with fused per-row square-sum reduction.
- Provides FP8 and INT8 einsum-style batched GEMMs with fused LHS permutation for `bhr,hdr->bhd`.

### JIT and Runtime

- Integrates the PPU toolchain with offline HGCC compilation and optional HGRTC runtime compilation.
- Provides C++ and Python-template JIT paths with runtime specialization for shapes, layouts, tile configurations, pipeline stages, and precision modes.
- Adds persistent kernel caching, in-process runtime caching, compile-only warm-up support, and configurable compiler diagnostics.
- Launches JIT-compiled kernels on the caller's current PyTorch stream.

### Performance and Scheduling

- Uses persistent warp-interleaved execution to overlap data movement, MMA instructions, and promotion operations.
- Provides unified rasterized block scheduling, unaligned tile sizes, and block tiles up to `256x256` for improved L2 reuse and wave utilization.
- Adds adaptive tile selection and prologue-overlap scheduling for BF16 DenseGEMM on M890.
- Adds deterministic fused MoE routing for both small and large routed-token workloads.

### Build and Validation

- Provides `develop.sh` for in-tree development builds and `install.sh` for wheel-based installation.
- Packages the JIT headers, ACTLIZE dependencies, and shipped tuning configurations with the Python package.
- Provides format- and caselist-driven correctness and profiling tools for dense, grouped, fused MoE, and attention workloads.
