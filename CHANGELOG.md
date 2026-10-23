# Changelog

This document records user-visible changes for formal DeepGEMM-for-sail releases, with the latest release listed first.

DeepGEMM-for-sail maintains its release identifiers independently of the upstream [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) project.

## Unreleased

### Added

- **Mega-mHC for DeepSeek V4.1 Flash**: adds `mega_mhc` for fused shifted
  mHC computation on Zhenwu M890P, with `hc_mult=4`, hidden size
  `5120`, and BF16 output. Writes the updated residual, mix coefficients,
  and RMSNorm output into caller-provided buffers; FP8 output is not supported.
- **MXFP4 sparse MQA**: adds `get_sparse_mqa_logits_metadata`,
  `fp8_fp4_sparse_mqa_logits`, `get_paged_sparse_mqa_logits_metadata`, and
  `fp8_fp4_paged_sparse_mqa_logits` for sparse prefill and paged decode on
  Zhenwu M890P. Supports 32 heads, head dimension 128, and sparse KV blocks of
  8 or 16 tokens. Returns BF16 logits compacted by selected block slot;
  paged Q uses `[T,1,32,64]` packed codes and `[T,1,32]` scales.
- **External GEMM tuning tables**: adds `SAIL_DEEPGEMM_LUT_PATH` for dense,
  grouped nopad, and grouped masked GEMMs, with device/precision-specific
  files and fallback to built-in configuration selection.
- **STEP FP4 epilogues**: adds `activation_type="silu_and_mul_step"` to
  FP4 grouped nopad, masked, and fused activation/quantization epilogues
  enabled by `out_scale`. This mode clamps the gate result after SiLU.
- **M890 W4A8 MoE GEMMs**: adds `m_grouped_gemm_fp8_fp4_bf16_nt_nopad`,
  `m_grouped_gemm_fp8_fp4_bf16_nt_masked`, and `m_grouped_gemm_fp8_fp4_bf16_nt_fused` for per-row
  E4M3 activations and group-32 MXFP4 weights. Uses FP4 MMA dequantization
  to E5M2, E4M3 x E5M2 MMA, FP32 accumulation, and BF16 output.
- **W4A8 scheduling and scale handling**: provides independent automatic
  tile/resource selection, fused gather with grouped unpadded output, and
  masked valid-row execution. Requires `N%64=0` and `K%128=0`.
  `W4A8_MXFP4_MMA` preprocessing returns a third scalar `weight_inv_scale`,
  applies a shared power-of-two exponent shift into `[114,140]`, and rejects
  exponent codes 0 and 255 or exponent ranges spanning more than 26. The
  inverse is applied to activation row scales exactly once per GEMM.
- **A8W4 INT8 MoE GEMMs**: adds `m_grouped_gemm_int8_int4_bf16_nt_masked`
  and `m_grouped_gemm_int8_int4_bf16_nt_nopad` for INT8 per-token activations and
  portable packed signed-INT4 per-channel weights on Zhenwu 610 / 610E / 810 / 810E
  (PPU001). Converts INT4 to INT8 in-register for INT8 x INT8 MMA with INT32
  accumulation and BF16 output. Requires `K%32=0` and `N%8=0`; checkpoint weights
  need `preprocess_a8w4_weight` for the device-specific permutation.
- **M890 standalone activation**: adds `silu_and_mul(mode, input, ...)`
  and `situ_and_mul(mode, input, *, beta, ...)`, with a required first
  `ActivationQuantMode` argument: `BF16`, `FP8_CHANNEL`, `FP8_BLOCK`,
  `MXFP4`, or `MXFP8`. BF16 returns a tensor; quantized modes return values
  and scales. Supports dense/batched inputs and masked expert inputs;
  `expected_m` is a scheduling hint and never limits valid-row coverage.
- **Activation quantization and execution**: supports FP32 per-row or
  group-128 FP8 scales, natural MXFP8 scales, and natural or reordered
  M-major MXFP4 scales. FP8-channel can fold a GEMM weight inverse into its
  returned row scales. Processes eight consecutive elements per
  thread with native packed conversions, vector stores, and a shared
  MXFP4/MXFP8 double-buffered pipeline. Partial tiles and unaligned input
  are supported.
  Clamp boundaries round to BF16; other intermediate arithmetic uses
  FP32 with native PPU sigmoid. SiTU supports configurable gate/up softcaps.

### Changed

- **Paged MQA page sizes**: extends regular paged MQA from page size 64
  to 32, 64, and 128.
- **Canonical GEMM bindings**: promotes existing short GEMM names to the
  C++ bindings, while retaining the corresponding old long names as
  Python top-level aliases. Callers importing old names directly from
  `deep_gemm.deep_gemm_cpp` must migrate to the short names or top-level
  `deep_gemm` imports.
- **Scale-layout helper**: moves `transform_sf_into_required_layout` from
  Python utils to a C++ binding. Import it from `deep_gemm`; the old
  `deep_gemm.utils` definition has been removed.
- **Unified weight preprocessing**: introduces
  `preprocess_weights_and_scales(mode, fuse_activation, weight, weight_scale)`
  and exports `PreprocessMode`. Both `mode` and `fuse_activation` are explicit
  arguments; the function returns the processed weights and scales (plus
  `weight_inv_scale` for `W4A8_MXFP4_MMA`). With `fuse_activation=False`,
  `MXFP4` and `W4A8_MXFP4_MMA` also accept omitted or `None` weights and
  return `None` in the weight slot. Scales remain required; W4A8 derives
  dimensions and alignment from `[E,N,K/32]` scales.
- **W4 preprocessing layouts**: accepts naturally packed weights and group-32
  scales for `W4A16_INT4`, `W4A16_MXFP4`, `W4A16_MXFP4_MMA`, and
  `W4A8_MXFP4_MMA` with `fuse_activation=False`. Applies the corresponding
  weight/scale rearrangements; MMA modes retain the natural packed weight layout.
  The W4A8 preprocessing mode rearranges the adjusted E8M0 scales in `[114, 140]`.
- **MXFP4 preprocessing**: `PreprocessMode.MXFP4` with
  `fuse_activation=False` preserves packed weights and converts E8M0 scales
  to padded uint16 M/N-major layout. With `fuse_activation=True`, gate/up
  weights and scales are interleaved before the same scale conversion.
  `preprocess_mxfp4_scales` remains as a deprecated scale-only alias, and
  `preprocess_mxfp4_weight_for_act_and_quant_fusing` remains as a deprecated
  activation-fusion alias. Unsupported mode/fusion combinations report an error.

### Removed

- **W4A16 MXFP4 with BF16 scales**: removes `fp4_use_bf16_scale` from
  `m_grouped_gemm_w4a16_nopad`, `m_grouped_gemm_w4a16_masked`, and
  `m_grouped_gemm_w4a16_fused`, together with the MXFP4-with-BF16-scale
  path. Explicitly passing the old keyword, even as `False`, is no longer
  accepted. Old MXFP4-with-BF16-scale data must be converted to a supported
  format and preprocessed with the matching W4A16 mode; deleting the
  keyword alone does not migrate that data. INT4 weights with BF16 scales
  remain supported.

### Fixed

- Defaults `fp8_paged_mqa_logits` to `clean_logits=False`, matching the
  existing requirement to mask out-of-context positions externally.
  `clean_logits=True` remains unsupported for paged MQA.
- Fixes compilation with PyTorch 2.13.
- Fixes the dynamic INT8 GEMM architecture selection on Zhenwu 810E and
  calculates INT8 shared-memory requirements from the final tile configuration.

## [1.1.0+v0.1.0](https://github.com/t-head/DeepGEMM-for-sail/tree/v1.1.0_v0.1.0_release) — Initial Public Release

This is the first formal public release of DeepGEMM for PPU. The version identifier follows the repository's current release naming convention.

### Platform Support

- Supports Zhenwu 610 / 610E / 810 / 810E / M890 platforms.

### Kernels and Operators

- Provides BF16, INT8, and FP8 dense NT GEMMs, plus grouped NT GEMMs in contiguous, masked, and no-pad layouts.
- Provides MXFP4 dense, grouped masked, and grouped no-pad NT GEMMs.
- Provides W4A16 grouped masked, no-pad, and fused MoE kernels, using MXFP4 weights on M890 and INT4 weights on Zhenwu 610 / 610E / 810 / 810E.
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
