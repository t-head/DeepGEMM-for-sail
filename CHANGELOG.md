# Changelog

All notable changes to DeepGemm for PPU will be documented in this file.

## [1.1.0] - 2026-08-06

### Added

- **Fused MoE GEMM path**: replaces scatter-based physical data movement with
  `moe_align` index remapping; fuses gather-load A + contiguous-load B +
  grouped GEMM + epilogue into a single kernel, covering all supported
  precisions:
  - BF16: `m_grouped_gemm_bf16_bf16_bf16_nt_fused` (890P / 810E)
  - INT8 per-channel: `m_grouped_gemm_int8_int8_bf16_nt_fused` (890P / 810E)
  - FP8 per-channel / blockwise: `m_grouped_gemm_fp8_fp8_bf16_nt_fused` (890P)
  - FP4 (mxfp4): `m_grouped_gemm_fp4_fp4_bf16_nt_fused`
  - W4A16: `m_grouped_gemm_w4a16_fused`
- **`moe_align_block_size` auxiliary kernel**: preprocesses `topk_ids` into the
  index format directly consumable by the GEMM kernels (`sorted_token_ids` +
  `expert_ids_and_cumsum` in uint4 layout + `inv_perm` + `m_indices`); a single
  call returns 7 values including the automatically queried GEMM config:
  - Small M (numel <= 16384): WarpOrdered single-kernel path (deterministic,
    tokens sorted within each expert)
  - Large M: deterministic 4-kernel path (zero global atomicAdd)
- **INT8 einsum**: new `int8_einsum` interface alongside `fp8_einsum`.
