# Changelog

All notable changes to DeepGemm for PPU will be documented in this file.

## [1.1.0] - 2026-08-06

### Added

- **W4A16 MXFP4 weight support**: extends the W4A16 grouped GEMM interface to
  accept FP4 (MXFP4) weights with E8M0 scales, adding `fp4_use_bf16_scale`
  keyword argument to all three kernel variants:
  - `m_grouped_gemm_w4a16_masked(x, y, out, masked_m, expected_m_per_group, *, fp4_use_bf16_scale=False)`
  - `m_grouped_gemm_w4a16_fused(x, y, out, m_rows, expert_ids_and_offset, sorted_token_ids, aligned_num_m_blocks, configs, *, fp4_use_bf16_scale=False)`
  - `m_grouped_gemm_w4a16_nopad(x, y, out, m_indices, *, fp4_use_bf16_scale=False)`
  The scale tensor `y[1]` dtype selects the weight path:
  - `uint8` → E8M0 scale — MXFP4 (`w4fa16`)
  - `bfloat16` (default) → INT4 quantized (`w4a16`)
  - `bfloat16` + `fp4_use_bf16_scale=True` → FP4 with BF16 scale (`w4fa16_s16`)
- **Fused silu_and_mul + MXFP4 post-quant epilogue (MoE gemm1)**: for the masked
  grouped FP4 GEMM, `silu_and_mul` and the MXFP4 post-quantization are fused into the
  GEMM epilogue, so gemm1 writes packed FP4 values plus E8M0 scales directly instead of
  a BF16 tensor that a standalone kernel has to read back. Opt-in through two new
  keyword arguments (**disabled by default**):
  - `m_grouped_gemm_fp4_fp4_bf16_nt_masked(lhs, rhs, bias, out, masked_m, expected_m, ..., out_scale=None, swiglu_limit=0.0)`
  - passing `out_scale` enables the fusion: `out` becomes `uint8` of shape
    `(num_groups, m, n // 4)` and `out_scale` `uint16` of shape
    `(num_groups, m, ceil_div(n // 4, 32))`. On return, `out_scale` is re-strided **in
    place** to the N-major layout `(sfm * sfn, 1, sfm)` expected by the gemm2 SFA reader.
  - `swiglu_limit > 0` applies the clamp before silu (gate clamped from above, up clamped
    on both sides); `0.0` or `None` disables it
  - constraints: `GroupedMasked` only (`GroupedNoPad` not supported yet), `ShapeN % 64 == 0`,
    `BlockN >= 64` (raised automatically), bias not supported
- **`preprocess_mxfp4_weight_for_act_and_quant_fusing(weight, weight_scale)`**: interleaves
  the gate/up (W1/W3) halves of the gemm1 weight and its E8M0 scales so that the fused
  epilogue can read gate/up pairs from adjacent N positions. Must be called before
  `preprocess_mxfp4_scales`.
- **Fused MoE GEMM path**: replaces scatter-based physical data movement with
  `moe_align` index remapping; fuses gather-load A + contiguous-load B +
  grouped GEMM + epilogue into a single kernel, covering all supported
  precisions:
  - BF16: `m_grouped_gemm_bf16_bf16_bf16_nt_fused` (890P / 810E)
  - INT8 per-channel: `m_grouped_gemm_int8_int8_bf16_nt_fused` (890P / 810E)
  - FP8 per-channel / blockwise: `m_grouped_gemm_fp8_fp8_bf16_nt_fused` (890P)
  - FP4 (mxfp4): `m_grouped_gemm_fp4_fp4_bf16_nt_fused` (890) (LHS_scale must be uint16 K-major / plain
    contiguous, unlike the uint16 M-major LHS_scale of the other FP4 interfaces; RHS_scale stays MN-major)
  - W4A16: `m_grouped_gemm_w4a16_fused`
- **`moe_align_block_size` auxiliary kernel**: preprocesses `topk_ids` into the
  index format directly consumable by the GEMM kernels (`sorted_token_ids` +
  `expert_ids_and_cumsum` in uint4 layout + `inv_perm` + `m_indices`); a single
  call returns 7 values including the automatically queried GEMM config:
  - Small M (numel <= 16384): WarpOrdered single-kernel path (deterministic,
    tokens sorted within each expert)
  - Large M: deterministic 4-kernel path (zero global atomicAdd)
- **INT8 einsum**: new `int8_einsum` interface alongside `fp8_einsum`.
