#pragma once

#include <cstdint>
#include "cutlass/bfloat16.h"
#include <deep_gemm/cute_free/arch/ppu_common.cuh>

namespace deep_gemm {
namespace arch {

// =============================================================================
// PPU 1.5 (tc02) low-level copy primitives.
//
// These are thin wrappers over the raw PTX instructions. They are templated on
// the tile shape constants so that the swizzle mode / strides are resolved at
// compile time, but they remain bf16-specific at the instruction level. Higher
// level, element-generic abstractions live in deep_gemm/cute_free/copy/*.cuh.
// =============================================================================

/// AIU DMA bulk-tensor load: global -> shared (swizzled, zero-padded).
/// Copies a CUBE_H x CUBE_W tile. Only one warp should issue this.
///   - smem_dst   : shared memory destination (swizzled layout handled by HW)
///   - gmem_base  : global memory base pointer
///   - rows / ld  : tensor rows and leading dimension (elements)
///   - row_offset : starting row in the source tensor
///   - col_offset : starting column (K offset) in the source tensor
template <int CUBE_H, int CUBE_W>
__device__ __forceinline__ void aiu_bulk_tensor_load_bf16(
    cutlass::bfloat16_t* smem_dst, cutlass::bfloat16_t const* gmem_base,
    int rows, int ld, int row_offset, int col_offset) {
  uint32_t s = smem_ptr_to_uint(smem_dst);
  uint64_t tensor_stride_w = (uint64_t)ld * sizeof(cutlass::bfloat16_t);
  uint64_t tensor_stride_n = tensor_stride_w * rows;
  static constexpr int swzl_mode = CUBE_W * (int)sizeof(cutlass::bfloat16_t) == 128 ? 0 : 1;
  asm volatile(
    "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.LLC::128B.padz.swzl.b16 "
    "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
    :: "r"(s), "l"(gmem_base),
      "r"(ld), "r"(rows), "r"(1),
      "r"(CUBE_W), "r"(CUBE_H), "r"(1),
      "l"(tensor_stride_w), "l"(tensor_stride_n),
      "r"(col_offset), "r"(row_offset), "r"(0),
      "r"(swzl_mode)
  );
}

/// TSM_LD: hardware fragment load from swizzled shared memory (ldmatrix.swzl).
/// Loads 4 registers (one 16x16 MMA operand fragment) using hardware de-swizzle.
/// Pure ldmatrix wrapper (mirrors cutlass3 PPU0015_TSM_LD_SWZL_IMPL): the caller
/// computes the final 16B-unit `tsm_add` (stage + sub-tile + row + within-K);
/// this primitive only derives lbo/sbo/swzl_mode from the swizzle-tile width.
///   - tsm_add    : final shared address in 16B units for this fragment
///   - frag       : output 4-register fragment
///   - SWZL_W     : swizzle-tile width in elements (<= 128B / sizeof(Element));
///                  sets row_stride and swzl_mode (128B mode 0 / 64B mode 1).
///   - SwapLboSbo : false for A operand (lbo=1, sbo=row_stride),
///                  true  for B operand (lbo=row_stride, sbo=1) -- hardware
///                  lbo/sbo swap that replaces the software frag[1]<->frag[2]
///                  swap (matches Cutlass3's Swap=true approach).
template <int SWZL_W, bool SwapLboSbo>
__device__ __forceinline__ void tsm_ldmatrix_swzl_bf16(
    int tsm_add, uint32_t (&frag)[4]) {
  static constexpr int kElemBytes = (int)sizeof(cutlass::bfloat16_t);
  static constexpr int row_stride = 8 * SWZL_W * kElemBytes / 16;
  static constexpr int lbo = SwapLboSbo ? row_stride : 1;
  static constexpr int sbo = SwapLboSbo ? 1 : row_stride;
  static constexpr int swzl_mode = SWZL_W * kElemBytes == 128 ? 0 : 1;
  asm volatile(
    "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%0, %1, %2, %3}, [%4], %5, %6, %7;"
    : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
    : "l"(tsm_add), "r"(lbo), "r"(sbo), "r"(swzl_mode)
  );
}

} // namespace arch
} // namespace deep_gemm
