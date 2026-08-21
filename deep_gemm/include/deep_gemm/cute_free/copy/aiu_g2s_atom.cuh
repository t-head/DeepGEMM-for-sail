#pragma once

#include <type_traits>
#include "cutlass/bfloat16.h"
#include "cute_free/arch/ppu_tc02_copy.cuh"
#include "copy_traits.cuh"

namespace deep_gemm {
namespace copy {

// =============================================================================
// AiuG2SAtom: Global -> Shared copy atom backed by the AIU bulk-tensor engine.
//
// Templated on the element type and the tile (cube) shape so that the same
// mainloop code can be reused across data types / tile configs by instantiating
// a different atom. The actual PTX is provided by deep_gemm::arch.
//
//   Element : data type (currently cutlass::bfloat16_t)
//   CubeH   : tile height (BLOCK_M for A, BLOCK_N for B)
//   CubeW   : tile width  (BLOCK_K)
// =============================================================================

template <typename Element, int CubeH, int CubeW>
struct AiuG2SAtom {
  static constexpr int kCubeH = CubeH;
  static constexpr int kCubeW = CubeW;
  using Traits = CopyElementTraits<Element>;

  // Swizzle-tile width (cutlass AiuContByteSize capped at 128B): a single AIU
  // instruction can cover at most 128B in the channel (K) direction, so a wider
  // CubeW (= BLOCK_K) is split into kInstNum sub-tiles along K.
  static constexpr int kElemBytes = Traits::kSize;
  static constexpr int kSwzlBytes = (CubeW * kElemBytes > 128) ? 128 : CubeW * kElemBytes;
  static constexpr int kSwzlW     = kSwzlBytes / kElemBytes;
  static constexpr int kInstNum   = CubeW / kSwzlW;

  /// Issue an asynchronous G2S copy of one CubeH x CubeW tile.
  /// Must be issued by a single warp; completion is observed via cp_async_wait.
  /// Split into kInstNum [CubeH x kSwzlW] sub-tiles: sub-tile j reads gmem K at
  /// col_offset + j*kSwzlW and writes smem at + j*CubeH*kSwzlW (matches the S2R
  /// sub-tile layout in TsmS2RAtom).
  static __device__ __forceinline__ void load(
      Element* smem_dst, Element const* gmem_base,
      int rows, int ld, int row_offset, int col_offset) {
    static_assert(std::is_same<Element, cutlass::bfloat16_t>::value,
                  "AiuG2SAtom: only bf16 is supported by the arch layer today");
    #pragma unroll
    for (int j = 0; j < kInstNum; ++j) {
      arch::aiu_bulk_tensor_load_bf16<CubeH, kSwzlW>(
          smem_dst + j * CubeH * kSwzlW, gmem_base, rows, ld,
          row_offset, col_offset + j * kSwzlW);
    }
  }
};

} // namespace copy
} // namespace deep_gemm
