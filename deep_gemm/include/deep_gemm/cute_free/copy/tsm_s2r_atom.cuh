#pragma once

#include <cstdint>
#include <type_traits>
#include "cutlass/bfloat16.h"
#include <deep_gemm/cute_free/arch/ppu_tc02_copy.cuh>
#include <deep_gemm/cute_free/copy/copy_traits.cuh>

namespace deep_gemm {
namespace copy {

// =============================================================================
// TsmS2RAtom: Shared -> Register copy atom backed by tc02 ldmatrix.swzl.
//
// Loads one 16x16 MMA operand fragment (4 registers) from swizzled shared
// memory using hardware de-swizzle. When BLOCK_K exceeds one swizzle tile
// (bf16: > 64 elems / 128B) the per-stage tile is stored as `kInstNum`
// [CUBE_H x kSwzlW] sub-tiles along K (mirrors cutlass3 DefaultGemm_AIU_Operand
// + PPU0015_TSM_LD_SWZL_DISPATCH). This atom maps a k_step to the right
// sub-tile and computes the final tsm address.
//
//   Element    : data type (currently cutlass::bfloat16_t)
//   CUBE_H     : tile height (BLOCK_M for A, BLOCK_N for B) -- sub-tile stride
//   BLOCK_K    : tile K dimension
//   SwapLboSbo : false for the A operand, true for the B operand. For B the
//                lbo/sbo are swapped in hardware, which replaces the software
//                frag[1]<->frag[2] swap (Cutlass3 Swap=true approach).
// =============================================================================

template <typename Element, int CUBE_H, int BLOCK_K, bool SwapLboSbo>
struct TsmS2RAtom {
  static constexpr int kCubeH = CUBE_H;
  static constexpr int kBlockK = BLOCK_K;
  static constexpr bool kSwapLboSbo = SwapLboSbo;
  using Traits = CopyElementTraits<Element>;

  // Swizzle-tile width (cutlass AiuContByteSize capped at 128B).
  static constexpr int kElemBytes = Traits::kSize;
  static constexpr int kSwzlBytes = (BLOCK_K * kElemBytes > 128) ? 128 : BLOCK_K * kElemBytes;
  static constexpr int kSwzlW     = kSwzlBytes / kElemBytes;
  static constexpr int kInstNum   = BLOCK_K / kSwzlW;
  static constexpr int kMmaK      = 16;
  // Element offset (16B units) to jump one sub-tile within a stage.
  static constexpr int kSubTileStride16B = CUBE_H * kSwzlW * kElemBytes / 16;

  /// Load one fragment for the given K-step into `frag`.
  /// `tsm_add_base` addresses row coord_h inside sub-tile 0 of the stage
  /// (16B units); this atom adds the sub-tile and within-tile K offsets.
  static __device__ __forceinline__ void load(
      int tsm_add_base, int k_step, uint32_t (&frag)[4]) {
    static_assert(std::is_same<Element, cutlass::bfloat16_t>::value,
                  "TsmS2RAtom: only bf16 is supported by the arch layer today");
    int k_global      = k_step * kMmaK;
    int cube_in_stage = k_global / kSwzlW;
    int within        = k_global % kSwzlW;
    int tsm_add = tsm_add_base
                + cube_in_stage * kSubTileStride16B
                + within * kElemBytes / 16;
    arch::tsm_ldmatrix_swzl_bf16<kSwzlW, SwapLboSbo>(tsm_add, frag);
  }
};

} // namespace copy
} // namespace deep_gemm
