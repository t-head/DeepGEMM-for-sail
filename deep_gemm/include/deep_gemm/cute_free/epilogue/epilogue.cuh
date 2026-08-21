#pragma once

#include <cstdint>
#include "cutlass/bfloat16.h"
#include "epilogue_traits.cuh"

namespace deep_gemm {
namespace epilogue {

// =============================================================================
// Epilogue: writes a warp's MMA accumulator tile back to global memory.
//
// Two orthogonal axes:
//   - CLayoutT : the accumulator register->coordinate layout, a property of the
//                MMA instruction SHAPE (shared across element types). Provides
//                kMmaM/kMmaN, the thread decomposition and the half-tile offsets.
//   - Element  : the OUTPUT element type, resolved via EpilogueElementTraits to
//                its storage/packed types and the f32->Element cvt instruction.
//
// The register->coordinate MAPPING below (which d[i] lands on which row/col) is
// specific to the tc02 16x16 CLayout; a genuinely different MMA layout needs its
// own epilogue mapping.
//
//   CLayoutT  : accumulator layout (e.g. mma::Tc02Mma16x16x16CLayout)
//   Element   : output element type (currently cutlass::bfloat16_t)
//   WARP_M    : warp tile M (sets kMmasPerWarpM = WARP_M / kMmaM)
//   WARP_N    : warp tile N (sets kMmasPerWarpN = WARP_N / kMmaN)
//   IsAlignedN: whether N is a multiple of BLOCK_N (eliminates col checks)
// =============================================================================

template <typename Element, typename CLayoutT, int WARP_M, int WARP_N, bool IsAlignedN>
struct Epilogue {
  using CLayout = CLayoutT;
  using ElemTraits  = EpilogueElementTraits<Element>;
  using StorageType = typename ElemTraits::StorageType;
  using PackedType  = typename ElemTraits::PackedType;

  static constexpr int kMmaM          = CLayout::kMmaM;
  static constexpr int kMmaN          = CLayout::kMmaN;
  static constexpr int kHalfMmaM      = CLayout::kHalfMmaM;
  static constexpr int kHalfMmaN      = CLayout::kHalfMmaN;
  static constexpr int kThreadsCol    = CLayout::kThreadsCol;
  static constexpr int kColsPerThread = CLayout::kColsPerThread;
  static constexpr int kAccumPerThread = CLayout::kAccumPerThread;
  static constexpr int kVecWidth      = ElemTraits::kVecWidth;
  static constexpr int kMmasPerWarpM  = WARP_M / kMmaM;
  static constexpr int kMmasPerWarpN  = WARP_N / kMmaN;
  static constexpr bool kIsAlignedN   = IsAlignedN;

  /// Store one warp's accumulator tile to global memory D.
  ///   - m_offset/n_offset : tile origin in the output matrix
  ///   - warp_row/warp_col : warp position within the tile
  ///   - lane              : lane id (0..31)
  ///   - ptr_D / ldd       : output base pointer and leading dimension
  ///   - max_row / max_col : output extents (M, N) for boundary checks
  ///   - accum             : per-thread accumulator fragment
  static __device__ __forceinline__ void store(
      int m_offset, int n_offset, int warp_row, int warp_col, int lane,
      Element* ptr_D, int ldd, int max_row, int max_col,
      float const (&accum)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread]) {
    const int t_row = lane / kThreadsCol;  // 0..kThreadsRow-1
    const int t_col = lane % kThreadsCol;  // 0..kThreadsCol-1

    // tc02 16x16 accumulator CLayout (from CuTe PPU0015_16x16_Row CLayout).
    // Register mapping (kHalfMmaM = kMmaM/2, kHalfMmaN = kMmaN/2):
    //   d[0] -> C[t_row         ][t_col*kColsPerThread           ]
    //   d[1] -> C[t_row         ][t_col*kColsPerThread + 1       ]
    //   d[2] -> C[t_row+kHalfMmaM][t_col*kColsPerThread          ]
    //   d[3] -> C[t_row+kHalfMmaM][t_col*kColsPerThread + 1      ]
    //   d[4] -> C[t_row         ][t_col*kColsPerThread + kHalfMmaN]
    //   d[5] -> C[t_row         ][t_col*kColsPerThread + kHalfMmaN + 1]
    //   d[6] -> C[t_row+kHalfMmaM][t_col*kColsPerThread + kHalfMmaN]
    //   d[7] -> C[t_row+kHalfMmaM][t_col*kColsPerThread + kHalfMmaN + 1]
    // Adjacent pairs packed by ElemTraits::cvt_pack: (d0,d1),(d2,d3),(d4,d5),(d6,d7)
    // Pointer-increment mode: only one multiply for base_ptr, rest are additions.

    const int first_row = m_offset + warp_row * WARP_M + t_row;
    const int first_col = n_offset + warp_col * WARP_N + t_col * kColsPerThread;

    StorageType* base_ptr = reinterpret_cast<StorageType*>(ptr_D)
                            + int64_t(first_row) * ldd + first_col;

    const int row_half_stride = kHalfMmaM * ldd;  // row0 -> row(kHalfMmaM) offset (elements)
    const int mma_m_stride    = kMmaM * ldd;      // mma_m inter-row stride

    if constexpr (kIsAlignedN) {
      StorageType* mma_m_base = base_ptr;

      #pragma unroll
      for (int mma_m = 0; mma_m < kMmasPerWarpM; ++mma_m) {
        const int row0    = first_row + mma_m * kMmaM;
        const int rowHalf = row0 + kHalfMmaM;

        // row0 stores: d[0],d[1] and d[4],d[5]
        if (row0 < max_row) {
          StorageType* mma_n_ptr = mma_m_base;
          #pragma unroll
          for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
            *reinterpret_cast<PackedType*>(mma_n_ptr) =
                ElemTraits::cvt_pack(accum[mma_m][mma_n][1], accum[mma_m][mma_n][0]);
            *reinterpret_cast<PackedType*>(mma_n_ptr + kHalfMmaN) =
                ElemTraits::cvt_pack(accum[mma_m][mma_n][5], accum[mma_m][mma_n][4]);
            mma_n_ptr += kMmaN;
          }
        }

        // rowHalf stores: d[2],d[3] and d[6],d[7]
        if (rowHalf < max_row) {
          StorageType* mma_n_ptr = mma_m_base;
          #pragma unroll
          for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
            *reinterpret_cast<PackedType*>(mma_n_ptr + row_half_stride) =
                ElemTraits::cvt_pack(accum[mma_m][mma_n][3], accum[mma_m][mma_n][2]);
            *reinterpret_cast<PackedType*>(mma_n_ptr + row_half_stride + kHalfMmaN) =
                ElemTraits::cvt_pack(accum[mma_m][mma_n][7], accum[mma_m][mma_n][6]);
            mma_n_ptr += kMmaN;
          }
        }

        mma_m_base += mma_m_stride;
      }
    } else {
      bool col0_valid[kMmasPerWarpN];
      bool colHalf_valid[kMmasPerWarpN];
      #pragma unroll
      for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
        const int col0 = first_col + mma_n * kMmaN;
        col0_valid[mma_n]    = col0 + kVecWidth - 1 < max_col;
        colHalf_valid[mma_n] = col0 + kHalfMmaN + kVecWidth - 1 < max_col;
      }

      StorageType* mma_m_base = base_ptr;

      #pragma unroll
      for (int mma_m = 0; mma_m < kMmasPerWarpM; ++mma_m) {
        const int row0    = first_row + mma_m * kMmaM;
        const int rowHalf = row0 + kHalfMmaM;
        const bool row0_valid    = row0 < max_row;
        const bool rowHalf_valid = rowHalf < max_row;
        StorageType* mma_n_ptr = mma_m_base;

        if (row0_valid || rowHalf_valid) {
          #pragma unroll
          for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
            const bool c0 = col0_valid[mma_n];
            const bool ch = colHalf_valid[mma_n];

            // d[0],d[1] -> (row0, col0:col0+kVecWidth-1)  -- packed store
            if (row0_valid && c0) {
              *reinterpret_cast<PackedType*>(mma_n_ptr) =
                  ElemTraits::cvt_pack(accum[mma_m][mma_n][1], accum[mma_m][mma_n][0]);
            }

            // d[4],d[5] -> (row0, colHalf:colHalf+kVecWidth-1)
            if (row0_valid && ch) {
              *reinterpret_cast<PackedType*>(mma_n_ptr + kHalfMmaN) =
                  ElemTraits::cvt_pack(accum[mma_m][mma_n][5], accum[mma_m][mma_n][4]);
            }

            // d[2],d[3] -> (rowHalf, col0:col0+kVecWidth-1)
            if (rowHalf_valid && c0) {
              *reinterpret_cast<PackedType*>(mma_n_ptr + row_half_stride) =
                  ElemTraits::cvt_pack(accum[mma_m][mma_n][3], accum[mma_m][mma_n][2]);
            }

            // d[6],d[7] -> (rowHalf, colHalf:colHalf+kVecWidth-1)
            if (rowHalf_valid && ch) {
              *reinterpret_cast<PackedType*>(mma_n_ptr + row_half_stride + kHalfMmaN) =
                  ElemTraits::cvt_pack(accum[mma_m][mma_n][7], accum[mma_m][mma_n][6]);
            }

            mma_n_ptr += kMmaN;  // column increment (kMmaN elements)
          }
        }
        mma_m_base += mma_m_stride;  // row increment
      }
    }
  }
};

} // namespace epilogue
} // namespace deep_gemm
