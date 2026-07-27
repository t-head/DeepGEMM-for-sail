/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * WarpOnK cross-warp reduction for DeepGemm kernels.
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////
// Compile-time reduction strategy selection based on tile parameters
///////////////////////////////////////////////////////////////////////////////

template <int BlockM_, int BlockN_, int BlockK_, int WarpM_, int WarpN_, int WarpK_, int Stages_, int ElemABytes_>
struct WarpOnKReductionPolicy {
  static constexpr int BlockM = BlockM_;
  static constexpr int BlockN = BlockN_;
  static constexpr int BlockK = BlockK_;
  static constexpr int WarpM  = WarpM_;
  static constexpr int WarpN  = WarpN_;
  static constexpr int WarpK  = WarpK_;
  static constexpr int Stages = Stages_;

  static constexpr int WarpOnK  = BlockK / WarpK;
  static constexpr int WarpOnM  = BlockM / WarpM;
  static constexpr int WarpOnN  = BlockN / WarpN;
  static constexpr int WarpOnMN = WarpOnM * WarpOnN;

  // Per-thread accumulator count (FP32)
  // Each thread in a 32-thread warp owns WarpM*WarpN/32 accumulator elements
  static constexpr int AccumPerThread = WarpM * WarpN / 32;
  static constexpr int TsmCopyNum = AccumPerThread / 4; // copy b32x4 once

  // Available SMEM bytes = mainloop A+B buffer (all stages, reusable after mainloop)
  static constexpr int AvailSmemBytes = (BlockM + BlockN) * BlockK * Stages * ElemABytes_;

  // SMEM bytes needed for ONE full k-copy (all M,N warps write simultaneously)
  // = WarpOnMN * 32 threads * AccumPerThread * sizeof(float)
  // = BlockM * BlockN * sizeof(float)
  static constexpr int BytesPerFullCopy = WarpOnMN * WarpM * WarpN * int(sizeof(float));

  // How many full k-copies fit in available SMEM
  static constexpr int FullCopiesThatFit = AvailSmemBytes / BytesPerFullCopy;

  // Helper: compile-time log2 for powers of 2
  static constexpr int Log2WarpOnK = (WarpOnK == 1) ? 0 : (WarpOnK == 2) ? 1 : (WarpOnK == 4) ? 2 : (WarpOnK == 8) ? 3 : (WarpOnK == 16) ? 4 : 0;

  // Is WarpOnK a power of 2 and >= 4?
  static constexpr bool IsPow2WarpOnK = (WarpOnK >= 4) && ((WarpOnK & (WarpOnK - 1)) == 0);

  // --- Strategy Selection (priority: A > D > B > C) ---
  // Strategy A: All (WarpOnK-1) copies fit at once -> single write+read pass (2 syncs)
  // Strategy D: Binary tree reduction -> 2*log2(WarpOnK) syncs (needs WarpOnK/2 copies)
  // Strategy B: One full copy fits -> sequential k-warp reduction (2*(WarpOnK-1) syncs)
  // Strategy C: Not even one full copy fits -> chunked reduction (tiled accumulator)
  static constexpr bool UseStrategyA = (FullCopiesThatFit >= (WarpOnK - 1));
  static constexpr bool UseStrategyD = (!UseStrategyA) && IsPow2WarpOnK && (FullCopiesThatFit >= (WarpOnK / 2));
  static constexpr bool UseStrategyB = (!UseStrategyA) && (!UseStrategyD) && (FullCopiesThatFit >= 1);
  static constexpr bool UseStrategyC = (!UseStrategyA) && (!UseStrategyD) && (FullCopiesThatFit < 1);

  // For Strategy C: chunk the accumulator elements per round
  static constexpr int ChunkPerThread = AvailSmemBytes / (WarpOnMN * 32 * int(sizeof(float)));
  static constexpr int NumChunks = (AccumPerThread + ChunkPerThread - 1) / ChunkPerThread;
};

///////////////////////////////////////////////////////////////////////////////
// Device-side K-reduction function
///////////////////////////////////////////////////////////////////////////////

template <typename Policy, typename AccumTensor>
CUTLASS_DEVICE void warp_on_k_reduce(
    AccumTensor& accumulators,
    float* smem_buf,
    int warp_idx,
    int lane_idx)
{
  constexpr int WarpOnK        = Policy::WarpOnK;
  constexpr int WarpOnMN       = Policy::WarpOnMN;
  constexpr int AccumPerThread = Policy::AccumPerThread;
  constexpr int TsmCopyNum = Policy::TsmCopyNum;

  if constexpr (WarpOnK <= 1) return;  // no-op when WarpOnK == 1

  int warp_idx_k  = warp_idx / WarpOnMN;
  int warp_idx_mn = warp_idx % WarpOnMN;

  if constexpr (Policy::UseStrategyA) {
    // --- Strategy A: All partials fit at once ---
    // k>0 warps write simultaneously to distinct SMEM regions
    if (warp_idx_k > 0) {
      // All offsets below are in float elements (not bytes).
      // base_offset: per-warp region (32 lanes * AccumPerThread elems) + per-lane slot (4 elems)
      int base_offset = ((warp_idx_k - 1) * WarpOnMN + warp_idx_mn) * 32 * AccumPerThread
                 + lane_idx * 4;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < TsmCopyNum; j++) {
        // (j << 7) = j * 128 elems = j * 32_lanes * 4_elems_per_lane
        int offset = base_offset + (j << 7);
        // offset_src: index into accumulator register fragment (4 elems per batch)
        int offset_src = j << 2;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) {
          smem_buf[offset + i] = accumulators(i + offset_src);
        }
      }
    }
    __syncthreads();

    // k=0 warps read all partials and accumulate
    if (warp_idx_k == 0) {
      CUTLASS_PRAGMA_UNROLL
      for (int k = 1; k < WarpOnK; ++k) {
        // All offsets in float elements (same layout as the write path above)
        int base_offset = ((k - 1) * WarpOnMN + warp_idx_mn) * 32 * AccumPerThread
                   + lane_idx * 4;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < TsmCopyNum; j++) {
          // (j << 7) = j * 128 elems = j * 32_lanes * 4_elems_per_lane
          int offset = base_offset + (j << 7);
          // offset_src: index into accumulator register fragment
          int offset_src = j << 2;
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < 4; ++i) {
            accumulators(i + offset_src) += smem_buf[offset + i];
          }
        }
      }
    }
    __syncthreads();

  } else if constexpr (Policy::UseStrategyD) {
    // --- Strategy D: Binary tree reduction ---
    // 2*log2(WarpOnK) syncs instead of 2*(WarpOnK-1) for Strategy B
    constexpr int NumRounds = Policy::Log2WarpOnK;

    CUTLASS_PRAGMA_NO_UNROLL
    for (int r = 0; r < NumRounds; ++r) {
      int stride = 1 << r;
      int group_size = stride << 1;  // stride * 2

      // Determine role for this round
      int pos_in_group = warp_idx_k % group_size;
      bool is_writer = (pos_in_group == stride);
      bool is_reader = (pos_in_group == 0);

      // Writer's slot index (for SMEM addressing)
      int slot = warp_idx_k / group_size;

      // Write phase
      if (is_writer) {
        int offset = slot * WarpOnMN * 32 * AccumPerThread
                   + warp_idx_mn * 32 * AccumPerThread
                   + lane_idx * AccumPerThread;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < AccumPerThread; ++i) {
          smem_buf[offset + i] = accumulators(i);
        }
      }
      __syncthreads();

      // Read phase
      if (is_reader) {
        int offset = slot * WarpOnMN * 32 * AccumPerThread
                   + warp_idx_mn * 32 * AccumPerThread
                   + lane_idx * AccumPerThread;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < AccumPerThread; ++i) {
          accumulators(i) += smem_buf[offset + i];
        }
      }
      __syncthreads();
    }

  } else if constexpr (Policy::UseStrategyB) {
    // --- Strategy B: Sequential, one k-warp at a time ---
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k = 1; k < WarpOnK; ++k) {
      if (warp_idx_k == k) {
        int offset = warp_idx_mn * 32 * AccumPerThread + lane_idx * AccumPerThread;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < AccumPerThread; ++i) {
          smem_buf[offset + i] = accumulators(i);
        }
      }
      __syncthreads();

      if (warp_idx_k == 0) {
        int offset = warp_idx_mn * 32 * AccumPerThread + lane_idx * AccumPerThread;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < AccumPerThread; ++i) {
          accumulators(i) += smem_buf[offset + i];
        }
      }
      __syncthreads();
    }

  } else {
    // --- Strategy C: Chunked reduction ---
    constexpr int ChunkSize = Policy::ChunkPerThread;
    constexpr int NumChunks = Policy::NumChunks;

    CUTLASS_PRAGMA_NO_UNROLL
    for (int k = 1; k < WarpOnK; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < NumChunks; ++c) {
        int elem_start = c * ChunkSize;
        int elem_count = (elem_start + ChunkSize <= AccumPerThread)
                       ? ChunkSize
                       : (AccumPerThread - elem_start);

        if (warp_idx_k == k) {
          int offset = warp_idx_mn * 32 * ChunkSize + lane_idx * ChunkSize;
          for (int i = 0; i < elem_count; ++i) {
            smem_buf[offset + i] = accumulators(elem_start + i);
          }
        }
        __syncthreads();

        if (warp_idx_k == 0) {
          int offset = warp_idx_mn * 32 * ChunkSize + lane_idx * ChunkSize;
          for (int i = 0; i < elem_count; ++i) {
            accumulators(elem_start + i) += smem_buf[offset + i];
          }
        }
        __syncthreads();
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
