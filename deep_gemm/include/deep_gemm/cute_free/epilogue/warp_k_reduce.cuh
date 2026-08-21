#pragma once

#include <cstdint>

namespace deep_gemm {
namespace epilogue {

// =============================================================================
// WarpKReduce: cross-warp reduction of the K-split partial accumulators.
//
// With WarpOnK > 1 each warp group [warp_k] holds a PARTIAL sum over its own K
// slice. This reduces them into warp_k == 0, which then runs the epilogue.
//
// Strategy: single write + single read pass (2 barriers), i.e. the equivalent of
// cutlass3's "Strategy A" in warp_on_k_reduction.cuh. All warp_k > 0 groups write
// to disjoint SMEM regions simultaneously, then warp_k == 0 reads and adds them.
// A static_assert guarantees the region fits; we deliberately do NOT fall back to
// the sequential/chunked strategies -- a config that does not fit fails to compile
// instead of silently getting slower.
//
// SMEM is the mainloop's A/B buffer, reused after the mainloop is done. The caller
// MUST have drained the async copies and executed a barrier before calling this
// (Mainloop::run() ends with cp_async_wait<0>() + __syncthreads(), which
// satisfies both).
//
// Layout: lane-interleaved within a warp region so consecutive lanes touch
// consecutive floats (conflict-free), rather than giving each lane a contiguous
// block (which would make all lanes hit the same banks).
//
//   region(warp_k, warp_mn) = ((warp_k - 1) * kWarpsMN + warp_mn) * 32 * kAccumTotal
//   slot(elem, lane)        = elem * 32 + lane
// =============================================================================

template <int kWarpsK, int kWarpsMN,
          int kMmasPerWarpM, int kMmasPerWarpN, int kAccumPerThread,
          int kSmemFloatCapacity>
struct WarpKReduce {
  static constexpr int kAccumTotal = kMmasPerWarpM * kMmasPerWarpN * kAccumPerThread;
  static constexpr int kRegionFloats = kWarpsMN * 32 * kAccumTotal;
  static constexpr int kFloatsNeeded = (kWarpsK > 1) ? (kWarpsK - 1) * kRegionFloats : 0;

  static_assert(kFloatsNeeded <= kSmemFloatCapacity,
                "WarpKReduce: mainloop SMEM cannot hold all K partials at once. "
                "Reduce WarpOnK, or implement the sequential/chunked strategy.");

  /// Reduce `accum` across warp_k into warp_k == 0. No-op when kWarpsK == 1.
  static __device__ __forceinline__ void run(
      float (&accum)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      float* smem, int warp_mn, int warp_k, int lane) {
    if constexpr (kWarpsK <= 1) {
      return;
    } else {
      if (warp_k > 0) {
        float* dst = smem + (warp_k - 1) * kRegionFloats + warp_mn * 32 * kAccumTotal + lane;
        int e = 0;
        #pragma unroll
        for (int mma_m = 0; mma_m < kMmasPerWarpM; ++mma_m) {
          #pragma unroll
          for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
            #pragma unroll
            for (int i = 0; i < kAccumPerThread; ++i, ++e) {
              dst[e * 32] = accum[mma_m][mma_n][i];
            }
          }
        }
      }
      __syncthreads();

      if (warp_k == 0) {
        #pragma unroll
        for (int k = 1; k < kWarpsK; ++k) {
          float const* src = smem + (k - 1) * kRegionFloats + warp_mn * 32 * kAccumTotal + lane;
          int e = 0;
          #pragma unroll
          for (int mma_m = 0; mma_m < kMmasPerWarpM; ++mma_m) {
            #pragma unroll
            for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
              #pragma unroll
              for (int i = 0; i < kAccumPerThread; ++i, ++e) {
                accum[mma_m][mma_n][i] += src[e * 32];
              }
            }
          }
        }
      }
      // Barrier before the caller's epilogue: warp_k>0 groups must not race ahead
      // into the next scheduler tile and overwrite SMEM while warp_k==0 still reads.
      __syncthreads();
    }
  }
};

} // namespace epilogue
} // namespace deep_gemm
