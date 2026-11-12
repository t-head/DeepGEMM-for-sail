#pragma once

#include <cstdint>

#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory_ppu.h>
#include <deep_gemm/cute_free/arch/ppu_common.cuh>
#include <deep_gemm/common/math.cuh>

#ifndef DG_DUAL_B_FULL_K_PREFETCH
#define DG_DUAL_B_FULL_K_PREFETCH 0
#endif
#ifndef DG_DUAL_B_EARLY_G2S
#define DG_DUAL_B_EARLY_G2S 0
#endif
#ifndef DG_DUAL_B_MMA_S2R_OVERLAP
#define DG_DUAL_B_MMA_S2R_OVERLAP 0
#endif
#ifndef DG_DUAL_B_BOUNDED_KLOOP
#define DG_DUAL_B_BOUNDED_KLOOP 0
#endif
#ifndef DG_DUAL_B_KLOOP_UNROLL
#define DG_DUAL_B_KLOOP_UNROLL 1
#endif
#ifndef DG_DUAL_B_KLOOP_CHUNK
#define DG_DUAL_B_KLOOP_CHUNK 0
#endif
#ifndef DG_DUAL_B_STATIC_STAGES
#define DG_DUAL_B_STATIC_STAGES 0
#endif
#ifndef DG_DUAL_B_STREAMED_LOW
#define DG_DUAL_B_STREAMED_LOW 0
#endif
#ifndef DG_DUAL_B_STREAMED_PAIRS
#define DG_DUAL_B_STREAMED_PAIRS 0
#endif
#ifndef DG_DUAL_B_PAIRS_M_FIRST
#define DG_DUAL_B_PAIRS_M_FIRST 0
#endif

namespace deep_gemm {
namespace mainloop {

// =============================================================================
// DualBMainloop: K-pipeline carrying two B operands (W_high / W_low) per stage.
//
// Variant of Mainloop for the BF16 x FP32 dual-B decomposition
// (kernel::GemmBf16xFp32Kernel). Instead of running the single-B pipeline twice
// per output tile -- which streams A twice from global memory and pays the
// stage warm-up / drain bubbles twice -- every pipeline stage holds
// A + B_high + B_low together, keeping the high/low pair resident in shared
// memory per stage:
//
//   - G2S loads all three tiles per stage: A is read from gmem ONCE per tile
//   - each k_step S2R-loads one A fragment, reused by two back-to-back MMAs
//     into two independent FP32 accumulators (high / low)
//   - one prologue fill and one drain per output tile
//
// The wait/G2S schedule is isomorphic to Mainloop: adding a third load inside
// an existing cp.async group does not change the pending-group counts the
// wait<STAGES-2> schedule relies on. SMEM footprint grows by
// STAGES * BLOCK_N * BLOCK_K elements; the host launcher must size shared
// memory from STAGES * (BLOCK_M + 2 * BLOCK_N) * BLOCK_K.
//
// Only the k_pair unroll shape (kUseKPairUnroll = true in Mainloop, the
// shape actually shipped) is implemented here.
//
// Template parameters: identical to Mainloop minus OverlapPrologue (the
// dual-B kernel never overlaps the next tile's prologue).
// =============================================================================
template <typename Element,
          typename G2SAtomA, typename G2SAtomB,
          typename S2RAtomA, typename S2RAtomB,
          typename MmaAtom,
          int BLOCK_M, int BLOCK_N, int BLOCK_K,
          int WARP_M, int WARP_N, int WARP_K, int STAGES,
          bool DenseS2Opt>
struct DualBMainloop {
  // ---- Constants derived from the MMA atom + tile config (same as Mainloop) ----
  static constexpr int kMmaM = MmaAtom::kMmaM;
  static constexpr int kMmaN = MmaAtom::kMmaN;
  static constexpr int kMmaK = MmaAtom::kMmaK;
  static constexpr int kAccumPerThread = MmaAtom::kAccumPerThread;

  static constexpr int kMmasPerWarpM = WARP_M / kMmaM;
  static constexpr int kMmasPerWarpN = WARP_N / kMmaN;

  // ---- WarpOnK: split BLOCK_K across kWarpsK warp groups ----
  static constexpr int kWarpsK = BLOCK_K / WARP_K;
  static constexpr int kKStepsPerWarp = WARP_K / kMmaK;
  static constexpr int kKSteps = BLOCK_K / kMmaK;

  // Swizzle-tile width along K (see Mainloop).
  static constexpr int kElemBytes = (int)sizeof(Element);
  static constexpr int kSwzlBytes = (BLOCK_K * kElemBytes > 128) ? 128 : BLOCK_K * kElemBytes;
  static constexpr int kSwzlW = kSwzlBytes / kElemBytes;

  static_assert(STAGES >= 2, "STAGES must be >= 2 for the async pipeline");
  static_assert(BLOCK_K % WARP_K == 0, "BLOCK_K must be a multiple of WARP_K");
  static_assert(WARP_K % kMmaK == 0, "WARP_K must be a multiple of kMmaK");
  static_assert(WARP_K % kSwzlW == 0, "WARP_K must be a multiple of the swizzle-tile width");
  static_assert(kKStepsPerWarp >= 2, "WARP_K / kMmaK must be >= 2 for G2S interleaving");
  static_assert(kKStepsPerWarp % 2 == 0,
                "WARP_K / kMmaK must be EVEN: the k_pair unroll relies on steps "
                "pairing up (see Mainloop)");

  // K offset (in 16B tsm units) that moves a base address by one warp-K slice.
  static constexpr int kWarpKSubTiles  = WARP_K / kSwzlW;
  static constexpr int kWarpKOffsetA   = kWarpKSubTiles * S2RAtomA::kSubTileStride16B;
  static constexpr int kWarpKOffsetB   = kWarpKSubTiles * S2RAtomB::kSubTileStride16B;

  // ---- SMEM stage layout (elements), mirrored by the kernel's SharedStorage:
  //   [ A: STAGES * BLOCK_M * BLOCK_K | B_high: STAGES * BLOCK_N * BLOCK_K
  //   | B_low: STAGES * BLOCK_N * BLOCK_K ]
  static constexpr int kSmemAElements     = STAGES * BLOCK_M * BLOCK_K;
  static constexpr int kSmemBHighElements = kSmemAElements + STAGES * BLOCK_N * BLOCK_K;

  // A + dual-B parameters only (decoupled from the kernel's full Params).
  struct Params {
    Element const* ptr_A;
    Element const* ptr_B_high;
    Element const* ptr_B_low;
    int M, N, K;
    int lda, ldb;
  };

  static constexpr int kKPairs = kKStepsPerWarp / 2;

  // A single non-boundary pair has no stage operations; explicit chunks and legacy bounded loops reuse the same dual accumulation chains.
  static __device__ __forceinline__ void run_one_inner_k_pair(
      int pair,
      int const (&base_a)[kMmasPerWarpM],
      int const (&base_bh)[kMmasPerWarpN],
      int const (&base_bl)[kMmasPerWarpN],
      float (&accum_high)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      float (&accum_low)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      uint32_t (&a_reg)[2][kMmasPerWarpM][4],
      uint32_t (&bh_reg)[2][kMmasPerWarpN][4],
      uint32_t (&bl_reg)[2][kMmasPerWarpN][4]) {
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int current = half;
      const int next = 1 - half;
      const int next_step = 2 * pair + half + 1;
      #pragma unroll
      for (int mi = 0; mi < kMmasPerWarpM; ++mi)
        S2RAtomA::load(base_a[mi], next_step, a_reg[next][mi]);
      #pragma unroll
      for (int ni = 0; ni < kMmasPerWarpN; ++ni)
        S2RAtomB::load(base_bh[ni], next_step, bh_reg[next][ni]);
      #pragma unroll
      for (int ni = 0; ni < kMmasPerWarpN; ++ni)
        S2RAtomB::load(base_bl[ni], next_step, bl_reg[next][ni]);
      #pragma unroll
      for (int mi = 0; mi < kMmasPerWarpM; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < kMmasPerWarpN; ++ni) {
          MmaAtom::mma(accum_high[mi][ni], a_reg[current][mi], bh_reg[current][ni], accum_high[mi][ni]);
          MmaAtom::mma(accum_low[mi][ni], a_reg[current][mi], bl_reg[current][ni], accum_low[mi][ni]);
        }
      }
    }
  }

  // Loop only over non-boundary K pairs; the final pair uses the original template, preserving stage sync points and full-K order.
  static __device__ __forceinline__ void run_inner_k_pairs_bounded(
      int const (&base_a)[kMmasPerWarpM],
      int const (&base_bh)[kMmasPerWarpN],
      int const (&base_bl)[kMmasPerWarpN],
      float (&accum_high)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      float (&accum_low)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      uint32_t (&a_reg)[2][kMmasPerWarpM][4],
      uint32_t (&bh_reg)[2][kMmasPerWarpN][4],
      uint32_t (&bl_reg)[2][kMmasPerWarpN][4]) {
    static_assert(DG_DUAL_B_KLOOP_UNROLL == 1 || DG_DUAL_B_KLOOP_UNROLL == 2 || DG_DUAL_B_KLOOP_UNROLL == 4,
                  "bounded K loop supports only the frozen partial unroll factors");
    if constexpr (DG_DUAL_B_KLOOP_CHUNK) {
      static_assert(!DG_DUAL_B_KLOOP_CHUNK || DG_DUAL_B_KLOOP_UNROLL == 2 || DG_DUAL_B_KLOOP_UNROLL == 4,
                    "explicit chunks support only two or four inner pairs");
      constexpr int kChunk = DG_DUAL_B_KLOOP_UNROLL;
      constexpr int kFullPairs = ((kKPairs - 1) / kChunk) * kChunk;
      #pragma unroll 1
      for (int base = 0; base < kFullPairs; base += kChunk) {
        #pragma unroll
        for (int offset = 0; offset < kChunk; ++offset)
          run_one_inner_k_pair(base + offset, base_a, base_bh, base_bl,
                               accum_high, accum_low, a_reg, bh_reg, bl_reg);
      }
      // Specialize the tail at compile time without repeating the final stage-boundary pair; carry buffers over from the previous chunk.
      #pragma unroll
      for (int pair = kFullPairs; pair < kKPairs - 1; ++pair)
        run_one_inner_k_pair(pair, base_a, base_bh, base_bl,
                             accum_high, accum_low, a_reg, bh_reg, bl_reg);
    } else {
      #if DG_DUAL_B_KLOOP_UNROLL == 4
      #pragma unroll 4
      #elif DG_DUAL_B_KLOOP_UNROLL == 2
      #pragma unroll 2
      #else
      #pragma unroll 1
      #endif
      for (int pair = 0; pair < kKPairs - 1; ++pair)
        run_one_inner_k_pair(pair, base_a, base_bh, base_bl,
                             accum_high, accum_low, a_reg, bh_reg, bl_reg);
    }
  }

  /// Issue the S2R loads for one k_step into the given fragment buffers,
  /// grouped by operand: all A fragments, then all B_high, then all B_low.
  /// SrcKStep is the source k_step inside the tile.
  template <int SrcKStep>
  static __device__ __forceinline__ void issue_s2r_triple(
      int const (&tsm_add_base_a)[kMmasPerWarpM],
      int const (&tsm_add_base_bh)[kMmasPerWarpN],
      int const (&tsm_add_base_bl)[kMmasPerWarpN],
      uint32_t (&a_dst)[kMmasPerWarpM][4],
      uint32_t (&bh_dst)[kMmasPerWarpN][4],
      uint32_t (&bl_dst)[kMmasPerWarpN][4]) {
    #pragma unroll
    for (int mma_m = 0; mma_m < kMmasPerWarpM; ++mma_m) {
      S2RAtomA::load(tsm_add_base_a[mma_m], SrcKStep, a_dst[mma_m]);
    }
    #pragma unroll
    for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
      S2RAtomB::load(tsm_add_base_bh[mma_n], SrcKStep, bh_dst[mma_n]);
    }
    #pragma unroll
    for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
      S2RAtomB::load(tsm_add_base_bl[mma_n], SrcKStep, bl_dst[mma_n]);
    }
  }

  /// G2S issue point (same placement / wait schedule as Mainloop): loads the
  /// next K tile's A + B_high + B_low into the recycled stage buffers.
  static __device__ __forceinline__ void issue_g2s_and_wait(
      int warp_id,
      bool issue_g2s,
      Element* g2s_smem_a,
      Element* g2s_smem_b_high,
      Element* g2s_smem_b_low,
      Params const& params,
      int m_offset, int n_offset, int k_offset_next) {
    if constexpr (DenseS2Opt) {
      if constexpr (STAGES > 2) {
        cutlass::arch::cp_async_wait<STAGES - 2>();
        __syncthreads();
      } else {
        __syncthreads();
      }
    } else {
      cutlass::arch::cp_async_wait<STAGES - 2>();
      __syncthreads();
    }
    if (issue_g2s) {
      if (warp_id == 0) {
        G2SAtomA::load(g2s_smem_a, params.ptr_A, params.M, params.lda, m_offset, k_offset_next);
        G2SAtomB::load(g2s_smem_b_high, params.ptr_B_high, params.N, params.ldb, n_offset, k_offset_next);
        G2SAtomB::load(g2s_smem_b_low, params.ptr_B_low, params.N, params.ldb, n_offset, k_offset_next);
      }
    }
    // Fence stays UNCONDITIONAL for the same FIFO-decay reason as Mainloop.
    cutlass::arch::cp_async_fence();
    if constexpr (DenseS2Opt && STAGES == 2) {
      cutlass::arch::cp_async_wait<1>();
      __syncthreads();
    }
  }

  template <int KStep>
  static __device__ __forceinline__ void load_next_fragments(
      char* smem,
      int next_stage,
      int warp_coord_a,
      int warp_coord_b,
      int const (&tsm_add_base_a)[kMmasPerWarpM],
      int const (&tsm_add_base_bh)[kMmasPerWarpN],
      int const (&tsm_add_base_bl)[kMmasPerWarpN],
      uint32_t (&a_next)[kMmasPerWarpM][4],
      uint32_t (&bh_next)[kMmasPerWarpN][4],
      uint32_t (&bl_next)[kMmasPerWarpN][4]) {
    static constexpr int kCoordScale = kSwzlW * (int)sizeof(Element) / 16;

    if constexpr (KStep < kKStepsPerWarp - 1) {
      issue_s2r_triple<KStep + 1>(tsm_add_base_a, tsm_add_base_bh, tsm_add_base_bl,
                                  a_next, bh_next, bl_next);
    } else {
      // Cross-tile prefetch: load step 0 from the NEXT stage.
      Element const* smem_elems = reinterpret_cast<Element const*>(smem);
      Element const* next_stage_a  = smem_elems + next_stage * BLOCK_M * BLOCK_K;
      Element const* next_stage_bh = smem_elems + kSmemAElements + next_stage * BLOCK_N * BLOCK_K;
      Element const* next_stage_bl = smem_elems + kSmemBHighElements + next_stage * BLOCK_N * BLOCK_K;
      const int next_base_a = arch::smem_ptr_to_tsm_base(next_stage_a, warp_coord_a);
      const int next_base_bh = arch::smem_ptr_to_tsm_base(next_stage_bh, warp_coord_b);
      const int next_base_bl = arch::smem_ptr_to_tsm_base(next_stage_bl, warp_coord_b);
      #pragma unroll
      for (int mma_m = 0; mma_m < kMmasPerWarpM; ++mma_m) {
        S2RAtomA::load(next_base_a + mma_m * kMmaM * kCoordScale, 0, a_next[mma_m]);
      }
      #pragma unroll
      for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
        S2RAtomB::load(next_base_bh + mma_n * kMmaN * kCoordScale, 0, bh_next[mma_n]);
      }
      #pragma unroll
      for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
        S2RAtomB::load(next_base_bl + mma_n * kMmaN * kCoordScale, 0, bl_next[mma_n]);
      }
    }
  }

  /// Execute one k_pair = 2 consecutive k_steps; per step, the SAME A fragment
  /// feeds two MMAs: (A, B_high) -> accum_high, then (A, B_low) -> accum_low.
  ///
  /// Pipeline layout per pair (isomorphic to Mainloop::do_k_pair):
  ///   S2R prefetch step 2*KPair+1 into buf_next
  ///   MMA step 2*KPair (buf_curr) x2 (high, low)
  ///   S2R prefetch step 2*KPair+2 into buf_curr
  ///     ↑ at the last pair this becomes the cross-tile prefetch
  ///   MMA step 2*KPair+1 (buf_next) x2 (high, low)
  ///   [G2S at KPair == kKPairs-1: wait + sync + issue + fence]
  template <int KPair>
  static __device__ __forceinline__ void do_k_pair(
      int warp_id,
      bool issue_g2s,
      Element* g2s_smem_a,
      Element* g2s_smem_b_high,
      Element* g2s_smem_b_low,
      Params const& params,
      int m_offset, int n_offset, int k_offset_next,
      char* smem,
      int next_stage,
      int warp_coord_a,
      int warp_coord_b,
      int const (&tsm_add_base_a)[kMmasPerWarpM],
      int const (&tsm_add_base_bh)[kMmasPerWarpN],
      int const (&tsm_add_base_bl)[kMmasPerWarpN],
      float (&accum_high)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      float (&accum_low)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      uint32_t (&a_buf0)[kMmasPerWarpM][4],
      uint32_t (&bh_buf0)[kMmasPerWarpN][4],
      uint32_t (&bl_buf0)[kMmasPerWarpN][4],
      uint32_t (&a_buf1)[kMmasPerWarpM][4],
      uint32_t (&bh_buf1)[kMmasPerWarpN][4],
      uint32_t (&bl_buf1)[kMmasPerWarpN][4]) {
    static constexpr int S0 = 2 * KPair;      // first step of this pair
    static constexpr int S1 = 2 * KPair + 1;  // second step of this pair

    // --- Step S0: prefetch S1's data into buf1, then two MMAs with buf0 ---
    if constexpr (!DG_DUAL_B_MMA_S2R_OVERLAP) {
      load_next_fragments<S0>(smem, next_stage, warp_coord_a, warp_coord_b,
                              tsm_add_base_a, tsm_add_base_bh, tsm_add_base_bl,
                              a_buf1, bh_buf1, bl_buf1);
    }
    // Place the next S2R between the current high/low MMAs; preserve each chain's K order and all synchronization points.
    if constexpr (DG_DUAL_B_MMA_S2R_OVERLAP) {
      #pragma unroll
      for (int mma_m = 0; mma_m < kMmasPerWarpM; ++mma_m) {
        #pragma unroll
        for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
          MmaAtom::mma(accum_high[mma_m][mma_n], a_buf0[mma_m], bh_buf0[mma_n], accum_high[mma_m][mma_n]);
        }
      }
      load_next_fragments<S0>(smem, next_stage, warp_coord_a, warp_coord_b,
                              tsm_add_base_a, tsm_add_base_bh, tsm_add_base_bl,
                              a_buf1, bh_buf1, bl_buf1);
    }
    if constexpr (DG_DUAL_B_EARLY_G2S && KPair == kKPairs - 1) {
      // All three operands for this stage's final two K steps are in registers; the original barrier ensures all warps
      // finish reading before stage reuse. Only issue copies earlier; preserve full-K accumulation order and the synchronization count.
      issue_g2s_and_wait(warp_id, issue_g2s, g2s_smem_a, g2s_smem_b_high, g2s_smem_b_low,
                         params, m_offset, n_offset, k_offset_next);
    }
    #pragma unroll
    for (int mma_m = 0; mma_m < kMmasPerWarpM; ++mma_m) {
      #pragma unroll
      for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
        if constexpr (!DG_DUAL_B_MMA_S2R_OVERLAP) {
          MmaAtom::mma(accum_high[mma_m][mma_n], a_buf0[mma_m], bh_buf0[mma_n], accum_high[mma_m][mma_n]);
        }
        MmaAtom::mma(accum_low[mma_m][mma_n], a_buf0[mma_m], bl_buf0[mma_n], accum_low[mma_m][mma_n]);
      }
    }
    if constexpr (!DG_DUAL_B_EARLY_G2S && KPair == kKPairs - 1) {
      issue_g2s_and_wait(warp_id, issue_g2s, g2s_smem_a, g2s_smem_b_high, g2s_smem_b_low,
                         params, m_offset, n_offset, k_offset_next);
    }

    // --- Step S1: prefetch (S1+1)'s data into buf0, then two MMAs with buf1 ---
    if constexpr (!DG_DUAL_B_MMA_S2R_OVERLAP) {
      load_next_fragments<S1>(smem, next_stage, warp_coord_a, warp_coord_b,
                              tsm_add_base_a, tsm_add_base_bh, tsm_add_base_bl,
                              a_buf0, bh_buf0, bl_buf0);
    }
    if constexpr (DG_DUAL_B_MMA_S2R_OVERLAP) {
      #pragma unroll
      for (int mma_m = 0; mma_m < kMmasPerWarpM; ++mma_m) {
        #pragma unroll
        for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
          MmaAtom::mma(accum_high[mma_m][mma_n], a_buf1[mma_m], bh_buf1[mma_n], accum_high[mma_m][mma_n]);
        }
      }
      load_next_fragments<S1>(smem, next_stage, warp_coord_a, warp_coord_b,
                              tsm_add_base_a, tsm_add_base_bh, tsm_add_base_bl,
                              a_buf0, bh_buf0, bl_buf0);
    }
    #pragma unroll
    for (int mma_m = 0; mma_m < kMmasPerWarpM; ++mma_m) {
      #pragma unroll
      for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
        if constexpr (!DG_DUAL_B_MMA_S2R_OVERLAP) {
          MmaAtom::mma(accum_high[mma_m][mma_n], a_buf1[mma_m], bh_buf1[mma_n], accum_high[mma_m][mma_n]);
        }
        MmaAtom::mma(accum_low[mma_m][mma_n], a_buf1[mma_m], bl_buf1[mma_n], accum_low[mma_m][mma_n]);
      }
    }
  }

  template <int KPair>
  static __device__ __forceinline__ void run_k_pairs(
      int warp_id,
      bool issue_g2s,
      Element* g2s_smem_a,
      Element* g2s_smem_b_high,
      Element* g2s_smem_b_low,
      Params const& params,
      int m_offset, int n_offset, int k_offset_next,
      char* smem,
      int next_stage,
      int warp_coord_a,
      int warp_coord_b,
      int const (&tsm_add_base_a)[kMmasPerWarpM],
      int const (&tsm_add_base_bh)[kMmasPerWarpN],
      int const (&tsm_add_base_bl)[kMmasPerWarpN],
      float (&accum_high)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      float (&accum_low)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      uint32_t (&a_buf0)[kMmasPerWarpM][4],
      uint32_t (&bh_buf0)[kMmasPerWarpN][4],
      uint32_t (&bl_buf0)[kMmasPerWarpN][4],
      uint32_t (&a_buf1)[kMmasPerWarpM][4],
      uint32_t (&bh_buf1)[kMmasPerWarpN][4],
      uint32_t (&bl_buf1)[kMmasPerWarpN][4]) {
    if constexpr (KPair < kKPairs) {
      do_k_pair<KPair>(warp_id, issue_g2s, g2s_smem_a, g2s_smem_b_high, g2s_smem_b_low,
                       params, m_offset, n_offset, k_offset_next,
                       smem, next_stage, warp_coord_a, warp_coord_b,
                       tsm_add_base_a, tsm_add_base_bh, tsm_add_base_bl,
                       accum_high, accum_low,
                       a_buf0, bh_buf0, bl_buf0, a_buf1, bh_buf1, bl_buf1);
      run_k_pairs<KPair + 1>(warp_id, issue_g2s, g2s_smem_a, g2s_smem_b_high, g2s_smem_b_low,
                             params, m_offset, n_offset, k_offset_next,
                             smem, next_stage, warp_coord_a, warp_coord_b,
                             tsm_add_base_a, tsm_add_base_bh, tsm_add_base_bl,
                             accum_high, accum_low,
                             a_buf0, bh_buf0, bl_buf0, a_buf1, bh_buf1, bl_buf1);
    }
  }

  // Fusion-only experiment: keep A/high double-buffered and load low on demand for the current K step and N subtile.
  // Shorten low-fragment lifetimes without changing any output element's full-K high/low accumulation order.
  template <int Step>
  static __device__ __forceinline__ void load_ah(
      int const (&base_a)[kMmasPerWarpM],
      int const (&base_bh)[kMmasPerWarpN],
      uint32_t (&a)[kMmasPerWarpM][4],
      uint32_t (&bh)[kMmasPerWarpN][4]) {
    #pragma unroll
    for (int mi = 0; mi < kMmasPerWarpM; ++mi)
      S2RAtomA::load(base_a[mi], Step, a[mi]);
    #pragma unroll
    for (int ni = 0; ni < kMmasPerWarpN; ++ni)
      S2RAtomB::load(base_bh[ni], Step, bh[ni]);
  }

  template <int Step>
  static __device__ __forceinline__ void run_streamed_steps(
      int const (&base_a)[kMmasPerWarpM],
      int const (&base_bh)[kMmasPerWarpN],
      int const (&base_bl)[kMmasPerWarpN],
      float (&accum_high)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      float (&accum_low)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      uint32_t (&a_reg)[2][kMmasPerWarpM][4],
      uint32_t (&bh_reg)[2][kMmasPerWarpN][4],
      int warp_id, bool issue_g2s, Element* g2s_a, Element* g2s_bh, Element* g2s_bl,
      Params const& params, int m_offset, int n_offset, int k_offset_next,
      char* smem, int next_stage, int warp_coord_a, int warp_coord_b) {
    if constexpr (Step < kKStepsPerWarp) {
      constexpr int current = Step % 2, next = 1 - current;
      if constexpr (Step + 1 < kKStepsPerWarp)
        load_ah<Step + 1>(base_a, base_bh, a_reg[next], bh_reg[next]);
      #pragma unroll
      for (int ni = 0; ni < kMmasPerWarpN; ++ni) {
        uint32_t low[4];
        S2RAtomB::load(base_bl[ni], Step, low);
        #pragma unroll
        for (int mi = 0; mi < kMmasPerWarpM; ++mi) {
          MmaAtom::mma(accum_high[mi][ni], a_reg[current][mi], bh_reg[current][ni], accum_high[mi][ni]);
          MmaAtom::mma(accum_low[mi][ni], a_reg[current][mi], low, accum_low[mi][ni]);
        }
      }
      if constexpr (Step + 1 == kKStepsPerWarp) {
        // Reuse the stage with the original barrier only after all warps have read the final low step.
        // Preserve wait/fence operations and empty-group counts; defer stage reuse until its MMAs complete.
        issue_g2s_and_wait(warp_id, issue_g2s, g2s_a, g2s_bh, g2s_bl,
                           params, m_offset, n_offset, k_offset_next);
        Element const* elems = reinterpret_cast<Element const*>(smem);
        constexpr int coord_scale = kSwzlW * (int)sizeof(Element) / 16;
        const int next_a = arch::smem_ptr_to_tsm_base(
            elems + next_stage * BLOCK_M * BLOCK_K, warp_coord_a);
        const int next_bh = arch::smem_ptr_to_tsm_base(
            elems + kSmemAElements + next_stage * BLOCK_N * BLOCK_K, warp_coord_b);
        int na[kMmasPerWarpM], nb[kMmasPerWarpN];
        #pragma unroll
        for (int mi = 0; mi < kMmasPerWarpM; ++mi)
          na[mi] = next_a + mi * kMmaM * coord_scale;
        #pragma unroll
        for (int ni = 0; ni < kMmasPerWarpN; ++ni)
          nb[ni] = next_bh + ni * kMmaN * coord_scale;
        load_ah<0>(na, nb, a_reg[0], bh_reg[0]);
      }
      run_streamed_steps<Step + 1>(base_a, base_bh, base_bl, accum_high, accum_low,
          a_reg, bh_reg, warp_id, issue_g2s, g2s_a, g2s_bh, g2s_bl,
          params, m_offset, n_offset, k_offset_next, smem, next_stage, warp_coord_a, warp_coord_b);
    }
  }

  // Accumulate each N subtile in S0, S1 order; reuse the stage only after both low steps for the final N subtile are in registers.
  // Do not overwrite low before other warps read it; preserve the original wait/fence operations and empty-group counts.
  template <int Pair>
  static __device__ __forceinline__ void run_streamed_pairs(
      int const (&base_a)[kMmasPerWarpM],
      int const (&base_bh)[kMmasPerWarpN],
      int const (&base_bl)[kMmasPerWarpN],
      float (&accum_high)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      float (&accum_low)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      uint32_t (&a_reg)[2][kMmasPerWarpM][4],
      uint32_t (&bh_reg)[2][kMmasPerWarpN][4],
      int warp_id, bool issue_g2s, Element* g2s_a, Element* g2s_bh, Element* g2s_bl,
      Params const& params, int m_offset, int n_offset, int k_offset_next,
      char* smem, int next_stage, int warp_coord_a, int warp_coord_b) {
    if constexpr (Pair < kKPairs) {
      constexpr int s0 = 2 * Pair, s1 = s0 + 1;
      load_ah<s1>(base_a, base_bh, a_reg[1], bh_reg[1]);
      #pragma unroll
      for (int ni = 0; ni < kMmasPerWarpN; ++ni) {
        uint32_t low0[4], low1[4];
        S2RAtomB::load(base_bl[ni], s0, low0);
        S2RAtomB::load(base_bl[ni], s1, low1);
        if constexpr (Pair == kKPairs - 1) {
          if (ni == kMmasPerWarpN - 1)
            issue_g2s_and_wait(warp_id, issue_g2s, g2s_a, g2s_bh, g2s_bl,
                               params, m_offset, n_offset, k_offset_next);
        }
        if constexpr (DG_DUAL_B_PAIRS_M_FIRST) {
          // For each N subtile, finish s0 for all M subtiles before issuing s1 to increase spacing between independent accumulation chains.
          // Each output retains s0, s1 K-step order; both accumulators, low loads, and stage synchronization points remain unchanged.
          #pragma unroll
          for (int mi = 0; mi < kMmasPerWarpM; ++mi) {
            MmaAtom::mma(accum_high[mi][ni], a_reg[0][mi], bh_reg[0][ni], accum_high[mi][ni]);
            MmaAtom::mma(accum_low[mi][ni], a_reg[0][mi], low0, accum_low[mi][ni]);
          }
          #pragma unroll
          for (int mi = 0; mi < kMmasPerWarpM; ++mi) {
            MmaAtom::mma(accum_high[mi][ni], a_reg[1][mi], bh_reg[1][ni], accum_high[mi][ni]);
            MmaAtom::mma(accum_low[mi][ni], a_reg[1][mi], low1, accum_low[mi][ni]);
          }
        } else {
          #pragma unroll
          for (int mi = 0; mi < kMmasPerWarpM; ++mi) {
            MmaAtom::mma(accum_high[mi][ni], a_reg[0][mi], bh_reg[0][ni], accum_high[mi][ni]);
            MmaAtom::mma(accum_low[mi][ni], a_reg[0][mi], low0, accum_low[mi][ni]);
            MmaAtom::mma(accum_high[mi][ni], a_reg[1][mi], bh_reg[1][ni], accum_high[mi][ni]);
            MmaAtom::mma(accum_low[mi][ni], a_reg[1][mi], low1, accum_low[mi][ni]);
          }
        }
      }
      if constexpr (Pair == kKPairs - 1) {
        Element const* elems = reinterpret_cast<Element const*>(smem);
        constexpr int coord_scale = kSwzlW * (int)sizeof(Element) / 16;
        const int next_a = arch::smem_ptr_to_tsm_base(
            elems + next_stage * BLOCK_M * BLOCK_K, warp_coord_a);
        const int next_bh = arch::smem_ptr_to_tsm_base(
            elems + kSmemAElements + next_stage * BLOCK_N * BLOCK_K, warp_coord_b);
        int na[kMmasPerWarpM], nb[kMmasPerWarpN];
        #pragma unroll
        for (int mi = 0; mi < kMmasPerWarpM; ++mi)
          na[mi] = next_a + mi * kMmaM * coord_scale;
        #pragma unroll
        for (int ni = 0; ni < kMmasPerWarpN; ++ni)
          nb[ni] = next_bh + ni * kMmaN * coord_scale;
        load_ah<0>(na, nb, a_reg[0], bh_reg[0]);
      } else {
        load_ah<s1 + 1>(base_a, base_bh, a_reg[0], bh_reg[0]);
      }
      run_streamed_pairs<Pair + 1>(base_a, base_bh, base_bl, accum_high, accum_low,
          a_reg, bh_reg, warp_id, issue_g2s, g2s_a, g2s_bh, g2s_bl,
          params, m_offset, n_offset, k_offset_next, smem, next_stage, warp_coord_a, warp_coord_b);
    }
  }

  /// S2R-first double-buffered mma_tile over both B operands. See
  /// Mainloop::mma_tile for the register-buffer design notes.
  static __device__ __forceinline__ void mma_tile(
      int stage, int warp_row, int warp_col, int warp_k,
      char* smem,
      float (&accum_high)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      float (&accum_low)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      // G2S pipeline parameters
      int warp_id,
      bool issue_g2s,
      Element* g2s_smem_a,
      Element* g2s_smem_b_high,
      Element* g2s_smem_b_low,
      Params const& params,
      int m_offset, int n_offset, int k_offset_next,
      bool skip_preload,
      int next_stage,
      uint32_t (&a_reg)[2][kMmasPerWarpM][4],
      uint32_t (&bh_reg)[2][kMmasPerWarpN][4],
      uint32_t (&bl_reg)[2][kMmasPerWarpN][4]) {

    Element const* smem_elems = reinterpret_cast<Element const*>(smem);
    Element const* stage_a  = smem_elems + stage * BLOCK_M * BLOCK_K;
    Element const* stage_bh = smem_elems + kSmemAElements + stage * BLOCK_N * BLOCK_K;
    Element const* stage_bl = smem_elems + kSmemBHighElements + stage * BLOCK_N * BLOCK_K;

    static constexpr int kCoordScale = kSwzlW * (int)sizeof(Element) / 16;

    const int warp_coord_a = warp_row * WARP_M * kCoordScale + warp_k * kWarpKOffsetA;
    const int warp_coord_b = warp_col * WARP_N * kCoordScale + warp_k * kWarpKOffsetB;

    const int warp_base_a = arch::smem_ptr_to_tsm_base(stage_a, warp_coord_a);
    const int warp_base_bh = arch::smem_ptr_to_tsm_base(stage_bh, warp_coord_b);
    const int warp_base_bl = arch::smem_ptr_to_tsm_base(stage_bl, warp_coord_b);

    int tsm_add_base_a[kMmasPerWarpM];
    #pragma unroll
    for (int mma_m = 0; mma_m < kMmasPerWarpM; ++mma_m) {
      tsm_add_base_a[mma_m] = warp_base_a + mma_m * kMmaM * kCoordScale;
    }
    int tsm_add_base_bh[kMmasPerWarpN];
    int tsm_add_base_bl[kMmasPerWarpN];
    #pragma unroll
    for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
      tsm_add_base_bh[mma_n] = warp_base_bh + mma_n * kMmaN * kCoordScale;
      tsm_add_base_bl[mma_n] = warp_base_bl + mma_n * kMmaN * kCoordScale;
    }

    if (!skip_preload) {
      if constexpr (DG_DUAL_B_STREAMED_LOW)
        load_ah<0>(tsm_add_base_a, tsm_add_base_bh, a_reg[0], bh_reg[0]);
      else
        issue_s2r_triple<0>(tsm_add_base_a, tsm_add_base_bh, tsm_add_base_bl,
                            a_reg[0], bh_reg[0], bl_reg[0]);
    }

    if constexpr (DG_DUAL_B_STREAMED_LOW) {
      static_assert(!DG_DUAL_B_STREAMED_LOW || (!DG_DUAL_B_FULL_K_PREFETCH
                    && !DG_DUAL_B_EARLY_G2S && !DG_DUAL_B_MMA_S2R_OVERLAP && !DG_DUAL_B_BOUNDED_KLOOP),
                    "streamed low uses its own stage release schedule");
      if constexpr (DG_DUAL_B_STREAMED_PAIRS) {
        run_streamed_pairs<0>(tsm_add_base_a, tsm_add_base_bh, tsm_add_base_bl,
            accum_high, accum_low, a_reg, bh_reg, warp_id, issue_g2s,
            g2s_smem_a, g2s_smem_b_high, g2s_smem_b_low, params, m_offset, n_offset, k_offset_next,
            smem, next_stage, warp_coord_a, warp_coord_b);
      } else {
        run_streamed_steps<0>(tsm_add_base_a, tsm_add_base_bh, tsm_add_base_bl,
            accum_high, accum_low, a_reg, bh_reg, warp_id, issue_g2s,
            g2s_smem_a, g2s_smem_b_high, g2s_smem_b_low, params, m_offset, n_offset, k_offset_next,
            smem, next_stage, warp_coord_a, warp_coord_b);
      }
    } else if constexpr (DG_DUAL_B_BOUNDED_KLOOP) {
      static_assert(!DG_DUAL_B_BOUNDED_KLOOP ||
                    (!DG_DUAL_B_FULL_K_PREFETCH && !DG_DUAL_B_EARLY_G2S && !DG_DUAL_B_MMA_S2R_OVERLAP),
                    "bounded K loop is an isolated instruction scheduling experiment");
      run_inner_k_pairs_bounded(tsm_add_base_a, tsm_add_base_bh, tsm_add_base_bl,
                                accum_high, accum_low, a_reg, bh_reg, bl_reg);
      do_k_pair<kKPairs - 1>(warp_id, issue_g2s, g2s_smem_a, g2s_smem_b_high, g2s_smem_b_low,
                             params, m_offset, n_offset, k_offset_next,
                             smem, next_stage, warp_coord_a, warp_coord_b,
                             tsm_add_base_a, tsm_add_base_bh, tsm_add_base_bl,
                             accum_high, accum_low,
                             a_reg[0], bh_reg[0], bl_reg[0], a_reg[1], bh_reg[1], bl_reg[1]);
    } else {
      run_k_pairs<0>(warp_id, issue_g2s, g2s_smem_a, g2s_smem_b_high, g2s_smem_b_low,
                     params, m_offset, n_offset, k_offset_next,
                     smem, next_stage, warp_coord_a, warp_coord_b,
                     tsm_add_base_a, tsm_add_base_bh, tsm_add_base_bl,
                     accum_high, accum_low,
                     a_reg[0], bh_reg[0], bl_reg[0], a_reg[1], bh_reg[1], bl_reg[1]);
    }
  }

  // Small-warp experiment: preload all three operands of the current K tile into registers before reusing the stage for G2S.
  // Preserve single-warp issue, wait/fence operations, and CTA barriers; overlap current-tile MMAs with next-tile copies.
  static __device__ __forceinline__ void mma_tile_prefetch_all(
      int stage, char* smem, int warp_row, int warp_col, int warp_id,
      bool issue_g2s, Params const& params, int m_offset, int n_offset, int k_offset_next,
      float (&accum_high)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      float (&accum_low)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread]) {
    static_assert(WARP_M == kMmaM && WARP_N == kMmaN && WARP_K == BLOCK_K,
                  "full register prefetch requires a single MMA tile per warp and full K");
    Element* elems = reinterpret_cast<Element*>(smem);
    Element* stage_a = elems + stage * BLOCK_M * BLOCK_K;
    Element* stage_bh = elems + kSmemAElements + stage * BLOCK_N * BLOCK_K;
    Element* stage_bl = elems + kSmemBHighElements + stage * BLOCK_N * BLOCK_K;
    constexpr int coord_scale = kSwzlW * (int)sizeof(Element) / 16;
    int base_a = arch::smem_ptr_to_tsm_base(stage_a, warp_row * WARP_M * coord_scale);
    int base_bh = arch::smem_ptr_to_tsm_base(stage_bh, warp_col * WARP_N * coord_scale);
    int base_bl = arch::smem_ptr_to_tsm_base(stage_bl, warp_col * WARP_N * coord_scale);
    uint32_t a[kKSteps][4], bh[kKSteps][4], bl[kKSteps][4];
    #pragma unroll
    for (int step = 0; step < kKSteps; ++step) {
      S2RAtomA::load(base_a, step, a[step]);
      S2RAtomB::load(base_bh, step, bh[step]);
      S2RAtomB::load(base_bl, step, bl[step]);
    }
    issue_g2s_and_wait(warp_id, issue_g2s, stage_a, stage_bh, stage_bl,
                       params, m_offset, n_offset, k_offset_next);
    #pragma unroll
    for (int step = 0; step < kKSteps; ++step) {
      // Preserve K order within each chain; high and low always accumulate independently.
      MmaAtom::mma(accum_high[0][0], a[step], bh[step], accum_high[0][0]);
      MmaAtom::mma(accum_low[0][0], a[step], bl[step], accum_low[0][0]);
    }
  }

  /// Run the whole K-pipeline for one output tile: prologue + mainloop, for
  /// both B operands at once.
  static __device__ __forceinline__ void run(
      Params const& params,
      char* smem,
      int warp_id, int warp_row, int warp_col, int warp_k,
      int m_offset, int n_offset,
      float (&accum_high)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      float (&accum_low)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      bool stage0_preloaded = false) {

    int total_k_tiles = math::ceil_div(params.K, BLOCK_K);

    // Smem base pointers
    Element* smem_a = reinterpret_cast<Element*>(smem);
    Element* smem_b_high = smem_a + kSmemAElements;
    Element* smem_b_low = smem_a + kSmemBHighElements;

    // Prologue: fill STAGES tiles (A + B_high + B_low per stage).
    int k_tile_to_load = stage0_preloaded ? 1 : 0;
    #pragma unroll
    for (int s = stage0_preloaded ? 1 : 0; s < STAGES && k_tile_to_load < total_k_tiles; ++s) {
      if (warp_id == 0) {
        G2SAtomA::load(smem_a + s * BLOCK_M * BLOCK_K, params.ptr_A,
                       params.M, params.lda, m_offset, k_tile_to_load * BLOCK_K);
        G2SAtomB::load(smem_b_high + s * BLOCK_N * BLOCK_K, params.ptr_B_high,
                       params.N, params.ldb, n_offset, k_tile_to_load * BLOCK_K);
        G2SAtomB::load(smem_b_low + s * BLOCK_N * BLOCK_K, params.ptr_B_low,
                       params.N, params.ldb, n_offset, k_tile_to_load * BLOCK_K);
      }
      cutlass::arch::cp_async_fence();
      ++k_tile_to_load;
    }
    if (k_tile_to_load >= STAGES) {
      cutlass::arch::cp_async_wait<STAGES - 1>();
    } else {
      cutlass::arch::cp_async_wait<0>();
    }
    __syncthreads();

    uint32_t a_reg[2][kMmasPerWarpM][4];
    uint32_t bh_reg[2][kMmasPerWarpN][4];
    uint32_t bl_reg[2][kMmasPerWarpN][4];

    if constexpr (DG_DUAL_B_STATIC_STAGES) {
      static_assert(!DG_DUAL_B_STATIC_STAGES || !DG_DUAL_B_FULL_K_PREFETCH,
                    "static stages uses the original double-buffered fragments");
      // Loop over complete stage cycles and unroll inner indices at compile time; preserve every original wait/fence,
      // CTA barrier, and unconditional empty group. The tail advances in K order without computing or loading extra K tiles.
      #pragma unroll 1
      for (int base = 0; base < total_k_tiles; base += STAGES) {
        #pragma unroll
        for (int stage = 0; stage < STAGES; ++stage) {
          if (base + stage < total_k_tiles) {
            const bool issue_g2s = k_tile_to_load < total_k_tiles;
            mma_tile(stage, warp_row, warp_col, warp_k, smem, accum_high, accum_low,
                     warp_id, issue_g2s,
                     smem_a + stage * BLOCK_M * BLOCK_K,
                     smem_b_high + stage * BLOCK_N * BLOCK_K,
                     smem_b_low + stage * BLOCK_N * BLOCK_K,
                     params, m_offset, n_offset, k_tile_to_load * BLOCK_K,
                     /*skip_preload=*/ (base + stage > 0),
                     /*next_stage=*/ (stage + 1 == STAGES ? 0 : stage + 1),
                     a_reg, bh_reg, bl_reg);
            if (issue_g2s) ++k_tile_to_load;
          }
        }
      }
    } else {
      int read_stage = 0;
      for (int k_iter = 0; k_iter < total_k_tiles; ++k_iter) {
        bool issue_g2s = (k_tile_to_load < total_k_tiles);
        int next_read_stage = (read_stage + 1 >= STAGES) ? 0 : read_stage + 1;

        if constexpr (DG_DUAL_B_FULL_K_PREFETCH && WARP_M == kMmaM && WARP_N == kMmaN
                      && WARP_K == BLOCK_K && BLOCK_K <= 128) {
          mma_tile_prefetch_all(read_stage, smem, warp_row, warp_col, warp_id, issue_g2s,
                                params, m_offset, n_offset, k_tile_to_load * BLOCK_K,
                                accum_high, accum_low);
        } else {
          mma_tile(read_stage, warp_row, warp_col, warp_k, smem, accum_high, accum_low,
                   warp_id, issue_g2s,
                   smem_a + read_stage * BLOCK_M * BLOCK_K,
                   smem_b_high + read_stage * BLOCK_N * BLOCK_K,
                   smem_b_low + read_stage * BLOCK_N * BLOCK_K,
                   params, m_offset, n_offset, k_tile_to_load * BLOCK_K,
                   /*skip_preload=*/ (k_iter > 0),
                   /*next_stage=*/ next_read_stage,
                   a_reg, bh_reg, bl_reg);
        }

        if (issue_g2s) ++k_tile_to_load;
        if (++read_stage >= STAGES) read_stage = 0;
      }
    }

    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
  }

  // As in standard BF16, the kernel preissues stage0; all three operands belong to the same async group.
  static __device__ __forceinline__ void issue_prologue_stage0(
      char* smem, int warp_id, Params const& params, int m_offset, int n_offset) {
    Element* elems = reinterpret_cast<Element*>(smem);
    if (warp_id == 0) {
      G2SAtomA::load(elems, params.ptr_A, params.M, params.lda, m_offset, 0);
      G2SAtomB::load(elems + kSmemAElements, params.ptr_B_high,
                     params.N, params.ldb, n_offset, 0);
      G2SAtomB::load(elems + kSmemBHighElements, params.ptr_B_low,
                     params.N, params.ldb, n_offset, 0);
    }
    cutlass::arch::cp_async_fence();
  }
};

} // namespace mainloop
} // namespace deep_gemm
