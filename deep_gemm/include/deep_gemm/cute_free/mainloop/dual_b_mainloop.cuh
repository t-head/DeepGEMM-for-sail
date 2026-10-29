#pragma once

#include <cstdint>

#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory_ppu.h>
#include <deep_gemm/cute_free/arch/ppu_common.cuh>
#include <deep_gemm/common/math.cuh>

namespace deep_gemm {
namespace mainloop {

// =============================================================================
// DualBMainloop: K-pipeline carrying two B operands (W_high / W_low) per stage.
//
// Variant of Mainloop for the BF16 x FP32 dual-B decomposition
// (kernel::GemmBf16xFp32Kernel). Instead of running the single-B pipeline twice
// per output tile -- which streams A twice from global memory and pays the
// stage warm-up / drain bubbles twice -- every pipeline stage holds
// A + B_high + B_low together (mirroring hpc-ops' SMEM layout that keeps the
// high/low pair resident per stage):
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
    load_next_fragments<S0>(smem, next_stage, warp_coord_a, warp_coord_b,
                            tsm_add_base_a, tsm_add_base_bh, tsm_add_base_bl,
                            a_buf1, bh_buf1, bl_buf1);
    #pragma unroll
    for (int mma_m = 0; mma_m < kMmasPerWarpM; ++mma_m) {
      #pragma unroll
      for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
        MmaAtom::mma(accum_high[mma_m][mma_n], a_buf0[mma_m], bh_buf0[mma_n], accum_high[mma_m][mma_n]);
        MmaAtom::mma(accum_low[mma_m][mma_n], a_buf0[mma_m], bl_buf0[mma_n], accum_low[mma_m][mma_n]);
      }
    }
    if constexpr (KPair == kKPairs - 1) {
      issue_g2s_and_wait(warp_id, issue_g2s, g2s_smem_a, g2s_smem_b_high, g2s_smem_b_low,
                         params, m_offset, n_offset, k_offset_next);
    }

    // --- Step S1: prefetch (S1+1)'s data into buf0, then two MMAs with buf1 ---
    load_next_fragments<S1>(smem, next_stage, warp_coord_a, warp_coord_b,
                            tsm_add_base_a, tsm_add_base_bh, tsm_add_base_bl,
                            a_buf0, bh_buf0, bl_buf0);
    #pragma unroll
    for (int mma_m = 0; mma_m < kMmasPerWarpM; ++mma_m) {
      #pragma unroll
      for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
        MmaAtom::mma(accum_high[mma_m][mma_n], a_buf1[mma_m], bh_buf1[mma_n], accum_high[mma_m][mma_n]);
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
      issue_s2r_triple<0>(tsm_add_base_a, tsm_add_base_bh, tsm_add_base_bl,
                          a_reg[0], bh_reg[0], bl_reg[0]);
    }

    run_k_pairs<0>(warp_id, issue_g2s, g2s_smem_a, g2s_smem_b_high, g2s_smem_b_low,
                   params, m_offset, n_offset, k_offset_next,
                   smem, next_stage, warp_coord_a, warp_coord_b,
                   tsm_add_base_a, tsm_add_base_bh, tsm_add_base_bl,
                   accum_high, accum_low,
                   a_reg[0], bh_reg[0], bl_reg[0], a_reg[1], bh_reg[1], bl_reg[1]);
  }

  /// Run the whole K-pipeline for one output tile: prologue + mainloop, for
  /// both B operands at once.
  static __device__ __forceinline__ void run(
      Params const& params,
      char* smem,
      int warp_id, int warp_row, int warp_col, int warp_k,
      int m_offset, int n_offset,
      float (&accum_high)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      float (&accum_low)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread]) {

    int total_k_tiles = math::ceil_div(params.K, BLOCK_K);

    // Smem base pointers
    Element* smem_a = reinterpret_cast<Element*>(smem);
    Element* smem_b_high = smem_a + kSmemAElements;
    Element* smem_b_low = smem_a + kSmemBHighElements;

    // Prologue: fill STAGES tiles (A + B_high + B_low per stage).
    int k_tile_to_load = 0;
    #pragma unroll
    for (int s = 0; s < STAGES && k_tile_to_load < total_k_tiles; ++s) {
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

    int read_stage = 0;

    uint32_t a_reg[2][kMmasPerWarpM][4];
    uint32_t bh_reg[2][kMmasPerWarpN][4];
    uint32_t bl_reg[2][kMmasPerWarpN][4];

    for (int k_iter = 0; k_iter < total_k_tiles; ++k_iter) {
      bool issue_g2s = (k_tile_to_load < total_k_tiles);
      int next_read_stage = (read_stage + 1 >= STAGES) ? 0 : read_stage + 1;

      mma_tile(read_stage, warp_row, warp_col, warp_k, smem, accum_high, accum_low,
               warp_id, issue_g2s,
               smem_a + read_stage * BLOCK_M * BLOCK_K,
               smem_b_high + read_stage * BLOCK_N * BLOCK_K,
               smem_b_low + read_stage * BLOCK_N * BLOCK_K,
               params, m_offset, n_offset, k_tile_to_load * BLOCK_K,
               /*skip_preload=*/ (k_iter > 0),
               /*next_stage=*/ next_read_stage,
               a_reg, bh_reg, bl_reg);

      if (issue_g2s) ++k_tile_to_load;
      if (++read_stage >= STAGES) read_stage = 0;
    }

    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
  }
};

} // namespace mainloop
} // namespace deep_gemm
