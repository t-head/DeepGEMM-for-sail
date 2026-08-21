#pragma once

#include <cstdint>

#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory_ppu.h"
#include "cute_free/arch/ppu_common.cuh"
#include "utils_rtc.cuh"

namespace deep_gemm {
namespace mainloop {

// =============================================================================
// Mainloop: the collective K-pipeline of the cute-free GEMM kernel.
//
// This is the "collective mainloop" layer in the cutlass3-style stacking,
// symmetric to epilogue::Epilogue. It owns the whole K-dimension
// pipeline and is agnostic to the concrete copy / mma atoms (injected as
// template parameters). The kernel layer (cute_free/kernel/gemm.cuh) only
// orchestrates: scheduler -> Mainloop::run() -> Epilogue::store().
//
// Pipeline (S2R-first double-buffered, G2S interleaved in k_step):
//   - prologue fills STAGES-1 tiles via G2S
//   - 2-level register double buffer indexed by k_step & 1
//   - S2R-first ordering: load next fragment before MMA compute
//   - cross-tile prefetch: at the last k_step, S2R from the next stage's k=0
//   - G2S for the next tile is issued at k_step == kKStepsPerWarp-2
//
// Template parameters:
//   Element     : matrix element type (e.g. cutlass::bfloat16_t)
//   G2SAtomA/B  : global->shared copy atoms for A / B
//   S2RAtomA/B  : shared->register copy atoms for A / B
//   MmaAtom     : warp-level MMA atom (exposes kMmaM/kMmaN/kMmaK/kAccumPerThread)
//   BLOCK_M/N/K : CTA tile shape
//   WARP_M/N    : warp tile shape
//   STAGES      : number of pipeline stages (>= 2)
// =============================================================================

template <typename Element,
          typename G2SAtomA, typename G2SAtomB,
          typename S2RAtomA, typename S2RAtomB,
          typename MmaAtom,
          int BLOCK_M, int BLOCK_N, int BLOCK_K,
          int WARP_M, int WARP_N, int WARP_K, int STAGES,
          bool DenseS2Opt>
struct Mainloop {
  // ---- Constants derived from the MMA atom + tile config ----
  static constexpr int kMmaM = MmaAtom::kMmaM;
  static constexpr int kMmaN = MmaAtom::kMmaN;
  static constexpr int kMmaK = MmaAtom::kMmaK;
  static constexpr int kAccumPerThread = MmaAtom::kAccumPerThread;

  static constexpr int kMmasPerWarpM = WARP_M / kMmaM;
  static constexpr int kMmasPerWarpN = WARP_N / kMmaN;

  // ---- WarpOnK: split BLOCK_K across kWarpsK warp groups ----
  // Each warp group owns the K range [warp_k*WARP_K, (warp_k+1)*WARP_K) and
  // produces a PARTIAL sum; the kernel layer reduces across warp_k afterwards.
  // kWarpsK == 1 reduces to the original single-K-group behaviour exactly.
  static constexpr int kWarpsK = BLOCK_K / WARP_K;
  static constexpr int kKStepsPerWarp = WARP_K / kMmaK;
  // Total k_steps in the CTA tile (kept for reference / smem sizing).
  static constexpr int kKSteps = BLOCK_K / kMmaK;


  // Swizzle-tile width along K (cutlass DefaultGemm_AIU_Operand: AiuContByteSize
  // capped at 128B). BLOCK_K wider than one swizzle tile is stored as
  // kInstNum = BLOCK_K/kSwzlW sub-tiles; the G2S/S2R atoms handle the split, the
  // mainloop only needs kSwzlW to scale the per-row tsm base (sub-tile 0 row).
  static constexpr int kElemBytes = (int)sizeof(Element);
  static constexpr int kSwzlBytes = (BLOCK_K * kElemBytes > 128) ? 128 : BLOCK_K * kElemBytes;
  static constexpr int kSwzlW = kSwzlBytes / kElemBytes;

  static_assert(STAGES >= 2, "STAGES must be >= 2 for the async pipeline");
  static_assert(BLOCK_K % WARP_K == 0, "BLOCK_K must be a multiple of WARP_K");
  static_assert(WARP_K % kMmaK == 0, "WARP_K must be a multiple of kMmaK");
  // A warp group's K slice must start on a swizzle sub-tile boundary, otherwise
  // its base address would straddle two sub-tiles and the tsm addressing breaks.
  static_assert(WARP_K % kSwzlW == 0, "WARP_K must be a multiple of the swizzle-tile width");
  // The G2S issue / wait are placed at kKStepsPerWarp-2, so each warp group needs
  // at least two k_steps of its own for the pipeline to interleave.
  static_assert(kKStepsPerWarp >= 2, "WARP_K / kMmaK must be >= 2 for G2S interleaving");
  // Both unroll shapes need an EVEN number of k_steps per warp group:
  //   - run_k_pairs: kKPairs = kKStepsPerWarp/2 would truncate on an odd count and
  //     silently drop the last k_step (wrong results, no diagnostic).
  //   - run_k_steps: the register ping-pong alternates by KStep parity, so an odd
  //     count leaves the final cross-tile prefetch in buf1 while the next k_iter
  //     starts reading buf0.
  static_assert(kKStepsPerWarp % 2 == 0,
                "WARP_K / kMmaK must be EVEN: both the k_pair and the k_step unroll "
                "rely on steps pairing up (see kUseKPairUnroll)");

  // K offset (in 16B tsm units) that moves a base address by one warp-K slice.
  // WARP_K/kSwzlW whole sub-tiles; A and B differ because the sub-tile stride
  // scales with the tile height (BLOCK_M for A, BLOCK_N for B).
  static constexpr int kWarpKSubTiles  = WARP_K / kSwzlW;
  static constexpr int kWarpKOffsetA   = kWarpKSubTiles * S2RAtomA::kSubTileStride16B;
  static constexpr int kWarpKOffsetB   = kWarpKSubTiles * S2RAtomB::kSubTileStride16B;


  // A/B-side parameters only (decoupled from the kernel's full Params).
  struct Params {
    Element const* ptr_A;
    Element const* ptr_B;
    int M, N, K;
    int lda, ldb;
  };

  // Number of k_pairs: each pair does 2 consecutive k_steps, matching cutlass3's
  // K_ATOM_PER_COPY=2 structure. This halves the template recursion depth (4 vs 8),
  // reducing the number of simultaneously live S2R fragments and eliminating the
  // ~166 s.mov register shuffling instructions the compiler inserted at depth 8.
  static constexpr int kKPairs = kKStepsPerWarp / 2;

  // Unroll shape of the K loop, switchable for A/B:
  //   true  -> run_k_pairs: 2 k_steps per recursion level (kKPairs levels), the
  //            half-unroll that mirrors cutlass3's K_ATOM_PER_COPY=2.
  //   false -> run_k_steps: 1 k_step per level (kKStepsPerWarp levels), the
  //            original full unroll.
  // The half-unroll was introduced to cut the s.mov register shuffling that full
  // unroll provokes, but it was benchmarked while the DenseS2Opt wait<0> bug was
  // still masking the pipeline, so its real effect is unverified. Both forms issue
  // G2S at the same point (after the MMA of step kKStepsPerWarp-2) and share
  // issue_g2s_and_wait(), so flipping this only changes the unroll shape.
  static constexpr bool kUseKPairUnroll = true;

  /// Issue the S2R loads for one k_step into the given fragment buffers, grouped
  /// by operand: all A fragments, then all B. SrcKStep is the source k_step
  /// inside the tile.
  template <int SrcKStep>
  static __device__ __forceinline__ void issue_s2r_pair(
      int const (&tsm_add_base_a)[kMmasPerWarpM],
      int const (&tsm_add_base_b)[kMmasPerWarpN],
      uint32_t (&a_dst)[kMmasPerWarpM][4],
      uint32_t (&b_dst)[kMmasPerWarpN][4]) {
    #pragma unroll
    for (int mma_m = 0; mma_m < kMmasPerWarpM; ++mma_m) {
      S2RAtomA::load(tsm_add_base_a[mma_m], SrcKStep, a_dst[mma_m]);
    }
    #pragma unroll
    for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
      S2RAtomB::load(tsm_add_base_b[mma_n], SrcKStep, b_dst[mma_n]);
    }
  }

  /// G2S issue point, shared by both unroll shapes. Placed after the MMA of step
  /// kKStepsPerWarp-2 and before the cross-tile prefetch of step kKStepsPerWarp-1,
  /// with the wait schedule kept isomorphic to cutlass3 (see do_k_pair).
  static __device__ __forceinline__ void issue_g2s_and_wait(
      int warp_id,
      bool issue_g2s,
      Element* g2s_smem_a,
      Element* g2s_smem_b,
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
        G2SAtomB::load(g2s_smem_b, params.ptr_B, params.N, params.ldb, n_offset, k_offset_next);
      }
    }
    // Fence stays UNCONDITIONAL: it creates an empty async group in the drain phase so
    // the pending count keeps decaying in FIFO order and wait<STAGES-2> still pops the
    // oldest real group, exactly as cutlass3 does.
    cutlass::arch::cp_async_fence();
    if constexpr (DenseS2Opt && STAGES == 2) {
      // Point of use: the caller's next cross-tile prefetch reads next_stage, so
      // next_stage's G2S must have landed. wait<1> drains the older group while
      // leaving the group just issued -- which targets the stage we have stopped
      // reading -- in flight. That surviving group is the DMA/MMA overlap a
      // wait<0> drain throws away.
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
      int const (&tsm_add_base_b)[kMmasPerWarpN],
      uint32_t (&a_next)[kMmasPerWarpM][4],
      uint32_t (&b_next)[kMmasPerWarpN][4]) {
    static constexpr int kCoordScale = kSwzlW * (int)sizeof(Element) / 16;

    if constexpr (KStep < kKStepsPerWarp - 1) {
      issue_s2r_pair<KStep + 1>(tsm_add_base_a, tsm_add_base_b, a_next, b_next);
    } else {
      // Cross-tile prefetch: load step 0 from the NEXT stage (matches cutlass3's
      // circular (k_block+1) % K_BLOCK_MAX indexing at the last k_block).
      Element const* next_stage_a =
          reinterpret_cast<Element const*>(smem) + next_stage * BLOCK_M * BLOCK_K;
      Element const* next_stage_b =
          reinterpret_cast<Element const*>(smem)
          + STAGES * BLOCK_M * BLOCK_K + next_stage * BLOCK_N * BLOCK_K;
      const int next_base_a = arch::smem_ptr_to_tsm_base(next_stage_a, warp_coord_a);
      const int next_base_b = arch::smem_ptr_to_tsm_base(next_stage_b, warp_coord_b);
      #pragma unroll
      for (int mma_m = 0; mma_m < kMmasPerWarpM; ++mma_m) {
        S2RAtomA::load(next_base_a + mma_m * kMmaM * kCoordScale, 0, a_next[mma_m]);
      }
      #pragma unroll
      for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
        S2RAtomB::load(next_base_b + mma_n * kMmaN * kCoordScale, 0, b_next[mma_n]);
      }
    }
  }

  /// Execute one k_pair = 2 consecutive k_steps: (2*KPair) and (2*KPair+1).
  /// This matches cutlass3's "K_ATOM_PER_COPY=2" inner loop that does 2 MMA atoms
  /// per S2R register-pipe slot, halving the recursion depth from 8 to 4.
  ///
  /// Pipeline layout per pair:
  ///   S2R prefetch step 2*KPair+1 into buf_next (for the SECOND step of this pair)
  ///   MMA step 2*KPair (consumes buf_curr)
  ///   S2R prefetch step 2*KPair+2 into buf_curr (for the FIRST step of next pair)
  ///     ↑ at the last pair this becomes the cross-tile prefetch
  ///   MMA step 2*KPair+1 (consumes buf_next)
  ///   [G2S at KPair == kKPairs-1: wait + sync + issue + fence]
  template <int KPair>
  static __device__ __forceinline__ void do_k_pair(
      int warp_id,
      bool issue_g2s,
      Element* g2s_smem_a,
      Element* g2s_smem_b,
      Params const& params,
      int m_offset, int n_offset, int k_offset_next,
      char* smem,
      int next_stage,
      int warp_coord_a,
      int warp_coord_b,
      int const (&tsm_add_base_a)[kMmasPerWarpM],
      int const (&tsm_add_base_b)[kMmasPerWarpN],
      float (&accum)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      uint32_t (&a_buf0)[kMmasPerWarpM][4],
      uint32_t (&b_buf0)[kMmasPerWarpN][4],
      uint32_t (&a_buf1)[kMmasPerWarpM][4],
      uint32_t (&b_buf1)[kMmasPerWarpN][4]) {
    static constexpr int S0 = 2 * KPair;      // first step of this pair
    static constexpr int S1 = 2 * KPair + 1;  // second step of this pair

    // --- Step S0: prefetch S1's data into buf1, then MMA with buf0 ---
    load_next_fragments<S0>(smem, next_stage, warp_coord_a, warp_coord_b,
                            tsm_add_base_a, tsm_add_base_b, a_buf1, b_buf1);
    #pragma unroll
    for (int mma_m = 0; mma_m < kMmasPerWarpM; ++mma_m) {
      #pragma unroll
      for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
        MmaAtom::mma(accum[mma_m][mma_n], a_buf0[mma_m], b_buf0[mma_n], accum[mma_m][mma_n]);
      }
    }
    if constexpr (KPair == kKPairs - 1) {
      issue_g2s_and_wait(warp_id, issue_g2s, g2s_smem_a, g2s_smem_b,
                         params, m_offset, n_offset, k_offset_next);
    }

    // --- Step S1: prefetch (S1+1)'s data into buf0, then MMA with buf1 ---
    // At S1 == kKStepsPerWarp-1 this becomes the cross-tile prefetch (next stage step 0).
    load_next_fragments<S1>(smem, next_stage, warp_coord_a, warp_coord_b,
                            tsm_add_base_a, tsm_add_base_b, a_buf0, b_buf0);
    #pragma unroll
    for (int mma_m = 0; mma_m < kMmasPerWarpM; ++mma_m) {
      #pragma unroll
      for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
        MmaAtom::mma(accum[mma_m][mma_n], a_buf1[mma_m], b_buf1[mma_n], accum[mma_m][mma_n]);
      }
    }
  }

  template <int KPair>
  static __device__ __forceinline__ void run_k_pairs(
      int warp_id,
      bool issue_g2s,
      Element* g2s_smem_a,
      Element* g2s_smem_b,
      Params const& params,
      int m_offset, int n_offset, int k_offset_next,
      char* smem,
      int next_stage,
      int warp_coord_a,
      int warp_coord_b,
      int const (&tsm_add_base_a)[kMmasPerWarpM],
      int const (&tsm_add_base_b)[kMmasPerWarpN],
      float (&accum)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      uint32_t (&a_buf0)[kMmasPerWarpM][4],
      uint32_t (&b_buf0)[kMmasPerWarpN][4],
      uint32_t (&a_buf1)[kMmasPerWarpM][4],
      uint32_t (&b_buf1)[kMmasPerWarpN][4]) {
    if constexpr (KPair < kKPairs) {
      do_k_pair<KPair>(warp_id, issue_g2s, g2s_smem_a, g2s_smem_b,
                       params, m_offset, n_offset, k_offset_next,
                       smem, next_stage, warp_coord_a, warp_coord_b,
                       tsm_add_base_a, tsm_add_base_b, accum,
                       a_buf0, b_buf0, a_buf1, b_buf1);
      run_k_pairs<KPair + 1>(warp_id, issue_g2s, g2s_smem_a, g2s_smem_b,
                             params, m_offset, n_offset, k_offset_next,
                             smem, next_stage, warp_coord_a, warp_coord_b,
                             tsm_add_base_a, tsm_add_base_b, accum,
                             a_buf0, b_buf0, a_buf1, b_buf1);
    }
  }

  template <int KStep>
  static __device__ __forceinline__ void do_k_step(
      int warp_id,
      bool issue_g2s,
      Element* g2s_smem_a,
      Element* g2s_smem_b,
      Params const& params,
      int m_offset, int n_offset, int k_offset_next,
      char* smem,
      int next_stage,
      int warp_coord_a,
      int warp_coord_b,
      int const (&tsm_add_base_a)[kMmasPerWarpM],
      int const (&tsm_add_base_b)[kMmasPerWarpN],
      float (&accum)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      uint32_t (&a_curr)[kMmasPerWarpM][4],
      uint32_t (&b_curr)[kMmasPerWarpN][4],
      uint32_t (&a_next)[kMmasPerWarpM][4],
      uint32_t (&b_next)[kMmasPerWarpN][4]) {
    load_next_fragments<KStep>(smem, next_stage, warp_coord_a, warp_coord_b,
                               tsm_add_base_a, tsm_add_base_b, a_next, b_next);
    #pragma unroll
    for (int mma_m = 0; mma_m < kMmasPerWarpM; ++mma_m) {
      #pragma unroll
      for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
        MmaAtom::mma(accum[mma_m][mma_n], a_curr[mma_m], b_curr[mma_n], accum[mma_m][mma_n]);
      }
    }

    if constexpr (KStep == kKStepsPerWarp - 2) {
      issue_g2s_and_wait(warp_id, issue_g2s, g2s_smem_a, g2s_smem_b,
                         params, m_offset, n_offset, k_offset_next);
    }
  }

  template <int KStep>
  static __device__ __forceinline__ void run_k_steps(
      int warp_id,
      bool issue_g2s,
      Element* g2s_smem_a,
      Element* g2s_smem_b,
      Params const& params,
      int m_offset, int n_offset, int k_offset_next,
      char* smem,
      int next_stage,
      int warp_coord_a,
      int warp_coord_b,
      int const (&tsm_add_base_a)[kMmasPerWarpM],
      int const (&tsm_add_base_b)[kMmasPerWarpN],
      float (&accum)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      uint32_t (&a_buf0)[kMmasPerWarpM][4],
      uint32_t (&b_buf0)[kMmasPerWarpN][4],
      uint32_t (&a_buf1)[kMmasPerWarpM][4],
      uint32_t (&b_buf1)[kMmasPerWarpN][4]) {
    if constexpr (KStep < kKStepsPerWarp) {
      // Even steps consume buf0 and prefetch into buf1; odd steps do the reverse.
      if constexpr ((KStep & 1) == 0) {
        do_k_step<KStep>(warp_id, issue_g2s, g2s_smem_a, g2s_smem_b,
                         params, m_offset, n_offset, k_offset_next,
                         smem, next_stage, warp_coord_a, warp_coord_b,
                         tsm_add_base_a, tsm_add_base_b, accum,
                         a_buf0, b_buf0, a_buf1, b_buf1);
      } else {
        do_k_step<KStep>(warp_id, issue_g2s, g2s_smem_a, g2s_smem_b,
                         params, m_offset, n_offset, k_offset_next,
                         smem, next_stage, warp_coord_a, warp_coord_b,
                         tsm_add_base_a, tsm_add_base_b, accum,
                         a_buf1, b_buf1, a_buf0, b_buf0);
      }
      run_k_steps<KStep + 1>(warp_id, issue_g2s, g2s_smem_a, g2s_smem_b,
                             params, m_offset, n_offset, k_offset_next,
                             smem, next_stage, warp_coord_a, warp_coord_b,
                             tsm_add_base_a, tsm_add_base_b, accum,
                             a_buf0, b_buf0, a_buf1, b_buf1);
    }
  }

  /// S2R-first double-buffered mma_tile, expressed with compile-time KStep
  /// selection and two independent register fragments. This avoids the first
  /// dimension `buf[2]` runtime-looking array access and gives the compiler a
  /// CuTe-like def-use shape while preserving the existing wait/G2S pipeline.
  ///
  /// IssueG2S: whether this tile still has a next tile to fetch. Runtime bool -- see
  /// the note in run() for why peeling it into a compile-time constant was rejected.
  static __device__ __forceinline__ void mma_tile(
      int stage, int warp_row, int warp_col, int warp_k,
      char* smem,
      float (&accum)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      // G2S pipeline parameters
      int warp_id,
      bool issue_g2s,
      Element* g2s_smem_a,
      Element* g2s_smem_b,
      Params const& params,
      int m_offset, int n_offset, int k_offset_next,
      bool skip_preload,
      int next_stage,
      // Single unified register buffer: [2][mma_dim][4] instead of separate buf0/buf1.
      // This lets the compiler see both ping-pong slots as offsets into the same object,
      // enabling register reuse without the s.mov shuffling it inserts for independent arrays.
      uint32_t (&a_reg)[2][kMmasPerWarpM][4],
      uint32_t (&b_reg)[2][kMmasPerWarpN][4]) {

    Element const* stage_a =
        reinterpret_cast<Element const*>(smem) + stage * BLOCK_M * BLOCK_K;
    Element const* stage_b =
        reinterpret_cast<Element const*>(smem)
        + STAGES * BLOCK_M * BLOCK_K + stage * BLOCK_N * BLOCK_K;

    static constexpr int kCoordScale = kSwzlW * (int)sizeof(Element) / 16;

    // Split the S2R base into a runtime part and a compile-time part.
    const int warp_coord_a = warp_row * WARP_M * kCoordScale + warp_k * kWarpKOffsetA;
    const int warp_coord_b = warp_col * WARP_N * kCoordScale + warp_k * kWarpKOffsetB;

    const int warp_base_a = arch::smem_ptr_to_tsm_base(stage_a, warp_coord_a);
    const int warp_base_b = arch::smem_ptr_to_tsm_base(stage_b, warp_coord_b);

    int tsm_add_base_a[kMmasPerWarpM];
    #pragma unroll
    for (int mma_m = 0; mma_m < kMmasPerWarpM; ++mma_m) {
      tsm_add_base_a[mma_m] = warp_base_a + mma_m * kMmaM * kCoordScale;
    }
    int tsm_add_base_b[kMmasPerWarpN];
    #pragma unroll
    for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
      tsm_add_base_b[mma_n] = warp_base_b + mma_n * kMmaN * kCoordScale;
    }

    if (!skip_preload) {
      issue_s2r_pair<0>(tsm_add_base_a, tsm_add_base_b, a_reg[0], b_reg[0]);
    }

    if constexpr (kUseKPairUnroll) {
      run_k_pairs<0>(warp_id, issue_g2s, g2s_smem_a, g2s_smem_b,
                     params, m_offset, n_offset, k_offset_next,
                     smem, next_stage, warp_coord_a, warp_coord_b,
                     tsm_add_base_a, tsm_add_base_b, accum,
                     a_reg[0], b_reg[0], a_reg[1], b_reg[1]);
    } else {
      run_k_steps<0>(warp_id, issue_g2s, g2s_smem_a, g2s_smem_b,
                     params, m_offset, n_offset, k_offset_next,
                     smem, next_stage, warp_coord_a, warp_coord_b,
                     tsm_add_base_a, tsm_add_base_b, accum,
                     a_reg[0], b_reg[0], a_reg[1], b_reg[1]);
    }
  }

  /// Run the whole K-pipeline for one output tile: prologue + mainloop.
  static __device__ __forceinline__ void run(
      Params const& params,
      char* smem,
      int warp_id, int warp_row, int warp_col, int warp_k,
      int m_offset, int n_offset,
      float (&accum)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread]) {

    int total_k_tiles = ceil_div(params.K, BLOCK_K);

    // Smem base pointers
    Element* smem_a = reinterpret_cast<Element*>(smem);
    Element* smem_b = smem_a + STAGES * BLOCK_M * BLOCK_K;

    // Prologue: fill STAGES tiles.
    int k_tile_to_load = 0;
    #pragma unroll
    for (int s = 0; s < STAGES && k_tile_to_load < total_k_tiles; ++s) {
      if (warp_id == 0) {
        G2SAtomA::load(smem_a + s * BLOCK_M * BLOCK_K, params.ptr_A,
                       params.M, params.lda, m_offset, k_tile_to_load * BLOCK_K);
        G2SAtomB::load(smem_b + s * BLOCK_N * BLOCK_K, params.ptr_B,
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
    uint32_t b_reg[2][kMmasPerWarpN][4];

    for (int k_iter = 0; k_iter < total_k_tiles; ++k_iter) {
      bool issue_g2s = (k_tile_to_load < total_k_tiles);
      int next_read_stage = (read_stage + 1 >= STAGES) ? 0 : read_stage + 1;

      mma_tile(read_stage, warp_row, warp_col, warp_k, smem, accum,
               warp_id, issue_g2s,
               smem_a + read_stage * BLOCK_M * BLOCK_K,
               smem_b + read_stage * BLOCK_N * BLOCK_K,
               params, m_offset, n_offset, k_tile_to_load * BLOCK_K,
               /*skip_preload=*/ (k_iter > 0),
               /*next_stage=*/ next_read_stage,
               a_reg, b_reg);

      if (issue_g2s) ++k_tile_to_load;
      if (++read_stage >= STAGES) read_stage = 0;
    }

    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
  }
};

} // namespace mainloop
} // namespace deep_gemm
