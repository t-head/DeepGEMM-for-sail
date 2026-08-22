#pragma once

#include <cstdint>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory_ppu.h"
#include "cute_free/arch/ppu_common.cuh"
#include "cute_free/mainloop/mainloop.cuh"
#include "cute_free/epilogue/warp_k_reduce.cuh"
#include "densegemm_scheduler_cutlass3.cuh"
#include "utils_cutlass3.h"

namespace deep_gemm {
namespace kernel {

// =============================================================================
// GemmKernel: data-type/platform-generic GEMM orchestrator.
//
// This is the "kernel" layer in the cutlass3-style stacking: it owns the
// top-level orchestration (scheduler -> mainloop -> epilogue) but is agnostic
// to the concrete copy / mma / epilogue implementations. The K-dimension
// pipeline lives in mainloop::Mainloop and the accumulator store in
// epilogue::Epilogue; those, together with the copy / mma atoms, are
// injected as template parameters (the "atom" layer), so a new data type or
// platform only needs a new set of atoms + a thin instance alias (see
// bf16_gemm_cute_free.cuh).
//
// Template parameters:
//   Element_    : matrix element type (e.g. cutlass::bfloat16_t)
//   G2SAtomA/B_ : global->shared copy atoms for A / B (forwarded to Mainloop)
//   S2RAtomA/B_ : shared->register copy atoms for A / B (forwarded to Mainloop)
//   MmaAtom_    : warp-level MMA atom (exposes kMmaM/kMmaN/kMmaK/kAccumPerThread)
//   Epilogue_   : accumulator->global store (epilogue::Epilogue<...>)
//   BLOCK_M/N/K : CTA tile shape
//   WARP_M/N    : warp tile shape
//   STAGES      : number of pipeline stages (>= 2)
//   DenseS2Opt  : host-selected pipeline-overlap opt, forwarded to the Mainloop's
//                 wait schedule. Mirrors the cutlass3 DispatchPolicy's third
//                 parameter (MainloopPPUAiuOpt<Stages, Schedule, DenseS2Opt>) so
//                 both paths react to the same host decision.
//   GemmTypeTag : std::integral_constant<GemmType, ...>
//
// NOTE: SHAPE_N / SHAPE_K / NUM_GROUPS are deliberately NOT template parameters.
// The DenseGemm path passes the problem shape at RUNTIME (params.scheduler holds
// shape_m/shape_n/shape_k), matching DenseGemmScheduler. The mainloop and epilogue
// already took M/N/K from Params, so only the scheduler needed the change.
// =============================================================================

template <typename Element_,
          typename G2SAtomA_, typename G2SAtomB_,
          typename S2RAtomA_, typename S2RAtomB_,
          typename MmaAtom_, typename Epilogue_,
          int BLOCK_M, int BLOCK_N, int BLOCK_K,
          int WARP_M, int WARP_N, int WARP_K, int STAGES,
          bool DenseS2Opt, bool OverlapPrologue,
          typename GemmTypeTag>
struct GemmKernel {
  // ---- Injected atoms / types ----
  using Element  = Element_;
  using G2SAtomA = G2SAtomA_;
  using G2SAtomB = G2SAtomB_;
  using S2RAtomA = S2RAtomA_;
  using S2RAtomB = S2RAtomB_;
  using MmaAtom  = MmaAtom_;
  using Epilogue = Epilogue_;

  // Collective mainloop assembled from the injected copy / mma atoms.
  using Mainloop = mainloop::Mainloop<
      Element, G2SAtomA, G2SAtomB, S2RAtomA, S2RAtomB, MmaAtom,
      BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, STAGES, DenseS2Opt, OverlapPrologue>;

  static constexpr GemmType kGemmType = GemmTypeTag::value;

  static_assert(kGemmType == GemmType::DenseGemm,
                "GemmKernel currently supports DenseGemm only");

  static constexpr int kMmaM = MmaAtom::kMmaM;
  static constexpr int kMmaN = MmaAtom::kMmaN;
  static constexpr int kMmaK = MmaAtom::kMmaK;
  static constexpr int kAccumPerThread = MmaAtom::kAccumPerThread;

  static constexpr int kWarpsM = BLOCK_M / WARP_M;
  static constexpr int kWarpsN = BLOCK_N / WARP_N;
  // WarpOnK: kWarpsK warp groups each cover WARP_K of the CTA's BLOCK_K.
  static constexpr int kWarpsK  = BLOCK_K / WARP_K;
  static constexpr int kWarpsMN = kWarpsM * kWarpsN;
  static constexpr int kThreads = kWarpsMN * kWarpsK * 32;

  static constexpr int kMmasPerWarpM = WARP_M / kMmaM;
  static constexpr int kMmasPerWarpN = WARP_N / kMmaN;
  static constexpr int kKSteps = BLOCK_K / kMmaK;

  static_assert(BLOCK_M % WARP_M == 0, "BLOCK_M must be a multiple of WARP_M");
  static_assert(BLOCK_N % WARP_N == 0, "BLOCK_N must be a multiple of WARP_N");
  static_assert(WARP_M % kMmaM == 0, "WARP_M must be a multiple of kMmaM");
  static_assert(WARP_N % kMmaN == 0, "WARP_N must be a multiple of kMmaN");
  static_assert(BLOCK_K % kMmaK == 0, "BLOCK_K must be a multiple of kMmaK");
  static_assert(BLOCK_K % WARP_K == 0, "BLOCK_K must be a multiple of WARP_K");
  static_assert(STAGES >= 2, "STAGES must be >= 2 for the async pipeline");

  // Must stay byte-identical to the host-side CuteFreeParams in
  // csrc/jit_kernels/impls/bf16_gemm.hpp. `scheduler` now carries the runtime
  // shape (m/n/k) because the DenseGemm scheduler is no longer templated on them.
  struct Params {
    Element const* ptr_A;
    Element const* ptr_B;
    Element* ptr_D;
    int M, N, K;
    int lda, ldb, ldd;
    DenseGemmTileSchedulerArguments scheduler;
  };

  // Shared memory: no padding (swizzled layout handled by AIU hardware)
  // A: STAGES * BLOCK_M * BLOCK_K elements
  // B: STAGES * BLOCK_N * BLOCK_K elements
  struct SharedStorage {
    alignas(128) Element smem[STAGES * (BLOCK_M + BLOCK_N) * BLOCK_K];
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static constexpr int MaxThreadsPerBlock = kThreads;
  static constexpr int MinBlocksPerMultiprocessor = 1;

  // The WarpOnK reduction reuses the mainloop's A/B SMEM as float scratch.
  static constexpr int kSmemFloatCapacity =
      int(sizeof(SharedStorage) / sizeof(float));
  using WarpKReduce = epilogue::WarpKReduce<
      kWarpsK, kWarpsMN, kMmasPerWarpM, kMmasPerWarpN, kAccumPerThread,
      kSmemFloatCapacity>;

  static_assert(kKSteps >= 2, "kKSteps must be >= 2 for G2S pipeline interleaving");

  __device__ void operator()(Params const& params, char* smem) {

    using Scheduler = DenseGemmScheduler<BLOCK_M, BLOCK_N>;
    Scheduler scheduler(params.scheduler);

    int warp_id = __shfl_sync(0xFFFFFFFFU, threadIdx.x / 32, 0);
    int lane = threadIdx.x % 32;
    int warp_mn  = warp_id % kWarpsMN;
    int warp_k   = warp_id / kWarpsMN;
    int warp_row = warp_mn / kWarpsN;
    int warp_col = warp_mn % kWarpsN;

    typename Mainloop::Params ml_params{
        params.ptr_A, params.ptr_B, params.M, params.N, params.K,
        params.lda, params.ldb};

    if constexpr (OverlapPrologue) {
      // --- Overlap path: hoist first tile's stage-0 G2S, then interleave
      //     next-tile prologue with current-tile epilogue. ---
      uint32_t m_block = 0, n_block = 0;
      bool tile_valid = scheduler.fetch_next_work(m_block, n_block);
      int m_offset = static_cast<int>(m_block) * BLOCK_M;
      int n_offset = static_cast<int>(n_block) * BLOCK_N;

      // Hoist first tile's stage-0 G2S
      if (tile_valid) {
        Mainloop::issue_prologue_stage0(smem, warp_id, ml_params, m_offset, n_offset);
      }

      while (tile_valid) {
        float accum[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread] = {};
        Mainloop::run(ml_params, smem, warp_id, warp_row, warp_col, warp_k,
                      m_offset, n_offset, accum);
        WarpKReduce::run(accum, reinterpret_cast<float*>(smem), warp_mn, warp_k, lane);

        // Fetch next tile + issue its stage-0 (overlaps epilogue below)
        tile_valid = scheduler.fetch_next_work(m_block, n_block);
        int m_next = static_cast<int>(m_block) * BLOCK_M;
        int n_next = static_cast<int>(n_block) * BLOCK_N;
        if (tile_valid) {
          Mainloop::issue_prologue_stage0(smem, warp_id, ml_params, m_next, n_next);
        }

        // Epilogue (NoTsm: no SMEM usage, safe to run with G2S in flight)
        if (warp_k == 0) {
          Epilogue::store(m_offset, n_offset, warp_row, warp_col, lane,
                          params.ptr_D, params.ldd, params.M, params.N, accum);
        }
        m_offset = m_next;
        n_offset = n_next;
      }
    } else {
      // --- Original non-overlap path (unchanged logic) ---
      uint32_t m_block = 0, n_block = 0;
      while (scheduler.fetch_next_work(m_block, n_block)) {
        int m_offset = static_cast<int>(m_block) * BLOCK_M;
        int n_offset = static_cast<int>(n_block) * BLOCK_N;
        float accum[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread] = {};
        Mainloop::run(ml_params, smem, warp_id, warp_row, warp_col, warp_k,
                      m_offset, n_offset, accum);
        WarpKReduce::run(accum, reinterpret_cast<float*>(smem), warp_mn, warp_k, lane);
        if (warp_k == 0) {
          Epilogue::store(m_offset, n_offset, warp_row, warp_col, lane,
                          params.ptr_D, params.ldd, params.M, params.N, accum);
        }
      }
    }
  }
};

} // namespace kernel
} // namespace deep_gemm
