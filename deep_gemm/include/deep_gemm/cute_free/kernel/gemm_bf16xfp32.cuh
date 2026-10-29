#pragma once

#include <cstdint>
#include <type_traits>

#include <cutlass/cutlass.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory_ppu.h>
#include <deep_gemm/cute_free/arch/ppu_common.cuh>
#include <deep_gemm/cute_free/mainloop/dual_b_mainloop.cuh>
#include <deep_gemm/cute_free/epilogue/warp_k_reduce.cuh>
#include <deep_gemm/scheduler/densegemm_scheduler_cutlass3.cuh>
#include <deep_gemm/common/utils_cutlass3.cuh>

namespace deep_gemm {
namespace kernel {

// =============================================================================
// GemmBf16xFp32Kernel: dual-B variant of GemmKernel for BF16 x FP32 GEMM.
//
// The FP32 weight is supplied as two BF16 matrices (high / residual parts):
//
//     W_fp32 ~= W_high + W_low * low_scale        (e.g. low_scale = 1 / 256)
//     D       = A @ W_fp32^T
//             = A @ W_high^T + low_scale * (A @ W_low^T)
//
// All MMA products accumulate in FP32. For every output tile the kernel runs
// the shared cute-free K-pipeline ONCE, with each pipeline stage holding
// A + B_high + B_low together (mainloop::DualBMainloop, mirroring hpc-ops'
// SMEM layout that keeps the high/low pair resident per stage): A is
// streamed from gmem once per tile, each k_step's A fragment feeds two
// back-to-back MMAs into two independent FP32 accumulators, and the result
// is written once through the standard epilogue after an in-register combine.
//
// Compared with two back-to-back BF16 GEMMs, this fuses the combination into
// the kernel: A is streamed once (not twice), the intermediate [M, N]
// round-trip through global memory is eliminated, and only one pipeline
// warm-up / drain is paid per tile.
//
// Differences vs. kernel::GemmKernel:
//   - Params carries two B pointers (ptr_B_high / ptr_B_low) and low_scale.
//   - SMEM holds an extra B operand per stage:
//       STAGES * (BLOCK_M + 2 * BLOCK_N) * BLOCK_K elements
//     (the host launcher must size shared memory from this formula).
//   - No OverlapPrologue variant: the next-tile prologue overlap assumes a
//     single mainloop pass per tile and does not compose with the dual-B
//     sequence. The host therefore never sets it for this kernel.
//
// Layout contract (identical to the BF16 dense NT path):
//   A          : row-major [M, K], lda = K
//   B_high/low : row-major [N, K], ldb = K   (i.e. B^T is column-major)
//   D          : row-major [M, N], ldd = N
//
// Accumulation precision: every MMA accumulates in FP32; the high/low
// combination (accum += low_scale * accum_low) is an FP32 FMA. Only the
// final store converts to ElementOutput (BF16 or FP32).
// =============================================================================

template <typename Element_,
          typename G2SAtomA_, typename G2SAtomB_,
          typename S2RAtomA_, typename S2RAtomB_,
          typename MmaAtom_, typename Epilogue_,
          int BLOCK_M, int BLOCK_N, int BLOCK_K,
          int WARP_M, int WARP_N, int WARP_K, int STAGES,
          bool DenseS2Opt, typename GemmTypeTag>
struct GemmBf16xFp32Kernel {
  // ---- Injected atoms / types ----
  using Element  = Element_;
  using G2SAtomA = G2SAtomA_;
  using G2SAtomB = G2SAtomB_;
  using S2RAtomA = S2RAtomA_;
  using S2RAtomB = S2RAtomB_;
  using MmaAtom  = MmaAtom_;
  using Epilogue = Epilogue_;
  using ElementOutput = typename Epilogue::ElementOutput;

  // Collective mainloop assembled from the injected copy / mma atoms; each
  // stage carries A + B_high + B_low together (see DualBMainloop).
  using Mainloop = mainloop::DualBMainloop<
      Element, G2SAtomA, G2SAtomB, S2RAtomA, S2RAtomB, MmaAtom,
      BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, STAGES, DenseS2Opt>;

  static constexpr GemmType kGemmType = GemmTypeTag::value;

  static_assert(kGemmType == GemmType::Normal,
                "GemmBf16xFp32Kernel currently supports DenseGemm only");

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

  // Must stay byte-identical to the host-side Bf16xFp32Params in
  // csrc/jit_kernels/impls/bf16xfp32_gemm.hpp (the kernel receives it by
  // value; the driver copies sizeof(device Params) bytes).
  struct Params {
    Element const* ptr_A;
    Element const* ptr_B_high;
    Element const* ptr_B_low;
    ElementOutput* ptr_D;
    float low_scale;
    int M, N, K;
    int lda, ldb, ldd;
    DenseGemmTileSchedulerArguments scheduler;
  };

  // Shared memory: one extra B operand per stage vs the single-B kernel
  // ([A stages | B_high stages | B_low stages]). The host launcher sizes
  // dynamic shared memory from the same formula.
  struct SharedStorage {
    alignas(128) Element smem[STAGES * (BLOCK_M + 2 * BLOCK_N) * BLOCK_K];
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
        params.ptr_A, params.ptr_B_high, params.ptr_B_low,
        params.M, params.N, params.K, params.lda, params.ldb};

    uint32_t m_block = 0, n_block = 0;
    while (scheduler.fetch_next_work(m_block, n_block)) {
      int m_offset = static_cast<int>(m_block) * BLOCK_M;
      int n_offset = static_cast<int>(n_block) * BLOCK_N;

      float accum_high[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread] = {};
      float accum_low[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread] = {};
      Mainloop::run(ml_params, smem, warp_id, warp_row, warp_col, warp_k,
                    m_offset, n_offset, accum_high, accum_low);

      // Combine the two FP32 accumulators before the (optional) WarpOnK
      // reduction: low_scale is uniform, so this is mathematically identical
      // to reducing first and scaling afterwards.
      #pragma unroll
      for (int mma_m = 0; mma_m < kMmasPerWarpM; ++mma_m) {
        #pragma unroll
        for (int mma_n = 0; mma_n < kMmasPerWarpN; ++mma_n) {
          #pragma unroll
          for (int i = 0; i < kAccumPerThread; ++i) {
            accum_high[mma_m][mma_n][i] +=
                params.low_scale * accum_low[mma_m][mma_n][i];
          }
        }
      }

      WarpKReduce::run(accum_high, reinterpret_cast<float*>(smem),
                       warp_mn, warp_k, lane);

      if (warp_k == 0) {
        Epilogue::store(m_offset, n_offset, warp_row, warp_col, lane,
                        params.ptr_D, params.ldd, params.M, params.N,
                        accum_high);
      }
    }
  }
};

} // namespace kernel
} // namespace deep_gemm
