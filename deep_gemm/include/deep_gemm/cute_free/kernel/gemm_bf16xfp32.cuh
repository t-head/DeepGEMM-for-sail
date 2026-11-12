#pragma once

#include <cstdint>
#include <type_traits>

#include <cutlass/cutlass.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory_ppu.h>
#include <deep_gemm/cute_free/arch/ppu_common.cuh>
#include <deep_gemm/cute_free/mainloop/dual_b_mainloop.cuh>
#include <deep_gemm/scheduler/densegemm_scheduler_cutlass3.cuh>
#include <deep_gemm/common/utils_cutlass3.cuh>

#ifndef DG_DUAL_B_OVERLAP_PROLOGUE
#define DG_DUAL_B_OVERLAP_PROLOGUE 0
#endif
#ifndef DG_DUAL_B_VALID_EPILOGUE
#define DG_DUAL_B_VALID_EPILOGUE 0
#endif
#ifndef DG_DUAL_B_ROUTER_M
#define DG_DUAL_B_ROUTER_M 0
#define DG_DUAL_B_ROUTER_N 0
#define DG_DUAL_B_ROUTER_K 0
#endif

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
//   - Fusion-only experiments can overlap the next output tile's three-operand stage0 with the current output store.
//     Cross-tile prefetch and valid-region merging default to off and leave the standard BF16 path unchanged.
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
  // Dual-B fusion forbids K splitting: both accumulators in a warp independently cover full K.
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
  static_assert(WARP_K == BLOCK_K, "dual-B fused kernel forbids K splitting");
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

  // Fusion-only output: merge high/low only at valid rows and columns; leave the shared standard BF16 epilogue unchanged.
  static __device__ __forceinline__ void store_valid_dual(
      Params const& params, int m_offset, int n_offset,
      int warp_row, int warp_col, int lane,
      float const (&high)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread],
      float const (&low)[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread]) {
    using Traits = typename Epilogue::ElemTraits;
    using Storage = typename Epilogue::StorageType;
    using Packed = typename Epilogue::PackedType;
    const int first_row = m_offset + warp_row * WARP_M + lane / Epilogue::kThreadsCol;
    const int first_col = n_offset + warp_col * WARP_N
                        + (lane % Epilogue::kThreadsCol) * Epilogue::kColsPerThread;
    #pragma unroll
    for (int mm = 0; mm < kMmasPerWarpM; ++mm) {
      #pragma unroll
      for (int half_m = 0; half_m < 2; ++half_m) {
        const int row = first_row + mm * kMmaM + half_m * Epilogue::kHalfMmaM;
        if (row < params.M) {
          Storage* row_ptr = reinterpret_cast<Storage*>(params.ptr_D)
                             + int64_t(row) * params.ldd + first_col;
          #pragma unroll
          for (int nn = 0; nn < kMmasPerWarpN; ++nn) {
            #pragma unroll
            for (int half_n = 0; half_n < 2; ++half_n) {
              const int col = first_col + nn * kMmaN + half_n * Epilogue::kHalfMmaN;
              if (Epilogue::kIsAlignedN || col + Traits::kVecWidth - 1 < params.N) {
                const int i = 2 * half_m + 4 * half_n;
                const float lo = high[mm][nn][i] + params.low_scale * low[mm][nn][i];
                const float hi = high[mm][nn][i + 1] + params.low_scale * low[mm][nn][i + 1];
                *reinterpret_cast<Packed*>(row_ptr + nn * kMmaN + half_n * Epilogue::kHalfMmaN)
                    = Traits::cvt_pack(hi, lo);
              }
            }
          }
        }
      }
    }
  }

  __device__ void operator()(Params const& runtime_params, char* smem) {
    Params params = runtime_params;
    if constexpr (DG_DUAL_B_ROUTER_M != 0) {
      params.M = DG_DUAL_B_ROUTER_M;
      params.N = DG_DUAL_B_ROUTER_N;
      params.K = params.lda = params.ldb = DG_DUAL_B_ROUTER_K;
      params.ldd = DG_DUAL_B_ROUTER_N;
      params.scheduler.shape_m = DG_DUAL_B_ROUTER_M;
      params.scheduler.shape_n = DG_DUAL_B_ROUTER_N;
      params.scheduler.shape_k = DG_DUAL_B_ROUTER_K;
    }
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
    bool tile_valid = scheduler.fetch_next_work(m_block, n_block);
    if constexpr (DG_DUAL_B_OVERLAP_PROLOGUE) {
      if (tile_valid)
        Mainloop::issue_prologue_stage0(smem, warp_id, ml_params,
            static_cast<int>(m_block) * BLOCK_M, static_cast<int>(n_block) * BLOCK_N);
    }
    while (tile_valid) {
      int m_offset = static_cast<int>(m_block) * BLOCK_M;
      int n_offset = static_cast<int>(n_block) * BLOCK_N;

      float accum_high[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread] = {};
      float accum_low[kMmasPerWarpM][kMmasPerWarpN][kAccumPerThread] = {};
      Mainloop::run(ml_params, smem, warp_id, warp_row, warp_col, warp_k,
                    m_offset, n_offset, accum_high, accum_low, DG_DUAL_B_OVERLAP_PROLOGUE);

      // Current K is complete and SMEM is free; overlap the next tile's three-operand stage0 with merging and storing.
      if constexpr (DG_DUAL_B_OVERLAP_PROLOGUE) {
        tile_valid = scheduler.fetch_next_work(m_block, n_block);
        if (tile_valid)
          Mainloop::issue_prologue_stage0(smem, warp_id, ml_params,
              static_cast<int>(m_block) * BLOCK_M, static_cast<int>(n_block) * BLOCK_N);
      }
      if constexpr (DG_DUAL_B_VALID_EPILOGUE) {
        store_valid_dual(params, m_offset, n_offset, warp_row, warp_col, lane,
                         accum_high, accum_low);
      } else {
        // After both independent FP32 accumulators cover full K, apply the scaled merge only in registers.
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

        Epilogue::store(m_offset, n_offset, warp_row, warp_col, lane,
                        params.ptr_D, params.ldd, params.M, params.N,
                        accum_high);
      }
      if constexpr (!DG_DUAL_B_OVERLAP_PROLOGUE)
        tile_valid = scheduler.fetch_next_work(m_block, n_block);
    }
  }
};

#if defined(DG_DUAL_B_PACKED_N) && DG_DUAL_B_PACKED_N
#ifndef DG_DUAL_B_DECODE_SPLIT_K
#define DG_DUAL_B_DECODE_SPLIT_K 1
#endif
// Small-M router experiment: pack 8 high columns and 8 low columns into the two N halves of one MMA.
// Each half independently accumulates its K range in FP32; optional CTA-local partitions reduce high/low separately before the store.
template <typename ElementOutput, int BLOCK_M, int BLOCK_N, int BLOCK_K, int STAGES>
struct PackedDualNRouterKernel {
  using Element = cutlass::bfloat16_t;
  using Mma = mma::Bf16MmaAtom;
  using S2RA = copy::TsmS2RAtom<Element, BLOCK_M, BLOCK_K, false>;
  using S2RB = copy::TsmS2RAtom<Element, BLOCK_N, BLOCK_K, true>;
  static_assert(std::is_same_v<ElementOutput, float>, "packed router requires FP32 output");
  static_assert(BLOCK_K % 64 == 0 && STAGES >= 2, "packed router requires complete K tiles");
  static_assert(BLOCK_M % 16 == 0 && BLOCK_N % 16 == 0, "packed router uses complete physical MMA tiles");
  static constexpr int kSplitK = DG_DUAL_B_DECODE_SPLIT_K;
  static_assert(kSplitK == 1 || kSplitK == 2 || kSplitK == 4 || kSplitK == 8,
                "decode K partition count must be a power of two up to eight");
  static constexpr int MaxThreadsPerBlock = (BLOCK_M / 16) * (BLOCK_N / 16) * kSplitK * 32;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int kSteps = BLOCK_K / 16;
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
  struct SharedStorage {
    alignas(128) Element smem[kSplitK * STAGES * (BLOCK_M + BLOCK_N) * BLOCK_K];
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  static __device__ __forceinline__ void load_stage(
      Element* a, Element* b, Params const& p, int mo, int no, int ko) {
    #pragma unroll
    for (int j = 0; j < BLOCK_K / 64; ++j) {
      arch::aiu_bulk_tensor_load_bf16<BLOCK_M, 64>(a + j * BLOCK_M * 64,
          p.ptr_A, p.M, p.lda, mo, ko + j * 64);
      // Each swizzle subtile stores high in the first 8 rows and low in the last 8; weights are not prepacked before timing.
      #pragma unroll
      for (int q = 0; q < BLOCK_N / 16; ++q) {
        arch::aiu_bulk_tensor_load_bf16<8, 64>(b + j * BLOCK_N * 64 + q * 16 * 64,
            p.ptr_B_high, p.N, p.ldb, no + q * 8, ko + j * 64);
        arch::aiu_bulk_tensor_load_bf16<8, 64>(b + j * BLOCK_N * 64 + q * 16 * 64 + 8 * 64,
            p.ptr_B_low, p.N, p.ldb, no + q * 8, ko + j * 64);
      }
    }
  }

  template <int Pair>
  static __device__ __forceinline__ void run_pairs(
      Params const& p, Element* a, Element* b, int stage, int next,
      int mo, int no, int next_kt, bool issue, int base_a, int base_b,
      int warp_id, int warp_row, int warp_col,
      float (&acc)[8], uint32_t (&ar)[2][4], uint32_t (&br)[2][4]) {
    if constexpr (Pair < kSteps / 2) {
      S2RA::load(base_a, 2 * Pair + 1, ar[1]);
      S2RB::load(base_b, 2 * Pair + 1, br[1]);
      Mma::mma(acc, ar[0], br[0], acc);
      if constexpr (Pair == kSteps / 2 - 1) {
        cutlass::arch::cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (issue && warp_id == 0)
          load_stage(a + stage * BLOCK_M * BLOCK_K, b + stage * BLOCK_N * BLOCK_K,
                     p, mo, no, next_kt * BLOCK_K);
        cutlass::arch::cp_async_fence();
        if constexpr (STAGES == 2) {
          cutlass::arch::cp_async_wait<1>();
          __syncthreads();
        }
        int na = arch::smem_ptr_to_tsm_base(a + next * BLOCK_M * BLOCK_K, warp_row * 16 * 8);
        int nb = arch::smem_ptr_to_tsm_base(b + next * BLOCK_N * BLOCK_K, warp_col * 16 * 8);
        S2RA::load(na, 0, ar[0]);
        S2RB::load(nb, 0, br[0]);
      } else {
        S2RA::load(base_a, 2 * Pair + 2, ar[0]);
        S2RB::load(base_b, 2 * Pair + 2, br[0]);
      }
      Mma::mma(acc, ar[1], br[1], acc);
      run_pairs<Pair + 1>(p, a, b, stage, next, mo, no, next_kt, issue,
                          base_a, base_b, warp_id, warp_row, warp_col, acc, ar, br);
    }
  }

  __device__ void operator()(Params const& runtime_params, char* smem) {
    Params p = runtime_params;
    if constexpr (DG_DUAL_B_ROUTER_M != 0) {
      p.M = DG_DUAL_B_ROUTER_M;
      p.N = DG_DUAL_B_ROUTER_N;
      p.K = p.lda = p.ldb = DG_DUAL_B_ROUTER_K;
      p.ldd = DG_DUAL_B_ROUTER_N;
      p.scheduler.shape_m = DG_DUAL_B_ROUTER_M;
      p.scheduler.shape_n = DG_DUAL_B_ROUTER_N;
      p.scheduler.shape_k = DG_DUAL_B_ROUTER_K;
    }
    DenseGemmScheduler<BLOCK_M, BLOCK_N / 2> scheduler(p.scheduler);
    const int lane = threadIdx.x % 32;
    const int physical_warp = __shfl_sync(0xFFFFFFFFU, threadIdx.x / 32, 0);
    const int split_id = physical_warp % kSplitK;
    const int warp_id = physical_warp / kSplitK;
    const int warp_row = warp_id / (BLOCK_N / 16);
    const int warp_col = warp_id % (BLOCK_N / 16);
    Element* a = reinterpret_cast<Element*>(smem)
        + split_id * STAGES * (BLOCK_M + BLOCK_N) * BLOCK_K;
    Element* b = a + STAGES * BLOCK_M * BLOCK_K;
    uint32_t mb = 0, nb = 0;
    while (scheduler.fetch_next_work(mb, nb)) {
      const int mo = int(mb) * BLOCK_M, no = int(nb) * (BLOCK_N / 2);
      // All partitions execute the same padded tile count, so CTA barriers remain convergent.
      // Tensor loads zero-fill the final global-K tail; no valid K element is omitted or duplicated.
      const int tiles = math::ceil_div(p.K, BLOCK_K * kSplitK);
      const int first_tile = split_id * tiles;
      int loaded = 0;
      #pragma unroll
      for (int s = 0; s < STAGES && loaded < tiles; ++s) {
        if (warp_id == 0)
          load_stage(a + s * BLOCK_M * BLOCK_K, b + s * BLOCK_N * BLOCK_K,
                     p, mo, no, (first_tile + loaded) * BLOCK_K);
        cutlass::arch::cp_async_fence();
        ++loaded;
      }
      if (loaded >= STAGES) cutlass::arch::cp_async_wait<STAGES - 1>();
      else cutlass::arch::cp_async_wait<0>();
      __syncthreads();
      float acc[8] = {};
      uint32_t ar[2][4], br[2][4];
      int stage = 0;
      for (int kt = 0; kt < tiles; ++kt) {
        int next = stage + 1 == STAGES ? 0 : stage + 1;
        int ba = arch::smem_ptr_to_tsm_base(a + stage * BLOCK_M * BLOCK_K, warp_row * 16 * 8);
        int bb = arch::smem_ptr_to_tsm_base(b + stage * BLOCK_N * BLOCK_K, warp_col * 16 * 8);
        if (kt == 0) {
          S2RA::load(ba, 0, ar[0]);
          S2RB::load(bb, 0, br[0]);
        }
        bool issue = loaded < tiles;
        run_pairs<0>(p, a, b, stage, next, mo, no, first_tile + loaded, issue,
                     ba, bb, warp_id, warp_row, warp_col, acc, ar, br);
        if (issue) ++loaded;
        stage = next;
      }
      if constexpr (kSplitK > 1) {
        cutlass::arch::cp_async_wait<0>();
        __syncthreads();
        // Reuse drained operand SMEM for deterministic FP32 partials; no global workspace or atomics.
        static_assert(SharedStorageSize >= MaxThreadsPerBlock * 8 * int(sizeof(float)),
                      "operand storage must fit all partial accumulators");
        float* partials = reinterpret_cast<float*>(smem);
        #pragma unroll
        for (int i = 0; i < 8; ++i) partials[threadIdx.x * 8 + i] = acc[i];
        __syncthreads();
        if (split_id == 0) {
          #pragma unroll
          for (int part = 1; part < kSplitK; ++part) {
            #pragma unroll
            for (int i = 0; i < 8; ++i)
              acc[i] += partials[((warp_id * kSplitK + part) * 32 + lane) * 8 + i];
          }
        }
      }
      #pragma unroll
      for (int half = 0; half < 2; ++half) {
        int row = mo + warp_row * 16 + lane / 4 + half * 8;
        int col = no + warp_col * 8 + (lane % 4) * 2;
        if (split_id == 0 && row < p.M && col + 1 < p.N) {
          const int i = half * 2;
          *reinterpret_cast<float2*>(p.ptr_D + int64_t(row) * p.ldd + col) =
              make_float2(acc[i] + p.low_scale * acc[i + 4],
                          acc[i + 1] + p.low_scale * acc[i + 5]);
        }
      }
      if constexpr (kSplitK > 1) __syncthreads();
    }
  }
};
#endif

} // namespace kernel
} // namespace deep_gemm
