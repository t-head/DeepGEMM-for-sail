#pragma once
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/epilogue_utils.cuh>
#include <deep_gemm/tix/cvt.cuh>

#include <cute/ppu_tensor_mix.hpp>
#include <cutlass/gemm/config/gemm_operands.hpp>
#include <cute/atom/copy_traits_ppu0010_aiu.hpp>
#include <cute/atom/copy_traits_ppu0015_aiu.hpp>
#include <cute/algorithm/ppu_copy.hpp>

#include <deep_gemm/common/fused_gemm_common.cuh>

// ============================================================
// Gemm_Hybrid_Operand: cp.async (GMEM->SMEM) + Swizzle<3,3,3> + SMEM->Reg
// 890P: TSM_LD_SWZL for SMEM->Reg; 810E: ldmatrix (same as TensorOpPPU)
// ============================================================
namespace cutlass {
namespace gemm {
namespace config {

template <typename ArchTag, typename Element, bool Trans,
          int Alignment, typename BLOCK_K, int ThreadNum, typename CopyInst,
          typename Block_MN>
struct Gemm_Hybrid_Operand;

// Trans=false (A matrix)
template <typename ArchTag, typename Element, int Alignment,
          typename BLOCK_K, int ThreadNum, typename CopyInst,
          typename Block_MN>
struct Gemm_Hybrid_Operand<ArchTag, Element, false,
                           Alignment, BLOCK_K, ThreadNum, CopyInst,
                           Block_MN> {
    // ---- GMEM->SMEM: cp.async (same as TensorOpPPU) ----
    static constexpr int ElemInLine = cute::min(int(BLOCK_K()), 128 / sizeof(Element));
    static constexpr int ThreadK = platform::min(ThreadNum, ElemInLine / Alignment);
    static constexpr int ThreadM = ThreadNum / ThreadK;
    using GmemTiledCopy = decltype(
        cute::make_tiled_copy(cute::Copy_Atom<CopyInst, Element>{},
                        cute::Layout<cute::Shape<cute::Int<ThreadM>, cute::Int<ThreadK>>,
                               cute::Stride<cute::Int<ThreadK>, cute::_1>>{},
                        cute::Layout<cute::Shape<cute::_1, cute::Int<Alignment>>>{}));

    // ---- SMEM Layout Atom: Swizzle<3,3,3> (same for both architectures) ----
    static constexpr int kFactor = 128 / ElemInLine / sizeof(Element);
    static constexpr int SwizzleIdx = kFactor == 1 ? 3 : (kFactor == 2 ? 2 : 1);
    static constexpr int SwizzleAtom = sizeof(Element) == 4 ? 2
                                     : sizeof(Element) == 2 ? 3 : 4;

    using BaseLayoutAtom = cute::Layout<cute::Shape<cute::_8, cute::Int<ElemInLine>>,
                                  cute::Stride<cute::Int<ElemInLine>, cute::_1>>;
    using SmemLayoutAtom = decltype(cute::composition(
        cute::Swizzle<SwizzleIdx, SwizzleAtom, 3>{}, BaseLayoutAtom{}));

    // ---- SMEM->Register: architecture-dependent ----
    static constexpr int InstNum = int(BLOCK_K()) / ElemInLine;
    static constexpr int CUBE_H = int(Block_MN{});
    static constexpr int CUBE_W = ElemInLine;

#if __HGGC_ARCH__ >= 150
    using SmemCopyOp = PPU0015_TSM_LD_SWZL<Element, CUBE_H, CUBE_W, false, false, InstNum>;
#else
    using SmemCopyOp = PPU_U32x4_LDSM_N;
#endif
    using SmemCopyAtom = cute::Copy_Atom<SmemCopyOp, Element>;
};

} // namespace config
} // namespace gemm
} // namespace cutlass

namespace deep_gemm {
using cute::_;

// GemmSmemConfig / BlkwiseQuantGemmSmemConfig live in common/fused_gemm_common.cuh
// (shared verbatim with the C++ JIT host runtimes); GemmSmemConfigFp4 stays below.

template <int kNumStages, int BLOCK_M, int BLOCK_N, int BLOCK_K, typename MainloopFp4,
          uint32_t kEpilogueSmemSize = 0>
struct GemmSmemConfigFp4 {
  // A is copied with cp.sync
  // the size of storage of cutlass::float4_t is uint8;
  static constexpr uint32_t kSmemASize = cute::round_up(kNumStages * BLOCK_M * BLOCK_K * sizeof(cutlass::float4_t), 128);
  // B is copied with aiu
  static constexpr uint32_t kSmemBSize = kNumStages * BLOCK_N * BLOCK_K * sizeof(cutlass::float4_t);

  static constexpr auto kSmemSFASize = cute::cosize_v<typename MainloopFp4::SmemLayoutSFA> * sizeof(typename MainloopFp4::ElementSFA);
  static constexpr auto kSmemSFBSize = cute::cosize_v<typename MainloopFp4::SmemLayoutSFB> * sizeof(typename MainloopFp4::ElementSFB);

  // A fused epilogue reuses the A/B segments as its activation tile, so the scale segments start
  // past whichever of the two is larger. That keeps the one-time SFA zero fill valid across
  // scheduler iterations, and only an epilogue larger than A+B costs extra shared memory.
  static constexpr uint32_t kSmemEpilogueSize = cute::round_up(kEpilogueSmemSize, 128);
  static constexpr uint32_t kSmemScaleOffset =
      (kSmemASize + kSmemBSize) > kSmemEpilogueSize ? (kSmemASize + kSmemBSize) : kSmemEpilogueSize;

  static constexpr uint32_t kTotalSize = kSmemScaleOffset + kSmemSFASize + kSmemSFBSize;
};

// Full copies skip invalid rows; clear their shared slots once before the K loop.
struct FusedACopyInst {
  CUTE_HOST_DEVICE static void copy(const cutlass::uint128_t& src, cutlass::uint128_t& dst, bool valid) {
    const uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&dst);
    asm volatile("{\n.reg .pred p;\n"
                 "ppu.setp.ne.u32 p, %2, 0;\n"
                 "@p ppu.cp.async.ca.shared.global.LLC::128B [%0], [%1], 16;\n}"
                 :: "r"(smem_addr), "l"(&src), "r"(uint32_t(valid)));
  }
};
struct FusedAClearInst {
  CUTE_HOST_DEVICE static void copy(const cutlass::uint128_t&, cutlass::uint128_t& dst, bool valid) {
    if (!valid) dst = cutlass::uint128_t{};
  }
};

template <typename SrcT, typename ACopyInst, typename TilerA,
         uint32_t BLOCK_M, uint32_t BLOCK_K, uint32_t STRIDE_AM,
         class TAsA, class... Ts>
__forceinline__ __device__ void copy_A_to_tsm(TAsA&& tAsA, const void* src_ptr, const int* blk_token_base,
                                              uint32_t stage_offset, uint32_t shape_m, uint32_t thread_idx) {
  static constexpr uint32_t TILER_M = cute::get<0>(TilerA{});
  static constexpr uint32_t TILER_N = cute::get<1>(TilerA{});
  static constexpr uint32_t KPerThread = 128 / cutlass::sizeof_bits<SrcT>::value;
  static constexpr uint32_t NumThreads_CPY_K = TILER_N / KPerThread; // max cont is 128B
  static constexpr uint32_t M_ITER = cute::ceil_div(BLOCK_M, TILER_M);
  static constexpr uint32_t K_ITER = BLOCK_K / TILER_N;

  uint32_t tid_k = thread_idx % NumThreads_CPY_K;
  uint32_t tid_m = thread_idx / NumThreads_CPY_K;
  if (tid_m < BLOCK_M) {
    CUTLASS_PRAGMA_UNROLL
    for(uint32_t m_iter = 0; m_iter < M_ITER; ++m_iter) {
      CUTLASS_PRAGMA_UNROLL
      for(uint32_t k_iter = 0; k_iter < K_ITER; ++k_iter) {
        uint32_t token_offset = __ldg(blk_token_base + (m_iter * TILER_M) + tid_m);
        bool token_mask = token_offset < shape_m;
        const cutlass::uint128_t* src_ptr128 = reinterpret_cast<const cutlass::uint128_t*>(reinterpret_cast<const SrcT*>(src_ptr)
              + token_offset * STRIDE_AM + (k_iter * TILER_N) + tid_k * KPerThread + stage_offset);
        cutlass::uint128_t* dst_ptr128 = reinterpret_cast<cutlass::uint128_t*>(cute::raw_pointer_cast(tAsA(_,m_iter,k_iter).data()));
        ACopyInst::copy(*src_ptr128, *dst_ptr128, token_mask);
      }
    }
  }
};

// Support prefetch token offsets and rebalance threads for A copy
template <typename SrcT, typename TilerA,
          uint32_t BLOCK_M, uint32_t BLOCK_K, int THREAD_NUM>
struct CopyAToTsmConfig {
  static constexpr uint32_t TILER_M = cute::get<0>(TilerA{});
  static constexpr uint32_t TILER_N = cute::get<1>(TilerA{});
  static constexpr uint32_t KPerThread = 128 / cutlass::sizeof_bits<SrcT>::value;
  static constexpr uint32_t NumThreads_CPY_K = TILER_N / KPerThread;
  static constexpr uint32_t ActiveM = cute::min(BLOCK_M, TILER_M);
  static constexpr uint32_t ThreadsPerKIter = ActiveM * NumThreads_CPY_K;
  static constexpr uint32_t M_ITER = cute::ceil_div(BLOCK_M, TILER_M);
  static constexpr uint32_t K_ITER = BLOCK_K / TILER_N;
  static_assert(THREAD_NUM != 0, "THREAD_NUM must be non-zero for CopyAToTsmConfig");
  static constexpr uint32_t MaxKCohorts = uint32_t(THREAD_NUM) / ThreadsPerKIter;
  static constexpr bool Rebalance = BLOCK_M < TILER_M && K_ITER > 1 &&
      ThreadsPerKIter % 32 == 0 && MaxKCohorts > 1;
  static constexpr uint32_t KCohorts = Rebalance ? cute::min(K_ITER, MaxKCohorts) : 1;

  __forceinline__ __device__ static uint32_t logical_thread_idx(uint32_t thread_idx) {
    return Rebalance ? thread_idx % ThreadsPerKIter : thread_idx;
  }
};

template <typename SrcT, typename TilerA,
          uint32_t BLOCK_M, uint32_t BLOCK_K, int THREAD_NUM,
          uint32_t M_ITER>
__forceinline__ __device__ void prefetch_A_token_offsets(
    uint32_t (&token_offsets)[M_ITER], const int* blk_token_base,
    uint32_t shape_m, uint32_t thread_idx) {
  using Config = CopyAToTsmConfig<SrcT, TilerA, BLOCK_M, BLOCK_K, THREAD_NUM>;
  static_assert(M_ITER == Config::M_ITER, "token offset buffer size must match A copy M iterations");
  uint32_t logical_thread_idx = Config::logical_thread_idx(thread_idx);
  uint32_t k_cohort = Config::Rebalance ? thread_idx / Config::ThreadsPerKIter : 0;
  uint32_t tid_m = logical_thread_idx / Config::NumThreads_CPY_K;
  bool active_copy_thread = k_cohort < Config::KCohorts && tid_m < BLOCK_M;
  CUTLASS_PRAGMA_UNROLL
  for (uint32_t m_iter = 0; m_iter < Config::M_ITER; ++m_iter) {
    token_offsets[m_iter] = active_copy_thread
        ? __ldg(blk_token_base + m_iter * Config::TILER_M + tid_m) : shape_m;
  }
}

template <typename SrcT, typename ACopyInst, typename TilerA,
          uint32_t BLOCK_M, uint32_t BLOCK_K, uint32_t STRIDE_AM,
          int THREAD_NUM, class TAsA>
__forceinline__ __device__ void copy_A_to_tsm_impl(
    TAsA&& tAsA, const void* src_ptr, const uint32_t* token_offsets,
    uint32_t stage_offset, uint32_t shape_m, uint32_t thread_idx) {
  using Config = CopyAToTsmConfig<SrcT, TilerA, BLOCK_M, BLOCK_K, THREAD_NUM>;
  uint32_t logical_thread_idx = Config::logical_thread_idx(thread_idx);
  uint32_t k_cohort = Config::Rebalance ? thread_idx / Config::ThreadsPerKIter : 0;
  uint32_t tid_k = logical_thread_idx % Config::NumThreads_CPY_K;
  uint32_t tid_m = logical_thread_idx / Config::NumThreads_CPY_K;
  if (k_cohort < Config::KCohorts && tid_m < BLOCK_M) {
    CUTLASS_PRAGMA_UNROLL
    for (uint32_t m_iter = 0; m_iter < Config::M_ITER; ++m_iter) {
      uint32_t token_offset = token_offsets[m_iter];
      bool token_mask = token_offset < shape_m;
      CUTLASS_PRAGMA_UNROLL
      for (uint32_t k_iter = k_cohort; k_iter < Config::K_ITER; k_iter += Config::KCohorts) {
        const cutlass::uint128_t* src_ptr128 = reinterpret_cast<const cutlass::uint128_t*>(
            reinterpret_cast<const SrcT*>(src_ptr) + token_offset * STRIDE_AM +
            k_iter * Config::TILER_N + tid_k * Config::KPerThread + stage_offset);
        cutlass::uint128_t* dst_ptr128 = reinterpret_cast<cutlass::uint128_t*>(
            cute::raw_pointer_cast(tAsA(_,m_iter,k_iter).data()));
        ACopyInst::copy(*src_ptr128, *dst_ptr128, token_mask);
      }
    }
  }
}

template <typename SrcT, typename ACopyInst, typename TilerA,
          uint32_t BLOCK_M, uint32_t BLOCK_K, uint32_t STRIDE_AM,
          int THREAD_NUM, class TAsA>
__forceinline__ __device__ void copy_A_to_tsm(
    TAsA&& tAsA, const void* src_ptr, const uint32_t* token_offsets,
    uint32_t stage_offset, uint32_t shape_m, uint32_t thread_idx,
    cute::Int<THREAD_NUM>) {
  copy_A_to_tsm_impl<SrcT, ACopyInst, TilerA, BLOCK_M, BLOCK_K, STRIDE_AM, THREAD_NUM>(
      static_cast<TAsA&&>(tAsA), src_ptr, token_offsets,
      stage_offset, shape_m, thread_idx);
}


template <typename SrcT, typename TilerA,
          uint32_t BLOCK_M, uint32_t BLOCK_K, uint32_t STRIDE_AM, int STAGES,
          int THREAD_NUM, class TAsA>
__forceinline__ __device__ void clear_invalid_A_to_tsm(
    TAsA&& tAsA, const void* src_ptr, const uint32_t* token_offsets,
    uint32_t shape_m, uint32_t thread_idx, cute::Int<THREAD_NUM> threads) {
  // Invalid rows stay zero across K; the caller's prologue barrier publishes these stores.
  CUTE_UNROLL
  for (int stage = 0; stage < STAGES; ++stage) {
    copy_A_to_tsm<SrcT, FusedAClearInst, TilerA, BLOCK_M, BLOCK_K, STRIDE_AM>(
        tAsA(_,_,_,stage), src_ptr, token_offsets, 0, shape_m, thread_idx, threads);
  }
}

template <typename AccT, typename DstT,
          uint32_t SHAPE_N, uint32_t BLOCK_N, uint32_t STRIDE_CM,
          typename EpilogueConfig, typename CopyAtomR2S,
          class TAcc, class TCcC, class TCC, class TileMMA, class... Ts>
__forceinline__ __device__ void epilogue_with_tsm(TAcc& accum, TCcC& tCcC, TCC& cC, TileMMA& tiled_mma,
            void* c_ptr, void* smem_buffer, uint32_t thread_idx,
            uint32_t c_row_base, uint32_t num_valid_rows, uint32_t blk_n_offset) {
  using SmemLayoutO = typename EpilogueConfig::SmemLayoutO;
  using TiledCopyS2R = typename EpilogueConfig::GmemTiledCopyO;
  using namespace cute;

  Tensor sAcc = make_tensor(make_smem_ptr(reinterpret_cast<DstT*>(smem_buffer)), SmemLayoutO{});

  // Partition sAcc to match the accumulator partitioning
  auto tiled_r2s  = make_tiled_copy_C(CopyAtomR2S{}, tiled_mma);
  auto thread_r2s = tiled_r2s.get_thread_slice(thread_idx);
  Tensor tRS_rAcc = thread_r2s.retile_S(accum);                        // ((Atom,AtomNum), MMA_M, MMA_N)
  Tensor tRS_sAcc = thread_r2s.partition_D(sAcc);                      // ((Atom,AtomNum),PIPE_M,PIPE_N)

  // Tile gC by the shape of SmemLayout first
  auto tile  = make_shape(size<0>(sAcc), size<1>(sAcc));

  // Partition sAcc, gC for the output
  auto tiled_s2r  = TiledCopyS2R{};
  auto thread_s2r = tiled_s2r.get_thread_slice(thread_idx);
  Tensor tSR_sAcc = thread_s2r.partition_S(sAcc);                      //               ((Atom,AtomNum),ATOM_M,ATOM_N)

  // Repeat the D-partitioning for coordinates and predication
  Tensor cCt  = flat_divide(cC, tile);                                 //                (SMEM_M,SMEM_N,TILE_M,TILE_N)
  Tensor tSR_cC = thread_s2r.partition_D(cCt);                         // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

  // Allocate intermediate registers on the dst tensors
  Tensor tSR_rAcc = make_tensor<DstT>(take<0,3>(shape(tSR_cC)));       // ((Atom,AtomNum),ATOM_M,ATOM_N)

  CUTE_STATIC_ASSERT(size<1>(tRS_rAcc) % size<3>(tSR_cC) == 0);  // TILE_M divides MMA_M
  CUTE_STATIC_ASSERT(size<2>(tRS_rAcc) % size<4>(tSR_cC) == 0);  // TILE_N divides MMA_N

  CUTLASS_PRAGMA_UNROLL
  for (int step_m = 0; step_m < size<2>(cCt); ++step_m) {
    CUTLASS_PRAGMA_UNROLL
    for (int step_n = 0; step_n < size<3>(cCt); ++step_n) {
      // Step 1. Copy to SMEM
      CUTLASS_PRAGMA_UNROLL
      for (int pipe_m = 0; pipe_m < size<1>(tRS_sAcc); ++pipe_m) {
        CUTLASS_PRAGMA_UNROLL
        for (int pipe_n = 0; pipe_n < size<2>(tRS_sAcc); ++pipe_n) {
          int mma_m = step_m * size<1>(tRS_sAcc) + pipe_m;
          int mma_n = step_n * size<2>(tRS_sAcc) + pipe_n;
          copy(tiled_r2s, tRS_rAcc(_,mma_m,mma_n), tRS_sAcc(_,pipe_m,pipe_n));
        }
      }
      // Step 2. Wait for SMEM writes to complete
      __syncthreads();

      // Step 3. Copy from SMEM into a fragment
      copy(tiled_s2r, tSR_sAcc, tSR_rAcc);

      // Step 4. Wait for SMEM reads to complete
      __syncthreads();

      Tensor tSR_cDmn = tSR_cC(_,_,_,step_m,step_n);
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < size<1>(tSR_cDmn); ++m) {
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < size<2>(tSR_cDmn); ++n) {
          size_t local_row = get<0>(tSR_cDmn(0,m,n));
          size_t n_offset = blk_n_offset + get<1>(tSR_cDmn(0,m,n));
          // Predication
          bool cond = local_row < num_valid_rows;
          if constexpr (SHAPE_N % BLOCK_N) {
            cond = cond &&  n_offset < SHAPE_N;
          }
          if (cond) {
            uint128_t* dst_ptr = (uint128_t*)((DstT*)c_ptr + (c_row_base + local_row) * STRIDE_CM + n_offset);
            uint128_t* acc_ptr = (uint128_t*)(raw_pointer_cast(tSR_rAcc(_,m,n).data()));
            *dst_ptr = *acc_ptr;
          }
        }
      }
    }
  }
};

// =============================================================================
// Single-kernel moe_align for numel <= 16384; expert-local order is unspecified.
// 1 block x 1024 threads.
// =============================================================================

template <int BLOCK_M, int kNumGroups, int kTopK, int BLOCK_SIZE = 1024>
__device__ __forceinline__ void
moe_align_single_block_kernel_impl(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ m_rows,
    int32_t* __restrict__ expert_ids_and_cumsum,
    int32_t* __restrict__ aligned_num_m_blocks,
    int32_t* __restrict__ inv_perm,
    int32_t* __restrict__ m_indices,
    int numel,
    int s_total_ub)
{
    constexpr int WARP_SIZE = 32;
    __shared__ int cursor[kNumGroups], padded[kNumGroups], compact[kNumGroups];
    __shared__ int2 totals[(kNumGroups + WARP_SIZE - 1) / WARP_SIZE];
    const int tid = threadIdx.x;
    // =======================================================================
    // Phase 1: Initialize expert counters and fill sorted_token_ids with PAD
    // =======================================================================
    if (tid < kNumGroups) cursor[tid] = 0;
    // Fill PAD via int4 stride-write (PAD value = shape_m = numel / kTopK)
    constexpr int shape_m_div = kTopK;  // divisor is compile-time constant
    int shape_m = numel / shape_m_div;
    int4 fill_val = make_int4(shape_m, shape_m, shape_m, shape_m);
    int4* out_vec = (int4*)sorted_token_ids;
    for (int i = threadIdx.x; i < s_total_ub / 4; i += BLOCK_SIZE) {
        out_vec[i] = fill_val;
    }
    for (int i = s_total_ub / 4 * 4 + threadIdx.x; i < s_total_ub; i += BLOCK_SIZE) {
        sorted_token_ids[i] = shape_m;
    }
    __syncthreads();
    // =======================================================================
    // Phase 2: Initialize m_indices, count routes and reserve expert-local slots
    // =======================================================================
    // offsets[c] stores the slot returned by atomicAdd for this thread's route.
    // Slots are unique within each expert; original route order is not preserved.
    // The C++ launcher selects this single block for at most 16,384 slots.
    static_assert(kNumGroups <= BLOCK_SIZE, "one thread per expert is required");
    constexpr int rounds = 16384 / BLOCK_SIZE;
    int offsets[rounds];
    #pragma unroll
    for (int c = 0; c < rounds; ++c) {
        const int i = tid + c * BLOCK_SIZE;
        if (c * BLOCK_SIZE >= numel) break;
        if (i < numel) m_indices[i] = -1;
        const int e = i < numel ? topk_ids[i] : -1;
        offsets[c] = (e >= 0 && e < kNumGroups) ? atomicAdd(&cursor[e], 1) : 0;
    }
    __syncthreads();
    // =======================================================================
    // Phase 3: Compute expert starts and emit GEMM block metadata
    // =======================================================================
    // One thread per expert: scan padded counts (p) and valid counts (c).
    const int count = tid < kNumGroups ? cursor[tid] : 0;
    const int extent = cute::round_up(count, BLOCK_M);
    int p = extent, c = count;
    const int warp = __ppu_read_warpid();
    const int lane = __ppu_read_laneid();
    if (warp < (kNumGroups + WARP_SIZE - 1) / WARP_SIZE) {
        #pragma unroll
        for (int d = 1; d < WARP_SIZE; d *= 2) {
            int a = __shfl_up_sync(0xFFFFFFFF, p, d);
            int b = __shfl_up_sync(0xFFFFFFFF, c, d);
            if (lane >= d) {
                p += a;
                c += b;
            }
        }
        if (lane == WARP_SIZE - 1) totals[warp] = make_int2(p, c);
    }
    __syncthreads();
    if (tid < kNumGroups) {
        // Add preceding warp totals to obtain block-wide inclusive sums.
        for (int w = 0; w < warp; ++w) {
            p += totals[w].x;
            c += totals[w].y;
        }
        // Subtract this expert's count to obtain its exclusive start offset.
        // padded indexes GEMM input slots; compact indexes valid intermediate rows.
        padded[tid] = p - extent;
        compact[tid] = c - count;
        m_rows[tid] = count;
        if (tid == kNumGroups - 1) *aligned_num_m_blocks = p / BLOCK_M;
        // Per block: expert ID, valid row count, first block, compact row start.
        const int4 info =
            make_int4(tid, count, (p - extent) / BLOCK_M, c - count);
        int4* metadata = reinterpret_cast<int4*>(expert_ids_and_cumsum);
        for (int pos = p - extent; pos < p; pos += BLOCK_M) {
            metadata[pos / BLOCK_M] = info;
        }
    }
    __syncthreads();
    // =======================================================================
    // Phase 4: Scatter token IDs and write inverse/compact expert mappings
    // =======================================================================
    // Valid routes overwrite PAD/invalid entries; unused m_indices stay -1.
    #pragma unroll
    for (int c = 0; c < rounds; ++c) {
        const int i = tid + c * BLOCK_SIZE;
        if (i >= numel) break;
        const int e = topk_ids[i];
        if (e < 0 || e >= kNumGroups) {
            inv_perm[i] = -1;
            continue;
        }
        sorted_token_ids[padded[e] + offsets[c]] = i / kTopK;
        inv_perm[i] = compact[e] + offsets[c];
        m_indices[compact[e] + offsets[c]] = e;
    }
}


// =============================================================================
// 4-kernel moe_align pipeline: K1 count+fill -> K2 reduce -> K3 cumsum+ids -> K4 scatter.
// Two traversals of topk_ids, no CPU round-trip; expert-local order is unspecified.
// =============================================================================

// -- K1: block_count_with_fill ----------------------------------------------
// Fills sorted_token_ids with PAD_ID (shape_m) via strided writes, then counts
// tokens per expert per block into block_counts[NUM_BLOCKS][kNumGroups].
// Each block processes BLOCK_SIZE elements of topk_ids.
//
// Grid: NUM_BLOCKS blocks.  SMEM: kNumGroups * sizeof(int32_t).
// ----------------------------------------------------------------------------
template <int BLOCK_SIZE, int kNumGroups, int kTopK>
__device__ __forceinline__ void
block_count_with_fill_impl(
    const int32_t* __restrict__ topk_ids,        // [numel] flattened topk_ids
    int32_t* __restrict__ sorted_token_ids,      // [s_total_ub] output, stride-filled with PAD_ID
    int32_t* __restrict__ block_counts,          // [NUM_BLOCKS * kNumGroups] output
    int32_t* __restrict__ m_indices,             // [numel] initialized for GEMM2
    int numel,                                   // total elements = M * topk
    int s_total_ub,                              // upper bound of s_total (= max_num_m_blocks * BLOCK_M)
    int bid)                                     // logical block index
{
    int tid = threadIdx.x;
    size_t idx = static_cast<size_t>(bid) * BLOCK_SIZE + tid;

    // -- Fill PAD: stride-write shape_m to all of sorted_token_ids --
    int shape_m = numel / kTopK;
    {
        int4 fill_val = make_int4(shape_m, shape_m, shape_m, shape_m);
        int4* out_vec = (int4*)sorted_token_ids;
        int total_vecs = s_total_ub / 4;
        if (idx < total_vecs) {
            out_vec[idx] = fill_val;
        } else if (idx < s_total_ub){
            sorted_token_ids[idx] = shape_m;
        }
    }

    extern __shared__ int32_t smem_counts[];  // [kNumGroups]
    if (tid < kNumGroups) {
        smem_counts[tid] = 0;
    }
    __syncthreads();

    if (idx < numel) {
        m_indices[idx] = -1;
        int eid = topk_ids[idx];
        if (eid >= 0 && eid < kNumGroups) {
            atomicAdd(&smem_counts[eid], 1);
        }
    }
    __syncthreads();
    int32_t* expert_count_per_block = block_counts + bid * kNumGroups;
    // -- Write back
    if (tid < kNumGroups) {
        expert_count_per_block[tid] = smem_counts[tid];
    }
}


// -- K2: Reduce counts and initialize the expert cursor ----------------------
// One block per expert. CUB BlockReduce uses static shared storage.
// Output: expert_cursors[0:kNumGroups] cursors and expert_counts[kNumGroups].
// ----------------------------------------------------------------------------
template <int BLOCK_SIZE>
__device__ __forceinline__ void
reduce_expert_counts_impl(
    const int32_t* block_counts,   // [num_blocks * kNumGroups]
    int32_t* expert_cursors,         // first kNumGroups entries: cursor
    int32_t* expert_counts,          // [kNumGroups] output
    int kNumGroups,                              // runtime
    int num_blocks,
    int eid)                                     // expert index handled by this block
{
    int sum = 0;
    for (int b = threadIdx.x; b < num_blocks; b += BLOCK_SIZE)
        sum += block_counts[b * kNumGroups + eid];
    using BlockReduce = cub::BlockReduce<int32_t, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    sum = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0) {
        expert_counts[eid] = sum;
        expert_cursors[eid] = 0;
    }
}


// -- K3: cumsum_expert_ids --------------------------------------------------
// Computes padded and compact expert starts from valid route counts (m_rows)
// via two CUB BlockScans, and writes expert_ids_and_cumsum in DeepGEMM format.
// All threads participate in scans; expert threads write their own M-blocks.
//
// Grid: 1 block. Dynamic SMEM: 2 * (kNumGroups + 1) * sizeof(int32_t).
// CUB BlockScan additionally uses static shared storage.
// Output: cumsum_out[2 * kNumGroups] (padded starts, then compact starts),
// expert_ids_and_cumsum and aligned_num_m_blocks.
// ----------------------------------------------------------------------------
template <int BLOCK_SIZE, int BLOCK_M, int kNumGroups>
__device__ __forceinline__ void
cumsum_expert_ids_impl(
    const int32_t* __restrict__ m_rows,              // [kNumGroups] expert_counts from K2
    int32_t* __restrict__ expert_ids_and_cumsum,     // [max_num_m_blocks * 4] output (DeepGEMM format, int4)
    int32_t* __restrict__ cumsum_out,                // [2 * kNumGroups] padded starts, then compact starts for K4
    int32_t* __restrict__ aligned_num_m_blocks)      // [1] output
{
    int tid = threadIdx.x;
    bool is_valid = (tid < kNumGroups);

    extern __shared__ int32_t smem[];
    int32_t* shared_cumsum = smem;                    // [kNumGroups + 1]
    int32_t* shared_nopad  = smem + (kNumGroups + 1); // [kNumGroups + 1]

    // Read m_rows[tid] directly (global is known, no SMEM copy needed)
    int group_num = 0;
    int padded = 0;
    if (is_valid) {
        group_num = m_rows[tid];
        padded = cute::round_up(group_num, BLOCK_M);
    }

    // CUB BlockScan: parallel prefix sum over padded counts
    using BlockScan = cub::BlockScan<int32_t, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int cumsum_val = 0;
    BlockScan(temp_storage).ExclusiveSum(padded, cumsum_val);

    // Write per-expert element-level cumsum to SMEM
    // Thread kNumGroups: after exclusive scan, its cumsum_val is the grand total
    if (tid <= kNumGroups) {
        shared_cumsum[tid] = cumsum_val;
    }
    __syncthreads();

    // Write aligned_num_m_blocks
    if (tid == kNumGroups) {
        *aligned_num_m_blocks = cumsum_val / BLOCK_M;
    }

    // Second scan: row-level compact cumsum (raw counts)
    int nopad_val = 0;
    BlockScan(temp_storage).ExclusiveSum(group_num, nopad_val);
    if (tid <= kNumGroups) {
        shared_nopad[tid] = nopad_val;
    }
    __syncthreads();

    // Each expert thread writes both starts and its GEMM block metadata (int4)
    if (is_valid) {
        cumsum_out[tid] = shared_cumsum[tid];
        cumsum_out[tid + kNumGroups] = shared_nopad[tid];
        int cumsum_aligned = shared_cumsum[tid] / BLOCK_M;
        int cumsum_compact = shared_nopad[tid];
        for (int i = shared_cumsum[tid]; i < shared_cumsum[tid + 1]; i += BLOCK_M) {
            int block_idx = i / BLOCK_M;
            int4* quad = (int4*)(expert_ids_and_cumsum + block_idx * 4);
            *quad = make_int4(tid, group_num, cumsum_aligned, cumsum_compact);
        }
    }
}


// -- K4: Atomic scatter -----------------------------------------------------
// Each valid route reserves one slot from its expert cursor.
// Grid: num_blocks blocks. BLOCK_SIZE = next_power_of_two(kNumGroups).
// Expert-local order is unspecified; inverse mappings preserve route identity.
// ----------------------------------------------------------------------------
template <int BLOCK_SIZE, int BLOCK_M, int kTopK>
__device__ __forceinline__ void
atomic_scatter_impl(
    const int32_t* __restrict__ topk_ids,            // [numel] flattened topk_ids
    int32_t* __restrict__ sorted_token_ids,           // [s_total_ub] output
    int32_t* __restrict__ inv_perm,                   // [numel] output
    int32_t* __restrict__ m_indices,                  // [numel] per-row expert ID for nopad GEMM2
    const int32_t* __restrict__ cumsum,                // [2 * kNumGroups] padded starts, then compact starts from K3
    int32_t* __restrict__ expert_cursors,         // first kNumGroups entries: cursor
    int numel,                                         // = M * topk
    int kNumGroups,                                    // runtime
    int bid)                                           // logical block index
{
    const int idx = bid * BLOCK_SIZE + threadIdx.x;
    const int eid = idx < numel ? topk_ids[idx] : -1;
    if (idx >= numel) return;
    if (static_cast<unsigned>(eid) >= static_cast<unsigned>(kNumGroups)) {
        inv_perm[idx] = -1;
        return;
    }
    const int offset = atomicAdd(&expert_cursors[eid], 1);
    sorted_token_ids[cumsum[eid] + offset] = idx / kTopK;
    inv_perm[idx] = cumsum[kNumGroups + eid] + offset;
    m_indices[cumsum[kNumGroups + eid] + offset] = eid;
}

} // namespace deep_gemm
