#pragma once
#include <cub/cub.cuh>

#include "cute/ppu_tensor_mix.hpp"
#include "cutlass/gemm/config/gemm_operands.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cute/atom/copy_traits_ppu0015_aiu.hpp"
#include "cute/algorithm/ppu_copy.hpp"

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

struct GemmArgs {
  const void *__restrict__ a_ptr;
  const void *__restrict__ b_ptr;
  void *__restrict__ c_ptr;

  const int *__restrict__ expert_ids_and_cumsum;
  const int *__restrict__ sorted_token_ids;
  const int *__restrict__ aligned_num_m_blocks;
  uint32_t shape_m;
};

struct QuantGemmArgs : public GemmArgs{
  const void *__restrict__ scale_a_ptr;
  const void *__restrict__ scale_b_ptr;
};

// tsm.ld.swzl need 128B aligned
template <typename SrcT, int kNumStages, int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct GemmSmemConfig {
  static constexpr uint32_t kSmemASize = cute::round_up(kNumStages * BLOCK_M * BLOCK_K * sizeof(SrcT), 128);
  static constexpr uint32_t kSmemBSize = cute::round_up(kNumStages * BLOCK_N * BLOCK_K * sizeof(SrcT), 128);
  static constexpr uint32_t kTotalSize = kSmemASize + kSmemBSize;
};

template <typename SrcT, int kNumStages, int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct BlkwiseQuantGemmSmemConfig : public GemmSmemConfig<SrcT, kNumStages, BLOCK_M, BLOCK_N, BLOCK_K> {
  using Base = GemmSmemConfig<SrcT, kNumStages, BLOCK_M, BLOCK_N, BLOCK_K>;
  static constexpr uint32_t kSmemScaleASize = cute::round_up(
      kNumStages * BLOCK_M * BLOCK_K / 128 * sizeof(float), 128);
  static constexpr uint32_t kSmemScaleBSize = cute::round_up(
      kNumStages * cute::ceil_div(BLOCK_N, 128) * BLOCK_K / 128 * sizeof(float), 256);
  static constexpr uint32_t kTotalSize = Base::kTotalSize + kSmemScaleASize + kSmemScaleBSize;
};

template <int kNumStages, int BLOCK_M, int BLOCK_N, int BLOCK_K, typename MainloopFp4>
struct GemmSmemConfigFp4 {
  // A is copied with cp.sync
  // the size of storage of cutlass::float4_t is uint8;
  static constexpr uint32_t kSmemASize = cute::round_up(kNumStages * BLOCK_M * BLOCK_K * sizeof(cutlass::float4_t), 128);
  // B is copied with aiu
  static constexpr uint32_t kSmemBSize = kNumStages * BLOCK_N * BLOCK_K * sizeof(cutlass::float4_t);

  static constexpr auto kSmemSFASize = cute::cosize_v<typename MainloopFp4::SmemLayoutSFA> * sizeof(typename MainloopFp4::ElementSFA);
  static constexpr auto kSmemSFBSize = cute::cosize_v<typename MainloopFp4::SmemLayoutSFB> * sizeof(typename MainloopFp4::ElementSFB);

  static constexpr uint32_t kTotalSize = kSmemASize + kSmemBSize + kSmemSFASize + kSmemSFBSize;
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

template <typename AccT, typename DstT,
         uint32_t SHAPE_N, uint32_t BLOCK_N, uint32_t STRIDE_CM,
         class TAcc, class TCcC, class... Ts>
__forceinline__ __device__ void epilogue_no_tsm(TAcc& accum, const TCcC& tCcC, void* c_ptr,
                                                uint32_t c_row_base, uint32_t num_valid_rows, uint32_t blk_n_offset) {
  using namespace cute;
#if __HGGC_ARCH__ == 150
  CUTLASS_PRAGMA_UNROLL
  for (int warp_m = 0; warp_m < size<1>(tCcC); ++warp_m) {
    CUTLASS_PRAGMA_UNROLL
    for (int mma_m = 0; mma_m < size<0,1>(tCcC); ++mma_m) {
      uint32_t local_row = get<0>(tCcC(make_coord(_0{}, mma_m, _0{}), warp_m, _0{}));
      bool cond = local_row < num_valid_rows;
      uint32_t row = c_row_base + local_row;
      if (cond) {
        DstT* cur_c_ptr = reinterpret_cast<DstT*>(c_ptr) + row * STRIDE_CM + blk_n_offset;
        CUTLASS_PRAGMA_UNROLL
        for (int warp_n = 0; warp_n < size<2>(tCcC); ++warp_n) {
          CUTLASS_PRAGMA_UNROLL
          for (int mma_n = 0; mma_n < size<0,2>(tCcC); ++mma_n) {
            AccT* acc_ptr = cute::raw_pointer_cast(accum(make_coord(_, mma_m, mma_n), warp_m, warp_n).data());
            uint32_t d;
            asm volatile("ppu.cvt.rtte.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(acc_ptr[1]), "f"(acc_ptr[0]));

            size_t n_idx = get<1>(tCcC(make_coord(_0{}, mma_m, mma_n), warp_m, warp_n));
            uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(cur_c_ptr + n_idx);
            if constexpr (SHAPE_N % BLOCK_N) {
              if (cond && n_idx < (SHAPE_N - blk_n_offset)) {
                *dst_ptr = d;
              }
            } else {
              *dst_ptr = d;
            }
          }
        }
      }
    }
  }
#else
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(tCcC); ++i) {
    size_t local_row = cute::get<0>(tCcC(i));
    bool cond = local_row < num_valid_rows;
    size_t row = c_row_base + local_row;
    if constexpr (SHAPE_N % BLOCK_N) {
      cond = cond && cute::get<1>(tCcC(i)) < (SHAPE_N - blk_n_offset);
    }
    if (cond) {
      AccT* acc_ptr = cute::raw_pointer_cast(accum.data()) + accum.layout()(i);
      DstT* dst_ptr = reinterpret_cast<DstT*>(c_ptr) + row * STRIDE_CM + blk_n_offset + cute::get<1>(tCcC(i));
      *dst_ptr = DstT(*acc_ptr);
    }
  }
#endif
};

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

template <typename T>
__device__ __host__  inline T round_up(T value, T alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

template <int N>
constexpr int next_pow2() {
    int v = N - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

// =============================================================================
// Single-kernel WarpOrdered deterministic moe_align (replaces 4-kernel for numel <= 16384)
// 1 block x 1024 threads, 4 phases, 3 syncs, zero grid.sync()
// =============================================================================

template <int BLOCK_M, int kNumGroups, int kTopK, int BLOCK_SIZE = 1024>
__global__ __launch_bounds__(BLOCK_SIZE, 1) void
moe_align_warp_ordered_kernel(
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
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    const int warp_id = __ppu_read_warpid();
    const int lane_id = __ppu_read_laneid();

    extern __shared__ int32_t smem[];
    int32_t* warp_counts = smem;
    int32_t* warp_prefix = smem;
    int32_t* shared_cumsum = smem + NUM_WARPS * kNumGroups;
    int32_t* shared_nopad  = shared_cumsum + (kNumGroups + 1);

    // peer_rank[c] = local rank within warp chunk c (used by Phase 4 scatter)
    int peer_rank[16];

    int numel_per_warp = (numel + NUM_WARPS - 1) / NUM_WARPS;
    int warp_start = warp_id * numel_per_warp;
    int warp_end = min(warp_start + numel_per_warp, numel);

    // =======================================================================
    // Phase 1: Fill PAD + Warp Count + peer_rank[]
    // =======================================================================
    {
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

        for (int i = threadIdx.x; i < NUM_WARPS * kNumGroups; i += BLOCK_SIZE) {
            warp_counts[i] = 0;
        }
        __syncthreads();

        for (int c = 0; c * WARP_SIZE < (warp_end - warp_start); c++) {
            int idx = warp_start + c * WARP_SIZE + lane_id;
            bool valid = (idx < warp_end);
            int eid = valid ? topk_ids[idx] : kNumGroups;
            uint32_t mask = __match_any_sync(0xFFFFFFFF, eid);
            uint32_t prefix_mask = mask & __ppu_read_lanemask_lt();

            if (valid) {
                int running_cnt = warp_counts[warp_id * kNumGroups + eid];
                if (prefix_mask == 0) {
                    warp_counts[warp_id * kNumGroups + eid] = running_cnt + __popc(mask);
                }
                peer_rank[c] = running_cnt + __popc(prefix_mask);
            }
        }
    }
    __syncthreads();

    // =======================================================================
    // Phase 2: Warp Prefix (Hillis-Steele, warp_counts -> warp_prefix)
    // =======================================================================
    {
        for (int eid = warp_id; eid < kNumGroups; eid += NUM_WARPS) {
            int val = (lane_id < NUM_WARPS) ? warp_counts[lane_id * kNumGroups + eid] : 0;
            int sum = val;
            #pragma unroll
            for (int k = 1; k < 32; k *= 2) {
                int n = __shfl_up_sync(0xFFFFFFFF, sum, k);
                if (lane_id >= k) sum += n;
            }
            if (lane_id < NUM_WARPS) {
                warp_prefix[lane_id * kNumGroups + eid] = sum - val;
            }
            if (lane_id == NUM_WARPS - 1) {
                shared_cumsum[eid] = sum;
            }
        }
    }
    __syncthreads();

    // =======================================================================
    // Phase 3: CUB BlockScan cumsum + expert_ids_and_cumsum
    // =======================================================================
    {
        int group_num = 0;
        int padded = 0;
        if (threadIdx.x < kNumGroups) {
            group_num = shared_cumsum[threadIdx.x];
            m_rows[threadIdx.x] = group_num;
            padded = round_up(group_num, BLOCK_M);
        }
        using BlockScan = cub::BlockScan<int32_t, BLOCK_SIZE>;
        __shared__ typename BlockScan::TempStorage temp_storage;
        int cumsum_val = 0;
        BlockScan(temp_storage).ExclusiveSum(padded, cumsum_val);
        if (threadIdx.x <= kNumGroups) {
            shared_cumsum[threadIdx.x] = cumsum_val;
        }
        __syncthreads();
        if (threadIdx.x == kNumGroups) {
            *aligned_num_m_blocks = cumsum_val / BLOCK_M;
        }
        // Second scan: row-level compact cumsum (raw, not padded)
        int nopad_val = 0;
        BlockScan(temp_storage).ExclusiveSum(group_num, nopad_val);
        if (threadIdx.x <= kNumGroups) {
            shared_nopad[threadIdx.x] = nopad_val;
        }
        __syncthreads();
        if (threadIdx.x < kNumGroups) {
            int eid = threadIdx.x;
            int cumsum_aligned = shared_cumsum[eid] / BLOCK_M;
            int cumsum_compact = shared_nopad[eid];
            for (int i = shared_cumsum[eid]; i < shared_cumsum[eid + 1]; i += BLOCK_M) {
                int block_idx = i / BLOCK_M;
                int4* quad = (int4*)(expert_ids_and_cumsum + block_idx * 4);
                *quad = make_int4(eid, group_num, cumsum_aligned, cumsum_compact);
            }
        }
    }

    // =======================================================================
    // Phase 4: Scatter
    // =======================================================================
    {
        for (int idx = warp_start + lane_id; idx < warp_end; idx += WARP_SIZE) {
            int eid = topk_ids[idx];
            int c = (idx - warp_start) / WARP_SIZE;
            int offset = warp_prefix[warp_id * kNumGroups + eid] + peer_rank[c];
            int rank = shared_cumsum[eid] + offset;
            int compact_rank = shared_nopad[eid] + offset;
            sorted_token_ids[rank] = idx / kTopK;
            inv_perm[idx] = compact_rank;
            m_indices[compact_rank] = eid;
        }
    }
}

// =============================================================================
// 4-kernel deterministic moe_align pipeline (ported from sglang final version)
// Replaces the original single-kernel atomicAdd scatter with a fully
// deterministic pipeline: K1 count+fill -> K2 local_scan -> K3 cumsum+ids -> K4 scatter.
// 2 traversals of topk_ids, zero CPU round-trip, bit-identical output across runs.
// =============================================================================

// -- K1: block_count_with_fill ----------------------------------------------
// Fills sorted_token_ids with PAD_ID (shape_m) via strided writes, then counts
// tokens per expert per block into block_counts[NUM_BLOCKS][kNumGroups].
// Each block processes BLOCK_SIZE elements of topk_ids.
//
// Grid: NUM_BLOCKS blocks.  SMEM: kNumGroups * sizeof(int32_t).
// ----------------------------------------------------------------------------
template <int BLOCK_SIZE, int kNumGroups, int kTopK>
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
block_count_with_fill(
    const int32_t* __restrict__ topk_ids,        // [numel] flattened topk_ids
    int32_t* __restrict__ sorted_token_ids,      // [s_total_ub] output, stride-filled with PAD_ID
    int32_t* __restrict__ block_counts,          // [NUM_BLOCKS * kNumGroups] output
    int numel,                                   // total elements = M * topk
    int s_total_ub)                              // upper bound of s_total (= max_num_m_blocks * BLOCK_M)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int num_blocks = gridDim.x;
    size_t idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

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
        int eid = topk_ids[idx];
        atomicAdd(&smem_counts[eid], 1);
    }
    __syncthreads();
    int32_t* expert_count_per_block = block_counts + bid * kNumGroups;
    // -- Write back
    if (tid < kNumGroups) {
        expert_count_per_block[tid] = smem_counts[tid];
    }
}

// -- K2: local_scan ---------------------------------------------------------
// Parallel exclusive scan across NUM_BLOCKS for each expert column.
// Uses CUB BlockScan for O(log N) inter-thread prefix.
//
// Grid: kNumGroups blocks.  SMEM: NUM_BLOCKS * sizeof(int32_t).
// Output: local_offsets[num_blocks][kNumGroups], expert_counts[kNumGroups].
// ----------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
local_scan(
    const int32_t* __restrict__ block_counts,   // [num_blocks * kNumGroups]
    int32_t* __restrict__ local_offsets,         // [num_blocks * kNumGroups] output
    int32_t* __restrict__ expert_counts,          // [kNumGroups] output
    int kNumGroups,                              // runtime
    int num_blocks)
{
    int eid = blockIdx.x;
    int tid = threadIdx.x;
    if (eid >= kNumGroups) return;

    extern __shared__ int32_t smem_vals[];

    // Coalesced strided load into SMEM (same as original)
    for (int b = tid; b < num_blocks; b += BLOCK_SIZE) {
        smem_vals[b] = block_counts[b * kNumGroups + eid];
    }
    __syncthreads();

    // Contiguous assignment: thread tid handles blocks [start, end)
    int ept = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int start = tid * ept;
    int end = min(start + ept, num_blocks);

    // Partial sum over contiguous blocks
    int my_partial = 0;
    for (int b = start; b < end; b++) {
        my_partial += smem_vals[b];
    }

    // Single BlockScan on partial sums (1 call, same as original)
    using BlockScan = cub::BlockScan<int32_t, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int inter_prefix = 0;
    BlockScan(temp_storage).ExclusiveSum(my_partial, inter_prefix);

    if (tid == BLOCK_SIZE - 1) {
        expert_counts[eid] = inter_prefix + my_partial;
    }

    // Sequential scan + writeback over contiguous blocks
    int local_total = 0;
    for (int b = start; b < end; b++) {
        local_offsets[b * kNumGroups + eid] = local_total + inter_prefix;
        local_total += smem_vals[b];
    }
}

// -- K3: cumsum_expert_ids --------------------------------------------------
// Block-aligns expert token counts (from m_rows), computes element-level cumsum
// via CUB BlockScan, and writes expert_ids_and_cumsum in DeepGEMM format.
// All threads participate: each handles its own expert's M-blocks.
//
// Grid: 1 block.  SMEM: (kNumGroups + 1) * sizeof(int32_t).
// Output: cumsum_out[kNumGroups] + expert_ids_and_cumsum + aligned_num_m_blocks.
// ----------------------------------------------------------------------------
template <int BLOCK_SIZE, int BLOCK_M, int kNumGroups>
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
cumsum_expert_ids(
    const int32_t* __restrict__ m_rows,              // [kNumGroups] expert_counts from K2
    int32_t* __restrict__ expert_ids_and_cumsum,     // [max_num_m_blocks * 4] output (DeepGEMM format, uint4)
    int32_t* __restrict__ cumsum_out,                // [kNumGroups] element-level padded cumsum for K4
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
        padded = round_up(group_num, BLOCK_M);
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

    // Each thread writes cumsum_out + expert_ids_and_cumsum (uint4) for its own expert
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

// -- K4: deterministic_scatter ----------------------------------------------
// Thread<->expert 1:1 binding.  SMEM tile cooperative load + scan.
// ZERO atomicAdd in scatter phase -- fully deterministic.
//
// Grid: num_blocks blocks.  BLOCK_SIZE = next_pow2(kNumGroups).
// Output: sorted_token_ids (filled with token_idx = idx / topk).
// ----------------------------------------------------------------------------
template <int BLOCK_SIZE, int BLOCK_M, int kTopK>
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
deterministic_scatter(
    const int32_t* __restrict__ topk_ids,            // [numel] flattened topk_ids
    int32_t* __restrict__ sorted_token_ids,           // [s_total_ub] output
    int32_t* __restrict__ inv_perm,                   // [numel] output
    int32_t* __restrict__ m_indices,                  // [numel] per-row expert ID for nopad GEMM2
    const int32_t* __restrict__ cumsum,                // [kNumGroups] element-level padded cumsum from K3
    const int32_t* __restrict__ local_offsets,         // [num_blocks * kNumGroups]
    int numel,                                         // = M * topk
    int kNumGroups)                                    // runtime
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int idx = bid * BLOCK_SIZE + tid;

    // Cooperative load one tile -> SMEM (all threads must reach __syncthreads)
    __align__(16) __shared__ int32_t tile[BLOCK_SIZE];
    tile[tid] = (idx < numel) ? topk_ids[idx] : -1;
    __syncthreads();

    // Vectorized scan: read 4 elements per iteration via int4
    if (tid < kNumGroups) {
        int eid = tid;
        int local_off = local_offsets[bid * kNumGroups + eid];
        int off = cumsum[eid] + local_off;
        int compact_off = cumsum[eid + kNumGroups] + local_off;
        int cnt = 0;
        int block_base = bid * BLOCK_SIZE;
        int4* tile4 = (int4*)tile;
        for (int j = 0; j < BLOCK_SIZE / 4; j++) {
            int4 v = tile4[j];
            int ib = block_base + j * 4;
            // Compute match mask: 4 independent comparisons → 4-bit mask
            int mask = ((v.x == eid) << 0) | ((v.y == eid) << 1) |
                       ((v.z == eid) << 2) | ((v.w == eid) << 3);
            int n = __popc(mask);
            if (n > 0) {
                int idxs[4] = {ib, ib + 1, ib + 2, ib + 3};
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    if (mask & (1 << k)) {
                        sorted_token_ids[off + cnt] = idxs[k] / kTopK;
                        inv_perm[idxs[k]] = compact_off + cnt;
                        m_indices[compact_off + cnt] = eid;
                        cnt++;
                    }
                }
            }
        }
    }
}

// -- Launcher ---------------------------------------------------------------
// Receives pre-allocated intermediate buffers from Python (torch tensors).
// Launches K1->K2->K3->K4 sequentially on the given stream.
// ---------------------------------------------------------------------------
template<int BLOCK_M, int kNumGroups, int kTopK>
void moe_align_block_size_kernel_launcher(
      int* m_rows,
      int* expert_ids_and_cumsum,
      int* sorted_token_ids, int* aligned_num_m_blocks,
      int* inv_perm, int* m_indices,
      const int* topk_ids, int numel, int max_num_m_blocks,
      int* intermediate_buffer,
      hggcStream_t stream) {

    int s_total_ub = max_num_m_blocks * BLOCK_M;

    const char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && std::atoi(pEnv_params) == 1) {
        printf("[moe_align] BLOCK_M=%d, kNumGroups=%d, numel=%d\n", BLOCK_M, kNumGroups, numel);
    }

    // -- WarpOrdered path: numel <= 16384 --
    constexpr int MOE_ALIGN_WARP_ORDERED_MAX_NUMEL = 16384;
    if (numel <= MOE_ALIGN_WARP_ORDERED_MAX_NUMEL) {
        constexpr int BLOCK_SIZE = 1024;
        constexpr int smem = (kNumGroups * (BLOCK_SIZE / 32) + 2 * (kNumGroups + 1)) * sizeof(int32_t);
        auto device_func = moe_align_warp_ordered_kernel<BLOCK_M, kNumGroups, kTopK, BLOCK_SIZE>;

        device_func<<<1, BLOCK_SIZE, smem, stream>>>(
                topk_ids, sorted_token_ids, m_rows,
                expert_ids_and_cumsum, aligned_num_m_blocks,
                inv_perm, m_indices,
                numel, s_total_ub);
        return;
    }

    // -- 4-kernel fallback --
    // -- Runtime parameters --
    constexpr int BLOCK_SIZE = next_pow2<kNumGroups>();      // K1, K4
    constexpr int K3_BLOCK  = next_pow2<kNumGroups + 1>();  // K3 only
    constexpr int K2_BLOCK  = 256;                          // K2 scan

    int num_blocks_pad = cute::ceil_div(s_total_ub, BLOCK_SIZE);
    int num_blocks     = cute::ceil_div(numel, BLOCK_SIZE);

    // Split intermediate buffer: [block_counts | local_offsets | cumsum]
    int* block_counts   = intermediate_buffer;
    int* local_offsets  = intermediate_buffer + num_blocks_pad * kNumGroups;
    int* cumsum_gpu     = intermediate_buffer + num_blocks_pad * kNumGroups * 2;

    // -- K1: fill PAD + per-block count --
    {
        constexpr int smem = kNumGroups * sizeof(int32_t);
        block_count_with_fill<BLOCK_SIZE, kNumGroups, kTopK>
            <<<num_blocks_pad, BLOCK_SIZE, smem, stream>>>(
                topk_ids, sorted_token_ids, block_counts,
                numel, s_total_ub);
    }

    // -- K2: 2-level parallel scan -> expert_counts + local_offsets --
    {
        int smem = num_blocks * sizeof(int32_t);
        local_scan<K2_BLOCK>
            <<<kNumGroups, K2_BLOCK, smem, stream>>>(
                block_counts, local_offsets, m_rows,
                kNumGroups, num_blocks);
    }

    // -- K3: cumsum + expert_ids (DeepGEMM format) + element-level cumsum --
    {
        constexpr int smem = 2 * (kNumGroups + 1) * sizeof(int32_t);
        cumsum_expert_ids<K3_BLOCK, BLOCK_M, kNumGroups>
            <<<1, K3_BLOCK, smem, stream>>>(
                m_rows, expert_ids_and_cumsum, cumsum_gpu,
                aligned_num_m_blocks);
    }

    // -- K4: deterministic scatter --
    {
        constexpr int smem = BLOCK_SIZE * sizeof(int32_t);
        deterministic_scatter<BLOCK_SIZE, BLOCK_M, kTopK>
            <<<num_blocks, BLOCK_SIZE, smem, stream>>>(
                topk_ids, sorted_token_ids, inv_perm, m_indices, cumsum_gpu, local_offsets,
                numel, kNumGroups);
    }
}

} // namespace deep_gemm
