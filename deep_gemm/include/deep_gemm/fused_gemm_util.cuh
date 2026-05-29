#pragma once
#include <cub/cub.cuh>

#include "cute/ppu_tensor_mix.hpp"
#include "cutlass/gemm/config/gemm_operands.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cute/atom/copy_traits_ppu0015_aiu.hpp"
#include "cute/algorithm/ppu_copy.hpp"

namespace deep_gemm {
using cute::_;

struct GemmArgs {
  const void *__restrict__ a_ptr;
  const void *__restrict__ b_ptr;
  void *__restrict__ c_ptr;

  const int *__restrict__ m_rows;
  const int *__restrict__ expert_ids_and_cumsum;
  const int *__restrict__ sorted_token_ids;
  const int *__restrict__ aligned_num_m_blocks;
  uint32_t shape_m;
  int topk;
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
      kNumStages * BLOCK_N / 128 * BLOCK_K / 128 * sizeof(float), 256);
  static constexpr uint32_t kTotalSize = Base::kTotalSize + kSmemScaleASize + kSmemScaleBSize;
};

template <typename SrcT, typename ACopyInst, typename TilerA,
         uint32_t BLOCK_M, uint32_t BLOCK_K, uint32_t STRIDE_AM,
         class TAsA, class... Ts>
__forceinline__ __device__ void copy_A_to_tsm(TAsA&& tAsA, const void* src_ptr, const int* blk_token_base,
                                              uint32_t stage_offset, uint32_t num_valid_tokens, int topk, uint32_t thread_idx) {
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
        bool token_mask = token_offset < num_valid_tokens;
        const cutlass::uint128_t* src_ptr128 = reinterpret_cast<const cutlass::uint128_t*>(reinterpret_cast<const SrcT*>(src_ptr)
              + token_offset / topk * STRIDE_AM + (k_iter * TILER_N) + tid_k * KPerThread + stage_offset);
        cutlass::uint128_t* dst_ptr128 = reinterpret_cast<cutlass::uint128_t*>(cute::raw_pointer_cast(tAsA(_,m_iter,k_iter).data()));
        ACopyInst::copy(*src_ptr128, *dst_ptr128, token_mask);
      }
    }
  }
};

template <typename AccT, typename DstT,
         uint32_t SHAPE_N, uint32_t BLOCK_N, uint32_t STRIDE_CM,
         class TAcc, class TCcC, class... Ts>
__forceinline__ __device__ void epilogue_no_tsm(TAcc& accum, TCcC& tCcC, void* c_ptr, const int* blk_token_base,
                                                uint32_t num_valid_tokens, uint32_t blk_n_offset) {
#if __HGGC_ARCH__ == 150
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(tCcC); i += 2) {
    size_t token_offset = __ldg(blk_token_base + cute::get<0>(tCcC(i)));
    bool cond = token_offset < num_valid_tokens;
    if constexpr (SHAPE_N % BLOCK_N) {
      cond = cond && cute::get<1>(tCcC(i)) < (SHAPE_N - blk_n_offset);
    }
    if (cond) {
      AccT* acc_ptr = cute::raw_pointer_cast(accum.data()) + accum.layout()(i);
      uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(reinterpret_cast<DstT*>(c_ptr)
                          + token_offset * STRIDE_CM + blk_n_offset + cute::get<1>(tCcC(i)));
      uint32_t d;
      asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(acc_ptr[1]), "f"(acc_ptr[0]));
      *dst_ptr = d;
    }
  }
#else
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(tCcC); ++i) {
    size_t token_offset = __ldg(blk_token_base + cute::get<0>(tCcC(i)));
    bool cond = token_offset < num_valid_tokens;
    if constexpr (SHAPE_N % BLOCK_N) {
      cond = cond && cute::get<1>(tCcC(i)) < (SHAPE_N - blk_n_offset);
    }
    if (cond) {
      AccT* acc_ptr = cute::raw_pointer_cast(accum.data()) + accum.layout()(i);
      DstT* dst_ptr = reinterpret_cast<DstT*>(c_ptr) + token_offset * STRIDE_CM + blk_n_offset + cute::get<1>(tCcC(i));
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
            void* c_ptr, void* smem_buffer, const int* blk_token_base, uint32_t thread_idx,
            uint32_t num_valid_tokens, uint32_t blk_n_offset) {
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
          size_t token_offset = __ldg(blk_token_base + get<0>(tSR_cDmn(0,m,n)));
          size_t n_offset = blk_n_offset + get<1>(tSR_cDmn(0,m,n));
          // Predication
          bool cond = token_offset < num_valid_tokens;
          if constexpr (SHAPE_N % BLOCK_N) {
            cond = cond &&  n_offset < SHAPE_N;
          }
          if (cond) {
            uint128_t* dst_ptr = (uint128_t*)((DstT*)c_ptr + token_offset * STRIDE_CM + n_offset);
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

template <int BLOCK_SIZE, int BLOCK_M, int kNumGroups>
__global__ void __launch_bounds__(BLOCK_SIZE, 1) moe_align_block_size_kernel(
    const int* __restrict__ topk_ids, int* __restrict__ m_rows,
    int* __restrict__ expert_ids_and_cumsum, int* __restrict__ sorted_token_ids,
    int* __restrict__ aligned_num_m_blocks, int numel, int max_num_tokens_padded) {

  uint32_t tid = threadIdx.x;
  // Initialize sorted_token_ids with numel
  for (uint32_t it = tid; it < max_num_tokens_padded; it += blockDim.x) {
    sorted_token_ids[it] = numel;
  }

  extern __shared__ int32_t shared_buffer[];
  int* shared_counts = shared_buffer;
  int* shared_cumsum = shared_buffer + kNumGroups;

  bool is_valid_experts = tid < kNumGroups;
  // init shared memory
  if (is_valid_experts) {
    shared_buffer[tid] = 0;
  }
  __syncthreads();

  // Count tokens per expert
  for (uint32_t i = tid; i < numel; i += blockDim.x) {
    int eid = topk_ids[i];
    atomicAdd(&shared_counts[eid], 1);
  }
  __syncthreads();

  // Compute prefix sum over token counts per expert
  using BlockScan = cub::BlockScan<int32_t, BLOCK_SIZE>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int expert_count = 0;
  int expert_padded_count = 0;

  if (is_valid_experts) {
    expert_count = shared_counts[tid];
    expert_padded_count = round_up(expert_count, BLOCK_M);
  }

  int cumsum_val = 0;
  BlockScan(temp_storage).ExclusiveSum(expert_padded_count, cumsum_val);
  if (tid < (kNumGroups + 1)) {
    shared_cumsum[tid] = cumsum_val;
  }
  __syncthreads();

  if (tid == kNumGroups) {
    *aligned_num_m_blocks = cumsum_val / BLOCK_M;
  }

  if (is_valid_experts) {
    m_rows[tid] = expert_count;  // m_rows
    for (int i = shared_cumsum[tid]; i < shared_cumsum[tid + 1]; i += BLOCK_M) {
      expert_ids_and_cumsum[i / BLOCK_M * 2] = tid;
      expert_ids_and_cumsum[i / BLOCK_M * 2 + 1] = shared_cumsum[tid] / BLOCK_M;
    }
  }
  // generate sorted token ids
  for (size_t i = tid; i < numel; i += blockDim.x) {
    int eid = topk_ids[i];
    int pos = atomicAdd(&shared_cumsum[eid], 1);
    sorted_token_ids[pos] = i;
  }
}

template<int BLOCK_M, int kNumGroups>
void moe_align_block_size_kernel_launcher(int* m_rows, int* expert_ids_and_cumsum,
      int* sorted_token_ids, int* aligned_num_m_blocks, const int* topk_ids,
      int numel, int max_num_m_blocks, cudaStream_t stream) {
  constexpr uint32_t BLOCK_SIZE = 1024;
  constexpr uint32_t SMEM_SIZE = (kNumGroups * 2 + 1) * sizeof(int);
  moe_align_block_size_kernel<BLOCK_SIZE, BLOCK_M, kNumGroups> <<<1, BLOCK_SIZE, SMEM_SIZE, stream>>>(topk_ids,
          m_rows, expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks,
          numel, max_num_m_blocks * BLOCK_M);
  // printf(" num_token, topk:%d, %d, max_num_m_blocks:%d, BLOCK_M:%d\n",  num_token, topk, max_num_m_blocks, BLOCK_M);
}

} // namespace deep_gemm
