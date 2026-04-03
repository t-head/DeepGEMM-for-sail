#pragma once
#include <cub/cub.cuh>

#include "ppu/cute/tensor_mix.hpp"
#include "ppu/gemm/config/gemm_operands.hpp"
#include "ppu/cute/atom/copy_traits_acompute10000_aiu.hpp"
#include "ppu/cute/atom/copy_traits_acompute10500_aiu.hpp"
#include "ppu/cute/algorithm/copy.hpp"

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
// #if __HGGC_ARCH__ == 150
#if 1
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

  // 统计每个专家的 token 数量, m_rows
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
