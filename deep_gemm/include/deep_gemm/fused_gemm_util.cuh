#pragma once
#include <cub/cub.cuh>

namespace deep_gemm {

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
