#pragma once
#include "utils.cuh"
namespace deep_gemm {

template <GemmType kGemmType,
    uint32_t SHAPE_N_, uint32_t SHAPE_K_,
    uint32_t BLOCK_M_, uint32_t BLOCK_N_,
    uint32_t kNumGroups_,
    uint32_t kNumNBlocks = ceil_div(SHAPE_N_, BLOCK_N_),
    uint32_t kNum1DBlocksPerGroup = 2>
struct FusedGemmScheduler {
public:
  struct Params {
    const int32_t* aligned_num_m_blocks;
    const int32_t* expert_ids_and_cumsum;
  };

  __device__ __forceinline__ explicit FusedGemmScheduler(Params params)
  : FusedGemmScheduler(params.aligned_num_m_blocks, params.expert_ids_and_cumsum) {};

  __device__ __forceinline__ explicit FusedGemmScheduler(const int32_t* aligned_num_m_blocks,
                                                         const int32_t* expert_ids_and_cumsum)
    : expert_ids_and_cumsum_((uint4*)expert_ids_and_cumsum), current_iter(0),
      curr_group_m(0), curr_group_idx(0), curr_cumsum_m(0), cumsum_m_block_idx(0),
      curr_block_m_offset(0), valid_m_in_block(0) {
    num_aligned_m_blocks_ = __ldg(aligned_num_m_blocks);
    num_blocks_ = num_aligned_m_blocks_ * kNumNBlocks;
  }

  __device__ __forceinline__ void get_swizzled_block_idx(const uint32_t num_m_blocks, int block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
    // Swizzle for better L2 usages
    const uint32_t& primary_num_blocks = num_m_blocks;
    constexpr uint32_t num_blocks_per_group = kNumNBlocks * kNum1DBlocksPerGroup;
    uint32_t group_idx = block_idx / num_blocks_per_group;
    uint32_t first_block_idx = group_idx * kNum1DBlocksPerGroup;
    uint32_t in_group_idx = block_idx % num_blocks_per_group;
    uint32_t num_blocks_in_group = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);

    m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
    n_block_idx = in_group_idx / num_blocks_in_group;
  }

  __device__ __forceinline__ bool fetch_next_work(uint32_t& m_block_idx, uint32_t& n_block_idx) {
    const auto next_block_idx = (current_iter++) * gridDim.x + blockIdx.x;
    if (next_block_idx >= num_blocks_) {
        m_block_idx = num_aligned_m_blocks_;
        n_block_idx = kNumNBlocks;
        return false;
    }
    uint32_t block_m_idx = next_block_idx / kNumNBlocks;
    uint4 data = __ldg(expert_ids_and_cumsum_ + block_m_idx);
    curr_group_idx = data.x;
    curr_group_m = data.y;
    uint32_t num_m_blocks = ceil_div(curr_group_m, BLOCK_M);
    uint32_t block_idx_in_m = next_block_idx - data.z * kNumNBlocks;
    get_swizzled_block_idx(num_m_blocks, block_idx_in_m, m_block_idx, n_block_idx);
    cumsum_m_block_idx = m_block_idx + data.z;
    curr_block_m_offset = data.w + m_block_idx * BLOCK_M;
    valid_m_in_block = min(BLOCK_M, curr_group_m - m_block_idx * BLOCK_M);
    return true;
  }

public:
  constexpr static uint32_t SHAPE_N = SHAPE_N_;
  constexpr static uint32_t SHAPE_K = SHAPE_K_;
  constexpr static uint32_t BLOCK_M = BLOCK_M_;
  constexpr static uint32_t BLOCK_N = BLOCK_N_;

  // Block configs
  uint32_t num_aligned_m_blocks_;
  uint32_t num_blocks_;

  const uint4* expert_ids_and_cumsum_;

  // iter
  int current_iter;
  uint32_t curr_group_idx;
  uint32_t curr_group_m;
  uint32_t curr_cumsum_m;
  uint32_t cumsum_m_block_idx;
  uint32_t curr_block_m_offset;
  uint32_t valid_m_in_block;
};

} // namespace deep_gemm
