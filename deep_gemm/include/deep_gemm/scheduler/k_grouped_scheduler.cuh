#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>

namespace deep_gemm {

template <GemmType kGemmType,
          uint32_t SHAPE_N_,
          uint32_t BLOCK_M_, uint32_t BLOCK_N_,
          uint32_t kNumGroups_,
          uint32_t kNumNBlocks = math::ceil_div(SHAPE_N_, BLOCK_N_),
          uint32_t kNum1DBlocksPerGroup = 2>
struct KGroupedScheduler {
    static_assert(is_k_grouped_contiguous(kGemmType), "KGroupedScheduler requires a K-grouped GEMM type");

    static constexpr uint32_t SHAPE_N = SHAPE_N_;
    static constexpr uint32_t BLOCK_M = BLOCK_M_;
    static constexpr uint32_t BLOCK_N = BLOCK_N_;
    static constexpr uint32_t kNumGroups = kNumGroups_;
    static constexpr uint32_t NUM_N_BLOCKS = kNumNBlocks;

    const int32_t* grouped_layout;
    uint32_t current_iter = 0;
    uint32_t num_blocks;
    uint32_t num_m_blocks;
    uint32_t curr_group_idx = 0;
    uint32_t current_shape_k = 0;
    uint32_t current_k_start = 0;
    bool group_loaded = false;

    __device__ __forceinline__ explicit KGroupedScheduler(const int32_t* grouped_layout_, uint32_t shape_m)
        : grouped_layout(grouped_layout_) {
        num_m_blocks = math::ceil_div(shape_m, BLOCK_M);
        num_blocks = kNumGroups * num_m_blocks * NUM_N_BLOCKS;
    }

    __device__ __forceinline__ void get_swizzled_block_idx(const uint32_t num_m_blocks, int block_idx,
                                                            uint32_t& m_block_idx, uint32_t& n_block_idx) {
        constexpr uint32_t num_blocks_per_group = kNumNBlocks * kNum1DBlocksPerGroup;
        const uint32_t group_idx = block_idx / num_blocks_per_group;
        const uint32_t first_block_idx = group_idx * kNum1DBlocksPerGroup;
        const uint32_t in_group_idx = block_idx % num_blocks_per_group;
        const uint32_t num_blocks_in_group = min(kNum1DBlocksPerGroup, num_m_blocks - first_block_idx);
        m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        n_block_idx = in_group_idx / num_blocks_in_group;
    }

    __device__ __forceinline__ void load_group_bounds(const uint32_t group_idx) {
        if (group_loaded && curr_group_idx == group_idx) {
            return;
        }
        // Persistent work indices only advance. Like the NoPad row-prefix
        // cursor, retain the K prefix and read each crossed group once.
        if (!group_loaded) {
            current_shape_k = grouped_layout[0];
            group_loaded = true;
        }
        while (curr_group_idx < group_idx) {
            current_k_start += current_shape_k;
            current_shape_k = grouped_layout[++curr_group_idx];
        }
    }

    __device__ __forceinline__ bool fetch_next_work(uint32_t& group_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const uint32_t blocks_per_group = num_m_blocks * NUM_N_BLOCKS;
        while (true) {
            const auto next_block_idx = (current_iter++) * gridDim.x + blockIdx.x;
            if (next_block_idx >= num_blocks) {
                return false;
            }

            group_idx = next_block_idx / blocks_per_group;
            load_group_bounds(group_idx);
            // beta=1: empty groups keep the existing output and have no work.
            if (current_shape_k == 0) {
                continue;
            }

            const uint32_t block_idx_in_group = next_block_idx - group_idx * blocks_per_group;
            get_swizzled_block_idx(num_m_blocks, block_idx_in_group, m_block_idx, n_block_idx);
            return true;
        }
    }
};

}  // namespace deep_gemm
