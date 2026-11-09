#pragma once

#include <cstdint>
#include <hggc_runtime.h>
#include <deep_gemm/common/math.cuh>

namespace deep_gemm::mega_moe_common {

#if defined(__HGGC__)

template <uint32_t kNumNBlocks, uint32_t kNum1DBlocksPerGroup>
__device__ __forceinline__ void get_swizzled_block_idx(
    const uint32_t num_m_blocks, int block_idx,
    uint32_t& m_block_idx, uint32_t& n_block_idx) {
    // Preserve small M groups and the EP8 bit-reversal mapping.
    const uint32_t group_span = kNum1DBlocksPerGroup >= 16 ? 2u : kNum1DBlocksPerGroup;
    const auto num_blocks_per_group = kNumNBlocks * group_span;
    const auto group_idx = block_idx / num_blocks_per_group;
    const auto first_block_idx = group_idx * group_span;
    const auto in_group_idx = block_idx % num_blocks_per_group;
    const uint32_t num_blocks_in_group = min(group_span, num_m_blocks - first_block_idx);
    m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
    n_block_idx = in_group_idx / num_blocks_in_group;
    if constexpr (kNum1DBlocksPerGroup >= 16) {
        if (num_m_blocks >= kNum1DBlocksPerGroup) {
            const uint32_t bits = 32u - __clz(num_m_blocks - 1u);
            // Skip out-of-range reversed inputs for a 3/4-full domain.
            const uint32_t input =
                bits >= 2 && num_m_blocks == (3u << (bits - 2))
                ? m_block_idx + m_block_idx / 3u : m_block_idx;
            const uint32_t permuted = __brev(input) >> (32u - bits);
            if (permuted < num_m_blocks)
                m_block_idx = permuted;
        }
    }
}

// Work assignment stays in each scheduler; tile decoding shares the same state fields.
template <class Scheduler>
__device__ __forceinline__ bool fetch_next_work(
    Scheduler& state, uint32_t next_block_idx,
    uint32_t& m_block_idx, uint32_t& n_block_idx) {
    if constexpr (Scheduler::kIsFusedDispatch) {
        if (next_block_idx >= state.num_blocks) {
            m_block_idx = state.num_aligned_m_blocks;
            n_block_idx = Scheduler::kNumNBlocks;
            return false;
        }
        // {expert, token count, first M-block, first row}.
        const uint32_t global_m_block = next_block_idx / Scheduler::kNumNBlocks;
        const uint4 data = (reinterpret_cast<const uint4*>(state.params.copy_grouped_layout) + 1)[global_m_block];
        state.curr_group_idx = data.x;
        state.curr_group_m = data.y;
        state.curr_cumsum_m = data.w;
        state.curr_global_block_m_idx = global_m_block;
        m_block_idx = global_m_block - data.z;
        n_block_idx = next_block_idx - global_m_block * Scheduler::kNumNBlocks;
    } else {
        uint32_t num_m_blocks;
        while (true) {
            if (state.curr_group_idx == Scheduler::kNumGroups) {
                m_block_idx = state.num_aligned_m_blocks;
                n_block_idx = Scheduler::kNumNBlocks;
                return false;
            }
            state.curr_group_m = static_cast<uint32_t>(__ldg(state.params.grouped_layout + state.curr_group_idx));
            num_m_blocks = math::ceil_div(state.curr_group_m, Scheduler::BLOCK_M);
            const auto current_m_block_cumsum = state.curr_cumsum + num_m_blocks;
            if (next_block_idx < current_m_block_cumsum * Scheduler::kNumNBlocks)
                break;
            state.curr_group_idx ++, state.curr_cumsum = current_m_block_cumsum;
            state.curr_cumsum_m += state.curr_group_m;
        }
        get_swizzled_block_idx<Scheduler::kNumNBlocks, Scheduler::kNum1DBlocksPerGroup>(
            num_m_blocks, next_block_idx - state.curr_cumsum * Scheduler::kNumNBlocks,
            m_block_idx, n_block_idx);
    }
    return true;
}

#endif

}  // namespace deep_gemm::mega_moe_common
