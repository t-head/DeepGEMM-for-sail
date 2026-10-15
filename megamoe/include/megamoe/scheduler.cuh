#pragma once

#include <megamoe/common/math.cuh>
#include <megamoe/common/tie.cuh>
#include <megamoe/layout/workspace.cuh>
#include <megamoe/ptx/atomic.cuh>
#include <megamoe/ptx/warp.cuh>

namespace megamoe::sched {

enum class BlockPhase { kNoBlock = 0, kL1Gemm = 1, kL2Gemm = 2 };

template <uint32_t BLOCK_M,
          uint32_t BLOCK_N,
          uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N,
          uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N,
          uint32_t L2_SHAPE_K,
          uint32_t kExpertsPerRank,
          uint32_t kExpertsPerGroup,
          uint32_t kSmCount,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kExpertsPerRank, 32u),
          uint32_t kNumL1NBlocks = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2NBlocks = L2_SHAPE_N / BLOCK_N,
          uint32_t kL1KTiles = L1_SHAPE_K / BLOCK_K,
          uint32_t kL2KTiles = L2_SHAPE_K / BLOCK_K>
struct BlockScheduler {
    DG_STATIC_ASSERT(L1_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(kExpertsPerGroup > 0 and kExpertsPerGroup <= kExpertsPerRank, "Invalid wave config");

    DG_STATIC_ASSERT(kNumL1NBlocks % 2 == 0, "L1 N block count must be even");
    DG_STATIC_ASSERT(kNumL2NBlocks % 2 == 0, "L2 N block count must be even");

    const layout::Workspace& workspace;

    BlockPhase next_phase = BlockPhase::kL1Gemm;

    uint32_t current_local_expert_idx = 0;
    uint32_t current_num_tokens = 0;
    uint32_t work_block_idx = 0;
    uint32_t m_block_idx = 0;
    uint32_t n_block_idx = 0;

    uint32_t l2_range_begin = 0;
    uint32_t l2_range_end = 0;
    uint32_t l2_block_idx = 0;

    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};

    uint32_t active_expert_mask[kNumExpertsPerLane] = {};

    DG_DEVICE void build_active_mask() {
#ifdef DG_SCHED_SKIP_EMPTY
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++i)
            active_expert_mask[i] = __ballot_sync(0xffffffff, stored_num_tokens_per_expert[i] != 0);
#endif
    }

    DG_DEVICE uint32_t find_next_active_expert(const uint32_t start_expert_idx) const {
        if (start_expert_idx >= kExpertsPerRank)
            return kExpertsPerRank;
        const uint32_t word_idx = start_expert_idx / 32;
        const uint32_t mask = active_expert_mask[word_idx] & (~0u << (start_expert_idx % 32));
        if (mask)
            return word_idx * 32 + (__ffs(mask) - 1);
        #pragma unroll
        for (uint32_t i = word_idx + 1; i < kNumExpertsPerLane; ++i)
            if (active_expert_mask[i])
                return i * 32 + (__ffs(active_expert_mask[i]) - 1);
        return kExpertsPerRank;
    }

    DG_DEVICE explicit BlockScheduler(const layout::Workspace& workspace) : workspace(workspace) { work_block_idx = blockIdx.x; }

    DG_DEVICE uint32_t get_expert_group_end_idx() const {
        const auto aligned = math::align_up(current_local_expert_idx + 1, kExpertsPerGroup);
        return megamoe::min(aligned, kExpertsPerRank);
    }

    DG_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
        uint32_t lane_token_count = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
            lane_token_count = (expert_idx == i * 32 + ptx::get_lane_idx()) ? stored_num_tokens_per_expert[i] : lane_token_count;
        }
        return ptx::broadcast_from_lane(lane_token_count, expert_idx % 32);
    }

    DG_DEVICE void advance_expert_idx() {
        const uint32_t next =
#ifdef DG_SCHED_SKIP_EMPTY
            find_next_active_expert(current_local_expert_idx + 1);
#else
            current_local_expert_idx + 1;
#endif
        current_local_expert_idx = next;
        current_num_tokens = get_num_tokens(next);
    }

    DG_DEVICE void set_local_expert_idx(const uint32_t& expert_idx) {
        current_local_expert_idx = expert_idx;
        current_num_tokens = get_num_tokens(expert_idx);
    }

    DG_DEVICE void compute_l2_range(const uint32_t& group_end_expert_idx) {
#ifdef DG_SCHED_LANE_MBLOCK_SUM
        uint32_t m_block_count_local = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
            const uint32_t expert_idx = i * 32 + ptx::get_lane_idx();
            if (expert_idx >= current_local_expert_idx && expert_idx < group_end_expert_idx)
                m_block_count_local += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        const uint32_t m_block_count_sum = __reduce_add_sync(0xffffffff, m_block_count_local);
#else
        uint32_t m_block_count_sum = 0;
#ifdef DG_SCHED_SERIAL_MBLOCK_SUM_NO_UNROLL
        // Unrolling this walk into the fused kernel's register file is what tips the 128-row tile
        // over the per-thread VReg budget (see launch._get_compile_defines).
        #pragma unroll 1
#endif
        for (uint32_t e = current_local_expert_idx; e < group_end_expert_idx; ++e)
            m_block_count_sum += math::ceil_div(get_num_tokens(e), BLOCK_M);
#endif
        const uint32_t num_l1_blocks = m_block_count_sum * kNumL1NBlocks;
        const uint32_t num_l2_blocks = m_block_count_sum * kNumL2NBlocks;
#ifdef DG_SCHED_L2_COST_MODEL
        constexpr uint64_t kCostScale = 16;
        constexpr uint64_t kL1ExtraCost = 72;
        constexpr uint64_t kL2ExtraCost = 110;
        const uint64_t l1_block_cost = static_cast<uint64_t>(kL1KTiles) * kCostScale + kL1ExtraCost;
        const uint64_t l2_block_cost = static_cast<uint64_t>(kL2KTiles) * kCostScale + kL2ExtraCost;
        const uint64_t total_cost =
            static_cast<uint64_t>(num_l1_blocks) * l1_block_cost + static_cast<uint64_t>(num_l2_blocks) * l2_block_cost;
#else
        const uint64_t l1_block_cost = kL1KTiles;
        const uint64_t l2_block_cost = kL2KTiles;
        const uint64_t total_cost =
            static_cast<uint64_t>(num_l1_blocks) * l1_block_cost + static_cast<uint64_t>(num_l2_blocks) * l2_block_cost;
#endif
        const uint32_t num_l1_blocks_min = num_l1_blocks / kSmCount;
        const uint32_t num_extra_sms = num_l1_blocks % kSmCount;
        const uint64_t budget_step = total_cost / kSmCount;
        const uint64_t budget_remainder = total_cost % kSmCount;
        const uint32_t num_low_sms = kSmCount - num_extra_sms;
        const uint32_t logical_sm_idx = blockIdx.x;
        const uint32_t balance_logical_sm_idx =
            (logical_sm_idx >= num_extra_sms) ? (logical_sm_idx - num_extra_sms) : (num_low_sms + logical_sm_idx);
        auto get_num_l2_blocks_below = [&](const uint32_t num_sms) -> uint32_t {
            const uint64_t budget =
                static_cast<uint64_t>(num_sms) * budget_step + static_cast<uint64_t>(num_sms) * budget_remainder / kSmCount;
            const uint64_t num_l1_blocks_below = (num_sms <= num_low_sms) ? static_cast<uint64_t>(num_sms) * num_l1_blocks_min
                                                                          : static_cast<uint64_t>(num_low_sms) * num_l1_blocks_min +
                    static_cast<uint64_t>(num_sms - num_low_sms) * (num_l1_blocks_min + 1);
            const uint64_t spent = l1_block_cost * num_l1_blocks_below;
            const uint64_t num_l2_blocks_in_budget = (budget > spent) ? (budget - spent) / l2_block_cost : 0ull;
            return (num_l2_blocks_in_budget > num_l2_blocks) ? num_l2_blocks : static_cast<uint32_t>(num_l2_blocks_in_budget);
        };
        l2_range_begin = get_num_l2_blocks_below(balance_logical_sm_idx);
        l2_range_end = get_num_l2_blocks_below(balance_logical_sm_idx + 1);
        if (l2_range_end < l2_range_begin)
            l2_range_end = l2_range_begin;
        l2_block_idx = l2_range_begin;
    }

    DG_DEVICE uint32_t get_current_m_block_count() const { return math::ceil_div(current_num_tokens, BLOCK_M); }

    DG_DEVICE uint32_t get_valid_row_count() const { return megamoe::min(current_num_tokens - m_block_idx * BLOCK_M, BLOCK_M); }

    DG_DEVICE bool fetch_next_l1_block() {
        const auto group_end_expert_idx = get_expert_group_end_idx();
        while (current_local_expert_idx < group_end_expert_idx) {
            const auto m_block_count = get_current_m_block_count();
            m_block_idx = work_block_idx / kNumL1NBlocks;
            if (m_block_idx < m_block_count)
                return true;

            work_block_idx -= m_block_count * kNumL1NBlocks;
            advance_expert_idx();
        }
        return false;
    }

    DG_DEVICE bool fetch_next_l2_block() {
        const auto group_end_expert_idx = get_expert_group_end_idx();
        while (current_local_expert_idx < group_end_expert_idx) {
            const auto m_block_count = get_current_m_block_count();
            if (work_block_idx < m_block_count * kNumL2NBlocks) {
                m_block_idx = work_block_idx / kNumL2NBlocks;
                return true;
            }

            work_block_idx -= m_block_count * kNumL2NBlocks;
            advance_expert_idx();
        }
        return false;
    }

    DG_DEVICE megamoe::tuple<BlockPhase, uint32_t, uint32_t, uint32_t> get_next_block() {
        while (true) {
            if (current_local_expert_idx >= kExpertsPerRank)
                break;

            if (next_phase == BlockPhase::kL1Gemm) {
                if (fetch_next_l1_block()) {
                    n_block_idx = work_block_idx - m_block_idx * kNumL1NBlocks;
                    work_block_idx += kSmCount;
                    return {BlockPhase::kL1Gemm, current_local_expert_idx, m_block_idx, n_block_idx};
                } else {
                    next_phase = BlockPhase::kL2Gemm;
                    set_local_expert_idx(math::align_down<uint32_t>(current_local_expert_idx - 1, kExpertsPerGroup));
                    compute_l2_range(get_expert_group_end_idx());
                    work_block_idx = l2_range_begin;
                }
            } else {
                if (l2_block_idx < l2_range_end && fetch_next_l2_block()) {
                    n_block_idx = work_block_idx - m_block_idx * kNumL2NBlocks;
                    work_block_idx += 1;
                    ++l2_block_idx;
                    return {BlockPhase::kL2Gemm, current_local_expert_idx, m_block_idx, n_block_idx};
                } else {
                    next_phase = BlockPhase::kL1Gemm;
                    work_block_idx = blockIdx.x;
                    set_local_expert_idx(get_expert_group_end_idx());
                }
            }
        }

        return {BlockPhase::kNoBlock, 0, 0, 0};
    }

    DG_DEVICE void fetch_local_expert_recv_counts(const uint32_t rank_idx) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kExpertsPerRank)
                value = ptx::ld_volatile(workspace.get_expert_recv_count_ptr(rank_idx, expert_idx));
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
        __syncwarp();
        build_active_mask();
    }
};

}  // namespace megamoe::sched
