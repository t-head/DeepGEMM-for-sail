#pragma once

#include <cstdint>
#include <type_traits>
#include <hggc_runtime.h>
#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm {

template <class Scheduler, class = void>
struct MegaMoEDispatchTraits {
    static constexpr bool kIsLinear1 = false;
    using Type = void;
};

template <class Scheduler>
struct MegaMoEDispatchTraits<Scheduler, std::void_t<typename Scheduler::Dispatch>> {
    static constexpr bool kIsLinear1 = Scheduler::kIsFusedDispatch;
    using Type = typename Scheduler::Dispatch;
};

}  // namespace deep_gemm

namespace deep_gemm::sched {

template <class Scale>
struct MegaMoESchedulerArgumentsT {
    int* grouped_layout = nullptr;
    // Copy and compute CTAs share the same M-block metadata.
    int* copy_grouped_layout = nullptr;
    uint32_t shape_m = 0;
    const uint32_t* expert_rank_prefix = nullptr;
    uint8_t* local_data_buf = nullptr;
    Scale* local_sfa_buf = nullptr;
    volatile uint32_t* copy_ready_flags = nullptr;
    uint32_t num_copy_blocks = 0;
    layout::SymBuffer<> sym_buffer{};
    layout::MegaMoEStaging staging{};
    const uint8_t* dispatch_data = nullptr;
    const uint32_t* dispatch_token_topk = nullptr;
    uint32_t* expert_base = nullptr;
    layout::TokenSrcMetadata* src_info = nullptr;
    int32_t* cumulative_local_expert_recv_stats = nullptr;

    CUTLASS_HOST_DEVICE
    MegaMoESchedulerArgumentsT(uint32_t shape_m = 0, int* grouped_layout_ptr = nullptr)
        : grouped_layout(grouped_layout_ptr), shape_m(shape_m) {}
};

using MegaMoESchedulerArguments = MegaMoESchedulerArgumentsT<uint16_t>;
using Int8MegaMoESchedulerArguments = MegaMoESchedulerArgumentsT<float>;

#if defined(__HGGC__)
// Tile ordering is independent of operand type and the expert-row layout.
template <uint32_t NumNBlocks, uint32_t NumRanks, uint32_t GroupSpan = 2>
struct MegaMoETileOrder {
    CUTLASS_DEVICE static void get_block_idx(
        uint32_t num_m_blocks, uint32_t block_idx,
        uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const uint32_t blocks_per_group = NumNBlocks * GroupSpan;
        const uint32_t first = block_idx / blocks_per_group * GroupSpan;
        const uint32_t in_group = block_idx % blocks_per_group;
        const uint32_t rows = min(GroupSpan, num_m_blocks - first);
        m_block_idx = first + in_group % rows;
        n_block_idx = in_group / rows;
        if constexpr (NumRanks == 8) {
            if (num_m_blocks >= 16) {
                const uint32_t bits = 32u - __clz(num_m_blocks - 1u);
                const uint32_t input = bits >= 2 && num_m_blocks == (3u << (bits - 2))
                    ? m_block_idx + m_block_idx / 3u : m_block_idx;
                const uint32_t permuted = __brev(input) >> (32u - bits);
                if (permuted < num_m_blocks)
                    m_block_idx = permuted;
            }
        }
    }
};

// Advance monotonically through unpadded expert rows for GEMM2 scatter.
template <uint32_t GroupSpan, class Scheduler>
CUTLASS_DEVICE bool fetch_scatter_work(
    Scheduler& state, uint32_t next_block_idx,
    uint32_t& m_block_idx, uint32_t& n_block_idx) {
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
    MegaMoETileOrder<Scheduler::kNumNBlocks, Scheduler::kNumRanks, GroupSpan>::get_block_idx(
        num_m_blocks, next_block_idx - state.curr_cumsum * Scheduler::kNumNBlocks,
        m_block_idx, n_block_idx);
    return true;
}
#endif

}  // namespace deep_gemm::sched
