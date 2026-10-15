#pragma once

#include <megamoe/layout/sym_buffer.cuh>
#include <megamoe/layout/workspace.cuh>
#include <megamoe/ptx/atomic.cuh>

namespace megamoe::comm {

template <uint32_t kNumSMs, uint32_t kGridSyncIndex = 0, typename sync_scope_t>
DG_DEVICE void grid_sync(const layout::Workspace& workspace,
                              const uint32_t& logical_sm_idx,
                              const uint32_t& thread_idx,
                              const sync_scope_t& sync_scope) {
    sync_scope();
    if (thread_idx == 0) {
        const auto counter_ptr = workspace.get_grid_sync_counter_ptr<kGridSyncIndex>();
        const auto old_value = ptx::atomic_add_rel(counter_ptr, logical_sm_idx == 0 ? (ptx::kFinishSumTag - (kNumSMs - 1)) : 1);
        uint32_t new_value;
        do {
            new_value = ptx::ld_acq_gpu(counter_ptr);
        } while (((new_value ^ old_value) & ptx::kFinishSumTag) == 0);
    }
    sync_scope();
}

template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kGridSyncIndex, uint32_t kTag, typename sync_scope_t>
DG_DEVICE void icn_barrier(const layout::Workspace& workspace,
                                const layout::SymBuffer<kNumRanks>& sym_buffer,
                                const uint32_t& logical_sm_idx,
                                const uint32_t& thread_idx,
                                const sync_scope_t& sync_scope,
                                const bool& sync_prologue = true,
                                const bool& sync_epilogue = true) {
    DG_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");

    if (sync_prologue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, logical_sm_idx, thread_idx, sync_scope);

    if (logical_sm_idx == 0) {
        auto* counter_ptr = workspace.get_icn_barrier_counter_ptr();
        const auto status = (*counter_ptr) & 3;
        const auto signal_phase = status & 1, signal_sign = status >> 1;
        auto* signal_ptr = workspace.get_icn_barrier_signal_ptr(signal_phase);

        // The arrival signal is a reduction on a peer-mapped address, which is why it is the sys-scope
        // form: the SDK has no remote-atomic form, so a plain red on the peer VA is what the dispatch
        // uses, and every combine of the fused kernel runs this barrier.
        if (thread_idx < kNumRanks)
            ptx::red_add_rel_sys(sym_buffer.map_pointer_to_rank(signal_ptr, thread_idx), signal_sign ? -1 : 1);
        sync_scope();

#ifndef DG_ICN_BARRIER_TIMEOUT_SEC
#define DG_ICN_BARRIER_TIMEOUT_SEC 300
#endif
        // The timeout is in seconds; the SM clock is ~2 GHz, hence the cycle factor.
        constexpr int64_t kNumTimeoutCycles = static_cast<int64_t>(DG_ICN_BARRIER_TIMEOUT_SEC) * 2000000000ll;
        if (thread_idx == 0) {
            ptx::red_add(counter_ptr, 1);
            const uint32_t target = signal_sign ? 0u : kNumRanks;
            const auto* const signal_u32_ptr = reinterpret_cast<const uint32_t*>(signal_ptr);
            uint32_t last_clock = static_cast<uint32_t>(clock64());
            uint64_t elapsed_cycles = 0;
#ifndef DG_ICN_BARRIER_DISABLE_TIMEOUT
#define DG_ICN_BARRIER_DISABLE_TIMEOUT 0
#endif
#if DG_ICN_BARRIER_DISABLE_TIMEOUT
            (void)last_clock;
            (void)elapsed_cycles;
            while (ptx::ld_acq_sys(signal_u32_ptr) != target) {
            }
#else
            while (ptx::ld_acq_sys(signal_u32_ptr) != target) {
                const uint32_t now_clock = static_cast<uint32_t>(clock64());
                elapsed_cycles += static_cast<uint32_t>(now_clock - last_clock);
                last_clock = now_clock;
                if (elapsed_cycles >= static_cast<uint64_t>(kNumTimeoutCycles)) {
                    printf("megamoe ICN barrier timeout (%us): rank=%u, counter=%u, signal=%u, target=%u, phase=%u, sign=%u, tag=%u\n",
                           static_cast<unsigned>(DG_ICN_BARRIER_TIMEOUT_SEC),
                           sym_buffer.rank_idx,
                           *counter_ptr,
                           ptx::ld_acq_sys(signal_u32_ptr),
                           target,
                           signal_phase,
                           signal_sign,
                           kTag);
                    DG_DEVICE_ASSERT(false and "ICN barrier timeout");
                }
            }
#endif
        }
    }

    if (sync_epilogue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, logical_sm_idx, thread_idx, sync_scope);
}

}  // namespace megamoe::comm
