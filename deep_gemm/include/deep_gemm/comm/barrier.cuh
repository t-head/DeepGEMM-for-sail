#pragma once

// Cross-rank and grid synchronization for MegaMoE using PPU TIX atomics.
// The `sync_scope` functor performs a block-wide barrier (`__syncthreads()`).
#include <deep_gemm/tix/sys_atomics.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm::comm {

// Grid-wide barrier across all SMs of one rank, driven by a global counter.
template <uint32_t kNumSMs, uint32_t kGridSyncIndex = 0,
          uint32_t kPollBackoffCycles = 0, typename sync_scope_t>
CUTLASS_DEVICE void grid_sync(const layout::Workspace& workspace,
                              const uint32_t& sm_idx, const uint32_t& thread_idx,
                              const sync_scope_t& sync_scope) {
    static constexpr uint32_t kFinishSumTag = 0x80000000u;
    sync_scope();
    if (thread_idx == 0) {
        const auto count_ptr = workspace.get_grid_sync_count_ptr<kGridSyncIndex>();
        const auto old_value = tix::atomic_add_rel(
            count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1)) : 1);
        uint32_t new_value;
        const uint32_t poll_backoff_cycles =
            kPollBackoffCycles + ((sm_idx * 37u) & 255u);
        do {
            new_value = tix::ld_acq(count_ptr);
            if constexpr (kPollBackoffCycles > 0)
                __nanosleep(poll_backoff_cycles);
        } while (((new_value ^ old_value) & kFinishSumTag) == 0);
    }
    sync_scope();
}

// Cross-rank barrier. Only SM 0 participates: every rank signals all
// remote ranks (release), then spins until its own signal reaches the target
// (acquire). Wrapped by grid syncs on both sides.
template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kGridSyncIndex, uint32_t kTag, typename sync_scope_t>
CUTLASS_DEVICE void cross_rank_barrier(const layout::Workspace& workspace,
                                   const layout::SymBuffer<kNumRanks>& sym_buffer,
                                   const uint32_t& sm_idx, const uint32_t& thread_idx,
                                   const sync_scope_t& sync_scope,
                                   const bool& sync_prologue = true,
                                   const bool& sync_epilogue = true) {
    DG_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");

    // Grid sync before cross-rank signaling
    if (sync_prologue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);

    // Cross-rank barrier, only SM 0 participates
    if (sm_idx == 0) {
        auto* counter_ptr = workspace.get_nvl_barrier_counter_ptr();
        const auto status = (*counter_ptr) & 3;
        const auto signal_phase = status & 1, signal_sign = status >> 1;
        auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);

        // Send signals to remote ranks
        if (thread_idx < kNumRanks)
            tix::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);
        sync_scope();

        // Update status and wait arrival (with 180s timeout, at 2 GHz)
        constexpr int64_t kNumTimeoutCycles = 180ll * 2000000000ll;
        if (thread_idx == 0) {
            tix::red_add(counter_ptr, 1u);
            const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
            const auto start_clock = clock64();
            while (tix::ld_acq_sys(signal_ptr) != target) {
                if (clock64() - start_clock >= kNumTimeoutCycles) {
                    printf("DeepGEMM cross-rank barrier timeout (180s): rank=%d, counter=%d, signal=%d, target=%d, phase=%d, sign=%d, tag=%d\n",
                           sym_buffer.rank_idx, *counter_ptr, tix::ld_acq_sys(signal_ptr), target, signal_phase, signal_sign, kTag);
                    DG_DEVICE_ASSERT(false and "cross-rank barrier timeout");
                }
            }
        }
    }

    // Grid sync after cross-rank completion
    if (sync_epilogue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);
}

} // namespace deep_gemm::comm
