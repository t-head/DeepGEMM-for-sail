#pragma once

#ifndef DG_MEGAMOE_OWNER_MODE
#define DG_MEGAMOE_OWNER_MODE 0
#endif

#include <cstdint>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/tix/ld_st.cuh>

namespace deep_gemm::comm {

__device__ __forceinline__ void grid_sync(
        layout::ScatterSignals& workspace,
        uint32_t num_blocks, uint32_t block_idx, uint32_t thread_idx) {
    constexpr uint32_t kFinishTag = 0x80000000u;
    __syncthreads();
    if (thread_idx == 0) {
        auto* counter = reinterpret_cast<uint32_t*>(&workspace.grid_counter);
        const uint32_t old_value = tix::atomic_add_rel(
            counter, block_idx == 0 ? kFinishTag - (num_blocks - 1) : 1);
        uint32_t new_value;
        do {
            new_value = tix::ld_acq(counter);
        } while (((new_value ^ old_value) & kFinishTag) == 0);
    }
    __syncthreads();
}

// Acquire every bank before exposing the combined writer mask to the reader.
// The upper unused bits stay set, so all-bulk remains a single UINT64_MAX test.
template <uint32_t kWorldSize>
__device__ __forceinline__ uint64_t fp4_scatter_acquire(
        const layout::ScatterSignals& workspace, uint32_t status) {
    static_assert(kWorldSize >= 1 && kWorldSize <= layout::kNumMaxRanks &&
                  (kWorldSize & (kWorldSize - 1)) == 0, "power-of-two EP through 64 required");
    const uint32_t phase = status & 1u;
    const bool negative = (status & 2u) != 0;
#if DG_MEGAMOE_OWNER_MODE
    constexpr uint32_t kBanks = (kWorldSize + layout::kScatterRanksPerBank - 1) / layout::kScatterRanksPerBank;
    constexpr uint32_t kWriters = kWorldSize < layout::kScatterRanksPerBank ? kWorldSize : layout::kScatterRanksPerBank;
    uint64_t writer_mask = UINT64_MAX;
    #pragma unroll
    for (uint32_t bank = 0; bank < kBanks; ++bank) {
        int32_t observed;
        do {
            observed = tix::ld_acq_sys(&workspace.phase_signal[phase][bank]);
        } while ((observed & 15) != (negative ? 0 : kWriters));
        constexpr uint32_t kMask = (1u << kWriters) - 1u;
        const uint64_t disabled = (~(uint32_t(observed) >> 4)) & kMask;
        writer_mask &= ~(disabled << (bank * layout::kScatterRanksPerBank));
    }
    return writer_mask;
#else
    while (tix::ld_acq_sys(&workspace.phase_signal[phase][0]) !=
           (negative ? 0 : static_cast<int32_t>(kWorldSize))) { }
    return UINT64_MAX;
#endif
}

// Release all local P2P stores, then acquire peer completion before reduce.
template <uint32_t kWorldSize>
__device__ __forceinline__ void fp4_scatter_release(
        layout::ScatterSignals& workspace,
        const layout::SymBuffer<kWorldSize>& sym_buffer,
        uint32_t num_blocks, uint32_t block_idx, uint32_t thread_idx,
        bool bulk_writer) {
    // PPU fences drain only the calling warp; each warp leader releases its stores.
    // The following grid sync joins these releases before peer notification.
    __syncwarp();
    if ((thread_idx & 31u) == 0) {
        __ppu_threadfence_system_uncache();
    }

    grid_sync(
        workspace, num_blocks, block_idx, thread_idx);

    if (block_idx == 0) {
        const int32_t status = workspace.phase_counter & 3;
        const uint32_t phase = status & 1;
        const bool negative = (status >> 1) != 0;
#if DG_MEGAMOE_OWNER_MODE
        static_assert(kWorldSize <= layout::kNumMaxRanks);
        const uint32_t rank_idx = sym_buffer.rank_idx;
        const uint32_t bank = rank_idx / layout::kScatterRanksPerBank;
        const int32_t old_bulk = __ldcg(workspace.writer_previous + phase);
        const int32_t notification = (negative ? -1 : 1) +
            (int32_t(bulk_writer) - old_bulk) * (1 << (rank_idx % layout::kScatterRanksPerBank + 4));
#else
        constexpr uint32_t bank = 0;
        const int32_t notification = negative ? -1 : 1;
#endif


        if (thread_idx < kWorldSize) {
            auto* remote_workspace = sym_buffer.map(&workspace, thread_idx);
            tix::red_add_rel_sys(
                &remote_workspace->phase_signal[phase][bank], notification);
        }
        __syncthreads();

        if (thread_idx == 0) {
#if DG_MEGAMOE_OWNER_MODE
            workspace.writer_previous[phase] = bulk_writer ? 1 : 0;
#endif
            tix::red_add(&workspace.phase_counter, 1);
            fp4_scatter_acquire<kWorldSize>(workspace, status);
        }
    }

    // Multi-node ranks join all CTA leaders after the peer acquire.
    if constexpr (kWorldSize != 8) {
        grid_sync(
            workspace, num_blocks, block_idx, thread_idx);
    }
}

}  // namespace deep_gemm::comm
