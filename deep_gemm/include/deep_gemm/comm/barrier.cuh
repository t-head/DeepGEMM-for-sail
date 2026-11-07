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
        layout::MegaMoESignals& signals,
        uint32_t num_blocks, uint32_t block_idx, uint32_t thread_idx) {
    constexpr uint32_t kFinishTag = 0x80000000u;
    __syncthreads();
    if (thread_idx == 0) {
        auto* counter = &signals.grid_sync_count;
        const uint32_t old_value = tix::atomic_add_rel(
            counter, block_idx == 0 ? kFinishTag - (num_blocks - 1) : 1);
        uint32_t new_value;
        do {
            new_value = tix::ld_acq(counter);
        } while (((new_value ^ old_value) & kFinishTag) == 0);
    }
    __syncthreads();
}

// Call after all local writers have released their stores.
template <uint32_t kWorldSize>
__device__ __forceinline__ void rank_arrive(
        const layout::SymBuffer<kWorldSize>& sym_buffer,
        uint32_t* arrivals, uint32_t notification) {
    if (threadIdx.x < kWorldSize)
        __stbl(sym_buffer.map(arrivals, threadIdx.x) + sym_buffer.rank_idx, notification);
}

// Each rank owns one slot. All threads in the CTA participate in the wait.
template <uint32_t kWorldSize, bool kWithWriterMask = false>
__device__ __forceinline__ void rank_barrier(
        const volatile uint32_t* arrivals, uint32_t generation,
        uint32_t* writer_mask = nullptr) {
    uint32_t observed = 0;
    if (threadIdx.x < kWorldSize && generation != 0) {
        do {
            if constexpr (kWithWriterMask) {
                observed = tix::ld_ready_flag_uncached(arrivals + threadIdx.x);
            } else {
                observed = arrivals[threadIdx.x];
            }
        } while ((kWithWriterMask ? observed >> 1 : observed) != generation);
    }
    if constexpr (kWithWriterMask) {
        if (threadIdx.x < 64) {
            const uint32_t mask = __ballot_sync(0xffffffffu,
                threadIdx.x >= kWorldSize || (observed & 1u));
            if ((threadIdx.x & 31u) == 0)
                writer_mask[threadIdx.x / 32] = mask;
        }
    }
    __syncthreads();
}

// Release all local P2P stores before notifying every peer.
template <uint32_t kWorldSize>
__device__ __forceinline__ void fp4_scatter_release(
        layout::MegaMoESignals& signals,
        const layout::SymBuffer<kWorldSize>& sym_buffer,
        uint32_t num_blocks, uint32_t block_idx, uint32_t thread_idx,
        uint32_t generation, bool bulk_writer) {
    // PPU fences drain only the calling warp; each warp leader releases its stores.
    // The following grid sync joins these releases before peer notification.
    __syncwarp();
    if ((thread_idx & 31u) == 0) {
        __ppu_threadfence_system_uncache();
    }

    grid_sync(
        signals, num_blocks, block_idx, thread_idx);

    if (block_idx == 0) {
        const uint32_t notification = (generation << 1) | uint32_t(bulk_writer);
        rank_arrive(sym_buffer, signals.scatter_arrivals, notification);
    }
}

}  // namespace deep_gemm::comm
