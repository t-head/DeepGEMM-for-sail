#pragma once

#include <cstdint>
#include <cutlass/cutlass.h>
#include <deep_gemm/tix/sys_atomics.cuh>

namespace deep_gemm::mega_moe {

// Rank-local monotonic tile queue. Reset before dispatch's grid rendezvous.
// claim() is CTA-collective but does not synchronize: the caller must
// publish the shared ticket with a CTA barrier and retire every read before
// the next claim (or before reusing its shared-memory storage for MMA).
struct WorkQueue {
    uint32_t* counter;
    uint32_t* shared_ticket;

    CUTLASS_DEVICE void claim(uint32_t thread, uint32_t count = 1) const {
        if (thread == 0) *shared_ticket = atomicAdd(counter, count);
    }
    CUTLASS_DEVICE void prime(uint32_t thread, uint32_t count = 1) const {
        claim(thread, count);
        __syncthreads();
    }
    CUTLASS_DEVICE uint32_t current() const { return *shared_ticket; }
};

// Arrival flags only increase during one dispatch. A warp acquires a window
// once; the CTA rendezvous publishes readiness and retires shared ticket reads.
// Keep this cache local to the kernel invocation, never in reusable workspace.
struct ArrivalCache {
    uint32_t base = ~uint32_t{0};
    uint32_t mask = 0;

    CUTLASS_DEVICE void wait(const uint32_t* arrivals, uint32_t first, uint32_t last,
                             uint32_t total, uint32_t required = 1) {
        if (threadIdx.x / 32 == 0) {
            const uint32_t lane = threadIdx.x % 32;
            for (uint32_t block = first; block <= last; ++block) {
                const uint32_t window = block & ~uint32_t{31};
                const uint32_t bit = 1u << (block & 31u);
                if (base != window || !(mask & bit)) {
                    const uint32_t peer = window + lane;
                    const bool ready = peer < total && tix::ld_acq(arrivals + peer) >= required;
                    mask = __ballot_sync(0xffffffffu, ready);
                    __syncwarp();
                    base = window;
                    if (!(mask & bit)) {
                        while (tix::ld_acq(arrivals + block) < required) {}
                        mask |= bit;
                    }
                }
            }
        }
        __syncthreads();
    }
};

} // namespace deep_gemm::mega_moe
