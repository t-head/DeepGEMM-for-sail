#pragma once

#include <cstdint>
#include <cutlass/cutlass.h>
#include <deep_gemm/scheduler/sm89_mega_moe.cuh>
#include <deep_gemm/tix/sys_atomics.cuh>

namespace deep_gemm::mega_moe {

using WorkQueue = sched::MegaMoEWorkQueue;

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
