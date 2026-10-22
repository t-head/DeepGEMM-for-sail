#pragma once

#include <cstdint>
#include <cutlass/cutlass.h>
#include <deep_gemm/ptx/sys_atomics.cuh>

namespace deep_gemm::mega_moe {

// Rank-local monotonic tile queue. Reset counter before the arrival-reset grid
// barrier. claim() is CTA-collective but does not synchronize: the caller must
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

} // namespace deep_gemm::mega_moe
