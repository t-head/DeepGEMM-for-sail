#pragma once

#include <cuda_bf16.h>

#include <megamoe/common/exception.cuh>
#include <cuda/std/cstdint>

namespace megamoe::ptx {

DG_DEVICE uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
}

DG_DEVICE void sync_threads_partial(const uint32_t& num_threads, const uint32_t& barrier_idx) {
    asm volatile("barrier.sync %0, %1;" : : "r"(barrier_idx), "r"(num_threads));
}

template <typename dtype_t>
DG_DEVICE dtype_t broadcast_from_lane(dtype_t send_value, const uint32_t& src_lane_idx) {
    DG_STATIC_ASSERT(sizeof(dtype_t) % sizeof(uint32_t) == 0, "");
    const auto send_int_values = reinterpret_cast<uint32_t*>(&send_value);
    dtype_t recv_dtype;
    auto recv_int_values = reinterpret_cast<uint32_t*>(&recv_dtype);
    #pragma unroll
    for (uint32_t i = 0; i < sizeof(dtype_t) / sizeof(uint32_t); ++i)
        recv_int_values[i] = __shfl_sync(0xffffffff, send_int_values[i], static_cast<int>(src_lane_idx));
    return recv_dtype;
}

}  // namespace megamoe::ptx
