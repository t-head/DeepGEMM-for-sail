#pragma once

#include <cstdint>

// HGGC inline-asm lane-id and shared-memory load primitives, lifted verbatim
// from impls/mqa_logits_utils.cuh.

namespace deep_gemm::tix {

__forceinline__ __device__ uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm("ppu.mov.u32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

__device__ __forceinline__ float ld_shared(const float* ptr) {
    float ret;
    asm volatile("ppu.ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

__device__ __inline__ uint32_t ld_shared_u32(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ppu.ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

} // namespace deep_gemm::tix
