#pragma once

#include <cstdint>

namespace deep_gemm::tix {

// HGGC inline-asm lane-id and shared-memory load primitives, lifted verbatim
// from impls/mqa_logits_utils.cuh.
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

__device__ __forceinline__ void red_add_rel_sys(
        int32_t* address, int32_t value) {
    asm volatile("ppu.atom.add.release.sys.global.s32 _, [%0], %1;"
                 :: "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ uint32_t ld_acq(
        const uint32_t* address) {
    uint32_t value;
    asm volatile("ppu.ld.acquire.gpu.global.b32 %0, [%1];"
                 : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ int32_t ld_acq_sys(
        const int32_t* address) {
    int32_t value;
    asm volatile("ppu.ld.acquire.sys.global.s32 %0, [%1];"
                 : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ uint32_t atomic_add_rel(
        uint32_t* address, uint32_t value) {
    uint32_t old_value;
    asm volatile("ppu.atom.add.release.gpu.global.u32 %0, [%1], %2;"
                 : "=r"(old_value) : "l"(address), "r"(value) : "memory");
    return old_value;
}

__device__ __forceinline__ void red_add(
        int32_t* address, int32_t value) {
    asm volatile("ppu.atom.add.relaxed.gpu.global.s32 _, [%0], %1;"
                 :: "l"(address), "r"(value) : "memory");
}

} // namespace deep_gemm::tix
