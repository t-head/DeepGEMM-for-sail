#pragma once

#include <cstdint>

namespace deep_gemm::tix {

__forceinline__ __device__ uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm("ppu.mov.u32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

#if defined(__HGGC__)

// Approximate reciprocal with flush-to-zero semantics for MXFP4 scales.
__device__ __forceinline__ float reciprocal_approx_ftz(float value) {
    float result;
    asm("ppu.rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ int atomic_add_release_global(int* addr, int value) {
    int ret;
    asm volatile ("ppu.atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(addr), "r"(value));
    return ret;
}

#endif  // __HGGC__

} // namespace deep_gemm::tix
