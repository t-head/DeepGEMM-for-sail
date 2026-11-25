#pragma once

#include <cstdint>

namespace deep_gemm::tix {

__forceinline__ __device__ uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm("ppu.mov.u32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

#if defined(__HGGC__)

// All Threads participate. Synchronize before reusing the per-warp scratch storage.
template <int Threads>
__device__ __forceinline__ float block_reduce_max(float value, float* warp_max, float minimum) {
    static_assert(Threads % 32 == 0 && Threads <= 1024);
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    const int lane = threadIdx.x % 32;
    if (lane == 0) warp_max[threadIdx.x / 32] = value;
    __syncthreads();
    value = lane < Threads / 32 ? warp_max[lane] : minimum;
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    return value;
}

// Approximate reciprocal with flush-to-zero semantics for MXFP4 scales.
__device__ __forceinline__ float reciprocal_approx_ftz(float value) {
    float result;
    asm("ppu.rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

// PPU 1.5 uses ordered integer keys for FP32 max; inputs must not be NaN.
__device__ __forceinline__ float reduce_max_sync(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t key = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
    const uint32_t best = __reduce_max_sync(0xffffffff, key);
    return __uint_as_float(best & 0x80000000u ? best ^ 0x80000000u : ~best);
}

__device__ __forceinline__ int atomic_add_release_global(int* addr, int value) {
    int ret;
    asm volatile ("ppu.atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(addr), "r"(value));
    return ret;
}

#endif  // __HGGC__

} // namespace deep_gemm::tix
