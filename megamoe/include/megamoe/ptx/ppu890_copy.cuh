#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace megamoe::ptx::ppu890 {

__device__ __forceinline__ void aiu_load_2d_linear_b8(
    void* smem_dst, const void* gmem_src, int dim_w, int dim_h, int cube_w, int cube_h, int coord_w, int coord_h) {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION >= 10500
    const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    const uint64_t stride_w = static_cast<uint64_t>(dim_w) * 1;
    const uint64_t stride_n = stride_w * static_cast<uint64_t>(dim_h);
    const int swzl_mode = (cube_w * 1 == 128) ? 0 : 1;
    asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.linear.b8 "
        "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n" ::"r"(smem_addr),
        "l"(gmem_src),
        "r"(dim_w),
        "r"(dim_h),
        "r"(1),
        "r"(cube_w),
        "r"(cube_h),
        "r"(1),
        "l"(stride_w),
        "l"(stride_n),
        "r"(coord_w),
        "r"(coord_h),
        "r"(0),
        "r"(swzl_mode));
#endif
}

__device__ __forceinline__ void aiu_load_2d_swzl_b8(
    void* smem_dst, const void* gmem_src, int dim_w, int dim_h, int cube_w, int cube_h, int coord_w, int coord_h) {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION >= 10500
    const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    const uint64_t stride_w = static_cast<uint64_t>(dim_w) * 1;
    const uint64_t stride_n = stride_w * static_cast<uint64_t>(dim_h);
    const int swzl_mode = (cube_w * 1 == 128) ? 0 : 1;
    asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.L2::128B.padz.swzl.b8 "
        "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n" ::"r"(smem_addr),
        "l"(gmem_src),
        "r"(dim_w),
        "r"(dim_h),
        "r"(1),
        "r"(cube_w),
        "r"(cube_h),
        "r"(1),
        "l"(stride_w),
        "l"(stride_n),
        "r"(coord_w),
        "r"(coord_h),
        "r"(0),
        "r"(swzl_mode));
#endif
}

template <int CUBE_H, int CUBE_W, bool kSwap = false>
__device__ __forceinline__ void ldmatrix_swzl_x4_b16(
    uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3, const void* smem_base, int coord_h, int coord_w) {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION >= 10500
    constexpr int row_stride = 8 * CUBE_W * 2 / 16;
    constexpr int lbo = kSwap ? row_stride : 1;
    constexpr int sbo = kSwap ? 1 : row_stride;
    constexpr int swzl_mode = (CUBE_W * 2 == 128) ? 0 : 1;
    const uint16_t* base = reinterpret_cast<const uint16_t*>(smem_base) + coord_h * CUBE_W + coord_w;
    const int smem_offset_16byte_units = static_cast<int>(static_cast<uint32_t>(__cvta_generic_to_shared(base)) / 16u);
    asm volatile("ppu.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4], %5, %6, %7;\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "l"(smem_offset_16byte_units), "r"(lbo), "r"(sbo), "r"(swzl_mode));
#endif
}

}  // namespace megamoe::ptx::ppu890
