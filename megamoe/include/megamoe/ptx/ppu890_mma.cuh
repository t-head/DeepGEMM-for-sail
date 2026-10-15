#pragma once

#include <megamoe/common/exception.cuh>
#include <megamoe/common/tie.cuh>
#include <megamoe/ptx/ppu890_copy.cuh>
#include <cstdint>

namespace megamoe::ptx::ppu890 {

struct FP4MmaM16N16K64 {
    __device__ __forceinline__ static void mma(float& d0,
                                               float& d1,
                                               float& d2,
                                               float& d3,
                                               float& d4,
                                               float& d5,
                                               float& d6,
                                               float& d7,
                                               uint32_t a0,
                                               uint32_t a1,
                                               uint32_t a2,
                                               uint32_t a3,
                                               uint32_t b0,
                                               uint32_t b1,
                                               uint32_t b2,
                                               uint32_t b3,
                                               float c0,
                                               float c1,
                                               float c2,
                                               float c3,
                                               float c4,
                                               float c5,
                                               float c6,
                                               float c7,
                                               uint32_t s0,
                                               uint32_t s1,
                                               uint32_t s2,
                                               uint32_t s3) {
        asm volatile(
            "ppu.mma.mx.sync.aligned.m16n16k64.row.col.f32.f4.f4.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, "
            "{%8, %9, %10, %11}, "
            "{%12, %13, %14, %15}, "
            "{%16, %17, %18, %19, %20, %21, %22, %23}, "
            "{%24, %25}, %26, %27, 0;\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
            : "r"(a0),
              "r"(a1),
              "r"(a2),
              "r"(a3),
              "r"(b0),
              "r"(b1),
              "r"(b2),
              "r"(b3),
              "f"(c0),
              "f"(c1),
              "f"(c2),
              "f"(c3),
              "f"(c4),
              "f"(c5),
              "f"(c6),
              "f"(c7),
              "r"(s0),
              "r"(s1),
              "r"(s2),
              "r"(s3));
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int kNumKIters, int kMmaMTilesPerWarp, int kMmaMTileRowStride = 16, class sf_fn_t>
__device__ __forceinline__ void mma_regpipe_swzl_fp4_native_accum_m_loop(
    const uint8_t* smem_a, const uint8_t* smem_b, int m_base, int k_base, sf_fn_t&& sf_fn, float* accum) {
    static constexpr int kNumN16Tiles = BLOCK_N / 16;
    static constexpr int CUBE_W = BLOCK_K / 4;
    DG_STATIC_ASSERT(BLOCK_N % 16 == 0, "m16n16k64 needs BLOCK_N divisible by 16");
    #pragma unroll
    for (int ki = 0; ki < kNumKIters; ++ki) {
        const int k_element_offset = k_base + ki * 64;
        uint32_t s0_by_m_tile[kMmaMTilesPerWarp];
        uint32_t s1_by_n16_tile[kNumN16Tiles];
        #pragma unroll
        for (int t = 0; t < kMmaMTilesPerWarp; ++t) {
            uint32_t s1_unused = 0;
            sf_fn(t, 0, ki, s0_by_m_tile[t], s1_unused);
        }
        #pragma unroll
        for (int nt = 0; nt < kNumN16Tiles; ++nt) {
            uint32_t s0_unused = 0;
            sf_fn(0, nt, ki, s0_unused, s1_by_n16_tile[nt]);
        }
        uint32_t b[2][4];
        ptx::ppu890::ldmatrix_swzl_x4_b16<BLOCK_N, CUBE_W, true>(b[0][0], b[0][1], b[0][2], b[0][3], smem_b, 0, k_element_offset / 4);
        // A fragment (m-tile, k) does not depend on the n16 tile, so one k-step's A tiles are loaded
        // once and held across the whole n16 sweep. Refetching them per n16 tile issued 4x the
        // ldmatrix ops for the same data and put 4x the bulk smem reads through the TSM queue.
        uint32_t a[kMmaMTilesPerWarp][4];
        #pragma unroll
        for (int t = 0; t < kMmaMTilesPerWarp; ++t) {
            ptx::ppu890::ldmatrix_swzl_x4_b16<BLOCK_M, CUBE_W>(
                a[t][0], a[t][1], a[t][2], a[t][3], smem_a, m_base + t * kMmaMTileRowStride, k_element_offset / 4);
        }
        #pragma unroll
        for (int nt = 0; nt < kNumN16Tiles; ++nt) {
            if (nt + 1 < kNumN16Tiles) {
                ptx::ppu890::ldmatrix_swzl_x4_b16<BLOCK_N, CUBE_W, true>(b[(nt + 1) & 1][0],
                                                                         b[(nt + 1) & 1][1],
                                                                         b[(nt + 1) & 1][2],
                                                                         b[(nt + 1) & 1][3],
                                                                         smem_b,
                                                                         (nt + 1) * 16,
                                                                         k_element_offset / 4);
            }
            const uint32_t b0 = b[nt & 1][0], b1 = b[nt & 1][1];
            const uint32_t b2 = b[nt & 1][2], b3 = b[nt & 1][3];
            const uint32_t s1 = s1_by_n16_tile[nt];
            #pragma unroll
            for (int t = 0; t < kMmaMTilesPerWarp; ++t) {
                float* d = accum + (t * kNumN16Tiles + nt) * 8;
                const uint32_t s0 = s0_by_m_tile[t];
                FP4MmaM16N16K64::mma(d[0],
                                     d[1],
                                     d[2],
                                     d[3],
                                     d[4],
                                     d[5],
                                     d[6],
                                     d[7],
                                     a[t][0],
                                     a[t][1],
                                     a[t][2],
                                     a[t][3],
                                     b0,
                                     b1,
                                     b2,
                                     b3,
                                     d[0],
                                     d[1],
                                     d[2],
                                     d[3],
                                     d[4],
                                     d[5],
                                     d[6],
                                     d[7],
                                     s0,
                                     s1,
                                     0u,
                                     0u);
            }
        }
    }
}
}  // namespace megamoe::ptx::ppu890
