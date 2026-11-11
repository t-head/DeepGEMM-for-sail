#pragma once

#include <cstdint>
#include <hggc_runtime.h>
#include <math.h>

namespace deep_gemm {

// Keep this POD byte-for-byte identical to FP32GemmRuntime::Params.
struct FP32GemmParams {
    const float* a;
    const float* b;
    float* d;
    int m, n, k;
};

// A warp partitions K and reuses each A load across NPerWarp output columns.
// Operands remain IEEE FP32 throughout; no TF32/BF16 conversion is involved.
template <int NPerWarp, int BlockWarps, bool Vectorized,
          int StaticM = 0, int StaticN = 0, int StaticK = 0, int Unroll = 1>
__device__ void fp32_gemm_nt(const FP32GemmParams p) {
    static_assert(Unroll > 0, "FP32 unroll must be positive");
    const int m = StaticM > 0 ? StaticM : p.m;
    const int n = StaticN > 0 ? StaticN : p.n;
    const int k = StaticK > 0 ? StaticK : p.k;
    const int lane = threadIdx.x & 31;
    const int warp = int(blockIdx.x) * BlockWarps + (threadIdx.x >> 5);
    const int column_groups = (n - 1) / NPerWarp + 1;
    const int row = warp / column_groups;
    const int col0 = (warp % column_groups) * NPerWarp;
    constexpr bool AllWarpsValid = StaticM > 0 && StaticN > 0 &&
        (StaticM * ((StaticN + NPerWarp - 1) / NPerWarp)) % BlockWarps == 0;
    constexpr bool AllColumnsValid = StaticN > 0 && StaticN % NPerWarp == 0;
    if constexpr (!AllWarpsValid)
        if (row >= m) return;  // A whole warp exits together.
    const float* row_a = p.a + int64_t(row) * k;
    float accum[NPerWarp] = {};
    if constexpr (Vectorized) {
        // Independent FMA chains hide arithmetic latency for the short, narrow
        // FP32 product. Keep all four components at full FP32 precision.
        float accum_y[NPerWarp] = {}, accum_z[NPerWarp] = {}, accum_w[NPerWarp] = {};
        constexpr bool FullChunks = StaticK > 0 && StaticK % (32 * 4 * Unroll) == 0;
        // Limit code growth to Unroll iterations while exposing independent
        // loads to the scheduler. The exact 4096-K case needs no inner tail check.
        #pragma unroll 1
        for (int kk = lane * 4; kk < k; kk += 32 * 4 * Unroll) {
            #pragma unroll
            for (int u = 0; u < Unroll; ++u) {
                const int offset = kk + u * 32 * 4;
                if (FullChunks || offset < k) {
                    const float4 av = *reinterpret_cast<const float4*>(row_a + offset);
                    #pragma unroll
                    for (int c = 0; c < NPerWarp; ++c) {
                        if (AllColumnsValid || col0 + c < n) {
                            const float4 bv = *reinterpret_cast<const float4*>(p.b + int64_t(col0 + c) * k + offset);
                            accum[c] = fmaf(av.x, bv.x, accum[c]);
                            accum_y[c] = fmaf(av.y, bv.y, accum_y[c]);
                            accum_z[c] = fmaf(av.z, bv.z, accum_z[c]);
                            accum_w[c] = fmaf(av.w, bv.w, accum_w[c]);
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int c = 0; c < NPerWarp; ++c)
            accum[c] = (accum[c] + accum_y[c]) + (accum_z[c] + accum_w[c]);
    } else {
        constexpr bool FullChunks = StaticK > 0 && StaticK % (32 * Unroll) == 0;
        #pragma unroll 1
        for (int kk = lane; kk < k; kk += 32 * Unroll) {
            #pragma unroll
            for (int u = 0; u < Unroll; ++u) {
                const int offset = kk + u * 32;
                if (FullChunks || offset < k) {
                    const float av = row_a[offset];
                    #pragma unroll
                    for (int c = 0; c < NPerWarp; ++c)
                        if (AllColumnsValid || col0 + c < n)
                            accum[c] = fmaf(av, p.b[int64_t(col0 + c) * k + offset], accum[c]);
                }
            }
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int c = 0; c < NPerWarp; ++c)
            accum[c] += __shfl_down_sync(0xffffffff, accum[c], offset);
    }
    if (lane == 0) {
        #pragma unroll
        for (int c = 0; c < NPerWarp; ++c)
            if (AllColumnsValid || col0 + c < n)
                p.d[int64_t(row) * n + col0 + c] = accum[c];
    }
}

} // namespace deep_gemm
