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

// Narrow products reuse both inputs across many outputs. One bounded copy
// aligns rows to the warp kernel's 32 lanes x float4 x 4 unrolled K chunk.
// Zero padding preserves FP32 precision and removes repeated scalar/tail loads.
// The host restricts this to M,N<129 and 1024<=K<=8192 on 810E.
__device__ void fp32_gemm_pack_inputs(const FP32GemmParams p, int64_t block_index) {
    const int padded_k=(p.k+511)/512*512;
    const int64_t i=block_index*blockDim.x+threadIdx.x;
    if(i>=(int64_t(p.m)+p.n)*padded_k)return;
    const int row=int(i/padded_k),kk=int(i%padded_k);
    const float* input=row<p.m ? p.a+int64_t(row)*p.k : p.b+int64_t(row-p.m)*p.k;
    p.d[i]=kk<p.k ? input[kk] : 0.0f;
}

// Cooperative K-contiguous loads are transposed in shared memory. Two padding
// elements distribute the float4 stores across banks, while adjacent M pairs
// let each thread reuse its operands across a register tile. The next K tile is
// prefetched before computing the current one. Scalar loads cover K tails and
// misaligned storage without changing the FP32 FMA accumulation order.
template <bool Vectorized, int BM = 32, int BN = 64, int Splits = 1, bool Persistent = false>
__device__ void fp32_gemm_tiled_nt(const FP32GemmParams p, int64_t block_index) {
    static_assert(BM == 32 || BM == 64);
    static_assert(BN == 32 || BN == 64);
    static_assert(Splits > 0 && Splits <= 32);
    constexpr int BK = 16, Threads = 128, TM = BM / 8, TN = BN / 16;
    constexpr int LA = BM * BK / (4 * Threads), LB = BN * BK / (4 * Threads);
    __shared__ float sa[BK][BM + 2], sb[BK][BN + 2];
    const int tid = int(threadIdx.x), ty = tid / 16, tx = tid % 16;
    int mi = int(blockIdx.y) * BM, ni = int(blockIdx.x) * BN, split = int(blockIdx.z);
    if constexpr (Persistent) {
        const int64_t tiles_m = (int64_t(p.m) + BM - 1) / BM;
        const int64_t tiles_n = (int64_t(p.n) + BN - 1) / BN;
        split = int(block_index / (tiles_m * tiles_n));
        mi = int((block_index / tiles_n) % tiles_m) * BM;
        ni = int(block_index % tiles_n) * BN;
    }
    // Partition whole BK tiles, retaining a possibly partial last tile. Each
    // partition writes FP32 partials; no conversion precedes their reduction.
    const int64_t k_tiles = (int64_t(p.k) + BK - 1) / BK;
    const int begin_k = int(k_tiles * split / Splits) * BK;
    const int padded_end = int(k_tiles * (split + 1) / Splits) * BK;
    const int end_k = padded_end < p.k ? padded_end : p.k;
    float* output = p.d + int64_t(split) * p.m * p.n;
    float accum[TM][TN] = {};
    float4 ra[LA], rb[LB];
    auto load = [&](int base) {
        #pragma unroll
        for (int j = 0; j < LA; ++j) {
            const int i = tid + j * Threads, row = i / (BK / 4), kk = base + (i % (BK / 4)) * 4;
            ra[j] = {0, 0, 0, 0};
            if (mi + row < p.m && kk < end_k) {
                const float* src = p.a + int64_t(mi + row) * p.k + kk;
                if constexpr (Vectorized) ra[j] = *reinterpret_cast<const float4*>(src);
                else ra[j] = {src[0], kk + 1 < end_k ? src[1] : 0.0f,
                              kk + 2 < end_k ? src[2] : 0.0f, kk + 3 < end_k ? src[3] : 0.0f};
            }
        }
        #pragma unroll
        for (int j = 0; j < LB; ++j) {
            const int i = tid + j * Threads, row = i / (BK / 4), kk = base + (i % (BK / 4)) * 4;
            rb[j] = {0, 0, 0, 0};
            if (ni + row < p.n && kk < end_k) {
                const float* src = p.b + int64_t(ni + row) * p.k + kk;
                if constexpr (Vectorized) rb[j] = *reinterpret_cast<const float4*>(src);
                else rb[j] = {src[0], kk + 1 < end_k ? src[1] : 0.0f,
                              kk + 2 < end_k ? src[2] : 0.0f, kk + 3 < end_k ? src[3] : 0.0f};
            }
        }
    };
    load(begin_k);
    for (int begin = begin_k; begin < end_k; begin += BK) {
        #pragma unroll
        for (int j = 0; j < LA; ++j) {
            const int i = tid + j * Threads, row = i / (BK / 4), kk = (i % (BK / 4)) * 4;
            sa[kk][row] = ra[j].x; sa[kk + 1][row] = ra[j].y;
            sa[kk + 2][row] = ra[j].z; sa[kk + 3][row] = ra[j].w;
        }
        #pragma unroll
        for (int j = 0; j < LB; ++j) {
            const int i = tid + j * Threads, row = i / (BK / 4), kk = (i % (BK / 4)) * 4;
            sb[kk][row] = rb[j].x; sb[kk + 1][row] = rb[j].y;
            sb[kk + 2][row] = rb[j].z; sb[kk + 3][row] = rb[j].w;
        }
        __syncthreads();
        if (begin + BK < end_k) load(begin + BK);
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float av[TM], bv[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) av[i] = sa[kk][ty * 2 + i % 2 + (i / 2) * 16];
            #pragma unroll
            for (int j = 0; j < TN; ++j) bv[j] = sb[kk][tx + j * 16];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    accum[i][j] = fmaf(av[i], bv[j], accum[i][j]);
        }
        // All warps finish reading the current tile before any shared overwrite.
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int row = mi + ty * 2 + i % 2 + (i / 2) * 16, col = ni + tx + j * 16;
            if (row < p.m && col < p.n) output[int64_t(row) * p.n + col] = accum[i][j];
        }
}

template <int Splits>
__device__ void fp32_gemm_split_reduce(const FP32GemmParams p, int64_t block_index) {
    const int64_t count = int64_t(p.m) * p.n;
    const int64_t index = block_index * blockDim.x + threadIdx.x;
    if (index < count) {
        float sum = 0.0f;
        #pragma unroll
        for (int part = 0; part < Splits; ++part)
            sum += p.a[int64_t(part) * count + index];
        p.d[index] = sum;
    }
}

// A warp partitions K and reuses each A load across NPerWarp output columns.
// Operands remain IEEE FP32 throughout; no TF32/BF16 conversion is involved.
template <int NPerWarp, int BlockWarps, bool Vectorized,
          int StaticM = 0, int StaticN = 0, int StaticK = 0, int Unroll = 1, bool ScalarQuad = false>
__device__ void fp32_gemm_nt(const FP32GemmParams p, int64_t block_index) {
    static_assert(Unroll > 0, "FP32 unroll must be positive");
    const int m = StaticM > 0 ? StaticM : p.m;
    const int n = StaticN > 0 ? StaticN : p.n;
    const int k = StaticK > 0 ? StaticK : p.k;
    const int lane = threadIdx.x & 31;
    const int warp = int(block_index) * BlockWarps + (threadIdx.x >> 5);
    const int column_groups = (n - 1) / NPerWarp + 1;
    const int row = warp / column_groups;
    const int col0 = (warp % column_groups) * NPerWarp;
    constexpr bool AllWarpsValid = StaticN > 0 &&
        (((StaticN + NPerWarp - 1) / NPerWarp) % BlockWarps == 0 ||
         (StaticM > 0 && (StaticM * ((StaticN + NPerWarp - 1) / NPerWarp)) % BlockWarps == 0));
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
        // loads to the scheduler. Complete K chunks need no inner tail check.
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
    } else if constexpr (ScalarQuad) {
        float accum_y[NPerWarp] = {}, accum_z[NPerWarp] = {}, accum_w[NPerWarp] = {};
        constexpr bool FullChunks = StaticK > 0 && StaticK % (32 * 4 * Unroll) == 0;
        #pragma unroll 1
        for (int kk = lane * 4; kk < k; kk += 32 * 4 * Unroll) {
            #pragma unroll
            for (int u = 0; u < Unroll; ++u) {
                const int offset = kk + u * 32 * 4;
                if (FullChunks || offset < k) {
                    const float ax=row_a[offset];
                    const float ay=FullChunks || offset+1<k ? row_a[offset+1] : 0.0f;
                    const float az=FullChunks || offset+2<k ? row_a[offset+2] : 0.0f;
                    const float aw=FullChunks || offset+3<k ? row_a[offset+3] : 0.0f;
                    #pragma unroll
                    for (int c = 0; c < NPerWarp; ++c) {
                        if (AllColumnsValid || col0 + c < n) {
                            const float* bp=p.b+int64_t(col0+c)*k+offset;
                            const float bx=bp[0];
                            const float by=FullChunks || offset+1<k ? bp[1] : 0.0f;
                            const float bz=FullChunks || offset+2<k ? bp[2] : 0.0f;
                            const float bw=FullChunks || offset+3<k ? bp[3] : 0.0f;
                            accum[c]=fmaf(ax,bx,accum[c]);
                            accum_y[c]=fmaf(ay,by,accum_y[c]);
                            accum_z[c]=fmaf(az,bz,accum_z[c]);
                            accum_w[c]=fmaf(aw,bw,accum_w[c]);
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int c = 0; c < NPerWarp; ++c)
            accum[c]=(accum[c]+accum_y[c])+(accum_z[c]+accum_w[c]);
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
