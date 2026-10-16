// SPDX-License-Identifier: Apache-2.0
// MXFP4 pipeline adapted from SGLang via DeepGemm17 origin/yidian (2354e23).
// Copyright 2023-2024 SGLang Team; see silu_quant_mxfp4.LICENSE.
#pragma once
#include <hggc_bf16.h>
#include <hggc_runtime.h>
#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>
#include <math.h>
#include <deep_gemm/activation/arguments.hpp>

namespace deep_gemm {
__device__ __forceinline__ float silu_and_mul_value(float gate, float up, float limit) {
    if (limit > 0.0f) {
        const float bound = float(cutlass::bfloat16_t(limit));
        gate = fminf(gate, bound);
        up = fminf(fmaxf(up, -bound), bound);
    }
    return (gate * __ppu_sgmdf(gate)) * up;
}

__device__ __forceinline__ float situ_and_mul_value(float gate, float up, float beta, float linear_beta) {
    const float gated = beta * __ppu_tanhf(gate / beta) * __ppu_sgmdf(gate);
    const float linear = linear_beta > 0.0f ? linear_beta * __ppu_tanhf(up / linear_beta) : up;
    return gated * linear;
}
} // namespace deep_gemm

namespace deep_gemm {
template<int N>
__device__ __forceinline__ void activation_load_eight(const cutlass::bfloat16_t* input,
        int64_t row, int col, cutlass::bfloat16_t* gate, cutlass::bfloat16_t* up) {
    const auto* g = input + row * (2 * N) + col;
    const auto* u = g + N;
    if constexpr (N % 8 == 0) {
        if (col < N && reinterpret_cast<uintptr_t>(g) % 16 == 0) {
            *reinterpret_cast<uint4*>(gate) = *reinterpret_cast<const uint4*>(g);
            *reinterpret_cast<uint4*>(up) = *reinterpret_cast<const uint4*>(u);
            return;
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        gate[i] = col + i < N ? g[i] : cutlass::bfloat16_t(0.0f);
        up[i] = col + i < N ? u[i] : cutlass::bfloat16_t(0.0f);
    }
}

__device__ __forceinline__ void activation_pack_fp8_eight(const float* values, float inverse, uint16_t* packed) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float lo = fminf(fmaxf(values[2*i] * inverse, -448.0f), 448.0f);
        const float hi = fminf(fmaxf(values[2*i+1] * inverse, -448.0f), 448.0f);
        asm("ppu.cvt.rtte.satfinite.e4m3x2.f32 %0, %1, %2;\n"
            : "=h"(packed[i]) : "f"(hi), "f"(lo));
    }
}

// BF16 and group128 FP8 use one warp per 256 columns, eight elements per thread.
template<int N, bool FP8Block, bool Situ, bool Masked>
__device__ void activation_tiled(const ActivationParams& p) {
    constexpr int Tiles = (N + 255) / 256;
    constexpr int Groups = N / 128;
    static_assert(!FP8Block || N % 128 == 0);
    const int tid = threadIdx.x;
    const int expert = blockIdx.x / Tiles;
    const int col = (blockIdx.x % Tiles) * 256 + tid * 8;
    const int group = col / 128;
    const int valid = Masked ? p.masked_m[expert] : p.rows_per_batch;
    const int aligned_rows = Masked ? (p.rows_per_batch + 3) / 4 * 4 : p.rows_per_batch;
    auto* scales = static_cast<float*>(p.scales);
    if constexpr (FP8Block && Masked) {
        if (tid % 16 == 0 && col < N)
            for (int64_t row = int64_t(valid) + blockIdx.y; row < aligned_rows; row += gridDim.y)
                scales[(int64_t(expert) * Groups + group) * aligned_rows + row] = 0.0f;
    }
    for (int local_row = blockIdx.y; local_row < valid; local_row += gridDim.y) {
        const int64_t row = int64_t(expert) * p.rows_per_batch + local_row;
        alignas(16) cutlass::bfloat16_t gate[8], up[8];
        activation_load_eight<N>(static_cast<const cutlass::bfloat16_t*>(p.input), row, col, gate, up);
        float values[8];
        float maximum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            values[i] = Situ ? situ_and_mul_value(float(gate[i]), float(up[i]), p.beta, p.linear_beta)
                             : silu_and_mul_value(float(gate[i]), float(up[i]), p.limit);
            if constexpr (FP8Block) maximum = fmaxf(maximum, fabsf(values[i]));
        }
        if constexpr (FP8Block) {
            #pragma unroll
            for (int delta = 8; delta > 0; delta /= 2)
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta, 16));
            const float amax = fmaxf(maximum, 1e-10f);
            if (tid % 16 == 0 && col < N) {
                const int64_t index = Masked ? (int64_t(expert) * Groups + group) * aligned_rows + local_row
                                            : row * Groups + group;
                scales[index] = amax * (1.0f / 448.0f);
            }
            alignas(8) uint16_t packed[4];
            activation_pack_fp8_eight(values, __fdividef(448.0f, amax), packed);
            if (col < N)
                *reinterpret_cast<uint2*>(static_cast<uint8_t*>(p.output) + row * N + col) =
                    *reinterpret_cast<const uint2*>(packed);
        } else {
            alignas(16) cutlass::bfloat16_t packed[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) packed[i] = cutlass::bfloat16_t(values[i]);
            auto* dst = static_cast<cutlass::bfloat16_t*>(p.output) + row * N + col;
            if constexpr (N % 8 == 0) {
                if (col < N) *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(packed);
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) if (col + i < N) dst[i] = packed[i];
            }
        }
    }
}
} // namespace deep_gemm

namespace deep_gemm {
// One CTA per row, eight consecutive columns per thread and work round.
template<int N, int Threads, bool Situ = false>
__device__ void activation_fp8_channel_impl(const cutlass::bfloat16_t* input,
                                  cutlass::float_e4m3_t* output, float* scales,
                                  float weight_inv_scale, float limit, int64_t row, float beta = 1.0f, float linear_beta = 0.0f) {
    constexpr int Rounds = (N + Threads * 8 - 1) / (Threads * 8);
    constexpr int Warps = Threads / 32;
    const int tid = threadIdx.x;
    float values[Rounds][8];
    float maximum = 0.0f;
    #pragma unroll
    for (int r = 0; r < Rounds; ++r) {
        const int col = (r * Threads + tid) * 8;
        alignas(16) cutlass::bfloat16_t gate[8], up[8];
        activation_load_eight<N>(input, row, col, gate, up);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float value = Situ ? situ_and_mul_value(float(gate[i]), float(up[i]), beta, linear_beta)
                                     : silu_and_mul_value(float(gate[i]), float(up[i]), limit);
            values[r][i] = value;
            maximum = fmaxf(maximum, fabsf(value));
        }
    }
    #pragma unroll
    for (int delta = 16; delta > 0; delta /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta));
    if constexpr (Warps > 1) {
        __shared__ float warp_max[Warps];
        if (tid % 32 == 0) warp_max[tid / 32] = maximum;
        __syncthreads();
        if (tid < 32) {
            maximum = tid < Warps ? warp_max[tid] : 0.0f;
            #pragma unroll
            for (int delta = 16; delta > 0; delta /= 2)
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta));
            if (tid == 0) warp_max[0] = maximum;
        }
        __syncthreads();
        maximum = warp_max[0];
    }
    const float amax = fmaxf(maximum, 1e-10f);
    const float scale = amax * (1.0f / 448.0f);
    const float inverse = __fdividef(448.0f, amax);
    if (tid == 0) scales[row] = scale * weight_inv_scale;
    #pragma unroll
    for (int r = 0; r < Rounds; ++r) {
        const int col = (r * Threads + tid) * 8;
        alignas(8) uint16_t packed[4];
        activation_pack_fp8_eight(values[r], inverse, packed);
        auto* dst = reinterpret_cast<uint8_t*>(output) + row * N + col;
        if constexpr (N % 8 == 0) {
            if (col < N) *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(packed);
        } else {
            const auto* bytes = reinterpret_cast<const uint8_t*>(packed);
            #pragma unroll
            for (int i = 0; i < 8; ++i) if (col + i < N) dst[i] = bytes[i];
        }
    }
}
} // namespace deep_gemm

namespace deep_gemm::activation_mx {
__device__ __forceinline__ void copy_16(void* dst, const void* src) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("ppu.cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(src));
}
__device__ __forceinline__ uint32_t pack_eight(const float* q) {
    uint8_t b[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        asm("ppu.cvt.rtte.satfinite.e2m1x2.f32 %0, %1, %2;\n"
            : "=c"(b[i]) : "f"(q[2*i+1]), "f"(q[2*i]));
    return uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24);
}

template<int N, bool Reordered, bool Masked, bool Clamp, bool VectorLoads, bool Situ, bool FP8>
__device__ void run(const cutlass::bfloat16_t* input, void* output, uint8_t* scales,
                   int rows_per_batch, const int32_t* masked_m, int row_slots,
                   float limit, float beta, float linear_beta) {
    constexpr int Groups = (N + 31) / 32;
    constexpr int Pairs = (Groups + 1) / 2;
    constexpr int HiddenBlocks = (N + 255) / 256;
    constexpr int Stages = 2;
    static_assert(FP8 || N % 2 == 0);
    static_assert(!FP8 || !Reordered);
    __shared__ alignas(16) __ppu_bfloat16 gate_smem[Stages][32][8];
    __shared__ alignas(16) __ppu_bfloat16 up_smem[Stages][32][8];
    const int expert = blockIdx.x / HiddenBlocks;
    const int hidden_block = blockIdx.x % HiddenBlocks;
    const int slot = blockIdx.y;
    const int tid = threadIdx.x;
    const int group = hidden_block * 8 + tid / 4;
    const int col = hidden_block * 256 + tid * 8;
    const int valid = Masked ? masked_m[expert] : rows_per_batch;
    const auto* in = reinterpret_cast<const __ppu_bfloat16*>(input) + int64_t(expert) * rows_per_batch * (2*N);
    auto* out = static_cast<uint8_t*>(output) + int64_t(expert) * rows_per_batch * (FP8 ? N : N/2);
    auto load = [&](int stage, int64_t t) {
        if (t < valid) {
            const auto* row = in + int64_t(t) * (2*N);
            if constexpr (VectorLoads) {
                copy_16(gate_smem[stage][tid], row + col);
                copy_16(up_smem[stage][tid], row + N + col);
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    gate_smem[stage][tid][i] = col+i < N ? row[col+i] : __float2bfloat16_rn(0);
                    up_smem[stage][tid][i] = col+i < N ? row[N+col+i] : __float2bfloat16_rn(0);
                }
            }
        }
        if constexpr (VectorLoads) asm volatile("ppu.cp.async.commit_group;\n" ::);
    };
    load(0, slot);
    int compute_stage = 0;
    for (int64_t t = slot; t < valid; t += row_slots) {
        load(1-compute_stage, t+row_slots);
        if constexpr (VectorLoads) asm volatile("ppu.cp.async.wait_group 1;\n" ::);
        __syncthreads();
        auto* gates = reinterpret_cast<__ppu_bfloat162*>(gate_smem[compute_stage][tid]);
        auto* ups = reinterpret_cast<__ppu_bfloat162*>(up_smem[compute_stage][tid]);
        float fp32[8], maximum = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            auto gate = gates[i], up = ups[i];
            if constexpr (Clamp && !Situ) {
                const auto hi = __float2bfloat162_rn(limit);
                const auto lo = __float2bfloat162_rn(-limit);
                gate = __hmin2(gate,hi);
                up = __hmin2(__hmax2(up,lo),hi);
            }
            const float g0 = __bfloat162float(__low2bfloat16(gate));
            const float g1 = __bfloat162float(__high2bfloat16(gate));
            const float u0 = __bfloat162float(__low2bfloat16(up));
            const float u1 = __bfloat162float(__high2bfloat16(up));
            if constexpr (Situ) {
                fp32[2*i] = situ_and_mul_value(g0,u0,beta,linear_beta);
                fp32[2*i+1] = situ_and_mul_value(g1,u1,beta,linear_beta);
            } else {
                fp32[2*i] = (g0 * __ppu_sgmdf(g0)) * u0;
                fp32[2*i+1] = (g1 * __ppu_sgmdf(g1)) * u1;
            }
            maximum = fmaxf(maximum,fmaxf(fabsf(fp32[2*i]),fabsf(fp32[2*i+1])));
        }
        maximum = fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,2));
        maximum = fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,1));
        const float raw_scale = fmaxf(maximum,FP8 ? 1e-30f : 1e-10f) * (FP8 ? 1.0f/448.0f : 1.0f/6.0f);
        const uint32_t bits = __float_as_uint(raw_scale);
        int exponent;
        float inverse;
        if constexpr (FP8) {
            exponent = min(254,max(1,int(bits>>23) + int((bits&0x7fffffu)!=0)));
            inverse = __uint_as_float(uint32_t(254-exponent)<<23);
        } else {
            const uint32_t rounded_scale = (bits + 0x007fffffu) & 0x7f800000u;
            exponent = rounded_scale >> 23;
            inverse = __uint_as_float(0x7f000000u - rounded_scale);
        }
        float q[8];
        #pragma unroll
        for (int i=0;i<8;++i) q[i]=fp32[i]*inverse;
        if constexpr (FP8) {
            uint16_t packed[4];
            #pragma unroll
            for (int i=0;i<4;++i)
                asm("ppu.cvt.rtte.satfinite.e4m3x2.f32 %0, %1, %2;\n"
                    : "=h"(packed[i]) : "f"(q[2*i+1]), "f"(q[2*i]));
            if constexpr (N % 8 == 0) {
                const uint64_t word = uint64_t(packed[0]) | (uint64_t(packed[1])<<16)
                    | (uint64_t(packed[2])<<32) | (uint64_t(packed[3])<<48);
                if (col<N) *reinterpret_cast<uint64_t*>(out+int64_t(t)*N+col)=word;
            } else if (col<N) {
                #pragma unroll
                for (int i=0;i<8;++i)
                    if (col+i<N) out[int64_t(t)*N+col+i]=uint8_t(packed[i/2]>>(8*(i%2)));
            }
        } else {
            const uint32_t packed = pack_eight(q);
            if constexpr (N % 8 == 0) {
                if (col < N) *reinterpret_cast<uint32_t*>(out + int64_t(t)*(N/2)+col/2)=packed;
            } else if (col < N) {
                const int bytes = min(8,N-col)/2;
                #pragma unroll
                for (int b=0;b<4;++b)
                    if (b<bytes) out[int64_t(t)*(N/2)+col/2+b]=uint8_t(packed>>(8*b));
            }
        }
        if constexpr (Reordered) {
            // Two neighboring groups form one PPU1.5 uint16 M-major word.
            const uint32_t byte = group < Groups ? uint32_t(exponent) : 0;
            const int base = tid/8*8;
            const uint32_t low = __shfl_sync(0xffffffffu,byte,base);
            const uint32_t high = __shfl_sync(0xffffffffu,byte,base+4);
            const int pair = hidden_block*4 + tid/8;
            if (tid%8==0 && pair<Pairs)
                reinterpret_cast<uint16_t*>(scales)[int64_t(expert)*rows_per_batch*Pairs + int64_t(pair)*rows_per_batch+t] = uint16_t(low|(high<<8));
        } else if (tid%4==0 && group<Groups) {
            scales[(int64_t(expert)*rows_per_batch+t)*Groups+group]=uint8_t(exponent);
        }
        compute_stage=1-compute_stage;
    }
    if constexpr (VectorLoads) asm volatile("ppu.cp.async.wait_group 0;\n" ::);
}
} // namespace deep_gemm::activation_mx

namespace deep_gemm {
template<int N, ActivationOutputMode Mode, bool Situ>
__device__ __forceinline__ void activation_quant_row(const ActivationParams& p, int64_t row) {
    const auto* input = static_cast<const cutlass::bfloat16_t*>(p.input);
    if constexpr (Mode == ActivationOutputMode::FP8_CHANNEL)
        activation_fp8_channel_impl<N, activation_fp8_channel_threads(N), Situ>(input, static_cast<cutlass::float_e4m3_t*>(p.output),
            static_cast<float*>(p.scales), p.weight_inv_scale, p.limit, row, p.beta, p.linear_beta);

}

template<int N, ActivationOutputMode Mode, bool Situ, bool Masked, bool Reordered, bool Clamp, bool VectorLoads>
__device__ void activation_quant(const ActivationParams& p) {
    if constexpr (Mode == ActivationOutputMode::BF16 || Mode == ActivationOutputMode::FP8_BLOCK) {
        activation_tiled<N, Mode == ActivationOutputMode::FP8_BLOCK, Situ, Masked>(p);
    } else if constexpr (Mode == ActivationOutputMode::MXFP4 || Mode == ActivationOutputMode::MXFP8) {
        activation_mx::run<N, Reordered, Masked, Clamp, VectorLoads, Situ, Mode == ActivationOutputMode::MXFP8>(
            static_cast<const cutlass::bfloat16_t*>(p.input), p.output, static_cast<uint8_t*>(p.scales),
            p.rows_per_batch, p.masked_m, p.row_slots, p.limit, p.beta, p.linear_beta);
    } else {
        const int work = blockIdx.x;
        if constexpr (Masked) {
            const int expert = work / p.row_slots;
            const int slot = work % p.row_slots;
            const int valid = p.masked_m[expert];
            for (int64_t local_row = slot; local_row < valid; local_row += p.row_slots) {
                const int64_t row = int64_t(expert) * p.rows_per_batch + local_row;
                activation_quant_row<N, Mode, Situ>(p, row);
                if constexpr (Mode == ActivationOutputMode::FP8_CHANNEL) __syncthreads();
            }
        } else {
            activation_quant_row<N, Mode, Situ>(p, work);
        }
    }
}
} // namespace deep_gemm
