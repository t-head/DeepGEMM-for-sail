#pragma once

#include <cstdint>
#include <type_traits>
#include <hggc_bf16.h>
#include <cutlass/numeric_conversion.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/tix/ld_st.cuh>
#include <deep_gemm/tix/utils.cuh>

namespace deep_gemm {

// Pack two FP32 values into one FP4 byte.
__device__ __forceinline__ uint8_t cvt_f32x2_to_fp4x2(float hi, float lo) {
    uint8_t b;
    asm volatile("ppu.cvt.rtte.satfinite.e2m1x2.f32 %0, %1, %2;\n"
                 : "=c"(b)
                 : "f"(hi), "f"(lo));
    return b;
}

__device__ __forceinline__ __ppu_bfloat16 fast_pow2_bf16(int x) {
    uint16_t bits = static_cast<uint16_t>((x + 127) << 7);
    return *reinterpret_cast<__ppu_bfloat16*>(&bits);
}

// Return the BF16 quantization multiplier and UE8M0 dequantization scale.
__device__ __forceinline__ void calculate_mxfp4_scales_bf16(
    __ppu_bfloat16 amax, __ppu_bfloat16& scale, uint8_t& scale_inv) {
    constexpr float kAmaxInvMXFP4 = 0.16666667f;  // 1/6.0, max representable E2M1 value
    __ppu_bfloat16 scale_inv_bf16 = __hmul(amax, __float2bfloat16(kAmaxInvMXFP4));
    uint16_t scale_inv_bits = *reinterpret_cast<uint16_t*>(&scale_inv_bf16);
    // Round the magnitude up to a UE8M0 exponent.
    uint16_t biased_exp_scale_inv = static_cast<uint16_t>(
        ((scale_inv_bits & 0x7fffu) + 0x7fu) >> 7);
    int exp_scale_inv = biased_exp_scale_inv - 127;
    scale = fast_pow2_bf16(-exp_scale_inv);
    scale_inv = static_cast<uint8_t>(biased_exp_scale_inv);
}

// Per-token INT8 scales follow the existing A8W8 GEMM contract.
template <int Hidden, bool kInputAligned>
__device__ __forceinline__ void quantize_int8_token(
    const __ppu_bfloat16* input, uint8_t* data, float* scales) {
    // Legacy DeepEP QUANT_SIZE_CHANNEL: vector loads, amax >= 1e-6,
    // multiplier 127/amax, FP32 dequant scale amax*(1/127), round-to-nearest.
    static_assert(Hidden % 8 == 0);
    __shared__ float warp_max[8];
    constexpr float epsilon = 1.0e-6f;
    float amax = epsilon;
    for (int i = threadIdx.x; i < Hidden / 8; i += 256) {
        int4 raw = tix::load_bf16x8<kInputAligned>(input + i * 8);
        const auto* values = reinterpret_cast<const __ppu_bfloat16*>(&raw);
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            amax = fmaxf(amax, fabsf(__bfloat162float(values[j])));
    }
    amax = tix::block_reduce_max<256>(amax, warp_max, epsilon);
    const float multiplier = 127.0f / amax;
    if (threadIdx.x == 0)
        scales[0] = amax * (1.0f / 127.0f);
    for (int i = threadIdx.x; i < Hidden / 8; i += 256) {
        int4 raw = tix::load_bf16x8<kInputAligned>(input + i * 8);
        const auto* values = reinterpret_cast<const __ppu_bfloat16*>(&raw);
        int2 packed;
        auto* quantized = reinterpret_cast<int8_t*>(&packed);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float value = __bfloat162float(values[j]) * multiplier;
            quantized[j] = cutlass::NumericConverter<int8_t, float>{}(value);
        }
        reinterpret_cast<int2*>(data)[i] = packed;
    }
}

// Quantize one FP4 row and pack its UE8M0 scales.
template <int Hidden, uint32_t Threads, bool kInputAligned>
__device__ __forceinline__ void quantize_fp4_token(
    const __ppu_bfloat16* row, int2* fp4, uint32_t* scales) {
    const uint32_t tid = threadIdx.x, lane = tid % 32;
    constexpr int kChunks = Hidden / 16;
    constexpr int kBlockThreads = Threads;
    constexpr int kQuantThreads = kBlockThreads - 32;
    constexpr int kIterations = (kChunks + kQuantThreads - 1) / kQuantThreads;
    int4 raw[kIterations][2];
    #pragma unroll
    for (int it = 0; it < kIterations; ++it) {
        const int chunk = it * kQuantThreads + tid;
        if (tid < kQuantThreads && chunk < kChunks) {
            raw[it][0] = tix::load_bf16x8<kInputAligned>(row + chunk * 16);
            raw[it][1] = tix::load_bf16x8<kInputAligned>(row + chunk * 16 + 8);
        }
    }
    #pragma unroll
    for (int it = 0; it < kIterations; ++it) {
        const int chunk = it * kQuantThreads + tid;
        if (tid >= kQuantThreads || chunk >= kChunks) continue;
        __ppu_bfloat162 values[8];
        __ppu_bfloat162 max2 = __float2bfloat162_rn(0.0f);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            values[j] = reinterpret_cast<const __ppu_bfloat162*>(&raw[it][j / 4])[j % 4];
            max2 = __hmax2(max2, __habs2(values[j]));
        }
        __ppu_bfloat16 amax = __hmax(max2.x, max2.y);
        amax = __hmax(amax, __shfl_xor_sync(0xffffffffu, amax, 1, 2));
        __ppu_bfloat16 scale;
        uint8_t exponent;
        calculate_mxfp4_scales_bf16(amax, scale, exponent);
        // Four pairs of lanes produce the four scale bytes in one aligned word.
        uint32_t word = uint32_t(exponent) << (8 * ((lane / 2) % 4));
        word |= __shfl_xor_sync(0xffffffffu, word, 2);
        word |= __shfl_xor_sync(0xffffffffu, word, 4);
        if (lane % 8 == 0) __stbl(scales + chunk / 8, word);
        const auto multiplier = __halves2bfloat162(scale, scale);
        int2 packed;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float2 value = __bfloat1622float2(__hmul2(values[j], multiplier));
            reinterpret_cast<uint8_t*>(&packed)[j] = cvt_f32x2_to_fp4x2(value.y, value.x);
        }
        __stbl(fp4 + chunk, packed);
    }
}

// Scale storage follows each precision's quantization format.
template <bool Int8>
struct MegaMoEScaleFormat {
    using Scale = std::conditional_t<Int8, float, uint16_t>;
    using Vector = std::conditional_t<Int8, float, int4>;
    static constexpr uint32_t kElementsPerVector = sizeof(Vector) / sizeof(Scale);

    __host__ __device__ static constexpr uint32_t elements(uint32_t hidden) {
        return Int8 ? 1 : hidden / 64;
    }
    __host__ __device__ static constexpr uint32_t vectors(uint32_t hidden) {
        return elements(hidden) / kElementsPerVector;
    }
    template <class Buffers>
    __device__ static const Vector* source_scales(
        const Buffers& buffers, void* base, uint32_t source, uint32_t capacity) {
        return buffers.staging_scales[0].template data<Vector>(base)
               + (Int8 ? uint64_t(source) * capacity : 0u);
    }
    // FP4 writes a packed vector across columns; INT8 copies one row scale.
    __device__ static __forceinline__ void copy_vector(
        const Scale* src, Scale* dst, uint32_t vector = 0, uint32_t row_stride = 1) {
        if constexpr (Int8) {
            __stbl(dst, __ldbl(src));
        } else {
            const int4 value = __ldbl(reinterpret_cast<const int4*>(src) + vector);
            const auto* elems = reinterpret_cast<const uint16_t*>(&value);
            #pragma unroll
            for (uint32_t e = 0; e < kElementsPerVector; ++e)
                __stbl(dst + static_cast<uint64_t>(vector * kElementsPerVector + e) * row_stride, elems[e]);
        }
    }
};

// Quantize locally and stage routes; GEMM1 publishes them to peers.
template <bool Int8, int Hidden, int TopK, uint32_t SourceCapacity,
          uint32_t NumRanks, uint32_t NumExperts, int Intermediate,
          int64_t StagingScales, int64_t StagingCounts, int64_t StagingTokens,
          typename TopkIdx, bool kInputAligned, uint32_t Threads = 256>
__global__ void mega_moe_pre_dispatch(
    const __ppu_bfloat16* __restrict__ input,    // [num_tokens, Hidden]
    const TopkIdx* __restrict__ topk_ids,       // [num_tokens, topk]
    const float* __restrict__ topk_weights,
    uint32_t num_tokens, layout::SymBuffer<> sym_buffer) {

    static_assert(Hidden >= 1024 && Hidden % 512 == 0);
    static_assert(SourceCapacity > 0 && TopK >= 1 && TopK <= 32);
    static_assert(is_supported_megamoe_rank_count(NumRanks));
    static_assert(NumExperts > 0 && NumExperts % NumRanks == 0);
    static_assert(Threads >= 256 && Threads <= 512 && Threads % 32 == 0);
    constexpr layout::MegaMoEBufferLayout<Int8> buffers(NumRanks, NumExperts, SourceCapacity, TopK,
        Hidden, Intermediate, {StagingScales, StagingCounts, StagingTokens});
    auto* base = sym_buffer.get_base_ptr();
    auto* ids = buffers.topk_idx.template data<int64_t>(base);
    auto* weights = buffers.topk_weights.template data<float>(base);
    const uint32_t token = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid / 32;
    const uint32_t lane = tid % 32;
    if (token == 0 && tid == 0) {
        auto* ready = buffers.preprocess_ready.template data<uint32_t>(base);
        const uint32_t prior = __ldbl(ready) & ~kPreprocessMetadataReadyBit;
        __stbl(ready, prior >= kFusedRoundGenerationLimit ? 2u : prior + 1u);
    }
    if (token >= num_tokens) {
        const uint32_t stride = (gridDim.x - num_tokens) * Threads;
        for (uint32_t slot = num_tokens * TopK + (token - num_tokens) * Threads + tid;
             slot < SourceCapacity * TopK; slot += stride) {
            ids[slot] = -1;
            weights[slot] = 0.0f;
        }
        return;
    }

    if ((Int8 ? tid < TopK : warp == (Threads - 32) / 32 && lane < TopK)) {
        const uint32_t slot = token * TopK + lane;
        const int64_t expert = static_cast<int64_t>(topk_ids[slot]);
        const bool valid = expert >= 0 && uint64_t(expert) < NumExperts;
        ids[slot] = valid ? expert : -1;
        weights[slot] = valid ? topk_weights[slot] : 0.0f;
    }
    const auto* row = input + static_cast<int64_t>(token) * Hidden;
    if constexpr (Int8) {
        static_assert(Threads == 256);
        auto* data = buffers.dispatch_data.data(base) + uint64_t(token) * Hidden;
        auto* scales = buffers.staging_scales[0].template data<float>(base)
                       + uint64_t(sym_buffer.rank_idx) * SourceCapacity + token;
        quantize_int8_token<Hidden, kInputAligned>(row, data, scales);
    } else {
        auto* data = reinterpret_cast<int2*>(buffers.dispatch_data.data(base)) + uint64_t(token) * (Hidden / 16);
        auto* scales = buffers.staging_scales[0].template data<uint32_t>(base) + uint64_t(token) * (Hidden / 128);
        quantize_fp4_token<Hidden, Threads, kInputAligned>(row, data, scales);
    }
}

}  // namespace deep_gemm
