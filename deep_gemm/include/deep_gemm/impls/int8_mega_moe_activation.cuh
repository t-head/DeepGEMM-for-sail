#pragma once

#include <cutlass/epilogue/thread/activation.h>
#include <deep_gemm/epilogue/gated_activation.cuh>
#include <deep_gemm/impls/int8_mega_moe_pre_dispatch.cuh>

namespace deep_gemm {

template <bool NormalQuantization, bool ApplyClamp, bool FastSigmoid = false>
__device__ __forceinline__ float int8_swiglu_value(float gate, float up, float limit) {
    using Activation = GatedActivationOp<GatedActivation::SwiGLU, ApplyClamp>;
    // Preserve Legacy's BF16 GEMM output boundary even inside the epilogue.
    gate = __bfloat162float(__float2bfloat16(gate));
    up = __bfloat162float(__float2bfloat16(up));
    gate = Activation::prepare_gate(gate, limit);
    float activated;
    if constexpr (FastSigmoid) {
        // Reuse FP4's Sigmoid primitive, which maps to native PPU sigmoid.
        // Keep exact division available to the independent unfused reference.
        activated = Activation::gate_times_sigmoid(
            gate, Activation::sigmoid(gate, limit), limit);
    } else {
        activated = gate / (1.0f + expf(-gate));
    }
    if constexpr (!NormalQuantization)
        activated = __bfloat162float(__float2bfloat16(activated));
    return __bfloat162float(__float2bfloat16(Activation::multiply_up(activated, up, limit)));
}

// The additional public activations keep FP4's formula and the BF16 GEMM/product boundaries.
template <GatedActivation Kind, bool ApplyClamp, bool HasLinearBeta, bool FastMath = true>
__device__ __forceinline__ float int8_gated_value(
        float gate, float up, float limit, float beta, float linear_beta, float alpha) {
    using Activation = GatedActivationOp<Kind, ApplyClamp, HasLinearBeta, FastMath>;
    gate = __bfloat162float(__float2bfloat16(gate));
    up = __bfloat162float(__float2bfloat16(up));
    float value;
    if constexpr (Kind == GatedActivation::SiTU) {
        value = Activation::apply(gate, up, {beta, linear_beta});
    } else {
        const typename Activation::Params params = [&] {
            if constexpr (Kind == GatedActivation::SwiGLUOAI)
                return typename Activation::Params{limit, alpha};
            else return limit;
        }();
        gate = Activation::prepare_gate(gate, params);
        const float activated = Activation::gate_times_sigmoid(
            gate, Activation::sigmoid(gate, params), params);
        value = Activation::multiply_up(activated, up, params);
    }
    return __bfloat162float(__float2bfloat16(value));
}

// Each warp owns a complete activated row, so independent rows do not wait on
// CTA-wide reductions. Keep the existing scale and rounding expressions.
template <int Intermediate, int Threads, bool NormalQuantization>
__device__ __forceinline__ void int8_quant_activated_warp(
        const __ppu_bfloat16* input, int8_t* output, float* scales,
        const uint32_t* rows) {
    constexpr int Vector = 8;
    constexpr int Warps = Threads / 32;
    static_assert(Threads % 32 == 0 && Intermediate % (32 * Vector) == 0);
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t count = *rows;
    for (uint32_t row = blockIdx.x * Warps + warp; row < count; row += gridDim.x * Warps) {
        float values[Intermediate / 32];
        float amax = 1.0e-10f;
        #pragma unroll
        for (int i = 0; i < Intermediate / (32 * Vector); ++i) {
            const uint64_t offset = uint64_t(row) * Intermediate + (i * 32 + lane) * Vector;
            const int4 packed = *reinterpret_cast<const int4*>(input + offset);
            const auto* source = reinterpret_cast<const __ppu_bfloat16*>(&packed);
            #pragma unroll
            for (int j = 0; j < Vector; ++j) {
                values[i * Vector + j] = __bfloat162float(source[j]);
                amax = fmaxf(amax, fabsf(values[i * Vector + j]));
            }
        }
        #pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
        const float scale = NormalQuantization ? amax * (1.0f / 127.0f) : amax / 127.0f;
        if (lane == 0) scales[row] = scale;
        #pragma unroll
        for (int i = 0; i < Intermediate / (32 * Vector); ++i) {
            uint2 packed{};
            auto* target = reinterpret_cast<int8_t*>(&packed);
            #pragma unroll
            for (int j = 0; j < Vector; ++j) {
                const float value = values[i * Vector + j];
                const float quantized = NormalQuantization
                    ? roundf(value * (127.0f * __ppu_rcpf(amax))) : value / scale;
                target[j] = static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, quantized)));
            }
            const uint64_t offset = uint64_t(row) * Intermediate + (i * 32 + lane) * Vector;
            *reinterpret_cast<uint2*>(output + offset) = packed;
        }
    }
}

// The original masked INT8 path rounds SiLU and its product to BF16, then
// truncates the normalized value to INT8. Input dispatch has a different
// contract (DeepEP round-to-nearest); keep that distinction explicit.
template <int Intermediate, int Threads = 256, bool ApplyClamp = false,
          bool NormalQuantization = false, bool InputIsActivated = false,
          GatedActivation Kind = GatedActivation::SwiGLU, bool HasLinearBeta = true>
__device__ __forceinline__ void int8_silu_mul_quant_impl(
        const __ppu_bfloat16* input, int8_t* output, float* scales,
        const uint32_t* rows, float limit, float beta = 0, float linear_beta = 0, float alpha = 1.702f) {
    static_assert(Intermediate % Threads == 0);
    __shared__ float warp_max[Threads / 32];
    for (uint32_t row = blockIdx.x; row < *rows; row += gridDim.x) {
        float values[Intermediate / Threads];
        float amax = 1.0e-10f;
        #pragma unroll
        for (int i = 0; i < Intermediate / Threads; ++i) {
            const int column = i * Threads + threadIdx.x;
            const auto* source = input + uint64_t(row) * Intermediate * (InputIsActivated ? 1 : 2);
            if constexpr (InputIsActivated)
                values[i] = __bfloat162float(source[column]);
            else if constexpr (Kind == GatedActivation::SwiGLU)
                values[i] = int8_swiglu_value<NormalQuantization, ApplyClamp>(
                    __bfloat162float(source[column]), __bfloat162float(source[column + Intermediate]), limit);
            else
                values[i] = int8_gated_value<Kind, ApplyClamp, HasLinearBeta>(
                    __bfloat162float(source[column]), __bfloat162float(source[column + Intermediate]),
                    limit, beta, linear_beta, alpha);
            amax = fmaxf(amax, fabsf(values[i]));
        }
        amax = int8_row_amax<Threads>(amax, warp_max, 1.0e-10f);
        const float scale = NormalQuantization ? amax * (1.0f / 127.0f) : amax / 127.0f;
        if (threadIdx.x == 0) scales[row] = scale;
        #pragma unroll
        for (int i = 0; i < Intermediate / Threads; ++i) {
            const int column = i * Threads + threadIdx.x;
            const float quantized = NormalQuantization
                ? roundf(values[i] * (127.0f * __ppu_rcpf(amax))) : values[i] / scale;
            output[uint64_t(row) * Intermediate + column] = static_cast<int8_t>(
                fminf(127.0f, fmaxf(-128.0f, quantized)));
        }
        // A later row may reuse warp_max only after all warps consumed it.
        __syncthreads();
    }
}

}  // namespace deep_gemm
