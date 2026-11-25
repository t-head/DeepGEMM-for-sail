#pragma once
#include <hggc_bf16.h>
#include <hggc_fp8.h>
#include <deep_gemm/common/math.cuh>

namespace deep_gemm::epilogue::quant {

struct FP8Scale { float scale, inverse; };

template <bool FastMath>
CUTLASS_DEVICE FP8Scale fp8_row_scale(float amax) {
    const float scale = math::fp32_divide<FastMath>(fmaxf(amax, 1e-10f), 448.0f);
    return {scale, math::fp32_divide<FastMath>(1.0f, scale)};
}

template <bool FastMath>
CUTLASS_DEVICE FP8Scale fp8_block_scale(float amax) {
    amax = fmaxf(amax, 1e-10f);
    return {amax * (1.0f / 448.0f), math::fp32_divide<FastMath>(448.0f, amax)};
}

template <int Lanes = 32>
CUTLASS_DEVICE float warp_amax(float value) {
    static_assert(Lanes > 0 && Lanes <= 32 && (Lanes & (Lanes - 1)) == 0);
    #pragma unroll
    for (int offset = Lanes / 2; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset, Lanes));
    return value;
}

CUTLASS_DEVICE __ppu_bfloat16 bf16(float value) { return __float2bfloat16(value); }

CUTLASS_DEVICE uint32_t bf16_pair(float lo, float hi) {
    auto pair = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&pair);
}

CUTLASS_DEVICE uint16_t fp8_pair(float lo, float hi) {
    return __hg_cvt_float2_to_fp8x2(make_float2(lo, hi), __HG_SATFINITE, __HG_E4M3);
}

// Preserve the existing fused FP4 epilogue's bit-based E8M0 rounding.
CUTLASS_DEVICE uint32_t mxfp4_scale_bits(uint32_t amax_bits) {
    return (((amax_bits > 0x2EDBE6FFu ? amax_bits : 0x2EDBE6FFu) - 0x1400000u)
            + 0x007FFFFFu) & 0x7F800000u;
}

CUTLASS_DEVICE float mxfp4_inverse(uint32_t scale_bits) {
    return __uint_as_float((254u << 23) - scale_bits);
}

CUTLASS_DEVICE uint8_t fp4_pair(float hi, float lo) {
    uint8_t packed;
    asm("ppu.cvt.rtte.satfinite.e2m1x2.f32 %0, %1, %2;\n" : "=c"(packed) : "f"(hi), "f"(lo));
    return packed;
}

} // namespace deep_gemm::epilogue::quant
