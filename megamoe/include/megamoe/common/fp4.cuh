#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace megamoe::fp4 {

// FP4 (E2M1) scale-factor computation and UE8M0 packing for the W4A4 requant path.

__device__ __forceinline__ void get_e2m1_sf_and_sf_inv(const float2& amax, float2& sf, float2& sf_inv) {
    const uint32_t amax_u0 = __float_as_uint(amax.x);
    const uint32_t amax_u1 = __float_as_uint(amax.y);
    const uint32_t clamped0 = max(amax_u0, 0x2EDBE6FFu);
    const uint32_t clamped1 = max(amax_u1, 0x2EDBE6FFu);
    const uint32_t scale_bits0 = clamped0 - 0x1400000u;
    const uint32_t scale_bits1 = clamped1 - 0x1400000u;
    const uint32_t scale_exp0 = (scale_bits0 + 0x007FFFFFu) & 0x7F800000u;
    const uint32_t scale_exp1 = (scale_bits1 + 0x007FFFFFu) & 0x7F800000u;
    sf.x = __uint_as_float(scale_exp0);
    sf.y = __uint_as_float(scale_exp1);
    sf_inv.x = __uint_as_float((254u << 23) - scale_exp0);
    sf_inv.y = __uint_as_float((254u << 23) - scale_exp1);
}

__device__ __forceinline__ uint32_t sf_to_ue8m0(float sf) {
    return (__float_as_uint(sf) >> 23) & 0xffu;
}

}  // namespace megamoe::fp4