#pragma once
#include <hggc_runtime.h>
#include <deep_gemm/common/nvfp4_decode.cuh>

namespace deep_gemm::nvfp4::detail {

// Every thread expands sixteen K16 records. Both inputs and the FP8 output
// are contiguous along K; vector stores avoid a transpose after decoding.
template <int Items = 16, int Threads = 128>
__device__ __forceinline__ void decode_fp4_fp8(
        const uint2* __restrict__ packed, const uint8_t* __restrict__ scales,
        uint4* __restrict__ output, uint32_t count) {
    constexpr int table_size = 127;
    __shared__ uint64_t table[table_size];
    const uint32_t t = threadIdx.x;
    if (t < table_size)
        table[t] = kNvfp4E4m3ScaleLut[t];
    __syncthreads();

    const uint32_t base = blockIdx.x * Items * Threads;
    #pragma unroll
    for (int j = 0; j < Items; ++j) {
        const uint32_t i = base + j * Threads + t;
        if (i < count) {
            const uint2 bits = packed[i];
            const uint8_t scale = scales[i];
            const uint64_t lut = table[scale & 0x7fu];
            const uint32_t lo = uint32_t(lut), hi = uint32_t(lut >> 32);
            output[i] = make_uint4(
                __byte_perm(lo, hi, bits.x & 0x7777u) | (bits.x & 0x80808080u),
                __byte_perm(lo, hi, (bits.x >> 16) & 0x7777u) | ((bits.x & 0x08080808u) << 4),
                __byte_perm(lo, hi, bits.y & 0x7777u) | (bits.y & 0x80808080u),
                __byte_perm(lo, hi, (bits.y >> 16) & 0x7777u) | ((bits.y & 0x08080808u) << 4));
        }
    }
}

} // namespace deep_gemm::nvfp4::detail
