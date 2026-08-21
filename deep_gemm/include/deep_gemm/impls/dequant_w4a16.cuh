#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/bfloat16.h"
#include "cutlass/float4.h"

namespace deep_gemm {

template <typename ElementA, typename ElementB, int InterleavedElems>
struct DequantW4A16;

template <>
struct DequantW4A16<cutlass::bfloat16_t, cutlass::int4b_t, 8>
{
    CUTLASS_DEVICE
    static void convert(const int* src, cutlass::bfloat16_t* dst)
    {
        uint32_t* h = reinterpret_cast<uint32_t*>(dst);
        uint32_t i4s = *reinterpret_cast<const uint32_t*>(src);

        static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
        static constexpr uint32_t MASK = 0x000f000f;
        static constexpr uint32_t I4s_TO_BF16s_MAGIC_NUM = 0x43004300;

        asm volatile("ppu.lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[0])
                     : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
        CUTLASS_PRAGMA_UNROLL
        for (int ii = 1; ii < 4; ++ii)
        {
            i4s >>= 4;
            asm volatile("ppu.lop3.b32 %0, %1, %2, %3, %4;\n"
                         : "=r"(h[ii])
                         : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
        }

        static constexpr uint32_t BF16_BIAS = 0xC308C308;
        static constexpr uint32_t BF16_ONE = 0x3F803F80;

        CUTLASS_PRAGMA_UNROLL
        for (int ii = 0; ii < 4; ++ii)
        {
            asm volatile("ppu.fma.rtte.bf16x2 %0, %1, %2, %3;\n"
                         : "=r"(h[ii])
                         : "r"(h[ii]), "r"(BF16_ONE), "r"(BF16_BIAS));
        }
    }
};

template <>
struct DequantW4A16<cutlass::bfloat16_t, cutlass::float4_t, 8>
{
    CUTLASS_DEVICE
    static void convert(const int* src, cutlass::bfloat16_t* dst)
    {
        uint32_t* h = reinterpret_cast<uint32_t*>(dst);
        uint32_t const source_fp4s = *reinterpret_cast<const uint32_t*>(src);

        // FP4 -> BF16 bit-field remap using HIGH nibble extraction with << 4 iteration.
        // The interleaved layout is:
        //   bits [ 3: 0] = e0   bits [ 7: 4] = e2   bits [11: 8] = e4   bits [15:12] = e6
        //   bits [19:16] = e1   bits [23:20] = e3   bits [27:24] = e5   bits [31:28] = e7

        static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
        static constexpr uint32_t SIGN_MASK = 0x80008000;
        static constexpr uint32_t DATA_MASK = 0x70007000;
        static constexpr uint32_t DATA_SHIFT = 6;

        // Parallel shifts: fp4s[ii] = source_fp4s << (4 * ii), all independent.
        uint32_t fp4s[4];
        fp4s[0] = source_fp4s;
        fp4s[1] = source_fp4s << 4;   // (e4, e5) -> h[2]
        fp4s[2] = source_fp4s << 8;   // (e2, e3) -> h[1]
        fp4s[3] = source_fp4s << 12;  // (e0, e1) -> h[0]
        CUTLASS_PRAGMA_UNROLL
        for (int ii = 0; ii < 4; ++ii)
        {
            uint32_t tmp;
            asm volatile("ppu.and.b32 %0, %1, %2;\n"
                         : "=r"(tmp) : "r"(fp4s[ii]), "n"(DATA_MASK));
            asm volatile("ppu.shr.b32 %0, %1, %2;\n"
                         : "=r"(tmp) : "r"(tmp), "n"(DATA_SHIFT));
            asm volatile("ppu.lop3.b32 %0, %1, %2, %3, %4;\n"
                         : "=r"(h[3 - ii]) : "r"(fp4s[ii]), "n"(SIGN_MASK), "r"(tmp), "n"(immLut));
        }

        // Exponent-bias compensation: x 2^126.
        static constexpr uint32_t EXP_OFFSET_126 = 0x7E807E80;
        CUTLASS_PRAGMA_UNROLL
        for (int ii = 0; ii < 4; ++ii)
        {
            asm volatile("ppu.mul.rtte.bf16x2 %0, %1, %2;\n"
                         : "=r"(h[ii]) : "r"(h[ii]), "r"(EXP_OFFSET_126));
        }
    }
};

template <int InterleavedElems>
struct DequantE8M0
{
    static_assert(InterleavedElems > 0 && InterleavedElems % 4 == 0,
                  "DequantE8M0 requires InterleavedElems to be a multiple of 4");

    CUTLASS_DEVICE
    static void convert(const uint8_t* src, cutlass::bfloat16_t* dst)
    {
        uint32_t* h = reinterpret_cast<uint32_t*>(dst);
        CUTLASS_PRAGMA_UNROLL
        for (int g = 0; g < InterleavedElems / 4; ++g)
        {
            // 4 e8m0 bytes (b0..b3) -> 2 bf16x2:
            //   low  bf16x2 = {b0<<7, b2<<7}  -> h[2g]
            //   high bf16x2 = {b1<<7, b3<<7}  -> h[2g+1]
            uint32_t q = *reinterpret_cast<const uint32_t*>(src + 4 * g);
            h[2 * g]     = (q << 7) & 0x7F807F80u;
            h[2 * g + 1] = (q & 0xFF00FF00u) >> 1;
        }
    }
};

} // namespace deep_gemm
