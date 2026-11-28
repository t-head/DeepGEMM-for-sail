#pragma once

#include <deep_gemm/common/types.cuh>
#include <deep_gemm/epilogue/activation_quant_ops.cuh>
#include <hggc_bf16.h>
#include <hggc_fp8.h>

namespace deep_gemm::epilogue::operators {

// Values are rounded to BF16 by the store epilogue before clamp and quantization.
template <bool kClamp = false, bool kCount = false>
struct CastToFP8WithClamp {
    static constexpr uint32_t kSFGranN = 32;
    const EpilogueOperatorArgs& args;
    uint32_t num_clamped = 0;

    CUTLASS_DEVICE explicit CastToFP8WithClamp(const EpilogueOperatorArgs& args): args(args) {}

    CUTLASS_DEVICE uint32_t prepare_fp8(const __ppu_bfloat162& lo, const __ppu_bfloat162& hi,
                                       const bool valid) {
        if constexpr (kClamp and kCount) {
            const auto bound = __float2bfloat162_rn(args.clamp_value);
            const auto packed_count = (__hgtu2_mask(__habs2(lo), bound) & 0x00010001u) +
                                      (__hgtu2_mask(__habs2(hi), bound) & 0x00010001u);
            if (valid)
                num_clamped += (packed_count & 0xffffu) + (packed_count >> 16);
        }
        const auto maximum = __hmax2(__habs2(lo), __habs2(hi));
        uint32_t amax = max(uint32_t(__bfloat16_as_ushort(maximum.x)),
                            uint32_t(__bfloat16_as_ushort(maximum.y)));
        // Eight adjacent lanes each own four columns of the same SF group.
        CUTLASS_PRAGMA_UNROLL
        for (int shift = 1; shift < 8; shift *= 2)
            amax = max(amax, __shfl_xor_sync(0xffffffff, amax, shift));
        if constexpr (kClamp) {
            const auto clipped = __hmin2_nan(__bfloat162bfloat162(__ushort_as_bfloat16(amax)),
                                             __float2bfloat162_rn(args.clamp_value));
            amax = __bfloat16_as_ushort(clipped.x);
        }
        // Upstream get_ue8m0_sf_exp<e4m3>: ceil(amax / 448) to a
        // power of two, with the same 2^-22 floor (amax clamp 1e-4).
        return max((amax + 31u) >> 7, 113u) - 8u;
    }

    CUTLASS_DEVICE uint32_t cast_fp8(const __ppu_bfloat162& lo, const __ppu_bfloat162& hi,
                                    const uint32_t exponent) const {
        const auto inverse = __bfloat162bfloat162(__ushort_as_bfloat16((254u - exponent) << 7));
        const auto scaled_lo = __bfloat1622float2(__hmul2(clamp(lo), inverse));
        const auto scaled_hi = __bfloat1622float2(__hmul2(clamp(hi), inverse));
        return __hg_fp8x4_e4m3(make_float4(scaled_lo.x, scaled_lo.y, scaled_hi.x, scaled_hi.y)).__x;
    }

    CUTLASS_DEVICE __ppu_bfloat162 clamp(const __ppu_bfloat162& value) const {
        if constexpr (not kClamp)
            return value;
        auto clipped = __hmin2_nan(__habs2(value), __float2bfloat162_rn(args.clamp_value));
        // Match upstream's symmetric clamp: preserve signs (including -0) and NaNs.
        clipped.x = __ushort_as_bfloat16(__bfloat16_as_ushort(clipped.x) | (__bfloat16_as_ushort(value.x) & 0x8000u));
        clipped.y = __ushort_as_bfloat16(__bfloat16_as_ushort(clipped.y) | (__bfloat16_as_ushort(value.y) & 0x8000u));
        return clipped;
    }

    CUTLASS_DEVICE void finalize() const {
        if constexpr (kClamp and kCount) {
            auto count = num_clamped;
            CUTLASS_PRAGMA_UNROLL
            for (int offset = 16; offset > 0; offset /= 2)
                count += __shfl_down_sync(0xffffffffu, count, offset);
            if (threadIdx.x % 32 == 0)
                atomicAdd(args.clamped_count, count);
        }
    }

    CUTLASS_DEVICE void store_sf(const uint32_t row_idx, const uint32_t group_n_idx,
                                 const uint32_t batch_idx, const uint8_t sf) const {
        const auto sf_idx = (batch_idx * args.shape_n + group_n_idx) / kSFGranN;
        auto* sf_ptr = reinterpret_cast<uint8_t*>(args.sfd + (sf_idx / 4) * args.sfd_stride + row_idx);
        sf_ptr[sf_idx % 4] = sf;
    }
};

struct CastToFP8PerToken {
    template <bool kIsAlignedN, int BM, int RowTiles,
              class Accumulator, class Coordinates, class MMA, class SharedValues>
    CUTLASS_DEVICE static void reduce_and_publish(
        const Accumulator& acc, const Coordinates& coords, MMA mma,
        SharedValues* shared_values, float* row_scales,
        int first, int rows, int cols, int tid) {
        using namespace cute;
        using WarpLayout = typename MMA::ThrLayoutVMNK;
        constexpr int warps_m = size<1>(WarpLayout{});
        constexpr int warps_n = size<2>(WarpLayout{});
        // Four adjacent PPU MMA lanes own columns from the same row.
        auto* partial = reinterpret_cast<float*>(shared_values);
        const int warp_n = tid / 32 / warps_m;
        cp_async_wait<0>();
        __syncthreads();
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size<1>(acc); ++m) {
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<1>(shape<0>(acc)); ++mi) {
                const int row = get<0>(coords(make_coord(0, mi, 0), m, 0));
                float top = 0.f;
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < size<2>(acc); ++k) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int ni = 0; ni < size<2>(shape<0>(acc)); ++ni) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int v = 0; v < size<0>(shape<0>(acc)); ++v) {
                            const auto pos = make_coord(v, mi, ni);
                            if (kIsAlignedN || get<1>(coords(pos, m, k)) < cols)
                                top = fmaxf(top, fabsf(float(acc(pos, m, k))));
                        }
                    }
                }
                top = epilogue::quant::warp_amax<4>(top);
                if (tid % 4 == 0) partial[warp_n * BM + row] = top;
            }
        }
        __syncthreads();
        for (int row = tid; row < rows; row += blockDim.x) {
            float top = partial[row];
            CUTLASS_PRAGMA_UNROLL
            for (int w = 1; w < warps_n; ++w)
                top = fmaxf(top, partial[w * BM + row]);
            const auto rounded = __float2bfloat16(top);
            auto* scale = row_scales + first + row;
            if constexpr (RowTiles == 1) {
                *scale = get_scale(float(rounded));
            } else {
                atomicMax(reinterpret_cast<unsigned*>(scale),
                    __float_as_uint(get_scale(float(rounded))));
                __threadfence();
            }
        }
        __syncthreads();
    }

    CUTLASS_DEVICE static float get_scale(float amax) {
        return fmaxf(amax, 1e-10f) * (1.f / 448.f);
    }
    CUTLASS_DEVICE static uint32_t cast_fp8(float x0, float x1, float x2, float x3, float inverse) {
        const auto bf16_lo = __floats2bfloat162_rn(x0, x1);
        const auto bf16_hi = __floats2bfloat162_rn(x2, x3);
        const float2 lo = __bfloat1622float2(bf16_lo);
        const float2 hi = __bfloat1622float2(bf16_hi);
        return uint32_t(quant::fp8_pair(lo.x * inverse, lo.y * inverse)) |
               (uint32_t(quant::fp8_pair(hi.x * inverse, hi.y * inverse)) << 16);
    }
};

} // namespace deep_gemm::epilogue::operators
