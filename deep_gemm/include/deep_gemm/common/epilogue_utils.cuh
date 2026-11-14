#pragma once

#include <cstdint>
#include <cute/tensor.hpp>
#include <cutlass/arch/memory.h>
#include <deep_gemm/tix/cvt.cuh>

namespace deep_gemm {

// Shared by fused MoE and K-grouped GEMM. Existing fused calls default to
// overwrite; FP32 callers can opt into beta=1 or an explicit zero fill.
template <typename AccT, typename DstT,
          uint32_t SHAPE_N, uint32_t BLOCK_N, uint32_t STRIDE_CM,
          bool kWithAccumulation = false, bool kZeroOutput = false,
          class TAcc, class TCcC, class... Ts>
__device__ __forceinline__ void epilogue_no_tsm(TAcc& accum, const TCcC& tCcC, void* c_ptr,
                                             uint32_t c_row_base, uint32_t num_valid_rows,
                                             uint32_t blk_n_offset) {
    using namespace cute;
    static_assert((!kWithAccumulation && !kZeroOutput) || cute::is_same_v<DstT, float>,
                  "Accumulation and zero-output modes require FP32 output");
#if __HGGC_ARCH__ == 150
    CUTLASS_PRAGMA_UNROLL
    for (int warp_m = 0; warp_m < size<1>(tCcC); ++warp_m) {
        CUTLASS_PRAGMA_UNROLL
        for (int mma_m = 0; mma_m < size<0,1>(tCcC); ++mma_m) {
            const uint32_t local_row = get<0>(tCcC(make_coord(_0{}, mma_m, _0{}), warp_m, _0{}));
            const uint32_t row = c_row_base + local_row;
            if (local_row < num_valid_rows) {
                DstT* cur_c_ptr = reinterpret_cast<DstT*>(c_ptr) + static_cast<size_t>(row) * STRIDE_CM + blk_n_offset;
                // Each fragment owns two adjacent output elements. Keep the
                // fused BF16 store and the FP32 store on the same traversal.
                // For FP32 beta=1, load a row's pairs before consuming any of
                // them so the compiler can overlap the independent C reads.
                if constexpr (cute::is_same_v<AccT, float> && cute::is_same_v<DstT, float> &&
                              STRIDE_CM % 2 == 0 && SHAPE_N % 2 == 0 && BLOCK_N % 2 == 0) {
                    // Contiguous tensors may still have a one-float storage
                    // offset. Keep the scalar path for such unaligned views.
                    if ((reinterpret_cast<uintptr_t>(cur_c_ptr) & (alignof(float2) - 1)) == 0) {
                        constexpr int kPairsPerMma = size<0,2>(TCcC{});
                        constexpr int kPairsPerRow = size<2>(TCcC{}) * kPairsPerMma;
                        float2 previous[kPairsPerRow] = {};
                        if constexpr (kWithAccumulation && !kZeroOutput) {
                            CUTLASS_PRAGMA_UNROLL
                            for (int warp_n = 0; warp_n < size<2>(tCcC); ++warp_n) {
                                CUTLASS_PRAGMA_UNROLL
                                for (int mma_n = 0; mma_n < kPairsPerMma; ++mma_n) {
                                    const size_t n_idx = get<1>(tCcC(make_coord(_0{}, mma_m, mma_n), warp_m, warp_n));
                                    cutlass::arch::global_load<float2, sizeof(float2)>(
                                        previous[warp_n * kPairsPerMma + mma_n], cur_c_ptr + n_idx,
                                        SHAPE_N % BLOCK_N == 0 || n_idx < SHAPE_N - blk_n_offset);
                                }
                            }
                        }
                        CUTLASS_PRAGMA_UNROLL
                        for (int warp_n = 0; warp_n < size<2>(tCcC); ++warp_n) {
                            CUTLASS_PRAGMA_UNROLL
                            for (int mma_n = 0; mma_n < kPairsPerMma; ++mma_n) {
                                const size_t n_idx = get<1>(tCcC(make_coord(_0{}, mma_m, mma_n), warp_m, warp_n));
                                float2 value = {};
                                if constexpr (!kZeroOutput) {
                                    const AccT* acc_ptr = cute::raw_pointer_cast(
                                        accum(make_coord(_, mma_m, mma_n), warp_m, warp_n).data());
                                    value = {acc_ptr[0], acc_ptr[1]};
                                    if constexpr (kWithAccumulation) {
                                        const auto old = previous[warp_n * kPairsPerMma + mma_n];
                                        value.x += old.x;
                                        value.y += old.y;
                                    }
                                }
                                cutlass::arch::global_store<float2, sizeof(float2)>(
                                    value, cur_c_ptr + n_idx,
                                    SHAPE_N % BLOCK_N == 0 || n_idx < SHAPE_N - blk_n_offset);
                            }
                        }
                        continue;
                    }
                }
                CUTLASS_PRAGMA_UNROLL
                for (int warp_n = 0; warp_n < size<2>(tCcC); ++warp_n) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int mma_n = 0; mma_n < size<0,2>(tCcC); ++mma_n) {
                        AccT* acc_ptr = cute::raw_pointer_cast(accum(make_coord(_, mma_m, mma_n), warp_m, warp_n).data());
                        const size_t n_idx = get<1>(tCcC(make_coord(_0{}, mma_m, mma_n), warp_m, warp_n));
                        if (SHAPE_N % BLOCK_N == 0 || n_idx < (SHAPE_N - blk_n_offset)) {
                            DstT* dst_ptr = cur_c_ptr + n_idx;
                            if constexpr (!cute::is_same_v<DstT, float>) {
                                // Preserve the existing fused BF16 conversion and packed store.
                                const uint32_t value = tix::cvt_rte_bf16x2_f32(acc_ptr[1], acc_ptr[0]);
                                *reinterpret_cast<uint32_t*>(dst_ptr) = value;
                            } else if constexpr (kZeroOutput) {
                                dst_ptr[0] = DstT(0);
                                if ((SHAPE_N % BLOCK_N == 0 && BLOCK_N % 2 == 0) || n_idx + 1 < SHAPE_N - blk_n_offset)
                                    dst_ptr[1] = DstT(0);
                            } else if constexpr (kWithAccumulation) {
                                dst_ptr[0] += acc_ptr[0];
                                if ((SHAPE_N % BLOCK_N == 0 && BLOCK_N % 2 == 0) || n_idx + 1 < SHAPE_N - blk_n_offset)
                                    dst_ptr[1] += acc_ptr[1];
                            } else {
                                dst_ptr[0] = acc_ptr[0];
                                if ((SHAPE_N % BLOCK_N == 0 && BLOCK_N % 2 == 0) || n_idx + 1 < SHAPE_N - blk_n_offset)
                                    dst_ptr[1] = acc_ptr[1];
                            }
                        }
                    }
                }
            }
        }
    }
#else
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tCcC); ++i) {
        const size_t local_row = cute::get<0>(tCcC(i));
        if (local_row < num_valid_rows) {
            const size_t n_idx = cute::get<1>(tCcC(i));
            if (n_idx < (SHAPE_N - blk_n_offset)) {
                AccT* acc_ptr = cute::raw_pointer_cast(accum.data()) + accum.layout()(i);
                DstT* dst_ptr = reinterpret_cast<DstT*>(c_ptr) + (static_cast<size_t>(c_row_base) + local_row) * STRIDE_CM + blk_n_offset + n_idx;
                if constexpr (kZeroOutput) {
                    *dst_ptr = DstT(0);
                } else if constexpr (kWithAccumulation) {
                    *dst_ptr += *acc_ptr;
                } else {
                    *dst_ptr = DstT(*acc_ptr);
                }
            }
        }
    }
#endif
}

// Full-CTA bounds and float4 alignment are checked by the caller.
// Adjacent N fragments permit pairing two MMA rows for contiguous C requests.
template <uint32_t STRIDE_CM, class TAcc, class TCcC>
__device__ __forceinline__ void epilogue_no_tsm_float4(
        TAcc& accum, const TCcC& tCcC, float* c_ptr, uint32_t row_base,
        uint32_t n_offset) {
    using namespace cute;
    constexpr int kVectors = size<2>(TCcC{});
    constexpr bool kPairRows =
#if __HGGC_ARCH__ == 150
        is_same_v<decltype(stride<2>(TCcC{})), ScaledBasis<_16, 1>> && kVectors % 2 == 0;
#else
        false;
#endif
    constexpr int kRows = kPairRows ? 2 : 1;
    const bool odd_lane = (threadIdx.x & 1) != 0;
    const int lane_column = (threadIdx.x >> 2) & (kRows - 1);
    CUTLASS_PRAGMA_UNROLL
    for (int wm = 0; wm < size<1>(tCcC); ++wm) {
        CUTLASS_PRAGMA_UNROLL
        for (int mm = 0; mm < size<0,1>(tCcC); ++mm) {
            const uint32_t row = get<0>(tCcC(make_coord(_0{}, mm, _0{}), wm, _0{}));
            const uint32_t first_row = row_base + (row & ~(kRows - 1));
            auto address = [&](int wn, int r) {
                const auto col = get<1>(tCcC(make_coord(_0{}, mm, _0{}), wm, wn + lane_column)) +
                    (odd_lane ? 6 : 0);
                return c_ptr + static_cast<size_t>(first_row + r) * STRIDE_CM + n_offset + col;
            };
            // Load the complete batch of old C before packing and storing.
            float4 previous[kVectors];
            CUTLASS_PRAGMA_UNROLL
            for (int wn = 0; wn < kVectors; wn += kRows) {
                CUTLASS_PRAGMA_UNROLL
                for (int r = 0; r < kRows; ++r) {
                    if constexpr (kPairRows) {
                        uint4 raw;
                        asm volatile("ppu.ld.global.bl.v4.u32 {%0, %1, %2, %3}, [%4];"
                                     : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w) : "l"(address(wn, r)));
                        previous[wn + r] = {__uint_as_float(raw.x), __uint_as_float(raw.y),
                                            __uint_as_float(raw.z), __uint_as_float(raw.w)};
                    } else {
                        cutlass::arch::global_load<float4, sizeof(float4)>(previous[wn + r], address(wn, r), true);
                    }
                }
            }
            CUTLASS_PRAGMA_UNROLL
            for (int wn = 0; wn < kVectors; wn += kRows) {
                float4 fragment[kRows];
                for_each(make_int_sequence<kRows>{}, [&](auto i) {
                    const auto* a = raw_pointer_cast(accum(make_coord(_, mm, _0{}), wm, wn + i).data());
                    const auto* b = raw_pointer_cast(accum(make_coord(_, mm, _1{}), wm, wn + i).data());
                    const float x = __shfl_xor_sync(0xffffffffu, odd_lane ? a[0] : b[0], 1);
                    const float y = __shfl_xor_sync(0xffffffffu, odd_lane ? a[1] : b[1], 1);
                    fragment[i] = odd_lane ? float4{x, y, b[0], b[1]} : float4{a[0], a[1], x, y};
                });
                if constexpr (kPairRows) {
                    // Lane bit 2 owns the adjacent row; exchange one float4.
                    const bool upper = (threadIdx.x & 4) != 0;
                    const auto send = upper ? fragment[0] : fragment[1];
                    const float4 peer{__shfl_xor_sync(0xffffffffu, send.x, 4),
                                      __shfl_xor_sync(0xffffffffu, send.y, 4),
                                      __shfl_xor_sync(0xffffffffu, send.z, 4),
                                      __shfl_xor_sync(0xffffffffu, send.w, 4)};
                    if (upper) fragment[0] = peer;
                    else fragment[1] = peer;
                }
                CUTLASS_PRAGMA_UNROLL
                for (int r = 0; r < kRows; ++r) {
                    const auto old = previous[wn + r];
                    const auto data = fragment[r];
                    const float4 value{old.x + data.x, old.y + data.y, old.z + data.z, old.w + data.w};
                    if constexpr (kPairRows) {
                        asm volatile("ppu.st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                                     :: "l"(address(wn, r)), "r"(__float_as_uint(value.x)), "r"(__float_as_uint(value.y)),
                                        "r"(__float_as_uint(value.z)), "r"(__float_as_uint(value.w)));
                    } else {
                        cutlass::arch::global_store<float4, sizeof(float4)>(value, address(wn, r), true);
                    }
                }
            }
        }
    }
}

}  // namespace deep_gemm
