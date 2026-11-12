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

template <uint32_t STRIDE_CM, class TAcc, class TCcC>
__device__ __forceinline__ void epilogue_no_tsm_float4(
        TAcc& accum, const TCcC& tCcC, float* c_ptr, uint32_t row_base,
        uint32_t n_offset) {
    using namespace cute;
    const bool odd_lane = (threadIdx.x & 1) != 0;
    CUTLASS_PRAGMA_UNROLL
    for (int wm = 0; wm < size<1>(tCcC); ++wm) {
        CUTLASS_PRAGMA_UNROLL
        for (int mm = 0; mm < size<0,1>(tCcC); ++mm) {
            const uint32_t row = get<0>(tCcC(make_coord(_0{}, mm, _0{}), wm, _0{}));
            constexpr int kVectors = size<2>(TCcC{});
            float4 old[kVectors];
            CUTLASS_PRAGMA_UNROLL
            for (int wn = 0; wn < kVectors; ++wn) {
                const uint32_t col = get<1>(tCcC(make_coord(_0{}, mm, _0{}), wm, wn)) + (odd_lane ? 6 : 0);
                const float* dst = c_ptr + static_cast<size_t>(row_base + row) * STRIDE_CM + n_offset + col;
                cutlass::arch::global_load<float4, sizeof(float4)>(old[wn], dst, true);
            }
            CUTLASS_PRAGMA_UNROLL
            for (int wn = 0; wn < kVectors; ++wn) {
                const float* p0 = raw_pointer_cast(accum(make_coord(_, mm, _0{}), wm, wn).data());
                const float* p1 = raw_pointer_cast(accum(make_coord(_, mm, _1{}), wm, wn).data());
                const float peer00 = __shfl_xor_sync(0xffffffffu, p0[0], 1);
                const float peer01 = __shfl_xor_sync(0xffffffffu, p0[1], 1);
                const float peer10 = __shfl_xor_sync(0xffffffffu, p1[0], 1);
                const float peer11 = __shfl_xor_sync(0xffffffffu, p1[1], 1);
                const float4 packed = odd_lane
                    ? float4{peer10, peer11, p1[0], p1[1]}
                    : float4{p0[0], p0[1], peer00, peer01};
                const float4 value = {packed.x + old[wn].x, packed.y + old[wn].y,
                                      packed.z + old[wn].z, packed.w + old[wn].w};
                const uint32_t col = get<1>(tCcC(make_coord(_0{}, mm, _0{}), wm, wn)) + (odd_lane ? 6 : 0);
                float* dst = c_ptr + static_cast<size_t>(row_base + row) * STRIDE_CM + n_offset + col;
                cutlass::arch::global_store<float4, sizeof(float4)>(value, dst, true);
            }
        }
    }
}

}  // namespace deep_gemm
