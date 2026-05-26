#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include "profiling_interface.hpp"

#include <iostream>

#include <cuda_runtime.h>
#include <cuda/std/cstdint>
#include <cuda_bf16.h>

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include <cutlass/tfloat32.h>


#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/cooperative_gemm.hpp"

#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/detail/layout.hpp"

#include "cute/ppu_util.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "tools/util/include/cutlass/util/packed_stride.hpp"
#include "scheduler_cutlass3.cuh"
#include "utils_cutlass3.h"
#include "utils.cuh"

#include "ppu_include.hpp"

using namespace cute;

namespace deep_gemm {

namespace sm80_hc_detail {

static constexpr uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}
CUTLASS_DEVICE uint32_t smem_addr(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

CUTLASS_DEVICE void cp_async_16(void* dst, const void* src, const bool pred) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 800)
    const uint32_t dst_addr = smem_addr(dst);
    const uint32_t src_size = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
                 "r"(dst_addr), "l"(src), "r"(src_size));
#endif
}

CUTLASS_DEVICE float quad_reduce_sum(float value) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 800)
    value += __shfl_down_sync(0xffffffff, value, 2, 4);
    value += __shfl_down_sync(0xffffffff, value, 1, 4);
#endif
    return value;
}

template <bool kFastBF16ToTF32>
struct ToTF32 {
    CUTLASS_DEVICE cutlass::tfloat32_t operator()(const __nv_bfloat16& value) const {
        if constexpr (kFastBF16ToTF32) {
            return cutlass::tfloat32_t::bitcast(static_cast<uint32_t>(__bfloat16_as_ushort(value)) << 16);
        } else {
            return cutlass::tfloat32_t(__bfloat162float(value));
        }
    }

    CUTLASS_DEVICE cutlass::tfloat32_t operator()(const float& value) const {
        return cutlass::tfloat32_t(value);
    }
};

template <bool kFastBF16ToTF32, class ThrMma, class SmemA, class SmemB, class Accum>
CUTLASS_DEVICE void gemm_explicit(ThrMma const& thr_mma,
                                  const uint32_t tid,
                                  SmemA const& sA,
                                  SmemB const& sB,
                                  Accum& accum,
                                  float& sqr_sum_acc_lo,
                                  float& sqr_sum_acc_hi) {
    using namespace cute;

    using InputTypeA = typename SmemA::value_type;
    using InputTypeB = typename SmemB::value_type;
    using ComputeTypeA = typename ThrMma::ValTypeA;
    using ComputeTypeB = typename ThrMma::ValTypeB;

    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCrAi = make_fragment_like<InputTypeA>(tCrA);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    Tensor tCrBi = make_fragment_like<InputTypeB>(tCrB);

    auto smem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<DefaultCopy, InputTypeA>{}, thr_mma);
    auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(tid);
    Tensor tCsA = smem_thr_copy_A.partition_S(sA);
    Tensor tCrAi_copy_view = smem_thr_copy_A.retile_D(tCrAi);

    auto smem_tiled_copy_B = make_tiled_copy_B(Copy_Atom<DefaultCopy, InputTypeB>{}, thr_mma);
    auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(tid);
    Tensor tCsB = smem_thr_copy_B.partition_S(sB);
    Tensor tCrBi_copy_view = smem_thr_copy_B.retile_D(tCrBi);

    copy(smem_tiled_copy_A, tCsA(_, _, Int<0>{}), tCrAi_copy_view(_, _, Int<0>{}));
    copy(smem_tiled_copy_B, tCsB(_, _, Int<0>{}), tCrBi_copy_view(_, _, Int<0>{}));

    constexpr int K_BLOCK_MAX = size<2>(tCrA);

    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        if (k_block < K_BLOCK_MAX - 1) {
            const int k_next = k_block + 1;
            copy(smem_tiled_copy_A, tCsA(_, _, k_next), tCrAi_copy_view(_, _, k_next));
            copy(smem_tiled_copy_B, tCsB(_, _, k_next), tCrBi_copy_view(_, _, k_next));
        }

        if constexpr (kFastBF16ToTF32) {
            auto tCrAi_k = tCrAi(_, _, k_block);
            CUTE_UNROLL
            for (int i = 0; i < size(tCrAi_k); i += 2) {
                const float value_lo = __bfloat162float(tCrAi_k(i));
                const float value_hi = __bfloat162float(tCrAi_k(i + 1));
                sqr_sum_acc_lo = fmaf(value_lo, value_lo, sqr_sum_acc_lo);
                sqr_sum_acc_hi = fmaf(value_hi, value_hi, sqr_sum_acc_hi);
            }
            cute::transform(tCrAi(_, _, k_block), tCrA(_, _, k_block), ToTF32<true>{});
        } else {
            auto tCrAi_k = tCrAi(_, _, k_block);
            auto tCrA_k = tCrA(_, _, k_block);
            CUTE_UNROLL
            for (int i = 0; i < size(tCrAi_k); i += 2) {
                const float value_lo = __bfloat162float(tCrAi_k(i));
                const float value_hi = __bfloat162float(tCrAi_k(i + 1));
                sqr_sum_acc_lo = fmaf(value_lo, value_lo, sqr_sum_acc_lo);
                sqr_sum_acc_hi = fmaf(value_hi, value_hi, sqr_sum_acc_hi);
                tCrA_k(i) = cutlass::tfloat32_t(value_lo);
                tCrA_k(i + 1) = cutlass::tfloat32_t(value_hi);
            }
        }

        cute::transform(tCrBi(_, _, k_block), tCrB(_, _, k_block), ToTF32<kFastBF16ToTF32>{});

        using Atom = typename ThrMma::Atom;
        gemm(static_cast<Atom const&>(thr_mma), tCrA(_, _, k_block), tCrB(_, _, k_block), accum);
    }
}

} // namespace sm80_hc_detail

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumSplits, uint32_t kNumThreads,
          bool kFastBF16ToTF32,
          bool kReduceSplits>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads, 2)
sm80_tf32_hc_prenorm_gemm_impl(const uint32_t shape_m,
                               const void* a_ptr,
                               const float* b,
                               float* d,
                               float* sqr_sum,
                               const int64_t stride_a_m,
                               const int64_t stride_b_n,
                               const int64_t stride_d_split,
                               const int64_t stride_d_m) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 800)) or defined(__CLION_IDE__)
    using namespace cute;

    DG_STATIC_ASSERT(BLOCK_M % 16 == 0 and BLOCK_M <= 256, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_K == 64 or BLOCK_K == 128, "Invalid block K");
    DG_STATIC_ASSERT(BLOCK_N % 8 == 0 and BLOCK_N <= 32, "Invalid block N");
    // DG_STATIC_ASSERT(BLOCK_N % 16 == 0 and BLOCK_N <= 32, "Invalid block N");
    DG_STATIC_ASSERT(kNumThreads == BLOCK_M * 2, "Invalid number of threads");
    DG_STATIC_ASSERT(SHAPE_N <= BLOCK_N, "Invalid shape N");
    DG_STATIC_ASSERT(SHAPE_K % BLOCK_K == 0, "Invalid shape K");
    DG_STATIC_ASSERT(BLOCK_K % 8 == 0, "Invalid block K for SM80 TF32 MMA");

    const auto* a = reinterpret_cast<const __nv_bfloat16*>(a_ptr);
    const uint32_t tid = threadIdx.x;

    extern __shared__ __align__(16) uint8_t smem_buffer[];
    constexpr uint32_t kSmemAStride = BLOCK_K + 8;
    constexpr uint32_t kSmemABytes = BLOCK_M * kSmemAStride * sizeof(__nv_bfloat16);
    constexpr uint32_t kSmemBStride = BLOCK_K + 4;
    constexpr uint32_t kSmemBBytes = BLOCK_N * kSmemBStride * sizeof(float);
    constexpr uint32_t kStageBytes = kSmemABytes + kSmemBBytes;

    constexpr uint32_t kNumKBlocks = sm80_hc_detail::ceil_div(SHAPE_K, BLOCK_K);
    constexpr uint32_t kNumKBlocksPerSplit = kNumKBlocks / kNumSplits;
    constexpr uint32_t kRemainKBlocks = kNumKBlocks % kNumSplits;

    const uint32_t block_idx = blockIdx.x;
    const uint32_t m_block_idx = block_idx / kNumSplits;
    const uint32_t k_split_idx = block_idx % kNumSplits;
    const uint32_t m_base = m_block_idx * BLOCK_M;
    const uint32_t k_offset = (k_split_idx * kNumKBlocksPerSplit +
                               (k_split_idx < kRemainKBlocks ? k_split_idx : kRemainKBlocks)) * BLOCK_K;
    const uint32_t num_total_stages = kNumKBlocksPerSplit + (k_split_idx < kRemainKBlocks);

#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
    using MmaAtom = MMA_Atom<PPU0015_16x8x8_F32TF32TF32F32_TN>;
#else
    using MmaAtom = MMA_Atom<PPU0010_16x16x8_F32TF32TF32F32_TN>;
#endif
    // using MmaAtom = MMA_Atom<PPU0015_16x16x8_F32TF32TF32F32_TN>;
    using TiledMma = cute::TiledMMA<MmaAtom, Layout<Shape<Int<BLOCK_M / 16>, _1, _1>>>;
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    auto d_layout = make_layout(make_shape(Int<BLOCK_M>{}, Int<BLOCK_N>{}),
                                make_stride(stride_d_m, int64_t(1)));
    auto d_tensor = make_tensor(make_gmem_ptr(d), d_layout);
    auto tdD = thr_mma.partition_C(d_tensor);
    auto accum = thr_mma.make_fragment_C(tdD);
    clear(accum);

    auto c_identity = make_identity_tensor(make_shape(Int<BLOCK_M>{}, Int<BLOCK_N>{}));
    auto tcC = thr_mma.partition_C(c_identity);

    auto smem_a_layout = make_layout(make_shape(Int<BLOCK_M>{}, Int<BLOCK_K>{}),
                                     make_stride(Int<kSmemAStride>{}, Int<1>{}));
    auto smem_b_layout = make_layout(make_shape(Int<BLOCK_N>{}, Int<BLOCK_K>{}),
                                     make_stride(Int<kSmemBStride>{}, Int<1>{}));

    float sqr_sum_acc_lo = 0.0f;
    float sqr_sum_acc_hi = 0.0f;

    constexpr uint32_t kAElementsPerVec = sizeof(uint4) / sizeof(__nv_bfloat16);
    constexpr uint32_t kAVectors = BLOCK_M * BLOCK_K / kAElementsPerVec;
    constexpr uint32_t kAVecsPerRow = BLOCK_K / kAElementsPerVec;
    constexpr uint32_t kBElementsPerVec = sizeof(uint4) / sizeof(float);
    constexpr uint32_t kBVectors = BLOCK_N * BLOCK_K / kBElementsPerVec;
    constexpr uint32_t kBVecsPerRow = BLOCK_K / kBElementsPerVec;

    auto issue_cp_async_stage = [&](const uint32_t s, const uint32_t stage) {
        const uint32_t k_base = k_offset + s * BLOCK_K;
        auto* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_buffer + stage * kStageBytes);
        auto* smem_b = reinterpret_cast<float*>(smem_buffer + stage * kStageBytes + kSmemABytes);

        for (uint32_t idx = tid; idx < kAVectors; idx += kNumThreads) {
            const uint32_t m = idx / kAVecsPerRow;
            const uint32_t k = (idx % kAVecsPerRow) * kAElementsPerVec;
            const bool valid_m = m_base + m < shape_m;
            const auto* src = reinterpret_cast<const uint4*>(
                a + (valid_m ? (m_base + m) : 0) * stride_a_m + k_base + k);
            sm80_hc_detail::cp_async_16(smem_a + m * kSmemAStride + k, src, valid_m);
        }

        for (uint32_t idx = tid; idx < kBVectors; idx += kNumThreads) {
            const uint32_t n = idx / kBVecsPerRow;
            const uint32_t k = (idx % kBVecsPerRow) * kBElementsPerVec;
            const bool valid_n = n < SHAPE_N;
            const auto* src = reinterpret_cast<const uint4*>(
                b + (valid_n ? n : 0) * stride_b_n + k_base + k);
            sm80_hc_detail::cp_async_16(smem_b + n * kSmemBStride + k, src, valid_n);
        }
        cp_async_fence();
    };

    if (num_total_stages > 0)
        issue_cp_async_stage(0, 0);

    #pragma unroll 1
    for (uint32_t s = 0; s < num_total_stages; ++s) {
        const uint32_t stage = s & 1;
        auto* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_buffer + stage * kStageBytes);
        auto* smem_b = reinterpret_cast<float*>(smem_buffer + stage * kStageBytes + kSmemABytes);

        cp_async_wait<0>();
        __syncthreads();

        if (s + 1 < num_total_stages)
            issue_cp_async_stage(s + 1, (s + 1) & 1);

        auto a_tensor = make_tensor(make_smem_ptr(smem_a), smem_a_layout);
        auto b_tensor = make_tensor(make_smem_ptr(smem_b), smem_b_layout);
        sm80_hc_detail::gemm_explicit<kFastBF16ToTF32>(
            thr_mma, tid, a_tensor, b_tensor, accum, sqr_sum_acc_lo, sqr_sum_acc_hi);
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t i = 0; i < size(accum); ++i) {
        const auto coord = tcC(i);
        const uint32_t m = get<0>(coord);
        const uint32_t n = get<1>(coord);
        if (m_base + m < shape_m and n < SHAPE_N) {
            float* out = d + (m_base + m) * stride_d_m + n;
            if constexpr (kReduceSplits) {
                atomicAdd(out, accum(i));
            } else {
                out[k_split_idx * stride_d_split] = accum(i);
            }
        }
    }

    const uint32_t lane_idx = tid & 31;
    const uint32_t warp_idx = tid >> 5;
    const uint32_t row_lo = warp_idx * 16 + lane_idx / 4;
    const uint32_t row_hi = row_lo + 8;
    const float sqr_sum_lo = sm80_hc_detail::quad_reduce_sum(sqr_sum_acc_lo);
    const float sqr_sum_hi = sm80_hc_detail::quad_reduce_sum(sqr_sum_acc_hi);

    if ((lane_idx & 3) == 0 and row_lo < BLOCK_M and m_base + row_lo < shape_m) {
        if constexpr (kReduceSplits) {
            atomicAdd(sqr_sum + m_base + row_lo, sqr_sum_lo);
        } else {
            sqr_sum[k_split_idx * shape_m + m_base + row_lo] = sqr_sum_lo;
        }
    }

    if ((lane_idx & 3) == 0 and row_hi < BLOCK_M and m_base + row_hi < shape_m) {
        if constexpr (kReduceSplits) {
            atomicAdd(sqr_sum + m_base + row_hi, sqr_sum_hi);
        } else {
            sqr_sum[k_split_idx * shape_m + m_base + row_hi] = sqr_sum_hi;
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_80 or newer");
#endif
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumSplits,
          bool kFastBF16ToTF32,
          bool kReduceSplits = true>
class HcPrenormGemm {
public:
    static void run(float* out,
                    float* sqr_sum,
                    uint32_t m,
                    const __nv_bfloat16* lhs,
                    const float* rhs,
                    cudaStream_t stream,
                    int /*num_sms*/ = 0,
                    uint32_t /*smem_size_from_python*/ = 0) {
        static_assert(BLOCK_N >= SHAPE_N, "BLOCK_N must >= SHAPE_N");
        static_assert(BLOCK_N <= 32, "BLOCK_N must <= 32 for SM80 TF32 MMA");
        static_assert(BLOCK_K == 64 || BLOCK_K == 128, "BLOCK_K must be 64 or 128");

        constexpr uint32_t kNumThreads = BLOCK_M * 2;

        constexpr uint32_t kSmemSize =
            2 * (BLOCK_M * (BLOCK_K + 8) * sizeof(uint16_t) +
                 BLOCK_N * (BLOCK_K + 4) * sizeof(float));

        auto* kernel = sm80_tf32_hc_prenorm_gemm_impl<
            SHAPE_N, SHAPE_K,
            BLOCK_M, BLOCK_N, BLOCK_K,
            kNumSplits, kNumThreads,
            kFastBF16ToTF32,
            kReduceSplits>;

        cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kSmemSize);

        if constexpr (kReduceSplits) {
            cudaMemsetAsync(out, 0, size_t(m) * SHAPE_N * sizeof(float), stream);
            cudaMemsetAsync(sqr_sum, 0, size_t(m) * sizeof(float), stream);
        }

        const uint32_t grid_m = sm80_hc_detail::ceil_div(m, BLOCK_M);
        dim3 grid(grid_m * kNumSplits);
        dim3 block(kNumThreads);

        kernel<<<grid, block, kSmemSize, stream>>>(
            m,
            lhs,
            rhs,
            out,
            sqr_sum,
            SHAPE_K,
            SHAPE_K,
            size_t(m) * SHAPE_N,
            SHAPE_N);
    }
};

} // namespace deep_gemm

#pragma clang diagnostic pop
