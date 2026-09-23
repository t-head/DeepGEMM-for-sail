#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

// NOTES: host-only helpers, skipped when the header is pulled in by the C++ JIT
#ifndef TF32_HC_PRENORM_HGRTC
    #include <deep_gemm/common/profiling_interface.cuh>
#endif

#include <hggc_runtime.h>
#include <hggc/std/cstdint>
#include <hggc_bf16.h>

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include <cutlass/tfloat32.h>
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/detail/layout.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cute/ppu_util.hpp"
#include <cute/arch/copy_ppu.hpp>

#ifndef TF32_HC_PRENORM_HGRTC
    #include "tools/util/include/cutlass/util/packed_stride.hpp"
#endif
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>
#include <deep_gemm/common/utils_cutlass3.cuh>
#include <deep_gemm/common/utils.cuh>
#include "ppu_include.hpp"

using namespace cute;

namespace deep_gemm {

namespace hc_detail {

static constexpr uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}
CUTLASS_DEVICE uint32_t smem_addr(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

CUTLASS_DEVICE void cp_async_16(void* dst, const void* src, const bool pred) {
#if defined(__HGGC_ARCH__)
    const uint32_t dst_addr = smem_addr(dst);
    const uint32_t src_size = pred ? 16 : 0;
    asm volatile("ppu.cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
                 "r"(dst_addr), "l"(src), "r"(src_size));
#endif
}

CUTLASS_DEVICE float quad_reduce_sum(float value) {
#if defined(__HGGC_ARCH__)
    value += __shfl_down_sync(0xffffffff, value, 2, 4);
    value += __shfl_down_sync(0xffffffff, value, 1, 4);
#endif
    return value;
}

CUTLASS_DEVICE uint32_t atom_add_acq_rel_gpu(uint32_t* ptr, uint32_t val) {
    uint32_t old = 0;
#if defined(__HGGC_ARCH__)
    asm volatile("ppu.atom.add.acq_rel.gpu.u32 %0,[%1],%2;"
                 : "=r"(old) : "l"(ptr), "r"(val) : "memory");
#endif
    return old;
}

// Two adjacent floats in one 8-byte store
CUTLASS_DEVICE void st_relaxed_gpu_pair(float* ptr, float lo, float hi) {
#if defined(__HGGC_ARCH__)
    asm volatile("ppu.st.relaxed.gpu.v2.u32 [%0], {%1, %2};" ::
                 "l"(ptr), "r"(__float_as_uint(lo)), "r"(__float_as_uint(hi)) : "memory");
#else
    ptr[0] = lo;
    ptr[1] = hi;
#endif
}

CUTLASS_DEVICE void st_relaxed_gpu(float* ptr, float val) {
#if defined(__HGGC_ARCH__)
    asm volatile("ppu.st.relaxed.gpu.b32 [%0], %1;" ::
                 "l"(ptr), "r"(__float_as_uint(val)) : "memory");
#else
    *ptr = val;
#endif
}

// kSplitMinor: ws[row][split][col]; otherwise ws[split][row][col].
template <uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t kNumSplits, uint32_t kNumThreads,
          bool kSplitMinor = false, bool kRotateSquareWork = false>
CUTLASS_DEVICE void reduce_splits_body(float* __restrict__ out,
                                       float* __restrict__ sqrsum,
                                       const float* __restrict__ ws,
                                       const float* __restrict__ ws_s,
                                       const uint32_t m_base,
                                       const uint32_t num_tokens,
                                       const uint32_t tid) {
#if defined(__HGGC_ARCH__)

    const uint32_t rows_left = num_tokens - m_base;
    const uint32_t num_rows = rows_left < BLOCK_M ? rows_left : BLOCK_M;

    constexpr uint32_t kVecPerRow = SHAPE_N / 4;
    for (uint32_t i = tid; i < num_rows * kVecPerRow; i += kNumThreads) {
        const uint32_t r = i / kVecPerRow;
        const uint32_t c = (i - r * kVecPerRow) * 4;
        const int64_t off = int64_t(m_base + r) * SHAPE_N + c;
        constexpr int64_t kSplitStride = int64_t(SHAPE_N);
        const int64_t off_ws = kSplitMinor
                                   ? int64_t(m_base + r) * int64_t(kNumSplits) * SHAPE_N + c
                                   : off;
        const float* base = ws + off_ws;
        float4 acc0 = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 acc1 = acc0;
        float4 acc2 = acc0;
        float4 acc3 = acc0;
        if constexpr (kNumSplits % 4 == 0) {
            #pragma unroll 4
            for (uint32_t s = 0; s < kNumSplits; s += 4) {
                const float4 v0 = *reinterpret_cast<const float4*>(
                    base + (kSplitMinor ? int64_t(s) * kSplitStride : int64_t(s) * num_tokens * kSplitStride));
                const float4 v1 = *reinterpret_cast<const float4*>(
                    base + (kSplitMinor ? int64_t(s + 1) * kSplitStride : int64_t(s + 1) * num_tokens * kSplitStride));
                const float4 v2 = *reinterpret_cast<const float4*>(
                    base + (kSplitMinor ? int64_t(s + 2) * kSplitStride : int64_t(s + 2) * num_tokens * kSplitStride));
                const float4 v3 = *reinterpret_cast<const float4*>(
                    base + (kSplitMinor ? int64_t(s + 3) * kSplitStride : int64_t(s + 3) * num_tokens * kSplitStride));
                acc0.x += v0.x; acc0.y += v0.y; acc0.z += v0.z; acc0.w += v0.w;
                acc1.x += v1.x; acc1.y += v1.y; acc1.z += v1.z; acc1.w += v1.w;
                acc2.x += v2.x; acc2.y += v2.y; acc2.z += v2.z; acc2.w += v2.w;
                acc3.x += v3.x; acc3.y += v3.y; acc3.z += v3.z; acc3.w += v3.w;
            }
        } else if constexpr (kNumSplits % 2 == 0) {
            #pragma unroll 4
            for (uint32_t s = 0; s < kNumSplits; s += 2) {
                const float4 v0 = *reinterpret_cast<const float4*>(
                    base + (kSplitMinor ? int64_t(s) * kSplitStride : int64_t(s) * num_tokens * kSplitStride));
                const float4 v1 = *reinterpret_cast<const float4*>(
                    base + (kSplitMinor ? int64_t(s + 1) * kSplitStride : int64_t(s + 1) * num_tokens * kSplitStride));
                acc0.x += v0.x; acc0.y += v0.y; acc0.z += v0.z; acc0.w += v0.w;
                acc1.x += v1.x; acc1.y += v1.y; acc1.z += v1.z; acc1.w += v1.w;
            }
        } else {
            #pragma unroll 4
            for (uint32_t s = 0; s < kNumSplits; ++s) {
                const float4 v0 = *reinterpret_cast<const float4*>(
                    base + (kSplitMinor ? int64_t(s) * kSplitStride : int64_t(s) * num_tokens * kSplitStride));
                acc0.x += v0.x; acc0.y += v0.y; acc0.z += v0.z; acc0.w += v0.w;
            }
        }
        float4 res;
        res.x = (acc0.x + acc1.x) + (acc2.x + acc3.x);
        res.y = (acc0.y + acc1.y) + (acc2.y + acc3.y);
        res.z = (acc0.z + acc1.z) + (acc2.z + acc3.z);
        res.w = (acc0.w + acc1.w) + (acc2.w + acc3.w);
        *reinterpret_cast<float4*>(out + off) = res;
    }

    // Rotate square-sum work without changing each output's split order.
    uint32_t square_tid = tid;
    if constexpr (kRotateSquareWork) {
        const uint32_t d_tail = (num_rows * kVecPerRow) % kNumThreads;
        const uint32_t first_idle_warp = ((d_tail + 31u) / 32u * 32u) % kNumThreads;
        square_tid = (tid + kNumThreads - first_idle_warp) % kNumThreads;
    }
    for (uint32_t r = square_tid; r < num_rows; r += kNumThreads) {
        const uint32_t row = m_base + r;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        if constexpr (kNumSplits % 4 == 0) {
            if constexpr (kSplitMinor) {
                #pragma unroll 8
                for (uint32_t s = 0; s < kNumSplits; s += 4) {
                    const float4 q = *reinterpret_cast<const float4*>(
                        &ws_s[int64_t(row) * kNumSplits + s]);
                    a0 += q.x; a1 += q.y; a2 += q.z; a3 += q.w;
                }
            } else {
                #pragma unroll 8
                for (uint32_t s = 0; s < kNumSplits; s += 4) {
                    a0 += ws_s[int64_t(s) * num_tokens + row];
                    a1 += ws_s[int64_t(s + 1) * num_tokens + row];
                    a2 += ws_s[int64_t(s + 2) * num_tokens + row];
                    a3 += ws_s[int64_t(s + 3) * num_tokens + row];
                }
            }
        } else if constexpr (kNumSplits % 2 == 0) {
            #pragma unroll 4
            for (uint32_t s = 0; s < kNumSplits; s += 2) {
                a0 += ws_s[kSplitMinor ? int64_t(row) * kNumSplits + (s) : int64_t(s) * num_tokens + row];
                a1 += ws_s[kSplitMinor ? int64_t(row) * kNumSplits + (s + 1) : int64_t(s + 1) * num_tokens + row];
            }
        } else {
            #pragma unroll 4
            for (uint32_t s = 0; s < kNumSplits; ++s)
                a0 += ws_s[kSplitMinor ? int64_t(row) * kNumSplits + (s) : int64_t(s) * num_tokens + row];
        }
        sqrsum[row] = (a0 + a1) + (a2 + a3);
    }
#endif
}

template <uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t kNumSplits, uint32_t kNumThreads,
          bool kSplitMinor = false, bool kRotateSquareWork = false>
CUTLASS_DEVICE void reduce_splits(float* __restrict__ out,
                                  float* __restrict__ sqrsum,
                                  const float* __restrict__ ws,
                                  const float* __restrict__ ws_s,
                                  int* __restrict__ counter,
                                  const uint32_t m_base,
                                  const uint32_t num_tokens,
                                  const uint32_t tid) {
#if defined(__HGGC_ARCH__)
    __syncthreads();

    __shared__ uint32_t s_is_last;
    if (tid == 0) {
        const uint32_t prev = atom_add_acq_rel_gpu(
            reinterpret_cast<uint32_t*>(counter) + blockIdx.x, 1u);
        s_is_last = (prev % kNumSplits) == (kNumSplits - 1) ? 1u : 0u;
    }
    __syncthreads();

    if (s_is_last == 0u)
        return;
    reduce_splits_body<SHAPE_N, BLOCK_M, kNumSplits, kNumThreads, kSplitMinor, kRotateSquareWork>(
        out, sqrsum, ws, ws_s, m_base, num_tokens, tid);
#endif
}

template <bool kFastBF16ToTF32>
struct ToTF32 {
    CUTLASS_DEVICE cutlass::tfloat32_t operator()(const __ppu_bfloat16& value) const {
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

// M8192/BM256: each warp shares B across two complete M16 groups.
template <bool kRoundedB, class SmemA, class SmemB, class Accum>
CUTLASS_DEVICE void gemm_reuse_tc01(const uint32_t tid, SmemA const& sA, SmemB const& sB,
                                  Accum& accum0, Accum& accum1, float* sqlo, float* sqhi) {
    using namespace cute;
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    const int lane = tid & 31;
    const int row = (tid >> 5) * 16 + lane / 4;
    constexpr int K = decltype(size<1>(sA))::value;
    constexpr int MHalf = decltype(size<0>(sA))::value / 2;
    #pragma unroll 8
    for (int kk = 0; kk < K; kk += 8) {
        uint32_t av[2][4];
        CUTE_UNROLL
        for (int wm = 0; wm < 2; ++wm) {
            const uint32_t lo = *reinterpret_cast<const uint32_t*>(&sA(row + wm * MHalf, kk + (lane & 3) * 2));
            const uint32_t hi = *reinterpret_cast<const uint32_t*>(&sA(row + wm * MHalf + 8, kk + (lane & 3) * 2));
            const int src = (lane & 3) >> 1;
            const int shift = (lane & 1) * 16;
            av[wm][0] = ((__shfl_sync(0xffffffff, lo, src, 4) >> shift) & 0xffffu) << 16;
            av[wm][1] = ((__shfl_sync(0xffffffff, lo, src + 2, 4) >> shift) & 0xffffu) << 16;
            av[wm][2] = ((__shfl_sync(0xffffffff, hi, src, 4) >> shift) & 0xffffu) << 16;
            av[wm][3] = ((__shfl_sync(0xffffffff, hi, src + 2, 4) >> shift) & 0xffffu) << 16;
            sqlo[wm] = fmaf(__uint_as_float(av[wm][0]), __uint_as_float(av[wm][0]), sqlo[wm]);
            sqlo[wm] = fmaf(__uint_as_float(av[wm][1]), __uint_as_float(av[wm][1]), sqlo[wm]);
            sqhi[wm] = fmaf(__uint_as_float(av[wm][2]), __uint_as_float(av[wm][2]), sqhi[wm]);
            sqhi[wm] = fmaf(__uint_as_float(av[wm][3]), __uint_as_float(av[wm][3]), sqhi[wm]);
        }
        CUTE_UNROLL
        for (int nb = 0; nb < 2; ++nb) {
            uint32_t bv[4];
            CUTE_UNROLL
            for (int i = 0; i < 4; ++i)
                bv[i] = kRoundedB ?
                    __float_as_uint(sB(nb * 16 + lane / 4 + (i / 2) * 8, kk + (i % 2) * 4 + (lane & 3))) :
                    cutlass::NumericConverter<cutlass::tfloat32_t, float>::convert(
                        sB(nb * 16 + lane / 4 + (i / 2) * 8, kk + (i % 2) * 4 + (lane & 3))).raw();
            const int c = nb * 8;
            CUTE_UNROLL
            for (int wm = 0; wm < 2; ++wm) {
                auto& accum = wm == 0 ? accum0 : accum1;
                cute::PPU0010_16x16x8_F32TF32TF32F32_TN::fma(
                    accum(c), accum(c+1), accum(c+2), accum(c+3), accum(c+4), accum(c+5), accum(c+6), accum(c+7),
                    av[wm][0], av[wm][1], av[wm][2], av[wm][3], bv[0], bv[1], bv[2], bv[3],
                    accum(c), accum(c+1), accum(c+2), accum(c+3), accum(c+4), accum(c+5), accum(c+6), accum(c+7));
            }
        }
    }
#endif
}

template <bool kFastBF16ToTF32, bool kPackedLoads = false, bool kShuffleA = false, bool kRoundedB = false, class ThrMma, class SmemA, class SmemB, class Accum>
CUTLASS_DEVICE void gemm_explicit(ThrMma const& thr_mma,
                                  const uint32_t tid,
                                  SmemA const& sA,
                                  SmemB const& sB,
                                  Accum& accum,
                                  float& sqr_sum_acc_lo,
                                  float& sqr_sum_acc_hi) {
    using namespace cute;

#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    if constexpr (kPackedLoads) {
        // Packed TSM loads preserve the TC01 MMA layout.
        static_assert(sizeof(typename SmemA::value_type) == 2);
        static_assert(decltype(size<0>(sB))::value == 32);
        static_assert(decltype(size(accum))::value == 16);
        const int lane = tid & 31;
        const int row = (tid >> 5) * 16 + lane / 4;
        constexpr int K = decltype(size<1>(sA))::value;
        CUTE_UNROLL
        for (int kk = 0; kk < K; kk += 8) {
            uint32_t av[4];
            if constexpr (kShuffleA) {
                // Share four BF16 pairs within each quad.
                const uint32_t lo = *reinterpret_cast<const uint32_t*>(&sA(row, kk + (lane & 3) * 2));
                const uint32_t hi = *reinterpret_cast<const uint32_t*>(&sA(row + 8, kk + (lane & 3) * 2));
                const int src = (lane & 3) >> 1;
                const int shift = (lane & 1) * 16;
                av[0] = ((__shfl_sync(0xffffffff, lo, src, 4) >> shift) & 0xffffu) << 16;
                av[1] = ((__shfl_sync(0xffffffff, lo, src + 2, 4) >> shift) & 0xffffu) << 16;
                av[2] = ((__shfl_sync(0xffffffff, hi, src, 4) >> shift) & 0xffffu) << 16;
                av[3] = ((__shfl_sync(0xffffffff, hi, src + 2, 4) >> shift) & 0xffffu) << 16;
            } else {
                CUTE_UNROLL
                for (int i = 0; i < 4; ++i) {
                    const auto* ptr = &sA(row + (i / 2) * 8, kk + (i % 2) * 4 + (lane & 2));
                    const uint32_t pair = *reinterpret_cast<const uint32_t*>(ptr);
                    av[i] = ((pair >> ((lane & 1) * 16)) & 0xffffu) << 16;
                }
            }
            sqr_sum_acc_lo = fmaf(__uint_as_float(av[0]), __uint_as_float(av[0]), sqr_sum_acc_lo);
            sqr_sum_acc_lo = fmaf(__uint_as_float(av[1]), __uint_as_float(av[1]), sqr_sum_acc_lo);
            sqr_sum_acc_hi = fmaf(__uint_as_float(av[2]), __uint_as_float(av[2]), sqr_sum_acc_hi);
            sqr_sum_acc_hi = fmaf(__uint_as_float(av[3]), __uint_as_float(av[3]), sqr_sum_acc_hi);
            CUTE_UNROLL
            for (int nb = 0; nb < 2; ++nb) {
                uint32_t bv[4];
                CUTE_UNROLL
                for (int i = 0; i < 4; ++i)
                    bv[i] = kRoundedB ? __float_as_uint(sB(nb * 16 + lane / 4 + (i / 2) * 8, kk + (i % 2) * 4 + (lane & 3))) :
                        cutlass::NumericConverter<cutlass::tfloat32_t, float>::convert(
                        sB(nb * 16 + lane / 4 + (i / 2) * 8, kk + (i % 2) * 4 + (lane & 3))).raw();
                const int c = nb * 8;
                cute::PPU0010_16x16x8_F32TF32TF32F32_TN::fma(
                    accum(c), accum(c + 1), accum(c + 2), accum(c + 3), accum(c + 4), accum(c + 5), accum(c + 6), accum(c + 7),
                    av[0], av[1], av[2], av[3], bv[0], bv[1], bv[2], bv[3],
                    accum(c), accum(c + 1), accum(c + 2), accum(c + 3), accum(c + 4), accum(c + 5), accum(c + 6), accum(c + 7));
            }
        }
        return;
    }
#endif

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
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
            CUTE_UNROLL
            for (int i = 0; i < size(tCrAi_k); i += 2) {
                const float value_lo = __bfloat162float(tCrAi_k(i));
                const float value_hi = __bfloat162float(tCrAi_k(i + 1));
                sqr_sum_acc_lo = fmaf(value_lo, value_lo, sqr_sum_acc_lo);
                sqr_sum_acc_hi = fmaf(value_hi, value_hi, sqr_sum_acc_hi);
            }
#else
            constexpr int kHalf = decltype(size(tCrAi_k))::value / 2;
            CUTE_UNROLL
            for (int i = 0; i < kHalf; ++i) {
                const float value = __bfloat162float(tCrAi_k(i));
                sqr_sum_acc_lo = fmaf(value, value, sqr_sum_acc_lo);
            }
            CUTE_UNROLL
            for (int i = kHalf; i < size(tCrAi_k); ++i) {
                const float value = __bfloat162float(tCrAi_k(i));
                sqr_sum_acc_hi = fmaf(value, value, sqr_sum_acc_hi);
            }
#endif
            cute::transform(tCrAi(_, _, k_block), tCrA(_, _, k_block), ToTF32<true>{});
        } else {
            auto tCrAi_k = tCrAi(_, _, k_block);
            auto tCrA_k = tCrA(_, _, k_block);
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
            CUTE_UNROLL
            for (int i = 0; i < size(tCrAi_k); i += 2) {
                const float value_lo = __bfloat162float(tCrAi_k(i));
                const float value_hi = __bfloat162float(tCrAi_k(i + 1));
                sqr_sum_acc_lo = fmaf(value_lo, value_lo, sqr_sum_acc_lo);
                sqr_sum_acc_hi = fmaf(value_hi, value_hi, sqr_sum_acc_hi);
                tCrA_k(i) = cutlass::tfloat32_t(value_lo);
                tCrA_k(i + 1) = cutlass::tfloat32_t(value_hi);
            }
#else
            constexpr int kHalf = decltype(size(tCrAi_k))::value / 2;
            CUTE_UNROLL
            for (int i = 0; i < kHalf; ++i) {
                const float value = __bfloat162float(tCrAi_k(i));
                sqr_sum_acc_lo = fmaf(value, value, sqr_sum_acc_lo);
                tCrA_k(i) = cutlass::tfloat32_t(value);
            }
            CUTE_UNROLL
            for (int i = kHalf; i < size(tCrAi_k); ++i) {
                const float value = __bfloat162float(tCrAi_k(i));
                sqr_sum_acc_hi = fmaf(value, value, sqr_sum_acc_hi);
                tCrA_k(i) = cutlass::tfloat32_t(value);
            }
#endif
        }

        cute::transform(tCrBi(_, _, k_block), tCrB(_, _, k_block), ToTF32<kFastBF16ToTF32>{});

        using Atom = typename ThrMma::Atom;
        gemm(static_cast<Atom const&>(thr_mma), tCrA(_, _, k_block), tCrB(_, _, k_block), accum);
    }
}

} // namespace hc_detail

#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
namespace fused_890p {

using AiuLoadA = cute::PPU0015_AIU_LOAD<
    cute::C<64 * 64 * 16>, __ppu_bfloat16, false, 64, 64, true>;
using AiuLoadB = cute::PPU0015_AIU_LOAD<
    cute::C<32 * 32 * 32>, float, false, 32, 32, true>;

static constexpr int kA_BUF_BF16  = 4096;
static constexpr int kB_CUBE_H    = 32;
static constexpr int kB_CUBE_W    = 32;
static constexpr int kB_INST      = 2;
static constexpr int kB_BUF_F     = kB_CUBE_H * kB_CUBE_W * kB_INST;
static constexpr int kA_BYTES     = 2 * kA_BUF_BF16 * 2;
static constexpr int kB_BYTES     = 2 * kB_BUF_F * 4;
static constexpr int kSMEM_BYTES  = kA_BYTES + kB_BYTES;  // 32768

__device__ __forceinline__ void tsm_ld_swzl_b32(
    void *frag, void *smem, int ch, int cw, int cube, int stage) {
    float *base = reinterpret_cast<float*>(smem) + kB_CUBE_H*kB_CUBE_W*(cube+stage*kB_INST) + ch*kB_CUBE_W+cw;
    int tsm = reinterpret_cast<uintptr_t>(base) / 16;
    int *v = reinterpret_cast<int*>(frag);
    asm volatile(
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%0,%1,%2,%3}, [%4], %5, %6, %7;"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
        : "l"(tsm), "r"(64), "r"(1), "r"(0));
}

__device__ __forceinline__ void mma_tf32(float* d, const uint32_t* a, const uint32_t* b) {
    cute::PPU0015_16x16x8_F32TF32TF32F32_TN::fma(
        d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],
        a[0],a[1],a[2],a[3], b[0],b[1],b[2],b[3],
        d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7]);
}

__device__ __forceinline__ float reduce4(float x) {
    x += __shfl_xor_sync(0xffffffff, x, 2);
    x += __shfl_xor_sync(0xffffffff, x, 1);
    return x;
}

// Fetch two consecutive 32x32 FP32 B planes in one AIU request.
__device__ __forceinline__ void b_cube_pair_copy(
    void* smem_ptr, const void* gmem_ptr, const cute::AiuDesc& desc,
    const uint32_t kb) {
    const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    const uint64_t stride_w = uint64_t(desc.stride_w) * sizeof(float);
    const uint64_t stride_n = 32 * sizeof(float);
    const uint32_t dim_n = uint32_t(desc.dim_w) / 32;
    const uint32_t start_n = kb / 32;
    asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.LLC::128B.padz.swzl.b32"
        " [%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
        :: "r"(dst), "l"(gmem_ptr),
           "r"(32), "r"(desc.dim_h), "r"(dim_n),
           "r"(32), "r"(32), "r"(2),
           "l"(stride_w), "l"(stride_n),
           "r"(0), "r"(0), "r"(start_n), "r"(0));
}

// Only the cached-A specialization owns a full FP32 A fragment.
template <bool CachedA>
struct WarpReuseAStorage {};
template <>
struct WarpReuseAStorage<true> {
    float values[2][32];
};

// Reuse B across two M16 tiles; CachedA selects full-fragment A caching.
template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M,
          uint32_t BLOCK_K, uint32_t kNumSplits, bool CachedA>
CUTLASS_DEVICE void warp_reuse2_prenorm(
    const float* __restrict__ fn,
    float* __restrict__ out,
    float* __restrict__ sqrsum,
    const __ppu_bfloat16* __restrict__ x,
    const uint32_t num_tokens,
    float* __restrict__ ws,
    float* __restrict__ ws_s,
    int* __restrict__ counter,
    const uint32_t m_base,
    const uint32_t k_split_idx) {
    constexpr uint32_t kWarpMGroups = 2;
    DG_STATIC_ASSERT(SHAPE_N == 24, "Warp-reuse prenorm requires N=24");
    DG_STATIC_ASSERT(BLOCK_M >= 64 && BLOCK_M <= 256 && BLOCK_M % 32 == 0,
                     "Warp-reuse prenorm requires complete row groups and two issuer warps");
    DG_STATIC_ASSERT(BLOCK_K == 64 || BLOCK_K == 128, "Invalid warp-reuse block K");
    DG_STATIC_ASSERT(SHAPE_K % BLOCK_K == 0, "K must contain complete blocks");
    const uint32_t tid = threadIdx.x;

    extern __shared__ __align__(1024) unsigned char buf_smem[];
    // NOTES: `kKH` is how many 64-wide K sub-tiles one AIU stage carries
    constexpr int kKH      = BLOCK_K / 64;
    constexpr int kATileRows = BLOCK_M;
    constexpr int kATileElements = kATileRows * 64;
    constexpr int lA_BUF = kATileElements * kKH;
    using AiuLoadATile = cute::PPU0015_AIU_LOAD<cute::C<kATileRows*64*16>, __ppu_bfloat16, false, kATileRows, 64, true>;
    constexpr int lB_BUF   = kB_CUBE_H * kB_CUBE_W * 2 * kKH;
    constexpr int lA_BYTES = 2 * lA_BUF * 2;
    float* sB = reinterpret_cast<float*>(buf_smem + lA_BYTES);

    constexpr uint32_t kKBlocks = SHAPE_K / BLOCK_K;
    constexpr uint32_t kKPerSplit = kKBlocks / kNumSplits;
    DG_STATIC_ASSERT(kKBlocks % kNumSplits == 0, "K blocks must divide evenly across splits");
    DG_STATIC_ASSERT(kKPerSplit >= 2, "Each split needs at least two K blocks");
    constexpr uint32_t NUM_STEPS = kKPerSplit;
    const uint32_t kb0 = k_split_idx * kKPerSplit * BLOCK_K;

    float of[kWarpMGroups][16], sq4[kWarpMGroups][2];
    WarpReuseAStorage<CachedA> a_cache;
    #pragma unroll
    for (int wm = 0; wm < kWarpMGroups; ++wm) {
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            *reinterpret_cast<float4*>(of[wm]+i*4) = make_float4(0,0,0,0);
        sq4[wm][0] = sq4[wm][1] = 0.f;
    }

    cute::AiuDesc dA; dA.dim_h=num_tokens; dA.dim_w=SHAPE_K; dA.stride_w=SHAPE_K;
    cute::AiuDesc dB; dB.dim_h=SHAPE_N;    dB.dim_w=SHAPE_K; dB.stride_w=SHAPE_K;

    auto ldA=[&](uint32_t s,int buf){
        if(tid==0){
            #pragma unroll
            for(int h=0;h<kKH;++h) AiuLoadATile::copy(
                (void*)&(((__ppu_bfloat16*)buf_smem)[buf*lA_BUF+h*kATileElements]),
                (const void*)x, dA, kb0+s*BLOCK_K+h*64, m_base);
        }
    };
    auto ldB=[&](uint32_t s,int buf){
        if(tid==(BLOCK_M==16 ? 0 : 32)){
            uint32_t kb=kb0+s*BLOCK_K;
            #pragma unroll
            for(int h=0;h<kKH;++h)
                b_cube_pair_copy((void*)&sB[buf*lB_BUF+h*2*kB_CUBE_H*kB_CUBE_W],
                                 (const void*)fn, dB, kb+h*64);
        }
    };
    auto rdA=[&](int buf,int kh){
        if constexpr (CachedA) {
            const uint32_t lane = tid & 31;
            const uint32_t swizzle = (lane & 28) << 1;
            const auto* source = reinterpret_cast<const __ppu_bfloat16*>(buf_smem);
            #pragma unroll
            for (int wm = 0; wm < kWarpMGroups; ++wm) {
                const uint32_t logical_warp = (tid >> 5) * kWarpMGroups + wm;
                const uint32_t base = buf*lA_BUF + kh*kATileElements + logical_warp*1024
                                    + (lane>>2)*64 + (lane&3);
                // Complete each BF16 temporary's use before loading the next M group.
                alignas(8) __ppu_bfloat16 xh_local[32];
                #pragma unroll
                for (int i = 0; i < 32; ++i)
                    xh_local[i] = source[base + (((i&30)<<1)^swizzle) + ((i&1)<<9)];
                #pragma unroll
                for(int i=0;i<8;++i){
                    uint2 raw=*(uint2*)(xh_local+i*4); float4 v_;
                    ((float2*)(&v_))[0]=__bfloat1622float2((reinterpret_cast<__ppu_bfloat162*>(&raw))[0]);
                    ((float2*)(&v_))[1]=__bfloat1622float2((reinterpret_cast<__ppu_bfloat162*>(&raw))[1]);
                    *(float4*)(a_cache.values[wm]+i*4)=v_;
                }
                for(int j=0;j<16;++j){
                    #pragma unroll
                    for(int i=0;i<2;++i){ float v=a_cache.values[wm][j*2+i]; sq4[wm][i]+=v*v; }
                }
            }
        }
    };
    auto doGemm=[&](int buf,int kh){
        float* Bb = sB + buf*lB_BUF;
        const uint32_t lane = tid & 31;
        const uint32_t swizzle = (lane & 28) << 1;
        const auto* source = reinterpret_cast<const __ppu_bfloat16*>(buf_smem);
        for (int ki = 0; ki < 8; ++ki) {
            const int gki = kh * 8 + ki;
            float Bl[8];
            #pragma unroll
            for (int g = 0; g < 2; ++g)
                tsm_ld_swzl_b32(Bl + g*4, Bb, g*16, (gki%4)*8, gki/4, 0);
            uint32_t b_tf32[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                b_tf32[i] = cutlass::NumericConverter<cutlass::tfloat32_t, float>::convert(Bl[i]).raw();
            // Both row groups consume the same rounded B fragment.
            #pragma unroll
            for (int wm = 0; wm < kWarpMGroups; ++wm) {
                if constexpr (CachedA) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j)
                        mma_tf32(of[wm]+j*8, reinterpret_cast<const uint32_t*>(a_cache.values[wm]+ki*4), b_tf32+j*4);
                } else {
                    const uint32_t logical_warp = (tid >> 5) * kWarpMGroups + wm;
                    const uint32_t base = buf*lA_BUF + kh*kATileElements + logical_warp*1024
                                        + (lane>>2)*64 + (lane&3);
                    alignas(8) __ppu_bfloat16 a_half[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int i = ki*4+j;
                        a_half[j] = source[base + (((i&30)<<1)^swizzle) + ((i&1)<<9)];
                    }
                    uint2 raw = *reinterpret_cast<const uint2*>(a_half);
                    float a_frag[4];
                    reinterpret_cast<float2*>(a_frag)[0] = __bfloat1622float2(
                        reinterpret_cast<const __ppu_bfloat162*>(&raw)[0]);
                    reinterpret_cast<float2*>(a_frag)[1] = __bfloat1622float2(
                        reinterpret_cast<const __ppu_bfloat162*>(&raw)[1]);
                    // Each row sees the original square accumulation sequence.
                    sq4[wm][0] += a_frag[0] * a_frag[0];
                    sq4[wm][1] += a_frag[1] * a_frag[1];
                    sq4[wm][0] += a_frag[2] * a_frag[2];
                    sq4[wm][1] += a_frag[3] * a_frag[3];
                    #pragma unroll
                    for (int j = 0; j < 2; ++j)
                        mma_tf32(of[wm]+j*8, reinterpret_cast<const uint32_t*>(a_frag), b_tf32+j*4);
                }
            }
        }
    };

    constexpr uint32_t kStages = NUM_STEPS < 2 ? NUM_STEPS : 2;
    #pragma unroll
    for(uint32_t stage=0;stage<kStages;++stage){
        ldA(stage,stage);ldB(stage,stage);cp_async_fence();
    }
    for(uint32_t p=0;p<NUM_STEPS;++p){
        __ppu_sched_bound();
        if(p+kStages<=NUM_STEPS) cp_async_wait<kStages-1>();
        else cp_async_wait<0>();
        __syncthreads();
        #pragma unroll
        for(int kh=0;kh<kKH;++kh){rdA(p%kStages,kh);doGemm(p%kStages,kh);}
        if(p+kStages<NUM_STEPS){
            __syncthreads();
            ldA(p+kStages,p%kStages);ldB(p+kStages,p%kStages);cp_async_fence();
        }
    }

    constexpr int64_t kDstStride  = kNumSplits == 1 ? (int64_t)SHAPE_N : (int64_t)kNumSplits * SHAPE_N;
    constexpr int64_t kDstSStride = kNumSplits == 1 ? (int64_t)1 : (int64_t)kNumSplits;
    float* const dst   = (kNumSplits == 1 ? out    : ws)   + (int64_t)k_split_idx * SHAPE_N;
    float* const dst_s = (kNumSplits == 1 ? sqrsum : ws_s) + (int64_t)k_split_idx;

    // Each physical warp owns kWarpMGroups disjoint 16-row output tiles.
    #pragma unroll
    for (int wm = 0; wm < kWarpMGroups; ++wm) {
        const uint32_t row_group = m_base + ((tid>>5)*kWarpMGroups+wm)*16;
        // sqrsum reduce: unchanged four-lane tree for each logical row.
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float sum = reduce4(sq4[wm][i]);
            const uint32_t row = row_group + i*8 + ((tid&31)>>2);
            if ((tid%4)==0 && row<num_tokens)
                hc_detail::st_relaxed_gpu(&dst_s[int64_t(row)*kDstSStride], sum);
        }
        #pragma unroll
        for(int i=0;i<8;++i){
            const uint32_t row = row_group + (i&1)*8 + ((tid&31)>>2);
            const int c0 = (i>>1)*8 + (tid&3)*2;
            if (row<num_tokens && c0<int(SHAPE_N)) {
                const int64_t wo = int64_t(row)*kDstStride;
                hc_detail::st_relaxed_gpu_pair(&dst[wo+c0], of[wm][i*2], of[wm][i*2+1]);
            }
        }
    }

    if constexpr (kNumSplits > 1)
    {
        // Reuse the head of the (fully consumed) A buffer as the "am I last" flag
        __syncthreads();
        uint32_t* const s_flag = reinterpret_cast<uint32_t*>(buf_smem);
        if (tid == 0) {
            const uint32_t prev = hc_detail::atom_add_acq_rel_gpu(
                reinterpret_cast<uint32_t*>(counter) + blockIdx.x, 1u);
            *s_flag = (prev % kNumSplits) == (kNumSplits - 1) ? 1u : 0u;
        }
        __syncthreads();
        if (*s_flag != 0u)
            hc_detail::reduce_splits_body<SHAPE_N, BLOCK_M, kNumSplits, BLOCK_M,
                                         /*kSplitMinor=*/true, /*kRotateSquareWork=*/false>(
                out, sqrsum, ws, ws_s, m_base, num_tokens, tid);
    }

}

} // namespace fused_890p
#endif // __HGGC_ARCH__ >= 150

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumSplits,
          bool kFastBF16ToTF32,
          uint32_t kNumThreadsIn = 0, uint32_t kNumStages = 2, uint32_t kKernelVariant = 0>
CUTLASS_DEVICE void
tf32_hc_prenorm_gemm_device(
    const float* __restrict__ fn,
    float* __restrict__ out,
    float* __restrict__ sqrsum,
    const __ppu_bfloat16* __restrict__ x,
    const uint32_t num_tokens,
    float* __restrict__ ws,
    float* __restrict__ ws_s,
    int* __restrict__ counter) {

    const uint32_t m_base = blockIdx.x * BLOCK_M;
    const uint32_t k_split_idx = blockIdx.y;

#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
    // Bits 64/128 select streamed/cached A with paired B loads.
    if constexpr ((kKernelVariant & (64u | 128u)) != 0u) {
        DG_STATIC_ASSERT(kKernelVariant == (64u | 7u) || kKernelVariant == (128u | 7u),
                         "Select exactly one warp-reuse mode with legacy bits equal to 7");
        DG_STATIC_ASSERT(kNumThreadsIn == BLOCK_M, "Warp-reuse prenorm launches BLOCK_M threads");
        DG_STATIC_ASSERT(kNumStages == 2, "Warp-reuse prenorm uses two pipeline stages");
        fused_890p::warp_reuse2_prenorm<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_K,
                                      kNumSplits, (kKernelVariant & 128u) != 0u>(
            fn, out, sqrsum, x, num_tokens, ws, ws_s, counter, m_base, k_split_idx);
    } else {
    using namespace fused_890p;
    const uint32_t tid = threadIdx.x;
    constexpr uint32_t kReduceThreads = kNumThreadsIn == 0 ? BLOCK_M * 2 : kNumThreadsIn;
    constexpr bool kExtraWarps = kReduceThreads > BLOCK_M * 2;
    // Extra warps only synchronize and reduce.
    const bool is_mma_thread = !kExtraWarps || tid < BLOCK_M * 2;

    extern __shared__ __align__(1024) unsigned char buf_smem[];
    // NOTES: `kKH` is how many 64-wide K sub-tiles one AIU stage carries
    constexpr int kKH      = BLOCK_K / 64;
    // Bits: 1 exact A, 2 factored address, 4 unified loop, 8 skip padded warps.
    // 16 rotates square-sum work; 32 unrolls short loops.
    constexpr int kATileRows = (kKernelVariant & 1) ? BLOCK_M : 64;
    constexpr int kATileElements = kATileRows * 64;
    constexpr int lA_BUF = kATileElements * kKH;
    using AiuLoadATile = cute::PPU0015_AIU_LOAD<cute::C<kATileRows*64*16>, __ppu_bfloat16, false, kATileRows, 64, true>;
    constexpr int lB_BUF   = kB_CUBE_H * kB_CUBE_W * 2 * kKH;
    constexpr int lA_BYTES = 2 * lA_BUF * 2;
    float* sB = reinterpret_cast<float*>(buf_smem + lA_BYTES);

    constexpr uint32_t kKBlocks = SHAPE_K / BLOCK_K;
    constexpr uint32_t kKPerSplit = kKBlocks / kNumSplits;
    DG_STATIC_ASSERT(kKBlocks % kNumSplits == 0, "K blocks must divide evenly across splits");
    DG_STATIC_ASSERT(kKPerSplit >= 2, "Each split needs at least two K blocks");
    constexpr uint32_t NUM_STEPS = kKPerSplit;
    constexpr uint32_t NUM_MAIN = NUM_STEPS >= 2 ? NUM_STEPS - 2 : 0;
    const uint32_t kb0 = k_split_idx * kKPerSplit * BLOCK_K;

    float of[16], sq4[2], sq_l[2], x_f[32], Bl[8];
    __ppu_bfloat16 xh[32];
    #pragma unroll
    for (int i=0;i<4;++i) *(float4*)(of+i*4)=make_float4(0,0,0,0);
    sq4[0]=sq4[1]=0.f;

    cute::AiuDesc dA; dA.dim_h=num_tokens; dA.dim_w=SHAPE_K; dA.stride_w=SHAPE_K;
    cute::AiuDesc dB; dB.dim_h=SHAPE_N;    dB.dim_w=SHAPE_K; dB.stride_w=SHAPE_K;

    auto ldA=[&](uint32_t s,int buf){
        if(tid==0){
            #pragma unroll
            for(int h=0;h<kKH;++h) AiuLoadATile::copy(
                (void*)&(((__ppu_bfloat16*)buf_smem)[buf*lA_BUF+h*kATileElements]),
                (const void*)x, dA, kb0+s*BLOCK_K+h*64, m_base);
        }
    };
    auto ldB=[&](uint32_t s,int buf){
        if(tid==(BLOCK_M==16 ? 0 : 32)){
            uint32_t kb=kb0+s*BLOCK_K;
            #pragma unroll
            for(int c=0;c<2*kKH;++c)
                AiuLoadB::copy((void*)&sB[buf*lB_BUF+c*kB_CUBE_H*kB_CUBE_W],
                               (const void*)fn,dB,kb+c*kB_CUBE_W,0);
        }
    };
    auto rdA=[&](int buf,int kh){
        if constexpr (kKernelVariant & 2) {
            const uint32_t lane = tid & 31;
            const uint32_t base = buf*lA_BUF + kh*kATileElements + (tid>>5)*1024
                                + (lane>>2)*64 + (lane&3);
            const uint32_t swizzle = (lane&28)<<1;
            const auto* source = reinterpret_cast<const uint16_t*>(buf_smem);
            #pragma unroll
            for(int i=0;i<32;++i){
                const uint32_t index = base + (((i&30)<<1)^swizzle) + ((i&1)<<9);
                xh[i] = reinterpret_cast<const __ppu_bfloat16*>(source)[index];
            }
        } else {
            #pragma unroll
            for(int i=0;i<32;++i)
                xh[i]=((__ppu_bfloat16*)buf_smem)[((((((((((buf*lA_BUF+kh*kATileElements))+((((int)threadIdx.x)>>5)*1024))+((i&1)*512))+(((((int)threadIdx.x)&31)>>2)*64))+((((i>>4)+((((int)threadIdx.x)&31)>>4))&1)*32))+(((((i&15)>>3)+((((int)threadIdx.x)&15)>>3))&1)*16))+(((((i&7)>>2)+((((int)threadIdx.x)&7)>>2))&1)*8))+(((i&3)>>1)*4))+(((int)threadIdx.x)&3))];
        }
    };
    auto cvtSq=[&](){
        #pragma unroll
        for(int i=0;i<8;++i){
            uint2 raw=*(uint2*)(xh+i*4); float4 v_;
            ((float2*)(&v_))[0]=__bfloat1622float2((reinterpret_cast<__ppu_bfloat162*>(&raw))[0]);
            ((float2*)(&v_))[1]=__bfloat1622float2((reinterpret_cast<__ppu_bfloat162*>(&raw))[1]);
            *(float4*)(x_f+i*4)=v_;
        }
        for(int j=0;j<16;++j){
            #pragma unroll
            for(int i=0;i<2;++i){ float v=x_f[j*2+i]; sq4[i]+=v*v; }
        }
    };
    auto doGemm=[&](int buf,int kh){
        float* Bb=sB+buf*lB_BUF;
        float Bl[8];
        for(int ki=0;ki<8;++ki){
            const int gki=kh*8+ki;
            #pragma unroll
            for(int g=0;g<2;++g) tsm_ld_swzl_b32(Bl+g*4,Bb,g*16,(gki%4)*8,gki/4,0);
            // Feed round-to-nearest-even TF32 bits directly to MMA.
            uint32_t b_tf32[8];
            #pragma unroll
            for(int i=0;i<8;++i)
                b_tf32[i] = cutlass::NumericConverter<cutlass::tfloat32_t, float>::convert(Bl[i]).raw();
            for(int j=0;j<2;++j)
                mma_tf32(of+j*8, reinterpret_cast<const uint32_t*>(x_f+ki*4), b_tf32+j*4);
        }
    };

    if constexpr ((kKernelVariant & 32) && NUM_STEPS <= 8) {
    if constexpr (kKernelVariant & 4) {
        constexpr uint32_t kStages = NUM_STEPS < 2 ? NUM_STEPS : 2;
        #pragma unroll
        for(uint32_t stage=0;stage<kStages;++stage){
            ldA(stage,stage);ldB(stage,stage);cp_async_fence();
        }
        #pragma unroll
        for(uint32_t p=0;p<NUM_STEPS;++p){
            __ppu_sched_bound();
            if(p+kStages<=NUM_STEPS) cp_async_wait<kStages-1>();
            else cp_async_wait<0>();
            __syncthreads();
            #pragma unroll
            for(int kh=0;kh<kKH;++kh) if(is_mma_thread){rdA(p%kStages,kh);cvtSq();doGemm(p%kStages,kh);}
            if(p+kStages<NUM_STEPS){
                __syncthreads();
                ldA(p+kStages,p%kStages);ldB(p+kStages,p%kStages);cp_async_fence();
            }
        }

    } else {
        ldA(0,0);ldB(0,0); cp_async_fence();
        ldA(1,1);ldB(1,1); cp_async_fence();
        #pragma unroll
        for(uint32_t p=0;p<NUM_MAIN;++p){
            __ppu_sched_bound();
            cp_async_wait<1>();__syncthreads();
            #pragma unroll
            for(int kh=0;kh<kKH;++kh) if(is_mma_thread && (!(kKernelVariant & 8) || m_base+(tid>>5)*16<num_tokens)){ rdA(p&1,kh);cvtSq();doGemm(p&1,kh); }
            __syncthreads();
            ldA(p+2,p&1);ldB(p+2,p&1);cp_async_fence();
        }
        __ppu_sched_bound();
        cp_async_wait<1>();__syncthreads();
        #pragma unroll
        for(int kh=0;kh<kKH;++kh) if(is_mma_thread && (!(kKernelVariant & 8) || m_base+(tid>>5)*16<num_tokens)){ rdA(NUM_MAIN&1,kh);cvtSq();doGemm(NUM_MAIN&1,kh); }
        __ppu_sched_bound();
        cp_async_wait<0>();__syncthreads();
        #pragma unroll
        for(int kh=0;kh<kKH;++kh) if(is_mma_thread && (!(kKernelVariant & 8) || m_base+(tid>>5)*16<num_tokens)){ rdA((NUM_MAIN+1)&1,kh);cvtSq();doGemm((NUM_MAIN+1)&1,kh); }

    }

    } else {
    if constexpr (kKernelVariant & 4) {
        constexpr uint32_t kStages = NUM_STEPS < 2 ? NUM_STEPS : 2;
        #pragma unroll
        for(uint32_t stage=0;stage<kStages;++stage){
            ldA(stage,stage);ldB(stage,stage);cp_async_fence();
        }
        for(uint32_t p=0;p<NUM_STEPS;++p){
            __ppu_sched_bound();
            if(p+kStages<=NUM_STEPS) cp_async_wait<kStages-1>();
            else cp_async_wait<0>();
            __syncthreads();
            #pragma unroll
            for(int kh=0;kh<kKH;++kh) if(is_mma_thread){rdA(p%kStages,kh);cvtSq();doGemm(p%kStages,kh);}
            if(p+kStages<NUM_STEPS){
                __syncthreads();
                ldA(p+kStages,p%kStages);ldB(p+kStages,p%kStages);cp_async_fence();
            }
        }

    } else {
        ldA(0,0);ldB(0,0); cp_async_fence();
        ldA(1,1);ldB(1,1); cp_async_fence();
        for(uint32_t p=0;p<NUM_MAIN;++p){
            __ppu_sched_bound();
            cp_async_wait<1>();__syncthreads();
            #pragma unroll
            for(int kh=0;kh<kKH;++kh) if(is_mma_thread && (!(kKernelVariant & 8) || m_base+(tid>>5)*16<num_tokens)){ rdA(p&1,kh);cvtSq();doGemm(p&1,kh); }
            __syncthreads();
            ldA(p+2,p&1);ldB(p+2,p&1);cp_async_fence();
        }
        __ppu_sched_bound();
        cp_async_wait<1>();__syncthreads();
        #pragma unroll
        for(int kh=0;kh<kKH;++kh) if(is_mma_thread && (!(kKernelVariant & 8) || m_base+(tid>>5)*16<num_tokens)){ rdA(NUM_MAIN&1,kh);cvtSq();doGemm(NUM_MAIN&1,kh); }
        __ppu_sched_bound();
        cp_async_wait<0>();__syncthreads();
        #pragma unroll
        for(int kh=0;kh<kKH;++kh) if(is_mma_thread && (!(kKernelVariant & 8) || m_base+(tid>>5)*16<num_tokens)){ rdA((NUM_MAIN+1)&1,kh);cvtSq();doGemm((NUM_MAIN+1)&1,kh); }

    }

    }

    // Bit 22 selects split-major scratch for measured small-M shapes.
    constexpr bool kSplitMinor = (kKernelVariant & (1u << 22)) == 0;
    const int64_t kDstStride = kNumSplits == 1 || !kSplitMinor ? SHAPE_N : int64_t(kNumSplits) * SHAPE_N;
    const int64_t kDstSStride = kNumSplits == 1 || !kSplitMinor ? 1 : kNumSplits;
    const int64_t split_row = kSplitMinor ? k_split_idx : int64_t(k_split_idx) * num_tokens;
    float* const dst = (kNumSplits == 1 ? out : ws) + split_row * SHAPE_N;
    float* const dst_s = (kNumSplits == 1 ? sqrsum : ws_s) + split_row;

    // sqrsum reduce
    #pragma unroll
    for(int i=0;i<2;++i) sq_l[i]=reduce4(sq4[i]);
    if((tid%4)==0){
        #pragma unroll
        for(int i=0;i<2;++i){
            uint32_t row=m_base+(tid>>5)*16+i*8+((tid&31)>>2);
            if(is_mma_thread && row<num_tokens)
                hc_detail::st_relaxed_gpu(&dst_s[(int64_t)row*kDstSStride], sq_l[i]);
        }
    }
    // out epilogue
    #pragma unroll
    for(int i=0;i<8;++i){
        uint32_t row=m_base+(tid>>5)*16+(i&1)*8+((tid&31)>>2);
        int c0=(i>>1)*8+(tid&3)*2;
        if(is_mma_thread && row<num_tokens){
            const int64_t wo=(int64_t)row*kDstStride;
            // `c0` is even and SHAPE_N is 24, so `c0 < SHAPE_N` implies `c1 == c0 + 1 < SHAPE_N`
            if(c0<(int)SHAPE_N) hc_detail::st_relaxed_gpu_pair(&dst[wo+c0], of[i*2], of[i*2+1]);
        }
    }

    if constexpr (kNumSplits > 1)
    {
        // Reuse the head of the (fully consumed) A buffer as the "am I last" flag
        __syncthreads();
        uint32_t* const s_flag = reinterpret_cast<uint32_t*>(buf_smem);
        if (tid == 0) {
            const uint32_t prev = hc_detail::atom_add_acq_rel_gpu(
                reinterpret_cast<uint32_t*>(counter) + blockIdx.x, 1u);
            *s_flag = (prev % kNumSplits) == (kNumSplits - 1) ? 1u : 0u;
        }
        __syncthreads();
        if (*s_flag != 0u)
            hc_detail::reduce_splits_body<SHAPE_N, BLOCK_M, kNumSplits, kReduceThreads,
                                         /*kSplitMinor=*/kSplitMinor, /*kRotateSquareWork=*/(kKernelVariant & 16) != 0>(
                out, sqrsum, ws, ws_s, m_base, num_tokens, tid);
    }

    } // legacy 890P path

#elif defined(__HGGC_ARCH__)
    using namespace cute;

    // Only the measured M=8192 full-tile dispatch sets bit 4096.
    constexpr bool kReuse = (kKernelVariant & 4096u) != 0;
    static_assert(!kReuse || BLOCK_M % 32 == 0, "Warp reuse needs full M32 groups");
    constexpr uint32_t kMmaRows = BLOCK_M / (kReuse ? 2 : 1);
    constexpr uint32_t kMmaThreads = kMmaRows * 2;
    constexpr uint32_t kNumThreads = kNumThreadsIn == 0 ? kMmaThreads : kNumThreadsIn;

    DG_STATIC_ASSERT(BLOCK_M % 16 == 0 and BLOCK_M <= 256, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_K == 64 or BLOCK_K == 128, "Invalid block K");
    DG_STATIC_ASSERT(BLOCK_N % 8 == 0 and BLOCK_N <= 32, "Invalid block N");
    DG_STATIC_ASSERT(kNumThreads >= kMmaThreads and kNumThreads % 32 == 0 and kNumThreads <= 1024,
                     "Invalid number of threads");
    DG_STATIC_ASSERT(kNumStages >= 2 and kNumStages <= 8, "Invalid number of stages");
    DG_STATIC_ASSERT(SHAPE_N <= BLOCK_N, "Invalid shape N");
    DG_STATIC_ASSERT(SHAPE_K % BLOCK_K == 0, "Invalid shape K");
    DG_STATIC_ASSERT(BLOCK_K % 8 == 0, "Invalid block K for PPU TF32 MMA");

    constexpr int64_t stride_a_m = SHAPE_K;
    constexpr int64_t stride_b_n = SHAPE_K;
    constexpr int64_t stride_d_m = SHAPE_N;

    const auto* a = x;
    const float* b = fn;
    float* d = out;
    const uint32_t shape_m = num_tokens;
    const uint32_t tid = threadIdx.x;

    extern __shared__ __align__(16) uint8_t smem_buffer[];
    constexpr uint32_t kSmemAStride = BLOCK_K + 8;
    constexpr uint32_t kSmemABytes = BLOCK_M * kSmemAStride * sizeof(__ppu_bfloat16);
    constexpr uint32_t kSmemBStride = BLOCK_K + 4;
    constexpr uint32_t kSmemBBytes = BLOCK_N * kSmemBStride * sizeof(float);
    constexpr uint32_t kStageBytes = kSmemABytes + kSmemBBytes;

    constexpr uint32_t kNumKBlocks = hc_detail::ceil_div(SHAPE_K, BLOCK_K);
    constexpr uint32_t kNumKBlocksPerSplit = kNumKBlocks / kNumSplits;
    constexpr uint32_t kRemainKBlocks = kNumKBlocks % kNumSplits;

    const uint32_t k_offset = (k_split_idx * kNumKBlocksPerSplit +
                               (k_split_idx < kRemainKBlocks ? k_split_idx : kRemainKBlocks)) * BLOCK_K;
    const uint32_t num_total_stages = kNumKBlocksPerSplit + (k_split_idx < kRemainKBlocks);

    using MmaAtom = MMA_Atom<PPU0010_16x16x8_F32TF32TF32F32_TN>;
    using TiledMma = cute::TiledMMA<MmaAtom, Layout<Shape<Int<kMmaRows / 16>, _1, _1>>>;
    TiledMma tiled_mma;
    // Load-only threads use slice 0 but must not execute MMA or output stores.
    const bool is_mma_thread = tid < kMmaThreads;
    auto thr_mma = tiled_mma.get_thread_slice(is_mma_thread ? tid : 0);

    auto d_layout = make_layout(make_shape(Int<kMmaRows>{}, Int<BLOCK_N>{}),
                                make_stride(stride_d_m, int64_t(1)));
    auto d_tensor = make_tensor(make_gmem_ptr(d), d_layout);
    auto tdD = thr_mma.partition_C(d_tensor);
    auto accum = thr_mma.make_fragment_C(tdD);
    clear(accum);
    auto accum1 = thr_mma.make_fragment_C(tdD);
    clear(accum1);
    float sqlo[2] = {0.0f, 0.0f}, sqhi[2] = {0.0f, 0.0f};

    auto c_identity = make_identity_tensor(make_shape(Int<kMmaRows>{}, Int<BLOCK_N>{}));
    auto tcC = thr_mma.partition_C(c_identity);

    auto smem_a_layout = make_layout(make_shape(Int<BLOCK_M>{}, Int<BLOCK_K>{}),
                                     make_stride(Int<kSmemAStride>{}, Int<1>{}));
    auto smem_b_layout = make_layout(make_shape(Int<BLOCK_N>{}, Int<BLOCK_K>{}),
                                     make_stride(Int<kSmemBStride>{}, Int<1>{}));

    float sqr_sum_acc_lo = 0.0f;
    float sqr_sum_acc_hi = 0.0f;

    constexpr uint32_t kAElementsPerVec = sizeof(uint4) / sizeof(__ppu_bfloat16);
    constexpr uint32_t kAVectors = BLOCK_M * BLOCK_K / kAElementsPerVec;
    constexpr uint32_t kAVecsPerRow = BLOCK_K / kAElementsPerVec;
    constexpr uint32_t kBElementsPerVec = sizeof(uint4) / sizeof(float);
    constexpr uint32_t kBVectors = BLOCK_N * BLOCK_K / kBElementsPerVec;
    constexpr uint32_t kBVecsPerRow = BLOCK_K / kBElementsPerVec;

    const uint32_t rows_left = shape_m > m_base ? shape_m - m_base : 0u;
    const uint32_t rows_capped = rows_left < BLOCK_M ? rows_left : BLOCK_M;
    const uint32_t rows_tiled = ((rows_capped + 15u) / 16u) * 16u;
    const uint32_t a_vectors_raw = rows_tiled * kAVecsPerRow;
    const uint32_t a_vectors = a_vectors_raw < kAVectors ? a_vectors_raw : kAVectors;
    constexpr uint32_t b_vectors_raw = SHAPE_N * kBVecsPerRow;
    constexpr uint32_t b_vectors = b_vectors_raw < kBVectors ? b_vectors_raw : kBVectors;

    auto issue_cp_async_stage = [&](const uint32_t s, const uint32_t stage) {
        const uint32_t k_base = k_offset + s * BLOCK_K;
        auto* smem_a = reinterpret_cast<__ppu_bfloat16*>(smem_buffer + stage * kStageBytes);
        auto* smem_b = reinterpret_cast<float*>(smem_buffer + stage * kStageBytes + kSmemABytes);

        // Full TC01 tiles can use constant copy-loop bounds and unconditional row validity.
        if constexpr ((kKernelVariant & 2048u) != 0) {
            if (rows_left >= BLOCK_M) {
                #pragma unroll
                for (uint32_t idx = tid; idx < kAVectors; idx += kNumThreads) {
                    const uint32_t m = idx / kAVecsPerRow;
                    const uint32_t k = (idx % kAVecsPerRow) * kAElementsPerVec;
                    constexpr bool valid_m = true;
                    const auto* src = reinterpret_cast<const uint4*>(
                        a + (valid_m ? (m_base + m) : 0) * stride_a_m + k_base + k);
                    hc_detail::cp_async_16(smem_a + m * kSmemAStride + k, src, valid_m);
                }

            } else {
                for (uint32_t idx = tid; idx < a_vectors; idx += kNumThreads) {
                    const uint32_t m = idx / kAVecsPerRow;
                    const uint32_t k = (idx % kAVecsPerRow) * kAElementsPerVec;
                    const bool valid_m = m_base + m < shape_m;
                    const auto* src = reinterpret_cast<const uint4*>(
                        a + (valid_m ? (m_base + m) : 0) * stride_a_m + k_base + k);
                    hc_detail::cp_async_16(smem_a + m * kSmemAStride + k, src, valid_m);
                }

            }

            #pragma unroll
            for (uint32_t idx = tid; idx < b_vectors; idx += kNumThreads) {
                const uint32_t n = idx / kBVecsPerRow;
                const uint32_t k = (idx % kBVecsPerRow) * kBElementsPerVec;
                const auto* src = reinterpret_cast<const uint4*>(
                    b + n * stride_b_n + k_base + k);
                hc_detail::cp_async_16(smem_b + n * kSmemBStride + k, src, true);
            }
        } else {
            for (uint32_t idx = tid; idx < a_vectors; idx += kNumThreads) {
                const uint32_t m = idx / kAVecsPerRow;
                const uint32_t k = (idx % kAVecsPerRow) * kAElementsPerVec;
                const bool valid_m = m_base + m < shape_m;
                const auto* src = reinterpret_cast<const uint4*>(
                    a + (valid_m ? (m_base + m) : 0) * stride_a_m + k_base + k);
                hc_detail::cp_async_16(smem_a + m * kSmemAStride + k, src, valid_m);
            }

            for (uint32_t idx = tid; idx < b_vectors; idx += kNumThreads) {
                const uint32_t n = idx / kBVecsPerRow;
                const uint32_t k = (idx % kBVecsPerRow) * kBElementsPerVec;
                const auto* src = reinterpret_cast<const uint4*>(
                    b + n * stride_b_n + k_base + k);
                hc_detail::cp_async_16(smem_b + n * kSmemBStride + k, src, true);
            }
        }
        cp_async_fence();
    };

    #pragma unroll
    for (uint32_t p = 0; p + 1 < kNumStages; ++p) {
        if (p < num_total_stages)
            issue_cp_async_stage(p, p);
        else
            cp_async_fence();
    }

    const bool warp_has_rows = is_mma_thread and m_base + (tid >> 5) * 16 < shape_m;

    uint32_t stage = 0;
    uint32_t stage_prefetch = kNumStages - 1;

    #pragma unroll 1
    for (uint32_t s = 0; s < num_total_stages; ++s) {
        auto* smem_a = reinterpret_cast<__ppu_bfloat16*>(smem_buffer + stage * kStageBytes);
        auto* smem_b = reinterpret_cast<float*>(smem_buffer + stage * kStageBytes + kSmemABytes);

        // Prevent the compiler from hoisting the wait and serializing the pipeline.
        asm volatile("" ::: "memory");
        cp_async_wait<kNumStages - 2>();
        __syncthreads();

        const uint32_t s_prefetch = s + kNumStages - 1;
        if (s_prefetch < num_total_stages)
            issue_cp_async_stage(s_prefetch, stage_prefetch);
        else
            cp_async_fence();

        auto a_tensor = make_tensor(make_smem_ptr(smem_a), smem_a_layout);
        auto b_tensor = make_tensor(make_smem_ptr(smem_b), smem_b_layout);
        if constexpr (kReuse) {
            if (warp_has_rows)
                hc_detail::gemm_reuse_tc01<(kKernelVariant & 1024u) != 0>(
                    tid, a_tensor, b_tensor, accum, accum1, sqlo, sqhi);
        } else if (warp_has_rows)
            hc_detail::gemm_explicit<kFastBF16ToTF32, (kKernelVariant & 256u) != 0, (kKernelVariant & 512u) != 0, (kKernelVariant & 1024u) != 0>(
                thr_mma, tid, a_tensor, b_tensor, accum, sqr_sum_acc_lo, sqr_sum_acc_hi);
        __syncthreads();

        stage = stage + 1 == kNumStages ? 0u : stage + 1;
        stage_prefetch = stage_prefetch + 1 == kNumStages ? 0u : stage_prefetch + 1;
    }

    float* const dst   = (kNumSplits == 1 ? out    : ws)   + int64_t(k_split_idx) * shape_m * SHAPE_N;
    float* const dst_s = (kNumSplits == 1 ? sqrsum : ws_s) + int64_t(k_split_idx) * shape_m;

    // Load-only threads alias slice 0 and must not store.
    if constexpr (kReuse) {
        if (is_mma_thread) {
            CUTE_UNROLL
            for (int wm = 0; wm < 2; ++wm) {
                auto& acc = wm == 0 ? accum : accum1;
                CUTE_UNROLL
                for (uint32_t i = 0; i < size(acc); ++i) {
                    const auto coord = tcC(i);
                    const uint32_t m = get<0>(coord) + wm * kMmaRows;
                    const uint32_t n = get<1>(coord);
                    if (m_base + m < shape_m && n < SHAPE_N)
                        dst[int64_t(m_base + m) * stride_d_m + n] = acc(i);
                }
                const uint32_t lane = tid & 31;
                const uint32_t row = (tid >> 5) * 16 + lane / 4 + wm * kMmaRows;
                const float lo = hc_detail::quad_reduce_sum(sqlo[wm]);
                const float hi = hc_detail::quad_reduce_sum(sqhi[wm]);
                if ((lane & 3) == 0 && m_base + row < shape_m) dst_s[m_base + row] = lo;
                if ((lane & 3) == 0 && m_base + row + 8 < shape_m) dst_s[m_base + row + 8] = hi;
            }
        }
    } else
    if (is_mma_thread) {
        #pragma unroll
        for (uint32_t i = 0; i < size(accum); ++i) {
            const auto coord = tcC(i);
            const uint32_t m = get<0>(coord);
            const uint32_t n = get<1>(coord);
            if (m_base + m < shape_m and n < SHAPE_N) {
                dst[int64_t(m_base + m) * stride_d_m + n] = accum(i);
            }
        }

        const uint32_t lane_idx = tid & 31;
        const uint32_t warp_idx = tid >> 5;
        const uint32_t row_lo = warp_idx * 16 + lane_idx / 4;
        const uint32_t row_hi = row_lo + 8;
        const float sqr_sum_lo = hc_detail::quad_reduce_sum(sqr_sum_acc_lo);
        const float sqr_sum_hi = hc_detail::quad_reduce_sum(sqr_sum_acc_hi);

        if ((lane_idx & 3) == 0 and row_lo < BLOCK_M and m_base + row_lo < shape_m)
            dst_s[m_base + row_lo] = sqr_sum_lo;

        if ((lane_idx & 3) == 0 and row_hi < BLOCK_M and m_base + row_hi < shape_m)
            dst_s[m_base + row_hi] = sqr_sum_hi;
    }

    if constexpr (kNumSplits > 1)
        hc_detail::reduce_splits<SHAPE_N, BLOCK_M, kNumSplits, kNumThreads, false,
                                 (kKernelVariant & 1048576u) != 0>(
            out, sqrsum, ws, ws_s, counter, m_base, shape_m, tid);
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports PPU or newer");
#endif
}

// Kernel entry used by the Python JIT (`HcPrenormGemm::run`)
template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumSplits,
          bool kFastBF16ToTF32,
          uint32_t kNumThreads = BLOCK_M * 2, uint32_t kNumStages = 2, uint32_t kKernelVariant = 0>
CUTLASS_GLOBAL void
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
__launch_bounds__(kNumThreads, 1)
#else
__launch_bounds__(kNumThreads, 2)
#endif
tf32_hc_prenorm_gemm_impl(
    const float* __restrict__ fn,
    float* __restrict__ out,
    float* __restrict__ sqrsum,
    const __ppu_bfloat16* __restrict__ x,
    const uint32_t num_tokens,
    float* __restrict__ ws,
    float* __restrict__ ws_s,
    int* __restrict__ counter) {
    tf32_hc_prenorm_gemm_device<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K,
                                kNumSplits, kFastBF16ToTF32, kNumThreads, kNumStages, kKernelVariant>(
        fn, out, sqrsum, x, num_tokens, ws, ws_s, counter);
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumSplits,
          bool kFastBF16ToTF32,
          uint32_t kNumThreads = BLOCK_M * 2, uint32_t kNumStages = 2, uint32_t kKernelVariant = 0>
class HcPrenormGemm {
public:
    static void run(float* out,
                    float* sqr_sum,
                    uint32_t m,
                    const __ppu_bfloat16* lhs,
                    const float* rhs,
                    hggcStream_t stream,
                    int /*num_sms*/ = 0,
                    uint32_t /*smem_size_from_python*/ = 0,
                    float* ws = nullptr,
                    float* ws_s = nullptr,
                    int* counter = nullptr) {
        static_assert(BLOCK_N >= SHAPE_N, "BLOCK_N must >= SHAPE_N");
        static_assert(BLOCK_N <= 32, "BLOCK_N must <= 32 for PPU TF32 MMA");
        static_assert(BLOCK_K == 64 || BLOCK_K == 128, "BLOCK_K must be 64 or 128");

        constexpr uint32_t kSmemCuTe =
            kNumStages * (BLOCK_M * (BLOCK_K + 8) * sizeof(uint16_t) +
                          BLOCK_N * (BLOCK_K + 4) * sizeof(float));
        constexpr uint32_t kSmemFused = 2 * ((kKernelVariant & 1 ? BLOCK_M * 64 * 2 : 8192) + 8192) * (BLOCK_K / 64);
        constexpr uint32_t kSmemSize = kSmemCuTe > kSmemFused ? kSmemCuTe : kSmemFused;

        auto* kernel = tf32_hc_prenorm_gemm_impl<
            SHAPE_N, SHAPE_K,
            BLOCK_M, BLOCK_N, BLOCK_K,
            kNumSplits,
            kFastBF16ToTF32,
            kNumThreads, kNumStages, kKernelVariant>;

        hggcFuncSetAttribute(
            kernel,
            hggcFuncAttributeMaxDynamicSharedMemorySize,
            kSmemSize);

        const uint32_t grid_m = hc_detail::ceil_div(m, BLOCK_M);
        dim3 grid(grid_m, kNumSplits, 1);
        dim3 block(kNumThreads);

        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()) {
            dg_prof_params.set_params(
                GemmType::DenseGemm, false, std::string("tf32"),
                1, m, SHAPE_N, SHAPE_K, 1, nullptr, stream);
            dg_prof_params.add_params("num_splits", int(kNumSplits));
        }

        ProfilingInterface::Instance().instrument(true, dg_prof_params);

        kernel<<<grid, block, kSmemSize, stream>>>(
            rhs, out, sqr_sum, lhs, m, ws, ws_s, counter);

        ProfilingInterface::Instance().instrument(false, dg_prof_params);
    }
};

} // namespace deep_gemm

#pragma clang diagnostic pop
