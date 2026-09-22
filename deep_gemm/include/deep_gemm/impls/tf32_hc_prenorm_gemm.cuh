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

// NOTES: the reduction body is split out so the 890P path can supply its own flag slot.
// `kSplitMinor` picks the workspace layout: false is `ws[split][row][col]`, true `ws[row][split][col]`
template <uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t kNumSplits, uint32_t kNumThreads,
          bool kSplitMinor = false>
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

    for (uint32_t r = tid; r < num_rows; r += kNumThreads) {
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
          bool kSplitMinor = false>
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
    reduce_splits_body<SHAPE_N, BLOCK_M, kNumSplits, kNumThreads, kSplitMinor>(
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

} // namespace fused_890p
#endif // __HGGC_ARCH__ >= 150

// ============================================================
// Kernel body (890P: AIU  | 810E: CuTe cp.async)
// ============================================================
// NOTES: the body lives in a `__device__` function so that both JIT flavours can share it:
// the Python JIT launches `tf32_hc_prenorm_gemm_impl` below via `HcPrenormGemm::run`, while the
// C++ JIT emits its own `extern "C" __global__` entry that forwards here.
// NOTES: `kNumThreadsIn == 0` keeps the historical `BLOCK_M * 2` block size, and `kNumStages == 2`
// the historical double buffer; both only take effect on the 810E CuTe path
template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumSplits,
          bool kFastBF16ToTF32,
          uint32_t kNumThreadsIn = 0, uint32_t kNumStages = 2>
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
    using namespace fused_890p;
    const uint32_t tid = threadIdx.x;

    extern __shared__ __align__(1024) unsigned char buf_smem[];
    // NOTES: `kKH` is how many 64-wide K sub-tiles one AIU stage carries
    constexpr int kKH      = BLOCK_K / 64;
    constexpr int lA_BUF   = kA_BUF_BF16 * kKH;
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
            for(int h=0;h<kKH;++h) AiuLoadA::copy(
                (void*)&(((__ppu_bfloat16*)buf_smem)[buf*lA_BUF+h*kA_BUF_BF16]),
                (const void*)x, dA, kb0+s*BLOCK_K+h*64, m_base);
        }
    };
    auto ldB=[&](uint32_t s,int buf){
        if(tid==32){
            uint32_t kb=kb0+s*BLOCK_K;
            #pragma unroll
            for(int c=0;c<2*kKH;++c)
                AiuLoadB::copy((void*)&sB[buf*lB_BUF+c*kB_CUBE_H*kB_CUBE_W],
                               (const void*)fn,dB,kb+c*kB_CUBE_W,0);
        }
    };
    auto rdA=[&](int buf,int kh){
        #pragma unroll
        for(int i=0;i<32;++i)
            xh[i]=((__ppu_bfloat16*)buf_smem)[((((((((((buf*lA_BUF+kh*kA_BUF_BF16))+((((int)threadIdx.x)>>5)*1024))+((i&1)*512))+(((((int)threadIdx.x)&31)>>2)*64))+((((i>>4)+((((int)threadIdx.x)&31)>>4))&1)*32))+(((((i&15)>>3)+((((int)threadIdx.x)&15)>>3))&1)*16))+(((((i&7)>>2)+((((int)threadIdx.x)&7)>>2))&1)*8))+(((i&3)>>1)*4))+(((int)threadIdx.x)&3))];
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
            // Match the generic path's FP32-to-TF32 rounding before MMA.
            // Passing raw FP32 bits instead silently truncates the weights.
            #pragma unroll
            for(int i=0;i<8;++i) Bl[i] = float(cutlass::tfloat32_t(Bl[i]));
            for(int j=0;j<2;++j)
                mma_tf32(of+j*8, reinterpret_cast<const uint32_t*>(x_f+ki*4),
                                 reinterpret_cast<const uint32_t*>(Bl+j*4));
        }
    };

    ldA(0,0);ldB(0,0); cp_async_fence();
    ldA(1,1);ldB(1,1); cp_async_fence();
    for(uint32_t p=0;p<NUM_MAIN;++p){
        __ppu_sched_bound();
        cp_async_wait<1>();__syncthreads();
        #pragma unroll
        for(int kh=0;kh<kKH;++kh){ rdA(p&1,kh);cvtSq();doGemm(p&1,kh); }
        __syncthreads();
        ldA(p+2,p&1);ldB(p+2,p&1);cp_async_fence();
    }
    __ppu_sched_bound();
    cp_async_wait<1>();__syncthreads();
    #pragma unroll
    for(int kh=0;kh<kKH;++kh){ rdA(NUM_MAIN&1,kh);cvtSq();doGemm(NUM_MAIN&1,kh); }
    __ppu_sched_bound();
    cp_async_wait<0>();__syncthreads();
    #pragma unroll
    for(int kh=0;kh<kKH;++kh){ rdA((NUM_MAIN+1)&1,kh);cvtSq();doGemm((NUM_MAIN+1)&1,kh); }

    constexpr int64_t kDstStride  = kNumSplits == 1 ? (int64_t)SHAPE_N : (int64_t)kNumSplits * SHAPE_N;
    constexpr int64_t kDstSStride = kNumSplits == 1 ? (int64_t)1 : (int64_t)kNumSplits;
    float* const dst   = (kNumSplits == 1 ? out    : ws)   + (int64_t)k_split_idx * SHAPE_N;
    float* const dst_s = (kNumSplits == 1 ? sqrsum : ws_s) + (int64_t)k_split_idx;

    // sqrsum reduce
    #pragma unroll
    for(int i=0;i<2;++i) sq_l[i]=reduce4(sq4[i]);
    if((tid%4)==0){
        #pragma unroll
        for(int i=0;i<2;++i){
            uint32_t row=m_base+(tid>>5)*16+i*8+((tid&31)>>2);
            if(row<num_tokens)
                hc_detail::st_relaxed_gpu(&dst_s[(int64_t)row*kDstSStride], sq_l[i]);
        }
    }
    // out epilogue
    #pragma unroll
    for(int i=0;i<8;++i){
        uint32_t row=m_base+(tid>>5)*16+(i&1)*8+((tid&31)>>2);
        int c0=(i>>1)*8+(tid&3)*2;
        if(row<num_tokens){
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
            hc_detail::reduce_splits_body<SHAPE_N, BLOCK_M, kNumSplits, BLOCK_M * 2,
                                         /*kSplitMinor=*/true>(
                out, sqrsum, ws, ws_s, m_base, num_tokens, tid);
    }

#elif defined(__HGGC_ARCH__)
    using namespace cute;

    constexpr uint32_t kMmaThreads = BLOCK_M * 2;
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
    using TiledMma = cute::TiledMMA<MmaAtom, Layout<Shape<Int<BLOCK_M / 16>, _1, _1>>>;
    TiledMma tiled_mma;
    // NOTES: the load-only threads take slice 0's layouts so every partition below stays in bounds;
    // `is_mma_thread` fences them off from the MMA and from the epilogue stores
    const bool is_mma_thread = tid < kMmaThreads;
    auto thr_mma = tiled_mma.get_thread_slice(is_mma_thread ? tid : 0);

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

        // NOTES: without this barrier the compiler hoists the `s.wait commit_group` above the address
        // maths and the pipeline collapses back to a single stage
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
        if (warp_has_rows)
            hc_detail::gemm_explicit<kFastBF16ToTF32>(
                thr_mma, tid, a_tensor, b_tensor, accum, sqr_sum_acc_lo, sqr_sum_acc_hi);
        __syncthreads();

        stage = stage + 1 == kNumStages ? 0u : stage + 1;
        stage_prefetch = stage_prefetch + 1 == kNumStages ? 0u : stage_prefetch + 1;
    }

    float* const dst   = (kNumSplits == 1 ? out    : ws)   + int64_t(k_split_idx) * shape_m * SHAPE_N;
    float* const dst_s = (kNumSplits == 1 ? sqrsum : ws_s) + int64_t(k_split_idx) * shape_m;

    // NOTES: the load-only threads never ran the MMA, so their `accum` is still zero and their `tcC`
    // coordinates alias thread 0's -- they must not store
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
        hc_detail::reduce_splits<SHAPE_N, BLOCK_M, kNumSplits, kNumThreads>(
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
          uint32_t kNumThreads = BLOCK_M * 2, uint32_t kNumStages = 2>
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
                                kNumSplits, kFastBF16ToTF32, kNumThreads, kNumStages>(
        fn, out, sqrsum, x, num_tokens, ws, ws_s, counter);
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumSplits,
          bool kFastBF16ToTF32,
          uint32_t kNumThreads = BLOCK_M * 2, uint32_t kNumStages = 2>
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
        constexpr uint32_t kSmemFused = 32768u;
        constexpr uint32_t kSmemSize = kSmemCuTe > kSmemFused ? kSmemCuTe : kSmemFused;

        auto* kernel = tf32_hc_prenorm_gemm_impl<
            SHAPE_N, SHAPE_K,
            BLOCK_M, BLOCK_N, BLOCK_K,
            kNumSplits,
            kFastBF16ToTF32,
            kNumThreads, kNumStages>;

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
