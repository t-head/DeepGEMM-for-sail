#pragma once
#include "ppu_include.hpp"
#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/utils_cutlass3.cuh>
#include <deep_gemm/common/profiling_interface.cuh>

// ============================================================
// Common device helpers shared across MQA logits kernels
// Defined at global scope (same as original per-file definitions)
// ============================================================

__forceinline__ __device__ uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm("ppu.mov.u32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

__device__ __forceinline__ float ld_shared(const float* ptr) {
    float ret;
    asm volatile("ppu.ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

__device__ __inline__ uint32_t ld_shared_u32(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ppu.ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

namespace deep_gemm {

// ============================================================
// Scale mode: selects what the kernel's `weights` argument slot carries
// and how it is applied in the epilogue
// ============================================================

// kScaleModeWeights: per-(q, head) weights, logits = sum_h(relu(qk) * w) * k_scale
// kScaleModeQRow:    avg variant, per-Q-row float scale in the weights slot,
//                    logits = sum_h(relu(qk)) * scale_q * k_scale / sqrt(head_dim)
// kScaleModeQScalar: avg variant, broadcast scalar float scale in the weights slot
//                    (same formula, scale_q shared by every Q row)
// kScaleModeUnity:   avg variant without any Q-row scale (scale_q = 1)
enum QScaleMode : uint32_t {
    kScaleModeWeights = 0,
    kScaleModeQRow = 1,
    kScaleModeQScalar = 2,
    kScaleModeUnity = 3,
};

// ============================================================
// Non-paged block scheduler helpers
// ============================================================

template <int BLOCK_Q, int kNumQStages, int BLOCK_KV, bool kIsCompressedLogits,
          typename SeqLenType, typename CuSeqLenPtrType>
__forceinline__ __device__
cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
load_schedule(uint32_t block_q_idx, uint32_t q_iter_idx, uint32_t q_iter_offset,
              SeqLenType seq_len_q, SeqLenType seq_len_k,
              CuSeqLenPtrType cu_seq_len_k_start, CuSeqLenPtrType cu_seq_len_k_end,
              uint32_t* seq_k_start, uint32_t* seq_k_end) {
    uint32_t start = cute::numeric_limits<uint32_t>::max();
    uint32_t end = cute::numeric_limits<uint32_t>::min();
    #pragma unroll
    for (uint32_t i = 0; i < BLOCK_Q; ++i) {
        const auto& q_idx = min(block_q_idx * BLOCK_Q + i, static_cast<uint32_t>(seq_len_q) - 1);
        if constexpr (kIsCompressedLogits) {
            seq_k_start[i] = static_cast<uint32_t>(min(__ldg(cu_seq_len_k_start + q_idx), seq_len_k));
            seq_k_end[i]   = static_cast<uint32_t>(min(__ldg(cu_seq_len_k_end + q_idx), seq_len_k));
            start = min(start, seq_k_start[i]);
            end   = max(end,   seq_k_end[i]);
        } else {
            start = min(start, static_cast<uint32_t>(min(__ldg(cu_seq_len_k_start + q_idx), seq_len_k)));
            end = max(end, static_cast<uint32_t>(min(__ldg(cu_seq_len_k_end + q_idx), seq_len_k)));
        }
    }
    start = start / 4 * 4;
    return {(q_iter_idx + q_iter_offset) % kNumQStages,
            ((q_iter_idx + q_iter_offset) / kNumQStages) & 1,
            start, end, ceil_div(end - start, static_cast<uint32_t>(BLOCK_KV))};
}

__forceinline__ __device__
cute::tuple<uint32_t, uint32_t>
get_next_block_q_idx(uint32_t block_q_idx, uint32_t q_iter_idx) {
    return {block_q_idx + gridDim.x, q_iter_idx + 1};
}

// ============================================================
// Weights s2r: load from smem copy view to weights register array
// Used by fp4 kernels (SmemTiledCopyWeights approach)
// ============================================================

// For SmemTiledCopyWeights approach (fp4 kernels):
// - bf16 weights or float logits: direct copy
// - float weights + bf16 logits: cvt f32 pairs to bf16x2
template <typename ElementWeights, typename ElementLogits,
          typename TCrWCopyView, typename TWeights>
__forceinline__ __device__ void load_weights_from_copy_view(
        TCrWCopyView& tCrW_copy_view, TWeights* weights, int elem_weights) {
    if constexpr (cute::is_same_v<ElementWeights, __ppu_bfloat16> ||
                  cute::is_same_v<ElementLogits, float>) {
        for (int j = 0; j < elem_weights; j++) {
            weights[j] = tCrW_copy_view(j);
        }
    } else {
        for (int j = 0; j < elem_weights; j += 2) {
            uint32_t d;
            asm volatile("ppu.cvt.rtte.bf16x2.f32 %0, %1, %2;\n"
                         : "=r"(d)
                         : "f"(tCrW_copy_view(j + 1)), "f"(tCrW_copy_view(j)));
            *reinterpret_cast<uint32_t*>(&weights[j]) = d;
        }
    }
}

// For ld_shared approach (int8/bf16/fp8 kernels):
// - bf16 weights: read as uint32_t pairs from smem
// - float weights + float logits: read as float from smem
// - float weights + bf16 logits: read as float, cvt to bf16x2
template <typename ElementWeights, typename ElementLogits, uint32_t kNumHeads, typename TSmemSlice>
__forceinline__ __device__ void load_weights_from_smem_ld_shared(
        const TSmemSlice& sSFB_slice, ElementLogits* weights,
        uint32_t lane_idx, uint32_t warp_q_idx) {
    constexpr uint32_t elem_weights = kNumHeads / 4;
    if constexpr (cute::is_same_v<ElementWeights, __ppu_bfloat16>) {
        // BF16 weights: read 2 BF16 at a time as uint32_t from smem
        uint32_t* smem_w_u32 = reinterpret_cast<uint32_t*>(sSFB_slice.data().get());
        smem_w_u32 += warp_q_idx * (kNumHeads / 2);
#if __HGGC_ARCH__ == 150
        #pragma unroll
        for (uint32_t j = 0; j < elem_weights; j += 2) {
            uint32_t off = ((j / 2) * 8 + (lane_idx % 4) * 2) / 2;
            uint32_t packed = ld_shared_u32(smem_w_u32 + off);
            *reinterpret_cast<uint32_t*>(&weights[j]) = packed;
        }
#else
        __ppu_bfloat16* smem_w = reinterpret_cast<__ppu_bfloat16*>(smem_w_u32);
        #pragma unroll
        for (uint32_t j = 0; j < elem_weights; ++j) {
            uint32_t off = (j / 2) * 8 + (j & 1) * 4 + lane_idx % 4;
            weights[j] = smem_w[off];
        }
#endif
    } else if constexpr (cute::is_same_v<ElementLogits, float>) {
        float* smem_weights_staged = sSFB_slice.data().get();
        smem_weights_staged += warp_q_idx * kNumHeads;
        #pragma unroll
        for (uint32_t j = 0; j < elem_weights; ++j) {
#if __HGGC_ARCH__ == 150
            weights[j] = ld_shared(smem_weights_staged + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
#else
            weights[j] = ld_shared(smem_weights_staged + (j / 2) * 8 + (j & 1) * 4 + lane_idx % 4);
#endif
        }
    } else {
        float* smem_weights_staged = sSFB_slice.data().get();
        smem_weights_staged += warp_q_idx * kNumHeads;
        #pragma unroll
        for (uint32_t j = 0; j < elem_weights; j += 2) {
            float w0, w1;
#if __HGGC_ARCH__ == 150
            w0 = ld_shared(smem_weights_staged + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
            w1 = ld_shared(smem_weights_staged + ((j + 1) / 2) * 8 + ((j + 1) & 1) + (lane_idx % 4) * 2);
#else
            w0 = ld_shared(smem_weights_staged + (j / 2) * 8 + (j & 1) * 4 + lane_idx % 4);
            w1 = ld_shared(smem_weights_staged + ((j + 1) / 2) * 8 + ((j + 1) & 1) * 4 + lane_idx % 4);
#endif
            uint32_t d;
            asm volatile("ppu.cvt.rtte.bf16x2.f32 %0, %1, %2;\n"
                         : "=r"(d)
                         : "f"(w1), "f"(w0));
            *reinterpret_cast<uint32_t*>(&weights[j]) = d;
        }
    }
}

// ============================================================
// Epilogue common patterns (shared across all 4 MQA logits kernels)
// ============================================================

// Float inter-thread reduction: 2 rounds of shfl_xor_sync
__forceinline__ __device__
void shfl_xor_reduce_2(float& v_0, float& v_1) {
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        const auto offset = static_cast<int>(1u << j);
        v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
        v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
    }
}

// BF16 packed inter-thread reduction: pack v0/v1 into bf162, 2 rounds of shfl_xor_sync
__forceinline__ __device__
__ppu_bfloat162 shfl_xor_reduce_bf16x2(__ppu_bfloat162 packed) {
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        uint32_t bits = reinterpret_cast<uint32_t&>(packed);
        uint32_t received_bits = __shfl_xor_sync(0xffffffffu, bits, 1u << j);
        __ppu_bfloat162 received = reinterpret_cast<__ppu_bfloat162&>(received_bits);
        packed = __hadd2(packed, received);
    }
    return packed;
}

// Cvt phase: convert accum to bf16x2 buffer
template <int kTotalTransforms, typename AccTensor>
__forceinline__ __device__
void cvt_accum_to_bf16x2_buf(const AccTensor& accum, int m,
                              uint32_t (&cvt_buf)[kTotalTransforms]) {
    #pragma unroll
    for (int idx = 0; idx < kTotalTransforms; idx++) {
        int n = idx / 4;
        int sub = idx % 4;
        int j = sub * 2;
        asm volatile("ppu.cvt.rtte.bf16x2.f32.relu %0, %1, %2;\n"
                     : "=r"(cvt_buf[idx])
                     : "f"((float)accum(j + 1, m, n)), "f"((float)accum(j, m, n)));
    }
}

// FP4 float epilogue reduce: weight access via tCrW tensor (no arch branching)
template <typename AccTensor, typename TCrW>
__forceinline__ __device__
void float_epilogue_reduce_tCrW(const AccTensor& accum, int m,
                                 const TCrW& tCrW,
                                 float& v_0, float& v_1) {
    auto transform = [&](int j, int n) {
        return fmaxf(accum(j, m, n), 0) * tCrW(j, n);
    };
    v_0 = 0; v_1 = 0;
    #pragma unroll
    for (int n = 0; n < cute::size<2>(accum); n++) {
        v_0 += transform(0, n);
        v_0 += transform(1, n);
        v_1 += transform(2, n);
        v_1 += transform(3, n);
        v_0 += transform(4, n);
        v_0 += transform(5, n);
        v_1 += transform(6, n);
        v_1 += transform(7, n);
    }
}

// Non-FP4 float epilogue reduce: weight access via weights[] array with arch branching
template <typename AccTensor, typename ElementLogits>
__forceinline__ __device__
void float_epilogue_reduce_weights(const AccTensor& accum, int m,
                                    ElementLogits* weights,
                                    float& v_0, float& v_1) {
    auto transform = [&](uint32_t j, uint32_t n) {
#if __HGGC_ARCH__ == 150
        return fmaxf(accum(j, m, n), 0) * weights[n * 4 + (j / 4) * 2 + (j & 1)];
#else
        return fmaxf(accum(j, m, n), 0) * weights[n * 4 + j % 4];
#endif
    };
    v_0 = 0; v_1 = 0;
    #pragma unroll
    for (uint32_t n = 0; n < cute::size<2>(accum); ++n) {
#if __HGGC_ARCH__ == 150
        v_0 += transform(0, n);
        v_0 += transform(1, n);
        v_1 += transform(2, n);
        v_1 += transform(3, n);
        v_0 += transform(4, n);
        v_0 += transform(5, n);
        v_1 += transform(6, n);
        v_1 += transform(7, n);
#else
        v_0 += transform(0, n);
        v_0 += transform(1, n);
        v_0 += transform(2, n);
        v_0 += transform(3, n);
        v_1 += transform(4, n);
        v_1 += transform(5, n);
        v_1 += transform(6, n);
        v_1 += transform(7, n);
#endif
    }
}

// Avg (unweighted) epilogue reduce: plain ReLU sum over heads, same j->v_0/v_1
// grouping as `float_epilogue_reduce_weights` (only the weights multiply is dropped)
template <typename AccTensor>
__forceinline__ __device__
void float_epilogue_reduce_avg(const AccTensor& accum, int m,
                               float& v_0, float& v_1) {
    auto transform = [&](uint32_t j, uint32_t n) {
        return fmaxf(accum(j, m, n), 0);
    };
    v_0 = 0; v_1 = 0;
    #pragma unroll
    for (uint32_t n = 0; n < cute::size<2>(accum); ++n) {
#if __HGGC_ARCH__ == 150
        v_0 += transform(0, n);
        v_0 += transform(1, n);
        v_1 += transform(2, n);
        v_1 += transform(3, n);
        v_0 += transform(4, n);
        v_0 += transform(5, n);
        v_1 += transform(6, n);
        v_1 += transform(7, n);
#else
        v_0 += transform(0, n);
        v_0 += transform(1, n);
        v_1 += transform(2, n);
        v_1 += transform(3, n);
        v_1 += transform(4, n);
        v_1 += transform(5, n);
        v_0 += transform(6, n);
        v_0 += transform(7, n);
#endif
    }
}

// ============================================================
// WARP_Q == 4 (4 heads) epilogue helpers: a warp covers 4 q tokens x 4 heads
// (WARP_QH = 16). C-fragment mapping per docs/mma/ppu_mma_thread_distribution.html.
// Arch 150 only: PPU 1.0 has no fp8 MMA, so the 4-head fp8 tiles that instantiate
// WARP_Q == 4 never compile there. Avg-only (the host rejects weighted H = 4).
// ============================================================

// 4-head in-thread head sums (requires MmaIterN == 1, accum(j, m, 0)):
// lane T holds heads {2*(T&1), 2*(T&1)+1} of q_{T/2} (j 0-3) and q_{2+T/2} (j 4-7).
// sums[t][r] is the row-G/row-(G+8) half of warp-local q token 2*t + T/2.
template <typename AccTensor>
__forceinline__ __device__
void float_epilogue_reduce_wq4(const AccTensor& accum, int m,
                               float (&sums)[2][2]) {
    auto transform = [&](int j) {
        return fmaxf(accum(j, m, 0), 0);
    };
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        sums[0][r] = transform(2 * r) + transform(2 * r + 1);         // q_{T/2}
        sums[1][r] = transform(4 + 2 * r) + transform(5 + 2 * r);     // q_{2+T/2}
    }
}

// Cross-lane merge for the 4-head layout: a single xor round suffices —
// lane T^1 holds the complementary head pair of the same q tokens
__forceinline__ __device__
void shfl_xor_reduce_wq4(float (&sums)[2][2]) {
    #pragma unroll
    for (int t = 0; t < 2; ++t)
        #pragma unroll
        for (int r = 0; r < 2; ++r)
            sums[t][r] += __shfl_xor_sync(0xffffffffu, sums[t][r], 1);
}

// FP4 fma2 phase: weight access via tCrW tensor (no arch branching)
template <int kTotalTransforms, typename TCrW>
__forceinline__ __device__
void fma2_phase_tCrW(uint32_t (&cvt_buf)[kTotalTransforms],
                      const TCrW& tCrW,
                      __ppu_bfloat162& sum_0, __ppu_bfloat162& sum_1) {
    #pragma unroll
    for (int idx = 0; idx < kTotalTransforms; idx++) {
        int n = idx / 4;
        int sub = idx % 4;
        int j = sub * 2;
        __ppu_bfloat162 a = reinterpret_cast<__ppu_bfloat162&>(cvt_buf[idx]);
        __ppu_bfloat162 b = {tCrW(j, n), tCrW(j + 1, n)};
        if (sub % 2 == 0) {
            sum_0 = __hfma2(a, b, sum_0);
        } else {
            sum_1 = __hfma2(a, b, sum_1);
        }
    }
}

// FP8/int8 fma2 phase: weight access via weights[] array with arch branching
template <int kTotalTransforms, typename ElementLogits>
__forceinline__ __device__
void fma2_phase_weights(uint32_t (&cvt_buf)[kTotalTransforms],
                         ElementLogits* weights,
                         __ppu_bfloat162& sum_0, __ppu_bfloat162& sum_1) {
    #pragma unroll
    for (int idx = 0; idx < kTotalTransforms; idx++) {
        int n = idx / 4;
        int sub = idx % 4;
        int j = sub * 2;
        __ppu_bfloat162 a = reinterpret_cast<__ppu_bfloat162&>(cvt_buf[idx]);
#if __HGGC_ARCH__ == 150
        __ppu_bfloat162 b = *reinterpret_cast<__ppu_bfloat162*>(&weights[n * 4 + (j / 4) * 2]);
        if (sub % 2 == 0) {
            sum_0 = __hfma2(a, b, sum_0);
        } else {
            sum_1 = __hfma2(a, b, sum_1);
        }
#else
        __ppu_bfloat162 b = *reinterpret_cast<__ppu_bfloat162*>(&weights[n * 4 + ((j % 4) / 2) * 2]);
        if (sub < 2) {
            sum_0 = __hfma2(a, b, sum_0);
        } else {
            sum_1 = __hfma2(a, b, sum_1);
        }
#endif
    }
}

} // namespace deep_gemm

// ============================================================
// Macro: Define FP4 SFA/SFB/Weight s2r layouts inside a kernel class
// Usage: Call inside class body with the appropriate template params
// ============================================================
// Identical between fp4_mqa_logits.cuh and fp4_paged_mqa_logits.cuh

#define MQA_DEFINE_FP4_S2R_LAYOUTS(WarpOnM_, MmaIterM_, kNumKVStages_, BLOCK_M_, \
                                    WarpOnN_, MmaIterN_, kNumQStages_, BLOCK_N_) \
    using sSFATransLayout = Layout<Shape<Shape<Int<WarpOnM_>, Int<MmaIterM_>>, Shape<_2, _8, _2>, Int<kNumKVStages_>>, \
                                   Stride<Stride<Int<MmaIterM_ * 32>, _32>, Stride<_16, _2, _1>, Int<BLOCK_M_ * 2>>>; \
    using SmemTiledCopySFA = decltype(make_tiled_copy( \
        Copy_Atom<UniversalCopy<uint16_t>, uint16_t>{}, \
        Layout<Shape<Shape<Int<WarpOnM_>, _4>, _8>, Stride<Stride<_32, _1>, _4>>{}, Layout<Shape<_1, _2>>{})); \
    \
    using sSFBTransLayout = Layout<Shape<Shape<Int<WarpOnN_>, Int<MmaIterN_>>, Shape<_2, _8, _2>, Int<kNumQStages_>>, \
                                   Stride<Stride<Int<MmaIterN_ * 32>, _32>, Stride<_16, _2, _1>, Int<BLOCK_N_ * 2>>>; \
    using SmemTiledCopySFB = decltype(make_tiled_copy( \
        Copy_Atom<UniversalCopy<uint16_t>, uint16_t>{}, \
        Layout<Shape<Shape<Int<WarpOnN_>, _4>, _8>, Stride<Stride<_32, _1>, _4>>{}, Layout<Shape<_1, _2>>{})); \
    \
    using sWCopyLayout = Layout< \
        Shape<Shape<_2, _4, Int<WarpOnN_>, _2, Int<MmaIterN_>>, _1, Int<kNumQStages_>>, \
        Stride<Stride<_1, _2, Int<WARP_N>, _8, _16>, _1, Int<BLOCK_N_>>, >; \
    using WeightRegLayout = Layout<Shape<Shape<_2, _2, _2>, Int<MmaIterN_>>, Stride<Stride<_1, _0, _2>, _4>>;

// ============================================================
// Warp interleave template functions
// Usage: deep_gemm::warp_interleave_start<WarpInterleaving>(warp_group_id, thread_count)
// ============================================================

namespace deep_gemm {

template <bool WarpInterleaving>
__forceinline__ __device__ void warp_interleave_start(int warp_group_id, int thread_count) {
    if constexpr (WarpInterleaving) {
        if (warp_group_id == 1) {
            __ppu_barrier_arrive(5, thread_count, 0);
        }
    }
}

template <bool WarpInterleaving>
__forceinline__ __device__ void warp_interleave_sync(int warp_group_id, int thread_count) {
    if constexpr (WarpInterleaving) {
        __ppu_barrier_sync(5 + warp_group_id, thread_count);
    }
}

template <bool WarpInterleaving>
__forceinline__ __device__ void warp_interleave_arrive(int warp_group_id, int thread_count) {
    if constexpr (WarpInterleaving) {
        __ppu_barrier_arrive(6 - warp_group_id, thread_count, 0);
    }
}

template <bool WarpInterleaving>
__forceinline__ __device__ void warp_interleave_end(int warp_group_id, int thread_count) {
    if constexpr (WarpInterleaving) {
        if (warp_group_id == 0) {
            __ppu_barrier_sync(5, thread_count);
        }
    }
}

} // namespace deep_gemm
