#pragma once

#include <cstdint>
#include <hggc.h>
#include <deep_gemm/tix/ld_st.cuh>

#ifndef DG_MEGAMOE_OWNER_MODE
#define DG_MEGAMOE_OWNER_MODE 0
#endif

#ifndef DG_MEGAMOE_OWNER_READER_OPT
#define DG_MEGAMOE_OWNER_READER_OPT 1
#endif

namespace deep_gemm {

template <bool kBulkRemote>
__device__ __forceinline__ int4 moe_reduce_load(
        const __ppu_bfloat16* source, bool remote) {
    if constexpr (kBulkRemote) {
        if (remote) {
            const uint4 bulk = __ppu_global_ldbl_bulk_b32x4(source);
            return *reinterpret_cast<const int4*>(&bulk);
        }
    }
    return __ldbl(reinterpret_cast<const int4*>(source));
}

template <bool kCheckInvalid, bool kOutputAligned, int kTopK, int kHidden, bool kBulkRemote,
          int kThreads, int kColumnSplits = 1>
__device__ __forceinline__ void moe_reduce_token(
        const __ppu_bfloat16* __restrict__ input,
        __ppu_bfloat16* __restrict__ output,
        const int64_t (&slots)[kTopK], const float (&weights)[kTopK],
        const bool (&remote_slots)[kTopK], int token, int column_part, unsigned int reduce_thread) {
    constexpr int kVector = sizeof(int4) / sizeof(__ppu_bfloat16);
    constexpr int kStride = kThreads * kVector;
    constexpr int kColumns = kHidden / kColumnSplits;
    for (int column = column_part * kColumns + reduce_thread * kVector;
         column < (column_part + 1) * kColumns; column += kStride) {
        int4 inputs[kTopK];
        // The fused GEMM grid has fewer resident warps. Overlap TopK
        // reads before consuming them, preserving the original sum order.
        #pragma unroll
        for (int index = 0; index < kTopK; ++index) {
            if constexpr (kCheckInvalid) {
                if (slots[index] < 0) continue;
            }
            inputs[index] = moe_reduce_load<kBulkRemote>(
                input + slots[index] * kHidden + column, remote_slots[index]);
        }
        float accumulator[kVector] = {0.0f};
        #pragma unroll
        for (int index = 0; index < kTopK; ++index) {
            if constexpr (kCheckInvalid) {
                if (slots[index] < 0) continue;
            }
            const int4 packed = inputs[index];
            const auto* values = reinterpret_cast<const __ppu_bfloat16*>(&packed);
            #pragma unroll
            for (int lane = 0; lane < kVector; ++lane) {
                accumulator[lane] += __bfloat162float(values[lane]) * weights[index];
            }
        }
        int4 packed_output;
        auto* pairs = reinterpret_cast<__ppu_bfloat162*>(&packed_output);
        #pragma unroll
        for (int pair = 0; pair < kVector / 2; ++pair) {
            pairs[pair] = __floats2bfloat162_rn(
                accumulator[2 * pair], accumulator[2 * pair + 1]);
        }
        tix::store_bf16x8<kOutputAligned>(output + static_cast<int64_t>(token) * kHidden + column, packed_output);
    }
}

// GEMM2 scatter completes peer synchronization before this weighted reduction.
template <bool kOutputAligned, int kTopK, int kHidden, bool kBulkRemote, int kThreads, int kColumnSplits = 1>
__device__ __forceinline__ void moe_reduce_impl(
        const __ppu_bfloat16* __restrict__ input,
        const int64_t* __restrict__ topk_indices,
        const float* __restrict__ topk_weights,
        __ppu_bfloat16* __restrict__ output,
        int max_m, int local_experts, int rank_idx, int token, int column_part, uint64_t writer_mask, unsigned int reduce_thread) {
    static_assert(kThreads % 32 == 0);
    static_assert(kHidden % kColumnSplits == 0 && (kHidden / kColumnSplits) % 256 == 0,
                  "each column split must contain complete 256-element warp rows");
    int64_t slots[kTopK];
    float weights[kTopK];
    bool remote_slots[kTopK];
    bool all_valid = true;
    // Expert IDs are contiguous within each rank. Reuse the local interval
    // across TopK instead of dividing every expert ID by local_experts.
    const int local_expert_begin = rank_idx * local_experts;
    const int local_expert_end = local_expert_begin + local_experts;
    // Publication validates each int64 route before storing an in-range ID or -1.
    int32_t expert_ids[kTopK];
    bool metadata_valid = true;
    #pragma unroll
    for (int index = 0; index < kTopK; ++index) {
        expert_ids[index] = static_cast<int32_t>(__ldg(topk_indices + token * kTopK + index));
        metadata_valid &= expert_ids[index] >= 0;
    }
    if (metadata_valid) {
        // The common all-valid path has no per-slot load dependency on
        // validity, allowing route and weight metadata reads to overlap.
        #pragma unroll
        for (int index = 0; index < kTopK; ++index) {
            slots[index] = index * static_cast<int64_t>(max_m) + token;
            weights[index] = __ldg(topk_weights + token * kTopK + index);
            remote_slots[index] = kBulkRemote &&
                (expert_ids[index] < local_expert_begin || expert_ids[index] >= local_expert_end);
#if DG_MEGAMOE_OWNER_MODE
#if DG_MEGAMOE_OWNER_READER_OPT
            // Uniform all-bulk launches need no per-route owner calculation.
            if (writer_mask != UINT64_MAX && remote_slots[index])
#endif
                remote_slots[index] &= ((writer_mask >> (expert_ids[index] / local_experts)) & 1) != 0;
#endif
        }
        moe_reduce_token<false, kOutputAligned, kTopK, kHidden, kBulkRemote, kThreads, kColumnSplits>(
            input, output, slots, weights, remote_slots, token, column_part, reduce_thread);
        return;
    }
    #pragma unroll
    for (int index = 0; index < kTopK; ++index) {
        const int32_t expert = static_cast<int32_t>(
            __ldg(topk_indices + token * kTopK + index));
        const bool valid = expert >= 0;
        all_valid &= valid;
        if constexpr (kBulkRemote) {
            remote_slots[index] = valid &&
                (expert < local_expert_begin || expert >= local_expert_end);
        } else {
            remote_slots[index] = false;
        }
#if DG_MEGAMOE_OWNER_MODE
#if DG_MEGAMOE_OWNER_READER_OPT
        if (writer_mask != UINT64_MAX && remote_slots[index])
#else
        if (valid)
#endif
            remote_slots[index] &= ((writer_mask >> (expert / local_experts)) & 1) != 0;
#endif
        slots[index] = valid
            ? index * static_cast<int64_t>(max_m) + token
            : -1;
        weights[index] = valid
            ? __ldg(topk_weights + token * kTopK + index)
            : 0.0f;
    }

    // Skip invalid slots before loading scatter data, even when their weight is zero.
    if (all_valid) {
        moe_reduce_token<false, kOutputAligned, kTopK, kHidden, kBulkRemote, kThreads, kColumnSplits>(
            input, output, slots, weights, remote_slots, token, column_part, reduce_thread);
    } else {
        moe_reduce_token<true, kOutputAligned, kTopK, kHidden, kBulkRemote, kThreads, kColumnSplits>(
            input, output, slots, weights, remote_slots, token, column_part, reduce_thread);
    }
}

// Distribute token/column tasks across warp-aligned groups for every operand type.
template <bool kOutputAligned, int kTopK, int kHidden, int kCtaThreads,
          int kColumnSplits = 1, int kReduceGroupThreads = 0>
__device__ __forceinline__ void moe_reduce_dispatch(
        const __ppu_bfloat16* input, const int64_t* ids, const float* weights,
        __ppu_bfloat16* output, int token_capacity, int local_experts, int rank_idx,
        int tokens, uint64_t writer_mask) {
    constexpr int kReduceThreads = kReduceGroupThreads > 0 ? kReduceGroupThreads : kCtaThreads;
    static_assert(kReduceThreads % 32 == 0 && kCtaThreads % kReduceThreads == 0);
    constexpr int kTokenGroups = kCtaThreads / kReduceThreads;
    unsigned int reduce_thread = threadIdx.x;
    int token_group = 0;
    if constexpr (kReduceGroupThreads > 0) {
        reduce_thread = threadIdx.x % kReduceThreads;
        // Group boundaries are warp-aligned; keep token metadata and owner decisions scalar.
        token_group = __ppu_read_firstlane(threadIdx.x / kReduceThreads);
    }
    if (kColumnSplits == 1 || reduce_thread * 8 < kHidden / kColumnSplits) {
        for (int work = blockIdx.x * kTokenGroups + token_group;
             work < tokens * kColumnSplits; work += gridDim.x * kTokenGroups) {
            moe_reduce_impl<kOutputAligned, kTopK, kHidden, true, kReduceThreads, kColumnSplits>(
                input,
                ids, weights, output,
                token_capacity, local_experts,
                rank_idx, work / kColumnSplits, work % kColumnSplits, writer_mask, reduce_thread);
        }
    }
}

}  // namespace deep_gemm
