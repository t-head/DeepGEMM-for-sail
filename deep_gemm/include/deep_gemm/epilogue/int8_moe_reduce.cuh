#pragma once

#ifndef DG_MEGAMOE_OWNER_MODE
#define DG_MEGAMOE_OWNER_MODE 0
#endif

#ifndef DG_MEGAMOE_OWNER_READER_OPT
#define DG_MEGAMOE_OWNER_READER_OPT 1
#endif

#ifndef DG_MEGAMOE_REDUCE_GROUP_THREADS
#define DG_MEGAMOE_REDUCE_GROUP_THREADS 0
#endif

#include <cstdint>
#include <hggc.h>
#include <hggc_bf16.h>
#include <hggc_runtime.h>

namespace deep_gemm::sm80 {

template <bool kBulkRemote>
__device__ __forceinline__ int4 moe_reduce_load(
        const __ppu_bfloat16* source, bool remote) {
    if constexpr (kBulkRemote) {
        if (remote) {
            const uint4 bulk = __ppu_global_ldcg_bulk_b32x4(source);
            return *reinterpret_cast<const int4*>(&bulk);
        }
    }
    return __ldcg(reinterpret_cast<const int4*>(source));
}

template <bool kCheckInvalid, int kTopK, int kHidden, bool kBulkRemote,
          int kThreads, int kColumnSplits = 1>
__device__ __forceinline__ void moe_reduce_token(
        const __ppu_bfloat16* __restrict__ input,
        __ppu_bfloat16* __restrict__ output,
        const int64_t (&slots)[kTopK], const float (&weights)[kTopK],
        const bool (&remote_slots)[kTopK], int token, int column_part = 0) {
    constexpr int kVector = sizeof(int4) / sizeof(__ppu_bfloat16);
    constexpr int kStride = kThreads * kVector;
    static_assert(kHidden % kColumnSplits == 0);
    constexpr int kColumns = kHidden / kColumnSplits;
#if DG_MEGAMOE_REDUCE_GROUP_THREADS
    const int reduce_thread = threadIdx.x % kThreads;
    for (int column = column_part * kColumns + reduce_thread * kVector;
#else
    for (int column = column_part * kColumns + threadIdx.x * kVector;
#endif
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
        *reinterpret_cast<int4*>(output + token * kHidden + column) = packed_output;
    }
}

// GEMM2 scatter completes peer synchronization before this weighted reduction.
template <int kTopK, int kHidden, bool kBulkRemote, int kThreads, int kColumnSplits = 1>
__device__ __forceinline__ void moe_reduce_impl(
        const __ppu_bfloat16* __restrict__ input,
        const int32_t* __restrict__ topk_indices,
        const float* __restrict__ topk_weights,
        __ppu_bfloat16* __restrict__ output,
        int max_m, int local_experts, int rank_idx, int token, int column_part = 0, uint64_t writer_mask = UINT64_MAX) {
    static_assert(kHidden % 256 == 0, "bulk reader requires complete warp rows");
    static_assert(kThreads % 32 == 0);
    static_assert(kHidden % kColumnSplits == 0 && (kHidden / kColumnSplits) % 256 == 0);
    int64_t slots[kTopK];
    float weights[kTopK];
    bool remote_slots[kTopK];
    bool all_valid = true;
    // Expert IDs are contiguous within each rank. Reuse the local interval
    // across TopK instead of dividing every expert ID by local_experts.
    const int local_expert_begin = rank_idx * local_experts;
    const int local_expert_end = local_expert_begin + local_experts;
    int32_t expert_ids[kTopK];
    bool metadata_valid = true;
    #pragma unroll
    for (int index = 0; index < kTopK; ++index) {
        expert_ids[index] = __ldg(topk_indices + token * kTopK + index);
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
        moe_reduce_token<false, kTopK, kHidden, kBulkRemote, kThreads, kColumnSplits>(
            input, output, slots, weights, remote_slots, token, column_part);
        return;
    }
    #pragma unroll
    for (int index = 0; index < kTopK; ++index) {
        const int32_t expert =
            __ldg(topk_indices + token * kTopK + index);
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
        moe_reduce_token<false, kTopK, kHidden, kBulkRemote, kThreads, kColumnSplits>(
            input, output, slots, weights, remote_slots, token, column_part);
    } else {
        moe_reduce_token<true, kTopK, kHidden, kBulkRemote, kThreads, kColumnSplits>(
            input, output, slots, weights, remote_slots, token, column_part);
    }
}

} // namespace deep_gemm::sm80
