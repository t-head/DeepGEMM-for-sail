#pragma once

// Quantization-neutral PPU FP8 MegaMoE routing and symmetric-memory transport.
#include <cstdint>
#include <hggc_runtime.h>
#include <hggc_bf16.h>
#include <hggc_fp8.h>
#include <cutlass/numeric_types.h>
#include <cute/arch/copy_ppu.hpp>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/comm/barrier.cuh>

namespace deep_gemm::mega_moe::detail {

// All GLM hidden sizes currently used here make rows 16-byte aligned.  Keep a
// byte tail so this helper remains correct for reduced unit-test shapes too.
CUTLASS_DEVICE void warp_copy_bytes(
        void* dst, const void* src, const uint32_t num_bytes,
        const uint32_t lane_idx) {
    auto* d16 = reinterpret_cast<uint4*>(dst);
    const auto* s16 = reinterpret_cast<const uint4*>(src);
    const uint32_t n16 = num_bytes / 16;
    for (uint32_t i = lane_idx; i < n16; i += 32)
        d16[i] = s16[i];

    auto* d1 = reinterpret_cast<uint8_t*>(dst);
    const auto* s1 = reinterpret_cast<const uint8_t*>(src);
    for (uint32_t i = n16 * 16 + lane_idx; i < num_bytes; i += 32)
        d1[i] = s1[i];
}

// Stage one aligned remote row through shared memory with the PPU cp.async
// wrapper.  Every lane owns an independent async group; callers can overlap
// scalar scale/router-weight loads before waiting on the group.
CUTLASS_DEVICE void warp_async_copy_global_to_shared(
        void* dst, const void* src, const uint32_t num_bytes,
        const uint32_t lane_idx) {
    auto* d16 = reinterpret_cast<uint4*>(dst);
    const auto* s16 = reinterpret_cast<const uint4*>(src);
    const uint32_t n16 = num_bytes / 16;
    for (uint32_t i = lane_idx; i < n16; i += 32)
        cute::PPU_CP_ASYNC_CACHEALWAYS<uint4>::copy(s16[i], d16[i]);
    cute::cp_async_fence();
}

CUTLASS_DEVICE uint32_t umin(const uint32_t a, const uint32_t b) {
    return a < b ? a : b;
}

template <int kHidden>
CUTLASS_DEVICE void warp_quantize_bf16_row(
        const __ppu_bfloat16* __restrict__ input,
        cutlass::float_e4m3_t* __restrict__ output,
        float* __restrict__ output_scale,
        const uint32_t row, const uint32_t lane_idx) {
    constexpr int kVec = 16;
    // The last iteration may use only part of the warp (e.g. 640 = 512+128).
    // Each active lane still loads/stores a complete aligned 16-element vector;
    // all lanes participate in the amax reduction after the predicated loop.
    static_assert(kHidden > 0 && kHidden % kVec == 0,
                  "whole-row FP8 quantizer requires 16-element alignment");

    const auto* src = input + static_cast<uint64_t>(row) * kHidden;
    float local_amax = 0.0f;
    for (int elem = static_cast<int>(lane_idx) * kVec;
         elem < kHidden; elem += 32 * kVec) {
        const auto* src2 =
            reinterpret_cast<const __ppu_bfloat162*>(src + elem);
        #pragma unroll
        for (int i = 0; i < kVec / 2; ++i) {
            const float2 value = __bfloat1622float2(src2[i]);
            local_amax = fmaxf(
                local_amax, fmaxf(fabsf(value.x), fabsf(value.y)));
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_amax = fmaxf(
            local_amax,
            __shfl_xor_sync(0xffffffffu, local_amax, offset));

    const float scale = local_amax / 448.0f;
    const float scale_inv = scale == 0.0f ? 0.0f : 1.0f / scale;
    if (lane_idx == 0)
        output_scale[row] = scale;

    auto* dst = output + static_cast<uint64_t>(row) * kHidden;
    for (int elem = static_cast<int>(lane_idx) * kVec;
         elem < kHidden; elem += 32 * kVec) {
        const auto* src2 =
            reinterpret_cast<const __ppu_bfloat162*>(src + elem);
        uint4 packed{};
        auto* packed2 = reinterpret_cast<uint16_t*>(&packed);
        #pragma unroll
        for (int i = 0; i < kVec / 2; ++i) {
            const float2 value = __bfloat1622float2(src2[i]);
            const float value0 = value.x * scale_inv;
            const float value1 = value.y * scale_inv;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n"
                : "=h"(packed2[i]) : "f"(value1), "f"(value0));
        }
        *reinterpret_cast<uint4*>(dst + elem) = packed;
    }
}

// Same counter and source-index protocol as fp4_mega_moe_dispatch_gemm1.cuh.
// The data dtype does not participate in Phase A, so keeping an independent,
// small copy lets this header avoid the FP4 Tensor-Core dependency.
template <
    uint32_t kNumRanks, uint32_t kNumCTAs, uint32_t kNumThreads,
    uint32_t kNumExperts, uint32_t kNumExpertsPerRank, uint32_t kNumTopk>
CUTLASS_DEVICE void dispatch_handshake(
        const layout::Workspace& workspace,
        const layout::SymBuffer<kNumRanks>& sym_buffer,
        const int64_t* __restrict__ input_topk_idx,
        uint32_t* __restrict__ smem_expert_count,
        const uint32_t num_tokens, const uint32_t cta_idx,
        const uint32_t thread_idx) {
    const auto block_sync = [=]() { __syncthreads(); };

    for (uint32_t i = thread_idx; i < kNumExperts; i += kNumThreads)
        smem_expert_count[i] = 0;
    __syncthreads();

    const auto for_each_routed = [&](const auto& process) {
        const uint32_t total = num_tokens * kNumTopk;
        for (uint32_t p = cta_idx * kNumThreads + thread_idx; p < total;
             p += kNumCTAs * kNumThreads) {
            const int expert_idx = static_cast<int>(__ldg(input_topk_idx + p));
            if (expert_idx >= 0 && expert_idx < static_cast<int>(kNumExperts))
                process(p, static_cast<uint32_t>(expert_idx));
        }
    };

    for_each_routed([&](const uint32_t, const uint32_t expert_idx) {
        atomicAdd_block(smem_expert_count + expert_idx, 1u);
    });
    __syncthreads();

    for (uint32_t i = thread_idx; i < kNumExperts; i += kNumThreads) {
        const uint64_t value = (1ull << 32) |
            static_cast<uint64_t>(smem_expert_count[i]);
        const uint64_t old = ptx::atomic_add_sys(
            workspace.get_expert_send_count_ptr(i), value);
        smem_expert_count[i] = static_cast<uint32_t>(old & 0xffffffffull);
    }
    __syncthreads();

    for_each_routed([&](const uint32_t token_topk_idx,
                        const uint32_t expert_idx) {
        const uint32_t dst_rank = expert_idx / kNumExpertsPerRank;
        const uint32_t local_expert = expert_idx % kNumExpertsPerRank;
        const uint32_t dst_slot = atomicAdd_block(
            smem_expert_count + expert_idx, 1u);
        auto* dst = workspace.get_src_token_topk_idx_ptr(
            local_expert, sym_buffer.rank_idx, dst_slot);
        *sym_buffer.map(dst, dst_rank) = token_topk_idx;
    });

    comm::grid_sync<kNumCTAs, /*kGridSyncIndex=*/0,
                    /*kPollBackoffCycles=*/128>(
        workspace, cta_idx, thread_idx, block_sync);

    if (cta_idx == 0) {
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumThreads) {
            const uint32_t dst_rank = i / kNumExpertsPerRank;
            const uint32_t local_expert = i % kNumExpertsPerRank;
            const uint64_t status = *workspace.get_expert_send_count_ptr(i);
            *sym_buffer.map(
                workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx,
                                                     local_expert),
                dst_rank) = status & 0xffffffffull;
            ptx::atomic_add_sys(
                sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(
                                    local_expert), dst_rank),
                status);
        }
    }
    __syncthreads();

    comm::cross_rank_barrier<kNumRanks, kNumCTAs, kNumThreads, /*idx=*/0,
                         /*tag=*/0>(
        workspace, sym_buffer, cta_idx, thread_idx, block_sync, true, true);
}

// Dispatch E4M3-token rows to expert-major local pools.  Every CTA remains
// resident for the two global barriers; then one CTA owns each local expert's
// pool rows, avoiding a second inter-CTA handoff before grouped GEMM.
template <
    uint32_t kNumRanks, uint32_t kNumCTAs, uint32_t kNumThreads,
    uint32_t kNumExperts, uint32_t kNumExpertsPerRank, uint32_t kNumTopk,
    uint32_t kBlockM, uint32_t kTokenBytes, uint32_t kNumScales = 1>
CUTLASS_DEVICE void
dispatch_fp8_device(
        void* workspace_base,
        const layout::SymBuffer<kNumRanks> sym_buffer,
        const int64_t* __restrict__ input_topk_idx,
        const float* __restrict__ input_topk_weights,
        const uint8_t* __restrict__ input_token,
        const float* __restrict__ input_scale,
        uint8_t* __restrict__ token_pool,
        float* __restrict__ scale_pool,
        float* __restrict__ topk_weight_pool,
        const uint32_t num_tokens,
        const uint32_t num_max_tokens_per_rank,
        unsigned char* smem_buf) {
    static_assert(kNumThreads >= 32 && kNumThreads % 32 == 0,
                  "dispatch CTA must contain whole warps");
    static_assert(kTokenBytes % 16 == 0,
                  "FP8 dispatch requires 16-byte aligned rows");

    const uint32_t cta_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = thread_idx / 32;
    const uint32_t lane_idx = thread_idx % 32;
    constexpr uint32_t kNumWarps = kNumThreads / 32;

    const layout::Workspace workspace(
        workspace_base, kNumRanks, kNumExperts, num_max_tokens_per_rank,
        kNumTopk);
    auto* smem_expert_count = reinterpret_cast<uint32_t*>(smem_buf);
    constexpr uint32_t kCountBytes = kNumExperts * sizeof(uint32_t);
    constexpr uint32_t kTokenScratchOffset = (kCountBytes + 15u) & ~15u;
    uint8_t* smem_token =
        smem_buf + kTokenScratchOffset + warp_idx * kTokenBytes;
    const auto block_sync = [=]() { __syncthreads(); };

    mega_moe::detail::dispatch_handshake<
        kNumRanks, kNumCTAs, kNumThreads, kNumExperts, kNumExpertsPerRank,
        kNumTopk>(
        workspace, sym_buffer, input_topk_idx, smem_expert_count, num_tokens,
        cta_idx, thread_idx);

    // The workspace reserves each expert an independently padded kBlockM
    // range.  This is exactly the layout consumed by prepare_moe_gemm2_rows.
    for (uint32_t e = cta_idx; e < kNumExpertsPerRank; e += kNumCTAs) {
        uint32_t pool_base = 0;
        for (uint32_t prior = 0; prior < e; ++prior) {
            const uint32_t prior_recv = static_cast<uint32_t>(
                *workspace.get_expert_recv_count_sum_ptr(prior));
            pool_base += math::ceil_div(prior_recv, kBlockM) * kBlockM;
        }
        const uint32_t num_recv = static_cast<uint32_t>(
            *workspace.get_expert_recv_count_sum_ptr(e));

        // The first kNumRanks lanes load this expert's source ranges in
        // parallel.  All rows assigned to this warp reuse the resulting
        // inclusive prefix instead of lane 0 scanning every rank per row.
        uint32_t src_count = lane_idx < kNumRanks
            ? static_cast<uint32_t>(
                  *workspace.get_expert_recv_count_ptr(lane_idx, e))
            : 0;
        uint32_t src_end = src_count;
        #pragma unroll
        for (uint32_t offset = 1; offset < kNumRanks; offset <<= 1) {
            const uint32_t value =
                __shfl_up_sync(0xffffffffu, src_end, offset);
            if (lane_idx >= offset)
                src_end += value;
        }
        const uint32_t src_begin = src_end - src_count;

        for (uint32_t local_row = warp_idx; local_row < num_recv;
             local_row += kNumWarps) {
            const uint32_t src_mask = __ballot_sync(
                0xffffffffu, lane_idx < kNumRanks &&
                    local_row >= src_begin && local_row < src_end);
            const uint32_t src_rank =
                static_cast<uint32_t>(__ffs(src_mask) - 1);
            const uint32_t range_begin = __shfl_sync(
                0xffffffffu, src_begin, src_rank);
            const uint32_t src_slot = local_row - range_begin;
            uint32_t src_token_topk = 0;
            if (lane_idx == 0)
                src_token_topk = *workspace.get_src_token_topk_idx_ptr(
                    e, src_rank, src_slot);
            src_token_topk = __shfl_sync(0xffffffffu, src_token_topk, 0);

            const uint32_t src_token = src_token_topk / kNumTopk;
            const uint32_t src_topk = src_token_topk % kNumTopk;
            const uint32_t pool_row = pool_base + local_row;
            const uint8_t* src_token_ptr = sym_buffer.map(
                input_token + static_cast<uint64_t>(src_token) * kTokenBytes,
                src_rank);

            mega_moe::detail::warp_async_copy_global_to_shared(
                smem_token, src_token_ptr, kTokenBytes, lane_idx);
            // Input scales are token-major; the landing pool is K-group
            // major so the native blockwise GEMM can consume it directly.
            for (uint32_t sf = lane_idx; sf < kNumScales; sf += 32) {
                const float value = *sym_buffer.map(
                    input_scale + static_cast<uint64_t>(src_token) * kNumScales + sf,
                    src_rank);
                scale_pool[static_cast<uint64_t>(sf)
                           * workspace.num_max_pool_tokens + pool_row] = value;
            }
            float topk_weight = 0.0f;
            if (lane_idx == 0) {
                topk_weight = *sym_buffer.map(
                    input_topk_weights + src_token_topk, src_rank);
            }
            asm volatile("" ::: "memory");
            cute::cp_async_wait<0>();
            __syncwarp();
            mega_moe::detail::warp_copy_bytes(
                token_pool + static_cast<uint64_t>(pool_row) * kTokenBytes,
                smem_token, kTokenBytes, lane_idx);

            if (lane_idx == 0) {
                topk_weight_pool[pool_row] = topk_weight;
                *workspace.get_token_src_metadata_ptr(pool_row) = {
                    src_rank, src_token, src_topk};
            }
            __syncwarp();
        }
    }

    // The caller reads recv_count_sum and token_src_metadata after this kernel;
    // do not clear dispatch counters until its G2/scatter phase completes.
    comm::grid_sync<kNumCTAs, /*kGridSyncIndex=*/1>(
        workspace, cta_idx, thread_idx, block_sync);
}

}  // namespace deep_gemm::mega_moe::detail
