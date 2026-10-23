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
            packed2[i] = __hg_cvt_float2_to_fp8x2(
                make_float2(value0, value1), __HG_SATFINITE, __HG_E4M3);
        }
        *reinterpret_cast<uint4*>(dst + elem) = packed;
    }
}

// Same counter and source-index protocol as fp4_mega_moe_dispatch_gemm1.cuh.
// The data dtype does not participate in Phase A, so keeping an independent,
// small copy lets this header avoid the FP4 Tensor-Core dependency.
// One source range per lane and per 32-rank chunk. EP64 needs two chunks;
// shuffling by 32 is not a prefix scan across two warps. All methods must be
// called by a converged warp, with the same expert/row in every lane.
template <uint32_t kNumRanks>
struct WarpSourceRanges {
    static_assert(kNumRanks > 0 && kNumRanks <= 64);
    static constexpr uint32_t kChunks = (kNumRanks + 31) / 32;
    uint32_t begin[kChunks], end[kChunks];

    CUTLASS_DEVICE WarpSourceRanges(const layout::Workspace& ws,
                                    uint32_t expert, uint32_t lane) {
        uint32_t carry = 0;
        #pragma unroll
        for (uint32_t chunk = 0; chunk < kChunks; ++chunk) {
            const uint32_t rank = chunk * 32 + lane;
            const uint32_t count = rank < kNumRanks
                ? uint32_t(*ws.get_expert_recv_count_ptr(rank, expert)) : 0;
            uint32_t prefix = count;
            #pragma unroll
            for (uint32_t offset = 1; offset < 32 && offset < kNumRanks; offset <<= 1) {
                const uint32_t value = __shfl_up_sync(0xffffffffu, prefix, offset);
                if (lane >= offset) prefix += value;
            }
            begin[chunk] = carry + prefix - count;
            end[chunk] = carry + prefix;
            if constexpr (kChunks > 1)
                carry += __shfl_sync(0xffffffffu, prefix, 31);
        }
    }

    struct Source { uint32_t rank, slot; };
    CUTLASS_DEVICE Source resolve(uint32_t row, uint32_t lane) const {
        Source source{kNumRanks, 0};
        #pragma unroll
        for (uint32_t chunk = 0; chunk < kChunks; ++chunk) {
            const uint32_t mask = __ballot_sync(0xffffffffu,
                chunk * 32 + lane < kNumRanks && row >= begin[chunk] && row < end[chunk]);
            if (mask) {
                const uint32_t owner = uint32_t(__ffs(mask) - 1);
                source = {chunk * 32 + owner,
                          row - __shfl_sync(0xffffffffu, begin[chunk], owner)};
            }
        }
        return source;
    }
};

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
        const uint64_t old = tix::atomic_add_sys(
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
            tix::atomic_add_sys(
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

}  // namespace deep_gemm::mega_moe::detail
