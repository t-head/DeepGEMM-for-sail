#pragma once

// Channelwise FP8 fused MegaMoE; shared routing is independent of GEMM dtype/layout.
#include <deep_gemm/impls/fp8_mega_moe_transport.cuh>
#include <deep_gemm/impls/fp8_channelwise_mega_moe_gemm1_mma.cuh>

namespace deep_gemm::mega_moe::fp8_channelwise {

// Cache the BF16 pairs loaded for amax and reuse them for FP8 conversion.
// The measured 3072-wide activation needs 48 packed pairs per lane. Keep
// the existing helper for other shapes to avoid increasing register demand.
template <int kHidden>
CUTLASS_DEVICE void warp_quantize_bf16_row_cached(
        const __ppu_bfloat16* __restrict__ input,
        cutlass::float_e4m3_t* __restrict__ output,
        float* __restrict__ output_scale,
        const uint32_t row, const uint32_t lane_idx) {
    constexpr int kVec = 16;
    static_assert(kHidden % (32 * kVec) == 0,
                  "whole-row FP8 quantizer requires 512-element alignment");

    const auto* src = input + static_cast<uint64_t>(row) * kHidden;
    __ppu_bfloat162 cached[kHidden / 64];
    float local_amax = 0.0f;
    #pragma unroll
    for (int elem = static_cast<int>(lane_idx) * kVec;
         elem < kHidden; elem += 32 * kVec) {
        const auto* src2 =
            reinterpret_cast<const __ppu_bfloat162*>(src + elem);
        #pragma unroll
        for (int i = 0; i < kVec / 2; ++i) {
            const __ppu_bfloat162 pair = src2[i];
            cached[(elem / (32 * kVec)) * (kVec / 2) + i] = pair;
            const float2 value = __bfloat1622float2(pair);
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
    #pragma unroll
    for (int elem = static_cast<int>(lane_idx) * kVec;
         elem < kHidden; elem += 32 * kVec) {
        uint4 packed{};
        auto* packed2 = reinterpret_cast<uint16_t*>(&packed);
        #pragma unroll
        for (int i = 0; i < kVec / 2; ++i) {
            const float2 value = __bfloat1622float2(
                cached[(elem / (32 * kVec)) * (kVec / 2) + i]);
            const float value0 = value.x * scale_inv;
            const float value1 = value.y * scale_inv;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n"
                : "=h"(packed2[i]) : "f"(value1), "f"(value0));
        }
        *reinterpret_cast<uint4*>(dst + elem) = packed;
    }
}

// Quantize valid expert-major rows after a stream-ordered GEMM boundary.
// Keeping this stage separate avoids extending persistent GEMM register
// lifetimes through the whole-grid quantization rendezvous.
template <uint32_t kEPR, uint32_t kWidth, uint32_t kBlockM>
CUTLASS_DEVICE void fp8_mega_moe_quantize_local_pool_kernel_device(
        const int64_t* counts, const __ppu_bfloat16* input,
        cutlass::float_e4m3_t* output, float* scales) {
    static_assert(kEPR > 0 && kEPR <= 128);
    constexpr uint32_t kChunks = (kEPR + 31) / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warps = blockDim.x / 32;
    const uint32_t warp = blockIdx.x * warps + threadIdx.x / 32;
    uint32_t rows[kChunks], begin[kChunks], end[kChunks];
    uint32_t base = 0;
    #pragma unroll
    for (uint32_t chunk = 0; chunk < kChunks; ++chunk) {
        const uint32_t expert = chunk * 32 + lane;
        rows[chunk] = expert < kEPR ? static_cast<uint32_t>(counts[expert]) : 0;
        const uint32_t blocks = math::ceil_div(rows[chunk], kBlockM);
        uint32_t prefix = blocks;
        #pragma unroll
        for (uint32_t off = 1; off < 32; off <<= 1) {
            const uint32_t value = __shfl_up_sync(0xffffffffu, prefix, off);
            if (lane >= off) prefix += value;
        }
        begin[chunk] = (base + prefix - blocks) * kBlockM;
        end[chunk] = (base + prefix) * kBlockM;
        base += __shfl_sync(0xffffffffu, prefix, 31);
    }
    const auto at = [&](const uint32_t* values, uint32_t expert) {
        uint32_t value = 0;
        #pragma unroll
        for (uint32_t chunk = 0; chunk < kChunks; ++chunk)
            if (chunk == expert / 32) value = values[chunk];
        return __shfl_sync(0xffffffffu, value, expert % 32);
    };
    uint32_t expert = 0;
    for (uint32_t row = warp; row < base * kBlockM; row += gridDim.x * warps) {
        while (expert < kEPR && row >= at(end, expert)) ++expert;
        if (expert >= kEPR) break;
        if (row - at(begin, expert) < at(rows, expert))
            warp_quantize_bf16_row_cached<kWidth>(input, output, scales, row, lane);
    }
}

template <uint32_t kEPR, uint32_t kWidth, uint32_t kBlockM>
__global__ void fp8_mega_moe_quantize_local_pool_kernel(
        const int64_t* counts, const __ppu_bfloat16* input,
        cutlass::float_e4m3_t* output, float* scales) {
    fp8_mega_moe_quantize_local_pool_kernel_device<kEPR, kWidth, kBlockM>(counts, input, output, scales);
}

template <
    uint32_t kNumRanks, uint32_t kNumCTAs, uint32_t kNumThreads,
    uint32_t kNumExperts, uint32_t kNumExpertsPerRank, uint32_t kNumTopk,
    uint32_t kBlockM, uint32_t kTokenBytes, uint32_t kGemm1N,
    uint32_t kBlockN, uint32_t kBlockK, uint32_t kWarpM,
    uint32_t kWarpN, uint32_t kStages, uint32_t kNumPullCTAs,
    bool kQuantizeOutput = false,
    bool kPairedBlockPull = false, bool kTripleBlockPull = false,
    bool kSwiGLUOAI = false>
CUTLASS_DEVICE void dispatch_gemm1_device(
        void* workspace_base,
        const layout::SymBuffer<kNumRanks>& sym_buffer,
        const int64_t* __restrict__ input_topk_idx,
        const float* __restrict__ input_topk_weights,
        const uint8_t* __restrict__ input_token,
        const float* __restrict__ input_scale,
        uint8_t* __restrict__ token_pool,
        float* __restrict__ scale_pool,
        float* __restrict__ topk_weight_pool,
        const cutlass::float_e4m3_t* __restrict__ gemm1_weight,
        const float* __restrict__ gemm1_weight_scale,
        const cutlass::float_e4m3_t* __restrict__ gemm1_weight_replicas,
        const float* __restrict__ gemm1_weight_scale_replicas,
        const uint32_t num_master_experts,
        __ppu_bfloat16* __restrict__ swiglu_output,
        cutlass::float_e4m3_t* __restrict__ quant_output,
        float* __restrict__ quant_scale,
        const float swiglu_limit,
        const float swiglu_alpha,
        const uint32_t num_tokens,
        const uint32_t num_max_tokens_per_rank) {
    using Cfg = mega_moe::fp8_channelwise::Gemm1MmaConfig<
        kBlockM, kBlockN, kBlockK, kWarpM, kWarpN, kStages>;
    static_assert(kNumExpertsPerRank > 0 && kNumExpertsPerRank <= 128,
                  "register-distributed scheduler supports up to 128 physical "
                  "experts per rank");
    static_assert(kGemm1N % kBlockN == 0 && kBlockN % 16 == 0,
                  "GEMM1 N must be tiled by complete interleave groups");
    static_assert(kTokenBytes % kBlockK == 0,
                  "GEMM1 K must be divisible by BLOCK_K");
    static_assert(kNumThreads == Cfg::ThreadNum,
                  "CTA threads must match the GEMM TiledMMA");
    static_assert(kNumPullCTAs > 0 && kNumPullCTAs < kNumCTAs,
                  "overlap requires separate pull and GEMM CTAs");
    static_assert(!kPairedBlockPull || kNumPullCTAs % 2 == 0,
                  "paired block pull requires two equal producer teams");
    static_assert(!kTripleBlockPull || kNumPullCTAs % 3 == 0,
                  "triple block pull requires three equal producer teams");
    static_assert(!(kPairedBlockPull && kTripleBlockPull),
                  "select at most one producer-team schedule");

    extern __shared__ __align__(16) unsigned char smem_buf[];
    const uint32_t cta_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = thread_idx / 32;
    const uint32_t lane_idx = thread_idx % 32;
    constexpr uint32_t kNumWarps = kNumThreads / 32;
    constexpr uint32_t kNumGemmCTAs = kNumCTAs - kNumPullCTAs;
    const layout::Workspace workspace(
        workspace_base, kNumRanks, kNumExperts, num_max_tokens_per_rank,
        kNumTopk);
    const auto block_sync = [=]() { __syncthreads(); };
    auto* smem_expert_count = reinterpret_cast<uint32_t*>(smem_buf);

    mega_moe::detail::dispatch_handshake<
        kNumRanks, kNumCTAs, kNumThreads, kNumExperts, kNumExpertsPerRank,
        kNumTopk>(
        workspace, sym_buffer, input_topk_idx, smem_expert_count, num_tokens,
        cta_idx, thread_idx);

    // Every warp holds the same distributed expert histogram. Each lane owns
    // one expert in every 32-expert chunk, retaining the register-only
    // scheduler for replica slots and Qwen's 128 experts/rank (four chunks).
    constexpr uint32_t kExpertChunks = (kNumExpertsPerRank + 31) / 32;
    uint32_t expert_m[kExpertChunks];
    uint32_t block_begin[kExpertChunks];
    uint32_t block_end[kExpertChunks];
    uint32_t chunk_base = 0;
    #pragma unroll
    for (uint32_t chunk = 0; chunk < kExpertChunks; ++chunk) {
        const uint32_t expert = chunk * 32 + lane_idx;
        expert_m[chunk] = expert < kNumExpertsPerRank
            ? static_cast<uint32_t>(
                  *workspace.get_expert_recv_count_sum_ptr(expert))
            : 0;
        const uint32_t expert_blocks =
            math::ceil_div(expert_m[chunk], kBlockM);
        uint32_t chunk_end = expert_blocks;
        #pragma unroll
        for (uint32_t offset = 1; offset < 32; offset <<= 1) {
            const uint32_t value =
                __shfl_up_sync(0xffffffffu, chunk_end, offset);
            if (lane_idx >= offset)
                chunk_end += value;
        }
        block_begin[chunk] =
            chunk_base + chunk_end - expert_blocks;
        block_end[chunk] = chunk_base + chunk_end;
        chunk_base += __shfl_sync(0xffffffffu, chunk_end, 31);
    }
    const uint32_t total_m_blocks = chunk_base;
    constexpr uint32_t kNumNBlocks = kGemm1N / kBlockN;
    constexpr uint32_t kMBlocksPerSwizzleGroup = 2;
    const uint32_t total_work = total_m_blocks * kNumNBlocks;

    const auto dget_m = [&](uint32_t e) {
        uint32_t local_m = 0;
        #pragma unroll
        for (uint32_t chunk = 0; chunk < kExpertChunks; ++chunk)
            if (chunk == e / 32)
                local_m = expert_m[chunk];
        return __shfl_sync(0xffffffffu, local_m, e % 32);
    };
    const auto dget_block_begin = [&](uint32_t e) {
        uint32_t local_begin = 0;
        #pragma unroll
        for (uint32_t chunk = 0; chunk < kExpertChunks; ++chunk)
            if (chunk == e / 32)
                local_begin = block_begin[chunk];
        return __shfl_sync(0xffffffffu, local_begin, e % 32);
    };

    // Arrival counters are persistent workspace state. Clear the exact active
    // prefix before producer and consumer CTAs diverge.
    for (uint32_t g = cta_idx * kNumThreads + thread_idx;
         g < total_m_blocks; g += kNumCTAs * kNumThreads)
        *workspace.get_l1_arrival_count_ptr(g) = 0;
    comm::grid_sync<kNumCTAs, /*kGridSyncIndex=*/2>(
        workspace, cta_idx, thread_idx, block_sync);

    struct BlockInfo {
        uint32_t expert;
        uint32_t expert_m;
        uint32_t expert_block_begin;
        uint32_t m_block;
        uint32_t valid_m;
        uint32_t pool_base;
    };

    const auto resolve_m_block = [&](uint32_t g) {
        uint32_t expert = kNumExpertsPerRank;
        #pragma unroll
        for (uint32_t chunk = 0; chunk < kExpertChunks; ++chunk) {
            const uint32_t lane_expert = chunk * 32 + lane_idx;
            const bool local_hit =
                lane_expert < kNumExpertsPerRank &&
                g >= block_begin[chunk] && g < block_end[chunk];
            const uint32_t hit_mask =
                __ballot_sync(0xffffffffu, local_hit);
            if (expert == kNumExpertsPerRank && hit_mask != 0)
                expert = chunk * 32
                    + static_cast<uint32_t>(__ffs(hit_mask) - 1);
        }
        const uint32_t group_m = dget_m(expert);
        const uint32_t group_block_begin = dget_block_begin(expert);
        const uint32_t m_block = g - group_block_begin;
        return BlockInfo{
            expert,
            group_m,
            group_block_begin,
            m_block,
            mega_moe::detail::umin(
                kBlockM, group_m - m_block * kBlockM),
            g * kBlockM};
    };

    // Pull CTAs cooperate on each M block. A hot expert is therefore split
    // across kNumPullCTAs instead of being pinned to one CTA. Each warp owns one
    // 6144-byte shared slot and publishes the block only after all local stores
    // and metadata writes are complete.
    if (cta_idx < kNumPullCTAs) {
        constexpr uint32_t kNumPullTeams =
            kTripleBlockPull ? 3 : (kPairedBlockPull ? 2 : 1);
        constexpr uint32_t kPullTeamSize =
            kNumPullCTAs / kNumPullTeams;
        const uint32_t pull_team =
            kNumPullTeams > 1 ? cta_idx / kPullTeamSize : 0;
        const uint32_t team_sm_idx =
            kNumPullTeams > 1 ? cta_idx % kPullTeamSize : cta_idx;
        uint8_t* smem_token =
            smem_buf + static_cast<uint64_t>(warp_idx) * kTokenBytes;
        for (uint32_t g = pull_team; g < total_m_blocks;
             g += kNumPullTeams) {
            const BlockInfo bi = resolve_m_block(g);

            uint32_t src_count = lane_idx < kNumRanks
                ? static_cast<uint32_t>(
                      *workspace.get_expert_recv_count_ptr(
                          lane_idx, bi.expert))
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

            const uint32_t rows_per_sm = bi.valid_m / kPullTeamSize;
            const uint32_t extra_rows = bi.valid_m % kPullTeamSize;
            const uint32_t my_begin =
                team_sm_idx * rows_per_sm
                + mega_moe::detail::umin(team_sm_idx, extra_rows);
            const uint32_t my_rows =
                rows_per_sm + (team_sm_idx < extra_rows ? 1u : 0u);
            const uint32_t num_batches =
                math::ceil_div(my_rows, kNumWarps);

            for (uint32_t batch = 0; batch < num_batches; ++batch) {
                const uint32_t row_in_block =
                    my_begin + batch * kNumWarps + warp_idx;
                if (row_in_block < my_begin + my_rows) {
                    const uint32_t row_in_expert =
                        bi.m_block * kBlockM + row_in_block;
                    const uint32_t src_mask = __ballot_sync(
                        0xffffffffu,
                        lane_idx < kNumRanks &&
                        row_in_expert >= src_begin &&
                        row_in_expert < src_end);
                    const uint32_t src_rank =
                        static_cast<uint32_t>(__ffs(src_mask) - 1);
                    const uint32_t range_begin = __shfl_sync(
                        0xffffffffu, src_begin, src_rank);
                    const uint32_t src_slot =
                        row_in_expert - range_begin;
                    uint32_t src_token_topk = 0;
                    if (lane_idx == 0)
                        src_token_topk =
                            *workspace.get_src_token_topk_idx_ptr(
                                bi.expert, src_rank, src_slot);
                    src_token_topk = __shfl_sync(
                        0xffffffffu, src_token_topk, 0);

                    const uint32_t src_token =
                        src_token_topk / kNumTopk;
                    const uint32_t src_topk =
                        src_token_topk % kNumTopk;
                    const uint8_t* src_token_ptr = sym_buffer.map(
                        input_token +
                            static_cast<uint64_t>(src_token) * kTokenBytes,
                        src_rank);
                    mega_moe::detail::warp_async_copy_global_to_shared(
                        smem_token, src_token_ptr, kTokenBytes, lane_idx);

                    float token_scale = 0.0f;
                    float topk_weight = 0.0f;
                    if (lane_idx == 0) {
                        token_scale = *sym_buffer.map(
                            input_scale + src_token, src_rank);
                        topk_weight = *sym_buffer.map(
                            input_topk_weights + src_token_topk, src_rank);
                    }
                    asm volatile("" ::: "memory");
                    cute::cp_async_wait<0>();
                    __syncwarp();

                    const uint32_t pool_row =
                        bi.pool_base + row_in_block;
                    auto* pool_token_row =
                        token_pool +
                        static_cast<uint64_t>(pool_row) * kTokenBytes;
                    mega_moe::detail::warp_copy_bytes(
                        pool_token_row, smem_token, kTokenBytes, lane_idx);
                    if (lane_idx == 0) {
                        scale_pool[pool_row] = token_scale;
                        topk_weight_pool[pool_row] = topk_weight;
                        *workspace.get_token_src_metadata_ptr(pool_row) = {
                            src_rank, src_token, src_topk};
                    }
                    __syncwarp();
                }
            }
            __syncthreads();
            if (thread_idx == 0)
                ptx::red_add_rel(
                    workspace.get_l1_arrival_count_ptr(g), 1u);
        }
    } else {
        // Consumer CTAs schedule the complete grouped-GEMM work list while
        // waiting only for the M block they need. This is the actual
        // producer/consumer overlap missing from the previous fused kernel.
        const uint32_t gemm_sm_idx = cta_idx - kNumPullCTAs;
        uint32_t current_iter = 0;
        uint32_t current_expert = 0;
        uint32_t current_block_begin = 0;
        while (true) {
            const uint32_t work_idx =
                current_iter++ * kNumGemmCTAs + gemm_sm_idx;
            if (work_idx >= total_work)
                break;

            while (current_expert < kNumExpertsPerRank) {
                const uint32_t group_m = dget_m(current_expert);
                const uint32_t group_block_begin =
                    dget_block_begin(current_expert);
                const uint32_t group_m_blocks =
                    math::ceil_div(group_m, kBlockM);
                const uint32_t group_block_end =
                    group_block_begin + group_m_blocks;
                if (work_idx < group_block_end * kNumNBlocks) {
                    current_block_begin = group_block_begin;
                    break;
                }
                ++current_expert;
            }
            if (current_expert >= kNumExpertsPerRank)
                break;

            const uint32_t group_m = dget_m(current_expert);
            const uint32_t group_m_blocks =
                math::ceil_div(group_m, kBlockM);
            const uint32_t in_group =
                work_idx - current_block_begin * kNumNBlocks;
            const uint32_t per_swizzle =
                kNumNBlocks * kMBlocksPerSwizzleGroup;
            const uint32_t swizzle_group = in_group / per_swizzle;
            const uint32_t first_m_block =
                swizzle_group * kMBlocksPerSwizzleGroup;
            const uint32_t in_swizzle = in_group % per_swizzle;
            const uint32_t m_blocks_in_swizzle = mega_moe::detail::umin(
                kMBlocksPerSwizzleGroup,
                group_m_blocks - first_m_block);
            const uint32_t m_block =
                first_m_block + in_swizzle % m_blocks_in_swizzle;
            const uint32_t n_block =
                in_swizzle / m_blocks_in_swizzle;
            const uint32_t global_m_block =
                current_block_begin + m_block;
            const uint32_t pool_base = global_m_block * kBlockM;
            const uint32_t valid_m = mega_moe::detail::umin(
                kBlockM, group_m - m_block * kBlockM);

            while (ptx::ld_acq(
                       workspace.get_l1_arrival_count_ptr(global_m_block))
                   < (kTripleBlockPull
                          ? kNumPullCTAs / 3
                          : (kPairedBlockPull
                                 ? kNumPullCTAs / 2 : kNumPullCTAs)));
            const bool use_replica =
                current_expert >= num_master_experts;
            const uint32_t weight_expert = use_replica
                ? current_expert - num_master_experts
                : current_expert;
            const auto* weight_base =
                use_replica ? gemm1_weight_replicas : gemm1_weight;
            const auto* weight_scale_base = use_replica
                ? gemm1_weight_scale_replicas : gemm1_weight_scale;
            mega_moe::fp8_channelwise::gemm1_swiglu_mma_block<Cfg, kSwiGLUOAI>(
                reinterpret_cast<const typename Cfg::ElementA*>(
                    token_pool +
                        static_cast<uint64_t>(pool_base) * kTokenBytes),
                scale_pool + pool_base,
                weight_base
                    + (static_cast<uint64_t>(weight_expert) * kGemm1N
                       + static_cast<uint64_t>(n_block) * kBlockN)
                        * kTokenBytes,
                weight_scale_base
                    + static_cast<uint64_t>(weight_expert) * kGemm1N
                    + static_cast<uint64_t>(n_block) * kBlockN,
                swiglu_output
                    + static_cast<uint64_t>(pool_base) * (kGemm1N / 2)
                    + static_cast<uint64_t>(n_block) * (kBlockN / 2),
                static_cast<int>(kGemm1N / 2),
                static_cast<int>(valid_m),
                static_cast<int>(kBlockM),
                static_cast<int>(kBlockN),
                static_cast<int>(kTokenBytes),
                static_cast<int>(thread_idx),
                reinterpret_cast<char*>(smem_buf),
                swiglu_limit, swiglu_alpha);
        }
    }

    if constexpr (kQuantizeOutput) {
        // Producers and consumers must rendezvous before the whole-row
        // quantizer can redistribute valid rows across the complete grid.
        comm::grid_sync<kNumCTAs, /*kGridSyncIndex=*/1,
                        /*kPollBackoffCycles=*/128>(
            workspace, cta_idx, thread_idx, block_sync);

        const uint32_t quant_warp = cta_idx * kNumWarps + warp_idx;
        constexpr uint32_t kQuantWarpStride = kNumCTAs * kNumWarps;
        uint32_t quant_expert = 0;
        for (uint32_t padded_row = quant_warp;
             padded_row < total_m_blocks * kBlockM;
             padded_row += kQuantWarpStride) {
            const uint32_t global_m_block = padded_row / kBlockM;
            const uint32_t row_in_block = padded_row % kBlockM;
            while (quant_expert < kNumExpertsPerRank) {
                const uint32_t begin = dget_block_begin(quant_expert);
                const uint32_t blocks =
                    math::ceil_div(dget_m(quant_expert), kBlockM);
                if (global_m_block < begin + blocks)
                    break;
                ++quant_expert;
            }
            if (quant_expert >= kNumExpertsPerRank)
                break;
            const uint32_t expert_block_begin =
                dget_block_begin(quant_expert);
            const uint32_t expert_row =
                (global_m_block - expert_block_begin) * kBlockM
                + row_in_block;
            if (expert_row < dget_m(quant_expert)) {
                if constexpr (kSwiGLUOAI && kGemm1N == 6144) {
                    mega_moe::fp8_channelwise::warp_quantize_bf16_row_cached<kGemm1N / 2>(
                        swiglu_output, quant_output, quant_scale,
                        padded_row, lane_idx);
                } else {
                    mega_moe::detail::warp_quantize_bf16_row<kGemm1N / 2>(
                        swiglu_output, quant_output, quant_scale,
                        padded_row, lane_idx);
                }
            }
        }
    }
}

template <
    uint32_t kNumRanks, uint32_t kNumCTAs, uint32_t kNumThreads,
    uint32_t kNumExperts, uint32_t kNumExpertsPerRank, uint32_t kNumTopk,
    uint32_t kBlockM, uint32_t kTokenBytes, uint32_t kGemm1N,
    uint32_t kBlockN, uint32_t kBlockK, uint32_t kWarpM,
    uint32_t kWarpN, uint32_t kStages, uint32_t kNumPullCTAs,
    bool kQuantizeOutput = false,
    bool kPairedBlockPull = false, bool kTripleBlockPull = false,
    bool kSwiGLUOAI = false>
__global__ __launch_bounds__(kNumThreads, 1) void
dispatch_gemm1_kernel(
        void* workspace_base,
        const layout::SymBuffer<kNumRanks> sym_buffer,
        const int64_t* __restrict__ input_topk_idx,
        const float* __restrict__ input_topk_weights,
        const uint8_t* __restrict__ input_token,
        const float* __restrict__ input_scale,
        uint8_t* __restrict__ token_pool,
        float* __restrict__ scale_pool,
        float* __restrict__ topk_weight_pool,
        const cutlass::float_e4m3_t* __restrict__ gemm1_weight,
        const float* __restrict__ gemm1_weight_scale,
        const cutlass::float_e4m3_t* __restrict__ gemm1_weight_replicas,
        const float* __restrict__ gemm1_weight_scale_replicas,
        const uint32_t num_master_experts,
        __ppu_bfloat16* __restrict__ swiglu_output,
        cutlass::float_e4m3_t* __restrict__ quant_output,
        float* __restrict__ quant_scale,
        const float swiglu_limit,
        const float swiglu_alpha,
        const uint32_t num_tokens,
        const uint32_t num_max_tokens_per_rank) {
    dispatch_gemm1_device<kNumRanks, kNumCTAs, kNumThreads, kNumExperts, kNumExpertsPerRank, kNumTopk, kBlockM, kTokenBytes, kGemm1N, kBlockN, kBlockK, kWarpM, kWarpN, kStages, kNumPullCTAs, kQuantizeOutput, kPairedBlockPull, kTripleBlockPull, kSwiGLUOAI>(workspace_base, sym_buffer, input_topk_idx, input_topk_weights, input_token, input_scale, token_pool, scale_pool, topk_weight_pool, gemm1_weight, gemm1_weight_scale, gemm1_weight_replicas, gemm1_weight_scale_replicas, num_master_experts, swiglu_output, quant_output, quant_scale, swiglu_limit, swiglu_alpha, num_tokens, num_max_tokens_per_rank);
}

}  // namespace deep_gemm::mega_moe::fp8_channelwise
