#pragma once

// Channelwise FP8 fused MegaMoE; shared routing is independent of GEMM dtype/layout.
#include <deep_gemm/impls/mega_moe_transport.cuh>
#include <deep_gemm/scheduler/mega_moe_work_queue.cuh>
#include <deep_gemm/scheduler/sm89_mega_moe.cuh>
#include <deep_gemm/impls/fp8_channelwise_mega_moe_gemm1_mma.cuh>

namespace deep_gemm::mega_moe::fp8_channelwise {

// Cache the BF16 pairs loaded for amax and reuse them for FP8 conversion.
// Complete 512-element warp iterations keep every cached vector in bounds.
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
        const uint4 vectors[2] = {
            *reinterpret_cast<const uint4*>(src + elem),
            *reinterpret_cast<const uint4*>(src + elem + 8),
        };
        const auto* src2 = reinterpret_cast<const __ppu_bfloat162*>(vectors);
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

    // Use the same finite-scale bound as the uncached row quantizer.
    const float scale = fmaxf(local_amax, 1e-10f) / 448.0f;
    const float scale_inv = 1.0f / scale;
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
            packed2[i] = __hg_cvt_float2_to_fp8x2(
                make_float2(value0, value1), __HG_SATFINITE, __HG_E4M3);
        }
        *reinterpret_cast<uint4*>(dst + elem) = packed;
    }
}

// Quantize one row using cached vectors or a predicated vector tail.
// Cached conversion requires complete 512-element warp iterations; the
// generic loader predicates the last 16-element vectors for narrower tails.
template <int kHidden>
CUTLASS_DEVICE void warp_quantize_bf16_row_common(
    const __ppu_bfloat16* input, cutlass::float_e4m3_t* output,
    float* scales, uint32_t row, uint32_t lane) {
    if constexpr (kHidden % 512 == 0)
        warp_quantize_bf16_row_cached<kHidden>(input, output, scales, row, lane);
    else
        mega_moe::detail::warp_quantize_bf16_row<kHidden>(input, output, scales, row, lane);
}

// Flatten valid expert rows for quantization, then map back to pool offsets.
// Quantized rows use the pool offsets described by the GEMM2 metadata.
template <uint32_t kEPR, uint32_t kWidth>
CUTLASS_DEVICE void fp8_mega_moe_quantize_valid_pool_kernel_device(
        const int32_t* metadata, const __ppu_bfloat16* input,
        cutlass::float_e4m3_t* output, float* scales) {
    static_assert(kEPR > 0 && kEPR <= 128);
    const layout::Gemm2Metadata meta(const_cast<int32_t*>(metadata), kEPR);
    constexpr uint32_t kChunks = (kEPR + 31) / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    uint32_t begin[kChunks], end[kChunks], source[kChunks];
    uint32_t total = 0;
    #pragma unroll
    for (uint32_t chunk = 0; chunk < kChunks; ++chunk) {
        const uint32_t expert = chunk * 32 + lane;
        const uint32_t rows = expert < kEPR ? meta.valid_rows[expert] : 0;
        source[chunk] = expert < kEPR ? meta.src_offsets[expert] : 0;
        uint32_t prefix = rows;
        #pragma unroll
        for (uint32_t off = 1; off < 32; off <<= 1) {
            const uint32_t value = __shfl_up_sync(0xffffffffu, prefix, off);
            if (lane >= off) prefix += value;
        }
        begin[chunk] = total + prefix - rows;
        end[chunk] = total + prefix;
        total += __shfl_sync(0xffffffffu, prefix, 31);
    }
    const auto at = [&](const uint32_t* values, uint32_t expert) {
        uint32_t value = 0;
        #pragma unroll
        for (uint32_t chunk = 0; chunk < kChunks; ++chunk)
            if (chunk == expert / 32) value = values[chunk];
        return __shfl_sync(0xffffffffu, value, expert % 32);
    };
    constexpr uint32_t search_steps = [] {
        uint32_t steps = 0;
        for (uint32_t n = kEPR; n; n >>= 1) ++steps;
        return steps;
    }();
    const uint32_t warps = gridDim.x * (blockDim.x / 32);
    // Sparse decode rows favor binary lookup. Dense prefill keeps a cursor
    // that visits each expert at most once per warp.
    const bool binary_search = total <= warps * kEPR / search_steps;
    uint32_t expert = 0;
    for (uint32_t row = warp; row < total; row += warps) {
        if (binary_search) {
            uint32_t lo = expert, hi = kEPR;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) / 2;
                if (row >= at(end, mid)) lo = mid + 1;
                else hi = mid;
            }
            expert = lo;
        } else {
            while (expert < kEPR && row >= at(end, expert)) ++expert;
        }
        if (expert >= kEPR) break;
        const uint32_t pool_row = at(source, expert) + row - at(begin, expert);
        warp_quantize_bf16_row_common<kWidth>(input, output, scales, pool_row, lane);
    }
}

template <
    uint32_t kNumRanks, uint32_t kNumCTAs, uint32_t kNumThreads,
    uint32_t kNumExperts, uint32_t kNumExpertsPerRank, uint32_t kNumTopk,
    uint32_t kBlockM, uint32_t kTokenBytes, uint32_t kGemm1N,
    uint32_t kBlockN, uint32_t kBlockK, uint32_t kWarpM,
    uint32_t kWarpN, uint32_t kStages, uint32_t kNumPullCTAs,
    bool kSwiGLUOAI = false, class Gemm1Backend = void,
    uint32_t kPullGroupSize = kNumPullCTAs / 3, uint32_t kGemm2BlockM = 0, bool kGemm2ValidRows = true>
CUTLASS_DEVICE void dispatch_gemm1_device(
        void* workspace_base,
        const layout::SymBuffer<>& sym_buffer,
        int64_t* __restrict__ input_topk_idx,
        const uint8_t* __restrict__ input_token,
        const float* __restrict__ input_scale,
        uint8_t* __restrict__ token_pool,
        float* __restrict__ scale_pool,
        int32_t* __restrict__ gemm2_metadata,
        uint4* __restrict__ gemm2_block_info,
        const cutlass::float_e4m3_t* __restrict__ gemm1_weight,
        const float* __restrict__ gemm1_weight_scale,
        const cutlass::float_e4m3_t* __restrict__ gemm1_weight_replicas,
        const float* __restrict__ gemm1_weight_scale_replicas,
        const uint32_t num_master_experts,
        __ppu_bfloat16* __restrict__ swiglu_output,
        const float swiglu_limit,
        const float activation_alpha,
        const uint32_t num_tokens,
        const uint32_t num_max_tokens_per_rank,
        const float weight_inv_scale = 1.0f, const float replica_inv_scale = 1.0f,
        const float situ_beta = 4.0f, const float situ_linear_beta = 25.0f) {
    using Cfg = mega_moe::fp8_channelwise::Gemm1MmaConfig<
        kBlockM, kBlockN, kBlockK, kWarpM, kWarpN, kStages>;
    static_assert(kNumExpertsPerRank > 0 && kNumExpertsPerRank <= 128,
                  "register-distributed scheduler supports up to 128 physical "
                  "experts per rank");
    static_assert(kGemm1N % kBlockN == 0 && kBlockN % 16 == 0,
                  "GEMM1 N must be tiled by complete interleave groups");
    static_assert(kTokenBytes % kBlockK == 0,
                  "GEMM1 K must be divisible by BLOCK_K");
    static_assert(kNumThreads == (kBlockM / kWarpM) * (kBlockN / kWarpN) * 32,
                  "CTA threads must match the GEMM TiledMMA");
    static_assert(kNumPullCTAs > 0 && kNumPullCTAs < kNumCTAs,
                  "overlap requires separate pull and GEMM CTAs");
    static_assert(kPullGroupSize > 0 && kNumPullCTAs % kPullGroupSize == 0,
                  "pull groups must have equal producer counts");

    extern __shared__ __align__(16) unsigned char smem_buf[];
    const uint32_t cta_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = thread_idx / 32;
    const uint32_t lane_idx = thread_idx % 32;
    constexpr uint32_t kNumWarps = kNumThreads / 32;
    const layout::Workspace workspace(
        workspace_base, kNumRanks, kNumExperts, num_max_tokens_per_rank,
        kNumTopk);
    const auto block_sync = [=]() { __syncthreads(); };
    auto* smem_expert_count = reinterpret_cast<uint32_t*>(smem_buf);

    if (cta_idx == 0 && thread_idx == 0)
        *workspace.get_gemm1_work_count_ptr() = 0;
    for (uint32_t g = cta_idx * kNumThreads + thread_idx;
         g < workspace.num_max_pool_tokens / kBlockM; g += kNumCTAs * kNumThreads)
        *workspace.get_l1_arrival_count_ptr(g) = 0;

    mega_moe::detail::dispatch_handshake<
        kNumRanks, kNumCTAs, kNumThreads, kNumExperts, kNumExpertsPerRank,
        kNumTopk, cute::is_same_v<Gemm1Backend, void>>(
        workspace, sym_buffer, input_topk_idx, smem_expert_count, num_tokens,
        cta_idx, thread_idx);

    const sched::DistributedMegaMoEBlocks<kNumExpertsPerRank, kBlockM> blocks(
        workspace.get_expert_recv_count_sum_ptr(), lane_idx);
    const layout::Gemm2Metadata gemm2_meta(
        gemm2_metadata, kNumExpertsPerRank);
    if (cta_idx == 0 && warp_idx == 0)
        gemm2_meta.store_expert_layout<kNumExpertsPerRank, kBlockM, kGemm2BlockM, kGemm2ValidRows>(
            blocks, lane_idx, gemm2_block_info);
    const uint32_t total_m_blocks = blocks.total_blocks;
    constexpr uint32_t kNumNBlocks = kGemm1N / kBlockN;

    auto* queue_counter = workspace.get_gemm1_work_count_ptr();
    // Mainloop returns after its final CTA barrier; its storage is free
    // until the next tile. The register-only activation leaves this word intact.
    // The scale copy occupies the first M*sizeof(float) bytes of one A stage.
    // Keep the ticket outside that range. MMA can overwrite it only after all
    // threads consumed its value and passed the epilogue rendezvous.
    static_assert(kBlockM * sizeof(float) + sizeof(uint32_t) <= kBlockM * kBlockK);
    auto& queue_ticket = *reinterpret_cast<uint32_t*>(smem_buf + kBlockM * sizeof(float));
    const mega_moe::WorkQueue queue{queue_counter, &queue_ticket};

    struct BlockInfo {
        uint32_t expert;
        uint32_t expert_m;
        uint32_t expert_block_begin;
        uint32_t m_block;
        uint32_t valid_m;
        uint32_t pool_base;
    };

    const auto resolve_m_block = [&](uint32_t g) {
        uint4 info;
        blocks.get_block(g, info);
        const uint32_t expert = info.x, group_m = info.y, group_block_begin = info.z;
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

    // Pull teams process interleaved M blocks. CTAs in each team partition
    // the block's rows, with one shared token slot per warp. Publish arrivals
    // only after all local stores and metadata writes are complete.
    if (cta_idx < kNumPullCTAs) {
        constexpr uint32_t kNumPullTeams = kNumPullCTAs / kPullGroupSize;
        constexpr uint32_t kPullTeamSize = kPullGroupSize;
        const uint32_t pull_team = cta_idx / kPullTeamSize;
        const uint32_t team_cta_idx = cta_idx % kPullTeamSize;
        uint8_t* smem_token =
            smem_buf + static_cast<uint64_t>(warp_idx) * kTokenBytes;
        for (uint32_t g = pull_team; g < total_m_blocks;
             g += kNumPullTeams) {
            const BlockInfo bi = resolve_m_block(g);

            const mega_moe::detail::WarpSourceRanges<kNumRanks> sources(
                workspace, bi.expert, lane_idx);

            const uint32_t rows_per_cta = bi.valid_m / kPullTeamSize;
            const uint32_t extra_rows = bi.valid_m % kPullTeamSize;
            const uint32_t my_begin =
                team_cta_idx * rows_per_cta
                + mega_moe::detail::umin(team_cta_idx, extra_rows);
            const uint32_t my_rows =
                rows_per_cta + (team_cta_idx < extra_rows ? 1u : 0u);
            const uint32_t num_batches =
                math::ceil_div(my_rows, kNumWarps);

            for (uint32_t batch = 0; batch < num_batches; ++batch) {
                const uint32_t row_in_block =
                    my_begin + batch * kNumWarps + warp_idx;
                if (row_in_block < my_begin + my_rows) {
                    const uint32_t row_in_expert =
                        bi.m_block * kBlockM + row_in_block;
                    const auto source = sources.resolve(row_in_expert, lane_idx);
                    const uint32_t src_rank = source.rank, src_slot = source.slot;
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
                    float token_scale = 0.0f;
                    const uint32_t pool_row = bi.pool_base + row_in_block;
                    auto* pool_token_row = token_pool + uint64_t(pool_row) * kTokenBytes;
                    mega_moe::detail::warp_async_copy_global_to_shared(
                        smem_token, src_token_ptr, kTokenBytes, lane_idx);
                    if (lane_idx == 0) {
                        token_scale = *sym_buffer.map(input_scale + src_token, src_rank);
                    }
                    asm volatile("" ::: "memory");
                    cute::cp_async_wait<0>();
                    __syncwarp();
                    mega_moe::detail::warp_copy_bytes(
                        pool_token_row, smem_token, kTokenBytes, lane_idx);
                    if (lane_idx == 0) {
                        scale_pool[pool_row] = token_scale;
                        *workspace.get_token_src_metadata_ptr(pool_row) = {
                            src_rank, src_token, src_topk};
                        gemm2_meta.store_route(pool_row, src_rank, src_token, src_topk);
                    }
                    __syncwarp();
                }
            }
            __syncthreads();
            if (thread_idx == 0)
                tix::red_add_rel(
                    workspace.get_l1_arrival_count_ptr(g), 1u);
        }
    }
    {
        mega_moe::ArrivalCache ready;
        queue.prime(thread_idx);
        uint32_t prefetched_work = queue.current();
        __syncthreads(); // first claim only: every warp retires the shared word
        // Consumer CTAs schedule the complete grouped-GEMM work list while
        // waiting only for the M block they need, overlapping data transport
        // with GEMM execution.
        sched::MegaMoEGemm1Scheduler scheduler;
        uint32_t m_block, n_block;
        while (scheduler.select_work<kBlockM, kNumNBlocks>(
                prefetched_work, m_block, n_block, blocks)) {
            const uint32_t current_expert = scheduler.curr_group_idx;
            const uint32_t pool_base = scheduler.curr_global_block_m_idx * kBlockM;
            const uint32_t valid_m = mega_moe::detail::umin(
                kBlockM, scheduler.curr_group_m - m_block * kBlockM);
            const uint32_t global_m_block = scheduler.curr_global_block_m_idx;

            ready.wait(workspace.get_l1_arrival_count_ptr(0), global_m_block, global_m_block,
                       total_m_blocks, kPullGroupSize);

            const bool use_replica =
                current_expert >= num_master_experts;
            const uint32_t weight_expert = use_replica
                ? current_expert - num_master_experts
                : current_expert;
            const auto* weight_base =
                use_replica ? gemm1_weight_replicas : gemm1_weight;
            const auto* weight_scale_base = use_replica
                ? gemm1_weight_scale_replicas : gemm1_weight_scale;
            if constexpr (!cute::is_same_v<Gemm1Backend, void>) {
                Gemm1Backend::run(
                    token_pool + uint64_t(pool_base) * kTokenBytes, scale_pool + pool_base,
                    reinterpret_cast<const uint8_t*>(weight_base)
                        + (uint64_t(weight_expert) * kGemm1N + n_block * kBlockN) * (kTokenBytes / 2),
                    reinterpret_cast<const uint8_t*>(weight_scale_base)
                        + (uint64_t(weight_expert) * kGemm1N + n_block * kBlockN) * (kTokenBytes / 32),
                    swiglu_output + uint64_t(pool_base) * (kGemm1N / 2) + n_block * (kBlockN / 2),
                    kGemm1N / 2, valid_m, reinterpret_cast<char*>(smem_buf),
                    swiglu_limit, activation_alpha, use_replica ? replica_inv_scale : weight_inv_scale,
                    situ_beta, situ_linear_beta,
                    queue_counter, &queue_ticket);
                prefetched_work = queue.current();
            } else {
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
                    swiglu_limit, activation_alpha,
                    queue_counter, &queue_ticket, &prefetched_work);
            }
        }
    }

}

}  // namespace deep_gemm::mega_moe::fp8_channelwise

namespace deep_gemm::mega_moe::fp8_channelwise {

template <int Experts, int Width, int Threads, uint64_t MetadataOffset>
__global__
__launch_bounds__(Threads, 1) void sm89_fp8_mega_moe_quantize_local_pool(int64_t sb_base, const __ppu_bfloat16* input,
                                                                         cutlass::float_e4m3_t* output, float* scales) {
    const auto* metadata = reinterpret_cast<const int32_t*>(sb_base + MetadataOffset);
    mega_moe::fp8_channelwise::fp8_mega_moe_quantize_valid_pool_kernel_device<Experts, (Width * 2) / 2>(metadata, input,
                                                                                                        output, scales);
}

} // namespace deep_gemm::mega_moe::fp8_channelwise
