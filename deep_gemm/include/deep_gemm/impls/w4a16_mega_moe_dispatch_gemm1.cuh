#pragma once
#include <deep_gemm/impls/mega_moe_transport.cuh>
#include <deep_gemm/impls/w4a16_gemm_cutlass3.cuh>

namespace deep_gemm::mega_moe::w4a16 {

// BF16 transport writes the same padded expert pool and route metadata used
// by SiTU GEMM1. Payloads are opaque bytes;
// no activation scale buffer is needed.
template <uint32_t Ranks, uint32_t CTAs, uint32_t Threads, uint32_t Experts, uint32_t ExpertsPerRank, uint32_t TopK,
          uint32_t BlockM, uint32_t PullSlots, uint32_t TokenBytes, uint32_t PullCTAs, uint32_t PullGroupSize>
CUTLASS_DEVICE void dispatch_tokens(void* workspace_base, const layout::SymBuffer<>& buffer,
                                    const int64_t* input_ids, const uint8_t* input_tokens, uint8_t* dispatched_tokens,
                                    void* metadata, uint32_t tokens, uint32_t capacity) {
    const uint32_t cta = blockIdx.x, thread = threadIdx.x, warp = thread / 32, lane = thread % 32;
    const layout::Workspace ws(workspace_base, Ranks, Experts, capacity, TopK);
    const layout::Gemm2Metadata meta(metadata, ExpertsPerRank);
    const auto block_sync = []() { __syncthreads(); };
    extern __shared__ __align__(16) unsigned char smem_buf[];
    if (cta == 0 && thread == 0)
        *ws.get_gemm1_work_count_ptr() = 0;
    for (uint32_t g = cta * Threads + thread; g < ws.num_max_pool_tokens / BlockM; g += CTAs * Threads)
        *ws.get_l1_arrival_count_ptr(g) = 0;
    mega_moe::detail::dispatch_handshake<Ranks, CTAs, Threads, Experts, ExpertsPerRank, TopK, true>(
        ws, buffer, input_ids, reinterpret_cast<uint32_t*>(smem_buf), tokens, cta, thread);
    constexpr uint32_t Chunks = (ExpertsPerRank + 31) / 32;
    uint32_t rows[Chunks], begins[Chunks], ends[Chunks];
    uint32_t total = 0;
#pragma unroll
    for (uint32_t chunk = 0; chunk < Chunks; ++chunk) {
        const uint32_t expert = chunk * 32 + lane;
        rows[chunk] = expert < ExpertsPerRank ? uint32_t(*ws.get_expert_recv_count_sum_ptr(expert)) : 0;
        const uint32_t blocks = math::ceil_div(rows[chunk], BlockM);
        uint32_t prefix = blocks;
#pragma unroll
        for (uint32_t offset = 1; offset < 32; offset *= 2) {
            const uint32_t value = __shfl_up_sync(0xffffffffu, prefix, offset);
            if (lane >= offset)
                prefix += value;
        }
        begins[chunk] = total + prefix - blocks;
        ends[chunk] = total + prefix;
        total += __shfl_sync(0xffffffffu, prefix, 31);
        if (cta == 0 && warp == 0 && expert < ExpertsPerRank) {
            const uint32_t padded = blocks * BlockM, base = begins[chunk] * BlockM;
            meta.store_rows(expert, rows[chunk], padded, base, padded, base);
        }
    }
    comm::grid_sync<CTAs, 1>(ws, cta, thread, block_sync);
    if (cta < PullCTAs) {
        static_assert(PullGroupSize > 0);
        constexpr uint32_t Teams = math::ceil_div(PullCTAs, PullGroupSize);
        const uint32_t team = cta / PullGroupSize, member = cta % PullGroupSize;
        const uint32_t members = cute::min(PullGroupSize, PullCTAs - team * PullGroupSize);
        for (uint32_t g = team; g < total; g += Teams) {
            uint32_t expert = 0, begin = 0, valid = 0;
#pragma unroll
            for (uint32_t chunk = 0; chunk < Chunks; ++chunk) {
                const bool owns = g >= begins[chunk] && g < ends[chunk];
                const uint32_t mask = __ballot_sync(0xffffffffu, owns);
                if (mask) {
                    const uint32_t owner = uint32_t(__ffs(mask) - 1);
                    expert = chunk * 32 + owner;
                    begin = __shfl_sync(0xffffffffu, begins[chunk], owner);
                    valid = __shfl_sync(0xffffffffu, rows[chunk], owner);
                }
            }
            const uint32_t row_start = (g - begin) * BlockM;
            const uint32_t count = mega_moe::detail::umin(BlockM, valid - row_start);
            const mega_moe::detail::WarpSourceRanges<Ranks> ranges(ws, expert, lane);
            if (warp < PullSlots) {
                auto* staged = smem_buf + uint64_t(warp) * TokenBytes;
                for (uint32_t row = member * PullSlots + warp; row < count; row += members * PullSlots) {
                    const auto source = ranges.resolve(row_start + row, lane);
                    uint32_t packed = 0;
                    if (lane == 0)
                        packed = *ws.get_src_token_topk_idx_ptr(expert, source.rank, source.slot);
                    packed = __shfl_sync(0xffffffffu, packed, 0);
                    const uint32_t token = packed / TopK, slot = packed % TopK, pool_row = g * BlockM + row;
                    const auto* src = buffer.map(input_tokens + uint64_t(token) * TokenBytes, source.rank);
                    mega_moe::detail::warp_async_copy_global_to_shared(staged, src, TokenBytes, lane);
                    cute::cp_async_wait<0>();
                    __syncwarp();
                    mega_moe::detail::warp_copy_bytes(dispatched_tokens + uint64_t(pool_row) * TokenBytes, staged,
                                                      TokenBytes, lane);
                    if (lane == 0)
                        meta.store_route(pool_row, source.rank, token, slot);
                    __syncwarp();
                }
            }
            __syncthreads();
#if __HGGC_ARCH__ == 150
            __ppu_wait_vstcnt(0);
#endif
            __threadfence();
            __syncthreads();
            if (thread == 0)
                // Complete a partial team's credit only after all of its
                // existing members have published their rows.
                tix::red_add_rel(ws.get_l1_arrival_count_ptr(g),
                    member + 1 == members ? PullGroupSize - members + 1 : 1u);
        }
    }
}

// BF16 dispatch and W4A16 GEMM1 share pool and metadata layouts.
// Dedicated pull CTAs publish
// each M block independently, then join consumers after all their own pulls.
template <uint32_t kNumRanks, uint32_t kNumCTAs, uint32_t kNumThreads, uint32_t kNumExperts,
          uint32_t kNumExpertsPerRank, uint32_t kNumTopk, uint32_t kDispatchBlockM,
          uint32_t kPullSlots, uint32_t kNumMaxTokensPerRank, uint32_t kShapeN, uint32_t kShapeK,
          int kW4BlockM, int kW4BlockN, int kW4BlockK, int kW4WarpM, int kW4WarpN,
          int kW4WarpK, int kW4Stages, uint32_t kNumPullCTAs, uint32_t kPullGroupSize>
CUTLASS_DEVICE void dispatch_gemm1_device(
    void* workspace_base, const layout::SymBuffer<>& sym_buffer, const int64_t* __restrict__ input_topk_idx,
    const uint8_t* __restrict__ input_token, uint8_t* __restrict__ dispatched_token_pool,
    void* __restrict__ gemm2_metadata, const uint32_t num_tokens, const uint8_t* __restrict__ gemm1_weight,
    const uint8_t* __restrict__ gemm1_weight_scale, const uint8_t* __restrict__ gemm1_weight_replicas,
    const uint8_t* __restrict__ gemm1_weight_scale_replicas, const uint32_t num_master_experts,
    __ppu_bfloat16* __restrict__ activated, const float situ_beta, const float situ_linear_beta,
    const uint32_t num_pool, const int rank) {
    extern __shared__ __align__(16) unsigned char smem_buf[];

    static_assert(kNumExpertsPerRank < 128, "fused W4A16 dispatch currently uses the direct no-pad scheduler");
    static_assert(kShapeK % 32 == 0, "W4A16 MXFP4 K must be divisible by 32");
    static_assert(kShapeN % 16 == 0, "fused SiTU needs complete G=8 gate/up pairs");

    static_assert(kNumPullCTAs > 0 && kNumPullCTAs < kNumCTAs);
    dispatch_tokens<kNumRanks, kNumCTAs, kNumThreads, kNumExperts, kNumExpertsPerRank, kNumTopk, kDispatchBlockM,
                    kPullSlots, kShapeK * sizeof(__ppu_bfloat16), kNumPullCTAs, kPullGroupSize>(
        workspace_base, sym_buffer, input_topk_idx, input_token, dispatched_token_pool, gemm2_metadata,
        num_tokens, kNumMaxTokensPerRank);

    const layout::Gemm2Metadata gemm2_meta(gemm2_metadata, kNumExpertsPerRank);

    using W4Kernel = cutlass::gemm::kernel::W4A16GEMM_MMA<kShapeN, kShapeK, kW4BlockM, kW4BlockN, kW4BlockK,
                                                          kW4WarpM, kW4WarpN, kW4WarpK, kNumExpertsPerRank,
                                                          kW4Stages, GemmType::MGroupedNoPad,
                                                          /*N_EXPAND=*/1,
                                                          /*kMoeScatter=*/false,
                                                          /*kFusedSitu=*/true, /*kUseReplicaWeights=*/false,
                                                          /*kDynamicWork=*/true>;
    static_assert(W4Kernel::MaxThreadsPerBlock == kNumThreads, "W4A16 GEMM phase requires the MMA CTA size");

    typename W4Kernel::Params params{reinterpret_cast<const cutlass::bfloat16_t*>(dispatched_token_pool),
                                     gemm1_weight, gemm1_weight_scale,
                                     reinterpret_cast<cutlass::bfloat16_t*>(activated)};
    // UltraEP routes use local physical slots: masters first, then replicas.
    params.ptr_b_replicas = gemm1_weight_replicas;
    params.ptr_scale_replicas = gemm1_weight_scale_replicas;
    params.num_master_groups = num_master_experts;
    params.scheduler = {num_pool, gemm2_meta.src_rows};
    params.valid_m_rows = gemm2_meta.valid_rows;
    params.situ_beta = situ_beta;
    params.situ_linear_beta = situ_linear_beta;
    const layout::Workspace ws(workspace_base, kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);
    params.queue_counter = ws.get_gemm1_work_count_ptr();
    params.arrival_counts = ws.get_l1_arrival_count_ptr(0);
    params.dispatch_block_m = kDispatchBlockM;
    params.dispatch_arrivals = kPullGroupSize;
    W4Kernel{}(params, reinterpret_cast<char*>(smem_buf));
}

} // namespace deep_gemm::mega_moe::w4a16
