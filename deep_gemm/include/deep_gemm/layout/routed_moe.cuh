#pragma once

// Symmetric MegaMoE workspace layout. Byte offsets describe counters,
// arrival counters, source-token metadata, dispatch pools and GEMM outputs.
#include <cstddef>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/mega_moe_common.cuh>

namespace deep_gemm::layout {

// PPU MegaMoE uses the same power-of-two M tiles as the PPU grouped FP4
// selector.  The pool is allocated once and reused by every runtime tile, so
// its capacity and block counters must cover this complete candidate set.
static constexpr int kNumCandidateBlockMs = 5;
static constexpr int kCandidateBlockM[kNumCandidateBlockMs] = {16, 32, 64, 128, 256};
static constexpr int kMaxCandidateBlockM = 256;
static constexpr int kMinCandidateBlockM = 16;
static constexpr int kLCMCandidateBlockM = 256;

// Pool capacity for shared expert token pool: worst-case total tokens + per-expert BLOCK_M alignment padding, among all possible BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_pool_tokens(T num_ranks, T num_max_tokens_per_rank, T num_topk,
                                                        T num_experts_per_rank) {
    const auto num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
    const auto num_max_experts_per_token = math::constexpr_min(num_topk, num_experts_per_rank);
    return math::constexpr_align(
        num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (static_cast<T>(kMaxCandidateBlockM) - 1),
        static_cast<T>(kLCMCandidateBlockM));
}

// Dispatch/GEMM1 publishes the GEMM2 views at its metadata base.
// GEMM1 and GEMM2 may use different expert padding.
struct Gemm2Metadata {
    int32_t *rows, *valid_rows, *offsets, *src_rows, *src_offsets;
    int64_t* packed_routes;

    CUTLASS_HOST_DEVICE static constexpr uint64_t rows_bytes(uint32_t experts) {
        return math::constexpr_align(uint64_t(5) * experts * sizeof(int32_t), uint64_t(16));
    }

    CUTLASS_HOST_DEVICE Gemm2Metadata(void* base, uint32_t experts) {
        rows = static_cast<int32_t*>(base);
        valid_rows = rows + experts;
        offsets = valid_rows + experts;
        src_rows = offsets + experts;
        src_offsets = src_rows + experts;
        packed_routes = math::advance_ptr<int64_t>(base, rows_bytes(experts));
    }

    CUTLASS_DEVICE void store_rows(uint32_t expert, uint32_t valid, uint32_t dst_m,
                                   uint32_t dst_offset, uint32_t src_m, uint32_t src_offset) const {
        rows[expert] = dst_m;
        valid_rows[expert] = valid;
        offsets[expert] = dst_offset;
        src_rows[expert] = src_m;
        src_offsets[expert] = src_offset;
    }

#if defined(__HGGC__)
    // Called by one complete warp after the dispatch histogram is available.
    // GEMM2 consumes this metadata after GEMM1 on the same stream. Its task
    // list carries physical pool offsets even when it schedules only valid rows.
    template <uint32_t Experts, uint32_t BlockM, uint32_t Gemm2BlockM = 0,
              bool UseValidRows = true, typename Blocks>
    CUTLASS_DEVICE void store_expert_layout(const Blocks& blocks, uint32_t lane,
                                           uint4* block_info = nullptr) const {
        uint32_t base = 0;
        #pragma unroll
        for (uint32_t chunk = 0; chunk < math::ceil_div(Experts, 32u); ++chunk) {
            const uint32_t expert = chunk * 32 + lane;
            const uint32_t padded = (blocks.end[chunk] - blocks.begin[chunk]) * BlockM;
            const uint32_t offset = blocks.begin[chunk] * BlockM;
            if (expert < Experts)
                store_rows(expert, blocks.rows[chunk], padded, offset, padded, offset);
            if constexpr (Gemm2BlockM > 0) {
                const uint32_t rows = UseValidRows ? blocks.rows[chunk] : padded;
                const uint32_t count = math::ceil_div(rows, Gemm2BlockM);
                uint32_t end = count;
                #pragma unroll
                for (uint32_t step = 1; step < 32; step <<= 1) {
                    const uint32_t value = __shfl_up_sync(0xffffffffu, end, step);
                    if (lane >= step) end += value;
                }
                const uint32_t begin = base + end - count;
                for (uint32_t i = 0; i < count; ++i)
                    block_info[1 + begin + i] = make_uint4(expert, rows, begin, offset);
                base += __shfl_sync(0xffffffffu, end, 31);
            }
        }
        if constexpr (Gemm2BlockM > 0)
            if (lane == 0) block_info[0] = make_uint4(base, 0, 0, 0);
    }
#endif

    CUTLASS_DEVICE void store_route(uint32_t row, uint32_t rank, uint32_t token, uint32_t slot) const {
        packed_routes[row] = (uint64_t(rank) << 32) | (uint64_t(slot) << 24) | token;
    }
};

// Allocation-owned state stays at the same addresses for every shape alias.
struct RoutedMoESignals {
    uint32_t grid[4], rank_counter;
    int rank_signals[2];
    alignas(16) int64_t peer_offsets[kNumMaxRanks];
    alignas(128) int32_t combine_flag;
    alignas(128) int32_t combine_done;
};

struct RoutedMoEStaging {
    uint64_t counts = 0, tokens = 0;

    CUTLASS_HOST_DEVICE static RoutedMoEStaging for_shape(uint32_t ranks, uint32_t experts, uint32_t capacity) {
        return {math::align<uint64_t>((uint64_t(2) * experts + experts / ranks) * sizeof(uint64_t), 256),
                math::align<uint64_t>(uint64_t(experts) * capacity * sizeof(uint32_t), 256)};
    }
};

struct Workspace {
    void* base;
    RoutedMoEStaging staging;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;

    // Pool capacity: all local experts share a contiguous token pool
    uint32_t num_max_pool_tokens;
    uint32_t num_max_pool_blocks;

    static constexpr uint64_t kNumSignalBytes = sizeof(RoutedMoESignals);

    CUTLASS_HOST_DEVICE
    Workspace(void* base,
              const uint32_t& num_ranks,
              const uint32_t& num_experts,
              const uint32_t& num_max_tokens_per_rank,
              const uint32_t& num_topk, RoutedMoEStaging staging = {}):
        base(base),
        staging(staging.counts ? staging : RoutedMoEStaging::for_shape(num_ranks, num_experts, num_max_tokens_per_rank)),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_pool_tokens = get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_max_pool_blocks = num_max_pool_tokens / kMinCandidateBlockM;
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;

        num_bytes += kNumSignalBytes;

        // Keep remote writes outside every alias's local compute/combine data.
        num_bytes += staging.counts + staging.tokens;

        // Per-block arrival counters; keep the following metadata 8-byte aligned.
        num_bytes += math::align(num_max_pool_blocks, 2u) * sizeof(uint32_t);

        // Source rank/token/top-k slots for scatter.
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

        // Rank-local G1 work queue, independent of per-block arrival counters.
        num_bytes += sizeof(uint32_t);

        // Align to TMA descriptor requirements
        num_bytes = math::align<uint64_t>(num_bytes, 16);
        return num_bytes;
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    template <uint32_t kIndex = 0>
    CUTLASS_HOST_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return static_cast<RoutedMoESignals*>(base)->grid + kIndex;
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return &static_cast<RoutedMoESignals*>(base)->rank_counter;
    }

    CUTLASS_HOST_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        // NOTES: the signal is signed, as we may minus
        return static_cast<RoutedMoESignals*>(base)->rank_signals + phase;
    }

    CUTLASS_HOST_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return math::advance_ptr<uint64_t>(base, kNumSignalBytes) + expert_idx;
    }

    CUTLASS_HOST_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts) + rank_idx * num_experts_per_rank + expert_idx;
    }

    CUTLASS_HOST_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts * 2) + expert_idx;
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_l1_arrival_count_ptr(const uint32_t& pool_block_idx = 0) const {
        return math::advance_ptr<uint32_t>(base, kNumSignalBytes + staging.counts + staging.tokens) + pool_block_idx;
    }

    // For dispatch pulling
    CUTLASS_HOST_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        return math::advance_ptr<uint32_t>(base, kNumSignalBytes + staging.counts) +
            (uint64_t(expert_idx) * num_ranks + rank_idx) * num_max_tokens_per_rank + token_idx;
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_gemm1_work_count_ptr() const {
        return reinterpret_cast<uint32_t*>(get_token_src_metadata_ptr(num_max_pool_tokens));
    }

    // For combine usages
    CUTLASS_HOST_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_l1_arrival_count_ptr(math::align(num_max_pool_blocks, 2u)));
        return base + pool_token_idx;
    }
};


// Shared dispatch prefix for host slicing and JIT argument generation.
struct RoutedMoEInputLayout {
    Workspace dispatch;
    uint64_t input_tokens, input_scales, topk_idx, topk_weights;
    uint64_t dispatched_tokens, dispatched_scales;
    uint64_t gemm2_metadata;

    CUTLASS_HOST_DEVICE
    RoutedMoEInputLayout(uint32_t ranks, uint32_t experts, uint32_t capacity, uint32_t topk,
                        uint64_t token_bytes, uint64_t scale_bytes, RoutedMoEStaging staging = {}):
        dispatch(nullptr, ranks, experts, capacity, topk, staging) {
        const uint64_t pool = dispatch.num_max_pool_tokens;
        const auto align16 = [](uint64_t n) { return math::align<uint64_t>(n, 16); };
        const auto align128 = [](uint64_t n) { return math::align<uint64_t>(n, 128); };
        input_tokens = align128(dispatch.get_num_bytes());
        input_scales = align16(input_tokens + capacity * token_bytes);
        topk_idx = align16(input_scales + capacity * scale_bytes);
        topk_weights = align16(topk_idx + uint64_t(capacity) * topk * sizeof(int64_t));
        dispatched_tokens = align128(topk_weights + uint64_t(capacity) * topk * sizeof(float));
        dispatched_scales = align16(dispatched_tokens + pool * token_bytes);
        gemm2_metadata = align16(dispatched_scales + pool * scale_bytes);
    }
};

// FP8 and BF16 dispatch share the communication prefix. Scale storage is
// precision-specific; all scratch and completion state belongs to this buffer.
struct RoutedMoEBuffer : RoutedMoEInputLayout {
    uint64_t packed_routes, l2_acts, l2_scales, bf16_scratch;
    uint64_t block_info, block_info_elements, peer_offsets;
    uint64_t combine, barrier_flag, barrier_done, num_bytes;

    CUTLASS_HOST_DEVICE
    RoutedMoEBuffer(uint32_t ranks, uint32_t experts, uint32_t capacity, uint32_t topk,
                    uint32_t hidden, uint32_t intermediate, bool fp8, bool blockwise, RoutedMoEStaging staging = {}):
        RoutedMoEInputLayout(ranks, experts, capacity, topk, uint64_t(hidden) * (fp8 ? 1 : 2),
                            fp8 ? (blockwise ? hidden / 128 : 1) * sizeof(float) : 0, staging) {
        const uint64_t pool = dispatch.num_max_pool_tokens, groups = experts / ranks;
        const auto align128 = [](uint64_t n) { return math::align<uint64_t>(n, 128); };
        packed_routes = gemm2_metadata + Gemm2Metadata::rows_bytes(groups);
        l2_acts = align128(packed_routes + pool * sizeof(int64_t));
        l2_scales = align128(l2_acts + pool * intermediate * (fp8 ? 1 : 2));
        bf16_scratch = align128(l2_scales + (fp8 ? pool * (blockwise ? intermediate / 128 : 1) * sizeof(float) : 0));
        block_info = align128(bf16_scratch + (fp8 && !blockwise ? pool * intermediate * 2 : 0));
        block_info_elements = groups >= 128 ? 4 * (1 + groups + pool / kMinCandidateBlockM) : 0;
        peer_offsets = offsetof(RoutedMoESignals, peer_offsets);
        barrier_flag = offsetof(RoutedMoESignals, combine_flag);
        barrier_done = offsetof(RoutedMoESignals, combine_done);
        combine = align128(block_info + block_info_elements * sizeof(int32_t));
        num_bytes = align128(combine + uint64_t(topk) * capacity * hidden * 2);
    }

    CUTLASS_HOST_DEVICE uint64_t get_num_bytes() const { return num_bytes; }
};

} // namespace deep_gemm::layout
