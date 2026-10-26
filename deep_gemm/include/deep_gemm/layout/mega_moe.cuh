#pragma once

// Symmetric MegaMoE workspace layout. Byte offsets describe counters,
// arrival counters, source-token metadata, dispatch pools and GEMM outputs.
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

// Per-token source metadata for combine write-back
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;
};

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

    CUTLASS_DEVICE void store_route(uint32_t row, uint32_t rank, uint32_t token, uint32_t slot) const {
        packed_routes[row] = (uint64_t(rank) << 32) | (uint64_t(slot) << 24) | token;
    }
};

struct Workspace {
    void* base;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;

    // Pool capacity: all local experts share a contiguous token pool
    uint32_t num_max_pool_tokens;
    uint32_t num_max_pool_blocks;

    // For grid and cross-rank generation barriers
    static constexpr uint64_t kNumBarrierSignalBytes = 32;

    CUTLASS_HOST_DEVICE
    Workspace(void* base,
              const uint32_t& num_ranks,
              const uint32_t& num_experts,
              const uint32_t& num_max_tokens_per_rank,
              const uint32_t& num_topk):
        base(base),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_pool_tokens = get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_max_pool_blocks = num_max_pool_tokens / kMinCandidateBlockM;
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;

        // Barrier
        num_bytes += kNumBarrierSignalBytes;

        // Expert send/recv count
        num_bytes += num_experts * sizeof(uint64_t) * 2;

        // Expert recv count sum
        num_bytes += num_experts_per_rank * sizeof(uint64_t);

        // Per-block arrival counters; keep the following routing table 8-byte aligned.
        num_bytes += math::align(num_max_pool_blocks, 2u) * sizeof(uint32_t);

        // Each (expert, source rank) list contains at most one entry per source token.
        num_bytes += uint64_t(num_experts_per_rank) * num_ranks * num_max_tokens_per_rank * sizeof(uint32_t);

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

    // Grid sync counters: `kNumBarrierSignalBytes` layout
    // [ 0..15]: 4 x `uint32_t` grid sync counters
    // [16..20]: `uint32_t` cross-rank barrier counter
    // [20..27]: 2 x `int` cross-rank barrier signals (phase 0 and 1)
    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    template <uint32_t kIndex = 0>
    CUTLASS_HOST_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return static_cast<uint32_t*>(base) + kIndex;
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters;
    }

    CUTLASS_HOST_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        // NOTES: the signal is signed, as we may minus
        return math::advance_ptr<int>(base, (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) + phase * sizeof(int));
    }

    CUTLASS_HOST_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes) + expert_idx;
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
        const auto base = get_expert_recv_count_sum_ptr(num_experts_per_rank);
        return reinterpret_cast<uint32_t*>(base) + pool_block_idx;
    }

    // For dispatch pulling
    CUTLASS_HOST_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto base = get_l1_arrival_count_ptr(math::align(num_max_pool_blocks, 2u));
        return reinterpret_cast<uint32_t*>(base) +
            (uint64_t(expert_idx) * num_ranks + rank_idx) * num_max_tokens_per_rank + token_idx;
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_gemm1_work_count_ptr() const {
        return reinterpret_cast<uint32_t*>(get_token_src_metadata_ptr(num_max_pool_tokens));
    }

    // For combine usages
    CUTLASS_HOST_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }
};

// Complete W4A4 storage. Dispatch keeps its existing prefix; all subsequent
// regions share the same allocation and peer-base mapping. Pool capacity
// covers every supported M tile, independently of runtime configuration.
struct MegaMoEBuffer {
    Workspace dispatch;
    uint64_t input_tokens, input_scales, topk_idx, topk_weights;
    uint64_t dispatched_tokens, dispatched_scales, gemm2_metadata, packed_routes;
    uint64_t l2_acts, l2_scales, sfa_scratch, block_info, block_info_elements;
    uint64_t combine, barrier_flag, barrier_done, num_bytes;

    CUTLASS_HOST_DEVICE
    MegaMoEBuffer(uint32_t ranks, uint32_t experts, uint32_t capacity,
                  uint32_t topk, uint32_t hidden, uint32_t intermediate):
        dispatch(nullptr, ranks, experts, capacity, topk) {
        const uint64_t pool = dispatch.num_max_pool_tokens;
        const uint64_t groups = experts / ranks;
        const auto align128 = [](uint64_t bytes) { return math::align<uint64_t>(bytes, 128); };
        const auto align16 = [](uint64_t bytes) { return math::align<uint64_t>(bytes, 16); };
        input_tokens = align128(dispatch.get_num_bytes());
        input_scales = align16(input_tokens + uint64_t(capacity) * hidden / 2);
        topk_idx = align16(input_scales + uint64_t(capacity) * hidden / 32);
        topk_weights = align16(topk_idx + uint64_t(capacity) * topk * sizeof(int64_t));
        dispatched_tokens = align128(topk_weights + uint64_t(capacity) * topk * sizeof(float));
        dispatched_scales = align16(dispatched_tokens + pool * hidden / 2);
        gemm2_metadata = align16(dispatched_scales + pool * hidden / 32);
        packed_routes = gemm2_metadata + Gemm2Metadata::rows_bytes(groups);
        l2_acts = align128(packed_routes + pool * sizeof(int64_t));
        l2_scales = align128(l2_acts + pool * intermediate / 2);
        sfa_scratch = align128(l2_scales + pool * intermediate / 32);
        block_info = align128(sfa_scratch + pool * hidden / 32);
        block_info_elements = groups >= 128 ? 4 * (1 + groups + pool / kMinCandidateBlockM) : 0;
        combine = align128(block_info + block_info_elements * sizeof(int32_t));
        barrier_flag = align128(combine + uint64_t(topk) * capacity * hidden * sizeof(uint16_t));
        barrier_done = align128(barrier_flag + sizeof(int32_t));
        num_bytes = align128(barrier_done + sizeof(int32_t));
    }

    CUTLASS_HOST_DEVICE uint64_t get_num_bytes() const { return num_bytes; }
};

} // namespace deep_gemm::layout
