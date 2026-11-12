#pragma once

#include <cstddef>
#include <cstdint>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/layout/dispatch_layout.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

namespace deep_gemm::layout {

static constexpr int kTokenAlignment = 16;

// Source token and route used by GEMM2 scatter.
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;
};
static_assert(sizeof(TokenSrcMetadata) == 12);

// Fixed protocol state shared by all views.
struct alignas(256) MegaMoESignals {
    static constexpr uint32_t kNumMaxExperts = 2048;
    // Full-pool BM16 flags, including 2048 experts at capacity 65536.
    static constexpr uint32_t kNumMaxCopyBlocks = 1u << 23;

    uint32_t grid_sync_count;
    uint64_t scatter_writer_mask;  // Passed from the standalone scatter wait to reduce.
    alignas(16) uint32_t dispatch_retire[4];
    // Each forward publishes dispatch, then scatter, before either slot is reused.
    alignas(128) uint32_t dispatch_arrivals[kNumMaxRanks];
    alignas(128) uint32_t scatter_arrivals[kNumMaxRanks];
    alignas(256) uint32_t dispatch_counts[2][kNumMaxExperts];
    alignas(256) uint32_t preprocess_ready[1 + kNumMaxExperts * FusedSfaPartitionLayout::kNumPartitions];
};

static_assert(offsetof(MegaMoESignals, dispatch_arrivals) == 128);
static_assert(sizeof(MegaMoESignals) % 256 == 0);

// Byte ranges only. Tensor dtype/shape/stride belong to the host API.
struct Buffer {
    int64_t offset = 0, bytes = 0;

    template <typename T = uint8_t>
    __host__ __device__ T* data(void* base) const {
        return reinterpret_cast<T*>(static_cast<uint8_t*>(base) + offset);
    }
};

// Valid routes for a token name distinct experts, including within one rank.
__host__ __device__ constexpr int64_t max_local_routed_rows(int64_t source_tokens, int topk, int64_t local_experts) {
    return source_tokens * (topk < local_experts ? topk : local_experts);
}

template <bool Int8>
struct MegaMoEStorage {
    __host__ __device__ static constexpr int64_t row_bytes(int64_t width) {
        return Int8 ? width : width / 2;
    }
    __host__ __device__ static constexpr int64_t scale_bytes(int64_t width) {
        return Int8 ? sizeof(float) : (width / 64) * sizeof(uint16_t);
    }
};

using Int8Storage = MegaMoEStorage<true>;

// Per-bank capacities inherited from the allocation owner.
struct MegaMoEStaging {
    int64_t scales = 0, counts = 0, tokens = 0;

    template <bool Int8 = false>
    __host__ __device__
    static constexpr MegaMoEStaging for_shape(int ranks, int experts, int capacity, int hidden) {
        return {math::constexpr_align<int64_t>(int64_t(ranks) * capacity * MegaMoEStorage<Int8>::scale_bytes(hidden), 256),
                math::constexpr_align<int64_t>(int64_t(experts) * sizeof(uint32_t), 256),
                math::constexpr_align<int64_t>(int64_t(experts) * capacity * sizeof(uint32_t), 256)};
    }
};

template <bool Int8>
struct MegaMoEBufferLayout {
    using Storage = MegaMoEStorage<Int8>;
    Buffer copy_ready;
    Buffer dispatch_counts[2], dispatch_data, dispatch_retire;
    Buffer expert_base, expert_rank_prefix, grouped_layout, local_data, local_sfa, preprocess_ready;
    Buffer scatter_storage, src_info, l2_acts, l2_acts_sf, activated;
    Buffer staging_scales[2], staging_counts[2], staging_tokens[2], topk_idx, topk_weights;
    MegaMoEStaging staging;
    int64_t bytes = 0, groups = 0, compact = 0, blocks = 0;

    __host__ __device__
    constexpr MegaMoEBufferLayout(int ranks, int experts, int capacity, int topk, int hidden, int intermediate,
                                 MegaMoEStaging staging_capacity = {}) {
        groups = experts / ranks;
        int64_t owner = int64_t(ranks) * capacity;
        compact = max_local_routed_rows(owner, topk, groups);
        blocks = groups * ((owner + 15) / 16);
        bytes = sizeof(MegaMoESignals);
        dispatch_retire = {offsetof(MegaMoESignals, dispatch_retire), sizeof(MegaMoESignals::dispatch_retire)};
        preprocess_ready = {offsetof(MegaMoESignals, preprocess_ready), sizeof(MegaMoESignals::preprocess_ready)};
        for (int p = 0; p < 2; ++p)
            dispatch_counts[p] = {static_cast<int64_t>(offsetof(MegaMoESignals, dispatch_counts) +
                p * sizeof(MegaMoESignals::dispatch_counts[0])), sizeof(MegaMoESignals::dispatch_counts[0])};

        staging = staging_capacity.scales ? staging_capacity : MegaMoEStaging::for_shape<Int8>(ranks, experts, capacity, hidden);
        // Token capacity bounds the full-pool BM16 flags for every alias.
        const int64_t max_copy_blocks = staging.tokens / (16 * sizeof(uint32_t));
        copy_ready = allocate(max_copy_blocks * sizeof(uint32_t));

        // Keep remote staging writes separate from every view's reduce inputs.
        for (auto& scales : staging_scales)
            scales = allocate(staging.scales);
        for (auto& counts : staging_counts)
            counts = allocate(staging.counts);
        for (auto& tokens : staging_tokens)
            tokens = allocate(staging.tokens);

        dispatch_data = allocate(int64_t(capacity) * Storage::row_bytes(hidden), 16);
        expert_base = allocate((groups + 1) * sizeof(int32_t));
        expert_rank_prefix = allocate(groups * ranks * sizeof(int32_t));
        grouped_layout = allocate(((1 + blocks) * 4 + groups) * sizeof(int32_t));
        local_data = allocate(compact * Storage::row_bytes(hidden));
        const auto sfa_rows = Int8 ? compact : FusedSfaPartitionLayout::row_offset(compact, groups);
        local_sfa = allocate(sfa_rows * Storage::scale_bytes(hidden));
        scatter_storage = allocate(int64_t(topk) * capacity * hidden * sizeof(uint16_t));
        src_info = allocate(compact * sizeof(TokenSrcMetadata));
        l2_acts = allocate(compact * Storage::row_bytes(intermediate));
        l2_acts_sf = allocate(compact * Storage::scale_bytes(intermediate));
        topk_idx = allocate(int64_t(capacity) * topk * sizeof(int64_t));
        topk_weights = allocate(int64_t(capacity) * topk * sizeof(float));
        bytes = math::constexpr_align<int64_t>(bytes, 256);
        // INT8 quantizes the BF16 activation rows after GEMM1.
        if constexpr (Int8)
            activated = allocate(compact * intermediate * sizeof(uint16_t));
    }

    int64_t get_num_bytes() const { return bytes; }

private:
    __host__ __device__ constexpr Buffer allocate(int64_t size, int64_t alignment = 256) {
        bytes = math::constexpr_align(bytes, alignment);
        Buffer result{bytes, size};
        bytes += size;
        return result;
    }
};

using MegaMoEBuffer = MegaMoEBufferLayout<false>;
using Int8MegaMoEBuffer = MegaMoEBufferLayout<true>;

}  // namespace deep_gemm::layout
