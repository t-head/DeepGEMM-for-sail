#pragma once

#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm::layout {

static constexpr int kInt8TokenAlignment = kTokenAlignment;

struct Int8Storage {
    __host__ __device__ static constexpr int64_t row_bytes(int64_t width) { return width; }
    __host__ __device__ static constexpr int64_t scale_bytes(int64_t) { return sizeof(float); }
    static constexpr uint32_t kScalePartitions = 1;
    static constexpr bool kTokenScaleBroadcast = true;
};

__host__ __device__
static constexpr MegaMoEStaging int8_staging_for_shape(int ranks, int experts, int capacity, int hidden) {
    return {math::constexpr_align<int64_t>(int64_t(ranks) * capacity * sizeof(float), 256),
            math::constexpr_align<int64_t>(int64_t(experts) * sizeof(uint32_t), 256),
            math::constexpr_align<int64_t>(int64_t(experts) * capacity * sizeof(uint32_t), 256)};
}

// The communication protocol and metadata are shared; only payload sizes differ.
struct Int8MegaMoEBuffer {
    using Storage = Int8Storage;
    Buffer copy_ready;
    Buffer dispatch_counts[2], dispatch_data, dispatch_retire;
    Buffer expert_base, expert_rank_prefix, grouped_layout, local_data, local_sfa, preprocess_ready;
    Buffer scatter_storage, src_info, stage1_data, stage1_scale_storage;
    Buffer staging_scales[2], staging_counts[2], staging_tokens[2], topk_idx, topk_weights;
    MegaMoEStaging staging;
    int64_t bytes = 0, groups = 0, compact = 0, blocks = 0;

    __host__ __device__ constexpr Buffer allocate(int64_t size, int64_t alignment = 256) {
        bytes = math::constexpr_align(bytes, alignment);
        Buffer result{bytes, size};
        bytes += size;
        return result;
    }

    __host__ __device__
    constexpr Int8MegaMoEBuffer(int ranks, int experts, int capacity, int topk, int hidden, int intermediate,
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
        staging = staging_capacity.scales ? staging_capacity :
            int8_staging_for_shape(ranks, experts, capacity, hidden);
        // Token capacity bounds the BM16 flags needed by every alias.
        const int64_t max_copy_blocks = staging.tokens / (16 * sizeof(uint32_t));
        copy_ready = allocate(max_copy_blocks * sizeof(uint32_t));

        for (auto& scales : staging_scales)
            scales = allocate(staging.scales);
        for (auto& counts : staging_counts)
            counts = allocate(staging.counts);
        for (auto& tokens : staging_tokens)
            tokens = allocate(staging.tokens);
        // Stage2 waits for all peers before the next round reuses the source rows.
        dispatch_data = allocate(int64_t(capacity) * Storage::row_bytes(hidden), 16);
        expert_base = allocate((groups + 1) * sizeof(int32_t));
        expert_rank_prefix = allocate(groups * ranks * sizeof(int32_t));
        grouped_layout = allocate(((1 + blocks) * 4 + groups) * sizeof(int32_t));
        local_data = allocate(compact * Storage::row_bytes(hidden));
        local_sfa = allocate(compact * Storage::scale_bytes(hidden));
        scatter_storage = allocate(int64_t(topk) * capacity * hidden * sizeof(uint16_t));
        src_info = allocate(compact * sizeof(TokenSrcMetadata));
        stage1_data = allocate(compact * Storage::row_bytes(intermediate));
        stage1_scale_storage = allocate(compact * Storage::scale_bytes(intermediate));
        topk_idx = allocate(int64_t(capacity) * topk * sizeof(int64_t));
        topk_weights = allocate(int64_t(capacity) * topk * sizeof(float));
        bytes = math::constexpr_align<int64_t>(bytes, 256);
    }

    int64_t get_num_bytes() const { return bytes; }
};

} // namespace deep_gemm::layout
