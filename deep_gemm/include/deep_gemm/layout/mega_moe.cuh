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

// Eight writers share one 32-bit atomic: four count bits plus eight mode bits.
constexpr uint32_t kScatterRanksPerBank = 8;
struct ScatterSignals {
    int32_t grid_counter;
    int32_t phase_counter;
    int32_t phase_signal[2][kNumMaxRanks / kScatterRanksPerBank];
    int32_t writer_previous[2];
};
static_assert(sizeof(ScatterSignals) == 80);

// Fixed protocol state shared by all views.
struct alignas(256) MegaMoESignals {
    static constexpr uint32_t kNumMaxExperts = 2048;
    // Full-pool BM16 flags, including 2048 experts at capacity 65536.
    static constexpr uint32_t kNumMaxCopyBlocks = 1u << 23;

    ScatterSignals scatter;
    alignas(16) uint32_t dispatch_retire[4];
    // Keep parity offsets independent of the active rank count.
    alignas(128) uint32_t dispatch_arrivals[2][kNumMaxRanks];
    alignas(256) uint32_t dispatch_counts[2][kNumMaxExperts];
    alignas(256) uint32_t preprocess_ready[1 + kNumMaxExperts * FusedSfaPartitionLayout::kNumPartitions];
};

static_assert(offsetof(MegaMoESignals, scatter) == 0);
static_assert(sizeof(MegaMoESignals) % 256 == 0);

// Byte ranges only. Tensor dtype/shape/stride belong to the host API.
struct Buffer {
    int64_t offset = 0, bytes = 0;

    template <typename T = uint8_t>
    __host__ __device__ T* data(void* base) const {
        return reinterpret_cast<T*>(static_cast<uint8_t*>(base) + offset);
    }
};

// Per-bank capacities inherited from the allocation owner.
struct MegaMoEStaging {
    int64_t scales = 0, counts = 0, tokens = 0;

    __host__ __device__
    static constexpr MegaMoEStaging for_shape(int ranks, int experts, int capacity, int hidden) {
        return {math::constexpr_align<int64_t>(int64_t(ranks) * capacity * (hidden / 64) * sizeof(uint16_t), 256),
                math::constexpr_align<int64_t>(int64_t(experts) * sizeof(uint32_t), 256),
                math::constexpr_align<int64_t>(int64_t(experts) * capacity * sizeof(uint32_t), 256)};
    }
};

struct MegaMoEBuffer {
    Buffer barrier_storage, copy_ready;
    Buffer dispatch_counts[2], dispatch_fp4, dispatch_retire, dispatch_arrivals[2];
    Buffer expert_base, expert_rank_prefix, grouped_layout, local_fp4, local_sfa, preprocess_ready;
    Buffer scatter_storage, src_info, l2_acts, l2_acts_sf;
    Buffer staging_scales[2], staging_counts[2], staging_tokens[2], topk_idx, topk_weights;
    MegaMoEStaging staging;
    int64_t bytes = 0, groups = 0, compact = 0, blocks = 0;

    __host__ __device__
    constexpr MegaMoEBuffer(int ranks, int experts, int capacity, int topk, int hidden, int intermediate,
                           MegaMoEStaging staging_capacity = {}) {
        groups = experts / ranks;
        int64_t owner = int64_t(ranks) * capacity;
        compact = owner * (topk < groups ? topk : groups);
        blocks = groups * ((owner + 15) / 16);
        bytes = sizeof(MegaMoESignals);
        barrier_storage = {offsetof(MegaMoESignals, scatter), sizeof(MegaMoESignals::scatter)};
        dispatch_retire = {offsetof(MegaMoESignals, dispatch_retire), sizeof(MegaMoESignals::dispatch_retire)};
        preprocess_ready = {offsetof(MegaMoESignals, preprocess_ready), sizeof(MegaMoESignals::preprocess_ready)};
        for (int p = 0; p < 2; ++p)
            dispatch_counts[p] = {static_cast<int64_t>(offsetof(MegaMoESignals, dispatch_counts) +
                p * sizeof(MegaMoESignals::dispatch_counts[0])), sizeof(MegaMoESignals::dispatch_counts[0])};
        for (int p = 0; p < 2; ++p)
            dispatch_arrivals[p] = {static_cast<int64_t>(offsetof(MegaMoESignals, dispatch_arrivals) +
                p * sizeof(MegaMoESignals::dispatch_arrivals[0])), sizeof(MegaMoESignals::dispatch_arrivals[0])};

        staging = staging_capacity.scales ? staging_capacity : MegaMoEStaging::for_shape(ranks, experts, capacity, hidden);
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

        dispatch_fp4 = allocate(int64_t(capacity) * (hidden / 2), 16);
        expert_base = allocate((groups + 1) * sizeof(int32_t));
        expert_rank_prefix = allocate(groups * ranks * sizeof(int32_t));
        grouped_layout = allocate(((1 + blocks) * 4 + groups) * sizeof(int32_t));
        local_fp4 = allocate(compact * (hidden / 2));
        const auto sfa_rows = FusedSfaPartitionLayout::row_offset(compact, groups);
        local_sfa = allocate(sfa_rows * (hidden / 64) * sizeof(uint16_t));
        scatter_storage = allocate(int64_t(topk) * capacity * hidden * sizeof(uint16_t));
        src_info = allocate(compact * sizeof(TokenSrcMetadata));
        l2_acts = allocate(compact * (intermediate / 2));
        l2_acts_sf = allocate((intermediate / 64) * compact * sizeof(uint16_t));
        topk_idx = allocate(int64_t(capacity) * topk * sizeof(int64_t));
        topk_weights = allocate(int64_t(capacity) * topk * sizeof(float));
        bytes = math::constexpr_align<int64_t>(bytes, 256);
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

}  // namespace deep_gemm::layout
