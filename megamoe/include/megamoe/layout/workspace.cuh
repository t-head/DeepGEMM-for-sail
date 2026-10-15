#pragma once

#include <megamoe/common/base.h>

#include <megamoe/common/exception.cuh>
#include <megamoe/common/math.cuh>

namespace megamoe::layout {

struct TokenSrcMetadata {
    uint32_t source_rank_idx;
    uint32_t source_token_idx;
    uint32_t topk_idx;
};

struct Workspace {
    void* base;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;

    static constexpr uint64_t kNumControlHeaderBytes = 32;

    DG_HOST_DEVICE
    Workspace(void* base, const uint32_t& num_ranks, const uint32_t& num_experts)
        : base(base), num_ranks(num_ranks), num_experts(num_experts) {
        num_experts_per_rank = num_experts / num_ranks;
    }

    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    template <uint32_t kIndex = 0>
    DG_DEVICE uint32_t* get_grid_sync_counter_ptr() const {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return static_cast<uint32_t*>(base) + kIndex;
    }

    DG_DEVICE
    uint32_t* get_icn_barrier_counter_ptr() const { return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters; }

    DG_DEVICE
    int* get_icn_barrier_signal_ptr(const uint32_t& phase) const {
        return math::advance_ptr<int>(base, (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) + phase * sizeof(int));
    }

    DG_DEVICE
    // Slots 0-3 are grid sync counters, slot 4 the ICN barrier counter and slots
    // 5-6 the ICN barrier signals; the dispatch arrival count follows at slot 7. It is
    // zero-initialized with the workspace and re-zeroed by the combine-phase
    // reset at the end of every iteration.
    uint32_t* get_dispatch_arrival_count_ptr() const { return static_cast<uint32_t*>(base) + 7; }

    DG_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return math::advance_ptr<uint64_t>(base, kNumControlHeaderBytes) + expert_idx;
    }

    DG_DEVICE
    uint64_t* get_expert_recv_count_ptr(const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts) + rank_idx * num_experts_per_rank + expert_idx;
    }

    DG_DEVICE
    // Address space only: the drain's arrival counter is get_dispatch_arrival_count_ptr, and the extent of
    // this reserved per-expert area is what positions get_dispatch_slot_base_ptr below, so its size stays
    // part of the workspace prefix (kWorkspacePrefixBytes).
    uint64_t* get_reserved_expert_count_region_ptr(const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts * 2) + expert_idx;
    }

    DG_DEVICE
    // Base of the dispatch slot region. The name and the zero-argument signature match what the
    // kernels address (the layout's region pointers, not a per-source index).
    //
    // The base is rounded up to 128B, because a slot line is the dispatch protocol's 128B-atomic unit
    // and a line store reaches the fabric as whole 128B requests only when the line starts on a 128B
    // boundary. A store to a non-128B-aligned address is instead split at the 128B line boundary and
    // its pieces can be delivered in either order, which would let a line's flag word (its last word)
    // become visible while the rest of that line still holds the previous launch's payload -- the one
    // thing the per-line certification assumes cannot happen. The raw prefix (the control header, the
    // two per-expert count tables and the reserved expert-count region) is a multiple of 8 and not of 128, so this
    // rounding is what actually puts a slot line on the boundary; the host pads the same prefix to
    // 128B (workspace._get_workspace_region_bytes), so the rounded base is also the offset the host
    // sized the data region for.
    uint32_t* get_dispatch_slot_base_ptr() const {
        auto* const slot_base_bytes = reinterpret_cast<uint8_t*>(get_reserved_expert_count_region_ptr(num_experts_per_rank));
        const uint64_t offset = static_cast<uint64_t>(slot_base_bytes - static_cast<uint8_t*>(base));
        return reinterpret_cast<uint32_t*>(math::advance_ptr<uint8_t>(base, math::align_up<uint64_t>(offset, 128ull)));
    }
};

namespace detail {

DG_HOST_DEVICE void fail_data_aiu_alignment() {
    DG_UNIFIED_ASSERT(false and "Data size must satisfy AIU alignment");
}

}  // namespace detail

struct DataLayout {
    uint32_t num_bytes;
    bool require_aiu_alignment;
    void* base;

    DG_HOST_DEVICE
    constexpr explicit DataLayout(const uint32_t& num_bytes, const bool& require_aiu_alignment = true, void* base = nullptr)
        : num_bytes(num_bytes), require_aiu_alignment(require_aiu_alignment), base(base) {
        if (require_aiu_alignment and num_bytes % 16 != 0)
            detail::fail_data_aiu_alignment();
    }

    template <typename T = uint32_t>
    DG_HOST_DEVICE constexpr T get_num_bytes() const {
        return static_cast<T>(num_bytes);
    }

    template <typename dtype_t = void>
    DG_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }
};

struct Buffer {
    DataLayout data_layout;
    uint32_t num_ranks;
    uint32_t num_rows_per_rank;

    void* base;

    DG_HOST_DEVICE
    Buffer(const DataLayout& data_layout, const uint32_t& num_ranks, const uint32_t& num_rows_per_rank, void* base = nullptr)
        : data_layout(data_layout), num_ranks(num_ranks), num_rows_per_rank(num_rows_per_rank), base(base) {}

    DG_HOST_DEVICE
    uint64_t get_num_bytes_per_rank() const { return num_rows_per_rank * data_layout.get_num_bytes<uint64_t>(); }

    DG_HOST_DEVICE
    uint64_t get_num_bytes() const { return get_num_bytes_per_rank() * num_ranks; }

    template <typename dtype_t = void>
    DG_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    DG_HOST_DEVICE
    void* get_end_ptr() const { return math::advance_ptr(base, get_num_bytes()); }

    DG_HOST_DEVICE
    Buffer get_rank_buffer(const uint32_t& rank_idx) const {
        return {data_layout, 1, num_rows_per_rank, math::advance_ptr(base, get_num_bytes_per_rank() * rank_idx)};
    }

    DG_HOST_DEVICE
    // The combine buffer's second dimension is one group per top-k entry rather than a communication
    // rank, so the combine views read through this named accessor; the generic rank accessor stays for
    // the rank-indexed buffers.
    Buffer get_topk_buffer(const uint32_t& topk_idx) const { return get_rank_buffer(topk_idx); }

    DG_HOST_DEVICE
    DataLayout get_token_layout(const uint32_t& token_idx) const {
        DG_DEVICE_ASSERT(num_ranks == 1);
        return DataLayout(data_layout.num_bytes,
                          data_layout.require_aiu_alignment,
                          math::advance_ptr(base, data_layout.get_num_bytes<uint64_t>() * token_idx));
    }
};

}  // namespace megamoe::layout
