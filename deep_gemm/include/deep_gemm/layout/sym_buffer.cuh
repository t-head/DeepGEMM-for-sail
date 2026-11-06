#pragma once

#include <cstdint>
#include <cutlass/cutlass.h>

namespace deep_gemm::layout {

constexpr uint32_t kNumMaxRanks = 64;

// Same address mapping contract as upstream: passed by value, not a GPU table.
template <uint32_t kNumRanks = kNumMaxRanks>
struct SymBuffer {
    int64_t base{};
    int64_t offsets[kNumRanks]{};
    uint32_t rank_idx{};

    CUTLASS_HOST_DEVICE SymBuffer() = default;

    template <uint32_t OtherRanks>
    CUTLASS_HOST_DEVICE explicit SymBuffer(const SymBuffer<OtherRanks>& other)
        : base(other.base), rank_idx(other.rank_idx) {
        static_assert(OtherRanks >= kNumRanks, "cannot invent peer offsets");
        for (uint32_t peer = 0; peer < kNumRanks; ++peer)
            offsets[peer] = other.offsets[peer];
    }

    template <typename Container>
    explicit SymBuffer(const Container& c, uint32_t rank_idx) : base(c[rank_idx]), rank_idx(rank_idx) {
        for (uint32_t i = 0; i < kNumRanks; ++i)
            offsets[i] = i < c.size() ? c[i] - base : 0;
    }

    template <typename ptr_t = void*>
    CUTLASS_HOST_DEVICE ptr_t get_base_ptr() const {
        return reinterpret_cast<ptr_t>(base);
    }

    template <typename ptr_t>
    CUTLASS_HOST_DEVICE ptr_t map(ptr_t ptr, uint32_t dst_rank_idx) const {
        return reinterpret_cast<ptr_t>(reinterpret_cast<int64_t>(ptr) + offsets[dst_rank_idx]);
    }
};

}  // namespace deep_gemm::layout
