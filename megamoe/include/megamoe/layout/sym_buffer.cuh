#pragma once

#include <megamoe/common/exception.cuh>

namespace megamoe::layout {

// Upper bound the launch ABI accepts for the rank count. The offset table is sized by the
// instance's kNumRanks, so a launch copies only the entries it can address.
static constexpr uint32_t kNumMaxRanks = 72;

template <uint32_t kNumRanks = kNumMaxRanks>
struct SymBuffer {
    int64_t base;
    int64_t rank_addr_deltas[kNumRanks];
    uint32_t rank_idx;

    DG_STATIC_ASSERT(kNumRanks <= kNumMaxRanks, "Too many ranks");

    SymBuffer() = default;

#if defined(__CUDA_ARCH__) or defined(__CLION_IDE__)
    template <typename ptr_t = void*>
    DG_DEVICE ptr_t get_base_ptr() const {
        return reinterpret_cast<ptr_t>(base);
    }

    template <typename ptr_t>
    DG_DEVICE ptr_t map_pointer_to_rank(const ptr_t& ptr, const uint32_t& dst_rank_idx) const {
        if constexpr (kNumRanks == 1)
            return ptr;

        int64_t mapped_ptr = rank_addr_deltas[dst_rank_idx] + reinterpret_cast<int64_t>(ptr);
        return *reinterpret_cast<ptr_t*>(&mapped_ptr);
    }
#endif
};

}  // namespace megamoe::layout
