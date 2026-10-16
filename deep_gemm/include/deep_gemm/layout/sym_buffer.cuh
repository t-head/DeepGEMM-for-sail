#pragma once

// Cross-rank pointer mapping for symmetric memory. Architecture-independent:
// pure pointer arithmetic over per-rank base offsets. Ported verbatim from
// sgl_DeepGEMM (deep_gemm/include/deep_gemm/layout/sym_buffer.cuh).
#include <deep_gemm/layout/mega_moe_common.cuh>

namespace deep_gemm::layout {

constexpr static uint32_t kNumMaxRanks = 72;

template <uint32_t kNumRanks = kNumMaxRanks>
struct SymBuffer {
    int64_t base;
    int64_t offsets[kNumMaxRanks];
    uint32_t rank_idx;

    DG_STATIC_ASSERT(kNumRanks <= kNumMaxRanks, "Too many ranks");

    SymBuffer() = default;

    // Build from a container of per-rank base pointers (as int64_t). `offsets`
    // store the signed delta from the local rank's base, so that
    // `local_ptr + offsets[r]` lands on the same logical address of rank `r`.
    template <typename Container>
    explicit SymBuffer(const Container& c, const uint32_t& rank_idx): rank_idx(rank_idx) {
        const auto size = static_cast<uint32_t>(c.size());
        base = c[rank_idx];
        for (uint32_t i = 0; i < kNumMaxRanks; ++ i)
            offsets[i] = i < size ? (c[i] - base) : 0;
    }

#if defined(__HGGC_ARCH__) or defined(__CLION_IDE__)
    template <typename ptr_t = void*>
    CUTLASS_DEVICE ptr_t get_base_ptr() const {
        return reinterpret_cast<ptr_t>(base);
    }

    template <typename ptr_t>
    CUTLASS_DEVICE ptr_t map(const ptr_t& ptr, const uint32_t& dst_rank_idx) const {
        int64_t mapped_ptr = offsets[dst_rank_idx] + reinterpret_cast<int64_t>(ptr);
        return *reinterpret_cast<ptr_t*>(&mapped_ptr);
    }
#endif
};

} // namespace deep_gemm::layout
