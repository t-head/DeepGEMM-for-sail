#pragma once

#include <cstdint>

namespace deep_gemm {

static constexpr uint32_t kNumMaxRanks = 72;
static constexpr uint32_t kTaggedMinGenerationBits = 20;
static constexpr uint32_t kTaggedMaxCountBits = 32 - kTaggedMinGenerationBits;
static constexpr uint32_t kTaggedMaxCountMask = (1u << kTaggedMaxCountBits) - 1u;

__host__ __device__ __forceinline__ uint32_t tagged_count_bits(uint32_t max_tokens_per_expert) {
    uint32_t bits = 1;
    uint32_t mask = 1u;
    while (mask < max_tokens_per_expert && bits < kTaggedMaxCountBits) {
        ++bits;
        mask = (1u << bits) - 1u;
    }
    return bits;
}

__host__ __device__ __forceinline__ uint32_t tagged_count_mask(uint32_t max_tokens_per_expert) {
    return (1u << tagged_count_bits(max_tokens_per_expert)) - 1u;
}

__host__ __device__ __forceinline__ uint32_t tagged_generation_mask(uint32_t max_tokens_per_expert) {
    return (1u << (32 - tagged_count_bits(max_tokens_per_expert))) - 1u;
}

__host__ __device__ __forceinline__ bool use_tagged_generation_counts(
    uint32_t generation, uint32_t max_tokens_per_expert) {
    return generation > 0 &&
           max_tokens_per_expert <= kTaggedMaxCountMask &&
           generation <= tagged_generation_mask(max_tokens_per_expert);
}

__host__ __device__ __forceinline__ uint32_t pack_count_generation(
    uint32_t generation, uint32_t max_tokens_per_expert) {
    return generation << tagged_count_bits(max_tokens_per_expert);
}

__host__ __device__ __forceinline__ uint32_t unpack_generation_count(
    uint32_t packed_count, uint32_t generation, uint32_t max_tokens_per_expert) {
    if (!use_tagged_generation_counts(generation, max_tokens_per_expert))
        return packed_count;
    const uint32_t count_mask = tagged_count_mask(max_tokens_per_expert);
    const uint32_t tag = pack_count_generation(generation, max_tokens_per_expert);
    return ((packed_count & ~count_mask) == tag) ? (packed_count & count_mask) : 0u;
}

// Symmetric buffer address mapping for NVLink peer-to-peer access.
// Translates local pointers to remote rank addresses via pre-computed offsets.
struct SymBuffer {
    int64_t base;
    int64_t offsets[kNumMaxRanks];
    uint32_t rank_idx;
    uint32_t num_ranks;

    SymBuffer() = default;

    template <typename Container>
    explicit SymBuffer(const Container& c, uint32_t rank_idx_, uint32_t num_ranks_)
        : rank_idx(rank_idx_), num_ranks(num_ranks_) {
        base = c[rank_idx_];
        for (uint32_t i = 0; i < kNumMaxRanks; ++i)
            offsets[i] = i < num_ranks_ ? (c[i] - base) : 0;
    }

    template <typename ptr_t = void*>
    __host__ __device__ ptr_t get_base_ptr() const {
        return reinterpret_cast<ptr_t>(base);
    }

    template <typename ptr_t>
    __host__ __device__ ptr_t map(const ptr_t& ptr, uint32_t dst_rank_idx) const {
        int64_t mapped_ptr = offsets[dst_rank_idx] + reinterpret_cast<int64_t>(ptr);
        return *reinterpret_cast<const ptr_t*>(&mapped_ptr);
    }
};

// Symmetric buffer layout for fused dispatch.
// Each rank's symmetric memory is organized as:
//   [metadata region][FP4 data region][scale data region]
//
// FP4 data: [num_local_experts][max_tokens_per_expert][K/2] bytes (K-Major / RowMajor)
// Scales:   [num_local_experts][ceil(K/32)][max_tokens_per_expert] uint16_t (M-Major / ColumnMajor)
// Metadata: expert_token_counts[num_total_experts] uint32_t
struct DispatchBufferLayout {
    uint32_t num_local_experts;
    uint32_t num_total_experts;
    uint32_t max_tokens_per_expert;
    uint32_t hidden_dim;

    __host__ __device__
    DispatchBufferLayout() = default;

    __host__ __device__
    DispatchBufferLayout(uint32_t num_local_experts_, uint32_t num_total_experts_,
                         uint32_t max_tokens_per_expert_, uint32_t hidden_dim_)
        : num_local_experts(num_local_experts_)
        , num_total_experts(num_total_experts_)
        , max_tokens_per_expert(max_tokens_per_expert_)
        , hidden_dim(hidden_dim_) {}

    // Metadata region: expert_token_counts array
    __host__ __device__
    uint64_t metadata_bytes() const {
        // Align to 16 bytes for AIU descriptor requirements
        uint64_t raw = num_total_experts * sizeof(uint32_t);
        return (raw + 15) & ~uint64_t(15);
    }

    // FP4 data region: packed FP4 values (2 values per byte)
    __host__ __device__
    uint64_t fp4_data_bytes_per_expert() const {
        return uint64_t(max_tokens_per_expert) * (hidden_dim / 2);
    }

    __host__ __device__
    uint64_t fp4_data_bytes() const {
        return num_local_experts * fp4_data_bytes_per_expert();
    }

    // Scale region: uint16_t M-Major (ColumnMajor)
    // Each uint16 packs two consecutive uint8 UE8M0 scales (little-endian),
    // matching preprocess_mxfp4_scales' view(torch.uint16) convention.
    // K dimension: ceil_div(k_blocks, 2) = ceil_div(hidden/32, 2) = ceil_div(K_packed, 32)
    __host__ __device__
    uint32_t k_scale_blocks() const {
        uint32_t k_blocks = (hidden_dim + 31) / 32;
        return (k_blocks + 1) / 2;
    }

    __host__ __device__
    uint64_t scale_elems_per_expert() const {
        return uint64_t(k_scale_blocks()) * max_tokens_per_expert;
    }

    __host__ __device__
    uint64_t scale_bytes_per_expert() const {
        return scale_elems_per_expert() * sizeof(uint16_t);
    }

    __host__ __device__
    uint64_t scale_bytes() const {
        return num_local_experts * scale_bytes_per_expert();
    }

    __host__ __device__
    uint64_t total_bytes() const {
        return metadata_bytes() + fp4_data_bytes() + scale_bytes();
    }

    // Double-buffering: the data region (metadata+fp4+scale) is duplicated so a producer
    // writing generation g+1 into buf[(g+1)&1] cannot clobber buf[g&1] that a 1-iteration-
    // slower consumer is still reading. The per-iteration all-to-all arrival barrier bounds
    // rank skew to 1 iteration, so 2 buffers are sufficient. Parity = generation & 1.
    __host__ __device__
    uint64_t parity_offset(uint32_t parity) const {
        return static_cast<uint64_t>(parity & 1u) * total_bytes();
    }

    // Base pointer of the parity buffer's data region.
    __host__ __device__
    void* parity_base(void* base, uint32_t parity) const {
        return static_cast<uint8_t*>(base) + parity_offset(parity);
    }

    __host__ __device__
    uint64_t ready_flag_offset() const {
        // Arrival slots live AFTER both data buffers (they are monotonic, not double-buffered).
        return (2 * total_bytes() + 15) & ~uint64_t(15);
    }

    __host__ __device__
    uint64_t total_bytes_with_flags() const {
        // Arrival slots: one uint32 per rank (atomic-arrival barrier). 128B = up to 32 ranks.
        return ready_flag_offset() + 128;
    }

    // DG_SFA_PUSH staging region (after the arrival flags). The quant kernel PUSHES
    // each token's ksb scales into the OWNER rank's staging (row-major, fixed
    // per-source-rank band), and the reshape reads it locally. Layout on one rank:
    //   [num_local_real][num_ranks][max_tokens_per_expert][ksb] uint16, x2 for parity.
    // With num_local(=num_total here) built layout, num_local_real*num_ranks == num_total,
    // so one parity of staging == scale_bytes() (num_total * ksb * max_tok * 2).
    __host__ __device__
    uint64_t staging_offset() const {
        // 16-aligned start after both data buffers + arrival flags.
        return (total_bytes_with_flags() + 15) & ~uint64_t(15);
    }
    __host__ __device__
    uint64_t staging_parity_bytes() const {
        // One parity of staging = num_total_experts * ksb * max_tokens * sizeof(uint16).
        return uint64_t(num_total_experts) * k_scale_blocks() * max_tokens_per_expert * sizeof(uint16_t);
    }
    __host__ __device__
    uint64_t total_bytes_with_staging() const {
        return staging_offset() + 2 * staging_parity_bytes();
    }

    // Offset getters
    __host__ __device__
    uint32_t* expert_token_counts_ptr(void* base) const {
        return static_cast<uint32_t*>(base);
    }

    __host__ __device__
    uint8_t* fp4_data_ptr(void* base, uint32_t expert_idx = 0) const {
        uint8_t* p = static_cast<uint8_t*>(base) + metadata_bytes();
        return p + expert_idx * fp4_data_bytes_per_expert();
    }

    __host__ __device__
    uint16_t* scale_ptr(void* base, uint32_t expert_idx = 0) const {
        uint8_t* p = static_cast<uint8_t*>(base) + metadata_bytes() + fp4_data_bytes();
        return reinterpret_cast<uint16_t*>(p) + expert_idx * scale_elems_per_expert();
    }
};

// Per-M-block dispatch metadata produced by the preprocess kernel.
// Extends the standard kIsNoPadPreprocessLayout grouped_layout with remote address info.
struct FusedDispatchParams {
    int* grouped_layout;        // [1 + total_m_blocks] as uint4, same as NoPad format
    uint64_t* remote_addr_a;    // [total_m_blocks] absolute remote FP4 data address
    uint64_t* remote_addr_sfa;  // [total_m_blocks] absolute remote scale address
    uint32_t total_m_blocks;
    uint32_t shape_m;           // total M (sum of all experts' tokens from all ranks)
};

}  // namespace deep_gemm
