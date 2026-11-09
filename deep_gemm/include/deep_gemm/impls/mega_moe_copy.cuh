#pragma once

#include <cstdint>
#include <hggc_runtime.h>
#include <deep_gemm/layout/dispatch_layout.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm::mega_moe_common {

// Copy CTAs reuse GEMM shared storage for the current M-block's rank metadata.
template <uint32_t BlockM, uint32_t NumRanks>
struct alignas(16) CopyRankMetadataShared {
    uint32_t rank_prefix[NumRanks];
    // Cache source row addresses for all K-stripes; destination rows are contiguous.
    uint64_t row_addr[BlockM];
};

__device__ __forceinline__ uint32_t resolve_fused_round_generation(
    const volatile layout::MegaMoESignals& signals) {
    // Stream order makes one cached load enough to read Quant's generation.
    return signals.preprocess_ready[0] & ~kPreprocessMetadataReadyBit;
}

// Turn CTA0's rank prefixes into row addresses. RowBytes is the physical byte stride,
// including packed FP4 rows; dispatch_data selects the caller's payload buffer.
template <uint32_t BlockM, uint32_t NumRanks,
          uint32_t RowBytes,
          uint32_t SourceCapacity, uint32_t TopK, class SchedulerArguments>
__device__ __forceinline__ void build_copy_row_map(
    const SchedulerArguments& sched, const uint8_t* dispatch_data, uint32_t expert_local,
    uint32_t m_block_in_expert, uint32_t generation,
    CopyRankMetadataShared<BlockM, NumRanks>& metadata) {
    const auto* token_topk = sched.dispatch_token_topk +
        (generation & 1u) * (sched.staging.tokens / sizeof(uint32_t));

    // Load all rank prefixes, then distribute real rows across the CTA.
    uint32_t block_begin = m_block_in_expert * BlockM;
    if (threadIdx.x < NumRanks) {
        metadata.rank_prefix[threadIdx.x] = __ldg(
            sched.expert_rank_prefix + expert_local * NumRanks + threadIdx.x);
    }
    __syncthreads();
    uint32_t total = metadata.rank_prefix[NumRanks - 1];
    for (uint32_t row = threadIdx.x; row < BlockM && block_begin + row < total;
         row += blockDim.x) {
        uint32_t query = block_begin + row;
        uint32_t owner = 0;
        for (uint32_t step = NumRanks / 2; step; step /= 2) {
            uint32_t candidate = owner + step;
            if (query >= metadata.rank_prefix[candidate - 1]) owner = candidate;
        }
        uint32_t begin = owner ? metadata.rank_prefix[owner - 1] : 0;
        const uint64_t source_index = (uint64_t(expert_local) * NumRanks + owner) * SourceCapacity +
            (query - begin);
        const uint32_t source_token = __ldbl(token_topk + source_index) / TopK;
        const uint8_t* source = sched.sym_buffer.map(dispatch_data, owner);
        metadata.row_addr[row] = reinterpret_cast<uint64_t>(source) +
            static_cast<uint64_t>(source_token) * RowBytes;
    }
    // Finish row lookup before replacing the global prefixes.
    __syncthreads();
    if (threadIdx.x < NumRanks) {
        uint32_t end = metadata.rank_prefix[threadIdx.x];
        metadata.rank_prefix[threadIdx.x] =
            min(end > block_begin ? end - block_begin : 0u, BlockM);
    }
    __syncthreads();
}

__device__ __forceinline__ void st_bulk_global(int4* ptr, const int4& v) {
    __ppu_global_stbl_bulk_b32x4(ptr, *reinterpret_cast<const uint4*>(&v));
}

// Pick the largest row window that fits 16 loads per warp.
__device__ __host__ inline constexpr uint32_t pbulk_row_depth(uint32_t cpr) {
    constexpr uint32_t kMaxBulkLoadsPerWarp = 16;
    uint32_t rows = 1;
    while (cpr != 0 && rows * 2u * cpr <= kMaxBulkLoadsPerWarp) rows *= 2u;
    return rows;
}

template <bool UseRemote>
__device__ __forceinline__ int4 ld_bulk_class(const int4* ptr) {
    uint4 v;
    if constexpr (UseRemote)
        v = __ppu_remote_load_bulk_b32x4(ptr);
    else
        v = __ppu_global_ldcg_bulk_b32x4(ptr);
    return *reinterpret_cast<const int4*>(&v);
}

}  // namespace deep_gemm::mega_moe_common
