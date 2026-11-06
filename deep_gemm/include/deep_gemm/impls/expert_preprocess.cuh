#pragma once

#include <cstdint>
#include <cute/algorithm/tuple_algorithms.hpp>
#include <cute/numeric/math.hpp>
#include <deep_gemm/layout/dispatch_layout.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/common/math.cuh>

namespace deep_gemm {

// Build rank prefixes and expert layouts before the copy CTAs start.
template <uint32_t BlockM, uint32_t NumRanks, uint32_t NumLocalExperts,
          uint32_t NumThreads>
__device__ __forceinline__ void prepare_expert_metadata(
    const volatile uint32_t* __restrict__ arrival_slots,
    uint32_t generation,
    int* __restrict__ copy_grouped_layout,
    uint32_t* __restrict__ expert_rank_prefix,
    uint32_t* __restrict__ expert_base,
    uint32_t* __restrict__ masked_m,
    uint32_t* smem_workspace,
    const uint32_t* __restrict__ local_pushed_counts,
    int32_t* cumulative_local_expert_recv_stats,
    uint32_t num_compute_ctas, uint32_t* sfa_ready_flags) {

    constexpr uint32_t total_pairs = NumLocalExperts * NumRanks;
    constexpr uint32_t kSfaPartitions = FusedSfaPartitionLayout::kNumPartitions;
    uint32_t* pair_counts = smem_workspace;
    uint32_t* expert_m_blocks = pair_counts + total_pairs;
    uint32_t* expert_block_offsets = expert_m_blocks + NumLocalExperts;
    uint32_t* expert_row_offsets = expert_block_offsets + NumLocalExperts;
    uint32_t* expert_counts = expert_row_offsets + NumLocalExperts;

    // Wait for this round's counts from every source rank.
    if (generation > 0 && threadIdx.x < NumRanks) {
        while (arrival_slots[threadIdx.x] != generation) { }
    }
    // All arrival waits must finish before reading counts.
    __syncthreads();

    // Keep each expert's rank counts next to each other in shared memory.
    #pragma unroll
    for (uint32_t tid = threadIdx.x; tid < total_pairs;
         tid += NumThreads) {
        uint32_t src_rank = tid / NumLocalExperts;
        uint32_t local_expert = tid % NumLocalExperts;
        pair_counts[local_expert * NumRanks + src_rank] = __ldbl(
            local_pushed_counts + tid);
    }
    __syncthreads();

    // Use aligned vector loads for the rank counts.
    constexpr uint32_t kCountVecElements = NumRanks < 4 ? NumRanks : 4;
    struct alignas(kCountVecElements * sizeof(uint32_t)) CountVec {
        uint32_t data[kCountVecElements];
    };
    constexpr uint32_t kNumCountVecs = NumRanks / kCountVecElements;
    constexpr uint32_t kBatchVecs = kNumCountVecs < 4 ? 1 : 4;
    for (uint32_t e = threadIdx.x; e < NumLocalExperts;
         e += NumThreads) {
        CountVec* rank_counts = reinterpret_cast<CountVec*>(
            pair_counts + e * NumRanks);
        uint32_t total = 0;
        #pragma unroll
        for (uint32_t base = 0; base < kNumCountVecs;
             base += kBatchVecs) {
            CountVec counts[kBatchVecs];

            #pragma unroll
            for (uint32_t v = 0; v < kBatchVecs; ++v)
                counts[v] = rank_counts[base + v];

            #pragma unroll
            for (uint32_t v = 0; v < kBatchVecs; ++v) {
                #pragma unroll
                for (uint32_t r = 0; r < kCountVecElements; ++r) {
                    total += counts[v].data[r];
                    counts[v].data[r] = total;
                }
            }

            #pragma unroll
            for (uint32_t v = 0; v < kBatchVecs; ++v)
                rank_counts[base + v] = counts[v];
        }
        masked_m[e] = total;
        expert_counts[e] = total;
        expert_m_blocks[e] = (total + BlockM - 1) / BlockM;
        // CTA0 visits each expert once. Count valid routes even when their weight is zero.
        if (cumulative_local_expert_recv_stats != nullptr)
            atomicAdd(cumulative_local_expert_recv_stats + e, static_cast<int32_t>(total));
    }
    __syncthreads();

    // Publish the rank prefixes before lane 0 reuses pair_counts[0].
    for (uint32_t pair = threadIdx.x; pair < total_pairs; pair += NumThreads)
        expert_rank_prefix[pair] = pair_counts[pair];

    uint32_t& share_rows = smem_workspace[0];
    // Scan experts a warp at a time, keeping both block and row offsets.
    if (threadIdx.x < 32) {
        uint32_t lane = threadIdx.x;
        uint32_t block_prefix = 0;
        uint32_t row_prefix = 0;
        uint32_t max_rows = 0;
        for (uint32_t base = 0; base < NumLocalExperts; base += 32) {
            uint32_t e = base + lane;
            uint32_t blocks = (e < NumLocalExperts)
                ? expert_m_blocks[e] : 0u;
            uint32_t rows = (e < NumLocalExperts)
                ? expert_counts[e] : 0u;
            max_rows = max(max_rows, rows);
            uint32_t block_scan = blocks;
            uint32_t row_scan = rows;
            #pragma unroll
            for (uint32_t offset = 1; offset < 32; offset <<= 1) {
                uint32_t prior_blocks = __shfl_up_sync(0xffffffff, block_scan, offset);
                uint32_t prior_rows = __shfl_up_sync(0xffffffff, row_scan, offset);
                if (lane >= offset) {
                    block_scan += prior_blocks;
                    row_scan += prior_rows;
                }
            }
            if (e < NumLocalExperts) {
                expert_block_offsets[e] = block_prefix + block_scan - blocks;
                expert_row_offsets[e] = row_prefix + row_scan - rows;
            }
            block_prefix += __shfl_sync(0xffffffff, block_scan, 31);
            row_prefix += __shfl_sync(0xffffffff, row_scan, 31);
        }
        const bool multiple_m_blocks = block_prefix > NumLocalExperts;
        if (multiple_m_blocks) {
            #pragma unroll
            for (uint32_t delta = 16; delta > 0; delta >>= 1)
                max_rows = max(max_rows, __shfl_down_sync(0xffffffffu, max_rows, delta));
        }
        if (lane == 0) {
            // Share rows only when a K partition exceeds twice the ideal work per CTA.
            share_rows = multiple_m_blocks &&
                uint64_t(max_rows) * num_compute_ctas > uint64_t(2) * kSfaPartitions * row_prefix;
            copy_grouped_layout[0] = block_prefix;
            copy_grouped_layout[2] = share_rows;
        }
    }
    __syncthreads();

    if (share_rows) {
        // The last completed row chunk brings the first partition tag to generation.
        for (uint32_t work = threadIdx.x; work < NumLocalExperts * kSfaPartitions; work += NumThreads)
            __stbl(sfa_ready_flags + work, generation - expert_m_blocks[work / kSfaPartitions]);
    }

    for (uint32_t e = threadIdx.x; e < NumLocalExperts;
         e += NumThreads) {
        // Entry: expert, token count, first M-block, first row.
        uint4 entry = {e, expert_counts[e], expert_block_offsets[e], expert_row_offsets[e]};
        expert_base[e] = entry.w;
        if (e + 1 == NumLocalExperts)
            expert_base[NumLocalExperts] = entry.w + entry.y;
        for (uint32_t mb = 0; mb < expert_m_blocks[e]; ++mb)
            reinterpret_cast<uint4*>(copy_grouped_layout + 4)[entry.z + mb] = entry;
    }
}

// Put each expert's scales in GEMM order after metadata is ready.
template <uint32_t NumRanks, uint32_t TopK, uint32_t SourceCapacity,
          uint32_t Hidden, uint32_t NumThreads, bool WholeK = false>
__device__ __forceinline__ void reshape_sfa(
    const uint16_t* __restrict__ staging,
    const uint32_t* __restrict__ token_base,
    const uint32_t* __restrict__ expert_rank_prefix,
    uint32_t expert_local,
    uint32_t partition_idx,
    uint16_t* __restrict__ local_sfa_buf,
    uint32_t* smem_rank_prefix,
    layout::TokenSrcMetadata* __restrict__ src_info,
    uint32_t row_begin = 0,
    uint32_t row_count = 0xffffffffu)
{
    constexpr uint32_t kScaleBlocks = Hidden / 64;
    static_assert(FusedSfaPartitionLayout::kNumPartitions == 3,
                  "preprocess readiness is three K partitions per expert");
    // The caller stages the expert's first row just after the rank prefixes.
    if (threadIdx.x < NumRanks)
        smem_rank_prefix[threadIdx.x] = __ldbl(
            expert_rank_prefix + expert_local * NumRanks + threadIdx.x);
    __syncthreads();

    uint32_t compact_base_m = smem_rank_prefix[NumRanks];
    // The three CTAs split K; they do not split source ranks.
    constexpr uint32_t kWarpThreads = 32;
    constexpr uint32_t kWarps = NumThreads / kWarpThreads;
    uint32_t partition_begin = WholeK ? 0u :
        FusedSfaPartitionLayout::partition_begin_vector(
            kScaleBlocks, partition_idx);
    uint32_t partition_end = WholeK ? kScaleBlocks / 8 :
        FusedSfaPartitionLayout::partition_begin_vector(
            kScaleBlocks, partition_idx + 1);
    uint32_t total_count = smem_rank_prefix[NumRanks - 1];
    const uint32_t rows_in_chunk = min(row_count, total_count - row_begin);
    const uint32_t row_end = row_begin + rows_in_chunk;

    const auto sfa_stride = FusedSfaPartitionLayout::align_rows(total_count);
    const auto sfa_base_m = FusedSfaPartitionLayout::row_offset(compact_base_m, expert_local);
    uint16_t* dst_expert = local_sfa_buf +
        static_cast<uint64_t>(sfa_base_m) * kScaleBlocks;

    // Use idle warps for more rows without adding vector work per row.
    constexpr uint32_t kVectors = kScaleBlocks / 8;
    constexpr uint32_t kMaxPartitionVectors = WholeK ? kVectors : (kVectors + 2) / 3;
    constexpr uint32_t kMinVectorWarps = math::next_power_of_two(kMaxPartitionVectors);
    // Each row group needs enough warps for the widest K partition.
    constexpr uint32_t kMaxRowGroups =
        kMaxPartitionVectors > 0 && kWarps >= kMinVectorWarps &&
        (kWarps & (kWarps - 1)) == 0
            ? kWarps / kMinVectorWarps : 1;
    // Stop adding row groups once they no longer save a pass.
    uint32_t row_groups = 1;
    if constexpr (kMaxRowGroups > 1) {
        uint32_t required_groups = math::next_power_of_two(
            rows_in_chunk / kWarpThreads + (rows_in_chunk % kWarpThreads != 0));
        row_groups = required_groups < kMaxRowGroups
            ? required_groups : kMaxRowGroups;
    }

    // Specialize each row-group count to keep warp mapping fixed.
    constexpr int kNumGroupChoices = cute::log_2(kMaxRowGroups) + 1;
    cute::for_each(cute::make_int_sequence<kNumGroupChoices>{}, [&](auto group_log2) {
        constexpr uint32_t kRowGroups = 1u << decltype(group_log2)::value;
        if (row_groups == kRowGroups) {
            constexpr uint32_t kVectorWarps = kWarps / kRowGroups;
            uint32_t warp = threadIdx.x / kWarpThreads;
            uint32_t lane = threadIdx.x % kWarpThreads;
            uint32_t vector_warp = kRowGroups == 1 ? warp : warp % kVectorWarps;
            uint32_t row_group = kRowGroups == 1 ? 0 : warp / kVectorWarps;
            // Keep the warp that writes token metadata active.
            if (partition_begin + vector_warp >= partition_end &&
                !(partition_idx == 0 && vector_warp == 0))
                return;
            // Groups split rows; warps within a group split scales.
            for (uint32_t row = row_begin + lane + row_group * kWarpThreads;
                 row < row_end; row += kWarpThreads * kRowGroups) {
                uint32_t source_rank = 0;
                uint32_t rank_begin = 0;
                uint32_t prefix = 0;
                bool found = false;
                #pragma unroll
                for (uint32_t r = 0; r < NumRanks; ++r) {
                    uint32_t rank_end = smem_rank_prefix[r];
                    if (!found && row < rank_end) {
                        source_rank = r;
                        rank_begin = prefix;
                        found = true;
                    }
                    prefix = rank_end;
                }
                uint64_t expert_rank = expert_local * NumRanks + source_rank;
                uint64_t route_index = expert_rank * SourceCapacity + (row - rank_begin);
                uint32_t token_topk = __ldbl(token_base + route_index);
                uint32_t source_token = token_topk / TopK;
                if (partition_idx == 0 && vector_warp == 0) {
                    auto* token_dst = src_info + compact_base_m;
                    token_dst[row] = {source_rank, source_token, token_topk % TopK};
                }
                uint64_t scale_row = uint64_t(source_rank) * SourceCapacity + source_token;
                const uint16_t* src = staging + scale_row * kScaleBlocks;
                for (uint32_t vec_k = partition_begin + vector_warp;
                     vec_k < partition_end; vec_k += kVectorWarps) {
                    int4 value = __ldbl(
                        reinterpret_cast<const int4*>(src) + vec_k);
                    const uint16_t* elems = reinterpret_cast<const uint16_t*>(&value);
                    uint32_t kb0 = vec_k * 8;
                    #pragma unroll
                    for (uint32_t e = 0; e < 8; ++e) {
                        __stbl(dst_expert + row + static_cast<uint64_t>(kb0 + e) * sfa_stride,
                               elems[e]);
                    }
                }
            }
        }
    });
}

}  // namespace deep_gemm
