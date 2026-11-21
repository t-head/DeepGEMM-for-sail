#pragma once

#include <deep_gemm/impls/mega_moe_copy.cuh>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/sm89_mega_moe.cuh>

#include <hggc_runtime.h>

#include <deep_gemm/common/exception.cuh>
#include <cute/algorithm/tuple_algorithms.hpp>
#include <cute/numeric/math.hpp>
#include <deep_gemm/mma/fp4_mma.cuh>

namespace deep_gemm {

using mega_moe_common::CopyRankMetadataShared;

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
    using ScaleFormat = MegaMoEScaleFormat<false>;
    constexpr uint32_t kScaleBlocks = ScaleFormat::elements(Hidden);
    constexpr uint32_t kVectors = ScaleFormat::vectors(Hidden);
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
    uint32_t partition_end = WholeK ? kVectors :
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
    constexpr uint32_t kMaxPartitionVectors = WholeK ? kVectors :
        math::ceil_div(kVectors, FusedSfaPartitionLayout::kNumPartitions);
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
                const auto source = load_route_source<NumRanks, SourceCapacity, TopK>(
                    token_base, smem_rank_prefix, expert_local, row);
                if (partition_idx == 0 && vector_warp == 0)
                    src_info[compact_base_m + row] = source;
                uint64_t scale_row = uint64_t(source.rank_idx) * SourceCapacity + source.token_idx;
                const uint16_t* src = staging + scale_row * kScaleBlocks;
                for (uint32_t vec_k = partition_begin + vector_warp;
                     vec_k < partition_end; vec_k += kVectorWarps) {
                    ScaleFormat::copy_vector(
                        src, dst_expert + row, vec_k, sfa_stride);
                }
            }
        }
    });
}

template <class TileScheduler, uint32_t BlockM, uint32_t NumThreads>
__device__ __forceinline__ void prepare_expert_layout_and_reshape_sfa(
    const sched::MegaMoESchedulerArguments& sched, layout::MegaMoESignals* signals,
    char* smem_buf, uint32_t generation) {
    constexpr uint32_t kNumRanks = TileScheduler::kNumRanks;
    constexpr uint32_t kNumLocalExperts = TileScheduler::kNumGroups;
    constexpr uint32_t kNumTotalExperts =
        kNumRanks * kNumLocalExperts;
    constexpr uint32_t kSourceCapacity =
        TileScheduler::kSourceCapacity;
    constexpr uint32_t kHidden = TileScheduler::SHAPE_K * 2;
    constexpr uint32_t kSfaPartitions =
        FusedSfaPartitionLayout::kNumPartitions;
    constexpr uint32_t kReshapeWork =
        kNumLocalExperts * kSfaPartitions;
    const layout::MegaMoEBuffer buffers(
        kNumRanks, kNumTotalExperts, kSourceCapacity, TileScheduler::kNumTopK,
        kHidden, TileScheduler::SHAPE_N / 2, sched.staging);

    volatile uint32_t* ready = signals->preprocess_ready;
    auto* smem = reinterpret_cast<uint32_t*>(smem_buf);
    uint32_t parity = generation & 1u;

    if (!DG_MEGAMOE_SPLIT_SYNC && blockIdx.x == 0) {
        prepare_stage1_metadata<TileScheduler, NumThreads>(
            sched, signals, buffers, smem_buf, generation,
            gridDim.x - TileScheduler::kNumCopyBlocks);
    } else {
        if constexpr (!DG_MEGAMOE_SPLIT_SYNC) {
            if (threadIdx.x == 0) comm::wait_expert_metadata(ready, generation);
        }
        if (blockIdx.x >= TileScheduler::kNumCopyBlocks) {
            __syncthreads();
            const uint32_t work_stride = gridDim.x - TileScheduler::kNumCopyBlocks;
            if (__ldbl(sched.copy_grouped_layout + 2) != 0) {
                const uint32_t total_m_blocks = __ldbl(sched.copy_grouped_layout);
                for (uint32_t work = blockIdx.x - TileScheduler::kNumCopyBlocks;
                     work < total_m_blocks; work += work_stride) {
                    const int4 entry = __ldbl(reinterpret_cast<const int4*>(sched.copy_grouped_layout + 4) + work);
                    if (threadIdx.x == 0)
                        smem[kNumRanks] = entry.w;
                    const uint32_t row_begin = (work - entry.z) * BlockM;
                    reshape_sfa<kNumRanks, TileScheduler::kNumTopK, kSourceCapacity, kHidden, NumThreads, true>(
                        buffers.staging_scales[1].data<uint16_t>(signals),
                        layout::bank_data<uint32_t>(buffers.staging_tokens, signals, parity), sched.expert_rank_prefix,
                        entry.x, 0, sched.local_sfa_buf, smem, sched.src_info, row_begin, BlockM);
                    __ppu_threadfence_uncache();
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        // Every chunk writes the whole K range. Count once;
                        // the final writer publishes the remaining partitions.
                        auto* flags = const_cast<uint32_t*>(ready + 1 +
                            entry.x * kSfaPartitions);
                        const uint32_t previous = atomicAdd(flags, 1u);
                        if (previous + 1u == generation) {
                            #pragma unroll
                            for (uint32_t part = 1; part < kSfaPartitions; ++part)
                                __stbl(flags + part, generation);
                        }
                    }
                }
            } else {
                for (uint32_t work = blockIdx.x - TileScheduler::kNumCopyBlocks;
                     work < kReshapeWork; work += work_stride) {
                    const uint32_t expert_local = work / kSfaPartitions;
                    if (threadIdx.x == 0)
                        smem[kNumRanks] = __ldbl(sched.expert_base + expert_local);
                    reshape_sfa<kNumRanks, TileScheduler::kNumTopK, kSourceCapacity, kHidden, NumThreads>(
                        buffers.staging_scales[1].data<uint16_t>(signals),
                        layout::bank_data<uint32_t>(buffers.staging_tokens, signals, parity), sched.expert_rank_prefix,
                        expert_local, work % kSfaPartitions, sched.local_sfa_buf, smem, sched.src_info);

                    __ppu_threadfence_uncache();
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        __stbl(const_cast<uint32_t*>(
                            ready + 1 + work), generation);
                    }
                }
            }
        }
    }

    // Finish readiness stores before returning.
    __syncthreads();
    __threadfence();
}

}  // namespace deep_gemm
