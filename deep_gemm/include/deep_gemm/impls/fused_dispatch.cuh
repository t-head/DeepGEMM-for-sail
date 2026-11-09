#pragma once

#include <deep_gemm/impls/mega_moe_copy.cuh>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/sm89_mega_moe.cuh>

#include <hggc_runtime.h>

#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/impls/expert_preprocess.cuh>
#include <deep_gemm/mma/fp4_mma.cuh>

namespace deep_gemm {

using mega_moe_common::CopyRankMetadataShared;
using mega_moe_common::resolve_fused_round_generation;
using mega_moe_common::build_copy_row_map;
using mega_moe_common::st_bulk_global;
using mega_moe_common::pbulk_row_depth;
using mega_moe_common::ld_bulk_class;

// Load a row window, then store it with the matching bulk instructions.
template <bool UseRemote, uint32_t ChunksPerRow, uint32_t RowDepth,
          bool HalfLastWave = false>
__device__ __forceinline__ void copy_bulk_window(
    const uint64_t* row_addr, int4* block_dst,
    uint32_t local_row_offset, uint32_t local_row_count, uint32_t row0, uint32_t row_stride,
    uint32_t pitch, uint32_t col_base, uint32_t lane) {
    constexpr uint32_t kBulkLanes = 32;
    const int4* sources[RowDepth];
    int4* destinations[RowDepth];
    cute::for_each(cute::make_int_sequence<RowDepth>{}, [&](auto r) {
        constexpr uint32_t kR = decltype(r)::value;
        uint32_t compact_row = row0 + kR * row_stride;
        uint32_t global_row;
        if constexpr (UseRemote) {
            // Remote rows skip the interval owned by the local rank.
            global_row = compact_row + (compact_row >= local_row_offset ? local_row_count : 0u);
        } else {
            global_row = local_row_offset + compact_row;
        }
        sources[kR] = reinterpret_cast<const int4*>(row_addr[global_row]) +
            col_base + lane;
        destinations[kR] = block_dst + global_row * pitch + col_base + lane;
    });
    int4 values[RowDepth][ChunksPerRow];
    uint2 half_values[RowDepth];
    cute::for_each(cute::make_int_sequence<RowDepth>{}, [&](auto r) {
        constexpr uint32_t kR = decltype(r)::value;
        cute::for_each(cute::make_int_sequence<ChunksPerRow>{}, [&](auto c) {
            constexpr uint32_t kC = decltype(c)::value;
            if constexpr (HalfLastWave && kC + 1 == ChunksPerRow) {
                const auto* tail = (sources[kR] - lane) + kC * kBulkLanes;
                if constexpr (UseRemote)
                    half_values[kR] = __ppu_remote_load_bulk_b32x2(tail);
                else
                    half_values[kR] = __ppu_global_ldcg_bulk_b32x2(tail);
            } else {
                values[kR][kC] = ld_bulk_class<UseRemote>(
                    sources[kR] + kC * kBulkLanes);
            }
        });
    });
    cute::for_each(cute::make_int_sequence<RowDepth>{}, [&](auto r) {
        constexpr uint32_t kR = decltype(r)::value;
        cute::for_each(cute::make_int_sequence<ChunksPerRow>{}, [&](auto c) {
            constexpr uint32_t kC = decltype(c)::value;
            if constexpr (HalfLastWave && kC + 1 == ChunksPerRow) {
                auto* tail = (destinations[kR] - lane) + kC * kBulkLanes;
                __ppu_global_stbl_bulk_b32x2(tail, half_values[kR]);
            } else {
                st_bulk_global(destinations[kR] + kC * kBulkLanes, values[kR][kC]);
            }
        });
    });
}

template <bool UseRemote, uint32_t ChunksPerRow, bool HalfLastWave = false,
          uint32_t RowDepth = pbulk_row_depth(ChunksPerRow)>
__device__ __forceinline__ void copy_bulk_rows(
    const uint64_t* row_addr, int4* block_dst,
    uint32_t local_row_offset, uint32_t local_row_count, uint32_t rows_total, uint32_t pitch,
    uint32_t col_base, uint32_t lane, uint32_t row_stride,
    uint32_t& row) {
    for (; row + (RowDepth - 1) * row_stride < rows_total;
         row += RowDepth * row_stride) {
        copy_bulk_window<
            UseRemote, ChunksPerRow, RowDepth, HalfLastWave>(
                row_addr, block_dst, local_row_offset,
                local_row_count, row, row_stride, pitch, col_base, lane);
    }
    if constexpr (RowDepth > 1) {
        copy_bulk_rows<
            UseRemote, ChunksPerRow, HalfLastWave, RowDepth / 2>(
                row_addr, block_dst, local_row_offset,
                local_row_count, rows_total, pitch, col_base, lane,
                row_stride, row);
    }
}

template <uint32_t FullStripeChunks, uint32_t LastStripeChunks, bool HalfLastStripe,
          uint32_t BlockM, uint32_t NumRanks>
__device__ __forceinline__ void copy_rank_rows(
    const CopyRankMetadataShared<BlockM, NumRanks>& metadata,
    int4* block_dst, uint32_t local_source_rank,
    uint32_t stripe_int4s, uint32_t pitch, uint32_t col_base,
    uint32_t capacity_int4s, uint32_t worker_thread,
    uint32_t worker_stride) {
    constexpr uint32_t kBulkLanes = 32;

    uint32_t lane = worker_thread & (kBulkLanes - 1);
    uint32_t warp = worker_thread / kBulkLanes;
    uint32_t num_warps = worker_stride / kBulkLanes;
    uint32_t rows_all = metadata.rank_prefix[NumRanks - 1];
    if (rows_all == 0) return;

    // Assign local and remote copy warps before copying rows.
    uint32_t local_row_offset = local_source_rank == 0
        ? 0u : metadata.rank_prefix[local_source_rank - 1];
    uint32_t rows_local = metadata.rank_prefix[local_source_rank] - local_row_offset;
    uint32_t rows_remote = rows_all - rows_local;

    static_assert(FullStripeChunks > 0 && LastStripeChunks > 0,
                  "copy stripes must contain at least one bulk chunk");
    auto copy_stripe = [&](auto cpr_tag, auto half_tag) {
        constexpr uint32_t kChunksPerRow = decltype(cpr_tag)::value;
        constexpr bool kHalfLastWave = decltype(half_tag)::value;

        // Let small copies overlap; split larger copies by rows and stripe width.
        constexpr uint32_t kLocalLoadsPerWarp = 16;
        constexpr uint32_t kRemoteLoadsPerWarp = 8;
        constexpr bool kWideTwoStripeSchedule = FullStripeChunks >= 3;
        bool overlap_ownership =
            kWideTwoStripeSchedule || rows_all <= 16;
        uint32_t warps_remote;
        uint32_t warps_local;
        if (overlap_ownership) {
            warps_remote = rows_remote == 0 ? 0u :
                (rows_local == 0 ? num_warps :
                 min(num_warps, max(1u,
                     (rows_remote * kChunksPerRow + kRemoteLoadsPerWarp - 1) /
                     kRemoteLoadsPerWarp)));
            warps_local = rows_local == 0 ? 0u :
                (rows_remote == 0 ? num_warps :
                 min(num_warps, max(1u,
                     (rows_local * kChunksPerRow + kLocalLoadsPerWarp - 1) /
                     kLocalLoadsPerWarp)));
        } else if (rows_local == 0) {
            warps_remote = num_warps;
            warps_local = 0;
        } else if (rows_remote == 0) {
            warps_remote = 0;
            warps_local = num_warps;
        } else {
            warps_remote =
                (rows_remote * num_warps + rows_all / 2) / rows_all;
            warps_remote = max(1u, min(num_warps - 1, warps_remote));
            warps_local = num_warps - warps_remote;
        }
        if (warp < warps_remote) {
            uint32_t row = warp;
            copy_bulk_rows<
                true, kChunksPerRow, kHalfLastWave>(
                    metadata.row_addr, block_dst,
                    local_row_offset, rows_local, rows_remote, pitch,
                    col_base, lane, warps_remote, row);
        }
        // Give local copy an idle warp pool when both pools fit.
        uint32_t local_warp_begin =
            warps_remote + warps_local <= num_warps ? warps_remote : 0u;
        if (warp >= local_warp_begin &&
            warp < local_warp_begin + warps_local) {
            uint32_t row = warp - local_warp_begin;
            copy_bulk_rows<
                false, kChunksPerRow, kHalfLastWave>(
                    metadata.row_addr, block_dst,
                    local_row_offset, rows_local, rows_local, pitch,
                    col_base, lane, warps_local, row);
        }
    };

    if (stripe_int4s == capacity_int4s) {
        copy_stripe(cute::Int<FullStripeChunks>{}, cute::Int<0>{});
    } else {
        copy_stripe(cute::Int<LastStripeChunks>{}, cute::Int<HalfLastStripe>{});
    }
}

template <uint32_t BlockM, uint32_t NumRanks,
          uint32_t BlockK,
          uint32_t KTilesPerFlag, uint32_t ShapeK, uint32_t NumThreads,
          uint32_t SourceCapacity, uint32_t TopK>
__device__ void run_copy_block(
    const sched::MegaMoESchedulerArguments& sched, uint32_t total_m_blocks,
    char* smem_buf, uint32_t generation)
{
    constexpr uint32_t kStripeVectors = KTilesPerFlag * BlockK / 16;
    constexpr uint32_t kRowVectors = ShapeK / 16;

    static_assert(ShapeK % 256 == 0,
                  "the packed K row must contain whole 256-byte bulk waves");
    static_assert(BlockK % 16 == 0,
                  "stripe capacity must be an exact int4 count: BlockK must "
                  "be a multiple of 16");
    static_assert(kStripeVectors % 32 == 0,
                  "each K stripe must contain a whole 32-int4 bulk wave");

    // Keep this runtime load; constexpr added scalar waits in the EP16 check.
    uint32_t num_copy = sched.num_copy_blocks;
    // Include the partial tail stripe when publishing readiness.
    constexpr uint32_t kNumStripes =
        math::ceil_div(kRowVectors, kStripeVectors);
    auto& shared_rank_metadata =
        *reinterpret_cast<CopyRankMetadataShared<BlockM, NumRanks>*>(smem_buf);
    constexpr uint32_t kPBulkLanes = 32;
    constexpr uint32_t kTailVectors =
        kRowVectors - (kNumStripes - 1) * kStripeVectors;
    constexpr uint32_t kFullStripeChunks =
        kStripeVectors / kPBulkLanes;
    constexpr uint32_t kFinalStripeChunks =
        math::ceil_div(kTailVectors, kPBulkLanes);
    constexpr bool kHalfLastStripe =
        kTailVectors % kPBulkLanes != 0;

    static_assert(NumThreads % 32 == 0, "bulk copy requires complete warps");

    for (uint32_t mb = blockIdx.x; mb < total_m_blocks; mb += num_copy) {
        uint4 gl = (reinterpret_cast<const uint4*>(sched.copy_grouped_layout) + 1)[mb];
        uint32_t m_block_in_expert = mb - gl.z;

        build_copy_row_map<
            BlockM, NumRanks, ShapeK,
            SourceCapacity, TopK>(
                sched, sched.dispatch_fp4, gl.x, m_block_in_expert,
                generation,
                shared_rank_metadata);

        uint64_t row_base = static_cast<uint64_t>(gl.w) +
            static_cast<uint64_t>(m_block_in_expert) * BlockM;
        auto* local_block_dst = reinterpret_cast<int4*>(sched.local_fp4_buf + row_base * ShapeK);

        for (uint32_t ks = 0; ks < kNumStripes; ++ks) {
            uint32_t stripe_col_start =
                ks * kStripeVectors;
            // Clamp the tail stripe to the row width.
            uint32_t stripe_int4s = min(
                kStripeVectors,
                kRowVectors - stripe_col_start);
            copy_rank_rows<
                kFullStripeChunks, kFinalStripeChunks, kHalfLastStripe>(
                    shared_rank_metadata,
                    local_block_dst, sched.sym_buffer.rank_idx,
                    stripe_int4s,
                    kRowVectors, stripe_col_start,
                    kStripeVectors, threadIdx.x,
                    NumThreads);
            // Finish stripe stores before publishing readiness.
            __ppu_threadfence_uncache();
            __syncthreads();
            if (threadIdx.x == 0) {
                __stbl(const_cast<uint32_t*>(sched.copy_ready_flags + mb),
                       (generation << 3) | (ks + 1u));
            }
        }
    }
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
    uint32_t metadata_ready = kPreprocessMetadataReadyBit | generation;
    auto* smem = reinterpret_cast<uint32_t*>(smem_buf);
    uint32_t parity = generation & 1u;
    const uint32_t* local_pushed_counts = buffers.staging_counts[parity].data<uint32_t>(signals);

    if (blockIdx.x == 0) {
        // Generation tags avoid clearing ready flags every round.
        if (generation == 2u) {
            // Clear every view's tagged flags before publishing metadata_ready.
            const uint32_t num_copy_ready_flags = buffers.copy_ready.bytes / sizeof(uint32_t);
            for (uint32_t i = threadIdx.x; i < num_copy_ready_flags;
                 i += NumThreads)
                __stbl(const_cast<uint32_t*>(sched.copy_ready_flags + i), 0u);
            constexpr uint32_t kNumPreprocessReadyFlags =
                layout::MegaMoESignals::kNumMaxExperts * kSfaPartitions;
            for (uint32_t i = threadIdx.x; i < kNumPreprocessReadyFlags;
                 i += NumThreads)
                __stbl(const_cast<uint32_t*>(ready + 1 + i), 0u);
        }

        prepare_expert_metadata<
            BlockM, kNumRanks, kNumLocalExperts, NumThreads>(
                signals->dispatch_arrivals, generation, sched.copy_grouped_layout,
                const_cast<uint32_t*>(sched.expert_rank_prefix), sched.expert_base,
                reinterpret_cast<uint32_t*>(sched.grouped_layout), smem, local_pushed_counts,
                sched.cumulative_local_expert_recv_stats, gridDim.x - TileScheduler::kNumCopyBlocks,
                const_cast<uint32_t*>(ready + 1));

        // Reset the GEMM1 tile counter before this round's metadata is ready.
        // Header word zero remains the M-block count; records start at word four.
        if (threadIdx.x == 0) sched.copy_grouped_layout[1] = 0;

        // Publish metadata before its ready flag.
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0)
            __stbl(const_cast<uint32_t*>(ready), metadata_ready);
    } else {
        const bool is_compute_cta = blockIdx.x >= TileScheduler::kNumCopyBlocks;
        if (threadIdx.x == 0) {
            while (tix::ld_ready_flag_uncached(ready) != metadata_ready) { }
        }
        if (is_compute_cta) {
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
                        buffers.staging_scales[parity].data<uint16_t>(signals),
                        buffers.staging_tokens[parity].data<uint32_t>(signals), sched.expert_rank_prefix,
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
                        buffers.staging_scales[parity].data<uint16_t>(signals),
                        buffers.staging_tokens[parity].data<uint32_t>(signals), sched.expert_rank_prefix,
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
