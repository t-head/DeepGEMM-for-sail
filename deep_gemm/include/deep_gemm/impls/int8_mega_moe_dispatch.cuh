#pragma once

#include <deep_gemm/impls/mega_moe_copy.cuh>

#include <hggc_runtime.h>
#include <type_traits>
#include <utility>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/int8_mega_moe.cuh>
#include <deep_gemm/scheduler/sm80_mega_moe.cuh>
#include <deep_gemm/comm/int8_dispatch.cuh>
#include <deep_gemm/comm/barrier.cuh>

namespace deep_gemm::sm80 {

// Build rank prefixes and expert layouts before the copy CTAs start.
template <uint32_t BlockM, uint32_t NumRanks, uint32_t NumLocalExperts,
          uint32_t NumThreads>
__device__ __forceinline__ void prepare_expert_metadata(
    const volatile uint32_t* __restrict__ arrival_slots,
    uint32_t generation,
    int* __restrict__ grouped_layout,
    uint32_t* __restrict__ expert_rank_prefix,
    uint32_t* __restrict__ expert_base,
    uint32_t* __restrict__ masked_m,
    uint32_t* smem_workspace,
    const uint32_t* __restrict__ local_pushed_counts,
    int32_t* cumulative_local_expert_recv_stats) {

    constexpr uint32_t total_pairs = NumLocalExperts * NumRanks;
    uint32_t* pair_counts = smem_workspace;
    uint32_t* expert_m_blocks = pair_counts + total_pairs;
    uint32_t* expert_block_offsets = expert_m_blocks + NumLocalExperts;
    uint32_t* expert_row_offsets = expert_block_offsets + NumLocalExperts;
    uint32_t* expert_counts = expert_row_offsets + NumLocalExperts;

    if constexpr (!DG_MEGAMOE_SPLIT_SYNC)
        comm::rank_barrier<NumRanks>(arrival_slots, generation);
    else
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
        // CTA0 visits each expert once, including valid routes with zero weight.
        if (cumulative_local_expert_recv_stats != nullptr)
            atomicAdd(cumulative_local_expert_recv_stats + e, static_cast<int32_t>(total));
    }
    __syncthreads();

    // Scan experts a warp at a time, keeping both block and row offsets.
    if (threadIdx.x < 32) {
        uint32_t lane = threadIdx.x;
        uint32_t block_prefix = 0;
        uint32_t row_prefix = 0;
        for (uint32_t base = 0; base < NumLocalExperts; base += 32) {
            uint32_t e = base + lane;
            uint32_t blocks = (e < NumLocalExperts)
                ? expert_m_blocks[e] : 0u;
            uint32_t rows = (e < NumLocalExperts)
                ? expert_counts[e] : 0u;
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
                if (e + 1 == NumLocalExperts) {
                    grouped_layout[0] = block_prefix + block_scan;
                }
            }
            // Carry offsets only if there is another batch.
            if (base + 32 < NumLocalExperts) {
                block_prefix += __shfl_sync(0xffffffff, block_scan, 31);
                row_prefix += __shfl_sync(0xffffffff, row_scan, 31);
            }
        }
    }
    __syncthreads();

    // Share rank prefixes with the copy CTAs.
    for (uint32_t pair = threadIdx.x; pair < total_pairs;
         pair += NumThreads)
        expert_rank_prefix[pair] = pair_counts[pair];

    for (uint32_t e = threadIdx.x; e < NumLocalExperts;
         e += NumThreads) {
        // Entry: expert, token count, first M-block, first row.
        uint4 entry = {e, expert_counts[e], expert_block_offsets[e], expert_row_offsets[e]};
        expert_base[e] = entry.w;
        if (e + 1 == NumLocalExperts)
            expert_base[NumLocalExperts] = entry.w + entry.y;
        for (uint32_t mb = 0; mb < expert_m_blocks[e]; ++mb)
            reinterpret_cast<uint4*>(grouped_layout + 4)[entry.z + mb] = entry;
    }
}

}  // namespace deep_gemm

namespace deep_gemm::sm80 {

// Compile-time indices keep bulk-copy temporaries in registers without a CuTe
// dependency, so both existing ACTLIZE generations share this implementation.
template <class Function, size_t... Indices>
__device__ __forceinline__ void for_each_copy_index(std::index_sequence<Indices...>, Function f) {
    (f(std::integral_constant<size_t, Indices>{}), ...);
}


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
    uint32_t group_row_begin, uint32_t local_row_offset,
    uint32_t local_row_count, uint32_t row0, uint32_t row_stride,
    uint32_t pitch, uint32_t col_base, uint32_t lane) {
    constexpr uint32_t kBulkLanes = 32;
    const int4* sources[RowDepth];
    int4* destinations[RowDepth];
    for_each_copy_index(std::make_index_sequence<RowDepth>{}, [&](auto r) {
        constexpr uint32_t kR = decltype(r)::value;
        uint32_t compact_row = row0 + kR * row_stride;
        uint32_t global_row;
        if constexpr (UseRemote) {
            global_row = group_row_begin + compact_row;
            // Skip local rows; local_row_count is zero if this group has none.
            global_row +=
                local_row_count != 0 && compact_row >= local_row_offset
                    ? local_row_count : 0u;
        } else {
            global_row = group_row_begin + local_row_offset + compact_row;
        }
        sources[kR] = reinterpret_cast<const int4*>(row_addr[global_row]) +
            col_base + lane;
        destinations[kR] = block_dst + global_row * pitch + col_base + lane;
    });
    int4 values[RowDepth][ChunksPerRow];
    uint2 half_values[RowDepth];
    for_each_copy_index(std::make_index_sequence<RowDepth>{}, [&](auto r) {
        constexpr uint32_t kR = decltype(r)::value;
        for_each_copy_index(std::make_index_sequence<ChunksPerRow>{}, [&](auto c) {
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
    for_each_copy_index(std::make_index_sequence<RowDepth>{}, [&](auto r) {
        constexpr uint32_t kR = decltype(r)::value;
        for_each_copy_index(std::make_index_sequence<ChunksPerRow>{}, [&](auto c) {
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
    uint32_t group_row_begin, uint32_t local_row_offset,
    uint32_t local_row_count, uint32_t rows_total, uint32_t pitch,
    uint32_t col_base, uint32_t lane, uint32_t row_stride,
    uint32_t& row) {
    for (; row + (RowDepth - 1) * row_stride < rows_total;
         row += RowDepth * row_stride) {
        copy_bulk_window<
            UseRemote, ChunksPerRow, RowDepth, HalfLastWave>(
                row_addr, block_dst, group_row_begin, local_row_offset,
                local_row_count, row, row_stride, pitch, col_base, lane);
    }
    if constexpr (RowDepth > 1) {
        copy_bulk_rows<
            UseRemote, ChunksPerRow, HalfLastWave, RowDepth / 2>(
                row_addr, block_dst, group_row_begin, local_row_offset,
                local_row_count, rows_total, pitch, col_base, lane,
                row_stride, row);
    }
}

template <uint32_t FullStripeChunks, uint32_t LastStripeChunks, bool HalfLastStripe,
          uint32_t BlockM, uint32_t NumRanks>
__device__ __forceinline__ void copy_rank_rows(
    const CopyRankMetadataShared<BlockM, NumRanks>& metadata,
    int4* block_dst, uint32_t local_source_rank, uint32_t group_base,
    uint32_t stripe_int4s, uint32_t pitch, uint32_t col_base,
    uint32_t capacity_int4s, uint32_t worker_thread,
    uint32_t worker_stride) {
    constexpr uint32_t kBulkLanes = 32;

    uint32_t lane = worker_thread & (kBulkLanes - 1);
    uint32_t warp = worker_thread / kBulkLanes;
    uint32_t num_warps = worker_stride / kBulkLanes;
    uint32_t group_row_begin = group_base == 0
        ? 0u : metadata.rank_prefix[group_base - 1];
    uint32_t group_row_end =
        metadata.rank_prefix[group_base + NumRanks - 1];
    uint32_t rows_all = group_row_end - group_row_begin;
    if (rows_all == 0) return;

    // Assign local and remote copy warps before copying rows.
    bool group_contains_local =
        local_source_rank >= group_base &&
        local_source_rank < group_base + NumRanks;
    uint32_t local_row_begin = local_source_rank == 0
        ? 0u : metadata.rank_prefix[local_source_rank - 1];
    uint32_t local_row_end = metadata.rank_prefix[local_source_rank];
    uint32_t rows_local = group_contains_local
        ? local_row_end - local_row_begin : 0u;
    uint32_t local_row_offset = group_contains_local
        ? local_row_begin - group_row_begin : 0u;
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
                    metadata.row_addr, block_dst, group_row_begin,
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
                    metadata.row_addr, block_dst, group_row_begin,
                    local_row_offset, rows_local, rows_local, pitch,
                    col_base, lane, warps_local, row);
        }
    };

    if (stripe_int4s == capacity_int4s) {
        copy_stripe(std::integral_constant<int, FullStripeChunks>{}, std::integral_constant<int, 0>{});
    } else {
        copy_stripe(std::integral_constant<int, LastStripeChunks>{}, std::integral_constant<int, HalfLastStripe>{});
    }
}

struct NoCopyScaleReadiness {
    template <class SchedulerArguments>
    __device__ static __forceinline__ void wait(
        const SchedulerArguments&, uint32_t, uint32_t, bool) {}
};

template <uint32_t BlockM, uint32_t NumRanks,
          uint32_t BlockK,
          uint32_t KTilesPerFlag, uint32_t ShapeK, uint32_t NumThreads,
          uint32_t SourceCapacity, uint32_t TopK,
          class ScaleReadiness = NoCopyScaleReadiness, class SchedulerArguments>
__device__ void run_copy_block(
    const SchedulerArguments& sched, uint32_t total_m_blocks,
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
                sched, sched.dispatch_data, gl.x, m_block_in_expert, generation,
                shared_rank_metadata);

        uint64_t row_base = static_cast<uint64_t>(gl.w) +
            static_cast<uint64_t>(m_block_in_expert) * BlockM;
        auto* local_block_dst = reinterpret_cast<int4*>(sched.local_data_buf + row_base * ShapeK);

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
                    local_block_dst, sched.local_source_rank,
                    0, stripe_int4s,
                    kRowVectors, stripe_col_start,
                    kStripeVectors, threadIdx.x,
                    NumThreads);
            // Finish stripe stores before publishing readiness.
            __ppu_threadfence_uncache();
            __syncthreads();
            if (threadIdx.x == 0) {
                // INT8 joins scale compaction only after the final payload
                // stripe has been copied; earlier stripes remain independent.
                ScaleReadiness::wait(sched, gl.x, generation, ks + 1 == kNumStripes);
                __stbl(const_cast<uint32_t*>(sched.copy_ready_flags + mb),
                       (generation << 3) | (ks + 1u));
            }
        }
    }
}

// CTA0 publishes one shared metadata representation for copy and compute.
template <uint32_t BlockM, uint32_t NumRanks, uint32_t NumLocalExperts,
          uint32_t NumThreads, uint32_t SourceCapacity, class SchedulerArguments>
__device__ __forceinline__ void publish_expert_metadata(
    const SchedulerArguments& sched, char* smem_buf, uint32_t generation,
    const volatile uint32_t* arrival_slots, const uint32_t* local_pushed_counts) {
    // Generation tags avoid clearing ready flags every round.
    prepare_expert_metadata<
        BlockM, NumRanks, NumLocalExperts, NumThreads>(
            arrival_slots, generation,
            sched.copy_grouped_layout,
            const_cast<uint32_t*>(sched.expert_rank_prefix),
            sched.expert_base,
            reinterpret_cast<uint32_t*>(sched.grouped_layout),
            reinterpret_cast<uint32_t*>(smem_buf), local_pushed_counts,
            sched.cumulative_local_expert_recv_stats);

    if (generation == 2u) {
        // Clear every alias's tagged flags before publishing metadata_ready.
        const uint32_t num_copy_ready_flags = sched.staging.tokens / (16 * sizeof(uint32_t));
        for (uint32_t i = threadIdx.x; i < num_copy_ready_flags;
             i += NumThreads)
            __stbl(const_cast<uint32_t*>(sched.copy_ready_flags + i), 0u);
        constexpr uint32_t kNumPreprocessReadyFlags =
            layout::MegaMoESignals::kNumMaxExperts * FusedSfaPartitionLayout::kNumPartitions;
        for (uint32_t i = threadIdx.x; i < kNumPreprocessReadyFlags; i += NumThreads)
            __stbl(const_cast<uint32_t*>(sched.preprocess_ready_flags + 1 + i), 0u);
    }

    // Publish metadata before its ready flag.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
        __stbl(const_cast<uint32_t*>(sched.preprocess_ready_flags),
               kPreprocessMetadataReadyBit | generation);
}

}  // namespace deep_gemm

namespace deep_gemm::sm80 {

template <int N, int K, int Groups, int Ranks, int Capacity, int TopK, int CopyBlocks>
inline sched::Int8MegaMoESchedulerArguments make_int8_dispatch_arguments(const layout::SymBuffer<>& symmetric) {
    constexpr layout::Int8MegaMoEBuffer buffers{Ranks, Groups * Ranks, Capacity, TopK, K, N / 2};
    auto sched = sched::make_int8_mega_moe_scheduler(buffers, symmetric.get_base_ptr(), Ranks * Capacity);
    sched.num_copy_blocks = CopyBlocks;
    sched.sym_buffer = symmetric;
    sched.local_source_rank = symmetric.rank_idx;

    return sched;
}

// INT8 broadcasts contiguous source-token scales, then compacts expert rows.
// The publication, metadata, row copy and scheduler protocols remain shared.
template <uint32_t BlockK, bool ReclaimCopyBlocks = false>
struct Int8FusedDispatch {
    static constexpr bool kEnabled = true;
    using Scale = float;

    template <class Scheduler>
    static constexpr uint32_t shared_bytes() {
        constexpr uint32_t metadata = (Scheduler::kNumRanks + 4) * Scheduler::kNumGroups * sizeof(uint32_t);
        constexpr uint32_t copy = sizeof(CopyRankMetadataShared<Scheduler::BLOCK_M, Scheduler::kNumRanks>);
        constexpr uint32_t reshape = (Scheduler::kNumRanks + 1) * sizeof(uint32_t);
        if constexpr (Scheduler::kNumCopyBlocks == 0) return reshape;
        return metadata > copy ? metadata : copy;
    }

    // Like FP4, compute CTAs compact scale rows while NCB copies payloads.
    // INT8 has one FP32 scale per token, so only the routed row order changes.
    template <class Scheduler, uint32_t Threads>
    __device__ static __forceinline__ void reshape_scales(
        const typename Scheduler::Arguments& sched, char* shared, uint32_t generation) {
        constexpr uint32_t ranks = Scheduler::kNumRanks;
        constexpr uint32_t groups = Scheduler::kNumGroups;
        constexpr uint32_t capacity = Scheduler::kSourceCapacity;
        constexpr uint32_t topk = Scheduler::kNumTopK;
        constexpr uint32_t copy_blocks = Scheduler::kNumCopyBlocks;
        const layout::Int8MegaMoEBuffer buffers(
            ranks, ranks * groups, capacity, topk, Scheduler::SHAPE_K, Scheduler::SHAPE_N / 2,
            sched.staging);
        auto* prefix = reinterpret_cast<uint32_t*>(shared);
        void* base = sched.sym_buffer.get_base_ptr();
        const auto* scales = buffers.staging_scales[0].data<float>(base) +
            uint64_t(generation & 1u) * (sched.staging.scales / sizeof(float));
        const auto* tokens = sched.dispatch_token_topk +
            uint64_t(generation & 1u) * (sched.staging.tokens / sizeof(uint32_t));
        auto* sources = static_cast<layout::TokenSrcMetadata*>(sched.src_info);
        for (uint32_t expert = blockIdx.x - copy_blocks;
             expert < groups; expert += gridDim.x - copy_blocks) {
            if (threadIdx.x < ranks)
                prefix[threadIdx.x] = __ldbl(sched.expert_rank_prefix + expert * ranks + threadIdx.x);
            if (threadIdx.x == 0) prefix[ranks] = __ldbl(sched.expert_base + expert);
            __syncthreads();
            const uint32_t row_base = prefix[ranks];
            for (uint32_t row = threadIdx.x; row < prefix[ranks - 1]; row += Threads) {
                uint32_t rank = 0;
                for (uint32_t step = ranks / 2; step; step /= 2) {
                    const uint32_t next = rank + step;
                    if (row >= prefix[next - 1]) rank = next;
                }
                const uint32_t begin = rank ? prefix[rank - 1] : 0;
                const uint64_t source = (uint64_t(expert) * ranks + rank) * capacity + row - begin;
                const uint32_t token = __ldbl(tokens + source);
                __stbl(sched.local_sfa_buf + row_base + row,
                       __ldbl(scales + uint64_t(rank) * capacity + token / topk));
                __stbl(&sources[row_base + row].rank_idx, rank);
                __stbl(&sources[row_base + row].token_idx, token / topk);
                __stbl(&sources[row_base + row].topk_idx, token % topk);
            }
            __ppu_threadfence_uncache();
            __syncthreads();
            if (threadIdx.x == 0)
                __stbl(const_cast<uint32_t*>(sched.preprocess_ready_flags + 1 + expert), generation);
            // Do not reuse prefix storage until every thread completed this row set.
            __syncthreads();
        }
    }

    template <class Scheduler, uint32_t Threads>
    __device__ static void prepare_metadata(const sched::Int8MegaMoESchedulerArguments& sched,
                                           char* smem, uint32_t generation) {
        constexpr uint32_t ranks = Scheduler::kNumRanks;
        constexpr uint32_t groups = Scheduler::kNumGroups;
        constexpr uint32_t capacity = Scheduler::kSourceCapacity;
        constexpr uint32_t topk = Scheduler::kNumTopK;
        const layout::Int8MegaMoEBuffer buffers(
            ranks, ranks * groups, capacity, topk, Scheduler::SHAPE_K, Scheduler::SHAPE_N / 2,
            sched.staging);
        auto* base = sched.sym_buffer.get_base_ptr<layout::MegaMoESignals*>();
        const auto* arrivals = base->dispatch_arrivals;
        const auto* counts = buffers.staging_counts[0].data<uint32_t>(base) +
            (generation & 1u) * (sched.staging.counts / sizeof(uint32_t));
        if (blockIdx.x == 0) {
            // Metadata publication also releases this round's work counter.
            if constexpr (ReclaimCopyBlocks) {
                if (threadIdx.x == 0) __stbl(sched.work_queue_counter(), 0u);
            }
            publish_expert_metadata<Scheduler::BLOCK_M, ranks, groups, Threads, capacity>(
                sched, smem, generation, arrivals, counts);
        }
        wait_metadata(sched, generation);
    }

    __device__ static __forceinline__ void wait_metadata(
        const sched::Int8MegaMoESchedulerArguments& sched, uint32_t generation) {
        if (threadIdx.x == 0)
            while (tix::ld_ready_flag_uncached(sched.preprocess_ready_flags) !=
                   (kPreprocessMetadataReadyBit | generation)) { }
        __syncthreads();
    }

    struct ScaleReadiness {
        template <class Arguments>
        __device__ static __forceinline__ void wait(
            const Arguments& sched, uint32_t expert, uint32_t generation, bool last_stripe) {
            if (last_stripe)
                while (tix::ld_ready_flag_uncached(sched.preprocess_ready_flags + 1 + expert) != generation) { }
        }
    };

    template <class Scheduler, uint32_t Threads>
    __device__ static void prepare_copy(const sched::Int8MegaMoESchedulerArguments& sched, char* smem) {
        const uint32_t generation = resolve_fused_round_generation(
            *sched.sym_buffer.get_base_ptr<layout::MegaMoESignals*>());
        prepare_metadata<Scheduler, Threads>(sched, smem, generation);
        run_copy_block<Scheduler::BLOCK_M, Scheduler::kNumRanks,
                       BlockK, Scheduler::kKTilesPerFlag, Scheduler::SHAPE_K, Threads,
                       Scheduler::kSourceCapacity, Scheduler::kNumTopK, ScaleReadiness>(
            sched, sched.copy_grouped_layout[0], smem, generation);
    }

    template <class Scheduler, uint32_t Threads>
    __device__ static bool prepare(const typename Scheduler::Arguments& sched, char* smem) {
        const uint32_t generation = resolve_fused_round_generation(
            *sched.sym_buffer.template get_base_ptr<layout::MegaMoESignals*>());
        if constexpr (Scheduler::kNumCopyBlocks > 0) {
            prepare_metadata<Scheduler, Threads>(sched, smem, generation);

            if (blockIdx.x < Scheduler::kNumCopyBlocks) {
                run_copy_block<Scheduler::BLOCK_M, Scheduler::kNumRanks,
                               BlockK, Scheduler::kKTilesPerFlag, Scheduler::SHAPE_K, Threads,
                               Scheduler::kSourceCapacity, Scheduler::kNumTopK, ScaleReadiness>(
                    sched, sched.copy_grouped_layout[0], smem, generation);
                if constexpr (!ReclaimCopyBlocks) return false;
            }
        } else {
            // Independent copy CTAs publish metadata. All GEMM CTAs consume
            // K-stripes through the existing mainloop readiness checks.
            wait_metadata(sched, generation);
        }

        if (blockIdx.x >= Scheduler::kNumCopyBlocks)
            reshape_scales<Scheduler, Threads>(sched, smem, generation);
        __threadfence();
        return true;
    }


};

}  // namespace deep_gemm

namespace deep_gemm::sm80 {

// Copy allocates only metadata, independently of the GEMM operand pipeline.
// Dispatch supplies the dtype-specific scale gather and metadata preparation.
template <class Dispatch, int N, int K, int BM, int BK, int Groups, int Ranks,
          int KTilesPerFlag, int SourceCapacity, int TopK, int Threads>
struct MegaMoECopyKernel {
    static_assert(is_supported_megamoe_rank_count(Ranks));
    static_assert(Threads >= Ranks && Threads >= 64 && Threads % 32 == 0);
    static_assert(KTilesPerFlag > 0 && math::ceil_div(K / BK, KTilesPerFlag) <= 7);
    static_assert(K > 0 && BK > 0 && K % BK == 0);
    struct PrepareConfig {
        using Arguments = sched::Int8MegaMoESchedulerArguments;
        static constexpr int kNumRanks = Ranks, kNumGroups = Groups;
        static constexpr int kSourceCapacity = SourceCapacity, kNumTopK = TopK;
        static constexpr int SHAPE_N = N, SHAPE_K = K, BLOCK_M = BM;
        static constexpr int kKTilesPerFlag = KTilesPerFlag;
    };
    using Arguments = sched::Int8MegaMoESchedulerArguments;
    static constexpr int CopyBytes = sizeof(CopyRankMetadataShared<BM, Ranks>);
    static constexpr int PrepareBytes = Groups * (Ranks + 4) * sizeof(uint32_t);
    static constexpr int SharedStorageSize = CopyBytes > PrepareBytes ? CopyBytes : PrepareBytes;

    __device__ static void run(const Arguments& sched, char* smem) {
        Dispatch::template prepare_copy<PrepareConfig, Threads>(sched, smem);
    }
};

}  // namespace deep_gemm
