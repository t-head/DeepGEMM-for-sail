#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>
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

// Turn CTA0's rank prefixes into row addresses. RowBytes is the physical byte stride,
// including packed FP4 rows.
template <uint32_t BlockM, uint32_t NumRanks,
          uint32_t RowBytes,
          uint32_t SourceCapacity, uint32_t TopK, class SchedulerArguments>
__device__ __forceinline__ void build_copy_row_map(
    const SchedulerArguments& sched, uint32_t expert_local,
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
        const uint8_t* source = sched.sym_buffer.map(sched.dispatch_data, owner);
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

// Compile-time indices keep bulk-copy temporaries in registers without tying
// the shared transport to one ACTLIZE generation.
template <class Function, size_t... Indices>
__device__ __forceinline__ void for_each_copy_index(std::index_sequence<Indices...>, Function f) {
    (f(std::integral_constant<size_t, Indices>{}), ...);
}

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
    for_each_copy_index(std::make_index_sequence<RowDepth>{}, [&](auto r) {
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
        copy_stripe(std::integral_constant<int, FullStripeChunks>{}, std::integral_constant<int, 0>{});
    } else {
        copy_stripe(std::integral_constant<int, LastStripeChunks>{}, std::integral_constant<int, HalfLastStripe>{});
    }
}

template <uint32_t BlockM, uint32_t NumRanks,
          uint32_t BlockK,
          uint32_t KTilesPerFlag, uint32_t ShapeK, uint32_t NumThreads,
          uint32_t SourceCapacity, uint32_t TopK, class SchedulerArguments, class BeforePublish>
__device__ void run_copy_block(
    const SchedulerArguments& sched, uint32_t total_m_blocks,
    char* smem_buf, uint32_t generation, BeforePublish before_publish)
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
                sched, gl.x, m_block_in_expert,
                generation,
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
                    local_block_dst, sched.sym_buffer.rank_idx,
                    stripe_int4s,
                    kRowVectors, stripe_col_start,
                    kStripeVectors, threadIdx.x,
                    NumThreads);
            // Finish stripe stores before publishing readiness.
            __ppu_threadfence_uncache();
            __syncthreads();
            if (threadIdx.x == 0) {
                before_publish(gl.x, ks + 1 == kNumStripes);
                __stbl(const_cast<uint32_t*>(sched.copy_ready_flags + mb),
                       (generation << 3) | (ks + 1u));
            }
        }
    }
}

}  // namespace deep_gemm::mega_moe_common
