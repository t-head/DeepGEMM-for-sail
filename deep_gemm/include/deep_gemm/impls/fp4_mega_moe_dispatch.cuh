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
using comm::resolve_fused_round_generation;

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
        prepare_expert_metadata<BlockM, kNumRanks, kNumLocalExperts, NumThreads>(
            sched, signals, generation, smem, local_pushed_counts,
            gridDim.x - TileScheduler::kNumCopyBlocks);
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
