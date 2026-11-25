#pragma once

#include <deep_gemm/impls/mega_moe_copy.cuh>
#include <deep_gemm/impls/expert_preprocess.cuh>

#include <hggc_runtime.h>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/mega_moe_common.cuh>
#include <deep_gemm/comm/barrier.cuh>

namespace deep_gemm::sm80 {

using mega_moe_common::CopyRankMetadataShared;
using comm::resolve_fused_round_generation;

// INT8 broadcasts contiguous source-token scales, then compacts expert rows.
// The publication, metadata, row copy and scheduler protocols remain shared.
template <uint32_t BlockK>
struct Int8FusedDispatch {
    template <class Scheduler>
    static constexpr uint32_t shared_bytes() {
        return (Scheduler::kNumRanks + 1) * sizeof(uint32_t);
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
        const layout::Int8MegaMoEBuffer buffers(
            ranks, ranks * groups, capacity, topk, Scheduler::SHAPE_K, Scheduler::SHAPE_N / 2,
            sched.staging);
        auto* prefix = reinterpret_cast<uint32_t*>(shared);
        auto* base = sched.sym_buffer.template get_base_ptr<layout::MegaMoESignals*>();
        auto* ready = base->preprocess_ready;
        const auto* scales = buffers.staging_scales[0].data<float>(base) +
            uint64_t(generation & 1u) * (sched.staging.scales / sizeof(float));
        const auto* tokens = sched.dispatch_token_topk +
            uint64_t(generation & 1u) * (sched.staging.tokens / sizeof(uint32_t));
        auto* sources = sched.src_info;
        for (uint32_t expert = blockIdx.x; expert < groups; expert += gridDim.x) {
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
                __stbl(const_cast<uint32_t*>(ready + 1 + expert), generation);
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
        const auto* counts = buffers.staging_counts[0].data<uint32_t>(base) +
            (generation & 1u) * (sched.staging.counts / sizeof(uint32_t));
        if (blockIdx.x == 0) {
            prepare_expert_metadata<Scheduler::BLOCK_M, ranks, groups, Threads, false>(
                sched, base, generation, reinterpret_cast<uint32_t*>(smem), counts);
        }
        wait_metadata(sched, generation);
    }

    __device__ static __forceinline__ void wait_metadata(
        const sched::Int8MegaMoESchedulerArguments& sched, uint32_t generation) {
        auto* ready = sched.sym_buffer.get_base_ptr<layout::MegaMoESignals*>()->preprocess_ready;
        if (threadIdx.x == 0)
            while (tix::ld_ready_flag_uncached(ready) !=
                   (kPreprocessMetadataReadyBit | generation)) { }
        __syncthreads();
    }

    template <class Scheduler, uint32_t Threads>
    __device__ static void prepare_copy(const sched::Int8MegaMoESchedulerArguments& sched, char* smem) {
        const uint32_t generation = resolve_fused_round_generation(
            *sched.sym_buffer.get_base_ptr<layout::MegaMoESignals*>());
        auto* ready = sched.sym_buffer.get_base_ptr<layout::MegaMoESignals*>()->preprocess_ready;
        prepare_metadata<Scheduler, Threads>(sched, smem, generation);
        mega_moe_common::run_copy_block<Scheduler::BLOCK_M, Scheduler::kNumRanks,
                       BlockK, Scheduler::kKTilesPerFlag, Scheduler::SHAPE_K, Threads,
                       Scheduler::kSourceCapacity, Scheduler::kNumTopK>(
            sched, sched.copy_grouped_layout[0], smem, generation,
            [&](uint32_t expert, bool last_stripe) {
                // Only the last payload stripe joins per-token scale compaction.
                if (last_stripe)
                    while (tix::ld_ready_flag_uncached(ready + 1 + expert) != generation) { }
            });
    }

    template <class Scheduler, uint32_t Threads>
    __device__ static void prepare(const typename Scheduler::Arguments& sched, char* smem) {
        const uint32_t generation = resolve_fused_round_generation(
            *sched.sym_buffer.template get_base_ptr<layout::MegaMoESignals*>());
        // Independent copy CTAs publish metadata. GEMM CTAs compact scales
        // and consume payload stripes through the mainloop readiness checks.
        wait_metadata(sched, generation);
        reshape_scales<Scheduler, Threads>(sched, smem, generation);
        __threadfence();
    }
};

// Copy allocates only metadata, independently of the GEMM operand pipeline.
template <int N, int K, int BM, int BK, int Groups, int Ranks,
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
        Int8FusedDispatch<BK>::template prepare_copy<PrepareConfig, Threads>(sched, smem);
    }
};

}  // namespace deep_gemm::sm80
