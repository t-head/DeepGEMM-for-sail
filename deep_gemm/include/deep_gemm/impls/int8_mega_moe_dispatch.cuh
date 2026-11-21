#pragma once

#include <deep_gemm/impls/mega_moe_copy.cuh>
#include <deep_gemm/impls/mega_moe_dispatch.cuh>

#include <hggc_runtime.h>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/mega_moe_common.cuh>
#include <deep_gemm/comm/barrier.cuh>

namespace deep_gemm::sm80 {

using mega_moe_common::CopyRankMetadataShared;
using comm::resolve_fused_round_generation;

// INT8 shares route publication and counts with FP4, then compacts FP32 scales.
template <uint32_t BlockK, uint32_t SharedBytes = 0, Stage1RoutePolicy Routes = Stage1RoutePolicy::PerToken>
struct Int8FusedDispatch {
    template <class Scheduler>
    static constexpr uint32_t shared_bytes() {
        constexpr uint32_t prefix_bytes = (Scheduler::kNumRanks + 1) * sizeof(uint32_t);
        return SharedBytes > prefix_bytes ? SharedBytes : prefix_bytes;
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
        const auto* scales = buffers.staging_scales[1].data<float>(base);
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
                const auto source = load_route_source<ranks, capacity, topk>(tokens, prefix, expert, row);
                MegaMoEScaleFormat<true>::copy_vector(
                    scales + uint64_t(source.rank_idx) * capacity + source.token_idx,
                    sched.local_sfa_buf + row_base + row);
                __stbl(&sources[row_base + row].rank_idx, source.rank_idx);
                __stbl(&sources[row_base + row].token_idx, source.token_idx);
                __stbl(&sources[row_base + row].topk_idx, source.topk_idx);
            }
            __ppu_threadfence_uncache();
            __syncthreads();
            if (threadIdx.x == 0)
                __stbl(const_cast<uint32_t*>(ready + 1 + expert), generation);
            // Do not reuse prefix storage until every thread completed this row set.
            __syncthreads();
        }
    }

    __device__ static __forceinline__ void wait_metadata(
        const sched::Int8MegaMoESchedulerArguments& sched, uint32_t generation) {
        auto* ready = sched.sym_buffer.get_base_ptr<layout::MegaMoESignals*>()->preprocess_ready;
        if (threadIdx.x == 0) comm::wait_expert_metadata(ready, generation);
        __syncthreads();
    }

    template <class Scheduler, uint32_t Threads>
    __device__ static void prepare_copy(const sched::Int8MegaMoESchedulerArguments& sched, char* smem) {
        auto* signals = sched.sym_buffer.get_base_ptr<layout::MegaMoESignals*>();
        const uint32_t generation = resolve_fused_round_generation(*signals);
        auto* ready = signals->preprocess_ready;
        if constexpr (!DG_MEGAMOE_SPLIT_SYNC) {
            if (blockIdx.x == 0) {
                const layout::Int8MegaMoEBuffer buffers(
                    Scheduler::kNumRanks, Scheduler::kNumRanks * Scheduler::kNumGroups,
                    Scheduler::kSourceCapacity, Scheduler::kNumTopK, Scheduler::SHAPE_K,
                    Scheduler::SHAPE_N / 2, sched.staging);
                prepare_stage1_metadata<Scheduler, Threads>(sched, signals, buffers, smem, generation);
            }
            wait_metadata(sched, generation);
        }
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
        if constexpr (!DG_MEGAMOE_SPLIT_SYNC) {
            publish_stage1_routes<Scheduler, Threads, SharedBytes, Routes, false>(sched, smem, generation);
            // Copy CTAs build metadata while GEMM CTAs publish routes and scales.
            wait_metadata(sched, generation);
        }
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
    static constexpr int PrepareBytes = expert_metadata_smem_bytes(Groups, Ranks);
    static constexpr int SharedStorageSize = CopyBytes > PrepareBytes ? CopyBytes : PrepareBytes;

    __device__ static void run(const Arguments& sched, char* smem) {
        Int8FusedDispatch<BK>::template prepare_copy<PrepareConfig, Threads>(sched, smem);
    }
};

}  // namespace deep_gemm::sm80
