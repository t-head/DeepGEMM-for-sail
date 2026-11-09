#pragma once

// Include this first to select the Actlize epilogue parameter layout.
#include <accutlass.hpp>
#include <cstdint>
#include <hggc.h>
#include <hggc_bf16.h>
#include <hggc_runtime.h>

#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/dispatch_layout.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/sm89_mega_moe.cuh>

#include <cutlass/gemm/config/gemm_configs.hpp>
#include <cutlass/gemm/config/gemm_operands.hpp>
#include <deep_gemm/mma/fp4_mma.cuh>
#include <deep_gemm/epilogue/fp4_epilogue_silu_and_mul_post_quant.cuh>
#include <deep_gemm/epilogue/fp4_scatter_epilogue.cuh>
#include <deep_gemm/impls/fused_dispatch.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/epilogue/mega_moe_reduce.cuh>

namespace deep_gemm {
using namespace cute;

template <int N, int K, int BM, int BN, int BK, int WM, int WN,
          int Groups, int Stages, int Ranks, MegaMoEPhaseKind Stage,
          int CopyBlocks = 0, int KTilesPerFlag = 0,
          int SourceCapacity = 1, int TopK = 1,
          bool ApplyClamp = false, bool WideScatterOffset = false,
          bool WarpPipeline = false,
          GatedActivation Activation = GatedActivation::SwiGLU,
          bool kHasLinearBeta = true, bool FastMath = true, bool HasReplicas = false>
struct MegaMoEMxfp4Kernel {
    static_assert(Stage == MegaMoEPhaseKind::Linear1 || Stage == MegaMoEPhaseKind::Linear2,
                  "unsupported MegaMoE stage");
    static constexpr bool kIsLinear1 = Stage == MegaMoEPhaseKind::Linear1;
    static_assert(!WarpPipeline || (!kIsLinear1 && BM == 128 && BN == 256 &&
                  BK == 128 && WM == 64 && WN == 64 && Stages == 3 && N % BN == 0),
                  "unsupported scatter warp pipeline geometry");
    using WarpOnM = Int<BM / WM>;
    using WarpOnN = Int<BN / WN>;
    using Arch = cutlass::arch::PPU0015;
    using StrideCD = cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>;
    using ElementD = std::conditional_t<kIsLinear1, uint8_t, cutlass::bfloat16_t>;
    static constexpr auto kGemmType = GemmType::MegaMoE;
    static constexpr auto kEpilogue = kIsLinear1 ? EpilogueType::SiluAndMulPostQuantFp4 : EpilogueType::Default;

    using OperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<
        Arch, cutlass::float4_t, false, Int<BM>, Int<BK>, false>;
    using OperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<
        Arch, cutlass::float4_t, false, Int<BN>, Int<BK>, true>;
    using OperandSFA = cutlass::gemm::config::DefaultGemm_AIU_Operand<
        Arch, uint16_t, true, Int<(BM < 16 ? 16 : BM)>, Int<BK / 32>, true, 0, false>;
    using OperandSFB = cutlass::gemm::config::DefaultGemm_AIU_Operand<
        Arch, uint16_t, true, Int<(BN < 16 ? 16 : BN)>, Int<BK / 32>, true, 0, false>;
    using Mma = TiledMMA<MMA_Atom<PPU0015_16x16x64_F32F4F4F32_TN>,
                         Layout<Shape<WarpOnM, WarpOnN, _1>>>;
    using Mainloop = cutlass::gemm::collective::CollectiveMmaScaleFp4<
        cutlass::gemm::MainloopWithScalePPU0015Aiu<Stages>, Shape<Int<BM>, Int<BN>, Int<BK>>,
        cutlass::float4_t, cutlass::detail::TagToStrideA_t<cutlass::layout::RowMajor>,
        cutlass::float4_t, cutlass::detail::TagToStrideB_t<cutlass::layout::ColumnMajor>, Mma,
        typename OperandA::GmemTiledCopy, typename OperandA::SmemLayoutAtom,
        typename OperandA::SmemCopyAtom, identity,
        typename OperandB::GmemTiledCopy, typename OperandB::SmemLayoutAtom,
        typename OperandB::SmemCopyAtom, identity,
        uint16_t, cutlass::detail::TagToStrideA_t<cutlass::layout::ColumnMajor>,
        typename OperandSFA::GmemTiledCopy, typename OperandSFA::SmemLayoutAtom,
        uint16_t, cutlass::detail::TagToStrideB_t<cutlass::layout::RowMajor>,
        typename OperandSFB::GmemTiledCopy, typename OperandSFB::SmemLayoutAtom,
        FP4DynamicTileId::Disabled,
        kIsLinear1 ? math::ceil_div(K * 2, 64) : 0,
        WarpPipeline>;
    using ActivationEpilogue = cutlass::epilogue::collective::EpilogueSiluAndMulPostQuant<
        StrideCD, StrideCD,
        cutlass::epilogue::thread::LinearCombination<
            ElementD, 2, float, float, cutlass::epilogue::thread::ScaleType::Nothing,
            cutlass::FloatRoundStyle::round_to_nearest, float>,
        cutlass::gemm::EpilogueDefault, N % BN == 0, ApplyClamp, kEpilogue,
        GatedActivationOp<Activation, ApplyClamp, kHasLinearBeta, FastMath>>;
    using ScatterEpilogue = epilogue::Fp4ScatterEpilogue<
        StrideCD, StrideCD, Layout<Shape<Int<(BM / WM) * 16>, Int<BN>>, cute::Stride<Int<BN>, _1>>,
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, float>, float, ElementD,
        cutlass::epilogue::thread::LinearCombination<ElementD, 16, float, float>,
        Ranks, WideScatterOffset, WarpPipeline>;
    using Epilogue = std::conditional_t<kIsLinear1, ActivationEpilogue, ScatterEpilogue>;
    using Scheduler = sched::SM89MegaMoEScheduler<
        N, K, BM, BN, Groups, Stage,
        Ranks, CopyBlocks, KTilesPerFlag, SourceCapacity, TopK>;
    struct Kernel {
        using ProblemShape = Shape<int, int, int, int>;
        using TileShape = typename Mainloop::TileShape;
        using MainloopArguments = typename Mainloop::ArgumentsBase;
        using EpilogueParams = typename Epilogue::Params;
        using SchedulerParams = typename Scheduler::Params;
        static constexpr uint32_t MaxThreadsPerBlock =
            WarpPipeline ? scatter_pipeline::kCtaThreads : CUTE_STATIC_V(size(Mma{}));
        static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

        union OverlaidStorage {
            typename Mainloop::SharedStorage mainloop;
            typename Epilogue::SharedStorage epilogue;
        };
        struct PipelineStorage {
            typename Mainloop::SharedStorage mainloop;
            typename Epilogue::SharedStorage epilogue;
        };
        using SharedStorage = std::conditional_t<WarpPipeline, PipelineStorage, OverlaidStorage>;
        static constexpr int SharedStorageSize = cute::max(static_cast<int>(sizeof(SharedStorage)),
            kIsLinear1 ? Groups * (Ranks + kExpertMetadataAuxFields) * static_cast<int>(sizeof(uint32_t)) : 0);

        CUTLASS_DEVICE void operator()(MainloopArguments mainloop, const EpilogueParams& epilogue_params,
                                      const SchedulerParams& scheduler_params, layout::MegaMoESignals* signals,
                                      char* smem_buf, const MegaMoEReplicaWeights& replicas, uint32_t generation,
                                      const layout::SymBuffer<Ranks>* scatter_buffer = nullptr) {
            if constexpr (kIsLinear1) {
                static_assert(Groups * (Ranks + kExpertMetadataAuxFields) * sizeof(uint32_t) <= SharedStorageSize,
                              "dispatch preprocess must fit the GEMM shared allocation");
                static_assert(sizeof(CopyRankMetadataShared<BM, Ranks>) <= SharedStorageSize,
                              "copy rank metadata must fit the GEMM shared allocation");
                prepare_expert_layout_and_reshape_sfa<Scheduler, BM, MaxThreadsPerBlock>(
                    scheduler_params, signals, smem_buf, generation);

                // Copy CTAs join the GEMM1 queue after finishing their own blocks.
                if (blockIdx.x < CopyBlocks) {
                    run_copy_block<BM, Ranks, BK, KTilesPerFlag, K, MaxThreadsPerBlock, SourceCapacity, TopK>(
                        scheduler_params, scheduler_params.copy_grouped_layout[0], smem_buf, generation);
                }
                __ppu_prefetch_KSD((void*)scheduler_params.copy_grouped_layout);
            } else {
                int warp = cutlass::canonical_warp_idx_sync();
                if (warp < cute::ceil_div(Groups, 32))
                    __ppu_prefetch_KSD((void*)(scheduler_params.grouped_layout + (warp << 5)));
            }

            auto& shared = *reinterpret_cast<SharedStorage*>(smem_buf);
            auto* epilogue_smem = reinterpret_cast<char*>(&shared.epilogue);
            const int n = kIsLinear1 ? get<1>(mainloop.problem_shape) : N;
            const int k = kIsLinear1 ? get<2>(mainloop.problem_shape) : K;
            Scheduler scheduler{scheduler_params};
            const int tid = threadIdx.x;
            uint32_t sequence = 0, m_block, n_block;

            // Hoist L1 staging pointers out of the LICM-disabled tile loop.
            const uint8_t* fp4 = nullptr;
            const uint16_t* sfa = nullptr;
            volatile uint32_t* ready = nullptr;
            if constexpr (kIsLinear1) {
                fp4 = scheduler_params.local_fp4_buf;
                sfa = scheduler_params.local_sfa_buf;
                ready = scheduler_params.copy_ready_flags;
            }
            if constexpr (!kIsLinear1) {
                get<0>(mainloop.dA) = K;
                get<0>(mainloop.dB) = K;
                get<1>(mainloop.dSFB) = N;
            }

            if (WarpPipeline && tid >= scatter_pipeline::kComputeThreads) {
                if constexpr (WarpPipeline) {
                    static_assert(CUTE_STATIC_V(size(Mma{})) == scatter_pipeline::kComputeThreads);
                    CUTLASS_PRAGMA_UNROLL
                    for (int slot = 0; slot < Epilogue::kPipelineSlots; ++slot)
                        __ppu_barrier_arrive(Epilogue::kFreeBarrier + slot, scatter_pipeline::kCtaThreads, 0);

                    Epilogue consumer{epilogue_params, shared.epilogue};
                    while (scheduler.fetch_next_work(m_block, n_block)) {
                        auto shape = ProblemShape{scheduler.curr_problem_m(), n, k, 1};
                        consumer.consume(shape, TileShape{}, make_coord(m_block, n_block, _, 0),
                            tid - scatter_pipeline::kComputeThreads, epilogue_smem,
                            static_cast<uint32_t>(scheduler.curr_offset_m()), *scatter_buffer, sequence);
                    }
                }
            } else {
                #pragma clang loop licm(disable)
                while (true) {
                    if constexpr (kIsLinear1) {
                        if (!scheduler.fetch_next_work(m_block, n_block, reinterpret_cast<uint32_t*>(smem_buf)))
                            break;
                    } else if (!scheduler.fetch_next_work(m_block, n_block)) {
                        break;
                    }
                    const uint32_t m = scheduler.curr_problem_m();
                    const auto shape = ProblemShape{m, n, k, 1};
                    const auto coord = make_coord(m_block, n_block, _, 0);
                    typename Mainloop::Params tile{};
                    tile.problem_shape = {m, n, k};
                    tile.dA = mainloop.dA;
                    tile.dB = mainloop.dB;
                    tile.dSFA = mainloop.dSFA;
                    tile.dSFB = mainloop.dSFB;
                    if constexpr (kIsLinear1) {
                        const uint64_t base_m = scheduler.curr_cumsum_m;
                        tile.ptr_A = reinterpret_cast<const typename Mainloop::ElementA*>(fp4 + base_m * K);
                        const auto sfa_base_m = FusedSfaPartitionLayout::row_offset(base_m, scheduler.problem_index());
                        tile.ptr_scale_A = sfa + sfa_base_m * ((K + 31u) / 32u);
                        get<1>(tile.dSFA) = FusedSfaPartitionLayout::align_rows(m);
                        tile.copy_ready_flags = ready + scheduler.curr_global_block_m_idx;
                        tile.generation = generation;
                        tile.copy_k_tiles_per_flag = KTilesPerFlag;
                        tile.sfa_ready_flags = signals->preprocess_ready + 1 +
                            static_cast<uint32_t>(scheduler.problem_index()) * FusedSfaPartitionLayout::kNumPartitions;
                    } else {
                        tile.ptr_A = mainloop.ptr_A + scheduler.curr_offset_a();
                        tile.ptr_scale_A = mainloop.ptr_scale_A + scheduler.curr_offset_mxfp4_scalea();
                    }
                    if constexpr (HasReplicas) {
                        int expert = scheduler.problem_index();
                        const bool is_replica = expert >= replicas.num_master_experts;
                        if (is_replica) expert -= replicas.num_master_experts;
                        const auto* weights = is_replica
                            ? reinterpret_cast<const typename Mainloop::ElementB*>(replicas.weights) : mainloop.ptr_B;
                        const auto* scales = is_replica ? replicas.scales : mainloop.ptr_scale_B;
                        tile.ptr_B = weights + static_cast<int64_t>(expert) * N * K;
                        tile.ptr_scale_B = scales + static_cast<int64_t>(expert) * N * math::ceil_div(K, 32);
                    } else {
                        tile.ptr_B = mainloop.ptr_B + scheduler.curr_offset_b(m_block);
                        tile.ptr_scale_B = mainloop.ptr_scale_B + scheduler.curr_offset_mxfp4_scaleb(m_block);
                    }

                    Mma tiled_mma;
                    Tensor accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));
                    auto residue = Mainloop::compute_tile(tile, shape, coord, accum, tid, smem_buf);
                    auto output = epilogue_params;
                    if constexpr (kIsLinear1) {
                        output.ptr_D += scheduler.curr_offset_c();
                        output.ptr_SFD += scheduler.curr_offset_c_scales();
                    }
                    Epilogue epilogue{output, shared.epilogue};
                    if constexpr (kIsLinear1) {
                        epilogue(shape, TileShape{}, coord, accum, tiled_mma, residue, tid, epilogue_smem);
                    } else if constexpr (WarpPipeline) {
                        epilogue.produce(TileShape{}, m_block, accum, tiled_mma, tid, epilogue_smem, m, sequence);
                    } else {
                        epilogue(shape, TileShape{}, coord, accum, tiled_mma, residue, tid, epilogue_smem,
                            static_cast<uint32_t>(scheduler.problem_index()),
                            static_cast<uint32_t>(scheduler.curr_offset_m()), *scatter_buffer);
                    }
                }

                // Only producers drain, including initially-free slots on CTAs without work.
                if constexpr (WarpPipeline) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int slot = 0; slot < Epilogue::kPipelineSlots; ++slot)
                        __ppu_barrier_sync(Epilogue::kFreeBarrier + slot, scatter_pipeline::kCtaThreads);
                }
            }

            if constexpr (WarpPipeline) __syncthreads();
            if constexpr (!kIsLinear1) {
                comm::fp4_scatter_release<Ranks>(*signals, *scatter_buffer,
                    gridDim.x, blockIdx.x, threadIdx.x, generation, BN == 256);
            }
        }
    };

    static_assert(is_supported_megamoe_rank_count(Ranks), "unsupported MegaMoE rank count");
    static_assert(BM == 16 || BM == 32 || BM == 64 || BM == 128 || BM == 256,
                  "unsupported MegaMoE block M");
    static_assert(!kIsLinear1 ||
                  (K > 0 && N > 0 && Groups > 0 && N % 64 == 0 &&
                   Kernel::MaxThreadsPerBlock >= 64 &&
                   CopyBlocks > 0 && KTilesPerFlag > 0 && SourceCapacity > 0 && TopK > 0 &&
                   math::ceil_div(K / BK, KTilesPerFlag) <= 7 &&
                   ActivationEpilogue::get_shared_storage_size(BM, BN) <= Kernel::SharedStorageSize),
                  "invalid fused dispatch resources");
};

template <typename Kernel>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock, Kernel::MinBlocksPerMultiprocessor)
void sm89_fp4_fp4_mega_moe_l1(typename Kernel::MainloopArguments mainloop, typename Kernel::EpilogueParams epilogue,
                              typename Kernel::SchedulerParams scheduler, layout::MegaMoESignals* signals,
                              MegaMoEReplicaWeights replicas) {
    extern __shared__ char smem[];
    Kernel op;
    op(mainloop, epilogue, scheduler, signals, smem, replicas, resolve_fused_round_generation(*signals));
}

template <typename Kernel, int Ranks, int TopK, int Hidden, int Groups,
          int kColumnSplits, int kReduceGroupThreads, bool kOutputAligned>
__device__ __forceinline__ void mega_moe_reduce(
        const layout::SymBuffer<Ranks>& scatter_buffer, uint32_t token_capacity,
        const MegaMoEReduceParams& reduce, uint64_t writer_mask) {
    constexpr int kReduceThreads = kReduceGroupThreads > 0 ? kReduceGroupThreads : Kernel::MaxThreadsPerBlock;
    constexpr int kTokenGroups = Kernel::MaxThreadsPerBlock / kReduceThreads;
    unsigned int reduce_thread = threadIdx.x;
    int token_group = 0;
    if constexpr (kReduceGroupThreads > 0) {
        reduce_thread = threadIdx.x % kReduceThreads;
        // Group boundaries are warp-aligned; keep token metadata and owner decisions scalar.
        token_group = __ppu_read_firstlane(threadIdx.x / kReduceThreads);
    }
    if (kColumnSplits == 1 || reduce_thread * 8 < Hidden / kColumnSplits) {
        for (int work = blockIdx.x * kTokenGroups + token_group;
             work < reduce.tokens * kColumnSplits; work += gridDim.x * kTokenGroups) {
            moe_reduce_impl<kOutputAligned, TopK, Hidden, true, kReduceThreads, kColumnSplits>(
                scatter_buffer.template get_base_ptr<const __ppu_bfloat16*>(),
                reduce.ids, reduce.weights, static_cast<__ppu_bfloat16*>(reduce.output),
                static_cast<int>(token_capacity), Groups,
                scatter_buffer.rank_idx, work / kColumnSplits, work % kColumnSplits, writer_mask, reduce_thread);
        }
    }
}

template <typename Kernel, int Ranks, int TopK, int Hidden, int Groups,
          int kColumnSplits, int kReduceGroupThreads, bool kOutputAligned>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock, Kernel::MinBlocksPerMultiprocessor)
void sm89_fp4_fp4_mega_moe_l2(typename Kernel::MainloopArguments mainloop, typename Kernel::EpilogueParams epilogue,
                              typename Kernel::SchedulerParams scheduler, layout::MegaMoESignals* signals,
                              layout::SymBuffer<Ranks> scatter_buffer, MegaMoEReduceParams reduce,
                              MegaMoEReplicaWeights replicas) {
    extern __shared__ char smem[];
    const uint32_t generation = __ldbl(signals->preprocess_ready) & ~kPreprocessMetadataReadyBit;
    Kernel op;
    op(mainloop, epilogue, scheduler, signals, smem, replicas, generation, &scatter_buffer);
    if constexpr (!DG_MEGAMOE_SPLIT_SYNC) {
        comm::rank_barrier<Ranks, true>(signals->scatter_arrivals, generation,
                                        reinterpret_cast<uint32_t*>(smem));
        mega_moe_reduce<Kernel, Ranks, TopK, Hidden, Groups, kColumnSplits, kReduceGroupThreads, kOutputAligned>(
            scatter_buffer, epilogue.token_capacity, reduce, *reinterpret_cast<const uint64_t*>(smem));
    }
}

template <typename Kernel, int Ranks, int TopK, int Hidden, int Groups,
          int kColumnSplits, int kReduceGroupThreads, bool kOutputAligned>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock, Kernel::MinBlocksPerMultiprocessor)
void sm89_fp4_fp4_mega_moe_reduce(layout::SymBuffer<Ranks> scatter_buffer, uint32_t token_capacity,
                                  const layout::MegaMoESignals* signals, MegaMoEReduceParams reduce) {
    mega_moe_reduce<Kernel, Ranks, TopK, Hidden, Groups, kColumnSplits, kReduceGroupThreads, kOutputAligned>(
        scatter_buffer, token_capacity, reduce, signals->scatter_writer_mask);
}

}  // namespace deep_gemm
