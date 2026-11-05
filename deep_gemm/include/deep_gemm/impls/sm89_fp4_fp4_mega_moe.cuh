#pragma once

// Include this first to select the Actlize epilogue parameter layout.
#include <accutlass.hpp>
#include <cstdint>
#include <hggc.h>
#include <hggc_bf16.h>
#include <hggc_runtime.h>

#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/sm80_mega_moe.cuh>

#include <cutlass/gemm/config/gemm_configs.hpp>
#include <cutlass/gemm/config/gemm_operands.hpp>
#include <deep_gemm/mma/fp4_mma.cuh>
#include <deep_gemm/epilogue/fp4_epilogue_silu_and_mul_post_quant.cuh>
#include <deep_gemm/epilogue/fp4_scatter_epilogue.cuh>
#include <deep_gemm/impls/fused_dispatch.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/tix/ld_st.cuh>

#ifndef DG_MEGAMOE_OWNER_READER_OPT
#define DG_MEGAMOE_OWNER_READER_OPT 1
#endif

namespace deep_gemm {
using namespace cute;

template <int N, int K, int BM, int BN, int BK, int WM, int WN,
          int Groups, int Stages, int Ranks, MegaMoEPhaseKind Stage,
          int CopyBlocks = 0, int KTilesPerFlag = 0,
          int SourceCapacity = 1, int TopK = 1,
          bool ApplyClamp = false, bool WideScatterOffset = false,
          bool WarpPipeline = false,
          GatedActivation Activation = GatedActivation::SwiGLU,
          bool kHasLinearBeta = true, bool FastMath = true>
struct MegaMoEFp4Kernel {
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
    using Scheduler = sched::SM80MegaMoEScheduler<
        N, K, BM, BN, Groups, Stage,
        Ranks, CopyBlocks, KTilesPerFlag, SourceCapacity, TopK>;
    struct Kernel {
        using ProblemShape = Shape<int, int, int, int>;
        using TileShape = typename Mainloop::TileShape;
        using MainloopParams = typename Mainloop::Params;
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
        static constexpr int SharedStorageSize = sizeof(SharedStorage);

        // Field order is shared with the host JIT launcher.
        struct ParamsBase {
            cutlass::gemm::GemmUniversalMode mode{};
            ProblemShape problem_shape{};
            MainloopParams mainloop{};
            typename Epilogue::Params epilogue{};
            cutlass::KernelHardwareInfo hw_info{};
            typename Scheduler::Params scheduler{};
            void* workspace{nullptr};
            int32_t* signal{nullptr};
#if SAIL_SYNC_IN_CE
            int* ptr_sync_ce = nullptr;
#endif
        };
        struct ScatterParams : ParamsBase {
            layout::SymBuffer<Ranks> scatter_buffer{};
            layout::ScatterSignals* scatter_barrier{nullptr};
        };
        using Params = std::conditional_t<kIsLinear1, ParamsBase, ScatterParams>;

        CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
            uint32_t generation = 0;
            if constexpr (kIsLinear1) {
                static_assert(Groups * (Ranks + 4) * sizeof(uint32_t) <= SharedStorageSize,
                              "dispatch preprocess must fit the GEMM shared allocation");
                static_assert(sizeof(CopyRankMetadataShared<BM, Ranks>) <= SharedStorageSize,
                              "copy rank metadata must fit the GEMM shared allocation");
                generation = resolve_fused_round_generation(params.scheduler);
                prepare_expert_layout_and_reshape_sfa<Scheduler, BM, MaxThreadsPerBlock>(
                    params.scheduler, smem_buf, generation);

                // Copy CTAs join the GEMM1 queue after finishing their own blocks.
                if (blockIdx.x < CopyBlocks) {
                    run_copy_block<BM, Ranks, BK, KTilesPerFlag, K, MaxThreadsPerBlock, SourceCapacity, TopK>(
                        params.scheduler, params.scheduler.copy_grouped_layout[0], smem_buf, generation);
                }
                __ppu_prefetch_KSD((void*)params.scheduler.copy_grouped_layout);
            } else {
                int warp = cutlass::canonical_warp_idx_sync();
                if (warp < cute::ceil_div(Groups, 32))
                    __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout + (warp << 5)));
            }

            auto& shared = *reinterpret_cast<SharedStorage*>(smem_buf);
            auto* epilogue_smem = reinterpret_cast<char*>(&shared.epilogue);
            const int n = kIsLinear1 ? get<1>(params.problem_shape) : N;
            const int k = kIsLinear1 ? get<2>(params.problem_shape) : K;
            const int batches = get<3>(params.problem_shape);
            Scheduler scheduler{params.scheduler};
            const int tid = threadIdx.x;
            uint32_t sequence = 0, m_block, n_block;

            // Hoist L1 staging pointers out of the LICM-disabled tile loop.
            const uint8_t* fp4 = nullptr;
            const uint16_t* sfa = nullptr;
            volatile uint32_t* ready = nullptr;
            if constexpr (kIsLinear1) {
                fp4 = params.scheduler.local_fp4_buf;
                sfa = params.scheduler.local_sfa_buf;
                ready = params.scheduler.copy_ready_flags;
            }
            auto mainloop = params.mainloop;
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

                    Epilogue consumer{params.epilogue, shared.epilogue};
                    while (scheduler.fetch_next_work(m_block, n_block)) {
                        auto shape = ProblemShape{scheduler.curr_problem_m(), n, k, batches};
                        consumer.consume(shape, TileShape{}, make_coord(m_block, n_block, _, 0),
                            tid - scatter_pipeline::kComputeThreads, epilogue_smem,
                            static_cast<uint32_t>(scheduler.curr_offset_m()), params.scatter_buffer, sequence);
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
                    const auto shape = ProblemShape{m, n, k, batches};
                    const auto coord = make_coord(m_block, n_block, _, 0);
                    MainloopParams tile{};
                    tile.problem_shape = {m, n, k};
                    tile.dA = mainloop.dA;
                    tile.dB = mainloop.dB;
                    tile.dSFA = mainloop.dSFA;
                    tile.dSFB = mainloop.dSFB;
                    if constexpr (kIsLinear1) {
                        const uint64_t base_m = scheduler.curr_cumsum_m;
                        tile.ptr_A = reinterpret_cast<const typename Mainloop::ElementA*>(fp4 + base_m * K);
                        tile.ptr_scale_A = sfa + base_m * ((K + 31u) / 32u);
                        get<1>(tile.dSFA) = static_cast<int64_t>(m);
                        tile.copy_ready_flags = ready + scheduler.curr_global_block_m_idx;
                        tile.copy_ready_generation = generation;
                        tile.copy_k_tiles_per_flag = KTilesPerFlag;
                        tile.sfa_ready_flags = params.scheduler.preprocess_ready_flags + 1 +
                            static_cast<uint32_t>(scheduler.problem_index()) * FusedSfaPartitionLayout::kNumPartitions;
                        tile.sfa_ready_generation = generation;
                    } else {
                        tile.ptr_A = mainloop.ptr_A + scheduler.curr_offset_a();
                        tile.ptr_scale_A = mainloop.ptr_scale_A + scheduler.curr_offset_mxfp4_scalea();
                    }
                    tile.ptr_B = mainloop.ptr_B + scheduler.curr_offset_b(m_block);
                    tile.ptr_scale_B = mainloop.ptr_scale_B + scheduler.curr_offset_mxfp4_scaleb(m_block);

                    Mma tiled_mma;
                    Tensor accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));
                    auto residue = Mainloop::compute_tile(tile, shape, coord, accum, tid, smem_buf);
                    auto output = params.epilogue;
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
                            static_cast<uint32_t>(scheduler.curr_offset_m()), params.scatter_buffer);
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
                comm::fp4_scatter_release<Ranks>(*params.scatter_barrier, params.scatter_buffer,
                    gridDim.x, blockIdx.x, threadIdx.x, BN == 256);
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
void sm89_fp4_fp4_mega_moe_l1(typename Kernel::Params params) {
    extern __shared__ char smem[];
    Kernel op;
    op(params, smem);
}

template <bool kBulkRemote>
__device__ __forceinline__ int4 moe_reduce_load(
        const __ppu_bfloat16* source, bool remote) {
    if constexpr (kBulkRemote) {
        if (remote) {
            const uint4 bulk = __ppu_global_ldcg_bulk_b32x4(source);
            return *reinterpret_cast<const int4*>(&bulk);
        }
    }
    return __ldcg(reinterpret_cast<const int4*>(source));
}

template <bool kCheckInvalid, int kTopK, int kHidden, bool kBulkRemote,
          int kThreads, int kColumnSplits = 1>
__device__ __forceinline__ void moe_reduce_token(
        const __ppu_bfloat16* __restrict__ input,
        __ppu_bfloat16* __restrict__ output,
        const int64_t (&slots)[kTopK], const float (&weights)[kTopK],
        const bool (&remote_slots)[kTopK], int token, int column_part, unsigned int reduce_thread) {
    constexpr int kVector = sizeof(int4) / sizeof(__ppu_bfloat16);
    constexpr int kStride = kThreads * kVector;
    constexpr int kColumns = kHidden / kColumnSplits;
    for (int column = column_part * kColumns + reduce_thread * kVector;
         column < (column_part + 1) * kColumns; column += kStride) {
        int4 inputs[kTopK];
        // The fused GEMM grid has fewer resident warps. Overlap TopK
        // reads before consuming them, preserving the original sum order.
        #pragma unroll
        for (int index = 0; index < kTopK; ++index) {
            if constexpr (kCheckInvalid) {
                if (slots[index] < 0) continue;
            }
            inputs[index] = moe_reduce_load<kBulkRemote>(
                input + slots[index] * kHidden + column, remote_slots[index]);
        }
        float accumulator[kVector] = {0.0f};
        #pragma unroll
        for (int index = 0; index < kTopK; ++index) {
            if constexpr (kCheckInvalid) {
                if (slots[index] < 0) continue;
            }
            const int4 packed = inputs[index];
            const auto* values = reinterpret_cast<const __ppu_bfloat16*>(&packed);
            #pragma unroll
            for (int lane = 0; lane < kVector; ++lane) {
                accumulator[lane] += __bfloat162float(values[lane]) * weights[index];
            }
        }
        int4 packed_output;
        auto* pairs = reinterpret_cast<__ppu_bfloat162*>(&packed_output);
        #pragma unroll
        for (int pair = 0; pair < kVector / 2; ++pair) {
            pairs[pair] = __floats2bfloat162_rn(
                accumulator[2 * pair], accumulator[2 * pair + 1]);
        }
        *reinterpret_cast<int4*>(output + static_cast<int64_t>(token) * kHidden + column) = packed_output;
    }
}

// GEMM2 scatter completes peer synchronization before this weighted reduction.
template <int kTopK, int kHidden, bool kBulkRemote, int kThreads, int kColumnSplits = 1>
__device__ __forceinline__ void moe_reduce_impl(
        const __ppu_bfloat16* __restrict__ input,
        const int64_t* __restrict__ topk_indices,
        const float* __restrict__ topk_weights,
        __ppu_bfloat16* __restrict__ output,
        int max_m, int local_experts, int rank_idx, int token, int column_part, uint64_t writer_mask, unsigned int reduce_thread) {
    static_assert(kThreads % 32 == 0);
    static_assert(kHidden % kColumnSplits == 0 && (kHidden / kColumnSplits) % 256 == 0,
                  "each column split must contain complete 256-element warp rows");
    int64_t slots[kTopK];
    float weights[kTopK];
    bool remote_slots[kTopK];
    bool all_valid = true;
    // Expert IDs are contiguous within each rank. Reuse the local interval
    // across TopK instead of dividing every expert ID by local_experts.
    const int local_expert_begin = rank_idx * local_experts;
    const int local_expert_end = local_expert_begin + local_experts;
    // Publication validates each int64 route before storing an in-range ID or -1.
    int32_t expert_ids[kTopK];
    bool metadata_valid = true;
    #pragma unroll
    for (int index = 0; index < kTopK; ++index) {
        expert_ids[index] = static_cast<int32_t>(__ldg(topk_indices + token * kTopK + index));
        metadata_valid &= expert_ids[index] >= 0;
    }
    if (metadata_valid) {
        // The common all-valid path has no per-slot load dependency on
        // validity, allowing route and weight metadata reads to overlap.
        #pragma unroll
        for (int index = 0; index < kTopK; ++index) {
            slots[index] = index * static_cast<int64_t>(max_m) + token;
            weights[index] = __ldg(topk_weights + token * kTopK + index);
            remote_slots[index] = kBulkRemote &&
                (expert_ids[index] < local_expert_begin || expert_ids[index] >= local_expert_end);
#if DG_MEGAMOE_OWNER_MODE
#if DG_MEGAMOE_OWNER_READER_OPT
            // Uniform all-bulk launches need no per-route owner calculation.
            if (writer_mask != UINT64_MAX && remote_slots[index])
#endif
                remote_slots[index] &= ((writer_mask >> (expert_ids[index] / local_experts)) & 1) != 0;
#endif
        }
        moe_reduce_token<false, kTopK, kHidden, kBulkRemote, kThreads, kColumnSplits>(
            input, output, slots, weights, remote_slots, token, column_part, reduce_thread);
        return;
    }
    #pragma unroll
    for (int index = 0; index < kTopK; ++index) {
        const int32_t expert = static_cast<int32_t>(
            __ldg(topk_indices + token * kTopK + index));
        const bool valid = expert >= 0;
        all_valid &= valid;
        if constexpr (kBulkRemote) {
            remote_slots[index] = valid &&
                (expert < local_expert_begin || expert >= local_expert_end);
        } else {
            remote_slots[index] = false;
        }
#if DG_MEGAMOE_OWNER_MODE
#if DG_MEGAMOE_OWNER_READER_OPT
        if (writer_mask != UINT64_MAX && remote_slots[index])
#else
        if (valid)
#endif
            remote_slots[index] &= ((writer_mask >> (expert / local_experts)) & 1) != 0;
#endif
        slots[index] = valid
            ? index * static_cast<int64_t>(max_m) + token
            : -1;
        weights[index] = valid
            ? __ldg(topk_weights + token * kTopK + index)
            : 0.0f;
    }

    // Skip invalid slots before loading scatter data, even when their weight is zero.
    if (all_valid) {
        moe_reduce_token<false, kTopK, kHidden, kBulkRemote, kThreads, kColumnSplits>(
            input, output, slots, weights, remote_slots, token, column_part, reduce_thread);
    } else {
        moe_reduce_token<true, kTopK, kHidden, kBulkRemote, kThreads, kColumnSplits>(
            input, output, slots, weights, remote_slots, token, column_part, reduce_thread);
    }
}

template <typename Kernel, int Ranks, int TopK, int Hidden, int Groups,
          int kColumnSplits, int kReduceGroupThreads>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock, Kernel::MinBlocksPerMultiprocessor)
void sm89_fp4_fp4_mega_moe_l2(typename Kernel::Params params, MegaMoEReduceParams reduce) {
    extern __shared__ char smem[];
    // Snapshot before GEMM's grid barrier advances the scatter phase.
    const int32_t reduce_phase = tix::ld_acq_sys(
        &params.scatter_barrier->phase_counter) & 3;
    Kernel op;
    op(params, smem);

    if constexpr (Ranks == 8 || DG_MEGAMOE_OWNER_MODE) {
        if (threadIdx.x == 0) {
            const uint64_t observed = comm::fp4_scatter_acquire<Ranks>(
                *params.scatter_barrier, reduce_phase);
#if DG_MEGAMOE_OWNER_MODE
            *reinterpret_cast<uint64_t*>(smem) = observed;
#endif
        }
        __syncthreads();
    }
    uint64_t writer_mask = UINT64_MAX;
#if DG_MEGAMOE_OWNER_MODE
    writer_mask = *reinterpret_cast<const uint64_t*>(smem);
#endif
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
            moe_reduce_impl<TopK, Hidden, true, kReduceThreads, kColumnSplits>(
                params.scatter_buffer.template get_base_ptr<const __ppu_bfloat16*>(),
                reduce.ids, reduce.weights, static_cast<__ppu_bfloat16*>(reduce.output),
                static_cast<int>(params.epilogue.token_capacity), Groups,
                params.scatter_buffer.rank_idx, work / kColumnSplits, work % kColumnSplits, writer_mask, reduce_thread);
        }
    }
}

}  // namespace deep_gemm
