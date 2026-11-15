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
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>

#include <cutlass/gemm/config/gemm_configs.hpp>
#include <cutlass/gemm/config/gemm_operands.hpp>
#include <deep_gemm/mma/fp4_mma.cuh>
#include <deep_gemm/epilogue/fp4_epilogue_silu_and_mul_post_quant.cuh>
#include <deep_gemm/epilogue/mega_moe_l2_epilogue.cuh>
#include <deep_gemm/impls/fp4_mega_moe_dispatch.cuh>
#include <deep_gemm/comm/barrier.cuh>

namespace deep_gemm {
using namespace cute;

template <int N, int K, int BM, int BN, int BK, int WM, int WN,
          int Groups, int Stages, int Ranks, MegaMoEPhaseKind Stage,
          int CopyBlocks = 0, int KTilesPerFlag = 0,
          int SourceCapacity = 1, int TopK = 1,
          bool ApplyClamp = false, bool WideScatterOffset = false,
          bool WarpPipeline = false,
          GatedActivation Activation = GatedActivation::SwiGLU,
          bool kHasLinearBeta = true, bool FastMath = true, bool HasReplicas = false, bool IsShared = false>
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
        kIsLinear1 && !IsShared ? math::ceil_div(K * 2, 64) : 0,
        WarpPipeline>;
    using ActivationEpilogue = cutlass::epilogue::collective::EpilogueSiluAndMulPostQuant<
        StrideCD, StrideCD,
        cutlass::epilogue::thread::LinearCombination<
            ElementD, 2, float, float, cutlass::epilogue::thread::ScaleType::Nothing,
            cutlass::FloatRoundStyle::round_to_nearest, float>,
        cutlass::gemm::EpilogueDefault, N % BN == 0, ApplyClamp, kEpilogue,
        GatedActivationOp<Activation, ApplyClamp, kHasLinearBeta, FastMath>>;
    using ScatterEpilogue = epilogue::MegaMoEScatterEpilogue<
        StrideCD, StrideCD, Layout<Shape<Int<(BM / WM) * 16>, Int<BN>>, cute::Stride<Int<BN>, _1>>,
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, float>, float, ElementD,
        cutlass::epilogue::thread::LinearCombination<ElementD, 16, float, float>,
        Ranks, WideScatterOffset, WarpPipeline>;
    using DenseEpilogue = cutlass::epilogue::collective::DefaultEpilogueNoTsm<
        StrideCD, StrideCD, cutlass::epilogue::thread::LinearCombination<
            ElementD, 2, float, float, cutlass::epilogue::thread::ScaleType::Nothing,
            cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::EpilogueDefault, N % BN == 0>;
    using Epilogue = std::conditional_t<kIsLinear1, ActivationEpilogue,
                     std::conditional_t<IsShared, DenseEpilogue, ScatterEpilogue>>;
    using RoutedScheduler = sched::SM89MegaMoEScheduler<
        N, K, BM, BN, Groups, Stage,
        Ranks, CopyBlocks, KTilesPerFlag, SourceCapacity, TopK>;
    using DenseScheduler = DeepGemmScheduler<GemmType::Normal, N, K, BM, BN, 1,
        math::ceil_div(N, BN), 2, kEpilogue>;
    using Scheduler = std::conditional_t<IsShared, DenseScheduler, RoutedScheduler>;
    struct Kernel {
        template<uint32_t Columns, uint32_t CapacityRows = 0>
        __device__ static __forceinline__ void map_shared_ticket(
            uint32_t ticket, uint32_t rows, uint32_t& m, uint32_t& n) {
            if constexpr (CapacityRows > 0 && CapacityRows <= 8) {
                if (rows == CapacityRows) {
                    m = ticket % CapacityRows;
                    n = ticket / CapacityRows;
                    return;
                }
            }
            constexpr uint32_t span = 8 * Columns;
            const uint32_t first = (ticket / span) * 8;
            const uint32_t offset = ticket % span;
            if (first + 8 <= rows) {
                m = first + (offset % 8);
                n = offset / 8;
            } else {
                const uint32_t tail = rows - first;
                m = first + (offset % tail);
                n = offset / tail;
            }
        }
        static constexpr bool kThreeSlots = WarpPipeline;
        static constexpr int kSharedCols = BN, kSharedRows = BM;
        using MainloopType = Mainloop;
        using EpilogueType = Epilogue;
        static constexpr bool kCanUnifyStage2 = !kIsLinear1 && !IsShared && !WarpPipeline;
        using ProblemShape = Shape<int, int, int, int>;
        using TileShape = typename Mainloop::TileShape;
        using MainloopArguments = typename Mainloop::ArgumentsBase;
        using MainloopParams = typename Mainloop::Params;
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
            kIsLinear1 && !IsShared ? Groups * (Ranks + kExpertMetadataAuxFields) * static_cast<int>(sizeof(uint32_t)) : 0);

        struct ParamsBase {
            static constexpr int kShapeN = N;
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
        using Params = ParamsBase;

        __device__ static __forceinline__ void prepare_weight_addresses(
                MainloopParams& tile, const MainloopArguments& mainloop, Scheduler& scheduler,
                uint32_t m_block, const MegaMoEReplicaWeights& replicas) {
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
        }

        // Routed first, then dynamically claim shared tiles in the same MMA loop.
        template<class Shared>
        CUTLASS_DEVICE void run_shared_routed(MainloopArguments mainloop, const EpilogueParams& epilogue_params,
                const SchedulerParams& scheduler_params, layout::MegaMoESignals* signals,
                typename Shared::Params const& dense_params, uint32_t dense_m,
                char* smem_buf, MegaMoEReplicaWeights const& replicas, int32_t* dense_counter, const layout::SymBuffer<Ranks>& scatter_buffer, uint32_t generation) {
            static_assert(kCanUnifyStage2);
            static_assert(std::is_same_v<Mainloop, typename Shared::MainloopType>);
            Scheduler scheduler{scheduler_params};
            auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
            auto& dense_storage = *reinterpret_cast<typename Shared::SharedStorage*>(smem_buf);
            const int tid = threadIdx.x;
            constexpr uint32_t dense_columns = math::ceil_div(Shared::Params::kShapeN, BN);
            const uint32_t dense_end = math::ceil_div(dense_m, uint32_t(BM)) * dense_columns;
            bool routed_phase = true;
            uint32_t m_block, n_block;
            int warp = cutlass::canonical_warp_idx_sync();
            if (warp < cute::ceil_div(Groups, 32))
                __ppu_prefetch_KSD((void*)(scheduler_params.grouped_layout + (warp << 5)));
            #pragma clang loop licm(disable)
            while (true) {
                bool dense = !routed_phase;
                if (routed_phase && !scheduler.fetch_next_work(m_block, n_block)) {
                    routed_phase = false;
                    dense = true;
                }
                if (dense) {
                    auto* ticket = reinterpret_cast<uint32_t*>(smem_buf);
                    // Join the previous epilogue before reusing shared memory for the ticket.
                    __syncthreads();
                    if (tid == 0) *ticket = atomicAdd(dense_counter, 1);
                    __syncthreads();
                    const uint32_t task = *reinterpret_cast<volatile uint32_t*>(ticket);
                    __syncthreads();
                    if (task >= dense_end) break;
                    map_shared_ticket<dense_columns>(task, math::ceil_div(dense_m, uint32_t(BM)), m_block, n_block);
                }

                const MainloopParams* input = dense ? &dense_params.mainloop : &mainloop;
                MainloopParams tile = *input;
                const auto shape = dense
                    ? ProblemShape{int(dense_m), get<1>(dense_params.problem_shape), get<2>(dense_params.problem_shape), get<3>(dense_params.problem_shape)}
                    : ProblemShape{scheduler.curr_problem_m(), N, K, 1};
                tile.problem_shape = {get<0>(shape), get<1>(shape), get<2>(shape)};
                if (!dense) {
                    get<0>(tile.dA) = K;
                    get<0>(tile.dB) = K;
                    get<1>(tile.dSFB) = N;
                    tile.ptr_A += scheduler.curr_offset_a();
                    tile.ptr_scale_A += scheduler.curr_offset_mxfp4_scalea();
                    prepare_weight_addresses(tile, mainloop, scheduler, m_block, replicas);
                }
                const auto coord = make_coord(m_block, n_block, _, 0);
                Mma tiled_mma;
                Tensor accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));
                auto residue = Mainloop::compute_tile(tile, shape, coord, accum, tid, smem_buf);
                if (dense) {
                    typename Shared::EpilogueType epilogue{dense_params.epilogue, dense_storage.epilogue};
                    epilogue(shape, TileShape{}, coord, accum, tiled_mma, residue, tid,
                        reinterpret_cast<char*>(&dense_storage.epilogue));
                } else {
                    Epilogue epilogue{epilogue_params, storage.epilogue};
                    epilogue(shape, TileShape{}, coord, accum, tiled_mma, residue, tid,
                        reinterpret_cast<char*>(&storage.epilogue),
                        static_cast<uint32_t>(scheduler.problem_index()),
                        static_cast<uint32_t>(scheduler.curr_offset_m()), scatter_buffer);
                }
            }
            comm::scatter_release<Ranks>(*signals, scatter_buffer,
                gridDim.x, blockIdx.x, threadIdx.x, generation, BN == 256);
        }

        template<class Dense = void, bool kVectorStore = false, class DenseParams = decltype(nullptr)>
        CUTLASS_DEVICE void operator()(MainloopArguments mainloop, const EpilogueParams& epilogue_params,
                                      const SchedulerParams& scheduler_params, layout::MegaMoESignals* signals,
                                      char* smem_buf, const MegaMoEReplicaWeights& replicas, uint32_t generation,
                                      const layout::SymBuffer<Ranks>* scatter_buffer = nullptr,
                                      DenseParams const& dense = nullptr, uint32_t rows = 0) {
            if constexpr (kIsLinear1 && !IsShared) {
                static_assert(Groups * (Ranks + kExpertMetadataAuxFields) * sizeof(uint32_t) <= SharedStorageSize,
                              "dispatch preprocess must fit the GEMM shared allocation");
                static_assert(sizeof(CopyRankMetadataShared<BM, Ranks>) <= SharedStorageSize,
                              "copy rank metadata must fit the GEMM shared allocation");
                prepare_expert_layout_and_reshape_sfa<Scheduler, BM, MaxThreadsPerBlock>(
                    scheduler_params, signals, smem_buf, generation);

                // Copy CTAs join the GEMM1 queue after finishing their own blocks.
                if (blockIdx.x < CopyBlocks) {
                    mega_moe_common::run_copy_block<BM, Ranks, BK, KTilesPerFlag, K, MaxThreadsPerBlock, SourceCapacity, TopK>(
                        scheduler_params, scheduler_params.copy_grouped_layout[0], smem_buf, generation,
                        [](uint32_t, bool) {});
                }
                __ppu_prefetch_KSD((void*)scheduler_params.copy_grouped_layout);
                if constexpr (!std::is_void_v<Dense>)
                    run_shared_prefix<Dense>(dense, rows, smem_buf);
            } else if constexpr (!IsShared) {
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
            if constexpr (kIsLinear1 && !IsShared) {
                fp4 = scheduler_params.local_data_buf;
                sfa = scheduler_params.local_sfa_buf;
                ready = scheduler_params.copy_ready_flags;
            }
            if constexpr (!kIsLinear1 && !IsShared) {
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
                    if constexpr (kIsLinear1 && !IsShared) {
                        if (!scheduler.fetch_next_work(m_block, n_block, reinterpret_cast<uint32_t*>(smem_buf)))
                            break;
                    } else if constexpr (IsShared) {
                        constexpr auto cols = math::ceil_div(N, BN);
                        // Give the last partial shared wave to the high CTA indices.
                        const uint32_t wave = scheduler.current_iter++ * gridDim.x;
                        if (wave >= scheduler.num_blocks) break;
                        const uint32_t ticket = wave + gridDim.x - 1u - blockIdx.x;
                        if (ticket >= scheduler.num_blocks) break;
                        map_shared_ticket<cols>(ticket, scheduler.num_aligned_m_blocks, m_block, n_block);
                        __syncthreads();
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
                    if constexpr (kIsLinear1 && !IsShared) {
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
                    prepare_weight_addresses(tile, mainloop, scheduler, m_block, replicas);

                    Mma tiled_mma;
                    Tensor accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));
                    auto residue = Mainloop::compute_tile(tile, shape, coord, accum, tid, smem_buf);
                    auto output = epilogue_params;
                    if constexpr (kIsLinear1 && !IsShared) {
                        output.ptr_D += scheduler.curr_offset_c();
                        output.ptr_SFD += scheduler.curr_offset_c_scales();
                    }
                    Epilogue epilogue{output, shared.epilogue};
                    if constexpr (kVectorStore) {
                        static_assert(IsShared && !kIsLinear1 && BN == 256);
                        auto* dest = reinterpret_cast<__ppu_bfloat16*>(output.ptr_D)
                            + int64_t(m_block) * BM * N + n_block * BN;
                        auto fr = tiled_mma.get_slice(tid).partition_C(make_identity_tensor(make_shape(Int<BM>{},Int<BN>{})));
                        constexpr int width = 4;
                        static_assert(size<2>(cute::shape<0>(fr)) == 2);
                        if constexpr (size<2>(fr) % (width/2) == 0) {
                            const int lane = tid % width;
                            CUTLASS_PRAGMA_UNROLL
                            for (int wm=0; wm<size<1>(fr); ++wm)
                                CUTLASS_PRAGMA_UNROLL
                                for (int mm=0; mm<size<1>(cute::shape<0>(fr)); ++mm)
                                    CUTLASS_PRAGMA_UNROLL
                                    for (int wn=0; wn<size<2>(fr); wn+=width/2) {
                                        uint32_t words[width];
                                        CUTLASS_PRAGMA_UNROLL
                                        for (int j=0; j<width; ++j) {
                                            auto lo=make_coord(Int<0>{},mm,j%2);
                                            auto hi=make_coord(Int<1>{},mm,j%2);
                                            auto packed=__floats2bfloat162_rn(accum(lo,wm,wn+j/2),accum(hi,wm,wn+j/2));
                                            static_assert(sizeof(packed)==sizeof(words[j]));
                                            __builtin_memcpy(&words[j],&packed,sizeof(packed));
                                        }
                                        // Transpose four packed words across four lanes.
                                        CUTLASS_PRAGMA_UNROLL
                                        for (int bit=1; bit<width; bit*=2)
                                            CUTLASS_PRAGMA_UNROLL
                                            for (int j=0; j<width; ++j) {
                                                if (j & bit) continue;
                                                uint32_t send=(lane & bit) ? words[j] : words[j+bit];
                                                uint32_t recv=__shfl_xor_sync(0xffffffffu,send,bit);
                                                if (lane & bit) words[j]=recv;
                                                else words[j+bit]=recv;
                                            }
                                        auto rc=fr(make_coord(Int<0>{},mm,Int<0>{}),wm,wn);
                                        const unsigned row=get<0>(rc);
                                        const auto owned = fr(make_coord(Int<0>{}, mm, lane % 2), wm, wn + lane / 2);
                                        const unsigned vector_col = get<1>(owned) / 8;
                                        if (m_block*BM+row<m) {
                                            static_assert(uint64_t(BM)*N/8<UINT32_MAX && N%8==0);
                                            const uint32_t offset=row*(N/8u)+vector_col;
                                            reinterpret_cast<uint4*>(dest)[offset]=make_uint4(words[0],words[1],words[2],words[3]);
                                        }
                                    }
                        } else {
                            epilogue(shape,TileShape{},coord,accum,tiled_mma,residue,tid,epilogue_smem);
                        }
                    } else if constexpr (kIsLinear1 || IsShared) {
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
            if constexpr (!kIsLinear1 && !IsShared) {
                comm::scatter_release<Ranks>(*signals, *scatter_buffer,
                    gridDim.x, blockIdx.x, threadIdx.x, generation, BN == 256);
            }
        }
        template<class Dense>
        CUTLASS_DEVICE void run_shared_prefix(const typename Dense::Params& params,
                                               uint32_t rows, char* smem) {
            if (blockIdx.x < CopyBlocks) return;
            constexpr uint32_t columns = math::ceil_div(Dense::Params::kShapeN, BN);
            const uint32_t m_tiles = math::ceil_div(rows, uint32_t(BM));
            const auto shape = typename Dense::ProblemShape{rows, Dense::Params::kShapeN, K, 1};
            auto tile = params.mainloop;
            tile.problem_shape = {rows, Dense::Params::kShapeN, K};
            get<0>(tile.dA) = K;
            get<0>(tile.dB) = K;
            get<1>(tile.dSFA) = SourceCapacity;
            get<1>(tile.dSFB) = Dense::Params::kShapeN;
            auto& storage = *reinterpret_cast<typename Dense::SharedStorage*>(smem);
            const uint32_t first_ticket = blockIdx.x - CopyBlocks;
            const uint32_t total = m_tiles * columns;
            const uint32_t base = first_ticket * 2u;
            if (base >= total) return;
            const uint32_t step = 2u * (gridDim.x - CopyBlocks);
            const uint32_t last = (total - 1u - base) / step;
            const uint32_t length = 2u * last + min(2u, total - (base + last * step));
            #pragma clang loop licm(disable)
            for (uint32_t i = 0; i < length; ++i) {
                const uint32_t ticket = base + (i / 2u) * step + i % 2u;
                uint32_t m, n;
                map_shared_ticket<columns, math::ceil_div(uint32_t(SourceCapacity),uint32_t(BM))>(
                    ticket, m_tiles, m, n);
                auto coord = make_coord(m, n, _, 0);
                Mma mma;
                auto accum = partition_fragment_C(mma, take<0,2>(TileShape{}));
                auto residue = Dense::MainloopType::compute_tile(tile, shape, coord, accum, threadIdx.x, smem);
                typename Dense::EpilogueType epilogue{params.epilogue, storage.epilogue};
                epilogue(shape, TileShape{}, coord, accum, mma, residue, threadIdx.x,
                         reinterpret_cast<char*>(&storage.epilogue));
                __syncthreads();
            }
        }
    };

    static_assert(is_supported_megamoe_rank_count(Ranks), "unsupported MegaMoE rank count");
    static_assert(BM == 16 || BM == 32 || BM == 64 || BM == 128 || BM == 256,
                  "unsupported MegaMoE block M");
    static_assert(IsShared || !kIsLinear1 ||
                  (K > 0 && N > 0 && Groups > 0 && N % 64 == 0 &&
                   Kernel::MaxThreadsPerBlock >= 64 &&
                   CopyBlocks > 0 && KTilesPerFlag > 0 && SourceCapacity > 0 && TopK > 0 &&
                   math::ceil_div(K / BK, KTilesPerFlag) <= 7 &&
                   ActivationEpilogue::get_shared_storage_size(BM, BN) <= Kernel::SharedStorageSize),
                  "invalid fused dispatch resources");
};

template <typename Kernel, typename Shared = void>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock, Kernel::MinBlocksPerMultiprocessor)
void sm89_fp4_fp4_mega_moe_l1(typename Kernel::MainloopArguments mainloop, typename Kernel::EpilogueParams epilogue,
                              typename Kernel::SchedulerParams scheduler, layout::MegaMoESignals* signals,
                              MegaMoEReplicaWeights replicas
#if DG_MEGAMOE_HAS_SHARED
    , const typename Shared::Params shared_params, const int32_t* shared_control
#endif
) {
    extern __shared__ char smem[];
    Kernel op;
#if DG_MEGAMOE_HAS_SHARED
    const uint32_t rows = min(shared_params.scheduler.shape_m, static_cast<uint32_t>(*shared_control));
    op.template operator()<Shared>(mainloop, epilogue, scheduler, signals, smem, replicas,
        resolve_fused_round_generation(*signals), nullptr, shared_params, rows);
#else
    op(mainloop, epilogue, scheduler, signals, smem, replicas, resolve_fused_round_generation(*signals));
#endif
}

template <typename Kernel, int Ranks, int TopK, int Hidden, int Groups,
          int kColumnSplits, int kReduceGroupThreads, bool kOutputAligned, typename Shared = void>
__device__ __forceinline__ void mega_moe_reduce(
        const layout::SymBuffer<Ranks>& scatter_buffer, uint32_t token_capacity,
        const MegaMoEReduceParams& reduce, uint64_t writer_mask,
        const __ppu_bfloat16* shared_output = nullptr, uint32_t shared_rows = 0) {
    moe_reduce_dispatch<kOutputAligned, TopK, Hidden, Kernel::MaxThreadsPerBlock,
                        kColumnSplits, kReduceGroupThreads, !std::is_void_v<Shared>>(
        scatter_buffer.template get_base_ptr<const __ppu_bfloat16*>(),
        reduce.ids, reduce.weights, static_cast<__ppu_bfloat16*>(reduce.output),
        static_cast<int>(token_capacity), Groups, scatter_buffer.rank_idx, reduce.tokens, writer_mask, shared_output, shared_rows);
}

template <typename Kernel, int Ranks, int TopK, int Hidden, int Groups,
          int kColumnSplits, int kReduceGroupThreads, bool kOutputAligned, typename Shared = void>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock, Kernel::MinBlocksPerMultiprocessor)
void sm89_fp4_fp4_mega_moe_l2(typename Kernel::MainloopArguments mainloop, typename Kernel::EpilogueParams epilogue,
                              typename Kernel::SchedulerParams scheduler, layout::MegaMoESignals* signals,
                              layout::SymBuffer<Ranks> scatter_buffer, MegaMoEReduceParams reduce,
                              MegaMoEReplicaWeights replicas
#if DG_MEGAMOE_HAS_SHARED
    , const typename Shared::Params shared_params, const int32_t* shared_control
#endif
) {
    extern __shared__ char smem[];
    const uint32_t generation = __ldbl(signals->preprocess_ready) & ~kPreprocessMetadataReadyBit;
    Kernel op;
#if DG_MEGAMOE_HAS_SHARED
    const uint32_t shared_m = min(shared_params.scheduler.shape_m, static_cast<uint32_t>(*shared_control));
    constexpr bool vector_store = Kernel::kThreeSlots && Shared::kSharedRows == 256 &&
        Shared::kSharedCols == 256 && Hidden % 256 == 0 && uint64_t(Shared::kSharedRows) * Hidden / 8 < UINT32_MAX;
    if constexpr (Kernel::kCanUnifyStage2 &&
                  std::is_same_v<typename Kernel::MainloopType, typename Shared::MainloopType>) {
        op.template run_shared_routed<Shared>(mainloop, epilogue, scheduler, signals, shared_params,
            shared_m, smem, replicas, const_cast<int32_t*>(shared_control) + 1, scatter_buffer, generation);
    } else {
        auto dense_scheduler = shared_params.scheduler;
        dense_scheduler.shape_m = shared_m;
        Shared{}.template operator()<void, vector_store>(shared_params.mainloop, shared_params.epilogue,
            dense_scheduler, signals, smem, MegaMoEReplicaWeights{}, generation);
        __syncthreads();
        op(mainloop, epilogue, scheduler, signals, smem, replicas, generation, &scatter_buffer);
    }
#else
    op(mainloop, epilogue, scheduler, signals, smem, replicas, generation, &scatter_buffer);
#endif
    if constexpr (!DG_MEGAMOE_SPLIT_SYNC) {
        comm::rank_barrier<Ranks, true>(signals->scatter_arrivals, generation,
                                        reinterpret_cast<uint32_t*>(smem));
        mega_moe_reduce<Kernel, Ranks, TopK, Hidden, Groups, kColumnSplits, kReduceGroupThreads, kOutputAligned, Shared>(
            scatter_buffer, epilogue.token_capacity, reduce, *reinterpret_cast<const uint64_t*>(smem)
#if DG_MEGAMOE_HAS_SHARED
            , reinterpret_cast<const __ppu_bfloat16*>(shared_params.epilogue.ptr_D), shared_m
#endif
        );
    }
}

template <typename Kernel, int Ranks, int TopK, int Hidden, int Groups,
          int kColumnSplits, int kReduceGroupThreads, bool kOutputAligned, typename Shared = void>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock, Kernel::MinBlocksPerMultiprocessor)
void sm89_fp4_fp4_mega_moe_reduce(layout::SymBuffer<Ranks> scatter_buffer, uint32_t token_capacity,
                                  const layout::MegaMoESignals* signals, MegaMoEReduceParams reduce
#if DG_MEGAMOE_HAS_SHARED
    , const typename Shared::Params shared_params, const int32_t* shared_control
#endif
) {
    mega_moe_reduce<Kernel, Ranks, TopK, Hidden, Groups, kColumnSplits, kReduceGroupThreads, kOutputAligned, Shared>(
        scatter_buffer, token_capacity, reduce, signals->scatter_writer_mask
#if DG_MEGAMOE_HAS_SHARED
        , reinterpret_cast<const __ppu_bfloat16*>(shared_params.epilogue.ptr_D),
        min(shared_params.scheduler.shape_m, static_cast<uint32_t>(*shared_control))
#endif
    );
}

}  // namespace deep_gemm
