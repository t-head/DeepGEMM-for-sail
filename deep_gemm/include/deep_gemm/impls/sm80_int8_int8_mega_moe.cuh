#pragma once

#include <accutlass.hpp>
#include <cutlass/gemm/config/gemm_configs.hpp>

#include <deep_gemm/impls/int8_gemm_cutlass3.cuh>
#include <deep_gemm/epilogue/int8_gated_epilogue.cuh>
#include <deep_gemm/impls/int8_mega_moe_dispatch.cuh>
#include <deep_gemm/epilogue/int8_moe_reduce.cuh>

namespace deep_gemm {

// Build the INT8 MegaMoE MMA from the ordinary A8W8 collective.
// Scheduling and communication are supplied by the kernel, not by the MMA.
template <class Arch, class Element, class Accumulator, class TileShape,
          class WarpShape, int Stages, class Schedule, int CopyKTilesPerFlag = 0>
struct Int8MmaBuilder {
    static constexpr int M = cute::size<0>(TileShape{});
    static constexpr int N = cute::size<1>(TileShape{});
    static constexpr int K = cute::size<2>(TileShape{});
    static constexpr int WarpsM = M / cute::size<0>(WarpShape{});
    static constexpr int WarpsN = N / cute::size<1>(WarpShape{});
    static constexpr bool Overlap = cute::is_same_v<Schedule, ::KernelAiuMultistageOverlapPrologue>;
    static_assert(!Overlap || CopyKTilesPerFlag == 0);
    using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<Arch, Element, Element, Accumulator>::type;
    using TiledMma = cute::TiledMMA<cute::MMA_Atom<MmaInst>,
        cute::Layout<cute::Shape<cute::Int<WarpsM>, cute::Int<WarpsN>, cute::_1>>,
        cute::Tile<cute::Int<WarpsM * 16>, cute::Int<WarpsN * 16>, cute::_32>>;
    using Policy = cute::conditional_t<Overlap,
        cutlass::gemm::MainloopPPUAiuA8W8OverlapPrologue<Stages, Schedule>,
        cutlass::gemm::MainloopPPUAiuA8W8<Stages, Schedule, CopyKTilesPerFlag>>;
    using OperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<Arch, Element, false,
        cute::Int<M>, cute::Int<K>, false, (Overlap ? M + N : M) * K>;
    using OperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<Arch, Element, false,
        cute::Int<N>, cute::Int<K>, true, (Overlap ? M + N : N) * K>;
    using Collective = cutlass::gemm::collective::CollectiveMma<Arch, Policy, TileShape,
        Element, cutlass::detail::TagToStrideA_t<cutlass::layout::RowMajor>,
        Element, cutlass::detail::TagToStrideB_t<cutlass::layout::ColumnMajor>, TiledMma,
        typename OperandA::GmemTiledCopy, typename OperandA::SmemLayoutAtom,
        typename OperandA::SmemCopyAtom, cute::identity,
        typename OperandB::GmemTiledCopy, typename OperandB::SmemLayoutAtom,
        typename OperandB::SmemCopyAtom, cute::identity>;
};

// The 8-bit entry reuses A8W8's mainloop and epilogue. FP4's kernel and scale
// permutation are not dependencies of this specialization.
template <class Arch, int N, int K, int BlockM, int BlockN, int BlockK,
          int WarpM, int WarpN, int Stages, int Groups, int Ranks,
          int Capacity, int TopK, int CopyBlocks, int KTilesPerFlag, class ElementOutput = float,
          bool FuseActivation = false, bool NormalQuantization = false, bool ApplyClamp = false,
          uint32_t ActivationLimitBits = 0xffffffffu,
          GatedActivation Activation = GatedActivation::SwiGLU, uint32_t BetaBits = 0,
          uint32_t LinearBetaBits = 0, uint32_t AlphaBits = 0>
struct MegaMoEInt8Linear1 {
    using Tile = cute::Shape<cute::Int<BlockM>, cute::Int<BlockN>, cute::Int<BlockK>>;
    using Warp = cute::Shape<cute::Int<WarpM>, cute::Int<WarpN>, cute::Int<BlockK>>;
    using Mainloop = typename Int8MmaBuilder<Arch, int8_t, int32_t, Tile, Warp,
        Stages, cutlass::gemm::KernelAiuMultistage, KTilesPerFlag>::Collective;
    using Scheduler = sched::SM80MegaMoEScheduler<N, K, BlockM, BlockN, Groups, MegaMoEPhaseKind::Linear1,
        Ranks, CopyBlocks, KTilesPerFlag, Capacity, TopK, sm80::Int8FusedDispatch<BlockK>>;
    using RowStride = cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>;
    // Match the native A8W8 path: scales are applied in the mainloop and the
    // output conversion needs no additional shared-memory staging.
    using DefaultEpilogue = cutlass::epilogue::collective::DefaultEpilogueNoTsm<
        RowStride, RowStride,
        cutlass::epilogue::thread::LinearCombination<ElementOutput, 2, float, float,
            cutlass::epilogue::thread::ScaleType::Nothing>,
        cutlass::gemm::EpilogueDefault, N % BlockN == 0>;
    // At 128 accumulator values per thread the fused gate/up boundary may
    // exceed the register budget. Smaller fragments retain the float path.
    static constexpr bool HighFragmentPressure = WarpM * WarpN / 32 >= 128;
    static constexpr int ComputeThreads = (BlockM / WarpM) * (BlockN / WarpN) * 32;
    static constexpr auto GatedFragment = !std::is_same_v<Arch, cutlass::arch::PPU0010> || !HighFragmentPressure
        ? epilogue::Int8GatedFragment::GateUpFloat
        : !NormalQuantization ? epilogue::Int8GatedFragment::GateUpPackedBf16
        : ComputeThreads >= 256 ? epilogue::Int8GatedFragment::ActivatedFloat
                                : epilogue::Int8GatedFragment::GateUpFloat;
    // Keep SM89 on the existing runtime-clamp path. Constant specialization
    // is needed only by the SM80 fused epilogue register policy.
    static constexpr uint32_t EffectiveLimitBits = std::is_same_v<Arch, cutlass::arch::PPU0010>
        ? ActivationLimitBits : 0xffffffffu;
    using Epilogue = cute::conditional_t<FuseActivation,
        epilogue::Int8GatedEpilogue<RowStride, RowStride,
            cutlass::epilogue::thread::LinearCombination<ElementOutput, 2, float, float,
                cutlass::epilogue::thread::ScaleType::Nothing>, NormalQuantization, ApplyClamp,
                std::is_same_v<Arch, cutlass::arch::PPU0010> ? 4 : 1, EffectiveLimitBits, GatedFragment,
                Activation, BetaBits, LinearBetaBits, AlphaBits>,
        DefaultEpilogue>;
    using Kernel = cutlass::gemm::kernel::DeepGemmUniversal<cute::Shape<int, int, int, int>,
        Mainloop, Epilogue, Scheduler>;
    static constexpr layout::Int8MegaMoEBuffer buffers{Ranks, Groups * Ranks, Capacity, TopK, K, N / 2};

    static constexpr size_t shared_bytes = Kernel::SharedStorageSize;

};

// The shared scatter epilogue consumes scaled FP32 accumulators, independently
// of the MMA input type. Keep INT8 computation in the A8W8 implementation.
template <class Kernel, int Hidden, int TopK, int Groups, int Ranks>
__device__ __forceinline__ void mega_moe_int8_scatter_impl(
        typename Kernel::Params const& params, char* smem, const int32_t* ids, const float* weights,
        __ppu_bfloat16* output, int tokens) {
    const int32_t phase = tix::ld_acq_sys(
        &params.scatter_barrier->phase_counter) & 3;
    Kernel{}(params, smem);

    // EP8 releases peers without a second grid barrier. Owner mode also needs
    // the completed writer mask before choosing the shared reducer's load path.
    if constexpr (Ranks == 8 || DG_MEGAMOE_OWNER_MODE) {
        if (threadIdx.x == 0) {
            const auto writers = comm::fp4_scatter_acquire<Ranks>(*params.scatter_barrier, phase);
#if DG_MEGAMOE_OWNER_MODE
            *reinterpret_cast<uint64_t*>(smem) = writers;
#endif
        }
        __syncthreads();
    }
    uint64_t writers = UINT64_MAX;
#if DG_MEGAMOE_OWNER_MODE
    writers = *reinterpret_cast<const uint64_t*>(smem);
#endif
    for (int token = blockIdx.x; token < tokens; token += gridDim.x) {
        sm80::moe_reduce_impl<TopK, Hidden, true, Kernel::MaxThreadsPerBlock>(
            reinterpret_cast<const __ppu_bfloat16*>(params.scatter_buffer.base),
            ids, weights, output, params.epilogue.token_capacity, Groups,
            params.scatter_buffer.rank_idx, token, 0, writers);
    }
}

template <class Kernel, int Hidden, int TopK, int Groups, int Ranks>
__launch_bounds__(Kernel::MaxThreadsPerBlock, Kernel::MinBlocksPerMultiprocessor)
__global__ void mega_moe_int8_scatter(
        typename Kernel::Params params, const int32_t* ids, const float* weights,
        __ppu_bfloat16* output, int tokens) {
    extern __shared__ char smem[];
    mega_moe_int8_scatter_impl<Kernel, Hidden, TopK, Groups, Ranks>(
        params, smem, ids, weights, output, tokens);
}

template <class Arch, int Hidden, int Intermediate, int BlockM, int BlockN, int BlockK,
          int WarpM, int WarpN, int Stages, int Groups, int Ranks, int Capacity, int TopK>
struct MegaMoEInt8Linear2 {
    static_assert(BlockN == 256 || DG_MEGAMOE_OWNER_MODE,
                  "BN128 scatter requires matching per-owner reduce metadata");
    using Tile = cute::Shape<cute::Int<BlockM>, cute::Int<BlockN>, cute::Int<BlockK>>;
    using Warp = cute::Shape<cute::Int<WarpM>, cute::Int<WarpN>, cute::Int<BlockK>>;
    using Mainloop = typename Int8MmaBuilder<Arch, int8_t, int32_t, Tile, Warp,
        Stages, cutlass::gemm::KernelAiuMultistage>::Collective;
    using Scheduler = sched::SM80MegaMoEScheduler<Hidden, Intermediate, BlockM, BlockN, Groups,
        MegaMoEPhaseKind::Linear2, Ranks>;
    using RowStride = cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>;
    using Epilogue = epilogue::Fp4ScatterEpilogue<RowStride, RowStride,
        cute::Layout<cute::Shape<cute::Int<(BlockM / WarpM) * 16>, cute::Int<BlockN>>,
                     cute::Stride<cute::Int<BlockN>, cute::_1>>,
        cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, float>,
        float, cutlass::bfloat16_t,
        cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 16, float, float>,
        Ranks, (int64_t(TopK) * Capacity * Hidden > UINT32_MAX)>;
    using Kernel = cutlass::gemm::kernel::DeepGemmUniversal<cute::Shape<int, int, int, int>,
        Mainloop, Epilogue, Scheduler>;
    static constexpr layout::Int8MegaMoEBuffer buffers{
        Ranks, Groups * Ranks, Capacity, TopK, Hidden, Intermediate};
    static constexpr size_t shared_bytes = Kernel::SharedStorageSize;

};

}  // namespace deep_gemm
