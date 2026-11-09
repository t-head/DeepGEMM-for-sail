#pragma once

#include <accutlass.hpp>
#include <cutlass/gemm/config/gemm_configs.hpp>

#include <deep_gemm/impls/int8_gemm_cutlass3.cuh>
#include <deep_gemm/epilogue/int8_gated_epilogue.cuh>
#include <deep_gemm/impls/int8_mega_moe_dispatch.cuh>
#include <deep_gemm/epilogue/mega_moe_reduce.cuh>

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
template <class Arch, int N, int K, int BM, int BN, int BK,
          int WM, int WN, int Stages, int Groups, int Ranks,
          int SourceCapacity, int TopK, int CopyBlocks, int KTilesPerFlag, class ElementD = float,
          bool FuseActivation = false, bool NormalQuantization = false, bool ApplyClamp = false,
          uint32_t ActivationLimitBits = 0xffffffffu,
          GatedActivation Activation = GatedActivation::SwiGLU, uint32_t BetaBits = 0,
          uint32_t LinearBetaBits = 0, uint32_t AlphaBits = 0, bool FastMath = true>
struct MegaMoEInt8Linear1 {
    using TileShape = cute::Shape<cute::Int<BM>, cute::Int<BN>, cute::Int<BK>>;
    using WarpShape = cute::Shape<cute::Int<WM>, cute::Int<WN>, cute::Int<BK>>;
    using Mainloop = typename Int8MmaBuilder<Arch, int8_t, int32_t, TileShape, WarpShape,
        Stages, cutlass::gemm::KernelAiuMultistage, KTilesPerFlag>::Collective;
    using Scheduler = sched::SM80MegaMoEScheduler<N, K, BM, BN, Groups, MegaMoEPhaseKind::Linear1,
        Ranks, CopyBlocks, KTilesPerFlag, SourceCapacity, TopK, sm80::Int8FusedDispatch<BK>>;
    using StrideCD = cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>;
    // Match the native A8W8 path: scales are applied in the mainloop and the
    // output conversion needs no additional shared-memory staging.
    using DefaultEpilogue = cutlass::epilogue::collective::DefaultEpilogueNoTsm<
        StrideCD, StrideCD,
        cutlass::epilogue::thread::LinearCombination<ElementD, 2, float, float,
            cutlass::epilogue::thread::ScaleType::Nothing>,
        cutlass::gemm::EpilogueDefault, N % BN == 0>;
    // At 128 accumulator values per thread the fused gate/up boundary may
    // exceed the register budget. Smaller fragments retain the float path.
    static constexpr bool kHighFragmentPressure = WM * WN / 32 >= 128;
    static constexpr int kComputeThreads = (BM / WM) * (BN / WN) * 32;
    static constexpr auto kGatedFragment = !std::is_same_v<Arch, cutlass::arch::PPU0010> || !kHighFragmentPressure
        ? epilogue::Int8GatedFragment::GateUpFloat
        : !NormalQuantization ? epilogue::Int8GatedFragment::GateUpPackedBf16
        : kComputeThreads >= 256 ? epilogue::Int8GatedFragment::ActivatedFloat
                                : epilogue::Int8GatedFragment::GateUpFloat;
    // Keep SM89 on the existing runtime-clamp path. Constant specialization
    // is needed only by the SM80 fused epilogue register policy.
    static constexpr uint32_t kEffectiveLimitBits = std::is_same_v<Arch, cutlass::arch::PPU0010>
        ? ActivationLimitBits : 0xffffffffu;
    using Epilogue = cute::conditional_t<FuseActivation,
        epilogue::Int8GatedEpilogue<StrideCD, StrideCD,
            cutlass::epilogue::thread::LinearCombination<ElementD, 2, float, float,
                cutlass::epilogue::thread::ScaleType::Nothing>, NormalQuantization, ApplyClamp,
                std::is_same_v<Arch, cutlass::arch::PPU0010> ? 4 : 1, kEffectiveLimitBits, kGatedFragment,
                Activation, BetaBits, LinearBetaBits, AlphaBits, FastMath>,
        DefaultEpilogue>;
    using Kernel = cutlass::gemm::kernel::DeepGemmUniversal<cute::Shape<int, int, int, int>,
        Mainloop, Epilogue, Scheduler>;
    static constexpr layout::Int8MegaMoEBuffer buffers{Ranks, Groups * Ranks, SourceCapacity, TopK, K, N / 2};

    static constexpr size_t shared_bytes = Kernel::SharedStorageSize;

};

// The shared scatter epilogue consumes scaled FP32 accumulators, independently
// of the MMA input type. Keep INT8 computation in the A8W8 implementation.
template <class Kernel, int Hidden, int TopK, int Groups, int Ranks, bool kOutputAligned>
__device__ __forceinline__ void mega_moe_int8_reduce_impl(
        typename Kernel::Params const& params, const int64_t* ids, const float* weights,
        __ppu_bfloat16* output, int tokens, uint64_t writer_mask) {
    for (int token = blockIdx.x; token < tokens; token += gridDim.x) {
        moe_reduce_impl<kOutputAligned, TopK, Hidden, true, Kernel::MaxThreadsPerBlock>(
            reinterpret_cast<const __ppu_bfloat16*>(params.scatter_buffer.base),
            ids, weights, output, params.epilogue.token_capacity, Groups,
            params.scatter_buffer.rank_idx, token, 0, writer_mask, threadIdx.x);
    }
}

template <class Kernel, int Hidden, int TopK, int Groups, int Ranks, bool kOutputAligned>
__device__ __forceinline__ void mega_moe_int8_scatter_impl(
        typename Kernel::Params const& params, char* smem, const int64_t* ids, const float* weights,
        __ppu_bfloat16* output, int tokens) {
    const uint32_t generation = __ldbl(params.signals->preprocess_ready) & ~kPreprocessMetadataReadyBit;
    Kernel{}(params, smem);
    if constexpr (!DG_MEGAMOE_SPLIT_SYNC) {
        comm::rank_barrier<Ranks, true>(params.signals->scatter_arrivals, generation,
                                        reinterpret_cast<uint32_t*>(smem));
        mega_moe_int8_reduce_impl<Kernel, Hidden, TopK, Groups, Ranks, kOutputAligned>(
            params, ids, weights, output, tokens, *reinterpret_cast<const uint64_t*>(smem));
    }
}

template <class Arch, int N, int K, int BM, int BN, int BK,
          int WM, int WN, int Stages, int Groups, int Ranks, int SourceCapacity, int TopK,
          bool WaitScatterPair = false>
struct MegaMoEInt8Linear2 {
    static_assert(BN == 256 || DG_MEGAMOE_OWNER_MODE,
                  "BN128 scatter requires matching per-owner reduce metadata");
    using TileShape = cute::Shape<cute::Int<BM>, cute::Int<BN>, cute::Int<BK>>;
    using WarpShape = cute::Shape<cute::Int<WM>, cute::Int<WN>, cute::Int<BK>>;
    using Mainloop = typename Int8MmaBuilder<Arch, int8_t, int32_t, TileShape, WarpShape,
        Stages, cutlass::gemm::KernelAiuMultistage>::Collective;
    using Scheduler = sched::SM80MegaMoEScheduler<N, K, BM, BN, Groups,
        MegaMoEPhaseKind::Linear2, Ranks>;
    using StrideCD = cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>;
    using Epilogue = epilogue::Fp4ScatterEpilogue<StrideCD, StrideCD,
        cute::Layout<cute::Shape<cute::Int<(BM / WM) * 16>, cute::Int<BN>>,
                     cute::Stride<cute::Int<BN>, cute::_1>>,
        cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, float>,
        float, cutlass::bfloat16_t,
        cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 16, float, float>,
        Ranks, (int64_t(TopK) * SourceCapacity * N > UINT32_MAX), false, WaitScatterPair>;
    using Kernel = cutlass::gemm::kernel::DeepGemmUniversal<cute::Shape<int, int, int, int>,
        Mainloop, Epilogue, Scheduler>;
    static constexpr layout::Int8MegaMoEBuffer buffers{
        Ranks, Groups * Ranks, SourceCapacity, TopK, N, K};
    static constexpr size_t shared_bytes = Kernel::SharedStorageSize;

};

}  // namespace deep_gemm
