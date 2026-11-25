#pragma once

#include <accutlass.hpp>
#include <cutlass/gemm/config/gemm_configs.hpp>

#include <deep_gemm/impls/int8_gemm_cutlass3.cuh>
#include <deep_gemm/epilogue/int8_gated_epilogue.cuh>
#include <deep_gemm/impls/int8_mega_moe_dispatch.cuh>
#include <deep_gemm/epilogue/mega_moe_l2_epilogue.cuh>
#include <deep_gemm/scheduler/sm80_mega_moe.cuh>

namespace deep_gemm {

template <int N, int K, int BM, int BN, int BK, int WM, int WN,
          int Groups, int Stages, int Ranks, MegaMoEPhaseKind Stage,
          int SourceCapacity, int TopK, int KTilesPerFlag = 0,
          bool WaitScatterPair = false, bool ApplyClamp = false,
          uint32_t ActivationLimitBits = 0xffffffffu,
          GatedActivation Activation = GatedActivation::SwiGLU, uint32_t BetaBits = 0,
          uint32_t LinearBetaBits = 0, uint32_t AlphaBits = 0, bool FastMath = true>
struct MegaMoEInt8Kernel {
    static_assert(Stage == MegaMoEPhaseKind::Linear1 || Stage == MegaMoEPhaseKind::Linear2,
                  "unsupported MegaMoE stage");
    static constexpr bool kIsLinear1 = Stage == MegaMoEPhaseKind::Linear1;
    static_assert(kIsLinear1 || BN == 256 || DG_MEGAMOE_OWNER_MODE,
                  "BN128 scatter requires matching per-owner reduce metadata");
    using WarpOnM = cute::Int<BM / WM>;
    using WarpOnN = cute::Int<BN / WN>;
    using Arch = cutlass::arch::PPU0010;
    using Element = int8_t;
    using StrideCD = cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>;
    using OperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<Arch, Element, false,
        cute::Int<BM>, cute::Int<BK>, false, BM * BK>;
    using OperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<Arch, Element, false,
        cute::Int<BN>, cute::Int<BK>, true, BN * BK>;
    using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<Arch, Element, Element, int32_t>::type;
    using Mma = cute::TiledMMA<cute::MMA_Atom<MmaInst>,
        cute::Layout<cute::Shape<WarpOnM, WarpOnN, cute::_1>>,
        cute::Tile<cute::Int<(BM / WM) * 16>, cute::Int<(BN / WN) * 16>, cute::_32>>;
    using Mainloop = cutlass::gemm::collective::CollectiveMma<Arch,
        cutlass::gemm::MainloopPPUAiuA8W8<Stages, cutlass::gemm::KernelAiuMultistage,
                                       kIsLinear1 ? KTilesPerFlag : 0>,
        cute::Shape<cute::Int<BM>, cute::Int<BN>, cute::Int<BK>>,
        Element, cutlass::detail::TagToStrideA_t<cutlass::layout::RowMajor>,
        Element, cutlass::detail::TagToStrideB_t<cutlass::layout::ColumnMajor>, Mma,
        typename OperandA::GmemTiledCopy, typename OperandA::SmemLayoutAtom,
        typename OperandA::SmemCopyAtom, cute::identity,
        typename OperandB::GmemTiledCopy, typename OperandB::SmemLayoutAtom,
        typename OperandB::SmemCopyAtom, cute::identity>;
    using ActivationEpilogue = epilogue::Int8GatedEpilogue<StrideCD, StrideCD,
        cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 2, float, float,
            cutlass::epilogue::thread::ScaleType::Nothing>, ApplyClamp, 4, ActivationLimitBits,
            Activation, BetaBits, LinearBetaBits, AlphaBits, FastMath>;
    using ScatterEpilogue = epilogue::MegaMoEScatterEpilogue<StrideCD, StrideCD,
        cute::Layout<cute::Shape<cute::Int<(BM / WM) * 16>, cute::Int<BN>>,
                     cute::Stride<cute::Int<BN>, cute::_1>>,
        cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, float>,
        float, cutlass::bfloat16_t,
        cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 16, float, float>,
        Ranks, (int64_t(TopK) * SourceCapacity * N > UINT32_MAX), false, WaitScatterPair>;
    using Epilogue = std::conditional_t<kIsLinear1, ActivationEpilogue, ScatterEpilogue>;
    using Scheduler = sched::SM80MegaMoEScheduler<N, K, BM, BN, Groups, Stage,
        Ranks, kIsLinear1 ? KTilesPerFlag : 0, SourceCapacity, TopK, sm80::Int8FusedDispatch<BK>>;
    using Kernel = cutlass::gemm::kernel::DeepGemmUniversal<cute::Shape<int, int, int, int>,
        Mainloop, Epilogue, Scheduler>;
};

// The shared scatter epilogue consumes scaled FP32 accumulators, independently
// of the MMA input type. Keep INT8 computation in the A8W8 implementation.
template <class Kernel, int Hidden, int TopK, int Groups, int Ranks, bool kOutputAligned,
          int kColumnSplits = 1, int kReduceGroupThreads = 0>
__device__ __forceinline__ void mega_moe_int8_reduce_impl(
        typename Kernel::Params const& params, const int64_t* ids, const float* weights,
        __ppu_bfloat16* output, int tokens, uint64_t writer_mask) {
    moe_reduce_dispatch<kOutputAligned, TopK, Hidden, Kernel::MaxThreadsPerBlock,
                        kColumnSplits, kReduceGroupThreads>(
        reinterpret_cast<const __ppu_bfloat16*>(params.scatter_buffer.base),
        ids, weights, output, params.epilogue.token_capacity, Groups,
        params.scatter_buffer.rank_idx, tokens, writer_mask);
}

template <class Kernel, int Hidden, int TopK, int Groups, int Ranks, bool kOutputAligned,
          int kColumnSplits = 1, int kReduceGroupThreads = 0>
__device__ __forceinline__ void mega_moe_int8_scatter_impl(
        typename Kernel::Params const& params, char* smem, const int64_t* ids, const float* weights,
        __ppu_bfloat16* output, int tokens) {
    const uint32_t generation = __ldbl(params.signals->preprocess_ready) & ~kPreprocessMetadataReadyBit;
    Kernel{}(params, smem);
    if constexpr (!DG_MEGAMOE_SPLIT_SYNC) {
        comm::rank_barrier<Ranks, true>(params.signals->scatter_arrivals, generation,
                                        reinterpret_cast<uint32_t*>(smem));
        mega_moe_int8_reduce_impl<Kernel, Hidden, TopK, Groups, Ranks, kOutputAligned,
                                 kColumnSplits, kReduceGroupThreads>(
            params, ids, weights, output, tokens, *reinterpret_cast<const uint64_t*>(smem));
    }
}

}  // namespace deep_gemm
