#pragma once

#include <deep_gemm/impls/fp8_channelwise_mega_moe_dispatch_gemm1.cuh>
#include <deep_gemm/impls/fp8_gemm_channelwise_scatter.cuh>

namespace deep_gemm {

template <int NUM_THREADS_, uint64_t LAYOUT_OFF_IT_, uint64_t LAYOUT_OFF_ISF_, uint64_t LAYOUT_OFF_ITI_,
          uint64_t LAYOUT_OFF_PT_, uint64_t LAYOUT_OFF_PSF_, uint64_t LAYOUT_OFF_GEMM2_META_, GatedActivation ACTIVATION_,
          int PULL_GROUP_SIZE_, int GEMM2_BLOCK_M_, bool GEMM2_VALID_ROWS_, int BLOCK_M_, int GEMM1_BLOCK_K_,
          int GEMM1_BLOCK_N_, int GEMM1_N_, int GEMM1_STAGES_, int GEMM1_WARP_M_, int GEMM1_WARP_N_, int NMTPR_,
          int NUM_CTAS_, int NUM_EXPERTS_, int NUM_EXPERTS_PER_RANK_, int NUM_RANKS_, int NUM_TOPK_,
          int PULL_CTAS_, int TOKEN_BYTES_, uint64_t StagingCounts_, uint64_t StagingTokens_>
struct MegaMoEFp8ChannelwiseGemm1Kernel {
    static constexpr layout::RoutedMoEStaging staging{StagingCounts_, StagingTokens_};
    static constexpr int NUM_THREADS = NUM_THREADS_;
    static constexpr uint64_t LAYOUT_OFF_IT = LAYOUT_OFF_IT_, LAYOUT_OFF_ISF = LAYOUT_OFF_ISF_;
    static constexpr uint64_t LAYOUT_OFF_ITI = LAYOUT_OFF_ITI_, LAYOUT_OFF_PT = LAYOUT_OFF_PT_;
    static constexpr uint64_t LAYOUT_OFF_PSF = LAYOUT_OFF_PSF_, LAYOUT_OFF_GEMM2_META = LAYOUT_OFF_GEMM2_META_;
    static constexpr GatedActivation ACTIVATION = ACTIVATION_;
    static constexpr int PULL_GROUP_SIZE = PULL_GROUP_SIZE_, GEMM2_BLOCK_M = GEMM2_BLOCK_M_;
    static constexpr bool GEMM2_VALID_ROWS = GEMM2_VALID_ROWS_;
    static constexpr int BLOCK_M = BLOCK_M_, GEMM1_BLOCK_K = GEMM1_BLOCK_K_, GEMM1_BLOCK_N = GEMM1_BLOCK_N_;
    static constexpr int GEMM1_N = GEMM1_N_, GEMM1_STAGES = GEMM1_STAGES_, GEMM1_WARP_M = GEMM1_WARP_M_;
    static constexpr int GEMM1_WARP_N = GEMM1_WARP_N_, NMTPR = NMTPR_, NUM_CTAS = NUM_CTAS_;
    static constexpr int NUM_EXPERTS = NUM_EXPERTS_, NUM_EXPERTS_PER_RANK = NUM_EXPERTS_PER_RANK_;
    static constexpr int NUM_RANKS = NUM_RANKS_, NUM_TOPK = NUM_TOPK_, PULL_CTAS = PULL_CTAS_;
    static constexpr int TOKEN_BYTES = TOKEN_BYTES_;

    static_assert(sizeof(layout::SymBuffer<>) == 528, "MegaMoE symmetric buffer ABI changed");
    static constexpr uint32_t NUM_WARPS = NUM_THREADS / 32;
    static constexpr uint32_t COUNT_BYTES = NUM_EXPERTS * sizeof(uint32_t);
    static constexpr uint32_t TOKEN_SCRATCH_OFFSET = (COUNT_BYTES + 15u) & ~15u;
    static constexpr uint32_t shared_memory_bytes() {
        using Cfg = mega_moe::fp8_channelwise::Gemm1MmaConfig<BLOCK_M, GEMM1_BLOCK_N, GEMM1_BLOCK_K, GEMM1_WARP_M,
                                                              GEMM1_WARP_N, GEMM1_STAGES>;
        uint32_t gemm = sizeof(typename Cfg::CollectiveMainloop::SharedStorage);
        uint32_t dispatch = ((NUM_EXPERTS * 4 + 15) & ~15u) + (NUM_THREADS / 32) * TOKEN_BYTES;
        return gemm > dispatch ? gemm : dispatch;
    }
    static constexpr int kNumThreads = NUM_THREADS;
    static constexpr int kMinBlocks = 1;
};

template <typename Builder>
__global__ __launch_bounds__(Builder::kNumThreads, Builder::kMinBlocks) void sm89_fp8_channelwise_mega_moe_l1(
    layout::SymBuffer<> sb, int rank_idx, int num_tokens, __hg_fp8_e4m3* gemm1_weight, float* gemm1_weight_scale,
    __hg_fp8_e4m3* gemm1_weight_replicas, float* gemm1_weight_scale_replicas, int num_master_experts,
    __ppu_bfloat16* swiglu_output, float activation_clamp, float activation_alpha, float situ_beta, float situ_linear_beta, uint4* gemm2_block_info) {
    using namespace deep_gemm;
    constexpr uint64_t off_ws = 0, off_it = Builder::LAYOUT_OFF_IT, off_isf = Builder::LAYOUT_OFF_ISF;
    constexpr uint64_t off_iti = Builder::LAYOUT_OFF_ITI;
    constexpr uint64_t off_pt = Builder::LAYOUT_OFF_PT, off_psf = Builder::LAYOUT_OFF_PSF,
        off_meta = Builder::LAYOUT_OFF_GEMM2_META;

    const int64_t local_base = sb.base;
    auto P = [&](uint64_t off) -> void* { return (void*)(local_base + (int64_t)off); };

    deep_gemm::mega_moe::fp8_channelwise::dispatch_gemm1_device<
        Builder::NUM_RANKS, Builder::NUM_CTAS, Builder::NUM_THREADS, Builder::NUM_EXPERTS, Builder::NUM_EXPERTS_PER_RANK,
            Builder::NUM_TOPK, Builder::BLOCK_M, Builder::TOKEN_BYTES, Builder::GEMM1_N,
        Builder::GEMM1_BLOCK_N, Builder::GEMM1_BLOCK_K, Builder::GEMM1_WARP_M, Builder::GEMM1_WARP_N, Builder::GEMM1_STAGES,
            Builder::PULL_CTAS, Builder::ACTIVATION, void,
        Builder::PULL_GROUP_SIZE, Builder::GEMM2_BLOCK_M, Builder::GEMM2_VALID_ROWS>(
        P(off_ws), sb, (int64_t*)P(off_iti), (const uint8_t*)P(off_it), (const float*)P(off_isf),
        (uint8_t*)P(off_pt), (float*)P(off_psf), (int32_t*)P(off_meta), gemm2_block_info,
        (const cutlass::float_e4m3_t*)gemm1_weight, gemm1_weight_scale,
        (const cutlass::float_e4m3_t*)gemm1_weight_replicas, gemm1_weight_scale_replicas, (uint32_t)num_master_experts,
        swiglu_output, activation_clamp, activation_alpha, (uint32_t)num_tokens, Builder::NMTPR, 1.0f, 1.0f, situ_beta, situ_linear_beta, Builder::staging);
}

template <bool USE_PADDED_VALID_ROWS, bool CHECK_PACKED_ROUTES, int N, int K, int BLOCK_M, int BLOCK_N,
          int BLOCK_K, int WARP_M, int WARP_N, int NUM_GROUPS, int NUM_STAGES, int NUM_RANKS, int SMEM_SIZE>
struct MegaMoEFp8ChannelwiseGemm2Kernel {
    using Scatter = mega_moe::detail::Bf16MoeScatter<USE_PADDED_VALID_ROWS, CHECK_PACKED_ROUTES>;
    using Family =
        ChannelwiseFp8GemmWithScatter<cutlass::float_e4m3_t, float, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                                      WARP_M, WARP_N, NUM_GROUPS, NUM_STAGES, GemmType::MGroupedNoPad,
                                      false, KernelStrategy::Default, Scatter, NUM_RANKS>;
    using Kernel = typename Family::GemmKernel;
    static_assert(Kernel::SharedStorageSize == SMEM_SIZE,
                  "channelwise scatter shared-memory estimate must match its arena");
    static constexpr uint32_t shared_memory_bytes() { return Kernel::SharedStorageSize; }
    static constexpr int kNumThreads = Kernel::MaxThreadsPerBlock;
    static constexpr int kMinBlocks = Kernel::MinBlocksPerMultiprocessor;
};

template <typename Builder>
__global__
__launch_bounds__(Builder::kNumThreads,
                  Builder::kMinBlocks) void sm89_fp8_fp8_mega_moe_l2(typename Builder::Kernel::Params params) {
    using Kernel = typename Builder::Kernel;
    extern __shared__ char smem[];
    Kernel op;
    op(params, smem);
}

} // namespace deep_gemm
