#pragma once

// Keep the blockwise epilogue parameter layout independent of the channelwise backend.
#include <deep_gemm/impls/fp8_gemm_blockwise_scatter.cuh>
#include <deep_gemm/impls/fp8_blockwise_mega_moe_dispatch_gemm1.cuh>

namespace deep_gemm {

template <int GEMM1_N_, int GEMM1_COMPUTE_M_, int NUM_THREADS_, uint64_t LAYOUT_OFF_IT_, uint64_t LAYOUT_OFF_ISF_,
          uint64_t LAYOUT_OFF_ITI_, uint64_t LAYOUT_OFF_PT_, uint64_t LAYOUT_OFF_PSF_,
          uint64_t LAYOUT_OFF_GEMM2_META_, int PULL_CTAS_, int CLAMP_, int MASTER_EXPERTS_, GatedActivation ACTIVATION_,
          int GEMM2_BLOCK_M_, int NMTPR_, int NUM_CTAS_, int NUM_EXPERTS_, int NUM_RANKS_, int NUM_TOPK_,
          int TOKEN_BYTES_>
struct MegaMoEFp8BlockwiseGemm1Kernel {
    static constexpr int GEMM1_N = GEMM1_N_, GEMM1_COMPUTE_M = GEMM1_COMPUTE_M_, NUM_THREADS = NUM_THREADS_;
    static constexpr uint64_t LAYOUT_OFF_IT = LAYOUT_OFF_IT_, LAYOUT_OFF_ISF = LAYOUT_OFF_ISF_;
    static constexpr uint64_t LAYOUT_OFF_ITI = LAYOUT_OFF_ITI_, LAYOUT_OFF_PT = LAYOUT_OFF_PT_;
    static constexpr uint64_t LAYOUT_OFF_PSF = LAYOUT_OFF_PSF_, LAYOUT_OFF_GEMM2_META = LAYOUT_OFF_GEMM2_META_;
    static constexpr int PULL_CTAS = PULL_CTAS_, CLAMP = CLAMP_, MASTER_EXPERTS = MASTER_EXPERTS_;
    static constexpr GatedActivation ACTIVATION = ACTIVATION_;
    static constexpr int GEMM2_BLOCK_M = GEMM2_BLOCK_M_, NMTPR = NMTPR_, NUM_CTAS = NUM_CTAS_;
    static constexpr int NUM_EXPERTS = NUM_EXPERTS_, NUM_RANKS = NUM_RANKS_, NUM_TOPK = NUM_TOPK_;
    static constexpr int TOKEN_BYTES = TOKEN_BYTES_;

    static_assert(sizeof(layout::SymBuffer<>) == 528, "MegaMoE symmetric buffer ABI changed");
    using G = mega_moe::fp8_blockwise::Gemm1<GEMM1_N, TOKEN_BYTES, GEMM1_COMPUTE_M>;
    static_assert(G::kNumThreads == NUM_THREADS, "blockwise GEMM1 launch must match its MMA tile");
    static constexpr uint32_t shared_memory_bytes() { return G::template shared_memory_bytes<NUM_EXPERTS>(); }
    static constexpr int kNumThreads = NUM_THREADS;
    static constexpr int kMinBlocks = 1;
};

template <typename Builder>
__global__ __launch_bounds__(Builder::kNumThreads, Builder::kMinBlocks) void sm89_fp8_blockwise_mega_moe_l1(
    layout::SymBuffer<> sb, uint8_t* local_pool, int rank_idx, int num_tokens, __hg_fp8_e4m3* weight,
    float* weight_scales, uint8_t* output, float* output_scales, float activation_clamp, float activation_alpha, float situ_beta, float situ_linear_beta,
    __hg_fp8_e4m3* weight_replicas, float* scale_replicas, uint4* gemm2_block_info) {
    constexpr uint64_t off_ws = 0, off_it = Builder::LAYOUT_OFF_IT, off_isf = Builder::LAYOUT_OFF_ISF;
    constexpr uint64_t off_iti = Builder::LAYOUT_OFF_ITI;
    constexpr uint64_t off_pt = Builder::LAYOUT_OFF_PT, off_psf = Builder::LAYOUT_OFF_PSF,
        off_meta = Builder::LAYOUT_OFF_GEMM2_META;

    const int64_t local_base = sb.base;
    auto P = [&](uint64_t off) -> void* {
        return off >= off_pt ? local_pool + (off - off_pt) : (void*)(local_base + off);
    };
    mega_moe::fp8_blockwise::dispatch_gemm1_kernel_device<Builder::NUM_RANKS, Builder::NUM_CTAS, Builder::NUM_EXPERTS,
        Builder::NUM_TOPK, Builder::TOKEN_BYTES,
                                                          Builder::GEMM1_N, Builder::PULL_CTAS, Builder::CLAMP,
                                                              Builder::MASTER_EXPERTS,
                                                          Builder::ACTIVATION, Builder::GEMM1_COMPUTE_M, Builder::GEMM2_BLOCK_M>(
        P(off_ws), sb, (int64_t*)P(off_iti), (const uint8_t*)P(off_it), (const float*)P(off_isf),
        (uint8_t*)P(off_pt), (float*)P(off_psf), (int32_t*)P(off_meta), gemm2_block_info, (const uint8_t*)weight,
        weight_scales, output, output_scales, (uint32_t)num_tokens, Builder::NMTPR, activation_clamp, activation_alpha, situ_beta, situ_linear_beta,
        (const uint8_t*)weight_replicas, scale_replicas);
}

template <int N, int K, int GROUPS, int MASTER_GROUPS, int NUM_STAGES, int BLOCK_M, int SCRATCH_ROWS,
          int NUM_RANKS, int NUM_THREADS, int SMEM_SIZE>
struct MegaMoEFp8BlockwiseGemm2Kernel {
    using Family =
        deep_gemm::mega_moe::fp8_blockwise::Gemm2Scatter<N, K, GROUPS, MASTER_GROUPS, NUM_STAGES,
                                                         BLOCK_M, SCRATCH_ROWS, NUM_RANKS>;
    using Kernel = typename Family::Kernel;
    static_assert(Family::kNumThreads == NUM_THREADS, "blockwise scatter launch must match its MMA tile");
    static_assert(Kernel::SharedStorageSize == SMEM_SIZE,
                  "blockwise scatter shared-memory estimate must match its arena");
    static constexpr uint32_t shared_memory_bytes() { return Kernel::SharedStorageSize; }
    static constexpr int kNumThreads = Kernel::MaxThreadsPerBlock;
    static constexpr int kMinBlocks = Kernel::MinBlocksPerMultiprocessor;
};

template <typename Builder>
__global__
__launch_bounds__(Builder::kNumThreads,
                  Builder::kMinBlocks) void sm89_fp8_blockwise_mega_moe_l2(typename Builder::Kernel::Params params) {
    using Kernel = typename Builder::Kernel;
    extern __shared__ char smem[];
    Kernel op;
    op(params, smem);
}

} // namespace deep_gemm
