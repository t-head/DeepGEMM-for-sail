#pragma once
#include <deep_gemm/impls/fp4_mega_moe_dispatch_gemm1.cuh>
#include <deep_gemm/impls/w4a16_gemm_cutlass3.cuh>

namespace deep_gemm::mega_moe::w4a16 {

// BF16 dispatch and W4A16 GEMM1 share pool and metadata layouts. Measured
// profiles specialize the same body into separate launches so dispatch threads
// do not constrain GEMM occupancy. The workspace fixes push/pull collectively.
//
// Measured profiles select the rank-local G1 queue. Dedicated pull CTAs publish
// each M block independently, then join consumers after all their own pulls.
// Other profiles use a serial pull-then-GEMM handoff.
template <
    uint32_t kNumRanks,
    uint32_t kNumCTAs,
    uint32_t kNumThreads,
    uint32_t kNumExperts,
    uint32_t kNumExpertsPerRank,
    uint32_t kNumTopk,
    uint32_t kDispatchBlockM,
    uint32_t kChunkBlocks,
    uint32_t kPullSlots,
    uint32_t kNumMaxTokensPerRank,
    uint32_t kShapeN,
    uint32_t kShapeK,
    uint32_t kDispatchBlockN,
    uint32_t kDispatchBlockK,
    uint32_t kDispatchWarpM,
    uint32_t kDispatchWarpN,
    uint32_t kDispatchStages,
    int kW4BlockM,
    int kW4BlockN,
    int kW4BlockK,
    int kW4WarpM,
    int kW4WarpN,
    int kW4WarpK,
    int kW4Stages,
    bool kDynamicWork = false,
    uint32_t kNumPullCTAs = kNumCTAs,
    bool kRunDispatch = true, bool kRunGemm = true, bool kPushDispatch = false>
CUTLASS_DEVICE void dispatch_gemm1_device(
    void* workspace_base,
    const layout::SymBuffer<kNumRanks>& sym_buffer,
    const int64_t* __restrict__ input_topk_idx,
    const uint8_t* __restrict__ input_token,
    const uint8_t* __restrict__ input_sf,
    uint8_t* __restrict__ dispatched_token_pool,
    uint8_t* __restrict__ dispatched_scale_pool,
    void* __restrict__ gemm2_metadata,
    const uint32_t num_tokens,
    const uint8_t* __restrict__ gemm1_weight,
    const uint8_t* __restrict__ gemm1_weight_scale,
    const uint8_t* __restrict__ gemm1_weight_replicas,
    const uint8_t* __restrict__ gemm1_weight_scale_replicas,
    const uint32_t num_master_experts,
    __ppu_bfloat16* __restrict__ activated,
    const float situ_beta,
    const float situ_linear_beta,
    const uint32_t num_pool,
    const int rank) {
  extern __shared__ __align__(16) unsigned char smem_buf[];

  static_assert(kNumExpertsPerRank < 128,
                "fused W4A16 dispatch currently uses the direct no-pad scheduler");
  static_assert(kShapeK % 32 == 0,
                "W4A16 MXFP4 K must be divisible by 32");
  static_assert(kShapeN % 16 == 0,
                "fused SiTU needs complete G=8 gate/up pairs");

  static_assert(kRunDispatch || kRunGemm);
  static_assert(!kPushDispatch || (!kDynamicWork && !(kRunDispatch && kRunGemm)));
  static_assert(!kDynamicWork || (kRunDispatch && kRunGemm));
  static_assert(!kDynamicWork || (kNumPullCTAs > 0 && kNumPullCTAs < kNumCTAs));
  if constexpr (kRunDispatch) {
    // Reuse the proven BF16 dispatch-only mode. Token payloads are opaque bytes;
    // kShapeK BF16 values therefore correspond to 2*kShapeK payload bytes and
    // there is no activation-scale payload.
    fp4::dispatch_gemm1_body<
        kNumRanks, kNumCTAs, kNumThreads, kNumExperts,
        kNumExpertsPerRank, kNumTopk, kDispatchBlockM, kChunkBlocks,
        /*kNumPullCTAs=*/kDynamicWork ? kNumPullCTAs : kNumCTAs,
        /*kNumGemmCTAs=*/kDynamicWork ? kNumCTAs - kNumPullCTAs : kNumCTAs,
        /*kSmSpecialization=*/kDynamicWork,
        kPullSlots,
        /*kEnableGemm1=*/false,
        kShapeN,
        /*kTokenBytes=*/kShapeK * sizeof(__ppu_bfloat16),
        kDispatchBlockN,
        kDispatchBlockK,
        /*kGemm1SfbBlockN=*/kDispatchBlockN,
        /*kEnableRemotePull=*/!kPushDispatch,
        /*kEnableRemotePush=*/kPushDispatch,
        kDispatchWarpM,
        kDispatchWarpN,
        kDispatchStages,
        /*kQuantizeOutput=*/false,
        /*kSituTanhBf16x2=*/false,
        /*kFusedSitu=*/false, /*kSiluThenClamp=*/false, kDispatchBlockM, kDynamicWork,
        /*kDecodeHandshake=*/true, /*kResetReceiveCounts=*/true>(
        workspace_base, sym_buffer, input_topk_idx,
        input_token, input_sf, dispatched_token_pool, dispatched_scale_pool,
        gemm2_metadata, num_tokens, kNumMaxTokensPerRank,
        kShapeK * sizeof(__ppu_bfloat16), /*sf_bytes=*/0,
        /*gemm1_weight=*/nullptr, /*gemm1_weight_sf=*/nullptr,
        /*gemm1_output=*/nullptr, /*gemm1_output_stride=*/0,
        /*gemm1_weight_sf_u16=*/nullptr,
        /*gemm1_weight_replicas=*/nullptr,
        /*gemm1_weight_sf_u16_replicas=*/nullptr,
        /*num_master_experts=*/kNumExpertsPerRank,
        /*sfa_scratch=*/nullptr, /*activation_clamp=*/0.0f,
        /*fused_swiglu=*/false, /*fused_situ=*/false,
        situ_beta, situ_linear_beta,
        /*fp4_output=*/nullptr, /*scale_output=*/nullptr, rank,
        0, 0xffffffffu, false, /*finalize_dispatch=*/!kDynamicWork,
        // Non-queue dispatch already joins the grid before finalizing counts.
        /*synchronize_pull=*/false);

  }
  if constexpr (kRunGemm) {

    const layout::Gemm2Metadata gemm2_meta(
        gemm2_metadata, kNumExpertsPerRank);

    using W4Kernel = cutlass::gemm::kernel::W4A16GEMM_MMA<
        kShapeN, kShapeK,
        kW4BlockM, kW4BlockN, kW4BlockK,
        kW4WarpM, kW4WarpN, kW4WarpK,
        kNumExpertsPerRank, kW4Stages,
        GemmType::MGroupedNoPad,
        /*N_EXPAND=*/1,
        /*kMoeScatter=*/false,
        /*kFusedSitu=*/true, /*kUseReplicaWeights=*/false,
        /*kDynamicWork=*/kDynamicWork>;
    static_assert(W4Kernel::MaxThreadsPerBlock == kNumThreads,
                  "W4A16 GEMM phase requires the MMA CTA size");

    typename W4Kernel::Params params{
        reinterpret_cast<const cutlass::bfloat16_t*>(dispatched_token_pool),
        gemm1_weight,
        gemm1_weight_scale,
        reinterpret_cast<cutlass::bfloat16_t*>(activated)};
    // UltraEP routes use local physical slots: masters first, then replicas.
    params.ptr_b_replicas = gemm1_weight_replicas;
    params.ptr_scale_replicas = gemm1_weight_scale_replicas;
    params.num_master_groups = num_master_experts;
    params.scheduler = {num_pool, gemm2_meta.src_rows};
    params.valid_m_rows = gemm2_meta.valid_rows;
    params.situ_beta = situ_beta;
    params.situ_linear_beta = situ_linear_beta;
    if constexpr (kDynamicWork) {
      const layout::Workspace ws(workspace_base, kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);
      params.queue_counter = ws.get_gemm1_work_count_ptr();
      params.arrival_counts = ws.get_l1_arrival_count_ptr(0);
      params.dispatch_block_m = kDispatchBlockM;
    }
    W4Kernel{}(params, reinterpret_cast<char*>(smem_buf));
    if constexpr (kDynamicWork) {
      const layout::Workspace ws(workspace_base, kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);
      const auto block_sync = []() { __syncthreads(); };
      comm::grid_sync<kNumCTAs, 1>(ws, blockIdx.x, threadIdx.x, block_sync);
      for (uint32_t e = blockIdx.x * kNumThreads + threadIdx.x; e < 2 * kNumExperts + kNumExpertsPerRank;
           e += kNumCTAs * kNumThreads) *ws.get_expert_send_count_ptr(e) = 0;
    }
  }
}

template <
    uint32_t kNumRanks,
    uint32_t kNumCTAs,
    uint32_t kNumThreads,
    uint32_t kNumExperts,
    uint32_t kNumExpertsPerRank,
    uint32_t kNumTopk,
    uint32_t kDispatchBlockM,
    uint32_t kChunkBlocks,
    uint32_t kPullSlots,
    uint32_t kNumMaxTokensPerRank,
    uint32_t kShapeN,
    uint32_t kShapeK,
    uint32_t kDispatchBlockN,
    uint32_t kDispatchBlockK,
    uint32_t kDispatchWarpM,
    uint32_t kDispatchWarpN,
    uint32_t kDispatchStages,
    int kW4BlockM,
    int kW4BlockN,
    int kW4BlockK,
    int kW4WarpM,
    int kW4WarpN,
    int kW4WarpK,
    int kW4Stages>
__global__ __launch_bounds__(kNumThreads, 1) void
dispatch_gemm1_kernel(
    void* workspace_base,
    const layout::SymBuffer<kNumRanks> sym_buffer,
    const int64_t* __restrict__ input_topk_idx,
    const uint8_t* __restrict__ input_token,
    const uint8_t* __restrict__ input_sf,
    uint8_t* __restrict__ dispatched_token_pool,
    uint8_t* __restrict__ dispatched_scale_pool,
    void* __restrict__ gemm2_metadata,
    const uint32_t num_tokens,
    const uint8_t* __restrict__ gemm1_weight,
    const uint8_t* __restrict__ gemm1_weight_scale,
    const uint8_t* __restrict__ gemm1_weight_replicas,
    const uint8_t* __restrict__ gemm1_weight_scale_replicas,
    const uint32_t num_master_experts,
    __ppu_bfloat16* __restrict__ activated,
    const float situ_beta,
    const float situ_linear_beta,
    const uint32_t num_pool,
    const int rank) {
    dispatch_gemm1_device<
        kNumRanks, kNumCTAs, kNumThreads, kNumExperts, kNumExpertsPerRank, kNumTopk,
        kDispatchBlockM, kChunkBlocks, kPullSlots, kNumMaxTokensPerRank, kShapeN, kShapeK,
        kDispatchBlockN, kDispatchBlockK, kDispatchWarpM, kDispatchWarpN, kDispatchStages,
        kW4BlockM, kW4BlockN, kW4BlockK, kW4WarpM, kW4WarpN, kW4WarpK, kW4Stages>(
        workspace_base, sym_buffer, input_topk_idx, input_token, input_sf,
        dispatched_token_pool, dispatched_scale_pool, gemm2_metadata, num_tokens,
        gemm1_weight, gemm1_weight_scale, gemm1_weight_replicas, gemm1_weight_scale_replicas,
        num_master_experts, activated, situ_beta, situ_linear_beta, num_pool, rank);
}

}  // namespace deep_gemm::mega_moe::w4a16
