#pragma once
#include <deep_gemm/impls/fp4_mega_moe_dispatch_gemm1.cuh>
#include <deep_gemm/impls/w4a16_gemm_cutlass3.cuh>

namespace deep_gemm::mega_moe::w4a16 {

// First fused BF16-dispatch + W4A16 GEMM1 implementation. The cross-rank pull
// still lands tokens in the symmetric pool (remote dispatch requires stable
// destination storage), but the grouped GEMM runs in the same persistent
// cooperative kernel immediately after dispatch. This removes the separate
// GEMM launch and the host-visible synchronization between the two stages.
//
// The initial implementation uses a serial pull-then-GEMM handoff. Once this
// path is correctness-locked, W4A16 tiles can consume the existing per-block
// arrival counters to overlap producer and consumer SMs as the FP4 path does.
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
CUTLASS_DEVICE void dispatch_gemm1_device(
    void* workspace_base,
    const layout::SymBuffer<kNumRanks>& sym_buffer,
    const int64_t* __restrict__ input_topk_idx,
    const float* __restrict__ input_topk_weights,
    const uint8_t* __restrict__ input_token,
    const uint8_t* __restrict__ input_sf,
    uint8_t* __restrict__ dispatched_token_pool,
    uint8_t* __restrict__ dispatched_scale_pool,
    float* __restrict__ dispatched_weights_pool,
    const uint32_t num_tokens,
    const uint8_t* __restrict__ gemm1_weight,
    const uint8_t* __restrict__ gemm1_weight_scale,
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

  // Reuse the proven BF16 dispatch-only mode. Token payloads are opaque bytes;
  // kShapeK BF16 values therefore correspond to 2*kShapeK payload bytes and
  // there is no activation-scale payload.
  fp4::dispatch_gemm1_body<
      kNumRanks, kNumCTAs, kNumThreads, kNumExperts,
      kNumExpertsPerRank, kNumTopk, kDispatchBlockM, kChunkBlocks,
      /*kNumPullCTAs=*/kNumCTAs,
      /*kNumGemmCTAs=*/kNumCTAs,
      /*kSmSpecialization=*/false,
      kPullSlots,
      /*kTailGemmBlocks=*/0,
      /*kTailGemmPercent=*/0,
      /*kEnableGemm1=*/false,
      kShapeN,
      /*kTokenBytes=*/kShapeK * sizeof(__ppu_bfloat16),
      kDispatchBlockN,
      kDispatchBlockK,
      /*kGemm1SfbBlockN=*/kDispatchBlockN,
      /*kEnableRemotePull=*/true,
      /*kEnableRemotePush=*/false,
      kDispatchWarpM,
      kDispatchWarpN,
      kDispatchStages,
      /*kQuantizeOutput=*/false,
      /*kSituTanhBf16x2=*/false,
      /*kFusedSitu=*/false, /*kSiluThenClamp=*/false, kDispatchBlockM>(
      workspace_base, sym_buffer, input_topk_idx, input_topk_weights,
      input_token, input_sf, dispatched_token_pool, dispatched_scale_pool,
      dispatched_weights_pool, num_tokens, kNumMaxTokensPerRank,
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
      /*fp4_output=*/nullptr, /*scale_output=*/nullptr, rank);

  const layout::Gemm2Metadata gemm2_meta(
      dispatched_weights_pool + num_pool, kNumExpertsPerRank);

  using W4Kernel = cutlass::gemm::kernel::W4A16GEMM_MMA<
      kShapeN, kShapeK,
      kW4BlockM, kW4BlockN, kW4BlockK,
      kW4WarpM, kW4WarpN, kW4WarpK,
      kNumExpertsPerRank, kW4Stages,
      GemmType::MGroupedNoPad,
      /*N_EXPAND=*/1,
      /*kMoeScatter=*/false,
      /*kFusedSitu=*/true>;
  static_assert(W4Kernel::MaxThreadsPerBlock == kNumThreads,
                "dispatch and W4A16 phases must use the same CTA size");

  typename W4Kernel::Params params{
      reinterpret_cast<const cutlass::bfloat16_t*>(dispatched_token_pool),
      gemm1_weight,
      gemm1_weight_scale,
      reinterpret_cast<cutlass::bfloat16_t*>(activated)};
  params.scheduler = {num_pool, gemm2_meta.src_rows};
  params.valid_m_rows = gemm2_meta.valid_rows;
  params.situ_beta = situ_beta;
  params.situ_linear_beta = situ_linear_beta;
  W4Kernel{}(params, reinterpret_cast<char*>(smem_buf));
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
    const float* __restrict__ input_topk_weights,
    const uint8_t* __restrict__ input_token,
    const uint8_t* __restrict__ input_sf,
    uint8_t* __restrict__ dispatched_token_pool,
    uint8_t* __restrict__ dispatched_scale_pool,
    float* __restrict__ dispatched_weights_pool,
    const uint32_t num_tokens,
    const uint8_t* __restrict__ gemm1_weight,
    const uint8_t* __restrict__ gemm1_weight_scale,
    __ppu_bfloat16* __restrict__ activated,
    const float situ_beta,
    const float situ_linear_beta,
    const uint32_t num_pool,
    const int rank) {
    dispatch_gemm1_device<kNumRanks, kNumCTAs, kNumThreads, kNumExperts, kNumExpertsPerRank, kNumTopk, kDispatchBlockM, kChunkBlocks, kPullSlots, kNumMaxTokensPerRank, kShapeN, kShapeK, kDispatchBlockN, kDispatchBlockK, kDispatchWarpM, kDispatchWarpN, kDispatchStages, kW4BlockM, kW4BlockN, kW4BlockK, kW4WarpM, kW4WarpN, kW4WarpK, kW4Stages>(workspace_base, sym_buffer, input_topk_idx, input_topk_weights, input_token, input_sf, dispatched_token_pool, dispatched_scale_pool, dispatched_weights_pool, num_tokens, gemm1_weight, gemm1_weight_scale, activated, situ_beta, situ_linear_beta, num_pool, rank);
}

}  // namespace deep_gemm::mega_moe::w4a16
