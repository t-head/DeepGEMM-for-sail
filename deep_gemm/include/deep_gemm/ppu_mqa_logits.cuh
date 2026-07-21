#pragma once
#include "mqa_logits_utils.cuh"

namespace cutlass::gemm::kernel {

template <typename ElementQK, typename ElementAcc, typename ElementLogits, typename ElementWeights,
          uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_QH, uint32_t BLOCK_KV,
          uint32_t WARP_QH, uint32_t WARP_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          typename StrideKType = uint32_t,
          bool kIsCompressedLogits = false>
class PPUMqaLogits {
  static_assert(cute::is_same_v<StrideKType, uint32_t> || cute::is_same_v<StrideKType, uint64_t>,
                "StrideKType must be uint32_t or uint64_t");
public:
  static constexpr int BLOCK_M = BLOCK_KV;
  static constexpr int BLOCK_N = BLOCK_QH;
  static constexpr int BLOCK_K = kHeadDim;
  static constexpr int WARP_M = WARP_KV;
  static constexpr int WARP_N = WARP_QH;
  static constexpr int BLOCK_Q = BLOCK_QH / kNumHeads;
  static constexpr int WARP_Q = WARP_QH / kNumHeads;

  using StrideAB = Stride<Int<BLOCK_K>, _1>;

  using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
  static constexpr int WarpOnM = BLOCK_M / WARP_M;
  static constexpr int WarpOnN = BLOCK_N / WARP_N;
#if __HGGC_ARCH__ == 100
    using ArchTag = cutlass::arch::PPU0010;
#else
    using ArchTag = cutlass::arch::PPU0015;
#endif

  using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ArchTag, ElementQK,ElementQK,ElementAcc>::type;
  using MmaK_type = typename cutlass::platform::conditional<sizeof(ElementQK) == 2, _16, _32 >::type;

  static constexpr int InstM = 16;
  static constexpr int InstN = 16;
  using WarpIterM = Int<BLOCK_M / WARP_M>;
  using WarpIterN = Int<BLOCK_N / WARP_N>;
  using MmaIterM = Int<WARP_M / InstM>;
  using MmaIterN = Int<WARP_N / InstN>;
  using PermutationMNK = Tile<
      Layout<Shape<Int<InstM>, WarpIterM, MmaIterM>, Stride<_1, Int<WARP_M>, Int<InstM>> >,
      Layout<Shape<Int<InstN>, WarpIterN, MmaIterN>, Stride<_1, Int<WARP_N>, Int<InstN>> >,
      MmaK_type
    >;
  using TiledMma = TiledMMA<MMA_Atom<MmaInst>, Layout<Shape<WarpIterM, WarpIterN, _1>>, PermutationMNK>;

  static constexpr int NumThreadsPerCTA = size(TiledMma{});
  // WarpInterleaving is enabled only when NumThreadsPerCTA is 512, which satisfy the condition that 2 warp group partitioned onto separate WEs.
  static constexpr bool WarpInterleaving = (NumThreadsPerCTA == 512);

  using DefaultOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementQK, false, Int<BLOCK_M>, Int<BLOCK_K>, false>;
  using DefaultOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementQK, false, Int<BLOCK_N>, Int<BLOCK_K>, true>;
  // A
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;
  // B
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<kNumKVStages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<kNumQStages>{})));

  constexpr static uint32_t CTA_M = shape<0>(TileShape{});
  constexpr static uint32_t CTA_N = shape<1>(TileShape{});
  constexpr static uint32_t CTA_K = shape<2>(TileShape{});

  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  // ScaleA (k_scales, float per KV row) -- AIU load
  using DefaultOperandSFA =
      cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, float, false, _1, Int<CTA_M>, false, 0, false>;
  using SmemLayoutAtomSFA = typename DefaultOperandSFA::SmemLayoutAtom;
  using GmemTiledCopySFA = typename DefaultOperandSFA::GmemTiledCopy;
  using SmemLayoutSFA = decltype(tile_to_shape(
      SmemLayoutAtomSFA{}, make_shape(_1{}, Int<CTA_M>{}, Int<kNumKVStages>{})));

  // ScaleB (weights) -- AIU load, use ElementWeights directly (like fp4_mqa_logits.cuh)
  using DefaultOperandSFB =
      cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementWeights, false, _1, Int<CTA_N>, false, 0, false>;
  using SmemLayoutAtomSFB = typename DefaultOperandSFB::SmemLayoutAtom;
  using GmemTiledCopySFB = typename DefaultOperandSFB::GmemTiledCopy;
  using SmemLayoutSFB = decltype(tile_to_shape(
      SmemLayoutAtomSFB{}, make_shape(_1{}, Int<CTA_N>{}, Int<kNumQStages>{})));

  using StrideSFA = Stride<Int<CTA_M>, _1>;
  using StrideSFB = Stride<Int<CTA_N>, _1>;

  // Kernel level shared memory storage
  struct SharedStorage {
    cute::array_aligned<ElementQK, cute::cosize_v<SmemLayoutA>> smem_k;
    cute::array_aligned<ElementQK, cute::cosize_v<SmemLayoutB>> smem_q;
    cute::array_aligned<float, cute::cosize_v<SmemLayoutSFA>> smem_k_scales;
    cute::array_aligned<ElementWeights, cute::cosize_v<SmemLayoutSFB>> smem_weight;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    const ElementQK * ptr_q;
    const ElementQK * ptr_k;
    const float * k_scales;
    const ElementWeights * weights;
    uint32_t* cu_seq_len_k_start;
    uint32_t* cu_seq_len_k_end;
    ElementLogits* logits;
    const uint32_t seq_len_q;
    const uint32_t seq_len_k;
    const StrideKType stride_k;
  };

  // Kernel entry point API
  using Params = Arguments;

  GmemTiledCopyA gmem_tiled_copy_A;
  GmemTiledCopyB gmem_tiled_copy_B;

  GmemTiledCopySFA gmem_tiled_copy_SFA;
  GmemTiledCopySFB gmem_tiled_copy_SFB;

  CUTLASS_DEVICE void
  init_aiu_copy() {
    using TilerA = typename GmemTiledCopyA::Tiler_MN;
    using TilerB = typename GmemTiledCopyB::Tiler_MN;
    using TilerSFA = typename GmemTiledCopySFA::Tiler_MN;
    using TilerSFB = typename GmemTiledCopySFB::Tiler_MN;

    gmem_tiled_copy_A.desc_.template init<ElementQK, false, get<0>(TilerA{}), get<1>(TilerA{})>(nullptr, CTA_M, CTA_K, StrideAB{});
    gmem_tiled_copy_B.desc_.template init<ElementQK, false, get<0>(TilerB{}), get<1>(TilerB{})>(nullptr, CTA_N, CTA_K, StrideAB{});

    gmem_tiled_copy_SFA.desc_.template init<float, false, get<0>(TilerSFA{}), get<1>(TilerSFA{})>(
        nullptr, 1, CTA_M, StrideSFA{});
    gmem_tiled_copy_SFB.desc_.template init<ElementWeights, false, get<0>(TilerSFB{}), get<1>(TilerSFB{})>(
        nullptr, 1, CTA_N, StrideSFB{});
  };

  CUTLASS_DEVICE auto
  load_init(Params const& params) {
    // load init A
    Tensor mA_mk = make_tensor(make_gmem_ptr(params.ptr_k), Shape<Int<CTA_M>, Int<CTA_K>>{}, StrideAB{});  // (m,k)
    Tensor mA_mk_mix = make_mix_tensor_like(mA_mk);                                                          // (m,k)
    Tensor gA = local_tile(mA_mk_mix, TileShape{}, make_coord(0, 0, _), Step<_1, X, _1>{});                  // (BLK_M,BLK_K,k)

    // load init B
    Tensor mB_nk = make_tensor(make_gmem_ptr(params.ptr_q), Shape<Int<CTA_N>, Int<CTA_K>>{}, StrideAB{});  //(n,k)
    Tensor mB_nk_mix = make_mix_tensor_like(mB_nk);                                                          // (n,k)
    Tensor gB = local_tile(mB_nk_mix, TileShape{}, make_coord(0, _, 0), Step<X, _1, _1>{});                  // (BLK_N,BLK_K,k)

    // load init scale A (k_scales) -- AIU style
    Tensor mSFA_m = make_tensor(make_gmem_ptr(params.k_scales), Shape<_1, Int<CTA_M>>{}, StrideSFA{});
    Tensor mSFA_m_mix = make_mix_tensor_like(mSFA_m);
    Tensor gSFA = local_tile(mSFA_m_mix, Shape<_1, Int<CTA_M>>{}, make_coord(_, 0));

    // load init scale B (weights) -- AIU style, use ElementWeights directly
    Tensor mSFB_n = make_tensor(make_gmem_ptr(params.weights), Shape<_1, Int<CTA_N>>{}, StrideSFB{});
    Tensor mSFB_n_mix = make_mix_tensor_like(mSFB_n);
    Tensor gSFB = local_tile(mSFB_n_mix, Shape<_1, Int<CTA_N>>{}, make_coord(0, _));

    return cute::make_tuple(gA, gB, gSFA, gSFB);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    int warp_idx = canonical_warp_idx_sync();
    int thread_idx = int(threadIdx.x);
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    const auto& num_q_blocks = ceil_div(params.seq_len_q, BLOCK_Q);

    // init aiu copy and async copy
    init_aiu_copy();

    // init input tensors
    auto load_inputs = load_init(params);
    Tensor gA = get<0>(load_inputs);
    Tensor gB = get<1>(load_inputs);
    Tensor gSFA = get<2>(load_inputs);
    Tensor gSFB = get<3>(load_inputs);

    Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)
    // Partition the copying of A and B tiles across the threads
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    Tensor sSFA = make_tensor(make_smem_ptr(shared_storage.smem_k_scales.data()), SmemLayoutSFA{});
    Tensor sSFB = make_tensor(make_smem_ptr(shared_storage.smem_weight.data()), SmemLayoutSFB{});

    auto gmem_thr_copy_SFA = gmem_tiled_copy_SFA.get_thread_slice(thread_idx);
    auto gmem_thr_copy_SFB = gmem_tiled_copy_SFB.get_thread_slice(thread_idx);

    Tensor tSFAgSFA = gmem_thr_copy_SFA.partition_S(gSFA);
    Tensor tSFAsSFA = gmem_thr_copy_SFA.partition_D(sSFA);

    Tensor tSFBgSFB = gmem_thr_copy_SFB.partition_S(gSFB);
    Tensor tSFBsSFB = gmem_thr_copy_SFB.partition_D(sSFB);

    TiledMma tiled_mma;
    Tensor accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                     // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                     // (MMA,MMA_N,MMA_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                  // MMA_K

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(warp_idx * 32);
    Tensor tCsA            = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));                  // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                   // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(warp_idx * 32);
    Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));                  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    constexpr bool enable_print = false;

    // Block scheduler
    uint32_t block_q_idx = blockIdx.x, q_iter_idx = 0;
    // Per-token KV range arrays for compressed logits mode (dead code when kIsCompressedLogits=false)
    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];

    ElementLogits weights[kNumHeads / 4];

    clear(accum);
    auto tKgK = tAgA;
    auto tSFKgSFK = tSFAgSFA;
    auto tQgQ = tBgB;
    auto tSFQgSFQ = tSFBgSFB;

    const auto& lane_idx = get_lane_idx();
    const auto& warp_offset = (warp_idx % WarpOnM) * WARP_M;
    const auto& v_0_offset = lane_idx / 4 + 0;
    const auto& v_1_offset = lane_idx / 4 + 8;
    uint32_t warp_q_idx = warp_idx / WarpOnM;
    int warp_group_id = warp_idx / 8;

    static constexpr uint32_t kMmaIterM = MmaIterM{};
    float scale_kv_array[kMmaIterM * 2];

    auto load_q_g2s = [&](uint32_t block_q_idx) {
        gmem_tiled_copy_B.desc_.dim_h = (params.seq_len_q - block_q_idx * BLOCK_Q) * kNumHeads;
        copy_aiu(gmem_tiled_copy_B, tBgB(_,_,_,0), tBsB(_,_,_,0), warp_idx);
        // SFB (weights) via AIU
        int residual_n_weights = (params.seq_len_q - block_q_idx * BLOCK_Q) * kNumHeads;
        gmem_tiled_copy_SFB.desc_.dim_w = residual_n_weights;
        copy_aiu(gmem_tiled_copy_SFB, tSFBgSFB(_,_,_,0), tSFBsSFB(_,_,_,0), warp_idx);
    };

    auto load_kv_g2s = [&](uint32_t kv_start, uint32_t kv_end, uint32_t kv_block_idx, uint32_t smem_stage) {
        int residual_m = kv_end - (kv_start + kv_block_idx * BLOCK_KV);
        gmem_tiled_copy_A.desc_.dim_h = residual_m;
        gmem_tiled_copy_SFA.desc_.dim_w = residual_m;
        if (enable_print) {
            printf("    kv_block_idx = %d, dim_h = %d\n", kv_block_idx, residual_m);
        }
        copy_aiu(gmem_tiled_copy_A, tAgA(_,_,_,0), tAsA(_,_,_,smem_stage), warp_idx);
        if constexpr(sizeof(ElementQK) == 1) {
            copy_aiu(gmem_tiled_copy_SFA, tSFAgSFA(_,_,_,0), tSFAsSFA(_,_,_,smem_stage), warp_idx);
        }
        tAgA.data() = tAgA.data() + BLOCK_KV * kHeadDim;
        tSFAgSFA.data() = tSFAgSFA.data() + BLOCK_KV;
    };

    auto load_q_s2r = [&]() {
        copy(smem_tiled_copy_B, tCsB(_,_,_,0), tCrB_copy_view);
        deep_gemm::load_weights_from_smem_ld_shared<ElementWeights, ElementLogits, kNumHeads>(
            sSFB(_,_,0), weights, lane_idx, warp_q_idx);
    };

    auto load_kv_scale_s2r = [&](uint32_t kv_stage_idx) {
        float * smem_kv_scales = sSFA(_,_,kv_stage_idx).data().get();
        for (int m = 0; m < kMmaIterM; m++) {
            uint32_t mma_offset = m * InstM;
            scale_kv_array[m * 2    ] = (sizeof(ElementQK) == 2) ? 1 : ld_shared(smem_kv_scales + warp_offset + mma_offset + v_0_offset);
            scale_kv_array[m * 2 + 1] = (sizeof(ElementQK) == 2) ? 1 : ld_shared(smem_kv_scales + warp_offset + mma_offset + v_1_offset);
        }
    };

    auto epilogue = [&](uint32_t block_q_idx, uint32_t kv_start, uint32_t kv_block_idx) {
        const auto& kv_offset = kv_start + kv_block_idx * BLOCK_KV + warp_offset;
        static constexpr uint32_t kNumAccumPerMma = 8;
        CUTE_STATIC_ASSERT(kNumHeads % 8 == 0);
        CUTE_STATIC_ASSERT(WARP_Q == 1);
        for (int m = 0; m < kMmaIterM; m++) {
            uint32_t mma_offset = m * InstM;
            float scale_kv_0 = scale_kv_array[m * 2];
            float scale_kv_1 = scale_kv_array[m * 2 + 1];

            if constexpr (cute::is_same_v<ElementLogits, float>) {
                // Float epilogue: reduce over heads, scale by per-row KV scale and store
                float v_0, v_1;
                deep_gemm::float_epilogue_reduce_weights(accum, m, weights, v_0, v_1);

                // Inter-thread reduction
                deep_gemm::shfl_xor_reduce_2(v_0, v_1);

                // Store into the global memory
                const uint32_t& q_idx = block_q_idx * BLOCK_Q + warp_q_idx;
                if constexpr (kIsCompressedLogits) {
                    const uint32_t rel_kv = kv_offset + mma_offset - seq_k_start[warp_q_idx];
                    const uint32_t len = seq_k_end[warp_q_idx] - seq_k_start[warp_q_idx];
                    if (rel_kv + v_0_offset < len)
                        params.logits[q_idx * params.stride_k + rel_kv + v_0_offset] = v_0 * scale_kv_0;
                    if (rel_kv + v_1_offset < len)
                        params.logits[q_idx * params.stride_k + rel_kv + v_1_offset] = v_1 * scale_kv_1;
                } else {
                    params.logits[q_idx * params.stride_k + kv_offset + mma_offset + v_0_offset] = v_0 * scale_kv_0;
                    params.logits[q_idx * params.stride_k + kv_offset + mma_offset + v_1_offset] = v_1 * scale_kv_1;
                }
            } else {
                // BF16 vectorized epilogue: separate cvt and fma2 phases with __ppu_sched_bound()
                constexpr int kTotalTransforms = 4 * (kNumHeads / InstN);
                __ppu_bfloat162 sum_0 = {0, 0}, sum_1 = {0, 0};
                uint32_t cvt_buf[kTotalTransforms];

                // cvt phase: issue all ppu.cvt.rtte.bf16x2.f32.relu
                deep_gemm::cvt_accum_to_bf16x2_buf<kTotalTransforms>(accum, m, cvt_buf);
                __ppu_sched_bound();
                // fma2 phase: issue all __hfma2
                deep_gemm::fma2_phase_weights<kTotalTransforms>(cvt_buf, weights, sum_0, sum_1);
                __ppu_bfloat16 v_0_bf = __hadd(__low2bfloat16(sum_0), __high2bfloat16(sum_0));
                __ppu_bfloat16 v_1_bf = __hadd(__low2bfloat16(sum_1), __high2bfloat16(sum_1));

                // Apply per-token KV scale
                v_0_bf = __hmul(v_0_bf, (__ppu_bfloat16)scale_kv_0);
                v_1_bf = __hmul(v_1_bf, (__ppu_bfloat16)scale_kv_1);

                // Packed cross-lane reduction
                __ppu_bfloat162 packed = deep_gemm::shfl_xor_reduce_bf16x2({v_0_bf, v_1_bf});

                // Store
                const uint32_t& q_idx = block_q_idx * BLOCK_Q + warp_q_idx;
                if constexpr (kIsCompressedLogits) {
                    const uint32_t rel_kv = kv_offset + mma_offset - seq_k_start[warp_q_idx];
                    const uint32_t len = seq_k_end[warp_q_idx] - seq_k_start[warp_q_idx];
                    if (rel_kv + v_0_offset < len)
                        params.logits[q_idx * params.stride_k + rel_kv + v_0_offset] = __low2bfloat16(packed);
                    if (rel_kv + v_1_offset < len)
                        params.logits[q_idx * params.stride_k + rel_kv + v_1_offset] = __high2bfloat16(packed);
                } else {
                    params.logits[q_idx * params.stride_k + kv_offset + mma_offset + v_0_offset] = __low2bfloat16(packed);
                    params.logits[q_idx * params.stride_k + kv_offset + mma_offset + v_1_offset] = __high2bfloat16(packed);
                }
            }
        }
    };

    while (block_q_idx < num_q_blocks) {
        CUTE_TIE_DECL(
            (deep_gemm::load_schedule<BLOCK_Q, kNumQStages, BLOCK_KV, kIsCompressedLogits>(
                block_q_idx, q_iter_idx, 1, params.seq_len_q, params.seq_len_k,
                params.cu_seq_len_k_start, params.cu_seq_len_k_end,
                seq_k_start, seq_k_end)),
            q_stage_idx, q_phase, kv_start, kv_end, num_kv_blocks);
        tAgA.data() = tKgK.data() + kv_start * kHeadDim;
        tSFAgSFA.data() = tSFKgSFK.data() + kv_start;
        tBgB.data() = tQgQ.data() + block_q_idx * BLOCK_Q * kNumHeads * kHeadDim;
        tSFBgSFB.data() = tSFQgSFQ.data() + block_q_idx * BLOCK_Q * kNumHeads;

        if (enable_print) {
            printf("block_q_idx = %d, kv_start = %d, kv_end=%d, num_kv_blocks = %d\n",
                block_q_idx, kv_start, kv_end, num_kv_blocks);
        }

        uint32_t current_stage_kv = 0;
        if (num_kv_blocks > 0) {
            // Issue AIU Q
            load_q_g2s(block_q_idx);

            // Issue AIU K in prologue
            for (int kv_pipe = 0; kv_pipe < kNumKVStages - 1; kv_pipe++) {
                if (kv_pipe < num_kv_blocks) {
                    load_kv_g2s(kv_start, kv_end, kv_pipe, kv_pipe);
                }
                current_stage_kv++;
                cp_async_fence();
            }

            // wait AIU Q and first K
            cp_async_wait<kNumKVStages - 2>();
            __syncthreads();

            // load Q + weights to vreg
            load_q_s2r();
        }

        deep_gemm::warp_interleave_start<WarpInterleaving>(warp_group_id, NumThreadsPerCTA);

        // Compute over KV blocks
        for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
            uint32_t kv_stage_idx = kv_block_idx % kNumKVStages;

            deep_gemm::warp_interleave_sync<WarpInterleaving>(warp_group_id, NumThreadsPerCTA);

            int kv_block_idx_copy = kv_block_idx + kNumKVStages - 1;
            if (kv_block_idx_copy < num_kv_blocks) {
                load_kv_g2s(kv_start, kv_end, kv_block_idx_copy, current_stage_kv);
            }
            cp_async_fence();
            current_stage_kv = (current_stage_kv + 1) % kNumKVStages;

            if (enable_print) {
                printf("    block_q_idx = %d, kv_block_idx = %d, kv_stage_idx = %d\n",
                    block_q_idx, kv_block_idx, kv_stage_idx);
                if (kv_stage_idx < 0) {
                    print("error error error, kv_stage_idx < 0, value =  ", kv_stage_idx);
                }
            }
            // load V to vreg
            Tensor tCsA_p = tCsA(_,_,_,kv_stage_idx);
            copy(smem_tiled_copy_A, tCsA_p, tCrA_copy_view);

            // compute
            cute::gemm(tiled_mma, accum, tCrA, tCrB, accum);

            // load scale from tsm to vreg before warp_interleave_arrive
            load_kv_scale_s2r(kv_stage_idx);

            deep_gemm::warp_interleave_arrive<WarpInterleaving>(warp_group_id, NumThreadsPerCTA);

            // Reduce + store logits
            epilogue(block_q_idx, kv_start, kv_block_idx);

            // wait for next K, load next K to vreg
            cp_async_wait<kNumKVStages - 2>();
            if constexpr (!WarpInterleaving) {
                __syncthreads();
            }

            clear(accum);
        }

        deep_gemm::warp_interleave_end<WarpInterleaving>(warp_group_id, NumThreadsPerCTA);

        cp_async_wait<0>();
        __syncthreads();

        // Jump to the next block
        CUTE_TIE(deep_gemm::get_next_block_q_idx(block_q_idx, q_iter_idx), block_q_idx, q_iter_idx);
    }

  }

};

} // namespace cutlass::gemm::kernel


namespace deep_gemm {

template <typename ElementQK, typename ElementAcc, typename ElementLogits, typename ElementWeights,
          uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_QH, uint32_t BLOCK_KV,
          uint32_t WARP_QH, uint32_t WARP_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          typename StrideKType = uint32_t,
          bool kIsCompressedLogits = false>
class Attention {

public:
    Attention() = default;

    static uint32_t generate_id() {
        static uint32_t id = 0;
        return ++id;
    }

    static void run(const ElementQK * ptr_q,
                    const ElementQK * ptr_k,
                    const float * k_scales,
                    const ElementWeights * weights,
                    uint32_t* cu_seq_len_k_start,
                    uint32_t* cu_seq_len_k_end,
                    ElementLogits* logits,
                    const uint32_t seq_len_q, const uint32_t seq_len_k, const StrideKType stride_k,
                    hggcStream_t stream, int num_sms) {

        using AttnKernel = cutlass::gemm::kernel::PPUMqaLogits<ElementQK, ElementAcc, ElementLogits, ElementWeights, kNumHeads, kHeadDim, BLOCK_QH, BLOCK_KV, WARP_QH, WARP_KV, kNumQStages, kNumKVStages, StrideKType, kIsCompressedLogits>;

        static constexpr int MaxThreadsPerBlock = AttnKernel::MaxThreadsPerBlock;

        int max_blocks_per_cu = compute_occupancy_for_kernel<AttnKernel>();

        typename AttnKernel::Arguments arguments{ptr_q, ptr_k, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits,
                                                 seq_len_q, seq_len_k, stride_k};
        auto params = arguments;
        dim3 const block(MaxThreadsPerBlock, 1, 1);
        dim3 const grid(num_sms * max_blocks_per_cu, 1, 1);
        int smem_size_kernel = AttnKernel::SharedStorageSize;

        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()){
            std::string data_type_str = "unknown";
            if (cute::is_same_v<ElementQK, cutlass::bfloat16_t>) {
                data_type_str = "bf16";
            } else if (cute::is_same_v<ElementQK, cutlass::float_e4m3_t>) {
                data_type_str = "fp8";
            } else if (cute::is_same_v<ElementQK, int8_t>) {
                data_type_str = "int8";
            }

            dg_prof_params.set_mqa_logits_params(data_type_str, seq_len_q, seq_len_k, kNumHeads, kHeadDim, stream);
            if constexpr (cute::is_same_v<ElementLogits, __ppu_bfloat16>) {
                dg_prof_params.add_params("logits_dtype", std::string("bf16"));
            }
            if constexpr (cute::is_same_v<ElementWeights, __ppu_bfloat16>) {
                dg_prof_params.add_params("weights_dtype", std::string("bf16"));
            }
        }

        int max_active_tb_num = max_blocks_per_cu;
        const int threadblock_count = num_sms * max_active_tb_num;
        char *pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            hggcFuncAttributes attr;
            hggcFuncGetAttributes(&attr, cutlass::device_kernel<AttnKernel>);

            printf("[mqa_logits:]\n");
            printf("kNumHeads:%u, kHeadDim:%u, seq_len_q:%u, seq_len_k:%u, stride_k:%llu\n",
                kNumHeads, kHeadDim, seq_len_q, seq_len_k, static_cast<unsigned long long>(stride_k));

            printf("ThreadblockShape[%d, %d], WarpShape[%d, %d], kNumQStages:%d, kNumKVStages:%d\n",
                BLOCK_QH, BLOCK_KV, WARP_QH, WARP_KV, kNumQStages, kNumKVStages);

            printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d, num_threads:%d\n", num_sms, max_active_tb_num, threadblock_count, block.x);
            printf("smem_size:%d, vreg:%d, stack:%d\n", smem_size_kernel, int(attr.numRegs), int(attr.localSizeBytes));
            printf("compressed_logits:%s, ", kIsCompressedLogits ? "true" : "false");
            printf("weights_bf16:%s\n", cute::is_same_v<ElementWeights, __ppu_bfloat16> ? "true" : "false");
        }

        ProfilingInterface::Instance().instrument(true, dg_prof_params);
        cutlass::device_kernel<AttnKernel><<<grid, block, smem_size_kernel, stream>>>(params);
        ProfilingInterface::Instance().instrument(false, dg_prof_params);
    }
};

};  // namespace deep_gemm