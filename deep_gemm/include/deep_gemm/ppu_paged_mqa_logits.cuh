#pragma once
#include "mqa_logits_utils.cuh"
#include "paged_mqa_logits_scheduler.cuh"


namespace cutlass::gemm::kernel {

template <typename ElementQK, typename ElementAcc, typename ElementLogits, typename ElementWeights,
          uint32_t kNextN, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV, uint32_t WARP_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV>
class PPUPagedMqaLogits {
public:
  static constexpr int BLOCK_M = BLOCK_KV;
  static constexpr int BLOCK_N = kNextN * kNumHeads;
  static constexpr int BLOCK_K = kHeadDim;
  static constexpr int WARP_M = WARP_KV;
  static constexpr int WARP_N = kNumHeads;

  static constexpr int BLOCK_Q = kNextN;
  static constexpr int WARP_Q = 1;
  static constexpr int kNumMathWarpGroups = SPLIT_KV / BLOCK_KV;

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


  static constexpr int MaxThreadsPerBlock = kNumMathWarpGroups * size(TiledMma{});
  static constexpr int WarpOnGroup = size(TiledMma{}) / 32;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr bool WarpInterleaving = (MaxThreadsPerBlock == 512);
  static_assert(!(WarpInterleaving && kNumQStages == 1), "warp-interleave does not support stage_q = 1");

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
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<kNumKVStages>{}, Int<kNumMathWarpGroups>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<kNumQStages>{})));

  constexpr static uint32_t CTA_M = shape<0>(TileShape{});
  constexpr static uint32_t CTA_N = shape<1>(TileShape{});
  constexpr static uint32_t CTA_K = shape<2>(TileShape{});

  // ScaleA (k_scales, float per KV row) -- AIU load, no pre-rearrangement
  using DefaultOperandSFA =
      cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, float, false, _1, Int<CTA_M>, false, 0, false>;
  using SmemLayoutAtomSFA = typename DefaultOperandSFA::SmemLayoutAtom;
  using GmemTiledCopySFA = typename DefaultOperandSFA::GmemTiledCopy;
  using SmemLayoutSFA = decltype(tile_to_shape(
      SmemLayoutAtomSFA{},
      make_shape(_1{}, Int<CTA_M>{}, Int<kNumKVStages>{}, Int<kNumMathWarpGroups>{})));

  // ScaleB (weights, ElementWeights per Q head) -- AIU load, no pre-rearrangement
  using DefaultOperandSFB =
      cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementWeights, false, _1, Int<CTA_N>, false, 0, false>;
  using SmemLayoutAtomSFB = typename DefaultOperandSFB::SmemLayoutAtom;
  using GmemTiledCopySFB = typename DefaultOperandSFB::GmemTiledCopy;
  using SmemLayoutSFB = decltype(tile_to_shape(
      SmemLayoutAtomSFB{},
      make_shape(_1{}, Int<CTA_N>{}, Int<kNumQStages>{})));

  using StrideSFA = Stride<Int<CTA_M>, _1>;
  using StrideSFB = Stride<Int<CTA_N>, _1>;

  static_assert(kNumQStages <= 2 && kNumKVStages >= 3, "q_stage <= 2 and kv_stage >= 3");

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
    const uint32_t batch_size;
    const uint64_t logits_stride;
    const uint64_t kv_cache_stride_bytes;
    const uint32_t block_table_stride;
    const uint32_t* context_lens;
    ElementLogits* logits;
    const uint32_t* block_table;
    const uint32_t* schedule_meta;
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

    // // load init scale A/B
    Tensor mSFA_m = make_tensor(make_gmem_ptr(params.k_scales), Shape<_1, Int<CTA_M>>{}, StrideSFA{});
    Tensor mSFA_m_mix = make_mix_tensor_like(mSFA_m);
    Tensor gSFA = local_tile(mSFA_m_mix, Shape<_1, Int<CTA_M>>{}, make_coord(_, 0));

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
    auto [warp_group_idx, local_warp_idx, local_thread_idx] = [&]() {
        if constexpr (kNumMathWarpGroups == 1) {
            return cute::make_tuple(0, warp_idx, thread_idx);
        } else {
            int warp_group_idx = warp_idx / WarpOnGroup;
            int local_warp_idx = warp_idx % WarpOnGroup;
            int local_thread_idx = thread_idx % (32 * WarpOnGroup);
            return cute::make_tuple(warp_group_idx, local_warp_idx, local_thread_idx);
        }
    }();

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // init aiu copy and async copy
    init_aiu_copy();

    // init input tensors
    auto load_inputs = load_init(params);
    Tensor gA = get<0>(load_inputs);
    Tensor gB = get<1>(load_inputs);
    Tensor gSFA = get<2>(load_inputs);
    Tensor gSFB = get<3>(load_inputs);

    // 4D KV smem: (BLOCK_M, BLOCK_K, kNumKVStages, kNumMathWarpGroups)
    // Slice by warp_group_idx to get this warp_group's 3D region
    Tensor sA_full = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutA{}); // 4D
    Tensor sA = sA_full(_, _, _, warp_group_idx); // 3D slice for this warp_group
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)
    // Partition the copying of A and B tiles across the threads (use local_thread_idx)
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(local_thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(local_thread_idx);
    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    // 4D SFA smem: (_1, BLOCK_M, kNumKVStages, kNumMathWarpGroups)
    Tensor sSFA_full = make_tensor(make_smem_ptr(shared_storage.smem_k_scales.data()), SmemLayoutSFA{}); // 4D
    Tensor sSFA = sSFA_full(_, _, _, warp_group_idx); // 3D slice for this warp_group
    Tensor sSFB = make_tensor(make_smem_ptr(shared_storage.smem_weight.data()), SmemLayoutSFB{});

    auto gmem_thr_copy_SFA = gmem_tiled_copy_SFA.get_thread_slice(local_thread_idx);
    auto gmem_thr_copy_SFB = gmem_tiled_copy_SFB.get_thread_slice(local_thread_idx);

    Tensor tSFAgSFA = gmem_thr_copy_SFA.partition_S(gSFA);
    Tensor tSFAsSFA = gmem_thr_copy_SFA.partition_D(sSFA);

    Tensor tSFBgSFB = gmem_thr_copy_SFB.partition_S(gSFB);
    Tensor tSFBsSFB = gmem_thr_copy_SFB.partition_D(sSFB);

    TiledMma tiled_mma;
    Tensor accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));
    auto thr_mma = tiled_mma.get_thread_slice(local_thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                     // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                     // (MMA,MMA_N,MMA_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                  // MMA_K

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(local_warp_idx * 32);
    Tensor tCsA            = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));                  // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                   // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(local_warp_idx * 32);
    Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));                  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    constexpr bool enable_print = false;
    bool thread_print = enable_print && cute::thread(0, 0);

    // Scheduler
    auto scheduler = deep_gemm::PagedMQALogitsScheduler<BLOCK_KV, kNumMathWarpGroups, kNextN>(params.batch_size, blockIdx.x, params.context_lens, params.schedule_meta);
    DG_STATIC_ASSERT(SPLIT_KV % BLOCK_KV == 0, "Unaligned SPLIT_KV");

    ElementLogits weights[kNumHeads / 4];

    clear(accum);
    auto tKgK = tAgA;
    auto tSFKgSFK = tSFAgSFA;
    auto tQgQ = tBgB;
    auto tSFQgSFQ = tSFBgSFB;

    constexpr bool load_kv_scale = sizeof(ElementQK) == 1;
    const auto& lane_idx = get_lane_idx();
    int warp_m_idx = local_warp_idx % WarpOnM;
    int warp_n_idx = local_warp_idx / WarpOnM;
    const auto& warp_offset = warp_m_idx * WARP_M;
    const auto& v_0_offset = lane_idx / 4 + 0;
    const auto& v_1_offset = lane_idx / 4 + 8;
    uint32_t warp_q_idx = warp_n_idx;
    int warp_group_id = warp_idx / 8;

    uint32_t q_idx = scheduler.current_q_idx;
    uint32_t kv_idx_base;
    uint32_t kv_idx_array[kNumKVStages];
    uint32_t smem_pipe_read_q = 0, smem_pipe_read_kv = 0;
    uint32_t smem_pipe_write_q = 0, smem_pipe_write_kv = 0;

    auto load_q_g2s = [&](uint32_t q_idx) {
        auto q_offset = q_idx * BLOCK_N;
        tBgB.data() = tQgQ.data() + q_offset * kHeadDim;
        tSFBgSFB.data() = tSFQgSFQ.data() + q_offset;
        copy_aiu(gmem_tiled_copy_B, tBgB(_,_,_,0), tBsB(_,_,_,smem_pipe_write_q), warp_idx);
        copy_aiu(gmem_tiled_copy_SFB, tSFBgSFB(_,_,_,0), tSFBsSFB(_,_,_,smem_pipe_write_q), warp_idx);
        if (thread_print) {
            printf("  copy_q q_idx = %d, q_offset = %d, stage = %d\n", q_idx, q_offset, smem_pipe_write_q);
        }
        smem_pipe_write_q = (smem_pipe_write_q + 1) % kNumQStages;
    };

    auto load_kv_g2s = [&](uint32_t q_idx, uint32_t kv_idx) {
        auto kv_offset = __ldg(params.block_table + q_idx * params.block_table_stride + kv_idx);
        tAgA.data() = tKgK.data() + kv_offset * params.kv_cache_stride_bytes;
        if constexpr(load_kv_scale) {
            tSFAgSFA.data() = tSFKgSFK.data() + kv_offset * params.kv_cache_stride_bytes / 4;
            copy_aiu<true>(gmem_tiled_copy_A, tAgA(_,_,_,0), tAsA(_,_,_,smem_pipe_write_kv),
                           gmem_tiled_copy_SFA, tSFAgSFA(_,_,_,0), tSFAsSFA(_,_,_,smem_pipe_write_kv), local_warp_idx);
        } else {
            copy_aiu(gmem_tiled_copy_A, tAgA(_,_,_,0), tAsA(_,_,_,smem_pipe_write_kv), local_warp_idx);
        }
        if (thread_print) {
            printf("  copy_k q_idx = %d, kv_idx = %d, kv_offset = %d, wg = %d, stage = %d\n", q_idx, kv_idx, kv_offset, warp_group_idx, smem_pipe_write_kv);
        }
    };

    auto load_q_s2r = [&]() {
        copy(smem_tiled_copy_B, tCsB(_,_,_,smem_pipe_read_q), tCrB_copy_view);
        if (thread_print) {
            printf("    copy q to vreg, q_stage_idx = %d,\n", smem_pipe_read_q);
        }

        // Read weights
        deep_gemm::load_weights_from_smem_ld_shared<ElementWeights, ElementLogits, kNumHeads>(
            sSFB(_,_,smem_pipe_read_q), weights, lane_idx, warp_q_idx);
        smem_pipe_read_q = (smem_pipe_read_q + 1) % kNumQStages;
    };

    auto load_next_qk_g2s = [&](bool load_q) {
        uint32_t q_idx, kv_idx_base, num_kv;
        if (scheduler.fetch_next_task(q_idx, kv_idx_base, num_kv)) {
            if (load_q) load_q_g2s(q_idx);
            uint32_t actual_kv = kv_idx_base + warp_group_idx;
            if (actual_kv < num_kv)
                load_kv_g2s(q_idx, actual_kv);
        }
        kv_idx_array[smem_pipe_write_kv] = kv_idx_base;
        smem_pipe_write_kv = (smem_pipe_write_kv + 1) % kNumKVStages;
        if (thread_print) printf("cp_async commit, write_stage = %d\n", smem_pipe_write_kv);
        cp_async_fence();
    };

    static constexpr uint32_t kMmaIterM = MmaIterM{};
    float scale_kv_array[kMmaIterM * 2];

    auto load_kv_scale_s2r = [&]() {
        float * smem_kv_scales = sSFA(_,_,smem_pipe_read_kv).data().get();
        for (int m = 0; m < kMmaIterM; m++) {
            uint32_t mma_offset = m * InstM;
            scale_kv_array[m * 2    ] = (sizeof(ElementQK) == 2) ? 1 : ld_shared(smem_kv_scales + warp_offset + mma_offset + v_0_offset);
            scale_kv_array[m * 2 + 1] = (sizeof(ElementQK) == 2) ? 1 : ld_shared(smem_kv_scales + warp_offset + mma_offset + v_1_offset);
        }
    };

    // Reduce over heads, scale by per-row KV scale and store logits
    auto epilogue = [&](uint32_t q_idx, uint32_t kv_idx) {
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
                // NOTES: we have redundant writes here, consider more carefully
                auto kv_offset = (q_idx * kNextN + warp_q_idx) * params.logits_stride + kv_idx * BLOCK_KV;
                params.logits[kv_offset + warp_offset + mma_offset + v_0_offset] = v_0 * scale_kv_0;
                params.logits[kv_offset + warp_offset + mma_offset + v_1_offset] = v_1 * scale_kv_1;
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

                // Store into the global memory
                auto kv_offset = (q_idx * kNextN + warp_q_idx) * params.logits_stride + kv_idx * BLOCK_KV;
                params.logits[kv_offset + warp_offset + mma_offset + v_0_offset] = __low2bfloat16(packed);
                params.logits[kv_offset + warp_offset + mma_offset + v_1_offset] = __high2bfloat16(packed);
            }
        }
    };
    // load qk
    for (int i = 0; i < kNumKVStages - 1; i++) {
        load_next_qk_g2s(i == 0);
    }
    kv_idx_base = kv_idx_array[0];
    // wait AIU Q and first K
    cp_async_wait<kNumKVStages - 2>();
    __syncthreads();

    deep_gemm::warp_interleave_start<WarpInterleaving>(warp_group_id, MaxThreadsPerBlock);

    auto handle_q_change = [&]() {
        if constexpr(kNumQStages > 1) {
            if (scheduler.exist_q_idx(q_idx + 1)) {
                load_q_g2s(q_idx + 1);
            }
            cp_async_fence();
        }
        load_q_s2r();
        if constexpr(kNumQStages == 1) { // not supported with warp interleaving
            __syncthreads();
            if (scheduler.exist_q_idx(q_idx + 1)) {
                load_q_g2s(q_idx + 1);
            }
            cp_async_fence();
        }
    };

    // Make `first` a compile-time constant via cute::true_type / cute::false_type
    // so the compiler can fold the first-iteration branch and avoid the runtime
    // check on the first iteration.
    auto mainloop = [&](auto first_ic) {
        constexpr bool first = decltype(first_ic)::value;
        if (thread_print) {
            printf("q_idx = %d, kv_idx_base = %d, wg = %d, read_stage = %d\n", q_idx, kv_idx_base, warp_group_idx, smem_pipe_read_kv);
        }

        deep_gemm::warp_interleave_sync<WarpInterleaving>(warp_group_id, MaxThreadsPerBlock);

        // Read weights if current Q changes. On the first iteration always load Q;
        // afterwards only when kv_idx_base == 0 (first group of new q).
        if constexpr (first) {
            handle_q_change();
        } else {
            if (kv_idx_base == 0) {
                handle_q_change();
            }
        }

        // Issue next KV load (overlap with current compute)
        load_next_qk_g2s(false);

        // s2r KV + compute GEMM from current read stage
        constexpr int K_BLOCK_MAX = size<2>(tCrA);
        uint32_t actual_kv = kv_idx_base + warp_group_idx;
        for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
            copy(smem_tiled_copy_A, tCsA(_,_,k_block,smem_pipe_read_kv), tCrA_copy_view(_,_,k_block));
            cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrB(_,_,k_block), accum);
        });

        deep_gemm::warp_interleave_arrive<WarpInterleaving>(warp_group_id, MaxThreadsPerBlock);

        load_kv_scale_s2r();
        epilogue(q_idx, actual_kv);

        cp_async_wait<kNumKVStages - 2>();
        if constexpr (!WarpInterleaving) {
            __syncthreads();
        }
        // move to next kv_block
        smem_pipe_read_kv = (smem_pipe_read_kv + 1) % kNumKVStages;
        kv_idx_base = kv_idx_array[smem_pipe_read_kv];
        if (kv_idx_base == 0) q_idx++;

        clear(accum);
    };

    if (!scheduler.is_last_task(q_idx, kv_idx_base)) {
        mainloop(cute::true_type{});
    }
    while (!scheduler.is_last_task(q_idx, kv_idx_base)) {
        mainloop(cute::false_type{});
    }

    deep_gemm::warp_interleave_end<WarpInterleaving>(warp_group_id, MaxThreadsPerBlock);

    cp_async_wait<0>();
    __syncthreads();

  }

};

} // namespace cutlass::gemm::kernel


namespace deep_gemm {

template <typename ElementQK, typename ElementAcc, typename ElementLogits, typename ElementWeights,
          uint32_t kNextN, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV, uint32_t WARP_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV>
class PagedAttention {

public:
    PagedAttention() = default;

    static uint32_t generate_id() {
        static uint32_t id = 0;
        return ++id;
    }

    static void run(const ElementQK * ptr_q,
                    const ElementQK * ptr_k,
                    const float * k_scales,
                    const ElementWeights * weights,
                    const uint32_t batch_size,
                    const uint64_t logits_stride, const uint64_t kv_cache_stride_bytes, const uint32_t block_table_stride,
                    const uint32_t* context_lens, ElementLogits* logits,
                    const uint32_t* block_table, const uint32_t* schedule_meta,
                    hggcStream_t stream, int num_sms, int num_blocks) {

        using AttnKernel = cutlass::gemm::kernel::PPUPagedMqaLogits<ElementQK, ElementAcc, ElementLogits, ElementWeights, kNextN, kNumHeads, kHeadDim, BLOCK_KV, WARP_KV, kNumQStages, kNumKVStages, SPLIT_KV>;

        static constexpr int BLOCK_M = AttnKernel::BLOCK_M;
        static constexpr int BLOCK_N = AttnKernel::BLOCK_N;
        static constexpr int BLOCK_K = AttnKernel::BLOCK_K;
        static constexpr int WARP_M  = AttnKernel::WARP_M ;
        static constexpr int WARP_N  = AttnKernel::WARP_N ;
        static constexpr int MaxThreadsPerBlock = AttnKernel::MaxThreadsPerBlock;

        int max_blocks_per_cu = compute_occupancy_for_kernel<AttnKernel>();

        typename AttnKernel::Arguments arguments{ptr_q, ptr_k, k_scales, weights, batch_size, logits_stride, kv_cache_stride_bytes, block_table_stride,
                                                 context_lens, logits, block_table, schedule_meta};
        auto params = arguments;
        dim3 const block(MaxThreadsPerBlock, 1, 1);
        dim3 const grid(num_blocks, 1, 1);
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

            dg_prof_params.set_paged_mqa_logits_params(data_type_str, batch_size, kNextN, kNumHeads, kHeadDim, reinterpret_cast<int*>(const_cast<uint32_t*>(context_lens)), stream);
            if constexpr (cute::is_same_v<ElementLogits, __ppu_bfloat16>) {
                dg_prof_params.add_params("logits_dtype", std::string("bf16"));
            }
            if constexpr (cute::is_same_v<ElementWeights, __ppu_bfloat16>) {
                dg_prof_params.add_params("weights_dtype", std::string("bf16"));
            }
        }

        char *pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            hggcFuncAttributes attr;
            hggcFuncGetAttributes(&attr, cutlass::device_kernel<AttnKernel>);

            printf("[paged_mqa_logits:]\n");
            printf("kNumHeads:%d, kHeadDim:%d, kNextN:%d, BLOCK_KV:%d, SPLIT_KV:%d\n",
                kNumHeads, kHeadDim, kNextN, BLOCK_KV, SPLIT_KV);

            printf("ThreadblockShape[%d, %d], WarpShape[%d, %d], kNumQStages:%d, kNumKVStages:%d\n",
                BLOCK_M, BLOCK_N, WARP_M, WARP_N, kNumQStages, kNumKVStages);

            printf("num_sms:%d, max_blocks_per_cu:%d, threadblock_count:%d, num_threads:%d\n", num_sms, max_blocks_per_cu, num_blocks, block.x);
            printf("smem_size:%d, vreg:%d, stack:%d\n", smem_size_kernel, int(attr.numRegs), int(attr.localSizeBytes));
            printf("weights_bf16:%s\n", cute::is_same_v<ElementWeights, __ppu_bfloat16> ? "true" : "false");
            if (num_sms * max_blocks_per_cu  != num_blocks) {
                printf("Warning: num_blocks(%d) should equal to num_sms(%d) * max_blocks_per_cu(%d) = %d\n", num_blocks, num_sms, max_blocks_per_cu, num_sms * max_blocks_per_cu);
            }
        }

        ProfilingInterface::Instance().instrument(true, dg_prof_params);
        cutlass::device_kernel<AttnKernel><<<grid, block, smem_size_kernel, stream>>>(params);
        ProfilingInterface::Instance().instrument(false, dg_prof_params);
    }
};

};  // namespace deep_gemm
