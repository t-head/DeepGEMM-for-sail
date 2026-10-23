#pragma once
#include <deep_gemm/tix/ld_st.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/impls/mqa_logits_utils.cuh>

namespace cutlass::gemm::kernel {

// ============================================================================
// FP4 (Non-Paged) MQA Logits Kernel
// ============================================================================
// Key differences from PPUMqaLogits (FP8):
//   - ElementQK is uint8_t (packed FP4, 2 elements per byte)
//   - kHeadDim is the *packed* dimension (original_head_dim / 2 = 64)
//   - Additional q_sf (Q scale factor, uint32_t packed from uint8 e8m0) parameter
//   - k_sf is uint32_t (packed e8m0), not float32 per-token scale
//   - Q/K/SFA/SFB/weights all use AIU load
//   - Output dtype controlled by ElementLogits (float or bfloat16)
//   - MMA atom: PPU10500_16x16x64_F32F4F4F32_TN (6-operand)
//   - Non-paged scheduling using cu_seq_len_k_start / cu_seq_len_k_end
// ============================================================================

template <typename ElementQK, typename ElementAcc, typename ElementLogits, typename ElementWeights,
          int kNumHeads, int kHeadDim, int BLOCK_QH,
          int BLOCK_KV, int WARP_QH, int WARP_KV, int kNumQStages, int kNumKVStages,
          typename StrideKType = uint32_t,
          bool kIsCompressedLogits = false,
          uint32_t kScaleMode = deep_gemm::kScaleModeWeights,
          int kMBlockGroup = cute::is_same_v<ElementLogits, float> ? 0 : 1,
          bool kQOwnerEpilogue = false, bool kBalanceOutputTiles = false>
class PPUMqaLogitsFP4 {
public:
    static_assert(cute::is_same_v<ElementQK, uint8_t>, "FP4 MQA logits requires uint8_t ElementQK");
    static_assert(kHeadDim == 64, "FP4 packed head_dim must be 64 (original 128 / 2)");
    static_assert(cute::is_same_v<StrideKType, uint32_t> || cute::is_same_v<StrideKType, uint64_t>,
                  "StrideKType must be uint32_t or uint64_t");

    static constexpr int BLOCK_M = BLOCK_KV;
    static constexpr int BLOCK_N = BLOCK_QH;
    static constexpr int BLOCK_K = kHeadDim; // 64 (packed)
    static constexpr int WARP_M = WARP_KV;
    static constexpr int WARP_N = WARP_QH;
    static constexpr int BLOCK_Q = BLOCK_QH / kNumHeads;
    static constexpr int WARP_Q = WARP_QH / kNumHeads;

    using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
    static constexpr int WarpOnM = BLOCK_M / WARP_M;
    static constexpr int WarpOnN = BLOCK_N / WARP_N;

    static constexpr float kInvSqrtHeadDim = 0x1.6a09e6p-4f;


    // ── MMA: F32F4F4F32 (16×16×64) ────────────────────────────────────
    using MmaInst = PPU0015_16x16x64_F32F4F4F32_TN;
    using MmaK_type = _32;

    static constexpr int InstM = 16;
    static constexpr int InstN = 16;
    static constexpr int MmaIterM = WARP_M / InstM;
    static constexpr int MmaIterN = WARP_N / InstN;
    using PermutationMNK =
        Tile<Layout<Shape<Int<InstM>, Int<WarpOnM>, Int<MmaIterM>>, Stride<_1, Int<WARP_M>, Int<InstM>>>,
             Layout<Shape<Int<InstN>, Int<WarpOnN>, Int<MmaIterN>>, Stride<_1, Int<WARP_N>, Int<InstN>>>, MmaK_type>;
    using TiledMma = TiledMMA<MMA_Atom<MmaInst>, Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, _1>>, PermutationMNK>;

    static constexpr int NumThreadsPerCTA = size(TiledMma{});
    static constexpr int MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));
    static constexpr int MinBlocksPerMultiprocessor = 1;
    // G=0 selects the original FP32 coarse schedule; positive G groups M blocks.
    static_assert(kMBlockGroup >= 0, "M-block group must be nonnegative");
    static_assert(cute::is_same_v<ElementLogits, float> || kMBlockGroup > 0,
                  "BF16 M-block group must be positive");
    static constexpr bool WarpInterleaving = NumThreadsPerCTA == 512;
    // Output ownership and task distribution are independent of M-block grouping.
    static constexpr bool kIsAvg = kScaleMode != deep_gemm::kScaleModeWeights;
    static_assert(!kQOwnerEpilogue ||
                  (kIsAvg && kNumHeads == 4 && WARP_Q == 4 && MmaIterN == 1),
                  "Q-owner requires H4 AVG with WARP_Q=4");
    static_assert(!kBalanceOutputTiles ||
                  (kIsAvg && kNumHeads == 4 && kNumQStages == 1),
                  "balanced output tiles require H4 AVG with one Q stage");


    // ── AIU operands for Q/K ────────────────────────────────────────────
    using DefaultOperandA =
        cutlass::gemm::config::DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, ElementQK, false, Int<BLOCK_M>, Int<BLOCK_K>, false>;
    using DefaultOperandB =
        cutlass::gemm::config::DefaultGemm_AIU_Operand<cutlass::arch::PPU0015,ElementQK, false, Int<BLOCK_N>, Int<BLOCK_K>, true>;
    // A (K)
    using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom;
    using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
    using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;
    // B (Q)
    using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom;
    using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
    using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

    static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");
    static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");

    static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");
    static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");

    using SmemLayoutA =
        decltype(tile_to_shape(SmemLayoutAtomA{}, Shape<Int<BLOCK_M>, Int<BLOCK_K>, Int<kNumKVStages>>{}));
    using SmemLayoutB =
        decltype(tile_to_shape(SmemLayoutAtomB{}, Shape<Int<BLOCK_N>, Int<BLOCK_K>, Int<kNumQStages>>{}));

    // ── SFA / SFB / Weight AIU operands ────────────────────────────────
    // SFA (k_sf): one uint32_t per KV row
    using DefaultOperandSFA =
        cutlass::gemm::config::DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, uint32_t, false, _1, Int<BLOCK_M>, false, 0, false>;
    using SmemLayoutAtomSFA = typename DefaultOperandSFA::SmemLayoutAtom;
    using GmemTiledCopySFA = typename DefaultOperandSFA::GmemTiledCopy;
    using SmemLayoutSFA =
        decltype(tile_to_shape(SmemLayoutAtomSFA{}, make_shape(_1{}, Int<BLOCK_M>{}, Int<kNumKVStages>{})));

    // SFB (q_sf / weights): one uint32_t/float32 per Q head
    using DefaultOperandSFB =
        cutlass::gemm::config::DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, uint32_t, false, _1, Int<BLOCK_N>, false, 0, false>;
    using SmemLayoutAtomSFB = typename DefaultOperandSFB::SmemLayoutAtom;
    using GmemTiledCopySFB = typename DefaultOperandSFB::GmemTiledCopy;
    using SmemLayoutSFB =
        decltype(tile_to_shape(SmemLayoutAtomSFB{}, make_shape(_1{}, Int<BLOCK_N>{}, Int<kNumQStages>{})));

    using DefaultOperandWeight =
        cutlass::gemm::config::DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, ElementWeights, false, _1, Int<BLOCK_N>, false, 0, false>;
    using SmemLayoutAtomWeight = typename DefaultOperandWeight::SmemLayoutAtom;
    using GmemTiledCopyWeight = typename DefaultOperandWeight::GmemTiledCopy;
    using SmemLayoutWeight =
        decltype(tile_to_shape(SmemLayoutAtomWeight{}, make_shape(_1{}, Int<BLOCK_N>{}, Int<kNumQStages>{})));

    // SFA/SFB/Weight s2r layouts
    MQA_DEFINE_FP4_S2R_LAYOUTS(WarpOnM, MmaIterM, kNumKVStages, BLOCK_M,
                               WarpOnN, MmaIterN, kNumQStages, BLOCK_N)
    // WeightCopyAtomType and SmemTiledCopyWeights are kernel-specific (depend on ElementWeights)
    using WeightCopyAtomType = cute::conditional_t<cute::is_same_v<ElementWeights, __ppu_bfloat16>, uint16_t, uint64_t>;
    using SmemTiledCopyWeights = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<WeightCopyAtomType>, ElementWeights>{},
        Layout<Shape<Shape<_4, Int<WarpOnN>>, _1>, Stride<Stride<_1, _4>, _1>>{}, Layout<Shape<_2, _1>>{}));

    // ── Shared memory ──────────────────────────────────────────────────
    struct SharedStorage {
        cute::array_aligned<ElementQK, cute::cosize_v<SmemLayoutA>> smem_k;       // K packed FP4
        cute::array_aligned<ElementQK, cute::cosize_v<SmemLayoutB>> smem_q;       // Q packed FP4
        cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutSFA>> smem_k_sf;   // K e8m0 scales (uint32_t)
        cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutSFB>> smem_q_sf;   // Q e8m0 scales (uint32_t)
        cute::array_aligned<ElementWeights, cute::cosize_v<SmemLayoutWeight>> smem_weight; // weights
    };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    // ── Device side arguments ──────────────────────────────────────────
    struct Arguments {
        const ElementQK* ptr_q;
        const uint32_t* q_sf; // Q scale factor (packed e8m0, uint32_t)
        const ElementQK* ptr_k;
        const uint32_t* k_sf; // K scale factor (packed e8m0, uint32_t)
        const ElementWeights* weights;
        int* cu_seq_len_k_start;
        int* cu_seq_len_k_end;
        ElementLogits* logits;
        const int seq_len_q;
        const int seq_len_k;
        const StrideKType stride_k;
    };

    using Params = Arguments;

    // ── AIU copy objects (compile-time layout → default-constructed) ────
    GmemTiledCopyA gmem_tiled_copy_A;
    GmemTiledCopyB gmem_tiled_copy_B;
    GmemTiledCopySFA gmem_tiled_copy_SFA;
    GmemTiledCopySFB gmem_tiled_copy_SFB;
    GmemTiledCopyWeight gmem_tiled_copy_weight;

    using StrideAB = Stride<Int<BLOCK_K>, _1>;
    using StrideSFA = Stride<Int<BLOCK_M>, _1>;
    using StrideSFB = Stride<Int<BLOCK_N>, _1>;

    CUTLASS_DEVICE void init_aiu_copy() {
        using TilerA = typename GmemTiledCopyA::Tiler_MN;
        using TilerB = typename GmemTiledCopyB::Tiler_MN;
        using TilerSFA = typename GmemTiledCopySFA::Tiler_MN;
        using TilerSFB = typename GmemTiledCopySFB::Tiler_MN;
        using TilerWeight = typename GmemTiledCopyWeight::Tiler_MN;

        gmem_tiled_copy_A.desc_.template init<ElementQK, false, get<0>(TilerA{}), get<1>(TilerA{})>(
            nullptr, BLOCK_M, BLOCK_K, StrideAB{});
        gmem_tiled_copy_B.desc_.template init<ElementQK, false, get<0>(TilerB{}), get<1>(TilerB{})>(
            nullptr, BLOCK_N, BLOCK_K, StrideAB{});
        gmem_tiled_copy_SFA.desc_.template init<uint32_t, false, get<0>(TilerSFA{}), get<1>(TilerSFA{})>(
            nullptr, 1, BLOCK_M, StrideSFA{});
        gmem_tiled_copy_SFB.desc_.template init<uint32_t, false, get<0>(TilerSFB{}), get<1>(TilerSFB{})>(
            nullptr, 1, BLOCK_N, StrideSFB{});
        gmem_tiled_copy_weight.desc_.template init<ElementWeights, false, get<0>(TilerWeight{}), get<1>(TilerWeight{})>(
            nullptr, 1, BLOCK_N, StrideSFB{});
    };

    CUTLASS_DEVICE auto load_init(Params const& params) {
        // K (A) — packed FP4, row-major [BLOCK_KV, kHeadDim]
        Tensor mA_mk = make_tensor(make_gmem_ptr(params.ptr_k), Shape<Int<BLOCK_M>, Int<BLOCK_K>>{}, StrideAB{});
        Tensor mA_mk_mix = make_mix_tensor_like(mA_mk);
        Tensor gA = local_tile(mA_mk_mix, TileShape{}, make_coord(0, 0, _), Step<_1, X, _1>{});

        // Q (B) — packed FP4, column-major-like [BLOCK_QH, kHeadDim]
        Tensor mB_nk = make_tensor(make_gmem_ptr(params.ptr_q), Shape<Int<BLOCK_N>, Int<BLOCK_K>>{}, StrideAB{});
        Tensor mB_nk_mix = make_mix_tensor_like(mB_nk);
        Tensor gB = local_tile(mB_nk_mix, TileShape{}, make_coord(0, _, 0), Step<X, _1, _1>{});

        // k_sf (SFA)
        Tensor mSFA_m = make_tensor(make_gmem_ptr(params.k_sf), Shape<_1, Int<BLOCK_M>>{}, StrideSFA{});
        Tensor mSFA_m_mix = make_mix_tensor_like(mSFA_m);
        Tensor gSFA = local_tile(mSFA_m_mix, Shape<_1, Int<BLOCK_M>>{}, make_coord(_, 0));

        // q_sf (SFB)
        Tensor mSFB_n = make_tensor(make_gmem_ptr(params.q_sf), Shape<_1, Int<BLOCK_N>>{}, StrideSFB{});
        Tensor mSFB_n_mix = make_mix_tensor_like(mSFB_n);
        Tensor gSFB = local_tile(mSFB_n_mix, Shape<_1, Int<BLOCK_N>>{}, make_coord(0, _));

        // weights
        Tensor mW_n = make_tensor(make_gmem_ptr(params.weights), Shape<_1, Int<BLOCK_N>>{}, StrideSFB{});
        Tensor mW_n_mix = make_mix_tensor_like(mW_n);
        Tensor gW = local_tile(mW_n_mix, Shape<_1, Int<BLOCK_N>>{}, make_coord(0, _));

        return cute::make_tuple(gA, gB, gSFA, gSFB, gW);
    }

    CUTLASS_DEVICE
    void operator()(Params const& params, char* smem_buf) {
#if __HGGC_ARCH__ == 100
        return;
#endif
        int warp_idx = canonical_warp_idx_sync();
        int thread_idx = int(threadIdx.x);
        int warp_m_idx = warp_idx % WarpOnM;
        int warp_n_idx = warp_idx / WarpOnM;
        int lane_idx = deep_gemm::tix::get_lane_idx();

        // ── Shared memory ─────────────────────────────────────────────────
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        // ── Init AIU copy ─────────────────────────────────────────────────
        init_aiu_copy();

        // ── Init input tensors ────────────────────────────────────────────
        auto [gA, gB, gSFA, gSFB, gW] = load_init(params);

        // ── Smem tensors ──────────────────────────────────────────────────
        Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutB{});

        auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
        auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
        Tensor tAgA = gmem_thr_copy_A.partition_S(gA);
        Tensor tAsA = gmem_thr_copy_A.partition_D(sA);
        Tensor tBgB = gmem_thr_copy_B.partition_S(gB);
        Tensor tBsB = gmem_thr_copy_B.partition_D(sB);

        // Scale / weight smem tensors
        Tensor sSFA = make_tensor(make_smem_ptr(shared_storage.smem_k_sf.data()), SmemLayoutSFA{});
        Tensor sSFB = make_tensor(make_smem_ptr(shared_storage.smem_q_sf.data()), SmemLayoutSFB{});
        Tensor sW = make_tensor(make_smem_ptr(shared_storage.smem_weight.data()), SmemLayoutWeight{});

        auto gmem_thr_copy_SFA = gmem_tiled_copy_SFA.get_thread_slice(thread_idx);
        auto gmem_thr_copy_SFB = gmem_tiled_copy_SFB.get_thread_slice(thread_idx);
        auto gmem_thr_copy_weight = gmem_tiled_copy_weight.get_thread_slice(thread_idx);

        Tensor tSFAgSFA = gmem_thr_copy_SFA.partition_S(gSFA);
        Tensor tSFAsSFA = gmem_thr_copy_SFA.partition_D(sSFA);
        Tensor tSFBgSFB = gmem_thr_copy_SFB.partition_S(gSFB);
        Tensor tSFBsSFB = gmem_thr_copy_SFB.partition_D(sSFB);
        Tensor tWgW = gmem_thr_copy_weight.partition_S(gW);
        Tensor tWsW = gmem_thr_copy_weight.partition_D(sW);

        // ── MMA setup ─────────────────────────────────────────────────────
        TiledMma tiled_mma;
        Tensor accum = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
        Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));

        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));

        // S2R copies for A / B
        auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
        auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(warp_idx * 32);
        Tensor tCsA = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));
        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
        CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));

        auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
        auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(warp_idx * 32);
        Tensor tCsB = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
        CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));

        // ── SFA s2r setup ─────────────────────────────────────────────────
        Tensor sSFAUint16 = recast<uint16_t>(sSFA);
        Tensor sSFATrans = make_tensor(sSFAUint16.data(), sSFATransLayout{});
        SmemTiledCopySFA smem_tiled_copy_SFA;
        auto smem_thr_copy_SFA = smem_tiled_copy_SFA.get_slice(warp_m_idx * 32 + lane_idx);
        Tensor tCsSFA = smem_thr_copy_SFA.partition_S(sSFATrans);
        Tensor tCrSFA_stage = smem_thr_copy_SFA.partition_D(sSFATrans(_, _, 0));
        using SFARegType = decltype(make_fragment_like(tCrSFA_stage));
        SFARegType tCrSFA_copy_view;

        // ── SFB s2r setup ─────────────────────────────────────────────────
        Tensor sSFBUint16 = recast<uint16_t>(sSFB);
        Tensor sSFBTrans = make_tensor(sSFBUint16.data(), sSFBTransLayout{});
        SmemTiledCopySFB smem_tiled_copy_SFB;
        auto smem_thr_copy_SFB = smem_tiled_copy_SFB.get_slice(warp_n_idx * 32 + lane_idx);
        Tensor tCsSFB = smem_thr_copy_SFB.partition_S(sSFBTrans);
        Tensor tCrSFB_stage = smem_thr_copy_SFB.partition_D(sSFBTrans(_, _, 0));
        using SFBRegType = decltype(make_fragment_like(tCrSFB_stage));
        SFBRegType tCrSFB_copy_view;

        // FP4 scale as uint32_t for the 6-operand gemm
        Tensor tCrSFA = recast<uint32_t>(tCrSFA_copy_view);
        Tensor tCrSFB = recast<uint32_t>(tCrSFB_copy_view);

        // ── Weights s2r setup ─────────────────────────────────────────────
        constexpr int elem_weights = WARP_QH / 4;
        Tensor sW_copy = make_tensor(sW.data(), sWCopyLayout{});
        SmemTiledCopyWeights smem_tiled_copy_weights;
        auto smem_thr_copy_weights = smem_tiled_copy_weights.get_slice(warp_n_idx * 4 + lane_idx % 4);
        Tensor tCsW = smem_thr_copy_weights.partition_S(sW_copy);
        Tensor tCrW_copy_view = make_tensor_like(tCsW(_, _, _, 0));
        using WeightRegLayout = Layout<Shape<Shape<_2, _2, _2>, Int<MmaIterN>>, Stride<Stride<_1, _0, _2>, _4>>;
        // For bf16 weights: tCrW points directly to tCrW_copy_view's data (no separate weights array)
        // For float weights: tCrW points to weights array (populated by cvt in load_q_s2r)
        ElementLogits weights[elem_weights];
        // Avg variants
        // FP32 Q-owner combines the two output factors once per Q load.
        // Arithmetic remains FP32; reassociation may change the final rounding.
        float scale_q = 1.0f;
        if constexpr (kScaleMode == deep_gemm::kScaleModeQScalar) {
            scale_q = __ldg(params.weights);
            if constexpr (cute::is_same_v<ElementLogits, float> && kQOwnerEpilogue)
                scale_q *= kInvSqrtHeadDim;
        }
        // Only kScaleModeQRow reads scale_q_row
        float scale_q_row[WARP_Q];

        Tensor tCrW = make_tensor(static_cast<ElementLogits*>(weights), WeightRegLayout{});

        // ── Block scheduler (non-paged) ───────────────────────────────────
        // The template permits balancing; every CTA probes the same global
        // rows and selects the same scheduler. Sampling is a performance
        // heuristic, not a proof that every row has a long/full window.
        bool balance_active = false;
        if constexpr (kBalanceOutputTiles) {
            if (params.seq_len_q <= 0 || params.seq_len_k <= 0) return;
            const uint32_t probe_q[3] = {
                0u, static_cast<uint32_t>(params.seq_len_q / 2),
                static_cast<uint32_t>(params.seq_len_q - 1)};
            balance_active = true;
            #pragma unroll
            for (int probe = 0; probe < 3; ++probe) {
                const int probe_start = min(max(__ldg(params.cu_seq_len_k_start + probe_q[probe]), 0), params.seq_len_k);
                const int probe_end = min(max(__ldg(params.cu_seq_len_k_end + probe_q[probe]), 0), params.seq_len_k);
                const uint32_t probe_span = probe_end > probe_start
                    ? static_cast<uint32_t>(probe_end - probe_start) : 0u;
                // Widen before multiplying: exact span >= 3*K/4, including
                // K values not divisible by four, without unsigned underflow.
                balance_active &= uint64_t(probe_span) * 4 >= uint64_t(params.seq_len_k) * 3;
            }
        }
        const auto& num_q_blocks = deep_gemm::math::ceil_div(params.seq_len_q, BLOCK_Q);
        uint32_t block_q_idx = blockIdx.x, q_iter_idx = 0;
        uint32_t full_kv_blocks = 0;
        uint64_t output_tile_idx = 0, output_tile_end = 0;
        if constexpr (kBalanceOutputTiles) {
            if (balance_active) {
                // Divide independent output tiles, never the dot-product dimension.
                full_kv_blocks = deep_gemm::math::ceil_div(params.seq_len_k, BLOCK_KV);
                const uint64_t full_tile_count = uint64_t(num_q_blocks) * full_kv_blocks;
                output_tile_idx = full_tile_count * blockIdx.x / gridDim.x;
                output_tile_end = full_tile_count * (uint64_t(blockIdx.x) + 1) / gridDim.x;
            }
        }
        // Per-token KV range arrays for compressed logits mode (dead code when kIsCompressedLogits=false)
        uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];

        // ── Variables ─────────────────────────────────────────────────────
        clear(accum);
        auto tKgK = tAgA;
        auto tSFKgSFK = tSFAgSFA;
        auto tQgQ = tBgB;
        auto tSFQgSFQ = tSFBgSFB;
        auto tWgW_base = tWgW;

        const auto& warp_offset = warp_m_idx * WARP_M;
        const auto& v_0_offset = lane_idx / 4 + 0;
        const auto& v_1_offset = lane_idx / 4 + 8;
        uint32_t warp_q_idx = warp_idx / WarpOnM;
        int warp_group_id = warp_idx / 8;
        // Lane-parity kv row half used by the kWq4 epilogue split
        const uint32_t r_mine = lane_idx & 1;
        const uint32_t v_offset_mine = r_mine ? v_1_offset : v_0_offset;

        auto load_q_g2s = [&](uint32_t block_q_idx, uint32_t q_stage_idx) {
            int residual_n = (params.seq_len_q - block_q_idx * BLOCK_Q) * kNumHeads;
            gmem_tiled_copy_B.desc_.dim_h = residual_n;
            gmem_tiled_copy_SFB.desc_.dim_w = residual_n;
            copy_aiu(gmem_tiled_copy_B, tBgB(_, _, _, 0), tBsB(_, _, _, q_stage_idx), warp_idx);
            if constexpr (kScaleMode == deep_gemm::kScaleModeWeights) {
                gmem_tiled_copy_weight.desc_.dim_w = residual_n;
                copy_aiu<true>(gmem_tiled_copy_SFB, tSFBgSFB(_, _, _, 0), tSFBsSFB(_, _, _, q_stage_idx), gmem_tiled_copy_weight,
                               tWgW(_, _, _, 0), tWsW(_, _, _, q_stage_idx), warp_idx);
            } else {
                copy_aiu(gmem_tiled_copy_SFB, tSFBgSFB(_, _, _, 0), tSFBsSFB(_, _, _, q_stage_idx), warp_idx);
            }
        };

        auto load_kv_g2s = [&](uint32_t kv_start, uint32_t kv_end, uint32_t kv_block_idx, uint32_t smem_stage) {
            int residual_m = kv_end - (kv_start + kv_block_idx * BLOCK_KV);
            gmem_tiled_copy_A.desc_.dim_h = residual_m;
            gmem_tiled_copy_SFA.desc_.dim_w = residual_m;
            copy_aiu<true>(gmem_tiled_copy_A, tAgA(_, _, _, 0), tAsA(_, _, _, smem_stage), gmem_tiled_copy_SFA,
                           tSFAgSFA(_, _, _, 0), tSFAsSFA(_, _, _, smem_stage), warp_idx);
            tAgA.data() = tAgA.data() + BLOCK_KV * BLOCK_K;
            tSFAgSFA.data() = tSFAgSFA.data() + BLOCK_KV;
        };

        auto load_q_s2r = [&](uint32_t q_stage_idx) {
            copy(smem_tiled_copy_B, tCsB(_, _, _, q_stage_idx), tCrB_copy_view);
            copy(smem_tiled_copy_SFB, tCsSFB(_, _, _, q_stage_idx), tCrSFB_copy_view);
            if constexpr (kScaleMode == deep_gemm::kScaleModeWeights) {
                copy(smem_tiled_copy_weights, tCsW(_, _, _, q_stage_idx), tCrW_copy_view);
                deep_gemm::load_weights_from_copy_view<ElementWeights, ElementLogits>(
                    tCrW_copy_view, weights, elem_weights);
            } else if constexpr (kScaleMode == deep_gemm::kScaleModeQRow) {
                // The weights slot carries the per-Q-row float scale; clamped for partial tail blocks
                #pragma unroll
                for (uint32_t i = 0; i < WARP_Q; ++i) {
                    const uint32_t q_idx = min(block_q_idx * BLOCK_Q + warp_q_idx * WARP_Q + i, params.seq_len_q - 1);
                    scale_q_row[i] = __ldg(params.weights + q_idx);
                    if constexpr (cute::is_same_v<ElementLogits, float> && kQOwnerEpilogue)
                        scale_q_row[i] *= kInvSqrtHeadDim;
                }
            }
        };

        auto load_kv_s2r = [&](int k_block, uint32_t kv_stage_idx) {
            copy(smem_tiled_copy_A, tCsA(_, _, k_block, kv_stage_idx), tCrA_copy_view(_, _, k_block));
            copy(smem_tiled_copy_SFA, tCsSFA(_, _, k_block, kv_stage_idx), tCrSFA_copy_view(_, _, k_block));
        };

        auto load_kv_a_s2r_mblock = [&](int k_block, uint32_t kv_stage_idx, int m_block) {
            copy(smem_tiled_copy_A, tCsA(_, m_block, k_block, kv_stage_idx), tCrA_copy_view(_, m_block, k_block));
        };

        auto load_kv_sfa_s2r_group = [&](int k_block, uint32_t kv_stage_idx, int scale_group) {
            copy(smem_tiled_copy_SFA, tCsSFA(_, scale_group, k_block, kv_stage_idx), tCrSFA_copy_view(_, scale_group, k_block));
        };

        auto epilogue_mblock = [&](uint32_t kv_start, uint32_t kv_block_idx, int m_block) {
            // Reduce over heads and store logits
            static constexpr int kNumAccumPerMma = size<0>(accum);
            CUTE_STATIC_ASSERT(kNumHeads % 4 == 0);
            const auto& kv_offset = kv_start + kv_block_idx * BLOCK_KV + warp_offset;
            {
                int m = m_block;
                int mma_offset = m * InstM;
                auto logits_q_offset = (block_q_idx * BLOCK_Q + warp_q_idx) * params.stride_k;
                auto logits_kv_offset = kv_offset + mma_offset;

                if constexpr (cute::is_same_v<ElementLogits, float>) {
                    float v_0, v_1;
                    if constexpr (kScaleMode == deep_gemm::kScaleModeWeights) {
                        deep_gemm::float_epilogue_reduce_tCrW(accum, m, tCrW, v_0, v_1);
                        deep_gemm::shfl_xor_reduce_2(v_0, v_1);
                        if constexpr (kIsCompressedLogits) {
                            const uint32_t rel_kv = static_cast<uint32_t>(logits_kv_offset) - seq_k_start[warp_q_idx];
                            const uint32_t len = seq_k_end[warp_q_idx] - seq_k_start[warp_q_idx];
                            if (rel_kv + v_0_offset < len)
                                params.logits[logits_q_offset + rel_kv + v_0_offset] = v_0;
                            if (rel_kv + v_1_offset < len)
                                params.logits[logits_q_offset + rel_kv + v_1_offset] = v_1;
                        } else {
                            params.logits[logits_q_offset + logits_kv_offset + v_0_offset] = v_0;
                            params.logits[logits_q_offset + logits_kv_offset + v_1_offset] = v_1;
                        }
                    } else {
                        if constexpr (kQOwnerEpilogue) {
                            // Adjacent lanes own
                            // different Q tokens, each writing both KV row halves.
                            // Exactly two xor shuffles, unchanged head-pair grouping.
                            CUTE_STATIC_ASSERT(MmaIterN == 1);
                            CUTE_STATIC_ASSERT(WARP_Q == 4 && kNumHeads == 4);
                            float sums[2][2];
                            deep_gemm::float_epilogue_reduce_wq4(accum, m, sums);
                            const uint32_t lane_in_quad = lane_idx % 4;
                            const uint32_t q_owner = lane_idx & 1;
                            const uint32_t local_q_warp = 2 * q_owner + lane_in_quad / 2;
                            const uint32_t q_idx = block_q_idx * BLOCK_Q + warp_q_idx * WARP_Q + local_q_warp;
                            float scale_q_t = kInvSqrtHeadDim;
                            if constexpr (kScaleMode == deep_gemm::kScaleModeQRow)
                                scale_q_t = scale_q_row[local_q_warp];
                            else if constexpr (kScaleMode == deep_gemm::kScaleModeQScalar)
                                scale_q_t = scale_q;

                            // A single Q-row base serves both outputs. Lane T^1 sends
                            // the same Q token's complementary head pair for each row.
                            const float recv_lo = __shfl_xor_sync(0xffffffffu, sums[q_owner ^ 1][0], 1);
                            const float total_lo = sums[q_owner][0] + recv_lo;
                            const float recv_hi = __shfl_xor_sync(0xffffffffu, sums[q_owner ^ 1][1], 1);
                            const float total_hi = sums[q_owner][1] + recv_hi;
                            const float out_lo = total_lo * scale_q_t;
                            const float out_hi = total_hi * scale_q_t;
                            if constexpr (kIsCompressedLogits) {
                                const uint32_t lq = warp_q_idx * WARP_Q + local_q_warp;
                                // Finish each unsigned relative column before adding the 64-bit row base.
                                const uint32_t rel_lo = kv_offset + mma_offset + v_0_offset - seq_k_start[lq];
                                const uint32_t rel_hi = kv_offset + mma_offset + v_1_offset - seq_k_start[lq];
                                const uint32_t len = seq_k_end[lq] - seq_k_start[lq];
                                if (rel_lo < len)
                                    params.logits[q_idx * params.stride_k + rel_lo] = out_lo;
                                if (rel_hi < len)
                                    params.logits[q_idx * params.stride_k + rel_hi] = out_hi;
                            } else {
                                auto* out_ptr = params.logits + q_idx * params.stride_k + kv_offset + mma_offset + v_0_offset;
                                out_ptr[0] = out_lo;
                                out_ptr[8] = out_hi;
                            }
                        } else {
                            static constexpr int kQPerLane = 2;
                            // CUTE_STATIC_ASSERT(MmaIterN{} == 1);
                            float sums[kQPerLane][2];
                            deep_gemm::float_epilogue_reduce_wq4(accum, m, sums);
                            const uint32_t q_base = block_q_idx * BLOCK_Q + warp_q_idx * WARP_Q;
                            const uint32_t lane_in_quad = lane_idx % 4;
                            #pragma unroll
                            for (int t = 0; t < kQPerLane; ++t) {
                                // sums[t] belongs to warp-local q token 2*t + lane_in_quad/2
                                const uint32_t local_q_warp = 2 * t + lane_in_quad / 2;
                                float scale_q_t = 1.0f;
                                if constexpr (kScaleMode == deep_gemm::kScaleModeQRow)
                                    scale_q_t = scale_q_row[local_q_warp];
                                else if constexpr (kScaleMode == deep_gemm::kScaleModeQScalar)
                                    scale_q_t = scale_q;
                                const float recv = __shfl_xor_sync(0xffffffffu, sums[t][r_mine ^ 1], 1);
                                const float total = sums[t][r_mine] + recv;
                                const uint32_t q_idx = q_base + local_q_warp;
                                if constexpr (kIsCompressedLogits) {
                                    const uint32_t lq = warp_q_idx * WARP_Q + local_q_warp;
                                    const uint32_t rel_kv = kv_offset + mma_offset + v_offset_mine - seq_k_start[lq];
                                    const uint32_t len = seq_k_end[lq] - seq_k_start[lq];
                                    if (rel_kv < len)
                                        params.logits[q_idx * params.stride_k + rel_kv] = total * kInvSqrtHeadDim * scale_q_t;
                                } else {
                                    params.logits[q_idx * params.stride_k + kv_offset + mma_offset + v_offset_mine] = total * kInvSqrtHeadDim * scale_q_t;
                                }
                            }
                        }
                    }
                } else {
                    if constexpr (kScaleMode == deep_gemm::kScaleModeWeights) {
                        // bf16 output: separate cvt and fma2 phases with __ppu_sched_bound()
                        constexpr int kTotalTransforms = 4 * size<2>(accum);
                        __ppu_bfloat162 sum_0 = {0, 0};
                        __ppu_bfloat162 sum_1 = {0, 0};
                        uint32_t cvt_buf[kTotalTransforms];

                        // cvt phase: issue all ppu.cvt.rtte.bf16x2.f32.relu
                        deep_gemm::cvt_accum_to_bf16x2_buf<kTotalTransforms>(accum, m, cvt_buf);
                        __ppu_sched_bound();
                        // fma2 phase: issue all __hfma2
                        deep_gemm::fma2_phase_tCrW<kTotalTransforms>(cvt_buf, tCrW, sum_0, sum_1);
                        __ppu_bfloat16 v_0_bf = __hadd(__low2bfloat16(sum_0), __high2bfloat16(sum_0));
                        __ppu_bfloat16 v_1_bf = __hadd(__low2bfloat16(sum_1), __high2bfloat16(sum_1));
                        __ppu_bfloat162 packed = deep_gemm::shfl_xor_reduce_bf16x2({v_0_bf, v_1_bf});
                        if constexpr (kIsCompressedLogits) {
                            const uint32_t rel_kv = static_cast<uint32_t>(logits_kv_offset) - seq_k_start[warp_q_idx];
                            const uint32_t len = seq_k_end[warp_q_idx] - seq_k_start[warp_q_idx];
                            if (rel_kv + v_0_offset < len)
                                params.logits[logits_q_offset + rel_kv + v_0_offset] = __low2bfloat16(packed);
                            if (rel_kv + v_1_offset < len)
                                params.logits[logits_q_offset + rel_kv + v_1_offset] = __high2bfloat16(packed);
                        } else {
                            params.logits[logits_q_offset + logits_kv_offset + v_0_offset] = __low2bfloat16(packed);
                            params.logits[logits_q_offset + logits_kv_offset + v_1_offset] = __high2bfloat16(packed);
                        }
                    } else {
                        if constexpr (kQOwnerEpilogue) {
                            // BF16 Q-owner: pack two KV rows of the same Q token.
                            // lo = KV row 0, hi = KV row 1; no FP32 reduction.
                            CUTE_STATIC_ASSERT(MmaIterN == 1);
                            CUTE_STATIC_ASSERT(WARP_Q == 4 && kNumHeads == 4);
                            const uint32_t lane_in_quad = lane_idx % 4;
                            const uint32_t q_owner = lane_idx & 1;
                            const uint32_t local_q_warp = 2 * q_owner + lane_in_quad / 2;
                            const uint32_t q_idx = block_q_idx * BLOCK_Q + warp_q_idx * WARP_Q + local_q_warp;
                            uint32_t pa0, pb0, pa1, pb1;
                            pa0 = deep_gemm::tix::cvt_rte_bf16x2_f32_relu(accum(2, m, 0), accum(0, m, 0));
                            pb0 = deep_gemm::tix::cvt_rte_bf16x2_f32_relu(accum(3, m, 0), accum(1, m, 0));
                            pa1 = deep_gemm::tix::cvt_rte_bf16x2_f32_relu(accum(6, m, 0), accum(4, m, 0));
                            pb1 = deep_gemm::tix::cvt_rte_bf16x2_f32_relu(accum(7, m, 0), accum(5, m, 0));
                            __ppu_bfloat162 partial0 = __hadd2(reinterpret_cast<__ppu_bfloat162&>(pa0),
                                                            reinterpret_cast<__ppu_bfloat162&>(pb0));
                            __ppu_bfloat162 partial1 = __hadd2(reinterpret_cast<__ppu_bfloat162&>(pa1),
                                                            reinterpret_cast<__ppu_bfloat162&>(pb1));
                            // The neighbor owns the opposite Q token. It must send
                            // its non-owned partial so this lane receives its own Q.
                            const __ppu_bfloat162 partial_mine = q_owner ? partial1 : partial0;
                            const __ppu_bfloat162 partial_send = q_owner ? partial0 : partial1;
                            const uint32_t recv = __shfl_xor_sync(0xffffffffu, reinterpret_cast<const uint32_t&>(partial_send), 1);
                            __ppu_bfloat162 total = __hadd2(partial_mine, reinterpret_cast<const __ppu_bfloat162&>(recv));
                            // Both packed values belong to one Q: preserve FP32
                            // scale construction, then BF16 scale cvt and BF16 mul.
                            float scale_q_t = kInvSqrtHeadDim;
                            if constexpr (kScaleMode == deep_gemm::kScaleModeQRow)
                                scale_q_t *= scale_q_row[local_q_warp];
                            else if constexpr (kScaleMode == deep_gemm::kScaleModeQScalar)
                                scale_q_t *= scale_q;
                            const uint32_t scale2 = deep_gemm::tix::cvt_rte_bf16x2_f32(scale_q_t, scale_q_t);
                            const __ppu_bfloat162 out2 = __hmul2(total, reinterpret_cast<const __ppu_bfloat162&>(scale2));
                            if constexpr (kIsCompressedLogits) {
                                const uint32_t lq = warp_q_idx * WARP_Q + local_q_warp;
                                // Finish each unsigned relative column before adding the 64-bit row base.
                                const uint32_t rel_lo = kv_offset + mma_offset + v_0_offset - seq_k_start[lq];
                                const uint32_t rel_hi = kv_offset + mma_offset + v_1_offset - seq_k_start[lq];
                                const uint32_t len = seq_k_end[lq] - seq_k_start[lq];
                                if (rel_lo < len)
                                    params.logits[q_idx * params.stride_k + rel_lo] = __low2bfloat16(out2);
                                if (rel_hi < len)
                                    params.logits[q_idx * params.stride_k + rel_hi] = __high2bfloat16(out2);
                            } else {
                                auto* out_ptr = params.logits + q_idx * params.stride_k + kv_offset + mma_offset + v_0_offset;
                                out_ptr[0] = __low2bfloat16(out2);
                                out_ptr[8] = __high2bfloat16(out2);
                            }
                        } else {
                            // BF16 t-packed epilogue: each bf16x2 packs the two q tokens of one
                            // kv row (lo = t0, hi = t1); select/scale/mul run once per m, not per t
                            CUTE_STATIC_ASSERT(MmaIterN == 1);
                            const uint32_t q_base = block_q_idx * BLOCK_Q + warp_q_idx * WARP_Q;
                            const uint32_t lane_in_quad = lane_idx % 4;

                            uint32_t pa0, pb0, pa1, pb1;
                            pa0 = deep_gemm::tix::cvt_rte_bf16x2_f32_relu(accum(4, m, 0), accum(0, m, 0));
                            pb0 = deep_gemm::tix::cvt_rte_bf16x2_f32_relu(accum(5, m, 0), accum(1, m, 0));
                            pa1 = deep_gemm::tix::cvt_rte_bf16x2_f32_relu(accum(6, m, 0), accum(2, m, 0));
                            pb1 = deep_gemm::tix::cvt_rte_bf16x2_f32_relu(accum(7, m, 0), accum(3, m, 0));
                            __ppu_bfloat162 partial0 = __hadd2(reinterpret_cast<__ppu_bfloat162&>(pa0),
                                                            reinterpret_cast<__ppu_bfloat162&>(pb0));
                            __ppu_bfloat162 partial1 = __hadd2(reinterpret_cast<__ppu_bfloat162&>(pa1),
                                                            reinterpret_cast<__ppu_bfloat162&>(pb1));
                            const uint32_t recv0 = __shfl_xor_sync(0xffffffffu, reinterpret_cast<const uint32_t&>(partial0), 1);
                            const uint32_t recv1 = __shfl_xor_sync(0xffffffffu, reinterpret_cast<const uint32_t&>(partial1), 1);
                            __ppu_bfloat162 total0 = __hadd2(partial0, reinterpret_cast<const __ppu_bfloat162&>(recv0));
                            __ppu_bfloat162 total1 = __hadd2(partial1, reinterpret_cast<const __ppu_bfloat162&>(recv1));
                            __ppu_bfloat162 total = r_mine ? total1 : total0;

                            // Scale both t outputs with one packed cvt + one packed mul;
                            // QRow's per-row scale_q rides in the two f32 cvt sources
                            float scale_lo = kInvSqrtHeadDim, scale_hi = kInvSqrtHeadDim;
                            if constexpr (kScaleMode == deep_gemm::kScaleModeQRow) {
                                scale_lo *= scale_q_row[lane_in_quad / 2];
                                scale_hi *= scale_q_row[2 + lane_in_quad / 2];
                            } else if constexpr (kScaleMode == deep_gemm::kScaleModeQScalar) {
                                scale_lo *= scale_q;
                                scale_hi *= scale_q;
                            }
                            const uint32_t scale2 = deep_gemm::tix::cvt_rte_bf16x2_f32(scale_hi, scale_lo);
                            const __ppu_bfloat162 out2 = __hmul2(total, reinterpret_cast<const __ppu_bfloat162&>(scale2));

                            const uint32_t q_idx_lo = q_base + lane_in_quad / 2;
                            const uint32_t q_idx_hi = q_base + 2 + lane_in_quad / 2;
                            if constexpr (kIsCompressedLogits) {
                                const uint32_t lq_lo = warp_q_idx * WARP_Q + lane_in_quad / 2;
                                const uint32_t lq_hi = warp_q_idx * WARP_Q + 2 + lane_in_quad / 2;
                                const uint32_t rel_lo = kv_offset + mma_offset + v_offset_mine - seq_k_start[lq_lo];
                                const uint32_t rel_hi = kv_offset + mma_offset + v_offset_mine - seq_k_start[lq_hi];
                                if (rel_lo < seq_k_end[lq_lo] - seq_k_start[lq_lo])
                                    params.logits[q_idx_lo * params.stride_k + rel_lo] = __low2bfloat16(out2);
                                if (rel_hi < seq_k_end[lq_hi] - seq_k_start[lq_hi])
                                    params.logits[q_idx_hi * params.stride_k + rel_hi] = __high2bfloat16(out2);
                            } else {
                                params.logits[q_idx_lo * params.stride_k + kv_offset + mma_offset + v_offset_mine] = __low2bfloat16(out2);
                                params.logits[q_idx_hi * params.stride_k + kv_offset + mma_offset + v_offset_mine] = __high2bfloat16(out2);
                            }
                        }
                    }
                }
            }
        };

        // ── Outer loop: Q blocks ──────────────────────────────────────────
        // Test the range before dividing by full_kv_blocks, including empty Q/K.
        while (balance_active ? output_tile_idx < output_tile_end
                              : block_q_idx < num_q_blocks) {
            uint64_t segment_end = 0;
            uint32_t full_first = 0, full_last = 0;
            if constexpr (kBalanceOutputTiles) {
                if (balance_active) {
                    block_q_idx = static_cast<uint32_t>(output_tile_idx / full_kv_blocks);
                    full_first = static_cast<uint32_t>(output_tile_idx % full_kv_blocks);
                    const uint64_t q_tile_end = (uint64_t(block_q_idx) + 1) * full_kv_blocks;
                    segment_end = q_tile_end < output_tile_end ? q_tile_end : output_tile_end;
                    full_last = full_first + static_cast<uint32_t>(segment_end - output_tile_idx);
                }
            }
            CUTE_TIE_DECL(
                (deep_gemm::load_schedule<BLOCK_Q, kNumQStages, BLOCK_KV, kIsCompressedLogits>(
                    block_q_idx, q_iter_idx, 0, params.seq_len_q, params.seq_len_k,
                    params.cu_seq_len_k_start, params.cu_seq_len_k_end,
                    seq_k_start, seq_k_end)),
                q_stage_idx, q_phase, kv_start, kv_end, num_kv_blocks);

            if constexpr (kBalanceOutputTiles) {
                if (balance_active) {
                    // Clip this worker's virtual segment to the scheduled window;
                    // retain load_schedule's original aligned KV origin.
                    const uint32_t first_kv_tile = min(full_first, num_kv_blocks);
                    const uint32_t last_kv_tile = min(full_last, num_kv_blocks);
                    const uint32_t original_kv_start = kv_start;
                    const uint32_t original_kv_end = kv_end;
                    kv_start = original_kv_start + first_kv_tile * BLOCK_KV;
                    num_kv_blocks = last_kv_tile - first_kv_tile;
                    kv_end = num_kv_blocks ? min(original_kv_end, original_kv_start + last_kv_tile * BLOCK_KV) : kv_start;
                }
            }


            // ── Offset Q / K / scale data pointers ──────────────────────────
            tAgA.data() = tKgK.data() + kv_start * BLOCK_K;
            tSFAgSFA.data() = tSFKgSFK.data() + kv_start;
            tBgB.data() = tQgQ.data() + block_q_idx * BLOCK_Q * kNumHeads * BLOCK_K;
            tSFBgSFB.data() = tSFQgSFQ.data() + block_q_idx * BLOCK_Q * kNumHeads;
            tWgW.data() = tWgW_base.data() + block_q_idx * BLOCK_Q * kNumHeads;

            uint32_t current_stage_kv = 0;
            if (num_kv_blocks > 0) {
                // ── Load Q + q_sf + weights (AIU) ──────────────────────────
                load_q_g2s(block_q_idx, q_stage_idx);

                // ── Prologue: load first (kNumKVStages - 1) K blocks (AIU) ──
                for (uint32_t kv_pipe = 0; kv_pipe < kNumKVStages - 1; kv_pipe++) {
                    if (kv_pipe < num_kv_blocks) {
                        load_kv_g2s(kv_start, kv_end, kv_pipe, kv_pipe);
                    }
                    current_stage_kv++;
                    cp_async_fence();
                }

                // ── Wait for Q + first K ───────────────────────────────────
                cp_async_wait<kNumKVStages - 2>();
                __syncthreads();

                // ── Load Q + q_sf + weights from smem to vreg ──────────────
                load_q_s2r(q_stage_idx);
            }

            deep_gemm::warp_interleave_start<WarpInterleaving>(warp_group_id, NumThreadsPerCTA);

            // ── Inner loop: KV blocks ──────────────────────────────────────
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                uint32_t kv_stage_idx = kv_block_idx % kNumKVStages;

                deep_gemm::warp_interleave_sync<WarpInterleaving>(warp_group_id, NumThreadsPerCTA);

                // Issue next K block
                uint32_t next_kv_block_idx = kv_block_idx + kNumKVStages - 1;
                if (next_kv_block_idx < num_kv_blocks) {
                    load_kv_g2s(kv_start, kv_end, next_kv_block_idx, current_stage_kv);
                }
                cp_async_fence();
                current_stage_kv = (current_stage_kv + 1) % kNumKVStages;

                constexpr int M_BLOCK = size<1>(accum);
                constexpr int N_BLOCK = size<2>(accum);
                constexpr int K_BLOCK = size<2>(tCrA);
                // FP32 defaults to the original coarse path; positive G shares the
                // grouped schedule below while retaining the FP32 epilogue.
                if constexpr (cute::is_same_v<ElementLogits, float> && kMBlockGroup == 0) {
                    for_each(make_int_sequence<K_BLOCK>{}, [&](auto k_block) {
                        load_kv_s2r(k_block, kv_stage_idx);
                        cute::gemm(tiled_mma, accum, tCrA(_, _, k_block), tCrSFA(_, _, k_block), tCrB(_, _, k_block),
                                    tCrSFB(_, _, k_block), accum);
                    });
                    deep_gemm::warp_interleave_arrive<WarpInterleaving>(warp_group_id, NumThreadsPerCTA);
                    #pragma unroll
                    for (int m_block = 0; m_block < size<1>(accum); m_block++) {
                        epilogue_mblock(kv_start, kv_block_idx, m_block);
                    }
                } else { // Grouped S2R/MMA and epilogue, independent of output precision
                    static_assert(kMBlockGroup > 0 && M_BLOCK % kMBlockGroup == 0, "invalid M-block group");
                    // Each packed SFA entry serves m_group MMA M blocks.
                    constexpr int m_group = M_BLOCK / size<1>(tCrSFA);
                    constexpr int n_group = N_BLOCK / size<1>(tCrSFB);
                    // G=1 keeps the original grouped schedule. Larger groups expose independent
                    // MMA work before each handoff; epilogue arithmetic stays dtype-specific.
                    if constexpr (kMBlockGroup == 1) {
                        for_each(make_int_sequence<M_BLOCK>{}, [&](auto m_block) {
                            if constexpr(m_block > 0) deep_gemm::warp_interleave_sync<WarpInterleaving>(warp_group_id, NumThreadsPerCTA);
                            for_each(make_int_sequence<K_BLOCK>{}, [&](auto k_block) {
                                load_kv_a_s2r_mblock(k_block, kv_stage_idx, m_block);
                                if constexpr (m_block % m_group == 0) {
                                    load_kv_sfa_s2r_group(k_block, kv_stage_idx, m_block / m_group);
                                }
                                for_each(make_int_sequence<N_BLOCK>{}, [&](auto n_block) {
                                    Tensor s = make_tensor<uint32_t>(Int<4>{});
                                    MMA_Atom<MmaInst> mma_atom;
                                    Tensor d = accum(_, m_block, n_block);
                                    Tensor a = tCrA(_, m_block, k_block);
                                    Tensor b = tCrB(_, n_block, k_block);
                                    Tensor c = accum(_, m_block, n_block);
                                    s[0] = tCrSFA(_, _, k_block)[m_block / m_group];
                                    s[1] = tCrSFB(_, _, k_block)[n_block / n_group];
                                    s[2] = m_block % m_group;
                                    s[3] = n_block % n_group;
                                    cute::mma_unpack(mma_atom, d, a, b, c, s);
                                });
                            });
                            deep_gemm::warp_interleave_arrive<WarpInterleaving>(warp_group_id, NumThreadsPerCTA);
                            epilogue_mblock(kv_start, kv_block_idx, m_block);
                        });
                    } else {
                        for_each(make_int_sequence<M_BLOCK / kMBlockGroup>{}, [&](auto group_block) {
                            if constexpr(group_block > 0) deep_gemm::warp_interleave_sync<WarpInterleaving>(warp_group_id, NumThreadsPerCTA);
                            for_each(make_int_sequence<kMBlockGroup>{}, [&](auto group_offset) {
                                constexpr int m_block = decltype(group_block)::value * kMBlockGroup + decltype(group_offset)::value;
                                for_each(make_int_sequence<K_BLOCK>{}, [&](auto k_block) {
                                    load_kv_a_s2r_mblock(k_block, kv_stage_idx, m_block);
                                    if constexpr (m_block % m_group == 0) {
                                        load_kv_sfa_s2r_group(k_block, kv_stage_idx, m_block / m_group);
                                    }
                                    for_each(make_int_sequence<N_BLOCK>{}, [&](auto n_block) {
                                        Tensor s = make_tensor<uint32_t>(Int<4>{});
                                        MMA_Atom<MmaInst> mma_atom;
                                        Tensor d = accum(_, m_block, n_block);
                                        Tensor a = tCrA(_, m_block, k_block);
                                        Tensor b = tCrB(_, n_block, k_block);
                                        Tensor c = accum(_, m_block, n_block);
                                        s[0] = tCrSFA(_, _, k_block)[m_block / m_group];
                                        s[1] = tCrSFB(_, _, k_block)[n_block / n_group];
                                        s[2] = m_block % m_group;
                                        s[3] = n_block % n_group;
                                        cute::mma_unpack(mma_atom, d, a, b, c, s);
                                    });
                                });
                            });
                            deep_gemm::warp_interleave_arrive<WarpInterleaving>(warp_group_id, NumThreadsPerCTA);
                            for_each(make_int_sequence<kMBlockGroup>{}, [&](auto group_offset) {
                                constexpr int m_block = decltype(group_block)::value * kMBlockGroup + decltype(group_offset)::value;
                                epilogue_mblock(kv_start, kv_block_idx, m_block);
                            });
                        });
                    }
                }

                // Wait for next K
                cp_async_wait<kNumKVStages - 2>();
                if constexpr (!WarpInterleaving) {
                    __syncthreads();
                }
                clear(accum);
            } // end KV loop

            deep_gemm::warp_interleave_end<WarpInterleaving>(warp_group_id, NumThreadsPerCTA);

            cp_async_wait<0>();
            __syncthreads();

            // Jump to next Q block
            if constexpr (kBalanceOutputTiles) {
                if (balance_active) {
                    output_tile_idx = segment_end;
                    ++q_iter_idx;
                } else {
                    CUTE_TIE(deep_gemm::get_next_block_q_idx(block_q_idx, q_iter_idx), block_q_idx, q_iter_idx);
                }
            } else {
                CUTE_TIE(deep_gemm::get_next_block_q_idx(block_q_idx, q_iter_idx), block_q_idx, q_iter_idx);
            }
        } // end Q loop
    }
};

} // namespace cutlass::gemm::kernel

namespace deep_gemm {

template <typename ElementQK, typename ElementAcc, typename ElementLogits, typename ElementWeights,
          int kNumHeads, int kHeadDim, int BLOCK_QH,
          int BLOCK_KV, int WARP_QH, int WARP_KV, int kNumQStages, int kNumKVStages,
          typename StrideKType = uint32_t,
          bool kIsCompressedLogits = false,
          uint32_t kScaleMode = deep_gemm::kScaleModeWeights>
class AttentionFP4 {
public:
    static void run(const ElementQK* ptr_q, const uint32_t* q_sf, const ElementQK* ptr_k, const uint32_t* k_sf,
                    const ElementWeights* weights, int* cu_seq_len_k_start, int* cu_seq_len_k_end, ElementLogits* logits,
                    const int seq_len_q, const int seq_len_k, const StrideKType stride_k, hggcStream_t stream,
                    int num_sms) {
        using AttnKernel =
            cutlass::gemm::kernel::PPUMqaLogitsFP4<ElementQK, ElementAcc, ElementLogits, ElementWeights, kNumHeads, kHeadDim, BLOCK_QH,
                                                    BLOCK_KV, WARP_QH, WARP_KV, kNumQStages, kNumKVStages, StrideKType, kIsCompressedLogits, kScaleMode>;

        static constexpr int BLOCK_M = AttnKernel::BLOCK_M;
        static constexpr int BLOCK_N = AttnKernel::BLOCK_N;
        static constexpr int BLOCK_K = AttnKernel::BLOCK_K;
        static constexpr int WARP_M = AttnKernel::WARP_M;
        static constexpr int WARP_N = AttnKernel::WARP_N;
        static constexpr int MaxThreadsPerBlock = AttnKernel::MaxThreadsPerBlock;

        int max_blocks_per_cu = compute_occupancy_for_kernel<AttnKernel>();
        int threadblock_count = num_sms * max_blocks_per_cu;

        typename AttnKernel::Arguments arguments{
            ptr_q,  q_sf,      ptr_k,     k_sf,    weights, cu_seq_len_k_start, cu_seq_len_k_end,
            logits, seq_len_q, seq_len_k, stride_k};
        auto params = arguments;
        dim3 const block(MaxThreadsPerBlock, 1, 1);
        dim3 const grid(threadblock_count, 1, 1);
        int smem_size_kernel = AttnKernel::SharedStorageSize;

        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()) {
            dg_prof_params.set_mqa_logits_params("fp4", seq_len_q, seq_len_k, kNumHeads, kHeadDim * 2, stream);
            if constexpr (cute::is_same_v<ElementLogits, __ppu_bfloat16>) {
                dg_prof_params.add_params("logits_dtype", std::string("bf16"));
            }
            if constexpr (cute::is_same_v<ElementWeights, __ppu_bfloat16>) {
                dg_prof_params.add_params("weights_dtype", std::string("bf16"));
            }
        }

        char* pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            hggcFuncAttributes attr;
            hggcFuncGetAttributes(&attr, cutlass::device_kernel<AttnKernel>);
            printf("[mqa_logits_fp4:]\n");
            printf("kNumHeads:%d, kHeadDim:%d(packed), BLOCK_QH:%d, BLOCK_KV:%d\n", kNumHeads, kHeadDim, BLOCK_QH,
                   BLOCK_KV);
            printf("ThreadblockShape[%d, %d], WarpShape[%d, %d], kNumQStages:%d, kNumKVStages:%d\n", BLOCK_M, BLOCK_N,
                   WARP_M, WARP_N, kNumQStages, kNumKVStages);
            printf("num_sms:%d, max_blocks_per_cu:%d, threadblock_count:%d, num_threads:%d\n", num_sms, max_blocks_per_cu,
                   threadblock_count, MaxThreadsPerBlock);
            printf("smem_size:%d, vreg:%d, stack:%d\n", smem_size_kernel, int(attr.numRegs), int(attr.localSizeBytes));
            printf("compressed_logits:%s, ", kIsCompressedLogits ? "true" : "false");
            printf("weights_bf16:%s\n", cute::is_same_v<ElementWeights, __ppu_bfloat16> ? "true" : "false");
        }

        ProfilingInterface::Instance().instrument(true, dg_prof_params);
        cutlass::device_kernel<AttnKernel><<<grid, block, smem_size_kernel, stream>>>(params);
        ProfilingInterface::Instance().instrument(false, dg_prof_params);
    }
};

}; // namespace deep_gemm
