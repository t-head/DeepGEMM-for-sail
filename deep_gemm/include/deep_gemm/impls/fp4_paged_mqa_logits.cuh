#pragma once

#include <type_traits>
#include <cutlass/arch/barrier.h>
#include <deep_gemm/tix/ld_st.cuh>
#include <deep_gemm/tix/utils.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/impls/mqa_logits_utils.cuh>
#include <deep_gemm/scheduler/paged_mqa_logits_scheduler.cuh>

namespace cutlass::gemm::kernel {

// ============================================================================
// FP4 Paged MQA Logits Kernel
// ============================================================================
// Key differences from FP8 PPUPagedMqaLogits:
//   - ElementQK is uint8_t (packed FP4, 2 elements per byte)
//   - kHeadDim is the *packed* dimension (original_head_dim / 2 = 64)
//   - Additional q_sf (Q scale factor, uint8 e8m0) parameter
//   - k_sf is uint8 e8m0 (not float32 per-token scale)
//   - Q/K/q_sf/k_sf and optional weights use AIU load
//   - Output dtype controlled by ElementLogits (float or bfloat16)
//   - MMA atom: F32F4F4F32 (16x16x64)
// ============================================================================

template <typename ElementQK, typename ElementLogits, typename ElementWeights>
struct FP4PagedArguments {
    const ElementQK* ptr_q;
    const uint32_t* q_sf; // Q scale factor (packed e8m0, 4×uint8 per int32)
    const ElementQK* ptr_k;
    const uint32_t* k_sf; // K scale factor (packed e8m0, uint32_t)
    const ElementWeights* weights;
    const uint32_t batch_size;
    const uint64_t logits_stride;
    const uint64_t kv_cache_stride_bytes;
    const uint32_t block_table_stride;
    const uint32_t* context_lens;
    ElementLogits* logits;
    const uint32_t* block_table;
    const uint32_t* schedule_meta;
    const uint32_t* grouped_meta;
};

template <typename ElementQK, typename ElementAcc, typename ElementLogits, typename ElementWeights, uint32_t kNextN, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t PAGE_KV, uint32_t SPLIT_KV, uint32_t BLOCK_KV, uint32_t WARP_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          bool SPLIT_MBLOCK, uint32_t kScaleMode = deep_gemm::kScaleModeWeights, bool kGrouped = false>
class PPUPagedMqaLogitsFP4 {
public:
    static_assert(PAGE_KV == 32 || (PAGE_KV > 0 && PAGE_KV % BLOCK_KV == 0),
                  "KV page must be 32 rows or a positive multiple of the compute tile");
    static_assert(cute::is_same_v<ElementQK, uint8_t>, "FP4 paged MQA logits requires uint8_t ElementQK");
    static_assert(kHeadDim == 64, "FP4 packed head_dim must be 64 (original 128 / 2)");
    static_assert(kScaleMode == deep_gemm::kScaleModeWeights || kScaleMode == deep_gemm::kScaleModeUnity,
                  "FP4 paged MQA supports weighted or unity-scale AVG mode");
    // Grouped Q stage 0 caches scales; later stages read them from global memory.
    static constexpr uint32_t kSharedQScaleStages = kGrouped ? 1 : kNumQStages;
    static_assert(!kGrouped || kNextN * kNumHeads == 128, "Invalid grouped Q geometry");
    static constexpr bool kIsAvg = kScaleMode == deep_gemm::kScaleModeUnity;
    static_assert(!kIsAvg || kNumHeads == 4 || kNumHeads == 8, "FP4 paged AVG requires 4 or 8 heads");
    static constexpr int kMmaNumHeads = kIsAvg ? 16 : kNumHeads;
    static constexpr int kQueriesPerWarp = kMmaNumHeads / kNumHeads;
    static constexpr float kInvSqrtHeadDim = 0.08838834764831844f; // 1 / sqrt(128)

    static constexpr int BLOCK_M = BLOCK_KV;
    static constexpr int BLOCK_N = ((kNextN + kQueriesPerWarp - 1) / kQueriesPerWarp) * kMmaNumHeads;
    static constexpr int BLOCK_K = 64; // kHeadDim = 64
    static constexpr int WARP_M = WARP_KV;
    static constexpr int WARP_N = kMmaNumHeads;

    static constexpr int kNumMathWarpGroups = SPLIT_KV / BLOCK_KV;

    using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
    static constexpr int WarpOnM = BLOCK_M / WARP_M;
    static constexpr int WarpOnN = BLOCK_N / WARP_N;
    // FP4 MMA: 16x16x64 F32F4F4F32
    using MmaInst = PPU0015_16x16x64_F32F4F4F32_TN;
    // For FP4 (uint8, 1 byte), MMA K dimension is 32 (same as FP8 int8)
    using MmaK_type = _32;

    static constexpr int InstM = 16;
    static constexpr int InstN = 16;
    static constexpr int MmaIterM = WARP_M / InstM;
    static constexpr int MmaIterN = WARP_N / InstN;
    using PermutationMNK =
        Tile<Layout<Shape<Int<InstM>, Int<WarpOnM>, Int<MmaIterM>>, Stride<_1, Int<WARP_M>, Int<InstM>>>,
             Layout<Shape<Int<InstN>, Int<WarpOnN>, Int<MmaIterN>>, Stride<_1, Int<WARP_N>, Int<InstN>>>, MmaK_type>;
    using TiledMma = TiledMMA<MMA_Atom<MmaInst>, Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, _1>>, PermutationMNK>;

    static constexpr int MaxThreadsPerBlock = kNumMathWarpGroups * size(TiledMma{});
    static constexpr int WarpOnGroup = size(TiledMma{}) / 32;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    // H8's 384-thread tile alternates two six-warp halves.
    static constexpr bool WarpInterleaving = !kGrouped &&
        (MaxThreadsPerBlock == 512 || (kIsAvg && kNumHeads == 8 && MaxThreadsPerBlock == 384));
    static_assert(!(WarpInterleaving && kNumQStages == 1), "warp-interleave does not support stage_q = 1");

    // A page32 copy fills half a 64-row KV stage. The TSM stage stride must
    // still span the whole compute tile, not one page-local AIU copy.
    using DefaultOperandA =
        cutlass::gemm::config::DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, ElementQK, false,
            Int<(PAGE_KV < BLOCK_KV ? PAGE_KV : BLOCK_KV)>, Int<BLOCK_K>, false,
            (kIsAvg || PAGE_KV < BLOCK_KV) ? BLOCK_M * BLOCK_K : 0>;
    using DefaultOperandB =
        cutlass::gemm::config::DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, ElementQK, false, Int<BLOCK_N>, Int<BLOCK_K>, true>;
    // A (K/V)
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

    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtomA{}, Shape<Int<BLOCK_M>, Int<BLOCK_K>, Int<kNumKVStages>, Int<kNumMathWarpGroups>>{}));
    using SmemLayoutB =
        decltype(tile_to_shape(SmemLayoutAtomB{}, Shape<Int<BLOCK_N>, Int<BLOCK_K>, Int<kNumQStages>>{}));

    // ===========================================================================
    // FP4 Scale (SFA/SFB) gmem tiled copy — async copy, no pre-rearrangement
    // ===========================================================================
    // K_sf (SFA): per-row e8m0 scale, shape [num_kv_blocks * block_kv, kv_sf_dim]
    //   kv_sf_dim = kHeadDim / 16 = 4  (for original head_dim=128, packed=64)
    //   We load kv_sf_dim = 4 bytes per KV row as one uint32_t
    // Q_sf (SFB): per-head-group e8m0 scale, shape [batch * next_n, num_heads, q_sf_dim]
    //   q_sf_dim = kHeadDim / 16 = 4 bytes per head, loaded as one uint32_t
    // Weights: ElementWeights per query/head (BF16 for weighted DSA; unused for AVG)

    // SFA: k_sf — async copy of uint32_t (4 bytes per KV row)
    // Layout: BLOCK_M rows × 1 column of uint32_t
    using SFCopyAtomWidth = cute::uint32_t;

    // SFA (k_sf): one uint32_t per KV row
    using DefaultOperandSFA =
        cutlass::gemm::config::DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, uint32_t, false, _1, Int<BLOCK_M>, false, 0, false>;
    using SmemLayoutAtomSFA = typename DefaultOperandSFA::SmemLayoutAtom;
    using GmemTiledCopySFA = typename DefaultOperandSFA::GmemTiledCopy;
    using SmemLayoutSFA = decltype(tile_to_shape(
        SmemLayoutAtomSFA{}, make_shape(_1{}, Int<BLOCK_M>{}, Int<kNumKVStages>{}, Int<kNumMathWarpGroups>{})));

    // SFB (q_sf): one packed uint32_t scale per Q head
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
    // The scale copy packs four MMA-block selector slots per warp. Narrow
    // AVG warps consume fewer slots; repeat their valid scales in the unused
    // slots instead of reading past the warp's scale rows (or the allocation).
    template <int WarpCount, int MmaIters, int Stages, int Rows>
    using AvgSFTransLayout = Layout<
        Shape<Shape<Int<WarpCount>, Shape<Int<MmaIters>, Int<(MmaIters < 4 ? 4 / MmaIters : 1)>>>,
              Shape<_2, _8, _2>, Int<Stages>>,
        Stride<Stride<Int<MmaIters * 32>, Stride<_32, _0>>,
               Stride<_16, _2, _1>, Int<Rows * 2>>>;
    // SmemTiledCopyWeights: use conditional copy atom type (same as fp4_mqa_logits.cuh)
    using WeightCopyAtomType = cute::conditional_t<cute::is_same_v<ElementWeights, __ppu_bfloat16>, uint32_t, uint64_t>;
    using SmemTiledCopyWeights = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<WeightCopyAtomType>, ElementWeights>{},
        Layout<Shape<Shape<_4, Int<WarpOnN>>, _1>, Stride<Stride<_1, _4>, _1>>{}, Layout<Shape<_2, _1>>{}));

    static_assert(kNumQStages <= 2 && kNumKVStages >= 3, "invalid pipeline stages");

    // Kernel level shared memory storage
    struct SharedStorage {
        cute::array_aligned<ElementQK, cute::cosize_v<SmemLayoutA>> smem_k;       // KV packed FP4
        cute::array_aligned<ElementQK, cute::cosize_v<SmemLayoutB>> smem_q;       // Q packed FP4
        cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutSFA>> smem_k_sf;   // KV e8m0 scales (uint32_t)
        // Only the first grouped Q stage caches scales; later stages load
        // them directly. The current H32/H64 shared-memory footprints allow
        // at most five/three CTAs respectively; actual residency also depends
        // on the other kernel resources.
        cute::array_aligned<uint32_t, kGrouped ? BLOCK_N * kSharedQScaleStages
                                             : cute::cosize_v<SmemLayoutSFB>> smem_q_sf;
        cute::array_aligned<ElementWeights, cute::cosize_v<SmemLayoutWeight>> smem_weight;
    };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    // Device side arguments
    using Arguments = FP4PagedArguments<ElementQK, ElementLogits, ElementWeights>;

    // Kernel entry point API
    using Params = Arguments;

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
        if constexpr (kIsAvg) {
            // Pad an incomplete final query group to the MMA tile. AIU fills
            // the remaining Q heads and scale words with zero.
            gmem_tiled_copy_B.desc_.dim_h = kNextN * kNumHeads;
            gmem_tiled_copy_SFB.desc_.dim_w = kNextN * kNumHeads;
        }
    };

    CUTLASS_DEVICE auto load_init(Params const& params) {
        // load init A (K/V — packed FP4)
        Tensor mA_mk =
            make_tensor(make_gmem_ptr(params.ptr_k), Shape<Int<BLOCK_M>, Int<BLOCK_K>>{}, StrideAB{}); // (m,k)
        Tensor mA_mk_mix = make_mix_tensor_like(mA_mk);
        Tensor gA = local_tile(mA_mk_mix, TileShape{}, make_coord(0, 0, _), Step<_1, X, _1>{}); // (BLK_M,BLK_K,k)

        // load init B (Q — packed FP4)
        Tensor mB_nk =
            make_tensor(make_gmem_ptr(params.ptr_q), Shape<Int<BLOCK_N>, Int<BLOCK_K>>{}, StrideAB{}); // (n,k)
        Tensor mB_nk_mix = make_mix_tensor_like(mB_nk);
        Tensor gB = local_tile(mB_nk_mix, TileShape{}, make_coord(0, _, 0), Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)

        // load init k_sf (SFA — e8m0, no pre-rearrangement)
        Tensor mSFA_m = make_tensor(make_gmem_ptr(params.k_sf), Shape<_1, Int<BLOCK_M>>{}, StrideSFA{});
        Tensor mSFA_m_mix = make_mix_tensor_like(mSFA_m);
        Tensor gSFA = local_tile(mSFA_m_mix, Shape<_1, Int<BLOCK_M>>{}, make_coord(_, 0));

        // load init q_sf (SFB — e8m0, no pre-rearrangement)
        Tensor mSFB_n = make_tensor(make_gmem_ptr(params.q_sf), Shape<_1, Int<BLOCK_N>>{}, StrideSFB{});
        Tensor mSFB_n_mix = make_mix_tensor_like(mSFB_n);
        Tensor gSFB = local_tile(mSFB_n_mix, Shape<_1, Int<BLOCK_N>>{}, make_coord(0, _));

        // Per-head weights use ElementWeights.
        Tensor mW_n = make_tensor(make_gmem_ptr(params.weights), Shape<_1, Int<BLOCK_N>>{}, StrideSFB{});
        Tensor mW_n_mix = make_mix_tensor_like(mW_n);
        Tensor gW = local_tile(mW_n_mix, Shape<_1, Int<BLOCK_N>>{}, make_coord(0, _));

        return cute::make_tuple(gA, gB, gSFA, gSFB, gW);
    }

    CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
        run_task(params, smem_buf, ~0u, 0, 0, 0,
                 Int<0>{}, cute::false_type{}, Int<0>{}, Int<0>{});
    }

    // Task-local tags preserve unrolling without making request state part of
    // the kernel geometry or shared-storage type.
    template <typename QSpan, typename ReuseKV, typename Valid0, typename Valid1>
    CUTLASS_DEVICE void run_task(Params const& params, char* smem_buf,
                                 uint32_t only_q, uint32_t kv_begin, uint32_t kv_end,
                                 uint32_t last_q, QSpan, ReuseKV, Valid0, Valid1) {
        constexpr uint32_t kFixedQSpan = QSpan::value;
        constexpr bool kReuseRequest = ReuseKV::value;
        constexpr uint32_t kRequestValid0 = Valid0::value, kRequestValid1 = Valid1::value;
        constexpr bool kStaticRequest = kRequestValid0 != 0;
        // A singleton uses pairs of Q-axis warps for the two M halves. This
        // needs two KV chunks per pipeline step, leaving a second shared stage.
        constexpr bool FoldSingleQ = kGrouped && !kReuseRequest && kFixedQSpan == 1 &&
            kNextN == 4 && kNumHeads == 32 && WARP_M == 64 && SPLIT_MBLOCK;
        constexpr int StorageQSpan = FoldSingleQ ? 2 : kFixedQSpan;
        static_assert(!kReuseRequest || !WarpInterleaving);
#if __HGGC_ARCH__ == 100
        return;
#endif
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
        int warp_m_idx = local_warp_idx % WarpOnM;
        int warp_n_idx = local_warp_idx / WarpOnM;
        int lane_idx = deep_gemm::tix::get_lane_idx();
        int warp_group_id = warp_idx / (WarpInterleaving ? MaxThreadsPerBlock / 64 : 8);

        // Kernel level shared memory storage
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        // init aiu copy and async copy
        init_aiu_copy();

        // init input tensors
        auto [gA, gB, gSFA, gSFB, gW] = load_init(params); // gSFA=k_sf, gSFB=q_sf, gW=weights

        // 4D KV smem: (BLOCK_M, BLOCK_K, kNumKVStages, kNumMathWarpGroups)
        // Slice by warp_group_idx to get this warp_group's 3D region: (BLOCK_M, BLOCK_K, kNumKVStages)
        Tensor sA_full = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutA{}); // 4D
        Tensor sA = sA_full(_, _, _, warp_group_idx); // 3D slice for this warp_group
        Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)
        // Partition the copying of A and B tiles across the threads (use local_thread_idx for 128-thread group)
        auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(local_thread_idx);
        auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(local_thread_idx);
        Tensor tAgA = gmem_thr_copy_A.partition_S(gA);
        Tensor tAsA = gmem_thr_copy_A.partition_D(sA);
        Tensor tBgB = gmem_thr_copy_B.partition_S(gB);
        Tensor tBsB = gmem_thr_copy_B.partition_D(sB);

        // 4D SFA smem: (_1, BLOCK_M, kNumKVStages, kNumMathWarpGroups)
        // Slice by warp_group_idx to get this warp_group's 3D region
        Tensor sSFA_full = make_tensor(make_smem_ptr(shared_storage.smem_k_sf.data()), SmemLayoutSFA{}); // 4D
        Tensor sSFA = sSFA_full(_, _, _, warp_group_idx); // 3D slice for this warp_group
        Tensor sSFB = make_tensor(make_smem_ptr(shared_storage.smem_q_sf.data()), SmemLayoutSFB{});
        Tensor sW = make_tensor(make_smem_ptr(shared_storage.smem_weight.data()), SmemLayoutWeight{});

        auto gmem_thr_copy_SFA = gmem_tiled_copy_SFA.get_thread_slice(local_thread_idx);
        auto gmem_thr_copy_SFB = gmem_tiled_copy_SFB.get_thread_slice(local_thread_idx);
        auto gmem_thr_copy_weight = gmem_tiled_copy_weight.get_thread_slice(local_thread_idx);

        Tensor tSFAgSFA = gmem_thr_copy_SFA.partition_S(gSFA);
        Tensor tSFAsSFA = gmem_thr_copy_SFA.partition_D(sSFA);

        Tensor tSFBgSFB = gmem_thr_copy_SFB.partition_S(gSFB);
        Tensor tSFBsSFB = gmem_thr_copy_SFB.partition_D(sSFB);

        Tensor tWgW = gmem_thr_copy_weight.partition_S(gW);
        Tensor tWsW = gmem_thr_copy_weight.partition_D(sW);

        TiledMma tiled_mma;
        auto full_accum = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
        constexpr bool CompactAccum = kGrouped && SPLIT_MBLOCK && MmaIterM % 2 == 0;
        auto accum = [&]() {
            if constexpr (CompactAccum)
                return make_tensor<ElementAcc>(make_shape(size<0>(full_accum), Int<1>{}, size<2>(full_accum)));
            else return full_accum;
        }();
        auto thr_mma = tiled_mma.get_thread_slice(local_thread_idx);
        Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
        Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));

        static_assert(CompactAccum || decltype(size<1>(tCrA))::value == decltype(size<1>(accum))::value);
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));

        auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
        auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(local_warp_idx * 32);
        Tensor tCsA = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));
        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
        CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));

        auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
        auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(local_warp_idx * 32);
        Tensor tCsB = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
        CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));

        // SFA s2r: recast smem as uint16_t, then apply transposed layout for MMA consumption
        Tensor sSFAUint16 = recast<uint16_t>(sSFA);
        using SFATransLayout = cute::conditional_t<kIsAvg,
            AvgSFTransLayout<WarpOnM, MmaIterM, kNumKVStages, BLOCK_M>, sSFATransLayout>;
        Tensor sSFATrans = make_tensor(sSFAUint16.data(), SFATransLayout{});

        SmemTiledCopySFA smem_tiled_copy_SFA;
        auto smem_thr_copy_SFA = smem_tiled_copy_SFA.get_slice(warp_m_idx * 32 + lane_idx);
        Tensor tCsSFA = smem_thr_copy_SFA.partition_S(sSFATrans);
        Tensor tCrSFA_stage = smem_thr_copy_SFA.partition_D(sSFATrans(_, _, 0));
        using SFARegType = decltype(make_fragment_like(tCrSFA_stage));
        SFARegType tCrSFA_copy_view;

        // SFB s2r: recast smem as uint16_t, then apply transposed layout for MMA consumption
        Tensor sSFBUint16 = recast<uint16_t>(sSFB);
        using SFBTransLayout = cute::conditional_t<kIsAvg,
            AvgSFTransLayout<WarpOnN, MmaIterN, kNumQStages, BLOCK_N>, sSFBTransLayout>;
        Tensor sSFBTrans = make_tensor(sSFBUint16.data(), SFBTransLayout{});

        SmemTiledCopySFB smem_tiled_copy_SFB;
        auto smem_thr_copy_SFB = smem_tiled_copy_SFB.get_slice(warp_n_idx * 32 + lane_idx);
        Tensor tCsSFB = smem_thr_copy_SFB.partition_S(sSFBTrans);
        Tensor tCrSFB_stage = smem_thr_copy_SFB.partition_D(sSFBTrans(_, _, 0));
        using SFBRegType = decltype(make_fragment_like(tCrSFB_stage));
        SFBRegType tCrSFB_copy_view;

        // FP4 scale as uint32_t for the 6-operand gemm
        Tensor tCrSFA = recast<uint32_t>(tCrSFA_copy_view);
        Tensor tCrSFB = recast<uint32_t>(tCrSFB_copy_view);

        constexpr bool enable_print = false;
        bool thread_print = enable_print && cute::thread(0, 0);

        // Scheduler
        using QIndex = uint32_t;
        auto scheduler = [&]() {
            if constexpr (kGrouped)
                return deep_gemm::FP4SingleGroupScheduler<BLOCK_KV, kNumMathWarpGroups, kNextN>(
                    only_q, kv_begin, kv_end, reinterpret_cast<const deep_gemm::FP4VarlenQBlock*>(params.grouped_meta)[only_q].context_len,
                    kReuseRequest ? 1u : kNextN / (kFixedQSpan ? StorageQSpan :
                        deep_gemm::fp4_varlen_q_span<kNextN>(reinterpret_cast<const deep_gemm::FP4VarlenQBlock*>(params.grouped_meta)[only_q].valid_q)));
            else
                return deep_gemm::PagedMQALogitsScheduler<BLOCK_KV, kNumMathWarpGroups, kNextN>(
                    params.batch_size, blockIdx.x, params.context_lens, params.schedule_meta);
        }();
        const auto group_desc = reinterpret_cast<const deep_gemm::FP4VarlenQBlock*>(params.grouped_meta);
        const uint32_t request_pack = [&]() {
            if constexpr (kGrouped && kFixedQSpan) return kNextN / StorageQSpan;
            else if constexpr (kGrouped) return kNextN / deep_gemm::fp4_varlen_q_span<kNextN>(group_desc[only_q].valid_q);
            else return 1u;
        }();
        const uint32_t pipeline_depth = kNumKVStages / request_pack;
        auto next_q_idx = [&](QIndex q) {
            if constexpr (kGrouped) return scheduler.next_q(q);
            else return q + 1;
        };
        DG_STATIC_ASSERT(SPLIT_KV % BLOCK_KV == 0, "Unaligned SPLIT_KV");

        constexpr int elem_weights = WARP_N / 4;
        ElementLogits weights[elem_weights];
        Tensor sW_copy = make_tensor(sW.data(), sWCopyLayout{});
        SmemTiledCopyWeights smem_tiled_copy_weights;
        auto smem_thr_copy_weights = smem_tiled_copy_weights.get_slice(warp_n_idx * 4 + lane_idx % 4);
        Tensor tCsW = smem_thr_copy_weights.partition_S(sW_copy);
        Tensor tCrW_copy_view = make_tensor_like(tCsW(_, _, _, 0));
        using WeightRegLayout = Layout<Shape<Shape<_2, _2, _2>, Int<MmaIterN>>, Stride<Stride<_1, _0, _2>, _4>>;
        Tensor tCrW = make_tensor(static_cast<ElementLogits*>(weights), WeightRegLayout{});

        clear(accum);
        auto tKgK = tAgA;
        auto tSFKgSFK = tSFAgSFA;
        auto tQgQ = tBgB;
        auto tSFQgSFQ = tSFBgSFB;
        auto tWgW_base = tWgW;

        const auto& warp_offset = warp_m_idx * WARP_M;
        const auto& v_0_offset = lane_idx / 4 + 0;
        const auto& v_1_offset = lane_idx / 4 + 8;
        const uint32_t r_mine = lane_idx & 1;
        const uint32_t v_offset_mine = r_mine ? v_1_offset : v_0_offset;

        QIndex q_idx = scheduler.current_q_idx;
        QIndex issued_q_idx = ~QIndex{0};
        uint32_t issued_table_row = 0, issued_context = 0;
        uint32_t issued_q_span = kNextN;
        uint32_t issued_q_shift = 0;
        uint32_t output_row_cached = 0;
        uint32_t output_context_cached = 0;
        bool output_valid_cached = false;
        uint32_t output_q_count_cached = 0, request_q_lane = 0;
        uint32_t request_valid = 0, request_source_q = 0;
        uint32_t output_q_span_cached = kNextN, read_kv_slot_cached = warp_group_idx;
        uint32_t kv_group_offset_cached = warp_group_idx;
        uint32_t kv_idx_base;
        uint32_t kv_idx_array[kNumKVStages];
        uint32_t smem_pipe_read_q = 0, smem_pipe_read_kv = 0;
        uint32_t smem_pipe_write_q = 0, smem_pipe_write_kv = 0;

        // Load Q (packed FP4) + q_sf (e8m0) + weights from gmem to smem
        auto load_q_g2s = [&](QIndex q_key) {
            const uint32_t q_idx = uint32_t(q_key);
            auto q_offset = (kGrouped ? group_desc[q_idx].source_q : q_idx * kNextN) * kNumHeads;
            if constexpr (kGrouped) {
                const uint32_t rows = group_desc[q_idx].valid_q * kNumHeads;
                gmem_tiled_copy_B.desc_.dim_h = rows;
                gmem_tiled_copy_SFB.desc_.dim_w = rows;
                gmem_tiled_copy_weight.desc_.dim_w = rows;
            }
            tBgB.data() = tQgQ.data() + q_offset * kHeadDim;
            tSFBgSFB.data() = tSFQgSFQ.data() + q_offset;
            copy_aiu(gmem_tiled_copy_B, tBgB(_, _, _, 0), tBsB(_, _, _, smem_pipe_write_q), warp_idx);
            if constexpr (kGrouped) {
                tWgW.data() = tWgW_base.data() + q_offset;
                copy_aiu(gmem_tiled_copy_weight, tWgW(_, _, _, 0), tWsW(_, _, _, smem_pipe_write_q), warp_idx);
                if (smem_pipe_write_q == 0)
                    copy_aiu(gmem_tiled_copy_SFB, tSFBgSFB(_, _, _, 0), tSFBsSFB(_, _, _, 0), warp_idx);
            } else if constexpr (kIsAvg) {
                // Quantization scales are still mandatory for FP4 AVG.
                copy_aiu(gmem_tiled_copy_SFB, tSFBgSFB(_, _, _, 0), tSFBsSFB(_, _, _, smem_pipe_write_q), warp_idx);
            } else {
                tWgW.data() = tWgW_base.data() + q_offset;
                copy_aiu<true>(gmem_tiled_copy_SFB, tSFBgSFB(_, _, _, 0), tSFBsSFB(_, _, _, smem_pipe_write_q),
                               gmem_tiled_copy_weight, tWgW(_, _, _, 0), tWsW(_, _, _, smem_pipe_write_q), warp_idx);
            }

            if (thread_print) {
                printf("  copy_q q_idx = %d, q_offset = %d, stage = %d\n", q_idx, q_offset, smem_pipe_write_q);
            }
            smem_pipe_write_q = (smem_pipe_write_q + 1) % kNumQStages;
        };

        // Load K (packed FP4) + k_sf (e8m0) from gmem to smem
        // Each warp_group loads into its own 3D smem slice, at stage = smem_pipe_write_kv
        auto prepare_kv_task = [&](QIndex q_key) {
            const uint32_t q_idx = uint32_t(q_key);
            if constexpr (kGrouped) {
                if (issued_q_idx != q_key) {
                    const auto d = group_desc[q_idx];
                    issued_table_row = d.table_row;
                    issued_context = d.context_len;
                    issued_q_span = kFixedQSpan ? StorageQSpan : deep_gemm::fp4_varlen_q_span<kNextN>(d.valid_q);
                    issued_q_shift = kFixedQSpan ? (StorageQSpan == 4 ? 2 : (StorageQSpan == 2 ? 1 : 0))
                        : deep_gemm::fp4_varlen_q_shift<kNextN>(d.valid_q);
                    issued_q_idx = q_key;
                }
            }
        };
        auto load_kv_g2s = [&](QIndex q_key, uint32_t kv_idx) {
            const uint32_t q_idx = uint32_t(q_key);
            const uint32_t table_row = kGrouped ? issued_table_row : q_idx;
            const uint32_t q_span = kGrouped ? issued_q_span : WarpOnN;
            const uint32_t kv_slot = warp_group_idx;
            const uint32_t packed_write_stage = kGrouped
                ? smem_pipe_write_kv * request_pack + (warp_n_idx >> issued_q_shift) : smem_pipe_write_kv;
            int issue_warp = kGrouped ? local_warp_idx & (q_span * WarpOnM - 1) : local_warp_idx;
            auto target_k = gmem_thr_copy_A.partition_D(sA_full(_, _, _, kv_slot));
            auto target_sf = gmem_thr_copy_SFA.partition_D(sSFA_full(_, _, _, kv_slot));
            if constexpr (PAGE_KV >= BLOCK_KV) {
                const uint32_t token = kv_idx * BLOCK_KV;
                const uint32_t row_in_page = token % PAGE_KV;
                auto kv_offset = __ldg(params.block_table + table_row * params.block_table_stride + token / PAGE_KV);
                tAgA.data() = tKgK.data() + kv_offset * params.kv_cache_stride_bytes + static_cast<uint64_t>(row_in_page) * BLOCK_K;
                tSFAgSFA.data() = tSFKgSFK.data() + kv_offset * params.kv_cache_stride_bytes / sizeof(uint32_t) + row_in_page;
                // Split the value/scale AIU copies across warps 0/1 only when the group
                // has >= 2 warps; otherwise warp 0 issues both
                constexpr bool SPLIT_AIU = (WarpOnGroup >= 2);
                if constexpr (kGrouped) {
                    copy_aiu(gmem_tiled_copy_A, tAgA(_, _, _, 0), target_k(_, _, _, packed_write_stage), issue_warp);
                    copy_aiu(gmem_tiled_copy_SFA, tSFAgSFA(_, _, _, 0), target_sf(_, _, _, packed_write_stage), issue_warp);
                } else {
                    copy_aiu<SPLIT_AIU>(gmem_tiled_copy_A, tAgA(_, _, _, 0), tAsA(_, _, _, packed_write_stage), gmem_tiled_copy_SFA,
                        tSFAgSFA(_, _, _, 0), tSFAsSFA(_, _, _, packed_write_stage), local_warp_idx);
                }
                if (thread_print) {
                    printf("  copy_k q_idx = %d, kv_idx = %d, kv_offset = %d, wg = %d, stage = %d\n", q_idx, kv_idx,
                        kv_offset, warp_group_idx, smem_pipe_write_kv);
                }
            } else {
                // Assemble one compute tile from page-local AIU copies.
                constexpr uint32_t kCopyRows = PAGE_KV < BLOCK_KV ? PAGE_KV : BLOCK_KV;
                using ChunkKV = DefaultOperandA;
                using ChunkSF = cutlass::gemm::config::DefaultGemm_AIU_Operand<
                    cutlass::arch::PPU0015, uint32_t, false, _1, Int<kCopyRows>, false, 0, false>;
                using ChunkKVCopy = typename ChunkKV::CopyInst;
                using ChunkSFCopy = typename ChunkSF::CopyInst;
                const uint32_t context_len = PAGE_KV < BLOCK_KV
                    ? (kGrouped ? issued_context : __ldg(params.context_lens + q_idx * kNextN + kNextN - 1)) : 0;
                const uint32_t first_page = kv_idx * BLOCK_KV / PAGE_KV;
                CUTE_UNROLL
                for (uint32_t row = 0; row < BLOCK_KV; row += kCopyRows) {
                    const uint32_t token = kv_idx * BLOCK_KV + row;
                    const bool valid_page = PAGE_KV >= BLOCK_KV || token < context_len;
                    const uint32_t logical_page = valid_page ? token / PAGE_KV : first_page;
                    const uint32_t physical_page = __ldg(params.block_table + table_row * params.block_table_stride + logical_page);
                    const uint32_t row_in_page = valid_page ? token % PAGE_KV : 0;
                    cute::AiuDesc kv_desc;
                    kv_desc.template init<ElementQK, false, kCopyRows, BLOCK_K>(
                        nullptr, kCopyRows, BLOCK_K, StrideAB{});
                    if (issue_warp == 0 && lane_idx == 0) {
                        auto src = reinterpret_cast<const ElementQK*>(
                            reinterpret_cast<const uint8_t*>(params.ptr_k) + physical_page * params.kv_cache_stride_bytes) + static_cast<uint64_t>(row_in_page) * BLOCK_K;
                        CUTE_UNROLL
                        for (int k = 0; k < BLOCK_K; k += ChunkKV::CUBE_W) {
                            auto dst_k = shared_storage.smem_k.data() + SmemLayoutA{}(make_coord(row, k, packed_write_stage, kv_slot));
                            ChunkKVCopy::copy(dst_k, src, kv_desc, k, 0, 0);
                        }
                    }
                    cute::AiuDesc sf_desc;
                    sf_desc.template init<uint32_t, false, 1, kCopyRows>(
                        nullptr, 1, kCopyRows, Stride<Int<kCopyRows>, _1>{});
                    constexpr uint32_t kScaleWarp = !kGrouped && WarpOnGroup >= 2 ? 1 : 0;
                    if (issue_warp == kScaleWarp && lane_idx == 0) {
                        auto dst = shared_storage.smem_k_sf.data() + SmemLayoutSFA{}(make_coord(0, row, packed_write_stage, kv_slot));
                        auto src = reinterpret_cast<const uint32_t*>(
                            reinterpret_cast<const uint8_t*>(params.k_sf) + physical_page * params.kv_cache_stride_bytes) + row_in_page;
                        CUTE_UNROLL
                        for (int k = 0; k < kCopyRows; k += ChunkSF::CUBE_W)
                            ChunkSFCopy::copy(dst + k, src, sf_desc, k, 0, 0);
                    }
                }
            }
        };

        // Load Q from smem to vreg + read weights
        auto load_q_s2r = [&]() {
            const uint32_t q_lane = kReuseRequest ? request_q_lane : (FoldSingleQ ? 0 : (kGrouped ? warp_n_idx & (output_q_span_cached - 1) : warp_n_idx));
            auto q_copy = smem_tiled_copy_B.get_thread_slice((q_lane * WarpOnM + warp_m_idx) * 32);
            auto q_source = q_copy.partition_S(make_mix_tensor_like(sB));
            auto sf_copy = smem_tiled_copy_SFB.get_slice(q_lane * 32 + lane_idx);
            auto sf_source = sf_copy.partition_S(sSFBTrans);
            const uint32_t source_q = kGrouped ? (kStaticRequest ? request_source_q : group_desc[q_idx].source_q) : 0;
            copy(smem_tiled_copy_B, q_source(_, _, _, smem_pipe_read_q), tCrB_copy_view);
            if constexpr (kGrouped) {
                if (smem_pipe_read_q == 0) {
                    copy(smem_tiled_copy_SFB, sf_source(_, _, _, 0), tCrSFB_copy_view);
                } else {
                    auto ptr = reinterpret_cast<const uint16_t*>(params.q_sf + source_q * kNumHeads);
                    // Repeat unused MMA selectors within each query's scale rows.
                    using GmemSFBTransLayout = AvgSFTransLayout<WarpOnN, MmaIterN, kNumQStages, BLOCK_N>;
                    auto global_sf = make_tensor(make_gmem_ptr(ptr), GmemSFBTransLayout{});
                    auto global_source = sf_copy.partition_S(global_sf);
                    if (output_valid_cached)
                        copy(smem_tiled_copy_SFB, global_source(_, _, _, 0), tCrSFB_copy_view);
                    else clear(tCrSFB_copy_view);
                }
            } else {
                copy(smem_tiled_copy_SFB, sf_source(_, _, _, smem_pipe_read_q), tCrSFB_copy_view);
            }
            if constexpr (!kIsAvg) {
                auto w_copy = smem_tiled_copy_weights.get_slice(q_lane * 4 + lane_idx % 4);
                auto w_source = w_copy.partition_S(sW_copy);
                copy(smem_tiled_copy_weights, w_source(_, _, _, smem_pipe_read_q), tCrW_copy_view);
                deep_gemm::load_weights_from_copy_view<ElementWeights, ElementLogits>(
                    tCrW_copy_view, weights, elem_weights);
            }

            if (thread_print) {
                printf("    copy q to vreg, q_stage_idx = %d,\n", smem_pipe_read_q);
            }
            smem_pipe_read_q = (smem_pipe_read_q + 1) % kNumQStages;
        };

        auto load_next_qk_g2s = [&](bool load_q) {
            QIndex q_idx;
            uint32_t kv_idx_base, num_kv;
            if (scheduler.fetch_next_task(q_idx, kv_idx_base, num_kv)) {
                if constexpr (!kReuseRequest) { if (load_q) load_q_g2s(q_idx); }
                prepare_kv_task(q_idx);
                uint32_t actual_kv = kv_idx_base + warp_group_idx;
                if constexpr (kGrouped) {
                    actual_kv += (warp_n_idx >> issued_q_shift) * kNumMathWarpGroups;
                }
                if (actual_kv < num_kv)
                    load_kv_g2s(q_idx, actual_kv);
            }
            if constexpr (!kGrouped) kv_idx_array[smem_pipe_write_kv] = kv_idx_base;
            smem_pipe_write_kv = (smem_pipe_write_kv + 1) % (kGrouped ? pipeline_depth : kNumKVStages);
            if (thread_print)
                printf("cp_async commit, write_stage = %d\n", smem_pipe_write_kv);
            cp_async_fence();
        };

        // Load KV (A + SFA) from smem to vreg — all M blocks for one k_block (float path)
        auto load_kv_s2r = [&](int k_block) {
            auto source_a = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA_full(_, _, _, read_kv_slot_cached)));
            auto sf = recast<uint16_t>(sSFA_full(_, _, _, read_kv_slot_cached));
            auto source_sf = smem_thr_copy_SFA.partition_S(make_tensor(sf.data(), SFATransLayout{}));
            copy(smem_tiled_copy_A, source_a(_, _, k_block, (kReuseRequest ? smem_pipe_read_kv : smem_pipe_read_kv * request_pack + (kGrouped ? warp_n_idx / output_q_span_cached : 0))), tCrA_copy_view(_, _, k_block));
            copy(smem_tiled_copy_SFA, source_sf(_, _, k_block, (kReuseRequest ? smem_pipe_read_kv : smem_pipe_read_kv * request_pack + (kGrouped ? warp_n_idx / output_q_span_cached : 0))), tCrSFA_copy_view(_, _, k_block));
        };

        // Load KV (A / SFA) from smem to vreg — single m_block for one k_block (bf16 path)
        auto load_kv_a_s2r_mblock = [&](int k_block, int m_block) {
            auto source = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA_full(_, _, _, read_kv_slot_cached)));
            copy(smem_tiled_copy_A, source(_, m_block, k_block, (kReuseRequest ? smem_pipe_read_kv : smem_pipe_read_kv * request_pack + (kGrouped ? warp_n_idx / output_q_span_cached : 0))), tCrA_copy_view(_, m_block, k_block));
        };

        auto load_kv_sfa_s2r_mblock = [&](int k_block, int m_block) {
            auto sf = recast<uint16_t>(sSFA_full(_, _, _, read_kv_slot_cached));
            auto source = smem_thr_copy_SFA.partition_S(make_tensor(sf.data(), SFATransLayout{}));
            copy(smem_tiled_copy_SFA, source(_, m_block, k_block, (kReuseRequest ? smem_pipe_read_kv : smem_pipe_read_kv * request_pack + (kGrouped ? warp_n_idx / output_q_span_cached : 0))), tCrSFA_copy_view(_, m_block, k_block));
        };

        // Reuse each lane's Q-row output bases across the KV tile's M blocks.
        static constexpr bool kReuseAvgLogitsBase = kIsAvg && !SPLIT_MBLOCK;
        static constexpr int kAvgOutputsPerLane = kNumHeads == 8 ? 1 : 2;
        ElementLogits* avg_logits_bases[kAvgOutputsPerLane];
        auto prepare_avg_logits_tile = [&](uint32_t current_q_idx, uint32_t current_kv_idx) {
            if constexpr (kReuseAvgLogitsBase) {
                #pragma unroll
                for (int t = 0; t < kAvgOutputsPerLane; ++t) {
                    const uint32_t token = warp_n_idx * kQueriesPerWarp +
                        (kNumHeads == 8 ? (lane_idx % 4) / 2 : 2 * t + (lane_idx % 4) / 2);
                    const bool valid_token = token < kNextN;
                    // Padded Q lanes retain an in-allocation base; the original
                    // output guard remains below, after the full-mask shuffle.
                    const uint64_t row = valid_token ? uint64_t(current_q_idx) * kNextN + token : 0;
                    const uint64_t kv0 = valid_token
                        ? uint64_t(current_kv_idx) * BLOCK_KV + warp_offset + v_offset_mine : 0;
                    avg_logits_bases[t] = params.logits + row * params.logits_stride + kv0;
                }
            }
        };

        auto epilogue_mblock = [&](QIndex q_key, int kv_idx, int m_block) {
            const uint32_t q_idx = uint32_t(q_key);
            // Reduce over heads and store logits
            static constexpr int kNumAccumPerMma = size<0>(accum);

            int m = m_block;
            if constexpr (kIsAvg && kNumHeads == 8) {
                CUTE_STATIC_ASSERT(MmaIterN == 1 && kQueriesPerWarp == 2);
                const uint32_t token = warp_n_idx * kQueriesPerWarp + (lane_idx % 4) / 2;
                static_assert(cute::is_same_v<ElementLogits, __ppu_bfloat16>, "H8 AVG requires BF16 output");
                const ElementLogits value = __hmul(deep_gemm::bf16_epilogue_reduce_wq2(accum, m, lane_idx),
                                                   (__ppu_bfloat16)kInvSqrtHeadDim);
                // The quad owns two queries x two KV rows, exactly once each.
                // Keep the full-mask head reduction above the padded-query guard.
                if (token < kNextN) {
                    if constexpr (kReuseAvgLogitsBase) {
                        avg_logits_bases[0][m * InstM] = value;
                    } else {
                        const auto logits_q_offset = (q_idx * kNextN + token) * params.logits_stride;
                        const uint32_t logits_kv_offset = kv_idx * BLOCK_KV + warp_offset + m * InstM + v_offset_mine;
                        params.logits[logits_q_offset + logits_kv_offset] = value;
                    }
                }
            } else if constexpr (kIsAvg) {
                CUTE_STATIC_ASSERT(MmaIterN == 1 && kQueriesPerWarp == 4);
                const uint32_t lane_in_quad = lane_idx % 4;
                const uint32_t token_base = warp_n_idx * kQueriesPerWarp;
                const uint32_t logits_kv_offset = kv_idx * BLOCK_KV + warp_offset + m * InstM + v_offset_mine;
                if constexpr (cute::is_same_v<ElementLogits, float>) {
                    float sums[2][2];
                    deep_gemm::float_epilogue_reduce_wq4(accum, m, sums);
                    #pragma unroll
                    for (int t = 0; t < 2; ++t) {
                        const float recv = __shfl_xor_sync(0xffffffffu, sums[t][r_mine ^ 1], 1);
                        const float total = sums[t][r_mine] + recv;
                        const uint32_t token = token_base + 2 * t + lane_in_quad / 2;
                        if (token < kNextN) {
                            if constexpr (kReuseAvgLogitsBase) {
                                avg_logits_bases[t][m * InstM] = total * kInvSqrtHeadDim;
                            } else {
                                const auto logits_q_offset = (q_idx * kNextN + token) * params.logits_stride;
                                params.logits[logits_q_offset + logits_kv_offset] = static_cast<ElementLogits>(total * kInvSqrtHeadDim);
                            }
                        }
                    }
                } else {
                    // Match FP8 AVG's BF16 rounding order. FP4 MMA has already
                    // applied Q/K block scales; only 1/sqrt(head_dim) remains.
                    #pragma unroll
                    for (int t = 0; t < 2; ++t) {
                        uint32_t pa = deep_gemm::tix::cvt_rte_bf16x2_f32_relu(
                            accum(4 * t + 2, m, 0), accum(4 * t + 0, m, 0));
                        uint32_t pb = deep_gemm::tix::cvt_rte_bf16x2_f32_relu(
                            accum(4 * t + 3, m, 0), accum(4 * t + 1, m, 0));
                        const __ppu_bfloat162 partial = __hadd2(
                            reinterpret_cast<__ppu_bfloat162&>(pa),
                            reinterpret_cast<__ppu_bfloat162&>(pb));
                        const uint32_t recv = __shfl_xor_sync(
                            0xffffffffu, reinterpret_cast<const uint32_t&>(partial), 1);
                        const __ppu_bfloat162 total2 = __hadd2(
                            partial, reinterpret_cast<const __ppu_bfloat162&>(recv));
                        const __ppu_bfloat16 total =
                            r_mine ? __high2bfloat16(total2) : __low2bfloat16(total2);
                        const uint32_t token = token_base + 2 * t + lane_in_quad / 2;
                        if (token < kNextN) {
                            if constexpr (kReuseAvgLogitsBase) {
                                avg_logits_bases[t][m * InstM] =
                                    __hmul(total, (__ppu_bfloat16)kInvSqrtHeadDim);
                            } else {
                                const auto logits_q_offset = (q_idx * kNextN + token) * params.logits_stride;
                                params.logits[logits_q_offset + logits_kv_offset] =
                                    __hmul(total, (__ppu_bfloat16)kInvSqrtHeadDim);
                            }
                        }
                    }
                }
            } else {
                CUTE_STATIC_ASSERT(kNumHeads % 8 == 0);
                if constexpr (kGrouped) {
                    if (!output_valid_cached) return;
                }
                int mma_offset = m * InstM;
                const uint32_t output_row = kGrouped ? output_row_cached : q_idx * kNextN + warp_n_idx;
                auto logits_q_offset = uint64_t(output_row) * params.logits_stride;
                auto logits_kv_offset = kv_idx * BLOCK_KV + warp_offset + mma_offset;

                if constexpr (cute::is_same_v<ElementLogits, float>) {
                    float v_0, v_1;
                    deep_gemm::float_epilogue_reduce_tCrW(accum, m, tCrW, v_0, v_1);

                    // Inter-thread reduction
                    deep_gemm::shfl_xor_reduce_2(v_0, v_1);
                    params.logits[logits_q_offset + logits_kv_offset + v_0_offset] = v_0;
                    params.logits[logits_q_offset + logits_kv_offset + v_1_offset] = v_1;
                } else {
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
                    // Intra-thread reduction in bfloat16
                    __ppu_bfloat16 v_0_bf = __hadd(__low2bfloat16(sum_0), __high2bfloat16(sum_0));
                    __ppu_bfloat16 v_1_bf = __hadd(__low2bfloat16(sum_1), __high2bfloat16(sum_1));

                    // Cross-lane reduction: pack v_0/v_1 into bfloat162 for single 32-bit shuffle
                    __ppu_bfloat162 packed = deep_gemm::shfl_xor_reduce_bf16x2({v_0_bf, v_1_bf});
                    if constexpr (kGrouped) {
                        // A quad has identical reduced values. Lanes 0/1 own
                        // the low/high KV row; lanes 2/3 do not duplicate stores.
                        const uint32_t column = logits_kv_offset + v_offset_mine;
                        if ((lane_idx & 2) == 0 && column < output_context_cached)
                            params.logits[logits_q_offset + column] = r_mine
                                ? __high2bfloat16(packed) : __low2bfloat16(packed);
                    } else {
                        params.logits[logits_q_offset + logits_kv_offset + v_0_offset] = __low2bfloat16(packed);
                        params.logits[logits_q_offset + logits_kv_offset + v_1_offset] = __high2bfloat16(packed);
                    }
                }
            }
        };

        auto reduce_grouped_mblock = [&](auto m) {
            constexpr int Transforms = 4 * size<2>(accum);
            uint32_t converted[Transforms];
            deep_gemm::cvt_accum_to_bf16x2_buf<Transforms>(accum, m, converted);
            __ppu_sched_bound();
            __ppu_bfloat162 sum0 = {0, 0}, sum1 = {0, 0};
            deep_gemm::fma2_phase_tCrW<Transforms>(converted, tCrW, sum0, sum1);
            const __ppu_bfloat16 v0 = __hadd(__low2bfloat16(sum0), __high2bfloat16(sum0));
            const __ppu_bfloat16 v1 = __hadd(__low2bfloat16(sum1), __high2bfloat16(sum1));
            return deep_gemm::shfl_xor_reduce_bf16x2({v0, v1});
        };

        auto store_reduced_grouped_pair = [&](uint32_t kv, auto pair,
                __ppu_bfloat162 low, __ppu_bfloat162 high) {
            if (!output_valid_cached) return;
            const uint32_t base_column = kv * BLOCK_KV + warp_offset;
            const bool full = base_column + WARP_M <= output_context_cached;
            const uint64_t row = uint64_t(output_row_cached) * params.logits_stride;
            const auto packed = (lane_idx & 2) ? high : low;
            const auto value = r_mine ? __high2bfloat16(packed) : __low2bfloat16(packed);
            const uint32_t column = base_column + (2 * pair) * InstM +
                ((lane_idx & 2) ? InstM : 0) + v_offset_mine;
            if (full) params.logits[row + column] = value;
            else if (column < output_context_cached) params.logits[row + column] = value;
        };
        auto store_grouped_pair = [&](uint32_t kv, auto pair) {
            if (!output_valid_cached) return;
            const auto low = reduce_grouped_mblock(Int<2 * pair>{});
            const auto high = reduce_grouped_mblock(Int<2 * pair + 1>{});
            store_reduced_grouped_pair(kv, pair, low, high);
        };
        auto store_grouped_pairs = [&](uint32_t kv) {
            for_each(make_int_sequence<size<1>(accum) / 2>{}, [&](auto pair) {
                store_grouped_pair(kv, pair);
            });
        };

        auto issue_request_kv = [&](uint32_t kv, uint32_t slot) {
            prepare_kv_task(only_q);
            smem_pipe_write_kv = slot;
            const uint32_t actual = kv + warp_group_idx;
            if (kv < kv_end && actual < scheduler.current_num_kv) load_kv_g2s(only_q, actual);
            cp_async_fence();
        };
        // A request window keeps two Q blocks resident while its KV is streamed once.
        if constexpr (kReuseRequest) {
            for (uint32_t q = only_q; q < last_q; ++q) load_q_g2s(q);
        }
        if constexpr (kReuseRequest) {
            issue_request_kv(kv_begin, 0);
            issue_request_kv(kv_begin + 3 * kNumMathWarpGroups, 3);
            issue_request_kv(kv_begin + kNumMathWarpGroups, 1);
            issue_request_kv(kv_begin + 2 * kNumMathWarpGroups, 2);
        } else {
            for (int i = 0; i < (pipeline_depth == 1 ? 1 : pipeline_depth - 1); i++)
                load_next_qk_g2s(i == 0);
        }
        if constexpr (kGrouped) kv_idx_base = kv_begin;
        else kv_idx_base = kv_idx_array[0];
        if constexpr (kGrouped) q_idx = only_q;
        // wait AIU Q and first K
        if constexpr (kGrouped) {
            if constexpr (kFixedQSpan != 0) {
                constexpr int Depth = kNumKVStages / (kNextN / StorageQSpan);
                if constexpr (kReuseRequest) cp_async_wait<kNumKVStages - 2>();
                else cp_async_wait<(Depth > 1 ? Depth - 2 : 0)>();
            } else cp_async_wait<0>();
        } else cp_async_wait<(kNumKVStages > 1 ? kNumKVStages - 2 : 0)>();
        __syncthreads();

        deep_gemm::warp_interleave_start<WarpInterleaving>(warp_group_id, MaxThreadsPerBlock);

        auto handle_q_change = [&]() {
            if constexpr (kGrouped) {
                const uint32_t q = uint32_t(q_idx);
                output_q_count_cached = kStaticRequest ? request_valid : group_desc[q].valid_q;
                output_q_span_cached = kReuseRequest ? deep_gemm::fp4_varlen_q_span<kNextN>(output_q_count_cached)
                    : (kFixedQSpan ? StorageQSpan : deep_gemm::fp4_varlen_q_span<kNextN>(output_q_count_cached));
                const uint32_t q_shift = kFixedQSpan ? (StorageQSpan == 4 ? 2 : (StorageQSpan == 2 ? 1 : 0))
                    : deep_gemm::fp4_varlen_q_shift<kNextN>(output_q_count_cached);
                const uint32_t q_lane = kReuseRequest ? request_q_lane : (FoldSingleQ ? 0 : warp_n_idx & (output_q_span_cached - 1));
                output_row_cached = (kStaticRequest ? request_source_q : group_desc[q].source_q) + q_lane;
                output_valid_cached = q_lane < output_q_count_cached;
                kv_group_offset_cached = warp_group_idx + (kReuseRequest ? 0 : warp_n_idx >> q_shift) * kNumMathWarpGroups;
                read_kv_slot_cached = warp_group_idx;
                output_context_cached = output_valid_cached
                    ? __ldg(params.context_lens + output_row_cached) : 0;
            }
            if constexpr (kNumQStages > 1 && !kReuseRequest) {
                if (scheduler.exist_q_idx(next_q_idx(q_idx))) {
                    load_q_g2s(next_q_idx(q_idx));
                }
                cp_async_fence();
            }
            load_q_s2r();
            if constexpr (!kGrouped && kNumQStages == 1) { // next-Q overwrite needs readers to finish
                __syncthreads();
                if (scheduler.exist_q_idx(next_q_idx(q_idx))) {
                    load_q_g2s(next_q_idx(q_idx));
                }
                cp_async_fence();
            }
        };

        auto compute_current_q = [&](auto recycle_kv) {
            // s2r KV + compute GEMM from current read stage
            constexpr int K_BLOCK_MAX = size<2>(tCrA);
            constexpr int M_BLOCK = CompactAccum ? size<1>(tCrA) : size<1>(accum);
            constexpr int N_BLOCK = size<2>(accum);
            uint32_t actual_kv = kv_idx_base + kv_group_offset_cached;
            if constexpr (not SPLIT_MBLOCK) {
                for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
                    load_kv_s2r(k_block);
                    cute::gemm(tiled_mma, accum, tCrA(_, _, k_block), tCrSFA(_, _, k_block), tCrB(_, _, k_block),
                               tCrSFB(_, _, k_block), accum);
                });
                if constexpr (kReuseRequest) recycle_kv();
                prepare_avg_logits_tile(q_idx, actual_kv);
                deep_gemm::warp_interleave_arrive<WarpInterleaving>(warp_group_id, MaxThreadsPerBlock);
                if constexpr (kGrouped && (M_BLOCK % 2 == 0)) {
                    store_grouped_pairs(actual_kv);
                } else {
                    #pragma unroll
                    for (int m_block = 0; m_block < size<1>(accum); m_block++) {
                        epilogue_mblock(q_idx, actual_kv, m_block);
                    }
                }
            } else if constexpr (kGrouped && !WarpInterleaving && (M_BLOCK % 2 == 0)) {
                constexpr int m_group = M_BLOCK / size<1>(tCrSFA);
                constexpr int n_group = N_BLOCK / size<1>(tCrSFB);
                constexpr bool ReusePackedSFA = M_BLOCK == 4 && size<1>(tCrSFA) == 1;
                if constexpr (ReusePackedSFA) {
                    for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
                        load_kv_sfa_s2r_mblock(k_block, 0);
                    });
                }
                for_each(make_int_sequence<FoldSingleQ ? 1 : M_BLOCK / 2>{}, [&](auto pair) {
                    auto compute_reduce_mblock = [&](auto m_block) {
                        clear(accum);
                        for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
                            load_kv_a_s2r_mblock(k_block, m_block);
                            if constexpr (!ReusePackedSFA) load_kv_sfa_s2r_mblock(k_block, m_block);
                            for_each(make_int_sequence<N_BLOCK>{}, [&](auto n_block) {
                                Tensor scales = make_tensor<uint32_t>(Int<4>{});
                                scales[0] = tCrSFA(_, _, k_block)[m_block / m_group];
                                scales[1] = tCrSFB(_, _, k_block)[n_block / n_group];
                                scales[2] = m_block % m_group;
                                scales[3] = n_block % n_group;
                                MMA_Atom<MmaInst> atom;
                                Tensor d = accum(_, 0, n_block);
                                Tensor a = tCrA(_, m_block, k_block);
                                Tensor b = tCrB(_, n_block, k_block);
                                Tensor c = accum(_, 0, n_block);
                                cute::mma_unpack(atom, d, a, b, c, scales);
                            });
                        });
                        // The last MMA has consumed shared KV and scales. Let
                        // the next load overlap reduction and output stores.
                        if constexpr (kReuseRequest && decltype(m_block)::value == M_BLOCK - 1)
                            recycle_kv();
                        return reduce_grouped_mblock(Int<0>{});
                    };
                    if constexpr (FoldSingleQ) {
                        // Both halves use the same Q; their M coordinates are warp-local.
                        if (warp_n_idx & 1) {
                            const auto low = compute_reduce_mblock(Int<2>{});
                            const auto high = compute_reduce_mblock(Int<3>{});
                            store_reduced_grouped_pair(actual_kv, Int<1>{}, low, high);
                        } else {
                            const auto low = compute_reduce_mblock(Int<0>{});
                            const auto high = compute_reduce_mblock(Int<1>{});
                            store_reduced_grouped_pair(actual_kv, Int<0>{}, low, high);
                        }
                    } else {
                        const auto low = compute_reduce_mblock(Int<2 * pair>{});
                        const auto high = compute_reduce_mblock(Int<2 * pair + 1>{});
                        store_reduced_grouped_pair(actual_kv, pair, low, high);
                    }
                });
            } else { // Interleaved mma and epilogue
                constexpr int m_group = M_BLOCK / size<1>(tCrSFA);
                constexpr int n_group = N_BLOCK / size<1>(tCrSFB);
                constexpr bool kReusePackedSFA = M_BLOCK == 4 and size<1>(tCrSFA) == 1;
                for_each(make_int_sequence<M_BLOCK>{}, [&](auto m_block) {
                    if constexpr(m_block > 0) deep_gemm::warp_interleave_sync<WarpInterleaving>(warp_group_id, MaxThreadsPerBlock);
                    for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
                        load_kv_a_s2r_mblock(k_block, m_block);
                        if constexpr (!kReusePackedSFA or m_block == 0) {
                            load_kv_sfa_s2r_mblock(k_block, m_block);
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
                    deep_gemm::warp_interleave_arrive<WarpInterleaving>(warp_group_id, MaxThreadsPerBlock);
                    epilogue_mblock(q_idx, actual_kv, m_block);
                });
            }

            if constexpr (kReuseRequest) clear(accum);
        };

        auto request_kv_sync = [&]() {
            if constexpr (kReuseRequest && kNumQStages == 2 && kNumHeads == 32)
                cutlass::arch::NamedBarrier::sync(WarpOnGroup * 32, warp_group_idx);
            else __syncthreads();
        };
        if constexpr (kReuseRequest) {
            // Static counts unroll common request shapes; the same traversal
            // also handles arbitrary run lengths through runtime counts.
            auto for_each_index = [&](auto count, auto body) {
                if constexpr (std::is_integral_v<decltype(count)>) {
                    for (uint32_t i = 0; i < uint32_t(count); ++i) body(i);
                } else {
                    for_each(make_int_sequence<decltype(count)::value>{}, body);
                }
            };
            const uint32_t source0 = group_desc[only_q].source_q;
            uint32_t cluster = kv_begin, rotation = 0;
            while (cluster < kv_end) {
                auto process_q_block = [&](auto qi, auto valid_count, auto blocks) {
                    constexpr bool StaticCount = !std::is_integral_v<decltype(valid_count)>;
                    constexpr uint32_t StaticValid = [&]() {
                        if constexpr (StaticCount) return uint32_t(decltype(valid_count)::value);
                        else return 0u;
                    }();
                    constexpr uint32_t StaticSpan = StaticValid <= 1 ? 1 : (StaticValid == 2 ? 2 : kNextN);
                    const uint32_t valid = uint32_t(valid_count);
                    auto span = [&]() {
                        if constexpr (StaticCount) return Int<StaticSpan>{};
                        else return deep_gemm::fp4_varlen_q_span<kNextN>(valid);
                    }();
                    const bool steal = kNextN == 4 && valid == 3;
                    const uint32_t pack = kNextN / uint32_t(span);
                    auto steps = [&]() {
                        if constexpr (StaticCount)
                            return Int<(kNextN == 4 && StaticValid == 3 ? 3 : 4 / (kNextN / StaticSpan))>{};
                        else return steal ? 3u : 4u / pack;
                    }();
                    q_idx = only_q + uint32_t(qi);
                    request_valid = valid;
                    request_source_q = source0 + (uint32_t(qi) == 0 ? 0 : kRequestValid0);
                    request_q_lane = steal && warp_n_idx == 3 ? 0 : warp_n_idx & (uint32_t(span) - 1);
                    smem_pipe_read_q = uint32_t(qi);
                    handle_q_change();
                    for_each_index(steps, [&](auto step_index) {
                        const uint32_t step = uint32_t(step_index);
                        // All warps skip empty KV steps together, including their
                        // waits and MMA. A partial cluster is the final cluster.
                        if (cluster + step * pack * kNumMathWarpGroups >= kv_end)
                            return;
                        if (uint32_t(qi) == 0) {
                            if (!steal && pack >= 4) cp_async_wait<0>();
                            else if (!steal && pack == 2) {
                                if (step == 0) cp_async_wait<1>();
                                else cp_async_wait<0>();
                            } else {
                                if (step == 0) cp_async_wait<2>();
                                else if (step == 1) cp_async_wait<1>();
                                else cp_async_wait<0>();
                            }
                            request_kv_sync();
                        }
                        if (steal && warp_n_idx == 3 && step) {
                            request_q_lane = step;
                            smem_pipe_read_q = uint32_t(qi);
                            handle_q_change();
                        }
                        const uint32_t offset = steal ? (warp_n_idx == 3 ? 3 : step)
                            : step * pack + warp_n_idx / uint32_t(span);
                        kv_idx_base = cluster + offset * kNumMathWarpGroups;
                        smem_pipe_read_kv = (offset + rotation) & 3;
                        output_valid_cached = request_q_lane < valid && kv_idx_base < kv_end;
                        auto recycle_kv = [&]() {
                            // A slot remains live until every Q block using it
                            // has finished its MMA, including the Q3 stealing warp.
                            if (uint32_t(qi) + 1 == uint32_t(blocks)) {
                                request_kv_sync();
                                auto free_count = [&]() {
                                    if constexpr (StaticCount && !std::is_integral_v<decltype(step_index)>) {
                                        constexpr uint32_t Step = decltype(step_index)::value;
                                        return Int<(kNextN == 4 && StaticValid == 3 ? (Step == 2 ? 2 : 1)
                                            : kNextN / StaticSpan)>{};
                                    } else return steal ? (step == 2 ? 2u : 1u) : pack;
                                }();
                                const uint32_t first_free = steal ? step : step * pack;
                                for_each_index(free_count, [&](auto i) {
                                    const uint32_t free = first_free + uint32_t(i);
                                    issue_request_kv(cluster + (4 + ((free + 3) & 3)) * kNumMathWarpGroups,
                                        (free + rotation) & 3);
                                });
                            }
                        };
                        compute_current_q(recycle_kv);
                    });
                    request_kv_sync();
                };
                if constexpr (kStaticRequest) {
                    constexpr uint32_t Blocks = kRequestValid1 ? 2 : 1;
                    for_each_index(Int<Blocks>{}, [&](auto qi) {
                        constexpr uint32_t Valid = qi == 0 ? kRequestValid0 : kRequestValid1;
                        process_q_block(qi, Int<Valid>{}, Int<Blocks>{});
                    });
                } else {
                    const uint32_t blocks = last_q - only_q;
                    for (uint32_t qi = 0; qi < blocks; ++qi)
                        process_q_block(qi, group_desc[only_q + qi].valid_q, blocks);
                }
                cluster += 4 * kNumMathWarpGroups;
                rotation = (rotation + 1) & 3;
            }
            cp_async_wait<0>();
            __syncthreads();
            return;
        }

        // Make `first` a compile-time constant via std::integral_constant<bool, ...>
        // so the compiler can fold the `first || kv_idx_base == 0` branch and avoid
        // the runtime check on the first iteration.
        auto mainloop = [&](auto first_ic) {
            constexpr bool first = decltype(first_ic)::value;
            if (thread_print) {
                printf("q_idx = %d, kv_idx_base = %d, wg = %d, read_stage = %d\n", q_idx, kv_idx_base, warp_group_idx, smem_pipe_read_kv);
            }

            deep_gemm::warp_interleave_sync<WarpInterleaving>(warp_group_id, MaxThreadsPerBlock);

            // Handle Q change: on the first iteration always load the next Q;
            // afterwards only when kv_idx_base == 0 (first group of new q).
            if constexpr (first) {
                handle_q_change();
            } else {
                if constexpr (!kGrouped)
                    if (kv_idx_base == 0) handle_q_change();
            }

            // Issue next KV load (overlap with current compute)
            if (pipeline_depth > 1) load_next_qk_g2s(false);

            compute_current_q([]() {});

            if (pipeline_depth == 1) {
                __syncthreads();
                load_next_qk_g2s(false);
            }
            if constexpr (kGrouped) {
                if constexpr (kFixedQSpan != 0) {
                    constexpr int Depth = kNumKVStages / (kNextN / StorageQSpan);
                    cp_async_wait<(Depth > 1 ? Depth - 2 : 0)>();
                } else cp_async_wait<0>();
            } else cp_async_wait<(kNumKVStages > 1 ? kNumKVStages - 2 : 0)>();
            if (!WarpInterleaving) {
                __syncthreads();
            }
            // move to next kv_block
            smem_pipe_read_kv = (smem_pipe_read_kv + 1) % (kGrouped ? pipeline_depth : kNumKVStages);
            if constexpr (kGrouped) kv_idx_base += kNumMathWarpGroups * request_pack;
            else kv_idx_base = kv_idx_array[smem_pipe_read_kv];
            if constexpr (kGrouped) q_idx = only_q;
            else if (kv_idx_base == 0) q_idx = next_q_idx(q_idx);
            clear(accum);
        };

        if (!scheduler.is_last_task(q_idx, kv_idx_base)) {
            mainloop(cute::true_type{});
        }
        while (!scheduler.is_last_task(q_idx, kv_idx_base)) {
            mainloop(cute::false_type{});
        }

        deep_gemm::warp_interleave_end<WarpInterleaving>(warp_group_id, MaxThreadsPerBlock);

        if constexpr (!kGrouped || WarpInterleaving) {
            cp_async_wait<0>();
            __syncthreads();
        }
    }
};

template <typename Q, typename Acc, typename Out, typename W, uint32_t BQ, uint32_t H,
          uint32_t D, uint32_t Page, uint32_t Split, uint32_t BK, uint32_t WK,
          uint32_t SQ, uint32_t SK, bool SplitM, uint32_t Scale,
          uint32_t WorkersPerCU>
class PPUGroupedPagedMqaLogitsFP4 {
    using Kernel = PPUPagedMqaLogitsFP4<Q, Acc, Out, W, BQ, H, D,
        Page, Split, BK, WK, SQ, SK, SplitM, Scale, true>;
public:
    using Params = typename Kernel::Params;
    static constexpr int SharedStorageSize = Kernel::SharedStorageSize;
    static constexpr int MaxThreadsPerBlock = Kernel::MaxThreadsPerBlock;
    static constexpr int MinBlocksPerMultiprocessor = WorkersPerCU;
    CUTLASS_DEVICE void operator()(Params const& p, char* smem) {
        deep_gemm::FP4GroupedPagedScheduler<BQ, Split, BK, WorkersPerCU> scheduler(
            reinterpret_cast<const deep_gemm::FP4VarlenCTAStart*>(p.schedule_meta),
            reinterpret_cast<const deep_gemm::FP4VarlenQBlock*>(p.grouped_meta));
        Kernel op;
        scheduler.for_each_task(
            [&](uint32_t q, uint32_t begin, uint32_t end, uint32_t last,
                uint32_t count0, uint32_t count1) {
                auto run_request = [&](auto valid0, auto valid1) {
                    op.run_task(p, smem, q, begin, end, last,
                                Int<BQ>{}, cute::true_type{}, valid0, valid1);
                };
                if constexpr (BQ == 4) {
                    if (count0 == 3 && count1 == 0) run_request(Int<3>{}, Int<0>{});
                    else if (count0 == 3 && count1 == 2) run_request(Int<3>{}, Int<2>{});
                    else if (count0 == 3 && count1 == 3) run_request(Int<3>{}, Int<3>{});
                    else run_request(Int<0>{}, Int<0>{});
                } else run_request(Int<0>{}, Int<0>{});
            },
            [&](uint32_t q, uint32_t begin, uint32_t end, uint32_t valid) {
                const uint32_t span = deep_gemm::fp4_varlen_q_span<BQ>(valid);
                auto run = [&](auto q_span) {
                    op.run_task(p, smem, q, begin, end, 0,
                                q_span, cute::false_type{}, Int<0>{}, Int<0>{});
                };
                if (span == 1) run(Int<1>{});
                else if (span == 2) run(Int<2>{});
                else if constexpr (BQ == 4) run(Int<4>{});
            });
    }
};
} // namespace cutlass::gemm::kernel
