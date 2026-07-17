#pragma once
#include "ppu_include.hpp"
#include "cute_tie.cuh"
#include "utils.cuh"
#include "utils_cutlass3.h"
#include "profiling_interface.hpp"

__forceinline__ __device__ int get_lane_idx() {
    int lane_id;
    asm("mov.u32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

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

template <typename ElementQK, typename ElementAcc, typename ElementLogits, int kNumHeads, int kHeadDim, int BLOCK_QH,
          int BLOCK_KV, int WARP_QH, int WARP_KV, int kNumQStages, int kNumKVStages,
          typename StrideKType = uint32_t>
class PPUMqaLogitsFP4 {
public:
    static_assert(std::is_same_v<ElementQK, uint8_t>, "FP4 MQA logits requires uint8_t ElementQK");
    static_assert(kHeadDim == 64, "FP4 packed head_dim must be 64 (original 128 / 2)");
    static_assert(std::is_same_v<StrideKType, uint32_t> || std::is_same_v<StrideKType, uint64_t>,
                  "StrideKType must be uint32_t or uint64_t");

    // ── Core types ──────────────────────────────────────────────────────
    using ElementC = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;
    using ElementCompute = float;
    using ElementScale = uint8_t;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    static constexpr int BLOCK_M = BLOCK_KV;
    static constexpr int BLOCK_N = BLOCK_QH;
    static constexpr int BLOCK_K = kHeadDim; // 64 (packed)
    static constexpr int WARP_M = WARP_KV;
    static constexpr int WARP_N = WARP_QH;
    static constexpr int BLOCK_Q = BLOCK_QH / kNumHeads;
    static constexpr int WARP_Q = WARP_QH / kNumHeads;

    static constexpr int kNumMathWarpGroups = 1;

    // ── Tile / Warp shapes ──────────────────────────────────────────────
    using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
    using WarpShape = Shape<Int<WARP_M>, Int<WARP_N>, Int<BLOCK_K>>;
    static constexpr int WarpOnM = BLOCK_M / WARP_M;
    static constexpr int WarpOnN = BLOCK_N / WARP_N;

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
    static constexpr bool WarpInterleaving = (NumThreadsPerCTA == 512);

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
        cutlass::gemm::config::DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, float, false, _1, Int<BLOCK_N>, false, 0, false>;
    using SmemLayoutAtomWeight = typename DefaultOperandWeight::SmemLayoutAtom;
    using GmemTiledCopyWeight = typename DefaultOperandWeight::GmemTiledCopy;
    using SmemLayoutWeight =
        decltype(tile_to_shape(SmemLayoutAtomWeight{}, make_shape(_1{}, Int<BLOCK_N>{}, Int<kNumQStages>{})));

    // ── SFA s2r: recast smem as uint16_t, then transposed layout for MMA ──
    using sSFATransLayout = Layout<Shape<Shape<Int<WarpOnM>, Int<MmaIterM>>, Shape<_2, _8, _2>, Int<kNumKVStages>>,
                                   Stride<Stride<Int<MmaIterM * 32>, _32>, Stride<_16, _2, _1>, Int<BLOCK_M * 2>>>;
    using SmemTiledCopySFA = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint16_t>, uint16_t>{},
        Layout<Shape<Shape<Int<WarpOnM>, _4>, _8>, Stride<Stride<_32, _1>, _4>>{}, Layout<Shape<_1, _2>>{}));

    // ── SFB s2r: recast smem as uint16_t, then transposed layout for MMA ──
    using sSFBTransLayout = Layout<Shape<Shape<Int<WarpOnN>, Int<MmaIterN>>, Shape<_2, _8, _2>, Int<kNumQStages>>,
                                   Stride<Stride<Int<MmaIterN * 32>, _32>, Stride<_16, _2, _1>, Int<BLOCK_N * 2>>>;
    using SmemTiledCopySFB = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint16_t>, uint16_t>{},
        Layout<Shape<Shape<Int<WarpOnN>, _4>, _8>, Stride<Stride<_32, _1>, _4>>{}, Layout<Shape<_1, _2>>{}));

    // ── Weights s2r copy ────────────────────────────────────────────────
    using sWCopyLayout = Layout<
        Shape<Shape<_2, _4, Int<WarpOnN>, _2, Int<MmaIterN>>, _1, Int<kNumKVStages>>, Stride<Stride<_1, _2, Int<WARP_N>, _8, _16>, _1, Int<BLOCK_N>>, >;
    using SmemTiledCopyWeights = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint64_t>, float>{},
        Layout<Shape<Shape<_4, Int<WarpOnN>>, _1>, Stride<Stride<_1, _4>, _1>>{}, Layout<Shape<_2, _1>>{}));

    // ── Shared memory ──────────────────────────────────────────────────
    struct SharedStorage {
        cute::array_aligned<ElementQK, cute::cosize_v<SmemLayoutA>> smem_k;       // K packed FP4
        cute::array_aligned<ElementQK, cute::cosize_v<SmemLayoutB>> smem_q;       // Q packed FP4
        cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutSFA>> smem_k_sf;   // K e8m0 scales (uint32_t)
        cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutSFB>> smem_q_sf;   // Q e8m0 scales (uint32_t)
        cute::array_aligned<float, cute::cosize_v<SmemLayoutWeight>> smem_weight; // weights (float32)
    };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    // ── Device side arguments ──────────────────────────────────────────
    struct Arguments {
        const ElementQK* ptr_q;
        const uint32_t* q_sf; // Q scale factor (packed e8m0, uint32_t)
        const ElementQK* ptr_k;
        const uint32_t* k_sf; // K scale factor (packed e8m0, uint32_t)
        const float* weights;
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

        gmem_tiled_copy_A.desc_.template init<ElementQK, false, get<0>(TilerA{}), get<1>(TilerA{})>(
            nullptr, BLOCK_M, BLOCK_K, StrideAB{});
        gmem_tiled_copy_B.desc_.template init<ElementQK, false, get<0>(TilerB{}), get<1>(TilerB{})>(
            nullptr, BLOCK_N, BLOCK_K, StrideAB{});
        gmem_tiled_copy_SFA.desc_.template init<uint32_t, false, get<0>(TilerSFA{}), get<1>(TilerSFA{})>(
            nullptr, 1, BLOCK_M, StrideSFA{});
        gmem_tiled_copy_SFB.desc_.template init<uint32_t, false, get<0>(TilerSFB{}), get<1>(TilerSFB{})>(
            nullptr, 1, BLOCK_N, StrideSFB{});
        gmem_tiled_copy_weight.desc_.template init<float, false, get<0>(TilerSFB{}), get<1>(TilerSFB{})>(
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
        int lane_idx = get_lane_idx();

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
        constexpr int elem_weights = kNumHeads / 4;
        ElementLogits weights[elem_weights];
        Tensor sW_copy = make_tensor(sW.data(), sWCopyLayout{});
        SmemTiledCopyWeights smem_tiled_copy_weights;
        auto smem_thr_copy_weights = smem_tiled_copy_weights.get_slice(warp_n_idx * 4 + lane_idx % 4);
        Tensor tCsW = smem_thr_copy_weights.partition_S(sW_copy);
        Tensor tCrW_copy_view = make_tensor_like(tCsW(_, _, _, 0));
        using WeightRegLayout = Layout<Shape<Shape<_2, _2, _2>, Int<MmaIterN>>, Stride<Stride<_1, _0, _2>, _4>>;
        Tensor tCrW = make_tensor(static_cast<ElementLogits*>(weights), WeightRegLayout{});

        // ── Block scheduler (non-paged) ───────────────────────────────────
        const auto& num_q_blocks = ceil_div(params.seq_len_q, BLOCK_Q);
        int block_q_idx = blockIdx.x, q_iter_idx = 0;

        const auto& get_next_block_q_idx = [&]() -> cute::tuple<int, int> {
            return {block_q_idx + gridDim.x, q_iter_idx + 1};
        };

        const auto& load_schedule = [&](const int& q_iter_offset = 0) -> cute::tuple<int, int, int, int, int> {
            int start = cute::numeric_limits<int>::max();
            int end = cute::numeric_limits<int>::min();
            #pragma unroll
            for (int i = 0; i < BLOCK_Q; ++i) {
                const auto& q_idx = min(block_q_idx * BLOCK_Q + i, params.seq_len_q - 1);
                start = min(start, min(__ldg(params.cu_seq_len_k_start + q_idx), params.seq_len_k));
                end = max(end, min(__ldg(params.cu_seq_len_k_end + q_idx), params.seq_len_k));
            }
            start = start / 4 * 4;
            return {(q_iter_idx + q_iter_offset) % kNumQStages, ((q_iter_idx + q_iter_offset) / kNumQStages) & 1, start,
                    end, ceil_div(end - start, BLOCK_KV)};
        };

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
        int warp_q_idx = warp_idx / WarpOnM;
        int warp_group_id = warp_idx / 8;

        auto warp_interleave_start = [&]() {
            if constexpr (WarpInterleaving) {
                if (warp_group_id == 1) {
                    __ppu_barrier_arrive(5, NumThreadsPerCTA, 0);
                }
            }
        };

        auto warp_interleave_sync = [&]() {
            if constexpr (WarpInterleaving) {
                __ppu_barrier_sync(5 + warp_group_id, NumThreadsPerCTA);
            }
        };

        auto warp_interleave_defer = [&]() {
            if constexpr (WarpInterleaving) {
                __ppu_barrier_arrive(6 - warp_group_id, NumThreadsPerCTA, 0);
            }
        };

        auto warp_interleave_end = [&]() {
            if constexpr (WarpInterleaving) {
                if (warp_group_id == 0) {
                    __ppu_barrier_sync(5, NumThreadsPerCTA);
                }
            }
        };

        auto load_q_g2s = [&](int block_q_idx, int q_stage_idx) {
            int residual_n = (params.seq_len_q - block_q_idx * BLOCK_Q) * kNumHeads;
            gmem_tiled_copy_B.desc_.dim_h = residual_n;
            gmem_tiled_copy_SFB.desc_.dim_w = residual_n;
            gmem_tiled_copy_weight.desc_.dim_w = residual_n;
            copy_aiu(gmem_tiled_copy_B, tBgB(_, _, _, 0), tBsB(_, _, _, q_stage_idx), warp_idx);
            copy_aiu<true>(gmem_tiled_copy_SFB, tSFBgSFB(_, _, _, 0), tSFBsSFB(_, _, _, q_stage_idx), gmem_tiled_copy_weight,
                           tWgW(_, _, _, 0), tWsW(_, _, _, q_stage_idx), warp_idx);
        };

        auto load_kv_g2s = [&](int kv_start, int kv_end, int kv_block_idx, int smem_stage) {
            int residual_m = kv_end - (kv_start + kv_block_idx * BLOCK_KV);
            gmem_tiled_copy_A.desc_.dim_h = residual_m;
            gmem_tiled_copy_SFA.desc_.dim_w = residual_m;
            copy_aiu<true>(gmem_tiled_copy_A, tAgA(_, _, _, 0), tAsA(_, _, _, smem_stage), gmem_tiled_copy_SFA,
                           tSFAgSFA(_, _, _, 0), tSFAsSFA(_, _, _, smem_stage), warp_idx);
            tAgA.data() = tAgA.data() + BLOCK_KV * BLOCK_K;
            tSFAgSFA.data() = tSFAgSFA.data() + BLOCK_KV;
        };

        auto load_q_s2r = [&](int q_stage_idx) {
            copy(smem_tiled_copy_B, tCsB(_, _, _, q_stage_idx), tCrB_copy_view);
            copy(smem_tiled_copy_SFB, tCsSFB(_, _, _, q_stage_idx), tCrSFB_copy_view);
            copy(smem_tiled_copy_weights, tCsW(_, _, _, q_stage_idx), tCrW_copy_view);
            if constexpr (cute::is_same_v<ElementLogits, float>) {
                for (int j = 0; j < elem_weights; j++) {
                    weights[j] = tCrW_copy_view(j);
                }
            } else {
                for (int j = 0; j < elem_weights; j += 2) {
                    uint32_t d;
                    asm volatile("ppu.cvt.rtte.bf16x2.f32 %0, %1, %2;\n"
                                 : "=r"(d)
                                 : "f"(tCrW_copy_view(j + 1)), "f"(tCrW_copy_view(j)));
                    *reinterpret_cast<uint32_t*>(&weights[j]) = d;
                }
            }
        };

        auto load_kv_s2r = [&](int k_block, int kv_stage_idx) {
            copy(smem_tiled_copy_A, tCsA(_, _, k_block, kv_stage_idx), tCrA_copy_view(_, _, k_block));
            copy(smem_tiled_copy_SFA, tCsSFA(_, _, k_block, kv_stage_idx), tCrSFA_copy_view(_, _, k_block));
        };

        auto load_kv_s2r_mblock = [&](int k_block, int kv_stage_idx, int m_block) {
            copy(smem_tiled_copy_A, tCsA(_, m_block, k_block, kv_stage_idx), tCrA_copy_view(_, m_block, k_block));
            copy(smem_tiled_copy_SFA, tCsSFA(_, m_block, k_block, kv_stage_idx), tCrSFA_copy_view(_, m_block, k_block));
        };

        auto epilogue_mblock = [&](int kv_start, int kv_block_idx, int m_block) {
            // Reduce over heads and store logits
            static constexpr int kNumAccumPerMma = size<0>(accum);
            CUTE_STATIC_ASSERT(kNumHeads % 8 == 0);
            const auto& kv_offset = kv_start + kv_block_idx * BLOCK_KV + warp_offset;
            {
                int m = m_block;
                int mma_offset = m * InstM;
                auto logits_q_offset = (block_q_idx * BLOCK_Q + warp_q_idx) * params.stride_k;
                auto logits_kv_offset = kv_offset + mma_offset;

                if constexpr (std::is_same_v<ElementLogits, float>) {
                    auto transform = [&](int j, int n) {
                        return fmaxf(accum(j, m, n), 0) * tCrW(j, n);
                    };
                    float v_0 = 0, v_1 = 0;
                    #pragma unroll
                    for (int n = 0; n < size<2>(accum); n++) {
                        v_0 += transform(0, n);
                        v_0 += transform(1, n);
                        v_1 += transform(2, n);
                        v_1 += transform(3, n);
                        v_0 += transform(4, n);
                        v_0 += transform(5, n);
                        v_1 += transform(6, n);
                        v_1 += transform(7, n);
                    }
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const auto& offset = static_cast<int>(1u << j);
                        v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
                        v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
                    }
                    params.logits[logits_q_offset + logits_kv_offset + v_0_offset] = v_0;
                    params.logits[logits_q_offset + logits_kv_offset + v_1_offset] = v_1;
                } else {
                    // bf16 output: separate cvt and fma2 phases with __ppu_sched_bound()
                    constexpr int kTotalTransforms = 4 * size<2>(accum);
                    __ppu_bfloat162 sum_0 = {0, 0};
                    __ppu_bfloat162 sum_1 = {0, 0};
                    uint32_t cvt_buf[kTotalTransforms];

                    // cvt phase: issue all ppu.cvt.rtte.bf16x2.f32.relu
                    #pragma unroll
                    for (int idx = 0; idx < kTotalTransforms; idx++) {
                        int n = idx / 4;
                        int sub = idx % 4;
                        int j = sub * 2;
                        asm volatile("ppu.cvt.rtte.bf16x2.f32.relu %0, %1, %2;\n"
                                     : "=r"(cvt_buf[idx])
                                     : "f"(accum(j + 1, m, n)), "f"(accum(j, m, n)));
                    }
                    __ppu_sched_bound();
                    // fma2 phase: issue all __hfma2
                    #pragma unroll
                    for (int idx = 0; idx < kTotalTransforms; idx++) {
                        int n = idx / 4;
                        int sub = idx % 4;
                        int j = sub * 2;
                        __ppu_bfloat162 a = reinterpret_cast<__ppu_bfloat162&>(cvt_buf[idx]);
                        __ppu_bfloat162 b = {tCrW(j, n), tCrW(j + 1, n)};
                        if (sub % 2 == 0) {
                            sum_0 = __hfma2(a, b, sum_0);
                        } else {
                            sum_1 = __hfma2(a, b, sum_1);
                        }
                    }
                    __ppu_bfloat16 v_0_bf = __hadd(__low2bfloat16(sum_0), __high2bfloat16(sum_0));
                    __ppu_bfloat16 v_1_bf = __hadd(__low2bfloat16(sum_1), __high2bfloat16(sum_1));
                    __ppu_bfloat162 packed = {v_0_bf, v_1_bf};
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        uint32_t bits = reinterpret_cast<uint32_t&>(packed);
                        uint32_t received_bits = __shfl_xor_sync(0xffffffffu, bits, 1u << j);
                        __ppu_bfloat162 received = reinterpret_cast<__ppu_bfloat162&>(received_bits);
                        packed = __hadd2(packed, received);
                    }
                    params.logits[logits_q_offset + logits_kv_offset + v_0_offset] = __low2bfloat16(packed);
                    params.logits[logits_q_offset + logits_kv_offset + v_1_offset] = __high2bfloat16(packed);
                }
            }
        };

        // ── Outer loop: Q blocks ──────────────────────────────────────────
        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(0), q_stage_idx, q_phase, kv_start, kv_end, num_kv_blocks);

            // ── Offset Q / K / scale data pointers ──────────────────────────
            tAgA.data() = tKgK.data() + kv_start * BLOCK_K;
            tSFAgSFA.data() = tSFKgSFK.data() + kv_start;
            tBgB.data() = tQgQ.data() + block_q_idx * BLOCK_Q * kNumHeads * BLOCK_K;
            tSFBgSFB.data() = tSFQgSFQ.data() + block_q_idx * BLOCK_Q * kNumHeads;
            tWgW.data() = tWgW_base.data() + block_q_idx * BLOCK_Q * kNumHeads;

            int current_stage_kv = 0;
            if (num_kv_blocks > 0) {
                // ── Load Q + q_sf + weights (AIU) ──────────────────────────
                load_q_g2s(block_q_idx, q_stage_idx);

                // ── Prologue: load first (kNumKVStages - 1) K blocks (AIU) ──
                for (int kv_pipe = 0; kv_pipe < kNumKVStages - 1; kv_pipe++) {
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

            warp_interleave_start();

            // ── Inner loop: KV blocks ──────────────────────────────────────
            for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
                int kv_stage_idx = kv_block_idx % kNumKVStages;

                warp_interleave_sync();

                // Issue next K block
                int next_kv_block_idx = kv_block_idx + kNumKVStages - 1;
                if (next_kv_block_idx < num_kv_blocks) {
                    load_kv_g2s(kv_start, kv_end, next_kv_block_idx, current_stage_kv);
                }
                cp_async_fence();
                current_stage_kv = (current_stage_kv + 1) % kNumKVStages;

                constexpr int M_BLOCK = size<1>(accum);
                constexpr int N_BLOCK = size<2>(accum);
                constexpr int K_BLOCK = size<2>(tCrA);
                if constexpr (std::is_same_v<ElementLogits, float>) {
                    for_each(make_int_sequence<K_BLOCK>{}, [&](auto k_block) {
                        load_kv_s2r(k_block, kv_stage_idx);
                        cute::gemm(tiled_mma, accum, tCrA(_, _, k_block), tCrSFA(_, _, k_block), tCrB(_, _, k_block),
                                    tCrSFB(_, _, k_block), accum);
                    });
                    warp_interleave_defer();
                    #pragma unroll
                    for (int m_block = 0; m_block < size<1>(accum); m_block++) {
                        epilogue_mblock(kv_start, kv_block_idx, m_block);
                    }
                } else { // Interleaved mma and epilogue
                    constexpr int m_group = M_BLOCK / size<1>(tCrSFA);
                    constexpr int n_group = N_BLOCK / size<1>(tCrSFB);
                    for_each(make_int_sequence<M_BLOCK>{}, [&](auto m_block) {
                        if constexpr(m_block > 0) warp_interleave_sync();
                        for_each(make_int_sequence<K_BLOCK>{}, [&](auto k_block) {
                            load_kv_s2r_mblock(k_block, kv_stage_idx, m_block);
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
                        warp_interleave_defer();
                        epilogue_mblock(kv_start, kv_block_idx, m_block);
                    });
                }

                // Wait for next K
                cp_async_wait<kNumKVStages - 2>();
                if constexpr (!WarpInterleaving) {
                    __syncthreads();
                }
                clear(accum);
            } // end KV loop

            warp_interleave_end();

            cp_async_wait<0>();
            __syncthreads();

            // Jump to next Q block
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        } // end Q loop
    }
};

} // namespace cutlass::gemm::kernel

namespace deep_gemm {

template <typename ElementQK, typename ElementAcc, typename ElementLogits, int kNumHeads, int kHeadDim, int BLOCK_QH,
          int BLOCK_KV, int WARP_QH, int WARP_KV, int kNumQStages, int kNumKVStages,
          typename StrideKType = uint32_t>
class AttentionFP4 {
public:
    static void run(const ElementQK* ptr_q, const uint32_t* q_sf, const ElementQK* ptr_k, const uint32_t* k_sf,
                    const float* weights, int* cu_seq_len_k_start, int* cu_seq_len_k_end, ElementLogits* logits,
                    const int seq_len_q, const int seq_len_k, const StrideKType stride_k, hggcStream_t stream,
                    int num_sms) {
        using AttnKernel =
            cutlass::gemm::kernel::PPUMqaLogitsFP4<ElementQK, ElementAcc, ElementLogits, kNumHeads, kHeadDim, BLOCK_QH,
                                                    BLOCK_KV, WARP_QH, WARP_KV, kNumQStages, kNumKVStages, StrideKType>;

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
            if constexpr (std::is_same_v<ElementLogits, __ppu_bfloat16>) {
                dg_prof_params.add_params("logits_dtype", std::string("bf16"));
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
            printf("num_sms:%d, max_blocks_per_cu:%d, threadblock_count:%d\n", num_sms, max_blocks_per_cu,
                   threadblock_count);
            printf("smem_size:%d, vreg:%d, stack:%d\n", smem_size_kernel, int(attr.numRegs), int(attr.localSizeBytes));
        }

        ProfilingInterface::Instance().instrument(true, dg_prof_params);
        cutlass::device_kernel<AttnKernel><<<grid, block, smem_size_kernel, stream>>>(params);
        ProfilingInterface::Instance().instrument(false, dg_prof_params);
    }
};

}; // namespace deep_gemm
