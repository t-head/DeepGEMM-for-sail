#pragma once
#include "ppu_include.hpp"
#include "cute_tie.cuh"
#include "utils.cuh"
#include "utils_cutlass3.h"
#include "profiling_interface.hpp"
// Reuse PagedMQALogitsScheduler, metadata kernel, and device helpers from FP8 paged MQA logits
#include "ppu_paged_mqa_logits.cuh"

namespace cutlass::gemm::kernel {

// ============================================================================
// FP4 Paged MQA Logits Kernel
// ============================================================================
// Key differences from FP8 PPUPagedMqaLogits:
//   - ElementQK is uint8_t (packed FP4, 2 elements per byte)
//   - kHeadDim is the *packed* dimension (original_head_dim / 2 = 64)
//   - Additional q_sf (Q scale factor, uint8 e8m0) parameter
//   - k_sf is uint8 e8m0 (not float32 per-token scale)
//   - Q/K/q_sf/k_sf/weights all use AIU load
//   - Output dtype controlled by ElementLogits (float or bfloat16)
//   - MMA atom: F32F4F4F32 (16x16x64)
// ============================================================================

template <typename ElementQK, typename ElementAcc, typename ElementLogits, uint32_t kNextN, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV, uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t SPLIT_KV>
class PPUPagedMqaLogitsFP4 {
public:
    static_assert(std::is_same_v<ElementQK, uint8_t>, "FP4 paged MQA logits requires uint8_t ElementQK");
    static_assert(kHeadDim == 64, "FP4 packed head_dim must be 64 (original 128 / 2)");

    using ElementC = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;
    using ElementCompute = float;
    // FP4 scale: uint8_t e8m0, packed as uint32_t for async copy
    using ElementScale = uint8_t;
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    static constexpr int BLOCK_M = BLOCK_KV;
    static constexpr int BLOCK_N = kNextN * kNumHeads;
    static constexpr int BLOCK_K = 64; // kHeadDim = 64
    static constexpr int WARP_M = 16;
    static constexpr int WARP_N = kNumHeads;

    static constexpr int kNumMathWarpGroups = SPLIT_KV / BLOCK_KV;

    using ProblemShape_MNKL = Shape<int, int, int, int>;

    using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
    using WarpShape = Shape<Int<WARP_M>, Int<WARP_N>, Int<BLOCK_K>>;
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
    static constexpr bool WarpInterleaving = (MaxThreadsPerBlock == 512);
    static_assert(!(WarpInterleaving && kNumQStages == 1), "warp-interleave does not support stage_q = 1");

    using DefaultOperandA =
        cutlass::gemm::config::DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, ElementQK, false, Int<BLOCK_M>, Int<BLOCK_K>, false>;
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
    // Weights: float32 per (batch * next_n, num_heads), same as FP8 paged MQA

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

    // SFB (q_sf/weights): one uint32_t/float32 per Q head
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

    // SFA smem->vreg: recast smem as uint16_t, then transpose layout for MMA consumption
    // sSFATrans shape: ((MmaIterM, WarpOnM), (2, 8, 2), kNumKVStages), stride: ((WarpOnM * 32, 32), (16, 2, 1),
    // BLOCK_M*2) Note: operates on per-warp_group 3D slice after slicing the warp_group dimension
    using sSFATransLayout = Layout<Shape<Shape<Int<WarpOnM>, Int<MmaIterM>>, Shape<_2, _8, _2>, Int<kNumKVStages>>,
                                   Stride<Stride<Int<MmaIterM * 32>, _32>, Stride<_16, _2, _1>, Int<BLOCK_M * 2>>>;
    // SFA s2r tiled copy: uint16_t copy atom
    // thr_layout: ((WarpOnM, 4), 8) stride: ((32, 1), 4), val_layout: (1, 2)
    using SmemTiledCopySFA = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint16_t>, uint16_t>{},
        Layout<Shape<Shape<Int<WarpOnM>, _4>, _8>, Stride<Stride<_32, _1>, _4>>{}, Layout<Shape<_1, _2>>{}));

    // SFB s2r tiled copy: same as SFA s2r tiled copy
    using sSFBTransLayout = Layout<Shape<Shape<Int<WarpOnN>, Int<MmaIterN>>, Shape<_2, _8, _2>, Int<kNumQStages>>,
                                   Stride<Stride<Int<MmaIterN * 32>, _32>, Stride<_16, _2, _1>, Int<BLOCK_N * 2>>>;
    using SmemTiledCopySFB = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint16_t>, uint16_t>{},
        Layout<Shape<Shape<Int<WarpOnN>, _4>, _8>, Stride<Stride<_32, _1>, _4>>{}, Layout<Shape<_1, _2>>{}));

    using sWCopyLayout = Layout<
        Shape<Shape<_2, _4, Int<WarpOnN>, _2, Int<MmaIterN>>, _1, Int<kNumQStages>>, Stride<Stride<_1, _2, Int<WARP_N>, _8, _16>, _1, Int<BLOCK_N>>, >;
    using SmemTiledCopyWeights = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint64_t>, float>{},
        Layout<Shape<Shape<_4, Int<WarpOnN>>, _1>, Stride<Stride<_1, _4>, _1>>{}, Layout<Shape<_2, _1>>{}));

    static_assert(kNumQStages <= 2 && kNumKVStages >= 3, "q_stage <= 2 and kv_stage >= 3");

    // Kernel level shared memory storage
    struct SharedStorage {
        cute::array_aligned<ElementQK, cute::cosize_v<SmemLayoutA>> smem_k;       // KV packed FP4
        cute::array_aligned<ElementQK, cute::cosize_v<SmemLayoutB>> smem_q;       // Q packed FP4
        cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutSFA>> smem_k_sf;   // KV e8m0 scales (uint32_t)
        cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutSFB>> smem_q_sf;   // Q e8m0 scales (uint32_t)
        cute::array_aligned<float, cute::cosize_v<SmemLayoutWeight>> smem_weight; // weights (float32)
    };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    // Device side arguments
    struct Arguments {
        const ElementQK* ptr_q;
        const uint32_t* q_sf; // Q scale factor (packed e8m0, 4×uint8 per int32)
        const ElementQK* ptr_k;
        const uint32_t* k_sf; // K scale factor (packed e8m0, uint32_t)
        const float* weights;
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

        // weights (float32, same as FP8 paged MQA)
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
        int lane_idx = get_lane_idx();
        int warp_group_id = warp_idx / 8;

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
        Tensor accum = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
        auto thr_mma = tiled_mma.get_thread_slice(local_thread_idx);
        Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
        Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));

        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));
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
        Tensor sSFATrans = make_tensor(sSFAUint16.data(), sSFATransLayout{});

        SmemTiledCopySFA smem_tiled_copy_SFA;
        auto smem_thr_copy_SFA = smem_tiled_copy_SFA.get_slice(warp_m_idx * 32 + lane_idx);
        Tensor tCsSFA = smem_thr_copy_SFA.partition_S(sSFATrans);
        Tensor tCrSFA_stage = smem_thr_copy_SFA.partition_D(sSFATrans(_, _, 0));
        using SFARegType = decltype(make_fragment_like(tCrSFA_stage));
        SFARegType tCrSFA_copy_view;

        // SFB s2r: recast smem as uint16_t, then apply transposed layout for MMA consumption
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

        constexpr bool enable_print = false;
        bool thread_print = enable_print && cute::thread(0, 0);

        // Scheduler
        auto scheduler = PagedMQALogitsScheduler<BLOCK_KV, kNumMathWarpGroups>(
            params.batch_size, blockIdx.x, params.context_lens, params.schedule_meta);
        DG_STATIC_ASSERT(SPLIT_KV % BLOCK_KV == 0, "Unaligned SPLIT_KV");

        constexpr int elem_weights = kNumHeads / 4;
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

        uint32_t q_idx = scheduler.current_q_idx;
        uint32_t kv_idx_base;
        uint32_t kv_idx_array[kNumKVStages];
        uint32_t smem_pipe_read_q = 0, smem_pipe_read_kv = 0;
        uint32_t smem_pipe_write_q = 0, smem_pipe_write_kv = 0;

        // Load Q (packed FP4) + q_sf (e8m0) + weights from gmem to smem
        auto load_q_g2s = [&](uint32_t q_idx) {
            auto q_offset = q_idx * BLOCK_N;
            tBgB.data() = tQgQ.data() + q_offset * kHeadDim;
            tSFBgSFB.data() = tSFQgSFQ.data() + q_offset;
            tWgW.data() = tWgW_base.data() + q_offset;
            copy_aiu(gmem_tiled_copy_B, tBgB(_, _, _, 0), tBsB(_, _, _, smem_pipe_write_q), warp_idx);
            copy_aiu<true>(gmem_tiled_copy_SFB, tSFBgSFB(_, _, _, 0), tSFBsSFB(_, _, _, smem_pipe_write_q),
                           gmem_tiled_copy_weight, tWgW(_, _, _, 0), tWsW(_, _, _, smem_pipe_write_q), warp_idx);

            if (thread_print) {
                printf("  copy_q q_idx = %d, q_offset = %d, stage = %d\n", q_idx, q_offset, smem_pipe_write_q);
            }
            smem_pipe_write_q = (smem_pipe_write_q + 1) % kNumQStages;
        };

        // Load K (packed FP4) + k_sf (e8m0) from gmem to smem
        // Each warp_group loads into its own 3D smem slice, at stage = smem_pipe_write_kv
        auto load_kv_g2s = [&](uint32_t q_idx, uint32_t kv_idx) {
            auto kv_offset = __ldg(params.block_table + q_idx * params.block_table_stride + kv_idx);
            tAgA.data() = tKgK.data() + kv_offset * params.kv_cache_stride_bytes;
            tSFAgSFA.data() = tSFKgSFK.data() + kv_offset * params.kv_cache_stride_bytes / sizeof(uint32_t);
            copy_aiu<true>(gmem_tiled_copy_A, tAgA(_, _, _, 0), tAsA(_, _, _, smem_pipe_write_kv), gmem_tiled_copy_SFA,
                           tSFAgSFA(_, _, _, 0), tSFAsSFA(_, _, _, smem_pipe_write_kv), local_warp_idx);

            if (thread_print) {
                printf("  copy_k q_idx = %d, kv_idx = %d, kv_offset = %d, wg = %d, stage = %d\n", q_idx, kv_idx,
                       kv_offset, warp_group_idx, smem_pipe_write_kv);
            }
        };

        // Load Q from smem to vreg + read weights
        auto load_q_s2r = [&]() {
            copy(smem_tiled_copy_B, tCsB(_, _, _, smem_pipe_read_q), tCrB_copy_view);
            copy(smem_tiled_copy_SFB, tCsSFB(_, _, _, smem_pipe_read_q), tCrSFB_copy_view);
            copy(smem_tiled_copy_weights, tCsW(_, _, _, smem_pipe_read_q), tCrW_copy_view);
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

            if (thread_print) {
                printf("    copy q to vreg, q_stage_idx = %d,\n", smem_pipe_read_q);
            }
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
            if (thread_print)
                printf("cp_async commit, write_stage = %d\n", smem_pipe_write_kv);
            cp_async_fence();
        };

        auto epilogue = [&](int q_idx, int kv_idx) {
            // Reduce over heads and store logits
            static constexpr int kNumAccumPerMma = size<0>(accum);
            CUTE_STATIC_ASSERT(kNumHeads % 8 == 0);

            for (int m = 0; m < size<1>(accum); m++) {
                int mma_offset = m * InstM;
                auto logits_q_offset = (q_idx * kNextN + warp_n_idx) * params.logits_stride;
                auto logits_kv_offset = kv_idx * BLOCK_KV + warp_offset + mma_offset;

                if constexpr (cute::is_same_v<ElementLogits, float>) {
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

                    // Inter-thread reduction
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
                    __ppu_sched_bound();
                    // Intra-thread reduction in bfloat16
                    __ppu_bfloat16 v_0_bf = __hadd(__low2bfloat16(sum_0), __high2bfloat16(sum_0));
                    __ppu_bfloat16 v_1_bf = __hadd(__low2bfloat16(sum_1), __high2bfloat16(sum_1));

                    // Cross-lane reduction: pack v_0/v_1 into bfloat162 for single 32-bit shuffle
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

        // load qk
        for (int i = 0; i < kNumKVStages - 1; i++) {
            load_next_qk_g2s(i == 0);
        }
        kv_idx_base = kv_idx_array[0];
        // wait AIU Q and first K
        cp_async_wait<kNumKVStages - 2>();
        __syncthreads();

        if constexpr (WarpInterleaving) {
            if (warp_group_id == 1) {
                __ppu_barrier_arrive(5, MaxThreadsPerBlock, 0);
            }
        }

        auto handle_q_change = [&]() {
            if constexpr (kNumQStages > 1) {
                if (scheduler.exist_q_idx(q_idx + 1)) {
                    load_q_g2s(q_idx + 1);
                }
                cp_async_fence();
            }
            load_q_s2r();
            if constexpr (kNumQStages == 1) { // not support warp interleaving
                __syncthreads();
                if (scheduler.exist_q_idx(q_idx + 1)) {
                    load_q_g2s(q_idx + 1);
                }
                cp_async_fence();
            }
        };

        // Make `first` a compile-time constant via std::integral_constant<bool, ...>
        // so the compiler can fold the `first || kv_idx_base == 0` branch and avoid
        // the runtime check on the first iteration.
        auto mainloop = [&](auto first_ic) {
            constexpr bool first = decltype(first_ic)::value;
            if (thread_print) {
                printf("q_idx = %d, kv_idx_base = %d, wg = %d, read_stage = %d\n", q_idx, kv_idx_base, warp_group_idx, smem_pipe_read_kv);
            }

            if constexpr (WarpInterleaving) {
                __ppu_barrier_sync(5 + warp_group_id, MaxThreadsPerBlock);
            }

            // Handle Q change: on the first iteration always load the next Q;
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
                copy(smem_tiled_copy_A, tCsA(_, _, k_block, smem_pipe_read_kv),
                     tCrA_copy_view(_, _, k_block));
                copy(smem_tiled_copy_SFA, tCsSFA(_, _, k_block, smem_pipe_read_kv),
                     tCrSFA_copy_view(_, _, k_block));
                cute::gemm(tiled_mma, accum, tCrA(_, _, k_block), tCrSFA(_, _, k_block), tCrB(_, _, k_block),
                           tCrSFB(_, _, k_block), accum);
            });

            if constexpr (WarpInterleaving) {
                __ppu_barrier_arrive(6 - warp_group_id, MaxThreadsPerBlock, 0);
            }

            epilogue(q_idx, actual_kv);

            cp_async_wait<kNumKVStages - 2>();
            if (!WarpInterleaving) {
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

        if constexpr (WarpInterleaving) {
            if (warp_group_id == 0) {
                __ppu_barrier_sync(5, MaxThreadsPerBlock);
            }
        }

        cp_async_wait<0>();
        __syncthreads();
    }
};

} // namespace cutlass::gemm::kernel

namespace deep_gemm {

template <typename ElementQK, typename ElementAcc, typename ElementLogits, uint32_t kNextN, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV, uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t SPLIT_KV>
class PagedAttentionFP4 {
public:
    static void run(const ElementQK* ptr_q, const uint32_t* q_sf, const ElementQK* ptr_k, const uint32_t* k_sf,
                    const float* weights, const uint32_t batch_size, const uint64_t logits_stride,
                    const uint64_t kv_cache_stride_bytes, const uint32_t block_table_stride,
                    const uint32_t* context_lens, ElementLogits* logits, const uint32_t* block_table,
                    const uint32_t* schedule_meta, hggcStream_t stream, int num_sms, int num_blocks) {
        using AttnKernel =
            cutlass::gemm::kernel::PPUPagedMqaLogitsFP4<ElementQK, ElementAcc, ElementLogits, kNextN, kNumHeads,
                                                         kHeadDim, BLOCK_KV, kNumQStages, kNumKVStages, SPLIT_KV>;

        static constexpr int BLOCK_M = AttnKernel::BLOCK_M;
        static constexpr int BLOCK_N = AttnKernel::BLOCK_N;
        static constexpr int BLOCK_K = AttnKernel::BLOCK_K;
        static constexpr int WARP_M = AttnKernel::WARP_M;
        static constexpr int WARP_N = AttnKernel::WARP_N;
        static constexpr int MaxThreadsPerBlock = AttnKernel::MaxThreadsPerBlock;

        int max_blocks_per_cu = compute_occupancy_for_kernel<AttnKernel>();

        typename AttnKernel::Arguments arguments{ptr_q,
                                                 q_sf,
                                                 ptr_k,
                                                 k_sf,
                                                 weights,
                                                 batch_size,
                                                 logits_stride,
                                                 kv_cache_stride_bytes,
                                                 block_table_stride,
                                                 context_lens,
                                                 logits,
                                                 block_table,
                                                 schedule_meta};
        auto params = arguments;
        dim3 const block(MaxThreadsPerBlock, 1, 1);
        dim3 const grid(num_blocks, 1, 1);
        int smem_size_kernel = AttnKernel::SharedStorageSize;

        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()) {
            dg_prof_params.set_paged_mqa_logits_params("fp4", batch_size, kNextN, kNumHeads, kHeadDim * 2,
                                                       reinterpret_cast<int*>(const_cast<uint32_t*>(context_lens)),
                                                       stream);
            if constexpr (std::is_same_v<ElementLogits, __ppu_bfloat16>) {
                dg_prof_params.add_params("logits_dtype", std::string("bf16"));
            }
        }

        char* pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            hggcFuncAttributes attr;
            hggcFuncGetAttributes(&attr, cutlass::device_kernel<AttnKernel>);

            printf("[paged_mqa_logits_fp4:]\n");
            printf("kNumHeads:%d, kHeadDim:%d(packed), kNextN:%d, BLOCK_KV:%d, SPLIT_KV:%d\n", kNumHeads, kHeadDim,
                   kNextN, BLOCK_KV, SPLIT_KV);

            printf("ThreadblockShape[%d, %d], WarpShape[%d, %d], kNumQStages:%d, kNumKVStages:%d\n", BLOCK_M, BLOCK_N,
                   WARP_M, WARP_N, kNumQStages, kNumKVStages);

            printf("num_sms:%d, max_blocks_per_cu:%d, threadblock_count:%d\n", num_sms, max_blocks_per_cu, num_blocks);
            printf("smem_size:%d, vreg:%d, stack:%d\n", smem_size_kernel, int(attr.numRegs), int(attr.localSizeBytes));
            std::cout << "block = " << block << std::endl;
            std::cout << "grid = " << grid << std::endl;
            if (num_sms * max_blocks_per_cu != num_blocks) {
                printf("Warning: num_blocks(%d) should equal to num_sms(%d) * max_blocks_per_cu(%d) = %d\n", num_blocks,
                       num_sms, max_blocks_per_cu, num_sms * max_blocks_per_cu);
            }
        }

        ProfilingInterface::Instance().instrument(true, dg_prof_params);
        cutlass::device_kernel<AttnKernel><<<grid, block, smem_size_kernel, stream>>>(params);
        ProfilingInterface::Instance().instrument(false, dg_prof_params);
    }
};

}; // namespace deep_gemm
