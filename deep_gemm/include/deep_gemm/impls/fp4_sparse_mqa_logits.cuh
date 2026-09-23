#pragma once
#include <deep_gemm/impls/sparse_mqa_logits_layout.cuh>
#include <deep_gemm/impls/mqa_logits_utils.cuh>

namespace cutlass::gemm::kernel {

namespace SML = deep_gemm::sparse_mqa_logits;

// ============================================================================
// FP4 Sparse MQA Logits Kernel (DeepSeek V4.1 DSA indexer, contiguous and paged KV)
// ============================================================================
// Consumes the metadata produced by the sparse metadata kernel: one
// ScheduleEntry per (Q-pair, 128-token KV tile) work unit, one column per
// schedule slot (num_cu x tb_per_cu resident CTAs). Persistent CTA loop:
//   for each wave: entry = schedule[wave * kNumSlots + blockIdx.x]
//     load Q (2 tokens x 32 heads packed fp4) + q_sf + weights once (AIU copy)
//     for each kv_split in the entry: gather the split's SPARSE_BLOCK_KV-token
//     blocks from the KV pool by physical block id (one AIU copy per block —
//     the AIU writer and the tc02.ldmatrix.swzl reader share the same hardware
//     smem swizzle, so manual row-major writes are NOT readable back correctly),
//     MMA (A=KV, B=Q), ReLU + weighted-sum epilogue, and write each bf16 logit
//     to its compressed column (slot_base + slot_offset) * sparse_block_kv +
//     token_in_block.
// Split metadata (slot bases / offsets) is read straight from global by the
// consuming threads. Both KV layouts use a three-stage asynchronous ring
// with the dense kernel's warp-interleave schedule and prefetched SF indices.
// Full contiguous splits use one whole-tile AIU copy; other splits retain
// per-block gathers.
// ============================================================================
template <typename ElementQK, typename ElementAcc, typename ElementLogits, typename ElementWeights,
          int BLOCK_QH, int BLOCK_KV, int WARP_QH, int WARP_KV, int kNumQStages, int kNumKVStages,
          uint32_t SPARSE_BLOCK_KV, uint32_t kNumSlots, bool kIsPaged = false, uint32_t PAGE_KV = 0>
class PPUSparseMqaLogitsFP4 {
public:
    static_assert(cute::is_same_v<ElementQK, uint8_t>, "FP4 sparse MQA logits requires uint8_t ElementQK");
    static_assert(SPARSE_BLOCK_KV == 8 or SPARSE_BLOCK_KV == 16, "Invalid sparse block size");
    static_assert(BLOCK_KV == deep_gemm::sparse_mqa_logits::kSplitKV, "BLOCK_KV must equal one split");
    static_assert(not kIsPaged or PAGE_KV % SPARSE_BLOCK_KV == 0, "Invalid page shape");

    static constexpr int kNumHeads = SML::kNumHeads;            // 32
    static constexpr int kHeadDim = 64;                          // packed (128 / 2)
    static constexpr int kNumKVBlocksPerSplit = SML::kSplitKV / SPARSE_BLOCK_KV;

    static constexpr int BLOCK_M = BLOCK_KV;
    static constexpr int BLOCK_N = BLOCK_QH;
    static constexpr int BLOCK_K = kHeadDim;                     // 64 (packed)
    static constexpr int WARP_M = WARP_KV;
    static constexpr int WARP_N = WARP_QH;
    static constexpr int BLOCK_Q = BLOCK_QH / kNumHeads;         // 2
    static constexpr int WARP_Q = WARP_QH / kNumHeads;           // 1

    using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
    static constexpr int WarpOnM = BLOCK_M / WARP_M;
    static constexpr int WarpOnN = BLOCK_N / WARP_N;

    // MMA: F32F4F4F32 (16x16x64)
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
    // Three-stage pipeline with dense-style warp interleaving.
    static_assert(kNumKVStages == 3, "KV pipeline requires exactly three stages");
    static constexpr bool WarpInterleaving = NumThreadsPerCTA == 512;
    static_assert(NumThreadsPerCTA == 512 && 32 + BLOCK_M <= 256,
                  "KV/SF pipeline producers must fit in warp group 0");

    // AIU operands for Q (B operand; KV uses per-block AIU gather)
    using DefaultOperandB =
        cutlass::gemm::config::DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, ElementQK, false, Int<BLOCK_N>, Int<BLOCK_K>, true>;
    using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom;
    using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
    using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;
    using SmemLayoutB =
        decltype(tile_to_shape(SmemLayoutAtomB{}, Shape<Int<BLOCK_N>, Int<BLOCK_K>, Int<kNumQStages>>{}));

    // Use the dense KV layout and hardware swizzle.
    using DefaultOperandA =
        cutlass::gemm::config::DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, ElementQK, false, Int<BLOCK_M>, Int<BLOCK_K>, false>;
    using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom;
    using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
    using SmemLayoutA =
        decltype(tile_to_shape(SmemLayoutAtomA{}, Shape<Int<BLOCK_M>, Int<BLOCK_K>, Int<kNumKVStages>>{}));

    // AIU gathers preserve the hardware swizzle; block offsets are swizzle-aligned.
    using DefaultOperandAGather =
        cutlass::gemm::config::DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, ElementQK, false, Int<SPARSE_BLOCK_KV>, Int<BLOCK_K>, false>;
    using GatherAiuCopy = typename DefaultOperandAGather::CopyInst;
    using ContiguousAiuCopy = typename DefaultOperandA::CopyInst;

    // SFA / SFB / Weight AIU operands
    using DefaultOperandSFA =
        cutlass::gemm::config::DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, uint32_t, false, _1, Int<BLOCK_M>, false, 0, false>;
    using SmemLayoutAtomSFA = typename DefaultOperandSFA::SmemLayoutAtom;
    using SmemLayoutSFA =
        decltype(tile_to_shape(SmemLayoutAtomSFA{}, make_shape(_1{}, Int<BLOCK_M>{}, Int<kNumKVStages>{})));

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

    MQA_DEFINE_FP4_S2R_LAYOUTS(WarpOnM, MmaIterM, kNumKVStages, BLOCK_M,
                               WarpOnN, MmaIterN, kNumQStages, BLOCK_N)
    using WeightCopyAtomType = cute::conditional_t<cute::is_same_v<ElementWeights, __ppu_bfloat16>, uint16_t, uint64_t>;
    using SmemTiledCopyWeights = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<WeightCopyAtomType>, ElementWeights>{},
        Layout<Shape<Shape<_4, Int<WarpOnN>>, _1>, Stride<Stride<_1, _4>, _1>>{}, Layout<Shape<_2, _1>>{}));

    // Shared memory
    struct SharedStorage {
        cute::array_aligned<ElementQK, cute::cosize_v<SmemLayoutA>> smem_k;       // gathered KV (packed FP4)
        cute::array_aligned<ElementQK, cute::cosize_v<SmemLayoutB>> smem_q;       // Q packed FP4
        cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutSFA>> smem_k_sf;   // KV e8m0 scales
        cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutSFB>> smem_q_sf;   // Q e8m0 scales
        cute::array_aligned<ElementWeights, cute::cosize_v<SmemLayoutWeight>> smem_weight;
    };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    // Contiguous KV uses ptr_k/k_sf; paged KV uses kv_cache/kv_page_stride_bytes.
    struct Arguments {
        const ElementQK* ptr_q;
        const uint32_t* q_sf;
        const ElementQK* ptr_k;
        const uint32_t* k_sf;
        const uint8_t* kv_cache;
        uint32_t kv_page_stride_bytes;
        const ElementWeights* weights;
        const uint8_t* metadata;
        ElementLogits* logits;
        const uint32_t logits_stride;
    };

    using Params = Arguments;

    CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
        int warp_idx = canonical_warp_idx_sync();
        int thread_idx = int(threadIdx.x);
        int warp_m_idx = warp_idx % WarpOnM;
        int warp_n_idx = warp_idx / WarpOnM;
        int lane_idx = get_lane_idx();
        uint32_t warp_q_idx = warp_idx / WarpOnM;

        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        // AIU copy init (Q / SFB / weights only)
        GmemTiledCopyB gmem_tiled_copy_B;
        GmemTiledCopySFB gmem_tiled_copy_SFB;
        GmemTiledCopyWeight gmem_tiled_copy_weight;
        using TilerB = typename GmemTiledCopyB::Tiler_MN;
        using TilerSFB = typename GmemTiledCopySFB::Tiler_MN;
        using TilerWeight = typename GmemTiledCopyWeight::Tiler_MN;
        using StrideAB = Stride<Int<BLOCK_K>, _1>;
        using StrideSFB = Stride<Int<BLOCK_N>, _1>;
        gmem_tiled_copy_B.desc_.template init<ElementQK, false, get<0>(TilerB{}), get<1>(TilerB{})>(
            nullptr, BLOCK_N, BLOCK_K, StrideAB{});
        gmem_tiled_copy_SFB.desc_.template init<uint32_t, false, get<0>(TilerSFB{}), get<1>(TilerSFB{})>(
            nullptr, 1, BLOCK_N, StrideSFB{});
        gmem_tiled_copy_weight.desc_.template init<ElementWeights, false, get<0>(TilerWeight{}), get<1>(TilerWeight{})>(
            nullptr, 1, BLOCK_N, StrideSFB{});

        // Q gmem/smem partitions
        Tensor mB_nk = make_tensor(make_gmem_ptr(params.ptr_q), Shape<Int<BLOCK_N>, Int<BLOCK_K>>{}, StrideAB{});
        Tensor mB_nk_mix = make_mix_tensor_like(mB_nk);
        Tensor gB = local_tile(mB_nk_mix, TileShape{}, make_coord(0, _, 0), Step<X, _1, _1>{});
        Tensor mSFB_n = make_tensor(make_gmem_ptr(params.q_sf), Shape<_1, Int<BLOCK_N>>{}, StrideSFB{});
        Tensor mSFB_n_mix = make_mix_tensor_like(mSFB_n);
        Tensor gSFB = local_tile(mSFB_n_mix, Shape<_1, Int<BLOCK_N>>{}, make_coord(0, _));
        Tensor mW_n = make_tensor(make_gmem_ptr(params.weights), Shape<_1, Int<BLOCK_N>>{}, StrideSFB{});
        Tensor mW_n_mix = make_mix_tensor_like(mW_n);
        Tensor gW = local_tile(mW_n_mix, Shape<_1, Int<BLOCK_N>>{}, make_coord(0, _));

        Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutB{});

        auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
        Tensor tBgB = gmem_thr_copy_B.partition_S(gB);
        Tensor tBsB = gmem_thr_copy_B.partition_D(sB);

        Tensor sSFA = make_tensor(make_smem_ptr(shared_storage.smem_k_sf.data()), SmemLayoutSFA{});
        Tensor sSFB = make_tensor(make_smem_ptr(shared_storage.smem_q_sf.data()), SmemLayoutSFB{});
        Tensor sW = make_tensor(make_smem_ptr(shared_storage.smem_weight.data()), SmemLayoutWeight{});

        auto gmem_thr_copy_SFB = gmem_tiled_copy_SFB.get_thread_slice(thread_idx);
        auto gmem_thr_copy_weight = gmem_tiled_copy_weight.get_thread_slice(thread_idx);
        Tensor tSFBgSFB = gmem_thr_copy_SFB.partition_S(gSFB);
        Tensor tSFBsSFB = gmem_thr_copy_SFB.partition_D(sSFB);
        Tensor tWgW = gmem_thr_copy_weight.partition_S(gW);
        Tensor tWsW = gmem_thr_copy_weight.partition_D(sW);

        // MMA setup
        TiledMma tiled_mma;
        Tensor accum = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
        Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));

        // S2R copies for A / B
        auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
        auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(warp_idx * 32);
        Tensor tCsA = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));
        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);

        auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
        auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(warp_idx * 32);
        Tensor tCsB = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);

        // SFA s2r setup
        Tensor sSFAUint16 = recast<uint16_t>(sSFA);
        Tensor sSFATrans = make_tensor(sSFAUint16.data(), sSFATransLayout{});
        SmemTiledCopySFA smem_tiled_copy_SFA;
        auto smem_thr_copy_SFA = smem_tiled_copy_SFA.get_slice(warp_m_idx * 32 + lane_idx);
        Tensor tCsSFA = smem_thr_copy_SFA.partition_S(sSFATrans);
        Tensor tCrSFA_stage = smem_thr_copy_SFA.partition_D(sSFATrans(_, _, 0));
        using SFARegType = decltype(make_fragment_like(tCrSFA_stage));
        SFARegType tCrSFA_copy_view;
        Tensor tCrSFA = recast<uint32_t>(tCrSFA_copy_view);

        // SFB s2r setup
        Tensor sSFBUint16 = recast<uint16_t>(sSFB);
        Tensor sSFBTrans = make_tensor(sSFBUint16.data(), sSFBTransLayout{});
        SmemTiledCopySFB smem_tiled_copy_SFB;
        auto smem_thr_copy_SFB = smem_tiled_copy_SFB.get_slice(warp_n_idx * 32 + lane_idx);
        Tensor tCsSFB = smem_thr_copy_SFB.partition_S(sSFBTrans);
        Tensor tCrSFB_stage = smem_thr_copy_SFB.partition_D(sSFBTrans(_, _, 0));
        using SFBRegType = decltype(make_fragment_like(tCrSFB_stage));
        SFBRegType tCrSFB_copy_view;
        Tensor tCrSFB = recast<uint32_t>(tCrSFB_copy_view);

        // Weights s2r setup
        constexpr int elem_weights = kNumHeads / 4;
        Tensor sW_copy = make_tensor(sW.data(), sWCopyLayout{});
        SmemTiledCopyWeights smem_tiled_copy_weights;
        auto smem_thr_copy_weights = smem_tiled_copy_weights.get_slice(warp_n_idx * 4 + lane_idx % 4);
        Tensor tCsW = smem_thr_copy_weights.partition_S(sW_copy);
        Tensor tCrW_copy_view = make_tensor_like(tCsW(_, _, _, 0));
        using WeightRegLayout = Layout<Shape<Shape<_2, _2, _2>, Int<MmaIterN>>, Stride<Stride<_1, _0, _2>, _4>>;
        ElementLogits weights[elem_weights];
        Tensor tCrW = make_tensor(static_cast<ElementLogits*>(weights), WeightRegLayout{});

        // Metadata handles
        const auto header = reinterpret_cast<const SML::MetadataHeader*>(params.metadata);
        const auto kv_splits = reinterpret_cast<const SML::KVSplit<kNumKVBlocksPerSplit>*>(
            params.metadata + sizeof(SML::MetadataHeader));
        const auto schedule_entries = reinterpret_cast<const SML::ScheduleEntry*>(
            kv_splits + header->num_kv_splits);

        // Warp 0 issues KV AIU copies; SF producers copy one u32 per token.

        cute::AiuDesc gather_desc;
        gather_desc.template init<ElementQK, false, SPARSE_BLOCK_KV, BLOCK_K>(
            nullptr, SPARSE_BLOCK_KV, BLOCK_K, StrideAB{});
        cute::AiuDesc contiguous_desc;
        if constexpr (not kIsPaged)
            contiguous_desc.template init<ElementQK, false, BLOCK_M, BLOCK_K>(
                nullptr, BLOCK_M, BLOCK_K, StrideAB{});
        constexpr int kStageBytes = BLOCK_M * BLOCK_K;
        constexpr int kGatherChunkBytes = SPARSE_BLOCK_KV * BLOCK_K;

        auto tQgQ = tBgB;
        auto tSFQgSFQ = tSFBgSFB;
        auto tWgW_base = tWgW;
        auto load_q_g2s = [&](const SML::ScheduleEntry& entry) {
            const uint32_t q_offset = entry.q_token_base * kNumHeads;
            gmem_tiled_copy_B.desc_.dim_h = entry.num_q_tokens * kNumHeads;
            gmem_tiled_copy_SFB.desc_.dim_w = entry.num_q_tokens * kNumHeads;
            gmem_tiled_copy_weight.desc_.dim_w = entry.num_q_tokens * kNumHeads;
            tBgB.data() = tQgQ.data() + q_offset * BLOCK_K;
            tSFBgSFB.data() = tSFQgSFQ.data() + q_offset;
            tWgW.data() = tWgW_base.data() + q_offset;
            copy_aiu(gmem_tiled_copy_B, tBgB(_, _, _, 0), tBsB(_, _, _, 0), warp_idx);
            copy_aiu<true>(gmem_tiled_copy_SFB, tSFBgSFB(_, _, _, 0), tSFBsSFB(_, _, _, 0), gmem_tiled_copy_weight,
                           tWgW(_, _, _, 0), tWsW(_, _, _, 0), warp_idx);
        };

        auto load_q_s2r = [&]() {
            copy(smem_tiled_copy_B, tCsB(_, _, _, 0), tCrB_copy_view);
            copy(smem_tiled_copy_SFB, tCsSFB(_, _, _, 0), tCrSFB_copy_view);
            copy(smem_tiled_copy_weights, tCsW(_, _, _, 0), tCrW_copy_view);
            deep_gemm::load_weights_from_copy_view<ElementWeights, ElementLogits>(
                tCrW_copy_view, weights, elem_weights);
        };

        // Contiguous physical IDs are token offsets; paged IDs are sparse-block offsets.
        // Each page stores packed KV followed by SF values.
        const auto resolve_block = [&](const uint32_t physical, const uint32_t token_in_block,
                                       const ElementQK** src_kv, uint32_t* src_sf) {
            if constexpr (kIsPaged) {
                constexpr uint32_t kNumKVBlocksPerPage = PAGE_KV / SPARSE_BLOCK_KV;
                const uint32_t page_idx = physical / kNumKVBlocksPerPage;
                const uint32_t block_in_page = physical % kNumKVBlocksPerPage;
                const uint8_t* page = params.kv_cache +
                                      static_cast<uint64_t>(page_idx) * params.kv_page_stride_bytes;
                *src_kv = reinterpret_cast<const ElementQK*>(page + block_in_page * (SPARSE_BLOCK_KV * BLOCK_K));
                *src_sf = *reinterpret_cast<const uint32_t*>(
                    page + PAGE_KV * BLOCK_K + (block_in_page * SPARSE_BLOCK_KV + token_in_block) * sizeof(uint32_t));
            } else {
                *src_kv = params.ptr_k + static_cast<uint64_t>(physical) * BLOCK_K + token_in_block * BLOCK_K;
                *src_sf = __ldg(params.k_sf + static_cast<uint64_t>(physical) + token_in_block);
            }
        };

        auto load_kv_gather = [&](uint32_t kv_split_idx, uint32_t kv_stage_idx,
                                  bool prefetched = false, uint32_t prefetched_physical = 0) {
            const auto& split = kv_splits[kv_split_idx];
            if (thread_idx == 0) {
                bool is_contiguous = false;
                if constexpr (not kIsPaged)
                    is_contiguous = SML::KVSplitHeader::is_contiguous(__ldg(&split.header.packed_num_kv_blocks));
                if (is_contiguous) {
                    // Copy full contiguous splits with one swizzled AIU transfer.
                    const uint32_t physical = __ldg(&split.kv_block_infos[0].physical_kv_block_idx);
                    ContiguousAiuCopy::copy(shared_storage.smem_k.data() + kv_stage_idx * kStageBytes,
                                           params.ptr_k + static_cast<uint64_t>(physical) * BLOCK_K,
                                           contiguous_desc, 0, 0, 0);
                } else {
                    CUTE_UNROLL
                    for (int b = 0; b < kNumKVBlocksPerSplit; ++b) {
                        // Resolve the block address for the selected KV layout.
                        const uint32_t physical = __ldg(&split.kv_block_infos[b].physical_kv_block_idx);
                        const ElementQK* src_kv;
                        uint32_t src_sf;
                        resolve_block(physical, 0, &src_kv, &src_sf);
                        ElementQK* dst_kv = shared_storage.smem_k.data() +
                                            kv_stage_idx * kStageBytes + b * kGatherChunkBytes;
                        GatherAiuCopy::copy(dst_kv, src_kv, gather_desc, 0, 0, 0);
                    }
                }
            }
            // Keep KV/SF producers in warp group 0 to avoid stale-stage writes.
            if (thread_idx >= 32 && thread_idx < 32 + BLOCK_M) {
                const uint32_t row = thread_idx - 32;
                const uint32_t physical = prefetched ? prefetched_physical : __ldg(
                    &split.kv_block_infos[row / SPARSE_BLOCK_KV].physical_kv_block_idx);
                const uint32_t* src;
                if constexpr (kIsPaged) {
                    constexpr uint32_t kBlocksPerPage = PAGE_KV / SPARSE_BLOCK_KV;
                    const uint8_t* page = params.kv_cache +
                        static_cast<uint64_t>(physical / kBlocksPerPage) * params.kv_page_stride_bytes;
                    // A page stores packed KV first, followed by the SF region.
                    src = reinterpret_cast<const uint32_t*>(page + PAGE_KV * BLOCK_K) +
                          (physical % kBlocksPerPage) * SPARSE_BLOCK_KV + row % SPARSE_BLOCK_KV;
                } else {
                    src = params.k_sf + physical + row % SPARSE_BLOCK_KV;
                }
                uint32_t* dst = shared_storage.smem_k_sf.data() +
                                SmemLayoutSFA{}(0, row, kv_stage_idx);
                PPU_CP_ASYNC_CACHEALWAYS<uint32_t>::copy(*src, *dst);
            }
        };

        // Epilogue: ReLU + weighted head-sum, written to compressed columns
        const auto& warp_offset = warp_m_idx * WARP_M;
        const auto& v_0_offset = lane_idx / 4 + 0;
        const auto& v_1_offset = lane_idx / 4 + 8;
        auto epilogue_mblock = [&](uint32_t kv_split_idx, int m_block,
                                   const uint32_t q0_slot_base, const uint32_t q1_slot_base,
                                   const uint32_t q_token_base, const uint32_t num_q_tokens, const uint32_t slot_offsets_0,
                                   const uint32_t slot_offsets_1) {
            constexpr int kTotalTransforms = 4 * size<2>(accum);
            __ppu_bfloat162 sum_0 = {0, 0};
            __ppu_bfloat162 sum_1 = {0, 0};
            uint32_t cvt_buf[kTotalTransforms];
            deep_gemm::cvt_accum_to_bf16x2_buf<kTotalTransforms>(accum, m_block, cvt_buf);
            __ppu_sched_bound();
            deep_gemm::fma2_phase_tCrW<kTotalTransforms>(cvt_buf, tCrW, sum_0, sum_1);
            __ppu_bfloat16 v_0_bf = __hadd(__low2bfloat16(sum_0), __high2bfloat16(sum_0));
            __ppu_bfloat16 v_1_bf = __hadd(__low2bfloat16(sum_1), __high2bfloat16(sum_1));
            __ppu_bfloat162 packed = deep_gemm::shfl_xor_reduce_bf16x2({v_0_bf, v_1_bf});
            if (warp_q_idx >= num_q_tokens)
                return;

            const uint32_t q_slot_base = warp_q_idx == 0 ? q0_slot_base : q1_slot_base;
            const uint32_t output_row_offset = (q_token_base + warp_q_idx) * params.logits_stride;
            const uint32_t rows[2] = {warp_offset + m_block * InstM + v_0_offset,
                                      warp_offset + m_block * InstM + v_1_offset};
            const __ppu_bfloat16 values[2] = {__low2bfloat16(packed), __high2bfloat16(packed)};
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const uint32_t token_in_block = rows[i] % SPARSE_BLOCK_KV;
                const uint32_t packed_slot_offsets =
                    i == 0 ? slot_offsets_0 : slot_offsets_1;
                const uint32_t q_slot_offset = (packed_slot_offsets >> (warp_q_idx * SML::kNumSparseSlotBits)) &
                                               SML::kInvalidSparseSlot;
                if (q_slot_offset != SML::kInvalidSparseSlot)
                    params.logits[output_row_offset + (q_slot_base + q_slot_offset) * SPARSE_BLOCK_KV + token_in_block] =
                        values[i];
            }
        };

        // Persistent wave loop
        clear(accum);  // partition_fragment_C leaves registers uninitialized
        const uint32_t num_waves = header->num_waves;
        for (uint32_t wave_idx = 0; wave_idx < num_waves; ++wave_idx) {
            const auto entry = schedule_entries[wave_idx * kNumSlots + blockIdx.x];
            if (entry.kv_split_begin == entry.kv_split_end)
                continue;

            // The previous entry drained copies and synchronized.
            load_q_g2s(entry);
            const uint32_t num_splits = entry.kv_split_end - entry.kv_split_begin;
            // Prime two stages; empty commits preserve wait<1> for short entries.
            for (uint32_t pipe = 0; pipe < kNumKVStages - 1; ++pipe) {
                if (pipe < num_splits)
                    load_kv_gather(entry.kv_split_begin + pipe, pipe);
                cp_async_fence();
            }
            cp_async_wait<kNumKVStages - 2>();
            __syncthreads();
            load_q_s2r();

            deep_gemm::warp_interleave_start<WarpInterleaving>(warp_idx / 8, NumThreadsPerCTA);

            // Prefetch SF indices before the warp-group handoff.
            uint32_t next_sf_physical = 0;
            {
                const uint32_t next = kNumKVStages - 1;
                if (next < num_splits && thread_idx >= 32 && thread_idx < 32 + BLOCK_M)
                    next_sf_physical = __ldg(&kv_splits[entry.kv_split_begin + next]
                        .kv_block_infos[(thread_idx - 32) / SPARSE_BLOCK_KV].physical_kv_block_idx);
            }

            uint32_t kv_stage_idx = 0;
            for (uint32_t split_in_entry = 0; split_in_entry < num_splits; ++split_in_entry) {
                const uint32_t kv_split_idx = entry.kv_split_begin + split_in_entry;
                const uint32_t q0_slot_base = __ldg(&kv_splits[kv_split_idx].header.q0_slot_base);
                const uint32_t q1_slot_base = __ldg(&kv_splits[kv_split_idx].header.q1_slot_base);

                deep_gemm::warp_interleave_sync<WarpInterleaving>(warp_idx / 8, NumThreadsPerCTA);
                {
                    // Wait for group 1 before reusing its stage.
                    const uint32_t next = split_in_entry + kNumKVStages - 1;
                    if (next < num_splits)
                        load_kv_gather(entry.kv_split_begin + next, kv_stage_idx == 0 ? kNumKVStages - 1 : kv_stage_idx - 1, true, next_sf_physical);
                    cp_async_fence();
                }
                next_sf_physical = 0;
                {
                    const uint32_t next = split_in_entry + kNumKVStages;
                    if (next < num_splits && thread_idx >= 32 && thread_idx < 32 + BLOCK_M)
                        next_sf_physical = __ldg(&kv_splits[entry.kv_split_begin + next]
                            .kv_block_infos[(thread_idx - 32) / SPARSE_BLOCK_KV].physical_kv_block_idx);
                }
                constexpr int M_BLOCK = size<1>(accum);
                constexpr int N_BLOCK = size<2>(accum);
                constexpr int K_BLOCK = size<2>(tCrA);
                constexpr int m_group = M_BLOCK / size<1>(tCrSFA);
                constexpr int n_group = N_BLOCK / size<1>(tCrSFB);
                for_each(make_int_sequence<M_BLOCK>{}, [&](auto m_block) {
                    if constexpr (m_block > 0)
                        deep_gemm::warp_interleave_sync<WarpInterleaving>(warp_idx / 8, NumThreadsPerCTA);
                    // Prefetch output metadata before MMA.
                    const uint32_t row0 = warp_offset + m_block * InstM + v_0_offset;
                    const uint32_t row1 = warp_offset + m_block * InstM + v_1_offset;
                    const uint32_t slot_offsets_0 = __ldg(&kv_splits[kv_split_idx].kv_block_infos[row0 / SPARSE_BLOCK_KV].packed_slot_offsets);
                    const uint32_t slot_offsets_1 = __ldg(&kv_splits[kv_split_idx].kv_block_infos[row1 / SPARSE_BLOCK_KV].packed_slot_offsets);
                    for_each(make_int_sequence<K_BLOCK>{}, [&](auto k_block) {
                        copy(smem_tiled_copy_A, tCsA(_, _, k_block, kv_stage_idx), tCrA_copy_view(_, _, k_block));
                        copy(smem_tiled_copy_SFA, tCsSFA(_, _, k_block, kv_stage_idx), tCrSFA_copy_view(_, _, k_block));
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
                    deep_gemm::warp_interleave_arrive<WarpInterleaving>(warp_idx / 8, NumThreadsPerCTA);
                    epilogue_mblock(kv_split_idx, m_block, q0_slot_base, q1_slot_base,
                                    entry.q_token_base, entry.num_q_tokens, slot_offsets_0, slot_offsets_1);
                });
                cp_async_wait<kNumKVStages - 2>();
                if constexpr (not WarpInterleaving)
                    __syncthreads();
                clear(accum);
                kv_stage_idx = kv_stage_idx + 1 == kNumKVStages ? 0 : kv_stage_idx + 1;
            }

            deep_gemm::warp_interleave_end<WarpInterleaving>(warp_idx / 8, NumThreadsPerCTA);
            cp_async_wait<0>();
            __syncthreads();
        }
    }
};

} // namespace cutlass::gemm::kernel
