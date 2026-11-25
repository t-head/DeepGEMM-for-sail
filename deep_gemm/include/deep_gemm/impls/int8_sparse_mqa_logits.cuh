#pragma once

#include <deep_gemm/impls/mqa_logits_utils.cuh>
#include <deep_gemm/impls/sparse_mqa_logits_layout.cuh>
#include <deep_gemm/tix/ld_st.cuh>
#include <deep_gemm/tix/utils.cuh>

namespace cutlass::gemm::kernel {

namespace SML = deep_gemm::sparse_mqa_logits;

// PPU0010-only INT8 sparse MQA. Non-paged physical indices are token offsets (including
// unaligned ks); paged physical indices are sparse-block offsets in the page pool.
// Q scales have already been folded into the BF16 per-head weights. The MMA and
// BF16 epilogue match PPUMqaLogits; only the KV gather and output slots differ.
template <typename ElementQK, typename ElementAcc, typename ElementLogits, typename ElementWeights,
          int BLOCK_QH, int BLOCK_KV, int WARP_QH, int WARP_KV, int kNumQStages, int kNumKVStages,
          uint32_t SPARSE_BLOCK_KV, uint32_t kNumSlots, bool kIsPaged = false, uint32_t PAGE_KV = 0>
class PPUSparseMqaLogitsINT8 {
public:
    static_assert(cute::is_same_v<ElementQK, int8_t> && cute::is_same_v<ElementAcc, int32_t>,
                  "INT8 sparse MQA requires signed INT8 inputs and INT32 accumulation");
    static_assert(cute::is_same_v<ElementLogits, __ppu_bfloat16> &&
                  cute::is_same_v<ElementWeights, __ppu_bfloat16>,
                  "INT8 sparse MQA requires BF16 weights and logits");
    static_assert(SPARSE_BLOCK_KV == 8 || SPARSE_BLOCK_KV == 16, "Invalid sparse block size");
    static_assert(!kIsPaged || (PAGE_KV > 0 && PAGE_KV % SPARSE_BLOCK_KV == 0), "Invalid page shape");
    static_assert(BLOCK_QH == SML::kBlockQ * SML::kNumHeads && BLOCK_KV == SML::kSplitKV,
                  "INT8 sparse MQA must preserve the metadata's Q-pair and KV-split layout");
    static_assert(WARP_QH == SML::kNumHeads && (WARP_KV == 16 || WARP_KV == 32),
                  "One warp must cover all heads of one Q token");
    static_assert(kNumQStages == 1 && kNumKVStages == 3, "Expected one Q and three KV stages");

    static constexpr int kNumHeads = SML::kNumHeads;
    static constexpr int kHeadDim = SML::kHeadDim;
    static constexpr int kNumKVBlocksPerSplit = BLOCK_KV / SPARSE_BLOCK_KV;
    static constexpr int BLOCK_M = BLOCK_KV;
    static constexpr int BLOCK_N = BLOCK_QH;
    static constexpr int BLOCK_K = kHeadDim;
    static constexpr int WarpOnM = BLOCK_M / WARP_KV;
    static constexpr int WarpOnN = BLOCK_N / WARP_QH;
    static constexpr int InstM = 16;
    static constexpr int InstN = 16;
    static constexpr int MmaIterM = WARP_KV / InstM;
    static constexpr int MmaIterN = WARP_QH / InstN;

    using ArchTag = cutlass::arch::PPU0010;
    using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<
        ArchTag, ElementQK, ElementQK, ElementAcc>::type;
    using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
    using PermutationMNK = Tile<
        Layout<Shape<Int<InstM>, Int<WarpOnM>, Int<MmaIterM>>, Stride<_1, Int<WARP_KV>, Int<InstM>>>,
        Layout<Shape<Int<InstN>, Int<WarpOnN>, Int<MmaIterN>>, Stride<_1, Int<WARP_QH>, Int<InstN>>>, _32>;
    using TiledMma = TiledMMA<MMA_Atom<MmaInst>, Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, _1>>,
                             PermutationMNK>;
    static constexpr int NumThreadsPerCTA = size(TiledMma{});
    static constexpr int MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));
    // Guide register allocation toward four resident CTAs on PPU0010. Actual
    // residency also depends on other resources; the host selects grid slots separately.
    static constexpr int MinBlocksPerMultiprocessor = 4;
    static_assert(NumThreadsPerCTA >= BLOCK_M, "KV scale producers require one thread per row");

    // Use the same PPU0010 AIU/swizzled S2R pair as dense INT8.
    using DefaultOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<
        ArchTag, ElementQK, false, Int<BLOCK_M>, Int<BLOCK_K>, false>;
    using DefaultOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<
        ArchTag, ElementQK, false, Int<BLOCK_N>, Int<BLOCK_K>, true>;
    using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
    using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
    using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;
    using ContiguousAiuCopy = typename DefaultOperandA::CopyInst;
    using SmemLayoutA = decltype(tile_to_shape(typename DefaultOperandA::SmemLayoutAtom{},
        Shape<Int<BLOCK_M>, Int<BLOCK_K>, Int<kNumKVStages>>{}));
    using SmemLayoutB = decltype(tile_to_shape(typename DefaultOperandB::SmemLayoutAtom{},
        Shape<Int<BLOCK_N>, Int<BLOCK_K>, Int<kNumQStages>>{}));

    using DefaultOperandSFA = cutlass::gemm::config::DefaultGemm_AIU_Operand<
        ArchTag, float, false, _1, Int<BLOCK_M>, false, 0, false>;
    using DefaultOperandWeights = cutlass::gemm::config::DefaultGemm_AIU_Operand<
        ArchTag, ElementWeights, false, _1, Int<BLOCK_N>, false, 0, false>;
    using GmemTiledCopyWeights = typename DefaultOperandWeights::GmemTiledCopy;
    using SmemLayoutSFA = decltype(tile_to_shape(typename DefaultOperandSFA::SmemLayoutAtom{},
        Shape<_1, Int<BLOCK_M>, Int<kNumKVStages>>{}));
    using SmemLayoutWeights = decltype(tile_to_shape(typename DefaultOperandWeights::SmemLayoutAtom{},
        Shape<_1, Int<BLOCK_N>, Int<kNumQStages>>{}));
    using StrideAB = Stride<Int<BLOCK_K>, _1>;
    using StrideWeights = Stride<Int<BLOCK_N>, _1>;
    using KVSplit = SML::KVSplit<kNumKVBlocksPerSplit>;
    // Iteration i commits metadata i+4 with KV i+2. Iteration i+1's wait<1>
    // and CTA barrier publish that metadata before iteration i+2 loads KV i+4.
    // Eight slots keep each record intact until its epilogue readers retire.
    // Host dynamic shared memory must include all eight metadata records.
    static constexpr uint32_t kNumMetadataStages = 8;
    static constexpr uint32_t kMetadataPrefetchDistance = kNumKVStages + 1;
    static constexpr uint32_t kMetadataWordsPerSplit = sizeof(KVSplit) / sizeof(uint32_t);
    static constexpr uint32_t kMetadataVectorsPerSplit = sizeof(KVSplit) / sizeof(uint4);
    static_assert(sizeof(KVSplit) % sizeof(uint4) == 0 && alignof(KVSplit) >= alignof(uint4),
                  "Async metadata requires whole aligned 16-byte vectors");

    struct SharedStorage {
        cute::array_aligned<ElementQK, cute::cosize_v<SmemLayoutA>> smem_k;
        cute::array_aligned<ElementQK, cute::cosize_v<SmemLayoutB>> smem_q;
        cute::array_aligned<float, cute::cosize_v<SmemLayoutSFA>> smem_k_scales;
        cute::array_aligned<ElementWeights, cute::cosize_v<SmemLayoutWeights>> smem_weights;
        cute::array_aligned<KVSplit, kNumMetadataStages> smem_split_metadata;
    };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    // PPU0010 stores each 32-byte channel slice as a separate BLOCK_M-row
    // plane. This is the inverse address map of ppu_tsm_ld_swzl_sim in
    // cute/arch/copy_ppu0010_aiu.hpp, at 16-byte copy granularity. Concatenating
    // independently swizzled 8/16-row AIU blocks would use the wrong plane
    // stride for the dense BLOCK_M-row ldmatrix reader.
    CUTE_HOST_DEVICE static constexpr uint32_t ppu0010_kv_vector_offset(uint32_t row, uint32_t channel_byte) {
        const uint32_t slice = channel_byte / 32;
        const uint32_t line = row / 4;
        const uint32_t vector_in_line = ((row % 4) * 2 + (channel_byte % 32) / 16) ^ (line & 1);
        const uint32_t slice_rotation = (((slice & 1) << 1) + ((slice & 2) >> 1)) * 2;
        return slice * BLOCK_M * 32 + line * 128 + ((vector_in_line + slice_rotation) % 8) * 16;
    }

    struct Arguments {
        const ElementQK* ptr_q;
        const ElementQK* ptr_k;
        const float* k_scales;
        const ElementWeights* weights;
        const uint8_t* metadata;
        ElementLogits* logits;
        uint32_t logits_stride;
        uint32_t num_q_tokens;
        // Non-paged: real K allocation length. Paged: num_pages * PAGE_KV,
        // including the fully allocated last page, not the logical context length.
        uint32_t num_kv_tokens;
        const uint8_t* kv_cache;
        uint32_t kv_page_stride_bytes;
    };
    using Params = Arguments;

    CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
        const int thread_idx = int(threadIdx.x);
        int warp_idx = canonical_warp_idx_sync();
        const int lane_idx = deep_gemm::tix::get_lane_idx();
        const uint32_t warp_q_idx = warp_idx / WarpOnM;
        const uint32_t warp_offset = (warp_idx % WarpOnM) * WARP_KV;
        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        GmemTiledCopyB gmem_tiled_copy_B;
        GmemTiledCopyWeights gmem_tiled_copy_weights;
        using TilerB = typename GmemTiledCopyB::Tiler_MN;
        using TilerWeights = typename GmemTiledCopyWeights::Tiler_MN;
        gmem_tiled_copy_B.desc_.template init<ElementQK, false, get<0>(TilerB{}), get<1>(TilerB{})>(
            nullptr, BLOCK_N, BLOCK_K, StrideAB{});
        gmem_tiled_copy_weights.desc_.template init<ElementWeights, false,
            get<0>(TilerWeights{}), get<1>(TilerWeights{})>(nullptr, 1, BLOCK_N, StrideWeights{});

        Tensor mB = make_tensor(make_gmem_ptr(params.ptr_q), Shape<Int<BLOCK_N>, Int<BLOCK_K>>{}, StrideAB{});
        Tensor gB = local_tile(make_mix_tensor_like(mB), TileShape{}, make_coord(0, _, 0), Step<X, _1, _1>{});
        Tensor mW = make_tensor(make_gmem_ptr(params.weights), Shape<_1, Int<BLOCK_N>>{}, StrideWeights{});
        Tensor gW = local_tile(make_mix_tensor_like(mW), Shape<_1, Int<BLOCK_N>>{}, make_coord(0, _));
        Tensor sA = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutB{});
        Tensor sW = make_tensor(make_smem_ptr(storage.smem_weights.data()), SmemLayoutWeights{});

        auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
        Tensor tBgB = gmem_thr_copy_B.partition_S(gB);
        Tensor tBsB = gmem_thr_copy_B.partition_D(sB);
        auto gmem_thr_copy_weights = gmem_tiled_copy_weights.get_thread_slice(thread_idx);
        Tensor tWgW = gmem_thr_copy_weights.partition_S(gW);
        Tensor tWsW = gmem_thr_copy_weights.partition_D(sW);
        const auto tQgQ = tBgB;
        const auto tWgW_base = tWgW;

        TiledMma tiled_mma;
        Tensor accum = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
        Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));
        auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
        auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(warp_idx * 32);
        Tensor tCsA = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));
        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
        auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
        auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(warp_idx * 32);
        Tensor tCsB = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
        ElementLogits weights[kNumHeads / 4];

        const auto* header = reinterpret_cast<const SML::MetadataHeader*>(params.metadata);
        const uint32_t num_waves = header->num_waves;
        const uint32_t use_unaligned_ks = header->use_unaligned_ks;
        const auto* splits = reinterpret_cast<const KVSplit*>(
            params.metadata + sizeof(SML::MetadataHeader));
        const auto* schedule = reinterpret_cast<const SML::ScheduleEntry*>(splits + header->num_kv_splits);
        cute::AiuDesc contiguous_desc;
        contiguous_desc.template init<ElementQK, false, BLOCK_M, BLOCK_K>(
            nullptr, BLOCK_M, BLOCK_K, StrideAB{});
        constexpr uint32_t kStageBytes = BLOCK_M * BLOCK_K;

        // The tensor view [pages, PAGE_KV, 1, 132] describes the page footprint,
        // not token-interleaved records: each page holds all K bytes, then all
        // FP32 scales. Resolve addresses only after checking the physical token.
        const auto resolve_k = [&](uint32_t physical, uint32_t token_in_block) -> const ElementQK* {
            if constexpr (kIsPaged) {
                constexpr uint32_t kBlocksPerPage = PAGE_KV / SPARSE_BLOCK_KV;
                const uint32_t page_idx = physical / kBlocksPerPage;
                const uint32_t row = (physical % kBlocksPerPage) * SPARSE_BLOCK_KV + token_in_block;
                const uint8_t* page = params.kv_cache + static_cast<uint64_t>(page_idx) * params.kv_page_stride_bytes;
                return reinterpret_cast<const ElementQK*>(page) + static_cast<uint64_t>(row) * BLOCK_K;
            } else {
                return params.ptr_k + (static_cast<uint64_t>(physical) + token_in_block) * BLOCK_K;
            }
        };
        const auto resolve_scale = [&](uint32_t physical, uint32_t token_in_block) -> const float* {
            if constexpr (kIsPaged) {
                constexpr uint32_t kBlocksPerPage = PAGE_KV / SPARSE_BLOCK_KV;
                const uint32_t page_idx = physical / kBlocksPerPage;
                const uint32_t row = (physical % kBlocksPerPage) * SPARSE_BLOCK_KV + token_in_block;
                const uint8_t* page = params.kv_cache + static_cast<uint64_t>(page_idx) * params.kv_page_stride_bytes;
                return reinterpret_cast<const float*>(page + static_cast<uint64_t>(PAGE_KV) * BLOCK_K) + row;
            } else {
                return params.k_scales + static_cast<uint64_t>(physical) + token_in_block;
            }
        };

        auto load_q = [&](const SML::ScheduleEntry& entry, uint32_t valid_q) {
            const uint64_t q_head_offset = static_cast<uint64_t>(entry.q_token_base) * kNumHeads;
            // AIU pads the second Q token on an odd tail; no padded input is required.
            gmem_tiled_copy_B.desc_.dim_h = valid_q * kNumHeads;
            gmem_tiled_copy_weights.desc_.dim_w = valid_q * kNumHeads;
            tBgB.data() = tQgQ.data() + q_head_offset * BLOCK_K;
            tWgW.data() = tWgW_base.data() + q_head_offset;
            copy_aiu(gmem_tiled_copy_B, tBgB(_, _, _, 0), tBsB(_, _, _, 0), warp_idx);
            copy_aiu(gmem_tiled_copy_weights, tWgW(_, _, _, 0), tWsW(_, _, _, 0), warp_idx);
        };

        auto load_metadata = [&](uint32_t split_idx, uint32_t metadata_stage) {
            // Copy each word once. The CTA's existing pipeline barrier publishes
            // these writes before either KV loading or the epilogue consumes it.
            if (thread_idx < kMetadataWordsPerSplit) {
                const auto* src = reinterpret_cast<const uint32_t*>(splits + split_idx);
                auto* dst = reinterpret_cast<uint32_t*>(&storage.smem_split_metadata[metadata_stage]);
                dst[thread_idx] = __ldg(src + thread_idx);
            }
        };

        auto load_metadata_async = [&](uint32_t split_idx, uint32_t metadata_stage) {
            // Nine producers for B8, five for B16. The record and every ring slot
            // are 16-byte aligned, and no vector crosses a KVSplit boundary.
            if (thread_idx < kMetadataVectorsPerSplit) {
                const auto* src = reinterpret_cast<const uint4*>(splits + split_idx);
                auto* dst = reinterpret_cast<uint4*>(&storage.smem_split_metadata[metadata_stage]);
                PPU_CP_ASYNC_CACHEALWAYS<uint4>::copy(src[thread_idx], dst[thread_idx]);
            }
        };

        auto load_kv = [&](uint32_t metadata_stage, uint32_t stage) {
            const auto& split = storage.smem_split_metadata[metadata_stage];
            const uint32_t packed_count = deep_gemm::tix::ld_shared_u32(&split.header.packed_num_kv_blocks);
            const uint32_t num_blocks = SML::KVSplitHeader::get_num_kv_blocks(packed_count);
            const uint32_t first = deep_gemm::tix::ld_shared_u32(&split.kv_block_infos[0].physical_kv_block_idx);
            bool is_contiguous;
            if constexpr (kIsPaged) {
                constexpr uint32_t kBlocksPerPage = PAGE_KV / SPARSE_BLOCK_KV;
                const bool contiguous_hint = SML::KVSplitHeader::is_contiguous(packed_count);
                // A metadata hint proves contiguous blocks in one logical page.
                // Without it, retain the physical-block vote for older metadata
                // and aliases across logical pages. Both paths must stay within
                // one physical page, excluding its scale region and stride gap.
                is_contiguous = false;
                if (num_blocks > 0 && first % kBlocksPerPage + num_blocks <= kBlocksPerPage &&
                    static_cast<uint64_t>(first) * SPARSE_BLOCK_KV < params.num_kv_tokens) {
                    is_contiguous = contiguous_hint;
                    if (!contiguous_hint) {
                        // Every warp independently evaluates the same live blocks.
                        bool consecutive = true;
                        if (lane_idx < num_blocks)
                            consecutive = deep_gemm::tix::ld_shared_u32(
                                &split.kv_block_infos[lane_idx].physical_kv_block_idx) == first + lane_idx;
                        is_contiguous = __all_sync(0xffffffffu, consecutive);
                    }
                }
            } else {
                // The metadata flag describes full tiles only. Aligned merged
                // non-paged IDs are sorted and unique, so endpoint distance
                // also proves contiguity for a partial tile.
                is_contiguous = num_blocks == kNumKVBlocksPerSplit &&
                    SML::KVSplitHeader::is_contiguous(packed_count);
                if (num_blocks > 0 && num_blocks < kNumKVBlocksPerSplit && !use_unaligned_ks) {
                    const uint32_t last = deep_gemm::tix::ld_shared_u32(
                        &split.kv_block_infos[num_blocks - 1].physical_kv_block_idx);
                    is_contiguous = last == first + (num_blocks - 1) * SPARSE_BLOCK_KV;
                }
                is_contiguous = is_contiguous && first < params.num_kv_tokens;
            }
            if (is_contiguous) {
                if (thread_idx == 0) {
                    if constexpr (kIsPaged)
                        contiguous_desc.dim_h = num_blocks * SPARSE_BLOCK_KV;
                    else
                        contiguous_desc.dim_h = min(num_blocks * SPARSE_BLOCK_KV, params.num_kv_tokens - first);
                    ContiguousAiuCopy::copy(storage.smem_k.data() + stage * kStageBytes,
                        resolve_k(first, 0), contiguous_desc, 0, 0, 0);
                }
            } else {
                // Coalesced 16-byte copies scatter directly into the full-stage
                // PPU0010 swizzle. They share the AIU commit/wait pipeline.
                constexpr int kVectorsPerThread = kStageBytes / (16 * NumThreadsPerCTA);
                CUTE_UNROLL
                for (int copy_idx = 0; copy_idx < kVectorsPerThread; ++copy_idx) {
                    const uint32_t vector = thread_idx + copy_idx * NumThreadsPerCTA;
                    const uint32_t row = vector / (BLOCK_K / 16);
                    const uint32_t channel = (vector % (BLOCK_K / 16)) * 16;
                    const uint32_t b = row / SPARSE_BLOCK_KV;
                    const uint32_t physical = b < num_blocks ?
                        deep_gemm::tix::ld_shared_u32(&split.kv_block_infos[b].physical_kv_block_idx) :
                        params.num_kv_tokens;
                    const uint64_t token = static_cast<uint64_t>(physical) * (kIsPaged ? SPARSE_BLOCK_KV : 1u) +
                                           row % SPARSE_BLOCK_KV;
                    auto* dst = reinterpret_cast<uint4*>(storage.smem_k.data() + stage * kStageBytes +
                                                        ppu0010_kv_vector_offset(row, channel));
                    if (token < params.num_kv_tokens) {
                        const auto* src = reinterpret_cast<const uint4*>(
                            resolve_k(physical, row % SPARSE_BLOCK_KV) + channel);
                        PPU_CP_ASYNC_CACHEALWAYS<uint4>::copy(*src, *dst);
                    } else {
                        // No input padding or out-of-bounds source address is needed.
                        *dst = uint4{0, 0, 0, 0};
                    }
                }
            }
            if constexpr (kIsPaged) {
                if (is_contiguous) {
                    // The contiguous path stays inside one physical page, so
                    // every live row uses the same scale base without reloading
                    // its block's physical index or resolving its page again.
                    if (thread_idx < BLOCK_M) {
                        const uint32_t row = thread_idx;
                        const float* scale_base = resolve_scale(first, 0);
                        auto& scale_dst = storage.smem_k_scales[SmemLayoutSFA{}(0, row, stage)];
                        if (row / SPARSE_BLOCK_KV < num_blocks &&
                            static_cast<uint64_t>(first) * SPARSE_BLOCK_KV + row < params.num_kv_tokens) {
                            PPU_CP_ASYNC_CACHEALWAYS<float>::copy(scale_base[row], scale_dst);
                        } else {
                            scale_dst = 0.0f;
                        }
                    }
                    // Only exit this loader; the caller still commits and waits
                    // for the shared K/scale group before the CTA barrier.
                    return;
                }
            }
            if (thread_idx < BLOCK_M) {
                const uint32_t row = thread_idx;
                const uint32_t b = row / SPARSE_BLOCK_KV;
                const uint32_t physical = b < num_blocks ?
                    deep_gemm::tix::ld_shared_u32(&split.kv_block_infos[b].physical_kv_block_idx) :
                    params.num_kv_tokens;
                const uint64_t token = static_cast<uint64_t>(physical) * (kIsPaged ? SPARSE_BLOCK_KV : 1u) +
                                       row % SPARSE_BLOCK_KV;
                auto& scale_dst = storage.smem_k_scales[SmemLayoutSFA{}(0, row, stage)];
                if (token < params.num_kv_tokens) {
                    // A 4-byte copy joins this KV stage's existing async group.
                    // The consumer's wait<1> + CTA barrier publishes both K and scales.
                    PPU_CP_ASYNC_CACHEALWAYS<float>::copy(*resolve_scale(physical, row % SPARSE_BLOCK_KV), scale_dst);
                } else {
                    // Keep invalid tails independent of input padding/address validity.
                    scale_dst = 0.0f;
                }
            }
        };

        auto epilogue = [&](uint32_t metadata_stage, uint32_t stage,
                            const SML::ScheduleEntry& entry, uint32_t valid_q) {
            const auto& split = storage.smem_split_metadata[metadata_stage];
            const uint32_t q_slot_base = warp_q_idx == 0 ?
                deep_gemm::tix::ld_shared_u32(&split.header.q0_slot_base) :
                deep_gemm::tix::ld_shared_u32(&split.header.q1_slot_base);
            const uint32_t num_blocks = SML::KVSplitHeader::get_num_kv_blocks(
                deep_gemm::tix::ld_shared_u32(&split.header.packed_num_kv_blocks));
            CUTE_UNROLL
            for (int m = 0; m < MmaIterM; ++m) {
                constexpr int kTotalTransforms = 4 * MmaIterN;
                uint32_t cvt_buf[kTotalTransforms];
                __ppu_bfloat162 sum_0 = {0, 0}, sum_1 = {0, 0};
                deep_gemm::cvt_accum_to_bf16x2_buf<kTotalTransforms>(accum, m, cvt_buf);
                __ppu_sched_bound();
                deep_gemm::fma2_phase_weights<kTotalTransforms>(cvt_buf, weights, sum_0, sum_1);
                __ppu_bfloat16 v0 = __hadd(__low2bfloat16(sum_0), __high2bfloat16(sum_0));
                __ppu_bfloat16 v1 = __hadd(__low2bfloat16(sum_1), __high2bfloat16(sum_1));
                const uint32_t row0 = warp_offset + m * InstM + lane_idx / 4;
                const uint32_t row1 = row0 + 8;
                // Match the dense BF16 epilogue's scale-before-cross-lane-reduce rounding.
                v0 = __hmul(v0, static_cast<__ppu_bfloat16>(deep_gemm::tix::ld_shared(
                    storage.smem_k_scales.data() + SmemLayoutSFA{}(0, row0, stage))));
                v1 = __hmul(v1, static_cast<__ppu_bfloat16>(deep_gemm::tix::ld_shared(
                    storage.smem_k_scales.data() + SmemLayoutSFA{}(0, row1, stage))));
                const __ppu_bfloat162 result = deep_gemm::shfl_xor_reduce_bf16x2({v0, v1});
                if (warp_q_idx < valid_q) {
                    // Each half covers eight rows. B8/B16 and the aligned warp/MMA
                    // origin make its block index and output base warp-uniform.
                    // Keep lane_idx out of this address chain so the compiler can
                    // use scalar slot predicates and a scalar global base pointer.
                    const uint32_t row_bases[2] = {warp_offset + m * InstM, warp_offset + m * InstM + 8};
                    const __ppu_bfloat16 values[2] = {__low2bfloat16(result), __high2bfloat16(result)};
                    CUTE_UNROLL
                    for (int i = 0; i < 2; ++i) {
                        const uint32_t b = row_bases[i] / SPARSE_BLOCK_KV;
                        if (b < num_blocks) {
                            const uint32_t slots = deep_gemm::tix::ld_shared_u32(
                                &split.kv_block_infos[b].packed_slot_offsets);
                            const uint32_t slot = (slots >> (warp_q_idx * SML::kNumSparseSlotBits)) &
                                                  SML::kInvalidSparseSlot;
                            if (slot != SML::kInvalidSparseSlot) {
                                const uint64_t output_row = static_cast<uint64_t>(entry.q_token_base + warp_q_idx) *
                                                            params.logits_stride;
                                ElementLogits* output_base = params.logits + output_row +
                                    (q_slot_base + slot) * SPARSE_BLOCK_KV + row_bases[i] % SPARSE_BLOCK_KV;
                                // Preserve the one-writer-per-four-lane mask.
                                if (lane_idx % 4 == 0)
                                    output_base[lane_idx / 4] = values[i];
                            }
                        }
                    }
                }
            }
        };

        clear(accum);
        for (uint32_t wave = 0; wave < num_waves; ++wave) {
            const auto entry = schedule[static_cast<uint64_t>(wave) * kNumSlots + blockIdx.x];
            if (entry.kv_split_begin == entry.kv_split_end || entry.q_token_base >= params.num_q_tokens)
                continue;
            const uint32_t valid_q = min(entry.num_q_tokens, params.num_q_tokens - entry.q_token_base);
            const uint32_t num_splits = entry.kv_split_end - entry.kv_split_begin;
            CUTE_UNROLL
            for (uint32_t pipe = 0; pipe < kMetadataPrefetchDistance; ++pipe) {
                if (pipe < num_splits)
                    load_metadata(entry.kv_split_begin + pipe, pipe);
            }
            __syncthreads();
            load_q(entry, valid_q);
            CUTE_UNROLL
            for (uint32_t pipe = 0; pipe < kNumKVStages - 1; ++pipe) {
                if (pipe < num_splits)
                    load_kv(pipe, pipe);
                // Empty groups keep wait<1> correct for one-split entries.
                cp_async_fence();
            }
            cp_async_wait<kNumKVStages - 2>();
            __syncthreads();
            copy(smem_tiled_copy_B, tCsB(_, _, _, 0), tCrB_copy_view);
            deep_gemm::load_weights_from_smem_ld_shared<ElementWeights, ElementLogits, kNumHeads>(
                sW(_, _, 0), weights, lane_idx, warp_q_idx);

            for (uint32_t split = 0; split < num_splits; ++split) {
                const uint32_t stage = split % kNumKVStages;
                const uint32_t next = split + kNumKVStages - 1;
                if (next < num_splits)
                    load_kv(next % kNumMetadataStages, next % kNumKVStages);
                const uint32_t next_metadata = split + kMetadataPrefetchDistance;
                if (next_metadata < num_splits)
                    load_metadata_async(entry.kv_split_begin + next_metadata, next_metadata % kNumMetadataStages);
                cp_async_fence();
                copy(smem_tiled_copy_A, tCsA(_, _, _, stage), tCrA_copy_view);
                cute::gemm(tiled_mma, accum, tCrA, tCrB, accum);
                epilogue(split % kNumMetadataStages, stage, entry, valid_q);
                cp_async_wait<kNumKVStages - 2>();
                // Publishes completed K/scale/metadata copies and retires shared
                // readers before any stage or metadata slot is reused.
                __syncthreads();
                clear(accum);
            }
            // The last iteration issues neither future KV/scales nor metadata.
            // Its empty commit is the only group that wait<1> may leave pending;
            // every real copy and shared-memory user has passed the CTA barrier.
            // No additional drain/barrier is required before the next entry.
        }
    }
};

} // namespace cutlass::gemm::kernel
