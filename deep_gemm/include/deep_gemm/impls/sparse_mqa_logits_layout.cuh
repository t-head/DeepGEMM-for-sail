#pragma once

#include <cstdint>

// Shared layout for sparse MQA logits (DeepSeek V4.1 DSA indexer), ported from open-source
// DeepGEMM 26/09 `layout/sparse_mqa_logits.cuh`. Pure POD + constexpr so that both the host
// translation unit (via csrc includes) and the hggc JIT device code can include it.
//
// MXFP4 only.
namespace deep_gemm {
namespace sparse_mqa_logits {

inline constexpr uint32_t kNumHeads = 32;
inline constexpr uint32_t kHeadDim = 128;
// Two paired Q tokens share one merged KV-block list (merge-path union in the metadata kernel)
inline constexpr uint32_t kBlockQ = 2;
inline constexpr uint32_t kNumSparseSlotBits = 16;
inline constexpr uint32_t kInvalidSparseSlot = (1u << kNumSparseSlotBits) - 1;
// One KV split is one 128-token KV tile of the sparse main kernel (BLOCK_KV = 128).
inline constexpr uint32_t kSplitKV = 128;
// Persistent CTAs per CU: 64 warp-engine slots / 16 warps per CTA = 4.
inline constexpr uint32_t kNumTbPerCu = 4;
inline constexpr uint32_t kContiguousFlag = 0x80000000u;

// Metadata buffer layout (all little-endian u32 words), produced by the metadata kernel:
//   [0, 16)                       MetadataHeader
//   [16, 16 + S * sizeof(KVSplit))  one KVSplit per split: KVSplitHeader + KVBlockInfo[blocks_per_split]
//   [schedule offset)             one ScheduleEntry per (wave, sm): [wave * kNumSMs + sm]
struct alignas(16) MetadataHeader {
    uint32_t num_kv_splits;
    uint32_t num_waves;
    uint32_t use_unaligned_ks;
};
static_assert(sizeof(MetadataHeader) == 16, "MetadataHeader must stay 16 bytes");

struct alignas(16) KVSplitHeader {
    uint32_t q_token_base;
    // num_kv_blocks | (is_contiguous ? kContiguousFlag : 0)
    uint32_t packed_num_kv_blocks;
    uint32_t q0_slot_base;
    uint32_t q1_slot_base;

    KVSplitHeader() = default;

    constexpr KVSplitHeader(const uint32_t q_token_base, const uint32_t num_kv_blocks,
                            const bool is_contiguous, const uint32_t q0_slot_base,
                            const uint32_t q1_slot_base)
        : q_token_base(q_token_base),
          packed_num_kv_blocks(num_kv_blocks | (is_contiguous ? kContiguousFlag : 0u)),
          q0_slot_base(q0_slot_base), q1_slot_base(q1_slot_base) {}

    static constexpr uint32_t get_num_kv_blocks(const uint32_t packed) {
        return packed & ~kContiguousFlag;
    }

    static constexpr bool is_contiguous(const uint32_t packed) {
        return (packed & kContiguousFlag) != 0;
    }
};
static_assert(sizeof(KVSplitHeader) == 16, "KVSplitHeader must stay 16 bytes");

// One sparse KV block within a split: where its 8/16 tokens live physically, and which of the two
// paired Q tokens selected it (present bit per 16-bit slot field) at which output slot offset
// (kInvalidSparseSlot = not selected by that token).
struct KVBlockInfo {
    uint32_t physical_kv_block_idx;
    uint32_t packed_slot_offsets;  // q0_slot_offset | (q1_slot_offset << kNumSparseSlotBits)

    KVBlockInfo() = default;

    constexpr KVBlockInfo(const uint32_t physical_kv_block_idx, const uint32_t q0_slot_offset,
                          const uint32_t q1_slot_offset)
        : physical_kv_block_idx(physical_kv_block_idx),
          packed_slot_offsets(q0_slot_offset | (q1_slot_offset << kNumSparseSlotBits)) {}
};
static_assert(sizeof(KVBlockInfo) == 8, "KVBlockInfo must stay 8 bytes");

struct alignas(16) ScheduleEntry {
    uint32_t kv_split_begin;
    uint32_t kv_split_end;
    uint32_t q_token_base;
    uint32_t num_q_tokens;

    ScheduleEntry() = default;

    constexpr ScheduleEntry(const uint32_t kv_split_begin, const uint32_t kv_split_end,
                            const uint32_t q_token_base, const uint32_t num_q_tokens)
        : kv_split_begin(kv_split_begin), kv_split_end(kv_split_end),
          q_token_base(q_token_base), num_q_tokens(num_q_tokens) {}
};
static_assert(sizeof(ScheduleEntry) == 16, "ScheduleEntry must stay 16 bytes");

template <uint32_t kNumKVBlocksPerSplit>
struct KVSplit {
    KVSplitHeader header;
    KVBlockInfo kv_block_infos[kNumKVBlocksPerSplit];
};

// Cross-CTA coordination for the metadata kernel: split-count reservation, persistent-CTA claim
// counter (paged path), and the last-CTA gate. Counters are reset by the last CTA so the same
// workspace can be reused without a memset.
struct alignas(128) WorkspaceState {
    uint32_t num_kv_splits;
    alignas(128) uint32_t next_q_offset;
    alignas(128) uint32_t num_finished_ctas;
};

struct alignas(8) QBlockInfo {
    uint32_t kv_split_base;
    uint32_t num_kv_splits;

    QBlockInfo() = default;

    constexpr QBlockInfo(const uint32_t kv_split_base, const uint32_t num_kv_splits)
        : kv_split_base(kv_split_base), num_kv_splits(num_kv_splits) {}
};

// Scratch capacity buckets for the metadata kernel: the runtime `num_max_sparse_blocks` row
// stride is rounded UP to one of these so the device smem/scratch is template-sized.
constexpr uint32_t kNumMaxSparseBlocksBuckets[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

constexpr uint32_t get_num_sparse_blocks_bucket(const uint32_t num_max_sparse_blocks) {
    for (const uint32_t bucket : kNumMaxSparseBlocksBuckets)
        if (bucket >= num_max_sparse_blocks)
            return bucket;
    return kNumMaxSparseBlocksBuckets[10];
}

constexpr uint32_t get_kv_blocks_per_split(const uint32_t sparse_block_kv) {
    return kSplitKV / sparse_block_kv;
}

constexpr uint32_t get_num_sparse_kv_split_bytes(const uint32_t sparse_block_kv) {
    return sizeof(KVSplitHeader) + get_kv_blocks_per_split(sparse_block_kv) * sizeof(KVBlockInfo);
}

} // namespace sparse_mqa_logits
} // namespace deep_gemm
