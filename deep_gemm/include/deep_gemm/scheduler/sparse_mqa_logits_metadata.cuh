#pragma once

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/impls/sparse_mqa_logits_layout.cuh>

// Sparse MQA logits metadata kernel, ported from open-source DeepGEMM 26/09
// `scheduler/sm100_sparse_mqa_logits_metadata.cuh`. One kernel serves both KV layouts via
// `kIsPaged`: contiguous (per-Q-block claim, KS/KE windows, unaligned-ks rescale) and paged
// (per-token claim, request-boundary Q pairing via `indices`, context_lens windows, block_table
// physical resolution, bounded kNumKVSplitsPerEntry entries + wave balancing).
//
// Per Q block (kBlockQ = 2 paired tokens), one CTA:
//   1. merge-path partitions the two tokens' sorted sparse index rows (read straight from global;
//      with unaligned ks, block ids are rescaled into logical token offsets on the fly) and dedups
//      the union, packing each merged KV block as two 16-bit (slot index | present bit) fields,
//   2. compacts via a CTA exclusive prefix sum and atomically reserves the block's KV-split range,
//   3. writes KVSplitHeader + KVBlockInfo (physical block id + per-token slot offsets), padding the
//      last partial split with invalid slots,
// and the last CTA (release-incremented counter + acquire load) builds the per-SM schedule table:
// KV splits are divided evenly across SMs, cut at Q-block boundaries, padded to whole waves.
//
// Note: logical index values are read straight from the global source rows (no shared-memory
// staging); only the packed merge output crosses threads, via dynamic shared memory around
// `__syncthreads()`.
//
// Differences vs upstream: no CDP/PDL (plain launch), shared-memory `atomicMax` replaced by a
// plain reduce (and paged wave balancing by warp histograms without shared atomics), PPU
// atomics/acquire via `ppu.*` asm, and logical index values read straight from global (no smem
// staging).

namespace deep_gemm::sparse_mqa_logits {

__device__ __forceinline__ uint32_t atomic_add_relaxed_global_u32(uint32_t* addr, uint32_t value) {
    uint32_t ret;
    asm volatile("ppu.atom.add.gpu.global.u32 %0, [%1], %2;" : "=r"(ret) : "l"(addr), "r"(value));
    return ret;
}

__device__ __forceinline__ uint32_t ld_acquire_global_u32(const uint32_t* addr) {
    uint32_t ret;
    asm volatile("ppu.ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(ret) : "l"(addr));
    return ret;
}

// Logical KV block id of `slot_idx` in token `global_q_idx`'s sparse index row. With unaligned ks,
// raw block ids are rescaled into logical token offsets (`block * sparse_block_kv + ks % sparse_block_kv`).
template <bool kUseUnalignedKs, uint32_t SPARSE_BLOCK_KV>
__device__ __forceinline__ uint32_t get_sparse_logical_block_value(
    const uint32_t* sparse_kv_block_indices, const uint32_t* cu_seq_len_k_start,
    const uint32_t global_q_idx, const uint32_t num_max_sparse_blocks, const uint32_t slot_idx) {
    const uint32_t raw =
        sparse_kv_block_indices[static_cast<uint64_t>(global_q_idx) * num_max_sparse_blocks + slot_idx];
    if constexpr (kUseUnalignedKs)
        return raw * SPARSE_BLOCK_KV + cu_seq_len_k_start[global_q_idx] % SPARSE_BLOCK_KV;
    return raw;
}

// CTA-wide exclusive prefix sum of per-thread `value` (kNumThreads power of two, <= 1024).
// Returns this thread's exclusive offset; `block_total` receives the sum of all threads.
template <uint32_t kNumThreads>
__device__ __forceinline__ uint32_t cta_exclusive_sum_u32(const uint32_t value, uint32_t* warp_sums,
                                                          uint32_t& block_total) {
    constexpr uint32_t kNumWarps = kNumThreads / 32;
    const uint32_t lane_idx = threadIdx.x % 32, warp_idx = threadIdx.x / 32;
    uint32_t inclusive = value;
    #pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        const uint32_t neighbor = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane_idx >= offset)
            inclusive += neighbor;
    }
    if (lane_idx == 31)
        warp_sums[warp_idx] = inclusive;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t warp = 0; warp < kNumWarps; ++warp) {
            const uint32_t warp_total = warp_sums[warp];
            warp_sums[warp] = sum;
            sum += warp_total;
        }
        warp_sums[kNumWarps] = sum;
    }
    __syncthreads();
    block_total = warp_sums[kNumWarps];
    // Add the warp-exclusive offset to the intra-warp prefix.
    return warp_sums[warp_idx] + inclusive - value;
}

// Divide `total` KV splits evenly across SMs, then cut each SM range at Q-block boundaries.
// `smem_max_entries[kNumSlots]` is per-thread scratch for the wave-count reduction.
template <uint32_t kNumKVBlocksPerSplit, uint32_t kNumSlots>
__device__ __forceinline__ uint32_t build_contiguous_schedule(
    ScheduleEntry* schedule_entries, const KVSplit<kNumKVBlocksPerSplit>* kv_splits,
    const QBlockInfo* q_block_infos, const uint32_t num_q_tokens, const uint32_t total_kv_splits,
    uint32_t* smem_num_waves, uint32_t* smem_max_entries) {
    constexpr uint32_t kNumThreads = 256;
    const uint32_t tid = threadIdx.x;

    if (tid == 0)
        *smem_num_waves = 1;
    __syncthreads();

    uint32_t num_slot_entries = 0;
    if (tid < kNumSlots) {
        uint32_t kv_split_idx = static_cast<uint32_t>(
            (static_cast<uint64_t>(total_kv_splits) * tid + kNumSlots - 1) / kNumSlots);
        const uint32_t kv_split_end = static_cast<uint32_t>(
            (static_cast<uint64_t>(total_kv_splits) * (tid + 1) + kNumSlots - 1) / kNumSlots);
        while (kv_split_idx < kv_split_end) {
            const uint32_t q_token_base = kv_splits[kv_split_idx].header.q_token_base;
            const QBlockInfo q_block_info = q_block_infos[q_token_base];
            const uint32_t entry_kv_split_end = min(kv_split_end, q_block_info.kv_split_base + q_block_info.num_kv_splits);
            schedule_entries[num_slot_entries * kNumSlots + tid] = ScheduleEntry(
                kv_split_idx, entry_kv_split_end, q_token_base, min(kBlockQ, num_q_tokens - q_token_base));
            kv_split_idx = entry_kv_split_end;
            ++num_slot_entries;
        }
        // Reduce per-thread maxima to obtain the wave count.
        smem_max_entries[tid] = num_slot_entries;
    }
    __syncthreads();
    if (tid == 0) {
        uint32_t max_entries = 1;
        for (uint32_t slot_idx = 0; slot_idx < kNumSlots; ++slot_idx)
            max_entries = max(max_entries, smem_max_entries[slot_idx]);
        *smem_num_waves = max_entries;
    }
    __syncthreads();

    const uint32_t num_waves = *smem_num_waves;
    if (tid < kNumSlots) {
        for (uint32_t wave_idx = num_slot_entries; wave_idx < num_waves; ++wave_idx)
            schedule_entries[wave_idx * kNumSlots + tid] = ScheduleEntry(0, 0, 0, 0);
    }
    return num_waves;
}

// Paged schedule: split each Q block's KV splits into bounded entries (<= kNumKVSplitsPerEntry),
// then sort each wave by split count and rotate heavy entries across slots (upstream
// `build_paged_schedule` + `balance_wave_entries`). Warp matching builds a histogram
// without shared atomics; a parallel stable counting sort preserves entry order within each bin.
// Schedule table is wave-major: entry ordinal e lives at schedule_entries[e] (wave e/kNumSlots,
// slot e%kNumSlots), and the main kernel reads [wave * kNumSlots + slot].
template <uint32_t kNumSlots, uint32_t kNumKVSplitsPerEntry, uint32_t kNumThreads>
__device__ __forceinline__ uint32_t build_paged_schedule(
    ScheduleEntry* schedule_entries, const QBlockInfo* q_block_infos,
    const uint32_t num_q_tokens, const uint32_t* indices,
    uint32_t* warp_sums, uint32_t* smem_histogram) {
    const uint32_t tid = threadIdx.x;

    uint32_t num_entries = 0;
    for (uint32_t q_token_begin = 0; q_token_begin < num_q_tokens; q_token_begin += kNumThreads) {
        const uint32_t q_token_idx = q_token_begin + tid;
        const QBlockInfo q_block_info = q_token_idx < num_q_tokens ? q_block_infos[q_token_idx] : QBlockInfo(0, 0);
        const uint32_t num_kv_splits = q_block_info.num_kv_splits;
        const uint32_t num_q_entries = ceil_div(num_kv_splits, kNumKVSplitsPerEntry);
        uint32_t num_batch_entries;
        const uint32_t entry_begin = num_entries + cta_exclusive_sum_u32<kNumThreads>(
            num_q_entries, warp_sums, num_batch_entries);

        const uint32_t kv_splits_per_entry = num_q_entries == 0 ? 0 : num_kv_splits / num_q_entries;
        const uint32_t num_larger_entries = num_q_entries == 0 ? 0 : num_kv_splits % num_q_entries;
        const uint32_t num_q_block_tokens =
            q_token_idx + 1 < num_q_tokens and indices[q_token_idx + 1] == indices[q_token_idx] ? kBlockQ : 1;
        uint32_t kv_split_begin = q_block_info.kv_split_base;
        for (uint32_t q_entry_idx = 0; q_entry_idx < num_q_entries; ++q_entry_idx) {
            const uint32_t kv_split_end = kv_split_begin + kv_splits_per_entry + (q_entry_idx < num_larger_entries);
            schedule_entries[entry_begin + q_entry_idx] =
                ScheduleEntry(kv_split_begin, kv_split_end, q_token_idx, num_q_block_tokens);
            kv_split_begin = kv_split_end;
        }
        num_entries += num_batch_entries;
        __syncthreads();
    }

    const uint32_t num_waves = max(1u, ceil_div(num_entries, kNumSlots));
    for (uint32_t entry_idx = num_entries + tid; entry_idx < num_waves * kNumSlots; entry_idx += kNumThreads)
        schedule_entries[entry_idx] = ScheduleEntry(0, 0, 0, 0);
    __syncthreads();

    // Each warp stably sorts one wave using a private histogram.
    constexpr uint32_t kNumWarps = kNumThreads / 32;
    constexpr uint32_t kNumBins = kNumKVSplitsPerEntry + 1;
    constexpr uint32_t kEntriesPerLane = (kNumSlots + 31) / 32;
    static_assert(kNumBins <= 32, "Wave histogram must fit in one warp");
    const uint32_t lane = tid % 32, warp = tid / 32;
    uint32_t* histogram = smem_histogram + warp * kNumBins;
    for (uint32_t wave_idx = warp; wave_idx < num_waves; wave_idx += kNumWarps) {
        ScheduleEntry* wave_entries = schedule_entries + wave_idx * kNumSlots;
        ScheduleEntry entries[kEntriesPerLane];
        uint32_t ranks[kEntriesPerLane];
        if (lane < kNumBins)
            histogram[lane] = 0;
        __syncwarp();
        #pragma unroll
        for (uint32_t i = 0; i < kEntriesPerLane; ++i) {
            const uint32_t slot = i * 32 + lane;
            entries[i] = slot < kNumSlots ? wave_entries[slot] : ScheduleEntry(0, 0, 0, 0);
            const uint32_t count = slot < kNumSlots ? entries[i].kv_split_end - entries[i].kv_split_begin : kNumBins;
            const uint32_t peers = __match_any_sync(0xffffffffu, count);
            const uint32_t preceding = peers & __ppu_read_lanemask_lt();
            const uint32_t base = slot < kNumSlots ? histogram[count] : 0;
            __syncwarp();
            if (slot < kNumSlots && preceding == 0)
                histogram[count] = base + __popc(peers);
            ranks[i] = base + __popc(preceding);
            __syncwarp();
        }
        const uint32_t total = lane < kNumBins ? histogram[lane] : 0;
        uint32_t inclusive = total;
        #pragma unroll
        for (uint32_t offset = 1; offset < kNumBins; offset <<= 1) {
            const uint32_t neighbor = __shfl_up_sync(0xffffffffu, inclusive, offset);
            if (lane >= offset)
                inclusive += neighbor;
        }
        if (lane < kNumBins)
            histogram[lane] = inclusive - total;
        __syncwarp();
        #pragma unroll
        for (uint32_t i = 0; i < kEntriesPerLane; ++i) {
            if (i * 32 + lane < kNumSlots) {
                const uint32_t count = entries[i].kv_split_end - entries[i].kv_split_begin;
                const uint32_t rank = histogram[count] + ranks[i];
                const uint32_t dst = num_waves == 2 && wave_idx == 1 ? kNumSlots - 1 - rank :
                    (rank + kNumSlots - wave_idx * kNumSlots / num_waves) % kNumSlots;
                wave_entries[dst] = entries[i];
            }
        }
        __syncwarp();
    }
    __syncthreads();
    return num_waves;
}

template <uint32_t kNumThreads, uint32_t kNumMaxBlocksCap, uint32_t kNumSlots,
          uint32_t SPARSE_BLOCK_KV, bool kUseUnalignedKs, bool kIsPaged, uint32_t PAGE_KV>
__device__ __forceinline__ void sparse_mqa_logits_metadata_device(
    const uint32_t num_q_tokens, const uint32_t num_kv_tokens,
    const uint32_t* cu_seq_len_k_start, const uint32_t* cu_seq_len_k_end,
    const uint32_t* context_lens, const uint32_t* block_table, const uint32_t block_table_stride,
    const uint32_t* indices,
    const uint32_t* sparse_kv_block_indices, const uint32_t num_max_sparse_blocks,
    uint8_t* metadata, uint8_t* workspace) {
    constexpr uint32_t kNumKVBlocksPerSplit = kSplitKV / SPARSE_BLOCK_KV;
    constexpr uint32_t kNumKVSplitsPerEntry = 8;  // upstream host constant (bounded paged entries)
    constexpr uint32_t kNumMergedBlocksCap = kBlockQ * kNumMaxBlocksCap;
    constexpr uint32_t kNumKVBlocksPerThread = (kNumMergedBlocksCap + kNumThreads - 1) / kNumThreads;
    DG_STATIC_ASSERT(not kIsPaged or PAGE_KV % SPARSE_BLOCK_KV == 0, "Invalid page shape");
    DG_STATIC_ASSERT(not kIsPaged or not kUseUnalignedKs, "Paged sparse MQA does not use ks");

    const uint32_t tid = threadIdx.x;

    const auto workspace_state = reinterpret_cast<WorkspaceState*>(workspace);
    const auto q_block_infos = reinterpret_cast<QBlockInfo*>(workspace + sizeof(WorkspaceState));
    const auto kv_splits = reinterpret_cast<KVSplit<kNumKVBlocksPerSplit>*>(metadata + sizeof(MetadataHeader));
    // Shared merge output; reduction scratch is statically allocated.
    extern __shared__ uint32_t smem_dynamic[];
    uint32_t* packed_slots_by_merged_kv_block = smem_dynamic;

    __shared__ uint32_t warp_sums[kNumThreads / 32 + 1];
    __shared__ uint32_t smem_num_waves;
    __shared__ uint32_t smem_max_entries[kNumSlots];
    __shared__ uint32_t smem_histogram[(kNumKVSplitsPerEntry + 1) * (kNumThreads / 32)];
    DG_STATIC_ASSERT(kNumSlots <= kNumThreads, "Metadata kernel needs one scratch slot per schedule slot");
    __shared__ struct {
        uint32_t q_token_base;
        uint32_t num_q_tokens;
        uint32_t num_kv_blocks[kBlockQ];
        uint32_t kv_split_base;
    } q_block;

    uint32_t q_token_idx = blockIdx.x * (kIsPaged ? 1u : kBlockQ);
    while (true) {
        if (tid == 0) {
            q_block.num_q_tokens = 0;
            while (q_token_idx < num_q_tokens) {
                uint32_t num_q_block_tokens = min(kBlockQ, num_q_tokens - q_token_idx);
                if constexpr (kIsPaged) {
                    // Pair tokens within request boundaries.
                    const uint32_t request_idx = indices[q_token_idx];
                    uint32_t request_q_token_base = q_token_idx;
                    while (request_q_token_base > 0 and indices[request_q_token_base - 1] == request_idx)
                        --request_q_token_base;
                    if ((q_token_idx - request_q_token_base) % kBlockQ != 0) {
                        // Clear stale workspace records for tokens paired with their predecessor.
                        q_block_infos[q_token_idx] = QBlockInfo(0, 0);
                        q_token_idx = gridDim.x + atomic_add_relaxed_global_u32(&workspace_state->next_q_offset, 1u);
                        continue;
                    }
                    num_q_block_tokens = q_token_idx + 1 < num_q_tokens and
                                         indices[q_token_idx + 1] == request_idx ? kBlockQ : 1;
                }
                const auto get_num_kv_blocks = [&](const uint32_t q_idx) {
                    uint32_t kv_begin, kv_end;
                    if constexpr (kIsPaged) {
                        kv_begin = 0;
                        kv_end = context_lens[q_idx];
                    } else {
                        kv_begin = min(cu_seq_len_k_start[q_idx], num_kv_tokens);
                        kv_end = max(kv_begin, min(cu_seq_len_k_end[q_idx], num_kv_tokens));
                        if constexpr (not kUseUnalignedKs)
                            DG_DEVICE_ASSERT(kv_end == kv_begin or kv_begin % SPARSE_BLOCK_KV == 0);
                    }
                    return min(num_max_sparse_blocks, ceil_div(kv_end - kv_begin, SPARSE_BLOCK_KV));
                };
                q_block.q_token_base = q_token_idx;
                q_block.num_q_tokens = num_q_block_tokens;
                q_block.num_kv_blocks[0] = get_num_kv_blocks(q_token_idx);
                q_block.num_kv_blocks[1] = num_q_block_tokens == kBlockQ ? get_num_kv_blocks(q_token_idx + 1) : 0;
                break;
            }
        }
        __syncthreads();

        const uint32_t num_q_block_tokens = q_block.num_q_tokens;
        if (num_q_block_tokens == 0)
            break;
        const uint32_t q_token_base = q_block.q_token_base;
        const uint32_t num_kv_blocks_in_q0 = q_block.num_kv_blocks[0];
        const uint32_t num_kv_blocks_in_q1 = q_block.num_kv_blocks[1];

        // Partition the two sorted logical index lists.
        const uint32_t num_input_kv_blocks = num_kv_blocks_in_q0 + num_kv_blocks_in_q1;
        const uint32_t merge_begin = tid * num_input_kv_blocks / kNumThreads;
        const uint32_t merge_end = (tid + 1) * num_input_kv_blocks / kNumThreads;
        uint32_t lo = max(merge_begin, num_kv_blocks_in_q1) - num_kv_blocks_in_q1;
        uint32_t hi = min(merge_begin, num_kv_blocks_in_q0);
        while (lo < hi) {
            const uint32_t q0_slot_idx = (lo + hi) / 2;
            const uint32_t q1_slot_idx = merge_begin - q0_slot_idx;
            if (q1_slot_idx > 0 and q0_slot_idx < num_kv_blocks_in_q0 and
                get_sparse_logical_block_value<kUseUnalignedKs, SPARSE_BLOCK_KV>(
                    sparse_kv_block_indices, cu_seq_len_k_start, q_token_base + 1,
                    num_max_sparse_blocks, q1_slot_idx - 1) >=
                get_sparse_logical_block_value<kUseUnalignedKs, SPARSE_BLOCK_KV>(
                    sparse_kv_block_indices, cu_seq_len_k_start, q_token_base,
                    num_max_sparse_blocks, q0_slot_idx))
                lo = q0_slot_idx + 1;
            else
                hi = q0_slot_idx;
        }
        uint32_t q0_slot_idx = lo;
        uint32_t q1_slot_idx = merge_begin - lo;
        uint32_t num_remaining_inputs = merge_end - merge_begin;
        uint32_t num_merged_kv_blocks_in_thread = 0;

        // Drop a duplicate carried across merge partitions
        if (num_remaining_inputs > 0 and q0_slot_idx > 0 and q1_slot_idx < num_kv_blocks_in_q1 and
            get_sparse_logical_block_value<kUseUnalignedKs, SPARSE_BLOCK_KV>(
                sparse_kv_block_indices, cu_seq_len_k_start, q_token_base,
                num_max_sparse_blocks, q0_slot_idx - 1) ==
            get_sparse_logical_block_value<kUseUnalignedKs, SPARSE_BLOCK_KV>(
                sparse_kv_block_indices, cu_seq_len_k_start, q_token_base + 1,
                num_max_sparse_blocks, q1_slot_idx)) {
            ++q1_slot_idx;
            --num_remaining_inputs;
        }

        // Merge and pack each Q's sparse slot and presence bit
        constexpr uint32_t kPresentBit = 1u << (kNumSparseSlotBits - 1);
        constexpr uint32_t kSlotIndexMask = kPresentBit - 1;
        const auto pack_slot = [](const uint32_t q_slot_idx, const bool is_present) {
            return q_slot_idx | (is_present ? kPresentBit : 0u);
        };
        uint32_t packed_slots_in_thread[kNumKVBlocksPerThread];
        for (uint32_t merged_offset_in_thread = 0; merged_offset_in_thread < kNumKVBlocksPerThread; ++merged_offset_in_thread) {
            if (num_remaining_inputs == 0)
                continue;
            const uint32_t q0_logical = q0_slot_idx < num_kv_blocks_in_q0 ?
                get_sparse_logical_block_value<kUseUnalignedKs, SPARSE_BLOCK_KV>(
                    sparse_kv_block_indices, cu_seq_len_k_start, q_token_base,
                    num_max_sparse_blocks, q0_slot_idx) : ~0u;
            const uint32_t q1_logical = q1_slot_idx < num_kv_blocks_in_q1 ?
                get_sparse_logical_block_value<kUseUnalignedKs, SPARSE_BLOCK_KV>(
                    sparse_kv_block_indices, cu_seq_len_k_start, q_token_base + 1,
                    num_max_sparse_blocks, q1_slot_idx) : ~0u;
            const bool in_q0 = q0_logical <= q1_logical;
            const bool in_q1 = q1_logical <= q0_logical;
            // Carry a final duplicate into the next partition
            const bool consume_q1 = in_q1 and num_remaining_inputs > in_q0;
            packed_slots_in_thread[merged_offset_in_thread] =
                pack_slot(q0_slot_idx, in_q0) | (pack_slot(q1_slot_idx, in_q1) << kNumSparseSlotBits);
            ++num_merged_kv_blocks_in_thread;
            q0_slot_idx += in_q0;
            q1_slot_idx += consume_q1;
            num_remaining_inputs -= in_q0 + consume_q1;
        }
        DG_DEVICE_ASSERT(num_remaining_inputs == 0);

        // Compact merged blocks and reserve their KV-split range
        uint32_t num_merged_kv_blocks;
        const uint32_t merged_kv_block_base = cta_exclusive_sum_u32<kNumThreads>(
            num_merged_kv_blocks_in_thread, warp_sums, num_merged_kv_blocks);
        const uint32_t num_kv_splits_in_q_block = ceil_div(num_merged_kv_blocks, kNumKVBlocksPerSplit);
        if (tid == 0) {
            q_block.kv_split_base = num_kv_splits_in_q_block == 0 ? 0 :
                atomic_add_relaxed_global_u32(&workspace_state->num_kv_splits, num_kv_splits_in_q_block);
            q_block_infos[q_token_base] = QBlockInfo(q_block.kv_split_base, num_kv_splits_in_q_block);
        }
        #pragma unroll
        for (uint32_t merged_offset_in_thread = 0; merged_offset_in_thread < kNumKVBlocksPerThread; ++merged_offset_in_thread) {
            if (merged_offset_in_thread >= num_merged_kv_blocks_in_thread)
                continue;
            packed_slots_by_merged_kv_block[merged_kv_block_base + merged_offset_in_thread] =
                packed_slots_in_thread[merged_offset_in_thread];
        }
        __syncthreads();

        const uint32_t kv_split_base = q_block.kv_split_base;
        // Keep split writes inline to avoid captured-lambda codegen issues.
        for (uint32_t merged_kv_block_idx = tid; merged_kv_block_idx < num_merged_kv_blocks; merged_kv_block_idx += kNumThreads) {
            const uint32_t kv_split_offset = merged_kv_block_idx / kNumKVBlocksPerSplit;
            const uint32_t kv_split_idx = kv_split_base + kv_split_offset;
            const uint32_t kv_block_idx_in_split = merged_kv_block_idx % kNumKVBlocksPerSplit;
            // The split's first merged block carries the per-token slot bases
            const uint32_t packed_slot_bases =
                packed_slots_by_merged_kv_block[merged_kv_block_idx - kv_block_idx_in_split];
            const uint32_t packed_slots = packed_slots_by_merged_kv_block[merged_kv_block_idx];
            const uint32_t q0_slot_idx = packed_slots & kSlotIndexMask;
            const uint32_t q1_slot_idx = (packed_slots >> kNumSparseSlotBits) & kSlotIndexMask;
            const uint32_t q0_slot_base = packed_slot_bases & kSlotIndexMask;
            const uint32_t q1_slot_base = (packed_slot_bases >> kNumSparseSlotBits) & kSlotIndexMask;
            const bool in_q0 = (packed_slots & kPresentBit) != 0;
            const bool in_q1 = ((packed_slots >> kNumSparseSlotBits) & kPresentBit) != 0;
            const uint32_t logical_kv_block_idx = in_q0 ?
                get_sparse_logical_block_value<kUseUnalignedKs, SPARSE_BLOCK_KV>(
                    sparse_kv_block_indices, cu_seq_len_k_start, q_token_base,
                    num_max_sparse_blocks, q0_slot_idx) :
                get_sparse_logical_block_value<kUseUnalignedKs, SPARSE_BLOCK_KV>(
                    sparse_kv_block_indices, cu_seq_len_k_start, q_token_base + 1,
                    num_max_sparse_blocks, q1_slot_idx);
            uint32_t physical_kv_block_idx;
            if constexpr (kIsPaged) {
                constexpr uint32_t kNumKVBlocksPerPage = PAGE_KV / SPARSE_BLOCK_KV;
                const uint32_t logical_page_idx = logical_kv_block_idx / kNumKVBlocksPerPage;
                DG_DEVICE_ASSERT(logical_page_idx < block_table_stride);
                const uint64_t block_table_idx = static_cast<uint64_t>(q_token_base) * block_table_stride + logical_page_idx;
                physical_kv_block_idx = block_table[block_table_idx] * kNumKVBlocksPerPage +
                                        logical_kv_block_idx % kNumKVBlocksPerPage;
            } else {
                physical_kv_block_idx = kUseUnalignedKs ? logical_kv_block_idx : logical_kv_block_idx * SPARSE_BLOCK_KV;
            }
            DG_DEVICE_ASSERT(not in_q0 or q0_slot_idx - q0_slot_base < kInvalidSparseSlot);
            DG_DEVICE_ASSERT(not in_q1 or q1_slot_idx - q1_slot_base < kInvalidSparseSlot);
            const uint32_t q0_slot_offset = in_q0 ? q0_slot_idx - q0_slot_base : kInvalidSparseSlot;
            const uint32_t q1_slot_offset = in_q1 ? q1_slot_idx - q1_slot_base : kInvalidSparseSlot;
            if (kv_block_idx_in_split == 0) {
                const uint32_t num_kv_blocks_in_split = min(kNumKVBlocksPerSplit, num_merged_kv_blocks - merged_kv_block_idx);
                bool is_contiguous = false;
                if constexpr (not kIsPaged and not kUseUnalignedKs) {
                    // Mark full, consecutive splits for the whole-tile AIU copy.
                    const uint32_t last_packed_slots =
                        packed_slots_by_merged_kv_block[merged_kv_block_idx + num_kv_blocks_in_split - 1];
                    const bool last_in_q0 = (last_packed_slots & kPresentBit) != 0;
                    const uint32_t last_q_slot_idx = last_in_q0 ? last_packed_slots & kSlotIndexMask
                                                                : (last_packed_slots >> kNumSparseSlotBits) & kSlotIndexMask;
                    const uint32_t last_logical_kv_block_idx = last_in_q0 ?
                        get_sparse_logical_block_value<kUseUnalignedKs, SPARSE_BLOCK_KV>(
                            sparse_kv_block_indices, cu_seq_len_k_start, q_token_base,
                            num_max_sparse_blocks, last_q_slot_idx) :
                        get_sparse_logical_block_value<kUseUnalignedKs, SPARSE_BLOCK_KV>(
                            sparse_kv_block_indices, cu_seq_len_k_start, q_token_base + 1,
                            num_max_sparse_blocks, last_q_slot_idx);
                    is_contiguous = num_kv_blocks_in_split == kNumKVBlocksPerSplit and
                                    last_logical_kv_block_idx == logical_kv_block_idx + num_kv_blocks_in_split - 1;
                }
                kv_splits[kv_split_idx].header.q_token_base = q_token_base;
                kv_splits[kv_split_idx].header.packed_num_kv_blocks =
                    num_kv_blocks_in_split | (is_contiguous ? kContiguousFlag : 0u);
                kv_splits[kv_split_idx].header.q0_slot_base = q0_slot_base;
                kv_splits[kv_split_idx].header.q1_slot_base =
                    num_q_block_tokens == kBlockQ ? q1_slot_base : kInvalidSparseSlot;
            }
            kv_splits[kv_split_idx].kv_block_infos[kv_block_idx_in_split].physical_kv_block_idx = physical_kv_block_idx;
            kv_splits[kv_split_idx].kv_block_infos[kv_block_idx_in_split].packed_slot_offsets =
                q0_slot_offset | (q1_slot_offset << kNumSparseSlotBits);
        }

        // Pad the last split so the main-kernel copy loop stays branch-free
        for (uint32_t padded_idx = num_merged_kv_blocks + tid;
             padded_idx < num_kv_splits_in_q_block * kNumKVBlocksPerSplit; padded_idx += kNumThreads) {
            auto& pad_block = kv_splits[kv_split_base + padded_idx / kNumKVBlocksPerSplit]
                                       .kv_block_infos[padded_idx % kNumKVBlocksPerSplit];
            pad_block.physical_kv_block_idx = 0;
            pad_block.packed_slot_offsets =
                kInvalidSparseSlot | (kInvalidSparseSlot << kNumSparseSlotBits);
        }
        if (tid == 0) {
            q_token_idx = kIsPaged ? gridDim.x + atomic_add_relaxed_global_u32(&workspace_state->next_q_offset, 1u)
                                   : q_token_idx + gridDim.x * kBlockQ;
        }
        __syncthreads();
    }

    // The last CTA acquires all producer writes and builds the schedule
    if (tid == 0) {
        const bool is_last_cta = atomic_add_release_global(reinterpret_cast<int*>(&workspace_state->num_finished_ctas), 1) +
                                     1 == static_cast<int>(gridDim.x);
        if (is_last_cta)
            ld_acquire_global_u32(&workspace_state->num_finished_ctas);
        warp_sums[0] = is_last_cta ? 1u : 0u;
    }
    __syncthreads();
    if (warp_sums[0] == 0)
        return;

    const uint32_t total_kv_splits = workspace_state->num_kv_splits;
    const auto schedule_entries = reinterpret_cast<ScheduleEntry*>(kv_splits + total_kv_splits);
    uint32_t num_waves;
    if constexpr (kIsPaged) {
        num_waves = build_paged_schedule<kNumSlots, kNumKVSplitsPerEntry, kNumThreads>(
            schedule_entries, q_block_infos, num_q_tokens, indices, warp_sums, smem_histogram);
    } else {
        num_waves = build_contiguous_schedule<kNumKVBlocksPerSplit, kNumSlots>(
            schedule_entries, kv_splits, q_block_infos, num_q_tokens, total_kv_splits, &smem_num_waves,
            smem_max_entries);
    }
    if (tid == 0) {
        const auto header = reinterpret_cast<MetadataHeader*>(metadata);
        header->num_kv_splits = total_kv_splits;
        header->num_waves = num_waves;
        header->use_unaligned_ks = kUseUnalignedKs ? 1u : 0u;
        workspace_state->num_kv_splits = 0;
        workspace_state->next_q_offset = 0;
        workspace_state->num_finished_ctas = 0;
    }
}

} // namespace deep_gemm::sparse_mqa_logits
