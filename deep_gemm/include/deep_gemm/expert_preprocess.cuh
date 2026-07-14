#pragma once

#include <cstdint>
#include "dispatch_layout.cuh"
#include "utils_rtc.cuh"

namespace deep_gemm {

// ===========================================================================
// Expert preprocessing is split into two stages.
//
// Prepare is BLOCK_M-independent: wait for this generation, read remote counts
// once, cache pair_counts, and produce the GroupedMasked counts plus total/max M.
// Finalize is BLOCK_M-dependent: consume the cached counts to build the compact
// expert layout and the per-(M-block, rank) copy metadata.
// ===========================================================================
__device__ void dispatch_expert_prepare_device(
    const int64_t* __restrict__ sym_buf_addrs,
    uint32_t rank_idx,
    uint32_t num_ranks,
    uint32_t num_local_experts,
    uint32_t num_total_experts,
    uint32_t max_tokens_per_expert,
    uint32_t hidden_dim,
    uint32_t local_expert_start,
    uint32_t generation,
    uint32_t* __restrict__ pair_counts,
    uint32_t* __restrict__ masked_m,
    uint32_t* __restrict__ out_shape_m,
    uint32_t* __restrict__ out_expected_m,
    uint64_t* __restrict__ dbg_cyc,
    uint32_t num_threads) {

    long long dbg_t0 = clock64();
    __shared__ SymBuffer smem_sym;
    if (threadIdx.x == 0) {
        smem_sym.rank_idx = rank_idx;
        smem_sym.num_ranks = num_ranks;
        smem_sym.base = sym_buf_addrs[rank_idx];
    }
    if (threadIdx.x < kNumMaxRanks) {
        smem_sym.offsets[threadIdx.x] = (threadIdx.x < num_ranks)
            ? (sym_buf_addrs[threadIdx.x] - sym_buf_addrs[rank_idx])
            : 0;
    }
    __syncthreads();

    const SymBuffer& sym_buffer = smem_sym;
    DispatchBufferLayout buf_layout(num_total_experts, num_total_experts,
                                     max_tokens_per_expert, hidden_dim);
    void* data_base = buf_layout.parity_base(
        sym_buffer.get_base_ptr<void*>(), generation & 1u);

    if (generation > 0 && threadIdx.x < num_ranks) {
        volatile uint32_t* arrival_slots = reinterpret_cast<volatile uint32_t*>(
            static_cast<uint8_t*>(sym_buffer.get_base_ptr<void*>()) +
            buf_layout.ready_flag_offset());
        while (arrival_slots[threadIdx.x] < generation) { }
    }
    if (generation > 0) __syncthreads();
    long long dbg_t_arrived = clock64();

    const uint32_t total_pairs = num_local_experts * num_ranks;
    for (uint32_t tid = threadIdx.x; tid < total_pairs; tid += num_threads) {
        uint32_t src_rank = tid / num_local_experts;
        uint32_t local_expert = tid % num_local_experts;
        uint32_t global_expert = local_expert_start + local_expert;
        uint32_t* local_counts = buf_layout.expert_token_counts_ptr(data_base);
        uint32_t* remote_counts = sym_buffer.map(local_counts, src_rank);
        pair_counts[tid] = unpack_generation_count(
            __ldg(remote_counts + global_expert), generation, max_tokens_per_expert);
    }
    __syncthreads();
    long long dbg_t_counts = clock64();

    if (threadIdx.x < num_local_experts) {
        uint32_t e = threadIdx.x;
        uint32_t total = 0;
        for (uint32_t r = 0; r < num_ranks; ++r)
            total += pair_counts[r * num_local_experts + e];
        masked_m[e] = total;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t shape_m = 0, expected_m = 0;
        for (uint32_t e = 0; e < num_local_experts; ++e) {
            uint32_t count = masked_m[e];
            shape_m += count;
            expected_m = max(expected_m, count);
        }
        *out_shape_m = shape_m;
        *out_expected_m = expected_m;
        if (dbg_cyc != nullptr) {
            dbg_cyc[0] = static_cast<uint64_t>(dbg_t_arrived - dbg_t0);
            dbg_cyc[1] = static_cast<uint64_t>(dbg_t_counts - dbg_t_arrived);
        }
    }
}

__global__ void dispatch_expert_prepare_kernel(
    const int64_t* __restrict__ sym_buf_addrs,
    uint32_t rank_idx,
    uint32_t num_ranks,
    uint32_t num_local_experts,
    uint32_t num_total_experts,
    uint32_t max_tokens_per_expert,
    uint32_t hidden_dim,
    uint32_t local_expert_start,
    uint32_t generation,
    uint32_t* __restrict__ pair_counts,
    uint32_t* __restrict__ masked_m,
    uint32_t* __restrict__ out_shape_m,
    uint32_t* __restrict__ out_expected_m,
    uint64_t* __restrict__ dbg_cyc) {

    dispatch_expert_prepare_device(
        sym_buf_addrs, rank_idx, num_ranks,
        num_local_experts, num_total_experts, max_tokens_per_expert, hidden_dim,
        local_expert_start, generation,
        pair_counts, masked_m, out_shape_m, out_expected_m, dbg_cyc,
        blockDim.x);
}

void launch_dispatch_expert_prepare(
    const int64_t* sym_buf_addrs,
    uint32_t rank_idx,
    uint32_t num_ranks,
    uint32_t num_local_experts,
    uint32_t num_total_experts,
    uint32_t max_tokens_per_expert,
    uint32_t hidden_dim,
    uint32_t local_expert_start,
    uint32_t generation,
    uint32_t* pair_counts,
    uint32_t* masked_m,
    uint32_t* out_shape_m,
    uint32_t* out_expected_m,
    uint64_t* dbg_cyc,
    cudaStream_t stream) {

    uint32_t total_pairs = num_local_experts * num_ranks;
    uint32_t threads = max(32u, min(max(total_pairs, num_local_experts), 256u));
    dispatch_expert_prepare_kernel<<<1, threads, 0, stream>>>(
        sym_buf_addrs, rank_idx, num_ranks,
        num_local_experts, num_total_experts, max_tokens_per_expert, hidden_dim,
        local_expert_start, generation,
        pair_counts, masked_m, out_shape_m, out_expected_m, dbg_cyc);
}

template <uint32_t BLOCK_M>
__device__ void dispatch_expert_finalize_device(
    const int64_t* __restrict__ sym_buf_addrs,
    uint32_t rank_idx,
    uint32_t num_ranks,
    uint32_t num_local_experts,
    uint32_t num_total_experts,
    uint32_t max_tokens_per_expert,
    uint32_t hidden_dim,
    uint32_t local_expert_start,
    uint32_t generation,
    const uint32_t* __restrict__ pair_counts,
    int* __restrict__ grouped_layout,
    uint64_t* __restrict__ rank_addr_a,
    uint64_t* __restrict__ rank_addr_sfa,
    uint32_t* __restrict__ rank_split_m,
    uint32_t* __restrict__ rank_counts,
    uint32_t* __restrict__ masked_m,
    uint32_t* __restrict__ out_total_m_blocks,
    uint32_t* __restrict__ out_shape_m,
    uint64_t* __restrict__ dbg_cyc,
    uint32_t* smem_workspace,
    uint32_t num_threads,
    const int64_t* __restrict__ staging_addrs = nullptr) {  // DG_SFA_PUSH: separate staging buffer peer bases

    long long dbg_t0 = clock64();
    __shared__ SymBuffer smem_sym;
    if (threadIdx.x == 0) {
        smem_sym.rank_idx = rank_idx;
        smem_sym.num_ranks = num_ranks;
        smem_sym.base = sym_buf_addrs[rank_idx];
    }
    if (threadIdx.x < kNumMaxRanks) {
        smem_sym.offsets[threadIdx.x] = (threadIdx.x < num_ranks)
            ? (sym_buf_addrs[threadIdx.x] - sym_buf_addrs[rank_idx])
            : 0;
    }
    __syncthreads();

    const SymBuffer& sym_buffer = smem_sym;
    DispatchBufferLayout buf_layout(num_total_experts, num_total_experts,
                                     max_tokens_per_expert, hidden_dim);

    void* data_base = buf_layout.parity_base(sym_buffer.get_base_ptr<void*>(), generation & 1u);

    uint32_t* expert_m_blocks_arr = smem_workspace;
    uint32_t* expert_cumsum_blocks = expert_m_blocks_arr + num_local_experts;
    uint32_t* expert_cumsum_m = expert_cumsum_blocks + num_local_experts;

    if (threadIdx.x < num_local_experts) {
        uint32_t e = threadIdx.x;
        uint32_t total = masked_m[e];
        expert_m_blocks_arr[e] = (total + BLOCK_M - 1) / BLOCK_M;
    }
    __syncthreads();

    // Serial expert-level prefix sum
    if (threadIdx.x == 0) {
        uint32_t cumsum_b = 0, cumsum_m = 0;
        for (uint32_t e = 0; e < num_local_experts; ++e) {
            expert_cumsum_blocks[e] = cumsum_b;
            expert_cumsum_m[e] = cumsum_m;
            cumsum_b += expert_m_blocks_arr[e];
            cumsum_m += masked_m[e];
        }
        grouped_layout[0] = cumsum_b;
        if (out_total_m_blocks) *out_total_m_blocks = cumsum_b;
        if (out_shape_m) *out_shape_m = cumsum_m;
    }
    __syncthreads();

    // Phase 5: Greedy packing — fill M-blocks, rank data may span blocks
    if (threadIdx.x < num_local_experts) {
        uint32_t e = threadIdx.x;
        uint32_t total_count = masked_m[e];
        uint32_t num_mblocks = expert_m_blocks_arr[e];
        uint32_t base_block = expert_cumsum_blocks[e];
        uint32_t base_m = expert_cumsum_m[e];
        uint32_t global_expert = local_expert_start + e;
        uint32_t k_half = hidden_dim / 2;

        // Masked scheduler metadata. The second half maps an expert-local
        // M-block back to the compact copy-ready flag index.
        masked_m[e] = total_count;
        masked_m[num_local_experts + e] = base_block;

        // Per-rank remaining tokens and offset tracking
        uint32_t remaining[kNumMaxRanks];
        uint32_t rank_offset[kNumMaxRanks];  // tokens already placed
        for (uint32_t r = 0; r < num_ranks; ++r) {
            remaining[r] = pair_counts[r * num_local_experts + e];
            rank_offset[r] = 0;
        }

        for (uint32_t mb = 0; mb < num_mblocks; ++mb) {
            uint32_t block_idx = base_block + mb;

            // [opt#1] Removed dead first pass: it computed tokens_in_block /
            // temp_remaining that were never used (entry.y uses total_count, and
            // tokens_placed_total was discarded). Saves ~num_ranks iters/block.

            // Write grouped_layout entry
            // entry.y = total tokens across ALL M-blocks for this expert (not per-block),
            // so the scheduler computes correct num_m_blocks = ceil(total/BLOCK_M).
            uint4 entry;
            entry.x = e;
            entry.y = total_count;
            entry.z = base_block;
            entry.w = base_m;
            reinterpret_cast<uint4*>(grouped_layout + 4)[block_idx] = entry;

            // Second pass: write per-rank split info and update state
            uint32_t smem_row = 0;
            for (uint32_t r = 0; r < num_ranks; ++r) {
                uint32_t idx = block_idx * num_ranks + r;

                if (remaining[r] == 0 || smem_row >= BLOCK_M) {
                    rank_addr_a[idx] = 0;
                    rank_addr_sfa[idx] = 0;
                    rank_split_m[idx] = 0;
                    rank_counts[idx] = 0;
                    continue;
                }

                uint32_t available = BLOCK_M - smem_row;
                uint32_t take = min(remaining[r], available);

                uint8_t* local_fp4 = buf_layout.fp4_data_ptr(data_base, global_expert);
                uint8_t* remote_fp4 = sym_buffer.map(local_fp4, r);
                uint16_t* local_scale = buf_layout.scale_ptr(data_base, global_expert);
                uint16_t* remote_scale = sym_buffer.map(local_scale, r);

                // Offset by tokens already placed from this rank
                rank_addr_a[idx] = reinterpret_cast<uint64_t>(
                    remote_fp4 + (uint64_t)rank_offset[r] * k_half);
#if defined(DG_SFA_PUSH)
                // PUSH: SFA is not pulled from the source. Quant pushed each token's
                // ksb scales (row-major) into THIS rank's SEPARATE staging buffer band
                // [local_expert e][src_rank r][slot][ksb]. rank_addr_sfa points at the
                // LOCAL staging slice start for (e, r, rank_offset[r]); the reshape
                // reads it row-major (see copy_mblock_sfa DG_SFA_PUSH path).
                {
                    uint32_t ksb = ((hidden_dim / 2) + 31u) / 32u;
                    uint64_t staging_local_base =
                        (staging_addrs != nullptr) ? (uint64_t)staging_addrs[rank_idx] : 0ull;
                    uint16_t* staging = reinterpret_cast<uint16_t*>(
                        staging_local_base
                        + (uint64_t)(generation & 1u) * buf_layout.staging_parity_bytes());
                    uint64_t sidx = ((uint64_t)(e * num_ranks + r) * max_tokens_per_expert
                                     + rank_offset[r]) * ksb;
                    rank_addr_sfa[idx] = reinterpret_cast<uint64_t>(staging + sidx);
                }
#elif defined(DG_SFA_ROWMAJOR_SRC)
                // Row-major source [max_tokens, ksb]: token offset strides by ksb
                // (each token's ksb scales are contiguous). ksb = ceil(hidden/64)
                // = K_SCALE_BLOCKS, matching mxfp4_quant / the GEMM's SFK.
                {
                    uint32_t ksb = ((hidden_dim / 2) + 31u) / 32u;
                    rank_addr_sfa[idx] = reinterpret_cast<uint64_t>(
                        remote_scale + (uint64_t)rank_offset[r] * ksb);
                }
#else
                rank_addr_sfa[idx] = reinterpret_cast<uint64_t>(
                    remote_scale + rank_offset[r]);
#endif
                rank_split_m[idx] = smem_row;
                rank_counts[idx] = take;

                smem_row += take;
                remaining[r] -= take;
                rank_offset[r] += take;
            }
        }
    }

    if (dbg_cyc != nullptr && threadIdx.x == 0) {
        dbg_cyc[2] = static_cast<uint64_t>(clock64() - dbg_t0);
    }
}

template <uint32_t BLOCK_M>
__global__ void dispatch_expert_finalize_kernel(
    const int64_t* __restrict__ sym_buf_addrs,
    uint32_t rank_idx,
    uint32_t num_ranks,
    uint32_t num_local_experts,
    uint32_t num_total_experts,
    uint32_t max_tokens_per_expert,
    uint32_t hidden_dim,
    uint32_t local_expert_start,
    uint32_t generation,
    const uint32_t* __restrict__ pair_counts,
    int* __restrict__ grouped_layout,
    uint64_t* __restrict__ rank_addr_a,
    uint64_t* __restrict__ rank_addr_sfa,
    uint32_t* __restrict__ rank_split_m,
    uint32_t* __restrict__ rank_counts,
    uint32_t* __restrict__ masked_m,
    uint32_t* __restrict__ out_total_m_blocks,
    uint32_t* __restrict__ out_shape_m,
    uint64_t* __restrict__ dbg_cyc,
    const int64_t* __restrict__ staging_addrs = nullptr) {

    extern __shared__ uint32_t smem[];
    dispatch_expert_finalize_device<BLOCK_M>(
        sym_buf_addrs, rank_idx, num_ranks,
        num_local_experts, num_total_experts, max_tokens_per_expert, hidden_dim,
        local_expert_start, generation,
        pair_counts, grouped_layout, rank_addr_a, rank_addr_sfa,
        rank_split_m, rank_counts, masked_m,
        out_total_m_blocks, out_shape_m,
        dbg_cyc, smem, blockDim.x, staging_addrs);
}

template <uint32_t BLOCK_M>
void launch_dispatch_expert_finalize(
    const int64_t* sym_buf_addrs,
    uint32_t rank_idx,
    uint32_t num_ranks,
    uint32_t num_local_experts,
    uint32_t num_total_experts,
    uint32_t max_tokens_per_expert,
    uint32_t hidden_dim,
    uint32_t local_expert_start,
    uint32_t generation,
    const uint32_t* pair_counts,
    int* grouped_layout,
    uint64_t* rank_addr_a,
    uint64_t* rank_addr_sfa,
    uint32_t* rank_split_m,
    uint32_t* rank_counts,
    uint32_t* masked_m,
    uint32_t* out_total_m_blocks,
    uint32_t* out_shape_m,
    uint64_t* dbg_cyc,
    cudaStream_t stream,
    const int64_t* __restrict__ staging_addrs = nullptr) {

    uint32_t num_scan_warps = (num_local_experts + 31) / 32;
    uint32_t min_threads = max(num_scan_warps * 32, min(num_local_experts, 256u));
    uint32_t smem_size = 3 * num_local_experts * sizeof(uint32_t);

    dispatch_expert_finalize_kernel<BLOCK_M><<<1, min_threads, smem_size, stream>>>(
        sym_buf_addrs, rank_idx, num_ranks,
        num_local_experts, num_total_experts, max_tokens_per_expert, hidden_dim,
        local_expert_start, generation,
        pair_counts, grouped_layout, rank_addr_a, rank_addr_sfa,
        rank_split_m, rank_counts, masked_m,
        out_total_m_blocks, out_shape_m, dbg_cyc, staging_addrs);
}


// ===========================================================================
// Merged prepare+finalize (single launch). Only valid when BLOCK_M is already
// known on host (fixed-config / masked path) -- i.e. no host readback of
// expected_m is needed between the two stages. Runs both device stages in one
// CTA with a __syncthreads() barrier: prepare writes pair_counts/masked_m to
// global, the barrier makes those writes visible within the block, finalize
// consumes them. Saves one kernel launch + the kernel-boundary gap vs the split
// prepare();finalize() sequence. Arrival barrier logic is unchanged.
// ===========================================================================
template <uint32_t BLOCK_M>
__global__ void dispatch_expert_preprocess_merged_kernel(
    const int64_t* __restrict__ sym_buf_addrs,
    uint32_t rank_idx,
    uint32_t num_ranks,
    uint32_t num_local_experts,
    uint32_t num_total_experts,
    uint32_t max_tokens_per_expert,
    uint32_t hidden_dim,
    uint32_t local_expert_start,
    uint32_t generation,
    uint32_t* __restrict__ pair_counts,
    int* __restrict__ grouped_layout,
    uint64_t* __restrict__ rank_addr_a,
    uint64_t* __restrict__ rank_addr_sfa,
    uint32_t* __restrict__ rank_split_m,
    uint32_t* __restrict__ rank_counts,
    uint32_t* __restrict__ masked_m,
    uint32_t* __restrict__ out_total_m_blocks,
    uint32_t* __restrict__ out_shape_m,
    uint32_t* __restrict__ out_expected_m,
    uint64_t* __restrict__ dbg_cyc,
    const int64_t* __restrict__ staging_addrs = nullptr) {

    extern __shared__ uint32_t smem[];

    dispatch_expert_prepare_device(
        sym_buf_addrs, rank_idx, num_ranks,
        num_local_experts, num_total_experts, max_tokens_per_expert, hidden_dim,
        local_expert_start, generation,
        pair_counts, masked_m, out_shape_m, out_expected_m, dbg_cyc,
        blockDim.x);

    __syncthreads();  // publish prepare's pair_counts/masked_m to the block

    dispatch_expert_finalize_device<BLOCK_M>(
        sym_buf_addrs, rank_idx, num_ranks,
        num_local_experts, num_total_experts, max_tokens_per_expert, hidden_dim,
        local_expert_start, generation,
        pair_counts, grouped_layout, rank_addr_a, rank_addr_sfa,
        rank_split_m, rank_counts, masked_m,
        out_total_m_blocks, out_shape_m,
        dbg_cyc, smem, blockDim.x, staging_addrs);
}

template <uint32_t BLOCK_M>
void launch_dispatch_expert_preprocess_merged(
    const int64_t* sym_buf_addrs,
    uint32_t rank_idx,
    uint32_t num_ranks,
    uint32_t num_local_experts,
    uint32_t num_total_experts,
    uint32_t max_tokens_per_expert,
    uint32_t hidden_dim,
    uint32_t local_expert_start,
    uint32_t generation,
    uint32_t* pair_counts,
    int* grouped_layout,
    uint64_t* rank_addr_a,
    uint64_t* rank_addr_sfa,
    uint32_t* rank_split_m,
    uint32_t* rank_counts,
    uint32_t* masked_m,
    uint32_t* out_total_m_blocks,
    uint32_t* out_shape_m,
    uint32_t* out_expected_m,
    uint64_t* dbg_cyc,
    cudaStream_t stream,
    const int64_t* __restrict__ staging_addrs = nullptr) {

    uint32_t total_pairs = num_local_experts * num_ranks;
    uint32_t prep_threads = max(32u, min(max(total_pairs, num_local_experts), 256u));
    uint32_t num_scan_warps = (num_local_experts + 31) / 32;
    uint32_t fin_threads = max(num_scan_warps * 32, min(num_local_experts, 256u));
    uint32_t threads = max(prep_threads, fin_threads);
    uint32_t smem_size = 3 * num_local_experts * sizeof(uint32_t);

    dispatch_expert_preprocess_merged_kernel<BLOCK_M><<<1, threads, smem_size, stream>>>(
        sym_buf_addrs, rank_idx, num_ranks,
        num_local_experts, num_total_experts, max_tokens_per_expert, hidden_dim,
        local_expert_start, generation,
        pair_counts, grouped_layout, rank_addr_a, rank_addr_sfa,
        rank_split_m, rank_counts, masked_m,
        out_total_m_blocks, out_shape_m, out_expected_m, dbg_cyc, staging_addrs);
}

}  // namespace deep_gemm
