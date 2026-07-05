#pragma once

#include <cstdint>
#include "dispatch_layout.cuh"
#include "utils_rtc.cuh"

namespace deep_gemm {

// Device function: core preprocess logic, callable from both the standalone kernel
// and the merged preprocess+GEMM wrapper kernel.
// smem_workspace must point to at least (4 * num_local_experts * num_ranks + 2 * ceil(num_local_experts/32)) uint32_t words.
template <uint32_t BLOCK_M>
__device__ void dispatch_preprocess_device(
    const int64_t* __restrict__ sym_buf_addrs,
    uint32_t rank_idx,
    uint32_t num_ranks,
    uint32_t num_local_experts,
    uint32_t num_total_experts,
    uint32_t max_tokens_per_expert,
    uint32_t hidden_dim,
    uint32_t local_expert_start,
    uint32_t generation,
    int* __restrict__ grouped_layout,
    uint64_t* __restrict__ remote_addr_a,
    uint64_t* __restrict__ remote_addr_sfa,
    uint32_t* __restrict__ out_total_m_blocks,
    uint32_t* __restrict__ out_shape_m,
    uint32_t* smem_workspace,
    uint32_t num_threads,
    uint64_t* __restrict__ profile_clocks = nullptr) {

    uint64_t _t0, _t1, _t2, _t3, _t4, _t5;
    if (threadIdx.x == 0 && profile_clocks) _t0 = clock64();

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

    // Double-buffering: this generation's data lives in buf[generation & 1].
    void* data_base = buf_layout.parity_base(sym_buffer.get_base_ptr<void*>(), generation & 1u);

    if (threadIdx.x == 0 && profile_clocks) _t1 = clock64();

    if (generation > 0 && threadIdx.x < num_ranks) {
        // Atomic-arrival barrier: each producer p pushes its generation into THIS rank's
        // local slot[p] (see arrival_push_kernel). We poll the LOCAL slot instead of
        // spinning on a remote NVLink flag read — far less P2P polling traffic + jitter.
        volatile uint32_t* arrival_slots = reinterpret_cast<volatile uint32_t*>(
            static_cast<uint8_t*>(sym_buffer.get_base_ptr<void*>()) +
            buf_layout.ready_flag_offset());
        while (arrival_slots[threadIdx.x] < generation) { }
    }
    if (generation > 0) __syncthreads();

    if (threadIdx.x == 0 && profile_clocks) _t2 = clock64();

    const uint32_t total_pairs = num_local_experts * num_ranks;

    uint32_t* pair_token_counts = smem_workspace;
    uint32_t* pair_m_blocks = smem_workspace + total_pairs;
    uint32_t* pair_cumsum_blocks = smem_workspace + 2 * total_pairs;
    uint32_t* pair_cumsum_m = smem_workspace + 3 * total_pairs;

    for (uint32_t tid = threadIdx.x; tid < total_pairs; tid += num_threads) {
        uint32_t src_rank = tid / num_local_experts;
        uint32_t local_expert = tid % num_local_experts;
        uint32_t global_expert = local_expert_start + local_expert;

        uint32_t* local_counts_ptr = buf_layout.expert_token_counts_ptr(data_base);
        uint32_t* remote_counts_ptr = sym_buffer.map(local_counts_ptr, src_rank);

        uint32_t count = __ldg(remote_counts_ptr + global_expert);
        pair_token_counts[tid] = count;
        pair_m_blocks[tid] = (count + BLOCK_M - 1) / BLOCK_M;
    }
    __syncthreads();

    if (threadIdx.x == 0 && profile_clocks) _t3 = clock64();

    // Prefix sum (expert-major order).
    // For large num_local_experts: parallel warp-shuffle scan.
    // For small sizes or insufficient threads: serial fallback.
    const uint32_t num_scan_warps = (num_local_experts + 31) / 32;
    const bool use_parallel_scan = (num_threads >= num_scan_warps * 32) && (num_local_experts >= 4);

    if (use_parallel_scan) {
        uint32_t* warp_scan_buf = smem_workspace + 4 * total_pairs;

        // Step 1: each thread reduces one expert across all ranks
        uint32_t my_exp_total_b = 0, my_exp_total_t = 0;
        if (threadIdx.x < num_local_experts) {
            for (uint32_t r = 0; r < num_ranks; ++r) {
                uint32_t idx = r * num_local_experts + threadIdx.x;
                my_exp_total_b += pair_m_blocks[idx];
                my_exp_total_t += pair_token_counts[idx];
            }
        }

        // Step 2: intra-warp inclusive prefix sum via shuffle
        uint32_t scan_b = (threadIdx.x < num_local_experts) ? my_exp_total_b : 0;
        uint32_t scan_t = (threadIdx.x < num_local_experts) ? my_exp_total_t : 0;
        if (threadIdx.x < num_scan_warps * 32) {
            uint32_t lane = threadIdx.x & 31;
            #pragma unroll
            for (uint32_t d = 1; d < 32; d <<= 1) {
                uint32_t nb = __shfl_up_sync(0xFFFFFFFF, scan_b, d);
                uint32_t nt = __shfl_up_sync(0xFFFFFFFF, scan_t, d);
                if (lane >= d) { scan_b += nb; scan_t += nt; }
            }
            if (lane == 31) {
                uint32_t wid = threadIdx.x >> 5;
                warp_scan_buf[wid * 2]     = scan_b;
                warp_scan_buf[wid * 2 + 1] = scan_t;
            }
        }
        __syncthreads();

        // Step 3: cross-warp exclusive prefix (thread 0, at most 8 iterations)
        if (threadIdx.x == 0) {
            uint32_t rb = 0, rt = 0;
            for (uint32_t w = 0; w < num_scan_warps; ++w) {
                uint32_t wb = warp_scan_buf[w * 2];
                uint32_t wt = warp_scan_buf[w * 2 + 1];
                warp_scan_buf[w * 2]     = rb;
                warp_scan_buf[w * 2 + 1] = rt;
                rb += wb;  rt += wt;
            }
            grouped_layout[0] = rb;
            if (out_total_m_blocks) *out_total_m_blocks = rb;
            if (out_shape_m) *out_shape_m = rt;
        }
        __syncthreads();

        // Step 4: expand within-expert cumulative offsets
        if (threadIdx.x < num_local_experts) {
            uint32_t wid = threadIdx.x >> 5;
            uint32_t excl_b = scan_b - my_exp_total_b + warp_scan_buf[wid * 2];
            uint32_t excl_t = scan_t - my_exp_total_t + warp_scan_buf[wid * 2 + 1];
            for (uint32_t r = 0; r < num_ranks; ++r) {
                uint32_t idx = r * num_local_experts + threadIdx.x;
                pair_cumsum_blocks[idx] = excl_b;
                pair_cumsum_m[idx] = excl_t;
                excl_b += pair_m_blocks[idx];
                excl_t += pair_token_counts[idx];
            }
        }
        __syncthreads();
    } else {
        // Serial prefix sum (fast for small num_local_experts)
        if (threadIdx.x == 0) {
            uint32_t cumsum_blocks = 0, cumsum_m = 0;
            for (uint32_t e = 0; e < num_local_experts; ++e) {
                for (uint32_t r = 0; r < num_ranks; ++r) {
                    uint32_t idx = r * num_local_experts + e;
                    pair_cumsum_blocks[idx] = cumsum_blocks;
                    pair_cumsum_m[idx] = cumsum_m;
                    cumsum_blocks += pair_m_blocks[idx];
                    cumsum_m += pair_token_counts[idx];
                }
            }
            grouped_layout[0] = cumsum_blocks;
            if (out_total_m_blocks) *out_total_m_blocks = cumsum_blocks;
            if (out_shape_m) *out_shape_m = cumsum_m;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && profile_clocks) _t4 = clock64();

    for (uint32_t tid = threadIdx.x; tid < total_pairs; tid += num_threads) {
        uint32_t src_rank = tid / num_local_experts;
        uint32_t local_expert = tid % num_local_experts;
        uint32_t global_expert = local_expert_start + local_expert;

        uint32_t count = pair_token_counts[tid];
        uint32_t num_mblocks = pair_m_blocks[tid];
        uint32_t base_block = pair_cumsum_blocks[tid];
        uint32_t base_m = pair_cumsum_m[tid];

        uint8_t* local_fp4_base = buf_layout.fp4_data_ptr(data_base, global_expert);
        uint8_t* remote_fp4_base = sym_buffer.map(local_fp4_base, src_rank);

        uint16_t* local_scale_base = buf_layout.scale_ptr(data_base, global_expert);
        uint16_t* remote_scale_base = sym_buffer.map(local_scale_base, src_rank);

        for (uint32_t mb = 0; mb < num_mblocks; ++mb) {
            uint32_t block_idx = base_block + mb;

            uint4 entry;
            entry.x = local_expert;
            entry.y = count;
            entry.z = base_block;
            entry.w = base_m;

            reinterpret_cast<uint4*>(grouped_layout + 4)[block_idx] = entry;

            remote_addr_a[block_idx] = reinterpret_cast<uint64_t>(remote_fp4_base);
            remote_addr_sfa[block_idx] = reinterpret_cast<uint64_t>(remote_scale_base);
        }
    }

    if (threadIdx.x == 0 && profile_clocks) {
        _t5 = clock64();
        profile_clocks[0] = _t1 - _t0;
        profile_clocks[1] = _t2 - _t1;
        profile_clocks[2] = _t3 - _t2;
        profile_clocks[3] = _t4 - _t3;
        profile_clocks[4] = _t5 - _t4;
        profile_clocks[5] = _t5 - _t0;
    }
}

// Standalone kernel wrapper around dispatch_preprocess_device.
template <uint32_t BLOCK_M>
__global__ void dispatch_preprocess_kernel(
    const int64_t* __restrict__ sym_buf_addrs,
    uint32_t rank_idx,
    uint32_t num_ranks,
    uint32_t num_local_experts,
    uint32_t num_total_experts,
    uint32_t max_tokens_per_expert,
    uint32_t hidden_dim,
    uint32_t local_expert_start,
    uint32_t generation,
    int* __restrict__ grouped_layout,
    uint64_t* __restrict__ remote_addr_a,
    uint64_t* __restrict__ remote_addr_sfa,
    uint32_t* __restrict__ out_total_m_blocks,
    uint32_t* __restrict__ out_shape_m,
    uint64_t* __restrict__ profile_clocks = nullptr) {

    extern __shared__ uint32_t smem[];
    dispatch_preprocess_device<BLOCK_M>(
        sym_buf_addrs, rank_idx, num_ranks,
        num_local_experts, num_total_experts, max_tokens_per_expert, hidden_dim,
        local_expert_start, generation,
        grouped_layout, remote_addr_a, remote_addr_sfa,
        out_total_m_blocks, out_shape_m,
        smem, blockDim.x, profile_clocks);
}

// Host-side launcher
template <uint32_t BLOCK_M>
void launch_dispatch_preprocess(
    const int64_t* sym_buf_addrs,
    uint32_t rank_idx,
    uint32_t num_ranks,
    uint32_t num_local_experts,
    uint32_t num_total_experts,
    uint32_t max_tokens_per_expert,
    uint32_t hidden_dim,
    uint32_t local_expert_start,
    uint32_t generation,
    int* grouped_layout,
    uint64_t* remote_addr_a,
    uint64_t* remote_addr_sfa,
    uint32_t* out_total_m_blocks,
    uint32_t* out_shape_m,
    cudaStream_t stream,
    uint64_t* profile_clocks = nullptr) {

    uint32_t total_pairs = num_local_experts * num_ranks;
    uint32_t num_scan_warps_t = (num_local_experts + 31) / 32;
    uint32_t min_threads_for_scan = num_scan_warps_t * 32;
    uint32_t threads = max(min_threads_for_scan, min(total_pairs, 256u));
    uint32_t num_scan_warps = (num_local_experts + 31) / 32;
    uint32_t smem_size = (4 * total_pairs + 2 * num_scan_warps) * sizeof(uint32_t);

    dispatch_preprocess_kernel<BLOCK_M><<<1, threads, smem_size, stream>>>(
        sym_buf_addrs, rank_idx, num_ranks,
        num_local_experts, num_total_experts, max_tokens_per_expert, hidden_dim,
        local_expert_start,
        generation,
        grouped_layout, remote_addr_a, remote_addr_sfa,
        out_total_m_blocks, out_shape_m,
        profile_clocks);
}

// ===========================================================================
// Expert-level preprocess for merged groups (Split-AIU approach).
// Groups by expert (not expert-rank pair), producing per-rank split metadata
// so the GEMM can issue separate AIU copies for each rank's A data.
//
// Output arrays (per M-block × per rank):
//   rank_addr_a[block_idx * num_ranks + r]:   FP4 data address for rank r
//   rank_addr_sfa[block_idx * num_ranks + r]:  scale address for rank r
//   rank_split_m[block_idx * num_ranks + r]:   SMEM row offset for rank r (even-aligned for 128B)
//   rank_counts[block_idx * num_ranks + r]:    actual token count for rank r
// ===========================================================================
template <uint32_t BLOCK_M>
__device__ void dispatch_expert_preprocess_device(
    const int64_t* __restrict__ sym_buf_addrs,
    uint32_t rank_idx,
    uint32_t num_ranks,
    uint32_t num_local_experts,
    uint32_t num_total_experts,
    uint32_t max_tokens_per_expert,
    uint32_t hidden_dim,
    uint32_t local_expert_start,
    uint32_t generation,
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
    uint32_t num_threads) {

    long long _dbg_t0 = clock64();
    // Phase 1: SymBuffer init (same as original)
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

    // Double-buffering: this generation's data lives in buf[generation & 1].
    void* data_base = buf_layout.parity_base(sym_buffer.get_base_ptr<void*>(), generation & 1u);

    // Phase 2: Flag polling (same as original)
    if (generation > 0 && threadIdx.x < num_ranks) {
        // Atomic-arrival barrier: each producer p pushes its generation into THIS rank's
        // local slot[p] (see arrival_push_kernel). We poll the LOCAL slot instead of
        // spinning on a remote NVLink flag read — far less P2P polling traffic + jitter.
        volatile uint32_t* arrival_slots = reinterpret_cast<volatile uint32_t*>(
            static_cast<uint8_t*>(sym_buffer.get_base_ptr<void*>()) +
            buf_layout.ready_flag_offset());
        while (arrival_slots[threadIdx.x] < generation) { }
    }
    if (generation > 0) __syncthreads();
    long long _dbg_tA = clock64();  // after Phase 1+2 (setup + arrival barrier)

    // Phase 3: Read pair token counts (same as original)
    const uint32_t total_pairs = num_local_experts * num_ranks;

    uint32_t* pair_token_counts = smem_workspace;
    // Reuse remaining SMEM for expert-level aggregation
    uint32_t* expert_total_count = smem_workspace + total_pairs;
    uint32_t* expert_m_blocks_arr = expert_total_count + num_local_experts;
    uint32_t* expert_cumsum_blocks = expert_m_blocks_arr + num_local_experts;
    uint32_t* expert_cumsum_m = expert_cumsum_blocks + num_local_experts;

    for (uint32_t tid = threadIdx.x; tid < total_pairs; tid += num_threads) {
        uint32_t src_rank = tid / num_local_experts;
        uint32_t local_expert = tid % num_local_experts;
        uint32_t global_expert = local_expert_start + local_expert;

        uint32_t* local_counts_ptr = buf_layout.expert_token_counts_ptr(data_base);
        uint32_t* remote_counts_ptr = sym_buffer.map(local_counts_ptr, src_rank);

        uint32_t count = __ldg(remote_counts_ptr + global_expert);
        pair_token_counts[tid] = count;
    }
    __syncthreads();
    long long _dbg_tB = clock64();  // after Phase 3 (remote count reads)

    // Phase 4: Expert-level aggregation + prefix sum
    // Greedy packing: fill each M-block to capacity, splitting rank data across blocks.
    // No padding: the GEMM kernel handles arbitrary rank boundaries via CuTe predication.
    if (threadIdx.x < num_local_experts) {
        uint32_t e = threadIdx.x;
        uint32_t padded_total = 0;
        for (uint32_t r = 0; r < num_ranks; ++r) {
            uint32_t c = pair_token_counts[r * num_local_experts + e];
            padded_total += c;
        }
        // Use padded_total as M: ensures CuTe SFA copy covers all rank positions
        // and output tensor includes all tokens at their 8-aligned SMEM positions
        expert_total_count[e] = padded_total;
        expert_m_blocks_arr[e] = (padded_total + BLOCK_M - 1) / BLOCK_M;
    }
    __syncthreads();

    // Serial expert-level prefix sum
    if (threadIdx.x == 0) {
        uint32_t cumsum_b = 0, cumsum_m = 0;
        for (uint32_t e = 0; e < num_local_experts; ++e) {
            expert_cumsum_blocks[e] = cumsum_b;
            expert_cumsum_m[e] = cumsum_m;
            cumsum_b += expert_m_blocks_arr[e];
            cumsum_m += expert_total_count[e];
        }
        grouped_layout[0] = cumsum_b;
        if (out_total_m_blocks) *out_total_m_blocks = cumsum_b;
        if (out_shape_m) *out_shape_m = cumsum_m;
    }
    __syncthreads();

    // Phase 5: Greedy packing — fill M-blocks, rank data may span blocks
    if (threadIdx.x < num_local_experts) {
        uint32_t e = threadIdx.x;
        uint32_t total_count = expert_total_count[e];
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
            remaining[r] = pair_token_counts[r * num_local_experts + e];
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
                rank_addr_sfa[idx] = reinterpret_cast<uint64_t>(
                    remote_scale + rank_offset[r]);
                rank_split_m[idx] = smem_row;
                rank_counts[idx] = take;

                smem_row += take;
                remaining[r] -= take;
                rank_offset[r] += take;
            }
        }
    }

    // Debug timing: [setup+barrier, remote-count-reads, compute(Phase4+5)] cycles
    if (dbg_cyc != nullptr && threadIdx.x == 0) {
        long long _dbg_t2 = clock64();
        dbg_cyc[0] = (uint64_t)(_dbg_tA - _dbg_t0);
        dbg_cyc[1] = (uint64_t)(_dbg_tB - _dbg_tA);
        dbg_cyc[2] = (uint64_t)(_dbg_t2 - _dbg_tB);
    }
}

// Standalone kernel for expert-level preprocess
template <uint32_t BLOCK_M>
__global__ void dispatch_expert_preprocess_kernel(
    const int64_t* __restrict__ sym_buf_addrs,
    uint32_t rank_idx,
    uint32_t num_ranks,
    uint32_t num_local_experts,
    uint32_t num_total_experts,
    uint32_t max_tokens_per_expert,
    uint32_t hidden_dim,
    uint32_t local_expert_start,
    uint32_t generation,
    int* __restrict__ grouped_layout,
    uint64_t* __restrict__ rank_addr_a,
    uint64_t* __restrict__ rank_addr_sfa,
    uint32_t* __restrict__ rank_split_m,
    uint32_t* __restrict__ rank_counts,
    uint32_t* __restrict__ masked_m,
    uint32_t* __restrict__ out_total_m_blocks,
    uint32_t* __restrict__ out_shape_m,
    uint64_t* __restrict__ dbg_cyc) {

    extern __shared__ uint32_t smem[];
    dispatch_expert_preprocess_device<BLOCK_M>(
        sym_buf_addrs, rank_idx, num_ranks,
        num_local_experts, num_total_experts, max_tokens_per_expert, hidden_dim,
        local_expert_start, generation,
        grouped_layout, rank_addr_a, rank_addr_sfa,
        rank_split_m, rank_counts, masked_m,
        out_total_m_blocks, out_shape_m,
        dbg_cyc, smem, blockDim.x);
}

// Host launcher for expert-level preprocess
template <uint32_t BLOCK_M>
void launch_dispatch_expert_preprocess(
    const int64_t* sym_buf_addrs,
    uint32_t rank_idx,
    uint32_t num_ranks,
    uint32_t num_local_experts,
    uint32_t num_total_experts,
    uint32_t max_tokens_per_expert,
    uint32_t hidden_dim,
    uint32_t local_expert_start,
    uint32_t generation,
    int* grouped_layout,
    uint64_t* rank_addr_a,
    uint64_t* rank_addr_sfa,
    uint32_t* rank_split_m,
    uint32_t* rank_counts,
    uint32_t* masked_m,
    uint32_t* out_total_m_blocks,
    uint32_t* out_shape_m,
    uint64_t* dbg_cyc,
    cudaStream_t stream) {

    uint32_t total_pairs = num_local_experts * num_ranks;
    uint32_t num_scan_warps = (num_local_experts + 31) / 32;
    uint32_t min_threads = max(num_scan_warps * 32, min(total_pairs, 256u));
    // SMEM: pair_token_counts[total_pairs] + expert arrays[5 * NLE] (includes is_merged)
    uint32_t smem_size = (total_pairs + 5 * num_local_experts) * sizeof(uint32_t);

    dispatch_expert_preprocess_kernel<BLOCK_M><<<1, min_threads, smem_size, stream>>>(
        sym_buf_addrs, rank_idx, num_ranks,
        num_local_experts, num_total_experts, max_tokens_per_expert, hidden_dim,
        local_expert_start, generation,
        grouped_layout, rank_addr_a, rank_addr_sfa,
        rank_split_m, rank_counts, masked_m,
        out_total_m_blocks, out_shape_m, dbg_cyc);
}

}  // namespace deep_gemm
