#pragma once

#include <cstdint>
#include "dispatch_layout.cuh"
#include "utils_rtc.cuh"

namespace deep_gemm {

// ===========================================================================
// Expert preprocessing is split into two stages.
//
// Prepare is BLOCK_M-independent: wait for this generation, read remote counts
// once, convert them in place to rank-inclusive prefixes, and produce the
// GroupedMasked counts plus total/max M. The production 8-rank path stores the
// prefix cache expert-major for an eight-lane shuffle scan; other topologies keep
// rank-major order. Finalize is BLOCK_M-dependent: consume the cached prefixes to
// build the compact expert layout and the per-(M-block, rank) copy metadata.
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
    uint32_t num_threads,
    const int64_t* __restrict__ staging_addrs = nullptr) {  // DG_SFA_PUSH_COUNTS: local counts

    long long dbg_t0 = 0, dbg_t_setup = 0;
    long long dbg_t_arrived = 0, dbg_t_counts = 0;
    if (dbg_cyc != nullptr) dbg_t0 = clock64();
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
    if (dbg_cyc != nullptr) dbg_t_setup = clock64();

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
    if (dbg_cyc != nullptr) dbg_t_arrived = clock64();

    const uint32_t total_pairs = num_local_experts * num_ranks;
#ifdef DG_SFA_PUSH_COUNTS
    // Quant pushed the per-(src_rank, local_expert) counts (unpacked) into THIS rank's
    // staging counts region, indexed exactly as pair_counts[tid]. Read it locally instead
    // of a remote NVLink read per pair. Falls back to the remote read if no staging buffer.
    // Only gen>0 has a folded arrival that pushed counts; gen=0 (diagnostic) uses remote.
    if (generation > 0 && staging_addrs != nullptr && staging_addrs[rank_idx] != 0) {
        const uint32_t* cbase = reinterpret_cast<const uint32_t*>(
            reinterpret_cast<uint8_t*>(static_cast<intptr_t>(staging_addrs[rank_idx]))
            + buf_layout.staging_counts_offset()
            + static_cast<uint64_t>(generation & 1u) * buf_layout.staging_counts_parity_bytes());
        for (uint32_t tid = threadIdx.x; tid < total_pairs; tid += num_threads) {
            uint32_t src_rank = tid / num_local_experts;
            uint32_t local_expert = tid % num_local_experts;
            uint32_t cache_idx = (num_ranks == 8 && (num_local_experts & 3u) == 0)
                ? local_expert * 8u + src_rank : tid;
            pair_counts[cache_idx] = cbase[tid];
        }
    } else
#endif
    for (uint32_t tid = threadIdx.x; tid < total_pairs; tid += num_threads) {
        uint32_t src_rank = tid / num_local_experts;
        uint32_t local_expert = tid % num_local_experts;
        uint32_t global_expert = local_expert_start + local_expert;
        uint32_t* local_counts = buf_layout.expert_token_counts_ptr(data_base);
        uint32_t* remote_counts = sym_buffer.map(local_counts, src_rank);
        uint32_t cache_idx = (num_ranks == 8 && (num_local_experts & 3u) == 0)
            ? local_expert * 8u + src_rank : tid;
        pair_counts[cache_idx] = unpack_generation_count(
            __ldg(remote_counts + global_expert), generation, max_tokens_per_expert);
    }
    __syncthreads();
    if (dbg_cyc != nullptr) dbg_t_counts = clock64();

    if (num_ranks == 8 && (num_local_experts & 3u) == 0 &&
        threadIdx.x < total_pairs) {
        uint32_t pair = threadIdx.x;
        uint32_t r = pair & 7u;
        uint32_t value = pair_counts[pair];
        uint32_t up = __shfl_up_sync(0xffffffff, value, 1);
        if (r >= 1) value += up;
        up = __shfl_up_sync(0xffffffff, value, 2);
        if (r >= 2) value += up;
        up = __shfl_up_sync(0xffffffff, value, 4);
        if (r >= 4) value += up;
        pair_counts[pair] = value;
        if (r == 7)
            masked_m[pair / 8u] = value;
    } else if ((num_ranks != 8 || (num_local_experts & 3u) != 0) &&
               threadIdx.x < num_local_experts) {
        uint32_t e = threadIdx.x;
        uint32_t total = 0;
        for (uint32_t r = 0; r < num_ranks; ++r) {
            uint32_t idx = r * num_local_experts + e;
            total += pair_counts[idx];
            pair_counts[idx] = total;
        }
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
    }
    if (dbg_cyc != nullptr) {
        // Profiling-only completion barrier: make the final prepare timestamp cover
        // the slowest participating thread without affecting production kernels.
        __syncthreads();
        if (threadIdx.x == 0) {
            long long dbg_t_done = clock64();
            dbg_cyc[0] = static_cast<uint64_t>(dbg_t_setup - dbg_t0);
            dbg_cyc[1] = static_cast<uint64_t>(dbg_t_arrived - dbg_t_setup);
            dbg_cyc[2] = static_cast<uint64_t>(dbg_t_counts - dbg_t_arrived);
            dbg_cyc[3] = static_cast<uint64_t>(dbg_t_done - dbg_t_counts);
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

    long long dbg_t0 = 0, dbg_t_setup = 0;
    long long dbg_t_mblocks = 0, dbg_t_prefix = 0;
    if (dbg_cyc != nullptr) dbg_t0 = clock64();
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
    if (dbg_cyc != nullptr) dbg_t_setup = clock64();

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
    if (dbg_cyc != nullptr) dbg_t_mblocks = clock64();

    // Warp-parallel expert-level prefix sum for the common <=32-expert case.
    // Both block and row offsets use the same shuffle schedule; the existing
    // barrier below publishes all prefix entries to metadata packing.
    if (num_local_experts <= 32 && threadIdx.x < 32) {
        uint32_t lane = threadIdx.x;
        uint32_t own_b = (lane < num_local_experts)
            ? expert_m_blocks_arr[lane] : 0u;
        uint32_t own_m = (lane < num_local_experts)
            ? masked_m[lane] : 0u;
        uint32_t scan_b = own_b;
        uint32_t scan_m = own_m;
        #pragma unroll
        for (uint32_t offset = 1; offset < 32; offset <<= 1) {
            uint32_t up_b = __shfl_up_sync(0xffffffff, scan_b, offset);
            uint32_t up_m = __shfl_up_sync(0xffffffff, scan_m, offset);
            if (lane >= offset) {
                scan_b += up_b;
                scan_m += up_m;
            }
        }
        if (lane < num_local_experts) {
            expert_cumsum_blocks[lane] = scan_b - own_b;
            expert_cumsum_m[lane] = scan_m - own_m;
            if (lane + 1 == num_local_experts) {
                grouped_layout[0] = scan_b;
                if (out_total_m_blocks) *out_total_m_blocks = scan_b;
                if (out_shape_m) *out_shape_m = scan_m;
            }
        }
    } else if (num_local_experts > 32 && threadIdx.x == 0) {
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
    if (dbg_cyc != nullptr) dbg_t_prefix = clock64();

    // Phase 5: parallel interval packing. One logical thread owns an
    // (expert, source-rank) pair. The rank's global token interval is intersected
    // with each M-block interval, removing the serial per-expert rank state machine
    // and making each block's rank metadata stores contiguous across a warp.
    const uint32_t total_pairs = num_local_experts * num_ranks;
    for (uint32_t pair = threadIdx.x; pair < total_pairs; pair += num_threads) {
        uint32_t e = pair / num_ranks;
        uint32_t r = pair % num_ranks;
        uint32_t total_count = masked_m[e];
        uint32_t num_mblocks = expert_m_blocks_arr[e];
        uint32_t base_block = expert_cumsum_blocks[e];
        uint32_t base_m = expert_cumsum_m[e];
        uint32_t global_expert = local_expert_start + e;
        uint32_t k_half = hidden_dim / 2;

        // prepare converted counts to inclusive prefixes, so every pair
        // reconstructs its interval with at most two shared loads.
        const bool fast_rank_scan =
            num_ranks == 8 && (num_local_experts & 3u) == 0;
        uint32_t prefix_idx = fast_rank_scan
            ? e * 8u + r : r * num_local_experts + e;
        uint32_t rank_end = pair_counts[prefix_idx];
        uint32_t rank_begin = (r == 0) ? 0u : pair_counts[
            fast_rank_scan ? (prefix_idx - 1u)
                           : ((r - 1) * num_local_experts + e)];
        uint32_t rank_count = rank_end - rank_begin;

        if (r == 0) {
            // The second half maps an expert-local M-block back to the compact
            // copy-ready flag index. masked_m[e] already contains total_count.
            masked_m[num_local_experts + e] = base_block;
        }

        uint8_t* local_fp4 = buf_layout.fp4_data_ptr(data_base, global_expert);
        uint8_t* remote_fp4 = sym_buffer.map(local_fp4, r);
#if !defined(DG_SFA_PUSH)
        uint16_t* local_scale = buf_layout.scale_ptr(data_base, global_expert);
        uint16_t* remote_scale = sym_buffer.map(local_scale, r);
#endif

        for (uint32_t mb = 0; mb < num_mblocks; ++mb) {
            uint32_t block_idx = base_block + mb;
            uint32_t block_begin = mb * BLOCK_M;
            uint32_t begin = max(rank_begin, block_begin);
            uint32_t end = min(rank_end, block_begin + BLOCK_M);
            uint32_t idx = block_idx * num_ranks + r;

            if (r == 0) {
                // entry.y is the expert total (not this block's count), so the
                // masked scheduler reconstructs ceil(total/BLOCK_M) consistently.
                uint4 entry;
                entry.x = e;
                entry.y = total_count;
                entry.z = base_block;
                entry.w = base_m;
                reinterpret_cast<uint4*>(grouped_layout + 4)[block_idx] = entry;
            }

            if (begin >= end) {
                rank_addr_a[idx] = 0;
                rank_addr_sfa[idx] = 0;
                rank_split_m[idx] = 0;
                rank_counts[idx] = 0;
                continue;
            }

            uint32_t rank_offset = begin - rank_begin;
            rank_addr_a[idx] = reinterpret_cast<uint64_t>(
                remote_fp4 + (uint64_t)rank_offset * k_half);
#if defined(DG_SFA_PUSH)
            // PUSH: SFA is not pulled from the source. Quant pushed each token's
            // ksb scales into this rank's separate staging buffer band.
            {
                uint32_t ksb = ((hidden_dim / 2) + 31u) / 32u;
                uint64_t staging_local_base =
                    (staging_addrs != nullptr) ? (uint64_t)staging_addrs[rank_idx] : 0ull;
                uint16_t* staging = reinterpret_cast<uint16_t*>(
                    staging_local_base
                    + (uint64_t)(generation & 1u) * buf_layout.staging_parity_bytes());
                uint64_t sidx = ((uint64_t)(e * num_ranks + r) * max_tokens_per_expert
                                 + rank_offset) * ksb;
                rank_addr_sfa[idx] = reinterpret_cast<uint64_t>(staging + sidx);
            }
#elif defined(DG_SFA_ROWMAJOR_SRC)
            {
                uint32_t ksb = ((hidden_dim / 2) + 31u) / 32u;
                rank_addr_sfa[idx] = reinterpret_cast<uint64_t>(
                    remote_scale + (uint64_t)rank_offset * ksb);
            }
#else
            rank_addr_sfa[idx] = reinterpret_cast<uint64_t>(remote_scale + rank_offset);
#endif
            rank_split_m[idx] = begin - block_begin;
            rank_counts[idx] = end - begin;
        }
    }

    if (dbg_cyc != nullptr) {
        // Greedy packing is distributed over expert threads. Wait for the slowest
        // one only in the profiling specialization so slot 7 is a CTA duration.
        __syncthreads();
        if (threadIdx.x == 0) {
            long long dbg_t_done = clock64();
            dbg_cyc[4] = static_cast<uint64_t>(dbg_t_setup - dbg_t0);
            dbg_cyc[5] = static_cast<uint64_t>(dbg_t_mblocks - dbg_t_setup);
            dbg_cyc[6] = static_cast<uint64_t>(dbg_t_prefix - dbg_t_mblocks);
            dbg_cyc[7] = static_cast<uint64_t>(dbg_t_done - dbg_t_prefix);
        }
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
// CTA with a __syncthreads() barrier. pair_counts stays in shared memory between
// the two stages; masked_m remains global because it is also a GEMM output.
// Saves one kernel launch, the kernel-boundary gap, and the intermediate global
// pair-count round trip vs the split prepare();finalize() sequence. Arrival
// barrier logic is unchanged.
// ===========================================================================
template <uint32_t BLOCK_M,
          uint32_t RANK_IDX_T,
          uint32_t NUM_RANKS_T,
          uint32_t NUM_LOCAL_EXPERTS_T,
          uint32_t NUM_TOTAL_EXPERTS_T,
          uint32_t MAX_TOKENS_PER_EXPERT_T,
          uint32_t HIDDEN_DIM_T,
          uint32_t LOCAL_EXPERT_START_T,
          bool PROFILE_ENABLED_T>
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
    const uint32_t total_pairs = NUM_LOCAL_EXPERTS_T * NUM_RANKS_T;
    uint32_t* smem_pair_counts = smem;
    uint32_t* finalize_smem = smem + total_pairs;

    // Kept in the launcher ABI so split and merged workspaces remain identical.
    // The merged path has no consumer for the global intermediate.
    (void)pair_counts;
    uint64_t* profile_cyc = nullptr;
    if constexpr (PROFILE_ENABLED_T) profile_cyc = dbg_cyc;

    dispatch_expert_prepare_device(
        sym_buf_addrs, RANK_IDX_T, NUM_RANKS_T,
        NUM_LOCAL_EXPERTS_T, NUM_TOTAL_EXPERTS_T,
        MAX_TOKENS_PER_EXPERT_T, HIDDEN_DIM_T,
        LOCAL_EXPERT_START_T, generation,
        smem_pair_counts, masked_m, out_shape_m, out_expected_m, profile_cyc,
        blockDim.x, staging_addrs);

    __syncthreads();  // publish prepare's masked_m to the block

    dispatch_expert_finalize_device<BLOCK_M>(
        sym_buf_addrs, RANK_IDX_T, NUM_RANKS_T,
        NUM_LOCAL_EXPERTS_T, NUM_TOTAL_EXPERTS_T,
        MAX_TOKENS_PER_EXPERT_T, HIDDEN_DIM_T,
        LOCAL_EXPERT_START_T, generation,
        smem_pair_counts, grouped_layout, rank_addr_a, rank_addr_sfa,
        rank_split_m, rank_counts, masked_m,
        out_total_m_blocks, out_shape_m,
        profile_cyc, finalize_smem, blockDim.x, staging_addrs);
}

template <uint32_t BLOCK_M,
          uint32_t RANK_IDX_T,
          uint32_t NUM_RANKS_T,
          uint32_t NUM_LOCAL_EXPERTS_T,
          uint32_t NUM_TOTAL_EXPERTS_T,
          uint32_t MAX_TOKENS_PER_EXPERT_T,
          uint32_t HIDDEN_DIM_T,
          uint32_t LOCAL_EXPERT_START_T,
          bool PROFILE_ENABLED_T>
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

    constexpr uint32_t total_pairs = NUM_LOCAL_EXPERTS_T * NUM_RANKS_T;
    uint32_t prep_threads = max(32u, min(max(total_pairs, NUM_LOCAL_EXPERTS_T), 256u));
    constexpr uint32_t num_scan_warps = (NUM_LOCAL_EXPERTS_T + 31) / 32;
    uint32_t fin_threads = max(num_scan_warps * 32, min(NUM_LOCAL_EXPERTS_T, 256u));
    uint32_t threads = max(prep_threads, fin_threads);
    constexpr uint32_t smem_size =
        (total_pairs + 3 * NUM_LOCAL_EXPERTS_T) * sizeof(uint32_t);

    dispatch_expert_preprocess_merged_kernel<
        BLOCK_M, RANK_IDX_T, NUM_RANKS_T, NUM_LOCAL_EXPERTS_T,
        NUM_TOTAL_EXPERTS_T, MAX_TOKENS_PER_EXPERT_T, HIDDEN_DIM_T,
        LOCAL_EXPERT_START_T, PROFILE_ENABLED_T><<<1, threads, smem_size, stream>>>(
        sym_buf_addrs, rank_idx, num_ranks,
        num_local_experts, num_total_experts, max_tokens_per_expert, hidden_dim,
        local_expert_start, generation,
        pair_counts, grouped_layout, rank_addr_a, rank_addr_sfa,
        rank_split_m, rank_counts, masked_m,
        out_total_m_blocks, out_shape_m, out_expected_m, dbg_cyc, staging_addrs);
}

// ===========================================================================
// Pre-GEMM overlap path for pushed SFA.
//
// SFA reshape does not require the materialized per-M-block metadata.  For one
// expert, the source rank counts alone define the same rank-concatenated row
// layout: rank r starts at prefix_sum(count[0:r]).  Reshape CTAs therefore wait
// for the peer-arrival publication, read the tagged counts directly from every
// source rank, derive their prefix locally, and transpose pushed row-major SFA
// into local_sfa_buf while CTA 0 builds the normal FP4/GEMM metadata.
//
// Keep the row transform as a device helper so a future larger pre-GEMM kernel
// can compose it with different CTA-role or synchronization policies.
// ===========================================================================
template <uint32_t RANK_IDX_T,
          uint32_t NUM_RANKS_T,
          uint32_t NUM_LOCAL_EXPERTS_T,
          uint32_t NUM_TOTAL_EXPERTS_T,
          uint32_t MAX_TOKENS_PER_EXPERT_T,
          uint32_t HIDDEN_DIM_T,
          uint32_t LOCAL_EXPERT_START_T>
__device__ __forceinline__ void dispatch_expert_reshape_pushed_sfa_device(
    const int64_t* __restrict__ sym_buf_addrs,
    const int64_t* __restrict__ staging_addrs,
    uint32_t generation,
    uint32_t expert_local,
    uint32_t rank_start,
    uint32_t rank_stride,
    uint16_t* __restrict__ local_sfa_buf,
    uint32_t* smem_rank_counts)
{
    constexpr uint32_t kScaleBlocks = ((HIDDEN_DIM_T / 2) + 31) / 32;
    constexpr uint32_t kVecsPerToken = kScaleBlocks / 8;
    static_assert(kScaleBlocks % 8 == 0,
                  "SFA overlap requires an int4-aligned scale K dimension");

    DispatchBufferLayout layout(NUM_TOTAL_EXPERTS_T, NUM_TOTAL_EXPERTS_T,
                                MAX_TOKENS_PER_EXPERT_T, HIDDEN_DIM_T);
    const uint32_t global_expert = LOCAL_EXPERT_START_T + expert_local;
    if (threadIdx.x < NUM_RANKS_T) {
        const uint32_t source_rank = threadIdx.x;
        void* peer_base = reinterpret_cast<void*>(
            static_cast<intptr_t>(sym_buf_addrs[source_rank]));
        void* peer_parity = layout.parity_base(peer_base, generation & 1u);
        const uint32_t* peer_counts = layout.expert_token_counts_ptr(peer_parity);
        smem_rank_counts[source_rank] = unpack_generation_count(
            __ldg(peer_counts + global_expert), generation,
            MAX_TOKENS_PER_EXPERT_T);
    }
    __syncthreads();

    uint8_t* staging_base = reinterpret_cast<uint8_t*>(
        static_cast<intptr_t>(staging_addrs[RANK_IDX_T]));
    const uint16_t* staging = reinterpret_cast<const uint16_t*>(
        staging_base + static_cast<uint64_t>(generation & 1u) *
        layout.staging_parity_bytes());
    uint16_t* dst_expert = local_sfa_buf +
        static_cast<uint64_t>(expert_local) * kScaleBlocks *
        MAX_TOKENS_PER_EXPERT_T;

    for (uint32_t r = rank_start; r < NUM_RANKS_T; r += rank_stride) {
        uint32_t rank_begin = 0;
        #pragma unroll
        for (uint32_t q = 0; q < r; ++q)
            rank_begin += smem_rank_counts[q];
        const uint32_t count = smem_rank_counts[r];

        const uint64_t src_index =
            (static_cast<uint64_t>(expert_local * NUM_RANKS_T + r) *
             MAX_TOKENS_PER_EXPERT_T) * kScaleBlocks;
        const uint16_t* src = staging + src_index;
        uint16_t* dst = dst_expert + rank_begin;
        const uint32_t total_vecs = count * kVecsPerToken;
        for (uint32_t linear = threadIdx.x; linear < total_vecs;
             linear += blockDim.x) {
            const uint32_t token = linear / kVecsPerToken;
            const uint32_t vec_k = linear - token * kVecsPerToken;
            int4 value = __ldbl(
                reinterpret_cast<const int4*>(src +
                    static_cast<uint64_t>(token) * kScaleBlocks) + vec_k);
            const uint16_t* elems = reinterpret_cast<const uint16_t*>(&value);
            const uint32_t kb0 = vec_k * 8;
            #pragma unroll
            for (uint32_t e = 0; e < 8; ++e) {
                dst[static_cast<uint64_t>(kb0 + e) *
                    MAX_TOKENS_PER_EXPERT_T + token] = elems[e];
            }
        }
    }
}

template <uint32_t BLOCK_M,
          uint32_t RANK_IDX_T,
          uint32_t NUM_RANKS_T,
          uint32_t NUM_LOCAL_EXPERTS_T,
          uint32_t NUM_TOTAL_EXPERTS_T,
          uint32_t MAX_TOKENS_PER_EXPERT_T,
          uint32_t HIDDEN_DIM_T,
          uint32_t LOCAL_EXPERT_START_T>
__global__ void dispatch_expert_sfa_overlap_kernel(
    const int64_t* __restrict__ sym_buf_addrs,
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
    uint16_t* __restrict__ local_sfa_buf,
    uint32_t* __restrict__ copy_ready_flags,
    uint32_t num_copy_ready_flags,
    const int64_t* __restrict__ staging_addrs)
{
    static_assert(BLOCK_M == 128 && NUM_RANKS_T == 8 &&
                  NUM_LOCAL_EXPERTS_T == 12 && NUM_TOTAL_EXPERTS_T == 96 &&
                  MAX_TOKENS_PER_EXPERT_T == 256 && HIDDEN_DIM_T == 7168,
                  "SFA overlap is currently specialized to the production shape");
    constexpr uint32_t kRankSubgroups = 3;
    constexpr uint32_t kReshapeWork =
        NUM_LOCAL_EXPERTS_T * kRankSubgroups;

    extern __shared__ uint32_t smem[];
    if (blockIdx.x == 0) {
        for (uint32_t i = threadIdx.x; i < num_copy_ready_flags;
             i += blockDim.x)
            copy_ready_flags[i] = 0;

        constexpr uint32_t total_pairs =
            NUM_LOCAL_EXPERTS_T * NUM_RANKS_T;
        uint32_t* smem_pair_counts = smem;
        uint32_t* finalize_smem = smem + total_pairs;
        (void)pair_counts;

        dispatch_expert_prepare_device(
            sym_buf_addrs, RANK_IDX_T, NUM_RANKS_T,
            NUM_LOCAL_EXPERTS_T, NUM_TOTAL_EXPERTS_T,
            MAX_TOKENS_PER_EXPERT_T, HIDDEN_DIM_T,
            LOCAL_EXPERT_START_T, generation,
            smem_pair_counts, masked_m, out_shape_m, out_expected_m,
            nullptr, blockDim.x, staging_addrs);
        __syncthreads();
        dispatch_expert_finalize_device<BLOCK_M>(
            sym_buf_addrs, RANK_IDX_T, NUM_RANKS_T,
            NUM_LOCAL_EXPERTS_T, NUM_TOTAL_EXPERTS_T,
            MAX_TOKENS_PER_EXPERT_T, HIDDEN_DIM_T,
            LOCAL_EXPERT_START_T, generation,
            smem_pair_counts, grouped_layout, rank_addr_a, rank_addr_sfa,
            rank_split_m, rank_counts, masked_m,
            out_total_m_blocks, out_shape_m,
            nullptr, finalize_smem, blockDim.x, staging_addrs);
        return;
    }

    const uint32_t work = blockIdx.x - 1;
    if (work >= kReshapeWork) return;

    DispatchBufferLayout layout(NUM_TOTAL_EXPERTS_T, NUM_TOTAL_EXPERTS_T,
                                MAX_TOKENS_PER_EXPERT_T, HIDDEN_DIM_T);
    volatile uint32_t* arrival_slots = reinterpret_cast<volatile uint32_t*>(
        reinterpret_cast<uint8_t*>(
            static_cast<intptr_t>(sym_buf_addrs[RANK_IDX_T])) +
        layout.ready_flag_offset());
    if (threadIdx.x < NUM_RANKS_T) {
        while (arrival_slots[threadIdx.x] < generation) { }
    }
    __syncthreads();
    asm volatile("" ::: "memory");

    const uint32_t expert_local = work / kRankSubgroups;
    const uint32_t rank_start = work - expert_local * kRankSubgroups;
    dispatch_expert_reshape_pushed_sfa_device<
        RANK_IDX_T, NUM_RANKS_T, NUM_LOCAL_EXPERTS_T,
        NUM_TOTAL_EXPERTS_T, MAX_TOKENS_PER_EXPERT_T,
        HIDDEN_DIM_T, LOCAL_EXPERT_START_T>(
            sym_buf_addrs, staging_addrs, generation, expert_local,
            rank_start, kRankSubgroups, local_sfa_buf, smem);
}

template <uint32_t BLOCK_M,
          uint32_t RANK_IDX_T,
          uint32_t NUM_RANKS_T,
          uint32_t NUM_LOCAL_EXPERTS_T,
          uint32_t NUM_TOTAL_EXPERTS_T,
          uint32_t MAX_TOKENS_PER_EXPERT_T,
          uint32_t HIDDEN_DIM_T,
          uint32_t LOCAL_EXPERT_START_T>
void launch_dispatch_expert_sfa_overlap(
    const int64_t* sym_buf_addrs,
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
    uint16_t* local_sfa_buf,
    uint32_t* copy_ready_flags,
    uint32_t num_copy_ready_flags,
    const int64_t* staging_addrs,
    cudaStream_t stream)
{
    constexpr uint32_t smem_size =
        (NUM_LOCAL_EXPERTS_T * NUM_RANKS_T +
         3 * NUM_LOCAL_EXPERTS_T) * sizeof(uint32_t);
    dispatch_expert_sfa_overlap_kernel<
        BLOCK_M, RANK_IDX_T, NUM_RANKS_T, NUM_LOCAL_EXPERTS_T,
        NUM_TOTAL_EXPERTS_T, MAX_TOKENS_PER_EXPERT_T, HIDDEN_DIM_T,
        LOCAL_EXPERT_START_T><<<39, 256, smem_size, stream>>>(
            sym_buf_addrs, generation, pair_counts, grouped_layout,
            rank_addr_a, rank_addr_sfa, rank_split_m, rank_counts, masked_m,
            out_total_m_blocks, out_shape_m, out_expected_m,
            local_sfa_buf, copy_ready_flags, num_copy_ready_flags,
            staging_addrs);
}

}  // namespace deep_gemm
