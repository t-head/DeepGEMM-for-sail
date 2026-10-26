#pragma once

// ============================================================================
// PPU mega-MoE: fused Dispatch + GEMM1 kernel.
// ----------------------------------------------------------------------------
// Design: cross-SM producer/consumer specialization with per-block arrival
// counters. PPU has no per-workgroup register allocation, so the communication
// and GEMM roles are assigned to different CTAs rather than different warps in
// one CTA.
//
//   Phase A  -- dispatch handshake (once):
//       count routed tokens per expert -> reserve send slots -> write the
//       source token-topk indices into the destination rank's workspace ->
//       grid_sync -> SM0 publishes recv counts -> cross_rank_barrier. After this
//       every rank's recv counts and src indices are globally consistent.
//
//   Phase B  -- pull + prefix gemm1 (overlapped):
//       Producer SMs split every M block's tokens, pull them into the local
//       landing pool, and release the block through `l1_arrival_count`.
//       Consumer SMs execute the prefix GEMM work list and can start each block
//       as soon as every producer has published its slice.
//
//   After a producer completes its pulls, it joins the same GEMM1 tile queue
//   as the initial consumers. Every claimed M block retains its arrival check.
//
// Data format: packed MXFP4 tokens with per-32 E8M0 scales. GEMM1 writes
// activated MXFP4 values and their scales for GEMM2.
//
// Cross-rank acquire/release operations use the scope-aware PPU wrappers in
// tix/sys_atomics.cuh.
// ============================================================================

#include <cstdint>
#include <cute/arch/copy_ppu.hpp>
#include <hggc_runtime.h>
#include <hggc_bf16.h>
#include <deep_gemm/layout/mega_moe_common.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/impls/fp4_mega_moe_gemm1_mma.cuh>
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>
#include <deep_gemm/scheduler/mega_moe_work_queue.cuh>

namespace deep_gemm::mega_moe::fp4 {

// Vectorized global->global copy of `num_bytes` bytes by the calling warp.
// `dst` / `src` are 16B-aligned when `num_bytes % 16 == 0` (mxfp4 token rows).
CUTLASS_DEVICE void warp_copy_bytes(void* dst, const void* src,
                                    const uint32_t& num_bytes,
                                    const uint32_t& lane_idx) {
    auto* d16 = reinterpret_cast<uint4*>(dst);
    const auto* s16 = reinterpret_cast<const uint4*>(src);
    const uint32_t n16 = num_bytes / 16;
    for (uint32_t i = lane_idx; i < n16; i += 32)
        d16[i] = s16[i];
    // Tail (should be empty for 16B-aligned mxfp4 rows; kept for safety)
    const uint32_t tail_base = n16 * 16;
    auto* d1 = reinterpret_cast<uint8_t*>(dst);
    const auto* s1 = reinterpret_cast<const uint8_t*>(src);
    for (uint32_t i = tail_base + lane_idx; i < num_bytes; i += 32)
        d1[i] = s1[i];
}

CUTLASS_DEVICE uint32_t umin(const uint32_t& a, const uint32_t& b) { return a < b ? a : b; }
CUTLASS_DEVICE uint32_t umax(const uint32_t& a, const uint32_t& b) { return a > b ? a : b; }

// FP4 e2m1 magnitude lookup (sign in bit 3). Matches tests' dequantize_fp4_torch.
CUTLASS_DEVICE float fp4_decode_nibble(const uint32_t& nib) {
    const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float mag = lut[nib & 0x7u];
    return (nib & 0x8u) ? -mag : mag;
}
// Per-32 mxfp4 scale: uint8 exponent byte -> float32 (byte << 23 reinterpreted).
CUTLASS_DEVICE float fp4_decode_scale(const uint8_t& sf) {
    return __uint_as_float(static_cast<uint32_t>(sf) << 23);
}

// ---------------------------------------------------------------------------
// Warp-distributed per-expert metadata helpers.
// Instead of every thread holding a full [kNumExpertsPerRank] array (which the
// compiler is forced to place in local memory because of runtime indexing in
// the consumer's expert resolve loop -> register spill, the gemm1 bottleneck),
// we distribute the array across the warp: blocked layout, lane L owns experts
// [L*S, L*S+S-1] in S=ceil(kNumExpertsPerRank/32) registers. Reads use a warp
// shuffle broadcast (the access index `e` is warp-uniform at every use site).
// ---------------------------------------------------------------------------
// Broadcast arr[e] (e warp-uniform) from its owning lane. `arr` is passed by
// reference so the unrolled (static) indexing keeps it in registers.
template <uint32_t S>
CUTLASS_DEVICE uint32_t dist_get(const uint32_t (&arr)[S], const uint32_t& e) {
    const uint32_t slot = e % S;
    uint32_t v = 0;
    #pragma unroll
    for (uint32_t i = 0; i < S; ++i)
        if (i == slot) v = arr[i];
    return __shfl_sync(0xffffffffu, v, e / S);
}

// Warp exclusive prefix-sum (Hillis-Steele) of `val` over the 32 lanes.
CUTLASS_DEVICE uint32_t lane_exclusive_scan(const uint32_t& val, const uint32_t& lane_idx) {
    uint32_t x = val;
    #pragma unroll
    for (uint32_t off = 1; off < 32; off <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, x, off);
        if (lane_idx >= off) x += y;
    }
    return x - val;  // inclusive - self = exclusive
}

// ----------------------------------------------------------------------------
// Phase A: dispatch handshake. After it returns, every rank's per-(src_rank,
// expert) recv counts and src token-topk indices are globally consistent, so
// the chunked pull can begin. Architecture-independent (counts + .sys atomics).
// ----------------------------------------------------------------------------
template <
    uint32_t kNumRanks, uint32_t kNumCTAs, uint32_t kNumThreads,
    uint32_t kNumExperts, uint32_t kNumExpertsPerRank, uint32_t kNumTopk>
CUTLASS_DEVICE void mega_moe_dispatch_handshake(
        const layout::Workspace& workspace,
        const layout::SymBuffer<kNumRanks>& sym_buffer,
        const int64_t* __restrict__ input_topk_idx,
        uint32_t* __restrict__ smem_expert_count,
        const uint32_t& num_tokens,
        const uint32_t& cta_idx,
        const uint32_t& thread_idx) {
    const auto block_sync = [=]() { __syncthreads(); };

    for (uint32_t i = thread_idx; i < kNumExperts; i += kNumThreads)
        smem_expert_count[i] = 0;
    __syncthreads();

    const auto for_each_routed = [&](const auto& process) {
        const uint32_t total = num_tokens * kNumTopk;
        const uint32_t first = cta_idx * kNumThreads + thread_idx;
        const uint32_t stride = kNumCTAs * kNumThreads;
        for (uint32_t p = first; p < total; p += stride) {
            const int64_t expert_idx = __ldg(input_topk_idx + p);
            if (expert_idx >= 0 && expert_idx < kNumExperts)
                process(p, static_cast<uint32_t>(expert_idx));
        }
    };
    for_each_routed([&](const uint32_t&, const uint32_t& expert_idx) {
        atomicAdd_block(smem_expert_count + expert_idx, 1u);
    });
    __syncthreads();

    for (uint32_t i = thread_idx; i < kNumExperts; i += kNumThreads) {
        const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
        const uint64_t old_value = tix::atomic_add_sys(workspace.get_expert_send_count_ptr(i), send_value);
        smem_expert_count[i] = static_cast<uint32_t>(old_value);
    }
    __syncthreads();

    for_each_routed([&](const uint32_t& token_topk_idx, const uint32_t& expert_idx) {
        const uint32_t dst_rank_idx = expert_idx / kNumExpertsPerRank;
        const uint32_t local_expert_idx = expert_idx % kNumExpertsPerRank;
        const uint32_t dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1u);
        auto* dst_ptr = workspace.get_src_token_topk_idx_ptr(local_expert_idx, sym_buffer.rank_idx, dst_slot_idx);
        *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
    });
    comm::grid_sync<kNumCTAs, /*kGridSyncIndex=*/0, /*kPollBackoffCycles=*/0>(
        workspace, cta_idx, thread_idx, block_sync);

    if (cta_idx == 0) {
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumThreads) {
            const uint32_t dst_rank_idx = i / kNumExpertsPerRank;
            const uint32_t local_expert_idx = i % kNumExpertsPerRank;
            const uint64_t expert_status =
                *workspace.get_expert_send_count_ptr(i);
            *sym_buffer.map(
                workspace.get_expert_recv_count_ptr(
                    sym_buffer.rank_idx, local_expert_idx),
                dst_rank_idx) = expert_status & 0xffffffffull;
            tix::atomic_add_sys(
                sym_buffer.map(
                    workspace.get_expert_recv_count_sum_ptr(local_expert_idx),
                    dst_rank_idx),
                expert_status);
        }
    }
    __syncthreads();

    // Route stores are already grid-synchronized above. Only CTA0 publishes
    // counts afterward, ordered by the preceding CTA barrier.
    comm::cross_rank_barrier<
        kNumRanks, kNumCTAs, kNumThreads, /*idx=*/0, /*tag=*/0>(
        workspace, sym_buffer, cta_idx, thread_idx, block_sync,
        false, true);
}

// ----------------------------------------------------------------------------
// Fused kernel entry.
//
// Dispatch handshake precedes overlapped pull and GEMM1. In queue mode,
// producers join the compute queue after their own pulls; each M block
// retains its arrival check.
// ----------------------------------------------------------------------------
template <
    uint32_t kNumRanks, uint32_t kNumCTAs, uint32_t kNumThreads,
    uint32_t kNumExperts, uint32_t kNumExpertsPerRank, uint32_t kNumTopk,
    uint32_t BLOCK_M,
    uint32_t kNumPullCTAs = kNumCTAs,
    uint32_t kNumGemmCTAs = kNumCTAs,
    uint32_t kPullSlots = 8,
    uint32_t kGemm1N = 0,
    uint32_t kTokenBytes = 0,
    uint32_t kGemm1BlockN = 0,  // gemm1 N-tile; 0 => full N (single tile)
    uint32_t kGemm1BlockK = 0,  // gemm1 K-tile; 0 => full K (single tile)
    uint32_t kGemm1SfbBlockN = kGemm1BlockN,  // physical canonical SFB N-tile
    uint32_t kGemm1WarpM = 32,  // gemm1 TiledMMA warp tile (M). Must divide BLOCK_M.
    uint32_t kGemm1WarpN = 64,  // gemm1 TiledMMA warp tile (N). Must divide kBlockN.
    uint32_t kGemm1Stages = 2,  // gemm1 mainloop pipeline stages (CUTLASS smem stages).
    bool kFusedSitu = false, // compile-time: eliminate inactive GLU epilogue
    bool kSiluThenClamp = false,
    uint32_t kGemm2BlockM = 0,
    bool kDynamicWork = false,
    bool kFastMath = true
>
// Called by the native JIT-generated global entry.
__device__ void
dispatch_gemm1_device(
        void* workspace_base,
        const layout::SymBuffer<kNumRanks>& sym_buffer,
        const int64_t* __restrict__ input_topk_idx,
        const uint8_t* __restrict__ input_token,
        const uint8_t* __restrict__ input_sf,
        uint8_t* __restrict__ dispatched_token_pool,
        uint8_t* __restrict__ dispatched_scale_pool,
        void* __restrict__ gemm2_metadata,
        int32_t* __restrict__ cumulative_local_expert_recv_stats,
        const uint32_t num_tokens,
        const uint32_t num_max_tokens_per_rank,
        const uint32_t token_bytes,
        const uint32_t sf_bytes,
        const uint8_t* __restrict__ gemm1_weight,     // [EPR, N, K/2] packed FP4
        const uint16_t* __restrict__ gemm1_weight_sf_u16,  // [EPR, N, K/32/2] uint16 N-major (TC)
        const uint8_t* __restrict__ gemm1_weight_replicas,
        const uint16_t* __restrict__ gemm1_weight_sf_u16_replicas,
        const uint32_t num_master_experts,
        uint16_t* __restrict__ sfa_scratch,             // [total_blocks, BLOCK_M*K/32/2] uint16 (TC)
        float activation_clamp = -1.0f,                   // >=0: clamp; negative: no clamp
        bool fused_swiglu = false,                        // fuse SwiGLU into epilogue
        bool fused_situ = false,                          // fuse Kimi K3 SiTU-GLU into epilogue
        float situ_beta = 4.0f,                           // gate tanh saturation
        float situ_linear_beta = 25.0f,                   // up-branch tanh saturation
        uint8_t* __restrict__ fp4_output = nullptr,       // [num_pool, N/4] packed FP4
        uint8_t* __restrict__ scale_output = nullptr,     // byte view of packed/transposed uint16 E8M0 scales
        int rank = 0) {

    const uint32_t cta_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = thread_idx / 32;
    const uint32_t lane_idx   = thread_idx % 32;

    constexpr uint32_t kNumWarpsPerBlock = kNumThreads / 32;

    // Reconstruct the architecture-independent workspace view.
    const layout::Workspace workspace(
        workspace_base, kNumRanks, kNumExperts, num_max_tokens_per_rank, kNumTopk);

    extern __shared__ __align__(16) unsigned char smem_buf[];
    auto* smem_expert_count = reinterpret_cast<uint32_t*>(smem_buf);  // [kNumExperts]
    const auto block_sync = [=]() { __syncthreads(); };

    // Reset the arrival span before the handshake's existing grid barriers.
    // The runtime tile determines the maximum reachable prefix; clearing that
    // prefix avoids a separate grid barrier after metadata construction.
    if constexpr (kDynamicWork)
        if (cta_idx == 0 && thread_idx == 0)
            *workspace.get_gemm1_work_count_ptr() = 0;
    for (uint32_t g = cta_idx * kNumThreads + thread_idx;
         g < workspace.num_max_pool_tokens / BLOCK_M;
         g += kNumCTAs * kNumThreads)
        *workspace.get_l1_arrival_count_ptr(g) = 0;

    // ==================== Phase A: dispatch handshake ====================

    mega_moe_dispatch_handshake<
        kNumRanks, kNumCTAs, kNumThreads, kNumExperts,
        kNumExpertsPerRank, kNumTopk>(
        workspace, sym_buffer, input_topk_idx, smem_expert_count,
        num_tokens, cta_idx, thread_idx);

    // ==================== Phase B: specialized pull + prefix GEMM ====================
    // Cache per-expert recv counts and pool-block prefix.
    // Warp-distributed: each lane stores ceil(EPR/32) experts to avoid spill.
    constexpr uint32_t kPerLaneSlots = (kNumExpertsPerRank + 31) / 32;
    uint32_t l_num_recv      [kPerLaneSlots] = {};
    uint32_t l_pool_block_off[kPerLaneSlots] = {};
    uint32_t my_total = 0;
    #pragma unroll
    for (uint32_t s = 0; s < kPerLaneSlots; ++s) {
        const uint32_t e = lane_idx * kPerLaneSlots + s;
        uint32_t nr = 0, nmb = 0;
        if (e < kNumExpertsPerRank) {
            nr = static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(e));
            nmb = math::ceil_div(nr, BLOCK_M);
        }
        l_num_recv[s]        = nr;
        l_pool_block_off[s]  = my_total;
        my_total            += nmb;
    }
    // Warp prefix scan: compute per-expert global block offset prefix
    const uint32_t my_global_off = lane_exclusive_scan(my_total, lane_idx);
    #pragma unroll
    for (uint32_t s = 0; s < kPerLaneSlots; ++s)
        l_pool_block_off[s] += my_global_off;
    const uint32_t total_blocks = __shfl_sync(0xffffffffu, my_total + my_global_off, 31);
    __syncwarp();

    const layout::Gemm2Metadata gemm2_meta(
        gemm2_metadata, kNumExpertsPerRank);
    uint32_t l_gemm2_off[kPerLaneSlots] = {};
    if constexpr (kGemm2BlockM != 0) {
        uint32_t total = 0;
        #pragma unroll
        for (uint32_t s = 0; s < kPerLaneSlots; ++s) {
            l_gemm2_off[s] = total;
            total += math::ceil_div(l_num_recv[s], kGemm2BlockM) * kGemm2BlockM;
        }
        const uint32_t prefix = lane_exclusive_scan(total, lane_idx);
        #pragma unroll
        for (uint32_t s = 0; s < kPerLaneSlots; ++s) {
            l_gemm2_off[s] += prefix;
            const uint32_t e = lane_idx * kPerLaneSlots + s;
            if (cta_idx == 0 && warp_idx == 0 && e < kNumExpertsPerRank) {
                const uint32_t nr = l_num_recv[s];
                gemm2_meta.store_rows(e, nr,
                    math::ceil_div(nr, kGemm2BlockM) * kGemm2BlockM, l_gemm2_off[s],
                    math::ceil_div(nr, BLOCK_M) * BLOCK_M, l_pool_block_off[s] * BLOCK_M);
                // Count each local physical expert once, before per-call counts
                // are recycled. Atomics allow a caller to share its accumulator.
                if (cumulative_local_expert_recv_stats != nullptr)
                    tix::red_add(reinterpret_cast<uint32_t*>(cumulative_local_expert_recv_stats + e), nr);
            }
        }
    }

    // Convenience: read warp-distributed arr[e].
    const auto dget_nr  = [&](uint32_t e) { return dist_get<kPerLaneSlots>(l_num_recv, e); };
    const auto dget_off = [&](uint32_t e) { return dist_get<kPerLaneSlots>(l_pool_block_off, e); };

    // -------- Cooperative pull + scheduler GEMM --------
    // Specialized producers publish each block independently, then join the
    // consumers' tile queue after their own pulls. Shared memory is reused
    // for pull staging and GEMM.

    constexpr uint32_t kSfBytes = kTokenBytes / 16;
    constexpr uint32_t kSfCols  = kSfBytes / 2;
    constexpr uint32_t kBlockN = (kGemm1BlockN == 0) ? kGemm1N : kGemm1BlockN;
    constexpr uint32_t kBlockK = (kGemm1BlockK == 0) ? kTokenBytes : kGemm1BlockK;
    constexpr uint32_t kSfbBlockN =
        (kGemm1SfbBlockN == 0) ? kBlockN : kGemm1SfbBlockN;
    static_assert(kGemm1N % kBlockN == 0, "GEMM1_N must be divisible by GEMM1_BLOCK_N");
    static_assert(kGemm1N % kSfbBlockN == 0,
                  "GEMM1_N must be divisible by canonical SFB BLOCK_N");
    static_assert(kSfbBlockN % kBlockN == 0,
                  "canonical SFB BLOCK_N must be divisible by launch BLOCK_N");
    static_assert(kTokenBytes % kBlockK == 0, "TOKEN_BYTES must be divisible by GEMM1_BLOCK_K");
    static_assert(kBlockK % 32 == 0, "GEMM1_BLOCK_K must be a multiple of 32 (per-32 SF)");
    using Cfg = Gemm1MmaConfig<BLOCK_M, kBlockN, kBlockK,
                                    kGemm1WarpM, kGemm1WarpN, kGemm1Stages,
                                    kSfbBlockN>;
    static_assert(kNumThreads >= Cfg::ThreadNum,
                  "CTA must contain every gemm1 TiledMMA thread");
    static_assert(kNumThreads == Cfg::ThreadNum,
                  "extra producer warps do not support fused quantization");
    static_assert(!kDynamicWork || kNumThreads == Cfg::ThreadNum,
                  "the GEMM1 queue requires specialized pull and a full MMA CTA");
    // Each pull warp owns private token and scale staging.
    static_assert(kNumPullCTAs > 0 && kNumPullCTAs <= kNumCTAs,
                  "producer SM count must be in [1, NUM_SMS]");
    static_assert(kNumGemmCTAs > 0 && kNumGemmCTAs <= kNumCTAs,
                  "GEMM SM count must be in [1, NUM_SMS]");

    static_assert(kNumPullCTAs + kNumGemmCTAs == kNumCTAs,
                  "specialized producer and consumer SMs must cover the grid");
    static_assert(kPullSlots > 0 && kPullSlots <= kNumWarpsPerBlock,
                  "pull slots must fit in the CTA warp count");
    constexpr uint32_t kPullStagingBytes =
        kPullSlots * (kSfBytes + kTokenBytes);
    static_assert(kPullStagingBytes <= 262144u,
                  "pull staging exceeds the per-CU shared-memory capacity");
    // Handshake, pull, and GEMM use this per-CTA buffer in disjoint phases;
    // specialized producer and consumer CTAs also have private shared memory.
    auto* pull_sf = reinterpret_cast<uint8_t*>(smem_buf);
    auto* pull_tok = pull_sf + kPullSlots * kSfBytes;

    struct BlockInfo {
        uint32_t e, mb, valid_m, pool_base, epool_base, tok_lo, tok_hi;
    };

    // Helper: resolve g -> (e, mb, valid_m, pool_token_base, ...)
    const auto resolve_block = [&](uint32_t g) {
        uint32_t e = 0;
        uint32_t off_e = 0;
        uint32_t nr_e = 0;
        uint32_t local_e = 0;
        uint32_t local_off = 0;
        uint32_t local_nr = 0;
        bool local_hit = false;
        #pragma unroll
        for (uint32_t s = 0; s < kPerLaneSlots; ++s) {
            const uint32_t cand_e = lane_idx * kPerLaneSlots + s;
            if (cand_e < kNumExpertsPerRank) {
                const uint32_t cand_nr = l_num_recv[s];
                const uint32_t cand_off = l_pool_block_off[s];
                const uint32_t cand_nmb = math::ceil_div(cand_nr, BLOCK_M);
                const bool hit = (g >= cand_off) && (g < cand_off + cand_nmb);
                if (hit && !local_hit) {
                    local_hit = true;
                    local_e = cand_e;
                    local_off = cand_off;
                    local_nr = cand_nr;
                }
            }
        }
        const uint32_t hit_mask = __ballot_sync(0xffffffffu, local_hit);
        const int src_lane = hit_mask ? (__ffs(hit_mask) - 1) : 31;
        e = hit_mask ? __shfl_sync(0xffffffffu, local_e, src_lane) : (kNumExpertsPerRank - 1);
        off_e = hit_mask ? __shfl_sync(0xffffffffu, local_off, src_lane) : dget_off(e);
        nr_e = hit_mask ? __shfl_sync(0xffffffffu, local_nr, src_lane) : dget_nr(e);
        const uint32_t mb = g - off_e;
        // Guard against unsigned underflow when nr=0 or mb*BLOCK_M >= nr
        const uint32_t valid_m = (nr_e == 0 || mb * BLOCK_M >= nr_e) ? 0
            : umin(nr_e - mb * BLOCK_M, BLOCK_M);
        return BlockInfo{e, mb,
            valid_m,
            off_e * BLOCK_M + mb * BLOCK_M,
            off_e * BLOCK_M,
            mb * BLOCK_M,
            umin(nr_e, (mb + 1) * BLOCK_M)};
    };

    // Pull one M block cooperatively. Every producer CTA owns a contiguous
    // token slice; each active warp stages one token and its scales at a time.
    const auto coop_pull = [&] (
            uint32_t g, uint32_t pull_idx, uint32_t pull_count) {
        auto bi = resolve_block(g);
        uint16_t* sfa_base = sfa_scratch + static_cast<uint64_t>(g) * (BLOCK_M * kSfCols);

        // Cache this expert's per-rank token ranges once per warp and
        // M block. The first kNumRanks lanes
        // load the counts in parallel and build an inclusive prefix;
        // each token then resolves its source with one warp ballot.
        uint32_t src_count = lane_idx < kNumRanks
            ? static_cast<uint32_t>(
                  *workspace.get_expert_recv_count_ptr(lane_idx, bi.e))
            : 0;
        uint32_t src_end = src_count;
        #pragma unroll
        for (uint32_t offset = 1; offset < kNumRanks; offset <<= 1) {
            const uint32_t v = __shfl_up_sync(
                0xffffffffu, src_end, offset);
            if (lane_idx >= offset)
                src_end += v;
        }
        const uint32_t src_begin = src_end - src_count;

        const uint32_t base = bi.valid_m / pull_count;
        const uint32_t rem  = bi.valid_m % pull_count;
        const uint32_t my_start =
            pull_idx * base + umin(pull_idx, rem);
        const uint32_t my_count =
            base + (pull_idx < rem ? 1u : 0u);
        const uint32_t num_batches = math::ceil_div(my_count, kPullSlots);
        for (uint32_t batch = 0; batch < num_batches; ++batch) {
            const uint32_t t = my_start + batch * kPullSlots + warp_idx;
            if (warp_idx < kPullSlots && t < my_start + my_count) {
                const uint32_t actual_t = bi.tok_lo + t;
                const uint32_t mask = __ballot_sync(0xffffffffu,
                    lane_idx < kNumRanks && actual_t >= src_begin && actual_t < src_end);
                uint32_t r = 0, slot = 0, stk = 0;
                if (mask) {
                    r = static_cast<uint32_t>(__ffs(mask) - 1);
                    slot = actual_t - __shfl_sync(0xffffffffu, src_begin, r);
                }
                if (lane_idx == 0)
                    stk = *workspace.get_src_token_topk_idx_ptr(bi.e, r, slot);
                stk = __shfl_sync(0xffffffffu, stk, 0);
                const uint32_t pool_idx = bi.epool_base + actual_t;
                const auto* rt = sym_buffer.map(
                    input_token + static_cast<uint64_t>(stk/kNumTopk) * token_bytes, r);
                const auto* rsf = sym_buffer.map(
                    input_sf + static_cast<uint64_t>(stk/kNumTopk) * sf_bytes, r);
                auto* mt = pull_tok + warp_idx * token_bytes;
                auto* ms = pull_sf + warp_idx * sf_bytes;
                // Commit scales first so their transpose can overlap the token copy.
                for (uint32_t i = lane_idx; i < sf_bytes/16; i += 32)
                    cute::PPU_CP_ASYNC_CACHEALWAYS<uint4>::copy(
                        *reinterpret_cast<const uint4*>(rsf + i*16),
                        *reinterpret_cast<uint4*>(ms + i*16));
                cute::cp_async_fence();
                for (uint32_t i = lane_idx; i < token_bytes/16; i += 32)
                    cute::PPU_CP_ASYNC_CACHEALWAYS<uint4>::copy(
                        *reinterpret_cast<const uint4*>(rt + i*16),
                        *reinterpret_cast<uint4*>(mt + i*16));
                cute::cp_async_fence();
                asm volatile("" ::: "memory");
                uint32_t gemm2_pool_idx = 0;
                if constexpr (kGemm2BlockM != 0)
                    gemm2_pool_idx = dist_get<kPerLaneSlots>(l_gemm2_off, bi.e) + actual_t;
                asm volatile("" ::: "memory");
                cute::cp_async_wait<1>();
                __syncwarp();
                // Each scale column keeps the allocated BLOCK_M row stride.
                for (uint32_t j = lane_idx; j < kSfCols; j += 32)
                    sfa_base[j * BLOCK_M + t] = static_cast<uint16_t>(ms[2*j]) |
                        (static_cast<uint16_t>(ms[2*j+1]) << 8);
                cute::cp_async_wait<0>();
                __syncwarp();
                warp_copy_bytes(
                    dispatched_token_pool + static_cast<uint64_t>(pool_idx) * token_bytes,
                    mt, token_bytes, lane_idx);
                if (lane_idx == 0) {
                    *workspace.get_token_src_metadata_ptr(pool_idx) = {r, stk/kNumTopk, stk%kNumTopk};
                    if constexpr (kGemm2BlockM != 0)
                        gemm2_meta.store_route(gemm2_pool_idx, r, stk/kNumTopk, stk%kNumTopk);
                }
                __syncwarp();
            }
        }

    };

    // A small M block fits one producer; distribute those blocks among
    // producers instead of splitting a few rows across every producer CTA.
    // Larger blocks retain cooperative pulls through the same group loop.
    constexpr uint32_t kPullGroupSize = BLOCK_M <= 32 ? 1 : kNumPullCTAs;
    constexpr uint32_t kNumPullGroups = math::ceil_div(kNumPullCTAs, kPullGroupSize);

    if (cta_idx < kNumPullCTAs) {
        const uint32_t group_idx = cta_idx / kPullGroupSize;
        const uint32_t group_size = umin(kPullGroupSize, kNumPullCTAs - group_idx * kPullGroupSize);
        for (uint32_t g = group_idx; g < total_blocks; g += kNumPullGroups) {
            coop_pull(g, cta_idx % kPullGroupSize, group_size);

            // Publish only after this producer's token, SFA and metadata stores.
            __syncthreads();
            if (thread_idx == 0)
                tix::red_add_rel(
                    workspace.get_l1_arrival_count_ptr(g), 1u);

        }
    }

    constexpr uint32_t kGemmCTAOffset = kNumPullCTAs;
    constexpr uint32_t kNumNBlocks = kGemm1N / kBlockN;
    // Finish the N tiles of one M block before advancing M.
    constexpr uint32_t kMBlocksPerSwizzleGroup = 1;
    const uint32_t total_gemm_work = total_blocks * kNumNBlocks;
    const bool initial_gemm_cta =
        cta_idx >= kGemmCTAOffset && cta_idx < kGemmCTAOffset + kNumGemmCTAs;
    const uint32_t initial_gemm_cta_idx = cta_idx - kGemmCTAOffset;

    // All queue participants retain per-M-block readiness checks.
    // Non-queue configurations use static assignment.
    const auto run_gemm1 = [&] () {
        uint32_t current_iter = 0;
        uint32_t ready_base = ~uint32_t{0};
        uint32_t ready_mask = 0;
        uint32_t curr_group_idx = 0;
        uint32_t curr_group_block_off = 0;
        // Every participant claims one tile from the same queue.
        // Tickets increase monotonically, so the expert cursor is reusable.
        const mega_moe::WorkQueue queue{
            workspace.get_gemm1_work_count_ptr(), reinterpret_cast<uint32_t*>(smem_buf)};
        const auto fetch_next_gemm1_work = [&] (
                uint32_t& group_idx,
                uint32_t& group_block_off,
                uint32_t& m_block_idx,
                uint32_t& n_block_idx,
                uint32_t& group_m) {
            uint32_t next_block_idx;
            if constexpr (kDynamicWork) {

                queue.prime(thread_idx);
                next_block_idx = queue.current();
                // Retire ticket reads in the readiness rendezvous below,
                // before MMA can reuse this shared storage.
            } else {
                next_block_idx = (current_iter++) * kNumGemmCTAs + initial_gemm_cta_idx;
            }
            if (next_block_idx >= total_gemm_work)
                return false;
            while (curr_group_idx < kNumExpertsPerRank) {
                const uint32_t curr_group_m = dget_nr(curr_group_idx);
                const uint32_t curr_group_m_blocks =
                    math::ceil_div(curr_group_m, BLOCK_M);
                const uint32_t next_group_block_off =
                    curr_group_block_off + curr_group_m_blocks;
                if (next_block_idx < next_group_block_off * kNumNBlocks) {
                    const uint32_t block_idx_in_group =
                        next_block_idx - curr_group_block_off * kNumNBlocks;
                    const uint32_t blocks_per_swizzle_group =
                        kNumNBlocks * kMBlocksPerSwizzleGroup;
                    const uint32_t swizzle_group_idx =
                        block_idx_in_group / blocks_per_swizzle_group;
                    const uint32_t first_m_block =
                        swizzle_group_idx * kMBlocksPerSwizzleGroup;
                    const uint32_t in_swizzle_group_idx =
                        block_idx_in_group % blocks_per_swizzle_group;
                    const uint32_t m_blocks_in_swizzle_group = umin(
                        kMBlocksPerSwizzleGroup,
                        curr_group_m_blocks - first_m_block);
                    group_idx = curr_group_idx;
                    group_block_off = curr_group_block_off;
                    m_block_idx = first_m_block +
                        in_swizzle_group_idx % m_blocks_in_swizzle_group;
                    n_block_idx =
                        in_swizzle_group_idx / m_blocks_in_swizzle_group;
                    group_m = curr_group_m;
                    return true;
                }
                curr_group_idx++;
                curr_group_block_off = next_group_block_off;
            }
            return false;
        };

        uint32_t group_idx, group_block_off, m_block_idx, n_block_idx, group_m;
        while (fetch_next_gemm1_work(
                group_idx, group_block_off, m_block_idx, n_block_idx, group_m)) {
            const uint32_t g = group_block_off + m_block_idx;
            const uint32_t valid_m =
                (group_m == 0 || m_block_idx * BLOCK_M >= group_m) ? 0
                : umin(group_m - m_block_idx * BLOCK_M, BLOCK_M);
            const BlockInfo bi{
                group_idx, m_block_idx, valid_m,
                group_block_off * BLOCK_M + m_block_idx * BLOCK_M,
                group_block_off * BLOCK_M,
                m_block_idx * BLOCK_M,
                umin(group_m, (m_block_idx + 1) * BLOCK_M)};

            const uint32_t group_size = umin(
                kPullGroupSize, kNumPullCTAs - (g % kNumPullGroups) * kPullGroupSize);
            // One warp observes producer releases for the whole CTA.
            // Its cache is kernel-local; all consumers join the same block
            // barrier, also retiring reads of the shared work ticket.
            if (warp_idx == 0) {
                const uint32_t window_base = g & ~uint32_t{31};
                const uint32_t ready_bit = 1u << (g & 31u);
                if (window_base != ready_base || !(ready_mask & ready_bit)) {
                    const uint32_t peer_block = window_base + lane_idx;
                    const uint32_t peer_group_size = umin(kPullGroupSize,
                        kNumPullCTAs - (peer_block % kNumPullGroups) * kPullGroupSize);
                    const bool ready = peer_block < total_blocks &&
                        tix::ld_acq(workspace.get_l1_arrival_count_ptr(peer_block)) >= peer_group_size;
                    ready_mask = __ballot_sync(0xffffffffu, ready);
                    // Carry each lane's acquire to the warp before caching readiness.
                    __syncwarp();
                    ready_base = window_base;
                    if (!(ready_mask & ready_bit)) {
                        while (tix::ld_acq(workspace.get_l1_arrival_count_ptr(g)) < group_size);
                        ready_mask |= ready_bit;
                    }
                }
            }
            __syncthreads();

            const bool is_gate = (n_block_idx * kBlockN < kGemm1N / 2);
            // Store one canonical SFB layout (normally BN256).  A
            // smaller launch tile loads the full contiguous SFB tile
            // and selects its half from shared memory. This keeps the
            // persistent layout unique without strided global loads.
            const bool use_replica = bi.e >= num_master_experts;
            const uint32_t weight_expert =
                use_replica ? bi.e - num_master_experts : bi.e;
            const uint8_t* weight_base =
                use_replica ? gemm1_weight_replicas : gemm1_weight;
            const uint16_t* weight_sf_u16_base =
                use_replica
                    ? gemm1_weight_sf_u16_replicas
                    : gemm1_weight_sf_u16;
            constexpr uint32_t kLaunchTilesPerSfbTile =
                kSfbBlockN / kBlockN;
            const uint32_t sfb_tile_idx =
                n_block_idx / kLaunchTilesPerSfbTile;
            const uint32_t sfb_subtile_idx =
                n_block_idx % kLaunchTilesPerSfbTile;
            const uint16_t* gemm1_sfb_tile =
                weight_sf_u16_base
                + (uint64_t)weight_expert * kGemm1N * kSfCols
                + (uint64_t)sfb_tile_idx * kSfbBlockN * kSfCols;
            gemm1_mma_block<
                Cfg, true, false, kFusedSitu, kSiluThenClamp, kFastMath>(
                dispatched_token_pool + (uint64_t)bi.pool_base * token_bytes,
                sfa_scratch + (uint64_t)g * (BLOCK_M * kSfCols),
                weight_base
                    + (uint64_t)weight_expert * kGemm1N * kTokenBytes
                    + (uint64_t)n_block_idx * kBlockN * kTokenBytes,
                gemm1_sfb_tile,
                static_cast<int>(kSfbBlockN),
                static_cast<int>(sfb_subtile_idx * kBlockN),
                nullptr,
                0,
                static_cast<int>(bi.valid_m),
                static_cast<int>(BLOCK_M), static_cast<int>(kBlockN),
                static_cast<int>(kTokenBytes),
                thread_idx, reinterpret_cast<char*>(smem_buf),
                activation_clamp, is_gate, fused_swiglu, fused_situ,
                situ_beta, situ_linear_beta,
                fp4_output
                    ? fp4_output + (uint64_t)bi.pool_base * (kGemm1N / 4)
                        + (uint64_t)n_block_idx * (kBlockN / 4)
                    : nullptr,
                scale_output
                    ? scale_output
                        + ((uint64_t)n_block_idx * (kBlockN / 128)
                           * workspace.num_max_pool_tokens
                           + bi.pool_base) * sizeof(uint16_t)
                    : nullptr,
                static_cast<int>(kGemm1N / 4),
                static_cast<int>(workspace.num_max_pool_tokens
                                 * sizeof(uint16_t)));
        }
    };

    if (initial_gemm_cta || kDynamicWork) {
        if (thread_idx < Cfg::ThreadNum)
            run_gemm1();
        __syncthreads();
    }

}

} // namespace deep_gemm::mega_moe::fp4
