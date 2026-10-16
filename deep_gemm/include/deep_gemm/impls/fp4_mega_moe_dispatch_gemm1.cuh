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
//   Phase C  -- tail handoff:
//       Producers rendezvous after the last pull while consumers finish the
//       prefix. The full grid then executes the reserved tail GEMM range, so
//       producer SMs do not remain idle after communication completes.
//
// Data format: PPU mxfp4 (per-32 scale, uint16 SF). The dispatch pull is byte-
// wise / format-agnostic (token payload = `token_bytes`, SF = `sf_bytes`), so
// routing can be validated independently of the FP4 details (gemm1 is gated by
// `kEnableGemm1`).
//
// All cross-rank "lock" primitives are the PPU-verified `.sys` PTX wrappers
// (see tests/test_ppu_sys_scope.py and ptx/sys_atomics.cuh).
// ============================================================================

#include <cstdint>
#include <hggc_runtime.h>
#include <hggc_bf16.h>
#include <deep_gemm/layout/mega_moe_common.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/impls/fp4_mega_moe_gemm1_mma.cuh>
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>

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
    uint32_t kNumExperts, uint32_t kNumExpertsPerRank, uint32_t kNumTopk,
    bool kWritePullIndices = true,
    bool kWritePushSlots = false
>
CUTLASS_DEVICE void mega_moe_dispatch_handshake(
        const layout::Workspace& workspace,
        const layout::SymBuffer<kNumRanks>& sym_buffer,
        const int64_t* __restrict__ input_topk_idx,   // [num_tokens, kNumTopk]
        uint32_t* __restrict__ smem_expert_count,      // [kNumExperts]
        const uint32_t& num_tokens,
        const uint32_t& cta_idx,
        const uint32_t& thread_idx) {

    const auto block_sync = [=]() { __syncthreads(); };

    // ---- 0) Zero shared expert count ----
    for (uint32_t i = thread_idx; i < kNumExperts; i += kNumThreads)
        smem_expert_count[i] = 0;
    __syncthreads();

    // Iterate this rank's (token, topk) pairs assigned to this thread, calling
    // `process(token_topk_idx, expert_idx)` for each valid routed pair.
    const auto for_each_routed = [&](const auto& process) {
        const uint32_t total = num_tokens * kNumTopk;
        for (uint32_t p = cta_idx * kNumThreads + thread_idx; p < total;
             p += kNumCTAs * kNumThreads) {
            const int expert_idx = static_cast<int>(__ldg(input_topk_idx + p));
            if (expert_idx >= 0 and expert_idx < static_cast<int>(kNumExperts))
                process(p, static_cast<uint32_t>(expert_idx));
        }
    };

    // ---- 1) Count tokens per (global) expert into shared ----
    for_each_routed([&](const uint32_t& /*ttid*/, const uint32_t& expert_idx) {
        atomicAdd_block(smem_expert_count + expert_idx, 1u);
    });
    __syncthreads();

    // ---- 2) Reserve a contiguous send-slot range per expert across all SMs ----
    // expert_send_count packs: high 32 bits = #SMs contributed, low 32 = #tokens.
    for (uint32_t i = thread_idx; i < kNumExperts; i += kNumThreads) {
        const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
        // old low-32 = this SM's starting slot for the expert
        const uint64_t old_val = ptx::atomic_add_sys(workspace.get_expert_send_count_ptr(i), send_value);
        smem_expert_count[i] = static_cast<uint32_t>(old_val & 0xffffffffu);
    }
    __syncthreads();

    // ---- 3) Write source token-topk indices into the DEST rank's workspace ----
    // Pull dispatch resolves source rows from this list. Push dispatch already
    // owns the local source row and can skip the extra remote metadata write.
    if constexpr (kWritePullIndices) {
        for_each_routed([&](const uint32_t& token_topk_idx, const uint32_t& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto local_expert_idx = expert_idx % kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1u);
            auto* dst_ptr = workspace.get_src_token_topk_idx_ptr(
                local_expert_idx, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });
    } else if constexpr (kWritePushSlots) {
        // Preserve the source-side slot selected by the same per-CTA prefix
        // protocol. Push dispatch consumes it later without a hot-loop global
        // atomic. The L2 mask region is otherwise unused by the BF16 path and
        // is capacity-sized for every routed pair.
        for_each_routed([&](const uint32_t& token_topk_idx, const uint32_t& expert_idx) {
            const auto dst_slot_idx =
                atomicAdd_block(smem_expert_count + expert_idx, 1u);
            *workspace.get_l2_arrival_mask_ptr(token_topk_idx) = dst_slot_idx;
        });
    }

    // ---- 4) Grid sync so every SM finished writing send counts / src indices ----
    // Stagger polling across SMs. Without backoff, rare launches can spend
    // milliseconds hammering this single counter cache line before SM0 flips
    // the generation bit, even though the routed-index work is already done.
    comm::grid_sync<kNumCTAs, /*kGridSyncIndex=*/0,
                    /*kPollBackoffCycles=*/128>(
        workspace, cta_idx, thread_idx, block_sync);

    // ---- 5) SM0 publishes per-(src_rank, local_expert) recv counts to dest ----
    if (cta_idx == 0) {
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumThreads) {
            const auto dst_rank_idx = i / kNumExpertsPerRank;
            const auto local_expert_idx = i % kNumExpertsPerRank;
            const uint64_t expert_status = *workspace.get_expert_send_count_ptr(i);
            // recv_count[this_rank][local_expert] on dest rank = #tokens we send
            *sym_buffer.map(
                workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, local_expert_idx),
                dst_rank_idx) = expert_status & 0xffffffffull;
            // recv_count_sum[local_expert] on dest rank += packed(#SMs, #tokens)
            ptx::atomic_add_sys(
                sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(local_expert_idx), dst_rank_idx),
                expert_status);
        }
    }
    __syncthreads();

    // ---- 6) Cross-rank barrier: all ranks done publishing counts/indices ----
    comm::cross_rank_barrier<kNumRanks, kNumCTAs, kNumThreads, /*idx=*/0, /*tag=*/0>(
        workspace, sym_buffer, cta_idx, thread_idx, block_sync, true, true);
}

// ----------------------------------------------------------------------------
// Fused kernel entry.
//
// Phase A: dispatch handshake. Phase B: overlapped pull/prefix GEMM.
// Phase C: optional full-grid tail GEMM. The compile-time probes can still
// isolate communication or compute for performance analysis.
// ----------------------------------------------------------------------------
template <
    uint32_t kNumRanks, uint32_t kNumCTAs, uint32_t kNumThreads,
    uint32_t kNumExperts, uint32_t kNumExpertsPerRank, uint32_t kNumTopk,
    uint32_t BLOCK_M,
    uint32_t kChunkBlocks = kNumCTAs,
    uint32_t kNumPullCTAs = kNumCTAs,
    uint32_t kNumGemmCTAs = kNumCTAs,
    bool kSmSpecialization = false,
    uint32_t kPullSlots = 8,
    uint32_t kTailGemmBlocks = 0,
    uint32_t kTailGemmPercent = 0,
    bool kEnableGemm1 = false,
    uint32_t kGemm1N = 0,
    uint32_t kTokenBytes = 0,
    uint32_t kGemm1BlockN = 0,  // gemm1 N-tile; 0 => full N (single tile, legacy)
    uint32_t kGemm1BlockK = 0,  // gemm1 K-tile; 0 => full K (single tile, legacy)
    uint32_t kGemm1SfbBlockN = kGemm1BlockN,  // physical canonical SFB N-tile
    bool kEnableRemotePull = true,  // false isolates GEMM1 from dispatch traffic
    bool kEnableRemotePush = false, // BF16 source rank pushes rows with ICN stores
    uint32_t kGemm1WarpM = 32,  // gemm1 TiledMMA warp tile (M). Must divide BLOCK_M.
    uint32_t kGemm1WarpN = 64,  // gemm1 TiledMMA warp tile (N). Must divide kBlockN.
    uint32_t kGemm1Stages = 2,  // gemm1 mainloop pipeline stages (CUTLASS smem stages).
    bool kQuantizeOutput = false, // compile-time: fuse MXFP4 quantization into epilogue
    bool kSituTanhBf16x2 = false, // compile-time: packed BF16x2 SiTU tanh
    bool kFusedSitu = false, // compile-time: eliminate inactive GLU epilogue
    bool kSiluThenClamp = false
>
// NOTE: the kernel body lives in this `__device__` impl (NOT the `__global__`)
// so it can be reused by two launch paths: the Python JIT launches the thin
// `__global__` template shim below directly, while the C++ JIT path generates an
// `extern "C" __global__` wrapper (it needs an unmangled symbol name for the
// exact-name `cuModuleGetFunction` lookup, and a template `__global__` cannot be
// `extern "C"` nor called from another `__global__`).
__device__ void
dispatch_gemm1_body(
        void* workspace_base,
        const layout::SymBuffer<kNumRanks> sym_buffer,
        const int64_t* __restrict__ input_topk_idx,
        const float* __restrict__ input_topk_weights,
        const uint8_t* __restrict__ input_token,
        const uint8_t* __restrict__ input_sf,
        uint8_t* __restrict__ dispatched_token_pool,
        uint8_t* __restrict__ dispatched_scale_pool,
        float* __restrict__ dispatched_weights_pool,
        const uint32_t num_tokens,
        const uint32_t num_max_tokens_per_rank,
        const uint32_t token_bytes,
        const uint32_t sf_bytes,
        const uint8_t* __restrict__ gemm1_weight,     // [EPR, N, K/2] packed FP4
        const uint8_t* __restrict__ gemm1_weight_sf,  // [EPR, N, K/32] uint8 SF
        __ppu_bfloat16* __restrict__ gemm1_output,
        const uint32_t gemm1_output_stride,
        const uint16_t* __restrict__ gemm1_weight_sf_u16,  // [EPR, N, K/32/2] uint16 N-major (TC)
        const uint8_t* __restrict__ gemm1_weight_replicas,
        const uint16_t* __restrict__ gemm1_weight_sf_u16_replicas,
        const uint32_t num_master_experts,
        uint16_t* __restrict__ sfa_scratch,             // [total_blocks, BLOCK_M*K/32/2] uint16 (TC)
        float activation_clamp = 0.0f,                    // >0: clamp; 0: no clamp
        bool fused_swiglu = false,                        // fuse SwiGLU into epilogue
        bool fused_situ = false,                          // fuse Kimi K3 SiTU-GLU into epilogue
        float situ_beta = 4.0f,                           // gate tanh saturation
        float situ_linear_beta = 25.0f,                   // up-branch tanh saturation
        uint8_t* __restrict__ fp4_output = nullptr,       // [num_pool, N/4] packed FP4
        uint8_t* __restrict__ scale_output = nullptr,     // byte view of packed/transposed uint16 E8M0 scales
        int rank = 0,                                      // rank index
        uint32_t pull_begin_block = 0,
        uint32_t pull_end_block = 0xffffffffu,
        bool resume_dispatch = false,
        bool finalize_dispatch = true,
        bool synchronize_pull = true,
        bool publish_arrivals = false) {

    const uint32_t cta_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = thread_idx / 32;
    const uint32_t lane_idx   = thread_idx % 32;

    constexpr uint32_t kNumWarpsPerBlock = kNumThreads / 32;
    constexpr uint32_t kNumGlobalWarps   = kNumCTAs * kNumWarpsPerBlock;
    const uint32_t global_warp_idx       = cta_idx * kNumWarpsPerBlock + warp_idx;

    // Reconstruct the architecture-independent workspace view.
    const layout::Workspace workspace(
        workspace_base, kNumRanks, kNumExperts, num_max_tokens_per_rank, kNumTopk);

    extern __shared__ __align__(16) unsigned char smem_buf[];
    auto* smem_expert_count = reinterpret_cast<uint32_t*>(smem_buf);  // [kNumExperts]
    const auto block_sync = [=]() { __syncthreads(); };

    // ==================== Phase A: dispatch handshake ====================
    if (!resume_dispatch) {
        mega_moe_dispatch_handshake<
            kNumRanks, kNumCTAs, kNumThreads, kNumExperts,
            kNumExpertsPerRank, kNumTopk,
            /*kWritePullIndices=*/!kEnableRemotePush,
            /*kWritePushSlots=*/kEnableRemotePush>(
            workspace, sym_buffer, input_topk_idx, smem_expert_count,
            num_tokens, cta_idx, thread_idx);
    }

    // ==================== Optional source-push dispatch ====================
    // This direction matches the efficient G2 scatter path: source warps read
    // their local contiguous token row and issue posted ICN stores directly
    // into the destination expert landing pool.  The pull-only source-index
    // list is omitted by the handshake specialization above.
    if constexpr (kEnableRemotePush) {
        static_assert(!kEnableRemotePull,
                      "push and pull dispatch are mutually exclusive");
        static_assert(!kEnableGemm1,
                      "source-push dispatch runs in dispatch-only mode");

        // The destination publishes its expert-padded row offsets in the
        // temporary L1-arrival prefix. Source slot indices were already
        // materialized per routed pair by the handshake.
        if (cta_idx == 0) {
            if (thread_idx == 0) {
                uint32_t pool_row_offset = 0;
                for (uint32_t e = 0; e < kNumExpertsPerRank; ++e) {
                    *workspace.get_l1_arrival_count_ptr(e) = pool_row_offset;
                    const uint32_t rows = static_cast<uint32_t>(
                        *workspace.get_expert_recv_count_sum_ptr(e));
                    pool_row_offset += math::ceil_div(rows, BLOCK_M) * BLOCK_M;
                }
            }
        }

        // All ranks must publish pool offsets before sources map remote
        // landing addresses.
        comm::cross_rank_barrier<
            kNumRanks, kNumCTAs, kNumThreads,
            /*kGridSyncIndex=*/2, /*kTag=*/1>(
            workspace, sym_buffer, cta_idx, thread_idx, block_sync, true, true);

        // Precompute one final destination row base per global expert:
        //   padded expert pool base + rows contributed by lower source ranks.
        // This removes both the system atomic and remote offset load from the
        // per-route hot loop.
        if (cta_idx == 0) {
            for (uint32_t expert_idx = thread_idx;
                 expert_idx < kNumExperts; expert_idx += kNumThreads) {
                const uint32_t dst_rank_idx =
                    expert_idx / kNumExpertsPerRank;
                const uint32_t local_expert_idx =
                    expert_idx % kNumExpertsPerRank;
                uint32_t src_prefix = 0;
                for (uint32_t src_rank_idx = 0;
                     src_rank_idx < sym_buffer.rank_idx; ++src_rank_idx) {
                    const auto* remote_count = sym_buffer.map(
                        workspace.get_expert_recv_count_ptr(
                            src_rank_idx, local_expert_idx),
                        dst_rank_idx);
                    src_prefix += static_cast<uint32_t>(
                        ptx::ld_acq_sys(remote_count));
                }
                const auto* remote_pool_offset = sym_buffer.map(
                    workspace.get_l1_arrival_count_ptr(local_expert_idx),
                    dst_rank_idx);
                const uint32_t final_base = src_prefix +
                    ptx::ld_acq_sys(remote_pool_offset);
                *workspace.get_expert_send_count_ptr(expert_idx) =
                    static_cast<uint64_t>(final_base) << 32;
            }
        }

        comm::grid_sync<kNumCTAs, /*kGridSyncIndex=*/3>(
            workspace, cta_idx, thread_idx, block_sync);

        const uint32_t total_routes = num_tokens * kNumTopk;
        for (uint32_t p = global_warp_idx; p < total_routes;
             p += kNumGlobalWarps) {
            const int expert_value = static_cast<int>(__ldg(input_topk_idx + p));
            if (expert_value < 0 || expert_value >= static_cast<int>(kNumExperts))
                continue;
            const uint32_t expert_idx = static_cast<uint32_t>(expert_value);
            const uint32_t dst_rank_idx = expert_idx / kNumExpertsPerRank;

            const uint32_t final_base = static_cast<uint32_t>(
                *workspace.get_expert_send_count_ptr(expert_idx) >> 32);
            const uint32_t source_slot = static_cast<uint32_t>(
                *workspace.get_l2_arrival_mask_ptr(p));
            const uint32_t pool_idx = final_base + source_slot;

            const auto* src_token = input_token
                + static_cast<uint64_t>(p / kNumTopk) * token_bytes;
            auto* local_dst_token = dispatched_token_pool
                + static_cast<uint64_t>(pool_idx) * token_bytes;
            auto* dst_token = sym_buffer.map(local_dst_token, dst_rank_idx);
            constexpr uint32_t kTokenChunks = kTokenBytes / sizeof(uint4);
            for (uint32_t i = lane_idx; i < kTokenChunks; i += 64) {
                const uint4 payload0 = __ppu_global_load_bulk_b32x4(
                    src_token + i * sizeof(uint4));
                uint4 payload1{};
                if (i + 32 < kTokenChunks) {
                    payload1 = __ppu_global_load_bulk_b32x4(
                        src_token + (i + 32) * sizeof(uint4));
                }
                if (dst_rank_idx == sym_buffer.rank_idx) {
                    __ppu_global_store_bulk_volatile_b32x4(
                        reinterpret_cast<volatile void*>(
                            dst_token + i * sizeof(uint4)),
                        payload0);
                    if (i + 32 < kTokenChunks) {
                        __ppu_global_store_bulk_volatile_b32x4(
                            reinterpret_cast<volatile void*>(
                                dst_token + (i + 32) * sizeof(uint4)),
                            payload1);
                    }
                } else {
                    __ppu_remote_store_bulk_volatile_b32x4(
                        reinterpret_cast<volatile void*>(
                            dst_token + i * sizeof(uint4)),
                        payload0);
                    if (i + 32 < kTokenChunks) {
                        __ppu_remote_store_bulk_volatile_b32x4(
                            reinterpret_cast<volatile void*>(
                                dst_token + (i + 32) * sizeof(uint4)),
                            payload1);
                    }
                }
            }

            if (lane_idx == 0) {
                const float topk_weight = __ldg(input_topk_weights + p);
                auto* dst_weight = sym_buffer.map(
                    dispatched_weights_pool + pool_idx, dst_rank_idx);
                auto* dst_meta = sym_buffer.map(
                    workspace.get_token_src_metadata_ptr(pool_idx),
                    dst_rank_idx);
                if (dst_rank_idx == sym_buffer.rank_idx) {
                    *dst_weight = topk_weight;
                    *dst_meta = {
                        sym_buffer.rank_idx, p / kNumTopk, p % kNumTopk};
                } else {
                    __ppu_remote_store_bulk_ncp_b32(
                        dst_weight, __float_as_uint(topk_weight));
                    __ppu_remote_store_bulk_ncp_b32(
                        &dst_meta->rank_idx, sym_buffer.rank_idx);
                    __ppu_remote_store_bulk_ncp_b32(
                        &dst_meta->token_idx, p / kNumTopk);
                    __ppu_remote_store_bulk_ncp_b32(
                        &dst_meta->topk_idx, p % kNumTopk);
                }
            }
            __syncwarp();
        }

        // Preserve the posted-store advantage: allow every warp to enqueue all
        // of its routes, then drain vector/scalar remote stores only once before
        // publishing completion to peer ranks.
        __ppu_wait_remote_vstcnt(0);
        if (lane_idx == 0)
            __ppu_wait_remote_sstcnt(0);
        __syncwarp();

        // The per-warp remote waits above establish payload completion. This
        // barrier makes every destination wait until all source CTAs/ranks have
        // completed before a following GEMM kernel consumes the landing pool.
        comm::cross_rank_barrier<
            kNumRanks, kNumCTAs, kNumThreads,
            /*kGridSyncIndex=*/2, /*kTag=*/2>(
            workspace, sym_buffer, cta_idx, thread_idx, block_sync, true, true);

        comm::grid_sync<kNumCTAs, /*kGridSyncIndex=*/1>(
            workspace, cta_idx, thread_idx, block_sync);
        const uint32_t flat = cta_idx * kNumThreads + thread_idx;
        const uint32_t stride = kNumCTAs * kNumThreads;
        for (uint32_t i = flat; i < kNumExperts; i += stride)
            *workspace.get_expert_send_count_ptr(i) = 0;
        return;
    }

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

    // Convenience: read warp-distributed arr[e].
    const auto dget_nr  = [&](uint32_t e) { return dist_get<kPerLaneSlots>(l_num_recv, e); };
    const auto dget_off = [&](uint32_t e) { return dist_get<kPerLaneSlots>(l_pool_block_off, e); };
    const auto dget_nmb = [&](uint32_t e) { return math::ceil_div(dget_nr(e), BLOCK_M); };

    // -------- Cooperative pull + scheduler GEMM --------
    // Specialized producer CTAs publish each block independently while
    // consumer CTAs run the prefix. A tail grid handoff reclaims all SMs for
    // the final GEMM range. Shared memory is reused for pull staging and GEMM.
    if constexpr (kEnableGemm1 || kEnableRemotePull) {
        // Dispatch-only callers may use the token pool as an opaque byte
        // transport (for example BF16 activations feeding a separate W4A16
        // grouped GEMM). In that mode there are no MXFP4 activation scales.
        constexpr uint32_t kSfBytes = kEnableGemm1 ? kTokenBytes / 16 : 0;
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
        static_assert(kNumThreads == Cfg::ThreadNum || !kQuantizeOutput,
                      "extra producer warps do not support fused quantization");

        // Each active pull warp owns one private staging slot. A warp reuses
        // that slot sequentially when a producer owns more than kPullSlots tokens.
        static_assert(kNumPullCTAs > 0 && kNumPullCTAs <= kNumCTAs,
                      "producer SM count must be in [1, NUM_SMS]");
        static_assert(kNumGemmCTAs > 0 && kNumGemmCTAs <= kNumCTAs,
                      "GEMM SM count must be in [1, NUM_SMS]");
        static_assert(!kSmSpecialization || (kEnableRemotePull || kEnableGemm1),
                      "SM specialization requires remote pull or GEMM1");
        static_assert(!kSmSpecialization || kNumPullCTAs + kNumGemmCTAs == kNumCTAs,
                      "specialized producer and consumer SMs must cover the grid");
        static_assert(kTailGemmPercent <= 100,
                      "tail GEMM percentage must be in [0, 100]");
        static_assert((kTailGemmBlocks == 0 && kTailGemmPercent == 0) ||
                          (kSmSpecialization && kEnableRemotePull && kEnableGemm1),
                      "tail GEMM stealing requires specialized pull+GEMM");
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

        // Arrival counters are persistent workspace state.  The full grid
        // clears the active prefix before specialized producer/consumer CTAs
        // diverge, then rendezvous once so no consumer can observe stale ready
        // counts from a previous launch.
        if constexpr (kSmSpecialization) {
            for (uint32_t g = cta_idx * kNumThreads + thread_idx; g < total_blocks;
                 g += kNumCTAs * kNumThreads)
                *workspace.get_l1_arrival_count_ptr(g) = 0;
            comm::grid_sync<kNumCTAs, /*kGridSyncIndex=*/2>(
                workspace, cta_idx, thread_idx, block_sync);
        }
        // Pull one M block cooperatively. Every producer CTA owns a contiguous
        // token slice; its active warps stage one token each with cp.async.
        const auto coop_pull = [&] (
                uint32_t g, uint32_t pull_idx, uint32_t pull_count) {
            auto bi = resolve_block(g);
            uint16_t* sfa_base = sfa_scratch + static_cast<uint64_t>(g) * (BLOCK_M * kSfCols);
            if constexpr (kEnableRemotePull) {
                // Cache this expert's per-rank token ranges once per warp and
                // M block.  The old path made lane 0 serially reload up to all
                // rank counts for every token.  Here the first kNumRanks lanes
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
                        uint32_t r = 0, slot = 0, stk = 0;
                        const uint32_t src_mask = __ballot_sync(
                            0xffffffffu,
                            lane_idx < kNumRanks &&
                                actual_t >= src_begin && actual_t < src_end);
                        if (src_mask) {
                            r = static_cast<uint32_t>(__ffs(src_mask) - 1);
                            const uint32_t range_begin = __shfl_sync(
                                0xffffffffu, src_begin, r);
                            slot = actual_t - range_begin;
                        }
                        if (lane_idx == 0)
                            stk = *workspace.get_src_token_topk_idx_ptr(bi.e, r, slot);
                        stk = __shfl_sync(0xffffffffu, stk, 0);
                        const auto* rt = sym_buffer.map(
                            input_token + (uint64_t)(stk/kNumTopk)*token_bytes, r);
                        const auto* rsf = sym_buffer.map(
                            input_sf + (uint64_t)(stk/kNumTopk)*sf_bytes, r);
                        float topk_weight = 0.0f;
                        if (lane_idx == 0)
                            topk_weight = *sym_buffer.map(input_topk_weights + stk, r);
                        const uint32_t pool_idx = bi.epool_base + actual_t;
                        auto* dst_token = dispatched_token_pool
                            + static_cast<uint64_t>(pool_idx) * token_bytes;
                        uint8_t* mt = pull_tok + warp_idx * token_bytes;
                        uint8_t* ms = pull_sf + warp_idx * sf_bytes;
                        for (uint32_t i = lane_idx; i < token_bytes/16; i += 32)
                            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                                :: "r"((unsigned)__cvta_generic_to_shared(mt+i*16)),
                                   "l"(rt+i*16));
                        for (uint32_t i = lane_idx; i < sf_bytes/16; i += 32)
                            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                                :: "r"((unsigned)__cvta_generic_to_shared(ms+i*16)),
                                   "l"(rsf+i*16));
                        asm volatile("cp.async.commit_group;\n" ::);
                        asm volatile("" ::: "memory");
                        asm volatile("cp.async.wait_all;\n" ::);
                        __syncwarp();
                        warp_copy_bytes(
                            dst_token, mt, token_bytes, lane_idx);
                        for (uint32_t j = lane_idx; j < kSfCols; j += 32) {
                            uint16_t v = (uint16_t)ms[2*j] | ((uint16_t)ms[2*j+1] << 8);
                            sfa_base[j*BLOCK_M + t] = v;
                        }
                        if (lane_idx == 0) {
                            dispatched_weights_pool[pool_idx] = topk_weight;
                            *workspace.get_token_src_metadata_ptr(pool_idx) =
                                {r, stk/kNumTopk, stk%kNumTopk};
                        }
                        __syncwarp();
                    }
                }
            }
        };

        if constexpr (kEnableRemotePull) {
            // Producer CTAs pull every block; specialized consumers can start
            // each block as soon as all producers publish its arrival count.
            if constexpr (!kEnableGemm1 && !kSmSpecialization) {
                // Give each producer an independent 64-row block. With all 39
                // CTAs on one block, each CTA owns only 1--2 rows and most of
                // its eight pull warps are idle. One CTA per block instead
                // pulls eight rows per batch and exposes many blocks to ICN.
                // A continuation launch may deliberately oversubscribe the 39
                // physical CUs with several lightweight pull CTAs per CU. The
                // remote-load bandwidth microbenchmark needs roughly 1.2k--
                // 2.5k resident warps to saturate ICN, much more than one
                // 256-thread CTA per CU provides. This path has no cooperative
                // grid barrier, so gridDim is the actual work distributor.
                const uint32_t pull_ctas = gridDim.x;
                const uint32_t pull_begin = umin(
                    pull_begin_block, total_blocks);
                const uint32_t pull_end = umin(
                    pull_end_block, total_blocks);
                for (uint32_t g = pull_begin + cta_idx; g < pull_end;
                     g += pull_ctas) {
                    coop_pull(g, 0, 1);
                    if (publish_arrivals) {
                        __syncthreads();
                        if (thread_idx == 0)
                            ptx::red_add_rel(
                                workspace.get_l1_arrival_count_ptr(g), 1u);
                        __syncthreads();
                    }
                }
            } else {
                if (cta_idx < kNumPullCTAs) {
                    for (uint32_t g = 0; g < total_blocks; ++g) {
                        coop_pull(g, cta_idx, kNumPullCTAs);
                        if constexpr (kSmSpecialization) {
                            // Release this block only after every producer warp has
                            // finished its token, SFA, weight, and metadata stores.
                            __syncthreads();
                            if (thread_idx == 0)
                                ptx::red_add_rel(
                                    workspace.get_l1_arrival_count_ptr(g), 1u);
                        }
                    }
                }
            }

            // Grid sync: ensure all SMs' pull writes are visible before GEMM.
            // Replaces per-block spin-wait barriers with 1 grid_sync.
            if constexpr (!kSmSpecialization) {
                if (synchronize_pull)
                    comm::grid_sync<kNumCTAs, /*kGridSyncIndex=*/2>(
                        workspace, cta_idx, thread_idx, block_sync);
            }
        }

        if constexpr (kEnableGemm1) {
            constexpr uint32_t kGemmSMOffset = kSmSpecialization ? kNumPullCTAs : 0;
            constexpr uint32_t kNumNBlocks = kGemm1N / kBlockN;
            constexpr uint32_t kMBlocksPerSwizzleGroup = 2;
            const uint32_t total_gemm_work = total_blocks * kNumNBlocks;
            const uint32_t percent_tail_m_blocks = kTailGemmPercent > 0
                ? umin(
                      total_blocks,
                      math::ceil_div(total_blocks * kTailGemmPercent, 100u))
                : 0;
            uint32_t tail_m_blocks = kTailGemmPercent > 0
                ? (kTailGemmBlocks > 0
                       ? umin(percent_tail_m_blocks, kTailGemmBlocks)
                       : percent_tail_m_blocks)
                : umin(total_blocks, kTailGemmBlocks);
            // The percentage-derived split can leave one partially occupied
            // scheduler wave on each side of the rendezvous. Starting from the
            // existing split, move only toward the configured cap and only
            // when the idealized wave count strictly decreases. Compute once
            // per warp and broadcast to avoid doing the tiny search per lane.
            if constexpr (kSmSpecialization && kTailGemmBlocks > 0) {
                const uint32_t max_tail_m_blocks =
                    umin(total_blocks, kTailGemmBlocks);
                if (tail_m_blocks < max_tail_m_blocks) {
                    if ((thread_idx & 31u) == 0) {
                        uint32_t best_tail_m_blocks = tail_m_blocks;
                        uint32_t best_waves =
                            math::ceil_div(
                                (total_blocks - tail_m_blocks) * kNumNBlocks,
                                kNumGemmCTAs) +
                            math::ceil_div(
                                tail_m_blocks * kNumNBlocks, kNumCTAs);
                        for (uint32_t candidate = tail_m_blocks + 1;
                             candidate <= max_tail_m_blocks; ++candidate) {
                            const uint32_t candidate_waves =
                                math::ceil_div(
                                    (total_blocks - candidate) * kNumNBlocks,
                                    kNumGemmCTAs) +
                                math::ceil_div(candidate * kNumNBlocks, kNumCTAs);
                            if (candidate_waves < best_waves) {
                                best_waves = candidate_waves;
                                best_tail_m_blocks = candidate;
                            }
                        }
                        tail_m_blocks = best_tail_m_blocks;
                    }
                    tail_m_blocks = __shfl_sync(0xffffffffu, tail_m_blocks, 0);
                }
            }
            const uint32_t tail_work_begin =
                (total_blocks - tail_m_blocks) * kNumNBlocks;
            const bool initial_gemm_sm =
                cta_idx >= kGemmSMOffset && cta_idx < kGemmSMOffset + kNumGemmCTAs;
            const uint32_t initial_gemm_sm_idx = cta_idx - kGemmSMOffset;

            // Execute a contiguous range of the swizzled grouped-GEMM work list.
            // The initial range uses only consumer SMs and waits per M block;
            // after all producers finish, the tail range can use all SMs.
            const auto run_gemm_range = [&] (
                    uint32_t work_begin, uint32_t work_end,
                    uint32_t scheduler_sms, uint32_t scheduler_sm_idx,
                    bool wait_for_arrival) {
                uint32_t current_iter = 0;
                uint32_t curr_group_idx = 0;
                uint32_t curr_group_block_off = 0;
                const auto fetch_next_gemm1_work = [&] (
                        uint32_t& group_idx,
                        uint32_t& group_block_off,
                        uint32_t& m_block_idx,
                        uint32_t& n_block_idx,
                        uint32_t& group_m) {
                    const uint32_t next_block_idx =
                        work_begin + (current_iter++) * scheduler_sms + scheduler_sm_idx;
                    if (next_block_idx >= work_end)
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
                    if constexpr (kSmSpecialization && kEnableRemotePull) {
                        if (wait_for_arrival) {
                            while (ptx::ld_acq(
                                       workspace.get_l1_arrival_count_ptr(g)) < kNumPullCTAs);
                        }
                    }

                    const bool is_gate = (n_block_idx * kBlockN < kGemm1N / 2);
                    // kQuantizeOutput writes FP4 + scales directly and never
                    // dereferences the BF16 output pointer. Keep the ABI
                    // argument valid, but do not form an out-of-range tile
                    // pointer from its one-element placeholder.
                    __ppu_bfloat16* gemm1_tile_output = gemm1_output;
                    if constexpr (!kQuantizeOutput) {
                        gemm1_tile_output +=
                            (uint64_t)bi.pool_base * gemm1_output_stride
                            + (uint64_t)n_block_idx
                                * ((fused_swiglu || fused_situ)
                                       ? (kBlockN / 2) : kBlockN);
                    }
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
                        Cfg, kQuantizeOutput, kSituTanhBf16x2, kFusedSitu, kSiluThenClamp>(
                        dispatched_token_pool + (uint64_t)bi.pool_base * token_bytes,
                        sfa_scratch + (uint64_t)g * (BLOCK_M * kSfCols),
                        weight_base
                            + (uint64_t)weight_expert * kGemm1N * kTokenBytes
                            + (uint64_t)n_block_idx * kBlockN * kTokenBytes,
                        gemm1_sfb_tile,
                        static_cast<int>(kSfbBlockN),
                        static_cast<int>(sfb_subtile_idx * kBlockN),
                        gemm1_tile_output,
                        static_cast<int>(gemm1_output_stride),
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
                                         * sizeof(uint16_t)), rank,
                        dispatched_weights_pool,
                        static_cast<int>(bi.pool_base));
                }
            };

            if (initial_gemm_sm) {
                if (thread_idx < Cfg::ThreadNum)
                    run_gemm_range(
                        0, tail_work_begin, kNumGemmCTAs,
                        initial_gemm_sm_idx, /*wait_for_arrival=*/true);
                __syncthreads();
            }

            if constexpr (kTailGemmBlocks > 0 || kTailGemmPercent > 0) {
                // Producers arrive after their final pull; consumers arrive
                // after finishing the prefix. Stagger polling across SMs so an
                // early side does not continuously hammer the barrier line.
                comm::grid_sync<kNumCTAs, /*kGridSyncIndex=*/3,
                                /*kPollBackoffCycles=*/128>(
                    workspace, cta_idx, thread_idx, block_sync);
                if (thread_idx < Cfg::ThreadNum)
                    run_gemm_range(
                        tail_work_begin, total_gemm_work, kNumCTAs, cta_idx,
                        /*wait_for_arrival=*/false);
                __syncthreads();
            }

        }  // kEnableGemm1
    }  // kEnableGemm1 || kEnableRemotePull

    if (!finalize_dispatch)
        return;

    comm::grid_sync<kNumCTAs, /*kGridSyncIndex=*/1>(workspace, cta_idx, thread_idx, block_sync);
    {
        const uint32_t flat   = cta_idx * kNumThreads + thread_idx;
        const uint32_t stride = kNumCTAs * kNumThreads;
        for (uint32_t i = flat; i < kNumExperts; i += stride)
            *workspace.get_expert_send_count_ptr(i) = 0;
    }
}

// ----------------------------------------------------------------------------
// Thin `__global__` shim around the `__device__` impl above. Keeps the original
// kernel entry name + signature (defaults included) so the Python JIT launch
// template (`fp4_mega_moe.py`) launches it unchanged. The C++ JIT path does NOT
// use this shim; it emits its own `extern "C" __global__` wrapper that forwards
// to `dispatch_gemm1_device`.
// ----------------------------------------------------------------------------
template <
    uint32_t kNumRanks, uint32_t kNumCTAs, uint32_t kNumThreads,
    uint32_t kNumExperts, uint32_t kNumExpertsPerRank, uint32_t kNumTopk,
    uint32_t BLOCK_M,
    uint32_t kChunkBlocks = kNumCTAs,
    uint32_t kNumPullCTAs = kNumCTAs,
    uint32_t kNumGemmCTAs = kNumCTAs,
    bool kSmSpecialization = false,
    uint32_t kPullSlots = 8,
    uint32_t kTailGemmBlocks = 0,
    uint32_t kTailGemmPercent = 0,
    bool kEnableGemm1 = false,
    uint32_t kGemm1N = 0,
    uint32_t kTokenBytes = 0,
    uint32_t kGemm1BlockN = 0,
    uint32_t kGemm1BlockK = 0,
    uint32_t kGemm1SfbBlockN = kGemm1BlockN,
    bool kEnableRemotePull = true,
    bool kEnableRemotePush = false,
    uint32_t kGemm1WarpM = 32,
    uint32_t kGemm1WarpN = 64,
    uint32_t kGemm1Stages = 2,
    bool kQuantizeOutput = false,
    bool kSituTanhBf16x2 = false,
    bool kFusedSitu = false,
    bool kSiluThenClamp = false
>
CUTLASS_DEVICE void dispatch_gemm1_device(
        void* workspace_base,
        const layout::SymBuffer<kNumRanks> sym_buffer,
        const int64_t* __restrict__ input_topk_idx,
        const float* __restrict__ input_topk_weights,
        const uint8_t* __restrict__ input_token,
        const uint8_t* __restrict__ input_sf,
        uint8_t* __restrict__ dispatched_token_pool,
        uint8_t* __restrict__ dispatched_scale_pool,
        float* __restrict__ dispatched_weights_pool,
        const uint32_t num_tokens,
        const uint32_t num_max_tokens_per_rank,
        const uint32_t token_bytes,
        const uint32_t sf_bytes,
        const uint8_t* __restrict__ gemm1_weight = nullptr,
        const uint8_t* __restrict__ gemm1_weight_sf = nullptr,
        __ppu_bfloat16* __restrict__ gemm1_output = nullptr,
        const uint32_t gemm1_output_stride = 0,
        const uint16_t* __restrict__ gemm1_weight_sf_u16 = nullptr,
        const uint8_t* __restrict__ gemm1_weight_replicas = nullptr,
        const uint16_t* __restrict__ gemm1_weight_sf_u16_replicas = nullptr,
        const uint32_t num_master_experts = kNumExpertsPerRank,
        uint16_t* __restrict__ sfa_scratch = nullptr,
        float activation_clamp = 0.0f,
        bool fused_swiglu = false,
        bool fused_situ = false,
        float situ_beta = 4.0f,
        float situ_linear_beta = 25.0f,
        uint8_t* __restrict__ fp4_output = nullptr,
        uint8_t* __restrict__ scale_output = nullptr,
        int rank = 0,
        uint32_t pull_begin_block = 0,
        uint32_t pull_end_block = 0xffffffffu,
        bool resume_dispatch = false,
        bool finalize_dispatch = true,
        bool synchronize_pull = true,
        bool publish_arrivals = false) {
    dispatch_gemm1_body<
        kNumRanks, kNumCTAs, kNumThreads, kNumExperts, kNumExpertsPerRank, kNumTopk,
        BLOCK_M, kChunkBlocks, kNumPullCTAs, kNumGemmCTAs,
        kSmSpecialization, kPullSlots, kTailGemmBlocks, kTailGemmPercent, kEnableGemm1,
        kGemm1N, kTokenBytes, kGemm1BlockN, kGemm1BlockK, kGemm1SfbBlockN,
        kEnableRemotePull, kEnableRemotePush,
        kGemm1WarpM, kGemm1WarpN, kGemm1Stages, kQuantizeOutput,
        kSituTanhBf16x2, kFusedSitu, kSiluThenClamp>(
        workspace_base, sym_buffer, input_topk_idx, input_topk_weights,
        input_token, input_sf, dispatched_token_pool, dispatched_scale_pool, dispatched_weights_pool,
        num_tokens, num_max_tokens_per_rank, token_bytes, sf_bytes,
        gemm1_weight, gemm1_weight_sf, gemm1_output, gemm1_output_stride,
        gemm1_weight_sf_u16, gemm1_weight_replicas,
        gemm1_weight_sf_u16_replicas, num_master_experts,
        sfa_scratch, activation_clamp, fused_swiglu,
        fused_situ, situ_beta, situ_linear_beta,
        fp4_output, scale_output, rank,
        pull_begin_block, pull_end_block,
        resume_dispatch, finalize_dispatch,
        synchronize_pull, publish_arrivals);
}

template <
    uint32_t kNumRanks, uint32_t kNumCTAs, uint32_t kNumThreads,
    uint32_t kNumExperts, uint32_t kNumExpertsPerRank, uint32_t kNumTopk,
    uint32_t BLOCK_M,
    uint32_t kChunkBlocks = kNumCTAs,
    uint32_t kNumPullCTAs = kNumCTAs,
    uint32_t kNumGemmCTAs = kNumCTAs,
    bool kSmSpecialization = false,
    uint32_t kPullSlots = 8,
    uint32_t kTailGemmBlocks = 0,
    uint32_t kTailGemmPercent = 0,
    bool kEnableGemm1 = false,
    uint32_t kGemm1N = 0,
    uint32_t kTokenBytes = 0,
    uint32_t kGemm1BlockN = 0,
    uint32_t kGemm1BlockK = 0,
    uint32_t kGemm1SfbBlockN = kGemm1BlockN,
    bool kEnableRemotePull = true,
    bool kEnableRemotePush = false,
    uint32_t kGemm1WarpM = 32,
    uint32_t kGemm1WarpN = 64,
    uint32_t kGemm1Stages = 2,
    bool kQuantizeOutput = false,
    bool kSituTanhBf16x2 = false,
    bool kFusedSitu = false,
    bool kSiluThenClamp = false
>
__global__ __launch_bounds__(kNumThreads, 1) void
dispatch_gemm1_kernel(
        void* workspace_base,
        const layout::SymBuffer<kNumRanks> sym_buffer,
        const int64_t* __restrict__ input_topk_idx,
        const float* __restrict__ input_topk_weights,
        const uint8_t* __restrict__ input_token,
        const uint8_t* __restrict__ input_sf,
        uint8_t* __restrict__ dispatched_token_pool,
        uint8_t* __restrict__ dispatched_scale_pool,
        float* __restrict__ dispatched_weights_pool,
        const uint32_t num_tokens,
        const uint32_t num_max_tokens_per_rank,
        const uint32_t token_bytes,
        const uint32_t sf_bytes,
        const uint8_t* __restrict__ gemm1_weight = nullptr,
        const uint8_t* __restrict__ gemm1_weight_sf = nullptr,
        __ppu_bfloat16* __restrict__ gemm1_output = nullptr,
        const uint32_t gemm1_output_stride = 0,
        const uint16_t* __restrict__ gemm1_weight_sf_u16 = nullptr,
        const uint8_t* __restrict__ gemm1_weight_replicas = nullptr,
        const uint16_t* __restrict__ gemm1_weight_sf_u16_replicas = nullptr,
        const uint32_t num_master_experts = kNumExpertsPerRank,
        uint16_t* __restrict__ sfa_scratch = nullptr,
        float activation_clamp = 0.0f,
        bool fused_swiglu = false,
        bool fused_situ = false,
        float situ_beta = 4.0f,
        float situ_linear_beta = 25.0f,
        uint8_t* __restrict__ fp4_output = nullptr,
        uint8_t* __restrict__ scale_output = nullptr,
        int rank = 0,
        uint32_t pull_begin_block = 0,
        uint32_t pull_end_block = 0xffffffffu,
        bool resume_dispatch = false,
        bool finalize_dispatch = true,
        bool synchronize_pull = true,
        bool publish_arrivals = false) {
    dispatch_gemm1_device<kNumRanks, kNumCTAs, kNumThreads, kNumExperts, kNumExpertsPerRank, kNumTopk, BLOCK_M, kChunkBlocks, kNumPullCTAs, kNumGemmCTAs, kSmSpecialization, kPullSlots, kTailGemmBlocks, kTailGemmPercent, kEnableGemm1, kGemm1N, kTokenBytes, kGemm1BlockN, kGemm1BlockK, kGemm1SfbBlockN, kEnableRemotePull, kEnableRemotePush, kGemm1WarpM, kGemm1WarpN, kGemm1Stages, kQuantizeOutput, kSituTanhBf16x2, kFusedSitu, kSiluThenClamp>(workspace_base, sym_buffer, input_topk_idx, input_topk_weights, input_token, input_sf, dispatched_token_pool, dispatched_scale_pool, dispatched_weights_pool, num_tokens, num_max_tokens_per_rank, token_bytes, sf_bytes, gemm1_weight, gemm1_weight_sf, gemm1_output, gemm1_output_stride, gemm1_weight_sf_u16, gemm1_weight_replicas, gemm1_weight_sf_u16_replicas, num_master_experts, sfa_scratch, activation_clamp, fused_swiglu, fused_situ, situ_beta, situ_linear_beta, fp4_output, scale_output, rank, pull_begin_block, pull_end_block, resume_dispatch, finalize_dispatch, synchronize_pull, publish_arrivals);
}

} // namespace deep_gemm::mega_moe::fp4
