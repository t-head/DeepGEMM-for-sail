#pragma once

#include <cstdint>
#include <cstdlib>
#include <cuda_bf16.h>
#include "dispatch_layout.cuh"

namespace deep_gemm {

// Grid-completion counter for the folded arrival push (see mxfp4_quantize_kernel
// tail). Self-resetting: atomicInc with wrap value gridDim.x-1 returns it to 0
// after the last block retires, so it needs no external zero-init even though the
// symmetric buffer is uninitialized. __device__ statics are zero-initialized at
// module load, so the very first (generation>0) launch also starts clean. Quant
// launches are serialized on a single stream, so one counter is race-free across
// generations.
__device__ uint32_t g_quant_arrival_retire = 0;

__device__ __forceinline__ uint32_t atomic_add_generation_count(
    uint32_t* counter, uint32_t generation, uint32_t max_tokens_per_expert) {
    if (!use_tagged_generation_counts(generation, max_tokens_per_expert))
        return atomicAdd(counter, 1u);

    const uint32_t count_mask = tagged_count_mask(max_tokens_per_expert);
    const uint32_t tag = pack_count_generation(generation, max_tokens_per_expert);
    uint32_t old = *counter;
    while ((old & ~count_mask) != tag) {
        uint32_t prev = atomicCAS(counter, old, tag);
        if (prev == old) {
            old = tag;
            break;
        }
        old = prev;
    }
    return atomicAdd(counter, 1u) & count_mask;
}

// Convert two FP32 values to packed E2M1x2 (two FP4 values in 1 byte)
__device__ __forceinline__ uint8_t cvt_f32x2_to_fp4x2(float hi, float lo) {
    uint16_t b;
    asm volatile("cvt.rn.satfinite.e2m1x2.f32 %0, %1, %2;\n"
                 : "=h"(b)
                 : "f"(hi), "f"(lo));
    return static_cast<uint8_t>(b);
}

// Fast power-of-2 for BF16: returns 2^x as __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 fast_pow2_bf16(int x) {
    uint16_t bits = static_cast<uint16_t>((x + 127) << 7);
    return *reinterpret_cast<__nv_bfloat16*>(&bits);
}

// Fast ceiling log2 for BF16
__device__ __forceinline__ int fast_log2_ceil_bf16(__nv_bfloat16 x) {
    uint16_t bits = *reinterpret_cast<uint16_t*>(&x);
    int exp_x = (bits >> 7) & 0xFF;
    int man_bits = bits & ((1 << 7) - 1);
    return exp_x - 127 + (man_bits != 0);
}

// Compute MXFP4 scale factors from BF16 amax
// Returns: scale (BF16 multiplier for quantization), scale_inv (UE8M0 for dequantization)
__device__ __forceinline__ void calculate_mxfp4_scales_bf16(
    __nv_bfloat16 amax, __nv_bfloat16& scale, uint8_t& scale_inv) {
    constexpr float kAmaxInvMXFP4 = 0.16666667f;  // 1/6.0, max representable E2M1 value
    __nv_bfloat16 scale_inv_bf16 = __hmul(amax, __float2bfloat16(kAmaxInvMXFP4));
    int exp_scale_inv = fast_log2_ceil_bf16(scale_inv_bf16);
    scale = fast_pow2_bf16(-exp_scale_inv);
    scale_inv_bf16 = fast_pow2_bf16(exp_scale_inv);
    scale_inv = static_cast<uint8_t>((*reinterpret_cast<uint16_t*>(&scale_inv_bf16)) >> 7);
}

#ifdef DG_SFA_PUSH_COUNTS
// DG_SFA_PUSH_COUNTS: push this rank's final per-expert token counts into each owner's
// staging counts region so the owner reads them locally (vs a remote NVLink read in
// expert_preprocess). Called by ALL threads of ONE block AFTER all quant blocks have
// finalized their local counts (retire==gridDim or grid.sync). Writes are __stbl (uncached
// / bypass L1); the caller's single lightweight uncache system fence orders them system-
// wide. Counts are UNPACKED here (owner reads them raw). Layout mirrors pair_counts[src_rank*nlr + le].
__device__ __forceinline__ void push_pair_counts(
    void* sym_buf_base, const int64_t* __restrict__ staging_addrs,
    const DispatchBufferLayout& layout, uint32_t num_total_experts, uint32_t num_ranks,
    uint32_t rank_idx, uint32_t generation, uint32_t max_tokens_per_expert) {
    if (staging_addrs == nullptr || staging_addrs[0] == 0) return;
    const uint32_t nlr = num_total_experts / num_ranks;   // real local experts per owner
    const uint32_t* counts = layout.expert_token_counts_ptr(sym_buf_base);
    const uint64_t coff = layout.staging_counts_offset()
        + static_cast<uint64_t>(generation & 1u) * layout.staging_counts_parity_bytes();
    for (uint32_t g = threadIdx.x; g < num_total_experts; g += blockDim.x) {
        uint32_t cnt = unpack_generation_count(counts[g], generation, max_tokens_per_expert);
        uint32_t d  = g / nlr;        // owner rank
        uint32_t le = g - d * nlr;    // local expert on owner
        uint32_t* cbase = reinterpret_cast<uint32_t*>(
            reinterpret_cast<uint8_t*>(staging_addrs[d]) + coff);
        __stbl(&cbase[rank_idx * nlr + le], cnt);   // peer-VA uint32 push, bypass L1
    }
}
#endif

// MXFP4 quantization kernel: BF16 input → MXFP4 in symmetric buffer (per-expert contiguous)
//
// Optimized: quantize each token ONCE into shared memory, then scatter-copy to topk expert slots.
// This avoids redundant BF16 reads and FP4 conversion for each topk destination.
//
// Template params:
//   HIDDEN: hidden dimension (must be multiple of 32)
//   MAX_TOPK: maximum topk value
template <int HIDDEN, int MAX_TOPK>
__global__ void mxfp4_quantize_kernel(
    const __nv_bfloat16* __restrict__ input,    // [num_tokens, HIDDEN]
    const int32_t* __restrict__ topk_ids,       // [num_tokens, topk]
    void* __restrict__ sym_buf_base,            // symmetric buffer base pointer
    uint32_t num_tokens,
    uint32_t topk,
    uint32_t num_local_experts,
    uint32_t num_total_experts,
    uint32_t max_tokens_per_expert,
    uint32_t hidden_dim,
    uint32_t generation,
    int64_t* __restrict__ profile_clocks = nullptr,
    // Folded arrival push (do_arrival != 0): the last block to retire signals every
    // consumer that this rank's generation is complete, replacing the standalone
    // arrival_push_kernel launch.
    const int64_t* __restrict__ sym_buf_addrs = nullptr,
    uint32_t rank_idx = 0,
    uint32_t num_ranks = 0,
    uint64_t arrival_offset = 0,
    uint32_t do_arrival = 0,
    // DG_SFA_PUSH: peer base ptrs of the SEPARATE staging symmetric buffer (row-major
    // [num_local_real][num_ranks][max_tokens][ksb] uint16, x2 parity). nullptr => no push.
    const int64_t* __restrict__ staging_addrs = nullptr,
    // last_fence != 0 (DG_SFA_PUSH only): replace the heavyweight cache-flushing
    // per-block fence with an uncache system fence. Pushes are uncached __stbl, so this
    // establishes the per-CTA release needed by the retire counter without flushing L1.
    uint32_t last_fence = 0) {

    constexpr int ELEMS_PER_THREAD = 16; // 16 BF16 values = 2 int4 loads
    constexpr int SCALE_GROUP_SIZE = 32;
    constexpr int LANES_PER_GROUP = SCALE_GROUP_SIZE / ELEMS_PER_THREAD;  // 2
    constexpr int QUANT_CHUNKS = HIDDEN / ELEMS_PER_THREAD;
    constexpr int FP4_INTS = HIDDEN / 8;  // number of int-sized FP4 output words
    constexpr int K_BLOCKS = (HIDDEN + 31) / 32;
    constexpr int K_SCALE_BLOCKS = (K_BLOCKS + 1) / 2;
    // Production HIDDEN=7168 has 448 sixteen-element chunks: seven warps process
    // exactly two chunks per lane while the eighth overlaps the topk slot atomics.
    constexpr int QUANT_WARPS = 7;
    constexpr int QUANT_THREADS = QUANT_WARPS * 32;
    constexpr bool SPECIALIZE_ATOMIC_WARP = (QUANT_CHUNKS % QUANT_THREADS) == 0;

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    DispatchBufferLayout layout(num_total_experts, num_total_experts,
                                max_tokens_per_expert, hidden_dim);

#ifdef DG_SFA_PUSH
    // A real staging base means the fused consumer reads scales exclusively from
    // owner-local staging. Hoist this uniform check out of the token loop.
    const bool sfa_staging_active =
        staging_addrs != nullptr && num_ranks > 0 && staging_addrs[0] != 0;
#endif

    // Shared memory: quantized FP4 data + packed scales (quantize once, scatter many)
    __shared__ __align__(16) int s_fp4_data[FP4_INTS];
    __shared__ uint8_t s_scale_inv[K_BLOCKS];
    __shared__ __align__(16) uint16_t s_packed_scale[K_SCALE_BLOCKS];  // 16B-aligned for int4 push
    __shared__ uint32_t s_slot[MAX_TOPK];
    __shared__ int s_expert[MAX_TOPK];

    // Profiling: record clocks at phase boundaries (block 0, first token only)
    uint64_t _prof_t0 = 0, _prof_t1 = 0, _prof_t2 = 0;
    if (profile_clocks && blockIdx.x == 0 && threadIdx.x == 0) _prof_t0 = clock64();

    // Grid-stride over tokens: launch ~num_resident_blocks (<= num_tokens) instead of
    // one block per token. This keeps the whole grid to a single wave, so the folded
    // arrival path fences once per block in ONE wave (vs 256 blocks / ~2 waves, which
    // exposed ~2x the remote-write round trip). Each block below fences once (Phase 3)
    // AFTER pushing all its tokens, covering every push it issued.
    for (int token_idx = blockIdx.x; token_idx < num_tokens; token_idx += gridDim.x) {
    const bool _prof_first = (blockIdx.x == 0 && token_idx == blockIdx.x);
    const __nv_bfloat16* token_data = input + static_cast<int64_t>(token_idx) * hidden_dim;
#ifdef DG_QUANT_PROFILE_P1SPLIT
    // Thread 32 is a quant worker but never a production topk atomic lane, so its
    // setup interval is not charged with the dedicated warp-7 atomics.
    const bool _p1s_record = profile_clocks && _prof_first && threadIdx.x == 32;
    uint64_t _p1s_t0 = 0, _p1s_t1 = 0, _p1s_t2 = 0, _p1s_t3 = 0;
    if (_p1s_record) _p1s_t0 = clock64();
#endif

    // ---- Phase 0: claim expert slots (issued BEFORE quantize to hide atomic latency) ----
    // The slot atomics are independent of the quantization (Phase 1 needs only the input),
    // and the slot is not used until the scatter/push below. Issuing them up front lets the
    // atomic round-trip overlap the other threads' Phase 1 work instead of stalling on the
    // critical path after it. One atomic per topk lane (topk experts of a token are distinct
    // -> no intra-token contention), so the topk atomics also fire in parallel rather than
    // serially on thread 0. s_slot/s_expert become visible at Phase 1's __syncthreads.
    if constexpr (SPECIALIZE_ATOMIC_WARP) {
        // Warp 7 owns the six production topk atomics and does no quant compute.
        // Warps 0..6 therefore reach the first barrier with equal quant work instead
        // of making every warp wait for warp 0's atomic+compute serial path.
        if (warp_id == QUANT_WARPS && lane_id < topk) {
            const int t = lane_id;
            uint32_t* counts = layout.expert_token_counts_ptr(sym_buf_base);
            int expert_idx = topk_ids[token_idx * topk + t];
            s_expert[t] = expert_idx;
            s_slot[t] = (expert_idx >= 0)
                ? atomic_add_generation_count(&counts[expert_idx], generation, max_tokens_per_expert)
                : max_tokens_per_expert;
        }
    } else {
        if (threadIdx.x < topk) {
            const int t = threadIdx.x;
            uint32_t* counts = layout.expert_token_counts_ptr(sym_buf_base);
            int expert_idx = topk_ids[token_idx * topk + t];
            s_expert[t] = expert_idx;
            s_slot[t] = (expert_idx >= 0)
                ? atomic_add_generation_count(&counts[expert_idx], generation, max_tokens_per_expert)
                : max_tokens_per_expert;
        }
    }

    // ---- Phase 1: Quantize token data ONCE into shared memory ----
#ifdef DG_QUANT_PROFILE_P1SPLIT
    if (_p1s_record) _p1s_t1 = clock64();
#endif
    // Compile-time worker stride lets the compiler unroll the fixed iteration count.
    // Loads are HOISTED into a register array first so the
    // P1_ITERS independent int4 fetches issue back-to-back and overlap (MLP): the
    // per-iteration `load; compute` schedule made the compiler emit `s.wait vldcnt(0)`
    // after every load (serializing the memory latency, acu No-Eligible ~48%);
    // separating loads from compute lets it wait with staggered vldcnt instead.
    constexpr int THREADS = SPECIALIZE_ATOMIC_WARP ? QUANT_THREADS : 256;
    constexpr int P1_ITERS = (QUANT_CHUNKS + THREADS - 1) / THREADS;
    const int4* token_v = reinterpret_cast<const int4*>(token_data);
    int4 raw[P1_ITERS][2];
    #pragma unroll
    for (int it = 0; it < P1_ITERS; ++it) {
        const int i = it * THREADS + threadIdx.x;
        if (threadIdx.x < THREADS && i < QUANT_CHUNKS) {
            raw[it][0] = token_v[2 * i];
            raw[it][1] = token_v[2 * i + 1];
        }
    }
    #pragma unroll
    for (int it = 0; it < P1_ITERS; ++it) {
        const int i = it * THREADS + threadIdx.x;
        if (threadIdx.x >= THREADS || i >= QUANT_CHUNKS) continue;
        __nv_bfloat162 local_v2[ELEMS_PER_THREAD / 2];

        __nv_bfloat162 amax2 = __float2bfloat162_rn(0.0f);
        for (int j = 0; j < ELEMS_PER_THREAD / 2; ++j) {
            local_v2[j] = reinterpret_cast<const __nv_bfloat162*>(&raw[it][j / 4])[j % 4];
            amax2 = __hmax2(amax2, __habs2(local_v2[j]));
        }
        __nv_bfloat16 amax = __hmax(amax2.x, amax2.y);

        amax = __hmax(amax, __shfl_xor_sync(0xffffffff, amax, 1, 2));

        // PROBE(dedup scale calc): all 4 lanes of a scale-group share `amax` after
        // the shfl reduction, so compute scale/scale_inv only on the group leader
        // (lane 0) and broadcast `scale` to the other 3. scale_inv is only stored by
        // the leader, so lanes 1-3 never needed it.
        __nv_bfloat16 scale;
        if ((lane_id & (LANES_PER_GROUP - 1)) == 0) {
            uint8_t scale_inv;
            calculate_mxfp4_scales_bf16(amax, scale, scale_inv);
            s_scale_inv[i / LANES_PER_GROUP] = scale_inv;
        }
        scale = __shfl_sync(0xffffffff, scale, (lane_id & ~(LANES_PER_GROUP - 1)) & 31, 32);

        __nv_bfloat162 scale2 = __halves2bfloat162(scale, scale);
        int2 int_value;
        uint8_t* fp4x2_values = reinterpret_cast<uint8_t*>(&int_value);
        for (int j = 0; j < ELEMS_PER_THREAD / 2; ++j) {
            __nv_bfloat162 scaled = __hmul2(local_v2[j], scale2);
            float2 f2 = __bfloat1622float2(scaled);
            fp4x2_values[j] = cvt_f32x2_to_fp4x2(f2.y, f2.x);
        }

        reinterpret_cast<int2*>(s_fp4_data)[i] = int_value;
    }
#ifdef DG_QUANT_PROFILE_P1SPLIT
    if (_p1s_record) _p1s_t2 = clock64();
#endif
    __syncthreads();
#ifdef DG_QUANT_PROFILE_P1SPLIT
    if (_p1s_record) _p1s_t3 = clock64();
#endif

    // Pack uint8 scales into uint16 pairs in shared memory
    for (int j = threadIdx.x; j < K_SCALE_BLOCKS; j += blockDim.x) {
        uint8_t s0 = s_scale_inv[2 * j];
        uint8_t s1 = (2 * j + 1 < K_BLOCKS) ? s_scale_inv[2 * j + 1] : 0;
        s_packed_scale[j] = static_cast<uint16_t>(s0) | (static_cast<uint16_t>(s1) << 8);
    }
    __syncthreads();

#ifdef DG_QUANT_PROFILE_P1SPLIT
    if (_p1s_record) {
        const uint64_t _p1s_t4 = clock64();
        profile_clocks[3] = _p1s_t1 - _p1s_t0;  // setup before quant work
        profile_clocks[4] = _p1s_t2 - _p1s_t1;  // BF16 loads + quant compute
        profile_clocks[5] = _p1s_t3 - _p1s_t2;  // first CTA barrier wait
        profile_clocks[6] = _p1s_t4 - _p1s_t3;  // scale pack + second barrier
    }
#endif

    if (profile_clocks && _prof_first && threadIdx.x == 0) _prof_t1 = clock64();

#ifdef DG_SFA_PUSH
    // Pre-issue the peer SFA stores before the much larger local FP4 scatter.
    // Each topk warp retains its own destination, preserving the six independent
    // peer streams. The uncached remote writes can complete while the same warps
    // issue the following 3.5-KB local FP4 stores; the final CTA barrier/fence
    // still publishes both streams together before folded arrival.
    if (sfa_staging_active) {
        const uint32_t nlr = num_total_experts / num_ranks;
        const uint64_t parity_stride = layout.staging_parity_bytes();
        if (warp_id < topk) {
            int t = warp_id;
            int e = s_expert[t];
            uint32_t slot = s_slot[t];
            if (e >= 0 && slot < max_tokens_per_expert) {
                uint32_t d  = static_cast<uint32_t>(e) / nlr;
                uint32_t le = static_cast<uint32_t>(e) - d * nlr;
                uint8_t* peer_base = reinterpret_cast<uint8_t*>(staging_addrs[d]);
                uint16_t* staging = reinterpret_cast<uint16_t*>(
                    peer_base + static_cast<uint64_t>(generation & 1u) * parity_stride);
                uint64_t sidx =
                    (static_cast<uint64_t>(le * num_ranks + rank_idx)
                     * max_tokens_per_expert + slot) * K_SCALE_BLOCKS;
                uint16_t* dst = staging + sidx;
#ifndef DG_SFA_PUSH_NOWRITE
                constexpr int KSB_I4 = K_SCALE_BLOCKS / 8;
                const int4* s4 = reinterpret_cast<const int4*>(s_packed_scale);
                int4* d4 = reinterpret_cast<int4*>(dst);
                for (int v = lane_id; v < KSB_I4; v += 32)
                    __stbl(&d4[v], s4[v]);
                for (int j = KSB_I4 * 8 + lane_id; j < K_SCALE_BLOCKS; j += 32)
                    __stbl(&dst[j], s_packed_scale[j]);
#else
                (void)dst;
#endif
            }
        }
    }
#endif

    // ---- Phase 2: Scatter-copy quantized data to topk expert slots ----
    // Slots were claimed in Phase 0 and are already visible after Phase 1's __syncthreads,
    // so no extra sync here.
    // Vectorized scatter: int4 (16-byte) writes for FP4 data
    constexpr int FP4_INT4S = FP4_INTS / 4;
    const int4* s_fp4_v = reinterpret_cast<const int4*>(s_fp4_data);

    // One warp owns one topk destination. This preserves contiguous 512-byte
    // transactions while exposing six independent destination streams at once.
    if (warp_id < topk) {
        int t = warp_id;
        int expert_idx = s_expert[t];
        uint32_t slot = s_slot[t];

        if (expert_idx >= 0 && slot < max_tokens_per_expert) {
            uint8_t* fp4_row_base = layout.fp4_data_ptr(sym_buf_base, expert_idx);
            int4* fp4_out = reinterpret_cast<int4*>(
                fp4_row_base + static_cast<int64_t>(slot) * (hidden_dim / 2));
            // Keep only two shared chunks in flight. Full seven-chunk hoisting
            // creates excessive register lifetime, while pairs still give the
            // scheduler an independent TSM load to overlap with each wait.
            constexpr int FP4_WARP_ITERS = (FP4_INT4S + 31) / 32;
            #pragma unroll
            for (int it = 0; it < FP4_WARP_ITERS; it += 2) {
                const int i0 = lane_id + it * 32;
                const int i1 = i0 + 32;
                int4 fp4_0;
                int4 fp4_1;
                if (i0 < FP4_INT4S) fp4_0 = s_fp4_v[i0];
                if (i1 < FP4_INT4S) fp4_1 = s_fp4_v[i1];
                if (i0 < FP4_INT4S) fp4_out[i0] = fp4_0;
                if (i1 < FP4_INT4S) fp4_out[i1] = fp4_1;
            }

#if defined(DG_SFA_PUSH) && !defined(DG_SFA_KEEP_LOCAL_SCALE)
            // With real push staging this sym-buffer scale copy has no fused
            // consumer. Keep it for generation-0/dummy-staging callers.
            if (!sfa_staging_active) {
#endif
            uint16_t* scale_out = layout.scale_ptr(sym_buf_base, expert_idx);
            for (int j = lane_id; j < K_SCALE_BLOCKS; j += 32) {
#ifdef DG_SFA_ROWMAJOR_SRC
                // Row-major [max_tokens, ksb]: this token's ksb scales are stored
                // contiguously, so the block-copy remote read is one contiguous
                // burst per rank instead of ksb strided segments. (This write is
                // also contiguous per token, vs the strided column-major write.)
                scale_out[static_cast<int64_t>(slot) * K_SCALE_BLOCKS + j] = s_packed_scale[j];
#else
                scale_out[static_cast<int64_t>(j) * max_tokens_per_expert + slot] = s_packed_scale[j];
#endif
            }
#if defined(DG_SFA_PUSH) && !defined(DG_SFA_KEEP_LOCAL_SCALE)
            }
#endif
        }
        // No __syncthreads() needed: all shared inputs are read-only here.
    }

    // Reuse of shared memory across grid-stride iterations: make sure this token's
    // readers (scatter/push of s_fp4_data/s_packed_scale/s_expert/s_slot) are done
    // before the next iteration's Phase 1 overwrites them.
    __syncthreads();

    // Record P2 only after the reuse barrier. Global/shared stores are asynchronous,
    // so sampling before this barrier measures issue time and can hide the completion
    // wait by shifting it outside the reported phase.
    if (profile_clocks && _prof_first && threadIdx.x == 0) {
        _prof_t2 = clock64();
        profile_clocks[0] = _prof_t1 - _prof_t0;  // Phase 1: quantize to SMEM
        profile_clocks[1] = _prof_t2 - _prof_t1;  // Phase 2: scatter/push + reuse barrier
        profile_clocks[2] = _prof_t2 - _prof_t0;  // Total through P2 completion
    }
    }  // end grid-stride token loop

    // ---- Phase 3 (optional): folded arrival push ----
    // Detect grid completion via a self-resetting atomic counter and have the last
    // block signal every consumer, saving the separate arrival_push_kernel launch.
    // Each block release-fences at DEVICE scope before its increment: this both
    // orders the block's scatter writes ahead of the counter bump and makes them
    // globally visible in this rank's HBM (which is what a consumer's NVLink read
    // observes — the same visibility the old kernel boundary provided). When the
    // last block sees the full count, every block's data is therefore consumer-
    // readable; it then release-fences SYSTEM-wide (matching the old arrival_push)
    // and publishes `generation` into each consumer's local slot[rank_idx].
    if (do_arrival) {
#ifdef DG_SFA_PUSH
        // Push writes targeted PEER HBM: device-scope fence only guarantees local-HBM
        // visibility (enough for the pull model). System scope is required so every
        // block's remote pushes are globally visible before the arrival flag.
        // Pushes use __stbl (uncached / bypass-L1), so the lightweight uncache fence
        // does not flush a per-SM write-back cache. Every CTA must nevertheless publish
        // its own writes before incrementing the retire counter: a fence executed only
        // by the last CTA cannot establish a release sequence for stores issued by the
        // other CTAs. The final fence below additionally orders the last CTA's counts
        // push and the arrival flags.
        if (last_fence) __ppu_threadfence_system_uncache();
        else            __threadfence_system();
#else
        __threadfence();
#endif
        __shared__ bool s_is_last_block;
        if (threadIdx.x == 0) {
            uint32_t ticket = atomicInc(&g_quant_arrival_retire, gridDim.x - 1);
            s_is_last_block = (ticket == gridDim.x - 1);
        }
        __syncthreads();
        if (s_is_last_block) {
#ifdef DG_SFA_PUSH_COUNTS
            // Counts are final (all blocks retired). Push them, then fence covers both the
            // SFA pushes (all blocks) and these counts pushes (this block).
            push_pair_counts(sym_buf_base, staging_addrs, layout, num_total_experts,
                             num_ranks, rank_idx, generation, max_tokens_per_expert);
            __syncthreads();
#endif
            if (threadIdx.x == 0) {
#ifdef DG_SFA_PUSH
                if (last_fence) __ppu_threadfence_system_uncache();
                else            __threadfence_system();
#else
                __threadfence_system();
#endif
                for (uint32_t c = 0; c < num_ranks; ++c) {
                    uint32_t* slot = reinterpret_cast<uint32_t*>(
                        sym_buf_addrs[c] + static_cast<int64_t>(arrival_offset)) + rank_idx;
                    *slot = generation;
                }
            }

        }
    }
}

// Atomic-arrival producer: push this rank's generation into every consumer's local
// slot[rank_idx]. Consumers poll their own local slot (no remote flag spinning).
// __threadfence_system() before the writes gives release semantics: a consumer that
// observes the slot is guaranteed to see this rank's quantized counts/data.
__global__ void arrival_push_kernel(
    const int64_t* __restrict__ sym_buf_addrs,
    uint32_t rank_idx, uint32_t num_ranks,
    uint64_t arrival_offset, uint32_t generation) {
    if (threadIdx.x == 0) {
        __threadfence_system();
        for (uint32_t c = 0; c < num_ranks; ++c) {
            uint32_t* slot = reinterpret_cast<uint32_t*>(
                sym_buf_addrs[c] + static_cast<int64_t>(arrival_offset)) + rank_idx;
            *slot = generation;
        }
    }
}

// Host-side launcher
template <int HIDDEN>
void launch_mxfp4_quantize(
    const __nv_bfloat16* input,
    const int32_t* topk_ids,
    void* sym_buf_base,
    uint32_t num_tokens,
    uint32_t topk,
    uint32_t num_local_experts,
    uint32_t num_total_experts,
    uint32_t max_tokens_per_expert,
    uint32_t generation,
    const int64_t* sym_buf_addrs,
    uint32_t rank_idx,
    uint32_t num_ranks,
    cudaStream_t stream,
    int64_t* profile_clocks = nullptr,
    const int64_t* staging_addrs = nullptr,
    uint32_t profile_enabled = 0) {

    DispatchBufferLayout layout(num_total_experts, num_total_experts,
                                max_tokens_per_expert, HIDDEN);
    // Double-buffering: write this generation into buf[generation & 1] so we never clobber
    // the buffer a 1-iteration-slower consumer is still reading. Arrival slots (below) are
    // NOT double-buffered — they live at the fixed ready_flag_offset from the true base.
    void* data_base = layout.parity_base(sym_buf_base, generation & 1u);
    if (!use_tagged_generation_counts(generation, max_tokens_per_expert)) {
        cudaMemsetAsync(data_base, 0, layout.metadata_bytes(), stream);
    }

    constexpr int THREADS = 256;
    // Grid-stride launch: cap blocks at the max resident count (one wave) instead of
    // one block per token. Fewer blocks => fewer folded-arrival system fences and no
    // second wave exposing the remote-write round trip; the kernel loops over tokens.
    int dev = 0;
    cudaGetDevice(&dev);
    int num_sms = 1;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
    if (num_sms <= 0) num_sms = 1;
    int blocks_per_sm = 1;
    if (topk <= 4)
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, mxfp4_quantize_kernel<HIDDEN, 4>, THREADS, 0);
    else
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, mxfp4_quantize_kernel<HIDDEN, 8>, THREADS, 0);
    if (blocks_per_sm <= 0) blocks_per_sm = 1;
    int resident = blocks_per_sm * num_sms;
    int blocks = static_cast<int>(num_tokens) < resident ? static_cast<int>(num_tokens) : resident;
    if (blocks < 1) blocks = 1;

    // FUSED_ARRIVAL_IN_QUANT=1 folds the arrival push into the quant kernel's last
    // block (saves one launch + lets consumers observe arrival ~1 launch earlier).
    // Default keeps the standalone arrival_push_kernel below as the A/B baseline.
    // Read once (env is process-static); generation is still a runtime arg.
    static const bool fold_arrival_env = []() {
        const char* e = std::getenv("FUSED_ARRIVAL_IN_QUANT");
        return e != nullptr && e[0] == '1';
    }();
    const bool arrival_needed = (generation > 0);
    const bool fold_arrival = arrival_needed && fold_arrival_env;
    const uint32_t do_arrival = fold_arrival ? 1u : 0u;
    const uint64_t arrival_offset = layout.ready_flag_offset();

    // Uncache-fence folded arrival (default ON under DG_SFA_PUSH): every CTA performs a
    // lightweight uncache system release before retiring, and the last CTA performs one
    // more after the optional counts push before publishing arrival. DG_QUANT_LASTFENCE=0
    // reverts to the heavyweight cache-flushing system-fence path for A/B.
    static const int lastfence_env = []() {
        const char* e = std::getenv("DG_QUANT_LASTFENCE");
        return (e != nullptr && e[0] == '0') ? 0 : 1;
    }();
    const uint32_t last_fence = (fold_arrival && lastfence_env) ? 1u : 0u;

    if (topk <= 4) {
        mxfp4_quantize_kernel<HIDDEN, 4><<<blocks, THREADS, 0, stream>>>(
            input, topk_ids, data_base,
            num_tokens, topk, num_local_experts, num_total_experts,
            max_tokens_per_expert, HIDDEN, generation,
            profile_enabled ? profile_clocks : nullptr,
            sym_buf_addrs, rank_idx, num_ranks, arrival_offset, do_arrival, staging_addrs,
            last_fence);
    } else {
        mxfp4_quantize_kernel<HIDDEN, 8><<<blocks, THREADS, 0, stream>>>(
            input, topk_ids, data_base,
            num_tokens, topk, num_local_experts, num_total_experts,
            max_tokens_per_expert, HIDDEN, generation,
            profile_enabled ? profile_clocks : nullptr,
            sym_buf_addrs, rank_idx, num_ranks, arrival_offset, do_arrival, staging_addrs,
            last_fence);
    }

    if (arrival_needed && !fold_arrival) {
        arrival_push_kernel<<<1, 1, 0, stream>>>(
            sym_buf_addrs, rank_idx, num_ranks,
            arrival_offset, generation);
    }
}

}  // namespace deep_gemm
