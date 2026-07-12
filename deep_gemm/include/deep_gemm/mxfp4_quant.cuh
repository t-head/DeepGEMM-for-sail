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
    uint32_t do_arrival = 0) {

    constexpr int ELEMS_PER_THREAD = 8;  // 8 BF16 values = 1 int4 load
    constexpr int SCALE_GROUP_SIZE = 32;
    constexpr int LANES_PER_GROUP = SCALE_GROUP_SIZE / ELEMS_PER_THREAD;  // 4
    constexpr int FP4_INTS = HIDDEN / ELEMS_PER_THREAD;  // number of int-sized FP4 words
    constexpr int K_BLOCKS = (HIDDEN + 31) / 32;
    constexpr int K_SCALE_BLOCKS = (K_BLOCKS + 1) / 2;

    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const int lane_id = threadIdx.x % 32;

    DispatchBufferLayout layout(num_total_experts, num_total_experts,
                                max_tokens_per_expert, hidden_dim);

    const __nv_bfloat16* token_data = input + static_cast<int64_t>(token_idx) * hidden_dim;

    // Shared memory: quantized FP4 data + packed scales (quantize once, scatter many)
    __shared__ int s_fp4_data[FP4_INTS];
    __shared__ uint8_t s_scale_inv[K_BLOCKS];
    __shared__ uint16_t s_packed_scale[K_SCALE_BLOCKS];
    __shared__ uint32_t s_slot[MAX_TOPK];
    __shared__ int s_expert[MAX_TOPK];

    // Profiling: record clocks at phase boundaries (block 0 only)
    uint64_t _prof_t0 = 0, _prof_t1 = 0, _prof_t2 = 0;
    if (profile_clocks && blockIdx.x == 0 && threadIdx.x == 0) _prof_t0 = clock64();

    // ---- Phase 1: Quantize token data ONCE into shared memory ----
    // Compile-time stride (launch is always THREADS=256) so the compiler can unroll
    // the fixed iteration count. Loads are HOISTED into a register array first so the
    // P1_ITERS independent int4 fetches issue back-to-back and overlap (MLP): the
    // per-iteration `load; compute` schedule made the compiler emit `s.wait vldcnt(0)`
    // after every load (serializing the memory latency, acu No-Eligible ~48%);
    // separating loads from compute lets it wait with staggered vldcnt instead.
    constexpr int THREADS = 256;
    constexpr int P1_ITERS = (FP4_INTS + THREADS - 1) / THREADS;
    const int4* token_v = reinterpret_cast<const int4*>(token_data);
    int4 raw[P1_ITERS];
    #pragma unroll
    for (int it = 0; it < P1_ITERS; ++it) {
        const int i = it * THREADS + threadIdx.x;
        if (i < FP4_INTS) raw[it] = token_v[i];
    }
    #pragma unroll
    for (int it = 0; it < P1_ITERS; ++it) {
        const int i = it * THREADS + threadIdx.x;
        if (i >= FP4_INTS) continue;
        int4 int4_val = raw[it];
        __nv_bfloat162 local_v2[ELEMS_PER_THREAD / 2];

        __nv_bfloat162 amax2 = __float2bfloat162_rn(0.0f);
        for (int j = 0; j < ELEMS_PER_THREAD / 2; ++j) {
            local_v2[j] = reinterpret_cast<const __nv_bfloat162*>(&int4_val)[j];
            amax2 = __hmax2(amax2, __habs2(local_v2[j]));
        }
        __nv_bfloat16 amax = __hmax(amax2.x, amax2.y);

        amax = __hmax(amax, __shfl_xor_sync(0xffffffff, amax, 2, 4));
        amax = __hmax(amax, __shfl_xor_sync(0xffffffff, amax, 1, 4));

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
        int int_value;
        uint8_t* fp4x2_values = reinterpret_cast<uint8_t*>(&int_value);
        for (int j = 0; j < ELEMS_PER_THREAD / 2; ++j) {
            __nv_bfloat162 scaled = __hmul2(local_v2[j], scale2);
            float2 f2 = __bfloat1622float2(scaled);
            fp4x2_values[j] = cvt_f32x2_to_fp4x2(f2.y, f2.x);
        }

        s_fp4_data[i] = int_value;
    }
    __syncthreads();

    // Pack uint8 scales into uint16 pairs in shared memory
    for (int j = threadIdx.x; j < K_SCALE_BLOCKS; j += blockDim.x) {
        uint8_t s0 = s_scale_inv[2 * j];
        uint8_t s1 = (2 * j + 1 < K_BLOCKS) ? s_scale_inv[2 * j + 1] : 0;
        s_packed_scale[j] = static_cast<uint16_t>(s0) | (static_cast<uint16_t>(s1) << 8);
    }
    __syncthreads();

    if (profile_clocks && blockIdx.x == 0 && threadIdx.x == 0) _prof_t1 = clock64();

    // ---- Phase 2: Scatter-copy quantized data to topk expert slots ----
    // Batch all atomicAdds upfront (thread 0 only), then ONE sync
    if (threadIdx.x == 0) {
        uint32_t* counts = layout.expert_token_counts_ptr(sym_buf_base);
        for (int t = 0; t < topk; ++t) {
            int expert_idx = topk_ids[token_idx * topk + t];
            s_expert[t] = expert_idx;
            if (expert_idx >= 0) {
                s_slot[t] = atomic_add_generation_count(
                    &counts[expert_idx], generation, max_tokens_per_expert);
            } else {
                s_slot[t] = max_tokens_per_expert;
            }
        }
    }
    __syncthreads();

    // Vectorized scatter: int4 (16-byte) writes for FP4 data
    constexpr int FP4_INT4S = FP4_INTS / 4;
    const int4* s_fp4_v = reinterpret_cast<const int4*>(s_fp4_data);

    for (int t = 0; t < topk; ++t) {
        int expert_idx = s_expert[t];
        uint32_t slot = s_slot[t];

        if (expert_idx >= 0 && slot < max_tokens_per_expert) {
            int4* fp4_out = reinterpret_cast<int4*>(
                layout.fp4_data_ptr(sym_buf_base, expert_idx) +
                static_cast<int64_t>(slot) * (hidden_dim / 2));
            for (int i = threadIdx.x; i < FP4_INT4S; i += blockDim.x) {
                fp4_out[i] = s_fp4_v[i];
            }

            uint16_t* scale_out = layout.scale_ptr(sym_buf_base, expert_idx);
            for (int j = threadIdx.x; j < K_SCALE_BLOCKS; j += blockDim.x) {
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
        }
        // No __syncthreads() needed: s_fp4_v, s_packed_scale, s_expert, s_slot are all read-only here
    }

    if (profile_clocks && blockIdx.x == 0 && threadIdx.x == 0) {
        _prof_t2 = clock64();
        profile_clocks[0] = _prof_t1 - _prof_t0;  // Phase 1: quantize to SMEM
        profile_clocks[1] = _prof_t2 - _prof_t1;  // Phase 2: scatter to global
        profile_clocks[2] = _prof_t2 - _prof_t0;  // Total
    }

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
        __threadfence();
        __shared__ bool s_is_last_block;
        if (threadIdx.x == 0) {
            uint32_t ticket = atomicInc(&g_quant_arrival_retire, gridDim.x - 1);
            s_is_last_block = (ticket == gridDim.x - 1);
        }
        __syncthreads();
        if (s_is_last_block && threadIdx.x == 0) {
            __threadfence_system();
            for (uint32_t c = 0; c < num_ranks; ++c) {
                uint32_t* slot = reinterpret_cast<uint32_t*>(
                    sym_buf_addrs[c] + static_cast<int64_t>(arrival_offset)) + rank_idx;
                *slot = generation;
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
    int64_t* profile_clocks = nullptr) {

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
    int blocks = num_tokens;

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

    if (topk <= 4) {
        mxfp4_quantize_kernel<HIDDEN, 4><<<blocks, THREADS, 0, stream>>>(
            input, topk_ids, data_base,
            num_tokens, topk, num_local_experts, num_total_experts,
            max_tokens_per_expert, HIDDEN, generation, profile_clocks,
            sym_buf_addrs, rank_idx, num_ranks, arrival_offset, do_arrival);
    } else {
        mxfp4_quantize_kernel<HIDDEN, 8><<<blocks, THREADS, 0, stream>>>(
            input, topk_ids, data_base,
            num_tokens, topk, num_local_experts, num_total_experts,
            max_tokens_per_expert, HIDDEN, generation, profile_clocks,
            sym_buf_addrs, rank_idx, num_ranks, arrival_offset, do_arrival);
    }

    if (arrival_needed && !fold_arrival) {
        arrival_push_kernel<<<1, 1, 0, stream>>>(
            sym_buf_addrs, rank_idx, num_ranks,
            arrival_offset, generation);
    }
}

}  // namespace deep_gemm
