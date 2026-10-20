#pragma once

#include <deep_gemm/impls/gemvt.cuh>

namespace deep_gemm {

// A8W4 GEMV consumes the preprocess_a8w4_weight layout directly. One 16-byte rhs load holds one
// K32 group as four uint32 words; byte j of word g stores logical k = 4g+j in its low nibble and
// k = 16+4g+j in its high nibble, both as unsigned values biased by 8. The matching lhs bytes are
// word g of the first and of the second 16-byte half of the K32 group, so
//   sum(a * (u - 8)) = dp4a(a, u) - 8 * dp4a(a, 0x01010101)
// and every weight word needs only a mask (low half) or a shift plus mask (high half).
HGGC_DEVICE_ONLY int a8w4_dp4a(uint32_t a, uint32_t b, int c) {
    int d;
    asm("ppu.dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// Load and accumulate kNumK32 K32 groups starting at `k32_idx`, `k32_stride` groups apart. All
// loads are issued before the dot products so that they overlap. With kGuard, groups at or past
// `num_k32` are skipped and contribute zero (a zero lhs word adds nothing to either dp4a sum).
template <int kNumK32, bool kGuard, int NPerThread, int NLoopStep>
HGGC_DEVICE_ONLY void a8w4_gemvt_accumulate(const int8_t* a_row, const uint8_t* b_rows, int64_t stride_bn,
                                            int k32_idx, int k32_stride, int num_k32, int (&accum)[NPerThread],
                                            int& a_sum) {
    constexpr uint32_t kNibbleMask = 0x0f0f0f0f;
    constexpr uint32_t kByteOnes = 0x01010101;

    int4 a_frag[kNumK32][2];
    int4 b_frag[kNumK32][NPerThread];
    HGGC_PRAGMA_UNROLL
    for (int u = 0; u < kNumK32; u++) {
        const int k32 = k32_idx + u * k32_stride;
        if (!kGuard || k32 < num_k32) {
            a_frag[u][0] = *reinterpret_cast<const int4*>(a_row + k32 * 32);
            a_frag[u][1] = *reinterpret_cast<const int4*>(a_row + k32 * 32 + 16);
            HGGC_PRAGMA_UNROLL
            for (int nloop = 0; nloop < NPerThread; nloop++) {
                b_frag[u][nloop] = *reinterpret_cast<const int4*>(b_rows + nloop * NLoopStep * stride_bn + k32 * 16);
            }
        } else {
            a_frag[u][0] = a_frag[u][1] = make_int4(0, 0, 0, 0);
            HGGC_PRAGMA_UNROLL
            for (int nloop = 0; nloop < NPerThread; nloop++)
                b_frag[u][nloop] = make_int4(0, 0, 0, 0);
        }
    }

    HGGC_PRAGMA_UNROLL
    for (int u = 0; u < kNumK32; u++) {
        const auto* a_lo = reinterpret_cast<const uint32_t*>(&a_frag[u][0]);
        const auto* a_hi = reinterpret_cast<const uint32_t*>(&a_frag[u][1]);
        HGGC_PRAGMA_UNROLL
        for (int g = 0; g < 4; g++) {
            a_sum = a8w4_dp4a(a_lo[g], kByteOnes, a_sum);
            a_sum = a8w4_dp4a(a_hi[g], kByteOnes, a_sum);
        }
        HGGC_PRAGMA_UNROLL
        for (int nloop = 0; nloop < NPerThread; nloop++) {
            const auto* b_words = reinterpret_cast<const uint32_t*>(&b_frag[u][nloop]);
            HGGC_PRAGMA_UNROLL
            for (int g = 0; g < 4; g++) {
                accum[nloop] = a8w4_dp4a(a_lo[g], b_words[g] & kNibbleMask, accum[nloop]);
                accum[nloop] = a8w4_dp4a(a_hi[g], (b_words[g] >> 4) & kNibbleMask, accum[nloop]);
            }
        }
    }
}

// One block computes NPerBlock output columns of one lhs row. ThreadPerN threads split K in K32
// groups; the host guarantees N % NPerBlock == 0, K % 32 == 0 and 16-byte aligned operands.
template <int BlockSize, int ThreadPerN, int NPerThread, int NUM_UNROLL, int GROUP_SIZE_M>
__device__ void a8w4_batched_gemvt_kernel_impl(const GemvtArgs args) {
    static_assert(ThreadPerN <= WARP_SIZE && WARP_SIZE % ThreadPerN == 0, "K threads must stay inside a warp");
    static_assert(BlockSize % WARP_SIZE == 0, "Block size must be a multiple of the warp size");
    constexpr int NLoopStep = BlockSize / ThreadPerN;
    constexpr int NPerBlock = NPerThread * NLoopStep;

    const int block_id = blockIdx.x;
    if (block_id >= args.total_blocks)
        return;

    const int num_pid_m = args.num_tokens;
    const int num_pid_n = args.N / NPerBlock;
    const int num_pid_in_group = GROUP_SIZE_M * num_pid_n;
    const int group_id = block_id / num_pid_in_group;
    const int first_pid_m = group_id * GROUP_SIZE_M;
    const int group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M);
    const int pid_m = first_pid_m + ((block_id % num_pid_in_group) % group_size_m);
    const int pid_n = (block_id % num_pid_in_group) / group_size_m;

    // Rows without a valid expert are skipped for the whole block, like the A8W8 GEMV.
    const int off_expert = args.expert_ids_ptr[pid_m];
    if (off_expert < 0 || off_expert >= args.num_experts)
        return;

    const int tid_k = threadIdx.x % ThreadPerN;
    const int tid_n = threadIdx.x / ThreadPerN;
    const int id_n = pid_n * NPerBlock + tid_n;

    const auto* a_row = static_cast<const int8_t*>(args.a_ptr) + pid_m * args.stride_am;
    const auto* b_rows = static_cast<const uint8_t*>(args.b_ptr) + off_expert * args.stride_be + id_n * args.stride_bn;

    int accum[NPerThread];
    HGGC_PRAGMA_UNROLL
    for (int nloop = 0; nloop < NPerThread; nloop++)
        accum[nloop] = 0;
    int a_sum = 0;

    const int num_k32 = args.K / 32;
    int k32_idx = tid_k;
    for (; k32_idx + (NUM_UNROLL - 1) * ThreadPerN < num_k32; k32_idx += NUM_UNROLL * ThreadPerN) {
        a8w4_gemvt_accumulate<NUM_UNROLL, false, NPerThread, NLoopStep>(a_row, b_rows, args.stride_bn, k32_idx,
                                                                        ThreadPerN, num_k32, accum, a_sum);
    }
    // Fewer than NUM_UNROLL groups remain per thread: one predicated step keeps their loads together.
    if (NUM_UNROLL > 1 && k32_idx < num_k32) {
        a8w4_gemvt_accumulate<NUM_UNROLL, true, NPerThread, NLoopStep>(a_row, b_rows, args.stride_bn, k32_idx,
                                                                       ThreadPerN, num_k32, accum, a_sum);
    }

    // Remove the zero point before the reduction; it is linear in the per-thread partial sums.
    HGGC_PRAGMA_UNROLL
    for (int nloop = 0; nloop < NPerThread; nloop++)
        accum[nloop] -= 8 * a_sum;

    HGGC_PRAGMA_UNROLL
    for (int offset = ThreadPerN >> 1; offset > 0; offset >>= 1) {
        HGGC_PRAGMA_UNROLL
        for (int nloop = 0; nloop < NPerThread; nloop++)
            accum[nloop] += __shfl_down_sync(0xffffffff, accum[nloop], offset);
    }

    if (tid_k == 0) {
        const float alpha_row = args.alphaRow[pid_m];
        auto* out = static_cast<__ppu_bfloat16*>(args.c_ptr) + pid_m * args.stride_cm + id_n;
        HGGC_PRAGMA_UNROLL
        for (int nloop = 0; nloop < NPerThread; nloop++) {
            const float alpha_col = args.alphaCol[off_expert * args.N + id_n + nloop * NLoopStep];
            out[nloop * NLoopStep] = (__ppu_bfloat16)(accum[nloop] * alpha_row * alpha_col);
        }
    }
}

} // namespace deep_gemm
