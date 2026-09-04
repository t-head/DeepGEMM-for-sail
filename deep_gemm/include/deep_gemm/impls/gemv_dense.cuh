#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#ifndef HGGC_PRAGMA_UNROLL
#define HGGC_PRAGMA_UNROLL _Pragma("unroll")
#endif
#ifndef HGGC_DEVICE_ONLY
#define HGGC_DEVICE_ONLY __forceinline__ __device__
#endif

#include <hggc_bf16.h>

namespace deep_gemm {

// Dense GEMV kernel for decode shapes (m == 1):
//   y[n] = sum_k x[k] * W[n, k]
//   - x (lhs): [K], contiguous
//   - W (rhs): [N, K] row-major, contiguous
//   - y (out): [N], contiguous
//
// Two properties below are load-bearing (measured; do not "simplify" without
// re-benchmarking):
//   - The main-loop guard keeps the trip count independent of tid_x: every
//     thread runs the same number of unrolled iterations and the residual
//     loop only mops up the shared `k % (ksize_ept * k_per_thread)` tail.
//     A flat `id_k + stride <= K` guard pushes high-tid_x tails into the
//     non-unrolled residual loop as divergent serial iterations.
//   - The compiler's register allocator is sensitive to this source form
//     (scalar params, results[] epilogue, NUM_X import stride, tid_x
//     addressing): structurally equivalent rewrites land at 48 VRegs vs
//     32-40 here (~9 occupancy points).
//
// Correctness notes:
//   - dot_op converts operands to acc_type before the multiply (fp32
//     accumulation, matching the tile path).
//   - Out-of-range rows: BlockX >= WARP_SIZE early-returns whole warps
//     (exited threads are discounted from the block barrier); BlockX <
//     WARP_SIZE clamps the row instead so every lane stays alive for the
//     0xffffffff shuffle below. The store is guarded either way.

template <typename src_type, typename acc_type, int ept>
HGGC_DEVICE_ONLY void gemv_dense_dot_op(const src_type* src_a, const src_type* src_x, acc_type& val) {
    HGGC_PRAGMA_UNROLL
    for (int i = 0; i < ept; i++) {
        val = val + acc_type(src_a[i]) * acc_type(src_x[i]);
    }
}

template <typename src_type, typename load_type, int Num>
HGGC_DEVICE_ONLY void gemv_dense_import_data(load_type* vreg, const src_type* v, int load_size) {
    HGGC_PRAGMA_UNROLL
    for (int i = 0; i < Num; i++) {
        vreg[i] = *(reinterpret_cast<const load_type*>(v + i * load_size));
    }
}

struct GemvDenseArgs {
    int N;
    int K;
    const void* x_ptr;   // lhs [1, K]
    const void* w_ptr;   // rhs [N, K] row-major
    void* y_ptr;         // out [1, N]
    int64_t stride_wn;   // W row stride (elements, == K for contiguous rhs)
};

// Core GEMV loop, specialized to m_per_thread == k_slices == batch_xy == 1.
template <typename src_type, typename acc_type, typename dst_type, typename load_atype,
          typename load_xtype, int BlockX, int BlockY, int m_per_thread, int k_per_thread,
          int k_slices, int batch_xy>
__device__ void gemv_dense_op(int m, int k, acc_type alpha, const src_type* A, int lda,
                              const src_type* x, int incx, acc_type beta, const dst_type* c,
                              dst_type* y, int incy) {
    acc_type results[m_per_thread * batch_xy];

    int tid = threadIdx.y * BlockX + threadIdx.x;

    constexpr int BlockSize = BlockX * BlockY;
    constexpr int WarpCount = BlockSize / WARP_SIZE;
    constexpr int WarpsPerN = BlockX / WARP_SIZE;
    __shared__ acc_type shared[WarpCount];

    constexpr int alignment_a = sizeof(load_atype) / sizeof(src_type);
    constexpr int alignment_x = sizeof(load_xtype) / sizeof(src_type);
    constexpr int alignmentMax = alignment_a;

    constexpr int NUM_X = alignment_a / alignment_x;

    constexpr int ksize_ept = BlockX * alignmentMax * k_slices;

    int tid_x = k_slices > 1 ? blockIdx.y * BlockX + threadIdx.x : threadIdx.x;
    int id_m = BlockY > 1 ? blockIdx.x * BlockY + threadIdx.y : blockIdx.x;
    int id_k = tid_x * alignmentMax;

    if (BlockY > 1 && id_m >= m) {
        if (BlockX >= WARP_SIZE) {
            return;
        }
        // BlockX < WARP_SIZE: a warp spans several rows here, so exiting
        // single lanes would leave the 0xffffffff shuffle mask below
        // unsatisfied. Clamp the row instead and rely on the guarded store.
    }
    const int id_m_row = (BlockY > 1 && id_m >= m) ? m - 1 : id_m;

    const src_type* a = A + id_m_row * (long)lda;
    dst_type* yy = y + id_m * incy;
    const src_type* xx = x + id_k * incx;
    const src_type* a1 = a + id_k;
    const dst_type* cc = c + id_m * incy;  // only read when beta != 0 (never here)

    load_atype vreg_a[k_per_thread];
    load_xtype vreg_x[NUM_X * k_per_thread];
    acc_type accum = 0.0;

    if (k_per_thread > 1) {
        for (; (id_k - tid_x * alignmentMax + ksize_ept * k_per_thread) <= k;
             id_k += ksize_ept * k_per_thread) {
            HGGC_PRAGMA_UNROLL
            for (int loop = 0; loop < k_per_thread; loop++) {
                gemv_dense_import_data<src_type, load_xtype, NUM_X>(vreg_x + loop, xx,
                                                                    alignment_x * incx);
                gemv_dense_import_data<src_type, load_atype, 1>(vreg_a + loop, a1, alignment_a);
                xx += ksize_ept * incx;
                a1 += ksize_ept;

                gemv_dense_dot_op<src_type, acc_type, alignmentMax>(
                    (const src_type*)(vreg_a + loop), (const src_type*)(vreg_x + loop), accum);
            }
        }
    }

    // process the residual k
    for (; id_k < k; id_k += ksize_ept) {
        gemv_dense_import_data<src_type, load_xtype, NUM_X>(vreg_x, xx, alignment_x * incx);
        gemv_dense_import_data<src_type, load_atype, 1>(vreg_a, a1, alignment_a);
        xx += ksize_ept * incx;
        a1 += ksize_ept;

        gemv_dense_dot_op<src_type, acc_type, alignmentMax>((const src_type*)vreg_a,
                                                            (const src_type*)vreg_x, accum);
    }

    // warp reduce
    constexpr int SHFL_THREAD = BlockX > WARP_SIZE ? WARP_SIZE : BlockX;
    HGGC_PRAGMA_UNROLL
    for (int offset = (SHFL_THREAD >> 1); offset > 0; offset >>= 1) {
        acc_type temp = __shfl_down_sync(0xffffffff, accum, offset);
        accum += temp;
    }

    if (WarpsPerN > 1) {
        if (tid % WARP_SIZE == 0) {
            shared[tid / WARP_SIZE] = accum;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && id_m < m) {
        if (WarpsPerN > 1) {
            HGGC_PRAGMA_UNROLL
            for (int i = 1; i < WarpsPerN; i++) {
                accum += shared[tid / WARP_SIZE + i];
            }
        }

        if (beta == acc_type(0)) {
            results[0] = acc_type(alpha * accum);
        } else {
            results[0] = acc_type(alpha * accum + beta * acc_type(*cc));
        }
        *yy = results[0];
    }
}

// JIT entry point: folds the decode-path constants (alpha == 1, beta == 0,
// unit strides, no C operand) into the core above.
template <typename src_type, typename dst_type, typename acc_type, typename load_atype,
          typename load_xtype, int BlockX, int BlockY, int k_per_thread>
__device__ void gemv_dense_kernel_impl(const GemvDenseArgs args) {
    gemv_dense_op<src_type, acc_type, dst_type, load_atype, load_xtype, BlockX, BlockY,
                  /*m_per_thread=*/1, k_per_thread, /*k_slices=*/1, /*batch_xy=*/1>(
        args.N, args.K, acc_type(1), (const src_type*)args.w_ptr, (int)args.stride_wn,
        (const src_type*)args.x_ptr, /*incx=*/1, acc_type(0), /*c=*/nullptr,
        (dst_type*)args.y_ptr, /*incy=*/1);
}

}  // namespace deep_gemm

#pragma clang diagnostic pop
