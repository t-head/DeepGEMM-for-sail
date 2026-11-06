#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#define WARP_SIZE 32
#define HGGC_PRAGMA_UNROLL _Pragma("unroll")
#define HGGC_DEVICE_ONLY __forceinline__ __device__

#include <hggc_bf16.h>
#include <hggc_fp16.h>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/tix/cp_async.cuh>

namespace deep_gemm {

struct GemvtArgs {
    int N;
    int K;
    void * a_ptr;
    void * b_ptr;
    void * c_ptr;

    //use for int8
    void * topk_weights_ptr {nullptr};
    const float* alphaCol {nullptr};
    const float* alphaRow {nullptr};

    int64_t num_tokens;
    int num_experts;
    int* expert_ids_ptr;

    int64_t stride_am;
    int64_t stride_ak;
    int64_t stride_be;
    int64_t stride_bk;
    int64_t stride_bn;
    int64_t stride_cm;
    int64_t stride_cn;
    int64_t stride_asm;
    int64_t stride_ask;
    int64_t stride_bse;
    int64_t stride_bsk;
    int64_t stride_bsn;
    int64_t total_blocks;
};

template <typename src_type, typename load_type, int Num>
HGGC_DEVICE_ONLY void import_data(load_type *vreg, const src_type *v, int load_size) {
    // TODO: checked again.
    HGGC_PRAGMA_UNROLL
    for (int i = 0; i < Num; i++) {
        vreg[i] = *(reinterpret_cast<const load_type *>(v + i * load_size));
    }
    return;
}

template <typename src_type, typename acc_type, int ept>
HGGC_DEVICE_ONLY void dot_op(src_type *src_a, src_type *src_x, acc_type &val) {
    HGGC_PRAGMA_UNROLL
    for (int i = 0; i < ept; i++) {
    val = val + acc_type(src_a[i] * src_x[i]);
    }
    return;
}

template <typename src_type, typename dst_type, typename acc_type,
          typename load_atype, typename load_btype,
          int BlockSize, int ThreadPerN = 32, int NPerThread = 1, int NUM_UNROLL=1,
          int GROUP_SIZE_M = 1, int Stages = 2>
__device__ void batched_gemvt_kernel_small_k_impl(const GemvtArgs args) {
  constexpr int WarpsPerN = ThreadPerN / WARP_SIZE;
  constexpr int WarpCount = BlockSize / WARP_SIZE;
  constexpr int NPerBlock = NPerThread * BlockSize / ThreadPerN;
  constexpr int NLoopStep = BlockSize / ThreadPerN;

  acc_type results[NPerThread];

  int num_pid_m = args.num_tokens;
  int num_pid_n = math::ceil_div(args.N, NPerBlock);
  int num_pid_in_group = GROUP_SIZE_M * num_pid_n;
  int group_id = blockIdx.x / num_pid_in_group;
  int first_pid_m = group_id * GROUP_SIZE_M;
  int group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M);
  int pid_m = first_pid_m + ((blockIdx.x % num_pid_in_group) % group_size_m);
  int pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;

  constexpr int alignmentB = sizeof(load_btype) / sizeof(src_type);
  constexpr int alignmentA = sizeof(load_atype) / sizeof(src_type);
  constexpr int alignmentMax = alignmentB;
  static_assert(alignmentB >= alignmentA, "only support alignmentB >= alignmentA");

  // for mutlistage B
  constexpr int TileM = ThreadPerN * alignmentB;
  __shared__ src_type shared[Stages][NLoopStep][TileM];

  int offs_token = pid_m;
  int off_expert = *(args.expert_ids_ptr + pid_m);

  int tid = threadIdx.x;
  int tid_k = tid % ThreadPerN;
  int tid_n = (ThreadPerN == 32) ? __ppu_read_firstlane(tid / ThreadPerN) : tid / ThreadPerN;
  int id_n = pid_n * NPerBlock + tid_n;
  int id_k = tid_k * alignmentMax;

  if ( off_expert >= args.num_experts
        || off_expert < 0
        || pid_m >= args.num_tokens
        || (id_n + (NPerThread - 1) * NLoopStep) >= args.N
        || id_k >= args.K) {
    return;
  }

  load_atype vreg_a;
  load_btype vreg_b[NPerThread];
  acc_type accum[NPerThread];

  HGGC_PRAGMA_UNROLL
  for (int i = 0; i < NPerThread; i++) {
    accum[i] = 0;
  }

  dst_type * out = (dst_type*)args.c_ptr + offs_token * args.stride_cm + id_n;

  const src_type *a_ptr_start = (src_type*)args.a_ptr + offs_token * args.stride_am + id_k;
  const src_type *b_ptr_start = (src_type*)args.b_ptr + off_expert * args.stride_be + id_n * args.stride_bn + id_k;

  // load A to vreg
  import_data<src_type, load_atype, 1>(&vreg_a, a_ptr_start, alignmentA);


  // load B to shared
  for (int stage = 0; stage < Stages; stage++) {
    tix::LdgSts128(&(shared[stage][tid_n][tid_k * alignmentB]), b_ptr_start + stage * NLoopStep * args.stride_bn);
    tix::LdgStsGroupCommit();
  }
  tix::LdgStsGroupWait<Stages - 1>();

  HGGC_PRAGMA_UNROLL
  for (int nloop = 0; nloop < NPerThread; nloop++) {
    // load B to vreg
    int current_stage = nloop % Stages;
    import_data<src_type, load_btype, 1>(vreg_b + nloop,
        &(shared[current_stage][tid_n][tid_k * alignmentB]), alignmentB);
    // import_data<src_type, load_btype, 1>(vreg_b + nloop,
    //   b_ptr_start + nloop * NLoopStep * args.stride_bn, alignmentB);

    // auto current_b_ptr = reinterpret_cast<src_type*>(vreg_b + nloop);
    // if (current_b_ptr[0] != shared[current_stage][tid_n][tid_k * alignmentB]) {
    //   printf("tid = %d, nloop = %d, shared[%d][%d][%d] = %.4f, current_b[0] = %.4f\n",
    //     tid, nloop, current_stage, tid_n, tid_k * alignmentB,
    //     float(shared[current_stage][tid_n][tid_k * alignmentB]),
    //     (float)current_b_ptr[0]);
    // }
    dot_op<src_type, acc_type, alignmentMax>((src_type *)&vreg_a,
                                             (src_type *)(vreg_b + nloop),
                                             accum[nloop]);
    tix::LdgStsGroupWait<Stages - 2>();

    int load_next_n = nloop + Stages;
    if (nloop < NPerThread - 2) {
      // load next B to shared
      int next_stage = load_next_n % Stages;
      auto b_ptr_next = b_ptr_start + load_next_n * NLoopStep * args.stride_bn;
      tix::LdgSts128(&(shared[next_stage][tid_n][tid_k * alignmentB]), b_ptr_next, load_next_n < NPerThread);
      tix::LdgStsGroupCommit();
    }
  }

  float alpha_row;
  float alpha_col[NPerThread];
  if (tid_k == 0) {
    // load scale
    if constexpr (sizeof(src_type) == 1) {
      alpha_row = args.alphaRow[offs_token];
      HGGC_PRAGMA_UNROLL
      for (int nloop = 0; nloop < NPerThread; nloop++) {
        alpha_col[nloop] = args.alphaCol[off_expert * args.N + id_n + nloop * NLoopStep];
      }
    }
  }

  // warp reduce
  constexpr int SHFL_THREAD = ThreadPerN > WARP_SIZE ? WARP_SIZE : ThreadPerN;
  HGGC_PRAGMA_UNROLL
  for (int offset = (SHFL_THREAD >> 1); offset > 0; offset >>= 1) {
    HGGC_PRAGMA_UNROLL
    for (int nloop = 0; nloop < NPerThread; nloop++) {
      acc_type temp = __shfl_down_sync(0xffffffff, accum[nloop], offset);
      accum[nloop] += temp;
    }
  }

  acc_type result;
  if (tid_k == 0) {
    // scale for int8
    if constexpr (sizeof(src_type) == 1) {
      HGGC_PRAGMA_UNROLL
      for (int nloop = 0; nloop < NPerThread; nloop++) {
        *(out + nloop * NLoopStep)  = (dst_type)(accum[nloop] * alpha_row * alpha_col[nloop]);
      }

    } else {
      HGGC_PRAGMA_UNROLL
      for (int nloop = 0; nloop < NPerThread; nloop++) {
        *(out + nloop * NLoopStep)  = (dst_type)(accum[nloop]);
      }
    }
  }
}


template <typename src_type, typename dst_type, typename acc_type,
          typename load_atype, typename load_btype,
          int BlockSize, int ThreadPerN = 32, int NPerThread = 1, int NUM_UNROLL=1,
          int GROUP_SIZE_M = 1>
__device__ void batched_gemvt_kernel_impl(const GemvtArgs args) {
    constexpr int BlockM = 1;
    constexpr int WarpsPerN = ThreadPerN / WARP_SIZE;
    //   constexpr const int NPerThread = 1;
    constexpr int WarpCount = BlockSize / WARP_SIZE;
    constexpr int NPerBlock = NPerThread * BlockSize / ThreadPerN;
    constexpr int NLoopStep = BlockSize / ThreadPerN;
    __shared__ acc_type shared[WarpCount * BlockM];

    acc_type results[NPerThread];

    constexpr int alignmentB = sizeof(load_btype) / sizeof(src_type);
    constexpr int alignmentA = sizeof(load_atype) / sizeof(src_type);
    constexpr int alignmentMax = alignmentB;
    // if (alignmentA > alignmentB) {
    //   FT_LOG_ERROR("only support alignmentA <= alignmentB");
    // }
    constexpr int NUM_X = alignmentB / alignmentA;
    constexpr int ksize_ept = ThreadPerN * alignmentMax;

    int tid = threadIdx.x;
    int tid_k = tid % ThreadPerN;
    int tid_n = tid / ThreadPerN;

    load_atype vreg_a[NUM_X * NUM_UNROLL * BlockM];
    load_btype vreg_b[NUM_UNROLL * NPerThread];
    acc_type accum[NUM_UNROLL * BlockM * NPerThread];

    int num_pid_m = args.num_tokens;
    int num_pid_n = math::ceil_div(args.N, NPerBlock);
    int num_pid_in_group = GROUP_SIZE_M * num_pid_n;

    int block_id = blockIdx.x;
    if(block_id < args.total_blocks) {
        int group_id = block_id / num_pid_in_group;
        int first_pid_m = group_id * GROUP_SIZE_M;
        int group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M);
        int pid_m = first_pid_m + ((block_id % num_pid_in_group) % group_size_m);
        int pid_n = (block_id % num_pid_in_group) / group_size_m;

        int offs_token = pid_m * BlockM;
        int off_expert = *(args.expert_ids_ptr + pid_m);
        int id_n = pid_n * NPerBlock + tid_n;

        if ( off_expert >= args.num_experts
            || off_expert < 0
            || pid_m >= args.num_tokens
            || (id_n + (NPerThread - 1) * NLoopStep) >= args.N) {
            return;
        }

        // if (tid == 0) {
        //   printf("total_blocks = %d, block_id = %d, pid_m = %d, id_n = %d, off_expert = %d, num_experts = %d\n",
        //     args.total_blocks, block_id, pid_m, id_n, off_expert, args.num_experts);
        // }

        HGGC_PRAGMA_UNROLL
        for (int i = 0; i < NUM_UNROLL * BlockM * NPerThread; i++) {
            accum[i] = 0;
        }

        int id_k = tid_k * alignmentMax;
        dst_type * out = (dst_type*)args.c_ptr + offs_token * args.stride_cm + id_n;

        const src_type *a_ptr_start = (src_type*)args.a_ptr + offs_token * args.stride_am + id_k;
        const src_type *b_ptr_start = (src_type*)args.b_ptr + off_expert * args.stride_be + id_n * args.stride_bn + id_k;

        for (; (id_k - tid_k * alignmentMax + ksize_ept * NUM_UNROLL) <= args.K;
            id_k += ksize_ept * NUM_UNROLL) {
            HGGC_PRAGMA_UNROLL
            for (int loop = 0; loop < NUM_UNROLL; loop++) {
                import_data<src_type, load_atype, NUM_X>(vreg_a + loop, a_ptr_start, alignmentA);
                a_ptr_start += ksize_ept;

                HGGC_PRAGMA_UNROLL
                for (int nloop = 0; nloop < NPerThread; nloop++) {
                    import_data<src_type, load_btype, 1>(vreg_b + loop + nloop * NUM_UNROLL,
                        b_ptr_start + nloop * NLoopStep * args.stride_bn, alignmentB);
                    dot_op<src_type, acc_type, alignmentMax>((src_type *)(vreg_a + loop),
                                                            (src_type *)(vreg_b + loop + nloop * NUM_UNROLL),
                                                            accum[nloop * NUM_UNROLL + loop]);
                }
                b_ptr_start += ksize_ept;
            }
        }

        // thread reduce
        acc_type accum_sum[NPerThread];
        HGGC_PRAGMA_UNROLL
        for (int nloop = 0; nloop < NPerThread; nloop++) {
            accum_sum[nloop] = 0;
            HGGC_PRAGMA_UNROLL
            for (int i = 0; i < NUM_UNROLL; i++) {
                accum_sum[nloop] += accum[nloop * NUM_UNROLL + i];
            }
        }

        // warp reduce
        constexpr int SHFL_THREAD = ThreadPerN > WARP_SIZE ? WARP_SIZE : ThreadPerN;
        HGGC_PRAGMA_UNROLL
        for (int offset = (SHFL_THREAD >> 1); offset > 0; offset >>= 1) {
            HGGC_PRAGMA_UNROLL
            for (int nloop = 0; nloop < NPerThread; nloop++) {
                acc_type temp = __shfl_down_sync(0xffffffff, accum_sum[nloop], offset);
                accum_sum[nloop] += temp;
            }
        }

        acc_type result;
        if (tid_k == 0) {
            // scale for int8
            if constexpr (sizeof(src_type) == 1) {
                float alpha_row = args.alphaRow[offs_token];
                HGGC_PRAGMA_UNROLL
                for (int nloop = 0; nloop < NPerThread; nloop++) {
                    float alpha_col = args.alphaCol[off_expert * args.N + id_n + nloop * NLoopStep];
                    *(out + nloop * NLoopStep)  = (dst_type)(accum_sum[nloop] * alpha_row * alpha_col);
                }
            } else {
                HGGC_PRAGMA_UNROLL
                for (int nloop = 0; nloop < NPerThread; nloop++) {
                    *(out + nloop * NLoopStep)  = (dst_type)(accum_sum[nloop]);
                }
            }
        }
    }
}

}  // namespace deepgemm
