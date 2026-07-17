#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#define WARP_SIZE 32
#define HGGC_PRAGMA_UNROLL _Pragma("unroll")
#define HGGC_DEVICE_ONLY __forceinline__ __device__

#include <iostream>
#include <hggc_bf16.h>
#include <hggc_fp16.h>
#include "utils.cuh"
#include "profiling_interface.hpp"

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

__device__ __forceinline__
uint32_t SmemU32Addr(const void *smemptr) {
    uint32_t u32addr;
    asm (
        "{.reg .u64 u64addr;\n"
        " cvta.to.shared.u64 u64addr, %1;\n"
        " cvt.u32.u64 %0, u64addr; }\n"
        : "=r"(u32addr)
        : "l"(smemptr)
    );
    return u32addr;
}

__device__ __forceinline__
void LdgSts128(const void* smemPtr,
               const void *gmemPtr,
               bool guard = true) {
    auto smemAddr = SmemU32Addr(smemPtr);
    asm volatile (
        "{.reg.pred p;\n"
        " ppu.cmpp.ne.b32 p, %2, 0;\n"
        " @p ppu.cp.async.cg.shared.global [%0], [%1], 16;}\n"
        :
        : "r"(smemAddr), "l"(gmemPtr), "r"((int)guard)
    );
}

__device__ __forceinline__
void LdgStsGroupCommit() {
    asm volatile ("ppu.cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__
void LdgStsGroupWait() {
    asm volatile ("ppu.cp.async.wait_group %0;\n" : : "n"(N));
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
  int num_pid_n = ceil_div(args.N, NPerBlock);
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
    LdgSts128(&(shared[stage][tid_n][tid_k * alignmentB]), b_ptr_start + stage * NLoopStep * args.stride_bn);
    LdgStsGroupCommit();
  }
  LdgStsGroupWait<Stages - 1>();

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
    LdgStsGroupWait<Stages - 2>();

    int load_next_n = nloop + Stages;
    if (nloop < NPerThread - 2) {
      // load next B to shared
      int next_stage = load_next_n % Stages;
      auto b_ptr_next = b_ptr_start + load_next_n * NLoopStep * args.stride_bn;
      LdgSts128(&(shared[next_stage][tid_n][tid_k * alignmentB]), b_ptr_next, load_next_n < NPerThread);
      LdgStsGroupCommit();
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
          int GROUP_SIZE_M = 1, int Stages = 2>
__global__ void batched_gemvt_kernel_small_k(const GemvtArgs args) {
    batched_gemvt_kernel_small_k_impl<src_type, dst_type, acc_type, load_atype, load_btype,
        BlockSize, ThreadPerN, NPerThread, NUM_UNROLL, GROUP_SIZE_M, Stages>(args);
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
    int num_pid_n = ceil_div(args.N, NPerBlock);
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

template <typename src_type, typename dst_type, typename acc_type,
          typename load_atype, typename load_btype,
          int BlockSize, int ThreadPerN = 32, int NPerThread = 1, int NUM_UNROLL=1,
          int GROUP_SIZE_M = 1>
__global__ void batched_gemvt_kernel(const GemvtArgs args) {
    batched_gemvt_kernel_impl<src_type, dst_type, acc_type, load_atype, load_btype,
        BlockSize, ThreadPerN, NPerThread, NUM_UNROLL, GROUP_SIZE_M>(args);
}

template <typename src_type, typename dst_type, typename acc_type,
          uint32_t SHAPE_N, uint32_t SHAPE_K, int32_t kNumGroups,
          int ThreadPerN, int NPerThread, int NUM_UNROLL,
          int SWZL_SIZE_M, int BlockSize = 256, bool SMALL_K = false>
class Gemvt {
    using load_atype = int4;
    using load_btype = int4;

public:
    Gemvt() = default;

    static void run(dst_type* gmem_d, int* grouped_layout,
                    uint32_t shape_m, src_type* gmem_a, src_type* gmem_b,
                    hggcStream_t stream, const float* lhs_scale = nullptr, const float* rhs_scale = nullptr) {

        GemvtArgs args;
        args.N = SHAPE_N;
        args.K = SHAPE_K;
        args.a_ptr = (void *)gmem_a;
        args.b_ptr = (void *)gmem_b;
        args.c_ptr = (void *)gmem_d;

        args.alphaCol = rhs_scale;
        args.alphaRow = lhs_scale;
        args.expert_ids_ptr = grouped_layout;
        args.num_tokens = shape_m;
        args.num_experts = kNumGroups;
        args.stride_am = SHAPE_K;
        args.stride_ak = 1;
        args.stride_be = SHAPE_N * SHAPE_K;
        args.stride_bk = 1;
        args.stride_bn = SHAPE_K;
        args.stride_cm = SHAPE_N;
        args.stride_cn = 1;

        DgProfParam dg_prof_params;
        // check src type
        std::string data_type = "";
        if constexpr (std::is_same_v<src_type, int8_t>) {
            data_type = "int8";
        }
        else if constexpr (std::is_same_v<src_type, __ppu_bfloat16>) {
            data_type = "bf16";
        }
        else {
            data_type = "unsupported";
        }
        if (ProfilingInterface::Instance().get_op_info()){
            dg_prof_params.set_params(
                GemmType::GroupedNoPad, true, data_type, kNumGroups, shape_m, SHAPE_N, SHAPE_K, 1,
                grouped_layout, stream
            );
        }

        if (SMALL_K == false) {
            size_t grid_x = shape_m;
            constexpr uint32_t NPerBlock = NPerThread * BlockSize / ThreadPerN;
            size_t grid_y = ceil_div(SHAPE_N, NPerBlock);
            args.total_blocks = grid_x * grid_y;
            // check GEMM_K alignment
            if(args.K % (NUM_UNROLL * ThreadPerN * sizeof(load_atype) / sizeof(src_type)) != 0) {
                 printf("K alignment mismatch, K = %d, NUM_UNROLL = %d, ThreadPerN = %d, sizeof(load_atype) = %d, sizeof(src_type) = %d",
                   args.K, NUM_UNROLL, ThreadPerN, sizeof(load_atype), sizeof(src_type));
                 return;
            }

            auto device_func = batched_gemvt_kernel<src_type, dst_type, acc_type, load_atype, load_btype,
                                BlockSize, ThreadPerN, NPerThread, NUM_UNROLL, SWZL_SIZE_M>;
            dim3 grid = grid_x * grid_y;

            char *pEnv_params = std::getenv("show_log");
            if (pEnv_params && isdigit(*pEnv_params)) {
                hggcFuncAttributes attr;
                hggcFuncGetAttributes(&attr, device_func);

                printf("[GemV-BF16:]\n");
                printf("group:%d, problem:[%d, %d, %d]\n",
                    kNumGroups, shape_m, SHAPE_N, SHAPE_K);
                printf("BlockSize:%d, NPerThread:%d, ThreadPerN:%d, NPerBlock:%d, NUM_UNROLL:%d, SWZL_SIZE_M:%d\n",
                    BlockSize, NPerThread, ThreadPerN, NPerBlock, NUM_UNROLL, SWZL_SIZE_M);
                printf("threadblock_count:%d, vreg:%d, stack:%d\n", args.total_blocks, int(attr.numRegs), int(attr.localSizeBytes));

            }
            ProfilingInterface::Instance().instrument(true, dg_prof_params);
            device_func<<<grid, BlockSize, 0, stream>>>(args);
            ProfilingInterface::Instance().instrument(false, dg_prof_params);
        } else {
            // launch small_k gemm_v
            size_t grid_x = args.num_tokens;
            constexpr int NPerBlock = NPerThread * BlockSize / ThreadPerN;
            size_t grid_y = args.N / NPerBlock;
            args.total_blocks = grid_x * grid_y;
            constexpr int MAX_K = NUM_UNROLL * ThreadPerN * sizeof(load_atype) / sizeof(src_type);
            constexpr int MIN_ALIGNMENT = 16 / sizeof(src_type); // for int4 copy

            if(args.K > MAX_K) {
                printf("unsupported K, K = %d, MAX_K = %d, NUM_UNROLL = %d, ThreadPerN = %d, sizeof(load_atype) = %d, sizeof(src_type) = %d",
                args.K, MAX_K, NUM_UNROLL, ThreadPerN, sizeof(load_atype), sizeof(src_type));
                return;
            }

            if (args.stride_am % MIN_ALIGNMENT != 0 || args.stride_bn % MIN_ALIGNMENT !=0) {
                printf("unsupported stride, stride_am = %d, stride_bn = %d\n", args.stride_am, args.stride_bn);
                return;
            }

            auto device_func = batched_gemvt_kernel_small_k<src_type, dst_type, acc_type, load_atype, load_btype,
                                                            BlockSize, ThreadPerN, NPerThread, 1, SWZL_SIZE_M, 5>;
            // printf("num_tokens = %d, N = %d, K = %d, top_k = %d, A = %p, B = %p, grid_x = %d, grid_y = %d",
            //     args.num_tokens, args.N, args.K, args.top_k, args.a_ptr, args.b_ptr, grid_x, grid_y);

            char *pEnv_params = std::getenv("show_log");
            if (pEnv_params && isdigit(*pEnv_params)) {
                hggcFuncAttributes attr;
                hggcFuncGetAttributes(&attr, device_func);

                printf("[GemV-Small-BF16:]\n");
                printf("group:%d, problem:[%d, %d, %d]\n",
                    kNumGroups, args.num_tokens, args.N, args.K);
                printf("BlockSize:%d, NPerThread:%d, ThreadPerN:%d, NPerBlock:%d, SWZL_SIZE_M:%d\n",
                    BlockSize, NPerThread, ThreadPerN, NPerBlock, SWZL_SIZE_M);

                printf("threadblock_count:%d, vreg:%d, stack:%d\n", args.total_blocks, int(attr.numRegs), int(attr.localSizeBytes));

            }

            ProfilingInterface::Instance().instrument(true, dg_prof_params);
            // batched_gemvt_kernel_small_k<src_type, dst_type, float, load_atype, load_btype,
            //     BlockSize, ThreadPerN, NPerThread, 1, SWZL_SIZE_M, Stages><<<grid_x * grid_y, BlockSize, 0, stream>>>(args);
            device_func<<<grid_x * grid_y, BlockSize, 0, stream>>>(args);
            ProfilingInterface::Instance().instrument(false, dg_prof_params);

        }
    }
};

}  // namespace deepgemm
