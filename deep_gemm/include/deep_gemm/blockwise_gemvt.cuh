#pragma once
#include "gemvt.cuh"

namespace deep_gemm {

struct BlockWiseGemvtArgs {
    int N;
    int K;
    void * a_ptr;
    void * b_ptr;
    void * c_ptr;

    //use for blockwise quant
    const float* scaleB {nullptr};
    const float* scaleA {nullptr};

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
};

template <typename src_type, typename acc_type, int ept, typename CvtType = float>
HGGC_DEVICE_ONLY void cvt_dot_op(src_type *src_a, src_type *src_x, acc_type &val) {
    HGGC_PRAGMA_UNROLL
    for (int i = 0; i < ept; i++) {
        val = val + acc_type(CvtType(src_a[i]) * CvtType(src_x[i]));
    }
    return;
}

template <typename src_type, typename dst_type, typename acc_type,
          typename load_atype, typename load_btype,
          int BlockSize, int ThreadPerN = 32, int NPerThread = 1, int NUM_UNROLL=1,
          int GROUP_SIZE_M = 1, int Stages = 2>
__global__ void batched_blockwise_gemvt_kernel_small_k(const BlockWiseGemvtArgs args) {
  constexpr int WarpCount = BlockSize / WARP_SIZE;
  constexpr int NPerBlock = NPerThread * BlockSize / ThreadPerN;
  constexpr int NLoopStep = BlockSize / ThreadPerN;

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
  constexpr int TileN = ThreadPerN * alignmentB;
  __shared__ src_type shared[Stages][NLoopStep][TileN];

  int offs_token = pid_m;
  int off_expert = *(args.expert_ids_ptr + pid_m);

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;

  int tid_k = tid % ThreadPerN;
  int tid_n = tid / ThreadPerN;
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
    cvt_dot_op<src_type, acc_type, alignmentMax>((src_type *)&vreg_a,
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
  float scale_ab;
  if (tid_k == 0) {
    // load scale
    if constexpr (sizeof(src_type) == 1) {
      float alpha_row = args.scaleA[id_k * args.stride_am / 128 + offs_token];
      float alpha_col = args.scaleB[off_expert * args.stride_be / 128 / 128 + id_n / 128];
      scale_ab = alpha_row * alpha_col;
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

  if (tid_k == 0) {
    HGGC_PRAGMA_UNROLL
    for (int nloop = 0; nloop < NPerThread; nloop++) {
        *(out + nloop * NLoopStep)  = (dst_type)(accum[nloop] * scale_ab);
    }
  }
}

template <typename src_type, typename dst_type, typename acc_type,
          uint32_t SHAPE_N, uint32_t SHAPE_K, int32_t kNumGroups,
          int ThreadPerN, int NPerThread, int NUM_UNROLL,
          int SWZL_SIZE_M, int BlockSize = 256, bool SMALL_K = false>
class BlockWiseGemvt {
    using load_atype = int4;
    using load_btype = int4;

public:
    BlockWiseGemvt() = default;

    static void run(dst_type* gmem_d, int* grouped_layout,
                    uint32_t shape_m, src_type* gmem_a, src_type* gmem_b,
                    hggcStream_t stream, const float* lhs_scale = nullptr, const float* rhs_scale = nullptr) {

        BlockWiseGemvtArgs args;
        args.N = SHAPE_N;
        args.K = SHAPE_K;
        args.a_ptr = (void *)gmem_a;
        args.b_ptr = (void *)gmem_b;
        args.c_ptr = (void *)gmem_d;

        args.scaleB = rhs_scale;
        args.scaleA = lhs_scale;
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
        else if constexpr (std::is_same_v<src_type, __hg_fp8_e4m3>) {
            data_type = "fp8";
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
        // for small k
        {
            // launch small_k gemm_v
            size_t grid_x = args.num_tokens;
            constexpr int NPerBlock = NPerThread * BlockSize / ThreadPerN;
            size_t grid_y = args.N / NPerBlock;
            int total_blocks = grid_x * grid_y;
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

            auto device_func = batched_blockwise_gemvt_kernel_small_k<src_type, dst_type, acc_type, load_atype, load_btype,
                                                            BlockSize, ThreadPerN, NPerThread, 1, SWZL_SIZE_M, 5>;

            char *pEnv_params = std::getenv("show_log");
            if (pEnv_params && isdigit(*pEnv_params)) {
                hggcFuncAttributes attr;
                hggcFuncGetAttributes(&attr, device_func);

                printf("[GemV-Small-FP8:]\n");
                printf("group:%d, problem:[%d, %d, %d]\n",
                    kNumGroups, args.num_tokens, args.N, args.K);
                printf("BlockSize:%d, NPerThread:%d, ThreadPerN:%d, NPerBlock:%d, SWZL_SIZE_M:%d\n",
                    BlockSize, NPerThread, ThreadPerN, NPerBlock, SWZL_SIZE_M);

                printf("threadblock_count:%d, vreg:%d, stack:%d\n", total_blocks, int(attr.numRegs), int(attr.localSizeBytes));

            }

            ProfilingInterface::Instance().instrument(true, dg_prof_params);
            device_func<<<grid_x * grid_y, BlockSize, 0, stream>>>(args);
            ProfilingInterface::Instance().instrument(false, dg_prof_params);

        }
    }
};

}  // namespace deepgemm
