#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#define WARP_SIZE 32
#define HGGC_PRAGMA_UNROLL _Pragma("unroll")
#define HGGC_DEVICE_ONLY __forceinline__ __device__

#include <iostream>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "utils.cuh"

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

HGGC_DEVICE_ONLY int cdiv(int a, int b) {
    return (a + b - 1) / b;
}


template <typename src_type, typename dst_type,
          typename load_atype, typename load_btype,
          int BlockSize, int ThreadPerN = 32, int NPerThread = 1, int NUM_UNROLL=1,
          int GROUP_SIZE_M = 1>
__global__ void batched_gemvt_kernel(const GemvtArgs args) {
    constexpr int BlockM = 1;
    using acc_type = float;
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
    int warp_id = tid / WARP_SIZE;

    int tid_k = tid % ThreadPerN;
    int tid_n = tid / ThreadPerN;

    load_atype vreg_a[NUM_X * NUM_UNROLL * BlockM];
    load_btype vreg_b[NUM_UNROLL * NPerThread];
    acc_type accum[NUM_UNROLL * BlockM * NPerThread];

    int num_pid_m = args.num_tokens;
    int num_pid_n = cdiv(args.N, NPerBlock);
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

template <typename src_type, typename dst_type,
          uint32_t SHAPE_N, uint32_t SHAPE_K, int32_t kNumGroups,
          int ThreadPerN, int NPerThread, int NUM_UNROLL,
          int SWZL_SIZE_M>
class Gemvt {
    using load_atype = int4;
    using load_btype = int4;

public:
    Gemvt() = default;

    static void run(dst_type* gmem_d, int* grouped_layout,
                    uint32_t shape_m, src_type* gmem_a, src_type* gmem_b,
                    cudaStream_t stream) {

        constexpr int BlockSize = 256;
        size_t grid_x = shape_m;
        constexpr uint32_t NPerBlock = NPerThread * BlockSize / ThreadPerN;
        size_t grid_y = ceil_div(SHAPE_N, NPerBlock);

        GemvtArgs args;
        args.N = SHAPE_N;
        args.K = SHAPE_K;
        args.a_ptr = (void *)gmem_a;
        args.b_ptr = (void *)gmem_b;
        args.c_ptr = (void *)gmem_d;
        
        // args.topk_weights_ptr = (void *)weight_scales;
        // args.alphaCol = prob_info->alphaCol;
        // args.alphaRow = prob_info->alphaRow;
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
        args.total_blocks = grid_x * grid_y;


        // check GEMM_K alignment
        //   if(args.K % (NUM_UNROLL * ThreadPerN * sizeof(load_atype) / sizeof(src_type)) != 0) {
        //     printf("K alignment mismatch, K = %d, NUM_UNROLL = %d, ThreadPerN = %d, sizeof(load_atype) = %d, sizeof(src_type) = %d",
        //       args.K, NUM_UNROLL, ThreadPerN, sizeof(load_atype), sizeof(src_type));
        //     return;
        //   }

        auto device_func = batched_gemvt_kernel<src_type, dst_type, load_atype, load_btype,
                            BlockSize, ThreadPerN, NPerThread, NUM_UNROLL, SWZL_SIZE_M>;
        dim3 grid = grid_x * grid_y;

        char *pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, device_func);

            printf("[GemV-BF16:]\n");
            printf("group:%d, problem:[%d, %d, %d]\n",
                kNumGroups, shape_m, SHAPE_N, SHAPE_K);
            printf("BlockSize:%d, NPerThread:%d, ThreadPerN:%d, NPerBlock:%d, NUM_UNROLL:%d, SWZL_SIZE_M:%d\n",
                BlockSize, NPerThread, ThreadPerN, NPerBlock, NUM_UNROLL, SWZL_SIZE_M);
            
            printf("threadblock_count:%d, verg:%d, stack:%d\n", args.total_blocks, int(attr.numRegs), int(attr.localSizeBytes));
            
        }

        device_func<<<grid, BlockSize, 0, stream>>>(args);
    }
};

}  // namespace deepgemm
