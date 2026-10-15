#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <megamoe/comm_barrier.cuh>
#include <megamoe/common/fp4.cuh>
#include <megamoe/common/math.cuh>
#include <megamoe/layout/sym_buffer.cuh>
#include <megamoe/layout/workspace.cuh>
#include <megamoe/ptx/atomic.cuh>
#include <megamoe/ptx/cp_async.cuh>
#include <megamoe/ptx/ppu890_copy.cuh>
#include <megamoe/ptx/ppu890_cvt.cuh>
#include <megamoe/ptx/ppu890_mma.cuh>
#include <megamoe/ptx/warp.cuh>
#include <megamoe/scheduler.cuh>
#include <cstdint>
#include <type_traits>

#ifndef DG_DYNAMIC_POOL
#define DG_DYNAMIC_POOL 0
#endif

#ifndef DG_PREFILL
#define DG_PREFILL 0
#endif
#ifndef DG_TILED_GEMM
#define DG_TILED_GEMM 0
#endif
#ifndef DG_TOPK_WEIGHT_SITE
#define DG_TOPK_WEIGHT_SITE 1
#endif

namespace megamoe {

#if DG_PREFILL || DG_TILED_GEMM
struct PrefillFp4Mma {
    __device__ __forceinline__ static void mma(float& d0,
                                               float& d1,
                                               float& d2,
                                               float& d3,
                                               float& d4,
                                               float& d5,
                                               float& d6,
                                               float& d7,
                                               uint32_t a0,
                                               uint32_t a1,
                                               uint32_t a2,
                                               uint32_t a3,
                                               uint32_t b0,
                                               uint32_t b1,
                                               uint32_t b2,
                                               uint32_t b3,
                                               float c0,
                                               float c1,
                                               float c2,
                                               float c3,
                                               float c4,
                                               float c5,
                                               float c6,
                                               float c7,
                                               uint32_t s0,
                                               uint32_t s1,
                                               uint32_t s2,
                                               uint32_t s3) {
        asm("ppu.mma.mx.sync.aligned.m16n16k64.row.col.f32.f4.f4.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, "
            "{%8, %9, %10, %11}, "
            "{%12, %13, %14, %15}, "
            "{%16, %17, %18, %19, %20, %21, %22, %23}, "
            "{%24, %25}, %26, %27, 0;\n"
            : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3), "=r"(d4), "=r"(d5), "=r"(d6), "=r"(d7)
            : "r"(a0),
              "r"(a1),
              "r"(a2),
              "r"(a3),
              "r"(b0),
              "r"(b1),
              "r"(b2),
              "r"(b3),
              "r"(c0),
              "r"(c1),
              "r"(c2),
              "r"(c3),
              "r"(c4),
              "r"(c5),
              "r"(c6),
              "r"(c7),
              "r"(s0),
              "r"(s1),
              "r"(s2),
              "r"(s3));
    }
};

template <bool kSwap = false>
__device__ __forceinline__ void load_prefill_matrix_fragment(
    uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3, const void* shared_base, int row, int column) {
    const uint32_t base_units = static_cast<uint32_t>(__cvta_generic_to_shared(shared_base)) / 16;
    const uint32_t address_units = base_units + row * 8 + column / 8;
    asm volatile("ppu.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4], %5, %6, %7;\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "l"(address_units), "r"(kSwap ? 64 : 1), "r"(kSwap ? 1 : 64), "r"(0));
}

template <bool kContiguous = false>
__device__ __forceinline__ uint32_t load_prefill_scale_fragment(const uint16_t* scales, int warp_coordinate, int k_step) {
    const int lane_idx = threadIdx.x & 31;
    const int scale_row = kContiguous ? warp_coordinate * 64 + (lane_idx / 2) * 16 + (lane_idx & 1) * 8
                                      : warp_coordinate * 16 + (lane_idx / 2) * 64 + (lane_idx & 1) * 8;
    const uint32_t base_address = static_cast<uint32_t>(__cvta_generic_to_shared(scales));
    const uint32_t shared_address = base_address + 2 * (k_step * 320 + (lane_idx < 8 ? (scale_row / 64) * 80 + scale_row % 64 : 0));
    uint32_t value;
    asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];" : "=r"(value) : "r"(shared_address));
    return value;
}

template <bool kContiguous>
__device__ __forceinline__ void load_prefill_operand_fragments(const uint8_t* activation,
                                                               const uint8_t* weight,
                                                               const uint16_t* activation_scale,
                                                               const uint16_t* weight_scale,
                                                               int warp_idx,
                                                               int k_step,
                                                               uint32_t (&activation_fragment)[4][4],
                                                               uint32_t (&weight_fragment)[4][4],
                                                               uint32_t& activation_scale_fragment,
                                                               uint32_t& weight_scale_fragment) {
    const int warp_m_idx = warp_idx & 3;
    const int warp_n_idx = warp_idx / 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        load_prefill_matrix_fragment<>(activation_fragment[i][0],
                                       activation_fragment[i][1],
                                       activation_fragment[i][2],
                                       activation_fragment[i][3],
                                       activation,
                                       warp_m_idx * 16 + i * 64,
                                       k_step * 16);
    }
    activation_scale_fragment = load_prefill_scale_fragment(activation_scale, warp_m_idx, k_step);
    weight_scale_fragment = load_prefill_scale_fragment<kContiguous>(weight_scale, warp_n_idx, k_step);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        load_prefill_matrix_fragment<true>(weight_fragment[i][0],
                                           weight_fragment[i][1],
                                           weight_fragment[i][2],
                                           weight_fragment[i][3],
                                           weight,
                                           (kContiguous ? warp_n_idx * 64 + i * 16 : warp_n_idx * 16 + i * 64),
                                           k_step * 16);
    }
}

template <int kOutputColumns, int kReductionElements, bool kL1, bool kHasActivationClamp, int kPoolRows>
__device__ __forceinline__ void compute_prefill_gemm_tile(const uint8_t* activation,
                                                          const uint16_t* activation_scale,
                                                          const uint8_t* weight,
                                                          const uint16_t* weight_scale,
                                                          uint8_t* quantized_output,
                                                          uint16_t* quantized_scale,
                                                          const int64_t* row_addresses,
                                                          const float* topk_weights,
                                                          const int4& task,
                                                          float activation_clamp,
                                                          uint8_t* shared_buffer) {
    constexpr int kPackedReductionBytes = kReductionElements / 2;
    constexpr int kReductionTiles = kReductionElements / 256;
    constexpr int kStages = 3;
    constexpr int kOperandBytes = 256 * 128;
    constexpr int kScaleElements = 4 * 320;
    const int thread_idx = threadIdx.x;
    const int lane_idx = thread_idx & 31;
    const int warp_idx = __shfl_sync(0xffffffff, thread_idx / 32, 0);
    const int warp_m_idx = warp_idx & 3;
    const int warp_n_idx = warp_idx / 4;

    const int expert_idx = task.x;
    const int row_start = task.y;
    const int task_valid_rows = task.z;
    const int column_start = task.w * 256;
    auto compute_task = [&](auto row_tag) {
        constexpr int valid_rows = decltype(row_tag)::value;
        const auto expert_weight = weight + static_cast<size_t>(expert_idx) * kOutputColumns * kPackedReductionBytes;
        const auto expert_scale = weight_scale + static_cast<size_t>(expert_idx) * kOutputColumns * (kReductionElements / 64);
        auto load_stage = [&](int reduction_tile, int stage_idx) {
            auto stage_activation = shared_buffer + stage_idx * kOperandBytes;
            auto stage_weight = shared_buffer + (kStages + stage_idx) * kOperandBytes;
            auto stage_activation_scale =
                reinterpret_cast<uint16_t*>(shared_buffer + 2 * kStages * kOperandBytes) + stage_idx * kScaleElements;
            auto stage_weight_scale = stage_activation_scale + kStages * kScaleElements;
            if (warp_idx == 0) {
                ptx::ppu890::aiu_load_2d_swzl_b8(
                    stage_activation, activation, kPackedReductionBytes, kPoolRows, 128, valid_rows, reduction_tile * 128, row_start);
                ptx::ppu890::aiu_load_2d_swzl_b8(
                    stage_weight, expert_weight, kPackedReductionBytes, kOutputColumns, 128, 256, reduction_tile * 128, column_start);
            }
            if (warp_idx >= 12 && warp_idx < 16) {
                const int scale_k_idx = warp_idx - 12;
                const int scale_row = (thread_idx & 31) * 8;
                if (scale_row < valid_rows) {
                    ptx::cp_async_16(
                        stage_activation_scale + scale_k_idx * 320 + (scale_row / 64) * 80 + scale_row % 64,
                        activation_scale + static_cast<size_t>(reduction_tile * 4 + scale_k_idx) * kPoolRows + row_start + scale_row);
                } else {
                    *reinterpret_cast<uint4*>(stage_activation_scale + scale_k_idx * 320 + (scale_row / 64) * 80 + scale_row % 64) =
                        make_uint4(0, 0, 0, 0);
                }
                ptx::cp_async_16(
                    stage_weight_scale + scale_k_idx * 320 + (scale_row / 64) * 80 + scale_row % 64,
                    expert_scale + static_cast<size_t>(reduction_tile * 4 + scale_k_idx) * kOutputColumns + column_start + scale_row);
            }
            ptx::cp_async_commit();
        };
        #pragma unroll
        for (int i = 0; i < kStages; ++i)
            load_stage(i, i);
        ptx::cp_async_wait<kStages - 1>();
        __syncthreads();
        uint32_t activation_fragment[2][4][4];
        uint32_t weight_fragment[2][4][4];
        uint32_t activation_scale_fragment[2];
        uint32_t weight_scale_fragment[2];
        float accumulator[4][4][8] = {};
        load_prefill_operand_fragments<true>(
            shared_buffer,
            shared_buffer + kStages * kOperandBytes,
            reinterpret_cast<uint16_t*>(shared_buffer + 2 * kStages * kOperandBytes),
            reinterpret_cast<uint16_t*>(shared_buffer + 2 * kStages * kOperandBytes) + kStages * kScaleElements,
            warp_idx,
            0,
            activation_fragment[0],
            weight_fragment[0],
            activation_scale_fragment[0],
            weight_scale_fragment[0]);
        int stage_idx = 0;
        {
            constexpr int k = 0;
            #pragma unroll
            for (int ki = 0; ki < 4; ++ki) {
                const int next_step = (ki + 1) % 4;
                int next_stage_idx = stage_idx;
                if (ki == 3) {
                    if (k + kStages < kReductionTiles) {
                        __syncthreads();
                        load_stage(k + kStages, stage_idx);
                    } else {
                        ptx::cp_async_commit();
                    }
                    ptx::cp_async_wait<kStages - 1>();
                    __syncthreads();
                    next_stage_idx = stage_idx == kStages - 1 ? 0 : stage_idx + 1;
                }
                const auto next_activation = shared_buffer + next_stage_idx * kOperandBytes;
                const auto next_weight = shared_buffer + (kStages + next_stage_idx) * kOperandBytes;
                const auto next_activation_scale =
                    reinterpret_cast<uint16_t*>(shared_buffer + 2 * kStages * kOperandBytes) + next_stage_idx * kScaleElements;
                load_prefill_operand_fragments<true>(next_activation,
                                                     next_weight,
                                                     next_activation_scale,
                                                     next_activation_scale + kStages * kScaleElements,
                                                     warp_idx,
                                                     next_step,
                                                     activation_fragment[(ki + 1) & 1],
                                                     weight_fragment[(ki + 1) & 1],
                                                     activation_scale_fragment[(ki + 1) & 1],
                                                     weight_scale_fragment[(ki + 1) & 1]);
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        const int m_tile = (n & 1) ? 3 - m : m;
                        auto& values = accumulator[m_tile][n];
                        const auto& activation_values = activation_fragment[ki & 1][m_tile];
                        const auto& weight_values = weight_fragment[ki & 1][n];
                        PrefillFp4Mma::mma(values[0],
                                           values[1],
                                           values[2],
                                           values[3],
                                           values[4],
                                           values[5],
                                           values[6],
                                           values[7],
                                           activation_values[0],
                                           activation_values[1],
                                           activation_values[2],
                                           activation_values[3],
                                           weight_values[0],
                                           weight_values[1],
                                           weight_values[2],
                                           weight_values[3],
                                           values[0],
                                           values[1],
                                           values[2],
                                           values[3],
                                           values[4],
                                           values[5],
                                           values[6],
                                           values[7],
                                           activation_scale_fragment[ki & 1],
                                           weight_scale_fragment[ki & 1],
                                           m_tile,
                                           n);
                    }
                }
            }
            stage_idx = stage_idx == kStages - 1 ? 0 : stage_idx + 1;
        }
        #pragma unroll 1
        for (int k = 1; k < kReductionTiles; ++k) {
            #pragma unroll
            for (int ki = 0; ki < 4; ++ki) {
                const int next_step = (ki + 1) % 4;
                int next_stage_idx = stage_idx;
                if (ki == 3) {
                    if (k + kStages < kReductionTiles) {
                        __syncthreads();
                        load_stage(k + kStages, stage_idx);
                    } else {
                        ptx::cp_async_commit();
                    }
                    ptx::cp_async_wait<kStages - 1>();
                    __syncthreads();
                    next_stage_idx = stage_idx == kStages - 1 ? 0 : stage_idx + 1;
                }
                const auto next_activation = shared_buffer + next_stage_idx * kOperandBytes;
                const auto next_weight = shared_buffer + (kStages + next_stage_idx) * kOperandBytes;
                const auto next_activation_scale =
                    reinterpret_cast<uint16_t*>(shared_buffer + 2 * kStages * kOperandBytes) + next_stage_idx * kScaleElements;
                load_prefill_operand_fragments<true>(next_activation,
                                                     next_weight,
                                                     next_activation_scale,
                                                     next_activation_scale + kStages * kScaleElements,
                                                     warp_idx,
                                                     next_step,
                                                     activation_fragment[(ki + 1) & 1],
                                                     weight_fragment[(ki + 1) & 1],
                                                     activation_scale_fragment[(ki + 1) & 1],
                                                     weight_scale_fragment[(ki + 1) & 1]);
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        const int m_tile = (n & 1) ? 3 - m : m;
                        auto& values = accumulator[m_tile][n];
                        const auto& activation_values = activation_fragment[ki & 1][m_tile];
                        const auto& weight_values = weight_fragment[ki & 1][n];
                        PrefillFp4Mma::mma(values[0],
                                           values[1],
                                           values[2],
                                           values[3],
                                           values[4],
                                           values[5],
                                           values[6],
                                           values[7],
                                           activation_values[0],
                                           activation_values[1],
                                           activation_values[2],
                                           activation_values[3],
                                           weight_values[0],
                                           weight_values[1],
                                           weight_values[2],
                                           weight_values[3],
                                           values[0],
                                           values[1],
                                           values[2],
                                           values[3],
                                           values[4],
                                           values[5],
                                           values[6],
                                           values[7],
                                           activation_scale_fragment[ki & 1],
                                           weight_scale_fragment[ki & 1],
                                           m_tile,
                                           n);
                    }
                }
            }
            stage_idx = stage_idx == kStages - 1 ? 0 : stage_idx + 1;
        }
        if constexpr (kL1) {
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int row = warp_m_idx * 16 + m * 64 + lane_idx / 4 + i * 8;
                    float values[4][2];
                    float maximum = 0.0f;
                    #pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        #pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            float gate = accumulator[m][n][i * 2 + j];
                            float up = accumulator[m][n][i * 2 + j + 4];
                            if constexpr (kHasActivationClamp) {
                                gate = fminf(gate, activation_clamp);
                                up = fminf(fmaxf(up, -activation_clamp), activation_clamp);
                            }
                            values[n][j] = (gate * __ppu_sgmdf(gate)) * up;
                            maximum = fmaxf(maximum, fabsf(values[n][j]));
                        }
                    }
                    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1));
                    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 2));
                    float2 scales, inverse_scales;
                    fp4::get_e2m1_sf_and_sf_inv(make_float2(maximum, maximum), scales, inverse_scales);
                    const int output_column = column_start / 2 + warp_n_idx * 32;
                    if (row < task_valid_rows) {
                        #pragma unroll
                        for (int n = 0; n < 4; ++n) {
                            const uint8_t packed =
                                ptx::ppu890::cvt_f4x2_f32x2(values[n][1] * inverse_scales.x, values[n][0] * inverse_scales.x);
                            quantized_output[static_cast<size_t>(row_start + row) * (kOutputColumns / 4) + output_column / 2 + n * 4 +
                                             (lane_idx & 3)] = packed;
                        }
                        if ((lane_idx & 3) == 0) {
                            const int scale_group = output_column / 32;
                            reinterpret_cast<uint8_t*>(
                                quantized_scale)[(static_cast<size_t>(scale_group / 2) * kPoolRows + row_start + row) * 2 +
                                                 (scale_group & 1)] = static_cast<uint8_t>(fp4::sf_to_ue8m0(scales.x));
                        }
                    }
                }
            }
        } else {
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int row = warp_m_idx * 16 + m * 64 + lane_idx / 4 + i * 8;
                    if (row < task_valid_rows) {
                        auto* const destination = reinterpret_cast<__nv_bfloat16*>(row_addresses[row_start + row]);
                        const float routed_weight = DG_TOPK_WEIGHT_SITE == 1 ? topk_weights[row_start + row] : 1.0f;
                        {
                            #pragma unroll
                            for (int half = 0; half < 2; ++half) {
                                union {
                                    __nv_bfloat162 pairs[4];
                                    uint4 vector;
                                } packed;
                                #pragma unroll
                                for (int w = 0; w < 4; ++w) {
                                    const int n = half * 2 + w / 2;
                                    const int component = i * 2 + (w & 1) * 4;
                                    packed.pairs[w] = __floats2bfloat162_rn(accumulator[m][n][component] * routed_weight,
                                                                            accumulator[m][n][component + 1] * routed_weight);
                                }
                                const int column = column_start + warp_n_idx * 64 + (lane_idx & 3) * 16 + half * 8;
                                *reinterpret_cast<uint4*>(destination + column) = packed.vector;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    };
    if (task_valid_rows > 128)
        compute_task(std::integral_constant<int, 256>{});
    else
        compute_task(std::integral_constant<int, 128>{});
}

#endif

template <bool kHasActivationClamp>
__device__ __forceinline__ float clamp_gate(float x, float activation_clamp) {
    if constexpr (kHasActivationClamp)
        x = megamoe::min(x, activation_clamp);
    return x;
}

template <bool kHasActivationClamp>
__device__ __forceinline__ float clamp_up(float x, float activation_clamp) {
    if constexpr (kHasActivationClamp)
        x = megamoe::min(megamoe::max(x, -activation_clamp), activation_clamp);
    return x;
}

__device__ __forceinline__ float ppu890_silu(float x) {
    return x * __ppu_sgmdf(x);
}

template <bool kHasActivationClamp>
__device__ __forceinline__ float ppu890_swiglu(float gate, float up, float activation_clamp) {
    gate = clamp_gate<kHasActivationClamp>(gate, activation_clamp);
    up = clamp_up<kHasActivationClamp>(up, activation_clamp);
    return ppu890_silu(gate) * up;
}

template <uint32_t kAlignedMaxTokensPerRank,
          uint32_t kDeclaredMaxTokensPerRank,
          uint32_t kHiddenDim,
          uint32_t kIntermediateDim,
          uint32_t kExpertCount,
          uint32_t kTopkCount,
          uint32_t kExpertsPerGroup,
          uint32_t BLOCK_M,
          uint32_t BLOCK_N,
          uint32_t BLOCK_K,
          uint32_t kPoolRowCapacity,
          uint32_t kSfPoolRowCapacity,
          uint32_t kThreadsPerBlock,
          uint32_t kSmCount,
          uint32_t kRankCount,
          bool kHasActivationClamp,
          uint32_t kMaxTokensPerSourceRank,
          uint32_t kReservedMBlocksPerExpert,
          uint32_t kDynamicSmemBytes = 0,
          uint32_t L1_SHAPE_N = kIntermediateDim * 2,
          uint32_t L1_SHAPE_K = kHiddenDim,
          uint32_t L2_SHAPE_N = kHiddenDim,
          uint32_t L2_SHAPE_K = kIntermediateDim,
          uint32_t kWarpsPerBlock = kThreadsPerBlock / 32,
          uint32_t kExpertsPerRank = kExpertCount / kRankCount>
DG_GLOBAL __launch_bounds__(kThreadsPerBlock,
                                 1) void ppu890_fp4_fp4_megamoe(void* y,
                                                                int* cumulative_local_expert_recv_counts,
                                                                const uint32_t num_tokens,
                                                                const float activation_clamp,
                                                                const __grid_constant__ layout::SymBuffer<kRankCount> sym_buffer,
                                                                const uint8_t* __restrict__ l1_weights,
                                                                const uint8_t* __restrict__ l1_weights_sf,
                                                                const uint8_t* __restrict__ l2_weights,
                                                                const uint8_t* __restrict__ l2_weights_sf,
                                                                const __nv_bfloat16* __restrict__ x_bf16,
                                                                const uint32_t physical_push_sm_mask_lo,
                                                                const uint32_t physical_push_sm_mask_hi,
                                                                const uint32_t physical_quant_sm_mask_lo,
                                                                const uint32_t physical_quant_sm_mask_hi) {
#if defined(__CUDA_ARCH__) && (!defined(__HGGC_ARCH__) || ACOMPUTE_VERSION < 10500)
#error "PPU890 megamoe kernel requires the PPU SDK (__HGGC_ARCH__ && ACOMPUTE_VERSION >= 10500) for AIU bulk-DMA + swzl ldmatrix."
#endif
    DG_STATIC_ASSERT(BLOCK_N == 128 || BLOCK_N == 256 || BLOCK_N == 512, "PPU890 MMA path: BLOCK_N in {128, 256, 512}");
    DG_STATIC_ASSERT(BLOCK_K == 128 || BLOCK_K == 256, "PPU890 MMA path: BLOCK_K in {128, 256}");
    DG_STATIC_ASSERT(BLOCK_M % 16 == 0 && BLOCK_M >= 16 && BLOCK_M <= 256,
                          "PPU890 MMA path: BLOCK_M must be multiple of 16 in [16, 256]");
    DG_STATIC_ASSERT(kExpertCount % kRankCount == 0, "kExpertCount must be divisible by kRankCount (else kExpertsPerRank truncates)");
    DG_STATIC_ASSERT(kReservedMBlocksPerExpert > 0, "the pool always reserves per-expert blocks");
#ifdef DG_M_TILES_PER_WARP
    constexpr uint32_t kMmaMTilesPerWarp = DG_M_TILES_PER_WARP;
#else
    constexpr uint32_t kMmaMTilesPerWarp = 1;
#endif

#ifndef DG_GEMM_K_LOOP_UNROLL
#define DG_GEMM_K_LOOP_UNROLL _Pragma("unroll 1")
#endif
    DG_STATIC_ASSERT(BLOCK_M % (16 * kMmaMTilesPerWarp) == 0, "BLOCK_M must be a multiple of 16*kMmaMTilesPerWarp");
    DG_STATIC_ASSERT(kWarpsPerBlock % (BLOCK_M / (16 * kMmaMTilesPerWarp)) == 0,
                          "kWarpsPerBlock must be a multiple of BLOCK_M/(16*kMmaMTilesPerWarp)");
    DG_STATIC_ASSERT((kWarpsPerBlock / (BLOCK_M / (16 * kMmaMTilesPerWarp))) <= 16, "WARP_N (N-split) supported up to 16");
    DG_STATIC_ASSERT(kTopkCount <= kExpertsPerRank, "Invalid topk");

    constexpr uint32_t kL1KTiles = L1_SHAPE_K / BLOCK_K;
    constexpr uint32_t kL2KTiles = L2_SHAPE_K / BLOCK_K;
    constexpr uint32_t kSwigluColsPerBlock = BLOCK_N / 2;
    constexpr uint32_t kWarpGridM = BLOCK_M / (16 * kMmaMTilesPerWarp);

    constexpr int kMmaMTileRowStride = (kMmaMTilesPerWarp == 2) ? static_cast<int>(kWarpGridM) * 16 : 16;
    constexpr uint32_t kWarpGridN = kWarpsPerBlock / kWarpGridM;
    constexpr uint32_t kGemmOutputColsPerWarp = BLOCK_N / kWarpGridN;
    constexpr uint32_t kMmaNTilesPerWarp = kGemmOutputColsPerWarp / 8;
    constexpr uint32_t kSwigluColsPerWarp = kSwigluColsPerBlock / kWarpGridN;
    constexpr uint32_t kStageReadyBarrierId = 8;
    constexpr uint32_t kStageReuseBarrierId = 9;
    constexpr uint32_t kCombineBlockBarrierId = 7;
    constexpr uint32_t kL1ReadyBarrierId = 11;
#ifndef DG_TOPK_WEIGHT_SITE
#define DG_TOPK_WEIGHT_SITE 1
#endif
    DG_STATIC_ASSERT(DG_TOPK_WEIGHT_SITE >= 1 && DG_TOPK_WEIGHT_SITE <= 2,
                          "DG_TOPK_WEIGHT_SITE must be 1 (L2 epilogue) or 2 (combine)");
    constexpr uint32_t kCombineGridBarrierSlot = 1;
    constexpr uint32_t kCombineIcnBarrierTag = 2;

    const uint32_t logical_sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;

    uint32_t physical_sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(physical_sm_id));
    const uint64_t physical_push_sm_mask =
        static_cast<uint64_t>(physical_push_sm_mask_lo) | (static_cast<uint64_t>(physical_push_sm_mask_hi) << 32);
    const uint64_t physical_quant_sm_mask =
        static_cast<uint64_t>(physical_quant_sm_mask_lo) | (static_cast<uint64_t>(physical_quant_sm_mask_hi) << 32);
    const uint64_t physical_sm_bit = 1ULL << physical_sm_id;
    const bool is_push_sm = (physical_push_sm_mask & physical_sm_bit) != 0;
    const bool is_quant_sm = (physical_quant_sm_mask & physical_sm_bit) != 0;
    const uint32_t push_sm_count = static_cast<uint32_t>(__popcll(physical_push_sm_mask));
    const uint32_t quant_sm_count = static_cast<uint32_t>(__popcll(physical_quant_sm_mask));
    const uint32_t push_sm_set_idx = static_cast<uint32_t>(__popcll(physical_push_sm_mask & (physical_sm_bit - 1ULL)));
    const uint32_t quant_sm_set_idx = static_cast<uint32_t>(__popcll(physical_quant_sm_mask & (physical_sm_bit - 1ULL)));
    const uint32_t warp_idx = thread_idx / 32;
    const uint32_t lane_idx = ptx::get_lane_idx();

    const auto workspace = layout::Workspace(sym_buffer.get_base_ptr(), kRankCount, kExpertCount);

    constexpr uint64_t kWorkspacePrefixBytes = math::constexpr_align_up<uint64_t>(
        32 + static_cast<uint64_t>(kExpertCount) * sizeof(uint64_t) * 2 + static_cast<uint64_t>(kExpertsPerRank) * sizeof(uint64_t), 128u);

    constexpr uint32_t kInputSfPayloadBytesPerToken = kHiddenDim / 32;
    constexpr uint32_t kInputSfRowStrideBytes = (kInputSfPayloadBytesPerToken + 127) / 128 * 128;
    constexpr uint32_t kInputTopkIdxPayloadBytesPerToken = kTopkCount * sizeof(int64_t);
    constexpr uint32_t kInputTopkIdxRowStrideBytes = (kInputTopkIdxPayloadBytesPerToken + 127) / 128 * 128;
    constexpr uint32_t kInputTopkWeightPayloadBytesPerToken = kTopkCount * sizeof(float);
    constexpr uint32_t kInputTopkWeightRowStrideBytes = (kInputTopkWeightPayloadBytesPerToken + 127) / 128 * 128;

    constexpr uint64_t kDispatchReservedBytesPerSourceRank = (512 + 2048) +
        static_cast<uint64_t>(kMaxTokensPerSourceRank) *
            (kHiddenDim / 2 + kInputSfRowStrideBytes + kInputTopkIdxRowStrideBytes + kInputTopkWeightRowStrideBytes);
    constexpr uint64_t kDispatchReservedBytes = static_cast<uint64_t>(kRankCount) * kDispatchReservedBytesPerSourceRank;

    constexpr uint32_t kSlotHeaderOffsetBytes = 0;
    constexpr uint32_t kSlotXOffsetBytes = 128;
    constexpr uint32_t kSlotSfOffsetBytes = 4224;
    constexpr uint32_t kSlotStrideBytes = 4480;

    constexpr uint32_t kFlagLane = 31u;

    constexpr uint32_t kLineReadyMask = 1u;
    constexpr uint32_t kSlotEmptyMask = 2u;
    constexpr uint32_t kNonemptySlotHeaderReadyWord = kLineReadyMask;
    constexpr uint32_t kEmptySlotHeaderReadyWord = kLineReadyMask | kSlotEmptyMask;
    constexpr uint32_t kTopkHeaderWordCount = kTopkCount * 2;

    constexpr uint32_t kXDisplacedBitsLaneIdx = 0u;
    constexpr uint32_t kSfDisplacedBitsLaneIdx = kTopkHeaderWordCount + 2u;

    DG_STATIC_ASSERT(kSlotStrideBytes % 128u == 0u, "slot stride must be a 128B multiple");
    DG_STATIC_ASSERT(kSlotXOffsetBytes % 128u == 0u && kSlotSfOffsetBytes % 128u == 0u, "every slot line start must be 128B aligned");
    DG_STATIC_ASSERT(kTopkHeaderWordCount + 1u < kSfDisplacedBitsLaneIdx && kSfDisplacedBitsLaneIdx < kFlagLane &&
                              kXDisplacedBitsLaneIdx != kSfDisplacedBitsLaneIdx,
                          "the topk pairs, the token id, the sf displaced lane and the flag lane must not collide");
    constexpr uint64_t kSlotAreaBytes = static_cast<uint64_t>(kRankCount) * kMaxTokensPerSourceRank * kSlotStrideBytes;
    constexpr uint64_t kPoolRowCursorRegionBytes = math::constexpr_align_up<uint32_t>(kExpertsPerRank * sizeof(uint32_t), 128u);
    constexpr uint64_t kDispatchHandoffLineOffsetBytes = kSlotAreaBytes + kPoolRowCursorRegionBytes;

#ifndef DG_PREFILL
#define DG_PREFILL 0
#endif
    constexpr uint64_t kSourceTokenCountsOffsetBytes = kDispatchHandoffLineOffsetBytes + 128;
    constexpr uint64_t kCountLineBytes = 128;
    constexpr uint64_t kSourceTokenCountsRegionBytes = static_cast<uint64_t>(kRankCount) * kCountLineBytes;

    constexpr uint64_t kPoolRowCursorsInitializedEpochOffsetBytes = kDispatchHandoffLineOffsetBytes + 4;

    constexpr uint64_t kStoredLaunchEpochOffsetBytes = kDispatchHandoffLineOffsetBytes + 8;

    constexpr uint64_t kConsumeTaskCursorOffsetBytes = kDispatchHandoffLineOffsetBytes + 12;
#if DG_PREFILL

    constexpr uint32_t kExpertCountWordsPerLine = 32;
    constexpr uint32_t kExpertCountLines = (kExpertsPerRank + kExpertCountWordsPerLine - 1) / kExpertCountWordsPerLine;
    constexpr uint64_t kExpertCountsOffset = kSourceTokenCountsOffsetBytes + kSourceTokenCountsRegionBytes;
    constexpr uint64_t kExpertCountsRegionBytes = static_cast<uint64_t>(kRankCount) * kExpertCountLines * kCountLineBytes;

    constexpr uint64_t kExpertPoolRowBasesOffsetBytes = kExpertCountsOffset + kExpertCountsRegionBytes;
    constexpr uint64_t kExpertPoolRowBasesRegionBytes = math::constexpr_align_up<uint32_t>(kExpertsPerRank * sizeof(uint32_t), 128u);

    constexpr uint64_t kExpertPoolLayoutReadyEpochOffsetBytes = kExpertPoolRowBasesOffsetBytes + kExpertPoolRowBasesRegionBytes;
    constexpr uint64_t kExpertPoolLayoutReadyEpochLineBytes = 128;
    constexpr uint64_t kDispatchRegionBytes = kExpertPoolLayoutReadyEpochOffsetBytes + kExpertPoolLayoutReadyEpochLineBytes;

    constexpr bool kUsesDynamicPool = (DG_DYNAMIC_POOL != 0);
    constexpr uint32_t kPoolRowAlignment = 384;
    {
        DG_STATIC_ASSERT(kExpertCountWordsPerLine == 32, "a 128B line is 32 words wide");
        if constexpr (kUsesDynamicPool) {
            DG_STATIC_ASSERT(kPoolRowAlignment % BLOCK_M == 0,
                                  "the dynamic pool prefix must be a multiple of BLOCK_M (workspace._WORKSPACE_BLOCK_M_CANDIDATES)");
            constexpr uint64_t kDynamicPoolRowCapacityBound =
                (static_cast<uint64_t>(kRankCount) * kMaxTokensPerSourceRank *
                     (kTopkCount < kExpertsPerRank ? kTopkCount : kExpertsPerRank) +
                 static_cast<uint64_t>(kExpertsPerRank) * (kPoolRowAlignment - 1) + kPoolRowAlignment - 1) /
                kPoolRowAlignment * kPoolRowAlignment;
            DG_STATIC_ASSERT(kDynamicPoolRowCapacityBound <= kPoolRowCapacity,
                                  "the packed pool bound exceeds the host-sized pool (workspace._uses_dynamic_pool disagrees)");
        }
    }
#else
    constexpr uint64_t kDispatchRegionBytes = kSourceTokenCountsOffsetBytes + kSourceTokenCountsRegionBytes;
#endif

    DG_STATIC_ASSERT(kDeclaredMaxTokensPerRank <= 0xFFFFu, "the per-source count word packs the count in 16 bits with the epoch");

    constexpr uint32_t kFlagPollSleepNs = 150u;
    constexpr uint64_t kDataRegionBytes = (kDispatchReservedBytes > kDispatchRegionBytes) ? kDispatchReservedBytes : kDispatchRegionBytes;
    constexpr uint64_t kPoolMetadataBytes = static_cast<uint64_t>(kPoolRowCapacity) * 12;
    constexpr uint64_t kL1CompletedBlockCountLineBytes = 128;
    constexpr uint64_t kL1CompletedBlockCountsOffsetBytes =
        ((kWorkspacePrefixBytes + kDataRegionBytes + kPoolMetadataBytes) + kL1CompletedBlockCountLineBytes - 1) /
        kL1CompletedBlockCountLineBytes * kL1CompletedBlockCountLineBytes;
    constexpr uint64_t kL1CompletedBlockCountsRegionBytes = static_cast<uint64_t>(kExpertsPerRank) * kL1CompletedBlockCountLineBytes;
    constexpr uint32_t kL1CompletedBlockCountLineWords = static_cast<uint32_t>(kL1CompletedBlockCountLineBytes / sizeof(uint32_t));
    constexpr uint64_t kWorkspaceCoreBytes = ((kL1CompletedBlockCountsOffsetBytes + kL1CompletedBlockCountsRegionBytes) + 15) / 16 * 16;
    constexpr uint64_t kWorkspaceTotalBytes = kWorkspaceCoreBytes;

    auto* pool_metadata_base =
        reinterpret_cast<layout::TokenSrcMetadata*>(math::advance_ptr<uint8_t>(workspace.base, kWorkspacePrefixBytes + kDataRegionBytes));
    auto* l1_completed_block_counts_base =
        reinterpret_cast<uint32_t*>(math::advance_ptr<uint8_t>(workspace.base, kL1CompletedBlockCountsOffsetBytes));

    constexpr auto hidden_fp4_row_layout = layout::DataLayout(kHiddenDim / 2);
    constexpr auto hidden_bf16_row_layout = layout::DataLayout(kHiddenDim * sizeof(nv_bfloat16));
    constexpr auto intermediate_fp4_row_layout = layout::DataLayout(kIntermediateDim / 2);

    constexpr auto activation_sf_layout = layout::DataLayout(kHiddenDim / 32);
    constexpr auto intermediate_sf_layout = layout::DataLayout(kIntermediateDim / 32);
    constexpr auto l1_topk_weights_layout = layout::DataLayout(sizeof(float), false);

    const auto input_token_buffer =
        layout::Buffer(hidden_fp4_row_layout, 1, kAlignedMaxTokensPerRank, math::advance_ptr(workspace.base, kWorkspaceTotalBytes));
    auto* input_sf_base = reinterpret_cast<uint8_t*>(input_token_buffer.get_end_ptr());
    auto* input_topk_expert_ids_base = input_sf_base + static_cast<uint64_t>(kAlignedMaxTokensPerRank) * kInputSfRowStrideBytes;
    auto* input_topk_weight_base =
        input_topk_expert_ids_base + static_cast<uint64_t>(kAlignedMaxTokensPerRank) * kInputTopkIdxRowStrideBytes;
    const auto l1_token_buffer =
        layout::Buffer(hidden_fp4_row_layout,
                       1,
                       kPoolRowCapacity,
                       input_topk_weight_base + static_cast<uint64_t>(kAlignedMaxTokensPerRank) * kInputTopkWeightRowStrideBytes);
    const auto l1_sf_buffer = layout::Buffer(activation_sf_layout, 1, kSfPoolRowCapacity, l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(l1_topk_weights_layout, 1, kPoolRowCapacity, l1_sf_buffer.get_end_ptr());
    const auto l2_token_buffer = layout::Buffer(intermediate_fp4_row_layout, 1, kPoolRowCapacity, l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer = layout::Buffer(intermediate_sf_layout, 1, kSfPoolRowCapacity, l2_token_buffer.get_end_ptr());
    const auto combine_token_buffer =
        layout::Buffer(hidden_bf16_row_layout, kTopkCount, kAlignedMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

    constexpr uint32_t kSmemAlignmentBytes = 1024;
    extern __shared__ __align__(kSmemAlignmentBytes) uint8_t smem_buffer[];
    constexpr uint32_t kSmemExpertScratchBytes = math::constexpr_align_up<uint32_t>(kExpertCount * sizeof(uint32_t), kSmemAlignmentBytes);
    constexpr uint32_t kSmemStageABytes = BLOCK_M * (BLOCK_K / 2);
    constexpr uint32_t kSmemStageBBytes = BLOCK_N * (BLOCK_K / 2);
    DG_STATIC_ASSERT(kGemmOutputColsPerWarp == 64, "SFB layout assumes 4 n16 tiles / 8 part0 threads per warp");
    constexpr uint32_t kSmemStageSfbBytes = BLOCK_N * (BLOCK_K / 64) * sizeof(uint16_t);

    DG_STATIC_ASSERT(kIntermediateDim <= kHiddenDim, "the activation-scale staging is sized by the hidden dim");

    constexpr uint32_t kActivationSfRows = kHiddenDim / 128;
    constexpr uint32_t kActivationSfTileWords = kActivationSfRows * BLOCK_M;
    constexpr uint32_t kSmemAsfBytes = kActivationSfTileWords * sizeof(uint32_t);
#ifdef DG_GEMM_STAGES
    constexpr uint32_t kGemmBStageCount = DG_GEMM_STAGES;
#else
    constexpr uint32_t kGemmBStageCount = 2;
#endif
    DG_STATIC_ASSERT(kGemmBStageCount >= 2, "GEMM bulk-DMA pipeline needs >= 2 stages");

    constexpr uint32_t kGemmAsfbStageCount = DG_GEMM_ASFB_STAGES;
    DG_STATIC_ASSERT(kGemmAsfbStageCount >= 2 && kGemmAsfbStageCount <= kGemmBStageCount, "A/SFB stages must lie in [2, stages]");
    DG_STATIC_ASSERT(kGemmBStageCount - kGemmAsfbStageCount <= 1, "B leads A/SFB by at most one k-block");

    constexpr int kAllowedPendingCopyGroups = (kGemmAsfbStageCount == kGemmBStageCount) ? static_cast<int>(kGemmBStageCount) - 2 : 1;

    constexpr uint32_t kSmemLayoutBytes = kSmemExpertScratchBytes + kGemmBStageCount * kSmemStageBBytes +
        kGemmAsfbStageCount * (kSmemStageABytes + kSmemStageSfbBytes) + kSmemAsfBytes;
    DG_STATIC_ASSERT(kSmemLayoutBytes <= kDynamicSmemBytes,
                          "GEMM smem layout overflows the host's dynamic allocation (see kDynamicSmemBytes)");
    DG_STATIC_ASSERT(kAlignedMaxTokensPerRank * sizeof(uint32_t) <= kDynamicSmemBytes,
                          "combine's smem_topk_valid_masks mask does not fit the dynamic allocation");

    {
        auto* dispatch_base = reinterpret_cast<uint8_t*>(workspace.get_dispatch_slot_base_ptr());
        const auto* const stored_launch_epoch_ptr = reinterpret_cast<const uint32_t*>(dispatch_base + kStoredLaunchEpochOffsetBytes);
        const uint32_t current_launch_epoch = (ptx::ld_volatile(stored_launch_epoch_ptr) % 0xFFFFu) + 1u;
        constexpr uint32_t kQuantGroupK = 32;
        constexpr uint32_t kQuantGroupsPerToken = kHiddenDim / kQuantGroupK;
        const uint32_t quant_group_count = num_tokens * kQuantGroupsPerToken;

        const uint32_t quant_global_thread_idx = quant_sm_set_idx * kThreadsPerBlock + thread_idx;
        const uint32_t quant_grid_stride = quant_sm_count * kThreadsPerBlock;
        auto* const x_quant_base = input_token_buffer.get_base_ptr<uint8_t>();

        const auto run_quant = [&]() [[clang::noinline]] {
            if (!is_quant_sm)
                return;

            DG_STATIC_ASSERT(kHiddenDim == 32u * kQuantGroupsPerToken, "the hidden row must be a whole number of 32-element groups");
            DG_STATIC_ASSERT(kHiddenDim / 2 == 16u * kQuantGroupsPerToken, "the packed fp4 row must be one group of 16B per group");
            DG_STATIC_ASSERT(kInputSfRowStrideBytes == kQuantGroupsPerToken, "the sf row stride must be one byte per group");
            const auto* const x_bf16_groups = reinterpret_cast<const int4*>(x_bf16);
            auto* const x_quant_groups = reinterpret_cast<int4*>(x_quant_base);

            const auto quantize_group = [&](const int4* const x_bf16_src, const uint32_t global_group_idx) {
                float input_values[32];
                float amax = 0.0f;
                #pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const int4 vec = x_bf16_src[v];
                    const __nv_bfloat162* bf16x2 = reinterpret_cast<const __nv_bfloat162*>(&vec);
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        float lo = __bfloat162float(bf16x2[i].x);
                        float hi = __bfloat162float(bf16x2[i].y);
                        input_values[v * 8 + i * 2] = lo;
                        input_values[v * 8 + i * 2 + 1] = hi;
                        amax = fmaxf(amax, fmaxf(fabsf(lo), fabsf(hi)));
                    }
                }

                uint32_t amax_bits = __float_as_uint(fmaxf(amax, 1e-10f));
                uint32_t unrounded_scale_bits = amax_bits - 0x1400000u;
                uint32_t scale_exponent_bits = (unrounded_scale_bits + 0x007FFFFFu) & 0x7F800000u;
                float sf_inv = __uint_as_float((254u << 23) - scale_exponent_bits);

                uint32_t quantized_words[4];
                #pragma unroll
                for (int w = 0; w < 4; w++) {
                    uint32_t quantized_word = 0;
                    #pragma unroll
                    for (int p = 0; p < 4; p++) {
                        int value_pair_base_idx = w * 8 + p * 2;
                        uint8_t fp4_pair_byte = ptx::ppu890::cvt_f4x2_f32x2(input_values[value_pair_base_idx + 1] * sf_inv,
                                                                            input_values[value_pair_base_idx] * sf_inv);
                        quantized_word |= (static_cast<uint32_t>(fp4_pair_byte) << (p * 8));
                    }
                    quantized_words[w] = quantized_word;
                }

                x_quant_groups[global_group_idx] = *reinterpret_cast<const int4*>(quantized_words);
                input_sf_base[global_group_idx] = static_cast<uint8_t>((scale_exponent_bits >> 23) & 0xFFu);
            };

            const uint32_t quant_round_count = (quant_group_count + quant_grid_stride - 1u) / quant_grid_stride;

            constexpr uint32_t kQuantStageBytes = kThreadsPerBlock * 64u;
            constexpr uint32_t kQuantStageRoom =
                (kDynamicSmemBytes > kSmemExpertScratchBytes) ? (kDynamicSmemBytes - kSmemExpertScratchBytes) : 0u;
            constexpr uint32_t kQuantStageCount =
                (kQuantStageRoom / kQuantStageBytes >= 4u) ? 4u : ((kQuantStageRoom / kQuantStageBytes >= 2u) ? 2u : 0u);
            if constexpr (kQuantStageCount >= 2u) {
                auto* const quant_stage_own = smem_buffer + kSmemExpertScratchBytes + static_cast<uint64_t>(thread_idx) * 64u;
                const auto issue_group_copy = [&](const uint32_t round) {
                    const uint32_t g = quant_global_thread_idx + round * quant_grid_stride;
                    if (g >= quant_group_count)
                        return;
                    const auto* const src = reinterpret_cast<const uint8_t*>(x_bf16_groups + 4u * g);
                    auto* const dst = quant_stage_own + (round % kQuantStageCount) * kQuantStageBytes;
                    #pragma unroll
                    for (uint32_t c = 0; c < 4u; ++c)
                        ptx::cp_async_16(dst + 16u * c, src + 16u * c);
                };
                #pragma unroll 1
                for (uint32_t r = 0; r < kQuantStageCount; ++r) {
                    issue_group_copy(r);
                    ptx::cp_async_commit();
                }
                #pragma unroll 1
                for (uint32_t quant_round_idx = 0; quant_round_idx < quant_round_count; ++quant_round_idx) {
                    ptx::cp_async_wait<kQuantStageCount - 1>();
                    const uint32_t global_group_idx = quant_global_thread_idx + quant_round_idx * quant_grid_stride;
                    if (global_group_idx < quant_group_count)
                        quantize_group(
                            reinterpret_cast<const int4*>(quant_stage_own + (quant_round_idx % kQuantStageCount) * kQuantStageBytes),
                            global_group_idx);
                    issue_group_copy(quant_round_idx + kQuantStageCount);
                    ptx::cp_async_commit();
                }
            } else {
                #pragma unroll 2
                for (uint32_t quant_round_idx = 0; quant_round_idx < quant_round_count; ++quant_round_idx) {
                    const uint32_t global_group_idx = quant_global_thread_idx + quant_round_idx * quant_grid_stride;
                    if (global_group_idx < quant_group_count)
                        quantize_group(x_bf16_groups + 4u * global_group_idx, global_group_idx);
                }
            }
        };
        run_quant();

        DG_STATIC_ASSERT(kSmCount <= 64, "the dispatch handoff counts participants in a 64-bit mask");
        const uint32_t active_quant_sm_count = megamoe::min(quant_sm_count, (quant_group_count + kThreadsPerBlock - 1) / kThreadsPerBlock);
#if DG_PREFILL

        auto* dispatch_handoff_counter_ptr = reinterpret_cast<uint32_t*>(dispatch_base + kDispatchHandoffLineOffsetBytes);
        if (logical_sm_idx == 0 && thread_idx == 0) {
            if constexpr (!kUsesDynamicPool) {
                auto* pool_row_cursor = reinterpret_cast<uint32_t*>(dispatch_base + kSlotAreaBytes);

                for (uint32_t e = 0; e < kExpertsPerRank; ++e)
                    ptx::st_volatile(pool_row_cursor + e, e * kReservedMBlocksPerExpert * BLOCK_M);

                ptx::st_volatile(reinterpret_cast<const uint32_t*>(dispatch_base + kConsumeTaskCursorOffsetBytes), 0u);
                ptx::atomic_exch_rel(reinterpret_cast<const uint32_t*>(dispatch_base + kPoolRowCursorsInitializedEpochOffsetBytes),
                                     current_launch_epoch & 0xFFFFu);
            }
        }

        if constexpr (!kUsesDynamicPool) {
            if (thread_idx == 0u) {
                const uint32_t* const pool_row_cursors_initialized_epoch_ptr =
                    reinterpret_cast<const uint32_t*>(dispatch_base + kPoolRowCursorsInitializedEpochOffsetBytes);
                while (ptx::ld_acq_gpu(pool_row_cursors_initialized_epoch_ptr) != (current_launch_epoch & 0xFFFFu))
                    __nanosleep(kFlagPollSleepNs);
            }
            __syncthreads();
        }
#else
        auto* dispatch_handoff_counter_ptr = reinterpret_cast<uint32_t*>(dispatch_base + kDispatchHandoffLineOffsetBytes);

        if (logical_sm_idx == 0 && thread_idx == 0) {
            auto* pool_row_cursor = reinterpret_cast<uint32_t*>(dispatch_base + kSlotAreaBytes);
            for (uint32_t e = 0; e < kExpertsPerRank; ++e)
                ptx::st_volatile(pool_row_cursor + e, e * kReservedMBlocksPerExpert * BLOCK_M);

            ptx::st_volatile(reinterpret_cast<const uint32_t*>(dispatch_base + kConsumeTaskCursorOffsetBytes), 0u);
            ptx::atomic_exch_rel(reinterpret_cast<const uint32_t*>(dispatch_base + kPoolRowCursorsInitializedEpochOffsetBytes),
                                 current_launch_epoch & 0xFFFFu);
        }

        if (thread_idx == 0u) {
            const uint32_t* const pool_row_cursors_initialized_epoch_ptr =
                reinterpret_cast<const uint32_t*>(dispatch_base + kPoolRowCursorsInitializedEpochOffsetBytes);
            while (ptx::ld_acq_gpu(pool_row_cursors_initialized_epoch_ptr) != (current_launch_epoch & 0xFFFFu))
                __nanosleep(kFlagPollSleepNs);
        }
        __syncthreads();
#endif

        if (!is_quant_sm && !is_push_sm) {
            const uint32_t count_word = ((current_launch_epoch & 0xFFFFu) << 16) | (num_tokens & 0xFFFFu);
            auto* const count_src = reinterpret_cast<uint8_t*>(workspace.get_dispatch_slot_base_ptr()) + kSourceTokenCountsOffsetBytes +
                static_cast<uint64_t>(sym_buffer.rank_idx) * kCountLineBytes;
            for (uint32_t d = warp_idx; d < kRankCount; d += kWarpsPerBlock) {
                uint8_t* const dst_line = sym_buffer.map_pointer_to_rank(count_src, d);
                if (d != sym_buffer.rank_idx) {
                    const uint32_t lane_word = (lane_idx == 0u) ? count_word : 0u;
                    __ppu_remote_store_bulk_volatile_b32(reinterpret_cast<volatile void*>(dst_line), lane_word);
                } else if (lane_idx == 0u) {
                    ptx::st_volatile(reinterpret_cast<const uint32_t*>(dst_line), count_word);
                }
            }
#if DG_PREFILL
            if constexpr (kUsesDynamicPool) {
                constexpr uint32_t kPoolRowHistogramWords = kRankCount * kExpertsPerRank;
                DG_STATIC_ASSERT(kPoolRowHistogramWords * sizeof(uint32_t) <= kSmemExpertScratchBytes,
                                      "the pool histogram must fit the expert-count carve");
                auto* const smem_histogram = reinterpret_cast<uint32_t*>(smem_buffer);
                for (uint32_t i = thread_idx; i < kPoolRowHistogramWords; i += kThreadsPerBlock)
                    smem_histogram[i] = 0u;
                __syncthreads();

                for (uint32_t token_idx = thread_idx; token_idx < num_tokens; token_idx += kThreadsPerBlock) {
                    const auto* const topk_row = reinterpret_cast<const int64_t*>(
                        input_topk_expert_ids_base + static_cast<uint64_t>(token_idx) * kInputTopkIdxRowStrideBytes);
                    #pragma unroll
                    for (uint32_t slot = 0; slot < kTopkCount; ++slot) {
                        const int32_t expert_idx = static_cast<int32_t>(topk_row[slot]);
                        if (expert_idx >= 0)
                            atomicAdd_block(smem_histogram + (static_cast<uint32_t>(expert_idx) / kExpertsPerRank) * kExpertsPerRank +
                                                (static_cast<uint32_t>(expert_idx) % kExpertsPerRank),
                                            1u);
                    }
                }
                __syncthreads();
                auto* const expert_count_src =
                    dispatch_base + kExpertCountsOffset + static_cast<uint64_t>(sym_buffer.rank_idx) * kExpertCountLines * kCountLineBytes;
                for (uint32_t item = warp_idx; item < kRankCount * kExpertCountLines; item += kWarpsPerBlock) {
                    const uint32_t dst_rank = item / kExpertCountLines;
                    const uint32_t line = item % kExpertCountLines;
                    const uint32_t expert = line * kExpertCountWordsPerLine + lane_idx;
                    const uint32_t count = (expert < kExpertsPerRank) ? smem_histogram[dst_rank * kExpertsPerRank + expert] : 0u;

                    const uint32_t expert_count_word = ((current_launch_epoch & 0xFFFFu) << 16) | (count > 0xFFFFu ? 0xFFFFu : count);
                    uint8_t* const dst_line =
                        sym_buffer.map_pointer_to_rank(expert_count_src + static_cast<uint64_t>(line) * kCountLineBytes, dst_rank);
                    if (dst_rank != sym_buffer.rank_idx)
                        __ppu_remote_store_bulk_volatile_b32(reinterpret_cast<volatile void*>(dst_line), expert_count_word);
                    else
                        __ppu_global_store_bulk_volatile_b32(reinterpret_cast<volatile void*>(dst_line), expert_count_word);
                }
                __threadfence();
                __syncthreads();

                if (warp_idx == 0) {
                    auto* const expert_pool_row_bases = reinterpret_cast<uint32_t*>(dispatch_base + kExpertPoolRowBasesOffsetBytes);
                    auto* const pool_row_cursor = reinterpret_cast<uint32_t*>(dispatch_base + kSlotAreaBytes);
                    constexpr uint32_t kPoolBasePrefixRounds = (kExpertsPerRank + 31u) / 32u;
                    uint32_t preceding_round_row_count = 0u;
                    for (uint32_t round = 0; round < kPoolBasePrefixRounds; ++round) {
                        const uint32_t expert = round * 32u + lane_idx;
                        uint32_t count = 0u;
                        if (expert < kExpertsPerRank) {
                            const uint32_t line = expert / kExpertCountWordsPerLine;
                            const uint32_t word_idx = expert % kExpertCountWordsPerLine;
                            for (uint32_t src = 0; src < kRankCount; ++src) {
                                const auto* const line_ptr = reinterpret_cast<const uint32_t*>(
                                    dispatch_base + kExpertCountsOffset +
                                    (static_cast<uint64_t>(src) * kExpertCountLines + line) * kCountLineBytes);
                                uint32_t count_word = 0u;
                                do {
                                    count_word = ptx::ld_volatile(line_ptr + word_idx);
                                    if ((count_word >> 16) != (current_launch_epoch & 0xFFFFu))
                                        __nanosleep(kFlagPollSleepNs);
                                } while ((count_word >> 16) != (current_launch_epoch & 0xFFFFu));
                                count += count_word & 0xFFFFu;
                            }
                        }
                        const uint32_t aligned_expert_row_count = (count + kPoolRowAlignment - 1u) / kPoolRowAlignment * kPoolRowAlignment;
                        uint32_t inclusive_row_sum = aligned_expert_row_count;
                        #pragma unroll
                        for (uint32_t shift = 1u; shift < 32u; shift <<= 1u) {
                            const uint32_t preceding_lane_row_sum = __shfl_up_sync(0xFFFFFFFFu, inclusive_row_sum, shift);
                            if (lane_idx >= shift)
                                inclusive_row_sum += preceding_lane_row_sum;
                        }
                        if (expert < kExpertsPerRank) {
                            const uint32_t expert_pool_row_base = inclusive_row_sum - aligned_expert_row_count + preceding_round_row_count;
                            expert_pool_row_bases[expert] = expert_pool_row_base;
                            pool_row_cursor[expert] = expert_pool_row_base;
                        }
                        preceding_round_row_count += __shfl_sync(0xFFFFFFFFu, inclusive_row_sum, 31u);
                    }
                }
                __syncthreads();
                if (thread_idx == 0) {
                    ptx::st_volatile(reinterpret_cast<const uint32_t*>(dispatch_base + kConsumeTaskCursorOffsetBytes), 0u);
                    __threadfence();
                    ptx::atomic_exch_rel(reinterpret_cast<uint32_t*>(dispatch_base + kExpertPoolLayoutReadyEpochOffsetBytes),
                                         current_launch_epoch & 0xFFFFu);
                }
            }
#endif
        }
        const bool in_dispatch_handoff = (is_push_sm || (is_quant_sm && quant_sm_set_idx < active_quant_sm_count));
        if (num_tokens != 0u && in_dispatch_handoff) {
            __syncthreads();
            if (thread_idx == 0) {
                const uint32_t handoff_role_count =
                    (is_push_sm ? 1u : 0u) + ((is_quant_sm && quant_sm_set_idx < active_quant_sm_count) ? 1u : 0u);
                const uint32_t handoff_required_role_count = push_sm_count + megamoe::min(active_quant_sm_count, quant_sm_count);
                const uint32_t handoff_addend = (is_push_sm && push_sm_set_idx == 0u)
                    ? (handoff_role_count + (ptx::kFinishSumTag - handoff_required_role_count))
                    : handoff_role_count;

                const uint32_t handoff_tag_before = ptx::ld_acq_gpu(dispatch_handoff_counter_ptr) & ptx::kFinishSumTag;
                ptx::red_add_rel(dispatch_handoff_counter_ptr, handoff_addend);
                if (is_push_sm) {
                    while ((ptx::ld_acq_gpu(dispatch_handoff_counter_ptr) & ptx::kFinishSumTag) == handoff_tag_before) {
                    }
                }
            }
            __syncthreads();
        }
    }

    {
        const auto run_dispatch = [&]() [[clang::noinline]] {
            DG_STATIC_ASSERT(kDispatchRegionBytes <= kDispatchReservedBytes, "the slot region must fit the dispatch-slot slack");
            DG_STATIC_ASSERT(kHiddenDim / 2 == 32 * 128 && kInputSfRowStrideBytes == 2 * 128,
                                  "slots are laid out for hidden=8192 (32 x lines + 2 sf lines)");

            constexpr uint32_t kXBulkTransfersPerToken = kHiddenDim / 2 / 512u;
            auto* const dispatch_base = reinterpret_cast<uint8_t*>(workspace.get_dispatch_slot_base_ptr());
            const auto* const stored_launch_epoch_ptr = reinterpret_cast<const uint32_t*>(dispatch_base + kStoredLaunchEpochOffsetBytes);
            const uint32_t current_launch_epoch = (ptx::ld_volatile(stored_launch_epoch_ptr) % 0xFFFFu) + 1u;
            auto get_slot_ptr = [&](uint32_t src_rank, uint32_t slot_idx) -> uint8_t* {
                return dispatch_base + (static_cast<uint64_t>(src_rank) * kMaxTokensPerSourceRank + slot_idx) * kSlotStrideBytes;
            };

            const auto store_dispatch_line = [&](uint8_t* const line_ptr, const uint32_t lane_word, const bool is_remote_destination) {
                if (is_remote_destination)
                    __ppu_remote_store_bulk_volatile_b32(reinterpret_cast<volatile void*>(line_ptr), lane_word);
                else
                    reinterpret_cast<volatile uint32_t*>(line_ptr)[lane_idx] = lane_word;
            };
            auto* const pool_row_cursor = reinterpret_cast<uint32_t*>(dispatch_base + kSlotAreaBytes);

            auto* const x_quant_base = input_token_buffer.get_base_ptr<uint8_t>();

            const auto store_dispatch_slot =
                [&](const uint32_t token_idx, uint8_t* const dst_slot, const bool is_remote_destination) [[clang::noinline]] {
                    uint32_t slot_header_word = 0u;
                    if (lane_idx >= 1u && lane_idx <= kTopkHeaderWordCount) {
                        const uint32_t topk_header_word_idx = lane_idx - 1u;
                        slot_header_word = (topk_header_word_idx % 2u == 0u)
                            ? *reinterpret_cast<const uint32_t*>(input_topk_expert_ids_base +
                                                                 static_cast<uint64_t>(token_idx) * kInputTopkIdxRowStrideBytes +
                                                                 static_cast<uint64_t>(topk_header_word_idx / 2u) * 8u)
                            : *reinterpret_cast<const uint32_t*>(input_topk_weight_base +
                                                                 static_cast<uint64_t>(token_idx) * kInputTopkWeightRowStrideBytes +
                                                                 static_cast<uint64_t>(topk_header_word_idx / 2u) * 4u);
                    } else if (lane_idx == kTopkHeaderWordCount + 1u) {
                        slot_header_word = token_idx;
                    }
                    const auto* const x_src = x_quant_base + static_cast<uint64_t>(token_idx) * (kHiddenDim / 2);
                    const uint2 sf_line_words = __ppu_global_load_bulk_volatile_b32x2(
                        reinterpret_cast<const void*>(input_sf_base + static_cast<uint64_t>(token_idx) * kInputSfRowStrideBytes));

                    uint32_t sf_displaced_bits = 0u;
                    if (lane_idx == kFlagLane)
                        sf_displaced_bits = (sf_line_words.x & kLineReadyMask) | ((sf_line_words.y & kLineReadyMask) << 1u);
                    sf_displaced_bits = __shfl_sync(0xFFFFFFFFu, sf_displaced_bits, kFlagLane);
                    if (lane_idx == kSfDisplacedBitsLaneIdx)
                        slot_header_word = sf_displaced_bits;
                    uint4 x_bulk_words[kXBulkTransfersPerToken];
                    #pragma unroll
                    for (uint32_t g = 0; g < kXBulkTransfersPerToken; ++g)
                        x_bulk_words[g] = __ppu_global_load_bulk_volatile_b32x4(reinterpret_cast<const void*>(x_src + 512u * g));

                    uint32_t x_displaced_bits = 0u;
                    if (lane_idx == kFlagLane) {
                        #pragma unroll
                        for (uint32_t g = 0; g < kXBulkTransfersPerToken; ++g)
                            x_displaced_bits |= ((x_bulk_words[g].x & kLineReadyMask) << (4u * g)) |
                                ((x_bulk_words[g].y & kLineReadyMask) << (4u * g + 1u)) |
                                ((x_bulk_words[g].z & kLineReadyMask) << (4u * g + 2u)) |
                                ((x_bulk_words[g].w & kLineReadyMask) << (4u * g + 3u));
                    }
                    x_displaced_bits = __shfl_sync(0xFFFFFFFFu, x_displaced_bits, kFlagLane);
                    if (lane_idx == kXDisplacedBitsLaneIdx)
                        slot_header_word = x_displaced_bits;

                    __syncwarp();

                    #pragma unroll
                    for (uint32_t m = 0; m < kXBulkTransfersPerToken; ++m) {
                        const uint4 x_bulk_words_with_flags = (lane_idx == kFlagLane) ? make_uint4(x_bulk_words[m].x | kLineReadyMask,
                                                                                                   x_bulk_words[m].y | kLineReadyMask,
                                                                                                   x_bulk_words[m].z | kLineReadyMask,
                                                                                                   x_bulk_words[m].w | kLineReadyMask)
                                                                                      : x_bulk_words[m];
                        if (is_remote_destination)
                            __ppu_remote_store_bulk_volatile_b32x4(
                                reinterpret_cast<volatile void*>(dst_slot + kSlotXOffsetBytes + 512u * m), x_bulk_words_with_flags);
                        else
                            __ppu_global_store_bulk_volatile_b32x4(
                                reinterpret_cast<volatile void*>(dst_slot + kSlotXOffsetBytes + 512u * m), x_bulk_words_with_flags);
                    }
                    const uint2 sf_bulk_words_with_flags = (lane_idx == kFlagLane)
                        ? make_uint2(sf_line_words.x | kLineReadyMask, sf_line_words.y | kLineReadyMask)
                        : make_uint2(sf_line_words.x, sf_line_words.y);
                    if (is_remote_destination)
                        __ppu_remote_store_bulk_volatile_b32x2(reinterpret_cast<volatile void*>(dst_slot + kSlotSfOffsetBytes),
                                                               sf_bulk_words_with_flags);
                    else
                        __ppu_global_store_bulk_volatile_b32x2(reinterpret_cast<volatile void*>(dst_slot + kSlotSfOffsetBytes),
                                                               sf_bulk_words_with_flags);

                    store_dispatch_line(dst_slot + kSlotHeaderOffsetBytes,
                                        (lane_idx == kFlagLane) ? kNonemptySlotHeaderReadyWord : slot_header_word,
                                        is_remote_destination);
                };

            if (is_push_sm) {
                const uint32_t token_begin = static_cast<uint32_t>(static_cast<uint64_t>(push_sm_set_idx) * num_tokens / push_sm_count);
                const uint32_t token_end = static_cast<uint32_t>(static_cast<uint64_t>(push_sm_set_idx + 1u) * num_tokens / push_sm_count);

                const uint32_t num_push_tokens = token_end - token_begin;
                if (num_push_tokens < kWarpsPerBlock * 2u) {
                    const uint32_t num_items = num_push_tokens * kRankCount;
                    for (uint32_t item = warp_idx; item < num_items; item += kWarpsPerBlock) {
                        const uint32_t token_idx = token_begin + item / kRankCount;
                        const uint32_t dst_rank = item % kRankCount;

                        if (item + kWarpsPerBlock < num_items) {
                            const uint32_t prefetch_token = token_begin + (item + kWarpsPerBlock) / kRankCount;
                            const auto* const next_x_src = x_quant_base + static_cast<uint64_t>(prefetch_token) * (kHiddenDim / 2);
                            #pragma unroll
                            for (uint32_t g = 0; g < kXBulkTransfersPerToken; ++g)
                                (void)__ppu_global_load_bulk_volatile_b32x4(reinterpret_cast<const void*>(next_x_src + 512u * g));
                            (void)__ppu_global_load_bulk_volatile_b32x2(reinterpret_cast<const void*>(
                                input_sf_base + static_cast<uint64_t>(prefetch_token) * kInputSfRowStrideBytes));
                        }
                        uint32_t lane_dst_rank = 0xFFFFFFFFu;
                        if (lane_idx < kTopkCount) {
                            const int32_t expert_idx = *reinterpret_cast<const int32_t*>(
                                input_topk_expert_ids_base + static_cast<uint64_t>(token_idx) * kInputTopkIdxRowStrideBytes +
                                static_cast<uint64_t>(lane_idx) * 8u);
                            if (expert_idx >= 0)
                                lane_dst_rank = static_cast<uint32_t>(expert_idx) / kExpertsPerRank;
                        }
                        uint8_t* const dst_slot = sym_buffer.map_pointer_to_rank(get_slot_ptr(sym_buffer.rank_idx, token_idx), dst_rank);
                        const bool is_remote_destination = (dst_rank != sym_buffer.rank_idx);
                        if (__ballot_sync(0xFFFFFFFFu, lane_dst_rank == dst_rank) == 0u) {
                            if (lane_idx == kFlagLane)
                                ptx::st_volatile(reinterpret_cast<const uint32_t*>(dst_slot + kSlotHeaderOffsetBytes + 124u),
                                                 kEmptySlotHeaderReadyWord);
                            continue;
                        }
                        store_dispatch_slot(token_idx, dst_slot, is_remote_destination);
                    }
                } else {
                    for (uint32_t token_idx = token_begin + warp_idx; token_idx < token_end; token_idx += kWarpsPerBlock) {
                        uint32_t lane_dst_rank = 0xFFFFFFFFu;
                        if (lane_idx < kTopkCount) {
                            const int32_t expert_idx = *reinterpret_cast<const int32_t*>(
                                input_topk_expert_ids_base + static_cast<uint64_t>(token_idx) * kInputTopkIdxRowStrideBytes +
                                static_cast<uint64_t>(lane_idx) * 8u);
                            if (expert_idx >= 0)
                                lane_dst_rank = static_cast<uint32_t>(expert_idx) / kExpertsPerRank;
                        }
                        uint32_t dst_local_expert_lane_mask = 0u;
                        #pragma unroll
                        for (uint32_t d = 0; d < kRankCount; ++d)
                            dst_local_expert_lane_mask |= (__ballot_sync(0xFFFFFFFFu, lane_dst_rank == d) != 0u) ? (1u << d) : 0u;

                        uint32_t slot_header_word = 0u;
                        if (lane_idx >= 1u && lane_idx <= kTopkHeaderWordCount) {
                            const uint32_t topk_header_word_idx = lane_idx - 1u;
                            slot_header_word = (topk_header_word_idx % 2u == 0u)
                                ? *reinterpret_cast<const uint32_t*>(input_topk_expert_ids_base +
                                                                     static_cast<uint64_t>(token_idx) * kInputTopkIdxRowStrideBytes +
                                                                     static_cast<uint64_t>(topk_header_word_idx / 2u) * 8u)
                                : *reinterpret_cast<const uint32_t*>(input_topk_weight_base +
                                                                     static_cast<uint64_t>(token_idx) * kInputTopkWeightRowStrideBytes +
                                                                     static_cast<uint64_t>(topk_header_word_idx / 2u) * 4u);
                        } else if (lane_idx == kTopkHeaderWordCount + 1u) {
                            slot_header_word = token_idx;
                        }
                        const auto* const x_src = x_quant_base + static_cast<uint64_t>(token_idx) * (kHiddenDim / 2);

                        if (token_idx + kWarpsPerBlock < token_end) {
                            const auto* const next_x_src = x_src + static_cast<uint64_t>(kWarpsPerBlock) * (kHiddenDim / 2);
                            #pragma unroll
                            for (uint32_t g = 0; g < kXBulkTransfersPerToken; ++g)
                                (void)__ppu_global_load_bulk_volatile_b32x4(reinterpret_cast<const void*>(next_x_src + 512u * g));
                        }
                        const uint2 sf_line_words = __ppu_global_load_bulk_volatile_b32x2(
                            reinterpret_cast<const void*>(input_sf_base + static_cast<uint64_t>(token_idx) * kInputSfRowStrideBytes));

                        uint32_t sf_displaced_bits = 0u;
                        if (lane_idx == kFlagLane)
                            sf_displaced_bits = (sf_line_words.x & kLineReadyMask) | ((sf_line_words.y & kLineReadyMask) << 1u);
                        sf_displaced_bits = __shfl_sync(0xFFFFFFFFu, sf_displaced_bits, kFlagLane);
                        if (lane_idx == kSfDisplacedBitsLaneIdx)
                            slot_header_word = sf_displaced_bits;

                        for (uint32_t d = 0; d < kRankCount; ++d) {
                            uint8_t* const dst_slot = sym_buffer.map_pointer_to_rank(get_slot_ptr(sym_buffer.rank_idx, token_idx), d);
                            if (((dst_local_expert_lane_mask >> d) & 1u) == 0u) {
                                if (lane_idx == kFlagLane)
                                    ptx::st_volatile(reinterpret_cast<const uint32_t*>(dst_slot + kSlotHeaderOffsetBytes + 124u),
                                                     kEmptySlotHeaderReadyWord);
                                continue;
                            }
                            const bool is_remote_destination = (d != sym_buffer.rank_idx);

                            uint4 x_bulk_words[kXBulkTransfersPerToken];
                            #pragma unroll
                            for (uint32_t g = 0; g < kXBulkTransfersPerToken; ++g)
                                x_bulk_words[g] = __ppu_global_load_bulk_volatile_b32x4(reinterpret_cast<const void*>(x_src + 512u * g));

                            uint32_t x_displaced_bits = 0u;
                            if (lane_idx == kFlagLane) {
                                #pragma unroll
                                for (uint32_t g = 0; g < kXBulkTransfersPerToken; ++g)
                                    x_displaced_bits |= ((x_bulk_words[g].x & kLineReadyMask) << (4u * g)) |
                                        ((x_bulk_words[g].y & kLineReadyMask) << (4u * g + 1u)) |
                                        ((x_bulk_words[g].z & kLineReadyMask) << (4u * g + 2u)) |
                                        ((x_bulk_words[g].w & kLineReadyMask) << (4u * g + 3u));
                            }
                            x_displaced_bits = __shfl_sync(0xFFFFFFFFu, x_displaced_bits, kFlagLane);
                            if (lane_idx == kXDisplacedBitsLaneIdx)
                                slot_header_word = x_displaced_bits;

                            __syncwarp();
                            #pragma unroll
                            for (uint32_t g = 0; g < kXBulkTransfersPerToken; ++g) {
                                const uint4 x_bulk_words_with_flags = (lane_idx == kFlagLane)
                                    ? make_uint4(x_bulk_words[g].x | kLineReadyMask,
                                                 x_bulk_words[g].y | kLineReadyMask,
                                                 x_bulk_words[g].z | kLineReadyMask,
                                                 x_bulk_words[g].w | kLineReadyMask)
                                    : x_bulk_words[g];
                                if (is_remote_destination)
                                    __ppu_remote_store_bulk_volatile_b32x4(
                                        reinterpret_cast<volatile void*>(dst_slot + kSlotXOffsetBytes + 512u * g), x_bulk_words_with_flags);
                                else
                                    __ppu_global_store_bulk_volatile_b32x4(
                                        reinterpret_cast<volatile void*>(dst_slot + kSlotXOffsetBytes + 512u * g), x_bulk_words_with_flags);
                            }

                            const uint2 sf_bulk_words_with_flags = (lane_idx == kFlagLane)
                                ? make_uint2(sf_line_words.x | kLineReadyMask, sf_line_words.y | kLineReadyMask)
                                : make_uint2(sf_line_words.x, sf_line_words.y);
                            if (is_remote_destination)
                                __ppu_remote_store_bulk_volatile_b32x2(reinterpret_cast<volatile void*>(dst_slot + kSlotSfOffsetBytes),
                                                                       sf_bulk_words_with_flags);
                            else
                                __ppu_global_store_bulk_volatile_b32x2(reinterpret_cast<volatile void*>(dst_slot + kSlotSfOffsetBytes),
                                                                       sf_bulk_words_with_flags);

                            store_dispatch_line(dst_slot + kSlotHeaderOffsetBytes,
                                                (lane_idx == kFlagLane) ? kNonemptySlotHeaderReadyWord : slot_header_word,
                                                is_remote_destination);
                        }
                    }
                }
            }

            const auto run_consume = [&]() [[clang::noinline]] {
#if DG_PREFILL
                if constexpr (kUsesDynamicPool) {
                    if (thread_idx == 0) {
                        auto* const expert_pool_layout_ready_epoch_ptr = reinterpret_cast<uint32_t*>(
                            reinterpret_cast<uint8_t*>(workspace.get_dispatch_slot_base_ptr()) + kExpertPoolLayoutReadyEpochOffsetBytes);
                        while (ptx::ld_acq_gpu(expert_pool_layout_ready_epoch_ptr) != (current_launch_epoch & 0xFFFFu))
                            __nanosleep(kFlagPollSleepNs);
                    }
                    __syncthreads();
                }
#endif
#if DG_PREFILL

                uint32_t source_token_count[kRankCount];
                uint32_t consume_max_source_count = 0u;
                #pragma unroll
                for (uint32_t s = 0; s < kRankCount; ++s) {
                    const uint32_t* const count_ptr = reinterpret_cast<const uint32_t*>(dispatch_base + kSourceTokenCountsOffsetBytes +
                                                                                        static_cast<uint64_t>(s) * kCountLineBytes);
                    uint32_t count_word = ptx::ld_volatile(count_ptr);
                    while ((count_word >> 16) != (current_launch_epoch & 0xFFFFu)) {
                        __nanosleep(kFlagPollSleepNs);
                        count_word = ptx::ld_volatile(count_ptr);
                    }
                    source_token_count[s] = megamoe::min(count_word & 0xFFFFu, kDeclaredMaxTokensPerRank);
                    consume_max_source_count = megamoe::max(consume_max_source_count, source_token_count[s]);
                }
                const uint32_t consume_push_rounds = (consume_max_source_count + push_sm_count - 1u) / push_sm_count;
                const uint32_t consume_task_count = consume_push_rounds * push_sm_count * kRankCount;
                auto* const consume_task_cursor = reinterpret_cast<uint32_t*>(dispatch_base + kConsumeTaskCursorOffsetBytes);

                const auto consume_task_token = [&](const uint32_t source, const uint32_t push_idx) -> uint32_t {
                    const uint32_t slice_begin = static_cast<uint32_t>(static_cast<uint64_t>(source_token_count[source]) *
                                                                       (push_idx % push_sm_count) / push_sm_count);
                    return slice_begin + push_idx / push_sm_count;
                };
                const auto consume_task_in_slice = [&](const uint32_t source, const uint32_t push_idx) -> bool {
                    return consume_task_token(source, push_idx) < static_cast<uint32_t>(static_cast<uint64_t>(source_token_count[source]) *
                                                                                        (push_idx % push_sm_count + 1u) / push_sm_count);
                };

                constexpr uint32_t kConsumeTaskBatch = 8u;
#else
                auto* const smem_source_token_counts = reinterpret_cast<uint32_t*>(smem_buffer);
                if (warp_idx == 0u) {
                    for (uint32_t s = lane_idx; s < kRankCount; s += 32u) {
                        const uint32_t* const count_ptr = reinterpret_cast<const uint32_t*>(dispatch_base + kSourceTokenCountsOffsetBytes +
                                                                                            static_cast<uint64_t>(s) * kCountLineBytes);
                        uint32_t count_word = ptx::ld_volatile(count_ptr);
                        while ((count_word >> 16) != (current_launch_epoch & 0xFFFFu)) {
                            __nanosleep(kFlagPollSleepNs);
                            count_word = ptx::ld_volatile(count_ptr);
                        }
                        smem_source_token_counts[s] = megamoe::min(count_word & 0xFFFFu, kDeclaredMaxTokensPerRank);
                    }
                }
                __syncthreads();
                uint32_t consume_token_rounds = 0u;
                #pragma unroll
                for (uint32_t s = 0; s < kRankCount; ++s)
                    consume_token_rounds = megamoe::max(consume_token_rounds, smem_source_token_counts[s]);
                const uint32_t consume_task_count = consume_token_rounds * kRankCount;
                auto* const consume_task_cursor = reinterpret_cast<uint32_t*>(dispatch_base + kConsumeTaskCursorOffsetBytes);

                const uint32_t consume_task_batch = megamoe::max(1u, consume_task_count / (kSmCount * kWarpsPerBlock));
#endif

                uint32_t consume_task_base = 0u, consume_task_off = 0u, consume_task_off_end = 0u;
                for (;;) {
                    uint32_t task_source = 0u, slot_idx = 0u;
#if DG_PREFILL
                    if (consume_task_off >= consume_task_off_end) {
                        uint32_t grabbed_task_base = 0u;
                        if (lane_idx == 0u)
                            grabbed_task_base = atomicAdd(consume_task_cursor, kConsumeTaskBatch);
                        consume_task_base = __shfl_sync(0xFFFFFFFFu, grabbed_task_base, 0u);
                        if (consume_task_base >= consume_task_count)
                            break;
                        consume_task_off = 0u;
                        consume_task_off_end = kConsumeTaskBatch;
                    }
                    const uint32_t consume_task_idx = consume_task_base + consume_task_off;
                    ++consume_task_off;
                    if (consume_task_idx >= consume_task_count)
                        break;
                    task_source = consume_task_idx % kRankCount;
                    const uint32_t consume_push_idx = consume_task_idx / kRankCount;
                    if (!consume_task_in_slice(task_source, consume_push_idx))
                        continue;
                    slot_idx = consume_task_token(task_source, consume_push_idx);
#else
                    if (consume_task_off >= consume_task_off_end) {
                        uint32_t grabbed_task_base = 0u;
                        if (lane_idx == 0u) {
                            grabbed_task_base = ptx::ld_volatile(reinterpret_cast<const uint32_t*>(consume_task_cursor));
                            if (grabbed_task_base < consume_task_count)
                                grabbed_task_base = atomicAdd(consume_task_cursor, consume_task_batch);
                        }
                        grabbed_task_base = __shfl_sync(0xFFFFFFFFu, grabbed_task_base, 0u);
                        if (grabbed_task_base >= consume_task_count)
                            break;
                        consume_task_base = grabbed_task_base;
                        consume_task_off = 0u;
                        consume_task_off_end = consume_task_batch;
                    }
                    const uint32_t consume_task_idx = consume_task_base + consume_task_off;
                    ++consume_task_off;
                    if (consume_task_idx >= consume_task_count)
                        break;
                    task_source = consume_task_idx % kRankCount;
                    const uint32_t consume_token_idx = consume_task_idx / kRankCount;
                    if (consume_token_idx >= smem_source_token_counts[task_source])
                        continue;
                    slot_idx = consume_token_idx;
#endif
                    uint8_t* const slot = get_slot_ptr(task_source, slot_idx);

                    uint32_t gate_word = 0u;
                    do {
                        gate_word = ptx::ld_volatile(reinterpret_cast<const uint32_t*>(slot + kSlotHeaderOffsetBytes + 124u));
                        if ((gate_word & kLineReadyMask) == 0u)
                            __nanosleep(kFlagPollSleepNs);
                    } while ((gate_word & kLineReadyMask) == 0u);
                    if ((gate_word & kSlotEmptyMask) != 0u) {
                        if (lane_idx == kFlagLane)
                            ptx::st_volatile(reinterpret_cast<const uint32_t*>(slot + kSlotHeaderOffsetBytes + 124u), 0u);
                        continue;
                    }

                    uint4 x_bulk_words[kXBulkTransfersPerToken];
                    uint32_t slot_header_word = 0u, sf_line0_word = 0u, sf_line1_word = 0u;

                    bool use_volatile_loads = false;
                    for (;;) {
                        #pragma unroll
                        for (uint32_t g = 0; g < kXBulkTransfersPerToken; ++g) {
                            const void* const src = reinterpret_cast<const void*>(slot + kSlotXOffsetBytes + 512u * g);
                            x_bulk_words[g] =
                                use_volatile_loads ? __ppu_global_load_bulk_volatile_b32x4(src) : __ppu_global_load_bulk_b32x4(src);
                        }
                        const uint32_t* const slot_header_word_ptr =
                            reinterpret_cast<const uint32_t*>(slot + kSlotHeaderOffsetBytes) + lane_idx;
                        const uint32_t* const sf_line0_word_ptr = reinterpret_cast<const uint32_t*>(slot + kSlotSfOffsetBytes) + lane_idx;
                        const uint32_t* const sf_line1_word_ptr =
                            reinterpret_cast<const uint32_t*>(slot + kSlotSfOffsetBytes + 128u) + lane_idx;
                        slot_header_word = use_volatile_loads ? ptx::ld_volatile(slot_header_word_ptr) : *slot_header_word_ptr;
                        sf_line0_word = use_volatile_loads ? ptx::ld_volatile(sf_line0_word_ptr) : *sf_line0_word_ptr;
                        sf_line1_word = use_volatile_loads ? ptx::ld_volatile(sf_line1_word_ptr) : *sf_line1_word_ptr;

                        uint32_t has_stale_line = 0u;
                        if (lane_idx == kFlagLane) {
                            #pragma unroll
                            for (uint32_t g = 0; g < kXBulkTransfersPerToken; ++g) {
                                #pragma unroll
                                for (uint32_t c = 0; c < 4u; ++c)
                                    has_stale_line |=
                                        (reinterpret_cast<const uint32_t*>(&x_bulk_words[g])[c] & kLineReadyMask) == 0u ? 1u : 0u;
                            }
                            has_stale_line |= (sf_line0_word & kLineReadyMask) == 0u ? 1u : 0u;
                            has_stale_line |= (sf_line1_word & kLineReadyMask) == 0u ? 1u : 0u;
                        }
                        if (__reduce_or_sync(0xFFFFFFFFu, has_stale_line) == 0u)
                            break;
                        use_volatile_loads = true;

                        __nanosleep(kFlagPollSleepNs);
                    }

                    const uint32_t x_displaced_bits = __shfl_sync(0xFFFFFFFFu, slot_header_word, kXDisplacedBitsLaneIdx);
                    const uint32_t sf_displaced_bits = __shfl_sync(0xFFFFFFFFu, slot_header_word, kSfDisplacedBitsLaneIdx);

                    if (lane_idx < kXBulkTransfersPerToken * 4u)
                        ptx::st_volatile(reinterpret_cast<const uint32_t*>(slot + kSlotXOffsetBytes + 128u * lane_idx + 124u), 0u);
                    if (lane_idx < 2u)
                        ptx::st_volatile(reinterpret_cast<const uint32_t*>(slot + kSlotSfOffsetBytes + 128u * lane_idx + 124u), 0u);
                    if (lane_idx == kFlagLane)
                        ptx::st_volatile(reinterpret_cast<const uint32_t*>(slot + kSlotHeaderOffsetBytes + 124u), 0u);

                    if (lane_idx == kFlagLane) {
                        #pragma unroll
                        for (uint32_t g = 0; g < kXBulkTransfersPerToken; ++g) {
                            #pragma unroll
                            for (uint32_t c = 0; c < 4u; ++c) {
                                uint32_t& word = reinterpret_cast<uint32_t*>(&x_bulk_words[g])[c];
                                word = (word & ~kLineReadyMask) | ((x_displaced_bits >> (4u * g + c)) & kLineReadyMask);
                            }
                        }
                        sf_line0_word = (sf_line0_word & ~kLineReadyMask) | (sf_displaced_bits & kLineReadyMask);
                        sf_line1_word = (sf_line1_word & ~kLineReadyMask) | ((sf_displaced_bits >> 1u) & kLineReadyMask);
                    }

                    uint32_t is_local_expert = 0u, allocated_pool_row = 0u;
                    if ((lane_idx & 1u) == 1u && lane_idx <= 2u * kTopkCount - 1u) {
                        const int32_t expert_idx = static_cast<int32_t>(slot_header_word);
                        if (expert_idx >= 0 && static_cast<uint32_t>(expert_idx) / kExpertsPerRank == sym_buffer.rank_idx) {
                            is_local_expert = 1u;
                            allocated_pool_row =
                                ptx::atomic_add(pool_row_cursor + (static_cast<uint32_t>(expert_idx) % kExpertsPerRank), 1u);
                        }
                    }
                    const uint32_t local_expert_lane_mask = __ballot_sync(0xFFFFFFFFu, is_local_expert);
                    if (local_expert_lane_mask == 0u)
                        continue;
                    const uint32_t slot_token_idx = __shfl_sync(0xFFFFFFFFu, slot_header_word, kTopkHeaderWordCount + 1u);

                    auto get_l1_sf_word_offset = [](uint32_t word_idx, uint32_t pool_row) -> uint32_t {
                        return (pool_row / BLOCK_M) * kActivationSfTileWords + word_idx * BLOCK_M + (pool_row % BLOCK_M);
                    };
                    uint32_t remaining_expert_lane_mask = local_expert_lane_mask;
                    while (remaining_expert_lane_mask != 0u) {
                        const uint32_t expert_lane_idx = __ffs(remaining_expert_lane_mask) - 1u;
                        remaining_expert_lane_mask &= remaining_expert_lane_mask - 1u;
                        const uint32_t pool_row = __shfl_sync(0xFFFFFFFFu, allocated_pool_row, expert_lane_idx);
                        const uint32_t weight_bits = __shfl_sync(0xFFFFFFFFu, slot_header_word, expert_lane_idx + 1u);
                        const uint32_t topk_idx = (expert_lane_idx - 1u) / 2u;
                        auto* const pool_x = l1_token_buffer.get_token_layout(pool_row).get_base_ptr<uint8_t>();

                        #pragma unroll
                        for (uint32_t g = 0; g < kXBulkTransfersPerToken; ++g)
                            __ppu_global_store_bulk_volatile_b32x4(reinterpret_cast<volatile void*>(pool_x + 512u * g), x_bulk_words[g]);
                        auto* const pool_sf = l1_sf_buffer.get_base_ptr<uint32_t>();

#if DG_PREFILL || DG_TILED_GEMM
                        if constexpr (kIntermediateDim >= 768 && kIntermediateDim % 256 == 0) {
                            auto* const scale_pairs = reinterpret_cast<uint16_t*>(pool_sf);
                            scale_pairs[(static_cast<uint64_t>(lane_idx * 2u) * kSfPoolRowCapacity) + pool_row] =
                                static_cast<uint16_t>(sf_line0_word);
                            scale_pairs[(static_cast<uint64_t>(lane_idx * 2u + 1u) * kSfPoolRowCapacity) + pool_row] =
                                static_cast<uint16_t>(sf_line0_word >> 16);
                            scale_pairs[(static_cast<uint64_t>(64u + lane_idx * 2u) * kSfPoolRowCapacity) + pool_row] =
                                static_cast<uint16_t>(sf_line1_word);
                            scale_pairs[(static_cast<uint64_t>(65u + lane_idx * 2u) * kSfPoolRowCapacity) + pool_row] =
                                static_cast<uint16_t>(sf_line1_word >> 16);
                        } else
#endif
                        {
                            pool_sf[get_l1_sf_word_offset(lane_idx, pool_row)] = sf_line0_word;
                            pool_sf[get_l1_sf_word_offset(32u + lane_idx, pool_row)] = sf_line1_word;
                        }
                        if (lane_idx == 0u) {
                            *l1_topk_weights_buffer.get_token_layout(pool_row).get_base_ptr<float>() = __uint_as_float(weight_bits);
#if DG_PREFILL || DG_TILED_GEMM
                            if constexpr (kIntermediateDim >= 768 && kIntermediateDim % 256 == 0) {
                                auto* const local_destination = combine_token_buffer.get_topk_buffer(topk_idx)
                                                                    .get_token_layout(slot_token_idx)
                                                                    .get_base_ptr<__nv_bfloat16>();
                                reinterpret_cast<int64_t*>(pool_metadata_base)[pool_row] =
                                    reinterpret_cast<int64_t>(sym_buffer.map_pointer_to_rank(local_destination, task_source));
                            } else
#endif
                            {
                                *(pool_metadata_base + pool_row) = {task_source, slot_token_idx, topk_idx};
                            }
                        }
                    }
                }
                __threadfence();
                __syncthreads();

                if (warp_idx == 0 && logical_sm_idx == 0u) {
                    if (lane_idx == 0u) {
                        auto* const dispatch_arrival_count_ptr =
                            reinterpret_cast<const uint32_t*>(workspace.get_dispatch_arrival_count_ptr());
                        while (static_cast<uint32_t>(ptx::ld_acq_gpu(dispatch_arrival_count_ptr)) < kSmCount - 1u)
                            __nanosleep(64u);
                    }
                    __syncwarp();
#if DG_PREFILL
                    const auto* const expert_pool_row_bases = reinterpret_cast<const uint32_t*>(
                        reinterpret_cast<const uint8_t*>(workspace.get_dispatch_slot_base_ptr()) + kExpertPoolRowBasesOffsetBytes);

                    for (uint32_t e = lane_idx; e < kExpertsPerRank; e += 32u) {
                        const uint32_t row_base = kUsesDynamicPool ? ptx::ld_volatile(expert_pool_row_bases + e)
                                                                   : static_cast<uint32_t>(e * kReservedMBlocksPerExpert * BLOCK_M);
                        const uint32_t count = ptx::ld_volatile(pool_row_cursor + e) - row_base;
                        *workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, e) = static_cast<uint64_t>(count);
                        if (count > 0 && cumulative_local_expert_recv_counts != nullptr)
                            ptx::red_add(cumulative_local_expert_recv_counts + e, static_cast<int>(count));
                    }
#else
                    for (uint32_t e = lane_idx; e < kExpertsPerRank; e += 32u) {
                        const uint32_t count =
                            ptx::ld_volatile(pool_row_cursor + e) - static_cast<uint32_t>(e * kReservedMBlocksPerExpert * BLOCK_M);
                        *workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, e) = static_cast<uint64_t>(count);
                        if (count > 0 && cumulative_local_expert_recv_counts != nullptr)
                            ptx::red_add(cumulative_local_expert_recv_counts + e, static_cast<int>(count));
                    }
#endif
                }
                if (warp_idx == 0) {
                    __threadfence();
                    __syncwarp();
                    if (lane_idx == 0) {
                        ptx::red_add_rel(reinterpret_cast<const uint32_t*>(workspace.get_dispatch_arrival_count_ptr()), 1u);
                    }
                }
            };
            run_consume();
        };
        run_dispatch();
    }

#if DG_PREFILL || DG_TILED_GEMM
    if constexpr (kIntermediateDim >= 768 && kIntermediateDim % 256 == 0) {
        auto run_prefill_gemm = [&]() {
            if (lane_idx == 0u) {
                while (static_cast<uint32_t>(ptx::ld_acq_gpu(workspace.get_dispatch_arrival_count_ptr())) < kSmCount)
                    __nanosleep(64u);
            }
            __syncwarp();
            auto* const expert_counts = reinterpret_cast<uint32_t*>(smem_buffer);
            auto* const pool_bases = expert_counts + kExpertsPerRank;
            for (uint32_t expert = thread_idx; expert < kExpertsPerRank; expert += kThreadsPerBlock) {
                expert_counts[expert] =
                    static_cast<uint32_t>(ptx::ld_volatile(workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, expert)));
#if DG_PREFILL
                if constexpr (kUsesDynamicPool) {
                    const auto* const published_bases = reinterpret_cast<const uint32_t*>(
                        reinterpret_cast<const uint8_t*>(workspace.get_dispatch_slot_base_ptr()) + kExpertPoolRowBasesOffsetBytes);
                    pool_bases[expert] = ptx::ld_volatile(published_bases + expert);
                } else
#endif
                {
                    pool_bases[expert] = expert * kReservedMBlocksPerExpert * BLOCK_M;
                }
            }
            __syncthreads();
            constexpr uint32_t shared_prefix = math::constexpr_align_up<uint32_t>(2 * kExpertsPerRank * sizeof(uint32_t), 1024u);
            constexpr uint32_t shared_tiles = 3 * (2 * 256 * 128 + 2 * 1280 * sizeof(uint16_t));
            DG_STATIC_ASSERT(shared_prefix + shared_tiles <= kDynamicSmemBytes, "prefill shared tiles exceed launch allocation");
            auto* const tile_buffer = smem_buffer + shared_prefix;
#if defined(DG_PREFILL_COLUMN_TILES)
            auto run_layer = [&](auto layer_tag) {
                constexpr bool kL1 = decltype(layer_tag)::value;
                constexpr int output_columns = kL1 ? L1_SHAPE_N : L2_SHAPE_N;
                constexpr int reduction_elements = kL1 ? L1_SHAPE_K : L2_SHAPE_K;
                constexpr uint32_t column_blocks = output_columns / 256;
                uint32_t tile_cursor = logical_sm_idx;
                for (uint32_t expert = 0; expert < kExpertsPerRank; ++expert) {
                    const uint32_t row_count = expert_counts[expert];
                    const uint32_t row_blocks = (row_count + 255u) / 256u;
                    const uint32_t tile_count = row_blocks * column_blocks;
                    for (; tile_cursor < tile_count; tile_cursor += kSmCount) {
                        const uint32_t row_block = tile_cursor / column_blocks;
                        int4 task = make_int4(expert,
                                              pool_bases[expert] + row_block * 256u,
                                              megamoe::min(256u, row_count - row_block * 256u),
                                              tile_cursor % column_blocks);
                        compute_prefill_gemm_tile<output_columns, reduction_elements, kL1, kHasActivationClamp, kPoolRowCapacity>(
                            kL1 ? l1_token_buffer.template get_base_ptr<uint8_t>() : l2_token_buffer.template get_base_ptr<uint8_t>(),
                            kL1 ? l1_sf_buffer.template get_base_ptr<uint16_t>() : l2_sf_buffer.template get_base_ptr<uint16_t>(),
                            kL1 ? l1_weights : l2_weights,
                            reinterpret_cast<const uint16_t*>(kL1 ? l1_weights_sf : l2_weights_sf),
                            l2_token_buffer.template get_base_ptr<uint8_t>(),
                            l2_sf_buffer.template get_base_ptr<uint16_t>(),
                            reinterpret_cast<const int64_t*>(pool_metadata_base),
                            l1_topk_weights_buffer.template get_base_ptr<float>(),
                            task,
                            activation_clamp,
                            tile_buffer);
                    }
                    tile_cursor -= tile_count;
                }
            };
            run_layer(std::true_type{});
            comm::grid_sync<kSmCount, 0>(workspace, logical_sm_idx, thread_idx, [&] { __syncthreads(); });
            run_layer(std::false_type{});
#else
            uint32_t block_cursor = logical_sm_idx;
            #pragma unroll 1
            for (uint32_t expert = 0; expert < kExpertsPerRank; ++expert) {
                const uint32_t row_count = expert_counts[expert];
                const uint32_t block_count = (row_count + 255u) / 256u;
                for (; block_cursor < block_count; block_cursor += kSmCount) {
                    int4 task =
                        make_int4(expert, pool_bases[expert] + block_cursor * 256u, megamoe::min(256u, row_count - block_cursor * 256u), 0);
                    #pragma unroll 1
                    for (int n = 0; n < L1_SHAPE_N / 256; ++n) {
                        task.w = n;
                        compute_prefill_gemm_tile<L1_SHAPE_N, L1_SHAPE_K, true, kHasActivationClamp, kPoolRowCapacity>(
                            l1_token_buffer.get_base_ptr<uint8_t>(),
                            l1_sf_buffer.get_base_ptr<uint16_t>(),
                            l1_weights,
                            reinterpret_cast<const uint16_t*>(l1_weights_sf),
                            l2_token_buffer.get_base_ptr<uint8_t>(),
                            l2_sf_buffer.get_base_ptr<uint16_t>(),
                            reinterpret_cast<const int64_t*>(pool_metadata_base),
                            l1_topk_weights_buffer.get_base_ptr<float>(),
                            task,
                            activation_clamp,
                            tile_buffer);
                    }
                    __syncthreads();
                    #pragma unroll 1
                    for (int n = 0; n < L2_SHAPE_N / 256; ++n) {
                        task.w = n;
                        compute_prefill_gemm_tile<L2_SHAPE_N, L2_SHAPE_K, false, kHasActivationClamp, kPoolRowCapacity>(
                            l2_token_buffer.get_base_ptr<uint8_t>(),
                            l2_sf_buffer.get_base_ptr<uint16_t>(),
                            l2_weights,
                            reinterpret_cast<const uint16_t*>(l2_weights_sf),
                            l2_token_buffer.get_base_ptr<uint8_t>(),
                            l2_sf_buffer.get_base_ptr<uint16_t>(),
                            reinterpret_cast<const int64_t*>(pool_metadata_base),
                            l1_topk_weights_buffer.get_base_ptr<float>(),
                            task,
                            activation_clamp,
                            tile_buffer);
                    }
                }
                block_cursor -= block_count;
            }
#endif
        };
        run_prefill_gemm();
    } else
#endif
    {

        auto smem_gemm_base = math::advance_ptr(smem_buffer, kSmemExpertScratchBytes);
        uint8_t* const smem_a_base = reinterpret_cast<uint8_t*>(smem_gemm_base);
        uint8_t* const smem_b_base = smem_a_base + kGemmAsfbStageCount * kSmemStageABytes;
        uint8_t* const smem_sfb_base = smem_b_base + kGemmBStageCount * kSmemStageBBytes;
        auto get_smem_a_stage_ptr = [=](uint32_t stage_idx) { return smem_a_base + stage_idx * kSmemStageABytes; };
        auto get_smem_b_stage_ptr = [=](uint32_t stage_idx) { return smem_b_base + stage_idx * kSmemStageBBytes; };
        auto get_smem_sfb_stage_ptr = [=](uint32_t stage_idx) {
            return reinterpret_cast<uint16_t*>(smem_sfb_base + stage_idx * kSmemStageSfbBytes);
        };
        uint8_t* const smem_asf_base_bytes = smem_sfb_base + kGemmAsfbStageCount * kSmemStageSfbBytes;
        uint32_t* const smem_asf = reinterpret_cast<uint32_t*>(smem_asf_base_bytes);

        auto scheduler = sched::BlockScheduler<BLOCK_M,
                                               BLOCK_N,
                                               BLOCK_K,
                                               L1_SHAPE_N,
                                               L1_SHAPE_K,
                                               L2_SHAPE_N,
                                               L2_SHAPE_K,
                                               kExpertsPerRank,
                                               kExpertsPerGroup,
                                               kSmCount>(workspace);

        struct GemmBlockDescriptor {
            sched::BlockPhase phase;
            uint32_t local_expert_idx, m_block_idx, n_block_idx, valid_row_count;
            uint32_t expert_m_block_count;
        };
        auto fetch_gemm_block = [&]() {
            DG_TIE_DECL(scheduler.get_next_block(), phase, expert_idx, m_block_idx, n_block_idx);
            GemmBlockDescriptor block;
            block.phase = phase;
            block.local_expert_idx = expert_idx;
            block.m_block_idx = m_block_idx;
            block.n_block_idx = n_block_idx;
            block.valid_row_count = scheduler.get_valid_row_count();
            block.expert_m_block_count = scheduler.get_current_m_block_count();
            return block;
        };

        constexpr uint32_t kWeightScaleRowsPerKTile = BLOCK_K / 64;

        constexpr uint32_t kL1WeightSfKRows = kHiddenDim / 64;
        constexpr uint32_t kL1WeightSfRowStrideElements = kIntermediateDim * 2;
        constexpr uint32_t kL2WeightSfKRows = kIntermediateDim / 64;
        constexpr uint32_t kL2WeightSfRowStrideElements = kHiddenDim;
        auto issue_first_weight_and_scale_tiles = [&](const GemmBlockDescriptor& block) {
            if (thread_idx != 0 || block.phase == sched::BlockPhase::kNoBlock)
                return;
            const bool is_l1 = (block.phase == sched::BlockPhase::kL1Gemm);
            const auto b_base = is_l1 ? reinterpret_cast<const uint8_t*>(l1_weights) +
                    (static_cast<uint64_t>(block.local_expert_idx) * kIntermediateDim * 2 + block.n_block_idx * BLOCK_N) * (kHiddenDim / 2)
                                      : reinterpret_cast<const uint8_t*>(l2_weights) +
                    (static_cast<uint64_t>(block.local_expert_idx) * kHiddenDim + block.n_block_idx * BLOCK_N) * (kIntermediateDim / 2);
            const uint32_t b_row_stride_bytes = (is_l1 ? kHiddenDim : kIntermediateDim) / 2;
            const auto sfb = is_l1 ? reinterpret_cast<const uint16_t*>(l1_weights_sf) +
                    (static_cast<uint64_t>(block.local_expert_idx) * kL1WeightSfKRows) * kL1WeightSfRowStrideElements +
                    block.n_block_idx * BLOCK_N
                                   : reinterpret_cast<const uint16_t*>(l2_weights_sf) +
                    (static_cast<uint64_t>(block.local_expert_idx) * kL2WeightSfKRows) * kL2WeightSfRowStrideElements +
                    block.n_block_idx * BLOCK_N;
            const uint32_t sfb_row_stride_bytes = (is_l1 ? kL1WeightSfRowStrideElements : kL2WeightSfRowStrideElements) * sizeof(uint16_t);
            ptx::ppu890::aiu_load_2d_swzl_b8(get_smem_b_stage_ptr(0), b_base, b_row_stride_bytes, BLOCK_N, BLOCK_K / 2, BLOCK_N, 0, 0);
            ptx::ppu890::aiu_load_2d_linear_b8(reinterpret_cast<uint8_t*>(get_smem_sfb_stage_ptr(0)),
                                               reinterpret_cast<const uint8_t*>(sfb),
                                               sfb_row_stride_bytes,
                                               is_l1 ? kL1WeightSfKRows : kL2WeightSfKRows,
                                               BLOCK_N * sizeof(uint16_t),
                                               kWeightScaleRowsPerKTile,
                                               0,
                                               0);
        };

        if ((threadIdx.x & 31u) == 0u) {
            auto* const dispatch_arrival_count_ptr = reinterpret_cast<const uint32_t*>(workspace.get_dispatch_arrival_count_ptr());
            while (static_cast<uint32_t>(ptx::ld_acq_gpu(dispatch_arrival_count_ptr)) < kSmCount)
                __nanosleep(64u);
        }
        __syncwarp();
        scheduler.fetch_local_expert_recv_counts(sym_buffer.rank_idx);
        scheduler.set_local_expert_idx(0);
        GemmBlockDescriptor current_block = fetch_gemm_block();
        issue_first_weight_and_scale_tiles(current_block);
        uint32_t l1_completed_local_expert_idx = 0xFFFFFFFFu;
#if DG_PREFILL

        auto* const smem_expert_pool_row_bases = reinterpret_cast<uint32_t*>(smem_buffer);
        if constexpr (kUsesDynamicPool) {
            const auto* const expert_pool_row_bases = reinterpret_cast<const uint32_t*>(
                reinterpret_cast<const uint8_t*>(workspace.get_dispatch_slot_base_ptr()) + kExpertPoolRowBasesOffsetBytes);
            for (uint32_t e = thread_idx; e < kExpertsPerRank; e += kThreadsPerBlock)
                smem_expert_pool_row_bases[e] = ptx::ld_volatile(expert_pool_row_bases + e);
            __syncthreads();
        }
        auto get_expert_pool_block_base = [&](const uint32_t expert_idx) -> uint32_t {
            if constexpr (!kUsesDynamicPool) {
                return expert_idx * kReservedMBlocksPerExpert;
            } else {
                return smem_expert_pool_row_bases[expert_idx] / BLOCK_M;
            }
        };
#else
        auto get_expert_pool_block_base = [&](const uint32_t expert_idx) -> uint32_t { return expert_idx * kReservedMBlocksPerExpert; };
#endif

        bool next_block_a_and_activation_scales_issued = false;
        auto issue_next_block_a_and_activation_scales = [&](const GemmBlockDescriptor& next_block) {
            const bool next_block_is_l1 = (next_block.phase == sched::BlockPhase::kL1Gemm);
            const uint32_t next_pool_block_idx = get_expert_pool_block_base(next_block.local_expert_idx) + next_block.m_block_idx;
            const uint32_t next_block_pool_row_base = next_pool_block_idx * BLOCK_M;
            const auto next_a_base = next_block_is_l1 ? l1_token_buffer.get_token_layout(next_block_pool_row_base).get_base_ptr<uint8_t>()
                                                      : l2_token_buffer.get_token_layout(next_block_pool_row_base).get_base_ptr<uint8_t>();
            const uint32_t next_a_row_stride_bytes = next_block_is_l1 ? kHiddenDim / 2 : kIntermediateDim / 2;
            ptx::ppu890::aiu_load_2d_swzl_b8(get_smem_a_stage_ptr(0),
                                             next_a_base,
                                             next_a_row_stride_bytes,
                                             next_block.valid_row_count,
                                             BLOCK_K / 2,
                                             next_block.valid_row_count,
                                             0,
                                             0);
            if (next_block_is_l1) {
                ptx::ppu890::aiu_load_2d_linear_b8(
                    reinterpret_cast<uint8_t*>(smem_asf),
                    reinterpret_cast<const uint8_t*>(l1_sf_buffer.get_base_ptr<uint32_t>() +
                                                     (uint64_t)(next_block_pool_row_base / BLOCK_M) * kActivationSfTileWords),
                    BLOCK_M * sizeof(uint32_t),
                    kActivationSfRows,
                    BLOCK_M * sizeof(uint32_t),
                    kActivationSfRows,
                    0,
                    0);
            } else {
                ptx::ppu890::aiu_load_2d_linear_b8(
                    reinterpret_cast<uint8_t*>(smem_asf),
                    reinterpret_cast<const uint8_t*>(l2_sf_buffer.get_base_ptr<uint32_t>() + next_block_pool_row_base),
                    kSfPoolRowCapacity * sizeof(uint32_t),
                    kIntermediateDim / 128,
                    BLOCK_M * sizeof(uint32_t),
                    kIntermediateDim / 128,
                    0,
                    0);
            }
        };
        while (current_block.phase != sched::BlockPhase::kNoBlock) {
            const sched::BlockPhase block_phase = current_block.phase;
            const uint32_t local_expert_idx = current_block.local_expert_idx;
            const uint32_t m_block_idx = current_block.m_block_idx;
            const uint32_t n_block_idx = current_block.n_block_idx;
            const uint32_t valid_row_count = current_block.valid_row_count;
            const uint32_t pool_block_idx = get_expert_pool_block_base(local_expert_idx) + m_block_idx;
            const uint32_t block_pool_row_base = pool_block_idx * BLOCK_M;

            const bool is_l1 = (block_phase == sched::BlockPhase::kL1Gemm);
            const uint32_t num_k_blocks = is_l1 ? kL1KTiles : kL2KTiles;

            const uint8_t* b_base;
            const uint16_t* sfb_base;
            uint32_t b_row_stride_bytes, sfb_row_stride_bytes;
            if (is_l1) {
                b_base = reinterpret_cast<const uint8_t*>(l1_weights) +
                    (static_cast<uint64_t>(local_expert_idx) * kIntermediateDim * 2 + n_block_idx * BLOCK_N) * (kHiddenDim / 2);
                sfb_base = reinterpret_cast<const uint16_t*>(l1_weights_sf) +
                    (static_cast<uint64_t>(local_expert_idx) * kL1WeightSfKRows) * kL1WeightSfRowStrideElements + n_block_idx * BLOCK_N;
                b_row_stride_bytes = kHiddenDim / 2;
                sfb_row_stride_bytes = kL1WeightSfRowStrideElements * sizeof(uint16_t);
            } else {
                b_base = reinterpret_cast<const uint8_t*>(l2_weights) +
                    (static_cast<uint64_t>(local_expert_idx) * kHiddenDim + n_block_idx * BLOCK_N) * (kIntermediateDim / 2);
                sfb_base = reinterpret_cast<const uint16_t*>(l2_weights_sf) +
                    (static_cast<uint64_t>(local_expert_idx) * kL2WeightSfKRows) * kL2WeightSfRowStrideElements + n_block_idx * BLOCK_N;
                b_row_stride_bytes = kIntermediateDim / 2;
                sfb_row_stride_bytes = kL2WeightSfRowStrideElements * sizeof(uint16_t);
            }
            auto load_sfb_tile = [&](uint32_t k_tile_idx, uint32_t stage_idx) {
                {
                    const uint32_t weight_sf_row_base = k_tile_idx * kWeightScaleRowsPerKTile;
                    ptx::ppu890::aiu_load_2d_linear_b8(
                        reinterpret_cast<uint8_t*>(get_smem_sfb_stage_ptr(stage_idx)),
                        reinterpret_cast<const uint8_t*>(
                            sfb_base + weight_sf_row_base * (is_l1 ? kL1WeightSfRowStrideElements : kL2WeightSfRowStrideElements)),
                        sfb_row_stride_bytes,
                        (is_l1 ? kL1WeightSfKRows : kL2WeightSfKRows),
                        BLOCK_N * sizeof(uint16_t),
                        kWeightScaleRowsPerKTile,
                        0,
                        0);
                }
            };
            const auto a_base_l1 = l1_token_buffer.get_token_layout(block_pool_row_base).get_base_ptr<uint8_t>();
            const auto a_base_l2 = l2_token_buffer.get_token_layout(block_pool_row_base).get_base_ptr<uint8_t>();
            const uint32_t a_row_stride_bytes = is_l1 ? kHiddenDim / 2 : kIntermediateDim / 2;
            const auto a_base = is_l1 ? a_base_l1 : a_base_l2;

            const uint32_t* const l1_activation_sf_base = l1_sf_buffer.get_base_ptr<uint32_t>();
            const uint32_t* const l2_activation_sf_base = l2_sf_buffer.get_base_ptr<uint32_t>();

            float mma_accum_by_m_tile[kMmaMTilesPerWarp * kMmaNTilesPerWarp * 4];
            #pragma unroll
            for (int i = 0; i < static_cast<int>(kMmaMTilesPerWarp * kMmaNTilesPerWarp * 4); ++i)
                mma_accum_by_m_tile[i] = 0.0f;

            auto load_asf_block = [&]() {
                if (is_l1) {
                    ptx::ppu890::aiu_load_2d_linear_b8(
                        reinterpret_cast<uint8_t*>(smem_asf),
                        reinterpret_cast<const uint8_t*>(l1_activation_sf_base +
                                                         (uint64_t)(block_pool_row_base / BLOCK_M) * kActivationSfTileWords),
                        BLOCK_M * sizeof(uint32_t),
                        kActivationSfRows,
                        BLOCK_M * sizeof(uint32_t),
                        kActivationSfRows,
                        0,
                        0);
                } else {
                    ptx::ppu890::aiu_load_2d_linear_b8(reinterpret_cast<uint8_t*>(smem_asf),
                                                       reinterpret_cast<const uint8_t*>(l2_activation_sf_base + block_pool_row_base),
                                                       kSfPoolRowCapacity * sizeof(uint32_t),
                                                       kIntermediateDim / 128,
                                                       BLOCK_M * sizeof(uint32_t),
                                                       kIntermediateDim / 128,
                                                       0,
                                                       0);
                }
            };
            constexpr uint32_t kATileIssuerThreadIdx = 0;
            constexpr uint32_t kBTileIssuerThreadIdx = 0;
            constexpr uint32_t kScaleTileIssuerThreadIdx = 0;

            auto load_k_tile = [&](uint32_t k_tile_idx, uint32_t stage_idx) {
                if (thread_idx == kATileIssuerThreadIdx)
                    ptx::ppu890::aiu_load_2d_swzl_b8(get_smem_a_stage_ptr(stage_idx),
                                                     a_base + k_tile_idx * (BLOCK_K / 2),
                                                     a_row_stride_bytes,
                                                     valid_row_count,
                                                     BLOCK_K / 2,
                                                     valid_row_count,
                                                     0,
                                                     0);
                if (thread_idx == kBTileIssuerThreadIdx)
                    ptx::ppu890::aiu_load_2d_swzl_b8(get_smem_b_stage_ptr(stage_idx),
                                                     b_base + k_tile_idx * (BLOCK_K / 2),
                                                     b_row_stride_bytes,
                                                     BLOCK_N,
                                                     BLOCK_K / 2,
                                                     BLOCK_N,
                                                     0,
                                                     0);
                if (thread_idx == kScaleTileIssuerThreadIdx)
                    load_sfb_tile(k_tile_idx, stage_idx);
            };
            auto load_a_and_weight_scale_tiles = [&](uint32_t k_tile_idx, uint32_t stage_idx) {
                if (thread_idx == kATileIssuerThreadIdx)
                    ptx::ppu890::aiu_load_2d_swzl_b8(get_smem_a_stage_ptr(stage_idx),
                                                     a_base + k_tile_idx * (BLOCK_K / 2),
                                                     a_row_stride_bytes,
                                                     valid_row_count,
                                                     BLOCK_K / 2,
                                                     valid_row_count,
                                                     0,
                                                     0);
                if (thread_idx == kScaleTileIssuerThreadIdx)
                    load_sfb_tile(k_tile_idx, stage_idx);
            };
            auto load_b_tile = [&](uint32_t k_tile_idx, uint32_t stage_idx) {
                if (thread_idx == kBTileIssuerThreadIdx)
                    ptx::ppu890::aiu_load_2d_swzl_b8(get_smem_b_stage_ptr(stage_idx),
                                                     b_base + k_tile_idx * (BLOCK_K / 2),
                                                     b_row_stride_bytes,
                                                     BLOCK_N,
                                                     BLOCK_K / 2,
                                                     BLOCK_N,
                                                     0,
                                                     0);
            };
            if (!is_l1 && local_expert_idx != l1_completed_local_expert_idx) {
                const auto l1_completed_block_count_ptr =
                    l1_completed_block_counts_base + local_expert_idx * kL1CompletedBlockCountLineWords;
                const uint32_t required_l1_completed_block_count = current_block.expert_m_block_count * (L1_SHAPE_N / BLOCK_N);
                if (thread_idx == 0)
                    while (ptx::ld_acq_gpu(l1_completed_block_count_ptr) < required_l1_completed_block_count) {
                    }
                ptx::sync_threads_partial(kThreadsPerBlock, kL1ReadyBarrierId);
                l1_completed_local_expert_idx = local_expert_idx;
            }
            if (thread_idx == 0 && !next_block_a_and_activation_scales_issued) {
                ptx::ppu890::aiu_load_2d_swzl_b8(get_smem_a_stage_ptr(0),
                                                 a_base + 0 * (BLOCK_K / 2),
                                                 a_row_stride_bytes,
                                                 valid_row_count,
                                                 BLOCK_K / 2,
                                                 valid_row_count,
                                                 0,
                                                 0);
                load_asf_block();
            }
            next_block_a_and_activation_scales_issued = false;
            ptx::cp_async_commit();

            #pragma unroll
            for (uint32_t s = 1; s + 1 < kGemmBStageCount; ++s) {
                if constexpr (kGemmAsfbStageCount == kGemmBStageCount) {
                    if (s < num_k_blocks)
                        load_k_tile(s, s);
                    ptx::cp_async_commit();
                } else {
                    if (s < num_k_blocks && s + 1 < kGemmAsfbStageCount)
                        load_a_and_weight_scale_tiles(s, s % kGemmAsfbStageCount);
                    ptx::cp_async_commit();
                    if (s < num_k_blocks)
                        load_b_tile(s, s % kGemmBStageCount);
                    ptx::cp_async_commit();
                }
            }

            const uint32_t n_group_idx = warp_idx / kWarpGridM;
            const int lane_group_idx = static_cast<int>(lane_idx) / 4;
            const int lane_in_group = static_cast<int>(lane_idx) % 4;
            DG_GEMM_K_LOOP_UNROLL
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
                ptx::cp_async_wait<kAllowedPendingCopyGroups>();
                ptx::sync_threads_partial(kThreadsPerBlock, kStageReadyBarrierId);
                if constexpr (kGemmAsfbStageCount == kGemmBStageCount) {
                    if (k_block_idx + (kGemmBStageCount - 1) < num_k_blocks) {
                        const uint32_t next_k_tile = k_block_idx + (kGemmBStageCount - 1);
                        load_k_tile(next_k_tile, next_k_tile % kGemmBStageCount);
                    }
                    ptx::cp_async_commit();
                } else {
                    if (k_block_idx + (kGemmAsfbStageCount - 1) < num_k_blocks) {
                        const uint32_t next_a_tile = k_block_idx + (kGemmAsfbStageCount - 1);
                        load_a_and_weight_scale_tiles(next_a_tile, next_a_tile % kGemmAsfbStageCount);
                    }
                    ptx::cp_async_commit();
                    if (k_block_idx + (kGemmBStageCount - 1) < num_k_blocks) {
                        const uint32_t next_b_tile = k_block_idx + (kGemmBStageCount - 1);
                        load_b_tile(next_b_tile, next_b_tile % kGemmBStageCount);
                    }
                    ptx::cp_async_commit();
                }
                auto smem_a = get_smem_a_stage_ptr(k_block_idx % kGemmAsfbStageCount);
                auto smem_b = get_smem_b_stage_ptr(k_block_idx % kGemmBStageCount);
                auto smem_sfb = get_smem_sfb_stage_ptr(k_block_idx % kGemmAsfbStageCount);

                {
                    const int contiguous_warp_row_offset =
                        static_cast<int>(warp_idx % kWarpGridM) * static_cast<int>(kMmaMTilesPerWarp) * 16;

                    const int first_warp_row_offset =
                        (kMmaMTilesPerWarp == 2) ? static_cast<int>(warp_idx % kWarpGridM) * 16 : contiguous_warp_row_offset;
                    if (static_cast<uint32_t>(first_warp_row_offset) < valid_row_count) {
                        const uint16_t* const sfb_stage = smem_sfb;
                        constexpr uint32_t kActivationSfWordsPerKTile = BLOCK_K / 128;
                        constexpr uint32_t kK64StepsPerTile = BLOCK_K / 64;
                        const uint32_t* const asf_base = smem_asf +
                            static_cast<uint32_t>(k_block_idx * kActivationSfWordsPerKTile) * BLOCK_M +
                            static_cast<uint32_t>(first_warp_row_offset) + static_cast<uint32_t>(lane_group_idx);

                        auto sf_fn = [&](int m_tile_idx,
                                         int n16_tile_idx,
                                         int k64_step_idx,
                                         uint32_t& activation_scale_word,
                                         uint32_t& weight_scale_word) {
                            const uint32_t sf_word_idx = static_cast<uint32_t>(k64_step_idx) >> 1;
                            const uint32_t asf_lo =
                                asf_base[sf_word_idx * BLOCK_M + static_cast<uint32_t>(m_tile_idx * kMmaMTileRowStride)];
                            const uint32_t asf_hi =
                                asf_base[sf_word_idx * BLOCK_M + static_cast<uint32_t>(m_tile_idx * kMmaMTileRowStride) + 8u];

                            const uint32_t asf_byte_perm_select = (static_cast<uint32_t>(k64_step_idx) & 1u) != 0u ? 0x7632u : 0x5410u;
                            activation_scale_word = __byte_perm(asf_lo, asf_hi, asf_byte_perm_select);
                            weight_scale_word =
                                reinterpret_cast<const uint32_t*>(sfb_stage + k64_step_idx * BLOCK_N +
                                                                  n_group_idx * kGemmOutputColsPerWarp + 8u * lane_group_idx)[n16_tile_idx];
                        };

                        if (is_l1) {
                            if constexpr (kMmaMTilesPerWarp != 2) {
                                ptx::ppu890::mma_regpipe_swzl_fp4_native_accum_m_loop<BLOCK_M,
                                                                                      kGemmOutputColsPerWarp,
                                                                                      BLOCK_K,
                                                                                      kK64StepsPerTile,
                                                                                      kMmaMTilesPerWarp>(
                                    smem_a,
                                    smem_b + static_cast<size_t>(n_group_idx) * kGemmOutputColsPerWarp * (BLOCK_K / 2),
                                    contiguous_warp_row_offset,
                                    0,
                                    sf_fn,
                                    mma_accum_by_m_tile);
                            } else {
                                if (first_warp_row_offset + kMmaMTileRowStride >= static_cast<int>(valid_row_count))
                                    ptx::ppu890::mma_regpipe_swzl_fp4_native_accum_m_loop<BLOCK_M,
                                                                                          kGemmOutputColsPerWarp,
                                                                                          BLOCK_K,
                                                                                          kK64StepsPerTile,
                                                                                          1,
                                                                                          kMmaMTileRowStride>(
                                        smem_a,
                                        smem_b + static_cast<size_t>(n_group_idx) * kGemmOutputColsPerWarp * (BLOCK_K / 2),
                                        first_warp_row_offset,
                                        0,
                                        sf_fn,
                                        mma_accum_by_m_tile);
                                else
                                    ptx::ppu890::mma_regpipe_swzl_fp4_native_accum_m_loop<BLOCK_M,
                                                                                          kGemmOutputColsPerWarp,
                                                                                          BLOCK_K,
                                                                                          kK64StepsPerTile,
                                                                                          2,
                                                                                          kMmaMTileRowStride>(
                                        smem_a,
                                        smem_b + static_cast<size_t>(n_group_idx) * kGemmOutputColsPerWarp * (BLOCK_K / 2),
                                        first_warp_row_offset,
                                        0,
                                        sf_fn,
                                        mma_accum_by_m_tile);
                            }
                        } else {
                            if constexpr (kMmaMTilesPerWarp != 2) {
                                ptx::ppu890::mma_regpipe_swzl_fp4_native_accum_m_loop<BLOCK_M,
                                                                                      kGemmOutputColsPerWarp,
                                                                                      BLOCK_K,
                                                                                      kK64StepsPerTile,
                                                                                      kMmaMTilesPerWarp>(
                                    smem_a,
                                    smem_b + static_cast<size_t>(n_group_idx) * kGemmOutputColsPerWarp * (BLOCK_K / 2),
                                    contiguous_warp_row_offset,
                                    0,
                                    sf_fn,
                                    mma_accum_by_m_tile);
                            } else {
                                if (first_warp_row_offset + kMmaMTileRowStride >= static_cast<int>(valid_row_count))
                                    ptx::ppu890::mma_regpipe_swzl_fp4_native_accum_m_loop<BLOCK_M,
                                                                                          kGemmOutputColsPerWarp,
                                                                                          BLOCK_K,
                                                                                          kK64StepsPerTile,
                                                                                          1,
                                                                                          kMmaMTileRowStride>(
                                        smem_a,
                                        smem_b + static_cast<size_t>(n_group_idx) * kGemmOutputColsPerWarp * (BLOCK_K / 2),
                                        first_warp_row_offset,
                                        0,
                                        sf_fn,
                                        mma_accum_by_m_tile);
                                else
                                    ptx::ppu890::mma_regpipe_swzl_fp4_native_accum_m_loop<BLOCK_M,
                                                                                          kGemmOutputColsPerWarp,
                                                                                          BLOCK_K,
                                                                                          kK64StepsPerTile,
                                                                                          2,
                                                                                          kMmaMTileRowStride>(
                                        smem_a,
                                        smem_b + static_cast<size_t>(n_group_idx) * kGemmOutputColsPerWarp * (BLOCK_K / 2),
                                        first_warp_row_offset,
                                        0,
                                        sf_fn,
                                        mma_accum_by_m_tile);
                            }
                        }
                    }
                }
            }
            ptx::sync_threads_partial(kThreadsPerBlock, kStageReuseBarrierId);
            const GemmBlockDescriptor next_block = fetch_gemm_block();
            issue_first_weight_and_scale_tiles(next_block);
            if (next_block.phase == sched::BlockPhase::kL1Gemm ||
                (next_block.phase == sched::BlockPhase::kL2Gemm && next_block.local_expert_idx == l1_completed_local_expert_idx &&
                 block_phase == sched::BlockPhase::kL2Gemm)) {
                if (thread_idx == 0)
                    issue_next_block_a_and_activation_scales(next_block);
                next_block_a_and_activation_scales_issued = true;
            }
            {
                const int epilogue_warp_row_offset = (kMmaMTilesPerWarp == 2)
                    ? static_cast<int>(warp_idx % kWarpGridM) * 16
                    : static_cast<int>(warp_idx % kWarpGridM) * static_cast<int>(kMmaMTilesPerWarp) * 16;
                #pragma unroll
                for (uint32_t m_tile_idx = 0; m_tile_idx < kMmaMTilesPerWarp; ++m_tile_idx) {
                    const int mma_tile_row_offset = epilogue_warp_row_offset + static_cast<int>(m_tile_idx) * kMmaMTileRowStride;
                    float* const mma_tile_accum = mma_accum_by_m_tile + m_tile_idx * kMmaNTilesPerWarp * 4;

                    if (is_l1) {
                        if (kSwigluColsPerWarp >= 64 ? (static_cast<uint32_t>(mma_tile_row_offset) < valid_row_count) : true) {
                            const uint32_t pool_row0 = block_pool_row_base + static_cast<uint32_t>(mma_tile_row_offset + lane_group_idx);
                            const uint32_t pool_row1 =
                                block_pool_row_base + static_cast<uint32_t>(mma_tile_row_offset + lane_group_idx + 8);
                            const bool is_row0_valid = static_cast<uint32_t>(mma_tile_row_offset + lane_group_idx) < valid_row_count;
                            const bool is_row1_valid = static_cast<uint32_t>(mma_tile_row_offset + lane_group_idx + 8) < valid_row_count;

                            float swiglu_out[kMmaNTilesPerWarp / 2 * 4];
                            #pragma unroll
                            for (int pair = 0; pair < static_cast<int>(kMmaNTilesPerWarp) / 2; ++pair) {
                                const int gate_tile_idx = pair * 2;
                                const int up_tile_idx = pair * 2 + 1;
                                float gate_value_0 = mma_tile_accum[gate_tile_idx * 4 + 0],
                                      up_value_0 = mma_tile_accum[up_tile_idx * 4 + 0];
                                float gate_value_1 = mma_tile_accum[gate_tile_idx * 4 + 1],
                                      up_value_1 = mma_tile_accum[up_tile_idx * 4 + 1];
                                float gate_value_2 = mma_tile_accum[gate_tile_idx * 4 + 2],
                                      up_value_2 = mma_tile_accum[up_tile_idx * 4 + 2];
                                float gate_value_3 = mma_tile_accum[gate_tile_idx * 4 + 3],
                                      up_value_3 = mma_tile_accum[up_tile_idx * 4 + 3];
                                swiglu_out[pair * 4 + 0] = ppu890_swiglu<kHasActivationClamp>(gate_value_0, up_value_0, activation_clamp) *
                                    (is_row0_valid ? 1.0f : 0.0f);
                                swiglu_out[pair * 4 + 1] = ppu890_swiglu<kHasActivationClamp>(gate_value_1, up_value_1, activation_clamp) *
                                    (is_row0_valid ? 1.0f : 0.0f);
                                swiglu_out[pair * 4 + 2] = ppu890_swiglu<kHasActivationClamp>(gate_value_2, up_value_2, activation_clamp) *
                                    (is_row1_valid ? 1.0f : 0.0f);
                                swiglu_out[pair * 4 + 3] = ppu890_swiglu<kHasActivationClamp>(gate_value_3, up_value_3, activation_clamp) *
                                    (is_row1_valid ? 1.0f : 0.0f);
                            }

                            constexpr uint32_t kSwigluScaleGroupsPerWarp = 1;
                            constexpr uint32_t kGateUpTilePairsPerWarp = kMmaNTilesPerWarp / 2;
                            constexpr uint32_t kGateUpTilePairsPerScaleGroup = kGateUpTilePairsPerWarp / kSwigluScaleGroupsPerWarp;
                            constexpr uint32_t kSwigluScaleGroupsPerNBlock = kSwigluColsPerBlock / 32;

                            auto l2_input_row0 = l2_token_buffer.get_token_layout(pool_row0).get_base_ptr<uint8_t>();
                            auto l2_input_row1 = l2_token_buffer.get_token_layout(pool_row1).get_base_ptr<uint8_t>();
                            const uint32_t out_col_base = n_block_idx * kSwigluColsPerBlock + n_group_idx * kSwigluColsPerWarp;

                            #pragma unroll
                            for (uint32_t sf_group_idx = 0; sf_group_idx < kSwigluScaleGroupsPerWarp; ++sf_group_idx) {
                                float amax_row0 = 0.0f, amax_row1 = 0.0f;
                                #pragma unroll
                                for (int pair = static_cast<int>(sf_group_idx * kGateUpTilePairsPerScaleGroup);
                                     pair < static_cast<int>((sf_group_idx + 1) * kGateUpTilePairsPerScaleGroup);
                                     ++pair) {
                                    amax_row0 = megamoe::max(amax_row0, fabsf(swiglu_out[pair * 4 + 0]));
                                    amax_row0 = megamoe::max(amax_row0, fabsf(swiglu_out[pair * 4 + 1]));
                                    amax_row1 = megamoe::max(amax_row1, fabsf(swiglu_out[pair * 4 + 2]));
                                    amax_row1 = megamoe::max(amax_row1, fabsf(swiglu_out[pair * 4 + 3]));
                                }
                                amax_row0 = megamoe::max(amax_row0, __shfl_xor_sync(0xffffffff, amax_row0, 1));
                                amax_row0 = megamoe::max(amax_row0, __shfl_xor_sync(0xffffffff, amax_row0, 2));
                                amax_row1 = megamoe::max(amax_row1, __shfl_xor_sync(0xffffffff, amax_row1, 1));
                                amax_row1 = megamoe::max(amax_row1, __shfl_xor_sync(0xffffffff, amax_row1, 2));

                                float2 row_scales, row_inverse_scales;
                                fp4::get_e2m1_sf_and_sf_inv(make_float2(amax_row0, amax_row1), row_scales, row_inverse_scales);

                                const uint32_t l2_sf_group =
                                    n_block_idx * kSwigluScaleGroupsPerNBlock + (n_group_idx * kSwigluColsPerWarp) / 32 + sf_group_idx;
                                const bool should_store_sf = (lane_in_group == 0);
                                if (should_store_sf) {
                                    auto l2_sf_base = l2_sf_buffer.get_base_ptr<uint8_t>();
                                    const uint32_t sf_word_group_base_offset = (l2_sf_group >> 2) * kSfPoolRowCapacity;
                                    const uint32_t sf_byte_in_word_idx = l2_sf_group & 3u;
                                    if (is_row0_valid)
                                        l2_sf_base[(sf_word_group_base_offset + pool_row0) * 4u + sf_byte_in_word_idx] =
                                            static_cast<uint8_t>(fp4::sf_to_ue8m0(row_scales.x));
                                    if (is_row1_valid)
                                        l2_sf_base[(sf_word_group_base_offset + pool_row1) * 4u + sf_byte_in_word_idx] =
                                            static_cast<uint8_t>(fp4::sf_to_ue8m0(row_scales.y));
                                }

                                DG_STATIC_ASSERT(kGateUpTilePairsPerScaleGroup == 4,
                                                      "K-permuted L1 store assumes a 4-byte run per lane");
                                uint32_t fp4_row0_word = 0u, fp4_row1_word = 0u;
                                #pragma unroll
                                for (int pair = static_cast<int>(sf_group_idx * kGateUpTilePairsPerScaleGroup);
                                     pair < static_cast<int>((sf_group_idx + 1) * kGateUpTilePairsPerScaleGroup);
                                     ++pair) {
                                    const uint32_t shift =
                                        static_cast<uint32_t>(pair - static_cast<int>(sf_group_idx * kGateUpTilePairsPerScaleGroup)) * 8u;
                                    const uint32_t fp4_row0_pair_byte = static_cast<uint32_t>(ptx::ppu890::cvt_f4x2_f32x2(
                                        swiglu_out[pair * 4 + 1] * row_inverse_scales.x, swiglu_out[pair * 4 + 0] * row_inverse_scales.x));
                                    const uint32_t fp4_row1_pair_byte = static_cast<uint32_t>(ptx::ppu890::cvt_f4x2_f32x2(
                                        swiglu_out[pair * 4 + 3] * row_inverse_scales.y, swiglu_out[pair * 4 + 2] * row_inverse_scales.y));
                                    fp4_row0_word |= fp4_row0_pair_byte << shift;
                                    fp4_row1_word |= fp4_row1_pair_byte << shift;
                                }
                                {
                                    const uint32_t output_byte_offset =
                                        (out_col_base >> 1) + sf_group_idx * 16u + static_cast<uint32_t>(lane_in_group) * 4u;
                                    if (is_row0_valid)
                                        *reinterpret_cast<uint32_t*>(l2_input_row0 + output_byte_offset) = fp4_row0_word;
                                    if (is_row1_valid)
                                        *reinterpret_cast<uint32_t*>(l2_input_row1 + output_byte_offset) = fp4_row1_word;
                                }
                            }
                        }

                    } else {
                        if (static_cast<uint32_t>(mma_tile_row_offset) < valid_row_count) {
                            const uint32_t n_base = n_block_idx * BLOCK_N;
                            const uint32_t pool_row0 = block_pool_row_base + static_cast<uint32_t>(mma_tile_row_offset + lane_group_idx);
                            const uint32_t pool_row1 =
                                block_pool_row_base + static_cast<uint32_t>(mma_tile_row_offset + lane_group_idx + 8);

                            const bool is_row0_valid = static_cast<uint32_t>(mma_tile_row_offset + lane_group_idx) < valid_row_count;
                            const bool is_row1_valid = static_cast<uint32_t>(mma_tile_row_offset + lane_group_idx + 8) < valid_row_count;
                            if (is_row0_valid || is_row1_valid) {
                                const auto metadata_row0 = *(pool_metadata_base + (is_row0_valid ? pool_row0 : pool_row1));
                                const auto metadata_row1 = *(pool_metadata_base + (is_row1_valid ? pool_row1 : pool_row0));
                                DG_STATIC_ASSERT(kMmaNTilesPerWarp == 8, "permuted combine layout assumes 8 n8 tiles/warp");
                                DG_STATIC_ASSERT(kHiddenDim % 64 == 0, "permuted combine layout works on 64-column blocks");
                                const uint32_t physical_hidden_col_idx =
                                    n_base + n_group_idx * kGemmOutputColsPerWarp + static_cast<uint32_t>(lane_in_group) * 16u;
                                const float combine_weight_0 = (DG_TOPK_WEIGHT_SITE == 1 && is_row0_valid)
                                    ? *l1_topk_weights_buffer.get_token_layout(pool_row0).get_base_ptr<float>()
                                    : 1.0f;
                                const float combine_weight_1 = (DG_TOPK_WEIGHT_SITE == 1 && is_row1_valid)
                                    ? *l1_topk_weights_buffer.get_token_layout(pool_row1).get_base_ptr<float>()
                                    : 1.0f;
                                auto store_combine_row =
                                    [&](const layout::TokenSrcMetadata& metadata_row, float combine_weight, uint32_t row_accum_offset) {
                                        auto dst_base = combine_token_buffer.get_topk_buffer(metadata_row.topk_idx)
                                                            .get_token_layout(metadata_row.source_token_idx)
                                                            .get_base_ptr<nv_bfloat16>();
                                        #pragma unroll
                                        for (uint32_t half_idx = 0; half_idx < 2u; ++half_idx) {
                                            union {
                                                __nv_bfloat162 bf16x2[4];
                                                int4 quad;
                                            } packed;
                                            #pragma unroll
                                            for (uint32_t w = 0; w < 4u; ++w) {
                                                const uint32_t n_tile_idx = half_idx * 4u + w;
                                                packed.bf16x2[w] = __floats2bfloat162_rn(
                                                    mma_tile_accum[n_tile_idx * 4 + row_accum_offset] * combine_weight,
                                                    mma_tile_accum[n_tile_idx * 4 + row_accum_offset + 1] * combine_weight);
                                            }
                                            *sym_buffer.map_pointer_to_rank(
                                                reinterpret_cast<int4*>(dst_base + physical_hidden_col_idx + half_idx * 8u),
                                                metadata_row.source_rank_idx) = packed.quad;
                                        }
                                    };
                                if (is_row0_valid)
                                    store_combine_row(metadata_row0, combine_weight_0, 0);
                                if (is_row1_valid)
                                    store_combine_row(metadata_row1, combine_weight_1, 2);
                            }
                        }
                    }
                }
            }
            if (is_l1) {
                ptx::sync_threads_partial(kThreadsPerBlock, kL1ReadyBarrierId);
                if (thread_idx == 0)
                    ptx::red_add_rel(l1_completed_block_counts_base + local_expert_idx * kL1CompletedBlockCountLineWords, 1u);
            }
            current_block = next_block;
        }
    }

    __syncthreads();

    {
        comm::grid_sync<kSmCount, kCombineGridBarrierSlot>(
            workspace, logical_sm_idx, thread_idx, []() { ptx::sync_threads_partial(kThreadsPerBlock, kCombineBlockBarrierId); });

        {
            const uint32_t reset_global_thread_idx = logical_sm_idx * kThreadsPerBlock + thread_idx;
            constexpr uint32_t kGlobalResetThreadCount = kSmCount * kThreadsPerBlock;
            for (uint32_t i = reset_global_thread_idx; i < kExpertCount; i += kGlobalResetThreadCount)
                *workspace.get_expert_send_count_ptr(i) = 0;
            for (uint32_t i = reset_global_thread_idx; i < kRankCount * kExpertsPerRank; i += kGlobalResetThreadCount)
                *workspace.get_expert_recv_count_ptr(i / kExpertsPerRank, i % kExpertsPerRank) = 0;
            if (reset_global_thread_idx == 0)
                *workspace.get_dispatch_arrival_count_ptr() = 0;
            for (uint32_t i = reset_global_thread_idx; i < kExpertsPerRank; i += kSmCount * kThreadsPerBlock)
                l1_completed_block_counts_base[i * kL1CompletedBlockCountLineWords] = 0;

            if (reset_global_thread_idx == 0) {
                auto* const stored_launch_epoch_ptr = reinterpret_cast<const uint32_t*>(
                    reinterpret_cast<uint8_t*>(workspace.get_dispatch_slot_base_ptr()) + kStoredLaunchEpochOffsetBytes);
                ptx::st_volatile(stored_launch_epoch_ptr, (ptx::ld_volatile(stored_launch_epoch_ptr) % 0xFFFFu) + 1u);
            }
        }
        __threadfence_system();

        if constexpr (kRankCount > 1) {
            comm::icn_barrier<kRankCount, kSmCount, kThreadsPerBlock, kCombineGridBarrierSlot, kCombineIcnBarrierTag>(
                workspace,
                sym_buffer,
                logical_sm_idx,
                thread_idx,
                []() { ptx::sync_threads_partial(kThreadsPerBlock, kCombineBlockBarrierId); },
                false,
                true);
        }

        auto smem_topk_valid_masks = reinterpret_cast<uint32_t*>(smem_buffer);
        {
            for (uint32_t t = thread_idx; t < num_tokens; t += kThreadsPerBlock) {
                const auto* topk_row =
                    reinterpret_cast<const int64_t*>(input_topk_expert_ids_base + static_cast<uint64_t>(t) * kInputTopkIdxRowStrideBytes);
                uint32_t topk_valid_mask = 0;
                #pragma unroll
                for (uint32_t s = 0; s < kTopkCount; ++s) {
                    if (__ldg(topk_row + s) >= 0)
                        topk_valid_mask |= (1u << s);
                }
                smem_topk_valid_masks[t] = topk_valid_mask;
            }
            ptx::sync_threads_partial(kThreadsPerBlock, kCombineBlockBarrierId);
        }
        const uint32_t combine_global_thread_idx = logical_sm_idx * kThreadsPerBlock + thread_idx;
        constexpr uint32_t kGlobalCombineThreadCount = kSmCount * kThreadsPerBlock;
        DG_STATIC_ASSERT(kHiddenDim % 2 == 0, "combine reduce loads bf16 pairs");
        constexpr uint32_t kHiddenBf16x2Count = kHiddenDim / 2;
        DG_STATIC_ASSERT(kHiddenBf16x2Count % 32 == 0, "combine reduce assumes whole 32-pair permutation groups per token");
        constexpr uint32_t kCombineUint4VectorsPerToken = kHiddenBf16x2Count / 4;
        auto y_bf16x2 = reinterpret_cast<nv_bfloat162*>(y);
        for (uint64_t idx = combine_global_thread_idx; idx < static_cast<uint64_t>(num_tokens) * kCombineUint4VectorsPerToken;
             idx += kGlobalCombineThreadCount) {
            const uint32_t uint4_vector_idx = static_cast<uint32_t>(idx % kCombineUint4VectorsPerToken);
            const uint32_t token_idx = static_cast<uint32_t>(idx / kCombineUint4VectorsPerToken);
            const uint32_t bf16x2_col_idx = uint4_vector_idx * 4u;
            const uint32_t bf16x2_group_base = bf16x2_col_idx & ~31u;
            const uint32_t bf16x2_group_offset = bf16x2_col_idx & 31u;
            const uint32_t output_bf16x2_group_offset = ((bf16x2_group_offset & 7u) << 2) + (bf16x2_group_offset >> 3);
            const uint32_t topk_valid_mask = smem_topk_valid_masks[token_idx];
            const float* topk_weight_row = (DG_TOPK_WEIGHT_SITE == 2)
                ? reinterpret_cast<const float*>(input_topk_weight_base + static_cast<uint64_t>(token_idx) * kInputTopkWeightRowStrideBytes)
                : nullptr;
            float combine_sum_0 = 0.0f, combine_sum_1 = 0.0f, combine_sum_2 = 0.0f, combine_sum_3 = 0.0f, combine_sum_4 = 0.0f,
                  combine_sum_5 = 0.0f, combine_sum_6 = 0.0f, combine_sum_7 = 0.0f;
            #pragma unroll
            for (uint32_t topk_idx = 0; topk_idx < kTopkCount; ++topk_idx) {
                auto combine_src =
                    reinterpret_cast<const nv_bfloat162*>(
                        combine_token_buffer.get_topk_buffer(topk_idx).get_token_layout(token_idx).get_base_ptr<nv_bfloat16>()) +
                    bf16x2_col_idx;
                const uint4 combine_bf16_vector = *reinterpret_cast<const uint4*>(combine_src);
                const nv_bfloat162 bf16_pair_0 = *reinterpret_cast<const nv_bfloat162*>(&combine_bf16_vector.x);
                const nv_bfloat162 bf16_pair_1 = *reinterpret_cast<const nv_bfloat162*>(&combine_bf16_vector.y);
                const nv_bfloat162 bf16_pair_2 = *reinterpret_cast<const nv_bfloat162*>(&combine_bf16_vector.z);
                const nv_bfloat162 bf16_pair_3 = *reinterpret_cast<const nv_bfloat162*>(&combine_bf16_vector.w);
                const bool is_topk_entry_valid = ((topk_valid_mask >> topk_idx) & 1u) != 0u;
                const float topk_weight = (DG_TOPK_WEIGHT_SITE == 2) ? __ldg(topk_weight_row + topk_idx) : 1.0f;
                combine_sum_0 += is_topk_entry_valid ? __low2float(bf16_pair_0) * topk_weight : 0.0f;
                combine_sum_1 += is_topk_entry_valid ? __high2float(bf16_pair_0) * topk_weight : 0.0f;
                combine_sum_2 += is_topk_entry_valid ? __low2float(bf16_pair_1) * topk_weight : 0.0f;
                combine_sum_3 += is_topk_entry_valid ? __high2float(bf16_pair_1) * topk_weight : 0.0f;
                combine_sum_4 += is_topk_entry_valid ? __low2float(bf16_pair_2) * topk_weight : 0.0f;
                combine_sum_5 += is_topk_entry_valid ? __high2float(bf16_pair_2) * topk_weight : 0.0f;
                combine_sum_6 += is_topk_entry_valid ? __low2float(bf16_pair_3) * topk_weight : 0.0f;
                combine_sum_7 += is_topk_entry_valid ? __high2float(bf16_pair_3) * topk_weight : 0.0f;
            }
            const uint64_t output_bf16x2_group_base = static_cast<uint64_t>(token_idx) * kHiddenBf16x2Count + bf16x2_group_base;
            y_bf16x2[output_bf16x2_group_base + output_bf16x2_group_offset] = __floats2bfloat162_rn(combine_sum_0, combine_sum_1);
            y_bf16x2[output_bf16x2_group_base + output_bf16x2_group_offset + 4u] = __floats2bfloat162_rn(combine_sum_2, combine_sum_3);
            y_bf16x2[output_bf16x2_group_base + output_bf16x2_group_offset + 8u] = __floats2bfloat162_rn(combine_sum_4, combine_sum_5);
            y_bf16x2[output_bf16x2_group_base + output_bf16x2_group_offset + 12u] = __floats2bfloat162_rn(combine_sum_6, combine_sum_7);
        }
    }
}

}

#pragma clang diagnostic pop
