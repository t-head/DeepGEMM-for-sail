#pragma once
// BF16 fused MoE GEMM C++ JIT Runtime — full end-to-end implementation.
//
// Generates a named __global__ wrapper that calls bf16_gemm_fused_moe_kernel_impl<...>
// via the __device__ trampoline in fused_moe_gemm.cuh.
// Registered in gemm.hpp and routed via __init__.py USE_CPP_JIT_FOR_PYTHON=1.

#include <cstdint>
#include <string>
#include <torch/python.h>
#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"
#include <deep_gemm/common/fused_gemm_common.cuh>
#include <deep_gemm/common/profiling_interface.cuh>

namespace deep_gemm {

// 7-element config tuple for fused MoE kernels: (num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages)
using FusedConfigTuple = std::tuple<int, int, int, int, int, int, int>;

class Bf16FusedMoeRuntime final : public LaunchRuntime<Bf16FusedMoeRuntime> {
public:
    struct LaunchInfo {
        int shape_n, shape_k, num_groups;
        int block_m, block_n, block_k;
        int warp_m, warp_n;
        int block_size;
        int num_stages;
        std::string kernel_name;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        GemmArgs   kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(
            R"(
#include <deep_gemm/impls/fused_moe_gemm.cuh>

namespace deep_gemm {{

constexpr uint32_t SHAPE_N    = {};
constexpr uint32_t SHAPE_K    = {};
constexpr uint32_t NUM_GROUPS = {};
constexpr uint32_t BLOCK_M    = {};
constexpr uint32_t BLOCK_N    = {};
constexpr uint32_t BLOCK_K    = {};
constexpr uint32_t WARP_M     = {};
constexpr uint32_t WARP_N     = {};
constexpr uint32_t BLOCK_SIZE = {};
constexpr int      STAGES     = {};

using SrcT = cutlass::bfloat16_t;
static constexpr GemmType kGemmType = GemmType::GroupedFused;

extern "C"
__launch_bounds__(BLOCK_SIZE, 1)
__global__ void {}(const GemmArgs args) {{
    bf16_gemm_fused_moe_kernel_impl<
        SrcT, kGemmType,
        SHAPE_N, SHAPE_K, NUM_GROUPS,
        BLOCK_M, BLOCK_N, BLOCK_K,
        WARP_M, WARP_N, BLOCK_SIZE, STAGES
    >(args);
}}

}}  // namespace deep_gemm
)",
            args.launch_info.shape_n,
            args.launch_info.shape_k,
            args.launch_info.num_groups,
            args.launch_info.block_m,
            args.launch_info.block_n,
            args.launch_info.block_k,
            args.launch_info.warp_m,
            args.launch_info.warp_n,
            args.launch_info.block_size,
            args.launch_info.num_stages,
            args.launch_info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

// --------------------------------------------------------------------------
// Top-level bf16 fused MoE GEMM implementation (C++ JIT path)
// --------------------------------------------------------------------------
static void m_grouped_gemm_bf16_bf16_bf16_nt_fused_impl(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const torch::Tensor& out,
    const torch::Tensor& m_rows,
    const torch::Tensor& expert_ids_and_cumsum,
    const torch::Tensor& sorted_token_ids,
    const torch::Tensor& aligned_num_m_blocks,
    const FusedConfigTuple& configs)
{
    const auto& [num_token, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_sum, n_] = get_shape<2>(out);

    if (m_sum == 0) return;

    // Unpack config
    auto [num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages] = configs;

    // Compute launch parameters
    int block_size = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(num_sms);

    // Smem: gemm_smem_total_size(elem 2, stages, BM, BN, BK) — sizeof(bf16) == 2.
    int smem_size = static_cast<int>(
        gemm_smem_total_size(2, num_stages, block_m, block_n, block_k));

    // Static kernel name (sibling-runtime convention) — the JIT cache key already
    // contains the full generated-code digest, so per-config cache entries are
    // automatic.
    const std::string kernel_name = "bf16_fused_moe_gemm";

    // Build args
    auto args = Bf16FusedMoeRuntime::Args{
        .launch_info = {(int)n, (int)k, (int)num_groups,
                        block_m, block_n, block_k,
                        warp_m, warp_n, block_size, num_stages,
                        kernel_name},
        .launch_args = {grid, dim3(block_size), smem_size},
        .kernel_params = {
            .a_ptr = lhs.data_ptr(),
            .b_ptr = rhs.data_ptr(),
            .c_ptr = out.data_ptr(),
            .expert_ids_and_cumsum = expert_ids_and_cumsum.data_ptr<int32_t>(),
            .sorted_token_ids = sorted_token_ids.data_ptr<int32_t>(),
            .aligned_num_m_blocks = aligned_num_m_blocks.data_ptr<int32_t>(),
            .shape_m = static_cast<uint32_t>(num_token)
        }
    };

    // Generate code and compile
    const auto& code = Bf16FusedMoeRuntime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code, block_size, smem_size);
    const auto& kernel = runtime->kernel;

    // Query occupancy and set grid
    int blocks_per_cu = 0;
    DG_HGGC_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_cu, kernel, block_size, smem_size));
    args.launch_args.grid_dim.x *= blocks_per_cu;

    // Profiling instrumentation
    hggcStream_t stream = (hggcStream_t)0;
    int topk = m_sum / num_token;
    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_fused_moe_params(
            std::string("bf16"), std::string("non_quantized"),
            (int)num_groups, (int)num_token, topk, (int)n, (int)k, m_rows.data_ptr<int32_t>(), stream);
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);

    // Launch
    Bf16FusedMoeRuntime::launch(runtime, args);

    ProfilingInterface::Instance().instrument(false, dg_prof_params);

    // Optional debug logging
    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        printf("[C++ JIT FusedMoeGemm-BF16:]\n");
        printf("group:%d, problem:[%d, %d, %ld]\n", (int)num_groups, (int)num_token, (int)n, k);
        printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], stages:%d\n",
               block_m, block_n, block_k, warp_m, warp_n, block_k, num_stages);
        printf("grid:%d, block:%d, smem:%d, tb_per_cu:%d\n",
               num_sms * blocks_per_cu, block_size, smem_size, blocks_per_cu);
    }
}

// ==========================================================================
// FP8/INT8 fused MoE GEMM C++ JIT runtimes (blockwise + per-channel quant)
// ==========================================================================

// FP8 blockwise-quant fused MoE GEMM runtime.
// The kernel name embeds "fp8_deep_gemm" so the C++ JIT compiler selects the
// warp-interleaving LLVM flags (compiler matches the 'gemm_fp8' name pattern).
class Fp8BlkwiseFusedMoeRuntime final : public LaunchRuntime<Fp8BlkwiseFusedMoeRuntime> {
public:
    struct LaunchInfo {
        int shape_n, shape_k, num_groups;
        int block_m, block_n, block_k;
        int warp_m, warp_n;
        int block_size;
        int stages;      // truncated: SHAPE_K < BLOCK_K * kNumStages ? SHAPE_K / BLOCK_K : kNumStages
        int n_expand;    // final N_EXPAND (after the kUseNStageKernel check)
        std::string kernel_name;
    };

    struct Args {
        LaunchInfo          launch_info;
        LaunchArgs          launch_args;
        QuantGemmArgs       kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(
            R"(
#include <deep_gemm/impls/fused_moe_gemm_with_blkwise_quant.cuh>

namespace deep_gemm {{

constexpr uint32_t SHAPE_N    = {};
constexpr uint32_t SHAPE_K    = {};
constexpr uint32_t NUM_GROUPS = {};
constexpr uint32_t BLOCK_M    = {};
constexpr uint32_t BLOCK_N    = {};
constexpr uint32_t BLOCK_K    = {};
constexpr uint32_t WARP_M     = {};
constexpr uint32_t WARP_N     = {};
constexpr uint32_t BLOCK_SIZE = {};
constexpr int      STAGES     = {};
constexpr int      N_EXPAND   = {};

using SrcT = __hg_fp8_e4m3;
static constexpr GemmType kGemmType = GemmType::GroupedFused;

extern "C"
__launch_bounds__(BLOCK_SIZE, 1)
__global__ void {}(const QuantGemmArgs args) {{
    fp8_blockwise_quant_gemm_fused_moe_kernel_impl<
        SrcT, kGemmType,
        SHAPE_N, SHAPE_K, NUM_GROUPS,
        BLOCK_M, BLOCK_N, BLOCK_K,
        WARP_M, WARP_N, BLOCK_SIZE, STAGES, N_EXPAND
    >(args);
}}

}}  // namespace deep_gemm
)",
            args.launch_info.shape_n,
            args.launch_info.shape_k,
            args.launch_info.num_groups,
            args.launch_info.block_m,
            args.launch_info.block_n,
            args.launch_info.block_k,
            args.launch_info.warp_m,
            args.launch_info.warp_n,
            args.launch_info.block_size,
            args.launch_info.stages,
            args.launch_info.n_expand,
            args.launch_info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

// A8W8 per-channel-quant fused MoE GEMM runtime (int8 or fp8 sources).
// The kernel name deliberately avoids "fp8_deep_gemm"/"fp8_grouped_deep_gemm"/
// "mqa_logits" substrings so it never matches the 'gemm_fp8' warp-interleaving pattern.
class A8W8PerchannelFusedMoeRuntime final : public LaunchRuntime<A8W8PerchannelFusedMoeRuntime> {
public:
    struct LaunchInfo {
        int shape_n, shape_k, num_groups;
        int block_m, block_n, block_k;
        int warp_m, warp_n;
        int block_size;
        int num_stages;
        std::string src_t;      // "int8_t" or "__hg_fp8_e4m3"
        std::string kernel_name;
    };

    struct Args {
        LaunchInfo          launch_info;
        LaunchArgs          launch_args;
        QuantGemmArgs       kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(
            R"(
#include <deep_gemm/impls/fused_moe_gemm_with_perchannel_quant.cuh>

namespace deep_gemm {{

constexpr uint32_t SHAPE_N    = {};
constexpr uint32_t SHAPE_K    = {};
constexpr uint32_t NUM_GROUPS = {};
constexpr uint32_t BLOCK_M    = {};
constexpr uint32_t BLOCK_N    = {};
constexpr uint32_t BLOCK_K    = {};
constexpr uint32_t WARP_M     = {};
constexpr uint32_t WARP_N     = {};
constexpr uint32_t BLOCK_SIZE = {};
constexpr int      STAGES     = {};

using SrcT = {};
static constexpr GemmType kGemmType = GemmType::GroupedFused;

extern "C"
__launch_bounds__(BLOCK_SIZE, 1)
__global__ void {}(const QuantGemmArgs args) {{
    a8w8_perchannel_quant_gemm_fused_moe_kernel_impl<
        SrcT, kGemmType,
        SHAPE_N, SHAPE_K, NUM_GROUPS,
        BLOCK_M, BLOCK_N, BLOCK_K,
        WARP_M, WARP_N, BLOCK_SIZE, STAGES
    >(args);
}}

}}  // namespace deep_gemm
)",
            args.launch_info.shape_n,
            args.launch_info.shape_k,
            args.launch_info.num_groups,
            args.launch_info.block_m,
            args.launch_info.block_n,
            args.launch_info.block_k,
            args.launch_info.warp_m,
            args.launch_info.warp_n,
            args.launch_info.block_size,
            args.launch_info.num_stages,
            args.launch_info.src_t,
            args.launch_info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

// --------------------------------------------------------------------------
// Shared per-channel fused MoE GEMM
// --------------------------------------------------------------------------
static void m_grouped_gemm_perchannel_nt_fused_impl(
    const torch::Tensor& lhs,
    const torch::Tensor& lhs_scales,
    const torch::Tensor& rhs,
    const torch::Tensor& rhs_scales,
    const torch::Tensor& out,
    const torch::Tensor& m_rows,
    const torch::Tensor& expert_ids_and_cumsum,
    const torch::Tensor& sorted_token_ids,
    const torch::Tensor& aligned_num_m_blocks,
    const FusedConfigTuple& configs)
{
    const auto& [num_token, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_sum, n_] = get_shape<2>(out);

    // Do nothing if `m_sum` is zero
    if (m_sum == 0) return;

    // Unpack config
    auto [num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages] = configs;

    // Compute launch parameters
    int block_size = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(num_sms);

    // Smem: gemm_smem_total_size(elem 1, stages, BM, BN, BK) — sizeof(SrcT) == 1;
    // the per-channel scales reuse the last-stage sA/sB smem.
    int smem_size = static_cast<int>(
        gemm_smem_total_size(1, num_stages, block_m, block_n, block_k));

    // Source type: fp8 or int8
    const bool is_fp8 = (lhs.scalar_type() == torch::kFloat8_e4m3fn);
    const char* src_t = is_fp8 ? "__hg_fp8_e4m3" : "int8_t";
    const char* type_tag = is_fp8 ? "fp8" : "int8";

    // One shared static kernel name for both fp8 and int8 sources: SrcT is baked
    // into the generated code, so the code digest in the JIT cache key already
    // distinguishes them. Deliberately dtype-neutral — must not contain "fp8",
    // which the compiler matches by name substring for warp-interleaving flags.
    const std::string kernel_name = "a8w8_fused_moe_gemm";

    // Build args (designated initializers cannot name base-class members, so
    // QuantGemmArgs is filled member-by-member)
    QuantGemmArgs kernel_params{};
    kernel_params.a_ptr = lhs.data_ptr();
    kernel_params.b_ptr = rhs.data_ptr();
    kernel_params.c_ptr = out.data_ptr();
    kernel_params.expert_ids_and_cumsum = expert_ids_and_cumsum.data_ptr<int32_t>();
    kernel_params.sorted_token_ids = sorted_token_ids.data_ptr<int32_t>();
    kernel_params.aligned_num_m_blocks = aligned_num_m_blocks.data_ptr<int32_t>();
    kernel_params.shape_m = static_cast<uint32_t>(num_token);
    kernel_params.scale_a_ptr = lhs_scales.data_ptr();
    kernel_params.scale_b_ptr = rhs_scales.data_ptr();
    auto args = A8W8PerchannelFusedMoeRuntime::Args{
        .launch_info = {(int)n, (int)k, (int)num_groups,
                        block_m, block_n, block_k,
                        warp_m, warp_n, block_size, num_stages,
                        src_t, kernel_name},
        .launch_args = {grid, dim3(block_size), smem_size},
        .kernel_params = kernel_params
    };

    // Generate code and compile
    const auto& code = A8W8PerchannelFusedMoeRuntime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code, block_size, smem_size);
    const auto& kernel = runtime->kernel;

    // Query occupancy and set grid
    int blocks_per_cu = 0;
    DG_HGGC_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_cu, kernel, block_size, smem_size));
    args.launch_args.grid_dim.x *= blocks_per_cu;

    // Profiling instrumentation
    hggcStream_t stream = (hggcStream_t)0;
    int topk = m_sum / num_token;
    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_fused_moe_params(
            std::string(type_tag), std::string("channel"),
            (int)num_groups, (int)num_token, topk, (int)n, (int)k, m_rows.data_ptr<int32_t>(), stream);
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);

    // Launch
    A8W8PerchannelFusedMoeRuntime::launch(runtime, args);

    ProfilingInterface::Instance().instrument(false, dg_prof_params);

    // Optional debug logging
    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        printf("[C++ JIT FusedMoeGemm-Perchannel-%s:]\n", type_tag);
        printf("group:%d, problem:[%d, %d, %d]\n", (int)num_groups, (int)num_token, (int)n, (int)k);
        printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], kNumStages:%d\n",
               block_m, block_n, block_k, warp_m, warp_n, block_k, num_stages);
        printf("grid:%d, block:%d, smem:%d, tb_per_cu:%d\n",
               num_sms * blocks_per_cu, block_size, smem_size, blocks_per_cu);
    }
}

// --------------------------------------------------------------------------
// FP8 blockwise fused MoE GEMM — pure blockwise path. The per-channel routing,
// the m_sum == 0 exit, the topk computation and the col-major scales transform
// live in the API layer.
// --------------------------------------------------------------------------
static void m_grouped_gemm_blkwise_nt_fused_impl(
    const torch::Tensor& lhs,
    const torch::Tensor& lhs_scales_col_major,
    const torch::Tensor& rhs,
    const torch::Tensor& rhs_scales,
    const torch::Tensor& out,
    const torch::Tensor& m_rows,
    const torch::Tensor& expert_ids_and_cumsum,
    const torch::Tensor& sorted_token_ids,
    const torch::Tensor& aligned_num_m_blocks,
    int topk,
    const FusedConfigTuple& configs)
{
    const auto& [num_token, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_sum, n_] = get_shape<2>(out);

    // Unpack config
    auto [num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages] = configs;

    // Compute launch parameters
    int block_size = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(num_sms);

    // Stage truncation: Stages = SHAPE_K < BLOCK_K * kNumStages ? SHAPE_K / BLOCK_K : kNumStages
    int stages = (k < block_k * num_stages) ? (k / block_k) : num_stages;

    // N_EXPAND dispatch (wave computation)
    int n_expand_sel = 1;
    if (k <= 512 && block_k == 128 && k % block_k == 0) {
        int expected_m = ceil_div(num_token * topk, num_groups);
        int wave = ceil_div(ceil_div(expected_m, block_m) * ceil_div(n, block_n), num_sms);
        switch (wave) {
            case 1:  n_expand_sel = 1; break;
            case 2:  n_expand_sel = 2; break;
            default: n_expand_sel = 4; break;
        }
    }
    // kUseNStageKernel = n_expand_ > 1 && (SHAPE_N % (BLOCK_N * n_expand_) == 0)
    const bool use_n_stage_kernel = (n_expand_sel > 1) && (n % (block_n * n_expand_sel) == 0);
    const int n_expand = use_n_stage_kernel ? n_expand_sel : 1;

    // Smem: blkwise_smem_total_size(elem 1, stages, BM, BN, BK) — sizeof(fp8) == 1.
    int smem_size = static_cast<int>(
        blkwise_smem_total_size(1, stages, block_m, block_n, block_k));

    // Static kernel name — the truncated stages and the final N_EXPAND are already
    // baked into the generated code (hence into the JIT cache-key digest). Still
    // contains the "fp8_deep_gemm" substring so the C++ JIT compiler selects the
    // warp-interleaving flags (compiler matches the 'gemm_fp8' name pattern).
    const std::string kernel_name = "fp8_deep_gemm_fused_moe_blkwise";

    // Build args (designated initializers cannot name base-class members, so
    // QuantGemmArgs is filled member-by-member)
    QuantGemmArgs kernel_params{};
    kernel_params.a_ptr = lhs.data_ptr();
    kernel_params.b_ptr = rhs.data_ptr();
    kernel_params.c_ptr = out.data_ptr();
    kernel_params.expert_ids_and_cumsum = expert_ids_and_cumsum.data_ptr<int32_t>();
    kernel_params.sorted_token_ids = sorted_token_ids.data_ptr<int32_t>();
    kernel_params.aligned_num_m_blocks = aligned_num_m_blocks.data_ptr<int32_t>();
    kernel_params.shape_m = static_cast<uint32_t>(num_token);
    kernel_params.scale_a_ptr = lhs_scales_col_major.data_ptr();
    kernel_params.scale_b_ptr = rhs_scales.data_ptr();
    auto args = Fp8BlkwiseFusedMoeRuntime::Args{
        .launch_info = {(int)n, (int)k, (int)num_groups,
                        block_m, block_n, block_k,
                        warp_m, warp_n, block_size, stages, n_expand,
                        kernel_name},
        .launch_args = {grid, dim3(block_size), smem_size},
        .kernel_params = kernel_params
    };

    // Generate code and compile
    const auto& code = Fp8BlkwiseFusedMoeRuntime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code, block_size, smem_size);
    const auto& kernel = runtime->kernel;

    // Query occupancy and set grid
    int blocks_per_cu = 0;
    DG_HGGC_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_cu, kernel, block_size, smem_size));
    args.launch_args.grid_dim.x *= blocks_per_cu;

    // Profiling instrumentation
    hggcStream_t stream = (hggcStream_t)0;
    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_fused_moe_params(
            std::string("fp8"), std::string("block"),
            (int)num_groups, (int)num_token, topk, (int)n, (int)k, m_rows.data_ptr<int32_t>(), stream);
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);

    // Launch
    Fp8BlkwiseFusedMoeRuntime::launch(runtime, args);

    ProfilingInterface::Instance().instrument(false, dg_prof_params);

    // Optional debug logging
    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        printf("[C++ JIT FusedMoeGemmWithBlkwiseQuant-FP8:]\n");
        printf("group:%d, problem:[%d, %d, %d], topk:%d\n", (int)num_groups, (int)num_token, (int)n, (int)k, topk);
        printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], Stages:%d, N_EXPAND:%d\n",
               block_m, block_n, block_k, warp_m, warp_n, block_k, stages, n_expand);
        printf("grid:%d, block:%d, smem:%d, tb_per_cu:%d\n",
               num_sms * blocks_per_cu, block_size, smem_size, blocks_per_cu);
    }
}

} // namespace deep_gemm
