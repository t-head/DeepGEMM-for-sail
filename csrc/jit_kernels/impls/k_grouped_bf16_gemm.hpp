#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <torch/python.h>
#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/utils.hpp"
#include "../heuristics/common_bf16.hpp"
#include <deep_gemm/common/fused_gemm_common.cuh>
#include <deep_gemm/common/profiling_interface.cuh>
#include <deep_gemm/common/types.cuh>

namespace deep_gemm {

class Bf16KGroupedRuntime final : public LaunchRuntime<Bf16KGroupedRuntime> {
public:
    struct LaunchInfo {
        int shape_m, shape_n, num_groups;
        int block_m, block_n, block_k;
        int warp_m, warp_n;
        int num_stages;
        std::string kernel_name;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        KGroupedGemmArgs kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(
            R"(
#include <deep_gemm/impls/k_grouped_bf16_gemm.cuh>

namespace deep_gemm {{

constexpr uint32_t SHAPE_M = {};
constexpr uint32_t SHAPE_N = {};
constexpr uint32_t NUM_GROUPS = {};
constexpr uint32_t BLOCK_M = {};
constexpr uint32_t BLOCK_N = {};
constexpr uint32_t BLOCK_K = {};
constexpr uint32_t WARP_M = {};
constexpr uint32_t WARP_N = {};
constexpr uint32_t NUM_STAGES = {};
constexpr uint32_t BLOCK_SIZE = 32 * (BLOCK_M / WARP_M) * (BLOCK_N / WARP_N);

extern "C"
__launch_bounds__(BLOCK_SIZE, 1)
__global__ void {}(const KGroupedGemmArgs args) {{
    bf16_k_grouped_gemm_impl<
        cutlass::bfloat16_t, float, GemmType::KGroupedContiguous,
        SHAPE_M, SHAPE_N, NUM_GROUPS,
        BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BLOCK_SIZE, NUM_STAGES, true
    >(args);
}}

}}  // namespace deep_gemm
)",
            args.launch_info.shape_m,
            args.launch_info.shape_n,
            args.launch_info.num_groups,
            args.launch_info.block_m,
            args.launch_info.block_n,
            args.launch_info.block_k,
            args.launch_info.warp_m,
            args.launch_info.warp_n,
            args.launch_info.num_stages,
            args.launch_info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

static void k_grouped_bf16_gemm_tn_contiguous_impl(
    const torch::Tensor& a, const torch::Tensor& b,
    const std::optional<torch::Tensor>& c, const torch::Tensor& d,
    const int& m, const int& n, const int& sum_k,
    const std::vector<int>& ks_cpu, const torch::Tensor& grouped_layout) {
    const auto num_groups = static_cast<int>(ks_cpu.size());
    // Follow upstream: choose the config using the largest group's K.
    const auto max_k = *std::max_element(ks_cpu.begin(), ks_cpu.end());
    const auto [num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, _smem_config] =
        deep_gemm_bf16_common::get_best_configs(m, n, max_k, num_groups, get_num_sms(),
                                               GemmType::KGroupedContiguous);
    const int block_size = 32 * (block_m / warp_m) * (block_n / warp_n);
    const int smem_size = static_cast<int>(gemm_smem_total_size(
        2, num_stages, block_m, block_n, block_k));
    const std::string kernel_name = "bf16_k_grouped_gemm";

    auto args = Bf16KGroupedRuntime::Args{
        .launch_info = {m, n, num_groups, block_m, block_n, block_k,
                        warp_m, warp_n, num_stages, kernel_name},
        .launch_args = {get_grid_shape(num_sms), dim3(block_size), smem_size},
        .kernel_params = {
            .a_ptr = a.data_ptr(),
            .b_ptr = b.data_ptr(),
            .c_ptr = d.data_ptr(),
            .grouped_layout = grouped_layout.data_ptr<int32_t>(),
        }
    };

    const auto& code = Bf16KGroupedRuntime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code);
    const auto& kernel = runtime->kernel;

    int blocks_per_cu = 0;
#if HGGCRT_VERSION >= 12080 and defined(DG_JIT_USE_RUNTIME_API)
    DG_HGGC_RUNTIME_CHECK(hggcOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_cu, kernel, block_size, smem_size));
#else
    DG_HGGC_DRIVER_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_cu, kernel, block_size, smem_size));
#endif
    DG_HOST_ASSERT(blocks_per_cu > 0 and "K-grouped tile exceeds the device's launch resources");
    args.launch_args.grid_dim.x *= blocks_per_cu;

    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_k_grouped_params(m, n, sum_k, ks_cpu,
                                           c->data_ptr() != d.data_ptr(), current_stream());
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);

    Bf16KGroupedRuntime::launch(runtime, args);

    ProfilingInterface::Instance().instrument(false, dg_prof_params);

    const char* p_env_params = std::getenv("show_log");
    if (p_env_params && std::isdigit(static_cast<unsigned char>(*p_env_params))) {
        int num_regs = 0, local_size = 0;
#if HGGCRT_VERSION >= 12080 and defined(DG_JIT_USE_RUNTIME_API)
        hggcFuncAttributes attributes;
        DG_HGGC_RUNTIME_CHECK(hggcFuncGetAttributes(&attributes, kernel));
        num_regs = attributes.numRegs;
        local_size = static_cast<int>(attributes.localSizeBytes);
#else
        DG_HGGC_DRIVER_CHECK(lazy_hgFuncGetAttribute(&num_regs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel));
        DG_HGGC_DRIVER_CHECK(lazy_hgFuncGetAttribute(&local_size, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel));
#endif

        std::printf("[KGroupedContiguous_BF16:]\n");
        std::printf("groups:%d, m:%d, n:%d, sum_k:%d, max_k:%d\n", num_groups, m, n, sum_k, max_k);
        std::printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%u, threads:%d\n",
                    num_sms, blocks_per_cu, args.launch_args.grid_dim.x, block_size);
        std::printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], num_stages:%d\n",
                    block_m, block_n, block_k, warp_m, warp_n, block_k, num_stages);
        std::printf("dynamic_smem_bytes:%d, vreg:%d, local_size_bytes:%d\n", smem_size, num_regs, local_size);
    }
}

}  // namespace deep_gemm
