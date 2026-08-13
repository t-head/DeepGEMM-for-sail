#pragma once

#include <torch/python.h>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"
#include "../heuristics/common_tf32.hpp"

using namespace deep_gemm_tf32_common;
namespace deep_gemm {

class Tf32HcPrenormGemmRuntime final : public LaunchRuntime<Tf32HcPrenormGemmRuntime> {
public:
    struct LaunchInfo {
        uint32_t shape_n, shape_k;
        uint32_t block_m, block_n, block_k;
        uint32_t num_splits;
        bool fast_bf16_to_tf32, reduce_splits;
        std::string kernel_name;
    };

    // NOTES: the pointers are deliberately non-const, `launch_kernel` collects `void*` addresses of
    // them and a `const T**` would not convert
    struct KernelArguments {
        float* rhs;     // `fn`, the FP32 weight
        float* out;     // `d`
        float* sqr_sum; // per-token squared sum
        void* lhs;      // `x`, the BF16 activation
        uint32_t num_tokens;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        KernelArguments kernel_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(
            R"(
#define TF32_HC_PRENORM_HGRTC
#include <tf32_hc_prenorm_gemm.cuh>
namespace deep_gemm {{

constexpr uint32_t SHAPE_N = {};
constexpr uint32_t SHAPE_K = {};
constexpr uint32_t BLOCK_M = {};
constexpr uint32_t BLOCK_N = {};
constexpr uint32_t BLOCK_K = {};
constexpr uint32_t NUM_SPLITS = {};
constexpr bool FAST_BF16_TO_TF32 = {};
constexpr bool REDUCE_SPLITS = {};

extern "C"
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
__launch_bounds__(BLOCK_M * 2, 1)
#else
__launch_bounds__(BLOCK_M * 2, 2)
#endif
__global__ void {}(
  const float* __restrict__ fn,
  float* __restrict__ out,
  float* __restrict__ sqrsum,
  const __ppu_bfloat16* __restrict__ x,
  const uint32_t num_tokens
) {{
  tf32_hc_prenorm_gemm_device<
    SHAPE_N, SHAPE_K,
    BLOCK_M, BLOCK_N, BLOCK_K,
    NUM_SPLITS,
    FAST_BF16_TO_TF32,
    REDUCE_SPLITS
  >(fn, out, sqrsum, x, num_tokens);
}}
}}
)",
            args.launch_info.shape_n, args.launch_info.shape_k, args.launch_info.block_m,
            args.launch_info.block_n, args.launch_info.block_k, args.launch_info.num_splits,
            args.launch_info.fast_bf16_to_tf32, args.launch_info.reduce_splits, args.launch_info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.kernel_args.rhs, args.kernel_args.out,
                                    args.kernel_args.sqr_sum, args.kernel_args.lhs,
                                    args.kernel_args.num_tokens));
    }
};

static void tf32_hc_prenorm_gemm(const torch::Tensor& lhs, const torch::Tensor& rhs, const torch::Tensor& out,
                                 const torch::Tensor& sqr_sum, const int& m, const int& n, const int& k) {
    const auto& [block_m, block_n, block_k, num_splits, smem_size] =
        deep_gemm_tf32_common::get_best_configs(m, n, k);

    constexpr bool fast_bf16_to_tf32 = true;
    // The splits are reduced in-kernel with `atomicAdd`, so a single output buffer of shape
    // `[m, n]` / `[m]` is enough
    constexpr bool reduce_splits = true;

    const uint32_t num_threads = block_m * 2;
    const dim3 grid(ceil_div(m, block_m), num_splits, 1);
    const dim3 block(num_threads, 1, 1);

    // NOTES: `LaunchRuntime::launch` always uses the default stream, so the zero-fill required by
    // the `atomicAdd` reduction must be ordered on that very stream
    const auto& stream = (hggcStream_t)0;
    if constexpr (reduce_splits) {
        DG_HGGC_RUNTIME_CHECK(
            hggcMemsetAsync(out.data_ptr(), 0, static_cast<size_t>(m) * n * sizeof(float), stream));
        DG_HGGC_RUNTIME_CHECK(
            hggcMemsetAsync(sqr_sum.data_ptr(), 0, static_cast<size_t>(m) * sizeof(float), stream));
    }

    const std::string kernel_name = "tf32_hc_prenorm_gemm";
    const auto& args = Tf32HcPrenormGemmRuntime::Args{
        .launch_info = {static_cast<uint32_t>(n), static_cast<uint32_t>(k), static_cast<uint32_t>(block_m),
                        static_cast<uint32_t>(block_n), static_cast<uint32_t>(block_k),
                        static_cast<uint32_t>(num_splits), fast_bf16_to_tf32, reduce_splits, kernel_name},
        .launch_args = {grid, block, smem_size},
        .kernel_args = {rhs.data_ptr<float>(), out.data_ptr<float>(), sqr_sum.data_ptr<float>(), lhs.data_ptr(),
                        static_cast<uint32_t>(m)},
    };

    const auto& code = Tf32HcPrenormGemmRuntime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code, num_threads, smem_size);
    Tf32HcPrenormGemmRuntime::launch(runtime, args);

    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        printf("[Tf32HcPrenormGemm:]\n");
        printf("problem:[%d, %d, %d]\n", m, n, k);
        printf("grid:[%u, %u], block:%u, num_splits:%d\n", grid.x, grid.y, block.x, num_splits);
        printf("ThreadblockShape[%d, %d, %d], SMSIZE:%d\n", block_m, block_n, block_k, int(smem_size));
    }
}

} // namespace deep_gemm
