#pragma once

#include <torch/python.h>
#include <cstdint>
#include <limits>
#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/math.hpp"

namespace deep_gemm {

class FP32GemmRuntime final : public LaunchRuntime<FP32GemmRuntime> {
public:
    struct Params { const float* a; const float* b; float* d; int m, n, k; };
    struct Args {
        LaunchArgs launch_args;
        Params params;
        int n_per_warp, block_warps;
        bool vectorized;
        int static_m, static_n, static_k, unroll;
    };
    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/fp32_gemm.cuh>
extern "C" __global__ void fp32_gemm(deep_gemm::FP32GemmParams p) {{
    deep_gemm::fp32_gemm_nt<{}, {}, {}, {}, {}, {}, {}>(p);
}}
)", args.n_per_warp, args.block_warps, args.vectorized ? "true" : "false",
            args.static_m, args.static_n, args.static_k, args.unroll);
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

static void fp32_gemm(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d) {
    constexpr int npw = 1, warps = 4, unroll = 4;
    const int64_t m64 = a.size(0), n64 = b.size(0), k64 = a.size(1);
    constexpr int64_t int_max = std::numeric_limits<int>::max();
    TORCH_CHECK(m64 > 0 && m64 <= int_max && n64 > 0 && n64 <= int_max &&
                k64 > 0 && k64 <= int_max, "FP32 GEMM requires positive int32 dimensions");
    // Leave room for the final K-loop increment and unrolled offsets in both
    // vectorized and scalar kernels, even when K itself fits in int32.
    TORCH_CHECK(k64 <= int_max - 32 * 4 * unroll, "FP32 GEMM K exceeds the int32 loop range");
    const int64_t column_groups = ceil_div(n64, int64_t(npw));
    const int64_t warp_count = m64 * column_groups;
    TORCH_CHECK(warp_count <= int_max, "FP32 GEMM warp count exceeds the int32 index range");
    const int64_t grid_x = ceil_div(warp_count, int64_t(warps));
    // Include padded warps in the bound on the device's signed linear index.
    TORCH_CHECK(grid_x * warps - 1 <= int_max, "FP32 GEMM padded warp index exceeds int32");
    TORCH_CHECK(grid_x <= device_runtime->get_prop()->maxGridSize[0], "FP32 GEMM exceeds grid.x limit");
    const int m = static_cast<int>(m64), n = static_cast<int>(n64), k = static_cast<int>(k64);
    const bool vectorized = k % 4 == 0 &&
        (reinterpret_cast<uintptr_t>(a.data_ptr()) & 15) == 0 &&
        (reinterpret_cast<uintptr_t>(b.data_ptr()) & 15) == 0;
    // Specialize this short, narrow product to eliminate loop/tail checks.
    const bool specialize = m == 36 && n == 32 && k == 4096;
    FP32GemmRuntime::Args args{
        {dim3(static_cast<unsigned int>(grid_x)), dim3(warps * 32), 0},
        {a.data_ptr<float>(), b.data_ptr<float>(), d.data_ptr<float>(), m, n, k}, npw, warps, vectorized,
        specialize ? 36 : 0, specialize ? 32 : 0, specialize ? 4096 : 0, unroll};
    auto code = FP32GemmRuntime::generate(args);
    auto runtime = compiler->build("fp32_gemm", code);
    FP32GemmRuntime::launch(runtime, args);
}

} // namespace deep_gemm
