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
        bool tiled = false;
        int block_m = 32;
        int block_n = 64, splits = 1;
        bool reduce = false;
        bool pack = false, scalar_quad = false;
    };
    static std::string generate_impl(const Args& args) {
        if (args.pack)
            return R"(
#include <deep_gemm/impls/fp32_gemm.cuh>
extern "C" __global__ void fp32_gemm(deep_gemm::FP32GemmParams p) {
    deep_gemm::fp32_gemm_pack_inputs(p);
}
)";
        if (args.reduce)
            return fmt::format(R"(
#include <deep_gemm/impls/fp32_gemm.cuh>
extern "C" __global__ void fp32_gemm(deep_gemm::FP32GemmParams p) {{
    deep_gemm::fp32_gemm_split_reduce<{}>(p);
}}
)", args.splits);
        if (args.tiled)
            return fmt::format(R"(
#include <deep_gemm/impls/fp32_gemm.cuh>
extern "C" __global__ void fp32_gemm(deep_gemm::FP32GemmParams p) {{
    {}
    deep_gemm::fp32_gemm_tiled_nt<{}, {}, {}, {}>(p);
}}
)", args.splits > 1 && args.static_k ? fmt::format("p.k = {};", args.static_k) : "",
                args.vectorized ? "true" : "false", args.block_m, args.block_n, args.splits);
        return fmt::format(R"(
#include <deep_gemm/impls/fp32_gemm.cuh>
extern "C" __global__ void fp32_gemm(deep_gemm::FP32GemmParams p) {{
    deep_gemm::fp32_gemm_nt<{}, {}, {}, {}, {}, {}, {}, {}>(p);
}}
)", args.n_per_warp, args.block_warps, args.vectorized ? "true" : "false",
            args.static_m, args.static_n, args.static_k, args.unroll, args.scalar_quad ? "true" : "false");
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
    // Bound the number of JIT variants while removing division and tail tests
    // for whole families of narrow outputs and complete unrolled K chunks.
    const bool is_810e = std::string(device_runtime->get_prop()->name).find("ZW810E") != std::string::npos;
    // Packing is worthwhile only when many narrow-output dot products reuse
    // the same long, unaligned rows. Both the copy and product are launches in
    // this invocation; compile-only/warm-up skips execution of both runtimes.
    if (is_810e && !vectorized && m >= 8 && m < 128 && n >= 16 && n <= 128 &&
        int64_t(m)*n >= 512 && k >= 1024 && k <= 8192) {
        const int padded_k=(k+511)/512*512;
        auto packed=torch::empty({int64_t(m)+n,int64_t(padded_k)},a.options());
        FP32GemmRuntime::Args x{{dim3((int64_t(m+n)*padded_k+255)/256),dim3(256),0},
            {a.data_ptr<float>(),b.data_ptr<float>(),packed.data_ptr<float>(),m,n,k},
            npw,warps,vectorized,0,0,0,unroll};
        x.pack=true;
        auto code=FP32GemmRuntime::generate(x);auto runtime=compiler->build("fp32_gemm",code);
        FP32GemmRuntime::launch(runtime,x);
        fp32_gemm(packed.narrow(0,0,m),packed.narrow(0,m,n),d);
        return;
    }
    // Preserve the previous specialization on other devices; only 810E has
    // been measured for the broader family and the shared-memory kernels.
    const bool legacy_specialize = !is_810e && m == 36 && n == 32 && k == 4096;
    const int static_m = legacy_specialize ? 36 : 0;
    const int static_n = (is_810e && n <= 128 && (n & (n - 1)) == 0) || legacy_specialize ? n : 0;
    const bool narrow_tiled = is_810e && m >= 128 && n >= 16 && n <= 64 && k >= 1024;
    const bool wide_tiled = is_810e && m >= 16 && n >= 128 && k >= 1024 && m64 * n64 >= 8192;
    const int static_k = is_810e ?
        (k <= 8192 && (narrow_tiled || k % (32 * 4 * unroll) == 0) ? k : 0) :
        (legacy_specialize ? 4096 : 0);
    const bool prefer_tiled = wide_tiled || (is_810e && m >= 128 && k >= 128 && (n >= 128 || narrow_tiled));
    // Long reductions can recover parallelism with split-K while larger M
    // tiles reuse operands. Other products still require four blocks per CU.
    const int64_t large_tiles = ceil_div(m64, int64_t(64)) * ceil_div(n64, int64_t(64));
    const int block_m = prefer_tiled && !narrow_tiled && k >= 512 &&
        (large_tiles >= 256 || (wide_tiled && m >= 128)) ? 64 : 32;
    // Large inputs supported by the linear warp grid need not fit a tiled
    // launch's smaller grid.y limit. Retain that launcher when this tile does
    // not fit, instead of rejecting an otherwise valid API call.
    const bool tiled = prefer_tiled && ceil_div(m64, int64_t(block_m)) <= device_runtime->get_prop()->maxGridSize[1];
    const int block_n = narrow_tiled ? 32 : 64;
    int splits = 1;
    if (tiled && (narrow_tiled || wide_tiled)) {
        // Four/six CTAs per general CU cover narrow/wide output grids. Retain
        // four BK16 tiles per partition; at most 32 partials bound the reduction
        // overhead and workspace without an exact-shape dispatch table.
        const int64_t tiles = ceil_div(m64, int64_t(block_m)) * ceil_div(n64, int64_t(block_n));
        const int target = narrow_tiled ? 256 : 384;
        while (splits < 32 && tiles * splits < target && k / 16 / (splits * 2) >= 4)
            splits *= 2;
    }
    auto workspace = splits == 1 ? d : torch::empty({int64_t(splits), m64, n64}, d.options());
    FP32GemmRuntime::Args args{
        {tiled ? dim3(ceil_div(n, block_n), ceil_div(m, block_m), splits) : dim3(static_cast<unsigned int>(grid_x)),
         dim3(warps * 32), 0},
        {a.data_ptr<float>(), b.data_ptr<float>(), workspace.data_ptr<float>(), m, n, k}, npw, warps, vectorized,
        static_m, static_n, static_k, unroll, tiled, block_m, block_n, splits};
    args.scalar_quad = is_810e;
    auto code = FP32GemmRuntime::generate(args);
    auto runtime = compiler->build("fp32_gemm", code);
    FP32GemmRuntime::launch(runtime, args);
    if (splits > 1) {
        constexpr int reduce_threads = 64;
        args.launch_args.grid_dim = dim3(static_cast<unsigned int>(ceil_div(m64 * n64, int64_t(reduce_threads))));
        args.launch_args.block_dim = dim3(reduce_threads);
        args.params = {workspace.data_ptr<float>(), nullptr, d.data_ptr<float>(), m, n, k};
        args.reduce = true;
        auto reduce_code = FP32GemmRuntime::generate(args);
        auto reduce_runtime = compiler->build("fp32_gemm", reduce_code);
        FP32GemmRuntime::launch(reduce_runtime, args);
    }
}

} // namespace deep_gemm
