#pragma once

#include <torch/python.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <list>
#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"
#include "../heuristics/common_tf32.hpp"
#include <deep_gemm/common/profiling_interface.cuh>

using namespace deep_gemm_tf32_common;
namespace deep_gemm {

class Tf32HcPrenormGemmRuntime final : public LaunchRuntime<Tf32HcPrenormGemmRuntime> {
public:
    struct LaunchInfo {
        uint32_t shape_n, shape_k;
        uint32_t block_m, block_n, block_k;
        uint32_t num_splits;
        uint32_t num_threads, num_stages;
        bool fast_bf16_to_tf32;
        uint32_t kernel_variant;
        std::string kernel_name;
    };

    // NOTES: the pointers are deliberately non-const, `launch_kernel` collects `void*` addresses of
    // them and a `const T**` would not convert
    struct KernelArguments {
        float* rhs;     // `fn`, the FP32 weight
        float* out;     // `d`
        float* sqr_sum; // per-token squared sum
        void* lhs;      // `x`, the BF16 activation
        float* ws;
        float* ws_s;
        int* counter;
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
#include <deep_gemm/impls/tf32_hc_prenorm_gemm.cuh>
namespace deep_gemm {{

constexpr uint32_t SHAPE_N = {};
constexpr uint32_t SHAPE_K = {};
constexpr uint32_t BLOCK_M = {};
constexpr uint32_t BLOCK_N = {};
constexpr uint32_t BLOCK_K = {};
constexpr uint32_t NUM_SPLITS = {};
constexpr uint32_t NUM_THREADS = {};
constexpr uint32_t NUM_STAGES = {};
constexpr bool FAST_BF16_TO_TF32 = {};
constexpr uint32_t KERNEL_VARIANT = {};

extern "C"
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
__launch_bounds__(NUM_THREADS, 1)
#else
__launch_bounds__(NUM_THREADS, 2)
#endif
__global__ void {}(
  const float* __restrict__ fn,
  float* __restrict__ out,
  float* __restrict__ sqrsum,
  const __ppu_bfloat16* __restrict__ x,
  const uint32_t num_tokens,
  float* __restrict__ ws,
  float* __restrict__ ws_s,
  int* __restrict__ counter
) {{
  tf32_hc_prenorm_gemm_device<
    SHAPE_N, SHAPE_K,
    BLOCK_M, BLOCK_N, BLOCK_K,
    NUM_SPLITS,
    FAST_BF16_TO_TF32,
    NUM_THREADS, NUM_STAGES, KERNEL_VARIANT
  >(fn, out, sqrsum, x, num_tokens, ws, ws_s, counter);
}}
}}
)",
            args.launch_info.shape_n, args.launch_info.shape_k, args.launch_info.block_m,
            args.launch_info.block_n, args.launch_info.block_k, args.launch_info.num_splits,
            args.launch_info.num_threads, args.launch_info.num_stages,
            args.launch_info.fast_bf16_to_tf32, args.launch_info.kernel_variant, args.launch_info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.kernel_args.rhs, args.kernel_args.out,
                                    args.kernel_args.sqr_sum, args.kernel_args.lhs,
                                    args.kernel_args.num_tokens,
                                    args.kernel_args.ws, args.kernel_args.ws_s, args.kernel_args.counter));
    }
};

class Tf32HcRoundBRuntime final : public LaunchRuntime<Tf32HcRoundBRuntime> {
public:
    struct Args { LaunchArgs launch_args; float* src; float* dst; uint32_t count; };
    static std::string generate_impl(const Args&) {
        return R"(
#define TF32_HC_PRENORM_HGRTC
#include <deep_gemm/impls/tf32_hc_prenorm_gemm.cuh>
extern "C" __global__ void tf32_hc_round_b(const float* src, float* dst, uint32_t count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const uint32_t bits = cutlass::NumericConverter<cutlass::tfloat32_t, float>::convert(src[i]).raw();
        dst[i] = __uint_as_float(bits);
    }
}
)";
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.src, args.dst, args.count));
    }
};

// Captures allocate separate scratch from the PyTorch graph-private pool.
struct Tf32ScratchBuffers {
    torch::Tensor rounded_b;
    torch::Tensor ws;
    torch::Tensor ws_s;
    std::unordered_map<int, torch::Tensor> counters;
};
// Serialize B prepass submission; isolate scratch by device and stream.
static std::mutex tf32_round_b_submit_mutex;
static std::mutex tf32_scratch_mutex;
static constexpr size_t kTf32ScratchStreams = 16;
using Tf32ScratchCache = std::list<std::pair<uintptr_t, Tf32ScratchBuffers>>;
static std::unordered_map<int64_t, Tf32ScratchCache> tf32_scratch;

static int64_t round_up_pow2(const int64_t& x) {
    int64_t p = 1;
    while (p < x)
        p *= 2;
    return p;
}

static void tf32_hc_prenorm_gemm(const torch::Tensor& lhs, const torch::Tensor& rhs, const torch::Tensor& out,
                                 const torch::Tensor& sqr_sum, const int& m, const int& n, const int& k) {
    const auto& [block_m, block_n, block_k, num_splits_i, num_threads_i, num_stages, smem_size, kernel_variant] =
        deep_gemm_tf32_common::get_best_configs(m, n, k);
    const uint32_t num_splits = static_cast<uint32_t>(num_splits_i);

    constexpr bool fast_bf16_to_tf32 = true;

    const uint32_t num_threads = static_cast<uint32_t>(num_threads_i);
    const uint32_t grid_m = ceil_div(m, block_m);
    const dim3 grid(grid_m, num_splits, 1);
    const dim3 block(num_threads, 1, 1);

    torch::Tensor ws, ws_s, counter, rounded_b;
    {
        Tf32ScratchBuffers graph_scratch;
        auto* selected = &graph_scratch;
        std::unique_lock<std::mutex> lock(tf32_scratch_mutex, std::defer_lock);
        if (c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None) {
            lock.lock();
            auto& cache = tf32_scratch[lhs.get_device()];
            const auto stream = reinterpret_cast<uintptr_t>(current_stream());
            const auto it = std::find_if(cache.begin(), cache.end(),
                                         [&](const auto& entry) { return entry.first == stream; });
            if (it == cache.end()) {
                cache.emplace_front(stream, Tf32ScratchBuffers{});
                if (cache.size() > kTf32ScratchStreams) cache.pop_back();
            } else {
                cache.splice(cache.begin(), cache, it);
            }
            selected = &cache.front().second;
        }
        auto& scratch = *selected;
        const auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(lhs.device());
        const auto ensure_capacity = [&](torch::Tensor& buffer, int64_t count, bool zero = false) {
            if (!buffer.defined() || buffer.numel() < count) {
                const auto capacity = round_up_pow2(count);
                buffer = zero ? torch::zeros({capacity}, float_opts.dtype(torch::kInt32))
                              : torch::empty({capacity}, float_opts);
            }
        };
        if (kernel_variant & 1024u) {
            ensure_capacity(scratch.rounded_b, int64_t(n) * k);
            rounded_b = scratch.rounded_b;
        }
        ensure_capacity(scratch.ws, int64_t(num_splits) * m * n);
        ensure_capacity(scratch.ws_s, int64_t(num_splits) * m);
        auto& counter_slot = scratch.counters[static_cast<int>(num_splits)];
        ensure_capacity(counter_slot, grid_m, true);
        ws = scratch.ws;
        ws_s = scratch.ws_s;
        counter = counter_slot;
    }

    const std::string kernel_name = "tf32_hc_prenorm_gemm";
    const auto& args = Tf32HcPrenormGemmRuntime::Args{
        .launch_info = {static_cast<uint32_t>(n), static_cast<uint32_t>(k), static_cast<uint32_t>(block_m),
                        static_cast<uint32_t>(block_n), static_cast<uint32_t>(block_k),
                        num_splits, num_threads, static_cast<uint32_t>(num_stages),
                        fast_bf16_to_tf32, kernel_variant, kernel_name},
        .launch_args = {grid, block, smem_size},
        .kernel_args = {(kernel_variant & 1024u) ? rounded_b.data_ptr<float>() : rhs.data_ptr<float>(), out.data_ptr<float>(), sqr_sum.data_ptr<float>(), lhs.data_ptr(),
                        ws.data_ptr<float>(), ws_s.data_ptr<float>(), counter.data_ptr<int>(),
                        static_cast<uint32_t>(m)},
    };

    const auto& code = Tf32HcPrenormGemmRuntime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code, num_threads, smem_size);

    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_params(
            GemmType::DenseGemm, false, std::string("tf32"),
            1, m, n, k, 1, nullptr, current_stream());
        dg_prof_params.add_params("num_splits", int(num_splits));
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);

    if (kernel_variant & 1024u) {
        Tf32HcRoundBRuntime::Args pack_args{{dim3(ceil_div(n * k, 256)), dim3(256), 0}, rhs.data_ptr<float>(), rounded_b.data_ptr<float>(), uint32_t(n * k)};
        const auto& pack_code = Tf32HcRoundBRuntime::generate(pack_args);
        const auto& pack_runtime = compiler->build("tf32_hc_round_b", pack_code, 256, 0);
        std::lock_guard<std::mutex> lock(tf32_round_b_submit_mutex);
        Tf32HcRoundBRuntime::launch(pack_runtime, pack_args);
        Tf32HcPrenormGemmRuntime::launch(runtime, args);
    } else {
        Tf32HcPrenormGemmRuntime::launch(runtime, args);
    }

    ProfilingInterface::Instance().instrument(false, dg_prof_params);

    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        printf("[Tf32HcPrenormGemm:]\n");
        printf("problem:[%d, %d, %d]\n", m, n, k);
        printf("grid:[%u, %u], block:%u, num_splits:%u, num_stages:%d\n",
               grid.x, grid.y, block.x, num_splits, num_stages);
        printf("ThreadblockShape[%d, %d, %d], SMSIZE:%d\n", block_m, block_n, block_k, int(smem_size));
    }
}

} // namespace deep_gemm
