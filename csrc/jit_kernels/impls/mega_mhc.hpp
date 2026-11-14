#pragma once

#include <torch/python.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/utils.hpp"
#include <deep_gemm/common/profiling_interface.cuh>
#define DG_MHC_ARGS_ONLY
#include <deep_gemm/impls/ppu_mega_mhc.cuh>
#undef DG_MHC_ARGS_ONLY

namespace deep_gemm {

enum class MegaMhcPhase { Main, Pre, Finish };

class MegaMhcRuntime final : public LaunchRuntime<MegaMhcRuntime> {
public:
    struct Config {
        int hidden, splits, iterations, tile_m, task_splits;
        bool full_rows, single, local, tail_rows, tail_tokens, vector_coefficients;
        bool store_bf16, store_fp8;
        int shared_sf_block_m;
        bool per_token_fp8;
        bool tc01 = false;
        MegaMhcPhase phase = MegaMhcPhase::Main;
        bool tc01_round_b = false;
        int tc01_pre_chunk = 1024;
    };
    struct Args {
        Config config;
        LaunchArgs launch_args;
        mhc::Args kernel_args;
        bool cooperative = false;
    };

    static std::string generate_impl(const Args& args) {
        const auto& c = args.config;
        if (c.phase == MegaMhcPhase::Pre) return fmt::format(R"(
#include <deep_gemm/impls/ppu_mega_mhc_tc01.cuh>
extern "C"
__global__ __launch_bounds__(128) void ppu_mega_mhc_tc01_pre(deep_gemm::mhc::Args args) {{
    deep_gemm::mhc::ppu_mega_mhc_tc01_prenorm_device<{},{},{},{}>(args);
}}
)", c.hidden, c.splits, c.tc01_round_b, c.tc01_pre_chunk);
        if (c.phase == MegaMhcPhase::Finish) return fmt::format(R"(
#include <deep_gemm/impls/ppu_mega_mhc_tc01.cuh>
extern "C"
__global__ __launch_bounds__(128) void ppu_mega_mhc_tc01_finish(deep_gemm::mhc::Args args) {{
    deep_gemm::mhc::ppu_mega_mhc_tc01_coeff_device<{},{},{}>(args);
}}
)", c.hidden, c.splits, c.iterations);
        if (c.tc01) return fmt::format(R"(
#include <deep_gemm/impls/ppu_mega_mhc_tc01.cuh>
extern "C"
__global__ __launch_bounds__(128) void ppu_mega_mhc_tc01(deep_gemm::mhc::Args args) {{
    deep_gemm::mhc::ppu_mega_mhc_tc01_device<{},{},{}>(args);
}}
)", c.hidden, c.splits, c.tc01_round_b);
        std::string specialization;
        if (c.single) {
            specialization = fmt::format(
                "ppu_mega_mhc_shifted_device<{},{},{},false,true,{},{},{},{},{},{},{},{},{},{}>",
                c.hidden, c.splits, c.iterations, c.local, c.tile_m, c.task_splits,
                c.tail_rows, c.tail_tokens, c.vector_coefficients, c.store_bf16, c.store_fp8, c.shared_sf_block_m, c.per_token_fp8);
        } else {
            specialization = fmt::format("ppu_mega_mhc_shifted_device<{},{},{},{},false,false,32,2,false,false,true,{},{},{},{}>",
                                         c.hidden, c.splits, c.iterations, c.full_rows,
                                         c.store_bf16, c.store_fp8, c.shared_sf_block_m, c.per_token_fp8);
        }
        return fmt::format(R"(
#include <deep_gemm/impls/ppu_mega_mhc_tc01.cuh>
extern "C"
__global__ __launch_bounds__({}) void {}(deep_gemm::mhc::Args args) {{
    deep_gemm::mhc::{}(args);
}}
)", args.launch_args.block_dim.x,
            c.single ? "ppu_mega_mhc_single" : "ppu_mega_mhc", specialization);
    }

    static void launch_impl(const KernelHandle& kernel, LaunchConfigHandle config, Args args) {
        LaunchAttrHandle attr{};
        if (args.cooperative) {
#if HGGCRT_VERSION >= 12080 && defined(DG_JIT_USE_RUNTIME_API)
            attr.id = hggcLaunchAttributeCooperative;
            attr.val.cooperative = 1;
#else
            attr.id = HG_LAUNCH_ATTRIBUTE_COOPERATIVE;
            attr.value.cooperative = 1;
#endif
            config.attrs = &attr;
            config.numAttrs = 1;
        }
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_args));
    }
};

struct MegaMhcPlan {
    MegaMhcRuntime::Config config;
    LaunchArgs launch_args;
    int num_sms;
    int64_t num_counters, num_partials;
    LaunchArgs pre_launch{}, finish_launch{};
};

struct PreparedMegaMhc {
    std::shared_ptr<KernelRuntime> runtime;
    MegaMhcRuntime::Args args;
};

static PreparedMegaMhc prepare_mega_mhc(MegaMhcRuntime::Args args, int num_sms,
                                       int physical_num_sms, bool cooperative_supported) {
    static auto mhc_compiler = LazyInit<HGCCCompiler>([]() { return std::make_shared<HGCCCompiler>(); });
    // Every specialization uses the same include closure for LaunchRuntime hashing.
    const auto name = args.config.phase == MegaMhcPhase::Pre ? "ppu_mega_mhc_tc01_pre" : args.config.phase == MegaMhcPhase::Finish ? "ppu_mega_mhc_tc01_finish" : args.config.tc01 ? "ppu_mega_mhc_tc01" : args.config.single ? "ppu_mega_mhc_single" : "ppu_mega_mhc";
    auto runtime = mhc_compiler->build(name, MegaMhcRuntime::generate(args));
    if (args.config.single && cooperative_supported) {
        int resident_blocks = 0;
#if HGGCRT_VERSION >= 12080 && defined(DG_JIT_USE_RUNTIME_API)
        DG_HGGC_RUNTIME_CHECK(hggcOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident_blocks, runtime->kernel, args.launch_args.block_dim.x, 0));
#else
        DG_HGGC_DRIVER_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident_blocks, runtime->kernel, args.launch_args.block_dim.x, 0));
#endif
        args.cooperative = resident_blocks * physical_num_sms >=
            args.launch_args.grid_dim.x * args.launch_args.grid_dim.y;
    }
    if (args.config.local && !args.cooperative) {
        args.launch_args.grid_dim.x *= args.launch_args.grid_dim.y;
        args.launch_args.grid_dim.y = 1;
        args.config.local = false;
        args.config.vector_coefficients = args.kernel_args.tokens != 1248;
        runtime = mhc_compiler->build(name, MegaMhcRuntime::generate(args));
    }
    if (get_env<int>("show_log", 0)) {
        const auto& c = args.config;
        const auto& launch = args.launch_args;
        int registers = 0, stack_bytes = 0, shared_bytes = 0, resident_blocks = 0;
#if HGGCRT_VERSION >= 12080 && defined(DG_JIT_USE_RUNTIME_API)
        hggcFuncAttributes attributes{};
        DG_HGGC_RUNTIME_CHECK(hggcFuncGetAttributes(&attributes, runtime->kernel));
        registers = attributes.numRegs;
        stack_bytes = attributes.localSizeBytes;
        shared_bytes = attributes.sharedSizeBytes;
        DG_HGGC_RUNTIME_CHECK(hggcOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident_blocks, runtime->kernel, launch.block_dim.x, launch.smem_size));
#else
        DG_HGGC_DRIVER_CHECK(hgFuncGetAttribute(&registers, HG_FUNC_ATTRIBUTE_NUM_REGS, runtime->kernel));
        DG_HGGC_DRIVER_CHECK(hgFuncGetAttribute(&stack_bytes, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, runtime->kernel));
        DG_HGGC_DRIVER_CHECK(hgFuncGetAttribute(&shared_bytes, HG_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, runtime->kernel));
        DG_HGGC_DRIVER_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident_blocks, runtime->kernel, launch.block_dim.x, launch.smem_size));
#endif
        const char* path = c.phase == MegaMhcPhase::Pre ? "prenorm" :
            c.phase == MegaMhcPhase::Finish ? "coefficient" :
            c.tc01 ? "projection" : c.single ? (c.local ? "local" : "persistent") : "general";
        const auto tile = c.phase == MegaMhcPhase::Pre ?
            fmt::format("tokens_per_cta:1, hidden_chunk:{}", c.tc01_pre_chunk) :
            c.phase == MegaMhcPhase::Finish ?
            fmt::format("tokens_per_cta:{}, coefficients:24", launch.block_dim.x / (c.splits < 32 ? 16 : 32)) :
            fmt::format("ThreadblockShape[{}, {}, 64]", c.single ? c.tile_m : 64, c.single ? 24 : 32);
        printf("[MegaMhc] kernel:%s path:%s hidden:%d sinkhorn_iters:%d\n"
               "problem:[%d, 24, %d]\n"
               "%s, num_splits:%d, task_splits:%d\n"
               "grid:[%u,%u,%u], BlockSize:%u, num_sms:%d, cooperative:%d\n"
               "output:%s sf_layout:%s shared_sf_block_m:%d\n"
               "smem_size:%d, vreg:%d, stack:%d, max_active_tb_num:%d\n",
               name, path,
               c.hidden, c.iterations, args.kernel_args.tokens, 4 * c.hidden,
               tile.c_str(), c.splits, c.single ? c.task_splits : 1,
               launch.grid_dim.x, launch.grid_dim.y, launch.grid_dim.z, launch.block_dim.x, num_sms, args.cooperative,
               c.store_fp8 ? (c.store_bf16 ? "bf16+fp8" : "fp8") : "bf16",
               c.store_fp8 ? (c.per_token_fp8 ? "token" : c.shared_sf_block_m > 0 ? "extra" : "col") : "none",
               c.shared_sf_block_m, shared_bytes + launch.smem_size, registers, stack_bytes, resident_blocks);
    }
    return {runtime, args};
}

static std::mutex mega_mhc_submit_mutex;

// A captured graph retains the allocation used during capture. Calls and graph
// replays sharing a capture stream must remain serialized on that stream.
static int* mega_mhc_counters(int64_t count, const torch::TensorOptions& options) {
    if (!count) return nullptr;
    static std::unordered_map<c10::cuda::CUDAStream, std::vector<torch::Tensor>> counters;
    auto& buffers = counters[at::cuda::getCurrentCUDAStream()];
    if (buffers.empty() || buffers.back().numel() < count) {
        DG_HOST_ASSERT(c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None and
                       "Warm up Mega-mHC on the capture stream with sufficient workspace capacity before capture");
        const int64_t capacity = buffers.empty() ? count : std::max(count, 2 * buffers.back().numel());
        buffers.push_back(torch::zeros({capacity}, options.dtype(torch::kInt)));
    }
    return buffers.back().data_ptr<int>();
}

static void run_mega_mhc(const MegaMhcPlan& plan, mhc::Args kernel_args,
                         const torch::TensorOptions& options) {
    std::lock_guard<std::mutex> lock(mega_mhc_submit_mutex);
    const int64_t partial_bytes = plan.num_partials * sizeof(float);
    const int64_t output_bytes = kernel_args.y ? 0 : int64_t(kernel_args.tokens) * plan.config.hidden * sizeof(__ppu_bfloat16);
    const auto scratch = torch::empty({partial_bytes + output_bytes}, options.dtype(torch::kByte));
    kernel_args.partials = static_cast<float*>(scratch.data_ptr());
    kernel_args.counters = mega_mhc_counters(plan.num_counters, options);
    if (!kernel_args.y)
        kernel_args.y = reinterpret_cast<__ppu_bfloat16*>(static_cast<uint8_t*>(scratch.data_ptr()) + partial_bytes);
    MegaMhcRuntime::Args args{plan.config, plan.launch_args, kernel_args};
    const int num_sms = plan.num_sms;
    const auto* prop = at::cuda::getCurrentDeviceProperties();
    std::array<PreparedMegaMhc, 3> kernels;
    int num_kernels = 0;
    if (!args.config.tc01) {
        kernels[num_kernels++] = prepare_mega_mhc(args, num_sms, prop->multiProcessorCount,
                                         !args.config.tc01 && prop->cooperativeLaunch);
    } else {
        args.config.phase = MegaMhcPhase::Pre;
        args.launch_args = plan.pre_launch;
        kernels[num_kernels++] = prepare_mega_mhc(args, num_sms, prop->multiProcessorCount, false);
        args.config.phase = MegaMhcPhase::Main;
        args.launch_args = plan.launch_args;
        kernels[num_kernels++] = prepare_mega_mhc(args, num_sms, prop->multiProcessorCount, false);
        args.config.phase = MegaMhcPhase::Finish;
        args.launch_args = plan.finish_launch;
        kernels[num_kernels++] = prepare_mega_mhc(args, num_sms, prop->multiProcessorCount, false);
    }
    DgProfParam dg_prof_params;
    if (deep_gemm::profiling_enabled()) {
        const auto& c = args.config;
        dg_prof_params.set_mega_mhc_params(args.kernel_args.tokens, c.hidden, c.iterations, current_stream());
        if (c.store_fp8) {
            dg_prof_params.add_params("data_type", "fp8");
            dg_prof_params.add_params("sf_layout", c.per_token_fp8 ? "token" : c.shared_sf_block_m > 0 ? "extra" : "col");
            dg_prof_params.add_params("store_bf16", c.store_bf16);
            if (c.shared_sf_block_m > 0)
                dg_prof_params.add_params("shared_sf_block_m", c.shared_sf_block_m);
        }
        dg_prof_params.add_params("num_splits", args.config.splits);
        dg_prof_params.add_params("num_sms", num_sms);
    }
    auto profiling_guard = show_gemm_ut(dg_prof_params);
    for (int i = 0; i < num_kernels; ++i)
        MegaMhcRuntime::launch(kernels[i].runtime, kernels[i].args);
}

} // namespace deep_gemm
