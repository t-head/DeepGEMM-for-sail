#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/utils.hpp"
#include <deep_gemm/common/profiling_interface.cuh>
#define DG_MHC_ARGS_ONLY
#include <deep_gemm/impls/ppu_mega_mhc_bf16.cuh>
#undef DG_MHC_ARGS_ONLY

namespace deep_gemm {

class MegaMhcRuntime final : public LaunchRuntime<MegaMhcRuntime> {
public:
    struct Config {
        int hidden, splits, iterations, tile_m, task_splits;
        bool full_rows, single, local, tail_rows, tail_tokens, vector_coefficients;
    };
    struct Args {
        Config config;
        LaunchArgs launch_args;
        mhc::Args kernel_args;
        bool cooperative = false;
    };

    static std::string generate_impl(const Args& args) {
        const auto& c = args.config;
        std::string specialization;
        if (c.single) {
            specialization = fmt::format(
                "ppu_mega_mhc_shifted_bf16_device<{},{},{},false,true,{},{},{},{},{},{}>",
                c.hidden, c.splits, c.iterations, c.local, c.tile_m, c.task_splits,
                c.tail_rows, c.tail_tokens, c.vector_coefficients);
        } else {
            specialization = fmt::format("ppu_mega_mhc_shifted_bf16_device<{},{},{},{}>",
                                         c.hidden, c.splits, c.iterations, c.full_rows);
        }
        return fmt::format(R"(
#include <deep_gemm/impls/ppu_mega_mhc_bf16.cuh>
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

static void launch_mega_mhc(MegaMhcRuntime::Args args, int num_sms, int physical_num_sms, bool cooperative_supported) {
    static auto mhc_compiler = LazyInit<HGCCCompiler>([]() { return std::make_shared<HGCCCompiler>(); });
    const auto name = args.config.single ? "ppu_mega_mhc_single" : "ppu_mega_mhc";
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
    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_mega_mhc_params(
            args.kernel_args.tokens, args.config.hidden, args.config.iterations, current_stream());
        dg_prof_params.add_params("num_splits", args.config.splits);
        dg_prof_params.add_params("num_sms", num_sms);
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);
    MegaMhcRuntime::launch(runtime, args);
    ProfilingInterface::Instance().instrument(false, dg_prof_params);
}

} // namespace deep_gemm
