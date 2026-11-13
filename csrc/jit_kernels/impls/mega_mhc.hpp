#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/utils.hpp"
#include <deep_gemm/common/profiling_interface.cuh>
#define DG_MHC_ARGS_ONLY
#include <deep_gemm/impls/ppu_mega_mhc.cuh>
#undef DG_MHC_ARGS_ONLY

namespace deep_gemm {

class MegaMhcRuntime final : public LaunchRuntime<MegaMhcRuntime> {
public:
    struct Config {
        int hidden, splits, iterations, tile_m, task_splits;
        bool full_rows, single, local, tail_rows, tail_tokens, vector_coefficients;
        bool store_bf16, store_fp8;
        int shared_sf_block_m;
        bool per_token_fp8;
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
                "ppu_mega_mhc_shifted_device<{},{},{},false,true,{},{},{},{},{},{},{},{},{},{}>",
                c.hidden, c.splits, c.iterations, c.local, c.tile_m, c.task_splits,
                c.tail_rows, c.tail_tokens, c.vector_coefficients, c.store_bf16, c.store_fp8, c.shared_sf_block_m, c.per_token_fp8);
        } else {
            specialization = fmt::format("ppu_mega_mhc_shifted_device<{},{},{},{},false,false,32,2,false,false,true,{},{},{},{}>",
                                         c.hidden, c.splits, c.iterations, c.full_rows,
                                         c.store_bf16, c.store_fp8, c.shared_sf_block_m, c.per_token_fp8);
        }
        return fmt::format(R"(
#include <deep_gemm/impls/ppu_mega_mhc.cuh>
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
        printf("[MegaMhc] kernel:%s path:%s hidden:%d sinkhorn_iters:%d\n"
               "problem:[%d, 24, %d]\n"
               "ThreadblockShape[%d, %d, 64], num_splits:%d, task_splits:%d\n"
               "grid:[%u,%u,%u], BlockSize:%u, num_sms:%d, cooperative:%d\n"
               "output:%s sf_layout:%s shared_sf_block_m:%d\n"
               "smem_size:%d, vreg:%d, stack:%d, max_active_tb_num:%d\n",
               name, c.single ? (c.local ? "local" : "persistent") : "general",
               c.hidden, c.iterations, args.kernel_args.tokens, 4 * c.hidden,
               c.single ? c.tile_m : 64, c.single ? 24 : 32, c.splits, c.single ? c.task_splits : 1,
               launch.grid_dim.x, launch.grid_dim.y, launch.grid_dim.z, launch.block_dim.x, num_sms, args.cooperative,
               c.store_fp8 ? (c.store_bf16 ? "bf16+fp8" : "fp8") : "bf16",
               c.store_fp8 ? (c.per_token_fp8 ? "token" : c.shared_sf_block_m > 0 ? "extra" : "col") : "none",
               c.shared_sf_block_m, shared_bytes + launch.smem_size, registers, stack_bytes, resident_blocks);
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
    {
        auto profiling_guard = deep_gemm::show_gemm_ut(dg_prof_params);
        MegaMhcRuntime::launch(runtime, args);
    }
}

} // namespace deep_gemm
