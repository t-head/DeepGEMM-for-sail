#pragma once

#include <memory>
#include <type_traits>
#include "../../jit/kernel_runtime.hpp"
#include "../../jit/compiler.hpp"
#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm::mega_moe_common {

class MegaMoESyncRuntime final : public LaunchRuntime<MegaMoESyncRuntime> {
public:
    struct Args {
        const char* kernel_name;
        int num_ranks;
        LaunchArgs launch_args;
        layout::MegaMoESignals* signals;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/mega_moe_sync.cuh>
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::{}<{}>);
}}
)", args.kernel_name, args.num_ranks);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.signals));
    }
};

static void launch_sync(const char* name, int num_ranks, layout::MegaMoESignals* signals) {
    MegaMoESyncRuntime::Args args{name, num_ranks, {dim3(1), dim3(64), 0}, signals};
    const auto runtime = compiler->build(name, MegaMoESyncRuntime::generate(args));
    MegaMoESyncRuntime::launch(runtime, args);
}

template <typename F>
static void for_ranks(int ranks, F&& f) {
    switch (ranks) {
        case 1: return f(std::integral_constant<int, 1>{});
        case 2: return f(std::integral_constant<int, 2>{});
        case 4: return f(std::integral_constant<int, 4>{});
        case 8: return f(std::integral_constant<int, 8>{});
        case 16: return f(std::integral_constant<int, 16>{});
        case 32: return f(std::integral_constant<int, 32>{});
        case 64: return f(std::integral_constant<int, 64>{});
        default: DG_HOST_UNREACHABLE("unsupported MegaMoE rank count");
    }
}

static int get_kernel_occupancy(const std::shared_ptr<KernelRuntime>& runtime, int threads, int dynamic_smem) {
    construct_launch_config(runtime->kernel, current_stream(), dynamic_smem, dim3(1), dim3(threads));
    int blocks = 0;
#if HGGCRT_VERSION >= 12080 && defined(DG_JIT_USE_RUNTIME_API)
    DG_HGGC_RUNTIME_CHECK(hggcOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, runtime->kernel, threads, dynamic_smem));
#else
    DG_HGGC_DRIVER_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, runtime->kernel, threads, dynamic_smem));
#endif
    DG_HOST_ASSERT(blocks > 0);
    return blocks;
}

struct KernelResources {
    int registers;
    int stack_bytes;
};

static KernelResources get_kernel_resources(const std::shared_ptr<KernelRuntime>& runtime) {
    int registers = 0, stack_bytes = 0;
#if HGGCRT_VERSION >= 12080 && defined(DG_JIT_USE_RUNTIME_API)
    hggcFuncAttributes attributes{};
    DG_HGGC_RUNTIME_CHECK(hggcFuncGetAttributes(&attributes, runtime->kernel));
    registers = attributes.numRegs;
    stack_bytes = attributes.localSizeBytes;
#else
    DG_HGGC_DRIVER_CHECK(hgFuncGetAttribute(&registers, HG_FUNC_ATTRIBUTE_NUM_REGS, runtime->kernel));
    DG_HGGC_DRIVER_CHECK(hgFuncGetAttribute(&stack_bytes, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, runtime->kernel));
#endif
    return {registers, stack_bytes};
}

}  // namespace deep_gemm::mega_moe_common
