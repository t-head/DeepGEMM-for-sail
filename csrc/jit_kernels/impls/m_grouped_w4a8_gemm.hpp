#pragma once
#include <torch/python.h>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"
#include "../heuristics/common_w4a8.hpp"
#include <deep_gemm/common/profiling_interface.cuh>
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>
#include "fp8_gemm.hpp" // `ComputeBlockInfoKernelRuntime`


namespace deep_gemm::w4a8 {
// Host mirror of the device-only FusedGemmScheduler::Params.
struct W4A8FusedSchedulerArguments {
    const int32_t* aligned_num_m_blocks;
    const int32_t* expert_ids_and_cumsum;
};

template<class Scheduler>
struct Arguments {
    const void* a;
    const void* b;
    const void* scale;
    void* d;
    const float* a_scale;
    int num_token;
    const int* sorted_token_ids;
    Scheduler scheduler;
};

template<class Scheduler>
class Runtime final : public LaunchRuntime<Runtime<Scheduler>> {
public:
    struct Args {
        W4A8Config config;
        int n, k, groups;
        GemmType kind;
        LaunchArgs launch_args;
        Arguments<Scheduler> params;
    };
    static std::string generate_impl(const Args& a) {
        const auto& c = a.config;
        return fmt::format(R"(
#include <deep_gemm/impls/w4fa8_gemm.cuh>
using Kernel = cutlass::gemm::kernel::W4A8GEMM_MMA<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, GemmType::{}, {}>;
static_assert(sizeof(Kernel::SharedStorage) == {}, "shared memory mismatch");
static_assert(Kernel::MaxThreadsPerBlock == {}, "thread count mismatch");
static_assert(sizeof(Kernel::Params) == {}, "kernel argument size mismatch");
extern "C" __global__ __launch_bounds__(Kernel::MaxThreadsPerBlock, Kernel::MinBlocksPerMultiprocessor)
void m_grouped_gemm_w4fa8_mma(Kernel::Params params) {{
    extern __shared__ char smem[];
    Kernel{{}}(params, smem);
}}
)", a.n, a.k, c.block_m, c.block_n, c.block_k, c.warp_m, c.warp_n, c.warp_k,
            a.groups, c.num_stages, get_gemm_type_name(a.kind), c.n_expand,
            get_smem_size(c), get_num_threads(c), sizeof(Arguments<Scheduler>));
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.params));
    }
};

template<class Scheduler>
static void launch(const W4A8Config& c, int m, int n, int k, int groups, GemmType kind, int expected_m,
                   int topk, int32_t* m_rows, const Arguments<Scheduler>& params) {
    using R = Runtime<Scheduler>;
    const int threads = get_num_threads(c), smem = get_smem_size(c);
    typename R::Args args{c, n, k, groups, kind, {dim3(c.num_sms), dim3(threads), smem}, params};
    const auto& runtime = compiler->build("m_grouped_gemm_w4fa8_mma", R::generate(args));
    construct_launch_config(runtime->kernel, current_stream(), smem, args.launch_args.grid_dim, args.launch_args.block_dim);
    int blocks = 0;
    DG_HGGC_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, runtime->kernel, threads, smem));
    DG_HOST_ASSERT(blocks > 0);
    args.launch_args.grid_dim = dim3(c.num_sms < 20 ? c.num_sms : c.num_sms * std::min(8, blocks));
    DgProfParam prof;
    if (ProfilingInterface::Instance().get_op_info()) {
        if (kind == GemmType::GroupedFused)
            prof.set_fused_moe_params("w4fa8_mma", "group", groups, m, topk, n, k, m_rows, current_stream());
        else {
            prof.set_params(kind, false, "w4fa8_mma", groups, m, n, k, expected_m, m_rows, current_stream());
            prof.add_params(std::string("quant_type"), std::string("group"));
        }
        prof.add_params(std::string("group_size"), 32);
    }
    if (const char* log = std::getenv("show_log"); log && isdigit(*log)) {
        int regs = 0, stack = 0;
        hgFuncGetAttribute(&regs, HG_FUNC_ATTRIBUTE_NUM_REGS, runtime->kernel);
        hgFuncGetAttribute(&stack, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, runtime->kernel);
        printf("[W4FA8] group:%d problem:[%d,%d,%d] block:[%d,%d,%d] warp:[%d,%d,%d] stages:%d smem:%d vreg:%d stack:%d tb_per_sm:%d threads:%d n_expand:%d\n",
               groups,m,n,k,c.block_m,c.block_n,c.block_k,c.warp_m,c.warp_n,c.warp_k,c.num_stages,smem,regs,stack,blocks,threads,c.n_expand);
    }
    ProfilingInterface::Instance().instrument(true, prof);
    R::launch(runtime, args);
    ProfilingInterface::Instance().instrument(false, prof);
}

static void run(const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b,
                const torch::Tensor& scale, const torch::Tensor& d, const torch::Tensor& rows,
                GemmType kind, int expected_m, const W4A8Config& c,
                const torch::Tensor& expert_ids = {}, const torch::Tensor& token_ids = {},
                const torch::Tensor& aligned_blocks = {}) {
    const int groups = b.size(0), n = b.size(1), k = a.size(-1);
    const int m = kind == GemmType::GroupedMasked ? a.size(1) : a.size(0);
    check_config(c, n, k, kind);
    if (kind == GemmType::GroupedFused) {
        const Arguments<W4A8FusedSchedulerArguments> params{a.data_ptr(), b.data_ptr(), scale.data_ptr(), d.data_ptr(),
            a_scale.data_ptr<float>(), m, token_ids.data_ptr<int32_t>(),
            {aligned_blocks.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>()}};
        launch(c,m,n,k,groups,kind,expected_m,d.size(0)/m,rows.data_ptr<int32_t>(),params);
    } else {
        torch::Tensor scratch;
        auto layout = rows.data_ptr<int32_t>();
        if (kind == GemmType::GroupedNoPad && groups >= 128) {
            TORCH_CHECK(groups <= 1024, "W4FA8 nopad supports at most 1024 experts");
            scratch = at::empty({(groups + ceil_div(m + 1 - groups, c.block_m))*4}, rows.options());
            auto args = ComputeBlockInfoKernelRuntime::Args{
                .launch_attr_args = {reinterpret_cast<const uint32_t*>(layout), static_cast<uint32_t>(groups),
                                    reinterpret_cast<uint32_t*>(scratch.data_ptr<int32_t>())},
                .launch_args = {1, std::max(32, next_power_of_two(groups)), 0}};
            const auto& code = ComputeBlockInfoKernelRuntime::generate(c.block_m);
            const auto& runtime = compiler->build("computeBlockInfoKernel", code);
            ComputeBlockInfoKernelRuntime::launch(runtime, args);
            layout = scratch.data_ptr<int32_t>();
        }
        const Arguments<TileSchedulerArguments> params{a.data_ptr(),b.data_ptr(),scale.data_ptr(),d.data_ptr(),
            a_scale.data_ptr<float>(),0,nullptr,TileSchedulerArguments(static_cast<uint32_t>(m),layout)};
        launch(c,m,n,k,groups,kind,expected_m,0,rows.data_ptr<int32_t>(),params);
    }
}
} // namespace deep_gemm::w4a8
