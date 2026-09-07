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
#include "../heuristics/common_w4a16.hpp"
#include <deep_gemm/common/profiling_interface.cuh>
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>
#include "fp8_gemm.hpp" // `ComputeBlockInfoKernelRuntime`

using namespace deep_gemm_w4a16_common;
namespace deep_gemm {

// Host-side mirror of `FusedGemmScheduler<...>::Params`.
//
// NOTES: `fused_scheduler.cuh` cannot be included from host code -- its constructors are plain
// `__device__ __forceinline__` -- so the two-pointer POD is replicated here. Field order matches, and
// the kernel is initialised the same way `W4A16Gemm::run` does it:
// `params.scheduler = {args.aligned_num_m_blocks, args.expert_ids_and_cumsum}`.
struct FusedSchedulerArguments {
    const int32_t* aligned_num_m_blocks;
    const int32_t* expert_ids_and_cumsum;
};

// Host-side mirror of `cutlass::gemm::kernel::W4A16GEMM<...>::Params` (and of `W4A16GEMM_MMA`, which
// declares the same fields).
//
// NOTES: the kernel takes this struct directly as its kernel argument -- we deliberately do NOT wrap
// it in a `HostParams` block that the kernel unpacks, since constructing `Params` inside the kernel
// costs registers on a pressure-sensitive kernel. The scheduler arm differs between the fused and
// non-fused kernels and changes the struct layout, hence the template; the value pointers stay
// `void*` because their element types only exist inside the generated device code.
template <typename SchedulerArgsT>
struct W4A16GemmArguments {
    const void* ptr_a;
    const void* ptr_b;
    const void* ptr_scale;
    void* ptr_d;
    int num_token;
    const int* sorted_token_ids;
    SchedulerArgsT scheduler;
};

// Everything the generated code needs, all resolved on the host
struct W4A16LaunchInfo {
    std::string element_b, element_scale;
    int n, k;
    int block_m, block_n, block_k;
    int warp_m, warp_n, warp_k;
    int num_groups, num_stages, group_size, n_expand;
    std::string gemm_type_name;
    bool use_mma_kernel;
    int smem_size, num_threads;
    std::string kernel_name;
};

// One runtime serves both W4A16 kernels: they only differ by class name and template parameter list,
// neither of which affects the kernel argument layout. The scheduler arm does, so that stays a
// template parameter.
template <typename SchedulerArgsT>
class W4A16GemmRuntime final : public LaunchRuntime<W4A16GemmRuntime<SchedulerArgsT>> {
public:
    struct Args {
        W4A16LaunchInfo launch_info;
        LaunchArgs launch_args;
        W4A16GemmArguments<SchedulerArgsT> kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        const auto& info = args.launch_info;
        // `W4A16GEMM_MMA` hardcodes the weight/scale element types and the group size, so its template
        // parameter list drops those three
        const std::string kernel_type =
            info.use_mma_kernel
                ? fmt::format("cutlass::gemm::kernel::W4A16GEMM_MMA<\n"
                              "  N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K,\n"
                              "  kNumGroups, kNumStages, GemmType::{}, N_EXPAND\n>",
                              info.gemm_type_name)
                : fmt::format("cutlass::gemm::kernel::W4A16GEMM<\n"
                              "  ElementB, ElementScale, N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K,\n"
                              "  kNumGroups, kNumStages, GemmType::{}, kGroupSize, N_EXPAND\n>",
                              info.gemm_type_name);
        return fmt::format(
            R"(
#define W4A16_HGRTC
#include <deep_gemm/impls/w4a16_gemm_cutlass3.cuh>
namespace deep_gemm {{
using namespace cute;

using ElementB = {};
using ElementScale = {};

constexpr auto N = {}, K = {};
constexpr auto BLOCK_M = {};
constexpr auto BLOCK_N = {};
constexpr auto WARP_M = {};
constexpr auto WARP_N = {};
constexpr auto WARP_K = {};
constexpr auto BLOCK_K = {};
constexpr auto kNumGroups = {};
constexpr auto kNumStages = {};
constexpr auto kGroupSize = {};
constexpr auto N_EXPAND = {};

using GemmKernel = {};

// The host computes these instead of reading them off the kernel type, so pin them down here
static_assert(sizeof(typename GemmKernel::SharedStorage) == {}, "host/device shared memory size mismatch");
static_assert(GemmKernel::MaxThreadsPerBlock == {}, "host/device thread count mismatch");

extern "C"
__launch_bounds__(GemmKernel::MaxThreadsPerBlock, GemmKernel::MinBlocksPerMultiprocessor)
__global__ void {}(
  typename GemmKernel::Params params
) {{
  extern __shared__ char smem[];
  GemmKernel op;
  op(params, smem);
}}
}} // namespace deep_gemm
)",
            info.element_b, info.element_scale, info.n, info.k, info.block_m, info.block_n, info.warp_m, info.warp_n,
            info.warp_k, info.block_k, info.num_groups, info.num_stages, info.group_size, info.n_expand, kernel_type,
            info.smem_size, info.num_threads, info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

// Dispatch helper: builds, launches and reports, for one concrete scheduler arm. Mirrors the launch
// tail of `W4A16Gemm::run`.
template <typename SchedulerArgsT>
static void launch_w4a16_gemm(const W4A16LaunchInfo& info, W4A16Type w4a16_type, GemmType gemm_type, int num_sms, int m,
                              int expected_m, int topk, int32_t* m_rows_ptr, bool no_pad_preprocess_layout,
                              const W4A16GemmArguments<SchedulerArgsT>& kernel_params) {
    using Runtime = W4A16GemmRuntime<SchedulerArgsT>;

    auto args = typename Runtime::Args{
        .launch_info = info,
        .launch_args = {dim3(num_sms, 1, 1), dim3(info.num_threads, 1, 1), info.smem_size},
        .kernel_params = kernel_params,
    };

    const auto& code = Runtime::generate(args);
    const auto& runtime = compiler->build(info.kernel_name, code, info.num_threads, info.smem_size);
    const auto& kernel = runtime->kernel;

    // Persistent kernel, replicating `W4A16Gemm::run`: occupancy is capped at 8 blocks per CU, and the
    // grid is only widened on devices with at least 20 CUs.
    // NOTES: the occupancy query reports 0 once the dynamic shared memory passes the default 48 KiB cap,
    // so raise the cap first -- `compute_occupancy_for_kernel`, which `run` used, did the same. The
    // config built here is only wanted for that side effect; `Runtime::launch` builds its own.
    construct_launch_config(kernel, (hggcStream_t)0, info.smem_size, args.launch_args.grid_dim,
                            args.launch_args.block_dim);
    int blocks_per_cu = 0;
    hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, info.num_threads, info.smem_size);
    const int tb_per_sm = std::min(8, blocks_per_cu);
    const int threadblock_count = num_sms < 20 ? num_sms : num_sms * tb_per_sm;
    args.launch_args.grid_dim = dim3(threadblock_count, 1, 1);

    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        // The fused arm is reported in token/topk terms instead of the m/em ones, and
        // `set_fused_moe_params` records `quant_type` itself, so only the other arm adds it
        if (gemm_type == GemmType::GroupedFused) {
            dg_prof_params.set_fused_moe_params(get_profiling_dtype_name(w4a16_type), std::string("group"),
                                                info.num_groups, m, topk, info.n, info.k, m_rows_ptr,
                                                (hggcStream_t)0);
        } else {
            dg_prof_params.set_params(gemm_type, false, get_profiling_dtype_name(w4a16_type), info.num_groups, m,
                                      info.n, info.k, expected_m, m_rows_ptr, (hggcStream_t)0);
            dg_prof_params.add_params(std::string("quant_type"), std::string("group"));
        }
        dg_prof_params.add_params(std::string("group_size"), info.group_size);
    }
    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        int numRegs = 0, localSize = 0;
        hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
        hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

        printf("[GemmGrouped-W4A16:] quant_type:%s, scale_type:%s\n",
               w4a16_type == W4A16Type::int4 ? "int4" : "fp4",
               info.element_scale == "uint8_t" ? "e8m0" : "bf16");
        printf("group:%d, problem:[%d, %d, %d], expected_m:%d, gemm_type:%s, NoPadPreprocessLayout: %d, N_EXPAND: %d\n",
               info.num_groups, m, info.n, info.k, expected_m, info.gemm_type_name.c_str(), no_pad_preprocess_layout,
               info.n_expand);
        printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], kNumStages:%d\n", info.block_m, info.block_n,
               info.block_k, info.warp_m, info.warp_n, info.warp_k, info.num_stages);
        printf("num_sms:%d, tb_per_sm:%d, threadblock_count:%d, num_threads: %d\n", num_sms, tb_per_sm,
               threadblock_count, info.num_threads);
        printf("smem_size:%d, vreg:%d, stack:%d\n", info.smem_size, int(numRegs), int(localSize));
    }

    ProfilingInterface::Instance().instrument(true, dg_prof_params);
    Runtime::launch(runtime, args);
    ProfilingInterface::Instance().instrument(false, dg_prof_params);
}

// Shared launch path for the three W4A16 grouped GEMM flavours, mirroring `m_grouped_gemm_w4a16_common`.
//
// NOTES: the operand shape/dtype contract is enforced by `check_w4a16_operands` in `apis/gemm.hpp`, which
// also derives `m` / `n` / `k` / `num_groups` / `group_size` and passes them in. Only the config-dependent
// assertions stay here, since `config` may be picked by `get_best_configs` below the API layer.
//
// Args:
//   w4a16_type: quantization layout and dequant path selector
//   lhs: activation tensor in BF16, shape nopad: (m, k), masked: (num_groups, m, k), fused: (num_token, k)
//   rhs / rhs_scales:
//     - normal weight: 4-bit weight stored in int32, shape (num_groups, k / 16, n * 2)
//     - mma weight: packed uint8 weight, shape (num_groups, n, k / 2)
//     - normal scale: BF16 numerical scale or uint8 raw E8M0 exponent bytes, shape (num_groups, k / group_size, n)
//     - mma scale: packed uint8 raw E8M0 exponent bytes, shape (num_groups, n // 64, k * 2)
//   out: output tensor in BF16, shape nopad/fused: (m, n), masked: (num_groups, m, n)
//   m / n / k / num_groups / group_size: problem sizes derived by the API layer
//   m_rows: number of rows per group, shape (num_groups,)
//   block_m_info: `GroupedNoPad` scratch buffer for the preprocessed layout
//   expert_ids_and_cumsum / sorted_token_ids / aligned_num_m_blocks / topk: `GroupedFused` scheduler inputs
static void m_grouped_gemm_w4a16_common(W4A16Type w4a16_type, GemmType gemm_type, int expected_m,
                                       const torch::Tensor& lhs, const torch::Tensor& rhs,
                                       const torch::Tensor& rhs_scales, const torch::Tensor& out,
                                       int m, int n, int k, int num_groups, int group_size,
                                       const W4A16Config& config, const torch::Tensor& m_rows,
                                       const torch::Tensor& block_m_info = torch::Tensor(),
                                       const torch::Tensor& expert_ids_and_cumsum = torch::Tensor(),
                                       const torch::Tensor& sorted_token_ids = torch::Tensor(),
                                       const torch::Tensor& aligned_num_m_blocks = torch::Tensor(),
                                       int topk = 0) {
    const bool is_mma = uses_mma_kernel(w4a16_type);
    if (is_mma) {
        // w4fa16_mma is only supported on PPU1.5
        DG_HOST_ASSERT(is_ppu1v5_device());
    }

    DG_HOST_ASSERT(config.warp_n == 64);
    if (is_mma)
        DG_HOST_ASSERT(config.block_k >= 128 and
                       (config.warp_k == 64 or (config.warp_k == config.block_k and config.block_k == 128)));

    const std::string gemm_type_name = get_gemm_type_name(gemm_type);
    // w4fa16_mma not use "-sort-copy-before-coalesce": the compiler keys that flag off the "w4a16"
    // substring in the kernel name, which is also the `extern "C"` symbol the generated code exports
    const std::string kernel_name = is_mma ? "m_grouped_gemm_w4fa16_mma" : "m_grouped_gemm_w4a16";

    const W4A16LaunchInfo info{
        get_element_b(w4a16_type),
        get_element_scale(w4a16_type),
        n,
        k,
        config.block_m,
        config.block_n,
        config.block_k,
        config.warp_m,
        config.warp_n,
        config.warp_k,
        num_groups,
        config.num_stages,
        group_size,
        config.n_expand,
        gemm_type_name,
        is_mma,
        get_smem_config(config, is_mma, static_cast<int>(rhs_scales.element_size()), group_size),
        get_num_threads(config, is_mma),
        kernel_name,
    };

    const void* ptr_a = lhs.data_ptr();
    const void* ptr_b = rhs.data_ptr();
    const void* ptr_scale = rhs_scales.data_ptr();
    void* ptr_d = out.data_ptr();
    int32_t* m_rows_ptr = m_rows.data_ptr<int32_t>();

    if (gemm_type == GemmType::GroupedFused) {
        const W4A16GemmArguments<FusedSchedulerArguments> kernel_params{
            ptr_a,
            ptr_b,
            ptr_scale,
            ptr_d,
            m,
            sorted_token_ids.data_ptr<int32_t>(),
            FusedSchedulerArguments{aligned_num_m_blocks.data_ptr<int32_t>(),
                                    expert_ids_and_cumsum.data_ptr<int32_t>()},
        };
        launch_w4a16_gemm(info, w4a16_type, gemm_type, config.num_sms, m, expected_m, topk, m_rows_ptr,
                          false /*no_pad_preprocess_layout*/, kernel_params);
        return;
    }

    int32_t* layout_info = m_rows_ptr;
    // `DeepGemmScheduler::kIsNoPadPreprocessLayout`; the `GroupedFused` half of that predicate is
    // handled by the branch above
    const bool no_pad_preprocess_layout = gemm_type == GemmType::GroupedNoPad and num_groups >= 128;
    if (no_pad_preprocess_layout) {
        const uint32_t block_size = std::max(32, next_power_of_two(num_groups));
        auto compute_block_info_args = ComputeBlockInfoKernelRuntime::Args{
            .launch_attr_args = {reinterpret_cast<const uint32_t*>(m_rows_ptr), static_cast<uint32_t>(num_groups),
                                 reinterpret_cast<uint32_t*>(block_m_info.data_ptr<int32_t>())},
            .launch_args = {1, block_size, 0},
        };
        const auto& code_blockinfo = ComputeBlockInfoKernelRuntime::generate(config.block_m);
        const auto& runtime_blockinfo = compiler->build("computeBlockInfoKernel", code_blockinfo);
        ComputeBlockInfoKernelRuntime::launch(runtime_blockinfo, compute_block_info_args);
        layout_info = block_m_info.data_ptr<int32_t>();
    }

    // NOTES: `num_token` / `sorted_token_ids` are only read by the fused scheduler, so they are zeroed
    // here -- the Python path left them uninitialised for the same reason
    const W4A16GemmArguments<TileSchedulerArguments> kernel_params{
        ptr_a,
        ptr_b,
        ptr_scale,
        ptr_d,
        0,
        nullptr,
        TileSchedulerArguments(static_cast<uint32_t>(m), layout_info),
    };
    launch_w4a16_gemm(info, w4a16_type, gemm_type, config.num_sms, m, expected_m, topk, m_rows_ptr,
                      no_pad_preprocess_layout, kernel_params);
}

// =============================================================================
// W4A16 Grouped GEMM API Implementations
// =============================================================================

static void m_grouped_gemm_w4a16_nopad_impl(const torch::Tensor& lhs, const torch::Tensor& rhs,
                                           const torch::Tensor& rhs_scales, const torch::Tensor& out,
                                           const torch::Tensor& m_indices, const torch::Tensor& m_rows,
                                           int m, int n, int k, int num_groups, int group_size,
                                           std::optional<W4A16ConfigTuple> configs = std::nullopt,
                                           bool fp4_use_bf16_scale = false) {
    const auto w4a16_type = get_w4a16_type(rhs.scalar_type(), rhs_scales.scalar_type(), fp4_use_bf16_scale);
    const int expected_m = ceil_div(m, num_groups);
    const auto& config = configs.has_value() ? unpack_w4a16_config(*configs)
                                             : get_best_configs(expected_m, n, k, num_groups, get_num_sms(),
                                                                GemmType::GroupedNoPad, w4a16_type);
    const int block_m = config.block_m;

    // Derive per-group row counts from `m_indices` when the caller did not supply them
    torch::Tensor m_rows_tensor;
    if (not m_rows.defined() or m_rows.numel() == 0) {
        const auto& counts = at::bincount(m_indices);
        const int64_t min_n = std::min<int64_t>(counts.size(0), num_groups);
        auto experts_for_rows =
            at::zeros({num_groups}, at::TensorOptions().dtype(at::kInt).device(m_indices.device()));
        if (min_n > 0)
            experts_for_rows.narrow(0, 0, min_n).copy_(counts.narrow(0, 0, min_n).to(at::kInt));
        m_rows_tensor = experts_for_rows;
    } else {
        m_rows_tensor = m_rows;
    }

    const auto& block_m_info =
        at::empty({(num_groups + ceil_div(m + 1 - num_groups, block_m)) * 4},
                  at::TensorOptions().dtype(at::kInt).device(m_rows_tensor.device()));

    m_grouped_gemm_w4a16_common(w4a16_type, GemmType::GroupedNoPad, expected_m, lhs, rhs, rhs_scales, out, m, n, k,
                                num_groups, group_size, config, m_rows_tensor, block_m_info);
}

static void m_grouped_gemm_w4a16_masked_impl(const torch::Tensor& lhs, const torch::Tensor& rhs,
                                            const torch::Tensor& rhs_scales, const torch::Tensor& out,
                                            const torch::Tensor& masked_m,
                                            int m, int n, int k, int num_groups, int group_size, int expected_m,
                                            std::optional<W4A16ConfigTuple> configs = std::nullopt,
                                            bool fp4_use_bf16_scale = false) {
    const auto w4a16_type = get_w4a16_type(rhs.scalar_type(), rhs_scales.scalar_type(), fp4_use_bf16_scale);
    const auto& config = configs.has_value() ? unpack_w4a16_config(*configs)
                                             : get_best_configs(expected_m, n, k, num_groups, get_num_sms(),
                                                                GemmType::GroupedMasked, w4a16_type);

    m_grouped_gemm_w4a16_common(w4a16_type, GemmType::GroupedMasked, expected_m, lhs, rhs, rhs_scales, out, m, n, k,
                                num_groups, group_size, config, masked_m);
}

static void m_grouped_gemm_w4a16_fused_impl(const torch::Tensor& lhs, const torch::Tensor& rhs,
                                           const torch::Tensor& rhs_scales, const torch::Tensor& out,
                                           const torch::Tensor& m_rows, const torch::Tensor& expert_ids_and_cumsum,
                                           const torch::Tensor& sorted_token_ids,
                                           const torch::Tensor& aligned_num_m_blocks,
                                           int m, int n, int k, int num_groups, int group_size,
                                           const W4A16ConfigTuple& configs, bool fp4_use_bf16_scale = false) {
    const auto w4a16_type = get_w4a16_type(rhs.scalar_type(), rhs_scales.scalar_type(), fp4_use_bf16_scale);
    const auto config = unpack_w4a16_config(configs);
    // `m` is the `num_token` the API layer read off `lhs`
    const int64_t num_token = m;
    const int64_t m_sum = out.size(0);
    // out rows must be divisible by num_token
    DG_HOST_ASSERT(m_sum % num_token == 0);
    const int64_t topk = m_sum / num_token;
    DG_HOST_ASSERT(num_groups >= topk);
    const int expected_m = ceil_div(static_cast<int>(m_sum), num_groups);

    m_grouped_gemm_w4a16_common(w4a16_type, GemmType::GroupedFused, expected_m, lhs, rhs, rhs_scales, out, m, n, k,
                                num_groups, group_size, config, m_rows, torch::Tensor(),
                                expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks,
                                static_cast<int>(topk));
}

} // namespace deep_gemm
