#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <type_traits>
#include <torch/extension.h>
#include <deep_gemm/common/profiling_interface.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/mega_moe_common.cuh>

#include "../../jit/compiler.hpp"
#include "../../utils/layout.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/mega_moe.hpp"

namespace deep_gemm::mega {

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

static_assert(sizeof(layout::SymBuffer<>) == 528, "MegaMoE symmetric buffer ABI changed");

// Large expert groups consume the task list published by GEMM1; smaller
// groups let the same scheduler traverse per-expert row counts directly.
inline int32_t* scatter_layout(int groups, int32_t* rows, int32_t* block_info) {
    return groups < 128 ? rows : block_info;
}

inline void* device_pointer(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "MegaMoE expects CUDA device tensors");
    return tensor.data_ptr();
}

template <typename T> inline T* device_pointer(const torch::Tensor& tensor) {
    return reinterpret_cast<T*>(device_pointer(tensor));
}

inline layout::SymBuffer<> make_sym_buffer(const std::vector<int64_t>& pointers, int rank_idx, int num_ranks) {
    TORCH_CHECK(int(pointers.size()) == num_ranks && num_ranks > 0 && num_ranks <= layout::kNumMaxRanks &&
                rank_idx >= 0 && rank_idx < num_ranks, "invalid symmetric pointer table or rank");
    for (const auto pointer : pointers)
        TORCH_CHECK(pointer != 0 && pointer % 128 == 0, "symmetric buffer bases must be 128-byte aligned");
    return layout::SymBuffer<>(pointers, rank_idx);
}

inline layout::SymBuffer<> make_sym_buffer(const torch::Tensor& buffer,
        const std::vector<int64_t>& pointers, int rank) {
    const auto transport = make_sym_buffer(pointers, rank, pointers.size());
    TORCH_CHECK(transport.base == reinterpret_cast<int64_t>(buffer.data_ptr()), "Invalid local workspace pointer");
    return transport;
}

inline int get_kernel_occupancy(const std::shared_ptr<KernelRuntime>& runtime,
                                int threads, int dynamic_smem = 0) {
    // Occupancy must use the same dynamic shared-memory limit as launch.
    construct_launch_config(runtime->kernel, current_stream(), dynamic_smem, dim3(1), dim3(threads));
    int blocks = 0;
#if HGGCRT_VERSION >= 12080 && defined(DG_JIT_USE_RUNTIME_API)
    DG_HGGC_UNIFIED_CHECK(hggcOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, runtime->kernel, threads, dynamic_smem));
#else
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, runtime->kernel, threads, dynamic_smem));
#endif
    DG_HOST_ASSERT(blocks > 0);
    return blocks;
}

template <bool Int8>
class MegaMoEPreDispatchRuntime final : public LaunchRuntime<MegaMoEPreDispatchRuntime<Int8>> {
public:
    static constexpr const char* kName = "mega_moe_pre_dispatch";
    struct Args {
        int hidden, num_topk, num_max_tokens_per_rank, num_ranks, num_experts, intermediate_hidden;
        LaunchArgs launch_args;
        const void* hidden_states;
        const torch::Tensor& topk_idx;
        const float* topk_weights;
        uint32_t num_tokens;
        layout::SymBuffer<> sym_buffer;
        layout::MegaMoEStaging staging;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/mega_moe_pre_dispatch.cuh>
using TopkIdx = {};
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::mega_moe_pre_dispatch<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, TopkIdx, {}>);
}}
)", args.topk_idx.scalar_type() == torch::kInt32 ? "int32_t" : "int64_t", Int8,
    args.hidden, args.num_topk, args.num_max_tokens_per_rank, args.num_ranks, args.num_experts, args.intermediate_hidden,
    args.staging.scales, args.staging.counts, args.staging.tokens, is_bf16x8_aligned(args.hidden_states));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.hidden_states, args.topk_idx.data_ptr(),
                                            args.topk_weights, args.num_tokens, args.sym_buffer));
    }
};

template <bool Int8>
static void mega_moe_pre_dispatch_impl(
    const torch::Tensor& hidden_states, const torch::Tensor& topk_idx, const torch::Tensor& topk_weights,
    const std::vector<int64_t>& sym_buffer_ptrs,
    int num_ranks, int num_experts, int num_max_tokens_per_rank, int rank_idx,
    int hidden, int intermediate_hidden, int num_topk, const layout::MegaMoEStaging& staging) {
    using Runtime = MegaMoEPreDispatchRuntime<Int8>;
    typename Runtime::Args args{hidden, num_topk, num_max_tokens_per_rank, num_ranks, num_experts, intermediate_hidden,
        {dim3(1), dim3(256), 0}, hidden_states.data_ptr(), topk_idx, topk_weights.data_ptr<float>(),
        static_cast<uint32_t>(hidden_states.size(0)), layout::SymBuffer<>(sym_buffer_ptrs, rank_idx), staging};
    const auto runtime = compiler->build(Runtime::kName, Runtime::generate(args));
    const int count_blocks = num_experts / num_ranks;
    const int resident = get_kernel_occupancy(runtime, 256, 0) * device_runtime->get_prop()->multiProcessorCount;
    const int token_blocks = std::max(1, std::min(static_cast<int>(args.num_tokens), std::max(1, resident - count_blocks)));
    args.launch_args.grid_dim.x = count_blocks + token_blocks;
    Runtime::launch(runtime, args);
}

template <class Scale, class BufferLayout>
static sched::MegaMoESchedulerArgumentsT<Scale> make_mega_moe_scheduler(
    const BufferLayout& buffers, void* base, uint32_t shape_m) {
    sched::MegaMoESchedulerArgumentsT<Scale> scheduler;
    scheduler.shape_m = shape_m;
    scheduler.copy_grouped_layout = buffers.grouped_layout.template data<int>(base);
    scheduler.grouped_layout = scheduler.copy_grouped_layout + (1 + buffers.blocks) * 4;
    scheduler.staging = buffers.staging;
    scheduler.expert_rank_prefix = buffers.expert_rank_prefix.template data<uint32_t>(base);
    scheduler.expert_base = buffers.expert_base.template data<uint32_t>(base);
    scheduler.local_data_buf = buffers.local_data.data(base);
    scheduler.local_sfa_buf = buffers.local_sfa.template data<Scale>(base);
    scheduler.copy_ready_flags = buffers.copy_ready.template data<uint32_t>(base);
    scheduler.dispatch_data = buffers.dispatch_data.data(base);
    scheduler.dispatch_token_topk = buffers.staging_tokens[0].template data<uint32_t>(base);
    scheduler.src_info = buffers.src_info.template data<layout::TokenSrcMetadata>(base);
    return scheduler;
}

struct MegaMoEGemmProfile {
    const char* data_type = nullptr;
    int groups = 0, m = 0, n = 0, k = 0, expected_m = 0;
    MegaMoEPhaseConfig tile{};
};

inline MegaMoEGemmProfile gemm_profile(const char* data_type, int rows, int expected_m, const Gemm1Config& config) {
    return {data_type, config.dispatch.num_experts_per_rank, rows, config.n, config.k, expected_m, config.tile};
}

inline MegaMoEGemmProfile gemm_profile(const char* data_type, int rows, int expected_m, const ScatterConfig& config) {
    return {data_type, config.num_groups, rows, config.n, config.k, expected_m,
            {config.num_sms, config.block_m, config.block_n, config.block_k,
             config.warp_m, config.warp_n, config.num_stages, config.smem_size}};
}

template <typename Runtime>
inline void launch_profiled(const std::shared_ptr<KernelRuntime>& runtime,
                            const typename Runtime::Args& args, const MegaMoEGemmProfile& info) {
    if (!info.data_type) {
        Runtime::launch(runtime, args);
        return;
    }
    DgProfParam profile;
    if (deep_gemm::profiling_enabled()) {
        profile.set_params(GemmType::MegaMoE, false, info.data_type, info.groups,
            info.m, info.n, info.k, info.expected_m, nullptr, current_stream());
        profile.add_params("stage", args.kernel_name);
    }
    auto profiling_guard = deep_gemm::show_gemm_ut(profile);
    Runtime::launch(runtime, args);
}

inline void log_gemm(const std::shared_ptr<KernelRuntime>& runtime, const char* name,
                     const LaunchArgs& launch, const MegaMoEGemmProfile& info, int blocks_per_sm) {
    if (!info.data_type || !get_env<int>("show_log", 0))
        return;
    const auto [registers, stack_bytes] = get_kernel_resources(runtime);
    const auto& tile = info.tile;
    printf("[%s] groups:%d problem:[%d,%d,%d] num_sms:%d blocks_per_sm:%d grid:%u\n"
           "tile:[%d,%d,%d] warp:[%d,%d] stages:%d smem:%d vreg:%d stack:%d\n",
        name, info.groups, info.m, info.n, info.k, tile.num_sms,
        blocks_per_sm, launch.grid_dim.x, tile.block_m, tile.block_n, tile.block_k,
        tile.warp_m, tile.warp_n, tile.num_stages, launch.smem_size, registers, stack_bytes);
}

// Cooperative kernels compile their grid size into the barrier protocol.
// Non-cooperative kernels may expand the grid by their measured occupancy.
// OccupancyLimit is zero for a fixed grid and negative for uncapped occupancy.
template <typename Runtime, bool Cooperative = false, int OccupancyLimit = 0>
inline void launch_mega_moe_kernel(typename Runtime::Args args, const MegaMoEGemmProfile& info = {}) {
    TORCH_CHECK(args.launch_args.grid_dim.x > 0 && args.launch_args.block_dim.x > 0 &&
                    args.launch_args.block_dim.x <= 1024,
                "invalid MegaMoE launch geometry");
    TORCH_CHECK(args.launch_args.smem_size >= 0 && args.launch_args.smem_size <= 256 * 1024,
                "MegaMoE kernel exceeds shared-memory capacity");
    for (;;) {
        const auto runtime = compiler->build(args.kernel_name, Runtime::generate(args));
        const int occupancy = get_kernel_occupancy(runtime, args.launch_args.block_dim.x, args.launch_args.smem_size);
        if constexpr (Cooperative) {
            const int resident = device_runtime->get_prop()->multiProcessorCount * occupancy;
            if (args.launch_args.grid_dim.x > static_cast<unsigned>(resident)) {
                args.launch_args.grid_dim.x = resident;
                args.launch_info.num_ctas = resident;
                continue;
            }
        } else if constexpr (OccupancyLimit != 0) {
            args.launch_args.grid_dim.x *= OccupancyLimit < 0 ? occupancy : std::min(OccupancyLimit, occupancy);
        }
        launch_profiled<Runtime>(runtime, args, info);
        log_gemm(runtime, args.kernel_name, args.launch_args, info, occupancy);
        return;
    }
}

} // namespace deep_gemm::mega
