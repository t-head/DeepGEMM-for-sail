#pragma once

#include <algorithm>
#include <torch/extension.h>
#include <deep_gemm/common/profiling_interface.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/mega_moe.hpp"

namespace deep_gemm::mega {

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
    return blocks;
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
        TORCH_CHECK(occupancy > 0, "MegaMoE kernel cannot keep a CTA resident");
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
        if (info.data_type) {
            DgProfParam profile;
            if (deep_gemm::profiling_enabled()) {
                profile.set_params(GemmType::MegaMoE, false, info.data_type, info.groups,
                    info.m, info.n, info.k, info.expected_m, nullptr, current_stream());
                profile.add_params("stage", args.kernel_name);
            }
            auto profiling_guard = deep_gemm::show_gemm_ut(profile);
            Runtime::launch(runtime, args);
        } else {
            Runtime::launch(runtime, args);
        }
        if (info.data_type && get_env<int>("show_log", 0)) {
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
            const auto& tile = info.tile;
            printf("[%s] groups:%d problem:[%d,%d,%d] num_sms:%d blocks_per_sm:%d grid:%u\n"
                   "tile:[%d,%d,%d] warp:[%d,%d] stages:%d smem:%d vreg:%d stack:%d\n",
                args.kernel_name, info.groups, info.m, info.n, info.k, tile.num_sms,
                occupancy, args.launch_args.grid_dim.x, tile.block_m, tile.block_n, tile.block_k,
                tile.warp_m, tile.warp_n, tile.num_stages, args.launch_args.smem_size, registers, stack_bytes);
        }
        return;
    }
}

} // namespace deep_gemm::mega
