#pragma once

#include <memory>
#include <torch/python.h>
#include "../../jit/kernel_runtime.hpp"
#include "../../jit/compiler.hpp"
#include <deep_gemm/gemm_comm/reduce_scatter.cuh>
#include <deep_gemm/common/profiling_interface.cuh>
#include "../../utils/exception.hpp"
#include "../heuristics/common_gemm_comm.hpp"

namespace deep_gemm::gemm_comm {

template <class NativeRuntime> struct Fp8GemmRSArgs {
    typename NativeRuntime::LaunchInfo launch_info;
    LaunchArgs launch_args;
    typename NativeRuntime::GemmKernelParams kernel_params;
    int m = 0, n = 0, k = 0, rank = 0, world = 1;
};

template <class Args>
inline void launch_gemm_rs(const KernelHandle& kernel, const LaunchConfigHandle& config, const Args& args) {
    LaunchAttrHandle attr{};
#if HGGCRT_VERSION >= 12080 && defined(DG_JIT_USE_RUNTIME_API)
    attr.id = hggcLaunchAttributeCooperative;
    attr.val.cooperative = 1;
#else
    attr.id = HG_LAUNCH_ATTRIBUTE_COOPERATIVE;
    attr.value.cooperative = 1;
#endif
    auto cooperative = config;
    cooperative.numAttrs = 1;
    cooperative.attrs = &attr;
    DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, cooperative, args.kernel_params));
}

inline void validate_gemm_rs_output_alignment(const torch::Tensor& out) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
                "GEMM/RS output must be 16-byte aligned");
}

inline std::pair<int, int> validate_fp8_gemm_rs(const std::pair<torch::Tensor, torch::Tensor>& a,
                                                const std::pair<torch::Tensor, torch::Tensor>& b,
                                                const torch::Tensor& out, const torch::Tensor& state, int m, int rank,
                                                int world, bool channelwise) {
    TORCH_CHECK(b.first.dim() == 2, "RHS must be a matrix");
    TORCH_CHECK(b.first.size(0) <= std::numeric_limits<int>::max() &&
                    b.first.size(1) <= std::numeric_limits<int>::max(),
                "GEMM/RS dimensions exceed int32 capacity");
    int n = b.first.size(0), k = b.first.size(1);
    validate_gemm_rs_shape(m, n, k, world, !channelwise);
    TORCH_CHECK(rank >= 0 && rank < world, "invalid GEMM/RS rank");
    TORCH_CHECK(a.first.sizes() == std::vector<int64_t>({m, k}), "invalid LHS shape");
    const auto a_shape = channelwise ? std::vector<int64_t>{m, 1} : std::vector<int64_t>{m, k / 128};
    const auto b_shape = channelwise ? std::vector<int64_t>{n, 1} : std::vector<int64_t>{(n + 127) / 128, k / 128};
    TORCH_CHECK(a.second.sizes() == a_shape && b.second.sizes() == b_shape, "invalid FP8 scale shapes");
    TORCH_CHECK(a.first.scalar_type() == torch::kFloat8_e4m3fn && b.first.scalar_type() == torch::kFloat8_e4m3fn,
                "FP8 E4M3 tensors required");
    TORCH_CHECK(a.second.scalar_type() == torch::kFloat && b.second.scalar_type() == torch::kFloat,
                "FP32 scales required");
    TORCH_CHECK(out.is_contiguous() && out.scalar_type() == torch::kBFloat16 &&
                    out.sizes() == std::vector<int64_t>({m / world, n}),
                "invalid BF16 output shard");
    validate_gemm_rs_output_alignment(out);
    TORCH_CHECK(a.first.is_cuda() && a.first.is_contiguous() && b.first.is_contiguous() && b.second.is_contiguous() &&
                    (!channelwise || a.second.is_contiguous()),
                "contiguous device operands required");
    TORCH_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == torch::kLong &&
                    state.numel() == GemmRSStateWords,
                "invalid GEMM/RS state table");
    TORCH_CHECK(a.first.device() == b.first.device() && a.first.device() == a.second.device() &&
                    a.first.device() == b.second.device() && a.first.device() == out.device() &&
                    a.first.device() == state.device(),
                "GEMM/RS operands must use the same device");
    return {n, k};
}

inline void validate_gemm_rs_config(const ConfigTuple& config, bool channelwise) {
    const auto& [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    TORCH_CHECK(sms > 0 && sms <= get_num_sms(), "invalid GEMM/RS SM budget");
    validate_gemm_rs_tile({bm, bn, bk, wm, wn, stages}, channelwise);
    TORCH_CHECK(std::get<0>(smem) > 0 && std::get<0>(smem) <= gemm_rs_smem_capacity(),
                "invalid GEMM/RS shared-memory budget");
}

template <class Args> inline std::string gemm_rs_defines(const Args& args, int shared_packets = 0, int compact_min_m = 0) {
    const auto& info = args.launch_info;
    return fmt::format("#define DG_GEMM_RS_M {}\n#define DG_GEMM_RS_N {}\n#define DG_GEMM_RS_K {}\n"
                       "#define DG_GEMM_RS_RANK {}\n#define DG_GEMM_RS_WORLD {}\n"
                       "#define DG_GEMM_RS_COMPACT_MIN_M {}\n"
                       "#define DG_GEMM_RS_SHARED_PACKETS {}\n#define DG_GEMM_RS_SHARED_BUDGET {}\n",
                       args.m, args.n, args.k, args.rank, args.world, compact_min_m > 0 ? compact_min_m : info.warp_m,
                       shared_packets, args.launch_args.smem_size);
}

// Query the admitted grid of the already-cached kernel. A zero grid lets a
// caller reject an optional implementation without changing the native tile.
inline int get_gemm_rs_grid(const std::shared_ptr<KernelRuntime>& runtime, const LaunchArgs& args, int tiles) {
    int blocks_per_cu = 0;
    TORCH_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks_per_cu, runtime->kernel, args.block_dim.x, args.smem_size) == HGGC_SUCCESS,
                "failed to inspect GEMM/RS kernel occupancy");
    return int(std::min<int64_t>(int64_t(args.grid_dim.x) * blocks_per_cu, tiles));
}

// Launch or inspect the selected cached kernel without rebuilding it or
// querying occupancy again. Tensor addresses and streams remain per-call.
template <class Runtime, class Args>
inline auto inspect_or_launch_gemm_rs(const std::shared_ptr<KernelRuntime>& runtime, const Args& args, bool launch,
                                      int shared_packets = 0, int warp_k = 0, int num_sms = 0) {
    int registers = 0, local_bytes = 0;
    if (launch) {
        if (profiling_enabled()) {
            DgProfParam profile("GemmReduceScatter", Runtime::DataType);
            const auto& i = args.launch_info;
            profile.add_params("quant_type", Runtime::QuantType)
                .add_params("m", args.m).add_params("n", args.n).add_params("k", args.k)
                .add_params("tp", args.world).add_params("rank", args.rank)
                .add_params("configs", fmt::format("[{},{},{},{},{},{},{}]", num_sms,
                    i.block_m, i.block_n, i.block_k, i.warp_m, i.warp_n, i.num_stages));
            if (warp_k > 0)
                profile.add_params("warp_k", warp_k);
            auto profiling_guard = show_gemm_ut(profile);
            Runtime::launch(runtime, args);
        } else {
            Runtime::launch(runtime, args);
        }
    }
    const bool show_log = launch && get_env<int>("show_log", 0);
    if (!launch || show_log) {
        TORCH_CHECK(hgFuncGetAttribute(&registers, HG_FUNC_ATTRIBUTE_NUM_REGS, runtime->kernel) == HGGC_SUCCESS &&
                        hgFuncGetAttribute(&local_bytes, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, runtime->kernel) ==
                            HGGC_SUCCESS,
                    "failed to inspect GEMM/RS kernel resources");
    }
    if (show_log) {
        const auto& i = args.launch_info;
        printf("[GemmReduceScatter_%s_%s:]\n", Runtime::DataType, Runtime::QuantType);
        printf("rank:%d, TP:%d, problem:[%d, %d, %d], path:%s, reduction:in-kernel\n",
               args.rank, args.world, args.m, args.n, args.k, shared_packets > 0 ? "staged-send" : "direct-send");
        printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], num_stages:%d\n",
               i.block_m, i.block_n, i.block_k, i.warp_m, i.warp_n, warp_k > 0 ? warp_k : i.block_k, i.num_stages);
        printf("num_sms:%d, threadblock_count:%u, threads:%u, shared_packets:%d, SMSIZE:%d, vreg:%d, stack:%d\n",
               num_sms, args.launch_args.grid_dim.x, args.launch_args.block_dim.x, shared_packets,
               args.launch_args.smem_size, registers, local_bytes);
    }
    return std::make_tuple(registers, local_bytes, int(args.launch_args.grid_dim.x), args.launch_args.smem_size);
}

// Resolve and launch the cached blockwise or BF16 kernel.
template <class Runtime, class Args>
inline auto inspect_or_launch_gemm_rs(Args args, int tiles, bool launch, int shared_packets = 0, int warp_k = 0) {
    const auto& runtime = compiler->build_cached<Runtime>(args.launch_info.kernel_name, args);
    const int grid = get_gemm_rs_grid(runtime, args.launch_args, tiles);
    TORCH_CHECK(grid > 0, "GEMM/RS kernel cannot reside");
    const int num_sms = args.launch_args.grid_dim.x;
    args.launch_args.grid_dim.x = grid;
    return inspect_or_launch_gemm_rs<Runtime>(runtime, args, launch, shared_packets, warp_k, num_sms);
}

} // namespace deep_gemm::gemm_comm
