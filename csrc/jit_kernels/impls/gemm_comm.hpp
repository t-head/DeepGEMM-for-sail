#pragma once

#include <torch/python.h>
#include "../../jit/kernel_runtime.hpp"
#include "../../jit/compiler.hpp"
#include <deep_gemm/gemm_comm/collectives.cuh>
#include "../../utils/exception.hpp"
#include "../heuristics/common_gemm_comm.hpp"

namespace deep_gemm::gemm_comm {

template <class NativeRuntime>
struct Fp8GemmRSArgs {
    typename NativeRuntime::LaunchInfo launch_info;
    LaunchArgs launch_args;
    typename NativeRuntime::GemmKernelParams kernel_params;
    bool direct_sfb = false;
    bool specialize = false;
    int m = 0, rank = 0, world = 1;
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

inline std::pair<int, int> validate_fp8_gemm_rs(const std::pair<torch::Tensor, torch::Tensor>& a,
                                                const std::pair<torch::Tensor, torch::Tensor>& b, const torch::Tensor& out,
                                                const torch::Tensor& state, int m, int rank, int world, bool channelwise) {
    TORCH_CHECK(b.first.dim() == 2, "RHS must be a matrix");
    int n = b.first.size(0), k = b.first.size(1);
    validate_gemm_rs_shape(m, n, k, world, channelwise);
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

} // namespace deep_gemm::gemm_comm
