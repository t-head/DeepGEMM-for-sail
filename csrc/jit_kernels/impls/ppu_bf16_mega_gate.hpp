#pragma once
#include <torch/python.h>
#include <deep_gemm/layout/mega_gate.cuh>
#include <deep_gemm/common/profiling_interface.cuh>
#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/mega_gate.hpp"

namespace deep_gemm {
#if !(HGGCRT_VERSION >= 12080 and defined(DG_JIT_USE_RUNTIME_API))
DECL_LAZY_HGGC_DRIVER_FUNCTION(hgOccupancyMaxActiveBlocksPerMultiprocessor);
#endif

class PPUMegaGateRuntime final : public LaunchRuntime<PPUMegaGateRuntime> {
public:
    struct Args {
        int hidden, aligned_experts, num_topk, num_split_k, block_experts, block_k, stages, warp_k;
        LaunchArgs launch_args;
        mega_gate_layout::KernelArgs kernel_args;
        bool cooperative = false;
        bool persistent = false;
        int block_tokens = 16;
        int features = -1;
        int warp_n = 16;
        int warp_m = 16;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/ppu_bf16_mega_gate.cuh>
static_assert(sizeof(deep_gemm::mega_gate_layout::KernelArgs) == {}, "Mega Gate argument layout mismatch");
extern "C" __global__ __launch_bounds__({})
void ppu_bf16_mega_gate(deep_gemm::mega_gate_layout::KernelArgs args) {{
    deep_gemm::ppu_bf16_mega_gate_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>(args);
}}
)", sizeof(mega_gate_layout::KernelArgs), args.launch_args.block_dim.x, args.hidden, args.aligned_experts,
            args.num_split_k, args.num_topk,
            args.kernel_args.routing.to_physical_map != nullptr,
            args.kernel_args.routing.unmapped_topk_idx != nullptr, args.block_experts, args.block_k, args.stages, args.warp_k, args.block_tokens, args.features, args.warp_n, args.warp_m, args.cooperative, args.persistent);
    }

    static int resident_blocks_per_cu(const KernelHandle& kernel, const LaunchArgs& args) {
        // Query the cooperative specialization itself: routing and warp layout
        // can change its register/shared-memory footprint.
        int blocks = 0;
#if HGGCRT_VERSION >= 12080 and defined(DG_JIT_USE_RUNTIME_API)
        DG_HGGC_RUNTIME_CHECK(hggcFuncSetAttribute(
            kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, args.smem_size));
        DG_HGGC_RUNTIME_CHECK(hggcOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks, kernel, args.block_dim.x, args.smem_size));
#else
        DG_HGGC_DRIVER_CHECK(lazy_hgFuncSetAttribute(
            kernel, HG_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, args.smem_size));
        DG_HGGC_DRIVER_CHECK(lazy_hgOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks, kernel, args.block_dim.x, args.smem_size));
#endif
        return blocks;
    }

    static void launch_impl(const KernelHandle& kernel, LaunchConfigHandle config, Args args) {
        LaunchAttrHandle cooperative{};
        if (args.cooperative) {
#if HGGCRT_VERSION >= 12080 and defined(DG_JIT_USE_RUNTIME_API)
            cooperative.id = hggcLaunchAttributeCooperative;
            cooperative.val.cooperative = 1;
#else
            cooperative.id = HG_LAUNCH_ATTRIBUTE_COOPERATIVE;
            cooperative.value.cooperative = 1;
#endif
            config.attrs = &cooperative;
            config.numAttrs = 1;
        }
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_args));
    }
};

static HgtxRangeGuard profile_mega_gate(
        int tokens, int hidden, int experts, int topk, bool use_shared_as_routed, int num_shared_experts,
        const std::string& scoring_func, const mega_gate_layout::RoutingArgs& routing,
        const std::optional<torch::Tensor>& bias, const std::optional<torch::Tensor>& image_bias,
        const std::optional<torch::Tensor>& image_token_mask, const std::optional<torch::Tensor>& mask,
        const std::optional<torch::Tensor>& fix_routing_mask, const std::optional<torch::Tensor>& force_random) {
    if (not profiling_enabled())
        return HgtxRangeGuard{};
    DgProfParam params;
    params.set_mega_gate_params(tokens, hidden, experts, topk, use_shared_as_routed, num_shared_experts,
                                routing.routed_scaling_factor, static_cast<int>(routing.rank_idx), current_stream());
    params.add_params("with_bias", bias.has_value());
    params.add_params("with_physical_map", routing.to_physical_map != nullptr);
    params.add_params("with_image_bias", image_bias.has_value());
    params.add_params("with_mask", mask.has_value());
    params.add_params("with_unmapped_topk_idx", routing.unmapped_topk_idx != nullptr);
    params.add_params("scoring_func", scoring_func);
    params.add_params("with_fix_routing_mask", fix_routing_mask.has_value());
    params.add_params("with_force_random", force_random.has_value());
    return show_gemm_ut(params);
}

static void ppu_bf16_mega_gate(const torch::Tensor& x, const torch::Tensor& weight,
        const std::optional<torch::Tensor>& bias, const std::optional<torch::Tensor>& image_bias,
        const std::optional<torch::Tensor>& image_token_mask, const std::optional<torch::Tensor>& mask,
        const std::optional<torch::Tensor>& fix_routing_mask, const std::optional<torch::Tensor>& force_random,
        const mega_gate_layout::RoutingArgs& routing, int tokens, int hidden, int experts, int topk,
        bool use_shared_as_routed, int num_shared_experts, const std::string& scoring_func, bool single_producer = false) {
    using namespace mega_gate_layout;
    DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16u == 0u);
    DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16u == 0u);
    DG_HOST_ASSERT(get_ppu_arch() == AC_PPU0010 or get_ppu_arch() == AC_PPU0015);
    const auto config = [&]() -> MegaGateConfig {
        if (get_ppu_arch() == AC_PPU0010 and !single_producer) {
            return mega_gate::get_sm80_mega_gate_config(tokens, hidden, experts,
                device_runtime->get_num_sms(), topk, bias.has_value() and !image_token_mask.has_value());
        }
        return mega_gate::get_sm89_mega_gate_config(tokens, hidden, experts, device_runtime->get_num_sms(),
                                    bias.has_value() and !image_token_mask.has_value(), topk, single_producer);
    }();
    const auto& [num_sms, m, n, k, wm, wn, stages, smem] = config.gemm;
    const int blocks = (tokens + m - 1) / m;
    const int padded_experts = config.aligned_experts, producers = config.producers;
    const bool cooperative = producers > 1 and device_runtime->get_prop()->cooperativeLaunch;
    const int launch_blocks = blocks * producers;
    PPUMegaGateRuntime::Args args{
        hidden, padded_experts, topk, config.num_split_k, n, k, stages, config.warp_k,
        {dim3(launch_blocks), dim3(config.threads()), std::get<0>(smem)},
        {x.data_ptr(), weight.data_ptr(), bias ? bias->data_ptr<float>() : nullptr,
         image_bias ? image_bias->data_ptr<float>() : nullptr,
         image_token_mask ? image_token_mask->data_ptr<bool>() : nullptr,
         mask ? mask->data_ptr<bool>() : nullptr,
         fix_routing_mask ? fix_routing_mask->data_ptr<bool>() : nullptr,
         force_random ? force_random->data_ptr<bool>() : nullptr,
         routing, nullptr, nullptr, uint32_t(tokens)}};
    args.cooperative = cooperative;
    args.persistent = cooperative and config.num_split_k == 1 and producers > 1;
    args.block_tokens = m;
    args.warp_n = wn;
    args.warp_m = wm;
    args.features = (mask ? kTokenMask : 0) | (force_random ? kForceRandom : 0) |
        (fix_routing_mask ? kFixRouting : 0) | (bias ? kExpertBias : 0) |
        (image_token_mask ? kImageTokenMask : 0) | (routing.num_shared_experts ? kSharedExperts : 0) |
        (scoring_func == "sigmoid" ? kSigmoidScore : 0);
    // Devices or CU budgets that cannot hold a producer group use one CTA per token tile.
    if (producers > 1 and !cooperative) {
        ppu_bf16_mega_gate(x, weight, bias, image_bias, image_token_mask, mask, fix_routing_mask, force_random,
                          routing, tokens, hidden, experts, topk, use_shared_as_routed, num_shared_experts, scoring_func, true);
        return;
    }
    auto kernel = compiler->build("ppu_bf16_mega_gate", PPUMegaGateRuntime::generate(args));
    if (producers > 1) {
        const int resident_blocks = PPUMegaGateRuntime::resident_blocks_per_cu(kernel->kernel, args.launch_args);
        // Grid sync requires every producer to be resident. Respect the caller's
        // CU budget while allowing multiple CTAs per CU when resources permit.
        if (args.persistent or int64_t(launch_blocks) > int64_t(resident_blocks) * device_runtime->get_num_sms()) {
            args.persistent = true;
            kernel = compiler->build("ppu_bf16_mega_gate", PPUMegaGateRuntime::generate(args));
            const int capacity = PPUMegaGateRuntime::resident_blocks_per_cu(kernel->kernel, args.launch_args) *
                                 device_runtime->get_num_sms();
            const int resident_groups = std::min(blocks, capacity / producers);
            if (resident_groups == 0) {
                ppu_bf16_mega_gate(x, weight, bias, image_bias, image_token_mask, mask, fix_routing_mask, force_random,
                                  routing, tokens, hidden, experts, topk, use_shared_as_routed, num_shared_experts, scoring_func, true);
                return;
            }
            args.launch_args.grid_dim = dim3(resident_groups * producers);
        }
    }
    auto scratch = torch::empty({producers == 1 ? 0 : int64_t(blocks) * m, config.num_split_k, padded_experts},
                                 x.options().dtype(torch::kFloat));
    // Only resident groups need counters. The fused kernel initializes them
    // before a grid barrier, so every call and Graph replay starts safely.
    auto barriers = torch::empty({args.persistent ? int64_t(args.launch_args.grid_dim.x) / producers : 0},
                                 x.options().dtype(torch::kInt));
    args.kernel_args.scratch = scratch.data_ptr<float>();
    args.kernel_args.score_barriers = barriers.data_ptr<int>();
    const auto profiling_range = profile_mega_gate(tokens, hidden, experts, topk, use_shared_as_routed,
        num_shared_experts, scoring_func, routing, bias, image_bias, image_token_mask, mask, fix_routing_mask, force_random);
    PPUMegaGateRuntime::launch(kernel, args);
}
} // namespace deep_gemm
