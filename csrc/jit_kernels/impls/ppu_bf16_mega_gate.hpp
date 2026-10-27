#pragma once
#include <torch/python.h>
#include <deep_gemm/layout/mega_gate.cuh>
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
        bool use_aiu;
        LaunchArgs launch_args;
        mega_gate_layout::KernelArgs kernel_args;
        bool topk_only = false;
        bool cooperative = false;
        int topk_threads = 128;
        int block_tokens = 16;
        bool producer_activation = false;
        int features = -1;
        int warp_n = 16;
        int warp_m = 16;
    };

    static std::string generate_impl(const Args& args) {
        if (args.topk_only) {
            return fmt::format(R"(
#include <deep_gemm/impls/ppu_bf16_mega_gate.cuh>
extern "C" __global__ __launch_bounds__({})
void ppu_mega_gate_topk(deep_gemm::mega_gate_layout::KernelArgs args) {{
    args.scratch_token_base = 0;
    deep_gemm::ppu_mega_gate_topk_impl<{}, 1, {}, false, false, {}, {}, {}, 1, false, 8>(args, blockIdx.x * {});
}}
)", args.topk_threads, args.aligned_experts, args.num_topk, args.topk_threads,
                args.topk_threads / 32, args.kernel_args.routing.num_routed_experts != args.aligned_experts,
                args.topk_threads / 32);
        }
        return fmt::format(R"(
#include <deep_gemm/impls/ppu_bf16_mega_gate.cuh>
static_assert(sizeof(deep_gemm::mega_gate_layout::KernelArgs) == {}, "Mega Gate argument layout mismatch");
extern "C" __global__ __launch_bounds__({})
void ppu_bf16_mega_gate(deep_gemm::mega_gate_layout::KernelArgs args) {{
    deep_gemm::ppu_bf16_mega_gate_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>(args);
}}
)", sizeof(mega_gate_layout::KernelArgs), args.launch_args.block_dim.x, args.hidden, args.aligned_experts,
            args.num_split_k, args.num_topk,
            args.kernel_args.routing.to_physical_map != nullptr,
            args.kernel_args.routing.unmapped_topk_idx != nullptr, args.block_experts, args.block_k, args.stages, args.warp_k, args.use_aiu, args.block_tokens, args.producer_activation, args.features, args.warp_n, args.warp_m, args.cooperative);
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
#else
            cooperative.id = HG_LAUNCH_ATTRIBUTE_COOPERATIVE;
#endif
            cooperative.value.cooperative = 1;
            config.attrs = &cooperative;
            config.numAttrs = 1;
        }
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.kernel_args));
    }
};

static void ppu_bf16_mega_gate(const torch::Tensor& x, const torch::Tensor& weight,
        const std::optional<torch::Tensor>& bias, const std::optional<torch::Tensor>& image_bias,
        const std::optional<torch::Tensor>& image_token_mask, const std::optional<torch::Tensor>& mask,
        const std::optional<torch::Tensor>& fix_routing_mask, const std::optional<torch::Tensor>& force_random,
        const mega_gate_layout::RoutingArgs& routing, int tokens, int hidden, int experts, int topk) {
    using namespace mega_gate_layout;
    const bool use_aiu = get_ppu_arch() == AC_PPU0015 and
        ((reinterpret_cast<uintptr_t>(x.data_ptr()) | reinterpret_cast<uintptr_t>(weight.data_ptr())) & 127) == 0;
    const auto config = get_mega_gate_config(tokens, hidden, experts, device_runtime->get_num_sms(), use_aiu,
                                              bias.has_value() and !image_token_mask.has_value(), topk);
    const auto& [num_sms, m, n, k, wm, wn, stages, smem] = config.gemm;
    const int blocks = (tokens + m - 1) / m;
    const int padded_experts = config.aligned_experts, producers = config.producers;
    const bool producer_activation = get_env<int>("DG_MEGA_GATE_PRODUCER_ACTIVATION", 1) != 0;
    const bool persistent_groups = get_env<int>("DG_MEGA_GATE_GROUPED", 0) != 0 and
        use_aiu and config.num_split_k == 1 and producers > 1 and
        producers <= device_runtime->get_num_sms() and device_runtime->get_prop()->cooperativeLaunch;
    // Both cooperative paths launch one resident wave. Expert groups loop over
    // token tiles; small split-K grids still process one tile per CTA.
    const bool cooperative = persistent_groups or (use_aiu and producer_activation and config.num_split_k > 1 and
        device_runtime->get_prop()->cooperativeLaunch);
    const int launch_blocks = persistent_groups ?
        std::min(blocks, device_runtime->get_num_sms() / producers) * producers : blocks * producers;
    PPUMegaGateRuntime::Args args{
        hidden, padded_experts, topk, config.num_split_k, n, k, stages, config.warp_k, use_aiu,
        {dim3(launch_blocks), dim3(config.threads()), std::get<0>(smem)},
        {x.data_ptr(), weight.data_ptr(), bias ? bias->data_ptr<float>() : nullptr,
         image_bias ? image_bias->data_ptr<float>() : nullptr,
         image_token_mask ? image_token_mask->data_ptr<bool>() : nullptr,
         mask ? mask->data_ptr<bool>() : nullptr,
         fix_routing_mask ? fix_routing_mask->data_ptr<bool>() : nullptr,
         force_random ? force_random->data_ptr<bool>() : nullptr,
         routing, nullptr, nullptr, uint32_t(tokens)}};
    args.cooperative = cooperative;
    args.block_tokens = m;
    args.warp_n = wn;
    args.warp_m = wm;
    args.producer_activation = producer_activation;
    args.features = (mask ? 1 : 0) | (force_random ? 2 : 0) | (fix_routing_mask ? 4 : 0) |
        (bias ? 8 : 0) | (image_token_mask ? 16 : 0) | (routing.num_shared_experts ? 32 : 0);
    auto kernel = compiler->build("ppu_bf16_mega_gate", PPUMegaGateRuntime::generate(args));
    if (cooperative and !persistent_groups) {
        const int resident_blocks = PPUMegaGateRuntime::resident_blocks_per_cu(kernel->kernel, args.launch_args);
        // Grid sync requires every producer to be resident. Respect the caller's
        // CU budget while allowing multiple CTAs per CU when resources permit.
        if (int64_t(launch_blocks) > int64_t(resident_blocks) * device_runtime->get_num_sms()) {
            args.cooperative = false;
            kernel = compiler->build("ppu_bf16_mega_gate", PPUMegaGateRuntime::generate(args));
        }
    }
    auto scratch = torch::empty({producers == 1 ? 0 : int64_t(blocks) * m, config.num_split_k, padded_experts},
                                 x.options().dtype(torch::kFloat));
    // Scratch and fallback counters remain invocation-private under Graph replay
    // and concurrent streams. Persistent groups initialize their counters in
    // the kernel; the small cooperative split-K path needs no counters.
    const int counters_per_tile = producer_activation ? padded_experts / n + 1 : 1;
    auto barriers = persistent_groups ? torch::empty({launch_blocks / producers}, x.options().dtype(torch::kInt)) :
        producers == 1 or args.cooperative ? torch::empty({0}, x.options().dtype(torch::kInt)) :
        torch::zeros({int64_t(blocks) * counters_per_tile}, x.options().dtype(torch::kInt));
    args.kernel_args.scratch = scratch.data_ptr<float>();
    args.kernel_args.score_barriers = barriers.data_ptr<int>();
    PPUMegaGateRuntime::launch(kernel, args);
}
} // namespace deep_gemm
