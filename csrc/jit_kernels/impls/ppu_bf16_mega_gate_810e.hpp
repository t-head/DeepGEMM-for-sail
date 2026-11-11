#pragma once
#include <torch/python.h>
#include <deep_gemm/layout/mega_gate.cuh>
#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/mega_gate_810e_config.hpp"
#include "ppu_bf16_mega_gate.hpp"

namespace deep_gemm {
class PPUMegaGate810ERuntime final : public LaunchRuntime<PPUMegaGate810ERuntime> {
public:
    struct Args {
        int hidden, aligned_experts, num_topk, num_split_k, block_experts, block_k, stages, warp_k;
        LaunchArgs launch_args;
        mega_gate_layout::KernelArgs kernel_args;
        bool cooperative = false;
        int block_tokens = 16;
        int features = -1;
        int warp_n = 16;
        int warp_m = 16;
        bool persistent_tokens = false;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/ppu_bf16_mega_gate_810e.cuh>
static_assert(sizeof(deep_gemm::mega_gate_layout::KernelArgs) == {}, "Mega Gate argument layout mismatch");
extern "C" __global__ __launch_bounds__({})
void ppu_bf16_mega_gate(deep_gemm::mega_gate_layout::KernelArgs args) {{
    deep_gemm::mega_gate_810e::ppu_bf16_mega_gate_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>(args);
}}
)", sizeof(mega_gate_layout::KernelArgs), args.launch_args.block_dim.x, args.hidden, args.aligned_experts,
            args.num_split_k, args.num_topk,
            args.kernel_args.routing.to_physical_map != nullptr,
            args.kernel_args.routing.unmapped_topk_idx != nullptr, args.block_experts, args.block_k, args.stages, args.warp_k, args.block_tokens, args.features, args.warp_n, args.warp_m, args.cooperative, args.persistent_tokens);
    }

    static void launch_impl(const KernelHandle& kernel, LaunchConfigHandle config, Args args) {
        PPUMegaGateRuntime::Args shared{};
        shared.cooperative = args.cooperative;
        shared.kernel_args = args.kernel_args;
        PPUMegaGateRuntime::launch_impl(kernel, config, shared);
    }
};

static void ppu_bf16_mega_gate_810e(const torch::Tensor& x, const torch::Tensor& weight,
        const std::optional<torch::Tensor>& bias, const std::optional<torch::Tensor>& image_bias,
        const std::optional<torch::Tensor>& image_token_mask, const std::optional<torch::Tensor>& mask,
        const std::optional<torch::Tensor>& fix_routing_mask, const std::optional<torch::Tensor>& force_random,
        const mega_gate_layout::RoutingArgs& routing, int tokens, int hidden, int experts, int topk,
        bool use_shared_as_routed, int num_shared_experts, const std::string& scoring_func) {
    using namespace mega_gate_layout;
    DG_HOST_ASSERT(get_ppu_arch() == AC_PPU0010);
    DG_HOST_ASSERT(((reinterpret_cast<uintptr_t>(x.data_ptr()) |
                     reinterpret_cast<uintptr_t>(weight.data_ptr())) & 127) == 0);
    const auto config = mega_gate_810e::get_mega_gate_config(tokens, hidden, experts, device_runtime->get_num_sms(), topk,
                                             bias.has_value() and !image_token_mask.has_value());
    const auto& [num_sms, m, n, k, wm, wn, stages, smem] = config.gemm;
    const int blocks = (tokens + m - 1) / m;
    const int padded_experts = config.aligned_experts, producers = config.producers;
    const bool cooperative = device_runtime->get_prop()->cooperativeLaunch and producers > 1;
    const int launch_blocks = blocks * producers;
    PPUMegaGate810ERuntime::Args args{
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
    args.block_tokens = m;
    args.warp_n = wn;
    args.warp_m = wm;
    args.features = (mask ? 1 : 0) | (force_random ? 2 : 0) | (fix_routing_mask ? 4 : 0) |
        (bias ? 8 : 0) | (image_token_mask ? 16 : 0) | (routing.num_shared_experts ? 32 : 0);
    // A non-cooperative device uses the full-row fallback below; do not
    // compile the unsupported multi-producer specialization first.
    std::shared_ptr<KernelRuntime> kernel;
    if (cooperative or producers == 1)
        kernel = compiler->build("ppu_bf16_mega_gate", PPUMegaGate810ERuntime::generate(args));
    if (cooperative) {
        const int resident_blocks = PPUMegaGateRuntime::resident_blocks_per_cu(kernel->kernel, args.launch_args);
        // Grid sync requires every producer to be resident. Respect the caller's
        // CU budget while allowing multiple CTAs per CU when resources permit.
        if (int64_t(launch_blocks) > int64_t(resident_blocks) * device_runtime->get_num_sms()) {
            // Reuse a resident producer grid across token tiles. Query this
            // specialization again because its loop can change occupancy.
            args.persistent_tokens = true;
            kernel = compiler->build("ppu_bf16_mega_gate", PPUMegaGate810ERuntime::generate(args));
            const int capacity = PPUMegaGateRuntime::resident_blocks_per_cu(
                kernel->kernel, args.launch_args) * device_runtime->get_num_sms();
            int resident_tiles = std::min(blocks, capacity / producers);
            // A deep K pipeline can consume a whole CU. Try half the K
            // staging only when it permits more resident producer groups.
            if (args.warp_k > 1 and args.block_k >= 128 and
                int64_t(args.launch_args.smem_size) * 2 > PpuHwParams::instance().tsm_per_cu) {
                auto shallower = args;
                shallower.block_k /= 2;
                shallower.warp_k = std::min(args.warp_k, shallower.block_k / 64);
                shallower.launch_args.block_dim.x = m / wm * (n / wn) * shallower.warp_k * 32;
                shallower.launch_args.smem_size = std::get<0>(deep_gemm_bf16_common::get_smem_config(
                    stages, hidden / args.num_split_k, m, n, shallower.block_k));
                auto shallower_kernel = compiler->build("ppu_bf16_mega_gate", PPUMegaGate810ERuntime::generate(shallower));
                const int shallower_capacity = PPUMegaGateRuntime::resident_blocks_per_cu(
                    shallower_kernel->kernel, shallower.launch_args) * device_runtime->get_num_sms();
                const int shallower_tiles = std::min(blocks, shallower_capacity / producers);
                if (shallower_tiles > resident_tiles) {
                    args = shallower;
                    kernel = shallower_kernel;
                    resident_tiles = shallower_tiles;
                }
            }
            if (resident_tiles > 0) args.launch_args.grid_dim.x = resident_tiles * producers;
            else args.cooperative = false;

        }
    }
    if (producers > 1 and !args.cooperative) {
        // If even one complete producer group cannot reside, let each CTA own
        // a full expert row. This also covers devices without cooperative launch.
        args.persistent_tokens = false;
        args.num_split_k = 1;
        args.block_tokens = 16;
        args.block_experts = padded_experts;
        args.block_k = 64;
        args.warp_k = 1;
        args.warp_m = args.warp_n = 16;
        args.stages = 2;
        const int bytes = std::max(16 * padded_experts * 4,
            std::get<0>(deep_gemm_bf16_common::get_smem_config(2, hidden, 16, padded_experts, 64)));
        args.launch_args = {dim3(ceil_div(tokens, 16)), dim3(padded_experts / 16 * 32), bytes};
        kernel = compiler->build("ppu_bf16_mega_gate", PPUMegaGate810ERuntime::generate(args));
    }
    const int final_producers = args.num_split_k * (padded_experts / args.block_experts);
    const int final_tiles = ceil_div(tokens, args.block_tokens);
    auto scratch = torch::empty({final_producers == 1 ? 0 : int64_t(final_tiles) * args.block_tokens,
                                 args.num_split_k, padded_experts}, x.options().dtype(torch::kFloat));
    // Scratch is invocation-private. The 810E cooperative path uses grid sync
    // and the full-row fallback uses CTA sync, so neither needs a reset kernel.
    DG_HOST_ASSERT(final_producers == 1 or args.cooperative);
    args.kernel_args.scratch = scratch.data_ptr<float>();
    args.kernel_args.score_barriers = nullptr;
    const auto profiling_range = profile_mega_gate(tokens, hidden, experts, topk, use_shared_as_routed,
        num_shared_experts, scoring_func, routing, bias, image_bias, image_token_mask, mask, fix_routing_mask, force_random);
    PPUMegaGate810ERuntime::launch(kernel, args);
}
} // namespace deep_gemm
