#pragma once

#include "mega_gate_810e.hpp"

namespace deep_gemm::mega_gate_810e {
static void finalize_mega_gate_resources(MegaGateConfig& config, int tokens, int hidden,
        int experts, int num_sms, bool deterministic) {
    namespace bf16 = deep_gemm_bf16_common;
    const auto& hw = PpuHwParams::instance();
    const int splits = config.num_split_k, warp_k = config.warp_k;
    auto& [sms, m, n, k, wm, wn, stages, smem] = config.gemm;
    DG_HOST_ASSERT(m >= 16 and m <= 512 and m % 16 == 0);
    DG_HOST_ASSERT(n >= 16 and n <= 512 and n % 16 == 0);
    DG_HOST_ASSERT(k == 64 or k == 128 or k == 256 or k == 512);
    DG_HOST_ASSERT(warp_k == 1 or warp_k == 2 or warp_k == 4 or warp_k == 8);
    DG_HOST_ASSERT(wm >= 16 and wm <= 256 and wm % 16 == 0 and m % wm == 0);
    DG_HOST_ASSERT(wn >= 16 and wn <= 128 and wn % 16 == 0 and n % wn == 0);
    // Each BF16 warp-K slice starts on a complete 128-byte swizzle tile
    // (or the whole K tile when it is narrower).
    DG_HOST_ASSERT(k % (warp_k * std::min(k, 64)) == 0);
    MegaGate810EArchSpec::validate_tile(config);
    DG_HOST_ASSERT(!deterministic or warp_k == 1);
    DG_HOST_ASSERT(stages >= 2 and stages <= 4);
    DG_HOST_ASSERT(hidden % (k * splits) == 0);
    const int alignment = std::lcm(n, 128);
    const int padded_experts = (experts + alignment - 1) / alignment * alignment;
    DG_HOST_ASSERT(padded_experts <= 512);
    const int producers = splits * (padded_experts / n);
    sms = std::min<int64_t>(num_sms, int64_t((tokens + m - 1) / m) * producers);

    // Reuse BF16's resource calculation, then account for FP32 score storage.
    smem = bf16::get_smem_config(stages, hidden / splits, m, n, k);
    config.aligned_experts = padded_experts;
    config.producers = producers;
    if (config.producers == 1) {
        auto& bytes = std::get<0>(std::get<7>(config.gemm));
        bytes = std::max(bytes, std::get<1>(config.gemm) * config.aligned_experts * 4);
    }
    const int values_per_thread = m * n / config.threads();
    DG_HOST_ASSERT(values_per_thread > 0 and values_per_thread % std::min(values_per_thread, 8) == 0);
    DG_HOST_ASSERT(config.threads() <= hw.max_threads_per_cta);
    DG_HOST_ASSERT(std::get<0>(smem) <= hw.tsm_per_cu);
}

static void print_mega_gate_config(const MegaGateConfig& config, int tokens, int hidden, int experts,
        int num_sms, int topk, bool allow_split_full_scores, bool deterministic) {
    if (!get_env<int>("DG_PRINT_CONFIGS", 0)) return;

    std::ostringstream message;
    message << "MegaGateConfig(num_tokens=" << tokens << ", hidden=" << hidden
            << ", num_routed_experts=" << experts << ", num_sms=" << num_sms
            << ", topk=" << topk << ", arch_path=" << MegaGate810EArchSpec::name
            << ", allow_split_full_scores=" << allow_split_full_scores
            << ", deterministic=" << deterministic << ", use_cutlass3=1" << "): " << config;
    // Include the result so changing a tuning override prints the new config too.
    static std::unordered_set<std::string> printed;
    static std::mutex mutex;
    const std::lock_guard<std::mutex> lock(mutex);
    if (printed.insert(message.str()).second)
        std::cout << message.str() << std::endl;
}

inline MegaGateConfig get_mega_gate_config(int tokens, int hidden, int experts, int num_sms,
                                           int topk, bool allow_split_full_scores = true) {
    DG_HOST_ASSERT(tokens > 0 and hidden > 0 and experts > 0 and experts <= 512 and num_sms > 0);
    DG_HOST_ASSERT(topk >= 0 and topk <= 32);
    const bool deterministic = heuristics_runtime->get_deterministic_algorithms();
    const MegaGateProblem problem{tokens, hidden, experts, num_sms, topk,
                                  allow_split_full_scores, deterministic};
    const auto task = MegaGate810EArchSpec::select_task(problem);
    DG_HOST_ASSERT(hidden % (64 * task.num_split_k) == 0);
    const auto base = MegaGate810EArchSpec::select_gemm(problem, task.num_split_k);
    auto config = MegaGate810EArchSpec::select_layout(problem, task, base);
    finalize_mega_gate_resources(config, tokens, hidden, experts, num_sms, deterministic);
    print_mega_gate_config(config, tokens, hidden, experts, num_sms, topk, allow_split_full_scores, deterministic);
    return config;
}

} // namespace deep_gemm::mega_gate_810e
