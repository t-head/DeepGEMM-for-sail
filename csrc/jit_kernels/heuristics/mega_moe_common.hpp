#pragma once

#include <cstdint>

namespace deep_gemm::mega_moe_common {

struct ReduceConfig {
    int column_splits = 1;
    int group_threads = 0;
};

// Share MXFP4's reduce policy across operand types. All inputs are host-known.
static constexpr ReduceConfig select_reduce_config(int tokens, int hidden, int cta_threads, int num_sms) {
    ReduceConfig config;
    const int parts = (hidden - 1) / (8 * cta_threads) + 1;
    if (tokens <= 128 && hidden % parts == 0 && (hidden / parts) % 256 == 0)
        config.column_splits = parts;
    constexpr int kGroupThreads = 128;
    // Keep at least one CTA's worth of token groups per configured CU.
    if (cta_threads > kGroupThreads && cta_threads % kGroupThreads == 0 &&
        int64_t(tokens) * config.column_splits >= int64_t(num_sms) * (cta_threads / kGroupThreads))
        config.group_threads = kGroupThreads;
    return config;
}

} // namespace deep_gemm::mega_moe_common
