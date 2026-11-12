#pragma once

#include <cstring>
#include <cstdlib>
#include <cstdint>

namespace deep_gemm_bf16_common {

struct MemoryExecutionPolicy {
    int b_cache_policy = 1;
    bool nonpersistent = false;
};

// Resolve execution policies after tile selection. A single M tile does not
// reuse B across output tiles, so use the streaming policy regardless of B size.
inline MemoryExecutionPolicy get_memory_execution_policy(
    int m, int n, int block_m, int block_n, int scheduled_sms,
    int blocks_per_cu, int num_sms, int physical_sms) {
    const int64_t tiles_m = (int64_t(m) + block_m - 1) / block_m;
    const int64_t tiles_n = (int64_t(n) + block_n - 1) / block_n;
    const int64_t tiles = tiles_m * tiles_n;
    const int64_t capacity = int64_t(scheduled_sms) * blocks_per_cu;
    MemoryExecutionPolicy policy;
    if (tiles_m == 1)
        policy.b_cache_policy = 3;

    // One-tile CTAs may spread over the whole device. A restricted CU budget
    // is respected only when the total CTA count itself fits that budget.
    const bool fits_budget = num_sms >= physical_sms || tiles <= num_sms;
    // One-tile CTAs remove scheduler-loop overhead through the first wave.
    // Multiple-M problems keep persistent traversal beyond that wave;
    // single-M problems may extend to a second wave.
    policy.nonpersistent = tiles > 0 && capacity > 0 && fits_budget &&
                           (tiles <= capacity || (tiles_m == 1 && tiles <= 2 * capacity));
    return policy;
}

inline bool memory_policies_enabled() {
    const char* enabled = std::getenv("DG_BF16_MEMORY_TUNING");
    return !enabled || std::strcmp(enabled, "0") != 0;
}

} // namespace deep_gemm_bf16_common
