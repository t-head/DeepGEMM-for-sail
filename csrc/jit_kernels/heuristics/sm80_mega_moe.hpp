#pragma once

#include <algorithm>
#include <array>
#include <numeric>
#include "common_int8.hpp"

namespace deep_gemm::mega_int8 {

struct MegaMoEStageConfig {
    int num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages;
    int smem_bytes;
};

struct MegaMoEStage1Config : MegaMoEStageConfig {
    int copy_blocks, k_tiles_per_flag;
};

struct MegaMoELaunchConfig {
    MegaMoEStage1Config l1;
    MegaMoEStageConfig l2;
};

struct MegaMoEHeuristicInput {
    int num_sms, num_ranks, num_experts_per_rank, num_max_tokens_per_rank;
    int hidden, intermediate_hidden, expected_m;
};

template <class Config>
static MegaMoEStageConfig make_stage_config(const Config& config) {
    const auto [num_sms, bm, bn, bk, wm, wn, stages, smem] = config;
    return {num_sms, bm, bn, bk, wm, wn, stages, std::get<0>(smem)};
}

enum class CopyLaunchMode { SharedGrid, SplitGrid };

// K and BK here count physical payload bytes, including packed FP4 storage.
// The caller converts its compute BK before selecting a publication stripe.
static int aligned_copy_ktpf(int k_bytes, int bk_bytes, int stripe_bytes) {
    DG_HOST_ASSERT(k_bytes > 0 && bk_bytes > 0 && k_bytes % bk_bytes == 0 && k_bytes % 256 == 0);
    const int alignment = 512 / std::gcd(bk_bytes, 512);
    const int minimum = std::max(ceil_div(stripe_bytes, bk_bytes), ceil_div(k_bytes / bk_bytes, 7));
    return ceil_div(minimum, alignment) * alignment;
}

static int initial_copy_stripe_bytes(int k_bytes) {
    int chunks = 1;
    while (chunks < ceil_div(k_bytes, 1024)) chunks *= 2;
    return 512 * chunks;
}

static int select_stage1_copy_blocks(
    int em, int n, int k_bytes, int groups, const MegaMoEStageConfig& c, int ktpf,
    CopyLaunchMode mode = CopyLaunchMode::SharedGrid, int copy_warps = 0) {
    if (em <= 1 && mode == CopyLaunchMode::SharedGrid) return 3;
    const int compute_warps = (c.block_m / c.warp_m) * (c.block_n / c.warp_n);
    const int warps = copy_warps > 0 ? copy_warps : compute_warps;
    DG_HOST_ASSERT(warps > 0 && compute_warps > 0);
    const int cta_slots = c.num_sms * std::min(4, 16 / compute_warps);
    const int m_tiles = ceil_div(em, c.block_m), warp_rows = m_tiles * c.warp_m;
    const int64_t copy_tasks = int64_t(groups) * m_tiles;
    const int64_t stripe_cost = int64_t(8) * (c.block_m / warps)
        * (warp_rows + std::min(warp_rows, em - 1)) / warp_rows + 12;
    const int64_t copy_cost = ceil_div<int64_t>(int64_t(em - 1) * ceil_div(k_bytes, 512), m_tiles * warps)
        + stripe_cost * ceil_div(k_bytes, c.block_k * ktpf);
    const int64_t gemm_cost = int64_t(c.warp_m) * c.warp_n * k_bytes / (16 * 1024);

    const auto score = [&](int ncb) -> int64_t {
        if (mode == CopyLaunchMode::SharedGrid && ncb >= c.num_sms)
            return std::numeric_limits<int64_t>::max();
        // Split-copy CTAs use a separate CU pool. They never reduce GEMM slots.
        const int compute_slots = cta_slots - (mode == CopyLaunchMode::SharedGrid ? ncb : 0);
        if (compute_slots <= 0) return std::numeric_limits<int64_t>::max();
        return std::max(ceil_div<int64_t>(copy_tasks, ncb) * copy_cost,
            ceil_div<int64_t>(copy_tasks * ceil_div(n, c.block_n), compute_slots) * gemm_cost);
    };
    // This is the common cost-model fallback. Datatype-specific measured rules
    // belong in that datatype's selector, with their measured shape scope.
    if (mode == CopyLaunchMode::SplitGrid)
        return std::min({2, 3, 4, 6}, [&](int a, int b) { return score(a) < score(b); });
    return std::min({3, 4, 6}, [&](int a, int b) { return score(a) < score(b); });
}

static MegaMoEStage1Config get_stage1_copy_config(
    int em, int n, int k_bytes, int groups, const MegaMoEStageConfig& config, int ktpf = 0,
    CopyLaunchMode mode = CopyLaunchMode::SharedGrid, int copy_warps = 0) {
    const int bk_bytes = config.block_k;
    DG_HOST_ASSERT(bk_bytes > 0 && k_bytes > 0 && k_bytes % bk_bytes == 0 && k_bytes % 256 == 0);
    if (ktpf == 0) {
        // Choose bytes first; BK is an MMA tile size, not a transfer target.
        const int stripe = std::max(initial_copy_stripe_bytes(k_bytes), std::min(k_bytes, 2048));
        // Preserve the existing FP4 shared-grid scheduling floor.
        ktpf = aligned_copy_ktpf(k_bytes, bk_bytes,
            mode == CopyLaunchMode::SharedGrid ? std::max(16 * bk_bytes, stripe) : stripe);
    }
    DG_HOST_ASSERT(ktpf > 0 && int64_t(ktpf) * bk_bytes % 512 == 0 &&
                   ceil_div(k_bytes / bk_bytes, ktpf) <= 7);
    return {config, select_stage1_copy_blocks(em, n, k_bytes, groups, config, ktpf, mode, copy_warps), ktpf};
}


// Split-copy uses an independent CU pool, so its bulk issue width need not
// follow the number of MMA warps. SM80 is validated with eight copy warps;
// retain the native compute-width default for other architectures.
static int get_int8_mega_moe_copy_threads(int compute_threads, int requested = 0) {
    if (requested != 0) return requested;
    return device_runtime->get_arch_pair() == std::make_pair(8, 0) ? 256 : compute_threads;
}

static MegaMoEStageConfig get_int8_mega_moe_compute_config(
    int em, int n, int k, int groups, int sms, GemmType gemm_type) {
    // Match the native Legacy A8W8 entry, including its runtime LUT.
    // FP4's packed K, MegaMoE tile overrides and scatter overrides do not apply.
    return make_stage_config(deep_gemm_int8::get_best_configs(
        em, n, k, groups, sms, gemm_type, 256,
        deep_gemm_lut::RuntimeLutKind::INT8));
}

static MegaMoELaunchConfig select_int8_mega_moe(
    const MegaMoEHeuristicInput& input, int copy_threads = 0, bool normal_quantization = false) {
    const auto gemm_type = normal_quantization ? GemmType::MGroupedNoPad : GemmType::MGroupedMasked;
    const auto up = get_int8_mega_moe_compute_config(
        input.expected_m, input.intermediate_hidden * 2, input.hidden,
        input.num_experts_per_rank, input.num_sms, gemm_type);
    auto down = get_int8_mega_moe_compute_config(
        input.expected_m, input.hidden, input.intermediate_hidden,
        input.num_experts_per_rank, input.num_sms, gemm_type);
    const int threads = (up.block_m / up.warp_m) * (up.block_n / up.warp_n) * 32;
    DG_HOST_ASSERT(threads >= 64 && input.hidden % up.block_k == 0);
    copy_threads = get_int8_mega_moe_copy_threads(threads, copy_threads);
    DG_HOST_ASSERT(copy_threads >= 64 && copy_threads % 32 == 0);
    // The current shared scatter implements 128/256-column vectors. Do not
    // silently replace a native INT8 tile with a tuned FP4 tile to satisfy it.
    DG_HOST_ASSERT(down.block_n == 128 || down.block_n == 256);
    down.smem_bytes = std::max(down.num_stages * (down.block_m + down.block_n) * down.block_k,
        2 * (down.block_m / down.warp_m) * 16 * down.block_n * int(sizeof(float)));
    auto copy = get_stage1_copy_config(input.expected_m, input.intermediate_hidden * 2,
        input.hidden, input.num_experts_per_rank, up, 0, CopyLaunchMode::SplitGrid, copy_threads / 32);
    return {copy, down};
}

} // namespace deep_gemm::mega_int8
