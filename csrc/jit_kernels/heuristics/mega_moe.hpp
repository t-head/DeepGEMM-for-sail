#pragma once

#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include "common_fp4.hpp"
#include <deep_gemm/layout/dispatch_layout.cuh>

namespace deep_gemm::mega {

// Rank-local token counts describe a uniform-routing estimate, not measured
// expert loads. Cap it by the owning workspace's per-expert row capacity.
inline int expected_expert_rows(int tokens, int topk, int groups, int ranks, int capacity) {
    DG_HOST_ASSERT(tokens >= 0 && topk > 0 && groups > 0 && ranks > 0 && capacity >= 0);
    const int64_t owner_capacity = int64_t(ranks) * capacity;
    return std::max<int64_t>(1, std::min(owner_capacity, int64_t(tokens) * topk / groups + 1));
}

enum class MmaKind { BF16, MXFP8FP4, MXFP4 };

static MmaKind parse_mma_kind(const std::string& mma_type_str) {
    if (mma_type_str == "bf16xbf16")
        return MmaKind::BF16;
    if (mma_type_str == "fp8xfp4" or mma_type_str == "fp8xfp8")
        return MmaKind::MXFP8FP4;
    DG_HOST_ASSERT(mma_type_str == "mxf4xmxf4");
    return MmaKind::MXFP4;
}

struct MegaMoEPhaseConfig {
    int num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages;
    int smem_size;
};

struct MegaMoEL1Config : MegaMoEPhaseConfig {
    int copy_blocks, k_tiles_per_flag;
};

struct MegaMoEL2Config : MegaMoEPhaseConfig {};

struct MegaMoEConfig {
    MegaMoEL1Config l1;
    MegaMoEL2Config l2;
};

static MegaMoEPhaseConfig make_phase_config(const deep_gemm_fp4_common::ConfigResult& config) {
    const auto [num_sms, bm, bn, bk, wm, wn, stages, smem] = config;
    return {num_sms, bm, bn, bk, wm, wn, stages, std::get<0>(smem)};
}

struct CopyTile {
    int bm, bn, bk, wm, wn, stages;
};

// K and BK count packed FP4 bytes. Select a physical stripe before converting
// to K tiles; the launch contract also limits one M block to seven publications.
static int aligned_copy_ktpf(int k, int bk) {
    DG_HOST_ASSERT(k > 0 && bk > 0 && k % bk == 0 && k % 256 == 0);
    const int initial_stripe = 512 * next_power_of_two(ceil_div(k, 1024));
    const int stripe_bytes = std::max({16 * bk, initial_stripe, std::min(k, 2048)});
    const int alignment = 512 / std::gcd(bk, 512);
    const int minimum = std::max(ceil_div(stripe_bytes, bk), ceil_div(k / bk, 7));
    return ceil_div(minimum, alignment) * alignment;
}

static bool fused_tile_fits(const MegaMoEPhaseConfig& config, int num_ranks, int num_experts_per_rank) {
    const int bm = config.block_m, bn = config.block_n;
    const int threads = (bm / config.warp_m) * (bn / config.warp_n) * 32;
    return threads >= 64 && bn >= 64 &&
        num_experts_per_rank * (num_ranks + kExpertMetadataAuxFields) *
            static_cast<int>(sizeof(uint32_t)) <= config.smem_size;
}

static std::optional<CopyTile> select_copy_tile(
    int n, int k, int num_experts_per_rank, int num_ranks, int expected_m, const MegaMoEPhaseConfig& base) {
    // DPSK V4 Pro geometry (H7168 / I3072); expert-count scopes are matched below.
    if (n == 6144 && k == 3584) {
        const bool shared_tile = num_ranks == 16 && num_experts_per_rank == 24 && expected_m >= 115 && expected_m <= 117;
        const bool family = base.block_m == 128 && base.block_n == 256 && base.block_k == 64;
        if (expected_m >= 16 && family && !shared_tile) {
            constexpr CopyTile wide{128, 512, 128, 64, 64, 3};
            constexpr CopyTile large{256, 256, 128, 64, 64, 3};
            return expected_m <= 117 ? wide : large;
        }
    }
    return std::nullopt;
}

static int select_l1_copy_blocks(
    int expected_m, int n, int k, int num_experts_per_rank, const MegaMoEPhaseConfig& config, int k_tiles_per_flag) {
    const int warps = (config.block_m / config.warp_m) * (config.block_n / config.warp_n);
    const int cta_slots = config.num_sms * std::min(4, 16 / warps);
    const int m_tiles = ceil_div(expected_m, config.block_m), warp_rows = m_tiles * config.warp_m;
    const int64_t copy_tasks = int64_t(num_experts_per_rank) * m_tiles;
    const int64_t stripe_cost = int64_t(8) * (config.block_m / warps)
        * (warp_rows + std::min(warp_rows, expected_m - 1)) / warp_rows + 12;
    const int64_t copy_cost = ceil_div<int64_t>(int64_t(expected_m - 1) * ceil_div(k, 512), m_tiles * warps)
        + stripe_cost * ceil_div(k, config.block_k * k_tiles_per_flag);
    const int64_t gemm_cost = int64_t(config.warp_m) * config.warp_n * k / (16 * 1024);

    int best_ncb = 3;
    int64_t best_cost = INT64_MAX;
    // Visit the existing candidates first; NCB2 must strictly reduce cost.
    for (const int ncb : {3, 4, 6, 2}) {
        if (ncb >= config.num_sms) continue;
        const int64_t copy = ceil_div<int64_t>(copy_tasks, ncb) * copy_cost;
        const int64_t compute = ceil_div<int64_t>(
            copy_tasks * ceil_div(n, config.block_n), cta_slots - ncb) * gemm_cost;
        const bool low_headroom = copy * 3 > compute * 2;
        // For Qwen3.8 (H8192/I2048), keep NCB6 when NCB3 has insufficient supply headroom.
        if (n == 4096 && k == 4096 && config.num_sms > 6 && ncb == 3 && low_headroom) return 6;
        // Two copy CTAs need headroom for unmodeled peer-load and publication latency.
        if (ncb == 2 && low_headroom) continue;

        const int64_t cost = std::max(copy, compute);
        if (cost >= best_cost) continue;
        best_ncb = ncb;
        best_cost = cost;
    }
    return best_ncb;
}

static MegaMoEL1Config get_l1_config(
    int shape_m, int n, int k, int num_experts_per_rank, int num_sms, int num_ranks, int expected_m) {
    auto config = make_phase_config(deep_gemm_fp4_common::get_best_configs(
        shape_m, expected_m, n, k, num_experts_per_rank, num_sms, false, false,
        GemmType::MegaMoE, 256, 32));
    // Metadata preprocessing reuses the GEMM shared allocation before compute.
    config.smem_size = std::max(config.smem_size,
        num_experts_per_rank * (num_ranks + kExpertMetadataAuxFields) * static_cast<int>(sizeof(uint32_t)));
    DG_HOST_ASSERT(fused_tile_fits(config, num_ranks, num_experts_per_rank));

    const auto record = select_copy_tile(n, k, num_experts_per_rank, num_ranks, expected_m, config);
    if (record) {
        const auto& t = *record;
        const MegaMoEPhaseConfig candidate{config.num_sms, t.bm, t.bn, t.bk,
            t.wm, t.wn, t.stages, std::get<0>(deep_gemm_fp4_common::get_smem_config_fp4(
                t.stages, t.bm, t.bn, t.wm, t.wn, t.bk, n, false, false))};
        if (fused_tile_fits(candidate, num_ranks, num_experts_per_rank)) {
            config = candidate;
        }
    }
    const int k_tiles_per_flag = aligned_copy_ktpf(k, config.block_k);
    return {config, select_l1_copy_blocks(expected_m, n, k, num_experts_per_rank, config, k_tiles_per_flag), k_tiles_per_flag};
}

static MegaMoEL2Config get_l2_config(
    int shape_m, int expected_m, int n, int k, int num_experts_per_rank, int num_sms) {
    auto config = make_phase_config(deep_gemm_fp4_common::get_best_configs(
        shape_m, expected_m, n, k, num_experts_per_rank, num_sms, false, false,
        GemmType::MegaMoE, 256, 128));
    // BN128 uses ordinary scatter; BN256 uses remote bulk scatter.
    // The owner protocol lets each reader match its writer's layout.
    const int old_wn = config.warp_n;
    config.warp_n = std::max(config.warp_n, 32);
    if (config.warp_n != old_wn)
        config.smem_size = std::get<0>(deep_gemm_fp4_common::get_smem_config_fp4(
            config.num_stages, config.block_m, config.block_n, config.warp_m, config.warp_n,
            config.block_k, n, false, false));
    DG_HOST_ASSERT(config.block_n == 128 || config.block_n == 256);
    // Scatter needs two accumulator tiles in addition to the mainloop resource contract.
    config.smem_size = std::max(config.smem_size, 2 * (config.block_m / config.warp_m) *
                                16 * config.block_n * static_cast<int>(sizeof(float)));
    return {config};
}

static MegaMoEConfig get_mega_moe_config(
    int num_ranks, int num_experts_per_rank, int num_max_tokens_per_rank,
    int hidden, int intermediate_hidden, int expected_m, int num_sms) {
    return {
        get_l1_config(expected_m * num_experts_per_rank,
            intermediate_hidden * 2, hidden / 2, num_experts_per_rank,
            num_sms, num_ranks, expected_m),
        get_l2_config(num_ranks * num_max_tokens_per_rank, expected_m,
            hidden, intermediate_hidden / 2, num_experts_per_rank,
            num_sms)
    };
}

} // namespace deep_gemm::mega
