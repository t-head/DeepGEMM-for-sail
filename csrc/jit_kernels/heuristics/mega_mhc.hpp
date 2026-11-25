#pragma once

#include <algorithm>
#include <climits>
#include <initializer_list>
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/math.hpp"

namespace deep_gemm {

enum class MegaMhcPhase { Main, Pre, Finish };

struct MegaMhcConfig {
    int hidden, splits, iterations, tile_m, task_splits;
    bool full_rows, single, local, tail_rows, tail_tokens, vector_coefficients;
    bool store_bf16, store_fp8;
    int shared_sf_block_m;
    bool per_token_fp8, shifted;
    int stream_chunk;
    bool compact_k;
    bool tc01 = false;
    MegaMhcPhase phase = MegaMhcPhase::Main;
    bool tc01_round_b = false;
    int tc01_pre_chunk = 1024;
};

struct MegaMhcPlan {
    MegaMhcConfig config;
    LaunchArgs launch_args;
    int num_sms;
    int64_t num_counters, num_partials;
    LaunchArgs pre_launch{}, finish_launch{};
};

static constexpr int kMegaMhcGeneralTileM = 64;

static int select_num_splits(int tokens, int iterations, bool shifted, int general_slots, int ring_slots) {
    if (shifted && iterations == 20 && tokens >= 128 && tokens <= 32768) {
        const int groups = ceil_div(tokens, 32);
        if (tokens >= 1024 && tokens % 32 == 0)
            return groups * 9 <= ring_slots ? 18 : 16;
        for (int splits : {80, 40, 32, 16, 12, 10, 8})
            if (groups * (splits / 2) <= ring_slots) return splits;
        int selected = 16, best = INT_MAX;
        for (int splits : {8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 40}) {
            const int waves = ceil_div(groups * (splits / 2), ring_slots);
            const int cost = 3 * waves * ceil_div(160, splits) + splits;
            if (cost <= best) { best = cost; selected = splits; }
        }
        const int s16_waves = ceil_div(groups * 8, ring_slots);
        if (groups > ring_slots && 30 * s16_waves + 16 <= best + 2) selected = 16;
        return selected;
    }
    const int blocks = ceil_div(tokens, kMegaMhcGeneralTileM);
    // Prefer direct dispatch within one wave, capped at 80 tasks to limit partials.
    constexpr int kDirectTaskBudget = 80;
    const int direct_split_budget = std::min(general_slots, kDirectTaskBudget) / blocks;
    for (const int splits : {80, 40})
        if (splits <= direct_split_budget) return splits;
    return std::clamp(5 * general_slots / blocks, 1, 16);
}

static int select_normal_splits(int tokens, int hidden, bool ring, int fallback, int general_slots, int ring_slots) {
    if (tokens <= 128 || tokens > 32768) return fallback;
    if (!ring) {
        const int blocks = ceil_div(tokens, kMegaMhcGeneralTileM);
        for (int splits : {40, 32, 24})
            if (blocks * splits <= general_slots) return splits;
        return fallback;
    }

    const int groups = ceil_div(tokens, 32);
    const int k_tiles = hidden / 64;
    const int max_local = std::min(hidden == 4096 ? k_tiles : 72, 2 * (ring_slots / groups));
    if (max_local >= 8) {
        const int paired_k_tiles = ceil_div(2 * k_tiles, max_local);
        return std::max(8, 2 * ceil_div(k_tiles, paired_k_tiles));
    }

    int selected = std::max(8, fallback / 2 * 2);
    if (groups >= 2 * ring_slots) return selected;
    int best = INT_MAX;
    for (int splits = 8; splits <= 32; splits += 2) {
        const int waves = ceil_div(groups * (splits / 2), ring_slots);
        const int cost = 8 * waves * ceil_div(2 * k_tiles, splits) + splits;
        if (cost < best) {
            best = cost;
            selected = splits;
        }
    }
    return selected;
}

static MegaMhcPlan select_mega_mhc_plan(int t, int h, int iterations, int sms,
                                       bool tc01, bool store_bf16, bool store_fp8,
                                       int shared_sf_block_m, bool per_token_fp8,
                                       int physical_sms, bool shifted) {
    DG_HOST_ASSERT(t > 0 and sms > 0);
    if (tc01) {
        int splits = 80;
        if (t >= 128) {
            int64_t best = INT64_MAX;
            for (int candidate : {4, 5, 6, 7, 8, 10, 12, 16, 20, 32, 40, 80}) {
                const int64_t waves = ceil_div<int64_t>(ceil_div(t, 64) * int64_t(candidate), 8 * sms);
                const int64_t cost = waves * (64 * ceil_div(80, candidate) + 20) + candidate;
                if (cost < best) { best = cost; splits = candidate; }
            }
        }
        const int tile_m = splits == 16 && t < 1024 && ceil_div(t, 16) <= sms ? 16 : 32;
        MegaMhcPlan plan{
            {h, splits, iterations, tile_m, 2, t % 64 == 0,
             false, false, t % tile_m != 0, t % (tile_m / 8) != 0, t != 1248,
             store_bf16, store_fp8, shared_sf_block_m, per_token_fp8, shifted, 0, false},
            {dim3(ceil_div(t, 64), splits), dim3(128), 0}, sms, 0,
            ceil_div<int64_t>(t, 64) * splits * 64 * 26};
        auto& c = plan.config;
        c.tc01 = true;
        c.tc01_round_b = ceil_div(t, 64) >= 3;
        if (c.tc01_round_b) plan.num_partials += 24 * 4 * h;
        c.tc01_pre_chunk = t > 8 * physical_sms ? 512 : 1024;
        plan.pre_launch = {dim3(t), dim3(128), 0};
        plan.finish_launch = {dim3(ceil_div(t, splits < 32 ? 8 : 4)), dim3(128), 0};
        return plan;
    }
    constexpr int kGeneralCtasPerSm = 4, kRingCtasPerSm = 8, kCompactCtasPerSm = 12;
    const int general_slots = sms * kGeneralCtasPerSm;
    const int ring_slots = sms * kRingCtasPerSm;
    const int compact_slots = sms * kCompactCtasPerSm;
    const int min_ring_tokens = shifted ? 128 : h == 7168 ? 129 : 320;
    const bool single_k_tile = !shifted && h == 4096 && t >= 256 && t < min_ring_tokens &&
                               ceil_div(t, 32) * (h / 128) <= ring_slots;
    const bool ring_size = (t >= min_ring_tokens || single_k_tile) && t <= 32768;
    int splits = std::min(h / 64, select_num_splits(t, iterations, shifted, general_slots, ring_slots));
    if (!shifted) splits = select_normal_splits(t, h, ring_size, splits, general_slots, ring_slots);
    const int normal_groups = ceil_div(t, 32);
    const int compact_splits = compact_slots / normal_groups;
    const int k_tiles = h / 64;
    const bool compact_k = ring_size && !shifted && h == 4096 && store_bf16 && !store_fp8 &&
                           compact_splits >= 20 && compact_splits <= 24 &&
                           2 * ceil_div(2 * k_tiles, splits) - ceil_div(2 * k_tiles, compact_splits) >= 3;
    const int single_slots = compact_k ? compact_slots : ring_slots;
    const bool single_split = compact_k || (ring_size && !shifted && h == 4096 && store_bf16 && !store_fp8 &&
                              splits >= 10 && splits <= 22 && normal_groups * (splits / 2) <= single_slots);
    if (compact_k) splits = compact_splits;
    else if (single_split) splits /= 2;
    int chunk = 0;
    if (single_split && !compact_k) {
        const int target = ceil_div(normal_groups * k_tiles, single_slots);
        const int previous = ceil_div(k_tiles, splits);
        for (int candidate = target; candidate < previous; ++candidate) {
            const int full = k_tiles / candidate, tail = k_tiles % candidate;
            const int tail_ctas = single_slots - normal_groups * full;
            if (tail && tail_ctas > 0 && tail_ctas < normal_groups &&
                ceil_div(normal_groups, tail_ctas) * tail <= candidate) {
                chunk = candidate;
                break;
            }
        }
    }
    const bool stream_k = chunk != 0;
    if (stream_k) splits = k_tiles / chunk + 1;
    const bool ring = ring_size && (single_split || (splits >= 8 && splits <= 80 && splits % 2 == 0));
    const int ring_m = shifted && splits == 16 && t < 1024 && ceil_div(t, 16) <= sms ? 16 : 32;
    const bool single = ring && (!shifted || iterations == 20);
    const int groups = ceil_div(t, ring_m);
    const bool local = single && (stream_k || groups * (splits / (single_split ? 1 : 2)) <= single_slots);
    int task_splits = single_split ? 1 : 2;
    if (single && !local && t % 32 != 0) {
        const int fine_waves = ceil_div(groups * splits, single_slots);
        const int pair_waves = ceil_div(groups * (splits / 2), single_slots);
        if (100 * fine_waves * ceil_div(80, splits) <
            98 * pair_waves * ceil_div(160, splits)) task_splits = 1;
    }
    const int general_blocks = ceil_div(t, kMegaMhcGeneralTileM);
    const int grid_x = stream_k ? single_slots : local ? groups :
                       single ? std::min(groups * (splits / task_splits), single_slots) :
                                std::min(general_blocks * splits, general_slots);
    return {
        {h, splits, iterations, ring_m, task_splits, t % 64 == 0,
         single, local, t % ring_m != 0, t % (ring_m / 8) != 0, local || t != 1248,
         store_bf16, store_fp8, shared_sf_block_m, per_token_fp8, shifted, chunk, compact_k},
        {dim3(grid_x, local && !stream_k ? splits / task_splits : 1),
         dim3(single ? ring_m * 2 : 128), 0}, sms,
        ceil_div<int64_t>(t, ring ? ring_m : kMegaMhcGeneralTileM) + 2,
        ceil_div<int64_t>(t, kMegaMhcGeneralTileM) * splits * kMegaMhcGeneralTileM * 26 +
            (ring ? (align<int64_t>(t, ring_m) - t) * 5 * h / 2 : 0)};
}

} // namespace deep_gemm
