#pragma once

#include <algorithm>
#include <atomic>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <unordered_set>

#include <deep_gemm/common/gemm_occ_model.cuh>
#include "common_bf16.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {
inline std::atomic<bool> mega_gate_deterministic{false};

struct MegaGateConfig {
    deep_gemm_bf16_common::ConfigResult gemm;
    int num_split_k, warp_k, aligned_experts, producers;

    int threads() const {
        const auto& [sms, m, n, k, wm, wn, stages, smem] = gemm;
        return m / wm * (n / wn) * warp_k * 32;
    }

    friend std::ostream& operator << (std::ostream& os, const MegaGateConfig& config) {
        const auto& [sms, m, n, k, wm, wn, stages, smem] = config.gemm;
        const auto& [smem_size, swizzle_mode, padding] = smem;
        return os << "MegaGateConfig(block_m=" << m << ", block_n=" << n << ", block_k=" << k
                  << ", warp_m=" << wm << ", warp_n=" << wn << ", warp_k=" << config.warp_k
                  << ", num_split_k=" << config.num_split_k << ", aligned_experts=" << config.aligned_experts
                  << ", producers=" << config.producers << ", num_launch_sms=" << sms
                  << ", threads=" << config.threads() << ", num_stages=" << stages
                  << ", smem_size=" << smem_size << ", swizzle_mode=" << swizzle_mode
                  << ", padding=" << padding << ")";
    }
};

struct MegaGateTaskConfig {
    int num_split_k, full_row_tile, split_row_tile;
    bool split_full_scores;
};

static MegaGateTaskConfig select_mega_gate_task_config(int tokens, int hidden, int aligned_experts,
        int num_sms, bool use_aiu, bool allow_split_full_scores, int topk, int requested, bool deterministic) {
    const int default_splits = deterministic ? 1 :
        requested ? requested : (tokens < 512 and hidden >= 2048 ? 2 : 1);
    // Estimate per-warp partial loads, bias loads, and seed/selection reductions.
    // One seed pass and two compact-list passes bound the routing work; larger
    // workloads retain the original grid and parallel activation path.
    constexpr int warp_size = 32, compact_candidates = 2 * warp_size;
    const int gate_warps = aligned_experts / warp_size;
    const int rows_per_warp = (32 + gate_warps - 1) / gate_warps;
    const int values_per_lane = aligned_experts / warp_size;
    const int routing_work = rows_per_warp * 2 * values_per_lane +
        (rows_per_warp + 1) * 2 * topk + values_per_lane;
    // Share the full-row tile choice with the non-split path. A new K split
    // may double its row reuse, but must not halve the active CU count again.
    int full_row_tile = 16;
    while (full_row_tile < 128 and (tokens + full_row_tile - 1) / full_row_tile > num_sms)
        full_row_tile *= 2;
    const bool short_split_grid = (tokens + 15) / 16 <= num_sms;
    // Cover the split producer wave without rounding the row tile up to the
    // next power of two. Keep the existing short-grid and row-resource limits.
    const int split_workers = std::max(1, num_sms / 2);
    const int split_rows = (tokens + split_workers - 1) / split_workers;
    const int split_row_tile = short_split_grid ? 32 : std::min((split_rows + 31) / 32 * 32, 128);
    const bool split_full_scores = use_aiu and allow_split_full_scores and
        routing_work <= warp_size + 2 * compact_candidates and !deterministic and !requested and
        default_splits == 1 and aligned_experts >= 384 and hidden >= 2048 and
        hidden % 128 == 0 and int64_t((tokens + 15) / 16) * 4 >= int64_t(num_sms) * 3 and
        (short_split_grid or int64_t((tokens + split_row_tile - 1) / split_row_tile) * 2 <= num_sms) and
        // The widest row tile needs enough K work to amortize its larger
        // register footprint and scratch reduction; share the K-plane crossover.
        (split_row_tile < 128 or hidden >= 16 * aligned_experts);
    const int splits = split_full_scores ? 2 : default_splits;
    DG_HOST_ASSERT(hidden % (64 * splits) == 0);
    return {splits, full_row_tile, split_row_tile, split_full_scores};
}

static void adapt_mega_gate_full_expert_tile(MegaGateConfig& config, int hidden, int full_row_tile) {
    namespace bf16 = deep_gemm_bf16_common;
    const auto& hw = PpuHwParams::instance();
    const int aligned_experts = config.aligned_experts;
    const auto bf16_gemm = config.gemm;
    auto& [sms, m, n, k, wm, wn, stages, smem] = config.gemm;
    // Size the full-score tile for a near-full wave before increasing
    // reuse. Expert padding is already priced by aligned_experts.
    m = full_row_tile;
    n = aligned_experts;
    k = 64;
    wm = m <= 32 ? m : m / 2;
    wn = m == 16 ? 32 : n / 8;
    stages = 3;
    // Retain larger BF16 tiles when the complete FP32 router row fits.
    // Round down by the original warp-M so the GEMM mapping stays valid.
    const auto& [base_sms, bm, bn, bk, bwm, bwn, bs, base_smem] = bf16_gemm;
    const int fit_m = std::min(bm, hw.tsm_per_cu / (aligned_experts * 4)) / bwm * bwm;
    int fit_wn = bwn;
    while (fit_m / bwm * (bn / fit_wn) < 16 and fit_wn % 32 == 0)
        fit_wn /= 2;
    if (bn == aligned_experts and fit_m > m and
        fit_m / bwm * (bn / fit_wn) * 32 <= hw.max_threads_per_cta and
        std::get<0>(bf16::get_smem_config(bs, hidden, fit_m, bn, bk)) <= hw.tsm_per_cu) {
        m = fit_m;
        n = bn;
        k = bk;
        wm = bwm;
        wn = fit_wn;
        stages = bs;
    }
}

static bool adapt_mega_gate_narrow_wave(MegaGateConfig& config, int tokens, int hidden, int num_sms) {
    namespace bf16 = deep_gemm_bf16_common;
    const auto& hw = PpuHwParams::instance();
    const int splits = config.num_split_k, experts = config.aligned_experts;
    if (experts != 128 or splits != 2 or hidden < 2048) return false;

    // Two expert groups retain N parallelism. Round the rows needed by one
    // resident wave to the MMA height; keep the original tiny-grid path.
    const int n = experts / 2, producers = splits * (experts / n);
    const int token_tiles = num_sms / producers;
    if (token_tiles == 0) return false;
    const int m = ((tokens + token_tiles - 1) / token_tiles + 15) / 16 * 16;
    if (m <= 16 or m > 512) return false;
    const int grid = (tokens + m - 1) / m * producers;
    // Wider CTAs need more than half a wave plus one producer group;
    // otherwise retain the original grid's CU coverage.
    if (grid < (num_sms + 1) / 2 + producers) return false;

    const auto& [base_sms, bm, bn, bk, bwm, bwn, bs, base_smem] = config.gemm;
    const int stages = std::clamp(bs, 2, 3);
    int k = bk;
    auto smem = bf16::get_smem_config(stages, hidden / splits, m, n, k);
    while (k > 64 and std::get<0>(smem) > hw.tsm_per_cu) {
        k /= 2;
        smem = bf16::get_smem_config(stages, hidden / splits, m, n, k);
    }
    if (hidden % (splits * k) or std::get<0>(smem) > hw.tsm_per_cu) return false;

    int wm = std::gcd(m, bwm);
    const int wn = 16;
    int independent_warps = m / wm * (n / wn), warp_k = 1;
    if (independent_warps >= 8) {
        // Give routing at most one row per warp before adding K groups.
        const int route_rows = (m + producers - 1) / producers;
        while (independent_warps < route_rows and wm > 16) {
            wm = std::gcd(m, wm - 16);
            independent_warps = m / wm * (n / wn);
        }
    } else {
        // Short M/N grids use the available K staging to fill a half CTA.
        while (independent_warps * warp_k < 16 and warp_k < 8 and
               k % (64 * warp_k * 2) == 0)
            warp_k *= 2;
    }
    const int threads = independent_warps * warp_k * 32;
    const int values = m * n / threads;
    if (threads > hw.max_threads_per_cta or values == 0 or values % std::min(values, 8)) return false;

    config.gemm = {num_sms, m, n, k, wm, wn, stages, smem};
    config.warp_k = warp_k;
    return true;
}

static void adapt_mega_gate_wide_wave(MegaGateConfig& config,
        const deep_gemm_bf16_common::ConfigResult& bf16_gemm, int tokens, int hidden, int num_sms) {
    namespace bf16 = deep_gemm_bf16_common;
    const auto& hw = PpuHwParams::instance();
    const int splits = config.num_split_k, experts = config.aligned_experts;
    const auto& [sms, bm, bn, bk, bwm, bwn, bs, old_smem] = config.gemm;
    if (splits <= 1 or config.warp_k <= 1 or
        int64_t((tokens + bm - 1) / bm) * splits * (experts / bn) <= num_sms) return;
    // Only replace an oversubscribed grid whose staging alone limits it to
    // one CTA per CU. Keep layouts that can already share a CU.
    const auto staging = bf16::get_smem_config(bs, hidden / splits, bm, bn, bk);
    if (int64_t(std::get<0>(staging)) * 2 <= hw.tsm_per_cu) return;

    const auto& [base_sms, base_m, base_n, base_k, base_wm, base_wn, base_stages, base_smem] = bf16_gemm;
    // Match the router's expert alignment and halve excess K parallelism.
    const int n = 128, k = std::min(base_k, bk / 2), stages = std::clamp(base_stages, 2, 3);
    if (k < 64 or hidden % (splits * k) or experts % n) return;
    const int warp_k = std::min(config.warp_k / 2, k / 64);
    const int wm = std::min(bwm, 32), wn = 32, producers = splits * (experts / n);
    const int max_m = hw.max_threads_per_cta / ((n / wn) * warp_k * 32) * wm;
    int m = bm;
    // Increase row reuse until the expert/K producer grid fits one wave.
    while (int64_t((tokens + m - 1) / m) * producers > num_sms and m * 2 <= max_m)
        m *= 2;
    if (int64_t((tokens + m - 1) / m) * producers > num_sms) return;
    const auto smem = bf16::get_smem_config(stages, hidden / splits, m, n, k);
    if (std::get<0>(smem) > hw.tsm_per_cu) return;
    config.gemm = {num_sms, m, n, k, wm, wn, stages, smem};
    config.warp_k = warp_k;
}

static void adapt_mega_gate_short_grid(MegaGateConfig& config, int tokens, int hidden,
        int num_sms, bool use_aiu, bool deterministic, int64_t small_grid) {
    namespace bf16 = deep_gemm_bf16_common;
    const auto& hw = PpuHwParams::instance();
    const int splits = config.num_split_k, aligned_experts = config.aligned_experts;
    auto& warp_k = config.warp_k;
    auto& [sms, m, n, k, wm, wn, stages, smem] = config.gemm;
    const auto bf16_gemm = config.gemm;
    // Short routing grids keep one token per gate warp; only the first
    // two waves add K-parallel warps for GEMM and routing latency.
    // Deterministic mode keeps the accumulation order batch invariant.
    if (use_aiu and !deterministic and adapt_mega_gate_narrow_wave(config, tokens, hidden, num_sms))
        return;
    m = 16;
    n = 128;
    k = 64;
    wm = 16;
    wn = 16;
    stages = 2;
    if (use_aiu and !deterministic) {
        if (small_grid < int64_t(num_sms) * 2 and hidden % (splits * 128) == 0) warp_k = 2;
        if (small_grid < num_sms) {
            while (warp_k < 8 and hidden % (splits * 64 * warp_k * 2) == 0)
                warp_k *= 2;
        }
        n = warp_k <= 2 ? 128 : 256 / warp_k;
        k = 64 * warp_k;
        if (splits == 2 and tokens >= 64 and small_grid * 2 >= num_sms and
            hidden % (splits * 128) == 0) {
            // Four expert groups keep the grid close to a wave while a
            // larger warp-M reuses weights across twice as many rows.
            m = wm = 32;
            n = aligned_experts / 4;
            wn = 16;
            warp_k = 2;
            k = 128;
            stages = 3;
            // Spend the otherwise idle CTA resources on K parallelism,
            // avoiding more global partials and giving routing more warps.
            const int wider_k = k * 2, wider_warp_k = warp_k * 2;
            if (hidden % (splits * wider_k) == 0 and
                m / wm * (n / wn) * wider_warp_k * 32 <= hw.max_threads_per_cta and
                std::get<0>(bf16::get_smem_config(stages, hidden / splits, m, n, wider_k)) <= hw.tsm_per_cu) {
                k = wider_k;
                warp_k = wider_warp_k;
            }
        } else if (splits == 1 and aligned_experts == 128 and
                   small_grid * 2 >= num_sms and small_grid < num_sms and
                   hidden >= 512 and hidden % 128 == 0) {
            // Half a wave of full-expert CTAs is cheaper than adding
            // expert producers solely to fill the final few CUs.
            n = 128;
            k = 128;
            warp_k = 1;
            stages = 3;
        }
        // A narrow router with up to two waves of 32-row tiles can
        // double row reuse and K staging while retaining two resident CTAs.
        const int tiles32 = (tokens + 31) / 32;
        if (splits == 1 and aligned_experts == 128 and hidden >= 2048 and hidden % 128 == 0 and
            tiles32 * 2 >= num_sms and tiles32 <= num_sms * 2) {
            m = 32;
            n = 128;
            k = 128;
            wm = 16;
            wn = 16;
            warp_k = 1;
            stages = 3;
        }
        adapt_mega_gate_wide_wave(config, bf16_gemm, tokens, hidden, num_sms);
    }
}

static void adapt_mega_gate_small_wave(MegaGateConfig& config,
        const deep_gemm_bf16_common::ConfigResult& bf16_gemm, int tokens, int hidden, int num_sms) {
    namespace bf16 = deep_gemm_bf16_common;
    const auto& hw = PpuHwParams::instance();
    const int experts = config.aligned_experts;
    const auto& [base_sms, base_m, base_n, base_k, base_wm, base_wn, base_stages, base_smem] = bf16_gemm;
    if (config.num_split_k != 2 or config.warp_k <= 1 or hidden < 2048) return;

    // Keep the short-grid expert granularity, then grow rows only when its
    // producer grid exceeds a wave. Price expert groups and K slices together.
    int splits = 2, m = 16, n = 32;
    if (int64_t((tokens + m - 1) / m) * splits * (experts / n) > num_sms) m *= 2;
    // An underfilled narrow router can spend the same resident wave on four
    // K slices. Retain at least three 256-element K chunks per slice.
    const int split_tiles = num_sms / (4 * (experts / n));
    if (2 * (experts / n) * 2 < num_sms and split_tiles > 0 and hidden % (4 * 256) == 0 and
        hidden / 4 >= 3 * 256) {
        const int split_m = ((tokens + split_tiles - 1) / split_tiles + 15) / 16 * 16;
        if (split_m <= std::min(base_m, 32)) splits = 4, m = split_m;
    }
    const int tiles = (tokens + m - 1) / m;
    n = std::max(n, ((tiles * splits * experts + num_sms - 1) / num_sms + 15) / 16 * 16);
    while (n <= experts / 4 and experts % n) n += 16;
    if (n > experts / 4 or int64_t(tiles) * splits * (experts / n) > num_sms) return;
    // The wider existing row/expert layouts already have adequate coverage.
    if (m >= 32 and std::get<1>(config.gemm) >= m and std::get<2>(config.gemm) <= n) return;

    const int stages = std::clamp(base_stages + (m < base_m), 2, 3);
    int k = base_k;
    // Growing M trades K staging for weight reuse within BF16's resource
    // budget. Tiny row tiles retain BF16's deeper K staging.
    const int smem_budget = m > 16 ? std::min(hw.tsm_per_cu, std::get<0>(base_smem)) : hw.tsm_per_cu;
    auto smem = bf16::get_smem_config(stages, hidden / splits, m, n, k);
    while (k > 64 and (hidden % (splits * k) or std::get<0>(smem) > smem_budget)) {
        k /= 2;
        smem = bf16::get_smem_config(stages, hidden / splits, m, n, k);
    }
    const int warp_k = std::min(4, k / 64), threads = (n / 16) * warp_k * 32;
    const int values = m * n / threads;
    if (hidden % (splits * k) or std::get<0>(smem) > smem_budget or
        threads > hw.max_threads_per_cta or values == 0 or values % std::min(values, 8)) return;
    config = {{num_sms, m, n, k, m, 16, stages, smem}, splits, warp_k, experts, 0};
}

static void adapt_mega_gate_bf16_tile(MegaGateConfig& config, int tokens, int experts, int num_sms) {
    const auto& hw = PpuHwParams::instance();
    const int splits = config.num_split_k, aligned_experts = config.aligned_experts;
    auto& [sms, m, n, k, wm, wn, stages, smem] = config.gemm;
    // Avoid additional expert padding and preserve cheap on-chip routing.
    n = std::clamp(n / 16 * 16, 16, 512);
    const int expert_alignment = std::lcm(n, 128);
    if ((experts + expert_alignment - 1) / expert_alignment * expert_alignment > aligned_experts) n = 128;
    if (aligned_experts == 128 and splits == 1) n = 128;
    const int score_rows = n == aligned_experts and splits == 1 ?
        std::min(512, hw.tsm_per_cu / (aligned_experts * 4)) / 16 * 16 : 256;
    m = std::clamp(m / 16 * 16, 16, score_rows);
    wm = std::gcd(m, std::clamp(wm / 16 * 16, 16, 256));
    wn = std::gcd(n, std::clamp(wn / 16 * 16, 16, 128));
    // Short staging leaves room for routing and concurrent producer CTAs.
    const bool large_bf16_tile = m > 256;
    const int target_sms = large_bf16_tile ? sms : num_sms;
    k = 64;
    stages = large_bf16_tile ? stages : 2;
    // Retain BF16's near-full wave for a large tile instead of halving
    // row reuse merely to fill the last one or two CUs.
    // Routing needs enough independent token tiles and gate warps.
    while (m > 16 and int64_t((tokens + m - 1) / m) *
           ((aligned_experts + n - 1) / n) * splits < target_sms) {
        // Halving an odd BF16 tile (e.g. 112 -> 48) can collapse
        // gcd(wm,m) to 16 and triple the warp count. Keep a balanced
        // power-of-two tile instead of manufacturing more gate warps.
        int smaller_m = std::max(m / 2 / 16 * 16, 16);
        int smaller_wm = std::gcd(wm, smaller_m);
        if (smaller_m / smaller_wm > m / wm) {
            smaller_m = 16;
            while (smaller_m * 2 < m) smaller_m *= 2;
            smaller_wm = std::min(smaller_m, 64);
            stages = 3;
        }
        m = smaller_m;
        wm = smaller_wm;
    }
    while (m / wm * (n / wn) < 8 and (wn > 16 or wm > 16)) {
        if (wn > 16) wn = std::gcd(n, wn - 16);
        else wm = std::gcd(m, wm - 16);
    }
}

static void adapt_mega_gate_fusion(MegaGateConfig& config, const MegaGateTaskConfig& task,
        int tokens, int hidden, int experts, int num_sms, bool use_aiu, bool deterministic) {
    const int splits = config.num_split_k, aligned_experts = config.aligned_experts;
    const int64_t small_grid = int64_t((tokens + 15) / 16) * (aligned_experts / 128) * splits;
    const bool full_expert_tile = use_aiu and !deterministic and splits == 1 and
        aligned_experts >= 384 and hidden / 64 >= 32 and
        int64_t((tokens + 15) / 16) * 4 >= int64_t(num_sms) * 3;
    auto& [sms, m, n, k, wm, wn, stages, smem] = config.gemm;
    const auto bf16_gemm = config.gemm;

    // Keep this priority: split full scores, full expert rows, short grids, BF16 tiles.
    if (task.split_full_scores) {
        // Two K slices share routing after one cooperative barrier.
        m = task.split_row_tile;
        n = aligned_experts;
        k = 64;
        wm = std::gcd(m, m >= 128 ? 64 : 32);
        wn = m == 32 ? 32 : n / 8;
        stages = 3;

        // Grouped producers can retain a larger BF16 K tile and row reuse.
        // Admit only a complete resident wave, without additional expert padding.
        const auto& hw = PpuHwParams::instance();
        const auto fits_wave = [&](const MegaGateConfig& candidate) {
            const auto& [csms, cm, cn, ck, cwm, cwn, cstages, csmem] = candidate.gemm;
            const int threads = candidate.threads(), values = cm * cn / threads;
            return cn < aligned_experts and aligned_experts % cn == 0 and
                hidden % (splits * ck) == 0 and ck % (candidate.warp_k * std::min(ck, 64)) == 0 and
                int64_t((tokens + cm - 1) / cm) * splits * (aligned_experts / cn) <= num_sms and
                threads <= hw.max_threads_per_cta and values > 0 and values % std::min(values, 8) == 0 and
                std::get<0>(deep_gemm_bf16_common::get_smem_config(cstages, hidden / splits, cm, cn, ck)) <= hw.tsm_per_cu;
        };
        MegaGateConfig grouped{bf16_gemm, splits, 1, aligned_experts, 0};
        adapt_mega_gate_short_grid(grouped, tokens, hidden, num_sms, use_aiu, deterministic, small_grid);
        const int grouped_producers = splits * (aligned_experts / std::get<2>(grouped.gemm));
        const int grouped_grid = (tokens + std::get<1>(grouped.gemm) - 1) / std::get<1>(grouped.gemm) * grouped_producers;
        if (fits_wave(grouped) and grouped_grid >= (num_sms + 1) / 2 + grouped_producers) {
            config = grouped;
        } else {
            // A shorter BF16 pipeline is another candidate when it keeps the
            // full producer grid resident and doubles the original K staging.
            grouped = {bf16_gemm, splits, 1, aligned_experts, 0};
            const int grouped_k = std::get<3>(grouped.gemm);
            grouped.warp_k = std::min(2, grouped_k / 64);
            std::get<6>(grouped.gemm) = 2;
            if (grouped_k > k) {
                if (fits_wave(grouped)) {
                    config = grouped;
                } else {
                    auto& [csms, cm, cn, ck, cwm, cwn, cstages, csmem] = grouped.gemm;
                    const int producers = splits * ((aligned_experts + cn - 1) / cn);
                    const int token_tiles = num_sms / producers;
                    if (token_tiles > 0) {
                        // Fit the BF16 expert groups into one resident wave.
                        // Bound row growth and per-CTA work to retain the
                        // full-row tile's cache-resident throughput.
                        cm = ((tokens + token_tiles - 1) / token_tiles + 15) / 16 * 16;
                        cwm = std::gcd(cm, cwm);
                        const int mn_warps = cm / cwm * (cn / cwn);
                        grouped.warp_k = 1;
                        while (mn_warps * grouped.warp_k < hw.max_threads_per_cta / (2 * 32) and
                               grouped.warp_k < 8 and grouped_k % (64 * grouped.warp_k * 2) == 0)
                            grouped.warp_k *= 2;
                        // Spend K parallelism only when M/N supply too few
                        // warps. Preserve the short BF16 pipeline's K depth
                        // with more stages when reducing its per-warp K tile.
                        ck = std::min(grouped_k, 64 * grouped.warp_k);
                        cstages = std::min(std::get<6>(bf16_gemm), 2 * grouped_k / ck);
                        if (cm <= 2 * m and cm * cn <= m * n and fits_wave(grouped))
                            config = grouped;
                    }
                }
            }
        }
    } else if (full_expert_tile) {
        adapt_mega_gate_full_expert_tile(config, hidden, task.full_row_tile);
    } else if (!use_aiu or deterministic or small_grid < int64_t(num_sms) * 8) {
        adapt_mega_gate_short_grid(config, tokens, hidden, num_sms, use_aiu, deterministic, small_grid);
    } else {
        adapt_mega_gate_bf16_tile(config, tokens, experts, num_sms);
    }

    // Preserve BF16's warp layout when the fused CTA would exceed the thread limit.
    if (config.threads() > PpuHwParams::instance().max_threads_per_cta) m = wm;
}

static void apply_mega_gate_overrides(MegaGateConfig& config) {
    auto& [sms, m, n, k, wm, wn, stages, smem] = config.gemm;
    auto& warp_k = config.warp_k;
    const int forced_m = get_env<int>("DG_MEGA_GATE_BLOCK_M", 0);
    const int forced_n = get_env<int>("DG_MEGA_GATE_BLOCK_N", 0);
    if (forced_m) {
        m = forced_m;
        wm = m >= 128 ? 64 : m >= 32 ? 32 : 16;
    }
    if (forced_n) {
        n = forced_n;
        wn = n >= 256 ? 64 : m >= 32 ? 32 : 16;
    }
    k = get_env<int>("DG_MEGA_GATE_BLOCK_K", k);
    stages = get_env<int>("DG_MEGA_GATE_STAGES", stages);
    warp_k = get_env<int>("DG_MEGA_GATE_WARP_K", warp_k);
    wm = get_env<int>("DG_MEGA_GATE_WARP_M", wm);
    wn = get_env<int>("DG_MEGA_GATE_WARP_N", wn);
}

static void finalize_mega_gate_resources(MegaGateConfig& config, int tokens, int hidden,
        int experts, int num_sms, bool use_aiu, bool deterministic) {
    namespace bf16 = deep_gemm_bf16_common;
    const auto& hw = PpuHwParams::instance();
    const int splits = config.num_split_k, warp_k = config.warp_k;
    auto& [sms, m, n, k, wm, wn, stages, smem] = config.gemm;
    DG_HOST_ASSERT(m >= 16 and m <= 512 and m % 16 == 0);
    DG_HOST_ASSERT(n >= 16 and n <= 512 and n % 16 == 0);
    DG_HOST_ASSERT((use_aiu and k == 32) or k == 64 or k == 128 or k == 256 or k == 512);
    DG_HOST_ASSERT(warp_k == 1 or warp_k == 2 or warp_k == 4 or warp_k == 8);
    DG_HOST_ASSERT(wm >= 16 and wm <= 256 and wm % 16 == 0 and m % wm == 0);
    DG_HOST_ASSERT(wn >= 16 and wn <= 128 and wn % 16 == 0 and n % wn == 0);
    // Each BF16 warp-K slice starts on a complete 128-byte swizzle tile
    // (or the whole K tile when it is narrower).
    DG_HOST_ASSERT(k % (warp_k * std::min(k, 64)) == 0);
    DG_HOST_ASSERT(use_aiu or (m == 16 and n == 128 and k == 64 and warp_k == 1 and wm == 16 and wn == 16));
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
    // Narrow split-K routers can outgrow a resident grid solely because
    // of staging. Prefer a shorter pipeline when it fits every producer;
    // the compiled cooperative kernel still gets the final occupancy check.
    if (use_aiu and splits > 1 and padded_experts == 128 and stages > 2 and
        !get_env<int>("DG_MEGA_GATE_STAGES", 0)) {
        const int64_t grid = int64_t((tokens + m - 1) / m) * producers;
        const int64_t required_blocks = (grid + num_sms - 1) / num_sms;
        const auto shorter = bf16::get_smem_config(2, hidden / splits, m, n, k);
        if (int64_t(std::get<0>(smem)) * required_blocks > hw.tsm_per_cu and
            int64_t(std::get<0>(shorter)) * required_blocks <= hw.tsm_per_cu) {
            stages = 2;
            smem = shorter;
        }
    }
    if (!use_aiu) std::get<0>(smem) = 0;
    if (use_aiu and producers == 1)
        std::get<0>(smem) = std::max(std::get<0>(smem), m * padded_experts * 4);
    config.aligned_experts = padded_experts;
    config.producers = producers;
    const int values_per_thread = m * n / config.threads();
    DG_HOST_ASSERT(values_per_thread > 0 and values_per_thread % std::min(values_per_thread, 8) == 0);
    DG_HOST_ASSERT(config.threads() <= hw.max_threads_per_cta);
    DG_HOST_ASSERT(std::get<0>(smem) <= hw.tsm_per_cu);
}

static void print_mega_gate_config(const MegaGateConfig& config, int tokens, int hidden, int experts,
        int num_sms, int topk, bool use_aiu, bool allow_split_full_scores, bool deterministic) {
    if (!get_env<int>("DG_PRINT_CONFIGS", 0)) return;

    std::ostringstream message;
    message << "MegaGateConfig(num_tokens=" << tokens << ", hidden=" << hidden
            << ", num_routed_experts=" << experts << ", num_sms=" << num_sms
            << ", topk=" << topk << ", use_aiu=" << use_aiu
            << ", allow_split_full_scores=" << allow_split_full_scores
            << ", deterministic=" << deterministic << "): " << config;
    // Include the result so changing a tuning override prints the new config too.
    static std::unordered_set<std::string> printed;
    static std::mutex mutex;
    const std::lock_guard<std::mutex> lock(mutex);
    if (printed.insert(message.str()).second)
        std::cout << message.str() << std::endl;
}

inline MegaGateConfig get_mega_gate_config(int tokens, int hidden, int experts, int num_sms,
        bool use_aiu, bool allow_split_full_scores = true, int topk = 0) {
    namespace bf16 = deep_gemm_bf16_common;
    DG_HOST_ASSERT(tokens > 0 and hidden > 0 and experts > 0 and experts <= 512 and num_sms > 0);
    DG_HOST_ASSERT(topk >= 0 and topk <= 32);
    const int requested = get_env<int>("DG_MEGA_GATE_SPLIT_K", 0);
    DG_HOST_ASSERT(requested == 0 or requested == 1 or requested == 2 or requested == 4 or requested == 8);
    const bool deterministic = mega_gate_deterministic.load();
    const int aligned_experts = (experts + 127) / 128 * 128;
    const auto task = select_mega_gate_task_config(tokens, hidden, aligned_experts, num_sms,
        use_aiu, allow_split_full_scores, topk, requested, deterministic);

    // Reuse BF16's LUT/adaptive/heuristic selection and configuration ABI.
    const auto gemm = use_aiu ? bf16::get_best_configs(tokens, experts, hidden / task.num_split_k,
        task.num_split_k, num_sms, GemmType::Normal) :
        bf16::ConfigResult{num_sms, 16, 128, 64, 16, 16, 2, {0, 128, 0}};
    MegaGateConfig config{gemm, task.num_split_k, 1, aligned_experts, 0};
    adapt_mega_gate_fusion(config, task, tokens, hidden, experts, num_sms, use_aiu, deterministic);
    if (use_aiu and !deterministic and !requested and !task.split_full_scores)
        adapt_mega_gate_small_wave(config, gemm, tokens, hidden, num_sms);
    // A narrow router may retain a larger row/K tile with two K slices even
    // after the initial task selector stops splitting. Reuse its resident-wave
    // adapter; rejected resource or grid layouts keep the original config.
    if (use_aiu and !deterministic and !requested and config.num_split_k == 1 and
        aligned_experts == 128 and hidden >= 2048 and hidden % 128 == 0) {
        MegaGateConfig split{bf16::get_best_configs(tokens, experts, hidden / 2, 2,
            num_sms, GemmType::Normal), 2, 1, aligned_experts, 0};
        if (adapt_mega_gate_narrow_wave(split, tokens, hidden, num_sms) and
            std::get<1>(split.gemm) > std::get<1>(config.gemm)) config = split;
    }
    // Explicit tuning overrides retain their priority over fusion adaptation.
    apply_mega_gate_overrides(config);
    finalize_mega_gate_resources(config, tokens, hidden, experts, num_sms, use_aiu, deterministic);
    print_mega_gate_config(config, tokens, hidden, experts, num_sms, topk,
                          use_aiu, allow_split_full_scores, deterministic);
    return config;
}
} // namespace deep_gemm
