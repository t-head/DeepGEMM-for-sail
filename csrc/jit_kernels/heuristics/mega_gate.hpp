#pragma once
#include <algorithm>
#include <iostream>
#include <deep_gemm/common/gemm_occ_model.cuh>
#include <deep_gemm/layout/mega_gate.cuh>
#include <numeric>
#include <sstream>
#include <mutex>
#include <unordered_set>
#include "common_bf16.hpp"
#include "runtime.hpp"
#include "../../utils/exception.hpp"
namespace deep_gemm {
struct MegaGateConfig {
    deep_gemm_bf16_common::ConfigResult gemm;
    int num_split_k, warp_k, aligned_experts, producers;
    int threads() const {
        const auto& [sms, m, n, k, wm, wn, stages, smem] = gemm;
        return m / wm * (n / wn) * warp_k * 32;
    }
};
inline void print_mega_gate_config_once(const std::string& message) {
    static std::unordered_set<std::string> printed;
    static std::mutex mutex;
    const std::lock_guard<std::mutex> lock(mutex);
    if (printed.insert(message).second) std::cout << message << std::endl;
}
namespace mega_gate {

inline MegaGateConfig get_sm89_mega_gate_config(int tokens, int hidden, int experts, int num_sms,
        bool allow_split_full_scores = true, int topk = 0, bool single_producer = false) {
    namespace bf16 = deep_gemm_bf16_common;
    const auto& hw = PpuHwParams::instance();
    DG_HOST_ASSERT(tokens > 0 && hidden > 0 && experts > 0 && experts <= 512 && num_sms > 0 && topk >= 0 && topk <= 32);
    const bool deterministic = heuristics_runtime->get_deterministic_algorithms() || single_producer;
    constexpr int expert_group = mega_gate_layout::kExpertAlignment, token_tile = mega_gate_layout::kBlockTokens;
    const int e = align(experts, expert_group);
    const bool split_wave = int64_t(ceil_div(tokens, token_tile)) * (e / expert_group) * 2 <= int64_t(num_sms) * 4;
    int splits = deterministic ? 1 : split_wave && hidden >= 2048 && hidden % 128 == 0 ? 2 : 1;
    const int full_m = next_power_of_two(std::clamp(ceil_div(tokens, num_sms), token_tile, 128));
    const int split_m = ceil_div(tokens, token_tile) <= num_sms ? 32 : std::min(align(ceil_div(tokens, std::max(1, num_sms / 2)), 32), 128);
    const bool full_split = allow_split_full_scores && !deterministic && splits == 1 &&
        e > 2 * expert_group && hidden >= 2048 && hidden % 128 == 0 &&
        int64_t(ceil_div(tokens, token_tile)) * 4 >= int64_t(num_sms) * 3 &&
        (ceil_div(tokens, token_tile) <= num_sms || ceil_div(tokens, split_m) * 2 <= num_sms) &&
        (split_m < 128 || hidden >= 16 * e);
    if (full_split) splits = 2;
    DG_HOST_ASSERT(hidden % (64 * splits) == 0);
    if (!deterministic && splits == 1 && e == expert_group && hidden >= 2048 && hidden % 128 == 0) {
        const int bm = align(ceil_div(tokens, std::max(1, num_sms / 4)), 32);
        if (bm <= 512 && bm / std::gcd(bm, 32) * 4 * 32 <= hw.max_threads_per_cta) splits = 2;
    }
    int sms, m, n, k, wm, wn, stages;
    std::tuple<int, int, int> smem;
    std::tie(sms, m, n, k, wm, wn, stages, smem) =
        bf16::get_best_configs(tokens, experts, hidden / splits, splits, num_sms, GemmType::Normal);
    const int base_warps = m / wm * (n / wn), base_warp_area = wm * wn;
    int warp_k = 1;
    const auto grid = [&](int bm, int bn, int sk) { return int64_t(ceil_div(tokens, bm)) * sk * (e / bn); };
    const auto take = [&](int bm, int bn, int bk, int bwm, int bwn, int wk, int st, int sk, int budget) {
        const int threads = bm / bwm * (bn / bwn) * wk * 32, values = bm * bn / threads;
        if (hidden % (sk * bk) || threads > hw.max_threads_per_cta || values <= 0 ||
            values % std::min(values, 8) || std::get<0>(bf16::get_smem_config(st, hidden / sk, bm, bn, bk)) > budget) return false;
        std::tie(m, n, k, wm, wn, warp_k, stages, splits) = std::tuple{bm, bn, bk, bwm, bwn, wk, st, sk};
        return true;
    };
    [&] {
        if (deterministic) { m = wm = wn = 16; n = single_producer ? e : 128; k = 64; stages = 2; return; }
        if (splits == 1 && e > 2 * expert_group && hidden >= 2048 &&
            int64_t(ceil_div(tokens, token_tile)) * 4 >= int64_t(num_sms) * 3) {
            const int fit_m = std::min(m, hw.tsm_per_cu / (e * 4)) / wm * wm;
            int fit_wn = wn;
            while (fit_m / wm * (n / fit_wn) < base_warps && fit_wn % 32 == 0) fit_wn /= 2;
            if (n == e && fit_m > full_m && take(fit_m, n, k, wm, fit_wn, 1, stages, splits, hw.tsm_per_cu)) return;
            m = full_m; n = e; k = 64; wm = m <= 32 ? m : m / 2; wn = m == 16 ? 32 : e / 8; stages = 3; return;
        }
        // Tiny grids trade some K staging for additional expert/K producers.
        if (splits == 2 && hidden >= 2048) {
            int sk = splits, bm = grid(16, 32, splits) > num_sms ? 32 : 16;
            const int slots = num_sms / (4 * (e / 32));
            if (4 * (e / 32) < num_sms && slots > 0 && hidden % 1024 == 0 && hidden / 4 >= 768 &&
                align(ceil_div(tokens, slots), 16) <= std::min(m, 32)) sk = 4, bm = align(ceil_div(tokens, slots), 16);
            int bn = std::max(32, align(ceil_div(ceil_div(tokens, bm) * sk * e, num_sms), 16));
            while (bn <= e / 4 && e % bn) bn += 16;
            int bk = k, st = std::clamp(stages + (bm < m), 2, 3);
            const int budget = bm > 16 ? std::min(hw.tsm_per_cu, std::get<0>(smem)) : hw.tsm_per_cu;
            while (bk > 64 && (hidden % (sk * bk) || std::get<0>(bf16::get_smem_config(st, hidden / sk, bm, bn, bk)) > budget)) bk /= 2;
            if (bn <= e / 4 && grid(bm, bn, sk) <= num_sms && take(bm, bn, bk, bm, 16, std::clamp(bk / 64, 1, 4), st, sk, budget)) return;
        }
        if (full_split && n < e && e % n == 0 && k > 64 && grid(m, n, splits) <= num_sms &&
            take(m, n, k, wm, wn, std::min(2, k / 64), 2, splits, hw.tsm_per_cu)) return;
        if (splits > 1 && hidden >= 2048) {
            const int bn = std::min(e / 2, expert_group), producers = splits * (e / bn), slots = num_sms / producers;
            int bm = slots ? align(ceil_div(tokens, slots), 16) : 0;
            if (bn == expert_group) bm = std::max(16, next_power_of_two(bm));
            else if (bm > 2 * bn) bm = align(bm, 32);
            if (bm > 0 && bm <= 512 && (!full_split || bm <= 2 * split_m) && grid(bm, bn, splits) <= num_sms) {
                int bwm = std::gcd(bm, 32), bwn = bn / 4;
                if (bn < expert_group && bm <= 2 * bn && bm / bwm * (bn / bwn) >= base_warps)
                    while (bwm > 16 && bm / bwm * (bn / bwn) < ceil_div(bm, producers)) bwm = std::gcd(bm, bwm - 16);
                int bk = k, st = std::clamp(stages, 2, 3), wk = 1;
                while (bk > 64 && (hidden % (splits * bk) || std::get<0>(bf16::get_smem_config(st, hidden / splits, bm, bn, bk)) > hw.tsm_per_cu)) bk /= 2;
                while (bm / bwm * (bn / bwn) * wk < base_warps && wk < 8 && bk % (128 * wk) == 0 && bm / bwm * (bn / bwn) * wk * 64 <= hw.max_threads_per_cta) wk *= 2;
                if (bn < expert_group && bm / bwm * (bn / bwn) >= base_warps) wk = 1;
                if (bn == expert_group) wk = std::min(2, bk / 64);
                if (take(bm, bn, bk, bwm, bwn, wk, st, splits, hw.tsm_per_cu)) return;
            }
        }
        if (full_split) {
            const int slots = num_sms / (splits * (e / n));
            const int bm = slots ? align(ceil_div(tokens, slots), 16) : 0, bwm = std::gcd(bm, wm);
            const int st = std::min(stages, 2 * k / 64);
            if (bm > 0 && bm <= 2 * split_m && bm * n <= split_m * e && e % n == 0 &&
                take(bm, n, 64, bwm, wn, 1, st, splits, hw.tsm_per_cu)) return;
            m = split_m; n = e; k = 64; wm = std::gcd(m, m >= 128 ? 64 : 32);
            wn = m == 32 ? 32 : e / 8; stages = 3; return;
        }
        if (splits == 1 && e == expert_group && grid(token_tile, expert_group, 1) < int64_t(num_sms) * 8) {
            const bool rows32 = hidden >= 2048 && hidden % 128 == 0 && ceil_div(tokens, 32) * 2 >= num_sms && ceil_div(tokens, 32) <= num_sms * 2;
            m = rows32 ? 32 : token_tile; n = e; k = rows32 ? 128 : 64; wm = wn = 16; stages = rows32 ? 3 : 2; return;
        }
        n = std::clamp(n / 16 * 16, 16, 512);
        if (align(experts, std::lcm(n, expert_group)) > e) n = expert_group;
        if (e == expert_group && splits == 1) n = e;
        const int max_m = n == e && splits == 1 ? std::min(512, hw.tsm_per_cu / (e * 4)) / 16 * 16 : 256;
        m = std::clamp(m / 16 * 16, 16, max_m);
        wm = std::gcd(m, std::clamp(wm / 16 * 16, 16, 256)); wn = std::gcd(n, std::clamp(wn / 16 * 16, 16, 128));
        const int target_sms = m > 256 ? sms : num_sms;
        while (m > 16 && grid(m, n, splits) < target_sms) {
            int smaller_m = std::max(m / 2 / 16 * 16, 16), smaller_wm = std::gcd(wm, smaller_m);
            if (smaller_m / smaller_wm > m / wm) {
                smaller_m = next_power_of_two(m / 2); smaller_wm = std::min(smaller_m, 64); stages = 3;
            }
            m = smaller_m; wm = smaller_wm;
        }
        while (m / wm * (n / wn) < std::min(base_warps, ceil_div(m * e, base_warp_area)) && (wn > 16 || wm > 16))
            if (wn > 16) wn = std::gcd(n, wn - 16); else wm = std::gcd(m, wm - 16);
        const int budget = std::max(m * n * int(sizeof(float)), hw.tsm_per_cu / int(ceil_div(grid(m, n, splits), int64_t(num_sms))));
        while (stages > 2 && std::get<0>(bf16::get_smem_config(stages, hidden / splits, m, n, k)) > budget) --stages;
        while (k > 64 && (hidden % (splits * k) || std::get<0>(bf16::get_smem_config(stages, hidden / splits, m, n, k)) > budget)) k /= 2;
    }();
    // Increase the warp tile only when normalization creates too many threads.
    while (m / wm * (n / wn) * warp_k * 32 > hw.max_threads_per_cta) {
        int next_wn = wn + 16;
        while (next_wn <= std::min(n, 128) && n % next_wn != 0) next_wn += 16;
        if (next_wn <= std::min(n, 128)) { wn = next_wn; continue; }
        int next_wm = wm + 16;
        while (next_wm <= std::min(m, 256) && m % next_wm != 0) next_wm += 16;
        if (next_wm > std::min(m, 256)) break;
        wm = next_wm;
    }
    // Preserve a resident producer grid while trading warp-K for larger M/N reuse.
    if (allow_split_full_scores && !deterministic && splits == 2 && (m > wm || warp_k == 2) && e % n == 0 && grid(m, n, splits) <= num_sms) {
        const int bm0 = m, bn0 = n, bk0 = k, st0 = stages, wm0 = wm, wn0 = wn, sk = splits * 2;
        const bool exchange_k = m == wm;
        const int target_warps = std::max(exchange_k ? 0 : base_warps, m / wm * (n / wn) * warp_k);
        const auto traffic = [&](int bm, int bn, int wk, int sk) {
            return grid(bm, bn, sk) * (int64_t(bm + bn) * (hidden / sk) * 2 +
                int64_t(bm) * bn * 4 * (2 + (wk > 1 ? 2 * wk : 0)));
        };
        auto best = traffic(m, n, warp_k, splits);
        for (auto [bm, bn] : {std::pair{bm0 * 2, bn0}, std::pair{bm0, bn0 * 2}}) {
            if (bm > 512 || bn > e || e % bn || grid(bm, bn, sk) != grid(bm0, bn0, 2)) continue;
            int bwm = exchange_k ? wm0 : std::gcd(bm, 32);
            int bwn = exchange_k ? wn0 : bn < expert_group ? 16 : std::gcd(bn, 32), wk = 1;
            if (const int mn_warps = bm / bwm * (bn / bwn); mn_warps < target_warps) {
                if (target_warps % mn_warps == 0 && bwm / (target_warps / mn_warps) >= 16) bwm /= target_warps / mn_warps;
                else wk = 2;
            }
            if (bm / bwm * (bn / bwn) * wk < target_warps) continue;
            const auto cost = traffic(bm, bn, wk, sk);
            if (cost > best || (cost == best && bn <= n)) continue;
            const int depth = (exchange_k ? std::min(st0 + 1, 4) : st0) * bk0;
            for (int bk = bk0; bk >= 64; bk /= 2) {
                const int st = exchange_k ? depth / bk0 : std::clamp(depth / bk, bn == e && bm <= sk * target_warps ? 4 : 3, 4);
                if (bk % (wk * 64) || st * bk > depth || hidden / sk < 2 * st * bk ||
                    !take(bm, bn, bk, bwm, bwn, wk, st, sk, hw.tsm_per_cu)) continue;
                best = cost; break;
            }
        }
    }
    DG_HOST_ASSERT(m >= 16 && m <= 512 && m % 16 == 0 && n >= 16 && n <= 512 && n % 16 == 0);
    DG_HOST_ASSERT((k == 32 || k == 64 || k == 128 || k == 256 || k == 512) && hidden % (k * splits) == 0);
    DG_HOST_ASSERT((warp_k == 1 || warp_k == 2 || warp_k == 4 || warp_k == 8) && k % (warp_k * std::min(k, 64)) == 0);
    DG_HOST_ASSERT(wm >= 16 && wm <= 256 && wm % 16 == 0 && m % wm == 0);
    DG_HOST_ASSERT(wn >= 16 && wn <= 128 && wn % 16 == 0 && n % wn == 0 && stages >= 2 && stages <= 4);
    DG_HOST_ASSERT(!deterministic || warp_k == 1);
    const int padded_e = align(experts, std::lcm(n, expert_group)), producers = splits * (padded_e / n);
    DG_HOST_ASSERT(padded_e <= 512);
    smem = bf16::get_smem_config(stages, hidden / splits, m, n, k);
    if (splits > 1 && padded_e == expert_group && stages > 2) {
        const auto shorter = bf16::get_smem_config(2, hidden / splits, m, n, k);
        const int needed = ceil_div(ceil_div(tokens, m) * producers, num_sms);
        if (int64_t(std::get<0>(smem)) * needed > hw.tsm_per_cu && int64_t(std::get<0>(shorter)) * needed <= hw.tsm_per_cu)
            stages = 2, smem = shorter;
    }
    if (producers == 1) std::get<0>(smem) = std::max(std::get<0>(smem), m * padded_e * 4);
    MegaGateConfig config{{std::min<int64_t>(num_sms, int64_t(ceil_div(tokens, m)) * producers), m, n, k, wm, wn, stages, smem}, splits, warp_k, padded_e, producers};
    const int values = m * n / config.threads();
    DG_HOST_ASSERT(values > 0 && values % std::min(values, 8) == 0 && config.threads() <= hw.max_threads_per_cta);
    DG_HOST_ASSERT(std::get<0>(smem) <= hw.tsm_per_cu);
    if (!get_env<int>("DG_PRINT_CONFIGS", 0)) return config;
    const auto message = fmt::format("MegaGateConfig(tokens={}, hidden={}, experts={}, topk={}, deterministic={}): "
        "MegaGateConfig(tile={}x{}x{}, warp={}x{}, stages={}, split_k={}, warp_k={}, threads={}, smem={}, sms={})",
        tokens, hidden, experts, topk, int(deterministic), m, n, k, wm, wn, stages, splits, warp_k,
        config.threads(), std::get<0>(smem), std::get<0>(config.gemm));
    print_mega_gate_config_once(message);
    return config;
}



static std::ostream& operator << (std::ostream& os, const MegaGateConfig& config) {
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

struct MegaGateGemmChoice {
    deep_gemm_bf16_common::ConfigResult tile;
    int dense_stages;
    bool from_lut;
};

struct MegaGateTaskConfig {
    int num_split_k, full_row_tile, split_row_tile;
    bool split_full_scores;
};

struct MegaGateProblem {
    int tokens, hidden, experts, num_sms, topk;
    bool allow_split_full_scores;
    bool deterministic;

    int aligned_experts() const { return ceil_div(experts, 128) * 128; }
};

static MegaGateGemmChoice get_mega_gate_bf16_choice(const MegaGateProblem& p, int splits) {
    bool from_lut = false;
    auto tile = deep_gemm_bf16_common::get_best_configs(p.tokens, p.experts, p.hidden / splits,
        splits, p.num_sms, GemmType::Normal, 256, &from_lut);
    return {tile, std::get<6>(tile), from_lut};
}

static MegaGateTaskConfig select_task(const MegaGateProblem& p) {
    const int aligned_experts = p.aligned_experts();
    const int default_splits = p.deterministic ? 1 :
        (ceil_div(p.tokens, 16) < 2 * p.num_sms and p.hidden >= 2048 ? 2 : 1);
    constexpr int warp_size = 32, compact_candidates = 2 * warp_size;
    const int gate_warps = aligned_experts / warp_size;
    const int rows_per_warp = (32 + gate_warps - 1) / gate_warps;
    const int values_per_lane = aligned_experts / warp_size;
    const int routing_work = rows_per_warp * 2 * values_per_lane +
        (rows_per_warp + 1) * 2 * p.topk + values_per_lane;
    int full_row_tile = 16;
    while (full_row_tile < 128 and (p.tokens + full_row_tile - 1) / full_row_tile > p.num_sms)
        full_row_tile *= 2;
    const bool short_split_grid = (p.tokens + 15) / 16 <= p.num_sms;
    // Round split tiles to the MMA row granularity.
    const int split_workers = std::max(1, p.num_sms / 2);
    const int split_rows = (p.tokens + split_workers - 1) / split_workers;
    const int split_row_tile = short_split_grid ? 32 : std::min((split_rows + 31) / 32 * 32, 128);
    const bool split_full_scores = p.allow_split_full_scores and
        routing_work <= warp_size + 2 * compact_candidates and !p.deterministic and
        default_splits == 1 and aligned_experts > 2 * 128 and p.hidden >= 2048 and
        p.hidden % 128 == 0 and int64_t((p.tokens + 15) / 16) * 4 >= int64_t(p.num_sms) * 3 and
        (short_split_grid or int64_t((p.tokens + split_row_tile - 1) / split_row_tile) * 2 <= p.num_sms) and
        // Wide tiles need enough K work to amortize their larger footprint.
        (split_row_tile < 128 or p.hidden >= 16 * aligned_experts);
    const int splits = split_full_scores ? 2 : default_splits;
    return {splits, full_row_tile, split_row_tile, split_full_scores};
}
static MegaGateGemmChoice select_gemm(const MegaGateProblem& p, int splits) {
    auto choice = get_mega_gate_bf16_choice(p, splits);
    if (!p.deterministic and !choice.from_lut)
        choice.tile = deep_gemm_bf16_common::get_adaptive_configs_bf16(
            p.tokens, p.experts, p.hidden / splits, p.num_sms);
    return choice;
}
static MegaGateConfig select_layout(const MegaGateProblem& p, const MegaGateTaskConfig& task,
                                    const MegaGateGemmChoice& base) {
    namespace bf16 = deep_gemm_bf16_common;
    const auto& hw = PpuHwParams::instance();
    const int tokens = p.tokens, hidden = p.hidden, experts = p.experts, num_sms = p.num_sms;
    const bool deterministic = p.deterministic, allow_split_full_scores = p.allow_split_full_scores;
    constexpr int expert_group = 128, token_tile = 16;
    const int e = p.aligned_experts(), full_m = task.full_row_tile, split_m = task.split_row_tile;
    const bool full_split = task.split_full_scores;
    int splits = task.num_split_k;
    int sms, m, n, k, wm, wn, stages;
    std::tuple<int, int, int> smem;
    std::tie(sms, m, n, k, wm, wn, stages, smem) = base.tile;
    const int base_warps = m / wm * (n / wn), base_warp_area = wm * wn;
    int warp_k = 1;
    const auto grid = [&](int bm, int bn, int sk) { return int64_t(ceil_div(tokens, bm)) * sk * (e / bn); };
    const auto take = [&](int bm, int bn, int bk, int bwm, int bwn, int wk, int st, int sk, int budget) {
        const int threads = bm / bwm * (bn / bwn) * wk * 32, values = bm * bn / threads;
        if (hidden % (sk * bk) || (wk - 1) * bm * bn * 4 > st * (bm + bn) * bk * 2 || threads > hw.max_threads_per_cta || values <= 0 ||
            values % std::min(values, 8) || std::get<0>(bf16::get_smem_config(st, hidden / sk, bm, bn, bk)) > budget) return false;
        std::tie(m, n, k, wm, wn, warp_k, stages, splits) = std::tuple{bm, bn, bk, bwm, bwn, wk, st, sk};
        return true;
    };
    [&] {
        if (deterministic) { m = wm = wn = 16; n = 128; k = 64; stages = 2; return; }
        if (splits == 1 && e > 2 * expert_group && hidden >= 2048 &&
            int64_t(ceil_div(tokens, token_tile)) * 4 >= int64_t(num_sms) * 3) {
            const int fit_m = std::min(m, hw.tsm_per_cu / (e * 4)) / wm * wm;
            int fit_wn = wn;
            while (fit_m / wm * (n / fit_wn) < base_warps && fit_wn % 32 == 0) fit_wn /= 2;
            if (n == e && fit_m > full_m && take(fit_m, n, k, wm, fit_wn, 1, stages, splits, hw.tsm_per_cu)) return;
            m = full_m; n = e; k = 64; wm = m <= 32 ? m : m / 2; wn = m == 16 ? 32 : e / 8; stages = 3; return;
        }
        if (splits == 2 && hidden >= 2048) {
            int sk = splits, bm = grid(16, 32, splits) > num_sms && tokens > 16 ? 32 : 16;
            const int slots = num_sms / (4 * (e / 32));
            if (4 * (e / 32) < num_sms && slots > 0 && hidden % 1024 == 0 && hidden / 4 >= 768 &&
                align(ceil_div(tokens, slots), 16) <= std::min(m, 32)) sk = 4, bm = align(ceil_div(tokens, slots), 16);
            int bn = std::max(32, align(ceil_div(ceil_div(tokens, bm) * sk * e, num_sms), 16)); const int max_bn = std::max(expert_group / 2, e / 4);
            while (bn <= max_bn && e % bn) bn += 16;
            if (bn <= max_bn && bn > 32 && bn % 32 == 0 && grid(bm, bn, sk) < ceil_div(num_sms, 2) + sk * (e / bn) && e % (bn / 2) == 0) bn /= 2;
            int bk = k, st = std::clamp(stages + (bm < m), 2, 3);
            const int budget = hw.tsm_per_cu / (grid(bm, bn, sk) > num_sms ? 2 : 1);
            while (grid(bm, bn, sk) <= num_sms && bk < 512 && hidden % (sk * bk * 2) == 0 && hidden / sk / (bk * 2) >= 2 * st &&
                   bk / 64 < std::min(8, bm / 4) && std::get<0>(bf16::get_smem_config(st, hidden / sk, bm, bn, bk * 2)) <= budget) bk *= 2;
            while (bk > 64 && (hidden % (sk * bk) || std::get<0>(bf16::get_smem_config(st, hidden / sk, bm, bn, bk)) > budget)) bk /= 2;
            if (bn <= max_bn && grid(bm, bn, sk) <= num_sms + sk * (e / bn) && take(bm, bn, bk, bm, 16, std::clamp(bk / 64, 1, std::min(8, bm / 4)), st, sk, budget)) return;
        }
        if (full_split && n < e && e % n == 0 && k > 64 && grid(m, n, splits) <= num_sms &&
            take(m, n, k, wm, wn, std::min(2, k / 64), 2, splits, hw.tsm_per_cu)) return;
        if (full_split) {
            const int slots = num_sms / (splits * (e / n));
            const int bm = slots ? align(ceil_div(tokens, slots), 16) : 0, bwm = std::gcd(bm, wm);
            const int st = std::min(stages, 2 * k / 64);
            if (bm > 0 && bm <= 2 * split_m && bm * n <= split_m * e && e % n == 0 &&
                take(bm, n, 64, bwm, wn, 1, st, splits, hw.tsm_per_cu)) return;
            m = split_m; n = e; k = 64; wm = std::gcd(m, m >= 128 ? 64 : 32);
            wn = m == 32 ? 32 : e / 8; stages = 3; return;
        }
        if (splits == 1 && e == expert_group && grid(token_tile, expert_group, 1) < int64_t(num_sms) * 8) {
            m = std::max(token_tile, align(ceil_div(tokens, num_sms), token_tile)); n = e; k = hidden % 128 == 0 ? 128 : 64;
            wm = 16; wn = 32; stages = hidden / k >= 3 ? 3 : 2; return;
        }
        n = std::clamp(n / 16 * 16, 16, 512);
        if (align(experts, std::lcm(n, expert_group)) > e) n = expert_group;
        if (e == expert_group && splits == 1) n = e;
        const int max_m = n == e && splits == 1 ? std::min(512, hw.tsm_per_cu / (e * 4)) / 16 * 16 : 256;
        const int clipped_m = std::clamp(m / 16 * 16, 16, max_m);
        const int step = std::lcm(16, wm);
        m = clipped_m < m && clipped_m >= step ? clipped_m / step * step : clipped_m;
        wm = std::gcd(m, std::clamp(wm / 16 * 16, 16, 256)); wn = std::gcd(n, std::clamp(wn / 16 * 16, 16, 128));
        const int target_sms = m > 256 ? sms : num_sms;
        while (m > 16 && grid(m, n, splits) < target_sms) {
            int smaller_m = std::max(m / 2 / 16 * 16, 16), smaller_wm = std::gcd(wm, smaller_m);
            if (smaller_m / smaller_wm > m / wm) {
                smaller_m = next_power_of_two(m / 2); smaller_wm = std::min(smaller_m, 64); stages = 3;
            }
            m = smaller_m; wm = smaller_wm;
        }
        while (m / wm * (n / wn) < std::min(base_warps, ceil_div(m * e, base_warp_area)) && (wn > 16 || wm > 16))
            if (wn > 16) wn = std::gcd(n, wn - 16); else wm = std::gcd(m, wm - 16);
        const int budget = std::max(m * n * int(sizeof(float)), hw.tsm_per_cu / int(ceil_div(grid(m, n, splits), int64_t(num_sms))));
        while (stages > 2 && std::get<0>(bf16::get_smem_config(stages, hidden / splits, m, n, k)) > budget) --stages;
        while (k > 64 && (hidden % (splits * k) || std::get<0>(bf16::get_smem_config(stages, hidden / splits, m, n, k)) > budget)) k /= 2;
    }();
    // Compare resident producer layouts using work, occupancy, and split reduction.
    if (!deterministic && splits == 2 && warp_k <= 2) {
        const auto cost = [&](const MegaGateConfig& c) {
            const auto& [cs, cm, cn, ck, cwm, cwn, st, unused] = c.gemm;
            const int we = deep_gemm_adaptive::WE_PER_CU, warps = c.threads() / 32;
            const int producers = c.num_split_k * (e / cn);
            const int waves = ceil_div(ceil_div(tokens, cm) * producers, num_sms);
            const double depth = double(hidden) / c.num_split_k;
            const double compute = double(cm) * cn * depth * ceil_div(warps, we) * we / warps / 96;
            const double reads = (cm + cn) * depth * std::max(1.0, 2.0 * we / (warps * (st - 1)));
            const double warp_reads = double(cm) * cn * depth * (1.0 / cwm + 1.0 / cwn) / we;
            const double routing = double(cm) * waves / producers *
                (4.0 * e * c.num_split_k + 2.0 * p.topk * c.num_split_k +
                 (e + p.topk) * (c.num_split_k - 1));
            return waves * (std::max(compute, reads) + warp_reads +
                   2.0 * (c.warp_k - 1) * cm * cn / we) + routing;
        };
        const MegaGateConfig original{{sms,m,n,k,wm,wn,stages,smem},splits,warp_k,e,0};
        const int wide_n = std::gcd(e, 128), narrow_n = std::max(64, e / 4), wide_slots = num_sms / (splits * (e / wide_n)), narrow_slots = num_sms / (splits * (e / narrow_n));
        const auto padded = [&](int slots) { const int rows = align(ceil_div(tokens, slots), 16); return ceil_div(tokens, rows) * rows; };
        const int bn = wide_slots && ceil_div(tokens, wide_slots) * 2 >= wide_n && (!narrow_slots || padded(wide_slots) <= padded(narrow_slots)) ? wide_n : narrow_n, groups = e / bn;
        bool selected = false;
        if (e % bn == 0) for (int resident : {2, 1}) {
            const int sk = 2 * resident, producers = sk * groups;
            const int slots = num_sms / producers * resident;
            int bk = std::min(std::max(128, std::get<3>(base.tile)), 256 / resident);
            const int st = std::clamp(std::get<6>(base.tile) + 1, 2, 3);
            if (!slots) continue;
            const int bm = align(ceil_div(tokens, slots), 16);
            const int bwm = resident == 2 && bm % 32 == 0 ? bm / 2 : std::gcd(bm, 32);
            while (bk > 64 && (hidden % (sk * bk) || int64_t(std::get<0>(bf16::get_smem_config(st, hidden / sk, bm, bn, bk))) * resident > hw.tsm_per_cu)) bk /= 2;
            if (bk < 128 || hidden % (sk * bk) || hidden / sk / bk < 2 * st) continue;
            const int wk = resident == 1 && groups < 4 && bn < bk && bm / bwm * (bn / 16) > 2 * deep_gemm_adaptive::WE_PER_CU ? 1 : bk / 64;
            if (bm > 512 || bwm > 256) continue;
            const auto bs = bf16::get_smem_config(st, hidden / sk, bm, bn, bk);
            if (int64_t(std::get<0>(bs)) * resident > hw.tsm_per_cu) continue;
            if (resident == 1 && wk <= warp_k && (groups < 4 ? grid(bm, bn, sk) >= grid(m, n, splits) :
                std::min<int64_t>(grid(bm, bn, sk), num_sms) <= std::min<int64_t>(grid(m, n, splits), num_sms))) continue;
            for (int bwn = 16; bwn <= std::min(bn, 128); bwn += 16) {
                if (bn % bwn) continue;
                const int warps = bm / bwm * (bn / bwn) * wk, threads = warps * 32;
                const int allocated = ceil_div(warps, deep_gemm_adaptive::WE_PER_CU) * deep_gemm_adaptive::WE_PER_CU * resident;
                const int regs = bwm * bwn / 32 + (bwm + bwn) * (bk / wk) / 64 + 24;
                if (warps < deep_gemm_adaptive::WE_PER_CU || warps > std::max((bm >= bn ? 4 : 3) * deep_gemm_adaptive::WE_PER_CU, ceil_div(bm, producers)) || threads > hw.max_threads_per_cta ||
                    warps * resident > hw.max_warps_per_cu || regs > std::min(256, hw.total_vreg_per_cu / allocated)) continue;
                const MegaGateConfig c{{num_sms,bm,bn,bk,bwm,bwn,st,bs},sk,wk,e,0};
                if (resident == 2 && cost(c) >= cost(original)) break;
                if (take(bm,bn,bk,bwm,bwn,wk,st,sk,hw.tsm_per_cu)) { selected = true; break; }
            }
            if (selected) break;
        }
    }
    if (!deterministic && splits > 1 && warp_k == 1 && n <= e && e % n == 0 && grid(m, n, splits) > num_sms) {
        const int slots = num_sms / (splits * (e / n)), bk = std::max(64, k / 2);
        if (slots) { const int bm = align(ceil_div(tokens, slots), 16), bwm = std::gcd(bm, wm);
            if (bm > m && bm <= 512) take(bm, n, bk, bwm, wn, 1, std::min(4, stages * k / bk), splits, hw.tsm_per_cu); }
    }
    if (m / wm * (n / wn) * warp_k * 32 > hw.max_threads_per_cta) m = wm;
    if (splits > 1 && grid(m, n, splits) <= num_sms)
        while (stages < (warp_k > 2 ? 4 : std::max(base.dense_stages, std::get<6>(base.tile))) && hidden / splits / k >= (warp_k > 2 ? 2 : 1) * (stages + 1) &&
               std::get<0>(bf16::get_smem_config(stages + 1, hidden / splits, m, n, k)) <= hw.tsm_per_cu) ++stages;
    if (splits == 1 && n == e)
        while (stages < 4 && hidden / k >= stages + 1 &&
               std::get<0>(bf16::get_smem_config(stages + 1, hidden, m, n, k)) <= m * n * 4) ++stages;
    return {{sms, m, n, k, wm, wn, stages, smem}, splits, warp_k, e, 0};
}

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
    DG_HOST_ASSERT((warp_k - 1) * m * n * 4 <= stages * (m + n) * k * 2);
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
            << ", topk=" << topk << ", arch_path=" << "810E/CuTe-free"
            << ", allow_split_full_scores=" << allow_split_full_scores
            << ", deterministic=" << deterministic << "): " << config;
    print_mega_gate_config_once(message.str());
}

inline MegaGateConfig get_sm80_mega_gate_config(int tokens, int hidden, int experts, int num_sms,
                                           int topk, bool allow_split_full_scores = true) {
    DG_HOST_ASSERT(tokens > 0 and hidden > 0 and experts > 0 and experts <= 512 and num_sms > 0);
    DG_HOST_ASSERT(topk >= 0 and topk <= 32);
    const bool deterministic = heuristics_runtime->get_deterministic_algorithms();
    const MegaGateProblem problem{tokens, hidden, experts, num_sms, topk,
                                  allow_split_full_scores, deterministic};
    const auto task = select_task(problem);
    DG_HOST_ASSERT(hidden % (64 * task.num_split_k) == 0);
    const auto base = select_gemm(problem, task.num_split_k);
    auto config = select_layout(problem, task, base);
    finalize_mega_gate_resources(config, tokens, hidden, experts, num_sms, deterministic);
    print_mega_gate_config(config, tokens, hidden, experts, num_sms, topk, allow_split_full_scores, deterministic);
    return config;
}

} // namespace mega_gate
} // namespace deep_gemm
