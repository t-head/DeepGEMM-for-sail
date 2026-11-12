#pragma once
#include <algorithm>
#include <iostream>
#include <deep_gemm/common/gemm_occ_model.cuh>
#include <deep_gemm/layout/mega_gate.cuh>
#include <numeric>
#include <mutex>
#include <unordered_set>
#include "common_bf16.hpp"
#include "runtime.hpp"
namespace deep_gemm {
struct MegaGateConfig {
    deep_gemm_bf16_common::ConfigResult gemm;
    int num_split_k, warp_k, aligned_experts, producers;
    int threads() const {
        const auto& [sms, m, n, k, wm, wn, stages, smem] = gemm;
        return m / wm * (n / wn) * warp_k * 32;
    }
};
inline MegaGateConfig get_mega_gate_config(int tokens, int hidden, int experts, int num_sms,
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
    static std::unordered_set<std::string> printed; static std::mutex mutex;
    const std::lock_guard<std::mutex> lock(mutex);
    if (printed.insert(message).second) std::cout << message << std::endl;
    return config;
}
} // namespace deep_gemm
