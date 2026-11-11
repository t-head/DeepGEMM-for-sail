#pragma once

#include "mega_gate_common.hpp"

namespace deep_gemm::mega_gate_810e {
struct MegaGate810EArchSpec {
    static constexpr auto name = "810E/CUTLASS3";
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
    static void validate_tile(const MegaGateConfig& config) {
        const auto& [sms, m, n, k, wm, wn, stages, smem] = config.gemm;
        DG_HOST_ASSERT((config.warp_k - 1) * m * n * 4 <= stages * (m + n) * k * 2);
    }
};
} // namespace deep_gemm::mega_gate_810e
