#pragma once
#include <deep_gemm/layout/mega_moe_types.cuh>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <map>
#include <optional>
#include <limits>
#include <tuple>
#include <vector>
#include "common_fp4.hpp"
#include "mega_moe_tuning.hpp"

namespace deep_gemm::mega_moe_config {

inline constexpr int kFp4SmemCapacity = 256 * 1024;

inline int64_t fp4_gemm1_scale_width(int n, int k) {
    return int64_t(n) * k / 64;
}

inline int64_t fp4_mainloop_smem(int stages, int bm, int bn, int bk, int sfb_bn) {
    const auto [sfa, invalid_a] = deep_gemm_fp4_common::get_sf_per_stage_size(bm, bk);
    const auto [sfb, invalid_b] = deep_gemm_fp4_common::get_sf_per_stage_size(sfb_bn, bk);
    return int64_t(stages) * ((int64_t(bm) + bn) * bk + sfa + sfb) - invalid_a - invalid_b;
}

inline int64_t fp4_pipeline_smem(const std::array<int, 6>& tile, bool scatter) {
    const auto [bm, bn, bk, wm, wn, stages] = tile;
    const int sfb_bn = scatter ? bn : mega_moe::kFp4Gemm1SfbBlockN;
    const int64_t scratch = scatter
        ? int64_t(bm) * (bn / 2 + mega_moe::kFp4ScatterSmemPadPairs) * sizeof(uint32_t) : 0;
    return std::max({int64_t(bm) * bn * 2, scratch,
                     fp4_mainloop_smem(stages, bm, bn, bk, sfb_bn)});
}

// MMA fragments are 16x16. Both scale-load fragments are limited to four
// iterations per warp. These constraints apply to LUTs and measured tiles too.
inline std::optional<std::array<int, 6>> fit_fp4_pipeline_tile(
    std::array<int, 6> tile, int n, int k, bool scatter) {
    auto& [bm, bn, bk, wm, wn, stages] = tile;
    if ((bm != 16 && bm != 32 && bm != 64 && bm != 128 && bm != 256) ||
        bn <= 0 || (scatter ? bn % 256 != 0 : (bn != 128 && bn != 256)) ||
        n <= 0 || n % bn || bk <= 0 || bk % 64 || (bk > 256 && bk % 256) ||
        k <= 0 || k % bk || stages < 2 ||
        wm < 16 || wm > 64 || wm % 16 || wn < 16 || wn > 64 || wn % 16 ||
        bm % wm || bn % wn || int64_t(bm / wm) * (bn / wn) * 32 > 1024)
        return std::nullopt;
    // AIU's contiguous FP4 span is BK/2 bytes: multiples of 32 up to
    // 128 bytes, then multiples of 128 (DefaultGemm_AIU_Operand).
    // Reject oversized operands before calculating the int-valued scale layout.
    if ((int64_t(bm) + bn) * bk > kFp4SmemCapacity || int64_t(bm) * bn * 2 > kFp4SmemCapacity)
        return std::nullopt;
    if (!scatter) {
        // Match the device stage picker, including the physical BN256 SFB.
        const int64_t two = fp4_mainloop_smem(2, bm, bn, bk, mega_moe::kFp4Gemm1SfbBlockN);
        const int64_t per_stage = two - fp4_mainloop_smem(1, bm, bn, bk, mega_moe::kFp4Gemm1SfbBlockN);
        if (two > kFp4SmemCapacity)
            return std::nullopt;
        const int max_stages = 2 + (kFp4SmemCapacity - two) / per_stage;
        stages = std::min(stages, max_stages);
    }
    if (fp4_pipeline_smem(tile, scatter) > kFp4SmemCapacity)
        return std::nullopt;
    return tile;
}

struct DispatchConfig {
    // Dispatch and GEMM1 row granularity.
    int block_m = 0;
    float activation_clamp = -1;
    bool fast_math = true;
    bool gemm1_dynamic_queue = false;
    bool fused_situ = false;
    bool fused_swiglu = false;
    int gemm2_block_m = 0;
    int gemm1_block_k = 0;
    int gemm1_block_n = 0;
    int gemm1_n = 0;
    int gemm1_stages = 0;
    int gemm1_warp_m = 0;
    int gemm1_warp_n = 0;
    int gemm_ctas = 0;
    int nmtpr = 0;
    int num_experts = 0;
    int num_experts_per_rank = 0;
    int num_ranks = 0;
    int num_sms = 0;
    int num_ctas = 0;
    int num_threads = 0;
    int num_topk = 0;
    int pull_slots = 0;
    int pull_ctas = 0;
    int sfb_layout_block_n = 0;
    int sf_bytes = 0;
    bool silu_then_clamp = false;
    float situ_beta = 0;
    float situ_linear_beta = 0;
    int token_bytes = 0;
};

struct PreDispatchConfig {
    int block_threads = 0;
    int hidden = 0;
    int num_ctas = 0;
    int num_topk = 0;
};

struct ScatterConfig {
    int block_k = 0;
    int block_m = 0;
    int block_n = 0;
    int k = 0;
    bool fp8_output = false;
    int fp8_exp_shift = 0;
    mega_moe::Fp8ScaleMode fp8_scale_mode = mega_moe::Fp8ScaleMode::Fixed;
    int n = 0;
    int num_groups = 0;
    int num_master_groups = 0;
    int m_tiles_per_group = 0;
    int num_stages = 0;
    int warp_m = 0;
    int warp_n = 0;
    int world_size = 0;
};

using Config = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;

struct MegaMoEConfigs {
    DispatchConfig dispatch;
    Config scatter;
    int m_tiles_per_group;
};

inline TuningConfig tuning_config(const TuningWorkload& workload) {
    return select_tuning(workload);
}

inline std::tuple<int, int, int> fp4_smem(int stages, int bm, int bn, int bk, bool scatter = false) {
    return {static_cast<int>(fp4_pipeline_smem({bm, bn, bk, 0, 0, stages}, scatter)), 128, 0};
}
inline Config fp4_config(int total_m, int m, int n, int k, int groups, int sms,
                         GemmType type = GemmType::MGroupedNoPad, int max_bn = 256, int min_bn = 32,
                         bool apply_tile_corrections = true,
                         const deep_gemm_fp4_common::TileConfig* tile_overrides = nullptr, bool scatter = false) {
    // Validate backend-specific resources before accepting a runtime LUT entry.
    // The ordinary GEMM selector keeps its default LUT behavior.
    if (auto entry = deep_gemm_lut::get_runtime_config(m, n, k * 2, groups, type,
                                                     deep_gemm_lut::RuntimeLutKind::FP4)) {
        const auto [bm, bn, bk, wm, wn, stages] = *entry;
        if (bn >= min_bn && bn <= max_bn)
            if (auto tile = fit_fp4_pipeline_tile({bm, bn, bk, wm, wn, stages}, n, k, scatter)) {
                const auto& t = *tile;
                return {sms, t[0], t[1], t[2], t[3], t[4], t[5], fp4_smem(t[5], t[0], t[1], t[2], scatter)};
            }
    }
    auto config = deep_gemm_fp4_common::get_best_configs(
        total_m, m, n, k, groups, sms, false, false, type, max_bn, min_bn,
        apply_tile_corrections, std::nullopt, tile_overrides, false);
    auto& [ns, bm, bn, bk, wm, wn, stages, smem] = config;
    if (auto tile = fit_fp4_pipeline_tile({bm, bn, bk, wm, wn, stages}, n, k, scatter))
        stages = (*tile)[5];
    smem = fp4_smem(stages, bm, bn, bk, scatter);
    return config;
}
inline int threads(int bm, int bn, int wm, int wn) {
    TORCH_CHECK(wm > 0 && wn > 0 && bm % wm == 0 && bn % wn == 0, "Invalid MegaMoE warp tile");
    return bm / wm * (bn / wn) * 32;
}
inline int pre_dispatch_threads(int hidden) {
    for (int t = std::min(hidden / 8, 512) / 32 * 32; t >= 32; t -= 32)
        if (hidden / 8 % t == 0)
            return t;
    return 32;
}
inline PreDispatchConfig fp4_pre_dispatch_config(int hidden, int topk, int tokens) {
    const int threads = pre_dispatch_threads(hidden);
    TORCH_CHECK(threads > 0 && threads % 32 == 0 && threads <= hidden / 8);
    return {threads, hidden, std::min(tokens, get_num_sms() * 4), topk};
}
inline int dispatch_pull_slots(int num_launch_threads, int token_bytes, int scale_bytes) {
    // Match the device pull-staging limit; pull and GEMM reuse this storage.
    constexpr int max_staging_bytes = 256 * 1024;
    TORCH_CHECK(num_launch_threads > 0 && num_launch_threads % 32 == 0);
    TORCH_CHECK(token_bytes > 0 && scale_bytes >= 0 && token_bytes <= max_staging_bytes - scale_bytes,
                "MegaMoE pull staging must fit at least one token and its scales in 256 KiB");
    return std::min({16, num_launch_threads / 32, max_staging_bytes / (token_bytes + scale_bytes)});
}
inline void set_gemm1_tile(DispatchConfig& c, const Tile& tile) {
    c.block_m = tile[0];
    c.gemm1_block_n = tile[1];
    c.gemm1_block_k = tile[2];
    c.gemm1_warp_m = tile[3];
    c.gemm1_warp_n = tile[4];
    c.gemm1_stages = tile[5];
    c.num_threads = threads(c.block_m, c.gemm1_block_n, c.gemm1_warp_m, c.gemm1_warp_n);
}
// Packed FP4 transport dimensions.
inline DispatchConfig dispatch_config(int ranks, int experts, int topk, int capacity,
                                      int token_bytes, int scale_bytes) {
    TORCH_CHECK(ranks > 0 && experts > 0 && experts % ranks == 0 && topk > 0 && capacity >= 0);
    TORCH_CHECK(token_bytes > 0 && scale_bytes >= 0);
    DispatchConfig c;
    c.num_ranks = ranks;
    c.num_sms = get_num_sms();
    c.num_experts = experts;
    c.num_experts_per_rank = experts / ranks;
    c.num_topk = topk;
    c.nmtpr = capacity;
    c.token_bytes = token_bytes;
    c.sf_bytes = scale_bytes;
    return c;
}

// Estimate the marginal expert load for distinct random top-k routing. The
// mode-centred recurrence avoids underflow; the bounded tail keeps selection
// cheap for prefill. Skewed routing remains correct, but is not this estimate.
inline std::vector<std::pair<int, double>> fp4_expert_rows(const DispatchConfig& c, int tokens) {
    const int trials = tokens * c.num_ranks;
    const double p = double(c.num_topk) / c.num_experts;
    if (trials == 0 || p >= 1)
        return {{p >= 1 ? trials : 0, 1.0}};
    const double mean = trials * p, sigma = std::sqrt(mean * (1 - p));
    const int mode = std::min(trials, int((trials + 1) * p));
    const int lo = std::max(0, int(mean - 12 * sigma - 32));
    const int hi = std::min(trials, int(std::ceil(mean + 12 * sigma + 32)));
    std::vector<std::pair<int, double>> rows{{mode, 1.0}};
    double probability = 1, total = 1;
    for (int r = mode; r > lo; --r) {
        probability *= double(r) / (trials - r + 1) * (1 - p) / p;
        rows.emplace_back(r - 1, probability);
        total += probability;
    }
    probability = 1;
    for (int r = mode; r < hi; ++r) {
        probability *= double(trials - r) / (r + 1) * p / (1 - p);
        rows.emplace_back(r + 1, probability);
        total += probability;
    }
    for (auto& row : rows)
        row.second /= total;
    return rows;
}

inline int fp4_gemm1_ctas_per_sm(const DispatchConfig& c, int n, int tokens, int bm, int bn,
                                 int num_threads, int pulls) {
    const int epr = c.num_experts_per_rank;
    const int em = std::max(1, ceil_div(tokens * std::min(c.num_topk, epr), epr));
    const int64_t tiles = int64_t(epr) * ceil_div(em, bm) * (n / bn);
    const int work = std::min<int64_t>(4, (tiles + pulls + c.num_sms - 1) / c.num_sms);
    return std::clamp(std::min(512 / num_threads, work), 1, 4);
}

inline int fp4_pull_ctas(const DispatchConfig& c, const Gemm1Tuning& tuning) {
    const int pulls = get_env<int>("MEGA_MOE_PULL_SMS", std::min(tuning.pull_ctas.value_or(6), c.num_sms - 1));
    TORCH_CHECK(pulls >= 1 && pulls < c.num_sms,
                "W4A4 pull CTAs must leave at least one compute CTA");
    return pulls;
}

inline int fp4_gemm1_residency(const DispatchConfig& c, int n, int tokens, const Tile& tile,
                                int pulls, const Gemm1Tuning& tuning) {
    const auto [bm, bn, bk, wm, wn, stages] = tile;
    const int target = tuning.ctas_per_sm.value_or(
        fp4_gemm1_ctas_per_sm(c, n, tokens, bm, bn, threads(bm, bn, wm, wn), pulls));
    TORCH_CHECK(target >= 1 && target <= 4, "invalid MegaMoE CTA residency target");
    return target;
}

inline bool valid_fp4_candidate(const Tile& tile, int n, int k, bool scatter = false) {
    return fit_fp4_pipeline_tile(tile, n, k, scatter).has_value();
}

inline const deep_gemm_fp4_common::TileConfig& fp4_scatter_tiles() {
    static const deep_gemm_fp4_common::TileConfig tiles = {
        {{16, 256, 64}, {16, 64, 2}},
        {{32, 256, 64}, {32, 32, 2}},
    };
    return tiles;
}

inline int fp4_resident_ctas(const Tile& tile, int thread_budget = 512, int smem_budget = 256 * 1024,
                              bool scatter = false) {
    const auto [bm, bn, bk, wm, wn, stages] = tile;
    // G1 defaults to its CTA policy budget; scatter uses the hardware budget.
    // Compilation and the driver still determine the actual residency.
    return std::min(std::max(1, thread_budget / threads(bm, bn, wm, wn)),
                    smem_budget / std::get<0>(fp4_smem(stages, bm, bn, bk, scatter)));
}

// Complete pipeline families, rather than patches to a selected tuple. Large
// accumulator tiles retain the grouped backend's pipeline. Small tiles also
// consider packed BK64/BK128 with the family's warp layout and stage count.
inline std::vector<Tile> fp4_tile_candidates(int total_m, int em, int n, int k, int groups,
                                             int sms, bool scatter) {
    std::vector<Tile> candidates;
    const auto append = [&](const Tile& tile) {
        if (auto fitted = fit_fp4_pipeline_tile(tile, n, k, scatter))
            if (std::find(candidates.begin(), candidates.end(), *fitted) == candidates.end())
                candidates.push_back(*fitted);
    };
    for (int bm : {16, 32, 64, 128, 256}) {
        for (int bn : {128, 256}) {
            if ((scatter && bn != 256) || n % bn)
                continue;
            const auto [ns, cm, cn, ck, wm, wn, stages, smem] =
                deep_gemm_fp4_common::get_best_configs(total_m, em, n, k, groups, sms, false, false,
                    GemmType::MGroupedNoPad, bn, bn, false, bm, scatter ? &fp4_scatter_tiles() : nullptr, false);
            append({cm, cn, ck, wm, wn, stages});
            if (cm <= 64)
                for (int bk : {64, 128})
                    append({cm, cn, bk, wm, wn, stages});
            // BF16 scatter also has a four-warp BM32 family. Keep both its
            // short and deeper pipelines available alongside the eight-warp family.
            if (scatter && cm == 32)
                for (int depth : {2, 3})
                    append({32, cn, 64, 32, 64, depth});
        }
    }
    return candidates;
}

// PPU1.5 operand/MMA/pipeline costs calibrated from producer/consumer traces.
// These estimate relative cost; they are not peak hardware throughput limits.
inline double fp4_compute_cost(double operand_bytes, double mma_atoms, double fill_bytes,
                                 double active_sms, double waves, int k_steps, int stages) {
    return .0109053 * operand_bytes / active_sms + .450894 * mma_atoms / active_sms +
           .137703 * fill_bytes + 403.829 * waves * k_steps + 25.4264 * waves * stages;
}

inline double fp4_gemm1_cost(const DispatchConfig& c, int n, int tokens, const Tile& tile,
                             const std::vector<std::pair<int, double>>& rows,
                             const Gemm1Tuning& tuning = {}) {
    const auto fitted = fit_fp4_pipeline_tile(tile, n, c.token_bytes, false);
    if (!fitted)
        return std::numeric_limits<double>::infinity();
    const auto [bm, bn, bk, wm, wn, stages] = *fitted;
    const int epr = c.num_experts_per_rank, k = c.token_bytes, sms = c.num_sms;
    const int nt = threads(bm, bn, wm, wn);
    const int smem = std::get<0>(fp4_smem(stages, bm, bn, bk));
    if (nt > 1024 || smem > 256 * 1024 || k % bk != 0)
        return std::numeric_limits<double>::infinity();
    const int pulls = fp4_pull_ctas(c, tuning), group = bm <= 32 ? 1 : pulls;
    const int pull_groups = ceil_div(pulls, group);
    const int slots = dispatch_pull_slots(nt, c.token_bytes, c.sf_bytes);
    const auto steps = [=](int r) { return ceil_div(ceil_div(r, group), slots); };
    double mean = 0, blocks = 0, batches = 0, first_steps = 0, nonempty = 0;
    for (const auto& [r, p] : rows) {
        mean += r * p;
        blocks += ceil_div(r, bm) * p;
        batches += ((r / bm) * steps(bm) + steps(r % bm)) * p;
        first_steps += steps(std::min(r, bm)) * p;
        if (r > 0)
            nonempty += p;
    }
    blocks *= epr;
    batches = std::max(epr * batches / pull_groups, nonempty > 0 ? first_steps / nonempty : 0.0);
    const double tasks = blocks * (n / bn);
    const int resident = std::min(fp4_gemm1_residency(c, n, tokens, tile, pulls, tuning),
                                  fp4_resident_ctas(*fitted));
    const int ctas = sms * resident;
    const double active_sms = std::min<double>(sms, std::max(1.0, tasks));
    const double waves = std::max(1.0, tasks / ctas);
    const double blocks_per_group = std::max(blocks / pull_groups, std::min(1.0, blocks));
    const int parse_ops = ceil_div(epr, 32) + int(std::ceil(std::log2(c.num_ranks))) + 1;
    const double operand_bytes = (blocks * n * k + mean * epr * k * (n / bn)) * 1.0625;
    const double mma_atoms = blocks * (bm / 16) * (n / 16) * (2 * k / 64);
    const double fill_bytes = double(bm + bn) * bk * waves / (stages - 1);
    const double gemm = fp4_compute_cost(operand_bytes, mma_atoms, fill_bytes, active_sms,
                                         waves, k / bk, stages);
    const double queue = 369.986 * waves * (1 + ceil_div(epr, 32));
    const double pull = 776.417 * batches * ceil_div(k, 512) +
        695.330 * blocks_per_group * parse_ops +
        47.5246 * batches * std::max(0, resident - 1) * smem / (256 * 1024);
    const double fill_blocks = std::min(blocks, std::ceil(double(std::max(1, ctas - pulls)) / (n / bn)));
    const double fill = pull / std::max(1.0, blocks_per_group) * std::max(1.0, fill_blocks / pull_groups);
    // Operand/MMA throughput already covers the expected work per SM. Round
    // only the per-CTA pipeline latency when estimating the incomplete wave.
    const double pipeline_wave = fp4_compute_cost(0, 0, fill_bytes / waves, active_sms, 1, k / bk, stages);
    const double tail = (std::ceil(waves) - waves) * pipeline_wave;
    return std::max(gemm + queue, pull) + fill + tail;
}

inline Tile select_fp4_gemm1_tile(const DispatchConfig& c, int n, int tokens, Tile fallback,
                                  const Gemm1Tuning& tuning = {}) {
    const int epr = c.num_experts_per_rank, k = c.token_bytes, sms = c.num_sms;
    const int total_m = std::max(1, tokens * std::min(c.num_topk, epr));
    const int em = std::max(1, ceil_div(total_m, epr));
    // Preserve a legal runtime LUT entry. Explicit tiles bypass this function;
    // measured tiles are resolved by the buffer's cached dispatch selection.
    if (auto tile = deep_gemm_lut::get_runtime_config(
            em, n, k * 2, epr, GemmType::MGroupedNoPad, deep_gemm_lut::RuntimeLutKind::FP4)) {
        const auto [bm, bn, bk, wm, wn, stages] = *tile;
        if (auto fitted = fit_fp4_pipeline_tile({bm, bn, bk, wm, wn, stages}, n, k, false))
            return *fitted;
    }
    const auto rows = fp4_expert_rows(c, tokens);
    Tile selected = fallback;
    double best_cost = fp4_gemm1_cost(c, n, tokens, selected, rows, tuning);
    for (const auto& candidate : fp4_tile_candidates(total_m, em, n, k, epr, sms, false)) {
        const double cost = fp4_gemm1_cost(c, n, tokens, candidate, rows, tuning);
        if (cost < best_cost) {
            best_cost = cost;
            selected = candidate;
        }
    }
    return selected;
}

// Select W4A4 dispatch threads and GEMM1 tiles from the transport byte count.
inline DispatchConfig w4_dispatch_config(DispatchConfig c, int intermediate, int tokens,
                                         const std::optional<Tile>& requested = std::nullopt,
                                         const Gemm1Tuning& tuning = {}) {
    TORCH_CHECK(intermediate > 0 && tokens >= 0 && tokens <= c.nmtpr);
    const int epr = c.num_experts_per_rank, bytes = c.token_bytes, sms = c.num_sms;
    Tile tile{};
    if (requested) {
        tile = *requested;
    } else {
        const int max_bn = std::min(256, intermediate);
        const int total_m = std::max(1, tokens * std::min(c.num_topk, epr));
        const int em = std::max(1, ceil_div(total_m, epr));
        auto [selected_sms, bm, bn, bk, wm, wn, stages, smem] =
            fp4_config(total_m, em, intermediate, bytes, epr, sms, GemmType::MGroupedNoPad, max_bn, 128);
        tile = select_fp4_gemm1_tile(c, intermediate, tokens, {bm, bn, bk, wm, wn, stages}, tuning);
    }
    const auto fitted = fit_fp4_pipeline_tile(tile, intermediate, bytes, false);
    TORCH_CHECK(fitted, "Invalid MegaMoE GEMM1 tile: unsupported MMA/scale fragments or "
                       "minimum two-stage BN256 SFB storage exceeds 256 KiB");
    set_gemm1_tile(c, *fitted);
    TORCH_CHECK(c.num_threads >= 32 && c.num_threads <= 1024);
    c.gemm1_n = intermediate;
    return c;
}
inline DispatchConfig fp4_dispatch_config(int ranks, int experts, int topk, int hidden, int intermediate,
                                          int capacity, int tokens,
                                          const std::optional<Tile>& requested = std::nullopt,
                                          const Gemm1Tuning& tuning = {}) {
    TORCH_CHECK(hidden > 0 && hidden % 32 == 0);
    return w4_dispatch_config(dispatch_config(ranks, experts, topk, capacity, hidden / 2, hidden / 32),
                              intermediate, tokens, requested, tuning);
}

inline double fp4_gemm2_cost(int n, int k, int groups, int sms, const Tile& tile,
                             const std::vector<std::pair<int, double>>& rows) {
    if (!valid_fp4_candidate(tile, n, k, true))
        return std::numeric_limits<double>::infinity();
    const auto [bm, bn, bk, wm, wn, stages] = tile;
    double blocks = 0, mean = 0;
    for (const auto& [r, p] : rows) {
        blocks += ceil_div(r, bm) * p;
        mean += r * p;
    }
    blocks *= groups;
    const double tasks = blocks * (n / bn);
    // Shared memory only gives a pre-JIT upper bound. The compiled kernel's
    // register/shared-memory occupancy still determines the actual launch.
    const auto prop = device_runtime->get_prop();
    const int resident = fp4_resident_ctas(tile, prop->maxThreadsPerMultiProcessor,
                                          static_cast<int>(prop->sharedMemPerMultiprocessor), true);
    const double active_sms = std::min<double>(sms, std::max(1.0, tasks));
    const double waves = std::max(1.0, tasks / (sms * resident));
    // B loads and scheduling follow the expected number of M blocks, not
    // ceil(mean rows / BM). Only valid rows are written to the remote rank.
    const double operand_bytes = (blocks * n * k + mean * groups * k * (n / bn)) * 1.0625 +
                                 mean * groups * n * 2;
    const double mma_atoms = blocks * (bm / 16) * (n / 16) * (2 * k / 64);
    const double fill_bytes = double(bm + bn) * bk * waves / (stages - 1);
    const double compute = fp4_compute_cost(operand_bytes, mma_atoms, fill_bytes, active_sms,
                                            waves, k / bk, stages);
    const double queue = 369.986 * waves;
    const double tail = (std::ceil(waves) - waves) * compute / waves;
    return compute + queue + tail;
}

inline Tile select_fp4_gemm2_tile(int n, int k, int groups, const TuningConfig& tuning, Tile fallback) {
    const auto& w = tuning.workload;
    const int total_m = std::max(1, w.tokens * w.topk), em = ceil_div(total_m, groups);
    if (auto tile = deep_gemm_lut::get_runtime_config(
            em, n, k * 2, groups, GemmType::MGroupedNoPad, deep_gemm_lut::RuntimeLutKind::FP4)) {
        const auto [bm, bn, bk, wm, wn, stages] = *tile;
        if (bn == 256)
            if (auto fitted = fit_fp4_pipeline_tile({bm, bn, bk, wm, wn, stages}, n, k, true))
                return *fitted;
    }
    const auto c = dispatch_config(w.ranks, w.experts, w.topk, std::max(1, w.tokens),
                                    w.hidden / 2, w.hidden / 32);
    const auto rows = fp4_expert_rows(c, w.tokens);
    const int sms = c.num_sms;
    Tile selected = fallback;
    double best_cost = fp4_gemm2_cost(n, k, groups, sms, selected, rows);
    for (const auto& candidate : fp4_tile_candidates(total_m, em, n, k, groups, sms, true)) {
        const double cost = fp4_gemm2_cost(n, k, groups, sms, candidate, rows);
        if (cost < best_cost) {
            best_cost = cost;
            selected = candidate;
        }
    }
    return selected;
}

// Resolve GEMM2 once, before GEMM1 produces its padded row metadata.
// Internal measured tiles override the generic configuration.
inline Config fp4_scatter_config(int n, int k, int groups, const TuningConfig& tuning) {
    TORCH_CHECK(groups > 0 && n > 0 && n % 256 == 0 && k > 0,
                "FP4 scatter requires N divisible by 256 and positive packed K");
    Config config;
    if (tuning.gemm2.tile) {
        const auto [bm, bn, bk, wm, wn, stages] = *tuning.gemm2.tile;
        config = {get_num_sms(), bm, bn, bk, wm, wn, stages, {0, 128, 0}};
    } else {
        // Full-warp BF16 stores constrain the candidate set before BM/BK,
        // warp layout and stages are selected. MegaMoE owns its measured
        // overrides; ordinary GEMM's shape corrections do not apply here.
        const int total_m = std::max(1, tuning.workload.tokens * tuning.workload.topk);
        config = fp4_config(total_m, ceil_div(total_m, groups), n, k, groups, get_num_sms(),
                            GemmType::MGroupedNoPad, 256, 256, false, &fp4_scatter_tiles(), true);
        const auto [sms, bm, bn, bk, wm, wn, stages, smem] = config;
        const auto [cm, cn, ck, cwm, cwn, depth] =
            select_fp4_gemm2_tile(n, k, groups, tuning, {bm, bn, bk, wm, wn, stages});
        config = {sms, cm, cn, ck, cwm, cwn, depth, fp4_smem(depth, cm, cn, ck, true)};
    }
    const auto [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    TORCH_CHECK(valid_fp4_candidate({bm, bn, bk, wm, wn, stages}, n, k, true),
                "Invalid MegaMoE GEMM2 tile: requires full 256-column scatter tiles, "
                "at most four scale fragments per warp, and shared storage within 256 KiB");
    return {sms, bm, bn, bk, wm, wn, stages, fp4_smem(stages, bm, bn, bk, true)};
}

// Keep dispatch scheduling independent of buffer ownership and kernel argument assembly.
inline void fp4_dispatch_schedule(DispatchConfig& c, const TuningConfig& tuning) {
    const auto& gemm1 = tuning.gemm1;
    const int sms = c.num_sms;
    const int pulls = fp4_pull_ctas(c, gemm1);
    c.gemm1_dynamic_queue = gemm1.dynamic_queue.value_or(true);
    c.pull_slots = dispatch_pull_slots(c.num_threads, c.token_bytes, c.sf_bytes);
    // Small tiles share the thread budget of a 512-thread CTA. Avoid adding
    // CTAs beyond the estimated G1 work; actual occupancy also caps launch.
    const int ctas_per_sm = fp4_gemm1_residency(c, c.gemm1_n, tuning.workload.tokens,
        {c.block_m, c.gemm1_block_n, c.gemm1_block_k, c.gemm1_warp_m, c.gemm1_warp_n, c.gemm1_stages},
        pulls, gemm1);
    c.num_ctas = sms * ctas_per_sm;
    c.pull_ctas = pulls;
    c.gemm_ctas = c.num_ctas - pulls;
}

inline const MegaMoEConfigs& get_mega_moe_configs(
    int ranks, int experts, int topk, int hidden, int intermediate, int capacity,
    int tokens, const std::string& activation, int device) {
    using Key = std::tuple<int, int, int, int, int, int, int, int, int, std::string>;
    static thread_local std::map<Key, MegaMoEConfigs> cache;
    const Key key{ranks, experts, topk, hidden, intermediate, capacity, tokens, device, get_num_sms(), activation};
    const auto found = cache.find(key);
    if (found != cache.end()) return found->second;
    const auto tuning = tuning_config({ranks, experts, topk, hidden, intermediate, tokens, activation});
    const auto scatter = fp4_scatter_config(hidden, intermediate / 2, experts / ranks, tuning);
    auto dispatch = fp4_dispatch_config(
        ranks, experts, topk, hidden, 2 * intermediate, capacity, tokens, tuning.gemm1.tile, tuning.gemm1);
    fp4_dispatch_schedule(dispatch, tuning);
    dispatch.gemm2_block_m = std::get<1>(scatter);
    const int m_tiles_per_group = tuning.gemm2.m_tiles_per_group.value_or(2);
    TORCH_CHECK(m_tiles_per_group == 1 || m_tiles_per_group == 2,
                "W4A4 scatter requires one or two M tiles per scheduler group");
    return cache.emplace(key, MegaMoEConfigs{dispatch, scatter, m_tiles_per_group}).first->second;
}

inline int combine_num_threads(int tokens, int hidden) {
    // One CTA per token. Use the actual local row count, not buffer capacity
    // or dispatch padding, to balance row parallelism and work per thread.
    if (tokens >= 512)
        return 256;
    if (tokens <= 64 && hidden > 4096)
        return 1024;
    return 512;
}

} // namespace deep_gemm::mega_moe_config
