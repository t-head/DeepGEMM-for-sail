#pragma once

#include "common_bf16.hpp"
#include "bf16_memory_policy.hpp"
#include "common_gemm_comm.hpp"

namespace deep_gemm::gemm_comm {
using Bf16GemmRSConfig = std::pair<ConfigTuple, int>;

inline bool can_pipeline_bf16_gemm_rs(const ConfigTuple& config, int m, int n, int world) {
    const auto& [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    return world > 1 && m % bm == 0 && m / world >= 16 && (m / world) % wm == 0 &&
           int64_t(m / bm) * ceil_div(n, bn) > sms;
}

// A native warp rectangle has one reduce-scatter owner. Use subsequent
// native work as the sender progress window.
inline int get_bf16_gemm_rs_shared_packets(const ConfigTuple& config, int m, int n, int world) {
    const auto& [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    if (!can_pipeline_bf16_gemm_rs(config, m, n, world))
        return 0;
    const int output_threads = (bm / wm) * (bn / wn) * 32;
    return std::min(wm * wn / 256,
                    int((gemm_rs_smem_capacity() - std::get<0>(smem)) / (output_threads * 16))) / 2 * 2;
}

inline void validate_bf16_gemm_rs_config(const ConfigTuple &config, bool require_smem = true) {
    const auto &[sms, bm, bn, bk, wm, wn, stages, smem] = config;
    TORCH_CHECK(sms > 0 && sms <= get_num_sms() && bm > 0 && bn > 0 && wm > 0 && wn > 0 && bm % wm == 0 &&
                    bn % wn == 0 && wm % 16 == 0 && wn % 16 == 0 && (bk == 64 || bk == 128 || bk == 256 || bk == 512) &&
                    stages >= 2,
                "invalid BF16 GEMM/RS tile geometry");
    const int64_t threads = int64_t(bm / wm) * (bn / wn) * 32;
    const int64_t capacity = gemm_rs_smem_capacity();
    const int64_t per_stage = (int64_t(bm) + bn) * bk * 2;
    TORCH_CHECK(int64_t(bm) * bn * 2 <= capacity && stages <= capacity / per_stage,
                "BF16 GEMM/RS tile exceeds shared-memory capacity");
    TORCH_CHECK(threads <= at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock &&
                    (!require_smem || (std::get<0>(smem) > 0 && std::get<0>(smem) <= capacity)),
                "invalid BF16 GEMM/RS resource budget");
}

inline Bf16GemmRSConfig get_bf16_gemm_rs_config(int m, int n, int k, int world, std::optional<ConfigTuple> configs,
                                                const std::string &tile_policy) {
    validate_gemm_rs_shape(m, n, k, world, false);
    TORCH_CHECK(tile_policy == "rs_aware" || tile_policy == "native", "invalid BF16 GEMM/RS tile_policy");
    const bool tune = !configs && world > 1 && tile_policy == "rs_aware" && is_ppu1v5_device();
    int warp_k = 0;
    auto config = configs ? *configs
                          : deep_gemm_bf16_common::get_best_configs(m, n, k, 1, get_num_sms(), GemmType::Normal, 256,
                                                                    nullptr, &warp_k);
    auto &[sms, bm, bn, bk, wm, wn, stages, smem] = config;
    validate_bf16_gemm_rs_config(config, false);
    if (warp_k == 0)
        warp_k = bk;
    TORCH_CHECK(warp_k >= 64 && warp_k % 64 == 0 && bk % warp_k == 0, "invalid BF16 GEMM/RS config");
    smem = deep_gemm_bf16_common::get_smem_config(stages, k, bm, bn, bk, 2);
    const int local_m = m / world, packet_rows = get_gemm_rs_packet_rows(local_m);
    const bool owner_tail = local_m >= 3 && local_m % packet_rows != 0;
    if (tune) {
        if (owner_tail && m <= wm && bm == wm && warp_k == bk &&
                int64_t(n) * k * 2 <= at::cuda::getCurrentDeviceProperties()->l2CacheSize) {
            const int narrow_bn = std::min(bn, 4 * wn);
            // When startup/drain spans most K work, use smaller K tiles to create
            // a useful prefetch window. Longer pipelines retain the native K tile.
            const bool short_pipeline = ceil_div(k, bk) < 2 * stages;
            const int input_stages = short_pipeline ? 3 : 2;
            int narrow_bk = bk;
            auto candidate = deep_gemm_bf16_common::get_smem_config(input_stages, k, bm, narrow_bn, narrow_bk, 2);
            while (short_pipeline && narrow_bk > 64 && std::get<0>(candidate) > gemm_rs_smem_capacity() / 2) {
                narrow_bk /= 2;
                candidate = deep_gemm_bf16_common::get_smem_config(input_stages, k, bm, narrow_bn, narrow_bk, 2);
            }
            // Keep every output tile resident. Otherwise the narrow grid adds a
            // second compute round to the tail instead of hiding communication.
            const int resident_tiles = sms * int(gemm_rs_smem_capacity() / std::get<0>(candidate));
            if ((short_pipeline || stages > 2) && std::get<0>(candidate) <= gemm_rs_smem_capacity() / 2 &&
                    ceil_div(n, narrow_bn) <= resident_tiles &&
                    (!short_pipeline || k >= input_stages * narrow_bk)) {
                bn = narrow_bn;
                bk = warp_k = narrow_bk;
                stages = input_stages;
                if (short_pipeline)
                    sms = get_num_sms();
                smem = candidate;
            }
        }
        if (can_pipeline_bf16_gemm_rs(config, m, n, world) &&
                get_bf16_gemm_rs_shared_packets(config, m, n, world) == 0) {
            const int threads = (bm / wm) * (bn / wn) * 32;
            const int packets = get_gemm_rs_preferred_shared_packets(config);
            const int64_t per_stage = int64_t(bm + bn) * bk * 2;
            const int64_t reduction_smem = int64_t(bk / warp_k - 1) * bm * bn * 4;
            const int minimum_stages = std::max(2, int((reduction_smem + per_stage - 1) / per_stage));
            if (packets > 0)
                adapt_gemm_rs_stages(config, minimum_stages, 1, int64_t(packets) * threads * 16,
                    [&](int input_stages) {
                        return deep_gemm_bf16_common::get_smem_config(input_stages, k, bm, bn, bk, 2);
                    });
        }
        if (owner_tail && m >= wm) {
            const int sm_budget = get_num_sms();
            int native_warps_n = bn / wn;
            if (m <= bm) {
                // Keep a whole-M tile in one wave with aligned transfer columns.
                const int narrow_bn = ceil_div(ceil_div(n, sm_budget), 64) * 64;
                while (narrow_bn % (native_warps_n * 16) != 0)
                    --native_warps_n;
                const int narrow_wn = narrow_bn / native_warps_n;
                const int warps = (bm / wm) * native_warps_n * (bk / warp_k);
                // Widen N only when it reduces the number of N warps.
                // Otherwise preserve the native geometry and input pipeline.
                if (narrow_bn != bn && (narrow_bn < bn || (stages > 3 && native_warps_n < bn / wn)) &&
                        ceil_div(n, narrow_bn) <= sm_budget &&
                        deep_gemm_adaptive::tile_spill_ok(wm, narrow_wn, warps)) {
                    const auto candidate = deep_gemm_bf16_common::get_smem_config(std::min(stages, 3), k, bm, narrow_bn, bk, 2);
                    const int64_t reduction_smem = int64_t(bk / warp_k - 1) * bm * narrow_bn * 4;
                    if (std::get<0>(candidate) <= gemm_rs_smem_capacity() &&
                            int64_t(bm) * narrow_bn * 2 <= gemm_rs_smem_capacity() &&
                            reduction_smem <= std::get<0>(candidate)) {
                        stages = std::min(stages, 3);
                        bn = narrow_bn;
                        wn = narrow_wn;
                        sms = sm_budget;
                        smem = candidate;
                    }
                }
            } else if (world > 2 && int64_t(ceil_div(m, bm)) * ceil_div(n, bn) <= sms &&
                       int64_t(n) * k * 2 <= at::cuda::getCurrentDeviceProperties()->l2CacheSize) {
                // Resident weights can sustain a second tile wave for communication.
                const int native_bn = bn;
                adapt_gemm_rs_n_grid(config, m, n, 2 * sms, false);
                if (bn != native_bn) {
                    stages = std::min(stages, 3);
                    sms = sm_budget;
                    smem = deep_gemm_bf16_common::get_smem_config(stages, k, bm, bn, bk, 2);
                }
            }
        }
    }
    TORCH_CHECK(std::get<0>(smem) <= gemm_rs_smem_capacity(), "BF16 GEMM/RS shared-memory capacity exceeded");
    validate_bf16_gemm_rs_config(config);
    return {config, warp_k};
}

struct Bf16GemmRSPlan {
    int threads = 0;
    int shared_packets = 0;
    int smem_size = 0;
    bool input_overlap = false;
    int b_cache_policy = 1;
    int compact_min_m = 0;
};

inline Bf16GemmRSPlan resolve_bf16_gemm_rs_plan(const Bf16GemmRSConfig& selected, int m, int n, int k, int world) {
    const auto& [config, wk] = selected;
    const auto& [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    const int output_threads = (bm / wm) * (bn / wn) * 32;
    const int shared_packets = get_bf16_gemm_rs_shared_packets(config, m, n, world);
    const int total_smem = std::get<0>(smem) + shared_packets * output_threads * 16;
    const int64_t tiles = int64_t((m + bm - 1) / bm) * ((n + bn - 1) / bn);
    const auto memory_policy = deep_gemm_bf16_common::memory_policies_enabled()
        ? deep_gemm_bf16_common::get_memory_execution_policy(m, n, bm, bn, sms, 1, sms,
              at::cuda::getCurrentDeviceProperties()->multiProcessorCount)
        : deep_gemm_bf16_common::MemoryExecutionPolicy{};
    // Higher source fan-in can amortize partial R8/R16 row compaction over
    // a steady K pipeline. Retain native R4 packets and K-split output.
    const bool compact_partial = world > 4 && m < wm && get_gemm_rs_packet_rows(m / world) > 4 &&
                                 wk == bk && k / bk >= 3 * stages;
    return {output_threads * (bk / wk), shared_packets, total_smem,
            tiles > sms && total_smem > PpuHwParams::instance().tsm_per_cu / 2,
            memory_policy.b_cache_policy, compact_partial ? 16 : wm};
}

} // namespace deep_gemm::gemm_comm
