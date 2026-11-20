#pragma once

#include "common_int8.hpp"
#include "common_gemm_comm.hpp"

namespace deep_gemm::gemm_comm {

// Share sender admission between input-stage planning and runtime selection.
inline bool can_pipeline_channelwise_gemm_rs(const ConfigTuple& config, int m, int world) {
    const auto& [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    return world > 1 && m >= bm && (m > bm || (m / world) % 16 == 0) &&
           m / world >= 16 && stages >= 3;
}

inline Fp8GemmRSConfig get_channelwise_gemm_rs_config(int m, int n, int k, int world,
                                                      std::optional<ConfigTuple> configs, const std::string& tile_policy) {
    auto tile = get_gemm_rs_tile(configs);
    const int sm_budget = configs ? std::get<0>(*configs) : get_num_sms();
    validate_gemm_rs_shape(m, n, k, world, false);
    TORCH_CHECK(tile_policy == "rs_aware" || tile_policy == "native", "channelwise tile_policy must be rs_aware or native");
    TORCH_CHECK(is_ppu1v5_device(), "channelwise FP8 GEMM/RS requires PPU0015");
    const bool tune = !configs && world > 1 && tile_policy == "rs_aware";
    ConfigTuple config;
    if (!tile) {
        auto native_tile = deep_gemm_lut::get_runtime_config(m, n, k, 1, GemmType::Normal,
                                                             deep_gemm_lut::RuntimeLutKind::FP8Channelwise);
        if (!native_tile)
            native_tile = deep_gemm_int8_lut::get_best_configs_from_lut(m, n, k, GemmType::Normal);
        if (native_tile) {
            const auto [bm, bn, bk, wm, wn, stages] = *native_tile;
            tile = Fp8GemmRSTile{bm, bn, bk, wm, wn, stages};
        } else {
            config = deep_gemm_int8::get_best_configs_dense_ppu1v5(m, n, k, 1, sm_budget);
        }
    }
    if (tile) {
        const auto [bm, bn, bk, wm, wn, stages] = *tile;
        config = {sm_budget, bm, bn, bk, wm, wn, stages, {}};
    }
    auto& [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    validate_gemm_rs_tile({bm, bn, bk, wm, wn, stages}, true);
    if (tune) {
        const auto native_smem = deep_gemm_int8::get_smem_config(stages, k, bm, bn, bk, 1);
        const auto* properties = at::cuda::getCurrentDeviceProperties();
        const int local_m = m / world;
        const int packet_rows = std::max(4, local_m);
        const int smem_blocks = std::max(1, int(properties->sharedMemPerMultiprocessor / std::get<0>(native_smem)));
        // Narrow N while the native tile grid leaves compute resources idle.
        const int packet_parallelism = std::min({smem_blocks, packet_rows / local_m, wn / 16});
        if ((bm / wm) * (bn / wn) <= 8)
            adapt_gemm_rs_n_grid(config, m, n, sm_budget * packet_parallelism, true);
    }
    validate_gemm_rs_tile({bm, bn, bk, wm, wn, stages}, true);
    smem = deep_gemm_int8::get_smem_config(stages, k, bm, bn, bk, 1);
    const int local_m = m / world, packet_rows = get_gemm_rs_packet_rows(local_m);
    const bool owner_tail = local_m >= 3 && local_m % packet_rows != 0;
    // Two-source reduction retains raw packets. Preserve its native M grid
    // when growing a tile; shrinking an existing tile does not change readiness.
    if (tune && owner_tail &&
            (world > 2 || m <= bm) &&
            (m >= bm || int64_t(n) * k <= at::cuda::getCurrentDeviceProperties()->l2CacheSize)) {
        adapt_gemm_rs_m_tile(config, m, n, world, sm_budget, 2,
            [&](int rows, int columns, int input_stages) {
                return deep_gemm_int8::get_smem_config(input_stages, k, rows, columns, bk, 1);
            });
        validate_gemm_rs_tile({bm, bn, bk, wm, wn, stages}, true);
    }
    if (tune) {
        // With one M tile and a deep BK64 pipeline, merge pairs of input
        // stages to reduce K-loop overhead without increasing input storage.
        if (m <= bm && bk == 64 && stages >= 8 && stages % 2 == 0 && k % (2 * bk) == 0) {
            bk *= 2;
            stages /= 2;
            smem = deep_gemm_int8::get_smem_config(stages, k, bm, bn, bk, 1);
        }
        const auto shared_size = [&](int input_stages) {
            return deep_gemm_int8::get_smem_config(input_stages, k, bm, bn, bk, 1);
        };
        // Fit a single-M-warp N grid in one compute round.
        if (m <= bm && bm == wm)
            adapt_gemm_rs_stages(config, 3, ceil_div(ceil_div(n, bn), sms), 0, shared_size);
        if (can_pipeline_channelwise_gemm_rs(config, m, world)) {
            // Balance the sender grid after owner-tail compaction. Two-source
            // reduction retains raw packets and its native wide-warp grid.
            if (world > 2) {
                adapt_gemm_rs_n_grid(config, m, n, sm_budget, true);
                smem = shared_size(stages);
            }
            int native_blocks = int(gemm_rs_smem_capacity() / std::get<0>(smem));
            const int64_t tiles = int64_t(ceil_div(m, bm)) * ceil_div(n, bn);
            const int packets = get_gemm_rs_preferred_shared_packets(config);
            int threads = (bm / wm) * (bn / wn) * 32;
            if (native_blocks > 0 && tiles > int64_t(sms) * native_blocks && packets > 0) {
                // Retain warp shapes while sharing a whole-M tile between two
                // resident CTAs, including their input and sender storage.
                // Each CTA retains at least two M warps for its input pipeline.
                if (m == bm && native_blocks == 1 && bm % (2 * wm) == 0 && bm >= 4 * wm) {
                    const int64_t sender_smem = int64_t(packets) * (threads / 2) * 16;
                    const auto minimum_smem = deep_gemm_int8::get_smem_config(3, k, bm / 2, bn, bk, 1);
                    if (std::get<0>(minimum_smem) + sender_smem <= gemm_rs_smem_capacity() / 2) {
                        bm /= 2;
                        threads /= 2;
                        native_blocks = 2;
                    }
                }
                adapt_gemm_rs_stages(config, 3, native_blocks, int64_t(packets) * threads * 16, shared_size);
            }
        }
        if (m > bm && bk == 128 && stages == 3 && k >= 2 * bk && k % (bk / 2) == 0) {
            const int64_t tiles = int64_t(ceil_div(m, bm)) * ceil_div(n, bn);
            const int resident_blocks = int(ceil_div(tiles, int64_t(sms)));
            const int64_t resident_smem = gemm_rs_smem_capacity() / resident_blocks;
            const bool short_pipeline = ceil_div(k, bk) <= 4 * stages;
            const auto narrow_k = deep_gemm_int8::get_smem_config(4, k, bm, bn, bk / 2, 1);
            // Prefer a single compute wave when the smaller K pipeline fits
            // both the shared-memory and register budgets.
            if (!short_pipeline && std::get<0>(smem) > resident_smem && std::get<0>(narrow_k) <= resident_smem &&
                    deep_gemm_adaptive::tile_spill_ok(wm, wn, (bm / wm) * (bn / wn) * resident_blocks)) {
                bk /= 2;
                stages = 4;
                smem = narrow_k;
            } else if (short_pipeline && tiles <= 2 * int64_t(sms) && wm == wn && wn >= 64 && n % bn == 0) {
                // Short K and wide accumulators benefit from a narrower N
                // tile while retaining the same input-buffer footprint.
                const int input_box = (bm + bn) * stages;
                adapt_gemm_rs_n_grid(config, m, n, int(tiles + 1), true);
                stages = input_box / (bm + bn);
                smem = shared_size(stages);
            }
        }
        // Fit an unpadded owner-aligned grid in two resident CTAs per CU.
        // Keeping the M warp count and widening warp N reduces producer warps.
        const int rows = ceil_div(m, bm);
        if (m % bm != 0 && rows > 2 && local_m >= 2 * wm && bk % 128 == 0 && stages >= 3 &&
                n % bn == 0 && bn % (2 * wn) == 0 && m % (rows - 1) == 0) {
            const int next_bm = m / (rows - 1), m_warps = bm / wm;
            const int next_wm = next_bm / m_warps, next_wn = 2 * wn;
            const int next_bk = bk / 2, next_stages = stages + 1;
            const auto next_smem = deep_gemm_int8::get_smem_config(next_stages, k, next_bm, bn, next_bk, 1);
            if (next_bm % (m_warps * 16) == 0 && local_m % next_wm == 0 &&
                    k % next_bk == 0 && ceil_div(k, bk) <= 4 * next_stages &&
                    ceil_div(m, next_bm) * ceil_div(n, bn) <= 2 * sms &&
                    std::get<0>(next_smem) <= gemm_rs_smem_capacity() / 2 &&
                    deep_gemm_adaptive::tile_spill_ok(next_wm, next_wn, 2 * m_warps * (bn / next_wn))) {
                bm = next_bm;
                bk = next_bk;
                wm = next_wm;
                wn = next_wn;
                stages = next_stages;
                smem = next_smem;
            }
        }
        // Keep a complete M tile on the direct column tail. A narrower input
        // tile can expose more column CTAs while reducing shared-memory pressure.
        if (m <= bm && local_m % wm == 0 && wn % 64 == 0 && bk % 128 == 0 && stages >= 3) {
            const int narrow_bn = bn / 4 * 3, narrow_bk = bk / 2;
            const auto narrow_smem = deep_gemm_int8::get_smem_config(stages, k, bm, narrow_bn, narrow_bk, 1);
            const auto half_capacity = gemm_rs_smem_capacity() / 2;
            if (ceil_div(n, narrow_bn) <= sm_budget && std::get<0>(smem) > half_capacity &&
                std::get<0>(narrow_smem) <= half_capacity) {
                bn = narrow_bn;
                wn = wn / 4 * 3;
                bk = narrow_bk;
                smem = narrow_smem;
            }
        }
        // Expose more resident columns for a complete compact M tile. Keep
        // both the input pipeline depth and the M warp layout.
        if (world > 2 && owner_tail && local_m >= 2 * packet_rows && m >= wm && m <= bm &&
                wn >= 64 && wn % 32 == 0 && std::get<0>(smem) > gemm_rs_smem_capacity() / 2) {
            const int narrow_wn = wn / 2;
            const int narrow_bn = ceil_div(ceil_div(n, sm_budget), narrow_wn) * narrow_wn;
            const int warps = (bm / wm) * (narrow_bn / narrow_wn);
            const auto narrow_smem = deep_gemm_int8::get_smem_config(stages, k, bm, narrow_bn, bk, 1);
            if (ceil_div(n, narrow_bn) > ceil_div(n, bn) && warps >= 2 &&
                    warps * 32 <= at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock &&
                    std::get<0>(narrow_smem) <= gemm_rs_smem_capacity() / 2 &&
                    deep_gemm_adaptive::tile_spill_ok(wm, narrow_wn, warps)) {
                bn = narrow_bn;
                wn = narrow_wn;
                sms = sm_budget;
                smem = narrow_smem;
            }
        }
        // Pair N tiles when the original producers need a second wave.
        // A 16-warp CTA also budgets staged sender storage when applicable.
        const int wide_bn = 2 * bn, wide_wn = 2 * wn;
        const auto capacity = gemm_rs_smem_capacity();
        const int warps = (bm / wm) * (bn / wn);
        const int sender_smem = warps >= 16 && can_pipeline_channelwise_gemm_rs(config, m, world)
            ? get_gemm_rs_preferred_shared_packets(config) * warps * 32 * 16 : 0;
        if (m <= bm && local_m >= 16 && !owner_tail && wide_wn <= 64 && n % wide_bn == 0 &&
                std::get<0>(smem) + sender_smem > capacity / 2 &&
                ceil_div(n, bn) > sms && ceil_div(n, wide_bn) <= sms) {
            const int wide_stages = std::min(stages, int(capacity / ((bm + wide_bn) * bk)));
            if (wide_stages >= 3 && deep_gemm_adaptive::tile_spill_ok(wm, wide_wn, warps)) {
                const auto wide_smem = deep_gemm_int8::get_smem_config(wide_stages, k, bm, wide_bn, bk, 1);
                if (std::get<0>(wide_smem) <= capacity) {
                    bn = wide_bn;
                    wn = wide_wn;
                    stages = wide_stages;
                    smem = wide_smem;
                }
            }
        }
    }
    TORCH_CHECK(std::get<0>(smem) <= gemm_rs_smem_capacity(), "channelwise GEMM exceeds shared-memory capacity");
    return config;
}

} // namespace deep_gemm::gemm_comm
