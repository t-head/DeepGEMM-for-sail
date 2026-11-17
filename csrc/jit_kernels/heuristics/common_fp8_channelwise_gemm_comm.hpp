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
        // Narrow N only while it increases the number of MMA warps for padded rows.
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
        const auto shared_size = [&](int input_stages) {
            return deep_gemm_int8::get_smem_config(input_stages, k, bm, bn, bk, 1);
        };
        // Fit a single-M-warp N grid in one compute round.
        if (m <= bm && bm == wm)
            adapt_gemm_rs_stages(config, 3, ceil_div(ceil_div(n, bn), sms), 0, shared_size);
        if (can_pipeline_channelwise_gemm_rs(config, m, world)) {
            const int native_blocks = int(gemm_rs_smem_capacity() / std::get<0>(smem));
            const int64_t tiles = int64_t(ceil_div(m, bm)) * ceil_div(n, bn);
            const int packets = get_gemm_rs_preferred_shared_packets(config);
            const int threads = (bm / wm) * (bn / wn) * 32;
            if (native_blocks > 0 && tiles > int64_t(sms) * native_blocks && packets > 0)
                adapt_gemm_rs_stages(config, 3, native_blocks, int64_t(packets) * threads * 16, shared_size);
        }
    }
    TORCH_CHECK(std::get<0>(smem) <= gemm_rs_smem_capacity(), "channelwise GEMM exceeds shared-memory capacity");
    return config;
}

} // namespace deep_gemm::gemm_comm
