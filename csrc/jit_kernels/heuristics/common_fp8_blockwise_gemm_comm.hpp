#pragma once

#include "common_fp8.hpp"
#include "common_gemm_comm.hpp"

namespace deep_gemm::gemm_comm {

inline Fp8GemmRSConfig get_blockwise_gemm_rs_config(int m, int n, int k, int world, std::optional<ConfigTuple> configs,
                                                    const std::string& tile_policy) {
    auto tile = get_gemm_rs_tile(configs);
    int sm_budget = configs ? std::get<0>(*configs) : get_num_sms();
    validate_gemm_rs_shape(m, n, k, world);
    TORCH_CHECK(tile_policy == "rs_aware" || tile_policy == "native", "blockwise tile_policy must be rs_aware or native");
    const bool tune = !configs && world > 1 && tile_policy == "rs_aware" && is_ppu1v5_device();
    ConfigTuple config;
    if (tile) {
        const auto [bm, bn, bk, wm, wn, stages] = *tile;
        validate_gemm_rs_tile(*tile);
        config = {sm_budget, bm, bn, bk, wm, wn, stages, {}};
    } else {
        config = deep_gemm_fp8_common::get_best_configs(m, n, k, 1, sm_budget);
    }
    auto& [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    validate_gemm_rs_tile({bm, bn, bk, wm, wn, stages});
    const auto native_tile = std::make_tuple(bm, bn, wm, wn);
    if (tune && m < bm) {
        // Avoid computing a mostly padded M tile. Preserve memory parallelism
        // after clipping, with at least eight warps where the native N tile permits.
        int rows = 16;
        while (rows < m)
            rows *= 2;
        bm = std::min(bm, rows);
        wm = std::min(wm, bm);
        wn = std::min(wn, std::max(16, ((bm / wm) * bn / 8) / 16 * 16));
    }
    // Long owners and K loops amortize packet routing without shrinking the
    // native compute tile. Other cases retain compact owner-aligned warps.
    const bool preserve_native_tile = m / world >= 16 * bm && k > 2 * stages * bk;
    if (tune && m > bm && !preserve_native_tile &&
        ceil_div(m, bm) > sms / ceil_div(n, bn) && (m / world) % 16 == 0 && (m / world) % wm != 0 &&
        (world > 2 || static_cast<int64_t>(n) * k <= at::cuda::getCurrentDeviceProperties()->l2CacheSize)) {
        // Keep a native tile that finishes in one CTA wave. Compact multi-source
        // traversal otherwise; for the two-source packed sum, avoid extra B reads
        // when the weights do not fit in L2 (also covers an unknown L2 capacity).
        // An owner-aligned warp exists only for a multiple of the 16-row MMA atom.
        int compact_wm = wm;
        while ((m / world) % compact_wm != 0)
            compact_wm -= 16;
        const int compact_bm = bm / wm * compact_wm;
        // Avoid extra B reads when a small owner would shrink to the minimum
        // MMA row atom merely to align warp boundaries; packet stores mask owners.
        if (compact_wm > 16 || m / world >= bm || ceil_div(m, compact_bm) <= ceil_div(m, bm)) {
            bm = compact_bm;
            wm = compact_wm;
        }
    }
    if (tune && (bm / wm) * (bn / wn) <= 8)
        adapt_gemm_rs_n_grid(config, m, n, sm_budget, false, is_blockwise_gemm_rs_n_tile);
    if (native_tile != std::make_tuple(bm, bn, wm, wn))
        sms = sm_budget; // A native CU budget was chosen for the original tile.
    validate_gemm_rs_tile({bm, bn, bk, wm, wn, stages});
    TORCH_CHECK(bk == 128, "blockwise FP8 GEMM/RS requires BK128");
    TORCH_CHECK(is_blockwise_gemm_rs_n_tile(bn),
                "blockwise BN must divide 128, be a multiple of 128, or equal 192");
    TORCH_CHECK(world != 1 || bn != 192, "TP1 native blockwise GEMM does not support BN192 weight-scale refinement");
    TORCH_CHECK(bn != 192 || 64 % (16 * (bn / wn)) == 0,
                "BN192 N64 scales require the tiled MMA N atom to divide 64; choose a compatible WN such as 48 or 96");
    const auto shared_size = [&](int rows, int columns, int input_stages) {
        auto selected = deep_gemm_fp8_common::get_smem_config(input_stages, k, rows, columns, bk);
        if (columns == 192) {
            // BN192 refines the two N128 scale slots into three N64 slots.
            const int slot = ceil_div(bk, 128) * sizeof(float);
            std::get<0>(selected) += ceil_div(input_stages * 3 * slot, 128) * 128 -
                                    ceil_div(input_stages * 2 * slot, 128) * 128;
        }
        return selected;
    };
    smem = shared_size(bm, bn, stages);
    if (tune && m > bm)
        adapt_gemm_rs_m_tile(config, m, n, world, sm_budget, 4, shared_size);
    const int local_m = m / world, packet_rows = get_gemm_rs_packet_rows(local_m);
    if (world > 2 && is_ppu1v5_device() && m <= bm && m >= wm && local_m >= 3 && local_m % packet_rows != 0)
        std::get<0>(smem) = std::max(std::get<0>(smem), bm * bn * 2);
    TORCH_CHECK(std::get<0>(smem) <= gemm_rs_smem_capacity(), "blockwise GEMM scales exceed shared-memory capacity");
    return config;
}

} // namespace deep_gemm::gemm_comm
