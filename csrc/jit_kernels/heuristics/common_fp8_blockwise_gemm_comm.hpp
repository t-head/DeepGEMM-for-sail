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
        static_cast<int64_t>(n) * k <= at::cuda::getCurrentDeviceProperties()->l2CacheSize) {
        // Keep a native tile that finishes in one CTA wave. Compact multi-source
        // traversal otherwise; avoid extra B reads
        // when the weights do not fit in L2 (also covers an unknown L2 capacity).
        // An owner-aligned warp exists only for a multiple of the 16-row MMA atom.
        int compact_wm = wm;
        while ((m / world) % compact_wm != 0)
            compact_wm -= 16;
        const int compact_bm = bm / wm * compact_wm;
        // Avoid extra B reads when a small owner would shrink to the minimum
        // MMA row atom merely to align warp boundaries; packet stores mask owners.
        const int64_t n_tiles = ceil_div(n, bn);
        const auto work_per_cta = [&](int rows) {
            const auto tiles = ceil_div(ceil_div(m, rows) * n_tiles, int64_t(sm_budget));
            // Padded output and A work scale with rows. Each tile also reloads B.
            return tiles * (2 * rows + bn);
        };
        // For two owners, keep the wider native tile when it reduces the
        // busiest CTA's work, with one fewer input stage for routing.
        if (world == 2 && stages == 4 && work_per_cta(bm) < work_per_cta(compact_bm)) {
            stages = 3;
        } else if (compact_wm > 16 || ceil_div(m, compact_bm) <= ceil_div(m, bm)) {
            bm = compact_bm;
            wm = compact_wm;
        }
    }
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
    const bool wide_mma = (bm / wm) * (bn / wn) > 8;
    const int local_m = m / world, packet_rows = get_gemm_rs_packet_rows(local_m);
    const bool paired_m_tiles = tune && wide_mma && world > 2 && m == 2 * bm && local_m <= wm &&
                                ceil_div(k, bk) <= 4 * stages;
    int narrow_stages = stages;
    bool resident_pair = false;
    if (paired_m_tiles) {
        const auto capacity = gemm_rs_smem_capacity() / 2;
        if (stages > 3 && ceil_div(k, bk) <= 4 * (stages - 1) &&
                std::get<0>(shared_size(bm, bn / 2, stages)) > capacity)
            --narrow_stages;
        resident_pair = std::get<0>(shared_size(bm, bn / 2, narrow_stages)) <= capacity;
    }
    const bool narrow_owner = world > 2 && local_m % packet_rows == 0 &&
                              ((m <= bm && packet_rows < 16) || resident_pair);
    if (tune && (!wide_mma || narrow_owner)) {
        // Use idle CUs without replacing an almost-full wave with a long
        // spillover wave. Two-M-tile producers can retain their warp N
        // width when the smaller input tiles fit together on a CU.
        const int tiles = ceil_div(m, bm) * ceil_div(n, bn);
        const int max_grid = wide_mma ? std::max(sm_budget, 2 * sm_budget - tiles)
                                     : std::numeric_limits<int>::max();
        const int original_bn = bn;
        adapt_gemm_rs_n_grid(config, m, n, sm_budget, false, is_blockwise_gemm_rs_n_tile, max_grid,
                             resident_pair);
        if (resident_pair && bn != original_bn)
            stages = narrow_stages;
    }
    // Refine N without adding a producer round. A partial owner pair must
    // fit in a single round and retain both M tiles after row compaction.
    const int refined_bn = bn / 4 * 3, refined_wn = wn / 4 * 3;
    const int refined_tiles = 2 * ceil_div(n, refined_bn);
    const int current_sms = native_tile == std::make_tuple(bm, bn, wm, wn) ? sms : sm_budget;
    const bool full_pair = m == 2 * bm && local_m % 16 == 0;
    const bool partial_pair = world > 2 && local_m >= 2 * packet_rows &&
                              2 * bm - packet_rows < m && m < 2 * bm && refined_tiles <= sm_budget;
    if (tune && (full_pair || partial_pair) && refined_bn == 192 && wn % 64 == 0 &&
            n % refined_bn == 0 && std::get<0>(shared_size(bm, bn, stages)) > gemm_rs_smem_capacity() / 2 &&
            ceil_div(refined_tiles, sm_budget) <= ceil_div(2 * ceil_div(n, bn), current_sms)) {
        bn = refined_bn;
        wn = refined_wn;
    }
    // A complete short owner tile can send native packets from one M
    // warp. Narrow N only when its larger producer grid stays resident.
    if (tune && world > 2 && local_m >= 3 && packet_rows == 4 && local_m % packet_rows != 0 &&
            wm <= m && m < bm && wn >= 32 && wn % 32 == 0 && is_blockwise_gemm_rs_n_tile(bn / 2)) {
        const int direct_bn = bn / 2, direct_wn = wn / 2, direct_stages = std::min(stages, 3);
        const int warps = direct_bn / direct_wn;
        const int tiles = ceil_div(n, direct_bn);
        const int resident_blocks = ceil_div(tiles, sm_budget);
        // Budget the spillover work at one additional warp per CU.
        if (warps >= 4 && tiles <= sm_budget + sm_budget / warps &&
                std::get<0>(shared_size(bm, direct_bn, direct_stages)) <= gemm_rs_smem_capacity() / resident_blocks &&
                deep_gemm_adaptive::tile_spill_ok(bm, direct_wn, warps * resident_blocks)) {
            bn = direct_bn;
            wm = bm;
            wn = direct_wn;
            stages = direct_stages;
        }
    }
    if (native_tile != std::make_tuple(bm, bn, wm, wn))
        sms = sm_budget; // A native CU budget was chosen for the original tile.
    validate_gemm_rs_tile({bm, bn, bk, wm, wn, stages});
    TORCH_CHECK(bk == 128, "blockwise FP8 GEMM/RS requires BK128");
    TORCH_CHECK(is_blockwise_gemm_rs_n_tile(bn),
                "blockwise BN must divide 128, be a multiple of 128, or equal 192");
    TORCH_CHECK(world != 1 || bn != 192, "TP1 native blockwise GEMM does not support BN192 weight-scale refinement");
    TORCH_CHECK(bn != 192 || 64 % (16 * (bn / wn)) == 0,
                "BN192 N64 scales require the tiled MMA N atom to divide 64; choose a compatible WN such as 48 or 96");
    smem = shared_size(bm, bn, stages);
    if (tune && m > bm)
        adapt_gemm_rs_m_tile(config, m, n, world, sm_budget, 4, shared_size);
    if (tune && world > 2 && m <= bm && local_m >= 2 * packet_rows && local_m % packet_rows != 0 &&
            m > bm / 2 && bm % (2 * wm) == 0 && bn == 128 && wn % 32 == 0 &&
            ceil_div(ceil_div(n, bn), sms) == 2) {
        // Split a complete M tile when both grids need two producer rounds.
        // The smaller input tile leaves space for an extra prefetch stage.
        constexpr int split_bn = 192;
        const int split_bm = bm / 2;
        const auto split_smem = shared_size(split_bm, split_bn, stages + 1);
        if (split_bm * (bn / wn) >= 4 * wm && split_bm + split_bn < bm + bn &&
                ceil_div(2 * ceil_div(n, split_bn), sm_budget) == 2 &&
                std::get<0>(split_smem) <= gemm_rs_smem_capacity()) {
            bm = split_bm;
            bn = split_bn;
            wn = wn / 2 * 3;
            ++stages;
            smem = split_smem;
            sms = sm_budget;
        }
    }
    if (world > 2 && is_ppu1v5_device() && m <= bm && m >= wm && local_m >= 3 && local_m % packet_rows != 0)
        std::get<0>(smem) = std::max(std::get<0>(smem), bm * bn * 2);
    TORCH_CHECK(std::get<0>(smem) <= gemm_rs_smem_capacity(), "blockwise GEMM scales exceed shared-memory capacity");
    return config;
}

} // namespace deep_gemm::gemm_comm
