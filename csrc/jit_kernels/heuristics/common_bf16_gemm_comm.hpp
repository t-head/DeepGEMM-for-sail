#pragma once

#include "common_bf16.hpp"
#include "bf16_memory_policy.hpp"
#include "common_gemm_comm.hpp"

namespace deep_gemm::gemm_comm {
using Bf16GemmRSConfig = std::pair<ConfigTuple, int>;

inline bool can_pipeline_bf16_gemm_rs(const ConfigTuple& config, int m, int n, int k, int world) {
    const auto& [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    const int local_m = m / world;
    const bool native_owners = m % bm == 0 && local_m % wm == 0;
    // Preserve staged sends when rebasing keeps whole owner tiles, or
    // when the owner has enough full tiles to amortize crossing boundaries.
    const int owner_rows = local_m / bm;
    // A few full owner waves can overlap a partial tail while streaming B.
    // Larger shards retain their original reduction and input pipeline.
    const bool partial_two_source_tail = world == 2 && owner_rows >= stages && 2 * owner_rows < sms &&
        local_m % 16 != 0 && local_m % bm < 16 &&
        int64_t(n) * k * sizeof(cutlass::bfloat16_t) > 4LL * at::cuda::getCurrentDeviceProperties()->l2CacheSize;
    const bool multiple_owner_tiles = partial_two_source_tail ||
        (world > 2 && (owner_rows >= 2 || can_align_owner_rows(m, world, bm)));
    return world > 1 && local_m >= 16 && (native_owners || multiple_owner_tiles) &&
           int64_t(ceil_div(m, bm)) * ceil_div(n, bn) > sms;
}

inline int get_bf16_gemm_rs_packet_capacity(const ConfigTuple& config, int64_t capacity) {
    const auto& [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    const int output_threads = (bm / wm) * (bn / wn) * 32;
    return std::min(wm * wn / 256, int((capacity - std::get<0>(smem)) / (output_threads * 16))) / 2 * 2;
}

// A native warp rectangle has one reduce-scatter owner. Use subsequent
// native work as the sender progress window.
inline int get_bf16_gemm_rs_shared_packets(const ConfigTuple& config, int m, int n, int k, int world) {
    return can_pipeline_bf16_gemm_rs(config, m, n, k, world)
        ? get_bf16_gemm_rs_packet_capacity(config, gemm_rs_smem_capacity()) : 0;
}

// Both input depth and shard scheduling use complete owner waves.
inline bool has_bf16_gemm_rs_owner_waves(const ConfigTuple& config, int local_m, int n, int input_stages) {
    const auto& [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    const int rows = local_m / bm;
    return local_m % bm == 0 && n % bn == 0 && 2 * rows >= sms && rows >= input_stages &&
           int64_t(rows) * (n / bn) >= int64_t(input_stages) * sms;
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
    const int budget = get_num_sms();
    int warp_k = 0;
    auto config = configs ? *configs
                          : deep_gemm_bf16_common::get_best_configs(m, n, k, 1, budget, GemmType::Normal, 256,
                                                                    nullptr, &warp_k);
    auto &[sms, bm, bn, bk, wm, wn, stages, smem] = config;
    validate_bf16_gemm_rs_config(config, false);
    if (warp_k == 0)
        warp_k = bk;
    TORCH_CHECK(warp_k >= 64 && warp_k % 64 == 0 && bk % warp_k == 0, "invalid BF16 GEMM/RS config");
    const GemmRSSmemBudget memory;
    const int64_t weight_bytes = int64_t(n) * k * sizeof(cutlass::bfloat16_t);
    const int64_t l2 = at::cuda::getCurrentDeviceProperties()->l2CacheSize;
    const bool cached_weights = weight_bytes <= l2;
    const auto shared_size = [k](int depth, int rows, int columns, int input_k) {
        return deep_gemm_bf16_common::get_smem_config(depth, k, rows, columns, input_k, 2);
    };
    const auto current_size = [&](int depth) { return shared_size(depth, bm, bn, bk); };
    smem = current_size(stages);
    const auto finish = [&] {
        TORCH_CHECK(memory.fits(smem), "BF16 GEMM/RS shared-memory capacity exceeded");
        validate_bf16_gemm_rs_config(config);
        return Bf16GemmRSConfig{config, warp_k};
    };
    if (configs || world == 1 || tile_policy == "native" || !is_ppu1v5_device())
        return finish();

    const int local_m = m / world, packet_rows = get_gemm_rs_packet_rows(local_m);
    const bool owner_tail = local_m >= 3 && local_m % packet_rows != 0;
    const bool compact_owner = !owner_tail && local_m <= packet_rows && m <= wm && bm == wm;
    if (compact_owner && (local_m < 16 || cached_weights)) {
        // A complete narrow output can go directly to the epilogue.
        warp_k = bk;
        const bool deepen_input =
            (stages == 2 && bk >= 2 * 128 && k >= bk && ceil_div(k, 128) <= 8 * stages) ||
            (stages == 3 && bk == 2 * 128 && ceil_div(k, bk) <= 8 * stages);
        if (deepen_input) {
            bk = warp_k = 128;
            stages = 4;
            smem = current_size(stages);
        }
    }
    if (owner_tail && m <= wm && bm == wm && warp_k == bk &&
            cached_weights) {
        const int narrow_bn = std::min(bn, 4 * wn);
        // When startup/drain spans most K work, use smaller K tiles to create
        // a useful prefetch window. Longer pipelines retain the native K tile.
        const bool short_pipeline = ceil_div(k, bk) < 2 * stages;
        const int input_stages = short_pipeline ? 3 : 2;
        int narrow_bk = bk;
        auto candidate = shared_size(input_stages, bm, narrow_bn, narrow_bk);
        while (short_pipeline && narrow_bk > 64 && !memory.fits(candidate, 2)) {
            narrow_bk /= 2;
            candidate = shared_size(input_stages, bm, narrow_bn, narrow_bk);
        }
        // Keep every output tile resident. Otherwise the narrow grid adds a
        // second compute round to the tail instead of hiding communication.
        const int resident_tiles = sms * memory.blocks(candidate);
        if ((short_pipeline || stages > 2) && memory.fits(candidate, 2) &&
                ceil_div(n, narrow_bn) <= resident_tiles &&
                (!short_pipeline || k >= input_stages * narrow_bk)) {
            config = {short_pipeline ? budget : sms, bm, narrow_bn, narrow_bk, wm, wn, input_stages, candidate};
            warp_k = narrow_bk;
        }
    }
    // Preserve the established column grid when doubling input depth
    // for boundary packets with two sources.
    if (world == 2 && owner_tail && packet_rows == 4 && m <= wm && bm == wm &&
            stages == 2 && bk == 2 * warp_k && warp_k >= 2 * 128 && k >= bk) {
        bk = warp_k;
        stages *= 2;
        smem = current_size(stages);
    }
    if (can_pipeline_bf16_gemm_rs(config, m, n, k, world)) {
        // Keep input depth for sustained streaming weights when owner
        // tiles cannot be isolated by rebasing.
        const bool full_owner_columns = local_m / bm > 2 * stages &&
            ceil_div(n, bn) * (stages + 1) >= sms * stages &&
            weight_bytes < int64_t(world) * stages * l2;
        const bool keep_input_depth = (world > 2 || full_owner_columns) &&
            owner_tail && local_m / bm >= stages && ceil_div(k, bk) > 4 * stages && !cached_weights &&
            !can_align_owner_rows(m, world, bm);
        if (!keep_input_depth && get_bf16_gemm_rs_packet_capacity(config, memory.capacity) == 0) {
            const int threads = (bm / wm) * (bn / wn) * 32;
            const int packets = get_gemm_rs_preferred_shared_packets(config);
            const int64_t per_stage = int64_t(bm + bn) * bk * 2;
            const int64_t reduction_smem = int64_t(bk / warp_k - 1) * bm * bn * 4;
            const int minimum_stages = std::max(2, int((reduction_smem + per_stage - 1) / per_stage));
            if (packets > 0)
                adapt_gemm_rs_stages(config, minimum_stages, 1, int64_t(packets) * threads * 16, current_size);
        }
        // Give the staged sender subsequent tile work without growing the
        // input box already balanced against its sender storage.
        const int64_t input_box = int64_t(bm + bn) * stages;
        // A grid below two rounds may add at most one sender-progress round.
        adapt_gemm_rs_n_grid(config, m, n, 2 * sms, false, nullptr, 3 * sms);
        stages = int(input_box / (bm + bn));
        smem = current_size(stages);
    }
    // A streaming partial owner needs both its weight column grid
    // and enough M warps to keep the input load and output work balanced.
    if (world >= 8 && owner_tail && packet_rows == 4 && m <= bm && bm == wm &&
            wm >= 3 * 16 && wn <= 2 * 16 && bk == warp_k && stages <= 3 &&
            int64_t(n) * k >= l2) {
        constexpr int row_warp = 32;
        const int rows = ceil_div(m, 2 * row_warp) * 2 * row_warp;
        for (const auto [column_warp, narrow_input] : {std::pair{64, false}, std::pair{96, false},
                                                      std::pair{80, true}, std::pair{64, true}}) {
            // Preserve a full native M tile when the alternate N warp
            // would introduce padded rows. A wide streaming grid can
            // retain two input stages when a wider input box cannot fit.
            if ((column_warp > 64 && m == bm && rows > bm) ||
                    (narrow_input && m == bm))
                continue;
            const int alignment = column_warp == 64 && !narrow_input ? 2 * column_warp : column_warp;
            const int columns = ceil_div(ceil_div(n, budget), alignment) * alignment;
            const int warps = rows / row_warp * (columns / column_warp);
            if (warps > (narrow_input ? 3 : 2) * deep_gemm_adaptive::WE_PER_CU ||
                    !deep_gemm_adaptive::tile_spill_ok(row_warp, column_warp, warps))
                continue;
            int input_k = narrow_input ? 64 : 128, input_stages = 2;
            auto bytes = shared_size(input_stages, rows, columns, input_k);
            if (!narrow_input && !memory.fits(bytes)) {
                input_k /= 2;
                ++input_stages;
                bytes = shared_size(input_stages, rows, columns, input_k);
            }
            if (memory.fits(bytes) && k % input_k == 0) {
                config = {budget, rows, columns, input_k, row_warp, column_warp, input_stages, bytes};
                warp_k = input_k;
                break;
            }
        }
    }
    // A short K loop and a padded owner leave room for a resident
    // split-M grid with smaller input slices.
    if (owner_tail && world > 2 && local_m < 2 * packet_rows && wm < m && m < bm &&
            bm == 2 * wm && wm % 32 == 0 && warp_k == bk && bk >= 256 && stages == 2 &&
            ceil_div(k, bk) <= 2 * stages) {
        const int next_bn = bn / 64 * 64, next_wm = wm / 2, next_wn = std::max(16, wn / 2);
        const int next_bk = bk / 2, next_stages = stages + 1;
        if (next_bn >= next_wn && next_bn % next_wn == 0 && next_bn < bn) {
            const auto next_smem = shared_size(next_stages, wm, next_bn, next_bk);
            const int blocks = memory.blocks(next_smem);
            const int warps = 2 * next_bn / next_wn;
            const int tiles = ceil_div(m, wm) * ceil_div(n, next_bn);
            if (blocks >= 2 && tiles <= 2 * budget + budget / warps &&
                    deep_gemm_adaptive::tile_spill_ok(next_wm, next_wn, 2 * warps)) {
                config = {budget, wm, next_bn, next_bk, next_wm, next_wn, next_stages, next_smem};
                warp_k = next_bk;
            }
        }
    }
    // Replace inter-warp K accumulation with two M warps and a
    // deeper input pipeline when a partial owner fits a single tile.
    if (owner_tail && world > 2 && local_m >= 16 && m < wm && wm == bm &&
            bk == 256 && warp_k == 128 && stages == 2) {
        const int rows = ceil_div(m, 32) * 32;
        const int columns = ceil_div(ceil_div(n, budget), 64) * 64;
        const int next_stages = 2 * stages;
        const auto next_smem = shared_size(next_stages, rows, columns, warp_k);
        const int warps = 2 * (columns / wn);
        if (columns % wn == 0 && warps <= 2 * deep_gemm_adaptive::WE_PER_CU &&
                4 * ceil_div(n, columns) >= 3 * budget &&
                memory.fits_warps(next_smem, 1, rows / 2, wn, warps)) {
            config = {budget, rows, columns, warp_k, rows / 2, wn, next_stages, next_smem};
        }
    }
    if (owner_tail && m >= wm) {
        int native_warps_n = bn / wn;
        if (m <= bm) {
            // Keep a whole-M tile in one wave with aligned transfer columns.
            const int narrow_bn = ceil_div(ceil_div(n, budget), 64) * 64;
            while (narrow_bn % (native_warps_n * 16) != 0)
                --native_warps_n;
            const int narrow_wn = narrow_bn / native_warps_n;
            const int warps = (bm / wm) * native_warps_n * (bk / warp_k);
            // Widen N only when it reduces the number of N warps.
            // Otherwise preserve the native geometry and input pipeline.
            if (narrow_bn != bn && (narrow_bn < bn || (stages > 3 && native_warps_n < bn / wn)) &&
                    ceil_div(n, narrow_bn) <= budget &&
                    deep_gemm_adaptive::tile_spill_ok(wm, narrow_wn, warps)) {
                const auto candidate = shared_size(std::min(stages, 3), bm, narrow_bn, bk);
                const int64_t reduction_smem = int64_t(bk / warp_k - 1) * bm * narrow_bn * 4;
                if (memory.fits(candidate) &&
                        int64_t(bm) * narrow_bn * 2 <= memory.capacity &&
                        reduction_smem <= std::get<0>(candidate)) {
                    config = {budget, bm, narrow_bn, bk, wm, narrow_wn, std::min(stages, 3), candidate};
                }
            }
        } else if (world > 2 && int64_t(ceil_div(m, bm)) * ceil_div(n, bn) <= sms &&
                   cached_weights) {
            // Resident weights can sustain a second tile wave for communication.
            const int native_bn = bn;
            adapt_gemm_rs_n_grid(config, m, n, 2 * sms, false);
            if (bn != native_bn) {
                stages = std::min(stages, 3);
                sms = budget;
                smem = current_size(stages);
            }
        }
    }
    // Keep a warp's complete packets in one row owner on the direct path.
    // Trading M for N retains both the accumulator area and CTA warp count.
    if (m == bm && wm < bm && local_m % 16 == 0 && wm == 2 * local_m && warp_k == bk &&
            bn % (2 * wn) == 0 && ceil_div(n, bn) <= sms &&
            deep_gemm_adaptive::tile_spill_ok(wm / 2, 2 * wn, (bm / wm) * (bn / wn))) {
        const int tiles = ceil_div(n, bn);
        // Splitting the M tile retains its N width for the staged sender.
        const ConfigTuple split{budget, bm / 2, bn, bk, wm / 2, wn, stages, shared_size(stages, bm / 2, bn, bk)};
        // Budget both input traversals when splitting a complete owner tile.
        const int64_t cache_elements = l2 / (2 * sizeof(cutlass::bfloat16_t));
        if (world > 2 && bm / wm >= 4 && wm / 2 > 2 * 16 && 2 * tiles > budget && tiles <= budget &&
                int64_t(n) * k < cache_elements &&
                get_bf16_gemm_rs_shared_packets(split, m, n, k, world) > 0) {
            config = split;
        } else {
            wm /= 2;
            wn *= 2;
        }
    }
    if (wm == wn && wn % 64 == 0 && bm % (2 * wm) == 0) {
        const bool staged_sender = (local_m % (2 * wm) == 0 || can_align_owner_rows(m, world, bm)) &&
            get_bf16_gemm_rs_shared_packets(config, m, n, k, world) > 0;
        const int64_t tiles = int64_t(ceil_div(m, bm)) * ceil_div(n, bn);
        const bool short_owner_tail = owner_tail && m > bm && warp_k == bk && stages == 4 &&
            tiles <= int64_t(sms) * stages && ceil_div(k, bk) <= 8 * stages &&
            deep_gemm_adaptive::tile_spill_ok(2 * wm, wn / 2, (bm / wm) * (bn / wn));
        // Keep the accumulator area while exposing more N warps. A short
        // tail grid uses the same orientation with a shallower pipeline.
        // Full producer waves per owner amortize one more input stage.
        // Retain balanced warps when the streaming weight footprint is
        // large enough to keep that deeper input window useful.
        const bool long_owner = staged_sender && world == 2 && warp_k == bk && stages == 3 &&
            has_bf16_gemm_rs_owner_waves(config, local_m, n, stages + 1) &&
            int64_t(n) * k > l2;
        const auto deeper_smem = long_owner ?
            current_size(stages + 1) : smem;
        if (long_owner && memory.fits(deeper_smem)) {
            ++stages;
            smem = deeper_smem;
        } else if (staged_sender || short_owner_tail) {
            wm *= 2;
            wn /= 2;
            if (short_owner_tail) {
                --stages;
                smem = current_size(stages);
            }
        }
    }
    if (owner_tail && world > 2 && packet_rows == 4 && m < wm && bm == wm && ceil_div(n, bn) <= sms) {
        // A full column wave with complete K accumulators already
        // occupies each CU. Keep the native K tile for that grid.
        const bool full_native_wave = world == 4 && warp_k == bk && ceil_div(n, bn) == sms;
        const int next_bk = ceil_div(k, 128) <= 8 * stages ? 128 : bk / 2;
        if (!full_native_wave && stages == 2 && bk >= 2 * 128 && k >= bk && k % next_bk == 0 &&
                (next_bk == 128 || warp_k < bk) && ceil_div(k, bk) <= 4 * stages) {
            bk = next_bk;
            stages *= 2;
            smem = current_size(stages);
        }
        warp_k = bk;
    }
    // Trim a complete compact M tile when one more M warp reduces
    // padding and accumulator pressure while preserving the column grid.
    if (owner_tail && world > 2 && wm <= m && m <= bm && bm > wm && warp_k == bk &&
            ceil_div(n, bn) <= sms) {
        const int m_warps = bm / wm + 1;
        const int compact_bm = ceil_div(m, m_warps * 16) * m_warps * 16;
        const int compact_wm = compact_bm / m_warps;
        const int warps = m_warps * (bn / wn);
        if (compact_bm < bm && warps <= 2 * deep_gemm_adaptive::WE_PER_CU &&
                deep_gemm_adaptive::tile_spill_ok(compact_wm, wn, warps)) {
            bm = compact_bm;
            wm = compact_wm;
            smem = current_size(stages);
        }
    }
    if (world >= 8 && local_m < 3 && m <= bm && bm == wm && warp_k == bk) {
        if (bk == 64 && (stages == 2 || std::get<0>(smem) * 8 >= 7 * memory.capacity) &&
                weight_bytes <= 4 * l2) {
            // With several LLCs of weights, keep the wider streaming tile.
            const int columns = 4 * wn, depth = 2;
            const int blocks = gemm_rs_column_rounds(n, columns, budget);
            const auto bytes = shared_size(depth, bm, columns, bk);
            if (columns < bn && memory.fits(bytes, blocks) &&
                    4 * blocks * 32 <= at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor)
                config = {budget, bm, columns, bk, wm, wn, depth, bytes};
        } else if (bk == 128 && wn == 16 && bn > 64 && stages == 4) {
            // Use spare CUs to retire a complete bulk-word group per warp.
            constexpr int columns = 64;
            const int tiles = ceil_div(n, bn);
            const auto bytes = shared_size(stages, bm, columns, bk);
            const bool underfilled = local_m == 1 && is_single_owner_packet(world, local_m, bn)
                ? 8 * tiles <= 7 * budget : tiles < budget;
            if (underfilled && ceil_div(n, columns) <= memory.blocks(bytes) * budget)
                config = {budget, bm, columns, bk, wm, wn, stages, bytes};
        }
    }
    // Pack each partial owner into three bulk words without adding a
    // producer wave. Narrow K only if enough of the input pipeline remains.
    if (world >= 8 && local_m == 3 && m <= bm && bm == 2 * 16 && k % 64 == 0) {
        constexpr int next_bn = 64, max_stages = 4;
        const int columns = ceil_div(n, next_bn);
        const int blocks = ceil_div(columns, budget);
        if (columns >= budget / 2) {
            for (int next_bk : {128, 64}) {
                if (k % next_bk != 0 || k < max_stages * next_bk)
                    continue;
                const int next_wn = next_bk == 64 && blocks < max_stages ? 32 : 16;
                const int min_stages = next_bk == 128 && k / next_bk <= 3 * max_stages ? 3 : max_stages;
                for (int next_stages = max_stages; next_stages >= min_stages; --next_stages) {
                    const auto next_smem = shared_size(next_stages, bm, next_bn, next_bk);
                    if (memory.fits_warps(next_smem, blocks, bm, next_wn, (next_bn / next_wn) * blocks)) {
                        const ConfigTuple candidate{budget, bm, next_bn, next_bk, bm, next_wn, next_stages, next_smem};
                        validate_bf16_gemm_rs_config(candidate);
                        return {candidate, next_bk};
                    }
                }
            }
        }
    }
    // Halve the M tile when two resident producers can preserve the
    // original waves while using a deeper input pipeline for ragged owners.
    if (world > 2 && owner_tail && m > bm && wm <= local_m && 2 * local_m < bm &&
            bm >= 4 * wm && bm % (2 * wm) == 0 && warp_k == bk && stages == 3 &&
            ceil_div(k, bk) <= 4 * (stages + 1) && !memory.fits(smem, 2)) {
        const int rows = ceil_div(m, bm), next_bm = bm / 2;
        const int next_rows = ceil_div(m, next_bm), columns = ceil_div(n, bn);
        const auto next_smem = shared_size(stages + 1, next_bm, bn, bk);
        const int next_warps = (next_bm / wm) * (bn / wn);
        const int waves = ceil_div(rows * columns, sms);
        const int tail = rows * columns - (waves - 1) * sms;
        // Keep wider tiles when the last producer wave is already busy.
        if (waves > 1 && 3 * tail <= 2 * sms && next_rows * next_bm <= rows * bm &&
                ceil_div(next_rows * columns, 2 * budget) <= waves &&
                gemm_rs_resident_blocks(wm, wn, next_warps, std::get<0>(next_smem)) >= 2) {
            config = {budget, next_bm, bn, bk, wm, wn, stages + 1, next_smem};
        }
    }
    // Use otherwise idle CUs to retire smaller square accumulators.
    // A second producer round is useful only if it lowers the work per CU.
    if (world > 2 && m > bm && m % bm == 0 && local_m >= wm && local_m < bm && bm == bn &&
            wm == wn && wm % 64 == 0 && warp_k == bk && stages >= 4 &&
            ceil_div(k, bk) <= (cached_weights ? 8 : 4) * stages &&
            8 * std::get<0>(smem) >= 7 * memory.capacity) {
        const int next_warp = wm / 4 * 3, next_tile = bm / 4 * 3;
        const int tiles = (m / bm) * ceil_div(n, bn);
        const int next_tiles = ceil_div(m, next_tile) * ceil_div(n, next_tile);
        const auto next_smem = shared_size(stages, next_tile, next_tile, bk);
        const int64_t next_work = int64_t(next_tiles) * next_tile * next_tile * std::min(sms, tiles);
        const int64_t native_work = int64_t(tiles) * bm * bn * budget;
        const int warps = (next_tile / next_warp) * (next_tile / next_warp);
        if (m % next_tile == 0 && tiles < budget && next_tiles <= 2 * budget && next_work < native_work &&
                memory.fits_warps(next_smem, 1, next_warp, next_warp, warps)) {
            config = {budget, next_tile, next_tile, bk, next_warp, next_warp, stages, next_smem};
        }
    }
    // A ragged owner with multiple row tiles can reduce column work on
    // the slowest CU without changing the number of warps per producer.
    if (world > 2 && owner_tail && local_m >= bm && !can_align_owner_rows(m, world, bm) &&
            warp_k == bk && m > bm && ceil_div(m, bm) <= budget && stages >= 3 && wn % 64 == 0) {
        const int next_bn = bn / 4 * 3, next_wn = wn / 4 * 3;
        const int rows = ceil_div(m, bm);
        const int warps = (bm / wm) * (bn / wn);
        const auto next_smem = shared_size(3, bm, next_bn, bk);
        const int64_t capacity = memory.capacity;
        const auto work = [&](int columns, int sms, int64_t bytes) {
            const int resident = std::max(1, std::min(int(capacity / bytes),
                3 * deep_gemm_adaptive::WE_PER_CU / warps));
            return ceil_div(rows * ceil_div(n, columns), sms * resident) * int64_t(columns);
        };
        if (n % next_bn == 0 && std::get<0>(next_smem) <= capacity &&
                work(next_bn, budget, std::get<0>(next_smem)) + (bn - next_bn) <= work(bn, sms, std::get<0>(smem)) &&
                deep_gemm_adaptive::tile_spill_ok(wm, next_wn, warps)) {
            config = {budget, bm, next_bn, bk, wm, next_wn, 3, next_smem};
        }
    }
    // A full column wave with wide N tiles benefits from shorter K slices
    // even when it does not need a second resident producer.
    const int input_tiles = ceil_div(k, bk);
    const bool full_column_wave = world >= 8 && local_m == packet_rows && bn > bm &&
                                  ceil_div(n, bn) == sms && bk == 2 * 64 &&
                                  input_tiles >= 2 * stages && input_tiles <= 4 * stages;
    // A narrower input tile can also make two whole-M producers resident.
    if (world > 2 && m == bm && local_m >= 16 && local_m % 16 == 0 &&
            (full_column_wave || bk >= 4 * 64) &&
            2 * std::get<0>(smem) > memory.capacity) {
        constexpr int input_k = 64, depth = 4;
        const auto bytes = shared_size(depth, bm, bn, input_k);
        if (k % input_k == 0 && ceil_div(k, input_k) <= 12 * depth &&
                memory.fits(bytes, full_column_wave ? 1 : 2)) {
            bk = warp_k = input_k;
            stages = depth;
            smem = bytes;
        }
    }
    // A long row-owner grid reuses each weight tile across more rows.
    // Keep the three-stage sender when widening a partial streaming shard.
    const bool partial_streaming = world == 2 && owner_tail && can_pipeline_bf16_gemm_rs(config, m, n, k, world);
    if (world == 2 && wm == wn && wm % 64 == 0 && n % bn == 0 &&
            (local_m / bm >= sms || partial_streaming) &&
            (stages == 4 || (partial_streaming && stages == 3)) &&
            warp_k == bk && input_tiles >= 8 * stages) {
        const int extended_wm = wm + 16, extended_bm = (bm / wm) * extended_wm;
        const int warps = (bm / wm) * (bn / wn), depth = 3;
        const auto bytes = shared_size(depth, extended_bm, bn, bk);
        if (memory.fits_warps(bytes, 1, extended_wm, wn, warps)) {
            bm = extended_bm;
            wm = extended_wm;
            stages = depth;
            smem = bytes;
        }
    }
    // A full short shard already fits in one column wave. Narrow the K
    // slices so an independent consumer warp can retain the native packet.
    if (world == 2 && m == bm && bm == wm && local_m == packet_rows &&
            warp_k == bk && bk >= 4 * 64 && stages == 2 && ceil_div(n, bn) <= sms) {
        constexpr int input_k = 64, depth = 6;
        const auto bytes = shared_size(depth, bm, bn, input_k);
        if (k % input_k == 0 && memory.fits(bytes, 2)) {
            bk = warp_k = input_k;
            stages = depth;
            smem = bytes;
        }
    }

    // A narrow input slice sustains a short complete owner's column grid.
    if (world > 2 && m == bm && bm == wm && bm <= 2 * 16 && !owner_tail &&
            local_m <= packet_rows && warp_k == bk && bk >= 128 && k % 64 == 0) {
        constexpr int input_k = 64;
        const int depth = std::clamp(stages * (bk / input_k), 6, 8);
        int columns = bn;
        if (world >= 8 && local_m <= 2 && ceil_div(k, input_k) <= 4 * depth) {
            // Keep a complete owner in one bulk payload and each N tile
            // on a separate CU when the input is already short.
            const int packet_columns = ceil_div(ceil_div(n, sms), 32) * 32;
            if (n % packet_columns == 0 && packet_columns % wn == 0 &&
                    is_single_owner_packet(world, local_m, packet_columns))
                columns = packet_columns;
        }
        const auto bytes = shared_size(depth, bm, columns, input_k);
        const int warps = columns / wn;
        const int blocks = gemm_rs_resident_blocks(wm, wn, warps, std::get<0>(bytes));
        if (blocks > 0 &&
                ceil_div(n, columns) <= blocks * budget) {
            config = {budget, bm, columns, input_k, wm, wn, depth, bytes};
            warp_k = input_k;
        }
    }
    return finish();
}

struct Bf16GemmRSKernelConfig {
    int threads = 0;
    int shared_packets = 0;
    int smem_size = 0;
    bool input_overlap = false;
    int b_cache_policy = 1;
    int compact_min_m = 0;
    bool nonpersistent = false;
    bool split_owner_shards = false;
};

inline Bf16GemmRSKernelConfig get_bf16_gemm_rs_kernel_config(const Bf16GemmRSConfig& selected, int m, int n, int k, int world) {
    const auto& [config, wk] = selected;
    const auto& [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    const int output_threads = (bm / wm) * (bn / wn) * 32;
    const int shared_packets = get_bf16_gemm_rs_shared_packets(config, m, n, k, world);
    const int total_smem = std::get<0>(smem) + shared_packets * output_threads * 16;
    const int64_t tiles = int64_t((m + bm - 1) / bm) * ((n + bn - 1) / bn);
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    const auto memory_policy = deep_gemm_bf16_common::memory_policies_enabled()
        ? deep_gemm_bf16_common::get_memory_execution_policy(m, n, bm, bn, sms, 1, sms,
              properties->multiProcessorCount)
        : deep_gemm_bf16_common::MemoryExecutionPolicy{};
    // Higher source fan-in can amortize partial R8/R16 row compaction over
    // a steady K pipeline. Retain native R4 packets and K-split output.
    const bool compact_partial = world > 4 && m < wm && get_gemm_rs_packet_rows(m / world) > 4 &&
                                 wk == bk && k / bk >= 3 * stages;
    // Leave the full vector register budget to the inlined one-tile reduction.
    const bool single_tile = memory_policy.nonpersistent && tiles <= sms && wk == bk &&
                             output_threads <= 2 * deep_gemm_adaptive::WE_PER_CU * 32;
    const int local_m = m / world, owner_rows = local_m / bm;
    // Splitting peer and local work needs at least a well-populated M wave
    // per shard. Column tiles fill the remaining slots for streaming weights.
    const bool split_owner_shards = shared_packets == 0 && world == 2 &&
        n % bn == 0 && wn % 32 == 0 && wk == bk &&
        (owner_rows >= sms || (stages >= 4 && wm == wn &&
            has_bf16_gemm_rs_owner_waves(config, local_m, n, stages) &&
            int64_t(n) * k > properties->l2CacheSize));
    return {output_threads * (bk / wk), shared_packets, total_smem,
            tiles > sms && total_smem > PpuHwParams::instance().tsm_per_cu / 2,
            memory_policy.b_cache_policy, compact_partial ? 16 : wm,
            single_tile, split_owner_shards};
}

} // namespace deep_gemm::gemm_comm
