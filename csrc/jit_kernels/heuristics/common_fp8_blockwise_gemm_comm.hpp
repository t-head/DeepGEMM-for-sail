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
    const bool supports_rs_aware = is_ppu1v5_device();
    ConfigTuple config;
    if (tile) {
        const auto [bm, bn, bk, wm, wn, stages] = *tile;
        validate_gemm_rs_tile(*tile);
        config = {sm_budget, bm, bn, bk, wm, wn, stages, {}};
    } else {
        config = deep_gemm_fp8_common::get_best_configs(m, n, k, 1, sm_budget);
    }
    const GemmRSSmemBudget memory;
    const int64_t weight_bytes = int64_t(n) * k;
    const int64_t l2 = at::cuda::getCurrentDeviceProperties()->l2CacheSize;
    const bool cached_weights = weight_bytes <= l2;
    auto& [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    validate_gemm_rs_tile({bm, bn, bk, wm, wn, stages});
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
    const int local_m = m / world, packet_rows = get_gemm_rs_packet_rows(local_m);
    const bool partial_owner = local_m % packet_rows != 0;
    const auto prepare_input = [&] {
        validate_gemm_rs_tile({bm, bn, bk, wm, wn, stages});
        TORCH_CHECK(bk == 128, "blockwise FP8 GEMM/RS requires BK128");
        TORCH_CHECK(is_blockwise_gemm_rs_n_tile(bn),
                    "blockwise BN must divide 128, be a multiple of 128, or equal 192");
        TORCH_CHECK(world != 1 || bn != 192, "TP1 native blockwise GEMM does not support BN192 weight-scale refinement");
        TORCH_CHECK(bn != 192 || 64 % (16 * (bn / wn)) == 0,
                    "BN192 N64 scales require the tiled MMA N atom to divide 64; choose a compatible WN such as 48 or 96");
        smem = shared_size(bm, bn, stages);
    };
    const auto reserve_output_storage = [&] {
        if (world > 2 && supports_rs_aware && m <= bm && m >= wm && local_m >= 3 && partial_owner)
            std::get<0>(smem) = std::max(std::get<0>(smem), bm * bn * 2);
    };
    const auto finish = [&] {
        TORCH_CHECK(memory.fits(smem), "blockwise GEMM scales exceed shared-memory capacity");
        return config;
    };
    if (configs || world == 1 || tile_policy == "native" || !supports_rs_aware) {
        prepare_input();
        reserve_output_storage();
        return finish();
    }

    const int k_tiles = ceil_div(k, bk);
    const auto native_tile = std::make_tuple(bm, bn, wm, wn);
    if (m < bm) {
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
    const bool preserve_native_tile = local_m >= 16 * bm && k > 2 * stages * bk;
    if (m > bm && !preserve_native_tile &&
        ceil_div(m, bm) > sms / ceil_div(n, bn) && local_m % 16 == 0 && local_m % wm != 0 &&
        cached_weights) {
        // Keep a native tile that finishes in one CTA wave. Compact multi-source
        // traversal otherwise; avoid extra B reads
        // when the weights do not fit in L2 (also covers an unknown L2 capacity).
        // An owner-aligned warp exists only for a multiple of the 16-row MMA atom.
        int compact_wm = wm;
        while (local_m % compact_wm != 0)
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
    const bool wide_mma = (bm / wm) * (bn / wn) > 8;
    const bool paired_m_tiles = world > 2 && wide_mma && m == 2 * bm && local_m <= wm;
    const bool whole_owner = world > 2 && m == bm && local_m == wm && local_m == packet_rows && bn % (2 * wn) == 0;
    int pair_stages = stages;
    bool resident_half_width = false;
    if ((paired_m_tiles || whole_owner) && k_tiles <= 4 * stages) {
        resident_half_width = memory.fits(shared_size(bm, bn / 2, stages), 2);
        // Trim prefetch depth for a pair only when the native grid leaves CUs idle.
        if (paired_m_tiles && !resident_half_width && stages > 3 &&
                8 * ceil_div(n, bn) < 3 * sm_budget && k_tiles <= 4 * (stages - 1)) {
            --pair_stages;
            resident_half_width = memory.fits(shared_size(bm, bn / 2, pair_stages), 2);
        }
    }
    const bool resident_pair = paired_m_tiles && resident_half_width;
    const bool narrow_owner = world > 2 && !partial_owner &&
                              ((m <= bm && packet_rows < 16) || resident_half_width);
    if (!wide_mma || narrow_owner) {
        // Use idle CUs without replacing an almost-full wave with a long
        // spillover wave. Two-M-tile producers can retain their warp N
        // width when the smaller input tiles fit together on a CU.
        const int tiles = ceil_div(m, bm) * ceil_div(n, bn);
        const int max_grid = resident_pair ? 2 * sm_budget
            : wide_mma ? std::max(sm_budget, 2 * sm_budget - tiles) : std::numeric_limits<int>::max();
        const int original_bn = bn;
        adapt_gemm_rs_n_grid(config, m, n, sm_budget, false, is_blockwise_gemm_rs_n_tile, max_grid,
                             resident_half_width);
        if (resident_pair && bn != original_bn)
            stages = pair_stages;
    }
    // Refine N without adding a producer round. A partial owner pair must
    // fit in a single round and retain both M tiles after row compaction.
    const int refined_bn = bn / 4 * 3, refined_wn = wn / 4 * 3;
    const int row_tiles = ceil_div(m, bm);
    const int refined_tiles = row_tiles * ceil_div(n, refined_bn);
    const int current_sms = native_tile == std::make_tuple(bm, bn, wm, wn) ? sms : sm_budget;
    const bool full_pair = m == 2 * bm && local_m % 16 == 0;
    const bool partial_pair = world > 2 && local_m >= 2 * packet_rows &&
                              2 * bm - packet_rows < m && m < 2 * bm && refined_tiles <= sm_budget;
    const bool full_owner_tile = world > 2 && m == bm && local_m == packet_rows && packet_rows == 16;
    const bool aligned = n % refined_bn == 0 && (full_pair || partial_pair || full_owner_tile);
    // Mask the final column tile when compact owners keep most of one wave resident.
    constexpr int mma_rows = 16;
    const bool masked_owner_tile = world > 2 && m <= bm && bm - m < wm &&
        local_m > mma_rows / 2 && local_m <= mma_rows && refined_tiles <= sm_budget &&
        refined_tiles >= sm_budget - sm_budget / 4;
    if ((aligned || masked_owner_tile) && refined_bn == 192 && wn % 64 == 0 &&
            !memory.fits(shared_size(bm, bn, stages), 2) &&
            ceil_div(refined_tiles, sm_budget) <= ceil_div(row_tiles * ceil_div(n, bn), current_sms)) {
        bn = refined_bn;
        wn = refined_wn;
    }
    // Narrow N for short partial owners when the larger grid can stay resident.
    // R4 packets can use a single M warp, while R8 keeps two M warps.
    if (world > 2 && local_m >= 3 && packet_rows <= 8 && partial_owner &&
            wm <= m && m < bm && wn >= 32 && wn % 32 == 0 && is_blockwise_gemm_rs_n_tile(bn / 2)) {
        const int direct_bn = bn / 2, direct_wn = wn / 2, direct_stages = std::min(stages, 3);
        const int direct_wm = packet_rows == 4 ? bm : bm / 2;
        const int warps = (bm / direct_wm) * (direct_bn / direct_wn);
        const int tiles = ceil_div(n, direct_bn);
        const int resident_blocks = ceil_div(tiles, sm_budget);
        // Budget the spillover work at one additional warp per CU.
        if (direct_wm % 16 == 0 && warps >= 4 && tiles <= sm_budget + sm_budget / warps &&
                memory.fits(shared_size(bm, direct_bn, direct_stages), resident_blocks) &&
                deep_gemm_adaptive::tile_spill_ok(direct_wm, direct_wn, warps * resident_blocks)) {
            bn = direct_bn;
            wm = direct_wm;
            wn = direct_wn;
            stages = direct_stages;
        }
    }
    // Use one full compact payload when a narrower column tile also
    // reduces the number of resident producer rounds.
    if (world >= 8 && local_m < 3 && m <= bm) {
        const int compact_bn = 256 / local_m;
        const int compact_wn = wn * compact_bn / bn;
        if (compact_bn < bn && compact_wn >= 16 && compact_wn % 16 == 0 &&
                compact_bn % compact_wn == 0 && is_blockwise_gemm_rs_n_tile(compact_bn)) {
            const int blocks = std::max(1, memory.blocks(shared_size(bm, compact_bn, stages)));
            const int native_blocks = std::max(1, memory.blocks(shared_size(bm, bn, stages)));
            if (gemm_rs_column_rounds(n, compact_bn, sm_budget * blocks) <
                    gemm_rs_column_rounds(n, bn, sms * native_blocks) &&
                    deep_gemm_adaptive::tile_spill_ok(wm, compact_wn, (bm / wm) * (compact_bn / compact_wn) * blocks)) {
                bn = compact_bn;
                wn = compact_wn;
            }
        }
    }
    const auto resident = [&](int columns, int warp_n, int depth, int cus) {
        const int bytes = std::get<0>(shared_size(bm, columns, depth));
        return cus * std::max(1, gemm_rs_resident_blocks(wm, warp_n, columns / warp_n, bytes));
    };
    // Split a complete N warp group for a raw four-source output packet.
    if (world == 4 && m <= bm && bm == wm && packet_rows == 4 && wn == 16 &&
            bn / wn == deep_gemm_adaptive::WE_PER_CU && ceil_div(n, bn) > sms) {
        const int narrower = bn / 2;
        if (ceil_div(n, narrower) <= resident(narrower, wn, stages, sm_budget)) {
            bn = narrower;
            sms = sm_budget;
        }
    }
    if (world >= 8 && local_m == 1 && m <= bm && bm == wm && bn >= 128) {
        constexpr int word_columns = 64;
        const int word_stages = std::min(stages, 3);
        if (ceil_div(n, word_columns) <= resident(word_columns, word_columns / 4, word_stages, sm_budget)) {
            bn = word_columns;
            wn = bn / 4;
            stages = word_stages;
        } else if (bn > 2 * word_columns) {
            const int next_bn = 2 * word_columns, next_wn = next_bn / 4;
            const int current_sms = native_tile == std::make_tuple(bm, bn, wm, wn) ? sms : sm_budget;
            if (ceil_div(ceil_div(n, next_bn), resident(next_bn, next_wn, stages, sm_budget)) <
                ceil_div(ceil_div(n, bn), resident(bn, wn, stages, current_sms))) {
                bn = next_bn;
                wn = next_wn;
            }
        }
    }
    // More column coverage can remove a whole tile round for cached weights.
    // When the short row wave shares a warp between owners, restore the input
    // depth and use the available CUs if the original N tile already suffices.
    if (world == 2 && m > bm && !partial_owner &&
            cached_weights) {
        const int rows = ceil_div(m, bm), resident_rounds = ceil_div(rows * ceil_div(n, bn), sms);
        const int refined_bn = bn / 2 * 3, full_bn = 2 * bn;
        // Use complete weight panels when the refined grid still needs a second round.
        const bool full_panel = bn == 128 && wn % 32 == 0 && rows == 2 && local_m >= bm - wm &&
            n % full_bn == 0 && rows * ceil_div(n, full_bn) <= sm_budget &&
            ceil_div(rows * ceil_div(n, refined_bn), sm_budget) >= resident_rounds;
        const int wide_bn = full_panel ? full_bn : refined_bn;
        const int wide_wn = full_panel ? 2 * wn : wn / 2 * 3;
        const int wide_stages = full_panel ? stages : 3;
        if ((full_panel || (wide_bn == 192 && 64 % (16 * (wide_bn / wide_wn)) == 0)) && wn % 32 == 0 &&
                ceil_div(rows * ceil_div(n, wide_bn), sm_budget) < resident_rounds &&
                memory.fits(shared_size(bm, wide_bn, wide_stages)) &&
                deep_gemm_adaptive::tile_spill_ok(wm, wide_wn, (bm / wm) * (bn / wn))) {
            bn = wide_bn;
            wn = wide_wn;
            stages = wide_stages;
        } else if (sms < sm_budget && m % bm != 0 && local_m % wm != 0 && stages == 3) {
            // More prefetch stages must not reduce producer residency.
            const int64_t capacity = memory.capacity;
            const int current_blocks = memory.blocks(shared_size(bm, bn, stages));
            const auto next_smem = shared_size(bm, bn, stages + 1);
            const int next_blocks = memory.blocks(next_smem);
            if (std::get<0>(next_smem) <= 7 * (capacity / 8) &&
                    sm_budget * next_blocks >= std::min(rows * ceil_div(n, bn), sms * current_blocks)) {
                sms = sm_budget;
                ++stages;
            }
        }
    }
    if (native_tile != std::make_tuple(bm, bn, wm, wn))
        sms = sm_budget; // A native CU budget was chosen for the original tile.
    prepare_input();
    // Keep multiple short column tiles resident for streaming weights.
    if (m <= bm && bm == wm &&
            (packet_rows == 4 || (world == 2 && packet_rows == 8)) &&
            bn == 256 && stages >= 3 && !memory.fits(smem, 2) &&
            !cached_weights) {
        constexpr int columns = 192, warp_columns = 48, depth = 3;
        const int slots = resident(columns, warp_columns, depth, sm_budget);
        const int tiles = ceil_div(n, columns);
        const int native_rounds = gemm_rs_column_rounds(n, bn, sms);
        const int rounds = ceil_div(tiles, slots);
        // Narrowing the columns trades larger per-warp fragments for more
        // resident output tiles. Preserve input depth when both choices need
        // the same number of producer waves.
        const bool sparse_tail = rounds > 1 && tiles <= rounds * slots - slots / 2;
        const int deep_slots = resident(columns, warp_columns, stages, sm_budget);
        // A deeper pipeline uses more input storage. Leave a third of
        // its final wave available to amortize the wider warp fragments.
        const bool keep_depth = world == 2 && rounds < native_rounds &&
            3 * (rounds * deep_slots - tiles) > deep_slots;
        const bool use_wide = keep_depth || 2 * rounds < native_rounds ||
                              (2 * rounds == native_rounds && sparse_tail);
        // Narrow fragments sustain more memory streams once both resident
        // producer slots on the second wide wave must process another tile.
        const bool crowded_wide = world == 2 && deep_slots == 2 * sm_budget && native_rounds == 3 &&
            tiles > deep_slots + sm_budget + ceil_div(sm_budget, deep_gemm_adaptive::WE_PER_CU) &&
            k_tiles >= 3 * deep_gemm_adaptive::WE_PER_CU * depth;
        const bool short_partial = !use_wide && world > 2 && local_m < packet_rows &&
            k_tiles <= 2 * deep_gemm_adaptive::WE_PER_CU * depth;
        constexpr int packet_n = 4 * 16, packet_wn = 16;
        const int packet_tiles = ceil_div(n, packet_n);
        const int packet_slots = crowded_wide || short_partial ? resident(packet_n, packet_wn, depth, sm_budget) : 0;
        const int packet_rounds = packet_slots > 0 ? ceil_div(packet_tiles, packet_slots) : native_rounds;
        // Keep the second narrow wave sparse when replacing a deep two-source tile.
        const bool sparse_packet_tail = !crowded_wide || 3 * (packet_tiles - packet_slots) <= packet_slots;
        // A sparse packet grid wins before the wider input candidate.
        if (packet_rounds <= 2 && packet_rounds < native_rounds && sparse_packet_tail) {
            config = {sm_budget, bm, packet_n, bk, wm, packet_wn, depth, shared_size(bm, packet_n, depth)};
        } else if (use_wide) {
            const int selected_depth = keep_depth ? stages : depth;
            config = {sm_budget, bm, columns, bk, wm, warp_columns, selected_depth,
                      shared_size(bm, columns, selected_depth)};
        }
    }
    if (m > bm) {
        // Keep an already resident producer grid if M compaction would
        // widen its warp pressure beyond two execution groups per CU.
        constexpr int warp_budget = 2 * deep_gemm_adaptive::WE_PER_CU;
        const int columns = ceil_div(n, bn), column_warps = bn / wn;
        const bool keep_native_rows = world == 2 && m - bm > wm &&
            ceil_div(m, bm) * columns <= sm_budget && (bm / wm) * column_warps <= warp_budget &&
            ceil_div(m, wm) * column_warps > warp_budget;
        if (!keep_native_rows)
            adapt_gemm_rs_m_tile(config, m, n, world, sm_budget, 4, shared_size);
        // Smaller warp accumulators can make two CTAs resident without
        // introducing padded M work for a full row-owner tile.
        if (world > 2 && local_m == bm && wm % (3 * 16) == 0 && wn % 32 == 0 &&
                !memory.fits(smem, 2) &&
                cached_weights) {
            const int next_bm = bm / 3 * 2, next_wm = wm / 3 * 2;
            const int next_bn = bn / 2, next_wn = wn / 2, next_stages = std::min(stages, 3);
            const auto next_smem = shared_size(next_bm, next_bn, next_stages);
            const int tiles = ceil_div(n, bn) * world;
            if (m % next_bm == 0 && is_blockwise_gemm_rs_n_tile(next_bn) &&
                    next_bn % next_wn == 0 && tiles > sms && tiles <= 2 * sms &&
                    memory.fits_warps(next_smem, 2, next_wm, next_wn, 2 * (bm / wm) * (bn / wn))) {
                config = {sm_budget, next_bm, next_bn, bk, next_wm, next_wn, next_stages, next_smem};
            }
        }
        // Exchange row and column atoms while keeping the warp count and
        // accumulator area. This limits work on irregular owner row tiles.
        if (world > 2 && local_m >= 2 * packet_rows && 3 * local_m <= 2 * bm && partial_owner &&
                wm % 48 == 0 && wn % 64 == 0 && bn % 256 == 0 && stages > 3 &&
                k_tiles >= 4 * stages) {
            const int next_bm = bm / 3 * 2, next_wm = wm / 3 * 2;
            const int next_bn = bn / 2 * 3, next_wn = wn / 2 * 3;
            const int current_tiles = ceil_div(m, bm) * ceil_div(n, bn);
            const int next_tiles = ceil_div(m, next_bm) * ceil_div(n, next_bn);
            const auto next_smem = shared_size(next_bm, next_bn, stages - 1);
            if (ceil_div(next_tiles, sm_budget) <= ceil_div(current_tiles, sms) &&
                memory.fits_warps(next_smem, 1, next_wm, next_wn, (bm / wm) * (bn / wn))) {
                config = {sm_budget, next_bm, next_bn, bk, next_wm, next_wn, stages - 1, next_smem};
            }
        }
        // A wider row tile can avoid a whole producer wave when the native
        // grid repeats the same weight panels across short owners.
        if (world > 2 && local_m > bm - wm && local_m <= bm && m > 2 * bm && wm % 32 == 0 && wn % 64 == 0 &&
                stages >= 4 && cached_weights) {
            const int expanded_m = bm / 2 * 3, expanded_wm = wm / 2 * 3;
            const int column_tiles = ceil_div(n, bn);
            const int rounds = ceil_div(ceil_div(m, bm) * column_tiles, sms);
            const int next_rounds = ceil_div(ceil_div(m, expanded_m) * column_tiles, sm_budget);
            const auto expanded_smem = shared_size(expanded_m, bn, stages - 1);
            const int tail_rows = m % expanded_m;
            if ((tail_rows == 0 || 2 * tail_rows >= expanded_m) && next_rounds < rounds &&
                next_rounds * expanded_m <= rounds * bm &&
                memory.fits_warps(expanded_smem, 1, expanded_wm, wn, (bm / wm) * (bn / wn))) {
                bm = expanded_m;
                wm = expanded_wm;
                --stages;
                sms = sm_budget;
                smem = expanded_smem;
            }
        }
        // Balance a nearly full ragged pair instead of producing a
        // third padded M tile. Narrow N to keep both tiles resident.
        if (world >= 8 && local_m >= 2 * packet_rows && partial_owner &&
                m > 2 * bm && ceil_div(m, bm) == 3 && wm % 32 == 0 && wn % 32 == 0 &&
                stages >= 4 && k_tiles <= 2 * stages && is_blockwise_gemm_rs_n_tile(bn / 2)) {
            const int next_wm = wm / 2 * 3, next_bn = bn / 2, next_wn = wn / 2;
            const int next_bm = ceil_div(m, 2 * next_wm) * next_wm;
            const auto next_smem = shared_size(next_bm, next_bn, stages - 1);
            const int slots = gemm_rs_resident_blocks(next_wm, next_wn, (next_bm / next_wm) * (next_bn / next_wn),
                                                     std::get<0>(next_smem));
            if (next_bm > bm && 2 * next_bm < 3 * bm && 2 * ceil_div(n, next_bn) <= slots * sm_budget) {
                config = {sm_budget, next_bm, next_bn, bk, next_wm, next_wn, stages - 1, next_smem};
            }
        }
        // Split an underfilled grid into tiles that stay close to each row
        // owner's extent, without increasing the padded M work.
        const int producer_tiles = ceil_div(m, bm) * ceil_div(n, bn);
        const bool split_owner = m % bm != 0 && local_m % 64 == 0 && bm / wm >= 4 &&
                                 k_tiles <= 4 * std::min(stages, 3);
        const int owner_bm = split_owner ? local_m / 2 : ceil_div(local_m, 32) * 32;
        const int owner_bn = bn / 2, owner_wn = wn / 2;
        if (world > 2 && owner_bm < bm && ceil_div(m, owner_bm) * owner_bm <= ceil_div(m, bm) * bm &&
                producer_tiles < sm_budget &&
                (2 * producer_tiles <= sm_budget || k_tiles <= 4 * stages) && wn >= 64 && wn % 32 == 0 &&
                is_blockwise_gemm_rs_n_tile(owner_bn) && owner_bn % owner_wn == 0) {
            const int warps = 2 * (owner_bn / owner_wn);
            const int tiles = ceil_div(m, owner_bm) * ceil_div(n, owner_bn);
            const int owner_stages = std::min(stages, 3);
            const auto owner_smem = shared_size(owner_bm, owner_bn, owner_stages);
            const int capacity = memory.blocks(owner_smem);
            const int blocks = split_owner ? std::min(ceil_div(tiles, sm_budget), capacity)
                                           : ceil_div(tiles, sm_budget);
            const int max_tiles = (split_owner ? 2 * blocks : 2) * sm_budget + sm_budget / warps;
            if (warps >= 4 && blocks > 0 && tiles <= max_tiles && (split_owner || capacity >= blocks) &&
                    deep_gemm_adaptive::tile_spill_ok(owner_bm / 2, owner_wn, blocks * warps)) {
                config = {sm_budget, owner_bm, owner_bn, bk, owner_bm / 2, owner_wn, owner_stages, owner_smem};
            }
        }
    }
    if (world > 2 && m <= bm && local_m >= packet_rows && partial_owner &&
            m > bm / 2 && bm % (2 * wm) == 0 && bn == 128 && wn % 32 == 0 &&
            gemm_rs_column_rounds(n, bn, sms) == 2) {
        // Split a complete M tile when both grids need two producer rounds.
        // The smaller input tile leaves space for an extra prefetch stage.
        constexpr int split_bn = 192;
        const int split_bm = bm / 2;
        const int split_wm = local_m < 2 * packet_rows && wm % 32 == 0 ? wm / 2 : wm;
        const auto split_smem = shared_size(split_bm, split_bn, stages + 1);
        if (split_bm * (bn / wn) >= 4 * wm && split_bm + split_bn < bm + bn &&
                ceil_div(2 * ceil_div(n, split_bn), sm_budget) == 2 &&
                memory.fits(split_smem)) {
            config = {sm_budget, split_bm, split_bn, bk, split_wm, wn / 2 * 3, stages + 1, split_smem};
        }
    }
    // Narrow the accumulator when a short R16 owner crosses an M warp.
    // Admit an extra producer round only for weights that can remain in L2.
    if (world > 2 && m > bm && local_m < bm / 2 && packet_rows == 16 &&
            local_m % wm != 0 && wn >= 64 && wn % 32 == 0 && wm > 32 &&
            k_tiles <= 4 * 3 && cached_weights &&
            is_blockwise_gemm_rs_n_tile(bn / 2) && ceil_div(m, bm) * ceil_div(n, bn) <= sm_budget) {
        const int next_bn = bn / 2, next_wn = wn / 2, next_stages = std::min(stages, 3);
        const auto next_smem = shared_size(bm, next_bn, next_stages);
        const int next_tiles = ceil_div(m, bm) * ceil_div(n, next_bn);
        if (next_tiles <= 2 * sm_budget && memory.fits_warps(next_smem, 2, wm, next_wn, (bm / wm) * (next_bn / next_wn))) {
            config = {sm_budget, bm, next_bn, bk, wm, next_wn, next_stages, next_smem};
        }
    }
    // The final M tile can use two row warps when the column grid is
    // already resident. This also covers a tile formed by M padding.
    if (world > 2 && m <= bm && m >= 2 * wm && bm == 3 * wm && wm % 32 == 0 &&
            ceil_div(n, bn) <= sm_budget && deep_gemm_adaptive::tile_spill_ok(bm / 2, wn, 2 * (bn / wn))) {
        wm = bm / 2;
        sms = sm_budget;
        // A single resident column wave can spend more shared memory on
        // input prefetch when the weights occupy most of L2.
        if (stages == 3 && 3 * weight_bytes >= 2 * l2) {
            const auto deeper = shared_size(bm, bn, stages + 1);
            if (memory.fits(deeper)) {
                ++stages;
                smem = deeper;
            }
        }
    }
    // Bound the producer's live fragments when the full M tile uses
    // many row warps. Reuse the native N warp count and input depth budget.
    const int row_warps = bm / wm, column_warps = bn / wn;
    const int split_m = 4 * wm;
    const int column_rounds = gemm_rs_column_rounds(n, bn, sms);
    const bool crowded = row_warps * column_warps > 3 * deep_gemm_adaptive::WE_PER_CU;
    const bool long_input = bn == 128 && wn % 32 == 0 && stages >= 3 &&
        k_tiles >= 4 * stages &&
        ((stages > 3 && (crowded || !cached_weights)) ||
         (crowded && column_rounds == 1));
    const int warps = row_warps * column_warps;
    // N64 scale refinement keeps another output fragment live on a
    // wide row warp. Check the final warp count after M compaction.
    const bool refined_register_bound = bn == 192 && wm > 2 * 16 &&
        warps >= 3 * deep_gemm_adaptive::WE_PER_CU;
    const bool long_spilling_tile = world == 2 && bn > 128 && k_tiles > 4 * stages &&
        (warps > 3 * deep_gemm_adaptive::WE_PER_CU || refined_register_bound);
    const bool register_bound = (long_spilling_tile || (world > 2 && partial_owner)) &&
        column_warps * 16 <= 64 &&
        ((world == 2 && refined_register_bound) || !deep_gemm_adaptive::tile_spill_ok(wm, wn, warps));
    if (split_m < m && m <= bm && m <= 2 * split_m && (long_input || register_bound)) {
        // Use the refined N tile while it can finish in the same CTA rounds.
        constexpr int refined_n = 192;
        const int budget = long_input ? column_rounds : 2;
        const auto fits = [&](int columns) { return 2 * ceil_div(n, columns) <= budget * sm_budget; };
        // A wider column tile can keep the split panel in the same wave
        // when the refined N192 grid would need another round.
        const bool short_owner = world > 2 && local_m < 2 * packet_rows && partial_owner;
        const bool keep_column_round = long_input && crowded && !short_owner && fits(2 * bn);
        const int split_n = fits(refined_n) ? refined_n : (keep_column_round ? 2 * bn : bn);
        const int split_warp_n = split_n / column_warps;
        const int split_stages = long_spilling_tile ? stages : std::min(stages, 3);
        const auto split_smem = shared_size(split_m, split_n, split_stages);
        if ((!long_input || split_n != bn) && split_warp_n % 16 == 0 &&
                64 % (16 * column_warps) == 0 &&
                memory.fits_warps(split_smem, 1, wm, split_warp_n, 4 * column_warps)) {
            config = {sm_budget, split_m, split_n, bk, wm, split_warp_n, split_stages, split_smem};
        }
    }
    // Give a whole two-source row tile more resident column producers
    // without changing the number of N warps.
    if (world == 2 && m <= bm && (bm / wm) * (bn / wn) >= 2 * deep_gemm_adaptive::WE_PER_CU) {
        const int next_bn = bn / 4 * 3, next_wn = wn / 4 * 3;
        const int columns = ceil_div(n, next_bn);
        const bool keep_n_warps = bn == 4 * wn && wn % 64 == 0 && is_blockwise_gemm_rs_n_tile(next_bn);
        const int row_warps = bm / wm, col_warps = bn / wn;
        const bool compact_rows = keep_n_warps && next_wn > wm &&
            ceil_div(m, next_wn) < row_warps && ceil_div(m, next_wn) <= col_warps;
        const int next_wm = compact_rows ? next_wn : wm;
        const int next_bm = compact_rows ? ceil_div(m, next_wm) * next_wm : bm;
        const int next_stages = compact_rows ? stages : 3;
        if ((compact_rows || (m == bm && local_m % wm == 0)) &&
                row_warps * col_warps > 2 * deep_gemm_adaptive::WE_PER_CU &&
                keep_n_warps && sms < columns && columns <= sm_budget &&
                stages >= 4 && k_tiles >= 4 * stages &&
                (wm <= 2 * 16 || row_warps * col_warps < 3 * deep_gemm_adaptive::WE_PER_CU) &&
                deep_gemm_adaptive::tile_spill_ok(next_wm, next_wn, (next_bm / next_wm) * col_warps)) {
            const auto bytes = shared_size(next_bm, next_bn, next_stages);
            if (memory.fits(bytes))
                config = {sm_budget, next_bm, next_bn, bk, next_wm, next_wn, next_stages, bytes};
        } else if (ceil_div(n, bn / 2) <= sm_budget) {
            const int previous_bn = bn;
            adapt_gemm_rs_n_grid(config, m, n, sm_budget, false, is_blockwise_gemm_rs_n_tile, sm_budget, true);
            if (bn != previous_bn) {
                sms = sm_budget;
                smem = shared_size(bm, bn, stages);
            }
        }
    }
    // Refine the selected streaming pair after M adaptation so a partial
    // final N panel does not change the native row tile or prefetch depth.
    if (world == 2 && bm < m && m <= 2 * bm && local_m > bm - wm &&
            bn == 256 && wn % 64 == 0 && stages >= 4 && k_tiles >= 4 * stages &&
            !cached_weights) {
        constexpr int columns = 192;
        const int native_rounds = ceil_div(2 * ceil_div(n, bn), sms);
        if (2 * ceil_div(n, columns) <= std::min(2, native_rounds) * sm_budget) {
            config = {sm_budget, bm, columns, bk, wm, wn / 4 * 3, stages, shared_size(bm, columns, stages)};
        }
    }
    adapt_partial_r4_fp8_tile(config, m, n, world, sm_budget, shared_size);
    reserve_output_storage();
    // With at most two tiles per producer, shorten the input pipeline
    // for a nearly full ragged owner tile and cache-resident K traversal.
    if (world > 2 && m > bm && local_m >= 2 * packet_rows && partial_owner &&
            2 * local_m < bm && m % bm >= 3 * bm / 4 && wn >= 64 &&
            stages > 3 && k_tiles <= 3 * stages &&
            ceil_div(m, bm) * ceil_div(n, bn) <= 2 * sms &&
            cached_weights) {
        --stages;
        smem = shared_size(bm, bn, stages);
    }
    // Trade excess column warps for more resident producers when a
    // whole-M streaming tile feeds complete row packets.
    if (world <= 4 && m == bm && bm == wm && bm > 16 && bn == 128 &&
            wn == 16 && local_m == packet_rows && m <= 2 * 16 &&
            k_tiles >= 8 * std::min(stages, 3)) {
        constexpr int columns = 64, depth = 3;
        const int rows = local_m >= 16 ? bm / 2 : wm;
        const auto bytes = shared_size(bm, columns, depth);
        const int blocks = gemm_rs_resident_blocks(rows, wn, (bm / rows) * (columns / wn),
                                                  std::get<0>(bytes));
        const int current_blocks = std::max(1, gemm_rs_resident_blocks(wm, wn, bn / wn, std::get<0>(smem)));
        if (blocks > 0 && gemm_rs_column_rounds(n, columns, blocks * sm_budget) <=
                gemm_rs_column_rounds(n, bn, current_blocks * sms)) {
            config = {sm_budget, bm, columns, bk, rows, wn, depth, bytes};
        }
    }

    // Compact owners can trade wider warp fragments for more resident N tiles.
    // Keep four compute warps and reject widths that add a producer round.
    if (world >= 8 && m == bm && local_m <= 8 &&
            (local_m == packet_rows || is_single_owner_packet(world, local_m, bn)) &&
            stages >= 3 && (bm / wm) <= 4) {
        for (const int columns : {bn / 4, bn / 2}) {
            const int warp_n = std::max(16, (bm / wm) * columns / 4);
            if (!is_blockwise_gemm_rs_n_tile(columns) || n % columns != 0 || columns % warp_n != 0)
                continue;
            const int warps = (bm / wm) * (columns / warp_n);
            if (warps < 4 || warp_n % 16 != 0)
                continue;
            constexpr int depth = 3;
            const auto bytes = shared_size(bm, columns, depth);
            const int blocks = gemm_rs_resident_blocks(wm, warp_n, warps, std::get<0>(bytes));
            const int tiles = ceil_div(n, columns);
            if (blocks > 0 && tiles <= blocks * sm_budget) {
                config = {sm_budget, bm, columns, bk, wm, warp_n, depth, bytes};
                break;
            }
        }
    }
    // Widen a three-row-tile grid when a full column wave can reuse its
    // B panels without increasing the number of warp fragments.
    if (world >= 8 && 2 * bm < m && m < 3 * bm && !partial_owner &&
            bn == 128 && n % (2 * bn) == 0 && stages == 3 && k_tiles >= 4 * stages &&
            3 * ceil_div(n, bn) > sms && 3 * ceil_div(n, 2 * bn) <= sm_budget) {
        const auto bytes = shared_size(bm, 2 * bn, stages);
        if (memory.fits_warps(bytes, 1, wm, 2 * wn, (bm / wm) * (bn / wn)))
            config = {sm_budget, bm, 2 * bn, bk, wm, 2 * wn, stages, bytes};
    }
    // A wide column grid can amortize its B panels across another MMA
    // row per warp while retaining the same input pipeline depth.
    if (world >= 8 && wm == 2 * 16 && bm == 4 * wm && local_m % bm == 0 &&
            local_m / bm >= 2 * stages && n % bn == 0 && 3 * ceil_div(n, bn) >= 2 * sm_budget &&
            stages >= 3 && k_tiles >= 2 * stages && k_tiles <= 4 * stages) {
        const int rows = wm + 16, block_rows = (bm / wm) * rows;
        const auto bytes = shared_size(block_rows, bn, stages);
        if (memory.fits_warps(bytes, 1, rows, wn, (bm / wm) * (bn / wn)))
            config = {sm_budget, block_rows, bn, bk, rows, wn, stages, bytes};
    }
    return finish();
}

} // namespace deep_gemm::gemm_comm
