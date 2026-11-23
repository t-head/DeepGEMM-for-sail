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

struct ChannelwiseGemmRSTileResources {
    Fp8GemmRSTile tile;
    std::tuple<int, int, int> smem;

    int warps() const {
        const auto& [bm, bn, bk, wm, wn, stages] = tile;
        return (bm / wm) * (bn / wn);
    }

    int resident_blocks(int output_bytes = 0) const {
        return gemm_rs_resident_blocks(tile[3], tile[4], warps(), std::get<0>(smem) + output_bytes);
    }
};

inline Fp8GemmRSConfig get_channelwise_gemm_rs_config(int m, int n, int k, int world,
                                                      std::optional<ConfigTuple> configs, const std::string& tile_policy) {
    auto tile = get_gemm_rs_tile(configs);
    const int sm_budget = configs ? std::get<0>(*configs) : get_num_sms();
    validate_gemm_rs_shape(m, n, k, world, false);
    TORCH_CHECK(tile_policy == "rs_aware" || tile_policy == "native", "channelwise tile_policy must be rs_aware or native");
    TORCH_CHECK(is_ppu1v5_device(), "channelwise FP8 GEMM/RS requires PPU0015");
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
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    const GemmRSSmemBudget memory;
    const auto capacity = memory.capacity;
    const int64_t weight_bytes = int64_t(n) * k;
    validate_gemm_rs_tile({bm, bn, bk, wm, wn, stages}, true);
    smem = deep_gemm_int8::get_smem_config(stages, k, bm, bn, bk, 1);
    const auto finish = [&] {
        TORCH_CHECK(std::get<0>(smem) <= capacity, "channelwise GEMM exceeds shared-memory capacity");
        return config;
    };
    if (configs || world == 1 || tile_policy == "native")
        return finish();

    const int local_m = m / world, packet_rows = get_gemm_rs_packet_rows(local_m);
    const bool owner_tail = local_m >= 3 && local_m % packet_rows != 0;
    // Narrow N to use idle CUs and reuse the native input footprint unless
    // the column grid changes.
    if ((bm / wm) * (bn / wn) <= 8) {
        const int native_bn = bn;
        const int smem_blocks = std::max(1, int(properties->sharedMemPerMultiprocessor / std::get<0>(smem)));
        const int packet_parallelism = std::min({smem_blocks, std::max(4, local_m) / local_m, wn / 16});
        adapt_gemm_rs_n_grid(config, m, n, sm_budget * packet_parallelism, true);
        if (bn != native_bn) {
            validate_gemm_rs_tile({bm, bn, bk, wm, wn, stages}, true);
            smem = deep_gemm_int8::get_smem_config(stages, k, bm, bn, bk, 1);
        }
    }

    const auto estimate = [&](const Fp8GemmRSTile& candidate) {
        const auto& [rows, columns, input_k, warp_m, warp_n, depth] = candidate;
        return ChannelwiseGemmRSTileResources{candidate,
            deep_gemm_int8::get_smem_config(depth, k, rows, columns, input_k, 1)};
    };
    const auto current = [&] { return ChannelwiseGemmRSTileResources{{bm, bn, bk, wm, wn, stages}, smem}; };
    const auto column_rounds = [&](int columns, int blocks, int cus) {
        return gemm_rs_column_rounds(n, columns, cus * std::max(1, blocks));
    };
    const auto use_tile = [&](const ChannelwiseGemmRSTileResources& candidate) {
        const auto& [rows, columns, input_k, warp_m, warp_n, depth] = candidate.tile;
        config = {sm_budget, rows, columns, input_k, warp_m, warp_n, depth, candidate.smem};
    };
    const auto use_resident_tile = [&](const ChannelwiseGemmRSTileResources& candidate, int blocks,
                                       int resident_blocks_for_spill = 0) {
        const int warps = candidate.warps();
        if (blocks <= 0 || warps < 2 || warps * 32 > properties->maxThreadsPerBlock ||
                !memory.fits_warps(candidate.smem, blocks, candidate.tile[3], candidate.tile[4],
                    warps * (resident_blocks_for_spill > 0 ? resident_blocks_for_spill : blocks)))
            return false;
        use_tile(candidate);
        return true;
    };
    // Two-source reduction retains raw packets. Preserve its native M grid
    // when growing a tile; shrinking an existing tile does not change readiness.
    if (owner_tail &&
            (world > 2 || m <= bm) &&
            (m >= bm || weight_bytes <= properties->l2CacheSize)) {
        adapt_gemm_rs_m_tile(config, m, n, world, sm_budget, 2,
            [&](int rows, int columns, int input_stages) {
                return deep_gemm_int8::get_smem_config(input_stages, k, rows, columns, bk, 1);
            });
        validate_gemm_rs_tile({bm, bn, bk, wm, wn, stages}, true);
    }
    const auto streaming_column_work = [&](const ChannelwiseGemmRSTileResources& candidate, int cus) {
        const int blocks = candidate.resident_blocks(), columns = candidate.tile[1];
        return blocks > 0 ? int64_t(column_rounds(columns, blocks, cus)) * (columns + 2 * candidate.tile[0])
                          : std::numeric_limits<int64_t>::max();
    };
    // A compact partial owner can trim padded M rows by switching from
    // two-row-atom warps to fewer, three-row-atom warps.
    if (world > 2 && owner_tail && packet_rows == 16 && m <= bm && wm == 2 * 16 && wn == 16 &&
            bm / wm > 3) {
        constexpr int row_warp = 3 * 16;
        const int rows = ceil_div(m, row_warp) * row_warp;
        const int blocks = gemm_rs_column_rounds(n, bn, sm_budget);
        if (rows < bm) {
            use_resident_tile(estimate({rows, bn, bk, row_warp, wn, stages}), blocks);
        } else if (world >= 8 && m < bm && blocks <= 2 && bn % (2 * wn) == 0) {
            // Keep the complete M tile and reduce N warp count.
            use_resident_tile(estimate({bm, bn, bk, wm, 2 * wn, stages}), blocks);
        }
    }
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
    // Resolve the mutually exclusive deep-input and BK128 narrowing
    // cases before considering a wider panel on the resulting TP2 tile.
    const bool short_streaming_owner = world == 2 || (world == 4 && packet_rows <= 8);
    if (short_streaming_owner && owner_tail && m <= bm && bm > wm && bk > 128 && stages >= 4 &&
            weight_bytes > properties->l2CacheSize) {
        constexpr int columns = 64, warp_n = 32, input_k = 128, depth = 2;
        const int rows = ceil_div(m, 16) * 16;
        const auto narrow = estimate({rows, columns, input_k, rows, warp_n, depth});
        const int blocks = narrow.resident_blocks();
        const int rounds = column_rounds(columns, blocks, sm_budget);
        if (blocks > 0 && column_rounds(bn, current().resident_blocks(), sm_budget) > (bk / input_k + 1) * rounds) {
            const auto wide = estimate({rows, 2 * columns, input_k, rows, 2 * warp_n, depth});
            const int wide_blocks = wide.resident_blocks();
            if (wide_blocks > 0 && column_rounds(2 * columns, wide_blocks, sm_budget) < rounds) {
                use_resident_tile(wide, wide_blocks);
            } else {
                const auto half_k = estimate({bm, columns, input_k / 2, wm, warp_n, 2 * depth});
                const int slots = half_k.resident_blocks() * sm_budget;
                if (world == 4 && m > wm && bm == 2 * wm && wm == warp_n && rounds > 1 && slots > 0 &&
                        ceil_div(n, columns) <= rounds * slots - slots / (2 * depth))
                    use_resident_tile(half_k, slots / sm_budget);
                else
                    use_resident_tile(narrow, blocks);
            }
        }
    } else if (world > 2 && owner_tail && packet_rows == 4 && m > wm && m <= bm && bm == 2 * wm &&
            wn == 2 * 16 && bn == 4 * wn && bk == 128 && stages >= 4 && ceil_div(k, bk) >= 2 * stages &&
            weight_bytes > properties->l2CacheSize) {
        const auto narrow = estimate({bm, bn, bk / 2, wm, wn, stages});
        const int blocks = narrow.resident_blocks();
        const int rounds = column_rounds(bn, blocks, sm_budget);
        const int native_rounds = column_rounds(bn, current().resident_blocks(), sms);
        if (native_rounds >= std::max(rounds + 2, 2 * rounds - 1))
            use_resident_tile(narrow, blocks);
    }
    if (world == 2 && owner_tail && m <= bm && bm == wm && wm == 3 * 16 &&
            wn == 2 * 16 && bn == 2 * wn && bk == 128 && stages == 2 &&
            k >= 2 * bk * stages * 16 && weight_bytes > properties->l2CacheSize) {
        const auto wide = estimate({bm, 2 * wm, 2 * bk, wm, wm, stages});
        const int old_blocks = current().resident_blocks();
        const int blocks = wide.resident_blocks();
        const int old_rounds = column_rounds(bn, old_blocks, sm_budget);
        const int rounds = column_rounds(2 * wm, blocks, sm_budget);
        // Protect the input overlap when the wider tile needs an extra wave.
        if (old_blocks > 0 && blocks > 0 && old_rounds > 1 && 2 * rounds <= 3 * old_rounds)
            use_resident_tile(wide, blocks);
    }
    // Fit a single-M-warp N grid in one compute round.
    if (m <= bm && bm == wm) {
        // A short packet can spend less TSM on each K stage while
        // preserving the number of active columns.
        const bool compact = is_single_owner_packet(world, local_m, bn);
        const int initial_columns = ceil_div(n, bn);
        const int column_blocks = ceil_div(initial_columns, sms);
        if (compact && stages > 4 && std::get<0>(shared_size(2)) > capacity / column_blocks) {
            const int four_stage_blocks = std::max(1, int(capacity / std::get<0>(shared_size(4))));
            adapt_gemm_rs_stages(config, 4, four_stage_blocks, 0, shared_size);
        } else {
            const int output_bytes = compact && local_m * bn == 64 ? bm * bn * 2 : 0;
            adapt_gemm_rs_stages(config, compact ? 2 : 3, column_blocks, output_bytes, shared_size);
            // A one-word owner packet benefits from a shallower resident input pipeline.
            if (output_bytes && stages == 4) {
                --stages;
                smem = shared_size(stages);
            }
        }
        if (compact && stages == 2 && bk == 128 && k % bk == 0 && k >= 4 * bk &&
                std::get<0>(smem) <= capacity / (column_blocks + 1)) {
            bk /= 2;
            stages *= 2;
            smem = shared_size(stages);
        }
        // A two-word packet can retain its resident column grid with
        // three input stages and reserve distinct output storage.
        if (compact && local_m == 2 && local_m * bn == 2 * 64 && stages > 3 &&
                std::get<0>(shared_size(3)) + bm * bn * 2 <= capacity / column_blocks) {
            stages = 3;
            smem = shared_size(stages);
        }
        // A short raw owner needs more memory requests in flight than
        // a wide, deep CTA can issue. Select the deepest half-width
        // pipeline that keeps its producer rounds.
        if (world == 4 && local_m < 4 && !compact && bm == 16 && bk == 128 &&
                wn == 32 && bn == 4 * wn && stages >= 3 &&
                weight_bytes > properties->l2CacheSize && initial_columns > 2 * sms) {
            const int next_bn = bn / 2, warps = next_bn / wn;
            const auto deep = estimate({bm, next_bn, bk, wm, wn, 3});
            const auto shallow = estimate({bm, next_bn, bk, wm, wn, 2});
            const auto rounds = [&](const ChannelwiseGemmRSTileResources& tile) {
                return gemm_rs_column_rounds(n, next_bn, sm_budget *
                    gemm_rs_resident_blocks(wm, wn, warps, std::get<0>(tile.smem)));
            };
            use_resident_tile(rounds(deep) > rounds(shallow) ? shallow : deep, 1);
        }
        // Spend a spare input stage on long, two-source single-row-warp
        // tiles, including owners that cross one native packet boundary.
        const int l2_bytes = properties->l2CacheSize;
        const bool extra_input_stage = weight_bytes >= l2_bytes ||
            (local_m < 16 && weight_bytes > l2_bytes / 2) ||
            initial_columns <= sms + sms / (bn / wn);
        if (world == 2 && local_m <= 16 && extra_input_stage &&
                stages == 3 && ceil_div(k, bk) >= 4 * stages && n % bn == 0) {
            const auto deeper = shared_size(stages + 1);
            if (std::get<0>(deeper) <= capacity / column_blocks) {
                ++stages;
                smem = deeper;
            }
        }
        if (world == 2 && local_m <= 16 && bk == 128 && wn == 32 && bn >= 4 * wn &&
                stages >= 3 && weight_bytes > l2_bytes && initial_columns > 2 * sms) {
            // Use two stages only when a full three-stage column wave
            // cannot sustain the short producer's input pipeline.
            const int narrow_columns = 2 * wn, wide_columns = 3 * wn;
            const bool short_input = bm > 16 && ceil_div(k, bk) > 8 * deep_gemm_adaptive::WE_PER_CU &&
                (streaming_column_work(estimate({bm, narrow_columns, bk, wm, wn, 2}), sm_budget) == narrow_columns + 2 * bm ||
                 streaming_column_work(estimate({bm, wide_columns, bk, wm, wn, 3}), sm_budget) > 2 * (wide_columns + 2 * bm));
            auto best = estimate({bm, bn, bk, wm, wn, stages});
            const int64_t native_work = streaming_column_work(best, sms);
            int64_t best_work = native_work;
            for (const int width : {2 * wn, wide_columns, bn}) {
                const int min_stages = width == 2 * wn && short_input ? 2 : 3;
                for (int depth = std::min(stages, 4); depth >= min_stages; --depth) {
                    const auto candidate = estimate({bm, width, bk, wm, wn, depth});
                    const auto work = streaming_column_work(candidate, sm_budget);
                    if (work < best_work) {
                        best_work = work;
                        best = candidate;
                    }
                }
            }
            // Keep shallow tiles unless a narrower input can retire a stage.
            if (best_work < native_work && (stages > 3 || best.tile[5] < stages))
                use_resident_tile(best, 1);
        }
        // One additional resident column group can trade input depth
        // for publication of a full resident column grid.
        if (compact && local_m < 3 && wn == 32 && bn == 4 * wn && column_blocks > 1 &&
                bk == 128 && stages >= 4 && k % bk == 0 && ceil_div(k, bk) <= 4 * stages &&
                std::get<0>(shared_size(2)) > capacity / column_blocks) {
            const auto narrow_smem = deep_gemm_int8::get_smem_config(3, k, bm, bn, bk / 2, 1);
            if (std::get<0>(narrow_smem) + bm * bn * 2 <= capacity / column_blocks &&
                    deep_gemm_adaptive::tile_spill_ok(wm, wn, (bn / wn) * column_blocks)) {
                config = {sms, bm, bn, bk / 2, wm, wn, 3, narrow_smem};
            }
        }
        // A long K loop benefits from wider warp accumulators when its
        // complete column grid still fits the two-stage input budget.
        if (compact && local_m < 3 && wn == 32 && bn == 4 * wn && bk <= 128 && column_blocks > 1 &&
                bm == 16 && bk * stages == 2 * 128 && k % 128 == 0 && k > 8 * stages * bk) {
            const auto wider = deep_gemm_int8::get_smem_config(2, k, bm, bn, 128, 1);
            if (std::get<0>(wider) <= capacity / column_blocks &&
                    deep_gemm_adaptive::tile_spill_ok(wm, 2 * wn, (bn / (2 * wn)) * column_blocks)) {
                config = {sms, bm, bn, 128, wm, wn * 2, 2, wider};
            }
        }
        if (compact && local_m == 1 && bn == 2 * 64 && bn == 4 * wn && stages == 4 && bk == 64) {
            const int wide_bn = bn + bn / 2;
            const int wide_blocks = gemm_rs_column_rounds(n, wide_bn, sms);
            const auto peak_columns = [&](int columns) {
                const int round = sms * columns;
                return ((n - 1) / round) * columns + std::min(columns, (n - 1) % round + 1);
            };
            const int tail = n % (sms * wide_bn);
            const int native_peak = peak_columns(bn), wide_peak = peak_columns(wide_bn);
            if (wide_peak < native_peak || (wide_peak == native_peak && tail > 0 && tail < wide_bn)) {
                const auto wide_smem = deep_gemm_int8::get_smem_config(stages, k, bm, wide_bn, bk, 1);
                if (std::get<0>(wide_smem) + bm * wide_bn * 2 <= capacity / wide_blocks &&
                        deep_gemm_adaptive::tile_spill_ok(wm, wn, wide_bn / wn * wide_blocks)) {
                    // Three full bulk words per owner reduce the busiest CU's column work.
                    bn = wide_bn;
                    smem = wide_smem;
                }
            }
        }
    }
    // A full short owner word replaces narrower partial packets
    // when all of its column producers can stay resident.
    if (world >= 8 && local_m == 3 && m <= bm && bm == 2 * 16 && bn < 64 &&
            k % 128 == 0 && k >= 4 * 128) {
        constexpr int packet_bn = 64, packet_bk = 128, packet_wn = 16, packet_stages = 4;
        const int blocks = gemm_rs_column_rounds(n, packet_bn, sm_budget);
        use_resident_tile(estimate({bm, packet_bn, packet_bk, bm, packet_wn, packet_stages}), blocks);
    }
    // Streaming weights need enough CTAs to cover the busiest column.
    // Charge both B and the repeatedly loaded A tiles to each wave.
    if (world >= 8 && local_m == 4 && m <= bm && bm == wm && wn == 32 &&
            bn == 4 * wn && bk == 128 && stages >= 3 && weight_bytes > properties->l2CacheSize) {
        constexpr int depth = 3;
        int64_t best_work = std::numeric_limits<int64_t>::max();
        auto best = estimate({bm, bn, bk, wm, wn, stages});
        for (const auto [columns, warp_n] : {std::pair{bn / 2, wn / 2},
                                              std::pair{3 * bn / 4, wn},
                                              std::pair{3 * bn / 2, 3 * wn / 2}}) {
            const auto candidate = estimate({bm, columns, bk, wm, warp_n, depth});
            const int64_t work = streaming_column_work(candidate, sm_budget);
            if (work == std::numeric_limits<int64_t>::max())
                continue;
            if (work < best_work || (work == best_work && columns > best.tile[1])) {
                best = candidate;
                best_work = work;
            }
        }
        use_resident_tile(best, 1);
    }
    // Favor resident short-owner tiles. Keep narrow packet columns
    // unless a wider tile can avoid another compute round.
    const bool short_owner = (world >= 8 && local_m == 3) ||
                             (world == 4 && local_m >= 4 && local_m <= 8);
    if (short_owner && m <= bm && bm == wm && bk == 128 && k % bk == 0 &&
            wn == 32 && bn >= 4 * wn && stages >= 3) {
        const auto resident = [&](int columns, const auto& bytes) {
            return gemm_rs_resident_blocks(wm, wn, columns / wn, std::get<0>(bytes)) * sm_budget;
        };
        if (ceil_div(n, bn) > resident(bn, smem)) {
            constexpr int packet_bn = 2 * 32;
            const int wide_bn = 2 * packet_bn;
            const auto packet_smem = deep_gemm_int8::get_smem_config(2, k, bm, packet_bn, bk, 1);
            const auto wide_smem = deep_gemm_int8::get_smem_config(3, k, bm, wide_bn, bk / 2, 1);
            const bool use_wide = ceil_div(n, packet_bn) > resident(packet_bn, packet_smem) &&
                                  ceil_div(n, wide_bn) <= resident(wide_bn, wide_smem);
            if (world == 4 && use_wide) {
                const auto packet_depth = deep_gemm_int8::get_smem_config(3, k, bm, packet_bn, bk, 1);
                if (ceil_div(n, packet_bn) <= 2 * resident(packet_bn, packet_depth))
                    config = {sm_budget, bm, packet_bn, bk, wm, wn, 3, packet_depth};
                else if (k > 8 * stages * bk)
                    config = {sm_budget, bm, wide_bn, bk, wm, wn, 3,
                              deep_gemm_int8::get_smem_config(3, k, bm, wide_bn, bk, 1)};
                else
                    config = {sm_budget, bm, wide_bn, bk / 2, wm, wn, 3, wide_smem};
            } else {
                config = {sm_budget, bm, use_wide ? wide_bn : packet_bn, use_wide ? bk / 2 : bk,
                          wm, wn, use_wide ? 3 : 2, use_wide ? wide_smem : packet_smem};
            }
        }
    }
    // A wide raw owner can keep three input stages and remove repeated
    // A loads when narrower N tiles would exceed two resident waves.
    if (world > 2 && m <= bm && bm == wm && wm == 16 && local_m <= packet_rows &&
            wn == 32 && (bn == 2 * wn || bn == 4 * wn) && bk <= 128 &&
            !is_single_owner_packet(world, local_m, bn) && weight_bytes > properties->l2CacheSize) {
        constexpr int next_wn = 3 * 16, next_bn = 2 * next_wn, input_k = 128, depth = 3;
        const int narrow_bn = 2 * wn;
        const auto narrow = estimate({bm, narrow_bn, input_k, wm, wn, depth});
        const auto wide = estimate({bm, next_bn, input_k, wm, next_wn, depth});
        const int blocks = wide.resident_blocks();
        if (blocks > 0 && ceil_div(n, narrow_bn) > 2 * narrow.resident_blocks() * sm_budget &&
                ceil_div(n, next_bn) <= 2 * blocks * sm_budget)
            use_resident_tile(wide, blocks);
    }
    if (can_pipeline_channelwise_gemm_rs(config, m, world)) {
        // Balance the sender grid after owner-tail compaction. Two-source
        // reduction retains raw packets and its native wide-warp grid.
        if (world > 2) {
            adapt_gemm_rs_n_grid(config, m, n, sm_budget, true);
            smem = shared_size(stages);
        }
        int native_blocks = int(capacity / std::get<0>(smem));
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
                if (std::get<0>(minimum_smem) + sender_smem <= capacity / 2) {
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
        const int64_t resident_smem = capacity / resident_blocks;
        const bool short_pipeline = ceil_div(k, bk) <= 4 * stages;
        const auto narrow_k = deep_gemm_int8::get_smem_config(4, k, bm, bn, bk / 2, 1);
        // Prefer a single compute wave when the smaller K pipeline fits
        // both the shared-memory and register budgets.
        if (!short_pipeline && std::get<0>(smem) > resident_smem && std::get<0>(narrow_k) <= resident_smem &&
                deep_gemm_adaptive::tile_spill_ok(wm, wn, (bm / wm) * (bn / wn) * resident_blocks)) {
            config = {sms, bm, bn, bk / 2, wm, wn, 4, narrow_k};
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
                std::get<0>(next_smem) <= capacity / 2 &&
                deep_gemm_adaptive::tile_spill_ok(next_wm, next_wn, 2 * m_warps * (bn / next_wn))) {
            config = {sms, next_bm, bn, next_bk, next_wm, next_wn, next_stages, next_smem};
        }
    }
    // Compare producer rounds with the input traffic of each tile.
    const bool partial_grid = world > 2 && m > bm && m % bm != 0 && local_m >= bm;
    if ((local_m % wm == 0 || partial_grid) && wn % 64 == 0 && bk % 128 == 0 && stages >= 3) {
        const int narrow_bn = bn / 4 * 3, narrow_wn = wn / 4 * 3;
        const int narrow_bk = m <= bm ? bk / 2 : bk;
        const auto narrow_smem = deep_gemm_int8::get_smem_config(stages, k, bm, narrow_bn, narrow_bk, 1);
        const auto half_capacity = capacity / 2;
        const int rows = ceil_div(m, bm);
        const int rounds = ceil_div(rows * ceil_div(n, bn), sms);
        const bool direct_tail = m <= bm && ceil_div(n, narrow_bn) <= sm_budget &&
                                 std::get<0>(narrow_smem) <= half_capacity;
        const int narrow_rounds = ceil_div(rows * ceil_div(n, narrow_bn), sm_budget);
        const bool sender_grid = world > 2 && m > bm && bk == 128 && stages == 3 &&
            ceil_div(k, bk) > 4 * stages && std::get<0>(narrow_smem) <= 2 * half_capacity &&
            ((m % bm == 0 && narrow_rounds <= rounds) ||
             (partial_grid && int64_t(narrow_rounds) * (narrow_bn + bk) <
                                  int64_t(rounds) * (bn + bk))) &&
            deep_gemm_adaptive::tile_spill_ok(wm, narrow_wn, (bm / wm) * (bn / wn));
        if (std::get<0>(smem) > half_capacity && (direct_tail || sender_grid)) {
            config = {sms, bm, narrow_bn, narrow_bk, wm, narrow_wn, stages, narrow_smem};
        }
    }
    // Keep a long-K, whole-M producer in two resident column rounds.
    // Compact padded M rows and keep pairs of active producer warps.
    if (world > 2 && packet_rows == 16 && m <= bm && m >= 2 * wm && wm >= 32 && wn >= 64 &&
            bk == 128 && stages >= 3 && ceil_div(k, bk) >= 4 * stages &&
            weight_bytes > properties->l2CacheSize) {
        const int warps = (bm / wm) * (bn / wn);
        const int native_blocks = std::max(1, gemm_rs_resident_blocks(wm, wn, warps, std::get<0>(smem)));
        constexpr int resident = 2, rounds = 2;
        if (ceil_div(n, bn) > rounds * native_blocks * sms) {
            constexpr int warp_n = 3 * 16;
            const int warp_m = wm == 2 * 16 ? 2 * wm : wm;
            const int row_warps = ceil_div(m, warp_m), rows = row_warps * warp_m;
            const int column_group = row_warps % 2 == 0 ? 1 : 2;
            const int columns = ceil_div(n, rounds * resident * sm_budget * warp_n * column_group) *
                                warp_n * column_group;
            // A wider warp group has more work in its final round.
            // Keep a spare half-wave when the N group must grow.
            const bool wide_tail = columns > 3 * warp_n &&
                ceil_div(n, columns) > (rounds * resident - resident / 2) * sm_budget;
            if (rows <= bm && row_warps < bm / wm && !wide_tail)
                use_resident_tile(estimate({rows, columns, bk, warp_m, warp_n, 3}), resident);
        }
    }
    // Expose more resident columns for a complete compact M tile. Keep
    // both the input pipeline depth and the M warp layout.
    if (world > 2 && owner_tail && local_m >= 2 * packet_rows && m >= wm && m <= bm &&
            wn >= 64 && wn % 32 == 0 && std::get<0>(smem) > capacity / 2) {
        const auto try_column_tile = [&](int columns, int warp_n, int blocks) {
            return ceil_div(n, columns) > ceil_div(n, bn) &&
                use_resident_tile(estimate({bm, columns, bk, wm, warp_n, stages}), blocks, 1);
        };
        const int narrow_wn = wn / 2;
        const int narrow_bn = ceil_div(ceil_div(n, sm_budget), narrow_wn) * narrow_wn;
        // Prefer the narrower two-resident tile, then a single-resident tile.
        if (!try_column_tile(narrow_bn, narrow_wn, 2) && wn % 64 == 0 &&
                ceil_div(n, bn / 4 * 3) <= sm_budget)
            try_column_tile(bn / 4 * 3, wn / 4 * 3, 1);
    }
    // Free accumulator registers for compact R16 owners while keeping
    // all columns resident. Narrow the CTA only if it can fill more CUs.
    const bool partial_r16_owner = world > 2 && owner_tail && local_m >= 16 && local_m < 2 * packet_rows;
    if (partial_r16_owner && m <= bm && m >= wm && wn >= 64 && wn % 32 == 0 && wm * wn / 32 >= 96) {
        const int next_wn = wn / 2;
        const int next_bn = bn % (2 * wn) == 0 && ceil_div(n, bn) < sm_budget ? bn / 2 : bn;
        const int resident = gemm_rs_column_rounds(n, next_bn, sm_budget);
        if (resident <= 2)
            use_resident_tile(estimate({bm, next_bn, bk, wm, next_wn, stages}), resident);
    }
    // A whole-M tile with several row warps can fill unused column
    // slots by splitting M. Spend the released TSM on the input prologue.
    if (world > 2 && m <= bm && m > bm / 2 && local_m >= 8 && local_m <= 16 &&
            bm == 4 * wm && wn == 32 && bn == 2 * wn && stages == 3 && ceil_div(k, bk) <= 3 * stages) {
        const int blocks = ceil_div(2 * ceil_div(n, bn), sm_budget);
        const int next_bm = bm / 2, next_wn = wn / 2, next_stages = stages + 1;
        if (blocks <= 3 && weight_bytes <= properties->l2CacheSize)
            use_resident_tile(estimate({next_bm, bn, bk, wm, next_wn, next_stages}), blocks);
    }
    // Split a wide bulk-packet fragment when the smaller output
    // columns can use the same number of resident input rounds.
    if (m <= bm && bm == wm && stages == 2 && wn >= 4 * 16 && wn % 32 == 0 &&
            is_single_owner_packet(world, local_m, bn) && weight_bytes > properties->l2CacheSize) {
        const int next_bn = bn / 2;
        const auto narrow = estimate({bm, next_bn, bk, wm, wn / 2, stages});
        const int blocks = narrow.resident_blocks(bm * next_bn * 2);
        if (blocks > 0 && column_rounds(next_bn, blocks, sm_budget) <= column_rounds(bn, current().resident_blocks(), sms))
            use_resident_tile(narrow, blocks);
    }
    // A short R4 owner can write native packets from one M warp.
    if (world > 2 && owner_tail && packet_rows == 4 && m >= wm && m < bm) {
        const int columns = ceil_div(n, bn), warps = bn / wn;
        const int resident_blocks = ceil_div(columns, sm_budget);
        if (warps >= 4 && columns <= sm_budget + sm_budget / warps)
            use_resident_tile(estimate({bm, bn, bk, bm, wn, stages}), resident_blocks);
    }
    // Pair N tiles when the original producers need a second wave.
    // A 16-warp CTA also budgets staged sender storage when applicable.
    const int wide_bn = 2 * bn, wide_wn = 2 * wn;
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
                config = {sms, bm, wide_bn, bk, wm, wide_wn, wide_stages, wide_smem};
            }
        }
    }
    // Reduce the warp fragment only when the original column tiles
    // need another resident wave.
    if (world > 2 && m == bm && bm == wm && local_m == packet_rows && wm % 32 == 0 &&
            wn >= 32 && bn % (2 * wn) == 0 && stages >= 3 && ceil_div(n, bn) > sms) {
        const int native_blocks = std::max(1, gemm_rs_resident_blocks(wm, wn, warps, std::get<0>(smem)));
        const int narrow_bn = bn / 2;
        const int blocks = gemm_rs_column_rounds(n, narrow_bn, sm_budget);
        const auto minimum_smem = deep_gemm_int8::get_smem_config(3, k, bm, narrow_bn, bk, 1);
        const int native_tiles = ceil_div(n, bn);
        const int free_slots = native_blocks * sms - native_tiles;
        // A nearly full native wave benefits from the wider warp. Keep
        // the same warp if a shallower pipe just admitted one more slot.
        const int deeper_blocks = stages == 3 ?
            gemm_rs_resident_blocks(wm, wn, warps, std::get<0>(shared_size(stages + 1))) : 0;
        const int extra_tiles = native_tiles - deeper_blocks * sms;
        const bool fits_after_shallowing = stages == 3 && free_slots >= 0 &&
            extra_tiles > 0 && 3 * extra_tiles <= sms;
        const bool filled_native = native_tiles > 2 * sms && free_slots >= 0 &&
            (3 * free_slots <= sms || fits_after_shallowing);
        if ((packet_rows == 4 || !filled_native) &&
                std::get<0>(minimum_smem) <= capacity / blocks &&
                deep_gemm_adaptive::tile_spill_ok(wm / 2, wn, warps * blocks)) {
            bn = narrow_bn;
            wm /= 2;
            sms = sm_budget;
            adapt_gemm_rs_stages(config, 3, blocks, 0, shared_size);
        }
    }
    // Use a compact whole-M tile when narrowing warp N or removing
    // a producer round keeps its columns in one wave.
    const bool short_k_tile = world >= 8 && m <= bm && bn == 2 * wn && wn == 32 &&
                              bk == 64 && k % 128 == 0 && wm >= 3 * 16;
    if (world > 2 && owner_tail && packet_rows == 16 && stages >= 3 &&
            ((bn >= 128 && bk == 128) || short_k_tile)) {
        constexpr int m_warps = 4, wide_wn = 3 * 16, wide_bn = 4 * wide_wn;
        // Use more N producers with a narrower warp accumulator when
        // replacing a large native fragment or a shallow K tile.
        const int compact_wm = ceil_div(m, m_warps * 16) * 16;
        const int next_wm = m <= bm && wm >= 3 * 16 ? wm : compact_wm;
        const int next_bm = m <= bm && next_wm == wm ? bm : m_warps * next_wm;
        // Use narrower accumulators to remove a producer wave or fill
        // both dimensions of a whole-M tile.
        constexpr int narrow_wn = 2 * 16;
        const int narrow_bn = ceil_div(ceil_div(n, sm_budget), narrow_wn) * narrow_wn;
        const bool can_fill_more = narrow_bn < wide_bn && int64_t(5) * m >= int64_t(4) * next_bm;
        const bool needs_columns = ceil_div(n, wide_bn) > sm_budget;
        const bool narrower_warps = world >= 8 &&
            ((wn >= 64 && (can_fill_more || needs_columns)) || short_k_tile);
        const int next_wn = narrower_warps ? narrow_wn : wide_wn;
        const int next_bn = narrower_warps ? narrow_bn : wide_bn;
        const int columns = ceil_div(n, next_bn);
        const int current_warps = (bm / wm) * (bn / wn);
            const int resident_blocks = std::max(1, std::min(int(capacity / std::get<0>(smem)),
            3 * deep_gemm_adaptive::WE_PER_CU / current_warps));
        const int current_tiles = ceil_div(m, bm) * ceil_div(n, bn);
        const bool removes_wave = current_tiles > sms * resident_blocks;
        const bool narrower_fragments = stages > 3 && wn > next_wn && columns > current_tiles && 2 * columns >= sm_budget;
        const bool whole_tile_n_wave = m <= bm && (removes_wave || narrower_fragments);
        if (columns <= sm_budget && (5 * columns >= 4 * sm_budget || whole_tile_n_wave) &&
                (wn > next_wn || short_k_tile || removes_wave)) {
            use_resident_tile(estimate({next_bm, next_bn, 128, next_wm, next_wn, 3}), 1);
        }
    }
    // Short streaming owners can keep their M warps when a deep input
    // tile still needs too many resident column rounds.
    if (world == 4 && owner_tail && packet_rows <= 8 && m <= bm && bm > wm && bk > 128 && stages >= 4 &&
            weight_bytes > properties->l2CacheSize) {
        constexpr int columns = 4 * 16, narrow_wn = 16, input_k = 128, depth = 3;
        const int rows = ceil_div(m, wm) * wm;
        const auto narrow = estimate({rows, columns, input_k, wm, narrow_wn, depth});
        const int blocks = narrow.resident_blocks();
        const int rounds = column_rounds(columns, blocks, sm_budget);
        const int native_rounds = column_rounds(bn, current().resident_blocks(), sms);
        if (rows > ceil_div(m, 16) * 16 && rows <= bm && blocks >= 2 && native_rounds >= (bk / input_k) * rounds) {
            // Prefer wider column warps only when they lower the busiest CU's work.
            constexpr int wide_warp_n = 3 * 16, wide_columns = 2 * wide_warp_n;
            const auto wide = estimate({rows, wide_columns, input_k, wm, wide_warp_n, depth});
            const int wide_blocks = wide.resident_blocks();
            if (wide_blocks >= 2 && column_rounds(wide_columns, wide_blocks, sm_budget) * (wide_columns + 2 * rows) <
                                    rounds * (columns + 2 * rows))
                use_resident_tile(wide, wide_blocks);
            else
                use_resident_tile(narrow, blocks);
        }
    }
    // A full second resident column wave can reuse a deeper K window
    // across the row warps without giving up sender occupancy.
    if (world == 2 && m <= bm && m > 2 * wm && bm / wm >= 4 && stages == 3 &&
            ceil_div(k, bk) > 4 * stages) {
        const int columns = ceil_div(n, bn);
        const auto deeper = estimate({bm, bn, bk, wm, wn, stages + 1});
        if (2 * columns >= 3 * sms && columns <= 2 * sms && deeper.resident_blocks() >= 2) {
            ++stages;
            smem = deeper.smem;
        }
    }
    // Short K can trade one input stage for two resident narrow producers.
    // Keep the tiles already narrowed by the main resident-grid rules.
    if (world >= 8 && partial_r16_owner && m <= bm && m >= wm && wn >= 64 && wn % 32 == 0 &&
            wm * wn / 32 >= 96 && bk == 128 && stages > 3 && ceil_div(k, bk) <= 4 * stages &&
            bn % (2 * wn) == 0 && ceil_div(n, bn) < sm_budget &&
            ceil_div(n, bn / 4) <= 4 * sm_budget) {
        use_resident_tile(estimate({bm, bn / 4, bk, wm, wn / 2, stages - 1}), 2);
    }
    // Keep one native R4 owner per narrow column group while increasing
    // producer parallelism for a single row warp.
    if (world == 4 && m == bm && bm == wm && local_m == 4 &&
            bk == 128 && bn >= 4 * 16 && ceil_div(k, bk) >= 4 * 3) {
        constexpr int columns = 2 * 16, warp_columns = 16, depth = 3;
        const auto narrow = estimate({bm, columns, bk, wm, warp_columns, depth});
        if (ceil_div(n, columns) <= narrow.resident_blocks() * sm_budget)
            use_tile(narrow);
    // These short-owner alternatives are mutually exclusive. The coalesced
    // two-row tile keeps its priority over subsequent grid refinements.
    } else if (world >= 8 && m == bm && local_m == packet_rows && packet_rows >= 8 &&
            bk >= 128 && stages >= 3) {
        const int rounds = column_rounds(bn, current().resident_blocks(), sms);
        for (const int width : {bn / 2, bn}) {
            const int input_k = width < bn && packet_rows <= 8 ? std::min(bk, 128) : bk / 2;
            const int depth = input_k == 128 ? 3 : std::max(4, stages);
            const int warp_n = width < bn && packet_rows <= 8 ? 16 : wn;
            if (input_k < 64 || width < 2 * 16 || width % warp_n != 0 || k % input_k != 0)
                continue;
            const auto candidate = estimate({bm, width, input_k, wm, warp_n, depth});
            if (ceil_div(n, width) <= candidate.resident_blocks() * sm_budget && (rounds > 1 || warp_n < wn)) {
                use_tile(candidate);
                break;
            }
        }
    // Finish the full-M tile in one wave with a wider row warp.
    } else if (world >= 8 && m == 2 * bm && local_m == wm && !owner_tail && bk == 64 &&
            wn % 32 == 0 && stages >= 4 && ceil_div(n, bn) <= sm_budget &&
            ceil_div(k, 2 * bk) <= 4 * 3) {
        constexpr int depth = 3;
        const auto bytes = deep_gemm_int8::get_smem_config(depth, k, m, bn, 2 * bk, 1);
        const int output_warps = 2 * (bm / wm) * (bn / wn);
        if (std::get<0>(bytes) <= capacity &&
                deep_gemm_adaptive::tile_spill_ok(2 * wm, wn / 2, output_warps))
            config = {sm_budget, m, bn, 2 * bk, 2 * wm, wn / 2, depth, bytes};
    // Short compact packets can shrink K to keep a third column tile resident.
    } else if (world >= 8 && m == bm && bm == wm && local_m <= 2 &&
            is_single_owner_packet(world, local_m, bn) && bk == 128 && stages == 3 && k % 64 == 0) {
        const int columns = ceil_div(n, bn);
        if (columns > 2 * sms && columns <= 3 * sms) {
            const auto narrow = estimate({bm, bn, 64, wm, wn, 4});
            if (columns <= narrow.resident_blocks() * sm_budget)
                use_tile(narrow);
        }
    }
    // Preserve the original column count when testing the later owner split,
    // which is deferred until the preceding N-widening rule has run.
    const int columns_before_widening = ceil_div(n, bn);
    // A full column wave can retain the narrower warp fragment while
    // sharing each M tile across more N columns.
    if (world > 2 && m == bm && bm / wm >= 4 && local_m % 16 == 0 &&
            bk == 128 && stages >= 3 && wn == 2 * 16 && bn == 2 * wn &&
            n % (2 * bn) == 0 && ceil_div(n, 2 * bn) * 4 >= 3 * sm_budget &&
            ceil_div(n, bn) > sms && ceil_div(n, 2 * bn) <= sm_budget &&
            ceil_div(k, bk) <= 8 * 3) {
        constexpr int depth = 3;
        const bool split_rows = world >= 8 && bm % (2 * wm) == 0 && ceil_div(k, bk) < 4 * depth;
        const int rows = split_rows ? bm / 2 : bm;
        const int active_warps = (rows / wm) * (2 * bn / wn);
        const auto bytes = deep_gemm_int8::get_smem_config(depth, k, rows, 2 * bn, bk, 1);
        if (std::get<0>(bytes) <= capacity &&
                deep_gemm_adaptive::tile_spill_ok(wm, wn, active_warps))
            config = {sm_budget, rows, 2 * bn, bk, wm, wn, depth, bytes};
    }
    // Use one complete owner per M tile when the shorter input pipeline
    // keeps the additional producer tiles resident.
    if (world >= 8 && !owner_tail && 2 * local_m < bm && bk == 128 && stages >= 3 &&
            local_m % 32 == 0 && ceil_div(k, bk) <= 2 * stages && columns_before_widening < sms) {
        const int warp_n = std::min(32, wn);
        const auto owner = estimate({local_m, bn, bk, 32, warp_n, stages - 1});
        if (bn % warp_n == 0 && world * columns_before_widening <= owner.resident_blocks() * sm_budget)
            use_tile(owner);
    }
    // Keep a balanced M/N warp grid for short partial owner shards when
    // the smaller fragments fit in the same resident producer rounds.
    const int panel = 4 * wm;
    if (world >= 8 && wm == 3 * 16 && local_m % packet_rows != 0 &&
            local_m >= 2 * packet_rows && local_m <= wm + 2 * packet_rows &&
            m > panel && m < 3 * panel && bm >= panel && (bm != panel || bn != panel) &&
            bk == 128 && stages == 3 && (bm / wm) * (bn / wn) >= 2 * deep_gemm_adaptive::WE_PER_CU) {
        const auto balanced = estimate({panel, panel, bk, wm, wm, stages});
        const int blocks = balanced.resident_blocks();
        const int current_blocks = std::max(1, current().resident_blocks());
        const int tiles = ceil_div(m, panel) * ceil_div(n, panel);
        const int current_tiles = ceil_div(m, bm) * ceil_div(n, bn);
        if (blocks > 0 && ceil_div(tiles, blocks * sm_budget) <= ceil_div(current_tiles, current_blocks * sms))
            use_tile(balanced);
    }
    return finish();
}

} // namespace deep_gemm::gemm_comm
