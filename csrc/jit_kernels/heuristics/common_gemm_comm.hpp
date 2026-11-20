#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <torch/python.h>
#include "../../utils/utils.hpp"
#include <deep_gemm/gemm_comm/reduce_scatter.cuh>

namespace deep_gemm::gemm_comm {

inline int get_gemm_rs_packet_rows(int64_t local_m) {
    return local_m < 8 ? 4 : (local_m < 16 ? 8 : 16);
}

using ConfigTuple = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;
using Fp8GemmRSTile = std::array<int, 6>;
using Fp8GemmRSConfig = ConfigTuple;

inline bool is_blockwise_gemm_rs_n_tile(int bn) {
    return bn > 0 && (bn == 192 || 128 % bn == 0 || bn % 128 == 0);
}

inline void validate_gemm_rs_shape(int m, int n, int k, int world, bool requires_block_scales = true) {
    TORCH_CHECK(world == 1 || world == 2 || world == 4 || world == 8, "GEMM/RS supports TP1/TP2/TP4/TP8");
    TORCH_CHECK(m > 0 && m % world == 0 && n > 0 && n % 16 == 0 && k > 0,
                "GEMM/RS requires positive shapes, M divisible by TP and N%16=0");
    TORCH_CHECK(requires_block_scales ? (k > 512 && k % 128 == 0) : k % 32 == 0,
                requires_block_scales ? "blockwise GEMM/RS requires K>512 and K%128=0" : "GEMM/RS requires K%32=0");
    // Packet offsets use signed 32-bit element indices, including padded rows.
    const int64_t lm = m / world;
    const int packet = get_gemm_rs_packet_rows(lm);
    const int64_t packet_rows = (lm + 2 * packet - 2) / packet * packet;
    TORCH_CHECK(packet_rows * world * n <= std::numeric_limits<int>::max(),
                "GEMM/RS packet indices exceed int32 capacity");
}

inline void validate_gemm_rs_tile(const Fp8GemmRSTile& tile, bool channelwise = false) {
    const auto [bm, bn, bk, wm, wn, stages] = tile;
    TORCH_CHECK(std::all_of(tile.begin(), tile.end(), [](int value) { return value > 0; }) && bm % wm == 0 &&
                    bn % wn == 0 && wm % 16 == 0 && wn % 16 == 0 && (bk == 64 || bk == 128 || bk == 256) && stages >= 2,
                "invalid GEMM/RS tile geometry; requires BK64/128/256 and at least two stages");
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    const int64_t threads = int64_t(bm / wm) * (bn / wn) * 32;
    TORCH_CHECK(threads >= (channelwise ? 64 : 128) && threads <= properties->maxThreadsPerBlock,
                "GEMM/RS requires at least two channelwise or four blockwise warps and a valid block size");
    const int64_t capacity = std::max(properties->sharedMemPerBlock, properties->sharedMemPerBlockOptin);
    TORCH_CHECK(int64_t(bm) * bn * 2 <= capacity, "GEMM/RS output tile exceeds shared-memory capacity");
    const int64_t per_stage = (int64_t(bm) + bn) * bk;
    TORCH_CHECK(per_stage <= capacity && stages <= capacity / per_stage,
                "GEMM/RS pipeline exceeds shared-memory capacity");
}

// Reuse the native runtime LUT and heuristics, with explicit configs first.
// Recompute SMEM for
// every selected tile; the caller-provided SMEM entry is not authoritative.
inline std::optional<Fp8GemmRSTile> get_gemm_rs_tile(const std::optional<ConfigTuple>& configs) {
    if (!configs)
        return std::nullopt;
    const auto& [sms, bm, bn, bk, wm, wn, stages, smem] = *configs;
    TORCH_CHECK(sms > 0 && sms <= get_num_sms(), "GEMM/RS configs exceed the current SM budget");
    return Fp8GemmRSTile{bm, bn, bk, wm, wn, stages};
}

inline int64_t gemm_rs_smem_capacity() {
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    return std::max(properties->sharedMemPerBlock, properties->sharedMemPerBlockOptin);
}

// Select the deepest input pipeline that preserves the required compute
// residency after reserving sender storage. Callers retain native stages if
// no valid candidate fits; runtime admission still checks compiled occupancy.
template <class SmemConfig>
inline void adapt_gemm_rs_stages(ConfigTuple& config, int minimum_stages, int resident_blocks,
                                 int64_t sender_smem, const SmemConfig& shared_size) {
    auto& [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    const int64_t capacity = gemm_rs_smem_capacity() / resident_blocks;
    for (int candidate = stages; candidate >= minimum_stages; --candidate) {
        const auto candidate_smem = shared_size(candidate);
        if (std::get<0>(candidate_smem) + sender_smem <= capacity) {
            stages = candidate;
            smem = candidate_smem;
            return;
        }
    }
}

// Keep the shared sender budget identical in stage planning and runtime admission.
inline int get_gemm_rs_preferred_shared_packets(const ConfigTuple& config,
                                                  int available_packets = std::numeric_limits<int>::max()) {
    const auto& [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    const int packets = wm * wn / 256;
    return std::min({packets, std::max(2, packets / 2), available_packets}) & ~1;
}

// Share N-grid adaptation across quantizations. A small native grid can leave
// CUs idle; split N while respecting the native minimum warp count.
inline void adapt_gemm_rs_n_grid(ConfigTuple& config, int m, int n, int target_grid, bool channelwise,
                                  bool (*supports_n_tile)(int) = nullptr,
                                  int max_grid = std::numeric_limits<int>::max(), bool keep_warp_n = false) {
    auto& [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    const int native_bn = bn;
    while (bn >= 64) {
        const int64_t tiles = int64_t((m + bm - 1) / bm) * ((n + bn - 1) / bn);
        if (tiles >= target_grid || (channelwise && wn == 16 && tiles >= sms))
            break;
        const int next_bn = bn / 2, next_wn = keep_warp_n ? wn : std::max(16, wn / 2);
        const int64_t next_tiles = int64_t((m + bm - 1) / bm) * ((n + next_bn - 1) / next_bn);
        if (next_tiles > max_grid || next_wn % 16 != 0 || next_bn % next_wn != 0 || (bm / wm) * (next_bn / next_wn) < (channelwise ? 2 : 4) ||
            (supports_n_tile && !supports_n_tile(next_bn)))
            break;
        bn = next_bn;
        wn = next_wn;
    }
    if (channelwise && bn != native_bn) {
        bk = std::min(bk, std::max(128, 2 * bn));
        // Keep the deepest native pipeline with the residency of three stages.
        // Output storage shares the input box; sender admission is checked later.
        const int64_t capacity = gemm_rs_smem_capacity();
        const int64_t stage_bytes = int64_t(bm + bn) * bk;
        const int64_t minimum_smem = std::max(int64_t(bm) * bn * 2, std::min(stages, 3) * stage_bytes);
        const int resident_blocks = int(capacity / minimum_smem);
        stages = std::min(stages, int(capacity / resident_blocks / stage_bytes));
    }
}

// Share the boundary-owner tile adjustment across FP8 quantizations.
template <class SmemConfig>
inline void adapt_gemm_rs_m_tile(ConfigTuple& config, int m, int n, int world, int sm_budget, int minimum_warps,
                                const SmemConfig& shared_size) {
    auto& [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    const int padded_m = ceil_div(m, bm) * bm;
    const int local_m = m / world, packet_rows = get_gemm_rs_packet_rows(local_m);
    const bool boundary = local_m >= 3 && local_m % packet_rows != 0;
    const int parallel_tiles = std::min(sms, ceil_div(m, bm) * ceil_div(n, bn));
    int rows = ceil_div(m, wm) * wm, columns = bn, warp_m = wm, warp_n = wn;
    int input_stages = stages;
    // Short owners amortize the duplicated boundary packets with compact
    // payloads. Longer owners and two-source reduction retain their native grid.
    const bool compact_equal_work = world > 2 && boundary && local_m < 2 * packet_rows &&
                                    m > bm && rows == padded_m;
    if (compact_equal_work) {
        warp_m = std::max(wm, wn);
        warp_n = std::min(wm, wn);
        rows = ceil_div(m, warp_m) * warp_m;
        if (m < warp_m || rows > padded_m)
            return;
        // Preserve the accumulator area while exposing enough N tiles.
        while (columns >= 128 && (columns & (columns - 1)) == 0 &&
                ceil_div(n, columns) < sm_budget && columns / 2 >= warp_n &&
                (columns / 2) % warp_n == 0)
            columns /= 2;
        // Avoid a new register/occupancy regime: retain the block's warp
        // budget, and its M-warp count when transposing the warp axes.
        if (int64_t(rows / warp_m) * (columns / warp_n) > int64_t(bm / wm) * (bn / wn) ||
                (warp_m != wm && rows / warp_m > bm / wm))
            return;
        input_stages = std::min(stages, 3);
    } else {
        if (rows >= padded_m)
            return;
        // Removing padded M work must retain compute parallelism.
        if (boundary && int64_t(ceil_div(m, bm)) * ceil_div(n, bn) < 2 * int64_t(sms)) {
            while (columns >= 128 && (columns & (columns - 1)) == 0 &&
                    (ceil_div(n, columns) < parallel_tiles ||
                     int64_t(rows / warp_m) * (columns / warp_n) * 32 >
                         at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock)) {
                columns /= 2;
                warp_n = std::max(16, warp_n / 2);
            }
        }
        if (columns != bn || rows < bm)
            input_stages = std::min(stages, 3);
    }
    const int64_t threads = int64_t(rows / warp_m) * (columns / warp_n) * 32;
    if (columns % warp_n != 0 || threads < minimum_warps * 32 ||
            threads > at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock ||
            int64_t(rows) * columns * 2 > gemm_rs_smem_capacity())
        return;
    const auto candidate = shared_size(rows, columns, input_stages);
    if (std::get<0>(candidate) > gemm_rs_smem_capacity())
        return;
    if (compact_equal_work || columns != bn || warp_n != wn)
        sms = sm_budget;
    bm = rows;
    bn = columns;
    wm = warp_m;
    wn = warp_n;
    stages = input_stages;
    smem = candidate;
}


} // namespace deep_gemm::gemm_comm
