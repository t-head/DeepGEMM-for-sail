#pragma once
#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"
#include <deep_gemm/impls/mqa_logits_types.cuh>

using namespace deep_gemm;
namespace deep_gemm_mqa_common {

// Tile / stage selection for the non-paged MQA logits kernel
struct MqaLogitsConfig {
    int block_q, block_qh, block_kv;
    int warp_qh, warp_kv;
    int num_q_stages, num_kv_stages;
    // FP4 groups M blocks with positive G; FP8 retains its output-specific default.
    int m_block_group = 0;
    bool q_owner_epilogue = false;
    bool balance_output_tiles = false;
};

// Logits row stride must be 1024-byte aligned
static int get_logits_stride_alignment(torch::ScalarType logits_dtype) {
    return 1024 / (logits_dtype == torch::kFloat32 ? 4 : 2);
}

// `stride_k` is `uint32_t` when `q_idx * stride_k` still fits in 32 bits, otherwise `uint64_t`.
// Chosen on the host and forwarded as a JIT template key.
static std::string select_stride_k_type(int aligned_seq_len, int aligned_seq_len_kv) {
    DG_HOST_ASSERT(aligned_seq_len_kv < (1LL << 31));
    const auto& product = static_cast<int64_t>(aligned_seq_len) * aligned_seq_len_kv;
    return product < (1LL << 32) ? "uint32_t" : "uint64_t";
}

// Dynamic shared memory of the non-paged kernel, mirroring `PPUMqaLogits::SharedStorage` (and
// `PPUMqaLogitsFP4::SharedStorage` when `is_fp4`):
//   smem_k / smem_q (values) + smem_k_scales + [smem_q_sf, FP4 only] + smem_weight
// Both flavours store 4 bytes per KV row scale (`float` vs packed `uint32_t` e8m0); FP4 adds a
// per-Q-row scale on top.
// The host supplies this size at launch time. The generated code carries a `static_assert`
// against `SharedStorageSize`, so a mismatch fails at compile time.
static int get_smem_config(const MqaLogitsConfig& config, int head_dim, int qk_element_size,
                           int weights_element_size, bool is_fp4) {
    const int smem_k = config.block_kv * head_dim * config.num_kv_stages * qk_element_size;
    const int smem_q = config.block_qh * head_dim * config.num_q_stages * qk_element_size;
    const int smem_k_scales = config.block_kv * config.num_kv_stages * 4;
    const int smem_q_sf = is_fp4 ? config.block_qh * config.num_q_stages * 4 : 0;
    const int smem_weight = config.block_qh * config.num_q_stages * weights_element_size;
    return smem_k + smem_q + smem_k_scales + smem_q_sf + smem_weight;
}

// Mirrors `MaxThreadsPerBlock = size(TiledMma{})`
static int get_num_threads(const MqaLogitsConfig& config) {
    return (config.block_qh / config.warp_qh) * (config.block_kv / config.warp_kv) * 32;
}

// ============================ Paged =============================

// Pipeline stages, KV splitting, warp shape and occupancy selected for paged MQA.
struct PagedTile {
    int stage_q, stage_k, split_kv, warp_kv;
    bool split_mblock;
    int tb_per_cu;
};

// Fixed M890 FP4 DSA grouped plan, resolved on the host and passed to JIT kernels.
// Query capacity is independent of the number of valid rows in a request.
struct FP4PagedVarlenConfig {
    static constexpr uint32_t kQueryHeadRows = 128;
    static constexpr uint32_t kBlockKV = 64;
    static constexpr uint32_t kWarpKV = 64;
    static constexpr uint32_t kMathWarps = 8;
    static constexpr uint32_t kQStages = 2;
    static constexpr uint32_t kKVSlots = 4;
    static constexpr bool kSplitMBlock = true;

    // Keep eight math warps per CTA for both supported head counts: H32
    // uses split128, H64 split256. Their shared storage fits five/three
    // resident blocks respectively; the schedule uses the same block counts.
    static constexpr uint32_t split_kv(uint32_t num_heads) {
        return kMathWarps * kWarpKV * num_heads / kQueryHeadRows;
    }
    static constexpr uint32_t blocks_per_cu(uint32_t num_heads) {
        return num_heads == 32 ? 5 : 3;
    }

    // Work scales with valid queries. Packed singletons need a bandwidth
    // floor because they hide less KV latency than multi-query blocks.
    static constexpr uint32_t kSingletonCostMultiplier = 2;
};

static PagedTile get_fp4_varlen_tile(int num_heads) {
    using Config = FP4PagedVarlenConfig;
    return {Config::kQStages, Config::kKVSlots, static_cast<int>(Config::split_kv(num_heads)), Config::kWarpKV,
            Config::kSplitMBlock, static_cast<int>(Config::blocks_per_cu(num_heads))};
}

// Dynamic shared memory of the paged kernel, mirroring `PPUPagedMqaLogits::SharedStorage`.
// NOTES: the M extent is `split_kv` (all math warp groups), not `block_kv` -- verified against
// `SharedStorageSize`, and the generated code carries a `static_assert` to keep it honest.
static int get_paged_smem_config(const PagedTile& tile, int next_n, int num_heads, int head_dim,
                                 int qk_element_size, int weights_element_size, bool is_fp4) {
    const int block_m = tile.split_kv;
    const int block_n = next_n * num_heads;
    const int smem_q = block_n * head_dim * tile.stage_q * qk_element_size;
    const int smem_k = block_m * head_dim * tile.stage_k * qk_element_size;
    const int smem_k_scale = block_m * tile.stage_k * 4;
    const int smem_q_scale = is_fp4 ? block_n * tile.stage_q * 4 : 0;
    const int smem_weight = block_n * tile.stage_q * weights_element_size;
    return smem_q + smem_k + smem_k_scale + smem_q_scale + smem_weight;
}

// Mirrors `MaxThreadsPerBlock = kNumMathWarpGroups * size(TiledMma{})`
static int get_paged_num_threads(const PagedTile& tile, int next_n) {
    return next_n * (tile.split_kv / tile.warp_kv) * 32;
}

// 810E INT8 AVG packs H4/H8 queries into 16-column warp tiles.
// Metadata and compute use this same plan; three KV stages hide page loads.
static PagedTile get_int8_avg_paged_tile() {
    return {2, 3, 256, 64, false, 2};
}

// Paged tile lookup with resource-based search for unlisted shapes.
// NOTES: `weights_size` / `logits_size` are hardcoded to 4 upstream even when the tensors are
// BF16; preserve that accounting for the existing weighted paths.
static PagedTile get_paged_mqa_logits_tile(int next_n, int page_kv, int num_heads, int head_dim, int datasize) {
    DG_HOST_ASSERT(page_kv == 32 or (page_kv > 0 and page_kv % 64 == 0));
    const bool is_fp4 = datasize == 1 and head_dim == 64;
    if (is_fp4 and num_heads == 8) {
        DG_HOST_ASSERT(next_n >= 1 and next_n <= 6);
        // H8 packs two queries into each 16-column warp tile.
        return {2, 3, 256, 64, false, next_n <= 4 ? 4 : 3};
    }
    constexpr int weights_size = 4, logits_size = 4;

    const auto& calc_smem = [&](int split_kv, int stage_q, int stage_k) {
        const int block_m = split_kv;
        const int block_n = next_n * num_heads;
        return block_n * head_dim * stage_q * datasize + block_m * head_dim * stage_k * datasize +
               block_m * stage_k * 4 + (is_fp4 ? block_n * stage_q * 4 : 0) + block_n * stage_q * weights_size;
    };
    const auto& calc_vreg = [&](int warp_kv) {
        const int k_block_max = head_dim * datasize / 32;
        const int mma_n = num_heads / 16;
        const int mma_m = warp_kv / 16;
        const int orig_vreg = 18 + 4 * mma_n * logits_size / 4 + 8 * mma_m * mma_n + 4 * mma_m * k_block_max +
                               4 * mma_n * k_block_max + (is_fp4 ? 2 : 0);
        return ceil_div(orig_vreg, 8) * 8;
    };
    const auto& calc_num_threads = [&](int split_kv, int warp_kv) {
        return next_n * (split_kv / warp_kv) * 32;
    };
    const auto& calc_tb_per_sm = [&](int num_threads, int vreg, int smem) {
        // max 8 warp slots per warp engine (64 warps / 8 WEs)
        const int warps_per_we = std::min(512 / vreg, 8);
        const int num_warps = 8 * warps_per_we;
        return std::min(num_warps / (num_threads / 32), 262144 / smem);
    };

    const auto& search_tile = [&]() {
        std::vector<PagedTile> tile_list;
        for (int split_kv : {64, 128, 256}) {
            for (int warp_kv : {16, 32, 64}) {
                const int num_threads = calc_num_threads(split_kv, warp_kv);
                if (num_threads < 128)
                    continue;
                if (split_kv / warp_kv > 4 and num_threads != 512)
                    continue;
                constexpr int stage_k = 3;
                for (bool split_mblock : {false, true}) {
                    if (split_mblock and not(is_fp4 and warp_kv >= 32))
                        continue;
                    const int vreg = calc_vreg(split_mblock ? 16 : warp_kv);
                    // Compare stage_q=1 and stage_q=2, pick one based on tb_per_sm
                    std::vector<std::pair<int, int>> candidates;
                    for (int stage_q : {1, 2}) {
                        // warp-interleave does not support stage_q = 1
                        if (num_threads == 512 and stage_q == 1)
                            continue;
                        const int tb_per_sm = calc_tb_per_sm(num_threads, vreg, calc_smem(split_kv, stage_q, stage_k));
                        if (tb_per_sm == 0)
                            continue;
                        candidates.emplace_back(stage_q, tb_per_sm);
                    }
                    if (candidates.empty())
                        continue;
                    const auto& pick = candidates.size() == 1 or candidates[0].second > candidates[1].second
                                           ? candidates[0]
                                           : candidates[1];
                    tile_list.push_back({pick.first, stage_k, split_kv, warp_kv, split_mblock, pick.second});
                }
            }
        }
        DG_HOST_ASSERT(not tile_list.empty());
        // Prefer more blocks per CU, then more Q stages, then unsplit M blocks.
        // std::max_element retains the first candidate when the keys tie.
        const auto& key = [](const PagedTile& t) {
            return std::make_tuple(t.tb_per_cu, t.stage_q, 1 - static_cast<int>(t.split_mblock));
        };
        return *std::max_element(tile_list.begin(), tile_list.end(),
                                 [&](const PagedTile& a, const PagedTile& b) { return key(a) < key(b); });
    };

    static const std::map<std::tuple<int, int, int, int>, PagedTile> tile_map = {
        // next_n = 1 tiles
        {{1, 1, 32, 64}, {2, 3, 256, 64, false, 4}},
        {{1, 1, 64, 64}, {2, 3, 256, 64, false, 4}},
        {{1, 1, 32, 128}, {1, 3, 64, 16, false, 8}},
        {{1, 1, 64, 128}, {2, 3, 64, 16, false, 6}},
        // AVG tiles use 16 MMA columns and ceil(next_n / 4) query groups.
        // FP4 D128 is packed into 64 bytes; FP8 D128 uses 128 bytes.
        {{1, 1, 16, 64}, {2, 3, 256, 64, false, 4}},
        {{1, 2, 16, 64}, {2, 3, 256, 64, false, 4}},
        {{1, 1, 16, 128}, {1, 3, 128, 32, false, 4}},
        {{1, 2, 16, 128}, {1, 3, 128, 32, false, 4}},
    };
    if (num_heads == 4) {
        num_heads = 16;
        next_n = (next_n + 3) / 4;
    }
    const auto& it = tile_map.find({datasize, next_n, num_heads, head_dim});
    if (it != tile_map.end())
        return it->second;
    return search_tile();
}

// Resident Q capacity before optional KV balancing. The resident-CTA count is
// a measured per-CU hint from the FP4 backend calibration, not a live occupancy
// query or tensor-utilization estimate. Recheck it when kernel resources change.
static int get_mqa_q_rows_per_wave(int block_q, int num_sms, int resident_ctas_per_cu) {
    return block_q * num_sms * resident_ctas_per_cu;
}

// Prefer more Q workers once Q32 exceeds its calibrated resident capacity.
// Medium KV uses the same Q16 tile through Q4096.
static bool prefer_fp4_avg_short_kv_tile(int seq_len_q, int seq_len_k,
                                        int num_sms, bool is_compressed) {
    if (is_compressed
        or seq_len_q <= std::max(2048, get_mqa_q_rows_per_wave(32, num_sms, 2)))
        return false;
    return seq_len_k < 2048 or (seq_len_q <= 4096 and seq_len_k <= 4096);
}

// Five geometry/stage candidates for BF16 FP4 AVG. Select G separately.
static MqaLogitsConfig select_fp4_avg_tile(int seq_len_q, int seq_len_k,
                                         int num_sms, bool is_compressed) {
    // H4, WARP_QH=16 and one Q stage are shared by all candidates.
    constexpr auto make_tile = [](int block_q, int block_kv, int warp_kv, int kv_stages) {
        return MqaLogitsConfig{block_q, block_q * 4, block_kv, 16, warp_kv, 1, kv_stages};
    };
    static constexpr auto tile_q16_kv128 = make_tile(16, 128, 64, 3);
    static constexpr auto tile_q16_kv256 = make_tile(16, 256, 64, 4);
    static constexpr auto tile_q16_kv512 = make_tile(16, 512, 128, 4);
    static constexpr auto tile_q32_kv128_s3 = make_tile(32, 128, 64, 3);
    static constexpr auto tile_q32_kv128_s4 = make_tile(32, 128, 64, 4);

    if (seq_len_k < 1024
        or prefer_fp4_avg_short_kv_tile(seq_len_q, seq_len_k, num_sms, is_compressed))
        return tile_q16_kv128;
    // KV is the total length, not compressed row capacity: a CTA processes
    // the union of Q windows. On 39 CUs these capacities are 624/1248/2496 Qs.
    if (seq_len_q <= get_mqa_q_rows_per_wave(tile_q16_kv512.block_q, num_sms, 1))
        return tile_q16_kv512;
    // Compressed output stays with BQ16: measured BQ32 variants use local storage.
    if (is_compressed or seq_len_q <= get_mqa_q_rows_per_wave(tile_q16_kv256.block_q, num_sms, 2))
        return tile_q16_kv256;
    // S4 avoids the QRow register increase measured with S3 at larger Q.
    return seq_len_q <= get_mqa_q_rows_per_wave(tile_q32_kv128_s3.block_q, num_sms, 2)
        ? tile_q32_kv128_s3 : tile_q32_kv128_s4;
}

// Group a full or half warp's M blocks; 16 mirrors PPUMqaLogitsFP4::InstM.
static int get_fp4_avg_m_block_group(int seq_len_q, int num_sms,
                                     bool is_compressed, const MqaLogitsConfig& config) {
    const int m_blocks = config.warp_kv / 16;
    if (is_compressed and config.block_kv >= 256) {
        // Full grouping measured two CTAs/CU; G2 measured three. After two
        // full-group Q waves, prefer G2's lower register pressure.
        const int full_group_q_wave = get_mqa_q_rows_per_wave(config.block_q, num_sms, 2);
        return seq_len_q > 2 * full_group_q_wave ? m_blocks / 2 : m_blocks;
    }
    if ((config.block_q == 16 and config.num_kv_stages == 3)
        or (config.block_q == 32 and config.num_kv_stages == 4))
        return m_blocks / 2;
    return m_blocks;
}

// FP8 AVG uses a fixed tile.
static MqaLogitsConfig get_fp8_avg_configs(torch::ScalarType logits_dtype) {
    return {32, 128, 128, 16, 64, 1, 3,
            logits_dtype == torch::kBFloat16 ? 1 : 0};
}

// Apply lane ownership and KV balancing after tile/G selection.
// This only sets host permissions; the kernel checks the actual KV windows.
static void configure_fp4_avg_lane_and_kv_balance(MqaLogitsConfig& config, int seq_len_q, int seq_len_k,
                                                 int num_sms, bool is_compressed) {
    if (seq_len_k < 1024)
        return;
    if (is_compressed) {
        // Runtime probes retain Q-only scheduling for short windows.
        config.balance_output_tiles = true;
        config.q_owner_epilogue = true;
        return;
    }
    config.balance_output_tiles = seq_len_k >= 8192 or (seq_len_q < 128 and seq_len_k >= 2048);
    config.q_owner_epilogue = config.balance_output_tiles
        or (seq_len_q >= 128 and seq_len_q <= get_mqa_q_rows_per_wave(32, num_sms, 2));
}

// FP4 AVG H4 policy: tile, M-block group, then lane ownership and KV balancing.
static MqaLogitsConfig get_fp4_avg_configs(int seq_len_q, int seq_len_k,
                                         int num_sms, bool is_compressed) {
    auto config = select_fp4_avg_tile(seq_len_q, seq_len_k, num_sms, is_compressed);
    config.m_block_group = get_fp4_avg_m_block_group(seq_len_q, num_sms, is_compressed, config);
    configure_fp4_avg_lane_and_kv_balance(config, seq_len_q, seq_len_k, num_sms, is_compressed);
    return config;
}

// The public API validates AVG head counts and FP4 packed head_dim == 64.
// Reuse the host's scale mode; no duplicate AVG/scale/dimension flags are needed.
static MqaLogitsConfig get_best_configs(torch::ScalarType qk_dtype, int seq_len_q, int seq_len_k,
                                       int num_heads, torch::ScalarType logits_dtype,
                                       bool is_fp4, int num_sms, bool is_compressed,
                                       QScaleMode scale_mode) {
    DG_HOST_ASSERT(num_sms > 0);
    // One geometry for H4/H8; CTA-synchronized 512-thread execution is faster than interleave on 810E.
    if (scale_mode == kScaleModeQHead)
        return {64 / num_heads, 64, 128, 16, 32, 1, 3};
    const bool is_avg = scale_mode != kScaleModeWeights;
    if (is_avg) {
        // One H8 geometry for all Q-scale modes.
        if (is_fp4 and num_heads == 8)
            return {8, 64, 256, 16, 64, 1, 3, 2, false, seq_len_k >= 1024};
        if (is_fp4)
            return get_fp4_avg_configs(seq_len_q, seq_len_k, num_sms, is_compressed);
        return get_fp8_avg_configs(logits_dtype);
    }

    MqaLogitsConfig config{};
    config.m_block_group = logits_dtype == torch::kBFloat16 ? 1 : 0;
    config.block_q = 4;
    config.block_qh = config.block_q * num_heads;
    config.warp_qh = num_heads;

    if (is_fp4) {
        // FP4 DSA is BF16-only with H32/H64; AVG returned above.
        config.block_kv = 256;
        config.warp_kv = 64;
        config.num_q_stages = 1;
        config.num_kv_stages = num_heads == 64 ? 4 : 3;
    } else {
        config.num_q_stages = 1;
        config.num_kv_stages = 3;
        // FP8 / INT8 share the same tiling.
        config.warp_kv = num_heads == 32 ? 32 : 64;
        config.block_kv = config.warp_kv * 4;
    }

    return config;
}

} // namespace deep_gemm_mqa_common
