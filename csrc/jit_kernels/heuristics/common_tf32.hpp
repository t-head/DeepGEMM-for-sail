#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"

using namespace deep_gemm;
namespace deep_gemm_tf32_common {

struct PrenormConfig {
    int block_m, block_n, block_k;
    int num_splits;
    int num_threads, num_stages;
    int smem_size;
    uint32_t kernel_variant;
};

// Keep shared-memory sizing consistent with the device template.
static int get_smem_config(const int& block_m, const int& block_n, const int& block_k,
                           const int& num_stages, uint32_t kernel_variant = 0) {
    if (is_ppu1v5_device()) {
        const int a_bytes = (kernel_variant & 1) ? block_m * 64 * 2 : 8192;
        return 2 * (a_bytes + 8192) * (block_k / 64);
    }
    const auto& smem_cute = static_cast<int>(num_stages * (block_m * (block_k + 8) * sizeof(uint16_t) +
                                                           block_n * (block_k + 4) * sizeof(float)));
    const int kSmemFused = block_k == 128 ? 65536 : 32768;
    return std::max(smem_cute, kSmemFused);
}

static int get_env_int(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr or *raw == '\0')
        return 0;
    const int parsed = std::atoi(raw);
    return parsed > 0 ? parsed : 0;
}

// Largest `m` for which raising the split cap from 32 to 64 is still a win, or 0 when it never is
static int get_split64_m_limit(const int& k_blocks) {
    if (is_ppu1v5_device())
        return k_blocks >= 448 ? 32 : 0;
    return k_blocks >= 448 ? 128 : (k_blocks >= 256 ? 32 : 0);
}

// The scheduler's CTA quantum on 810E, measured at block_m=64 / block_k=64 / n<=32
static constexpr int kOneWave810E = 140;

static int get_wave_model_target(const int& grid_m, const int& k_blocks, const int& cap) {
    if (grid_m * 2 <= kOneWave810E)
        return kOneWave810E / grid_m;
    int best_ns = 2;
    int64_t best_cost = INT64_MAX;
    for (int ns = 2; ns <= cap; ns *= 2) {
        const int64_t cta = int64_t(grid_m) * ns;
        const int64_t waves = (cta + kOneWave810E - 1) / kOneWave810E;
        const int64_t cost = waves * kOneWave810E * int64_t(k_blocks) * 1024 / cta +
                             int64_t(ns) * 1024;
        if (cost < best_cost) {
            best_cost = cost;
            best_ns = ns;
        }
    }
    return best_ns;
}

// Split K so that every split still owns a couple of K blocks
static int get_num_splits(const int& m, const int& k, const int& block_m, const int& block_k) {
    const int k_blocks = k / block_k;
    const int forced = get_env_int("DG_TF32_NUM_SPLITS");
    // NOTES: must stay a power of two -- the kernel's last-CTA test relies on `2^32 % num_splits == 0`
    if (forced > 0 and (forced & (forced - 1)) == 0 and k_blocks % forced == 0 and k_blocks / forced >= 2)
        return forced;
    const int grid_m = (m + block_m - 1) / block_m;
    const int cap = grid_m > 2 ? 16 : (m <= get_split64_m_limit(k_blocks) ? 64 : 32);
    int target = std::min(cap, std::max(1, 256 / grid_m));
    if (not is_ppu1v5_device() and grid_m >= 16 and k_blocks >= 256)
        target = std::min(cap, get_wave_model_target(grid_m, k_blocks, cap));
    int num_splits = 1;
    while (num_splits * 2 <= target)
        num_splits *= 2;
    while (num_splits > 1 and (k_blocks % num_splits != 0 or k_blocks / num_splits < 2))
        num_splits /= 2;
    if (num_splits == 1 and is_ppu1v5_device()) {
        const int one_wave = get_num_sms() * (block_k == 128 ? 4 : 8);
        if (grid_m * 2 <= one_wave and k_blocks % 2 == 0 and k_blocks / 2 >= 2)
            num_splits = 2;
    }
    return num_splits;
}

// NOTES: `block_m` defaults to 64, from which the kernel derives `BLOCK_M * 2` threads per block
static PrenormConfig get_best_configs(const int& m, const int& n, const int& k) {
    const int forced_block_m = get_env_int("DG_TF32_BLOCK_M");
    int block_m =
        (forced_block_m > 0 and forced_block_m % 16 == 0 and forced_block_m <= 256) ? forced_block_m : 64;
    const int forced_block_k = get_env_int("DG_TF32_BLOCK_K");
    const int block_k =
        ((forced_block_k == 64 or forced_block_k == 128) and k % forced_block_k == 0) ? forced_block_k : 64;
    // NOTE: bind by value, not `const int&` -- std::min returns a reference to one of its
    // temporary arguments, which would dangle past the end of this statement.
    const int block_n = std::min(align(n, is_ppu1v5_device() ? 8 : 16), 32);

    DG_HOST_ASSERT(n <= block_n);
    DG_HOST_ASSERT(n <= 32 and n % 8 == 0);
    DG_HOST_ASSERT(k % block_k == 0);

    const int mma_threads = block_m * 2;
    const int forced_threads = is_ppu1v5_device() ? 0 : get_env_int("DG_TF32_NUM_THREADS");
    const int num_threads =
        (forced_threads >= mma_threads and forced_threads % 32 == 0 and forced_threads <= 1024)
            ? forced_threads : mma_threads;
    const int forced_stages = is_ppu1v5_device() ? 0 : get_env_int("DG_TF32_NUM_STAGES");
    const int num_stages = (forced_stages >= 2 and forced_stages <= 8) ? forced_stages : 2;

    // Preserve explicit overrides.
    int num_splits = get_num_splits(m, k, block_m, block_k);
    const bool tune_890p = is_ppu1v5_device() and n == 24 and
                          forced_block_m == 0 and forced_block_k == 0 and
                          get_env_int("DG_TF32_NUM_SPLITS") == 0;
    if (tune_890p) {
        const bool shorter_k = k == 16384 or k == 20480;
        if ((m <= 20 and shorter_k) or (m <= 32 and k == 20480))
            num_splits = 64;
    }
    const bool tune_810e = n == 24 and block_n == 32 and
                          forced_block_m == 0 and forced_block_k == 0 and
                          forced_threads == 0 and forced_stages == 0 and
                          get_env_int("DG_TF32_NUM_SPLITS") == 0;
    if (tune_810e) {
        // Warp 3 reduces the square sum without changing its summation order.
        if (not is_ppu1v5_device() and m == 16 and k == 28672) {
            constexpr int bm = 16, bk = 64, threads = 128, stages = 2;
            constexpr uint32_t variant = 1u << 20;
            constexpr int smem = stages * (bm * (bk + 8) * sizeof(uint16_t) +
                                          32 * (bk + 4) * sizeof(float));
            return PrenormConfig{bm, block_n, bk, 64, threads, stages, smem, variant};
        }
        // Load-only warps help these small shapes; the MMA and reduction stay unchanged.
        const bool known_k = k == 16384 or k == 20480 or k == 28672;
        // Use exact TC01 stage storage.
        if (known_k and (m == 60 or m == 128 or (m == 32 and k == 16384))) {
            constexpr int bm = 32, bk = 64, stages = 2;
            const int threads = m == 32 ? 256 : 128;
            constexpr uint32_t variant = 256u | 512u;
            constexpr int smem = stages * (bm * (bk + 8) * sizeof(uint16_t) +
                                          32 * (bk + 4) * sizeof(float));
            return PrenormConfig{bm, block_n, bk, m == 128 ? 32 : 64,
                                threads, stages, smem, variant};
        }
        if (known_k and (m == 24 or (m == 32 and k != 16384))) {
            return PrenormConfig{64, block_n, 64, 64, 256, 2,
                get_smem_config(64, block_n, 64, 2), 0};
        }
        // TC01: bit 256 = packed loads; bit 512 = quad reuse.
        struct Tuned810Config { int m, block_m, num_splits; };
        static constexpr Tuned810Config configs[] = {
            { 1248, 128,  8},
            { 4096, 128, 16},
            { 8192, 256, 16},
            { 9984, 128,  1},
            {19968, 128,  1},
        };
        if (k == 16384 or k == 20480 or k == 28672) {
            for (const auto& c : configs) {
                if (m == c.m) {
                    const bool shuffle_a = m <= 8192 or (m == 9984 and k != 28672) or
                                           (m == 19968 and k == 16384);
                    // Round B once per call for reuse across CTAs.
                    const bool rounded_b = m == 4096 or m == 8192 or m == 9984 or m == 19968;
                    // Specialize async-copy addressing for complete M tiles; keep a tail fallback.
                    const bool full_tile_copy = m == 8192 or m == 9984 or m == 19968;
                    // Each warp reuses B across two M16 groups.
                    const bool reuse_b = m == 8192;
                    const uint32_t variant = 256u | ((shuffle_a or rounded_b) ? 512u : 0u) |
                                             (rounded_b ? 1024u : 0u) | (full_tile_copy ? 2048u : 0u) |
                                             (reuse_b ? 4096u : 0u);
                    return PrenormConfig{c.block_m, block_n, 64, c.num_splits,
                        c.block_m * 2, 2, get_smem_config(c.block_m, block_n, 64, 2, variant), variant};
                }
            }
        }
    }
    uint32_t kernel_variant = 0;
    if (tune_890p) {
        // Measured N=24 configurations; variant bits are defined in the device template.
        struct TileConfig { int block_m, num_splits; uint32_t variant; };
        struct TunedConfig { int m; TileConfig by_k[3]; };
        // K columns: 16384, 20480, 28672; entries: {block_m, splits, variant}.
        const int k_index = k == 16384 ? 0 : k == 20480 ? 1 : k == 28672 ? 2 : -1;
        static constexpr TunedConfig configs[] = {
            {   16, {{ 64, 64,  27}, { 64, 64,  27}, { 64, 64,  27}}},
            {   20, {{ 64, 64,  43}, { 64, 64,  43}, { 64, 64,  43}}},
            {   24, {{ 64, 64,  19}, { 64, 64,  19}, { 64, 64,  59}}},
            {   32, {{ 64, 64,  19}, { 64, 64,  27}, { 64, 64,  19}}},
            {   60, {{ 32, 64,  19}, { 32, 64,  19}, { 32, 32,   3}}},
            {  128, {{ 32, 32,   3}, { 32, 32,   7}, { 32, 32,   3}}},
            { 1248, {{ 96, 16,   7}, { 96, 16,   7}, { 96, 16,   7}}},
            { 4096, {{ 64,  4,   7}, {112,  4,   7}, { 96,  4,  71}}},
            { 8192, {{ 64,  2,   7}, { 64,  2, 135}, {112,  2,   7}}},
            { 9984, {{ 64,  2,   5}, { 64,  2,   5}, { 64,  2,   5}}},
            {19968, {{ 64,  1,   7}, {128,  1,   7}, {128,  1,   5}}},
        };
        for (const auto& c : configs) {
            if (m == c.m and k_index >= 0) {
                const auto& tile = c.by_k[k_index];
                block_m = tile.block_m;
                num_splits = tile.num_splits;
                kernel_variant = tile.variant;
                // Split-major scratch and extra reduction warps preserve the sum order.
                if (m == 24 || m == 32 || m == 60 ||
                    (m == 128 && (k == 16384 || k == 20480)))
                    kernel_variant |= (1u << 22) | (1u << 23);
                break;
            }
        }
    }
    // The two measured M-reuse schedules cover 32 rows per physical warp.
    const int warp_m_groups = (kernel_variant & (64u | 128u)) ? 2 : 1;
    const int launch_threads = (tune_890p && (kernel_variant & (1u << 23))) ? 256 :
                               (tune_890p ? block_m * 2 / warp_m_groups : num_threads);
    return PrenormConfig{block_m, block_n, block_k, num_splits,
                         launch_threads, num_stages,
                         get_smem_config(block_m, block_n, block_k, num_stages, kernel_variant), kernel_variant};
}

} // namespace deep_gemm_tf32_common
