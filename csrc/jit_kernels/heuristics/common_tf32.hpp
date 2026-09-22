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
};

// Dynamic shared memory consumed by the kernel: the CuTe path keeps `num_stages` padded A/B tiles,
// while the fused 890P path always claims 32KB.
// NOTES: this mirrors the `kSmemSize` computed inside `HcPrenormGemm::run` in
// `tf32_hc_prenorm_gemm.cuh`, keep both in sync
static int get_smem_config(const int& block_m, const int& block_n, const int& block_k,
                           const int& num_stages) {
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
    const int block_m =
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

    return PrenormConfig{block_m, block_n, block_k, get_num_splits(m, k, block_m, block_k),
                         num_threads, num_stages,
                         get_smem_config(block_m, block_n, block_k, num_stages)};
}

} // namespace deep_gemm_tf32_common
