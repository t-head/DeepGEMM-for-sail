#pragma once
#include <algorithm>
#include <cstdint>

#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"

using namespace deep_gemm;
namespace deep_gemm_tf32_common {

struct PrenormConfig {
    int block_m, block_n, block_k;
    int num_splits;
    int smem_size;
};

// Dynamic shared memory consumed by the kernel: the CuTe path double-buffers padded A/B tiles,
// while the fused 890P path always claims 32KB.
// NOTES: this mirrors the `kSmemSize` computed inside `HcPrenormGemm::run` in
// `tf32_hc_prenorm_gemm.cuh`, keep both in sync
static int get_smem_config(const int& block_m, const int& block_n, const int& block_k) {
    const auto& smem_cute = static_cast<int>(2 * (block_m * (block_k + 8) * sizeof(uint16_t) +
                                                  block_n * (block_k + 4) * sizeof(float)));
    constexpr int kSmemFused = 32768;
    return std::max(smem_cute, kSmemFused);
}

// Split K so that every split still owns a couple of K blocks
static int get_num_splits(const int& k, const int& block_k) {
    const int& k_blocks = k / block_k;
    const int& num_splits = k_blocks >= 448 /* K >= 28672 */ ? 16 : std::min(32, k_blocks);
    return std::max(num_splits, 1);
}

// NOTES: `m` does not take part in the selection yet, `block_m` is pinned to 64 because the kernel
// hardcodes `BLOCK_M * 2` threads per block
static PrenormConfig get_best_configs(const int& m, const int& n, const int& k) {
    constexpr int block_m = 64;
    constexpr int block_k = 64;
    // NOTE: bind by value, not `const int&` -- std::min returns a reference to one of its
    // temporary arguments, which would dangle past the end of this statement.
    const int block_n = std::min(align(n, is_ppu1v5_device() ? 8 : 16), 32);

    DG_HOST_ASSERT(n <= block_n);
    DG_HOST_ASSERT(n <= 32 and n % 8 == 0);
    DG_HOST_ASSERT(k % block_k == 0);

    return PrenormConfig{block_m, block_n, block_k, get_num_splits(k, block_k),
                         get_smem_config(block_m, block_n, block_k)};
}

} // namespace deep_gemm_tf32_common
