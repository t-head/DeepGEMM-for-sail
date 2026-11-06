#pragma once

namespace deep_gemm::nvfp4::detail {

// Shared by host resource accounting and device specializations. These are
// layout capabilities, independent of the heuristic's measured M/N/K ranges.
constexpr bool is_shared_decode_tile(int block_m, int block_n, int block_k,
                                     int warp_m, int warp_n, int warp_k) {
    return block_k == 128 && warp_k == 128 && warp_n == 32 &&
        ((warp_m == 64 && ((block_m == 128 && block_n == 128) ||
                          (block_m == 256 && block_n == 64))) ||
         (block_m == 64 && block_n == 64 && warp_m == 32));
}

constexpr bool use_fast_block_scales(int k, int block_k, int warp_k, int n_expand,
                                     int a_scale_group, int b_scale_block) {
    return a_scale_group == 128 && b_scale_block == 128 &&
        (block_k == 128 || block_k == 256) && warp_k == 128 &&
        k % block_k == 0 && n_expand == 1;
}

constexpr bool can_stage_block_scales(int block_m, int block_n, int block_k, int warp_m) {
    return block_k == 128 && block_m >= 32 &&
        (block_n % 128 == 0 || (block_m == 256 && block_n == 64) ||
         (block_m == 64 && block_n == 64 && warp_m == 32));
}

} // namespace deep_gemm::nvfp4::detail
