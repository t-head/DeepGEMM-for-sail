#pragma once
#include <algorithm>
#include <tuple>
#include "../../utils/exception.hpp"
#include "../../utils/utils.hpp"

namespace deep_gemm::w4a8 {
struct W4A8Config {
    int num_sms;
    int block_m, block_n, block_k;
    int warp_m, warp_n, warp_k;
    int num_stages, n_expand;
};
using W4A8ConfigTuple = std::tuple<int, int, int, int, int, int, int, int, int>;

static W4A8Config unpack_config(const W4A8ConfigTuple& tuple) {
    const auto& [sms,bm,bn,bk,wm,wn,wk,stages,expand] = tuple;
    return {sms,bm,bn,bk,wm,wn,wk,stages,expand};
}
static W4A8ConfigTuple pack_config(const W4A8Config& c) {
    return {c.num_sms,c.block_m,c.block_n,c.block_k,c.warp_m,c.warp_n,c.warp_k,c.num_stages,c.n_expand};
}
static int get_num_threads(const W4A8Config& c) {
    return (c.block_m/c.warp_m)*(c.block_n/c.warp_n)*(c.block_k/c.warp_k)*32;
}

static void check_config(const W4A8Config& c, int n, int k, GemmType kind) {
    DG_HOST_ASSERT(c.num_sms > 0 and c.num_sms <= get_num_sms());
    DG_HOST_ASSERT(c.block_m == 16 or c.block_m == 32 or c.block_m == 64 or c.block_m == 128);
    DG_HOST_ASSERT(c.warp_m == 16 or c.warp_m == 32 or c.warp_m == 64);
    DG_HOST_ASSERT(c.block_m % c.warp_m == 0 and c.warp_n == 64);
    DG_HOST_ASSERT(c.block_n >= 64 and c.block_n % 64 == 0);
    // PPU1.5 AIU minimum 64B swizzle requires 128 packed-FP4 K elements.
    DG_HOST_ASSERT(c.block_k == 128 and (c.warp_k == 64 or c.warp_k == 128));
    DG_HOST_ASSERT(k % c.block_k == 0 and c.num_stages >= 2 and c.num_stages <= 4);
    DG_HOST_ASSERT(c.n_expand >= 1 and c.n_expand <= 4);
    if (c.n_expand > 1) {
        DG_HOST_ASSERT(kind == GemmType::GroupedNoPad and c.num_stages == 2);
        DG_HOST_ASSERT(c.block_k == c.warp_k);
        DG_HOST_ASSERT(n % (c.block_n * c.n_expand) == 0);
    }
    DG_HOST_ASSERT(c.block_m == c.warp_m or c.block_k == c.warp_k);
    DG_HOST_ASSERT(get_num_threads(c) <= 1024);
}

static int get_smem_size(const W4A8Config& c) {
    const int mainloop = (c.block_m*c.block_k + c.block_n*c.block_k/2 +
                         c.block_n*c.block_k/32) * c.num_stages;
    const int reduction = 16*64*(c.block_n/64)*(c.block_k/c.warp_k)*4;
    return std::max(mainloop, reduction);
}

// Bound M before rounding so even a large tuning hint cannot overflow.
static W4A8Config get_best_configs(int expected_m, int n, int k, int groups, GemmType kind) {
    DG_HOST_ASSERT(expected_m >= 0 && groups > 0 && n > 0 && n % 64 == 0 && k > 0);
    DG_HOST_ASSERT(kind == GemmType::GroupedFused || kind == GemmType::GroupedNoPad || kind == GemmType::GroupedMasked);
    int bm = 16;
    while (bm < expected_m && bm < 64) bm *= 2;
    const int bn = n >= 256 ? 256 : n >= 128 ? 128 : 64;
    // M32/N256 has enough shared memory for three stages at four CTAs/SM
    // (64,512 bytes per CTA). M16 and M64 lose residency with a third stage.
    const int stages = kind == GemmType::GroupedMasked && bm == 32 && bn == 256 ? 3 : 2;
    W4A8Config c{get_num_sms(),bm,bn,128,bm,64,128,stages,1};
    if (kind == GemmType::GroupedNoPad and k <= 512 and k % c.block_k == 0 and
        c.num_stages == 2 and c.block_k == c.warp_k) {
        for (const int candidate : {4, 3, 2}) {
            if (n % (bn * candidate) == 0) {
                c.n_expand = candidate;
                break;
            }
        }
    }
    check_config(c,n,k,kind);
    return c;
}
static const char* get_gemm_type_name(GemmType kind) {
    switch (kind) {
        case GemmType::GroupedFused: return "GroupedFused";
        case GemmType::GroupedNoPad: return "GroupedNoPad";
        case GemmType::GroupedMasked: return "GroupedMasked";
        default: DG_HOST_ASSERT(false);
    }
    return "";
}
} // namespace deep_gemm::w4a8
