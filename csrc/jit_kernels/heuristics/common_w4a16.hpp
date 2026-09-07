#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <torch/torch.h>

#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"

using namespace deep_gemm;
namespace deep_gemm_w4a16_common {

// Mirrors `W4A16Type` in `deep_gemm/jit_kernels/m_grouped_gemm_w4a16.py`: the quantization layout
// and dequant path selector
enum class W4A16Type {
    int4,
    mxfp4_e8m0,
    mxfp4_bf16,
    mxfp4_e8m0_mma,
};

// Faithful port of `get_w4a16_type`: a `uint8` weight means the packed-FP4 MMA kernel, otherwise the
// scale dtype (or the caller's override) picks between int4, raw E8M0 and BF16 scales
static W4A16Type get_w4a16_type(torch::ScalarType rhs_dtype, torch::ScalarType rhs_scales_dtype,
                                bool fp4_use_bf16_scale) {
    if (rhs_dtype == torch::kUInt8) {
        // w4fa16_mma is only supported on PPU1.5
        DG_HOST_ASSERT(is_ppu1v5_device());
        return W4A16Type::mxfp4_e8m0_mma;
    }
    if (fp4_use_bf16_scale)
        return W4A16Type::mxfp4_bf16;
    if (rhs_scales_dtype == torch::kUInt8)
        return W4A16Type::mxfp4_e8m0;
    return W4A16Type::int4;
}

// `W4A16Gemm` routes `uint8_t` weights to `W4A16GEMM_MMA` and everything else to `W4A16GEMM`
static bool uses_mma_kernel(W4A16Type w4a16_type) {
    return w4a16_type == W4A16Type::mxfp4_e8m0_mma;
}

// `ElementB` / `ElementScale` spelled out for the generated code. Kept as the exact strings the
// Python template used: `W4A16Gemm::run` branches on `is_same_v<ElementB, cutlass::int4b_t>`, so
// renaming these would change which dequant path the kernel picks.
static std::string get_element_b(W4A16Type w4a16_type) {
    if (w4a16_type == W4A16Type::int4)
        return "int4_t";
    return w4a16_type == W4A16Type::mxfp4_e8m0_mma ? "uint8_t" : "cutlass::float4_t";
}

static std::string get_element_scale(W4A16Type w4a16_type) {
    return w4a16_type == W4A16Type::mxfp4_e8m0 or w4a16_type == W4A16Type::mxfp4_e8m0_mma ? "uint8_t" : "bfloat16_t";
}

// `data_type` reported to the profiler, mirroring the `dtype_name` ladder in `W4A16Gemm::run`
static std::string get_profiling_dtype_name(W4A16Type w4a16_type) {
    switch (w4a16_type) {
        case W4A16Type::mxfp4_e8m0_mma: return "w4fa16_mma";
        case W4A16Type::int4:           return "w4a16";
        case W4A16Type::mxfp4_e8m0:     return "w4fa16";
        default:                        return "w4fa16_s16";
    }
}

// `GemmType::{}` as spelled in the generated code, mirroring the Python `gemm_type.name`
static std::string get_gemm_type_name(GemmType gemm_type) {
    switch (gemm_type) {
        case GemmType::GroupedNoPad:  return "GroupedNoPad";
        case GemmType::GroupedMasked: return "GroupedMasked";
        case GemmType::GroupedFused:  return "GroupedFused";
        default: DG_HOST_ASSERT(false);
    }
    return {};
}

// The 9-tuple returned by `w4a16_get_best_configs`
struct W4A16Config {
    int num_sms;
    int block_m, block_n, block_k;
    int warp_m, warp_n, warp_k;
    int num_stages;
    int n_expand;
};

// The same 9-tuple as the callers spell it through pybind:
// (num_sms, block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages, n_expand)
using W4A16ConfigTuple = std::tuple<int, int, int, int, int, int, int, int, int>;

// NOTES: unlike the FP8 path there is nothing to recompute while unpacking -- `W4A16Config` carries no
// derived member, so the tuple maps onto it field by field
static W4A16Config unpack_w4a16_config(const W4A16ConfigTuple& configs) {
    const auto& [num_sms, block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages, n_expand] = configs;
    return W4A16Config{num_sms, block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages, n_expand};
}

// Faithful port of `w4a16_get_best_configs`
// NOTES: `num_groups` is unused upstream too -- kept so the signature still reads like the Python one
static W4A16Config get_best_configs(int expected_m, int n, int k, int num_groups, int num_sms, GemmType gemm_type,
                                    W4A16Type w4a16_type) {
    const bool is_ppu1v5 = is_ppu1v5_device();

    // `for block_m in [16, 32, 64, 128]: if expected_m / block_m < 0.9: break` -- the loop variable
    // keeps the value it broke on and falls through to 128 when nothing breaks
    int block_m = 16;
    for (const int candidate : {16, 32, 64, 128}) {
        block_m = candidate;
        if (static_cast<double>(expected_m) / block_m < 0.9)
            break;
    }
    const int warp_m = block_m <= 64 ? block_m : 64;

    int block_n, warp_n, block_k, warp_k, num_stages;
    if (w4a16_type == W4A16Type::mxfp4_e8m0_mma) {
        // w4fa16_mma is only supported on PPU1.5
        DG_HOST_ASSERT(is_ppu1v5);
        const bool warps_on_k = k >= 2048;
        static const std::map<std::pair<int, bool>, std::array<int, 5>> tile_list = {
            {{64, false}, {512, 64, 128, 128, 2}},
            {{64, true},  {256, 64, 128, 64, 3}},
            {{32, true},  {256, 64, 128, 64, 3}},
            // {{16, true},  {256, 64, 128, 64, 3}},
        };
        const auto& it = tile_list.find({block_m, warps_on_k});
        const std::array<int, 5> tile = it != tile_list.end() ? it->second : std::array<int, 5>{256, 64, 128, 128, 2};
        block_n = tile[0], warp_n = tile[1], block_k = tile[2], warp_k = tile[3], num_stages = tile[4];
    } else {
        const bool warps_on_k = block_m == warp_m and k >= 2048;
        if (warps_on_k) {
            if (is_ppu1v5) // warps_on_n = 4, warps_on_k = 4
                block_n = 256, warp_n = 64, block_k = 128, warp_k = 32;
            else // warps_on_n = 2, warps_on_k = 8
                block_n = 128, warp_n = 64, block_k = 256, warp_k = 32;
        } else {
            block_n = 256, warp_n = 64, block_k = 64, warp_k = 64;
        }
        num_stages = k <= 512 ? 2 : 3;
    }

    int n_expand = 1;
    if (is_ppu1v5 and k <= 512 and k % block_k == 0 and num_stages == 2 and block_k == warp_k) {
        for (const int candidate : {4, 3, 2, 1}) {
            n_expand = candidate;
            if (n % (block_n * n_expand) == 0)
                break;
        }
        if (warp_m == 64 and gemm_type == GemmType::GroupedFused and w4a16_type == W4A16Type::mxfp4_e8m0_mma)
            n_expand = 1; // n_expand > 1 will cause vreg exceeds the 256 limit
    }

    return W4A16Config{num_sms, block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages, n_expand};
}

// Mirrors `MaxThreadsPerBlock = size(TiledMma{})`.
// NOTES: `W4A16GEMM` hardcodes `WarpsOnN = BlockN / 64` while `W4A16GEMM_MMA` uses `BlockN / WarpN`;
// both agree because `warp_n` is asserted to be 64, but each is spelled out as written.
static int get_warps_on_n(const W4A16Config& config, bool is_mma) {
    return is_mma ? config.block_n / config.warp_n : config.block_n / 64;
}

static int get_num_threads(const W4A16Config& config, bool is_mma) {
    return (config.block_m / config.warp_m) * get_warps_on_n(config, is_mma) * (config.block_k / config.warp_k) * 32;
}

// Dynamic shared memory of the W4A16 kernels, mirroring `W4A16GEMM::SharedStorage` (and
// `W4A16GEMM_MMA::SharedStorage` when `is_mma`). A union overlays the mainloop buffers with the two
// epilogue ones, so the total is the largest arm:
//   mainloop            : smem_a + smem_b + smem_scale
//   epilogue cta reduce : smem_reduce
//   epilogue            : smem_c -- the MMA kernel stores straight out, so it has no such arm
// NOTES: `W4A16Gemm::run` just took `sizeof(Kernel::SharedStorage)`; the C++ JIT has to know the size
// before the kernel is compiled, hence the replica here. The generated code carries a `static_assert`
// against the real `sizeof`, so any drift fails at compile time rather than silently.
static int get_smem_config(const W4A16Config& config, bool is_mma, int scale_element_size, int group_size) {
    constexpr int kElementASize = 2;   // cutlass::bfloat16_t
    constexpr int kElementAccSize = 4; // float
    constexpr int kElementDSize = 2;   // cutlass::bfloat16_t

    const int warps_on_m = config.block_m / config.warp_m;
    const int warps_on_n = get_warps_on_n(config, is_mma);
    const int warps_on_k = config.block_k / config.warp_k;

    const int smem_a = config.block_m * config.block_k * config.num_stages * kElementASize;
    // `SmemLayoutB` is (BlockK / 16, BlockN * 2) of `int`, or (BlockN, BlockK / 2) of `uint8_t` for MMA
    const int smem_b = is_mma ? config.block_n * (config.block_k / 2) * config.num_stages
                              : (config.block_k / 16) * (config.block_n * 2) * config.num_stages * 4;
    // `SmemLayoutScale` is (BlockK / kGroupSize, BlockN), or (BlockN / 64, BlockK * 2) of `uint8_t` for MMA
    const int smem_scale = is_mma
                               ? (config.block_n / 64) * (config.block_k * 2) * config.num_stages
                               : (config.block_k / group_size) * config.block_n * config.num_stages * scale_element_size;
    const int smem_reduce = 16 * 64 * warps_on_n * warps_on_k * kElementAccSize;
    // `SmemLayoutC` is `EpilogueConfig::SmemLayoutO`, shaped (WarpsOnM * 16, BlockN)
    const int smem_c = is_mma ? 0 : warps_on_m * 16 * config.block_n * kElementDSize;

    return std::max({smem_a + smem_b + smem_scale, smem_reduce, smem_c});
}

} // namespace deep_gemm_w4a16_common
