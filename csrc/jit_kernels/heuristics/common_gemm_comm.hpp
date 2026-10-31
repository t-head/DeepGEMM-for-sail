#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <tuple>
#include <torch/python.h>
#include "common_fp8.hpp"
#include "common_int8.hpp"

namespace deep_gemm::gemm_comm {

using ConfigTuple = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;
using Fp8GemmRSTile = std::array<int, 6>;
using Fp8GemmRSConfig = std::pair<ConfigTuple, bool>;

inline void validate_gemm_rs_shape(int m, int n, int k, int world, bool channelwise = false) {
    TORCH_CHECK(world == 2 || world == 4, "GEMM/RS supports TP2/TP4");
    TORCH_CHECK(m > 0 && m % world == 0 && n > 0 && n % 16 == 0 && k > 0,
                "GEMM/RS requires positive shapes, M divisible by TP and N%16=0");
    TORCH_CHECK(channelwise ? k % 32 == 0 : (k > 512 && k % 128 == 0),
                "channelwise requires K%32=0; blockwise requires K>512 and K%128=0");
    // Packet offsets use signed 32-bit element indices, including padded rows.
    TORCH_CHECK(int64_t(std::max(m, 4 * world)) * n <= std::numeric_limits<int>::max(),
                "GEMM/RS packet indices exceed int32 capacity");
    const int lm = m / world;
    TORCH_CHECK(lm == 1 || lm == 2 || lm == 4 || lm == 8 || lm % 16 == 0,
                "GEMM/RS requires local M in {1,2,4,8} or a multiple of 16");
}

inline void validate_gemm_rs_tile(const Fp8GemmRSTile& tile, bool channelwise = false) {
    const auto [bm, bn, bk, wm, wn, stages] = tile;
    TORCH_CHECK(std::all_of(tile.begin(), tile.end(), [](int value) { return value > 0; }) &&
                    bm % wm == 0 && bn % wn == 0 && wm % 16 == 0 && wn % 16 == 0 &&
                    (bk == 64 || bk == 128 || bk == 256) && stages >= 2,
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

// Configuration precedence: explicit configs, fused decode profiles, then
// native runtime LUT, built-in LUT and native heuristics. Recompute SMEM for
// every selected tile; the caller-provided SMEM entry is not authoritative.
inline std::optional<Fp8GemmRSTile> get_gemm_rs_tile(const std::optional<ConfigTuple>& configs) {
    if (!configs)
        return std::nullopt;
    const auto& [sms, bm, bn, bk, wm, wn, stages, smem] = *configs;
    TORCH_CHECK(sms > 0 && sms <= get_num_sms(), "GEMM/RS configs exceed the current SM budget");
    return Fp8GemmRSTile{bm, bn, bk, wm, wn, stages};
}

inline Fp8GemmRSConfig get_blockwise_gemm_rs_config(int m, int n, int k, int world,
                                                  std::optional<ConfigTuple> configs, const std::string& preset) {
    auto tile = get_gemm_rs_tile(configs);
    const int sm_budget = configs ? std::get<0>(*configs) : get_num_sms();
    validate_gemm_rs_shape(m, n, k, world);
    TORCH_CHECK(preset == "decode" || preset == "native", "blockwise preset must be decode or native");
    if (!tile && preset == "decode" && n == 7168 && int64_t(k) * world == 24576 &&
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount == 39) {
        switch (m) {
            case 8:
            case 16: tile = Fp8GemmRSTile{16, 192, 128, 16, 16, 7}; break;
            case 32: tile = Fp8GemmRSTile{32, 192, 128, 32, 16, 7}; break;
            case 64: tile = Fp8GemmRSTile{64, 192, 128, 16, 48, 6}; break;
            case 128: tile = Fp8GemmRSTile{128, 192, 128, 32, 48, 4}; break;
            case 256: tile = Fp8GemmRSTile{256, 192, 128, 64, 48, 3}; break;
        }
    }
    ConfigTuple config;
    if (tile) {
        const auto [bm, bn, bk, wm, wn, stages] = *tile;
        validate_gemm_rs_tile(*tile);
        TORCH_CHECK(bk == 128, "blockwise tile requires BK128");
        config = ConfigTuple{sm_budget, bm, bn, bk, wm, wn, stages, std::tuple<int, int, int>{}};
    } else {
        config = deep_gemm_fp8_common::get_best_configs(m, n, k, 1, sm_budget);
    }
    auto& [num_sms, bm, bn, bk, wm, wn, stages, smem] = config;
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    const int64_t capacity = std::max(properties->sharedMemPerBlock, properties->sharedMemPerBlockOptin);
    validate_gemm_rs_tile(Fp8GemmRSTile{bm, bn, bk, wm, wn, stages});
    TORCH_CHECK(bn == 192 || 128 % bn == 0 || bn % 128 == 0,
                "blockwise BN must divide 128, be a multiple of 128, or equal 192");
    smem = deep_gemm_fp8_common::get_smem_config_scaled_n(stages, k, bm, bn, bk, bn == 192 ? 64 : 128);
    TORCH_CHECK(std::get<0>(smem) <= capacity, "blockwise GEMM scales exceed shared-memory capacity");
    return {config, false};
}

inline Fp8GemmRSConfig get_channelwise_gemm_rs_config(int m, int n, int k, int world,
                                                    std::optional<ConfigTuple> configs, const std::string& preset) {
    auto tile = get_gemm_rs_tile(configs);
    const int sm_budget = configs ? std::get<0>(*configs) : get_num_sms();
    validate_gemm_rs_shape(m, n, k, world, true);
    TORCH_CHECK(preset == "decode" || preset == "native", "channelwise preset must be decode or native");
    TORCH_CHECK(is_ppu1v5_device(), "channelwise FP8 GEMM/RS requires PPU0015");
    bool specialize = false;
    const bool decode = !tile && preset == "decode" && m >= 8 && m <= 1024 &&
                        (m & (m - 1)) == 0 && n == 7168 && int64_t(k) * world == 24576 &&
                        at::cuda::getCurrentDeviceProperties()->multiProcessorCount == 39;
    if (decode) {
        // Keep small decode tiles resident through the fused reduction tail.
        if (m <= 16) {
            tile = Fp8GemmRSTile{16, 32, 256, 16, 16, 3};
        } else if (m == 32) {
            tile = Fp8GemmRSTile{32, 32, 128, 32, 16, 3};
        } else if (m == 64) {
            tile = Fp8GemmRSTile{64, 32, 128, 32, 16, 3};
        } else if (m == 128) {
            tile = world == 2 ? Fp8GemmRSTile{128, 192, 128, 32, 32, 4}
                              : Fp8GemmRSTile{128, 96, 128, 32, 32, 3};
            specialize = world == 4;
        } else {
            tile = m <= 512 ? Fp8GemmRSTile{256, 192, 128, 64, 48, 3}
                            : Fp8GemmRSTile{256, 256, 128, 64, 64, 4};
        }
        if (world == 2 && (m == 16 || m == 64)) {
            tile = m == 16 ? Fp8GemmRSTile{16, 192, 256, 16, 32, 3} : Fp8GemmRSTile{64, 192, 128, 32, 32, 4};
            specialize = true;
        }
    }
    ConfigTuple config;
    if (!tile) {
        auto native_tile = deep_gemm_lut::get_runtime_config(
            m, n, k, 1, GemmType::Normal, deep_gemm_lut::RuntimeLutKind::FP8Channelwise);
        if (!native_tile)
            native_tile = deep_gemm_int8_lut::get_best_configs_from_lut(m, n, k, GemmType::Normal);
        if (native_tile) {
            const auto [bm, bn, bk, wm, wn, stages] = *native_tile;
            tile = Fp8GemmRSTile{bm, bn, bk, wm, wn, stages};
        } else {
            // This runtime uses the native multistage mainloop without K-split
            // or the dense-only single-stage adaptive mainloop.
            config = deep_gemm_int8::get_best_configs_dense_ppu1v5(m, n, k, 1, sm_budget);
        }
    }
    if (tile) {
        validate_gemm_rs_tile(*tile, true);
        const auto [bm, bn, bk, wm, wn, stages] = *tile;
        config = ConfigTuple{sm_budget, bm, bn, bk, wm, wn, stages,
                             deep_gemm_int8::get_smem_config(stages, k, bm, bn, bk, 1)};
    }
    const auto& [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    validate_gemm_rs_tile(Fp8GemmRSTile{bm, bn, bk, wm, wn, stages}, true);
    return {config, specialize};
}

} // namespace deep_gemm::gemm_comm
