#pragma once

#include <algorithm>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <unordered_set>

#include <deep_gemm/common/gemm_occ_model.cuh>
#include "common_bf16.hpp"
#include "runtime.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm::mega_gate_810e {

struct MegaGateConfig {
    deep_gemm_bf16_common::ConfigResult gemm;
    int num_split_k, warp_k, aligned_experts, producers;

    int threads() const {
        const auto& [sms, m, n, k, wm, wn, stages, smem] = gemm;
        return m / wm * (n / wn) * warp_k * 32;
    }

    friend std::ostream& operator << (std::ostream& os, const MegaGateConfig& config) {
        const auto& [sms, m, n, k, wm, wn, stages, smem] = config.gemm;
        const auto& [smem_size, swizzle_mode, padding] = smem;
        return os << "MegaGateConfig(block_m=" << m << ", block_n=" << n << ", block_k=" << k
                  << ", warp_m=" << wm << ", warp_n=" << wn << ", warp_k=" << config.warp_k
                  << ", num_split_k=" << config.num_split_k << ", aligned_experts=" << config.aligned_experts
                  << ", producers=" << config.producers << ", num_launch_sms=" << sms
                  << ", threads=" << config.threads() << ", num_stages=" << stages
                  << ", smem_size=" << smem_size << ", swizzle_mode=" << swizzle_mode
                  << ", padding=" << padding << ")";
    }
};

struct MegaGateGemmChoice {
    deep_gemm_bf16_common::ConfigResult tile;
    int dense_stages;
    bool from_lut;
};

struct MegaGateTaskConfig {
    int num_split_k, full_row_tile, split_row_tile;
    bool split_full_scores;
};

struct MegaGateProblem {
    int tokens, hidden, experts, num_sms, topk;
    bool allow_split_full_scores;
    bool deterministic;

    int aligned_experts() const { return ceil_div(experts, 128) * 128; }
};

static MegaGateGemmChoice get_mega_gate_bf16_choice(const MegaGateProblem& p, int splits) {
    bool from_lut = false;
    auto tile = deep_gemm_bf16_common::get_best_configs(p.tokens, p.experts, p.hidden / splits,
        splits, p.num_sms, GemmType::Normal, 256, &from_lut);
    return {tile, std::get<6>(tile), from_lut};
}


} // namespace deep_gemm::mega_gate_810e
