#pragma once

#include <array>
#include <functional>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <deep_gemm/layout/mega_moe.cuh>

#include "../jit_kernels/impls/sm89_fp4_fp4_mega_moe.hpp"
#include "../jit_kernels/heuristics/mega_moe.hpp"

namespace deep_gemm::mega {

static int get_token_alignment_for_mega_moe() {
    return layout::kTokenAlignment;
}

static GatedActivation parse_gated_activation(const std::string& activation) {
    if (activation == "swiglu") return GatedActivation::SwiGLU;
    if (activation == "swiglu_step") return GatedActivation::SwiGLUStep;
    if (activation == "situ") return GatedActivation::SiTU;
    if (activation == "swiglu_oai") return GatedActivation::SwiGLUOAI;
    TORCH_CHECK(false, "Unsupported MegaMoE activation: ", activation);
}

static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int& num_shared_experts = 0,
    const std::array<int, 3>& staging_config = {}) {
    const auto mma_kind = parse_mma_kind(mma_type);
    DG_HOST_ASSERT(mma_kind == MmaKind::MXFP4);
    parse_gated_activation(activation);
    TORCH_CHECK(num_shared_experts == 0, "MegaMoE shared experts are not supported yet");
    const auto staging = get_staging_capacity(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, staging_config);
    const auto mega_buffer = layout::MegaMoEBuffer(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, staging);
    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        check_mega_moe_buffer(buffer, mega_buffer);
        auto* base = buffer.data_ptr<uint8_t>();
        auto make_tensor = [&](torch::ScalarType dtype, const layout::Buffer& region, torch::IntArrayRef shape,
                               torch::IntArrayRef strides = {}) {
            auto* ptr = region.data(base);
            const auto options = buffer.options().dtype(dtype);
            return strides.empty() ? torch::from_blob(ptr, shape, options)
                                   : torch::from_blob(ptr, shape, strides, options);
        };
        auto x = make_tensor(torch::kUInt8, mega_buffer.dispatch_fp4, {num_max_tokens_per_rank, hidden / 2});
        auto x_sf = make_tensor(torch::kUInt16, mega_buffer.staging_scales[0],
            {2, num_ranks, num_max_tokens_per_rank, hidden / 64},
            {mega_buffer.staging.scales / int64_t(sizeof(uint16_t)),
             int64_t(num_max_tokens_per_rank) * (hidden / 64), hidden / 64, 1});
        auto topk_idx = make_tensor(torch::kInt64, mega_buffer.topk_idx, {num_max_tokens_per_rank, num_topk});
        auto topk_weights = make_tensor(torch::kFloat32, mega_buffer.topk_weights, {num_max_tokens_per_rank, num_topk});
        auto l1_acts = make_tensor(torch::kUInt8, mega_buffer.local_fp4, {mega_buffer.compact * (hidden / 2)});
        auto l1_acts_sf = make_tensor(torch::kUInt16, mega_buffer.local_sfa, {mega_buffer.compact * (hidden / 64)});
        auto l2_acts = make_tensor(torch::kUInt8, mega_buffer.l2_acts, {mega_buffer.compact, intermediate_hidden / 2});
        auto l2_acts_sf = make_tensor(torch::kUInt16, mega_buffer.l2_acts_sf,
            {mega_buffer.compact, intermediate_hidden / 64}, {1, mega_buffer.compact});
        return std::make_tuple(x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
    };
    return {mega_buffer.get_num_bytes(), slice_input_buffers};
}

static void mega_moe_pre_dispatch(
    const torch::Tensor& hidden_states, const torch::Tensor& topk_idx, const torch::Tensor& topk_weights,
    const torch::Tensor& sym_buffer, const std::vector<int64_t>& sym_buffer_ptrs,
    int num_ranks, int num_experts, int num_max_tokens_per_rank, int rank_idx,
    int intermediate_hidden, const std::string& mma_type,
    const std::array<int, 3>& staging_config) {
    const auto mma_kind = parse_mma_kind(mma_type);
    DG_HOST_ASSERT(mma_kind == MmaKind::MXFP4);
    DG_HOST_ASSERT(sym_buffer.is_cuda());
    if (hidden_states.dim() != 2 || topk_idx.dim() != 2)
        throw pybind11::value_error("hidden_states and topk_idx must be 2D tensors");
    const int hidden = hidden_states.size(1), num_topk = topk_idx.size(1);
    const auto num_tokens = hidden_states.size(0);
    if (hidden_states.scalar_type() != torch::kBFloat16 || !hidden_states.is_contiguous())
        throw pybind11::value_error(c10::str("hidden_states must be contiguous BF16 with shape [tokens, ", hidden, "]"));
    if (num_tokens > num_max_tokens_per_rank)
        throw pybind11::value_error(c10::str("token count must be in [0, ", num_max_tokens_per_rank, "], got ", num_tokens));
    if (topk_idx.size(0) != num_tokens || !topk_idx.is_contiguous() ||
        (topk_idx.scalar_type() != torch::kInt32 && topk_idx.scalar_type() != torch::kInt64))
        throw pybind11::value_error(c10::str("topk_idx must be contiguous int32 or int64 with shape (", num_tokens, ", ", num_topk, ")"));
    if (topk_weights.sizes() != topk_idx.sizes() || topk_weights.scalar_type() != torch::kFloat32 ||
        !topk_weights.is_contiguous())
        throw pybind11::value_error("topk_weights must be contiguous float32 and match topk_idx");
    if (hidden_states.device() != sym_buffer.device() || topk_idx.device() != sym_buffer.device() ||
        topk_weights.device() != sym_buffer.device())
        throw pybind11::value_error("activations and routing must be on the buffer device");
    const auto staging = get_staging_capacity(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, staging_config);
    const auto mega_buffer = layout::MegaMoEBuffer(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, staging);
    DG_HOST_ASSERT(rank_idx >= 0 && rank_idx < num_ranks);
    DG_HOST_ASSERT(sym_buffer_ptrs.size() == num_ranks);
    check_mega_moe_buffer(sym_buffer, mega_buffer);
    fp4_mega_moe_pre_dispatch_impl(
        hidden_states, topk_idx, topk_weights, sym_buffer_ptrs,
        num_ranks, num_experts, num_max_tokens_per_rank, rank_idx,
        hidden, intermediate_hidden, num_topk, mega_buffer.staging);
}

// Python owns one symmetric allocation; C++ slices it and submits computation.
static void fp4_fp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe, const std::string& activation,
    const std::optional<float>& activation_clamp_opt, const bool& fast_math,
    const std::optional<int>& expected_m, const std::function<void()>& stage1_done,
    const std::optional<float>& situ_beta_opt, const std::optional<float>& situ_linear_beta_opt,
    const std::optional<float>& activation_alpha_opt,
    const std::array<int, 3>& staging_config) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    // Config checks
    DG_HOST_ASSERT(y.dim() == 2);
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    const auto activation_kind = parse_gated_activation(activation);

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);
    const auto activation_alpha = activation_alpha_opt.value_or(1.702f);
    if (activation_kind == GatedActivation::SwiGLUOAI) {
        DG_HOST_ASSERT(std::isfinite(activation_alpha) && activation_alpha > 0);
    } else {
        DG_HOST_ASSERT(!activation_alpha_opt.has_value());
    }
    const auto situ_beta = situ_beta_opt.value_or(0.0f);
    const auto situ_linear_beta = situ_linear_beta_opt.value_or(0.0f);
    if (activation_kind == GatedActivation::SiTU) {
        DG_HOST_ASSERT(!activation_clamp_opt.has_value());
        DG_HOST_ASSERT(std::isfinite(situ_beta) && situ_beta > 0);
        DG_HOST_ASSERT(!situ_linear_beta_opt.has_value() ||
                       (std::isfinite(situ_linear_beta) && situ_linear_beta > 0));
    } else {
        DG_HOST_ASSERT(!situ_beta_opt.has_value() && !situ_linear_beta_opt.has_value());
    }

    // Tensor checks
    DG_HOST_ASSERT(sym_buffer.is_cuda());
    for (const auto& tensor : {y, l1_weights, l1_weights_sf, l2_weights, l2_weights_sf})
        DG_HOST_ASSERT(tensor.device() == sym_buffer.device());
    DG_HOST_ASSERT(l1_weights.dim() == 3 && l2_weights.dim() == 3);
    DG_HOST_ASSERT(l1_weights.is_contiguous() && l2_weights.is_contiguous());
    DG_HOST_ASSERT(l1_weights.scalar_type() == torch::kUInt8 && l2_weights.scalar_type() == torch::kUInt8);
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden_packed] = get_shape<3>(l1_weights);
    const auto [num_experts_per_rank_, hidden, intermediate_hidden_packed] = get_shape<3>(l2_weights);
    DG_HOST_ASSERT(y.size(1) == hidden);
    DG_HOST_ASSERT(y.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(y.is_contiguous());
    const auto intermediate_hidden = intermediate_hidden_packed * 2;
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    // MXFP4 packs two K elements per byte; GEMM1 has two SwiGLU branches.
    DG_HOST_ASSERT(hidden == 2 * hidden_packed);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);

    // Check scale extents against the weights before any conversion or launch.
    // Two FP4 elements share one weight byte; two UE8M0 scales share one uint16.
    auto normalize_sf = [](torch::Tensor sf, const torch::Tensor& weight) {
        DG_HOST_ASSERT(sf.dim() == 3);
        DG_HOST_ASSERT(sf.scalar_type() == torch::kUInt8 || sf.scalar_type() == torch::kUInt16);
        DG_HOST_ASSERT(sf.size(0) == weight.size(0) && sf.size(1) == weight.size(1));
        const int64_t packed_k_per_scale = sf.scalar_type() == torch::kUInt8 ? 16 : 32;
        DG_HOST_ASSERT(sf.size(2) == ceil_div<int64_t>(weight.size(2), packed_k_per_scale));
        if (deep_gemm_fp4_common::check_mxfp4_scales_layout(sf))
            return sf;
        sf = deep_gemm_fp4_common::preprocess_mxfp4_scales(sf);
        DG_HOST_ASSERT(deep_gemm_fp4_common::check_mxfp4_scales_layout(sf));
        return sf;
    };
    const auto l1_weights_sf_t = normalize_sf(l1_weights_sf, l1_weights);
    const auto l2_weights_sf_t = normalize_sf(l2_weights_sf, l2_weights);

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->device() == sym_buffer.device());
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    DG_HOST_ASSERT(rank_idx >= 0 && rank_idx < num_ranks);
    DG_HOST_ASSERT(num_experts == num_experts_);
    const auto staging = get_staging_capacity(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, staging_config);
    const auto mega_buffer = layout::MegaMoEBuffer(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, staging);
    check_mega_moe_buffer(sym_buffer, mega_buffer);
    const int64_t tokens = num_tokens, groups = num_experts_per_rank;
    const int64_t owner_capacity = static_cast<int64_t>(num_ranks) * num_max_tokens_per_rank;
    int em;
    if (!expected_m.has_value()) {
        em = std::max<int64_t>(1, std::min(owner_capacity,
            tokens * num_topk / groups + 1));
    } else {
        DG_HOST_ASSERT(expected_m.value() >= 1 && expected_m.value() <= owner_capacity);
        em = expected_m.value();
    }

    const auto [arch_major, arch_minor] = device_runtime->get_arch_pair();
    if (arch_major != 8 || arch_minor != 9)
        DG_HOST_UNREACHABLE("FP4 MegaMoE requires SM89 (M890)");

    sm89_fp4_fp4_mega_moe(
        y, l1_weights, l2_weights, l1_weights_sf_t, l2_weights_sf_t,
        cumulative_local_expert_recv_stats, sym_buffer.data_ptr<uint8_t>(), mega_buffer,
        sym_buffer_ptrs, rank_idx, num_max_tokens_per_rank,
        num_experts_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        activation_clamp, em, stage1_done, activation_kind,
        situ_beta, situ_linear_beta, activation_alpha, fast_math);
}

static void register_apis(pybind11::module_& m) {
    m.def("fp4_fp4_mega_moe", &fp4_fp4_mega_moe);
    m.def("mega_moe_pre_dispatch", &mega_moe_pre_dispatch);
    m.def("get_token_alignment_for_mega_moe", &get_token_alignment_for_mega_moe);
    m.def("get_symm_buffer_size_for_mega_moe", &get_symm_buffer_size_for_mega_moe,
          pybind11::arg("num_ranks"), pybind11::arg("num_experts"),
          pybind11::arg("num_max_tokens_per_rank"), pybind11::arg("num_topk"),
          pybind11::arg("hidden"), pybind11::arg("intermediate_hidden"),
          pybind11::arg("mma_type"), pybind11::arg("activation"),
          pybind11::arg("num_shared_experts") = 0,
          pybind11::arg("staging_config") = std::array<int, 3>{});
}

}  // namespace deep_gemm::mega
