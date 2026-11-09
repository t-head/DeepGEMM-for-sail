#pragma once

#include <functional>
#include "mega_moe.hpp"
#include <limits>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "../jit_kernels/impls/sm80_int8_int8_mega_moe.hpp"

namespace deep_gemm::mega_int8 {

static Shape check_shape(int ranks, int experts, int capacity, int topk, int hidden, int intermediate,
                         const std::array<int, 3>& staging_config = {}) {
    TORCH_CHECK(ranks >= 1 && ranks <= 64 && (ranks & (ranks - 1)) == 0,
                "INT8 MegaMoE requires a power-of-two EP size in [1, 64]");
    TORCH_CHECK(experts > 0 && experts % ranks == 0, "experts must be uniformly sharded across ranks");
    TORCH_CHECK(capacity > 0 && capacity % layout::kInt8TokenAlignment == 0 && topk >= 1 && topk <= 32,
                "capacity must be positive and aligned to 16; topk must be in [1, 32]");
    TORCH_CHECK(hidden >= 2048 && hidden % 512 == 0 && intermediate > 0 && intermediate % 256 == 0,
                "INT8 MegaMoE requires H >= 2048 divisible by 512 and I divisible by 256");
    TORCH_CHECK(int64_t(ranks) * capacity * topk <= INT32_MAX &&
                int64_t(experts) * capacity <= INT32_MAX && experts / ranks <= 1024,
                "INT8 MegaMoE dimensions exceed the scheduler's indexing limits");
    const auto arch = device_runtime->get_arch_pair();
    TORCH_CHECK(arch.first == 8 && arch.second == 0,
                "INT8 MegaMoE only support sm80 now");
    const auto staging = get_mega_moe_staging(
        ranks, experts, capacity, hidden, staging_config);
    Shape shape{hidden, intermediate, experts, ranks, capacity, topk, staging};
    TORCH_CHECK(shape.layout().blocks <= layout::MegaMoESignals::kNumMaxCopyBlocks,
                "INT8 MegaMoE view exceeds the protocol's M-block limit");
    return shape;
}

static void check_tensor(const torch::Tensor& t, torch::ScalarType dtype,
        const torch::Device& device, torch::IntArrayRef shape, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.device() == device && t.scalar_type() == dtype &&
                t.is_contiguous() && t.sizes() == shape, name, " has an invalid shape, dtype, layout or device");
}

static void check_buffer(const torch::Tensor& buffer, const Shape& s) {
    TORCH_CHECK(buffer.is_cuda() && buffer.scalar_type() == torch::kUInt8 && buffer.dim() == 1 &&
                buffer.is_contiguous() && buffer.storage_offset() == 0 &&
                buffer.numel() >= s.layout().bytes && buffer.get_device() == at::cuda::current_device(),
                "invalid INT8 MegaMoE symmetric buffer or current device");
}

static auto get_symm_buffer_size_for_int8_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int& num_shared_experts = 0,
    const std::array<int, 3>& staging_config = {}) {
    TORCH_CHECK(mma_type == "int8xint8", "INT8 MegaMoE requires mma_type=int8xint8");
    TORCH_CHECK(activation == "swiglu" || activation == "swiglu_step" || activation == "situ" || activation == "swiglu_oai",
                "unsupported INT8 MegaMoE activation");
    TORCH_CHECK(num_shared_experts == 0, "MegaMoE shared experts are not supported yet");
    const auto s = check_shape(num_ranks, num_experts, num_max_tokens_per_rank,
                               num_topk, hidden, intermediate_hidden, staging_config);
    const auto b = s.layout();
    auto slice = [=](const torch::Tensor& buffer) {
        check_buffer(buffer, s);
        return std::make_tuple(
            view(buffer, b.dispatch_data, torch::kInt8, {num_max_tokens_per_rank, hidden}),
            view(buffer, {b.staging_scales[0].offset, 2 * b.staging_scales[0].bytes}, torch::kFloat32,
                 {2, 1, num_ranks, num_max_tokens_per_rank},
                 {b.staging.scales / int64_t(sizeof(float)),
                  int64_t(num_ranks) * num_max_tokens_per_rank, num_max_tokens_per_rank, 1}),
            view(buffer, b.topk_idx, torch::kInt64, {num_max_tokens_per_rank, num_topk}),
            view(buffer, b.topk_weights, torch::kFloat32, {num_max_tokens_per_rank, num_topk}),
            view(buffer, b.local_data, torch::kInt8, {b.compact, hidden}),
            view(buffer, b.local_sfa, torch::kFloat32, {b.compact}),
            view(buffer, b.stage1_data, torch::kInt8, {b.compact, intermediate_hidden}),
            view(buffer, b.stage1_scale_storage, torch::kFloat32, {b.compact}));
    };
    using Views = decltype(slice(std::declval<const torch::Tensor&>()));
    return std::make_tuple(b.bytes, std::function<Views(const torch::Tensor&)>(slice));
}

static void quant(const torch::Tensor& x, const torch::Tensor& ids, const torch::Tensor& weights,
                  const torch::Tensor& sym_buffer, const std::vector<int64_t>& buffer_ptrs,
                  int ranks, int experts, int capacity, int rank,
                  int intermediate, const std::string& mma_type,
                  const std::array<int, 3>& staging_config) {
    TORCH_CHECK(mma_type == "int8xint8", "INT8 MegaMoE requires mma_type=int8xint8");
    TORCH_CHECK(x.dim() == 2 && ids.dim() == 2, "inputs and routing must be 2D");
    const int hidden = x.size(1), topk = ids.size(1);
    const auto s = check_shape(ranks, experts, capacity, topk, hidden, intermediate, staging_config);
    check_buffer(sym_buffer, s);
    TORCH_CHECK(buffer_ptrs.size() == ranks && rank >= 0 && rank < ranks &&
                buffer_ptrs[rank] == reinterpret_cast<int64_t>(sym_buffer.data_ptr()),
                "invalid INT8 MegaMoE peer pointers");
    TORCH_CHECK(x.dim() == 2 && x.size(0) <= capacity, "input token count exceeds capacity");
    check_tensor(x, torch::kBFloat16, sym_buffer.device(), {x.size(0), hidden}, "hidden_states");
    TORCH_CHECK(ids.scalar_type() == torch::kInt32 || ids.scalar_type() == torch::kInt64,
                "topk_ids must be int32 or int64");
    check_tensor(ids, ids.scalar_type(), sym_buffer.device(), {x.size(0), topk}, "topk_ids");
    check_tensor(weights, torch::kFloat32, sym_buffer.device(), {x.size(0), topk}, "topk_weights");
    int8_mega_moe_quant_impl(s, x, ids, weights, buffer_ptrs, rank);
}

static void int8_int8_mega_moe(
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

    // INT8 uses one scale per activation row and weight output channel.
    // rk=0 denotes the full K dimension of each GEMM (H for L1, I for L2).
    const auto [rm, rn, rk] = recipe;
    TORCH_CHECK(rm == 1 && rn == 1 && rk == 0,
                "INT8 MegaMoE requires recipe=(1, 1, 0), where 0 means the full K dimension");
    TORCH_CHECK(l1_weights.dim() == 3 && l2_weights.dim() == 3, "weights must be 3D");
    const int hidden = l2_weights.size(1), intermediate_hidden = l2_weights.size(2);
    const auto s = check_shape(sym_buffer_ptrs.size(), num_experts, num_max_tokens_per_rank,
                               num_topk, hidden, intermediate_hidden, staging_config);
    check_buffer(sym_buffer, s);
    TORCH_CHECK(rank_idx >= 0 && rank_idx < s.ranks &&
                sym_buffer_ptrs[rank_idx] == reinterpret_cast<int64_t>(sym_buffer.data_ptr()),
                "invalid INT8 MegaMoE peer pointers");
    TORCH_CHECK(y.dim() == 2 && y.size(0) <= num_max_tokens_per_rank, "invalid output token count");
    const int num_tokens = static_cast<int>(y.size(0));
    check_tensor(y, torch::kBFloat16, sym_buffer.device(), {num_tokens, hidden}, "y");
    const int num_experts_per_rank = num_experts / s.ranks;
    check_tensor(l1_weights, torch::kInt8, sym_buffer.device(),
                 {num_experts_per_rank, 2 * intermediate_hidden, hidden}, "l1_weights");
    check_tensor(l1_weights_sf, torch::kFloat32, sym_buffer.device(),
                 {num_experts_per_rank, 2 * intermediate_hidden}, "l1_weights_sf");
    check_tensor(l2_weights, torch::kInt8, sym_buffer.device(),
                 {num_experts_per_rank, hidden, intermediate_hidden}, "l2_weights");
    check_tensor(l2_weights_sf, torch::kFloat32, sym_buffer.device(),
                 {num_experts_per_rank, hidden}, "l2_weights_sf");
    if (cumulative_local_expert_recv_stats.has_value())
        check_tensor(*cumulative_local_expert_recv_stats, torch::kInt32, sym_buffer.device(),
                     {num_experts_per_rank}, "cumulative_local_expert_recv_stats");
    const auto activation_clamp = activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    TORCH_CHECK(activation_clamp >= 0, "activation_clamp must be nonnegative");
    GatedActivation activation_kind;
    if (activation == "situ") {
        TORCH_CHECK(!activation_clamp_opt && situ_beta_opt &&
                    std::isfinite(*situ_beta_opt) && *situ_beta_opt > 0 && !activation_alpha_opt,
                    "SiTU requires positive finite situ_beta and no clamp/alpha");
        TORCH_CHECK(!situ_linear_beta_opt ||
                    (std::isfinite(*situ_linear_beta_opt) && *situ_linear_beta_opt > 0),
                    "situ_linear_beta must be positive and finite");
        activation_kind = GatedActivation::SiTU;
    } else {
        TORCH_CHECK(!situ_beta_opt && !situ_linear_beta_opt, "SiTU parameters require activation=situ");
        if (activation == "swiglu_oai") {
            TORCH_CHECK(!activation_alpha_opt ||
                        (std::isfinite(*activation_alpha_opt) && *activation_alpha_opt > 0),
                        "activation_alpha must be positive and finite");
            activation_kind = GatedActivation::SwiGLUOAI;
        } else {
            TORCH_CHECK(!activation_alpha_opt, "activation_alpha requires activation=swiglu_oai");
            TORCH_CHECK(activation == "swiglu" || activation == "swiglu_step", "unsupported INT8 MegaMoE activation");
            activation_kind = activation == "swiglu" ? GatedActivation::SwiGLU : GatedActivation::SwiGLUStep;
        }
    }
    const int64_t owner_capacity = int64_t(s.ranks) * num_max_tokens_per_rank;
    const int em = expected_m.value_or(std::max<int64_t>(1,
        std::min<int64_t>(owner_capacity, int64_t(num_tokens) * num_topk / num_experts_per_rank + 1)));
    TORCH_CHECK(em >= 1 && em <= owner_capacity, "expected_m is outside the owner capacity");
    int8_int8_mega_moe_impl(s, sym_buffer, sym_buffer_ptrs, rank_idx, y,
        l1_weights_tuple, l2_weights_tuple, cumulative_local_expert_recv_stats,
        em, stage1_done, activation_kind, activation_clamp,
        situ_beta_opt.value_or(0), situ_linear_beta_opt.value_or(0), activation_alpha_opt.value_or(1.702f), fast_math);
}

static void int8_mega_moe_pre_dispatch(
        const torch::Tensor& x, const torch::Tensor& topk_idx, const torch::Tensor& topk_weights,
        const torch::Tensor& buf_x, const torch::Tensor& buf_x_sf,
        const torch::Tensor& buf_topk_idx, const torch::Tensor& buf_topk_weights,
        const int& num_tokens, const int& group_size, const std::string& mma_type) {
    TORCH_CHECK(group_size == 0, "INT8 MegaMoE requires group_size=0 (the full hidden row)");
    TORCH_CHECK(x.dim() == 2 && num_tokens == x.size(0), "num_tokens must match hidden_states.shape[0]");
    const auto owner = mega::get_mega_moe_buffer_owner(buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights);
    TORCH_CHECK(owner.attr("mma_type").cast<std::string>() == "int8xint8",
                "INT8 pre-dispatch requires an Int8SymmBuffer");
    TORCH_CHECK(x.size(1) == owner.attr("hidden").cast<int>() &&
                topk_idx.dim() == 2 && topk_idx.size(1) == owner.attr("num_topk").cast<int>(),
                "input shape must match the INT8 MegaMoE buffer");
    quant(x, topk_idx, topk_weights, owner.attr("buffer").cast<torch::Tensor>(),
          owner.attr("handle").attr("buffer_ptrs").cast<std::vector<int64_t>>(),
          owner.attr("group").attr("size")().cast<int>(), owner.attr("num_experts").cast<int>(),
          owner.attr("num_max_tokens_per_rank").cast<int>(), owner.attr("group").attr("rank")().cast<int>(),
          owner.attr("intermediate_hidden").cast<int>(), mma_type,
          owner.attr("staging_config").cast<std::array<int, 3>>());
}

static void register_apis(pybind11::module_& m) {
    m.def("get_symm_buffer_size_for_int8_mega_moe", &get_symm_buffer_size_for_int8_mega_moe,
          pybind11::arg("num_ranks"), pybind11::arg("num_experts"),
          pybind11::arg("num_max_tokens_per_rank"), pybind11::arg("num_topk"),
          pybind11::arg("hidden"), pybind11::arg("intermediate_hidden"),
          pybind11::arg("mma_type"), pybind11::arg("activation"),
          pybind11::arg("num_shared_experts") = 0,
          pybind11::arg("staging_config") = std::array<int, 3>{});
    m.def("int8_mega_moe_pre_dispatch", &int8_mega_moe_pre_dispatch);
    m.def("int8_int8_mega_moe", &int8_int8_mega_moe);
}

}  // namespace deep_gemm::mega_int8
