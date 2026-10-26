#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "../jit_kernels/impls/mega_moe.hpp"
#include "../jit_kernels/heuristics/mega_moe.hpp"

namespace deep_gemm::mega_moe {

// Shared API helpers.
namespace py = pybind11;
using Tensor = torch::Tensor;
using OptTensor = std::optional<Tensor>;

static void check_device() {
    TORCH_CHECK(is_ppu1v5_device(), "MegaMoE requires PPU1.5");
}

static void check_tensor(const Tensor& tensor, const torch::ScalarType& dtype, const bool& contiguous = true) {
    TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == dtype && (!contiguous || tensor.is_contiguous()),
                "Invalid MegaMoE tensor device, dtype or layout");
}
static void check_shape(const Tensor& tensor, c10::IntArrayRef shape) {
    TORCH_CHECK(tensor.sizes() == shape, "Invalid MegaMoE tensor shape: ", tensor.sizes());
}
static Tensor mega_moe_slice(const Tensor& buffer, uint64_t offset,
                             const std::vector<int64_t>& shape, torch::ScalarType dtype) {
    int64_t elements = 1;
    for (const auto size : shape) elements *= size;
    return buffer.narrow(0, offset, elements * c10::elementSize(dtype)).view(dtype).view(shape);
}

static void fp4_mega_moe_pre_dispatch(const Tensor& hidden_states, const Tensor& topk_idx,
                               const Tensor& topk_weights, const Tensor& x, const Tensor& x_sf,
                               const Tensor& staged_topk_idx, const Tensor& staged_topk_weights, const int& num_tokens) {
    check_device();
    check_tensor(hidden_states, torch::kBFloat16);
    check_tensor(topk_idx, torch::kInt64);
    check_tensor(topk_weights, torch::kFloat32);
    check_tensor(x, torch::kUInt8);
    check_tensor(x_sf, torch::kUInt8);
    check_tensor(staged_topk_idx, torch::kInt64);
    check_tensor(staged_topk_weights, torch::kFloat32);
    const int hidden = hidden_states.size(1), num_topk = topk_idx.size(1);
    TORCH_CHECK(hidden > 0 && hidden % 64 == 0 && num_tokens >= 0 && num_tokens <= hidden_states.size(0));
    TORCH_CHECK(x.size(1) == hidden / 2 && x_sf.size(1) == hidden / 32 &&
                staged_topk_idx.size(1) == num_topk && staged_topk_weights.size(1) == num_topk);
    for (const auto& tensor : {x, x_sf, staged_topk_idx, staged_topk_weights, topk_idx, topk_weights})
        TORCH_CHECK(tensor.size(0) >= num_tokens);
    fp4_mega_moe_pre_dispatch_impl(hidden_states, topk_idx, topk_weights, x,
                                   x_sf, staged_topk_idx, staged_topk_weights, num_tokens,
                                   hidden, num_topk);
}

static bool is_fp4_scale_layout(const Tensor& scale) {
    return scale.scalar_type() == torch::kUInt16 && scale.dim() == 3 &&
           scale.transpose(1, 2).is_contiguous();
}
static int get_token_alignment_for_mega_moe() {
    // PPU supports arbitrary token counts. Byte regions, rather than token
    // capacity, satisfy the 128-byte transport alignment in MegaMoEBuffer.
    return 1;
}

static layout::MegaMoEBuffer checked_layout(
    int num_ranks, int num_experts, int capacity, int num_topk,
    int hidden, int intermediate_hidden) {
    TORCH_CHECK(num_ranks > 0 && num_ranks <= 32 && num_experts > 0 && num_experts % num_ranks == 0 &&
                num_topk > 0 && num_topk <= 16 && num_topk <= num_experts &&
                capacity > 0 && capacity < (1 << 24),
                "Invalid MegaMoE ranks, expert count, top-k or token capacity");
    TORCH_CHECK(num_experts / num_ranks <= 1024,
                "MegaMoE supports at most 1024 local physical experts per rank");
    TORCH_CHECK(intermediate_hidden > 0 && intermediate_hidden <= std::numeric_limits<int>::max() / 2 &&
                intermediate_hidden % 128 == 0,
                "intermediate_hidden must be the activated width I, a positive multiple of 128");
    TORCH_CHECK(hidden > 0 && hidden % 512 == 0, "MegaMoE hidden must be divisible by 512 for W4A4");
    const auto pool = layout::get_num_max_pool_tokens<uint64_t>(num_ranks, capacity, num_topk,
                                                               num_experts / num_ranks);
    TORCH_CHECK(pool <= std::numeric_limits<int32_t>::max(),
                "MegaMoE token pool exceeds the signed 32-bit row index range");
    const layout::MegaMoEBuffer result(num_ranks, num_experts, capacity, num_topk, hidden, intermediate_hidden);
    TORCH_CHECK(result.get_num_bytes() <= uint64_t(std::numeric_limits<int64_t>::max()),
                "MegaMoE workspace exceeds the tensor byte range");
    return result;
}

using MegaMoEBufferViews = std::tuple<Tensor, Tensor, MegaMoEWorkspace>;
using MegaMoEBufferSlicer = std::function<MegaMoEBufferViews(const Tensor&)>;

// Allocation and complete calls share the same layout and typed view builder.
// The Python adapter below exposes only the four public input views.
static std::tuple<int64_t, MegaMoEBufferSlicer> get_mega_moe_buffer_slices(
    int num_ranks, int num_experts, int capacity, int num_topk,
    int hidden, int intermediate_hidden) {
    const auto offsets = checked_layout(num_ranks, num_experts, capacity, num_topk,
                                        hidden, intermediate_hidden);
    const int64_t num_bytes = offsets.get_num_bytes();
    auto slice_buffers = [=](const Tensor& buffer) {
        check_tensor(buffer, torch::kUInt8);
        TORCH_CHECK(buffer.dim() == 1 && buffer.nbytes() >= uint64_t(num_bytes),
                    "symmetric buffer is smaller than the MegaMoE workspace");
        const int64_t pool = offsets.dispatch.num_max_pool_tokens;
        const int64_t groups = num_experts / num_ranks;
        const auto rows = mega_moe_slice(buffer, offsets.gemm2_metadata, {5, groups}, torch::kInt32);
        OptTensor block_info;
        if (offsets.block_info_elements)
            block_info = mega_moe_slice(buffer, offsets.block_info,
                                        {int64_t(offsets.block_info_elements)}, torch::kInt32);
        MegaMoEWorkspace workspace{
            {}, kFp4Gemm1SfbBlockN, {},
            mega_moe_slice(buffer, offsets.l2_acts, {pool, intermediate_hidden / 2}, torch::kUInt8),
            mega_moe_slice(buffer, offsets.l2_scales, {intermediate_hidden / 64, pool}, torch::kUInt16).t(),
            mega_moe_slice(buffer, offsets.sfa_scratch, {pool * hidden / 64}, torch::kUInt16),
            rows[0], rows[1], rows[4],
            mega_moe_slice(buffer, offsets.packed_routes, {pool}, torch::kInt64), block_info,
            mega_moe_slice(buffer, offsets.combine, {int64_t(num_topk) * capacity * hidden}, torch::kBFloat16),
            buffer.narrow(0, layout::Workspace::kNumBarrierSignalBytes,
                          (2LL * num_experts + groups) * sizeof(uint64_t)),
            mega_moe_slice(buffer, offsets.topk_weights, {capacity, num_topk}, torch::kFloat32),
            mega_moe_slice(buffer, offsets.topk_idx, {capacity, num_topk}, torch::kInt64),
            mega_moe_slice(buffer, offsets.barrier_flag, {1}, torch::kInt32),
            mega_moe_slice(buffer, offsets.barrier_done, {1}, torch::kInt32)};
        return std::make_tuple(
            mega_moe_slice(buffer, offsets.input_tokens, {capacity, hidden / 2}, torch::kUInt8),
            mega_moe_slice(buffer, offsets.input_scales, {capacity, hidden / 32}, torch::kUInt8),
            std::move(workspace));
    };
    return {num_bytes, slice_buffers};
}

static std::tuple<int64_t, std::function<py::tuple(const Tensor&)>>
get_symm_buffer_size_for_mega_moe(
    const int& num_ranks, const int& num_experts, const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden, const std::string& mma_type,
    const std::string& activation, const int& num_shared_experts) {
    check_device();
    TORCH_CHECK(mma_type == "fp4xfp4", "PPU MegaMoE requires mma_type='fp4xfp4' (MXFP4)");
    TORCH_CHECK(activation == "swiglu" || activation == "situ" || activation == "step",
                "unsupported precision/activation combination for W4A4: ", activation);
    TORCH_CHECK(num_shared_experts == 0, "PPU MegaMoE does not support shared experts");
    auto [num_bytes, slice] = get_mega_moe_buffer_slices(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden);
    auto slice_inputs = [slice = std::move(slice)](const Tensor& buffer) {
        auto [x, x_sf, workspace] = slice(buffer);
        return py::make_tuple(x, x_sf, workspace.topk_idx, workspace.topk_weights);
    };
    return {num_bytes, slice_inputs};
}

void mega_moe_pre_dispatch(const Tensor& hidden_states, const Tensor& topk_idx,
                           const Tensor& topk_weights, const Tensor& x, const Tensor& x_sf,
                           const Tensor& staged_topk_idx, const Tensor& staged_topk_weights,
                           const std::optional<int>& requested_tokens) {
    check_device();
    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be [tokens, hidden]");
    TORCH_CHECK(x.dim() == 2 && x_sf.dim() == 2 && staged_topk_idx.dim() == 2 &&
                staged_topk_weights.dim() == 2, "pre-dispatch destinations must be matrices");
    const int num_tokens = requested_tokens.value_or(hidden_states.size(0));
    const int capacity = x.size(0), hidden = x.size(1) * 2, num_topk = staged_topk_idx.size(1);
    TORCH_CHECK(hidden > 0 && hidden % 512 == 0,
                "W4A4 dispatch requires hidden divisible by 512 for scale copies");
    check_shape(x_sf, {capacity, hidden / 32});
    check_shape(staged_topk_idx, {capacity, num_topk});
    check_shape(staged_topk_weights, {capacity, num_topk});
    const auto device = x.device();
    TORCH_CHECK(hidden_states.device() == device && topk_idx.device() == device && topk_weights.device() == device,
                "pre-dispatch inputs must be on the workspace device");
    TORCH_CHECK(x_sf.device() == device && staged_topk_idx.device() == device &&
                staged_topk_weights.device() == device, "pre-dispatch destinations must be on the workspace device");
    TORCH_CHECK(topk_idx.dim() == 2 && topk_weights.dim() == 2,
                "topk_ids and topk_weights must have shape [tokens, topk]");
    TORCH_CHECK(num_tokens >= 0 && num_tokens <= capacity,
                "pre-dispatch token count must be within workspace capacity");
    TORCH_CHECK(hidden_states.size(0) >= num_tokens && topk_idx.size(0) >= num_tokens &&
                topk_weights.sizes() == topk_idx.sizes(),
                "pre-dispatch inputs must contain num_tokens rows and matching routing shapes");
    TORCH_CHECK(hidden_states.size(1) == hidden && topk_idx.size(1) == num_topk,
                "pre-dispatch hidden and topk dimensions must match the workspace");
    const auto input = hidden_states.narrow(0, 0, num_tokens);
    const auto ids = topk_idx.narrow(0, 0, num_tokens), weights = topk_weights.narrow(0, 0, num_tokens);
    fp4_mega_moe_pre_dispatch(input, ids, weights, x, x_sf, staged_topk_idx, staged_topk_weights, num_tokens);
}

static std::string check_activation(const std::string& activation, const std::optional<float>& activation_clamp,
                                       const float& situ_beta, const float& situ_linear_beta) {
    TORCH_CHECK(activation == "swiglu" || activation == "step" || activation == "situ",
                "unsupported precision/activation combination for W4A4: ", activation);
    TORCH_CHECK(!activation_clamp || (*activation_clamp >= 0 && activation != "situ"),
                "activation_clamp requires SwiGLU or Step and a nonnegative limit");
    TORCH_CHECK(activation == "situ" || (situ_beta == 4.0f && situ_linear_beta == 25.0f),
                "SiTU parameters require activation='situ'");
    TORCH_CHECK(activation != "situ" || (std::isfinite(situ_beta) && situ_beta > 0 &&
                                  std::isfinite(situ_linear_beta) && situ_linear_beta > 0),
                "situ_beta and situ_linear_beta must be finite and positive");
    return activation == "swiglu" ? "silu_and_mul" : activation == "step" ? "silu_and_mul_step" : "situ_and_mul";
}

// Validate both stages before entering a cross-rank kernel. A malformed GEMM2
// weight must not leave peers waiting after GEMM1 has already advanced the protocol.
static void check_pipeline_weights(const std::tuple<torch::Tensor, torch::Tensor>& l1_weights,
                                   const std::tuple<torch::Tensor, torch::Tensor>& l2_weights,
                                   int num_experts_per_rank, int hidden, int intermediate_hidden,
                                   const torch::Device& device) {
    const auto check = [&](const std::tuple<torch::Tensor, torch::Tensor>& weights_tuple,
                           int n, int k, bool is_l1) {
        const auto& [weights, weights_sf] = weights_tuple;
        check_tensor(weights, torch::kUInt8);
        check_shape(weights, {num_experts_per_rank, n, k / 2});
        TORCH_CHECK(weights.device() == device && weights_sf.device() == device,
                    "weights must be on the workspace device");
        check_tensor(weights_sf, torch::kUInt16, is_l1);
        if (is_l1) check_shape(weights_sf, {num_experts_per_rank, mega_moe_config::fp4_gemm1_scale_width(n, k)});
        else {
            check_shape(weights_sf, {num_experts_per_rank, n, k / 64});
            TORCH_CHECK(is_fp4_scale_layout(weights_sf), "invalid transformed W4A4 GEMM2 scales");
        }
    };
    check(l1_weights, 2 * intermediate_hidden, hidden, true);
    check(l2_weights, hidden, intermediate_hidden, false);
}

void fp4_fp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    // PPU extensions: independently allocated UltraEP weights and SiTU coefficients.
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& l1_weights_replicas,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& l2_weights_replicas,
    const float& situ_beta,
    const float& situ_linear_beta
) {
    const auto& [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto& [l2_weights, l2_weights_sf] = l2_weights_tuple;
    check_device();
    check_tensor(sym_buffer, torch::kUInt8);
    check_tensor(y, torch::kBFloat16);
    TORCH_CHECK(y.dim() == 2, "output must be [tokens, hidden]");
    TORCH_CHECK(l1_weights.dim() == 3 && l2_weights.dim() == 3,
                "weights must be rank-3 tensors of local physical experts");
    const int num_tokens = y.size(0), hidden = y.size(1), num_ranks = sym_buffer_ptrs.size();
    const int intermediate_hidden = l1_weights.size(1) / 2;
    const auto [num_bytes, slice] = get_mega_moe_buffer_slices(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden);
    TORCH_CHECK(sym_buffer.dim() == 1 && sym_buffer.nbytes() >= uint64_t(num_bytes),
                "symmetric buffer is smaller than the MegaMoE workspace");
    const auto transport = make_sym_buffer(sym_buffer_ptrs, rank_idx, num_ranks);
    TORCH_CHECK(transport.base == reinterpret_cast<int64_t>(sym_buffer.data_ptr()),
                "symmetric pointer table must reference the local workspace buffer");
    TORCH_CHECK(num_tokens <= num_max_tokens_per_rank, "output token count exceeds workspace capacity");
    const auto device = sym_buffer.device();
    TORCH_CHECK(y.device() == device, "output must be on the workspace device");
    TORCH_CHECK(recipe == std::make_tuple(1, 1, 32), "PPU MXFP4 MegaMoE requires recipe=(1, 1, 32)");
    if (cumulative_local_expert_recv_stats) {
        const auto& stats = *cumulative_local_expert_recv_stats;
        check_tensor(stats, torch::kInt32);
        TORCH_CHECK(stats.device() == device && stats.numel() == num_experts / num_ranks,
                    "cumulative_local_expert_recv_stats must have one INT32 element per local physical expert "
                    "on the workspace device");
    }
    const auto activation_op = check_activation(activation, activation_clamp_opt, situ_beta, situ_linear_beta);
    TORCH_CHECK(l1_weights_replicas.has_value() == l2_weights_replicas.has_value(),
                "l1_weights_replicas and l2_weights_replicas must be provided together");
    const int num_experts_per_rank = num_experts / num_ranks;
    TORCH_CHECK(l1_weights.size(0) > 0 && l1_weights.size(0) <= num_experts_per_rank,
                "master expert count must be within the local physical expert count");
    const int num_master_experts = l1_weights.size(0);
    if (l1_weights_replicas) {
        TORCH_CHECK(num_master_experts < num_experts_per_rank,
                    "separate replica weights require at least one replica slot per rank");
        check_pipeline_weights(*l1_weights_replicas, *l2_weights_replicas,
                               num_experts_per_rank - num_master_experts, hidden, intermediate_hidden, device);
    } else {
        TORCH_CHECK(num_master_experts == num_experts_per_rank,
                    "weights must contain all local physical experts when replicas are not provided");
    }
    check_pipeline_weights(l1_weights_tuple, l2_weights_tuple, num_master_experts, hidden, intermediate_hidden, device);
    const auto& configs = mega_moe_config::get_mega_moe_configs(num_ranks, num_experts, num_topk, hidden,
        intermediate_hidden, num_max_tokens_per_rank, num_tokens, activation, device.index());
    auto [x, x_sf, workspace] = slice(sym_buffer);
    workspace.dispatch_config = configs.dispatch;
    workspace.sym_buffer = transport;
    const float clamp = activation_clamp_opt && std::isfinite(*activation_clamp_opt) ? *activation_clamp_opt : -1.0f;
    fp4_fp4_mega_moe_impl(y, l1_weights_tuple, l2_weights_tuple,
        l1_weights_replicas ? *l1_weights_replicas : l1_weights_tuple,
        l2_weights_replicas ? *l2_weights_replicas : l2_weights_tuple,
        workspace, rank_idx, num_tokens, activation_op,
        clamp, fast_math, situ_beta, situ_linear_beta, configs, cumulative_local_expert_recv_stats);
}

// Python bindings.
void register_apis(py::module_& m) {
    m.def("fp4_fp4_mega_moe", &fp4_fp4_mega_moe);
    m.def("mega_moe_pre_dispatch", &mega_moe_pre_dispatch);
    m.def("get_token_alignment_for_mega_moe", &get_token_alignment_for_mega_moe);
    m.def("get_symm_buffer_size_for_mega_moe", &get_symm_buffer_size_for_mega_moe);
}

} // namespace deep_gemm::mega_moe
