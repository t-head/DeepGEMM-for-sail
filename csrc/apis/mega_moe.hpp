#pragma once

#include <array>
#include <limits>
#include <pybind11/stl.h>
#include <deep_gemm/layout/mega_moe.cuh>

#include "../jit_kernels/impls/sm80_int8_int8_mega_moe.hpp"
#include "../jit_kernels/impls/sm89_fp4_fp4_mega_moe.hpp"
#include "../jit_kernels/impls/sm89_fp8_fp8_mega_moe.hpp"
#include "../jit_kernels/impls/sm89_bf16_fp4_mega_moe.hpp"
#include "../jit_kernels/impls/sm89_fp8_fp4_mega_moe.hpp"
#include "../jit_kernels/heuristics/mega_moe.hpp"

namespace deep_gemm::mega {

// Shared API helpers.
namespace py = pybind11;
using Tensor = torch::Tensor;
using OptTensor = std::optional<Tensor>;

static void check_tensor(const Tensor& tensor, const torch::ScalarType& dtype, const bool& contiguous = true) {
    TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == dtype && (!contiguous || tensor.is_contiguous()),
                "Invalid MegaMoE tensor device, dtype or layout");
}
static void check_shape(const Tensor& tensor, c10::IntArrayRef shape) {
    TORCH_CHECK(tensor.sizes() == shape, "Invalid MegaMoE tensor shape: ", tensor.sizes());
}
using WeightPair = std::tuple<Tensor, Tensor>;

static void check_tensor(const Tensor& tensor, torch::ScalarType dtype,
                         const torch::Device& device, torch::IntArrayRef shape, const char* name) {
    check_tensor(tensor, dtype);
    check_shape(tensor, shape);
    TORCH_CHECK(tensor.device() == device, name, " must share the workspace device");
}

struct MegaMoEActivationParams {
    float alpha, beta, linear_beta;
};

static MegaMoEActivationParams check_mega_moe_activation(
    GatedActivation activation, const std::optional<float>& activation_clamp_opt,
    const std::optional<float>& situ_beta_opt, const std::optional<float>& situ_linear_beta_opt,
    const std::optional<float>& activation_alpha_opt) {
    TORCH_CHECK(!activation_clamp_opt || (!std::isnan(*activation_clamp_opt) && *activation_clamp_opt >= 0),
                "activation_clamp must be nonnegative or None");
    const float alpha = activation_alpha_opt.value_or(1.702f);
    if (activation == GatedActivation::SwiGLUOAI) {
        TORCH_CHECK(std::isfinite(alpha) && alpha > 0, "activation_alpha must be positive and finite");
    } else {
        TORCH_CHECK(!activation_alpha_opt, "activation_alpha applies only to swiglu_oai");
    }
    const float beta = situ_beta_opt.value_or(0.0f);
    const float linear_beta = situ_linear_beta_opt.value_or(0.0f);
    if (activation == GatedActivation::SiTU) {
        TORCH_CHECK(!activation_clamp_opt, "SiTU does not support activation_clamp");
        TORCH_CHECK(std::isfinite(beta) && beta > 0, "situ_beta must be positive and finite");
        TORCH_CHECK(!situ_linear_beta_opt || (std::isfinite(linear_beta) && linear_beta > 0),
                    "situ_linear_beta must be positive and finite or None");
    } else {
        TORCH_CHECK(!situ_beta_opt && !situ_linear_beta_opt, "SiTU parameters apply only to situ");
    }
    return {alpha, beta, linear_beta};
}

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

// The workspace owns the allocation. Its views borrow addresses within it.
static torch::Tensor mega_moe_slice(const torch::Tensor& buffer, uint64_t offset,
                                    torch::IntArrayRef shape, torch::ScalarType dtype,
                                    torch::IntArrayRef strides = {}) {
    auto* ptr = buffer.data_ptr<uint8_t>() + offset;
    const auto options = buffer.options().dtype(dtype);
    return strides.empty() ? torch::from_blob(ptr, shape, options)
                           : torch::from_blob(ptr, shape, strides, options);
}

static std::tuple<int64_t, pybind11::function> get_mega_moe_buffer_slices_for_precision(
    const int& num_ranks, const int& num_experts, const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden, const std::string& precision,
    const std::array<int, 3>& staging_config);

static std::string fp8_mega_moe_precision(const std::string& quantization) {
    TORCH_CHECK(quantization == "channelwise" || quantization == "blockwise",
                "FP8 quantization must be 'channelwise' or 'blockwise'");
    return "fp8_" + quantization;
}

template <class BufferLayout = layout::MegaMoEBuffer>
static BufferLayout get_mega_moe_layout(
    int num_ranks, int num_experts, int capacity, int topk,
    int hidden, int intermediate, const std::array<int, 3>& staging_config) {
    constexpr bool int8 = std::is_same_v<BufferLayout, layout::Int8MegaMoEBuffer>;
    TORCH_CHECK(is_supported_megamoe_rank_count(num_ranks), "MegaMoE requires a power-of-two EP size in [1, 64]");
    TORCH_CHECK(num_experts > 0 && num_experts <= layout::MegaMoESignals::kNumMaxExperts &&
                num_experts % num_ranks == 0, "experts must be uniformly sharded within the protocol capacity");
    TORCH_CHECK(capacity > 0 && capacity % layout::kTokenAlignment == 0 && topk >= 1 && topk <= 32,
                "capacity must be positive and aligned to 16; topk must be in [1, 32]");
    TORCH_CHECK(hidden >= (int8 ? 2048 : 1024) && hidden % 512 == 0 &&
                intermediate > 0 && intermediate % (int8 ? 256 : 64) == 0,
                "unsupported MegaMoE hidden or intermediate dimension");
    TORCH_CHECK(int64_t(num_experts) * (capacity / 16) <= layout::MegaMoESignals::kNumMaxCopyBlocks &&
                int64_t(num_ranks) * capacity * topk <= INT32_MAX,
                "MegaMoE dimensions exceed the protocol's indexing limits");
    if constexpr (int8) {
        TORCH_CHECK(device_runtime->get_arch_pair() == std::make_pair(8, 0), "INT8 MegaMoE requires SM80");
    }
    if (staging_config == std::array<int, 3>{} ||
        staging_config == std::array<int, 3>{num_experts, capacity, hidden})
        return {num_ranks, num_experts, capacity, topk, hidden, intermediate};
    // Aliases inherit the allocation owner's bank strides for peer writes.
    const auto [owner_experts, owner_capacity, owner_hidden] = staging_config;
    TORCH_CHECK(owner_experts > 0 && owner_experts % num_ranks == 0 &&
                owner_experts <= layout::MegaMoESignals::kNumMaxExperts &&
                owner_capacity > 0 && owner_capacity % layout::kTokenAlignment == 0 &&
                owner_hidden >= (int8 ? 2048 : 1024) && owner_hidden % 512 == 0,
                "invalid MegaMoE allocation staging configuration");
    const auto owner = layout::MegaMoEStaging::for_shape<int8>(num_ranks, owner_experts, owner_capacity, owner_hidden);
    const auto required = layout::MegaMoEStaging::for_shape<int8>(num_ranks, num_experts, capacity, hidden);
    TORCH_CHECK(required.scales <= owner.scales && required.counts <= owner.counts && required.tokens <= owner.tokens,
                "MegaMoE view exceeds allocation staging capacity");
    return {num_ranks, num_experts, capacity, topk, hidden, intermediate, owner};
}

template <class BufferLayout>
static void check_mega_moe_buffer(const torch::Tensor& buffer, const BufferLayout& layout) {
    TORCH_CHECK(buffer.is_cuda() && buffer.scalar_type() == torch::kUInt8 && buffer.dim() == 1 &&
                buffer.is_contiguous() && buffer.storage_offset() == 0 &&
                buffer.numel() >= layout.get_num_bytes() && buffer.get_device() == at::cuda::current_device(),
                "invalid MegaMoE symmetric buffer size, layout or current device");
}

template <class BufferLayout>
static std::tuple<int64_t, pybind11::function> get_mega_moe_buffer_slices(
    int num_ranks, int num_experts, int capacity, int topk,
    int hidden, int intermediate, const std::array<int, 3>& staging_config) {
    const auto b = get_mega_moe_layout<BufferLayout>(
        num_ranks, num_experts, capacity, topk, hidden, intermediate, staging_config);
    auto slice_input_buffers = [=](const Tensor& buffer) {
        check_mega_moe_buffer(buffer, b);
        auto make_tensor = [&](torch::ScalarType dtype, const layout::Buffer& region, torch::IntArrayRef shape,
                               torch::IntArrayRef strides = {}) {
            return mega_moe_slice(buffer, region.offset, shape, dtype, strides);
        };
        const auto [x, x_sf, l1, l1_sf, l2, l2_sf] = [&] {
            if constexpr (std::is_same_v<BufferLayout, layout::Int8MegaMoEBuffer>) {
                const auto owner = py::cast(buffer);
                if (!py::hasattr(owner, "_mega_moe_copy_stream")) {
                    owner.attr("_mega_moe_copy_stream") = py::capsule(
                        new mega_int8::CopyStreamResources(buffer.get_device()),
                        [](void* p) { delete static_cast<mega_int8::CopyStreamResources*>(p); });
                }
                return std::make_tuple(
                    make_tensor(torch::kInt8, b.dispatch_data, {capacity, hidden}),
                    make_tensor(torch::kFloat32, b.staging_scales[0], {2, 1, num_ranks, capacity},
                        {b.staging.scales / int64_t(sizeof(float)), int64_t(num_ranks) * capacity, capacity, 1}),
                    make_tensor(torch::kInt8, b.local_data, {b.compact, hidden}),
                    make_tensor(torch::kFloat32, b.local_sfa, {b.compact}),
                    make_tensor(torch::kInt8, b.l2_acts, {b.compact, intermediate}),
                    make_tensor(torch::kFloat32, b.l2_acts_sf, {b.compact}));
            } else {
                return std::make_tuple(
                    make_tensor(torch::kUInt8, b.dispatch_data, {capacity, hidden / 2}),
                    make_tensor(torch::kUInt16, b.staging_scales[0], {2, num_ranks, capacity, hidden / 64},
                        {b.staging.scales / int64_t(sizeof(uint16_t)), int64_t(capacity) * (hidden / 64), hidden / 64, 1}),
                    make_tensor(torch::kUInt8, b.local_data, {b.compact * (hidden / 2)}),
                    make_tensor(torch::kUInt16, b.local_sfa, {b.local_sfa.bytes / int64_t(sizeof(uint16_t))}),
                    make_tensor(torch::kUInt8, b.l2_acts, {b.compact, intermediate / 2}),
                    make_tensor(torch::kUInt16, b.l2_acts_sf, {b.compact, intermediate / 64}, {1, b.compact}));
            }
        }();
        return std::make_tuple(x, x_sf,
            make_tensor(torch::kInt64, b.topk_idx, {capacity, topk}),
            make_tensor(torch::kFloat32, b.topk_weights, {capacity, topk}), l1, l1_sf, l2, l2_sf);
    };
    return {b.get_num_bytes(), pybind11::cpp_function(slice_input_buffers)};
}

static std::tuple<int64_t, pybind11::function>
get_symm_buffer_size_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int& num_shared_experts = 0,
    const std::array<int, 3>& staging_config = {},
    const std::optional<std::string>& quantization = std::nullopt) {
    const auto activation_kind = parse_gated_activation(activation);
    TORCH_CHECK(num_shared_experts == 0, "MegaMoE shared experts are not supported yet");
    TORCH_CHECK(mma_type == "fp8xfp8" || !quantization,
                "quantization applies only to fp8xfp8 workspaces");
    if (is_fp4_mma_type(mma_type))
        return get_mega_moe_buffer_slices<layout::MegaMoEBuffer>(
            num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, staging_config);
    if (mma_type == "int8xint8")
        return get_mega_moe_buffer_slices<layout::Int8MegaMoEBuffer>(
            num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, staging_config);
    if (mma_type == "fp8xfp8" || mma_type == "fp8xfp4" || mma_type == "bf16xfp4") {
        std::string precision;
        if (mma_type == "fp8xfp8")
            precision = fp8_mega_moe_precision(quantization.value_or("channelwise"));
        else if (mma_type == "fp8xfp4")
            precision = "w4a8";
        else
            precision = "w4a16";
        return get_mega_moe_buffer_slices_for_precision(
            num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, precision, staging_config);
    }
    TORCH_CHECK(false, "Unsupported MegaMoE mma_type: ", mma_type);
}

static void routed_mega_moe_pre_dispatch(
    const torch::Tensor& hidden_states, const torch::Tensor& topk_idx, const torch::Tensor& topk_weights,
    const torch::Tensor& x, const torch::Tensor& x_sf, const torch::Tensor& staged_topk_idx,
    const torch::Tensor& staged_topk_weights, const std::optional<int>& num_tokens);

static void mega_moe_pre_dispatch(
    const torch::Tensor& x,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights,
    const torch::Tensor& buf_x,
    const torch::Tensor& buf_x_sf,
    const torch::Tensor& buf_topk_idx,
    const torch::Tensor& buf_topk_weights,
    const int& num_tokens,
    const int& group_size,
    const std::string& mma_type) {
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
    TORCH_CHECK(num_tokens >= 0 && num_tokens == x.size(0), "num_tokens must match x.shape[0]");
    // Fused peer publication needs the workspace metadata behind the output views.
    const auto owner_ref = pybind11::getattr(pybind11::cast(buf_x), "_mega_moe_buffer", pybind11::none());
    if (!pybind11::isinstance<pybind11::weakref>(owner_ref))
        throw pybind11::value_error("a live MegaMoE SymmBuffer is required");
    const auto owner = owner_ref();
    if (owner.is_none() || owner.attr("buffer").is_none())
        throw pybind11::value_error("a live MegaMoE SymmBuffer is required");
    const auto same_view = [&owner](const char* name, const torch::Tensor& view) {
        return owner.attr(name).cast<torch::Tensor>().is_same(view);
    };
    if (!same_view("x", buf_x) || !same_view("x_sf", buf_x_sf) ||
        !same_view("topk_idx", buf_topk_idx) || !same_view("topk_weights", buf_topk_weights))
        throw pybind11::value_error("output views must belong to the same live MegaMoE SymmBuffer");
    const auto owner_mma_type = owner.attr("mma_type").cast<std::string>();
    if (owner_mma_type != mma_type && !(is_fp4_mma_type(owner_mma_type) && is_fp4_mma_type(mma_type)))
        throw pybind11::value_error("mma_type must match the output views' MegaMoE SymmBuffer");
    if (!is_fp4_mma_type(mma_type) && mma_type != "int8xint8") {
        TORCH_CHECK(mma_type == "fp8xfp8" || mma_type == "fp8xfp4" || mma_type == "bf16xfp4",
                    "Unsupported MegaMoE mma_type: ", mma_type);
        routed_mega_moe_pre_dispatch(x, topk_idx, topk_weights, buf_x, buf_x_sf,
                                    buf_topk_idx, buf_topk_weights, num_tokens);
        return;
    }
    const bool int8 = mma_type == "int8xint8";
    TORCH_CHECK(group_size == (int8 ? 0 : 32), "invalid MegaMoE quantization group size");
    const auto buffer = owner.attr("buffer").cast<Tensor>();
    const auto pointers = owner.attr("handle").attr("buffer_ptrs").cast<std::vector<int64_t>>();
    const auto group = owner.attr("group");
    const int ranks = group.attr("size")().cast<int>(), rank = group.attr("rank")().cast<int>();
    const int experts = owner.attr("num_experts").cast<int>();
    const int capacity = owner.attr("num_max_tokens_per_rank").cast<int>();
    const int hidden = owner.attr("hidden").cast<int>();
    const int intermediate = owner.attr("intermediate_hidden").cast<int>();
    const int topk = owner.attr("num_topk").cast<int>();
    const auto staging_config = owner.attr("staging_config").cast<std::array<int, 3>>();
    TORCH_CHECK(num_tokens <= capacity, "input token count exceeds capacity");
    check_tensor(x, torch::kBFloat16, buffer.device(), {num_tokens, hidden}, "hidden_states");
    TORCH_CHECK(topk_idx.scalar_type() == torch::kInt32 || topk_idx.scalar_type() == torch::kInt64,
                "topk_ids must be int32 or int64");
    check_tensor(topk_idx, topk_idx.scalar_type(), buffer.device(), {num_tokens, topk}, "topk_ids");
    check_tensor(topk_weights, torch::kFloat32, buffer.device(), {num_tokens, topk}, "topk_weights");
    check_tensor(buf_x, int8 ? torch::kInt8 : torch::kUInt8,
                 buffer.device(), {capacity, int8 ? hidden : hidden / 2}, "dispatch data");
    check_tensor(buf_x_sf, int8 ? torch::kFloat32 : torch::kUInt16, false);
    TORCH_CHECK(buf_x_sf.dim() == 4, "invalid MegaMoE scale view");
    check_tensor(buf_topk_idx, torch::kInt64, buffer.device(), {capacity, topk}, "staged topk_ids");
    check_tensor(buf_topk_weights, torch::kFloat32, buffer.device(), {capacity, topk}, "staged topk_weights");
    make_sym_buffer(buffer, pointers, rank);
    if (int8) {
        const auto b = get_mega_moe_layout<layout::Int8MegaMoEBuffer>(
            ranks, experts, capacity, topk, hidden, intermediate, staging_config);
        check_mega_moe_buffer(buffer, b);
        mega_moe_pre_dispatch_impl<true>(x, topk_idx, topk_weights, pointers,
            ranks, experts, capacity, rank, hidden, intermediate, topk, b.staging);
    } else {
        const auto b = get_mega_moe_layout(ranks, experts, capacity, topk, hidden, intermediate, staging_config);
        check_mega_moe_buffer(buffer, b);
        mega_moe_pre_dispatch_impl<false>(x, topk_idx, topk_weights, pointers,
            ranks, experts, capacity, rank, hidden, intermediate, topk, b.staging);
    }
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
    const std::optional<float>& situ_beta_opt, const std::optional<float>& situ_linear_beta_opt,
    const std::optional<float>& activation_alpha_opt,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& l1_weights_replicas,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& l2_weights_replicas,
    const std::array<int, 3>& staging_config) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    // Config checks
    DG_HOST_ASSERT(y.dim() == 2);
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    const auto activation_kind = parse_gated_activation(activation);

    const auto activation_params = check_mega_moe_activation(
        activation_kind, activation_clamp_opt, situ_beta_opt, situ_linear_beta_opt, activation_alpha_opt);
    const auto activation_clamp = activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    const auto [activation_alpha, situ_beta, situ_linear_beta] = activation_params;

    // Tensor checks
    DG_HOST_ASSERT(sym_buffer.is_cuda());
    for (const auto& tensor : {y, l1_weights, l1_weights_sf, l2_weights, l2_weights_sf})
        DG_HOST_ASSERT(tensor.device() == sym_buffer.device());
    DG_HOST_ASSERT(l1_weights.dim() == 3 && l2_weights.dim() == 3);
    DG_HOST_ASSERT(l1_weights.is_contiguous() && l2_weights.is_contiguous());
    DG_HOST_ASSERT(l1_weights.scalar_type() == torch::kUInt8 && l2_weights.scalar_type() == torch::kUInt8);
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden_packed] = get_shape<3>(l1_weights);
    const auto [num_experts_per_rank_, hidden, intermediate_hidden_packed] = get_shape<3>(l2_weights);
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    DG_HOST_ASSERT(num_ranks > 0 && num_experts > 0 && num_experts % num_ranks == 0);
    const int num_experts_per_rank_with_replicas = num_experts / num_ranks;
    DG_HOST_ASSERT(y.size(1) == hidden);
    DG_HOST_ASSERT(y.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(y.is_contiguous());
    const auto intermediate_hidden = intermediate_hidden_packed * 2;
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    // MXFP4 packs two K elements per byte; GEMM1 has two SwiGLU branches.
    DG_HOST_ASSERT(hidden == 2 * hidden_packed);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    TORCH_CHECK(l1_weights_replicas.has_value() == l2_weights_replicas.has_value(),
                "l1_weights_replicas and l2_weights_replicas must be provided together");
    TORCH_CHECK(num_experts_per_rank > 0 && num_experts_per_rank <= num_experts_per_rank_with_replicas,
                "master expert count must be within the local physical expert count");

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

    torch::Tensor l1_replica_sf, l2_replica_sf;
    MegaMoEReplicaWeights l1_replicas{}, l2_replicas{};
    if (l1_weights_replicas) {
        TORCH_CHECK(num_experts_per_rank < num_experts_per_rank_with_replicas,
                    "separate replica weights require at least one replica slot per rank");
        const int replicas = num_experts_per_rank_with_replicas - num_experts_per_rank;
        auto check_replicas = [&](const auto& pair, int n, int k, torch::Tensor& normalized_sf) {
            const auto& [weight, sf] = pair;
            TORCH_CHECK(weight.dim() == 3 && weight.is_contiguous() &&
                        weight.scalar_type() == torch::kUInt8 && weight.size(0) == replicas &&
                        weight.size(1) == n && weight.size(2) == k &&
                        weight.device() == sym_buffer.device() && sf.device() == sym_buffer.device(),
                        "replica weights must match local physical expert slots, dimensions and device");
            normalized_sf = normalize_sf(sf, weight);
            return MegaMoEReplicaWeights{weight.template data_ptr<uint8_t>(),
                normalized_sf.data_ptr<uint16_t>(), static_cast<int>(num_experts_per_rank)};
        };
        l1_replicas = check_replicas(*l1_weights_replicas, intermediate_hidden_2, hidden_packed, l1_replica_sf);
        l2_replicas = check_replicas(*l2_weights_replicas, hidden, intermediate_hidden_packed, l2_replica_sf);
    } else {
        TORCH_CHECK(num_experts_per_rank == num_experts_per_rank_with_replicas,
                    "weights must contain all local physical experts when replicas are not provided");
    }

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->device() == sym_buffer.device());
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank_with_replicas);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_experts_ = num_experts_per_rank_with_replicas * num_ranks;
    make_sym_buffer(sym_buffer, sym_buffer_ptrs, rank_idx);
    DG_HOST_ASSERT(num_experts == num_experts_);
    const auto mega_buffer = get_mega_moe_layout(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, staging_config);
    check_mega_moe_buffer(sym_buffer, mega_buffer);
    const int em = expected_expert_rows(num_tokens, num_topk, num_experts_per_rank_with_replicas,
                                        num_ranks, num_max_tokens_per_rank);

    const auto [arch_major, arch_minor] = device_runtime->get_arch_pair();
    if (arch_major != 8 || arch_minor != 9)
        DG_HOST_UNREACHABLE("FP4 MegaMoE requires SM89 (M890)");

    sm89_fp4_fp4_mega_moe(
        y, l1_weights, l2_weights, l1_weights_sf_t, l2_weights_sf_t,
        l1_replicas, l2_replicas,
        cumulative_local_expert_recv_stats, sym_buffer.data_ptr<uint8_t>(), mega_buffer,
        sym_buffer_ptrs, rank_idx, num_max_tokens_per_rank,
        num_experts_per_rank_with_replicas, num_tokens, num_topk, hidden, intermediate_hidden,
        activation_clamp, em, activation_kind,
        situ_beta, situ_linear_beta, activation_alpha, fast_math);
}

static void check_device() {
    TORCH_CHECK(is_ppu1v5_device(), "MegaMoE requires PPU1.5");
}

static Precision routed_precision(const std::string& precision) {
    using P = Precision;
    if (precision == "w4a16")
        return P::W4A16;
    if (precision == "w4a8")
        return P::W4A8;
    if (precision == "fp8_channelwise")
        return P::FP8Channelwise;
    TORCH_CHECK(precision == "fp8_blockwise", "Unsupported MegaMoE precision: ", precision);
    return P::FP8Blockwise;
}

static layout::RoutedMoEBuffer checked_routed_layout(
    int num_ranks, int num_experts, int num_max_tokens_per_rank, int num_topk, int hidden,
    int intermediate_hidden, const std::string& precision, const std::array<int, 3>& staging_config) {
    using P = Precision;
    const auto p = routed_precision(precision);
    TORCH_CHECK(num_ranks > 0 && num_ranks <= layout::kNumMaxRanks &&
                    num_experts > 0 && num_experts % num_ranks == 0 &&
                    num_topk > 0 && num_topk <= 16 && num_topk <= num_experts && num_max_tokens_per_rank > 0 && num_max_tokens_per_rank < (1 << 24),
                "Invalid MegaMoE ranks, experts, top-k or token capacity for ", precision);
    TORCH_CHECK(num_experts / num_ranks <= (p == P::W4A16 ? 127 : 128), "Unsupported local expert count for ", precision);
    TORCH_CHECK(hidden > 0 && hidden % 256 == 0 && intermediate_hidden > 0 && intermediate_hidden % 128 == 0 &&
                    intermediate_hidden <= std::numeric_limits<int>::max() / 2,
                "MegaMoE requires hidden divisible by 256 and intermediate_hidden=I divisible by 128");
    const auto pool = layout::get_num_max_pool_tokens<uint64_t>(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts / num_ranks);
    TORCH_CHECK(pool <= std::numeric_limits<int32_t>::max(), "MegaMoE pool exceeds signed 32-bit rows");
    auto staging = layout::RoutedMoEStaging::for_shape(num_ranks, num_experts, num_max_tokens_per_rank);
    if (staging_config != std::array<int, 3>{}) {
        const int owner_experts = staging_config[0], owner_capacity = staging_config[1];
        TORCH_CHECK(owner_experts > 0 && owner_experts % num_ranks == 0 &&
                        owner_experts / num_ranks <= (p == P::W4A16 ? 127 : 128) &&
                        owner_capacity > 0 && owner_capacity < (1 << 24),
                    "Invalid MegaMoE allocation staging configuration");
        const auto owner = layout::RoutedMoEStaging::for_shape(num_ranks, owner_experts, owner_capacity);
        TORCH_CHECK(staging.counts <= owner.counts && staging.tokens <= owner.tokens,
                    "MegaMoE view exceeds allocation staging capacity");
        staging = owner;
    }
    return {uint32_t(num_ranks), uint32_t(num_experts), uint32_t(num_max_tokens_per_rank), uint32_t(num_topk),
            uint32_t(hidden), uint32_t(intermediate_hidden), p != P::W4A16, p == P::FP8Blockwise, staging};
}

static void check_routed_buffer(const Tensor& buffer, const layout::RoutedMoEBuffer& layout) {
    check_tensor(buffer, torch::kUInt8);
    TORCH_CHECK(buffer.dim() == 1 && buffer.nbytes() >= layout.get_num_bytes(), "MegaMoE buffer is too small");
}

static std::tuple<int64_t, pybind11::function> get_mega_moe_buffer_slices_for_precision(
    const int& num_ranks, const int& num_experts, const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden, const std::string& precision,
    const std::array<int, 3>& staging_config) {
    check_device();
    const auto layout = checked_routed_layout(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, precision, staging_config);
    auto slice_input_buffers =
        [=](const Tensor& buffer, const std::vector<int64_t>& pointers, const int& rank_idx) {
            check_routed_buffer(buffer, layout);
            const auto transport = make_sym_buffer(pointers, rank_idx, num_ranks);
            TORCH_CHECK(transport.base == reinterpret_cast<int64_t>(buffer.data_ptr()),
                        "Invalid local symmetric pointer");
            const bool fp8 = precision != "w4a16", blockwise = precision == "fp8_blockwise";
            auto x = mega_moe_slice(buffer, layout.input_tokens, {num_max_tokens_per_rank, hidden},
                                    fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16);
            auto sf = mega_moe_slice(buffer, layout.input_scales,
                {num_max_tokens_per_rank, fp8 ? (blockwise ? hidden / 128 : 1) : 0}, torch::kFloat32);
            auto ids = mega_moe_slice(buffer, layout.topk_idx, {num_max_tokens_per_rank, num_topk}, torch::kInt64);
            auto weights = mega_moe_slice(buffer, layout.topk_weights, {num_max_tokens_per_rank, num_topk}, torch::kFloat32);
            std::vector<int64_t> offsets(num_ranks);
            for (int r = 0; r < num_ranks; ++r)
                offsets[r] = pointers[r] - pointers[rank_idx];
            auto peer_offsets = mega_moe_slice(buffer, layout.peer_offsets, {num_ranks}, torch::kInt64);
            peer_offsets.copy_(torch::tensor(offsets, torch::TensorOptions().dtype(torch::kInt64)));
            return py::make_tuple(x, sf, ids, weights);
        };
    return {int64_t(layout.get_num_bytes()), pybind11::cpp_function(slice_input_buffers)};
}

static void fp8_w4a16_mega_moe_pre_dispatch_checked(const Tensor& hidden_states, const Tensor& topk_idx,
                                         const Tensor& topk_weights, const Tensor& x, const Tensor& x_sf,
                                         const Tensor& staged_topk_idx, const Tensor& staged_topk_weights,
                                         const std::optional<int>& requested_num_tokens) {
    check_device();
    check_tensor(hidden_states, torch::kBFloat16);
    TORCH_CHECK(topk_idx.scalar_type() == torch::kInt32 || topk_idx.scalar_type() == torch::kInt64,
                "topk_idx must be contiguous int32 or int64");
    check_tensor(topk_idx, topk_idx.scalar_type());
    check_tensor(topk_weights, torch::kFloat32);
    TORCH_CHECK(hidden_states.dim() == 2 && x.dim() == 2 && topk_idx.dim() == 2 && x_sf.dim() == 2 &&
                    staged_topk_idx.dim() == 2 && staged_topk_weights.dim() == 2,
                "pre-dispatch inputs and views must be matrices");
    const int num_tokens = requested_num_tokens.value_or(hidden_states.size(0));
    const int hidden = x.size(1), num_topk = staged_topk_idx.size(1);
    const int num_max_tokens_per_rank = x.size(0);
    TORCH_CHECK(num_tokens >= 0 && num_tokens <= num_max_tokens_per_rank,
                "pre-dispatch token count must be within workspace capacity");
    TORCH_CHECK(hidden > 0 && hidden % 256 == 0, "pre-dispatch hidden must be divisible by 256");
    check_shape(hidden_states, {hidden_states.size(0), hidden});
    TORCH_CHECK(hidden_states.size(0) >= num_tokens && topk_idx.size(0) >= num_tokens,
                "pre-dispatch inputs must contain num_tokens rows");
    check_shape(topk_idx, {topk_idx.size(0), num_topk});
    check_shape(topk_weights, topk_idx.sizes());
    check_shape(staged_topk_idx, {num_max_tokens_per_rank, num_topk});
    check_shape(staged_topk_weights, staged_topk_idx.sizes());
    check_tensor(staged_topk_idx, torch::kInt64);
    check_tensor(staged_topk_weights, torch::kFloat32);
    check_tensor(x_sf, torch::kFloat32);
    const auto device = x.device();
    for (const auto& t : {hidden_states, topk_idx, topk_weights, x_sf, staged_topk_idx, staged_topk_weights})
        TORCH_CHECK(t.device() == device, "pre-dispatch tensors must share the workspace device");
    if (x.scalar_type() == torch::kBFloat16) {
        check_tensor(x, torch::kBFloat16);
        check_shape(x_sf, {num_max_tokens_per_rank, 0});
        w4a16_mega_moe_pre_dispatch_impl(
            hidden_states, topk_idx, topk_weights, x, staged_topk_idx,
            staged_topk_weights, num_tokens, hidden, num_topk);
    } else {
        check_tensor(x, torch::kFloat8_e4m3fn);
        const bool blockwise = x_sf.size(1) != 1;
        check_shape(x_sf, {num_max_tokens_per_rank, blockwise ? hidden / 128 : 1});
        if (blockwise)
            fp8_blockwise_mega_moe_pre_dispatch_impl(
                hidden, num_topk, hidden_states, topk_idx, topk_weights, x, x_sf, staged_topk_idx, staged_topk_weights, num_tokens);
        else
            fp8_channelwise_mega_moe_pre_dispatch_impl(
                hidden, num_topk, hidden_states, topk_idx, topk_weights, x,
                x_sf, staged_topk_idx, staged_topk_weights, num_tokens);
    }
}

static void routed_mega_moe_pre_dispatch(
    const Tensor& hidden_states, const Tensor& topk_idx, const Tensor& topk_weights,
    const Tensor& x, const Tensor& x_sf, const Tensor& staged_topk_idx, const Tensor& staged_topk_weights,
    const std::optional<int>& num_tokens) {
    fp8_w4a16_mega_moe_pre_dispatch_checked(hidden_states, topk_idx, topk_weights, x, x_sf,
                                           staged_topk_idx, staged_topk_weights, num_tokens);
}

template <bool kFp8>
static void checked_mega_moe(const Tensor& y, const WeightPair& l1_weights, const WeightPair& l2_weights,
                            const OptTensor& cumulative_local_expert_recv_stats, const Tensor& sym_buffer,
                            const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx, const int& num_max_tokens_per_rank,
                            const int& num_experts, const int& num_topk, const std::string& precision,
                            const std::string& activation, const std::optional<float>& activation_clamp_opt,
                            const bool& fast_math, const std::optional<float>& situ_beta_opt,
                            const std::optional<float>& situ_linear_beta_opt,
                            const std::optional<float>& activation_alpha_opt,
                            const std::optional<WeightPair>& l1_weights_replicas,
                            const std::optional<WeightPair>& l2_weights_replicas,
                            const std::array<int, 3>& staging_config) {
    check_device();
    const auto kind = routed_precision(precision);
    const bool fp8 = precision != "w4a16", blockwise = precision == "fp8_blockwise";
    TORCH_CHECK(fp8 == kFp8, "Compute entry does not match workspace precision");
    const auto activation_kind = parse_gated_activation(activation);
    const auto activation_params = check_mega_moe_activation(
        activation_kind, activation_clamp_opt, situ_beta_opt, situ_linear_beta_opt, activation_alpha_opt);
    check_tensor(y, torch::kBFloat16);
    TORCH_CHECK(y.dim() == 2, "output must be [num_tokens, hidden]");
    const auto& [w1, s1] = l1_weights;
    const auto& [w2, s2] = l2_weights;
    TORCH_CHECK(w1.dim() == 3 && w2.dim() == 3, "weights must be rank-3 tensors");
    const int hidden = y.size(1), intermediate_hidden = w1.size(1) / 2;
    const int num_tokens = y.size(0), num_ranks = sym_buffer_ptrs.size();
    const auto l = checked_routed_layout(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, precision, staging_config);
    if (cumulative_local_expert_recv_stats) {
        check_tensor(*cumulative_local_expert_recv_stats, torch::kInt32);
        TORCH_CHECK(cumulative_local_expert_recv_stats->device() == y.device() &&
                        cumulative_local_expert_recv_stats->numel() == num_experts / num_ranks,
                    "Receive statistics must contain one INT32 per local physical expert on the output device");
    }
    TORCH_CHECK(num_tokens <= num_max_tokens_per_rank, "output token count exceeds workspace capacity");
    TORCH_CHECK(y.device() == sym_buffer.device(), "output and workspace must share a device");
    const auto transport = make_sym_buffer(sym_buffer_ptrs, rank_idx, num_ranks);
    TORCH_CHECK(transport.base == reinterpret_cast<int64_t>(sym_buffer.data_ptr()),
                "Invalid local workspace pointer");
    TORCH_CHECK(l1_weights_replicas.has_value() == l2_weights_replicas.has_value(),
                "Both replica banks must be provided together");
    const auto check = [&](const WeightPair& w, int n, int k, int groups) {
        const auto& [values, scales] = w;
        check_tensor(values, fp8 ? torch::kFloat8_e4m3fn : torch::kUInt8);
        check_tensor(scales, fp8 ? torch::kFloat32 : torch::kUInt8);
        TORCH_CHECK(values.device() == y.device() && scales.device() == y.device(),
                    "Weights must share the workspace device");
        check_shape(values, {groups, n, fp8 ? k : k / 2});
        if (fp8)
            check_shape(scales, {groups, blockwise ? n / 128 : n, blockwise ? k / 128 : 1});
        else
            check_shape(scales, {groups, n / 64, int64_t(k) * 2});
    };
    const int masters = w1.size(0), groups = num_experts / num_ranks;
    TORCH_CHECK(masters > 0 && masters <= groups, "Invalid master expert count");
    check(l1_weights, 2 * intermediate_hidden, hidden, masters);
    check(l2_weights, hidden, intermediate_hidden, masters);
    if (l1_weights_replicas) {
        TORCH_CHECK(masters < groups, "Replica banks need replica slots");
        check(*l1_weights_replicas, 2 * intermediate_hidden, hidden, groups - masters);
        check(*l2_weights_replicas, hidden, intermediate_hidden, groups - masters);
    } else
        TORCH_CHECK(masters == groups, "Weights must contain all local physical experts");
    check_routed_buffer(sym_buffer, l);
    const auto& configs = get_mega_moe_precision_configs(
        kind, num_ranks, num_experts, num_topk, hidden, intermediate_hidden, num_max_tokens_per_rank,
        num_tokens, masters, activation, y.device().index());
    const auto& l1_replicas = l1_weights_replicas ? *l1_weights_replicas : l1_weights;
    const auto& l2_replicas = l2_weights_replicas ? *l2_weights_replicas : l2_weights;
    if constexpr (kFp8) {
        fp8_fp8_mega_moe_impl(
            y, l1_weights, l2_weights, l1_replicas, l2_replicas, sym_buffer.data_ptr<uint8_t>(), l, transport, rank_idx, num_tokens,
            activation_kind, activation_clamp_opt && std::isfinite(*activation_clamp_opt) ? *activation_clamp_opt : -1.0f,
            fast_math, activation_params.alpha, activation_params.beta, activation_params.linear_beta,
            configs, kind, cumulative_local_expert_recv_stats);
    } else {
        bf16_fp4_mega_moe_impl(
            y, l1_weights, l2_weights, l1_replicas, l2_replicas, sym_buffer.data_ptr<uint8_t>(), l, transport, rank_idx, num_tokens,
            activation_kind, activation_clamp_opt && std::isfinite(*activation_clamp_opt) ? *activation_clamp_opt : -1.0f,
            fast_math, activation_params.alpha, activation_params.beta, activation_params.linear_beta,
            configs, cumulative_local_expert_recv_stats);
    }
}

void fp8_fp8_mega_moe(
    const Tensor& y, const WeightPair& l1_weights_tuple, const WeightPair& l2_weights_tuple,
    const OptTensor& cumulative_local_expert_recv_stats, const Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::string& activation, const std::optional<float>& activation_clamp_opt, const bool& fast_math,
    const std::optional<float>& activation_alpha_opt,
    const std::optional<WeightPair>& l1_weights_replicas, const std::optional<WeightPair>& l2_weights_replicas,
    const std::string& quantization, const std::optional<float>& situ_beta_opt,
    const std::optional<float>& situ_linear_beta_opt, const std::array<int, 3>& staging_config) {
    checked_mega_moe<true>(
        y, l1_weights_tuple, l2_weights_tuple, cumulative_local_expert_recv_stats, sym_buffer,
        sym_buffer_ptrs, rank_idx, num_max_tokens_per_rank, num_experts, num_topk,
        fp8_mega_moe_precision(quantization), activation, activation_clamp_opt, fast_math,
        situ_beta_opt, situ_linear_beta_opt, activation_alpha_opt, l1_weights_replicas, l2_weights_replicas, staging_config);
}

void bf16_fp4_mega_moe(
    const Tensor& y, const WeightPair& l1_weights_tuple, const WeightPair& l2_weights_tuple,
    const OptTensor& cumulative_local_expert_recv_stats, const Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::string& activation, const std::optional<float>& activation_clamp_opt, const bool& fast_math,
    const std::optional<float>& situ_beta_opt, const std::optional<float>& situ_linear_beta_opt,
    const std::optional<WeightPair>& l1_weights_replicas, const std::optional<WeightPair>& l2_weights_replicas,
    const std::optional<float>& activation_alpha_opt, const std::array<int, 3>& staging_config) {
    checked_mega_moe<false>(
        y, l1_weights_tuple, l2_weights_tuple, cumulative_local_expert_recv_stats, sym_buffer,
        sym_buffer_ptrs, rank_idx, num_max_tokens_per_rank, num_experts, num_topk, "w4a16",
        activation, activation_clamp_opt, fast_math, situ_beta_opt, situ_linear_beta_opt, activation_alpha_opt,
        l1_weights_replicas, l2_weights_replicas, staging_config);
}


void fp8_fp4_mega_moe(
    const Tensor& y, const W4A8Weights& l1_weights_tuple, const W4A8Weights& l2_weights_tuple,
    const OptTensor& cumulative_local_expert_recv_stats, const Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::string& activation, const std::optional<float>& activation_clamp_opt, const bool& fast_math,
    const std::optional<float>& situ_beta_opt, const std::optional<float>& situ_linear_beta_opt,
    const std::optional<float>& activation_alpha_opt,
    const std::optional<W4A8Weights>& l1_weights_replicas, const std::optional<W4A8Weights>& l2_weights_replicas,
    const std::array<int, 3>& staging_config) {
    check_device();
    const auto activation_kind = parse_gated_activation(activation);
    const auto activation_params = check_mega_moe_activation(
        activation_kind, activation_clamp_opt, situ_beta_opt, situ_linear_beta_opt, activation_alpha_opt);
    check_tensor(y, torch::kBFloat16);
    TORCH_CHECK(y.dim() == 2 && std::get<0>(l1_weights_tuple).dim() == 3, "Invalid output or GEMM1 weight rank");
    const int hidden = y.size(1), intermediate = std::get<0>(l1_weights_tuple).size(1) / 2;
    const int tokens = y.size(0), ranks = sym_buffer_ptrs.size();
    const auto l = checked_routed_layout(ranks, num_experts, num_max_tokens_per_rank, num_topk,
                                         hidden, intermediate, "w4a8", staging_config);
    TORCH_CHECK(tokens <= num_max_tokens_per_rank, "output token count exceeds workspace capacity");
    TORCH_CHECK(y.device() == sym_buffer.device(), "output and workspace must share a device");
    const auto transport = make_sym_buffer(sym_buffer_ptrs, rank_idx, ranks);
    TORCH_CHECK(transport.base == reinterpret_cast<int64_t>(sym_buffer.data_ptr()), "Invalid local workspace pointer");
    TORCH_CHECK(l1_weights_replicas.has_value() == l2_weights_replicas.has_value(),
                "Both replica banks must be provided together");
    const int groups = num_experts / ranks, masters = std::get<0>(l1_weights_tuple).size(0);
    TORCH_CHECK(masters > 0 && masters <= groups, "Invalid master expert count");
    const auto check = [&](const W4A8Weights& weights, int n, int k, int experts) {
        const auto& [w, s, inv] = weights;
        check_tensor(w, torch::kUInt8); check_tensor(s, torch::kUInt8);
        TORCH_CHECK(w.device() == y.device() && s.device() == y.device(), "Weights must share the workspace device");
        check_shape(w, {experts, n, k / 2});
        check_shape(s, {experts, n / 64, int64_t(k) * 2});
        TORCH_CHECK(std::isfinite(inv) && inv > 0, "weight_inv_scale must be finite and positive");
    };
    check(l1_weights_tuple, 2 * intermediate, hidden, masters);
    check(l2_weights_tuple, hidden, intermediate, masters);
    if (l1_weights_replicas) {
        TORCH_CHECK(masters < groups, "Replica banks need replica slots");
        check(*l1_weights_replicas, 2 * intermediate, hidden, groups - masters);
        check(*l2_weights_replicas, hidden, intermediate, groups - masters);
    } else
        TORCH_CHECK(masters == groups, "Weights must contain all local physical experts");
    if (cumulative_local_expert_recv_stats) {
        check_tensor(*cumulative_local_expert_recv_stats, torch::kInt32);
        TORCH_CHECK(cumulative_local_expert_recv_stats->device() == y.device() &&
                    cumulative_local_expert_recv_stats->numel() == groups,
                    "Receive statistics must contain one INT32 per local physical expert on the output device");
    }
    check_routed_buffer(sym_buffer, l);
    const auto& configs = get_mega_moe_precision_configs(
        Precision::W4A8, ranks, num_experts, num_topk, hidden, intermediate,
        num_max_tokens_per_rank, tokens, masters, activation, y.device().index());
    fp8_fp4_mega_moe_impl(y, l1_weights_tuple, l2_weights_tuple,
        l1_weights_replicas ? *l1_weights_replicas : l1_weights_tuple,
        l2_weights_replicas ? *l2_weights_replicas : l2_weights_tuple,
        sym_buffer.data_ptr<uint8_t>(), l, transport, rank_idx, tokens, activation_kind,
        activation_clamp_opt && std::isfinite(*activation_clamp_opt) ? *activation_clamp_opt : -1.0f,
        fast_math, activation_params.alpha, activation_params.beta, activation_params.linear_beta,
        configs, cumulative_local_expert_recv_stats);
}

static void int8_int8_mega_moe(
    const Tensor& y, const WeightPair& l1_weights_tuple, const WeightPair& l2_weights_tuple,
    const OptTensor& cumulative_local_expert_recv_stats,
    const Tensor& sym_buffer, const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk, const std::tuple<int, int, int>& recipe,
    const std::string& activation, const std::optional<float>& activation_clamp_opt,
    const bool& fast_math, const std::optional<float>& situ_beta_opt,
    const std::optional<float>& situ_linear_beta_opt, const std::optional<float>& activation_alpha_opt,
    const std::array<int, 3>& staging_config) {
    const auto& [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto& [l2_weights, l2_weights_sf] = l2_weights_tuple;

    // INT8 uses one scale per activation row and weight output channel.
    // rk=0 denotes the full K dimension of each GEMM (H for L1, I for L2).
    const auto [rm, rn, rk] = recipe;
    TORCH_CHECK(rm == 1 && rn == 1 && rk == 0,
                "INT8 MegaMoE requires recipe=(1, 1, 0), where 0 means the full K dimension");
    TORCH_CHECK(l1_weights.dim() == 3 && l2_weights.dim() == 3, "weights must be 3D");
    const int hidden = l2_weights.size(1), intermediate = l2_weights.size(2);
    const int num_ranks = sym_buffer_ptrs.size();
    const auto b = get_mega_moe_layout<layout::Int8MegaMoEBuffer>(num_ranks, num_experts, num_max_tokens_per_rank,
                                       num_topk, hidden, intermediate, staging_config);
    check_mega_moe_buffer(sym_buffer, b);
    make_sym_buffer(sym_buffer, sym_buffer_ptrs, rank_idx);
    TORCH_CHECK(y.dim() == 2 && y.size(0) <= num_max_tokens_per_rank, "invalid output token count");
    const int num_tokens = y.size(0), groups = num_experts / num_ranks;
    const auto device = sym_buffer.device();
    check_tensor(y, torch::kBFloat16, device, {num_tokens, hidden}, "y");
    check_tensor(l1_weights, torch::kInt8, device, {groups, 2 * intermediate, hidden}, "l1_weights");
    check_tensor(l1_weights_sf, torch::kFloat32, device, {groups, 2 * intermediate}, "l1_weights_sf");
    check_tensor(l2_weights, torch::kInt8, device, {groups, hidden, intermediate}, "l2_weights");
    check_tensor(l2_weights_sf, torch::kFloat32, device, {groups, hidden}, "l2_weights_sf");
    if (cumulative_local_expert_recv_stats)
        check_tensor(*cumulative_local_expert_recv_stats, torch::kInt32, device,
                          {groups}, "cumulative_local_expert_recv_stats");

    const auto activation_kind = parse_gated_activation(activation);
    const auto activation_params = check_mega_moe_activation(
        activation_kind, activation_clamp_opt, situ_beta_opt, situ_linear_beta_opt, activation_alpha_opt);
    const auto activation_clamp = activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    const auto expected_m = expected_expert_rows(num_tokens, num_topk, groups, num_ranks, num_max_tokens_per_rank);
    auto& streams = *py::cast(sym_buffer).attr("_mega_moe_copy_stream").cast<py::capsule>()
                        .get_pointer<mega_int8::CopyStreamResources>();
    mega_int8::int8_int8_mega_moe_impl(sym_buffer.data_ptr<uint8_t>(), b, num_max_tokens_per_rank, num_topk, streams,
        sym_buffer_ptrs, rank_idx, y, l1_weights_tuple, l2_weights_tuple, cumulative_local_expert_recv_stats,
        expected_m, activation_kind, activation_clamp,
        activation_params.beta, activation_params.linear_beta, activation_params.alpha, fast_math);
}

static void register_apis(pybind11::module_& m) {
    m.def("fp8_fp4_mega_moe", &fp8_fp4_mega_moe);
    m.def("bf16_fp4_mega_moe", &bf16_fp4_mega_moe);
    m.def("fp8_fp8_mega_moe", &fp8_fp8_mega_moe);
    m.def("fp4_fp4_mega_moe", &fp4_fp4_mega_moe);
    m.def("int8_int8_mega_moe", &int8_int8_mega_moe);
    m.def("mega_moe_pre_dispatch", &mega_moe_pre_dispatch);
    m.def("get_token_alignment_for_mega_moe", &get_token_alignment_for_mega_moe);
    m.def("get_symm_buffer_size_for_mega_moe", &get_symm_buffer_size_for_mega_moe,
          pybind11::arg("num_ranks"), pybind11::arg("num_experts"),
          pybind11::arg("num_max_tokens_per_rank"), pybind11::arg("num_topk"),
          pybind11::arg("hidden"), pybind11::arg("intermediate_hidden"),
          pybind11::arg("mma_type"), pybind11::arg("activation"),
          pybind11::arg("num_shared_experts") = 0,
          pybind11::arg("staging_config") = std::array<int, 3>{},
          pybind11::arg("quantization") = pybind11::none());
}

}  // namespace deep_gemm::mega
