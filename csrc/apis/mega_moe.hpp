#pragma once

#include <array>
#include <functional>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <deep_gemm/layout/mega_moe.cuh>

#include "../jit_kernels/impls/sm89_fp4_fp4_mega_moe.hpp"
#include "../jit_kernels/impls/mega_moe_fp8_w4a16.hpp"
#include "../jit_kernels/impls/mega_moe_w4a8.hpp"
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

using SymmBufferSlices = std::function<pybind11::tuple(
    const torch::Tensor&, const std::vector<int64_t>&, const int&)>;

static std::tuple<int64_t, SymmBufferSlices> get_mega_moe_buffer_slices_for_precision(
    const int& num_ranks, const int& num_experts, const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden, const std::string& precision);

static std::string fp8_mega_moe_precision(const std::string& quantization) {
    TORCH_CHECK(quantization == "channelwise" || quantization == "blockwise",
                "FP8 quantization must be 'channelwise' or 'blockwise'");
    return "fp8_" + quantization;
}

static std::tuple<int64_t, SymmBufferSlices>
get_symm_buffer_size_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int& num_shared_experts = 0,
    const std::array<int, 3>& staging_config = {},
    const std::optional<std::string>& quantization = std::nullopt) {
    TORCH_CHECK(num_shared_experts == 0, "MegaMoE shared experts are not supported yet");
    TORCH_CHECK(mma_type == "fp8xfp8" || !quantization,
                "quantization applies only to fp8xfp8 workspaces");
    if (mma_type != "mxf4xmxf4") {
        std::string precision;
        if (mma_type == "fp8xfp8")
            precision = fp8_mega_moe_precision(quantization.value_or("channelwise"));
        else if (mma_type == "fp8xmxfp4")
            precision = "w4a8";
        else {
            TORCH_CHECK(mma_type == "bf16xmxfp4", "Unsupported MegaMoE mma_type: ", mma_type);
            precision = "w4a16";
        }
        const bool supported_activation = precision == "w4a16" ? activation == "situ" :
            activation == "swiglu" || activation == "swiglu_oai" ||
            (precision == "w4a8" && activation == "situ");
        TORCH_CHECK(supported_activation, "unsupported activation for ", precision, ": ", activation);
        TORCH_CHECK((staging_config == std::array<int, 3>{} ||
                     staging_config == std::array<int, 3>{num_experts, num_max_tokens_per_rank, hidden}),
                    "MegaMoE base must have the same workspace dimensions for ", precision);
        return get_mega_moe_buffer_slices_for_precision(
            num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, precision);
    }
    parse_gated_activation(activation);
    const auto staging = get_staging_capacity(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, staging_config);
    const auto mega_buffer = layout::MegaMoEBuffer(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, staging);
    auto slice_input_buffers = [=](const torch::Tensor& buffer, const std::vector<int64_t>&, const int&) {
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
        auto l1_acts_sf = make_tensor(torch::kUInt16, mega_buffer.local_sfa,
            {mega_buffer.local_sfa.bytes / int64_t(sizeof(uint16_t))});
        auto l2_acts = make_tensor(torch::kUInt8, mega_buffer.l2_acts, {mega_buffer.compact, intermediate_hidden / 2});
        auto l2_acts_sf = make_tensor(torch::kUInt16, mega_buffer.l2_acts_sf,
            {mega_buffer.compact, intermediate_hidden / 64}, {1, mega_buffer.compact});
        return pybind11::make_tuple(x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
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
    DG_HOST_ASSERT(rank_idx >= 0 && rank_idx < num_ranks);
    DG_HOST_ASSERT(num_experts == num_experts_);
    const auto staging = get_staging_capacity(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, staging_config);
    const auto mega_buffer = layout::MegaMoEBuffer(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, staging);
    check_mega_moe_buffer(sym_buffer, mega_buffer);
    const int64_t tokens = num_tokens, groups = num_experts_per_rank_with_replicas;
    const int64_t owner_capacity = static_cast<int64_t>(num_ranks) * num_max_tokens_per_rank;
    const int em = std::max<int64_t>(1, std::min(owner_capacity,
        tokens * num_topk / groups + 1));

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

using RoutedMoEWorkspace = mega_moe_fp8_w4a16::RoutedMoEWorkspace;
using RoutedMoEViews = std::tuple<Tensor, Tensor, RoutedMoEWorkspace>;
using WeightPair = std::tuple<Tensor, Tensor>;

static mega_moe_config::fp8_w4a16::Precision routed_precision(const std::string& precision) {
    using P = mega_moe_config::fp8_w4a16::Precision;
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
    int intermediate_hidden, const std::string& precision) {
    using P = mega_moe_config::fp8_w4a16::Precision;
    const auto p = routed_precision(precision);
    const int max_ranks = p == P::FP8Channelwise ? 8 : (p == P::W4A16 || p == P::W4A8) ? 32 : 64;
    TORCH_CHECK(num_ranks > 0 && num_ranks <= max_ranks && num_experts > 0 && num_experts % num_ranks == 0 &&
                    num_topk > 0 && num_topk <= 16 && num_topk <= num_experts && num_max_tokens_per_rank > 0 && num_max_tokens_per_rank < (1 << 24),
                "Invalid MegaMoE ranks, experts, top-k or token capacity for ", precision);
    TORCH_CHECK(num_experts / num_ranks <= (p == P::W4A16 ? 127 : 128), "Unsupported local expert count for ", precision);
    TORCH_CHECK(hidden > 0 && hidden % 256 == 0 && intermediate_hidden > 0 && intermediate_hidden % 128 == 0 &&
                    intermediate_hidden <= std::numeric_limits<int>::max() / 2,
                "MegaMoE requires hidden divisible by 256 and intermediate_hidden=I divisible by 128");
    const auto pool = layout::get_num_max_pool_tokens<uint64_t>(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts / num_ranks);
    TORCH_CHECK(pool <= std::numeric_limits<int32_t>::max(), "MegaMoE pool exceeds signed 32-bit rows");
    return {uint32_t(num_ranks), uint32_t(num_experts), uint32_t(num_max_tokens_per_rank), uint32_t(num_topk),
            uint32_t(hidden), uint32_t(intermediate_hidden), p != P::W4A16, p == P::FP8Blockwise};
}

static RoutedMoEViews slice_routed_buffer(
    const Tensor& buffer, const layout::RoutedMoEBuffer& l, int num_experts, int num_ranks,
    int num_max_tokens_per_rank, int num_topk, int hidden, int intermediate_hidden, const std::string& precision) {
    check_tensor(buffer, torch::kUInt8);
    TORCH_CHECK(buffer.dim() == 1 && buffer.nbytes() >= l.get_num_bytes(), "MegaMoE buffer is too small");
    const bool fp8 = precision != "w4a16", blockwise = precision == "fp8_blockwise";
    const int64_t pool = l.dispatch.num_max_pool_tokens, groups = num_experts / num_ranks;
    auto view = [&](uint64_t off, std::vector<int64_t> shape, torch::ScalarType dtype) {
        return mega_moe_slice(buffer, off, shape, dtype);
    };
    const auto rows = view(l.gemm2_metadata, {5, groups}, torch::kInt32);
    auto scales =
        fp8 ? view(l.l2_scales,
                   blockwise ? std::vector<int64_t>{intermediate_hidden / 128, pool} : std::vector<int64_t>{pool, 1},
                   torch::kFloat32)
            : view(l.l2_scales, {0}, torch::kFloat32);
    if (blockwise)
        scales = scales.t();
    RoutedMoEWorkspace w{
        view(l.dispatched_tokens, {pool, int64_t(hidden) * (fp8 ? 1 : 2)}, torch::kUInt8),
        view(l.l2_acts, {pool, intermediate_hidden}, fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16),
        scales,
        view(l.bf16_scratch, fp8 && !blockwise ? std::vector<int64_t>{pool, intermediate_hidden} : std::vector<int64_t>{0},
             torch::kBFloat16),
        view(l.block_info, {int64_t(l.block_info_elements)}, torch::kInt32),
        rows[0],
        rows[1],
        rows[2],
        view(l.packed_routes, {pool}, torch::kInt64),
        view(l.combine, {int64_t(num_topk) * num_max_tokens_per_rank * hidden}, torch::kBFloat16),
        view(l.peer_offsets, {num_ranks}, torch::kInt64),
        buffer.narrow(0, layout::Workspace::kNumBarrierSignalBytes, (2LL * num_experts + groups) * sizeof(uint64_t)),
        view(l.topk_idx, {num_max_tokens_per_rank, num_topk}, torch::kInt64),
        view(l.topk_weights, {num_max_tokens_per_rank, num_topk}, torch::kFloat32),
        view(l.barrier_flag, {1}, torch::kInt32),
        view(l.barrier_done, {1}, torch::kInt32)};
    return {view(l.input_tokens, {num_max_tokens_per_rank, hidden}, fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16),
            view(l.input_scales, {num_max_tokens_per_rank, fp8 ? (blockwise ? hidden / 128 : 1) : 0}, torch::kFloat32),
            std::move(w)};
}

static std::tuple<int64_t, SymmBufferSlices> get_mega_moe_buffer_slices_for_precision(
    const int& num_ranks, const int& num_experts, const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden, const std::string& precision) {
    check_device();
    const auto l = checked_routed_layout(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, precision);
    SymmBufferSlices slices =
        [=](const Tensor& buffer, const std::vector<int64_t>& pointers, const int& rank_idx) {
            const auto transport = make_sym_buffer(pointers, rank_idx, num_ranks);
            TORCH_CHECK(transport.base == reinterpret_cast<int64_t>(buffer.data_ptr()),
                        "Invalid local symmetric pointer");
            auto [x, sf, w] =
                slice_routed_buffer(buffer, l, num_experts, num_ranks, num_max_tokens_per_rank,
                                    num_topk, hidden, intermediate_hidden, precision);
            std::vector<int64_t> offsets(num_ranks);
            for (int r = 0; r < num_ranks; ++r)
                offsets[r] = pointers[r] - pointers[rank_idx];
            w.peer_offsets.copy_(torch::tensor(offsets, torch::TensorOptions().dtype(torch::kInt64)));
            return py::make_tuple(x, sf, w.topk_idx, w.topk_weights);
        };
    return std::make_tuple(int64_t(l.get_num_bytes()), slices);
}

static void fp8_w4a16_mega_moe_pre_dispatch_checked(const Tensor& hidden_states, const Tensor& topk_idx,
                                         const Tensor& topk_weights, const Tensor& x, const Tensor& x_sf,
                                         const Tensor& staged_topk_idx, const Tensor& staged_topk_weights,
                                         const std::optional<int>& requested_num_tokens) {
    check_device();
    check_tensor(hidden_states, torch::kBFloat16);
    check_tensor(topk_idx, torch::kInt64);
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
        mega_moe_fp8_w4a16::w4a16_mega_moe_pre_dispatch_impl(
            hidden_states, topk_idx, topk_weights, x, staged_topk_idx,
            staged_topk_weights, num_tokens, hidden, num_topk);
    } else {
        check_tensor(x, torch::kFloat8_e4m3fn);
        const bool blockwise = x_sf.size(1) != 1;
        check_shape(x_sf, {num_max_tokens_per_rank, blockwise ? hidden / 128 : 1});
        if (blockwise)
            mega_moe_fp8_w4a16::fp8_blockwise_mega_moe_pre_dispatch_impl(
                hidden, num_topk, hidden_states, topk_idx, topk_weights, x, x_sf, staged_topk_idx, staged_topk_weights, num_tokens);
        else
            mega_moe_fp8_w4a16::fp8_channelwise_mega_moe_pre_dispatch_impl(
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
                            const bool& fast_math, const std::optional<WeightPair>& l1_weights_replicas,
                            const std::optional<WeightPair>& l2_weights_replicas, const float& swiglu_alpha,
                            const float& situ_beta, const float& situ_linear_beta) {
    check_device();
    const auto kind = routed_precision(precision);
    const bool fp8 = precision != "w4a16", blockwise = precision == "fp8_blockwise";
    TORCH_CHECK(fp8 == kFp8, "Compute entry does not match workspace precision");
    TORCH_CHECK(fp8 ? (activation == "swiglu" || activation == "swiglu_oai") : activation == "situ",
                "Unsupported activation for ", precision);
    TORCH_CHECK(!activation_clamp_opt || (!std::isnan(*activation_clamp_opt) && *activation_clamp_opt >= 0),
                "activation_clamp must be nonnegative or None");
    TORCH_CHECK(fp8 || !activation_clamp_opt, "W4A16 SiTU does not support activation_clamp");
    TORCH_CHECK(std::isfinite(swiglu_alpha) && swiglu_alpha > 0 && std::isfinite(situ_beta) && situ_beta > 0 &&
                    std::isfinite(situ_linear_beta) && situ_linear_beta > 0,
                "Invalid activation coefficients");
    check_tensor(y, torch::kBFloat16);
    TORCH_CHECK(y.dim() == 2, "output must be [num_tokens, hidden]");
    const auto& [w1, s1] = l1_weights;
    const auto& [w2, s2] = l2_weights;
    TORCH_CHECK(w1.dim() == 3 && w2.dim() == 3, "weights must be rank-3 tensors");
    const int hidden = y.size(1), intermediate_hidden = w1.size(1) / 2;
    const int num_tokens = y.size(0), num_ranks = sym_buffer_ptrs.size();
    const auto l = checked_routed_layout(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, precision);
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
    auto [x, sf, workspace] =
        slice_routed_buffer(sym_buffer, l, num_experts, num_ranks, num_max_tokens_per_rank,
                            num_topk, hidden, intermediate_hidden, precision);
    const auto& configs = mega_moe_config::fp8_w4a16::get_routed_moe_configs(
        kind, num_ranks, num_experts, num_topk, hidden, intermediate_hidden, num_max_tokens_per_rank,
        num_tokens, masters, activation, y.device().index());
    workspace.sym_buffer = transport;
    const auto& l1_replicas = l1_weights_replicas ? *l1_weights_replicas : l1_weights;
    const auto& l2_replicas = l2_weights_replicas ? *l2_weights_replicas : l2_weights;
    if constexpr (kFp8) {
        mega_moe_fp8_w4a16::fp8_fp8_mega_moe_impl(
            y, l1_weights, l2_weights, l1_replicas, l2_replicas, workspace, rank_idx, num_tokens,
            activation, activation_clamp_opt && std::isfinite(*activation_clamp_opt) ? *activation_clamp_opt : -1.0f,
            fast_math, swiglu_alpha, configs, kind, cumulative_local_expert_recv_stats);
    } else {
        mega_moe_fp8_w4a16::bf16_fp4_mega_moe_impl(
            y, l1_weights, l2_weights, l1_replicas, l2_replicas, workspace, rank_idx, num_tokens,
            fast_math, situ_beta, situ_linear_beta, configs, cumulative_local_expert_recv_stats);
    }
}

void fp8_fp8_mega_moe(
    const Tensor& y, const WeightPair& l1_weights_tuple, const WeightPair& l2_weights_tuple,
    const OptTensor& cumulative_local_expert_recv_stats, const Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk, const std::string& quantization,
    const std::string& activation, const std::optional<float>& activation_clamp_opt, const bool& fast_math,
    const std::optional<WeightPair>& l1_weights_replicas, const std::optional<WeightPair>& l2_weights_replicas,
    const float& swiglu_alpha) {
    checked_mega_moe<true>(
        y, l1_weights_tuple, l2_weights_tuple, cumulative_local_expert_recv_stats, sym_buffer,
        sym_buffer_ptrs, rank_idx, num_max_tokens_per_rank, num_experts, num_topk,
        fp8_mega_moe_precision(quantization), activation, activation_clamp_opt, fast_math,
        l1_weights_replicas, l2_weights_replicas, swiglu_alpha, 4.0f, 25.0f);
}

void bf16_fp4_mega_moe(
    const Tensor& y, const WeightPair& l1_weights_tuple, const WeightPair& l2_weights_tuple,
    const OptTensor& cumulative_local_expert_recv_stats, const Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::string& activation, const std::optional<float>& activation_clamp_opt, const bool& fast_math,
    const std::optional<WeightPair>& l1_weights_replicas, const std::optional<WeightPair>& l2_weights_replicas,
    const float& situ_beta, const float& situ_linear_beta) {
    checked_mega_moe<false>(
        y, l1_weights_tuple, l2_weights_tuple, cumulative_local_expert_recv_stats, sym_buffer,
        sym_buffer_ptrs, rank_idx, num_max_tokens_per_rank, num_experts, num_topk, "w4a16",
        activation, activation_clamp_opt, fast_math, l1_weights_replicas, l2_weights_replicas,
        1.702f, situ_beta, situ_linear_beta);
}

using W4A8Weights = mega_moe_w4a8::W4A8Weights;

void fp8_fp4_mega_moe(
    const Tensor& y, const W4A8Weights& l1_weights_tuple, const W4A8Weights& l2_weights_tuple,
    const OptTensor& cumulative_local_expert_recv_stats, const Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::string& activation, const std::optional<float>& activation_clamp_opt, const bool& fast_math,
    const std::optional<W4A8Weights>& l1_weights_replicas, const std::optional<W4A8Weights>& l2_weights_replicas,
    const float& swiglu_alpha, const float& situ_beta, const float& situ_linear_beta) {
    check_device();
    TORCH_CHECK(activation == "swiglu" || activation == "swiglu_oai" || activation == "situ",
                "W4A8 supports swiglu, swiglu_oai and situ");
    TORCH_CHECK(!activation_clamp_opt || (!std::isnan(*activation_clamp_opt) && *activation_clamp_opt >= 0),
                "activation_clamp must be nonnegative or None");
    TORCH_CHECK(activation != "situ" || !activation_clamp_opt, "SiTU does not support activation_clamp");
    TORCH_CHECK(std::isfinite(swiglu_alpha) && swiglu_alpha > 0 && std::isfinite(situ_beta) && situ_beta > 0 &&
                    std::isfinite(situ_linear_beta) && situ_linear_beta > 0, "Invalid activation coefficients");
    check_tensor(y, torch::kBFloat16);
    TORCH_CHECK(y.dim() == 2 && std::get<0>(l1_weights_tuple).dim() == 3, "Invalid output or GEMM1 weight rank");
    const int hidden = y.size(1), intermediate = std::get<0>(l1_weights_tuple).size(1) / 2;
    const int tokens = y.size(0), ranks = sym_buffer_ptrs.size();
    const auto l = checked_routed_layout(ranks, num_experts, num_max_tokens_per_rank, num_topk,
                                         hidden, intermediate, "w4a8");
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
    auto [x, sf, workspace] = slice_routed_buffer(sym_buffer, l, num_experts, ranks,
        num_max_tokens_per_rank, num_topk, hidden, intermediate, "w4a8");
    workspace.sym_buffer = transport;
    const auto& configs = mega_moe_config::fp8_w4a16::get_routed_moe_configs(
        mega_moe_config::fp8_w4a16::Precision::W4A8, ranks, num_experts, num_topk, hidden, intermediate,
        num_max_tokens_per_rank, tokens, masters, activation, y.device().index());
    mega_moe_w4a8::fp8_fp4_mega_moe_impl(y, l1_weights_tuple, l2_weights_tuple,
        l1_weights_replicas ? *l1_weights_replicas : l1_weights_tuple,
        l2_weights_replicas ? *l2_weights_replicas : l2_weights_tuple, workspace, rank_idx, tokens, activation,
        activation_clamp_opt && std::isfinite(*activation_clamp_opt) ? *activation_clamp_opt : -1.0f,
        fast_math, swiglu_alpha, situ_beta, situ_linear_beta, configs, cumulative_local_expert_recv_stats);
}

static void register_apis(pybind11::module_& m) {
    m.def("fp8_fp4_mega_moe", &fp8_fp4_mega_moe);
    m.def("bf16_fp4_mega_moe", &bf16_fp4_mega_moe);
    m.def("fp8_fp8_mega_moe", &fp8_fp8_mega_moe);
    m.def("routed_mega_moe_pre_dispatch", &routed_mega_moe_pre_dispatch);
    m.def("fp4_fp4_mega_moe", &fp4_fp4_mega_moe);
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
