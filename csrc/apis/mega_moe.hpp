#pragma once

#include <cmath>
#include <pybind11/stl.h>
#include "../jit_kernels/impls/mega_moe.hpp"
#include "../jit_kernels/impls/mega_moe_scatter.hpp"
#include "../jit_kernels/impls/mega_moe_combine.hpp"
#include "../jit_kernels/impls/m_grouped_w4a16_gemm.hpp"
#include "../jit_kernels/heuristics/mega_moe.hpp"

namespace deep_gemm::mega_moe {

// Shared API helpers.
namespace py = pybind11;
using Tensor = torch::Tensor;
using Pair = std::pair<Tensor, Tensor>;
using OptTensor = std::optional<Tensor>;
using OptPair = std::optional<Pair>;
using Config = mega_moe_config::Config;
using mega_moe_config::DispatchConfig;
using mega_moe_config::flag;
using mega_moe_config::integer;
using mega_moe_config::Precision;
using mega_moe_config::TuningConfig;
using OptTuning = std::optional<TuningConfig>;

static Precision parse_precision(const std::string& precision) {
    if (precision == "w4a4") return Precision::W4A4;
    if (precision == "w4a16") return Precision::W4A16;
    if (precision == "fp8_channelwise") return Precision::FP8Channelwise;
    if (precision == "fp8_blockwise") return Precision::FP8Blockwise;
    TORCH_CHECK(false, "precision must be w4a4, w4a16, fp8_channelwise or fp8_blockwise");
}

TuningConfig get_mega_moe_tuning_config(int ranks, int experts, int topk, int hidden,
                                       int intermediate, int tokens, const std::string& precision,
                                       const std::string& act) {
    TORCH_CHECK(ranks > 0 && ranks <= 32 && experts > 0 && experts % ranks == 0 &&
                topk > 0 && topk <= 16 && topk <= experts && hidden > 0 && intermediate > 0 && tokens >= 0,
                "Invalid MegaMoE tuning workload");
    TORCH_CHECK(act == "swiglu" || act == "swiglu_oai" || act == "situ" || act == "step",
                "Invalid MegaMoE tuning activation");
    return mega_moe_config::tuning_config(
        {parse_precision(precision), ranks, experts, topk, hidden, intermediate, tokens, act});
}

static void check_tuning(const OptTuning& tuning, Precision precision, int ranks, int experts,
                         int topk, int hidden, int intermediate, std::optional<int> tokens = std::nullopt) {
    if (!tuning)
        return;
    const auto& w = tuning->workload;
    TORCH_CHECK(w.precision == precision && w.ranks == ranks && w.experts == experts &&
                w.topk == topk && w.hidden == hidden && w.intermediate == intermediate &&
                (!tokens || w.tokens == *tokens), "MegaMoE tuning configuration does not match this workload");
}

template <typename T> static T field(const py::object& buffer, const char* key) {
    return buffer.attr(key).cast<T>();
}
static DispatchConfig read_dispatch_config(const py::dict& values) {
    DispatchConfig c;
    for (const auto& [key, member] : mega_moe_config::dispatch_config_fields)
        if (values.contains(key))
            c.*member = values[key].cast<int>();
    return c;
}

static std::map<std::string, int> dispatch_config_dict(const DispatchConfig& c, bool fp4 = true) {
    std::map<std::string, int> values;
    for (const auto& [key, member] : mega_moe_config::dispatch_config_fields) {
        const std::string name = key;
        if (fp4 || (name != "CHUNK_BLOCKS" && name.rfind("GEMM1_", 0) != 0))
            values[key] = c.*member;
    }
    return values;
}

static void check_tensor(const Tensor& t, torch::ScalarType dtype, bool contiguous = true) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == dtype && (!contiguous || t.is_contiguous()),
                "Invalid MegaMoE tensor device, dtype or layout");
}
static void check_shape(const Tensor& t, c10::IntArrayRef shape) {
    TORCH_CHECK(t.sizes() == shape, "Invalid MegaMoE tensor shape: ", t.sizes());
}
static std::string activation(std::optional<std::string> selected, bool swiglu = false, bool situ = false,
                              std::optional<float> alpha = std::nullopt, bool fp8 = false) {
    TORCH_CHECK(int(swiglu) + int(situ) + int(alpha.has_value()) <= 1,
                "Conflicting legacy activation selectors");
    const std::string legacy = situ ? "situ_and_mul" : swiglu ? "silu_and_mul" : alpha ? "swiglu_oai" : "";
    if (selected && !legacy.empty())
        TORCH_CHECK(*selected == legacy || (swiglu && *selected == "silu_and_mul_step"),
                    "Conflicting activation selectors");
    const std::string result = selected.value_or(legacy.empty() && fp8 ? "silu_and_mul" : legacy);
    TORCH_CHECK(fp8 ? (result == "silu_and_mul" || result == "swiglu_oai")
                    : (result.empty() || result == "silu_and_mul" || result == "silu_and_mul_step" ||
                       result == "situ_and_mul"),
                "Unsupported MegaMoE activation: ", result);
    TORCH_CHECK(result != "swiglu_oai" || alpha, "swiglu_oai requires swiglu_alpha");
    return result;
}
static DispatchConfig launch_config(const py::object& buffer, int tokens) {
    // The buffer owns the cache and allocations; the selector itself is native.
    TORCH_CHECK(tokens >= 0 && tokens <= field<int>(buffer, "num_max_tokens_per_rank"));
    auto cache = field<py::dict>(buffer, "_launch_cfg_cache");
    const auto key = py::int_(tokens);
    if (!cache.contains(key)) {
        DispatchConfig c;
        if (py::hasattr(buffer, "bf16_hidden")) {
            c = mega_moe_config::w4a16_dispatch_config(
                field<int>(buffer, "num_ranks"), field<int>(buffer, "num_experts"),
                field<int>(buffer, "num_topk"), field<int>(buffer, "bf16_hidden"),
                field<int>(buffer, "intermediate_hidden"), field<int>(buffer, "num_max_tokens_per_rank"),
                tokens, read_dispatch_config(field<py::dict>(buffer, "_capacity_cfg")).block_m);
        } else {
            const auto overrides = field<std::optional<mega_moe_config::Overrides>>(buffer, "_config_override");
            c = mega_moe_config::fp4_dispatch_config(
                field<int>(buffer, "num_ranks"), field<int>(buffer, "num_experts"),
                field<int>(buffer, "num_topk"), field<int>(buffer, "hidden"),
                field<int>(buffer, "intermediate_hidden"), field<int>(buffer, "num_max_tokens_per_rank"), tokens,
                field<int>(buffer, "_gemm1_block_n"), field<int>(buffer, "_gemm1_block_k"), 32, 64, 3, overrides);
            if (!overrides || overrides->empty()) {
                const int bm = c.block_m, bn = c.gemm1_block_n;
                TORCH_CHECK((bm == 16 || bm == 32 || bm == 64 || bm == 128 || bm == 256) &&
                            (bn == 128 || bn == 256));
            }
        }
        cache[key] = dispatch_config_dict(c);
    }
    return read_dispatch_config(cache[key].cast<py::dict>());
}
static std::pair<bool, float> fp8_activation_config(std::optional<float> alpha,
                                                    std::optional<std::string> selected) {
    const auto act = activation(selected, false, false, alpha, true);
    const float a = alpha.value_or(1.0f);
    TORCH_CHECK(std::isfinite(a) && a > 0, "swiglu_alpha must be finite and positive");
    return {act == "swiglu_oai", a};
}
static void validate_rank(const DispatchConfig& c, int rank, int tokens) {
    TORCH_CHECK(is_ppu1v5_device(), "MegaMoE requires PPU1.5");
    TORCH_CHECK(rank >= 0 && rank < c.num_ranks && tokens >= 0 && tokens <= c.nmtpr);
}

// Internal allocation configuration for the Python symmetric-buffer factory.
static std::map<std::string, int> dispatch_config(
    int num_ranks, int num_experts, int num_topk, int hidden, int num_max_tokens_per_rank,
    const std::string& precision, std::optional<int> intermediate_hidden,
    std::optional<int> block_m, std::optional<int> gemm1_block_n, std::optional<int> gemm1_block_k,
    const std::optional<mega_moe_config::Overrides>& overrides) {
    const auto kind = parse_precision(precision);
    if (kind == Precision::W4A4) {
        TORCH_CHECK(intermediate_hidden && !block_m,
                    "w4a4 requires intermediate_hidden; set BLOCK_M through overrides");
        return dispatch_config_dict(mega_moe_config::fp4_dispatch_config(
            num_ranks, num_experts, num_topk, hidden, *intermediate_hidden, num_max_tokens_per_rank,
            num_max_tokens_per_rank, gemm1_block_n.value_or(256), gemm1_block_k.value_or(128),
            32, 64, 3, overrides));
    }
    TORCH_CHECK(!gemm1_block_n && !gemm1_block_k && !overrides,
                "GEMM1 tile options and overrides are only supported for w4a4");
    if (kind == Precision::W4A16) {
        TORCH_CHECK(intermediate_hidden, "w4a16 requires intermediate_hidden");
        return dispatch_config_dict(mega_moe_config::w4a16_dispatch_config(
            num_ranks, num_experts, num_topk, hidden, *intermediate_hidden, num_max_tokens_per_rank,
            num_max_tokens_per_rank, block_m.value_or(64)));
    }
    TORCH_CHECK(!intermediate_hidden, "FP8 dispatch configuration does not select GEMM1 tiles");
    const bool blockwise = kind == Precision::FP8Blockwise;
    return dispatch_config_dict(mega_moe_config::fp8_dispatch_config(
        num_ranks, num_experts, num_topk, hidden, num_max_tokens_per_rank,
        block_m.value_or(blockwise ? 128 : 256), blockwise), false);
}

// BF16 and FP4 dispatch / fused GEMM1.
static void bf16_mega_moe_pre_dispatch(const Tensor& hidden_states, const Tensor& topk_ids,
                                const Tensor& topk_weights, const py::object& buffer, int num_tokens) {
    check_tensor(hidden_states, torch::kBFloat16);
    check_tensor(topk_ids, torch::kInt64);
    check_tensor(topk_weights, torch::kFloat32);
    check_shape(hidden_states, {num_tokens, field<int>(buffer, "bf16_hidden")});
    check_shape(topk_ids, {num_tokens, field<int>(buffer, "num_topk")});
    check_shape(topk_weights, topk_ids.sizes());
    TORCH_CHECK(num_tokens >= 0 && num_tokens <= field<int>(buffer, "num_max_tokens_per_rank"));
    if (num_tokens) {
        field<Tensor>(buffer, "input_token_bf16").narrow(0, 0, num_tokens).copy_(hidden_states);
        field<Tensor>(buffer, "input_topk_idx").narrow(0, 0, num_tokens).copy_(topk_ids);
        field<Tensor>(buffer, "input_topk_weights").narrow(0, 0, num_tokens).copy_(topk_weights);
    }
}
static void fp4_mega_moe_pre_dispatch(const Tensor& hidden_states, const Tensor& topk_ids,
                               const Tensor& topk_weights, const Tensor& input_token, const Tensor& input_sf,
                               const Tensor& input_topk_idx, const Tensor& input_topk_weights, int num_tokens,
                               int groups_per_block, std::optional<int> vec_block_threads) {
    TORCH_CHECK(is_ppu1v5_device());
    check_tensor(hidden_states, torch::kBFloat16);
    check_tensor(topk_ids, torch::kInt64);
    check_tensor(topk_weights, torch::kFloat32);
    check_tensor(input_token, torch::kUInt8);
    check_tensor(input_sf, torch::kUInt8);
    check_tensor(input_topk_idx, torch::kInt64);
    check_tensor(input_topk_weights, torch::kFloat32);
    const int hidden = hidden_states.size(1), topk = topk_ids.size(1);
    TORCH_CHECK(hidden > 0 && hidden % 64 == 0 && num_tokens >= 0 && num_tokens <= hidden_states.size(0));
    TORCH_CHECK(input_token.size(1) == hidden / 2 && input_sf.size(1) == hidden / 32 &&
                input_topk_idx.size(1) == topk && input_topk_weights.size(1) == topk);
    for (const auto& t : {input_token, input_sf, input_topk_idx, input_topk_weights, topk_ids, topk_weights})
        TORCH_CHECK(t.size(0) >= num_tokens);
    TORCH_CHECK(groups_per_block > 0 && groups_per_block <= 32);
    const int threads = vec_block_threads.value_or(
        integer("MEGA_MOE_PRE_DISPATCH_VEC_BLOCK_THREADS", mega_moe_config::pre_dispatch_threads(hidden)));
    TORCH_CHECK(threads > 0 && threads % 32 == 0 && threads <= hidden / 8);
    mega_moe_native::fp4_mega_moe_pre_dispatch_impl(hidden_states, topk_ids, topk_weights, input_token,
                                                    input_sf, input_topk_idx, input_topk_weights, num_tokens,
                                                    hidden, topk, groups_per_block, threads);
}

static int fp4_mega_moe_dispatch_gemm1(const py::object& buffer, int rank, int num_tokens, Tensor gemm1_weight,
                                Tensor gemm1_weight_sf, Tensor gemm1_weight_sf_u16, float activation_clamp,
                                bool fused_swiglu, OptTensor fp4_output, OptTensor scale_output,
                                bool quantize_output, OptTensor gemm1_weight_replicas,
                                OptTensor gemm1_weight_sf_u16_replicas, bool fused_situ, float situ_beta,
                                float situ_linear_beta, bool situ_tanh_bf16x2, bool dispatch_only,
                                int pull_begin_block, int pull_end_block, bool resume_dispatch,
                                bool finalize_dispatch, bool synchronize_pull, bool publish_arrivals,
                                int num_launch_blocks, std::optional<std::string> activation_type,
                                OptTensor output_override = std::nullopt, int gemm2_block_m = 0,
                                const mega_moe_config::Gemm1Tuning& tuning = {}) {
    const auto act = activation(activation_type, fused_swiglu, fused_situ);
    const auto capacity = read_dispatch_config(field<py::dict>(buffer, "_capacity_cfg"));
    validate_rank(capacity, rank, num_tokens);
    const bool glu = !act.empty(),
               split = pull_begin_block != 0 || pull_end_block >= 0 || resume_dispatch || !finalize_dispatch;
    TORCH_CHECK(!situ_tanh_bf16x2 || act == "situ_and_mul");
    TORCH_CHECK(!split || dispatch_only);
    TORCH_CHECK(pull_begin_block >= 0 && (pull_end_block < 0 || pull_end_block >= pull_begin_block));
    TORCH_CHECK(!resume_dispatch || pull_begin_block > 0 || publish_arrivals);
    if (num_launch_blocks < 0)
        num_launch_blocks = capacity.num_sms;
    TORCH_CHECK(num_launch_blocks >= 1 && num_launch_blocks <= 8 * capacity.num_sms);
    TORCH_CHECK(num_launch_blocks == capacity.num_sms ||
                (dispatch_only && resume_dispatch && !synchronize_pull && !finalize_dispatch));
    TORCH_CHECK(synchronize_pull || !finalize_dispatch);
    TORCH_CHECK(!publish_arrivals || resume_dispatch);
    if (act == "situ_and_mul")
        TORCH_CHECK(activation_clamp == 0 && std::isfinite(situ_beta) && situ_beta > 0 &&
                    std::isfinite(situ_linear_beta) && situ_linear_beta > 0);
    const bool dynamic = flag("MEGA_MOE_DYNAMIC_CONFIG", true);
    auto c = dynamic ? launch_config(buffer, num_tokens) : capacity;
    const auto selected_tuning = dynamic ? tuning : mega_moe_config::Gemm1Tuning{};
    mega_moe_config::apply_gemm1_tuning(c, selected_tuning,
        field<std::optional<mega_moe_config::Overrides>>(buffer, "_config_override"),
        field<int>(buffer, "_gemm1_block_n"), field<int>(buffer, "_gemm1_block_k"));
    const int sfb_bn = field<int>(buffer, "_sfb_layout_block_n");
    TORCH_CHECK(sfb_bn % c.gemm1_block_n == 0);
    if (gemm1_weight_sf_u16.scalar_type() != torch::kUInt16)
        gemm1_weight_sf_u16 = gemm1_weight_sf_u16.view(torch::kUInt16);
    TORCH_CHECK(gemm1_weight_replicas.has_value() == gemm1_weight_sf_u16_replicas.has_value());
    int masters = capacity.num_experts_per_rank;
    Tensor replica = gemm1_weight_replicas.value_or(gemm1_weight),
           replica_scale = gemm1_weight_sf_u16_replicas.value_or(gemm1_weight_sf_u16);
    if (gemm1_weight_replicas) {
        masters = gemm1_weight.size(0);
        TORCH_CHECK(masters + replica.size(0) == capacity.num_experts_per_rank);
        TORCH_CHECK(replica.sizes().slice(1) == gemm1_weight.sizes().slice(1));
        if (replica_scale.scalar_type() != torch::kUInt16)
            replica_scale = replica_scale.view(torch::kUInt16);
        TORCH_CHECK(replica_scale[0].numel() == gemm1_weight_sf_u16.reshape({masters, -1})[0].numel());
    }
    check_tensor(gemm1_weight, torch::kUInt8, false);
    check_tensor(gemm1_weight_sf, torch::kUInt8, false);
    check_tensor(replica, torch::kUInt8, false);
    gemm1_weight = gemm1_weight.contiguous();
    gemm1_weight_sf = gemm1_weight_sf.contiguous();
    gemm1_weight_sf_u16 = gemm1_weight_sf_u16.contiguous();
    replica = replica.contiguous();
    replica_scale = replica_scale.contiguous();
    const int pool = field<int>(buffer, "num_pool");
    Tensor output;
    int stride = 0;
    if (dispatch_only) {
        TORCH_CHECK(!glu && !quantize_output);
        output = torch::empty({1}, gemm1_weight.options().dtype(torch::kBFloat16));
    } else if (quantize_output) {
        TORCH_CHECK(glu && fp4_output && scale_output);
        check_tensor(*fp4_output, torch::kUInt8);
        check_tensor(*scale_output, torch::kUInt16, false);
        check_shape(*fp4_output, {pool, capacity.gemm1_n / 4});
        check_shape(*scale_output, {pool, capacity.gemm1_n / 128});
        TORCH_CHECK(scale_output->stride(0) == 1 && scale_output->stride(1) == pool);
        output = fp4_output->view(torch::kBFloat16).reshape({-1}).narrow(0, 0, 1);
    } else {
        output = output_override ? *output_override : buffer.attr("get_gemm1_output")(glu).cast<Tensor>();
        check_tensor(output, torch::kBFloat16);
        check_shape(output, {pool, glu ? capacity.gemm1_n / 2 : capacity.gemm1_n});
        stride = output.stride(0);
    }
    const auto scratch = buffer.attr("get_sfa_scratch")(pool * (capacity.sf_bytes / 2)).cast<Tensor>();

    return mega_moe_native::fp4_mega_moe_dispatch_gemm1_impl(
        c, capacity, field<Tensor>(buffer, "buf_ptrs"), gemm1_weight, gemm1_weight_sf, gemm1_weight_sf_u16,
        replica, replica_scale, output, scratch, rank, num_tokens, masters, stride, sfb_bn, fp4_output,
        scale_output, dynamic, py::hasattr(buffer, "bf16_hidden"), dispatch_only, split, glu, quantize_output,
        act, activation_clamp, situ_beta, situ_linear_beta, situ_tanh_bf16x2, pull_begin_block,
        pull_end_block, resume_dispatch, finalize_dispatch, synchronize_pull, publish_arrivals,
        num_launch_blocks, gemm2_block_m, selected_tuning);
}
// FP8 dispatch / fused GEMM1.
static void fp8_mega_moe_pre_dispatch(const Tensor& hidden_states, const Tensor& topk_ids,
                               const Tensor& topk_weights, const Tensor& input_token,
                               const Tensor& input_scale, const Tensor& input_topk_idx,
                               const Tensor& input_topk_weights, int num_tokens) {
    TORCH_CHECK(is_ppu1v5_device());
    check_tensor(hidden_states, torch::kBFloat16);
    check_tensor(topk_ids, torch::kInt64);
    check_tensor(topk_weights, torch::kFloat32);
    check_tensor(input_token, torch::kUInt8);
    check_tensor(input_scale, torch::kFloat32);
    check_tensor(input_topk_idx, torch::kInt64);
    check_tensor(input_topk_weights, torch::kFloat32);
    const int hidden = hidden_states.size(1), topk = topk_ids.size(1);
    TORCH_CHECK(hidden > 0 && hidden % 8 == 0 && num_tokens >= 0 && num_tokens <= hidden_states.size(0));
    TORCH_CHECK(topk_weights.sizes() == topk_ids.sizes() && input_token.size(1) == hidden &&
                input_scale.size(1) == 1);
    TORCH_CHECK(input_topk_idx.size(1) == topk && input_topk_weights.size(1) == topk);
    for (const auto& t : {input_token, input_scale, input_topk_idx, input_topk_weights, topk_ids})
        TORCH_CHECK(t.size(0) >= num_tokens);
    mega_moe_native::fp8_mega_moe_pre_dispatch_impl(hidden, topk, hidden_states, topk_ids, topk_weights,
                                                    input_token, input_scale, input_topk_idx,
                                                    input_topk_weights, num_tokens);
}

static int run_fp8_gemm1(const py::object& buffer, int rank, int num_tokens, const Tensor& weight,
                         const Tensor& scales, const Tensor& output, OptTensor quant, OptTensor quant_scale,
                         std::optional<float> limit, std::optional<bool> paired,
                         std::optional<bool> triple, OptTensor replicas, OptTensor replica_scales,
                         std::optional<float> alpha, std::optional<bool> separate,
                         std::optional<std::string> selected, const mega_moe_config::Gemm1Tuning& tuning = {}) {
    auto c = read_dispatch_config(field<py::dict>(buffer, "_cfg"));
    validate_rank(c, rank, num_tokens);
    check_tensor(weight, torch::kFloat8_e4m3fn);
    check_tensor(scales, torch::kFloat32);
    check_tensor(output, torch::kBFloat16);
    TORCH_CHECK(weight.dim() == 3 && field<int>(buffer, "num_scales") == 1);
    const int masters = weight.size(0), n = weight.size(1), hidden = weight.size(2),
              pool = field<int>(buffer, "num_pool");
    check_shape(scales, {masters, n, 1});
    TORCH_CHECK(replicas.has_value() == replica_scales.has_value());
    const auto rw = replicas.value_or(weight), rs = replica_scales.value_or(scales);
    if (replicas) {
        check_shape(rw, {rw.size(0), n, hidden});
        check_shape(rs, {rw.size(0), n, 1});
    }
    check_tensor(rw, torch::kFloat8_e4m3fn);
    check_tensor(rs, torch::kFloat32);
    TORCH_CHECK(masters + (replicas ? rw.size(0) : 0) == c.num_experts_per_rank);
    TORCH_CHECK(hidden == field<int>(buffer, "hidden") && hidden % 128 == 0 && n % 256 == 0 &&
                c.block_m == 256 && c.num_threads == 512);
    check_shape(output, {pool, n / 2});
    if (quant) {
        TORCH_CHECK(quant_scale);
        check_tensor(*quant, torch::kFloat8_e4m3fn);
        check_tensor(*quant_scale, torch::kFloat32);
        check_shape(*quant, {pool, n / 2});
        TORCH_CHECK(quant_scale->numel() == pool);
    }
    const auto [oai, activation_alpha] = fp8_activation_config(alpha, selected);
    return mega_moe_native::fp8_mega_moe_dispatch_gemm1_impl(
        c, field<Tensor>(buffer, "buf_ptrs"), rank, num_tokens, weight, scales, rw, rs, masters, output,
        quant, quant_scale, limit, paired, triple, oai, activation_alpha, separate, tuning);
}
static void fp8_blockwise_mega_moe_pre_dispatch(const py::object& buffer, const Tensor& hidden_states,
                                         const Tensor& topk_ids, const Tensor& topk_weights) {
    check_tensor(hidden_states, torch::kBFloat16);
    check_tensor(topk_ids, torch::kInt64);
    check_tensor(topk_weights, torch::kFloat32);
    const int m = hidden_states.size(0), hidden = hidden_states.size(1),
              topk = field<int>(buffer, "num_topk");
    TORCH_CHECK(hidden == field<int>(buffer, "hidden") && field<int>(buffer, "num_scales") == hidden / 128 &&
                m <= field<int>(buffer, "num_max_tokens_per_rank"));
    check_shape(topk_ids, {m, topk});
    check_shape(topk_weights, {m, topk});
    mega_moe_native::fp8_blockwise_mega_moe_pre_dispatch_impl(
        hidden, topk, hidden_states, topk_ids, topk_weights, field<Tensor>(buffer, "input_token"),
        field<Tensor>(buffer, "input_scale"), field<Tensor>(buffer, "input_topk_idx"),
        field<Tensor>(buffer, "input_topk_weights"), m);
}
static void check_blockwise_buffer(const py::object& buffer, int rank, int tokens) {
    const auto c = read_dispatch_config(field<py::dict>(buffer, "_cfg"));
    validate_rank(c, rank, tokens);
    const int ranks = c.num_ranks, hidden = field<int>(buffer, "hidden");
    TORCH_CHECK((ranks == 1 || ranks == 2 || ranks == 4 || ranks == 8) && hidden % 128 == 0 &&
                field<int>(buffer, "num_scales") == hidden / 128);
}
static int fp8_blockwise_mega_moe_dispatch(const py::object& buffer, int rank, int num_tokens) {
    check_blockwise_buffer(buffer, rank, num_tokens);
    const auto c = read_dispatch_config(field<py::dict>(buffer, "_cfg"));
    mega_moe_native::fp8_blockwise_mega_moe_dispatch_impl(c, field<Tensor>(buffer, "buf_ptrs"), rank,
                                                          num_tokens);
    return c.block_m;
}
static int fp8_blockwise_mega_moe_dispatch_gemm1(const py::object& buffer, int rank, int num_tokens,
                                          const Tensor& weight, const Tensor& weight_scales,
                                          const Tensor& output, OptTensor output_scales,
                                          std::optional<float> swiglu_limit, OptPair weight_replicas,
                                          std::optional<std::string> activation_type) {
    check_blockwise_buffer(buffer, rank, num_tokens);
    auto c = read_dispatch_config(field<py::dict>(buffer, "_cfg"));
    const bool quant = output_scales.has_value(), clamp = swiglu_limit.has_value();
    const float limit = swiglu_limit.value_or(0.0f);
    TORCH_CHECK(!activation_type || (quant && *activation_type == "silu_and_mul"));
    TORCH_CHECK(!clamp || (quant && std::isfinite(limit) && limit >= 0));
    check_tensor(weight, torch::kFloat8_e4m3fn);
    check_tensor(weight_scales, torch::kFloat32);
    check_tensor(output, quant ? torch::kFloat8_e4m3fn : torch::kBFloat16);
    TORCH_CHECK(weight.dim() == 3);
    const int e = weight.size(0), n = weight.size(1), hidden = weight.size(2),
              pool = field<int>(buffer, "num_pool");
    TORCH_CHECK(hidden == field<int>(buffer, "hidden") && hidden % 128 == 0 && n % 128 == 0);
    const auto rw = weight_replicas ? weight_replicas->first : weight,
               rs = weight_replicas ? weight_replicas->second : weight_scales;
    check_tensor(rw, torch::kFloat8_e4m3fn);
    check_tensor(rs, torch::kFloat32);
    if (weight_replicas) {
        check_shape(rw, {rw.size(0), n, hidden});
        check_shape(rs, {rw.size(0), n / (quant ? 8 : 128), hidden / 128});
    }
    const int groups = e + (weight_replicas ? rw.size(0) : 0);
    TORCH_CHECK(e > 0 && groups == c.num_experts_per_rank && groups <= 128);
    check_shape(weight_scales, {e, n / (quant ? 8 : 128), hidden / 128});
    check_shape(output, {pool, quant ? n / 2 : n});
    if (quant) {
        check_tensor(*output_scales, torch::kFloat32, false);
        check_shape(*output_scales, {pool, n / 256});
        TORCH_CHECK(n % 256 == 0 && output_scales->stride(0) == 1 && output_scales->stride(1) == pool);
    }
    TORCH_CHECK(c.block_m == 128);
    return mega_moe_native::fp8_blockwise_mega_moe_dispatch_gemm1_impl(
        c, field<Tensor>(buffer, "buf_ptrs"), weight, weight_scales, output, rw, rs,
        rank, num_tokens, n, e, quant, clamp, limit, output_scales);
}
Pair interleave_fp8_blockwise_mega_moe_weights(const Tensor& weight, const Tensor& scales) {
    TORCH_CHECK(weight.dim() == 3 && weight.scalar_type() == torch::kFloat8_e4m3fn);
    const int e = weight.size(0), n = weight.size(1), k = weight.size(2);
    TORCH_CHECK(n % 256 == 0 && k % 128 == 0 && scales.scalar_type() == torch::kFloat32);
    check_shape(scales, {e, n / 128, k / 128});
    auto packed = weight.view(torch::kUInt8).reshape({e, 2, n / 16, 8, k}).transpose(1, 2).contiguous();
    auto sf = scales.repeat_interleave(16, 1).reshape({e, 2, n / 16, k / 128}).transpose(1, 2).contiguous();
    return {packed.reshape({e, n, k}).view(weight.scalar_type()), sf.reshape({e, n / 8, k / 128})};
}

// Grouped GEMM2 with scatter.
static bool moe_fp4_scale_layout(const Tensor& scale) {
    return scale.scalar_type() == torch::kUInt16 &&
           ((scale.dim() == 2 && (scale.stride(0) == 1 || scale.size(0) == 1)) ||
            (scale.dim() == 3 && (scale.stride(1) == 1 || scale.size(1) == 1)));
}
static Tensor fp4_scales(const Tensor& scale) {
    return moe_fp4_scale_layout(scale) ? scale : deep_gemm_fp4_common::preprocess_mxfp4_scales(scale);
}

static void m_grouped_gemm_fp4_with_scatter(const Pair& lhs_, const Pair& rhs_, OptTensor m_indices,
                                     const Tensor& token_meta, const Tensor& combine_buf,
                                     const Tensor& sym_offsets, int rank, int world_size, int max_tokens,
                                     int hidden, OptTensor m_rows, std::optional<Config> configs,
                                     int num_topk, OptTensor valid_m_rows,
                                     OptTensor combine_scales, OptTensor scale_offsets, int scale_block_n,
                                     std::optional<bool> force_fp8_scatter, OptTensor src_m_offsets,
                                     std::optional<int> expected_m, OptPair rhs_replicas_) {
    const auto& lhs = lhs_.first;
    const auto& rhs = rhs_.first;
    TORCH_CHECK(lhs.dim() == 2 && rhs.dim() == 3);
    const int m = lhs.size(0), k = lhs.size(1), masters = rhs.size(0), n = rhs.size(1);
    auto sa = fp4_scales(lhs_.second), sb = fp4_scales(rhs_.second);
    const auto replica = rhs_replicas_ ? rhs_replicas_->first : rhs;
    const auto rs = rhs_replicas_ ? fp4_scales(rhs_replicas_->second) : sb;
    TORCH_CHECK(replica.dim() == 3 && replica.sizes().slice(1) == rhs.sizes().slice(1) && rhs.size(2) == k);
    check_tensor(lhs, torch::kUInt8);
    check_tensor(rhs, torch::kUInt8);
    check_tensor(replica, torch::kUInt8);
    check_tensor(sa, torch::kUInt16, false);
    check_tensor(sb, torch::kUInt16, false);
    check_tensor(rs, torch::kUInt16, false);
    const int groups = masters + (rhs_replicas_ ? replica.size(0) : 0);
    TORCH_CHECK(groups > 0 && hidden == n && rank >= 0 && rank < world_size && world_size <= 64 &&
                max_tokens > 0 && num_topk > 0);
    const bool fp8 = force_fp8_scatter.value_or(get_env<std::string>("DG_MOE_SCATTER_FP8", "0") == "1");
    const bool unit = fp8 && mega_moe_config::fp8_unit_scale();
    int shift = integer("DG_MOE_SCATTER_FP8_EXP_SHIFT", 0);
    TORCH_CHECK(shift >= 0 && shift <= 12);
    if (unit)
        shift = 0;
    check_tensor(combine_buf, fp8 ? torch::kUInt8 : torch::kBFloat16);
    TORCH_CHECK(combine_buf.numel() >= int64_t(num_topk) * max_tokens * hidden);
    check_tensor(token_meta, torch::kInt64);
    TORCH_CHECK(token_meta.dim() == 1 && token_meta.numel() >= m);
    const auto rows = m_rows.value_or(Tensor());
    TORCH_CHECK(rows.defined() || m_indices, "m_indices is required when m_rows is omitted");
    Tensor physical =
        rows.defined() ? rows : torch::bincount(*m_indices, std::nullopt, groups).to(torch::kInt32);
    const auto valid = valid_m_rows.value_or(physical);
    check_tensor(physical, torch::kInt32);
    check_tensor(valid, torch::kInt32);
    check_shape(physical, {groups});
    check_shape(valid, {groups});
    if (src_m_offsets) {
        check_tensor(*src_m_offsets, torch::kInt32);
        TORCH_CHECK(src_m_offsets->numel() >= groups);
    }
    mega_moe_native::m_grouped_gemm_fp4_with_scatter_impl(
        lhs, sa, rhs, sb, replica, rs, physical, valid, token_meta, combine_buf, sym_offsets, m, n, k, groups,
        masters, rank, world_size, max_tokens, hidden, num_topk, m_rows.has_value(), fp8, shift,
        configs, expected_m, combine_scales, scale_offsets, scale_block_n, src_m_offsets);
}
static void m_grouped_gemm_fp8_blockwise_with_scatter(const Pair& lhs, const Pair& rhs, OptTensor scratch,
                                               const Tensor& padded_rows, const Tensor& valid_rows,
                                               const Tensor& row_offsets, const Tensor& packed_src_info,
                                               const Tensor& combine, const Tensor& peer_offsets,
                                               int max_tokens, int topk, int rank, OptPair rhs_replicas) {
    const auto& [a, sa] = lhs;
    const auto& [b, sb] = rhs;
    check_tensor(a, torch::kFloat8_e4m3fn);
    check_tensor(b, torch::kFloat8_e4m3fn);
    check_tensor(sa, torch::kFloat32, false);
    check_tensor(sb, torch::kFloat32);
    TORCH_CHECK(a.dim() == 2 && b.dim() == 3);
    const int m = a.size(0), k = a.size(1), masters = b.size(0), n = b.size(1);
    const auto out = scratch.value_or(combine.reshape({-1}).narrow(0, 0, n).view({1, n}));
    const auto rw = rhs_replicas ? rhs_replicas->first : b, rs = rhs_replicas ? rhs_replicas->second : sb;
    const int groups = masters + (rhs_replicas ? rw.size(0) : 0), ranks = peer_offsets.numel();
    check_tensor(rw, torch::kFloat8_e4m3fn);
    check_tensor(rs, torch::kFloat32);
    check_shape(rw, {rw.size(0), n, k});
    check_shape(rs, {rw.size(0), n / 128, k / 128});
    for (const auto& t : {out, combine})
        check_tensor(t, torch::kBFloat16);
    for (const auto& t : {padded_rows, valid_rows, row_offsets}) {
        check_tensor(t, torch::kInt32);
        check_shape(t, {groups});
    }
    for (const auto& t : {packed_src_info, peer_offsets})
        check_tensor(t, torch::kInt64);
    for (const auto& t : {sa, b, sb, out, padded_rows, valid_rows, row_offsets, packed_src_info, combine,
                          peer_offsets, rw, rs})
        TORCH_CHECK(t.device() == a.device());
    TORCH_CHECK(k == b.size(2) && k % 128 == 0 && n % 256 == 0 && groups >= 1 && groups <= 1024);
    check_shape(sa, {m, k / 128});
    TORCH_CHECK(sa.stride(0) == 1 && sa.stride(1) == m);
    check_shape(sb, {masters, n / 128, k / 128});
    TORCH_CHECK(out.dim() == 2 && (out.size(0) == m || out.size(0) == 1) && out.size(1) == n);
    check_shape(packed_src_info, {m});
    TORCH_CHECK((ranks == 1 || ranks == 2 || ranks == 4 || ranks == 8) && rank >= 0 && rank < ranks);
    TORCH_CHECK(max_tokens > 0 && max_tokens < (1 << 24) && topk > 0 && topk <= 256 &&
                combine.numel() == int64_t(topk) * max_tokens * n);
    mega_moe_native::m_grouped_gemm_fp8_blockwise_with_scatter_impl(
        a, sa, b, sb, out, padded_rows, valid_rows, row_offsets, packed_src_info, combine, peer_offsets, rw,
        rs, m, n, k, groups, masters, max_tokens, topk, rank, ranks);
}

static void validate_grouped_rows(const Tensor& indices, const Tensor& rows, const Tensor& valid, int m,
                                  int groups) {
    const auto snapshot = torch::cat({rows, valid, indices.reshape({-1})}).detach().cpu();
    const auto values = snapshot.data_ptr<int32_t>();
    int64_t sum = 0;
    for (int e = 0; e < groups; ++e) {
        TORCH_CHECK(values[e] >= 0 && values[groups + e] >= 0 && values[groups + e] <= values[e]);
        sum += values[e];
    }
    TORCH_CHECK(sum == m, "m_rows must sum to GEMM M");
    int cursor = 0;
    for (int e = 0; e < groups; ++e)
        for (int i = 0; i < values[e]; ++i, ++cursor)
            TORCH_CHECK(values[2 * groups + cursor] == e,
                        "m_indices must contain contiguous expert segments");
}
static void m_grouped_gemm_fp8_channelwise_with_scatter(
    const Pair& lhs_, const Pair& rhs_, const Tensor& m_indices, const Tensor& token_meta,
    const Tensor& combine_buf, const Tensor& sym_offsets, int rank, int world_size, int max_tokens,
    int hidden, OptTensor m_rows, std::optional<Config> configs, int num_topk, OptTensor valid_m_rows,
    std::optional<std::string> kernel_type_override, bool validate_metadata,
    std::optional<bool> check_packed_routes, std::optional<bool> rolling_pair_scatter,
    std::optional<int> num_sms_override, OptPair rhs_replicas_,
    const mega_moe_config::Gemm2Tuning& tuning = {}) {
    const auto& [lhs, sa] = lhs_;
    const auto& [rhs, sb] = rhs_;
    TORCH_CHECK(lhs.dim() == 2 && rhs.dim() == 3);
    const int m = lhs.size(0), k = lhs.size(1), masters = rhs.size(0), n = rhs.size(1);
    const auto rw = rhs_replicas_ ? rhs_replicas_->first : rhs,
               rs = rhs_replicas_ ? rhs_replicas_->second : sb;
    TORCH_CHECK(rw.dim() == 3);
    const int groups = masters + (rhs_replicas_ ? rw.size(0) : 0);
    TORCH_CHECK(groups > 0 && rhs.size(2) == k && n == hidden && n % 256 == 0);
    for (const auto& t : {lhs, rhs, rw})
        check_tensor(t, torch::kFloat8_e4m3fn);
    for (const auto& t : {sa, sb, rs})
        check_tensor(t, torch::kFloat32);
    check_shape(sa, {m, 1});
    check_shape(sb, {masters, n, 1});
    check_shape(rw, {rw.size(0), n, k});
    check_shape(rs, {rw.size(0), n, 1});
    check_tensor(combine_buf, torch::kBFloat16);
    check_tensor(m_indices, torch::kInt32);
    check_shape(m_indices, {m});
    check_tensor(token_meta, torch::kInt64);
    TORCH_CHECK(token_meta.numel() >= m);
    check_tensor(sym_offsets, torch::kInt64);
    TORCH_CHECK(sym_offsets.numel() >= world_size);
    TORCH_CHECK(rank >= 0 && rank < world_size && world_size <= 8 && max_tokens > 0 &&
                max_tokens <= (1 << 24) && num_topk > 0 && num_topk <= 256);
    TORCH_CHECK(combine_buf.numel() >= int64_t(num_topk) * max_tokens * hidden);
    int device;
    DG_HGGC_RUNTIME_CHECK(hggcGetDevice(&device));
    TORCH_CHECK(lhs.get_device() == device);
    for (const auto& t : {sa, rhs, sb, rw, rs, m_indices, token_meta, combine_buf, sym_offsets})
        TORCH_CHECK(t.device() == lhs.device());
    if (m == 0)
        return;
    const auto rows = m_rows.value_or(Tensor());
    const auto physical =
        rows.defined()
            ? rows
            : torch::bincount(m_indices, std::nullopt, groups).narrow(0, 0, groups).to(torch::kInt32);
    const auto valid = valid_m_rows.value_or(physical);
    for (const auto& t : {physical, valid}) {
        check_tensor(t, torch::kInt32);
        check_shape(t, {groups});
        TORCH_CHECK(t.device() == lhs.device());
    }
    if (validate_metadata)
        validate_grouped_rows(m_indices, physical, valid, m, groups);
    const bool checked = check_packed_routes.value_or(validate_metadata);
    TORCH_CHECK(!validate_metadata || checked);
    mega_moe_native::m_grouped_gemm_fp8_channelwise_with_scatter_impl(
        lhs, sa, rhs, sb, rw, rs, m_indices, token_meta, combine_buf, sym_offsets, physical, valid, m,
        n, k, groups, masters, world_size, rank, max_tokens, hidden, num_topk, checked, configs,
        kernel_type_override, rolling_pair_scatter, num_sms_override, tuning);
}
static void check_w4a16_scatter(const torch::Tensor& lhs, const torch::Tensor& rhs,
                                const torch::Tensor& scales, const torch::Tensor& combine,
                                const torch::Tensor& rows, const torch::Tensor& packed_src_info,
                                const torch::Tensor& sym_offsets, const torch::Tensor& valid_m_rows, int rank,
                                int max_tokens, int topk,
                                const std::optional<std::pair<torch::Tensor, torch::Tensor>>& replicas) {
    constexpr int group_size = 32;
    TORCH_CHECK(lhs.dim() == 2, "Invalid W4A16 activation rank");
    const int m = lhs.size(0), k = lhs.size(-1);
    if (m == 0)
        return;
    TORCH_CHECK(rhs.dim() == 3 and scales.dim() == 3, "Invalid W4A16 weight layout");
    const int n = rhs.size(1), master_groups = rhs.size(0);
    const torch::Tensor replica_rhs = replicas ? replicas->first : rhs;
    const torch::Tensor replica_scales = replicas ? replicas->second : scales;
    const int groups = master_groups + (replicas ? replica_rhs.size(0) : 0);
    TORCH_CHECK(is_ppu1v5_device(), "W4A16 MXFP4 requires PPU1.5 and group size 32");
    TORCH_CHECK(n > 0 and n % 64 == 0 and k > 0 and k % group_size == 0, "Invalid W4A16 N/K/group size");
    const auto check_weight = [&](const torch::Tensor& weight, const torch::Tensor& scale) {
        TORCH_CHECK(weight.dim() == 3 and scale.dim() == 3 and scale.size(0) == weight.size(0),
                    "Invalid W4A16 weight/scale rank");
        TORCH_CHECK(weight.size(1) == n and weight.size(2) == k / 2 and scale.size(1) == n / 64 and
                        scale.size(2) == ceil_div(k, 64) * 128,
                    "MXFP4 requires mainline [E,N,K/2] weights and [E,N/64,ceil(K/64)*128] scales");
    };
    check_weight(rhs, scales);
    TORCH_CHECK(lhs.scalar_type() == torch::kBFloat16 and combine.scalar_type() == torch::kBFloat16 and
                    rhs.scalar_type() == torch::kUInt8 and replica_rhs.scalar_type() == rhs.scalar_type(),
                "Invalid W4A16 operand dtype");
    TORCH_CHECK(scales.scalar_type() == torch::kUInt8 and replica_scales.scalar_type() == torch::kUInt8,
                "Invalid W4A16 scale dtype");
    TORCH_CHECK(lhs.is_contiguous() and rhs.is_contiguous() and scales.is_contiguous() and
                    combine.is_contiguous() and replica_rhs.is_contiguous() and
                    replica_scales.is_contiguous() and rows.is_contiguous() and
                    rows.scalar_type() == torch::kInt32 and rows.numel() == groups,
                "Invalid W4A16 layout or row counts");
    TORCH_CHECK(packed_src_info.scalar_type() == torch::kInt64 and packed_src_info.is_contiguous() and
                    packed_src_info.dim() == 1 and packed_src_info.numel() >= m,
                "Invalid W4A16 scatter metadata");
    TORCH_CHECK(sym_offsets.scalar_type() == torch::kInt64 and sym_offsets.is_contiguous() and
                    sym_offsets.dim() == 1 and rank >= 0 and rank < sym_offsets.numel(),
                "Invalid W4A16 scatter rank offsets");
    TORCH_CHECK(valid_m_rows.scalar_type() == torch::kInt32 and valid_m_rows.is_contiguous() and
                    valid_m_rows.numel() == groups,
                "Invalid W4A16 scatter valid rows");
    TORCH_CHECK(max_tokens > 0 and topk > 0 and combine.numel() >= int64_t(topk) * max_tokens * n,
                "Invalid W4A16 combine capacity");
    if (replicas)
        check_weight(replica_rhs, replica_scales);
}

static void m_grouped_gemm_w4a16_mxfp4_with_scatter(const Tensor& lhs, const Pair& rhs_,
                                             const Tensor& packed_src_info, const Tensor& combine_buf,
                                             const Tensor& sym_offsets, const Tensor& valid_m_rows,
                                             const Tensor& m_rows, int rank, int max_tokens, int topk,
                                             std::optional<PPUW4ConfigTuple> configs, OptPair rhs_replicas_) {
    TORCH_CHECK(lhs.dim() == 2 && rhs_.first.dim() == 3);
    const int m = lhs.size(0), n = rhs_.first.size(1),
              groups = rhs_.first.size(0) + (rhs_replicas_ ? rhs_replicas_->first.size(0) : 0);
    TORCH_CHECK(m > 0 && groups > 0 && n > 0 && n % 64 == 0);
    check_w4a16_scatter(lhs, rhs_.first, rhs_.second, combine_buf, m_rows, packed_src_info, sym_offsets,
                        valid_m_rows, rank, max_tokens, topk, rhs_replicas_);
    mega_moe_native::m_grouped_gemm_w4a16_mxfp4_with_scatter_impl(
        lhs, rhs_.first, rhs_.second, combine_buf, m_rows, packed_src_info, sym_offsets, valid_m_rows, rank,
        max_tokens, topk, configs, rhs_replicas_);
}
Pair preprocess_w4a16_mxfp4(const Pair& rhs_) {
    return {ppu_w4a16_preprocess_weights(rhs_.first),
            deep_gemm_w4a16_common::ppu_w4a16_preprocess_scales(rhs_.second)};
}

// Synchronization and combine.
// All ranks call this after GEMM2/scatter on the same stream, including empty
// ranks. Zero weights mask unwritten slots; valid routes have already scattered.
// Counters start at zero and must not be shared by concurrent MoE invocations.
torch::Tensor mega_moe_combine(const torch::Tensor& buf, const torch::Tensor& weights, int tokens,
                             int hidden, int topk, int capacity, const torch::Tensor& pointers,
                             int rank, int world, const torch::Tensor& done,
                             OptTensor scales, int scale_block_n, OptTuning tuning) {
    TORCH_CHECK(topk >= 1 && topk <= 16, "unsupported MoE reduce topk: ", topk);
    TORCH_CHECK(tokens >= 0 && tokens <= capacity && capacity > 0 && hidden > 0 && hidden % 8 == 0);
    const bool fp8 = buf.scalar_type() == torch::kFloat8_e4m3fn;
    if (tuning) {
        const auto& w = tuning->workload;
        TORCH_CHECK(w.tokens == tokens && w.hidden == hidden && w.topk == topk && w.ranks == world,
                    "MegaMoE tuning configuration does not match this combine workload");
        TORCH_CHECK(!fp8 || tuning->profile.empty(), "measured MegaMoE profiles require BF16 scatter");
    }
    TORCH_CHECK(fp8 || buf.scalar_type() == torch::kBFloat16, "combine buffer must be BF16 or E4M3 FP8");
    check_tensor(buf, buf.scalar_type());
    check_tensor(weights, torch::kFloat32);
    TORCH_CHECK(weights.numel() >= int64_t(tokens) * topk);
    TORCH_CHECK(buf.numel() >= int64_t(capacity) * hidden * topk);
    TORCH_CHECK(world > 0 && world <= 32 && rank >= 0 && rank < world);
    check_tensor(pointers, torch::kInt64);
    check_tensor(done, torch::kInt32);
    TORCH_CHECK(pointers.numel() == world && done.numel() == 1);
    for (const auto& tensor : {weights, pointers, done})
        TORCH_CHECK(tensor.device() == buf.device());
    if (scales) {
        TORCH_CHECK(fp8 && scale_block_n >= 64 && scale_block_n % 8 == 0 && hidden % scale_block_n == 0);
        check_tensor(*scales, torch::kInt64);
        TORCH_CHECK(scales->device() == buf.device());
        TORCH_CHECK(scales->numel() >= int64_t(topk) * capacity * ceil_div(hidden, scale_block_n));
    } else {
        TORCH_CHECK(scale_block_n == 0, "scale_block_n requires combine_scales");
    }
    auto out = torch::empty({tokens, hidden}, buf.options().dtype(torch::kBFloat16));
    mega_moe_aux::barrier_reduce_impl(buf, weights, tokens, hidden, topk, capacity, pointers, rank, world,
                                      done, scales, scale_block_n, out,
                                      tuning ? tuning->combine : mega_moe_config::CombineTuning{});
    return out;
}

// Symmetric layout and row metadata.
static std::map<std::string, int64_t> symmetric_layout(const std::map<std::string, int64_t>& config,
                                                       bool fp4) {
    const auto value = [&](const char* key) {
        const auto found = config.find(key);
        TORCH_CHECK(found != config.end(), "missing MegaMoE configuration key: ", key);
        return uint32_t(found->second);
    };
    const auto ranks = value("NUM_RANKS"), experts = value("NUM_EXPERTS");
    TORCH_CHECK(ranks > 0 && experts % ranks == 0, "invalid MegaMoE expert/rank layout");
    const auto capacity = value("NMTPR"), topk = value("NUM_TOPK");
    TORCH_CHECK(capacity < (1u << 24) && topk <= 256,
                "packed routes require tokens < 2^24 and topk <= 256");
    const uint64_t token_bytes = value("TOKEN_BYTES"), scale_bytes = value("SF_BYTES");
    const layout::Workspace workspace(nullptr, ranks, experts, capacity, topk);
    const auto align16 = [](uint64_t bytes) { return (bytes + 15) / 16 * 16; };
    const uint64_t ws_bytes = workspace.get_num_bytes();
    const uint64_t pool = workspace.num_max_pool_tokens;
    const uint64_t off_it = align16(ws_bytes);
    const uint64_t off_isf = align16(off_it + uint64_t(capacity) * token_bytes);
    const uint64_t off_iti = align16(off_isf + uint64_t(capacity) * scale_bytes);
    const uint64_t off_itw = align16(off_iti + uint64_t(capacity) * topk * sizeof(int64_t));
    const uint64_t off_pt = align16(off_itw + uint64_t(capacity) * topk * sizeof(float));
    const uint64_t off_psf = align16(off_pt + pool * token_bytes);
    const uint64_t off_pw = align16(off_psf + pool * scale_bytes);
    const uint64_t off_meta = align16(off_pw + pool * sizeof(float));
    const uint64_t off_packed = off_meta + layout::Gemm2Metadata::rows_bytes(experts / ranks);
    std::map<std::string, int64_t> output{
        {"total", off_packed + pool * sizeof(int64_t)},
        {"off_gemm2_meta", off_meta},
        {"off_gemm2_packed", off_packed},
        {"off_ws", 0},
        {"off_it", off_it},
        {"off_isf", off_isf},
        {"off_iti", off_iti},
        {"off_itw", off_itw},
        {"off_pt", off_pt},
        {"off_psf", off_psf},
        {"off_pw", off_pw},
        {"num_pool", pool},
        {"num_pool_blocks", workspace.num_max_pool_blocks},
        {"woff_rc", reinterpret_cast<uintptr_t>(workspace.get_expert_recv_count_ptr(0, 0))},
        {"woff_rcs", reinterpret_cast<uintptr_t>(workspace.get_expert_recv_count_sum_ptr(0))},
        {"woff_meta", reinterpret_cast<uintptr_t>(workspace.get_token_src_metadata_ptr(0))},
        {"ws_bytes", ws_bytes},
    };
    if (fp4) {
        output["off_out"] = 0;
        output["woff_arr"] = reinterpret_cast<uintptr_t>(workspace.get_l1_arrival_count_ptr(0));
    }
    return output;
}

// The four public stages select a backend with explicit, bounded options.
// Backend adapters above validate the physical tensor and workspace layouts.
static void check_buffer_precision(const py::object& buffer, Precision precision) {
    const bool bf16 = py::hasattr(buffer, "bf16_hidden");
    const bool fp8 = py::hasattr(buffer, "num_scales");
    if (precision == Precision::W4A4 || precision == Precision::W4A16) {
        TORCH_CHECK(!fp8 && bf16 == (precision == Precision::W4A16),
                    "buffer does not match the selected precision");
        if (precision == Precision::W4A4)
            TORCH_CHECK(field<int>(buffer, "hidden") % 512 == 0,
                        "W4A4 dispatch requires hidden divisible by 512 for scale copies");
    } else {
        TORCH_CHECK(fp8 && !bf16, "buffer does not match the selected precision");
        const int scales = precision == Precision::FP8Channelwise ? 1 : field<int>(buffer, "hidden") / 128;
        TORCH_CHECK(field<int>(buffer, "num_scales") == scales,
                    "buffer scale granularity does not match precision");
    }
}

void mega_moe_pre_dispatch(const Tensor& hidden_states, const Tensor& topk_ids,
                           const Tensor& topk_weights, const py::object& buffer, int num_tokens,
                           const std::string& precision) {
    const auto kind = parse_precision(precision);
    check_buffer_precision(buffer, kind);
    TORCH_CHECK(hidden_states.dim() == 2 && topk_ids.dim() == 2 && topk_weights.dim() == 2);
    TORCH_CHECK(num_tokens >= 0 && num_tokens <= field<int>(buffer, "num_max_tokens_per_rank"));
    TORCH_CHECK(hidden_states.size(0) >= num_tokens && topk_ids.size(0) >= num_tokens &&
                topk_weights.sizes() == topk_ids.sizes());
    const int hidden = field<int>(buffer, kind == Precision::W4A16 ? "bf16_hidden" : "hidden");
    TORCH_CHECK(hidden_states.size(1) == hidden && topk_ids.size(1) == field<int>(buffer, "num_topk"));
    const auto input = hidden_states.narrow(0, 0, num_tokens);
    const auto ids = topk_ids.narrow(0, 0, num_tokens), weights = topk_weights.narrow(0, 0, num_tokens);
    if (kind == Precision::W4A16)
        bf16_mega_moe_pre_dispatch(input, ids, weights, buffer, num_tokens);
    else if (kind == Precision::W4A4)
        fp4_mega_moe_pre_dispatch(input, ids, weights, field<Tensor>(buffer, "input_token"),
                                 field<Tensor>(buffer, "input_sf"), field<Tensor>(buffer, "input_topk_idx"),
                                 field<Tensor>(buffer, "input_topk_weights"), num_tokens, 8, std::nullopt);
    else if (kind == Precision::FP8Channelwise)
        fp8_mega_moe_pre_dispatch(input, ids, weights, field<Tensor>(buffer, "input_token").view(torch::kUInt8),
                                 field<Tensor>(buffer, "input_scale"), field<Tensor>(buffer, "input_topk_idx"),
                                 field<Tensor>(buffer, "input_topk_weights"), num_tokens);
    else
        fp8_blockwise_mega_moe_pre_dispatch(buffer, input, ids, weights);
}

static std::string check_gemm1_options(Precision kind, const std::optional<std::string>& activation,
                                      const std::optional<std::string>& output_quantization) {
    TORCH_CHECK(!activation || !activation->empty(), "use activation=None for a raw projection");
    TORCH_CHECK(!output_quantization || !output_quantization->empty(),
                "use output_quantization=None for BF16 output");
    const std::string act = activation.value_or(""), quant = output_quantization.value_or("");
    bool supported = false;
    if (kind == Precision::W4A4)
        supported = (act.empty() || act == "swiglu" || act == "step" || act == "situ") &&
                    (quant.empty() || (quant == "mxfp4" && !act.empty()));
    else if (kind == Precision::W4A16)
        supported = act == "situ" && quant.empty();
    else if (kind == Precision::FP8Channelwise)
        supported = (act == "swiglu" || act == "swiglu_oai") && (quant.empty() || quant == "fp8");
    else
        supported = (act.empty() && quant.empty()) || (act == "swiglu" && quant == "fp8");
    TORCH_CHECK(supported, "unsupported precision/activation/output_quantization combination");
    return act == "swiglu" ? "silu_and_mul" : act == "step" ? "silu_and_mul_step" :
           act == "situ" ? "situ_and_mul" : act;
}

int mega_moe_dispatch_gemm1(const py::object& buffer, const Pair& rhs, const Tensor& out,
                            int rank, int num_tokens, const std::string& precision,
                            std::optional<std::string> activation,
                            std::optional<std::string> output_quantization, OptTensor out_scales,
                            OptPair rhs_replicas,
                            std::optional<float> activation_limit, std::optional<float> activation_alpha,
                            float situ_beta, float situ_linear_beta, OptTensor bf16_scratch, int gemm2_block_m,
                            OptTuning tuning) {
    const auto kind = parse_precision(precision);
    const auto act = check_gemm1_options(kind, activation, output_quantization);
    const bool quant = output_quantization.has_value();
    TORCH_CHECK(gemm2_block_m == 0 ||
                (kind == Precision::W4A4 &&
                 (gemm2_block_m == 16 || gemm2_block_m == 32 || gemm2_block_m == 64 ||
                  gemm2_block_m == 128 || gemm2_block_m == 256)),
                "gemm2_block_m requires W4A4 and BLOCK_M in {16,32,64,128,256}");
    TORCH_CHECK(field<int>(buffer, "num_max_tokens_per_rank") < (1 << 24) &&
                field<int>(buffer, "num_topk") <= 256,
                "packed routes require tokens < 2^24 and topk <= 256");
    TORCH_CHECK(quant == out_scales.has_value(), "out_scales is required exactly when output_quantization is set");
    TORCH_CHECK(!activation_limit || (std::isfinite(*activation_limit) && *activation_limit > 0 &&
                                      !act.empty() && act != "situ_and_mul"),
                "activation_limit requires a supported SwiGLU activation and a finite positive limit");
    TORCH_CHECK(!activation_alpha || act == "swiglu_oai", "activation_alpha is only supported by swiglu_oai");
    TORCH_CHECK(act == "situ_and_mul" || (situ_beta == 4.0f && situ_linear_beta == 25.0f),
                "SiTU parameters require activation='situ'");
    TORCH_CHECK(!bf16_scratch || (kind == Precision::FP8Channelwise && quant),
                "bf16_scratch is only used by quantized FP8 channelwise GEMM1");
    check_buffer_precision(buffer, kind);
    if (tuning) {
        TORCH_CHECK(rhs.first.dim() == 3);
        check_tuning(tuning, kind, field<int>(buffer, "num_ranks"), field<int>(buffer, "num_experts"),
                     field<int>(buffer, "num_topk"), field<int>(buffer, "hidden"), rhs.first.size(1) / 2,
                     num_tokens);
        TORCH_CHECK(tuning->workload.activation == activation.value_or(""),
                    "MegaMoE tuning activation does not match GEMM1");
        TORCH_CHECK(tuning->profile.empty() || quant == (kind != Precision::W4A16),
                    "measured MegaMoE profiles require quantized GEMM1 output except W4A16");
    }
    const auto gemm1_tuning = tuning ? tuning->gemm1 : mega_moe_config::Gemm1Tuning{};
    if (kind == Precision::W4A4) {
        check_tensor(rhs.second, torch::kUInt16);
        if (quant && !gemm2_block_m) {
            const int pool = field<int>(buffer, "num_pool");
            const int groups = field<int>(buffer, "num_experts") / field<int>(buffer, "num_ranks");
            // Use the same BF16 scatter selector as GEMM2. The tuning workload
            // carries the actual token count; without it both stages use capacity.
            const auto config = mega_moe_config::fp4_pipeline_scatter_config(
                pool, field<int>(buffer, "hidden"), field<int>(buffer, "intermediate_hidden") / 4,
                groups, std::nullopt, std::nullopt, tuning);
            gemm2_block_m = std::get<1>(config);
            TORCH_CHECK(gemm2_block_m == 16 || gemm2_block_m == 32 || gemm2_block_m == 64 ||
                        gemm2_block_m == 128 || gemm2_block_m == 256,
                        "W4A4 GEMM2 metadata requires BLOCK_M in {16,32,64,128,256}");
        }
        // The TC backend consumes the packed uint16 scales. Its legacy raw
        // scale pointer is unused; no second scale layout is part of this API.
        return fp4_mega_moe_dispatch_gemm1(
            buffer, rank, num_tokens, rhs.first, rhs.second.view(torch::kUInt8), rhs.second,
            activation_limit.value_or(0.0f), false, quant ? OptTensor(out) : std::nullopt, out_scales, quant,
            rhs_replicas ? OptTensor(rhs_replicas->first) : std::nullopt,
            rhs_replicas ? OptTensor(rhs_replicas->second) : std::nullopt, false, situ_beta, situ_linear_beta,
            false, false, 0, -1, false, true, true, false, -1, act, quant ? std::nullopt : OptTensor(out),
            gemm2_block_m, gemm1_tuning);
    }
    if (kind == Precision::W4A16) {
        TORCH_CHECK(!rhs_replicas, "W4A16 GEMM1 does not support split replica weights");
        const auto capacity = read_dispatch_config(field<py::dict>(buffer, "_capacity_cfg"));
        validate_rank(capacity, rank, num_tokens);
        const int epr = capacity.num_experts_per_rank, k = field<int>(buffer, "bf16_hidden"),
                  n = field<int>(buffer, "intermediate_hidden");
        TORCH_CHECK(k % 64 == 0 && n % 16 == 0 && capacity.token_bytes == k * 2);
        TORCH_CHECK(std::isfinite(situ_beta) && situ_beta > 0 && std::isfinite(situ_linear_beta) &&
                    situ_linear_beta > 0);
        check_tensor(rhs.first, torch::kUInt8);
        check_tensor(rhs.second, torch::kUInt8);
        check_shape(rhs.first, {epr, n, k / 2});
        check_shape(rhs.second, {epr, n / 64, k * 2});
        check_tensor(out, torch::kBFloat16);
        check_shape(out, {field<int>(buffer, "num_pool"), n / 2});
        const mega_moe_native::W4A16ActivationConfig activation_config{
            mega_moe_native::W4A16ActivationConfig::Type::SiTU, situ_beta, situ_linear_beta};
        auto config = launch_config(buffer, num_tokens);
        mega_moe_config::apply_gemm1_tuning(config, gemm1_tuning,
            field<std::optional<mega_moe_config::Overrides>>(buffer, "_config_override"));
        return mega_moe_native::w4a16_dispatch_gemm1_impl(
            config, field<Tensor>(buffer, "buf_ptrs"), rhs.first,
            rhs.second, out, rank, num_tokens, n, activation_config);
    }
    if (kind == Precision::FP8Channelwise) {
        TORCH_CHECK(!quant || bf16_scratch, "quantized FP8 channelwise GEMM1 requires bf16_scratch");
        const auto alpha = act == "swiglu_oai" ? std::optional<float>(activation_alpha.value_or(1.702f))
                                               : std::nullopt;
        return run_fp8_gemm1(buffer, rank, num_tokens, rhs.first, rhs.second,
                             quant ? *bf16_scratch : out, quant ? OptTensor(out) : std::nullopt,
                             out_scales, activation_limit, std::nullopt, std::nullopt,
                             rhs_replicas ? OptTensor(rhs_replicas->first) : std::nullopt,
                             rhs_replicas ? OptTensor(rhs_replicas->second) : std::nullopt,
                             alpha, std::nullopt, act, gemm1_tuning);
    }
    return fp8_blockwise_mega_moe_dispatch_gemm1(buffer, rank, num_tokens, rhs.first, rhs.second, out,
                                                out_scales, activation_limit, rhs_replicas,
                                                act.empty() ? std::nullopt : std::optional<std::string>(act));
}

using MoEInput = std::variant<Tensor, Pair>;

void mega_moe_gemm2_scatter(const MoEInput& lhs, const Pair& rhs, const Tensor& token_meta,
                           const Tensor& combine_buf, const Tensor& sym_offsets, const Tensor& m_rows,
                           const Tensor& valid_m_rows, int rank, int max_tokens, int topk,
                           const std::string& precision, OptTensor m_indices, OptTensor row_offsets,
                           const py::object& configs, OptPair rhs_replicas,
                           std::optional<std::string> output_quantization, OptTensor combine_scales,
                           OptTensor scale_offsets, int scale_block_n, std::optional<int> expected_m,
                           const std::string& kernel_type, bool validate_metadata,
                           std::optional<bool> check_packed_routes, std::optional<bool> rolling_pair_scatter,
                           std::optional<int> num_sms, OptTuning tuning) {
    const auto kind = parse_precision(precision);
    const bool fp8 = output_quantization.has_value();
    TORCH_CHECK(!fp8 || (*output_quantization == "fp8" && kind == Precision::W4A4),
                "FP8 scatter output is only supported by W4A4 GEMM2");
    check_tensor(combine_buf, fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16);
    TORCH_CHECK(fp8 || (!combine_scales && !scale_offsets && scale_block_n == 0),
                "combine scales require output_quantization='fp8'");
    TORCH_CHECK(combine_scales.has_value() == scale_offsets.has_value(),
                "combine_scales and scale_offsets must be supplied together");
    TORCH_CHECK(combine_scales ? scale_block_n >= 64 && scale_block_n % 8 == 0 : scale_block_n == 0,
                "combine_scales requires scale_block_n >= 64 aligned to 8; otherwise it must be zero");
    TORCH_CHECK(!expected_m || kind == Precision::W4A4, "expected_m is only supported by W4A4 GEMM2");
    TORCH_CHECK(kind == Precision::FP8Channelwise ||
                (!m_indices && kernel_type == "Default" && validate_metadata &&
                 !check_packed_routes && !rolling_pair_scatter && !num_sms),
                "channelwise scheduling options require precision='fp8_channelwise'");
    TORCH_CHECK(!row_offsets || kind == Precision::W4A4 || kind == Precision::FP8Blockwise,
                "row_offsets is only supported by W4A4 or FP8 blockwise GEMM2");
    TORCH_CHECK(rhs.first.dim() == 3);
    const int hidden = rhs.first.size(1), world = sym_offsets.numel();
    TORCH_CHECK(world > 0 && world <= 32 && rank >= 0 && rank < world,
                "the unified pipeline supports at most 32 ranks");
    if (tuning) {
        const int groups = rhs.first.size(0) + (rhs_replicas ? rhs_replicas->first.size(0) : 0);
        const int intermediate = rhs.first.size(2) *
            ((kind == Precision::W4A4 || kind == Precision::W4A16) ? 2 : 1);
        check_tuning(tuning, kind, world, groups * world, topk, hidden, intermediate);
        TORCH_CHECK(tuning->workload.tokens <= max_tokens, "tuning workload exceeds buffer capacity");
        TORCH_CHECK(!fp8 || tuning->profile.empty(), "measured MegaMoE profiles require BF16 scatter");
    }
    if (kind == Precision::W4A16) {
        TORCH_CHECK(std::holds_alternative<Tensor>(lhs), "W4A16 lhs must be a BF16 tensor");
        return m_grouped_gemm_w4a16_mxfp4_with_scatter(
            std::get<Tensor>(lhs), rhs, token_meta, combine_buf, sym_offsets, valid_m_rows, m_rows,
            rank, max_tokens, topk, configs.cast<std::optional<PPUW4ConfigTuple>>(), rhs_replicas);
    }
    TORCH_CHECK(std::holds_alternative<Pair>(lhs), "quantized lhs must be a (tensor, scales) pair");
    const auto& input = std::get<Pair>(lhs);
    if (kind == Precision::W4A4) {
        TORCH_CHECK(row_offsets || !configs.is_none(),
                    "automatic W4A4 scatter configuration requires metadata.src_m_offsets as row_offsets");
        const auto config = mega_moe_config::fp4_pipeline_scatter_config(
            input.first.size(0), hidden, rhs.first.size(2),
            rhs.first.size(0) + (rhs_replicas ? rhs_replicas->first.size(0) : 0),
            configs.cast<std::optional<Config>>(), expected_m, tuning);
        return m_grouped_gemm_fp4_with_scatter(
            input, rhs, std::nullopt, token_meta, fp8 ? combine_buf.view(torch::kUInt8) : combine_buf,
            sym_offsets, rank, world, max_tokens, hidden, m_rows, config,
            topk, valid_m_rows, combine_scales, scale_offsets, scale_block_n, fp8, row_offsets,
            expected_m, rhs_replicas);
    }
    if (kind == Precision::FP8Channelwise) {
        TORCH_CHECK(m_indices, "FP8 channelwise GEMM2 requires m_indices");
        return m_grouped_gemm_fp8_channelwise_with_scatter(
            input, rhs, *m_indices, token_meta, combine_buf, sym_offsets, rank, world, max_tokens, hidden,
            m_rows, configs.cast<std::optional<Config>>(), topk, valid_m_rows, kernel_type,
            validate_metadata, check_packed_routes, rolling_pair_scatter, num_sms, rhs_replicas,
            tuning ? tuning->gemm2 : mega_moe_config::Gemm2Tuning{});
    }
    TORCH_CHECK(row_offsets && configs.is_none(),
                "FP8 blockwise GEMM2 requires row_offsets and uses its built-in tile configuration");
    return m_grouped_gemm_fp8_blockwise_with_scatter(input, rhs, std::nullopt, m_rows, valid_m_rows,
                                                   *row_offsets, token_meta, combine_buf, sym_offsets,
                                                   max_tokens, topk, rank, rhs_replicas);
}

// Python bindings.
void register_apis(py::module_& m) {
    py::class_<TuningConfig>(m, "MegaMoETuningConfig")
        .def_readonly("profile", &TuningConfig::profile)
        .def_property_readonly("num_tokens", [](const TuningConfig& c) { return c.workload.tokens; })
        .def_property_readonly("gemm1_tile", [](const TuningConfig& c) { return c.gemm1.tile; })
        .def_property_readonly("gemm1_block_m", [](const TuningConfig& c) { return c.gemm1.block_m; })
        .def_property_readonly("pull_sms", [](const TuningConfig& c) { return c.gemm1.pull_sms; })
        .def_property_readonly("pull_mode", [](const TuningConfig& c) -> std::optional<std::string> {
            return c.gemm1.pull_mode ? std::optional<std::string>(mega_moe_config::pull_mode_name(*c.gemm1.pull_mode))
                                    : std::nullopt;
        })
        .def_property_readonly("separate_quant", [](const TuningConfig& c) { return c.gemm1.separate_quant; })
        .def_property_readonly("gemm2_tile", [](const TuningConfig& c) { return c.gemm2.tile; })
        .def_property_readonly("reduce_chunked", [](const TuningConfig& c) { return c.combine.chunked; });
    m.def("get_mega_moe_tuning_config", &get_mega_moe_tuning_config,
          py::arg("num_ranks"), py::arg("num_experts"), py::arg("num_topk"), py::arg("hidden"),
          py::arg("intermediate_hidden"), py::arg("num_tokens"), py::kw_only(),
          py::arg("precision"), py::arg("activation") = "swiglu");
    m.def("mega_moe_pre_dispatch", &mega_moe_pre_dispatch, py::arg("hidden_states"), py::arg("topk_ids"),
          py::arg("topk_weights"), py::arg("buffer"), py::arg("num_tokens"), py::kw_only(), py::arg("precision"));
    m.def("mega_moe_dispatch_gemm1", &mega_moe_dispatch_gemm1, py::arg("buffer"), py::arg("rhs"),
          py::arg("out"), py::arg("rank"), py::arg("num_tokens"), py::kw_only(), py::arg("precision"),
          py::arg("activation") = "swiglu", py::arg("output_quantization") = py::none(),
          py::arg("out_scales") = py::none(),
          py::arg("rhs_replicas") = py::none(), py::arg("activation_limit") = py::none(),
          py::arg("activation_alpha") = py::none(), py::arg("situ_beta") = 4.0,
          py::arg("situ_linear_beta") = 25.0, py::arg("bf16_scratch") = py::none(),
          py::arg("gemm2_block_m") = 0, py::arg("tuning") = py::none());
    m.def("mega_moe_gemm2_scatter", &mega_moe_gemm2_scatter, py::arg("lhs"), py::arg("rhs"),
          py::arg("token_meta"), py::arg("combine_buf"), py::arg("sym_offsets"), py::arg("m_rows"),
          py::arg("valid_m_rows"), py::arg("rank"), py::arg("max_tokens"), py::arg("topk"), py::kw_only(),
          py::arg("precision"), py::arg("m_indices") = py::none(), py::arg("row_offsets") = py::none(),
          py::arg("configs") = py::none(), py::arg("rhs_replicas") = py::none(),
          py::arg("output_quantization") = py::none(), py::arg("combine_scales") = py::none(),
          py::arg("scale_offsets") = py::none(), py::arg("scale_block_n") = 0, py::arg("expected_m") = py::none(),
          py::arg("kernel_type") = "Default", py::arg("validate_metadata") = true,
          py::arg("check_packed_routes") = py::none(), py::arg("rolling_pair_scatter") = py::none(),
          py::arg("num_sms") = py::none(), py::arg("tuning") = py::none());
    m.def("mega_moe_combine", &mega_moe_combine, py::arg("combine_buf"), py::arg("weights"),
          py::arg("num_tokens"), py::arg("hidden"), py::arg("topk"), py::arg("max_tokens"),
          py::arg("flag_ptrs"), py::arg("rank"), py::arg("world_size"), py::arg("done_flag"),
          py::kw_only(), py::arg("combine_scales") = py::none(), py::arg("scale_block_n") = 0,
          py::arg("tuning") = py::none());
    m.def("_ppu_mega_moe_layout", &symmetric_layout, py::arg("config"), py::arg("fp4"));
    m.def("_select_glm_channelwise_scatter_config", &mega_moe_config::channel_scatter_config,
          py::arg("expected_m"), py::arg("n"), py::arg("k"), py::arg("num_groups"), py::arg("num_sms"));
    m.def("_mega_moe_dispatch_config", &dispatch_config, py::arg("num_ranks"),
          py::arg("num_experts"), py::arg("num_topk"), py::arg("hidden"), py::arg("num_max_tokens_per_rank"),
          py::kw_only(), py::arg("precision"), py::arg("intermediate_hidden") = py::none(),
          py::arg("block_m") = py::none(),
          py::arg("gemm1_block_n") = py::none(), py::arg("gemm1_block_k") = py::none(),
          py::arg("overrides") = py::none());
    m.def("_fp8_blockwise_mega_moe_dispatch", &fp8_blockwise_mega_moe_dispatch, py::arg("buffer"),
          py::arg("rank"), py::arg("num_tokens"));
    m.def("interleave_fp8_blockwise_mega_moe_weights", &interleave_fp8_blockwise_mega_moe_weights,
          py::arg("weight"), py::arg("scales"));
    m.def("preprocess_w4a16_mxfp4", &preprocess_w4a16_mxfp4, py::arg("rhs_"));
}

} // namespace deep_gemm::mega_moe
