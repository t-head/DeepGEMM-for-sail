#pragma once
#include <algorithm>
#include <limits>
#include <type_traits>
#include <torch/extension.h>
#include <deep_gemm/activation/arguments.hpp>
#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"

namespace deep_gemm {
// Masked wrappers use the same row implementation as dense wrappers. expected_m
// controls row_slots, while a grid-stride loop always covers masked_m[expert].
struct ActivationSoftcapArgs {
    ActivationMode mode = ActivationMode::SILU;
    float beta = 1.0f;
    float linear_beta = 0.0f;
};
struct ActivationMaskArgs {
    const int32_t* masked_m = nullptr;
    int rows_per_batch = 0;
    int row_slots = 0;
    bool enabled = false;
    bool route_masked = false;
    bool masked() const { return enabled; }
};
struct ActivationLaunchArgs {
    int n;
    bool reordered;
    ActivationMaskArgs mask;
    ActivationSoftcapArgs softcap;
    LaunchArgs launch_args;
    ActivationParams params;
    QuantMode mode = QuantMode::BF16;
};
class PPUActivationQuantRuntime final : public LaunchRuntime<PPUActivationQuantRuntime> {
public:
    using Args = ActivationLaunchArgs;
    static const char* get_kernel_name(const Args& args) {
        return args.mode == QuantMode::BF16
            ? ((args.softcap.mode == ActivationMode::SITU) ? "situ_and_mul" : "silu_and_mul")
            : args.mode == QuantMode::FP8_CHANNEL
                ? ((args.softcap.mode == ActivationMode::SITU) ? "situ_and_mul_quant_fp8" : "silu_and_mul_quant_fp8")
                : args.mode == QuantMode::FP8_BLOCK
                    ? ((args.softcap.mode == ActivationMode::SITU) ? "situ_and_mul_quant_fp8_block" : "silu_and_mul_quant_fp8_block")
                    : ((args.softcap.mode == ActivationMode::SITU) ? "situ_and_mul_quant_mx" : "silu_and_mul_quant_mx");
    }
    static std::string generate_impl(const Args& args) {
        const char* mode = args.mode == QuantMode::BF16 ? "BF16"
                         : args.mode == QuantMode::FP8_CHANNEL ? "FP8_CHANNEL"
                         : args.mode == QuantMode::MXFP4 ? "MXFP4"
                         : args.mode == QuantMode::MXFP8 ? "MXFP8" : "FP8_BLOCK";
        const bool vector_loads = args.n % 256 == 0 && reinterpret_cast<uintptr_t>(args.params.input) % 16 == 0;
        return fmt::format(R"(
#include <deep_gemm/activation/activation_quant.cuh>
static_assert(sizeof(deep_gemm::ActivationParams) == {}, "activation argument size mismatch");
extern "C" __global__ __launch_bounds__({})
void {}(const void* input, void* output, void* scales, const int32_t* masked_m, int rows_per_batch, int row_slots, float limit, float weight_inv_scale, float beta, float linear_beta) {{
    deep_gemm::ActivationParams params{{input, output, scales, masked_m, rows_per_batch, row_slots, limit, weight_inv_scale, beta, linear_beta}};
    deep_gemm::activation_quant<{}, deep_gemm::QuantMode::{}, {}, {}, {}, {}, {}, {}, {}>(params);
}}
)", sizeof(ActivationParams), args.mode == QuantMode::FP8_CHANNEL ? activation_fp8_channel_threads(args.n) : 32, get_kernel_name(args),
            args.n, mode, (args.softcap.mode == ActivationMode::SITU) ? "true" : "false", args.mask.masked() ? "true" : "false",
            args.reordered ? "true" : "false", args.params.limit > 0 ? "true" : "false",
            vector_loads ? "true" : "false", args.mask.route_masked ? "true" : "false",
            args.softcap.mode == ActivationMode::SILU_STEP ? "true" : "false");
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        auto& p = args.params;
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, p.input, p.output, p.scales, p.masked_m, p.rows_per_batch, p.row_slots, p.limit, p.weight_inv_scale, p.beta, p.linear_beta));
    }
};
namespace activation {
inline ActivationMaskArgs make_activation_mask(const torch::Tensor& input,
        const torch::Tensor& masked_m, const std::optional<int>& expected_m) {
    const int64_t capacity = input.size(1);
    // Cap grid size by the hint, then iterate to each device-side
    // valid count. The hint must never truncate computation; no count readback.
    const int hint = expected_m.value_or(static_cast<int>(capacity));
    const int slots = std::min<int64_t>(capacity,
        std::min(std::max<int64_t>(1, 2048 / std::max<int64_t>(1, input.size(0))), hint > 0 ? hint : capacity));
    return {masked_m.data_ptr<int32_t>(), static_cast<int>(capacity), slots, true};
}
inline ActivationLaunchArgs make_activation_args(int n, const torch::Tensor& input,
        const torch::Tensor& output, void* scales, float limit, const ActivationMaskArgs& mask,
        const ActivationSoftcapArgs& softcap, LaunchArgs launch, int rows_per_batch = 0,
        float inverse = 1.0f) {
    ActivationParams params{input.data_ptr(), output.data_ptr(), scales, mask.masked_m,
        rows_per_batch ? rows_per_batch : mask.rows_per_batch,
        mask.row_slots, limit, inverse, softcap.beta, softcap.linear_beta};
    return {n, false, mask, softcap, launch, params};
}
static int64_t launch_rows(const torch::Tensor& input, int64_t rows, const ActivationMaskArgs& mask) {
    return mask.masked() ? input.size(0) * mask.row_slots : rows;
}

template<bool FP8Block>
static std::pair<torch::Tensor, torch::Tensor> run_activation_tiled(
        const torch::Tensor& input, int n, int64_t rows, float limit,
        const ActivationMaskArgs& mask = {}, const ActivationSoftcapArgs& softcap = {}) {
    const int rows_per_batch = input.size(-2);
    const int64_t batches = rows_per_batch ? rows / rows_per_batch : 0;
    const int slots = mask.masked() ? mask.row_slots : std::min<int64_t>(rows_per_batch,
        std::max<int64_t>(1, 2048 / std::max<int64_t>(1, batches)));
    const int64_t blocks = batches * ((n + 255) / 256);
    TORCH_CHECK(blocks <= std::numeric_limits<int>::max(), "activation grid exceeds indexing limits");
    auto shape = input.sizes().vec();
    shape.back() = n;
    auto output = at::empty(shape, input.options().dtype(FP8Block ? torch::kFloat8_e4m3fn : torch::kBFloat16));
    torch::Tensor scales;
    if constexpr (FP8Block) {
        if (mask.masked()) {
            const int64_t aligned_rows = (int64_t(rows_per_batch) + 3) / 4 * 4;
            scales = at::empty({input.size(0), n / 128, aligned_rows}, input.options().dtype(torch::kFloat32))
                         .transpose(1, 2).narrow(1, 0, rows_per_batch);
        } else {
            shape.back() = n / 128;
            scales = at::empty(shape, input.options().dtype(torch::kFloat32));
        }
    }
    if (rows) {
        auto args = make_activation_args(n, input, output, FP8Block ? scales.data_ptr() : nullptr,
            limit, mask, softcap, {dim3(blocks, slots), dim3(32), 0}, rows_per_batch);
        args.mode = FP8Block ? QuantMode::FP8_BLOCK : QuantMode::BF16;
        args.params.row_slots = slots;
        const auto& runtime = compiler->build(PPUActivationQuantRuntime::get_kernel_name(args), PPUActivationQuantRuntime::generate(args));
        PPUActivationQuantRuntime::launch(runtime, args);
    }
    return {output, scales};
}

static torch::Tensor run_activation(const torch::Tensor& input, int n, int64_t rows, float limit,
        const ActivationMaskArgs& mask = {}, const ActivationSoftcapArgs& softcap = {}) {
    return run_activation_tiled<false>(input, n, rows, limit, mask, softcap).first;
}

// Output row scales include GEMM2's inverse weight scale; quantization codes do not.
static std::pair<torch::Tensor, torch::Tensor> run_activation_fp8_channel(
        const torch::Tensor& input, int64_t n, int64_t rows, float limit, float weight_inv_scale, const ActivationMaskArgs& mask = {},
        const ActivationSoftcapArgs& softcap = {}) {
    auto shape = input.sizes().vec();
    shape.back() = n;
    auto output = at::empty(shape, input.options().dtype(torch::kFloat8_e4m3fn));
    shape.back() = 1;
    auto scales = at::empty(shape, input.options().dtype(torch::kFloat32));
    if (rows) {
        auto args = make_activation_args(n, input, output, scales.data_ptr(), limit, mask, softcap,
            {dim3(launch_rows(input, rows, mask)), dim3(activation_fp8_channel_threads(n)), 0}, 0, weight_inv_scale);
        args.mode = QuantMode::FP8_CHANNEL;
        const auto& runtime = compiler->build(PPUActivationQuantRuntime::get_kernel_name(args), PPUActivationQuantRuntime::generate(args));
        PPUActivationQuantRuntime::launch(runtime, args);
    }
    return {output, scales};
}

template<bool FP4, bool Reordered = false>
static std::pair<torch::Tensor, torch::Tensor> run_activation_mx(
        const torch::Tensor& input, int64_t n, int64_t rows, float limit, const ActivationMaskArgs& mask = {},
        const ActivationSoftcapArgs& softcap = {}) {
    const int groups = (n + 31) / 32;
    const int pairs = (groups + 1) / 2;
    // MXFP4/MXFP8: one warp per 256 columns, persistent token slots for async overlap.
    const int rows_per_batch = input.size(-2);
    const int64_t batches = rows_per_batch ? rows / rows_per_batch : 0;
    const int slots = mask.masked() ? mask.row_slots : std::min<int64_t>(rows_per_batch,
        std::max<int64_t>(1, 2048 / std::max<int64_t>(1, batches)));
    const int64_t blocks = batches * ((n + 255) / 256);
    TORCH_CHECK(blocks <= std::numeric_limits<int>::max(), "MX activation grid exceeds indexing limits");
    auto shape = input.sizes().vec();
    shape.back() = FP4 ? n / 2 : n;
    auto output = at::empty(shape, input.options().dtype(FP4 ? torch::kUInt8 : torch::kFloat8_e4m3fn));
    torch::Tensor scales;
    if constexpr (Reordered) {
        if (input.dim() == 2)
            scales = at::empty({pairs, rows_per_batch}, input.options().dtype(torch::kUInt16)).t();
        else
            scales = at::empty({input.size(0), pairs, rows_per_batch}, input.options().dtype(torch::kUInt16))
                         .permute({0, 2, 1});
    } else {
        shape.back() = groups;
        scales = at::empty(shape, input.options().dtype(torch::kUInt8));
    }
    if (rows) {
        auto args = make_activation_args(n, input, output, scales.data_ptr(), limit, mask, softcap,
            {dim3(blocks, slots), dim3(32), 0}, rows_per_batch);
        args.mode = FP4 ? QuantMode::MXFP4 : QuantMode::MXFP8;
        args.reordered = Reordered;
        args.params.row_slots = slots;
        const auto& runtime = compiler->build(PPUActivationQuantRuntime::get_kernel_name(args), PPUActivationQuantRuntime::generate(args));
        PPUActivationQuantRuntime::launch(runtime, args);
    }
    return {output, scales};
}
static std::pair<torch::Tensor, torch::Tensor> run_activation_mxfp4(
        const torch::Tensor& input, int64_t n, int64_t rows, float limit, bool reorder_scale, const ActivationMaskArgs& mask = {},
        const ActivationSoftcapArgs& softcap = {}) {
    return reorder_scale ? run_activation_mx<true, true>(input, n, rows, limit, mask, softcap)
                         : run_activation_mx<true>(input, n, rows, limit, mask, softcap);
}

} // namespace activation
} // namespace deep_gemm
