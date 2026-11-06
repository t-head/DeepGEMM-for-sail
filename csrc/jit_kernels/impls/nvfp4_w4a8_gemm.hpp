#pragma once
#include <c10/core/DeviceGuard.h>
#include <limits>
#include <variant>
#include "m_grouped_fp8_gemm.hpp"
#include "fused_moe_gemm.hpp"
#include "nvfp4_decode.hpp"
#include "../heuristics/common_nvfp4_w4a8.hpp"
#include "m_grouped_w4a16_gemm.hpp"

namespace deep_gemm::nvfp4::detail {

template <typename Scheduler>
struct Arguments {
    const void* ptr_a;
    const void* ptr_b;
    const void* ptr_scale;
    void* ptr_d;
    int num_token;
    int topk;
    const int32_t* sorted_token_ids;
    Scheduler scheduler;
    const float* ptr_a_scale;
    const float* ptr_weight_global_scale;
    int weight_global_scale_stride;
    const float* ptr_b_scale;
};

struct LaunchInfo {
    Config tile;
    int n, k, groups, group_size, a_scale_group, b_scale_block;
    bool has_a_scale, has_weight_global_scale, has_b_scale;
    GemmType type;
    std::string name;
    int smem, threads;
};

struct Routing {
    int topk = 0;
    torch::Tensor expert_ids_and_cumsum;
    torch::Tensor sorted_token_ids;
    torch::Tensor aligned_num_m_blocks;
};

template <typename Scheduler>
class Runtime final : public LaunchRuntime<Runtime<Scheduler>> {
public:
    struct Args {
        LaunchInfo info;
        LaunchArgs launch_args;
        Arguments<Scheduler> kernel_params;
    };
    static std::string generate_impl(const Args& args) {
        const auto& i = args.info;
        const auto& c = i.tile;
        return fmt::format(R"(
#define W4A16_HGRTC
#include <deep_gemm/impls/nvfp4_w4a8_gemm.cuh>
using Kernel = cutlass::gemm::kernel::W4A8GEMM<{}, {}, {}, {}, {}, {}, {}, {}, {}, {},
    deep_gemm::GemmType::{}, {}, {}, {}, {}, {}, {}, {}>;
static_assert(sizeof(Kernel::SharedStorage) == {}, "W4A8 shared memory mismatch");
static_assert(Kernel::MaxThreadsPerBlock == {}, "W4A8 thread count mismatch");
static_assert(sizeof(Kernel::Params) == {}, "W4A8 Params ABI mismatch");
static_assert(offsetof(Kernel::Params, scheduler) == {}, "W4A8 scheduler ABI mismatch");
static_assert(offsetof(Kernel::Params, ptr_a_scale) == {}, "W4A8 scale ABI mismatch");
static_assert(offsetof(Kernel::Params, ptr_b_scale) == {}, "W4A8 channel scale ABI mismatch");
extern "C" __global__ __launch_bounds__(Kernel::MaxThreadsPerBlock, Kernel::MinBlocksPerMultiprocessor)
void {}(Kernel::Params params) {{
    extern __shared__ char smem[];
    Kernel{{}}(params, smem);
}}
)", i.n, i.k, c.block_m, c.block_n, c.block_k, c.warp_m, c.warp_n, c.warp_k,
            i.groups, c.num_stages, cpp_gemm_type_name(i.type), i.group_size, c.n_expand,
            i.a_scale_group, i.has_a_scale, i.has_weight_global_scale, i.has_b_scale,
            i.b_scale_block,
            i.smem, i.threads,
            sizeof(Arguments<Scheduler>), offsetof(Arguments<Scheduler>, scheduler),
            offsetof(Arguments<Scheduler>, ptr_a_scale), offsetof(Arguments<Scheduler>, ptr_b_scale), i.name);
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

template <typename Scheduler>
static void launch(const LaunchInfo& info, const Arguments<Scheduler>& params) {
    using R = Runtime<Scheduler>;
    typename R::Args args{info, {dim3(info.tile.num_sms), dim3(info.threads), info.smem}, params};
    const auto& code = R::generate(args);
    const auto& compiled = compiler->build(info.name, code);
    construct_launch_config(compiled->kernel, current_stream(), info.smem,
                            args.launch_args.grid_dim, args.launch_args.block_dim);
    int occupancy = 0;
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, compiled->kernel, info.threads, info.smem));
    TORCH_CHECK(occupancy > 0, "W4A8 tile exceeds device resources");
    args.launch_args.grid_dim = dim3(info.tile.num_sms < 20 ? info.tile.num_sms
        : info.tile.num_sms * std::min(8, occupancy));
    R::launch(compiled, args);
}

static void run(GemmType type, int expected_m, const torch::Tensor& a,
                 const std::pair<torch::Tensor, torch::Tensor>& rhs, const torch::Tensor& out,
                 const std::optional<ConfigTuple>& requested,
                 const torch::Tensor& rows,
                 const std::optional<torch::Tensor>& a_scales, int a_scale_group,
                 const std::optional<torch::Tensor>& global_scale,
                 const std::optional<torch::Tensor>& b_scales, int b_scale_block,
                 const Routing& routing = {}) {
    const bool masked = type == GemmType::MGroupedMasked;
    const bool fused = type == GemmType::MGroupedFused;
    const auto& [b, s] = rhs;
    const int m = a.size(masked ? 1 : 0), k = a.size(-1), groups = b.size(0), n = b.size(2) / 4;
    const int group_size = 16;
    if (m == 0) return;
    const auto c = config(type, expected_m, n, k, requested, a_scale_group, b_scale_block);
    TORCH_CHECK(!fused || routing.sorted_token_ids.size(1) == c.block_m,
                "Routing metadata block_m must match the GEMM config");
    const int threads = (c.block_m/c.warp_m) * (c.block_n/c.warp_n) * (c.block_k/c.warp_k) * 32;
    TORCH_CHECK(threads <= 1024, "W4A8 tile has too many threads");
    const LaunchInfo info{c, n, k, groups, group_size, a_scale_group, b_scale_block,
        bool(a_scales), bool(global_scale), bool(b_scales), type,
        "w4a8_nvfp4",
        smem_size(c, group_size, b_scale_block, a_scale_group, k, fused), threads};
    auto fill = [&](auto& p) {
        p.ptr_a = a.data_ptr(); p.ptr_b = b.data_ptr(); p.ptr_scale = s.data_ptr(); p.ptr_d = out.data_ptr();
        p.num_token = m;
        p.ptr_a_scale = a_scales ? a_scales->data_ptr<float>() : nullptr;
        p.ptr_weight_global_scale = global_scale ? global_scale->data_ptr<float>() : nullptr;
        p.weight_global_scale_stride = global_scale && global_scale->numel() > 1 ? 1 : 0;
        p.ptr_b_scale = b_scales ? b_scales->data_ptr<float>() : nullptr;
    };
    if (fused) {
        Arguments<FusedSchedulerArguments> p{}; fill(p);
        p.topk = routing.topk;
        p.sorted_token_ids = routing.sorted_token_ids.data_ptr<int32_t>();
        p.scheduler = {routing.aligned_num_m_blocks.data_ptr<int32_t>(),
                       routing.expert_ids_and_cumsum.data_ptr<int32_t>()};
        launch(info,p);
    } else {
        auto layout = type == GemmType::Normal ? nullptr : rows.data_ptr<int32_t>();
        torch::Tensor scratch;
        if (type == GemmType::MGroupedNoPad && groups >= 128) {
            const int needed = (groups + ceil_div(m, c.block_m) + 1) * 4;
            scratch = at::empty({needed}, rows.options());
            const auto block_size = std::max(32, int(next_power_of_two(uint32_t(groups))));
            TORCH_CHECK(block_size <= 1024, "W4A8 supports at most 1024 no-pad experts");
            auto args = ComputeBlockInfoKernelRuntime::Args{
                .launch_attr_args = {reinterpret_cast<const uint32_t*>(layout), uint32_t(groups),
                    reinterpret_cast<uint32_t*>(scratch.data_ptr<int32_t>())},
                .launch_args = {1, block_size, 0}};
            const auto& code = ComputeBlockInfoKernelRuntime::generate(c.block_m);
            const auto& rt = compiler->build("computeBlockInfoKernel", code);
            ComputeBlockInfoKernelRuntime::launch(rt,args);
            layout = scratch.data_ptr<int32_t>();
        }
        Arguments<TileSchedulerArguments> p{}; fill(p);
        p.scheduler = TileSchedulerArguments(m,layout);
        launch(info,p);
    }
}

} // namespace deep_gemm::nvfp4::detail

namespace deep_gemm::nvfp4 {

using OptionalTensor = std::optional<torch::Tensor>;
using Activation = std::pair<torch::Tensor, OptionalTensor>;
using Weight = std::pair<torch::Tensor, torch::Tensor>;
using ConfigTuple = deep_gemm::nvfp4::detail::ConfigTuple;
using ExecutionConfig = std::variant<ConfigTuple, deep_gemm::ConfigTuple, FusedCommonConfigTuple>;

struct Options {
    std::optional<ExecutionConfig> configs;
    int lhs_scale_group_size;
    OptionalTensor weight_global_scale;
    OptionalTensor rhs_scales;
    int rhs_scale_block_size;
    std::string execution;
    OptionalTensor fp8_workspace;
};

static void check_tensor(const torch::Tensor& tensor, const torch::Tensor& reference,
                         bool contiguous = true) {
    TORCH_CHECK(tensor.defined() && tensor.device() == reference.device() &&
                    (!contiguous || tensor.is_contiguous()),
                "NVFP4 tensors must share the activation device and required contiguous layout");
}

static void check_indices(const torch::Tensor& tensor, const torch::Tensor& reference) {
    check_tensor(tensor, reference);
    TORCH_CHECK(tensor.scalar_type() == torch::kInt32, "NVFP4 routing tensors must be int32");
}

static void dequantize_nvfp4_fp8_impl(const Weight& rhs, const torch::Tensor& out) {
    TORCH_CHECK(out.is_cuda(), "NVFP4 decoding requires PPU tensors");
    c10::DeviceGuard guard(out.device());
    TORCH_CHECK(is_ppu1v5_device(), "NVFP4 decoding requires PPU1.5");
    const auto& [packed, scales] = rhs;
    TORCH_CHECK((out.dim() == 2 || out.dim() == 3) && packed.dim() == out.dim() &&
                    scales.dim() == out.dim(), "NVFP4 decoding expects dense or grouped NT weights");
    const int64_t n = out.size(-2), k = out.size(-1);
    auto packed_shape = out.sizes().vec(), scale_shape = out.sizes().vec();
    packed_shape.back() = k / 8;
    scale_shape.back() = k / 16;
    TORCH_CHECK(out.numel() > 0 && n % 64 == 0 && k % 32 == 0 &&
                    packed.sizes() == packed_shape && scales.sizes() == scale_shape,
                "NVFP4 decoding requires [E,N,K/8] int32 weights and [E,N,K/16] scales; "
                "N must be divisible by 64 and K by 32 (omit E for dense)");
    TORCH_CHECK(packed.scalar_type() == torch::kInt32 && scales.scalar_type() == torch::kUInt8 &&
                    out.scalar_type() == torch::kFloat8_e4m3fn,
                "NVFP4 decoding expects prepared int32 weights, uint8 scales and E4M3FN output");
    for (const auto& tensor : {packed, scales, out}) check_tensor(tensor, out);
    TORCH_CHECK(out.numel() / 16 <= std::numeric_limits<int32_t>::max(),
                "NVFP4 decoding supports at most INT32_MAX K16 records");
    const auto begin = reinterpret_cast<uintptr_t>(out.data_ptr());
    const auto end = begin + out.nbytes();
    TORCH_CHECK(begin % 16 == 0 && reinterpret_cast<uintptr_t>(packed.data_ptr()) % 8 == 0,
                "NVFP4 decoding requires 16-byte aligned output and 8-byte aligned packed weights");
    for (const auto& input : {packed, scales}) {
        const auto input_begin = reinterpret_cast<uintptr_t>(input.data_ptr());
        TORCH_CHECK(end <= input_begin || input_begin + input.nbytes() <= begin,
                    "NVFP4 decode output must not overlap its inputs");
    }
    detail::decode_fp4_fp8(packed, scales, out);
}

static void check_workspace_overlap(const torch::Tensor& workspace, const torch::Tensor& tensor) {
    if (!tensor.defined() || tensor.numel() == 0) return;
    int64_t span = 1;
    for (int i = 0; i < tensor.dim(); ++i) span += (tensor.size(i) - 1) * tensor.stride(i);
    const auto begin = reinterpret_cast<uintptr_t>(workspace.data_ptr());
    const auto input_begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    TORCH_CHECK(begin + workspace.nbytes() <= input_begin ||
                    input_begin + span * tensor.element_size() <= begin,
                "fp8_workspace must not overlap GEMM inputs, output or routing metadata");
}

static void run_dequantized(GemmType kind, const Activation& lhs, const Weight& rhs,
                            const torch::Tensor& out, const torch::Tensor& rows,
                            const torch::Tensor& indices, int expected_m,
                            const detail::Routing& routing, const Options& options) {
    const auto& [a, activation_scale] = lhs;
    TORCH_CHECK(activation_scale && options.rhs_scales,
                "Dequantize execution requires explicit FP32 activation and weight scales");
    TORCH_CHECK(options.lhs_scale_group_size == 0 || options.lhs_scale_group_size == 128,
                "Dequantize execution supports per-token or K128 activation scales");
    TORCH_CHECK(options.rhs_scale_block_size == 128,
                "NVFP4 dequantize execution requires outer blockwise FP8 scales (128)");
    std::optional<deep_gemm::ConfigTuple> fp8_config;
    std::optional<FusedCommonConfigTuple> routing_config;
    if (options.configs) {
        if (kind == GemmType::MGroupedFused) {
            const auto* config = std::get_if<FusedCommonConfigTuple>(&*options.configs);
            TORCH_CHECK(config, "Dequantize routing requires a seven-entry FP8 config");
            routing_config = *config;
        } else {
            const auto* config = std::get_if<deep_gemm::ConfigTuple>(&*options.configs);
            TORCH_CHECK(config, "Dequantize execution requires an FP8 config, not a W4A8 config");
            fp8_config = *config;
        }
    }
    TORCH_CHECK(kind != GemmType::MGroupedFused || routing_config,
                "Fused routing requires an explicit FP8 config matching the routing metadata");
    TORCH_CHECK(!routing_config || routing.sorted_token_ids.size(1) == std::get<1>(*routing_config),
                "Routing metadata block_m must match the GEMM config");
    const bool dense = kind == GemmType::Normal;
    const int64_t n = out.size(-1), k = a.size(-1), groups = dense ? 1 : rhs.first.size(0);
    auto workspace = *options.fp8_workspace;
    auto weight = rhs;
    auto b_scale = *options.rhs_scales;
    if (dense) {
        if (workspace.dim() == 3 && workspace.size(0) == 1) workspace = workspace.squeeze(0);
        weight.first = weight.first.squeeze(0);
        weight.second = weight.second.squeeze(0);
        if (b_scale.dim() == 3 && b_scale.size(0) == 1) b_scale = b_scale.squeeze(0);
    }
    check_tensor(workspace, a);
    const auto shape = dense ? std::vector<int64_t>{n, k} : std::vector<int64_t>{groups, n, k};
    TORCH_CHECK(workspace.sizes() == shape && workspace.scalar_type() == torch::kFloat8_e4m3fn,
                "fp8_workspace must be contiguous E4M3FN [N,K] or [E,N,K]");
    for (const auto& tensor : {a, out, *activation_scale, b_scale, rows, indices,
                              routing.expert_ids_and_cumsum, routing.sorted_token_ids,
                              routing.aligned_num_m_blocks}) check_workspace_overlap(workspace, tensor);
    if (options.weight_global_scale) {
        check_workspace_overlap(workspace, *options.weight_global_scale);
        const auto scale_shape = options.weight_global_scale->numel() == 1
            ? std::vector<int64_t>{} : std::vector<int64_t>{groups, 1, 1};
        b_scale = b_scale * options.weight_global_scale->reshape(scale_shape);
    }
    auto a_scale = *activation_scale;
    if (options.lhs_scale_group_size == 0) {
        auto scale_shape = a.sizes().vec();
        scale_shape.back() = ceil_div(k, int64_t(128));
        a_scale = a_scale.expand(scale_shape);
    }
    dequantize_nvfp4_fp8_impl(weight, workspace);
    if (a.size(-2) == 0) return;
    const int m = a.size(-2);
    if (dense) {
        deep_gemm::fp8_gemm(a, a_scale, workspace, b_scale, out, m, n, k, fp8_config);
    } else if (kind == GemmType::MGroupedNoPad) {
        m_grouped_gemm_fp8_fp8_bf16_nt_nopad_impl(
            a, a_scale, workspace, b_scale, out, indices, m, n, k, groups, rows, fp8_config);
    } else if (kind == GemmType::MGroupedMasked) {
        auto signal = at::empty({0}, a.options().dtype(at::kInt));
        m_grouped_gemm_fp8_fp8_bf16_nt_masked_impl(
            a, a_scale, workspace, b_scale, out, rows, m, n, k, groups, expected_m,
            fp8_config, 256, false, signal);
    } else {
        TORCH_CHECK(n % 128 == 0 && k % 128 == 0 && std::get<3>(*routing_config) == 128,
                    "FP8 fused routing requires N/K divisible by 128 and block_k=128");
        auto aligned_scale = get_mn_major_tma_aligned_tensor(a_scale);
        m_grouped_gemm_blkwise_nt_fused_impl(a, aligned_scale, workspace, b_scale, out, rows,
            routing.expert_ids_and_cumsum, routing.sorted_token_ids,
            routing.aligned_num_m_blocks, routing.topk, *routing_config);
    }
}

static void dispatch(GemmType kind, const Activation& lhs, Weight rhs, const torch::Tensor& out,
                     std::optional<torch::Tensor> rows, const torch::Tensor& indices,
                     int expected_m, detail::Routing routing, const Options& options) {
    const auto& [a, a_scale] = lhs;
    TORCH_CHECK(a.is_cuda(), "NVFP4 W4A8 requires PPU tensors");
    c10::DeviceGuard guard(a.device());
    TORCH_CHECK(is_ppu1v5_device(), "NVFP4 W4A8 requires PPU1.5");
    TORCH_CHECK(options.execution == "fused" || options.execution == "dequantize",
                "execution must be 'fused' or 'dequantize'");
    const bool decode = options.execution == "dequantize";
    TORCH_CHECK(decode == options.fp8_workspace.has_value(),
                "execution='dequantize' requires fp8_workspace; fused execution does not use it");
    const bool dense = kind == GemmType::Normal, masked = kind == GemmType::MGroupedMasked;
    const bool fused = kind == GemmType::MGroupedFused;
    TORCH_CHECK(a.dim() == (masked ? 3 : 2) && out.dim() == a.dim(), "Invalid NVFP4 operand ranks");
    const int m = a.size(-2), k = a.size(-1), n = out.size(-1);
    if (dense) {
        if (rhs.first.dim() == 2) rhs.first = rhs.first.unsqueeze(0);
        if (rhs.second.dim() == 2) rhs.second = rhs.second.unsqueeze(0);
    }
    TORCH_CHECK(rhs.first.dim() == 3 && rhs.second.dim() == 3, "Invalid NVFP4 weight ranks");
    const int groups = rhs.first.size(0);
    TORCH_CHECK(groups > 0 && n > 0 && n % 64 == 0 && k > 0 && k % 32 == 0,
                "NVFP4 W4A8 requires E>0, N divisible by 64 and K divisible by 32");
    const auto weight_shape = decode ? std::vector<int64_t>{groups, n, k / 8}
                                     : std::vector<int64_t>{groups, k / 32, 4 * n};
    const auto scale_shape = decode ? std::vector<int64_t>{groups, n, k / 16}
                                    : std::vector<int64_t>{groups, k / 16, n};
    TORCH_CHECK(rhs.first.sizes() == weight_shape && rhs.second.sizes() == scale_shape,
                "Invalid NVFP4 prepared weight/scale shape for the selected execution");
    TORCH_CHECK(!dense || groups == 1, "Dense NVFP4 requires one weight matrix");
    TORCH_CHECK(!masked || (a.size(0) == groups && out.size(0) == groups), "Masked expert count mismatch");
    TORCH_CHECK(fused || out.size(-2) == m, "Invalid NVFP4 output shape");
    TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn && rhs.first.scalar_type() == torch::kInt32 &&
                    rhs.second.scalar_type() == torch::kUInt8 && out.scalar_type() == torch::kBFloat16,
                "NVFP4 W4A8 expects E4M3FN activations, int32 weights, uint8 scales and BF16 output");
    for (const auto& tensor : {a, rhs.first, rhs.second, out}) check_tensor(tensor, a);
    TORCH_CHECK(options.lhs_scale_group_size == 0 || options.lhs_scale_group_size == 16 ||
                    (options.lhs_scale_group_size > 0 && options.lhs_scale_group_size % 32 == 0),
                "Activation scale group must be 0 (per row), 16, or a positive multiple of 32");
    if (a_scale) {
        check_tensor(*a_scale, a, !decode);
        auto shape = a.sizes().vec();
        shape.back() = options.lhs_scale_group_size ? ceil_div(k, options.lhs_scale_group_size) : 1;
        TORCH_CHECK(a_scale->scalar_type() == torch::kFloat32 && a_scale->sizes() == shape,
                    "Invalid FP32 activation scale shape");
    } else {
        TORCH_CHECK(options.lhs_scale_group_size == 0, "Blockwise activations require activation scales");
    }
    if (options.weight_global_scale) {
        check_tensor(*options.weight_global_scale, a);
        TORCH_CHECK(options.weight_global_scale->scalar_type() == torch::kFloat32 &&
                        (options.weight_global_scale->numel() == 1 || options.weight_global_scale->numel() == groups),
                    "Weight global scale must be FP32 scalar or one value per expert");
    }
    TORCH_CHECK(options.rhs_scale_block_size == 0 || options.rhs_scale_block_size == 128,
                "rhs_scale_block_size must be 0 (per channel) or 128 (128x128 blocks)");
    TORCH_CHECK(options.rhs_scale_block_size == 0 || options.rhs_scales,
                "Blockwise FP8 weights require rhs_scales");
    if (options.rhs_scales) {
        const auto& scales = *options.rhs_scales;
        check_tensor(scales, a);
        bool valid;
        if (options.rhs_scale_block_size) {
            const int nb = ceil_div(n, 128), kb = ceil_div(k, 128);
            valid = scales.sizes() == std::vector<int64_t>{groups, nb, kb} ||
                (dense && scales.sizes() == std::vector<int64_t>{nb, kb});
        } else {
            valid = scales.sizes() == std::vector<int64_t>{groups, n} ||
                scales.sizes() == std::vector<int64_t>{groups, n, 1} ||
                (dense && (scales.sizes() == std::vector<int64_t>{n} ||
                           scales.sizes() == std::vector<int64_t>{n, 1}));
        }
        TORCH_CHECK(scales.scalar_type() == torch::kFloat32 && valid, "Invalid FP32 rhs_scales shape");
    }
    torch::Tensor counts;
    if (!dense) {
        if (kind == GemmType::MGroupedNoPad) {
            check_indices(indices, a);
            TORCH_CHECK(indices.dim() == 1 && indices.numel() == m, "m_indices must be int32 [M]");
            expected_m = ceil_div(m, groups);
        }
        counts = rows ? *rows : at::bincount(indices.to(at::kLong), {}, groups).to(at::kInt);
        check_indices(counts, a);
        TORCH_CHECK(counts.dim() == 1 && counts.numel() == groups, "m_rows must be int32 [E]");
    }
    if (masked) TORCH_CHECK(expected_m > 0, "expected_m must be positive");
    if (fused) {
        TORCH_CHECK(m == 0 ? out.size(0) == 0 : out.size(0) % m == 0,
                    "Fused output M must be num_tokens * topk");
        routing.topk = m == 0 ? 1 : out.size(0) / m;
        TORCH_CHECK(routing.topk > 0 && routing.topk <= groups, "Invalid fused top-k");
        for (const auto& tensor : {routing.expert_ids_and_cumsum, routing.sorted_token_ids,
                                  routing.aligned_num_m_blocks}) check_indices(tensor, a);
        TORCH_CHECK(routing.expert_ids_and_cumsum.dim() == 2 &&
                        routing.expert_ids_and_cumsum.size(1) == 4 && routing.sorted_token_ids.dim() == 2 &&
                        routing.sorted_token_ids.size(0) == routing.expert_ids_and_cumsum.size(0) &&
                        routing.aligned_num_m_blocks.numel() == 1, "Invalid NVFP4 routing shapes");
        expected_m = ceil_div(int(out.size(0)), groups);
    }
    if (decode) {
        run_dequantized(kind, lhs, rhs, out, counts, indices, expected_m, routing, options);
    } else {
        std::optional<ConfigTuple> config;
        if (options.configs) {
            const auto* requested = std::get_if<ConfigTuple>(&*options.configs);
            TORCH_CHECK(requested, "Fused NVFP4 execution requires an eight-entry W4A8 config");
            config = *requested;
        }
        detail::run(kind, dense ? m : expected_m, a, rhs, out, config, counts, a_scale,
            options.lhs_scale_group_size, options.weight_global_scale, options.rhs_scales,
            options.rhs_scale_block_size, routing);
    }
}

static void gemm_fp8_nvfp4_bf16_nt_impl(const Activation& lhs, const Weight& rhs, const torch::Tensor& out,
                             std::optional<ExecutionConfig> configs, int lhs_scale_group_size,
                             OptionalTensor weight_global_scale, OptionalTensor rhs_scales,
                             int rhs_scale_block_size, const std::string& execution, OptionalTensor fp8_workspace) {
    dispatch(GemmType::Normal, lhs, rhs, out, std::nullopt, {}, 0, {},
        {configs, lhs_scale_group_size, weight_global_scale, rhs_scales, rhs_scale_block_size, execution, fp8_workspace});
}

static void m_grouped_gemm_fp8_nvfp4_bf16_nt_nopad_impl(const Activation& lhs, const Weight& rhs, const torch::Tensor& out,
        const torch::Tensor& m_indices, OptionalTensor m_rows, std::optional<ExecutionConfig> configs,
        int lhs_scale_group_size, OptionalTensor weight_global_scale, OptionalTensor rhs_scales,
        int rhs_scale_block_size, const std::string& execution, OptionalTensor fp8_workspace) {
    dispatch(GemmType::MGroupedNoPad, lhs, rhs, out, m_rows, m_indices, 0, {},
        {configs, lhs_scale_group_size, weight_global_scale, rhs_scales, rhs_scale_block_size, execution, fp8_workspace});
}

static void m_grouped_gemm_fp8_nvfp4_bf16_nt_masked_impl(const Activation& lhs, const Weight& rhs, const torch::Tensor& out,
        const torch::Tensor& masked_m, int expected_m, std::optional<ExecutionConfig> configs,
        int lhs_scale_group_size, OptionalTensor weight_global_scale, OptionalTensor rhs_scales,
        int rhs_scale_block_size, const std::string& execution, OptionalTensor fp8_workspace) {
    dispatch(GemmType::MGroupedMasked, lhs, rhs, out, masked_m, {}, expected_m, {},
        {configs, lhs_scale_group_size, weight_global_scale, rhs_scales, rhs_scale_block_size, execution, fp8_workspace});
}

static void m_grouped_gemm_fp8_nvfp4_bf16_nt_fused_impl(const Activation& lhs, const Weight& rhs, const torch::Tensor& out,
        const torch::Tensor& m_rows, const torch::Tensor& expert_ids_and_cumsum,
        const torch::Tensor& sorted_token_ids, const torch::Tensor& aligned_num_m_blocks,
        std::optional<ExecutionConfig> configs, int lhs_scale_group_size, OptionalTensor weight_global_scale,
        OptionalTensor rhs_scales, int rhs_scale_block_size, const std::string& execution, OptionalTensor fp8_workspace) {
    dispatch(GemmType::MGroupedFused, lhs, rhs, out, m_rows, {}, 0,
        {0, expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks},
        {configs, lhs_scale_group_size, weight_global_scale, rhs_scales, rhs_scale_block_size, execution, fp8_workspace});
}

} // namespace deep_gemm::nvfp4
