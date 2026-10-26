#pragma once

#include <cstddef>
#include <utility>
#include "fp4_gemm.hpp"
#include "fp8_gemm.hpp" // Existing grouped-scheduler block-info preprocessing.
#include "mega_moe_dispatch.hpp"

namespace deep_gemm {

using PackedStride = cute::Stride<int64_t, cute::Int<1>, int64_t>;
using ScaleStride = cute::Stride<cute::Int<1>, int64_t, int64_t>;
using Problem = FP4GemmRuntime::GemmProblemSize;
using Mode = FP4GemmRuntime::GemmUniversalMode;

using ScatterScheduler = TileSchedulerArguments;

struct Fp4CombineBuffer {
    void* base = nullptr;
    uint32_t hidden_bytes = 0, max_tokens_per_rank = 0, num_topk = 0;
    uint64_t* scales = nullptr;
    uint32_t scale_tiles = 0, scale_block_n = 0, use_side_scale = 0;
    int64_t scale_offsets[64]{};
};

struct Fp4SymmetricBuffer {
    int64_t base = 0, offsets[64]{};
    uint32_t rank_idx = 0, num_ranks = 0;
};

template <typename Epilogue>
struct Fp4ReplicaParams {
    Mode mode{};
    Problem problem_shape{};
    FP4GemmRuntime::MainLoopArguments mainloop{};
    Epilogue epilogue{};
    cutlass::KernelHardwareInfo hw_info{};
    ScatterScheduler scheduler{};
    void* workspace = nullptr;
    // The NExpand == 1 specialization has a signal field even without SBO.
    int32_t* signal = nullptr;
    const uint8_t* ptr_B_replica = nullptr;
    const uint16_t* ptr_scale_B_replica = nullptr;
    int32_t num_master_groups = 0;
    int32_t* ptr_sync_ce = nullptr;
};

struct Fp4ScatterState {
    Fp4CombineBuffer moe_combine_buf;
    Fp4SymmetricBuffer moe_sym_buffer;
    const int64_t* moe_token_meta = nullptr;
    const int32_t* moe_valid_m_rows = nullptr;
    const int32_t* moe_src_m_offsets = nullptr;
};

template <typename Epilogue>
struct Fp4ScatterParams : Fp4ReplicaParams<Epilogue>, Fp4ScatterState {};

inline PackedStride packed_stride(int rows, int cols) {
    return cutlass::make_cute_packed_stride(PackedStride{}, cute::make_shape(rows, cols, 1));
}

template <typename Params>
inline std::string scatter_source(std::string body, const std::string& name) {
    body += R"DGABI(
static_assert(sizeof(Kernel::Params) == {ABI_SIZE}, "scatter Params ABI mismatch");
static_assert(alignof(Kernel::Params) == {ABI_ALIGN}, "scatter Params alignment mismatch");
static_assert(sizeof(Kernel::MainloopParams) == {ABI_MAINLOOP_SIZE}, "scatter mainloop ABI mismatch");
static_assert(sizeof(Kernel::EpilogueParams) == {ABI_EPILOGUE_SIZE}, "scatter epilogue ABI mismatch");
static_assert(offsetof(Kernel::Params,mainloop) == {ABI_MAINLOOP_OFFSET}, "scatter mainloop offset mismatch");
static_assert(offsetof(Kernel::Params,epilogue) == {ABI_EPILOGUE_OFFSET}, "scatter epilogue offset mismatch");
static_assert(offsetof(Kernel::Params,hw_info) == {ABI_HW_OFFSET}, "scatter hardware offset mismatch");
static_assert(offsetof(Kernel::Params,scheduler) == {ABI_SCHEDULER_OFFSET}, "scatter scheduler offset mismatch");
static_assert(offsetof(Kernel::Params,workspace) == {ABI_WORKSPACE_OFFSET}, "scatter workspace offset mismatch");
static_assert(offsetof(Kernel::Params,signal) == {ABI_SIGNAL_OFFSET}, "scatter signal offset mismatch");
extern "C" __device__ __constant__ uint32_t ppu_mega_moe_smem_bytes = Kernel::SharedStorageSize;
extern "C" __global__
__launch_bounds__(Kernel::MaxThreadsPerBlock,Kernel::MinBlocksPerMultiprocessor)
void PPU_SCATTER_ENTRY(typename Kernel::Params params) {
    extern __shared__ char smem[];
    Kernel op;
    op(params,smem);
}
)DGABI";
    const std::string marker = "PPU_SCATTER_ENTRY";
    body.replace(body.find(marker), marker.size(), name);
    replace_source_field(body, "ABI_SIZE", std::to_string(sizeof(Params)));
    replace_source_field(body, "ABI_ALIGN", std::to_string(alignof(Params)));
    replace_source_field(body, "ABI_MAINLOOP_SIZE", std::to_string(sizeof(Params{}.mainloop)));
    replace_source_field(body, "ABI_EPILOGUE_SIZE", std::to_string(sizeof(Params{}.epilogue)));
    replace_source_field(body, "ABI_MAINLOOP_OFFSET", std::to_string(offsetof(Params, mainloop)));
    replace_source_field(body, "ABI_EPILOGUE_OFFSET", std::to_string(offsetof(Params, epilogue)));
    replace_source_field(body, "ABI_HW_OFFSET", std::to_string(offsetof(Params, hw_info)));
    replace_source_field(body, "ABI_SCHEDULER_OFFSET", std::to_string(offsetof(Params, scheduler)));
    replace_source_field(body, "ABI_WORKSPACE_OFFSET", std::to_string(offsetof(Params, workspace)));
    replace_source_field(body, "ABI_SIGNAL_OFFSET", std::to_string(offsetof(Params, signal)));
    return body;
}

using FP4MegaMoELinear2Params = Fp4ScatterParams<FP4GemmRuntime::EpilogueArgsNoTsm>;

struct FP4MegaMoELinear2Args {
    ScatterConfig launch_info;
    LaunchArgs launch_args;
    FP4MegaMoELinear2Params kernel_params;
    // The optional FP8 epilogue scratch must outlive parameter preparation.
    torch::Tensor output_scratch{};
};

inline std::string generate_fp4_mega_moe_linear2(const FP4MegaMoELinear2Args& args) {
    using KernelParams = FP4MegaMoELinear2Params;
    const auto& config = args.launch_info;
    const int n = config.n, k = config.k, groups = config.num_groups;
    const int threads = args.launch_args.block_dim.x;
    FP4GemmRuntime::LaunchInfo info{config.block_m,
                                    config.block_n,
                                    config.block_k,
                                    config.warp_m,
                                    config.warp_n,
                                    groups,
                                    config.num_stages,
                                    n,
                                    k,
                                    "MGroupedNoPad",
                                    "m_grouped_gemm_fp4_moe_scatter",
                                    false,
                                    1,
                                    false};
    FP4GemmRuntime::Args shared{info, {dim3(1), dim3(threads), 0}, {}};
    const char* scale_mode = config.fp8_scale_mode == mega_moe::Fp8ScaleMode::SideDynamic ? "SideDynamic"
        : config.fp8_scale_mode == mega_moe::Fp8ScaleMode::SideStatic ? "SideStatic" : "Fixed";
    const auto policy = fmt::format("deep_gemm::mega_moe::fp4::ScatterPolicy<{}, deep_gemm::mega_moe::Fp8ScaleMode::{}, {}>",
                                    config.fp8_output, scale_mode, config.fp8_exp_shift);
    auto body = FP4GemmRuntime::generate_code(shared, true, policy,
                                              "#include <deep_gemm/impls/fp4_gemm_scatter.cuh>", false,
                                              config.m_tiles_per_group);
    body += "using Kernel = deep_gemm::GemmKernel;\n";
    body += "static_assert(sizeof(deep_gemm::mega_moe::detail::CombineBuffer)==560, \"FP4 combine ABI mismatch\");\n";
    body += fmt::format("static_assert(sizeof(deep_gemm::mega_moe::detail::SymBufferLite<>) == {}, \"FP4 symmetric ABI mismatch\");\n", sizeof(Fp4SymmetricBuffer));
    const std::pair<const char*, size_t> abi_offsets[] = {
        {"ptr_B_replica", offsetof(KernelParams, ptr_B_replica)},
        {"ptr_scale_B_replica", offsetof(KernelParams, ptr_scale_B_replica)},
        {"num_master_groups", offsetof(KernelParams, num_master_groups)},
        {"ptr_sync_ce", offsetof(KernelParams, ptr_sync_ce)},
        {"moe_combine_buf", offsetof(KernelParams, moe_combine_buf)},
        {"moe_sym_buffer", offsetof(KernelParams, moe_sym_buffer)},
        {"moe_token_meta", offsetof(KernelParams, moe_token_meta)},
        {"moe_valid_m_rows", offsetof(KernelParams, moe_valid_m_rows)},
        {"moe_src_m_offsets", offsetof(KernelParams, moe_src_m_offsets)},
    };
    for (const auto& [field, offset] : abi_offsets)
        body += fmt::format("static_assert(offsetof(Kernel::Params, {}) == {}, "
                            "\"FP4 scatter {} offset mismatch\");\n",
                            field, offset, field);
    return scatter_source<KernelParams>(std::move(body), "m_grouped_gemm_fp4_moe_scatter");
}

inline int32_t* prepare_scatter_layout(int groups, int block_m, const torch::Tensor& rows,
                                      const torch::Tensor& block_info) {
    TORCH_CHECK(groups > 0 && groups <= 1024, "scatter preprocessing supports 1 to 1024 local experts");
    auto* layout = rows.data_ptr<int32_t>();
    if (groups < 128)
        return layout;
    TORCH_CHECK(block_info.defined() && block_info.scalar_type() == torch::kInt32,
                "Grouped no-pad scheduler needs an int32 block-info workspace");
    const auto args = ComputeBlockInfoKernelRuntime::Args{
        .launch_attr_args = {reinterpret_cast<const uint32_t*>(layout), static_cast<uint32_t>(groups),
                            reinterpret_cast<uint32_t*>(block_info.data_ptr<int32_t>())},
        .launch_args = {dim3(1), dim3(std::max(32, next_power_of_two(groups))), 0},
    };
    const auto code = ComputeBlockInfoKernelRuntime::generate(block_m);
    const auto runtime = compiler->build("computeBlockInfoKernel", code);
    ComputeBlockInfoKernelRuntime::launch(runtime, args);
    return block_info.data_ptr<int32_t>();
}

inline FP4MegaMoELinear2Args make_fp4_mega_moe_scatter_args(
    const ScatterConfig& config, const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
    const torch::Tensor& rhs, const torch::Tensor& rhs_scales, const torch::Tensor& rhs_replicas,
    const torch::Tensor& rhs_replica_scales, const torch::Tensor& y, int num_rows,
    const torch::Tensor& rows, const torch::Tensor& block_info, const torch::Tensor& token_meta,
    const torch::Tensor& combine, int hidden_bytes, int num_max_tokens_per_rank, int num_topk, const torch::Tensor& scales,
    const torch::Tensor& scale_offsets, int scale_tiles, int scale_block_n, int use_side_scale,
    const layout::SymBuffer<1>& peer_offsets, int rank_idx, int num_ranks,
    const torch::Tensor& valid_rows, const torch::Tensor& src_offsets) {
    const int m = num_rows, n = config.n, k = config.k, groups = config.num_groups;
    const auto layout = prepare_scatter_layout(groups, config.block_m, rows, block_info);
    const auto sa = packed_stride(m, k),
               sb = packed_stride(n, k),
               sc = packed_stride(m, 0),
               sd = packed_stride(m, n);
    const auto sfa = cutlass::make_cute_packed_stride(ScaleStride{}, cute::make_shape(m, ceil_div(k, 32), 1));
    const auto sfb = cutlass::make_cute_packed_stride(ScaleStride{}, cute::make_shape(n, ceil_div(k, 32), 1));
    using Params = FP4MegaMoELinear2Params;
    Params params{};
    params.problem_shape = {m, n, k, 1};
    params.mainloop = {
        {m, n, k}, device_pointer<uint8_t>(lhs),         sa,  device_pointer<uint8_t>(rhs),
        sb,        device_pointer<uint16_t>(lhs_scales), sfa, device_pointer<uint16_t>(rhs_scales),
        sfb};
    params.epilogue = {{1, 0}, nullptr, sc, device_pointer<cutlass::bfloat16_t>(y), sd};
    params.scheduler = {uint32_t(m), layout};
    params.moe_combine_buf.base = device_pointer(combine);
    params.moe_combine_buf.hidden_bytes = hidden_bytes;
    params.moe_combine_buf.max_tokens_per_rank = num_max_tokens_per_rank;
    params.moe_combine_buf.num_topk = num_topk;
    params.moe_combine_buf.scales = scales.defined() ? device_pointer<uint64_t>(scales) : nullptr;
    params.moe_combine_buf.scale_tiles = scale_tiles;
    params.moe_combine_buf.scale_block_n = scale_block_n;
    params.moe_combine_buf.use_side_scale = use_side_scale;
    TORCH_CHECK(num_ranks > 0 && num_ranks <= 64 && rank_idx >= 0 && rank_idx < num_ranks, "invalid FP4 scatter rank");
    params.moe_sym_buffer.rank_idx = rank_idx;
    params.moe_sym_buffer.num_ranks = num_ranks;
    std::copy_n(peer_offsets.offsets, num_ranks, params.moe_sym_buffer.offsets);
    if (params.moe_combine_buf.use_side_scale) {
        TORCH_CHECK(!scale_offsets.is_cuda() && scale_offsets.scalar_type() == torch::kInt64 &&
                        scale_offsets.is_contiguous() && scale_offsets.numel() >= num_ranks,
                    "FP4 side-scale offsets must be a CPU int64 table");
        std::copy_n(scale_offsets.data_ptr<int64_t>(), num_ranks, params.moe_combine_buf.scale_offsets);
    }
    params.moe_token_meta = device_pointer<int64_t>(token_meta);
    params.moe_valid_m_rows = device_pointer<int32_t>(valid_rows);
    params.moe_src_m_offsets = device_pointer<int32_t>(src_offsets);
    params.ptr_B_replica = device_pointer<uint8_t>(rhs_replicas);
    params.ptr_scale_B_replica = device_pointer<uint16_t>(rhs_replica_scales);
    params.num_master_groups = config.num_master_groups;
    const int threads = (config.block_m / config.warp_m) * (config.block_n / config.warp_n) * 32;
    const int sms = device_runtime->get_prop()->multiProcessorCount;
    TORCH_CHECK(sms > 0 && threads > 0 && threads <= 1024, "invalid scatter geometry");
    return FP4MegaMoELinear2Args{config, {dim3(sms), dim3(threads), 0}, params};
}


inline torch::Tensor scatter_block_info(const torch::Tensor& rows, int m, int groups, int block_m,
                                       const std::optional<torch::Tensor>& workspace) {
    if (groups < 128)
        return rows;
    const int64_t required = 4LL * (1 + groups + ceil_div(m, block_m));
    TORCH_CHECK(workspace, "MegaMoE GEMM2 requires workspace-owned block-info scratch");
    TORCH_CHECK(workspace->scalar_type() == torch::kInt32 && workspace->is_contiguous() &&
                    workspace->device() == rows.device() && workspace->numel() >= required,
                "MegaMoE GEMM2 block-info workspace has an invalid device, dtype, layout or size");
    return *workspace;
}

inline FP4MegaMoELinear2Args make_fp4_mega_moe_linear2_args(
    const torch::Tensor& lhs, const torch::Tensor& sa, const torch::Tensor& rhs, const torch::Tensor& sb,
    const torch::Tensor& replica, const torch::Tensor& rs, const torch::Tensor& physical,
    const torch::Tensor& valid, const torch::Tensor& token_meta, const torch::Tensor& combine_buffer,
    const layout::SymBuffer<1>& sym_offsets, int m, int n, int k, int groups, int num_master_experts, int rank_idx, int num_ranks,
    int num_max_tokens_per_rank, int hidden, int num_topk, bool fp8, int shift,
    const mega_moe_config::Config& config,
    std::optional<torch::Tensor> combine_scales,
    std::optional<torch::Tensor> scale_offsets, int scale_block_n,
    const torch::Tensor& src_m_offsets, int m_tiles_per_group,
    const std::optional<torch::Tensor>& workspace = std::nullopt) {
    const auto [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    const auto info = scatter_block_info(physical, m, groups, bm, workspace);
    const auto y = fp8 ? torch::empty({m, n}, lhs.options().dtype(torch::kBFloat16))
                         : combine_buffer.reshape({-1}).narrow(0, 0, n).view({1, n});
    const bool side = fp8 && combine_scales.has_value();
    int side_mode = 0, tiles = 1;
    if (!scale_block_n)
        scale_block_n = bn;
    torch::Tensor cs, so;
    if (side) {
        side_mode = 1;
        TORCH_CHECK(combine_scales->is_cuda() && combine_scales->scalar_type() == torch::kInt64 &&
                    combine_scales->is_contiguous());
        TORCH_CHECK(scale_offsets && scale_offsets->scalar_type() == torch::kInt64 &&
                    scale_offsets->numel() >= num_ranks);
        TORCH_CHECK(scale_block_n >= 64 && scale_block_n <= bn && bn % scale_block_n == 0 &&
                    scale_block_n % 8 == 0 && bn / scale_block_n <= 4,
                    "FP8 scatter scale_block_n must divide BLOCK_N into at most four groups, "
                    "each aligned to 8 and at least 64 columns");
        tiles = ceil_div(hidden, scale_block_n);
        TORCH_CHECK(combine_scales->numel() >= int64_t(num_topk) * num_max_tokens_per_rank * tiles);
        cs = *combine_scales;
        so = *scale_offsets;
        shift = 0;
    }
    ScatterConfig c;
    c.n = n;
    c.k = k;
    c.block_m = bm;
    c.block_n = bn;
    c.block_k = bk;
    c.warp_m = wm;
    c.warp_n = wn;
    c.num_groups = groups;
    c.num_master_groups = num_master_experts;
    c.num_stages = stages;
    c.fp8_output = fp8;
    c.fp8_exp_shift = shift;
    c.m_tiles_per_group = m_tiles_per_group;
    c.fp8_scale_mode = !side ? mega_moe::Fp8ScaleMode::Fixed
        : side_mode == 2 ? mega_moe::Fp8ScaleMode::SideDynamic : mega_moe::Fp8ScaleMode::SideStatic;

    auto args = make_fp4_mega_moe_scatter_args(
        c, lhs, sa, rhs, sb, replica, rs, y, m, physical, info, token_meta, combine_buffer,
        fp8 ? hidden : hidden * 2, num_max_tokens_per_rank, num_topk, cs, so, tiles, scale_block_n, side_mode, sym_offsets,
        rank_idx, num_ranks, valid, src_m_offsets);
    if (fp8)
        args.output_scratch = y;
    return args;
}


} // namespace deep_gemm
