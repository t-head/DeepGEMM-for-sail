#pragma once

#include <cstddef>
#include <utility>
#include "fp4_gemm.hpp"
#include "m_grouped_w4a16_gemm.hpp"
#include "fp8_gemm.hpp"
#include "mega_moe.hpp"

namespace deep_gemm::mega_moe_native {

using PackedStride = cute::Stride<int64_t, cute::Int<1>, int64_t>;
using ScaleStride = cute::Stride<cute::Int<1>, int64_t, int64_t>;
using Problem = FP4GemmRuntime::GemmProblemSize;
using Mode = FP4GemmRuntime::GemmUniversalMode;

using ScatterScheduler = WorkRangeTileSchedulerArguments;

struct ChannelScatter {
    void* combine_base = nullptr;
    uint32_t hidden_bytes = 0, max_tokens_per_rank = 0, num_topk = 0;
    const int64_t* packed_src_info = nullptr;
    const int32_t* valid_m_rows = nullptr;
    const int64_t* peer_offsets = nullptr;
    uint32_t rank_idx = 0, num_ranks = 0;
    const int32_t* row_group_indices = nullptr;
    uint32_t num_groups = 0;
};

struct BlockScatter {
    ChannelScatter base;
    const int32_t* row_offsets = nullptr;
};

template <typename Epilogue, typename Scatter> struct ScatterEpilogue {
    Epilogue base;
    Scatter scatter;
    void* base_ptr_D = nullptr;
};

struct ChannelDefaultParams {
    Mode mode = Mode::kGemm;
    Problem problem_shape;
    DenseINT8GemmCutlass3Runtime::MainLoopArguments mainloop;
    DenseINT8GemmCutlass3Runtime::EpilogueArgs epilogue;
    cutlass::KernelHardwareInfo hw_info;
    ScatterScheduler scheduler;
    void* workspace = nullptr;
    int32_t* signal = nullptr;
    ChannelScatter scatter;
    const void* rhs_replicas = nullptr;
    const float* rhs_replica_scales = nullptr;
    uint32_t num_master_groups = 0;
};

struct ChannelOverlapParams {
    Mode mode = Mode::kGemm;
    Problem problem_shape;
    DenseINT8GemmCutlass3Runtime::MainLoopArguments mainloop;
    ScatterEpilogue<DenseINT8GemmCutlass3Runtime::EpilogueArgs, ChannelScatter> epilogue;
    cutlass::KernelHardwareInfo hw_info;
    ScatterScheduler scheduler;
    void* workspace = nullptr;
    int32_t* signal = nullptr;
    const void* rhs_replicas = nullptr;
    const float* rhs_replica_scales = nullptr;
    uint32_t num_master_groups = 0;
};

struct BlockScatterParams {
    Mode mode = Mode::kGemm;
    Problem problem_shape;
    FP8GemmRuntime::MainLoopArguments mainloop;
    ScatterEpilogue<FP8GemmRuntime::EpilogueArgs, BlockScatter> epilogue;
    cutlass::KernelHardwareInfo hw_info;
    struct Scheduler {
        ScatterScheduler base;
        const int32_t* pool_row_offsets = nullptr;
    } scheduler;
    void* workspace = nullptr;
    int32_t* signal = nullptr;
    const void* rhs_replicas = nullptr;
    const float* rhs_replica_scales = nullptr;
    uint32_t num_master_groups = 0;
};

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

// The TSM epilogue without bias uses LinearCombination's full parameter
// struct, whereas the bias epilogue uses the smaller elementwise struct.
struct Fp4ScatterEpilogueWithTsm {
    decltype(std::declval<FP4GemmRuntime::EpilogueArgsNoTsm>().thread) thread;
    const float* ptr_C = nullptr;
    cute::Stride<int64_t, cute::Int<1>, int64_t> stride_C;
    cutlass::bfloat16_t* ptr_D = nullptr;
    cute::Stride<int64_t, cute::Int<1>, int64_t> stride_D;
    const float* ptr_Bias = nullptr;
    cute::Stride<cute::Int<0>, cute::Int<1>, int64_t> stride_Bias;
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
inline std::string scatter_source(std::string body, const ScatterConfig& config, const std::string& name) {
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
extern "C" __device__ __constant__ uint32_t ppu_native_smem_bytes = Kernel::SharedStorageSize;
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
    return render(std::move(body), config);
}

template <typename Tag, typename Params>
inline void scatter_launch(const std::string& name, std::string body, const ScatterConfig& config,
                           Params params, int sms, int threads) {
    TORCH_CHECK(sms > 0 && threads > 0 && threads <= 1024, "invalid scatter geometry");
    NativeKernelArgs args{scatter_source<Params>(std::move(body), config, name),
                          {dim3(sms), dim3(threads), 0},
                          {native_pod_argument(params)}};
    const auto runtime = compile_native_kernel<Tag>(name, args);
    args.launch_args.smem_size = native_kernel_smem(runtime);
    const int occupancy = native_kernel_occupancy(runtime, threads, args.launch_args.smem_size);
    TORCH_CHECK(occupancy > 0, "scatter launch cannot keep a block resident");
    args.launch_args.grid_dim.x = sms * occupancy;
    params.hw_info = {0, sms * occupancy};
    args.kernel_args = {native_pod_argument(params)};
    NativeKernelRuntime<Tag>::launch(runtime, args);
}

inline std::string channel_scatter_family() {
    return R"DGSOURCE(
#include <deep_gemm/impls/fp8_gemm_channelwise_scatter.cuh>
using namespace deep_gemm;
using Scatter = mega_moe::detail::Bf16MoeScatter<{USE_PADDED_VALID_ROWS}, {CHECK_PACKED_ROUTES},
    {ROLLING_PAIR_SCATTER}>;
using Family = ChannelwiseFp8GemmWithScatter<cutlass::float_e4m3_t,float,
    {N},{K},{BLOCK_M},{BLOCK_N},{BLOCK_K},{WARP_M},{WARP_N},{NUM_GROUPS},{NUM_STAGES},
    GemmType::MGroupedNoPad,false,KernelStrategy::{KERNEL_TYPE},Scatter,false,{BM128_OVERLAP_PAIR_STAGING}>;
using Kernel = typename Family::GemmKernel;
)DGSOURCE";
}

struct ChannelDefaultScatterTag {};
struct ChannelOverlapScatterTag {};
inline void channel_scatter_kernel(const ScatterConfig& config, const torch::Tensor& lhs,
                                   const torch::Tensor& lhs_scales, const torch::Tensor& rhs,
                                   const torch::Tensor& rhs_scales, const torch::Tensor& rhs_replicas,
                                   const torch::Tensor& rhs_replica_scales, const torch::Tensor& output,
                                   int num_rows, const torch::Tensor& rows, const torch::Tensor& block_info,
                                   int num_sms, const torch::Tensor& packed_src, const torch::Tensor& combine,
                                   int hidden_bytes, int max_tokens, int topk,
                                   const torch::Tensor& peer_offsets, int rank, int ranks,
                                   const torch::Tensor& valid_rows,
                                   const torch::Tensor& indices, int row_groups) {
    const int m = num_rows, n = config.n, k = config.k;
    const int groups = config.num_groups, masters = config.num_master_groups;
    if (m == 0)
        return;
    const auto layout = prepare_grouped_nopad_layout(GemmType::MGroupedNoPad, groups, config.block_m,
                                                     rows, block_info);
    const auto sa = packed_stride(m, k), sb = packed_stride(n, k), sd = packed_stride(m, n);
    DenseINT8GemmCutlass3Runtime::MainLoopArguments mainloop{device_pointer(lhs),
                                                             sa,
                                                             device_pointer(rhs),
                                                             sb,
                                                             device_pointer<float>(lhs_scales),
                                                             device_pointer<float>(rhs_scales)};
    DenseINT8GemmCutlass3Runtime::EpilogueArgs epilogue{{1, 0},
                                                        device_pointer<cutlass::bfloat16_t>(output),
                                                        sd,
                                                        device_pointer<cutlass::bfloat16_t>(output),
                                                        sd};
    ChannelScatter scatter{device_pointer(combine),
                           uint32_t(hidden_bytes),
                           uint32_t(max_tokens),
                           uint32_t(topk),
                           device_pointer<int64_t>(packed_src),
                           device_pointer<int32_t>(valid_rows),
                           device_pointer<int64_t>(peer_offsets),
                           uint32_t(rank),
                           uint32_t(ranks),
                           device_pointer<int32_t>(indices),
                           uint32_t(row_groups)};
    const int sms = num_sms,
              threads = (config.block_m / config.warp_m) * (config.block_n / config.warp_n) * 32;
    if (config.kernel_type == "OverlapPrologue") {
        ChannelOverlapParams params{};
        params.problem_shape = {m, n, k, 1};
        params.mainloop = mainloop;
        params.epilogue = {epilogue, scatter, device_pointer(output)};
        params.scheduler = {layout, uint32_t(m)};
        params.rhs_replicas = device_pointer(rhs_replicas);
        params.rhs_replica_scales = device_pointer<float>(rhs_replica_scales);
        params.num_master_groups = masters;
        scatter_launch<ChannelOverlapScatterTag>("m_grouped_gemm_fp8_channelwise_moe_scatter",
                                                 channel_scatter_family(), config, params, sms, threads);
    } else {
        ChannelDefaultParams params{};
        params.problem_shape = {m, n, k, 1};
        params.mainloop = mainloop;
        params.epilogue = epilogue;
        params.scheduler = {layout, uint32_t(m)};
        params.scatter = scatter;
        params.rhs_replicas = device_pointer(rhs_replicas);
        params.rhs_replica_scales = device_pointer<float>(rhs_replica_scales);
        params.num_master_groups = masters;
        scatter_launch<ChannelDefaultScatterTag>("m_grouped_gemm_fp8_channelwise_moe_scatter",
                                                 channel_scatter_family(), config, params, sms, threads);
    }
}

struct BlockFp8ScatterTag {};
inline void block_scatter_kernel(const ScatterConfig& config,
                                 const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
                                 const torch::Tensor& rhs, const torch::Tensor& rhs_scales,
                                 const torch::Tensor& output, const torch::Tensor& rows,
                                 const torch::Tensor& valid_rows, const torch::Tensor& row_offsets,
                                 const torch::Tensor& packed_src, const torch::Tensor& combine,
                                 const torch::Tensor& peer_offsets, const torch::Tensor& block_info,
                                 int num_rows, int max_tokens, int topk, int rank, int ranks, int num_sms,
                                 const torch::Tensor& rhs_replicas, const torch::Tensor& rhs_replica_scales) {
    const int m = num_rows, n = config.n, k = config.k, groups = config.num_groups;
    if (m == 0)
        return;
    const auto layout = prepare_grouped_nopad_layout(GemmType::MGroupedNoPad, groups, config.block_m, valid_rows, block_info);
    const auto sa = packed_stride(m, k), sb = packed_stride(n, k), sd = packed_stride(m, n);
    const auto problem = cute::make_shape(m, n, k, 1);
    BlockScatterParams params{};
    params.problem_shape = {m, n, k, 1};
    params.mainloop = {device_pointer<cutlass::float_e4m3_t>(lhs),
                       sa,
                       device_pointer<cutlass::float_e4m3_t>(rhs),
                       sb,
                       4,
                       device_pointer<float>(lhs_scales),
                       FP8GemmRuntime::ScaleConfig::tile_atom_to_shape_SFA(problem),
                       device_pointer<float>(rhs_scales),
                       FP8GemmRuntime::ScaleConfig::tile_atom_to_shape_SFB(problem)};
    params.epilogue.base = {{1, 0}, nullptr, sd, device_pointer<cutlass::bfloat16_t>(output), sd};
    params.epilogue.scatter = {{device_pointer(combine), uint32_t(n * 2), uint32_t(max_tokens),
                                uint32_t(topk), device_pointer<int64_t>(packed_src),
                                device_pointer<int32_t>(valid_rows), device_pointer<int64_t>(peer_offsets),
                                uint32_t(rank), uint32_t(ranks), nullptr, uint32_t(groups)},
                               device_pointer<int32_t>(row_offsets)};
    params.epilogue.base_ptr_D = device_pointer(output);
    params.scheduler = {{layout, uint32_t(m)}, device_pointer<int32_t>(row_offsets)};
    params.rhs_replicas = device_pointer(rhs_replicas);
    params.rhs_replica_scales = device_pointer<float>(rhs_replica_scales);
    params.num_master_groups = config.num_master_groups;
    // Chunked BM192 uses the grouped FP8 compiler scheduling policy.
    const std::string entry = config.scratch_rows ? "fp8_grouped_deep_gemm_mega_moe_blockwise_scatter"
                                                 : "fp8_blockwise_mega_moe_gemm2_scatter";
    scatter_launch<BlockFp8ScatterTag>(entry, R"DGSOURCE(
#define PPU_FP8_DEVICE_ONLY
#include <deep_gemm/impls/fp8_gemm_blockwise_scatter.cuh>
using Family = deep_gemm::mega_moe::fp8_blockwise::Gemm2Scatter<{N},{K},{GROUPS},{MASTER_GROUPS},{NUM_STAGES},{BLOCK_M},{SCRATCH_ROWS}>;
using Kernel = typename Family::Kernel;
static_assert(Family::kNumThreads == {NUM_THREADS}, "blockwise scatter launch must match its MMA tile");
)DGSOURCE",
                                       config, params, num_sms, config.num_threads);
}

struct Fp4MegaScatterTag {};
struct Fp4MegaScatterWithTsmTag {};

template <typename Epilogue, typename Tag>
inline void fp4_scatter_kernel(
    const ScatterConfig& config, const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
    const torch::Tensor& rhs, const torch::Tensor& rhs_scales, const torch::Tensor& rhs_replicas,
    const torch::Tensor& rhs_replica_scales, const torch::Tensor& output, int num_rows,
    const torch::Tensor& rows, const torch::Tensor& block_info, const torch::Tensor& token_meta,
    const torch::Tensor& combine, int hidden_bytes, int max_tokens, int topk, const torch::Tensor& scales,
    const torch::Tensor& scale_offsets, int scale_tiles, int scale_block_n, int use_side_scale,
    const torch::Tensor& peer_offsets, int rank_idx, int num_ranks,
    const torch::Tensor& valid_rows, const torch::Tensor& src_offsets) {
    const int m = num_rows, n = config.n, k = config.k, groups = config.num_groups;
    if (m == 0)
        return;
    const auto layout = prepare_grouped_nopad_layout(GemmType::MGroupedNoPad, groups, config.block_m,
                                                     rows, block_info);
    const auto sa = packed_stride(m, k), sb = packed_stride(n, k), sc = packed_stride(m, 0),
               sd = packed_stride(m, n);
    const auto sfa = cutlass::make_cute_packed_stride(ScaleStride{}, cute::make_shape(m, ceil_div(k, 32), 1));
    const auto sfb = cutlass::make_cute_packed_stride(ScaleStride{}, cute::make_shape(n, ceil_div(k, 32), 1));
    using Params = Fp4ScatterParams<Epilogue>;
    Params params{};
    params.problem_shape = {m, n, k, 1};
    params.mainloop = {
        {m, n, k}, device_pointer<uint8_t>(lhs),         sa,  device_pointer<uint8_t>(rhs),
        sb,        device_pointer<uint16_t>(lhs_scales), sfa, device_pointer<uint16_t>(rhs_scales),
        sfb};
    params.epilogue = {{1, 0}, nullptr, sc, device_pointer<cutlass::bfloat16_t>(output), sd};
    params.scheduler = {layout, uint32_t(m)};
    params.moe_combine_buf.base = device_pointer(combine);
    params.moe_combine_buf.hidden_bytes = hidden_bytes;
    params.moe_combine_buf.max_tokens_per_rank = max_tokens;
    params.moe_combine_buf.num_topk = topk;
    params.moe_combine_buf.scales = device_pointer<uint64_t>(scales);
    params.moe_combine_buf.scale_tiles = scale_tiles;
    params.moe_combine_buf.scale_block_n = scale_block_n;
    params.moe_combine_buf.use_side_scale = use_side_scale;
    const int rank = rank_idx, ranks = num_ranks;
    TORCH_CHECK(ranks > 0 && ranks <= 64 && rank >= 0 && rank < ranks, "invalid FP4 scatter rank");
    const auto offsets = peer_offsets;
    TORCH_CHECK(!offsets.is_cuda() && offsets.scalar_type() == torch::kInt64 && offsets.is_contiguous() &&
                    offsets.numel() >= ranks,
                "FP4 scatter peer offsets must be a CPU int64 table");
    params.moe_sym_buffer.rank_idx = rank;
    params.moe_sym_buffer.num_ranks = ranks;
    std::copy_n(offsets.data_ptr<int64_t>(), ranks, params.moe_sym_buffer.offsets);
    if (params.moe_combine_buf.use_side_scale) {
        TORCH_CHECK(!scale_offsets.is_cuda() && scale_offsets.scalar_type() == torch::kInt64 &&
                        scale_offsets.is_contiguous() && scale_offsets.numel() >= ranks,
                    "FP4 side-scale offsets must be a CPU int64 table");
        std::copy_n(scale_offsets.data_ptr<int64_t>(), ranks, params.moe_combine_buf.scale_offsets);
    }
    params.moe_token_meta = device_pointer<int64_t>(token_meta);
    params.moe_valid_m_rows = device_pointer<int32_t>(valid_rows);
    params.moe_src_m_offsets = device_pointer<int32_t>(src_offsets);
    params.ptr_B_replica = device_pointer<uint8_t>(rhs_replicas);
    params.ptr_scale_B_replica = device_pointer<uint16_t>(rhs_replica_scales);
    params.num_master_groups = config.num_master_groups;
    const int threads = (config.block_m / config.warp_m) * (config.block_n / config.warp_n) * 32;
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
        {"ptr_B_replica", offsetof(Params, ptr_B_replica)},
        {"ptr_scale_B_replica", offsetof(Params, ptr_scale_B_replica)},
        {"num_master_groups", offsetof(Params, num_master_groups)},
        {"ptr_sync_ce", offsetof(Params, ptr_sync_ce)},
        {"moe_combine_buf", offsetof(Params, moe_combine_buf)},
        {"moe_sym_buffer", offsetof(Params, moe_sym_buffer)},
        {"moe_token_meta", offsetof(Params, moe_token_meta)},
        {"moe_valid_m_rows", offsetof(Params, moe_valid_m_rows)},
        {"moe_src_m_offsets", offsetof(Params, moe_src_m_offsets)},
    };
    for (const auto& [field, offset] : abi_offsets)
        body += fmt::format("static_assert(offsetof(Kernel::Params, {}) == {}, "
                            "\"FP4 scatter {} offset mismatch\");\n",
                            field, offset, field);
    scatter_launch<Tag>("m_grouped_gemm_fp4_moe_scatter", std::move(body), config, params,
                        device_runtime->get_prop()->multiProcessorCount, threads);
}

inline void fp4_scatter_kernel_dispatch(
    const ScatterConfig& config, const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
    const torch::Tensor& rhs, const torch::Tensor& rhs_scales, const torch::Tensor& rhs_replicas,
    const torch::Tensor& rhs_replica_scales, const torch::Tensor& output, int num_rows,
    const torch::Tensor& rows, const torch::Tensor& block_info, const torch::Tensor& token_meta,
    const torch::Tensor& combine, int hidden_bytes, int max_tokens, int topk, const torch::Tensor& scales,
    const torch::Tensor& scale_offsets, int scale_tiles, int scale_block_n, int use_side_scale,
    const torch::Tensor& peer_offsets, int rank_idx, int num_ranks,
    const torch::Tensor& valid_rows, const torch::Tensor& src_offsets) {
    if (config.n % 2)
        fp4_scatter_kernel<Fp4ScatterEpilogueWithTsm, Fp4MegaScatterWithTsmTag>(
            config, lhs, lhs_scales, rhs, rhs_scales, rhs_replicas, rhs_replica_scales, output, num_rows,
            rows, block_info, token_meta, combine, hidden_bytes, max_tokens, topk, scales, scale_offsets,
            scale_tiles, scale_block_n, use_side_scale, peer_offsets, rank_idx, num_ranks,
            valid_rows, src_offsets);
    else
        fp4_scatter_kernel<FP4GemmRuntime::EpilogueArgsNoTsm, Fp4MegaScatterTag>(
            config, lhs, lhs_scales, rhs, rhs_scales, rhs_replicas, rhs_replica_scales, output, num_rows,
            rows, block_info, token_meta, combine, hidden_bytes, max_tokens, topk, scales, scale_offsets,
            scale_tiles, scale_block_n, use_side_scale, peer_offsets, rank_idx, num_ranks,
            valid_rows, src_offsets);
}

inline torch::Tensor scatter_block_info(const torch::Tensor& rows, int m, int groups, int block_m,
                                       const std::optional<torch::Tensor>& workspace) {
    if (groups < 128)
        return rows;
    const int64_t required = 4LL * (1 + groups + ceil_div(m, block_m));
    // Complete entries reuse workspace storage. Standalone staged calls retain
    // their internal allocation when no workspace was supplied.
    if (!workspace)
        return torch::empty({required}, rows.options());
    TORCH_CHECK(workspace->scalar_type() == torch::kInt32 && workspace->is_contiguous() &&
                    workspace->device() == rows.device() && workspace->numel() >= required,
                "MegaMoE GEMM2 block-info workspace has an invalid device, dtype, layout or size");
    return *workspace;
}

inline void m_grouped_gemm_fp8_blockwise_with_scatter_impl(
    const torch::Tensor& a, const torch::Tensor& sa, const torch::Tensor& b, const torch::Tensor& sb,
    const torch::Tensor& out, const torch::Tensor& padded_rows, const torch::Tensor& valid_rows,
    const torch::Tensor& row_offsets, const torch::Tensor& packed_src_info, const torch::Tensor& combine,
    const torch::Tensor& peer_offsets, const torch::Tensor& rw, const torch::Tensor& rs, int m, int n, int k,
    int groups, int masters, int max_tokens, int topk, int rank, int ranks,
    std::optional<int> expected, const mega_moe_config::Gemm2Tuning& tuning,
    const std::optional<torch::Tensor>& workspace = std::nullopt) {
    const auto c = mega_moe_config::fp8_blockwise_scatter_config(n, k, groups, masters, expected, tuning);
    const auto info = scatter_block_info(padded_rows, m, groups, c.block_m, workspace);
    mega_moe_native::block_scatter_kernel(
        c, a, sa, b, sb, out, padded_rows, valid_rows, row_offsets, packed_src_info, combine, peer_offsets,
        info, m, max_tokens, topk, rank, ranks, get_num_sms(), rw, rs);
}

inline void m_grouped_gemm_fp8_channelwise_with_scatter_impl(
    const torch::Tensor& lhs, const torch::Tensor& sa, const torch::Tensor& rhs, const torch::Tensor& sb,
    const torch::Tensor& rw, const torch::Tensor& rs, const torch::Tensor& m_indices,
    const torch::Tensor& token_meta, const torch::Tensor& combine_buf, const torch::Tensor& sym_offsets,
    const torch::Tensor& physical, const torch::Tensor& valid, int m, int n,
    int k, int groups, int masters, int world_size, int rank, int max_tokens, int hidden, int num_topk,
    bool checked, std::optional<mega_moe_config::Config> configs,
    std::optional<std::string> kernel_type_override, std::optional<bool> rolling_pair_scatter,
    std::optional<int> num_sms_override, const mega_moe_config::Gemm2Tuning& tuning = {},
    const std::optional<torch::Tensor>& workspace = std::nullopt) {
    auto [config, c] = mega_moe_config::channel_scatter_policy(
        m, n, k, groups, masters, world_size, max_tokens, num_topk,
        valid.data_ptr() != physical.data_ptr(), configs, kernel_type_override,
        rolling_pair_scatter, num_sms_override, tuning);
    const auto [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    c.check_packed_routes = checked;
    const auto info = scatter_block_info(physical, m, groups, bm, workspace);
    const auto out = combine_buf.reshape({-1}).narrow(0, 0, n).view({1, n});
    mega_moe_native::channel_scatter_kernel(c, lhs, sa, rhs, sb, rw, rs, out, m, physical, info, sms,
                                            token_meta, combine_buf, hidden * 2, max_tokens, num_topk,
                                            sym_offsets, rank, world_size, valid, m_indices, groups);
}

static torch::Tensor zero_pointer_table(const torch::Tensor& reference, int size) {
    static std::map<std::pair<int, int>, torch::Tensor> cache;
    static std::mutex mutex;
    const std::lock_guard<std::mutex> guard(mutex);
    auto& tensor = cache[{reference.get_device(), size}];
    if (!tensor.defined())
        tensor = torch::zeros({size}, reference.options().dtype(torch::kInt64));
    return tensor;
}

inline void m_grouped_gemm_fp4_with_scatter_impl(
    const torch::Tensor& lhs, const torch::Tensor& sa, const torch::Tensor& rhs, const torch::Tensor& sb,
    const torch::Tensor& replica, const torch::Tensor& rs, const torch::Tensor& physical,
    const torch::Tensor& valid, const torch::Tensor& token_meta, const torch::Tensor& combine_buf,
    const torch::Tensor& sym_offsets, int m, int n, int k, int groups, int masters, int rank, int world_size,
    int max_tokens, int hidden, int num_topk, bool provided_rows, bool fp8, int shift,
    std::optional<mega_moe_config::Config> configs, std::optional<int> expected_m,
    std::optional<torch::Tensor> combine_scales,
    std::optional<torch::Tensor> scale_offsets, int scale_block_n,
    const torch::Tensor& src_m_offsets, const mega_moe_config::Gemm2Tuning& tuning = {},
    const std::optional<torch::Tensor>& workspace = std::nullopt) {
    const int em = expected_m ? std::max(1, *expected_m) : ceil_div(m, groups);
    const auto cfg =
        mega_moe_config::fp4_scatter_config(m, n, k, groups, configs, provided_rows, false, fp8, em);
    const auto [sms, bm, bn, bk, wm, wn, stages, smem] = cfg;
    const auto info = scatter_block_info(physical, m, groups, bm, workspace);
    const auto out = fp8 ? torch::empty({m, n}, lhs.options().dtype(torch::kBFloat16))
                         : combine_buf.reshape({-1}).narrow(0, 0, n).view({1, n});
    const bool side = fp8 && combine_scales.has_value();
    int side_mode = 0, tiles = 1;
    if (!scale_block_n)
        scale_block_n = bn;
    torch::Tensor cs, so;
    if (side) {
        side_mode = get_env<std::string>("DG_MOE_SCATTER_FP8_SIDE_SCALE_DYNAMIC", "0") == "1" ? 2 : 1;
        TORCH_CHECK(combine_scales->is_cuda() && combine_scales->scalar_type() == torch::kInt64 &&
                    combine_scales->is_contiguous());
        TORCH_CHECK(scale_offsets && scale_offsets->scalar_type() == torch::kInt64 &&
                    scale_offsets->numel() >= world_size);
        TORCH_CHECK(scale_block_n >= 64 && scale_block_n <= bn && bn % scale_block_n == 0 &&
                    scale_block_n % 8 == 0 && bn / scale_block_n <= 4,
                    "FP8 scatter scale_block_n must divide BLOCK_N into at most four groups, "
                    "each aligned to 8 and at least 64 columns");
        tiles = ceil_div(hidden, scale_block_n);
        TORCH_CHECK(combine_scales->numel() >= int64_t(num_topk) * max_tokens * tiles);
        cs = *combine_scales;
        so = *scale_offsets;
        shift = 0;
    } else {
        cs = zero_pointer_table(out, 1);
        so = zero_pointer_table(out, world_size);
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
    c.num_master_groups = masters;
    c.num_stages = stages;
    c.fp8_output = fp8;
    c.fp8_exp_shift = shift;
    c.m_tiles_per_group = tuning.m_tiles_per_group.value_or(2);
    TORCH_CHECK(c.m_tiles_per_group == 1 || c.m_tiles_per_group == 2,
                "W4A4 scatter requires one or two M tiles per scheduler group");
    c.fp8_scale_mode = !side ? mega_moe::Fp8ScaleMode::Fixed
        : side_mode == 2 ? mega_moe::Fp8ScaleMode::SideDynamic : mega_moe::Fp8ScaleMode::SideStatic;

    mega_moe_native::fp4_scatter_kernel_dispatch(
        c, lhs, sa, rhs, sb, replica, rs, out, m, physical, info, token_meta, combine_buf,
        fp8 ? hidden : hidden * 2, max_tokens, num_topk, cs, so, tiles, scale_block_n, side_mode, sym_offsets,
        rank, world_size, valid, src_m_offsets);
}

inline void m_grouped_gemm_w4a16_mxfp4_with_scatter_impl(
    const torch::Tensor& lhs, const torch::Tensor& rhs, const torch::Tensor& scales,
    const torch::Tensor& combine, const torch::Tensor& rows, const torch::Tensor& packed_src_info,
    const torch::Tensor& sym_offsets, const torch::Tensor& valid_m_rows, int rank, int max_tokens, int topk,
    std::optional<PPUW4ConfigTuple> requested,
    const std::optional<std::pair<torch::Tensor, torch::Tensor>>& replicas,
    std::optional<int> expected, const mega_moe_config::Gemm2Tuning& tuning = {},
    const std::optional<torch::Tensor>& workspace = std::nullopt) {
    constexpr auto type = GemmType::MGroupedNoPad;
    constexpr int group_size = 32;
    const int m = lhs.size(0), k = lhs.size(1), n = rhs.size(1), master_groups = rhs.size(0);
    const auto replica_rhs = replicas ? replicas->first : rhs;
    const auto replica_scales = replicas ? replicas->second : scales;
    const int groups = master_groups + (replicas ? replica_rhs.size(0) : 0);
    const int expected_m = expected.value_or(ceil_div(m, groups));
    TORCH_CHECK(expected_m >= 0);
    const auto selected = mega_moe_config::w4a16_scatter_config(expected_m, n, k, requested, tuning);
    const PPUW4ConfigTuple selected_tuple{selected.num_sms, selected.block_m,   selected.block_n,
                                          selected.block_k, selected.warp_m,    selected.warp_n,
                                          selected.warp_k,  selected.num_stages};
    const auto block_info = scatter_block_info(rows, m, groups, selected.block_m, workspace);
    auto c = ppu_w4_config(expected_m, n, k, selected_tuple);
    TORCH_CHECK(c.warp_n == 64 and c.warp_k % 64 == 0, "Invalid W4A16 warp shape");
    TORCH_CHECK(c.block_k == c.warp_k and c.block_n % 256 == 0 and n % c.block_n == 0,
                "W4 scatter requires a single K warp and complete N tiles");
    W4A16ScatterLaunchInfo info{c,
                                type,
                                n,
                                k,
                                groups,
                                group_size,
                                deep_gemm_w4a16_common::get_num_threads(c, false),
                                get_smem_config(c, true, 1, group_size, true),
                                "ppu_w4a16_mxfp4_scatter"};
    W4A16ScatterArguments<TileSchedulerArguments> params{};
    params.ptr_a = lhs.data_ptr();
    params.ptr_b = rhs.data_ptr();
    params.ptr_scale = scales.data_ptr();
    params.ptr_b_replicas = replica_rhs.data_ptr();
    params.ptr_scale_replicas = replica_scales.data_ptr();
    params.num_master_groups = master_groups;
    params.ptr_d = combine.data_ptr();
    params.num_token = m;
    params.packed_src_info = packed_src_info.data_ptr<int64_t>();
    params.sym_offsets = sym_offsets.data_ptr<int64_t>();
    params.valid_m_rows = valid_m_rows.data_ptr<int32_t>();
    params.rank = rank;
    params.max_tokens = max_tokens;
    params.topk = topk;
    params.scheduler = {static_cast<uint32_t>(m),
                        prepare_grouped_nopad_layout(type, groups, c.block_m, rows, block_info)};
    launch_w4a16_scatter(info, params, m, expected_m, rows);
}

} // namespace deep_gemm::mega_moe_native
