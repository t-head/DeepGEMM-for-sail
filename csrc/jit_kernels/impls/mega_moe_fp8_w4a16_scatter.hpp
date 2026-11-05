#pragma once
#include "fp8_gemm.hpp"
#include "int8_gemm.hpp"
#include "mega_moe_fp8_w4a16_dispatch.hpp"
#include "fp4_gemm.hpp"
#include <cstddef>

namespace deep_gemm::mega_moe_fp8_w4a16 {
using PackedStride = cute::Stride<int64_t, cute::Int<1>, int64_t>;
using Problem = FP4GemmRuntime::GemmProblemSize;
using Mode = FP4GemmRuntime::GemmUniversalMode;

using ScatterScheduler = TileSchedulerArguments;

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
    const int32_t* src_m_offsets = nullptr;
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
    const int32_t* src_m_offsets = nullptr;
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

template <typename Tag, typename Params>
class FP8ScatterRuntime final : public LaunchRuntime<FP8ScatterRuntime<Tag, Params>> {
  public:
    struct Args {
        std::string source;
        LaunchArgs launch_args;
        Params kernel_params;
    };
    static std::string generate_impl(const Args& args) { return args.source; }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, const Args& args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};
inline PackedStride packed_stride(int m_rows, int cols) {
    return cutlass::make_cute_packed_stride(PackedStride{}, cute::make_shape(m_rows, cols, 1));
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
    for (const auto& [field, value] : source_fields(config))
        replace_source_field(body, field, value);
    return body;
}

template <typename Tag, typename Params>
inline void scatter_launch(const std::string& name, std::string body, const ScatterConfig& config, Params params,
                           int num_ctas, int threads) {
    TORCH_CHECK(num_ctas > 0 && threads > 0 && threads <= 1024, "invalid scatter geometry");
    using Runtime = FP8ScatterRuntime<Tag, Params>;
    typename Runtime::Args args{
        scatter_source<Params>(std::move(body), config, name), {dim3(num_ctas), dim3(threads), 0}, params};
    const auto runtime = compiler->build(name, Runtime::generate(args));
    args.launch_args.smem_size = mega_moe_runtime::get_kernel_smem_size(runtime);
    const int occupancy = mega_moe_runtime::get_kernel_occupancy(runtime, threads, args.launch_args.smem_size);
    TORCH_CHECK(occupancy > 0, "scatter cannot keep a CTA resident");
    args.launch_args.grid_dim.x *= occupancy;
    args.kernel_params.hw_info = {0, num_ctas * occupancy};
    Runtime::launch(runtime, args);
}

inline std::string channel_scatter_family() {
    return R"DGSOURCE(
#include <deep_gemm/impls/fp8_gemm_channelwise_scatter.cuh>
using namespace deep_gemm;
using Scatter = mega_moe::detail::Bf16MoeScatter<{USE_PADDED_VALID_ROWS}, {CHECK_PACKED_ROUTES}>;
using Family = ChannelwiseFp8GemmWithScatter<cutlass::float_e4m3_t,float,
    {N},{K},{BLOCK_M},{BLOCK_N},{BLOCK_K},{WARP_M},{WARP_N},{NUM_GROUPS},{NUM_STAGES},
    GemmType::MGroupedNoPad,false,KernelStrategy::{KERNEL_TYPE},Scatter>;
using Kernel = typename Family::GemmKernel;
)DGSOURCE";
}

struct Fp8ChannelwiseDefaultScatterTag {};
struct Fp8ChannelwiseOverlapScatterTag {};

inline int32_t* prepare_channelwise_scatter_layout(const ScatterConfig& config,
                                                  const torch::Tensor& rows,
                                                  const torch::Tensor& valid_rows,
                                                  const torch::Tensor& block_info) {
    if (channel_scatter_uses_worklist(config))
        return block_info.data_ptr<int32_t>();
    if (config.num_groups < 128 && config.use_padded_valid_rows && config.kernel_type == "Default")
        return valid_rows.data_ptr<int32_t>();
    return prepare_scatter_layout(config.num_groups, config.block_m, rows, block_info);
}

inline void fp8_channelwise_mega_moe_scatter_kernel(const ScatterConfig& config, const torch::Tensor& lhs,
                                   const torch::Tensor& lhs_scales, const torch::Tensor& rhs,
                                   const torch::Tensor& rhs_scales, const torch::Tensor& rhs_replicas,
                                   const torch::Tensor& rhs_replica_scales, const torch::Tensor& output, int num_rows,
                                   const torch::Tensor& m_rows, const torch::Tensor& block_info, int num_ctas,
                                   const torch::Tensor& token_meta, const torch::Tensor& combine, int hidden_bytes,
                                   int num_max_tokens_per_rank, int num_topk, const torch::Tensor& peer_offsets, int rank_idx, int num_ranks,
                                   const torch::Tensor& valid_m_rows, const torch::Tensor& src_offsets,
                                   const torch::Tensor& indices, int row_groups,
                                   const std::string& entry) {
    const int m = num_rows, n = config.n, k = config.k;
    const int groups = config.num_groups, masters = config.num_master_groups;
    if (m == 0)
        return;
    const auto layout = prepare_channelwise_scatter_layout(config, m_rows, valid_m_rows, block_info);
    const auto sa = packed_stride(m, k), sb = packed_stride(n, k), sd = packed_stride(m, n);
    DenseINT8GemmCutlass3Runtime::MainLoopArguments mainloop{device_pointer(lhs),
                                                             sa,
                                                             device_pointer(rhs),
                                                             sb,
                                                             device_pointer<float>(lhs_scales),
                                                             device_pointer<float>(rhs_scales)};
    DenseINT8GemmCutlass3Runtime::EpilogueArgs epilogue{
        {1, 0}, device_pointer<cutlass::bfloat16_t>(output), sd, device_pointer<cutlass::bfloat16_t>(output), sd};
    ChannelScatter scatter{device_pointer(combine),
                           uint32_t(hidden_bytes),
                           uint32_t(num_max_tokens_per_rank),
                           uint32_t(num_topk),
                           device_pointer<int64_t>(token_meta),
                           device_pointer<int32_t>(valid_m_rows),
                           device_pointer<int64_t>(peer_offsets),
                           uint32_t(rank_idx),
                           uint32_t(num_ranks),
                           device_pointer<int32_t>(indices),
                           uint32_t(row_groups)};
    const int threads = (config.block_m / config.warp_m) * (config.block_n / config.warp_n) * 32;
    if (config.kernel_type == "OverlapPrologue") {
        ChannelOverlapParams params{};
        params.problem_shape = {m, n, k, 1};
        params.mainloop = mainloop;
        params.epilogue = {epilogue, scatter, device_pointer(output)};
        params.scheduler = {uint32_t(m), layout};
        params.rhs_replicas = device_pointer(rhs_replicas);
        params.rhs_replica_scales = device_pointer<float>(rhs_replica_scales);
        params.num_master_groups = masters;
        scatter_launch<Fp8ChannelwiseOverlapScatterTag>(entry,
                                                 channel_scatter_family(), config, params, num_ctas, threads);
    } else {
        ChannelDefaultParams params{};
        params.problem_shape = {m, n, k, 1};
        params.mainloop = mainloop;
        params.epilogue = epilogue;
        params.scheduler = {uint32_t(m), layout};
        params.scatter = scatter;
        params.src_m_offsets = device_pointer<int32_t>(src_offsets);
        params.rhs_replicas = device_pointer(rhs_replicas);
        params.rhs_replica_scales = device_pointer<float>(rhs_replica_scales);
        params.num_master_groups = masters;
        scatter_launch<Fp8ChannelwiseDefaultScatterTag>(entry,
                                                 channel_scatter_family(), config, params, num_ctas, threads);
    }
}

struct Fp8BlockwiseScatterTag {};
inline void fp8_blockwise_mega_moe_scatter_kernel(const ScatterConfig& config, const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
                     const torch::Tensor& rhs, const torch::Tensor& rhs_scales, const torch::Tensor& output,
                     const torch::Tensor& m_rows, const torch::Tensor& valid_m_rows, const torch::Tensor& src_m_offsets,
                     const torch::Tensor& token_meta, const torch::Tensor& combine, const torch::Tensor& peer_offsets,
                     const torch::Tensor& block_info, int num_rows, int num_max_tokens_per_rank, int num_topk, int rank_idx, int num_ranks,
                     int num_ctas, const torch::Tensor& rhs_replicas, const torch::Tensor& rhs_replica_scales,
                     const std::string& entry) {
    const int m = num_rows, n = config.n, k = config.k, groups = config.num_groups;
    if (m == 0)
        return;
    const auto layout = prepare_scatter_layout(groups, config.block_m, valid_m_rows, block_info);
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
    params.epilogue.scatter = {{device_pointer(combine), uint32_t(n * 2), uint32_t(num_max_tokens_per_rank), uint32_t(num_topk),
                                device_pointer<int64_t>(token_meta), device_pointer<int32_t>(valid_m_rows),
                                device_pointer<int64_t>(peer_offsets), uint32_t(rank_idx), uint32_t(num_ranks), nullptr,
                                uint32_t(groups)},
                               device_pointer<int32_t>(src_m_offsets)};
    params.epilogue.base_ptr_D = device_pointer(output);
    params.scheduler = {{uint32_t(m), layout}, device_pointer<int32_t>(src_m_offsets)};
    params.rhs_replicas = device_pointer(rhs_replicas);
    params.rhs_replica_scales = device_pointer<float>(rhs_replica_scales);
    params.num_master_groups = config.num_master_groups;
    scatter_launch<Fp8BlockwiseScatterTag>(entry, R"DGSOURCE(
#include <deep_gemm/impls/fp8_gemm_blockwise_scatter.cuh>
using Family = deep_gemm::mega_moe::fp8_blockwise::Gemm2Scatter<{N},{K},{GROUPS},{MASTER_GROUPS},{NUM_STAGES},{BLOCK_M},{SCRATCH_ROWS}>;
using Kernel = typename Family::Kernel;
static_assert(Family::kNumThreads == {NUM_THREADS}, "blockwise scatter launch must match its MMA tile");
static_assert(Kernel::SharedStorageSize == {SMEM_SIZE}, "blockwise scatter shared-memory estimate must match its arena");
)DGSOURCE",
                                       config, params, num_ctas, config.num_threads);
}

} // namespace deep_gemm::mega_moe_fp8_w4a16
