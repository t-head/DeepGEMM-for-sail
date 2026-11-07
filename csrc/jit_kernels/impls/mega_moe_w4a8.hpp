#pragma once

#include "mega_moe_fp8_w4a16.hpp"

namespace deep_gemm::mega_moe_w4a8 {
using namespace mega_moe_fp8_w4a16;
using DispatchConfig = mega_moe_fp8_w4a16::DispatchConfig;
using ScatterConfig = mega_moe_fp8_w4a16::ScatterConfig;

using W4A8Weights = std::tuple<torch::Tensor, torch::Tensor, float>;

class PPUW4A8MegaMoERuntime final : public LaunchRuntime<PPUW4A8MegaMoERuntime> {
public:
    enum class KernelPhase { Linear1, Linear2 };
    struct Args;
    static void compile_and_launch(const std::string& name, const Args& args);
    template <typename Tag, typename Params>
    static std::string generate_impl(const PrecisionLaunchArgs<Tag, Params>& args) { return args.source; }
    template <typename Tag, typename Params>
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config,
                            const PrecisionLaunchArgs<Tag, Params>& args) {
        std::apply([&](const auto&... p) { DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, p...)); },
                   args.kernel_params);
    }
};
struct W4A8DispatchTag { using Runtime = PPUW4A8MegaMoERuntime; };
struct W4A8ScatterTag { using Runtime = PPUW4A8MegaMoERuntime; };

inline void w4a8_source_policy(std::string& source, const w4fa8::NativeKernelConfig& kernel) {
    replace_source_field(source, "FRAGMENT_POLICY",
                         kernel.fragment == w4fa8::FragmentPolicy::FullK ? "FullK" : "RollingK32");
    replace_source_field(source, "A_LOAD_POLICY",
                         kernel.a_load() == w4fa8::ALoadPolicy::Bulk ? "Bulk" : "Ring2");
}

inline void w4a8_dispatch_gemm1_impl(const DispatchConfig& c, const RoutedMoEWorkspace& buffer,
                                    const w4fa8::NativeKernelConfig& kernel,
                                    const W4A8Weights& weights, const W4A8Weights& replicas,
                                    int num_tokens, bool situ,
                                    float situ_beta, float situ_linear_beta, const std::string& name) {
    auto source = std::string(R"(
#include <deep_gemm/impls/w4a8_mega_moe.cuh>
#include <deep_gemm/impls/fp8_channelwise_mega_moe_dispatch_gemm1.cuh>
using namespace deep_gemm;
using Backend = mega_moe::w4a8::Gemm1<{BLOCK_M}, {GEMM1_BLOCK_N}, {TOKEN_BYTES},
    {GEMM1_WARP_M}, {GEMM1_WARP_N}, {GEMM1_STAGES}, {SWIGLU_OAI}, {SITU},
    w4fa8::FragmentPolicy::{FRAGMENT_POLICY}, w4fa8::ALoadPolicy::{A_LOAD_POLICY}>;
constexpr uint32_t compute_bytes = sizeof(typename Backend::Kernel::SharedStorage);
constexpr uint32_t dispatch_bytes = (({NUM_EXPERTS} * 4 + 15) & ~15u) + {NUM_THREADS} / 32 * {TOKEN_BYTES};
extern "C" __device__ __constant__ uint32_t ppu_mega_moe_smem_bytes =
    compute_bytes > dispatch_bytes ? compute_bytes : dispatch_bytes;
extern "C" __global__ __launch_bounds__({NUM_THREADS}, 1)
void PPU_KERNEL_ENTRY(layout::SymBuffer<> sb, int tokens,
    const uint8_t* w, const uint8_t* s, float inv, const uint8_t* rw, const uint8_t* rs, float rinv,
    int masters, __ppu_bfloat16* out, float limit, uint4* gemm2_block_info) {
    auto p = [&](int64_t off) { return reinterpret_cast<uint8_t*>(sb.base + off); };
    mega_moe::fp8_channelwise::dispatch_gemm1_device<
        {NUM_RANKS}, {NUM_CTAS}, {NUM_THREADS}, {NUM_EXPERTS}, {NUM_EXPERTS_PER_RANK},
        {NUM_TOPK}, {BLOCK_M}, {TOKEN_BYTES}, {GEMM1_N}, {GEMM1_BLOCK_N},
        128, {GEMM1_WARP_M}, {GEMM1_WARP_N}, {GEMM1_STAGES}, {PULL_CTAS}, {SWIGLU_OAI}, Backend,
        {PULL_GROUP_SIZE}, {GEMM2_BLOCK_M}, {GEMM2_VALID_ROWS}>(
        p(0), sb, reinterpret_cast<int64_t*>(p({LAYOUT_OFF_ITI})),
        p({LAYOUT_OFF_IT}),
        reinterpret_cast<float*>(p({LAYOUT_OFF_ISF})), p({LAYOUT_OFF_PT}),
        reinterpret_cast<float*>(p({LAYOUT_OFF_PSF})), reinterpret_cast<int32_t*>(p({LAYOUT_OFF_GEMM2_META})), gemm2_block_info,
        reinterpret_cast<const cutlass::float_e4m3_t*>(w), reinterpret_cast<const float*>(s),
        reinterpret_cast<const cutlass::float_e4m3_t*>(rw), reinterpret_cast<const float*>(rs),
        masters, out, limit, {ACTIVATION_ALPHA}, tokens, {NMTPR}, inv, rinv, {SITU_BETA}, {SITU_LINEAR_BETA});
}
)");
    w4a8_source_policy(source, kernel);
    replace_source_field(source, "SITU", situ ? "true" : "false");
    replace_source_field(source, "SITU_BETA", source_float_literal(situ_beta));
    replace_source_field(source, "SITU_LINEAR_BETA", source_float_literal(situ_linear_beta));
    const auto& [w, s, inv] = weights;
    const auto& [rw, rs, rinv] = replicas;
    launch<W4A8DispatchTag>(name, source, c,
        std::make_tuple(buffer.sym_buffer, int32_t(num_tokens), w.data_ptr(), s.data_ptr(), inv,
                        rw.data_ptr(), rs.data_ptr(), rinv, int32_t(w.size(0)),
                        buffer.bf16_scratch.data_ptr(), c.activation_clamp, device_pointer(buffer.block_info)),
        dim3(c.num_ctas), c.num_threads, false, true);
}

inline void w4a8_scatter_impl(const w4fa8::NativeKernelConfig& kernel, const RoutedMoEWorkspace& buffer,
                              const W4A8Weights& weights, const W4A8Weights& replicas,
                              int groups, int capacity, int rank_idx, int num_ranks, const std::string& name) {
    const auto& tile = kernel.tile;
    const auto& [w, s, inv] = weights;
    const auto& [rw, rs, rinv] = replicas;
    auto* rows = scatter_layout(groups, buffer.m_rows, buffer.block_info);
    ScatterConfig c;
    c.num_ranks = num_ranks;
    c.n = w.size(1); c.k = buffer.l2_acts.size(1); c.num_groups = groups;
    c.block_m = tile.block_m; c.block_n = tile.block_n; c.block_k = tile.block_k;
    c.warp_m = tile.warp_m; c.warp_n = tile.warp_n; c.num_stages = tile.num_stages;
    c.num_threads = w4fa8::get_num_threads(tile);
    auto source = std::string(R"(
#include <deep_gemm/impls/w4a8_mega_moe.cuh>
#include <deep_gemm/scheduler/sm89_mega_moe.cuh>
using namespace deep_gemm;
using Kernel = cutlass::gemm::kernel::W4A8GEMM_MMA<
    {N}, {K}, {BLOCK_M}, {BLOCK_N}, 128, {WARP_M}, {WARP_N}, 128,
    {NUM_GROUPS}, {NUM_STAGES}, GemmType::MGroupedNoPad, 1, false,
    w4fa8::FragmentPolicy::{FRAGMENT_POLICY}, w4fa8::ALoadPolicy::{A_LOAD_POLICY}>;
using Epilogue = mega_moe::w4a8::Scatter<{N}, {K}, {BLOCK_M}, {BLOCK_N}>;
extern "C" __device__ __constant__ uint32_t ppu_mega_moe_smem_bytes =
    Epilogue::SharedBytes > sizeof(Kernel::SharedStorage) ? Epilogue::SharedBytes : sizeof(Kernel::SharedStorage);
extern "C" __global__ __launch_bounds__({NUM_THREADS}, 1)
void PPU_KERNEL_ENTRY(const cutlass::float_e4m3_t* a, const float* a_scale,
    const uint8_t* w, const uint8_t* s, float inv, const uint8_t* rw, const uint8_t* rs, float rinv,
    int masters, int32_t* rows, const int32_t* valid, const int64_t* routes,
    __ppu_bfloat16* output, const int64_t* offsets, int capacity, int pool, int self_rank) {
    extern __shared__ char smem[];
    Kernel::Params params{};
    params.ptr_a = a; params.ptr_b = w; params.ptr_scale = s;
    params.ptr_d = reinterpret_cast<cutlass::bfloat16_t*>(output);
    params.ptr_a_scale = a_scale;
    params.scheduler = TileSchedulerArguments(pool, rows);
    sched::RoutedMegaMoEScheduler<{N}, {K}, {BLOCK_M}, {BLOCK_N}, {NUM_GROUPS},
        {NUM_RANKS}, ({BLOCK_M} == 16 ? 1 : 2)> scheduler(params.scheduler);
    const Epilogue epilogue{
        output, a_scale, valid, routes, offsets, rw, rs, masters, capacity, inv, rinv, smem, self_rank};
    Kernel{}.run(params, smem, scheduler, epilogue);
#if __HGGC_ARCH__ == 150
    // Complete every producer's stores before combine publishes this rank.
    __ppu_wait_remote_vstcnt(0);
    __ppu_wait_vstcnt(0);
#endif
}
)");
    w4a8_source_policy(source, kernel);
    launch<W4A8ScatterTag>(name, source, c,
        std::make_tuple(buffer.l2_acts.data_ptr(), buffer.l2_acts_sf.data_ptr(), w.data_ptr(), s.data_ptr(), inv,
        rw.data_ptr(), rs.data_ptr(), rinv, int32_t(w.size(0)), rows, buffer.valid_m_rows.data_ptr(),
        buffer.token_meta.data_ptr(), buffer.combine.data_ptr(), buffer.peer_offsets.data_ptr(),
        int32_t(capacity), int32_t(buffer.l2_acts.size(0)), int32_t(rank_idx)),
        dim3(tile.num_sms), c.num_threads, true);
}

struct PPUW4A8MegaMoERuntime::Args {
    KernelPhase kernel_phase;
    const RoutedMoEWorkspace& buffer;
    const torch::Tensor& y;
    DispatchConfig config;
    const w4fa8::NativeKernelConfig& gemm1;
    const w4fa8::NativeKernelConfig& scatter;
    const W4A8Weights& l1_weights;
    const W4A8Weights& l2_weights;
    const W4A8Weights& l1_weights_replicas;
    const W4A8Weights& l2_weights_replicas;
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats;
    int rank_idx, num_tokens;
    bool situ;
    float situ_beta, situ_linear_beta;
};

inline void PPUW4A8MegaMoERuntime::compile_and_launch(const std::string& name, const Args& args) {
    if (args.kernel_phase == KernelPhase::Linear1) {
        w4a8_dispatch_gemm1_impl(args.config, args.buffer, args.gemm1, args.l1_weights, args.l1_weights_replicas,
            args.num_tokens, args.situ, args.situ_beta, args.situ_linear_beta, name);
        fp8_channelwise_mega_moe_quantize_local_pool_kernel(args.config, args.buffer.sym_buffer,
            args.rank_idx, args.buffer.bf16_scratch, args.buffer.l2_acts, args.buffer.l2_acts_sf);
        update_recv_stats(args.buffer, args.cumulative_local_expert_recv_stats);
        return;
    }
    TORCH_CHECK(args.kernel_phase == KernelPhase::Linear2, "Invalid MegaMoE kernel phase");
    w4a8_scatter_impl(args.scatter, args.buffer, args.l2_weights, args.l2_weights_replicas,
                     args.config.num_experts_per_rank, args.config.nmtpr, args.rank_idx, args.config.num_ranks, name);
    combine_mega_moe(args.buffer, args.config, args.y, args.rank_idx, args.num_tokens);
}

inline void fp8_fp4_mega_moe_impl(
    const torch::Tensor& y, const W4A8Weights& l1_weights, const W4A8Weights& l2_weights,
    const W4A8Weights& l1_weights_replicas, const W4A8Weights& l2_weights_replicas,
    const RoutedMoEWorkspace& buffer, int rank_idx, int num_tokens, const std::string& activation,
    float activation_clamp, bool fast_math, float activation_alpha, float situ_beta, float situ_linear_beta,
    const RoutedMoEConfigs& configs, const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats) {
    auto c = configs.dispatch;
    c.fast_math = fast_math; c.swiglu_oai = activation == "swiglu_oai";
    c.activation_alpha = activation_alpha; c.activation_clamp = activation_clamp;
    auto args = PPUW4A8MegaMoERuntime::Args{PPUW4A8MegaMoERuntime::KernelPhase::Linear1,
        buffer, y, c, configs.w4a8_gemm1, configs.w4a8_scatter,
        l1_weights, l2_weights, l1_weights_replicas, l2_weights_replicas,
        cumulative_local_expert_recv_stats, rank_idx, num_tokens, activation == "situ", situ_beta, situ_linear_beta};
    const auto launch_with_phase = [&](PPUW4A8MegaMoERuntime::KernelPhase phase, const char* name) {
        args.kernel_phase = phase;
        PPUW4A8MegaMoERuntime::compile_and_launch(name, args);
    };
    launch_with_phase(PPUW4A8MegaMoERuntime::KernelPhase::Linear1, "w4a8_mega_moe_dispatch_gemm1");
    launch_with_phase(PPUW4A8MegaMoERuntime::KernelPhase::Linear2, "m_grouped_gemm_w4a8_moe_scatter");
}

} // namespace deep_gemm::mega_moe_w4a8
