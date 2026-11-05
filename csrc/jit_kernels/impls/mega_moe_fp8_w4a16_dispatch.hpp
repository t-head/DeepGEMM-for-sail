#pragma once

#include "../heuristics/mega_moe_fp8_w4a16.hpp"
#include "mega_moe_dispatch.hpp"

namespace deep_gemm::mega_moe_fp8_w4a16 {
using namespace mega_moe_config::fp8_w4a16;
using DispatchConfig = mega_moe_config::fp8_w4a16::DispatchConfig;
using PreDispatchConfig = mega_moe_config::fp8_w4a16::PreDispatchConfig;
using ScatterConfig = mega_moe_config::fp8_w4a16::ScatterConfig;
using deep_gemm::device_pointer;
using deep_gemm::replace_source_field;
using deep_gemm::source_float_literal;
using SourceFields = deep_gemm::SourceFields;

inline SourceFields source_fields(const DispatchConfig& c) {
    SourceFields values{
        {"FAST_MATH", c.fast_math ? "1" : "0"},
        {"GEMM1_COMPUTE_M", std::to_string(c.gemm1_compute_m)},
        {"BLOCK_M", std::to_string(c.block_m)},
        {"CLAMP", c.activation_clamp >= 0 ? "1" : "0"},
        {"GEMM1_BLOCK_K", std::to_string(c.gemm1_block_k)},
        {"GEMM1_BLOCK_N", std::to_string(c.gemm1_block_n)},
        {"GEMM1_N", std::to_string(c.gemm1_n)},
        {"GEMM1_STAGES", std::to_string(c.gemm1_stages)},
        {"GEMM1_WARP_M", std::to_string(c.gemm1_warp_m)},
        {"GEMM1_WARP_N", std::to_string(c.gemm1_warp_n)},
        {"GEMM2_BLOCK_M", std::to_string(c.gemm2_block_m)},
        {"MASTER_EXPERTS", std::to_string(c.master_experts)},
        {"NMTPR", std::to_string(c.nmtpr)},
        {"NUM_EXPERTS", std::to_string(c.num_experts)},
        {"NUM_EXPERTS_PER_RANK", std::to_string(c.num_experts_per_rank)},
        {"NUM_RANKS", std::to_string(c.num_ranks)},
        {"NUM_CTAS", std::to_string(c.num_ctas)},
        {"NUM_THREADS", std::to_string(c.num_threads)},
        {"NUM_TOPK", std::to_string(c.num_topk)},
        {"PULL_SLOTS", std::to_string(c.pull_slots)},
        {"PULL_CTAS", std::to_string(c.pull_ctas)},
        {"PULL_GROUP_SIZE", std::to_string(c.pull_group_size)},
        {"SF_BYTES", std::to_string(c.sf_bytes)},
        {"SWIGLU_ALPHA", source_float_literal(c.swiglu_alpha)},
        {"SWIGLU_OAI", (c.swiglu_oai ? "true" : "false")},
        {"TOKEN_BYTES", std::to_string(c.token_bytes)},
        {"W4_BLOCK_K", std::to_string(c.w4_block_k)},
        {"W4_BLOCK_M", std::to_string(c.w4_block_m)},
        {"W4_BLOCK_N", std::to_string(c.w4_block_n)},
        {"W4_STAGES", std::to_string(c.w4_stages)},
        {"W4_WARP_K", std::to_string(c.w4_warp_k)},
        {"W4_WARP_M", std::to_string(c.w4_warp_m)},
        {"W4_WARP_N", std::to_string(c.w4_warp_n)},
    };
    TORCH_CHECK(c.num_ranks > 0 && c.num_experts % c.num_ranks == 0, "invalid MegaMoE layout");
    const layout::Workspace workspace(nullptr, c.num_ranks, c.num_experts, c.nmtpr, c.num_topk);
    const auto align16 = [](uint64_t bytes) { return (bytes + 15) / 16 * 16; };
    // Match the source and landing token alignment used by symmetric_layout().
    const auto align128 = [](uint64_t bytes) { return (bytes + 127) / 128 * 128; };
    const uint64_t pool = workspace.num_max_pool_tokens, num_max_tokens_per_rank = c.nmtpr;
    const uint64_t it = align128(workspace.get_num_bytes());
    const uint64_t isf = align16(it + num_max_tokens_per_rank * c.token_bytes);
    const uint64_t iti = align16(isf + num_max_tokens_per_rank * c.sf_bytes);
    const uint64_t itw = align16(iti + num_max_tokens_per_rank * c.num_topk * sizeof(int64_t));
    const uint64_t pt = align128(itw + num_max_tokens_per_rank * c.num_topk * sizeof(float));
    const uint64_t psf = align16(pt + pool * c.token_bytes);
    const uint64_t metadata = align16(psf + pool * c.sf_bytes);
    for (const auto& [key, value] : std::vector<std::pair<const char*, uint64_t>>{
             {"LAYOUT_WS_BYTES", workspace.get_num_bytes()},
             {"LAYOUT_NUM_POOL", pool},
             {"LAYOUT_NUM_POOL_BLOCKS", workspace.num_max_pool_blocks},
             {"LAYOUT_OFF_IT", it},
             {"LAYOUT_OFF_ISF", isf},
             {"LAYOUT_OFF_ITI", iti},
             {"LAYOUT_OFF_PT", pt},
             {"LAYOUT_OFF_PSF", psf},
             {"LAYOUT_OFF_GEMM2_META", metadata},
             {"LAYOUT_WOFF_RCS", reinterpret_cast<uintptr_t>(workspace.get_expert_recv_count_sum_ptr(0))}})
        values.emplace_back(key, std::to_string(value));
    return values;
}

inline SourceFields source_fields(const PreDispatchConfig& c) {
    SourceFields values{
        {"BLOCK_THREADS", std::to_string(c.block_threads)},
        {"HIDDEN", std::to_string(c.hidden)},
        {"NUM_CTAS", std::to_string(c.num_ctas)},
        {"TOPK", std::to_string(c.num_topk)},
        {"TOP_K", std::to_string(c.num_topk)},
    };
    return values;
}

inline SourceFields source_fields(const ScatterConfig& c) {
    SourceFields values{
        {"BLOCK_K", std::to_string(c.block_k)},
        {"BLOCK_M", std::to_string(c.block_m)},
        {"BLOCK_N", std::to_string(c.block_n)},
        {"SCRATCH_ROWS", std::to_string(c.scratch_rows)},
        {"SMEM_SIZE", std::to_string(c.smem_size)},
        {"CHECK_PACKED_ROUTES", (c.check_packed_routes ? "true" : "false")},
        {"GROUPS", std::to_string(c.num_groups)},
        {"K", std::to_string(c.k)},
        {"KERNEL_TYPE", c.kernel_type},
        {"MASTER_GROUPS", std::to_string(c.num_master_groups)},
        {"N", std::to_string(c.n)},
        {"NUM_GROUPS", std::to_string(c.num_groups)},
        {"NUM_STAGES", std::to_string(c.num_stages)},
        {"NUM_THREADS", std::to_string(c.num_threads)},
        {"USE_PADDED_VALID_ROWS", (c.use_padded_valid_rows ? "true" : "false")},
        {"WARP_M", std::to_string(c.warp_m)},
        {"WARP_N", std::to_string(c.warp_n)},
    };
    return values;
}

template <typename Tag, typename Params> struct PrecisionLaunchArgs {
    std::string source;
    LaunchArgs launch_args;
    Params kernel_params;
};

class PPUFP8MegaMoERuntime final : public LaunchRuntime<PPUFP8MegaMoERuntime> {
public:
    enum class KernelPhase { Linear1, Linear2 };
    struct Args;
    static void compile_and_launch(const std::string& tag, const Args& args);
    template <typename Tag, typename Params>
    static std::string generate_impl(const PrecisionLaunchArgs<Tag, Params>& args) {
        return args.source;
    }
    template <typename Tag, typename Params>
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config,
                            const PrecisionLaunchArgs<Tag, Params>& args) {
        std::apply([&](const auto&... p) { DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, p...)); },
                   args.kernel_params);
    }
};

template <typename Tag, typename Config, typename... Params>
inline void launch(const std::string& name, const std::string& source, const Config& config,
                   const std::tuple<Params...>& params, dim3 grid, int threads,
                   bool occupancy_scaled = false, bool cooperative = false) {
    using Runtime = typename Tag::Runtime;
    for (;;) {
        auto fields = source_fields(config);
        if (cooperative)
            for (auto& [field, value] : fields)
                if (field == "NUM_CTAS") value = std::to_string(grid.x);
        std::string rendered = source;
        if constexpr (std::is_same_v<Config, DispatchConfig>)
            rendered = "#define DG_MOE_FAST_MATH {FAST_MATH}\n" + rendered;
        PrecisionLaunchArgs<Tag, std::tuple<Params...>> args{
            rendered_source<Tag>(name, std::move(rendered), fields), {grid, dim3(threads), 0}, params};
        const auto runtime = compiler->build(name, Runtime::generate(args));
        args.launch_args.smem_size = mega_moe_runtime::get_kernel_smem_size(runtime);
        const int occupancy = mega_moe_runtime::get_kernel_occupancy(runtime, threads, args.launch_args.smem_size);
        TORCH_CHECK(occupancy > 0, "MegaMoE kernel cannot keep a CTA resident");
        if (cooperative) {
            const int resident = device_runtime->get_prop()->multiProcessorCount * occupancy;
            if (grid.x > resident) {
                // Grid-barrier participation is compiled into the kernel.
                // Rebuild and recheck the smaller specialization before launch.
                grid.x = resident;
                continue;
            }
        } else if (occupancy_scaled) {
            args.launch_args.grid_dim.x *= std::min(8, occupancy);
        }
        Runtime::launch(runtime, args);
        return;
    }
}

struct Fp8ChannelwiseMegaMoeDispatchGemm1Tag {
    using Runtime = PPUFP8MegaMoERuntime;
};
inline void fp8_channelwise_mega_moe_dispatch_gemm1_kernel(
    const DispatchConfig& config, const layout::SymBuffer<>& sym_buffer, int rank_idx, int num_tokens,
    const torch::Tensor& gemm1_weight, const torch::Tensor& gemm1_weight_scale,
    const torch::Tensor& gemm1_weight_replicas, const torch::Tensor& gemm1_weight_scale_replicas,
    int num_master_experts, const torch::Tensor& swiglu_output, float activation_clamp,
    const torch::Tensor& block_info, const std::string& entry) {
    launch<Fp8ChannelwiseMegaMoeDispatchGemm1Tag>(
        entry, R"DGMEGA(#include <deep_gemm/impls/fp8_channelwise_mega_moe_dispatch_gemm1.cuh>
using namespace deep_gemm;
static_assert(sizeof(layout::SymBuffer<>) == 528, "MegaMoE symmetric buffer ABI changed");
constexpr uint32_t NUM_RANKS = {NUM_RANKS};
constexpr uint32_t NUM_CTAS = {NUM_CTAS};
constexpr uint32_t NUM_THREADS = {NUM_THREADS};
constexpr uint32_t NUM_EXPERTS = {NUM_EXPERTS};
constexpr uint32_t NUM_EXPERTS_PER_RANK = {NUM_EXPERTS_PER_RANK};
constexpr uint32_t NUM_TOPK = {NUM_TOPK};
constexpr uint32_t BLOCK_M = {BLOCK_M};
constexpr uint32_t NMTPR = {NMTPR};
constexpr uint32_t TOKEN_BYTES = {TOKEN_BYTES};
constexpr uint32_t SF_BYTES = {SF_BYTES};
constexpr uint32_t GEMM1_N = {GEMM1_N};
constexpr uint32_t GEMM1_BLOCK_N = {GEMM1_BLOCK_N};
constexpr uint32_t GEMM1_BLOCK_K = {GEMM1_BLOCK_K};
constexpr uint32_t GEMM1_WARP_M = {GEMM1_WARP_M};
constexpr uint32_t GEMM1_WARP_N = {GEMM1_WARP_N};
constexpr uint32_t GEMM1_STAGES = {GEMM1_STAGES};
constexpr uint32_t PULL_CTAS = {PULL_CTAS};
constexpr uint32_t NUM_WARPS = NUM_THREADS / 32;
constexpr uint32_t COUNT_BYTES = NUM_EXPERTS * sizeof(uint32_t);
constexpr uint32_t TOKEN_SCRATCH_OFFSET = (COUNT_BYTES + 15u) & ~15u;
constexpr uint32_t ppu_smem_compute() {
using Cfg = mega_moe::fp8_channelwise::Gemm1MmaConfig<BLOCK_M, GEMM1_BLOCK_N, GEMM1_BLOCK_K,
    GEMM1_WARP_M, GEMM1_WARP_N, GEMM1_STAGES>;
uint32_t gemm = sizeof(typename Cfg::CollectiveMainloop::SharedStorage);
uint32_t dispatch = ((NUM_EXPERTS * 4 + 15) & ~15u) + (NUM_THREADS / 32) * TOKEN_BYTES;
return gemm > dispatch ? gemm : dispatch;
}
extern "C" __device__ __constant__ uint32_t ppu_mega_moe_smem_bytes = ppu_smem_compute();
extern "C" __global__ __launch_bounds__(NUM_THREADS, 1)
void PPU_KERNEL_ENTRY(layout::SymBuffer<> sb, int rank_idx, int num_tokens, __hg_fp8_e4m3* gemm1_weight, float* gemm1_weight_scale, __hg_fp8_e4m3* gemm1_weight_replicas, float* gemm1_weight_scale_replicas, int num_master_experts, __ppu_bfloat16* swiglu_output, float activation_clamp, uint4* gemm2_block_info) {

using namespace deep_gemm;

constexpr uint32_t NUM_RANKS = {NUM_RANKS};
constexpr uint32_t NUM_CTAS = {NUM_CTAS};
constexpr uint32_t NUM_THREADS = {NUM_THREADS};
constexpr uint32_t NUM_EXPERTS = {NUM_EXPERTS};
constexpr uint32_t NUM_EXPERTS_PER_RANK = {NUM_EXPERTS_PER_RANK};
constexpr uint32_t NUM_TOPK = {NUM_TOPK};
constexpr uint32_t BLOCK_M = {BLOCK_M};
constexpr uint32_t NMTPR = {NMTPR};
constexpr uint32_t TOKEN_BYTES = {TOKEN_BYTES};
constexpr uint32_t SF_BYTES = {SF_BYTES};

constexpr uint64_t ws_bytes = {LAYOUT_WS_BYTES};
constexpr uint64_t num_pool = {LAYOUT_NUM_POOL};
constexpr uint64_t num_pool_blocks = {LAYOUT_NUM_POOL_BLOCKS};
constexpr uint64_t off_ws = 0, off_it = {LAYOUT_OFF_IT}, off_isf = {LAYOUT_OFF_ISF};
constexpr uint64_t off_iti = {LAYOUT_OFF_ITI};
constexpr uint64_t off_pt = {LAYOUT_OFF_PT}, off_psf = {LAYOUT_OFF_PSF}, off_meta = {LAYOUT_OFF_GEMM2_META};

constexpr uint32_t GEMM1_N = {GEMM1_N};
constexpr uint32_t GEMM1_BLOCK_N = {GEMM1_BLOCK_N};
constexpr uint32_t GEMM1_BLOCK_K = {GEMM1_BLOCK_K};
constexpr uint32_t GEMM1_WARP_M = {GEMM1_WARP_M};
constexpr uint32_t GEMM1_WARP_N = {GEMM1_WARP_N};
constexpr uint32_t GEMM1_STAGES = {GEMM1_STAGES};
constexpr uint32_t PULL_CTAS = {PULL_CTAS};

const int64_t local_base = sb.base;
auto P = [&](uint64_t off) -> void* { return (void*)(local_base + (int64_t)off); };

using Cfg = mega_moe::fp8_channelwise::Gemm1MmaConfig<
    BLOCK_M, GEMM1_BLOCK_N, GEMM1_BLOCK_K,
    GEMM1_WARP_M, GEMM1_WARP_N, GEMM1_STAGES>;
constexpr uint32_t NUM_WARPS = NUM_THREADS / 32;
constexpr uint32_t COUNT_BYTES = NUM_EXPERTS * sizeof(uint32_t);
constexpr uint32_t TOKEN_SCRATCH_OFFSET = (COUNT_BYTES + 15u) & ~15u;
uint32_t smem_dispatch = TOKEN_SCRATCH_OFFSET + NUM_WARPS * TOKEN_BYTES;
uint32_t smem_gemm =
    (uint32_t)sizeof(typename Cfg::CollectiveMainloop::SharedStorage);
uint32_t smem = smem_dispatch > smem_gemm ? smem_dispatch : smem_gemm;

deep_gemm::mega_moe::fp8_channelwise::dispatch_gemm1_device<
    NUM_RANKS, NUM_CTAS, NUM_THREADS, NUM_EXPERTS, NUM_EXPERTS_PER_RANK,
    NUM_TOPK, BLOCK_M, TOKEN_BYTES, GEMM1_N, GEMM1_BLOCK_N,
    GEMM1_BLOCK_K, GEMM1_WARP_M, GEMM1_WARP_N, GEMM1_STAGES,
    PULL_CTAS, {SWIGLU_OAI}, void, {PULL_GROUP_SIZE}, {GEMM2_BLOCK_M}>(
    P(off_ws), sb,
    (const int64_t*)P(off_iti),
    (const uint8_t*)P(off_it), (const float*)P(off_isf),
    (uint8_t*)P(off_pt), (float*)P(off_psf), (int32_t*)P(off_meta), gemm2_block_info,
    (const cutlass::float_e4m3_t*)gemm1_weight,
    gemm1_weight_scale,
    (const cutlass::float_e4m3_t*)gemm1_weight_replicas, gemm1_weight_scale_replicas,
    (uint32_t)num_master_experts, swiglu_output,
    (float)activation_clamp, {SWIGLU_ALPHA},
    (uint32_t)num_tokens, NMTPR);

}
)DGMEGA", config,
        std::make_tuple(sym_buffer, int32_t(rank_idx), int32_t(num_tokens), device_pointer(gemm1_weight),
                        device_pointer(gemm1_weight_scale), device_pointer(gemm1_weight_replicas),
                        device_pointer(gemm1_weight_scale_replicas), int32_t(num_master_experts),
                        device_pointer(swiglu_output), activation_clamp, device_pointer(block_info)),
        dim3(config.num_ctas), config.num_threads, false, /*cooperative=*/true);
}

struct Fp8ChannelwiseMegaMoeQuantizeLocalPoolTag {
    using Runtime = PPUFP8MegaMoERuntime;
};
inline void fp8_channelwise_mega_moe_quantize_local_pool_kernel(const DispatchConfig& config,
                                                    const layout::SymBuffer<>& sym_buffer, int rank_idx,
                                                    const torch::Tensor& input, const torch::Tensor& output,
                                                    const torch::Tensor& scales) {
    // Four warps per CTA distribute small batches across more CTAs; four
    // CTAs per SM retain the warp budget of the 512-thread GEMM launch.
    auto quant_config = config;
    quant_config.num_threads = 128;
    launch<Fp8ChannelwiseMegaMoeQuantizeLocalPoolTag>(
        "fp8_channelwise_mega_moe_quantize_local_pool",
        R"DGMEGA(#include <deep_gemm/impls/fp8_channelwise_mega_moe_dispatch_gemm1.cuh>
using namespace deep_gemm;
extern "C" __device__ __constant__ uint32_t ppu_mega_moe_smem_bytes = 0;
extern "C" __global__ __launch_bounds__({NUM_THREADS}, 1)
void PPU_KERNEL_ENTRY(int64_t sb_base,
                      const __ppu_bfloat16* input, cutlass::float_e4m3_t* output, float* scales) {
    const auto* metadata = reinterpret_cast<const int32_t*>(
        sb_base + {LAYOUT_OFF_GEMM2_META});
    mega_moe::fp8_channelwise::fp8_mega_moe_quantize_valid_pool_kernel_device<
        {NUM_EXPERTS_PER_RANK}, {GEMM1_N} / 2>(metadata, input, output, scales);
}
)DGMEGA",
        quant_config,
        std::make_tuple(int64_t(sym_buffer.base), device_pointer(input), device_pointer(output),
                        device_pointer(scales)),
        dim3(config.num_sms * 4), quant_config.num_threads);
}

struct Fp8BlockwiseMegaMoeDispatchGemm1Tag {
    using Runtime = PPUFP8MegaMoERuntime;
};
inline void fp8_blockwise_mega_moe_dispatch_gemm1_kernel(
    const DispatchConfig& config, const layout::SymBuffer<>& sym_buffer, const torch::Tensor& local_pool, int rank_idx,
    int num_tokens, const torch::Tensor& weight, const torch::Tensor& weight_scales, const torch::Tensor& output,
    const torch::Tensor& output_scales, float activation_clamp, float swiglu_alpha, const torch::Tensor& weight_replicas,
    const torch::Tensor& scale_replicas, const std::string& entry) {
    launch<Fp8BlockwiseMegaMoeDispatchGemm1Tag>(
        entry, R"DGMEGA(#include <deep_gemm/impls/fp8_blockwise_mega_moe_dispatch_gemm1.cuh>
using namespace deep_gemm;
static_assert(sizeof(layout::SymBuffer<>) == 528, "MegaMoE symmetric buffer ABI changed");
constexpr uint32_t NUM_RANKS = {NUM_RANKS};
constexpr uint32_t NUM_CTAS = {NUM_CTAS};
constexpr uint32_t NUM_THREADS = {NUM_THREADS};
constexpr uint32_t NUM_EXPERTS = {NUM_EXPERTS};
constexpr uint32_t NUM_TOPK = {NUM_TOPK};
constexpr uint32_t NMTPR = {NMTPR};
constexpr uint32_t TOKEN_BYTES = {TOKEN_BYTES};
using G = mega_moe::fp8_blockwise::Gemm1<{GEMM1_N}, TOKEN_BYTES, {GEMM1_COMPUTE_M}>;
static_assert(G::kNumThreads == NUM_THREADS, "blockwise GEMM1 launch must match its MMA tile");
constexpr uint32_t ppu_smem_compute() {
    return G::shared_memory_bytes<NUM_EXPERTS>();
}
extern "C" __device__ __constant__ uint32_t ppu_mega_moe_smem_bytes = ppu_smem_compute();
extern "C" __global__ __launch_bounds__(NUM_THREADS, 1)
void PPU_KERNEL_ENTRY(layout::SymBuffer<> sb, uint8_t* local_pool, int rank_idx, int num_tokens, __hg_fp8_e4m3* weight, float* weight_scales, uint8_t* output, float* output_scales, float activation_clamp, float swiglu_alpha, __hg_fp8_e4m3* weight_replicas, float* scale_replicas) {

constexpr uint64_t off_ws = 0, off_it = {LAYOUT_OFF_IT}, off_isf = {LAYOUT_OFF_ISF};
constexpr uint64_t off_iti = {LAYOUT_OFF_ITI};
constexpr uint64_t off_pt = {LAYOUT_OFF_PT}, off_psf = {LAYOUT_OFF_PSF}, off_meta = {LAYOUT_OFF_GEMM2_META};

const int64_t local_base = sb.base;
auto P = [&](uint64_t off) -> void* {
    return off >= off_pt ? local_pool + (off - off_pt) : (void*)(local_base + off);
};
mega_moe::fp8_blockwise::dispatch_gemm1_kernel_device<
    NUM_RANKS, NUM_CTAS, NUM_EXPERTS, NUM_TOPK, TOKEN_BYTES, {GEMM1_N}, {PULL_CTAS}, {CLAMP}, {MASTER_EXPERTS}, {SWIGLU_OAI}, {GEMM1_COMPUTE_M}>(
    P(off_ws), sb, (const int64_t*)P(off_iti),
    (const uint8_t*)P(off_it), (const float*)P(off_isf),
    (uint8_t*)P(off_pt), (float*)P(off_psf), (int32_t*)P(off_meta),
    (const uint8_t*)weight, weight_scales, output,
    output_scales,
    (uint32_t)num_tokens, NMTPR, activation_clamp, swiglu_alpha,
    (const uint8_t*)weight_replicas, scale_replicas);
}
)DGMEGA",
        config,
        std::make_tuple(sym_buffer, device_pointer(local_pool), int32_t(rank_idx), int32_t(num_tokens),
                        device_pointer(weight), device_pointer(weight_scales), device_pointer(output),
                        device_pointer(output_scales), activation_clamp, swiglu_alpha, device_pointer(weight_replicas),
                        device_pointer(scale_replicas)),
        dim3(config.num_ctas), config.num_threads, false, /*cooperative=*/true);
}

struct Fp8ChannelwiseMegaMoePreDispatchTag {
    using Runtime = PPUFP8MegaMoERuntime;
};
inline void fp8_channelwise_mega_moe_pre_dispatch_impl(
    int hidden, int num_topk, const torch::Tensor& hidden_states,
    const torch::Tensor& topk_ids, const torch::Tensor& topk_weights,
    const torch::Tensor& input_token, const torch::Tensor& input_scale,
    const torch::Tensor& input_topk_idx, const torch::Tensor& input_topk_weights,
    int num_tokens, const std::string& entry = "fp8_channelwise_mega_moe_pre_dispatch") {
    PreDispatchConfig config;
    config.hidden = hidden;
    config.num_topk = num_topk;
    if (num_tokens == 0)
        return;
    launch<Fp8ChannelwiseMegaMoePreDispatchTag>(
        entry, R"DGMEGA(#include <deep_gemm/impls/fp8_channelwise_mega_moe_pre_dispatch.cuh>
using namespace deep_gemm;
extern "C" __device__ __constant__ uint32_t ppu_mega_moe_smem_bytes = 0;
extern "C" __global__ __launch_bounds__(256, 1)
void PPU_KERNEL_ENTRY(__ppu_bfloat16* hidden_states, int64_t* topk_ids, float* topk_weights, uint8_t* input_token, float* input_scale, int64_t* input_topk_idx, float* input_topk_weights, int num_tokens) {
mega_moe::fp8_channelwise::fp8_mega_moe_pre_dispatch_kernel_device<{HIDDEN}, {TOP_K}, 256>(hidden_states,topk_ids,topk_weights,input_token,input_scale,input_topk_idx,input_topk_weights,(uint32_t)num_tokens);
}
)DGMEGA",
        config,
        std::make_tuple(device_pointer(hidden_states), device_pointer(topk_ids), device_pointer(topk_weights),
                        device_pointer(input_token), device_pointer(input_scale), device_pointer(input_topk_idx),
                        device_pointer(input_topk_weights), int32_t(num_tokens)),
        dim3(num_tokens), 256);
}

struct Fp8BlockwiseMegaMoePreDispatchTag {
    using Runtime = PPUFP8MegaMoERuntime;
};
inline void fp8_blockwise_mega_moe_pre_dispatch_impl(
    int hidden, int num_topk, const torch::Tensor& hidden_states, const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights, const torch::Tensor& input_token, const torch::Tensor& input_scale,
    const torch::Tensor& input_topk_idx, const torch::Tensor& input_topk_weights, int num_tokens,
    const std::string& entry = "fp8_blockwise_mega_moe_pre_dispatch") {
    PreDispatchConfig config;
    config.hidden = hidden;
    config.num_topk = num_topk;
    if (num_tokens == 0)
        return;
    launch<Fp8BlockwiseMegaMoePreDispatchTag>(
        entry, R"DGMEGA(#include <deep_gemm/impls/fp8_blockwise_mega_moe_pre_dispatch.cuh>
using namespace deep_gemm;
extern "C" __device__ __constant__ uint32_t ppu_mega_moe_smem_bytes = 0;
extern "C" __global__ __launch_bounds__(256, 1)
void PPU_KERNEL_ENTRY(__ppu_bfloat16* hidden_states, int64_t* topk_ids, float* topk_weights, __hg_fp8_e4m3* input_token, float* input_scale, int64_t* input_topk_idx, float* input_topk_weights, int num_tokens) {
mega_moe::fp8_blockwise::pre_dispatch_kernel_device<{HIDDEN}, {TOPK}>(hidden_states,topk_ids,topk_weights,input_token,input_scale,input_topk_idx,input_topk_weights,(uint32_t)num_tokens);
}
)DGMEGA",
        config,
        std::make_tuple(device_pointer(hidden_states), device_pointer(topk_ids), device_pointer(topk_weights),
                        device_pointer(input_token), device_pointer(input_scale), device_pointer(input_topk_idx),
                        device_pointer(input_topk_weights), int32_t(num_tokens)),
        dim3(num_tokens), 256);
}

} // namespace deep_gemm::mega_moe_fp8_w4a16
