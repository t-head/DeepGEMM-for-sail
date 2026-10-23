#pragma once

#include <torch/extension.h>
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/mega_moe.hpp"

namespace deep_gemm::mega_moe_native {

static_assert(sizeof(layout::SymBuffer<1>) == 592, "MegaMoE symmetric buffer ABI changed");

using mega_moe_config::DispatchConfig;
using mega_moe_config::PreDispatchConfig;
using mega_moe_config::ScatterConfig;
using SourceFields = std::vector<std::pair<std::string, std::string>>;

inline std::string source_float_literal(float value) {
    TORCH_CHECK(std::isfinite(value), "MegaMoE specialization must be finite");
    return fmt::format("{:.9e}f", value);
}

inline void replace_source_field(std::string& code, const std::string& key, const std::string& text) {
    const auto marker = "{" + key + "}";
    for (size_t pos = code.find(marker); pos != std::string::npos; pos = code.find(marker, pos + text.size()))
        code.replace(pos, marker.size(), text);
}

inline SourceFields source_fields(const DispatchConfig& c) {
    SourceFields values{
        {"GEMM1_DYNAMIC_QUEUE", c.gemm1_dynamic_queue ? "true" : "false"},
        {"GEMM1_COMPUTE_M", std::to_string(c.gemm1_compute_m)},
        {"BLOCK_M", std::to_string(c.block_m)},
        {"CHUNK_BLOCKS", std::to_string(c.chunk_blocks)},
        {"CLAMP", std::to_string(c.clamp)},
        {"CLAMP_VALUE", source_float_literal(c.clamp_value)},
        {"ENABLE_GEMM1", (c.enable_gemm1 ? "true" : "false")},
        {"ENABLE_REMOTE_PULL", (c.enable_remote_pull ? "true" : "false")},
        {"ENABLE_REMOTE_PUSH", (c.enable_remote_push ? "true" : "false")},
        {"QUANTIZE_OUTPUT", (c.quantize_output ? "true" : "false")},
        {"FUSED_SITU", (c.fused_situ ? "true" : "false")},
        {"FUSED_SWIGLU", (c.fused_swiglu ? "true" : "false")},
        {"GEMM2_BLOCK_M", std::to_string(c.gemm2_block_m)},
        {"GEMM1_BLOCK_K", std::to_string(c.gemm1_block_k)},
        {"GEMM1_BLOCK_N", std::to_string(c.gemm1_block_n)},
        {"GEMM1_N", std::to_string(c.gemm1_n)},
        {"GEMM1_STAGES", std::to_string(c.gemm1_stages)},
        {"GEMM1_WARP_M", std::to_string(c.gemm1_warp_m)},
        {"GEMM1_WARP_N", std::to_string(c.gemm1_warp_n)},
        {"GEMM_SMS", std::to_string(c.gemm_sms)},
        {"MASTER_EXPERTS", std::to_string(c.master_experts)},
        {"NMTPR", std::to_string(c.nmtpr)},
        {"NUM_EXPERTS", std::to_string(c.num_experts)},
        {"NUM_EXPERTS_PER_RANK", std::to_string(c.num_experts_per_rank)},
        {"NUM_PULL_SMS", std::to_string(c.num_pull_sms)},
        {"NUM_RANKS", std::to_string(c.num_ranks)},
        {"NUM_SMS", std::to_string(c.num_sms)},
        {"NUM_THREADS", std::to_string(c.num_threads)},
        {"NUM_TOPK", std::to_string(c.num_topk)},
        {"PAIRED_BLOCK_PULL", std::to_string(c.paired_block_pull)},
        {"PRODUCER_WARPS", std::to_string(c.producer_warps)},
        {"PULL_SLOTS", std::to_string(c.pull_slots)},
        {"PULL_SMS", std::to_string(c.pull_sms)},
        {"GEMM1_EPILOGUE", c.gemm1_epilogue == mega_moe::Gemm1Epilogue::SwiGluFp8
            ? "deep_gemm::mega_moe::Gemm1Epilogue::SwiGluFp8"
            : "deep_gemm::mega_moe::Gemm1Epilogue::IdentityBf16"},
        {"SFB_LAYOUT_BLOCK_N", std::to_string(c.sfb_layout_block_n)},
        {"SF_BYTES", std::to_string(c.sf_bytes)},
        {"SILU_THEN_CLAMP", (c.silu_then_clamp ? "true" : "false")},
        {"SITU_BETA", source_float_literal(c.situ_beta)},
        {"SITU_LINEAR_BETA", source_float_literal(c.situ_linear_beta)},
        {"SITU_TANH_BF16X2", (c.situ_tanh_bf16x2 ? "true" : "false")},
        {"SM_SPECIALIZATION", (c.sm_specialization ? "true" : "false")},
        {"SWIGLU_ALPHA", source_float_literal(c.swiglu_alpha)},
        {"SWIGLU_OAI", (c.swiglu_oai ? "true" : "false")},
        {"TOKEN_BYTES", std::to_string(c.token_bytes)},
        {"TRIPLE_BLOCK_PULL", std::to_string(c.triple_block_pull)},
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
    const uint64_t pool = workspace.num_max_pool_tokens, capacity = c.nmtpr;
    const uint64_t it = align128(workspace.get_num_bytes());
    const uint64_t isf = align16(it + capacity * c.token_bytes);
    const uint64_t iti = align16(isf + capacity * c.sf_bytes);
    const uint64_t itw = align16(iti + capacity * c.num_topk * sizeof(int64_t));
    const uint64_t pt = align128(itw + capacity * c.num_topk * sizeof(float));
    const uint64_t psf = align16(pt + pool * c.token_bytes);
    const uint64_t pw = align16(psf + pool * c.sf_bytes);
    for (const auto& [key, value] : std::vector<std::pair<const char*, uint64_t>>{
             {"LAYOUT_WS_BYTES", workspace.get_num_bytes()},
             {"LAYOUT_NUM_POOL", pool},
             {"LAYOUT_NUM_POOL_BLOCKS", workspace.num_max_pool_blocks},
             {"LAYOUT_OFF_IT", it},
             {"LAYOUT_OFF_ISF", isf},
             {"LAYOUT_OFF_ITI", iti},
             {"LAYOUT_OFF_ITW", itw},
             {"LAYOUT_OFF_PT", pt},
             {"LAYOUT_OFF_PSF", psf},
             {"LAYOUT_OFF_PW", pw},
             {"LAYOUT_OFF_W4_GEMM2_META", pw},
             {"LAYOUT_WOFF_RCS", reinterpret_cast<uintptr_t>(workspace.get_expert_recv_count_sum_ptr(0))}})
        values.emplace_back(key, std::to_string(value));
    return values;
}

inline SourceFields source_fields(const PreDispatchConfig& c) {
    SourceFields values{
        {"BLOCK_THREADS", std::to_string(c.block_threads)},
        {"HIDDEN", std::to_string(c.hidden)},
        {"NUM_SMS", std::to_string(c.num_sms)},
        {"TOPK", std::to_string(c.topk)},
        {"TOP_K", std::to_string(c.topk)},
    };
    return values;
}

inline SourceFields source_fields(const ScatterConfig& c) {
    SourceFields values{
        {"BLOCK_K", std::to_string(c.block_k)},
        {"BLOCK_M", std::to_string(c.block_m)},
        {"BLOCK_N", std::to_string(c.block_n)},
        {"SCRATCH_ROWS", std::to_string(c.scratch_rows)},
        {"BM128_OVERLAP_PAIR_STAGING", (c.bm128_overlap_pair_staging ? "true" : "false")},
        {"CHECK_PACKED_ROUTES", (c.check_packed_routes ? "true" : "false")},
        {"GROUPS", std::to_string(c.num_groups)},
        {"K", std::to_string(c.k)},
        {"KERNEL_TYPE", c.kernel_type},
        {"MASTER_GROUPS", std::to_string(c.num_master_groups)},
        {"M_TILES_PER_GROUP", std::to_string(c.m_tiles_per_group)},
        {"N", std::to_string(c.n)},
        {"NUM_GROUPS", std::to_string(c.num_groups)},
        {"NUM_MASTER_GROUPS", std::to_string(c.num_master_groups)},
        {"NUM_STAGES", std::to_string(c.num_stages)},
        {"NUM_THREADS", std::to_string(c.num_threads)},
        {"ROLLING_PAIR_SCATTER", (c.rolling_pair_scatter ? "true" : "false")},
        {"USE_PADDED_VALID_ROWS", (c.use_padded_valid_rows ? "true" : "false")},
        {"WARP_M", std::to_string(c.warp_m)},
        {"WARP_N", std::to_string(c.warp_n)},
        {"WORLD_SIZE", std::to_string(c.world_size)},
    };
    return values;
}

template <typename Config> inline std::string render(std::string code, const Config& config) {
    for (const auto& [key, value] : source_fields(config))
        replace_source_field(code, key, value);
    return code;
}

template <typename Tag, typename Config>
inline std::string rendered_source(const std::string& name, std::string source, const Config& config) {
    const auto fields = source_fields(config);
    std::string key = name;
    for (const auto& [field, value] : fields)
        key += ":" + field + "=" + value;
    static std::mutex mutex;
    static std::unordered_map<std::string, std::string> sources;
    const std::lock_guard<std::mutex> guard(mutex);
    const auto found = sources.find(key);
    if (found != sources.end())
        return found->second;
    const std::string marker = "PPU_KERNEL_ENTRY";
    for (size_t pos = source.find(marker); pos != std::string::npos;
         pos = source.find(marker, pos + name.size()))
        source.replace(pos, marker.size(), name);
    for (const auto& [field, value] : fields)
        replace_source_field(source, field, value);
    return sources.emplace(key, std::move(source)).first->second;
}

inline void* device_pointer(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "MegaMoE expects CUDA device tensors");
    return tensor.data_ptr();
}

template <typename T> inline T* device_pointer(const torch::Tensor& tensor) {
    return reinterpret_cast<T*>(device_pointer(tensor));
}

inline layout::SymBuffer<1> symmetric_buffer(const torch::Tensor& pointers, int rank, int ranks) {
    TORCH_CHECK(!pointers.is_cuda() && pointers.scalar_type() == torch::kInt64 && pointers.is_contiguous() &&
                    pointers.numel() == ranks && ranks > 0 && ranks <= layout::kNumMaxRanks && rank >= 0 &&
                    rank < ranks,
                "invalid CPU symmetric pointer table");
    layout::SymBuffer<1> buffer{};
    buffer.base = pointers.data_ptr<int64_t>()[rank];
    buffer.rank_idx = rank;
    for (int peer = 0; peer < ranks; ++peer)
        buffer.offsets[peer] = pointers.data_ptr<int64_t>()[peer] - buffer.base;
    return buffer;
}

template <typename Tag, typename Config>
inline void launch(const std::string& name, std::string source, const Config& config,
                   std::vector<NativeKernelArgument> arguments, dim3 grid, int threads, bool occupancy_scaled = false) {
    NativeKernelArgs native{rendered_source<Tag>(name, std::move(source), config),
                            {grid, dim3(threads), 0},
                            std::move(arguments)};
    const auto runtime = compile_native_kernel<Tag>(name, native);
    native.launch_args.smem_size = native_kernel_smem(runtime);
    const int occupancy = native_kernel_occupancy(runtime, threads, native.launch_args.smem_size);
    TORCH_CHECK(occupancy >= 1, "MegaMoE launch cannot keep a block resident");
    if (occupancy_scaled) native.launch_args.grid_dim.x *= std::min(8, occupancy);
    NativeKernelRuntime<Tag>::launch(runtime, native);
}

struct Fp4MegaMoeDispatchGemm1Tag {};
inline void fp4_dispatch_gemm1_kernel(
    const DispatchConfig& config, const torch::Tensor& buffer_ptrs, int rank, int num_tokens,
    const torch::Tensor& gemm1_weight, const torch::Tensor& gemm1_weight_sf,
    const torch::Tensor& gemm1_weight_sf_u16, const torch::Tensor& gemm1_weight_replicas,
    const torch::Tensor& gemm1_weight_sf_u16_replicas, int num_master_experts,
    const torch::Tensor& gemm1_output, int gemm1_output_stride, const torch::Tensor& sfa_scratch,
    const torch::Tensor& fp4_output, const torch::Tensor& scale_output, int pull_begin_block,
    int pull_end_block, int resume_dispatch, int finalize_dispatch, int synchronize_pull,
    int publish_arrivals, int num_launch_blocks, const std::string& entry = "fp4_mega_moe_dispatch_gemm1") {
    launch<Fp4MegaMoeDispatchGemm1Tag>(
        entry, R"DGMEGA(#include <deep_gemm/impls/fp4_mega_moe_dispatch_gemm1.cuh>
using namespace deep_gemm;
static_assert(sizeof(layout::SymBuffer<{NUM_RANKS}>) == 592, "MegaMoE symmetric buffer ABI changed");
constexpr uint32_t NUM_RANKS = {NUM_RANKS};
constexpr uint32_t NUM_SMS = {NUM_SMS};
constexpr uint32_t NUM_THREADS = {NUM_THREADS};
constexpr uint32_t NUM_EXPERTS = {NUM_EXPERTS};
constexpr uint32_t NUM_EXPERTS_PER_RANK = {NUM_EXPERTS_PER_RANK};
constexpr uint32_t NUM_TOPK = {NUM_TOPK};
constexpr uint32_t BLOCK_M = {BLOCK_M};
constexpr uint32_t CHUNK_BLOCKS = {CHUNK_BLOCKS};
constexpr uint32_t NMTPR = {NMTPR};
constexpr uint32_t TOKEN_BYTES = {TOKEN_BYTES};
constexpr uint32_t SF_BYTES = {SF_BYTES};
constexpr uint32_t GEMM1_N = {GEMM1_N};
constexpr uint32_t GEMM1_BLOCK_N = {GEMM1_BLOCK_N};
constexpr uint32_t GEMM1_BLOCK_K = {GEMM1_BLOCK_K};
constexpr uint32_t GEMM1_WARP_M = {GEMM1_WARP_M};
constexpr uint32_t GEMM1_WARP_N = {GEMM1_WARP_N};
constexpr uint32_t GEMM1_STAGES = {GEMM1_STAGES};
constexpr uint32_t NUM_PRODUCER_WARPS = {PRODUCER_WARPS};
constexpr uint32_t NUM_LAUNCH_THREADS = NUM_THREADS + NUM_PRODUCER_WARPS * 32;
constexpr uint32_t ppu_smem_compute() {
constexpr int stages = mega_moe::fp4::Gemm1PickStage<BLOCK_M, GEMM1_BLOCK_N, GEMM1_BLOCK_K,
    GEMM1_WARP_M, GEMM1_WARP_N, GEMM1_STAGES, 262144, {SFB_LAYOUT_BLOCK_N}>::value;
using Cfg = mega_moe::fp4::Gemm1MmaConfig<BLOCK_M, GEMM1_BLOCK_N, GEMM1_BLOCK_K,
    GEMM1_WARP_M, GEMM1_WARP_N, stages, {SFB_LAYOUT_BLOCK_N}>;
uint32_t collective = {ENABLE_GEMM1} ? sizeof(typename Cfg::CollectiveMainloop::SharedStorage) : 0;
uint32_t phase_a = NUM_EXPERTS * sizeof(uint32_t);
uint32_t pull = {ENABLE_REMOTE_PULL} ? {PULL_SLOTS} * (TOKEN_BYTES + SF_BYTES) : 0;
return collective > phase_a ? (collective > pull ? collective : pull) : (phase_a > pull ? phase_a : pull);
}
extern "C" __device__ __constant__ uint32_t ppu_native_smem_bytes = ppu_smem_compute();
extern "C" __global__ __launch_bounds__(NUM_THREADS + {PRODUCER_WARPS} * 32, 1)
void PPU_KERNEL_ENTRY(layout::SymBuffer<{NUM_RANKS}> sb, int rank, int num_tokens, uint8_t* gemm1_weight, uint8_t* gemm1_weight_sf, uint16_t* gemm1_weight_sf_u16, uint8_t* gemm1_weight_replicas, uint16_t* gemm1_weight_sf_u16_replicas, int num_master_experts, __ppu_bfloat16* gemm1_output, int gemm1_output_stride, uint16_t* sfa_scratch, uint8_t* fp4_output, uint16_t* scale_output, int pull_begin_block, int pull_end_block, int resume_dispatch, int finalize_dispatch, int synchronize_pull, int publish_arrivals, int num_launch_blocks) {

using namespace deep_gemm;

constexpr uint32_t NUM_RANKS = {NUM_RANKS};
constexpr uint32_t NUM_SMS = {NUM_SMS};
constexpr uint32_t NUM_THREADS = {NUM_THREADS};
constexpr uint32_t NUM_EXPERTS = {NUM_EXPERTS};
constexpr uint32_t NUM_EXPERTS_PER_RANK = {NUM_EXPERTS_PER_RANK};
constexpr uint32_t NUM_TOPK = {NUM_TOPK};
constexpr uint32_t BLOCK_M = {BLOCK_M};
constexpr uint32_t CHUNK_BLOCKS = {CHUNK_BLOCKS};
constexpr uint32_t NMTPR = {NMTPR};
constexpr uint32_t TOKEN_BYTES = {TOKEN_BYTES};
constexpr uint32_t SF_BYTES = {SF_BYTES};
constexpr uint32_t GEMM1_N = {GEMM1_N};
constexpr uint32_t GEMM1_BLOCK_N = {GEMM1_BLOCK_N};
constexpr uint32_t GEMM1_BLOCK_K = {GEMM1_BLOCK_K};
constexpr uint32_t GEMM1_WARP_M = {GEMM1_WARP_M};
constexpr uint32_t GEMM1_WARP_N = {GEMM1_WARP_N};
constexpr uint32_t GEMM1_STAGES = {GEMM1_STAGES};

constexpr uint64_t ws_bytes = {LAYOUT_WS_BYTES};
constexpr uint64_t num_pool = {LAYOUT_NUM_POOL};
constexpr uint64_t num_pool_blocks = {LAYOUT_NUM_POOL_BLOCKS};
constexpr uint64_t off_ws = 0, off_it = {LAYOUT_OFF_IT}, off_isf = {LAYOUT_OFF_ISF};
constexpr uint64_t off_iti = {LAYOUT_OFF_ITI};
constexpr uint64_t off_pt = {LAYOUT_OFF_PT}, off_psf = {LAYOUT_OFF_PSF};

constexpr uint32_t NUM_PRODUCER_WARPS = {PRODUCER_WARPS};
constexpr uint32_t NUM_LAUNCH_THREADS = NUM_THREADS + NUM_PRODUCER_WARPS * 32;

const int64_t local_base = sb.base;
auto P = [&](uint64_t off) -> void* { return (void*)(local_base + (int64_t)off); };
void* ws_base          = P(off_ws);
const int64_t* itopk   = (const int64_t*)P(off_iti);
const uint8_t* itok    = (const uint8_t*)P(off_it);
const uint8_t* isf     = (const uint8_t*)P(off_isf);
uint8_t* ptok          = (uint8_t*)P(off_pt);
uint8_t* psf           = (uint8_t*)P(off_psf);
void* gemm2_metadata  = P({LAYOUT_OFF_W4_GEMM2_META});
__ppu_bfloat16*  gout   = gemm1_output;

const uint8_t*  gw     = gemm1_weight;
const uint8_t*  gwsf   = gemm1_weight_sf;
const uint16_t* gwsf16 = gemm1_weight_sf_u16;
const uint8_t*  gwr    = gemm1_weight_replicas;
const uint16_t* gwsf16r = gemm1_weight_sf_u16_replicas;

void* fp4_output_ptr = (void*)fp4_output;
void* scale_output_ptr = (void*)scale_output;

constexpr int kGemm1Stages = mega_moe::fp4::Gemm1PickStage<
    BLOCK_M, GEMM1_BLOCK_N, GEMM1_BLOCK_K, GEMM1_WARP_M, GEMM1_WARP_N,
    (int)GEMM1_STAGES, 262144, {SFB_LAYOUT_BLOCK_N}>::value;
static_assert(kGemm1Stages >= 1, "gemm1 needs at least 1 pipeline stage");

using Cfg = mega_moe::fp4::Gemm1MmaConfig<BLOCK_M, GEMM1_BLOCK_N, GEMM1_BLOCK_K,
                                GEMM1_WARP_M, GEMM1_WARP_N, kGemm1Stages,
                                {SFB_LAYOUT_BLOCK_N}>;
uint32_t smem_collective = {ENABLE_GEMM1}
    ? (uint32_t)sizeof(typename Cfg::CollectiveMainloop::SharedStorage) : 0;
uint32_t smem_phaseA = NUM_EXPERTS * sizeof(uint32_t);
uint32_t smem = smem_collective > smem_phaseA ? smem_collective : smem_phaseA;
uint32_t smem_pull = {ENABLE_REMOTE_PULL}
    ? (uint32_t){PULL_SLOTS} * (TOKEN_BYTES + SF_BYTES) : 0;
smem = smem > smem_pull ? smem : smem_pull;

float activation_clamp = {CLAMP_VALUE};
bool fused_swiglu = {FUSED_SWIGLU};
bool fused_situ = {FUSED_SITU};
float situ_beta = {SITU_BETA};
float situ_linear_beta = {SITU_LINEAR_BETA};
uint8_t* fp4_out = (uint8_t*)fp4_output_ptr;
uint8_t* scale_out = (uint8_t*)scale_output_ptr;

deep_gemm::mega_moe::fp4::dispatch_gemm1_device<
    NUM_RANKS, NUM_SMS, NUM_LAUNCH_THREADS, NUM_EXPERTS,
    NUM_EXPERTS_PER_RANK, NUM_TOPK, BLOCK_M, CHUNK_BLOCKS,
    {PULL_SMS},
    {GEMM_SMS},
    {SM_SPECIALIZATION},
    {PULL_SLOTS},
    {ENABLE_GEMM1}, GEMM1_N, TOKEN_BYTES, GEMM1_BLOCK_N, GEMM1_BLOCK_K,
    {SFB_LAYOUT_BLOCK_N},
    {ENABLE_REMOTE_PULL},
    {ENABLE_REMOTE_PUSH},
    GEMM1_WARP_M, GEMM1_WARP_N, kGemm1Stages,
    {QUANTIZE_OUTPUT},
    {SITU_TANH_BF16X2},
    {FUSED_SITU},
    {SILU_THEN_CLAMP}, {GEMM2_BLOCK_M}, false, {GEMM1_DYNAMIC_QUEUE}>(
    ws_base, sb, itopk, itok, isf, ptok, psf, gemm2_metadata,
    (uint32_t)num_tokens, NMTPR, TOKEN_BYTES, SF_BYTES,
    gw, gwsf, gout, (uint32_t)gemm1_output_stride, gwsf16,
    gwr, gwsf16r, (uint32_t)num_master_experts, sfa_scratch,
    activation_clamp, fused_swiglu, fused_situ,
    situ_beta, situ_linear_beta,
    fp4_out, scale_out, (int)rank,
    (uint32_t)pull_begin_block, (uint32_t)pull_end_block,
    resume_dispatch, finalize_dispatch,
    synchronize_pull, publish_arrivals);

}
)DGMEGA",
        config,
        {native_pod_argument(symmetric_buffer(buffer_ptrs, rank, config.num_ranks)),
         int32_t(rank),
         int32_t(num_tokens),
         device_pointer(gemm1_weight),
         device_pointer(gemm1_weight_sf),
         device_pointer(gemm1_weight_sf_u16),
         device_pointer(gemm1_weight_replicas),
         device_pointer(gemm1_weight_sf_u16_replicas),
         int32_t(num_master_experts),
         device_pointer(gemm1_output),
         int32_t(gemm1_output_stride),
         device_pointer(sfa_scratch),
         device_pointer(fp4_output),
         device_pointer(scale_output),
         int32_t(pull_begin_block),
         int32_t(pull_end_block),
         int32_t(resume_dispatch),
         int32_t(finalize_dispatch),
         int32_t(synchronize_pull),
         int32_t(publish_arrivals),
         int32_t(num_launch_blocks)},
        dim3(num_launch_blocks), config.num_threads + config.producer_warps * 32);
}

struct Bf16MegaMoePreDispatchTag {};
inline void w4a16_pre_dispatch_impl(
    const torch::Tensor& input, const torch::Tensor& ids, const torch::Tensor& weights,
    const torch::Tensor& staged_tokens, const torch::Tensor& staged_ids,
    const torch::Tensor& staged_weights, int tokens, int hidden, int topk) {
    const bool copy_tokens = input.data_ptr() != staged_tokens.data_ptr();
    // Even inputs already in the workspace need invalid routing weights masked.
    if (!tokens) return;
    const auto c = mega_moe_config::w4a16_pre_dispatch_config(hidden, topk);
    const int vector = (hidden % 8 == 0 &&
        ((reinterpret_cast<uintptr_t>(input.data_ptr()) |
          reinterpret_cast<uintptr_t>(staged_tokens.data_ptr())) & 15) == 0) ? 8 : 1;
    const int token_blocks = copy_tokens ? ceil_div(int64_t(tokens) * hidden / vector, int64_t(c.block_threads)) : 0;
    const int route_blocks = ceil_div(int64_t(tokens) * topk, int64_t(c.block_threads));
    launch<Bf16MegaMoePreDispatchTag>("w4a16_mega_moe_pre_dispatch", R"DGMEGA(
#include <deep_gemm/impls/w4a16_mega_moe_pre_dispatch.cuh>
extern "C" __device__ __constant__ uint32_t ppu_native_smem_bytes = 0;
extern "C" __global__ __launch_bounds__({BLOCK_THREADS})
void PPU_KERNEL_ENTRY(const __ppu_bfloat16* hidden_states, const int64_t* topk_ids, const float* topk_weights,
                     __ppu_bfloat16* staged_tokens, int64_t* staged_ids, float* staged_weights,
                     int num_tokens, int token_blocks, int vector_size) {
    deep_gemm::mega_moe::w4a16::pre_dispatch_kernel_device<{HIDDEN}, {TOP_K}>(
        hidden_states, topk_ids, topk_weights, staged_tokens, staged_ids, staged_weights,
        num_tokens, token_blocks, vector_size);
}
)DGMEGA", c, {input.data_ptr(), ids.data_ptr(), weights.data_ptr(), staged_tokens.data_ptr(),
               staged_ids.data_ptr(), staged_weights.data_ptr(), int32_t(tokens),
               int32_t(token_blocks), int32_t(vector)}, dim3(token_blocks + route_blocks), c.block_threads);
}

enum class W4A16Phase { Fused, Dispatch, Gemm1 };
struct W4A16DispatchGemm1Tag {};
inline void w4a16_dispatch_gemm1_kernel(
    const DispatchConfig& config, const torch::Tensor& buffer_ptrs, int rank, int num_tokens,
    const torch::Tensor& rhs, const torch::Tensor& rhs_scales, const torch::Tensor& rhs_replicas,
    const torch::Tensor& rhs_replica_scales, int num_master_experts, const torch::Tensor& activated,
    float situ_beta, float situ_linear_beta,
    W4A16Phase phase = W4A16Phase::Fused) {
    const bool dispatch = phase != W4A16Phase::Gemm1;
    const bool gemm = phase != W4A16Phase::Dispatch;
    const std::string entry = phase == W4A16Phase::Dispatch ? "w4a16_mega_moe_dispatch_prepare"
                           : phase == W4A16Phase::Gemm1 ? "w4a16_mega_moe_gemm1_situ"
                           : "bf16_mega_moe_dispatch_w4a16_situ";
    std::string source = R"DGMEGA(#include <deep_gemm/impls/w4a16_mega_moe_dispatch_gemm1.cuh>
using namespace deep_gemm;
static_assert(sizeof(layout::SymBuffer<{NUM_RANKS}>) == 592, "MegaMoE symmetric buffer ABI changed");
constexpr uint32_t NUM_RANKS = {NUM_RANKS};
constexpr uint32_t NUM_SMS = {NUM_SMS};
constexpr uint32_t NUM_THREADS = {NUM_THREADS};
constexpr uint32_t NUM_EXPERTS = {NUM_EXPERTS};
constexpr uint32_t NUM_EXPERTS_PER_RANK = {NUM_EXPERTS_PER_RANK};
constexpr uint32_t NUM_TOPK = {NUM_TOPK};
constexpr uint32_t BLOCK_M = {BLOCK_M};
constexpr uint32_t CHUNK_BLOCKS = {CHUNK_BLOCKS};
constexpr uint32_t NMTPR = {NMTPR};
constexpr uint32_t TOKEN_BYTES = {TOKEN_BYTES};
constexpr uint32_t SF_BYTES = {SF_BYTES};
constexpr uint32_t GEMM1_N = {GEMM1_N};
constexpr uint32_t GEMM1_BLOCK_N = {GEMM1_BLOCK_N};
constexpr uint32_t GEMM1_BLOCK_K = {GEMM1_BLOCK_K};
constexpr uint32_t GEMM1_WARP_M = {GEMM1_WARP_M};
constexpr uint32_t GEMM1_WARP_N = {GEMM1_WARP_N};
constexpr uint32_t GEMM1_STAGES = {GEMM1_STAGES};
constexpr uint32_t PULL_SLOTS = {PULL_SLOTS};
constexpr uint32_t ppu_smem_compute() {
using Kernel = cutlass::gemm::kernel::W4A16GEMM_MMA<
    GEMM1_N, TOKEN_BYTES / 2,
    {W4_BLOCK_M}, {W4_BLOCK_N}, {W4_BLOCK_K}, {W4_WARP_M}, {W4_WARP_N}, {W4_WARP_K},
    NUM_EXPERTS_PER_RANK, {W4_STAGES}, GemmType::MGroupedNoPad, 1, false, true>;
uint32_t pull = {RUN_DISPATCH} && !{ENABLE_REMOTE_PUSH} ? PULL_SLOTS * (TOKEN_BYTES + SF_BYTES) : 0;
uint32_t counts = {RUN_DISPATCH} ? NUM_EXPERTS * 4 : 0;
uint32_t w4 = {RUN_GEMM} ? sizeof(typename Kernel::SharedStorage) : 0;
return pull > counts ? (pull > w4 ? pull : w4) : (counts > w4 ? counts : w4);
}
extern "C" __device__ __constant__ uint32_t ppu_native_smem_bytes = ppu_smem_compute();
extern "C" __global__ __launch_bounds__(NUM_THREADS, 1)
void PPU_KERNEL_ENTRY({BUFFER_ARGUMENT}, int rank, int num_tokens, uint8_t* rhs, uint8_t* rhs_scales, uint8_t* rhs_replicas, uint8_t* rhs_replica_scales, int num_master_experts, __ppu_bfloat16* activated, float situ_beta, float situ_linear_beta) {

using namespace deep_gemm;
{LOCAL_BUFFER}

constexpr uint32_t NUM_RANKS = {NUM_RANKS};
constexpr uint32_t NUM_SMS = {NUM_SMS};
constexpr uint32_t NUM_THREADS = {NUM_THREADS};
constexpr uint32_t NUM_EXPERTS = {NUM_EXPERTS};
constexpr uint32_t NUM_EXPERTS_PER_RANK = {NUM_EXPERTS_PER_RANK};
constexpr uint32_t NUM_TOPK = {NUM_TOPK};
constexpr uint32_t BLOCK_M = {BLOCK_M};
constexpr uint32_t CHUNK_BLOCKS = {CHUNK_BLOCKS};
constexpr uint32_t NMTPR = {NMTPR};
constexpr uint32_t TOKEN_BYTES = {TOKEN_BYTES};
constexpr uint32_t SF_BYTES = {SF_BYTES};
constexpr uint32_t GEMM1_N = {GEMM1_N};
constexpr uint32_t GEMM1_BLOCK_N = {GEMM1_BLOCK_N};
constexpr uint32_t GEMM1_BLOCK_K = {GEMM1_BLOCK_K};
constexpr uint32_t GEMM1_WARP_M = {GEMM1_WARP_M};
constexpr uint32_t GEMM1_WARP_N = {GEMM1_WARP_N};
constexpr uint32_t GEMM1_STAGES = {GEMM1_STAGES};

constexpr uint64_t ws_bytes = {LAYOUT_WS_BYTES};
constexpr uint64_t num_pool = {LAYOUT_NUM_POOL};
constexpr uint64_t num_pool_blocks = {LAYOUT_NUM_POOL_BLOCKS};
constexpr uint64_t off_ws = 0, off_it = {LAYOUT_OFF_IT}, off_isf = {LAYOUT_OFF_ISF};
constexpr uint64_t off_iti = {LAYOUT_OFF_ITI};
constexpr uint64_t off_pt = {LAYOUT_OFF_PT}, off_psf = {LAYOUT_OFF_PSF};

const int64_t local_base = sb.base;
auto P = [&](uint64_t off) -> void* {{
    return (void*)(local_base + (int64_t)off);
}};
void* ws_base          = P(off_ws);
const int64_t* itopk   = (const int64_t*)P(off_iti);
const uint8_t* itok    = (const uint8_t*)P(off_it);
const uint8_t* isf     = (const uint8_t*)P(off_isf);
uint8_t* ptok          = (uint8_t*)P(off_pt);
uint8_t* psf           = (uint8_t*)P(off_psf);
void* gemm2_metadata  = P({LAYOUT_OFF_W4_GEMM2_META});

constexpr int W4_BLOCK_M = {W4_BLOCK_M};
constexpr int W4_BLOCK_N = {W4_BLOCK_N};
constexpr int W4_BLOCK_K = {W4_BLOCK_K};
constexpr int W4_WARP_M = {W4_WARP_M};
constexpr int W4_WARP_N = {W4_WARP_N};
constexpr int W4_WARP_K = {W4_WARP_K};
constexpr int W4_STAGES = {W4_STAGES};
constexpr uint32_t PULL_SLOTS = {PULL_SLOTS};

mega_moe::w4a16::dispatch_gemm1_device<
    NUM_RANKS, NUM_SMS, NUM_THREADS, NUM_EXPERTS,
    NUM_EXPERTS_PER_RANK, NUM_TOPK, BLOCK_M, CHUNK_BLOCKS,
    PULL_SLOTS, NMTPR,
    GEMM1_N, TOKEN_BYTES / 2,
    GEMM1_BLOCK_N, GEMM1_BLOCK_K,
    GEMM1_WARP_M, GEMM1_WARP_N, GEMM1_STAGES,
    W4_BLOCK_M, W4_BLOCK_N, W4_BLOCK_K,
    W4_WARP_M, W4_WARP_N, W4_WARP_K, W4_STAGES,
    {GEMM1_DYNAMIC_QUEUE}, {NUM_PULL_SMS}, {RUN_DISPATCH}, {RUN_GEMM}, {ENABLE_REMOTE_PUSH}>(
    ws_base, sb, itopk, itok, isf, ptok, psf, gemm2_metadata,
    (uint32_t)num_tokens, rhs, rhs_scales, rhs_replicas, rhs_replica_scales,
    (uint32_t)num_master_experts, activated,
    situ_beta, situ_linear_beta, (uint32_t)num_pool, (int)rank);

}
)DGMEGA";
    // Compute-only GEMM1 uses no peer addresses. Keep its entry arguments small:
    // passing the full communication descriptor also changes generated GEMM code.
    replace_source_field(source, "BUFFER_ARGUMENT", dispatch
        ? "layout::SymBuffer<{NUM_RANKS}> sb" : "int64_t sb_base");
    replace_source_field(source, "LOCAL_BUFFER", dispatch
        ? "" : "layout::SymBuffer<{NUM_RANKS}> sb{}; sb.base = sb_base;");
    replace_source_field(source, "RUN_DISPATCH", dispatch ? "true" : "false");
    replace_source_field(source, "RUN_GEMM", gemm ? "true" : "false");
    const auto sb = symmetric_buffer(buffer_ptrs, rank, config.num_ranks);
    const NativeKernelArgument buffer_argument = dispatch
        ? native_pod_argument(sb) : NativeKernelArgument{int64_t(sb.base)};
    launch<W4A16DispatchGemm1Tag>(entry, std::move(source), config,
        {buffer_argument, int32_t(rank),
         int32_t(num_tokens), device_pointer(rhs), device_pointer(rhs_scales),
         device_pointer(rhs_replicas), device_pointer(rhs_replica_scales), int32_t(num_master_experts),
         device_pointer(activated),
         situ_beta, situ_linear_beta},
        dim3(config.num_sms), config.num_threads, phase == W4A16Phase::Gemm1);
}

struct Fp8MegaMoeDispatchGemm1SwigluTag {};
inline void fp8_channelwise_dispatch_gemm1_kernel(
    const DispatchConfig& config, const torch::Tensor& buffer_ptrs, int rank, int num_tokens,
    const torch::Tensor& gemm1_weight, const torch::Tensor& gemm1_weight_scale,
    const torch::Tensor& gemm1_weight_replicas, const torch::Tensor& gemm1_weight_scale_replicas,
    int num_master_experts, const torch::Tensor& swiglu_output, float swiglu_limit,
    const std::string& entry = "fp8_mega_moe_dispatch_gemm1_swiglu") {
    launch<Fp8MegaMoeDispatchGemm1SwigluTag>(
        entry, R"DGMEGA(#include <deep_gemm/impls/fp8_channelwise_mega_moe_dispatch_gemm1.cuh>
using namespace deep_gemm;
static_assert(sizeof(layout::SymBuffer<{NUM_RANKS}>) == 592, "MegaMoE symmetric buffer ABI changed");
constexpr uint32_t NUM_RANKS = {NUM_RANKS};
constexpr uint32_t NUM_SMS = {NUM_SMS};
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
constexpr uint32_t NUM_PULL_SMS = {NUM_PULL_SMS};
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
extern "C" __device__ __constant__ uint32_t ppu_native_smem_bytes = ppu_smem_compute();
extern "C" __global__ __launch_bounds__(NUM_THREADS, 1)
void PPU_KERNEL_ENTRY(layout::SymBuffer<{NUM_RANKS}> sb, int rank, int num_tokens, __hg_fp8_e4m3* gemm1_weight, float* gemm1_weight_scale, __hg_fp8_e4m3* gemm1_weight_replicas, float* gemm1_weight_scale_replicas, int num_master_experts, __ppu_bfloat16* swiglu_output, float swiglu_limit) {

using namespace deep_gemm;

constexpr uint32_t NUM_RANKS = {NUM_RANKS};
constexpr uint32_t NUM_SMS = {NUM_SMS};
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
constexpr uint64_t off_iti = {LAYOUT_OFF_ITI}, off_itw = {LAYOUT_OFF_ITW};
constexpr uint64_t off_pt = {LAYOUT_OFF_PT}, off_psf = {LAYOUT_OFF_PSF}, off_pw = {LAYOUT_OFF_PW};

constexpr uint32_t GEMM1_N = {GEMM1_N};
constexpr uint32_t GEMM1_BLOCK_N = {GEMM1_BLOCK_N};
constexpr uint32_t GEMM1_BLOCK_K = {GEMM1_BLOCK_K};
constexpr uint32_t GEMM1_WARP_M = {GEMM1_WARP_M};
constexpr uint32_t GEMM1_WARP_N = {GEMM1_WARP_N};
constexpr uint32_t GEMM1_STAGES = {GEMM1_STAGES};
constexpr uint32_t NUM_PULL_SMS = {NUM_PULL_SMS};

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
    NUM_RANKS, NUM_SMS, NUM_THREADS, NUM_EXPERTS, NUM_EXPERTS_PER_RANK,
    NUM_TOPK, BLOCK_M, TOKEN_BYTES, GEMM1_N, GEMM1_BLOCK_N,
    GEMM1_BLOCK_K, GEMM1_WARP_M, GEMM1_WARP_N, GEMM1_STAGES,
    NUM_PULL_SMS, {PAIRED_BLOCK_PULL},
    {TRIPLE_BLOCK_PULL}, {SWIGLU_OAI}, {GEMM1_DYNAMIC_QUEUE}>(
    P(off_ws), sb,
    (const int64_t*)P(off_iti), (const float*)P(off_itw),
    (const uint8_t*)P(off_it), (const float*)P(off_isf),
    (uint8_t*)P(off_pt), (float*)P(off_psf), (float*)P(off_pw),
    (const cutlass::float_e4m3_t*)gemm1_weight,
    gemm1_weight_scale,
    (const cutlass::float_e4m3_t*)gemm1_weight_replicas, gemm1_weight_scale_replicas,
    (uint32_t)num_master_experts, swiglu_output,
    (float)swiglu_limit, {SWIGLU_ALPHA},
    (uint32_t)num_tokens, NMTPR);

}
)DGMEGA",
        config,
        {native_pod_argument(symmetric_buffer(buffer_ptrs, rank, config.num_ranks)), int32_t(rank),
         int32_t(num_tokens), device_pointer(gemm1_weight), device_pointer(gemm1_weight_scale),
         device_pointer(gemm1_weight_replicas), device_pointer(gemm1_weight_scale_replicas),
         int32_t(num_master_experts), device_pointer(swiglu_output), swiglu_limit},
        dim3(config.num_sms), config.num_threads);
}

struct Fp8MegaMoeQuantizeLocalPoolTag {};
inline void fp8_mega_moe_quantize_local_pool_kernel(
    const DispatchConfig& config, const torch::Tensor& buffer_ptrs, int rank,
    const torch::Tensor& input, const torch::Tensor& output, const torch::Tensor& scales) {
    // Four warps per CTA distribute small batches across more CTAs; four
    // CTAs per SM retain the warp budget of the 512-thread GEMM launch.
    auto quant_config = config;
    quant_config.num_threads = 128;
    launch<Fp8MegaMoeQuantizeLocalPoolTag>(
        "fp8_mega_moe_quantize_local_pool",
        R"DGMEGA(#include <deep_gemm/impls/fp8_channelwise_mega_moe_dispatch_gemm1.cuh>
using namespace deep_gemm;
extern "C" __device__ __constant__ uint32_t ppu_native_smem_bytes = 0;
extern "C" __global__ __launch_bounds__({NUM_THREADS}, 1)
void PPU_KERNEL_ENTRY(int64_t sb_base,
                      const __ppu_bfloat16* input, cutlass::float_e4m3_t* output, float* scales) {
    const auto* metadata = reinterpret_cast<const int32_t*>(
        sb_base + {LAYOUT_OFF_PW} + {LAYOUT_NUM_POOL} * sizeof(float));
    mega_moe::fp8_channelwise::fp8_mega_moe_quantize_valid_pool_kernel_device<
        {NUM_EXPERTS_PER_RANK}, {GEMM1_N} / 2>(metadata, input, output, scales);
}
)DGMEGA",
        quant_config,
        {int64_t(symmetric_buffer(buffer_ptrs, rank, config.num_ranks).base),
         device_pointer(input), device_pointer(output), device_pointer(scales)},
        dim3(config.num_sms * 4), quant_config.num_threads);
}

struct Fp8BlockwiseMegaMoeDispatchGemm1Tag {};
inline void fp8_blockwise_mega_moe_dispatch_gemm1_kernel(
    const DispatchConfig& config, const torch::Tensor& buffer_ptrs, const torch::Tensor& local_pool,
    int rank, int num_tokens,
    const torch::Tensor& weight, const torch::Tensor& weight_scales, const torch::Tensor& output,
    const torch::Tensor& output_scales, float swiglu_limit, float swiglu_alpha,
    const torch::Tensor& weight_replicas, const torch::Tensor& scale_replicas,
    const std::string& entry = "fp8_blockwise_mega_moe_dispatch_gemm1") {
    launch<Fp8BlockwiseMegaMoeDispatchGemm1Tag>(
        entry, R"DGMEGA(#include <deep_gemm/impls/fp8_blockwise_mega_moe_dispatch_gemm1.cuh>
using namespace deep_gemm;
static_assert(sizeof(layout::SymBuffer<{NUM_RANKS}>) == 592, "MegaMoE symmetric buffer ABI changed");
constexpr uint32_t NUM_RANKS = {NUM_RANKS};
constexpr uint32_t NUM_SMS = {NUM_SMS};
constexpr uint32_t NUM_THREADS = {NUM_THREADS};
constexpr uint32_t NUM_EXPERTS = {NUM_EXPERTS};
constexpr uint32_t NUM_TOPK = {NUM_TOPK};
constexpr uint32_t NMTPR = {NMTPR};
constexpr uint32_t TOKEN_BYTES = {TOKEN_BYTES};
using G = mega_moe::fp8_blockwise::Gemm1<{GEMM1_N}, TOKEN_BYTES, {GEMM1_EPILOGUE}, {GEMM1_COMPUTE_M}>;
static_assert(G::kNumThreads == NUM_THREADS, "blockwise GEMM1 launch must match its MMA tile");
constexpr uint32_t ppu_smem_compute() {
    return G::shared_memory_bytes<NUM_EXPERTS>();
}
extern "C" __device__ __constant__ uint32_t ppu_native_smem_bytes = ppu_smem_compute();
extern "C" __global__ __launch_bounds__(NUM_THREADS, 1)
void PPU_KERNEL_ENTRY(layout::SymBuffer<{NUM_RANKS}> sb, uint8_t* local_pool, int rank, int num_tokens, __hg_fp8_e4m3* weight, float* weight_scales, void* output, float* output_scales, float swiglu_limit, float swiglu_alpha, __hg_fp8_e4m3* weight_replicas, float* scale_replicas) {

constexpr uint64_t off_ws = 0, off_it = {LAYOUT_OFF_IT}, off_isf = {LAYOUT_OFF_ISF};
constexpr uint64_t off_iti = {LAYOUT_OFF_ITI}, off_itw = {LAYOUT_OFF_ITW};
constexpr uint64_t off_pt = {LAYOUT_OFF_PT}, off_psf = {LAYOUT_OFF_PSF}, off_pw = {LAYOUT_OFF_PW};

const int64_t local_base = sb.base;
auto P = [&](uint64_t off) -> void* {
    return off >= off_pt ? local_pool + (off - off_pt) : (void*)(local_base + off);
};
mega_moe::fp8_blockwise::dispatch_gemm1_kernel_device<
    NUM_RANKS, NUM_SMS, NUM_EXPERTS, NUM_TOPK, TOKEN_BYTES, {GEMM1_N}, {PULL_SMS}, {GEMM1_EPILOGUE}, {CLAMP}, {MASTER_EXPERTS}, {SWIGLU_OAI}, {GEMM1_COMPUTE_M}, {GEMM1_DYNAMIC_QUEUE}>(
    P(off_ws), sb, (const int64_t*)P(off_iti), (const float*)P(off_itw),
    (const uint8_t*)P(off_it), (const float*)P(off_isf),
    (uint8_t*)P(off_pt), (float*)P(off_psf), (float*)P(off_pw),
    (const uint8_t*)weight, weight_scales, output,
    ({GEMM1_EPILOGUE} == mega_moe::Gemm1Epilogue::SwiGluFp8) ? output_scales : nullptr,
    (uint32_t)num_tokens, NMTPR, swiglu_limit, swiglu_alpha,
    (const uint8_t*)weight_replicas, scale_replicas);
}
)DGMEGA",
        config,
        {native_pod_argument(symmetric_buffer(buffer_ptrs, rank, config.num_ranks)),
         device_pointer(local_pool), int32_t(rank),
         int32_t(num_tokens), device_pointer(weight), device_pointer(weight_scales), device_pointer(output),
         device_pointer(output_scales), swiglu_limit, swiglu_alpha,
         device_pointer(weight_replicas), device_pointer(scale_replicas)},
        dim3(config.num_sms), config.num_threads);
}

struct Fp8MegaMoePreDispatchTag {};
inline void fp8_mega_moe_pre_dispatch_impl(int hidden, int topk, const torch::Tensor& hidden_states,
                                           const torch::Tensor& topk_ids, const torch::Tensor& topk_weights,
                                           const torch::Tensor& input_token, const torch::Tensor& input_scale,
                                           const torch::Tensor& input_topk_idx,
                                           const torch::Tensor& input_topk_weights, int num_tokens,
                                           const std::string& entry = "fp8_mega_moe_pre_dispatch") {
    PreDispatchConfig config;
    config.hidden = hidden;
    config.topk = topk;
    if (num_tokens == 0)
        return;
    launch<Fp8MegaMoePreDispatchTag>(entry, R"DGMEGA(#include <deep_gemm/impls/fp8_channelwise_mega_moe_pre_dispatch.cuh>
using namespace deep_gemm;
extern "C" __device__ __constant__ uint32_t ppu_native_smem_bytes = 0;
extern "C" __global__ __launch_bounds__(256, 1)
void PPU_KERNEL_ENTRY(__ppu_bfloat16* hidden_states, int64_t* topk_ids, float* topk_weights, uint8_t* input_token, float* input_scale, int64_t* input_topk_idx, float* input_topk_weights, int num_tokens) {
mega_moe::fp8_channelwise::fp8_mega_moe_pre_dispatch_kernel_device<{HIDDEN}, {TOP_K}, 256>(hidden_states,topk_ids,topk_weights,input_token,input_scale,input_topk_idx,input_topk_weights,(uint32_t)num_tokens);
}
)DGMEGA",
                                     config,
                                     {device_pointer(hidden_states), device_pointer(topk_ids),
                                      device_pointer(topk_weights), device_pointer(input_token),
                                      device_pointer(input_scale), device_pointer(input_topk_idx),
                                      device_pointer(input_topk_weights), int32_t(num_tokens)},
                                     dim3(num_tokens), 256);
}

struct Fp8BlockwiseMegaMoePreDispatchTag {};
inline void fp8_blockwise_mega_moe_pre_dispatch_impl(
    int hidden, int topk, const torch::Tensor& hidden_states, const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights, const torch::Tensor& input_token, const torch::Tensor& input_scale,
    const torch::Tensor& input_topk_idx, const torch::Tensor& input_topk_weights, int num_tokens,
    const std::string& entry = "fp8_blockwise_mega_moe_pre_dispatch") {
    PreDispatchConfig config;
    config.hidden = hidden;
    config.topk = topk;
    if (num_tokens == 0)
        return;
    launch<Fp8BlockwiseMegaMoePreDispatchTag>(
        entry, R"DGMEGA(#include <deep_gemm/impls/fp8_blockwise_mega_moe_pre_dispatch.cuh>
using namespace deep_gemm;
extern "C" __device__ __constant__ uint32_t ppu_native_smem_bytes = 0;
extern "C" __global__ __launch_bounds__(256, 1)
void PPU_KERNEL_ENTRY(__ppu_bfloat16* hidden_states, int64_t* topk_ids, float* topk_weights, __hg_fp8_e4m3* input_token, float* input_scale, int64_t* input_topk_idx, float* input_topk_weights, int num_tokens) {
mega_moe::fp8_blockwise::pre_dispatch_kernel_device<{HIDDEN}, {TOPK}>(hidden_states,topk_ids,topk_weights,input_token,input_scale,input_topk_idx,input_topk_weights,(uint32_t)num_tokens);
}
)DGMEGA",
        config,
        {device_pointer(hidden_states), device_pointer(topk_ids), device_pointer(topk_weights),
         device_pointer(input_token), device_pointer(input_scale), device_pointer(input_topk_idx),
         device_pointer(input_topk_weights), int32_t(num_tokens)},
        dim3(num_tokens), 256);
}

struct Fp4MegaMoePreDispatchTag {};
inline void fp4_mega_moe_pre_dispatch_kernel(
    const PreDispatchConfig& config, const torch::Tensor& hidden_states, const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights, const torch::Tensor& input_token, const torch::Tensor& input_sf,
    const torch::Tensor& input_topk_idx, const torch::Tensor& input_topk_weights, int num_tokens,
    const std::string& entry = "fp4_mega_moe_pre_dispatch") {
    launch<Fp4MegaMoePreDispatchTag>(
        entry, R"DGMEGA(#include <deep_gemm/impls/fp4_mega_moe_pre_dispatch.cuh>
using namespace deep_gemm;
extern "C" __device__ __constant__ uint32_t ppu_native_smem_bytes = 0;
extern "C" __global__ __launch_bounds__({BLOCK_THREADS}, 4)
void PPU_KERNEL_ENTRY(__ppu_bfloat16* hidden_states, int64_t* topk_ids, float* topk_weights, uint8_t* input_token, uint8_t* input_sf, int64_t* input_topk_idx, float* input_topk_weights, int num_tokens) {
mega_moe::fp4::fp4_mega_moe_pre_dispatch_kernel_device<{HIDDEN}, {TOP_K}, {BLOCK_THREADS}>(hidden_states,topk_ids,topk_weights,input_token,input_sf,input_topk_idx,input_topk_weights,(uint32_t)num_tokens);
}
)DGMEGA",
        config,
        {device_pointer(hidden_states), device_pointer(topk_ids), device_pointer(topk_weights),
         device_pointer(input_token), device_pointer(input_sf), device_pointer(input_topk_idx),
         device_pointer(input_topk_weights), int32_t(num_tokens)},
        dim3(std::min(num_tokens, config.num_sms * 4)), config.block_threads);
}

inline void fp4_mega_moe_pre_dispatch_impl(const torch::Tensor& hidden_states, const torch::Tensor& topk_ids,
                                           const torch::Tensor& topk_weights,
                                           const torch::Tensor& input_token, const torch::Tensor& input_sf,
                                           const torch::Tensor& input_topk_idx,
                                           const torch::Tensor& input_topk_weights, int num_tokens,
                                           int hidden, int topk, int threads) {
    if (num_tokens == 0)
        return;
    PreDispatchConfig c;
    c.hidden = hidden;
    c.topk = topk;
    c.block_threads = threads;
    c.num_sms = get_num_sms();
    mega_moe_native::fp4_mega_moe_pre_dispatch_kernel(c, hidden_states, topk_ids, topk_weights,
                                                      input_token, input_sf, input_topk_idx,
                                                      input_topk_weights, num_tokens);
}

inline int fp4_mega_moe_dispatch_gemm1_impl(
    DispatchConfig c, const DispatchConfig& capacity, const torch::Tensor& buffer_ptrs,
    const torch::Tensor& gemm1_weight, const torch::Tensor& gemm1_weight_sf,
    const torch::Tensor& gemm1_weight_sf_u16, const torch::Tensor& replica,
    const torch::Tensor& replica_scale, const torch::Tensor& output, const torch::Tensor& scratch, int rank,
    int num_tokens, int masters, int stride, int sfb_bn, std::optional<torch::Tensor> fp4_output,
    std::optional<torch::Tensor> scale_output, bool dynamic, bool bf16, bool dispatch_only, bool split,
    bool glu, bool quantize_output, const std::string& act, float activation_clamp, float situ_beta,
    float situ_linear_beta, bool situ_tanh_bf16x2, int pull_begin_block, int pull_end_block,
    bool resume_dispatch, bool finalize_dispatch, bool synchronize_pull, bool publish_arrivals,
    int num_launch_blocks, int gemm2_block_m = 0,
    const mega_moe_config::Gemm1Tuning& tuning = {}) {
    mega_moe_config::fp4_dispatch_schedule(c, capacity, num_tokens, dynamic, bf16, dispatch_only, split, glu,
                                           quantize_output, tuning);
    c.gemm2_block_m = gemm2_block_m ? gemm2_block_m : c.block_m;
    TORCH_CHECK(c.enable_remote_pull, "MegaMoE GEMM1 metadata requires remote-pull dispatch");
    c.sfb_layout_block_n = sfb_bn;
    c.clamp_value = std::max(activation_clamp, 0.0f);
    c.fused_swiglu = act == "silu_and_mul" || act == "silu_and_mul_step";
    c.silu_then_clamp = act == "silu_and_mul_step";
    c.fused_situ = act == "situ_and_mul";
    c.situ_beta = situ_beta;
    c.situ_linear_beta = situ_linear_beta;
    c.situ_tanh_bf16x2 = situ_tanh_bf16x2;
    c.quantize_output = quantize_output;
    mega_moe_native::fp4_dispatch_gemm1_kernel(
        c, buffer_ptrs, rank, num_tokens, gemm1_weight, gemm1_weight_sf, gemm1_weight_sf_u16, replica,
        replica_scale, masters, output, stride, scratch,
        fp4_output.value_or(torch::empty({0}, gemm1_weight.options())),
        scale_output.value_or(torch::empty({0}, gemm1_weight_sf_u16.options())), pull_begin_block,
        pull_end_block < 0 ? int64_t(0xffffffffu) : int64_t(pull_end_block), int(resume_dispatch),
        int(finalize_dispatch), int(synchronize_pull), int(publish_arrivals), num_launch_blocks);
    return c.block_m;
}

struct W4A16ActivationConfig {
    enum class Type { SiTU };
    Type type;
    float beta;
    float linear_beta;
};

inline int w4a16_dispatch_gemm1_impl(
    const mega_moe_config::W4A16Gemm1Config& config,
    const torch::Tensor& buffer_ptrs, const torch::Tensor& gemm1_weight,
    const torch::Tensor& gemm1_weight_scale, const torch::Tensor& replica,
    const torch::Tensor& replica_scale, int masters, const torch::Tensor& activated,
    int rank, int num_tokens, const W4A16ActivationConfig& activation) {
    DG_HOST_ASSERT(activation.type == W4A16ActivationConfig::Type::SiTU);
    auto c = config.dispatch;
    if (config.separate_dispatch) {
        w4a16_dispatch_gemm1_kernel(c, buffer_ptrs, rank, num_tokens, gemm1_weight,
            gemm1_weight_scale, replica, replica_scale, masters, activated,
            activation.beta, activation.linear_beta, W4A16Phase::Dispatch);
        c.num_threads = config.gemm_threads;
        w4a16_dispatch_gemm1_kernel(c, buffer_ptrs, rank, num_tokens, gemm1_weight,
            gemm1_weight_scale, replica, replica_scale, masters, activated,
            activation.beta, activation.linear_beta, W4A16Phase::Gemm1);
        return c.block_m;
    }
    w4a16_dispatch_gemm1_kernel(
        c, buffer_ptrs, rank, num_tokens, gemm1_weight, gemm1_weight_scale, replica, replica_scale,
        masters, activated,
        activation.beta, activation.linear_beta);
    return c.block_m;
}

inline int fp8_mega_moe_dispatch_gemm1_impl(
    DispatchConfig c, const torch::Tensor& ptrs, int rank, int num_tokens, const torch::Tensor& weight,
    const torch::Tensor& scales, const torch::Tensor& rw, const torch::Tensor& rs, int masters,
    const torch::Tensor& output, std::optional<torch::Tensor> quant, std::optional<torch::Tensor> quant_scale,
    std::optional<float> limit, std::optional<bool> paired,
    std::optional<bool> triple, bool oai, float alpha,
    const mega_moe_config::Gemm1Tuning& tuning = {}, std::optional<int> block_m = std::nullopt) {
    const int n = weight.size(1);
    c.gemm1_n = n;
    mega_moe_config::fp8_gemm1_config(c, paired, triple, tuning, block_m);
    c.swiglu_oai = oai;
    c.swiglu_alpha = alpha;
    const float clamp = limit.value_or(-1.0f);
    mega_moe_native::fp8_channelwise_dispatch_gemm1_kernel(
        c, ptrs, rank, num_tokens, weight, scales, rw, rs, masters, output, clamp);
    if (quant)
        mega_moe_native::fp8_mega_moe_quantize_local_pool_kernel(
            c, ptrs, rank, output, *quant, *quant_scale);
    return c.block_m;
}

inline int fp8_blockwise_mega_moe_dispatch_gemm1_impl(
    DispatchConfig c, const torch::Tensor& ptrs, const torch::Tensor& local_pool,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scales, const torch::Tensor& output, const torch::Tensor& rw,
    const torch::Tensor& rs, int rank, int num_tokens, int n, int e,
    bool quant, bool clamp, float limit, std::optional<torch::Tensor> output_scales,
    bool oai, float alpha, const mega_moe_config::Gemm1Tuning& tuning) {
    mega_moe_config::fp8_blockwise_gemm1_config(c, num_tokens, n, quant, tuning);
    c.swiglu_oai = oai;
    c.gemm1_n = n;
    c.master_experts = e;
    c.gemm1_epilogue = quant ? mega_moe::Gemm1Epilogue::SwiGluFp8
                            : mega_moe::Gemm1Epilogue::IdentityBf16;
    c.clamp = int(clamp);
    mega_moe_native::fp8_blockwise_mega_moe_dispatch_gemm1_kernel(
        c, ptrs, local_pool, rank, num_tokens, weight, weight_scales, output, output_scales.value_or(weight_scales),
        limit, alpha, rw, rs,
        // Use the FP8 kernel family name to retain its compiler scheduling policy.
        quant ? "fp8_deep_gemm_mega_moe_blockwise_dispatch_gemm1"
              : "fp8_blockwise_mega_moe_dispatch_gemm1");
    return c.block_m;
}

} // namespace deep_gemm::mega_moe_native
