#pragma once

#include <torch/python.h>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/layout.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"
#include "../heuristics/common_mqa.hpp"
#include <deep_gemm/common/profiling_interface.cuh>

namespace deep_gemm {

// Compute tile rows, independent of the physical page size.
static constexpr int kPagedMqaBlockKV = 64;

// ---------------------------------------------------------------- metadata

// Builds the `schedule_metadata` table consumed by the paged kernel. A tiny single-block kernel.
class PagedMqaLogitsMetadataRuntime final : public LaunchRuntime<PagedMqaLogitsMetadataRuntime> {
public:
    struct KernelArguments {
        uint32_t batch_size;
        uint32_t next_n;
        const uint32_t* context_lens;
        uint32_t* schedule_metadata;
        const uint32_t* indices;
    };

    struct LaunchInfo {
        int split_kv, num_sms;
        int group_q;
        std::string kernel_name;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        KernelArguments kernel_args;
    };

    static std::string generate_impl(const Args& args) {
        const bool grouped = args.launch_info.group_q != 0;
        return fmt::format(
            R"(
#include <deep_gemm/scheduler/paged_mqa_logits_scheduler.cuh>
namespace deep_gemm {{

constexpr uint32_t SPLIT_KV = {};
constexpr uint32_t kNumSMs = {};

extern "C"
__global__ void {}(
  const uint32_t batch_size,
  const uint32_t next_n,
  const uint32_t* __restrict__ context_lens,
  uint32_t* __restrict__ schedule_metadata{}
) {{
  smxx_paged_mqa_logits_metadata_device<SPLIT_KV, kNumSMs{}>(
      batch_size, next_n, context_lens, schedule_metadata{});
}}
}}
)",
            args.launch_info.split_kv, args.launch_info.num_sms, args.launch_info.kernel_name,
            grouped ? ", const uint32_t* __restrict__ indices" : "",
            grouped ? fmt::format(", {}", args.launch_info.group_q) : "", grouped ? ", indices" : "");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        const auto& p = args.kernel_args;
        if (args.launch_info.group_q)
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, p.batch_size, p.next_n, p.context_lens, p.schedule_metadata, p.indices));
        else
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, p.batch_size, p.next_n, p.context_lens, p.schedule_metadata));
    }
};

class FP4VarlenMetadataRuntime final : public LaunchRuntime<FP4VarlenMetadataRuntime> {
public:
    static constexpr int kSharedWorkspaceRows = 1024;

    struct Args {
        LaunchArgs launch_args;
        int block_q, split_kv, workers, singleton_cost_multiplier, shared_rows;
        uint32_t nq;
        const uint32_t* lengths;
        const uint32_t* indices;
        uint32_t* metadata;
        uint32_t* units;
        uint64_t* suffix;
    };
    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/scheduler/paged_mqa_logits_scheduler.cuh>
extern "C" __global__ void attention_paged_mqa_logits_metadata_varlen(
    uint32_t nq, const uint32_t* lengths, const uint32_t* indices,
    uint32_t* metadata, uint32_t* units, uint64_t* suffix) {{
    deep_gemm::fp4_varlen_metadata<{}, {}, {}, {}, {}>(nq, lengths, indices,
        reinterpret_cast<deep_gemm::FP4VarlenCTAStart*>(metadata),
        units, suffix);
}}
)", args.block_q, args.split_kv, args.workers, args.singleton_cost_multiplier, args.shared_rows);
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.nq, args.lengths,
                                            args.indices, args.metadata, args.units, args.suffix));
    }
};

// ---------------------------------------------------------------- paged logits

// Host-side mirror of `cutlass::gemm::kernel::PPUPagedMqaLogits<...>::Arguments`.
//
// NOTES: the paged kernel takes this struct directly as its kernel argument -- unlike the non-paged
// one we deliberately do NOT wrap it in a `HostParams` block that the kernel unpacks. Constructing
// `Params` inside the kernel costs registers, and this kernel's vreg budget is already tight
// (measured 216/256): the wrapper pushed it to 232 with a 16-byte stack spill and demoted address
// arithmetic from scalar to vector registers, costing ~20% throughput.
struct PagedMqaLogitsArguments {
    const void* ptr_q;
    const void* ptr_k;
    const float* k_scales;
    const void* weights;
    uint32_t batch_size;
    uint64_t logits_stride;
    uint64_t kv_cache_stride_bytes;
    uint64_t block_table_stride;
    const uint32_t* context_lens;
    void* logits;
    const uint32_t* block_table;
    const uint32_t* schedule_meta;
    const uint32_t* indices;
};

// Host-side mirror of `PPUPagedMqaLogitsFP4<...>::Arguments`: same as above plus the packed e8m0
// scale pointers for Q and K.
template <typename TableOffset>
struct PagedMqaLogitsFP4Arguments {
    const void* ptr_q;
    const uint32_t* q_sf;
    const void* ptr_k;
    const uint32_t* k_sf;
    const void* weights;
    uint32_t batch_size;
    uint64_t logits_stride;
    uint64_t kv_cache_stride_bytes;
    TableOffset block_table_stride;
    const uint32_t* context_lens;
    void* logits;
    const uint32_t* block_table;
    const uint32_t* schedule_meta;
    const uint32_t* indices;
};

// One runtime for both paged flavours; they share the full template parameter list
// (SPLIT_MBLOCK is always false for non-FP4).
template <typename ArgumentsT>
class PagedMqaLogitsRuntime final : public LaunchRuntime<PagedMqaLogitsRuntime<ArgumentsT>> {
public:
    struct LaunchInfo {
        std::string include_header, kernel_class;
        std::string element_qk, element_acc, element_logits, element_weights;
        int next_n, num_heads, head_dim;
        int page_kv, split_kv, block_kv, warp_kv, num_q_stages, num_kv_stages;
        bool split_mblock;
        std::string scale_mode;
        int smem_size, num_threads;
        std::string kernel_name;
        std::string kernel_template_args;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        ArgumentsT kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        const auto& info = args.launch_info;
        return fmt::format(
            R"(
#include <{}>
namespace deep_gemm {{
using namespace cute;

using ElementQK = {};
using ElementAcc = {};
using ElementLogits = {};
using ElementWeights = {};
constexpr uint32_t kNextN = {};
constexpr uint32_t kNumHeads = {};
constexpr uint32_t kHeadDim = {};
constexpr uint32_t PAGE_KV = {};
constexpr uint32_t SPLIT_KV = {};
constexpr uint32_t BLOCK_KV = {};
constexpr uint32_t WARP_KV = {};
constexpr uint32_t kNumQStages = {};
constexpr uint32_t kNumKVStages = {};
constexpr bool SPLIT_MBLOCK = {};
constexpr uint32_t kScaleMode = {};

using AttnKernel = cutlass::gemm::kernel::{}<
  ElementQK, ElementAcc, ElementLogits, ElementWeights,
  kNextN, kNumHeads, kHeadDim, PAGE_KV, SPLIT_KV, BLOCK_KV, WARP_KV,
  kNumQStages, kNumKVStages, SPLIT_MBLOCK, kScaleMode{}
>;

// The host computes these instead of reading them off the kernel type, so pin them down here
static_assert(AttnKernel::SharedStorageSize == {}, "host/device shared memory size mismatch");
static_assert(AttnKernel::MaxThreadsPerBlock == {}, "host/device thread count mismatch");

extern "C"
__launch_bounds__(AttnKernel::MaxThreadsPerBlock, AttnKernel::MinBlocksPerMultiprocessor)
__global__ void {}(
  typename AttnKernel::Params params
) {{
  extern __shared__ char smem[];
  AttnKernel op;
  op(params, smem);
}}
}}
)",
            info.include_header, info.element_qk, info.element_acc, info.element_logits, info.element_weights,
            info.next_n, info.num_heads, info.head_dim, info.page_kv, info.split_kv, info.block_kv, info.warp_kv,
            info.num_q_stages, info.num_kv_stages, info.split_mblock, info.scale_mode, info.kernel_class,
            info.kernel_template_args,
            info.smem_size, info.num_threads, info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

// Debug-only readback of normalized request runs. Production dispatch and Graph
// replay consume the GPU metadata directly and never inspect its contents here.
static void add_paged_varlen_dump_params(DgProfParam& params, const uint32_t* indices,
                                       const uint32_t* context_lens, uint32_t nq) {
    // Compile-only launches do not populate scheduling metadata.
    if (is_compile_only())
        return;
    hggcStreamCaptureStatus capture = hggcStreamCaptureStatusNone;
    DG_HGGC_RUNTIME_CHECK(hggcStreamIsCapturing(current_stream(), &capture));
    if (capture != hggcStreamCaptureStatusNone)
        return;
    std::vector<uint32_t> lengths(nq), ids(nq), bases;
    DG_HGGC_RUNTIME_CHECK(hggcMemcpyAsync(ids.data(), indices, ids.size() * sizeof(uint32_t),
                                          hggcMemcpyDeviceToHost, current_stream()));
    DG_HGGC_RUNTIME_CHECK(hggcMemcpyAsync(lengths.data(), context_lens, nq * sizeof(uint32_t),
                                          hggcMemcpyDeviceToHost, current_stream()));
    DG_HGGC_RUNTIME_CHECK(hggcStreamSynchronize(current_stream()));
    uint32_t previous_id = 0;
    for (uint32_t q = 0; q < nq; ++q) {
        const uint32_t id = ids[q];
        if (q == 0 or id != previous_id) bases.push_back(lengths[q]);
        ids[q] = static_cast<uint32_t>(bases.size() - 1);
        previous_id = id;
    }
    auto render = [](const std::vector<uint32_t>& values) {
        std::string out = "[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i) out += ',';
            out += std::to_string(values[i]);
        }
        return out + "]";
    };
    params.add_params("varlen_indices", render(ids));
    params.add_params("context_lens", render(lengths));
    params.add_params("distribution", render(bases));
}

// Grouped dumps already carry per-request bases and exact per-query lengths.
// The shared legacy helper would overwrite distribution with per-query values.
static HgtxRangeGuard show_paged_varlen_ut(DgProfParam& params) {
    if (not profiling_enabled())
        return HgtxRangeGuard();
    const auto message = params.format();
    if (ProfilingInterface::instance().show_params())
        std::cout << message << std::endl;
    return HgtxRangeGuard(message);
}

// The grid has one CTA per metadata worker. Occupancy is queried only for diagnostics.
template <typename ArgumentsT>
static void launch_paged_mqa_logits(const std::string& include_header, const std::string& kernel_class,
                                    const std::string& scale_mode, const deep_gemm_mqa_common::PagedMqaConfig& config,
                                    const LaunchArgs& launch, const std::string& element_qk,
                                    const std::string& element_acc, const std::string& element_logits,
                                    const std::string& element_weights, const std::string& dtype_tag,
                                    int num_heads, int head_dim, int page_kv, int block_kv, int max_context_len,
                                    const std::string& kernel_name, bool is_avg,
                                    const ArgumentsT& kernel_params, bool wide_table_offset) {
    using Runtime = PagedMqaLogitsRuntime<ArgumentsT>;
    const bool is_fp4 = dtype_tag == "fp4";
    const auto& tile = config.tile;
    const bool grouped = config.grouped;
    const int next_n = config.next_n, num_threads = launch.block_dim.x, smem_size = launch.smem_size;

    std::string kernel_template_args;
    if (not is_fp4)
        kernel_template_args = fmt::format(", {}, {}", tile.direct_k_scale, grouped);
    else if (grouped)
        kernel_template_args = fmt::format(", {}", tile.tb_per_cu);
    if (wide_table_offset)
        kernel_template_args += is_fp4 and not grouped ? ", false, uint64_t" : ", uint64_t";

    const auto& args = typename Runtime::Args{
        .launch_info = {include_header, kernel_class, element_qk, element_acc, element_logits, element_weights,
                        next_n, num_heads, head_dim, page_kv, tile.split_kv, block_kv, tile.warp_kv,
                        tile.stage_q, tile.stage_k, tile.split_mblock, scale_mode, smem_size, num_threads, kernel_name,
                        kernel_template_args},
        .launch_args = launch,
        .kernel_params = kernel_params,
    };

    const auto& code = Runtime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code);

    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        const int num_sms = get_num_sms(), num_blocks = launch.grid_dim.x;
        int blocks_per_cu = 0;
        hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, runtime->kernel, num_threads, smem_size);
        int num_regs = 0, local_size = 0;
        hgFuncGetAttribute(&num_regs, HG_FUNC_ATTRIBUTE_NUM_REGS, runtime->kernel);
        hgFuncGetAttribute(&local_size, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, runtime->kernel);

        printf("[paged_mqa_logits%s:]\n", is_fp4 ? "_fp4" : "");
        printf("kNumHeads:%d, kHeadDim:%d%s, kNextN:%d, PAGE_KV:%d, SPLIT_KV:%d\n", num_heads, head_dim,
               is_fp4 ? "(packed)" : "", next_n, page_kv, tile.split_kv);
        printf("ThreadblockShape[%d, %d], WarpShape[%d, %d], kNumQStages:%d, kNumKVStages:%d\n", block_kv,
               config.block_n(), tile.warp_kv, config.mma_heads, tile.stage_q, tile.stage_k);
        printf("num_sms:%d, max_blocks_per_cu:%d, threadblock_count:%d, num_threads:%d, scheduled_blocks_per_cu:%d\n",
               num_sms, blocks_per_cu, num_blocks, num_threads, tile.tb_per_cu);
        printf("smem_size:%d, vreg:%d, stack:%d\n", smem_size, num_regs, local_size);
        printf("weights_bf16:%s\n", element_weights == "__ppu_bfloat16" ? "true" : "false");
        if (grouped)
            printf("schedule:request_grouped, block_q:%d\n", next_n);
        if (num_sms * tile.tb_per_cu != num_blocks) {
            printf("Warning: num_blocks(%d) differs from metadata plan: num_sms(%d) * scheduled_blocks_per_cu(%d) = %d\n",
                   num_blocks, num_sms, tile.tb_per_cu, num_sms * tile.tb_per_cu);
        }
    }

    DgProfParam dg_prof_params;
    if (deep_gemm::profiling_enabled()) {
        dg_prof_params.set_paged_mqa_logits_params(
            dtype_tag, static_cast<int>(kernel_params.batch_size), grouped ? 1 : next_n, num_heads,
            is_fp4 ? head_dim * 2 : head_dim,
            reinterpret_cast<int*>(const_cast<uint32_t*>(kernel_params.context_lens)), current_stream(), is_avg);
        dg_prof_params.add_params("page_kv", page_kv);
        dg_prof_params.add_params("max_context_len", max_context_len);
        if (element_logits == "__ppu_bfloat16")
            dg_prof_params.add_params("logits_dtype", std::string("bf16"));
        if (element_weights == "__ppu_bfloat16")
            dg_prof_params.add_params("weights_dtype", std::string("bf16"));
        if (grouped) {
            dg_prof_params.add_params("is_varlen", true);
            add_paged_varlen_dump_params(dg_prof_params, kernel_params.indices,
                                         kernel_params.context_lens, kernel_params.batch_size);
        }
    }
    {
        auto profiling_guard = grouped ? show_paged_varlen_ut(dg_prof_params) : deep_gemm::show_gemm_ut(dg_prof_params);
        Runtime::launch(runtime, args);
    }
}

// metadata_extra = (next_n, num_heads, logical_head_dim, qk_dtype).
// M890 torch::kInt8 denotes MXFP4; 810E torch::kInt8 denotes true INT8.
static torch::Tensor paged_mqa_logits_metadata(const torch::Tensor& context_lens, int batch_size, int page_kv,
                                               int num_sms,
                                               const std::tuple<int, int, int, torch::ScalarType>& metadata_extra,
                                               const std::optional<torch::Tensor>& indices = std::nullopt) {
    DG_HOST_ASSERT(page_kv == 32 or (page_kv > 0 and page_kv % 64 == 0));
    const auto& [next_n, num_heads, logical_head_dim, qk_dtype] = metadata_extra;
    const bool is_fp4 = get_ppu_arch() == AC_PPU0015 and qk_dtype == torch::kInt8;
    const int head_dim = is_fp4 ? logical_head_dim / 2 : logical_head_dim;
    const auto config = deep_gemm_mqa_common::get_paged_mqa_logits_metadata_config(
        next_n, page_kv, num_heads, head_dim, qk_dtype, batch_size, num_sms, is_fp4, indices.has_value());
    const bool grouped = config.block_q != 0;
    if (grouped and is_fp4) {
        const int rows = config.num_workers + 1;
        const int metadata_words = rows * 2;
        constexpr int SharedRows = FP4VarlenMetadataRuntime::kSharedWorkspaceRows;
        const int shared_rows = batch_size <= SharedRows ? SharedRows : 0;
        const int suffix_offset = align(metadata_words + batch_size, 2);
        // Retain the builder workspace through graph capture with the metadata owner.
        // Only the visible two-column view is consumed by the logits kernel.
        const int storage_words = shared_rows ? metadata_words : suffix_offset + batch_size * 2;
        auto storage = torch::empty({storage_words}, context_lens.options());
        auto metadata = storage.narrow(0, 0, metadata_words).view({rows, 2});
        auto* scratch = reinterpret_cast<uint32_t*>(storage.data_ptr());
        const auto& args = FP4VarlenMetadataRuntime::Args{
            .launch_args = {dim3(1, 1, 1), dim3(config.num_threads, 1, 1), config.smem_size},
            .block_q = config.block_q, .split_kv = config.split_kv, .workers = config.num_workers,
            .singleton_cost_multiplier = deep_gemm_mqa_common::FP4PagedVarlenConfig::kSingletonCostMultiplier,
            .shared_rows = shared_rows,
            .nq = static_cast<uint32_t>(batch_size),
            .lengths = reinterpret_cast<const uint32_t*>(context_lens.data_ptr()),
            .indices = reinterpret_cast<const uint32_t*>(indices->data_ptr()),
            .metadata = reinterpret_cast<uint32_t*>(metadata.data_ptr()),
            .units = shared_rows ? nullptr : scratch + metadata_words,
            .suffix = shared_rows ? nullptr : reinterpret_cast<uint64_t*>(scratch + suffix_offset),
        };
        const auto& runtime = compiler->build("attention_paged_mqa_logits_metadata_varlen",
                                              FP4VarlenMetadataRuntime::generate(args));
        FP4VarlenMetadataRuntime::launch(runtime, args);
        return metadata;
    }
    auto schedule_metadata = torch::empty(
        {config.num_workers + 1, 2}, context_lens.options());

    const std::string kernel_name = "attention_paged_mqa_logits_metadata";
    // kNumSMs is the total block count, rather than the number of physical CUs.
    const auto& args = PagedMqaLogitsMetadataRuntime::Args{
        .launch_info = {config.split_kv, config.num_workers, config.block_q, kernel_name},
        .launch_args = {dim3(1, 1, 1), dim3(config.num_threads, 1, 1), config.smem_size},
        .kernel_args = {static_cast<uint32_t>(batch_size), static_cast<uint32_t>(next_n),
                        reinterpret_cast<const uint32_t*>(context_lens.data_ptr()),
                        reinterpret_cast<uint32_t*>(schedule_metadata.data_ptr()),
                        grouped ? reinterpret_cast<const uint32_t*>(indices->data_ptr()) : nullptr},
    };
    const auto& code = PagedMqaLogitsMetadataRuntime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code);
    PagedMqaLogitsMetadataRuntime::launch(runtime, args);

    return schedule_metadata;
}

// Host entry for the paged MQA logits kernels. Recovers the value / scale views out of the fused KV
// cache, picks the tile config, allocates the padded logits buffer and launches.
//
// NOTES: shapes and dtypes have already been validated by `apis/attention.hpp`; the scalars are
// passed in so they are not re-extracted here.
static torch::Tensor paged_mqa_logits(const torch::Tensor& q, const torch::Tensor& fused_kv_cache,
                                      const torch::Tensor& weights, const torch::Tensor& context_lens,
                                      const torch::Tensor& block_table, const torch::Tensor& schedule_meta,
                                      int batch_size, int next_n, int num_heads, int head_dim, int num_kv_blocks,
                                      int page_kv, int schedule_meta_size, int max_context_len,
                                      torch::ScalarType logits_dtype, bool is_avg, const std::optional<torch::Tensor>& q_sf,
                                      const std::optional<torch::Tensor>& indices) {
    const bool is_fp4 = q_sf.has_value();
    const auto& qk_dtype = q.scalar_type();
    const int64_t kv_cache_stride_bytes = fused_kv_cache.stride(0);
    const int64_t block_table_stride = block_table.stride(0);

    // Each quantized page stores all K values first, then its per-token scales.
    auto k = fused_kv_cache.as_strided({num_kv_blocks, page_kv, head_dim}, {kv_cache_stride_bytes, head_dim, 1});
    if (not is_fp4)
        k = k.view(qk_dtype);
    auto k_scales = fused_kv_cache
                   .as_strided({num_kv_blocks, page_kv * 4LL}, {kv_cache_stride_bytes, 1},
                               fused_kv_cache.storage_offset() + static_cast<int64_t>(page_kv) * head_dim)
                   .view(is_fp4 ? torch::kInt32 : torch::kFloat32);

    const bool is_int8_avg = is_avg and not is_fp4 and qk_dtype == torch::kInt8;
    // Grouped dispatch follows indices, never the opaque metadata shape.
    const auto config = deep_gemm_mqa_common::get_paged_mqa_config(
        next_n, page_kv, num_heads, head_dim, qk_dtype, is_fp4, indices.has_value());
    const bool grouped = config.grouped;
    const auto& tile = config.tile;
    if (grouped) {
        TORCH_CHECK(schedule_meta_size == get_num_sms() * tile.tb_per_cu + 1,
                    is_fp4 ? "request indices require grouped MXFP4 metadata generated with the same indices and current num_sms"
                           : "request indices require grouped FP8 metadata generated with the same indices and current num_sms");
    } else if (is_fp4) {
        TORCH_CHECK(schedule_meta_size == get_num_sms() * tile.tb_per_cu + 1,
                    "FP4 paged logits require matching metadata_extra=(next_n, H, 128, torch.int8) on M890 "
                    "and the current num_sms");
    }
    if (not is_fp4)
        TORCH_CHECK(not tile.split_mblock, "SPLIT_MBLOCK is only supported for FP4 paged kernel");

    const int logits_stride_alignment = deep_gemm_mqa_common::get_logits_stride_alignment(logits_dtype);
    DG_HOST_ASSERT(logits_stride_alignment % tile.split_kv == 0);
    const int aligned_max_context_len = align(max_context_len, logits_stride_alignment);
    auto logits = torch::empty({batch_size * next_n, aligned_max_context_len},
                               torch::TensorOptions().dtype(logits_dtype).device(q.device()));

    // INT8 AVG uses the weights slot for Q scales; FP8/FP4 AVG needs no weights.
    const void* weights_ptr = is_avg and not is_int8_avg ? nullptr : weights.data_ptr();
    const std::string scale_mode = is_int8_avg ? "deep_gemm::kScaleModeQHead" : is_avg ? "deep_gemm::kScaleModeUnity" : "deep_gemm::kScaleModeWeights";

    const bool weights_bf16 = logits_dtype == torch::kBFloat16 and weights.scalar_type() == torch::kBFloat16;
    const std::string element_weights = weights_bf16 ? "__ppu_bfloat16" : "float";
    const std::string element_logits = logits_dtype == torch::kFloat32 ? "float" : "__ppu_bfloat16";

    // NOTES: `dtype_tag` names the kernel because the C++ JIT uses it as the `extern "C"` symbol
    // name, and the `cutlass::` qualifier in `element_qk` is not a valid identifier
    std::string element_qk, element_acc, dtype_tag;
    if (is_fp4) {
        element_qk = "uint8_t", element_acc = "float", dtype_tag = "fp4";
    } else if (qk_dtype == torch::kFloat8_e4m3fn) {
        element_qk = "cutlass::float_e4m3_t", element_acc = "float", dtype_tag = "fp8";
    } else {
        element_qk = "int8_t", element_acc = "int32_t", dtype_tag = "int8";
    }

    const int smem_size = deep_gemm_mqa_common::get_paged_smem_config(
        tile, config.block_n(), head_dim, static_cast<int>(q.element_size()), static_cast<int>(weights.element_size()),
        is_fp4, grouped);
    const int num_threads = deep_gemm_mqa_common::get_paged_num_threads(config.query_tiles, tile.split_kv, tile.warp_kv);
    const LaunchArgs launch{dim3(schedule_meta_size - 1), dim3(num_threads), smem_size};
    const auto& kernel_name = "attention_paged_mqa_logits_" + dtype_tag + (is_avg ? "_avg" : "");

    const auto* context_lens_ptr = reinterpret_cast<const uint32_t*>(context_lens.data_ptr());
    const auto* block_table_ptr = reinterpret_cast<const uint32_t*>(block_table.data_ptr());
    const auto* schedule_meta_ptr = reinterpret_cast<const uint32_t*>(schedule_meta.data_ptr());

    // Row offsets count int32 page IDs. Check the last row without overflowing the product.
    const bool wide_table_offset = batch_size > 1 and
        block_table_stride > std::numeric_limits<uint32_t>::max() / (batch_size - 1);
    const auto* group_ids = grouped ? reinterpret_cast<const uint32_t*>(indices->data_ptr()) : nullptr;
    if (is_fp4) {
        auto launch_fp4 = [&](auto table_stride) {
            auto args = PagedMqaLogitsFP4Arguments<decltype(table_stride)>{q.data_ptr(), reinterpret_cast<const uint32_t*>(q_sf->data_ptr()),
                k.data_ptr(), reinterpret_cast<const uint32_t*>(k_scales.data_ptr()), weights_ptr,
                static_cast<uint32_t>(batch_size), static_cast<uint64_t>(aligned_max_context_len),
                static_cast<uint64_t>(kv_cache_stride_bytes), table_stride,
                context_lens_ptr, logits.data_ptr(), block_table_ptr, schedule_meta_ptr, group_ids};
            launch_paged_mqa_logits(
                "deep_gemm/impls/fp4_paged_mqa_logits.cuh", grouped ? "PPUGroupedPagedMqaLogitsFP4" : "PPUPagedMqaLogitsFP4",
                scale_mode, config, launch, element_qk, element_acc, element_logits, element_weights, dtype_tag,
                num_heads, head_dim, page_kv, grouped ? deep_gemm_mqa_common::FP4PagedVarlenConfig::kBlockKV : kPagedMqaBlockKV,
                max_context_len, grouped ? kernel_name + "_grouped" : kernel_name, is_avg, args, wide_table_offset);
        };
        if (wide_table_offset)
            launch_fp4(static_cast<uint64_t>(block_table_stride));
        else
            launch_fp4(static_cast<uint32_t>(block_table_stride));
    } else {
        launch_paged_mqa_logits(
            "deep_gemm/impls/ppu_paged_mqa_logits.cuh", "PPUPagedMqaLogits", scale_mode, config, launch, element_qk, element_acc, element_logits,
            element_weights, dtype_tag, num_heads, head_dim, page_kv, kPagedMqaBlockKV, max_context_len,
            grouped ? kernel_name + "_grouped" : kernel_name, is_avg,
            PagedMqaLogitsArguments{q.data_ptr(), k.data_ptr(),
                                    k_scales.defined() ? k_scales.data_ptr<float>() : nullptr, weights_ptr,
                                    static_cast<uint32_t>(batch_size),
                                    static_cast<uint64_t>(aligned_max_context_len),
                                    static_cast<uint64_t>(kv_cache_stride_bytes),
                                    static_cast<uint64_t>(block_table_stride), context_lens_ptr, logits.data_ptr(),
                                    block_table_ptr, schedule_meta_ptr, group_ids}, wide_table_offset);
    }

    // NOTES: the kernel writes into the padded buffer, the caller only sees the valid window
    return logits.slice(1, 0, max_context_len);
}

} // namespace deep_gemm
