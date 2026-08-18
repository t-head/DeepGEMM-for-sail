#pragma once

#include <torch/python.h>
#include <cctype>
#include <cstdint>
#include <string>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/layout.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"
#include "../heuristics/common_mqa.hpp"

namespace deep_gemm {

// ---------------------------------------------------------------- metadata

// Builds the `schedule_metadata` table consumed by the paged kernel. A tiny single-block kernel.
class PagedMqaLogitsMetadataRuntime final : public LaunchRuntime<PagedMqaLogitsMetadataRuntime> {
public:
    struct KernelArguments {
        uint32_t batch_size;
        uint32_t next_n;
        const uint32_t* context_lens;
        uint32_t* schedule_metadata;
    };

    struct LaunchInfo {
        int split_kv, num_sms;
        std::string kernel_name;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        KernelArguments kernel_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(
            R"(
#include <paged_mqa_logits_scheduler.cuh>
namespace deep_gemm {{

constexpr uint32_t SPLIT_KV = {};
constexpr uint32_t kNumSMs = {};

extern "C"
__global__ void {}(
  const uint32_t batch_size,
  const uint32_t next_n,
  const uint32_t* __restrict__ context_lens,
  uint32_t* __restrict__ schedule_metadata
) {{
  smxx_paged_mqa_logits_metadata_device<SPLIT_KV, kNumSMs>(batch_size, next_n, context_lens, schedule_metadata);
}}
}}
)",
            args.launch_info.split_kv, args.launch_info.num_sms, args.launch_info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.kernel_args.batch_size, args.kernel_args.next_n,
                                    args.kernel_args.context_lens, args.kernel_args.schedule_metadata));
    }
};

// ---------------------------------------------------------------- paged logits

// Parameter block handed to the generated paged kernels -- our own fixed layout, not a mirror of
// `PPUPagedMqaLogits<...>::Arguments`. The generated code unpacks it and builds the real `Params`
// device-side, so a change to the kernel's struct cannot silently misalign the launch. One block
// serves both flavours; FP4-only fields are left null for the others.
struct PagedMqaLogitsHostParams {
    const void* ptr_q;
    const uint32_t* q_sf;   // FP4 only (packed e8m0), null otherwise
    const void* ptr_k;
    const uint32_t* k_sf;   // FP4 only (packed e8m0), null otherwise
    const float* k_scales;  // FP8 / INT8 only, null for BF16 and FP4
    const void* weights;
    uint32_t batch_size;
    uint64_t logits_stride;
    uint64_t kv_cache_stride_bytes;
    uint32_t block_table_stride;
    const uint32_t* context_lens;
    void* logits;
    const uint32_t* block_table;
    const uint32_t* schedule_meta;
};

// Initialiser lists for `AttnKernel::Params`, evaluated in the generated device code. The field order
// follows each kernel's own `Arguments` declaration.
static constexpr const char* kPagedParamsInit =
    "(const ElementQK*)hp.ptr_q, (const ElementQK*)hp.ptr_k, hp.k_scales, "
    "(const ElementWeights*)hp.weights, hp.batch_size, hp.logits_stride, hp.kv_cache_stride_bytes, "
    "hp.block_table_stride, hp.context_lens, (ElementLogits*)hp.logits, hp.block_table, "
    "hp.schedule_meta";
static constexpr const char* kPagedFP4ParamsInit =
    "(const ElementQK*)hp.ptr_q, hp.q_sf, (const ElementQK*)hp.ptr_k, hp.k_sf, "
    "(const ElementWeights*)hp.weights, hp.batch_size, hp.logits_stride, hp.kv_cache_stride_bytes, "
    "hp.block_table_stride, hp.context_lens, (ElementLogits*)hp.logits, hp.block_table, "
    "hp.schedule_meta";

// One runtime for both paged flavours; they share the template parameter list up to the trailing
// `SPLIT_MBLOCK` that only FP4 takes, emitted via `extra_template_args`.
class PagedMqaLogitsRuntime final : public LaunchRuntime<PagedMqaLogitsRuntime> {
public:
    struct LaunchInfo {
        std::string include_header, kernel_class;
        std::string element_qk, element_acc, element_logits, element_weights;
        int next_n, num_heads, head_dim;
        int block_kv, warp_kv, num_q_stages, num_kv_stages, split_kv;
        std::string extra_template_args, params_init;
        int smem_size, num_threads;
        std::string kernel_name;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        PagedMqaLogitsHostParams kernel_params;
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
constexpr uint32_t BLOCK_KV = {};
constexpr uint32_t WARP_KV = {};
constexpr uint32_t kNumQStages = {};
constexpr uint32_t kNumKVStages = {};
constexpr uint32_t SPLIT_KV = {};

using AttnKernel = cutlass::gemm::kernel::{}<
  ElementQK, ElementAcc, ElementLogits, ElementWeights,
  kNextN, kNumHeads, kHeadDim, BLOCK_KV, WARP_KV,
  kNumQStages, kNumKVStages, SPLIT_KV{}
>;

// Must stay byte-identical to `deep_gemm::PagedMqaLogitsHostParams` on the host side
struct HostParams {{
  const void* ptr_q;
  const uint32_t* q_sf;
  const void* ptr_k;
  const uint32_t* k_sf;
  const float* k_scales;
  const void* weights;
  uint32_t batch_size;
  uint64_t logits_stride;
  uint64_t kv_cache_stride_bytes;
  uint32_t block_table_stride;
  const uint32_t* context_lens;
  void* logits;
  const uint32_t* block_table;
  const uint32_t* schedule_meta;
}};
static_assert(sizeof(HostParams) == {}, "host/device parameter block size mismatch");

// The host computes these instead of reading them off the kernel type, so pin them down here
static_assert(AttnKernel::SharedStorageSize == {}, "host/device shared memory size mismatch");
static_assert(AttnKernel::MaxThreadsPerBlock == {}, "host/device thread count mismatch");

extern "C"
__launch_bounds__(AttnKernel::MaxThreadsPerBlock, AttnKernel::MinBlocksPerMultiprocessor)
__global__ void {}(
  HostParams hp
) {{
  typename AttnKernel::Params params{{{}}};
  extern __shared__ char smem[];
  AttnKernel op;
  op(params, smem);
}}
}}
)",
            info.include_header, info.element_qk, info.element_acc, info.element_logits, info.element_weights,
            info.next_n, info.num_heads, info.head_dim, info.block_kv, info.warp_kv, info.num_q_stages,
            info.num_kv_stages, info.split_kv, info.kernel_class, info.extra_template_args,
            sizeof(PagedMqaLogitsHostParams), info.smem_size, info.num_threads, info.kernel_name, info.params_init);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

// Dispatch helper. Unlike the non-paged kernel the grid is not derived from occupancy: it is
// `num_blocks`, which the caller took from the `schedule_meta` table built earlier.
static void launch_paged_mqa_logits(const std::string& include_header, const std::string& kernel_class,
                                    const std::string& extra_template_args, const std::string& params_init,
                                    const deep_gemm_mqa_common::PagedTile& tile, const std::string& element_qk,
                                    const std::string& element_acc, const std::string& element_logits,
                                    const std::string& element_weights, int next_n, int num_heads, int head_dim,
                                    int block_kv, int smem_size, int num_threads, int num_blocks,
                                    const std::string& kernel_name,
                                    const PagedMqaLogitsHostParams& kernel_params) {
    using Runtime = PagedMqaLogitsRuntime;
    const dim3 block(num_threads, 1, 1);
    const dim3 grid(num_blocks, 1, 1);

    const auto& args = typename Runtime::Args{
        .launch_info = {include_header, kernel_class, element_qk, element_acc, element_logits, element_weights,
                        next_n, num_heads, head_dim, block_kv, tile.warp_kv, tile.stage_q, tile.stage_k,
                        tile.split_kv, extra_template_args, params_init, smem_size, num_threads, kernel_name},
        .launch_args = {grid, block, smem_size},
        .kernel_params = kernel_params,
    };

    const auto& code = Runtime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code, num_threads, smem_size);
    Runtime::launch(runtime, args);

    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        printf("[paged_mqa_logits:]\n");
        printf("kNumHeads:%d, kHeadDim:%d, kNextN:%d, BLOCK_KV:%d, SPLIT_KV:%d\n", num_heads, head_dim, next_n,
               block_kv, tile.split_kv);
        printf("WARP_KV:%d, kNumQStages:%d, kNumKVStages:%d, split_mblock:%s\n", tile.warp_kv, tile.stage_q,
               tile.stage_k, tile.split_mblock ? "true" : "false");
        printf("threadblock_count:%d, num_threads:%d, smem_size:%d\n", num_blocks, num_threads, smem_size);
    }
}

// Host entry for the `schedule_metadata` table. `metadata_extra` = (next_n, num_heads, head_dim,
// element_size); when absent we fall back to the compatibility tile.
static torch::Tensor paged_mqa_logits_metadata(const torch::Tensor& context_lens, int batch_size, int block_kv,
                                              int num_sms,
                                              const std::optional<std::tuple<int, int, int, int>>& metadata_extra) {
    int next_n = context_lens.size(1);
    int tb_per_cu = 1;
    // fallback: assume max num_math_warpgroups=4 to avoid crash
    int split_kv = block_kv * 4;
    if (metadata_extra.has_value()) {
        const auto& [extra_next_n, num_heads, head_dim, element_size] = *metadata_extra;
        next_n = extra_next_n;
        const auto& tile =
            deep_gemm_mqa_common::get_paged_mqa_logits_tile(next_n, block_kv, num_heads, head_dim, element_size);
        split_kv = tile.split_kv, tb_per_cu = tile.tb_per_cu;
    }

    const int num_blocks = num_sms * tb_per_cu;
    auto schedule_metadata = torch::empty(
        {num_blocks + 1, 2}, torch::TensorOptions().dtype(context_lens.scalar_type()).device(context_lens.device()));

    const std::string kernel_name = "attention_paged_mqa_logits_metadata";
    const int num_threads = std::min(batch_size, 1024);
    const int smem_size = batch_size * 4;
    // NOTES: `kNumSMs` is the total block count, not `num_sms` -- matching the Python template key
    const auto& args = PagedMqaLogitsMetadataRuntime::Args{
        .launch_info = {split_kv, num_blocks, kernel_name},
        .launch_args = {dim3(1, 1, 1), dim3(num_threads, 1, 1), smem_size},
        .kernel_args = {static_cast<uint32_t>(batch_size), static_cast<uint32_t>(next_n),
                        reinterpret_cast<const uint32_t*>(context_lens.data_ptr()),
                        reinterpret_cast<uint32_t*>(schedule_metadata.data_ptr())},
    };
    const auto& code = PagedMqaLogitsMetadataRuntime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code, num_threads, smem_size);
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
                                      int block_kv, int schedule_meta_size, int max_context_len,
                                      torch::ScalarType logits_dtype, const std::optional<torch::Tensor>& q_sf) {
    const bool is_fp4 = q_sf.has_value();
    const auto& qk_dtype = q.scalar_type();
    const int64_t kv_cache_stride_bytes = fused_kv_cache.stride(0);
    const int64_t block_table_stride = block_table.stride(0);

    // Derive the value and scale-factor views out of the fused KV cache: each row stores the values
    // first, immediately followed by the scale bytes
    auto k = fused_kv_cache.as_strided({num_kv_blocks, block_kv, head_dim}, {kv_cache_stride_bytes, head_dim, 1});
    if (not is_fp4)
        k = k.view(qk_dtype);
    torch::Tensor k_scales;
    if (is_fp4 or qk_dtype != torch::kBFloat16) {
        k_scales = fused_kv_cache
                       .as_strided({num_kv_blocks, block_kv * 4}, {kv_cache_stride_bytes, 1}, block_kv * head_dim)
                       .view(is_fp4 ? torch::kInt32 : torch::kFloat32);
    }

    const auto& tile = deep_gemm_mqa_common::get_paged_mqa_logits_tile(next_n, block_kv, num_heads, head_dim,
                                                                      static_cast<int>(q.element_size()));
    if (not is_fp4)
        TORCH_CHECK(not tile.split_mblock, "SPLIT_MBLOCK is only supported for FP4 paged kernel");

    const int logits_stride_alignment = deep_gemm_mqa_common::get_logits_stride_alignment(logits_dtype);
    DG_HOST_ASSERT(logits_stride_alignment % tile.split_kv == 0);
    const int aligned_max_context_len = align(max_context_len, logits_stride_alignment);
    auto logits = torch::empty({batch_size * next_n, aligned_max_context_len},
                               torch::TensorOptions().dtype(logits_dtype).device(q.device()));

    const bool weights_bf16 = logits_dtype == torch::kBFloat16 and weights.scalar_type() == torch::kBFloat16;
    const std::string element_weights = weights_bf16 ? "__ppu_bfloat16" : "float";
    const std::string element_logits = logits_dtype == torch::kFloat32 ? "float" : "__ppu_bfloat16";

    // NOTES: `dtype_tag` names the kernel because the C++ JIT uses it as the `extern "C"` symbol
    // name, and the `cutlass::` qualifier in `element_qk` is not a valid identifier
    std::string element_qk, element_acc, dtype_tag;
    if (is_fp4) {
        element_qk = "uint8_t", element_acc = "float", dtype_tag = "fp4";
    } else if (qk_dtype == torch::kBFloat16) {
        element_qk = "cutlass::bfloat16_t", element_acc = "float", dtype_tag = "bf16";
    } else if (qk_dtype == torch::kFloat8_e4m3fn) {
        element_qk = "cutlass::float_e4m3_t", element_acc = "float", dtype_tag = "fp8";
    } else {
        element_qk = "int8_t", element_acc = "int32_t", dtype_tag = "int8";
    }

    const int smem_size = deep_gemm_mqa_common::get_paged_smem_config(
        tile, next_n, num_heads, head_dim, static_cast<int>(q.element_size()),
        static_cast<int>(weights.element_size()), is_fp4);
    const int num_threads = deep_gemm_mqa_common::get_paged_num_threads(tile, next_n);
    const int num_blocks = schedule_meta_size - 1;
    const auto& kernel_name = "attention_paged_mqa_logits_" + dtype_tag;

    const auto* context_lens_ptr = reinterpret_cast<const uint32_t*>(context_lens.data_ptr());
    const auto* block_table_ptr = reinterpret_cast<const uint32_t*>(block_table.data_ptr());
    const auto* schedule_meta_ptr = reinterpret_cast<const uint32_t*>(schedule_meta.data_ptr());

    // One parameter block for both flavours; the generated code picks the fields it needs
    const PagedMqaLogitsHostParams params{
        q.data_ptr(),
        is_fp4 ? reinterpret_cast<const uint32_t*>(q_sf->data_ptr()) : nullptr,
        k.data_ptr(),
        is_fp4 ? reinterpret_cast<const uint32_t*>(k_scales.data_ptr()) : nullptr,
        (not is_fp4 and k_scales.defined()) ? k_scales.data_ptr<float>() : nullptr,
        weights.data_ptr(),
        static_cast<uint32_t>(batch_size),
        static_cast<uint64_t>(aligned_max_context_len),
        static_cast<uint64_t>(kv_cache_stride_bytes),
        static_cast<uint32_t>(block_table_stride),
        context_lens_ptr,
        logits.data_ptr(),
        block_table_ptr,
        schedule_meta_ptr,
    };

    launch_paged_mqa_logits(is_fp4 ? "fp4_paged_mqa_logits.cuh" : "ppu_paged_mqa_logits.cuh",
                            is_fp4 ? "PPUPagedMqaLogitsFP4" : "PPUPagedMqaLogits",
                            is_fp4 ? (tile.split_mblock ? ", true" : ", false") : "",
                            is_fp4 ? kPagedFP4ParamsInit : kPagedParamsInit, tile, element_qk, element_acc,
                            element_logits, element_weights, next_n, num_heads, head_dim, block_kv, smem_size,
                            num_threads, num_blocks, kernel_name, params);

    // NOTES: the kernel writes into the padded buffer, the caller only sees the valid window
    return logits.slice(1, 0, max_context_len);
}

} // namespace deep_gemm
