#pragma once

#include <torch/python.h>
#include <cctype>
#include <cstdint>
#include <limits>
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

// Parameter block handed to the generated kernel. This is *our own* fixed layout, not a mirror of
// `PPUMqaLogits<...>::Arguments`: the generated code unpacks it and builds the real `Params` on the
// device side. That buys two things:
//   - `stride_k` always travels as 64-bit and is narrowed to `StrideKType` by an explicit cast in
//     device code, so the host needs no template and makes no assumptions about byte order;
//   - the host never has to reproduce the kernel's struct layout, so a change to `Arguments` cannot
//     silently misalign the launch (the generated code carries a `sizeof` assertion as a backstop).
// One block serves both flavours; the fields a flavour does not use are left null.
struct MqaLogitsHostParams {
    const void* ptr_q;
    const uint32_t* q_sf;   // FP4 only (packed e8m0), null otherwise
    const void* ptr_k;
    const uint32_t* k_sf;   // FP4 only (packed e8m0), null otherwise
    const float* k_scales;  // FP8 / INT8 only, null for BF16 and FP4
    const void* weights;
    void* cu_seq_len_k_start;
    void* cu_seq_len_k_end;
    void* logits;
    uint32_t seq_len_q;
    uint32_t seq_len_k;
    uint64_t stride_k;
};

// Initialiser lists for `AttnKernel::Params`, evaluated in the generated device code where the
// element types exist. The field order follows each kernel's own `Arguments` declaration.
static constexpr const char* kMqaLogitsParamsInit =
    "(const ElementQK*)hp.ptr_q, (const ElementQK*)hp.ptr_k, hp.k_scales, "
    "(const ElementWeights*)hp.weights, (uint32_t*)hp.cu_seq_len_k_start, "
    "(uint32_t*)hp.cu_seq_len_k_end, (ElementLogits*)hp.logits, hp.seq_len_q, hp.seq_len_k, "
    "static_cast<StrideKType>(hp.stride_k)";
static constexpr const char* kMqaLogitsFP4ParamsInit =
    "(const ElementQK*)hp.ptr_q, hp.q_sf, (const ElementQK*)hp.ptr_k, hp.k_sf, "
    "(const ElementWeights*)hp.weights, (int*)hp.cu_seq_len_k_start, (int*)hp.cu_seq_len_k_end, "
    "(ElementLogits*)hp.logits, (int)hp.seq_len_q, (int)hp.seq_len_k, "
    "static_cast<StrideKType>(hp.stride_k)";

// NOTES: one runtime serves both flavours -- they take an identical template parameter list and now
// also share a single host-side parameter block, so nothing here needs templating.
class MqaLogitsRuntime final : public LaunchRuntime<MqaLogitsRuntime> {
public:
    struct LaunchInfo {
        std::string include_header, kernel_class;
        std::string element_qk, element_acc, element_logits, element_weights;
        int num_heads, head_dim;
        int block_qh, block_kv, warp_qh, warp_kv;
        int num_q_stages, num_kv_stages;
        std::string stride_k_type, params_init;
        bool is_compressed_logits;
        int smem_size, num_threads;
        std::string kernel_name;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        MqaLogitsHostParams kernel_params;
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
constexpr uint32_t kNumHeads = {};
constexpr uint32_t kHeadDim = {};
constexpr uint32_t BLOCK_QH = {};
constexpr uint32_t BLOCK_KV = {};
constexpr uint32_t WARP_QH = {};
constexpr uint32_t WARP_KV = {};
constexpr uint32_t kNumQStages = {};
constexpr uint32_t kNumKVStages = {};
using StrideKType = {};
constexpr bool kIsCompressedLogits = {};

using AttnKernel = cutlass::gemm::kernel::{}<
  ElementQK, ElementAcc, ElementLogits, ElementWeights,
  kNumHeads, kHeadDim, BLOCK_QH, BLOCK_KV, WARP_QH, WARP_KV,
  kNumQStages, kNumKVStages, StrideKType, kIsCompressedLogits
>;

// Must stay byte-identical to `deep_gemm::MqaLogitsHostParams` on the host side
struct HostParams {{
  const void* ptr_q;
  const uint32_t* q_sf;
  const void* ptr_k;
  const uint32_t* k_sf;
  const float* k_scales;
  const void* weights;
  void* cu_seq_len_k_start;
  void* cu_seq_len_k_end;
  void* logits;
  uint32_t seq_len_q;
  uint32_t seq_len_k;
  uint64_t stride_k;
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
  // `stride_k` arrives as 64-bit and is narrowed here, which is why the host needs no template
  typename AttnKernel::Params params{{{}}};
  extern __shared__ char smem[];
  AttnKernel op;
  op(params, smem);
}}
}}
)",
            info.include_header, info.element_qk, info.element_acc, info.element_logits, info.element_weights,
            info.num_heads, info.head_dim, info.block_qh, info.block_kv, info.warp_qh, info.warp_kv,
            info.num_q_stages, info.num_kv_stages, info.stride_k_type, info.is_compressed_logits, info.kernel_class,
            sizeof(MqaLogitsHostParams), info.smem_size, info.num_threads, info.kernel_name, info.params_init);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

// Dispatch helper: builds, launches and reports
static void launch_mqa_logits(const std::string& include_header, const std::string& kernel_class,
                              const std::string& stride_k_type, const std::string& params_init,
                              const deep_gemm_mqa_common::MqaLogitsConfig& config, const std::string& element_qk,
                              const std::string& element_acc, const std::string& element_logits,
                              const std::string& element_weights, int num_heads, int head_dim, bool is_compressed,
                              int smem_size, int num_threads, const std::string& kernel_name,
                              const MqaLogitsHostParams& kernel_params) {
    using Runtime = MqaLogitsRuntime;
    const int num_sms = get_num_sms();
    const dim3 block(num_threads, 1, 1);
    const dim3 grid(num_sms, 1, 1);

    auto args = typename Runtime::Args{
        .launch_info = {include_header, kernel_class, element_qk, element_acc, element_logits, element_weights,
                        num_heads, head_dim, config.block_qh, config.block_kv, config.warp_qh, config.warp_kv,
                        config.num_q_stages, config.num_kv_stages, stride_k_type, params_init, is_compressed,
                        smem_size, num_threads, kernel_name},
        .launch_args = {grid, block, smem_size},
        .kernel_params = kernel_params,
    };

    const auto& code = Runtime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code, num_threads, smem_size);

    // Persistent kernel: one wave of `num_sms * blocks_per_cu` blocks, mirroring
    // `num_sms * compute_occupancy_for_kernel<AttnKernel>()` in `Attention::run`
    int blocks_per_cu = 0;
    hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, runtime->kernel, num_threads, smem_size);
    args.launch_args.grid_dim.x *= blocks_per_cu;

    Runtime::launch(runtime, args);

    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        printf("[mqa_logits:]\n");
        printf("kNumHeads:%d, kHeadDim:%d, seq_len_q:%u, seq_len_k:%u\n", num_heads, head_dim,
               kernel_params.seq_len_q, kernel_params.seq_len_k);
        printf("ThreadblockShape[%d, %d], WarpShape[%d, %d], kNumQStages:%d, kNumKVStages:%d\n", config.block_qh,
               config.block_kv, config.warp_qh, config.warp_kv, config.num_q_stages, config.num_kv_stages);
        printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%u, num_threads:%d\n", num_sms, blocks_per_cu,
               args.launch_args.grid_dim.x, num_threads);
        printf("smem_size:%d, compressed_logits:%s\n", smem_size, is_compressed ? "true" : "false");
    }
}

// Host entry for the non-paged MQA logits kernels. Picks the tile config, allocates the padded
// logits buffer, dispatches on `StrideKType` / FP4-ness, and applies the optional out-of-range
// masking. Returns the logits sliced down to the caller-visible window.
//
// NOTES: shapes and dtypes have already been validated by `apis/attention.hpp`; the scalars are
// passed in so they are not re-extracted here.
static torch::Tensor mqa_logits(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& k_scales,
                                const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
                                const torch::Tensor& cu_seq_len_k_end, int seq_len_q, int seq_len_k, int num_heads,
                                int head_dim, bool clean_logits, int max_seqlen_k, torch::ScalarType logits_dtype,
                                const std::optional<torch::Tensor>& q_sf,
                                const std::optional<torch::Tensor>& k_sf) {
    const bool is_fp4 = q_sf.has_value();
    const auto& qk_dtype = q.scalar_type();
    const bool is_compressed = max_seqlen_k > 0;

    const auto& config =
        deep_gemm_mqa_common::get_best_configs(qk_dtype, num_heads, seq_len_k, logits_dtype, is_fp4);

    const int aligned_seq_len = align(seq_len_q, config.block_q);
    const int logits_stride_alignment = deep_gemm_mqa_common::get_logits_stride_alignment(logits_dtype);
    DG_HOST_ASSERT(logits_stride_alignment % config.block_kv == 0);
    const int aligned_seq_len_kv = is_compressed ? align(max_seqlen_k, logits_stride_alignment)
                                                  : align(seq_len_k + config.block_kv, logits_stride_alignment);
    const auto& stride_k_type = deep_gemm_mqa_common::select_stride_k_type(aligned_seq_len, aligned_seq_len_kv);

    auto logits = torch::empty({aligned_seq_len, aligned_seq_len_kv},
                               torch::TensorOptions().dtype(logits_dtype).device(q.device()));
    const int logits_cols = is_compressed ? max_seqlen_k : seq_len_k;

    const bool weights_bf16 = logits_dtype == torch::kBFloat16 and weights.scalar_type() == torch::kBFloat16;
    const std::string element_weights = weights_bf16 ? "__ppu_bfloat16" : "float";
    const std::string element_logits = logits_dtype == torch::kFloat32 ? "float" : "__ppu_bfloat16";

    // NOTES: `dtype_tag` (rather than `element_qk`) names the kernel, because the C++ JIT uses this
    // string as the `extern "C"` symbol name -- the `cutlass::` qualifier in `element_qk` is not a
    // valid identifier. The Python path only used it as a cache directory name, so it got away with it.
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

    const int smem_size = deep_gemm_mqa_common::get_smem_config(
        config, head_dim, static_cast<int>(q.element_size()), static_cast<int>(weights.element_size()), is_fp4);
    const int num_threads = deep_gemm_mqa_common::get_num_threads(config);
    const auto& kernel_name = "attention_mqa_logits_" + dtype_tag;

    // `k_scales` is an empty tensor for BF16 and FP4 (FP4 uses `k_sf` instead)
    const float* k_scales_ptr = (is_fp4 or qk_dtype == torch::kBFloat16) ? nullptr : k_scales.data_ptr<float>();

    // One parameter block for both flavours; the generated code picks the fields it needs and
    // narrows `stride_k` to `StrideKType` on the device side
    const MqaLogitsHostParams params{
        q.data_ptr(),
        is_fp4 ? reinterpret_cast<const uint32_t*>(q_sf->data_ptr()) : nullptr,
        k.data_ptr(),
        is_fp4 ? reinterpret_cast<const uint32_t*>(k_sf->data_ptr()) : nullptr,
        k_scales_ptr,
        weights.data_ptr(),
        cu_seq_len_k_start.data_ptr(),
        cu_seq_len_k_end.data_ptr(),
        logits.data_ptr(),
        static_cast<uint32_t>(seq_len_q),
        static_cast<uint32_t>(seq_len_k),
        static_cast<uint64_t>(aligned_seq_len_kv),
    };

    launch_mqa_logits(is_fp4 ? "fp4_mqa_logits.cuh" : "ppu_mqa_logits.cuh",
                      is_fp4 ? "PPUMqaLogitsFP4" : "PPUMqaLogits", stride_k_type,
                      is_fp4 ? kMqaLogitsFP4ParamsInit : kMqaLogitsParamsInit, config, element_qk, element_acc,
                      element_logits, element_weights, num_heads, head_dim, is_compressed, smem_size, num_threads,
                      kernel_name, params);

    // NOTES: the kernel writes into the padded buffer, the caller only sees the valid window
    logits = logits.slice(0, 0, seq_len_q).slice(1, 0, logits_cols);

    if (clean_logits) {
        const auto& positions =
            torch::arange(0, seq_len_k, torch::TensorOptions().dtype(torch::kInt32).device(q.device()));
        const auto& mask = positions.unsqueeze(0).ge(cu_seq_len_k_start.unsqueeze(1)) &
                           positions.unsqueeze(0).lt(cu_seq_len_k_end.unsqueeze(1));
        logits = logits.masked_fill(mask.logical_not(), -std::numeric_limits<float>::infinity());
    }
    return logits;
}

} // namespace deep_gemm
