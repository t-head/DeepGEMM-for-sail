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
#include <deep_gemm/common/profiling_interface.cuh>

namespace deep_gemm {

// Spells out `StrideKType` for the generated code, tied to the template argument so the two can
// never drift apart.
template <typename T>
struct StrideKTypeName;
template <>
struct StrideKTypeName<uint32_t> {
    static constexpr const char* value = "uint32_t";
};
template <>
struct StrideKTypeName<uint64_t> {
    static constexpr const char* value = "uint64_t";
};

// Host-side mirror of `cutlass::gemm::kernel::PPUMqaLogits<...>::Arguments`.
//
// NOTES: the kernel takes this struct directly as its kernel argument -- we deliberately do NOT
// wrap it in a `HostParams` block that the kernel unpacks. Constructing `Params` inside the kernel
// costs registers and can push a pressure-sensitive kernel over its vreg budget (a stack spill was
// observed on the paged kernel from exactly that pattern). `StrideKType` is picked at runtime
// (`uint32_t` or `uint64_t`) and changes the struct layout, hence the template; the value pointers
// stay `void*` because their element types only exist inside the generated device code.
template <typename StrideKType>
struct MqaLogitsArguments {
    const void* ptr_q;
    const void* ptr_k;
    const float* k_scales;
    const void* weights;
    uint32_t* cu_seq_len_k_start;
    uint32_t* cu_seq_len_k_end;
    void* logits;
    uint32_t seq_len_q;
    uint32_t seq_len_k;
    StrideKType stride_k;
};

// Host-side mirror of `cutlass::gemm::kernel::PPUMqaLogitsFP4<...>::Arguments`. Same shape as the
// non-FP4 one plus the packed e8m0 scale pointers for Q and K.
template <typename StrideKType>
struct MqaLogitsFP4Arguments {
    const void* ptr_q;
    const uint32_t* q_sf;
    const void* ptr_k;
    const uint32_t* k_sf;
    const void* weights;
    int* cu_seq_len_k_start;
    int* cu_seq_len_k_end;
    void* logits;
    int seq_len_q;
    int seq_len_k;
    StrideKType stride_k;
};

// The FP4 and non-FP4 kernels take an identical template parameter list, differing only in
// header and class name, so one runtime serves both -- parameterised by the `Arguments` mirror.
template <typename StrideKType, typename ArgumentsT>
class MqaLogitsRuntime final : public LaunchRuntime<MqaLogitsRuntime<StrideKType, ArgumentsT>> {
public:
    struct LaunchInfo {
        std::string include_header, kernel_class;
        std::string element_qk, element_acc, element_logits, element_weights;
        int num_heads, head_dim;
        int block_qh, block_kv, warp_qh, warp_kv;
        int num_q_stages, num_kv_stages;
        std::string stride_k_type;
        bool is_compressed_logits;
        std::string scale_mode;
        int smem_size, num_threads;
        std::string kernel_name;
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
constexpr uint32_t kScaleMode = {};

using AttnKernel = cutlass::gemm::kernel::{}<
  ElementQK, ElementAcc, ElementLogits, ElementWeights,
  kNumHeads, kHeadDim, BLOCK_QH, BLOCK_KV, WARP_QH, WARP_KV,
  kNumQStages, kNumKVStages, StrideKType, kIsCompressedLogits, kScaleMode
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
            info.num_heads, info.head_dim, info.block_qh, info.block_kv, info.warp_qh, info.warp_kv,
            info.num_q_stages, info.num_kv_stages, info.stride_k_type, info.is_compressed_logits, info.scale_mode,
            info.kernel_class, info.smem_size, info.num_threads, info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

// Dispatch helper: builds, launches and reports, for one concrete `StrideKType` / `Arguments` pair
template <typename StrideKType, typename ArgumentsT>
static void launch_mqa_logits(const std::string& include_header, const std::string& kernel_class,
                              const deep_gemm_mqa_common::MqaLogitsConfig& config, const std::string& element_qk,
                              const std::string& element_acc, const std::string& element_logits,
                              const std::string& element_weights, const std::string& dtype_tag, int num_heads,
                              int head_dim, bool is_compressed, const std::string& scale_mode, int smem_size,
                              int num_threads, const std::string& kernel_name, const ArgumentsT& kernel_params) {
    using Runtime = MqaLogitsRuntime<StrideKType, ArgumentsT>;
    const bool is_fp4 = dtype_tag == "fp4";
    const bool is_avg = scale_mode != "deep_gemm::kScaleModeWeights";
    const int num_sms = get_num_sms();
    const dim3 block(num_threads, 1, 1);
    const dim3 grid(num_sms, 1, 1);

    auto args = typename Runtime::Args{
        .launch_info = {include_header, kernel_class, element_qk, element_acc, element_logits, element_weights,
                        num_heads, head_dim, config.block_qh, config.block_kv, config.warp_qh, config.warp_kv,
                        config.num_q_stages, config.num_kv_stages, StrideKTypeName<StrideKType>::value,
                        is_compressed, scale_mode, smem_size, num_threads, kernel_name},
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

    // Reported before the launch, as `Attention::run` does, so the configuration is on screen even
    // when the launch itself fails
    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {

        int num_regs = 0, local_size = 0;
        hgFuncGetAttribute(&num_regs, HG_FUNC_ATTRIBUTE_NUM_REGS, runtime->kernel);
        hgFuncGetAttribute(&local_size, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, runtime->kernel);

        if (is_fp4) {
            printf("[mqa_logits_fp4:]\n");
            printf("kNumHeads:%d, kHeadDim:%d(packed), BLOCK_QH:%d, BLOCK_KV:%d\n", num_heads, head_dim,
                   config.block_qh, config.block_kv);
            printf("ThreadblockShape[%d, %d], WarpShape[%d, %d], kNumQStages:%d, kNumKVStages:%d\n", config.block_kv,
                   config.block_qh, config.warp_kv, config.warp_qh, config.num_q_stages, config.num_kv_stages);
            printf("num_sms:%d, max_blocks_per_cu:%d, threadblock_count:%d, num_threads:%d\n", num_sms, blocks_per_cu,
                   static_cast<int>(args.launch_args.grid_dim.x), num_threads);
        } else {
            printf("[mqa_logits:]\n");
            printf("kNumHeads:%d, kHeadDim:%d, seq_len_q:%u, seq_len_k:%u, stride_k:%llu\n", num_heads, head_dim,
                   static_cast<unsigned>(kernel_params.seq_len_q), static_cast<unsigned>(kernel_params.seq_len_k),
                   static_cast<unsigned long long>(kernel_params.stride_k));
            printf("ThreadblockShape[%d, %d], WarpShape[%d, %d], kNumQStages:%d, kNumKVStages:%d\n", config.block_qh,
                   config.block_kv, config.warp_qh, config.warp_kv, config.num_q_stages, config.num_kv_stages);
            printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d, num_threads:%d\n", num_sms, blocks_per_cu,
                   static_cast<int>(args.launch_args.grid_dim.x), num_threads);
        }
        printf("smem_size:%d, vreg:%d, stack:%d\n", smem_size, num_regs, local_size);
        printf("compressed_logits:%s, ", is_compressed ? "true" : "false");
        printf("weights_bf16:%s\n", element_weights == "__ppu_bfloat16" ? "true" : "false");
    }

    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_mqa_logits_params(dtype_tag, static_cast<int>(kernel_params.seq_len_q),
                                             static_cast<int>(kernel_params.seq_len_k), num_heads,
                                             is_fp4 ? head_dim * 2 : head_dim, current_stream(), is_avg);
        if (element_logits == "__ppu_bfloat16")
            dg_prof_params.add_params("logits_dtype", std::string("bf16"));
        if (element_weights == "__ppu_bfloat16")
            dg_prof_params.add_params("weights_dtype", std::string("bf16"));
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);

    Runtime::launch(runtime, args);

    ProfilingInterface::Instance().instrument(false, dg_prof_params);
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
                                const std::optional<torch::Tensor>& k_sf,
                                const std::optional<torch::Tensor>& q_scale = std::nullopt) {
    const bool is_fp4 = q_sf.has_value();
    const auto& qk_dtype = q.scalar_type();
    const bool is_compressed = max_seqlen_k > 0;
    // Avg variant: an empty weights tensor selects it, and the `weights` kernel slot then
    // carries `q_scale` (nullptr for the unity mode -- the kernel never dereferences it)
    const bool is_avg = weights.numel() == 0;
    std::string scale_mode = "deep_gemm::kScaleModeWeights";
    const void* weights_ptr = weights.data_ptr();
    if (is_avg) {
        if (not q_scale.has_value()) {
            scale_mode = "deep_gemm::kScaleModeUnity";
            weights_ptr = nullptr;
        } else if (q_scale->numel() == 1) {
            scale_mode = "deep_gemm::kScaleModeQScalar";
            weights_ptr = q_scale->data_ptr();
        } else {
            scale_mode = "deep_gemm::kScaleModeQRow";
            weights_ptr = q_scale->data_ptr();
        }
    }

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
    const auto& kernel_name = "attention_mqa_logits_" + dtype_tag + (is_avg ? "_avg" : "");

    // `k_scales` is an empty tensor for BF16 and FP4 (FP4 uses `k_sf` instead)
    const float* k_scales_ptr = (is_fp4 or qk_dtype == torch::kBFloat16) ? nullptr : k_scales.data_ptr<float>();

    if (is_fp4) {
        const auto* q_sf_ptr = reinterpret_cast<const uint32_t*>(q_sf->data_ptr());
        const auto* k_sf_ptr = reinterpret_cast<const uint32_t*>(k_sf->data_ptr());
        auto* ks_ptr = reinterpret_cast<int*>(cu_seq_len_k_start.data_ptr());
        auto* ke_ptr = reinterpret_cast<int*>(cu_seq_len_k_end.data_ptr());
        if (stride_k_type == "uint32_t") {
            launch_mqa_logits<uint32_t>(
                "deep_gemm/impls/fp4_mqa_logits.cuh", "PPUMqaLogitsFP4", config, element_qk, element_acc, element_logits,
                element_weights, dtype_tag, num_heads, head_dim, is_compressed, scale_mode, smem_size, num_threads,
                kernel_name,
                MqaLogitsFP4Arguments<uint32_t>{q.data_ptr(), q_sf_ptr, k.data_ptr(), k_sf_ptr, weights.data_ptr(),
                                                ks_ptr, ke_ptr, logits.data_ptr(), seq_len_q, seq_len_k,
                                                static_cast<uint32_t>(aligned_seq_len_kv)});
        } else {
            launch_mqa_logits<uint64_t>(
                "deep_gemm/impls/fp4_mqa_logits.cuh", "PPUMqaLogitsFP4", config, element_qk, element_acc, element_logits,
                element_weights, dtype_tag, num_heads, head_dim, is_compressed, scale_mode, smem_size, num_threads,
                kernel_name,
                MqaLogitsFP4Arguments<uint64_t>{q.data_ptr(), q_sf_ptr, k.data_ptr(), k_sf_ptr, weights.data_ptr(),
                                                ks_ptr, ke_ptr, logits.data_ptr(), seq_len_q, seq_len_k,
                                                static_cast<uint64_t>(aligned_seq_len_kv)});
        }
    } else if (stride_k_type == "uint32_t") {
        launch_mqa_logits<uint32_t>(
            "deep_gemm/impls/ppu_mqa_logits.cuh", "PPUMqaLogits", config, element_qk, element_acc, element_logits, element_weights,
            dtype_tag, num_heads, head_dim, is_compressed, scale_mode, smem_size, num_threads, kernel_name,
            MqaLogitsArguments<uint32_t>{q.data_ptr(), k.data_ptr(), k_scales_ptr, weights_ptr,
                                         reinterpret_cast<uint32_t*>(cu_seq_len_k_start.data_ptr()),
                                         reinterpret_cast<uint32_t*>(cu_seq_len_k_end.data_ptr()), logits.data_ptr(),
                                         static_cast<uint32_t>(seq_len_q), static_cast<uint32_t>(seq_len_k),
                                         static_cast<uint32_t>(aligned_seq_len_kv)});
    } else {
        launch_mqa_logits<uint64_t>(
            "deep_gemm/impls/ppu_mqa_logits.cuh", "PPUMqaLogits", config, element_qk, element_acc, element_logits, element_weights,
            dtype_tag, num_heads, head_dim, is_compressed, scale_mode, smem_size, num_threads, kernel_name,
            MqaLogitsArguments<uint64_t>{q.data_ptr(), k.data_ptr(), k_scales_ptr, weights_ptr,
                                         reinterpret_cast<uint32_t*>(cu_seq_len_k_start.data_ptr()),
                                         reinterpret_cast<uint32_t*>(cu_seq_len_k_end.data_ptr()), logits.data_ptr(),
                                         static_cast<uint32_t>(seq_len_q), static_cast<uint32_t>(seq_len_k),
                                         static_cast<uint64_t>(aligned_seq_len_kv)});
    }

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
