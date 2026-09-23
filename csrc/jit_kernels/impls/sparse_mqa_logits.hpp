#pragma once

#include <cctype>
#include <cstdint>
#include <limits>

#include <torch/extension.h>

#include <deep_gemm/common/profiling_interface.cuh>
#include <deep_gemm/impls/sparse_mqa_logits_layout.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/common_mqa.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"

namespace deep_gemm {

// Host launchers for sparse MQA logits and metadata generation.
// Layout structs/constants live in the shared device+host header
// `deep_gemm/impls/sparse_mqa_logits_layout.cuh`.
//
// MXFP4 only.
namespace sparse_mqa_logits {

// Schedule table columns: one slot per resident CTA of the sparse main kernel.
inline int get_num_sparse_slots() {
    return get_num_sms() * kNumTbPerCu;
}

static uint32_t get_sparse_split_kv(const at::ScalarType& qk_dtype) {
    // MXFP4 only.
    DG_HOST_ASSERT(qk_dtype == torch::kInt8);
    return kSplitKV;
}

// Worst-case metadata buffer size. Each Q block (kBlockQ = 2 tokens) merges at most
// kBlockQ * num_max_sparse_blocks KV blocks, each split holds split_kv / sparse_block_kv of them;
// the schedule table is padded to a whole number of waves. Paged bounds splits per Q token
// instead (one token can own up to num_max_sparse_blocks blocks → cap/split splits).
static int64_t get_num_sparse_metadata_bytes(const int num_q_tokens, const int num_max_sparse_blocks,
                                             const int sparse_block_kv, const int num_slots,
                                             const bool is_paged = false) {
    DG_HOST_ASSERT(num_max_sparse_blocks > 0 and num_max_sparse_blocks % 4 == 0 and num_max_sparse_blocks <= 4096);
    DG_HOST_ASSERT(sparse_block_kv == 8 or sparse_block_kv == 16);
    const int64_t num_kv_blocks_per_split = kSplitKV / sparse_block_kv;
    const int64_t num_kv_split_bytes = sizeof(KVSplitHeader) + num_kv_blocks_per_split * sizeof(KVBlockInfo);
    const int64_t num_max_kv_splits = is_paged ?
        static_cast<int64_t>(num_q_tokens) * ceil_div<int64_t>(num_max_sparse_blocks, num_kv_blocks_per_split) :
        ceil_div<int64_t>(num_q_tokens, kBlockQ) *
            ceil_div<int64_t>(kBlockQ * static_cast<int64_t>(num_max_sparse_blocks), num_kv_blocks_per_split);
    DG_HOST_ASSERT(num_max_kv_splits <= std::numeric_limits<uint32_t>::max());
    const int64_t num_max_schedule_entries = align<int64_t>(num_max_kv_splits, num_slots);
    return static_cast<int64_t>(sizeof(MetadataHeader)) + num_max_kv_splits * num_kv_split_bytes +
           num_max_schedule_entries * sizeof(ScheduleEntry);
}

// ---------------------------------------------------------------- metadata kernel

class SparseMqaLogitsMetadataRuntime final : public LaunchRuntime<SparseMqaLogitsMetadataRuntime> {
public:
    struct KernelArguments {
        uint32_t num_q_tokens, num_kv_tokens;
        const uint32_t* cu_seq_len_k_start;
        const uint32_t* cu_seq_len_k_end;
        const uint32_t* context_lens;
        const uint32_t* block_table;
        uint32_t block_table_stride;
        const uint32_t* indices;
        const uint32_t* sparse_kv_block_indices;
        uint32_t num_max_sparse_blocks;
        uint8_t* metadata;
        uint8_t* workspace;
    };

    struct LaunchInfo {
        int num_max_blocks_cap, num_slots, sparse_block_kv, num_threads, smem_size;
        bool use_unaligned_ks, is_paged;
        uint32_t page_kv;
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
#include <deep_gemm/scheduler/sparse_mqa_logits_metadata.cuh>
namespace deep_gemm {{

constexpr uint32_t kNumThreads = {};
constexpr uint32_t kNumMaxBlocksCap = {};
constexpr uint32_t kNumSlots = {};
constexpr uint32_t kSparseBlockKV = {};
constexpr bool kUseUnalignedKs = {};
constexpr bool kIsPaged = {};
constexpr uint32_t kPageKV = {};

extern "C"
__global__ void {}(
  const uint32_t num_q_tokens, const uint32_t num_kv_tokens,
  const uint32_t* __restrict__ cu_seq_len_k_start, const uint32_t* __restrict__ cu_seq_len_k_end,
  const uint32_t* __restrict__ context_lens, const uint32_t* __restrict__ block_table,
  const uint32_t block_table_stride, const uint32_t* __restrict__ indices,
  const uint32_t* __restrict__ sparse_kv_block_indices, const uint32_t num_max_sparse_blocks,
  uint8_t* __restrict__ metadata, uint8_t* __restrict__ workspace) {{
  sparse_mqa_logits::sparse_mqa_logits_metadata_device<
      kNumThreads, kNumMaxBlocksCap, kNumSlots, kSparseBlockKV, kUseUnalignedKs, kIsPaged, kPageKV>(
      num_q_tokens, num_kv_tokens, cu_seq_len_k_start, cu_seq_len_k_end, context_lens, block_table,
      block_table_stride, indices, sparse_kv_block_indices, num_max_sparse_blocks, metadata, workspace);
}}
}}
)",
            args.launch_info.num_threads, args.launch_info.num_max_blocks_cap, args.launch_info.num_slots,
            args.launch_info.sparse_block_kv, args.launch_info.use_unaligned_ks ? "true" : "false",
            args.launch_info.is_paged ? "true" : "false", args.launch_info.page_kv,
            args.launch_info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config,
            args.kernel_args.num_q_tokens, args.kernel_args.num_kv_tokens,
            args.kernel_args.cu_seq_len_k_start, args.kernel_args.cu_seq_len_k_end,
            args.kernel_args.context_lens, args.kernel_args.block_table, args.kernel_args.block_table_stride,
            args.kernel_args.indices,
            args.kernel_args.sparse_kv_block_indices, args.kernel_args.num_max_sparse_blocks,
            args.kernel_args.metadata, args.kernel_args.workspace));
    }
};

// Launches the sparse metadata kernel for one contiguous-KV call. The workspace is a process-wide
// cache, grown on demand and self-cleaning (the kernel resets the WorkspaceState counters at the
// end), so the tensor outlives the API call while the async launch is still in flight.
static torch::Tensor sparse_workspace_cache;

static torch::Tensor& get_sparse_workspace(const torch::TensorOptions& options, const int64_t num_bytes) {
    if (not sparse_workspace_cache.defined() or sparse_workspace_cache.numel() < num_bytes)
        sparse_workspace_cache = torch::zeros({num_bytes}, options);
    return sparse_workspace_cache;
}

static void launch_sparse_mqa_logits_metadata(const torch::Tensor& metadata, torch::Tensor& workspace,
                                              const uint32_t num_q_tokens, const uint32_t num_kv_tokens,
                                              const uint32_t num_max_sparse_blocks, const uint32_t sparse_block_kv,
                                              const uint32_t* cu_seq_len_k_start_ptr,
                                              const uint32_t* cu_seq_len_k_end_ptr,
                                              const uint32_t* sparse_kv_block_indices_ptr,
                                              const bool use_unaligned_ks,
                                              const bool is_paged = false, const uint32_t page_kv = 0,
                                              const uint32_t* context_lens_ptr = nullptr,
                                              const uint32_t* block_table_ptr = nullptr,
                                              const uint32_t block_table_stride = 0,
                                              const uint32_t* indices_ptr = nullptr) {
    constexpr uint32_t kNumThreads = 256;
    const int num_slots = get_num_sparse_slots();
    const uint32_t num_max_blocks_cap = get_num_sparse_blocks_bucket(num_max_sparse_blocks);
    // Dynamic smem: the packed merge output (2 × Cap words)
    const int smem_size = static_cast<int>(2 * num_max_blocks_cap * sizeof(uint32_t));
    const auto kernel_name = "sparse_mqa_logits_metadata";

    auto args = SparseMqaLogitsMetadataRuntime::Args{
        .launch_info = {static_cast<int>(num_max_blocks_cap), num_slots, static_cast<int>(sparse_block_kv),
                        static_cast<int>(kNumThreads), smem_size, use_unaligned_ks, is_paged, page_kv, kernel_name},
        .launch_args = {dim3(is_paged ? static_cast<int>(std::min(num_q_tokens, static_cast<uint32_t>(num_slots)))
                                      : ceil_div<int>(static_cast<int>(num_q_tokens), static_cast<int>(kBlockQ)),
                                1, 1),
                        dim3(static_cast<int>(kNumThreads), 1, 1), smem_size},
        .kernel_args = {num_q_tokens, num_kv_tokens,
                        cu_seq_len_k_start_ptr, cu_seq_len_k_end_ptr,
                        context_lens_ptr, block_table_ptr, block_table_stride, indices_ptr,
                        sparse_kv_block_indices_ptr,
                        num_max_sparse_blocks,
                        reinterpret_cast<uint8_t*>(metadata.data_ptr()),
                        reinterpret_cast<uint8_t*>(workspace.data_ptr())},
    };
    const auto& code = SparseMqaLogitsMetadataRuntime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code, static_cast<int>(kNumThreads), smem_size);
    SparseMqaLogitsMetadataRuntime::launch(runtime, args);
}

// ---------------------------------------------------------------- main kernel

// Host-side mirror of `cutlass::gemm::kernel::PPUSparseMqaLogitsFP4<...>::Arguments`
// (Params == Arguments): bf16 logits / bf16 weights, packed fp4 q/k, packed ue8m0 scales.
// Contiguous fills ptr_k/k_sf; paged fills kv_cache/kv_page_stride_bytes (per-token 4B SF
// trailing each 64B packed row inside the fused page).
struct SparseMqaLogitsFP4Arguments {
    const uint8_t* ptr_q;
    const uint32_t* q_sf;
    const uint8_t* ptr_k;
    const uint32_t* k_sf;
    const uint8_t* kv_cache;
    uint32_t kv_page_stride_bytes;
    const void* weights;
    const uint8_t* metadata;
    void* logits;
    uint32_t logits_stride;
};

class SparseMqaLogitsFP4Runtime final : public LaunchRuntime<SparseMqaLogitsFP4Runtime> {
public:
    struct LaunchInfo {
        std::string kernel_name;
        uint32_t num_slots, sparse_block_kv;
        bool is_paged;
        uint32_t page_kv;
        int num_threads, smem_size;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        SparseMqaLogitsFP4Arguments kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(
            R"(
#include <deep_gemm/impls/fp4_sparse_mqa_logits.cuh>
namespace deep_gemm {{

using AttnKernel = cutlass::gemm::kernel::PPUSparseMqaLogitsFP4<
  uint8_t, float, __ppu_bfloat16, __ppu_bfloat16,
  64, deep_gemm::sparse_mqa_logits::kSplitKV, 32, 16, 1, 3, {}, {}, {}, {}>;

// The host computes these instead of reading them off the kernel type, so pin them down here
static_assert(AttnKernel::SharedStorageSize == {}, "host/device shared memory size mismatch");
static_assert(AttnKernel::MaxThreadsPerBlock == {}, "host/device thread count mismatch");

extern "C"
__global__ void {}(
  typename AttnKernel::Params params
) {{
  extern __shared__ char smem[];
  AttnKernel op;
  op(params, smem);
}}
}}
)",
            args.launch_info.sparse_block_kv, args.launch_info.num_slots,
            args.launch_info.is_paged ? "true" : "false", args.launch_info.page_kv,
            args.launch_info.smem_size, args.launch_info.num_threads, args.launch_info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

// Shared launcher for the sparse main kernel (contiguous + paged). Contiguous passes the packed
// KV pool (`kv`/`kv_sf`) and leaves `kv_cache` empty; paged passes the fused cache and leaves
// `kv`/`kv_sf` empty.
static void launch_fp4_sparse_mqa_logits_impl(
    const torch::Tensor& q, const torch::Tensor& q_sf, const torch::Tensor& kv, const torch::Tensor& kv_sf,
    const torch::Tensor& kv_cache, const torch::Tensor& weights, const torch::Tensor& metadata,
    torch::Tensor& logits, const uint32_t logits_stride, const uint32_t sparse_block_kv,
    const uint32_t seq_len_q, const uint32_t seq_len_kv, const uint32_t num_max_sparse_blocks,
    const bool use_unaligned_ks, const bool is_paged) {
    constexpr int kNumThreads = 512;
    constexpr uint32_t kQStages = 1, kKVStages = 3;
    const uint32_t num_slots = get_num_sparse_slots();
    // Mirrors `deep_gemm_mqa_common::get_smem_config` for the frozen sparse tile
    // (block_qh=64, block_kv=128, warp_qh=32, warp_kv=16); pinned by static_assert
    const deep_gemm_mqa_common::MqaLogitsConfig config{
        .block_q = 2, .block_qh = 64, .block_kv = 128, .warp_qh = 32, .warp_kv = 16,
        .num_q_stages = kQStages, .num_kv_stages = kKVStages};
    const int smem_size = deep_gemm_mqa_common::get_smem_config(config, 64, 1, 2, true);
    const auto kernel_name = "attention_sparse_mqa_logits_fp4";
    const uint32_t page_kv = is_paged ? static_cast<uint32_t>(kv_cache.size(1)) : 0;
    const uint32_t page_stride = is_paged ? static_cast<uint32_t>(kv_cache.stride(0)) : 0;

    auto args = typename SparseMqaLogitsFP4Runtime::Args{
        .launch_info = {kernel_name, num_slots, sparse_block_kv, is_paged, page_kv, kNumThreads, smem_size},
        .launch_args = {dim3(static_cast<int>(num_slots), 1, 1), dim3(kNumThreads, 1, 1), smem_size},
        .kernel_params = {reinterpret_cast<const uint8_t*>(q.data_ptr()),
                          reinterpret_cast<const uint32_t*>(q_sf.data_ptr()),
                          is_paged ? nullptr : reinterpret_cast<const uint8_t*>(kv.data_ptr()),
                          is_paged ? nullptr : reinterpret_cast<const uint32_t*>(kv_sf.data_ptr()),
                          is_paged ? reinterpret_cast<const uint8_t*>(kv_cache.data_ptr()) : nullptr,
                          page_stride,
                          weights.data_ptr(),
                          reinterpret_cast<const uint8_t*>(metadata.data_ptr()),
                          logits.data_ptr(),
                          logits_stride},
    };
    const auto& code = SparseMqaLogitsFP4Runtime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code, kNumThreads, smem_size);

    // Reported before the launch, as the dense `launch_mqa_logits` does, so the configuration is
    // on screen even when the launch itself fails
    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        int num_regs = 0, local_size = 0;
        hgFuncGetAttribute(&num_regs, HG_FUNC_ATTRIBUTE_NUM_REGS, runtime->kernel);
        hgFuncGetAttribute(&local_size, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, runtime->kernel);
        int max_blocks_per_cu = 0;
        hgOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_cu, runtime->kernel, kNumThreads, smem_size);

        constexpr int kNumHeads = 32;      // sparse layout constant
        constexpr int kHeadDimPacked = 64; // 128 / 2 packed fp4
        printf("[sparse_mqa_logits_fp4:]\n");
        printf("kNumHeads:%d, kHeadDim:%d(packed), BLOCK_QH:%d, BLOCK_KV:%d, SPARSE_BLOCK_KV:%u\n",
               kNumHeads, kHeadDimPacked, config.block_qh, config.block_kv, sparse_block_kv);
        printf("ThreadblockShape[%d, %d], WarpShape[%d, %d], kNumQStages:%d, kNumKVStages:%d\n",
               config.block_kv, config.block_qh, config.warp_kv, config.warp_qh, kQStages, kKVStages);
        printf("num_sms:%d, max_blocks_per_cu:%d, threadblock_count:%d, num_threads:%d\n",
               get_num_sms(), max_blocks_per_cu, num_slots, kNumThreads);
        printf("smem_size:%d, vreg:%d, stack:%d\n", smem_size, num_regs, local_size);
        printf("weights_bf16:true\n");
    }
    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_sparse_mqa_logits_params(
            "fp4", static_cast<int>(seq_len_q), static_cast<int>(seq_len_kv),
            static_cast<int>(sparse_mqa_logits::kNumHeads), static_cast<int>(sparse_mqa_logits::kHeadDim),
            static_cast<int>(sparse_block_kv), static_cast<int>(num_max_sparse_blocks), use_unaligned_ks,
            is_paged);
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);
    SparseMqaLogitsFP4Runtime::launch(runtime, args);
    ProfilingInterface::Instance().instrument(false, dg_prof_params);
}

static void launch_fp4_sparse_mqa_logits(
    const torch::Tensor& q, const torch::Tensor& q_sf, const torch::Tensor& kv, const torch::Tensor& kv_sf,
    const torch::Tensor& weights, const torch::Tensor& metadata, torch::Tensor& logits,
    const uint32_t logits_stride, const uint32_t sparse_block_kv,
    const uint32_t seq_len_q, const uint32_t seq_len_kv, const uint32_t num_max_sparse_blocks,
    const bool use_unaligned_ks) {
    launch_fp4_sparse_mqa_logits_impl(q, q_sf, kv, kv_sf, /*kv_cache=*/kv, weights, metadata, logits,
                                      logits_stride, sparse_block_kv, seq_len_q, seq_len_kv,
                                      num_max_sparse_blocks, use_unaligned_ks, /*is_paged=*/false);
}

static void launch_fp4_paged_sparse_mqa_logits(
    const torch::Tensor& q, const torch::Tensor& q_sf, const torch::Tensor& kv_cache,
    const torch::Tensor& weights, const torch::Tensor& metadata, torch::Tensor& logits,
    const uint32_t logits_stride, const uint32_t sparse_block_kv,
    const uint32_t seq_len_q, const uint32_t seq_len_kv, const uint32_t num_max_sparse_blocks) {
    const auto empty = torch::Tensor{};
    launch_fp4_sparse_mqa_logits_impl(q, q_sf, empty, empty, kv_cache, weights, metadata, logits,
                                      logits_stride, sparse_block_kv, seq_len_q, seq_len_kv,
                                      num_max_sparse_blocks, /*use_unaligned_ks=*/false, /*is_paged=*/true);
}

} // namespace sparse_mqa_logits

} // namespace deep_gemm
