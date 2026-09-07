#pragma once
// C++ JIT for moe_align_block_size — completes the fused MoE
// zero-Python-JIT path. Mirrors deep_gemm/jit_kernels/a_fused_m_grouped_gemm.py
// tensor allocation + the fused_gemm_util.cuh moe_align_block_size_kernel_launcher
// branch (numel <= 16384 uses single warp-ordered kernel; otherwise the
// deterministic K1->K2->K3->K4 phases are built and launched one by one).

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"
#include "../heuristics/common_bf16.hpp"
#include "../heuristics/common_fp8.hpp"
#include "../heuristics/common_int8.hpp"
#include "../heuristics/common_w4a16.hpp"
#include "fused_moe_gemm.hpp"  // FusedCommonConfigTuple

namespace deep_gemm {

// =============================================================================
// K0: single warp-ordered kernel (numel <= 16384)
// =============================================================================
class MoeAlignWarpOrderedRuntime final : public LaunchRuntime<MoeAlignWarpOrderedRuntime> {
public:
    struct KernelParams {
        const int32_t* topk_ids;
        int32_t*       sorted_token_ids;
        int32_t*       m_rows;
        int32_t*       expert_ids_and_cumsum;
        int32_t*       aligned_num_m_blocks;
        int32_t*       inv_perm;
        int32_t*       m_indices;
        int            numel;
        int            s_total_ub;
    };
    struct LaunchInfo {
        int block_m, num_groups, topk, block_size;
        std::string kernel_name;
    };
    struct Args {
        LaunchInfo   launch_info;
        LaunchArgs   launch_args;
        KernelParams kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(
            R"(
#include <deep_gemm/impls/fused_gemm_util.cuh>
namespace deep_gemm {{
extern "C" __launch_bounds__({}, 1)
__global__ void {}(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ m_rows,
    int32_t* __restrict__ expert_ids_and_cumsum,
    int32_t* __restrict__ aligned_num_m_blocks,
    int32_t* __restrict__ inv_perm,
    int32_t* __restrict__ m_indices,
    int numel, int s_total_ub)
{{
    moe_align_warp_ordered_kernel_impl<{}, {}, {}, {}>(
        topk_ids, sorted_token_ids, m_rows, expert_ids_and_cumsum,
        aligned_num_m_blocks, inv_perm, m_indices, numel, s_total_ub);
}}
}}  // namespace deep_gemm
)",
            args.launch_info.block_size,
            args.launch_info.kernel_name,
            args.launch_info.block_m,
            args.launch_info.num_groups,
            args.launch_info.topk,
            args.launch_info.block_size);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config,
            args.kernel_params.topk_ids,
            args.kernel_params.sorted_token_ids,
            args.kernel_params.m_rows,
            args.kernel_params.expert_ids_and_cumsum,
            args.kernel_params.aligned_num_m_blocks,
            args.kernel_params.inv_perm,
            args.kernel_params.m_indices,
            args.kernel_params.numel,
            args.kernel_params.s_total_ub));
    }
};

// =============================================================================
// Fallback path: deterministic K1->K2->K3->K4 (numel > 16384)
// Each phase is generated and built on its own, so every HGBIN keeps a single
// entry symbol and every phase keeps its own launch config.
// =============================================================================
class MoeAlignFallbackRuntime final : public LaunchRuntime<MoeAlignFallbackRuntime> {
public:
    enum class Phase { kCount, kScan, kCumsum, kScatter };

    struct KernelParams {
        const int32_t* topk_ids;
        int32_t*       sorted_token_ids;
        int32_t*       m_rows;
        int32_t*       expert_ids_and_cumsum;
        int32_t*       aligned_num_m_blocks;
        int32_t*       inv_perm;
        int32_t*       m_indices;
        int32_t*       block_counts;
        int32_t*       local_offsets;
        int32_t*       cumsum;
        int            numel;
        int            s_total_ub;
        int            num_groups;
        int            num_blocks;
    };
    struct LaunchInfo {
        int block_m, num_groups, topk, block_size, k2_block, k3_block;
    };
    struct Args {
        Phase        phase;
        LaunchInfo   launch_info;
        LaunchArgs   launch_args;
        KernelParams kernel_params;
    };

    // Wrapper name of a phase; also the build name, so the two cannot drift apart
    static std::string symbol_name(const Phase& phase) {
        switch (phase) {
        case Phase::kCount:   return "moe_align_p1";
        case Phase::kScan:    return "moe_align_p2";
        case Phase::kCumsum:  return "moe_align_p3";
        case Phase::kScatter: return "moe_align_p4";
        }
        DG_HOST_UNREACHABLE("Unknown moe_align phase");
    }

    static std::string generate_impl(const Args& args) {
        const auto& info = args.launch_info;
        switch (args.phase) {
        case Phase::kCount:
            return fmt::format(
                R"(
#include <deep_gemm/impls/fused_gemm_util.cuh>
namespace deep_gemm {{

extern "C" __launch_bounds__({0}, 1)
__global__ void moe_align_p1(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ block_counts,
    int numel, int s_total_ub)
{{
    block_count_with_fill_impl<{0}, {1}, {2}>(
        topk_ids, sorted_token_ids, block_counts, numel, s_total_ub, blockIdx.x);
}}

}}  // namespace deep_gemm
)",
                info.block_size, info.num_groups, info.topk);

        case Phase::kScan:
            return fmt::format(
                R"(
#include <deep_gemm/impls/fused_gemm_util.cuh>
namespace deep_gemm {{

extern "C" __launch_bounds__({0}, 1)
__global__ void moe_align_p2(
    const int32_t* __restrict__ block_counts,
    int32_t* __restrict__ local_offsets,
    int32_t* __restrict__ expert_counts,
    int num_groups, int num_blocks)
{{
    local_scan_impl<{0}>(
        block_counts, local_offsets, expert_counts, num_groups, num_blocks, blockIdx.x);
}}

}}  // namespace deep_gemm
)",
                info.k2_block);

        case Phase::kCumsum:
            return fmt::format(
                R"(
#include <deep_gemm/impls/fused_gemm_util.cuh>
namespace deep_gemm {{

extern "C" __launch_bounds__({0}, 1)
__global__ void moe_align_p3(
    const int32_t* __restrict__ m_rows,
    int32_t* __restrict__ expert_ids_and_cumsum,
    int32_t* __restrict__ cumsum_out,
    int32_t* __restrict__ aligned_num_m_blocks)
{{
    cumsum_expert_ids_impl<{0}, {1}, {2}>(
        m_rows, expert_ids_and_cumsum, cumsum_out, aligned_num_m_blocks);
}}

}}  // namespace deep_gemm
)",
                info.k3_block, info.block_m, info.num_groups);

        case Phase::kScatter:
            return fmt::format(
                R"(
#include <deep_gemm/impls/fused_gemm_util.cuh>
namespace deep_gemm {{

extern "C" __launch_bounds__({0}, 1)
__global__ void moe_align_p4(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ inv_perm,
    int32_t* __restrict__ m_indices,
    const int32_t* __restrict__ cumsum,
    const int32_t* __restrict__ local_offsets,
    int numel, int num_groups)
{{
    deterministic_scatter_impl<{0}, {1}, {2}>(
        topk_ids, sorted_token_ids, inv_perm, m_indices,
        cumsum, local_offsets, numel, num_groups, blockIdx.x);
}}

}}  // namespace deep_gemm
)",
                info.block_size, info.block_m, info.topk);
        }
        DG_HOST_UNREACHABLE("Unknown moe_align phase");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        auto& p = args.kernel_params;
        switch (args.phase) {
        case Phase::kCount:
            DG_HGGC_CHECK(launch_kernel(kernel, config,
                p.topk_ids, p.sorted_token_ids, p.block_counts, p.numel, p.s_total_ub));
            break;
        case Phase::kScan:
            DG_HGGC_CHECK(launch_kernel(kernel, config,
                p.block_counts, p.local_offsets, p.m_rows, p.num_groups, p.num_blocks));
            break;
        case Phase::kCumsum:
            DG_HGGC_CHECK(launch_kernel(kernel, config,
                p.m_rows, p.expert_ids_and_cumsum, p.cumsum, p.aligned_num_m_blocks));
            break;
        case Phase::kScatter:
            DG_HGGC_CHECK(launch_kernel(kernel, config,
                p.topk_ids, p.sorted_token_ids, p.inv_perm, p.m_indices,
                p.cumsum, p.local_offsets, p.numel, p.num_groups));
            break;
        }
    }
};

// =============================================================================
// Host impl fn — mirrors Python moe_align_block_size in a_fused_m_grouped_gemm.py
// =============================================================================
// The config slot is dtype-dependent, exactly as in Python (`return_config = config if w4a16_type is
// not None else config[:7]`): the W4A16 flavours hand back the full 9-tuple their fused GEMM unpacks,
// while every other dtype drops the trailing `smem_config` and forwards only the first 7 tuning params.
using FusedConfigTuple = std::variant<FusedCommonConfigTuple, deep_gemm_w4a16_common::W4A16ConfigTuple>;

using MoeAlignReturn = std::tuple<
    FusedConfigTuple,        // config (7-tuple, or the W4A16 9-tuple)
    torch::Tensor,         // m_rows
    torch::Tensor,         // expert_ids_and_cumsum
    torch::Tensor,         // sorted_token_ids
    torch::Tensor,         // aligned_num_m_blocks
    torch::Tensor,         // inv_perm
    torch::Tensor>;        // m_indices

static MoeAlignReturn moe_align_block_size_impl(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const torch::Tensor& topk_ids,
    bool perchannel_quant = false,
    std::optional<FusedConfigTuple> config_in = std::nullopt)
{
    // Input validation (topk_ids dtype/shape) lives in the apis layer
    // (csrc/apis/gemm.hpp); this impl only orchestrates config resolution
    // + kernel dispatch and no longer re-checks its inputs.
    const auto& [num_token, k] = get_shape<2>(lhs);
    const auto& [num_groups_l, rhs_dim1, rhs_dim2] = get_shape<3>(rhs);
    int num_groups = static_cast<int>(num_groups_l);
    int n = rhs_dim1;
    // NOTES: only the weight is in scope here, not its scales, so an `int32` weight cannot be told
    // apart from `mxfp4_e8m0` / `mxfp4_bf16` and upstream just labels it `int4`. Harmless for config
    // selection -- `get_best_configs` only branches on `mxfp4_e8m0_mma` -- so the label is mirrored.
    std::optional<deep_gemm_w4a16_common::W4A16Type> w4a16_type;
    if (rhs.dtype() == torch::kInt32) {
        n = rhs_dim2 / 2;
        w4a16_type = deep_gemm_w4a16_common::W4A16Type::int4;
    } else if (lhs.dtype() == torch::kBFloat16 and rhs.dtype() == torch::kUInt8) {
        w4a16_type = deep_gemm_w4a16_common::W4A16Type::mxfp4_e8m0_mma;
    }

    int numel = static_cast<int>(topk_ids.numel());
    int topk  = static_cast<int>(topk_ids.size(1));
    // Resolve config (mirrors Python dtype/perchannel dispatch)
    FusedConfigTuple config;
    if (config_in.has_value()) {
        config = *config_in;
    } else {
        const int num_sms = get_num_sms();
        const int expected_m = ceil_div(numel, num_groups);
        auto take_first_7 = [](const auto& full) -> FusedCommonConfigTuple {
            return FusedCommonConfigTuple{std::get<0>(full), std::get<1>(full), std::get<2>(full),
                                    std::get<3>(full), std::get<4>(full), std::get<5>(full),
                                    std::get<6>(full)};
        };
        if (w4a16_type.has_value()) {
            // Forwarded whole: the W4A16 fused GEMM unpacks all 9 params, `warp_k` and `n_expand`
            // included, so there is nothing to truncate here
            const auto& c = deep_gemm_w4a16_common::get_best_configs(
                expected_m, n, static_cast<int>(k), num_groups, num_sms, GemmType::GroupedFused, *w4a16_type);
            config = deep_gemm_w4a16_common::W4A16ConfigTuple{c.num_sms, c.block_m, c.block_n, c.block_k,
                                                             c.warp_m, c.warp_n, c.warp_k, c.num_stages,
                                                             c.n_expand};
        } else if (lhs.dtype() == torch::kBFloat16) {
            config = take_first_7(deep_gemm_bf16_common::get_best_configs(
                expected_m, static_cast<int>(n), static_cast<int>(k), num_groups, num_sms));
        } else if (perchannel_quant) {
            config = take_first_7(deep_gemm_int8::get_best_configs(
                expected_m, static_cast<int>(n), static_cast<int>(k), num_groups, num_sms));
        } else if (lhs.dtype() == torch::kFloat8_e4m3fn) {
            config = take_first_7(deep_gemm_fp8_common::get_best_configs(
                expected_m, static_cast<int>(n), static_cast<int>(k), num_groups, num_sms));
        } else if (lhs.dtype() == torch::kUInt8) {
            TORCH_CHECK(false,
                        "moe_align_block_size (C++ JIT): auto-config for fp4 (uint8 lhs) is "
                        "not ported yet; pass an explicit `config` or unset USE_CPP_JIT_FOR_PYTHON");
        } else {
            TORCH_CHECK(false,
                        "moe_align_block_size (C++ JIT): unsupported lhs dtype "
                        "(supported: bf16 / int8-perchannel / fp8-e4m3fn / w4a16; fp4 pending)");
        }
    }
    const int block_m = std::visit([](const auto& c) { return std::get<1>(c); }, config);
    DG_HOST_ASSERT(num_groups > 0);
    DG_HOST_ASSERT(block_m > 0);

    // Sizes (mirror Python)
    int max_num_m_blocks = num_groups - 1 + ceil_div(numel + 1 - num_groups, block_m);
    int block_size_pow2  = static_cast<int>(next_power_of_two(static_cast<uint32_t>(num_groups)));
    int s_total_ub       = max_num_m_blocks * block_m;
    int num_blocks_pad   = ceil_div(s_total_ub, block_size_pow2);

    // Allocate output tensors (mirror Python)
    auto opt_i32 = at::TensorOptions().dtype(at::kInt).device(topk_ids.device());
    auto expert_ids_and_cumsum = at::empty({max_num_m_blocks, 4},        opt_i32);
    auto sorted_token_ids      = at::empty({max_num_m_blocks, block_m},  opt_i32);
    auto aligned_num_m_blocks  = at::empty({1},                          opt_i32);
    auto inv_perm              = at::empty({numel},                      opt_i32);
    auto m_rows                = at::empty({num_groups},                 opt_i32);
    auto m_indices             = at::empty({numel},                      opt_i32);

    // Intermediate buffer for fallback path (K1..K4). Alloc always: matches Python.
    auto intermediate_buffer = at::empty({2 * num_blocks_pad + 2, num_groups}, opt_i32);

    // Branch: warp-ordered (numel <= 16384) OR 4-kernel fallback
    constexpr int WARP_ORDERED_MAX_NUMEL = 16384;
    if (numel <= WARP_ORDERED_MAX_NUMEL) {
        // K0 path
        constexpr int K0_BLOCK_SIZE = 1024;
        int smem = (num_groups * (K0_BLOCK_SIZE / 32) + 2 * (num_groups + 1)) * static_cast<int>(sizeof(int32_t));

        const std::string kernel_name = "moe_align_warp_ordered";

        MoeAlignWarpOrderedRuntime::Args args{
            .launch_info = {block_m, num_groups, topk, K0_BLOCK_SIZE, kernel_name},
            .launch_args = {dim3(1), dim3(K0_BLOCK_SIZE), smem},
            .kernel_params = {
                .topk_ids              = topk_ids.data_ptr<int32_t>(),
                .sorted_token_ids      = sorted_token_ids.data_ptr<int32_t>(),
                .m_rows                = m_rows.data_ptr<int32_t>(),
                .expert_ids_and_cumsum = expert_ids_and_cumsum.data_ptr<int32_t>(),
                .aligned_num_m_blocks  = aligned_num_m_blocks.data_ptr<int32_t>(),
                .inv_perm              = inv_perm.data_ptr<int32_t>(),
                .m_indices             = m_indices.data_ptr<int32_t>(),
                .numel                 = numel,
                .s_total_ub            = s_total_ub,
            },
        };

        const auto& code    = MoeAlignWarpOrderedRuntime::generate(args);
        const auto& runtime = compiler->build(kernel_name, code, K0_BLOCK_SIZE, smem);
        MoeAlignWarpOrderedRuntime::launch(runtime, args);
    } else {
        // Per-phase block widths, matching the reference launcher: BLOCK_SIZE is a template
        // constant driving index math, CUB width and static SMEM, so each phase keeps its own.
        const int bs       = block_size_pow2;  // K1, K4
        const int k2_block = 256;              // K2 scan
        const int k3_block = static_cast<int>(next_power_of_two(static_cast<uint32_t>(num_groups + 1)));
        TORCH_CHECK(bs <= 1024 and k3_block <= 1024,
                    "moe_align_block_size (C++ JIT): num_groups=", num_groups,
                    " needs blocks of ", bs, "/", k3_block, " threads, over the 1024 limit");

        int num_blocks = ceil_div(numel, bs);

        // Split intermediate buffer: [block_counts | local_offsets | cumsum]
        int32_t* base          = intermediate_buffer.data_ptr<int32_t>();
        int32_t* block_counts  = base;
        int32_t* local_offsets = base + num_blocks_pad * num_groups;
        int32_t* cumsum_gpu    = base + num_blocks_pad * num_groups * 2;

        const int smem_p1 = num_groups * static_cast<int>(sizeof(int32_t));
        const int smem_p2 = num_blocks * static_cast<int>(sizeof(int32_t));
        const int smem_p3 = 2 * (num_groups + 1) * static_cast<int>(sizeof(int32_t));
        const int smem_p4 = bs * static_cast<int>(sizeof(int32_t));

        const MoeAlignFallbackRuntime::LaunchInfo info{block_m, num_groups, topk, bs, k2_block, k3_block};
        const MoeAlignFallbackRuntime::KernelParams params{
            .topk_ids              = topk_ids.data_ptr<int32_t>(),
            .sorted_token_ids      = sorted_token_ids.data_ptr<int32_t>(),
            .m_rows                = m_rows.data_ptr<int32_t>(),
            .expert_ids_and_cumsum = expert_ids_and_cumsum.data_ptr<int32_t>(),
            .aligned_num_m_blocks  = aligned_num_m_blocks.data_ptr<int32_t>(),
            .inv_perm              = inv_perm.data_ptr<int32_t>(),
            .m_indices             = m_indices.data_ptr<int32_t>(),
            .block_counts          = block_counts,
            .local_offsets         = local_offsets,
            .cumsum                = cumsum_gpu,
            .numel                 = numel,
            .s_total_ub            = s_total_ub,
            .num_groups            = num_groups,
            .num_blocks            = num_blocks,
        };

        using Phase = MoeAlignFallbackRuntime::Phase;
        const MoeAlignFallbackRuntime::Args phases[] = {
            {Phase::kCount,   info, {dim3(num_blocks_pad), dim3(bs),       smem_p1}, params},
            {Phase::kScan,    info, {dim3(num_groups),     dim3(k2_block), smem_p2}, params},
            {Phase::kCumsum,  info, {dim3(1),              dim3(k3_block), smem_p3}, params},
            {Phase::kScatter, info, {dim3(num_blocks),     dim3(bs),       smem_p4}, params},
        };

        // One build per phase, so each HGBIN keeps a single entry symbol. Launches stay
        // ordered K1->K2->K3->K4 because they all go to the same stream.
        for (const auto& phase_args : phases) {
            const auto& code    = MoeAlignFallbackRuntime::generate(phase_args);
            const auto& runtime = compiler->build(MoeAlignFallbackRuntime::symbol_name(phase_args.phase), code,
                                                 static_cast<int>(phase_args.launch_args.block_dim.x),
                                                 phase_args.launch_args.smem_size);
            MoeAlignFallbackRuntime::launch(runtime, phase_args);
        }
    }

    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        printf("[C++ JIT moe_align_block_size] BLOCK_M=%d, num_groups=%d, topk=%d, numel=%d, path=%s\n",
               block_m, num_groups, topk, numel, (numel <= WARP_ORDERED_MAX_NUMEL) ? "warp_ordered" : "4kernel");
    }

    return std::make_tuple(
        config, m_rows, expert_ids_and_cumsum, sorted_token_ids,
        aligned_num_m_blocks, inv_perm, m_indices);
}

} // namespace deep_gemm
