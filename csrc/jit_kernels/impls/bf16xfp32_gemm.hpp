#pragma once

// =============================================================================
// BF16 x FP32 DenseGemm (cute-free dual-B kernel + PPU1.0 fallback).
//
// Computes  D = A @ (W_high + W_low * scale)^T  with FP32 accumulation, where
// the FP32 weight is supplied as two BF16 matrices:
//     W_high = W_fp32.to(torch.bfloat16)
//     W_low  = ((W_fp32 - W_high.float()) / scale).to(torch.bfloat16)
// (scale is typically 1/256). Inputs / layouts:
//     A (lhs)     : BF16, row-major [m, k]
//     W_high / W_low : BF16, row-major [n, k]  (NT layout: rhs.T is col-major)
//     D (out)     : BF16 or FP32, row-major [m, n]
//
// PPU1.5 (890P): fused cute-free kernel -- a single interleaved K pipeline
//     per output tile, each stage carrying A + W_high + W_low together (A is
//     streamed once), two back-to-back MMAs per k_step into two FP32
//     accumulators, combined in registers and stored once. m == 1 / 2 dispatch
//     to the dedicated GEMV kernel (two passes composed in FP32).
// PPU1.0 (810E): fallback composition of two BF16 GEMMs through an FP32
//     intermediate (still BF16 tensor-core rate, higher precision than
//     BF16 x BF16 because the low part recovers the residual bits).
// =============================================================================

#include <torch/python.h>
#include <cstdint>
#include <type_traits>
#include <hggc_fp8.h>
#include "cute/tensor.hpp"
#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"
#include "../../utils/layout_type_name.hpp"
#include "cute/arch/mma.hpp"
#include <deep_gemm/common/profiling_interface.cuh>
#include "../heuristics/common_bf16.hpp"
#include <deep_gemm/scheduler/densegemm_scheduler_cutlass3.cuh>
#include "bf16_gemm.hpp"

using namespace deep_gemm_bf16_common;
namespace deep_gemm {

// JIT runtime for the fused dual-B cute-free kernel. Mirrors
// DenseBF16GemmCuteFreeRuntimeT; the host-side GemmKernelParams must stay
// byte-identical to kernel::GemmBf16xFp32Kernel::Params (the kernel receives
// it by value).
template <typename ElementOutput>
class DenseBf16xFp32GemmCuteFreeRuntimeT final : public LaunchRuntime<DenseBf16xFp32GemmCuteFreeRuntimeT<ElementOutput>> {
public:
    struct GemmKernelParams {
        cutlass::bfloat16_t const* ptr_A;
        cutlass::bfloat16_t const* ptr_B_high;
        cutlass::bfloat16_t const* ptr_B_low;
        ElementOutput* ptr_D;
        float low_scale;
        int M, N, K;
        int lda, ldb, ldd;
        DenseGemmTileSchedulerArguments scheduler;
    };

    struct LaunchInfo {
        int block_m, block_n, block_k, warp_m, warp_n;
        int warp_k;
        int num_stages;
        std::string gemm_type, kernel_name;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        GemmKernelParams kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        const auto& info = args.launch_info;
        const int shape_n = args.kernel_params.N;
        const bool is_aligned_n = (shape_n % info.block_n == 0);

        return fmt::format(
            R"(
#define BF16_HGRTC
#include <deep_gemm/impls/bf16xfp32_gemm_cute_free.cuh>

namespace deep_gemm {{

using GemmTypeTag = std::integral_constant<GemmType, GemmType::{0}>;

using Kernel = BF16xFp32GemmCuteFreeKernel<{1}, {2}, {3}, {4}, {5}, {6}, {7}, true, {8}, GemmTypeTag{9}>;

}} // namespace deep_gemm

extern "C"
__launch_bounds__(deep_gemm::Kernel::MaxThreadsPerBlock,
                  deep_gemm::Kernel::MinBlocksPerMultiprocessor)
__global__ void {10}(typename deep_gemm::Kernel::Params params) {{
  extern __shared__ char smem[];
  deep_gemm::Kernel kernel;
  kernel(params, smem);
}}
)",
            info.gemm_type,
            info.block_m, info.block_n, info.block_k,
            info.warp_m, info.warp_n, info.warp_k, info.num_stages,
            is_aligned_n ? "true" : "false",
            std::is_same_v<ElementOutput, float> ? ", float" : "",
            info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

// Composition fallback: out = x@w_high^T + scale * x@w_low^T via two plain BF16
// GEMMs through an FP32 intermediate (BF16 tensor-core rate, FP32 accumulation;
// the low part recovers the residual bits). Used for PPU1.0 and for tile
// configs whose dual-B SMEM footprint does not fit the per-CU budget.
static void bf16xfp32_gemm_composed(const torch::Tensor& lhs, const torch::Tensor& w_high,
                                    const torch::Tensor& w_low, const torch::Tensor& out,
                                    const double& scale, const int& m, const int& n, const int& k,
                                    std::optional<ConfigTuple> configs = std::nullopt) {
    torch::Tensor out_f32 = (out.scalar_type() == torch::kFloat)
        ? out
        : torch::empty({m, n}, out.options().dtype(torch::kFloat));
    bf16_gemm(lhs, w_high, out_f32, m, n, k, configs);
    torch::Tensor low_f32 = torch::empty({m, n}, out_f32.options());
    bf16_gemm(lhs, w_low, low_f32, m, n, k, configs);
    out_f32.add_(low_f32, scale);
    if (out.scalar_type() != torch::kFloat)
        out.copy_(out_f32);
}

// Fused dual-B launcher (PPU1.5 cute-free path). Tile/warp config selection is
// shared with the plain BF16 dense path; the dual-B kernel carries one extra B
// operand per pipeline stage (SMEM = STAGES * (BLOCK_M + 2*BLOCK_N) * BLOCK_K),
// so the stage count is pruned against the per-CU SMEM budget and configs
// that still do not fit fall back to the composed two-GEMM path.
template <typename ElementOutput>
static void bf16xfp32_gemm_cute_free_impl(const torch::Tensor& lhs, const torch::Tensor& w_high,
                                          const torch::Tensor& w_low, const torch::Tensor& out,
                                          const float& scale, const int& m, const int& n, const int& k,
                                          std::optional<ConfigTuple> configs = std::nullopt) {
    int num_sms = get_num_sms();
    static constexpr GemmType kGemmType = GemmType::Normal;

    using Config = std::tuple<int, int, int, int, int, int, int, int, std::tuple<int, int, int>>;

    Config cfg;
    if (configs.has_value()) {
        // Explicit config path: unpack the 8-element public tuple; force warp_k = block_k
        // (same contract as the plain BF16 dense path).
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        cfg = std::make_tuple(ns, bm, bn, bk, wm, wn, /*warp_k=*/bk, nst,
            deep_gemm_bf16_common::get_smem_config(nst, k, bm, bn, bk, 2));
    } else {
        bool from_runtime_lut = false;
        auto best = deep_gemm_bf16_common::get_best_configs(
            m, n, k, 1, num_sms, kGemmType, 256, &from_runtime_lut);
        auto [ns, bm, bn, bk, wm, wn, nst, sc] = best;
        int warp_k = bk;
        if (!from_runtime_lut && is_ppu1v5_device() &&
            deep_gemm_adaptive::bf16_adaptive_enabled(m, n, k)) {
            warp_k = std::get<6>(deep_gemm_adaptive::get_adaptive_configs(m, n, k, num_sms));
        }
        cfg = std::make_tuple(ns, bm, bn, bk, wm, wn, warp_k, nst, sc);
    }
    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages, smem_config] = cfg;
    (void)smem_config;  // single-B footprint; the dual-B kernel sizes its own SMEM

    // Dual-B shared memory: STAGES * (BLOCK_M + 2*BLOCK_N) * BLOCK_K elements,
    // rounded up to the SharedStorage 128-byte alignment. The tile config came
    // from single-B heuristics, so prune stages until the dual-B footprint fits
    // the 256 KB per-CU budget (PPU 890P).
    constexpr int kSmemPerCU = 256 * 1024;
    constexpr int kElemBytes = static_cast<int>(sizeof(cutlass::bfloat16_t));
    auto dual_smem_bytes = [&](int stages) {
        int bytes = stages * (block_m + 2 * block_n) * block_k * kElemBytes;
        return (bytes + 127) & ~127;
    };
    while (num_stages > 2 && dual_smem_bytes(num_stages) > kSmemPerCU)
        --num_stages;
    const int SMSIZE = dual_smem_bytes(num_stages);
    if (SMSIZE > kSmemPerCU) {
        // Even the minimum pipeline (2 stages) does not fit for these huge
        // tiles: compose two plain BF16 GEMMs instead (they size their own SMEM).
        bf16xfp32_gemm_composed(lhs, w_high, w_low, out, scale, m, n, k, configs);
        return;
    }

    cutlass::bfloat16_t* input_a = reinterpret_cast<cutlass::bfloat16_t*>(lhs.data_ptr<at::BFloat16>());
    cutlass::bfloat16_t* input_b_high = reinterpret_cast<cutlass::bfloat16_t*>(w_high.data_ptr<at::BFloat16>());
    cutlass::bfloat16_t* input_b_low = reinterpret_cast<cutlass::bfloat16_t*>(w_low.data_ptr<at::BFloat16>());
    auto* output = static_cast<ElementOutput*>(out.data_ptr());

    using Runtime = DenseBf16xFp32GemmCuteFreeRuntimeT<ElementOutput>;

    typename Runtime::GemmKernelParams hparams;
    hparams.ptr_A = input_a;
    hparams.ptr_B_high = input_b_high;
    hparams.ptr_B_low = input_b_low;
    hparams.ptr_D = output;
    hparams.low_scale = scale;
    hparams.M = m;
    hparams.N = n;
    hparams.K = k;
    hparams.lda = k;   // A is row-major MxK
    hparams.ldb = k;   // B is col-major KxN, stored as row-major NxK
    hparams.ldd = n;   // D is row-major MxN
    hparams.scheduler = DenseGemmTileSchedulerArguments{(uint32_t)m, (uint32_t)n, (uint32_t)k, nullptr};

    const int warps_k = block_k / warp_k;
    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * warps_k * 32;

    // SMEM-based occupancy heuristic on the dual-B footprint.
    int blocks_per_cu = kSmemPerCU / SMSIZE;
    if (blocks_per_cu < 1) blocks_per_cu = 1;

    dim3 grid;
    grid.x = blocks_per_cu * num_sms_new;

    // Output dtype is part of the JIT cache key (separate kernel name).
    std::string kernel_name = "bf16xfp32_gemm_cute_free";
    if (std::is_same_v<ElementOutput, float>) kernel_name += "_fp32";

    auto args = typename Runtime::Args{
        .launch_info = {block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages,
                        "Normal", kernel_name},
        .launch_args = {grid, block, SMSIZE},
        .kernel_params = hparams,
    };

    const auto& code = Runtime::generate(args);
    auto runtime = compiler->build(kernel_name, code);
    if (is_compile_only())
        return;
    auto kernel = runtime->kernel;

    const auto& stream = current_stream();
    auto config = construct_launch_config(kernel, stream, SMSIZE, grid, block);

    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_params(kGemmType, false, std::string("bf16xfp32"), 1, m, n, k, 0, nullptr, stream);
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);

    DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, hparams));

    ProfilingInterface::Instance().instrument(false, dg_prof_params);

    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        int numRegs = 0, localSize = 0;
        hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
        hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

        printf("[DenseGemm_BF16xFP32_CuteFree:]\n");
        printf("problem:[%d, %d, %d], scale:%g\n", m, n, k, scale);
        printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu, grid.x);
        printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], num_stages:%d\n",
               block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages);
        printf("SMSIZE:%d, vreg:%d, stack:%d\n", SMSIZE, int(numRegs), int(localSize));
    }
}

// Public launcher: dispatches on the device generation.
//   PPU1.5 : m == 1 / 2 -> two GEMV passes composed in FP32 (avoids the tile
//            path's BM=16 padding waste); otherwise the fused dual-B cute-free
//            kernel (out is BF16 or FP32).
//   PPU1.0 : composed two-GEMM fallback (out = x@w_high^T, then out +=
//            scale * x@w_low^T).
// `configs` (optional) follows the 8-element public ConfigTuple contract of
// the plain BF16 dense path.
static void bf16xfp32_gemm(const torch::Tensor& lhs, const torch::Tensor& w_high, const torch::Tensor& w_low,
                          const torch::Tensor& out, const double& scale,
                          const int& m, const int& n, const int& k,
                          std::optional<ConfigTuple> configs = std::nullopt) {
    if (is_ppu1v5_device()) {
        // Small-m fast path: the dedicated GEMV kernel (mirrors the plain BF16
        // dense path's dispatch) runs two passes composed in FP32 instead of
        // the tile path, which pads m to 16 lanes. Shapes the GEMV rejects
        // fall through to the fused dual-B tile kernel; both calls share the
        // same (m, n, k), so feasibility is consistent across the pair.
        if ((m == 1 || m == 2) && !configs.has_value()) {
            torch::Tensor out_f32 = (out.scalar_type() == torch::kFloat)
                ? out
                : torch::empty({m, n}, out.options().dtype(torch::kFloat));
            torch::Tensor low_f32 = torch::empty({m, n}, out_f32.options());
            if (gemv_dense_bf16(lhs, w_high, out_f32, m, n, k) &&
                gemv_dense_bf16(lhs, w_low, low_f32, m, n, k)) {
                out_f32.add_(low_f32, scale);
                if (out.scalar_type() != torch::kFloat)
                    out.copy_(out_f32);
                return;
            }
        }
        if (out.scalar_type() == torch::kFloat)
            bf16xfp32_gemm_cute_free_impl<float>(lhs, w_high, w_low, out, static_cast<float>(scale),
                                                 m, n, k, configs);
        else
            bf16xfp32_gemm_cute_free_impl<cutlass::bfloat16_t>(lhs, w_high, w_low, out, static_cast<float>(scale),
                                                              m, n, k, configs);
        return;
    }

    // --- PPU1.0 fallback: compose two BF16 GEMMs (both accumulate in FP32). ---
    bf16xfp32_gemm_composed(lhs, w_high, w_low, out, scale, m, n, k, configs);
}

} // namespace deep_gemm
