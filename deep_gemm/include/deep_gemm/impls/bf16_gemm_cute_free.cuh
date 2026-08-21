#pragma once

// =============================================================================
// BF16 instance of the cute-free GEMM kernel.
//
// The pipeline / orchestration lives in the data-type-generic
// kernel::GemmKernel (deep_gemm/cute_free/kernel/gemm.cuh). This
// header only assembles the bf16 copy / mma / epilogue atoms and binds them to
// that orchestrator. To support a new data type or platform, add the
// corresponding atoms and write a thin instance header like this one.
//
// Layering (cutlass3-style, all under deep_gemm/cute_free/):
//   kernel/gemm.cuh          : generic orchestrator (prologue/mainloop/epilogue)
//   copy/*, mma/*, epilogue/*: atoms (data type / platform specific)
//   arch/*                   : low-level PTX wrappers
// =============================================================================

#include <cstdint>
#include "cutlass/bfloat16.h"
#include <deep_gemm/cute_free/copy/aiu_g2s_atom.cuh>
#include <deep_gemm/cute_free/copy/tsm_s2r_atom.cuh>
#include <deep_gemm/cute_free/mma/mma_atom.cuh>
#include <deep_gemm/cute_free/epilogue/epilogue.cuh>
#include <deep_gemm/cute_free/kernel/gemm.cuh>

namespace deep_gemm {

// IsAlignedN replaces the old SHAPE_N template parameter: the DenseGemm path no
// longer passes the problem shape at compile time, so the host computes
// (shape_n % block_n == 0) and hands the result in as a bool literal -- same
// convention as DenseBF16GemmCuteFreeRuntime::generate_impl. This keeps the
// epilogue's compile-time boundary-check elimination intact.
template <int BLOCK_M, int BLOCK_N, int BLOCK_K,
          int WARP_M, int WARP_N, int WARP_K, int STAGES,
          bool DenseS2Opt,
          bool OverlapPrologue,
          bool IsAlignedN,
          typename GemmTypeTag>
using BF16GemmCuteFreeKernel = kernel::GemmKernel<
    cutlass::bfloat16_t,
    copy::AiuG2SAtom<cutlass::bfloat16_t, BLOCK_M, BLOCK_K>,   // G2S A
    copy::AiuG2SAtom<cutlass::bfloat16_t, BLOCK_N, BLOCK_K>,   // G2S B
    copy::TsmS2RAtom<cutlass::bfloat16_t, BLOCK_M, BLOCK_K, /*SwapLboSbo=*/false>,  // S2R A
    copy::TsmS2RAtom<cutlass::bfloat16_t, BLOCK_N, BLOCK_K, /*SwapLboSbo=*/true>,   // S2R B
    mma::Bf16MmaAtom,                                          // MMA
    epilogue::Epilogue<cutlass::bfloat16_t, mma::Bf16MmaAtom::CLayout,
                             WARP_M, WARP_N, IsAlignedN>,      // Epilogue
    BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, STAGES, DenseS2Opt, OverlapPrologue,
    GemmTypeTag>;

} // namespace deep_gemm
