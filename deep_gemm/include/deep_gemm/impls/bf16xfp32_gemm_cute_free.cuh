#pragma once

// =============================================================================
// BF16 x FP32 instance of the cute-free GEMM kernel.
//
// Same layering as bf16_gemm_cute_free.cuh, but binds the atoms to the dual-B
// orchestrator (cute_free/kernel/gemm_bf16xfp32.cuh). The FP32 weight is
// passed as two BF16 matrices (high + low * scale); the interleaved dual-B
// pipeline carries A + both B operands per stage on the BF16 tensor cores
// with FP32 accumulation, and the epilogue writes a single combined result.
//
// IsAlignedN: host-computed (shape_n % block_n == 0), same convention as the
// single-B DenseGemm cute-free path.
// =============================================================================

#include <cstdint>
#include <cutlass/bfloat16.h>
#include <deep_gemm/cute_free/copy/aiu_g2s_atom.cuh>
#include <deep_gemm/cute_free/copy/tsm_s2r_atom.cuh>
#include <deep_gemm/cute_free/mma/mma_atom.cuh>
#include <deep_gemm/cute_free/epilogue/epilogue.cuh>
#include <deep_gemm/cute_free/kernel/gemm_bf16xfp32.cuh>

namespace deep_gemm {

template <int BLOCK_M, int BLOCK_N, int BLOCK_K,
          int WARP_M, int WARP_N, int WARP_K, int STAGES,
          bool DenseS2Opt,
          bool IsAlignedN,
          typename GemmTypeTag, typename ElementOutput = cutlass::bfloat16_t>
using BF16xFp32GemmCuteFreeKernel = kernel::GemmBf16xFp32Kernel<
    cutlass::bfloat16_t,
    copy::AiuG2SAtom<cutlass::bfloat16_t, BLOCK_M, BLOCK_K>,   // G2S A
    copy::AiuG2SAtom<cutlass::bfloat16_t, BLOCK_N, BLOCK_K>,   // G2S B
    copy::TsmS2RAtom<cutlass::bfloat16_t, BLOCK_M, BLOCK_K, /*SwapLboSbo=*/false>,  // S2R A
    copy::TsmS2RAtom<cutlass::bfloat16_t, BLOCK_N, BLOCK_K, /*SwapLboSbo=*/true>,   // S2R B
    mma::Bf16MmaAtom,                                          // MMA
    epilogue::Epilogue<ElementOutput, mma::Bf16MmaAtom::CLayout,
                             WARP_M, WARP_N, IsAlignedN>,      // Epilogue
    BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, STAGES, DenseS2Opt,
    GemmTypeTag>;

} // namespace deep_gemm
