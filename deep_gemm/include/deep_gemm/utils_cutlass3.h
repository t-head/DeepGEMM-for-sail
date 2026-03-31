#ifndef DEEP_GEMM_UTILS_CUTLASS3_H
#define DEEP_GEMM_UTILS_CUTLASS3_H

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"

struct KernelAiuMultistageOnN {
  constexpr static int N_EXPAND = 4;
};

struct KernelAiuMultistageOverlapPrologue {};
struct KernelAiuMultistageOverlapMainloop {};

template <typename GemmKernel>
inline int compute_occupancy_for_kernel() {
  cudaError_t result;
  int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    result = cudaFuncSetAttribute(cutlass::device_kernel<GemmKernel>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  smem_size);
    if (cudaSuccess != result) {
      result = cudaGetLastError(); // to clear the error bit
      std::cout << "  cudaFuncSetAttribute() returned error: " << cudaGetErrorString(result) << std::endl;
    }
  }

  int max_active_blocks = -1;
  result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks, cutlass::device_kernel<GemmKernel>, GemmKernel::MaxThreadsPerBlock, smem_size);

  if (cudaSuccess != result) {
    result = cudaGetLastError(); // to clear the error bit
    std::cout << "  cudaOccupancyMaxActiveBlocksPerMultiprocessor() returned error: " << cudaGetErrorString(result) << std::endl;
  }

  return max_active_blocks;
}

template <typename Element> class ToCutlassType {
public:
  using Element_if_bf16 = typename cutlass::platform::conditional<cutlass::platform::is_same<Element, __nv_bfloat16>::value,
                                                    cutlass::bfloat16_t, Element>::type;
  using Element_if_fp16 = typename cutlass::platform::conditional<cutlass::platform::is_same<Element, half>::value,
                                                    cutlass::half_t, Element_if_bf16>::type;
  using Element_if_fp8_e4m3 = typename cutlass::platform::conditional<cutlass::platform::is_same<Element, __nv_fp8_e4m3>::value,
                                                    cutlass::float_e4m3_t, Element_if_fp16>::type;
  using type = Element_if_fp8_e4m3;
};
#endif // DEEP_GEMM_UTILS_CUTLASS3_H
