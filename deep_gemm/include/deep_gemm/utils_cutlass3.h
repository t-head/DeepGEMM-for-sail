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
  hggcError_t result;
  int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    result = hggcFuncSetAttribute(cutlass::device_kernel<GemmKernel>,
                                  hggcFuncAttributeMaxDynamicSharedMemorySize,
                                  smem_size);
    if (hggcSuccess != result) {
      result = hggcGetLastError(); // to clear the error bit
      std::cout << "  hggcFuncSetAttribute() returned error: " << hggcGetErrorString(result) << std::endl;
    }
  }

  int max_active_blocks = -1;
  result = hggcOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks, cutlass::device_kernel<GemmKernel>, GemmKernel::MaxThreadsPerBlock, smem_size);

  if (hggcSuccess != result) {
    result = hggcGetLastError(); // to clear the error bit
    std::cout << "  hggcOccupancyMaxActiveBlocksPerMultiprocessor() returned error: " << hggcGetErrorString(result) << std::endl;
  }

  return max_active_blocks;
}

template <typename Element> class ToCutlassType {
public:
  using Element_if_bf16 = typename cutlass::platform::conditional<cutlass::platform::is_same<Element, __ppu_bfloat16>::value,
                                                    cutlass::bfloat16_t, Element>::type;
  using Element_if_fp16 = typename cutlass::platform::conditional<cutlass::platform::is_same<Element, half>::value,
                                                    cutlass::half_t, Element_if_bf16>::type;
  using Element_if_fp8_e4m3 = typename cutlass::platform::conditional<cutlass::platform::is_same<Element, __hg_fp8_e4m3>::value,
                                                    cutlass::float_e4m3_t, Element_if_fp16>::type;
  using type = Element_if_fp8_e4m3;
};
#endif // DEEP_GEMM_UTILS_CUTLASS3_H
