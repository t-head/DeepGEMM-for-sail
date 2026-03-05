#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"

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
