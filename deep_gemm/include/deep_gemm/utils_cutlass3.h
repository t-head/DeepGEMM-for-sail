#ifndef DEEP_GEMM_UTILS_CUTLASS3_H
#define DEEP_GEMM_UTILS_CUTLASS3_H

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#if !defined(__CUDACC_RTC__)
#include "cuda_ad.h"
#endif

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

template <typename GemmKernel>
inline void launch_kernel(typename GemmKernel::Params& params, cudaStream_t stream, int max_blocks_per_cu = 1) {
    dim3 const block = GemmKernel::get_block_shape();
    dim3 const grid = GemmKernel::get_grid_shape(params);
    const int smem_size = GemmKernel::SharedStorageSize;
#if !defined(__CUDACC_RTC__)
    if (GemmKernel::TileScheduler::EnableHWDispatchStrategy) {
      // set dispatch strategy
      auto kernel = cutlass::device_kernel<GemmKernel>;
      const void *gemmfunc = reinterpret_cast<const void*>(kernel);
      CUfunction func = static_cast<CUfunction>(NULL);
      cudaGetFuncBySymbol(reinterpret_cast<cudaFunction_t*>(&func), gemmfunc);
      void* kernel_args[] = {&params};
      CUlaunchAttributeAD LaunchAttr = {CUAD_LAUNCH_ATTRIBUTE_SCHED_PREFERENCE};
      CUlaunchConfigAD LaunchCfg = {grid.x, grid.y, grid.z, block.x, block.y, block.z, smem_size, stream, &LaunchAttr, 1};
      max_blocks_per_cu = min(max_blocks_per_cu, 8);
      LaunchAttr.value.schedPreference.blocksPerMultiprocessor = max_blocks_per_cu;//schedule.bits.tb_per_cu;
      LaunchAttr.value.schedPreference.gridStepX = 1;
      LaunchAttr.value.schedPreference.gridStepY = 1;
      LaunchAttr.value.schedPreference.flags = CUAD_SCHED_PREFER_UNIFORM; // HGAD_SCHED_PREFER_UNIFORM: 8 (auto-startce)
      CHECK_DRIVER_API(cuLaunchKernelExAD(&LaunchCfg, func, kernel_args, nullptr));
    } else {
      cutlass::device_kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params);
    }
#else
      cutlass::device_kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params);
#endif
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
