#ifndef DEEP_GEMM_UTILS_CUTLASS3_H
#define DEEP_GEMM_UTILS_CUTLASS3_H

#include <hggc_fp8.h>
#include <cutlass/cutlass.h>

struct KernelAiuMultistageOnN {
  static constexpr int N_EXPAND = 4;

  // A8W8 operands use one byte per element.
  static constexpr size_t smem_size(size_t k, size_t block_m, size_t block_n, size_t block_k, size_t stages) {
    const size_t smem_a = block_m * k;
    const size_t smem_b = stages * block_n * block_k;
    const size_t smem_scales = (block_m + 2 * block_n) * sizeof(float);
    return smem_a + smem_b + smem_scales;
  }
};

struct KernelAiuMultistageOverlapPrologue {};
struct KernelAiuMultistageOverlapMainloop {};


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
