#ifndef DEEP_GEMM_UTILS_CUTLASS3_H
#define DEEP_GEMM_UTILS_CUTLASS3_H

#include <hggc_fp8.h>
#include <cutlass/cutlass.h>

struct KernelAiuMultistageOnN {
  constexpr static int N_EXPAND = 4;
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
