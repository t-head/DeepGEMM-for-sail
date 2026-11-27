#pragma once
#include <cstdint>
#include <deep_gemm/layout/sym_buffer.cuh>

namespace deep_gemm::layout {
struct RoutedMoESignals;
}

namespace deep_gemm::mega_moe {
struct FusedReduceParams {
    layout::SymBuffer<> sym_buffer;
    const float* weights = nullptr;
    const int64_t* ids = nullptr;
    void* output = nullptr;
    layout::RoutedMoESignals* signals = nullptr;
    int tokens = 0, num_experts = 0;
};
} // namespace deep_gemm::mega_moe

namespace deep_gemm::w4fa8 {
enum class FragmentPolicy { FullK, RollingK32 };
enum class ALoadPolicy { Bulk, Ring2, Ring3 };

constexpr bool uses_ring_a(int warp_m) {
  return warp_m >= 80 && warp_m <= 192 && warp_m % 16 == 0;
}

template <class Storage, int Rows> struct RowScaleStorage : Storage {
  alignas(128) float row_scales[2][Rows];
};
template <class Storage> struct RowScaleStorage<Storage, 0> : Storage {};
} // namespace deep_gemm::w4fa8

namespace deep_gemm::w4fa8_streamed {
inline constexpr int kBlockM = 256;
struct GroupedRowRangeArguments {
  uint32_t shape_m;
  int32_t* grouped_layout;
  const int32_t* row_range;
};
struct GatherParams {
  const uint8_t *a;
  const float *a_scale;
  uint8_t *grouped_a;
  float *grouped_scale;
  const int32_t *experts;
  const int32_t *tokens;
  const int32_t *aligned;
  int num_tokens;
};
struct DecodeParams {
  const uint8_t* packed;
  const uint8_t* scales;
  uint8_t* output;
  int n_begin;
  const int32_t* all_rows;
  int group_start;
};
} // namespace deep_gemm::w4fa8_streamed
