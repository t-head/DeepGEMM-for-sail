#pragma once
#include <cstdint>
namespace deep_gemm::w4fa8 {
enum class NativeStrategy { Default, Compact, Routed };
template <class Storage, int Rows> struct RowScaleStorage : Storage {
  alignas(128) float row_scales[2][Rows];
};
template <class Storage> struct RowScaleStorage<Storage, 0> : Storage {};
} // namespace deep_gemm::w4fa8

namespace deep_gemm::w4fa8_streamed {
struct GroupedRowRangeArguments {
  uint32_t shape_m;
  int32_t* grouped_layout;
  const int32_t* row_range;
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

namespace deep_gemm::w4fa8 {
// Task records retain the measured alignment/layout used by routed kernels.
inline constexpr int kRoutedMetadataBytes = 1024;
struct RoutedMetadataArguments {
  const int32_t* rows;
  uint8_t* lower;
  uint8_t* upper;
};
} // namespace deep_gemm::w4fa8
