#pragma once
#include <cstdint>
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
  uint8_t* fast_lo;
  uint8_t* fast_hi;
  uint8_t* old_lo;
  uint8_t* old_hi;
  int max_extra_rows;
};
struct RoutedGemmArguments {
  const void* a;
  const uint8_t* b;
  const uint8_t* scale;
  void* d;
  const float* a_scale;
  int num_token;
  const int32_t* sorted_token_ids;
  const uint8_t* fast_metadata;
  const uint8_t* old_metadata;
};
} // namespace deep_gemm::w4fa8
