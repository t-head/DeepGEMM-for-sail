#pragma once

#include <cstdint>

namespace deep_gemm {

/// Compute PredicatedTileIteratorParams values without templates
///
/// This function computes all 8 int64_t values that define the PredicatedTileIteratorParams
/// for an epilogue tile iterator. The values are computed based on the tile dimensions
/// and element properties.
///
/// Parameters (runtime):
///   block_m: Threadblock shape dimension M
///   block_n: Threadblock shape dimension N
///   block_k: Threadblock shape dimension K
///   warp_m: Warp shape dimension M
///   warp_n: Warp shape dimension N
///   warp_k: Warp shape dimension K
///   elements_per_access: Number of elements accessed per memory operation
///   element_size_bits: Size of each element in bits (16 for bfloat16, 32 for float, 8 for int8)
///   shape_n: Number of columns in the output matrix (used for stride calculation)
///
/// Output parameters (all 8 int64_t values):
///   out_stride:          bytes between rows
///   out_increment_row:   bytes to advance when moving between rows
///   out_increment_group: bytes to advance when moving to the next group
///   out_increment_cluster: bytes to advance when moving to the next cluster
///   out_advance_row:     bytes to add to move to the next 'row' position
///   out_advance_group:   bytes to add to move to the next 'group' position
///   out_advance_cluster: bytes to add to move to the next 'cluster' position
///   out_advance_tile:    bytes to add to move to the next 'tile'
CUTLASS_HOST_DEVICE
void compute_predicated_tile_iterator_params(int block_m, int block_n, int block_k, int warp_m, int warp_n, int warp_k,
                                             int elements_per_access, int element_size_bits, int shape_n,
                                             int64_t* out_stride, int64_t* out_increment_row,
                                             int64_t* out_increment_group, int64_t* out_increment_cluster,
                                             int64_t* out_advance_row, int64_t* out_advance_group,
                                             int64_t* out_advance_cluster, int64_t* out_advance_tile) {
    const int WARP_SIZE = 32;
    const int K_TENSOR_OP_ROWS = 8;

    int warp_count_m = block_m / warp_m;
    int warp_count_n = block_n / warp_n;
    int total_warps = warp_count_m * warp_count_n;

    int shape_column = block_n;       // BLOCK_N
    int shape_row = K_TENSOR_OP_ROWS;
    int shape_group = warp_count_m;   // BLOCK_M / WARP_M
    int shape_cluster = 1;
    int shape_tile = 1;
    // Count
    int count_column = 1;
    int count_row = warp_m / K_TENSOR_OP_ROWS; // WARP_M / 8
    int count_group = 1;
    int count_cluster = 1;
    int count_tile = warp_m / K_TENSOR_OP_ROWS;

    int warps_remaining_for_groups;
    if (shape_cluster > total_warps) {
        warps_remaining_for_groups = 1;
    } else {
        warps_remaining_for_groups = total_warps / shape_cluster;
    }
    int warps_remaining_for_rows;
    if (shape_group > warps_remaining_for_groups) {
        warps_remaining_for_rows = 1;
    } else {
        warps_remaining_for_rows = warps_remaining_for_groups / shape_group;
    }

    // Detail::kShapeRow
    int detail_shape_row = shape_row / warps_remaining_for_rows;

    // Detail::kShapeWidth
    int detail_shape_width = shape_column / elements_per_access;

    // Detail::kTargetMemoryAccessWidth
    int target_memory_access_width = 128 / (elements_per_access * element_size_bits / 8);

    // Detail::kTargetAccessRows
    int target_access_rows = WARP_SIZE / target_memory_access_width;

    // kAccessWidth - AIU
    int access_width;
    if (target_access_rows > detail_shape_row) {
        access_width = WARP_SIZE / detail_shape_row;
    } else {
        access_width = std::min(WARP_SIZE, 128 / (elements_per_access * element_size_bits / 8));
    }
    // kAccessRows
    int access_rows;
    if (target_access_rows > detail_shape_row) {
        access_rows = detail_shape_row;
    } else {
        access_rows = std::min(shape_row, WARP_SIZE / access_width);
    }

    // delta_row
    int delta_row = access_rows;

    // iterations_row
    int iterations_row = detail_shape_row / access_rows;

    int iterations_group;
    int delta_group;
    if (shape_group > warps_remaining_for_rows) {
        iterations_group = shape_group / warps_remaining_for_rows;
        delta_group = shape_row * count_row * shape_group / iterations_group;
    } else {
        iterations_group = 1;
        delta_group = 1;
    }
    int iterations_cluster;
    int delta_cluster;
    if (shape_cluster > total_warps) {
        iterations_cluster = shape_cluster / total_warps;
        delta_cluster = shape_row * count_row * shape_group * count_group * shape_cluster / iterations_cluster;
    } else {
        iterations_cluster = 1;
        delta_cluster = 1;
    }

    int delta_column = access_width * elements_per_access;

    // 1. stride
    *out_stride = (int64_t)shape_n * (element_size_bits / 8);

    // 2. increment_row
    *out_increment_row = *out_stride * delta_row;

    // 3. increment_group
    *out_increment_group = *out_stride * delta_group - *out_stride * delta_row * (iterations_row - 1);

    // 4. increment_cluster
    *out_increment_cluster = *out_stride * delta_cluster - *out_stride * delta_group * (iterations_group - 1) -
                             *out_stride * delta_row * (iterations_row - 1);

    // 5. advance_row
    *out_advance_row = *out_stride * shape_row;

    // 6. advance_group
    *out_advance_group = *out_stride * (shape_group - 1) * shape_row * count_row;

    // 7. advance_cluster
    *out_advance_cluster = *out_stride * count_group * shape_group * count_row * shape_row;

    // 8. advance_tile
    *out_advance_tile = *out_stride * shape_group * shape_row * shape_cluster * shape_tile;
}

} // namespace deep_gemm
