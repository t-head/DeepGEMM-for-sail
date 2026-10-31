#pragma once
#include <hggc_runtime.h>
#include <deep_gemm/common/w4a8_arguments.cuh>
namespace deep_gemm::w4fa8_streamed {
// Convert the public padded token layout to contiguous expert rows. Both
// activations and row scales are regenerated on every invocation/Graph replay.
template <int K, int BlockM, int Groups> __device__ __forceinline__ void gather_body(GatherParams p, int group_start) {
  constexpr int VectorsPerRow = K / 16;
  const int blocks = __ldg(p.aligned);
  for (int block = blockIdx.x; block < blocks; block += gridDim.x) {
    const uint4 expert = __ldg(reinterpret_cast<const uint4 *>(p.experts) + block);
    if (expert.x < uint32_t(group_start) || expert.x >= uint32_t(group_start + Groups))
      continue;
    const int row = (block - int(expert.z)) * BlockM;
    const int valid = min(BlockM, int(expert.y) - row);
    const int out_row = int(expert.w) + row;
    for (int vector = threadIdx.x; vector < valid * VectorsPerRow; vector += blockDim.x) {
      const int local_row = vector / VectorsPerRow, column = vector % VectorsPerRow;
      const int token = __ldg(p.tokens + int64_t(block) * BlockM + local_row);
      if (uint32_t(token) < uint32_t(p.num_tokens))
        reinterpret_cast<uint4 *>(p.grouped_a)[int64_t(out_row + local_row) * VectorsPerRow + column] =
            __ldg(reinterpret_cast<const uint4 *>(p.a) + int64_t(token) * VectorsPerRow + column);
    }
    for (int local_row = threadIdx.x; local_row < valid; local_row += blockDim.x) {
      const int token = __ldg(p.tokens + int64_t(block) * BlockM + local_row);
      if (uint32_t(token) < uint32_t(p.num_tokens))
        p.grouped_scale[out_row + local_row] = __ldg(p.a_scale + token);
    }
  }
}

// W4A8 preprocess confines adjusted UE8M0 to [114,140], which maps every
// finite E2M1 magnitude exactly to a normal finite E5M2 value.
__device__ __forceinline__ uint32_t fp4_four_to_e5(uint32_t codes, uint32_t low, uint32_t high) {
  uint32_t magnitude = __byte_perm(low, high, codes & 0x7777u);
  uint32_t sign = __byte_perm(0u, 0x80808080u, (codes >> 1) & 0x7777u);
  return magnitude | sign;
}
template <int N, int K, int TileN, int Groups>
__device__ __forceinline__ void decode_tile_body(const uint8_t* __restrict__ packed, const uint8_t* __restrict__ scales,
                                                 uint8_t* __restrict__ output, int n_begin,
                                                 const int32_t* __restrict__ all_rows, int group_start) {
  constexpr int VectorsPerRow = K / 16;
  const int expert_step = min(Groups, int(gridDim.x));
  const int first_expert = blockIdx.x % expert_step, local_block = blockIdx.x / expert_step;
  const int expert_blocks = (int(gridDim.x) + expert_step - 1 - first_expert) / expert_step;
  for (int expert = first_expert; expert < Groups; expert += expert_step) {
    // Prune the complete expert before its K/N vector loop, including replay.
    if (__ldg(all_rows + group_start + expert) == 0)
      continue;
    for (int64_t vector = int64_t(local_block) * 256 + threadIdx.x; vector < int64_t(TileN) * VectorsPerRow;
         vector += int64_t(expert_blocks) * 256) {
      const int nr = int(vector / VectorsPerRow) + n_begin, kr = int(vector % VectorsPerRow) * 16;
      const int64_t scale_offset = int64_t(expert) * N * K / 32 + int64_t(nr / 64) * (K * 2) +
                                   (kr / 64) * 128 + 16 * (nr % 8) + 2 * ((nr % 64) / 8) + (kr / 32 % 2);
      const uint32_t bias = (uint32_t(__ldg(scales + scale_offset)) - 114u) * 0x04040404u;
      const uint32_t low = (0x0a080400u + bias) & 0xffffff00u, high = 0x12100e0cu + bias;
      const uint2 p = __ldg(reinterpret_cast<const uint2*>(packed + (int64_t(expert) * N + nr) * (K / 2) + kr / 2));
      const uint4 decoded = {fp4_four_to_e5(p.x, low, high), fp4_four_to_e5(p.x >> 16, low, high),
                             fp4_four_to_e5(p.y, low, high), fp4_four_to_e5(p.y >> 16, low, high)};
      *reinterpret_cast<uint4*>(output + (int64_t(expert) * TileN * VectorsPerRow + vector) * 16) = decoded;
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int prefix = 0, count = 0;
    for (int i = 0; i < group_start; ++i)
      prefix += all_rows[i];
    for (int i = 0; i < Groups; ++i)
      count += all_rows[group_start + i];
    int32_t* metadata = reinterpret_cast<int32_t*>(output + int64_t(Groups) * TileN * K);
    metadata[0] = prefix;
    metadata[1] = count;
  }
}
// Exact P41 mixed-sign expansion, isolated from the legacy partial-N decoder.
__device__ __forceinline__ uint32_t fp4_four_to_e5_rows32(uint32_t codes, uint32_t low, uint32_t high) {
  uint32_t magnitude = __byte_perm(low, high, codes & 0x7777u);
  uint32_t expanded = __byte_perm(codes, codes << 4, 0x1504u);
  return magnitude | (expanded & 0x80808080u);
}
// Full-N rows reuse one adjusted UE8M0 scale for two adjacent 16-byte outputs.
// The generated global wrapper supplies maxnreg32 without launch_bounds.
template <int N, int K, int TileN, int Groups, bool SkipFence = false>
__device__ __forceinline__ void decode_fulln_rows32_body(const uint8_t* __restrict__ packed,
                                                         const uint8_t* __restrict__ scales,
                                                         uint8_t* __restrict__ output, int n_begin,
                                                         const int32_t* __restrict__ all_rows, int group_start) {
  static_assert(TileN == N, "Flat full-N decoder requires a complete physical N range");
  static_assert(uint64_t(Groups) * N * K <= 0xffffffffull, "Partial decoded buffer must fit uint32 addressing");
  constexpr uint32_t VectorsPerRow = K / 32;
  constexpr uint32_t VectorCount = uint32_t(uint64_t(Groups) * N * K / 16);
  // With fullN, expert-major rows are already contiguous: no per-vector expert/N divmod.
  // n_begin is zero by the complete physical-N caller contract.
  constexpr uint32_t TotalRows = Groups * N;
  const uint32_t lane = threadIdx.x & 31u;
  for (uint32_t row = uint32_t(blockIdx.x) * 8u + threadIdx.x / 32u; row < TotalRows; row += uint32_t(gridDim.x) * 8u) {
    const uint32_t row_scale = (row / 64u) * uint32_t(K * 2) + 16u * (row % 8u) + 2u * ((row % 64u) / 8u);
    const uint32_t packed_base = row * uint32_t(K / 2), output_base = row * uint32_t(K);
#pragma unroll 1
    for (uint32_t chunk = 0; chunk < (VectorsPerRow + 31u) / 32u; ++chunk) {
      const uint32_t column = 32u * chunk + lane;
      if (column < VectorsPerRow) {
        const uint32_t scale_offset = row_scale + (column / 2u) * 128u + (column % 2u);
        const uint32_t bias = (uint32_t(__ldg(scales + scale_offset)) - 114u) * 0x04040404u;
        const uint32_t low = (0x0a080400u + bias) & 0xffffff00u, high = 0x12100e0cu + bias;
#pragma unroll
        for (uint32_t half = 0; half < 2; ++half) {
          uint2 p = __ldg(reinterpret_cast<const uint2*>(packed + packed_base + column * 16u + half * 8u));
          uint4 decoded = {fp4_four_to_e5_rows32(p.x, low, high), fp4_four_to_e5_rows32(p.x >> 16, low, high),
                           fp4_four_to_e5_rows32(p.y, low, high), fp4_four_to_e5_rows32(p.y >> 16, low, high)};
          *reinterpret_cast<uint4*>(output + output_base + column * 32u + half * 16u) = decoded;
          // SDK 2.2 lowers this legacy clobber to a block VMEM fence.
          // The selected short-call tail decoder omits it after measured validation.
          if constexpr (!SkipFence)
            asm volatile("" ::: "memory");
        }
      }
    }
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int prefix = 0, count = 0;
    for (int i = 0; i < group_start; ++i)
      prefix += all_rows[i];
    for (int i = 0; i < Groups; ++i)
      count += all_rows[group_start + i];
    int32_t* metadata = reinterpret_cast<int32_t*>(output + int64_t(Groups) * TileN * K);
    metadata[0] = prefix;
    metadata[1] = count;
  }
}

} // namespace deep_gemm::w4fa8_streamed

namespace deep_gemm::w4fa8_empty_flat {
template <int N, int K, int TileN, int Groups, int Threads, bool SkipFence>
__device__ __forceinline__ void decode_empty_flat_body(const uint8_t* __restrict__ packed,
                                                       const uint8_t* __restrict__ scales, uint8_t* __restrict__ output,
                                                       int n_begin, const int32_t* __restrict__ all_rows,
                                                       int group_start) {
  static_assert(TileN == N, "Flat full-N decoder requires a complete physical N range");
  static_assert(uint64_t(Groups) * N * K <= 0xffffffffull, "Partial decoded buffer must fit uint32 addressing");
  constexpr uint32_t VectorCount = uint32_t(uint64_t(Groups) * N * K / 16);
  static_assert(K % 128 == 0 && Threads % 32 == 0);
  // Each CTA is assigned to one expert; its row count is read once before
  // touching packed weights. Empty experts retain their untouched weight slot.
  const uint32_t vote_lane = threadIdx.x & 31u;
  uint32_t local_valid = 1;
  if (vote_lane == 0) {
#pragma unroll
    for (int expert = 0; expert < Groups; ++expert)
      local_valid &= uint32_t(__ldg(all_rows + group_start + expert) != 0);
  }
  const bool all_nonempty = __shfl_sync(0xffffffffu, local_valid, 0) != 0;
  constexpr uint32_t VectorsPerRow = K / 32;
  const uint32_t lane = threadIdx.x & 31u;
  for (uint32_t row = uint32_t(blockIdx.x) * 8u + threadIdx.x / 32u; row < uint32_t(Groups * N);
       row += uint32_t(gridDim.x) * 8u) {
    if (!all_nonempty && __ldg(all_rows + group_start + row / uint32_t(N)) == 0)
      continue;
    const uint32_t row_scale = (row / 64u) * uint32_t(K * 2) + 16u * (row % 8u) + 2u * ((row % 64u) / 8u);
    const uint32_t packed_base = row * uint32_t(K / 2), output_base = row * uint32_t(K);
#pragma unroll 1
    for (uint32_t chunk = 0; chunk < (VectorsPerRow + 31u) / 32u; ++chunk) {
      const uint32_t column = 32u * chunk + lane;
      if (column < VectorsPerRow) {
        const uint32_t scale_offset = row_scale + (column / 2u) * 128u + (column % 2u);
        const uint32_t bias = (uint32_t(__ldg(scales + scale_offset)) - 114u) * 0x04040404u;
        const uint32_t low = (0x0a080400u + bias) & 0xffffff00u, high = 0x12100e0cu + bias;
#pragma unroll
        for (uint32_t half = 0; half < 2; ++half) {
          uint2 p = __ldg(reinterpret_cast<const uint2*>(packed + packed_base + column * 16u + half * 8u));
          uint4 decoded = {w4fa8_streamed::fp4_four_to_e5_rows32(p.x, low, high),
                           w4fa8_streamed::fp4_four_to_e5_rows32(p.x >> 16, low, high),
                           w4fa8_streamed::fp4_four_to_e5_rows32(p.y, low, high),
                           w4fa8_streamed::fp4_four_to_e5_rows32(p.y >> 16, low, high)};
          *reinterpret_cast<uint4*>(output + output_base + column * 32u + half * 16u) = decoded;
          if constexpr (!SkipFence)
            asm volatile("" ::: "memory");
        }
      }
    }
  }
  // Metadata is always produced, including when expert zero is empty.
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    int prefix = 0, count = 0;
    for (int i = 0; i < group_start; ++i)
      prefix += all_rows[i];
    for (int i = 0; i < Groups; ++i)
      count += all_rows[group_start + i];
    int32_t* metadata = reinterpret_cast<int32_t*>(output + int64_t(Groups) * TileN * K);
    metadata[0] = prefix;
    metadata[1] = count;
  }
}
} // namespace deep_gemm::w4fa8_empty_flat
