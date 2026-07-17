#pragma once

#ifndef USE_HGGC
__device__ __forceinline__ int atomic_add_release_global(int* addr, int value) {
    int ret;
    asm volatile ("ppu.atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(addr), "r"(value));
    return ret;
}
#endif
// When not compiling with hgcc (e.g. gcc host compilation via CppExtension),
// device function qualifiers are unknown and must be defined as empty macros.
#ifndef USE_HGGC
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif
#endif  // !USE_HGGC


enum class GemmType {
    DenseGemm,
    GroupedContiguous,
    GroupedMasked,
    GroupedNoPad,
    GroupedFused,
    BatchGemm,
};

const char* GemmTypeS[] = { "DenseGemm", "GroupedContiguous", "GroupedMasked", "GroupedNoPad", "GroupedFused", "BatchGemm"};

enum class KernelType {
    Default,
    MultistageOnN,
    MoeDynamicTile,
    OverlapPrologue,
    OverlapMainloop
};

const char* KernelTypeS[] = { "Default", "MultistageOnN", "MoeDynamicTile", "OverlapPrologue", "OverlapMainloop"};


template <typename T>
__device__ __host__ constexpr inline T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
__device__ __host__ constexpr T constexpr_gcd(T a, T b) {
    return b == 0 ? a : constexpr_gcd(b, a % b);
}

#if defined(__HGGC__)
__device__ __forceinline__ int atomic_add_release_global(int* addr, int value) {
    int ret;
    asm volatile ("ppu.atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(addr), "r"(value));
    return ret;
}
#endif  // __HGGC__

uint32_t next_power_of_two(uint32_t n) {
  if (n == 0) return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

#if defined(__HGGC__)
template <uint32_t BlockM>
__global__ void computeBlockInfoKernel(
    const uint32_t* __restrict__ group_num_list,
    const uint32_t group_num,
    uint32_t* __restrict__ block_info)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warp_id = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t num_warps = blockDim.x / 32;
    uint32_t group_val = tid < group_num ? group_num_list[tid] : 0;
    uint32_t block_val = (group_val + BlockM - 1) / BlockM;
    uint32_t warp_group_scan = group_val;
    uint32_t warp_block_scan = block_val;
    for (uint32_t offset = 1; offset < 32; offset *= 2) {
        uint32_t tmp_group = __shfl_up_sync(0xFFFFFFFF, warp_group_scan, offset);
        uint32_t tmp_block = __shfl_up_sync(0xFFFFFFFF, warp_block_scan, offset);
        if (lane >= offset) {
            warp_group_scan += tmp_group;
            warp_block_scan += tmp_block;
        }
    }

    __shared__ uint32_t warp_group_totals[32];
    __shared__ uint32_t warp_block_totals[32];

    if (lane == 31) {
        warp_group_totals[warp_id] = warp_group_scan;
        warp_block_totals[warp_id] = warp_block_scan;
    }
    __syncthreads();

    __shared__ uint32_t warp_group_prefix[32];
    __shared__ uint32_t warp_block_prefix[32];

    if (warp_id == 0) {
        uint32_t group_sum = 0;
        uint32_t block_sum = 0;
        for (uint32_t w = 0; w < num_warps; ++w) {
            warp_group_prefix[w] = group_sum;
            warp_block_prefix[w] = block_sum;
            group_sum += warp_group_totals[w];
            block_sum += warp_block_totals[w];
        }
        if (tid == 0) {
            *block_info = block_sum;
        }
    }
    __syncthreads();

    uint32_t group_prefix = warp_group_prefix[warp_id];
    uint32_t block_prefix = warp_block_prefix[warp_id];
    uint32_t warp_group_exclusive = warp_group_scan - group_val;
    uint32_t warp_block_exclusive = warp_block_scan - block_val;
    uint32_t global_group_prefix = group_prefix + warp_group_exclusive;
    uint32_t global_block_prefix = block_prefix + warp_block_exclusive;

    uint32_t base_offset = global_block_prefix * 4;
    uint32_t* output_info = block_info + 4;
    for (uint32_t i = 0; i < block_val; ++i) {
        uint32_t block_idx = base_offset + i * 4;
        output_info[block_idx]     = tid;          // group_idx
        output_info[block_idx + 1] = group_val;    // group_num
        output_info[block_idx + 2] = global_block_prefix; // prefix_block_m_idx
        output_info[block_idx + 3] = global_group_prefix; // prefix_group_sum
    }
}
#endif  // __HGGC__
