#pragma once

/*! \file
    \brief Parameters structures for deepgemm schedulers
*/

#include "utils_rtc.cuh"
#include "cutlass/coord.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/workspace.h"
#include "cutlass/platform/platform.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm_coord.hpp"
#include "cutlass/cutlass.h"
////////////////////////////////////////////////////////////////////////////////

// namespace cutlass::gemm::kernel {
namespace deep_gemm {
using cutlass::KernelHardwareInfo;

struct TileSchedulerArguments
{
    uint32_t shape_m;
    int tb_per_cu;
    int cu_count;
    int* grouped_layout;

    //
    // Methods
    //

    /// Ctor
    CUTLASS_HOST_DEVICE
    TileSchedulerArguments()
        : shape_m(0), tb_per_cu(0), cu_count(0), grouped_layout(nullptr)
    {
    }

    /// Ctor
    CUTLASS_HOST_DEVICE
    TileSchedulerArguments(uint32_t shape_m, int tb_per_cu, int cu_count, int* grouped_layout_ptr = nullptr)
        : shape_m(shape_m), tb_per_cu(tb_per_cu), cu_count(cu_count), grouped_layout(grouped_layout_ptr)
    {
    }

};
using TileSchedulerParams = TileSchedulerArguments;

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
template <GemmType kGemmType,
          uint32_t SHAPE_N_, uint32_t SHAPE_K_,
          uint32_t BLOCK_M_, uint32_t BLOCK_N_,
          uint32_t kNumGroups_,
          uint32_t kNumNBlocks_ = ceil_div(SHAPE_N_, BLOCK_N_),
          uint32_t kNum1DBlocksPerGroup = 2
          >
struct DeepGemmScheduler {
    constexpr static uint32_t SHAPE_N = SHAPE_N_;
    constexpr static uint32_t SHAPE_K = SHAPE_K_;
    constexpr static uint32_t BLOCK_M = BLOCK_M_;
    constexpr static uint32_t BLOCK_N = BLOCK_N_;
    constexpr static uint32_t kNumGroups = kNumGroups_;
    constexpr static uint32_t kNumNBlocks = kNumNBlocks_;
    int current_iter = 0;
    uint32_t num_aligned_m_blocks;
    constexpr static GemmType GEMM_TYPE = kGemmType;
    constexpr static bool kIsTMAMulticastOnA = false;
    // disable tileM=16 or Occpuancy = 1
    static constexpr bool EnableHWDispatchStrategy = BLOCK_M >= 32 && BLOCK_M <= 128 || (BLOCK_M == 192 && BLOCK_N == 128);
    constexpr static bool kIsNoPadPreprocessLayout = kGemmType == GemmType::GroupedNoPad || kGemmType == GemmType::GroupedFused;

    // For normal GEMM
    // Maybe not used in the masked grouped GEMM
    uint32_t num_blocks;

    // Only used for masked layout
    uint32_t curr_group_idx, curr_cumsum, curr_cumsum_blocks, curr_group_m, curr_cumsum_m;

    // last round
    int last_round_wave;
    int last_round_idx;

    using Arguments = TileSchedulerArguments;
    using Params = TileSchedulerArguments;
    Params const& params;

    CUTLASS_DEVICE explicit DeepGemmScheduler(Params const& params_, const int total_blocks = 0, const int warp_group_id = 0) : params(params_), current_iter(warp_group_id) {
        num_aligned_m_blocks = ceil_div(params_.shape_m, BLOCK_M);
        if (kGemmType == GemmType::DenseGemm) {
            num_blocks = num_aligned_m_blocks * kNumNBlocks;
        } else if (kGemmType == GemmType::GroupedContiguous) {
            num_blocks = num_aligned_m_blocks * kNumNBlocks;
        } else if (kIsNoPadPreprocessLayout) {
            num_aligned_m_blocks = __ld_smem(params_.grouped_layout); // total blocks in m, block_m_sum
            curr_group_idx = curr_cumsum = curr_cumsum_blocks = curr_group_m = curr_cumsum_m = 0;
            num_blocks = num_aligned_m_blocks * kNumNBlocks;
        } else if (kGemmType == GemmType::GroupedMasked) {
            if constexpr (EnableHWDispatchStrategy) {
                if (total_blocks == 0) {
                    int m_blocks_sum = 0;
                    for (int i = 0; i < kNumGroups; i++) {
                        int curr_group_m = __ld_smem(params.grouped_layout + i);
                        m_blocks_sum += cute::ceil_div(curr_group_m, BLOCK_M);
                    }
                    num_aligned_m_blocks = m_blocks_sum; // total blocks in m, block_m_sum
                    num_blocks = num_aligned_m_blocks * kNumNBlocks;
                } else {
                    // dynamic tile, compute total blocks in kernel
                    num_blocks = total_blocks;
                }
            }
            curr_group_idx = curr_cumsum = curr_group_m = curr_cumsum_blocks = curr_cumsum_m = 0;
        } else if (kGemmType == GemmType::GroupedNoPad || kGemmType == GemmType::GroupedFused) {
            curr_group_idx = curr_cumsum = curr_cumsum_blocks = curr_group_m = curr_cumsum_m = 0;
            num_blocks = 0;
        }
        // compute last round idx, to keep work balance in last round
        if constexpr (EnableHWDispatchStrategy) {
            if (params_.cu_count == 39) {
                last_round_wave = num_blocks / gridDim.x;
                int ce_idx = blockIdx.x / (params_.tb_per_cu * 4);
                int tb_idx_in_ce = blockIdx.x % (params_.tb_per_cu * 4);
                int cu_idx = ce_idx < 9 ? tb_idx_in_ce % 4 : tb_idx_in_ce % 3;
                int tb_idx_in_cu = ce_idx < 9 ? tb_idx_in_ce / 4 : tb_idx_in_ce / 3;
                last_round_idx = ce_idx * 4 + cu_idx + tb_idx_in_cu * params_.cu_count;
            } else if (params_.cu_count % 4 == 0) {
                last_round_wave = num_blocks / gridDim.x;
                int ce_idx = blockIdx.x / (params_.tb_per_cu * 4);
                int tb_idx_in_ce = blockIdx.x % (params_.tb_per_cu * 4);
                int cu_idx = tb_idx_in_ce % 4;
                int tb_idx_in_cu = tb_idx_in_ce / 4;
                last_round_idx = ce_idx * 4 + cu_idx + tb_idx_in_cu * params_.cu_count;
            } else {
                last_round_idx = blockIdx.x;
            }
        } else {
            last_round_wave = 0; // not used
            last_round_idx = blockIdx.x; // not used
        }
    }

    CUTLASS_DEVICE void get_swizzled_block_idx(const uint32_t num_m_blocks, int block_idx,
                                               uint32_t& m_block_idx, uint32_t& n_block_idx, int n_expand=1) {
        // Swizzle for better L2 usages
        auto primary_num_blocks = kIsTMAMulticastOnA ? kNumNBlocks : num_m_blocks;
        auto secondary_num_blocks = kIsTMAMulticastOnA ? num_m_blocks : (kNumNBlocks / n_expand);
        auto num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;
        auto group_idx = block_idx / num_blocks_per_group;
        auto first_block_idx = group_idx * kNum1DBlocksPerGroup;
        auto in_group_idx = block_idx % num_blocks_per_group;
        uint32_t num_blocks_in_group = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);

        // Convert to final M/N block indices
        if constexpr (kIsTMAMulticastOnA) {
            m_block_idx = in_group_idx / num_blocks_in_group;
            n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        } else if constexpr (kNum1DBlocksPerGroup == 2) {
            auto sel = (num_blocks_in_group >> 1) & 1;
            m_block_idx = first_block_idx + (((in_group_idx ^ (in_group_idx >> 2)) & 1) & sel);
            n_block_idx = in_group_idx >> sel;
        } else {
            m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
            n_block_idx = in_group_idx / num_blocks_in_group;
        }
    }


    template <bool kIgnoreGroupedForGroupedContiguous=true>
    CUTLASS_DEVICE uint32_t get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
                                           const uint32_t& block_idx, const uint32_t& m_block_idx=0) {
        if (kGemmType == GemmType::DenseGemm) {
            return block_idx * block_size;
        } else if (kGemmType == GemmType::GroupedContiguous) {
            auto offset = kIgnoreGroupedForGroupedContiguous ? 0 : __ldg(params.grouped_layout + m_block_idx * BLOCK_M);
            return offset * shape_dim + block_idx * block_size;
        } else if (kGemmType == GemmType::GroupedMasked) {
            return curr_group_idx * shape_dim + block_idx * block_size;
        }
    }

    CUTLASS_DEVICE bool fetch_next_work(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        int tb_idx = blockIdx.x;
        if constexpr (EnableHWDispatchStrategy) {
            tb_idx = current_iter < last_round_wave ? blockIdx.x : last_round_idx;
        }
        const auto next_block_idx = current_iter++ * gridDim.x + tb_idx;
        if (kIsNoPadPreprocessLayout) {
            if (next_block_idx >= num_blocks) {
                m_block_idx = num_aligned_m_blocks;
                n_block_idx = kNumNBlocks;
                return false;
            }
            int block_m_idx = next_block_idx / kNumNBlocks;
            const uint4 data = __ld_smem((const uint4*)params.grouped_layout + 1 + block_m_idx);
            curr_group_idx = data.x;
            curr_group_m = data.y;
            uint32_t block_idx_in_m = next_block_idx - data.z * kNumNBlocks;
            uint32_t num_m_blocks = ceil_div(curr_group_m, BLOCK_M);
            get_swizzled_block_idx(num_m_blocks, block_idx_in_m, m_block_idx, n_block_idx);
            if constexpr(kGemmType == GemmType::GroupedFused) {
                m_block_idx += data.z;
            } else {
                curr_cumsum_m = data.w;
            }
        } else if constexpr(kGemmType == GemmType::GroupedMasked || kGemmType == GemmType::GroupedNoPad || kGemmType == GemmType::GroupedFused) {
            uint32_t num_m_blocks;
            while (true) {
                // End of the task
                if (curr_group_idx == kNumGroups) {
                    m_block_idx = num_m_blocks;
                    n_block_idx = kNumNBlocks;
                    return false;
                }
                // Within the current group
                curr_group_m = static_cast<uint32_t>(__ld_smem(params.grouped_layout + curr_group_idx));
                num_m_blocks = ceil_div(curr_group_m, BLOCK_M);
                auto current_m_block_cumsum = curr_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * kNumNBlocks)
                    break;
                // Move to check the next group
                curr_group_idx ++, curr_cumsum = current_m_block_cumsum;
                curr_cumsum_m += curr_group_m;
            }
            get_swizzled_block_idx(num_m_blocks, next_block_idx - curr_cumsum * kNumNBlocks, m_block_idx, n_block_idx);
            if constexpr(kGemmType == GemmType::GroupedFused) {
                m_block_idx += curr_cumsum;
            }
        } else {
            if (next_block_idx >= num_blocks) {
                m_block_idx = num_aligned_m_blocks;
                n_block_idx = kNumNBlocks;
                return false;
            }
            get_swizzled_block_idx(num_aligned_m_blocks, next_block_idx, m_block_idx, n_block_idx);
        }
        return true;
    }

    template<bool kEnableNExpand>
    CUTLASS_DEVICE int get_n_expand(int curr_group_m) {
        int n_expand = 1;
        if constexpr (kEnableNExpand) {
            if ((SHAPE_K > 2048 && curr_group_m > 32 && curr_group_m <= 64)
                || (SHAPE_K <= 2048 && curr_group_m <= 64)) {
                n_expand = 2;
            }
            return n_expand;
        } else {
            return 1;
        }
    }

    template<bool kEnableNExpand = true>
    CUTLASS_DEVICE bool fetch_next_work_dynamic_tile(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        int tb_idx = blockIdx.x;
        if constexpr (EnableHWDispatchStrategy) {
            tb_idx = current_iter < last_round_wave ? blockIdx.x : last_round_idx;
        }
        const auto next_block_idx = current_iter++ * gridDim.x + tb_idx;
        if (kIsNoPadPreprocessLayout) {
            if (next_block_idx >= num_blocks) {
                m_block_idx = num_aligned_m_blocks;
                n_block_idx = kNumNBlocks;
                return false;
            }
            int block_m_idx = next_block_idx / kNumNBlocks;
            const uint4 data = __ld_smem((const uint4*)params.grouped_layout + 1 + block_m_idx);
            curr_group_idx = data.x;
            curr_group_m = data.y;
            uint32_t block_idx_in_m = next_block_idx - data.z * kNumNBlocks;
            uint32_t num_m_blocks = ceil_div(curr_group_m, BLOCK_M);
            curr_cumsum_m = data.w;
            get_swizzled_block_idx(num_m_blocks, block_idx_in_m, m_block_idx, n_block_idx);
        } else if (kGemmType == GemmType::GroupedMasked || kGemmType == GemmType::GroupedNoPad) {
            uint32_t num_m_blocks;
            int n_expand = 1;
            int curr_cumsum_blocks_prev;

            while (true) {
                // End of the task
                if (curr_group_idx == kNumGroups)
                    return false;

                // Within the current group
                curr_group_m = static_cast<uint32_t>(__ld_smem(params.grouped_layout + curr_group_idx));
                n_expand = get_n_expand<kEnableNExpand>(curr_group_m);
                num_m_blocks = ceil_div(curr_group_m, BLOCK_M);
                auto current_m_block_cumsum = curr_cumsum + num_m_blocks;
                curr_cumsum_blocks_prev = curr_cumsum_blocks;
                int curr_cumsum_blocks_next = curr_cumsum_blocks + num_m_blocks * kNumNBlocks / n_expand;

                // if (cute::thread0()) {
                //     printf("curr_group_idx = %d, curr_cumsum_blocks = %d, num_m_blocks = %d, next_block_idx = %d, curr_cumsum_blocks_prev = %d, curr_cumsum_blocks_next = %d, n_expand = %d\n",
                //         curr_group_idx, curr_cumsum_blocks, num_m_blocks, next_block_idx, curr_cumsum_blocks_prev, curr_cumsum_blocks_next, n_expand);
                // }
                if (next_block_idx < curr_cumsum_blocks_next)
                    break;

                // Move to check the next group
                curr_cumsum_blocks = curr_cumsum_blocks_next;
                curr_group_idx ++, curr_cumsum = current_m_block_cumsum;
                curr_cumsum_m += curr_group_m;
            }

            get_swizzled_block_idx(num_m_blocks, next_block_idx - curr_cumsum_blocks_prev, m_block_idx, n_block_idx, n_expand);
        } else {
            if (next_block_idx >= num_blocks)
                return false;

            get_swizzled_block_idx(num_aligned_m_blocks, next_block_idx, m_block_idx, n_block_idx);
        }
        return true;
    }

    // The basic tile scheduler does not require any additional workspace
    template <class ProblemShape, class ElementAccumulator>
    static size_t
    get_workspace_size(Arguments const&, ProblemShape, KernelHardwareInfo const&, uint32_t, const uint32_t = 1, uint32_t = 1) {
        return 0;
    }

    // Returns the problem size for the current problem
    __device__ __forceinline__ int32_t curr_problem_m() const
    {
        if constexpr (kGemmType == GemmType::DenseGemm || kGemmType == GemmType::GroupedContiguous) {
            return params.shape_m;
        } else if constexpr (kGemmType == GemmType::GroupedMasked) {
            return curr_group_m;
        } else if constexpr (kGemmType == GemmType::GroupedNoPad) {
            return curr_group_m;
        } else {
            return 0;
        }
    }

    // Gets the index of the problem
    __device__ __forceinline__ int32_t problem_index() const
    {
        return curr_group_idx;
    }

    // Gets the pointer offset of matrix A
    __device__ __forceinline__ int64_t curr_offset_a() const
    {
        if constexpr (kGemmType == GemmType::GroupedMasked) {
            return int64_t(curr_group_idx) * params.shape_m * SHAPE_K;
        } else if constexpr (kGemmType == GemmType::GroupedNoPad) {
            return int64_t(curr_cumsum_m) * SHAPE_K;
        } else {
            return 0;
        }
    }

    __device__ __forceinline__ int64_t curr_offset_scalea() const
    {
        if constexpr (kGemmType == GemmType::GroupedMasked) {
            return int64_t(curr_group_idx) * params.shape_m * SHAPE_K / 128;
        } else if constexpr (kGemmType == GemmType::GroupedNoPad) {
            return int64_t(curr_cumsum_m);
        } else {
            return 0;
        }
    }

    // Gets the pointer offset of matrix A_scale for MXFP4, scale is padded to uint64_t but load as uint32_t.
    __device__ __forceinline__ int64_t curr_offset_mxfp4_scalea() const
    {
        // scales are organized in uint16_t
        uint32_t shape_k_scale = ceil_div(SHAPE_K, (uint32_t)32);
        if constexpr (kGemmType == GemmType::GroupedNoPad) {
            // /4 means uint8_t to uint32_t;
            return int64_t(curr_cumsum_m);
        } else if constexpr (kGemmType == GemmType::GroupedMasked) {
            return int64_t(curr_group_idx) * params.shape_m * shape_k_scale;
        } else {
            return 0;
        }
    }

    /// Gets the pointer offset of matrix A
    __device__ __forceinline__ int64_t curr_offset_m() const
    {
        if constexpr (kGemmType == GemmType::GroupedMasked) {
            return int64_t(curr_group_idx) * params.shape_m;
        } else if constexpr (kGemmType == GemmType::GroupedNoPad) {
            return curr_cumsum_m;
        } else {
            return 0;
        }
    }

    // Gets the pointer offset of matrix B
    __device__ __forceinline__ int64_t curr_offset_b(const int m_block_idx = 0) const
    {
        if constexpr (kGemmType == GemmType::GroupedContiguous) {
            int64_t offset = __ldg(params.grouped_layout + m_block_idx * BLOCK_M);
            return offset * SHAPE_N * SHAPE_K;
        } else {
            return int64_t(curr_group_idx) * SHAPE_N * SHAPE_K;
        }
    }

    // Gets the pointer offset of matrix B_scale for MXFP4, scale is padded to uint64_t but load as uint32_t.
    __device__ __forceinline__ int64_t curr_offset_mxfp4_scaleb(const int m_block_idx = 0) const
    {
        // scale are organized in uint16_t
        uint32_t shape_k_scale = ceil_div(SHAPE_K, (uint32_t)32);

        return int64_t(curr_group_idx) * SHAPE_N * shape_k_scale;
    }

    // Gets the pointer offset of matrix C
    __device__ __forceinline__ int64_t curr_offset_mxfp4_c() const
    {
        if constexpr (kGemmType == GemmType::GroupedNoPad || kGemmType == GemmType::GroupedMasked) {
            return int64_t(curr_group_idx) * SHAPE_N;
        } else {
            return 0;
        }
    }

    __device__ __forceinline__ int64_t curr_offset_c() const
    {
        if constexpr (kGemmType == GemmType::GroupedMasked || kGemmType == GemmType::GroupedContiguous) {
            return int64_t(curr_group_idx) * params.shape_m * SHAPE_N;
        } else if constexpr (kGemmType == GemmType::GroupedNoPad) {
            return int64_t(curr_cumsum_m) * SHAPE_N;
        } else {
            return 0;
        }
    }
};

#pragma clang diagnostic pop

}

