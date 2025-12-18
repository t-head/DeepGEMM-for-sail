#pragma once

/*! \file
    \brief Parameters structures for deepgemm schedulers
*/

#include "utils.cuh"
#include "cutlass/coord.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/workspace.h"
#include "cutlass/platform/platform.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm_coord.hpp"
////////////////////////////////////////////////////////////////////////////////

// namespace cutlass::gemm::kernel {
namespace deep_gemm {
using cutlass::KernelHardwareInfo;

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
template <GemmType kGemmType,
          uint32_t SHAPE_N_, uint32_t SHAPE_K_,
          uint32_t BLOCK_M_, uint32_t BLOCK_N_,
          uint32_t kNumGroups,
          uint32_t kNumNBlocks = ceil_div(SHAPE_N_, BLOCK_N_),
          uint32_t kNum1DBlocksPerGroup = 2>
struct DeepGemmScheduler {
    constexpr static uint32_t SHAPE_N = SHAPE_N_;
    constexpr static uint32_t SHAPE_K = SHAPE_K_;
    constexpr static uint32_t BLOCK_M = BLOCK_M_;
    constexpr static uint32_t BLOCK_N = BLOCK_N_;
    int current_iter = 0;
    uint32_t num_aligned_m_blocks;
    constexpr static GemmType GEMM_TYPE = kGemmType;
    constexpr static bool kIsTMAMulticastOnA = false;

    // For normal GEMM
    // Maybe not used in the masked grouped GEMM
    uint32_t num_blocks;
    uint32_t num_n_blocks = kNumNBlocks;

    // Only used for masked layout
    uint32_t curr_group_idx, curr_cumsum, curr_group_m, curr_cumsum_m;

    struct Arguments
    {
        int* grouped_layout;
        uint32_t shape_m;

        //
        // Methods
        //

        /// Ctor
        CUTLASS_HOST_DEVICE
        Arguments()
            : grouped_layout(nullptr)
            , shape_m(0)
        {
        }

        /// Ctor
        CUTLASS_HOST_DEVICE
        Arguments(uint32_t shape_m, int* grouped_layout_ptr = nullptr)
            : grouped_layout(grouped_layout_ptr)
            , shape_m(shape_m)
        {
        }

    };

    using Params = Arguments;
    Params const& params;

    CUTLASS_DEVICE explicit DeepGemmScheduler(Params const& params_, const int warp_group_id = 0) : params(params_), current_iter(warp_group_id) {
        num_aligned_m_blocks = ceil_div(params_.shape_m, BLOCK_M);
        if (kGemmType == GemmType::DenseGemm) {
            num_blocks = num_aligned_m_blocks * num_n_blocks;
        } else if (kGemmType == GemmType::GroupedContiguous) {
            num_blocks = num_aligned_m_blocks * num_n_blocks;
        } else if (kGemmType == GemmType::GroupedMasked || kGemmType == GemmType::GroupedNoPad) {
            curr_group_idx = curr_cumsum = curr_group_m = curr_cumsum_m = 0;
        }
    }

    CUTLASS_DEVICE void get_swizzled_block_idx(const uint32_t num_m_blocks, int block_idx,
                                               uint32_t& m_block_idx, uint32_t& n_block_idx) {
        // Swizzle for better L2 usages
        auto primary_num_blocks = kIsTMAMulticastOnA ? kNumNBlocks : num_m_blocks;
        auto secondary_num_blocks = kIsTMAMulticastOnA ? num_m_blocks : kNumNBlocks;
        auto num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;
        auto group_idx = block_idx / num_blocks_per_group;
        auto first_block_idx = group_idx * kNum1DBlocksPerGroup;
        auto in_group_idx = block_idx % num_blocks_per_group;
        uint32_t num_blocks_in_group = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);

        // Convert to final M/N block indices
        if constexpr (kIsTMAMulticastOnA) {
            m_block_idx = in_group_idx / num_blocks_in_group;
            n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
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
        const auto next_block_idx = (current_iter++) * gridDim.x + blockIdx.x;

        if (kGemmType == GemmType::GroupedMasked || kGemmType == GemmType::GroupedNoPad) {
            uint32_t num_m_blocks;
            while (true) {
                // End of the task
                if (curr_group_idx == kNumGroups)
                    return false;

                // Within the current group
                curr_group_m = static_cast<uint32_t>(__ldg(params.grouped_layout + curr_group_idx));
                num_m_blocks = ceil_div(curr_group_m, BLOCK_M);
                auto current_m_block_cumsum = curr_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * kNumNBlocks)
                    break;

                // Move to check the next group
                curr_group_idx ++, curr_cumsum = current_m_block_cumsum;
                curr_cumsum_m += curr_group_m;
            }

            get_swizzled_block_idx(num_m_blocks, next_block_idx - curr_cumsum * kNumNBlocks, m_block_idx, n_block_idx);
        } else {
            if (next_block_idx >= num_blocks)
                return false;

            get_swizzled_block_idx(num_aligned_m_blocks, next_block_idx, m_block_idx, n_block_idx);
        }
        return true;
    }


    template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
    static Params
    to_underlying_arguments(
      int* groups_layout,
      ProblemShapeMNKL problem_shape_mnkl,
      TileShape tile_shape,
      ClusterShape cluster_shape,
      [[maybe_unused]] KernelHardwareInfo const& hw_info,
      Arguments const& arguments,
      [[maybe_unused]] void* workspace=nullptr,
      [[maybe_unused]] const uint32_t epilogue_subtile = 1,
      [[maybe_unused]] uint32_t ktile_start_alignment_count = 1u) {

        // cutlass3 change
        // rtc will use this to get grid size on host
        #ifndef ACOMPUTE_VERSION
            // We only need the tile and cluster shape during scheduler setup, so let FTAD do the magic
            static_assert(cute::is_static<TileShape>::value);
            static_assert(cute::is_static<ClusterShape>::value);
        #endif

        // dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);
        auto problem_shape = cutlass::gemm::to_gemm_coord(problem_shape_mnkl);

        Params params(problem_shape.m(), groups_layout);
        return params;
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
            return params.shape_m;
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

    // Gets the pointer offset of matrix C
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

