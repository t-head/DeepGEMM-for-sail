#pragma once

/*! \file
    \brief Simplified tile scheduler for DenseGemm (no SHAPE_N/SHAPE_K template params).
    num_n_blocks is computed at RUNTIME from params.shape_n.
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

namespace deep_gemm {
using cutlass::KernelHardwareInfo;

/// Arguments / Params for DenseGemm tile scheduler.
/// Contains runtime shape_n and shape_k (no longer template params).
struct DenseGemmTileSchedulerArguments {
    int* grouped_layout;   // nullptr for DenseGemm (interface compat)
    uint32_t shape_m;
    uint32_t shape_n;
    uint32_t shape_k;

    CUTLASS_HOST_DEVICE
    DenseGemmTileSchedulerArguments()
        : grouped_layout(nullptr), shape_m(0), shape_n(0), shape_k(0) {}

    CUTLASS_HOST_DEVICE
    DenseGemmTileSchedulerArguments(uint32_t shape_m_, uint32_t shape_n_, uint32_t shape_k_,
                                    int* grouped_layout_ptr = nullptr)
        : grouped_layout(grouped_layout_ptr), shape_m(shape_m_), shape_n(shape_n_), shape_k(shape_k_) {}
};

using DenseGemmTileSchedulerParams = DenseGemmTileSchedulerArguments;

////////////////////////////////////////////////////////////////////////////////

#ifdef __clang__
#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
#endif

/// Simplified DenseGemm scheduler.
/// Template only on BLOCK_M and BLOCK_N (which includes N_EXPAND).
/// SHAPE_N and SHAPE_K are stored at runtime in params.
template <uint32_t BLOCK_M_, uint32_t BLOCK_N_, uint32_t kNum1DBlocksPerGroup = 2>
struct DenseGemmScheduler {
    constexpr static uint32_t BLOCK_M = BLOCK_M_;
    constexpr static uint32_t BLOCK_N = BLOCK_N_;
    constexpr static GemmType GEMM_TYPE = GemmType::DenseGemm;
    constexpr static bool kIsTMAMulticastOnA = false;
    constexpr static bool kIsNoPadPreprocessLayout = false;
    constexpr static uint32_t SHAPE_N = 0;    // Runtime N; 0 signals shape-agnostic
    constexpr static uint32_t SHAPE_K = 0;    // Runtime K; 0 signals shape-agnostic
    constexpr static uint32_t kNumGroups = 1; // DenseGemm is always single-problem

    int current_iter = 0;
    uint32_t num_aligned_m_blocks;
    uint32_t num_blocks;
    uint32_t num_n_blocks;  // computed at runtime: ceil_div(shape_n, BLOCK_N)

    // Host-callable type definitions
    using Arguments = DenseGemmTileSchedulerArguments;
    using Params = DenseGemmTileSchedulerParams;
    Params const& params;

#if defined(__HGGC__)
    // --- Device-only methods below ---

    CUTLASS_DEVICE explicit DenseGemmScheduler(Params const& params_, const int warp_group_id = 0)
        : current_iter(warp_group_id), params(params_) {
        num_aligned_m_blocks = ceil_div(params_.shape_m, BLOCK_M);
        num_n_blocks = ceil_div(params_.shape_n, BLOCK_N);
        num_blocks = num_aligned_m_blocks * num_n_blocks;
    }

    CUTLASS_DEVICE void get_swizzled_block_idx(const uint32_t num_m_blocks, int block_idx,
                                               uint32_t& m_block_idx, uint32_t& n_block_idx) {
        // Swizzle for better L2 usages
        auto primary_num_blocks = num_m_blocks;
        auto secondary_num_blocks = num_n_blocks;
        auto num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;
        auto group_idx = block_idx / num_blocks_per_group;
        auto first_block_idx = group_idx * kNum1DBlocksPerGroup;
        auto in_group_idx = block_idx % num_blocks_per_group;
        uint32_t num_blocks_in_group = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);

        // Convert to final M/N block indices
        m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        n_block_idx = in_group_idx / num_blocks_in_group;
    }

    CUTLASS_DEVICE uint32_t get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
                                           const uint32_t& block_idx, const uint32_t& m_block_idx = 0) {
        return block_idx * block_size;
    }

    CUTLASS_DEVICE bool fetch_next_work(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const auto next_block_idx = (current_iter++) * gridDim.x + blockIdx.x;
        if (next_block_idx >= num_blocks) {
            m_block_idx = num_aligned_m_blocks;
            n_block_idx = num_n_blocks;
            return false;
        }
        get_swizzled_block_idx(num_aligned_m_blocks, next_block_idx, m_block_idx, n_block_idx);
        return true;
    }

    // Returns the problem size M for the current problem
    __device__ __forceinline__ int32_t curr_problem_m() const {
        return params.shape_m;
    }

    // DenseGemm always has a single problem (index 0)
    __device__ __forceinline__ int32_t problem_index() const {
        return 0;
    }

    // Offset stubs — DenseGemm offsets are always 0
    __device__ __forceinline__ int64_t curr_offset_a() const { return 0; }
    __device__ __forceinline__ int64_t curr_offset_b(const int m_block_idx = 0) const { return 0; }
    __device__ __forceinline__ int64_t curr_offset_c() const { return 0; }
    __device__ __forceinline__ int64_t curr_offset_scalea() const { return 0; }
    __device__ __forceinline__ int64_t curr_offset_m() const { return 0; }

#endif  // defined(__HGGC__)

    // Host-callable: to_underlying_arguments
    template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
    static Params
    to_underlying_arguments(
      int* groups_layout,
      ProblemShapeMNKL problem_shape_mnkl,
      TileShape tile_shape,
      ClusterShape cluster_shape,
      [[maybe_unused]] KernelHardwareInfo const& hw_info,
      Arguments const& arguments,
      [[maybe_unused]] void* workspace = nullptr,
      [[maybe_unused]] const uint32_t epilogue_subtile = 1,
      [[maybe_unused]] uint32_t ktile_start_alignment_count = 1u) {

        static_assert(cute::is_static<TileShape>::value);
        static_assert(cute::is_static<ClusterShape>::value);

        auto problem_shape = cutlass::gemm::to_gemm_coord(problem_shape_mnkl);
        Params p(problem_shape.m(), problem_shape.n(), problem_shape.k(), groups_layout);
        return p;
    }

    // The DenseGemm scheduler does not require any additional workspace
    template <class ProblemShape, class ElementAccumulator>
    static size_t
    get_workspace_size(Arguments const&, ProblemShape, KernelHardwareInfo const&, uint32_t, const uint32_t = 1, uint32_t = 1) {
        return 0;
    }
};

#ifdef __clang__
#pragma clang diagnostic pop
#endif

}  // namespace deep_gemm
