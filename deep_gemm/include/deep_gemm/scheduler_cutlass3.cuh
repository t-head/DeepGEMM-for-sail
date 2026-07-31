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

#define EnableGroupNoPadOpt

// namespace cutlass::gemm::kernel {
namespace deep_gemm {
using cutlass::KernelHardwareInfo;
struct TileSchedulerArguments
{
    int* grouped_layout;
    uint32_t shape_m;

    //
    // Methods
    //

    /// Ctor
    CUTLASS_HOST_DEVICE
    TileSchedulerArguments()
        : grouped_layout(nullptr)
        , shape_m(0)
    {
    }

    /// Ctor
    CUTLASS_HOST_DEVICE
    TileSchedulerArguments(uint32_t shape_m, int* grouped_layout_ptr = nullptr)
        : grouped_layout(grouped_layout_ptr)
        , shape_m(shape_m)
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
          uint32_t kNumNBlocks = ceil_div(SHAPE_N_, BLOCK_N_),
          uint32_t kNum1DBlocksPerGroup = 2>
struct DeepGemmScheduler {
    constexpr static uint32_t SHAPE_N = SHAPE_N_;
    constexpr static uint32_t SHAPE_K = SHAPE_K_;
    constexpr static uint32_t BLOCK_M = BLOCK_M_;
    constexpr static uint32_t BLOCK_N = BLOCK_N_;
    constexpr static uint32_t kNumGroups = kNumGroups_;
    int current_iter = 0;
    uint32_t num_aligned_m_blocks;
    constexpr static GemmType GEMM_TYPE = kGemmType;
    constexpr static bool kIsTMAMulticastOnA = false;
#ifdef EnableGroupNoPadOpt
    constexpr static bool kIsNoPadPreprocessLayout = (kGemmType == GemmType::GroupedNoPad||kGemmType == GemmType::GroupedFused) && kNumGroups >= 128;
#else
    constexpr static bool kIsNoPadPreprocessLayout = false;
#endif

    // For normal GEMM
    // Maybe not used in the masked grouped GEMM
    uint32_t num_blocks;
    uint32_t num_n_blocks = kNumNBlocks;

    // Only used for masked layout
    uint32_t curr_group_idx, curr_cumsum, curr_cumsum_blocks, curr_group_m, curr_cumsum_m;
    // Host-callable methods and type definitions
    using Arguments = TileSchedulerArguments;
    using Params = TileSchedulerParams;
    Params const& params;

#if defined(__HGGC__)
    // --- Device-only methods below (guarded from host compilation) ---

    CUTLASS_DEVICE explicit DeepGemmScheduler(Params const& params_, const int warp_group_id = 0) : params(params_), current_iter(warp_group_id) {
        num_aligned_m_blocks = ceil_div(params_.shape_m, BLOCK_M);
        if constexpr(kGemmType == GemmType::DenseGemm || kGemmType == GemmType::BatchGemm) {
            num_blocks = num_aligned_m_blocks * num_n_blocks;
        } else if constexpr(kGemmType == GemmType::GroupedContiguous) {
            num_blocks = num_aligned_m_blocks * num_n_blocks;
        } else if constexpr(kGemmType == GemmType::GroupedMasked) {
            curr_group_idx = curr_cumsum = curr_group_m = curr_cumsum_blocks = curr_cumsum_m = 0;
        } else if constexpr(kGemmType == GemmType::GroupedNoPad || kGemmType == GemmType::GroupedFused) {
            if constexpr(kIsNoPadPreprocessLayout) {
                num_aligned_m_blocks = params_.grouped_layout[0]; // total blocks in m, block_m_sum
                curr_group_idx = curr_cumsum = curr_cumsum_blocks = curr_group_m = curr_cumsum_m = 0;
                num_blocks = num_aligned_m_blocks * num_n_blocks;
            } else {
                curr_group_idx = curr_cumsum = curr_cumsum_blocks = curr_group_m = curr_cumsum_m = 0;
            }
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
        if constexpr(kIsNoPadPreprocessLayout) {
            if (next_block_idx >= num_blocks) {
                m_block_idx = num_aligned_m_blocks;
                n_block_idx = kNumNBlocks;
                return false;
            }
            int block_m_idx = next_block_idx / kNumNBlocks;
            uint4 data = (((const uint4*)params.grouped_layout) + 1)[block_m_idx];
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
            if constexpr(kGemmType == GemmType::GroupedFused) {
                m_block_idx += curr_cumsum;
            }
        } else if constexpr (kGemmType == GemmType::BatchGemm) {
            if (next_block_idx >= num_blocks * kNumGroups)
                return false;

            curr_group_idx = next_block_idx / num_blocks;
            const auto& block_idx = next_block_idx - curr_group_idx * num_blocks;
            m_block_idx = block_idx / kNumNBlocks;
            n_block_idx = block_idx % kNumNBlocks;
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
        const auto next_block_idx = (current_iter++) * gridDim.x + blockIdx.x;
        if (kIsNoPadPreprocessLayout) {
            if (next_block_idx >= num_blocks) {
                m_block_idx = num_aligned_m_blocks;
                n_block_idx = kNumNBlocks;
                return false;
            }
            int block_m_idx = next_block_idx / kNumNBlocks;
            uint4 data = (((const uint4*)params.grouped_layout) + 1)[block_m_idx];
            curr_group_idx = data.x;
            curr_group_m = data.y;
            uint32_t block_idx_in_m = data.z * kNumNBlocks + next_block_idx % kNumNBlocks;
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
                curr_group_m = static_cast<uint32_t>(__ldg(params.grouped_layout + curr_group_idx));
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
#endif  // defined(__HGGC__)

    // Host-callable: to_underlying_arguments and get_workspace_size
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
        // We only need the tile and cluster shape during scheduler setup, so let FTAD do the magic
        static_assert(cute::is_static<TileShape>::value);
        static_assert(cute::is_static<ClusterShape>::value);

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

#if defined(__HGGC__)
    // --- Device-only accessor methods below ---

    // Returns the problem size for the current problem
    __device__ __forceinline__ int32_t curr_problem_m() const
    {
        if constexpr (kGemmType == GemmType::DenseGemm || kGemmType == GemmType::BatchGemm || kGemmType == GemmType::GroupedContiguous) {
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
        if constexpr (kGemmType == GemmType::GroupedMasked || kGemmType == GemmType::BatchGemm) {
            return int64_t(curr_group_idx) * params.shape_m * SHAPE_K;
        } else if constexpr (kGemmType == GemmType::GroupedNoPad) {
            return int64_t(curr_cumsum_m) * SHAPE_K;
        } else {
            return 0;
        }
    }

    __device__ __forceinline__ int64_t curr_offset_scalea() const
    {
        if constexpr (kGemmType == GemmType::GroupedMasked || kGemmType == GemmType::BatchGemm) {
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
        if constexpr (kGemmType == GemmType::GroupedMasked || kGemmType == GemmType::BatchGemm) {
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
        } else if constexpr(kGemmType == GemmType::BatchGemm) {
            return int64_t(curr_group_idx) * SHAPE_N;
        } else {
            return 0;
        }
    }
#endif  // defined(__HGGC__)

};

// ==================== Dynamic Tile Builder Config & Scheduler ====================

/// Configuration for dynamic tile builder selection.
/// Describes each builder's BLOCK_M and BLOCK_N, and provides builder selection logic.
template <int BM0, int BM1, int BM2, int BM3, int BM4,
          int BN0, int BN1, int BN2, int BN3, int BN4>
struct DynamicTileBuilderConfig {
    static constexpr int kNumBuilders = 5;
    static constexpr int BlockM[kNumBuilders] = {BM0, BM1, BM2, BM3, BM4};
    static constexpr int BlockN[kNumBuilders] = {BN0, BN1, BN2, BN3, BN4};

#if defined(__HGGC__)
    // Select builder: pick the first builder whose BLOCK_M >= group_m.
    // If group_m exceeds all builders, use the last (largest) builder.
    static CUTLASS_DEVICE int select_builder(int group_m) {
        if (group_m <= BlockM[0]) return 0;
        if (group_m <= BlockM[1]) return 1;
        if (group_m <= BlockM[2]) return 2;
        if (group_m <= BlockM[3]) return 3;
        return kNumBuilders - 1;
    }

    static CUTLASS_DEVICE int get_block_m(int idx) { return BlockM[idx]; }
    static CUTLASS_DEVICE int get_block_n(int idx) { return BlockN[idx]; }
#endif  // defined(__HGGC__)
};

/// Dynamic tile scheduler that selects builder per group and distributes tiles
/// at the selected builder's granularity.
/// Inherits from DeepGemmScheduler to reuse offset helpers, params, and swizzle logic.
template <GemmType kGemmType, int SHAPE_N_, int SHAPE_K_,
          typename BuilderConfig_, int kNumGroups_, int kNum1DBlocksPerGroup = 2>
struct DynamicTileScheduler
    : public DeepGemmScheduler<kGemmType, SHAPE_N_, SHAPE_K_,
                               BuilderConfig_::BlockM[BuilderConfig_::kNumBuilders - 1],
                               BuilderConfig_::BlockN[BuilderConfig_::kNumBuilders - 1],
                               kNumGroups_,
                               ceil_div((uint32_t)SHAPE_N_, (uint32_t)BuilderConfig_::BlockN[BuilderConfig_::kNumBuilders - 1]),
                               kNum1DBlocksPerGroup> {
    using Base = DeepGemmScheduler<kGemmType, SHAPE_N_, SHAPE_K_,
                                    BuilderConfig_::BlockM[BuilderConfig_::kNumBuilders - 1],
                                    BuilderConfig_::BlockN[BuilderConfig_::kNumBuilders - 1],
                                    kNumGroups_,
                                    ceil_div((uint32_t)SHAPE_N_, (uint32_t)BuilderConfig_::BlockN[BuilderConfig_::kNumBuilders - 1]),
                                    kNum1DBlocksPerGroup>;
    using BuilderConfig = BuilderConfig_;
    using Params = typename Base::Params;

    static constexpr bool kIsDynamicTile = true;
    // Override base: dynamic tile does not support NoPadPreprocess layout
    static constexpr bool kIsNoPadPreprocessLayout = false;

    // Dynamic tile specific members
    int curr_builder_idx;
    uint32_t curr_num_blocks;
    uint32_t curr_n_expanded;

#if defined(__HGGC__)
    CUTLASS_DEVICE
    DynamicTileScheduler(Params const& params_)
        : Base(params_),
          curr_builder_idx(0),
          curr_num_blocks(0),
          curr_n_expanded(0) {}

    /// Fetch next tile with builder selection.
    /// Unlike base class fetch_next_work, also outputs builder_idx.
    CUTLASS_DEVICE
    bool fetch_next_work(uint32_t& m_block_idx, uint32_t& n_block_idx, int& builder_idx) {
        const int next_block_idx = (this->current_iter++) * gridDim.x + blockIdx.x;

        if constexpr (kGemmType == GemmType::GroupedMasked || kGemmType == GemmType::GroupedNoPad) {
            uint32_t num_m_blocks;
            int n_expand = 1;
            int curr_cumsum_blocks_prev;
            while (true) {
                if (this->curr_group_idx == (uint32_t)Base::kNumGroups)
                    return false;

                // Read group M and select builder
                this->curr_group_m = static_cast<uint32_t>(__ldg(this->params.grouped_layout + this->curr_group_idx));
                curr_builder_idx = BuilderConfig::select_builder((int)this->curr_group_m);
                int builder_block_m = BuilderConfig::get_block_m(curr_builder_idx);
                int builder_block_n = BuilderConfig::get_block_n(curr_builder_idx);

                num_m_blocks = ceil_div(this->curr_group_m, (uint32_t)builder_block_m);
                int num_n_blocks = ceil_div((uint32_t)SHAPE_N_, (uint32_t)builder_block_n);
                curr_num_blocks = num_m_blocks * num_n_blocks;

                curr_cumsum_blocks_prev = this->curr_cumsum_blocks;
                int curr_cumsum_blocks_next = this->curr_cumsum_blocks + curr_num_blocks;

                if (next_block_idx < curr_cumsum_blocks_next)
                    break;

                // Move to next group
                this->curr_cumsum_blocks = curr_cumsum_blocks_next;
                this->curr_group_idx++;
                this->curr_cumsum += num_m_blocks;
                this->curr_cumsum_m += this->curr_group_m;
            }

            this->get_swizzled_block_idx(num_m_blocks, next_block_idx - curr_cumsum_blocks_prev,
                                         m_block_idx, n_block_idx, n_expand);
            builder_idx = curr_builder_idx;
        } else {
            // Fallback for non-grouped types
            if (next_block_idx >= this->num_blocks)
                return false;
            this->get_swizzled_block_idx(this->num_aligned_m_blocks, next_block_idx,
                                         m_block_idx, n_block_idx);
            builder_idx = BuilderConfig::kNumBuilders - 1;
        }
        return true;
    }
#endif  // defined(__HGGC__)
};

#pragma clang diagnostic pop

}

