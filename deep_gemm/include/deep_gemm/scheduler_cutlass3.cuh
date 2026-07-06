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
    // FusedDispatchMasked schedules from grouped_layout (= masked_m), while
    // copy blocks still consume the block-granular NoPad metadata below.
    int* copy_grouped_layout;
    uint32_t shape_m;

    // Block-copy fused dispatch: per-(M-block × rank) split metadata
    const uint64_t* rank_addr_a;
    const uint64_t* rank_addr_sfa;
    const uint32_t* rank_split_m;
    const uint32_t* rank_counts;
    uint32_t num_ranks;
    const uint64_t* remote_addr_sfa;

    // Local HBM staging buffer for block-copy
    uint8_t* local_fp4_buf;
    uint32_t local_buf_k_half;
    uint32_t local_buf_max_tokens;
    // Local HBM staging buffer for GPU-side SFA (scale) copy/repack. The copy
    // blocks repack each rank's column-major scales into per-expert layout
    // [k_scale_blocks, max_tokens] (K-stride = local_buf_max_tokens) so the GEMM
    // reads this rank's SFA directly, without a host-built merged_sfa.
    uint16_t* local_sfa_buf;
    uint32_t local_buf_k_scale_blocks;

    // Dedicated copy blocks
    volatile uint32_t* copy_ready_flags;
    uint32_t num_copy_blocks;

    // Optional per-tile k-stripe wait profiling buffer (nullptr => off).
    // Indexed by tile_idx (= next_block_idx); 4 int64 per tile:
    // [0]=stripe-wait cycles, [1]=mainloop cycles, [2]=wave, [3]=gemm CTA id.
    // kstripe_profile_max_tiles = capacity (numel/4); writes past it are skipped.
    uint64_t* kstripe_profile_buf;
    uint32_t kstripe_profile_max_tiles;

    // Copy strategy: 0 = static round-robin (each block owns whole M-blocks),
    // 1 = cooperative (all blocks co-copy each M-block in order). KTPF=0 only.
    uint32_t copy_mode;

    //
    // Methods
    //

    /// Ctor
    CUTLASS_HOST_DEVICE
    TileSchedulerArguments()
        : grouped_layout(nullptr)
        , copy_grouped_layout(nullptr)
        , shape_m(0)
        , rank_addr_a(nullptr)
        , rank_addr_sfa(nullptr)
        , rank_split_m(nullptr)
        , rank_counts(nullptr)
        , num_ranks(0)
        , remote_addr_sfa(nullptr)
        , local_fp4_buf(nullptr)
        , local_buf_k_half(0)
        , local_buf_max_tokens(0)
        , local_sfa_buf(nullptr)
        , local_buf_k_scale_blocks(0)
        , copy_ready_flags(nullptr)
        , num_copy_blocks(0)
        , kstripe_profile_buf(nullptr)
        , kstripe_profile_max_tiles(0)
        , copy_mode(0)
    {
    }

    /// Ctor (non-fused)
    CUTLASS_HOST_DEVICE
    TileSchedulerArguments(uint32_t shape_m, int* grouped_layout_ptr = nullptr)
        : grouped_layout(grouped_layout_ptr)
        , copy_grouped_layout(nullptr)
        , shape_m(shape_m)
        , rank_addr_a(nullptr)
        , rank_addr_sfa(nullptr)
        , rank_split_m(nullptr)
        , rank_counts(nullptr)
        , num_ranks(0)
        , remote_addr_sfa(nullptr)
        , local_fp4_buf(nullptr)
        , local_buf_k_half(0)
        , local_buf_max_tokens(0)
        , local_sfa_buf(nullptr)
        , local_buf_k_scale_blocks(0)
        , copy_ready_flags(nullptr)
        , num_copy_blocks(0)
        , kstripe_profile_buf(nullptr)
        , kstripe_profile_max_tiles(0)
        , copy_mode(0)
    {
    }

    /// Ctor (block-copy fused dispatch)
    CUTLASS_HOST_DEVICE
    TileSchedulerArguments(uint32_t shape_m, int* grouped_layout_ptr,
                           int* copy_grouped_layout_ptr,
                           const uint64_t* rank_addr_a_, const uint64_t* rank_addr_sfa_,
                           const uint32_t* rank_split_m_, const uint32_t* rank_counts_,
                           uint32_t num_ranks_,
                           const uint64_t* remote_addr_sfa_,
                           uint8_t* local_fp4_buf_,
                           uint32_t local_buf_k_half_,
                           uint32_t local_buf_max_tokens_,
                           uint16_t* local_sfa_buf_,
                           uint32_t local_buf_k_scale_blocks_,
                           volatile uint32_t* copy_ready_flags_,
                           uint32_t num_copy_blocks_,
                           uint64_t* kstripe_profile_buf_ = nullptr,
                           uint32_t kstripe_profile_max_tiles_ = 0,
                           uint32_t copy_mode_ = 0)
        : grouped_layout(grouped_layout_ptr)
        , copy_grouped_layout(copy_grouped_layout_ptr)
        , shape_m(shape_m)
        , rank_addr_a(rank_addr_a_)
        , rank_addr_sfa(rank_addr_sfa_)
        , rank_split_m(rank_split_m_)
        , rank_counts(rank_counts_)
        , num_ranks(num_ranks_)
        , remote_addr_sfa(remote_addr_sfa_)
        , local_fp4_buf(local_fp4_buf_)
        , local_buf_k_half(local_buf_k_half_)
        , local_buf_max_tokens(local_buf_max_tokens_)
        , local_sfa_buf(local_sfa_buf_)
        , local_buf_k_scale_blocks(local_buf_k_scale_blocks_)
        , copy_ready_flags(copy_ready_flags_)
        , num_copy_blocks(num_copy_blocks_)
        , kstripe_profile_buf(kstripe_profile_buf_)
        , kstripe_profile_max_tiles(kstripe_profile_max_tiles_)
        , copy_mode(copy_mode_)
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
          uint32_t kNum1DBlocksPerGroup_ = 2,
          uint32_t kNumRanks_ = 1,
          uint32_t kNumCopyBlocks_ = 0,
          uint32_t kKTilesPerFlag_ = 0>
struct DeepGemmScheduler {
    constexpr static uint32_t SHAPE_N = SHAPE_N_;
    constexpr static uint32_t SHAPE_K = SHAPE_K_;
    constexpr static uint32_t BLOCK_M = BLOCK_M_;
    constexpr static uint32_t BLOCK_N = BLOCK_N_;
    constexpr static uint32_t kNumGroups = kNumGroups_;
    constexpr static uint32_t kNum1DBlocksPerGroup = kNum1DBlocksPerGroup_;
    constexpr static uint32_t kNumRanks = kNumRanks_;
    constexpr static uint32_t kNumCopyBlocks = kNumCopyBlocks_;
    constexpr static uint32_t kKTilesPerFlag = kKTilesPerFlag_;
    int current_iter = 0;
    uint32_t num_aligned_m_blocks;
    constexpr static GemmType GEMM_TYPE = kGemmType;
    constexpr static bool kIsFusedDispatch =
        kGemmType == GemmType::FusedDispatch || kGemmType == GemmType::FusedDispatchMasked;
    constexpr static bool kIsMaskedLayout =
        kGemmType == GemmType::GroupedMasked || kGemmType == GemmType::FusedDispatchMasked;
    constexpr static bool kIsTMAMulticastOnA = false;
#ifdef EnableGroupNoPadOpt
    constexpr static bool kIsNoPadPreprocessLayout = ((kGemmType == GemmType::GroupedNoPad||kGemmType == GemmType::GroupedFused) && kNumGroups >= 128) || kGemmType == GemmType::FusedDispatch;
#else
    constexpr static bool kIsNoPadPreprocessLayout = false;
#endif

    // For normal GEMM
    // Maybe not used in the masked grouped GEMM
    uint32_t num_blocks;
    uint32_t num_n_blocks = kNumNBlocks;

    // Only used for masked layout
    uint32_t curr_group_idx, curr_cumsum, curr_cumsum_blocks, curr_group_m, curr_cumsum_m;
    uint32_t curr_global_block_m_idx;
    // Per-tile profiling identity (set in fetch_next_work): linear tile index
    // (= next_block_idx), persistent-loop wave number (= current_iter), and
    // the GEMM CTA id (= eff_bidx = blockIdx.x - kNumCopyBlocks).
    uint32_t curr_tile_idx;
    uint32_t curr_wave;
    uint32_t curr_cta_id;
    using Arguments = TileSchedulerArguments;
    using Params = TileSchedulerParams;
    Params const& params;

    CUTLASS_DEVICE explicit DeepGemmScheduler(Params const& params_, const int warp_group_id = 0) : params(params_), current_iter(warp_group_id) {
        num_aligned_m_blocks = ceil_div(params_.shape_m, BLOCK_M);
        if constexpr(kGemmType == GemmType::DenseGemm || kGemmType == GemmType::BatchGemm) {
            num_blocks = num_aligned_m_blocks * num_n_blocks;
        } else if constexpr(kGemmType == GemmType::GroupedContiguous) {
            num_blocks = num_aligned_m_blocks * num_n_blocks;
        } else if constexpr(kIsMaskedLayout) {
            curr_group_idx = curr_cumsum = curr_group_m = curr_cumsum_blocks = curr_cumsum_m = 0;
        } else if constexpr(kGemmType == GemmType::GroupedNoPad || kGemmType == GemmType::GroupedFused || kGemmType == GemmType::FusedDispatch) {
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
        } else if (kIsMaskedLayout) {
            return curr_group_idx * shape_dim + block_idx * block_size;
        }
    }

    CUTLASS_DEVICE bool fetch_next_work(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        uint32_t next_block_idx;
        {
            uint32_t eff_grid = gridDim.x;
            uint32_t eff_bidx = blockIdx.x;
            if constexpr (kNumCopyBlocks > 0) {
                eff_grid = gridDim.x - kNumCopyBlocks;
                eff_bidx = blockIdx.x - kNumCopyBlocks;
            }
            next_block_idx = (current_iter++) * eff_grid + eff_bidx;
            // Profiling identity for this work tile (harmless when profiling off).
            curr_tile_idx = next_block_idx;
            curr_wave = (uint32_t)(current_iter - 1);
            curr_cta_id = eff_bidx;
        }
        if constexpr(kIsNoPadPreprocessLayout) {
            if (next_block_idx >= num_blocks) {
                m_block_idx = num_aligned_m_blocks;
                n_block_idx = kNumNBlocks;
                return false;
            }
            if constexpr(kGemmType == GemmType::FusedDispatch) {
                // M-major without swizzle: matches copy block order (M-block 0 copied first)
                // Swizzle disabled: prevents cross-M-block mapping that causes copy_ready_flags race
                int block_m_idx = next_block_idx / kNumNBlocks;
                n_block_idx = next_block_idx % kNumNBlocks;
                curr_global_block_m_idx = block_m_idx;
                uint4 data = (((const uint4*)params.grouped_layout) + 1)[block_m_idx];
                curr_group_idx = data.x;
                curr_group_m = data.y;
                curr_cumsum_m = data.w;
                m_block_idx = block_m_idx - data.z;
            } else {
            int block_m_idx = next_block_idx / kNumNBlocks;
            curr_global_block_m_idx = block_m_idx;
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
            }
        } else if constexpr(kIsMaskedLayout || kGemmType == GemmType::GroupedNoPad || kGemmType == GemmType::GroupedFused || kGemmType == GemmType::FusedDispatch) {
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
            if constexpr(kGemmType == GemmType::FusedDispatchMasked) {
                // masked_m is laid out as [counts[kNumGroups], block_offsets[kNumGroups]].
                curr_global_block_m_idx = static_cast<uint32_t>(__ldg(
                    params.grouped_layout + kNumGroups + curr_group_idx)) + m_block_idx;
            }
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
        uint32_t eff_grid = gridDim.x;
        uint32_t eff_bidx = blockIdx.x;
        if constexpr (kNumCopyBlocks > 0) {
            eff_grid = gridDim.x - kNumCopyBlocks;
            eff_bidx = blockIdx.x - kNumCopyBlocks;
        }
        const auto next_block_idx = (current_iter++) * eff_grid + eff_bidx;
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
        } else if (kIsMaskedLayout || kGemmType == GemmType::GroupedNoPad) {
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
        Params params = arguments;
        params.grouped_layout = groups_layout;
        params.shape_m = static_cast<uint32_t>(problem_shape.m());
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
        if constexpr (kGemmType == GemmType::DenseGemm || kGemmType == GemmType::BatchGemm || kGemmType == GemmType::GroupedContiguous) {
            return params.shape_m;
        } else if constexpr (kIsMaskedLayout) {
            return curr_group_m;
        } else if constexpr (kGemmType == GemmType::GroupedNoPad || kGemmType == GemmType::FusedDispatch) {
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
        if constexpr (kIsMaskedLayout || kGemmType == GemmType::BatchGemm) {
            return int64_t(curr_group_idx) * params.shape_m * SHAPE_K;
        } else if constexpr (kGemmType == GemmType::GroupedNoPad) {
            return int64_t(curr_cumsum_m) * SHAPE_K;
        } else {
            return 0;
        }
    }

    __device__ __forceinline__ int64_t curr_offset_scalea() const
    {
        if constexpr (kIsMaskedLayout || kGemmType == GemmType::BatchGemm) {
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
        } else if constexpr (kIsMaskedLayout) {
            return int64_t(curr_group_idx) * params.shape_m * shape_k_scale;
        } else {
            return 0;
        }
    }

    /// Gets the pointer offset of matrix A
    __device__ __forceinline__ int64_t curr_offset_m() const
    {
        if constexpr (kIsMaskedLayout || kGemmType == GemmType::BatchGemm) {
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
        if constexpr (kGemmType == GemmType::GroupedNoPad || kIsMaskedLayout || kGemmType == GemmType::FusedDispatch) {
            return int64_t(curr_group_idx) * SHAPE_N;
        } else {
            return 0;
        }
    }

    __device__ __forceinline__ int64_t curr_offset_c() const
    {
        if constexpr (kIsMaskedLayout || kGemmType == GemmType::GroupedContiguous) {
            return int64_t(curr_group_idx) * params.shape_m * SHAPE_N;
        } else if constexpr (kGemmType == GemmType::GroupedNoPad || kGemmType == GemmType::FusedDispatch) {
            return int64_t(curr_cumsum_m) * SHAPE_N;
        } else if constexpr(kGemmType == GemmType::BatchGemm) {
            return int64_t(curr_group_idx) * SHAPE_N;
        } else {
            return 0;
        }
    }
};

#pragma clang diagnostic pop

}
