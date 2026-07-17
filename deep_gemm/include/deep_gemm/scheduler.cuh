#pragma once

#include "utils_rtc.cuh"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/fast_math.h"
#define EnableGroupNoPadOpt
namespace deep_gemm {


#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
/////////////////////////////////////////////////////////////////////////////////////////////////
// ProblemVisitor that same as deepgemm
//
template <GemmType GemmType_, uint32_t SHAPE_N, typename ThreadblockShape_, uint32_t kNumGroups_, uint32_t kNum1DBlocksPerGroup = 16>
struct Scheduler
{
    /// use for base_group compatiblity
    struct ProblemInfo
    {
        static int32_t const kNoPrefetchEntry = -1;
        int32_t problem_idx;
        int32_t problem_start;

        CUTLASS_DEVICE
        ProblemInfo()
            : problem_idx(kNoPrefetchEntry)
            , problem_start(kNoPrefetchEntry)
        {
        }

        CUTLASS_DEVICE
        ProblemInfo(int32_t problem_idx_, int32_t problem_start_)
            : problem_idx(problem_idx_)
            , problem_start(problem_start_)
        {
        }
    };

    using ThreadblockShape = ThreadblockShape_;

    static const GemmType kGemmType = GemmType_;
#ifdef EnableGroupNoPadOpt
    constexpr static bool kIsNoPadPreprocessLayout = kGemmType == GemmType::GroupedNoPad and kNumGroups_ >= 128;
#else
    constexpr static bool kIsNoPadPreprocessLayout = false;
#endif

    struct Params
    {
        int const* grouped_layout;
        int64_t gemm_n;
        int64_t gemm_k;
        int64_t gemm_m;
        int32_t problem_count;

        //
        // Methods
        //

        /// Ctor
        CUTLASS_HOST_DEVICE
        Params()
            : grouped_layout(nullptr)
            , gemm_n(0)
            , gemm_k(0)
            , gemm_m(0)
            , problem_count(0)
        {
        }

        /// Ctor
        CUTLASS_HOST_DEVICE
        Params(int64_t gemm_m, int64_t gemm_n, int64_t gemm_k, int const* groups_layout, int32_t num_groups)
            : grouped_layout(groups_layout)
            , gemm_n(gemm_n)
            , gemm_k(gemm_k)
            , gemm_m(gemm_m)
            , problem_count(num_groups)
        {
        }
    };

    // using Params = typename Params;
    Params const& params;
    static bool const kRequiresPrecomputation = false;

    struct SharedStorage
    {
    };

    SharedStorage& shared_storage;

    // Only used for masked layout
    uint32_t curr_group_idx, curr_cumsum, curr_m_start, curr_m;
    int curr_group_m;

    int current_iter = 0;
    uint32_t num_aligned_m_blocks;

    uint32_t num_blocks;

    uint32_t num_n_blocks;

    CUTLASS_DEVICE
    Scheduler(Params const& params_, SharedStorage& shared_storage_, int32_t block_idx)
        : params(params_)
        , shared_storage(shared_storage_)
    {
        num_aligned_m_blocks = cutlass::ceil_div(params_.gemm_m, ThreadblockShape::kM);
        num_n_blocks = cutlass::ceil_div(params_.gemm_n, ThreadblockShape::kN);

        curr_group_idx = 0;
        if constexpr (kGemmType == GemmType::DenseGemm || kGemmType == GemmType::BatchGemm) {
            num_blocks = num_aligned_m_blocks * num_n_blocks;
        } else if (kGemmType == GemmType::GroupedContiguous) {
            num_blocks = num_aligned_m_blocks * num_n_blocks;
        } else if (kGemmType == GemmType::GroupedMasked) {
            curr_cumsum = curr_m_start = curr_m = 0;
        } else if (kGemmType == GemmType::GroupedNoPad) {
            if (kIsNoPadPreprocessLayout) {
                num_aligned_m_blocks = params_.grouped_layout[0]; // total blocks in m, block_m_sum
                curr_cumsum = curr_group_m = curr_m = 0;
                num_blocks = num_aligned_m_blocks * num_n_blocks;
            } else {
                curr_cumsum = curr_m_start = curr_m = curr_group_m = 0;
            }
        }
    }

    CUTLASS_DEVICE uint32_t get_curr_m() {
        return curr_m;
    }

    template <bool kIgnoreGroupedForGroupedContiguous=true>
    CUTLASS_DEVICE uint32_t get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
                                                    const uint32_t& block_idx, const uint32_t& m_block_idx=0) {
        if constexpr (kGemmType == GemmType::DenseGemm || kGemmType == GemmType::BatchGemm) {
            return block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::GroupedContiguous) {
            auto offset = kIgnoreGroupedForGroupedContiguous ? 0 : __ldg(params.grouped_layout + m_block_idx * ThreadblockShape::kM);
            return offset * shape_dim + block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::GroupedMasked) {
            return curr_group_idx * shape_dim + block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::GroupedNoPad) {
            return (kIgnoreGroupedForGroupedContiguous ?
                curr_m_start : curr_group_idx * shape_dim) + block_idx * block_size;
        }
    }

    CUTLASS_DEVICE void get_swizzled_block_idx(const uint32_t num_m_blocks, int block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
        // Swizzle for better L2 usages
        // TODO: unify these 2 branches
        if constexpr (true) {
            auto num_blocks_per_group = num_m_blocks * kNum1DBlocksPerGroup;
            auto group_idx = block_idx / num_blocks_per_group;
            auto first_n_block_idx = group_idx * kNum1DBlocksPerGroup;
            auto num_n_blocks_in_group = min(kNum1DBlocksPerGroup, num_n_blocks - first_n_block_idx);
            auto in_group_idx = block_idx % num_blocks_per_group;
            m_block_idx = in_group_idx / num_n_blocks_in_group;
            n_block_idx = first_n_block_idx + in_group_idx % num_n_blocks_in_group;
        } else {
            auto num_blocks_per_group = num_n_blocks * kNum1DBlocksPerGroup;
            auto group_idx = block_idx / num_blocks_per_group;
            auto first_m_block_idx = group_idx * kNum1DBlocksPerGroup;
            auto num_m_blocks_in_group = min(kNum1DBlocksPerGroup, num_m_blocks - first_m_block_idx);
            auto in_group_idx = block_idx % num_blocks_per_group;
            m_block_idx = first_m_block_idx + in_group_idx % num_m_blocks_in_group;
            n_block_idx = in_group_idx / num_m_blocks_in_group;
        }
    }

    CUTLASS_DEVICE
    bool next_tile(uint32_t& m_block_idx, uint32_t& n_block_idx)
    {
        const auto next_block_idx = (current_iter++) * gridDim.x + blockIdx.x;

        if (kIsNoPadPreprocessLayout) {
            if (next_block_idx >= num_blocks) {
                m_block_idx = num_aligned_m_blocks;
                n_block_idx = num_n_blocks;
                return false;
            }
            int block_m_idx = next_block_idx / num_n_blocks;
            uint4 data = (((const uint4*)params.grouped_layout) + 1)[block_m_idx];
            curr_group_idx = data.x;
            curr_group_m = data.y;
            uint32_t block_idx_in_m = next_block_idx - data.z * num_n_blocks;
            uint32_t num_m_blocks = ceil_div(curr_group_m, ThreadblockShape::kM);
            curr_m_start = data.w;
            get_swizzled_block_idx(num_m_blocks, block_idx_in_m, m_block_idx, n_block_idx);
            curr_m = (curr_group_m - (m_block_idx * ThreadblockShape::kM)) < ThreadblockShape::kM
                     ? (curr_group_m - (m_block_idx * ThreadblockShape::kM))
                     : ThreadblockShape::kM;
        } else if (kGemmType == GemmType::GroupedMasked || kGemmType == GemmType::GroupedNoPad) {
            uint32_t num_m_blocks;
            while (true) {
                // End of the task
                if (curr_group_idx == params.problem_count) {
                    m_block_idx = num_m_blocks;
                    n_block_idx = num_n_blocks;
                    return false;
                }
                // Within the current group
                curr_group_m = static_cast<uint32_t>(__ldg(params.grouped_layout + curr_group_idx));
                num_m_blocks = ceil_div(curr_group_m, ThreadblockShape::kM);
                auto current_m_block_cumsum = curr_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * num_n_blocks)
                    break;
                // Move to check the next group
                curr_group_idx ++, curr_cumsum = current_m_block_cumsum;
                curr_m_start += curr_group_m;
            }
            get_swizzled_block_idx(num_m_blocks, next_block_idx - curr_cumsum * num_n_blocks, m_block_idx, n_block_idx);
            curr_m = (curr_group_m - (m_block_idx * ThreadblockShape::kM)) < ThreadblockShape::kM
                     ? (curr_group_m - (m_block_idx * ThreadblockShape::kM))
                     : ThreadblockShape::kM;
        } else if constexpr (kGemmType == GemmType::BatchGemm) {
            if (next_block_idx >= num_blocks * params.problem_count)
                return false;
            curr_group_idx = next_block_idx / num_blocks;
            const auto& in_group_block_idx = next_block_idx - curr_group_idx * num_blocks;
            get_swizzled_block_idx(num_aligned_m_blocks, in_group_block_idx, m_block_idx, n_block_idx);
        } else {
            if (next_block_idx >= num_blocks)
                return false;

            get_swizzled_block_idx(num_aligned_m_blocks, next_block_idx, m_block_idx, n_block_idx);

            if (kGemmType == GemmType::GroupedContiguous)
                curr_group_idx = __ldg(params.grouped_layout + m_block_idx * ThreadblockShape::kM);
        }

        return true;
    }

    static size_t get_workspace_size(
        const cutlass::gemm::GemmCoord* host_problem_sizes_ptr, int32_t problem_count, int32_t block_count)
    {
        return 0;
    }

    static void host_precompute(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr, int32_t problem_count,
        int32_t block_count, void* host_workspace_ptr)
    {
    }

    /// Returns the problem size for the current problem
    CUTLASS_HOST_DEVICE
    cutlass::gemm::GemmCoord problem_size() const
    {
        cutlass::gemm::GemmCoord problem(cutlass::gemm::GemmCoord::Index(params.gemm_m), cutlass::gemm::GemmCoord::Index(params.gemm_n), cutlass::gemm::GemmCoord::Index(params.gemm_k));
        return problem;
    }

    /// Gets the index of the problem
    CUTLASS_HOST_DEVICE
    int32_t problem_index() const
    {
        return curr_group_idx;
    }

    /// use for base_group compatiblity
    CUTLASS_HOST_DEVICE
    static cutlass::gemm::GemmCoord grid_shape(const cutlass::gemm::GemmCoord& problem)
    {
        return cutlass::gemm::GemmCoord(((problem.m() - 1 + ThreadblockShape::kM) / ThreadblockShape::kM),
            ((problem.n() - 1 + ThreadblockShape::kN) / ThreadblockShape::kN), 1);
    }

    CUTLASS_HOST_DEVICE
    int32_t threadblock_idx() const
    {
        return 0;
    }

    CUTLASS_HOST_DEVICE
    static void possibly_transpose_problem(cutlass::gemm::GemmCoord& problem)
    {
    }

    CUTLASS_HOST_DEVICE
    static int32_t tile_count(const cutlass::gemm::GemmCoord& grid)
    {
        return grid.m() * grid.n();
    }
};


#pragma clang diagnostic pop

} // namespace deep_gemm
