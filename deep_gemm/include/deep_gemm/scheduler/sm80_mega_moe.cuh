#pragma once

#include <type_traits>
#include <deep_gemm/scheduler/mega_moe_common.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>

#if defined(__HGGC__)
#include <deep_gemm/comm/barrier.cuh>
#endif

namespace deep_gemm::sched {

// Two-stage PPU scheduler. Keep the GEMM1 tile queue and scatter traversal
// separate from the ordinary grouped GEMM scheduler.
template <uint32_t SHAPE_N_, uint32_t SHAPE_K_,
          uint32_t BLOCK_M_, uint32_t BLOCK_N_, uint32_t kNumGroups_,
          MegaMoEPhaseKind Stage, uint32_t kNumRanks_,
          uint32_t kKTilesPerFlag_ = 0,
          uint32_t kSourceCapacity_ = 1, uint32_t kNumTopK_ = 1,
          class Dispatch_ = void>
struct SM80MegaMoEScheduler {
    using Dispatch = Dispatch_;
    static constexpr uint32_t SHAPE_N = SHAPE_N_;
    static constexpr uint32_t SHAPE_K = SHAPE_K_;
    static constexpr uint32_t BLOCK_M = BLOCK_M_;
    static constexpr uint32_t BLOCK_N = BLOCK_N_;
    static constexpr uint32_t kNumGroups = kNumGroups_;
    static constexpr uint32_t kNumRanks = kNumRanks_;
    static constexpr uint32_t kKTilesPerFlag = kKTilesPerFlag_;
    static constexpr uint32_t kSourceCapacity = kSourceCapacity_;
    static constexpr uint32_t kNumTopK = kNumTopK_;
    static constexpr GemmType GEMM_TYPE = GemmType::MegaMoE;
    static constexpr bool kIsMegaMoE = true;
    static constexpr bool kIsFusedDispatch = Stage == MegaMoEPhaseKind::Linear1;
    static constexpr bool kIsMaskedLayout = kIsFusedDispatch;
    static constexpr bool kIsNoPadLayout = !kIsFusedDispatch;
    static constexpr bool kIsNoPadPreprocessLayout = false;
    static constexpr uint32_t kNumNBlocks = math::ceil_div(SHAPE_N, BLOCK_N);
    // GEMM1 stores gated BF16 with half the MMA output columns.
    static constexpr uint32_t SHAPE_N_OUT = kIsFusedDispatch ? SHAPE_N / 2 : SHAPE_N;
    static constexpr uint32_t kGroupSpan = 2;

    using Arguments = std::conditional_t<kIsFusedDispatch,
                                         Int8MegaMoESchedulerArguments, TileSchedulerArguments>;
    using Params = Arguments;
    Params const& params;
    int current_iter = 0;
    uint32_t num_aligned_m_blocks;
    uint32_t num_blocks;
    uint32_t curr_group_idx = 0, curr_cumsum = 0, curr_group_m = 0, curr_cumsum_m = 0;
    uint32_t curr_global_block_m_idx = 0;
    uint32_t round_generation = 0;

    template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
    static Params to_underlying_arguments(
        int* groups_layout, ProblemShapeMNKL problem_shape_mnkl,
        TileShape, ClusterShape, KernelHardwareInfo const&, Arguments const& arguments,
        void* = nullptr, const uint32_t = 1, uint32_t = 1u) {
        static_assert(cute::is_static<TileShape>::value);
        static_assert(cute::is_static<ClusterShape>::value);
        const auto problem_shape = cutlass::gemm::to_gemm_coord(problem_shape_mnkl);
        Params params = arguments;
        params.grouped_layout = groups_layout;
        params.shape_m = static_cast<uint32_t>(problem_shape.m());
        return params;
    }

    template <class ProblemShape, class ElementAccumulator>
    static size_t get_workspace_size(
        Arguments const&, ProblemShape, KernelHardwareInfo const&, uint32_t,
        const uint32_t = 1, uint32_t = 1) {
        return 0;
    }

#if defined(__HGGC__)
    CUTLASS_DEVICE explicit SM80MegaMoEScheduler(Params const& params_, const int warp_group_id = 0)
        : params(params_), current_iter(warp_group_id) {
        if constexpr (kIsFusedDispatch)
            round_generation = comm::resolve_fused_round_generation(
                *params_.sym_buffer.template get_base_ptr<layout::MegaMoESignals*>());
        num_aligned_m_blocks = math::ceil_div(params_.shape_m, BLOCK_M);
        if constexpr (kIsFusedDispatch) {
            // Reuse the block-granular metadata already produced for copy CTAs.
            num_aligned_m_blocks = params_.copy_grouped_layout[0];
            num_blocks = num_aligned_m_blocks * kNumNBlocks;
        }
    }

    CUTLASS_DEVICE bool fetch_next_work(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const uint32_t next_block_idx = current_iter++ * gridDim.x + blockIdx.x;
        if constexpr (kIsFusedDispatch) {
            if (next_block_idx >= num_blocks) {
                m_block_idx = num_aligned_m_blocks;
                n_block_idx = kNumNBlocks;
                return false;
            }
            const uint32_t global_m = next_block_idx / kNumNBlocks;
            const uint4 entry = (reinterpret_cast<const uint4*>(params.copy_grouped_layout) + 1)[global_m];
            curr_group_idx = entry.x;
            curr_group_m = entry.y;
            curr_cumsum_m = entry.w;
            curr_global_block_m_idx = global_m;
            m_block_idx = global_m - entry.z;
            n_block_idx = next_block_idx - global_m * kNumNBlocks;
            return true;
        } else {
            return fetch_scatter_work<kGroupSpan>(*this, next_block_idx, m_block_idx, n_block_idx);
        }
    }

    __device__ __forceinline__ int32_t curr_problem_m() const { return curr_group_m; }
    __device__ __forceinline__ int32_t problem_index() const { return curr_group_idx; }
    __device__ __forceinline__ int64_t curr_offset_a(int64_t = 0) const {
        return int64_t(curr_cumsum_m) * SHAPE_K;
    }
    __device__ __forceinline__ int64_t curr_offset_m() const { return curr_cumsum_m; }
    __device__ __forceinline__ int64_t curr_offset_b(const int = 0, int64_t = 0) const {
        return int64_t(curr_group_idx) * SHAPE_N * SHAPE_K;
    }
    __device__ __forceinline__ int64_t curr_offset_c(int64_t = 0) const {
        return int64_t(curr_cumsum_m) * SHAPE_N_OUT;
    }
    __device__ __forceinline__ int64_t curr_offset_c_scales() const { return int64_t(curr_cumsum_m); }
#endif
};

}  // namespace deep_gemm::sched
