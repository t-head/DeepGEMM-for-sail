#pragma once

#include <type_traits>
#include <deep_gemm/scheduler/mega_moe_common.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>

namespace deep_gemm::sched {

struct MegaMoESchedulerArguments {
    int* grouped_layout = nullptr;
    // Copy and compute CTAs share the same M-block metadata.
    int* copy_grouped_layout = nullptr;
    uint32_t shape_m = 0;
    const uint32_t* expert_rank_prefix = nullptr;
    uint8_t* local_fp4_buf = nullptr;
    uint16_t* local_sfa_buf = nullptr;
    volatile uint32_t* copy_ready_flags = nullptr;
    uint32_t num_copy_blocks = 0;
    layout::SymBuffer<> sym_buffer{};
    layout::MegaMoEStaging staging{};
    const uint8_t* dispatch_fp4 = nullptr;
    const uint32_t* dispatch_token_topk = nullptr;
    uint32_t* expert_base = nullptr;
    layout::TokenSrcMetadata* src_info = nullptr;
    int32_t* cumulative_local_expert_recv_stats = nullptr;

    CUTLASS_HOST_DEVICE
    MegaMoESchedulerArguments(uint32_t shape_m = 0, int* grouped_layout_ptr = nullptr)
        : grouped_layout(grouped_layout_ptr), shape_m(shape_m) {}
};

// Two-stage PPU scheduler. Keep the GEMM1 tile queue and scatter traversal
// separate from the ordinary grouped GEMM scheduler.
template <uint32_t SHAPE_N_, uint32_t SHAPE_K_,
          uint32_t BLOCK_M_, uint32_t BLOCK_N_, uint32_t kNumGroups_,
          MegaMoEPhaseKind Stage, uint32_t kNumRanks_,
          uint32_t kNumCopyBlocks_ = 0, uint32_t kKTilesPerFlag_ = 0,
          uint32_t kSourceCapacity_ = 1, uint32_t kNumTopK_ = 1>
struct SM89MegaMoEScheduler {
    static constexpr uint32_t SHAPE_N = SHAPE_N_;
    static constexpr uint32_t SHAPE_K = SHAPE_K_;
    static constexpr uint32_t BLOCK_M = BLOCK_M_;
    static constexpr uint32_t BLOCK_N = BLOCK_N_;
    static constexpr uint32_t kNumGroups = kNumGroups_;
    static constexpr uint32_t kNumRanks = kNumRanks_;
    static constexpr uint32_t kNumCopyBlocks = kNumCopyBlocks_;
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
    static constexpr uint32_t SHAPE_N_OUT = EpilogueTraits<
        kIsFusedDispatch ? EpilogueType::SiluAndMulPostQuantFp4 : EpilogueType::Default
    >::get_shape_n_out(SHAPE_N);
    static constexpr uint32_t kNum1DBlocksPerGroup = !kIsFusedDispatch && kNumRanks == 8 ? 16 : 2;

    using Arguments = std::conditional_t<kIsFusedDispatch,
                                         MegaMoESchedulerArguments, TileSchedulerArguments>;
    using Params = Arguments;
    Params const& params;
    int current_iter = 0;
    uint32_t num_aligned_m_blocks;
    uint32_t num_blocks;
    uint32_t curr_group_idx = 0, curr_cumsum = 0, curr_group_m = 0, curr_cumsum_m = 0;
    uint32_t curr_global_block_m_idx = 0;

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
    CUTLASS_DEVICE explicit SM89MegaMoEScheduler(Params const& params_, const int warp_group_id = 0)
        : params(params_), current_iter(warp_group_id) {
        num_aligned_m_blocks = math::ceil_div(params_.shape_m, BLOCK_M);
        if constexpr (kIsFusedDispatch) {
            // Reuse the block-granular metadata already produced for copy CTAs.
            num_aligned_m_blocks = params_.copy_grouped_layout[0];
            num_blocks = num_aligned_m_blocks * kNumNBlocks;
        }
    }

    CUTLASS_DEVICE bool fetch_next_work(uint32_t& m_block_idx, uint32_t& n_block_idx,
                                        uint32_t* ticket_storage = nullptr) {
        uint32_t next_block_idx;
        if constexpr (kIsFusedDispatch && kNumCopyBlocks > 0) {
            // Finish the previous tile before reusing GEMM shared storage for a ticket.
            __syncthreads();
            if (threadIdx.x == 0)
                *ticket_storage = atomicAdd(params.copy_grouped_layout + 1, 1);
            __syncthreads();
            next_block_idx = *reinterpret_cast<volatile uint32_t*>(ticket_storage);
            __syncthreads();
        } else {
            next_block_idx = current_iter++ * gridDim.x + blockIdx.x;
        }
        return mega_moe_common::fetch_next_work(*this, next_block_idx, m_block_idx, n_block_idx);
    }

    __device__ __forceinline__ int32_t curr_problem_m() const { return curr_group_m; }
    __device__ __forceinline__ int32_t problem_index() const { return curr_group_idx; }
    __device__ __forceinline__ int64_t curr_offset_a() const {
        return int64_t(curr_cumsum_m) * SHAPE_K;
    }
    __device__ __forceinline__ int64_t curr_offset_mxfp4_scalea() const { return int64_t(curr_cumsum_m); }
    __device__ __forceinline__ int64_t curr_offset_m() const { return curr_cumsum_m; }
    __device__ __forceinline__ int64_t curr_offset_b(const int = 0) const {
        return int64_t(curr_group_idx) * SHAPE_N * SHAPE_K;
    }
    __device__ __forceinline__ int64_t curr_offset_mxfp4_scaleb(const int = 0) const {
        const uint32_t shape_k_scale = math::ceil_div(SHAPE_K, uint32_t(32));
        return int64_t(curr_group_idx) * SHAPE_N * shape_k_scale;
    }
    __device__ __forceinline__ int64_t curr_offset_mxfp4_c() const {
        return int64_t(curr_group_idx) * SHAPE_N;
    }
    __device__ __forceinline__ int64_t curr_offset_c() const {
        return int64_t(curr_cumsum_m) * SHAPE_N_OUT;
    }
    __device__ __forceinline__ int64_t curr_offset_c_scales() const { return int64_t(curr_cumsum_m); }
#endif
};

}  // namespace deep_gemm::sched
