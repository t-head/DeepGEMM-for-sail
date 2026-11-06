#pragma once

#include <type_traits>
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
    // First flag publishes metadata; the rest cover three SFA partitions per expert.
    volatile uint32_t* preprocess_ready_flags = nullptr;
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

    CUTLASS_DEVICE void get_swizzled_block_idx(
        const uint32_t num_m_blocks, int block_idx,
        uint32_t& m_block_idx, uint32_t& n_block_idx) {
        // Preserve small M groups and the EP8 bit-reversal mapping.
        const uint32_t group_span = kNum1DBlocksPerGroup >= 16 ? 2u : kNum1DBlocksPerGroup;
        const auto num_blocks_per_group = kNumNBlocks * group_span;
        const auto group_idx = block_idx / num_blocks_per_group;
        const auto first_block_idx = group_idx * group_span;
        const auto in_group_idx = block_idx % num_blocks_per_group;
        const uint32_t num_blocks_in_group = min(group_span, num_m_blocks - first_block_idx);
        m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        n_block_idx = in_group_idx / num_blocks_in_group;
        if constexpr (kNum1DBlocksPerGroup >= 16) {
            if (num_m_blocks >= kNum1DBlocksPerGroup) {
                const uint32_t bits = 32u - __clz(num_m_blocks - 1u);
                // Skip out-of-range reversed inputs for a 3/4-full domain.
                const uint32_t input =
                    bits >= 2 && num_m_blocks == (3u << (bits - 2))
                    ? m_block_idx + m_block_idx / 3u : m_block_idx;
                const uint32_t permuted = __brev(input) >> (32u - bits);
                if (permuted < num_m_blocks)
                    m_block_idx = permuted;
            }
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
        if constexpr (kIsFusedDispatch) {
            if (next_block_idx >= num_blocks) {
                m_block_idx = num_aligned_m_blocks;
                n_block_idx = kNumNBlocks;
                return false;
            }
            // {expert, token count, first M-block, first row}.
            const uint32_t global_m_block = next_block_idx / kNumNBlocks;
            const uint4 data = (reinterpret_cast<const uint4*>(params.copy_grouped_layout) + 1)[global_m_block];
            curr_group_idx = data.x;
            curr_group_m = data.y;
            curr_cumsum_m = data.w;
            curr_global_block_m_idx = global_m_block;
            m_block_idx = global_m_block - data.z;
            n_block_idx = next_block_idx - global_m_block * kNumNBlocks;
        } else {
            uint32_t num_m_blocks;
            while (true) {
                if (curr_group_idx == kNumGroups) {
                    m_block_idx = num_aligned_m_blocks;
                    n_block_idx = kNumNBlocks;
                    return false;
                }
                curr_group_m = static_cast<uint32_t>(__ldg(params.grouped_layout + curr_group_idx));
                num_m_blocks = math::ceil_div(curr_group_m, BLOCK_M);
                const auto current_m_block_cumsum = curr_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * kNumNBlocks)
                    break;
                curr_group_idx ++, curr_cumsum = current_m_block_cumsum;
                curr_cumsum_m += curr_group_m;
            }
            get_swizzled_block_idx(num_m_blocks, next_block_idx - curr_cumsum * kNumNBlocks,
                                  m_block_idx, n_block_idx);
        }
        return true;
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
