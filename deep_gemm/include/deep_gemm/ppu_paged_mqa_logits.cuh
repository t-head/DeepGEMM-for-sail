#pragma once
#include "tools/util/include/cutlass/util/packed_stride.hpp"
#include "ppu_include.hpp"
#include "cute_tie.cuh"
#include "utils.cuh"
#include "utils_cutlass3.h"
#include "profiling_interface.hpp"

#define ENABLE_WARP_CONTIG_LAYOUT 1

__forceinline__ __device__ uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm ("mov.u32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

__device__  __forceinline__ float ld_shared(const float* ptr) {
    float ret;
    asm volatile("ppu.ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}


namespace deep_gemm {

template <uint32_t SPLIT_KV, uint32_t kNumSMs>
__global__
void smxx_paged_mqa_logits_metadata(const uint32_t batch_size, const uint32_t* context_lens, uint32_t* schedule_metadata) {
    extern __shared__ uint32_t prefix_sum[];
    const uint32_t tid = threadIdx.x;

    // load context lens
    for (uint32_t k = tid; k < batch_size; k += blockDim.x) {
        prefix_sum[k] = ceil_div(__ldg(context_lens + k), SPLIT_KV);
    }
    __syncthreads();

    // calculate prefix sum
    uint32_t sum = 0;
    uint32_t* temp_prefix_sum = prefix_sum;
    uint32_t loop_num = batch_size / blockDim.x;
    for (uint32_t k = 0; k < loop_num; k++) {
        uint32_t val = temp_prefix_sum[tid];
        for (uint32_t offset = 1; offset < blockDim.x; offset <<= 1) {
            uint32_t temp = 0;
            if (tid >= offset) {
                temp = temp_prefix_sum[tid - offset];
            }
            __syncthreads();
            val += temp;
            __syncthreads();
            temp_prefix_sum[tid] = val;
        }
        temp_prefix_sum[tid] += sum;
        __syncthreads();
        sum = temp_prefix_sum[blockDim.x - 1];
        temp_prefix_sum += blockDim.x;
    }
    // blockDim.x < 1024: only last loop
    uint32_t last = batch_size - loop_num * blockDim.x;
    uint32_t val = (tid < last) ? temp_prefix_sum[tid] : 0;
    for (uint32_t offset = 1; offset < last; offset <<= 1) {
        uint32_t temp = (tid >= offset && tid < last) ? temp_prefix_sum[tid - offset] : 0;
        __syncthreads();
        val += temp;
        __syncthreads();
        if (tid < last) {
            temp_prefix_sum[tid] = val;
        }
    }
    if (tid < last) {
        temp_prefix_sum[tid] += sum;
    }
    __syncthreads();
    sum = prefix_sum[batch_size - 1];

    // binary search
    const uint32_t& q = sum / kNumSMs, r = sum % kNumSMs;
    for (uint32_t sm_idx = tid; sm_idx < kNumSMs + 1; sm_idx += blockDim.x) {
        uint32_t seg_starts = sm_idx * q + min(sm_idx, r);

        int left = 0;
        int right = batch_size - 1;
        int found_idx = batch_size;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (prefix_sum[mid] > seg_starts) {
                found_idx = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        uint32_t q_idx = found_idx;
        uint32_t prev_sum = (q_idx == 0) ? 0 : prefix_sum[q_idx - 1];
        uint32_t kv_split_idx = seg_starts - prev_sum;

        schedule_metadata[sm_idx * 2] = q_idx;
        schedule_metadata[sm_idx * 2 + 1] = kv_split_idx;
    }
}

template <uint32_t SPLIT_KV, uint32_t kNumSMs>
void launch_paged_mqa_logits_metadata(const uint32_t batch_size, const uint32_t* context_lens, uint32_t* schedule_metadata,
                                      hggcStream_t stream) {
    int grid = 1;
    int block = min(batch_size, 1024);
    int smem_size = batch_size * 4;
    smxx_paged_mqa_logits_metadata<SPLIT_KV, kNumSMs><<<grid, block, smem_size, stream>>>(
        batch_size, context_lens, schedule_metadata);

};

}// namespace deep_gemm

namespace cutlass::gemm::kernel {


template <uint32_t BLOCK_KV, uint32_t kNumMathWarpGroups>
struct PagedMQALogitsScheduler {
    uint32_t batch_size;
    const uint32_t* context_lens;

    uint32_t current_q_idx, current_kv_idx;
    uint32_t end_q_idx, end_kv_idx;
    uint32_t current_num_kv;

    __device__ __forceinline__ explicit PagedMQALogitsScheduler(const uint32_t& batch_size, const uint32_t& sm_idx,
                                                                const uint32_t* context_lens, const uint32_t* schedule_meta) {
        this->batch_size = batch_size;
        this->context_lens = context_lens;

        const auto& current_pack = __ldg(reinterpret_cast<const uint2*>(schedule_meta) + sm_idx);
        const auto& end_pack = __ldg(reinterpret_cast<const uint2*>(schedule_meta) + sm_idx + 1);
        current_q_idx = current_pack.x, current_kv_idx = current_pack.y * kNumMathWarpGroups;
        end_q_idx = end_pack.x, end_kv_idx = end_pack.y * kNumMathWarpGroups;

        current_num_kv = current_q_idx < batch_size ? ceil_div(__ldg(this->context_lens + current_q_idx), BLOCK_KV) : 0;
    }

    __device__ __forceinline__ bool fetch_next_task(uint32_t &q_idx, uint32_t &kv_idx, uint32_t &num_kv) {
        q_idx = current_q_idx;
        kv_idx = current_kv_idx;
        num_kv = current_num_kv;

        if (is_last_task(q_idx, kv_idx))
            return false;

        current_kv_idx += kNumMathWarpGroups;
        if (current_kv_idx >= current_num_kv) {
            ++ current_q_idx;
            current_kv_idx = 0;
            current_num_kv = current_q_idx < batch_size ? ceil_div(__ldg(this->context_lens + current_q_idx), BLOCK_KV) : 0;
        }

        return true;
    }

    __device__ __forceinline__ bool exist_q_idx(const uint32_t& q_idx) const {
        return q_idx < end_q_idx or q_idx == end_q_idx and 0 < end_kv_idx;
    }

    __device__ __forceinline__ bool is_last_task(const uint32_t& q_idx, const uint32_t& kv_idx) const {
        return (q_idx > end_q_idx) or (q_idx == end_q_idx and kv_idx == end_kv_idx);
    }
};

template <typename ElementQK, typename ElementAcc, typename ElementLogits,
          uint32_t kNextN, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV>
class PPUPagedMqaLogits {
public:
  using ElementC            = float;
  using LayoutA             = cutlass::layout::RowMajor;
  using LayoutB             = cutlass::layout::ColumnMajor;
  using LayoutC             = cutlass::layout::RowMajor;
  using ElementD            = ElementC;
  using LayoutD             = cutlass::layout::RowMajor;
  using ElementCompute      = float;
  using ElementScale        = float;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  static constexpr int BLOCK_M = BLOCK_KV;
  static constexpr int BLOCK_N = kNextN * kNumHeads;
  static constexpr int BLOCK_K = kHeadDim;
  static constexpr int WARP_M = 16;
  static constexpr int WARP_N = kNumHeads;

  static constexpr int BLOCK_Q = kNextN;
  static constexpr int WARP_Q = 1;
  static constexpr int kNumMathWarpGroups = SPLIT_KV / BLOCK_KV;

  using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
  using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
  using ProblemShape_MNKL = Shape<int,int,int,int>;

  using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
  using WarpShape = Shape<Int<WARP_M>, Int<WARP_N>, Int<BLOCK_K>>;
  static constexpr int WarpOnM = BLOCK_M / WARP_M;
  static constexpr int WarpOnN = BLOCK_N / WARP_N;
#if __HGGC_ARCH__ == 100
    using ArchTag = cutlass::arch::PPU0010;
#else
    using ArchTag = cutlass::arch::PPU0015;
#endif

  using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ArchTag, ElementQK,ElementQK,ElementAcc>::type;
  using MmaK_type = typename cutlass::platform::conditional<sizeof(ElementQK) == 2, _16, _32 >::type;

#if ENABLE_WARP_CONTIG_LAYOUT
  static constexpr int InstM = 16;
  static constexpr int InstN = 16;
  using WarpIterM = Int<BLOCK_M / WARP_M>;
  using WarpIterN = Int<BLOCK_N / WARP_N>;
  using MmaIterM = Int<WARP_M / InstM>;
  using MmaIterN = Int<WARP_N / InstN>;
  using PermutationMNK = Tile<
      Layout<Shape<Int<InstM>, WarpIterM, MmaIterM>, Stride<_1, Int<WARP_M>, Int<InstM>> >,
      Layout<Shape<Int<InstN>, WarpIterN, MmaIterN>, Stride<_1, Int<WARP_N>, Int<InstN>> >,
      MmaK_type
    >;
  using TiledMma = TiledMMA<MMA_Atom<MmaInst>, Layout<Shape<WarpIterM, WarpIterN, _1>>, PermutationMNK>;
#else
  using TiledMma = TiledMMA<
      MMA_Atom<MmaInst>,
      Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, _1>>,  // 1x4x1 thread group
      Tile<Int<WarpOnM * 16>, Int<WarpOnN * 16>, MmaK_type
      >>;       // 1x1x1 value group
#endif

  static constexpr int MaxThreadsPerBlock = kNumMathWarpGroups * size(TiledMma{});
  static constexpr int WarpOnGroup = size(TiledMma{}) / 32;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr bool WarpInterleaving = (MaxThreadsPerBlock == 512);
  static_assert(!(WarpInterleaving && kNumQStages == 1), "warp-interleave does not support stage_q = 1");

  static constexpr bool TransA = cutlass::platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = cutlass::platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;
  using DefaultOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementQK, TransA, Int<BLOCK_M>, Int<BLOCK_K>, false>;
  using DefaultOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementQK, TransB, Int<BLOCK_N>, Int<BLOCK_K>, true>;

  // A
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;
  // B
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<kNumKVStages>{}, Int<kNumMathWarpGroups>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<kNumQStages>{})));

  constexpr static uint32_t CTA_M = shape<0>(TileShape{});
  constexpr static uint32_t CTA_N = shape<1>(TileShape{});
  constexpr static uint32_t CTA_K = shape<2>(TileShape{});

  // ScaleA (k_scales, float per KV row) -- AIU load, no pre-rearrangement
  using DefaultOperandSFA =
      cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, float, false, _1, Int<CTA_M>, false, 0, false>;
  using SmemLayoutAtomSFA = typename DefaultOperandSFA::SmemLayoutAtom;
  using GmemTiledCopySFA = typename DefaultOperandSFA::GmemTiledCopy;
  using SmemLayoutSFA = decltype(tile_to_shape(
      SmemLayoutAtomSFA{},
      make_shape(_1{}, Int<CTA_M>{}, Int<kNumKVStages>{}, Int<kNumMathWarpGroups>{})));

  // ScaleB (weights, float per Q head) -- AIU load, no pre-rearrangement
  using DefaultOperandSFB =
      cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, float, false, _1, Int<CTA_N>, false, 0, false>;
  using SmemLayoutAtomSFB = typename DefaultOperandSFB::SmemLayoutAtom;
  using GmemTiledCopySFB = typename DefaultOperandSFB::GmemTiledCopy;
  using SmemLayoutSFB = decltype(tile_to_shape(
      SmemLayoutAtomSFB{},
      make_shape(_1{}, Int<CTA_N>{}, Int<kNumQStages>{})));

  using StrideSFA = Stride<Int<CTA_M>, _1>;
  using StrideSFB = Stride<Int<CTA_N>, _1>;

  static_assert(kNumQStages <= 2 && kNumKVStages >= 3, "q_stage <= 2 and kv_stage >= 3");

  // Kernel level shared memory storage
  struct SharedStorage {
    cute::array_aligned<ElementQK, cute::cosize_v<SmemLayoutA>> smem_k;
    cute::array_aligned<ElementQK, cute::cosize_v<SmemLayoutB>> smem_q;
    cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutSFA>> smem_k_scales;
    cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutSFB>> smem_weight;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    const ElementQK * ptr_q;
    const ElementQK * ptr_k;
    const float * k_scales;
    const float * weights;
    const uint32_t batch_size;
    const uint64_t logits_stride;
    const uint64_t kv_cache_stride_bytes;
    const uint32_t block_table_stride;
    const uint32_t* context_lens;
    ElementLogits* logits;
    const uint32_t* block_table;
    const uint32_t* schedule_meta;
    StrideA dA;
    StrideB dB;
    KernelHardwareInfo hw_info{};
  };

  // Kernel entry point API
  using Params = Arguments;

  GmemTiledCopyA gmem_tiled_copy_A;
  GmemTiledCopyB gmem_tiled_copy_B;

  GmemTiledCopySFA gmem_tiled_copy_SFA;
  GmemTiledCopySFB gmem_tiled_copy_SFB;

  // // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    return dim3(params.hw_info.cu_count, 1, 1);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE void
  init_aiu_copy(ProblemShape_MNKL const& problem_shape_mnkl, Params params) {
    auto [M,N,K,L] = problem_shape_mnkl;
    using TilerA = typename GmemTiledCopyA::Tiler_MN;
    using TilerB = typename GmemTiledCopyB::Tiler_MN;
    using TilerSFA = typename GmemTiledCopySFA::Tiler_MN;
    using TilerSFB = typename GmemTiledCopySFB::Tiler_MN;

    gmem_tiled_copy_A.desc_.template init<ElementQK, TransA, get<0>(TilerA{}), get<1>(TilerA{})>(nullptr, CTA_M, CTA_K, params.dA);
    gmem_tiled_copy_B.desc_.template init<ElementQK, TransB, get<0>(TilerB{}), get<1>(TilerB{})>(nullptr, CTA_N, CTA_K, params.dB);

    gmem_tiled_copy_SFA.desc_.template init<float, false, get<0>(TilerSFA{}), get<1>(TilerSFA{})>(
        nullptr, 1, CTA_M, StrideSFA{});
    gmem_tiled_copy_SFB.desc_.template init<float, false, get<0>(TilerSFB{}), get<1>(TilerSFB{})>(
        nullptr, 1, CTA_N, StrideSFB{});
  };

  template <class BlockCoord_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_mnkl, BlockCoord_MNKL const& blk_coord_mnkl, Params const& params) {
    auto [M,N,K,L] = problem_shape_mnkl;
    auto [m_coord, n_coord, _, l_coord] = blk_coord_mnkl;
    // load init A
    Tensor mA_mkl = make_tensor(make_gmem_ptr(params.ptr_k), make_shape(M,K,L), params.dA);   // (m,k,l)
    Tensor mA_mk = make_mix_tensor_like(mA_mkl(_,_,l_coord));                                 // (m,k)
    Tensor gA = local_tile(mA_mk, TileShape{}, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)

    // load init B
    Tensor mB_nkl = make_tensor(make_gmem_ptr(params.ptr_q), make_shape(N,K,L), params.dB);   //(n,k,l)
    Tensor mB_nk = make_mix_tensor_like(mB_nkl(_,_,l_coord));                                 // (n,k)
    Tensor gB = local_tile(mB_nk, TileShape{}, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)

    // // load init scale A/B
    Tensor mSFA_m = make_tensor(make_gmem_ptr(params.k_scales), Shape<_1, Int<CTA_M>>{}, StrideSFA{});
    Tensor mSFA_m_mix = make_mix_tensor_like(mSFA_m);
    Tensor gSFA = local_tile(mSFA_m_mix, Shape<_1, Int<CTA_M>>{}, make_coord(_, 0));

    Tensor mSFB_n = make_tensor(make_gmem_ptr(params.weights), Shape<_1, Int<CTA_N>>{}, StrideSFB{});
    Tensor mSFB_n_mix = make_mix_tensor_like(mSFB_n);
    Tensor gSFB = local_tile(mSFB_n_mix, Shape<_1, Int<CTA_N>>{}, make_coord(0, _));

    return cute::make_tuple(gA, gB, gSFA, gSFB);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    int warp_idx = canonical_warp_idx_sync();
    int thread_idx = int(threadIdx.x);
    auto [warp_group_idx, local_warp_idx, local_thread_idx] = [&]() {
        if constexpr (kNumMathWarpGroups == 1) {
            return cute::make_tuple(0, warp_idx, thread_idx);
        } else {
            int warp_group_idx = warp_idx / WarpOnGroup;
            int local_warp_idx = warp_idx % WarpOnGroup;
            int local_thread_idx = thread_idx % (32 * WarpOnGroup);
            return cute::make_tuple(warp_group_idx, local_warp_idx, local_thread_idx);
        }
    }();

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto M = BLOCK_M;
    auto N = BLOCK_N;
    auto K = kHeadDim;
    auto L = 1;
    auto problem_shape_mnkl = ProblemShape_MNKL{M, N, K, L};
    auto blk_coord_mnkl = make_coord(0, 0, _, 0);

    // init aiu copy and async copy
    init_aiu_copy(problem_shape_mnkl, params);

    // init input tensors
    auto load_inputs = load_init(problem_shape_mnkl, blk_coord_mnkl, params);
    Tensor gA = get<0>(load_inputs);
    Tensor gB = get<1>(load_inputs);
    Tensor gSFA = get<2>(load_inputs);
    Tensor gSFB = get<3>(load_inputs);

    // 4D KV smem: (BLOCK_M, BLOCK_K, kNumKVStages, kNumMathWarpGroups)
    // Slice by warp_group_idx to get this warp_group's 3D region
    Tensor sA_full = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutA{}); // 4D
    Tensor sA = sA_full(_, _, _, warp_group_idx); // 3D slice for this warp_group
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)
    // Partition the copying of A and B tiles across the threads (use local_thread_idx)
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(local_thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(local_thread_idx);
    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    // 4D SFA smem: (_1, BLOCK_M, kNumKVStages, kNumMathWarpGroups)
    Tensor sSFA_full = make_tensor(make_smem_ptr(shared_storage.smem_k_scales.data()), SmemLayoutSFA{}); // 4D
    Tensor sSFA = sSFA_full(_, _, _, warp_group_idx); // 3D slice for this warp_group
    Tensor sSFB = make_tensor(make_smem_ptr(shared_storage.smem_weight.data()), SmemLayoutSFB{});

    auto gmem_thr_copy_SFA = gmem_tiled_copy_SFA.get_thread_slice(local_thread_idx);
    auto gmem_thr_copy_SFB = gmem_tiled_copy_SFB.get_thread_slice(local_thread_idx);

    Tensor tSFAgSFA = gmem_thr_copy_SFA.partition_S(gSFA);
    Tensor tSFAsSFA = gmem_thr_copy_SFA.partition_D(sSFA);

    Tensor tSFBgSFB = gmem_thr_copy_SFB.partition_S(gSFB);
    Tensor tSFBsSFB = gmem_thr_copy_SFB.partition_D(sSFB);

    TiledMma tiled_mma;
    Tensor accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));
    auto thr_mma = tiled_mma.get_thread_slice(local_thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                     // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                     // (MMA,MMA_N,MMA_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                  // MMA_K

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(local_warp_idx * 32);
    Tensor tCsA            = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));                  // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                   // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(local_warp_idx * 32);
    Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));                  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    constexpr bool enable_print = false;
    bool thread_print = enable_print && cute::thread(0, 0);

    // Scheduler
    auto scheduler = PagedMQALogitsScheduler<BLOCK_KV, kNumMathWarpGroups>(params.batch_size, blockIdx.x, params.context_lens, params.schedule_meta);
    DG_STATIC_ASSERT(SPLIT_KV % BLOCK_KV == 0, "Unaligned SPLIT_KV");

    ElementLogits weights[kNumHeads / 4];

    clear(accum);
    auto tKgK = tAgA;
    auto tSFKgSFK = tSFAgSFA;
    auto tQgQ = tBgB;
    auto tSFQgSFQ = tSFBgSFB;

    constexpr bool load_kv_scale = sizeof(ElementQK) == 1;
    const auto& lane_idx = get_lane_idx();
    int warp_m_idx = local_warp_idx % WarpOnM;
    int warp_n_idx = local_warp_idx / WarpOnM;
    const auto& warp_offset = warp_m_idx * WARP_M;
    const auto& v_0_offset = lane_idx / 4 + 0;
    const auto& v_1_offset = lane_idx / 4 + 8;
    uint32_t warp_q_idx = warp_n_idx;
    int warp_group_id = warp_idx / 8;

    uint32_t q_idx = scheduler.current_q_idx;
    uint32_t kv_idx_base;
    uint32_t kv_idx_array[kNumKVStages];
    uint32_t smem_pipe_read_q = 0, smem_pipe_read_kv = 0;
    uint32_t smem_pipe_write_q = 0, smem_pipe_write_kv = 0;

    auto load_q_g2s = [&](uint32_t q_idx) {
        auto q_offset = q_idx * BLOCK_N;
        tBgB.data() = tQgQ.data() + q_offset * kHeadDim;
        tSFBgSFB.data() = tSFQgSFQ.data() + q_offset;
        copy_aiu(gmem_tiled_copy_B, tBgB(_,_,_,0), tBsB(_,_,_,smem_pipe_write_q), warp_idx);
        copy_aiu(gmem_tiled_copy_SFB, tSFBgSFB(_,_,_,0), tSFBsSFB(_,_,_,smem_pipe_write_q), warp_idx);
        if (thread_print) {
            printf("  copy_q q_idx = %d, q_offset = %d, stage = %d\n", q_idx, q_offset, smem_pipe_write_q);
        }
        smem_pipe_write_q = (smem_pipe_write_q + 1) % kNumQStages;
    };

    auto load_kv_g2s = [&](uint32_t q_idx, uint32_t kv_idx) {
        auto kv_offset = __ldg(params.block_table + q_idx * params.block_table_stride + kv_idx);
        tAgA.data() = tKgK.data() + kv_offset * params.kv_cache_stride_bytes;
        if constexpr(load_kv_scale) {
            tSFAgSFA.data() = tSFKgSFK.data() + kv_offset * params.kv_cache_stride_bytes / 4;
            copy_aiu<true>(gmem_tiled_copy_A, tAgA(_,_,_,0), tAsA(_,_,_,smem_pipe_write_kv),
                           gmem_tiled_copy_SFA, tSFAgSFA(_,_,_,0), tSFAsSFA(_,_,_,smem_pipe_write_kv), local_warp_idx);
        } else {
            copy_aiu(gmem_tiled_copy_A, tAgA(_,_,_,0), tAsA(_,_,_,smem_pipe_write_kv), local_warp_idx);
        }
        if (thread_print) {
            printf("  copy_k q_idx = %d, kv_idx = %d, kv_offset = %d, wg = %d, stage = %d\n", q_idx, kv_idx, kv_offset, warp_group_idx, smem_pipe_write_kv);
        }
    };

    auto load_q_s2r = [&]() {
        copy(smem_tiled_copy_B, tCsB(_,_,_,smem_pipe_read_q), tCrB_copy_view);
        if (thread_print) {
            printf("    copy q to vreg, q_stage_idx = %d,\n", smem_pipe_read_q);
        }

        // Read weights
        float * smem_weights_staged = sSFB(_,_,smem_pipe_read_q).data().get() + warp_q_idx * kNumHeads;
        if constexpr (std::is_same_v<ElementLogits, float>) {
            #pragma unroll
            for (uint32_t j = 0; j < kNumHeads / 4; ++ j) {
#if __HGGC_ARCH__ == 150
                weights[j] = ld_shared(smem_weights_staged + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
#else
                weights[j] = ld_shared(smem_weights_staged + (j / 2) * 8 + (j & 1) * 4 + lane_idx % 4);
#endif
            }
        } else {
            #pragma unroll
            for (uint32_t j = 0; j < kNumHeads / 4; j += 2) {
                float w0, w1;
#if __HGGC_ARCH__ == 150
                w0 = ld_shared(smem_weights_staged + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
                w1 = ld_shared(smem_weights_staged + ((j + 1) / 2) * 8 + ((j + 1) & 1) + (lane_idx % 4) * 2);
#else
                w0 = ld_shared(smem_weights_staged + (j / 2) * 8 + (j & 1) * 4 + lane_idx % 4);
                w1 = ld_shared(smem_weights_staged + ((j + 1) / 2) * 8 + ((j + 1) & 1) * 4 + lane_idx % 4);
#endif
                uint32_t d;
                asm volatile("ppu.cvt.rtte.bf16x2.f32 %0, %1, %2;\n"
                             : "=r"(d)
                             : "f"(w1), "f"(w0));
                *reinterpret_cast<uint32_t*>(&weights[j]) = d;
            }
        }
        smem_pipe_read_q = (smem_pipe_read_q + 1) % kNumQStages;
    };

    auto load_next_qk_g2s = [&](bool load_q) {
        uint32_t q_idx, kv_idx_base, num_kv;
        if (scheduler.fetch_next_task(q_idx, kv_idx_base, num_kv)) {
            if (load_q) load_q_g2s(q_idx);
            uint32_t actual_kv = kv_idx_base + warp_group_idx;
            if (actual_kv < num_kv)
                load_kv_g2s(q_idx, actual_kv);
        }
        kv_idx_array[smem_pipe_write_kv] = kv_idx_base;
        smem_pipe_write_kv = (smem_pipe_write_kv + 1) % kNumKVStages;
        if (thread_print) printf("cp_async commit, write_stage = %d\n", smem_pipe_write_kv);
        cp_async_fence();
    };

    static constexpr uint32_t kMmaIterM = MmaIterM{};
    float scale_kv_array[kMmaIterM * 2];

    auto load_kv_scale_s2r = [&]() {
        float * smem_kv_scales = sSFA(_,_,smem_pipe_read_kv).data().get();
        for (int m = 0; m < kMmaIterM; m++) {
            uint32_t mma_offset = m * InstM;
            scale_kv_array[m * 2    ] = (sizeof(ElementQK) == 2) ? 1 : ld_shared(smem_kv_scales + warp_offset + mma_offset + v_0_offset);
            scale_kv_array[m * 2 + 1] = (sizeof(ElementQK) == 2) ? 1 : ld_shared(smem_kv_scales + warp_offset + mma_offset + v_1_offset);
        }
    };

    // Reduce over heads, scale by per-row KV scale and store logits
    auto epilogue = [&](uint32_t q_idx, uint32_t kv_idx) {
        static constexpr uint32_t kNumAccumPerMma = 8;
        static constexpr uint32_t kAccumStrideN16 = kNumAccumPerMma * kMmaIterM;
        CUTE_STATIC_ASSERT(kNumHeads % 8 == 0);
        CUTE_STATIC_ASSERT(WARP_Q == 1);
        for (int m = 0; m < kMmaIterM; m++) {
            uint32_t mma_offset = m * InstM;
            float scale_kv_0 = scale_kv_array[m * 2];
            float scale_kv_1 = scale_kv_array[m * 2 + 1];

            if constexpr (std::is_same_v<ElementLogits, float>) {
                // Float epilogue: reduce over heads, scale by per-row KV scale and store
                auto shifted_accum = accum.data() + m * kNumAccumPerMma;
                const auto& transform = [&](const uint32_t& j, const uint32_t& n = 0) {
#if __HGGC_ARCH__ == 150
                    return fmaxf(shifted_accum[n * kAccumStrideN16 + j], 0) * weights[n * 4 + (j / 4) * 2 + (j & 1)];
#else
                    return fmaxf(shifted_accum[n * kAccumStrideN16 + j], 0) * weights[n * 4 + j % 4];
#endif
                };

                // Intra-thread reduction
                float sum[8] = {transform(0), transform(1), transform(2), transform(3),
                                transform(4), transform(5), transform(6), transform(7)};
                #pragma unroll
                for (uint32_t n = 1; n < kNumHeads / InstN; ++ n) {
                    #pragma unroll
                    for (uint32_t k = 0; k < kNumAccumPerMma; k ++)
                        sum[k] += transform(k, n);
                }
#if __HGGC_ARCH__ == 150
                float v_0 = sum[0] + sum[1] + sum[4] + sum[5];
                float v_1 = sum[2] + sum[3] + sum[6] + sum[7];
#else
                float v_0 = sum[0] + sum[1] + sum[2] + sum[3];
                float v_1 = sum[4] + sum[5] + sum[6] + sum[7];
#endif

                // Inter-thread reduction
                #pragma unroll
                for (uint32_t j = 0; j < 2; ++ j) {
                    const auto& offset = static_cast<int>(1u << j);
                    v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
                    v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
                }

                // Store into the global memory
                // NOTES: we have redundant writes here, consider more carefully
                auto kv_offset = (q_idx * kNextN + warp_q_idx) * params.logits_stride + kv_idx * BLOCK_KV;
                params.logits[kv_offset + warp_offset + mma_offset + v_0_offset] = v_0 * scale_kv_0;
                params.logits[kv_offset + warp_offset + mma_offset + v_1_offset] = v_1 * scale_kv_1;
            } else {
                // BF16 vectorized epilogue: separate cvt and fma2 phases with __ppu_sched_bound()
                constexpr int kTotalTransforms = 4 * (kNumHeads / InstN);
                __ppu_bfloat162 sum_0 = {0, 0}, sum_1 = {0, 0};
                uint32_t cvt_buf[kTotalTransforms];

                // cvt phase: issue all ppu.cvt.rtte.bf16x2.f32.relu
                #pragma unroll
                for (int idx = 0; idx < kTotalTransforms; idx++) {
                    int n = idx / 4;
                    int sub = idx % 4;
                    int j = sub * 2;
                    asm volatile("ppu.cvt.rtte.bf16x2.f32.relu %0, %1, %2;\n"
                                 : "=r"(cvt_buf[idx])
                                 : "f"(accum(j + 1, m, n)), "f"(accum(j, m, n)));
                }
                __ppu_sched_bound();
                // fma2 phase: issue all __hfma2
                #pragma unroll
                for (int idx = 0; idx < kTotalTransforms; idx++) {
                    int n = idx / 4;
                    int sub = idx % 4;
                    int j = sub * 2;
                    __ppu_bfloat162 a = reinterpret_cast<__ppu_bfloat162&>(cvt_buf[idx]);
#if __HGGC_ARCH__ == 150
                    __ppu_bfloat162 b = *reinterpret_cast<__ppu_bfloat162*>(&weights[n * 4 + (j / 4) * 2]);
                    if (sub % 2 == 0) {
                        sum_0 = __hfma2(a, b, sum_0);
                    } else {
                        sum_1 = __hfma2(a, b, sum_1);
                    }
#else
                    __ppu_bfloat162 b = *reinterpret_cast<__ppu_bfloat162*>(&weights[n * 4 + ((j % 4) / 2) * 2]);
                    if (sub < 2) {
                        sum_0 = __hfma2(a, b, sum_0);
                    } else {
                        sum_1 = __hfma2(a, b, sum_1);
                    }
#endif
                }
                __ppu_sched_bound();
                __ppu_bfloat16 v_0_bf = __hadd(__low2bfloat16(sum_0), __high2bfloat16(sum_0));
                __ppu_bfloat16 v_1_bf = __hadd(__low2bfloat16(sum_1), __high2bfloat16(sum_1));

                // Apply per-token KV scale
                v_0_bf = __hmul(v_0_bf, (__ppu_bfloat16)scale_kv_0);
                v_1_bf = __hmul(v_1_bf, (__ppu_bfloat16)scale_kv_1);

                // Packed cross-lane reduction
                __ppu_bfloat162 packed = {v_0_bf, v_1_bf};
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    uint32_t bits = reinterpret_cast<uint32_t&>(packed);
                    uint32_t received_bits = __shfl_xor_sync(0xffffffffu, bits, 1u << j);
                    __ppu_bfloat162 received = reinterpret_cast<__ppu_bfloat162&>(received_bits);
                    packed = __hadd2(packed, received);
                }

                // Store into the global memory
                auto kv_offset = (q_idx * kNextN + warp_q_idx) * params.logits_stride + kv_idx * BLOCK_KV;
                params.logits[kv_offset + warp_offset + mma_offset + v_0_offset] = __low2bfloat16(packed);
                params.logits[kv_offset + warp_offset + mma_offset + v_1_offset] = __high2bfloat16(packed);
            }
        }
    };
    // load qk
    for (int i = 0; i < kNumKVStages - 1; i++) {
        load_next_qk_g2s(i == 0);
    }
    kv_idx_base = kv_idx_array[0];
    // wait AIU Q and first K
    cp_async_wait<kNumKVStages - 2>();
    __syncthreads();

    if constexpr (WarpInterleaving) {
        if (warp_group_id == 1) {
            __ppu_barrier_arrive(5, MaxThreadsPerBlock, 0);
        }
    }

    auto handle_q_change = [&]() {
        if constexpr(kNumQStages > 1) {
            if (scheduler.exist_q_idx(q_idx + 1)) {
                load_q_g2s(q_idx + 1);
            }
            cp_async_fence();
        }
        load_q_s2r();
        if constexpr(kNumQStages == 1) { // not supported with warp interleaving
            __syncthreads();
            if (scheduler.exist_q_idx(q_idx + 1)) {
                load_q_g2s(q_idx + 1);
            }
            cp_async_fence();
        }
    };

    // Make `first` a compile-time constant via cute::true_type / cute::false_type
    // so the compiler can fold the first-iteration branch and avoid the runtime
    // check on the first iteration.
    auto mainloop = [&](auto first_ic) {
        constexpr bool first = decltype(first_ic)::value;
        if (thread_print) {
            printf("q_idx = %d, kv_idx_base = %d, wg = %d, read_stage = %d\n", q_idx, kv_idx_base, warp_group_idx, smem_pipe_read_kv);
        }

        if constexpr (WarpInterleaving) {
            __ppu_barrier_sync(5 + warp_group_id, MaxThreadsPerBlock);
        }

        // Read weights if current Q changes. On the first iteration always load Q;
        // afterwards only when kv_idx_base == 0 (first group of new q).
        if constexpr (first) {
            handle_q_change();
        } else {
            if (kv_idx_base == 0) {
                handle_q_change();
            }
        }

        // Issue next KV load (overlap with current compute)
        load_next_qk_g2s(false);

        // s2r KV + compute GEMM from current read stage
        constexpr int K_BLOCK_MAX = size<2>(tCrA);
        uint32_t actual_kv = kv_idx_base + warp_group_idx;
        for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
            copy(smem_tiled_copy_A, tCsA(_,_,k_block,smem_pipe_read_kv), tCrA_copy_view(_,_,k_block));
            cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrB(_,_,k_block), accum);
        });

        if constexpr (WarpInterleaving) {
            __ppu_barrier_arrive(6 - warp_group_id, MaxThreadsPerBlock, 0);
        }

        load_kv_scale_s2r();
        epilogue(q_idx, actual_kv);

        cp_async_wait<kNumKVStages - 2>();
        if constexpr (!WarpInterleaving) {
            __syncthreads();
        }
        // move to next kv_block
        smem_pipe_read_kv = (smem_pipe_read_kv + 1) % kNumKVStages;
        kv_idx_base = kv_idx_array[smem_pipe_read_kv];
        if (kv_idx_base == 0) q_idx++;

        clear(accum);
    };

    if (!scheduler.is_last_task(q_idx, kv_idx_base)) {
        mainloop(cute::true_type{});
    }
    while (!scheduler.is_last_task(q_idx, kv_idx_base)) {
        mainloop(cute::false_type{});
    }

    if constexpr (WarpInterleaving) {
        if (warp_group_id == 0) {
            __ppu_barrier_sync(5, MaxThreadsPerBlock);
        }
    }

    cp_async_wait<0>();
    __syncthreads();

  }

};

} // namespace cutlass::gemm::kernel


namespace deep_gemm {

template <typename ElementQK, typename ElementAcc, typename ElementLogits,
          uint32_t kNextN, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV>
class PagedAttention {

public:
    PagedAttention() = default;

    static uint32_t generate_id() {
        static uint32_t id = 0;
        return ++id;
    }

    static void run(const ElementQK * ptr_q,
                    const ElementQK * ptr_k,
                    const float * k_scales,
                    const float * weights,
                    const uint32_t batch_size,
                    const uint64_t logits_stride, const uint64_t kv_cache_stride_bytes, const uint32_t block_table_stride,
                    const uint32_t* context_lens, ElementLogits* logits,
                    const uint32_t* block_table, const uint32_t* schedule_meta,
                    hggcStream_t stream, int num_sms, int num_blocks) {

        using AttnKernel = cutlass::gemm::kernel::PPUPagedMqaLogits<ElementQK, ElementAcc, ElementLogits, kNextN, kNumHeads, kHeadDim, BLOCK_KV, kNumQStages, kNumKVStages, SPLIT_KV>;

        using StrideA = typename AttnKernel::StrideA;
        using StrideB = typename AttnKernel::StrideB;

        static constexpr int BLOCK_M = AttnKernel::BLOCK_M;
        static constexpr int BLOCK_N = AttnKernel::BLOCK_N;
        static constexpr int BLOCK_K = AttnKernel::BLOCK_K;
        static constexpr int WARP_M  = AttnKernel::WARP_M ;
        static constexpr int WARP_N  = AttnKernel::WARP_N ;

        auto SHAPE_M = BLOCK_M;
        auto SHAPE_N = BLOCK_N;
        auto SHAPE_K = kHeadDim;
        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape((int)SHAPE_M, (int)SHAPE_K, 1));
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape((int)SHAPE_N, (int)SHAPE_K, 1));
        int max_blocks_per_cu = compute_occupancy_for_kernel<AttnKernel>();

        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.cu_count = num_blocks;

        typename AttnKernel::Arguments arguments{ptr_q, ptr_k, k_scales, weights, batch_size, logits_stride, kv_cache_stride_bytes, block_table_stride,
                                                 context_lens, logits, block_table, schedule_meta, stride_A, stride_B, hw_info};
        auto params = arguments;
        dim3 const block = AttnKernel::get_block_shape();
        dim3 const grid = AttnKernel::get_grid_shape(params);
        int smem_size_kernel = AttnKernel::SharedStorageSize;

        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()){
            std::string data_type_str = "unknown";
            if (std::is_same_v<ElementQK, cutlass::bfloat16_t>) {
                data_type_str = "bf16";
            } else if (std::is_same_v<ElementQK, cutlass::float_e4m3_t>) {
                data_type_str = "fp8";
            } else if (std::is_same_v<ElementQK, int8_t>) {
                data_type_str = "int8";
            }

            dg_prof_params.set_paged_mqa_logits_params(data_type_str, batch_size, kNextN, kNumHeads, kHeadDim, reinterpret_cast<int*>(const_cast<uint32_t*>(context_lens)), stream);
            if constexpr (std::is_same_v<ElementLogits, __ppu_bfloat16>) {
                dg_prof_params.add_params("logits_dtype", std::string("bf16"));
            }
        }

        char *pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            hggcFuncAttributes attr;
            hggcFuncGetAttributes(&attr, cutlass::device_kernel<AttnKernel>);

            printf("[paged_mqa_logits:]\n");
            printf("kNumHeads:%d, kHeadDim:%d, kNextN:%d, BLOCK_KV:%d, SPLIT_KV:%d\n",
                kNumHeads, kHeadDim, kNextN, BLOCK_KV, SPLIT_KV);

            printf("ThreadblockShape[%d, %d], WarpShape[%d, %d], kNumQStages:%d, kNumKVStages:%d\n",
                BLOCK_M, BLOCK_N, WARP_M, WARP_N, kNumQStages, kNumKVStages);

            printf("num_sms:%d, max_blocks_per_cu:%d, threadblock_count:%d\n", num_sms, max_blocks_per_cu, num_blocks);
            printf("smem_size:%d, vreg:%d, stack:%d\n", smem_size_kernel, int(attr.numRegs), int(attr.localSizeBytes));
            std::cout << "block = " << block << std::endl;
            std::cout << "grid = " << grid << std::endl;
            if (num_sms * max_blocks_per_cu  != num_blocks) {
                printf("Warning: num_blocks(%d) should equal to num_sms(%d) * max_blocks_per_cu(%d) = %d\n", num_blocks, num_sms, max_blocks_per_cu, num_sms * max_blocks_per_cu);
            }
        }

        ProfilingInterface::Instance().instrument(true, dg_prof_params);
        cutlass::device_kernel<AttnKernel><<<grid, block, smem_size_kernel, stream>>>(params);
        ProfilingInterface::Instance().instrument(false, dg_prof_params);
    }
};

};  // namespace deep_gemm
