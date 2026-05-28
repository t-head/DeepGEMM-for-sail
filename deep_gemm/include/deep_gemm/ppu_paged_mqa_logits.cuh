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
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
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
        #pragma unroll
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
    #pragma unroll
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
                                      cudaStream_t stream) {
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
        while (current_kv_idx >= current_num_kv) {
            ++ current_q_idx;
            current_kv_idx = 0;
            if (current_q_idx < batch_size) {
                current_num_kv = ceil_div(__ldg(this->context_lens + current_q_idx), BLOCK_KV);
            } else {
                current_num_kv = 0;
                break;
            }
        }

        return true;
    }

    __device__ __forceinline__ bool exist_q_idx(const uint32_t& q_idx) const {
        return q_idx < end_q_idx or q_idx == end_q_idx and 0 < end_kv_idx;
    }

    // return 0 means no valid next q
    __device__ __forceinline__ uint32_t next_valid_q_idx(const uint32_t& q_idx) const {
        int next_q_idx = q_idx + 1;
        while (true) {
            if (!exist_q_idx(next_q_idx)) return 0;
            if (__ldg(this->context_lens + next_q_idx) != 0) return next_q_idx;
            next_q_idx++;
        }
        return 0;
    }

    __device__ __forceinline__ bool is_last_task(const uint32_t& q_idx, const uint32_t& kv_idx) const {
        return q_idx > end_q_idx or (q_idx == end_q_idx and kv_idx == end_kv_idx);
    }
};

template <typename ElementQK, typename ElementAcc,
          uint32_t kNextN, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV>
class Sm80PagedMqaLogits {
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
  static constexpr int BLOCK_M = SPLIT_KV;
  static constexpr int BLOCK_N = kNextN * kNumHeads;
  static constexpr int BLOCK_K = kHeadDim;
  static constexpr int WARP_M = 16;
  static constexpr int WARP_N = kNumHeads;

  static constexpr int BLOCK_Q = kNextN;
  static constexpr int WARP_Q = 1;
  static constexpr int kNumMathWarpGroups = 1;

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

  static constexpr int NumThreadsPerCTA = size(TiledMma{});
  // WarpInterleaving is enabled only when NumThreadsPerCTA is 512, which satisfy the condition that 2 warp group partitioned onto separate WEs.
  static constexpr bool WarpInterleaving = (NumThreadsPerCTA == 512);

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
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<kNumKVStages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<kNumQStages>{})));

  constexpr static uint32_t CTA_M = shape<0>(TileShape{});
  constexpr static uint32_t CTA_N = shape<1>(TileShape{});
  constexpr static uint32_t CTA_K = shape<2>(TileShape{});

  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  // ScaleA (k_scales, float per KV row) -- AIU load, no pre-rearrangement
  using DefaultOperandSFA =
      cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, float, false, _1, Int<CTA_M>, false, 0, false>;
  using SmemLayoutAtomSFA = typename DefaultOperandSFA::SmemLayoutAtom;
  using GmemTiledCopySFA = typename DefaultOperandSFA::GmemTiledCopy;
  using SmemLayoutSFA = decltype(tile_to_shape(
      SmemLayoutAtomSFA{},
      make_shape(_1{}, Int<CTA_M>{}, Int<kNumKVStages>{})));

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
    float* logits;
    const uint32_t* block_table;
    const uint32_t* schedule_meta;
    StrideA dA;
    StrideB dB;
    KernelHardwareInfo hw_info{};

    CUTLASS_DEVICE void
    print() const {
        printf("templates, kNumHeads=%d, kHeadDim=%d, tile=(%d, %d, %d, %d, %d, %d), kNextN=%d\n",
                kNumHeads, kHeadDim, BLOCK_M, BLOCK_N, WARP_M, WARP_N, kNumQStages, kNumKVStages, kNextN);
        printf("arguments, ptr_q=%p, ptr_k=%p, k_scales=%p, weights=%p, context_lens=%p, logits=%p, block_table=%p, schedule_meta=%p\n",
                ptr_q, ptr_k, k_scales, weights, context_lens, logits, block_table, schedule_meta);
        printf("arguments, batch_size=%d, logits_stride=%ld, kv_cache_stride_bytes=%ld, block_table_stride=%u\n",
                batch_size, logits_stride, kv_cache_stride_bytes, block_table_stride);
    }
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
    return dim3(params.hw_info.sm_count, 1, 1);
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
    // printf("run ppu aiu deepgemm persistent!!!");
    using namespace cute;
    using X = Underscore;

    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShape>::value);

    int warp_idx = canonical_warp_idx_sync();
    int thread_idx = int(threadIdx.x);

    // if (thread0()) {
    //   printf("EpilogueSharedStorage size = %d\n", sizeof(CollectiveEpilogue::SharedStorage));
    // }

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

    Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)
    // Partition the copying of A and B tiles across the threads
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    Tensor sSFA = make_tensor(make_smem_ptr(shared_storage.smem_k_scales.data()), SmemLayoutSFA{});
    Tensor sSFB = make_tensor(make_smem_ptr(shared_storage.smem_weight.data()), SmemLayoutSFB{});

    auto gmem_thr_copy_SFA = gmem_tiled_copy_SFA.get_thread_slice(thread_idx);
    auto gmem_thr_copy_SFB = gmem_tiled_copy_SFB.get_thread_slice(thread_idx);

    Tensor tSFAgSFA = gmem_thr_copy_SFA.partition_S(gSFA);
    Tensor tSFAsSFA = gmem_thr_copy_SFA.partition_D(sSFA);

    Tensor tSFBgSFB = gmem_thr_copy_SFB.partition_S(gSFB);
    Tensor tSFBsSFB = gmem_thr_copy_SFB.partition_D(sSFB);

    TiledMma tiled_mma;
    Tensor accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                     // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                     // (MMA,MMA_N,MMA_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                  // MMA_K

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(warp_idx * 32);
    Tensor tCsA            = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));                  // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                   // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(warp_idx * 32);
    Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));                  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    constexpr bool enable_print = false;
    bool thread_print = enable_print && cute::thread(0, 0);

    // Scheduler
    auto scheduler = PagedMQALogitsScheduler<BLOCK_KV, kNumMathWarpGroups>(params.batch_size, blockIdx.x, params.context_lens, params.schedule_meta);
    DG_STATIC_ASSERT(SPLIT_KV % BLOCK_KV == 0, "Unaligned SPLIT_KV");

    float weights[kNumHeads / 4];

    clear(accum);
    auto tKgK = tAgA;
    auto tSFKgSFK = tSFAgSFA;
    auto tQgQ = tBgB;
    auto tSFQgSFQ = tSFBgSFB;

    constexpr bool load_kv_scale = sizeof(ElementQK) == 1;
    const auto& lane_idx = get_lane_idx();
    const auto& warp_offset = (warp_idx % WarpOnM) * WARP_M;
    const auto& v_0_offset = lane_idx / 4 + 0;
    const auto& v_1_offset = lane_idx / 4 + 8;
    uint32_t warp_q_idx = warp_idx / WarpOnM;
    int warp_group_id = warp_idx / 8;

    uint32_t q_idx_array[kNumKVStages];
    uint32_t kv_idx_array[kNumKVStages];
    kv_idx_array[kNumKVStages - 1] = UINT32_MAX;
    uint32_t num_kv; // num_kv is not used
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
                           gmem_tiled_copy_SFA, tSFAgSFA(_,_,_,0), tSFAsSFA(_,_,_,smem_pipe_write_kv), warp_idx);
        } else {
            copy_aiu(gmem_tiled_copy_A, tAgA(_,_,_,0), tAsA(_,_,_,smem_pipe_write_kv), warp_idx);
        }
        if (thread_print) {
            printf("  copy_k q_idx = %d, kv_idx = %d, kv_offset = %d, stage = %d\n", q_idx, kv_idx, kv_offset, smem_pipe_write_kv);
        }
    };

    auto load_q_s2r = [&]() {
        copy(smem_tiled_copy_B, tCsB(_,_,_,smem_pipe_read_q), tCrB_copy_view);
        if (thread_print) {
            printf("    copy q to vreg, q_stage_idx = %d,\n", smem_pipe_read_q);
        }

        // Read weights
        float * smem_weights_staged = sSFB(_,_,smem_pipe_read_q).data().get() + warp_q_idx * kNumHeads;
        #pragma unroll
        for (uint32_t j = 0; j < kNumHeads / 4; ++ j) {
            #if __HGGC_ARCH__ == 150
                weights[j] = ld_shared(smem_weights_staged + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
            #else
                weights[j] = ld_shared(smem_weights_staged + (j / 2) * 8 + (j & 1) * 4 + lane_idx % 4);
            #endif
        }
        smem_pipe_read_q = (smem_pipe_read_q + 1) % kNumQStages;
    };

    auto load_next_qk_g2s = [&](bool load_q) {
        uint32_t& q_idx = q_idx_array[smem_pipe_write_kv];
        uint32_t& kv_idx = kv_idx_array[smem_pipe_write_kv];
        if (scheduler.fetch_next_task(q_idx, kv_idx, num_kv)) {
            if (load_q) load_q_g2s(q_idx);
            load_kv_g2s(q_idx, kv_idx);
        }
        smem_pipe_write_kv = (smem_pipe_write_kv + 1) % kNumKVStages;
        if (thread_print) printf("cp_async commit\n");
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

            // Reduce over the head dim and store
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

            if (thread_print) {
                printf("        sum[0] = %.4f, sum[1] = %f, sum[2] = %f, sum[3] = %f, sum[4] = %.4f, sum[5] = %f, sum[6] = %f, sum[7] = %f\n",
                    sum[0], sum[1], sum[2], sum[3], sum[4], sum[5], sum[6], sum[7]);
                printf("        v_0 = %.4f, q_idx = %d, warp_q_idx = %d, kMmaIterM = %d, kAccumStrideN16 = %d\n",
                    v_0, q_idx, warp_q_idx, kMmaIterM, kAccumStrideN16);
            }
        }
    };
    // load qk
    for (int i = 0; i < kNumKVStages - 1; i++) {
        load_next_qk_g2s(i == 0);
    }
    // wait AIU Q and first K
    cp_async_wait<kNumKVStages - 2>();
    __syncthreads();
    copy(smem_tiled_copy_A, tCsA(_,_,0,0), tCrA_copy_view(_,_,0));

    while (true) {
        // Get current Q and KV index
        const uint32_t& q_idx = q_idx_array[smem_pipe_read_kv];
        const uint32_t& kv_idx = kv_idx_array[smem_pipe_read_kv];

        if (scheduler.is_last_task(q_idx, kv_idx)) break;
        if (thread_print) {
            printf("q_idx = %d, kv_idx = %d\n", q_idx, kv_idx);
        }

        // Read weights if current Q changes
        if (kv_idx == 0 || kv_idx_array[kNumKVStages - 1] == UINT32_MAX) {
            if constexpr(kNumQStages > 1) {
                uint32_t next_q_idx = scheduler.next_valid_q_idx(q_idx);
                if (next_q_idx != 0) {
                    load_q_g2s(next_q_idx);
                }
                cp_async_fence();
            }
            load_q_s2r();
            if constexpr(kNumQStages == 1) {
                __syncthreads();
                uint32_t next_q_idx = scheduler.next_valid_q_idx(q_idx);
                if (next_q_idx != 0) {
                    load_q_g2s(next_q_idx);
                }
                cp_async_fence();
            }
        }
        load_next_qk_g2s(false);

        load_kv_scale_s2r();

        constexpr int K_BLOCK_MAX = size<2>(tCrA);
        for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
            auto k_block_next = (k_block + 1) % K_BLOCK_MAX;
            if (k_block_next == 0) {
                cp_async_wait<kNumKVStages - 2>();
                __syncthreads();
                smem_pipe_read_kv = (smem_pipe_read_kv + 1) % kNumKVStages;
                if (thread_print) {
                    printf("    copy kv to vreg, stage = %d,\n", smem_pipe_read_kv);
                }
            }
            copy(smem_tiled_copy_A, tCsA(_,_,k_block_next,smem_pipe_read_kv), tCrA_copy_view(_,_,k_block_next));
            cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrB(_,_,k_block), accum);
        });

        epilogue(q_idx, kv_idx);

        clear(accum);
    } // end of while loop

    cp_async_wait<0>();
    __syncthreads();

  }

};

} // namespace cutlass::gemm::kernel


namespace deep_gemm {

template <typename ElementQK, typename ElementAcc,
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
                    const uint32_t* context_lens, float* logits,
                    const uint32_t* block_table, const uint32_t* schedule_meta,
                    cudaStream_t stream, int num_sms, int num_blocks) {

        using AttnKernel = cutlass::gemm::kernel::Sm80PagedMqaLogits<ElementQK, ElementAcc, kNextN, kNumHeads, kHeadDim, BLOCK_KV, kNumQStages, kNumKVStages, SPLIT_KV>;

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
        hw_info.sm_count = num_blocks;

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
        }

        char *pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, cutlass::device_kernel<AttnKernel>);

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
