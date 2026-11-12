#pragma once
#include <hggc_runtime.h>
#include <deep_gemm/common/w4a8_arguments.cuh>
namespace deep_gemm::w4fa8_streamed {
// Convert the public padded token layout to contiguous expert rows. Both
// activations and row scales are regenerated on every invocation/Graph replay.
template <int K, int BlockM, int Groups> __device__ __forceinline__ void gather_body(GatherParams p, int group_start) {
  constexpr int VectorsPerRow = K / 16;
  constexpr int CopyRows = 64;
  static_assert(BlockM % CopyRows == 0);
  constexpr int CopiesPerBlock = BlockM / CopyRows;
  const int tasks = __ldg(p.aligned) * CopiesPerBlock;
  // Copy tasks stay small enough to fill the device without changing routing BlockM.
  for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
    const int block = task / CopiesPerBlock;
    const int copy_row = task % CopiesPerBlock * CopyRows;
    const uint4 expert = __ldg(reinterpret_cast<const uint4 *>(p.experts) + block);
    if (expert.x < uint32_t(group_start) || expert.x >= uint32_t(group_start + Groups))
      continue;
    const int row = (block - int(expert.z)) * BlockM + copy_row;
    const int valid = min(CopyRows, int(expert.y) - row);
    const int out_row = int(expert.w) + row;
    // A warp owns a row: share its routing lookup and keep both vector streams
    // contiguous, without a row/column division for every 16-byte copy.
    const int lane = threadIdx.x & 31;
    for (int local_row = threadIdx.x / 32; local_row < valid; local_row += blockDim.x / 32) {
      int token = lane == 0 ? __ldg(p.tokens + int64_t(block) * BlockM + copy_row + local_row) : 0;
      token = __shfl_sync(0xffffffffu, token, 0);
      if (uint32_t(token) < uint32_t(p.num_tokens)) {
        const auto src = reinterpret_cast<const uint4 *>(p.a) + int64_t(token) * VectorsPerRow;
        auto dst = reinterpret_cast<uint4 *>(p.grouped_a) + int64_t(out_row + local_row) * VectorsPerRow;
        for (int column = lane; column < VectorsPerRow; column += 32)
          dst[column] = __ldg(src + column);
        if (lane == 0)
          p.grouped_scale[out_row + local_row] = __ldg(p.a_scale + token);
      }
    }
  }
}

// W4A8 preprocess confines adjusted UE8M0 to [114,140], which maps every
// finite E2M1 magnitude exactly to a normal finite E5M2 value.
__device__ __forceinline__ uint32_t fp4_four_to_e5(uint32_t codes, uint32_t low, uint32_t high) {
  uint32_t magnitude = __byte_perm(low, high, codes & 0x7777u);
  uint32_t sign = __byte_perm(0u, 0x80808080u, (codes >> 1) & 0x7777u);
  return magnitude | sign;
}
template <int N, int K, int TileN, int Groups>
__device__ __forceinline__ void decode_tile_body(const uint8_t* __restrict__ packed, const uint8_t* __restrict__ scales,
                                                 uint8_t* __restrict__ output, int n_begin,
                                                 const int32_t* __restrict__ all_rows, int group_start) {
  constexpr int VectorsPerRow = K / 16;
  const int expert_step = min(Groups, int(gridDim.x));
  const int first_expert = blockIdx.x % expert_step, local_block = blockIdx.x / expert_step;
  const int expert_blocks = (int(gridDim.x) + expert_step - 1 - first_expert) / expert_step;
  for (int expert = first_expert; expert < Groups; expert += expert_step) {
    // Prune the complete expert before its K/N vector loop, including replay.
    if (__ldg(all_rows + group_start + expert) == 0)
      continue;
    for (int64_t vector = int64_t(local_block) * 256 + threadIdx.x; vector < int64_t(TileN) * VectorsPerRow;
         vector += int64_t(expert_blocks) * 256) {
      const int nr = int(vector / VectorsPerRow) + n_begin, kr = int(vector % VectorsPerRow) * 16;
      const int64_t scale_offset = int64_t(expert) * N * K / 32 + int64_t(nr / 64) * (K * 2) +
                                   (kr / 64) * 128 + 16 * (nr % 8) + 2 * ((nr % 64) / 8) + (kr / 32 % 2);
      const uint32_t bias = (uint32_t(__ldg(scales + scale_offset)) - 114u) * 0x04040404u;
      const uint32_t low = (0x0a080400u + bias) & 0xffffff00u, high = 0x12100e0cu + bias;
      const uint2 p = __ldg(reinterpret_cast<const uint2*>(packed + (int64_t(expert) * N + nr) * (K / 2) + kr / 2));
      const uint4 decoded = {fp4_four_to_e5(p.x, low, high), fp4_four_to_e5(p.x >> 16, low, high),
                             fp4_four_to_e5(p.y, low, high), fp4_four_to_e5(p.y >> 16, low, high)};
      *reinterpret_cast<uint4*>(output + (int64_t(expert) * TileN * VectorsPerRow + vector) * 16) = decoded;
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int prefix = 0, count = 0;
    for (int i = 0; i < group_start; ++i)
      prefix += all_rows[i];
    for (int i = 0; i < Groups; ++i)
      count += all_rows[group_start + i];
    int32_t* metadata = reinterpret_cast<int32_t*>(output + int64_t(Groups) * TileN * K);
    metadata[0] = prefix;
    metadata[1] = count;
  }
}
// Exact P41 mixed-sign expansion, isolated from the legacy partial-N decoder.
__device__ __forceinline__ uint32_t fp4_four_to_e5_rows32(uint32_t codes, uint32_t low, uint32_t high) {
  uint32_t magnitude = __byte_perm(low, high, codes & 0x7777u);
  uint32_t expanded = __byte_perm(codes, codes << 4, 0x1504u);
  return magnitude | (expanded & 0x80808080u);
}
// Full-N rows reuse one adjusted UE8M0 scale for two adjacent 16-byte outputs.
// The generated global wrapper supplies maxnreg32 without launch_bounds.
template <int N, int K, int TileN, int Groups, bool SkipFence = false>
__device__ __forceinline__ void decode_fulln_rows32_body(const uint8_t* __restrict__ packed,
                                                         const uint8_t* __restrict__ scales,
                                                         uint8_t* __restrict__ output, int n_begin,
                                                         const int32_t* __restrict__ all_rows, int group_start) {
  static_assert(TileN == N, "Flat full-N decoder requires a complete physical N range");
  static_assert(uint64_t(Groups) * N * K <= 0xffffffffull, "Partial decoded buffer must fit uint32 addressing");
  constexpr uint32_t VectorsPerRow = K / 32;
  constexpr uint32_t VectorCount = uint32_t(uint64_t(Groups) * N * K / 16);
  // With fullN, expert-major rows are already contiguous: no per-vector expert/N divmod.
  // n_begin is zero by the complete physical-N caller contract.
  constexpr uint32_t TotalRows = Groups * N;
  const uint32_t lane = threadIdx.x & 31u;
  for (uint32_t row = uint32_t(blockIdx.x) * 8u + threadIdx.x / 32u; row < TotalRows; row += uint32_t(gridDim.x) * 8u) {
    const uint32_t row_scale = (row / 64u) * uint32_t(K * 2) + 16u * (row % 8u) + 2u * ((row % 64u) / 8u);
    const uint32_t packed_base = row * uint32_t(K / 2), output_base = row * uint32_t(K);
#pragma unroll 1
    for (uint32_t chunk = 0; chunk < (VectorsPerRow + 31u) / 32u; ++chunk) {
      const uint32_t column = 32u * chunk + lane;
      if (column < VectorsPerRow) {
        const uint32_t scale_offset = row_scale + (column / 2u) * 128u + (column % 2u);
        const uint32_t bias = (uint32_t(__ldg(scales + scale_offset)) - 114u) * 0x04040404u;
        const uint32_t low = (0x0a080400u + bias) & 0xffffff00u, high = 0x12100e0cu + bias;
#pragma unroll
        for (uint32_t half = 0; half < 2; ++half) {
          uint2 p = __ldg(reinterpret_cast<const uint2*>(packed + packed_base + column * 16u + half * 8u));
          uint4 decoded = {fp4_four_to_e5_rows32(p.x, low, high), fp4_four_to_e5_rows32(p.x >> 16, low, high),
                           fp4_four_to_e5_rows32(p.y, low, high), fp4_four_to_e5_rows32(p.y >> 16, low, high)};
          *reinterpret_cast<uint4*>(output + output_base + column * 32u + half * 16u) = decoded;
          // SDK 2.2 lowers this legacy clobber to a block VMEM fence.
          // The selected short-call tail decoder omits it after measured validation.
          if constexpr (!SkipFence)
            asm volatile("" ::: "memory");
        }
      }
    }
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int prefix = 0, count = 0;
    for (int i = 0; i < group_start; ++i)
      prefix += all_rows[i];
    for (int i = 0; i < Groups; ++i)
      count += all_rows[group_start + i];
    int32_t* metadata = reinterpret_cast<int32_t*>(output + int64_t(Groups) * TileN * K);
    metadata[0] = prefix;
    metadata[1] = count;
  }
}

} // namespace deep_gemm::w4fa8_streamed

namespace deep_gemm::w4fa8_empty_flat {
template <int N, int K, int TileN, int Groups, int Threads, bool SkipFence>
__device__ __forceinline__ void decode_empty_flat_body(const uint8_t* __restrict__ packed,
                                                       const uint8_t* __restrict__ scales, uint8_t* __restrict__ output,
                                                       int n_begin, const int32_t* __restrict__ all_rows,
                                                       int group_start) {
  static_assert(TileN == N, "Flat full-N decoder requires a complete physical N range");
  static_assert(uint64_t(Groups) * N * K <= 0xffffffffull, "Partial decoded buffer must fit uint32 addressing");
  constexpr uint32_t VectorCount = uint32_t(uint64_t(Groups) * N * K / 16);
  static_assert(K % 128 == 0 && Threads % 32 == 0);
  // Each CTA is assigned to one expert; its row count is read once before
  // touching packed weights. Empty experts retain their untouched weight slot.
  const uint32_t vote_lane = threadIdx.x & 31u;
  uint32_t local_valid = 1;
  if (vote_lane == 0) {
#pragma unroll
    for (int expert = 0; expert < Groups; ++expert)
      local_valid &= uint32_t(__ldg(all_rows + group_start + expert) != 0);
  }
  const bool all_nonempty = __shfl_sync(0xffffffffu, local_valid, 0) != 0;
  constexpr uint32_t VectorsPerRow = K / 32;
  const uint32_t lane = threadIdx.x & 31u;
  for (uint32_t row = uint32_t(blockIdx.x) * 8u + threadIdx.x / 32u; row < uint32_t(Groups * N);
       row += uint32_t(gridDim.x) * 8u) {
    if (!all_nonempty && __ldg(all_rows + group_start + row / uint32_t(N)) == 0)
      continue;
    const uint32_t row_scale = (row / 64u) * uint32_t(K * 2) + 16u * (row % 8u) + 2u * ((row % 64u) / 8u);
    const uint32_t packed_base = row * uint32_t(K / 2), output_base = row * uint32_t(K);
#pragma unroll 1
    for (uint32_t chunk = 0; chunk < (VectorsPerRow + 31u) / 32u; ++chunk) {
      const uint32_t column = 32u * chunk + lane;
      if (column < VectorsPerRow) {
        const uint32_t scale_offset = row_scale + (column / 2u) * 128u + (column % 2u);
        const uint32_t bias = (uint32_t(__ldg(scales + scale_offset)) - 114u) * 0x04040404u;
        const uint32_t low = (0x0a080400u + bias) & 0xffffff00u, high = 0x12100e0cu + bias;
#pragma unroll
        for (uint32_t half = 0; half < 2; ++half) {
          uint2 p = __ldg(reinterpret_cast<const uint2*>(packed + packed_base + column * 16u + half * 8u));
          uint4 decoded = {w4fa8_streamed::fp4_four_to_e5_rows32(p.x, low, high),
                           w4fa8_streamed::fp4_four_to_e5_rows32(p.x >> 16, low, high),
                           w4fa8_streamed::fp4_four_to_e5_rows32(p.y, low, high),
                           w4fa8_streamed::fp4_four_to_e5_rows32(p.y >> 16, low, high)};
          *reinterpret_cast<uint4*>(output + output_base + column * 32u + half * 16u) = decoded;
          if constexpr (!SkipFence)
            asm volatile("" ::: "memory");
        }
      }
    }
  }
  // Metadata is always produced, including when expert zero is empty.
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    int prefix = 0, count = 0;
    for (int i = 0; i < group_start; ++i)
      prefix += all_rows[i];
    for (int i = 0; i < Groups; ++i)
      count += all_rows[group_start + i];
    int32_t* metadata = reinterpret_cast<int32_t*>(output + int64_t(Groups) * TileN * K);
    metadata[0] = prefix;
    metadata[1] = count;
  }
}
} // namespace deep_gemm::w4fa8_empty_flat

#include <deep_gemm/impls/int8_gemm_cutlass3.cuh>
namespace deep_gemm::w4fa8_streamed {
using namespace cute;
// The decoder already incorporates weight scales. Keep the stock pipeline and
// specialize its final scaling step without cloning the collective mainloop.
template <class Base, bool ApplyInvScale> struct UnitBScaleMainloop : Base {
  struct ScaledParams : Base::Params { float inv_scale = 1.f; };
  using Params = cute::conditional_t<ApplyInvScale, ScaledParams, typename Base::Params>;
  using Arguments = Params;
  float inv_scale = 1.f;
  template <class Shape> static Params to_underlying_arguments(Shape const&, Params const& p, void*) { return p; }
  using Policy = typename Base::DispatchPolicy;
  using ElementScale = typename Base::ElementScale;
  template <class Shape> CUTLASS_DEVICE UnitBScaleMainloop(Params p, Shape shape) : Base(p, shape) {
    if constexpr (ApplyInvScale) inv_scale = p.inv_scale;
  }
  template <class Src, class Dst>
  CUTLASS_DEVICE void dequant(Src& mma_acc, Dst& accum, int thread_idx, int smem_pipe_scale, char* smem_buf) {
    using namespace cute;
    auto& storage = *reinterpret_cast<typename Base::SharedStorage*>(smem_buf);
    auto sA = make_tensor(make_smem_ptr(storage.smem_a.data()), typename Base::SmemLayoutAInterleave{});
    auto gmem_thr = this->gmem_tiled_copy_A.get_slice(thread_idx);
    auto tAsA = gmem_thr.partition_D(sA);
    auto scale_ptr = reinterpret_cast<ElementScale*>(tAsA(_, _, _, smem_pipe_scale).data().get());
    auto sSA = make_tensor(make_smem_ptr(scale_ptr), typename Base::SmemLayoutScaleA{});
    typename Base::TiledMma mma;
    auto thr = mma.get_thread_slice(thread_idx);
    auto tCrSA = make_fragment_like<ElementScale>(thr.partition_fragment_C(sSA(_, _, Int<0>{})));
    using Atom =
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<cutlass::sizeof_bits<ElementScale>::value>, ElementScale>;
    auto copy_scale = make_tiled_copy_C(Atom{}, mma);
    auto thr_copy = copy_scale.get_thread_slice(thread_idx);
    auto src = thr_copy.partition_S(sSA);
    auto dst = thr_copy.retile_D(tCrSA);
    auto src_pipe = src(_, _, _, Int<0>{});
    cp_async_wait<Policy::Stages - 1>();
    __syncthreads();
    copy(copy_scale, src_pipe(_, _, Int<0>{}), dst(_, _, Int<0>{}));
    CUTE_UNROLL
    for (int n = 0; n < cute::size<2>(accum); ++n) {
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(accum); ++m) {
        CUTE_UNROLL
        for (int i = 0; i < cute::size<0>(accum); ++i) {
          float scale = tCrSA(i, m, Int<0>{});
          if constexpr (ApplyInvScale) scale *= inv_scale;
          accum(i, m, n) = mma_acc(i, m, n) * scale;
        }
      }
    }
  }
};
} // namespace deep_gemm::w4fa8_streamed
namespace cutlass::gemm::collective {
// Keep the stock mainloop construction, loading and parameter ABI. Only the
// operand-register schedule changes; a shared stage is recycled after its last read.
template <class Base, bool ApplyInvScale = false>
struct W4A8SingleRegisterMainloop : deep_gemm::w4fa8_streamed::UnitBScaleMainloop<Base, ApplyInvScale> {
  using Parent = deep_gemm::w4fa8_streamed::UnitBScaleMainloop<Base, ApplyInvScale>;
  using Params = typename Parent::Params;
  using DispatchPolicy = typename Base::DispatchPolicy;
  using TileShape = typename Base::TileShape;
  using SharedStorage = typename Base::SharedStorage;
  using ElementScale = typename Base::ElementScale;
  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutB = typename Base::SmemLayoutB;
  using SmemLayoutAInterleave = typename Base::SmemLayoutAInterleave;
  using SmemLayoutBInterleave = typename Base::SmemLayoutBInterleave;
  using SmemLayoutScaleA = typename Base::SmemLayoutScaleA;
  using TiledMma = typename Base::TiledMma;
  using SmemCopyAtomA = typename Base::SmemCopyAtomA;
  using SmemCopyAtomB = typename Base::SmemCopyAtomB;
  using TransformA = typename Base::TransformA;
  using TransformB = typename Base::TransformB;
  using Base::gmem_tiled_copy_A;
  using Base::gmem_tiled_copy_B;
  using Base::gmem_tiled_copy_scaleA;
  static constexpr bool SplitAIU = Base::SplitAIU;
  template <class Shape> CUTLASS_DEVICE W4A8SingleRegisterMainloop(Params p, Shape shape) : Parent(p, shape) {}
  template <class... Ts, class FrgTensorD, class FrgTensorC, class ResidueMNK>
  CUTLASS_DEVICE void operator()(FrgTensorD& accum, cute::tuple<Ts...> const& load_inputs, FrgTensorC const& src_accum,
                                 int k_tile_iter, int k_tile_count, ResidueMNK residue_mnk, int thread_idx,
                                 int smem_pipe_write, int smem_pipe_scale, char* smem_buf) {
    using namespace cute;
    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 3, "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");
    static_assert(rank(SmemLayoutB{}) == 3, "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");
    int warp_idx = canonical_warp_idx_sync();
    Tensor gA = get<0>(load_inputs);
    Tensor gB = get<1>(load_inputs);
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutAInterleave{});
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_a.data() + size<0>(TileShape{}) * size<2>(TileShape{})),
                            SmemLayoutBInterleave{});
    CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(sA));
    CUTE_STATIC_ASSERT_V(size<1>(gA) == size<1>(sA));
    CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sB));
    CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sB));
    CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);
    Tensor gSFA = get<2, 0>(load_inputs);
    Tensor cSFA = get<2, 1>(load_inputs);
    ElementScale* scale_a_smem_ptr = reinterpret_cast<ElementScale*>(tAsA(_, _, _, smem_pipe_scale).data().get());
    Tensor sSA = make_tensor(make_smem_ptr(scale_a_smem_ptr), SmemLayoutScaleA{});
    auto gmem_thr_copy_scaleA = gmem_tiled_copy_scaleA.get_slice(thread_idx);
    Tensor tSgSA = gmem_thr_copy_scaleA.partition_S(gSFA);
    Tensor tSsSA = gmem_thr_copy_scaleA.partition_D(sSA);
    Tensor tSFAcSFA = gmem_thr_copy_scaleA.partition_S(cSFA);
    Tensor tSpSA = make_tensor<bool>(shape(tSsSA));
    for (int i = 0; i < size(tSpSA); ++i) {
      tSpSA(i) = get<0>(tSFAcSFA(i)) < get<0>(residue_mnk);
    }
    bool scaleCopySendDone = false;
    if (k_tile_count > 0) {
      copy_aiu<SplitAIU>(gmem_tiled_copy_A, tAgA(_, _, _, k_tile_iter), tAsA(_, _, _, smem_pipe_write),
                         gmem_tiled_copy_B, tBgB(_, _, _, k_tile_iter), tBsB(_, _, _, smem_pipe_write), warp_idx);
    }
    cp_async_fence();
    --k_tile_count;
    ++k_tile_iter;
    ++smem_pipe_write;
    smem_pipe_write = (smem_pipe_write == DispatchPolicy::Stages) ? 0 : smem_pipe_write;
    int smem_pipe_read = smem_pipe_write;
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(src_accum));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(src_accum));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(warp_idx * 32);
    Tensor tCsA = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));
    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(warp_idx * 32);
    Tensor tCsB = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));
    Tensor tCsA_p = tCsA(_, _, _, smem_pipe_read);
    Tensor tCsB_p = tCsB(_, _, _, smem_pipe_read);
    auto K_BLOCK_MAX = size<2>(tCrA_copy_view);
    if (K_BLOCK_MAX > 1) {
      cp_async_wait<DispatchPolicy::Stages - 1>();
      __syncthreads();
    }
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<2>(tCrB) == size<2>(tCrB_copy_view));
    auto process_kblock_iterations = [&](int k_block) {
      copy(smem_tiled_copy_A, tCsA_p(_, _, k_block), tCrA_copy_view(_, _, Int<0>{}));
      copy(smem_tiled_copy_B, tCsB_p(_, _, k_block), tCrB_copy_view(_, _, Int<0>{}));
      if (k_block == K_BLOCK_MAX - 1) {
        cp_async_wait<DispatchPolicy::Stages - 2>();
        __syncthreads();
        if (k_tile_count > 0) {
          copy_aiu<SplitAIU>(gmem_tiled_copy_A, tAgA(_, _, _, k_tile_iter), tAsA(_, _, _, smem_pipe_write),
                             gmem_tiled_copy_B, tBgB(_, _, _, k_tile_iter), tBsB(_, _, _, smem_pipe_write), warp_idx);
        } else if (!scaleCopySendDone) {
          copy_if(gmem_tiled_copy_scaleA, tSpSA, tSgSA(_, _, _, 0), tSsSA(_, _, _, 0));
          scaleCopySendDone = true;
        }
        cp_async_fence();
        --k_tile_count;
        ++k_tile_iter;
        ++smem_pipe_read;
        smem_pipe_read = smem_pipe_read == DispatchPolicy::Stages ? 0 : smem_pipe_read;
        smem_pipe_write = smem_pipe_read;
        tCsA_p = tCsA(_, _, _, smem_pipe_read);
        tCsB_p = tCsB(_, _, _, smem_pipe_read);
      }
      cute::transform(tCrA(_, _, Int<0>{}), TransformA{});
      cute::transform(tCrB(_, _, Int<0>{}), TransformB{});
      cute::gemm(tiled_mma, accum, tCrA(_, _, Int<0>{}), tCrB(_, _, Int<0>{}), accum);
    };
    for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) { process_kblock_iterations(k_block); });
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > -(DispatchPolicy::Stages)) {
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) { process_kblock_iterations(k_block); });
    }
  }
};
} // namespace cutlass::gemm::collective

namespace deep_gemm::w4fa8_streamed {
using namespace cute;
template <int OutputN, int K, int TileN, int Groups, int BM = kBlockM, int BN = 256, int WM = 64, int WN = 64,
          int Stages = 4, bool ApplyInvScale = false>
struct CoreTraits {
  using Arch = cutlass::arch::PPU0015;
  using A = cutlass::float_e4m3_t;
  using B = cutlass::float_e5m2_t;
  using D = cutlass::bfloat16_t;
  using Row = cutlass::layout::RowMajor;
  using Col = cutlass::layout::ColumnMajor;
  using Tile = Shape<Int<BM>, Int<BN>, _128>;
  using MMA = cute::PPU0015_16x16x32_F32E4M3E5M2F32_TN;
  using TM = TiledMMA<MMA_Atom<MMA>, Layout<Shape<Int<BM / WM>, Int<BN / WN>, _1>>,
                      cute::Tile<Int<(BM / WM) * 16>, Int<(BN / WN) * 16>, _32>>;
  using OA = cutlass::gemm::config::DefaultGemm_AIU_Operand<Arch, A, false, Int<BM>, _128, false, (BM + BN) * 128>;
  using OB = cutlass::gemm::config::DefaultGemm_AIU_Operand<Arch, B, false, Int<BN>, _128, true, (BM + BN) * 128>;
  using MLBase = cutlass::gemm::collective::CollectiveMma<
      Arch, cutlass::gemm::MainloopPPUAiuA8W8OverlapPrologue<Stages, ::KernelAiuMultistageOverlapPrologue>, Tile, A,
      cutlass::detail::TagToStrideA_t<Row>, B, cutlass::detail::TagToStrideB_t<Col>, TM, typename OA::GmemTiledCopy,
      typename OA::SmemLayoutAtom, typename OA::SmemCopyAtom, cute::identity, typename OB::GmemTiledCopy,
      typename OB::SmemLayoutAtom, typename OB::SmemCopyAtom, cute::identity>;
  using ML = cutlass::gemm::collective::W4A8SingleRegisterMainloop<MLBase, ApplyInvScale>;
  using EP = cutlass::epilogue::collective::DefaultEpilogueNoTsm<
      cutlass::detail::TagToStrideA_t<Row>, cutlass::detail::TagToStrideA_t<Row>,
      cutlass::epilogue::thread::LinearCombination<D, 2, float, float, cutlass::epilogue::thread::ScaleType::Nothing>,
      cutlass::gemm::EpilogueDefault, (TileN % BN == 0)>;
  using BaseScheduler = deep_gemm::DeepGemmScheduler<deep_gemm::GemmType::MGroupedNoPad, TileN, K, BM, BN, Groups,
                                                     (TileN + BN - 1) / BN, 2, deep_gemm::EpilogueType::Default, false>;
  using SchedulerArguments = GroupedRowRangeArguments;
  struct SchedulerStorage {
    typename BaseScheduler::Params base_params;
    int32_t row_prefix;
    __device__ explicit SchedulerStorage(SchedulerArguments const& a)
        : base_params{uint32_t(a.row_range[1]), a.grouped_layout}, row_prefix(a.row_range[0]) {}
  };
  // The first base owns the Params object referenced by the second base.
  // Construction order guarantees its lifetime throughout the persistent loop.
  struct Scheduler : SchedulerStorage, BaseScheduler {
    using Arguments = SchedulerArguments;
    using Params = Arguments;
    __device__ explicit Scheduler(Arguments const& a)
        : SchedulerStorage(a), BaseScheduler(SchedulerStorage::base_params) {}
    __device__ __forceinline__ int64_t curr_offset_a() const {
      return BaseScheduler::curr_offset_a() + int64_t(this->row_prefix) * K;
    }
    __device__ __forceinline__ int64_t curr_offset_m() const {
      return BaseScheduler::curr_offset_m() + int64_t(this->row_prefix);
    }
    __device__ __forceinline__ int64_t curr_offset_c() const {
      return (BaseScheduler::curr_offset_a() / K + this->row_prefix) * int64_t(OutputN);
    }
  };
  using Kernel = cutlass::gemm::kernel::DeepGemmUniversal<Shape<int, int, int, int>, ML, EP, Scheduler, false>;
};
template <int N, int K, int TileN, int Groups, int BM = kBlockM, int BN = 256, int WM = 64, int WN = 64, int Stages = 4>
__global__ __launch_bounds__(
    CoreTraits<N, K, TileN, Groups, BM, BN, WM, WN, Stages>::Kernel::MaxThreadsPerBlock,
    1) void mixed_core(typename CoreTraits<N, K, TileN, Groups, BM, BN, WM, WN, Stages>::Kernel::Params params) {
  extern __shared__ char smem[];
  using Kernel = typename CoreTraits<N, K, TileN, Groups, BM, BN, WM, WN, Stages>::Kernel;
  Kernel{}(params, smem);
}
} // namespace deep_gemm::w4fa8_streamed
