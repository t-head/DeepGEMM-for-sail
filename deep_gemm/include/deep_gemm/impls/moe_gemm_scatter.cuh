#pragma once

#include <stdint.h>
#include <deep_gemm/impls/int8_gemm_cutlass3.cuh>
#include <ppu_include.hpp>

// BF16 routing and overlap epilogue shared by both FP8 weight layouts.
namespace deep_gemm::mega_moe::detail {

__device__ __forceinline__ uint4 scatter_load_bf16x8(
    const float4* ptr) {
#if __HGGC_ARCH__ == 150
    return __ppu_global_load_bulk_b32x4((const void*)ptr);
#else
    return *reinterpret_cast<const uint4*>(ptr);
#endif
}

__device__ __forceinline__ uint4 scatter_load_bf16x8_shared(
    const uint4* ptr) {
#if __HGGC_ARCH__ == 150
    return __ppu_shared_load_bulk_b32x4((const void*)ptr);
#else
    return *ptr;
#endif
}

} // namespace deep_gemm::mega_moe::detail

namespace deep_gemm::mega_moe::detail {
template <
    bool kUsePaddedValidRows = false,
    bool kCheckPackedRoutes = true>
struct Bf16MoeScatter {
  static constexpr bool kEnabled = true;
  static constexpr bool kUsesPaddedValidRows =
      kUsePaddedValidRows;
  static constexpr bool kSharedFromAccumulators = true;

  struct Arguments {
    void* combine_base{nullptr};
    uint32_t hidden_bytes{0};
    uint32_t max_tokens_per_rank{0};
    uint32_t num_topk{0};
    const int64_t* packed_src_info{nullptr};
    const int32_t* valid_m_rows{nullptr};
    const int64_t* peer_offsets{nullptr};
    uint32_t rank_idx{0};
    uint32_t num_ranks{0};
    const int32_t* row_group_indices{nullptr};
    uint32_t num_groups{0};
  };
  using Params = Arguments;

  static Params to_underlying_arguments(Arguments const& args) {
    return args;
  }

  CUTLASS_DEVICE
  static uint32_t get_valid_m(
      Params const& params, uint32_t group_idx, uint32_t fallback_m) {
    if constexpr (!kUsePaddedValidRows) {
      return fallback_m;
    } else {
      if (params.valid_m_rows == nullptr ||
          group_idx >= params.num_groups) {
        return 0;
      }
      const int32_t signed_valid_m =
          __ldg(params.valid_m_rows + group_idx);
      if (signed_valid_m <= 0) {
        return 0;
      }
      const uint32_t valid_m =
          static_cast<uint32_t>(signed_valid_m);
      return valid_m < fallback_m ? valid_m : fallback_m;
    }
  }

  CUTLASS_DEVICE
  static uint32_t get_overlap_valid_m(
      Params const& params, uint32_t pool_row_base,
      uint32_t row_start, uint32_t fallback_m) {
    if constexpr (!kUsePaddedValidRows) {
      return fallback_m;
    } else {
      if (params.row_group_indices == nullptr ||
          params.valid_m_rows == nullptr) {
        return 0;
      }
      const int32_t group_idx =
          __ldg(params.row_group_indices + pool_row_base);
      if (group_idx < 0 ||
          static_cast<uint32_t>(group_idx) >= params.num_groups) {
        return 0;
      }
      const int32_t signed_group_valid_m =
          __ldg(params.valid_m_rows + group_idx);
      if (signed_group_valid_m <= 0) {
        return 0;
      }
      const uint32_t group_valid_m =
          static_cast<uint32_t>(signed_group_valid_m);
      if (row_start >= group_valid_m) {
        return 0;
      }
      const uint32_t remaining = group_valid_m - row_start;
      return remaining < fallback_m ? remaining : fallback_m;
    }
  }

  CUTLASS_DEVICE
  static int64_t peer_offset(
      Params const& params, uint32_t rank) {
    return params.peer_offsets != nullptr
        ? __ldg(params.peer_offsets + rank)
        : 0;
  }

  // Use only for values shared by every active lane in a warp. Explicit
  // scalarization keeps routed-scatter metadata out of GEMM vector registers.
  CUTLASS_DEVICE
  static uint32_t warp_uniform(uint32_t value) {
#if __HGGC_ARCH__ == 150
    return __ppu_read_firstlane(value);
#else
    return value;
#endif
  }

  CUTLASS_DEVICE
  static __ppu_bfloat16* destination_row(
      Params const& params, uint32_t topk_idx,
      uint32_t token_idx) {
    const uint64_t combine_row =
        static_cast<uint64_t>(topk_idx) *
            params.max_tokens_per_rank +
        token_idx;
    return reinterpret_cast<__ppu_bfloat16*>(
        static_cast<uint8_t*>(params.combine_base) +
        combine_row * params.hidden_bytes);
  }

  template <uint32_t kThreadNum>
  CUTLASS_DEVICE
  static void prefetch_routes(
      Params const& params, uint32_t pool_row_base, uint32_t valid_m) {
    if (params.packed_src_info == nullptr) return;
    // Use the same 128-byte spacing as the grouped-layout KSD prefetch.
    constexpr uint32_t kRoutesPerHint = 128 / sizeof(int64_t);
    for (uint32_t row = warp_uniform(uint32_t(threadIdx.x) / 32) * kRoutesPerHint;
         row < valid_m; row += (kThreadNum / 32) * kRoutesPerHint)
      __ppu_prefetch_KSD(static_cast<void*>(const_cast<int64_t*>(
          params.packed_src_info + warp_uniform(pool_row_base) + row)));
  }

  template <int kBlockM, int kBlockN, uint32_t kThreadNum>
  CUTLASS_DEVICE
  static void scatter(
      Params const& params, const __ppu_bfloat16* tile_d,
      uint32_t valid_m, uint32_t n_start,
      uint32_t shape_n, uint32_t pool_row_base) {
    static_assert(kBlockN % 8 == 0,
                  "BF16 scatter requires eight-element vectors");
    if (params.packed_src_info == nullptr) return;

    __syncthreads();
    constexpr uint32_t kVec = 8;
    constexpr uint32_t kNVecs = kBlockN / kVec;
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t lane = tid % 32;
    for (uint32_t row = tid / 32; row < valid_m;
         row += kThreadNum / 32) {
      const int64_t packed =
          params.packed_src_info[pool_row_base + row];
      const uint32_t packed_low = static_cast<uint32_t>(packed);
      const uint32_t rank = static_cast<uint32_t>(packed >> 32);
      const uint32_t token_idx = packed_low & 0x00FFFFFF;
      const uint32_t topk_idx = packed_low >> 24;
      if constexpr (kCheckPackedRoutes) {
        if (rank >= params.num_ranks ||
            topk_idx >= params.num_topk ||
            token_idx >= params.max_tokens_per_rank) {
          continue;
        }
      }
      auto* destination = destination_row(
          params, topk_idx, token_idx);
      auto* dst_base = reinterpret_cast<uint4*>(
          peer_offset(params, rank) +
          reinterpret_cast<int64_t>(destination + n_start));
      const auto* src_base = reinterpret_cast<const float4*>(
          tile_d + row * shape_n + n_start);

      CUTLASS_PRAGMA_UNROLL
      for (uint32_t vector = lane;
           vector < kNVecs; vector += 32) {
        const uint4 payload =
            mega_moe::detail::scatter_load_bf16x8(
                src_base + vector);
#if __HGGC_ARCH__ == 150
        if (rank != params.rank_idx) {
          __ppu_remote_store_bulk_ncp_b32x4(
              static_cast<void*>(dst_base + vector), payload);
        } else {
          __ppu_global_store_bulk_b32x4(
              static_cast<void*>(dst_base + vector), payload);
        }
#else
        dst_base[vector] = payload;
#endif
      }
    }
  }

  template <int kBlockM, int kBlockN, uint32_t kThreadNum>
  CUTLASS_DEVICE
  static void scatter_staged_bf16(
      Params const& params, const __ppu_bfloat16* shared_tile,
      uint32_t valid_m, uint32_t n_start,
      uint32_t pool_row_base) {
    static_assert(
        kSharedFromAccumulators,
        "staged BF16 scatter requires shared epilogue mode");
    static_assert(kBlockN % 8 == 0,
                  "BF16 scatter requires eight-element vectors");
    if (params.packed_src_info == nullptr) return;

    constexpr uint32_t kVec = 8;
    constexpr uint32_t kNVecs = kBlockN / kVec;
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t lane = tid % 32;
    pool_row_base = warp_uniform(pool_row_base);
    for (uint32_t row = warp_uniform(tid / 32); row < valid_m;
         row += kThreadNum / 32) {
      const int64_t packed =
          __ldg(params.packed_src_info + pool_row_base + row);
      const uint32_t packed_low = warp_uniform(static_cast<uint32_t>(packed));
      const uint32_t rank = warp_uniform(static_cast<uint32_t>(packed >> 32));
      const uint32_t token_idx = packed_low & 0x00FFFFFF;
      const uint32_t topk_idx = packed_low >> 24;
      if constexpr (kCheckPackedRoutes) {
        if (rank >= params.num_ranks ||
            topk_idx >= params.num_topk ||
            token_idx >= params.max_tokens_per_rank) {
          continue;
        }
      }
      auto* destination = destination_row(
          params, topk_idx, token_idx);
      auto* dst_base = reinterpret_cast<uint4*>(
          peer_offset(params, rank) +
          reinterpret_cast<int64_t>(destination + n_start));
      const auto* src_base = reinterpret_cast<const uint4*>(
          shared_tile + row * kBlockN);

      CUTLASS_PRAGMA_UNROLL
      for (uint32_t vector = lane;
           vector < kNVecs; vector += 32) {
        const uint4 payload =
            mega_moe::detail::scatter_load_bf16x8_shared(
                src_base + vector);
#if __HGGC_ARCH__ == 150
        if (rank != params.rank_idx) {
          __ppu_remote_store_bulk_ncp_b32x4(
              static_cast<void*>(dst_base + vector), payload);
        } else {
          __ppu_global_store_bulk_b32x4(
              static_cast<void*>(dst_base + vector), payload);
        }
#else
        dst_base[vector] = payload;
#endif
      }
    }
  }

  // Every producer warp drains its stores before this kernel can complete.
  // Cross-rank completion is handled exclusively by barrier_reduce.
  CUTLASS_DEVICE
  static void complete_stores() {
    __ppu_wait_remote_vstcnt(0);
    __ppu_wait_vstcnt(0);
  }
};

template <class MainloopSharedStorage_, class BaseEpilogueSharedStorage_,
          uint32_t kBf16Elements>
struct alignas(128) MoeOverlapBf16ScratchStorage {
  using MainloopSharedStorage = MainloopSharedStorage_;
  using BaseEpilogueSharedStorage = BaseEpilogueSharedStorage_;

  union {
    MainloopSharedStorage mainloop_lifetime;
    BaseEpilogueSharedStorage base_epilogue;
    // The read-only overlap kernel names this member in an unevaluated
    // sizeof expression. Keep it in the prefix union at no extra cost.
    uint8_t smem_epilogue[1];
  };
  alignas(128) __ppu_bfloat16 scatter_tile[kBf16Elements];
};

template <class MainloopSharedStorage_, class BaseEpilogueSharedStorage_>
struct MoeOverlapGlobalDStorage {
  using MainloopSharedStorage = MainloopSharedStorage_;
  using BaseEpilogueSharedStorage = BaseEpilogueSharedStorage_;

  union {
    MainloopSharedStorage mainloop_lifetime;
    BaseEpilogueSharedStorage base_epilogue;
    uint8_t smem_epilogue[1];
  };
};

// Scatter-aware overlap epilogue using the GEMM mainloop's parameter ABI.
template <
    class BaseCollectiveEpilogue_,
    class MainloopSharedStorage_,
    class GemmScatter_,
    class TileShape_,
    uint32_t kThreadCount,
    uint32_t kShapeN>
class MoeOverlapScatterEpilogue {
public:
  using BaseCollectiveEpilogue = BaseCollectiveEpilogue_;
  using MainloopSharedStorage = MainloopSharedStorage_;
  using GemmScatter = GemmScatter_;
  using TileShape = TileShape_;

  using EpilogueSchedule =
      typename BaseCollectiveEpilogue::EpilogueSchedule;
  using DispatchPolicy =
      typename BaseCollectiveEpilogue::DispatchPolicy;
  using ThreadEpilogueOp =
      typename BaseCollectiveEpilogue::ThreadEpilogueOp;
  using ElementOutput =
      typename BaseCollectiveEpilogue::ElementOutput;
  using FragmentAccumulator =
      typename BaseCollectiveEpilogue::FragmentAccumulator;
  using ElementAccumulator =
      typename BaseCollectiveEpilogue::ElementAccumulator;
  using ElementCompute =
      typename BaseCollectiveEpilogue::ElementCompute;
  using ElementScalar =
      typename BaseCollectiveEpilogue::ElementScalar;
  using ElementC = typename BaseCollectiveEpilogue::ElementC;
  using StrideC = typename BaseCollectiveEpilogue::StrideC;
  using ElementD = typename BaseCollectiveEpilogue::ElementD;
  using StrideD = typename BaseCollectiveEpilogue::StrideD;
  using GmemTiledCopyC =
      typename BaseCollectiveEpilogue::GmemTiledCopyC;
  using GmemTiledCopyD =
      typename BaseCollectiveEpilogue::GmemTiledCopyD;
  using BaseSharedStorage =
      typename BaseCollectiveEpilogue::SharedStorage;

  static constexpr uint32_t BlockM =
      CUTE_STATIC_V(cute::size<0>(TileShape{}));
  static constexpr uint32_t BlockN =
      CUTE_STATIC_V(cute::size<1>(TileShape{}));
  static constexpr uint32_t Bf16Elements = BlockM * BlockN;
  static constexpr uint32_t Bf16ScratchBytes =
      Bf16Elements * sizeof(__ppu_bfloat16);
  static constexpr uint32_t PpuSharedMemoryCapacity = 256 * 1024;

  using ScratchCandidate = MoeOverlapBf16ScratchStorage<
      MainloopSharedStorage, BaseSharedStorage, Bf16Elements>;
  using GlobalDStorage = MoeOverlapGlobalDStorage<
      MainloopSharedStorage, BaseSharedStorage>;

  static constexpr bool kUsesDisjointBf16Scratch =
      GemmScatter::kSharedFromAccumulators &&
      sizeof(ScratchCandidate) <= PpuSharedMemoryCapacity;
  using SharedStorage = cute::conditional_t<
      kUsesDisjointBf16Scratch, ScratchCandidate, GlobalDStorage>;
  using TensorStorage = SharedStorage;

  static_assert(
      sizeof(__ppu_bfloat16) == 2,
      "channelwise overlap scratch must contain BF16 elements");
  static_assert(
      Bf16ScratchBytes == BlockM * BlockN * 2,
      "channelwise overlap scratch must be exactly BM*BN*2 bytes");
  static_assert(
      BlockM != 128 || BlockN != 256 ||
          Bf16ScratchBytes == 65536,
      "BM128/BN256 overlap scratch must be exactly 65536 bytes");
  static_assert(
      __builtin_offsetof(ScratchCandidate, scatter_tile) >=
          sizeof(MainloopSharedStorage),
      "overlap BF16 scratch must begin after mainloop shared storage");
  static_assert(
      __builtin_offsetof(ScratchCandidate, scatter_tile) % 128 == 0,
      "overlap BF16 scratch must be 128-byte aligned");
  static_assert(
      sizeof(decltype(ScratchCandidate::scatter_tile)) ==
          Bf16ScratchBytes,
      "overlap BF16 scratch array has an unexpected size");
  static_assert(
      !kUsesDisjointBf16Scratch ||
          sizeof(SharedStorage) <= PpuSharedMemoryCapacity,
      "overlap mainloop plus BF16 scratch exceeds 256 KiB");

  struct Arguments {
    typename ThreadEpilogueOp::Params thread{};
    ElementC const* ptr_C{nullptr};
    StrideC dC{};
    ElementD* ptr_D{nullptr};
    StrideD dD{};
    typename GemmScatter::Arguments scatter{};
    ElementD* base_ptr_D{nullptr};
  };

  struct Params {
    typename ThreadEpilogueOp::Params thread{};
    ElementC const* ptr_C{nullptr};
    StrideC dC{};
    ElementD* ptr_D{nullptr};
    StrideD dD{};
    typename GemmScatter::Params scatter{};
    ElementD* base_ptr_D{nullptr};
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const&, Arguments const& args, void*) {
    return {
        args.thread,
        args.ptr_C,
        args.dC,
        args.ptr_D,
        args.dD,
        GemmScatter::to_underlying_arguments(args.scatter),
        args.base_ptr_D};
  }

  template <class ProblemShape>
  static size_t get_workspace_size(
      ProblemShape const&, Arguments const&) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const&, Arguments const&, void*,
      hggcStream_t, cutlass::HostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template <class ProblemShape>
  static bool can_implement(
      ProblemShape const&, Arguments const&) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  MoeOverlapScatterEpilogue(
      Params const& params, SharedStorage& shared_storage)
      : params_(params), shared_storage_(shared_storage) {}

  CUTLASS_DEVICE
  bool is_source_needed() {
    return false;
  }

  template <
      class ProblemShapeMNKL,
      class BlockShapeMNK,
      class BlockCoordMNKL,
      class FrgEngine,
      class FrgLayout,
      class TiledMma,
      class ResidueMNK>
  CUTLASS_DEVICE
  void operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK block_shape_mnk,
      BlockCoordMNKL block_coord_mnkl,
      cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char*) {
    using namespace cute;

    const uint32_t m_block_idx =
        static_cast<uint32_t>(get<0>(block_coord_mnkl));
    const uint32_t n_block_idx =
        static_cast<uint32_t>(get<1>(block_coord_mnkl));
    const uint32_t row_start = m_block_idx * BlockM;
    const uint32_t n_start = n_block_idx * BlockN;
    const uint32_t residue_m =
        static_cast<uint32_t>(get<0>(residue_mnk));
    uint32_t valid_m = cute::min(BlockM, residue_m);

    uint32_t pool_row_base = row_start;
    if constexpr (GemmScatter::kEnabled) {
      const auto group_output_elements =
          params_.ptr_D - params_.base_ptr_D;
      const uint32_t group_row_base =
          static_cast<uint32_t>(
              group_output_elements /
              static_cast<decltype(group_output_elements)>(kShapeN));
      pool_row_base += group_row_base;
      valid_m = GemmScatter::get_overlap_valid_m(
          params_.scatter, pool_row_base, row_start, valid_m);
    }

    if constexpr (kUsesDisjointBf16Scratch) {
      // The next tile prologue may still be writing mainloop shared memory.
      // Only the disjoint tail allocation below is legal for this tile.
      auto* shared_tile = shared_storage_.scatter_tile;
      auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
      auto identity = make_identity_tensor(
          make_shape(Int<BlockM>{}, Int<BlockN>{}));
      auto accumulator_coords = thread_mma.partition_C(identity);
      auto shared_output = make_tensor(
          make_smem_ptr(shared_tile),
          make_layout(
              make_shape(Int<BlockM>{}, Int<BlockN>{}),
              make_stride(Int<BlockN>{}, _1{})));
      auto thread_shared_output =
          thread_mma.partition_C(shared_output);
      CUTE_STATIC_ASSERT_V(
          size(accumulator_coords) == size(accumulators));
      CUTE_STATIC_ASSERT_V(
          size(thread_shared_output) == size(accumulators));

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accumulators); ++i) {
        const auto coord = accumulator_coords(i);
        if (elem_less(
                coord,
                make_coord(
                    get<0>(residue_mnk),
                    get<1>(residue_mnk)))) {
          const __ppu_bfloat16 value =
              __float2bfloat16_rn(
                  static_cast<float>(accumulators(i)));
          thread_shared_output(i) = value;
        }
      }
#if __HGGC_ARCH__ == 150
      __ppu_barrier_sync_nocnt(0, 1);
#else
      __syncthreads();
#endif
      if constexpr (GemmScatter::kEnabled) {
        GemmScatter::template scatter_staged_bf16<
            BlockM, BlockN, kThreadCount>(
                params_.scatter, shared_tile, valid_m,
                n_start, pool_row_base);
      }
    } else {
      typename BaseCollectiveEpilogue::Params base_params{
          params_.thread,
          params_.ptr_C,
          params_.dC,
          params_.ptr_D,
          params_.dD};
      BaseCollectiveEpilogue base_epilogue{
          base_params, shared_storage_.base_epilogue};
      base_epilogue(
          problem_shape_mnkl,
          block_shape_mnk,
          block_coord_mnkl,
          accumulators,
          tiled_mma,
          residue_mnk,
          thread_idx,
          reinterpret_cast<char*>(
              &shared_storage_.base_epilogue));

      if constexpr (GemmScatter::kEnabled) {
        auto* tile_d =
            reinterpret_cast<const __ppu_bfloat16*>(
                params_.ptr_D + row_start * kShapeN);
        GemmScatter::template scatter<
            BlockM, BlockN, kThreadCount>(
                params_.scatter, tile_d, valid_m,
                n_start, kShapeN, pool_row_base);
      }
    }
  }

private:
  Params params_;
  SharedStorage& shared_storage_;
};

} // namespace deep_gemm::mega_moe::detail
