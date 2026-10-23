#pragma once

#include <stdint.h>
#include <deep_gemm/impls/int8_gemm_cutlass3.cuh>
#include <ppu_include.hpp>

// BF16 routing and overlap epilogue shared by both FP8 weight layouts.
namespace deep_gemm::mega_moe::detail {

template <int kSourceLoadMode>
__device__ __forceinline__ uint4 scatter_load_bf16x8_policy(
    const float4* ptr) {
#if __HGGC_ARCH__ == 150
    if constexpr (kSourceLoadMode == 1) {
        return __ppu_global_ldcv_bulk_b32x4((const void*)ptr);
    } else if constexpr (kSourceLoadMode == 2) {
        return __ppu_global_ldcg_bulk_b32x4((const void*)ptr);
    } else if constexpr (kSourceLoadMode == 3) {
        return __ppu_global_ldca_bulk_b32x4((const void*)ptr);
    } else {
        return __ppu_global_load_bulk_b32x4((const void*)ptr);
    }
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
    bool kCheckPackedRoutes = true,
    bool kRollingPairScatter = false>
struct Bf16MoeScatter {
  static constexpr bool kEnabled = true;
  static constexpr bool kUsesPaddedValidRows =
      kUsePaddedValidRows;
  static constexpr bool kPipelinePreviousTile = false;
  static constexpr bool kSharedFromAccumulators = true;
  static constexpr bool kPairedNSharedEpilogue = false;
  static constexpr bool kPairedNRegisterEpilogue = false;
  static constexpr bool kPairedNEpilogue = false;
  static constexpr int kPairedNSharedScratchBytes = 0;
  static constexpr bool kScatterLocalPairWork = false;
  static constexpr bool kRemotePairScheduler = false;
  static constexpr bool kExactFullGroupM = false;
  static constexpr uint32_t kExactGroupM = 1024;
  static constexpr bool kAssumesFullScatterTile = false;
  static constexpr bool kDrainRemoteAtLocalTransition = false;
  static constexpr bool kStoresLocalInD = false;
  static constexpr bool kExactRemoteFirstSegmentSchedule = false;
  static constexpr bool kUsesScalarizedExactRoute = false;

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
  static bool is_local_tile(
      Params const& params, uint32_t pool_row_base) {
    if constexpr (!kStoresLocalInD) {
      return false;
    } else {
      if (params.packed_src_info == nullptr) {
        return false;
      }
      const int64_t packed =
          __ldg(params.packed_src_info + pool_row_base);
      const uint32_t rank =
          static_cast<uint32_t>(
              static_cast<uint64_t>(packed) >> 32);
      return rank == params.rank_idx;
    }
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
            mega_moe::detail::scatter_load_bf16x8_policy<0>(
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
  static void scatter_staged_bf16_exact_scalar_route(
      Params const& params, const __ppu_bfloat16* shared_tile,
      uint32_t valid_m, uint32_t pool_row_base,
      uint4* peer_n_base) {
    static_assert(
        kBlockM == 256 && kBlockN == 256 && kThreadNum == 512,
        "scalarized exact route requires BM256/BN256/512 threads");

    // Strict trusted GLM-5.2 EP8/T4096 contract:
    //   hidden BF16 bytes / sizeof(uint4) = 12288 / 16 = 768 vectors;
    //   one combine row is topk * 4096 + token.
    // The largest vector index is
    //   (8 * 4096 - 1) * 768 + 736 + 31 = 25,165,823,
    // so every row-relative calculation is losslessly uint32_t. The peer
    // offset and base pointer remain signed/unsigned 64-bit quantities.
    constexpr uint32_t kWarps = kThreadNum / 32;
    constexpr uint32_t kMaxTokensPerRank = 4096;
    constexpr uint32_t kRowStrideVec = 768;
    constexpr uint32_t kBf16PerVec = sizeof(uint4) /
                                      sizeof(__ppu_bfloat16);
    static_assert(kWarps == 16);
    static_assert(kBf16PerVec == 8);
    static_assert(
        (8u * kMaxTokensPerRank - 1u) * kRowStrideVec +
                (6144u - kBlockN) / kBf16PerVec + 31u <
            UINT32_MAX,
        "exact GLM combine vector index must fit uint32_t");

    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t lane = tid & 31u;
    const uint32_t warp =
#if __HGGC_ARCH__ == 150
        __ppu_read_firstlane(tid >> 5);
#else
        tid >> 5;
#endif
    const auto* packed_words =
        reinterpret_cast<const uint32_t*>(params.packed_src_info);

    for (uint32_t row0 = warp; row0 < valid_m;
         row0 += 2 * kWarps) {
      const uint32_t packed_low0 =
#if __HGGC_ARCH__ == 150
          __ld_smem(
              packed_words + ((pool_row_base + row0) << 1));
#else
          static_cast<uint32_t>(
              params.packed_src_info[pool_row_base + row0]);
#endif
      const uint32_t token0 = packed_low0 & 0x00FFFFFFu;
      const uint32_t topk0 = packed_low0 >> 24;
      const uint32_t combine_row0 =
          topk0 * kMaxTokensPerRank + token0;
      auto* const destination0 =
          peer_n_base + combine_row0 * kRowStrideVec;
      const auto* const source0 =
          reinterpret_cast<const uint4*>(
              shared_tile + row0 * kBlockN);
      const uint4 payload0 =
          mega_moe::detail::scatter_load_bf16x8_shared(source0 + lane);

      const uint32_t row1 = row0 + kWarps;
      const bool active1 = row1 < valid_m;
      uint4* destination1 = nullptr;
      uint4 payload1{};
      if (active1) {
        const uint32_t packed_low1 =
#if __HGGC_ARCH__ == 150
            __ld_smem(
                packed_words + ((pool_row_base + row1) << 1));
#else
            static_cast<uint32_t>(
                params.packed_src_info[pool_row_base + row1]);
#endif
        const uint32_t token1 = packed_low1 & 0x00FFFFFFu;
        const uint32_t topk1 = packed_low1 >> 24;
        const uint32_t combine_row1 =
            topk1 * kMaxTokensPerRank + token1;
        destination1 =
            peer_n_base + combine_row1 * kRowStrideVec;
        const auto* const source1 =
            reinterpret_cast<const uint4*>(
                shared_tile + row1 * kBlockN);
        payload1 =
            mega_moe::detail::scatter_load_bf16x8_shared(source1 + lane);
      }

#if __HGGC_ARCH__ == 150
      __ppu_remote_store_bulk_ncp_b32x4(
          static_cast<void*>(destination0 + lane), payload0);
      if (active1) {
        __ppu_remote_store_bulk_ncp_b32x4(
            static_cast<void*>(destination1 + lane), payload1);
      }
#else
      destination0[lane] = payload0;
      if (active1) {
        destination1[lane] = payload1;
      }
#endif
    }
  }

  template <int kBlockM, int kBlockN, uint32_t kThreadNum>
  CUTLASS_DEVICE
  static void scatter_staged_bf16(
      Params const& params, const __ppu_bfloat16* shared_tile,
      uint32_t valid_m, uint32_t n_start,
      uint32_t pool_row_base, uint32_t exact_segment_rank = 0,
      uint4* exact_remote_vec_base = nullptr) {
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
    if constexpr (
        kExactRemoteFirstSegmentSchedule &&
            kRollingPairScatter) {
      static_assert(
          kBlockM == 256 && kBlockN == 256 && kThreadNum == 512,
          "exact rolling-pair scatter requires BM256/BN256/512 threads");
      if constexpr (kUsesScalarizedExactRoute) {
        scatter_staged_bf16_exact_scalar_route<
            kBlockM, kBlockN, kThreadNum>(
                params, shared_tile, valid_m, pool_row_base,
                exact_remote_vec_base);
        return;
      }
      // The actual first route supplies one rank for the whole host-validated
      // BM256 segment. Keep its peer base outside the row loop. The trusted
      // path loads only low route words and issues remote stores without a
      // per-row rank branch; the checked path still verifies every row.
      // packed_src_info uses the little-endian int64 ABI:
      // [31:0] token/top-k, [63:32] rank.
      if constexpr (kCheckPackedRoutes) {
        if (exact_segment_rank >= params.num_ranks ||
            exact_segment_rank == params.rank_idx) {
          return;
        }
      }
      const auto* packed_low_words =
          reinterpret_cast<const uint32_t*>(params.packed_src_info);
      const int64_t remote_n_base =
          reinterpret_cast<int64_t>(params.combine_base) +
          peer_offset(params, exact_segment_rank) +
          static_cast<int64_t>(n_start) *
              static_cast<int64_t>(sizeof(__ppu_bfloat16));
      constexpr uint32_t kWarps = kThreadNum / 32;
      for (uint32_t row_base = tid / 32; row_base < valid_m;
           row_base += 2 * kWarps) {
        uint32_t packed_low0;
        bool rank_matches0 = true;
        if constexpr (kCheckPackedRoutes) {
          const uint64_t packed0 =
              static_cast<uint64_t>(
                  __ldg(
                      params.packed_src_info +
                      pool_row_base + row_base));
          packed_low0 = static_cast<uint32_t>(packed0);
          rank_matches0 =
              static_cast<uint32_t>(packed0 >> 32) ==
              exact_segment_rank;
        } else {
          packed_low0 = __ldg(
              packed_low_words +
              ((pool_row_base + row_base) << 1));
        }
        const uint32_t token_idx0 =
            packed_low0 & 0x00FFFFFF;
        const uint32_t topk_idx0 = packed_low0 >> 24;
        bool active0 = true;
        if constexpr (kCheckPackedRoutes) {
          active0 =
              rank_matches0 &&
              topk_idx0 < params.num_topk &&
              token_idx0 < params.max_tokens_per_rank;
        }
        int64_t destination_address0 = 0;
        uint4 payload0{};
        if (active0) {
          const uint64_t combine_row0 =
              static_cast<uint64_t>(topk_idx0) *
                  params.max_tokens_per_rank +
              token_idx0;
          destination_address0 =
              remote_n_base +
              static_cast<int64_t>(
                  combine_row0 * params.hidden_bytes);
          const auto* source0 =
              reinterpret_cast<const uint4*>(
                  shared_tile + row_base * kBlockN);
          payload0 =
              mega_moe::detail::scatter_load_bf16x8_shared(source0 + lane);
        }

        const uint32_t row1 = row_base + kWarps;
        bool active1 = row1 < valid_m;
        int64_t destination_address1 = 0;
        uint4 payload1{};
        if (active1) {
          uint32_t packed_low1;
          bool rank_matches1 = true;
          if constexpr (kCheckPackedRoutes) {
            const uint64_t packed1 =
                static_cast<uint64_t>(
                    __ldg(
                        params.packed_src_info +
                        pool_row_base + row1));
            packed_low1 = static_cast<uint32_t>(packed1);
            rank_matches1 =
                static_cast<uint32_t>(packed1 >> 32) ==
                exact_segment_rank;
          } else {
            packed_low1 = __ldg(
                packed_low_words +
                ((pool_row_base + row1) << 1));
          }
          const uint32_t token_idx1 =
              packed_low1 & 0x00FFFFFF;
          const uint32_t topk_idx1 = packed_low1 >> 24;
          if constexpr (kCheckPackedRoutes) {
            active1 =
                rank_matches1 &&
                topk_idx1 < params.num_topk &&
                token_idx1 < params.max_tokens_per_rank;
          }
          if (active1) {
            const uint64_t combine_row1 =
                static_cast<uint64_t>(topk_idx1) *
                    params.max_tokens_per_rank +
                token_idx1;
            destination_address1 =
                remote_n_base +
                static_cast<int64_t>(
                    combine_row1 * params.hidden_bytes);
            const auto* source1 =
                reinterpret_cast<const uint4*>(
                    shared_tile + row1 * kBlockN);
            payload1 =
                mega_moe::detail::scatter_load_bf16x8_shared(
                    source1 + lane);
          }
        }

        if (active0) {
          void* destination0 = reinterpret_cast<void*>(
              destination_address0 + lane * sizeof(uint4));
#if __HGGC_ARCH__ == 150
          __ppu_remote_store_bulk_ncp_b32x4(
              destination0, payload0);
#else
          *reinterpret_cast<uint4*>(destination0) = payload0;
#endif
        }
        if (active1) {
          void* destination1 = reinterpret_cast<void*>(
              destination_address1 + lane * sizeof(uint4));
#if __HGGC_ARCH__ == 150
          __ppu_remote_store_bulk_ncp_b32x4(
              destination1, payload1);
#else
          *reinterpret_cast<uint4*>(destination1) = payload1;
#endif
        }
      }
      return;
    } else if constexpr (kRollingPairScatter) {
      static_assert(
          kBlockM == 256 && kBlockN == 256 && kThreadNum == 512,
          "rolling-pair scatter requires BM256/BN256/512 threads");
      constexpr uint32_t kWarps = kThreadNum / 32;
      // Every lane in a warp scatters the same row. Keep that index scalar
      // so it does not compete with GEMM accumulators for vector registers.
      const uint32_t warp = warp_uniform(tid / 32);
      for (uint32_t row_base = warp; row_base < valid_m;
           row_base += 2 * kWarps) {
        const int64_t packed0 =
            params.packed_src_info[pool_row_base + row_base];
        const uint32_t packed_low0 =
            warp_uniform(static_cast<uint32_t>(packed0));
        const uint32_t rank0 =
            warp_uniform(static_cast<uint32_t>(packed0 >> 32));
        const uint32_t token_idx0 =
            packed_low0 & 0x00FFFFFF;
        const uint32_t topk_idx0 = packed_low0 >> 24;
        bool active0 = true;
        if constexpr (kCheckPackedRoutes) {
          active0 =
              rank0 < params.num_ranks &&
              topk_idx0 < params.num_topk &&
              token_idx0 < params.max_tokens_per_rank;
        }
        int64_t destination_address0 = 0;
        int64_t peer_offset0 = 0;
        uint4 payload0{};
        if (active0) {
          auto* destination0 = destination_row(
              params, topk_idx0, token_idx0);
          peer_offset0 = peer_offset(params, rank0);
          destination_address0 =
              peer_offset0 +
              reinterpret_cast<int64_t>(destination0 + n_start);
          const auto* source0 = reinterpret_cast<const uint4*>(
              shared_tile + row_base * kBlockN);
          payload0 =
              mega_moe::detail::scatter_load_bf16x8_shared(source0 + lane);
        }

        const uint32_t row1 = row_base + kWarps;
        bool active1 = row1 < valid_m;
        uint32_t rank1 = 0;
        int64_t destination_address1 = 0;
        uint4 payload1{};
        if (active1) {
          const int64_t packed1 =
              params.packed_src_info[pool_row_base + row1];
          const uint32_t packed_low1 =
              warp_uniform(static_cast<uint32_t>(packed1));
          rank1 = warp_uniform(static_cast<uint32_t>(packed1 >> 32));
          const uint32_t token_idx1 =
              packed_low1 & 0x00FFFFFF;
          const uint32_t topk_idx1 = packed_low1 >> 24;
          if constexpr (kCheckPackedRoutes) {
            active1 =
                rank1 < params.num_ranks &&
                topk_idx1 < params.num_topk &&
                token_idx1 < params.max_tokens_per_rank;
          }
          if (active1) {
            auto* destination1 = destination_row(
                params, topk_idx1, token_idx1);
            // Dispatch normally groups adjacent rows by source rank. Reuse
            // the peer base for that common case without assuming it.
            const int64_t peer_offset1 =
                active0 && rank1 == rank0
                    ? peer_offset0 : peer_offset(params, rank1);
            destination_address1 =
                peer_offset1 +
                reinterpret_cast<int64_t>(destination1 + n_start);
            const auto* source1 = reinterpret_cast<const uint4*>(
                shared_tile + row1 * kBlockN);
            payload1 =
                mega_moe::detail::scatter_load_bf16x8_shared(source1 + lane);
          }
        }

        if (active0) {
          void* destination0 = reinterpret_cast<void*>(
              destination_address0 + lane * sizeof(uint4));
#if __HGGC_ARCH__ == 150
          if (rank0 != params.rank_idx) {
            __ppu_remote_store_bulk_ncp_b32x4(
                destination0, payload0);
          } else {
            __ppu_global_store_bulk_b32x4(
                destination0, payload0);
          }
#else
          *reinterpret_cast<uint4*>(destination0) = payload0;
#endif
        }
        if (active1) {
          void* destination1 = reinterpret_cast<void*>(
              destination_address1 + lane * sizeof(uint4));
#if __HGGC_ARCH__ == 150
          if (rank1 != params.rank_idx) {
            __ppu_remote_store_bulk_ncp_b32x4(
                destination1, payload1);
          } else {
            __ppu_global_store_bulk_b32x4(
                destination1, payload1);
          }
#else
          *reinterpret_cast<uint4*>(destination1) = payload1;
#endif
        }
      }
    } else {
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
    uint32_t kShapeN,
    bool kStageOnlyDebug = false,
    bool kUseAccumulatorPairStaging = false>
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
  static_assert(
      !GemmScatter::kPipelinePreviousTile,
      "channelwise overlap scatter does not pipeline a previous tile");
  static_assert(
      !GemmScatter::kRemotePairScheduler,
      "channelwise overlap scatter does not use remote-pair scheduling");
  static_assert(
      !GemmScatter::kStoresLocalInD,
      "channelwise local-D bypass is restricted to the Default kernel");
  static_assert(
      !GemmScatter::kExactRemoteFirstSegmentSchedule,
      "channelwise exact segment scheduling is restricted to the "
      "Default kernel");

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

      if constexpr (kUseAccumulatorPairStaging) {
        // The innermost MMA fragment is one adjacent N pair,
        // so convert/store two accumulators with one BF16x2 transaction.
        static_assert(
            BlockN == 256 && kThreadCount >= 128 && kThreadCount % 128 == 0,
            "pair staging requires BN256 and complete N-warp groups");
        static_assert(
            decltype(size<1>(accumulator_coords))::value * (kThreadCount / 8) == BlockM,
            "pair staging repeat-M groups must span the M tile");
        static_assert(
            decltype(size<2>(accumulator_coords))::value == 4,
            "pair staging requires four repeat-N groups");
        static_assert(
            decltype(size<0>(shape<0>(accumulator_coords)))::value == 2,
            "pair staging requires two adjacent fragment values");
        static_assert(
            decltype(size<1>(shape<0>(accumulator_coords)))::value == 2,
            "pair staging requires two MMA-M atoms");
        static_assert(
            decltype(size<2>(shape<0>(accumulator_coords)))::value == 2,
            "pair staging requires two MMA-N atoms");
        using ExpectedAtomLayoutC = Layout<
            Shape<Shape<_4, _8>, Shape<_2, _2, _2>>,
            Stride<Stride<_32, _1>, Stride<_16, _8, _128>>>;
        static_assert(
            cute::is_same_v<
                typename TiledMma::AtomLayoutC_TV,
                ExpectedAtomLayoutC>,
            "pair staging requires PPU0015 row-major accumulators");
        constexpr auto probe_identity =
            make_identity_tensor(make_shape(Int<BlockM>{}, _256{}));
        constexpr auto probe_coords =
            TiledMma{}.get_thread_slice(_0{}).partition_C(
                probe_identity);
        constexpr auto probe_fragment =
            TiledMma::make_fragment_C(probe_coords);
        static_assert(
            cute::is_same_v<
                typename decltype(probe_fragment)::value_type, float>,
            "pair staging fragment values must be FP32");
        CUTE_STATIC_ASSERT_V(
            (stride<0, 0>(probe_fragment) == _1{}));

        constexpr auto pair_0 = probe_coords(
            make_coord(_0{}, _0{}, _0{}), _0{}, _0{});
        constexpr auto pair_1 = probe_coords(
            make_coord(_1{}, _0{}, _0{}), _0{}, _0{});
        constexpr auto mma_n_1 = probe_coords(
            make_coord(_0{}, _0{}, _1{}), _0{}, _0{});
        constexpr auto repeat_n_1 = probe_coords(
            make_coord(_0{}, _0{}, _0{}), _0{}, _1{});
        constexpr auto last_n_pair = probe_coords(
            make_coord(_0{}, _0{}, _1{}), _0{}, _3{});
        CUTE_STATIC_ASSERT_V(
            get<0>(pair_0) == get<0>(pair_1));
        CUTE_STATIC_ASSERT_V(
            get<1>(pair_1) == get<1>(pair_0) + _1{});
        CUTE_STATIC_ASSERT_V(
            get<0>(mma_n_1) == get<0>(pair_0));
        CUTE_STATIC_ASSERT_V(
            get<0>(repeat_n_1) == get<0>(pair_0));
        CUTE_STATIC_ASSERT_V(
            get<1>(pair_0) % _2{} == _0{});
        CUTE_STATIC_ASSERT_V(
            get<1>(mma_n_1) % _2{} == _0{});
        CUTE_STATIC_ASSERT_V(
            get<1>(repeat_n_1) % _2{} == _0{});
        CUTE_STATIC_ASSERT_V(
            get<1>(last_n_pair) + _1{} < Int<BlockN>{});

        using PairAccumulator = cutlass::Array<float, 2>;
        union PackedBf16x2 {
          __ppu_bfloat162 bf16;
          uint32_t bits;
        };

        cute::for_each(
            cute::make_seq<
                CUTE_STATIC_V(size<1>(accumulator_coords))>{},
            [&](auto repeat_m) {
          cute::for_each(
              cute::make_seq<CUTE_STATIC_V(
                  size<1>(shape<0>(accumulator_coords)))>{},
              [&](auto mma_m) {
            const auto row_coord = accumulator_coords(
                make_coord(Int<0>{}, mma_m, Int<0>{}),
                repeat_m, Int<0>{});
            const uint32_t row =
                static_cast<uint32_t>(get<0>(row_coord));
            if (row < residue_m) {
              cute::for_each(
                  cute::make_seq<
                      CUTE_STATIC_V(size<2>(accumulator_coords))>{},
                  [&](auto repeat_n) {
                cute::for_each(
                    cute::make_seq<CUTE_STATIC_V(
                        size<2>(shape<0>(accumulator_coords)))>{},
                    [&](auto mma_n) {
                  const auto coord = accumulator_coords(
                      make_coord(Int<0>{}, mma_m, mma_n),
                      repeat_m, repeat_n);
                  const uint32_t col =
                      static_cast<uint32_t>(get<1>(coord));
                  const PairAccumulator input =
                      *reinterpret_cast<const PairAccumulator*>(
                          accumulators(
                              make_coord(_, mma_m, mma_n),
                              repeat_m, repeat_n)
                              .data());
                  PackedBf16x2 value;
                  value.bf16 = __floats2bfloat162_rn(
                      input[0], input[1]);
                  cutlass::arch::shared_store<4>(
                      cutlass::arch::cutlass_get_smem_pointer(
                          shared_tile + row * BlockN + col),
                      &value.bits);
                });
              });
            }
          });
        });
      } else {
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
      }
#if __HGGC_ARCH__ == 150
      __ppu_barrier_sync_nocnt(0, 1);
#else
      __syncthreads();
#endif
      if constexpr (kStageOnlyDebug) {
        // Diagnostic only: read the staged shared tile back through the same
        // thread partition and materialize a contiguous D tile. This cleanly
        // separates accumulator->scratch mapping from routed scatter.
        auto* tile_d = reinterpret_cast<__ppu_bfloat16*>(
            params_.ptr_D + row_start * kShapeN + n_start);
        auto global_output = make_tensor(
            make_gmem_ptr(tile_d),
            make_layout(
                make_shape(Int<BlockM>{}, Int<BlockN>{}),
                make_stride(Int<kShapeN>{}, _1{})));
        auto thread_global_output =
            thread_mma.partition_C(global_output);
        CUTE_STATIC_ASSERT_V(
            size(thread_global_output) == size(accumulators));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accumulators); ++i) {
          const auto coord = accumulator_coords(i);
          if (elem_less(
                  coord,
                  make_coord(
                      get<0>(residue_mnk),
                      get<1>(residue_mnk)))) {
            thread_global_output(i) = thread_shared_output(i);
          }
        }
      } else if constexpr (GemmScatter::kEnabled) {
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
