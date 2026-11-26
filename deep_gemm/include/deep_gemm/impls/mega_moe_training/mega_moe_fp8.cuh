#pragma once

#include <deep_gemm/impls/fp8_gemm.cuh>
#include <deep_gemm/impls/mega_moe_training/mega_moe_gemm.cuh>

#if DG_TRAINING_FP8_TILE_PRODUCER

#ifndef DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_PEER_MOE_FP8_DISPATCH_MAINLOOP_CUH
#define DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_PEER_MOE_FP8_DISPATCH_MAINLOOP_CUH
/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once
#include <deep_gemm/impls/mega_moe_training/mega_moe_training.cuh>

namespace deep_gemm {

// Native blockwise mainloop, with peer BF16 gather/quant replacing A's AIU
// load in the first N tile. The other warps continue native MMA while these
// producers fill the next K128 stage. All later N tiles read the local cache.
template <class Native>
struct TrainingFp8DispatchMainloop : Native {
    using DispatchPolicy = typename Native::DispatchPolicy;
    using TileShape = typename Native::TileShape;
    using TiledMma = typename Native::TiledMma;
    using ElementAccumulator = typename Native::ElementAccumulator;
    using SmemCopyAtomA = typename Native::SmemCopyAtomA;
    using SmemCopyAtomB = typename Native::SmemCopyAtomB;
    using TransformA = typename Native::TransformA;
    using TransformB = typename Native::TransformB;
    using ElementScale = typename Native::ElementScale;
    using SmemLayoutA = typename Native::SmemLayoutA;
    using SmemLayoutB = typename Native::SmemLayoutB;
    using SmemLayoutSFA = typename Native::SmemLayoutSFA;
    using SmemLayoutSFB = typename Native::SmemLayoutSFB;
    using RealSmemLayoutSFA = typename Native::RealSmemLayoutSFA;
    using RealSmemLayoutSFB = typename Native::RealSmemLayoutSFB;
    using Params = typename Native::Params;
    static constexpr auto NumThreadsPerCTA = Native::NumThreadsPerCTA;
    static constexpr auto WarpInterleaving = Native::WarpInterleaving;
    static constexpr auto ScaleGranularityK = Native::ScaleGranularityK;
    static constexpr auto ScalePromotionInterval = Native::ScalePromotionInterval;
    static constexpr auto ScaleMsPerTile = Native::ScaleMsPerTile;
    static constexpr auto ScaleNsPerTile = Native::ScaleNsPerTile;
    static constexpr auto ScaleKsPerTile = Native::ScaleKsPerTile;
    using Native::gmem_tiled_copy_A;
    using Native::gmem_tiled_copy_B;
    using Native::gmem_tiled_copy_SFA;
    using Native::gmem_tiled_copy_SFB;
    using ElementA = typename Native::ElementA;
    using SharedStorage = typename Native::SharedStorage;
    uint64_t* addresses_ = nullptr;
    __ppu_bfloat16* packed_ = nullptr;
    int global_row_ = 0, local_row_ = 0, capacity_ = 0, width_ = 0;
    bool produce_ = false;
    mega_moe_training::MegaMoEFp8ProducerGradient gradient_{};
    float* scores_ = nullptr;

    CUTLASS_DEVICE void configure(const mega_moe_training::MegaMoEGemmParams& peers, const mega_moe_training::MegaMoEFp8ProducerGradient& gradient, float* scores, uint32_t* status,
        uint64_t* addresses, int global_row, int local_row, int capacity, int width, bool produce) {
      gradient_ = gradient;scores_ = scores;
      addresses_ = addresses;packed_ = reinterpret_cast<__ppu_bfloat16*>(peers.dispatch_scratch);
      global_row_ = global_row;local_row_ = local_row;capacity_ = capacity;width_ = width;produce_ = produce;
      if (produce) {
        if (threadIdx.x < 128) {
          const int route = peers.permutation[global_row + threadIdx.x];
          const uint64_t token = route >= 0 ? uint64_t(route) / DG_TRAINING_FP8_PRODUCER_TOPK : 0;
          const uint32_t owner = token / peers.source_rows_per_rank;
          const bool valid = route >= 0 && owner < peers.num_ranks;
          if (route >= 0 && !valid) atomicOr(status, 2u);
#if DG_TRAINING_FP8_GRADIENT_PRODUCER
          scores[threadIdx.x] = valid ? reinterpret_cast<const float*>(gradient.score_peers[owner])[
              uint64_t(route) % (uint64_t(peers.source_rows_per_rank) * DG_TRAINING_FP8_PRODUCER_TOPK)] : 0.f;
#endif
          addresses[threadIdx.x] = valid ? peers.source_peers[owner] +
              (token % peers.source_rows_per_rank) * uint64_t(width) * sizeof(__ppu_bfloat16) : 0;
        }
        __syncthreads();
      }
    }
  template <
    class... Ts,
    class FrgTensorC,
    class KTileIterator
  >
  CUTLASS_DEVICE void
  operator() (
      Params const& mainloop_params,
      cute::tuple<Ts...> const& load_inputs,
      FrgTensorC &accum,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      char *smem_buf)
  {
    using namespace cute;
    static_assert((NumThreadsPerCTA == 256 || NumThreadsPerCTA == 512) && size<0>(TileShape{}) == 128 && size<2>(TileShape{}) == 128);
    if (!produce_) {
      Native::operator()(mainloop_params, load_inputs, accum, k_tile_iter, k_tile_count, thread_idx, smem_buf);
      return;
    }

    // static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");
    static_assert(rank(SmemLayoutB{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int aiu_warp_group_thread_idx = warp_idx * 32;
    int warp_group_id = warp_idx / 8;

    Tensor gA = get<0>(load_inputs);
    Tensor gB = get<1>(load_inputs);

    // Construct shared memory tiles
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(sA));                          // BLK_M
    CUTE_STATIC_ASSERT_V(size<1>(gA) == size<1>(sA));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sB));                          // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));        // PIPE

    // Partition the copying of A and B tiles across the threads
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx & 31);

    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    Tensor gSFA = get<2>(load_inputs);
    Tensor gSFB = get<3>(load_inputs);

    auto sSFA = make_tensor(cute::make_smem_ptr(storage.smem_SFA.data()), RealSmemLayoutSFA{});
    auto sSFB = make_tensor(cute::make_smem_ptr(storage.smem_SFB.data()), RealSmemLayoutSFB{});

    auto gmem_thr_copy_SFA = gmem_tiled_copy_SFA.get_slice((thread_idx & 31) + 64);
    Tensor tSFAgSFA = gmem_thr_copy_SFA.partition_S(gSFA);
    Tensor tSFAsSFA = gmem_thr_copy_SFA.partition_D(sSFA);

    auto gmem_thr_copy_SFB = gmem_tiled_copy_SFB.get_slice((thread_idx & 31) + 32);
    Tensor tSFBgSFB = gmem_thr_copy_SFB.partition_S(gSFB);
    Tensor tSFBsSFB = gmem_thr_copy_SFB.partition_D(sSFB);

    auto copy_to_tsm = [&](auto producer_tag, int k_pipe_write, int k_idx) {
      constexpr int ProducerWarpIdx = decltype(producer_tag)::value;
      if constexpr (ProducerWarpIdx >= 0 && ProducerWarpIdx < 4) {
        // Four producer warps own disjoint rows. Quantize each K128 block
        // directly into the native swizzled A/SFA shared-memory stage.
        const int lane = thread_idx & 15;
        for (int row = ProducerWarpIdx * 2 + ((thread_idx & 31) >> 4); row < 128; row += 8) {
          const auto* source = reinterpret_cast<const __ppu_bfloat16*>(addresses_[row]);
          const int column = k_idx * 128 + lane * 8;
          uint4 raw = source ? *reinterpret_cast<const uint4*>(source + column) : uint4{};
#if DG_TRAINING_FP8_GRADIENT_PRODUCER
          const uint4 raw_y = source ? *reinterpret_cast<const uint4*>(
              reinterpret_cast<const __ppu_bfloat16*>(gradient_.unweighted_y) + int64_t(global_row_ + row) * width_ + column) : uint4{};
          const auto* dy = reinterpret_cast<const __ppu_bfloat16*>(&raw);
          const auto* y = reinterpret_cast<const __ppu_bfloat16*>(&raw_y);
          uint4 weighted_raw{};auto* weighted = reinterpret_cast<__ppu_bfloat16*>(&weighted_raw);
          float dot = 0.f;
          #pragma unroll
          for (int j = 0; j < 8; ++j) {
            const float value = __bfloat162float(dy[j]);
            weighted[j] = __float2bfloat16(value * scores_[row]);
            dot += value * __bfloat162float(y[j]);
          }
          const unsigned mask = 0xffffu << ((thread_idx & 31) >= 16 ? 16 : 0);
          #pragma unroll
          for (int delta = 8; delta > 0; delta >>= 1) dot += __shfl_down_sync(mask, dot, delta, 16);
          if (lane == 0) gradient_.score_partials[int64_t(global_row_ + row) * (width_ / 128) + k_idx] = dot;
          raw = weighted_raw;
#endif
          float scale;
          const uint2 quantized = mega_moe_training::peer_quantize_eight(raw, scale);
          if (lane == 0) {
            const_cast<ElementScale*>(mainloop_params.ptr_scale_A)[int64_t(k_idx) * capacity_ + local_row_ + row] = scale;
          }
          const int64_t at = int64_t(local_row_ + row) * width_ + column;
          *reinterpret_cast<uint2*>(const_cast<ElementA*>(mainloop_params.ptr_A) + at) = quantized;
          *reinterpret_cast<uint4*>(packed_ + int64_t(global_row_ + row) * width_ + column) = raw;
        }
        // Native AIU and scale-copy atoms preserve their physical TSM layout.
        // This is still K128 pipelined: no complete-input dispatch barrier.
        __threadfence();
        __ppu_barrier_sync(7, 128, 0);
        if constexpr (ProducerWarpIdx == 0) {
          copy(gmem_tiled_copy_A, tAgA(_,_,_,k_idx), tAsA(_,_,_,k_pipe_write));
          copy(gmem_tiled_copy_SFA, tSFAgSFA(_,_,_,k_idx), tSFAsSFA(_,_,_,k_pipe_write));
        }
      } else if constexpr (ProducerWarpIdx == 4) {
        copy(gmem_tiled_copy_B, tBgB(_,_,_,k_idx), tBsB(_,_,_,k_pipe_write));
        copy(gmem_tiled_copy_SFB, tSFBgSFB(_,_,_,k_idx), tSFBsSFB(_,_,_,k_pipe_write));
      }
    };

    // Start async loads for all pipes but the last
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages-1; ++k_pipe) {
      if (k_tile_count > 0) {
        // All three producers belong to physical group 0; preserve prologue dispatch.
        switch (warp_idx) {
          case 0: copy_to_tsm(Int<0>{}, k_pipe, *k_tile_iter); break;
          case 1: copy_to_tsm(Int<1>{}, k_pipe, *k_tile_iter); break;
          case 2: copy_to_tsm(Int<2>{}, k_pipe, *k_tile_iter); break;
          case 3: copy_to_tsm(Int<3>{}, k_pipe, *k_tile_iter); break;
          case 4: copy_to_tsm(Int<4>{}, k_pipe, *k_tile_iter); break;
        }
        ++k_tile_iter;
      }
      cp_async_fence();
      --k_tile_count;
    }

    //
    // MMA Atom partitioning
    //

    // Tile MMA compute thread partitions and allocate accumulators
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,_0{}));                     // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,_0{}));                     // (MMA,MMA_N,MMA_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K

    using EngineAccum =  typename FrgTensorC::engine_type;
    using LayoutAccum =  typename FrgTensorC::layout_type;
    cutlass::gemm::collective::MixAccumulation<EngineAccum, LayoutAccum, ElementAccumulator> accumulation(accum, ScalePromotionInterval, size<2>(tCrA));

    //
    // Copy Atom retiling
    //
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));                  // (CPY,CPY_M,CPY_K,PIPE)

    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                   // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));                  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    // Block scaling
    auto layout_sSFA_copy = make_layout(
      make_shape(shape<0>(SmemLayoutSFA{}), get<1>(TileShape{}),
                 make_shape(shape<1>(SmemLayoutSFA{}), shape<2>(SmemLayoutSFA{}))),
      make_stride(stride<0>(SmemLayoutSFA{}), _0{},
                  make_stride(stride<1>(SmemLayoutSFA{}), stride<2>(SmemLayoutSFA{}))));
    auto layout_sSFB_copy = make_layout(
      make_shape(get<0>(TileShape{}), shape<0>(SmemLayoutSFB{}),
                 make_shape(shape<1>(SmemLayoutSFB{}), shape<2>(SmemLayoutSFB{}))),
      make_stride(_0{}, stride<0>(SmemLayoutSFB{}),
                  make_stride(stride<1>(SmemLayoutSFB{}), stride<2>(SmemLayoutSFB{}))));
    Tensor sSFA_copy = make_tensor(cute::make_smem_ptr(storage.smem_SFA.data()), layout_sSFA_copy);
    Tensor sSFB_copy = make_tensor(cute::make_smem_ptr(storage.smem_SFB.data()), layout_sSFB_copy);

    Tensor tCsSFA = tiled_mma.get_slice(thread_idx).partition_C(sSFA_copy);
    Tensor tCsSFB = tiled_mma.get_slice(thread_idx).partition_C(sSFB_copy);
    Tensor tCrSFA = make_fragment_like<ElementScale>(tCsSFA(_, _, _, _0{}));
    Tensor tCrSFB = make_fragment_like<ElementScale>(tCsSFB(_, _, _, _0{}));

    //
    // PIPELINED MAIN LOOP
    //

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = DispatchPolicy::Stages-1;

    int smem_scale_pipe_read = smem_pipe_read;

    Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA_copy_view);
    auto K_ATOM_PER_COPY = size<2>(tCrA) / size<2>(tCrA_copy_view);
    auto K_BLOCK_GROUP = K_BLOCK_MAX / Int<ScaleKsPerTile>{};

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
      // Wait until our first prefetched tile is loaded in
      cp_async_wait<DispatchPolicy::Stages-2>();
      __syncthreads();

      // Prefetch the first rmem from the first k-tile
      copy(smem_tiled_copy_A, tCsA_p(_,_,_0{}), tCrA_copy_view(_,_,_0{}));
      copy(smem_tiled_copy_B, tCsB_p(_,_,_0{}), tCrB_copy_view(_,_,_0{}));
      copy(tCsSFA(_,_,_,make_coord(_0{}, _0{})), tCrSFA);
      copy(tCsSFB(_,_,_,make_coord(_0{}, _0{})), tCrSFB);
    }
    if constexpr (WarpInterleaving) {
      if (warp_group_id == 1) {
        __ppu_barrier_arrive(5, NumThreadsPerCTA, 0);
      }
    }

    // Specialize barrier IDs by group while preserving the shared loop body.
    auto run_group = [&](auto group_tag, auto producer_tag) {
      constexpr int WarpGroupId = decltype(group_tag)::value;
      auto preprocess_scale = [&]() {
        if constexpr (WarpInterleaving) {
          __ppu_barrier_sync(5 + WarpGroupId, NumThreadsPerCTA, 0); // group 0 wait bar 5, group 1 wait bar 6
        }
        clear(accumulation());
        if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
          tCrSFA(_0{}) = tCrSFA(_0{}) * tCrSFB(_0{});
        }
        if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
          ElementScale scale_b = tCrSFB(_0{});
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(filter_zeros(tCrSFA)); i++) {
            filter_zeros(tCrSFA)(i) = filter_zeros(tCrSFA)(i) * scale_b;
          }
        }
        if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
          ElementScale scale_a = tCrSFA(_0{});
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(filter_zeros(tCrSFB)); i++) {
            filter_zeros(tCrSFB)(i) = filter_zeros(tCrSFB)(i) * scale_a;
          }
        }
      };

      auto process_scale_and_copy = [&](int k_pipe_read, int k_idx = _0{}) {
        if constexpr (WarpInterleaving) {
          __ppu_barrier_arrive(6 - WarpGroupId, NumThreadsPerCTA, 0);
        }
        // Block scale the accumulators with reg tensor `tCrSFA` and `tCrSFB`
        if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
          ElementScale scale_ab = tCrSFA(_0{});
          accumulation.scale(scale_ab);
        }
        if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
          accumulation.scale(tCrSFA);
        }
        if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
          accumulation.scale(tCrSFB);
        }
        if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
          accumulation.scale(tCrSFA, tCrSFB);
        }

        copy(tCsSFB(_,_,_,make_coord(k_idx, k_pipe_read)), tCrSFB);
        copy(tCsSFA(_,_,_,make_coord(k_idx, k_pipe_read)), tCrSFA);
      };

      CUTLASS_PRAGMA_NO_UNROLL
      while (k_tile_count > -(DispatchPolicy::Stages-1)) {
        // Pipeline the outer products with a static for loop.
        //
        // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
        for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
          if (k_block == K_BLOCK_MAX - 1) {
            // Slice the smem_pipe_read smem
            tCsA_p = tCsA(_,_,_,smem_pipe_read);
            tCsB_p = tCsB(_,_,_,smem_pipe_read);

            // Commit the smem for smem_pipe_read
            cp_async_wait<DispatchPolicy::Stages-2>();
            if constexpr (WarpInterleaving) {
              __ppu_barrier_sync(3 + WarpGroupId, NumThreadsPerCTA / 2, 0);   // warp group sync
            } else {
              __syncthreads();
            }
          }
          if (k_block % K_BLOCK_GROUP == 0) {
            preprocess_scale();
          }
          // Load A, B shmem->regs for k_block+1
          auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
          copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
          copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));

          CUTLASS_PRAGMA_UNROLL
          for (int k_loop = 0; k_loop < K_ATOM_PER_COPY; k_loop++) {
            auto atom_idx = k_block * K_ATOM_PER_COPY + k_loop;
            // Transform before compute
            cute::transform(tCrA(_,_,atom_idx), TransformA{});
            cute::transform(tCrB(_,_,atom_idx), TransformB{});
            // gemm for one tiled_mma atom on K
            cute::gemm(tiled_mma, tCrA(_,_,atom_idx), tCrB(_,_,atom_idx), accumulation());
          }
          // Issue the next tile after the first MMA block.
          if (k_block == 0) {
            copy_to_tsm(producer_tag, smem_pipe_write, *k_tile_iter);
            cp_async_fence();

            // Advance the tile
            --k_tile_count;
            if (k_tile_count > 0) { ++k_tile_iter; }

            // Advance the pipe before the last-block wait.
            smem_pipe_write = smem_pipe_read;
            ++smem_pipe_read;
            smem_pipe_read = (smem_pipe_read == DispatchPolicy::Stages) ? 0 : smem_pipe_read;
          }
          if constexpr (ScaleKsPerTile > 1) {
            if (k_block_next % K_BLOCK_GROUP == 0) {    // ctaK = 256
              process_scale_and_copy(smem_scale_pipe_read, k_block_next / K_BLOCK_GROUP * ScaleGranularityK);
              smem_scale_pipe_read = smem_pipe_read;
            }
          } else {                                      // ctaK = 128
            if (k_block_next == 0) {
              process_scale_and_copy(smem_pipe_read);
            }
          }
        }); // for_each
      }
    };

    // Non-producer warps keep their original physical group; -1 disables gmem copies.
    auto run_non_producer = [&]() {
      if constexpr (WarpInterleaving) {
        if (warp_group_id == 0) {
          run_group(Int<0>{}, Int<-1>{});
        } else {
          run_group(Int<1>{}, Int<-1>{});
        }
      } else {
        run_group(Int<0>{}, Int<-1>{});
      }
    };

    // Select a producer role once, before entering the steady-state loop.
    switch (warp_idx) {
      case 0: run_group(Int<0>{}, Int<0>{}); break;  // A/B
      case 1: run_group(Int<0>{}, Int<1>{}); break;  // SFB
      case 2: run_group(Int<0>{}, Int<2>{}); break;
      case 3: run_group(Int<0>{}, Int<3>{}); break;
      case 4: run_group(Int<0>{}, Int<4>{}); break;
      default: run_non_producer(); break;
    }
    if constexpr (WarpInterleaving) {
      if (warp_group_id == 0) {
        __ppu_barrier_sync(5, NumThreadsPerCTA, 0);
      }
    }

    cp_async_wait<0>();
    __syncthreads();
  }
};
} // namespace deep_gemm

#endif
#endif
#if DG_TRAINING_FP8_PEER_INPUT && DG_TRAINING_FP8_INPUT_K_SLICE

#ifndef DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_PEER_MOE_FP8_KREADY_MAINLOOP_CUH
#define DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_PEER_MOE_FP8_KREADY_MAINLOOP_CUH
/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once
#include <deep_gemm/tix/sys_atomics.cuh>

namespace deep_gemm {

// Keep the native TC mainloop free of readiness polls. A CTA starts as soon
// as one K stripe is published and coalesces adjacent published stripes into
// a native run. Later stripes are gathered/quantized concurrently with that run.
template<class Native, uint32_t FullK>
struct TrainingFp8KReadyMainloop : Native {
  using Params=typename Native::Params;
  using TileShape=typename Native::TileShape;
  uint32_t* ready_=nullptr;
  uint32_t* status_=nullptr;
  uint64_t timeout_=0;
  CUTLASS_DEVICE void configure(const mega_moe_training::MegaMoEDispatchPipeline& input,
      uint32_t row,uint32_t valid,uint32_t width) {
    static_assert(cute::size<0>(TileShape{})==128);
    ready_=input.ready+row/128;status_=input.status;timeout_=input.timeout_cycles;
    static_cast<void>(valid);static_cast<void>(width);
  }
  struct BoundedKIterator {
    int index,last;
    CUTLASS_DEVICE int operator*() const {return index<last?index:last;}
    CUTLASS_DEVICE BoundedKIterator& operator++(){++index;return *this;}
  };
  template<class...Ts,class FrgTensorC,class KTileIterator>
  CUTLASS_DEVICE void operator()(Params const& params,cute::tuple<Ts...> const& inputs,
      FrgTensorC& accum,KTileIterator,int count,int thread,char* smem) {
    constexpr uint32_t stripes=DG_TRAINING_FP8_INPUT_READY_COUNT;
    constexpr uint32_t subtiles=DG_TRAINING_FP8_INPUT_SUBTILES;
    static_assert(stripes*subtiles<=32);
    constexpr uint32_t submask=(1u<<subtiles)-1u;
    static_assert(FullK%(128*stripes)==0);
    constexpr uint32_t tiles_per_stripe=FullK/(128*stripes);
    __shared__ uint32_t ready_end,ready_failed;
    int first=0;
    CUTLASS_PRAGMA_NO_UNROLL
    while(first<count) {
      if(threadIdx.x==0) {
        ready_failed=0;
        const uint32_t start_stripe=uint32_t(first)/tiles_per_stripe;
        const uint32_t required=submask<<(start_stripe*subtiles);
        const uint64_t begin=clock64();
        uint32_t mask;
        while(true) {
          mask=tix::ld_acq(ready_);
          if((mask&required)==required)break;
          if(tix::ld_acq(status_) || clock64()-begin>timeout_) {
            atomicOr(status_,8u);ready_failed=1;break;
          }
        }
        uint32_t last_stripe=start_stripe+1;
        while(last_stripe<stripes && ((mask>>(last_stripe*subtiles))&submask)==submask)++last_stripe;
        ready_end=last_stripe*tiles_per_stripe;
      }
      __syncthreads();
      if(ready_failed)return;
      const int end=int(ready_end);
      // Native drain may issue unused prefetches. Clamp them to the final
      // acquired tile so it never reads a not-yet-published stripe.
      Native::operator()(params,inputs,accum,BoundedKIterator{first,end-1},
          end-first,thread,smem);
      first=end;
    }
  }
};
} // namespace deep_gemm

#endif
#endif
#ifndef DG_TRAINING_FP8_INLINE_SOURCE
#define DG_TRAINING_FP8_INLINE_SOURCE 0
#endif
#if DG_TRAINING_FP8_INLINE_SOURCE
#include <deep_gemm/impls/mega_moe_training/fp8_inline_source.cuh>

#ifndef DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_PEER_MOE_FP8_BROADCAST_SF_CUH
#define DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_PEER_MOE_FP8_BROADCAST_SF_CUH
namespace deep_gemm {
// The source publishes one feature-scale tile, reused by every row tile and
// expert. Data A/B and SFB retain their original descriptors and coordinates.
template<class Native>
struct TrainingFp8BroadcastScaleMainloop : Native {
    const float* broadcast_sf_=nullptr;
    CUTLASS_DEVICE void configure_broadcast(const float* pointer){broadcast_sf_=pointer;}
    template<class ProblemShape,class BlockCoord,class Params>
    CUTLASS_DEVICE auto load_init(const ProblemShape& problem,const BlockCoord& block,const Params& params) {
        using namespace cute;
        auto normal=Native::load_init(problem,block,params);
        constexpr int Rows=size<0>(typename Native::TileShape{});
        const auto blocks=ceil_div(get<2>(problem),128);
        const auto stride_sf=make_stride(_1{},Int<Rows>{});
        auto matrix=make_tensor(make_gmem_ptr(broadcast_sf_),make_layout(make_shape(Int<Rows>{},blocks),stride_sf));
        auto tile=local_tile(make_mix_tensor_like(matrix),
            make_tile(Int<Native::ScaleMsPerTile>{},Int<Native::ScaleKsPerTile>{}),make_coord(0,_));
        using Tiler=typename Native::GmemTiledCopySFA::Tiler_MN;
        this->gmem_tiled_copy_SFA.desc_.template init<typename Native::ElementScale,Native::TransSFA,
            get<0>(Tiler{}),get<1>(Tiler{})>(nullptr,Rows,blocks,stride_sf);
        return make_tuple(get<0>(normal),get<1>(normal),tile,get<3>(normal));
    }
};
} // namespace deep_gemm

#endif
#endif
#ifndef DG_TRAINING_FP8_INLINE_SOURCE_ROTATION
#define DG_TRAINING_FP8_INLINE_SOURCE_ROTATION 0
#endif
namespace deep_gemm {


#if DG_TRAINING_FP8_EXPERT_END
// Native expert-major/two-M swizzle over a range. Global row/weight offsets
// remain native; only work ordinals restart at zero for this independent batch.
template<class NativeScheduler>
struct TrainingFp8ExpertBatchScheduler : NativeScheduler {
    CUTLASS_DEVICE explicit TrainingFp8ExpertBatchScheduler(typename NativeScheduler::Params const& p):NativeScheduler(p) {
        static_assert(!NativeScheduler::kIsNoPadPreprocessLayout);
        this->curr_group_idx=DG_TRAINING_FP8_EXPERT_BEGIN;
        this->curr_cumsum=this->curr_cumsum_m=0;
        for(uint32_t e=0;e<DG_TRAINING_FP8_EXPERT_BEGIN;++e)this->curr_cumsum_m+=__ldg(p.grouped_layout+e);
    }
    CUTLASS_DEVICE bool fetch_next_work(uint32_t& mi,uint32_t& ni) {
        constexpr uint32_t columns=(NativeScheduler::SHAPE_N+NativeScheduler::BLOCK_N-1)/NativeScheduler::BLOCK_N;
        const uint32_t work=(this->current_iter++)*gridDim.x+blockIdx.x;
        while(this->curr_group_idx<DG_TRAINING_FP8_EXPERT_END) {
            this->curr_group_m=__ldg(this->params.grouped_layout+this->curr_group_idx);
            const uint32_t blocks=(this->curr_group_m+NativeScheduler::BLOCK_M-1)/NativeScheduler::BLOCK_M;
            if(work<(this->curr_cumsum+blocks)*columns) {
                this->get_swizzled_block_idx(blocks,work-this->curr_cumsum*columns,mi,ni);return true;
            }
            this->curr_cumsum+=blocks;this->curr_cumsum_m+=this->curr_group_m;++this->curr_group_idx;
        }
        mi=0;ni=columns;return false;
    }
};
#endif

template <class NativeScheduler>
struct TrainingFp8TileOwnerScheduler : NativeScheduler {
    uint32_t owned_m = blockIdx.x, next_n = 0;
    CUTLASS_DEVICE explicit TrainingFp8TileOwnerScheduler(typename NativeScheduler::Params const& p) : NativeScheduler(p) {}
    CUTLASS_DEVICE bool fetch_next_work(uint32_t& mi, uint32_t& ni) {
      constexpr uint32_t columns = (NativeScheduler::SHAPE_N + NativeScheduler::BLOCK_N - 1) / NativeScheduler::BLOCK_N;
      if (next_n == columns) { owned_m += gridDim.x;next_n = 0; }
      while (this->curr_group_idx < NativeScheduler::kNumGroups) {
        this->curr_group_m = __ldg(this->params.grouped_layout + this->curr_group_idx);
        const uint32_t blocks = (this->curr_group_m + NativeScheduler::BLOCK_M - 1) / NativeScheduler::BLOCK_M;
        if (owned_m < this->curr_cumsum + blocks) {
          mi = owned_m - this->curr_cumsum;ni = next_n++;return true;
        }
        this->curr_cumsum += blocks;this->curr_cumsum_m += this->curr_group_m;++this->curr_group_idx;
      }
      return false;
    }
};

// Native expert-major/two-M swizzle with a shared atomic work ordinal. Two
// independently launched worker grids can join/leave without losing tiles.
template<class NativeScheduler>
struct TrainingFp8InputWorkScheduler : NativeScheduler {
    uint32_t* counter;
    CUTLASS_DEVICE explicit TrainingFp8InputWorkScheduler(typename NativeScheduler::Params const& params,
                                                         uint32_t* work_counter)
        : NativeScheduler(params),counter(work_counter) {}
    CUTLASS_DEVICE bool fetch_next_work(uint32_t& mi,uint32_t& ni) {
        static_assert(NativeScheduler::GEMM_TYPE==GemmType::MGroupedNoPad &&
                      !NativeScheduler::kIsNoPadPreprocessLayout);
        constexpr uint32_t columns=(NativeScheduler::SHAPE_N+NativeScheduler::BLOCK_N-1)/NativeScheduler::BLOCK_N;
        __shared__ uint32_t ordinal;
        __syncthreads();
        if(threadIdx.x==0)ordinal=atomicAdd(counter,1u);
        __syncthreads();
        const uint32_t work=ordinal;
        uint32_t mblocks=0;
        while(this->curr_group_idx<NativeScheduler::kNumGroups) {
            this->curr_group_m=uint32_t(__ldg(this->params.grouped_layout+this->curr_group_idx));
            mblocks=(this->curr_group_m+NativeScheduler::BLOCK_M-1)/NativeScheduler::BLOCK_M;
            const uint32_t end=this->curr_cumsum+mblocks;
            if(work<end*columns) {
                this->get_swizzled_block_idx(mblocks,work-this->curr_cumsum*columns,mi,ni);
                return true;
            }
            ++this->curr_group_idx;
            this->curr_cumsum=end;
            this->curr_cumsum_m+=this->curr_group_m;
        }
        mi=0;ni=columns;
        return false;
    }
};

// One already-resident CTA owns one full M tile and advances all N tiles.
// This supplies39 independent source tiles rather than a few CAS winners.
template<class NativeScheduler>
struct TrainingFp8ExternalReadyScheduler : NativeScheduler {
    uint32_t ready_ordinal=0,source_rank=0,source_ranks=1;
    CUTLASS_DEVICE explicit TrainingFp8ExternalReadyScheduler(typename NativeScheduler::Params const& p):NativeScheduler(p){}
    CUTLASS_DEVICE bool fetch_next_work(uint32_t& mi,uint32_t& ni) {
        if(!NativeScheduler::fetch_next_work(mi,ni))return false;
        constexpr uint32_t Columns=(NativeScheduler::SHAPE_N+NativeScheduler::BLOCK_N-1)/NativeScheduler::BLOCK_N;
        const uint32_t work=(this->current_iter-1)*gridDim.x+blockIdx.x;
        const uint4 record=reinterpret_cast<const uint4*>(this->params.grouped_layout)[1+work/Columns];
#if DG_TRAINING_FP8_INLINE_SOURCE_ROTATION
        const uint32_t blocks=(this->curr_group_m+NativeScheduler::BLOCK_M-1)/NativeScheduler::BLOCK_M;
        mi=(mi+source_rank*(blocks/source_ranks+1))%blocks;
#endif
        ready_ordinal=record.z+mi;
        return true;
    }
};

// A global ordinal lets already-resident workers consume more tiles while
// other CUs remain occupied by communication. Preserve the native two-M swizzle.
template<class NativeScheduler>
struct TrainingFp8ExternalWorkScheduler : TrainingFp8ExternalReadyScheduler<NativeScheduler> {
    using Parent=TrainingFp8ExternalReadyScheduler<NativeScheduler>;
    uint32_t* counter;
    CUTLASS_DEVICE TrainingFp8ExternalWorkScheduler(typename NativeScheduler::Params const& p,uint32_t* value)
        :Parent(p),counter(value){}
    CUTLASS_DEVICE bool fetch_next_work(uint32_t& mi,uint32_t& ni) {
        static_assert(NativeScheduler::kIsNoPadPreprocessLayout);
        constexpr uint32_t Columns=(NativeScheduler::SHAPE_N+NativeScheduler::BLOCK_N-1)/NativeScheduler::BLOCK_N;
        __shared__ uint32_t work;
        __syncthreads();if(threadIdx.x==0)work=atomicAdd(counter,1u);__syncthreads();
        if(work>=this->num_aligned_m_blocks*Columns)return false;
        const uint4 record=reinterpret_cast<const uint4*>(this->params.grouped_layout)[1+work/Columns];
        this->curr_group_idx=record.x;this->curr_group_m=record.y;
        this->curr_cumsum=record.z;this->curr_cumsum_m=record.w;
        const uint32_t blocks=(this->curr_group_m+NativeScheduler::BLOCK_M-1)/NativeScheduler::BLOCK_M;
        this->get_swizzled_block_idx(blocks,work-record.z*Columns,mi,ni);
#if DG_TRAINING_FP8_INLINE_SOURCE_ROTATION
        mi=(mi+this->source_rank*(blocks/this->source_ranks+1))%blocks;
#endif
        this->ready_ordinal=record.z+mi;
        return true;
    }
};

template<class NativeScheduler>
struct TrainingFp8InlineOwnerScheduler : NativeScheduler {
    uint32_t owned_m=blockIdx.x,next_n=0,source_rank=0,source_ranks=1;
    CUTLASS_DEVICE explicit TrainingFp8InlineOwnerScheduler(typename NativeScheduler::Params const& p):NativeScheduler(p){}
    CUTLASS_DEVICE bool fetch_next_work(uint32_t& mi,uint32_t& ni) {
        static_assert(NativeScheduler::kIsNoPadPreprocessLayout);
        constexpr uint32_t Columns=(NativeScheduler::SHAPE_N+NativeScheduler::BLOCK_N-1)/NativeScheduler::BLOCK_N;
        if(next_n==Columns){owned_m+=gridDim.x;next_n=0;}
        if(owned_m>=this->num_aligned_m_blocks)return false;
        const uint4 record=reinterpret_cast<const uint4*>(this->params.grouped_layout)[1+owned_m];
        this->curr_group_idx=record.x;this->curr_group_m=record.y;
        this->curr_cumsum=record.z;this->curr_cumsum_m=record.w;
        mi=owned_m-record.z;ni=next_n++;
#if DG_TRAINING_FP8_INLINE_SOURCE_ROTATION
        const uint32_t blocks=(this->curr_group_m+NativeScheduler::BLOCK_M-1)/NativeScheduler::BLOCK_M;
        mi=(mi+source_rank*(blocks/source_ranks+1))%blocks;
#endif
        return true;
    }
};

// Native blockwise FP8 mainloop and scheduler, with direct BF16 peer stores.
// Input is already locally packed/quantized; this primitive fuses the return.
template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_,
          bool kEnableSboOverlap, bool kUseNStageKernel>
class TrainingFp8PeerReturn {
public:
    static_assert(!kEnableSboOverlap && !kUseNStageKernel);
#if DG_TRAINING_FP8_INLINE_SOURCE==4
    using CollectiveMainloop = TrainingFp8BroadcastScaleMainloop<CollectiveMainloop_>;
#elif DG_TRAINING_FP8_TILE_PRODUCER
    using CollectiveMainloop = TrainingFp8DispatchMainloop<CollectiveMainloop_>;
#elif DG_TRAINING_FP8_PEER_INPUT && DG_TRAINING_FP8_INPUT_K_SLICE
    using CollectiveMainloop = TrainingFp8KReadyMainloop<CollectiveMainloop_, TileScheduler_::SHAPE_K>;
#else
    using CollectiveMainloop = CollectiveMainloop_;
#endif
    using CollectiveEpilogue = CollectiveEpilogue_;
    using Base = DeepGemmUniversal<ProblemShape_, CollectiveMainloop, CollectiveEpilogue_, TileScheduler_, false, false>;
    using ProblemShape = typename Base::ProblemShape;
    using TileShape = typename Base::TileShape;
    using TiledMma = typename Base::TiledMma;
    using TileScheduler = typename Base::TileScheduler;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementScale = typename Base::ElementScale;
    using LayoutSFA = typename Base::LayoutSFA;
    using MainloopParams = typename Base::MainloopParams;
    using SharedStorage = typename Base::SharedStorage;
    using Params = typename Base::Params;
    using StrideA = typename Base::StrideA;
    using StrideB = typename Base::StrideB;
    using StrideC = typename Base::StrideC;
    using StrideD = typename Base::StrideD;
    static_assert(TileScheduler::GEMM_TYPE == GemmType::MGroupedNoPad);
    static constexpr int N_PREFETCH_CACHELINE = Base::N_PREFETCH_CACHELINE;
    static constexpr bool kUseReplicaWeights = false;
    static constexpr int SharedStorageSize = (DG_TRAINING_FP8_INLINE_SOURCE || DG_TRAINING_FP8_RETURN_PIPELINE) ? ((Base::SharedStorageSize+127)/128)*128+128 : Base::SharedStorageSize;
    static constexpr int MaxThreadsPerBlock = Base::MaxThreadsPerBlock;
    static constexpr int MinBlocksPerMultiprocessor = Base::MinBlocksPerMultiprocessor;
    CUTLASS_DEVICE void
  operator()(Params const& params, const mega_moe_training::MegaMoEGemmParams& peers,
             const mega_moe_training::MegaMoEDispatchPipeline& input, const mega_moe_training::MegaMoEDispatchPipeline& output,
             uint32_t* input_work_counter,const mega_moe_training::MegaMoEFp8ProducerGradient& producer_gradient,char* smem_buf, const mega_moe_training::MegaMoEFp8InlineSource& inline_source={}) {
    using namespace cute;
    using X = Underscore;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    if (TileScheduler::GEMM_TYPE == GemmType::MGroupedMasked) {
      // group is small 8|16, just prefetch one cacheline
      __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout));
    } else if (TileScheduler::GEMM_TYPE == GemmType::MGroupedNoPad) {
      // each warp prefetch one cacheline
      if (warp_idx < N_PREFETCH_CACHELINE) {
        __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout + (warp_idx<<5)));
      }
    }

#if DG_TRAINING_FP8_RETURN_PIPELINE || (DG_TRAINING_FP8_INPUT_DYNAMIC && DG_TRAINING_FP8_INLINE_SOURCE<3) || DG_TRAINING_FP8_TILE_PRODUCER || DG_TRAINING_FP8_INPUT_K_SLICE
#if DG_TRAINING_FP8_RETURN_PIPELINE
    int& return_layout_bad=*reinterpret_cast<int*>(smem_buf+((Base::SharedStorageSize+127)/128)*128);
#else
    __shared__ int return_layout_bad;
#endif
    if(threadIdx.x==0) {
      return_layout_bad=0;int64_t total_rows=0;
      for(uint32_t g=0;g<TileScheduler::kNumGroups;++g) {
        const int rows=params.scheduler.grouped_layout[g];
        if(rows<0 || rows%128)return_layout_bad=1;
        total_rows+=rows;
      }
      if(total_rows>params.scheduler.shape_m)return_layout_bad=1;
      if(return_layout_bad)atomicOr((DG_TRAINING_FP8_INPUT_DYNAMIC || DG_TRAINING_FP8_TILE_PRODUCER || DG_TRAINING_FP8_INPUT_K_SLICE) ? input.status : output.status,2u);
    }
    __syncthreads();if(return_layout_bad)return;
#endif
#if DG_TRAINING_FP8_INLINE_SOURCE>=3 && DG_TRAINING_FP8_INPUT_DYNAMIC
    // External-ready input uses preprocessed uint4 records, not a lengths
    // array. Its status pointer also belongs to inline_source, not input.
    __shared__ int external_metadata_bad;
    if(threadIdx.x==0) {
        const int count=params.scheduler.grouped_layout[0];
        external_metadata_bad=count<0 || uint32_t(count)>inline_source.signal_capacity;
        if(!external_metadata_bad) {
            const auto* records=reinterpret_cast<const uint4*>(params.scheduler.grouped_layout)+1;
            for(uint32_t at=0;at<uint32_t(count);++at) {
                const uint4 r=records[at];
                const uint32_t blocks=(r.y+TileScheduler::BLOCK_M-1)/TileScheduler::BLOCK_M;
                if(r.x>=TileScheduler::kNumGroups || !r.y || r.y%128 || r.w%128 ||
                    uint64_t(r.w)+r.y>params.scheduler.shape_m || r.z>at || at-r.z>=blocks)
                    external_metadata_bad=1;
            }
        }
        if(external_metadata_bad)atomicOr(inline_source.status,2u);
    }
    __syncthreads();if(external_metadata_bad)return;
#endif
#if DG_TRAINING_FP8_INLINE_SOURCE>=3
#if DG_TRAINING_FP8_INPUT_DYNAMIC
    TrainingFp8ExternalWorkScheduler<TileScheduler> deep_scheduler{params.scheduler,input_work_counter};
#else
    TrainingFp8ExternalReadyScheduler<TileScheduler> deep_scheduler{params.scheduler};
#endif
#elif DG_TRAINING_FP8_INLINE_SOURCE
    TrainingFp8InlineOwnerScheduler<TileScheduler> deep_scheduler{params.scheduler};
#elif DG_TRAINING_FP8_EXPERT_END
    TrainingFp8ExpertBatchScheduler<TileScheduler> deep_scheduler{params.scheduler};
#elif DG_TRAINING_FP8_TILE_PRODUCER
    TrainingFp8TileOwnerScheduler<TileScheduler> deep_scheduler{params.scheduler};
#elif DG_TRAINING_FP8_INPUT_DYNAMIC
    TrainingFp8InputWorkScheduler<TileScheduler> deep_scheduler{params.scheduler,input_work_counter};
#else
    TileScheduler deep_scheduler{params.scheduler};
#endif

#if DG_TRAINING_FP8_INLINE_SOURCE && DG_TRAINING_FP8_INLINE_SOURCE_ROTATION
    deep_scheduler.source_rank=inline_source.local_rank;
    deep_scheduler.source_ranks=inline_source.num_ranks;
#endif
    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShape>::value);


    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    // Preconditions
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{}; // (BLK_M,BLK_N,BLK_K)
    auto strides = make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, M));
    uint32_t m_block_idx, n_block_idx;
    #pragma clang loop licm(disable)
    while (deep_scheduler.fetch_next_work(m_block_idx, n_block_idx)) {
      auto m_coord = m_block_idx;
      auto n_coord = n_block_idx;
      auto l_coord = 0;
      M = deep_scheduler.curr_problem_m();
      auto offset_scalea = deep_scheduler.curr_offset_scalea();
      auto scale_offset_b = deep_scheduler.curr_offset_b(m_block_idx);
      const auto offset_a = deep_scheduler.curr_offset_a(get<2>(params.mainloop.dA));
#if DG_TRAINING_FP8_INLINE_SOURCE>=3
      if(!mega_moe_training::external_fp8_source_tile(inline_source,
          deep_scheduler.ready_ordinal,
          reinterpret_cast<int*>(smem_buf+((Base::SharedStorageSize+127)/128)*128)))return;
#elif DG_TRAINING_FP8_INLINE_SOURCE
      constexpr uint32_t InlineBM=size<0>(TileShape{}),InlineK=TileScheduler::SHAPE_K;
      static_assert(Base::SharedStorageSize>=InlineBM*(20+8*(InlineK/128+1)));
      const uint32_t inline_base=uint32_t(offset_a/K)+uint32_t(m_coord)*InlineBM;
      const uint32_t inline_valid=uint32_t(cute::min(int(InlineBM),int(M)-int(m_coord)*int(InlineBM)));
      const uint32_t inline_ordinal=deep_scheduler.owned_m;
      if(!mega_moe_training::inline_fp8_source_tile<InlineK,InlineBM,DG_TRAINING_FP8_INLINE_SOURCE==2,DG_TRAINING_FP8_INLINE_SOURCE_BULK>(
          inline_source,inline_ordinal,inline_base,inline_valid,uint32_t(get<0>(params.problem_shape)),
          reinterpret_cast<uint8_t*>(const_cast<ElementA*>(params.mainloop.ptr_A)),
          const_cast<float*>(params.mainloop.ptr_scale_A),smem_buf,
          reinterpret_cast<int*>(smem_buf+((Base::SharedStorageSize+127)/128)*128)))return;
#endif

#if DG_TRAINING_FP8_PEER_INPUT && !DG_TRAINING_FP8_INPUT_K_SLICE
      const uint32_t input_row_base = uint32_t(offset_a / K) + uint32_t(m_coord) * size<0>(TileShape{});
      const uint32_t input_valid_rows = uint32_t(cute::min(int(size<0>(TileShape{})), int(M) - int(m_coord) * int(size<0>(TileShape{}))));
      __shared__ int input_failed;
      if (threadIdx.x == 0) {
        input_failed = 0;
        const uint64_t begin = clock64();
        // Native M192 can cover two route-aligned M128 publication tiles.
        for (uint32_t tile = input_row_base / 128;
             tile < (input_row_base + input_valid_rows + 127) / 128; ++tile) {
          if (tile >= input.signal_capacity) { atomicOr(input.status, 2u); input_failed = 1; break; }
          while (tix::ld_acq(input.ready + tile) != uint32_t(DG_TRAINING_FP8_INPUT_READY_COUNT)) {
            if (tix::ld_acq(input.status) || clock64() - begin > input.timeout_cycles) {
              atomicOr(input.status, 8u); input_failed = 1; break;
            }
          }
          if (input_failed) break;
        }
      }
      __syncthreads();
      if (input_failed) return;
#endif
      auto offset_b = deep_scheduler.curr_offset_b(m_block_idx, get<2>(params.mainloop.dB));
      const ElementA* ptr_A = reinterpret_cast<const ElementA*>(params.mainloop.ptr_A) + offset_a;
      const ElementB* weight_base = reinterpret_cast<const ElementB*>(params.mainloop.ptr_B);
      const ElementScale* scale_base = reinterpret_cast<const ElementScale*>(params.mainloop.ptr_scale_B);
      if constexpr (kUseReplicaWeights) {
        const uint32_t expert = deep_scheduler.curr_group_idx;
        if (expert >= params.num_master_groups) {
          offset_b = int64_t(expert - params.num_master_groups) * N * K;
          scale_offset_b = offset_b;
          weight_base = params.rhs_replicas;
          scale_base = params.rhs_replica_scales;
        }
      }
      const ElementB* ptr_B = weight_base + offset_b;
      const ElementScale* ptr_scale_A = reinterpret_cast<const ElementScale*>(params.mainloop.ptr_scale_A) + offset_scalea;
      // Each expert owns ceil(N/128) * ceil(K/128) scale blocks.
      const auto expert_idx = scale_offset_b / (int64_t(N) * int64_t(K));
      const auto scale_stride_b = int64_t(cute::ceil_div(N, 128)) * cute::ceil_div(K, 128);
      const ElementScale* ptr_scale_B = scale_base + expert_idx * scale_stride_b;
      auto mk_layout = make_layout(
        make_shape(make_shape(Int<1>{}, M),
                  make_shape(Int<128>{}, cute::ceil_div(K, 128))),
        strides
      );
      LayoutSFA layout_SFA = make_layout(append(shape(mk_layout), L), append(stride(mk_layout), size(filter_zeros(mk_layout))));

      auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);
      CollectiveMainloop collective_mainloop;
#if DG_TRAINING_FP8_PEER_INPUT && DG_TRAINING_FP8_INPUT_K_SLICE
      collective_mainloop.configure(input, uint32_t(offset_a/K)+uint32_t(m_coord)*size<0>(TileShape{}),
          cute::min(int(size<0>(TileShape{})),int(M)-int(m_coord)*int(size<0>(TileShape{}))),K);
#endif
#if DG_TRAINING_FP8_TILE_PRODUCER
      __shared__ uint64_t producer_addresses[128];
      __shared__ float producer_scores[128];
      collective_mainloop.configure(peers, producer_gradient, producer_scores, input.status, producer_addresses,
          uint32_t(offset_a / K) + uint32_t(m_coord) * 128, uint32_t(m_coord) * 128,
          params.scheduler.shape_m, K, n_coord == 0);
#endif
      // update actual global ptr offset
      MainloopParams update_params = {
        ptr_A, params.mainloop.dA, ptr_B, params.mainloop.dB, 4,
        ptr_scale_A, layout_SFA,
        ptr_scale_B, params.mainloop.layout_SFB
      };
      problem_shape_MNKL = ProblemShape{M, N, K, L};
#if DG_TRAINING_FP8_INLINE_SOURCE==4
      collective_mainloop.configure_broadcast(inline_source.broadcast_sf_tile);
#endif
      auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, blk_coord_mnkl, update_params);
      static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");

      Tensor gA = get<0>(load_inputs);
      Tensor gB = get<1>(load_inputs);
      // Compute tile residues for predication
      auto m_max_coord = M - size<0>(gA) * get<0>(blk_coord_mnkl);                             // M - BLK_M * m_coord
      auto n_max_coord = N - size<0>(gB) * get<1>(blk_coord_mnkl);                             // N - BLK_N * n_coord
      auto k_residue   = K - size<1>(gA) * size<2>(gA);                                        // K - BLK_K * k_coord_max
      auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);
      // Allocate the tiled_mma and the accumulators for the (M,N) blk_shape
      TiledMma tiled_mma;
      Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape)); // (MMA,MMA_M,MMA_N)
      clear(accumulators);

      auto k_tile_iter  = cute::make_coord_iterator(shape<2>(gA));
      int  k_tile_count = size<2>(gA);

      // Perform the collective scoped MMA
      collective_mainloop(
        update_params,
        load_inputs,
        accumulators,
        k_tile_iter, k_tile_count,
        thread_idx,
        smem_buf
      );
#if DG_TRAINING_FP8_TILE_PRODUCER
      if (n_coord == 0) { __threadfence();__syncthreads(); }
#endif
      auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
      const auto coordinates = thread_mma.partition_C(make_identity_tensor(take<0, 2>(blk_shape)));
      const uint32_t row_base = uint32_t(offset_a / K) + uint32_t(m_coord) * size<0>(TileShape{});
      const uint32_t valid_rows = uint32_t(cute::min(int(size<0>(TileShape{})), int(M) - int(m_coord) * int(size<0>(TileShape{}))));
#if DG_TRAINING_FP8_INLINE_SOURCE==4
      CUTLASS_PRAGMA_UNROLL
      for(int element=0;element<cute::size(accumulators);++element) {
          const uint32_t row=cute::get<0>(coordinates(element));
          if(row<valid_rows)accumulators(element)*=__ldg(inline_source.output_row_scores+row_base+row);
      }
#endif
#if DG_TRAINING_FP8_ACTIVATION_BACKWARD
      // Reuse the BF16 dHidden boundary and established split-layout SwiGLU
      // derivative. Quantization follows on the same stream, so its output
      // can safely alias a consumed dY quantization buffer.
      mega_moe_training::activation_backward_epilogue<TileScheduler::SHAPE_N,false>(
          accumulators,coordinates,peers,row_base,valid_rows,uint32_t(n_coord)*size<1>(TileShape{}));
#endif
#if !DG_TRAINING_FP8_EXPERT_END && !DG_TRAINING_FP8_PEER_INPUT && !DG_TRAINING_FP8_RETURN_PIPELINE && !DG_TRAINING_FP8_ACTIVATION_BACKWARD && !DG_TRAINING_FP8_TILE_PRODUCER && !DG_TRAINING_FP8_INLINE_SOURCE
#if DG_TRAINING_FP8_RETURN_STORE_BYTES == 16
      // The Tensor Cell mainloop is finished. Reuse its TSM for one rounded
      // BF16 tile, then issue contiguous local/peer 128-bit stores. No extra
      // global output staging buffer or copy kernel is needed.
      constexpr uint32_t BM=size<0>(TileShape{}),BN=size<1>(TileShape{}),StageN=BN+16;
      constexpr uint32_t Width=TileScheduler::SHAPE_N,Stride=DG_TRAINING_FP8_RETURN_STRIDE;
      static_assert(BN%8==0 && Width%8==0 && Stride>0);
      auto* stage=reinterpret_cast<cutlass::bfloat16_t*>(smem_buf);
      __shared__ uint64_t return_destinations[BM];
      __syncthreads();
      for(uint32_t row=threadIdx.x;row<valid_rows;row+=blockDim.x) {
        const int route=__ldg(peers.permutation+row_base+row);
        uint64_t address=0;
        if(route>=0 && uint64_t(route)<uint64_t(Stride)*peers.num_ranks) {
          address=__ldg(peers.destination_peers+uint32_t(route)/Stride)+
              (uint64_t(uint32_t(route)%Stride)*Width+uint32_t(n_coord)*BN)*sizeof(cutlass::bfloat16_t);
        }
#if DG_TRAINING_FP8_RETURN_REMOTE_MODE
        if constexpr(BN==256)if(address && uint32_t(route)/Stride!=output.communication_ctas-1)address|=uint64_t(1);
#endif
        return_destinations[row]=address;
      }
      CUTLASS_PRAGMA_UNROLL
      for(int wm=0;wm<cute::size<1>(coordinates);++wm) {
        CUTLASS_PRAGMA_UNROLL
        for(int mm=0;mm<cute::size<0,1>(coordinates);++mm) {
          const uint32_t row=cute::get<0>(coordinates(cute::make_coord(cute::_0{},mm,cute::_0{}),wm,cute::_0{}));
          if(row>=valid_rows)continue;
          CUTLASS_PRAGMA_UNROLL
          for(int wn=0;wn<cute::size<2>(coordinates);++wn) {
            CUTLASS_PRAGMA_UNROLL
            for(int mn=0;mn<cute::size<0,2>(coordinates);++mn) {
              const uint32_t column=cute::get<1>(coordinates(cute::make_coord(cute::_0{},mm,mn),wm,wn));
              const auto* values=cute::raw_pointer_cast(accumulators(cute::make_coord(cute::_,mm,mn),wm,wn).data());
              *reinterpret_cast<uint32_t*>(stage+uint64_t(row)*StageN+column)=tix::cvt_rte_bf16x2_f32(values[1],values[0]);
            }
          }
        }
      }
      __syncthreads();
      for(uint32_t index=threadIdx.x;index<valid_rows*(BN/8);index+=blockDim.x) {
        const uint32_t row=index/(BN/8),column=(index%(BN/8))*8;
        if(uint32_t(n_coord)*BN+column>=Width)continue;
        uint4 value;
#if DG_TRAINING_FP8_RETURN_REMOTE_MODE
        if constexpr(BN==256) {
          // Bulk x4 consumes component-major warp registers, unlike a C++
          // thread-major uint4 load. A row contains128 dwords across32 lanes.
          const auto* words=reinterpret_cast<const uint32_t*>(stage+uint64_t(row)*StageN);
          const uint32_t lane=column/8;
          value=uint4{words[lane],words[lane+32],words[lane+64],words[lane+96]};
        } else
#endif
        {value=*reinterpret_cast<const uint4*>(stage+uint64_t(row)*StageN+column);}
        if(return_destinations[row]) {
#if DG_TRAINING_FP8_RETURN_REMOTE_MODE
          if constexpr(BN==256) {
            const uint64_t tagged=return_destinations[row];
            auto* target=reinterpret_cast<void*>((tagged&~uint64_t(1))+column*sizeof(cutlass::bfloat16_t));
            if(tagged&1) {
#if DG_TRAINING_FP8_RETURN_REMOTE_MODE==2 || DG_TRAINING_FP8_RETURN_REMOTE_MODE==4
              __ppu_remote_store_bulk_ncp_b32x4(target,value);
#else
              __ppu_remote_store_bulk_volatile_b32x4(target,value);
#endif
            } else __ppu_global_store_bulk_b32x4(target,value);
          } else *reinterpret_cast<uint4*>(return_destinations[row]+column*sizeof(cutlass::bfloat16_t))=value;
#else
          *reinterpret_cast<uint4*>(return_destinations[row]+column*sizeof(cutlass::bfloat16_t))=value;
#endif
        }
#if DG_TRAINING_FP8_PEER_KEEP_LOCAL
#if DG_TRAINING_FP8_RETURN_REMOTE_MODE
        if constexpr(BN==256)__ppu_global_store_bulk_b32x4(params.epilogue.ptr_D+uint64_t(row_base+row)*Width+uint32_t(n_coord)*BN+column,value);
        else
#endif
        {        *reinterpret_cast<uint4*>(params.epilogue.ptr_D+uint64_t(row_base+row)*Width+uint32_t(n_coord)*BN+column)=value;}
#endif
      }
#if DG_TRAINING_FP8_RETURN_REMOTE_MODE>=3
      __ppu_wait_remote_vstcnt(0);
#endif
      __syncthreads();
#else
      mega_moe_training::scatter_peer_output<cutlass::bfloat16_t, TileScheduler::SHAPE_N, false>(
          accumulators, coordinates, peers, row_base, valid_rows,
          uint32_t(n_coord) * size<1>(TileShape{}));
#endif
#endif
#if DG_TRAINING_FP8_PEER_KEEP_LOCAL && DG_TRAINING_FP8_RETURN_STORE_BYTES != 16 && !DG_TRAINING_FP8_ACTIVATION_BACKWARD
      // update params.epilogue for ptrC and ptrD
      auto params_epilogue_local = params.epilogue;
      params_epilogue_local.ptr_D += deep_scheduler.curr_offset_c(cute::get<2>(params_epilogue_local.dD));
      // Epilogue and write to gD
      CollectiveEpilogue epilogue{params_epilogue_local, shared_storage.tensors.epilogue};
      epilogue(
        problem_shape_MNKL,
        blk_shape,
        blk_coord_mnkl,
        accumulators,
        tiled_mma,
        residue_mnk,
        thread_idx,
        (char*)&shared_storage.tensors.epilogue
      );
#endif
#if DG_TRAINING_FP8_RETURN_PIPELINE
      static_assert((size<0>(TileShape{})==128 || size<0>(TileShape{})==192 || size<0>(TileShape{})==256) && size<1>(TileShape{})==256);
      __threadfence();
      __syncthreads();
      if(threadIdx.x==0) {
        const uint32_t tile=(deep_scheduler.curr_cumsum+m_block_idx)*(TileScheduler::SHAPE_N/256)+uint32_t(n_coord);
        if(tile>=output.signal_capacity)atomicOr(output.status,2u);
        else tix::red_add_rel(output.ready+tile,1u);
      }
      __syncthreads();
#endif
      if constexpr(kEnableSboOverlap && TileScheduler::GEMM_TYPE == GemmType::MGroupedMasked) {
        cp_async_wait<0>();
        __syncthreads();

        if (threadIdx.x == 0) {
          tix::atomic_add_release_global(params.signal + deep_scheduler.curr_group_idx
                  * math::ceil_div(deep_scheduler.params.shape_m, TileScheduler::BLOCK_M) + m_block_idx, 1);
        }
      }
    } // Scheduler work fetch loop
#if DG_TRAINING_FP8_RETURN_REMOTE_MODE && !DG_TRAINING_FP8_RETURN_PIPELINE
    // Match inference CompletedScatterKernel: retain remote writes across MMA
    // tiles, then publish completion before the owner-side IPC barrier.
    __ppu_wait_remote_vstcnt(0);
    __ppu_wait_vstcnt(0);
    __threadfence_system();
    __syncthreads();
#endif
  }
};

// Copy consumers share the producer scheduler, including expert groups and
// its two-M-tile swizzle. They retain row destinations while N tiles advance.
template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_,
          bool EnableSbo, bool UseNStage>
class TrainingFp8PeerTileCopy {
public:
    using Producer=TrainingFp8PeerReturn<ProblemShape_,CollectiveMainloop_,CollectiveEpilogue_,TileScheduler_,EnableSbo,UseNStage>;
    using Params=typename Producer::Params;
    using TileScheduler=TileScheduler_;
    static constexpr int MaxThreadsPerBlock=1024;
    static constexpr int SharedStorageSize=0; // Dynamic TSM; row addresses are static shared.
    static constexpr int MinBlocksPerMultiprocessor=1;
    CUTLASS_DEVICE void operator()(Params const& params,const mega_moe_training::MegaMoEGemmParams& peers,
        const mega_moe_training::MegaMoEDispatchPipeline&,const mega_moe_training::MegaMoEDispatchPipeline& output,uint32_t*,const mega_moe_training::MegaMoEFp8ProducerGradient&,char*) {
        static_assert((TileScheduler::BLOCK_M==128 || TileScheduler::BLOCK_M==192 || TileScheduler::BLOCK_M==256) && TileScheduler::BLOCK_N==256);
        constexpr uint32_t BM=TileScheduler::BLOCK_M;
        static_assert(TileScheduler::GEMM_TYPE==GemmType::MGroupedNoPad);
        constexpr uint32_t width=TileScheduler::SHAPE_N,stride=DG_TRAINING_FP8_RETURN_STRIDE;
        static_assert(width%256==0 && stride>0);
        __shared__ int failed;
        __shared__ uint64_t destinations[BM];
        if(threadIdx.x==0) {
            failed=0;int64_t total=0;
            for(uint32_t g=0;g<TileScheduler::kNumGroups;++g) {
                const int rows=params.scheduler.grouped_layout[g];
                if(rows<0 || rows%128)failed=1;
                total+=rows;
            }
            if(total>params.scheduler.shape_m)failed=1;
            if(failed)atomicOr(output.status,2u);
        }
        __syncthreads();
        if(failed)return;
        TileScheduler scheduler{params.scheduler};
        uint32_t mi,ni;int last_row=-1;
        while(scheduler.fetch_next_work(mi,ni)) {
            const uint32_t row_base=uint32_t(scheduler.curr_offset_m())+mi*BM;
            const uint32_t valid_rows=cute::min(uint32_t(BM),uint32_t(scheduler.curr_problem_m())-mi*BM);
            if(int(row_base)!=last_row) {
                for(uint32_t row=threadIdx.x;row<valid_rows;row+=blockDim.x) {
                    const int route=__ldg(peers.permutation+row_base+row);
                    const bool valid=route>=0 && uint64_t(route)<uint64_t(stride)*peers.num_ranks;
                    if(route>=0 && !valid)atomicOr(output.status,2u);
                    destinations[row]=valid?__ldg(peers.destination_peers+uint32_t(route)/stride)+
                        uint64_t(uint32_t(route)%stride)*width*sizeof(cutlass::bfloat16_t):0;
#if DG_TRAINING_FP8_RETURN_REMOTE_MODE
                    if(valid && uint32_t(route)/stride!=output.communication_ctas-1)destinations[row]|=uint64_t(1);
#endif
                }
                __syncthreads();last_row=row_base;
            }
            if(threadIdx.x==0) {
                failed=0;const uint32_t tile=(scheduler.curr_cumsum+mi)*(width/256)+ni;
                const uint64_t begin=clock64();
                if(tile>=output.signal_capacity){failed=1;atomicOr(output.status,2u);}
                else while(tix::ld_acq(output.ready+tile)!=1u) {
                    if(tix::ld_acq(output.status) || clock64()-begin>output.timeout_cycles) {
                        failed=1;atomicOr(output.status,8u);break;
                    }
                }
            }
            __syncthreads();if(failed)return;
            for(int base=threadIdx.x;base<valid_rows*(256/8);base+=blockDim.x*4) {
                uint4 values[4];uint64_t addresses[4];
                #pragma unroll
                for(int j=0;j<4;++j) {
                    const int item=base+j*blockDim.x,row=item/(256/8),column=(item%(256/8))*8;
                    uint64_t address=0;uint4 value{};
                    if(item<valid_rows*(256/8) && destinations[row]) {
                        address=destinations[row]+uint64_t(ni*256+column)*sizeof(cutlass::bfloat16_t);
#if DG_TRAINING_FP8_RETURN_REMOTE_MODE
                        const auto* words=reinterpret_cast<const uint32_t*>(params.epilogue.ptr_D+
                            int64_t(row_base+row)*width+ni*256);
                        const uint32_t lane=column/8;
                        value=uint4{words[lane],words[lane+32],words[lane+64],words[lane+96]};
#else
                        value=*reinterpret_cast<const uint4*>(params.epilogue.ptr_D+
                            int64_t(row_base+row)*width+ni*256+column);
#endif
                    }
                    values[j]=value;addresses[j]=address;
                }
                #pragma unroll
                for(int j=0;j<4;++j)if(addresses[j]) {
#if DG_TRAINING_FP8_RETURN_REMOTE_MODE
                    auto* target=reinterpret_cast<void*>(addresses[j]&~uint64_t(1));
                    if(addresses[j]&1) {
#if DG_TRAINING_FP8_RETURN_REMOTE_MODE==2 || DG_TRAINING_FP8_RETURN_REMOTE_MODE==4
                        __ppu_remote_store_bulk_ncp_b32x4(target,values[j]);
#else
                        __ppu_remote_store_bulk_volatile_b32x4(target,values[j]);
#endif
                    } else __ppu_global_store_bulk_b32x4(target,values[j]);
#else
                    *reinterpret_cast<uint4*>(addresses[j])=values[j];
#endif
                }
            }
#if DG_TRAINING_FP8_RETURN_REMOTE_MODE>=3
            __ppu_wait_remote_vstcnt(0);
#endif
            __syncthreads();
        }
#if DG_TRAINING_FP8_RETURN_REMOTE_MODE
        __ppu_wait_remote_vstcnt(0);
        __ppu_wait_vstcnt(0);
        __threadfence_system();
#endif
        __threadfence();
    }
};

} // namespace deep_gemm
