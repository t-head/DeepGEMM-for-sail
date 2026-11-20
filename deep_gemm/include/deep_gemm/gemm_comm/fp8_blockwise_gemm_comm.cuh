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

#include <deep_gemm/gemm_comm/reduce_scatter.cuh>

#if __HGGC_ARCH__ == 150
#define __syncthreads() deep_gemm::gemm_comm::mainloop_sync()
#endif
#include <deep_gemm/impls/fp8_gemm.cuh>
#if __HGGC_ARCH__ == 150
#undef __syncthreads
#endif
#if DG_GEMM_RS_REFINED_SFB
#include <cutlass/gemm/collective/ppu_mma_aiu_multistage_with_scale.hpp>
#endif

namespace deep_gemm::gemm_comm {

#if DG_GEMM_RS_REFINED_SFB
namespace detail {

// Keep the native mainloop and replace only its B-scale copy operand.
// BN192 uses three internal N64 slots backed by the original W128 scales.
template <class NativeCopy>
struct Fp8BlockwiseSfbCopy {
    using Tiler_MN = typename NativeCopy::Tiler_MN;

    struct Descriptor {
        int n64, k128;

        template <class Element, bool Trans, int TileN, int TileK, class Stride>
        CUTLASS_DEVICE void init(void*, int n, int k, Stride const&) {
            static_assert(!Trans);
            n64 = n;
            k128 = k;
        }
    } desc_;

    const float* scales;
    int n64_begin;

    struct Slice {
        template <class Engine, class Layout>
        CUTLASS_DEVICE auto partition_S(cute::Tensor<Engine, Layout> const& tensor) const {
            // Carry the K-scale index instead of forming addresses in a fictitious N64 allocation.
            return cute::make_counting_tensor(cute::make_layout(
                cute::make_shape(cute::_1{}, cute::_3{}, cute::_1{}, cute::size<2>(tensor)),
                cute::make_stride(cute::_0{}, cute::_0{}, cute::_0{}, cute::_1{})));
        }

        template <class Engine, class Layout>
        CUTLASS_DEVICE auto partition_D(cute::Tensor<Engine, Layout> const& tensor) const {
            return cute::make_tensor(tensor.data(), cute::make_layout(
                cute::prepend(cute::shape(tensor), cute::_1{}),
                cute::prepend(cute::stride(tensor), cute::_0{})));
        }
    };

    CUTLASS_DEVICE Slice get_slice(int) const { return {}; }
};

struct SfbCopyPredicate {
    bool valid;
    CUTLASS_DEVICE bool operator()() const { return valid; }
};

template <class NativeCopy, class SrcEngine, class SrcLayout, class DstEngine, class DstLayout>
CUTLASS_DEVICE void copy(Fp8BlockwiseSfbCopy<NativeCopy> const& operand,
                        cute::Tensor<SrcEngine, SrcLayout> const& src,
                        cute::Tensor<DstEngine, DstLayout>& dst) {
    const int lane = threadIdx.x % 32;
    if (lane < 3) {
        const int n64 = operand.n64_begin + lane;
        const int k128 = src(0, 0, 0);
        const bool valid = n64 < operand.desc_.n64 && k128 >= 0 && k128 < operand.desc_.k128;
        const float* ptr = operand.scales + (valid ? (n64 / 2) * operand.desc_.k128 + k128 : 0);
        auto src_one = cute::make_tensor(cute::make_gmem_ptr(ptr), cute::Layout<cute::Shape<cute::_1>>{});
        auto dst_one = cute::make_tensor(cute::make_smem_ptr(&dst(0, lane, 0)),
                                         cute::Layout<cute::Shape<cute::_1>>{});
        using Atom = cute::Copy_Atom<cute::PPU_CP_ASYNC_CACHEALWAYS_ZFILL<float>, float>;
        cute::copy_if(Atom{}, SfbCopyPredicate{valid}, src_one, dst_one);
    }
}

template <class Native>
struct WithFp8BlockwiseSfbCopy;

template <class Arch, class Policy, class Tile, class A, class StrideA, class B, class StrideB,
          class Mma, class CopyA, class SmemA, class SmemCopyA, class TransformA,
          class CopyB, class SmemB, class SmemCopyB, class TransformB>
struct WithFp8BlockwiseSfbCopy<cutlass::gemm::collective::CollectiveMma<
    Arch, Policy, Tile, A, StrideA, B, StrideB, Mma,
    CopyA, SmemA, SmemCopyA, TransformA, CopyB, SmemB, SmemCopyB, TransformB>> {
    using Type = cutlass::gemm::collective::CollectiveMma<
        Arch, Policy, Tile, A, StrideA, B, StrideB, Mma,
        CopyA, SmemA, SmemCopyA, TransformA,
        cute::tuple<cute::tuple_element_t<0, CopyB>, Fp8BlockwiseSfbCopy<cute::tuple_element_t<1, CopyB>>>,
        SmemB, SmemCopyB, TransformB>;
};
} // namespace detail

template <class Native>
struct Fp8BlockwiseSfbMainloop : detail::WithFp8BlockwiseSfbCopy<Native>::Type {
    using Base = typename detail::WithFp8BlockwiseSfbCopy<Native>::Type;
    static_assert(Base::ScaleGranularityN == 64 && Base::ScaleGranularityK == 128 && Base::blockK == 128);
    static_assert(!Base::TransSFB && Base::ScaleNsPerTile == 3);

    template <class ProblemShape, class BlockCoord>
    CUTLASS_DEVICE auto load_init(ProblemShape const& shape, BlockCoord const& coord,
                                 typename Base::Params const& params) {
        this->gmem_tiled_copy_SFB.scales = params.ptr_scale_B;
        this->gmem_tiled_copy_SFB.n64_begin = int(cute::get<1>(coord)) * Base::ScaleNsPerTile;
        return Base::load_init(shape, coord, params);
    }
};
#endif

#if __HGGC_ARCH__ == 150
template <int M, int N, int World, int Rank, class Acc, class Mma, class Shape>
__device__ __forceinline__ bool store_full_owner_tile(GemmRSState state, int epoch, const Acc& acc,
                                                      Mma mma, Shape shape, int mt, int nt) {
    using namespace cute;
    constexpr int BM = size<0>(Shape{}), BN = size<1>(Shape{}), LocalM = M / World;
    static_assert(LocalM >= BM && N % BN == 0);
    const int row = int(__ppu_read_firstlane(unsigned(mt))) * BM;
    const int owner = row / LocalM;
    const int owner_begin = owner * LocalM;
    if (row + BM > owner_begin + LocalM)
        return false;

    const int64_t phase = int64_t((epoch - 1) & 1) * state.phase_elements() / 2;
    const int packet_row = row / 16 - owner_begin / 16;
    auto* base = reinterpret_cast<uint32_t*>(state.peer(owner)) + phase +
                 int64_t(Rank) * GemmRSPackets<LocalM>::StorageRows * N / 2 +
                 (packet_row * (N / 16) + int(__ppu_read_firstlane(unsigned(nt))) * (BN / 16)) * 128;
    const auto coordinates =
        mma.get_slice((threadIdx.x / 32) * 32).partition_C(make_identity_tensor(take<0, 2>(shape)));
    const auto store = [&](auto remote) {
#pragma unroll
        for (int i = 0; i < size(Acc{}) / 8; ++i) {
            const auto point = coordinates(i * 8);
            auto* target = base + (get<0>(point) / 16 * (N / 16) + get<1>(point) / 16) * 128;
            const auto packet = pack_accumulator_packet(acc, i * 8);
            if constexpr (decltype(remote)::value)
                __ppu_remote_store_bulk_ncp_b32x4(target, packet);
            else
                __ppu_global_store_bulk_b32x4(target, packet);
        }
    };
    if (owner == Rank)
        store(std::false_type{});
    else
        store(std::true_type{});
    return true;
}
#endif

// Reuse native arithmetic, storage and parameter layouts. Only the scheduler,
// output transport and reduction differ from the ordinary dense GEMM.
template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_,
          bool kEnableSboOverlap, bool kUseNStageKernel>
class Fp8BlockwiseGemmRS {
  public:
#if DG_GEMM_RS_REFINED_SFB
    using CollectiveMainloop = Fp8BlockwiseSfbMainloop<CollectiveMainloop_>;
    static_assert(sizeof(typename CollectiveMainloop::Params) == sizeof(typename CollectiveMainloop_::Params));
    static_assert(alignof(typename CollectiveMainloop::Params) == alignof(typename CollectiveMainloop_::Params));
    using RefinedParams = typename CollectiveMainloop::Params;
    using NativeParams = typename CollectiveMainloop_::Params;
    static_assert(offsetof(RefinedParams, ptr_scale_B) == offsetof(NativeParams, ptr_scale_B));
    static_assert(offsetof(RefinedParams, layout_SFB) == offsetof(NativeParams, layout_SFB));
    static_assert(sizeof(typename CollectiveMainloop::SharedStorage) ==
                  sizeof(typename CollectiveMainloop_::SharedStorage));
#else
    using CollectiveMainloop = CollectiveMainloop_;
#endif
    using Base = DeepGemmUniversal<ProblemShape_, CollectiveMainloop, CollectiveEpilogue_, TileScheduler_,
                                   kEnableSboOverlap, kUseNStageKernel>;
    static_assert(!kEnableSboOverlap && !kUseNStageKernel);
    static_assert(TileScheduler_::GEMM_TYPE == GemmType::Normal);
    using ProblemShape = typename Base::ProblemShape;
    using TileShape = typename Base::TileShape;
    using TiledMma = typename Base::TiledMma;
    using TileScheduler = typename Base::TileScheduler;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementScale = typename Base::ElementScale;
    using LayoutSFA = typename Base::LayoutSFA;
    using MainloopParams = typename CollectiveMainloop::Params;
    using SharedStorage = typename Base::SharedStorage;
    using Params = typename Base::Params;
    static_assert(DG_GEMM_RS_SHARED_BUDGET >= Base::SharedStorageSize);
    static constexpr int SharedStorageSize = DG_GEMM_RS_SHARED_BUDGET;
    static constexpr int MaxThreadsPerBlock = Base::MaxThreadsPerBlock;
    static constexpr int MinBlocksPerMultiprocessor = Base::MinBlocksPerMultiprocessor;

    CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
        using namespace cute;
        constexpr int M = DG_GEMM_RS_M, N = TileScheduler::SHAPE_N, K = TileScheduler::SHAPE_K;
        constexpr int World = DG_GEMM_RS_WORLD, Rank = DG_GEMM_RS_RANK;
        GemmRSState state{static_cast<const int64_t*>(params.workspace)};
        const int epoch = begin_epoch(state, smem_buf);
        auto scheduler_params = params.scheduler;
        scheduler_params.shape_m = M;
        TileScheduler scheduler{scheduler_params};
        auto tile_shape = TileShape{};
        auto strides = make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, M));
        uint32_t mt, nt;
#pragma clang loop licm(disable)
        while (scheduler.fetch_next_work(mt, nt)) {
            mt = owner_balanced_m_tile<M, World, Rank, size<0>(TileShape{})>(mt);
            const ElementA* ptr_a = reinterpret_cast<const ElementA*>(params.mainloop.ptr_A);
            const ElementB* ptr_b = reinterpret_cast<const ElementB*>(params.mainloop.ptr_B);
            auto scale_layout = make_layout(
                make_shape(make_shape(Int<1>{}, M), make_shape(Int<128>{}, cute::ceil_div(K, 128))), strides);
            LayoutSFA layout_sfa = make_layout(append(shape(scale_layout), 1),
                                               append(stride(scale_layout), size(filter_zeros(scale_layout))));
            MainloopParams mainloop_params = {ptr_a,
                                              params.mainloop.dA,
                                              ptr_b,
                                              params.mainloop.dB,
                                              4,
                                              reinterpret_cast<const ElementScale*>(params.mainloop.ptr_scale_A),
                                              layout_sfa,
                                              reinterpret_cast<const ElementScale*>(params.mainloop.ptr_scale_B),
                                              params.mainloop.layout_SFB};
            CollectiveMainloop mainloop;
            auto inputs = mainloop.load_init(ProblemShape{M, N, K, 1}, make_coord(mt, nt, _, 0), mainloop_params);
            Tensor gA = get<0>(inputs);
            TiledMma mma;
            Tensor acc = partition_fragment_C(mma, take<0, 2>(tile_shape));
            clear(acc);
            auto k_iter = make_coord_iterator(shape<2>(gA));
            const int k_count = size<2>(gA);
            mainloop(mainloop_params, inputs, acc, k_iter, k_count, int(threadIdx.x), smem_buf);
#if __HGGC_ARCH__ == 150
            constexpr int LocalM = M / World;
            // Reuse the peer address when generic packets can cross a shard boundary.
            if constexpr (World > 1 && LocalM >= size<0>(TileShape{}) && N % size<1>(TileShape{}) == 0 &&
                          (World == 2 || !GemmRSPackets<LocalM>::Aligned)) {
                if (store_full_owner_tile<M, N, World, Rank>(state, epoch, acc, mma, tile_shape, mt, nt))
                    continue;
            }
#endif
            store_packets<M, N, World, Rank>(state, epoch, acc, mma, tile_shape, mt, nt, smem_buf);
        }
        finish_tiles<M, N, World, Rank, size<0>(TileShape{}), size<1>(TileShape{})>(
            state, epoch, reinterpret_cast<__ppu_bfloat16*>(params.epilogue.ptr_D));
    }
};

} // namespace deep_gemm::gemm_comm
