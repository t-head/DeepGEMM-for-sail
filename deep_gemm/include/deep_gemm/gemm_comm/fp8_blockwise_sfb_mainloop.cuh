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

#include <cutlass/gemm/collective/ppu_mma_aiu_multistage_with_scale.hpp>

namespace deep_gemm::gemm_comm {
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
} // namespace deep_gemm::gemm_comm
