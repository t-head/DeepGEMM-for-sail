#pragma once

#include <deep_gemm/impls/fp8_gemm.cuh>
#include <deep_gemm/common/epilogue_utils.cuh>
#include <deep_gemm/scheduler/k_grouped_scheduler.cuh>
#include <cutlass/gemm/collective/collective_builder.hpp>

namespace deep_gemm {
using namespace cute;
template <uint32_t STRIDE_CM, class TAcc, class TCcC>
__device__ __forceinline__ void native_dw_overwrite_float4(
        TAcc& accum, const TCcC& tCcC, float* c_ptr, uint32_t row_base,
        uint32_t n_offset) {
    using namespace cute;
    const bool odd_lane = (threadIdx.x & 1) != 0;
    CUTLASS_PRAGMA_UNROLL
    for (int wm = 0; wm < size<1>(tCcC); ++wm) {
        CUTLASS_PRAGMA_UNROLL
        for (int mm = 0; mm < size<0,1>(tCcC); ++mm) {
            const uint32_t row = get<0>(tCcC(make_coord(_0{}, mm, _0{}), wm, _0{}));
            constexpr int kVectors = size<2>(TCcC{});
            CUTLASS_PRAGMA_UNROLL
            for (int wn = 0; wn < kVectors; ++wn) {
                const float* p0 = raw_pointer_cast(accum(make_coord(_, mm, _0{}), wm, wn).data());
                const float* p1 = raw_pointer_cast(accum(make_coord(_, mm, _1{}), wm, wn).data());
                const float peer00 = __shfl_xor_sync(0xffffffffu, p0[0], 1);
                const float peer01 = __shfl_xor_sync(0xffffffffu, p0[1], 1);
                const float peer10 = __shfl_xor_sync(0xffffffffu, p1[0], 1);
                const float peer11 = __shfl_xor_sync(0xffffffffu, p1[1], 1);
                const float4 packed = odd_lane
                    ? float4{peer10, peer11, p1[0], p1[1]}
                    : float4{p0[0], p0[1], peer00, peer01};
                const float4 value = packed;
                const uint32_t col = get<1>(tCcC(make_coord(_0{}, mm, _0{}), wm, wn)) + (odd_lane ? 6 : 0);
                float* dst = c_ptr + static_cast<size_t>(row_base + row) * STRIDE_CM + n_offset + col;
                cutlass::arch::global_store<float4, sizeof(float4)>(value, dst, true);
            }
        }
    }
}



// Preserve the native blockwise collective, including its scale copy pipeline
// and FP32 MixAccumulation. Only expert K bounds and the FP32 epilogue differ.
template<uint32_t M,uint32_t N,uint32_t G,uint32_t BM,uint32_t BN,
         uint32_t BK,uint32_t WM,uint32_t WN,int Stages,bool FirstOverwrite,uint32_t SwizzleM=2,bool GlobalLengths=false,uint32_t ScaleM=1>
__device__ __forceinline__ void native_fp8_k_grouped_gemm_impl(
        KGroupedGemmArgs args,const float* sa,const float* sb,
        const int32_t* offsets,const int32_t* first_flags) {
    using Tile=cute::Shape<Int<BM>,Int<BN>,Int<BK>>;
    static_assert(BM%ScaleM==0 && M%ScaleM==0);
    using ScaleShape=cute::Shape<Int<ScaleM>,_128,_128>;
    using ScaleConfig=decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<ScaleShape,false,true>(ScaleShape{}));
    using SFA=decltype(ScaleConfig::deduce_layoutSFA());
    using SFB=decltype(ScaleConfig::deduce_layoutSFB());
    using Mainloop=typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::PPU0015,cutlass::arch::OpClassTensorOp,
        cutlass::float_e4m3_t,cute::tuple<cutlass::layout::RowMajor,SFA>,1,
        cutlass::float_e4m3_t,cute::tuple<cutlass::layout::ColumnMajor,SFB>,1,
        float,Tile,cute::Shape<Int<WM>,Int<WN>,Int<BK>>,Int<Stages>,
        cutlass::gemm::KernelAiuMultistageWithBlockWiseScale>::CollectiveOp;
    using Mma=typename Mainloop::TiledMma;
    using Scheduler=KGroupedScheduler<GemmType::KGroupedContiguous,N,BM,BN,G,math::ceil_div(N,BN),SwizzleM>;
    extern __shared__ __align__(128) uint8_t native_smem[];
    Scheduler scheduler(args.grouped_layout,M);
    uint32_t expert,mi,ni;
    Mma mma;
    Tensor accum=partition_fragment_C(mma,take<0,2>(Tile{}));
    Tensor identity=make_identity_tensor(cute::Shape<Int<BM>,Int<BN>>{});
    Tensor coords=mma.get_thread_slice(threadIdx.x).partition_C(identity);
    while(scheduler.fetch_next_work(expert,mi,ni)) {
        const uint32_t k=GlobalLengths?math::ceil_div(scheduler.current_shape_k,uint32_t(128))*128:scheduler.current_shape_k;
        const uint32_t start=uint32_t(offsets[expert]);
        const auto shape=make_shape(int(M),int(N),int(k),1);
        const auto stride_a=make_stride(int64_t(k),_1{},int64_t(M)*k);
        const auto stride_b=make_stride(int64_t(k),_1{},int64_t(N)*k);
        typename Mainloop::Params params={
            reinterpret_cast<const cutlass::float_e4m3_t*>(args.a_ptr)+uint64_t(start)*M,stride_a,
            reinterpret_cast<const cutlass::float_e4m3_t*>(args.b_ptr)+uint64_t(start)*N,stride_b,4,
            sa+uint64_t(start/128)*(M/ScaleM),ScaleConfig::tile_atom_to_shape_SFA(shape),
            sb+uint64_t(start/128)*(N/128),ScaleConfig::tile_atom_to_shape_SFB(shape)};
        Mainloop mainloop;
        const auto block_coord=make_coord(mi,ni,_,0);
        auto inputs=mainloop.load_init(shape,block_coord,params);
        auto ga=get<0>(inputs);
        auto k_iter=make_coord_iterator(cute::shape<2>(ga));
        int k_count=size<2>(ga);
        clear(accum);
        mainloop(params,inputs,accum,k_iter,k_count,int(threadIdx.x),reinterpret_cast<char*>(native_smem));
        float* output=reinterpret_cast<float*>(args.c_ptr)+uint64_t(expert)*M*N;
        const uint32_t row=mi*BM,column=ni*BN,valid=min(BM,M-row);
        auto store=[&](auto accumulate_tag) {
            constexpr bool Add=decltype(accumulate_tag)::value;
            using Coord=decltype(coords);
            if constexpr(N%BN==0 && N%4==0 && size<0,2>(Coord{})==2) {
                if(valid==BM && (reinterpret_cast<uintptr_t>(output)&15)==0) {
                    if constexpr(Add)epilogue_no_tsm_float4<N>(accum,coords,output,row,column);
                    else native_dw_overwrite_float4<N>(accum,coords,output,row,column);
                    return;
                }
            }
            epilogue_no_tsm<float,float,N,BN,N,Add,false>(accum,coords,output,row,valid,column);
        };
        if constexpr(FirstOverwrite) {
            if(__ldg(first_flags+expert))store(cute::false_type{});
            else store(cute::true_type{});
        } else store(cute::true_type{});
        cp_async_wait<0>();
        __syncthreads();
    }
}
} // namespace deep_gemm
