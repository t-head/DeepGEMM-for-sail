#pragma once
#include <deep_gemm/impls/mega_moe_training/mega_moe_training.cuh>

#ifndef DG_TRAINING_FP8_INLINE_SOURCE_BULK
#define DG_TRAINING_FP8_INLINE_SOURCE_BULK 0
#endif
#ifndef DG_TRAINING_FP8_INLINE_SOURCE_BULK_ILP
#define DG_TRAINING_FP8_INLINE_SOURCE_BULK_ILP 4
#endif
namespace deep_gemm::mega_moe_training {
__device__ __forceinline__ bool external_fp8_source_tile(
        const MegaMoEFp8InlineSource& src,uint32_t ordinal,int* control) {
    if(threadIdx.x==0) {
        control[0]=0;
        if(ordinal>=src.signal_capacity){atomicOr(src.status,2u);control[0]=1;}
        else {
            const uint64_t begin=clock64();
            while(tix::ld_acq_sys(src.ready+ordinal)!=2u) {
                if(tix::ld_acq_sys(src.status) || clock64()-begin>src.timeout_cycles) {
                    atomicOr(src.status,8u);control[0]=1;break;
                }
            }
        }
    }
    __syncthreads();return control[0]==0;
}
// Source scratch is dead before the native AIU prologue uses this arena.
template<uint32_t Width,uint32_t BM,bool Gradient,bool Bulk=false>
__device__ __forceinline__ bool inline_fp8_source_tile(
        const MegaMoEFp8InlineSource& src,uint32_t ordinal,uint32_t row_base,uint32_t valid,
        uint32_t capacity,uint8_t* quantized,float* scales,char* arena,int* control) {
    constexpr uint32_t Segments=Width/128,ILP=8;
    static_assert(Width%128==0 && BM%4==0);
    const uint32_t thread=threadIdx.x;
    if(thread==0) {
        control[1]=0;
        if(ordinal>=src.signal_capacity){atomicOr(src.status,2u);control[1]=1;}
        else control[0]=int(atomicCAS(src.ready+ordinal,0u,1u));
    }
    __syncthreads();
    if(control[1])return false;
    if(control[0]==0) {
        auto* addresses=reinterpret_cast<uint64_t*>(arena);
        auto* auxiliaries=addresses+BM;
        auto* scores=reinterpret_cast<float*>(auxiliaries+BM);
        auto* tile_scales=scores+BM;
        auto* tile_dots=tile_scales+BM*(Segments+1);
        constexpr uint32_t Pitch=Segments+1;
        const uint64_t stride=uint64_t(src.tokens)*src.topk;
        for(uint32_t row=thread;row<valid;row+=blockDim.x) {
            const int route=__ldg(src.permutation+row_base+row);
            const bool good=route>=0 && uint64_t(route)<stride*src.num_ranks;
            if(route>=0 && !good)atomicOr(src.status,2u);
            const uint32_t owner=good?uint64_t(route)/stride:0;
            const uint64_t token=good?(uint64_t(route)%stride)/src.topk:0;
            addresses[row]=good?__ldg(src.value_peers+owner)+token*Width*(Gradient?2:1):0;
            if constexpr(Bulk && !Gradient)if(good && owner!=src.local_rank)addresses[row]|=uint64_t(1);
            auxiliaries[row]=good?__ldg(src.aux_peers+owner):0;
            if constexpr(Gradient)scores[row]=good?reinterpret_cast<const float*>(auxiliaries[row])[uint64_t(route)%stride]:0.f;
            else auxiliaries[row]=good?auxiliaries[row]+token*Segments*sizeof(float):0;
        }
        __syncthreads();
        if constexpr(Bulk && !Gradient && Width%512==0) {
            constexpr uint32_t BulkILP=DG_TRAINING_FP8_INLINE_SOURCE_BULK_ILP;
            static_assert(BulkILP==4 || BulkILP==8 || BulkILP==16);
            const uint32_t lane=thread&31,warp=thread>>5;
            for(uint32_t row=warp;row<valid;row+=blockDim.x/32) {
                const uint64_t tagged=addresses[row];
                const auto* source=reinterpret_cast<const uint4*>(tagged&~uint64_t(1));
                auto* target=reinterpret_cast<uint4*>(quantized+uint64_t(row_base+row)*Width);
                for(uint32_t first=0;first<Width/512;first+=BulkILP) {
                    uint4 values[BulkILP];
                    #pragma unroll
                    for(uint32_t j=0;j<BulkILP;++j) {
                        values[j]=uint4{};
                        if(first+j<Width/512 && tagged) {
                            // Hardware byte gates require warp-uniform bulk bases here.
                            const auto* at=source+(first+j)*32;
                            values[j]=(tagged&1)?__ppu_remote_load_bulk_b32x4(at):__ppu_global_ldcg_bulk_b32x4(at);
                        }
                    }
                    #pragma unroll
                    for(uint32_t j=0;j<BulkILP;++j)if(first+j<Width/512)
                        __ppu_global_store_bulk_b32x4(target+(first+j)*32,values[j]);
                }
                for(uint32_t segment=lane;segment<Segments;segment+=32)
                    tile_scales[row*Pitch+segment]=tagged?reinterpret_cast<const float*>(auxiliaries[row])[segment]:0x1p-120f;
            }
            __ppu_wait_vstcnt(0);
        } else {
        constexpr uint32_t Lanes=Gradient?16:8;
        const uint32_t lane=thread&(Lanes-1),groups=blockDim.x/Lanes;
        for(uint32_t first=thread/Lanes;first<valid*Segments;first+=groups*ILP) {
            uint4 fetched[ILP],fetched_y[Gradient?ILP:1];
            float fetched_scale[Gradient?1:ILP];
            #pragma unroll
            for(uint32_t j=0;j<ILP;++j) {
                const uint32_t group=first+j*groups,row=group/Segments,segment=group%Segments;
                fetched[j]=make_uint4(0,0,0,0);
                if constexpr(Gradient)fetched_y[j]=make_uint4(0,0,0,0);
                else fetched_scale[j]=0x1p-120f;
                if(group<valid*Segments && addresses[row]) {
                    const uint32_t col=segment*128+lane*(Gradient?8:16);
                    fetched[j]=*reinterpret_cast<const uint4*>(addresses[row]+col*(Gradient?2:1));
                    if constexpr(Gradient)fetched_y[j]=*reinterpret_cast<const uint4*>(
                        reinterpret_cast<const __ppu_bfloat16*>(src.unweighted_y)+uint64_t(row_base+row)*Width+col);
                    else if(lane==0)fetched_scale[j]=reinterpret_cast<const float*>(auxiliaries[row])[segment];
                }
            }
            #pragma unroll
            for(uint32_t j=0;j<ILP;++j) {
                const uint32_t group=first+j*groups,row=group/Segments,segment=group%Segments;
                if(group>=valid*Segments)continue;
                const uint32_t col=segment*128+lane*(Gradient?8:16);
                const uint64_t at=uint64_t(row_base+row)*Width+col;
                if constexpr(Gradient) {
                    const auto* dy=reinterpret_cast<const __ppu_bfloat16*>(&fetched[j]);
                    const auto* y=reinterpret_cast<const __ppu_bfloat16*>(&fetched_y[j]);
                    uint4 weighted{};auto* values=reinterpret_cast<__ppu_bfloat16*>(&weighted);float dot=0.f;
                    #pragma unroll
                    for(uint32_t v=0;v<8;++v){const float d=__bfloat162float(dy[v]);values[v]=__float2bfloat16(d*scores[row]);dot+=d*__bfloat162float(y[v]);}
                    const unsigned mask=0xffffu<<((thread&31)>=16?16:0);
                    #pragma unroll
                    for(int delta=8;delta>0;delta>>=1)dot+=__shfl_down_sync(mask,dot,delta,16);
                    if(lane==0)tile_dots[row*Pitch+segment]=dot;
                    *reinterpret_cast<uint4*>(reinterpret_cast<__ppu_bfloat16*>(src.packed_dy)+at)=weighted;
                    float sf;const uint2 q=peer_quantize_eight(weighted,sf);
                    *reinterpret_cast<uint2*>(quantized+at)=q;
                    if(lane==0)tile_scales[row*Pitch+segment]=sf;
                } else {
                    *reinterpret_cast<uint4*>(quantized+at)=fetched[j];
                    if(lane==0)tile_scales[row*Pitch+segment]=fetched_scale[j];
                }
            }
        }
        }
        __syncthreads();
        for(uint32_t at=thread;at<Segments*(valid/4);at+=blockDim.x) {
            const uint32_t segment=at/(valid/4),row=(at%(valid/4))*4;
            *reinterpret_cast<uint4*>(scales+uint64_t(segment)*capacity+row_base+row)=make_uint4(
                __float_as_uint(tile_scales[row*Pitch+segment]),__float_as_uint(tile_scales[(row+1)*Pitch+segment]),
                __float_as_uint(tile_scales[(row+2)*Pitch+segment]),__float_as_uint(tile_scales[(row+3)*Pitch+segment]));
        }
        if constexpr(Gradient) {
            for(uint32_t at=thread;at<valid*Segments;at+=blockDim.x)
                src.score_partials[uint64_t(row_base+at/Segments)*Segments+at%Segments]=tile_dots[(at/Segments)*Pitch+at%Segments];
        }
        __threadfence_system();
        __syncthreads();
        if(thread==0)tix::red_add_rel_sys(reinterpret_cast<int*>(src.ready+ordinal),1);
    } else if(thread==0) {
        const uint64_t begin=clock64();
        while(tix::ld_acq_sys(src.ready+ordinal)!=2u) {
            if(tix::ld_acq_sys(src.status) || clock64()-begin>src.timeout_cycles){atomicOr(src.status,8u);control[1]=1;break;}
        }
    }
    __syncthreads();
    return control[1]==0;
}
} // namespace deep_gemm::mega_moe_training
