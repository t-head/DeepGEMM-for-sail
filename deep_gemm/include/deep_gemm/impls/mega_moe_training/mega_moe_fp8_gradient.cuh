#pragma once


#ifndef DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_MOE_FP8_GRADIENT_WEIGHTED_REFERENCE_CUH
#define DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_MOE_FP8_GRADIENT_WEIGHTED_REFERENCE_CUH
#include <deep_gemm/impls/mega_moe_training/mega_moe_training.cuh>

namespace deep_gemm::mega_moe_training {

// Independent power2 scales permit a single owner-side pass. The low packet
// retains BF16 gradient detail for later DW preparation; dHidden reads high.
template<int Op, int Width, int Threads>
__device__ __forceinline__ void fp8_gradient_weighted(const Params& p) {
    constexpr int segments=Width/128;
    const int thread=threadIdx.x,lane=thread&15;
    if constexpr(Op==0) {
        const int tokens=p.dim[0],topk=p.dim[1],experts=p.dim[3];
        const int valid=__ldg(peer_pointer<int32_t>(p,4));
        if(valid<0 || valid>tokens) {
            if(thread==0 && blockIdx.x==0)atomicOr(peer_pointer<uint32_t>(p,9),2u);
            return;
        }
        const unsigned mask=0xffffu<<((thread&31)>=16?16:0);
        for(int64_t route=int64_t(blockIdx.x)*(Threads/16)+thread/16;
            route<int64_t(tokens)*topk;route+=int64_t(gridDim.x)*(Threads/16)) {
            const int token=route/topk;
            const int64_t expert=__ldg(peer_pointer<int64_t>(p,3)+route);
            const bool good=token<valid && expert>=0 && expert<experts;
            if(token<valid && expert>=experts && lane==0)atomicOr(peer_pointer<uint32_t>(p,9),2u);
            const float score=good?__ldg(peer_pointer<float>(p,1)+route):0.f;
            for(int segment=0;segment<segments;++segment) {
                const int column=segment*128+lane*8;
                const uint4 dy=good?*reinterpret_cast<const uint4*>(
                    peer_pointer<__ppu_bfloat16>(p,0)+int64_t(token)*Width+column):uint4{};
                const uint4 y=good?*reinterpret_cast<const uint4*>(
                    peer_pointer<__ppu_bfloat16>(p,2)+route*Width+column):uint4{};
                const auto* dv=reinterpret_cast<const __ppu_bfloat16*>(&dy);
                const auto* yv=reinterpret_cast<const __ppu_bfloat16*>(&y);
                uint4 weighted{};
                auto* wv=reinterpret_cast<__ppu_bfloat16*>(&weighted);
                float dot=0.f;
                #pragma unroll
                for(int j=0;j<8;++j) {
                    const float d=__bfloat162float(dv[j]);
                    wv[j]=__float2bfloat16(d*score);
                    dot+=d*__bfloat162float(yv[j]);
                }
                unsigned words[4]={weighted.x,weighted.y,weighted.z,weighted.w};
                unsigned maximum=0;
                #pragma unroll
                for(int j=0;j<4;++j) {
                    maximum=max(maximum,words[j]&0x7fffu);
                    maximum=max(maximum,(words[j]>>16)&0x7fffu);
                }
                #pragma unroll
                for(int delta=8;delta;delta>>=1) {
                    maximum=max(maximum,__shfl_down_sync(mask,maximum,delta,16));
                    dot+=__shfl_down_sync(mask,dot,delta,16);
                }
                maximum=__shfl_sync(mask,maximum,0,16);
                const float target=fmaxf(__uint_as_float(maximum<<16)/448.f,0x1p-120f);
                const unsigned bits=__float_as_uint(target);
                const unsigned exponent=(bits&0x7f800000u)+((bits&0x7fffffu)?0x800000u:0u);
                const float scale=__uint_as_float(exponent),low_scale=scale*(1.f/16.f);
                unsigned high_words[4];
                #pragma unroll
                for(int j=0;j<4;++j)high_words[j]=epilogue::quant::fp8_pair(
                    __uint_as_float(words[j]<<16)/scale,__uint_as_float(words[j]&0xffff0000u)/scale);
                const uint2 high=make_uint2(high_words[0]|(high_words[1]<<16),high_words[2]|(high_words[3]<<16));
                const auto* hv=reinterpret_cast<const cutlass::float_e4m3_t*>(&high);
                unsigned low_words[4];
                #pragma unroll
                for(int j=0;j<4;++j)low_words[j]=epilogue::quant::fp8_pair(
                    (__uint_as_float(words[j]<<16)-float(hv[2*j])*scale)/low_scale,
                    (__uint_as_float(words[j]&0xffff0000u)-float(hv[2*j+1])*scale)/low_scale);
                const uint2 low=make_uint2(low_words[0]|(low_words[1]<<16),low_words[2]|(low_words[3]<<16));
                *reinterpret_cast<uint2*>(peer_pointer<uint8_t>(p,5)+route*Width+column)=high;
                *reinterpret_cast<uint2*>(peer_pointer<uint8_t>(p,6)+route*Width+column)=low;
                if(lane==0) {
                    peer_pointer<float>(p,7)[route*segments+segment]=scale;
                    peer_pointer<float>(p,8)[route*segments+segment]=dot;
                }
            }
        }
    } else {
        // Match the existing 32-lane score-return sum and its feature order.
        const int warp=thread/32,warp_lane=thread&31;
        for(int64_t row=int64_t(blockIdx.x)*(Threads/32)+warp;row<p.dim[0];
            row+=int64_t(gridDim.x)*(Threads/32)) {
            float dot=0.f;
            for(int column=warp_lane;column<segments;column+=32)
                dot+=peer_pointer<float>(p,0)[row*segments+column];
            #pragma unroll
            for(int delta=16;delta;delta>>=1)dot+=__shfl_down_sync(0xffffffffu,dot,delta);
            if(warp_lane==0)peer_pointer<float>(p,1)[row]=dot;
        }
    }
}
} // namespace deep_gemm::mega_moe_training

#endif

#ifndef DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_MOE_FP8_GRADIENT_RAW_OWNER_CUH
#define DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_MOE_FP8_GRADIENT_RAW_OWNER_CUH

#ifndef DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_MOE_FP8_GRADIENT_CODEC_CUH
#define DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_MOE_FP8_GRADIENT_CODEC_CUH
#include <deep_gemm/impls/mega_moe_training/mega_moe_training.cuh>
namespace deep_gemm::mega_moe_training {
__device__ __forceinline__ uint2 gradient_quantize_with_scale(const uint4 raw,float scale) {
    const unsigned words[4]={raw.x,raw.y,raw.z,raw.w};unsigned q[4];
    #pragma unroll
    for(int j=0;j<4;++j)q[j]=epilogue::quant::fp8_pair(
        __uint_as_float(words[j]<<16)/scale,__uint_as_float(words[j]&0xffff0000u)/scale);
    return make_uint2(q[0]|(q[1]<<16),q[2]|(q[3]<<16));
}

__device__ __forceinline__ uint2 gradient_raw_quantize_eight(const uint4 dy,float& scale,uint2& low) {
    const unsigned mask=0xffffu<<((threadIdx.x&31)>=16?16:0);
            const unsigned words[4]={dy.x,dy.y,dy.z,dy.w};
            unsigned maximum=0;
            #pragma unroll
            for(int j=0;j<4;++j) {maximum=max(maximum,words[j]&0x7fffu);maximum=max(maximum,(words[j]>>16)&0x7fffu);}
            #pragma unroll
            for(int delta=8;delta;delta>>=1)maximum=max(maximum,__shfl_down_sync(mask,maximum,delta,16));
            maximum=__shfl_sync(mask,maximum,0,16);
            const unsigned bits=__float_as_uint(fmaxf(__uint_as_float(maximum<<16)/448.f,0x1p-120f));
            scale=__uint_as_float((bits&0x7f800000u)+((bits&0x7fffffu)?0x800000u:0u));
            unsigned h[4];
            #pragma unroll
            for(int j=0;j<4;++j)h[j]=epilogue::quant::fp8_pair(__uint_as_float(words[j]<<16)/scale,__uint_as_float(words[j]&0xffff0000u)/scale);
            const uint2 high=make_uint2(h[0]|(h[1]<<16),h[2]|(h[3]<<16));
            const auto* hv=reinterpret_cast<const cutlass::float_e4m3_t*>(&high);
            unsigned l[4];
            #pragma unroll
            for(int j=0;j<4;++j)l[j]=epilogue::quant::fp8_pair(
                (__uint_as_float(words[j]<<16)-float(hv[2*j])*scale)/(scale*(1.f/16.f)),
                (__uint_as_float(words[j]&0xffff0000u)-float(hv[2*j+1])*scale)/(scale*(1.f/16.f)));
            low=make_uint2(l[0]|(l[1]<<16),l[2]|(l[3]<<16));
    return high;
}
}

#endif
namespace deep_gemm::mega_moe_training {
template<int Width,int Threads>
__device__ __forceinline__ void gradient_raw_owner(const Params& p) {
    constexpr int segments=Width/128;
    const int thread=threadIdx.x,lane=thread&15,tokens=p.dim[0],topk=p.dim[1],experts=p.dim[3];
    const int valid=__ldg(peer_pointer<int32_t>(p,4));
    if(valid<0 || valid>tokens) {
        if(thread==0 && blockIdx.x==0)atomicOr(peer_pointer<uint32_t>(p,9),2u);
        return;
    }
    const unsigned mask=0xffffu<<((thread&31)>=16?16:0);
    for(int token=blockIdx.x*(Threads/16)+thread/16;token<tokens;token+=gridDim.x*(Threads/16)) {
        for(int segment=0;segment<segments;++segment) {
            const int column=segment*128+lane*8;
            const uint4 dy=token<valid?*reinterpret_cast<const uint4*>(
                peer_pointer<__ppu_bfloat16>(p,0)+int64_t(token)*Width+column):uint4{};
            const auto* dv=reinterpret_cast<const __ppu_bfloat16*>(&dy);
            // Complete every owner-Y reader before token-ready publication.
            for(int choice=0;choice<topk;++choice) {
                const int64_t route=int64_t(token)*topk+choice;
                const int64_t expert=__ldg(peer_pointer<int64_t>(p,3)+route);
                const bool good=token<valid && expert>=0 && expert<experts;
                if(token<valid && expert>=experts && lane==0)atomicOr(peer_pointer<uint32_t>(p,9),2u);
                const uint4 y=good?*reinterpret_cast<const uint4*>(peer_pointer<__ppu_bfloat16>(p,2)+route*Width+column):uint4{};
                const auto* yv=reinterpret_cast<const __ppu_bfloat16*>(&y);
                float dot=0.f;
                #pragma unroll
                for(int j=0;j<8;++j)dot+=__bfloat162float(dv[j])*__bfloat162float(yv[j]);
                #pragma unroll
                for(int delta=8;delta;delta>>=1)dot+=__shfl_down_sync(mask,dot,delta,16);
                if(lane==0)peer_pointer<float>(p,8)[route*segments+segment]=dot;
            }
            float scale;uint2 low;
            const uint2 high=gradient_raw_quantize_eight(dy,scale,low);
            *reinterpret_cast<uint2*>(peer_pointer<uint8_t>(p,5)+int64_t(token)*Width+column)=high;
            *reinterpret_cast<uint2*>(peer_pointer<uint8_t>(p,6)+int64_t(token)*Width+column)=low;
            if(lane==0)peer_pointer<float>(p,7)[int64_t(token)*segments+segment]=scale;
        }
    }
}
} // namespace deep_gemm::mega_moe_training

#endif

#ifndef DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_MOE_FP8_GRADIENT_TRANSPORT_CUH
#define DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_MOE_FP8_GRADIENT_TRANSPORT_CUH
#include <deep_gemm/impls/mega_moe_training/mega_moe_dw_block128.cuh>

#ifndef DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_MOE_FP8_GRADIENT_REFERENCE_CUH
#define DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_MOE_FP8_GRADIENT_REFERENCE_CUH
namespace deep_gemm::mega_moe_training {

template<int Threads>
__device__ __forceinline__ void gradient_subset_metadata(const Params& p,int pointer) {
    if(blockIdx.x!=0 || p.ptr[pointer]==nullptr || p.dim[9]==0)return;
    auto* records=reinterpret_cast<uint4*>(p.ptr[pointer]);
    uint32_t block_prefix=0,global_prefix=0;
    for(int expert=0;expert<p.dim[6];++expert) {
        const uint32_t length=uint32_t(__ldg(peer_pointer<int32_t>(p,4)+expert));
        if(expert>=p.dim[5]) {
            const uint32_t blocks=(uint64_t(length)+p.dim[9]-1)/p.dim[9];
            for(uint32_t block=threadIdx.x;block<blocks;block+=Threads)
                records[1+block_prefix+block]=make_uint4(expert,length,block_prefix,global_prefix);
            block_prefix+=blocks;
        }
        global_prefix+=length;
    }
    if(threadIdx.x==0)reinterpret_cast<int32_t*>(records)[0]=block_prefix;
}

// Numerical control only: BF16 transport, followed by the same raw-packet
// encode/decode recipe on the expert. This is not a performance baseline.
template<int Width,int Threads,bool Shared=false>
__device__ __forceinline__ void gradient_gather_bf16_reference(const Params& p) {
    constexpr int segments=Width/128;
    const int thread=threadIdx.x,lane=thread&15;
    int64_t prefix=0,rows=0;
    for(int expert=0;expert<p.dim[6];++expert) {
        const int64_t count=__ldg(peer_pointer<int32_t>(p,4)+expert);
        if(count<0 || count%128) {
            if(thread==0 && blockIdx.x==0)atomicOr(peer_pointer<uint32_t>(p,9),2u);
            return;
        }
        if(expert<p.dim[5])prefix+=count;else rows+=count;
    }
    if(prefix+rows>p.dim[0]) {
        if(thread==0 && blockIdx.x==0)atomicOr(peer_pointer<uint32_t>(p,9),2u);
        return;
    }
    gradient_subset_metadata<Threads>(p,11);
    const uint64_t stride=uint64_t(p.dim[2])*p.dim[3];
    for(int64_t group=int64_t(blockIdx.x)*(Threads/16)+thread/16;group<rows*segments;group+=int64_t(gridDim.x)*(Threads/16)) {
        const int64_t row=prefix+group/segments;
        const int segment=group%segments,column=segment*128+lane*8;
        const int route=__ldg(peer_pointer<int32_t>(p,3)+row);
        const bool good=route>=0 && uint64_t(route)<stride*p.dim[4];
        if(route>=0 && !good && lane==0)atomicOr(peer_pointer<uint32_t>(p,9),2u);
        const uint32_t owner=good?uint64_t(route)/stride:0;
        const uint64_t local=good?uint64_t(route)%stride:0,token=local/p.dim[3];
        const float score=good?reinterpret_cast<const float*>(__ldg(peer_pointer<uint64_t>(p,2)+owner))[local]:0.f;
        const uint4 dy=good?*reinterpret_cast<const uint4*>(__ldg(peer_pointer<uint64_t>(p,0)+owner)+
            (token*Width+column)*sizeof(__ppu_bfloat16)):uint4{};
        float scale;uint2 low;
        const uint2 high=[&]() {
            if constexpr(Shared) {
                scale=good?reinterpret_cast<const float*>(__ldg(peer_pointer<uint64_t>(p,1)+owner))[token*segments+segment]:0x1p-120f;
                low=uint2{};return gradient_quantize_with_scale(dy,scale);
            } else return gradient_raw_quantize_eight(dy,scale,low);
        }();
        uint2 weighted_high=high;
        const uint32_t sign=(__float_as_uint(score)>>31)?0x80808080u:0u;
        weighted_high.x^=sign;weighted_high.y^=sign;
        *reinterpret_cast<uint2*>(peer_pointer<uint8_t>(p,5)+row*Width+column)=weighted_high;
        const auto* hv=reinterpret_cast<const cutlass::float_e4m3_t*>(&high);
        const auto* lv=reinterpret_cast<const cutlass::float_e4m3_t*>(&low);
        uint4 weighted{};auto* output=reinterpret_cast<__ppu_bfloat16*>(&weighted);
        #pragma unroll
        for(int j=0;j<8;++j) {
            const auto raw=__float2bfloat16(float(hv[j])*scale+float(lv[j])*(scale*(1.f/16.f)));
            output[j]=__float2bfloat16(__bfloat162float(raw)*score);
        }
        *reinterpret_cast<uint4*>(peer_pointer<__ppu_bfloat16>(p,10)+row*Width+column)=weighted;
        if(lane==0) {
            peer_pointer<float>(p,6)[int64_t(segment)*p.dim[0]+row]=scale*fabsf(score);
            peer_pointer<float>(p,7)[int64_t(segment)*p.dim[0]+row]=scale;
            if(segment==0)peer_pointer<float>(p,8)[row]=score;
        }
    }
}
} // namespace deep_gemm::mega_moe_training

#endif
namespace deep_gemm::mega_moe_training {
template<int Width,int Threads>
__device__ __forceinline__ void gradient_gather_high(const Params& p) {
    constexpr int segments=Width/128;
    const int thread=threadIdx.x,lane=thread&7;
    const auto* lengths=peer_pointer<int32_t>(p,4);
    int64_t prefix=0,rows=0;
    for(int g=0;g<p.dim[6];++g) {
        const int64_t count=__ldg(lengths+g);
        if(count<0 || count%128) {
            if(thread==0 && blockIdx.x==0)atomicOr(peer_pointer<uint32_t>(p,9),2u);
            return;
        }
        if(g<p.dim[5])prefix+=count;else rows+=count;
    }
    if(prefix+rows>p.dim[0]) {
        if(thread==0 && blockIdx.x==0)atomicOr(peer_pointer<uint32_t>(p,9),2u);
        return;
    }
    gradient_subset_metadata<Threads>(p,10);
    const uint64_t stride=uint64_t(p.dim[2])*p.dim[3];
    for(int64_t group=int64_t(blockIdx.x)*(Threads/8)+thread/8;group<rows*segments;group+=int64_t(gridDim.x)*(Threads/8)) {
        const int64_t row=prefix+group/segments;
        const int feature=group%segments;
        const int route=__ldg(peer_pointer<int32_t>(p,3)+row);
        const bool good=route>=0 && uint64_t(route)<stride*p.dim[4];
        if(route>=0 && !good && lane==0)atomicOr(peer_pointer<uint32_t>(p,9),2u);
        const uint32_t owner=good?uint64_t(route)/stride:0;
        const uint64_t local=good?uint64_t(route)%stride:0,token=local/p.dim[3];
        const float score=good?reinterpret_cast<const float*>(__ldg(peer_pointer<uint64_t>(p,2)+owner))[local]:0.f;
        const uint32_t sign=(__float_as_uint(score)>>31)?0x80808080u:0u;
        uint4 q{};
        if(good)q=*reinterpret_cast<const uint4*>(__ldg(peer_pointer<uint64_t>(p,0)+owner)+token*Width+feature*128+lane*16);
        q.x^=sign;q.y^=sign;q.z^=sign;q.w^=sign;
        *reinterpret_cast<uint4*>(peer_pointer<uint8_t>(p,5)+row*Width+feature*128+lane*16)=q;
        if(lane==0) {
            const float scale=good?reinterpret_cast<const float*>(__ldg(peer_pointer<uint64_t>(p,1)+owner))[token*segments+feature]:0x1p-120f;
            peer_pointer<float>(p,6)[int64_t(feature)*p.dim[0]+row]=scale*fabsf(score);
            peer_pointer<float>(p,7)[int64_t(feature)*p.dim[0]+row]=scale;
            if(feature==0)peer_pointer<float>(p,8)[row]=score;
        }
    }
}

// Functional reference; profile this retained BF16 panel before introducing
// a two-pass shared-memory implementation for co-residency.
template<int Width,int Threads>
__device__ __forceinline__ void gradient_dw_left(const Params& p) {
    constexpr int segments=Width/128,Cached=128*16/Threads;
    const int thread=threadIdx.x,lane=thread&15;
    __shared__ int starts[128],counts[128],padded[128],tiles[129],valid;
    __shared__ uint32_t tile[128*32],maxima[Threads/32];
    if(thread==0) {
        bool okay=true;tiles[0]=0;
        for(int g=0;g<p.dim[0];++g) {
            const int64_t count=__ldg(peer_pointer<int32_t>(p,5)+g);
            const int64_t start=__ldg(peer_pointer<int32_t>(p,6)+g),end=__ldg(peer_pointer<int32_t>(p,6)+g+1);
            const int64_t rounded=count<0?0:((count+127)/128)*128;
            const bool good=count>=0 && start>=0 && start%128==0 && end>=start+rounded && end<=p.dim[1];
            okay=okay && good;starts[g]=good?start:0;counts[g]=good?count:0;padded[g]=good?rounded:0;
            tiles[g+1]=tiles[g]+(good?rounded/128:0);
        }
        valid=okay;
        if(!okay)atomicOr(peer_pointer<uint32_t>(p,9),2u);
    }
    __syncthreads();
    if(!valid)return;
    const uint64_t stride=uint64_t(p.dim[3])*p.dim[4];
    for(int task=blockIdx.x;task<tiles[p.dim[0]]*segments;task+=gridDim.x) {
        const int ordinal=task/segments,feature_base=(task%segments)*128;
        int expert=0;
        while(ordinal>=tiles[expert+1])++expert;
        const int token_base=(ordinal-tiles[expert])*128,start=starts[expert],base=start+token_base;
        uint4 weighted[Cached];
        #pragma unroll
        for(int j=0;j<Cached;++j) {
            const int row=thread/16+j*(Threads/16);
            weighted[j]=uint4{};
            if(token_base+row>=counts[expert])continue;
            const int route=__ldg(peer_pointer<int32_t>(p,3)+base+row);
            const bool good=route>=0 && uint64_t(route)<stride*p.dim[5];
            if(route>=0 && !good && lane==0)atomicOr(peer_pointer<uint32_t>(p,9),2u);
            if(!good)continue;
            const uint32_t owner=uint64_t(route)/stride;
            const uint64_t token=(uint64_t(route)%stride)/p.dim[4];
            const int column=feature_base+lane*8;
            const float score=__ldg(peer_pointer<float>(p,2)+base+row);
            const uint32_t sign=(__float_as_uint(score)>>31)?0x80808080u:0u;
            uint2 high=*reinterpret_cast<const uint2*>(peer_pointer<uint8_t>(p,0)+int64_t(base+row)*Width+column);
            high.x^=sign;high.y^=sign;
            const uint2 low=*reinterpret_cast<const uint2*>(__ldg(peer_pointer<uint64_t>(p,4)+owner)+token*Width+column);
            const float scale=__ldg(peer_pointer<float>(p,1)+int64_t(feature_base/128)*p.dim[1]+base+row);
            const auto* hv=reinterpret_cast<const cutlass::float_e4m3_t*>(&high);
            const auto* lv=reinterpret_cast<const cutlass::float_e4m3_t*>(&low);
            auto* out=reinterpret_cast<__ppu_bfloat16*>(&weighted[j]);
            #pragma unroll
            for(int k=0;k<8;++k) {
                const auto raw=__float2bfloat16(float(hv[k])*scale+float(lv[k])*(scale*(1.f/16.f)));
                out[k]=__float2bfloat16(__bfloat162float(raw)*score);
            }
        }
        mega_moe_dw_emit_block128<128,Threads,true,false,true>(weighted,peer_pointer<uint8_t>(p,7),peer_pointer<float>(p,8),
            Width,start,padded[expert],token_base,feature_base,tile,maxima);
    }
}
} // namespace deep_gemm::mega_moe_training

#endif

#ifndef DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_MOE_FP8_GRADIENT_TILED_CUH
#define DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_MOE_FP8_GRADIENT_TILED_CUH
namespace deep_gemm::mega_moe_training {

template<int Width,int Threads,bool Pure=false,bool Bf16=false,bool WaitOwner=false,int ILP=2>
__device__ __forceinline__ void gradient_gather_high_tiled(const Params& p) {
    constexpr int segments=Width/128;
    const auto* meta=reinterpret_cast<const uint4*>(p.ptr[10]);
    auto* ready=peer_pointer<uint32_t>(p,11);
    const int count=reinterpret_cast<const int32_t*>(meta)[0];
    // API requires enough signals for every capacity row plus expert tails.
    const int signal_capacity=(p.dim[0]+p.dim[9]-1)/p.dim[9]+p.dim[6];
    if(count<0 || count>signal_capacity) {
        if(threadIdx.x==0)atomicOr(peer_pointer<uint32_t>(p,9),2u);
        return;
    }
    if(threadIdx.x==0)tix::red_add_rel_sys(reinterpret_cast<int*>(ready+signal_capacity),1);
    constexpr int Lanes=Bf16?16:8;
    const int thread=threadIdx.x,lane=thread&(Lanes-1);
    __shared__ uint64_t row_address[256],row_scale_address[Pure?1:256];
    __shared__ float row_score[256];
    const uint64_t stride=uint64_t(p.dim[2])*p.dim[3];
    for(int task=blockIdx.x;task<count;task+=gridDim.x) {
        const uint4 record=meta[1+task];
        const uint32_t local=(uint32_t(task)-record.z)*p.dim[9];
        const uint32_t valid=min(uint32_t(p.dim[9]),record.y-local),base=record.w+local;
        if(record.x>=uint32_t(p.dim[6]) || record.z>uint32_t(task) || local>=record.y ||
                uint64_t(base)+valid>uint32_t(p.dim[0])) {
            if(thread==0)atomicOr(peer_pointer<uint32_t>(p,9),2u);
            return;
        }
        for(uint32_t row=thread;row<valid;row+=Threads) {
            const int route=__ldg(peer_pointer<int32_t>(p,3)+base+row);
            const bool good=route>=0 && uint64_t(route)<stride*p.dim[4];
            if(route>=0 && !good)atomicOr(peer_pointer<uint32_t>(p,9),2u);
            const uint32_t owner=good?uint64_t(route)/stride:0;
            const uint64_t local_route=good?uint64_t(route)%stride:0,token=local_route/p.dim[3];
            if constexpr(WaitOwner)if(good) {
                const auto* token_ready=reinterpret_cast<const uint32_t*>(__ldg(peer_pointer<uint64_t>(p,1)+owner));
                const uint64_t began=clock64();
                while(tix::ld_acq_sys(token_ready+token/128)!=1u) {
                    if(tix::ld_acq_sys(peer_pointer<uint32_t>(p,9)) || clock64()-began>39000000000ull) {
                        atomicOr(peer_pointer<uint32_t>(p,9),8u);return;
                    }
                }
            }
            row_address[row]=good?__ldg(peer_pointer<uint64_t>(p,0)+owner)+token*Width*(Bf16?2:1):0;
            if constexpr(!Pure)row_scale_address[row]=good?__ldg(peer_pointer<uint64_t>(p,1)+owner)+token*segments*sizeof(float):0;
            row_score[row]=good?reinterpret_cast<const float*>(__ldg(peer_pointer<uint64_t>(p,2)+owner))[local_route]:0.f;
        }
        __syncthreads();
        for(uint32_t first=thread/Lanes;first<valid*segments;first+=(Threads/Lanes)*ILP) {
            uint4 data[ILP];float fetched_sf[ILP];
            #pragma unroll
            for(int j=0;j<ILP;++j) {
                const uint32_t group=first+j*(Threads/Lanes),row=group/segments,feature=group%segments;
                data[j]=uint4{};fetched_sf[j]=0x1p-120f;
                if(group<valid*segments && row_address[row]) {
                    data[j]=*reinterpret_cast<const uint4*>(row_address[row]+(feature*128+lane*(Bf16?8:16))*(Bf16?2:1));
                    if constexpr(!Pure)if(lane==0)fetched_sf[j]=reinterpret_cast<const float*>(row_scale_address[row])[feature];
                }
            }
            #pragma unroll
            for(int j=0;j<ILP;++j) {
                const uint32_t group=first+j*(Threads/Lanes),within=group/segments,feature=group%segments,row=base+within;
                if(group>=valid*segments)continue;
                const float score=row_score[within];
                if constexpr(Pure) {
                    if constexpr(Bf16) {
                        const float scale=__ldg(peer_pointer<float>(p,1)+feature*p.dim[9]);
                        const uint2 q=gradient_quantize_with_scale(data[j],scale);
                        *reinterpret_cast<uint2*>(peer_pointer<uint8_t>(p,5)+int64_t(row)*Width+feature*128+lane*8)=q;
                    } else *reinterpret_cast<uint4*>(peer_pointer<uint8_t>(p,5)+int64_t(row)*Width+feature*128+lane*16)=data[j];
                    if(lane==0 && feature==0)peer_pointer<float>(p,8)[row]=score;
                } else {
                    const uint32_t sign=(__float_as_uint(score)>>31)?0x80808080u:0u;
                    auto q=data[j];q.x^=sign;q.y^=sign;q.z^=sign;q.w^=sign;
                    *reinterpret_cast<uint4*>(peer_pointer<uint8_t>(p,5)+int64_t(row)*Width+feature*128+lane*16)=q;
                    if(lane==0) {
                        peer_pointer<float>(p,6)[int64_t(feature)*p.dim[0]+row]=fetched_sf[j]*fabsf(score);
                        peer_pointer<float>(p,7)[int64_t(feature)*p.dim[0]+row]=fetched_sf[j];
                        if(feature==0)peer_pointer<float>(p,8)[row]=score;
                    }
                }
            }
        }
        __threadfence_system();__syncthreads();
        if(thread==0)tix::red_add_rel_sys(reinterpret_cast<int*>(ready+task),2);
        __syncthreads();
    }
}


// Warp-uniform 512-byte rows pair PPU bulk load/store register layouts.
// The ordinary path remains available for every configuration and narrow width.
template<int Width,int Threads,bool WaitOwner,int ILP=4,bool Rotate=false>
__device__ __forceinline__ void gradient_gather_high_bulk(const Params& p) {
    if constexpr(Width<512 || Width%512) {
        gradient_gather_high_tiled<Width,Threads,true,false,WaitOwner>(p);
        return;
    } else {
        const auto* meta=reinterpret_cast<const uint4*>(p.ptr[10]);
        auto* ready=peer_pointer<uint32_t>(p,11);
        const int count=reinterpret_cast<const int32_t*>(meta)[0];
        const int signal_capacity=(p.dim[0]+p.dim[9]-1)/p.dim[9]+p.dim[6];
        if(count<0 || count>signal_capacity) {
            if(threadIdx.x==0)atomicOr(peer_pointer<uint32_t>(p,9),2u);
            return;
        }
        if(threadIdx.x==0)tix::red_add_rel_sys(reinterpret_cast<int*>(ready+signal_capacity),1);
        __shared__ uint64_t row_address[256];
        __shared__ float row_score[256];
        const uint64_t stride=uint64_t(p.dim[2])*p.dim[3];
        const int thread=threadIdx.x,lane=thread&31,warp=thread>>5;
        for(int ordinal=blockIdx.x;ordinal<count;ordinal+=gridDim.x) {
            const uint4 record=meta[1+ordinal];
            uint32_t task=ordinal;
            if constexpr(Rotate) {
                if(record.x>=uint32_t(p.dim[6]) || !record.y || record.z>uint32_t(ordinal)) {
                    if(thread==0)atomicOr(peer_pointer<uint32_t>(p,9),2u);
                    return;
                }
                const uint32_t blocks=(record.y+uint32_t(p.dim[9])-1)/uint32_t(p.dim[9]);
                task=record.z+(uint32_t(ordinal)-record.z+uint32_t(p.dim[11])*(blocks/uint32_t(p.dim[4])+1))%blocks;
            }
            const uint32_t local=(uint32_t(task)-record.z)*p.dim[9];
            const uint32_t valid=min(uint32_t(p.dim[9]),record.y-local),base=record.w+local;
            if(record.x>=uint32_t(p.dim[6]) || record.z>uint32_t(task) || local>=record.y ||
                    uint64_t(base)+valid>uint32_t(p.dim[0])) {
                if(thread==0)atomicOr(peer_pointer<uint32_t>(p,9),2u);
                return;
            }
            for(uint32_t row=thread;row<valid;row+=Threads) {
                const int route=__ldg(peer_pointer<int32_t>(p,3)+base+row);
                const bool good=route>=0 && uint64_t(route)<stride*p.dim[4];
                if(route>=0 && !good)atomicOr(peer_pointer<uint32_t>(p,9),2u);
                const uint32_t owner=good?uint64_t(route)/stride:0;
                const uint64_t local_route=good?uint64_t(route)%stride:0,token=local_route/p.dim[3];
                if constexpr(WaitOwner)if(good) {
                    const auto* token_ready=reinterpret_cast<const uint32_t*>(__ldg(peer_pointer<uint64_t>(p,1)+owner));
                    const uint64_t began=clock64();
                    while(tix::ld_acq_sys(token_ready+token/128)!=1u) {
                        if(tix::ld_acq_sys(peer_pointer<uint32_t>(p,9)) || clock64()-began>39000000000ull) {
                            atomicOr(peer_pointer<uint32_t>(p,9),8u);return;
                        }
                    }
                }
                row_address[row]=good?(__ldg(peer_pointer<uint64_t>(p,0)+owner)+token*Width)|uint64_t(owner!=p.dim[11]):0;
                row_score[row]=good?reinterpret_cast<const float*>(__ldg(peer_pointer<uint64_t>(p,2)+owner))[local_route]:0.f;
            }
            __syncthreads();
            for(uint32_t row=warp;row<valid;row+=Threads/32) {
                const uint64_t tagged=row_address[row];
                const auto* source=reinterpret_cast<const uint4*>(tagged&~uint64_t(1));
                auto* target=reinterpret_cast<uint4*>(peer_pointer<uint8_t>(p,5)+int64_t(base+row)*Width);
                for(int first=0;first<Width/512;first+=ILP) {
                    uint4 values[ILP];
                    #pragma unroll
                    for(int j=0;j<ILP;++j) {
                        values[j]=uint4{};
                        if(first+j<Width/512 && tagged) {
                            const auto* at=source+(first+j)*32;
                            values[j]=(tagged&1)?__ppu_remote_load_bulk_b32x4(at):__ppu_global_ldcg_bulk_b32x4(at);
                        }
                    }
                    #pragma unroll
                    for(int j=0;j<ILP;++j)if(first+j<Width/512)
                        __ppu_global_store_bulk_b32x4(target+(first+j)*32,values[j]);
                }
                if(lane==0)peer_pointer<float>(p,8)[base+row]=row_score[row];
            }
            __ppu_wait_vstcnt(0);
            __threadfence_system();__syncthreads();
            if(thread==0)tix::red_add_rel_sys(reinterpret_cast<int*>(ready+task),2);
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void gradient_producer_started(const Params& p) {
    if(threadIdx.x!=0)return;
    const auto* ready=peer_pointer<uint32_t>(p,0);
    auto* status=peer_pointer<uint32_t>(p,1);
    const uint64_t begin=clock64();
    while(tix::ld_acq_sys(ready+p.dim[0]-1)<uint32_t(p.dim[1])) {
        if(tix::ld_acq_sys(status) || clock64()-begin>uint64_t(39000000000ull)) {
            atomicOr(status,8u);return;
        }
    }
}
} // namespace deep_gemm::mega_moe_training

#endif

#ifndef DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_MOE_FP8_GRADIENT_SHARED_CUH
#define DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_MOE_FP8_GRADIENT_SHARED_CUH
namespace deep_gemm::mega_moe_training {

template<int Width,int Threads>
__device__ __forceinline__ void gradient_shared_owner(const Params& p) {
    constexpr int segments=Width/128;
    const int thread=threadIdx.x,lane=thread&15,valid=__ldg(peer_pointer<int32_t>(p,4));
    if(valid<0 || valid>p.dim[0]) {
        if(thread==0 && blockIdx.x==0)atomicOr(peer_pointer<uint32_t>(p,9),2u);
        return;
    }
    for(int64_t group=int64_t(blockIdx.x)*(Threads/16)+thread/16;
            group<int64_t(p.dim[0])*segments;group+=int64_t(gridDim.x)*(Threads/16)) {
        const int token=group/segments,segment=group%segments;
        const int64_t at=int64_t(token)*Width+segment*128+lane*8;
        const float scale=fmaxf(__ldg(peer_pointer<float>(p,10)+segment)/448.f,0x1p-120f);
        const uint4 raw=token<valid?*reinterpret_cast<const uint4*>(peer_pointer<__ppu_bfloat16>(p,0)+at):uint4{};
        *reinterpret_cast<uint2*>(peer_pointer<uint8_t>(p,5)+at)=gradient_quantize_with_scale(raw,scale);
        if(lane==0)peer_pointer<float>(p,7)[group]=scale;
    }
}

template<int Width,int Threads,int ILP=4>
__device__ __forceinline__ void gradient_owner_score_parallel(const Params& p) {
    constexpr int segments=Width/128;
    const int thread=threadIdx.x,lane=thread&15,groups=Threads/16;
    const int valid=__ldg(peer_pointer<int32_t>(p,4));
    if(valid<0 || valid>p.dim[0]) {
        if(thread==0 && blockIdx.x==0)atomicOr(peer_pointer<uint32_t>(p,9),2u);
        return;
    }
    const unsigned mask=0xffffu<<((thread&31)>=16?16:0);
    const int64_t total=int64_t(p.dim[0])*p.dim[1]*segments;
    for(int64_t first=int64_t(blockIdx.x)*groups*ILP+thread/16;first<total;
            first+=int64_t(gridDim.x)*groups*ILP) {
        uint4 dy[ILP],y[ILP];
        #pragma unroll
        for(int j=0;j<ILP;++j) {
            const int64_t at=first+j*groups,route=at/segments,token=route/p.dim[1];
            const int column=(at%segments)*128+lane*8;
            const int64_t expert=at<total?__ldg(peer_pointer<int64_t>(p,3)+route):-1;
            const bool good=at<total && token<valid && expert>=0 && expert<p.dim[3];
            dy[j]=good?*reinterpret_cast<const uint4*>(peer_pointer<__ppu_bfloat16>(p,0)+token*Width+column):uint4{};
            y[j]=good?*reinterpret_cast<const uint4*>(peer_pointer<__ppu_bfloat16>(p,2)+route*Width+column):uint4{};
        }
        #pragma unroll
        for(int j=0;j<ILP;++j) {
            const int64_t at=first+j*groups;
            if(at>=total)continue;
            const auto* dv=reinterpret_cast<const __ppu_bfloat16*>(&dy[j]);
            const auto* yv=reinterpret_cast<const __ppu_bfloat16*>(&y[j]);
            float dot=0.f;
            #pragma unroll
            for(int k=0;k<8;++k)dot+=__bfloat162float(dv[k])*__bfloat162float(yv[k]);
            #pragma unroll
            for(int delta=8;delta;delta>>=1)dot+=__shfl_down_sync(mask,dot,delta,16);
            if(lane==0)peer_pointer<float>(p,8)[at]=dot;
        }
    }
}

template<int Width,int Threads,bool Signed=false>
__device__ __forceinline__ void gradient_score_folded_hidden(const Params& p) {
    constexpr int segments=Width/128,Cached=128*16/Threads;
    const int thread=threadIdx.x;
    const auto* lengths=peer_pointer<int32_t>(p,2);
    const auto* offsets=peer_pointer<int32_t>(p,3);
    int total=0;
    for(int g=0;g<p.dim[0];++g) {
        const int64_t count=__ldg(lengths+g),start=__ldg(offsets+g),end=__ldg(offsets+g+1);
        const int64_t padded=count<0?0:((count+127)/128)*128;
        if(count<0 || start<0 || start%128 || end<start+padded || end>p.dim[1]) {
            if(thread==0 && blockIdx.x==0)atomicOr(peer_pointer<uint32_t>(p,6),2u);
            return;
        }
        total+=padded/128;
    }
    __shared__ uint32_t tile[128*32],maxima[Threads/32];
    for(int task=blockIdx.x;task<total*segments;task+=gridDim.x) {
        int ordinal=task/segments,start=0,count=0,padded=0,base=0;
        for(int g=0;g<p.dim[0];++g) {
            const int s=__ldg(offsets+g),c=__ldg(lengths+g),k=((int64_t(c)+127)/128)*128;
            if(ordinal<k/128){start=s;count=c;padded=k;base=s+ordinal*128;break;}
            ordinal-=k/128;
        }
        const int feature=(task%segments)*128;
        uint4 cached[Cached];
        #pragma unroll
        for(int j=0;j<Cached;++j) {
            const int vector=thread+j*Threads,row=vector/16,column=(vector%16)*8;
            cached[j]=uint4{};
            if(base+row<start+count) {
                cached[j]=*reinterpret_cast<const uint4*>(peer_pointer<__ppu_bfloat16>(p,0)+int64_t(base+row)*Width+feature+column);
                const float value=__ldg(peer_pointer<float>(p,1)+base+row);
                const float score=Signed?value:fabsf(value);
                auto* values=reinterpret_cast<__ppu_bfloat16*>(&cached[j]);
                #pragma unroll
                for(int k=0;k<8;++k)values[k]=__float2bfloat16(__bfloat162float(values[k])*score);
            }
        }
        mega_moe_dw_emit_block128<128,Threads,false,false,true>(cached,peer_pointer<uint8_t>(p,4),peer_pointer<float>(p,5),
            Width,start,padded,base-start,feature,tile,maxima);
    }
}
} // namespace deep_gemm::mega_moe_training

#endif

#ifndef DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_MOE_FP8_GRADIENT_BYTES_CUH
#define DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_MOE_FP8_GRADIENT_BYTES_CUH
namespace deep_gemm::mega_moe_training {

// Independent token tiles: every CTA publishes only its own current DY bytes.
// Pointer ABI: DY, q, valid, SFtile, token_ready, status. Six pointers total.
template<int Width,int Threads>
__device__ __forceinline__ void gradient_token_owner(const Params& p) {
    constexpr int segments=Width/128;
    const int thread=threadIdx.x,lane=thread&15;
    auto* ready=peer_pointer<uint32_t>(p,4);
    const int tiles=(p.dim[0]+127)/128;
    if(thread==0)tix::red_add_rel_sys(reinterpret_cast<int*>(ready+tiles),1);
    const int valid=__ldg(peer_pointer<int32_t>(p,2));
    if(valid<0 || valid>p.dim[0]) {
        if(thread==0)atomicOr(peer_pointer<uint32_t>(p,5),2u);
        return;
    }
    for(int token_tile=blockIdx.x;token_tile<tiles;token_tile+=gridDim.x) {
    const int begin=token_tile*128,end=min(int(p.dim[0]),begin+128);
    for(int group=thread/16;group<(end-begin)*segments;group+=Threads/16) {
        const int token=begin+group/segments,feature=group%segments;
        const int64_t at=int64_t(token)*Width+feature*128+lane*8;
        const float sf=__ldg(peer_pointer<float>(p,3)+feature*p.dim[2]);
        const uint4 value=token<valid?*reinterpret_cast<const uint4*>(peer_pointer<__ppu_bfloat16>(p,0)+at):uint4{};
        *reinterpret_cast<uint2*>(peer_pointer<uint8_t>(p,1)+at)=gradient_quantize_with_scale(value,sf);
    }
    __threadfence_system();__syncthreads();
    if(thread==0)tix::red_add_rel_sys(reinterpret_cast<int*>(ready+token_tile),1);
    __syncthreads();
    }
}

template<int Width,int Threads>
__device__ __forceinline__ void gradient_expand_shared_scales(const Params& p) {
    constexpr int segments=Width/128;
    for(int at=blockIdx.x*Threads+threadIdx.x;at<segments*p.dim[1];at+=gridDim.x*Threads)
        peer_pointer<float>(p,1)[at]=fmaxf(__ldg(peer_pointer<float>(p,0)+at/p.dim[1])/448.f,0x1p-120f);
}

template<int Width,int Threads>
__device__ __forceinline__ void gradient_transpose_shared_left(const Params& p) {
    constexpr int segments=Width/128;
    const int thread=threadIdx.x;
    const auto* lengths=peer_pointer<int32_t>(p,2);
    const auto* offsets=peer_pointer<int32_t>(p,3);
    int total=0;
    for(int g=0;g<p.dim[0];++g) {
        const int64_t count=__ldg(lengths+g),start=__ldg(offsets+g),end=__ldg(offsets+g+1);
        const int64_t padded=count<0?0:((count+127)/128)*128;
        if(count<0 || start<0 || start%128 || end<start+padded || end>p.dim[1]) {
            if(thread==0 && blockIdx.x==0)atomicOr(peer_pointer<uint32_t>(p,6),2u);
            return;
        }
        total+=padded/128;
    }
    __shared__ uint32_t tile[128*32];
    auto slot=[](int row,int word){return row*32+(word^(row&3)^((row>>2)&3)^(row>>4));};
    for(int task=blockIdx.x;task<total*segments;task+=gridDim.x) {
        int ordinal=task/segments,start=0,count=0,padded=0,base=0;
        for(int g=0;g<p.dim[0];++g) {
            const int s=__ldg(offsets+g),c=__ldg(lengths+g),k=((int64_t(c)+127)/128)*128;
            if(ordinal<k/128){start=s;count=c;padded=k;base=s+ordinal*128;break;}
            ordinal-=k/128;
        }
        const int feature=(task%segments)*128;
        const float scale=__ldg(peer_pointer<float>(p,1)+(feature/128)*p.dim[3]);
        if(thread<128)peer_pointer<float>(p,5)[int64_t(base/128)*Width+feature+thread]=scale;
        for(int vector=thread;vector<1024;vector+=Threads) {
            const int row=vector/8,column=(vector%8)*16,word=column/4;
            const uint4 raw=base+row<start+count?*reinterpret_cast<const uint4*>(peer_pointer<uint8_t>(p,0)+
                int64_t(base+row)*Width+feature+column):uint4{};
            tile[slot(row,word)]=raw.x;tile[slot(row,word+1)]=raw.y;
            tile[slot(row,word+2)]=raw.z;tile[slot(row,word+3)]=raw.w;
        }
        __syncthreads();
        const int lane4=thread&3;
        for(int group=thread/4;group<256;group+=Threads/4) {
            const int col=(group/8)*4,token=(group%8)*16,row=token+lane4*4,word=col/4;
            const unsigned a=tile[slot(row,word)],b=tile[slot(row+1,word)],c=tile[slot(row+2,word)],d=tile[slot(row+3,word)];
            const unsigned ab01=__byte_perm(a,b,0x5140u),cd01=__byte_perm(c,d,0x5140u);
            const unsigned ab23=__byte_perm(a,b,0x7362u),cd23=__byte_perm(c,d,0x7362u);
            unsigned v0=__byte_perm(ab01,cd01,0x5410u),v1=__byte_perm(ab01,cd01,0x7632u);
            unsigned v2=__byte_perm(ab23,cd23,0x5410u),v3=__byte_perm(ab23,cd23,0x7632u);
            auto swap=[&](unsigned& lo,unsigned& hi,int bit){const unsigned send=(lane4&bit)?lo:hi;const unsigned other=__shfl_xor_sync(0xffffffffu,send,bit);if(lane4&bit)lo=other;else hi=other;};
            swap(v0,v1,1);swap(v2,v3,1);swap(v0,v2,2);swap(v1,v3,2);
            const int64_t at=int64_t(start)*Width+int64_t(feature+col+lane4)*padded+(base-start)+token;
            *reinterpret_cast<uint4*>(peer_pointer<uint8_t>(p,4)+at)=make_uint4(v0,v1,v2,v3);
        }
        __syncthreads();
    }
}
} // namespace deep_gemm::mega_moe_training

#endif
namespace deep_gemm::mega_moe_training {
template<int Op,int Width,int Threads>
__device__ __forceinline__ void fp8_gradient(const Params& p) {
    if constexpr(Op==0)gradient_raw_owner<Width,Threads>(p);
    else if constexpr(Op==1)fp8_gradient_weighted<1,Width,Threads>(p);
    else if constexpr(Op==2)gradient_gather_high<Width,Threads>(p);
    else if constexpr(Op==3)gradient_dw_left<Width,Threads>(p);
    else if constexpr(Op==4)gradient_gather_bf16_reference<Width,Threads>(p);
    else if constexpr(Op==5)gradient_gather_high_tiled<Width,Threads>(p);
    else if constexpr(Op==6)gradient_producer_started(p);
    else if constexpr(Op==7)gradient_shared_owner<Width,Threads>(p);
    else if constexpr(Op==8)gradient_score_folded_hidden<Width,Threads>(p);
    else if constexpr(Op==9)gradient_gather_bf16_reference<Width,Threads,true>(p);
    else if constexpr(Op==10)gradient_owner_score_parallel<Width,Threads>(p);
    else if constexpr(Op==11)gradient_gather_high_tiled<Width,Threads,true,false>(p);
    else if constexpr(Op==12)gradient_score_folded_hidden<Width,Threads,true>(p);
    else if constexpr(Op==13)gradient_gather_high_tiled<Width,Threads,true,true>(p);
    else if constexpr(Op==14)gradient_transpose_shared_left<Width,Threads>(p);
    else if constexpr(Op==15)gradient_expand_shared_scales<Width,Threads>(p);
    else if constexpr(Op==16)gradient_token_owner<Width,Threads>(p);
    else if constexpr(Op==17)gradient_gather_high_tiled<Width,Threads,true,false,true>(p);
    else if constexpr(Op==18)gradient_owner_score_parallel<Width,Threads,1>(p);
    else if constexpr(Op==19)gradient_owner_score_parallel<Width,Threads,2>(p);
    else if constexpr(Op==20)gradient_gather_high_tiled<Width,Threads,true,false,true,1>(p);
    else if constexpr(Op==21)gradient_gather_high_tiled<Width,Threads,true,false,false,1>(p);
    else if constexpr(Op==22)gradient_gather_high_tiled<Width,Threads,true,false,true,4>(p);
    else if constexpr(Op==23)gradient_gather_high_tiled<Width,Threads,true,false,false,4>(p);
    else if constexpr(Op==24)gradient_gather_high_tiled<Width,Threads,true,false,true,8>(p);
    else if constexpr(Op==25)gradient_gather_high_tiled<Width,Threads,true,false,false,8>(p);
    else if constexpr(Op==26)gradient_gather_high_tiled<Width,Threads,true,false,true,16>(p);
    else if constexpr(Op==27)gradient_gather_high_tiled<Width,Threads,true,false,false,16>(p);
    else if constexpr(Op==28)gradient_gather_high_tiled<Width,Threads,true,false,true,32>(p);
    else if constexpr(Op==29)gradient_gather_high_tiled<Width,Threads,true,false,false,32>(p);
    else if constexpr(Op==30)gradient_gather_high_bulk<Width,Threads,true>(p);
    else if constexpr(Op==32)gradient_gather_high_bulk<Width,Threads,true,4,true>(p);
    else if constexpr(Op==33)gradient_gather_high_bulk<Width,Threads,false,4,true>(p);
    else if constexpr(Op==35)gradient_gather_high_bulk<Width,Threads,true,8,false>(p);
    else if constexpr(Op==36)gradient_gather_high_bulk<Width,Threads,true,16,false>(p);
    else if constexpr(Op==37)gradient_gather_high_bulk<Width,Threads,true,8,true>(p);
    else if constexpr(Op==38)gradient_gather_high_bulk<Width,Threads,true,16,true>(p);
    else gradient_gather_high_bulk<Width,Threads,false>(p);
}
} // namespace deep_gemm::mega_moe_training


#include <deep_gemm/impls/mega_moe_training/fp8_inline_source.cuh>
namespace deep_gemm::mega_moe_training {
template<int Width,int BM>
__device__ __forceinline__ void forward_fp8_tile_source(const Params& p,char* arena) {
    __shared__ int control[2];
    const auto* records=reinterpret_cast<const uint4*>(p.ptr[3]);
    const int count=reinterpret_cast<const int32_t*>(records)[0];
    MegaMoEFp8InlineSource source{};
    source.value_peers=peer_pointer<uint64_t>(p,0);
    source.aux_peers=peer_pointer<uint64_t>(p,1);
    source.permutation=peer_pointer<int32_t>(p,2);
    source.ready=peer_pointer<uint32_t>(p,6);source.status=peer_pointer<uint32_t>(p,7);
    source.tokens=p.dim[2];source.topk=p.dim[3];source.num_ranks=p.dim[4];
    source.local_rank=p.dim[8];source.signal_capacity=p.dim[9]-1;
    if(count<0 || count>source.signal_capacity) {
        if(threadIdx.x==0)atomicOr(source.status,2u);return;
    }
    for(int ordinal=blockIdx.x;ordinal<count;ordinal+=gridDim.x) {
        const uint4 record=records[1+ordinal];
        const uint32_t local=(uint32_t(ordinal)-record.z)*BM;
        const uint32_t valid=min(uint32_t(BM),record.y-local);
        if(record.x>=uint32_t(p.dim[10]) || record.z>uint32_t(ordinal) || local>=record.y ||
                uint64_t(record.w)+local+valid>uint32_t(p.dim[0]) || record.y%128 || record.w%128) {
            if(threadIdx.x==0)atomicOr(source.status,2u);
            return;
        }
        if(!inline_fp8_source_tile<Width,BM,false,true>(source,ordinal,record.w+local,valid,p.dim[0],
                peer_pointer<uint8_t>(p,4),peer_pointer<float>(p,5),arena,control))return;
    }
}
}
