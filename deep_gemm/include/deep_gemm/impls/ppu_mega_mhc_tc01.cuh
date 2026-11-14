#pragma once

#include <type_traits>

#include <deep_gemm/impls/ppu_mega_mhc.cuh>
#include <cute/arch/mma_ppu0010.hpp>
#include <cute/arch/copy_ppu0010_aiu.hpp>
#include <cutlass/arch/memory_ppu.h>
#include <cutlass/numeric_conversion.h>

namespace deep_gemm::mhc {

template<int H, int S, bool RoundB, int Chunk>
__device__ __forceinline__ void ppu_mega_mhc_tc01_prenorm_device(Args a) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    static_assert(H==5120 && (Chunk==512 || Chunk==1024));
    constexpr int V=Chunk/128;
    using Pack=std::conditional_t<Chunk==512,uint2,uint4>;
    __shared__ float shared_coeff[24], sums[4][2], inv;
    __shared__ __align__(16) __ppu_bfloat16 input[2][5][Chunk], u_buffer[H];
    const int tid=threadIdx.x, t=blockIdx.x, lane=tid%32, warp=tid/32;
    float local_coeff[24];
    if constexpr(Chunk==512) {
        #pragma unroll
        for(int i=0;i<24;++i) local_coeff[i]=i<4 ? a.pre[t*4+i] : i<8 ? a.post[t*4+i-4] : a.comb[t*16+i-8];
    } else {
        if(tid<24) shared_coeff[tid]=tid<4 ? a.pre[t*4+tid] : tid<8 ? a.post[t*4+tid-4] : a.comb[t*16+tid-8];
        __syncthreads();
    }
    const auto coeff=[&](int i) {if constexpr(Chunk==512) return local_coeff[i];else return shared_coeff[i];};
    const auto copy=[&](void* dst,const void* src) {
        if constexpr(Chunk==512) cute::PPU_CP_ASYNC_CACHEGLOBAL<Pack>::copy(*reinterpret_cast<const Pack*>(src),*reinterpret_cast<Pack*>(dst));
        else cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(dst,src,true);
    };
    const auto unpack=[](uint32_t bits) {
        return make_float2(__uint_as_float(__byte_perm(bits,0,0x1054)),
                           __uint_as_float(__byte_perm(bits,0,0x3254)));
    };
    const auto load=[&](int base,int stage) {
        #pragma unroll
        for(int j=0;j<4;++j)
            copy(&input[stage][j][tid*V],a.residual+((int64_t)t*4+j)*H+base+tid*V);
        copy(&input[stage][4][tid*V],a.x+(int64_t)t*H+base+tid*V);
        cute::cp_async_fence();
    };
    float sq=0,usq=0;
    load(0,0);
    uint4 norm_weight[H/(128*8)];
    #pragma unroll
    for(int i=0;i<H/(128*8);++i)
        norm_weight[i]=*reinterpret_cast<const uint4*>(a.weight+tid*8+i*128*8);
    #pragma unroll 1
    for(int base=0;base<H;base+=Chunk) {
        const int stage=(base/Chunk)%2;
        cute::cp_async_wait<0>();__syncthreads();
        if(base+Chunk<H) load(base+Chunk,stage^1);
        Pack packed[5];
        #pragma unroll
        for(int j=0;j<5;++j) packed[j]=*reinterpret_cast<const Pack*>(&input[stage][j][tid*V]);
        float u[V]={};
        #pragma unroll
        for(int route=0;route<4;++route) {
            Pack output;
            #pragma unroll
            for(int z=0;z<V;z+=2) {
                const float2 x=unpack(reinterpret_cast<const uint32_t*>(&packed[4])[z/2]);
                float v0=x.x*coeff(4+route),v1=x.y*coeff(4+route);
                #pragma unroll
                for(int j=0;j<4;++j) {
                    const float2 old=unpack(reinterpret_cast<const uint32_t*>(&packed[j])[z/2]);
                    v0=fmaf(old.x,coeff(8+j*4+route),v0);
                    v1=fmaf(old.y,coeff(8+j*4+route),v1);
                }
                const auto rounded=__floats2bfloat162_rn(v0,v1);
                reinterpret_cast<__ppu_bfloat162*>(&output)[z/2]=rounded;
                const float2 v=unpack(*reinterpret_cast<const uint32_t*>(&rounded));
                sq=fmaf(v.x,v.x,sq);sq=fmaf(v.y,v.y,sq);
                u[z]=fmaf(v.x,coeff(route),u[z]);
                u[z+1]=fmaf(v.y,coeff(route),u[z+1]);
            }
            *reinterpret_cast<Pack*>(a.new_residual+((int64_t)t*4+route)*H+base+tid*V)=output;
        }
        Pack output;
        #pragma unroll
        for(int z=0;z<V;z+=2) {
            const auto rounded=__floats2bfloat162_rn(u[z],u[z+1]);
            reinterpret_cast<__ppu_bfloat162*>(&output)[z/2]=rounded;
            const float2 v=unpack(*reinterpret_cast<const uint32_t*>(&rounded));
            usq=fmaf(v.x,v.x,usq);usq=fmaf(v.y,v.y,usq);
        }
        *reinterpret_cast<Pack*>(u_buffer+base+tid*V)=output;
    }
    #pragma unroll
    for(int delta=1;delta<32;delta*=2) {
        sq+=__shfl_xor_sync(0xffffffff,sq,delta);
        usq+=__shfl_xor_sync(0xffffffff,usq,delta);
    }
    if(lane==0) {sums[warp][0]=sq;sums[warp][1]=usq;}
    __syncthreads();
    float* p=a.partials+(int64_t)(t/64)*S*64*P+(t%64)*P;
    if(tid==0) {
        sq=0;usq=0;
        #pragma unroll
        for(int i=0;i<4;++i) {sq+=sums[i][0];usq+=sums[i][1];}
        p[24]=sq;
        inv=rsqrtf(usq/H+a.norm_eps);
    }
    __syncthreads();
    constexpr int E=8;
    using OutPack=uint4;
    #pragma unroll (H/(128*8))
    for(int i=0;i<H/(128*8);++i) {
        const int h=tid*8+i*128*8;
        OutPack up=*reinterpret_cast<const OutPack*>(u_buffer+h);
        const OutPack wp=norm_weight[i];
        OutPack output;
        #pragma unroll
        for(int z=0;z<E/2;++z) {
            const float2 u=unpack(reinterpret_cast<const uint32_t*>(&up)[z]);
            const float2 w=unpack(reinterpret_cast<const uint32_t*>(&wp)[z]);
            reinterpret_cast<__ppu_bfloat162*>(&output)[z]=__floats2bfloat162_rn(
                (u.x*inv)*(w.x*a.norm_scale),(u.y*inv)*(w.y*a.norm_scale));
        }
        *reinterpret_cast<OutPack*>(a.y+(int64_t)t*H+h)=output;
    }
    if constexpr(RoundB) {
        auto* rounded=reinterpret_cast<uint32_t*>(a.partials+(int64_t)((a.tokens+63)/64)*S*64*P);
        const int64_t stride=int64_t(gridDim.x)*128*4;
        for(int64_t i=(int64_t(blockIdx.x)*128+tid)*4;i<24*4*H;i+=stride) {
            const float4 values=*reinterpret_cast<const float4*>(a.fn+i);
            uint4 bits;
            #pragma unroll
            for(int z=0;z<4;++z) reinterpret_cast<uint32_t*>(&bits)[z]=cutlass::NumericConverter<cutlass::tfloat32_t,float>::convert(reinterpret_cast<const float*>(&values)[z]).raw();
            *reinterpret_cast<uint4*>(rounded+i)=bits;
        }
    }
#endif
}

template<int H, int Iters, int W, bool Unroll>
__device__ __forceinline__ void ppu_mega_mhc_tc01_store_coeff(Args a, int t, float v, float v1, float sq) {
    const int lane=threadIdx.x%W;
    const uint32_t mask=W==32?0xffffffffu:0xffffu<<((threadIdx.x%32)/16*16);
    if(lane>=24) v=0.f;
    if(lane<24) {
        v=v*rsqrtf(sq/(4*H)+a.hc_eps);
        v=v*a.scales[lane<4?0:lane<8?1:2]+a.bases[lane];
    }
    if constexpr(W==16) if(lane<8) {
        v1=v1*rsqrtf(sq/(4*H)+a.hc_eps);
        v1=v1*a.scales[2]+a.bases[lane+16];
    }
    if(lane<4) a.new_pre[t*4+lane]=__ppu_sgmdf(v)+a.pre_eps;
    if(lane>=4 && lane<8) a.new_post[t*4+lane-4]=__ppu_sgmdf(v)*a.post_scale;
    float c;
    if constexpr(W==16) {
        float c0=__shfl_sync(mask,v,(lane%8)+8,W);
        float c1=__shfl_sync(mask,v1,lane%8,W);
        c=lane<8?c0:c1;
    } else c=__shfl_sync(mask,v,(lane%16)+8,W);
    float mx=c;
    mx=fmaxf(mx,__shfl_xor_sync(mask,mx,1,W));
    mx=fmaxf(mx,__shfl_xor_sync(mask,mx,2,W));
    c=expf(c-mx);
    float sum=c+__shfl_xor_sync(mask,c,1,W);
    sum+=__shfl_xor_sync(mask,sum,2,W);
    c=c*__ppu_rcpf(sum)+a.sink_eps;
    #pragma unroll (Unroll?Iters:1)
    for(int iter=0;iter<Iters;++iter) {
        if(iter) {
            sum=c+__shfl_xor_sync(mask,c,1,W);
            sum+=__shfl_xor_sync(mask,sum,2,W);
            c=c*__ppu_rcpf(sum+a.sink_eps);
        }
        sum=c+__shfl_xor_sync(mask,c,4,W);
        sum+=__shfl_xor_sync(mask,sum,8,W);
        c=c*__ppu_rcpf(sum+a.sink_eps);
    }
    if(lane<16) a.new_comb[t*16+lane]=c;
}

template<int H,int S,int Iters>
__device__ __forceinline__ void ppu_mega_mhc_tc01_coeff_device(Args a) {
    constexpr int W=S<32?16:32;
    const int t=blockIdx.x*(128/W)+threadIdx.x/W,lane=threadIdx.x%W;
    if(t>=a.tokens) return;
    const volatile float* p=a.partials+(int64_t)(t/64)*S*64*P+(t%64)*P;
    float v=0,v1=0;
    #pragma unroll
    for(int split=0;split<S;++split) {
        if(lane<24) v+=p[split*64*P+lane];
        if constexpr(W==16) v1+=p[split*64*P+16+lane%8];
    }
    // Prenorm produces the complete residual square sum in split zero.
    float sq=p[24];
    ppu_mega_mhc_tc01_store_coeff<H,Iters,W,(S<32)>(a,t,v,v1,sq);
}

template<int H, int S, bool RoundB>
__device__ __forceinline__ void ppu_mega_mhc_tc01_device(Args a) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    constexpr int BM=64, BK=64;
    static_assert(H==5120 && H%BK==0);
    struct Storage {
        __ppu_bfloat16 input[2][BM*BK];
        float b[2][2][32*32];
    };
    __shared__ __align__(128) Storage sm;
    const int tid=threadIdx.x, lane=tid%32, warp=tid/32;
    const int mb=blockIdx.x, split=blockIdx.y;
    const int begin=(H/BK)*split/S, end=(H/BK)*(split+1)/S;
    float accum[16]={};
    const float* rounded=RoundB ? a.partials+(int64_t)((a.tokens+63)/64)*S*64*P : a.fn;
    cute::AiuDesc ad{},bd{};
    ad.dim_h=a.tokens;ad.dim_w=4*H;ad.cube_h=64;ad.cube_w=64;
    bd.dim_h=24;bd.dim_w=4*H;bd.cube_h=32;bd.cube_w=32;
    const auto load = [&](int step, int stage) {
        const int kb=begin+step/4,route=step%4;
        if(warp==0) {
            cute::PPU0010_AIU_LOAD<cute::C<64*64*16>,__ppu_bfloat16,false,true>::copy(
                reinterpret_cast<void*>(__cvta_generic_to_shared(sm.input[stage])),a.new_residual,ad,route*H+kb*BK,mb*64);
            #pragma unroll
            for(int part=0;part<2;++part)
                cute::PPU0010_AIU_LOAD<cute::C<32*32*32>,float,false,true>::copy(
                    reinterpret_cast<void*>(__cvta_generic_to_shared(sm.b[stage][part])),rounded,bd,route*H+kb*BK+part*32,0);
            cute::cp_async_fence();
        }
    };
    load(0,0);
    for(int step=0;step<(end-begin)*4;++step) {
        const int stage=step%2;
        if(warp==0) cute::cp_async_wait<0>();
        __syncthreads();
        if(step+1<(end-begin)*4) load(step+1,stage^1);
        #pragma unroll
        for(int kk=0;kk<BK;kk+=16) {
            uint32_t av[2][4],bv[2][2][4],packed[4];
            cute::PPU0010_TSM_LD_SWZL<__ppu_bfloat16,64,64,false,false>::copy(
                packed,reinterpret_cast<void*>(__cvta_generic_to_shared(sm.input[stage])),kk,warp*16);
            #pragma unroll
            for(int ki=0;ki<2;++ki) {
                const int src=(lane%4)/2,perm=(lane%2) ? 0x3254 : 0x1054;
                av[ki][0]=__byte_perm(__shfl_sync(0xffffffff,packed[ki],src,4),0,perm);
                av[ki][1]=__byte_perm(__shfl_sync(0xffffffff,packed[ki],src+2,4),0,perm);
                av[ki][2]=__byte_perm(__shfl_sync(0xffffffff,packed[ki+2],src,4),0,perm);
                av[ki][3]=__byte_perm(__shfl_sync(0xffffffff,packed[ki+2],src+2,4),0,perm);
                #pragma unroll
                for(int nb=0;nb<2;++nb) {
                    float values[4];
                    cute::PPU0010_TSM_LD_SWZL<float,32,32,false,false>::copy(
                        values,reinterpret_cast<void*>(__cvta_generic_to_shared(sm.b[stage][kk/32])),kk%32+ki*8,nb*16);
                    #pragma unroll
                    for(int i=0;i<4;++i) {
                        if constexpr(RoundB) bv[ki][nb][i]=__float_as_uint(values[i]);
                        else bv[ki][nb][i]=cutlass::NumericConverter<cutlass::tfloat32_t,float>::convert(values[i]).raw();
                    }
                }
            }
            #pragma unroll
            for(int ki=0;ki<2;++ki) {
                #pragma unroll
                for(int nb=0;nb<2;++nb) {
                    float* d=accum+nb*8;
                    cute::PPU0010_16x16x8_F32TF32TF32F32_TN::fma(
                        d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],
                        av[ki][0],av[ki][1],av[ki][2],av[ki][3],bv[ki][nb][0],bv[ki][nb][1],bv[ki][nb][2],bv[ki][nb][3],
                        d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7]);
                }
            }
        }
    }

    float* partial=a.partials+((int64_t)mb*S+split)*BM*P;
    #pragma unroll
    for(int nb=0;nb<2;++nb) {
        #pragma unroll
        for(int i=0;i<8;++i) {
            const int r=warp*16+lane/4+(i/4)*8;
            const int n=nb*16+lane%4+(i%4)*4;
            if(n<24) partial[r*P+n]=accum[nb*8+i];
        }
    }
#endif
}
} // namespace deep_gemm::mhc
