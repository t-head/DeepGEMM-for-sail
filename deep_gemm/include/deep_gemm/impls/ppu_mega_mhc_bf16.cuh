#pragma once
#include <type_traits>
#include <hggc_runtime.h>
#include <hggc_bf16.h>
#include <cstdint>
namespace deep_gemm::mhc {
struct Args {
    const __ppu_bfloat16 *x, *residual, *weight;
    const float *pre, *post, *comb, *fn, *scales, *bases;
    __ppu_bfloat16 *new_residual, *y;
    float *new_pre, *new_post, *new_comb, *partials;
    int *counters;
    int tokens;
    float hc_eps, pre_eps, post_scale, sink_eps, norm_eps, norm_scale;
};
}

#ifndef DG_MHC_ARGS_ONLY
#include <cutlass/cutlass.h>
#include <cute/arch/copy_ppu.hpp>
#include <cute/arch/copy_ppu0015_aiu.hpp>
#include <cute/arch/mma_ppu0015.hpp>
#include <cute/arch/util.hpp>

namespace deep_gemm::mhc::primitives {

CUTLASS_DEVICE void cp_async_16(void* dst, const void* src, const bool pred) {
#if defined(__HGGC_ARCH__)
    const uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst);
    const uint32_t src_size = pred ? 16 : 0;
    asm volatile("ppu.cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
                 "r"(dst_addr), "l"(src), "r"(src_size));
#endif
}

#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
namespace tc02 {

using AiuLoadA = cute::PPU0015_AIU_LOAD<
    cute::C<64 * 64 * 16>, __ppu_bfloat16, false, 64, 64, true>;
using AiuLoadB = cute::PPU0015_AIU_LOAD<
    cute::C<32 * 32 * 32>, float, false, 32, 32, true>;

// SDK coordinates: (column, row, cube, stage).
using BTileLoad = cute::PPU0015_TSM_LD_SWZL<float, 32, 32, true, false, 2>;
static_assert(BTileLoad::lbo == 64 && BTileLoad::sbo == 1 && BTileLoad::swzl_mode == 0);

} // namespace tc02
#endif

} // namespace deep_gemm::mhc::primitives

namespace deep_gemm::mhc {
// Partials: [tile, split, 64 tokens, 24 projections + residual square sum + pre-mix square sum].
// Round residual and pre-mix results to BF16 before reuse.
// Preserve route-then-element, TF32 and Split-K accumulation order.
constexpr int M = 64, K = 64, P = 26;

template<int H, int S, int Iters, bool FullRows=false>
__device__ void finish(Args a, int mb, __ppu_bfloat16* norm_tile, int first_row=0, int row_limit=M) {
    constexpr bool InterleaveNorm = Iters==20;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    // Padded rows must still participate in CTA barriers.
    int remaining=a.tokens-mb*M;
    int rows=FullRows ? M : (remaining<M ? ((remaining+3)&~3) : M);
    rows=min(rows,row_limit);
    for (int row=first_row+warp; row<rows; row+=4) {
        int t=mb*M+row;
        bool valid=FullRows || t<a.tokens;
        constexpr bool EarlyPartials = S==5 || S==12;
        float early[EarlyPartials ? S : 1];
        if constexpr(EarlyPartials) {
            const volatile float* source=a.partials+(int64_t)mb*S*M*P+row*P;
            #pragma unroll
            for(int split=0;split<S;++split)
                if(lane<26) early[split]=source[split*M*P+lane];
        }
        {
            for(int pos=threadIdx.x*8;pos<4*H;pos+=128*8) {
                int token=mb*M+row-warp+pos/H;
                primitives::cp_async_16(norm_tile+pos,a.y+(int64_t)token*H+pos%H,FullRows || token<a.tokens);
            }
            if constexpr(S==5 || S==6 || S==12) {
                if(mb*16+first_row/4==blockIdx.x) {
                    for(int pos=threadIdx.x*8;pos<H;pos+=128*8)
                        primitives::cp_async_16(norm_tile+4*H+pos,a.weight+pos,true);
                }
            } else {
                for(int pos=threadIdx.x*8;pos<H;pos+=128*8)
                    primitives::cp_async_16(norm_tile+4*H+pos,a.weight+pos,true);
            }
            cute::cp_async_fence();
        }
        float v=0, sq=0, usq=0;
        const volatile float* p=a.partials+(int64_t)mb*S*M*P+row*P;
        if constexpr(EarlyPartials) {
            #pragma unroll
            for(int split=0;split<S;++split)
                if(lane<26) v+=early[split];
        } else {
            for(int split=0;split<S;++split)
                if(lane<26) v+=p[split*M*P+lane];
        }
        sq=__shfl_sync(0xffffffff,v,24);
        usq=__shfl_sync(0xffffffff,v,25);
        if(lane>=24) v=0.f;
        if(lane<24) {
            v=v*rsqrtf(sq/(4*H)+a.hc_eps);
            v=v*a.scales[lane<4?0:lane<8?1:2]+a.bases[lane];
        }
        float pre=__ppu_sgmdf(v)+a.pre_eps;
        if(lane<4 && valid) a.new_pre[t*4+lane]=pre;
        if(lane>=4 && lane<8 && valid) a.new_post[t*4+lane-4]=__ppu_sgmdf(v)*a.post_scale;
        float c=__shfl_sync(0xffffffff,v,(lane%16)+8);
        float mx=c;
        mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1));
        mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));
        c=expf(c-mx);
        float sum=c+__shfl_xor_sync(0xffffffff,c,1);
        sum+=__shfl_xor_sync(0xffffffff,sum,2);
        c=c*__ppu_rcpf(sum)+a.sink_eps;
        float interleaved_inv=0.f;
        if constexpr(InterleaveNorm) {
            cute::cp_async_wait<0>(); __syncthreads();
            interleaved_inv=rsqrtf(usq/H+a.norm_eps);
        }
        for(int iter=0;iter<Iters;++iter) {
            if(iter) {
                sum=c+__shfl_xor_sync(0xffffffff,c,1);
                sum+=__shfl_xor_sync(0xffffffff,sum,2);
                c=c*__ppu_rcpf(sum+a.sink_eps);
            }
            if constexpr(InterleaveNorm) {
                int h=iter*256+lane*8;
                uint4 input=*reinterpret_cast<const uint4*>(norm_tile+warp*H+h);
                uint4 weight=*reinterpret_cast<const uint4*>(norm_tile+4*H+h),output;
                const auto* uv=reinterpret_cast<const __ppu_bfloat16*>(&input);
                const auto* wv=reinterpret_cast<const __ppu_bfloat16*>(&weight);
                #pragma unroll
                for(int z=0;z<4;z+=2) {
                    float u0=__bfloat162float(uv[z]),u1=__bfloat162float(uv[z+1]);
                    float w0=__bfloat162float(wv[z])*a.norm_scale,w1=__bfloat162float(wv[z+1])*a.norm_scale;
                    reinterpret_cast<__ppu_bfloat162*>(&output)[z/2]=__floats2bfloat162_rn((u0*interleaved_inv)*w0,(u1*interleaved_inv)*w1);
                }
                sum=c+__shfl_xor_sync(0xffffffff,c,4);
                sum+=__shfl_xor_sync(0xffffffff,sum,8);
                c=c*__ppu_rcpf(sum+a.sink_eps);
                #pragma unroll
                for(int z=4;z<8;z+=2) {
                    float u0=__bfloat162float(uv[z]),u1=__bfloat162float(uv[z+1]);
                    float w0=__bfloat162float(wv[z])*a.norm_scale,w1=__bfloat162float(wv[z+1])*a.norm_scale;
                    reinterpret_cast<__ppu_bfloat162*>(&output)[z/2]=__floats2bfloat162_rn((u0*interleaved_inv)*w0,(u1*interleaved_inv)*w1);
                }
                if(valid) *reinterpret_cast<uint4*>(a.y+(int64_t)t*H+h)=output;
            } else {
                sum=c+__shfl_xor_sync(0xffffffff,c,4);
                sum+=__shfl_xor_sync(0xffffffff,sum,8);
                c=c*__ppu_rcpf(sum+a.sink_eps);
            }
        }
        if(lane<16 && valid) a.new_comb[t*16+lane]=c;
        if constexpr(!InterleaveNorm) {
            cute::cp_async_wait<0>(); __syncthreads();
            float inv=rsqrtf(usq/H+a.norm_eps);
            for(int base=0;base<H;base+=256) {
                int h=base+lane*8;
                const __ppu_bfloat16* input_ptr=norm_tile+warp*H+h;
                uint4 input=*reinterpret_cast<const uint4*>(input_ptr);
                const __ppu_bfloat16* weight_ptr=norm_tile+4*H+h;
                uint4 weight=*reinterpret_cast<const uint4*>(weight_ptr),output;
                const auto* uv=reinterpret_cast<const __ppu_bfloat16*>(&input);
                const auto* wv=reinterpret_cast<const __ppu_bfloat16*>(&weight);
                #pragma unroll
                for(int z=0;z<8;z+=2) {
                    float u0=__bfloat162float(uv[z]),u1=__bfloat162float(uv[z+1]);
                    float w0=__bfloat162float(wv[z])*a.norm_scale,w1=__bfloat162float(wv[z+1])*a.norm_scale;
                    reinterpret_cast<__ppu_bfloat162*>(&output)[z/2]=__floats2bfloat162_rn((u0*inv)*w0,(u1*inv)*w1);
                }
                if(valid) *reinterpret_cast<uint4*>(a.y+(int64_t)t*H+h)=output;
            }
        }
        __syncthreads();
    }
}

}

#include <cutlass/arch/memory.h>
#include <cutlass/barrier.h>

namespace deep_gemm::mhc {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
template<int BCubeStride=1024>
__device__ __forceinline__ void manual_lifetime_addresses(
        void* smem, int ki, int* addr) {
    #pragma unroll
    for (int k=0;k<2;++k) {
        #pragma unroll
        for (int g=0;g<2;++g) {
            float* base = reinterpret_cast<float*>(smem)
                + BCubeStride*((ki+k)/4) + g*16*32 + ((ki+k)%4)*8;
            addr[k*2+g] = reinterpret_cast<uintptr_t>(base)/16;
        }
    }
}

// TSM operand waits rely on compiler scoreboarding.
__device__ __forceinline__ void manual_tf32_pair(float* d, const uint32_t* a,
                                                 void* smem, int ki) {
    int addr[4];
    manual_lifetime_addresses(smem, ki, addr);
    asm volatile(
        "{\n"
        ".reg .b32 b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15;\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {b0,b1,b2,b3}, [%24], %28, %29, %30;\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {b4,b5,b6,b7}, [%25], %28, %29, %30;\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {b8,b9,b10,b11}, [%26], %28, %29, %30;\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {b12,b13,b14,b15}, [%27], %28, %29, %30;\n"
        "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%16,%17,%18,%19}, {b0,b1,b2,b3}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 {%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, {b4,b5,b6,b7}, {%8,%9,%10,%11,%12,%13,%14,%15};\n"
        "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%20,%21,%22,%23}, {b8,b9,b10,b11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 {%8,%9,%10,%11,%12,%13,%14,%15}, {%20,%21,%22,%23}, {b12,b13,b14,b15}, {%8,%9,%10,%11,%12,%13,%14,%15};\n"
        "}\n"
        : "+&r"(d[0]), "+&r"(d[1]), "+&r"(d[2]), "+&r"(d[3]), "+&r"(d[4]), "+&r"(d[5]), "+&r"(d[6]), "+&r"(d[7]), "+&r"(d[8]), "+&r"(d[9]), "+&r"(d[10]), "+&r"(d[11]), "+&r"(d[12]), "+&r"(d[13]), "+&r"(d[14]), "+&r"(d[15])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]), "l"(addr[0]), "l"(addr[1]), "l"(addr[2]), "l"(addr[3]), "r"(64), "r"(1), "r"(0)
        : "memory");
}
#endif


#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
__device__ __forceinline__ void manual_lifetime_load(
        uint32_t* next, void* smem) {
    int addr[4];
    manual_lifetime_addresses(smem, 0, addr);
    asm volatile(
        "{\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%0,%1,%2,%3}, [%16], %20, %21, %22;\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%4,%5,%6,%7}, [%17], %20, %21, %22;\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%8,%9,%10,%11}, [%18], %20, %21, %22;\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%12,%13,%14,%15}, [%19], %20, %21, %22;\n"
        "}\n"
        : "=&r"(next[0]), "=&r"(next[1]), "=&r"(next[2]), "=&r"(next[3]), "=&r"(next[4]), "=&r"(next[5]), "=&r"(next[6]), "=&r"(next[7]), "=&r"(next[8]), "=&r"(next[9]), "=&r"(next[10]), "=&r"(next[11]), "=&r"(next[12]), "=&r"(next[13]), "=&r"(next[14]), "=&r"(next[15])
        : "l"(addr[0]), "l"(addr[1]), "l"(addr[2]), "l"(addr[3]), "r"(64), "r"(1), "r"(0)
        : "memory");

}
__device__ __forceinline__ void manual_lifetime_advance(
        float* d, const uint32_t* a, const uint32_t* current,
        uint32_t* next, void* smem, int next_ki) {
    int addr[4];
    manual_lifetime_addresses(smem, next_ki, addr);
    asm volatile(
        "{\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%16,%17,%18,%19}, [%56], %60, %61, %62;\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%20,%21,%22,%23}, [%57], %60, %61, %62;\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%24,%25,%26,%27}, [%58], %60, %61, %62;\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%28,%29,%30,%31}, [%59], %60, %61, %62;\n"
        "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%32,%33,%34,%35}, {%40,%41,%42,%43}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 {%8,%9,%10,%11,%12,%13,%14,%15}, {%32,%33,%34,%35}, {%44,%45,%46,%47}, {%8,%9,%10,%11,%12,%13,%14,%15};\n"
        "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%36,%37,%38,%39}, {%48,%49,%50,%51}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 {%8,%9,%10,%11,%12,%13,%14,%15}, {%36,%37,%38,%39}, {%52,%53,%54,%55}, {%8,%9,%10,%11,%12,%13,%14,%15};\n"
        "}\n"
        : "+&r"(d[0]), "+&r"(d[1]), "+&r"(d[2]), "+&r"(d[3]), "+&r"(d[4]), "+&r"(d[5]), "+&r"(d[6]), "+&r"(d[7]), "+&r"(d[8]), "+&r"(d[9]), "+&r"(d[10]), "+&r"(d[11]), "+&r"(d[12]), "+&r"(d[13]), "+&r"(d[14]), "+&r"(d[15]), "=&r"(next[0]), "=&r"(next[1]), "=&r"(next[2]), "=&r"(next[3]), "=&r"(next[4]), "=&r"(next[5]), "=&r"(next[6]), "=&r"(next[7]), "=&r"(next[8]), "=&r"(next[9]), "=&r"(next[10]), "=&r"(next[11]), "=&r"(next[12]), "=&r"(next[13]), "=&r"(next[14]), "=&r"(next[15])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]), "r"(current[0]), "r"(current[1]), "r"(current[2]), "r"(current[3]), "r"(current[4]), "r"(current[5]), "r"(current[6]), "r"(current[7]), "r"(current[8]), "r"(current[9]), "r"(current[10]), "r"(current[11]), "r"(current[12]), "r"(current[13]), "r"(current[14]), "r"(current[15]), "l"(addr[0]), "l"(addr[1]), "l"(addr[2]), "l"(addr[3]), "r"(64), "r"(1), "r"(0)
        : "memory");

}
__device__ __forceinline__ void manual_lifetime_drain(
        float* d, const uint32_t* a, const uint32_t* current) {
    asm volatile(
        "{\n"
        "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%16,%17,%18,%19}, {%24,%25,%26,%27}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 {%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, {%28,%29,%30,%31}, {%8,%9,%10,%11,%12,%13,%14,%15};\n"
        "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%20,%21,%22,%23}, {%32,%33,%34,%35}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 {%8,%9,%10,%11,%12,%13,%14,%15}, {%20,%21,%22,%23}, {%36,%37,%38,%39}, {%8,%9,%10,%11,%12,%13,%14,%15};\n"
        "}\n"
        : "+&r"(d[0]), "+&r"(d[1]), "+&r"(d[2]), "+&r"(d[3]), "+&r"(d[4]), "+&r"(d[5]), "+&r"(d[6]), "+&r"(d[7]), "+&r"(d[8]), "+&r"(d[9]), "+&r"(d[10]), "+&r"(d[11]), "+&r"(d[12]), "+&r"(d[13]), "+&r"(d[14]), "+&r"(d[15])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]), "r"(current[0]), "r"(current[1]), "r"(current[2]), "r"(current[3]), "r"(current[4]), "r"(current[5]), "r"(current[6]), "r"(current[7]), "r"(current[8]), "r"(current[9]), "r"(current[10]), "r"(current[11]), "r"(current[12]), "r"(current[13]), "r"(current[14]), "r"(current[15])
        : "memory");

}

// N24 packet: [K0 N16:4 words, K0 N8:2 words, K1 N16:4 words, K1 N8:2 words].
// N8 D/C require the SDK float register constraints.
template<int BCubeStride=1024>
__device__ __forceinline__ void manual_lifetime_load_n24(
        uint32_t* next, void* smem, int ki=0) {
    int addr[4];
    manual_lifetime_addresses<BCubeStride>(smem, ki, addr);
    asm volatile(
        "{\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%0,%1,%2,%3}, [%12], %16, %17, %18;\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x2.b16 {%4,%5}, [%13], %17, %17, %18;\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%6,%7,%8,%9}, [%14], %16, %17, %18;\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x2.b16 {%10,%11}, [%15], %17, %17, %18;\n"
        "}\n"
        : "=&r"(next[0]), "=&r"(next[1]), "=&r"(next[2]), "=&r"(next[3]), "=&r"(next[4]), "=&r"(next[5]), "=&r"(next[6]), "=&r"(next[7]), "=&r"(next[8]), "=&r"(next[9]), "=&r"(next[10]), "=&r"(next[11])
        : "l"(addr[0]), "l"(addr[1]), "l"(addr[2]), "l"(addr[3]), "r"(64), "r"(1), "r"(0)
        : "memory");
}
template<int BCubeStride=1024>
__device__ __forceinline__ void manual_lifetime_advance_n24(
        float* d, const uint32_t* a, const uint32_t* current,
        uint32_t* next, void* smem, int next_ki) {
    int addr[4];
    manual_lifetime_addresses<BCubeStride>(smem, next_ki, addr);
    asm volatile(
        "{\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%12,%13,%14,%15}, [%44], %48, %49, %50;\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x2.b16 {%16,%17}, [%45], %49, %49, %50;\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%18,%19,%20,%21}, [%46], %48, %49, %50;\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x2.b16 {%22,%23}, [%47], %49, %49, %50;\n"
        "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%24,%25,%26,%27}, {%32,%33,%34,%35}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        "ppu.tc02.mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%8,%9,%10,%11}, {%24,%25,%26,%27}, {%36,%37}, {%8,%9,%10,%11};\n"
        "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%28,%29,%30,%31}, {%38,%39,%40,%41}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        "ppu.tc02.mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%8,%9,%10,%11}, {%28,%29,%30,%31}, {%42,%43}, {%8,%9,%10,%11};\n"
        "}\n"
        : "+&r"(d[0]), "+&r"(d[1]), "+&r"(d[2]), "+&r"(d[3]), "+&r"(d[4]), "+&r"(d[5]), "+&r"(d[6]), "+&r"(d[7]), "+&f"(d[8]), "+&f"(d[9]), "+&f"(d[10]), "+&f"(d[11]), "=&r"(next[0]), "=&r"(next[1]), "=&r"(next[2]), "=&r"(next[3]), "=&r"(next[4]), "=&r"(next[5]), "=&r"(next[6]), "=&r"(next[7]), "=&r"(next[8]), "=&r"(next[9]), "=&r"(next[10]), "=&r"(next[11])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]), "r"(current[0]), "r"(current[1]), "r"(current[2]), "r"(current[3]), "r"(current[4]), "r"(current[5]), "r"(current[6]), "r"(current[7]), "r"(current[8]), "r"(current[9]), "r"(current[10]), "r"(current[11]), "l"(addr[0]), "l"(addr[1]), "l"(addr[2]), "l"(addr[3]), "r"(64), "r"(1), "r"(0)
        : "memory");
}
__device__ __forceinline__ void manual_lifetime_drain_n24(
        float* d, const uint32_t* a, const uint32_t* current) {
    asm volatile(
        "{\n"
        "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%12,%13,%14,%15}, {%20,%21,%22,%23}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        "ppu.tc02.mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%8,%9,%10,%11}, {%12,%13,%14,%15}, {%24,%25}, {%8,%9,%10,%11};\n"
        "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%16,%17,%18,%19}, {%26,%27,%28,%29}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        "ppu.tc02.mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%8,%9,%10,%11}, {%16,%17,%18,%19}, {%30,%31}, {%8,%9,%10,%11};\n"
        "}\n"
        : "+&r"(d[0]), "+&r"(d[1]), "+&r"(d[2]), "+&r"(d[3]), "+&r"(d[4]), "+&r"(d[5]), "+&r"(d[6]), "+&r"(d[7]), "+&f"(d[8]), "+&f"(d[9]), "+&f"(d[10]), "+&f"(d[11])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]), "r"(current[0]), "r"(current[1]), "r"(current[2]), "r"(current[3]), "r"(current[4]), "r"(current[5]), "r"(current[6]), "r"(current[7]), "r"(current[8]), "r"(current[9]), "r"(current[10]), "r"(current[11])
        : "memory");
}
#endif

#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
// One token per half-warp; Sinkhorn shuffles must not cross the boundary.
template<int S, bool LocalCompletion=false, int TileM=32, bool TailRows=false,
         bool TailTokens=false, bool VectorCoefficients=true>
__device__ __forceinline__ void single_ring_finish(Args a, void* scratch, uint32_t producer_task=0) {
    static_assert(S>=8 && S<=80 && S%2==0,"Unsupported split count");
    constexpr int H=5120, Iters=20;
    constexpr int Threads=TileM*2, Rows=TileM/8;
    const int tokens=a.tokens;
    struct __align__(1024) Storage {
        float weight[H];
        uint4 input[4][2][Threads];
    };
    static_assert(sizeof(Storage)==20480+128*Threads,"two BF16 tokens per warp, four ring slots");
    Storage& storage=*reinterpret_cast<Storage*>(scratch);
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32;
    auto* padding=reinterpret_cast<__ppu_bfloat16*>(a.partials+((tokens+63)/64)*S*64*P);
    auto* raw_weight=reinterpret_cast<__ppu_bfloat16*>(storage.weight);
    for(int pos=tid*8;pos<H;pos+=Threads*8)
        primitives::cp_async_16(raw_weight+pos,a.weight+pos,true);
    cute::cp_async_fence();cute::cp_async_wait<0>();__syncthreads();
    {
        uint4 packed_weight[H/(Threads*8)];
        #pragma unroll
        for(int pack=0;pack<H/(Threads*8);++pack)
            packed_weight[pack]=*reinterpret_cast<const uint4*>(raw_weight+(pack*Threads+tid)*8);
        __syncthreads();
        #pragma unroll
        for(int pack=0;pack<H/(Threads*8);++pack) {
            const auto* raw=reinterpret_cast<const __ppu_bfloat16*>(&packed_weight[pack]);
            float scaled[8];
            #pragma unroll
            for(int z=0;z<8;++z) scaled[z]=__bfloat162float(raw[z])*a.norm_scale;
            const int wi=(pack*Threads+tid)*8;
            float* dest=storage.weight+(wi^((wi>>3)&4));
            *reinterpret_cast<float4*>(dest)=*reinterpret_cast<float4*>(scaled);
            *reinterpret_cast<float4*>(storage.weight+((wi+4)^(((wi+4)>>3)&4)))=*reinterpret_cast<float4*>(scaled+4);
        }
    }
    __syncthreads();
    // Acquire producer stores before reading partials or U.
    if constexpr(LocalCompletion) {
        const int group=producer_task/S;
        cutlass::Barrier::wait_eq(a.counters,tid,group,S);
    } else cutlass::Barrier::wait_eq(a.counters,tid,0,((tokens+TileM-1)/TileM)*S);
    const int first_task=LocalCompletion ? (S==16 ? int(producer_task/2) :
        int(producer_task/S)*(TileM/Rows)+int(producer_task%S)/2) : int(blockIdx.x);
    const int end_task=LocalCompletion && S!=16 ?
        (TailRows ? min((tokens+Rows-1)/Rows,(int(producer_task/S)+1)*(TileM/Rows)) :
         (int(producer_task/S)+1)*(TileM/Rows)) :
        (LocalCompletion && !TailRows ? first_task+1 : (tokens+Rows-1)/Rows);
    for(int task=first_task;task<end_task;task+=LocalCompletion && S!=16 ? S/2 : int(gridDim.x)*(LocalCompletion ? S/2 : 1)) {
        const int t0=task*Rows+warp,t1=t0+Rows/2;
        const int mb=t0/64,row=t0%64;
        const volatile float* p=a.partials+(int64_t)mb*S*M*P+row*P;
        const int t=lane<16?t0:t1;
        const int h_lane=(lane%16)*8;
        auto* y_row=a.y+(int64_t)t*H;
        if constexpr(TailRows) if(t>=tokens)
            y_row=padding+(t-tokens)*5*H+4*H;
        float values[S][2];
        #pragma unroll
        for(int split=0;split<S;++split) if(lane<26) {
            values[split][0]=p[split*M*P+lane];
            values[split][1]=p[split*M*P+(Rows/2)*P+lane];
        }
        #pragma unroll
        for(int pack=0;pack<4;++pack) {
            #pragma unroll
            for(int part=0;part<2;++part)
                primitives::cp_async_16(&storage.input[pack][part][tid],
                    y_row+pack*256+h_lane+part*128,true);
            cute::cp_async_fence();
        }
        float v[2]={0,0},inv[2];
        #pragma unroll
        for(int split=0;split<S;++split) if(lane<26) {
            v[0]+=values[split][0];v[1]+=values[split][1];
        }
        float half_coefficient_inverse=0.f,half_output_inverse=0.f;
        float coefficient_inverse0=0.f,coefficient_inverse1=0.f;
        if constexpr(VectorCoefficients) {
            const float sq0=__shfl_sync(0xffffffff,v[0],24);
            const float sq1=__shfl_sync(0xffffffff,v[1],24);
            const float usq0=__shfl_sync(0xffffffff,v[0],25);
            const float usq1=__shfl_sync(0xffffffff,v[1],25);
            const float sq=lane<16?sq0:sq1;
            const float usq=lane<16?usq0:usq1;
            half_coefficient_inverse=rsqrtf(sq/(4*H)+a.hc_eps);
            half_output_inverse=rsqrtf(usq/H+a.norm_eps);
            asm volatile("" : "+f"(half_coefficient_inverse), "+f"(half_output_inverse));
            coefficient_inverse0=__shfl_sync(0xffffffff,half_coefficient_inverse,0);
            coefficient_inverse1=__shfl_sync(0xffffffff,half_coefficient_inverse,16);
        }
        #pragma unroll
        for(int token=0;token<2;++token) {
            const float sq=VectorCoefficients?0.f:__shfl_sync(0xffffffff,v[token],24);
            const float usq=VectorCoefficients?0.f:__shfl_sync(0xffffffff,v[token],25);
            if(lane>=24) v[token]=0.f;
            if(lane<24) {
                v[token]=v[token]*(VectorCoefficients?(token==0?coefficient_inverse0:coefficient_inverse1):rsqrtf(sq/(4*H)+a.hc_eps));
                v[token]=v[token]*a.scales[lane<4?0:lane<8?1:2]+a.bases[lane];
            }
            if constexpr(!VectorCoefficients) {
            const int token_id=t0+token*(Rows/2);
            const float pre=__ppu_sgmdf(v[token])+a.pre_eps;
            if(lane<4 && (!TailTokens || token_id<tokens)) a.new_pre[token_id*4+lane]=pre;
            if(lane>=4 && lane<8 && (!TailTokens || token_id<tokens))
                a.new_post[token_id*4+lane-4]=__ppu_sgmdf(v[token])*a.post_scale;
            }
            if constexpr(!VectorCoefficients) inv[token]=rsqrtf(usq/H+a.norm_eps);
        }
        if constexpr(VectorCoefficients) {
            const float gate1=__shfl_sync(0xffffffff,v[1],lane%16);
            const float logit=lane<16?v[0]:gate1;
            float gate=__ppu_sgmdf(logit);
            asm volatile("" : "+f"(gate));
            const int column=lane%16,token_id=t0+(lane/16)*(Rows/2);
            if(column<4 && (!TailTokens || token_id<tokens)) a.new_pre[token_id*4+column]=gate+a.pre_eps;
            if(column>=4 && column<8 && (!TailTokens || token_id<tokens))
                a.new_post[token_id*4+column-4]=gate*a.post_scale;
        }
        const float c0=__shfl_sync(0xffffffff,v[0],(lane%16)+8);
        const float c1=__shfl_sync(0xffffffff,v[1],(lane%16)+8);
        float c=lane<16?c0:c1;
        float mx=c;
        mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1));
        mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));
        c=expf(c-mx);
        float sum=c+__shfl_xor_sync(0xffffffff,c,1);
        sum+=__shfl_xor_sync(0xffffffff,sum,2);
        c=c*__ppu_rcpf(sum)+a.sink_eps;
        const float inverse=VectorCoefficients?half_output_inverse:(lane<16?inv[0]:inv[1]);
        auto consume=[&](int iter) {
            if(iter) {
                sum=c+__shfl_xor_sync(0xffffffff,c,1);
                sum+=__shfl_xor_sync(0xffffffff,sum,2);
                c=c*__ppu_rcpf(sum+a.sink_eps);
            }
            const int h=iter*256+h_lane;
            #pragma unroll
            for(int part=0;part<2;++part) {
                uint4 input=storage.input[iter&3][part][tid],output;
                const auto* uv=reinterpret_cast<const __ppu_bfloat16*>(&input);
                float scaled[8];
                *reinterpret_cast<float4*>(scaled)=*reinterpret_cast<const float4*>(storage.weight+((h+part*128)^(((h+part*128)>>3)&4)));
                *reinterpret_cast<float4*>(scaled+4)=*reinterpret_cast<const float4*>(storage.weight+((h+part*128+4)^(((h+part*128+4)>>3)&4)));
                #pragma unroll
                for(int z=0;z<8;z+=2) {
                    const float u0=__bfloat162float(uv[z]),u1=__bfloat162float(uv[z+1]);
                    reinterpret_cast<__ppu_bfloat162*>(&output)[z/2]=
                        __floats2bfloat162_rn((u0*inverse)*scaled[z],(u1*inverse)*scaled[z+1]);
                }
                *reinterpret_cast<uint4*>(y_row+h+part*128)=output;
                if(part==0) {
                    sum=c+__shfl_xor_sync(0xffffffff,c,4);
                    sum+=__shfl_xor_sync(0xffffffff,sum,8);
                    c=c*__ppu_rcpf(sum+a.sink_eps);
                }
            }
        };
        for(int iter=0;iter<Iters-4;++iter) {
            cute::cp_async_wait<3>();consume(iter);
            #pragma unroll
            for(int part=0;part<2;++part)
                primitives::cp_async_16(&storage.input[iter&3][part][tid],
                    y_row+(iter+4)*256+h_lane+part*128,true);
            cute::cp_async_fence();
        }
        cute::cp_async_wait<3>();consume(Iters-4);
        cute::cp_async_wait<2>();consume(Iters-3);
        cute::cp_async_wait<1>();consume(Iters-2);
        cute::cp_async_wait<0>();consume(Iters-1);
        if(!TailTokens || t<tokens) a.new_comb[t*16+lane%16]=c;
        __syncthreads();
        if constexpr(LocalCompletion && !TailRows && S>=16) break;
    }
}

#endif

#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
__device__ __forceinline__ void manual_tf32_pair_n24(float* d, const uint32_t* a,
                                                   void* smem, int ki) {
    int addr[4];
    manual_lifetime_addresses(smem, ki, addr);
    asm volatile(
        "{\n"
        ".reg .b32 b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11;\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {b0,b1,b2,b3}, [%20], %24, %25, %26;\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x2.b16 {b4,b5}, [%21], %25, %25, %26;\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {b6,b7,b8,b9}, [%22], %24, %25, %26;\n"
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x2.b16 {b10,b11}, [%23], %25, %25, %26;\n"
        "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%12,%13,%14,%15}, {b0,b1,b2,b3}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        "ppu.tc02.mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%8,%9,%10,%11}, {%12,%13,%14,%15}, {b4,b5}, {%8,%9,%10,%11};\n"
        "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%16,%17,%18,%19}, {b6,b7,b8,b9}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        "ppu.tc02.mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%8,%9,%10,%11}, {%16,%17,%18,%19}, {b10,b11}, {%8,%9,%10,%11};\n"
        "}\n"
        : "+&r"(d[0]), "+&r"(d[1]), "+&r"(d[2]), "+&r"(d[3]), "+&r"(d[4]), "+&r"(d[5]), "+&r"(d[6]), "+&r"(d[7]), "+&f"(d[8]), "+&f"(d[9]), "+&f"(d[10]), "+&f"(d[11])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]), "l"(addr[0]), "l"(addr[1]), "l"(addr[2]), "l"(addr[3]), "r"(64), "r"(1), "r"(0)
        : "memory");
}
#endif

template<int H, int S, int Iters, bool FullRows=false, bool SingleKernel=false, bool LocalCompletion=false,
         int SingleTileM=32, int TaskSplits=2, bool TailRows=false, bool TailTokens=false, bool VectorCoefficients=true>
__device__ __forceinline__ void ppu_mega_mhc_shifted_bf16_device(Args a) {
    static_assert(H==5120, "Mega-mHC requires hidden=5120");
    static_assert(!SingleKernel || ((S>=8 && S<=80 && S%2==0) && Iters==20 && !FullRows),
                  "Single-kernel specialization requires shifted BF16 H5120/I20");
    static_assert(TaskSplits==1 || TaskSplits==2, "One or two splits per producer task");
    static_assert(!LocalCompletion || TaskSplits==2, "Local completion uses paired splits");
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
    using namespace primitives::tc02;
    constexpr bool SmallTile=SingleKernel && SingleTileM==16;
    constexpr int M=SingleKernel ? (SmallTile ? 16 : 32) : 64;
    constexpr int Threads=M*2;
    constexpr int BCubeStride = SingleKernel ? 24*32 : 32*32;
    struct __align__(1024) Storage {
        float weights[2][2*BCubeStride];
        __ppu_bfloat16 inputs[SingleKernel ? 4 : 5][M*K];
        union {
            struct { float coeff[M][24]; int next; };
            __ppu_bfloat16 x_scratch[SmallTile ? 2048 : SingleKernel ? M*K : 1];
        };
    };
    static_assert(!SingleKernel || sizeof(Storage)==(SmallTile ? 24576 : 32768),"X alias must retain occupancy");
    static_assert(SingleKernel || sizeof(Storage)>=5*H*sizeof(__ppu_bfloat16),
                  "Storage must cover the epilogue input and weight staging area");
    __shared__ __align__(1024) Storage storage;
    auto& weights=storage.weights;
    auto& inputs=storage.inputs;
    auto& next=storage.next;
    int tid=threadIdx.x, lane=tid%32, warp=tid/32;
    const int tokens=a.tokens;
    static_assert(!SingleKernel || (SingleTileM==16 || SingleTileM==32), "Unsupported tile size");
    auto* padding=reinterpret_cast<__ppu_bfloat16*>(a.partials+((tokens+63)/64)*S*64*P);
    const int SingleGroups=LocalCompletion ? int(gridDim.x) : (tokens+M-1)/M;
    int blocks=LocalCompletion ? SingleGroups : (tokens+63)/64;
    static_assert(!LocalCompletion || SingleKernel,"Local completion requires the single-kernel path");
    const uint32_t cooperative_task=blockIdx.x*S+blockIdx.y*2;
    int produced=0;
    while(true) {
        using TaskIndex=std::conditional_t<SingleKernel,uint32_t,int>;
        TaskIndex first_task;
        if constexpr(LocalCompletion) {
            // Cooperative launch requires all CTAs to be resident.
            first_task=cooperative_task;
        } else {
            if(tid==0) next=atomicAdd(a.counters+blocks,SingleKernel ? TaskSplits : 1);
            __syncthreads();
            first_task=static_cast<TaskIndex>(next);
            if constexpr(SingleKernel) __syncthreads();
            if(first_task>=(SingleKernel ? SingleGroups : blocks)*S) break;
        }
        float4 resident_coeff[2][6];
        if constexpr(SingleKernel) {
            #pragma unroll
            for(int part=0;part<2;++part) {
                const int t=(first_task/S)*M+warp*16+part*8+lane/4;
                if constexpr(TailRows) if(t>=tokens) {
                    #pragma unroll
                    for(int j=0;j<6;++j) resident_coeff[part][j]=make_float4(0,0,0,0);
                    continue;
                }
                resident_coeff[part][0]=*reinterpret_cast<const float4*>(a.pre+t*4);
                resident_coeff[part][1]=*reinterpret_cast<const float4*>(a.post+t*4);
                #pragma unroll
                for(int j=0;j<4;++j)
                    resident_coeff[part][2+j]=*reinterpret_cast<const float4*>(a.comb+t*16+j*4);
            }
        }
        __ppu_bfloat16* residual_row[2];
        __ppu_bfloat16* mixed_row[2];
        if constexpr(SingleKernel && TaskSplits==1) {
            #pragma unroll
            for(int part=0;part<2;++part) {
                const int t=(first_task/S)*M+warp*16+part*8+lane/4;
                residual_row[part]=a.new_residual+t*(4*H);
                mixed_row[part]=a.y+t*H;
                if constexpr(TailRows) if(t>=tokens) {
                    residual_row[part]=padding+(t-tokens)*5*H;
                    mixed_row[part]=padding+(t-tokens)*5*H+4*H;
                }
            }
        }
        #pragma unroll 1
        for(int subtask=0;subtask<(SingleKernel ? TaskSplits : 1);++subtask) {
        TaskIndex task=first_task+subtask;
        TaskIndex mb=task/S, split=task%S;
        int begin=(H/K)*split/S, end=(H/K)*(split+1)/S;
        if(subtask==0 && !SingleKernel)
        #pragma unroll
        for(int z=tid;z<M*6;z+=Threads) {
            int row=z/6,c=z%6,t=mb*M+row;
            const float* source=c==0 ? a.pre+t*4 : c==1 ? a.post+t*4 : a.comb+t*16+(c-2)*4;
            primitives::cp_async_16(&storage.coeff[row][c*4],source,(FullRows || t<tokens));
        }
        float acc[16]={}, sq[2]={}, usq[2]={};
        cute::AiuDesc desc; desc.dim_h=24; desc.dim_w=4*H; desc.stride_w=4*H;
        auto load_b_pair=[&](float* dest,int k) {
            // Two 24x32 FP32 cubes with a 32-element plane stride.
            asm volatile(
                "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.LLC::128B.padz.swzl.b32"
                "[%0], [%1], {%2,%3,%4}, {%5,%6,%7}, {%8,%9}, {%10,%11,%12}, %13;"
                :: "r"((void*)dest), "l"(a.fn),
                "r"(4*H), "r"(24), "r"(2), "r"(32), "r"(24), "r"(2),
                "l"((uint64_t)(4*H*4)), "l"((uint64_t)(32*4)),
                "r"(k), "r"(0), "r"(0), "r"(0));
        };
        auto load_x=[&](int xkb) {
            if constexpr(SingleKernel) {
                if(tid==(SmallTile ? 16 : 32)) {
                    asm volatile(
                        "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.rt.LLC::128B.padz.swzl.b16"
                        "[%0], [%1], {%2,%3,%4}, {%5,%6,%7}, {%8,%9}, {%10,%11,%12}, %13;"
                        :: "r"((void*)storage.x_scratch), "l"(a.x),
                        "r"(H), "r"(tokens), "r"(1), "r"(K), "r"(M), "r"(1),
                        "l"((uint64_t)(H*2)), "l"((uint64_t)(H*2*tokens)),
                        "r"(xkb*K), "r"(mb*M), "r"(0), "r"(0));
                }
            }
        };
        for(int kb=begin;kb<end;++kb) {
            cute::AiuDesc dr;dr.dim_h=tokens;dr.dim_w=H;dr.stride_w=4*H;
            cute::AiuDesc dx;dx.dim_h=tokens;dx.dim_w=H;dx.stride_w=H;
            if constexpr(S==5 || S==6) {
                if(kb==begin) {
                    if(tid==0) {
                        AiuLoadA::copy(inputs[0],a.residual,dr,kb*K,mb*M);
                        AiuLoadA::copy(inputs[4],a.x,dx,kb*K,mb*M);
                    }
                    if(tid==32) {
                        AiuLoadA::copy(inputs[1],a.residual+H,dr,kb*K,mb*M);
                        AiuLoadB::copy(weights[0],a.fn,desc,kb*K,0);
                    }
                    if(tid==64) {
                        AiuLoadA::copy(inputs[2],a.residual+2*H,dr,kb*K,mb*M);
                        AiuLoadB::copy(weights[0]+BCubeStride,a.fn,desc,kb*K+32,0);
                    }
                    if(tid==96) AiuLoadA::copy(inputs[3],a.residual+3*H,dr,kb*K,mb*M);
                }
            } else {
                if(tid==0 && kb==begin && (!SingleKernel || subtask==0)) {
                    if constexpr(SingleKernel) {
                        asm volatile(
                            "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.rt.LLC::128B.padz.swzl.b16"
                            "[%0], [%1], {%2,%3,%4}, {%5,%6,%7}, {%8,%9}, {%10,%11,%12}, %13;"
                            :: "r"((void*)inputs[0]), "l"(a.residual),
                            "r"(H), "r"(tokens), "r"(4), "r"(K), "r"(M), "r"(4),
                            "l"((uint64_t)(4*H*2)), "l"((uint64_t)(H*2)),
                            "r"(kb*K), "r"(mb*M), "r"(0), "r"(0));
                    } else {
                        #pragma unroll
                        for(int j=0;j<4;++j) AiuLoadA::copy(inputs[j],a.residual+j*H,dr,kb*K,mb*M);
                    }
                    if constexpr(!SingleKernel) AiuLoadA::copy(inputs[4],a.x,dx,kb*K,mb*M);
                    if constexpr(SingleKernel) load_b_pair(weights[0],kb*K);
                    else {
                    AiuLoadB::copy(weights[0],a.fn,desc,kb*K,0);
                    AiuLoadB::copy(weights[0]+BCubeStride,a.fn,desc,kb*K+32,0);
                    }
                }
            }
            uint4 delayed_u[2];
            uint4 x_preload[2][2];
            if constexpr(SingleKernel) if(kb==begin && subtask==0) load_x(kb);
            cute::cp_async_fence();cute::cp_async_wait<0>();__syncthreads();
            if constexpr(SingleKernel)
            #pragma unroll
            for(int xp=0;xp<2;++xp) {
                const int row=warp*16+xp*8+lane/4;
                const int offset=row*K+((lane%4)*16^((row&7)*8));
                #pragma unroll
                for(int xh=0;xh<2;++xh)
                    x_preload[xp][xh]=*reinterpret_cast<const uint4*>(storage.x_scratch+(offset^(xh*8)));
            }

            // The tile-start barrier retires route3 reads before staging route1.
            if constexpr(S==5) {
                if(tid==0) AiuLoadB::copy(weights[1],a.fn,desc,H+kb*K,0);
                if(tid==32) AiuLoadB::copy(weights[1]+BCubeStride,a.fn,desc,H+kb*K+32,0);
                cute::cp_async_fence();
            }
            #pragma unroll
            for(int part=0;part<2;++part) {
                if constexpr(S==80 || S==40 || S==8 || S==12 || S==6 || S==16 || SingleKernel) {
                    if constexpr(SingleKernel) {
                        int row=warp*16+part*8+lane/4, k=(lane%4)*16;
                        int t=mb*M+row, offset=row*64+(k^((row&7)*8));
                        float4 coeff_vec[5];
                        #pragma unroll
                        for(int j=0;j<5;++j)
                            coeff_vec[j]=resident_coeff[part][j+1];
                        const float* local_coeff=reinterpret_cast<const float*>(coeff_vec);
                        __ppu_bfloat16* residual_out=a.new_residual+(t*(4*H)+kb*K+k);
                        if constexpr(TaskSplits==1) residual_out=residual_row[part]+kb*K+k;
                        else if constexpr(TailRows) if(t>=tokens)
                            residual_out=padding+(t-tokens)*5*H+kb*K+k;
                        uint4 packed_r[4][2];
                        uint4 old_preload[4][2];
                        #pragma unroll
                        for(int half=0;half<2;++half)
                            #pragma unroll
                            for(int j=0;j<4;++j)
                                old_preload[j][half]=*reinterpret_cast<const uint4*>(inputs[j]+(offset^(half*8)));
                        asm volatile("" ::: "memory");
                        #pragma unroll 2
                        for(int half=0;half<2;++half) {
                            uint4 packed[5];
                            #pragma unroll
                            for(int j=0;j<4;++j)
                                packed[j]=old_preload[j][half];
                            packed[4]=x_preload[part][half];
                            #pragma unroll
                            for(int z=0;z<8;z+=2) {
                            float old[5][2];
                            #pragma unroll
                            for(int j=0;j<5;++j) {
                                const auto* value=reinterpret_cast<const __ppu_bfloat16*>(&packed[j]);
                                old[j][0]=__bfloat162float(value[z]);
                                old[j][1]=__bfloat162float(value[z+1]);
                            }
                            #pragma unroll
                            for(int route=0;route<4;++route) {
                                const float post=local_coeff[route];
                                float v0=old[4][0]*post,v1=old[4][1]*post;
                                #pragma unroll
                                for(int j=0;j<4;++j) {
                                    const float comb=local_coeff[4+j*4+route];
                                    v0=fmaf(old[j][0],comb,v0);
                                    v1=fmaf(old[j][1],comb,v1);
                                }
                                const auto rounded=__floats2bfloat162_rn(v0,v1);
                                reinterpret_cast<__ppu_bfloat162*>(&packed_r[route][half])[z/2]=rounded;
                            }
                        }
                        }
                        float u[16]={};
                        float4 pre_vector=resident_coeff[part][0];
                        #pragma unroll
                        for(int route=0;route<4;++route) {
                            const float pre=reinterpret_cast<const float*>(&pre_vector)[route];
                            #pragma unroll
                            for(int z=0;z<16;z+=2) {
                                const auto rounded=reinterpret_cast<const __ppu_bfloat162*>(packed_r[route])[z/2];
                                const float2 v=__bfloat1622float2(rounded);
                                sq[part]=fmaf(v.x,v.x,sq[part]);sq[part]=fmaf(v.y,v.y,sq[part]);
                                {u[z]=fmaf(v.x,pre,u[z]);u[z+1]=fmaf(v.y,pre,u[z+1]);}
                            }
                            #pragma unroll
                            for(int half=0;half<2;++half) {
                                const uint4 v=packed_r[route][half];
                                const uint4 matrix_packed=make_uint4(
                                    __byte_perm(v.x,v.z,0x5410), __byte_perm(v.x,v.z,0x7632),
                                    __byte_perm(v.y,v.w,0x5410), __byte_perm(v.y,v.w,0x7632));
                                cutlass::arch::shared_store<16>(
                                    static_cast<uint32_t>(__cvta_generic_to_shared(inputs[route]+(offset^(half*8)))),
                                    &matrix_packed);
                                asm volatile("ppu.st.global.v4.u32 [%0], {%1,%2,%3,%4};"
                                    :: "l"(residual_out+route*H+half*8),
                                       "r"(v.x),"r"(v.y),"r"(v.z),"r"(v.w));
                            }
                        }
                        {
                            uint4 packed_u[2];
                            #pragma unroll
                            for(int z=0;z<16;z+=2) {
                                auto packed=__floats2bfloat162_rn(u[z],u[z+1]);
                                reinterpret_cast<__ppu_bfloat162*>(&packed_u)[z/2]=packed;
                                if(part!=1) {
                                    float2 v=__bfloat1622float2(packed);
                                    usq[part]=fmaf(v.x,v.x,usq[part]);usq[part]=fmaf(v.y,v.y,usq[part]);
                                }
                            }
                            #pragma unroll
                            for(int half=0;half<2;++half) {
                                if(part==1) delayed_u[half]=packed_u[half];
                                else {
                                    auto* dest=a.y+(t*H+kb*K+k+half*8);
                                    if constexpr(TaskSplits==1) dest=mixed_row[part]+kb*K+k+half*8;
                                    else if constexpr(TailRows) if(t>=tokens)
                                        dest=padding+(t-tokens)*5*H+4*H+kb*K+k+half*8;
                                    *reinterpret_cast<uint4*>(dest)=packed_u[half];
                                }
                            }
                        }
                        if(part==0) {
                            if(tid==0) load_b_pair(weights[1],H+kb*K);
                            cute::cp_async_fence();
                        }
                    } else {
                        int row=warp*16+part*8+lane/4, k=(lane%4)*16;
                        int t=mb*M+row, offset=row*64+(k^((row&7)*8));
                        uint4 packed[5][2];
                        #pragma unroll
                        for(int j=0;j<5;++j) {
                            #pragma unroll
                            for(int half=0;half<2;++half) packed[j][half]=*reinterpret_cast<uint4*>(inputs[j]+(offset^(half*8)));
                        }
                        float4 coeff_vec[6];
                        #pragma unroll
                        for(int j=0;j<6;++j) coeff_vec[j]=*reinterpret_cast<const float4*>(storage.coeff[row]+j*4);
                        const float* local_coeff=reinterpret_cast<const float*>(coeff_vec);
                        __ppu_bfloat16* residual_out=a.new_residual+(int64_t)t*(4*H)+kb*K+k;
                        uint4 packed_r[4][2];
                        #pragma unroll
                        for(int z=0;z<16;z+=2) {
                            float old[5][2];
                            #pragma unroll
                            for(int j=0;j<5;++j) {
                                const auto* value=reinterpret_cast<const __ppu_bfloat16*>(&packed[j]);
                                old[j][0]=__bfloat162float(value[z]);
                                old[j][1]=__bfloat162float(value[z+1]);
                            }
                            #pragma unroll
                            for(int route=0;route<4;++route) {
                                const float post=local_coeff[4+route];
                                float v0=old[4][0]*post,v1=old[4][1]*post;
                                #pragma unroll
                                for(int j=0;j<4;++j) {
                                    const float comb=local_coeff[8+j*4+route];
                                    v0=fmaf(old[j][0],comb,v0);
                                    v1=fmaf(old[j][1],comb,v1);
                                }
                                const auto rounded=__floats2bfloat162_rn(v0,v1);
                                reinterpret_cast<__ppu_bfloat162*>(packed_r[route])[z/2]=rounded;
                            }
                        }
                        float u[16]={};
                        #pragma unroll
                        for(int route=0;route<4;++route) {
                            const float pre=local_coeff[route];
                            #pragma unroll
                            for(int z=0;z<16;z+=2) {
                                const auto rounded=reinterpret_cast<const __ppu_bfloat162*>(packed_r[route])[z/2];
                                const float2 v=__bfloat1622float2(rounded);
                                sq[part]=fmaf(v.x,v.x,sq[part]);sq[part]=fmaf(v.y,v.y,sq[part]);
                                {u[z]=fmaf(v.x,pre,u[z]);u[z+1]=fmaf(v.y,pre,u[z+1]);}
                            }
                            #pragma unroll
                            for(int half=0;half<2;++half) {
                                cutlass::arch::shared_store<16>(
                                    static_cast<uint32_t>(__cvta_generic_to_shared(inputs[route]+(offset^(half*8)))),
                                    &packed_r[route][half]);
                                cutlass::arch::global_store<uint4,16>(packed_r[route][half], residual_out+route*H+half*8, (FullRows || t<tokens));
                            }
                        }
                        {
                            uint4 packed_u[2];
                            #pragma unroll
                            for(int z=0;z<16;z+=2) {
                                auto packed=__floats2bfloat162_rn(u[z],u[z+1]);
                                reinterpret_cast<__ppu_bfloat162*>(&packed_u)[z/2]=packed;
                                float2 v=__bfloat1622float2(packed);
                                usq[part]=fmaf(v.x,v.x,usq[part]);usq[part]=fmaf(v.y,v.y,usq[part]);
                            }
                            #pragma unroll
                            for(int half=0;half<2;++half)
                                if((FullRows || t<tokens)) *reinterpret_cast<uint4*>(a.y+(int64_t)t*H+kb*K+k+half*8)=packed_u[half];
                        }
                    }
                } else {
                    int row=warp*16+part*8+lane/4, k=(lane%4)*16;
                    int t=mb*M+row, offset=row*64+(k^((row&7)*8));
                    uint4 packed[5][2];
                    #pragma unroll
                    for(int j=0;j<5;++j) {
                        #pragma unroll
                        for(int half=0;half<2;++half) packed[j][half]=*reinterpret_cast<uint4*>(inputs[j]+(offset^(half*8)));
                    }
                    float old[5][16];
                    #pragma unroll
                    for(int j=0;j<5;++j) {
                        const auto* v=reinterpret_cast<const __ppu_bfloat16*>(&packed[j]);
                        #pragma unroll
                        for(int z=0;z<16;++z) old[j][z]=__bfloat162float(v[z]);
                    }
                    float4 coeff_vec[6];
                    #pragma unroll
                    for(int j=0;j<6;++j) coeff_vec[j]=*reinterpret_cast<const float4*>(storage.coeff[row]+j*4);
                    const float* local_coeff=reinterpret_cast<const float*>(coeff_vec);
                    __ppu_bfloat16* residual_out=a.new_residual+(int64_t)t*(4*H)+kb*K+k;
                    float u[16]={};
                    #pragma unroll
                    for(int route=0;route<4;++route) {
                        float post=local_coeff[4+route];
                        float pre=local_coeff[route];
                        float comb[4];
                        #pragma unroll
                        for(int j=0;j<4;++j) comb[j]=local_coeff[8+j*4+route];
                        uint4 packed_r[2];
                        #pragma unroll
                        for(int z=0;z<16;z+=2) {
                            float v0=old[4][z]*post,v1=old[4][z+1]*post;
                            #pragma unroll
                            for(int j=0;j<4;++j) {
                                v0=fmaf(old[j][z],comb[j],v0);
                                v1=fmaf(old[j][z+1],comb[j],v1);
                            }
                            auto packed=__floats2bfloat162_rn(v0,v1);
                            reinterpret_cast<__ppu_bfloat162*>(&packed_r)[z/2]=packed;
                            float2 v=__bfloat1622float2(packed);
                            sq[part]=fmaf(v.x,v.x,sq[part]);sq[part]=fmaf(v.y,v.y,sq[part]);
                            {u[z]=fmaf(v.x,pre,u[z]);u[z+1]=fmaf(v.y,pre,u[z+1]);}
                        }
                        #pragma unroll
                        for(int half=0;half<2;++half) {
                            *reinterpret_cast<uint4*>(inputs[route]+(offset^(half*8)))=packed_r[half];
                            if((FullRows || t<tokens)) *reinterpret_cast<uint4*>(residual_out+route*H+half*8)=packed_r[half];
                        }
                    }
                    {
                        uint4 packed_u[2];
                        #pragma unroll
                        for(int z=0;z<16;z+=2) {
                            auto packed=__floats2bfloat162_rn(u[z],u[z+1]);
                            reinterpret_cast<__ppu_bfloat162*>(&packed_u)[z/2]=packed;
                            float2 v=__bfloat1622float2(packed);
                            usq[part]=fmaf(v.x,v.x,usq[part]);usq[part]=fmaf(v.y,v.y,usq[part]);
                        }
                        #pragma unroll
                        for(int half=0;half<2;++half)
                            if((FullRows || t<tokens)) *reinterpret_cast<uint4*>(a.y+(int64_t)t*H+kb*K+k+half*8)=packed_u[half];
                    }
                }
            }
            __syncwarp();
            uint32_t next_route_a[SingleKernel?16:1];
            #pragma unroll (SingleKernel ? 4 : 1)
            for(int route=0;route<4;++route) {
                if(!SingleKernel && route<3 && (S!=5 || route>0)) {
                    if constexpr(S==5 || S==6) {
                        if(tid==0) AiuLoadB::copy(weights[(route+1)%2],a.fn,desc,(route+1)*H+kb*K,0);
                        if(tid==32) AiuLoadB::copy(weights[(route+1)%2]+BCubeStride,a.fn,desc,(route+1)*H+kb*K+32,0);
                    } else {
                        if(tid==0) {
                            AiuLoadB::copy(weights[(route+1)%2],a.fn,desc,(route+1)*H+kb*K,0);
                            AiuLoadB::copy(weights[(route+1)%2]+BCubeStride,a.fn,desc,(route+1)*H+kb*K+32,0);
                        }
                    }
                    cute::cp_async_fence();
                }
                float r[32];
                auto load_packed_a=[&](int packet) {
                    uint32_t packed[4];
                    cute::PPU0015_TSM_LD_SWZL<float,32,32,false,false,1>::copy(
                        packed,inputs[route],packet*8,warp*16);
                    #pragma unroll
                    for(int j=0;j<4;++j) {
                        const int i=packet*8+(j/2)*4+(j&1);
                        r[i]=__uint_as_float(packed[j]<<16);
                        r[i+2]=__uint_as_float(packed[j]&0xffff0000u);
                    }
                };
                if constexpr(SingleKernel) {
                    if(route==3) {
                        #pragma unroll
                        for(int packet=0;packet<4;++packet) {
                            #pragma unroll
                            for(int j=0;j<4;++j) {
                                const uint32_t v=next_route_a[packet*4+j];
                                const int i=packet*8+(j/2)*4+(j&1);
                                r[i]=__uint_as_float(v<<16);
                                r[i+2]=__uint_as_float(v&0xffff0000u);
                            }
                        }
                    }
                } else {
                #pragma unroll
                for(int i=0;i<32;++i) {
                    int row=warp*16+(i&1)*8+lane/4;
                    int k=(i/4)*8+((i%4)/2)*4+lane%4;
                    r[i]=__bfloat162float(inputs[route][row*64+(k^((row&7)*8))]);
                }
                }
                if(route==3 && !SingleKernel) {
                    __syncthreads();
                    if constexpr(S==5 || S==6) {
                        if(kb+1<end) {
                            if(tid==0) {
                                AiuLoadA::copy(inputs[0],a.residual,dr,(kb+1)*K,mb*M);
                                AiuLoadA::copy(inputs[4],a.x,dx,(kb+1)*K,mb*M);
                            }
                            if(tid==32) {
                                AiuLoadA::copy(inputs[1],a.residual+H,dr,(kb+1)*K,mb*M);
                                AiuLoadB::copy(weights[0],a.fn,desc,(kb+1)*K,0);
                            }
                            if(tid==64) {
                                AiuLoadA::copy(inputs[2],a.residual+2*H,dr,(kb+1)*K,mb*M);
                                AiuLoadB::copy(weights[0]+BCubeStride,a.fn,desc,(kb+1)*K+32,0);
                            }
                            if(tid==96) AiuLoadA::copy(inputs[3],a.residual+3*H,dr,(kb+1)*K,mb*M);
                        }
                    } else {
                        if(tid==0 && kb+1<end) {
                            #pragma unroll
                            for(int j=0;j<4;++j) AiuLoadA::copy(inputs[j],a.residual+j*H,dr,(kb+1)*K,mb*M);
                            AiuLoadA::copy(inputs[4],a.x,dx,(kb+1)*K,mb*M);
                            AiuLoadB::copy(weights[0],a.fn,desc,(kb+1)*K,0);
                            AiuLoadB::copy(weights[0]+BCubeStride,a.fn,desc,(kb+1)*K+32,0);
                        }
                    }
                    cute::cp_async_fence();
                }
                if constexpr(S==12 || S==16 || SingleKernel) {
                    if constexpr(SingleKernel) {
                        if(route==3) {
                            if(warp==0) {
                                // Wait for B3 while leaving next-tile A in flight.
                                if(kb+1<end || subtask<TaskSplits-1) cute::cp_async_wait<1>();
                                else cute::cp_async_wait<0>();
                            }
                            __syncthreads();
                        }
                        uint32_t b_packet[4][12];
                        #pragma unroll
                        for(int packet=0;packet<4;++packet)
                            manual_lifetime_load_n24<BCubeStride>(b_packet[packet],weights[route%2],packet*2);

                        if(route<2) {
                            if(warp==0) cute::cp_async_wait<0>();
                            __syncthreads();
                            if(tid==0) load_b_pair(weights[route%2],(route+2)*H+kb*K);
                            if(route==0 && (kb+1<end || subtask<TaskSplits-1)) load_x(kb+1);
                            cute::cp_async_fence();
                        }
                        if(route<3) {
                            #pragma unroll
                            for(int packet=0;packet<4;++packet) load_packed_a(packet);
                        }
                        if(route==2) {
                            #pragma unroll
                            for(int packet=0;packet<4;++packet)
                                cute::PPU0015_TSM_LD_SWZL<float,32,32,false,false,1>::copy(
                                    next_route_a+packet*4,inputs[3],packet*8,warp*16);
                            // Both warps must retire current readers before TSM reuse.
                            __syncthreads();

                            if(tid==0 && (kb+1<end || subtask<TaskSplits-1)) {
                                asm volatile(
                                    "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.rt.LLC::128B.padz.swzl.b16"
                                    "[%0], [%1], {%2,%3,%4}, {%5,%6,%7}, {%8,%9}, {%10,%11,%12}, %13;"
                                    :: "r"((void*)inputs[0]), "l"(a.residual),
                                    "r"(H), "r"(tokens), "r"(4), "r"(K), "r"(M), "r"(4),
                                    "l"((uint64_t)(4*H*2)), "l"((uint64_t)(H*2)),
                                    "r"((kb+1)*K), "r"(mb*M), "r"(0), "r"(0));
                            }
                            if(tid==(SmallTile ? 16 : 32) && (kb+1<end || subtask<TaskSplits-1))
                                load_b_pair(weights[0],(kb+1)*K);
                            cute::cp_async_fence();
                        }
                        #pragma unroll
                        for(int packet=0;packet<4;++packet)
                            manual_lifetime_drain_n24(acc,reinterpret_cast<const uint32_t*>(r+packet*8),b_packet[packet]);
                        if(route==2) {
                            #pragma unroll
                            for(int z=0;z<16;z+=2) {
                                const auto rounded=reinterpret_cast<const __ppu_bfloat162*>(delayed_u)[z/2];
                                const float2 v=__bfloat1622float2(rounded);
                                usq[1]=fmaf(v.x,v.x,usq[1]);
                                usq[1]=fmaf(v.y,v.y,usq[1]);
                            }
                            const int t=mb*M+warp*16+8+lane/4,k=(lane%4)*16;
                            #pragma unroll
                            for(int half=0;half<2;++half) {
                                auto* dest=a.y+(t*H+kb*K+k+half*8);
                                if constexpr(TaskSplits==1) dest=mixed_row[1]+kb*K+k+half*8;
                                else if constexpr(TailRows) if(t>=tokens)
                                    dest=padding+(t-tokens)*5*H+4*H+kb*K+k+half*8;
                                *reinterpret_cast<uint4*>(dest)=delayed_u[half];
                            }
                        }

                    } else {
                    uint32_t b_packet[2][12];
                    manual_lifetime_load_n24<BCubeStride>(b_packet[0], weights[route%2]);
                    manual_lifetime_advance_n24<BCubeStride>(acc, reinterpret_cast<const uint32_t*>(r),
                                                b_packet[0], b_packet[1], weights[route%2], 2);
                    manual_lifetime_advance_n24<BCubeStride>(acc, reinterpret_cast<const uint32_t*>(r+8),
                                                b_packet[1], b_packet[0], weights[route%2], 4);
                    manual_lifetime_advance_n24<BCubeStride>(acc, reinterpret_cast<const uint32_t*>(r+16),
                                                b_packet[0], b_packet[1], weights[route%2], 6);
                    manual_lifetime_drain_n24(acc, reinterpret_cast<const uint32_t*>(r+24),
                                              b_packet[1]);
                    }
                } else if constexpr(S==80 || S==8) {
                    uint32_t b_packet[2][16];
                    manual_lifetime_load(b_packet[0], weights[route%2]);
                    manual_lifetime_advance(acc, reinterpret_cast<const uint32_t*>(r),
                                            b_packet[0], b_packet[1], weights[route%2], 2);
                    manual_lifetime_advance(acc, reinterpret_cast<const uint32_t*>(r+8),
                                            b_packet[1], b_packet[0], weights[route%2], 4);
                    manual_lifetime_advance(acc, reinterpret_cast<const uint32_t*>(r+16),
                                            b_packet[0], b_packet[1], weights[route%2], 6);
                    manual_lifetime_drain(acc, reinterpret_cast<const uint32_t*>(r+24),
                                          b_packet[1]);
                } else if constexpr(S==5 || S==6) {
                    #pragma unroll
                    for(int ki=0;ki<8;ki+=2) {
                        manual_tf32_pair_n24(acc, reinterpret_cast<const uint32_t*>(r+ki*4),
                                             weights[route%2], ki);
                    }
                } else if constexpr(S==40) {
                    #pragma unroll
                    for(int ki=0;ki<8;ki+=2) {
                        manual_tf32_pair(acc, reinterpret_cast<const uint32_t*>(r+ki*4),
                                         weights[route%2], ki);
                    }
                } else {
                    #pragma unroll
                    for(int ki=0;ki<8;++ki) {
                        float b[8];
                        #pragma unroll
                        for(int g=0;g<2;++g) BTileLoad::copy(b+g*4,weights[route%2],(ki%4)*8,g*16,ki/4,0);
                        #pragma unroll
                        for(int g=0;g<2;++g) {
                            float* d=acc+g*8;
                            const auto* a_frag=reinterpret_cast<const uint32_t*>(r+ki*4);
                            const auto* b_frag=reinterpret_cast<const uint32_t*>(b+g*4);
                            cute::PPU0015_16x16x8_F32TF32TF32F32_TN::fma(
                                d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],
                                a_frag[0],a_frag[1],a_frag[2],a_frag[3],
                                b_frag[0],b_frag[1],b_frag[2],b_frag[3],
                                d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7]);
                        }
                    }
                }
                if(route<3 && !SingleKernel) {
                    cute::cp_async_wait<0>();
                    __syncthreads();
                }
            }
        }
        float* p=a.partials+(SingleKernel
            ? ((mb/(64/M))*S*64+split*64+(mb%(64/M))*M)*P
            : (int64_t)task*M*P);
        #pragma unroll
        for(int i=0;i<8;++i) {
            int row=warp*16+(i&1)*8+lane/4;
            int c=(i/2)*8+(lane%4)*2;
            if(c<24) {p[row*P+c]=acc[i*2];p[row*P+c+1]=acc[i*2+1];}
        }
        #pragma unroll
        for(int i=0;i<2;++i) {
            float ss=sq[i], uu=usq[i];
            #pragma unroll
            for(int d=2;d>=1;d/=2) {
                ss+=__shfl_xor_sync(0xffffffff,ss,d);
                uu+=__shfl_xor_sync(0xffffffff,uu,d);
            }
            if(lane%4==0) {int row=warp*16+i*8+lane/4;p[row*P+24]=ss;p[row*P+25]=uu;}
        }
        if constexpr(SingleKernel) ++produced;
        if constexpr(!SingleKernel) {
            // Publish every thread's stores before the leader increments the counter.
            __threadfence();
            __syncthreads();
            if(tid==0) atomicAdd(a.counters+mb,1);
            if constexpr(S!=12) __syncthreads();
        } else {
            if(subtask<TaskSplits-1) __syncthreads();
        }
        }
        if constexpr(LocalCompletion) break;
    }
    // Schedule all producers before waiting; arrive_inc releases their stores.
    if constexpr(SingleKernel) {
        if constexpr(LocalCompletion) {
            cutlass::Barrier::arrive_inc(a.counters,tid,cooperative_task/S,2);
        } else if(produced) cutlass::Barrier::arrive_inc(a.counters,tid,0,produced);
        else __syncthreads();
        single_ring_finish<S,LocalCompletion,SingleTileM,TailRows,TailTokens,LocalCompletion || VectorCoefficients>(a,&storage,cooperative_task);
    }
    if constexpr(!SingleKernel) {
        for(int task=blockIdx.x;task<(tokens+3)/4;task+=gridDim.x) {
            int mb=task/16, row=(task%16)*4;
            if(tid==0) while(atomicAdd(a.counters+mb,0)!=S) {}
            __syncthreads();
            finish<H,S,Iters,FullRows>(a,mb,reinterpret_cast<__ppu_bfloat16*>(&storage),row,row+4);
            // finish() already ends each four-row task with a CTA barrier.
            if constexpr(S!=12) __syncthreads();
        }
    }
    // Reset after every CTA reaches exit; later launches must be stream-ordered.
    if constexpr(!SingleKernel) __threadfence();
    __syncthreads();
    if(tid==0 && atomicAdd(a.counters+blocks+1,1)==gridDim.x*(LocalCompletion ? S/2 : 1)-1) {
        if constexpr(LocalCompletion) {
            #pragma unroll
            for(int i=0;i<(SingleGroups+2)/4*4;i+=4) {
                asm volatile("ppu.st.global.v4.u32 [%0], {%1,%1,%1,%1};"
                    :: "l"(a.counters+i), "r"(0) : "memory");
            }
            #pragma unroll
            for(int i=(SingleGroups+2)/4*4;i<SingleGroups+2;++i)
                asm volatile("ppu.st.global.u32 [%0], %1;"
                    :: "l"(a.counters+i), "r"(0) : "memory");
        } else {
        if constexpr(SingleKernel) atomicExch(a.counters,0);
        else
            for(int mb=0;mb<blocks;++mb) atomicExch(a.counters+mb,0);
        atomicExch(a.counters+blocks,0);
        atomicExch(a.counters+blocks+1,0);
        }
    }
#endif
}
}

#endif // DG_MHC_ARGS_ONLY
