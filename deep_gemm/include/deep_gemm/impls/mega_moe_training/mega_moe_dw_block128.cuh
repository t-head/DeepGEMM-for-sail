#pragma once
#include <deep_gemm/impls/mega_moe_training/mega_moe_training.cuh>
#include <deep_gemm/common/mega_moe_training.cuh>

namespace deep_gemm::mega_moe_training {

template<int Rows,int Threads,bool ScaleA,bool ReuseRowScales=false,bool PreciseScale=false>
__device__ __forceinline__ void mega_moe_dw_emit_block128(
        const uint4 (&cached)[Rows*16/Threads],uint8_t* output,float* scales,
        int width,int start,int expert_k,int token_base,int feature_base,
        uint32_t* tile,uint32_t* maxima,const float* row_scales=nullptr) {
    const int thread=threadIdx.x,lane=thread&31,warp=thread>>5;
    unsigned maximum=0;
    if constexpr(ReuseRowScales) {
        // Row scales use the same monotone FP32 clamp/divide operation.
        // Their positive bitwise maximum equals the block's original scale.
        maximum=thread<Rows?__float_as_uint(row_scales[thread]):0u;
    } else {
    #pragma unroll
    for(int j=0;j<Rows*16/Threads;++j) {
        const unsigned words[4]={cached[j].x,cached[j].y,cached[j].z,cached[j].w};
        #pragma unroll
        for(int pair=0;pair<4;++pair) {
            const unsigned lo=words[pair]&0x7fffu,hi=(words[pair]>>16)&0x7fffu;
            if(lo<=0x7f80u)maximum=max(maximum,lo);
            if(hi<=0x7f80u)maximum=max(maximum,hi);
        }
    }
    }
    #pragma unroll
    for(int delta=16;delta;delta>>=1)maximum=max(maximum,__shfl_down_sync(0xffffffffu,maximum,delta));
    if(lane==0)maxima[warp]=maximum;
    __syncthreads();
    if(warp==0) {
        maximum=lane<Threads/32?maxima[lane]:0;
        #pragma unroll
        for(int delta=16;delta;delta>>=1)maximum=max(maximum,__shfl_down_sync(0xffffffffu,maximum,delta));
        if(lane==0) {
            if constexpr(ReuseRowScales)maxima[0]=maximum;
            else if constexpr(PreciseScale)maxima[0]=__float_as_uint(fmaxf(__uint_as_float(maximum<<16)/448.f,0x1p-120f));
            else maxima[0]=__float_as_uint(fmaxf(__uint_as_float(maximum<<16),1e-4f)/448.f);
        }
    }
    __syncthreads();
    const float scale=__uint_as_float(maxima[0]),inverse=1.f/scale;
    if constexpr(ScaleA) {
        if(thread<128)for(int part=0;part<Rows/128 && token_base+part*128<expert_k;++part)scales[int64_t((start+token_base)/128+part)*width+feature_base+thread]=scale;
    } else {
        if(thread==0)for(int part=0;part<Rows/128 && token_base+part*128<expert_k;++part)scales[int64_t(start/128)*(width/128)+int64_t(feature_base/128)*(expert_k/128)+token_base/128+part]=scale;
    }
    auto slot=[](int row,int word){return row*32+(word^(row&3)^((row>>2)&3)^(row>>4));};
    #pragma unroll
    for(int j=0;j<Rows*16/Threads;++j) {
        const unsigned words[4]={cached[j].x,cached[j].y,cached[j].z,cached[j].w};
        unsigned q[4];
        #pragma unroll
        for(int pair=0;pair<4;++pair)q[pair]=epilogue::quant::fp8_pair(__uint_as_float(words[pair]<<16)*inverse,__uint_as_float(words[pair]&0xffff0000u)*inverse);
        const int row=thread/16+j*(Threads/16),word=(thread%16)*2;
        tile[slot(row,word)]=q[0]|(q[1]<<16);tile[slot(row,word+1)]=q[2]|(q[3]<<16);
    }
    __syncthreads();
    // Four lanes own a4feature x16token rectangle. Each lane loads
    // four words, transposes4x4bytes, then exchanges word ownership.
    const int lane4=thread&3;
    for(int group=thread/4;group<Rows*2;group+=Threads/4) {
        const int feature=(group/(Rows/16))*4,token=(group%(Rows/16))*16;
        const int row=token+lane4*4,word=feature/4;
        const unsigned a=tile[slot(row,word)],b=tile[slot(row+1,word)];
        const unsigned c=tile[slot(row+2,word)],d=tile[slot(row+3,word)];
        const unsigned ab01=__byte_perm(a,b,0x5140u),cd01=__byte_perm(c,d,0x5140u);
        const unsigned ab23=__byte_perm(a,b,0x7362u),cd23=__byte_perm(c,d,0x7362u);
        unsigned v0=__byte_perm(ab01,cd01,0x5410u),v1=__byte_perm(ab01,cd01,0x7632u);
        unsigned v2=__byte_perm(ab23,cd23,0x5410u),v3=__byte_perm(ab23,cd23,0x7632u);
        auto swap=[&](unsigned& low,unsigned& high,int bit) {
            const unsigned send=(lane4&bit)?low:high;
            const unsigned received=__shfl_xor_sync(0xffffffffu,send,bit);
            if(lane4&bit)low=received;else high=received;
        };
        swap(v0,v1,1);swap(v2,v3,1);
        swap(v0,v2,2);swap(v1,v3,2);
        const int64_t at=int64_t(start)*width+int64_t(feature_base+feature+lane4)*expert_k+token_base+token;
        if(token_base+token<expert_k)*reinterpret_cast<uint4*>(output+at)=make_uint4(v0,v1,v2,v3);
    }
    __syncthreads();
}

template<int Width,int Tokens,int Topk,int MetadataBM,bool Gradient,int Rows=128,int Threads=Rows==512?512:256>
__device__ __forceinline__ void mega_moe_dw_block128(const Params& p,const MegaMoEDwBlock128Operands& dw) {
    constexpr int Segments=Width/128;
    static_assert(Rows==128 || Rows==512);
    const int thread=threadIdx.x,lane=thread&15;
    const unsigned mask=0xffffu<<((thread&31)>=16?16:0);
    const int* lengths=peer_pointer<int32_t>(p,Gradient?6:4);
    auto* status=peer_pointer<uint32_t>(p,Gradient?7:3);
    const int first=p.dim[5],last=p.dim[6];
    int64_t prefix64=0,rows64=0;int token_tiles=0;
    for(int expert=0;expert<last;++expert) {
        const int count=__ldg(lengths+expert);
        if(count<0 || count%128){if(thread==0 && blockIdx.x==0)atomicOr(status,2u);return;}
        if(expert<first)prefix64+=count;else {rows64+=count;token_tiles+=(int64_t(count)+Rows-1)/Rows;}
    }
    if(prefix64+rows64>p.dim[0]){if(thread==0 && blockIdx.x==0)atomicOr(status,2u);return;}
    const int prefix=int(prefix64),rows=int(rows64);
    if(blockIdx.x==0) {
        auto* metadata=reinterpret_cast<uint4*>(peer_pointer<int32_t>(p,Gradient?11:7));
        int blocks=0,global_prefix=0;
        for(int expert=0;expert<last;++expert) {
            const int count=__ldg(lengths+expert);
            if(expert>=first) {
                const int tiles=(count+MetadataBM-1)/MetadataBM;
                for(int tile=thread;tile<tiles;tile+=blockDim.x)metadata[1+blocks+tile]=make_uint4(expert,count,blocks,global_prefix);
                blocks+=tiles;
            }
            global_prefix+=count;
        }
        if(thread==0)reinterpret_cast<int32_t*>(metadata)[0]=blocks;
    }
    __shared__ uint64_t addresses[Rows];
    __shared__ float row_scores[Gradient?Rows:1],row_scales[Rows];
    __shared__ uint32_t tile[Rows*32],maxima[Threads/32];
    for(int work=blockIdx.x;work<token_tiles*Segments;work+=gridDim.x) {
        int token_tile=work/Segments;
        const int feature_base=(work%Segments)*128;
        int expert_start=prefix,expert=first,expert_k=0;
        for(;expert<last;++expert) {
            expert_k=__ldg(lengths+expert);
            const int tiles=(expert_k+Rows-1)/Rows;
            if(token_tile<tiles)break;
            token_tile-=tiles;expert_start+=expert_k;
        }
        const int base=expert_start+token_tile*Rows;
        for(int row=thread;row<Rows;row+=blockDim.x) {
            const int route=base+row<expert_start+expert_k?__ldg(peer_pointer<int32_t>(p,Gradient?3:1)+base+row):-1;
            constexpr int64_t stride=int64_t(Tokens)*Topk;
            const bool valid=route>=0 && int64_t(route)<p.dim[4]*stride;
            if(route>=0 && !valid)atomicOr(status,2u);
            const int owner=valid?int64_t(route)/stride:0;
            const int64_t token=valid?(int64_t(route)%stride)/Topk:0;
            addresses[row]=valid?peer_pointer<uint64_t>(p,0)[owner]+token*Width*sizeof(__ppu_bfloat16):0;
            if constexpr(Gradient)row_scores[row]=valid?reinterpret_cast<const float*>(peer_pointer<uint64_t>(p,1)[owner])[int64_t(route)%stride]:0.f;
        }
        __syncthreads();
        uint4 cached[Rows*16/Threads];
        #pragma unroll
        for(int j=0;j<Rows*16/Threads;++j) {
            const int row=thread/16+j*(Threads/16);
            const int64_t at=int64_t(base+row)*Width+feature_base+lane*8;
            uint4 raw{};
            if(addresses[row])raw=*reinterpret_cast<const uint4*>(addresses[row]+(feature_base+lane*8)*sizeof(__ppu_bfloat16));
            if constexpr(Gradient) {
                uint4 raw_y{};
                if(addresses[row])raw_y=*reinterpret_cast<const uint4*>(peer_pointer<__ppu_bfloat16>(p,4)+at);
                const auto* dy=reinterpret_cast<const __ppu_bfloat16*>(&raw);
                const auto* y=reinterpret_cast<const __ppu_bfloat16*>(&raw_y);
                uint4 weighted{};auto* value=reinterpret_cast<__ppu_bfloat16*>(&weighted);float dot=0.f;
                #pragma unroll
                for(int v=0;v<8;++v){const float d=__bfloat162float(dy[v]);value[v]=__float2bfloat16(d*row_scores[row]);dot+=d*__bfloat162float(y[v]);}
                #pragma unroll
                for(int delta=8;delta>0;delta>>=1)dot+=__shfl_down_sync(mask,dot,delta,16);
                if(lane==0 && base+row<expert_start+expert_k)peer_pointer<float>(p,10)[int64_t(base+row)*Segments+feature_base/128]=dot;
                raw=weighted;
            }
            if(base+row<expert_start+expert_k)*reinterpret_cast<uint4*>(peer_pointer<__ppu_bfloat16>(p,Gradient?5:2)+at)=raw;
            float scale;const uint2 q=peer_quantize_eight(raw,scale);
            if(base+row<expert_start+expert_k)*reinterpret_cast<uint2*>(peer_pointer<uint8_t>(p,Gradient?8:5)+at)=q;
            if(lane==0)row_scales[row]=scale;
        }
        __syncthreads();
        if(thread<Rows && base+thread<expert_start+expert_k)peer_pointer<float>(p,Gradient?9:6)[int64_t(feature_base/128)*p.dim[0]+base+thread]=row_scales[thread];
        // Reload the required packed BF16 output after rowwise quantization,
        // so its DW panel does not remain live during peer/score arithmetic.
        #pragma unroll
        for(int j=0;j<Rows*16/Threads;++j) {
            const int row=thread/16+j*(Threads/16);
            const int64_t at=int64_t(base+row)*Width+feature_base+lane*8;
            cached[j]=base+row<expert_start+expert_k?*reinterpret_cast<const uint4*>(
                peer_pointer<__ppu_bfloat16>(p,Gradient?5:2)+at):make_uint4(0,0,0,0);
        }
        mega_moe_dw_emit_block128<Rows,Threads,Gradient,true>(cached,reinterpret_cast<uint8_t*>(dw.quantized0),dw.scales0,
            Width,expert_start,expert_k,base-expert_start,feature_base,tile,maxima,row_scales);
        if constexpr(Gradient) {
            for(int column=feature_base;column<dw.intermediate;column+=Width) {
                #pragma unroll
                for(int j=0;j<Rows*16/Threads;++j) {
                    const int row=thread/16+j*(Threads/16);
                    cached[j]=base+row<expert_start+expert_k?*reinterpret_cast<const uint4*>(reinterpret_cast<const __ppu_bfloat16*>(dw.hidden)+int64_t(base+row)*dw.intermediate+column+lane*8):make_uint4(0,0,0,0);
                }
                mega_moe_dw_emit_block128<Rows,Threads,false>(cached,reinterpret_cast<uint8_t*>(dw.quantized1),dw.scales1,
                    dw.intermediate,expert_start,expert_k,base-expert_start,column,tile,maxima);
            }
        }
    }
}
}
