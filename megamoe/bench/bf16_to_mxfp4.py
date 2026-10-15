"""JIT-compiled BF16 -> MXFP4 input quantizer (packed FP4 + packed UE8M0 scales).

The fused kernel quantizes its bf16 x input in-kernel; the bench reference
quantizes x through this same kernel so the two sides see identical inputs.

Layout:
  - input:  bf16 [num_tokens, hidden], contiguous
  - output x_fp4: packed int8 [num_tokens, hidden//2], contiguous
            i.e. each token occupies hidden//2 bytes
  - output sf: uint8 [num_tokens, hidden//32], contiguous row-major
            i.e. each token occupies hidden//32 bytes

Quantization (per 32-element group):
  scale = ceil_to_ue8m0(amax(abs(group)) / 6.0)
  code = bucketize(abs(x/scale), E2M1 midpoints)  // 0..7
  code |= (sign_bit << 3)                          // negative non-zero -> bit 3
  packed[i] = code[2i] | (code[2i+1] << 4)        // lo=even, hi=odd
"""

import torch

from megamoe.jit import build_runtime

_KERNEL_ARG_DEFS = (
    ('x', torch.bfloat16),
    ('out', torch.int8),
    ('sf', torch.uint8),
    ('num_tokens', int),
    ('hidden', int),
    ('stream', torch.cuda.Stream),
)

_KERNEL_SOURCE_TEMPLATE = """\
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ uint8_t cvt_f4x2_f32x2(float hi, float lo) {{
    uint16_t packed;
    asm volatile ("ppu.cvt.rtte.satfinite.e2m1x2.f32 %0, %1, %2;\\n"
      : "=h"(packed) : "f"(hi), "f"(lo));
    return static_cast<uint8_t>(packed);
}}

constexpr int K_GRANULARITY = 32;
constexpr int THREADS_PER_BLOCK = {threads_per_block};
constexpr int GROUPS_PER_CTA = THREADS_PER_BLOCK;

__global__ __launch_bounds__(THREADS_PER_BLOCK)
void bf16_to_mxfp4_kernel(
    const __nv_bfloat16* __restrict__ x,
    int8_t* __restrict__ out,
    uint8_t* __restrict__ sf,
    int num_tokens,
    int hidden
) {{
    const int num_groups_per_token = hidden / K_GRANULARITY;
    const int total_groups = num_tokens * num_groups_per_token;
    const int global_group_idx = blockIdx.x * GROUPS_PER_CTA + threadIdx.x;
    if (global_group_idx >= total_groups) return;

    const int token_idx = global_group_idx / num_groups_per_token;
    const int group_idx = global_group_idx % num_groups_per_token;

    const int4* x_bf16_src = reinterpret_cast<const int4*>(x + token_idx * hidden + group_idx * K_GRANULARITY);
    int4 quad0 = x_bf16_src[0];
    int4 quad1 = x_bf16_src[1];
    int4 quad2 = x_bf16_src[2];
    int4 quad3 = x_bf16_src[3];

    const __nv_bfloat162* bf16x2_0 = reinterpret_cast<const __nv_bfloat162*>(&quad0);
    const __nv_bfloat162* bf16x2_1 = reinterpret_cast<const __nv_bfloat162*>(&quad1);
    const __nv_bfloat162* bf16x2_2 = reinterpret_cast<const __nv_bfloat162*>(&quad2);
    const __nv_bfloat162* bf16x2_3 = reinterpret_cast<const __nv_bfloat162*>(&quad3);

    float bf16_values[32];
    float amax = 0.0f;

    #pragma unroll
    for (int i = 0; i < 4; i++) {{
        float lo = __bfloat162float(bf16x2_0[i].x);
        float hi = __bfloat162float(bf16x2_0[i].y);
        bf16_values[i*2]   = lo;
        bf16_values[i*2+1] = hi;
        amax = fmaxf(amax, fmaxf(fabsf(lo), fabsf(hi)));
    }}
    #pragma unroll
    for (int i = 0; i < 4; i++) {{
        float lo = __bfloat162float(bf16x2_1[i].x);
        float hi = __bfloat162float(bf16x2_1[i].y);
        bf16_values[8+i*2]   = lo;
        bf16_values[8+i*2+1] = hi;
        amax = fmaxf(amax, fmaxf(fabsf(lo), fabsf(hi)));
    }}
    #pragma unroll
    for (int i = 0; i < 4; i++) {{
        float lo = __bfloat162float(bf16x2_2[i].x);
        float hi = __bfloat162float(bf16x2_2[i].y);
        bf16_values[16+i*2]   = lo;
        bf16_values[16+i*2+1] = hi;
        amax = fmaxf(amax, fmaxf(fabsf(lo), fabsf(hi)));
    }}
    #pragma unroll
    for (int i = 0; i < 4; i++) {{
        float lo = __bfloat162float(bf16x2_3[i].x);
        float hi = __bfloat162float(bf16x2_3[i].y);
        bf16_values[24+i*2]   = lo;
        bf16_values[24+i*2+1] = hi;
        amax = fmaxf(amax, fmaxf(fabsf(lo), fabsf(hi)));
    }}

    uint32_t amax_u = __float_as_uint(fmaxf(amax, 1e-10f));
    uint32_t scale_bits = amax_u - 0x1400000u;
    uint32_t scale_exp = (scale_bits + 0x007FFFFFu) & 0x7F800000u;
    float sf_inv = __uint_as_float((254u << 23) - scale_exp);

    uint32_t packed[4];
    #pragma unroll
    for (int w = 0; w < 4; w++) {{
        uint32_t packed_word = 0;
        #pragma unroll
        for (int p = 0; p < 4; p++) {{
            int base = w * 8 + p * 2;
            float scaled_lo = bf16_values[base]     * sf_inv;
            float scaled_hi = bf16_values[base + 1] * sf_inv;
            uint8_t byte = cvt_f4x2_f32x2(scaled_hi, scaled_lo);
            packed_word |= (static_cast<uint32_t>(byte) << (p * 8));
        }}
        packed[w] = packed_word;
    }}

    int4* dst = reinterpret_cast<int4*>(out + token_idx * hidden + group_idx * 16);
    *dst = *reinterpret_cast<int4*>(packed);

    sf[token_idx * num_groups_per_token + group_idx] =
        static_cast<uint8_t>((scale_exp >> 23) & 0xFFu);
}}

extern "C" void launch(
    void* __raw_x, void* __raw_out, void* __raw_sf,
    int num_tokens, int hidden,
    void* __raw_stream, int& __return_code
) {{
    auto x = reinterpret_cast<__nv_bfloat16*>(__raw_x);
    auto out = reinterpret_cast<int8_t*>(__raw_out);
    auto sf = reinterpret_cast<uint8_t*>(__raw_sf);
    auto stream = reinterpret_cast<cudaStream_t>(__raw_stream);

    int num_groups_per_token = hidden / K_GRANULARITY;
    int total_groups = num_tokens * num_groups_per_token;
    int grid_size = (total_groups + GROUPS_PER_CTA - 1) / GROUPS_PER_CTA;
    if (grid_size > 0)
        bf16_to_mxfp4_kernel<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(
            x, out, sf, num_tokens, hidden);
    __return_code = 0;
}}
"""


def _make_kernel_source(threads_per_block):
    """The quantizer source for one block size; the template's only field is the block size."""
    return _KERNEL_SOURCE_TEMPLATE.format(threads_per_block=threads_per_block)


_runtime_cache = {}


def _get_runtime(name, source_fn, *args):
    if name not in _runtime_cache:
        src = source_fn(*args)
        runtime = build_runtime(name, _KERNEL_ARG_DEFS, src)
        _runtime_cache[name] = runtime
    return _runtime_cache[name]


def get_bf16_to_mxfp4_runtime(threads_per_block: int = 256):
    """Build (or fetch from cache) the quantizer runtime; building it is also the prewarm."""
    return _get_runtime(f'bf16_to_mxfp4_t{threads_per_block}', _make_kernel_source, threads_per_block)


def bf16_to_mxfp4_cuda(x: torch.Tensor, out: torch.Tensor = None, sf: torch.Tensor = None, threads_per_block: int = 256):
    """Launch BF16->MXFP4 CUDA kernel (JIT compiled, Megamoe layout)."""
    assert x.dim() == 2 and x.dtype == torch.bfloat16 and x.is_contiguous()
    num_tokens, hidden = x.shape
    assert hidden % 32 == 0

    num_groups_per_token = hidden // 32
    if out is None:
        out = torch.empty((num_tokens, hidden), dtype=torch.int8, device=x.device)
    if sf is None:
        sf = torch.empty((num_tokens, num_groups_per_token), dtype=torch.uint8, device=x.device)

    runtime = get_bf16_to_mxfp4_runtime(threads_per_block)
    stream = torch.cuda.current_stream()
    runtime(x, out, sf, num_tokens, hidden, stream)
    return out[:, :hidden // 2].contiguous(), sf
