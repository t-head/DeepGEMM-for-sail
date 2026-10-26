#pragma once
#include <deep_gemm/layout/mega_moe_types.cuh>

#include <stdint.h>
#include <hggc_fp8.h>
#include <hggc_bf16.h>
#include <deep_gemm/impls/fp4_gemm_cutlass3.cuh>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <ppu_include.hpp>
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>

// MoE routing and FP8 packing for the PPU symmetric combine buffer.
namespace deep_gemm::mega_moe::detail {

template <uint32_t kMaxRanks = 64>
struct SymBufferLite {
    int64_t base;
    int64_t offsets[kMaxRanks];
    uint32_t rank_idx;
    uint32_t num_ranks;
    SymBufferLite() : base(0), rank_idx(0), num_ranks(0) {
        for (auto& o : offsets) o = 0;
    }
    template <typename T>
    __host__ __device__ __forceinline__ T* map(T* ptr, uint32_t dst) const {
        return reinterpret_cast<T*>(offsets[dst] + reinterpret_cast<int64_t>(ptr));
    }
};

struct CombineBuffer {
    void* base; uint32_t hidden_bytes, max_tokens_per_rank, num_topk;
    uint64_t* scales; uint32_t scale_tiles, scale_block_n, use_side_scale;
    int64_t scale_offsets[64];
    CombineBuffer()
        : base(nullptr), hidden_bytes(0), max_tokens_per_rank(0), num_topk(0),
          scales(nullptr), scale_tiles(0), scale_block_n(0), use_side_scale(0) {
        for (auto& o : scale_offsets) o = 0;
    }
    CombineBuffer(void* b, uint32_t hb, uint32_t mt)
        : CombineBuffer(b, hb, mt, 0) {}
    CombineBuffer(void* b, uint32_t hb, uint32_t mt, uint32_t nt)
        : base(b), hidden_bytes(hb), max_tokens_per_rank(mt), num_topk(nt),
          scales(nullptr), scale_tiles(0), scale_block_n(0), use_side_scale(0) {
        for (auto& o : scale_offsets) o = 0;
    }
    CombineBuffer(void* b, uint32_t hb, uint32_t mt, uint32_t nt,
                  uint64_t* sc, uint32_t st, uint32_t sbn, uint32_t use_scale)
        : base(b), hidden_bytes(hb), max_tokens_per_rank(mt), num_topk(nt),
          scales(sc), scale_tiles(st), scale_block_n(sbn), use_side_scale(use_scale) {
        for (auto& o : scale_offsets) o = 0;
    }
    __host__ __device__ __forceinline__
    uint8_t* get_bytes(uint32_t slot, uint32_t tok) const {
        return static_cast<uint8_t*>(base)
            + static_cast<uint64_t>(slot) * max_tokens_per_rank * hidden_bytes
            + static_cast<uint64_t>(tok) * hidden_bytes;
    }
    __host__ __device__ __forceinline__
    __ppu_bfloat16* get(uint32_t slot, uint32_t tok) const {
        return reinterpret_cast<__ppu_bfloat16*>(get_bytes(slot, tok));
    }
    __host__ __device__ __forceinline__
    uint64_t* get_scale(uint32_t slot, uint32_t tok, uint32_t n_start) const {
        uint32_t tile = scale_block_n == 0 ? 0 : n_start / scale_block_n;
        return scales + (static_cast<uint64_t>(slot) * max_tokens_per_rank + tok) * scale_tiles + tile;
    }
    __device__ __forceinline__
    void store_scale(uint32_t self_rank, uint32_t dst_rank, uint32_t slot, uint32_t tok,
                     uint32_t n_start, uint16_t raw_scale) const {
        if (use_side_scale == 0 || scales == nullptr || scale_tiles == 0) return;
        uint64_t* local_ptr = get_scale(slot, tok, n_start);
        uint64_t* dst_ptr = reinterpret_cast<uint64_t*>(
            scale_offsets[dst_rank] + reinterpret_cast<int64_t>(local_ptr));
#if __HGGC_ARCH__ == 150
        uint32_t payload = static_cast<uint32_t>(raw_scale);
        uint32_t* dst_ptr32 = reinterpret_cast<uint32_t*>(dst_ptr);
        if (dst_rank != self_rank) {
            __ppu_remote_store_bulk_ncp_b32((void*)dst_ptr32, payload);
        } else {
            __ppu_global_store_bulk_b32((void*)dst_ptr32, payload);
        }
#else
        *reinterpret_cast<uint32_t*>(dst_ptr) = static_cast<uint32_t>(raw_scale);
#endif
    }
};


__device__ __forceinline__ uint2 bf16x8_bits_to_e4m3x8(uint4 vals, int exp_shift);
__device__ __forceinline__ uint2 bf16x8_bits_to_e4m3x8_scaled(uint4 vals, float inv_scale);
__device__ __forceinline__ uint2 bf16x8_bits_to_e4m3x8_scaled_guarded(
    uint4 vals, const uint16_t* raw_scales, const float* inv_scales, uint32_t scale_idx, uint32_t side_scale_mode);

__device__ __forceinline__ float bf16_bits_to_float(uint16_t h) {
    return __uint_as_float(static_cast<uint32_t>(h) << 16);
}


__device__ __forceinline__ uint16_t float_to_bf16_bits(float x) {
    uint32_t bits = __float_as_uint(x);
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

__device__ __forceinline__ float bf16x8_absmax(uint4 vals) {
    uint32_t xs[4] = {vals.x, vals.y, vals.z, vals.w};
    float m = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint16_t lo = static_cast<uint16_t>(xs[i] & 0xffffu);
        uint16_t hi = static_cast<uint16_t>((xs[i] >> 16) & 0xffffu);
        m = fmaxf(m, fabsf(bf16_bits_to_float(lo)));
        m = fmaxf(m, fabsf(bf16_bits_to_float(hi)));
    }
    return m;
}

__device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_down_sync(0xffffffffu, x, offset));
    }
    return x;
}

__device__ __forceinline__ uint4 scatter_load_bf16x8(const uint4* ptr) {
#if __HGGC_ARCH__ == 150
    return __ppu_global_load_bulk_b32x4((const void*)ptr);
#else
    return *ptr;
#endif
}

__device__ __forceinline__ uint4 scatter_load_bf16x8(const float4* ptr) {
#if __HGGC_ARCH__ == 150
    return __ppu_global_load_bulk_b32x4((const void*)ptr);
#else
    return *reinterpret_cast<const uint4*>(ptr);
#endif
}

// FP8 packing reads individual vector members. On PPU, keep this as an
// ordinary vector load because the bulk-load wrapper only materializes .x.
__device__ __forceinline__ uint4 scatter_load_bf16x8_for_fp8(const uint4* ptr) {
    return *ptr;
}

// Keep the FP8 payload as one 16-byte store without passing a uint4 through
// the PPU bulk-intrinsic ABI. Explicit scalar operands preserve all four
// words and force the backend to allocate one contiguous v4 register tuple.
__device__ __forceinline__ void scatter_store_fp8x16(
    uint4* ptr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
#if __HGGC_ARCH__ == 150
    asm volatile(
        "ppu.st.global.v4.b32 [%0], {%1, %2, %3, %4};"
        :: "l"(ptr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
#else
    *ptr = make_uint4(x, y, z, w);
#endif
}

__device__ __forceinline__ int fp8_shift_from_absmax(float absmax) {
    constexpr float kMaxE4M3 = 448.0f;
    if (!(absmax > kMaxE4M3)) return 0;
    int shift = static_cast<int>(ceilf(log2f(absmax / kMaxE4M3)));
    return shift < 0 ? 0 : (shift > 15 ? 15 : shift);
}

__device__ __forceinline__ uint16_t fp8_scale_raw_from_shift(int shift) {
    shift = shift < 0 ? 0 : (shift > 15 ? 15 : shift);
    return static_cast<uint16_t>((127 + shift) << 7);
}


__device__ __forceinline__ int fp8_dynamic_row_shift(
    const uint4* __restrict__ src_base, uint32_t n_vecs, uint32_t lane, int fallback_shift) {
    float local_absmax = 0.0f;
#pragma unroll
    for (uint32_t v = lane; v < n_vecs; v += 32) {
#if __HGGC_ARCH__ == 150
        uint4 vals = scatter_load_bf16x8_for_fp8(src_base + v);
#else
        uint4 vals = scatter_load_bf16x8_for_fp8(src_base + v);
#endif
        local_absmax = fmaxf(local_absmax, bf16x8_absmax(vals));
    }
    float row_absmax = warp_reduce_max(local_absmax);
    int row_shift = fallback_shift;
    if (lane == 0) row_shift = fp8_shift_from_absmax(row_absmax);
    return __shfl_sync(0xffffffffu, row_shift, 0);
}


__device__ __forceinline__ void fp8_scale_raw_and_inv_from_absmax(
    float absmax, uint16_t* raw_scale, float* inv_scale) {
    // The sideband path scans D before packing. If that scan observes a stale
    // non-finite or extreme value, publish a benign fixed-scale side scale and
    // zero the corresponding FP8 payload chunk instead of emitting saturated
    // bytes that would dominate the reduce.
    if (!isfinite(absmax) || !(absmax > 0.0f) || absmax > 16.0f) {
        *raw_scale = fp8_scale_raw_from_shift(0);
        *inv_scale = 0.0f;
        return;
    }
    float scale = fmaxf(absmax * (1.0f / 448.0f), 0x1.0p-24f);
    *raw_scale = float_to_bf16_bits(scale);
    *inv_scale = 1.0f / bf16_bits_to_float(*raw_scale);
}

__device__ __forceinline__ uint16_t fp8_scale_raw_from_absmax(float absmax) {
    uint16_t raw_scale;
    float inv_scale;
    fp8_scale_raw_and_inv_from_absmax(absmax, &raw_scale, &inv_scale);
    return raw_scale;
}

__device__ __forceinline__ uint16_t fp8_dynamic_row_scale_raw(
    const uint4* __restrict__ src_base, uint32_t n_vecs, uint32_t lane) {
    float local_absmax = 0.0f;
#pragma unroll
    for (uint32_t v = lane; v < n_vecs; v += 32) {
#if __HGGC_ARCH__ == 150
        uint4 vals = scatter_load_bf16x8_for_fp8(src_base + v);
#else
        uint4 vals = scatter_load_bf16x8_for_fp8(src_base + v);
#endif
        local_absmax = fmaxf(local_absmax, bf16x8_absmax(vals));
    }
    float row_absmax = warp_reduce_max(local_absmax);
    uint32_t raw = fp8_scale_raw_from_shift(0);
    if (lane == 0) raw = static_cast<uint32_t>(fp8_scale_raw_from_absmax(row_absmax));
    return static_cast<uint16_t>(__shfl_sync(0xffffffffu, raw, 0));
}


template <uint32_t kBlockN>
__device__ __forceinline__ uint32_t fp8_dynamic_tile_scales(
    const uint4* __restrict__ src_base, uint32_t lane, const CombineBuffer& combine_buf,
    uint16_t* raw_scales, float* inv_scales, uint32_t* actual_scale_block_n) {
    constexpr uint32_t kVec = 8;
    constexpr uint32_t n_vecs = kBlockN / kVec;
    // Validated against the selected tile before launch. Never change the grouping here.
    const uint32_t scale_block_n = combine_buf.scale_block_n;
    *actual_scale_block_n = scale_block_n;
    const uint32_t chunks = kBlockN / scale_block_n;

#pragma unroll
    for (uint32_t c = 0; c < 4; ++c) {
        if (c < chunks) {
            uint32_t vec_begin = (c * scale_block_n) / kVec;
            uint32_t vec_end = ((c + 1) * scale_block_n) / kVec;
            if (vec_end > n_vecs) vec_end = n_vecs;
            float local_absmax = 0.0f;
#pragma unroll
            for (uint32_t v = vec_begin + lane; v < vec_end; v += 32) {
#if __HGGC_ARCH__ == 150
                uint4 vals = scatter_load_bf16x8_for_fp8(src_base + v);
#else
                uint4 vals = scatter_load_bf16x8_for_fp8(src_base + v);
#endif
                local_absmax = fmaxf(local_absmax, bf16x8_absmax(vals));
            }
            float row_absmax = warp_reduce_max(local_absmax);
            uint32_t raw = fp8_scale_raw_from_shift(0);
            float inv_scale = 0.0f;
            if (lane == 0) {
                uint16_t raw_scale;
                fp8_scale_raw_and_inv_from_absmax(row_absmax, &raw_scale, &inv_scale);
                raw = static_cast<uint32_t>(raw_scale);
            }
            raw = __shfl_sync(0xffffffffu, raw, 0);
            uint32_t inv_bits = __shfl_sync(0xffffffffu, __float_as_uint(inv_scale), 0);
            raw_scales[c] = static_cast<uint16_t>(raw);
            inv_scales[c] = __uint_as_float(inv_bits);
        }
    }
    return chunks;
}

template <uint32_t kBlockN>
__device__ __forceinline__ uint32_t fp8_static_tile_scales(
    const CombineBuffer& combine_buf, uint16_t* raw_scales, float* inv_scales,
    uint32_t* actual_scale_block_n, int exp_shift) {
    // Validated against the selected tile before launch. Never change the grouping here.
    const uint32_t scale_block_n = combine_buf.scale_block_n;
    *actual_scale_block_n = scale_block_n;
    const uint32_t chunks = kBlockN / scale_block_n;
#pragma unroll
    for (uint32_t c = 0; c < 4; ++c) {
        if (c < chunks) {
            raw_scales[c] = fp8_scale_raw_from_shift(exp_shift);
            inv_scales[c] = 1.0f;
        }
    }
    return chunks;
}

template <Fp8ScaleMode kSideScaleMode>
__device__ __forceinline__ uint2
scatter_pack_fp8(
    uint4 vals, const uint16_t* raw_scales, const float* inv_scales,
    uint32_t scale_idx, int exp_shift);

template <uint32_t kBlockN, Fp8ScaleMode kSideScaleMode>
__device__ __forceinline__ void scatter_pack_store_fp8_row(
    const uint4* src_base, uint4* dst_base, uint32_t lane,
    const uint16_t* raw_scales, const float* inv_scales,
    uint32_t scale_block_n, int exp_shift, bool is_remote);

// Packed route: rank in bits [63:32], top-k slot in [31:24], token in [23:0].
// Pool-row packed scatter that transfers E4M3 bytes.
// The packed metadata is trusted for valid rows, matching the expert-major fast path.
template <uint32_t kBlockM, uint32_t kBlockN, uint32_t kThreadNum,
          Fp8ScaleMode kSideScaleMode, uint32_t kMaxRanks = 64>
__device__ __forceinline__ void
scatter_fp8(
    const __ppu_bfloat16* __restrict__ tile_d,
    uint32_t valid_m, uint32_t n_start, uint32_t shape_n,
    uint32_t pool_row_base,
    const int64_t* __restrict__ token_meta,
    const CombineBuffer& combine_buf,
    const SymBufferLite<kMaxRanks>& sym_buffer,
    int exp_shift)
{
    constexpr uint32_t kVec = 16;
    const uint32_t tid = threadIdx.x;
    constexpr uint32_t n_vecs = kBlockN / kVec;

    for (uint32_t row = tid / 32; row < valid_m; row += kThreadNum / 32) {
        uint32_t lane = tid % 32;
        uint32_t pool_row = pool_row_base + row;
        const uint64_t packed = __ldg(token_meta + pool_row);
        const uint32_t rank_idx = packed >> 32;
        const uint32_t token_idx = packed & 0x00FFFFFFu;
        const uint32_t topk_idx = (packed >> 24) & 0xFFu;

        uint4* dst_base = reinterpret_cast<uint4*>(
            sym_buffer.offsets[rank_idx] +
            reinterpret_cast<int64_t>(combine_buf.get_bytes(topk_idx, token_idx) + n_start));
        const uint4* src_base = reinterpret_cast<const uint4*>(tile_d + row * shape_n + n_start);
        int row_exp_shift = exp_shift;
        uint16_t row_scale_raws[4];
        float row_inv_scales[4];
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            row_scale_raws[i] = fp8_scale_raw_from_shift(exp_shift);
            row_inv_scales[i] = 1.0f;
        }
        uint32_t row_scale_block_n = kBlockN;
        uint32_t row_scale_chunks = 1;
        if constexpr (kSideScaleMode != Fp8ScaleMode::Fixed) {
            if constexpr (kSideScaleMode == Fp8ScaleMode::SideDynamic) {
                row_scale_chunks = fp8_dynamic_tile_scales<kBlockN>(
                    src_base, lane, combine_buf, row_scale_raws, row_inv_scales, &row_scale_block_n);
            } else {
                row_scale_chunks = fp8_static_tile_scales<kBlockN>(
                    combine_buf, row_scale_raws, row_inv_scales, &row_scale_block_n, exp_shift);
            }
            if (lane == 0) {
#pragma unroll
                for (uint32_t c = 0; c < 4; ++c) {
                    if (c < row_scale_chunks) {
                        combine_buf.store_scale(
                            sym_buffer.rank_idx, rank_idx, topk_idx, token_idx,
                            n_start + c * row_scale_block_n, row_scale_raws[c]);
                    }
                }
            }
        }

#if __HGGC_ARCH__ == 150
        bool is_remote = (rank_idx != sym_buffer.rank_idx);
        scatter_pack_store_fp8_row<kBlockN, kSideScaleMode>(
            src_base, dst_base, lane, row_scale_raws, row_inv_scales,
            row_scale_block_n, row_exp_shift, is_remote);
#else
        #pragma unroll
        for (uint32_t v = lane; v < n_vecs; v += 32) {
            uint4 vals0 = scatter_load_bf16x8_for_fp8(src_base + v * 2);
            uint4 vals1 = scatter_load_bf16x8_for_fp8(src_base + v * 2 + 1);
            uint32_t scale_idx0 = ((v * kVec) / row_scale_block_n);
            uint32_t scale_idx1 = ((v * kVec + 8) / row_scale_block_n);
            uint2 packed0 = scatter_pack_fp8<kSideScaleMode>(vals0, row_scale_raws, row_inv_scales, scale_idx0, row_exp_shift);
            uint2 packed1 = scatter_pack_fp8<kSideScaleMode>(vals1, row_scale_raws, row_inv_scales, scale_idx1, row_exp_shift);
            scatter_store_fp8x16(dst_base + v, packed0.x, packed0.y, packed1.x, packed1.y);
        }
#endif
    }
}

__device__ __forceinline__ uint8_t bf16_bits_to_e4m3_finite(uint16_t h, int exp_shift) {
    uint32_t sign = static_cast<uint32_t>(h >> 15);
    uint32_t exp = static_cast<uint32_t>((h >> 7) & 0xffu);
    uint32_t mant = static_cast<uint32_t>(h & 0x7fu);
    if (exp == 0) return static_cast<uint8_t>(sign << 7);

    int e = static_cast<int>(exp) - 127 - exp_shift;
    int fp8_exp = e + 7;
    uint32_t fp8_mant = 0;

    if (fp8_exp <= 0) {
        int shift = 1 - fp8_exp;
        uint32_t sig = 128u + mant;
        uint32_t rounded = (shift >= 8) ? 0u : ((sig + (1u << (shift + 3))) >> (shift + 4));
        fp8_exp = 0;
        fp8_mant = rounded > 7u ? 7u : rounded;
    } else {
        fp8_mant = (mant + 8u) >> 4;
        if (fp8_mant == 8u) {
            fp8_mant = 0;
            fp8_exp += 1;
        }
        if (fp8_exp > 15 || (fp8_exp == 15 && fp8_mant > 6u)) {
            fp8_exp = 15;
            fp8_mant = 6u;
        }
    }
    return static_cast<uint8_t>((sign << 7) | (static_cast<uint32_t>(fp8_exp) << 3) | fp8_mant);
}

__device__ __forceinline__ uint16_t bf16x2_bits_to_e4m3x2_native(uint32_t bf16x2) {
    __ppu_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(bf16x2 & 0xffffu);
    raw.y = static_cast<unsigned short>((bf16x2 >> 16) & 0xffffu);
    return static_cast<uint16_t>(__hg_cvt_bfloat16raw2_to_fp8x2(raw, __HG_SATFINITE, __HG_E4M3));
}

__device__ __forceinline__ uint16_t bf16x2_bits_to_e4m3x2_shifted(uint32_t bf16x2, int exp_shift) {
    uint8_t lo = bf16_bits_to_e4m3_finite(static_cast<uint16_t>(bf16x2 & 0xffffu), exp_shift);
    uint8_t hi = bf16_bits_to_e4m3_finite(static_cast<uint16_t>((bf16x2 >> 16) & 0xffffu), exp_shift);
    return static_cast<uint16_t>(lo) | (static_cast<uint16_t>(hi) << 8);
}

__device__ __forceinline__ uint2 bf16x8_bits_to_e4m3x8_native(uint4 vals) {
    uint16_t p0 = bf16x2_bits_to_e4m3x2_native(vals.x);
    uint16_t p1 = bf16x2_bits_to_e4m3x2_native(vals.y);
    uint16_t p2 = bf16x2_bits_to_e4m3x2_native(vals.z);
    uint16_t p3 = bf16x2_bits_to_e4m3x2_native(vals.w);
    return make_uint2(static_cast<uint32_t>(p0) | (static_cast<uint32_t>(p1) << 16),
                      static_cast<uint32_t>(p2) | (static_cast<uint32_t>(p3) << 16));
}

__device__ __forceinline__ uint2 bf16x8_bits_to_e4m3x8_shifted(uint4 vals, int exp_shift) {
    uint16_t p0 = bf16x2_bits_to_e4m3x2_shifted(vals.x, exp_shift);
    uint16_t p1 = bf16x2_bits_to_e4m3x2_shifted(vals.y, exp_shift);
    uint16_t p2 = bf16x2_bits_to_e4m3x2_shifted(vals.z, exp_shift);
    uint16_t p3 = bf16x2_bits_to_e4m3x2_shifted(vals.w, exp_shift);
    return make_uint2(static_cast<uint32_t>(p0) | (static_cast<uint32_t>(p1) << 16),
                      static_cast<uint32_t>(p2) | (static_cast<uint32_t>(p3) << 16));
}

__device__ __forceinline__ uint2 bf16x8_bits_to_e4m3x8(uint4 vals, int exp_shift) {
    return exp_shift == 0 ? bf16x8_bits_to_e4m3x8_native(vals)
                          : bf16x8_bits_to_e4m3x8_shifted(vals, exp_shift);
}


__device__ __forceinline__ uint16_t bf16x2_bits_to_e4m3x2_scaled(uint32_t bf16x2, float inv_scale) {
    uint16_t lo = static_cast<uint16_t>(bf16x2 & 0xffffu);
    uint16_t hi = static_cast<uint16_t>((bf16x2 >> 16) & 0xffffu);
    __ppu_bfloat162_raw raw;
    raw.x = float_to_bf16_bits(bf16_bits_to_float(lo) * inv_scale);
    raw.y = float_to_bf16_bits(bf16_bits_to_float(hi) * inv_scale);
    return static_cast<uint16_t>(__hg_cvt_bfloat16raw2_to_fp8x2(raw, __HG_SATFINITE, __HG_E4M3));
}

__device__ __forceinline__ uint2 bf16x8_bits_to_e4m3x8_scaled(uint4 vals, float inv_scale) {
    if (inv_scale == 0.0f) return make_uint2(0u, 0u);
    uint16_t p0 = bf16x2_bits_to_e4m3x2_scaled(vals.x, inv_scale);
    uint16_t p1 = bf16x2_bits_to_e4m3x2_scaled(vals.y, inv_scale);
    uint16_t p2 = bf16x2_bits_to_e4m3x2_scaled(vals.z, inv_scale);
    uint16_t p3 = bf16x2_bits_to_e4m3x2_scaled(vals.w, inv_scale);
    return make_uint2(static_cast<uint32_t>(p0) | (static_cast<uint32_t>(p1) << 16),
                      static_cast<uint32_t>(p2) | (static_cast<uint32_t>(p3) << 16));
}

template <Fp8ScaleMode kSideScaleMode>
__device__ __forceinline__ uint2 bf16x8_bits_to_e4m3x8_scaled_guarded(
    uint4 vals, const uint16_t* raw_scales, const float* inv_scales, uint32_t scale_idx) {
    if constexpr (kSideScaleMode == Fp8ScaleMode::SideDynamic) {
        if (raw_scales[scale_idx] == fp8_scale_raw_from_shift(0)) {
            return make_uint2(0u, 0u);
        }
    }
    return bf16x8_bits_to_e4m3x8_scaled(vals, inv_scales[scale_idx]);
}

template <Fp8ScaleMode kSideScaleMode>
__device__ __forceinline__ uint2 scatter_pack_fp8(
    uint4 vals, const uint16_t* raw_scales, const float* inv_scales,
    uint32_t scale_idx, int exp_shift) {
    if constexpr (kSideScaleMode == Fp8ScaleMode::SideStatic) {
        return bf16x8_bits_to_e4m3x8_native(vals);
    } else if constexpr (kSideScaleMode == Fp8ScaleMode::SideDynamic) {
        return bf16x8_bits_to_e4m3x8_scaled_guarded<kSideScaleMode>(
            vals, raw_scales, inv_scales, scale_idx);
    } else {
        return bf16x8_bits_to_e4m3x8(vals, exp_shift);
    }
}

template <Fp8ScaleMode kSideScaleMode>
__device__ __forceinline__ uint32_t scatter_pack_fp8x4(
    uint2 vals, const uint16_t* raw_scales, const float* inv_scales,
    uint32_t scale_idx, int exp_shift) {
    uint16_t p0, p1;
    if constexpr (kSideScaleMode == Fp8ScaleMode::SideStatic) {
        p0 = bf16x2_bits_to_e4m3x2_native(vals.x);
        p1 = bf16x2_bits_to_e4m3x2_native(vals.y);
    } else if constexpr (kSideScaleMode == Fp8ScaleMode::SideDynamic) {
        if (raw_scales[scale_idx] == fp8_scale_raw_from_shift(0)) return 0u;
        float inv_scale = inv_scales[scale_idx];
        if (inv_scale == 0.0f) return 0u;
        p0 = bf16x2_bits_to_e4m3x2_scaled(vals.x, inv_scale);
        p1 = bf16x2_bits_to_e4m3x2_scaled(vals.y, inv_scale);
    } else if (exp_shift == 0) {
        p0 = bf16x2_bits_to_e4m3x2_native(vals.x);
        p1 = bf16x2_bits_to_e4m3x2_native(vals.y);
    } else {
        p0 = bf16x2_bits_to_e4m3x2_shifted(vals.x, exp_shift);
        p1 = bf16x2_bits_to_e4m3x2_shifted(vals.y, exp_shift);
    }
    return static_cast<uint32_t>(p0) | (static_cast<uint32_t>(p1) << 16);
}

// A PPU bulk b32x2 store writes register-major data across all 32 lanes:
// first every lane's .x, then every lane's .y. Feed it two 128-byte stripes
// so one instruction emits a correctly ordered 256-byte FP8 tile.
template <uint32_t kBlockN, Fp8ScaleMode kSideScaleMode>
__device__ __forceinline__ void scatter_pack_store_fp8_row(
    const uint4* src_base, uint4* dst_base, uint32_t lane,
    const uint16_t* raw_scales, const float* inv_scales,
    uint32_t scale_block_n, int exp_shift, bool is_remote) {
#if __HGGC_ARCH__ == 150
    if constexpr (kBlockN == 256) {
        const uint2* src4 = reinterpret_cast<const uint2*>(src_base);
        uint2* dst8 = reinterpret_cast<uint2*>(dst_base);
        uint32_t n0 = lane * 4;
        uint32_t n1 = n0 + 128;
        uint2 vals0 = *(src4 + lane);
        uint2 vals1 = *(src4 + lane + 32);
        uint32_t scale_idx0 = (n0 / scale_block_n);
        uint32_t scale_idx1 = (n1 / scale_block_n);
        uint2 payload = make_uint2(
            scatter_pack_fp8x4<kSideScaleMode>(
                vals0, raw_scales, inv_scales, scale_idx0, exp_shift),
            scatter_pack_fp8x4<kSideScaleMode>(
                vals1, raw_scales, inv_scales, scale_idx1, exp_shift));
        if (is_remote) {
            __ppu_remote_store_bulk_ncp_b32x2((void*)(dst8 + lane), payload);
        } else {
            __ppu_global_store_bulk_b32x2((void*)(dst8 + lane), payload);
        }
    } else {
        constexpr uint32_t kVec = 16;
        constexpr uint32_t n_vecs = kBlockN / kVec;
#pragma unroll
        for (uint32_t v = lane; v < n_vecs; v += 32) {
            uint4 vals0 = scatter_load_bf16x8_for_fp8(src_base + v * 2);
            uint4 vals1 = scatter_load_bf16x8_for_fp8(src_base + v * 2 + 1);
            uint32_t scale_idx0 = ((v * kVec) / scale_block_n);
            uint32_t scale_idx1 = ((v * kVec + 8) / scale_block_n);
            uint2 packed0 = scatter_pack_fp8<kSideScaleMode>(
                vals0, raw_scales, inv_scales, scale_idx0, exp_shift);
            uint2 packed1 = scatter_pack_fp8<kSideScaleMode>(
                vals1, raw_scales, inv_scales, scale_idx1, exp_shift);
            scatter_store_fp8x16(
                dst_base + v, packed0.x, packed0.y, packed1.x, packed1.y);
        }
    }
#else
    constexpr uint32_t kVec = 16;
    constexpr uint32_t n_vecs = kBlockN / kVec;
#pragma unroll
    for (uint32_t v = lane; v < n_vecs; v += 32) {
        uint4 vals0 = scatter_load_bf16x8_for_fp8(src_base + v * 2);
        uint4 vals1 = scatter_load_bf16x8_for_fp8(src_base + v * 2 + 1);
        uint32_t scale_idx0 = ((v * kVec) / scale_block_n);
        uint32_t scale_idx1 = ((v * kVec + 8) / scale_block_n);
        uint2 packed0 = scatter_pack_fp8<kSideScaleMode>(
            vals0, raw_scales, inv_scales, scale_idx0, exp_shift);
        uint2 packed1 = scatter_pack_fp8<kSideScaleMode>(
            vals1, raw_scales, inv_scales, scale_idx1, exp_shift);
        scatter_store_fp8x16(
            dst_base + v, packed0.x, packed0.y, packed1.x, packed1.y);
    }
#endif
}

} // namespace deep_gemm::mega_moe::detail


namespace deep_gemm::mega_moe::fp4 {
// Only routing and fused epilogue behavior live here. GEMM, SF loads, replicas,
// and the persistent tile loop belong to the mainline FP4 universal kernel.
template <bool kFp8Output = false, mega_moe::Fp8ScaleMode kScaleMode = mega_moe::Fp8ScaleMode::Fixed,
          int kFp8ExpShift = 0>
struct ScatterPolicy {
  static constexpr bool kEnabled = true;
  template <typename TileShape> struct SharedStorage {
    static constexpr int kRowStridePairs =
        CUTE_STATIC_V(cute::size<1>(TileShape{})) / 2 + mega_moe::kFp4ScatterSmemPadPairs;
    uint32_t transpose[kFp8Output ? 1 : CUTE_STATIC_V(cute::size<0>(TileShape{})) * kRowStridePairs];
  };
  struct State {
    mega_moe::detail::CombineBuffer moe_combine_buf{};
    mega_moe::detail::SymBufferLite<> moe_sym_buffer{};
    const int64_t* moe_token_meta{nullptr};
    const int32_t* moe_valid_m_rows{nullptr};
    const int32_t* moe_src_m_offsets{nullptr};
  };
  template <typename Base> struct Arguments : Base, State {};
  template <typename Base> struct Params : Base, State {};

  template <typename Output, typename Input>
  static void copy_arguments(Output& out, Input const& in) {
    static_cast<State&>(out) = static_cast<State const&>(in);
  }
  template <typename P, typename Scheduler>
  CUTLASS_DEVICE static uint32_t problem_m(P const& p, Scheduler const& scheduler, uint32_t m) {
    if (p.moe_valid_m_rows != nullptr)
      return static_cast<uint32_t>(__ldg(p.moe_valid_m_rows + scheduler.curr_group_idx));
    return m;
  }
  template <typename P, typename Scheduler, typename OffsetA, typename OffsetSF>
  CUTLASS_DEVICE static void input_offsets(P const& p, Scheduler const& scheduler, OffsetA& a, OffsetSF& sf) {
    if (p.moe_src_m_offsets != nullptr) {
      const int64_t row = __ldg(p.moe_src_m_offsets + scheduler.curr_group_idx);
      a = row * int64_t(Scheduler::SHAPE_K);
      sf = row;
    }
  }
  template <typename Kernel, typename P, typename Scheduler, typename ProblemShape,
            typename Accumulators, typename Mma, typename Residue, typename Storage>
  CUTLASS_DEVICE static void epilogue(P const& params, Scheduler const& deep_scheduler,
      uint32_t m_block_idx, uint32_t n_block_idx, ProblemShape const& problem_shape_MNKL,
      Accumulators& accumulators, Mma const& tiled_mma, Residue const& residue_mnk,
      int thread_idx, char* smem_buf, Storage& shared_storage) {
    using namespace cute;
    using TileShape = typename Kernel::TileShape;
    using CollectiveEpilogue = typename Kernel::CollectiveEpilogue;
    constexpr bool hasBias_ = Kernel::kHasBias;
    constexpr int MaxThreadsPerBlock = Kernel::MaxThreadsPerBlock;
    constexpr int SharedStorageSize = Kernel::SharedStorageSize;
    const auto N = get<1>(problem_shape_MNKL);
    const auto m_coord = m_block_idx, n_coord = n_block_idx;
    const int l_coord = 0;
    auto blk_shape = TileShape{};
      constexpr int kTileM_ = CUTE_STATIC_V(get<0>(TileShape{}));
      constexpr int kTileN_ = CUTE_STATIC_V(get<1>(TileShape{}));

      uint32_t valid_m_sc = 0;
      uint32_t route_row_base = 0;
      uint32_t n_start_sc = n_block_idx * kTileN_;
      if (params.moe_token_meta != nullptr) {
        uint32_t group_valid_m_sc = params.moe_valid_m_rows != nullptr
            ? static_cast<uint32_t>(__ldg(
                  params.moe_valid_m_rows + deep_scheduler.curr_group_idx))
            : static_cast<uint32_t>(deep_scheduler.curr_group_m);
        uint32_t row_start_sc = m_block_idx * kTileM_;
        uint32_t remaining_m_sc =
            group_valid_m_sc > row_start_sc ? group_valid_m_sc - row_start_sc : 0;
        valid_m_sc = cute::min(static_cast<uint32_t>(kTileM_), remaining_m_sc);
        // Packed routes follow GEMM2 padding; A/SFA keep their source offsets.
        route_row_base = deep_scheduler.curr_cumsum_m + m_block_idx * kTileM_;
      }

      // Update the pointer to C and D of MXFP4 in the epilogue parameters
      auto params_epilogue_local = params.epilogue;
      auto epilogue_m_coord = m_coord;
      params_epilogue_local.ptr_D += deep_scheduler.curr_offset_c();
      if constexpr (hasBias_) {
        params_epilogue_local.ptr_Bias += deep_scheduler.curr_offset_mxfp4_c();
      }

      if constexpr (!kFp8Output) {
        static_assert(!hasBias_,
            "direct fragment scatter does not support a bias epilogue");
        static_assert(CollectiveEpilogue::kIsAlignedN,
            "direct fragment scatter requires an aligned N tile");

        if (params.moe_token_meta != nullptr) {
          using ThreadEpilogueOp =
              typename CollectiveEpilogue::ThreadEpilogueOp;
          using FragmentAccumulator =
              typename CollectiveEpilogue::FragmentAccumulator;
          using VectorType = typename CollectiveEpilogue::VectorType;
          static_assert(sizeof(VectorType) == sizeof(uint32_t),
              "direct fragment scatter expects one BF16 pair per fragment");

          auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
          auto cD = make_identity_tensor(
              make_shape(Int<kTileM_>{}, Int<kTileN_>{}));
          auto tCcD = thr_mma.partition_C(cD);
          CUTE_STATIC_ASSERT_V(size(tCcD) == size(accumulators));
          ThreadEpilogueOp epilogue_op(params_epilogue_local.thread);

          // PPU0015 bulk stores are warp-collective, while the accumulator
          // fragments are distributed across the CTA's 2-D warp layout.
          // Reuse the mainloop arena, sized for this policy's padded tile,
          // to transpose BF16 fragments to row-major, then assign each whole
          // row to one warp for a direct store into the symmetric combine
          // buffer. This removes the global GEMM2 scratch round trip.
          constexpr int kPairsPerRow = kTileN_ / 2;
          constexpr int kRowStridePairs = SharedStorage<typename Kernel::TileShape>::kRowStridePairs;
          static_assert(
              kTileM_ * kRowStridePairs * sizeof(uint32_t) <= SharedStorageSize,
              "direct fragment scatter shared-memory tile exceeds its arena");
          constexpr int kNumWarps = MaxThreadsPerBlock / 32;
          static_assert(kTileN_ % 256 == 0,
              "direct fragment scatter requires full warp stores (256 columns)");
          const int lane_idx = thread_idx % 32;
          const int warp_idx = thread_idx / 32;
          auto* tile_smem = reinterpret_cast<uint32_t*>(smem_buf);

          // CollectiveMainloop uses the same shared-memory arena.
          __syncthreads();
          CUTLASS_PRAGMA_UNROLL
          for (int warp_m = 0; warp_m < size<1>(tCcD); ++warp_m) {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_m = 0;
                 mma_m < size<1>(shape<0>(tCcD)); ++mma_m) {
              CUTLASS_PRAGMA_UNROLL
              for (int warp_n = 0; warp_n < size<2>(tCcD); ++warp_n) {
                CUTLASS_PRAGMA_UNROLL
                for (int mma_n = 0;
                     mma_n < size<2>(shape<0>(tCcD)); ++mma_n) {
                  auto coord = tCcD(
                      make_coord(Int<0>{}, mma_m, mma_n),
                      warp_m, warp_n);
                  int row = static_cast<int>(get<0>(coord));
                  int col = static_cast<int>(get<1>(coord));
                  FragmentAccumulator input_array =
                      *reinterpret_cast<const FragmentAccumulator*>(
                          accumulators(
                              make_coord(_, mma_m, mma_n),
                              warp_m, warp_n).data());
                  auto output_array = epilogue_op(input_array);
                  uint32_t payload =
                      *reinterpret_cast<const uint32_t*>(output_array.data());
                  tile_smem[row * kRowStridePairs + col / 2] = payload;
                }
              }
            }
          }
          __syncthreads();

          CUTLASS_PRAGMA_UNROLL
          for (uint32_t row = warp_idx;
               row < valid_m_sc; row += kNumWarps) {
            const uint64_t packed = __ldg(params.moe_token_meta + route_row_base + row);
            const uint32_t rank_idx = packed >> 32;
            const uint32_t token_idx = packed & 0x00FFFFFFu;
            const uint32_t topk_idx = (packed >> 24) & 0xFFu;
            auto* dst_row = reinterpret_cast<__ppu_bfloat16*>(
                params.moe_sym_buffer.offsets[rank_idx] +
                reinterpret_cast<int64_t>(
                    params.moe_combine_buf.get(topk_idx, token_idx)));
            CUTLASS_PRAGMA_UNROLL
            for (int vector_base = 0;
                 vector_base < kTileN_ / 8; vector_base += 32) {
              const auto* src_vectors = reinterpret_cast<const uint4*>(
                  tile_smem + row * kRowStridePairs);
#if __HGGC_ARCH__ == 150
              uint4 payload = __ppu_shared_load_bulk_b32x4(
                  src_vectors + vector_base + lane_idx);
#else
              uint4 payload = src_vectors[vector_base + lane_idx];
#endif
              auto* dst = reinterpret_cast<uint4*>(
                  dst_row + n_start_sc) + vector_base + lane_idx;
#if __HGGC_ARCH__ == 150
              if (rank_idx != params.moe_sym_buffer.rank_idx)
                __ppu_remote_store_bulk_ncp_b32x4(
                    reinterpret_cast<void*>(dst), payload);
              else
                __ppu_global_store_bulk_b32x4(
                    reinterpret_cast<void*>(dst), payload);
#else
              *dst = payload;
#endif
            }
          }
          // Persistent CTAs may reuse the shared arena for the next tile.
          __syncthreads();
        } else {
          CollectiveEpilogue epilogue{
              params_epilogue_local, shared_storage.tensors.epilogue};
          auto epilogue_blk_coord_mnkl =
              make_coord(epilogue_m_coord, n_coord, _, l_coord);
          epilogue(
              problem_shape_MNKL, blk_shape, epilogue_blk_coord_mnkl,
              accumulators, tiled_mma, residue_mnk, thread_idx,
              (char*)&shared_storage.tensors.epilogue);
        }
      } else {
        CollectiveEpilogue epilogue{
            params_epilogue_local, shared_storage.tensors.epilogue};
        auto epilogue_blk_coord_mnkl =
            make_coord(epilogue_m_coord, n_coord, _, l_coord);
        epilogue(
            problem_shape_MNKL, blk_shape, epilogue_blk_coord_mnkl,
            accumulators, tiled_mma, residue_mnk, thread_idx,
            (char*)&shared_storage.tensors.epilogue);
      }

      // Optional FP8 combine format is materialized in D before packing.
      // BF16 output has already scattered directly from the accumulator tile.
      if constexpr (kFp8Output) {
        if (params.moe_token_meta == nullptr)
          return;
        const auto* tile_d = reinterpret_cast<const __ppu_bfloat16*>(
            params_epilogue_local.ptr_D + m_block_idx * kTileM_ * N);
        __syncthreads();
        mega_moe::detail::scatter_fp8<
            kTileM_, kTileN_, MaxThreadsPerBlock, kScaleMode>(
            tile_d, valid_m_sc, n_start_sc, static_cast<uint32_t>(N),
            route_row_base, params.moe_token_meta,
            params.moe_combine_buf, params.moe_sym_buffer,
            kFp8ExpShift);
      }
  }
};
} // namespace deep_gemm::mega_moe::fp4
