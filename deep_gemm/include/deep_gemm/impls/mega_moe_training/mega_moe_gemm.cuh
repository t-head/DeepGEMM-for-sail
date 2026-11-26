#pragma once

#ifndef DG_BF16_RETURN_NCP_MODE
#define DG_BF16_RETURN_NCP_MODE 0
#endif
#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <type_traits>

#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/mega_moe_training.cuh>

#include <cute/ppu_tensor_mix.hpp>
#include <cutlass/gemm/config/gemm_operands.hpp>

#include <cute/atom/mma_traits_ppu0010.hpp>
#include <cute/atom/mma_traits_ppu0015.hpp>
#include <cute/atom/copy_traits_ppu0010_aiu.hpp>
#include <cute/atom/copy_traits_ppu0015_aiu.hpp>
#include <cute/algorithm/ppu_copy.hpp>

#include <deep_gemm/scheduler/fused_scheduler.cuh>
#include <deep_gemm/scheduler/mega_moe_training_scheduler.cuh>
#include <deep_gemm/impls/fused_gemm_util.cuh>
#include <deep_gemm/common/utils_cutlass3.cuh>

using namespace cute;

namespace deep_gemm {

namespace mega_moe_training {

template <uint32_t Threads = 0>
__device__ __forceinline__ void peer_compute_sync() {
    if constexpr (Threads != 0) __ppu_barrier_sync(0, Threads);
    else __syncthreads();
}

template <uint32_t BM, uint32_t BN, uint32_t N, uint32_t Threads>
__device__ __forceinline__ void return_pipeline_consumer(
        const MegaMoEGemmParams& params, MegaMoEDispatchPipeline control) {
    constexpr uint32_t ntiles = (N + BN - 1) / BN;
    const uint64_t begin = clock64();
    const uint32_t blocks = __ldg(params.gemm.aligned_num_m_blocks);
    const auto* output = reinterpret_cast<const uint4*>(params.gemm.c_ptr);
    uint64_t wait_cycles = 0, returned = 0;
    __shared__ int failed;
    __shared__ uint64_t destination_addresses[BM];
    for (uint32_t task = blockIdx.x; task < blocks * ntiles; task += control.communication_ctas) {
        const uint32_t tile = task / ntiles, column_base = (task % ntiles) * BN;
        const uint4 record = __ldg(reinterpret_cast<const uint4*>(params.gemm.expert_ids_and_cumsum) + tile);
        const uint32_t base = record.w + (tile - record.z) * BM;
        const uint32_t valid = min(BM, record.y - (tile - record.z) * BM);
        if (threadIdx.x == 0) {
            const uint64_t wait_begin = clock64();
            const uint32_t signal = control.serialize ? control.signal_capacity : task;
            const uint32_t target = control.serialize ? blocks * ntiles : 1u;
            while (tix::ld_acq(control.ready + signal) != target) {
                if ((tix::ld_acq(control.status) & 8u) || clock64() - wait_begin > control.timeout_cycles) {
                    atomicOr(control.status, 8u);
                    break;
                }
            }
            wait_cycles += clock64() - wait_begin;
            failed = (tix::ld_acq(control.status) & 8u) != 0;
        }
        __syncthreads();
        if (failed) break;
        // Resolve ownership once per row. Repeating the permutation load,
        // peer-table dependency and division for every BF16x8 vector stalls
        // the payload loop, especially when only a few communication CTAs
        // share the device with GEMM consumers.
        for (uint32_t row = threadIdx.x; row < valid; row += Threads) {
            const int route = __ldg(params.permutation + base + row);
            uint64_t address = 0;
            if (route >= 0) {
                const uint32_t owner = uint32_t(route) / params.destination_rows_per_rank;
                if (owner < params.num_ranks) {
                    const uint64_t local_row = uint32_t(route) % params.destination_rows_per_rank;
                    address = __ldg(params.destination_peers + owner)
                              + (local_row * N + column_base) * sizeof(cutlass::bfloat16_t);
                }
            }
            destination_addresses[row] = address;
        }
        __syncthreads();
        for (uint32_t item = threadIdx.x; item < valid * (BN / 8); item += Threads) {
            const uint32_t row = item / (BN / 8), column = column_base + (item % (BN / 8)) * 8;
            if (column >= N) continue;
            const uint64_t address = destination_addresses[row];
            if (address)
                reinterpret_cast<uint4*>(address)[item % (BN / 8)] =
                    output[uint64_t(base + row) * (N / 8) + column / 8];
        }
        // Prevent the next tile's failure flag update racing another warp's
        // read, and complete this CTA's stores before its kernel can return.
        __syncthreads();
        returned += uint64_t(valid) * BN * sizeof(cutlass::bfloat16_t);
    }
    if (threadIdx.x == 0 && control.statistics) {
        control.statistics[blockIdx.x * 4] = clock64() - begin;
        control.statistics[blockIdx.x * 4 + 1] = wait_cycles;
        control.statistics[blockIdx.x * 4 + 2] = returned;
        control.statistics[blockIdx.x * 4 + 3] = 2;
    }
}

template <uint32_t BM, uint32_t K, uint32_t Threads>
__device__ __forceinline__ void dispatch_pipeline_producer(
        const MegaMoEGemmParams& params, MegaMoEDispatchPipeline control) {
    const uint64_t begin = clock64();
    const uint32_t blocks = __ldg(params.gemm.aligned_num_m_blocks);
    uint64_t produced = 0;
    __shared__ uint64_t source_addresses[BM];
    auto* packed = reinterpret_cast<uint4*>(const_cast<void*>(params.gemm.a_ptr));
    for (uint32_t tile = blockIdx.x; tile < blocks; tile += control.communication_ctas) {
        const auto* records = reinterpret_cast<const uint4*>(params.gemm.expert_ids_and_cumsum);
        const uint4 record = __ldg(records + tile);
        const uint32_t base = record.w + (tile - record.z) * BM;
        for (uint32_t row = threadIdx.x; row < BM; row += Threads) {
            const uint32_t source_row = __ldg(params.gemm.sorted_token_ids + base + row);
            uint64_t address = 0;
            if (source_row < uint64_t(params.num_ranks) * params.source_rows_per_rank) {
                const uint32_t rank = source_row / params.source_rows_per_rank;
                const uint32_t local_row = source_row % params.source_rows_per_rank;
                address = __ldg(params.source_peers + rank)
                          + uint64_t(local_row) * K * sizeof(cutlass::bfloat16_t);
            }
            source_addresses[row] = address;
        }
        __syncthreads();
        for (uint32_t element = threadIdx.x; element < BM * (K / 8); element += Threads) {
            const uint32_t row = element / (K / 8), column = element % (K / 8);
            const uint64_t address = source_addresses[row];
            const uint4 value = address ? reinterpret_cast<const uint4*>(address)[column] : uint4{};
            packed[uint64_t(base) * (K / 8) + element] = value;
        }
        // Every producer thread publishes its stores before thread0 releases
        // the tile. Consumer AIU loads use the acquired tile's local buffer.
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            tix::red_add_rel(control.ready + tile, 1u);
            tix::red_add_rel(control.ready + control.signal_capacity, 1u);
        }
        __syncthreads();
        produced += BM * uint64_t(K) * sizeof(cutlass::bfloat16_t);
    }
    if (threadIdx.x == 0 && control.statistics) {
        control.statistics[blockIdx.x * 4] = clock64() - begin;
        control.statistics[blockIdx.x * 4 + 1] = 0;
        control.statistics[blockIdx.x * 4 + 2] = produced;
        control.statistics[blockIdx.x * 4 + 3] = 0;
    }
}

// The producer owns every row of its M tile. Materialize the packed input once,
// then feed this CTA's selected N sweep through AIU. No other CTA is waited on.
template <uint32_t BM, uint32_t K, uint32_t Threads, uint32_t CopyILP = 1>
__device__ __forceinline__ void pack_producer_input_rows(
        const uint64_t* addresses, const MegaMoEGemmParams& params, uint32_t packed_row_base) {
    auto* packed = reinterpret_cast<uint4*>(const_cast<void*>(params.gemm.a_ptr))
        + uint64_t(packed_row_base) * (K / 8);
    static_assert(CopyILP == 1 || CopyILP == 2 || CopyILP == 4 || CopyILP == 8);
    if constexpr (CopyILP == 1) {
    for (uint32_t position = threadIdx.x; position < BM * (K / 8); position += Threads) {
        const uint32_t row = position / (K / 8), column = position % (K / 8);
        const uint64_t address = addresses[row];
        packed[position] = address ? reinterpret_cast<const uint4*>(address)[column] : uint4{};
    }
    } else {
        // Issue independent vector loads before their dependent stores. Keep
        // raw BF16 bits, padding and publication ordering unchanged. The
        // register arrays have compile-time indices and no dynamic indexing.
        for (uint32_t base = threadIdx.x; base < BM * (K / 8); base += Threads * CopyILP) {
            uint4 values[CopyILP];
            #pragma unroll
            for (uint32_t j = 0; j < CopyILP; ++j) {
                const uint32_t position = base + j * Threads;
                uint4 value{};
                if (position < BM * (K / 8)) {
                    const uint32_t row = position / (K / 8), column = position % (K / 8);
                    const uint64_t address = addresses[row];
                    if (address) value = reinterpret_cast<const uint4*>(address)[column];
                }
                values[j] = value;
            }
            #pragma unroll
            for (uint32_t j = 0; j < CopyILP; ++j) {
                const uint32_t position = base + j * Threads;
                if (position < BM * (K / 8)) packed[position] = values[j];
            }
        }
    }
    __threadfence();
    __syncthreads();
}

// The first dHidden N slice owns the complete M tile. Gather raw DY once,
// retain weighted BF16 DY for DW2, and run real dHidden/activation/dScore work.
template <uint32_t BM,uint32_t K,uint32_t Threads,uint32_t CopyILP>
__device__ __forceinline__ void pack_dhidden_producer_rows(
        const uint64_t* addresses,const MegaMoEGemmParams& params,
        const MegaMoEDHiddenScores& scores,uint32_t base) {
    __shared__ float row_scores[BM];
    for(uint32_t row=threadIdx.x;row<BM;row+=Threads) {
        const int route=params.permutation[base+row];float score=0.f;
        if(route>=0) {
            const uint32_t owner=uint32_t(route)/params.destination_rows_per_rank;
            if(owner<params.num_ranks) {
                score=reinterpret_cast<const float*>(scores.score_peers[owner])[
                    uint32_t(route)%params.destination_rows_per_rank];
            } else atomicOr(scores.status,2u);
        }
        row_scores[row]=score;
        const_cast<float*>(scores.row_scores)[base+row]=score;
    }
    __syncthreads();
    auto* raw_out=reinterpret_cast<uint4*>(const_cast<void*>(params.gemm.a_ptr))+uint64_t(base)*(K/8);
    auto* weighted_out=reinterpret_cast<uint4*>(scores.packed_dy)+uint64_t(base)*(K/8);
    for(uint32_t item=threadIdx.x;item<BM*(K/8);item+=Threads*CopyILP) {
        uint4 values[CopyILP];
        #pragma unroll
        for(uint32_t j=0;j<CopyILP;++j) {
            const uint32_t at=item+j*Threads;uint4 value{};
            if(at<BM*(K/8)) {
                const uint32_t row=at/(K/8),column=at%(K/8);
                if(addresses[row])value=reinterpret_cast<const uint4*>(addresses[row])[column];
            }
            values[j]=value;
        }
        #pragma unroll
        for(uint32_t j=0;j<CopyILP;++j) {
            const uint32_t at=item+j*Threads;
            if(at<BM*(K/8)) {
                const uint4 raw=values[j];raw_out[at]=raw;
                const auto* input=reinterpret_cast<const __ppu_bfloat16*>(&raw);
                uint4 weighted;auto* pairs=reinterpret_cast<uint32_t*>(&weighted);
                const float score=row_scores[at/(K/8)];
                #pragma unroll
                for(int q=0;q<4;++q)pairs[q]=tix::cvt_rte_bf16x2_f32(
                    __bfloat162float(input[2*q+1])*score,__bfloat162float(input[2*q])*score);
                weighted_out[at]=weighted;
            }
        }
    }
    __threadfence();__syncthreads();
}

template <uint32_t BM, uint32_t K, uint32_t Threads>
__device__ __forceinline__ void cache_peer_input_rows(
        uint64_t* addresses, const MegaMoEGemmParams& params,
        uint32_t packed_row_base, bool pack_input) {
    auto* cache = reinterpret_cast<uint4*>(params.dispatch_scratch) + uint64_t(blockIdx.x) * BM * (K / 8);
    auto* packed = reinterpret_cast<uint4*>(const_cast<void*>(params.gemm.a_ptr));
    for (uint32_t position = threadIdx.x; position < BM * (K / 8); position += Threads) {
        const uint32_t row = position / (K / 8), column = position % (K / 8);
        const uint64_t address = addresses[row];
        const uint4 value = address ? reinterpret_cast<const uint4*>(address)[column] : uint4{};
        cache[position] = value;
        if (pack_input)
            packed[uint64_t(packed_row_base + row) * (K / 8) + column] = value;
    }
    // Consumers bypass L1 below. Make every thread's writes visible before
    // another warp starts the local-cache-to-TSM pipeline.
    __threadfence();
    __syncthreads();
    for (uint32_t row = threadIdx.x; row < BM; row += Threads)
        addresses[row] = addresses[row] ? reinterpret_cast<uint64_t>(cache + uint64_t(row) * (K / 8)) : 0;
    __syncthreads();
}

// The native implicit-permute mainloop already gathers token rows directly
// into its shared-memory pipeline. Resolve each row through a peer pointer
// table so that the GEMM itself performs the dispatch reads over ICN/fabric.
template <typename SrcT, typename ACopyInst, typename TilerA,
          uint32_t BLOCK_M, uint32_t BLOCK_K, uint32_t STRIDE_AM, class TAsA>
__device__ __forceinline__ void copy_peer_A_to_tsm(
        TAsA&& destination, const uint64_t* source_addresses, const uint64_t* peers,
        uint32_t stage_offset, uint32_t thread_idx,
        void* packed_input, uint32_t packed_row_base, bool pack_input) {
    constexpr uint32_t TILE_M = cute::get<0>(TilerA{});
    constexpr uint32_t TILE_K = cute::get<1>(TilerA{});
    constexpr uint32_t VECTOR = 128 / cutlass::sizeof_bits<SrcT>::value;
    constexpr uint32_t K_THREADS = TILE_K / VECTOR;
    constexpr uint32_t M_ITER = cute::ceil_div(BLOCK_M, TILE_M);
    constexpr uint32_t K_ITER = BLOCK_K / TILE_K;
    const uint32_t tid_k = thread_idx % K_THREADS, tid_m = thread_idx / K_THREADS;
    if (tid_m < BLOCK_M) {
        CUTLASS_PRAGMA_UNROLL
        for (uint32_t mi = 0; mi < M_ITER; ++mi) {
            // Resolve peer ownership and the row's byte offset once per work
            // tile, rather than repeating divisions during every K stage.
            const uint64_t address = source_addresses[mi * TILE_M + tid_m];
            const bool valid = address != 0;
            const auto* source = reinterpret_cast<const SrcT*>(valid ? address : __ldg(peers));
            CUTLASS_PRAGMA_UNROLL
            for (uint32_t ki = 0; ki < K_ITER; ++ki) {
                const auto* src = reinterpret_cast<const cutlass::uint128_t*>(source
                    + ki * TILE_K + tid_k * VECTOR + stage_offset);
                auto* dst = reinterpret_cast<cutlass::uint128_t*>(cute::raw_pointer_cast(destination(_, mi, ki).data()));
                ACopyInst::copy(*src, *dst, valid);
                if (pack_input) {
                    cutlass::uint128_t value{};
                    if (valid) value = *src;
                    auto* packed = reinterpret_cast<SrcT*>(packed_input)
                        + uint64_t(packed_row_base + mi * TILE_M + tid_m) * STRIDE_AM
                        + ki * TILE_K + tid_k * VECTOR + stage_offset;
                    *reinterpret_cast<cutlass::uint128_t*>(packed) = value;
                }
            }
        }
    }
}

template <typename DstT, uint32_t SHAPE_N, bool Coherent, bool Cached = false, class Accum, class Coordinates>
__device__ __forceinline__ void scatter_peer_output(
        const Accum& accum, const Coordinates& coordinates, const MegaMoEGemmParams& params,
        uint32_t row_base, uint32_t valid_rows, uint32_t column_base,
        const uint64_t* cached_destinations = nullptr) {
    // tc02 fragments own adjacent pairs. Resolve the destination once per
    // fragment row and use the native fused BF16x2 conversion/store.
    CUTLASS_PRAGMA_UNROLL
    for (int wm = 0; wm < cute::size<1>(coordinates); ++wm) {
        CUTLASS_PRAGMA_UNROLL
        for (int mm = 0; mm < cute::size<0, 1>(coordinates); ++mm) {
            const uint32_t row = cute::get<0>(coordinates(cute::make_coord(cute::_0{}, mm, cute::_0{}), wm, cute::_0{}));
            if (row >= valid_rows) continue;
            DstT* destination;
            if constexpr (Cached) {
                const uint64_t address = cached_destinations[row];
                if (!address) continue;
                destination = reinterpret_cast<DstT*>(address);
            } else {
                const auto* route_ptr = params.permutation + row_base + row;
                const int route = Coherent ? load_pipeline_metadata(route_ptr) : __ldg(route_ptr);
                if (route < 0) continue;
                const uint32_t rank = uint32_t(route) / params.destination_rows_per_rank;
                if (rank >= params.num_ranks) continue;
                destination = reinterpret_cast<DstT*>(__ldg(params.destination_peers + rank));
                destination += uint64_t(uint32_t(route) % params.destination_rows_per_rank) * SHAPE_N + column_base;
            }
            CUTLASS_PRAGMA_UNROLL
            for (int wn = 0; wn < cute::size<2>(coordinates); ++wn) {
                CUTLASS_PRAGMA_UNROLL
                for (int mn = 0; mn < cute::size<0, 2>(coordinates); ++mn) {
                    const uint32_t column = cute::get<1>(coordinates(cute::make_coord(cute::_0{}, mm, mn), wm, wn));
                    if (column_base + column < SHAPE_N) {
                        const auto* values = cute::raw_pointer_cast(accum(cute::make_coord(cute::_, mm, mn), wm, wn).data());
                        *reinterpret_cast<uint32_t*>(destination + column) = tix::cvt_rte_bf16x2_f32(values[1], values[0]);
                    }
                }
            }
        }
    }
}

// Adjacent tc02 lanes own neighbouring BF16 pairs. Pack two pairs into one
// eight-byte store without routing the accumulator through shared memory.
// Every lane executes both shuffles, including invalid/padded route rows.
template <typename DstT, uint32_t SHAPE_N, bool Coherent, bool Cached,
          class Accum, class Coordinates>
__device__ __forceinline__ void scatter_peer_output_vector8(
        const Accum& accum, const Coordinates& coordinates, const MegaMoEGemmParams& params,
        uint32_t row_base, uint32_t valid_rows, uint32_t column_base,
        const uint64_t* cached_destinations = nullptr) {
    static_assert(cute::size<0, 2>(Coordinates{}) == 2);
    static_assert(SHAPE_N % 16 == 0);
    const bool odd = (threadIdx.x & 1) != 0;
    CUTLASS_PRAGMA_UNROLL
    for (int wm = 0; wm < cute::size<1>(coordinates); ++wm) {
        CUTLASS_PRAGMA_UNROLL
        for (int mm = 0; mm < cute::size<0, 1>(coordinates); ++mm) {
            const uint32_t row = cute::get<0>(coordinates(cute::make_coord(cute::_0{}, mm, cute::_0{}), wm, cute::_0{}));
            DstT* destination = nullptr;
            if (row < valid_rows) {
                if constexpr (Cached) {
                    destination = reinterpret_cast<DstT*>(cached_destinations[row]);
                } else {
                    const auto* route_ptr = params.permutation + row_base + row;
                    const int route = Coherent ? load_pipeline_metadata(route_ptr) : __ldg(route_ptr);
                    if (route >= 0) {
                        const uint32_t rank = uint32_t(route) / params.destination_rows_per_rank;
                        if (rank < params.num_ranks) {
                            destination = reinterpret_cast<DstT*>(__ldg(params.destination_peers + rank));
                            destination += uint64_t(uint32_t(route) % params.destination_rows_per_rank) * SHAPE_N + column_base;
                        }
                    }
                }
            }
            CUTLASS_PRAGMA_UNROLL
            for (int wn = 0; wn < cute::size<2>(coordinates); ++wn) {
                const auto* p0 = cute::raw_pointer_cast(accum(cute::make_coord(cute::_, mm, cute::_0{}), wm, wn).data());
                const auto* p1 = cute::raw_pointer_cast(accum(cute::make_coord(cute::_, mm, cute::_1{}), wm, wn).data());
                const uint32_t pair0 = tix::cvt_rte_bf16x2_f32(p0[1], p0[0]);
                const uint32_t pair1 = tix::cvt_rte_bf16x2_f32(p1[1], p1[0]);
                const uint32_t neighbour0 = __shfl_xor_sync(0xffffffffu, pair0, 1);
                const uint32_t neighbour1 = __shfl_xor_sync(0xffffffffu, pair1, 1);
                const uint2 value = odd ? uint2{neighbour1, pair1} : uint2{pair0, neighbour0};
                const uint32_t column = cute::get<1>(coordinates(cute::make_coord(cute::_0{}, mm, cute::_0{}), wm, wn)) + (odd ? 6 : 0);
                if (destination && column_base + column + 3 < SHAPE_N)
                    *reinterpret_cast<uint2*>(destination + column) = value;
            }
        }
    }
}

template <typename DstT, uint32_t BM, uint32_t BN, uint32_t SHAPE_N, bool Coherent,
          class Accum, class Coordinates>
__device__ __forceinline__ void scatter_peer_output_staged(
        const Accum& accum, const Coordinates& coordinates, const MegaMoEGemmParams& params,
        DstT* stage, uint32_t row_base, uint32_t valid_rows, uint32_t column_base) {
    // Keep adjacent fragment rows at different shared-memory bank offsets.
    // The pitch stays 128-bit aligned for the subsequent vector reads.
    constexpr uint32_t StageN = BN + 16;
    // All MMA readers must finish before reusing the input pipeline storage.
    __syncthreads();
    CUTLASS_PRAGMA_UNROLL
    for (int wm = 0; wm < cute::size<1>(coordinates); ++wm) {
        CUTLASS_PRAGMA_UNROLL
        for (int mm = 0; mm < cute::size<0, 1>(coordinates); ++mm) {
            const uint32_t row = cute::get<0>(coordinates(cute::make_coord(cute::_0{}, mm, cute::_0{}), wm, cute::_0{}));
            if (row >= valid_rows) continue;
            CUTLASS_PRAGMA_UNROLL
            for (int wn = 0; wn < cute::size<2>(coordinates); ++wn) {
                CUTLASS_PRAGMA_UNROLL
                for (int mn = 0; mn < cute::size<0, 2>(coordinates); ++mn) {
                    const uint32_t col = cute::get<1>(coordinates(cute::make_coord(cute::_0{}, mm, mn), wm, wn));
                    const auto* values = cute::raw_pointer_cast(accum(cute::make_coord(cute::_, mm, mn), wm, wn).data());
                    *reinterpret_cast<uint32_t*>(stage + uint64_t(row) * StageN + col)
                        = tix::cvt_rte_bf16x2_f32(values[1], values[0]);
                }
            }
        }
    }
    __syncthreads();
    // Redistribute fragment pairs into contiguous BF16x8 remote stores. No
    // extra global output buffer or separate return kernel is introduced.
    constexpr uint32_t vectors = BN / 8;
    for (uint32_t index = threadIdx.x; index < valid_rows * vectors; index += blockDim.x) {
        const uint32_t row = index / vectors, col = (index % vectors) * 8;
        if (column_base + col >= SHAPE_N) continue;
        const int route = Coherent ? load_pipeline_metadata(params.permutation + row_base + row)
                                   : __ldg(params.permutation + row_base + row);
        if (route < 0) continue;
        const uint32_t rank = uint32_t(route) / params.destination_rows_per_rank;
        if (rank >= params.num_ranks) continue;
        auto* destination = reinterpret_cast<DstT*>(__ldg(params.destination_peers + rank));
        destination += uint64_t(uint32_t(route) % params.destination_rows_per_rank) * SHAPE_N + column_base + col;
#if DG_BF16_RETURN_NCP_MODE
        static_assert(BN==256 && sizeof(DstT)==2);
        const auto* words=reinterpret_cast<const uint32_t*>(stage+uint64_t(row)*StageN);
        const uint32_t lane=col/8;
        const uint4 value{words[lane],words[lane+32],words[lane+64],words[lane+96]};
        if(rank==params.return_local_rank)__ppu_global_store_bulk_b32x4(destination,value);
        else __ppu_remote_store_bulk_ncp_b32x4(destination,value);
#else
        *reinterpret_cast<uint4*>(destination) = reinterpret_cast<const uint4*>(stage)[
            uint64_t(row) * (StageN / 8) + index % vectors];
#endif
    }
    __syncthreads();
}

}  // namespace mega_moe_training

namespace mega_moe_training {

template <uint32_t Width, bool G8 = false, class Accum, class Coordinates>
__device__ __forceinline__ void activation_backward_epilogue(
        const Accum& accum, const Coordinates& coordinates, const MegaMoEGemmParams& params,
        uint32_t row_base, uint32_t valid_rows, uint32_t column_base) {
    auto* gate_up = reinterpret_cast<__ppu_bfloat16*>(params.activation_gate_up);
    CUTLASS_PRAGMA_UNROLL
    for (int wm = 0; wm < cute::size<1>(coordinates); ++wm) {
        CUTLASS_PRAGMA_UNROLL
        for (int mm = 0; mm < cute::size<0, 1>(coordinates); ++mm) {
            const uint32_t row = cute::get<0>(coordinates(cute::make_coord(cute::_0{}, mm, cute::_0{}), wm, cute::_0{}));
            if (row >= valid_rows) continue;
            CUTLASS_PRAGMA_UNROLL
            for (int wn = 0; wn < cute::size<2>(coordinates); ++wn) {
                CUTLASS_PRAGMA_UNROLL
                for (int mn = 0; mn < cute::size<0, 2>(coordinates); ++mn) {
                    const uint32_t column = column_base + cute::get<1>(coordinates(cute::make_coord(cute::_0{}, mm, mn), wm, wn));
                    if (column >= Width) continue;
                    const auto* values = cute::raw_pointer_cast(accum(cute::make_coord(cute::_, mm, mn), wm, wn).data());
                    float dg[2], du[2];
                    CUTLASS_PRAGMA_UNROLL
                    for (int element = 0; element < 2; ++element) {
                        const uint32_t feature = column + element;
                        const uint64_t offset = uint64_t(row_base + row) * 2 * Width
                            + (G8 ? feature / 8 * 16 + feature % 8 : feature);
                        const float gate = __bfloat162float(gate_up[offset]);
                        const float up = __bfloat162float(gate_up[offset + (G8 ? 8 : Width)]);
                        // Preserve the existing GEMM's BF16 dHidden boundary.
                        const float dh = __bfloat162float(__float2bfloat16(values[element]));
                        const float sigmoid = 1.f / (1.f + expf(-gate));
                        dg[element] = dh * up * sigmoid * (1.f + gate * (1.f - sigmoid));
                        du[element] = dh * (gate * sigmoid);
                    }
                    const uint64_t base = uint64_t(row_base + row) * 2 * Width
                        + (G8 ? column / 8 * 16 + column % 8 : column);
                    *reinterpret_cast<uint32_t*>(gate_up + base) = tix::cvt_rte_bf16x2_f32(dg[1], dg[0]);
                    *reinterpret_cast<uint32_t*>(gate_up + base + (G8 ? 8 : Width)) = tix::cvt_rte_bf16x2_f32(du[1], du[0]);
                }
            }
        }
    }
}

// dScore = dot(hidden, rawDY @ W2), before route-score scaling. This
// eliminates FC2 replay and is defined for zero, tiny and negative scores.
// BF16 scaling is reassociated from DY to dHidden; independent FP32 and
// mega_moe_training-quality checks are required before enabling this optional path.
template <uint32_t Width, uint32_t BlockM, class Accum, class Coordinates>
__device__ __forceinline__ void dhidden_score_epilogue(
        const Accum& accum, const Coordinates& coordinates, const MegaMoEGemmParams& params,
        const MegaMoEDHiddenScores& scores, uint32_t row_base, uint32_t valid_rows,
        uint32_t column_base, uint32_t n_tile) {
    __shared__ __align__(16) float row_dots[BlockM];
    for (uint32_t row = threadIdx.x; row < BlockM; row += blockDim.x)
        row_dots[row] = 0.f;
    __syncthreads();
    auto* gate_up = reinterpret_cast<__ppu_bfloat16*>(params.activation_gate_up);
    const auto* hidden = reinterpret_cast<const __ppu_bfloat16*>(scores.hidden);
    CUTLASS_PRAGMA_UNROLL
    for (int wm = 0; wm < cute::size<1>(coordinates); ++wm) {
        CUTLASS_PRAGMA_UNROLL
        for (int mm = 0; mm < cute::size<0, 1>(coordinates); ++mm) {
            const uint32_t row = cute::get<0>(coordinates(cute::make_coord(cute::_0{}, mm, cute::_0{}), wm, cute::_0{}));
            float dot = 0.f;
            const float scale = row < valid_rows ? scores.row_scores[row_base + row] : 0.f;
            CUTLASS_PRAGMA_UNROLL
            for (int wn = 0; wn < cute::size<2>(coordinates); ++wn) {
                CUTLASS_PRAGMA_UNROLL
                for (int mn = 0; mn < cute::size<0, 2>(coordinates); ++mn) {
                    const uint32_t column = column_base + cute::get<1>(coordinates(cute::make_coord(cute::_0{}, mm, mn), wm, wn));
                    if (row >= valid_rows || column >= Width) continue;
                    const auto* values = cute::raw_pointer_cast(accum(cute::make_coord(cute::_, mm, mn), wm, wn).data());
                    float dg[2], du[2];
                    CUTLASS_PRAGMA_UNROLL
                    for (int element = 0; element < 2; ++element) {
                        const uint32_t feature = column + element;
                        dot += values[element] * __bfloat162float(hidden[uint64_t(row_base + row) * Width + feature]);
                        const uint64_t offset = uint64_t(row_base + row) * 2 * Width + feature / 8 * 16 + feature % 8;
                        const float gate = __bfloat162float(gate_up[offset]);
                        const float up = __bfloat162float(gate_up[offset + 8]);
                        const float dh = __bfloat162float(__float2bfloat16(values[element] * scale));
                        const float sigmoid = 1.f / (1.f + expf(-gate));
                        dg[element] = dh * up * sigmoid * (1.f + gate * (1.f - sigmoid));
                        du[element] = dh * (gate * sigmoid);
                    }
                    const uint64_t base = uint64_t(row_base + row) * 2 * Width + column / 8 * 16 + column % 8;
                    *reinterpret_cast<uint32_t*>(gate_up + base) = tix::cvt_rte_bf16x2_f32(dg[1], dg[0]);
                    *reinterpret_cast<uint32_t*>(gate_up + base + 8) = tix::cvt_rte_bf16x2_f32(du[1], du[0]);
                }
            }
            // All lanes participate, including ragged rows. Same tc02 row
            // lane-group reduction as the independently checked replay path.
            uint32_t leader = threadIdx.x & 31;
            #pragma unroll
            for (int delta = 1; delta < 32; delta <<= 1) {
                const auto other_row = __shfl_xor_sync(0xffffffffu, row, delta);
                const auto other_dot = __shfl_xor_sync(0xffffffffu, dot, delta);
                const auto other_leader = __shfl_xor_sync(0xffffffffu, leader, delta);
                if (other_row == row) { dot += other_dot; leader = min(leader, other_leader); }
            }
            if (row < valid_rows && (threadIdx.x & 31) == leader)
                atomicAdd(row_dots + row, dot);
        }
    }
    __syncthreads();
    for (uint32_t row = threadIdx.x; row < valid_rows; row += blockDim.x)
        scores.score_partials[uint64_t(row_base + row) * scores.num_n_tiles + n_tile] = row_dots[row];
}

// G=8 interleaving puts each thread's gate/up pair in consecutive tc02 MMA_N
// fragments. Match the BF16 mega_moe_training boundary before applying SwiGLU.
template <uint32_t Width, bool KeepGateUp, class Accum, class Coordinates>
__device__ __forceinline__ void fc1_swiglu_epilogue(
        const Accum& accum, const Coordinates& coordinates, const MegaMoEGemmParams& params,
        uint32_t row_base, uint32_t valid_rows, uint32_t column_base) {
    static_assert(cute::size<0, 2>(Coordinates{}) % 2 == 0);
    static_assert(Width % 16 == 0);
    auto* gate_up = reinterpret_cast<__ppu_bfloat16*>(params.gemm.c_ptr);
    auto* hidden = reinterpret_cast<__ppu_bfloat16*>(params.activation_gate_up);
    CUTLASS_PRAGMA_UNROLL
    for (int wm = 0; wm < cute::size<1>(coordinates); ++wm) {
        CUTLASS_PRAGMA_UNROLL
        for (int mm = 0; mm < cute::size<0, 1>(coordinates); ++mm) {
            const uint32_t row = cute::get<0>(coordinates(cute::make_coord(cute::_0{}, mm, cute::_0{}), wm, cute::_0{}));
            if (row >= valid_rows) continue;
            CUTLASS_PRAGMA_UNROLL
            for (int wn = 0; wn < cute::size<2>(coordinates); ++wn) {
                CUTLASS_PRAGMA_UNROLL
                for (int mn = 0; mn < cute::size<0, 2>(coordinates); mn += 2) {
                    const uint32_t column = column_base + cute::get<1>(coordinates(cute::make_coord(cute::_0{}, mm, mn), wm, wn));
                    if (column >= Width) continue;
                    const auto* g = cute::raw_pointer_cast(accum(cute::make_coord(cute::_, mm, mn), wm, wn).data());
                    const auto* u = cute::raw_pointer_cast(accum(cute::make_coord(cute::_, mm, mn + 1), wm, wn).data());
                    const uint32_t gate_bits = tix::cvt_rte_bf16x2_f32(g[1], g[0]);
                    const uint32_t up_bits = tix::cvt_rte_bf16x2_f32(u[1], u[0]);
                    union Scalar {uint16_t bits; __ppu_bfloat16 value;};
                    float activated[2];
                    CUTLASS_PRAGMA_UNROLL
                    for (int element = 0; element < 2; ++element) {
                        const Scalar gb{uint16_t(gate_bits >> (16 * element))};
                        const Scalar ub{uint16_t(up_bits >> (16 * element))};
                        const float gf = __bfloat162float(gb.value), uf = __bfloat162float(ub.value);
                        const float sigmoid = 1.f / (1.f + expf(-gf));
                        activated[element] = (gf * sigmoid) * uf;
                    }
                    const uint32_t compact = column / 16 * 8 + column % 8;
                    *reinterpret_cast<uint32_t*>(hidden + uint64_t(row_base + row) * (Width / 2) + compact)
                        = tix::cvt_rte_bf16x2_f32(activated[1], activated[0]);
                    if constexpr (KeepGateUp) {
                        const uint64_t offset = uint64_t(row_base + row) * Width + column;
                        *reinterpret_cast<uint32_t*>(gate_up + offset) = gate_bits;
                        *reinterpret_cast<uint32_t*>(gate_up + offset + 8) = up_bits;
                    }
                }
            }
        }
    }
}

}  // namespace mega_moe_training

namespace mega_moe_training {

template <uint32_t Width, uint32_t BlockM, bool WarpReduction, class Accum, class Coordinates>
__device__ __forceinline__ void replay_gradient_epilogue(
        const Accum& accum, const Coordinates& coordinates, const MegaMoEGemmParams& params,
        const MegaMoEReplayGradients& gradients, uint32_t row_base,
        uint32_t valid_rows, uint32_t column_base, uint32_t n_tile) {
    __shared__ __align__(16) uint64_t dy_addresses[BlockM];
    __shared__ __align__(16) float row_scores[BlockM];
    __shared__ __align__(16) float row_dots[BlockM];
    const uint64_t stride = uint64_t(gradients.source_tokens) * gradients.topk;
    for (uint32_t row = threadIdx.x; row < BlockM; row += blockDim.x) {
        const int route = row < valid_rows ? params.permutation[row_base + row] : -1;
        const bool valid = route >= 0 && uint64_t(route) < gradients.num_ranks * stride;
        if (route >= 0 && !valid) atomicOr(gradients.status, 2u);
        const uint32_t owner = valid ? route / stride : 0;
        const uint64_t local = valid ? route % stride : 0;
        dy_addresses[row] = valid ? gradients.dy_peers[owner]
            + (local / gradients.topk) * Width * sizeof(__ppu_bfloat16) : 0;
        row_scores[row] = valid ? reinterpret_cast<const float*>(gradients.score_peers[owner])[local] : 0.f;
        row_dots[row] = 0.f;
    }
    __syncthreads();
    auto* packed = reinterpret_cast<__ppu_bfloat16*>(gradients.packed_dy);
    CUTLASS_PRAGMA_UNROLL
    for (int wm = 0; wm < cute::size<1>(coordinates); ++wm) {
        CUTLASS_PRAGMA_UNROLL
        for (int mm = 0; mm < cute::size<0, 1>(coordinates); ++mm) {
            const uint32_t row = cute::get<0>(coordinates(cute::make_coord(cute::_0{}, mm, cute::_0{}), wm, cute::_0{}));
            if constexpr (!WarpReduction) {
                if (row >= valid_rows) continue;
            }
            float dot = 0.f;
            CUTLASS_PRAGMA_UNROLL
            for (int wn = 0; wn < cute::size<2>(coordinates); ++wn) {
                CUTLASS_PRAGMA_UNROLL
                for (int mn = 0; mn < cute::size<0, 2>(coordinates); ++mn) {
                    const uint32_t column = column_base + cute::get<1>(coordinates(cute::make_coord(cute::_0{}, mm, mn), wm, wn));
                    if (column >= Width) continue;
                    const auto* y = cute::raw_pointer_cast(accum(cute::make_coord(cute::_, mm, mn), wm, wn).data());
                    const uint32_t raw = dy_addresses[row]
                        ? *reinterpret_cast<const uint32_t*>(dy_addresses[row] + uint64_t(column) * 2) : 0;
                    const auto* dy = reinterpret_cast<const __ppu_bfloat16*>(&raw);
                    const float d0 = __bfloat162float(dy[0]), d1 = __bfloat162float(dy[1]);
                    // Match the existing BF16 Y and weighted-DY boundaries.
                    const float y0 = __bfloat162float(__float2bfloat16(y[0]));
                    const float y1 = __bfloat162float(__float2bfloat16(y[1]));
                    dot += d0 * y0;
                    dot += d1 * y1;
                    if (row < valid_rows)
                        *reinterpret_cast<uint32_t*>(packed + uint64_t(row_base + row) * Width + column)
                            = tix::cvt_rte_bf16x2_f32(d1 * row_scores[row], d0 * row_scores[row]);
                }
            }
            if constexpr (WarpReduction) {
                // tc02 C-coordinate lane groups must remain closed under
                // these XOR steps; strict dynamic/FP32 checks gate this mode.
                // Keep every lane participating even for ragged M tails.
                if (row >= valid_rows) dot = 0.f;
                uint32_t leader = threadIdx.x & 31;
                #pragma unroll
                for (int delta = 1; delta < 32; delta <<= 1) {
                    const auto other_row = __shfl_xor_sync(0xffffffffu, row, delta);
                    const auto other_dot = __shfl_xor_sync(0xffffffffu, dot, delta);
                    const auto other_leader = __shfl_xor_sync(0xffffffffu, leader, delta);
                    if (other_row == row) {
                        dot += other_dot;
                        leader = min(leader, other_leader);
                    }
                }
                if (row < valid_rows && (threadIdx.x & 31) == leader)
                    atomicAdd(row_dots + row, dot);
            } else {
                atomicAdd(row_dots + row, dot);
            }
        }
    }
    __syncthreads();
    for (uint32_t row = threadIdx.x; row < valid_rows; row += blockDim.x)
        gradients.score_partials[uint64_t(row_base + row) * gradients.num_n_tiles + n_tile] = row_dots[row];
}

}  // namespace mega_moe_training

template <class _SrcT, GemmType kGemmType,
          uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t WARP_M, uint32_t WARP_N,
          uint32_t BLOCK_SIZE, int kNumStages, bool PeerRead, bool PeerWrite,
          bool Coherent = false, bool ActivationBackward = false, uint32_t DispatchCacheN = 0,
          bool DispatchCacheAiu = false, bool StagedReturn = false,
          uint32_t FirstNTile = 0, uint32_t NumNTiles = 0, bool LocalInputAiu = false,
          bool ProducerInputAiu = false, bool DispatchPipeline = false,
          bool InlineDispatch = false, bool CacheReturn = false, bool FixedMTile = false,
          uint32_t ReturnVectorBytes = 4, bool GateUpG8 = false,
          bool Fc1Activation = false, bool KeepGateUp = true, uint32_t InputCopyILP = 1,
          bool ExternalInput = false, uint32_t PollYieldEvery = 0, uint32_t PollPauseCycles = 0,
          bool ReplayGradients = false, bool ReplayWarpReduction = false, bool DHiddenScores = false,
          bool WeightKN = false, uint32_t SyncThreads = 0, uint32_t MetadataBlockM = BLOCK_M>
__device__ __forceinline__ void
training_bf16_peer_gemm_impl(const mega_moe_training::MegaMoEGemmParams params,
                           const mega_moe_training::MegaMoEDispatchPipeline pipeline = {},
                           const uint32_t selected_m_tile = 0,
                           const uint32_t selected_n_tile = ~0u,
                           const mega_moe_training::MegaMoEReplayGradients gradients = {},
                           const mega_moe_training::MegaMoEDHiddenScores dhidden_scores = {}) {
    static_assert(SyncThreads == 0 || SyncThreads == BLOCK_SIZE);
    const uint64_t pipeline_begin = DispatchPipeline ? clock64() : 0;
    static_assert(!InlineDispatch || (DispatchPipeline && !PeerRead && !PeerWrite));
    static_assert(!ExternalInput || (DispatchPipeline && !InlineDispatch && !PeerRead && !PeerWrite));
    static_assert(PollYieldEvery == 0 || (PollYieldEvery <= 1024 && (PollYieldEvery & (PollYieldEvery - 1)) == 0));
    static_assert(PollPauseCycles <= 16384);
    static_assert(!CacheReturn || (PeerWrite && !DispatchPipeline && !StagedReturn));
    static_assert(ReturnVectorBytes == 4 || ReturnVectorBytes == 8);
    static_assert(ReturnVectorBytes == 4 || (PeerWrite && !DispatchPipeline && !StagedReturn));
    static_assert(!Fc1Activation || (GateUpG8 && !ActivationBackward && !PeerWrite));
    static_assert(!ReplayGradients || (!PeerRead && !PeerWrite && !ActivationBackward
                                      && !Fc1Activation && !DispatchPipeline));
    if constexpr (DispatchPipeline) {
        static_assert(!PeerRead && !ActivationBackward && DispatchCacheN == 0
                      && FirstNTile == 0 && NumNTiles == 0);
        if constexpr (!InlineDispatch && !ExternalInput) {
            if (blockIdx.x < pipeline.communication_ctas) {
                if constexpr (PeerWrite)
                    mega_moe_training::return_pipeline_consumer<BLOCK_M, BLOCK_N, SHAPE_N, BLOCK_SIZE>(params, pipeline);
                else
                    mega_moe_training::dispatch_pipeline_producer<BLOCK_M, SHAPE_K, BLOCK_SIZE>(params, pipeline);
                return;
            }
        }
    }
    static_assert(!DHiddenScores || (ActivationBackward && GateUpG8 && !ReplayGradients
                                    && (!PeerRead || ProducerInputAiu) && !PeerWrite && !DispatchPipeline));
    const auto& args = params.gemm;
    constexpr uint32_t STRIDE_AM = SHAPE_K;
    constexpr uint32_t STRIDE_BE = SHAPE_N * SHAPE_K;
    constexpr uint32_t STRIDE_CM = SHAPE_N;

    using SrcT = typename ToCutlassType<_SrcT>::type;
    using DstT = cutlass::bfloat16_t;
    using AccT = float;
    using TileShape = cute::Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_N>, cute::Int<BLOCK_K>>;
    using WarpShape = cute::Shape<cute::Int<WARP_M>, cute::Int<WARP_N>, cute::Int<BLOCK_K>>;
#if __HGGC_ARCH__ == 100
    using ArchTag = cutlass::arch::PPU0010;
#else
    using ArchTag = cutlass::arch::PPU0015;
#endif

    using MmaInst = typename cutlass::gemm::config::GetMmaInst<ArchTag, SrcT, SrcT, AccT>::type;
    using TiledMma = cute::TiledMMA<
      cute::MMA_Atom<MmaInst>,
      cute::Layout<Shape< Int<BLOCK_M / WARP_M>, Int<BLOCK_N / WARP_N>, _1>>>;

    static_assert(DispatchCacheN == 0 || (PeerRead && !Coherent));
    static_assert(!DispatchCacheAiu || DispatchCacheN != 0);
    static_assert(!LocalInputAiu || (!PeerRead && (FixedMTile || (!PeerWrite && FirstNTile > 0))));
    static_assert(!ProducerInputAiu || (PeerRead && !PeerWrite && FirstNTile == 0
                  && (NumNTiles == 1 || NumNTiles == 0) && DispatchCacheN == 0));
    static_assert((FirstNTile == 0 && NumNTiles == 0) || (!Coherent && DispatchCacheN == 0));
    using OriginalScheduler = std::conditional_t<Coherent,
        mega_moe_training::CoherentFusedScheduler<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, kNumGroups>,
        FusedGemmScheduler<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, kNumGroups>>;
    using UnslicedScheduler = std::conditional_t<DispatchCacheN != 0,
        mega_moe_training::CachedPeerScheduler<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, kNumGroups,
                                     DispatchCacheN == 0 ? 1 : DispatchCacheN>, OriginalScheduler>;
    using SlicedScheduler = std::conditional_t<FirstNTile != 0 || NumNTiles != 0,
        mega_moe_training::SlicedPeerScheduler<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, kNumGroups,
                                     FirstNTile, NumNTiles>, UnslicedScheduler>;
    using PipelineScheduler = std::conditional_t<DispatchPipeline,
        mega_moe_training::DispatchPipelineScheduler<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, kNumGroups,
                                             !PeerWrite, InlineDispatch, PollYieldEvery, PollPauseCycles>,
        SlicedScheduler>;
    using SelectedScheduler = std::conditional_t<FixedMTile,
        mega_moe_training::SelectedMTileScheduler<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, kNumGroups, MetadataBlockM>,
        PipelineScheduler>;
    constexpr bool OwnSource=ProducerInputAiu && NumNTiles==0;
    static_assert(!OwnSource || (!FixedMTile && !DispatchPipeline && !Coherent));
    using TileScheduler=std::conditional_t<OwnSource,
        mega_moe_training::OwnedSourceScheduler<kGemmType,SHAPE_N,SHAPE_K,BLOCK_M,BLOCK_N,kNumGroups>,SelectedScheduler>;
    static_assert(!FixedMTile || (!DispatchPipeline && !PeerRead && DispatchCacheN == 0));

    // Shared memory
    constexpr uint32_t kSmemASize = smem_a_size(sizeof(SrcT), kNumStages, BLOCK_M, BLOCK_K);
    extern __shared__ __align__(128) uint8_t smem_buffer[];
    SrcT* smem_a = reinterpret_cast<SrcT*>(smem_buffer);
    SrcT* smem_b = reinterpret_cast<SrcT*>(smem_buffer + kSmemASize);

    uint32_t thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    uint32_t shape_m = args.shape_m;

    // load A from hbm to tsm: use async copy.
    constexpr int Alignment = 128 / cutlass::sizeof_bits<SrcT>::value;
    using ACopyInst = std::conditional_t<DispatchCacheN != 0,
        cute::PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<cutlass::uint128_t>,
        cute::PPU_CP_ASYNC_CACHEALWAYS_ZFILL<cutlass::uint128_t>>;
    using GatherOperandA = cutlass::gemm::config::Gemm_Hybrid_Operand<
            ArchTag, SrcT, false, Alignment, cute::Int<BLOCK_K>, BLOCK_SIZE,
            ACopyInst, cute::Int<BLOCK_M>>;
    constexpr bool UseAiuA = DispatchCacheAiu || LocalInputAiu || ProducerInputAiu || DispatchPipeline;
    using GemmOperandA = std::conditional_t<UseAiuA,
        cutlass::gemm::config::DefaultGemm_AIU_Operand<
            ArchTag, SrcT, false, cute::Int<BLOCK_M>, cute::Int<BLOCK_K>, false>, GatherOperandA>;

    using TilerA = typename GemmOperandA::GmemTiledCopy::Tiler_MN;
    using SmemLayoutA = decltype(tile_to_shape(typename GemmOperandA::SmemLayoutAtom{},
            Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_K>, cute::Int<kNumStages>>{}));

    typename GemmOperandA::GmemTiledCopy gmem_tiled_copy_A;
    if constexpr (UseAiuA) {
        auto stride_A = cute::make_stride(cute::Int<SHAPE_K>{}, cute::_1{});
        gmem_tiled_copy_A.desc_.template init<SrcT, false, get<0>(TilerA{}), get<1>(TilerA{})>(
            nullptr, BLOCK_M, SHAPE_K, stride_A);
    }
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    Tensor sA = cute::make_tensor(cute::make_smem_ptr(smem_a), SmemLayoutA{});
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);

    // load B from hbm to tsm: use aiu load.
    using GemmOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<
            ArchTag, SrcT, WeightKN, cute::Int<BLOCK_N>, cute::Int<BLOCK_K>, true>;
    using SmemLayoutB = decltype(tile_to_shape(typename GemmOperandB::SmemLayoutAtom{},
            Shape<cute::Int<BLOCK_N>, cute::Int<BLOCK_K>, cute::Int<kNumStages>>{}));

    // init aiu desc
    typename GemmOperandB::GmemTiledCopy gmem_tiled_copy_B;
    using TilerB = typename GemmOperandB::GmemTiledCopy::Tiler_MN;
    auto shape_B = cute::make_shape(Int<SHAPE_N>{}, Int<SHAPE_K>{});
    auto stride_B = [] {
        if constexpr (WeightKN) return cute::make_stride(_1{}, Int<SHAPE_N>{});
        else return cute::make_shape((int)SHAPE_K, _1{});
    }();
    gmem_tiled_copy_B.desc_.template init<SrcT, WeightKN, get<0>(TilerB{}), get<1>(TilerB{})>(nullptr, SHAPE_N, SHAPE_K, stride_B);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
    Tensor sB = cute::make_tensor(cute::make_smem_ptr(smem_b), SmemLayoutB{});
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);
    static_assert(cute::cosize_v<SmemLayoutB> * sizeof(SrcT)
        <= smem_b_size(sizeof(SrcT), kNumStages, BLOCK_N, BLOCK_K));

    //
    // MMA Atom partitioning
    //

    // Tile MMA compute thread partitions and allocate accumulators
    TiledMma tiled_mma;

    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                     // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                     // (MMA,MMA_N,MMA_K)
    Tensor accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{})); // (MMA,MMA_M,MMA_N)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K

    //
    // Copy Atom retiling
    //
    using SmemCopyAtomA = typename GemmOperandA::SmemCopyAtom;
    using SmemCopyAtomB = typename GemmOperandB::SmemCopyAtom;

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
#if __HGGC_ARCH__ >= 150
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(warp_idx * 32);
    Tensor tCsA            = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));
#else
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S((sA));
#endif
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(warp_idx * 32);
    Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));                  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    TileScheduler deep_scheduler = [&]() {
        if constexpr (FixedMTile)
            return TileScheduler(args.aligned_num_m_blocks, args.expert_ids_and_cumsum, selected_m_tile, selected_n_tile);
        else if constexpr (DispatchPipeline)
            return TileScheduler(args.aligned_num_m_blocks, args.expert_ids_and_cumsum, pipeline);
        else
            return TileScheduler(args.aligned_num_m_blocks, args.expert_ids_and_cumsum);
    }();

    #pragma clang loop licm(disable)
    while (deep_scheduler.fetch_next_work(m_block_idx, n_block_idx)) {
      auto blk_coord_mnkl = make_coord(m_block_idx, n_block_idx, _, _1{});
      uint32_t blk_n_offset = n_block_idx * BLOCK_N;
      const int* blk_token_base = args.sorted_token_ids + deep_scheduler.cumsum_m_block_idx * BLOCK_M;
      if constexpr (InlineDispatch) {
          if (n_block_idx == 0) {
              __shared__ int inline_claimed;
              if (thread_idx == 0)
                  inline_claimed = atomicCAS(pipeline.ready + deep_scheduler.cumsum_m_block_idx, 0u, 2u) == 0u;
              mega_moe_training::peer_compute_sync<SyncThreads>();
              if (inline_claimed) {
              __shared__ uint64_t inline_source_addresses[BLOCK_M];
              for (uint32_t row = thread_idx; row < BLOCK_M; row += BLOCK_SIZE) {
                  const uint32_t source_row = __ldg(blk_token_base + row);
                  uint64_t address = 0;
                  if (source_row < uint64_t(params.num_ranks) * params.source_rows_per_rank) {
                      const uint32_t rank = source_row / params.source_rows_per_rank;
                      const uint32_t local_row = source_row % params.source_rows_per_rank;
                      address = __ldg(params.source_peers + rank)
                              + uint64_t(local_row) * SHAPE_K * sizeof(SrcT);
                  }
                  inline_source_addresses[row] = address;
              }
              mega_moe_training::peer_compute_sync<SyncThreads>();
              mega_moe_training::pack_producer_input_rows<BLOCK_M, SHAPE_K, BLOCK_SIZE, InputCopyILP>(
                  inline_source_addresses, params, deep_scheduler.curr_block_m_offset);
              if (thread_idx == 0) {
                  tix::red_add_rel(pipeline.ready + deep_scheduler.cumsum_m_block_idx, uint32_t(-1));
                  tix::red_add_rel(pipeline.ready + pipeline.signal_capacity, 1u);
              }
              mega_moe_training::peer_compute_sync<SyncThreads>();
              } else {
                  __shared__ int inline_failed;
                  if (thread_idx == 0) {
                      const uint64_t begin = clock64();
                      uint32_t polls = 0;
                      while (tix::ld_acq(pipeline.ready + deep_scheduler.cumsum_m_block_idx) != 1u) {
                          if ((tix::ld_acq(pipeline.status) & 8u) || clock64() - begin > pipeline.timeout_cycles) {
                              atomicOr(pipeline.status, 8u);
                              break;
                          }
                          if constexpr (PollYieldEvery != 0)
                              if ((++polls & (PollYieldEvery - 1)) == 0) __ppu_yield();
                          if constexpr (PollPauseCycles != 0) {
                              const uint64_t pause_begin = clock64();
                              while (clock64() - pause_begin < PollPauseCycles) {}
                          }
                      }
                      inline_failed = (tix::ld_acq(pipeline.status) & 8u) != 0;
                  }
                  mega_moe_training::peer_compute_sync<SyncThreads>();
                  if (inline_failed) return;
              }
          }
      }
      const uint64_t* source_addresses = nullptr;
      if constexpr (PeerRead) {
          __shared__ uint64_t cached_source_addresses[BLOCK_M];
          mega_moe_training::peer_compute_sync<SyncThreads>();
          bool first_in_group = true;
          if constexpr (OwnSource) first_in_group = n_block_idx == 0;
          if constexpr (DispatchCacheN != 0) first_in_group = deep_scheduler.first_in_group;
          if (first_in_group) {
            for (int row = thread_idx; row < BLOCK_M; row += BLOCK_SIZE) {
              const uint32_t source_row = Coherent ? mega_moe_training::load_pipeline_metadata(blk_token_base + row)
                                                    : __ldg(blk_token_base + row);
              uint64_t address = 0;
              if (source_row < shape_m) {
                  const uint32_t rank = source_row / params.source_rows_per_rank;
                  const uint32_t local_row = source_row % params.source_rows_per_rank;
                  address = __ldg(params.source_peers + rank) + uint64_t(local_row) * STRIDE_AM * sizeof(SrcT);
              }
              cached_source_addresses[row] = address;
            }
            mega_moe_training::peer_compute_sync<SyncThreads>();
            if constexpr (DispatchCacheN != 0)
                mega_moe_training::cache_peer_input_rows<BLOCK_M, SHAPE_K, BLOCK_SIZE>(cached_source_addresses,
                    params, deep_scheduler.curr_block_m_offset, params.pack_input && n_block_idx == 0);
            if constexpr (ProducerInputAiu) {
                if constexpr (DHiddenScores)
                    mega_moe_training::pack_dhidden_producer_rows<BLOCK_M,SHAPE_K,BLOCK_SIZE,InputCopyILP>(
                        cached_source_addresses,params,dhidden_scores,deep_scheduler.curr_block_m_offset);
                else mega_moe_training::pack_producer_input_rows<BLOCK_M, SHAPE_K, BLOCK_SIZE, InputCopyILP>(
                    cached_source_addresses, params, deep_scheduler.curr_block_m_offset);
            }
          }
          mega_moe_training::peer_compute_sync<SyncThreads>();
          source_addresses = cached_source_addresses;
      }
      const uint64_t* output_addresses = nullptr;
      if constexpr (CacheReturn) {
          __shared__ uint64_t cached_output_addresses[BLOCK_M];
          // The preceding tile's epilogue can still read this cache in a
          // slower warp. Join every warp before reusing its destination rows.
          mega_moe_training::peer_compute_sync<SyncThreads>();
          for (uint32_t row = thread_idx; row < deep_scheduler.valid_m_in_block; row += BLOCK_SIZE) {
              const auto* route_ptr = params.permutation + deep_scheduler.curr_block_m_offset + row;
              const int route = Coherent ? mega_moe_training::load_pipeline_metadata(route_ptr) : __ldg(route_ptr);
              uint64_t address = 0;
              if (route >= 0) {
                  const uint32_t rank = uint32_t(route) / params.destination_rows_per_rank;
                  if (rank < params.num_ranks) {
                      const uint64_t local_row = uint32_t(route) % params.destination_rows_per_rank;
                      address = __ldg(params.destination_peers + rank)
                              + (local_row * SHAPE_N + blk_n_offset) * sizeof(DstT);
                  }
              }
              cached_output_addresses[row] = address;
          }
          mega_moe_training::peer_compute_sync<SyncThreads>();
          output_addresses = cached_output_addresses;
      }
      // gmem_b in block
      SrcT* gmem_b = (SrcT*)args.b_ptr + deep_scheduler.curr_group_idx * STRIDE_BE;
      Tensor mB_nk = cute::make_tensor(cute::make_gmem_ptr(gmem_b), shape_B, stride_B);
      Tensor gB = cute::local_tile(cute::make_mix_tensor_like(mB_nk), TileShape{}, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});           // (BLK_N,BLK_K,k)
      Tensor tBgB = gmem_thr_copy_B.partition_S(gB);

      auto copy_input_stage = [&](int tile, int stage) {
          if constexpr (UseAiuA) {
              // Arbitrary peer rows have already been gathered into this
              // CTA's contiguous cache. AIU is legal for this local operand;
              // its swizzled TSM layout uses the matching A load atom above.
              const auto* cache = (LocalInputAiu || ProducerInputAiu || DispatchPipeline)
                  ? reinterpret_cast<const SrcT*>(args.a_ptr)
                      + uint64_t(deep_scheduler.curr_block_m_offset) * SHAPE_K
                  : reinterpret_cast<const SrcT*>(params.dispatch_scratch)
                      + uint64_t(blockIdx.x) * BLOCK_M * SHAPE_K;
              auto shape_A = cute::make_shape(cute::Int<BLOCK_M>{}, cute::Int<SHAPE_K>{});
              auto stride_A = cute::make_stride(cute::Int<SHAPE_K>{}, cute::_1{});
              Tensor mA = cute::make_tensor(cute::make_gmem_ptr(cache), shape_A, stride_A);
              Tensor gA = cute::local_tile(cute::make_mix_tensor_like(mA), TileShape{},
                  cute::make_coord(cute::_0{}, cute::_, cute::_), Step<_1,X,_1>{});
              Tensor tAgA = gmem_thr_copy_A.partition_S(gA);
              copy_aiu(gmem_tiled_copy_A, tAgA(_,_,_,tile), tAsA(_,_,_,stage), warp_idx);
          } else if constexpr (PeerRead) {
              mega_moe_training::copy_peer_A_to_tsm<SrcT, ACopyInst, TilerA, BLOCK_M, BLOCK_K, STRIDE_AM>(
                  tAsA(_,_,_,stage), source_addresses, params.source_peers,
                  BLOCK_K * tile, thread_idx,
                  const_cast<void*>(args.a_ptr), deep_scheduler.curr_block_m_offset,
                  DispatchCacheN == 0 && params.pack_input && n_block_idx == 0);
          } else {
              copy_A_to_tsm<SrcT, ACopyInst, TilerA, BLOCK_M, BLOCK_K, STRIDE_AM>(
                  tAsA(_,_,_,stage), args.a_ptr,
                  blk_token_base, BLOCK_K * tile, shape_m, thread_idx);
          }
      };

      int k_tile_iter  = 0;
      int k_tile_count = size<2>(gB);

      CUTLASS_PRAGMA_UNROLL
      for (int k_pipe = 0; k_pipe < kNumStages; ++k_pipe) {
        if (k_tile_count > 0) {
          copy_input_stage(k_tile_iter, k_pipe);
          copy_aiu(gmem_tiled_copy_B, tBgB(_,_,_,k_tile_iter), tBsB(_,_,_,k_pipe), warp_idx);
          ++k_tile_iter;
        }
        cp_async_fence();
        --k_tile_count;
      }

      clear(accum);

      //
      // PIPELINED MAIN LOOP
      //

      // Current pipe index in smem to read from
      int smem_pipe_read  = 0;
      // Current pipe index in smem to write to
      int smem_pipe_write = 0;

      Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
      Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

      // Size of the register pipeline
      auto K_BLOCK_MAX = size<2>(tCrA_copy_view);
      auto K_ATOM_PER_COPY = size<2>(tCrA) / size<2>(tCrA_copy_view);

      // PREFETCH register pipeline
      if (K_BLOCK_MAX > 1) {
        // Wait until our first prefetched tile is loaded in
        cp_async_wait<kNumStages-1>();
        mega_moe_training::peer_compute_sync<SyncThreads>();

        // Prefetch the first rmem from the first k-tile
        copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
        copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
      }
      auto process_kblock_iterations = [&](int k_block) {
        // Load A, B shmem->regs for k_block+1
        // Copy gmem to smem before computing gemm on each k-pipe
        if (k_block == K_BLOCK_MAX - 1) {
          // Commit the smem for smem_pipe_read
          cp_async_wait<kNumStages-2>();
          mega_moe_training::peer_compute_sync<SyncThreads>();
          if (k_tile_count > 0) {
            copy_input_stage(k_tile_iter, smem_pipe_write);
            copy_aiu(gmem_tiled_copy_B, tBgB(_,_,_,k_tile_iter), tBsB(_,_,_,smem_pipe_write), warp_idx);
            ++k_tile_iter;
          }
          cp_async_fence();
          // Advance the tile
          --k_tile_count;

          // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
          ++smem_pipe_read;
          smem_pipe_read = (smem_pipe_read == kNumStages) ? 0 : smem_pipe_read;
          smem_pipe_write = smem_pipe_read;

          // Slice the smem_pipe_read smem
          tCsA_p = tCsA(_,_,_,smem_pipe_read);
          tCsB_p = tCsB(_,_,_,smem_pipe_read);
        }
        // Load A, B shmem->regs for k_block+1
        auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
        copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));

        CUTLASS_PRAGMA_UNROLL
        for (int k_loop = 0; k_loop < K_ATOM_PER_COPY; k_loop++) {
          auto atom_idx = k_block * K_ATOM_PER_COPY + k_loop;
          // gemm for one tiled_mma atom on K
          cute::gemm(tiled_mma, accum, tCrA(_,_,atom_idx), tCrB(_,_,atom_idx), accum);
        }
      };

      // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
      //
      // Split out the first loop iteration to facilitate the use of mm instructions
      // for_each(make_int_sequence<K_BLOCK_MAX >{}, [&] (auto k_block) {
      //   process_kblock_iterations(k_block);
      // }); // for_each

      CUTLASS_PRAGMA_NO_UNROLL
      while (k_tile_count > -(kNumStages)) {
        // Pipeline the outer products with a static for loop.
        //
        // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
        for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
          process_kblock_iterations(k_block);
        }); // for_each
      }

      // acc write back
      auto blk_mn_shape = Shape<Int<BLOCK_M>, Int<BLOCK_N>>{};
      Tensor cC = make_identity_tensor(blk_mn_shape);
      Tensor tCcC = thr_mma.partition_C(cC);
      CUTE_STATIC_ASSERT_V(size(tCcC) == size(accum),
          "Accumulator count must have the same destination element count.");

#if __HGGC_ARCH__ == 150
      if constexpr (ReplayGradients) {
          mega_moe_training::replay_gradient_epilogue<SHAPE_N, BLOCK_M, ReplayWarpReduction>(accum, tCcC, params,
              gradients, deep_scheduler.curr_block_m_offset, deep_scheduler.valid_m_in_block,
              blk_n_offset, blk_n_offset / BLOCK_N);
      } else if constexpr (Fc1Activation) {
          mega_moe_training::fc1_swiglu_epilogue<SHAPE_N, KeepGateUp>(accum, tCcC, params,
              deep_scheduler.curr_block_m_offset, deep_scheduler.valid_m_in_block, blk_n_offset);
      } else if constexpr (DHiddenScores) {
          mega_moe_training::dhidden_score_epilogue<SHAPE_N, BLOCK_M>(accum, tCcC, params, dhidden_scores,
              deep_scheduler.curr_block_m_offset, deep_scheduler.valid_m_in_block,
              blk_n_offset, blk_n_offset / BLOCK_N);
      } else if constexpr (ActivationBackward) {
          mega_moe_training::activation_backward_epilogue<SHAPE_N, GateUpG8>(accum, tCcC, params,
              deep_scheduler.curr_block_m_offset, deep_scheduler.valid_m_in_block, blk_n_offset);
      } else if constexpr (PeerWrite && !DispatchPipeline) {
          if constexpr (StagedReturn) {
              static_assert(BLOCK_M * (BLOCK_N + 16) * sizeof(DstT)
                  <= kSmemASize + smem_b_size(sizeof(SrcT), kNumStages, BLOCK_N, BLOCK_K));
              mega_moe_training::scatter_peer_output_staged<DstT, BLOCK_M, BLOCK_N, SHAPE_N, Coherent>(
                  accum, tCcC, params, reinterpret_cast<DstT*>(smem_buffer),
                  deep_scheduler.curr_block_m_offset, deep_scheduler.valid_m_in_block, blk_n_offset);
          } else if constexpr (ReturnVectorBytes == 8) {
              mega_moe_training::scatter_peer_output_vector8<DstT, SHAPE_N, Coherent, CacheReturn>(accum, tCcC, params,
                  deep_scheduler.curr_block_m_offset, deep_scheduler.valid_m_in_block, blk_n_offset, output_addresses);
          } else {
              mega_moe_training::scatter_peer_output<DstT, SHAPE_N, Coherent, CacheReturn>(accum, tCcC, params,
                  deep_scheduler.curr_block_m_offset, deep_scheduler.valid_m_in_block, blk_n_offset, output_addresses);
          }
      } else {
          epilogue_no_tsm<AccT, DstT, SHAPE_N, BLOCK_N, STRIDE_CM>(accum, tCcC, args.c_ptr,
              deep_scheduler.curr_block_m_offset, deep_scheduler.valid_m_in_block, blk_n_offset);
      }
#else
      using EpilogueCopyInst = AutoVectorizingCopyWithAssumedAlignment<128>;
      static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<DstT>::value;
      using EpilogueConfig = cutlass::gemm::config::DefaultGemm_Epilogue_Configuration<
              EpilogueCopyInst, DstT, AlignmentD, Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_M / WARP_M>, BLOCK_SIZE>;
      using CopyAtomR2S = Copy_Atom<EpilogueCopyInst, DstT>;
      epilogue_with_tsm<AccT, DstT, SHAPE_N, BLOCK_N, STRIDE_CM, EpilogueConfig, CopyAtomR2S>(
        accum, tCcC, cC, tiled_mma, args.c_ptr, smem_buffer, thread_idx,
        deep_scheduler.curr_block_m_offset, deep_scheduler.valid_m_in_block, blk_n_offset);
#endif
      if constexpr (DispatchPipeline && PeerWrite) {
          __threadfence();
          mega_moe_training::peer_compute_sync<SyncThreads>();
          if (threadIdx.x == 0) {
              constexpr uint32_t ntiles = (SHAPE_N + BLOCK_N - 1) / BLOCK_N;
              tix::red_add_rel(pipeline.ready + deep_scheduler.cumsum_m_block_idx * ntiles + n_block_idx, 1u);
              tix::red_add_rel(pipeline.ready + pipeline.signal_capacity, 1u);
          }
          mega_moe_training::peer_compute_sync<SyncThreads>();
      }
    }
#if DG_BF16_RETURN_NCP_MODE
    if constexpr(PeerWrite && StagedReturn) {
        __ppu_wait_remote_vstcnt(0);
        __ppu_wait_vstcnt(0);
        __threadfence_system();
        __syncthreads();
    }
#endif
    if constexpr (DispatchPipeline) {
        if (threadIdx.x == 0 && pipeline.statistics) {
            pipeline.statistics[blockIdx.x * 4] = clock64() - pipeline_begin;
            pipeline.statistics[blockIdx.x * 4 + 1] = deep_scheduler.wait_cycles;
            pipeline.statistics[blockIdx.x * 4 + 2] = 0;
            pipeline.statistics[blockIdx.x * 4 + 3] = 1;
        }
    }
}

} // namespace deep_gemm

#pragma clang diagnostic pop


#ifndef DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_PEER_MOE_INPUT_HELPERS_CUH
#define DEEP_GEMM_TRAINING_MERGED_DEEP_GEMM_INCLUDE_DEEP_GEMM_IMPLS_PEER_MOE_INPUT_HELPERS_CUH
// Included after mega_moe_gemm.cuh's normal implementation. All payloads retain
// their existing POD ABI. Helpers and Tensor Cell consumers share one CTA.
namespace deep_gemm {

template <uint32_t ShapeN, uint32_t ShapeK, uint32_t Groups,
          uint32_t BM, uint32_t BN, uint32_t BK, uint32_t WM, uint32_t WN,
          uint32_t ComputeThreads, int Stages, uint32_t HelperThreads,
          bool KeepGateUp, uint32_t CopyILP, uint32_t MetadataBM = BM>
__device__ __forceinline__ void training_bf16_input_helpers_impl(
        const mega_moe_training::MegaMoEGemmParams params) {
    static_assert(HelperThreads == 32 || HelperThreads == 64);
    static_assert(CopyILP == 1 || CopyILP == 2 || CopyILP == 4 || CopyILP == 8);
    static_assert(ShapeK % 8 == 0);
    static_assert(MetadataBM >= BM && MetadataBM % BM == 0);
    constexpr uint32_t Ratio = MetadataBM / BM;
    constexpr uint32_t TotalThreads = ComputeThreads + HelperThreads;
    const uint32_t tiles = __ldg(params.gemm.aligned_num_m_blocks) * Ratio;
    if (threadIdx.x >= ComputeThreads) {
        const uint32_t helper = threadIdx.x - ComputeThreads;
        __shared__ uint64_t addresses[BM];
        for (uint32_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
            const uint4 record = __ldg(
                reinterpret_cast<const uint4*>(params.gemm.expert_ids_and_cumsum) + tile / Ratio);
            const uint32_t row_base = record.w + (tile / Ratio - record.z) * MetadataBM
                + (tile % Ratio) * BM;
            for (uint32_t row = helper; row < BM; row += HelperThreads) {
                const uint32_t source = __ldg(params.gemm.sorted_token_ids + tile * BM + row);
                uint64_t address = 0;
                if (source < uint64_t(params.num_ranks) * params.source_rows_per_rank) {
                    const uint32_t owner = source / params.source_rows_per_rank;
                    const uint32_t local = source % params.source_rows_per_rank;
                    address = __ldg(params.source_peers + owner)
                        + uint64_t(local) * ShapeK * sizeof(cutlass::bfloat16_t);
                }
                addresses[row] = address;
            }
            __ppu_barrier_sync(2, HelperThreads);
            auto* packed = reinterpret_cast<uint4*>(const_cast<void*>(params.gemm.a_ptr))
                + uint64_t(row_base) * (ShapeK / 8);
            for (uint32_t base = helper; base < BM * (ShapeK / 8);
                 base += HelperThreads * CopyILP) {
                uint4 values[CopyILP];
                #pragma unroll
                for (uint32_t j = 0; j < CopyILP; ++j) {
                    const uint32_t position = base + j * HelperThreads;
                    uint4 value{};
                    if (position < BM * (ShapeK / 8)) {
                        const uint32_t row = position / (ShapeK / 8);
                        const uint32_t column = position % (ShapeK / 8);
                        const uint64_t address = addresses[row];
                        if (address) value = reinterpret_cast<const uint4*>(address)[column];
                    }
                    values[j] = value;
                }
                #pragma unroll
                for (uint32_t j = 0; j < CopyILP; ++j) {
                    const uint32_t position = base + j * HelperThreads;
                    if (position < BM * (ShapeK / 8)) packed[position] = values[j];
                }
            }
            // Every helper completes its global stores before the team handoff.
            __threadfence();
            __ppu_barrier_sync(1, TotalThreads);
            // Consumers never access addresses[]. Join helper warps before
            // one can start replacing another's last-stage source addresses.
            __ppu_barrier_sync(2, HelperThreads);
        }
        return;
    }
    for (uint32_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        __ppu_barrier_sync(1, TotalThreads);
        training_bf16_peer_gemm_impl<
            cutlass::bfloat16_t, GemmType::MGroupedFused,
            ShapeN, ShapeK, Groups, BM, BN, BK, WM, WN, ComputeThreads, Stages,
            false, false,                      // peer reads/writes
            false, false, 0, false, false,      // coherent/activation/cache/staging
            0, 0, true, false,                 // all N, local AIU
            false, false, false, true,         // no pipeline; selected M tile
            4, true, true, KeepGateUp, CopyILP, // G8 FC1/SwiGLU
            false, 0, 0,                      // no external readiness waits
            false, false, false, false,        // no gradient/KN sidecars
            ComputeThreads, MetadataBM        // subgroup and original metadata geometry
        >(params, mega_moe_training::MegaMoEDispatchPipeline{}, tile);
    }
}

} // namespace deep_gemm

#endif
