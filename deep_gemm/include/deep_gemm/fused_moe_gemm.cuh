#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#define WARP_SIZE 32
#define HGGC_PRAGMA_UNROLL _Pragma("unroll")
#define HGGC_DEVICE_ONLY __forceinline__ __device__

#include <iostream>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "utils.cuh"
#include "profiling_interface.hpp"

#include "ppu/cute/tensor_mix.hpp"
#include "ppu/gemm/config/gemm_operands.hpp"

#include "ppu/cute/atom/mma_traits_acompute10000.hpp"
#include "ppu/cute/atom/mma_traits_acompute10500.hpp"
#include "ppu/cute/atom/copy_traits_acompute10000_aiu.hpp"
#include "ppu/cute/atom/copy_traits_acompute10500_aiu.hpp"
#include "ppu/cute/algorithm/copy.hpp"

#include "scheduler_cutlass3.cuh"
#include "utils_cutlass3.h"
#include "cuda_ad.h"

using namespace cute;

namespace deep_gemm {

struct GemmArgs {
    void * a_ptr;
    void * b_ptr;
    void * c_ptr;

    int64_t shape_m;

    int* grouped_layout;
    int* sorted_token_ids;  // permute maps
    int* expert_ids;        // block load which expert
    int* block_m_offset;    // real block_m offset without padding

    //optimized to constexpr
    // int64_t stride_am;
    // int64_t stride_ak;
    // int64_t stride_be;
    // int64_t stride_bk;
    // int64_t stride_bn;
    // int64_t stride_cm;
    // int64_t stride_cn;
};

template <class TileScheduler, class _SrcT,
          uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t WARP_M, uint32_t WARP_N,
          uint32_t BLOCK_SIZE, int kNumStages>
__global__ __launch_bounds__(BLOCK_SIZE, 1) void
bf16_fused_moe_gemm_impl(const GemmArgs args) {
    constexpr uint32_t STRIDE_AM = SHAPE_K;
    constexpr uint32_t STRIDE_BE = SHAPE_N * SHAPE_K;
    // constexpr uint32_t STRIDE_BN = SHAPE_K;   // aiu load only need mB.
    constexpr uint32_t STRIDE_CM = SHAPE_N;
    using SrcT = typename ToCutlassType<_SrcT>::type;
    using DstT = cutlass::bfloat16_t;
    using AccT = float;
    using TileShape = cute::Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_N>, cute::Int<BLOCK_K>>;
    using WarpShape = cute::Shape<cute::Int<WARP_M>, cute::Int<WARP_N>, cute::Int<BLOCK_K>>;

    using MmaInst = typename cutlass::gemm::config::GetMmaInst<SrcT, SrcT, AccT>::type;
    using TiledMma = cute::TiledMMA<
      cute::MMA_Atom<MmaInst>,
      cute::Layout<Shape< Int<BLOCK_M / WARP_M>, Int<BLOCK_N / WARP_N>, _1>>>;

    // ppu1.5 tsm.ld.swzl need 128B aligned
    constexpr uint32_t SMEM_A_SIZE = cute::round_up(kNumStages * BLOCK_M * BLOCK_K * sizeof(SrcT), 128);

    // Shared memory
    extern __shared__ __align__(128) uint8_t smem_buffer[];
    SrcT* smem_a = reinterpret_cast<SrcT*>(smem_buffer);
    SrcT* smem_b = reinterpret_cast<SrcT*>(smem_buffer + SMEM_A_SIZE);

    uint32_t thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();

    // load A from hbm to tsm: use async copy.
    constexpr int Alignment = 128 / cutlass::sizeof_bits<SrcT>::value;
    using ACopyInst = cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<cutlass::uint128_t>;
    using GemmOperandA = cutlass::gemm::config::DefaultGemm_TensorOpSm80_Operand<
            SrcT, false, Alignment, cute::Int<BLOCK_K>, BLOCK_SIZE,
            ACopyInst>;

    using TilerA = typename GemmOperandA::GmemTiledCopy::Tiler_MN;
    using SmemLayoutA = decltype(tile_to_shape(typename GemmOperandA::SmemLayoutAtom{},
            Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_K>, cute::Int<kNumStages>>{}));

    typename GemmOperandA::GmemTiledCopy gmem_tiled_copy_A;
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    Tensor sA = cute::make_tensor(cute::make_smem_ptr(smem_a), SmemLayoutA{});
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);

    // load B from hbm to tsm: use aiu load.
    using GemmOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<
            SrcT, false, cute::Int<BLOCK_N>, cute::Int<BLOCK_K>, true>;
    using SmemLayoutB = decltype(tile_to_shape(typename GemmOperandB::SmemLayoutAtom{},
            Shape<cute::Int<BLOCK_N>, cute::Int<BLOCK_K>, cute::Int<kNumStages>>{}));

    // init aiu desc
    typename GemmOperandB::GmemTiledCopy gmem_tiled_copy_B;
    using TilerB = typename GemmOperandB::GmemTiledCopy::Tiler_MN;
    auto shape_B = cute::make_shape(Int<SHAPE_N>{}, Int<SHAPE_K>{});
    auto stride_B = cute::make_shape((int)SHAPE_K, _1{});
    gmem_tiled_copy_B.desc_.template init<SrcT, false, get<0>(TilerB{}), get<1>(TilerB{})>(nullptr, SHAPE_N, SHAPE_K, stride_B);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
    Tensor sB = cute::make_tensor(cute::make_smem_ptr(smem_b), SmemLayoutB{});
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);

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
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S((sA));                  // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                   // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(warp_idx * 32);
    Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));                  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K


    uint32_t num_tokens = args.shape_m * SHAPE_K;
    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    TileScheduler deep_scheduler(typename TileScheduler::Params(args.shape_m, args.grouped_layout), 0);

    #pragma clang loop licm(disable)
    while (deep_scheduler.fetch_next_work(m_block_idx, n_block_idx)) {
      auto blk_coord_mnkl = make_coord(m_block_idx, n_block_idx, _, _1{});
      uint32_t blk_n_offset = n_block_idx * BLOCK_N;
      // gmem_b in block
      SrcT* gmem_b = (SrcT*)args.b_ptr + __ldg(args.expert_ids + m_block_idx) * STRIDE_BE;
      Tensor mB_nk = cute::make_tensor(cute::make_gmem_ptr(gmem_b), shape_B, stride_B);
      Tensor mB_nk_mix = cute::make_mix_tensor_like(mB_nk);
      Tensor gB = cute::local_tile(mB_nk_mix, TileShape{}, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});           // (BLK_N,BLK_K,k)
      Tensor tBgB = gmem_thr_copy_B.partition_S(gB);

      int k_tile_iter  = 0;
      int k_tile_count = size<2>(gB);

      auto copy_A_to_tsm = [&](int pipe_write, uint32_t stage_offset) {
        static constexpr uint32_t KPerThread = 128 / cutlass::sizeof_bits<SrcT>::value;
        static constexpr uint32_t NumThreads_CPY_K = get<1>(TilerA{}) / KPerThread; // max cont is 128B
        static constexpr uint32_t M_ITER = BLOCK_M / get<0>(TilerA{});
        static constexpr uint32_t K_ITER = BLOCK_K / get<1>(TilerA{});
        CUTLASS_PRAGMA_UNROLL
        for(uint32_t m_iter = 0; m_iter < M_ITER; ++m_iter) {
          CUTLASS_PRAGMA_UNROLL
          for(uint32_t k_iter = 0; k_iter < K_ITER; ++k_iter) {
            uint32_t tid_k = thread_idx % NumThreads_CPY_K;
            uint32_t tid_m = thread_idx / NumThreads_CPY_K;
            uint32_t token_offset = __ldg(args.sorted_token_ids + m_block_idx * BLOCK_M + (m_iter * get<0>(TilerA{})) + tid_m) * STRIDE_AM
                                  + (k_iter * get<1>(TilerA{})) + tid_k * KPerThread + stage_offset;
            bool token_mask = token_offset < num_tokens;
            cutlass::uint128_t* src_ptr = reinterpret_cast<cutlass::uint128_t*>((SrcT*)args.a_ptr + token_offset);
            cutlass::uint128_t* dst_ptr = reinterpret_cast<cutlass::uint128_t*>(raw_pointer_cast(tAsA(_,m_iter,k_iter,pipe_write).data()));
            ACopyInst::copy(*src_ptr, *dst_ptr, token_mask);
          }
        }
      };
      CUTLASS_PRAGMA_UNROLL
      for (int k_pipe = 0; k_pipe < kNumStages; ++k_pipe) {
        if (k_tile_count > 0) {
          copy_A_to_tsm(k_pipe, BLOCK_K * k_tile_iter);
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
        __syncthreads();

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
          __syncthreads();
          if (k_tile_count > 0) {
            copy_A_to_tsm(smem_pipe_write, BLOCK_K * k_tile_iter);
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
      DstT* gC_ptr = (DstT*)args.c_ptr + __ldg(args.block_m_offset + m_block_idx) * STRIDE_CM + blk_n_offset;
      auto blk_mn_shape = Shape<Int<BLOCK_M>, Int<BLOCK_N>>{};
      Tensor gC = cute::make_tensor(cute::make_gmem_ptr(gC_ptr), blk_mn_shape, cute::make_stride(Int<SHAPE_N>{}, _1{}));
      // Partition source and destination tiles to match the accumulator partitioning
      Tensor tCgC = thr_mma.partition_C(gC);
      CUTE_STATIC_ASSERT_V(size(tCgC) == size(accum),
          "Accumulator count must have the same destination element count.");

      Tensor cC = make_identity_tensor(blk_mn_shape);
      Tensor tCcC = thr_mma.partition_C(cC);
      // auto epilogue_NoTsm
      // epilogue_no_tsm only available for ppu1.5
      auto epilogue_no_tsm = [&]() {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCgC); i += 2) {
          bool cond = __ldg(args.sorted_token_ids + m_block_idx * BLOCK_M + get<0>(tCcC(i))) < args.shape_m;
          if constexpr (SHAPE_N % BLOCK_N) {
            cond = cond && get<1>(tCcC(i)) < (SHAPE_N - blk_n_offset);
          }
          if (cond) {
            AccT* acc_ptr = raw_pointer_cast(accum.data()) + accum.layout()(i);
            uint32_t* dst_ptr = (uint32_t*)((raw_pointer_cast(tCgC.data()) + tCgC.layout()(i)));
            uint32_t d;
            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(acc_ptr[1]), "f"(acc_ptr[0]));
            *dst_ptr = d;
          }
        }
      };

      auto epilogue_with_tsm = [&]() {
        using EpilogueCopyInst = AutoVectorizingCopyWithAssumedAlignment<128>;
        static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<DstT>::value;
        using EpilogueConfig = cutlass::gemm::config::DefaultGemm_Epilogue_Configuration<
                EpilogueCopyInst, AccT, AlignmentD, Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_M / WARP_M>, BLOCK_SIZE>;
        using SmemLayoutO = typename EpilogueConfig::SmemLayoutO;
        using CopyAtomR2S = Copy_Atom<EpilogueCopyInst, AccT>;
        using TiledCopyS2R = typename EpilogueConfig::GmemTiledCopyO;
        using CopyAtomR2G = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(DstT) * AlignmentD * 8>, DstT>;
        Tensor sAcc = make_tensor(make_smem_ptr(reinterpret_cast<AccT*>(smem_buffer)), SmemLayoutO{});

        // Partition sAcc to match the accumulator partitioning
        auto tiled_r2s = make_tiled_copy_C(CopyAtomR2S{}, tiled_mma);
        auto thread_r2s     = tiled_r2s.get_thread_slice(thread_idx);
        Tensor tRS_rAcc = thread_r2s.retile_S(accum);                        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor tRS_sAcc = thread_r2s.partition_D(sAcc);                      // ((Atom,AtomNum),PIPE_M,PIPE_N)

        // Tile gC by the shape of SmemLayout first
        auto tile  = make_shape(size<0>(sAcc), size<1>(sAcc));
        Tensor gCt = flat_divide(gC, tile);                                  // (SMEM_M,SMEM_N,TILE_M,TILE_N)

        // Partition sAcc, gC for the output
        auto tiled_s2r = TiledCopyS2R{};
        auto thread_s2r     = tiled_s2r.get_thread_slice(thread_idx);
        Tensor tSR_sAcc = thread_s2r.partition_S(sAcc);                      //               ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tSR_gC = thread_s2r.partition_D(gCt);                         // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

        // Allocate intermediate registers on the dst tensors
        Tensor tSR_rAcc = make_tensor<AccT>(take<0,3>(shape(tSR_gC)));       // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tSR_rC = make_tensor<DstT>(shape(tSR_rAcc));                  // ((Atom,AtomNum),ATOM_M,ATOM_N)

        // Repeat the D-partitioning for coordinates and predication
        Tensor cCt  = flat_divide(cC, tile);                                 //                (SMEM_M,SMEM_N,TILE_M,TILE_N)
        Tensor tSR_cC = thread_s2r.partition_D(cCt);                         // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

        CUTE_STATIC_ASSERT(size<1>(tRS_rAcc) % size<3>(tSR_gC) == 0);  // TILE_M divides MMA_M
        CUTE_STATIC_ASSERT(size<2>(tRS_rAcc) % size<4>(tSR_gC) == 0);  // TILE_N divides MMA_N


        CUTLASS_PRAGMA_UNROLL
        for (int step_m = 0; step_m < size<2>(cCt); ++step_m)
        {
          CUTLASS_PRAGMA_UNROLL
          for (int step_n = 0; step_n < size<3>(cCt); ++step_n)
          {
            // Step 1. Copy to SMEM
            CUTLASS_PRAGMA_UNROLL
            for (int pipe_m = 0; pipe_m < size<1>(tRS_sAcc); ++pipe_m) {
              CUTLASS_PRAGMA_UNROLL
              for (int pipe_n = 0; pipe_n < size<2>(tRS_sAcc); ++pipe_n) {
                int mma_m = step_m * size<1>(tRS_sAcc) + pipe_m;
                int mma_n = step_n * size<2>(tRS_sAcc) + pipe_n;
                copy(tiled_r2s, tRS_rAcc(_,mma_m,mma_n), tRS_sAcc(_,pipe_m,pipe_n));
              }
            }
            // Step 2. Wait for SMEM writes to complete
            __syncthreads();

            // Step 3. Copy from SMEM into a fragment
            copy(tiled_s2r, tSR_sAcc, tSR_rAcc);

            // Step 4. Wait for SMEM reads to complete
            __syncthreads();

            Tensor tSR_gDmn = tSR_gC(_,_,_,step_m,step_n);
            Tensor tSR_cDmn = tSR_cC(_,_,_,step_m,step_n);

            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < size<1>(tSR_gDmn); ++m) {
              CUTLASS_PRAGMA_UNROLL
              for (int n = 0; n < size<2>(tSR_gDmn); ++n) {
                // Predication
                bool cond = __ldg(args.sorted_token_ids + m_block_idx * BLOCK_M + get<0>(tSR_cDmn(0,m,n))) < args.shape_m;
                if constexpr (SHAPE_N % BLOCK_N) {
                  cond = cond && get<1>(tSR_cDmn(0,m,n)) < (SHAPE_N - blk_n_offset);
                }
                if (cond) {
                  // The Last Step. Copy to GMEM
                  copy(CopyAtomR2G{}, tSR_rAcc(_,m,n), tSR_gDmn(_,m,n));
                }
              }
            }
          }
        }
      };
      // epilogue_no_tsm();
      epilogue_with_tsm();
    }
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t WARP_M, uint32_t WARP_N, int32_t kNumStages,
          GemmType kGemmType, bool kEnableSboOverlap = false,
          KernelType kKernelType = KernelType::Default>
class FusedMoeGemm {
    using SrcT = __nv_bfloat16;
    using DstT = __nv_bfloat16;

    using TileScheduler = DeepGemmScheduler<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, kNumGroups>;

public:
    FusedMoeGemm() = default;

    static void run(DstT* gmem_d, int* grouped_layout, int* block_m_info,
                    int* sorted_token_ids, int* expert_ids, int* block_m_offset,
                    uint32_t shape_m, SrcT* gmem_a, SrcT* gmem_b,
                    cudaStream_t stream, int num_sms) {

        GemmArgs args;

        args.a_ptr = (void *)gmem_a;
        args.b_ptr = (void *)gmem_b;
        args.c_ptr = (void *)gmem_d;

        args.grouped_layout = grouped_layout;
        args.sorted_token_ids = sorted_token_ids;
        args.expert_ids = expert_ids;
        args.block_m_offset = block_m_offset;
        args.shape_m = shape_m;

        DgProfParam dg_prof_params;
        // check src type
        std::string data_type = "bf16";
        if (ProfilingInterface::Instance().get_op_info()){
            dg_prof_params.set_params(
                GemmType::GroupedFused, false, data_type, kNumGroups, shape_m, SHAPE_N, SHAPE_K, 1,
                grouped_layout, stream
            );
        }
        // compute block_m_info
        if (TileScheduler::kIsNoPadPreprocessLayout) {
            uint32_t block_size = max(32, next_power_of_two(kNumGroups));
            computeBlockInfoKernel<BLOCK_M><<<1, block_size, 0, stream>>>(reinterpret_cast<const uint32_t*>(grouped_layout), kNumGroups, reinterpret_cast<uint32_t*>(block_m_info));
            args.grouped_layout = block_m_info;
        }

        // dispatch and launch kernel
        constexpr int BlockSize = BLOCK_M / WARP_M * BLOCK_N / WARP_N * 32;

        auto device_func = bf16_fused_moe_gemm_impl<TileScheduler, SrcT, SHAPE_N, SHAPE_K, kNumGroups, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BlockSize, kNumStages>;
        // ppu1.5 tsm.ld.swzl need 128B aligned
        constexpr uint32_t SMEM_A_SIZE = cute::round_up(kNumStages * BLOCK_M * BLOCK_K * sizeof(SrcT), 128);
        constexpr uint32_t SMEM_B_SIZE = cute::round_up(kNumStages * BLOCK_N * BLOCK_K * sizeof(SrcT), 128);
        constexpr uint32_t smem_size =  SMEM_A_SIZE + SMEM_B_SIZE;
        CHECK_CUDA(cudaFuncSetAttribute(device_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        int max_blocks_per_cu = -1;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_cu, device_func, BlockSize, smem_size));

        int sm_count = num_sms * max_blocks_per_cu;
        dim3 grid(sm_count, 1, 1);

        char *pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, device_func);

            printf("[FusedMoeGemm-BF16:]\n");
            printf("group:%d, problem:[%d, %d, %d], gemm_type:%s, kernel_type:%s, kIsNoPadPreprocessLayout: %d\n",
                kNumGroups, shape_m, SHAPE_N, SHAPE_K, GemmTypeS[static_cast<int>(kGemmType)], KernelTypeS[static_cast<int>(kKernelType)], TileScheduler::kIsNoPadPreprocessLayout);

            printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], kNumStages:%d\n",
                BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BLOCK_K, kNumStages);

            printf("grid:%d, vreg:%d, smem_size: %d, tb_per_cu:%d, stack:%d\n",
                sm_count, int(attr.numRegs), smem_size, max_blocks_per_cu, int(attr.localSizeBytes));
        }
        ProfilingInterface::Instance().instrument(true, dg_prof_params);
        device_func<<<grid, BlockSize, smem_size, stream>>>(args);
        ProfilingInterface::Instance().instrument(false, dg_prof_params);
    }
};


} // namespace deep_gemm


#pragma clang diagnostic pop
