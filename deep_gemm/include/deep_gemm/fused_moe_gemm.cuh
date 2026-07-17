#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include "utils.cuh"
#include "profiling_interface.hpp"

#include "cute/ppu_tensor_mix.hpp"
#include "cutlass/gemm/config/gemm_operands.hpp"

#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cute/atom/copy_traits_ppu0015_aiu.hpp"
#include "cute/algorithm/ppu_copy.hpp"

#include "fused_scheduler.cuh"
#include "fused_gemm_util.cuh"
#include "utils_cutlass3.h"

using namespace cute;

namespace deep_gemm {

template <class _SrcT, GemmType kGemmType,
          uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t WARP_M, uint32_t WARP_N,
          uint32_t BLOCK_SIZE, int kNumStages>
__global__ __launch_bounds__(BLOCK_SIZE, 1) void
bf16_gemm_fused_moe_kernel(const GemmArgs args) {
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

    using TileScheduler = FusedGemmScheduler<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, kNumGroups>;

    // Shared memory
    using TsmCfg = GemmSmemConfig<SrcT, kNumStages, BLOCK_M, BLOCK_N, BLOCK_K>;
    extern __shared__ __align__(128) uint8_t smem_buffer[];
    SrcT* smem_a = reinterpret_cast<SrcT*>(smem_buffer);
    SrcT* smem_b = reinterpret_cast<SrcT*>(smem_buffer + TsmCfg::kSmemASize);

    uint32_t thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    uint32_t shape_m = args.shape_m;

    // load A from hbm to tsm: use async copy.
    constexpr int Alignment = 128 / cutlass::sizeof_bits<SrcT>::value;
    using ACopyInst = cute::PPU_CP_ASYNC_CACHEALWAYS_ZFILL<cutlass::uint128_t>;
    using GemmOperandA = cutlass::gemm::config::Gemm_Hybrid_Operand<
            ArchTag, SrcT, false, Alignment, cute::Int<BLOCK_K>, BLOCK_SIZE,
            ACopyInst, cute::Int<BLOCK_M>>;

    using TilerA = typename GemmOperandA::GmemTiledCopy::Tiler_MN;
    using SmemLayoutA = decltype(tile_to_shape(typename GemmOperandA::SmemLayoutAtom{},
            Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_K>, cute::Int<kNumStages>>{}));

    typename GemmOperandA::GmemTiledCopy gmem_tiled_copy_A;
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    Tensor sA = cute::make_tensor(cute::make_smem_ptr(smem_a), SmemLayoutA{});
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);

    // load B from hbm to tsm: use aiu load.
    using GemmOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<
            ArchTag, SrcT, false, cute::Int<BLOCK_N>, cute::Int<BLOCK_K>, true>;
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
    TileScheduler deep_scheduler(args.aligned_num_m_blocks, args.expert_ids_and_cumsum);

    #pragma clang loop licm(disable)
    while (deep_scheduler.fetch_next_work(m_block_idx, n_block_idx)) {
      auto blk_coord_mnkl = make_coord(m_block_idx, n_block_idx, _, _1{});
      uint32_t blk_n_offset = n_block_idx * BLOCK_N;
      const int* blk_token_base = args.sorted_token_ids + deep_scheduler.cumsum_m_block_idx * BLOCK_M;
      // gmem_b in block
      SrcT* gmem_b = (SrcT*)args.b_ptr + deep_scheduler.curr_group_idx * STRIDE_BE;
      Tensor mB_nk = cute::make_tensor(cute::make_gmem_ptr(gmem_b), shape_B, stride_B);
      Tensor gB = cute::local_tile(cute::make_mix_tensor_like(mB_nk), TileShape{}, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});           // (BLK_N,BLK_K,k)
      Tensor tBgB = gmem_thr_copy_B.partition_S(gB);

      int k_tile_iter  = 0;
      int k_tile_count = size<2>(gB);

      CUTLASS_PRAGMA_UNROLL
      for (int k_pipe = 0; k_pipe < kNumStages; ++k_pipe) {
        if (k_tile_count > 0) {
          copy_A_to_tsm<SrcT, ACopyInst, TilerA, BLOCK_M, BLOCK_K, STRIDE_AM>(
              tAsA(_,_,_,k_pipe), args.a_ptr,
              blk_token_base, BLOCK_K * k_tile_iter, shape_m, thread_idx);
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
            copy_A_to_tsm<SrcT, ACopyInst, TilerA, BLOCK_M, BLOCK_K, STRIDE_AM>(
              tAsA(_,_,_,smem_pipe_write), args.a_ptr,
              blk_token_base, BLOCK_K * k_tile_iter, shape_m, thread_idx);
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
      epilogue_no_tsm<AccT, DstT, SHAPE_N, BLOCK_N, STRIDE_CM>(accum, tCcC, args.c_ptr,
          deep_scheduler.curr_block_m_offset, deep_scheduler.valid_m_in_block, blk_n_offset);
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
    }
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t WARP_M, uint32_t WARP_N, int32_t kNumStages,
          GemmType kGemmType, bool kEnableSboOverlap = false,
          KernelType kKernelType = KernelType::Default>
class FusedMoeGemm {
    using SrcT = __ppu_bfloat16;
    using DstT = __ppu_bfloat16;

public:
    FusedMoeGemm() = default;

    static void run(DstT* gmem_d, SrcT* gmem_a, SrcT* gmem_b,
                    int* m_rows, int* expert_ids_and_cumsum, int* sorted_token_ids,
                    int* aligned_num_m_blocks, uint32_t shape_m,
                    hggcStream_t stream, int num_sms) {

        GemmArgs args;

        args.a_ptr = (void *)gmem_a;
        args.b_ptr = (void *)gmem_b;
        args.c_ptr = (void *)gmem_d;

        args.expert_ids_and_cumsum = expert_ids_and_cumsum;
        args.sorted_token_ids = sorted_token_ids;
        args.aligned_num_m_blocks = aligned_num_m_blocks;
        args.shape_m = shape_m;

        DgProfParam dg_prof_params;
        // check src type
        std::string data_type = "bf16";
        if (ProfilingInterface::Instance().get_op_info()){
            dg_prof_params.set_params(
                GemmType::GroupedFused, false, data_type, kNumGroups, shape_m, SHAPE_N, SHAPE_K, 1,
                m_rows, stream
            );
        }
        // dispatch and launch kernel
        constexpr int BlockSize = BLOCK_M / WARP_M * BLOCK_N / WARP_N * 32;

        auto device_func = bf16_gemm_fused_moe_kernel<SrcT, kGemmType, SHAPE_N, SHAPE_K, kNumGroups, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BlockSize, kNumStages>;
        constexpr uint32_t smem_size = GemmSmemConfig<SrcT, kNumStages, BLOCK_M, BLOCK_N, BLOCK_K>::kTotalSize;
        CHECK_HGGC(hggcFuncSetAttribute(device_func, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        int max_blocks_per_cu = -1;
        CHECK_HGGC(hggcOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_cu, device_func, BlockSize, smem_size));

        int sm_count = num_sms * max_blocks_per_cu;
        dim3 grid(sm_count, 1, 1);

        const char* pEnv_params = std::getenv("show_log");
        if (pEnv_params && std::atoi(pEnv_params) == 1) {
            hggcFuncAttributes attr;
            hggcFuncGetAttributes(&attr, device_func);

            printf("[FusedMoeGemm-BF16:]\n");
            printf("group:%d, problem:[%d, %d, %d], gemm_type:%s, kernel_type:%s\n",
                kNumGroups, shape_m, SHAPE_N, SHAPE_K, GemmTypeS[static_cast<int>(kGemmType)], KernelTypeS[static_cast<int>(kKernelType)]);

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
