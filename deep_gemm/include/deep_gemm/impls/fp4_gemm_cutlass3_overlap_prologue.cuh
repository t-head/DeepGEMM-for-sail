#pragma once

#include <type_traits>
#include <deep_gemm/impls/fp4_gemm_cutlass3.cuh>

namespace cutlass::gemm::collective {

using namespace cute;

template <class Base_>
struct Fp4CollectiveMmaOverlapPrologue : Base_ {
  using Base = Base_;
  using Params = typename Base::Params;
  using SharedStorage = typename Base::SharedStorage;
  using DispatchPolicy = typename Base::DispatchPolicy;
  using TiledMma = typename Base::TiledMma;
  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutB = typename Base::SmemLayoutB;
  using SmemLayoutSFA = typename Base::SmemLayoutSFA;
  using SmemLayoutSFB = typename Base::SmemLayoutSFB;
  using SmemCopyAtomA = typename Base::SmemCopyAtomA;
  using SmemCopyAtomB = typename Base::SmemCopyAtomB;
  using TransformA = typename Base::TransformA;
  using TransformB = typename Base::TransformB;

  using Base::params_;
  using Base::gmem_tiled_copy_A;
  using Base::gmem_tiled_copy_B;
  using Base::gmem_tiled_copy_SFA;
  using Base::gmem_tiled_copy_SFB;
  using Base::warp_on_m;
  using Base::SplitAIU;
  using Base::AIU_SFA_WARP;
  using Base::AIU_SFB_WARP;
  using Base::cal_ldmat_offset;
  using Base::sf_s2r;

  template <class ProblemShape>
  CUTLASS_DEVICE
  Fp4CollectiveMmaOverlapPrologue(Params params, ProblemShape problem_shape_MNKL, int batch_idx = 0)
      : Base(params, problem_shape_MNKL, batch_idx) {}

  template <class... Ts>
  CUTLASS_DEVICE void
  prologue(cute::tuple<Ts...> const& load_inputs,
           int& k_tile_iter, int& k_tile_count,
           int thread_idx, char* smem_buf) {
    int warp_idx = canonical_warp_idx_sync();

    Tensor gA = get<0>(load_inputs);
    Tensor gB = get<1>(load_inputs);
    Tensor gSFA = get<2>(load_inputs);
    Tensor gSFB = get<3>(load_inputs);

    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{});
    Tensor sSFA = make_tensor(make_smem_ptr(storage.smem_sfa.data()), SmemLayoutSFA{});
    Tensor sSFB = make_tensor(make_smem_ptr(storage.smem_sfb.data()), SmemLayoutSFB{});

    CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(sA));
    CUTE_STATIC_ASSERT_V(size<1>(gA) == size<1>(sA));
    CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sB));
    CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sB));
    CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));

    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
    auto gmem_thr_copy_SFA = gmem_tiled_copy_SFA.get_thread_slice(thread_idx);
    auto gmem_thr_copy_SFB = gmem_tiled_copy_SFB.get_thread_slice(thread_idx);

    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);
    Tensor tSFAgSFA = gmem_thr_copy_SFA.partition_S(gSFA);
    Tensor tSFAsSFA = gmem_thr_copy_SFA.partition_D(sSFA);
    Tensor tSFBgSFB = gmem_thr_copy_SFB.partition_S(gSFB);
    Tensor tSFBsSFB = gmem_thr_copy_SFB.partition_D(sSFB);

    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages; ++k_pipe) {
      if (k_tile_count > 0) {
        auto k_idx = k_tile_iter;
        copy_aiu<SplitAIU>(
          gmem_tiled_copy_A, tAgA(_,_,_,k_idx), tAsA(_,_,_,k_pipe),
          gmem_tiled_copy_B, tBgB(_,_,_,k_idx), tBsB(_,_,_,k_pipe),
          warp_idx
        );
        if (warp_idx == AIU_SFA_WARP) {
          copy(gmem_tiled_copy_SFA, tSFAgSFA(_,_,_,k_idx), tSFAsSFA(_,_,_,k_pipe));
        }
        if (warp_idx == AIU_SFB_WARP) {
          copy(gmem_tiled_copy_SFB, tSFBgSFB(_,_,_,k_idx), tSFBsSFB(_,_,_,k_pipe));
        }
        ++k_tile_iter;
      }
      cp_async_fence();
      --k_tile_count;
    }
  }

  template <class... Ts, class FrgTensorD, class FrgTensorC,
            class ResidueMNK>
  CUTLASS_DEVICE void
  operator()(FrgTensorD& accum,
             cute::tuple<Ts...> const& load_inputs,
             FrgTensorC const& src_accum,
             int k_tile_iter, int k_tile_count,
             ResidueMNK residue_mnk,
             int thread_idx, char* smem_buf) {
    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");
    static_assert(rank(SmemLayoutB{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");

    int warp_idx = canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;
    int aiu_warp_group_thread_idx = warp_idx * 32;
    int warp_m_idx = warp_idx % warp_on_m;
    int warp_n_idx = warp_idx / warp_on_m;

    Tensor gA = get<0>(load_inputs);
    Tensor gB = get<1>(load_inputs);

    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{});

    CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(sA));
    CUTE_STATIC_ASSERT_V(size<1>(gA) == size<1>(sA));
    CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sB));
    CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sB));
    CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));

    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);

    Tensor gSFA = get<2>(load_inputs);
    Tensor gSFB = get<3>(load_inputs);
    Tensor sSFA = make_tensor(make_smem_ptr(storage.smem_sfa.data()), SmemLayoutSFA{});
    Tensor sSFB = make_tensor(make_smem_ptr(storage.smem_sfb.data()), SmemLayoutSFB{});

    auto gmem_thr_copy_SFA = gmem_tiled_copy_SFA.get_thread_slice(thread_idx);
    auto gmem_thr_copy_SFB = gmem_tiled_copy_SFB.get_thread_slice(thread_idx);
    Tensor tSFAgSFA = gmem_thr_copy_SFA.partition_S(gSFA);
    Tensor tSFAsSFA = gmem_thr_copy_SFA.partition_D(sSFA);
    Tensor tSFBgSFB = gmem_thr_copy_SFB.partition_S(gSFB);
    Tensor tSFBsSFB = gmem_thr_copy_SFB.partition_D(sSFB);

    auto copy_to_tsm = [&](int k_pipe_write, int k_idx, int warp_idx) {
      copy_aiu<SplitAIU>(
        gmem_tiled_copy_A, tAgA(_,_,_,k_idx), tAsA(_,_,_,k_pipe_write),
        gmem_tiled_copy_B, tBgB(_,_,_,k_idx), tBsB(_,_,_,k_pipe_write),
        warp_idx
      );
      if (warp_idx == AIU_SFA_WARP) {
        copy(gmem_tiled_copy_SFA, tSFAgSFA(_,_,_,k_idx), tSFAsSFA(_,_,_,k_pipe_write));
      }
      if (warp_idx == AIU_SFB_WARP) {
        copy(gmem_tiled_copy_SFB, tSFBgSFB(_,_,_,k_idx), tSFBsSFB(_,_,_,k_pipe_write));
      }
    };

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(src_accum));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(src_accum));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    smem_tiled_copy_A.smem_base_ = storage.smem_a.data();
    auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsA = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    smem_tiled_copy_B.smem_base_ = storage.smem_b.data();
    auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsB = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));

    constexpr auto warp_iter_num_sfa = size<1>(tCsA);
    constexpr auto block_iter_num_sfa = size<2>(tCsA);
    constexpr auto warp_iter_stride_sfa = [&]() constexpr {
      if constexpr (cute::is_scaled_basis<decltype(stride<1>(tCsA))>::value) {
        return stride<1>(tCsA).value();
      } else {
        return stride<1>(tCsA);
      }
    }();
    constexpr int ldmat_iter_num_sfa = cute::ceil_div(warp_iter_num_sfa, Int<4>{});
    Tensor tCrSFA = make_tensor<uint32_t>(Shape<Int<1>, Int<ldmat_iter_num_sfa>, Int<block_iter_num_sfa>>{});
    static_assert(warp_iter_num_sfa <= 4, "warp_iter_num_sfa > 4 is not valid now.");

    constexpr auto warp_iter_num_sfb = size<1>(tCsB);
    constexpr auto block_iter_num_sfb = size<2>(tCsB);
    constexpr auto warp_iter_stride_sfb = [&]() constexpr {
      if constexpr (cute::is_scaled_basis<decltype(stride<1>(tCsB))>::value) {
        return stride<1>(tCsB).value();
      } else {
        return stride<1>(tCsB);
      }
    }();
    constexpr int ldmat_iter_num_sfb = cute::ceil_div(warp_iter_num_sfb, Int<4>{});
    Tensor tCrSFB = make_tensor<uint32_t>(Shape<Int<1>, Int<ldmat_iter_num_sfb>, Int<block_iter_num_sfb>>{});
    static_assert(warp_iter_num_sfb <= 4, "warp_iter_num_sfb > 4 is not valid now.");

    int smem_pipe_read = 0;
    int smem_pipe_write = 0;
    Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

    auto K_BLOCK_MAX = size<2>(tCrA);
    static_assert(K_BLOCK_MAX > 1, "K_BLOCK_MAX should be large than 1.");

    int base_offset_sfa = 0;
    int base_offset_sfb = 0;
    auto smem_layout_sfa = SmemLayoutSFA{};
    auto smem_layout_sfb = SmemLayoutSFB{};
    cal_ldmat_offset(base_offset_sfa, smem_layout_sfa, lane_idx, warp_m_idx,
                     Int<warp_iter_num_sfa>{}, Int<warp_iter_stride_sfa>{});
    cal_ldmat_offset(base_offset_sfb, smem_layout_sfb, lane_idx, warp_n_idx,
                     Int<warp_iter_num_sfb>{}, Int<warp_iter_stride_sfb>{});

    if (K_BLOCK_MAX > 1) {
      cp_async_wait<DispatchPolicy::Stages-1>();
      __syncthreads();
      copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
      copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
      sf_s2r(sSFA, tCrSFA, base_offset_sfa, lane_idx, Int<0>{}, 0);
      sf_s2r(sSFB, tCrSFB, base_offset_sfb, lane_idx, Int<0>{}, 0);
    }

    auto process_kblock_iterations = [&](auto k_block) {
      if constexpr (k_block == K_BLOCK_MAX - 1) {
        tCsA_p = tCsA(_,_,_,smem_pipe_read);
        tCsB_p = tCsB(_,_,_,smem_pipe_read);
        copy(smem_tiled_copy_A, tCsA_p(_,_,_0{}), tCrA_copy_view(_,_,_0{}));
        copy(smem_tiled_copy_B, tCsB_p(_,_,_0{}), tCrB_copy_view(_,_,_0{}));
        sf_s2r(sSFA, tCrSFA, base_offset_sfa, lane_idx, Int<0>{}, smem_pipe_read);
        sf_s2r(sSFB, tCrSFB, base_offset_sfb, lane_idx, Int<0>{}, smem_pipe_read);
      } else {
        auto k_block_next = k_block + Int<1>{};
        copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
        sf_s2r(sSFA, tCrSFA, base_offset_sfa, lane_idx, k_block_next, smem_pipe_read);
        sf_s2r(sSFB, tCrSFB, base_offset_sfb, lane_idx, k_block_next, smem_pipe_read);
      }
      cute::transform(tCrA(_,_,k_block), TransformA{});
      cute::transform(tCrB(_,_,k_block), TransformB{});
      cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrSFA(_,_,k_block), tCrB(_,_,k_block), tCrSFB(_,_,k_block), src_accum);
      if constexpr (k_block == K_BLOCK_MAX - 2) {
        cp_async_wait<DispatchPolicy::Stages-2>();
        __syncthreads();
        if (k_tile_count > 0) {
          copy_to_tsm(smem_pipe_write, k_tile_iter, warp_idx);
          ++k_tile_iter;
        }
        cp_async_fence();
        ++smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == DispatchPolicy::Stages) ? 0 : smem_pipe_read;
        smem_pipe_write = smem_pipe_read;
      }
    };

    for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
      process_kblock_iterations(k_block);
    });
    --k_tile_count;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > -(DispatchPolicy::Stages); --k_tile_count) {
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
        process_kblock_iterations(k_block);
      });
    }

    cp_async_wait<0>();
    __syncthreads();
  }
};

}

namespace deep_gemm {

template <class ProblemShape_, class CollectiveMainloop_,
          class CollectiveEpilogue_, class TileScheduler_>
class Fp4OverlapPrologue
    : public DeepGemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_,
                               TileScheduler_, false, 1, false, EpilogueType::Default> {
public:
  using Base = DeepGemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_,
                                 TileScheduler_, false, 1, false, EpilogueType::Default>;
  using Params = typename Base::Params;
  using SharedStorage = typename Base::SharedStorage;
  using CollectiveMainloop = typename Base::CollectiveMainloop;
  using CollectiveEpilogue = typename Base::CollectiveEpilogue;
  using TileScheduler = typename Base::TileScheduler;
  using MainloopParams = typename Base::MainloopParams;
  using TileShape = typename Base::TileShape;
  using TiledMma = typename Base::TiledMma;
  using ElementA = typename Base::ElementA;
  using ElementB = typename Base::ElementB;
  using ElementSFA = typename Base::ElementSFA;
  using ElementSFB = typename Base::ElementSFB;
  using StrideA = typename Base::StrideA;
  using StrideB = typename Base::StrideB;
  using StrideC = typename Base::StrideC;
  using StrideD = typename Base::StrideD;

  static_assert(TileScheduler::GEMM_TYPE == GemmType::MGroupedContiguous,
    "FP4 OverlapPrologue only supports GroupedContiguous.");
  static_assert(std::is_empty<typename CollectiveEpilogue::SharedStorage>::value,
    "FP4 OverlapPrologue requires an epilogue that does not use shared memory.");

  CUTLASS_DEVICE void
  operator()(Params const& params, char* smem_buf) {
    using namespace cute;

    CUTE_STATIC_ASSERT(is_static<TileShape>::value);
    static_assert(rank(StrideA{}) == 3, "StrideA must be rank-3.");
    static_assert(rank(StrideB{}) == 3, "StrideB must be rank-3.");
    static_assert(rank(StrideC{}) == 3, "StrideC must be rank-3.");
    static_assert(rank(StrideD{}) == 3, "StrideD must be rank-3.");

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    TileScheduler deep_scheduler{params.scheduler};
    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{};

    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    uint32_t m_block_idx = 0;
    uint32_t n_block_idx = 0;

    auto fetch_next_valid_tile = [&]() {
      #pragma clang loop licm(disable)
      while (deep_scheduler.fetch_next_work(m_block_idx, n_block_idx)) {
        if (__ldg(params.scheduler.grouped_layout + m_block_idx * TileScheduler::BLOCK_M) == -1) {
          continue;
        }
        return true;
      }
      return false;
    };

    auto make_mainloop_params = [&](uint32_t M) {
      auto offset_a = deep_scheduler.curr_offset_a();
      auto offset_b = deep_scheduler.curr_offset_b(m_block_idx);
      auto offset_scalea = deep_scheduler.curr_offset_mxfp4_scalea();
      auto offset_scaleb = deep_scheduler.curr_offset_mxfp4_scaleb(m_block_idx);
      const ElementA* ptr_A = reinterpret_cast<const ElementA*>(params.mainloop.ptr_A) + offset_a;
      const ElementB* ptr_B = reinterpret_cast<const ElementB*>(params.mainloop.ptr_B) + offset_b;
      const ElementSFA* ptr_scale_A = reinterpret_cast<const ElementSFA*>(params.mainloop.ptr_scale_A) + offset_scalea;
      const ElementSFB* ptr_scale_B = reinterpret_cast<const ElementSFB*>(params.mainloop.ptr_scale_B) + offset_scaleb;
      return MainloopParams{
        {M, N, K}, ptr_A, params.mainloop.dA, ptr_B, params.mainloop.dB,
        ptr_scale_A, params.mainloop.dSFA, ptr_scale_B, params.mainloop.dSFB
      };
    };

    bool tile_valid = fetch_next_valid_tile();
    if (!tile_valid) {
      return;
    }

    uint32_t M = deep_scheduler.curr_problem_m();
    get<0>(problem_shape_MNKL) = M;
    auto blk_coord_mnkl = make_coord(m_block_idx, n_block_idx, _, 0);
    auto update_params = make_mainloop_params(M);
    CollectiveMainloop collective_mma(update_params, problem_shape_MNKL);
    auto load_inputs = collective_mma.load_init(problem_shape_MNKL, blk_coord_mnkl, update_params);
    auto k_tile_iter = 0;
    int k_tile_count = size<2>(get<0>(load_inputs));
    collective_mma.prologue(load_inputs, k_tile_iter, k_tile_count, thread_idx, smem_buf);

    TiledMma tiled_mma;
    Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));

    CUTLASS_PRAGMA_NO_UNROLL
    while (tile_valid) {
      auto current_problem_shape_MNKL = problem_shape_MNKL;
      auto current_blk_coord_mnkl = blk_coord_mnkl;
      auto params_epilogue_local = params.epilogue;
      params_epilogue_local.ptr_D += deep_scheduler.curr_offset_c();

      Tensor gA = get<0>(load_inputs);
      Tensor gB = get<1>(load_inputs);
      auto m_max_coord = M - size<0>(gA) * get<0>(current_blk_coord_mnkl);
      auto n_max_coord = N - size<0>(gB) * get<1>(current_blk_coord_mnkl);
      auto k_residue = K - size<1>(gA) * size<2>(gA);
      auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

      clear(accumulators);
      collective_mma(
        accumulators,
        load_inputs,
        accumulators,
        k_tile_iter, k_tile_count,
        residue_mnk,
        thread_idx,
        smem_buf
      );

      tile_valid = fetch_next_valid_tile();
      if (tile_valid) {
        M = deep_scheduler.curr_problem_m();
        get<0>(problem_shape_MNKL) = M;
        blk_coord_mnkl = make_coord(m_block_idx, n_block_idx, _, 0);
        update_params = make_mainloop_params(M);
        collective_mma = CollectiveMainloop(update_params, problem_shape_MNKL);
        load_inputs = collective_mma.load_init(problem_shape_MNKL, blk_coord_mnkl, update_params);
        k_tile_iter = 0;
        k_tile_count = size<2>(get<0>(load_inputs));
        // The empty epilogue leaves all drained mainloop stages available.
        collective_mma.prologue(load_inputs, k_tile_iter, k_tile_count, thread_idx, smem_buf);
      }

      CollectiveEpilogue epilogue{params_epilogue_local, shared_storage.tensors.epilogue};
      epilogue(
        current_problem_shape_MNKL,
        blk_shape,
        current_blk_coord_mnkl,
        accumulators,
        tiled_mma,
        residue_mnk,
        thread_idx,
        reinterpret_cast<char*>(&shared_storage.tensors.epilogue)
      );
      __syncthreads();
    }
  }
};

}
