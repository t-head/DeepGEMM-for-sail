#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

#include "cute/tensor.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "utils_rtc.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies an element wise operation to all elements within the fragment
/// and writes them out to destination storage.
template <
  class StrideC_,
  class StrideD_,
  class ThreadEpilogueOp_,
  class EpilogueSchedule_,
  bool IsAlignedN = false, // ProblemN % blockN == 0, used for deepgemm
  bool kApplySwigluLimit = false
>
class EpilogueNoTsmSiluAndMulQuant: public DefaultEpilogueNoTsm<StrideC_, StrideD_, ThreadEpilogueOp_, EpilogueSchedule_, IsAlignedN>{
public:
  // Base class alias
  using Base = DefaultEpilogueNoTsm<StrideC_, StrideD_, ThreadEpilogueOp_, EpilogueSchedule_, IsAlignedN>;
  using typename Base::SharedStorage;
  using typename Base::EpilogueSchedule;
  using typename Base::ThreadEpilogueOp;
  using typename Base::ElementAccumulator;
  using typename Base::FragmentAccumulator;
  using Base::kIsAlignedN;

  using VectorTypeOutputScale = Array<uint8_t, 2>;
  static constexpr int TSM_PADDING = 4; // padding 1 float for bankconflict

  static constexpr size_t get_shared_storage_size(int block_m, int block_n) {
    return size_t(block_m) * (block_n / 2 + TSM_PADDING) * sizeof(ElementAccumulator);
  }

  struct Arguments : Base::Arguments {
    uint16_t* ptr_SFD = nullptr;
    uint32_t shape_m = 0;
    float swiglu_limit = 0.f;
  };
  using Params = Arguments;

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      [[maybe_unused]] ProblemShape const& _,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    return args;
  }

  // Note: assumed that params_ are cutted by compiler for Base class initialization.
  CUTLASS_HOST_DEVICE
  EpilogueNoTsmSiluAndMulQuant(Params const& params_, SharedStorage const& shared_storage = SharedStorage())
      : DefaultEpilogueNoTsm<StrideC_, StrideD_, ThreadEpilogueOp_, EpilogueSchedule_, IsAlignedN>(params_, shared_storage), params(params_), epilogue_op(params_.thread) { }

  static CUTLASS_DEVICE uint8_t pcnvt_f4x2_f32x2(float hi, float lo) {
    uint8_t b;
    asm ("ppu.cvt.rtte.satfinite.e2m1x2.f32 %0, %1, %2;\n"
      : "=c"(b)
      : "f"(hi), "f"(lo));
    return b;
  }

  // `smem_act_tensor_ptr` is already advanced to the row of the current step_m, so all the
  // tsm offsets here are compile-time constants and no per-step address math is emitted.
  template<
    int WARP_ITER_NUM_N,
    int QUANT_LAYOUT_STRIDE_WARP_ITER_N
  >
  static CUTLASS_DEVICE void downcast_to_mxfp4(
    const float* smem_act_tensor_ptr,
    ElementAccumulator *rC,
    uint32_t *scale_bits_u32,
    uint8_t *val_2f4_packeds,
    VectorTypeOutputScale *scale_e8m0s
  ) {
    // 0. read val from tsm to vreg
    cute::for_each(make_int_sequence<WARP_ITER_NUM_N>{}, [&](auto step_n) {
      constexpr uint32_t offset = step_n * QUANT_LAYOUT_STRIDE_WARP_ITER_N;
      rC[step_n] = smem_act_tensor_ptr[offset];
    });

    // 1. calculate amax inner group by __reduce_max_sync
    cute::for_each(make_int_sequence<WARP_ITER_NUM_N>{}, [&](auto step_n) {
      auto val_u32 = __float_as_uint(fabsf(rC[step_n]));
      scale_bits_u32[step_n] = __reduce_max_sync(0xffffffff, val_u32);
    });

    // 2. calculate scales
    cute::for_each(make_int_sequence<WARP_ITER_NUM_N>{}, [&](auto step_n) {
      // Clamp absmax to avoid division by zero in scale computation
      // float group_max_fp32 = fmaxf(__uint_as_float(scale_bits_u32[step_n]), 1e-10f);
      uint32_t group_max_fp32 = max(scale_bits_u32[step_n], 0x2EDBE6FF); // 0x2EDBE6FF is 1e-10f (u32);

      // Scale Round Up as OCP decalared.
      // scale_bits_u32[step_n] = __float_as_uint(group_max_fp32 * (1.f / 6.f));
      // 0x1400000u means that 6.0 can be termed as 4 * 1.5 which can be represented by u32.
      scale_bits_u32[step_n] = group_max_fp32 - 0x1400000u; // ceil to power of 2;
    });

    // 3. calculate scales-0: round up
    cute::for_each(make_int_sequence<WARP_ITER_NUM_N>{}, [&](auto step_n) {
      scale_bits_u32[step_n] = (scale_bits_u32[step_n] + 0x007FFFFFu);
    });

    // 4. calculate scales-1: remove mantissa and do quant
    cute::for_each(make_int_sequence<WARP_ITER_NUM_N>{}, [&](auto step_n) {
      unsigned scale_exp_bit = scale_bits_u32[step_n] & 0x7F800000u; // + 1 and remove mantissa

      // save e8m0 scale into vreg for tsm.st
      constexpr int scale_step = step_n / 2;
      constexpr int scale_step_offset = step_n % 2;
      scale_e8m0s[scale_step][scale_step_offset] = (uint8_t)(scale_exp_bit >> 23);

      // equal to 1 / __uint_as_float(scale_exp_bits)
      // exponent << 23 termed as 2 ^ (exponent - 127)
      // we want 2 ^ (127 - exponent) which means exponent^{'} is (127 - exponent) + 127
      // bits: exponent^{'} << 23 = (254 - exponent) << 23 = 254 << 23 - exponent << 23
      float scale_inv = __uint_as_float((254u << 23) - scale_exp_bit);

      // do quant
      auto val_quanted_tXv0 = rC[step_n] * scale_inv;
      auto val_quanted_tXv0_adjecent = __shfl_xor_sync(0xffffffff, val_quanted_tXv0, 1);
      val_2f4_packeds[step_n] = pcnvt_f4x2_f32x2(val_quanted_tXv0_adjecent, val_quanted_tXv0);
    });
  }

  // IsNBoundary: whether this block is cutted along the N coord (ProblemN % BlockN != 0).
  // The M coord boundary is folded into the quant loop trip count instead of being a template
  // arg, so boundary and non-boundary blocks share a single instantiation. That keeps the
  // instruction footprint of the epilogue small (it is instruction fetch bound).
  template<
    bool IsNBoundary,
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  apply(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char* smem_buf)
  {
    using namespace cute;
    using X = Underscore;

    static_assert(cute::rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<BlockShapeMNK>::value, "ThreadBlock tile shape must be static");
    static_assert(cute::rank(BlockShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
    static_assert(cute::rank(BlockCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 4");

    // Separate out problem shape for convenience
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);

    using ThrLayoutVMNK = typename TiledMma::ThrLayoutVMNK;
    // WARP_ON_M * WARP_ITER_M_NUM * 16(means 8 threads * 2(mma_m) values) = BlockM
    // WARP_ON_N * WARP_ITER_N_NUM * 16(means 4 threads * 2(mma_n) * 2 values) = BlockN
    static constexpr int WARP_ON_M = size<1>(ThrLayoutVMNK{});
    static constexpr int WARP_ON_N = size<2>(ThrLayoutVMNK{});
    static constexpr int WARP_ITER_M_NUM = decltype(size<1>(accumulators))::value;
    static constexpr int WARP_ITER_N_NUM = decltype(size<2>(accumulators))::value;

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx % 32;
    int warp_m_idx = warp_idx % WARP_ON_M;
    int warp_n_idx = warp_idx / WARP_ON_M;

    //  BlockM, BlockN and BlockK are static vals, but M/N/N are not;
    auto BlockM = get<0>(blk_shape_MNK);
    auto BlockN = get<1>(blk_shape_MNK);
    auto BlockK = get<2>(blk_shape_MNK);
    constexpr bool IsBlockN64 = (BlockN == 64); // BlockN64 will only has one e8m0 scale in a tile;

    auto stride_d = detail::get_epilogue_stride<EpilogueSchedule>(params.dD);

    // Represent the full output tensor; gC(bias) is not supported;
    Tensor mD_mnl = make_tensor(make_gmem_ptr(params.ptr_D), make_shape(M,N,L), stride_d);                 // (m,n,l)
    Tensor gD_mnl = local_tile(mD_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});    // (BLK_M,BLK_N,m,n,l)

    // Slice to get the tile this CTA is responsible for
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
    Tensor gD = gD_mnl(_,_,m_coord,n_coord,l_coord);                                                 // (BLK_M,BLK_N)

    constexpr int kOutElemsPerByte = EpilogueTraits<EpilogueType::SiluAndMulPostQuantFp4>::kOutElemsPerByte;
    auto gD_ptr = params.ptr_D + (cute::raw_pointer_cast(gD.data()) - params.ptr_D) / kOutElemsPerByte;

    // Partition source and destination tiles to match the accumulator partitioning
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCgD = thr_mma.partition_C(gD);                                       // (VEC,THR_M,THR_N)

    // 0. The result of activation function tensor Layout for tsm.st: 
    // ((WARP_ITER_M_NUM, WARP_ON_M, (2, 8)), (WARP_ITER_N_NUM, WARP_ON_N, (2, 4))), RowMajor with padding in row
    constexpr int BlockNAct = BlockN / 2; // Row length for silu_and_mul

    // constexpr int act_layout_stride_thread_n = 1;
    constexpr int act_layout_stride_mma_n = 4;  // across 4 threads T0/T1/T2/T3
    constexpr int act_layout_stride_warp_n = 8; // 8 values in each warp row
    constexpr int act_layout_stride_warp_iter_n = WARP_ON_N * act_layout_stride_warp_n; // across tiled_mma_m
    constexpr int act_layout_stride_thread_m = BlockNAct + TSM_PADDING; // across a row, add padding;
    constexpr int act_layout_stride_mma_m = act_layout_stride_thread_m * 8; // across mma_m, 8 threads T0/T4/.../T28
    constexpr int act_layout_stride_warp_m = 2 * act_layout_stride_mma_m; //across a warp col, 2 groups * 8 threads;
    constexpr int act_layout_stride_warp_iter_m = WARP_ON_M * act_layout_stride_warp_m;

    auto smem_act_tensor_ptr = reinterpret_cast<float*>(smem_buf);

    static_assert(is_static<FrgLayout>::value, "Accumulator layout must be static");
    CUTE_STATIC_ASSERT_V(size(tCgD) == size(accumulators),
        "Accumulator count must have the same destination element count.");

    static constexpr int MMA_M = decltype(size<1>(shape<0>(accumulators)))::value;
    static constexpr int MMA_N = decltype(size<2>(shape<0>(accumulators)))::value;
    int warp_base_offset = warp_m_idx * act_layout_stride_warp_m + (lane_idx / 4) * act_layout_stride_thread_m + warp_n_idx * act_layout_stride_warp_n + (lane_idx % 4); // * act_layout_stride_thread_n=1
    auto smem_act_tensor_ptr_per_lane = smem_act_tensor_ptr + warp_base_offset;
    cutlass::epilogue::thread::Sigmoid<ElementAccumulator> sigmoid_accumulator;
    multiplies<ElementAccumulator> mul_accumulator;
    cute::for_each(make_int_sequence<WARP_ITER_M_NUM>{}, [&](auto warp_iter_m) {
      cute::for_each(make_int_sequence<MMA_M>{}, [&](auto mma_m) {
        // unroll inner loops manually to mitigate compute deps
        ElementAccumulator sigmoid_accums[WARP_ITER_N_NUM][MMA_N];
        ElementAccumulator mul_accums[WARP_ITER_N_NUM][MMA_N];
        cute::for_each(make_int_sequence<WARP_ITER_N_NUM>{}, [&](auto warp_iter_n) {
          cute::for_each(make_int_sequence<MMA_N>{}, [&](auto mma_n) {
            FragmentAccumulator input_array = *reinterpret_cast<const FragmentAccumulator*>(accumulators(make_coord(_, mma_m, mma_n), warp_iter_m, warp_iter_n).data());

            // 0. sigmoid only for better compute deps
            if constexpr (kApplySwigluLimit) {
              // using mul_accums[warp_iter_n][mma_n] to store gate_clamped
              mul_accums[warp_iter_n][mma_n] = min(input_array[0], params.swiglu_limit);
              sigmoid_accums[warp_iter_n][mma_n] = sigmoid_accumulator(mul_accums[warp_iter_n][mma_n]);
            } else {
              sigmoid_accums[warp_iter_n][mma_n] = sigmoid_accumulator(input_array[0]);
            }
          });
        });

        cute::for_each(make_int_sequence<WARP_ITER_N_NUM>{}, [&](auto warp_iter_n) {
          cute::for_each(make_int_sequence<MMA_N>{}, [&](auto mma_n) {
            if constexpr (kApplySwigluLimit) {
              mul_accums[warp_iter_n][mma_n] = mul_accumulator(mul_accums[warp_iter_n][mma_n], sigmoid_accums[warp_iter_n][mma_n]);
            } else {
              FragmentAccumulator input_array = *reinterpret_cast<const FragmentAccumulator*>(accumulators(make_coord(_, mma_m, mma_n), warp_iter_m, warp_iter_n).data());
              mul_accums[warp_iter_n][mma_n] = mul_accumulator(input_array[0], sigmoid_accums[warp_iter_n][mma_n]);
            }
          });
        });

        cute::for_each(make_int_sequence<WARP_ITER_N_NUM>{}, [&](auto warp_iter_n) {
          cute::for_each(make_int_sequence<MMA_N>{}, [&](auto mma_n) {
            FragmentAccumulator input_array = *reinterpret_cast<const FragmentAccumulator*>(accumulators(make_coord(_, mma_m, mma_n), warp_iter_m, warp_iter_n).data());
            if constexpr (kApplySwigluLimit) {
              auto up_clamped = max(min(input_array[1], params.swiglu_limit), -params.swiglu_limit);
              sigmoid_accums[warp_iter_n][mma_n] = mul_accumulator(mul_accums[warp_iter_n][mma_n], up_clamped);
            } else {
              sigmoid_accums[warp_iter_n][mma_n] = mul_accumulator(mul_accums[warp_iter_n][mma_n], input_array[1]);
            }

            // dump vals after silu_and_mul
            constexpr int act_offset = warp_iter_m * act_layout_stride_warp_iter_m + mma_m * act_layout_stride_mma_m + warp_iter_n * act_layout_stride_warp_iter_n + mma_n * act_layout_stride_mma_n;
            smem_act_tensor_ptr_per_lane[act_offset] = sigmoid_accums[warp_iter_n][mma_n];
          });
        });
      });
    });

    // 1. The input of Quantizationg tensor Layout TV for tsm.ld:
    // ((WARP_ITER_M_NUM, WARP_NUM), (WARP_ITER_N_NUM, 32))
    constexpr int WARP_NUM = WARP_ON_M * WARP_ON_N;
    constexpr int WARP_ITER_NUM_M = cute::ceil_div(BlockM, WARP_NUM); // WARP_NUM always less than BlockM
    constexpr int WARP_ITER_NUM_N = cute::ceil_div(BlockNAct, 32); // BlockNAct >= 32 and BlockNAct % 32 = 0
    // constexpr int quant_layout_stride_thread_n = 1;
    constexpr int quant_layout_stride_warp_iter_n = 32;
    constexpr int quant_layout_stride_warp_m = BlockNAct + TSM_PADDING;
    constexpr int quant_layout_stride_warp_iter_m = WARP_NUM * quant_layout_stride_warp_m;

    // 2. The result of Quantization tensor Layout for vmem.st
    constexpr int out_quant_layout_stride_warp_iter_n = 16; // 32 values would be packed into 16 uint8
    int out_quant_layout_stride_warp_m = N / kOutElemsPerByte;  // N must be divideable by 4.
    int out_quant_layout_stride_warp_iter_m = WARP_NUM * out_quant_layout_stride_warp_m;

    // 3.1 construct Scale Tensor Layout for tsm.st(Put here to cover the tsm.st latency)
    constexpr int BlockSFN = IsBlockN64 ? 1 : BlockN / Int<128>{};  // 64 means 4(kOutElemsPerByte) * 16(16 u8)
    int SFM = params.shape_m;  // use max_m for groupedMasked and total_m for groupedNoPad because SF is M/N Major
    int SFN = cute::ceil_div(N / kOutElemsPerByte, 32); // assume SFN is padding for u16

    auto gSFD_mnl_layout = make_layout(make_shape(SFM, SFN, L), make_stride(_1{}, SFM, SFM * SFN));
    Tensor mSFD_mnl = make_tensor(make_gmem_ptr(params.ptr_SFD), gSFD_mnl_layout);
    Tensor gSFD_mnl = local_tile(mSFD_mnl, make_shape(Int<BlockM>{}, Int<BlockSFN>{}, Int<BlockK>{}), make_coord(_,_, _), Step<_1,_1, X>{});    // (BLK_M,BlockSFN,m,n,l)
    auto n_coord_ = n_coord;
    if constexpr (IsBlockN64) {
      n_coord_ /= 2; // to find the u16 base ptr offset of uint8 e8m0 scale;
    }
    Tensor gSFD = gSFD_mnl(_,_,m_coord,n_coord_,l_coord);
    auto gSFD_ptr = cute::raw_pointer_cast(gSFD.data());
    int parity = n_coord & 1;
    // When BlockN == 64 and ceil(N / 64) is odd, the high byte of the last uint16 scale slot in a
    // row belongs to no tile. Let the last (even n_coord) tile zero it so that the output does not
    // depend on how the caller allocated out_scale.
    const bool pad_hi_byte = IsBlockN64 && (parity == 0) && ((n_coord + 1) * BlockN >= N);

    // copy thread layout is (WARP, 32):(32, 1)
    // each warp traverse in rows T0V0, T1V0, ..., T31V0
    constexpr int SCALE_WARP_ITER_NUM_N = ceil_div(WARP_ITER_NUM_N, 2);
    int warp_offset_base = warp_idx * quant_layout_stride_warp_m + lane_idx;
    auto smem_act_tensor_ptr_read_per_lane = smem_act_tensor_ptr + warp_offset_base;
    int out_warp_offset_base = warp_idx * out_quant_layout_stride_warp_m + (lane_idx / 2);
    auto gD_ptr_per_lane = gD_ptr + out_warp_offset_base;
    int out_lane_offset_base = (lane_idx / 2);

    // 4. Each warp owns the rows [warp_idx, residue_m) with a stride of WARP_NUM, so the M
    // boundary predicate is monotonic along step_m and can be folded into a trip count. That
    // removes one taken branch per unrolled step and lets the body be emitted only once.
    const int rows_left = get<0>(residue_mnk) - warp_idx;
    const int num_iter_m = rows_left <= 0 ? 0 : min(WARP_ITER_NUM_M, cute::ceil_div(rows_left, WARP_NUM));

    // 4.1 store predicates are loop invariant along M, hoist them out of the quant loop
    const bool is_store_lane = (lane_idx % 2 == 0);
    bool store_val_preds[WARP_ITER_NUM_N];
    bool store_scale_preds[SCALE_WARP_ITER_NUM_N];
    cute::for_each(make_int_sequence<WARP_ITER_NUM_N>{}, [&](auto step_n) {
      if constexpr (IsNBoundary) {
        constexpr int out_offset_ = step_n * out_quant_layout_stride_warp_iter_n;
        store_val_preds[step_n] = is_store_lane && ((out_offset_ + out_lane_offset_base) < (get<1>(residue_mnk) / 4));
      } else {
        store_val_preds[step_n] = is_store_lane;
      }
    });
    // The scale residue predicates below are uint16 slot granular: a slot packs 2 uint8 e8m0 scales,
    // that is 2 * 32(quant group) * 2(silu_and_mul) = 128 output columns. `residue_n` is a multiple
    // of 64, so it can end in the middle of a slot: the high scale is then out of range and has to
    // be zeroed out, as the reference pads it with zero.
    static_assert(!(IsBlockN64 && IsNBoundary),
                  "BlockN == 64 must imply an aligned N: a uint16 scale slot then covers 64 output "
                  "columns instead of 128, which would break the residue predicates below.");
    uint16_t store_scale_masks[SCALE_WARP_ITER_NUM_N];
    cute::for_each(make_int_sequence<SCALE_WARP_ITER_NUM_N>{}, [&](auto step_n_scale) {
      // BlockN=64 always means IsAlignedN is TRUE
      if constexpr (IsNBoundary) {
        // Only for BlockN >= 128
        constexpr int last_step_n_scale_offset = step_n_scale * 128;
        store_scale_preds[step_n_scale] = last_step_n_scale_offset < get<1>(residue_mnk);
        store_scale_masks[step_n_scale] =
            ((last_step_n_scale_offset + 64) < get<1>(residue_mnk)) ? 0xFFFFu : 0x00FFu;
      } else {
        store_scale_preds[step_n_scale] = true;
      }
    });

    // 4.2 walk the tensors with base + increment to keep the loop body free of address math
    auto smem_act_tensor_ptr_iter = smem_act_tensor_ptr_read_per_lane;
    auto gD_ptr_iter = gD_ptr_per_lane;
    int scale_row = warp_idx;
    __syncthreads();  // wait for activation tensor tsm store
    // keep the loop rolled: the body is big and the epilogue is instruction fetch bound,
    #pragma unroll 1
    for (int step_m = 0; step_m < num_iter_m; ++step_m) {
      // unroll manually inner loops to mitigate compte deps
      ElementAccumulator rC[WARP_ITER_NUM_N]; // float values for load
      uint32_t scale_bits_u32[WARP_ITER_NUM_N]; // vreg for calculating e8m0 scale.
      uint8_t val_2f4_packeds[WARP_ITER_NUM_N]; // packed f4 values
      VectorTypeOutputScale scale_e8m0s[SCALE_WARP_ITER_NUM_N];

      downcast_to_mxfp4<
        WARP_ITER_NUM_N,
        quant_layout_stride_warp_iter_n>(
          smem_act_tensor_ptr_iter,
          rC,
          scale_bits_u32,
          val_2f4_packeds,
          scale_e8m0s
        );

      // 5-0. dump sacles into gmem
      cute::for_each(make_int_sequence<SCALE_WARP_ITER_NUM_N>{}, [&](auto step_n_scale) {
        // scales are in sreg, all lane write into the same location will be collapsed.
        if (store_scale_preds[step_n_scale]) {
          if constexpr (IsBlockN64) {
            // BlockN == 64 implies ShapeN % BlockN == 0, so this tile owns exactly one uint8
            // scale and the neighbour parity byte belongs to the neighbour N tile.
            uint8_t* dst_sfd = reinterpret_cast<uint8_t*>(&gSFD_ptr[scale_row + step_n_scale * SFM]);
            dst_sfd[parity] = *reinterpret_cast<uint8_t*>(&scale_e8m0s[step_n_scale]);
            if (pad_hi_byte) dst_sfd[1] = 0;
          } else if constexpr (IsNBoundary) {
            gSFD_ptr[scale_row + step_n_scale * SFM] =
                *reinterpret_cast<uint16_t*>(&scale_e8m0s[step_n_scale]) & store_scale_masks[step_n_scale];
          } else {
            gSFD_ptr[scale_row + step_n_scale * SFM] =
                *reinterpret_cast<uint16_t*>(&scale_e8m0s[step_n_scale]);
          }
        }
      });

      // 5. dump quanted values into gmem
      cute::for_each(make_int_sequence<WARP_ITER_NUM_N>{}, [&](auto step_n) {
        constexpr int out_offset_ = step_n * out_quant_layout_stride_warp_iter_n;
        if (store_val_preds[step_n]) {
          gD_ptr_iter[out_offset_] = val_2f4_packeds[step_n];
        }
      });

      smem_act_tensor_ptr_iter += quant_layout_stride_warp_iter_m;
      gD_ptr_iter += out_quant_layout_stride_warp_iter_m;
      scale_row += WARP_NUM;
    }
  }

  template<
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char* smem_buf) {

    // the M boundary is handled by the quant loop trip count inside `apply`, so only the N
    // boundary needs a dedicated instantiation.
    if constexpr (kIsAlignedN) {
      apply<false>(problem_shape_mnkl, blk_shape_MNK, blk_coord_mnkl, accumulators, tiled_mma, residue_mnk, thread_idx, smem_buf);
    } else {
      const bool n_boundary = get<1>(residue_mnk) < get<1>(blk_shape_MNK);
      if (n_boundary) {
        apply<true>(problem_shape_mnkl, blk_shape_MNK, blk_coord_mnkl, accumulators, tiled_mma, residue_mnk, thread_idx, smem_buf);
      } else {
        apply<false>(problem_shape_mnkl, blk_shape_MNK, blk_coord_mnkl, accumulators, tiled_mma, residue_mnk, thread_idx, smem_buf);
      }
    }
  }
  
private:
  Params params;
  ThreadEpilogueOp epilogue_op;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
