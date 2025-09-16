#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#define ACOMPUTE_VERSION 10500

#include <iostream>
#include <cuda_fp8.h>
#include "profiling_interface.hpp"
#include "utils.cuh"
#include <sys/file.h>
#include "cutlass/cutlass.h"

#include "cutlass/workspace.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cute/tensor.hpp"
#include "cutlass/trace.h"
#include "ppu/cute/util.hpp"
#include "ppu/cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_conversion.h"
#include "ppu/gemm/config/gemm_configs.hpp"
#include "ppu/cutlass/epilogue/fusion/ppu_callbacks.hpp"
#include "tools/util/include/cutlass/util/host_tensor.h"
#include "tools/util/include/cutlass/util/packed_stride.hpp"
#include "scheduler_cutlass3.cuh"
#include "fp8_mainloop_with_scale.hpp"

namespace deep_gemm {
using namespace cute;
using cutlass::KernelHardwareInfo;

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class DeepGemmUniversal
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(ProblemShape{}) == 3 or cute::rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");
  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using ElementScale  = typename CollectiveMainloop::ElementScale;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using TileScheduler = TileScheduler_;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;
  using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;

  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));

  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumMmaWarpGroups = 1;

  // Kernel level shared memory storage
  struct SharedStorage {
    // Mainloop and epilogue don't use smem concurrently since kernel is non-persistent, so we can use a union
    union SharedTensorStorage {
      using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
      using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;

      MainloopSharedStorage mainloop;
      EpilogueSharedStorage epilogue;
    } tensors;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
    void* workspace{nullptr};
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace, int* grouped_layout) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    auto problem_shape = args.problem_shape;
    if constexpr (cutlass::gemm::kernel::detail::Has_SwapAB_v<CollectiveMainloop>) {
      // swap M/N
      cute::get<0>(problem_shape) = cute::get<1>(args.problem_shape);
      cute::get<1>(problem_shape) = cute::get<0>(args.problem_shape);
    }
    auto problem_shape_MNKL = cute::append<4>(problem_shape, 1);

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    void* scheduler_workspace = workspace_ptr;
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_offset = cutlass::round_nearest(workspace_offset,  cutlass::MinWorkspaceAlignment);

    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = cutlass::round_nearest(workspace_offset,  cutlass::MinWorkspaceAlignment);

    void* mainloop_workspace = nullptr;
    // Precompute the sub tiles numbers in epilogue, pass into tile scheduler.  Therefore it will be used
    // in separate reduction scheme for streamk case, NumEpilogueSubTiles default value is 1, which means
    // subtile will not be used, therefore separate reduction will not be enabled.
    constexpr uint32_t NumEpilogueSubTiles = 1; //CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(grouped_layout,
      problem_shape_MNKL, TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace, NumEpilogueSubTiles);

    return {
      args.mode,
      problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, mainloop_workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, epilogue_workspace),
      hw_info,
      scheduler,
      workspace
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = (args.mode == GemmUniversalMode::kGemm) or
        (args.mode == GemmUniversalMode::kBatched && cute::rank(ProblemShape{}) == 4);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return implementable;
    }
    implementable &= TileScheduler::can_implement(args.scheduler);
    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    return 0;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    cutlass::CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    return dim3(params.hw_info.sm_count, 1, 1);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

    TileScheduler deep_scheduler{params.scheduler};

    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShape>::value);


    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    // Preconditions
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{}; // (BLK_M,BLK_N,BLK_K)

    uint32_t m_block_idx, n_block_idx;
    while (deep_scheduler.fetch_next_work(m_block_idx, n_block_idx)) {
      auto m_coord = m_block_idx;
      auto n_coord = n_block_idx;
      auto l_coord = 0;
      M = deep_scheduler.curr_problem_m(params.scheduler);
      auto offset_a = deep_scheduler.curr_offset_a(params.scheduler);
      auto offset_b = deep_scheduler.curr_offset_b(params.scheduler, m_block_idx);
      const ElementA* ptr_A = reinterpret_cast<const ElementA*>(params.mainloop.ptr_A) + offset_a;
      const ElementB* ptr_B = reinterpret_cast<const ElementB*>(params.mainloop.ptr_B) + offset_b;
      // scaleA(shape_m, shape_k / 128), offset_a is A's offset
      const ElementScale* ptr_scale_A = reinterpret_cast<const ElementScale*>(params.mainloop.ptr_scale_A) + offset_a / 128;
      const ElementScale* ptr_scale_B = reinterpret_cast<const ElementScale*>(params.mainloop.ptr_scale_B) + offset_b / 128 / 128;

      auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);
      CollectiveMainloop collective_mainloop;
      // update actual global ptr offset
      MainloopParams update_params = {
        ptr_A, params.mainloop.dA, ptr_B, params.mainloop.dB,
        ptr_scale_A, params.mainloop.dScaleA,
        ptr_scale_B, params.mainloop.dScaleB
      };
      auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, blk_coord_mnkl, update_params);
      static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");

      Tensor gA = get<0>(load_inputs);
      Tensor gB = get<1>(load_inputs);
      Tensor gSA = get<2>(load_inputs);
      Tensor gSB = get<3>(load_inputs);
      // Compute tile residues for predication
      auto m_max_coord = M - size<0>(gA) * get<0>(blk_coord_mnkl);                             // M - BLK_M * m_coord
      auto n_max_coord = N - size<0>(gB) * get<1>(blk_coord_mnkl);                             // N - BLK_N * n_coord
      auto k_residue   = K - size<1>(gA) * size<2>(gA);                                        // K - BLK_K * k_coord_max
      auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

      // Allocate the tiled_mma and the accumulators for the (M,N) blk_shape
      TiledMma tiled_mma;
      Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape)); // (MMA,MMA_M,MMA_N)
      clear(accumulators);

      auto k_tile_iter  = cute::make_coord_iterator(shape<2>(gA));
      int  k_tile_count = size<2>(gA);

      // Perform the collective scoped MMA
      collective_mainloop(
        accumulators,
        gA,
        gB,
        gSA,
        gSB,
        accumulators,
        k_tile_iter, k_tile_count,
        residue_mnk,
        thread_idx,
        smem_buf
      );

      // update params.epilogue for ptrC and ptrD
      auto params_epilogue_local = params.epilogue;
      params_epilogue_local.ptr_D += deep_scheduler.curr_offset_c(params.scheduler);

      // Epilogue and write to gD
      CollectiveEpilogue epilogue{params_epilogue_local, shared_storage.tensors.epilogue};
      epilogue(
        problem_shape_MNKL,
        blk_shape,
        blk_coord_mnkl,
        accumulators,
        tiled_mma,
        residue_mnk,
        thread_idx,
        (char*)&shared_storage.tensors.epilogue
      );

    } // Scheduler work fetch loop
  }

};

template <int32_t SHAPE_N, int32_t SHAPE_K,
          int32_t BLOCK_M, int32_t BLOCK_N, int32_t BLOCK_K,
          int32_t WARP_M, int32_t WARP_N,
          int32_t BLOCK_N_PADDING,
          int32_t kSwizzleDMode,
          int32_t kNumGroups, int32_t kNumStages,
          GemmType kGemmType>
class Fp8Gemm {

public:
    Fp8Gemm() = default;

    static uint32_t generate_id() {
        static uint32_t id = 0;
        return ++id;
    }
    static constexpr bool UseAIU = true;
    // A matrix configuration
    using         ElementA    = cutlass::float_e4m3_t;                          // Element type for A matrix operand
    using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
    static constexpr int AlignmentA  = UseAIU ? 1 : 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    using         ElementB    = cutlass::float_e4m3_t;                          // Element type for B matrix operand
    using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
    static constexpr int AlignmentB  = UseAIU ? 1 : 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

    // D matrix configuration
    using         ElementD    = cutlass::bfloat16_t;
    using         LayoutD     = cutlass::layout::RowMajor;
    static constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;

    // C matrix configuration
    using         ElementC    = ElementD;
    using         LayoutC     = LayoutD;
    static constexpr int AlignmentC  = AlignmentD;


    // Core kernel configurations
    using ElementAccumulator  = float;                                          // Element type for internal accumulation
    using ElementCompute      = float;                                          // Element type for epilogue computation
    using ElementScalar    = ElementCompute;

    static constexpr int BlockM = BLOCK_M;
    static constexpr int BlockN = BLOCK_N;
    static constexpr int BlockK = BLOCK_K;
    static constexpr int WarpM = WARP_M;
    static constexpr int WarpN = WARP_N;
    static constexpr int WarpK = BLOCK_K;
    static constexpr int Stage = kNumStages;
    using TileShape = Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>;
    using WarpShape = Shape<Int<WarpM>, Int<WarpN>, Int<BlockK>>;

    using WarpOnM = Int<BlockM / WarpM>;
    using WarpOnN = Int<BlockN / WarpN>;
    static constexpr bool TransA = cutlass::platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
    static constexpr bool TransB = cutlass::platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;

    using DispatchPolicy = cutlass::gemm::MainloopAcomputeAiuFP8<Stage, cutlass::gemm::KernelAiuMultistage>;

    using GemmOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ElementA, TransA, Int<BlockM>, Int<BlockK>, false>;//operation
    using GemmOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ElementB, TransB, Int<BlockN>, Int<BlockK>, true>;

    using TransformA = typename cutlass::platform::conditional<
      cutlass::platform::is_same<ElementA, float>::value && cutlass::platform::is_same<ElementB, cutlass::tfloat32_t>::value,
      cute::convert<cutlass::tfloat32_t>,//change to tf32
      cute::identity
    >::type;

    using TransformB = typename cutlass::platform::conditional<
      cutlass::platform::is_same<ElementB, float>::value && cutlass::platform::is_same<ElementB, cutlass::tfloat32_t>::value,
      cute::convert<cutlass::tfloat32_t>,
      cute::identity
    >::type;

    using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ElementB>::type;
    using TiledMma = cute::TiledMMA<
        cute::MMA_Atom<MmaInst>,
        cute::Layout<Shape<WarpOnM, WarpOnN, _1>>>;

    // ElemA/B and LayoutA/B is already transfered
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveMmaBlockWise<
      DispatchPolicy, TileShape,
      ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
      ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
      TiledMma,
      typename GemmOperandA::GmemTiledCopy, typename GemmOperandA::SmemLayoutAtom, typename GemmOperandA::SmemCopyAtom, TransformA,
      typename GemmOperandB::GmemTiledCopy, typename GemmOperandB::SmemLayoutAtom, typename GemmOperandB::SmemCopyAtom, TransformB
    >;

    using EpilogueDispatchPolicy = cutlass::epilogue::EpilogueSimtVectorized;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm80, cutlass::arch::OpClassTensorOp,
      TileShape, WarpShape,
      EpilogueTileType,
      ElementCompute, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueDispatchPolicy
    >::CollectiveOp;

    static void run(__nv_bfloat16* gmem_d,
                    __nv_fp8_e4m3* input_a,
                    __nv_fp8_e4m3* input_b,
                    float* scales_a,
                    float* scales_b,
                    int* grouped_layout,
                    int32_t shape_m, uint32_t expected_m,
                    cudaStream_t stream,
                    int num_sms, uint32_t smem_size) {
        using TileScheduler = DeepGemmScheduler<kGemmType, SHAPE_N, SHAPE_K, BlockM, BlockN, kNumGroups>;
        using GemmKernel = typename deep_gemm::DeepGemmUniversal<
          Shape<int,int,int,int>,
          CollectiveMainloop,
          CollectiveEpilogue,
          TileScheduler
        >;

        using StrideA = typename GemmKernel::StrideA;
        using StrideB = typename GemmKernel::StrideB;
        using StrideC = typename GemmKernel::StrideC;
        using StrideD = typename GemmKernel::StrideD;
        using StrideS = typename CollectiveMainloop::StrideScale;

        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(shape_m, SHAPE_K, 1));
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(SHAPE_N, SHAPE_K, 1));
        StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(shape_m, SHAPE_N, 1));
        const int scale_k = (SHAPE_K + 127 - 1) / 128;
        const int scale_n = (SHAPE_N + 127 - 1) / 128;
        StrideS stride_scale_A = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(shape_m, scale_k, 1));
        StrideB stride_scale_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(scale_n, scale_k, 1));

        cutlass::float_e4m3_t* converted_input_b = reinterpret_cast<cutlass::float_e4m3_t*>(input_b);
        cutlass::float_e4m3_t* converted_input_a = reinterpret_cast<cutlass::float_e4m3_t*>(input_a);
        cutlass::bfloat16_t* converted_output = reinterpret_cast<cutlass::bfloat16_t*>(gmem_d);
        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.sm_count = num_sms;
        typename GemmKernel::Arguments arguments{
          cutlass::gemm::GemmUniversalMode::kGemm,
          {shape_m, SHAPE_N, SHAPE_K, 1},
          {converted_input_a, stride_A, converted_input_b, stride_B,
           scales_a, stride_scale_A, scales_b, stride_scale_B},
          {
            {1, 0},
            nullptr, stride_D,
            converted_output, stride_D
          },
          hw_info,
        };
        // Using the arguments, query for extra workspace required for matrix multiplication computation
        size_t workspace_size = GemmKernel::get_workspace_size(arguments);

        // Allocate workspace memory
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        // evt realization must construct evt params on device, can't use GemmUniversalAdapter
        typename GemmKernel::Params params = GemmKernel::to_underlying_arguments(arguments, workspace.get(), grouped_layout);

        dim3 const block = GemmKernel::get_block_shape();
        dim3 const grid = GemmKernel::get_grid_shape(params);
        int sharemem_size = GemmKernel::SharedStorageSize;
        // std::cout << "block = " << block << std::endl;
        // std::cout << "grid = " << grid << std::endl;
        // std::cout << "smem_size_kernel = " << sharemem_size << std::endl;

        // TODO: query max_active_tb_num
        int max_active_tb_num = 8; //GemmGrouped::maximum_active_blocks();

        const int threadblock_count = num_sms < 20 ? num_sms : num_sms * max_active_tb_num;
        char *pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, cutlass::device_kernel<GemmKernel>);

            printf("[GemmGrouped-FP8:]\n");
            printf("group:%d, problem:[%d, %d, %d], expected_m:%d, gemm_type:%s\n",
                kNumGroups, shape_m, SHAPE_N, SHAPE_K, expected_m, GemmTypeS[static_cast<int>(kGemmType)]);

            printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], kNumStages:%d\n",
                BlockM, BlockN, BlockK, WarpM, WarpN, BlockK, kNumStages);

            printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms, max_active_tb_num, threadblock_count);

            printf("smem_size:%d, vreg:%d, stack:%d\n", sharemem_size, int(attr.numRegs), int(attr.localSizeBytes));
        }

        // export PPU_LIB_PERF_INSTRUMENT=1
        int id = generate_id();
        int pid = getpid();
        int device_id = -1;
        cudaError_t result = cudaGetDevice(&device_id);
        if (result != cudaSuccess) {
            printf("get device id failed\n");
            return;
        }

        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()){
            dg_prof_params.set_deep_gemm_params(
                GemmTypeS[static_cast<int>(kGemmType)], std::string("fp8"), id, kNumGroups, shape_m, SHAPE_N, SHAPE_K, expected_m, device_id, pid
            );
        }
        ProfilingInterface::Instance().instrument(true, dg_prof_params);

        cutlass::device_kernel<GemmKernel><<<grid, block, sharemem_size, stream>>>(params);
        ProfilingInterface::Instance().instrument(false, dg_prof_params);



        char *pEnv_params_dump = std::getenv("PPU_LIB_SHOW_PARAMS");
        char *pEnv_dump_device = std::getenv("PPU_LIB_DUMP_DEVICE");
        static int target_device_id = pEnv_dump_device != nullptr ? std::stoi(pEnv_dump_device) : 0;

        if (pEnv_params_dump && std::string(pEnv_params_dump) == "2" && kGemmType != GemmType::Normal && target_device_id == device_id) {
            // check if cuda graph captured
            cudaStreamCaptureStatus captureStatus;
            cudaStreamIsCapturing(stream, &captureStatus);
            // add cuda graph mode later
            if (captureStatus != cudaStreamCaptureStatusNone) {
                printf("dump_group_m not supported in cuda graph mode.");
                return;
            }
            std::ostringstream filename;
            filename << "case" << id << "_"
                     << GemmTypeS[static_cast<int>(kGemmType)] << "_"
                     << "fp8" << "_"
                     << "groups" << kNumGroups << "_"
                     << "m" << shape_m << "_"
                     << "n" << SHAPE_N << "_"
                     << "k" << SHAPE_K << "_"
                     << "em" << expected_m << "_"
                     << "gpu" << device_id << "_"
                     << "pid" << pid << ".dump";

            std::ifstream file(filename.str().c_str());
            int fd = open(filename.str().c_str(), O_CREAT | O_WRONLY | O_APPEND, 0666);
            if (fd != -1 && !file) {
                if (flock(fd, LOCK_EX | LOCK_NB) != -1)
                    print_to_file(grouped_layout, kGemmType == GemmType::GroupedContiguous ? shape_m : kNumGroups,  filename.str().c_str(), stream);
                close(fd);
            }
        }
    }
};

};  // namespace deep_gemm

#pragma clang diagnostic pop
