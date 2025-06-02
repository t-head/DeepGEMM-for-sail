#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include "scheduler.cuh"
#include "utils.cuh"
#include "profiling_interface.hpp"

#include "accutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/tensor_ref.h"

#include "aiu/gemm/device/aiugemm_grouped.h"
#include "aiu/gemm/kernel/default_gemm_grouped.h"
#include "aiu/gemm/kernel/default_gemm.h"
#include "aiu/gemm/threadblock/default_mma.h"


namespace deep_gemm {

template <typename Mma_,          ///! Threadblock-scoped matrix multiply-accumulate
    typename Epilogue_,           ///! Epilogue
    typename ProblemVisitor_
    >
struct GemmKernel {
public:
    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using EpilogueOutputOp = typename Epilogue::OutputOp;

    static bool const kTransposed = false;
    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    using ElementA = typename Mma::IteratorA::Element;
    using LayoutA = typename Mma::IteratorA::Layout;
    using ElementB = typename Mma::IteratorB::Element;
    using LayoutB = typename Mma::IteratorB::Layout;
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using LayoutC = typename Epilogue::OutputTileIterator::Layout;

    // Type definitions about the mainloop.
    using Operator = typename Mma::Operator;
    using OperatorClass = typename Mma::Operator::OperatorClass;
    using ThreadblockShape = typename Mma::Shape;
    using WarpShape = typename Mma::Operator::Shape;
    using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
    using ArchTag = typename Mma::ArchTag;

    static int const kStages = Mma::kStages;

    static int const kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static int const kAlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    /// Warp count (concept: GemmShape)
    using WarpCount = typename Mma::WarpCount;
    static int const kThreadCount = 32 * WarpCount::kCount;

    static constexpr int kInterleave = 1;

    // using ProblemVisitor =  DeepGeemProblemVisitor<ThreadblockShape>;
    using ProblemVisitor = ProblemVisitor_;

     /// Argument structure
    struct Arguments
    {
        //
        // Data members
        //

        typename EpilogueOutputOp::Params output_op;

        typename Mma::IteratorA::TensorRef ref_A;
        typename Mma::IteratorB::TensorRef ref_B;

        typename Epilogue::OutputTileIterator::TensorRef ref_D;

        int64_t gemm_m;
        int64_t gemm_n;
        int64_t gemm_k;

        int32_t* grouped_layout;

        // in-order to compatility with base_group
        cutlass::gemm::GemmCoord* host_problem_sizes {nullptr};
        int problem_count {0};
        int threadblock_count {0};

        //
        // Methods
        //

        /// Default ctor
        CUTLASS_HOST_DEVICE
        Arguments()
            : problem_count(0)
            , threadblock_count(0)
            , gemm_m(0)
            , gemm_n(0)
            , gemm_k(0)
            , grouped_layout(nullptr)
        {
        }

        /// Ctor
        CUTLASS_HOST_DEVICE
        Arguments(int problem_count, int threadblock_count, typename EpilogueOutputOp::Params output_op,
            typename Mma::IteratorA::TensorRef ref_A, typename Mma::IteratorB::TensorRef ref_B,
            typename Epilogue::OutputTileIterator::TensorRef ref_D,
            int64_t gemm_m, int64_t gemm_n, int64_t gemm_k,
            int32_t* grouped_layout)
            : problem_count(problem_count)
            , threadblock_count(threadblock_count)
            , output_op(output_op)
            , ref_A(ref_A)
            , ref_B(ref_B)
            , ref_D(ref_D)
            , gemm_m(gemm_m)
            , gemm_n(gemm_n)
            , gemm_k(gemm_k)
            , grouped_layout(grouped_layout)
        {
        }
    };

    //
    // Structure for precomputing values in host memory and passing to kernels
    //

    /// Parameters structure
    struct Params
    {
        typename ProblemVisitor::Params problem_visitor;
        int threadblock_count;
        int problem_count;

        typename EpilogueOutputOp::Params output_op;

        ElementA* ptr_A;
        typename Mma::IteratorA::Params params_A;
        ElementB* ptr_B;
        typename Mma::IteratorB::Params params_B;
        ElementC* ptr_D;
        typename Epilogue::OutputTileIterator::Params params_D;

        //
        // Methods
        //

        CUTLASS_HOST_DEVICE
        Params()
            : ptr_A(nullptr)
            , ptr_B(nullptr)
            , ptr_D(nullptr)
        {
        }

        CUTLASS_HOST_DEVICE
        Params(Arguments const& args, void* workspace = nullptr, int tile_count = 0)
            : problem_visitor(args.gemm_m, args.gemm_n, args.gemm_k, args.grouped_layout, args.problem_count)
            , problem_count(args.problem_count)
            , threadblock_count(args.threadblock_count)
            , output_op(args.output_op)
            , ptr_A(args.ref_A.data())
            , params_A(args.ref_A.layout())
            , ptr_B(args.ref_B.data())
            , params_B(args.ref_B.layout())
            , ptr_D(args.ref_D.data())
            , params_D(args.ref_D.layout())
        {
        }

        CUTLASS_HOST_DEVICE
        void update(Arguments const& args, void* workspace = nullptr, int tile_count = 0)
        {
            problem_visitor = typename ProblemVisitor::Params(
                args.gemm_m, args.gemm_n, args.gemm_k, args.grouped_layout, args.problem_count);
            threadblock_count = args.threadblock_count;
            problem_count = args.problem_count;
            output_op = args.output_op;
            ptr_A = args.ptr_A;
            ptr_B = args.ptr_B;
            ptr_D = args.ptr_D;
        }
    };

    /// Shared memory storage structure
    union SharedStorage
    {
        typename ProblemVisitor::SharedStorage problem_visitor;
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };

public:
     //
    // Methods
    //

    CUTLASS_DEVICE
    GemmKernel() {}

    /// Determines whether kernel satisfies alignment
    static cutlass::Status can_implement(cutlass::gemm::GemmCoord const& problem_size)
    {
        return cutlass::Status::kSuccess;
    }

    static cutlass::Status can_implement(Arguments const& args)
    {
        // Handle the case the input is too short
        if (args.gemm_n < Mma::IteratorB::AccessType::kElements)
        {
            CUTLASS_TRACE_HOST("MoeFCGemm::can_implement() - gemm_n is smaller than the input alignment");
            return cutlass::Status::kInvalid;
        }
        return cutlass::Status::kSuccess;
    }

    static size_t get_extra_workspace_size(Arguments const& args, cutlass::gemm::GemmCoord const& grid_tiled_shape)
    {
        return 0;
    }

    CUTLASS_DEVICE
    void operator()(Params const& params, SharedStorage& shared_storage)
    {
        //
        // These types shadow the type-level definitions and support the ability to implement
        // a 'transposed' GEMM that computes the transposed problems.
        //
        using ElementA = typename Mma::IteratorA::Element;
        using LayoutA = typename Mma::IteratorA::Layout;
        using ElementB = typename Mma::IteratorB::Element;
        using LayoutB = typename Mma::IteratorB::Layout;
        using ElementC = typename Epilogue::OutputTileIterator::Element;
        using LayoutC = typename Epilogue::OutputTileIterator::Layout;

        //
        // Problem visitor.
        //
        ProblemVisitor problem_visitor(params.problem_visitor, shared_storage.problem_visitor, blockIdx.x);

        const int64_t gemm_m = params.problem_visitor.gemm_m;
        const int64_t gemm_n = params.problem_visitor.gemm_n;
        const int64_t gemm_k = params.problem_visitor.gemm_k;

        // Outer 'persistent' loop to iterate over tiles
        int loop = 0;
        uint32_t m_block_idx, n_block_idx;
        while (problem_visitor.next_tile(m_block_idx, n_block_idx))
        {

            loop++;

            cutlass::gemm::GemmCoord problem_size = problem_visitor.problem_size();
            int32_t problem_idx = problem_visitor.problem_index();

            int32_t index_m = problem_visitor.get_global_idx(gemm_m, ThreadblockShape::kM, m_block_idx);
            int32_t index_n = problem_visitor.get_global_idx<false>(gemm_n, ThreadblockShape::kN, n_block_idx, m_block_idx);

            cutlass::gemm::GemmCoord threadblock_offset(
                index_m,
                index_n,
                0
            );

            // Compute initial location in logical coordinates
            cutlass::MatrixCoord tb_offset_A{
                threadblock_offset.m(),
                0,
            };

            cutlass::MatrixCoord tb_offset_B{0, threadblock_offset.n()};

            // Compute position within threadblock
            int thread_idx = threadIdx.x;

            // Construct iterators to A and B operands
            typename Mma::IteratorA iterator_A(
                params.params_A, params.ptr_A,
                {problem_size.m() * (ProblemVisitor::kGemmType == GemmType::GroupedMasked ? params.problem_count : 1), problem_size.k()}, thread_idx, tb_offset_A);

            typename Mma::IteratorB iterator_B(params.params_B,
                reinterpret_cast<ElementB*>(params.ptr_B),
                {problem_size.k(), problem_size.n() * params.problem_count}, thread_idx, tb_offset_B);

            typename Mma::FragmentC accumulators;

            accumulators.clear();

            // Broadcast the warp_id computed by lane 0 to ensure dependent code
            // is compiled as warp-uniform.
            int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

            int lane_idx = threadIdx.x % 32;

            //
            // Matrix multiply phase
            //

            Mma mma = Mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

            // Compute threadblock-scoped matrix multiply-add
            int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

            // Wait for all threads to finish their epilogue phases from the previous tile.
            __syncthreads();

            mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

            // Epilogue
            //

            EpilogueOutputOp output_op(params.output_op);

            LayoutC layout_D(gemm_n);

            typename Epilogue::OutputTileIterator::Params params_D(layout_D);

            // Tile iterator writing to destination tensor.

            cutlass::gemm::GemmCoord threadblock_offset_output(
                problem_visitor.get_global_idx(gemm_m, ThreadblockShape::kM, m_block_idx),
                n_block_idx * Mma::Shape::kN,
                0
            );

            cutlass::gemm::GemmCoord problem_size_output(
                problem_size.m() * (ProblemVisitor::kGemmType == GemmType::GroupedMasked ? problem_idx + 1 : 1),
                problem_size.n(),
                problem_size.k()
            );

            typename Epilogue::OutputTileIterator iterator_D(
                params_D, params.ptr_D, problem_size_output.mn(), thread_idx, threadblock_offset_output.mn());

            Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

            epilogue(output_op, iterator_D, accumulators, iterator_D);
        }
    }
};

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t WARP_M, uint32_t WARP_N,
          uint32_t kNumGroups, uint32_t kNumStages,
          GemmType kGemmType>
class Gemm {

public:
    Gemm() = default;

    static uint32_t generate_id() {
        static uint32_t id = 0;
        return ++id;
    }

    static void run(__nv_bfloat16* gmem_d, int* grouped_layout,
                    uint32_t shape_m, __nv_bfloat16* gmem_a, __nv_bfloat16* gmem_b,
                    cudaStream_t stream, int num_sms, uint32_t smem_size) {
        using ThreadblockShape = cutlass::gemm::GemmShape<BLOCK_M, BLOCK_N, BLOCK_K>;
        using WarpShape = cutlass::gemm::GemmShape<WARP_M, WARP_N, BLOCK_K>;
        using ElementType = cutlass::bfloat16_t;

        using OperatorClass = cutlass::arch::OpClassTensorOp;
        using ElementAccumulator = float;

        static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<ElementType>::value;
        static constexpr int LimitedPerAccessC_ = ((ThreadblockShape::kN) * 8 / (ThreadblockShape::kN / WarpShape::kN) / 32);
        static constexpr int ElementsPerAccessC = LimitedPerAccessC_ < ElementsPerAccess ? LimitedPerAccessC_ : ElementsPerAccess;
        static constexpr int ThreadblockK = ThreadblockShape::kK;
        using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

        using EpilogueOp = typename cutlass::epilogue::thread::LinearCombination<ElementType, ElementsPerAccessC,
                ElementAccumulator, ElementAccumulator>;

        using DefaultGemm = typename aiu::gemm::kernel::DefaultGemmGrouped<ElementType, cutlass::layout::RowMajor, ElementsPerAccess,
                                                                            ElementType, cutlass::layout::ColumnMajor, ElementsPerAccess,
                                                                            ElementType, cutlass::layout::RowMajor, ElementAccumulator,
                                                                            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, ThreadblockShape, WarpShape,
                                                                            InstructionShape, EpilogueOp,
                                                                            cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, kNumStages,
                                                                            cutlass::gemm::kernel::GroupScheduleMode::kDeepGemm,
                                                                            cutlass::arch::OpMultiplyAdd>::GemmKernel;

        using ProblemVisitor = Scheduler<kGemmType, SHAPE_N, ThreadblockShape>;

        using GemmKernel = GemmKernel<typename DefaultGemm::Mma, typename DefaultGemm::Epilogue, ProblemVisitor>;

        using GemmGrouped = aiu::gemm::device::GemmGrouped<GemmKernel>;

        int max_active_tb_num = GemmGrouped::maximum_active_blocks();

        const int threadblock_count = num_sms < 20 ? num_sms : num_sms * max_active_tb_num;

        char *pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, cutlass::Kernel<GemmKernel>);
    
            printf("[GemmGrouped-BF16:]\n");
            printf("group:%d, problem:[%d, %d, %d], gemm_type:%s\n",
                kNumGroups, shape_m, SHAPE_N, SHAPE_K, GemmTypeS[static_cast<int>(kGemmType)]);

            printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], kNumStages:%d\n",
                ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK,
                WarpShape::kM, WarpShape::kN, WarpShape::kK, kNumStages);

            printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms, max_active_tb_num, threadblock_count);

            printf("smem_size:%d, verg:%d, stack:%d\n", smem_size, int(attr.numRegs), int(attr.localSizeBytes));
        }

        // export PPU_LIB_SHOW_PARAMS=1
        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()){
            dg_prof_params.set_deep_gemm_params(
                GemmTypeS[static_cast<int>(kGemmType)], std::string("bf16"), kNumGroups, shape_m, SHAPE_N, SHAPE_K
            );
        }
    
        char *pEnv_params_dump = std::getenv("dump_group_m");
        if (pEnv_params_dump && isdigit(*pEnv_params_dump) && kGemmType != GemmType::Normal) {
            // check if cuda graph captured
            cudaStreamCaptureStatus captureStatus;
            cudaStreamIsCapturing(stream, &captureStatus);
            // add cuda graph mode later
            if (captureStatus != cudaStreamCaptureStatusNone) {
                printf("[moe gemm]: dump_group_m not supported in cuda graph mode.");
                return;
            }

            static int casedId = 0;
            std::ostringstream filename;
            int id = generate_id();
            printf("id:%d\n", id);
            filename << "case" << id << "_"
                     << "groups" << kNumGroups << "_"
                     << "m" << shape_m << "_"
                     << "n" << SHAPE_N << "_"
                     << "k" << SHAPE_K << "_"
                     << GemmTypeS[static_cast<int>(kGemmType)] << ".dump";
            print_to_file(grouped_layout, kGemmType == GemmType::GroupedContiguous ? shape_m : kNumGroups, filename.str().c_str(), stream);
        }

        typename EpilogueOp::Params epilogue_op(
            ElementAccumulator(1.f), ElementAccumulator(0.f));

        typename GemmGrouped::Arguments args(kNumGroups, threadblock_count, epilogue_op,
            // {gmem_a, SHAPE_K},
            {reinterpret_cast<ElementType*>(gmem_a), SHAPE_K},
            // {gmem_b, SHAPE_K},
            {reinterpret_cast<ElementType*>(gmem_b), SHAPE_K},
            // {gmem_d, SHAPE_N},
            {reinterpret_cast<ElementType*>(gmem_d), SHAPE_N},
            shape_m, SHAPE_N, SHAPE_K,
            grouped_layout
        );

        GemmGrouped gemm;

        auto can_implement = gemm.can_implement(args);

        if (can_implement != cutlass::Status::kSuccess)
            printf("Gemm kernel will fail for params. Error: %s\n",
                std::string(cutlassGetStatusString(can_implement)).c_str());

        auto init_status = gemm.initialize(args);

        if(init_status != cutlass::Status::kSuccess)
            printf("Failed to initialize cutlass variable batched gemm. Error: %s\n",
                std::string(cutlassGetStatusString(init_status)).c_str());

        ProfilingInterface::Instance().instrument(true,  dg_prof_params);
        auto run_status = gemm.run(stream);
        ProfilingInterface::Instance().instrument(false, dg_prof_params);

        if(run_status != cutlass::Status::kSuccess)
            printf("Failed to run cutlass variable batched gemm. Error: %s\n",
                std::string(cutlassGetStatusString(run_status)).c_str());
    }
};

};  // namespace deep_gemm

#pragma clang diagnostic pop
