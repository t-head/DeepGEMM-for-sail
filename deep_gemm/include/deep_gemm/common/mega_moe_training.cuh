#pragma once

#include <cstdint>

namespace deep_gemm::mega_moe_training {

// Fixed POD ABI shared by the host dispatcher and the JIT device entry points.
// Tensor storage remains caller-owned for the entire asynchronous operation.
struct Params {
    void* ptr[12]{};
    int64_t dim[12]{};
};

enum class Operation : int {
    CountRoutes,
    PrefixRoutes,
    InitializeLayout,
    PackInput,
    UnpackRoutes,
    PackRouteGrad,
    Combine,
    Activation,
    ActivationBackward,
    QuantizeRows,
    QuantizeWeights,
    WeightedGradient,
    ScoreGradient,
    ReadChunk,
    ActivationQuantize,
    ActivationBackwardQuantize,
    ActivationDeviceRows,
    TransformDeviceRows,
    ZeroDeviceRows,
    BuildPermutationOnly,
    PackInputDeviceRows,
    TransformDeviceRowsVector,
    TransposeWeights,
    DeepEPReduceRows,
    DeepEPBackwardRows,
    DeepEPReduceWindowRows,
    QuantizeRowsDeviceRows,
    ActivationQuantizeDeviceRows,
    ActivationBackwardQuantizeDeviceRows,
    QuantizeWeightsCachedFloat,
    QuantizeWeightsCachedInteger,
    PackQuantizedRouteRows,
    TransposeQuantizedWeights,
    PackFp8WeightGradients,
    QuantizeRowsVectorDeviceRows,
    TransposeQuantizedWeightsSwizzled,
    QuantizeWeightsAndTranspose,
    DeepEPZeroActiveReceive,
    DeepEPCastActiveReceive,
    DeepEPReduceCompactReceive,
    PrefixCompactRoutes,
    CompactReplayPrefix,
};

}  // namespace deep_gemm::mega_moe_training


#include <deep_gemm/common/fused_gemm_common.cuh>

namespace deep_gemm::mega_moe_training {

enum class MegaMoEOperation : int {
    CountRoutes,
    InitializeLayout,
    BuildPermutation,
    BuildWorkInfo,
    Pack,
    Barrier,
    Scatter,
    PackActive,
    ActivationActive,
    ActivationBackwardActive,
    LocalGradients,
    PackActiveVector,
    ScatterVector,
    ExpertLocalGradients,
    SetRouteWindow,
    PrefetchRawDy,
    AssistInputTiles,
    ReduceScorePartials,
    StageRawWeightedDy,
    StageFp8InputTiles,
    PrepareContextReplay,
    PackActiveQuantized,
    ExpertLocalGradientsQuantized,
    ExpertLocalGradientsQuantizedTiles,
    ExpertLocalGradientsQuantizedTilesCached,
    PackQuantizedInputTiles,
    ReturnGemmTiles,
    GradientQuantizedInputTiles,
    PackExpertBatch,
    GradientExpertBatch,
    FinalizeEmptyWgrad=43,
    AggregateFirstWgradFlags=44,
    BuildGemmMetadata=45,
};

struct MegaMoEGemmParams {
    GemmArgs gemm;
    const uint64_t* source_peers;
    const uint64_t* destination_peers;
    const int32_t* permutation;
    uint32_t source_rows_per_rank;
    uint32_t destination_rows_per_rank;
    uint32_t num_ranks;
    bool pack_input;
    void* activation_gate_up = nullptr;
    void* dispatch_scratch = nullptr;
    uint32_t return_local_rank = 0;
};

// Optional FP8 gradient producer state; existing GPU argument structs stay unchanged.
struct MegaMoEFp8ProducerGradient {
    const uint64_t* score_peers = nullptr;
    const void* unweighted_y = nullptr;
    float* score_partials = nullptr;
};

// Separate launch argument preserves the existing MegaMoEGemmParams ABI.
struct MegaMoEDispatchPipeline {
    uint32_t* ready = nullptr;
    uint32_t* status = nullptr;
    uint64_t* statistics = nullptr;
    uint32_t communication_ctas = 0;
    uint64_t timeout_cycles = uint64_t(180) * 2'000'000'000;
    uint32_t signal_capacity = 0;
    bool serialize = false;
};

// Separate optional launch argument: preserve MegaMoEGemmParams and old kernels.
struct MegaMoEReplayGradients {
    const uint64_t* dy_peers = nullptr;
    const uint64_t* score_peers = nullptr;
    void* packed_dy = nullptr;
    float* score_partials = nullptr;
    uint32_t* status = nullptr;
    uint32_t source_tokens = 0;
    uint32_t topk = 0;
    uint32_t num_ranks = 0;
    uint32_t num_n_tiles = 0;
};

// Optional dHidden epilogue argument, separate from existing GPU ABI structs.
struct MegaMoEDHiddenScores {
    const void* hidden = nullptr;
    const float* row_scores = nullptr;
    float* score_partials = nullptr;
    uint32_t num_n_tiles = 0;
    const uint64_t* score_peers = nullptr;
    void* packed_dy = nullptr;
    uint32_t* status = nullptr;
};

}  // namespace deep_gemm::mega_moe_training

namespace deep_gemm::mega_moe_training {
struct MegaMoEFp8InlineSource {
    const uint64_t* value_peers=nullptr;
    const uint64_t* aux_peers=nullptr;
    const int32_t* permutation=nullptr;
    const void* unweighted_y=nullptr;
    void* packed_dy=nullptr;
    float* score_partials=nullptr;
    const float* broadcast_sf_tile=nullptr;
    const float* output_row_scores=nullptr;
    uint32_t* ready=nullptr;
    uint32_t* status=nullptr;
    uint32_t tokens=0,topk=0,num_ranks=0,signal_capacity=0;
    uint64_t timeout_cycles=uint64_t(20)*2'000'000'000;
    uint32_t local_rank=0;
};
}


namespace deep_gemm::mega_moe_training {
struct DwActivationOperands {
    const void* x=nullptr;
    void* quantized_x=nullptr;
    float* scales_x=nullptr;
    int32_t hidden=0;
};
}


#include <cstdint>
namespace deep_gemm::mega_moe_training {
struct Fp8WgradParams {
    const int32_t* lengths;
    const int32_t* offsets;
    float* output;
    uint32_t* status;
    uint32_t groups, m, n, capacity;
};
}


namespace deep_gemm::mega_moe_training {
// Separate optional operands preserve the established12-pointer Params ABI.
struct MegaMoEDwBlock128Operands {
    void* quantized0=nullptr;
    float* scales0=nullptr;
    void* quantized1=nullptr;
    float* scales1=nullptr;
    const void* hidden=nullptr;
    int32_t intermediate=0;
};
}


#include <cstdint>
namespace deep_gemm {
// Host/device launch parameter: fixed-size, no device allocation or pointer chase.
struct KGroupedSegmentArgs {
    const void* a[8]{};
    const void* b[8]{};
    const int32_t* offsets[8]{};
};
} // namespace deep_gemm
