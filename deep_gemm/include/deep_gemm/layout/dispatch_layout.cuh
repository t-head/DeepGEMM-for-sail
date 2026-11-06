#pragma once

#include <cstdint>
#include <deep_gemm/common/math.cuh>

namespace deep_gemm {

// preprocess_ready_flags[0] is also the graph-safe round handoff. Quant writes
// the raw generation; fused GEMM1 sets this bit after metadata is complete.
static constexpr uint32_t kPreprocessMetadataReadyBit = 1u << 31;
// Copy flags encode (generation << 3) | K-stripe watermark.
// Wrap to generation 2 after clearing flags, preserving parity alternation.
static constexpr uint32_t kFusedRoundGenerationLimit = (1u << 27) - 1u;

// M-block count, block offset, row offset and token count per expert.
static constexpr int kExpertMetadataAuxFields = 4;

// Metadata owner search requires a power-of-two rank count.
__host__ __device__
static constexpr bool is_supported_megamoe_rank_count(uint32_t num_ranks) {
    return num_ranks >= 1 && num_ranks <= 64 &&
        (num_ranks & (num_ranks - 1)) == 0;
}

// Map packed-scale producer partitions to GEMM K tiles.
// SFA readiness uses this mapping independently of FP4 copy stripes.
struct FusedSfaPartitionLayout {
    static constexpr uint32_t kNumPartitions = 3;
    // One int4 load/store in the reshaper carries eight uint16_t scale blocks.
    static constexpr uint32_t kScaleBlocksPerVector = 8;
    // Align SFA rows for hardware memory accesses.
    static constexpr uint32_t kRowAlignment = 4;

    template <typename T>
    __host__ __device__
    static constexpr T align_rows(T rows) {
        return math::constexpr_align<T>(rows, kRowAlignment);
    }

    template <typename T>
    __host__ __device__
    static constexpr T row_offset(T compact_prefix, uint32_t experts_before) {
        return align_rows(compact_prefix) + experts_before * kRowAlignment;
    }

    __host__ __device__
    static constexpr uint32_t partition_begin_vector(
            uint32_t scale_blocks, uint32_t partition) {
        const uint32_t vectors = scale_blocks / kScaleBlocksPerVector;
        return (vectors * partition + kNumPartitions - 1) /
            kNumPartitions;
    }

    __host__ __device__
    static constexpr uint32_t partition_begin_scale_block(
            uint32_t scale_blocks, uint32_t partition) {
        return partition_begin_vector(scale_blocks, partition) *
            kScaleBlocksPerVector;
    }

    __host__ __device__
    static constexpr uint32_t first_k_tile(
            uint32_t scale_blocks, uint32_t block_k,
            uint32_t partition) {
        // Use floor, not ceil: when a K tile straddles a producer boundary it
        // must wait for both partitions before issuing the SFA load.
        return partition_begin_scale_block(scale_blocks, partition) /
            (block_k / 32);
    }
};

}  // namespace deep_gemm
