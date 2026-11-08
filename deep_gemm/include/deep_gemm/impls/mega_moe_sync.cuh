#pragma once

#include <deep_gemm/comm/barrier.cuh>

namespace deep_gemm {

template <uint32_t NumRanks>
__global__ void mega_moe_dispatch_sync(const layout::MegaMoESignals* signals) {
    const uint32_t generation = signals->preprocess_ready[0] & ~kPreprocessMetadataReadyBit;
    comm::rank_barrier<NumRanks>(signals->dispatch_arrivals, generation);
}

template <uint32_t NumRanks>
__global__ void mega_moe_scatter_sync(layout::MegaMoESignals* signals) {
    const uint32_t generation = signals->preprocess_ready[0] & ~kPreprocessMetadataReadyBit;
    __shared__ uint32_t mask[2];
    comm::rank_barrier<NumRanks, true>(signals->scatter_arrivals, generation, mask);
    if (threadIdx.x == 0)
        signals->scatter_writer_mask = (uint64_t(mask[1]) << 32) | mask[0];
}

}  // namespace deep_gemm
