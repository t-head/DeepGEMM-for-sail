#pragma once

#include <cuda.h>
#include <cstdint>

namespace deep_gemm {
namespace profiling {

struct GemmProfileRecord {
  uint64_t aiu_issue_cycles;
  uint64_t aiu_wait_cycles;
  uint64_t gemm_compute_cycles;
  uint64_t smem_copy_cycles;
  uint64_t tile_total_cycles;
  uint32_t num_tiles;
  uint32_t sm_id;
  uint64_t sync_cycles;
  uint32_t _pad[2];
};
static_assert(sizeof(GemmProfileRecord) == 64, "GemmProfileRecord must be 64 bytes");

struct GemmProfiler {
  GemmProfileRecord* records;
  bool active;

  __device__ __forceinline__ void init() {
    if (!active || threadIdx.x != 0) return;
    auto& rec = records[blockIdx.x];
    rec.aiu_issue_cycles = 0;
    rec.aiu_wait_cycles = 0;
    rec.gemm_compute_cycles = 0;
    rec.smem_copy_cycles = 0;
    rec.tile_total_cycles = 0;
    rec.sync_cycles = 0;
    rec.num_tiles = 0;
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    rec.sm_id = smid;
  }

  __device__ __forceinline__ uint64_t timestamp() const {
    return clock64();
  }

  __device__ __forceinline__ void accum_aiu_issue(uint64_t start, uint64_t end) {
    if (!active || threadIdx.x != 0) return;
    records[blockIdx.x].aiu_issue_cycles += (end - start);
  }

  __device__ __forceinline__ void accum_aiu_wait(uint64_t start, uint64_t end) {
    if (!active || threadIdx.x != 0) return;
    records[blockIdx.x].aiu_wait_cycles += (end - start);
  }

  __device__ __forceinline__ void accum_gemm_compute(uint64_t start, uint64_t end) {
    if (!active || threadIdx.x != 0) return;
    records[blockIdx.x].gemm_compute_cycles += (end - start);
  }

  __device__ __forceinline__ void accum_smem_copy(uint64_t start, uint64_t end) {
    if (!active || threadIdx.x != 0) return;
    records[blockIdx.x].smem_copy_cycles += (end - start);
  }

  __device__ __forceinline__ void accum_sync(uint64_t start, uint64_t end) {
    if (!active || threadIdx.x != 0) return;
    records[blockIdx.x].sync_cycles += (end - start);
  }

  __device__ __forceinline__ void accum_tile_total(uint64_t start, uint64_t end) {
    if (!active || threadIdx.x != 0) return;
    records[blockIdx.x].tile_total_cycles += (end - start);
    records[blockIdx.x].num_tiles += 1;
  }
};

} // namespace profiling
} // namespace deep_gemm
