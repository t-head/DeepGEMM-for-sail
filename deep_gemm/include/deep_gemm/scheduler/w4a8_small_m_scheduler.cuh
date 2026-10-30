#pragma once
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>

namespace deep_gemm {
template <GemmType Kind, uint32_t N, uint32_t K, uint32_t BM, uint32_t BN, uint32_t Groups, uint32_t NB,
          uint32_t Swizzle>
struct WarpPrefixScheduler : DeepGemmScheduler<Kind, N, K, BM, BN, Groups, NB, Swizzle> {
  using Base = DeepGemmScheduler<Kind, N, K, BM, BN, Groups, NB, Swizzle>;
  using Params = typename Base::Params;
  static_assert(Kind == GemmType::MGroupedNoPad && Groups <= 64);
  uint32_t row0, row1, endrow0, endrow1, endm0, endm1, total;
  CUTLASS_DEVICE explicit WarpPrefixScheduler(Params const& p) : Base(p) {
    const unsigned lane = threadIdx.x % 32;
    row0 = lane < Groups ? static_cast<uint32_t>(__ldg(p.grouped_layout + lane)) : 0u;
    row1 = lane + 32u < Groups ? static_cast<uint32_t>(__ldg(p.grouped_layout + lane + 32u)) : 0u;
    endrow0 = row0;
    endrow1 = row1;
    endm0 = (row0 + BM - 1) / BM;
    endm1 = (row1 + BM - 1) / BM;
#pragma unroll
    for (unsigned d = 1; d < 32; d *= 2) {
      uint32_t r0 = __shfl_up_sync(0xffffffffu, endrow0, d), r1 = __shfl_up_sync(0xffffffffu, endrow1, d);
      uint32_t m0 = __shfl_up_sync(0xffffffffu, endm0, d), m1 = __shfl_up_sync(0xffffffffu, endm1, d);
      if (lane >= d) {
        endrow0 += r0;
        endrow1 += r1;
        endm0 += m0;
        endm1 += m1;
      }
    }
    endrow1 += __shfl_sync(0xffffffffu, endrow0, 31);
    endm1 += __shfl_sync(0xffffffffu, endm0, 31);
    total = __shfl_sync(0xffffffffu, endm1, 31) * NB;
  }
  CUTLASS_DEVICE bool fetch_next_work(uint32_t& mi, uint32_t& ni) {
    const uint32_t next = (this->current_iter++) * gridDim.x + blockIdx.x;
    if (next >= total)
      return false;
    const uint32_t mt = next / NB;
    const bool first = mt < __shfl_sync(0xffffffffu, endm0, 31);
    const uint32_t mask = __ballot_sync(0xffffffffu, mt < (first ? endm0 : endm1));
    const int lane = __ffs(mask) - 1;
    this->curr_group_idx = lane + (first ? 0 : 32);
    this->curr_group_m = __shfl_sync(0xffffffffu, first ? row0 : row1, lane);
    this->curr_cumsum_m = __shfl_sync(0xffffffffu, first ? endrow0 - row0 : endrow1 - row1, lane);
    const uint32_t nm = (this->curr_group_m + BM - 1) / BM;
    this->curr_cumsum = __shfl_sync(0xffffffffu, first ? endm0 : endm1, lane) - nm;
    this->get_swizzled_block_idx(nm, next - this->curr_cumsum * NB, mi, ni);
    return true;
  }
};
} // namespace deep_gemm

namespace deep_gemm {
template <GemmType Kind, uint32_t N, uint32_t K, uint32_t BM, uint32_t BN, uint32_t Groups, uint32_t NB,
          uint32_t Swizzle>
struct SharedWarpPrefixScheduler : DeepGemmScheduler<Kind, N, K, BM, BN, Groups, NB, Swizzle> {
  using Base = DeepGemmScheduler<Kind, N, K, BM, BN, Groups, NB, Swizzle>;
  using Params = typename Base::Params;
  static_assert(Kind == GemmType::MGroupedNoPad && Groups <= 64);
  struct alignas(16) State {
    uint32_t end_rows[64];
    uint32_t end_m[64];
  };
  State& state;
  uint32_t total;
  CUTLASS_DEVICE explicit SharedWarpPrefixScheduler(Params const& p, State& shared) : Base(p), state(shared) {
    const unsigned lane = threadIdx.x % 32;
    if (threadIdx.x / 32 == 0) {
      uint32_t r0 = lane < Groups ? static_cast<uint32_t>(__ldg(p.grouped_layout + lane)) : 0u;
      uint32_t r1 = lane + 32u < Groups ? static_cast<uint32_t>(__ldg(p.grouped_layout + lane + 32u)) : 0u;
      uint32_t e0 = r0, e1 = r1, m0 = (r0 + BM - 1) / BM, m1 = (r1 + BM - 1) / BM;
#pragma unroll
      for (unsigned d = 1; d < 32; d *= 2) {
        uint32_t a = __shfl_up_sync(0xffffffffu, e0, d), b = __shfl_up_sync(0xffffffffu, e1, d);
        uint32_t c = __shfl_up_sync(0xffffffffu, m0, d), f = __shfl_up_sync(0xffffffffu, m1, d);
        if (lane >= d) {
          e0 += a;
          e1 += b;
          m0 += c;
          m1 += f;
        }
      }
      e1 += __shfl_sync(0xffffffffu, e0, 31);
      m1 += __shfl_sync(0xffffffffu, m0, 31);
      state.end_rows[lane] = e0;
      state.end_rows[lane + 32] = e1;
      state.end_m[lane] = m0;
      state.end_m[lane + 32] = m1;
    }
    __syncthreads();
    total = state.end_m[63] * NB;
  }
  CUTLASS_DEVICE bool fetch_next_work(uint32_t& mi, uint32_t& ni) {
    const uint32_t next = (this->current_iter++) * gridDim.x + blockIdx.x;
    if (next >= total)
      return false;
    const unsigned lane = threadIdx.x % 32;
    const uint32_t mt = next / NB, end0 = state.end_m[lane], end1 = state.end_m[lane + 32];
    const bool first = mt < __shfl_sync(0xffffffffu, end0, 31);
    const uint32_t mask = __ballot_sync(0xffffffffu, mt < (first ? end0 : end1));
    const int found = __ffs(mask) - 1, group = found + (first ? 0 : 32);
    const uint32_t begin = group ? state.end_rows[group - 1] : 0u;
    this->curr_group_idx = group;
    this->curr_group_m = state.end_rows[group] - begin;
    this->curr_cumsum_m = begin;
    const uint32_t nm = (this->curr_group_m + BM - 1) / BM;
    this->curr_cumsum = __shfl_sync(0xffffffffu, first ? end0 : end1, found) - nm;
    this->get_swizzled_block_idx(nm, next - this->curr_cumsum * NB, mi, ni);
    return true;
  }
};
} // namespace deep_gemm

namespace deep_gemm {
template <GemmType Kind, uint32_t N, uint32_t K, uint32_t BM, uint32_t BN, uint32_t Groups, uint32_t NB,
          uint32_t Swizzle, bool Fast>
struct WarpPrefixPairScheduler : WarpPrefixScheduler<Kind, N, K, BM, BN, Groups, NB, Swizzle> {
  using Base = WarpPrefixScheduler<Kind, N, K, BM, BN, Groups, NB, Swizzle>;
  using Params = typename Base::Params;
  static_assert(Swizzle == 2);
  CUTLASS_DEVICE explicit WarpPrefixPairScheduler(Params const& p) : Base(p) {}
  CUTLASS_DEVICE bool fetch_next_work(uint32_t& mi, uint32_t& ni) {
    const uint32_t next = (this->current_iter++) * gridDim.x + blockIdx.x;
    if (next >= this->total)
      return false;
    const uint32_t mt = next / NB;
    const bool first = mt < __shfl_sync(0xffffffffu, this->endm0, 31);
    const uint32_t mask = __ballot_sync(0xffffffffu, mt < (first ? this->endm0 : this->endm1));
    const int lane = __ffs(mask) - 1;
    this->curr_group_idx = lane + (first ? 0 : 32);
    this->curr_group_m = __shfl_sync(0xffffffffu, first ? this->row0 : this->row1, lane);
    this->curr_cumsum_m =
        __shfl_sync(0xffffffffu, first ? this->endrow0 - this->row0 : this->endrow1 - this->row1, lane);
    const uint32_t nm = (this->curr_group_m + BM - 1) / BM;
    this->curr_cumsum = __shfl_sync(0xffffffffu, first ? this->endm0 : this->endm1, lane) - nm;
    const uint32_t index = next - this->curr_cumsum * NB;
    if constexpr (!Fast)
      this->get_swizzled_block_idx(nm, index, mi, ni);
    else if (nm == 1) {
      mi = 0;
      ni = index;
    } else {
      const uint32_t group = index / (NB * 2), inside = index % (NB * 2), start = group * 2;
      if (start + 1 < nm) {
        mi = start + (inside & 1u);
        ni = inside >> 1u;
      } else {
        mi = start;
        ni = inside;
      }
    }
    return true;
  }
};
} // namespace deep_gemm

namespace deep_gemm {
template <class Storage, class Scheduler, bool Shared> struct SmallMSchedulerStorage : Storage {};

template <class Storage, class Scheduler> struct SmallMSchedulerStorage<Storage, Scheduler, true> : Storage {
  typename Scheduler::State scheduler_state;
};
} // namespace deep_gemm
