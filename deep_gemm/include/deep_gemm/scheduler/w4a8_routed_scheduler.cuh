#pragma once
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>
#include <deep_gemm/common/w4a8_arguments.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
namespace deep_gemm::w4fa8 {
using namespace cute;
struct Triple {
  int r, p, b;
};
struct Add {
  CUTLASS_DEVICE Triple operator()(Triple a, Triple b) const { return {a.r + b.r, a.p + b.p, a.b + b.b}; }
};
template <int N, int BM, int BN> struct RoutedWorkScheduler {
  const uint4* records;
  int next = 0, begin, end, start;
  int expert, rows, out_row, token_base, valid;
  __device__ RoutedWorkScheduler(const uint8_t* meta, int chunkid, int start_)
      : records(reinterpret_cast<const uint4*>(meta + kRoutedMetadataBytes)), start(start_) {
    auto p = reinterpret_cast<const int32_t*>(meta);
    begin = __ldg(p + 2 * chunkid);
    end = __ldg(p + 2 * chunkid + 1);
  }
  __device__ bool fetch(uint32_t& mb, uint32_t& nb) {
    constexpr int NC = N / BN;
    int work = (next++) * gridDim.x + blockIdx.x + begin * NC;
    if (work >= end * NC)
      return false;
    auto d = __ldg(records + 2 * (work / NC));
    auto t = __ldg(records + 2 * (work / NC) + 1);
    int count = (d.y + BM - 1) / BM, local = work - int(d.z) * NC;
    int first = (local / (NC * 2)) * 2, in = local % (NC * 2), width = min(2, count - first);
    mb = first + in % width;
    nb = in / width;
    expert = int(d.x) - start;
    rows = d.y;
    out_row = d.w + mb * BM;
    token_base = t.x + mb * BM;
    valid = min(BM, rows - int(mb) * BM);
    return true;
  }
};
template <int N, int K, int BM, int BN> struct W4A8RoutedScheduler {
  struct Params {
    const uint8_t* meta;
  };
  RoutedWorkScheduler<N, BM, BN> inner;
  int curr_group_idx, curr_group_m, curr_block_m_offset, valid_m_in_block, token_base, group_base;
  CUTLASS_DEVICE explicit W4A8RoutedScheduler(Params p) : inner(p.meta, 0, 0) {}
  CUTLASS_DEVICE bool fetch_next_work(uint32_t& m, uint32_t& n) {
    if (!inner.fetch(m, n))
      return false;
    curr_group_idx = inner.expert;
    curr_group_m = inner.rows;
    curr_block_m_offset = inner.out_row;
    valid_m_in_block = inner.valid;
    token_base = inner.token_base;
    group_base = inner.out_row - int(m) * BM;
    return true;
  }
  CUTLASS_DEVICE int64_t curr_offset_a() { return int64_t(group_base) * K; }
  CUTLASS_DEVICE int64_t curr_offset_c() { return int64_t(group_base) * N; }
};
template <int Groups, int SmallM, int LargeM, int TokenBlockM>
CUTLASS_DEVICE void build_w4a8_routed_metadata(const int32_t* rows, uint8_t* lower, uint8_t* upper) {
  const int tid = threadIdx.x;
  constexpr int Items = (Groups + 255) / 256;
  int row_count[Items], extra = 0;
#pragma unroll
  for (int j = 0; j < Items; ++j) {
    const int e = tid * Items + j;
    const int r = e < Groups ? __ldg(rows + e) : 0;
    row_count[j] = r;
    extra += (r + SmallM - 1) / SmallM - (r + LargeM - 1) / LargeM;
  }
  using Reduce = cub::BlockReduce<int, 256>;
  __shared__ typename Reduce::TempStorage reduction;
  __shared__ int small;
  const int total = Reduce(reduction).Sum(extra);
  if (tid == 0)
    small = total == 0;
  __syncthreads();
  if (tid == 0) {
    auto* inactive = reinterpret_cast<int32_t*>(small ? upper : lower);
    inactive[0] = inactive[1] = 0;
  }
  Triple counts[Items], sum{0, 0, 0};
#pragma unroll
  for (int j = 0; j < Items; ++j) {
    const int r = row_count[j];
    counts[j] = {r, (r + TokenBlockM - 1) / TokenBlockM * TokenBlockM,
                 small ? (r + SmallM - 1) / SmallM : (r + LargeM - 1) / LargeM};
    sum = Add{}(sum, counts[j]);
  }
  using Scan = cub::BlockScan<Triple, 256>;
  __shared__ typename Scan::TempStorage scan;
  Triple prefix;
  Scan(scan).ExclusiveScan(sum, prefix, Triple{0, 0, 0}, Add{});
  auto* range = reinterpret_cast<int32_t*>(small ? lower : upper);
  auto* records = reinterpret_cast<uint4*>((small ? lower : upper) + kRoutedMetadataBytes);
#pragma unroll
  for (int j = 0; j < Items; ++j) {
    const int e = tid * Items + j;
    const auto count = counts[j];
    if (e < Groups) {
      if (e == 0)
        range[0] = prefix.b;
      if (e == Groups - 1)
        range[1] = prefix.b + count.b;
      for (int tile = 0; tile < count.b; ++tile) {
        records[2 * (prefix.b + tile)] = make_uint4(e, count.r, prefix.b, prefix.r);
        records[2 * (prefix.b + tile) + 1] = make_uint4(prefix.p, 0, 0, 0);
      }
    }
    prefix = Add{}(prefix, count);
  }
}
} // namespace deep_gemm::w4fa8
