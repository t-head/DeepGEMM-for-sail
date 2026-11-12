#pragma once
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/impls/mqa_logits_types.cuh>


namespace deep_gemm {

// The C++ JIT emits its own `extern "C" __global__` entry that forwards here.
template <uint32_t SPLIT_KV, uint32_t kNumSMs>
__device__ __forceinline__
void smxx_paged_mqa_logits_metadata_device(const uint32_t batch_size, const uint32_t next_n,
                                          const uint32_t* context_lens, uint32_t* schedule_metadata) {
    extern __shared__ uint32_t prefix_sum[];
    const uint32_t tid = threadIdx.x;

    // load context lens (2D indexing: take last token's context_len per request)
    for (uint32_t k = tid; k < batch_size; k += blockDim.x) {
        prefix_sum[k] = math::ceil_div(__ldg(context_lens + k * next_n + next_n - 1), SPLIT_KV);
    }
    __syncthreads();

    // calculate prefix sum
    uint32_t sum = 0;
    uint32_t* temp_prefix_sum = prefix_sum;
    uint32_t loop_num = batch_size / blockDim.x;
    for (uint32_t k = 0; k < loop_num; k++) {
        uint32_t val = temp_prefix_sum[tid];
        for (uint32_t offset = 1; offset < blockDim.x; offset <<= 1) {
            uint32_t temp = 0;
            if (tid >= offset) {
                temp = temp_prefix_sum[tid - offset];
            }
            __syncthreads();
            val += temp;
            temp_prefix_sum[tid] = val;
            // Publish this round before other warps read it in the next round.
            __syncthreads();
        }
        temp_prefix_sum[tid] += sum;
        __syncthreads();
        sum = temp_prefix_sum[blockDim.x - 1];
        temp_prefix_sum += blockDim.x;
    }
    // blockDim.x < 1024: only last loop
    uint32_t last = batch_size - loop_num * blockDim.x;
    uint32_t val = (tid < last) ? temp_prefix_sum[tid] : 0;
    for (uint32_t offset = 1; offset < last; offset <<= 1) {
        uint32_t temp = (tid >= offset && tid < last) ? temp_prefix_sum[tid - offset] : 0;
        __syncthreads();
        val += temp;
        if (tid < last) {
            temp_prefix_sum[tid] = val;
        }
        // All threads must participate, including those outside the tail.
        __syncthreads();
    }
    if (tid < last) {
        temp_prefix_sum[tid] += sum;
    }
    __syncthreads();
    sum = prefix_sum[batch_size - 1];

    // binary search
    const uint32_t& q = sum / kNumSMs, r = sum % kNumSMs;
    for (uint32_t sm_idx = tid; sm_idx < kNumSMs + 1; sm_idx += blockDim.x) {
        uint32_t seg_starts = sm_idx * q + min(sm_idx, r);

        int left = 0;
        int right = batch_size - 1;
        int found_idx = batch_size;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (prefix_sum[mid] > seg_starts) {
                found_idx = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        uint32_t q_idx = found_idx;
        uint32_t prev_sum = (q_idx == 0) ? 0 : prefix_sum[q_idx - 1];
        uint32_t kv_split_idx = seg_starts - prev_sum;

        schedule_metadata[sm_idx * 2] = q_idx;
        schedule_metadata[sm_idx * 2 + 1] = kv_split_idx;
    }
}


template <uint32_t BLOCK_KV, uint32_t kNumMathWarpGroups, uint32_t kNextN>
struct PagedMQALogitsScheduler {
    uint32_t batch_size;
    const uint32_t* context_lens;

    uint32_t current_q_idx, current_kv_idx;
    uint32_t end_q_idx, end_kv_idx;
    uint32_t current_num_kv;

    __device__ __forceinline__ explicit PagedMQALogitsScheduler(const uint32_t& batch_size, const uint32_t& sm_idx,
                                                                const uint32_t* context_lens, const uint32_t* schedule_meta) {
        this->batch_size = batch_size;
        this->context_lens = context_lens;

        const auto& current_pack = __ldg(reinterpret_cast<const uint2*>(schedule_meta) + sm_idx);
        const auto& end_pack = __ldg(reinterpret_cast<const uint2*>(schedule_meta) + sm_idx + 1);
        current_q_idx = current_pack.x, current_kv_idx = current_pack.y * kNumMathWarpGroups;
        end_q_idx = end_pack.x, end_kv_idx = end_pack.y * kNumMathWarpGroups;

        uint32_t idx = current_q_idx * kNextN + kNextN - 1;
        current_num_kv = current_q_idx < batch_size ? math::ceil_div(__ldg(this->context_lens + idx), BLOCK_KV) : 0;
    }

    __device__ __forceinline__ bool fetch_next_task(uint32_t &q_idx, uint32_t &kv_idx, uint32_t &num_kv) {
        q_idx = current_q_idx;
        kv_idx = current_kv_idx;
        num_kv = current_num_kv;

        if (is_last_task(q_idx, kv_idx))
            return false;

        current_kv_idx += kNumMathWarpGroups;
        if (current_kv_idx >= current_num_kv) {
            ++ current_q_idx;
            current_kv_idx = 0;
            uint32_t idx = current_q_idx * kNextN + kNextN - 1;
            current_num_kv = current_q_idx < batch_size ? math::ceil_div(__ldg(this->context_lens + idx), BLOCK_KV) : 0;
        }

        return true;
    }

    __device__ __forceinline__ bool exist_q_idx(const uint32_t& q_idx) const {
        return q_idx < end_q_idx or q_idx == end_q_idx and 0 < end_kv_idx;
    }

    __device__ __forceinline__ bool is_last_task(const uint32_t& q_idx, const uint32_t& kv_idx) const {
        return q_idx == end_q_idx and kv_idx == end_kv_idx;
    }
};

// Request-grouped FP4 paged scheduling.
template <uint32_t BlockQ> __device__ __forceinline__ uint32_t fp4_varlen_q_shift(uint32_t valid) {
    return valid == 1 ? 0 : (valid <= 2 ? 1 : 2);
}
template <uint32_t BlockQ> __device__ __forceinline__ uint32_t fp4_varlen_q_span(uint32_t valid) {
    return 1u << fp4_varlen_q_shift<BlockQ>(valid);
}

template <uint32_t BlockQ, uint32_t SingletonCostMultiplier>
__device__ __forceinline__ uint32_t fp4_varlen_request_unit(const FP4VarlenQBlock *desc, uint32_t q, uint32_t groups) {
    const FP4VarlenQBlock first = desc[q];
    const uint32_t pack = BlockQ >> fp4_varlen_q_shift<BlockQ>(first.valid_q);
    uint32_t unit = 0;
    for (uint32_t i = q; i < groups && desc[i].table_row == first.table_row; ++i)
        unit += desc[i].valid_q;
    return pack == 1 ? unit
                     : first.valid_q * pack *
                           (pack == 4 ? SingletonCostMultiplier : 1);
}

// CTA starts followed by dense (source Q, valid Q, max KV, table row)
// descriptors. Original Q count bounds the allocation, not shared memory.
template <uint32_t BlockQ, uint32_t SplitKV, uint32_t NumWorkers, uint32_t SingletonCostMultiplier>
__device__ __forceinline__ void fp4_varlen_metadata(uint32_t nq, const uint32_t *lengths, const uint32_t *indices,
                                                    FP4VarlenCTAStart *metadata, uint64_t *prefix) {
    constexpr uint32_t Threads = 256;
    const uint32_t tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    __shared__ uint64_t warp_cost[8], carried_cost;
    __shared__ uint32_t warp_count[8], carried_count;
    auto desc = reinterpret_cast<FP4VarlenQBlock*>(metadata + NumWorkers + 1);
    for (uint32_t q = tid; q < nq; q += Threads) {
        desc[q] = FP4VarlenQBlock{0, 0, 0, 0};
    }
    __syncthreads();
    for (uint32_t q = tid; q < nq; q += Threads) {
        if (q && indices[q] == indices[q - 1])
            continue;
        uint32_t end = q + 1;
        while (end < nq && indices[end] == indices[q]) ++end;
        // The caller guarantees this row has the run's maximum context.
        const uint32_t context = lengths[end - 1];
        const uint32_t count = end - q, blocks = math::ceil_div(count, BlockQ);
        const uint32_t base = count / blocks, remainder = count % blocks;
        uint32_t start = q;
        for (uint32_t b = 0; b < blocks; ++b) {
            const uint32_t valid = base + (b < remainder);
            desc[start] = FP4VarlenQBlock{start, valid, context, q};
            start += valid;
        }
    }
    __syncthreads();
    if (!tid)
        carried_count = 0;
    __syncthreads();
    // All inputs of a chunk are read before compacted outputs overwrite them.
    for (uint32_t base = 0; base < nq; base += Threads) {
        const uint32_t q = base + tid;
        FP4VarlenQBlock d = FP4VarlenQBlock{0, 0, 0, 0};
        if (q < nq) {
            const volatile FP4VarlenQBlock* source = desc + q;
            d = FP4VarlenQBlock{source->source_q, source->valid_q, source->context_len, source->table_row};
        }
        const bool valid = d.valid_q != 0;
        uint32_t count = valid;
#pragma unroll
        for (uint32_t offset = 1; offset < 32; offset <<= 1) {
            uint32_t n = __shfl_up_sync(0xffffffffu, count, offset);
            if (lane >= offset)
                count += n;
        }
        if (lane == 31)
            warp_count[warp] = count;
        __syncthreads();
        if (!tid) {
            uint32_t n = carried_count;
#pragma unroll
            for (uint32_t w = 0; w < 8; ++w) {
                const uint32_t next_n = n + warp_count[w];
                warp_count[w] = n;
                n = next_n;
            }
            carried_count = n;
        }
        __syncthreads();
        if (valid) {
            const uint32_t out = warp_count[warp] + count - 1;
            desc[out] = d;
        }
        __syncthreads();
    }
    const uint32_t groups = carried_count;
    if (!tid)
        carried_cost = 0;
    __syncthreads();
    for (uint32_t base = 0; base < groups; base += Threads) {
        const uint32_t q = base + tid;
        uint64_t cost = 0;
        if (q < groups) {
            const uint32_t pack = BlockQ / fp4_varlen_q_span<BlockQ>(desc[q].valid_q);
            if (desc[q].source_q == desc[q].table_row)
                cost = uint64_t(math::ceil_div(desc[q].context_len, SplitKV * pack)) *
                       fp4_varlen_request_unit<BlockQ, SingletonCostMultiplier>(desc, q, groups);
        }
#pragma unroll
        for (uint32_t offset = 1; offset < 32; offset <<= 1) {
            const uint32_t lo = __shfl_up_sync(0xffffffffu, uint32_t(cost), offset);
            const uint32_t hi = __shfl_up_sync(0xffffffffu, uint32_t(cost >> 32), offset);
            if (lane >= offset)
                cost += (uint64_t(hi) << 32) | lo;
        }
        if (lane == 31)
            warp_cost[warp] = cost;
        __syncthreads();
        if (!tid) {
            uint64_t c = carried_cost;
#pragma unroll
            for (uint32_t w = 0; w < 8; ++w) {
                const uint64_t next = c + warp_cost[w];
                warp_cost[w] = c;
                c = next;
            }
            carried_cost = c;
        }
        __syncthreads();
        if (q < groups)
            prefix[q] = warp_cost[warp] + cost;
        __syncthreads();
    }
    const uint64_t total = carried_cost;
    for (uint32_t worker = tid; worker <= NumWorkers; worker += Threads) {
        // floor(total * worker / NumWorkers), without overflowing uint64_t.
        // Divide proportional cost intervals before aligning request boundaries.
        const uint64_t target = (total / NumWorkers) * worker + ((total % NumWorkers) * worker) / NumWorkers;
        uint32_t lo = 0, hi = groups;
        while (lo < hi) {
            const uint32_t mid = lo + (hi - lo) / 2;
            if (prefix[mid] <= target)
                lo = mid + 1;
            else
                hi = mid;
        }
        const uint64_t before = lo ? prefix[lo - 1] : 0;
        uint32_t split = 0;
        if (lo < groups) {
            const uint32_t unit = fp4_varlen_request_unit<BlockQ, SingletonCostMultiplier>(desc, lo, groups);
            const uint64_t from_start = target - before, to_end = prefix[lo] - target;
            // Move at most one split, within half a worker's cost budget,
            // to avoid loading two requests' Q for a nearly aligned interval.
            const uint64_t margin = min(uint64_t(unit), total / NumWorkers / 2);
            if (min(from_start, to_end) <= margin) {
                if (to_end < from_start) {
                    const uint64_t boundary = prefix[lo];
                    do { ++lo; } while (lo < groups && prefix[lo] <= boundary);
                }
            } else {
                split = uint32_t(from_start / unit);
            }
        }
        metadata[worker] = FP4VarlenCTAStart{lo, split, groups, SplitKV};
    }
    __syncthreads();
}

// Owns worker permutation, request ranges and two-Q-block reuse windows.
template <uint32_t BlockQ, uint32_t SplitKV, uint32_t BlockKV, uint32_t WorkersPerCU>
struct FP4GroupedPagedScheduler {
    const FP4VarlenCTAStart* starts;
    const FP4VarlenQBlock* groups;
    __device__ __forceinline__ FP4GroupedPagedScheduler(const FP4VarlenCTAStart* starts,
                                                        const FP4VarlenQBlock* groups)
        : starts(starts), groups(groups) {}
    template <typename RequestFn, typename SingleFn>
    __device__ __forceinline__ void for_each_task(RequestFn&& on_request, SingleFn&& on_single) const {
        const uint32_t num_cus = gridDim.x / WorkersPerCU;
        const uint32_t slot = blockIdx.x / num_cus;
        // Offset each scheduling layer to spread sparse nonempty ranges
        // across CU positions while preserving the original cost mixing.
        const uint32_t worker = slot * num_cus + (blockIdx.x % num_cus + slot) % num_cus;
        const FP4VarlenCTAStart first = starts[worker], last = starts[worker + 1];
        constexpr uint32_t MathGroups = SplitKV / BlockKV;
        for (uint32_t q = first.group; q <= last.group && q < starts[0].num_groups;) {
            if (q == last.group && last.kv_split == 0) break;
            const FP4VarlenQBlock d = groups[q];
            uint32_t run_end = q + 1;
            while (run_end < starts[0].num_groups && groups[run_end].table_row == d.table_row) ++run_end;
            // Q spans are powers of two. Keep the quotient explicit so local
            // task dispatch cannot turn it into a runtime integer division.
            const uint32_t pack = BlockQ >> fp4_varlen_q_shift<BlockQ>(d.valid_q);
            const uint32_t begin = q == first.group ? first.kv_split * MathGroups * pack : 0;
            const uint32_t end = q == last.group ? last.kv_split * MathGroups * pack
                : deep_gemm::math::ceil_div(d.context_len, BlockKV);
            if (begin < end) {
                if (run_end > q + 1 || (BlockQ == 4 && d.valid_q == 3)) {
                    for (uint32_t window = q; window < run_end; window += 2) {
                        const uint32_t last_block = min(window + 2, run_end);
                        const uint32_t count0 = groups[window].valid_q;
                        const uint32_t count1 = window + 1 < last_block ? groups[window + 1].valid_q : 0;
                        on_request(window, begin, end, last_block, count0, count1);
                    }
                } else {
                    on_single(q, begin, end, d.valid_q);
                }
            }
            q = run_end;
        }
    }
};

template <uint32_t BlockKV, uint32_t MathGroups, uint32_t BlockQ> struct FP4SingleGroupScheduler {
    const uint32_t current_q_idx, end_kv_idx, current_num_kv, step;
    uint32_t current_kv_idx;
    __device__ __forceinline__ FP4SingleGroupScheduler(uint32_t q, uint32_t begin, uint32_t end, uint32_t context,
                                                       uint32_t pack)
        : current_q_idx(q), end_kv_idx(end), current_num_kv(math::ceil_div(context, BlockKV)), step(MathGroups * pack),
          current_kv_idx(begin) {}
    __device__ __forceinline__ uint32_t next_q(uint32_t q) const { return q + 1; }
    __device__ __forceinline__ bool exist_q_idx(uint32_t) const { return false; }
    __device__ __forceinline__ bool is_last_task(uint32_t, uint32_t kv) const { return kv >= end_kv_idx; }
    __device__ __forceinline__ bool fetch_next_task(uint32_t &q, uint32_t &kv, uint32_t &count) {
        q = current_q_idx;
        kv = current_kv_idx;
        count = current_num_kv;
        current_kv_idx += step;
        return kv < end_kv_idx;
    }
};

} // namespace deep_gemm
