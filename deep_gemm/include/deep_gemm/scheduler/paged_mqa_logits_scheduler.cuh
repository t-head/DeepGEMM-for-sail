#pragma once
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/impls/mqa_logits_types.cuh>


namespace deep_gemm {

// Grouped FP8 uses contiguous query rows within a request. Workers start on
// group boundaries, so a bounded forward scan also handles the final tail.
template <uint32_t GroupQ>
__device__ __forceinline__ uint32_t paged_mqa_group_size(
        const uint32_t* indices, uint32_t row, uint32_t nq) {
    static_assert(GroupQ > 0 && GroupQ < 32);
    const uint32_t id = __ldg(indices + row);
    const uint32_t available = min(GroupQ, nq - row);
    // Read the small window independently. The sentinel ends a tail with no ID change.
    uint32_t boundaries = 1u << available;
    #pragma unroll
    for (uint32_t i = 1; i < GroupQ; ++i) {
        const uint32_t next_id = __ldg(indices + row + min(i, available - 1));
        boundaries |= static_cast<uint32_t>(next_id != id) << i;
    }
    return __ffs(boundaries) - 1;
}

__device__ __forceinline__ uint32_t paged_mqa_group_context(
        const uint32_t* lengths, uint32_t row, uint32_t count) {
    uint32_t length = 0;
    for (uint32_t i = 0; i < count; ++i)
        length = max(length, __ldg(lengths + row + i));
    return length;
}

// The C++ JIT emits its own `extern "C" __global__` entry that forwards here.
template <uint32_t SPLIT_KV, uint32_t kNumSMs, uint32_t GroupQ = 0>
__device__ __forceinline__
void smxx_paged_mqa_logits_metadata_device(const uint32_t batch_size, const uint32_t next_n,
                                          const uint32_t* context_lens, uint32_t* schedule_metadata,
                                          const uint32_t* indices = nullptr) {
    extern __shared__ uint32_t prefix_sum[];
    const uint32_t tid = threadIdx.x;

    if constexpr (GroupQ != 0) {
        for (uint32_t q = tid; q < batch_size; q += blockDim.x)
            prefix_sum[q] = 0;
        __syncthreads();
        // A run head owns its group's task counts. Interior rows contribute
        // zero so the same prefix scan emits flattened group starts.
        for (uint32_t q = tid; q < batch_size; q += blockDim.x) {
            const uint32_t id = __ldg(indices + q);
            if (q > 0 && __ldg(indices + q - 1) == id)
                continue;
            for (uint32_t row = q; row < batch_size && __ldg(indices + row) == id; row += GroupQ) {
                const uint32_t count = paged_mqa_group_size<GroupQ>(indices, row, batch_size);
                prefix_sum[row] = math::ceil_div(paged_mqa_group_context(context_lens, row, count), SPLIT_KV);
                if (count < GroupQ)
                    break;
            }
        }
    } else {
        for (uint32_t q = tid; q < batch_size; q += blockDim.x)
            prefix_sum[q] = math::ceil_div(__ldg(context_lens + q * next_n + next_n - 1), SPLIT_KV);
    }
    __syncthreads();

    // Grouped launches use whole warps. Scan within each warp, then
    // carry its total to the following warps and chunks.
    uint32_t sum = 0;
    if constexpr (GroupQ != 0) {
        const uint32_t lane = tid % 32, warp = tid / 32;
        const uint32_t warps = blockDim.x / 32;
        const auto warp_prefix_sum = [lane](uint32_t value) {
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                const uint32_t previous = __shfl_up_sync(0xffffffffu, value, offset);
                if (lane >= offset)
                    value += previous;
            }
            return value;
        };
        for (uint32_t base = 0; base < batch_size; base += blockDim.x) {
            const uint32_t row = base + tid;
            const uint32_t val = warp_prefix_sum(row < batch_size ? prefix_sum[row] : 0);
            // Reuse the first entries of this chunk for warp totals after
            // every lane has loaded its input. Only live warps need a slot.
            const uint32_t live_warps = min(warps, math::ceil_div(batch_size - base, 32u));
            __syncthreads();
            if (lane == 31 && warp < live_warps)
                prefix_sum[base + warp] = val;
            __syncthreads();
            if (warp == 0) {
                const uint32_t total = warp_prefix_sum(lane < live_warps ? prefix_sum[base + lane] : 0);
                if (lane < live_warps)
                    prefix_sum[base + lane] = total;
            }
            __syncthreads();
            const uint32_t before = warp && warp < live_warps ? prefix_sum[base + warp - 1] : 0;
            const uint32_t chunk = prefix_sum[base + live_warps - 1];
            __syncthreads();
            if (row < batch_size)
                prefix_sum[row] = sum + before + val;
            sum += chunk;
        }
        __syncthreads();
    } else {
        // calculate prefix sum
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
    }

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


template <uint32_t BLOCK_KV, uint32_t kNumMathWarpGroups, uint32_t kNextN,
          bool kSkipEmpty = false, uint32_t GroupQ = 0>
struct PagedMQALogitsScheduler {
    uint32_t batch_size;
    const uint32_t* context_lens;
    const uint32_t* indices;

    uint32_t current_q_idx, current_kv_idx;
    uint32_t end_q_idx, end_kv_idx;
    uint32_t current_num_kv;

    __device__ __forceinline__ explicit PagedMQALogitsScheduler(const uint32_t& batch_size, const uint32_t& sm_idx,
                                                                const uint32_t* context_lens, const uint32_t* schedule_meta,
                                                                const uint32_t* indices = nullptr)
        : batch_size(batch_size), context_lens(context_lens), indices(indices) {

        const auto& current_pack = __ldg(reinterpret_cast<const uint2*>(schedule_meta) + sm_idx);
        const auto& end_pack = __ldg(reinterpret_cast<const uint2*>(schedule_meta) + sm_idx + 1);
        current_q_idx = current_pack.x, current_kv_idx = current_pack.y * kNumMathWarpGroups;
        end_q_idx = end_pack.x, end_kv_idx = end_pack.y * kNumMathWarpGroups;

        current_num_kv = current_q_idx < batch_size ? math::ceil_div(
            context_len(current_q_idx), BLOCK_KV) : 0;
    }

    __device__ __forceinline__ bool fetch_next_task(uint32_t &q_idx, uint32_t &kv_idx, uint32_t &num_kv) {
        q_idx = current_q_idx;
        kv_idx = current_kv_idx;
        num_kv = current_num_kv;

        if (is_last_task(q_idx, kv_idx))
            return false;

        current_kv_idx += kNumMathWarpGroups;
        const bool advance_q = current_kv_idx >= current_num_kv;
        if (GroupQ != 0 ? __builtin_expect(advance_q, false) : advance_q) {
            current_q_idx = next_q_idx(current_q_idx);
            current_kv_idx = 0;
            current_num_kv = current_q_idx < batch_size ? math::ceil_div(
                context_len(current_q_idx), BLOCK_KV) : 0;
        }

        return true;
    }

    __device__ __forceinline__ uint32_t group_size(uint32_t row) const {
        if constexpr (GroupQ != 0)
            return paged_mqa_group_size<GroupQ>(indices, row, batch_size);
        return 1;
    }

    __device__ __forceinline__ uint32_t context_len(uint32_t row) const {
        if constexpr (GroupQ != 0)
            return paged_mqa_group_context(context_lens, row, group_size(row));
        const uint32_t idx = row * kNextN + kNextN - 1;
        return __ldg(context_lens + idx);
    }

    __device__ __forceinline__ uint32_t next_q_idx(uint32_t q_idx) const {
        q_idx += group_size(q_idx);
        if constexpr (GroupQ != 0) {
            while (q_idx < end_q_idx) {
                const uint32_t count = group_size(q_idx);
                if (paged_mqa_group_context(context_lens, q_idx, count) != 0)
                    break;
                q_idx += count;
            }
        } else if constexpr (kSkipEmpty) {
            // H8 metadata assigns no KV tasks to empty requests. Keep the
            // scheduler and Q lookahead on the same next nonempty request.
            if (__builtin_expect(q_idx < end_q_idx &&
                    context_len(q_idx) == 0, false)) {
                // All lanes enter together. Scan long empty runs a warp at a
                // time; the usual adjacent nonempty request stays scalar.
                const uint32_t lane = threadIdx.x % 32;
                while (q_idx < end_q_idx) {
                    const uint32_t candidate = q_idx + lane;
                    const bool live = candidate < end_q_idx &&
                        context_len(candidate) != 0;
                    const uint32_t mask = __ballot_sync(0xffffffffu, live);
                    if (mask != 0)
                        return q_idx + __ffs(mask) - 1;
                    q_idx = min(q_idx + 32, end_q_idx);
                }
            }
        }
        return q_idx;
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

// Only CTA boundaries are returned. Scan request cost from the end so a
// single CTA can find the next request head across tiles with bounded storage.
template <uint32_t BlockQ, uint32_t SplitKV, uint32_t NumWorkers, uint32_t SingletonCostMultiplier, uint32_t SharedRows>
__device__ __forceinline__ void fp4_varlen_metadata(uint32_t nq, const uint32_t *lengths, const uint32_t *indices,
                                                    FP4VarlenCTAStart *metadata, uint32_t *units,
                                                    uint64_t *suffix) {
    static_assert(BlockQ == 2 || BlockQ == 4);
    constexpr uint32_t Threads = 256, Warps = Threads / 32;
    const uint32_t tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    __shared__ uint64_t warp_cost[Warps], carried_cost;
    __shared__ uint32_t warp_first[Warps], next_run;
    if constexpr (SharedRows > 0) {
        __shared__ uint32_t shared_units[SharedRows];
        __shared__ uint64_t shared_suffix[SharedRows];
        units = shared_units;
        suffix = shared_suffix;
    }
    if (!tid) {
        carried_cost = 0;
        next_run = nq;
    }
    __syncthreads();
    for (uint32_t tile = math::ceil_div(nq, Threads); tile > 0; --tile) {
        const uint32_t q = (tile - 1) * Threads + tid;
        const uint32_t id = q < nq ? indices[q] : 0;
        const uint32_t previous = __shfl_up_sync(0xffffffffu, id, 1);
        const bool head = q < nq && (q == 0 || id != (lane ? previous : indices[q - 1]));
        const uint32_t heads = __ballot_sync(0xffffffffu, head);
        if (!lane)
            warp_first[warp] = heads ? q + __ffs(heads) - 1 : nq;
        __syncthreads();
        uint64_t cost = 0;
        if (head) {
            uint32_t end = nq;
            const uint32_t following = heads & (0xfffffffeu << lane);
            if (following) {
                end = (q & ~31u) + __ffs(following) - 1;
            } else {
                for (uint32_t w = warp + 1; w < Warps && end == nq; ++w)
                    end = warp_first[w];
                if (end == nq)
                    end = next_run;
            }
            // The largest Q block has at least three queries for H32
            // runs of length >= 3, and two queries for H64 runs of length >= 2.
            const uint32_t count = end - q;
            const uint32_t shift = count == 1 ? (BlockQ == 4 ? 2u : 1u)
                : (BlockQ == 4 && count == 2 ? 1u : 0u);
            const uint32_t pack = 1u << shift;
            const uint32_t unit = (count << shift) * (shift == 2 ? SingletonCostMultiplier : 1);
            const uint32_t splits = math::ceil_div(lengths[end - 1], SplitKV);
            cost = uint64_t((splits + pack - 1) >> shift) * unit;
            units[q] = unit;
        }
        const uint64_t own_cost = cost;
#pragma unroll
        for (uint32_t offset = 1; offset < 32; offset <<= 1) {
            const uint32_t lo = __shfl_down_sync(0xffffffffu, uint32_t(cost), offset);
            const uint32_t hi = __shfl_down_sync(0xffffffffu, uint32_t(cost >> 32), offset);
            if (lane + offset < 32)
                cost += (uint64_t(hi) << 32) | lo;
        }
        if (!lane)
            warp_cost[warp] = cost;
        __syncthreads();
        if (!tid) {
            uint64_t c = carried_cost;
            uint32_t next = next_run;
#pragma unroll
            for (uint32_t w = Warps; w > 0; --w) {
                const uint64_t new_cost = c + warp_cost[w - 1];
                warp_cost[w - 1] = c;
                c = new_cost;
                next = min(next, warp_first[w - 1]);
            }
            carried_cost = c;
            next_run = next;
        }
        __syncthreads();
        if (q < nq)
            suffix[q] = warp_cost[warp] + cost - own_cost;
        __syncthreads();
    }
    const uint64_t total = carried_cost;
    // The exclusive suffix at q equals total minus the original prefix
    // at q. Equal suffix values skip rows with no work.
    auto first_below = [&](uint32_t lo, uint64_t remaining) {
        uint32_t hi = nq;
        while (lo < hi) {
            const uint32_t mid = lo + (hi - lo) / 2;
            if (suffix[mid] >= remaining)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    };
    for (uint32_t worker = tid; worker <= NumWorkers; worker += Threads) {
        // Quotient and remainder avoid overflow in total * worker.
        const uint64_t target = (total / NumWorkers) * worker + ((total % NumWorkers) * worker) / NumWorkers;
        const uint64_t remaining = total - target;
        uint32_t q = first_below(0, remaining), split = 0;
        if (q < nq) {
            // The exclusive suffix drops only at nonempty request heads.
            const uint32_t unit = units[q];
            const uint64_t before = q ? suffix[q - 1] : total;
            const uint64_t after = suffix[q];
            const uint64_t from_start = before - remaining, to_end = remaining - after;
            const uint64_t margin = min(uint64_t(unit), total / NumWorkers / 2);
            if (min(from_start, to_end) <= margin) {
                if (to_end < from_start)
                    q = first_below(q + 1, after);
            } else {
                split = uint32_t(from_start / unit);
            }
        }
        metadata[worker] = FP4VarlenCTAStart{q, split};
    }
    __syncthreads();
}

// Owns worker permutation, request ranges and two-Q-block reuse windows.
template <uint32_t BlockQ, uint32_t SplitKV, uint32_t BlockKV, uint32_t WorkersPerCU>
struct FP4GroupedPagedScheduler {
    const FP4VarlenCTAStart* starts;
    const uint32_t* indices;
    const uint32_t* lengths;
    uint32_t nq;
    template <typename RequestFn, typename SingleFn>
    __device__ __forceinline__ void for_each_task(RequestFn&& on_request, SingleFn&& on_single) const {
        const uint32_t num_cus = gridDim.x / WorkersPerCU;
        const uint32_t slot = blockIdx.x / num_cus;
        // Offset each scheduling layer to spread sparse nonempty ranges
        // across CU positions while preserving the original cost mixing.
        const uint32_t worker = slot * num_cus + (blockIdx.x % num_cus + slot) % num_cus;
        const FP4VarlenCTAStart first = starts[worker], last = starts[worker + 1];
        constexpr uint32_t MathGroups = SplitKV / BlockKV;
        for (uint32_t q = first.source_q; q <= last.source_q && q < nq;) {
            if (q == last.source_q && last.kv_split == 0) break;
            const uint32_t id = __ldg(indices + q);
            uint32_t run_end = q + 1;
            while (run_end < nq && __ldg(indices + run_end) == id) ++run_end;
            const uint32_t count = run_end - q;
            const uint32_t blocks = math::ceil_div(count, BlockQ);
            const uint32_t base = count / blocks, remainder = count % blocks;
            const uint32_t valid = base + (remainder != 0);
            const uint32_t context = __ldg(lengths + run_end - 1);
            // Q spans are powers of two. Keep the quotient explicit so local
            // task dispatch cannot turn it into a runtime integer division.
            const uint32_t pack = BlockQ >> fp4_varlen_q_shift<BlockQ>(valid);
            const uint32_t begin = q == first.source_q ? first.kv_split * MathGroups * pack : 0;
            const uint32_t end = q == last.source_q ? last.kv_split * MathGroups * pack
                : deep_gemm::math::ceil_div(context, BlockKV);
            if (begin < end) {
                if (blocks > 1 || (BlockQ == 4 && valid == 3)) {
                    uint32_t source = q;
                    for (uint32_t block = 0; block < blocks; block += 2) {
                        const uint32_t count0 = base + (block < remainder);
                        const uint32_t count1 = block + 1 < blocks ? base + (block + 1 < remainder) : 0;
                        on_request(FP4VarlenQBlock{source, count0, context, q},
                                   FP4VarlenQBlock{source + count0, count1, context, q}, begin, end);
                        source += count0 + count1;
                    }
                } else {
                    on_single(FP4VarlenQBlock{q, valid, context, q}, begin, end);
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
