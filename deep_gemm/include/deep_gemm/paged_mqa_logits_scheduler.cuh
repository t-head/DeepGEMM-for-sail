#pragma once
#include "utils.cuh"


namespace deep_gemm {

// NOTES: the body lives in a `__device__` function so that both JIT flavours can share it: the
// Python JIT launches the `__global__` below via `launch_paged_mqa_logits_metadata`, while the C++
// JIT emits its own `extern "C" __global__` entry that forwards here.
template <uint32_t SPLIT_KV, uint32_t kNumSMs>
__device__ __forceinline__
void smxx_paged_mqa_logits_metadata_device(const uint32_t batch_size, const uint32_t next_n,
                                          const uint32_t* context_lens, uint32_t* schedule_metadata) {
    extern __shared__ uint32_t prefix_sum[];
    const uint32_t tid = threadIdx.x;

    // load context lens (2D indexing: take last token's context_len per request)
    for (uint32_t k = tid; k < batch_size; k += blockDim.x) {
        prefix_sum[k] = ceil_div(__ldg(context_lens + k * next_n + next_n - 1), SPLIT_KV);
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
            __syncthreads();
            temp_prefix_sum[tid] = val;
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
        __syncthreads();
        if (tid < last) {
            temp_prefix_sum[tid] = val;
        }
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

// Kernel entry used by the Python JIT (`launch_paged_mqa_logits_metadata`)
template <uint32_t SPLIT_KV, uint32_t kNumSMs>
__global__
void smxx_paged_mqa_logits_metadata(const uint32_t batch_size, const uint32_t next_n,
                                     const uint32_t* context_lens, uint32_t* schedule_metadata) {
    smxx_paged_mqa_logits_metadata_device<SPLIT_KV, kNumSMs>(batch_size, next_n, context_lens, schedule_metadata);
}

template <uint32_t SPLIT_KV, uint32_t kNumSMs>
void launch_paged_mqa_logits_metadata(const uint32_t batch_size, const uint32_t next_n,
                                      const uint32_t* context_lens, uint32_t* schedule_metadata,
                                      hggcStream_t stream) {
    int grid = 1;
    int block = min(batch_size, 1024);
    int smem_size = batch_size * 4;
    smxx_paged_mqa_logits_metadata<SPLIT_KV, kNumSMs><<<grid, block, smem_size, stream>>>(
        batch_size, next_n, context_lens, schedule_metadata);

};

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
        current_num_kv = current_q_idx < batch_size ? ceil_div(__ldg(this->context_lens + idx), BLOCK_KV) : 0;
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
            current_num_kv = current_q_idx < batch_size ? ceil_div(__ldg(this->context_lens + idx), BLOCK_KV) : 0;
        }

        return true;
    }

    __device__ __forceinline__ bool exist_q_idx(const uint32_t& q_idx) const {
        return q_idx < end_q_idx or q_idx == end_q_idx and 0 < end_kv_idx;
    }

    __device__ __forceinline__ bool is_last_task(const uint32_t& q_idx, const uint32_t& kv_idx) const {
        return (q_idx > end_q_idx) or (q_idx == end_q_idx and kv_idx == end_kv_idx);
    }
};

}// namespace deep_gemm
