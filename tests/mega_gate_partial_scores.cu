#include <deep_gemm/epilogue/ppu_mega_gate.cuh>
#include <cstdio>

using deep_gemm::epilogue::mega_gate::store_checked_scores;
using deep_gemm::mega_gate_layout::KernelArgs;

template <int Splits>
__device__ bool accepts(bool fixed, bool random, float lo, float hi) {
    KernelArgs args{};
    args.num_tokens = 1;
    args.routing.num_routed_experts = 2;
    args.fix_routing_mask = &fixed;
    args.force_random = &random;
    alignas(8) float scratch[2];
    bool valid = true;
    store_checked_scores<true, Splits>(args, 0, 0, scratch, lo, hi, false, valid);
    return valid;
}

__global__ void check(int* result) {
    // One thread per control: learned, fixed or random routing.
    const int route = threadIdx.x;
    const bool fixed = route == 1, random = route == 2;
    const bool valid = accepts<1>(fixed, random, INFINITY, -INFINITY) &&
                       !accepts<1>(fixed, random, NAN, 1.0f) &&
                       accepts<2>(fixed, random, 1.0f, -2.0f) &&
                       (accepts<2>(fixed, random, INFINITY, 0.0f) == (route == 0)) &&
                       (accepts<2>(fixed, random, 0.0f, -INFINITY) == (route == 0)) &&
                       (accepts<2>(fixed, random, NAN, 0.0f) == (route == 0));
    result[route] = valid;
}

int main() {
    int* device;
    int results[3]{};
    if (hggcMalloc(reinterpret_cast<void**>(&device), sizeof(results)) != hggcSuccess)
        return 1;
    check<<<1, 3>>>(device);
    const auto status = hggcMemcpy(results, device, sizeof(results), hggcMemcpyDeviceToHost);
    hggcFree(device);
    if (status != hggcSuccess)
        return 1;
    for (int route = 0; route < 3; ++route) {
        if (!results[route]) {
            std::fprintf(stderr, "Unexpected sigmoid partial-score validity for route %d\n", route);
            return 1;
        }
    }
    return 0;
}
