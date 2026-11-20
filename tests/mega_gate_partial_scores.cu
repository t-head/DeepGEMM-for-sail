#include <deep_gemm/epilogue/ppu_mega_gate.cuh>
#include <cstdio>

using deep_gemm::epilogue::mega_gate::store_scores;

__device__ bool preserves(float lo, float hi) {
    alignas(8) float scratch[6] = {17, 17, 17, 17, 17, 17};
    store_scores(scratch, lo, hi);
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    constexpr int hi_offset = 4;
#else
    constexpr int hi_offset = 1;
#endif
    bool valid = __float_as_uint(scratch[0]) == __float_as_uint(lo) &&
                 __float_as_uint(scratch[hi_offset]) == __float_as_uint(hi);
    for (int i = 1; i < 6; ++i)
        if (i != hi_offset) valid &= scratch[i] == 17;
    return valid;
}

__global__ void check(int* result) {
    *result = preserves(1.0f, -2.0f) && preserves(INFINITY, -INFINITY) &&
              preserves(NAN, 1.0f) && preserves(-1.0f, NAN);
}

int main() {
    int* device;
    int result = 0;
    if (hggcMalloc(reinterpret_cast<void**>(&device), sizeof(result)) != hggcSuccess)
        return 1;
    check<<<1, 1>>>(device);
    const auto status = hggcMemcpy(&result, device, sizeof(result), hggcMemcpyDeviceToHost);
    hggcFree(device);
    if (status != hggcSuccess)
        return 1;
    if (!result) {
        std::fprintf(stderr, "GEMM score storage changed values or accumulator layout\n");
        return 1;
    }
    return 0;
}
