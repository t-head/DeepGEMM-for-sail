#pragma once

namespace deep_gemm {
namespace dual_b {

constexpr int shared_storage_bytes(int bm, int bn, int bk, int wm, int wn, int wk,
                                   int stages, bool packed_n, int partitions,
                                   bool cta_k, int element_bytes = 2) {
    const int operands = partitions * stages * (bm + (packed_n ? 1 : 2) * bn) * bk * element_bytes;
    const int threads = (bm / wm) * (bn / wn) * (bk / wk) * partitions * 32;
    const int partials = cta_k ? threads * (wm * wn / 32) * 2 * 4 : 0;
    return ((operands > partials ? operands : partials) + 127) & ~127;
}

} // namespace dual_b
} // namespace deep_gemm
