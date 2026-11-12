#pragma once

#include <cassert>
#include <torch/python.h>

namespace deep_gemm {

torch::Tensor get_mn_major_tma_aligned_tensor(const torch::Tensor& x) {
    assert(x.dim() == 2 || x.dim() == 3);

    bool remove_dim = false;
    int64_t m = x.size(-2);
    int64_t n = x.size(-1);
    auto dtype = x.dtype();
    auto device = x.device();

    int64_t element_size = x.element_size();
    // int64_t aligned_m = get_tma_aligned_size(m, element_size);
    int64_t aligned_m = m;
    torch::Tensor x_view = x;

    if (x.dim() == 2) {
        if (x.stride(0) == 1 && x.stride(1) == aligned_m) {
            return x;
        }
        x_view = x.unsqueeze(0);
        remove_dim = true;
    }

    int64_t b = x_view.size(0);

    if (x_view.stride(0) == aligned_m * n && x_view.stride(1) == 1 && x_view.stride(2) == aligned_m) {
        return remove_dim ? x_view.squeeze(0) : x_view;
    }

    auto options = torch::TensorOptions().dtype(dtype).device(device);
    torch::Tensor aligned_x = torch::transpose(torch::empty({b, n, aligned_m}, options), 1, 2);

    aligned_x.slice(1, 0, m).copy_(x_view);

    aligned_x = aligned_x.slice(1, 0, m);

    return remove_dim ? aligned_x.squeeze(0) : aligned_x;
}

} // namespace deep_gemm
