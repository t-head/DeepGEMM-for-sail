#pragma once

#include <torch/python.h>

#include "math.hpp"
#include "exception.hpp"
#include "../jit/device_runtime.hpp"

namespace deep_gemm {

// Major-ness stuffs
// Hand-written replacement for cute::UMMA::Major: the PPU actlize port does not
// provide cute::UMMA, so define the minimal major-ness enum used by the APIs here.
enum class MajorType { K = 0, MN = 1 };

static void major_check(const torch::Tensor& t) {
    const auto dim = t.dim();
    DG_HOST_ASSERT(dim == 2 or dim == 3);
    if (dim == 3)
        DG_HOST_ASSERT(t.stride(0) == t.size(-2) * t.size(-1));
    DG_HOST_ASSERT(t.stride(-2) == 1 or t.stride(-1) == 1);
}

static MajorType get_major_type_ab(const torch::Tensor& t) {
    major_check(t);
    return t.stride(-1) == 1 ? MajorType::K : MajorType::MN;
}

static void check_major_type_cd(const torch::Tensor& t) {
    // NOTES: the library only supports row-major output layouts
    major_check(t);
    DG_HOST_ASSERT(t.stride(-1) == 1);
}

static bool fp8_requires_k_major() {
    return device_runtime->get_arch_major() == 9;
}

// Tensor utils
template <int N>
static auto get_shape(const torch::Tensor& t) {
    return [&t] <size_t... Is> (std::index_sequence<Is...>) {
        return std::make_tuple(static_cast<int>(t.sizes()[Is])...);
    }(std::make_index_sequence<N>());
}

// Recipe
static std::tuple<int, int, int>
get_default_recipe(const torch::ScalarType& sfa_dtype, const torch::ScalarType& sfb_dtype) {
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        DG_HOST_ASSERT(sfa_dtype == torch::kFloat and sfb_dtype == torch::kFloat);
        return {1, 128, 128};
    } else if (arch_major == 10) {
        DG_HOST_ASSERT(sfb_dtype == torch::kFloat or sfb_dtype == torch::kInt);
        return sfb_dtype == torch::kFloat ?
            std::make_tuple(1, 128, 128):   // Legacy format
            std::make_tuple(1,   1, 128);   // 1D1D kernels
    }
    DG_HOST_UNREACHABLE("Unknown recipe");
}

// SF layouts
static torch::Tensor check_sf_layout(const torch::Tensor& sf,
                                     const int& mn, const int& k,
                                     const int& gran_mn, const int& gran_k,
                                     const std::optional<int>& num_groups,
                                     const bool& tma_stride_check = false,
                                     const bool& sfb_check = false,
                                     const std::optional<torch::ScalarType>& type_check = std::nullopt) {
    // Type check
    if (type_check.has_value())
        DG_HOST_ASSERT(sf.scalar_type() == type_check.value());

    // Always do shape checks
    const auto sf_dtype = sf.scalar_type();
    DG_HOST_ASSERT(sf_dtype == torch::kFloat or sf_dtype == torch::kInt);
    DG_HOST_ASSERT(sf.dim() == static_cast<int>(num_groups.has_value()) + 2);
    if (num_groups.has_value())
        DG_HOST_ASSERT(sf.size(-3) == num_groups.value());
    DG_HOST_ASSERT(sf.size(-2) == ceil_div(mn, gran_mn));
    DG_HOST_ASSERT(sf.size(-1) == ceil_div(k, gran_k * (sf_dtype == torch::kFloat ? 1 : 4)));

    // TMA stride checks: TMA aligned and MN-major
    if (tma_stride_check) {
        if (num_groups.has_value())
            DG_HOST_ASSERT(sf.stride(-3) == sf.stride(-1) * sf.size(-1));
        // Check contiguity in the MN direction
        DG_HOST_ASSERT(sf.stride(-2) == 1 or mn == 1);
        DG_HOST_ASSERT(sf.stride(-1) == get_tma_aligned_size(mn, sf.element_size()));
    }

    if (sfb_check) {
        if (num_groups.has_value())
            DG_HOST_ASSERT(sf.stride(-3) == sf.size(-2) * sf.size(-1));
        DG_HOST_ASSERT((sf.stride(-1) == 1 and sf.stride(-2) == sf.size(-1)) or
                       (sf.stride(-1) == sf.size(-2) and sf.stride(-2) == 1));
    }
    return sf;
}

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

torch::Tensor get_col_major_tensor(const torch::Tensor& x) {
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "Only 2-D or 3-D tensors supported");

    bool squeeze_dim = false;
    torch::Tensor x_view = x;
    if (x.dim() == 2) {
        x_view = x.unsqueeze(0);
        squeeze_dim = true;
    }

    const int64_t b = x_view.size(0);
    const int64_t m = x_view.size(1);
    const int64_t n = x_view.size(2);

    // Allocate (B, N, M) then transpose the last two dims, so the result is column-major over (M, N).
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor col_major = torch::empty({b, n, m}, options).transpose(-2, -1);
    col_major.copy_(x_view);

    return squeeze_dim ? col_major.squeeze(0) : col_major;
}

int get_mk_alignment_for_contiguous_layout() {
    /*
    When we do a grouped GEMM in contiguous format, LHS are grouped into several batches along the M axis.
    Since we deal with exactly one sub-matrix of RHS for each GEMM block, batch sizes above should align well
        with GEMM block shape.
    Returns:
        Group-level alignment requirement for grouped contiguous layout, which is always 128.
    */
    return 128;
}

} // namespace deep_gemm
