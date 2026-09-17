#pragma once

#include <acblasLt.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>

#include "../../jit/device_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/compatibility.hpp"

namespace deep_gemm {

static auto get_acblaslt_layout(const hggcDataType& type, const int& rows, const int& cols, const int& ld,
                                const std::optional<int>& batch_count = std::nullopt,
                                const std::optional<int>& batch_offset = std::nullopt) {
    acblasLtMatrixLayout_t layout;
    DG_ACBLASLT_CHECK(acblasLtMatrixLayoutCreate(&layout, type, rows, cols, ld));
    if (batch_count.has_value()) {
        DG_HOST_ASSERT(batch_offset.has_value());

        const int64_t batch_offset_int64 = batch_offset.value();
        DG_ACBLASLT_CHECK(acblasLtMatrixLayoutSetAttribute(layout, ACBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count.value(), sizeof(batch_count.value())));
        DG_ACBLASLT_CHECK(acblasLtMatrixLayoutSetAttribute(layout, ACBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_offset_int64, sizeof(batch_offset_int64)));
    }
    return layout;
}

static void call_acblaslt_api(const acblasOperation_t& trans_a,
                              const acblasOperation_t& trans_b,
                              const acblasLtMatrixLayout_t& layout_a,
                              const acblasLtMatrixLayout_t& layout_b,
                              const acblasLtMatrixLayout_t& layout_d,
                              const torch::Tensor& a,
                              const torch::Tensor& b,
                              const torch::Tensor& d,
                              const bool& accumulate) {
    acblasComputeType_t compute_type = ACBLAS_COMPUTE_32F_FAST_TF32;
    hggcDataType scale_type = HGGC_R_32F;

    // Operation description
    acblasLtMatmulDesc_t desc;
    DG_ACBLASLT_CHECK(acblasLtMatmulDescCreate(&desc, compute_type, scale_type));
    DG_ACBLASLT_CHECK(acblasLtMatmulDescSetAttribute(desc, ACBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    DG_ACBLASLT_CHECK(acblasLtMatmulDescSetAttribute(desc, ACBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
    DG_ACBLASLT_CHECK(acblasLtMatmulDescSetAttribute(desc, ACBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

#if DG_ACBLASLT_ADVANCED_FEATURES_COMPATIBLE
    const int math_sms = device_runtime->get_num_sms();
    DG_ACBLASLT_CHECK(acblasLtMatmulDescSetAttribute(desc, ACBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &math_sms, sizeof(math_sms)));
#endif

#if DG_FP8_COMPATIBLE and DG_ACBLASLT_ADVANCED_FEATURES_COMPATIBLE
    bool fp8_fast_accumulate = false;
    if (a.scalar_type() == torch::kFloat8_e4m3fn)
        DG_ACBLASLT_CHECK(acblasLtMatmulDescSetAttribute(desc, ACBLASLT_MATMUL_DESC_FAST_ACCUM, &fp8_fast_accumulate, sizeof(fp8_fast_accumulate)));
#endif

    // Get acBLASLt handle, workspace, and stream
    const auto handle = device_runtime->get_acblaslt_handle();
    const auto workspace = device_runtime->get_acblaslt_workspace();
    const auto workspace_bytes = workspace.nbytes();
    const auto stream = current_stream();

    // Algorithm selection
    acblasLtMatmulPreference_t pref;
    acblasLtMatmulHeuristicResult_t heuristic;
    int num_heuristic_results = 0;
    uint32_t reduction_scheme_mask = ACBLASLT_REDUCTION_SCHEME_NONE | ACBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    DG_ACBLASLT_CHECK(acblasLtMatmulPreferenceCreate(&pref));
    DG_ACBLASLT_CHECK(acblasLtMatmulPreferenceSetAttribute(pref, ACBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                           &workspace_bytes, sizeof(workspace_bytes)));
    DG_ACBLASLT_CHECK(acblasLtMatmulPreferenceSetAttribute(pref, ACBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                                           &reduction_scheme_mask, sizeof(reduction_scheme_mask)));
    DG_ACBLASLT_CHECK(acblasLtMatmulAlgoGetHeuristic(handle, desc, layout_a, layout_b, layout_d, layout_d,
                                                     pref, 1, &heuristic, &num_heuristic_results));
    DG_HOST_ASSERT(num_heuristic_results == 1 and "Unable to find any algorithm for the GEMM");

    // Call: D = alpha * (A @ B) + beta * C
    const float alpha = 1.0, beta = accumulate ? 1.0 : 0.0;
    DG_ACBLASLT_CHECK(acblasLtMatmul(handle,                                // Light handle
                                     desc,                                  // Operation description
                                     &alpha,                                // Alpha
                                     b.data_ptr(), layout_a,                // A
                                     a.data_ptr(), layout_b,                // B
                                     &beta,                                 // Beta
                                     d.data_ptr(), layout_d,                // C
                                     d.data_ptr(), layout_d,                // D
                                     &heuristic.algo,                       // Algorithm
                                     workspace.data_ptr(), workspace_bytes, // Workspace
                                     stream));                              // Stream

    // Free memory
    DG_ACBLASLT_CHECK(acblasLtMatmulPreferenceDestroy(pref));
    DG_ACBLASLT_CHECK(acblasLtMatrixLayoutDestroy(layout_a));
    DG_ACBLASLT_CHECK(acblasLtMatrixLayoutDestroy(layout_b));
    DG_ACBLASLT_CHECK(acblasLtMatrixLayoutDestroy(layout_d));
    DG_ACBLASLT_CHECK(acblasLtMatmulDescDestroy(desc));
}

static void acblaslt_gemm(const torch::Tensor& lhs, const torch::Tensor& rhs,
                          const torch::Tensor& out,
                          const int& m, const int& n, const int& k,
                          const MajorType& a_major, const MajorType& b_major,
                          const bool& accumulate) {
    const auto trans_a = b_major == MajorType::K ? ACBLAS_OP_T : ACBLAS_OP_N;
    const auto trans_b = a_major == MajorType::K ? ACBLAS_OP_N : ACBLAS_OP_T;

    // Matrix layouts
    const auto hggc_type_a = at::cuda::ScalarTypeToCudaDataType(rhs.scalar_type());
    const auto hggc_type_b = at::cuda::ScalarTypeToCudaDataType(lhs.scalar_type());
    const auto hggc_type_d = at::cuda::ScalarTypeToCudaDataType(out.scalar_type());
    const auto layout_a = b_major == MajorType::K ? get_acblaslt_layout(hggc_type_a, k, n, rhs.stride(0))
                                                           : get_acblaslt_layout(hggc_type_a, n, k, rhs.stride(1));
    const auto layout_b = a_major == MajorType::K ? get_acblaslt_layout(hggc_type_b, k, m, lhs.stride(0))
                                                          : get_acblaslt_layout(hggc_type_b, m, k, lhs.stride(1));
    const auto layout_d = get_acblaslt_layout(hggc_type_d, n, m, out.stride(0));

    call_acblaslt_api(trans_a, trans_b, layout_a, layout_b, layout_d, lhs, rhs, out, accumulate);
}

static void acblaslt_bhr_hdr_bhd(const torch::Tensor& lhs, const torch::Tensor& rhs, const torch::Tensor& out,
                                 const int& b, const int& h, const int& r, const int& d) {
    const auto m = d, n = b, k = r;
    const auto trans_a = ACBLAS_OP_T;
    const auto trans_b = ACBLAS_OP_N;

    // Matrix layouts
    const auto layout_a = get_acblaslt_layout(HGGC_R_16BF, k, m, rhs.stride(1), h, rhs.stride(0));
    const auto layout_b = get_acblaslt_layout(HGGC_R_16BF, k, n, lhs.stride(0), h, lhs.stride(1));
    const auto layout_d = get_acblaslt_layout(HGGC_R_16BF, m, n, out.stride(0), h, out.stride(1));

    call_acblaslt_api(trans_a, trans_b, layout_a, layout_b, layout_d, lhs, rhs, out, false);
}


static void acblaslt_bhd_hdr_bhr(const torch::Tensor& lhs, const torch::Tensor& rhs, const torch::Tensor& out,
                                 const int& b, const int& h, const int& r, const int& d) {
    const auto m = r, n = b, k = d;
    const auto trans_a = ACBLAS_OP_N;
    const auto trans_b = ACBLAS_OP_N;

    // Matrix layouts
    const auto layout_a = get_acblaslt_layout(HGGC_R_16BF, m, k, rhs.stride(1), h, rhs.stride(0));
    const auto layout_b = get_acblaslt_layout(HGGC_R_16BF, k, n, lhs.stride(0), h, lhs.stride(1));
    const auto layout_d = get_acblaslt_layout(HGGC_R_16BF, m, n, out.stride(0), h, out.stride(1));

    call_acblaslt_api(trans_a, trans_b, layout_a, layout_b, layout_d, lhs, rhs, out, false);
}

} // namespace deep_gemm
