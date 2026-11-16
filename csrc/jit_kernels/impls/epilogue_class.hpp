#pragma once

#include <memory>
#include <optional>
#include <string>
#include <torch/python.h>
#include <deep_gemm/common/types.cuh>
#include "../../utils/exception.hpp"

namespace deep_gemm {

// Selects the device operator and constructs its runtime state.
class EpilogueClass {
public:
    virtual ~EpilogueClass() = default;

    virtual std::string get_epilogue_operator_type() const = 0;
    virtual bool is_per_token() const { return false; }
    virtual EpilogueOperatorArgs make_epilogue_operator_args(const int& m, const int& n) const { return {}; }
    virtual std::optional<torch::Tensor> output_sfd() const { return std::nullopt; }
    virtual void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const {
        DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    }
};

class FP8CastWithClampEpilogue final: public EpilogueClass {
    torch::Tensor sfd;
    std::optional<float> clamp_value;
    std::optional<torch::Tensor> clamped_count;
public:
    explicit FP8CastWithClampEpilogue(const torch::Tensor& sfd,
                                     const std::optional<float>& clamp_value = std::nullopt,
                                     const std::optional<torch::Tensor>& clamped_count = std::nullopt):
        sfd(sfd), clamp_value(clamp_value), clamped_count(clamped_count) {
        DG_HOST_ASSERT(not clamp_value.has_value() or clamp_value.value() > 0.0f);
        if (clamped_count.has_value())
            DG_HOST_ASSERT(clamped_count->is_cuda() and clamped_count->scalar_type() == torch::kUInt32 and
                           clamped_count->numel() == 1);
    }

    std::string get_epilogue_operator_type() const override {
        // Specialize optional work away from the default PPU output path.
        return std::string("epilogue::operators::CastToFP8WithClamp<") +
               (clamp_value.has_value() ? "true" : "false") + ", " +
               (clamp_value.has_value() and clamped_count.has_value() ? "true>" : "false>");
    }

    EpilogueOperatorArgs make_epilogue_operator_args(const int& m, const int& n) const override {
        return {.sfd = static_cast<uint32_t*>(sfd.data_ptr()),
                .sfd_stride = sfd.stride(-1),
                .shape_m = static_cast<uint32_t>(m), .shape_n = static_cast<uint32_t>(n),
                .clamp_value = clamp_value.value_or(0.0f),
                .clamped_count = clamped_count.has_value() ? static_cast<uint32_t*>(clamped_count->data_ptr()) : nullptr};
    }

    std::optional<torch::Tensor> output_sfd() const override { return sfd; }
    void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const override {
        DG_HOST_ASSERT(not c.has_value() and d.scalar_type() == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(sfd.device() == d.device());
        if (clamped_count.has_value())
            DG_HOST_ASSERT(clamped_count->device() == d.device());
    }
};

// PPU extension: one FP32 scale for the flattened output row across all groups.
class FP8PerTokenCastEpilogue final: public EpilogueClass {
    torch::Tensor sfd;
public:
    explicit FP8PerTokenCastEpilogue(const torch::Tensor& sfd): sfd(sfd) {}

    bool is_per_token() const override { return true; }
    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::CastToFP8PerToken";
    }
    EpilogueOperatorArgs make_epilogue_operator_args(const int&, const int&) const override {
        return {.row_scales = sfd.data_ptr<float>()};
    }
    std::optional<torch::Tensor> output_sfd() const override { return sfd; }
    void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const override {
        DG_HOST_ASSERT(not c.has_value() and d.scalar_type() == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(sfd.device() == d.device() and sfd.scalar_type() == torch::kFloat32);
        DG_HOST_ASSERT(d.transpose(0, 1).is_contiguous() and sfd.is_contiguous());
        DG_HOST_ASSERT(sfd.sizes() == std::vector<int64_t>({d.size(1), 1}));
    }
};

} // namespace deep_gemm
