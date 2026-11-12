#pragma once

#include <memory>
#include <string>

namespace deep_gemm {

// Upstream-compatible interface type. Custom epilogues are currently rejected
// by the PPU K-grouped host API; no device implementation is connected yet.
class EpilogueClass {
public:
    virtual ~EpilogueClass() = default;

    virtual std::string get_epilogue_operator_type() const = 0;
};

} // namespace deep_gemm
