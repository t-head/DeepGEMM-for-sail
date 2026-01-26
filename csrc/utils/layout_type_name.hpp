// layout_type_name.h
#pragma once
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"

template <typename T>
constexpr const char* get_layout_type_name() {
    return "UNKNOWN_LAYOUT"; // fallback
}

#define REGISTER_LAYOUT(LAYOUT_TYPE) \
    template<> constexpr const char* get_layout_type_name<LAYOUT_TYPE>() { \
        return #LAYOUT_TYPE; \
    }

REGISTER_LAYOUT(cutlass::layout::RowMajor)
REGISTER_LAYOUT(cutlass::layout::ColumnMajor)
REGISTER_LAYOUT(cutlass::layout::PitchLinear)
// REGISTER_LAYOUT(cutlass::layout::TensorNHWC)
// REGISTER_LAYOUT(cutlass::layout::TensorNDHWC)

#undef REGISTER_LAYOUT