#pragma once

#include <optional>
#include <cstring>
#include <cstdlib>

namespace deep_gemm_bf16_common {

// Measured BF16 NT configurations for the 39-CU PPU1.5. Keep this separate
// from the shared adaptive selector: INT8 and grouped GEMMs use that selector
// too. Explicit public configs bypass this table in bf16_gemm().
struct MemoryTuningConfig {
    int bm, bn, bk, wm, wn, wk, stages;
    bool nonpersistent;
    int b_cache_policy;
};

// Restrict the measured table to the validated SKU and full scheduling budget.
// A caller may reduce num_sms; that budget must not identify physical hardware.
inline bool is_memory_tuning_device(const char* name, int major, int minor,
                                    int physical_sms, int num_sms) {
    return major == 8 && minor == 9 && physical_sms == 39 && num_sms == 39 &&
           std::strcmp(name, "ZW-M890P") == 0;
}

inline bool memory_tuning_enabled() {
    const char* enabled = std::getenv("DG_BF16_MEMORY_TUNING");
    return !enabled || std::strcmp(enabled, "0") != 0;
}

// Pure selector: capability, environment and public-config gates belong to the
// caller. Exact matches keep neighboring workloads on the existing selector.
inline std::optional<MemoryTuningConfig> get_memory_tuning(int m, int n, int k) {

    // Stream each B tile once per CTA, preserving the original K traversal order.
    if (n == 49152 && k == 7168) {
        if (m == 1 || m == 2 || m == 4 || m == 8 || m == 16)
            return MemoryTuningConfig{16, 640, 64, 16, 32, 64, 3, true, 3};
        if (m == 32)
            return MemoryTuningConfig{32, 640, 64, 32, 32, 64, 3, true, 3};
        if (m == 64)
            return MemoryTuningConfig{64, 640, 64, 64, 32, 64, 2, true, 3};
    }

    // Smaller BK with a deeper pipeline avoids the coarse BK256 WarpOnK
    // reduction on these underfilled memory-bound problems.
    if (m == 32 && n == 7168 && k == 12288)
        return MemoryTuningConfig{32, 192, 64, 16, 32, 64, 6, true, 1};
    if (m == 32 && n == 7168 && k == 8192)
        return MemoryTuningConfig{32, 192, 64, 32, 16, 64, 6, false, 1};
    if (m == 64 && n == 7168 && k == 8192)
        return MemoryTuningConfig{64, 192, 64, 32, 32, 64, 7, true, 1};
    if (m == 64 && n == 12288 && k == 1536)
        return MemoryTuningConfig{64, 320, 64, 64, 32, 64, 4, true, 1};
    return std::nullopt;
}

} // namespace deep_gemm_bf16_common
