#pragma once
#include <string>

enum acArch_t {
    AC_PPU0010,
    AC_PPU0015,
    AC_PPU_MAX_ARCH,
};

static std::string acArchNames[]{
    "PPU0010",
    "PPU0015",
    "PPU_UNKNOWN",
};
