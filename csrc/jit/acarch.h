#pragma once
#include <string>

enum acArch_t {
    AC_PPU0010,
    AC_PPU0015,
    AC_PPU0017,
    AC_PPU0020,
    AC_PPU_MAX_ARCH,
};

static std::string acArchNames[]{
    "PPU0010",
    "PPU0015",
    "PPU0017",
    "PPU0020",
    "PPU_UNKNOWN",
};
