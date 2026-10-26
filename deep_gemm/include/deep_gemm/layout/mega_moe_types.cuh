#pragma once

namespace deep_gemm::mega_moe {

enum class Fp8ScaleMode { Fixed, SideStatic, SideDynamic };

inline constexpr int kFp4Gemm1SfbBlockN = 256;
inline constexpr int kFp4ScatterSmemPadPairs = 16;

} // namespace deep_gemm::mega_moe
