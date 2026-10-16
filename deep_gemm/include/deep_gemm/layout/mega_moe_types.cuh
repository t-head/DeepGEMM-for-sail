#pragma once

namespace deep_gemm::mega_moe {

// GEMM1 output contracts include activation, element type and logical width.
enum class Gemm1Epilogue { IdentityBf16, SwiGluFp8 };
enum class Fp8ScaleMode { Fixed, SideStatic, SideDynamic };

} // namespace deep_gemm::mega_moe
