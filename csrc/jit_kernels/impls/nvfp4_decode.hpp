#pragma once
#include <tuple>
#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/math.hpp"

namespace deep_gemm::nvfp4::detail {

class Fp4DecodeRuntime final : public LaunchRuntime<Fp4DecodeRuntime> {
public:
    struct Args {
        const void* packed;
        const void* scales;
        void* output;
        uint32_t count;
        LaunchArgs launch_args;
    };

    static std::tuple<> compile_key(const Args&) { return {}; }

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/nvfp4_decode.cuh>
extern "C" __global__ __launch_bounds__(128)
void dequantize_nvfp4_fp8(const uint2* packed, const uint8_t* scales, uint4* output, uint32_t count) {{
    deep_gemm::nvfp4::detail::decode_fp4_fp8<>(packed, scales, output, count);
}}
)");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.packed, args.scales, args.output, args.count));
    }
};

static void decode_fp4_fp8(const torch::Tensor& packed, const torch::Tensor& scales,
                          const torch::Tensor& output) {
    const uint32_t count = output.numel() / 16;
    Fp4DecodeRuntime::Args args{packed.data_ptr(), scales.data_ptr(), output.data_ptr(), count,
        {dim3(ceil_div(count, uint32_t(16 * 128))), dim3(128), 0}};
    const auto runtime = compiler->build_cached<Fp4DecodeRuntime>("dequantize_nvfp4_fp8", args);
    Fp4DecodeRuntime::launch(runtime, args);
}

} // namespace deep_gemm::nvfp4::detail
