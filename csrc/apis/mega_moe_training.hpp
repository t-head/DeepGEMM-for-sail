#pragma once

#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <deep_gemm/common/mega_moe_training.cuh>
#include "../jit/compiler.hpp"
#include "../jit/kernel_runtime.hpp"
#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/utils.hpp"
#include "../jit_kernels/heuristics/common_bf16.hpp"
#include "../jit_kernels/heuristics/common_fp8.hpp"

namespace deep_gemm::mega_moe_training {

class Runtime final : public LaunchRuntime<Runtime> {
public:
    struct Args {
        int operation;
        LaunchArgs launch_args;
        Params params;
    };

    static bool active_quantization(const int operation) {
        return operation >= int(Operation::QuantizeRowsDeviceRows) &&
               operation <= int(Operation::ActivationBackwardQuantizeDeviceRows) ||
               operation == int(Operation::QuantizeRowsVectorDeviceRows);
    }

    static int vector_activation(const Args& args) {
        return (args.operation==int(Operation::ActivationQuantizeDeviceRows) ||
                args.operation==int(Operation::ActivationBackwardQuantizeDeviceRows)) ? args.params.dim[3] : 0;
    }
    static std::string kernel_name(const Args& args) {
        const auto name = fmt::format("mega_moe_training_common_{}", args.operation);
        if(args.operation==int(Operation::PackFp8WeightGradients) && args.params.dim[4]>=6)return name+fmt::format("_dw_block{}",args.params.dim[4]==6?128:512);
        if(args.operation==int(Operation::PackFp8WeightGradients) && args.params.dim[4]>=4 && args.params.dim[4]<=5)
            return name+fmt::format("_cache_{}x{}_reciprocal",args.params.dim[4]==4?32:16,args.params.dim[4]==4?3072:6144);
        if (args.operation == int(Operation::PackFp8WeightGradients) && args.params.dim[4])
            return name + (args.params.dim[4]==1 || args.params.dim[4]==2 ? "_quantize_before_transpose" : "") +
                (args.params.dim[4]>=2 ? "_reciprocal" : "");
        if (!active_quantization(args.operation)) return name;
        return name + (args.params.dim[2] ? "_mn_scales" : "_row_scales") + (vector_activation(args)?fmt::format("_vec{}",vector_activation(args)==2?32:vector_activation(args)==4?8:16):"");
    }

    static std::string generate_impl(const Args& args) {
        auto source=fmt::format(R"(
#include <deep_gemm/impls/mega_moe_training/mega_moe_training_common.cuh>
extern "C" __launch_bounds__({}, 1)
__global__ void {}(const deep_gemm::mega_moe_training::Params params) {{
    deep_gemm::mega_moe_training::run<static_cast<deep_gemm::mega_moe_training::Operation>({}){}>(params);
}}
)", args.launch_args.block_dim.x,kernel_name(args), args.operation,
       args.operation==int(Operation::PackFp8WeightGradients) && args.params.dim[4]>=6 ? fmt::format(", false, false, true, 0, 0, true, {}",args.params.dim[4]==6?128:512) :
       args.operation==int(Operation::PackFp8WeightGradients) && args.params.dim[4]>=4 && args.params.dim[4]<=5 ?
           fmt::format(", false, false, true, {}, {}",args.params.dim[4]==4?32:16,args.params.dim[4]==4?3072:6144) :
       args.operation == int(Operation::PackFp8WeightGradients) && args.params.dim[4] ?
           fmt::format(", false, {}, {}",args.params.dim[4]==1 || args.params.dim[4]==2 ? "true" : "false",
                       args.params.dim[4]>=2 ? "true" : "false") :
       !active_quantization(args.operation) ? "" : args.params.dim[2] ? ", true" : ", false");
        return fmt::format("#define DG_TRAINING_VECTOR_ACTIVATION {}\n",int(vector_activation(args)))+source;
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

// Internal, allocation-free tensor dispatcher. All kernels use the caller's
// current stream. It validates host-known shapes/types without reading device
// metadata, so warmed calls are legal during graph capture.
static void launch(const int operation, const std::vector<torch::Tensor>& tensors,
                   const std::vector<int64_t>& dims) {
    DG_HOST_ASSERT(operation >= 0 && operation <= int(Operation::CompactReplayPrefix));
    DG_HOST_ASSERT(!tensors.empty() && tensors.size() <= 12 && dims.size() <= 12);
    DG_HOST_ASSERT(tensors[0].is_cuda());
    const c10::cuda::CUDAGuard guard(tensors[0].device());
    for (const auto& tensor : tensors)
        DG_HOST_ASSERT(tensor.device() == tensors[0].device() && tensor.is_contiguous());
    for (const auto dimension : dims)
        DG_HOST_ASSERT(dimension >= 0 && dimension <= INT32_MAX);

    const auto schema = [&](size_t nt, size_t nd) {
        DG_HOST_ASSERT(tensors.size() == nt && dims.size() == nd);
    };
    const auto expect = [&](size_t i, int64_t elements, torch::ScalarType dtype) {
        DG_HOST_ASSERT(tensors[i].numel() == elements && tensors[i].scalar_type() == dtype);
    };
    constexpr auto bf = torch::kBFloat16, i32 = torch::kInt32, i64 = torch::kInt64;
    constexpr auto fp = torch::kFloat, f8 = torch::kFloat8_e4m3fn;
    int64_t work = 0, blocks = -1;
    switch (static_cast<Operation>(operation)) {
        case Operation::QuantizeWeightsAndTranspose: {
            DG_HOST_ASSERT(tensors.size()==10 && (dims.size()==3 || dims.size()==4));
            const auto e=dims[0],h=dims[1],i=dims[2],ctas=dims.size()==4?dims[3]:4;
            DG_HOST_ASSERT(e>0 && e<128 && h>0 && i>0 && h%128==0 && i%128==0 && ctas>0 && ctas<=16);
            DG_HOST_ASSERT(e*h*2*i<=INT32_MAX);
            expect(0,e*2*i*h,bf);expect(1,e*h*i,bf);
            expect(2,e*2*i*h,f8);expect(3,e*(2*i/128)*(h/128),fp);
            expect(4,e*h*i,f8);expect(5,e*(h/128)*(i/128),fp);
            expect(6,e*h*2*i,f8);expect(7,e*(h/128)*(2*i/128),fp);
            expect(8,e*i*h,f8);expect(9,e*(i/128)*(h/128),fp);
            for(const auto index:{0,1,2,4,6,8})
                DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[index].data_ptr())%16==0);
            blocks=std::min<int64_t>(e*((2*i/128)*(h/128)+(h/128)*(i/128)),int64_t(get_num_sms())*ctas);
            break;
        }
        case Operation::PackFp8WeightGradients: {
            DG_HOST_ASSERT((tensors.size()==9 && (dims.size()==4 || dims.size()==6)) ||
                           (tensors.size()==10 && dims.size()==7 && dims[4]>=6));
            const auto mode=dims.size()>=6?dims[4]:0;
            const auto ctas=dims.size()>=6?dims[5]:4;
            DG_HOST_ASSERT(mode<=7 && ctas>0 && ctas<=16);
            const auto g=dims[0],r=dims[1],m=dims[2],n=dims[3];
            DG_HOST_ASSERT(g>0 && g<128 && r>0 && r%128==0 && m>0 && n>0 && m%128==0 && n%128==0);
            DG_HOST_ASSERT(r*m<=INT32_MAX && r*n<=INT32_MAX);
            expect(0,r*m,bf);expect(1,r*n,bf);expect(2,r*m,f8);expect(3,mode>=6?(r/128)*m:g*m,fp);
            expect(4,r*n,f8);expect(5,mode>=6?(n/128)*(r/128):g*n,fp);expect(6,g,i32);expect(7,g+1,i32);expect(8,1,i32);
            if(tensors.size()==10) {expect(9,r*(m/128),fp);DG_HOST_ASSERT(dims[6]<=1);}
            if(mode>=6)for(const auto index:{0,1,2,4})DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[index].data_ptr())%16==0);
            for(const auto index:{0,1}) DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[index].data_ptr())%8==0);
            for(const auto index:{2,4}) DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[index].data_ptr())%4==0);
            blocks=std::min<int64_t>(g*((m+n)/(mode==4?32:mode==5?16:128)),int64_t(get_num_sms())*ctas);
            break;
        }
        case Operation::TransposeQuantizedWeightsSwizzled:
        case Operation::TransposeQuantizedWeights: {
            schema(8, 3);
            const auto e = dims[0], h = dims[1], i = dims[2];
            DG_HOST_ASSERT(e > 0 && h > 0 && i > 0 && h % 128 == 0 && i % 128 == 0);
            DG_HOST_ASSERT(e * h * 2 * i <= INT32_MAX);
            expect(0, e * 2 * i * h, f8); expect(1, e * (2 * i / 128) * (h / 128), fp);
            expect(2, e * h * i, f8); expect(3, e * (h / 128) * (i / 128), fp);
            expect(4, e * h * 2 * i, f8); expect(5, e * (h / 128) * (2 * i / 128), fp);
            expect(6, e * i * h, f8); expect(7, e * (i / 128) * (h / 128), fp);
            for (const auto index : {0, 2, 4, 6})
                DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[index].data_ptr()) % 16 == 0);
            const auto tiles = e * (2 * i / 128) * (h / 128) + e * (h / 128) * (i / 128);
            blocks = std::min<int64_t>(tiles, int64_t(get_num_sms()) * 4);
            break;
        }
        case Operation::PackQuantizedRouteRows:
            schema(7, 4);
            DG_HOST_ASSERT(dims[0] > 0 && dims[1] > 0 && dims[1] % 128 == 0 && dims[2] > 0 && dims[3] > 0);
            DG_HOST_ASSERT(dims[0] * dims[1] <= INT32_MAX && dims[2] * dims[1] <= INT32_MAX);
            expect(0, dims[2] * dims[1], f8); expect(1, dims[2] * (dims[1] / 128), fp);
            expect(2, dims[0], i32); expect(3, dims[0] * dims[1], f8);
            expect(4, dims[0] * (dims[1] / 128), fp); expect(5, 1, i32); expect(6, 1, i32);
            blocks = int64_t(get_num_sms()) * 4;
            break;
        case Operation::CountRoutes: {
            schema(5, 4);
            const auto r = dims[0], e = dims[1];
            DG_HOST_ASSERT(e > 0 && dims[3] > 0 && dims[2] < dims[3]);
            expect(0, r, i64); expect(1, e, i32); expect(2, r, i32);
            expect(3, r, i32); expect(4, 1, i32);
            work = r;
            break;
        }
        case Operation::PrefixRoutes:
            schema(3, 2);
            DG_HOST_ASSERT(dims[0] > 0 && dims[1] > 0 && dims[1] % 128 == 0);
            expect(0, dims[0], i32); expect(1, dims[0] + 1, i32); expect(2, dims[0], i32);
            blocks = 1;
            break;
        case Operation::InitializeLayout:
            schema(3, 2);
            DG_HOST_ASSERT(dims[1] > 0);
            expect(0, dims[1] + 1, i32); expect(1, dims[0], i32); expect(2, dims[0], i32);
            work = dims[0];
            break;
        case Operation::PackInput: {
            schema(7, 4);
            const auto r = dims[0], h = dims[1], k = dims[2], m = dims[3];
            DG_HOST_ASSERT(h > 0 && k > 0 && r % k == 0 && tensors[3].numel() > 1);
            expect(0, r / k * h, bf); expect(1, r, i32); expect(2, r, i32);
            DG_HOST_ASSERT(tensors[3].scalar_type() == i32);
            expect(4, m * h, bf); expect(5, m, i32); expect(6, 1, i32);
            blocks = r;
            break;
        }
        case Operation::UnpackRoutes:
        case Operation::PackRouteGrad: {
            schema(3, 3);
            const auto m = dims[0], h = dims[1], r = dims[2];
            DG_HOST_ASSERT(h > 0);
            expect(1, m, i32);
            if (static_cast<Operation>(operation) == Operation::UnpackRoutes) {
                expect(0, m * h, bf); expect(2, r * h, bf);
            } else {
                expect(0, r * h, bf); expect(2, m * h, bf);
            }
            work = m * h;
            break;
        }
        case Operation::Combine:
            DG_HOST_ASSERT(tensors.size() == 3 || tensors.size() == 4);
            schema(tensors.size(), 6);
            if (tensors.size() == 4) expect(3, dims[0] * dims[2], i64);
            DG_HOST_ASSERT(dims[1] > 0 && dims[2] > 0 && dims[5] <= 1);
            expect(0, dims[0] * dims[1] * dims[2], bf);
            expect(1, dims[3] * dims[2], fp); expect(2, dims[3] * dims[1], bf);
            work = dims[0] * dims[1];
            if (dims[1] % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(tensors[0].data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(tensors[2].data_ptr()) % 16 == 0)
                work /= 8;
            break;
        case Operation::Activation:
        case Operation::ActivationBackward:
            DG_HOST_ASSERT(dims.size() == 2 || dims.size() == 3);
            schema(static_cast<Operation>(operation) == Operation::Activation ? 2 : 3, dims.size());
            if (dims.size() == 3) DG_HOST_ASSERT(dims[2] == 8);
            DG_HOST_ASSERT(dims[1] > 0);
            expect(0, dims[0] * dims[1] * 2, bf); expect(1, dims[0] * dims[1], bf);
            if (tensors.size() == 3) expect(2, dims[0] * dims[1] * 2, bf);
            work = dims[0] * dims[1];
            break;
        case Operation::QuantizeRows:
            schema(3, 2);
            DG_HOST_ASSERT(dims[1] > 0 && dims[1] % 128 == 0);
            expect(0, dims[0] * dims[1], bf); expect(1, dims[0] * dims[1], f8);
            expect(2, dims[0] * (dims[1] / 128), fp);
            blocks = (dims[0] * (dims[1] / 128) + 7) / 8;
            break;
        case Operation::ActivationQuantize:
        case Operation::ActivationBackwardQuantize: {
            const bool backward = static_cast<Operation>(operation) == Operation::ActivationBackwardQuantize;
            schema(backward ? 5 : 4, 2);
            const auto m = dims[0], width = dims[1];
            DG_HOST_ASSERT(width > 0 && width % 128 == 0);
            expect(0, m * 2 * width, bf); expect(1, m * width, bf);
            if (backward) expect(2, m * 2 * width, bf);
            const auto output_width = backward ? 2 * width : width;
            expect(backward ? 3 : 2, m * output_width, f8);
            expect(backward ? 4 : 3, m * (output_width / 128), fp);
            blocks = (m * (width / 128) + 7) / 8;
            break;
        }
        case Operation::QuantizeRowsVectorDeviceRows:
        case Operation::QuantizeRowsDeviceRows:
        case Operation::ActivationQuantizeDeviceRows:
        case Operation::ActivationBackwardQuantizeDeviceRows: {
            const bool row = static_cast<Operation>(operation) == Operation::QuantizeRowsDeviceRows ||
                             static_cast<Operation>(operation) == Operation::QuantizeRowsVectorDeviceRows;
            const bool backward = static_cast<Operation>(operation) == Operation::ActivationBackwardQuantizeDeviceRows;
            const size_t count_index = row ? 3 : backward ? 5 : 4;
            schema(count_index + 2, !row && dims.size()==4 ? 4 : 3);
            DG_HOST_ASSERT(dims.size()==3 || (!row && (dims[3]==0 || dims[3]==1 || dims[3]==2 || dims[3]==4)));
            if(!row && dims.size()==4 && dims[3]) {
                for(int j=0;j<(backward?3:2);++j)DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[j].data_ptr())%16==0);
                DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[backward?3:2].data_ptr())%8==0);
            }
            const auto m = dims[0], width = dims[1];
            DG_HOST_ASSERT(m > 0 && width > 0 && width % 128 == 0 && dims[2] <= 1);
            expect(0, m * width * (row ? 1 : 2), bf);
            if (!row) expect(1, m * width, bf);
            if (backward) expect(2, m * 2 * width, bf);
            const auto output_width = backward ? 2 * width : width;
            if(static_cast<Operation>(operation)==Operation::QuantizeRowsVectorDeviceRows) {
                DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[0].data_ptr())%16==0);
                DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[1].data_ptr())%8==0);
            }
            expect(row ? 1 : backward ? 3 : 2, m * output_width, f8);
            expect(row ? 2 : backward ? 4 : 3, m * (output_width / 128), fp);
            expect(count_index, 1, i32); expect(count_index + 1, 1, i32);
            blocks = int64_t(get_num_sms()) * 4;
            break;
        }
        case Operation::QuantizeWeights:
        case Operation::QuantizeWeightsCachedFloat:
        case Operation::QuantizeWeightsCachedInteger:
            schema(3, 3);
            DG_HOST_ASSERT(dims[0] > 0 && dims[1] > 0 && dims[2] > 0);
            DG_HOST_ASSERT(dims[1] % 128 == 0 && dims[2] % 128 == 0);
            expect(0, dims[0] * dims[1] * dims[2], bf);
            expect(1, dims[0] * dims[1] * dims[2], f8);
            if (static_cast<Operation>(operation) != Operation::QuantizeWeights) {
                DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[0].data_ptr()) % 16 == 0);
                DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[1].data_ptr()) % 8 == 0);
            }
            blocks = dims[0] * (dims[1] / 128) * (dims[2] / 128);
            expect(2, blocks, fp);
            break;
        case Operation::WeightedGradient:
        case Operation::ScoreGradient:
            schema(3, 5);
            DG_HOST_ASSERT(dims[1] > 0 && dims[2] > 0);
            if (static_cast<Operation>(operation) == Operation::WeightedGradient) {
                expect(0, dims[3] * dims[1], bf); expect(1, dims[3] * dims[2], fp);
                expect(2, dims[0] * dims[1] * dims[2], bf);
                work = dims[0] * dims[1] * dims[2];
            } else {
                expect(0, dims[0] * dims[1] * dims[2], bf);
                expect(1, dims[3] * dims[1], bf); expect(2, dims[3] * dims[2], fp);
                blocks = dims[0] * dims[2];
            }
            break;
        case Operation::ReadChunk:
            schema(6, 5);
            DG_HOST_ASSERT(dims[1] > 0 && dims[2] > 0 && dims[3] > 0);
            expect(0, dims[0] * dims[2], bf); expect(1, dims[0] * dims[3], i64);
            expect(2, dims[1] * dims[2], bf); expect(3, dims[1] * dims[3], i64);
            expect(4, 1, i32); expect(5, 1, i32);
            // Match the device's eight-BF16 copy when both row matrices are
            // aligned; offset views retain the original scalar contract.
            if (dims[2] % 8 == 0 &&
                reinterpret_cast<uintptr_t>(tensors[0].data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(tensors[2].data_ptr()) % 16 == 0) {
                work = dims[1] * std::max(dims[2] / 8, dims[3]);
            } else {
                work = dims[1] * std::max(dims[2], dims[3]);
            }
            break;
        case Operation::ActivationDeviceRows:
            DG_HOST_ASSERT(dims.size() == 3 || dims.size() == 4);
            schema(4, dims.size());
            if (dims.size() == 4) DG_HOST_ASSERT(dims[3] == 8);
            DG_HOST_ASSERT(dims[1] > 0 && dims[2] <= 1);
            expect(0, dims[0] * 2 * dims[1], bf); expect(1, dims[0] * dims[1], bf);
            expect(2, 1, i32); expect(3, 1, i32);
            blocks = int64_t(get_num_sms()) * 4;
            break;
        case Operation::TransformDeviceRows:
        case Operation::TransformDeviceRowsVector: {
            schema(5, 4);
            DG_HOST_ASSERT(dims[1] > 0 && dims[3] <= 1);
            const bool backward = dims[3];
            expect(0, (backward ? dims[2] : dims[0]) * dims[1], bf);
            expect(1, dims[0], i32);
            expect(2, (backward ? dims[0] : dims[2]) * dims[1], bf);
            expect(3, 1, i32); expect(4, 1, i32);
            if (static_cast<Operation>(operation) == Operation::TransformDeviceRowsVector)
                DG_HOST_ASSERT(dims[1] % 8 == 0);
            blocks = int64_t(get_num_sms()) * 4;
            break;
        }
        case Operation::ZeroDeviceRows:
            schema(3, 2);
            DG_HOST_ASSERT(dims[1] > 0);
            expect(0, dims[0] * dims[1], bf); expect(1, 1, i32); expect(2, 1, i32);
            blocks = int64_t(get_num_sms()) * 4;
            break;
        case Operation::BuildPermutationOnly:
            schema(5, 2);
            expect(0, dims[0], i32); expect(1, dims[0], i32);
            DG_HOST_ASSERT(tensors[2].numel() > 1 && tensors[2].scalar_type() == i32);
            expect(3, dims[1], i32); expect(4, 1, i32);
            work = dims[0];
            break;
        case Operation::PackInputDeviceRows:
            schema(5, 4);
            DG_HOST_ASSERT(dims[1] > 0 && dims[1] % 8 == 0 && dims[2] > 0 && dims[3] > 0);
            expect(0, dims[2] * dims[1], bf); expect(1, dims[0], i32);
            expect(2, dims[0] * dims[1], bf); expect(3, 1, i32); expect(4, 1, i32);
            blocks = int64_t(get_num_sms()) * 4;
            break;
        case Operation::DeepEPZeroActiveReceive:
        case Operation::DeepEPCastActiveReceive: {
            const bool cast=operation==int(Operation::DeepEPCastActiveReceive);
            schema(cast?4:3,2);
            const auto capacity=dims[0],width=dims[1];
            DG_HOST_ASSERT(capacity>0 && width>0 && width%8==0);
            expect(0,capacity*width,fp);
            if(cast)expect(1,capacity*width,bf);
            expect(cast?2:1,1,i32);expect(cast?3:2,1,i32);
            DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[0].data_ptr())%16==0);
            if(cast)DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[1].data_ptr())%16==0);
            blocks=int64_t(get_num_sms())*4;
            break;
        }
        case Operation::PrefixCompactRoutes: {
            schema(4, 3);
            const auto e=dims[0], alignment=dims[1], capacity=dims[2];
            DG_HOST_ASSERT(e>0 && e<=1024 && alignment>0 && capacity>0 && capacity%alignment==0);
            expect(0,e,i32);expect(1,e+1,i32);expect(2,e,i32);expect(3,1,i32);
            blocks=1;
            break;
        }
        case Operation::CompactReplayPrefix: {
            schema(7, 3);
            const auto e=dims[0], capacity=dims[1];
            DG_HOST_ASSERT(e>0 && e<=1024 && capacity>0 && capacity%128==0 && dims[2]%128==0);
            expect(0,e+1,i32);expect(1,e,i32);expect(2,e+1,i32);expect(3,e,i32);
            expect(4,1,i32);expect(5,1,i64);expect(6,1,i32);
            blocks=1;
            break;
        }
        case Operation::DeepEPReduceCompactReceive: {
            schema(8, 6);
            const auto m=dims[0], h=dims[1], n=dims[2], k=dims[3], e=dims[4];
            DG_HOST_ASSERT(m>0 && h>0 && h%256==0 && n>0 && k>0 && k<=32 && e>0 && dims[5]<=1);
            expect(0,m*h,bf);expect(1,n*k,i32);expect(2,n*k,i32);expect(3,e+1,i32);
            expect(4,n*k,fp);expect(5,n*h,bf);expect(6,1,i32);expect(7,1,i32);
            for(const auto at:{0,5})DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[at].data_ptr())%16==0);
            blocks=int64_t(get_num_sms())*4;
            break;
        }
        case Operation::DeepEPReduceRows: {
            schema(7, 6);
            const auto m = dims[0], h = dims[1], n = dims[2], k = dims[3];
            DG_HOST_ASSERT(h > 0 && h % 8 == 0 && k > 0 && dims[4] > 0 && dims[5] <= 1);
            expect(0, m * h, bf); expect(1, n * k, i32); expect(2, n * k, i32);
            expect(3, dims[4] + 1, i32); expect(4, n * k, fp);
            expect(5, n * h, bf); expect(6, 1, i32);
            blocks = int64_t(get_num_sms()) * 4;
            break;
        }
        case Operation::DeepEPReduceWindowRows: {
            schema(8, 8);
            const auto m = dims[0], h = dims[1], n = dims[2], k = dims[3];
            DG_HOST_ASSERT(h > 0 && h % 8 == 0 && k > 0 && k <= 32 && dims[4] > 0
                           && dims[5] % 128 == 0 && dims[6] <= 1 && dims[7] >= m);
            expect(0, m * h, bf); expect(1, n * k, i32); expect(2, n * k, i32);
            expect(3, dims[4] + 1, i32); expect(4, n * k, fp); expect(5, n * h, fp);
            expect(6, 1, i32); expect(7, 1, i32);
            blocks = int64_t(get_num_sms()) * 4;
            break;
        }
        case Operation::DeepEPBackwardRows: {
            schema(8, 4);
            const auto m = dims[0], h = dims[1], n = dims[2], k = dims[3];
            DG_HOST_ASSERT(h > 0 && k > 0);
            expect(0, n * h, bf); expect(1, n * k, fp); expect(2, m, i32);
            expect(3, m * h, bf); expect(4, m * h, bf); expect(5, n * k, fp);
            expect(6, 1, i32); expect(7, 1, i32);
            blocks = int64_t(get_num_sms()) * 4;
            break;
        }
        case Operation::TransposeWeights: {
            schema(4, 3);
            const auto e = dims[0], h = dims[1], i = dims[2];
            DG_HOST_ASSERT(e > 0 && h > 0 && i > 0 && h % 128 == 0 && i % 128 == 0);
            expect(0, e * 2 * i * h, bf); expect(1, e * h * i, bf);
            expect(2, e * h * 2 * i, bf); expect(3, e * i * h, bf);
            for (const auto& tensor : tensors)
                DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0);
            blocks = e * (2 * i / 128) * (h / 128) + e * (h / 128) * (i / 128);
            break;
        }
    }
    if (blocks < 0) blocks = (work + 255) / 256;
    if (!blocks) return;
    DG_HOST_ASSERT(blocks <= INT32_MAX);
    Params params{};
    for (size_t i = 0; i < tensors.size(); ++i) params.ptr[i] = tensors[i].data_ptr();
    for (size_t i = 0; i < dims.size(); ++i) params.dim[i] = dims[i];
    Runtime::Args args{operation, {dim3(blocks), dim3(operation==int(Operation::PackFp8WeightGradients) && params.dim[4]==7 ? 512 : operation==int(Operation::PackFp8WeightGradients) && params.dim[4]>=4 && params.dim[4]<=5 ? 1024 : 256), 0}, params};
    const auto code = Runtime::generate(args);
    const auto runtime = compiler->build(Runtime::kernel_name(args), code);
    Runtime::launch(runtime, args);
}

// Compile the fixed pointwise variants without publishing EP payloads. GEMM
// preparation alone cannot protect peers from a later pointwise compiler error.
static void precompile(const int device, const std::vector<int>& operations,
                       const int vector_activation = 0) {
    DG_HOST_ASSERT(vector_activation == 0 || vector_activation == 1 ||
                   vector_activation == 2 || vector_activation == 4);
    const c10::cuda::CUDAGuard guard(device);
    std::vector<std::pair<std::string, std::string>> kernels;
    for (const auto operation : operations) {
        DG_HOST_ASSERT(operation >= 0 && operation <= int(Operation::CompactReplayPrefix));
        const int variants = Runtime::active_quantization(operation) ? 2 : 1;
        for (int layout = 0; layout < variants; ++layout) {
            Runtime::Args args{operation, {dim3(1), dim3(256), 0}, Params{}};
            args.params.dim[2] = layout;
            args.params.dim[3] = vector_activation;
            kernels.emplace_back(Runtime::kernel_name(args), Runtime::generate(args));
        }
    }
    for (const auto& kernel : kernels)
        compiler->build(kernel.first, kernel.second);
}

static std::tuple<int,int,int,int,int,int,int,std::tuple<int,int,int>>
resolve_nopad_config(const int device, const std::string& compute,
                     const int rows, const int n, const int k, const int groups) {
    DG_HOST_ASSERT(rows>0 && n>0 && k>0 && groups>0 && (compute=="bf16" || compute=="fp8"));
    const c10::cuda::CUDAGuard guard(device);
    const int expected_m=rows/groups+(rows%groups!=0);
    if(compute=="bf16")
        return deep_gemm_bf16_common::get_best_configs(expected_m,n,k,groups,get_num_sms(),GemmType::MGroupedNoPad);
    return deep_gemm_fp8_common::get_best_configs(expected_m,n,k,groups,get_num_sms(),GemmType::MGroupedNoPad);
}

static void register_apis(pybind11::module_& m) {
    m.def("_mega_moe_training_common_launch", &launch, py::arg("operation"), py::arg("tensors"), py::arg("dims"));
    m.def("_mega_moe_training_precompile", &precompile, py::arg("device"), py::arg("operations"),
          py::arg("vector_activation") = 0);
    m.def("_mega_moe_training_nopad_config", &resolve_nopad_config, py::arg("device"),py::arg("compute"),
          py::arg("rows"),py::arg("n"),py::arg("k"),py::arg("groups"));
}

}  // namespace deep_gemm::mega_moe_training


#include <c10/cuda/CUDAGuard.h>
#include "../jit_kernels/impls/k_grouped_bf16_gemm.hpp"
#include <deep_gemm/common/mega_moe_training.cuh>
#include "../jit_kernels/heuristics/common_fp8.hpp"
namespace deep_gemm::mega_moe_training {
class Fp8NtWgradRuntime final:public LaunchRuntime<Fp8NtWgradRuntime> {
public:
    struct Args {
        Bf16KGroupedRuntime::LaunchInfo info;
        LaunchArgs launch_args;
        KGroupedGemmArgs params;
        Fp8WgradParams wgrad;
        const float* sa;
        const float* sb;
        int promotion_k;
        bool block_scales;
        bool native_blockwise;
        int swizzle_m;
        int metadata_mode;
        int scale_m;
        const int32_t* first_flags;
    };
    static auto compile_key(const Args& a) {
        const auto& i=a.info;
        return std::make_tuple(i.shape_m,i.shape_n,i.num_groups,i.block_m,i.block_n,i.block_k,i.warp_m,i.warp_n,i.num_stages,a.promotion_k,a.block_scales,a.first_flags!=nullptr,a.native_blockwise,a.swizzle_m,a.metadata_mode,a.scale_m);
    }
    static std::string generate_impl(const Args& a) {
        const auto& i=a.info;
        std::string source=fmt::format(R"(
#include <deep_gemm/impls/k_grouped_bf16_gemm.cuh>
#include <deep_gemm/common/mega_moe_training.cuh>
namespace deep_gemm {{
extern "C" __launch_bounds__({0},1)
__global__ void {13}(KGroupedGemmArgs args,const mega_moe_training::Fp8WgradParams dw,
                                    const float* sa,const float* sb,const int32_t* first_flags) {{
    __shared__ int32_t rounded[{3}];
    for(uint32_t g=threadIdx.x;g<{3};g+=blockDim.x) {{
        const int32_t count=dw.lengths[g],start=dw.offsets[g],end=dw.offsets[g+1];
        const bool valid=count>=0 && start>=0 && start%128==0 && end>=start && end<=dw.capacity &&
            int64_t(start)+((int64_t(count)+127)/128)*128<=end;
        rounded[g]=valid?((count+127)/128)*128:0;
        if(!valid)atomicOr(dw.status,2u);
    }}
    __syncthreads();
    args.grouped_layout=rounded;
    bf16_k_grouped_gemm_impl<cutlass::float_e4m3_t,float,GemmType::KGroupedContiguous,
        {1},{2},{3},{4},{5},{6},{7},{8},{0},{9},true,0,true,0,0,true,true,{12},{10},{11}>(args,0,~0u,sa,sb,dw.offsets,first_flags);
}}
}}
)",(i.block_m/i.warp_m)*(i.block_n/i.warp_n)*32,i.shape_m,i.shape_n,i.num_groups,
            i.block_m,i.block_n,i.block_k,i.warp_m,i.warp_n,i.num_stages,a.promotion_k,a.block_scales,a.first_flags!=nullptr,i.kernel_name);
        if(a.native_blockwise) {
            if(a.metadata_mode) {
                const auto validation_begin=source.find("    __shared__ int32_t rounded[");
                const std::string old_end="    args.grouped_layout=rounded;";
                const auto validation_end=source.find(old_end,validation_begin)+old_end.size();
                DG_HOST_ASSERT(validation_begin!=std::string::npos && validation_end!=std::string::npos);
                std::string replacement=fmt::format(R"(
    // Every thread validates the same immutable GPU metadata. No static TSM
    // allocation is needed; actual per-expert K is rounded at its use site.
    bool metadata_valid=true;
    #pragma unroll
    for(uint32_t g=0;g<{};++g) {{
        const int64_t count=__ldg(dw.lengths+g),start=__ldg(dw.offsets+g),end=__ldg(dw.offsets+g+1);
        metadata_valid=metadata_valid && count>=0 && start>=0 && start%128==0 && end>=start && end<=dw.capacity &&
            start+((count+127)/128)*128<=end;
    }}
    if(!metadata_valid) {{if(threadIdx.x==0)atomicOr(dw.status,2u);return;}}
)",i.num_groups);
                if(a.metadata_mode==2)replacement+=R"(
    // Diagnostic control: restore exactly128 bytes of static shared storage
    // while retaining the same global-K data path and native math.
    __shared__ volatile int32_t metadata_padding[32];
    if(threadIdx.x<32)metadata_padding[threadIdx.x]=int32_t(threadIdx.x);
    __syncthreads();
    if(metadata_padding[threadIdx.x&31]>=32)return;
)";
                replacement+="    args.grouped_layout=dw.lengths;";
                source.replace(validation_begin,validation_end-validation_begin,replacement);
            }
            const std::string include="#include <deep_gemm/impls/k_grouped_bf16_gemm.cuh>";
            source.replace(source.find(include),include.size(),"#include <deep_gemm/impls/mega_moe_training/fp8_k_grouped_native.cuh>");
            const auto begin=source.find("    bf16_k_grouped_gemm_impl<");
            const auto end=source.find(";",begin);
            source.replace(begin,end-begin+1,fmt::format(
                "    native_fp8_k_grouped_gemm_impl<{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}>(args,sa,sb,dw.offsets,first_flags);",
                i.shape_m,i.shape_n,i.num_groups,i.block_m,i.block_n,i.block_k,i.warp_m,i.warp_n,i.num_stages,a.first_flags!=nullptr,a.swizzle_m,a.metadata_mode!=0,a.scale_m));
        }
        return source;
    }
    static void launch_impl(const KernelHandle& kernel,const LaunchConfigHandle& config,Args a) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel,config,a.params,a.wgrad,a.sa,a.sb,a.first_flags));
    }
};
static int fp8_wgrad_nt(const std::pair<torch::Tensor,torch::Tensor>& lhs,const std::pair<torch::Tensor,torch::Tensor>& rhs,
                       const torch::Tensor& output,const torch::Tensor& lengths,const torch::Tensor& offsets,
                       const torch::Tensor& status,bool compile_only,const std::vector<int>& config,int promotion_k,bool block_scales,const std::optional<torch::Tensor>& first_flags,bool native_blockwise,int swizzle_m,int metadata_mode,int scale_m) {
    DG_HOST_ASSERT(scale_m==1 || scale_m==16 || scale_m==32 || scale_m==64 || scale_m==128);
    DG_HOST_ASSERT(scale_m==1 || (native_blockwise && block_scales));
    DG_HOST_ASSERT(metadata_mode>=0 && metadata_mode<=2 && (native_blockwise || metadata_mode==0));
    DG_HOST_ASSERT(swizzle_m>=1 && swizzle_m<=64 && (swizzle_m&(swizzle_m-1))==0 && (native_blockwise || swizzle_m==2));
    const auto& a=lhs.first;const auto& sa=lhs.second;const auto& b=rhs.first;const auto& sb=rhs.second;
    DG_HOST_ASSERT(a.is_cuda() && a.dim()==2 && b.dim()==2 && output.dim()==3);
    const c10::cuda::CUDAGuard guard(a.device());DG_HOST_ASSERT(get_ppu_arch()==AC_PPU0015);
    for(const auto& t:{a,sa,b,sb,output,lengths,offsets,status})
        DG_HOST_ASSERT(t.is_cuda() && t.device()==a.device() && t.is_contiguous());
    const auto [m,r]=get_shape<2>(a);const auto [n,rb]=get_shape<2>(b);const auto g=output.size(0);
    DG_HOST_ASSERT(m>0 && n>0 && r>0 && r==rb && m%128==0 && n%128==0 && r%128==0);
    DG_HOST_ASSERT(g>0 && g<128 && output.size(1)==m && output.size(2)==n);
    DG_HOST_ASSERT(a.numel()<=INT32_MAX && b.numel()<=INT32_MAX && output.numel()<=INT32_MAX);
    DG_HOST_ASSERT(a.scalar_type()==torch::kFloat8_e4m3fn && b.scalar_type()==torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(sa.scalar_type()==torch::kFloat && sb.scalar_type()==torch::kFloat && output.scalar_type()==torch::kFloat);
    DG_HOST_ASSERT(sa.dim()==2 && sb.dim()==2);
    if(block_scales)DG_HOST_ASSERT(sa.size(0)==r/128 && sa.size(1)==m/scale_m && sb.size(0)==n/128 && sb.size(1)==r/128);
    else DG_HOST_ASSERT(sa.size(0)==g && sa.size(1)==m && sb.size(0)==g && sb.size(1)==n);
    DG_HOST_ASSERT(lengths.scalar_type()==torch::kInt && offsets.scalar_type()==torch::kInt && status.scalar_type()==torch::kInt);
    DG_HOST_ASSERT(lengths.numel()==g && offsets.numel()==g+1 && status.numel()==1);
    if(first_flags)DG_HOST_ASSERT(first_flags->is_cuda() && first_flags->device()==a.device() && first_flags->is_contiguous() && first_flags->scalar_type()==torch::kInt && first_flags->numel()==g);
    const auto overlaps_output = [&](const torch::Tensor& tensor) {
        const auto begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
        const auto output_begin = reinterpret_cast<uintptr_t>(output.data_ptr());
        return tensor.numel() && begin < output_begin + output.nbytes()
            && output_begin < begin + tensor.nbytes();
    };
    for (const auto& tensor : {a, sa, b, sb, lengths, offsets, status})
        DG_HOST_ASSERT(!overlaps_output(tensor));
    if (first_flags)
        DG_HOST_ASSERT(!overlaps_output(*first_flags));
    int bm=256,bn=256,bk=128,wm=64,wn=64,stages=3;
    if(!config.empty()) {DG_HOST_ASSERT(config.size()==6);bm=config[0];bn=config[1];bk=config[2];wm=config[3];wn=config[4];stages=config[5];}
    DG_HOST_ASSERT(bm>0 && bn>0 && bk>0 && (native_blockwise?(bk==128 || bk==256):128%bk==0) && wm>0 && wn>0 && bm%wm==0 && bn%wn==0 && stages>=2);
    DG_HOST_ASSERT(bm%scale_m==0 && m%scale_m==0);
    int bytes=int(gemm_smem_total_size(1,stages,bm,bn,bk));
    if(native_blockwise)bytes=std::get<0>(deep_gemm_fp8_common::get_smem_config(stages,1024,bm,bn,bk));DG_HOST_ASSERT(bytes+g*4<=256*1024);
    if(block_scales) {if(promotion_k==0)promotion_k=128;DG_HOST_ASSERT(promotion_k==128 || promotion_k==512);if(!native_blockwise)DG_HOST_ASSERT(128%wm==0 && 128%wn==0 && (bm%128==0 || 128%bm==0) && (bn%128==0 || 128%bn==0));}
    DG_HOST_ASSERT(promotion_k>=0 && promotion_k<=16384 && (native_blockwise || promotion_k%bk==0));
    std::string name=block_scales?fmt::format("fp8_training_nt_wgrad_block128_k{}",promotion_k):promotion_k?fmt::format("fp8_training_nt_wgrad_promote{}",promotion_k):"fp8_training_nt_wgrad";
    if(first_flags)name+="_first";
    if(scale_m!=1)name+=fmt::format("_sfm{}",scale_m);
    if(metadata_mode)name+=fmt::format("_metadata{}",metadata_mode);
    if(swizzle_m!=2)name+=fmt::format("_swizzle{}",swizzle_m);
    if(native_blockwise) {DG_HOST_ASSERT(block_scales && promotion_k==128 && (bk==128 || bk==256));name+="_native_blockwise";}
    auto args=Fp8NtWgradRuntime::Args{
        .info={int(m),int(n),int(g),bm,bn,bk,wm,wn,stages,0,true,name},
        .launch_args={get_grid_shape(get_num_sms()),dim3((bm/wm)*(bn/wn)*32),bytes},
        .params={a.data_ptr(),b.data_ptr(),output.data_ptr(),lengths.data_ptr<int32_t>()},
        .wgrad={lengths.data_ptr<int32_t>(),offsets.data_ptr<int32_t>(),output.data_ptr<float>(),
                reinterpret_cast<uint32_t*>(status.data_ptr<int32_t>()),uint32_t(g),uint32_t(m),uint32_t(n),uint32_t(r)},
        .sa=sa.data_ptr<float>(),.sb=sb.data_ptr<float>(),.promotion_k=promotion_k,.block_scales=block_scales,.native_blockwise=native_blockwise,.swizzle_m=swizzle_m,.metadata_mode=metadata_mode,.scale_m=scale_m,.first_flags=first_flags?first_flags->data_ptr<int32_t>():nullptr};
    const auto runtime=compiler->build(name,Fp8NtWgradRuntime::generate(args));int occ=0;
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(&occ,runtime->kernel,args.launch_args.block_dim.x,args.launch_args.smem_size));
    DG_HOST_ASSERT(occ>0);args.launch_args.grid_dim.x*=occ;
    if(!compile_only)Fp8NtWgradRuntime::launch(runtime,args);
    return args.launch_args.grid_dim.x;
}
static void register_fp8_wgrad_nt_apis(pybind11::module_& m) {
    m.def("_mega_moe_training_fp8_wgrad_nt",&fp8_wgrad_nt,py::arg("a"),py::arg("b"),py::arg("output"),py::arg("lengths"),
          py::arg("offsets"),py::arg("status"),py::arg("compile_only")=false,py::arg("config")=std::vector<int>{},py::arg("promotion_k")=0,py::arg("block_scales")=false,py::arg("first_flags")=std::nullopt,py::arg("native_blockwise")=false,py::arg("swizzle_m")=2,py::arg("metadata_mode")=0,py::arg("scale_m")=1);
}
}

#include <algorithm>
#include <cstdio>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <deep_gemm/common/mega_moe_training.cuh>
#include "../jit/compiler.hpp"
#include "../jit/kernel_runtime.hpp"
#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/layout.hpp"
#include "../utils/math.hpp"
#include "../utils/utils.hpp"

namespace deep_gemm::mega_moe_training {

class MegaMoERuntime final : public LaunchRuntime<MegaMoERuntime> {
public:
    struct Args {
        int operation;
        LaunchArgs launch_args;
        Params params;
    };
    static std::string generate_impl(const Args& args) {
        const int specialize = get_env<int>("DG_INPUT_ASSIST_STATIC_GEOMETRY", 1);
        DG_HOST_ASSERT(specialize == 0 || specialize == 1);
        const int warp_rows = get_env<int>("DG_INPUT_ASSIST_WARP_ROWS", 1);
        DG_HOST_ASSERT(warp_rows == 0 || warp_rows == 1);
        const bool assist = args.operation == int(MegaMoEOperation::AssistInputTiles) && specialize;
        const bool tiled_quant=args.operation==int(MegaMoEOperation::ExpertLocalGradientsQuantizedTiles) ||
                               args.operation==int(MegaMoEOperation::ExpertLocalGradientsQuantizedTilesCached);
        const bool gradient_input = args.operation == int(MegaMoEOperation::GradientQuantizedInputTiles);
        const bool input_tiles = args.operation == int(MegaMoEOperation::PackQuantizedInputTiles) || gradient_input;
        const bool first_flags=args.operation==int(MegaMoEOperation::BuildWorkInfo) && args.params.ptr[4]!=nullptr;
        const bool expert_batch=args.operation==int(MegaMoEOperation::PackExpertBatch) || args.operation==int(MegaMoEOperation::GradientExpertBatch);
        const int source_ilp=get_env<int>("DG_MOE_FP8_EXPERT_SOURCE_ILP",1);
        DG_HOST_ASSERT(source_ilp==1 || source_ilp==2 || source_ilp==4);
        const bool fused_quant = expert_batch || input_tiles || args.operation == int(MegaMoEOperation::ReturnGemmTiles) || args.operation == int(MegaMoEOperation::PackActiveQuantized) ||
                                 args.operation == int(MegaMoEOperation::ExpertLocalGradientsQuantized) ||
                                 tiled_quant;
        const bool cache_fp8_scales = args.operation == int(MegaMoEOperation::StageFp8InputTiles) && args.params.dim[7];
        return fmt::format(R"(
#include <deep_gemm/impls/mega_moe_training/mega_moe_training.cuh>
extern "C" __launch_bounds__({}, 1)
__global__ void mega_moe_training_{}(const deep_gemm::mega_moe_training::Params params) {{
    deep_gemm::mega_moe_training::run_mega_moe<static_cast<deep_gemm::mega_moe_training::MegaMoEOperation>({}), {}, {}, {}{}>(params);
}}
)", args.launch_args.block_dim.x, args.operation, args.operation,
       fused_quant || assist || cache_fp8_scales || (args.operation == int(MegaMoEOperation::StageFp8InputTiles) && specialize) ? args.params.dim[1] : 0, assist ? args.params.dim[4] : (expert_batch || tiled_quant || input_tiles) ? args.params.dim[2] : 0,
       args.operation == int(MegaMoEOperation::AssistInputTiles) && warp_rows ? "true" : "false",
       first_flags ? ", false, 0, 1, false, 128, 0, true" :
       input_tiles ? fmt::format(", false, {}, {}, {}, {}",args.params.dim[3],args.params.dim[8] ? args.params.dim[8] : 1,gradient_input || args.params.dim[9] ? "true" : "false",
           gradient_input ? (args.params.dim[9] ? args.params.dim[9] : 128) : (args.params.dim[10] ? args.params.dim[10] : 128)) :
       expert_batch ? fmt::format(", false, {}, 1, false, {}, {}, false, {}",args.params.dim[3],args.params.dim[8] ? args.params.dim[8] : 16,args.params.dim[9],source_ilp)+(args.params.dim[10]==2 ? ", true, true" : args.params.dim[10] ? ", true" : "") :
       tiled_quant ? fmt::format(", false, {}",args.params.dim[3]) : cache_fp8_scales ? ", true" : "");
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

static void launch_mega_moe(const int operation, const std::vector<torch::Tensor>& tensors,
                        const std::vector<int64_t>& dims, bool compile_only) {
    DG_HOST_ASSERT((operation >= 0 && operation <= int(MegaMoEOperation::GradientExpertBatch)) || operation==int(MegaMoEOperation::FinalizeEmptyWgrad) || operation==int(MegaMoEOperation::AggregateFirstWgradFlags) || operation==int(MegaMoEOperation::BuildGemmMetadata));
    DG_HOST_ASSERT(!tensors.empty() && tensors.size() <= 12 && dims.size() <= 12);
    DG_HOST_ASSERT(tensors[0].is_cuda());
    const c10::cuda::CUDAGuard guard(tensors[0].device());
    for (const auto& tensor : tensors)
        DG_HOST_ASSERT(tensor.device() == tensors[0].device() && tensor.is_contiguous());
    for (const auto dimension : dims) DG_HOST_ASSERT(dimension >= 0);
    const auto schema = [&](size_t nt, size_t nd) {
        DG_HOST_ASSERT(tensors.size() == nt && dims.size() == nd);
    };
    const auto expect = [&](size_t i, int64_t elements, torch::ScalarType dtype) {
        DG_HOST_ASSERT(tensors[i].numel() == elements && tensors[i].scalar_type() == dtype);
    };
    constexpr auto bf = torch::kBFloat16, i32 = torch::kInt32, i64 = torch::kInt64;
    int64_t work = 0, blocks = -1;
    int threads = 256;
    switch (static_cast<MegaMoEOperation>(operation)) {
        case MegaMoEOperation::PackExpertBatch:
        case MegaMoEOperation::GradientExpertBatch: {
            const bool grad=operation==int(MegaMoEOperation::GradientExpertBatch);
            const bool source_fp8=!grad && dims.size()==11;
            DG_HOST_ASSERT(((dims.size()==8 || dims.size()==9 || dims.size()==10) &&
                            tensors.size()==(grad?11:7)+(dims.size()==10?1:0)) ||
                           (source_fp8 && tensors.size()==10 && (dims[10]==1 || dims[10]==2)));
            if(dims.size()>=9)DG_HOST_ASSERT(dims[8]==16 || dims[8]==32 || dims[8]==64 || dims[8]==128);
            if(dims.size()>=10) {
                DG_HOST_ASSERT(dims[9]==32 || dims[9]==64 || dims[9]==128 || dims[9]==192 || dims[9]==256);
                const auto& metadata=tensors[grad?11:7];
                DG_HOST_ASSERT(metadata.scalar_type()==i32 && metadata.dim()==1 && metadata.numel()>=
                    std::max<int64_t>(128,4*(1+ceil_div(dims[0],dims[9])+tensors[grad?6:4].numel())));
            }
            DG_HOST_ASSERT(dims[0]>0 && dims[0]%128==0 && dims[1]>=128 && dims[1]<=8192
                && dims[1]%128==0 && dims[2]>0 && dims[3]>0 && dims[4]>0 && dims[4]<=64
                && dims[5]<dims[6] && dims[6]<=tensors[grad?6:4].numel()
                && dims[7]>0 && dims[7]<=16*get_num_sms() && dims[0]*dims[1]<=INT32_MAX);
            expect(0,dims[4],i64);expect(grad?3:1,dims[0],i32);
            expect(grad?5:2,dims[0]*dims[1],bf);expect(grad?7:3,1,i32);
            DG_HOST_ASSERT(tensors[grad?6:4].dim()==1 && tensors[grad?6:4].scalar_type()==i32);
            expect(grad?8:5,dims[0]*dims[1],torch::kFloat8_e4m3fn);
            expect(grad?9:6,dims[0]*(dims[1]/128),torch::kFloat32);
            if(grad) {
                expect(1,dims[4],i64);expect(2,dims[4],i64);expect(4,dims[0]*dims[1],bf);
                expect(10,dims[0]*(dims[1]/128),torch::kFloat32);
            }
            DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[grad?5:2].data_ptr())%16==0
                && reinterpret_cast<uintptr_t>(tensors[grad?8:5].data_ptr())%8==0);
            if(source_fp8) {expect(8,dims[4],i64);expect(9,dims[4],i64);}
            blocks=dims[7];break;
        }

        case MegaMoEOperation::ExpertLocalGradientsQuantizedTilesCached:
        case MegaMoEOperation::ExpertLocalGradientsQuantizedTiles:
            schema(11,6);
            DG_HOST_ASSERT(dims[0]>0 && dims[0]%128==0 && dims[1]>0 && dims[1]%128==0 &&
                           dims[2]>0 && dims[3]>0 && dims[4]>0 && dims[4]<=64 && dims[5]==128);
            expect(0,dims[4],i64);expect(1,dims[4],i64);expect(2,dims[4],i64);
            expect(3,dims[0],i32);expect(4,dims[0]*dims[1],bf);expect(5,dims[0]*dims[1],bf);
            expect(6,1,i32);expect(7,1,i32);expect(8,dims[0]*dims[1],torch::kFloat8_e4m3fn);
            expect(9,dims[0]*(dims[1]/128),torch::kFloat32);
            expect(10,dims[0]*(dims[1]/128),torch::kFloat32);
            DG_HOST_ASSERT(dims[0]*dims[1]<=INT32_MAX &&
                reinterpret_cast<uintptr_t>(tensors[4].data_ptr())%16==0 &&
                reinterpret_cast<uintptr_t>(tensors[5].data_ptr())%16==0 &&
                reinterpret_cast<uintptr_t>(tensors[8].data_ptr())%8==0);
            blocks=int64_t(get_num_sms())*4;
            break;
        case MegaMoEOperation::ReturnGemmTiles:
            schema(6,6);
            DG_HOST_ASSERT(dims[0]>0 && dims[0]%128==0 && dims[1]>0 && dims[1]%256==0 &&
                           dims[2]>0 && dims[3]>0 && dims[3]<=64 && dims[4]==256 &&
                           dims[5]>0 && dims[5]<get_num_sms());
            expect(0,dims[0]*dims[1],bf);expect(1,dims[0],i32);expect(2,dims[3],i64);
            expect(3,1,i32);expect(4,(dims[0]/128)*(dims[1]/256),i32);expect(5,1,i32);
            DG_HOST_ASSERT(dims[0]*dims[1]<=INT32_MAX && reinterpret_cast<uintptr_t>(tensors[0].data_ptr())%16==0);
            blocks=dims[5];
            break;
        case MegaMoEOperation::PackQuantizedInputTiles:
            DG_HOST_ASSERT(tensors.size()==8 && (dims.size()==7 || dims.size()==8 || dims.size()==9));
            DG_HOST_ASSERT(dims.size()==7 || (dims[7]<=2 &&
                (dims[7]!=1 || dims[6]==get_num_sms()) &&
                (dims[7]!=2 || (dims[6]>=get_num_sms() && dims[6]<=4*get_num_sms() && dims[6]%get_num_sms()==0))));
            if(dims.size()==9)DG_HOST_ASSERT(dims[8]>=1 && dims[8]<=16 &&
                (dims[8]&(dims[8]-1))==0 && dims[1]%(128*dims[8])==0);
            DG_HOST_ASSERT(dims[0]>0 && dims[0]%128==0 && dims[1]>=128 && dims[1]<=8192 &&
                           dims[1]%128==0 && dims[2]>0 && dims[3]>0 && dims[4]>0 &&
                           dims[4]<=64 && dims[5]==128 && dims[6]>0 && dims[6]<=8*get_num_sms());
            expect(0,dims[4],i64);expect(1,dims[0],i32);expect(2,dims[0]*dims[1],bf);
            expect(3,1,i32);expect(4,1,i32);expect(5,dims[0]*dims[1],torch::kFloat8_e4m3fn);
            expect(6,dims[0]*(dims[1]/128),torch::kFloat32);expect(7,dims[0]/128,i32);
            DG_HOST_ASSERT(dims[0]*dims[1]<=INT32_MAX &&
                           reinterpret_cast<uintptr_t>(tensors[2].data_ptr())%16==0 &&
                           reinterpret_cast<uintptr_t>(tensors[5].data_ptr())%8==0);
            blocks=dims[6];
            break;
        case MegaMoEOperation::PackActiveQuantized:
            schema(7,6);
            DG_HOST_ASSERT(dims[0]>0 && dims[1]>0 && dims[1]%128==0 && dims[2]>0 && dims[3]>0 &&
                           dims[4]>0 && dims[4]<=64 && dims[5]==128 && dims[0]%128==0);
            expect(0,dims[4],i64);expect(1,dims[0],i32);expect(2,dims[0]*dims[1],bf);
            expect(3,1,i32);expect(4,1,i32);expect(5,dims[0]*dims[1],torch::kFloat8_e4m3fn);
            expect(6,dims[0]*(dims[1]/128),torch::kFloat32);
            DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[2].data_ptr())%16==0 &&
                           reinterpret_cast<uintptr_t>(tensors[5].data_ptr())%8==0);
            blocks=int64_t(get_num_sms())*4;
            break;
        case MegaMoEOperation::PrepareContextReplay:
            schema(11, 3);
            DG_HOST_ASSERT(dims[0] > 0 && dims[0] < 128 && dims[1] > 0 && dims[2] % dims[1] == 0);
            expect(0, dims[0] + 1, i32); expect(1, dims[0], i32); expect(2, 1, i32);
            expect(3, dims[0] + 1, i32); expect(4, dims[0] + 1, i32);
            expect(5, dims[0], i32); expect(6, 1, i32); expect(7, 1, i32);
            expect(8, 1, i32); expect(9, 1, i32); expect(10, 1, i64);
            blocks = 1;
            break;
        case MegaMoEOperation::StageFp8InputTiles:
            // Three source pointer tables (FP8, row scales, original BF16),
            // global source rows, local FP8/MN scales/BF16, count, ready, status.
            DG_HOST_ASSERT(dims.size() >= 6 && dims.size() <= 8);
            schema(10, dims.size());
            DG_HOST_ASSERT(dims[0] > 0 && dims[0] % 128 == 0 && dims[1] > 0 && dims[1] % 128 == 0
                           && dims[2] > 0 && dims[3] > 0 && dims[3] <= 64
                           && dims[4] > 0 && dims[4] < get_num_sms() && dims[5] <= 1);
            expect(0, dims[3], i64); expect(1, dims[3], i64); expect(2, dims[3], i64);
            expect(3, dims[0], i32); expect(4, dims[0] * dims[1], torch::kFloat8_e4m3fn);
            expect(5, dims[0] * (dims[1] / 128), torch::kFloat32);
            expect(6, dims[0] * dims[1], bf); expect(7, 1, i32);
            expect(8, dims[0] / 128, i32); expect(9, 1, i32);
            DG_HOST_ASSERT(dims[0] * dims[1] <= INT32_MAX);
            if (dims.size() == 8) DG_HOST_ASSERT(dims[7] <= 1 && (!dims[7] || dims[1] <= 8192));
            threads = dims.size() >= 7 ? dims[6] : 128;
            DG_HOST_ASSERT(threads == 128 || threads == 256 || threads == 512);
            blocks = dims[4];
            break;
        case MegaMoEOperation::AssistInputTiles:
            schema(8, 7);
            DG_HOST_ASSERT(dims[0] > 0 && dims[1] > 0 && dims[1] % 8 == 0
                           && dims[2] > 0 && dims[3] > 0 && dims[3] <= 64
                           && (dims[4] == 128 || dims[4] == 256)
                           && dims[0] % dims[4] == 0 && dims[5] > 0 && dims[5] <= 156
                           && (dims[6] == 1 || dims[6] == 2 || dims[6] == 4 || dims[6] == 8));
            expect(0, dims[3], i64); expect(1, dims[0], i32);
            expect(2, dims[0] * dims[1], bf); expect(3, dims[0] / dims[4] * 4, i32);
            expect(4, 1, i32); expect(5, dims[0] / dims[4] + 1, i32);
            expect(6, 1, i32); expect(7, 1, i32);
            threads = 128;
            blocks = dims[5];
            break;
        case MegaMoEOperation::PrefetchRawDy:
            schema(5, 6);
            DG_HOST_ASSERT(dims[0] > 0 && dims[1] > 0 && dims[1] % 8 == 0
                           && dims[2] > 0 && dims[3] > 0 && dims[4] > 0 && dims[4] <= 64
                           && (dims[5] == 128 || dims[5] == 256));
            expect(0, dims[4], i64); expect(1, dims[0], i32);
            expect(2, dims[0] * dims[1], bf); expect(3, 1, i32); expect(4, 1, i32);
            threads = 128;
            blocks = int64_t(get_num_sms()) * 2;
            break;
        case MegaMoEOperation::SetRouteWindow:
            schema(8, 6);
            DG_HOST_ASSERT(dims[0] > 0 && dims[1] > 0 && dims[2] > 0 && dims[2] % 128 == 0
                           && dims[3] % 128 == 0 && dims[4] > 0 && dims[5] > 0);
            expect(0, dims[0] + 1, i32); expect(1, dims[1], i32);
            expect(2, dims[0] + 1, i32); expect(3, dims[0], i32);
            expect(4, dims[2], i32); expect(5, dims[2], i32); expect(6, 1, i32);
            expect(7, dims[2], i32);
            blocks = int64_t(get_num_sms()) * 2;
            break;
        case MegaMoEOperation::ReduceScorePartials:
            schema(5, 6);
            DG_HOST_ASSERT(dims[0] > 0 && dims[1] > 0 && dims[2] > 0 && dims[3] > 0
                           && dims[4] > 0 && dims[4] <= 64 && (dims[5] == 128 || dims[5] == 256));
            expect(0, dims[0] * dims[1], torch::kFloat32); expect(1, dims[0], i32);
            expect(2, 1, i32); expect(3, dims[4], i64); expect(4, 1, i32);
            blocks = int64_t(get_num_sms()) * 2;
            break;
        case MegaMoEOperation::StageRawWeightedDy:
            schema(8, 6);
            DG_HOST_ASSERT(dims[0] > 0 && dims[1] > 0 && dims[1] % 8 == 0
                           && dims[2] > 0 && dims[3] > 0 && dims[4] > 0 && dims[4] <= 64
                           && (dims[5] == 128 || dims[5] == 256) && dims[0] % dims[5] == 0);
            expect(0, dims[4], i64); expect(1, dims[4], i64); expect(2, dims[0], i32);
            expect(3, dims[0] * dims[1], bf); expect(4, dims[0] * dims[1], bf);
            expect(5, dims[0], torch::kFloat32); expect(6, 1, i32); expect(7, 1, i32);
            DG_HOST_ASSERT(!tensors[3].is_alias_of(tensors[4]));
            blocks = int64_t(get_num_sms()) * 2;
            break;
        case MegaMoEOperation::ExpertLocalGradientsQuantized:
        case MegaMoEOperation::ExpertLocalGradients: {
            DG_HOST_ASSERT(dims.size() == 5 || dims.size() == 6);
            const bool quantized = static_cast<MegaMoEOperation>(operation)==MegaMoEOperation::ExpertLocalGradientsQuantized;
            DG_HOST_ASSERT(quantized ? tensors.size()==10 : (tensors.size() == 8 || tensors.size() == 9));
            if (dims.size() == 6) DG_HOST_ASSERT(dims[5] == 128 || dims[5] == 256);
            DG_HOST_ASSERT(dims[1] > 0 && dims[1] % 8 == 0 && dims[2] > 0 && dims[3] > 0
                           && dims[4] > 0 && dims[4] <= 64);
            expect(0, dims[4], i64); expect(1, dims[4], i64); expect(2, dims[4], i64);
            expect(3, dims[0], i32); expect(4, dims[0] * dims[1], bf);
            expect(5, dims[0] * dims[1], bf); expect(6, 1, i32); expect(7, 1, i32);
            if (quantized) {
                DG_HOST_ASSERT(dims.size()==6 && dims[5]==128 && dims[1]%128==0);
                expect(8,dims[0]*dims[1],torch::kFloat8_e4m3fn);
                expect(9,dims[0]*(dims[1]/128),torch::kFloat32);
                DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[5].data_ptr())%16==0 &&
                               reinterpret_cast<uintptr_t>(tensors[8].data_ptr())%8==0);
            } else if (tensors.size() == 9) expect(8, dims[0] * dims[1], bf);
            blocks = int64_t(get_num_sms()) * 2;
            break;
        }
        case MegaMoEOperation::CountRoutes:
            schema(5, 5);
            DG_HOST_ASSERT(dims[1] > 0 && dims[3] > 0 && dims[3] <= 64 && dims[2] < dims[3]);
            DG_HOST_ASSERT(dims[4] > 0 && dims[0] == dims[3] * dims[4]);
            expect(0, dims[3], i64); expect(1, dims[1], i32);
            expect(2, dims[0], i32); expect(3, dims[0], i32); expect(4, 1, i32);
            work = dims[0];
            break;
        case MegaMoEOperation::InitializeLayout:
            schema(2, 2);
            expect(0, dims[0], i32); expect(1, dims[0], i32);
            work = dims[0];
            break;
        case MegaMoEOperation::BuildPermutation:
            schema(6, 3);
            DG_HOST_ASSERT(dims[2] > 0 && tensors[2].numel() > 1);
            expect(0, dims[0], i32); expect(1, dims[0], i32);
            DG_HOST_ASSERT(tensors[2].scalar_type() == i32);
            expect(3, dims[1], i32); expect(4, dims[1], i32); expect(5, 1, i32);
            work = dims[0];
            break;
        case MegaMoEOperation::AggregateFirstWgradFlags:
            DG_HOST_ASSERT(dims.size()==2 && dims[0]>0 && dims[0]<=512 && dims[1]>0 && dims[1]<=8);
            DG_HOST_ASSERT(tensors.size()==size_t(2+dims[1]));
            expect(0,dims[0],i32);expect(1,ceil_div(dims[1],int64_t(2))*dims[0],i32);
            DG_HOST_ASSERT(!tensors[0].is_alias_of(tensors[1]));
            for(int part=0;part<dims[1];++part)expect(2+part,dims[0]+1,i32);
            threads=128;blocks=ceil_div(dims[0],int64_t(threads));break;
        case MegaMoEOperation::FinalizeEmptyWgrad:
            schema(2,2);DG_HOST_ASSERT(dims[0]>0 && dims[0]<=512 && dims[1]>0 && dims[1]%4==0);
            expect(0,dims[0]*dims[1],torch::kFloat32);expect(1,dims[0],i32);
            DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[0].data_ptr())%16==0);
            blocks=std::max<int64_t>(dims[0],(16*get_num_sms()/dims[0])*dims[0]);break;
        case MegaMoEOperation::BuildGemmMetadata:
            schema(3,3);
            DG_HOST_ASSERT(dims[0]>0 && dims[0]<=64 && dims[1]>0 && dims[1]%128==0 && dims[1]<=INT32_MAX &&
                (dims[2]==32 || dims[2]==64 || dims[2]==128 || dims[2]==192 || dims[2]==256));
            expect(0,dims[0],i32);expect(2,1,i32);
            DG_HOST_ASSERT(tensors[1].scalar_type()==i32 && tensors[1].dim()==1 &&
                tensors[1].numel()>=std::max<int64_t>(128,4*(1+ceil_div(dims[1],dims[2])+dims[0])));
            DG_HOST_ASSERT(!tensors[0].is_alias_of(tensors[1]) && !tensors[2].is_alias_of(tensors[1]) && !tensors[0].is_alias_of(tensors[2]));
            blocks=1;break;
        case MegaMoEOperation::BuildWorkInfo:
            DG_HOST_ASSERT(dims.size()==3 && (tensors.size()==4 || tensors.size()==6));
            if(tensors.size()==6) {
                expect(4,dims[0],i32);expect(5,dims[0],i32);
                DG_HOST_ASSERT(!tensors[4].is_alias_of(tensors[5]));
            }
            DG_HOST_ASSERT(dims[0] > 0 && dims[2] > 0 && dims[1] % dims[2] == 0);
            expect(0, dims[0] + 1, i32); expect(1, dims[0], i32);
            expect(2, (dims[1] / dims[2]) * 4, i32); expect(3, 1, i32);
            work = dims[1] / dims[2];
            break;
        case MegaMoEOperation::Pack:
            schema(4, 5);
            DG_HOST_ASSERT(dims[1] > 0 && dims[2] > 0 && dims[3] > 0 && dims[4] > 0 && dims[4] <= 64);
            expect(0, dims[4], i64); expect(1, dims[0], i32);
            expect(2, dims[0] * dims[1], bf); expect(3, 1, i32);
            work = dims[0] * dims[1];
            break;
        case MegaMoEOperation::Barrier:
            schema(3, 4);
            DG_HOST_ASSERT(dims[0] > 0 && dims[0] <= 64 && dims[1] < dims[0] && dims[2] > 0);
            expect(0, dims[0], i64); expect(1, 4, i32); expect(2, 1, i32);
            blocks = 1;
            break;
        case MegaMoEOperation::Scatter:
        case MegaMoEOperation::ScatterVector:
            DG_HOST_ASSERT(dims.size() == 4 || dims.size() == 5);
            schema(5, dims.size());
            if (dims.size() == 5) DG_HOST_ASSERT(dims[4] == 128 || dims[4] == 256);
            DG_HOST_ASSERT(dims[1] > 0 && dims[2] > 0 && dims[3] > 0 && dims[3] <= 64);
            expect(0, dims[0] * dims[1], bf); expect(1, dims[0], i32);
            expect(2, dims[3], i64); expect(3, 1, i32);
            expect(4, 1, i32);
            if (static_cast<MegaMoEOperation>(operation) == MegaMoEOperation::ScatterVector)
                DG_HOST_ASSERT(dims[1] % 8 == 0);
            work = dims[0] * dims[1];
            blocks = std::min((work + 255) / 256, int64_t(get_num_sms()) * 2);
            break;
        case MegaMoEOperation::PackActive:
        case MegaMoEOperation::PackActiveVector:
            DG_HOST_ASSERT(dims.size() == 5 || dims.size() == 6);
            schema(5, dims.size());
            if (dims.size() == 6) DG_HOST_ASSERT(dims[5] == 128 || dims[5] == 256);
            DG_HOST_ASSERT(dims[1] > 0 && dims[2] > 0 && dims[3] > 0 && dims[4] > 0 && dims[4] <= 64);
            expect(0, dims[4], i64); expect(1, dims[0], i32);
            expect(2, dims[0] * dims[1], bf); expect(3, 1, i32); expect(4, 1, i32);
            if (static_cast<MegaMoEOperation>(operation) == MegaMoEOperation::PackActiveVector)
                DG_HOST_ASSERT(dims[1] % 8 == 0);
            work = dims[0] * dims[1];
            blocks = std::min((work + 255) / 256, int64_t(get_num_sms()) * 2);
            break;
        case MegaMoEOperation::ActivationActive:
        case MegaMoEOperation::ActivationBackwardActive:
            DG_HOST_ASSERT(dims.size() >= 2 && dims.size() <= 4);
            schema(4, dims.size());
            if (dims.size() >= 3) DG_HOST_ASSERT(dims[2] == 128 || dims[2] == 256);
            if (dims.size() == 4) DG_HOST_ASSERT(dims[3] == 8);
            DG_HOST_ASSERT(dims[0] >= 0 && dims[1] > 0);
            expect(0, dims[0] * 2 * dims[1], bf); expect(1, dims[0] * dims[1], bf);
            expect(2, 1, i32); expect(3, 1, i32);
            blocks = int64_t(get_num_sms()) * 4;
            break;
        case MegaMoEOperation::LocalGradients:
            schema(5, 5);
            DG_HOST_ASSERT(dims[0] > 0 && dims[1] > 0 && dims[2] > 0);
            expect(0, dims[0] * dims[2] * dims[1], bf);
            expect(1, dims[3] * dims[1], bf); expect(2, dims[3] * dims[2], torch::kFloat);
            expect(3, dims[3] * dims[2], torch::kFloat); expect(4, dims[0] * dims[2] * dims[1], bf);
            blocks = dims[0] * dims[2];
            break;
    }
    if(operation==int(MegaMoEOperation::PackActiveVector) || operation==int(MegaMoEOperation::ExpertLocalGradients) ||
       operation==int(MegaMoEOperation::StageRawWeightedDy)) {
        const int ctas=get_env<int>("DG_MOE_PEER_STAGE_CTAS_PER_CU",0);
        DG_HOST_ASSERT(ctas==0 || ctas==1 || ctas==2 || ctas==4 || ctas==8 || ctas==16);
        if(ctas)blocks=int64_t(get_num_sms())*ctas;
    }
    if(operation==int(MegaMoEOperation::PackActiveQuantized) || operation==int(MegaMoEOperation::ExpertLocalGradientsQuantized) || operation==int(MegaMoEOperation::ExpertLocalGradientsQuantizedTiles) || operation==int(MegaMoEOperation::ExpertLocalGradientsQuantizedTilesCached)) {
        const int ctas=get_env<int>("DG_MOE_FP8_FUSED_QUANT_CTAS_PER_CU",0);
        DG_HOST_ASSERT(ctas==0 || ctas==1 || ctas==2 || ctas==4 || ctas==8 || ctas==16);
        if(ctas)blocks=int64_t(get_num_sms())*ctas;
    }
    if (blocks < 0) blocks = (work + 255) / 256;
    if (!blocks) return;
    DG_HOST_ASSERT(blocks <= INT32_MAX);
    Params params{};
    for (size_t i = 0; i < tensors.size(); ++i) params.ptr[i] = tensors[i].data_ptr();
    for (size_t i = 0; i < dims.size(); ++i) params.dim[i] = dims[i];
    MegaMoERuntime::Args args{operation, {dim3(blocks), dim3(threads), 0}, params};
    const auto code = MegaMoERuntime::generate(args);
    const auto runtime = compiler->build(fmt::format("mega_moe_training_{}", operation), code);
    if(operation==int(MegaMoEOperation::PackQuantizedInputTiles) && dims.size()>=8 && dims[7]) {
        int occupancy=0;
        DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(
            &occupancy,runtime->kernel,threads,0));
        TORCH_CHECK(occupancy>=2,"co-resident input producer must permit at least two resident CTAs per CU");
    }
    if (!compile_only) MegaMoERuntime::launch(runtime, args);
}

static int mega_moe_input_occupancy(const int device,const std::vector<int64_t>& dims) {
    const c10::cuda::CUDAGuard guard(device);
    DG_HOST_ASSERT((dims.size()==7 || dims.size()==8 || dims.size()==9) && dims[1]>=128 && dims[1]<=8192 &&
                   dims[1]%128==0 && dims[2]>0 && dims[3]>0 && dims[4]>0 && dims[4]<=64);
    if(dims.size()==9)DG_HOST_ASSERT(dims[8]>=1 && dims[8]<=16 &&
        (dims[8]&(dims[8]-1))==0 && dims[1]%(128*dims[8])==0);
    Params params{};
    for(size_t i=0;i<dims.size();++i)params.dim[i]=dims[i];
    MegaMoERuntime::Args args{int(MegaMoEOperation::PackQuantizedInputTiles),{dim3(get_num_sms()),dim3(256),0},params};
    const auto runtime=compiler->build("mega_moe_training_25",MegaMoERuntime::generate(args));
    int occupancy=0;
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,runtime->kernel,256,0));
    return occupancy;
}

class MegaMoEGemmRuntime final : public LaunchRuntime<MegaMoEGemmRuntime> {
public:
    struct Args {
        int n, k, groups;
        bool read_peer, write_peer;
        bool activation_backward;
        int dispatch_cache_n;
        bool dispatch_cache_aiu;
        bool staged_return;
        int block_n, block_k, warp_m, warp_n, stages, threads;
        int dispatch_cache_ctas;
        int first_n_tile, num_n_tiles;
        bool local_input_aiu;
        bool producer_input_aiu;
        LaunchArgs launch_args;
        MegaMoEGemmParams params;
        MegaMoEDispatchPipeline pipeline{};
        bool cache_return_addresses = false;
        int block_m = 128;
        int return_vector_bytes = 4;
        int return_ncp_mode = 0;
        bool interleaved_gate_up = false;
        bool fc1_activation = false;
        bool keep_gate_up = true;
        int input_copy_ilp = 1;
        bool external_input = false;
        int poll_yield_every = 0;
        int poll_pause_cycles = 0;
        bool replay_gradients = false;
        bool replay_warp_reduce = false;
        MegaMoEReplayGradients gradients{};
        bool dhidden_scores = false;
        MegaMoEDHiddenScores scores{};
        bool weight_kn = false;
        int fc1_helper_threads = 0;
        int metadata_block_m = 0;
    };
    static std::string name(const Args& args) {
        auto name = fmt::format("bf16_peer_r{}_w{}_n{}_k{}_bn{}_bk{}_wm{}_wn{}_s{}",
                           args.read_peer, args.write_peer, args.n, args.k, args.block_n,
                           args.block_k, args.warp_m, args.warp_n, args.stages);
        if (args.activation_backward) name += "_activation_backward";
        if (args.dispatch_cache_n) name += fmt::format("_cache{}", args.dispatch_cache_n);
        if (args.dispatch_cache_aiu) name += "_aiu";
        if (args.staged_return) name += "_staged_return";
        if (args.dispatch_cache_n && args.dispatch_cache_ctas != 2)
            name += fmt::format("_ctas{}", args.dispatch_cache_ctas);
        if (args.first_n_tile || args.num_n_tiles)
            name += fmt::format("_nslice{}_{}", args.first_n_tile, args.num_n_tiles);
        if (args.local_input_aiu) name += "_local_aiu";
        if (args.producer_input_aiu) name += args.num_n_tiles==0 ? "_producer_owned" : "_producer_aiu";
        if (args.pipeline.ready) name += args.write_peer ? "_return_pipeline" : "_dispatch_pipeline";
        if (args.pipeline.ready && args.pipeline.serialize) name += "_serialized";
        if (args.poll_yield_every) name += fmt::format("_yield{}", args.poll_yield_every);
        if (args.poll_pause_cycles) name += fmt::format("_pause{}", args.poll_pause_cycles);
        if (args.replay_gradients) name += "_replay_grad";
        if (args.dhidden_scores) name += "_dhidden_scores";
        if (args.weight_kn) name += "_weight_kn";
        if (args.fc1_helper_threads) name += fmt::format("_input_helpers{}", args.fc1_helper_threads);
        if (args.metadata_block_m && args.metadata_block_m != args.block_m)
            name += fmt::format("_meta_bm{}", args.metadata_block_m);
        if (args.replay_warp_reduce) name += "_warp_reduce";
        if (args.pipeline.ready && !args.write_peer && args.pipeline.communication_ctas == 0)
            name += args.external_input ? "_external_dispatch" : "_owned_dispatch";
        if (args.cache_return_addresses) name += "_cached_return";
        if (args.block_m != 128) name += fmt::format("_bm{}", args.block_m);
        if (args.return_vector_bytes != 4) name += fmt::format("_return{}", args.return_vector_bytes);
        if(args.return_ncp_mode)name+=fmt::format("_ncp{}",args.return_ncp_mode);
        if (args.interleaved_gate_up) name += "_g8";
        if (args.fc1_activation) name += args.keep_gate_up ? "_swiglu_keep" : "_swiglu_discard";
        if (args.input_copy_ilp != 1) name += fmt::format("_copy_ilp{}", args.input_copy_ilp);
        return name;
    }
    static std::string generate_impl(const Args& args) {
        if (args.fc1_helper_threads) {
            return fmt::format(R"(
#include <deep_gemm/impls/mega_moe_training/mega_moe_gemm.cuh>
extern "C" __launch_bounds__({}, 1)
__global__ void {}(const deep_gemm::mega_moe_training::MegaMoEGemmParams params) {{
    deep_gemm::training_bf16_input_helpers_impl<
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    >(params);
}}
)", args.threads + args.fc1_helper_threads, name(args),
            args.n, args.k, args.groups, args.block_m, args.block_n, args.block_k,
            args.warp_m, args.warp_n, args.threads, args.stages,
            args.fc1_helper_threads, args.keep_gate_up ? "true" : "false", args.input_copy_ilp,
            args.metadata_block_m ? args.metadata_block_m : args.block_m);
        }
        return fmt::format("#define DG_BF16_RETURN_NCP_MODE {}\n",args.return_ncp_mode)+fmt::format(R"(
#include <deep_gemm/impls/mega_moe_training/mega_moe_gemm.cuh>
extern "C" __launch_bounds__({}, 1)
__global__ void {}(const deep_gemm::mega_moe_training::MegaMoEGemmParams params{}) {{
    deep_gemm::training_bf16_peer_gemm_impl<
        cutlass::bfloat16_t, deep_gemm::GemmType::MGroupedFused,
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, false, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, false, {}, {}, {}, {}, {}, {}, {}, {}{}
    >(params{});
}}
)", args.threads, name(args),
       args.pipeline.ready ? ", const deep_gemm::mega_moe_training::MegaMoEDispatchPipeline pipeline"
                           : args.replay_gradients ? ", const deep_gemm::mega_moe_training::MegaMoEReplayGradients gradients"
                           : args.dhidden_scores ? ", const deep_gemm::mega_moe_training::MegaMoEDHiddenScores scores" : "",
       args.n, args.k, args.groups, args.block_m, args.block_n, args.block_k,
       args.warp_m, args.warp_n, args.threads, args.stages,
       args.read_peer ? "true" : "false", args.write_peer ? "true" : "false",
       args.activation_backward ? "true" : "false", args.dispatch_cache_n,
       args.dispatch_cache_aiu ? "true" : "false", args.staged_return ? "true" : "false",
       args.first_n_tile, args.num_n_tiles, args.local_input_aiu ? "true" : "false",
       args.producer_input_aiu ? "true" : "false",
       args.pipeline.ready ? "true" : "false",
       args.pipeline.ready && !args.write_peer && args.pipeline.communication_ctas == 0 && !args.external_input ? "true" : "false",
       args.cache_return_addresses ? "true" : "false",
       args.return_vector_bytes,
       args.interleaved_gate_up ? "true" : "false",
       args.fc1_activation ? "true" : "false", args.keep_gate_up ? "true" : "false",
       args.input_copy_ilp, args.external_input ? "true" : "false", args.poll_yield_every, args.poll_pause_cycles,
       args.weight_kn ? (args.dhidden_scores ? ", false, false, true, true" : ", false, false, false, true")
                           : args.dhidden_scores ? ", false, false, true"
                           : args.replay_gradients ? (args.replay_warp_reduce ? ", true, true" : ", true") : "",
       args.pipeline.ready ? ", pipeline"
                           : args.replay_gradients ? ", deep_gemm::mega_moe_training::MegaMoEDispatchPipeline{}, 0, ~0u, gradients"
                           : args.dhidden_scores ? ", deep_gemm::mega_moe_training::MegaMoEDispatchPipeline{}, 0, ~0u, deep_gemm::mega_moe_training::MegaMoEReplayGradients{}, scores" : "");
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        if (args.dhidden_scores)
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params, args.scores));
        else if (args.replay_gradients)
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params, args.gradients));
        else if (args.pipeline.ready)
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params, args.pipeline));
        else
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

static int mega_moe_gemm(const torch::Tensor& a, const torch::Tensor& weight, const torch::Tensor& out,
                      const torch::Tensor& source_peers, const torch::Tensor& destination_peers,
                      const torch::Tensor& input_rows, const torch::Tensor& permutation,
                      const torch::Tensor& work_info, const torch::Tensor& num_blocks,
                      int source_rows_per_rank, int destination_rows_per_rank,
                      bool read_peer, bool write_peer, bool compile_only,
                      bool pack_input, const std::vector<int>& config,
                      const c10::optional<torch::Tensor>& activation_gate_up,
                      int dispatch_cache_n, const c10::optional<torch::Tensor>& dispatch_scratch,
                      bool dispatch_cache_aiu, bool staged_return, int dispatch_cache_ctas,
                      int first_n_tile, int num_n_tiles, bool local_input_aiu, bool producer_input_aiu,
                      const c10::optional<torch::Tensor>& pipeline_ready,
                      const c10::optional<torch::Tensor>& pipeline_status,
                      const c10::optional<torch::Tensor>& pipeline_statistics,
                      int pipeline_comm_ctas, bool pipeline_serialize, bool cache_return_addresses,
                      int block_m = 128, int return_vector_bytes = 4,
                      bool interleaved_gate_up = false, bool fc1_activation = false, bool keep_gate_up = true,
                      int input_copy_ilp = 1, bool external_input = false, int compute_cus = 0,
                      const std::vector<torch::Tensor>& replay_gradient_tensors = {},
                      const std::vector<torch::Tensor>& dhidden_score_tensors = {}, bool weight_kn = false,
                      int fc1_helper_threads = 0, int fc1_compute_block_m = 0, int return_ncp_mode = 0, int return_local_rank = 0) {
    DG_HOST_ASSERT(return_ncp_mode==0 || return_ncp_mode==2);
    if(return_ncp_mode) {
        DG_HOST_ASSERT(write_peer && !read_peer && !pipeline_ready.has_value() && !activation_gate_up.has_value() &&
            return_vector_bytes==4 && !cache_return_addresses && !fc1_helper_threads && !fc1_activation);
        DG_HOST_ASSERT(return_local_rank>=0 && return_local_rank<destination_peers.numel());
        staged_return=true;
    }
    DG_HOST_ASSERT(fc1_helper_threads == 0 || fc1_helper_threads == 32 || fc1_helper_threads == 64);
    DG_HOST_ASSERT(fc1_compute_block_m == 0 || (fc1_helper_threads && (fc1_compute_block_m == 128 || fc1_compute_block_m == 256)));
    const int compute_block_m = fc1_compute_block_m ? fc1_compute_block_m : block_m;
    DG_HOST_ASSERT(compute_block_m <= block_m && block_m % compute_block_m == 0);
    DG_HOST_ASSERT(!fc1_helper_threads || (read_peer && !write_peer && pack_input
        && fc1_activation && interleaved_gate_up && !pipeline_ready.has_value()
        && !dispatch_cache_n && !staged_return && !local_input_aiu && !producer_input_aiu
        && !first_n_tile && !num_n_tiles && !weight_kn && !external_input
        && replay_gradient_tensors.empty() && dhidden_score_tensors.empty()));
    DG_HOST_ASSERT(!external_input || (pipeline_ready.has_value() && pipeline_comm_ctas == 0 && !read_peer && !write_peer));
    DG_HOST_ASSERT(input_copy_ilp == 1 || input_copy_ilp == 2 || input_copy_ilp == 4 || input_copy_ilp == 8);
    DG_HOST_ASSERT(input_copy_ilp == 1 || producer_input_aiu || fc1_helper_threads
                   || (pipeline_ready.has_value() && pipeline_comm_ctas == 0 && !write_peer));
    DG_HOST_ASSERT(block_m == 128 || block_m == 256);
    DG_HOST_ASSERT(return_vector_bytes == 4 || return_vector_bytes == 8);
    DG_HOST_ASSERT(return_vector_bytes == 4 || (write_peer && !pipeline_ready.has_value() && !staged_return));
    DG_HOST_ASSERT(!fc1_activation || (interleaved_gate_up && !write_peer
                                     && (!pipeline_ready.has_value() || pipeline_comm_ctas == 0)
                                     && activation_gate_up.has_value()));
    DG_HOST_ASSERT(!interleaved_gate_up || activation_gate_up.has_value());
    DG_HOST_ASSERT(a.is_cuda() && a.dim() == 2 && weight.dim() == 3 && out.dim() == 2);
    const c10::cuda::CUDAGuard guard(a.device());
    DG_HOST_ASSERT(compute_cus >= 0 && compute_cus <= get_num_sms());
    for (const auto& tensor : {a, weight, out, source_peers, destination_peers, input_rows,
                               permutation, work_info, num_blocks})
        DG_HOST_ASSERT(tensor.device() == a.device() && tensor.is_contiguous());
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16 && weight.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(out.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(a.numel() <= INT32_MAX && weight.numel() <= INT32_MAX && out.numel() <= INT32_MAX);
    const auto [m, k] = get_shape<2>(a);
    const auto [groups, weight_rows, weight_columns] = get_shape<3>(weight);
    const int n = weight_kn ? weight_columns : weight_rows;
    const int k_ = weight_kn ? weight_rows : weight_columns;
    DG_HOST_ASSERT(k == k_ && out.size(0) == m && out.size(1) == n && m > 0 && m % block_m == 0);
    DG_HOST_ASSERT(n > 0 && n % 128 == 0 && k > 0 && k % 128 == 0 && groups > 0);
    DG_HOST_ASSERT(source_peers.dim() == 1 && source_peers.numel() > 0 && source_peers.numel() <= 64);
    DG_HOST_ASSERT(source_peers.scalar_type() == torch::kInt64 && destination_peers.scalar_type() == torch::kInt64);
    DG_HOST_ASSERT(source_peers.sizes() == destination_peers.sizes());
    DG_HOST_ASSERT(input_rows.numel() == m && permutation.numel() == m);
    DG_HOST_ASSERT(input_rows.scalar_type() == torch::kInt32 && permutation.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(work_info.sizes() == at::IntArrayRef({m / block_m, 4}) && work_info.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(num_blocks.numel() == 1 && num_blocks.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(source_rows_per_rank > 0 && destination_rows_per_rank > 0);
    DG_HOST_ASSERT(!pack_input || read_peer);
    DG_HOST_ASSERT(block_m == 128 || ((!pipeline_ready.has_value() || (pipeline_comm_ctas == 0 && !write_peer))
                                     && !dispatch_cache_n && (!staged_return || return_ncp_mode==2)));
    DG_HOST_ASSERT(get_ppu_arch() == AC_PPU0015);
    const int ranks = source_peers.numel();
    DG_HOST_ASSERT(int64_t(ranks) * source_rows_per_rank <= INT32_MAX);
    MegaMoEGemmParams params{};
    params.gemm.a_ptr = a.data_ptr();
    params.gemm.b_ptr = weight.data_ptr();
    params.gemm.c_ptr = out.data_ptr();
    params.gemm.expert_ids_and_cumsum = work_info.data_ptr<int32_t>();
    params.gemm.sorted_token_ids = input_rows.data_ptr<int32_t>();
    params.gemm.aligned_num_m_blocks = num_blocks.data_ptr<int32_t>();
    params.gemm.shape_m = read_peer ? ranks * source_rows_per_rank : m;
    params.source_peers = reinterpret_cast<uint64_t*>(source_peers.data_ptr<int64_t>());
    params.destination_peers = reinterpret_cast<uint64_t*>(destination_peers.data_ptr<int64_t>());
    params.permutation = permutation.data_ptr<int32_t>();
    params.source_rows_per_rank = source_rows_per_rank;
    params.destination_rows_per_rank = destination_rows_per_rank;
    params.num_ranks = ranks;
    params.pack_input = pack_input;
    DG_HOST_ASSERT(dispatch_cache_n == 0 || dispatch_cache_n == 2 || dispatch_cache_n == 4
                   || dispatch_cache_n == 8 || dispatch_cache_n == 16);
    if (!read_peer) dispatch_cache_n = 0;
    DG_HOST_ASSERT(!dispatch_cache_aiu || (read_peer && dispatch_cache_n));
    DG_HOST_ASSERT(!staged_return || write_peer);
    DG_HOST_ASSERT(dispatch_cache_ctas == 1 || dispatch_cache_ctas == 2 || dispatch_cache_ctas == 4);
    if (dispatch_cache_n) {
        DG_HOST_ASSERT(dispatch_scratch.has_value());
        const auto& scratch = dispatch_scratch.value();
        DG_HOST_ASSERT(scratch.is_contiguous() && scratch.device() == a.device()
                       && scratch.scalar_type() == torch::kBFloat16 && scratch.dim() == 2);
        DG_HOST_ASSERT(scratch.size(1) == k
                       && scratch.size(0) >= get_num_sms() * dispatch_cache_ctas * 128);
        DG_HOST_ASSERT(scratch.numel() <= INT32_MAX);
        params.dispatch_scratch = scratch.data_ptr();
    }
    if (activation_gate_up.has_value()) {
        const auto& activation = activation_gate_up.value();
        DG_HOST_ASSERT(!write_peer && activation.device() == a.device() && activation.is_contiguous());
        DG_HOST_ASSERT(activation.scalar_type() == torch::kBFloat16 && activation.dim() == 2);
        DG_HOST_ASSERT(activation.size(0) == m && activation.size(1) == (fc1_activation ? n / 2 : 2 * n));
        DG_HOST_ASSERT(!fc1_activation || n % 16 == 0);
        DG_HOST_ASSERT(activation.numel() <= INT32_MAX);
        params.activation_gate_up = activation.data_ptr();
    }
    DG_HOST_ASSERT(config.empty() || config.size() == 5);
    const int bn = config.empty() ? 128 : config[0];
    const int bk = config.empty() ? 64 : config[1];
    const int wm = config.empty() ? 64 : config[2];
    const int wn = config.empty() ? 64 : config[3];
    const int stages = config.empty() ? 3 : config[4];
    DG_HOST_ASSERT(return_vector_bytes == 4 || (wm == 64 && wn == 64 && n % 16 == 0));
    DG_HOST_ASSERT(bn == 64 || bn == 128 || bn == 256 || (fc1_helper_threads && bn == 512 && compute_block_m == 128));
    DG_HOST_ASSERT(bk == 64 || bk == 128);
    DG_HOST_ASSERT(wm == 32 || wm == 64 || wm == 128);
    DG_HOST_ASSERT(wn == 32 || wn == 64 || wn == 128);
    DG_HOST_ASSERT(bn % wn == 0 && stages >= 2 && stages <= 4);
    DG_HOST_ASSERT(first_n_tile >= 0 && first_n_tile < (n + bn - 1) / bn && num_n_tiles >= 0);
    DG_HOST_ASSERT(num_n_tiles == 0 || first_n_tile + num_n_tiles <= (n + bn - 1) / bn);
    DG_HOST_ASSERT((first_n_tile == 0 && num_n_tiles == 0) || dispatch_cache_n == 0);
    DG_HOST_ASSERT(!local_input_aiu || (!read_peer && !write_peer && first_n_tile > 0));
    DG_HOST_ASSERT(!producer_input_aiu || (read_peer && !write_peer && pack_input
                   && first_n_tile == 0 && (num_n_tiles == 1 || num_n_tiles == 0) && dispatch_cache_n == 0));
    DG_HOST_ASSERT(!fc1_helper_threads || ((bn == 256 || bn == 512) && bk == 64
        && ((wm == 64 && wn == 64) || (wm == 32 && wn == 128)) && stages == 3));
    const int threads = (compute_block_m / wm) * (bn / wn) * 32;
    const int launch_threads = threads + fc1_helper_threads;
    DG_HOST_ASSERT(launch_threads <= 1024);
    DG_HOST_ASSERT(threads >= 32 && threads <= 1024);
    const int smem = gemm_smem_total_size(2, stages, compute_block_m, bn, bk);
    DG_HOST_ASSERT(smem <= 256 * 1024);
    if(return_ncp_mode)DG_HOST_ASSERT(bn==256 && n%256==0);
    params.return_local_rank=uint32_t(return_local_rank);
    MegaMoEGemmRuntime::Args args{n, k, groups, read_peer, write_peer, activation_gate_up.has_value(), dispatch_cache_n,
                              dispatch_cache_aiu,
                              staged_return,
                              bn, bk, wm, wn, stages, threads,
                              dispatch_cache_ctas,
                              first_n_tile, num_n_tiles,
                              local_input_aiu,
                              producer_input_aiu,
                              {get_grid_shape(get_num_sms()), dim3(launch_threads), smem}, params};
    DG_HOST_ASSERT(!cache_return_addresses || (write_peer && !staged_return && !pipeline_ready.has_value()));
    args.cache_return_addresses = cache_return_addresses;
    args.block_m = compute_block_m;
    if (fc1_helper_threads) args.metadata_block_m = block_m;
    args.return_vector_bytes = return_vector_bytes;
    args.return_ncp_mode=return_ncp_mode;
    args.activation_backward = activation_gate_up.has_value() && !fc1_activation;
    args.interleaved_gate_up = interleaved_gate_up;
    args.fc1_activation = fc1_activation;
    args.keep_gate_up = keep_gate_up;
    args.input_copy_ilp = input_copy_ilp;
    args.external_input = external_input;
    const bool dhidden_producer=dhidden_score_tensors.size()==6;
    const bool dhidden_local_slice=dhidden_score_tensors.size()==3 && first_n_tile==1 && num_n_tiles==0 && local_input_aiu;
    DG_HOST_ASSERT(!weight_kn || ((!read_peer && !fc1_activation && !pack_input
                   && !dispatch_cache_n && !pipeline_ready.has_value()
                   && replay_gradient_tensors.empty() && !local_input_aiu && !producer_input_aiu
                   && !first_n_tile && !num_n_tiles && (!staged_return || return_ncp_mode==2)) || dhidden_producer || dhidden_local_slice));
    args.weight_kn = weight_kn;
    args.fc1_helper_threads = fc1_helper_threads;
    if (fc1_helper_threads) args.params.gemm.shape_m = m;
    if (!dhidden_score_tensors.empty()) {
        const auto& g = dhidden_score_tensors;
        DG_HOST_ASSERT((g.size()==3 || g.size()==6) && replay_gradient_tensors.empty()
                       && args.activation_backward && interleaved_gate_up && !write_peer
                       && !pipeline_ready.has_value() && !dispatch_cache_n && !staged_return && !fc1_activation);
        DG_HOST_ASSERT(dhidden_producer ? (read_peer && pack_input && producer_input_aiu && !local_input_aiu && first_n_tile==0 && (num_n_tiles==1 || num_n_tiles==0))
                       : (!read_peer && !pack_input && !producer_input_aiu &&
                          ((!local_input_aiu && !first_n_tile && !num_n_tiles) || dhidden_local_slice)));
        for (const auto& tensor : g)
            DG_HOST_ASSERT(tensor.device() == a.device() && tensor.is_contiguous());
        const uint32_t ntiles = (n + bn - 1) / bn;
        DG_HOST_ASSERT(g[0].scalar_type() == torch::kBFloat16 && g[0].sizes() == out.sizes());
        DG_HOST_ASSERT(g[1].scalar_type() == torch::kFloat32 && g[1].sizes() == at::IntArrayRef({m}));
        DG_HOST_ASSERT(g[2].scalar_type() == torch::kFloat32
                       && g[2].sizes() == at::IntArrayRef({m, int64_t(ntiles)}));
        // out aliases the retained hidden tensor by design: this epilogue
        // writes gate/up gradients and score partials, never out/hidden.
        DG_HOST_ASSERT(!a.is_alias_of(g[0]) && !g[2].is_alias_of(g[1]));
        args.dhidden_scores = true;
        args.scores = {g[0].data_ptr(), g[1].data_ptr<float>(), g[2].data_ptr<float>(), ntiles};
        if(dhidden_producer) {
            DG_HOST_ASSERT(g[3].scalar_type()==torch::kInt64 && g[3].numel()==ranks);
            DG_HOST_ASSERT(g[4].scalar_type()==torch::kBFloat16 && g[4].sizes()==a.sizes() && !g[4].is_alias_of(a));
            DG_HOST_ASSERT(g[5].scalar_type()==torch::kInt32 && g[5].numel()==1);
            args.scores.score_peers=reinterpret_cast<const uint64_t*>(g[3].data_ptr<int64_t>());
            args.scores.packed_dy=g[4].data_ptr();args.scores.status=reinterpret_cast<uint32_t*>(g[5].data_ptr<int32_t>());
        }
    }
    if (!replay_gradient_tensors.empty()) {
        const auto& g = replay_gradient_tensors;
        DG_HOST_ASSERT(g.size() == 5 && !read_peer && !write_peer && !activation_gate_up.has_value()
                       && !pipeline_ready.has_value() && !dispatch_cache_n && !pack_input
                       && !local_input_aiu && !producer_input_aiu && !first_n_tile
                       && !num_n_tiles && !staged_return);
        for (const auto& tensor : g)
            DG_HOST_ASSERT(tensor.device() == a.device() && tensor.is_contiguous());
        for (int index : {0, 1})
            DG_HOST_ASSERT(g[index].scalar_type() == torch::kInt64 && g[index].dim() == 1
                           && g[index].numel() == source_peers.numel());
        DG_HOST_ASSERT(g[2].scalar_type() == torch::kBFloat16 && g[2].sizes() == out.sizes());
        const uint32_t ntiles = (n + bn - 1) / bn;
        DG_HOST_ASSERT(g[3].scalar_type() == torch::kFloat32
                       && g[3].sizes() == at::IntArrayRef({m, int64_t(ntiles)}));
        DG_HOST_ASSERT(g[4].scalar_type() == torch::kInt32 && g[4].numel() == 1);
        DG_HOST_ASSERT(destination_rows_per_rank % source_rows_per_rank == 0);
        args.replay_gradients = true;
        const int warp_reduce = get_env<int>("DG_REPLAY_GRAD_WARP_REDUCTION", 0);
        DG_HOST_ASSERT(warp_reduce == 0 || warp_reduce == 1);
        args.replay_warp_reduce = warp_reduce;
        args.gradients = {
            reinterpret_cast<const uint64_t*>(g[0].data_ptr<int64_t>()),
            reinterpret_cast<const uint64_t*>(g[1].data_ptr<int64_t>()),
            g[2].data_ptr(), g[3].data_ptr<float>(),
            reinterpret_cast<uint32_t*>(g[4].data_ptr<int32_t>()),
            uint32_t(source_rows_per_rank), uint32_t(destination_rows_per_rank / source_rows_per_rank),
            uint32_t(source_peers.numel()), ntiles};
    }
    if (pipeline_ready.has_value()) {
        args.poll_pause_cycles = get_env<int>("DG_INPUT_READY_PAUSE_CYCLES", 0);
        DG_HOST_ASSERT(args.poll_pause_cycles >= 0 && args.poll_pause_cycles <= 16384);
        args.poll_yield_every = get_env<int>("DG_INPUT_READY_YIELD_EVERY", 0);
        DG_HOST_ASSERT(args.poll_yield_every >= 0 && args.poll_yield_every <= 1024
                       && (args.poll_yield_every == 0 || (args.poll_yield_every & (args.poll_yield_every - 1)) == 0));
        DG_HOST_ASSERT(!read_peer && (!activation_gate_up.has_value() || fc1_activation)
                       && !dispatch_cache_n && !local_input_aiu && !producer_input_aiu
                       && !first_n_tile && !num_n_tiles && !pack_input);
        const auto& ready = pipeline_ready.value();
        DG_HOST_ASSERT(ready.device() == a.device() && ready.is_contiguous()
                       && ready.scalar_type() == torch::kInt32
                       && ready.numel() > (m / block_m) * (write_peer ? ((n + bn - 1) / bn) : 1)
                       && ready.numel() <= INT32_MAX);
        DG_HOST_ASSERT(pipeline_status.has_value());
        const auto& status = pipeline_status.value();
        DG_HOST_ASSERT(status.device() == a.device() && status.is_contiguous()
                       && status.scalar_type() == torch::kInt32 && status.numel() == 1);
        DG_HOST_ASSERT(get_num_sms() >= 2 && pipeline_comm_ctas >= 0);
        DG_HOST_ASSERT(pipeline_comm_ctas != 0 || (!write_peer && !pipeline_serialize));
        args.pipeline.ready = reinterpret_cast<uint32_t*>(ready.data_ptr<int32_t>());
        args.pipeline.status = reinterpret_cast<uint32_t*>(status.data_ptr<int32_t>());
        args.pipeline.communication_ctas = pipeline_comm_ctas;
        args.pipeline.signal_capacity = ready.numel() - 1;
        args.pipeline.serialize = pipeline_serialize;
        if (pipeline_statistics.has_value()) {
            const auto& statistics = pipeline_statistics.value();
            DG_HOST_ASSERT(statistics.device() == a.device() && statistics.is_contiguous()
                           && statistics.scalar_type() == torch::kInt64
                           && statistics.dim() == 2 && statistics.size(1) == 4);
            args.pipeline.statistics = reinterpret_cast<uint64_t*>(statistics.data_ptr<int64_t>());
        }
    } else {
        DG_HOST_ASSERT(!pipeline_status.has_value() && !pipeline_statistics.has_value());
    }
    const auto code = MegaMoEGemmRuntime::generate(args);
    const auto runtime = compiler->build(MegaMoEGemmRuntime::name(args), code);
    int occupancy = 0;
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, runtime->kernel, launch_threads, smem));
    DG_HOST_ASSERT(occupancy > 0);
    if (args.pipeline.ready) {
        // A CTA is not a CU: the actual kernel can have multiple resident
        // CTAs per CU. Fill that resident pool while reserving one CU's worth
        // of slots for framework progress. No grid-level oversubscription.
        int compute_cus = get_num_sms() - (pipeline_comm_ctas == 0 && !external_input ? 0 : 1);
        if (external_input) {
            // This caps the persistent compute grid; it does not pin CTAs to
            // particular CUs or promise hardware partitioning across streams.
            const int requested = get_env<int>("DG_EXTERNAL_INPUT_COMPUTE_CUS", 0);
            DG_HOST_ASSERT(requested >= 0 && requested < get_num_sms());
            if (requested) compute_cus = requested;
        }
        args.launch_args.grid_dim.x = compute_cus * occupancy;
        DG_HOST_ASSERT(pipeline_comm_ctas < args.launch_args.grid_dim.x);
        if (pipeline_statistics.has_value())
            DG_HOST_ASSERT(pipeline_statistics.value().numel() >= args.launch_args.grid_dim.x * 4);
    } else
        args.launch_args.grid_dim.x *= dispatch_cache_n ? std::min(occupancy, dispatch_cache_ctas) : occupancy;
    // The scheduler strides by gridDim.x, so reducing its persistent CTA
    // budget preserves complete work coverage. This is not CU affinity.
    if (compute_cus) {
        args.launch_args.grid_dim.x = std::min(args.launch_args.grid_dim.x,
                                              uint32_t(compute_cus * occupancy));
        if (args.pipeline.ready)
            DG_HOST_ASSERT(pipeline_comm_ctas < args.launch_args.grid_dim.x);
    }
    if (get_env<int>("show_log")) {
        int registers = 0, local_bytes = 0;
#if HGGCRT_VERSION >= 12080 and defined(DG_JIT_USE_RUNTIME_API)
        hggcFuncAttributes attributes;
        DG_HGGC_RUNTIME_CHECK(hggcFuncGetAttributes(&attributes, runtime->kernel));
        registers = attributes.numRegs;
        local_bytes = static_cast<int>(attributes.localSizeBytes);
#else
        DG_HGGC_DRIVER_CHECK(lazy_hgFuncGetAttribute(&registers, HG_FUNC_ATTRIBUTE_NUM_REGS, runtime->kernel));
        DG_HGGC_DRIVER_CHECK(lazy_hgFuncGetAttribute(&local_bytes, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, runtime->kernel));
#endif
        std::printf("[MegaMoETraining_BF16:]\n");
        std::printf("kernel:%s, vreg:%d, local_size_bytes:%d\n",
                    MegaMoEGemmRuntime::name(args).c_str(), registers, local_bytes);
        std::printf("groups:%d, n:%d, k:%d, read_peer:%d, write_peer:%d, capacity_rows:%d\n",
                    int(groups), int(n), int(k), read_peer, write_peer, int(m));
        std::printf("tile:[%d,%d,%d], warp:[%d,%d], stages:%d, threads:%d, "
                    "dynamic_smem_bytes:%d, blocks_per_cu:%d, grid:%u\n",
                    block_m, bn, bk, wm, wn, stages, launch_threads, smem, occupancy, args.launch_args.grid_dim.x);
    }
    if (!compile_only) MegaMoEGemmRuntime::launch(runtime, args);
    return int(args.launch_args.grid_dim.x);
}

static void register_mega_moe_apis(pybind11::module_& m) {
    m.def("_mega_moe_training_launch", &launch_mega_moe, py::arg("operation"), py::arg("tensors"),
          py::arg("dims"), py::arg("compile_only") = false);
    m.def("_mega_moe_training_input_occupancy", &mega_moe_input_occupancy,py::arg("device"),py::arg("dims"));
    m.def("_mega_moe_training_gemm", &mega_moe_gemm,
          py::arg("a"), py::arg("weight"), py::arg("out"), py::arg("source_peers"), py::arg("destination_peers"),
          py::arg("input_rows"), py::arg("permutation"), py::arg("work_info"), py::arg("num_blocks"),
          py::arg("source_rows_per_rank"), py::arg("destination_rows_per_rank"),
          py::arg("read_peer"), py::arg("write_peer"), py::arg("compile_only") = false,
          py::arg("pack_input") = false, py::arg("config") = std::vector<int>{},
          py::arg("activation_gate_up") = py::none(), py::arg("dispatch_cache_n") = 0,
          py::arg("dispatch_scratch") = py::none(), py::arg("dispatch_cache_aiu") = false,
          py::arg("staged_return") = false, py::arg("dispatch_cache_ctas") = 2,
          py::arg("first_n_tile") = 0, py::arg("num_n_tiles") = 0,
          py::arg("local_input_aiu") = false, py::arg("producer_input_aiu") = false,
          py::arg("pipeline_ready") = py::none(), py::arg("pipeline_status") = py::none(),
          py::arg("pipeline_statistics") = py::none(), py::arg("pipeline_comm_ctas") = 2,
          py::arg("pipeline_serialize") = false, py::arg("cache_return_addresses") = false,
          py::arg("block_m") = 128, py::arg("return_vector_bytes") = 4,
          py::arg("interleaved_gate_up") = false, py::arg("fc1_activation") = false,
          py::arg("keep_gate_up") = true, py::arg("input_copy_ilp") = 1,
          py::arg("external_input") = false, py::arg("compute_cus") = 0,
          py::arg("replay_gradient_tensors") = std::vector<torch::Tensor>{},
          py::arg("dhidden_score_tensors") = std::vector<torch::Tensor>{},
          py::arg("weight_kn") = false, py::arg("fc1_helper_threads") = 0,
          py::arg("fc1_compute_block_m") = 0, py::arg("return_ncp_mode") = 0, py::arg("return_local_rank") = 0);
}

}  // namespace deep_gemm::mega_moe_training


#include <deep_gemm/common/mega_moe_training.cuh>
namespace deep_gemm::mega_moe_training {
class MegaMoEDwBlock128Runtime final:public LaunchRuntime<MegaMoEDwBlock128Runtime> {
public:
    struct Args {
        bool gradient;
        LaunchArgs launch_args;
        Params params;
        MegaMoEDwBlock128Operands dw;
    };
    static auto compile_key(const Args& a) {return std::make_tuple(a.gradient,a.params.dim[1],a.params.dim[2],a.params.dim[3],a.params.dim[9],a.params.dim[8]);}
    static std::string name(const Args& a){return a.gradient?"mega_moe_training_dw_block128_grad":"mega_moe_training_dw_block128_input";}
    static std::string generate_impl(const Args& a) {
        return fmt::format(R"(
#include <deep_gemm/impls/mega_moe_training/mega_moe_dw_block128.cuh>
extern "C" __launch_bounds__({},1)
__global__ void {}(const deep_gemm::mega_moe_training::Params params,const deep_gemm::mega_moe_training::MegaMoEDwBlock128Operands dw) {{
    deep_gemm::mega_moe_training::mega_moe_dw_block128<{},{},{},{},{},{}>(params,dw);
}}
)",a.params.dim[8]==512?512:256,name(a),a.params.dim[1],a.params.dim[2],a.params.dim[3],a.params.dim[9],a.gradient,a.params.dim[8]);
    }
    static void launch_impl(const KernelHandle& kernel,const LaunchConfigHandle& config,Args a) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel,config,a.params,a.dw));
    }
};
static void mega_moe_dw_block128_launch(const std::vector<torch::Tensor>& tensors,const std::vector<int64_t>& dims,bool gradient,bool compile_only) {
    DG_HOST_ASSERT(dims.size()==(gradient?11:10) && tensors.size()==(gradient?17:10));
    DG_HOST_ASSERT(tensors[0].is_cuda());const c10::cuda::CUDAGuard guard(tensors[0].device());
    for(const auto& t:tensors)DG_HOST_ASSERT(t.device()==tensors[0].device() && t.is_contiguous());
    for(auto d:dims)DG_HOST_ASSERT(d>=0 && d<=INT32_MAX);
    const auto expect=[&](int index,int64_t elements,torch::ScalarType dtype){DG_HOST_ASSERT(tensors[index].numel()==elements && tensors[index].scalar_type()==dtype);};
    const auto r=dims[0],h=dims[1],tokens=dims[2],topk=dims[3],ranks=dims[4],first=dims[5],last=dims[6],ctas=dims[7],bm=dims[9];
    const auto bf=torch::kBFloat16,i32=torch::kInt32,i64=torch::kInt64,fp=torch::kFloat,f8=torch::kFloat8_e4m3fn;
    DG_HOST_ASSERT(r>0 && r%128==0 && h>=128 && h<=8192 && h%128==0 && tokens>0 && topk>0 && ranks>0 && ranks<=64 && first<last && ctas>0 && ctas<=16*get_num_sms() && (dims[8]==128 || dims[8]==512));
    DG_HOST_ASSERT(bm==32 || bm==64 || bm==128 || bm==192 || bm==256);
    const auto& lengths=tensors[gradient?6:4];DG_HOST_ASSERT(lengths.dim()==1 && lengths.scalar_type()==i32 && lengths.numel()<128 && last<=lengths.numel() && r*h<=INT32_MAX);
    expect(0,ranks,i64);expect(gradient?3:1,r,i32);expect(gradient?5:2,r*h,bf);expect(gradient?7:3,1,i32);
    expect(gradient?8:5,r*h,f8);expect(gradient?9:6,r*(h/128),fp);
    const auto& meta=tensors[gradient?11:7];DG_HOST_ASSERT(meta.dim()==1 && meta.scalar_type()==i32 && meta.numel()>=std::max<int64_t>(128,4*(1+ceil_div(r,bm)+lengths.numel())));
    expect(gradient?12:8,r*h,f8);expect(gradient?13:9,(r/128)*(gradient?h:h/128),fp);
    MegaMoEDwBlock128Operands dw{tensors[gradient?12:8].data_ptr(),tensors[gradient?13:9].data_ptr<float>()};
    if(gradient) {
        const auto i=dims[10];DG_HOST_ASSERT(i>0 && i%128==0 && r*i<=INT32_MAX);
        expect(1,ranks,i64);expect(2,ranks,i64);expect(4,r*h,bf);expect(10,r*(h/128),fp);
        expect(14,r*i,f8);expect(15,(i/128)*(r/128),fp);expect(16,r*i,bf);
        dw.quantized1=tensors[14].data_ptr();dw.scales1=tensors[15].data_ptr<float>();dw.hidden=tensors[16].data_ptr();dw.intermediate=i;
    }
    for(int index:{gradient?5:2,gradient?8:5,gradient?12:8})DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[index].data_ptr())%16==0);
    if(gradient)for(int index:{14,16})DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[index].data_ptr())%16==0);
    auto a=MegaMoEDwBlock128Runtime::Args{gradient,{dim3(ctas),dim3(dims[8]==512?512:256),0},{},dw};
    for(int j=0;j<(gradient?12:8);++j)a.params.ptr[j]=tensors[j].data_ptr();
    for(size_t j=0;j<dims.size();++j)a.params.dim[j]=dims[j];
    const auto runtime=compiler->build(MegaMoEDwBlock128Runtime::name(a),MegaMoEDwBlock128Runtime::generate(a));
    if(!compile_only)MegaMoEDwBlock128Runtime::launch(runtime,a);
}
static void register_mega_moe_dw_block128_apis(pybind11::module_& m) {
    m.def("_mega_moe_training_dw_block128",&mega_moe_dw_block128_launch,py::arg("tensors"),py::arg("dims"),py::arg("gradient")=false,py::arg("compile_only")=false);
}
}


#include <deep_gemm/common/mega_moe_training.cuh>
namespace deep_gemm::mega_moe_training {
class DwActivationRuntime final:public LaunchRuntime<DwActivationRuntime> {
public:
    struct Args {LaunchArgs launch_args;Params params;DwActivationOperands x;};
    static auto compile_key(const Args& a){return std::make_tuple(a.params.dim[1],a.params.dim[3],a.params.dim[4]);}
    static std::string name(const Args& a){return fmt::format("mega_moe_training_dw_activation_k{}",a.params.dim[4]);}
    static std::string generate_impl(const Args& a) {
        return fmt::format(R"(
#include <deep_gemm/impls/mega_moe_training/mega_moe_training_common.cuh>
extern "C" __launch_bounds__({},1)
__global__ void {}(const deep_gemm::mega_moe_training::Params params,const deep_gemm::mega_moe_training::DwActivationOperands x) {{
    deep_gemm::mega_moe_training::dw_activation_block<{},{},{}>(params,x);
}}
)",a.launch_args.block_dim.x,name(a),a.params.dim[1],a.params.dim[4],bool(a.params.dim[3]));
    }
    static void launch_impl(const KernelHandle& kernel,const LaunchConfigHandle& config,Args a){DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel,config,a.params,a.x));}
};
static void dw_activation_launch(const std::vector<torch::Tensor>& tensors,const std::vector<int64_t>& dims,bool compile_only) {
    DG_HOST_ASSERT(tensors.size()==13 && dims.size()==7 && tensors[0].is_cuda());
    const c10::cuda::CUDAGuard guard(tensors[0].device());
    for(const auto& t:tensors)DG_HOST_ASSERT(t.is_cuda() && t.device()==tensors[0].device() && t.is_contiguous());
    for(auto d:dims)DG_HOST_ASSERT(d>=0 && d<=INT32_MAX);
    const auto r=dims[0],i=dims[1],g=dims[2],mn=dims[3],rows=dims[4],h=dims[5],ctas=dims[6];
    DG_HOST_ASSERT(r>0 && r%128==0 && i>0 && i%128==0 && g>0 && g<128 && mn<=1 && (rows==128 || rows==512) && h>0 && h%128==0 && ctas>0 && ctas<=16*get_num_sms() && r*2*i<=INT32_MAX && r*h<=INT32_MAX);
    const auto bf=torch::kBFloat16,f8=torch::kFloat8_e4m3fn,fp=torch::kFloat,i32=torch::kInt32;
    const auto expect=[&](int index,int64_t elements,torch::ScalarType dtype){DG_HOST_ASSERT(tensors[index].numel()==elements && tensors[index].scalar_type()==dtype);};
    expect(0,r*2*i,bf);expect(1,r*i,bf);expect(2,r*2*i,f8);expect(3,r*(2*i/128),fp);
    expect(4,r*2*i,f8);expect(5,(r/128)*2*i,fp);expect(6,g,i32);expect(7,g+1,i32);expect(8,1,i32);expect(9,1,i32);
    expect(10,r*h,bf);expect(11,r*h,f8);expect(12,(h/128)*(r/128),fp);
    for(int index:{0,1,2,4,10,11})DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[index].data_ptr())%16==0);
    auto a=DwActivationRuntime::Args{{dim3(ctas),dim3(rows==512?512:256),0},{},{tensors[10].data_ptr(),tensors[11].data_ptr(),tensors[12].data_ptr<float>(),int32_t(h)}};
    for(int j=0;j<10;++j)a.params.ptr[j]=tensors[j].data_ptr();
    for(int j=0;j<7;++j)a.params.dim[j]=dims[j];
    const auto runtime=compiler->build(DwActivationRuntime::name(a),DwActivationRuntime::generate(a));
    if(!compile_only)DwActivationRuntime::launch(runtime,a);
}
static void register_dw_activation_apis(pybind11::module_& m){m.def("_mega_moe_training_activation_dw",&dw_activation_launch,py::arg("tensors"),py::arg("dims"),py::arg("compile_only")=false);}
}


namespace deep_gemm::mega_moe_training {
class StableRoutesRuntime final:public LaunchRuntime<StableRoutesRuntime> {
public:
    struct Args {int mode,block_routes;LaunchArgs launch_args;Params params;};
    static auto compile_key(const Args& a){return std::make_tuple(a.mode,a.block_routes,a.params.dim[1]);}
    static std::string name(const Args& a) {
        return fmt::format("mega_moe_training_stable_routes_{}_b{}",a.mode==0?"histogram":a.mode==1?"prefix":"finish",a.block_routes);
    }
    static std::string generate_impl(const Args& a) {
        return fmt::format(R"(
#include <deep_gemm/impls/mega_moe_training/mega_moe_training_common.cuh>
extern "C" __launch_bounds__(256,1)
__global__ void {}(const deep_gemm::mega_moe_training::Params params) {{
    deep_gemm::mega_moe_training::stable_routes<{},{},{}>(params);
}}
)",name(a),a.params.dim[1],a.mode,a.block_routes);
    }
    static void launch_impl(const KernelHandle& kernel,const LaunchConfigHandle& config,Args a){DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel,config,a.params));}
};
static void stable_routes_launch(const std::vector<torch::Tensor>& tensors,const std::vector<int64_t>& dims,
                                bool finish,bool compile_only,int block_routes,bool scan_prefix) {
    DG_HOST_ASSERT(tensors.size()==7 && dims.size()==5 && tensors[0].is_cuda());
    const c10::cuda::CUDAGuard guard(tensors[0].device());
    for(const auto& t:tensors)DG_HOST_ASSERT(t.is_cuda() && t.device()==tensors[0].device() && t.is_contiguous());
    for(auto d:dims)DG_HOST_ASSERT(d>=0 && d<=INT32_MAX);
    DG_HOST_ASSERT(block_routes==256 || block_routes==512 || block_routes==1024 || block_routes==2048);
    DG_HOST_ASSERT(!scan_prefix || finish);
    const auto r=dims[0],g=dims[1],rank=dims[2],ranks=dims[3],per_source=dims[4],blocks=(r+block_routes-1)/block_routes;
    DG_HOST_ASSERT(r>0 && g>0 && g<128 && rank<ranks && ranks>0 && ranks<=64 && per_source>0 && r==ranks*per_source);
    const auto expect=[&](int i,int64_t count,torch::ScalarType dtype){DG_HOST_ASSERT(tensors[i].numel()==count && tensors[i].scalar_type()==dtype);};
    expect(0,ranks,torch::kInt64);expect(1,g,torch::kInt32);expect(2,r,torch::kInt32);expect(3,r,torch::kInt32);expect(4,1,torch::kInt32);
    for(int i:{5,6}) {expect(i,blocks*g,torch::kInt32);DG_HOST_ASSERT(tensors[i].dim()==2 && tensors[i].size(0)==blocks && tensors[i].size(1)==g);}
    const auto phase=[&](int mode) {
        auto a=StableRoutesRuntime::Args{mode,block_routes,{dim3(mode==1?g:blocks),dim3(256),0},{}};
        for(int j=0;j<7;++j)a.params.ptr[j]=tensors[j].data_ptr();
        for(int j=0;j<5;++j)a.params.dim[j]=dims[j];
        const auto runtime=compiler->build(StableRoutesRuntime::name(a),StableRoutesRuntime::generate(a));
        if(!compile_only)StableRoutesRuntime::launch(runtime,a);
    };
    if(finish && scan_prefix)phase(1);
    phase(finish?2:0);
}
static void register_stable_routes_apis(pybind11::module_& m) {
    m.def("_mega_moe_training_stable_routes",&stable_routes_launch,py::arg("tensors"),py::arg("dims"),
          py::arg("finish")=false,py::arg("compile_only")=false,py::arg("block_routes")=256,py::arg("scan_prefix")=false);
}
}



namespace deep_gemm::mega_moe_training {
class Fp8ReuseRuntime final:public LaunchRuntime<Fp8ReuseRuntime> {
public:
    struct Args {int operation,width;LaunchArgs launch_args;Params params;};
    static auto compile_key(const Args& a){return std::make_tuple(a.operation,a.width);}
    static std::string name(const Args& a){return fmt::format("moe_fp8_reuse_{}_w{}",a.operation,a.width);}
    static std::string generate_impl(const Args& a) {
        return fmt::format(R"(
#include <deep_gemm/impls/mega_moe_training/mega_moe_training_common.cuh>
extern "C" __launch_bounds__(256,1)
__global__ void {}(const deep_gemm::mega_moe_training::Params params) {{
    deep_gemm::mega_moe_training::fp8_reuse<{},{}>(params);
}}
)",name(a),a.operation,a.width);
    }
    static void launch_impl(const KernelHandle& kernel,const LaunchConfigHandle& config,Args a){DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel,config,a.params));}
};

static void fp8_reuse_launch(int operation,const std::vector<torch::Tensor>& tensors,
                            const std::vector<int64_t>& dims,bool compile_only) {
    DG_HOST_ASSERT(operation>=0 && operation<=7 && !tensors.empty() && tensors[0].is_cuda());
    const c10::cuda::CUDAGuard guard(tensors[0].device());
    for(const auto& t:tensors)DG_HOST_ASSERT(t.is_cuda() && t.device()==tensors[0].device() && t.is_contiguous());
    for(auto d:dims)DG_HOST_ASSERT(d>=0 && d<=INT32_MAX);
    const auto expect=[&](int index,int64_t elements,torch::ScalarType dtype){DG_HOST_ASSERT(tensors[index].numel()==elements && tensors[index].scalar_type()==dtype);};
    const auto bf=torch::kBFloat16,fp=torch::kFloat,f8=torch::kFloat8_e4m3fn,i32=torch::kInt32,i64=torch::kInt64;
    int64_t width=0,blocks=0;
    if(operation<=3) {
        DG_HOST_ASSERT(dims.size()==2 && dims[0]>0);
        width=dims[1];DG_HOST_ASSERT(width>=128 && width<=8192 && width%128==0);
        const int64_t rows=dims[0],segments=width/128;
        if(operation==2) {
            DG_HOST_ASSERT(tensors.size()==2 && rows<=64);expect(0,rows,i64);expect(1,segments,fp);
            blocks=ceil_div<int64_t>(segments,256);
        } else {
            DG_HOST_ASSERT(rows*width<=INT32_MAX);
            if(operation==0) {
                DG_HOST_ASSERT(tensors.size()==2);expect(0,rows*width,bf);expect(1,rows*segments,fp);
                DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[0].data_ptr())%16==0);
            } else if(operation==1) {
                DG_HOST_ASSERT(tensors.size()==2);expect(0,rows*segments,fp);expect(1,segments,fp);
            } else {
                DG_HOST_ASSERT(tensors.size()==5);expect(0,rows*width,bf);expect(1,rows*segments,fp);
                expect(2,segments,fp);expect(3,rows*width,f8);expect(4,rows*segments,fp);
                DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[0].data_ptr())%16==0 &&
                               reinterpret_cast<uintptr_t>(tensors[3].data_ptr())%8==0);
            }
            blocks=operation==1?segments:std::min<int64_t>(ceil_div<int64_t>(rows*segments,16),get_num_sms()*4);
        }
    } else if(operation==4 || operation==7) {
        DG_HOST_ASSERT(tensors.size()==7 && dims.size()==5);
        const int64_t groups=dims[0],capacity=dims[1];width=dims[2];
        DG_HOST_ASSERT(groups>0 && groups<128 && capacity>0 && capacity%128==0 && width>=128 && width<=8192 &&
            width%128==0 && capacity*width<=INT32_MAX && dims[3]<=1 && dims[4]>0 && dims[4]<=16);
        expect(0,capacity*width,f8);expect(1,capacity*(width/128),fp);expect(2,capacity*width,f8);
        expect(3,(capacity/128)*(operation==7?width:width/128),fp);expect(4,groups,i32);expect(5,groups+1,i32);expect(6,1,i32);
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[0].data_ptr())%16==0 &&
                       reinterpret_cast<uintptr_t>(tensors[2].data_ptr())%16==0);
        blocks=std::min<int64_t>((capacity/128)*(width/128),get_num_sms()*dims[4]);
    } else {
        DG_HOST_ASSERT(tensors.size()==6 && dims.size()==4);
        const int64_t groups=dims[0],capacity=dims[1];width=dims[2];
        DG_HOST_ASSERT(groups>0 && groups<128 && capacity>0 && capacity%128==0 && width>=128 && width<=8192 &&
            width%128==0 && capacity*width<=INT32_MAX && dims[3]>0 && dims[3]<=16);
        expect(0,capacity*width,bf);expect(1,capacity*width,f8);
        expect(2,(capacity/128)*(operation==5?width:width/128),fp);
        expect(3,groups,i32);expect(4,groups+1,i32);expect(5,1,i32);
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[0].data_ptr())%16==0 &&
                       reinterpret_cast<uintptr_t>(tensors[1].data_ptr())%16==0);
        blocks=std::min<int64_t>((capacity/128)*(width/128),get_num_sms()*dims[3]);
    }
    auto a=Fp8ReuseRuntime::Args{operation,int(width),{dim3(blocks),dim3(256),0},{}};
    for(size_t j=0;j<tensors.size();++j)a.params.ptr[j]=tensors[j].data_ptr();
    for(size_t j=0;j<dims.size();++j)a.params.dim[j]=dims[j];
    const auto runtime=compiler->build(Fp8ReuseRuntime::name(a),Fp8ReuseRuntime::generate(a));
    if(!compile_only)Fp8ReuseRuntime::launch(runtime,a);
}
static void register_fp8_reuse_apis(pybind11::module_& m) {
    m.def("_mega_moe_training_fp8_reuse",&fp8_reuse_launch,py::arg("operation"),py::arg("tensors"),py::arg("dims"),py::arg("compile_only")=false);
}
}



namespace deep_gemm::mega_moe_training {
class Fp8GradientRuntime final:public LaunchRuntime<Fp8GradientRuntime> {
public:
    struct Args {int operation,width,threads;LaunchArgs launch_args;Params params;};
    static auto compile_key(const Args& a){return std::make_tuple(a.operation,a.width,a.threads);}
    static std::string name(const Args& a){return fmt::format("moe_fp8_gradient_{}_w{}_t{}",a.operation,a.width,a.threads);}
    static std::string generate_impl(const Args& a) {
        return fmt::format(R"(
#include <deep_gemm/impls/mega_moe_training/mega_moe_fp8_gradient.cuh>
extern "C" __launch_bounds__({0},1)
__global__ void {1}(const deep_gemm::mega_moe_training::Params params) {{
    deep_gemm::mega_moe_training::fp8_gradient<{2},{3},{0}>(params);
}}
)",a.threads,name(a),a.operation,a.width);
    }
    static void launch_impl(const KernelHandle& kernel,const LaunchConfigHandle& config,Args a){DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel,config,a.params));}
};

class Fp8ForwardTileSourceRuntime final:public LaunchRuntime<Fp8ForwardTileSourceRuntime> {
public:
    struct Args {int width,bm,threads;LaunchArgs launch_args;Params params;};
    static auto compile_key(const Args& a){return std::make_tuple(a.width,a.bm,a.threads);}
    static std::string name(const Args& a){return fmt::format("moe_fp8_forward_tile_source_w{}_m{}_t{}",a.width,a.bm,a.threads);}
    static std::string generate_impl(const Args& a){return fmt::format(R"(
#include <deep_gemm/impls/mega_moe_training/mega_moe_fp8_gradient.cuh>
extern "C" __launch_bounds__({0},1)
__global__ void {1}(const deep_gemm::mega_moe_training::Params params) {{
    extern __shared__ __align__(128) char arena[];
    deep_gemm::mega_moe_training::forward_fp8_tile_source<{2},{3}>(params,arena);
}}
)",a.threads,name(a),a.width,a.bm);}
    static void launch_impl(const KernelHandle& k,const LaunchConfigHandle& c,Args a){DG_HGGC_UNIFIED_CHECK(launch_kernel(k,c,a.params));}
};
static void fp8_forward_tile_source_launch(const std::vector<torch::Tensor>& tensors,
        const std::vector<int64_t>& dims,bool compile_only) {
    DG_HOST_ASSERT(tensors.size()==8 && dims.size()==11 && tensors[0].is_cuda());
    const c10::cuda::CUDAGuard guard(tensors[0].device());
    for(const auto& t:tensors)DG_HOST_ASSERT(t.is_cuda() && t.device()==tensors[0].device() && t.is_contiguous());
    for(auto d:dims)DG_HOST_ASSERT(d>=0 && d<=INT32_MAX);
    const int64_t rows=dims[0],width=dims[1],tokens=dims[2],topk=dims[3],ranks=dims[4],bm=dims[5],ctas=dims[6],threads=dims[7],groups=dims[10];
    DG_HOST_ASSERT(rows>0 && rows%128==0 && width>=512 && width<=8192 && width%512==0 && rows*width<=INT32_MAX);
    DG_HOST_ASSERT(tokens>0 && topk>0 && topk<=64 && ranks>0 && ranks<=64 && tokens*topk*ranks<=INT32_MAX);
    DG_HOST_ASSERT((bm==32 || bm==64 || bm==128 || bm==192 || bm==256) && ctas>0 && ctas<=get_num_sms() && (threads==128 || threads==256 || threads==512 || threads==1024));
    DG_HOST_ASSERT(dims[8]<ranks && groups>0 && groups<128 && dims[9]>=ceil_div(rows,bm)+groups+1);
    const auto expect=[&](int index,int64_t n,torch::ScalarType dt){DG_HOST_ASSERT(tensors[index].numel()==n && tensors[index].scalar_type()==dt);};
    expect(0,ranks,torch::kInt64);expect(1,ranks,torch::kInt64);expect(2,rows,torch::kInt32);
    DG_HOST_ASSERT(tensors[3].scalar_type()==torch::kInt32 && tensors[3].numel()>=4*(1+ceil_div(rows,bm)+groups));
    expect(4,rows*width,torch::kFloat8_e4m3fn);expect(5,rows*(width/128),torch::kFloat);
    expect(6,dims[9],torch::kInt32);expect(7,1,torch::kInt32);
    DG_HOST_ASSERT(!tensors[6].is_alias_of(tensors[7]));
    for(int index:{0,1,3,4,5})DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[index].data_ptr())%16==0);
    for(int index:{2,6,7})DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[index].data_ptr())%4==0);
    const int shared=ceil_div(int(bm*(20+8*(width/128+1))),128)*128+128;
    auto a=Fp8ForwardTileSourceRuntime::Args{int(width),int(bm),int(threads),{dim3(ctas),dim3(threads),shared},{}};
    for(size_t i=0;i<tensors.size();++i)a.params.ptr[i]=tensors[i].data_ptr();
    for(size_t i=0;i<dims.size();++i)a.params.dim[i]=dims[i];
    const auto runtime=compiler->build(Fp8ForwardTileSourceRuntime::name(a),Fp8ForwardTileSourceRuntime::generate(a));
    if(!compile_only)Fp8ForwardTileSourceRuntime::launch(runtime,a);
}

static void fp8_gradient_launch(int operation,const std::vector<torch::Tensor>& tensors,
                                const std::vector<int64_t>& dims,bool compile_only) {
    if(operation==34){fp8_forward_tile_source_launch(tensors,dims,compile_only);return;}
    DG_HOST_ASSERT(tensors.size()<=12 && dims.size()<=12);
    DG_HOST_ASSERT((operation>=0 && operation<=38) && !tensors.empty() && tensors[0].is_cuda());
    const c10::cuda::CUDAGuard guard(tensors[0].device());
    for(const auto& t:tensors)DG_HOST_ASSERT(t.is_cuda() && t.device()==tensors[0].device() && t.is_contiguous());
    for(auto d:dims)DG_HOST_ASSERT(d>=0 && d<=INT32_MAX);
    const auto expect=[&](int index,int64_t elements,torch::ScalarType dtype){DG_HOST_ASSERT(tensors[index].numel()==elements && tensors[index].scalar_type()==dtype);};
    const auto bf=torch::kBFloat16,fp=torch::kFloat,f8=torch::kFloat8_e4m3fn,i32=torch::kInt32,i64=torch::kInt64;
    int64_t width,rows,ctas,threads;
    const bool tiled_source=operation==5 || operation==11 || operation==13 || operation==17 || (operation>=20 && operation<=38);
    if(operation==16) {
        DG_HOST_ASSERT(tensors.size()==6 && (dims.size()==4 || dims.size()==5));
        const int64_t tokens=dims[0],bm=dims[2];width=dims[1];threads=dims[3];ctas=1;rows=tokens;
        DG_HOST_ASSERT(tokens>0 && width>=128 && width<=8192 && width%128==0 && tokens*width<=INT32_MAX);
        DG_HOST_ASSERT(bm==32 || bm==64 || bm==128 || bm==192 || bm==256);
        DG_HOST_ASSERT(ceil_div(tokens,int64_t(128))<=get_num_sms()*16);
        if(dims.size()==5)DG_HOST_ASSERT(dims[4]>0 && dims[4]<=std::min<int64_t>(ceil_div(tokens,int64_t(128)),get_num_sms()*16));
        expect(0,tokens*width,bf);expect(1,tokens*width,f8);expect(2,1,i32);
        expect(3,bm*(width/128),fp);expect(4,ceil_div(tokens,int64_t(128))+1,i32);expect(5,1,i32);
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[0].data_ptr())%16==0 && reinterpret_cast<uintptr_t>(tensors[1].data_ptr())%8==0);
    } else if(operation==14) {
        DG_HOST_ASSERT(tensors.size()==7 && dims.size()==6);
        const int64_t groups=dims[0],capacity=dims[1],bm=dims[3];width=dims[2];ctas=dims[4];threads=dims[5];rows=(capacity/128)*(width/128);
        DG_HOST_ASSERT(groups>0 && groups<128 && capacity>0 && capacity%128==0 && width>=128 && width<=8192 && width%128==0 && capacity*width<=INT32_MAX);
        DG_HOST_ASSERT(bm==32 || bm==64 || bm==128 || bm==192 || bm==256);
        expect(0,capacity*width,f8);expect(1,bm*(width/128),fp);expect(2,groups,i32);expect(3,groups+1,i32);
        expect(4,capacity*width,f8);expect(5,(capacity/128)*width,fp);expect(6,1,i32);
    } else if(operation==15) {
        DG_HOST_ASSERT(tensors.size()==3 && dims.size()==3);
        width=dims[0];const int64_t bm=dims[1];threads=dims[2];ctas=4;
        DG_HOST_ASSERT(width>=128 && width<=8192 && width%128==0 && (bm==32 || bm==64 || bm==128 || bm==192 || bm==256));
        rows=(width/128)*bm;expect(0,width/128,fp);expect(1,rows,fp);expect(2,1,i32);
    } else if(operation==8 || operation==12) {
        DG_HOST_ASSERT(tensors.size()==7 && dims.size()==5);
        const int64_t groups=dims[0],capacity=dims[1];width=dims[2];ctas=dims[3];threads=dims[4];rows=capacity*(width/128);
        DG_HOST_ASSERT(groups>0 && groups<128 && capacity>0 && capacity%128==0 && width>=128 && width<=8192 && width%128==0 && capacity*width<=INT32_MAX);
        expect(0,capacity*width,bf);expect(1,capacity,fp);expect(2,groups,i32);expect(3,groups+1,i32);
        expect(4,capacity*width,f8);expect(5,(capacity/128)*(width/128),fp);expect(6,1,i32);
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[0].data_ptr())%16==0 && reinterpret_cast<uintptr_t>(tensors[4].data_ptr())%16==0);
    } else if(operation==6) {
        DG_HOST_ASSERT(tensors.size()==2 && dims.size()==3);
        expect(0,dims[0],i32);expect(1,1,i32);
        DG_HOST_ASSERT(dims[0]>=2 && dims[1]>0 && dims[1]<=get_num_sms()*16 && dims[2]==128);
        rows=1;width=128;ctas=1;threads=128;
    } else if(operation==0 || operation==7 || (operation==10 || operation==18 || operation==19)) {
        DG_HOST_ASSERT(tensors.size()==(operation==7?11:10) && (dims.size()==6 || ((operation==10 || operation==18 || operation==19) && dims.size()==7)));
        if(dims.size()==7)DG_HOST_ASSERT(dims[6]>0 && dims[6]<=get_num_sms()*16);
        const int64_t tokens=dims[0],topk=dims[1];width=dims[2];rows=tokens*topk;
        DG_HOST_ASSERT(tokens>0 && topk>0 && topk<=64 && dims[3]>0 && dims[3]<=65536);
        ctas=dims[4];threads=dims[5];
        DG_HOST_ASSERT(width>=128 && width<=8192 && width%128==0 && rows*width<=INT32_MAX);
        expect(0,tokens*width,bf);expect(1,rows,fp);expect(2,rows*width,bf);
        expect(3,rows,i64);expect(4,1,i32);expect(5,tokens*width,f8);expect(6,tokens*width,f8);
        if(operation==7)expect(10,width/128,fp);
        expect(7,tokens*(width/128),fp);expect(8,rows*(width/128),fp);expect(9,1,i32);
        for(int i:{0,2})DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[i].data_ptr())%16==0);
        for(int i:{5,6})DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[i].data_ptr())%8==0);
    } else if(operation==1) {
        DG_HOST_ASSERT(tensors.size()==3 && dims.size()==4);
        rows=dims[0];width=dims[1];ctas=dims[2];threads=dims[3];
        DG_HOST_ASSERT(rows>0 && width>=128 && width<=8192 && width%128==0 && rows*(width/128)<=INT32_MAX);
        expect(0,rows*(width/128),fp);expect(1,rows,fp);expect(2,1,i32);
    } else if(operation==2 || operation==4 || operation==9 || tiled_source) {
        DG_HOST_ASSERT((operation==2 && ((tensors.size()==10 && dims.size()==9) || (tensors.size()==11 && dims.size()==10)))
            || ((operation==4 || operation==9) && tensors.size()==12 && dims.size()==10)
            || (tiled_source && tensors.size()==12 && (dims.size()==10 || dims.size()==11 || ((operation==30 || operation==31 || operation==32 || operation==33 || operation>=35) && dims.size()==12))));
        if(tiled_source && dims.size()>=11)DG_HOST_ASSERT(dims[10]>0 && dims[10]<=get_num_sms()*16);
        const int64_t capacity=dims[0],tokens=dims[2],topk=dims[3],ranks=dims[4],groups=tensors[4].numel();
        if(operation==30 || operation==31 || operation==32 || operation==33 || operation>=35)DG_HOST_ASSERT(dims.size()==12 && dims[11]>=0 && dims[11]<ranks);
        width=dims[1];rows=capacity*(width/128);ctas=dims[7];threads=dims[8];
        DG_HOST_ASSERT(capacity>0 && capacity%128==0 && width>=128 && width<=8192 && width%128==0 && capacity*width<=INT32_MAX);
        DG_HOST_ASSERT(tokens>0 && topk>0 && topk<=64 && ranks>0 && ranks<=64 && tokens*topk*ranks<=INT32_MAX);
        DG_HOST_ASSERT(groups>0 && groups<128 && dims[5]<dims[6] && dims[6]<=groups);
        expect(0,ranks,i64);
        if(operation==13) {
            DG_HOST_ASSERT(dims[9]==32 || dims[9]==64 || dims[9]==128 || dims[9]==192 || dims[9]==256);
            expect(1,dims[9]*(width/128),fp);
        } else expect(1,ranks,i64);
        expect(2,ranks,i64);expect(3,capacity,i32);expect(4,groups,i32);
        expect(5,capacity*width,f8);expect(6,capacity*(width/128),fp);expect(7,capacity*(width/128),fp);expect(8,capacity,fp);expect(9,1,i32);
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[5].data_ptr())%16==0);
        if(tiled_source) {
            DG_HOST_ASSERT(dims[9]==32 || dims[9]==64 || dims[9]==128 || dims[9]==192 || dims[9]==256);
            DG_HOST_ASSERT(tensors[11].scalar_type()==i32 && tensors[11].numel()>=ceil_div(capacity,dims[9])+groups+1);
            DG_HOST_ASSERT(dims[5]==0 && dims[6]==groups);

        }
        if(operation==4 || operation==9) {
            expect(10,capacity*width,bf);
            DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[10].data_ptr())%16==0);
        }
        if(dims.size()>=10) {
            const int64_t bm=dims[9];
            DG_HOST_ASSERT(bm==32 || bm==64 || bm==128 || bm==192 || bm==256);
            const auto& meta=tensors[(operation==4 || operation==9)?11:10];
            DG_HOST_ASSERT(meta.scalar_type()==i32 && meta.numel()>=4*(1+(capacity+bm-1)/bm+groups));
        }
    } else {
        DG_HOST_ASSERT(tensors.size()==10 && dims.size()==8);
        const int64_t groups=dims[0],capacity=dims[1],tokens=dims[3],topk=dims[4],ranks=dims[5];
        width=dims[2];rows=(capacity/128)*(width/128);ctas=dims[6];threads=dims[7];
        DG_HOST_ASSERT(groups>0 && groups<128 && capacity>0 && capacity%128==0 && width>=128 && width<=8192 && width%128==0 && capacity*width<=INT32_MAX);
        DG_HOST_ASSERT(tokens>0 && topk>0 && topk<=64 && ranks>0 && ranks<=64 && tokens*topk*ranks<=INT32_MAX);
        expect(0,capacity*width,f8);expect(1,capacity*(width/128),fp);expect(2,capacity,fp);expect(3,capacity,i32);expect(4,ranks,i64);
        expect(5,groups,i32);expect(6,groups+1,i32);expect(7,capacity*width,f8);expect(8,(capacity/128)*width,fp);expect(9,1,i32);
        for(int i:{0,7})DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[i].data_ptr())%8==0);
    }
    DG_HOST_ASSERT(ctas>0 && ctas<=16 && (threads==128 || threads==256 || ((threads==64 || threads==96 || threads==512 || threads==1024) && tiled_source)));
    const int64_t work_rows=(operation==0 || operation==7)?dims[0]:(operation==10 || operation==18 || operation==19)?dims[0]*dims[1]*(width/128):rows;
    const int divisor=operation==15?threads:(operation==0 || operation==7 || (operation==10 || operation==18 || operation==19))?threads/16:operation==1?threads/32:operation==2?threads/8:(operation==4 || operation==9)?threads/16:1;
    const int64_t blocks=(operation==10 || operation==18 || operation==19) && dims.size()==7?std::min<int64_t>(ceil_div<int64_t>(work_rows,divisor),dims[6]):operation==16?(dims.size()==5?dims[4]:ceil_div(dims[0],int64_t(128))):operation==6?1:tiled_source?(dims.size()>=11?dims[10]:get_num_sms()*ctas):std::min<int64_t>(ceil_div<int64_t>(work_rows,divisor),get_num_sms()*ctas);
    auto a=Fp8GradientRuntime::Args{operation,int(width),int(threads),{dim3(blocks),dim3(threads),0},{}};
    for(size_t i=0;i<tensors.size();++i)a.params.ptr[i]=tensors[i].data_ptr();
    for(size_t i=0;i<dims.size();++i)a.params.dim[i]=dims[i];
    const auto runtime=compiler->build(Fp8GradientRuntime::name(a),Fp8GradientRuntime::generate(a));
    if(!compile_only)Fp8GradientRuntime::launch(runtime,a);
}
static void register_fp8_gradient_apis(pybind11::module_& m) {
    m.def("_mega_moe_training_fp8_gradient",&fp8_gradient_launch,py::arg("operation"),py::arg("tensors"),py::arg("dims"),py::arg("compile_only")=false);
}
} // namespace deep_gemm::mega_moe_training


#include <c10/cuda/CUDAGuard.h>
#include "../jit_kernels/impls/fp8_gemm.hpp"
#include <deep_gemm/common/mega_moe_training.cuh>

namespace deep_gemm::mega_moe_training {

class MegaMoEFp8ReturnRuntime final : public LaunchRuntime<MegaMoEFp8ReturnRuntime> {
public:
    struct Args {
        FP8GemmRuntime::LaunchInfo launch_info;
        LaunchArgs launch_args;
        FP8GemmRuntime::GemmKernelParams kernel_params;
        MegaMoEGemmParams peers;
        bool keep_local;
        MegaMoEDispatchPipeline input;
        bool input_mode;
        MegaMoEDispatchPipeline output;
        bool output_mode;
        bool copy_only;
        int copy_threads;
        int return_store_bytes;
        bool activation_backward;
        uint32_t* input_work_counter;
        int input_ready_count;
        bool tile_producer;
        int producer_topk;
        MegaMoEFp8ProducerGradient producer_gradient;
        bool gradient_producer;
        bool k_slice_input;
        int input_sub_m;
        int expert_begin=0,expert_end=0;
        bool preprocessed_experts=false;
        int inline_kind=0;
        int inline_bulk_ilp=4;
        int inline_max_regs=0;
        int return_remote_mode=0;
        bool inline_source_bulk=false;
        bool inline_source_rotation=false;
        MegaMoEFp8InlineSource inline_source{};
    };
    static auto compile_key(const Args& args) {
        return std::make_tuple(FP8GemmRuntime::compile_key(
            {args.launch_info, args.launch_args, args.kernel_params}), args.keep_local, args.input_mode, args.output_mode, args.copy_only,
            args.copy_only ? args.copy_threads : 0, args.return_store_bytes, args.activation_backward,
            args.input_work_counter!=nullptr, args.input_ready_count, args.tile_producer, args.producer_topk, args.gradient_producer, args.k_slice_input, args.input_sub_m, args.expert_begin, args.expert_end, args.preprocessed_experts, args.inline_kind,args.inline_bulk_ilp,args.inline_max_regs,args.return_remote_mode,args.inline_source_bulk,args.inline_source_rotation,
            args.copy_only || args.return_store_bytes==16 ? args.peers.destination_rows_per_rank : 0);
    }
    static std::string generate_impl(const Args& args) {
        auto source = FP8GemmRuntime::generate_impl({args.launch_info, args.launch_args, args.kernel_params});
        if(args.preprocessed_experts && !args.inline_kind)return source;
        const auto replace = [&](const std::string& from, const std::string& to) {
            const auto position = source.find(from);
            TORCH_CHECK(position != std::string::npos && source.find(from, position + from.size()) == std::string::npos,
                        "expected one native FP8 declaration: ", from);
            source.replace(position, from.size(), to);
        };
        replace("#include <deep_gemm/impls/fp8_gemm.cuh>", "#include <deep_gemm/impls/mega_moe_training/mega_moe_fp8.cuh>");
        if(args.expert_end) {
            // Preserve the actual native universal mainloop and epilogue.
            // Full-range control uses the original scheduler too.
            replace("  TileScheduler,\n  kEnableSboOverlap,",
                "  std::conditional_t<DG_TRAINING_FP8_EXPERT_BEGIN==0 && DG_TRAINING_FP8_EXPERT_END==NUM_GROUPS, TileScheduler, TrainingFp8ExpertBatchScheduler<TileScheduler>>,\n  kEnableSboOverlap,");
        } else replace("using GemmKernel = DeepGemmUniversal<", args.copy_only ? "using GemmKernel = TrainingFp8PeerTileCopy<" : "using GemmKernel = TrainingFp8PeerReturn<");
        if(args.copy_only) {
            replace("__launch_bounds__(GemmKernel::MaxThreadsPerBlock, GemmOcc::kMinBlocksPerMultiprocessor)",
                fmt::format("__launch_bounds__({}, 1)",args.copy_threads));
            // The copy kernel uses static row-address scratch and no MMA or
            // dynamic TSM. Query its real function occupancy at launch below.
            const auto begin=source.find("using GemmOcc = GemmOccModel<");
            TORCH_CHECK(begin!=std::string::npos,"expected native GEMM occupancy declaration");
            const auto end=source.find(';',begin);
            TORCH_CHECK(end!=std::string::npos,"expected end of GEMM occupancy declaration");
            source.erase(begin,end-begin+1);
        }
        if(!args.expert_end) {
            replace("typename GemmKernel::Params params", "typename GemmKernel::Params params, const deep_gemm::mega_moe_training::MegaMoEGemmParams peers, const deep_gemm::mega_moe_training::MegaMoEDispatchPipeline input, const deep_gemm::mega_moe_training::MegaMoEDispatchPipeline output, uint32_t* input_work_counter, const deep_gemm::mega_moe_training::MegaMoEFp8ProducerGradient producer_gradient" + std::string(args.inline_kind?", const deep_gemm::mega_moe_training::MegaMoEFp8InlineSource inline_source":""));
            replace("op(params, smem);", args.inline_kind ? "op(params, peers, input, output, input_work_counter, producer_gradient, smem, inline_source);" : "op(params, peers, input, output, input_work_counter, producer_gradient, smem);");
        }
        if(args.inline_max_regs) {
            replace("__launch_bounds__(GemmKernel::MaxThreadsPerBlock, GemmOcc::kMinBlocksPerMultiprocessor)",
                fmt::format("__maxnreg__({})",args.inline_max_regs));
        }
        source=fmt::format("#define DG_TRAINING_FP8_RETURN_REMOTE_MODE {}\n",args.return_remote_mode)+source;
        source=fmt::format("#define DG_TRAINING_FP8_INLINE_SOURCE_BULK_ILP {}\n",args.inline_bulk_ilp)+source;
        source=fmt::format("#define DG_TRAINING_FP8_INLINE_SOURCE_BULK {}\n",int(args.inline_source_bulk))+source;
        source=fmt::format("#define DG_TRAINING_FP8_INLINE_SOURCE_ROTATION {}\n",int(args.inline_source_rotation))+source;
        source=fmt::format("#define DG_TRAINING_FP8_INLINE_SOURCE {}\n",args.inline_kind)+source;
        source = fmt::format("#define DG_TRAINING_FP8_INPUT_K_SLICE {}\n", int(args.k_slice_input)) + source;
        source=fmt::format("#define DG_TRAINING_FP8_INPUT_SUBTILES {}\n",128/args.input_sub_m)+source;
        source = fmt::format("#define DG_TRAINING_FP8_TILE_PRODUCER {}\n#define DG_TRAINING_FP8_PRODUCER_TOPK {}\n#define DG_TRAINING_FP8_GRADIENT_PRODUCER {}\n", int(args.tile_producer), args.producer_topk, int(args.gradient_producer)) + source;
        source=fmt::format("#define DG_TRAINING_FP8_EXPERT_BEGIN {}\n#define DG_TRAINING_FP8_EXPERT_END {}\n",args.expert_begin,args.expert_end)+source;
        return fmt::format("#define DG_TRAINING_FP8_PEER_KEEP_LOCAL {}\n#define DG_TRAINING_FP8_PEER_INPUT {}\n#define DG_TRAINING_FP8_RETURN_PIPELINE {}\n#define DG_TRAINING_FP8_RETURN_STRIDE {}\n#define DG_TRAINING_FP8_RETURN_STORE_BYTES {}\n#define DG_TRAINING_FP8_ACTIVATION_BACKWARD {}\n#define DG_TRAINING_FP8_INPUT_DYNAMIC {}\n#define DG_TRAINING_FP8_INPUT_READY_COUNT {}\n",
                           int(args.keep_local), int(args.input_mode), int(args.output_mode), args.peers.destination_rows_per_rank, args.return_store_bytes, int(args.activation_backward), int(args.input_work_counter!=nullptr), args.input_ready_count) + source;
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        if(args.expert_end || (args.preprocessed_experts && !args.inline_kind))DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel,config,args.kernel_params));
        else if(args.inline_kind)DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel,config,args.kernel_params,args.peers,args.input,args.output,args.input_work_counter,args.producer_gradient,args.inline_source));
        else DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params, args.peers, args.input, args.output,args.input_work_counter,args.producer_gradient));
    }
};

// Blockwise E4M3 compute, with the same BF16 route return boundary as the
// BF16 mega_moe_training path. Local input packing/quantization precedes this call.
static int mega_moe_fp8_return(const std::pair<torch::Tensor, torch::Tensor>& lhs,
                           const std::pair<torch::Tensor, torch::Tensor>& rhs,
                           const torch::Tensor& out, const torch::Tensor& lengths,
                           const torch::Tensor& permutation, const torch::Tensor& destinations,
                           int destination_rows_per_rank, bool keep_local, bool compile_only,
                           const std::vector<int>& config,
                           const std::vector<torch::Tensor>& input_control, int reserved_input_ctas,
                           const std::vector<torch::Tensor>& return_control, int reserved_return_ctas, bool copy_only, int copy_threads,
                           int return_store_bytes, const std::optional<torch::Tensor>& activation_gate_up,
                           bool co_resident_input, int input_ready_count,
                           const std::vector<torch::Tensor>& producer_sources, int producer_tokens, int producer_topk,
                           const std::vector<torch::Tensor>& producer_gradient_sources, bool k_slice_input, int input_sub_m, int expert_begin, int expert_end, const std::optional<torch::Tensor>& expert_metadata, int metadata_block_m, int metadata_ctas, const std::vector<torch::Tensor>& inline_sources, int inline_tokens, int inline_topk, int inline_ranks, int return_block_m, int inline_max_regs, int return_max_regs, int return_remote_mode, int return_local_rank, bool inline_source_bulk, bool inline_source_rotation, const std::optional<torch::Tensor>& tile_work_counter,int inline_bulk_ilp) {
    const auto& a = lhs.first;
    const auto& b = rhs.first;
    const auto& sa = lhs.second;
    const auto& sb = rhs.second;
    DG_HOST_ASSERT(a.is_cuda() && a.dim() == 2 && b.dim() == 3 && out.dim() == 2);
    const c10::cuda::CUDAGuard guard(a.device());
    for (const auto& t : {b, sa, sb, out, lengths, permutation, destinations})
        DG_HOST_ASSERT(t.is_cuda() && t.device() == a.device());
    for (const auto& t : {a, b, sb, out, lengths, permutation, destinations})
        DG_HOST_ASSERT(t.is_contiguous());
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn && b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(sa.scalar_type() == torch::kFloat32 && sb.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(out.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(lengths.scalar_type() == torch::kInt32 && permutation.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(destinations.scalar_type() == torch::kInt64);
    const auto [m, k] = get_shape<2>(a);
    const auto [groups, n, weight_k] = get_shape<3>(b);
    const bool preprocessed_mode=expert_metadata.has_value();
    DG_HOST_ASSERT(return_remote_mode>=0 && return_remote_mode<=4);
    DG_HOST_ASSERT(return_local_rank>=0 && return_local_rank<destinations.numel());
    TORCH_CHECK(!return_remote_mode || (!preprocessed_mode && inline_sources.empty() && input_control.empty()
        && !activation_gate_up.has_value() && producer_sources.empty() && producer_gradient_sources.empty()
        && !k_slice_input && expert_begin==0 && expert_end==0 &&
        (copy_only || (!return_control.empty()) || return_store_bytes==16)),
        "remote bulk is available only for BF16 direct wide return or return copy");
    DG_HOST_ASSERT(return_block_m==0 || return_block_m==64 || return_block_m==128 || return_block_m==192 || return_block_m==256);
    DG_HOST_ASSERT(!return_block_m || (!preprocessed_mode && inline_sources.empty() && input_control.empty()
        && (!copy_only || !return_control.empty()) && !activation_gate_up.has_value() && producer_sources.empty() && producer_gradient_sources.empty()
        && !k_slice_input && expert_begin==0 && expert_end==0 && config.size()==5));
    DG_HOST_ASSERT(inline_sources.empty() || inline_sources.size()==2 || inline_sources.size()==4 || inline_sources.size()==5 || inline_sources.size()==8);
    const int inline_kind=inline_sources.empty()?0:inline_sources.size()==2?3:inline_sources.size()==4?4:inline_sources.size()==5?1:2;
    DG_HOST_ASSERT(inline_bulk_ilp==4 || inline_bulk_ilp==8 || inline_bulk_ilp==16);
    TORCH_CHECK(inline_bulk_ilp==4 || (inline_kind==1 && inline_source_bulk),"source ILP override requires forward bulk inline input");
    TORCH_CHECK(!inline_source_bulk || (inline_kind==1 && k>=512 && k%512==0),"inline bulk source requires forward FP8 width divisible by512");
    DG_HOST_ASSERT(inline_max_regs==0 || inline_max_regs==224 || inline_max_regs==232 || inline_max_regs==240 || inline_max_regs==248);
    DG_HOST_ASSERT(!inline_max_regs || inline_kind>=3);
    DG_HOST_ASSERT(return_max_regs==0 || return_max_regs==224 || return_max_regs==232 || return_max_regs==240 || return_max_regs==248);
    DG_HOST_ASSERT(!return_max_regs || (!preprocessed_mode && inline_sources.empty() && input_control.empty() && !copy_only
        && !activation_gate_up.has_value() && producer_sources.empty() && producer_gradient_sources.empty() && !k_slice_input
        && expert_begin==0 && expert_end==0 && !inline_max_regs));
    DG_HOST_ASSERT(inline_kind ? preprocessed_mode : (inline_tokens==0 && inline_topk==0 && inline_ranks==0));
    DG_HOST_ASSERT(metadata_ctas>=0 && metadata_ctas<=2*get_num_sms() && (preprocessed_mode || metadata_ctas==0));
    DG_HOST_ASSERT(preprocessed_mode ? (metadata_block_m==32 || metadata_block_m==64 || metadata_block_m==128 || metadata_block_m==192 || metadata_block_m==256) : metadata_block_m==0);
    DG_HOST_ASSERT(!preprocessed_mode || (keep_local && input_control.empty() && return_control.empty()
        && !copy_only && return_store_bytes==4 && !activation_gate_up.has_value()
        && producer_sources.empty() && producer_gradient_sources.empty() && !k_slice_input && expert_begin==0 && expert_end==0));
    DG_HOST_ASSERT(expert_begin>=0 && expert_end>=0 && (expert_end==0 ? expert_begin==0 : expert_begin<expert_end && expert_end<=groups));
    DG_HOST_ASSERT(!expert_end || (keep_local && input_control.empty() && return_control.empty()
        && !copy_only && return_store_bytes==4 && !activation_gate_up.has_value()
        && producer_sources.empty() && producer_gradient_sources.empty() && !k_slice_input));
    DG_HOST_ASSERT(m > 0 && m % 128 == 0 && n > 0 && n % 128 == 0 && k > 0 && k % 128 == 0);
    DG_HOST_ASSERT(k == weight_k && groups > 0 && groups < 128);
    DG_HOST_ASSERT(a.numel() <= INT32_MAX && b.numel() <= INT32_MAX && out.numel() <= INT32_MAX);
    DG_HOST_ASSERT(out.size(0) == m && out.size(1) == n);
    DG_HOST_ASSERT(sa.dim() == 2 && sa.size(0) == m && sa.size(1) == k / 128);
    DG_HOST_ASSERT(sb.numel() == groups * (n / 128) * (k / 128));
    DG_HOST_ASSERT(lengths.numel() == groups && permutation.numel() == m);
    DG_HOST_ASSERT(destinations.numel() > 0 && destinations.numel() <= 64 && destination_rows_per_rank > 0);
    auto aligned_sa = get_mn_major_tma_aligned_tensor(sa);
    const int num_sms = get_num_sms();
    const bool tile_producer = !producer_sources.empty();
    if (tile_producer) {
        TORCH_CHECK(producer_sources.size() == 3 && keep_local && input_control.empty() &&
                    return_control.empty() && !copy_only && !activation_gate_up && !co_resident_input && return_store_bytes == 4,
                    "tile producer requires local FP8 GEMM with source pointers, BF16 cache and status");
        const auto& table = producer_sources[0];const auto& packed = producer_sources[1];const auto& status = producer_sources[2];
        for (const auto& t : producer_sources) DG_HOST_ASSERT(t.is_cuda() && t.device() == a.device() && t.is_contiguous());
        DG_HOST_ASSERT(table.scalar_type() == torch::kInt64 && table.numel() == destinations.numel());
        DG_HOST_ASSERT(packed.scalar_type() == torch::kBFloat16 && packed.dim() == 2 && packed.size(0) == m && packed.size(1) == k);
        DG_HOST_ASSERT(status.scalar_type() == torch::kInt32 && status.numel() == 1);
        DG_HOST_ASSERT(producer_tokens > 0 && producer_topk > 0 && producer_topk <= 64 &&
                       int64_t(producer_tokens) * producer_topk == destination_rows_per_rank);
    } else DG_HOST_ASSERT(producer_tokens == 0 && producer_topk == 0);
    const bool gradient_producer = !producer_gradient_sources.empty();
    MegaMoEFp8ProducerGradient producer_gradient{};
    if (gradient_producer) {
        DG_HOST_ASSERT(tile_producer && producer_gradient_sources.size() == 3);
        for (const auto& t : producer_gradient_sources) DG_HOST_ASSERT(t.is_cuda() && t.device() == a.device() && t.is_contiguous());
        const auto& scores = producer_gradient_sources[0];const auto& y = producer_gradient_sources[1];const auto& partials = producer_gradient_sources[2];
        DG_HOST_ASSERT(scores.scalar_type() == torch::kInt64 && scores.numel() == destinations.numel());
        DG_HOST_ASSERT(y.scalar_type() == torch::kBFloat16 && y.dim() == 2 && y.size(0) == m && y.size(1) == k);
        DG_HOST_ASSERT(partials.scalar_type() == torch::kFloat32 && partials.dim() == 2 && partials.size(0) == m && partials.size(1) == k / 128);
        producer_gradient = {reinterpret_cast<const uint64_t*>(scores.data_ptr<int64_t>()), y.data_ptr(), partials.data_ptr<float>()};
    }
    const bool input_mode = !input_control.empty();
    DG_HOST_ASSERT(input_ready_count>=1 && input_ready_count<=16 &&
        (input_ready_count&(input_ready_count-1))==0 &&
        (input_mode ? k%(128*input_ready_count)==0 : input_ready_count==1));
    TORCH_CHECK(!k_slice_input || (input_mode && input_ready_count>=1 && !tile_producer && k%(128*input_ready_count)==0),
                "K-slice input requires explicit bitmap control and aligned K stripes");
    DG_HOST_ASSERT(input_sub_m==16 || input_sub_m==32 || input_sub_m==64 || input_sub_m==128);
    DG_HOST_ASSERT(input_ready_count*(128/input_sub_m)<=32 && (input_sub_m==128 || k_slice_input));
    const bool dynamic_input=input_control.size()==3;
    MegaMoEDispatchPipeline input{}, output{};
    const bool output_mode = !return_control.empty();
    DG_HOST_ASSERT(!(input_mode && output_mode) && (!copy_only || output_mode));
    DG_HOST_ASSERT(!co_resident_input || (input_mode && !output_mode && !copy_only && !activation_gate_up));
    DG_HOST_ASSERT(!dynamic_input || co_resident_input);
    DG_HOST_ASSERT(return_store_bytes==4 || return_store_bytes==16);
    DG_HOST_ASSERT(return_store_bytes==4 || (!input_mode && !output_mode));
    const bool activation_backward=activation_gate_up.has_value();
    if(activation_backward) {
        const auto& gu=*activation_gate_up;
        DG_HOST_ASSERT(!input_mode && !output_mode && !copy_only && !keep_local && return_store_bytes==4);
        DG_HOST_ASSERT(gu.is_cuda() && gu.device()==a.device() && gu.is_contiguous() &&
                       gu.scalar_type()==torch::kBFloat16 && gu.dim()==2 && gu.size(0)==m && gu.size(1)==2*n);
        DG_HOST_ASSERT(gu.numel()<=INT32_MAX);
    }
    DG_HOST_ASSERT(copy_threads==64 || copy_threads==128 || copy_threads==256 || copy_threads==512 || copy_threads==1024);
    if (input_mode) {
        TORCH_CHECK((input_control.size() == 2 || input_control.size()==3) && keep_local,
                    "FP8 input pipeline requires ready/status and a local output");
        for (const auto& tensor : input_control)
            DG_HOST_ASSERT(tensor.is_cuda() && tensor.device() == a.device() && tensor.is_contiguous()
                           && tensor.scalar_type() == torch::kInt32);
        DG_HOST_ASSERT(input_control[0].numel() == m / 128 && input_control[1].numel() == 1);
        DG_HOST_ASSERT(co_resident_input ? (dynamic_input ? reserved_input_ctas>=num_sms && reserved_input_ctas<=(input_sub_m<128?16:4)*num_sms && reserved_input_ctas%num_sms==0 : reserved_input_ctas==num_sms) :
                       reserved_input_ctas>0 && reserved_input_ctas<num_sms);
        if(dynamic_input)DG_HOST_ASSERT(input_control[2].numel()==1);
        input.ready = reinterpret_cast<uint32_t*>(input_control[0].data_ptr<int32_t>());
        input.status = reinterpret_cast<uint32_t*>(input_control[1].data_ptr<int32_t>());
        input.signal_capacity = m / 128;
    } else {
        DG_HOST_ASSERT(reserved_input_ctas == 0);
    }
    if (output_mode) {
        DG_HOST_ASSERT(return_control.size()==2 && keep_local && n%256==0 &&
                       reserved_return_ctas>0 && reserved_return_ctas<num_sms);
        for(const auto& tensor:return_control)
            DG_HOST_ASSERT(tensor.is_cuda() && tensor.device()==a.device() && tensor.is_contiguous()
                           && tensor.scalar_type()==torch::kInt32);
        DG_HOST_ASSERT(return_control[0].numel()==(m/128)*(n/256) && return_control[1].numel()==1);
        output.ready=reinterpret_cast<uint32_t*>(return_control[0].data_ptr<int32_t>());
        output.status=reinterpret_cast<uint32_t*>(return_control[1].data_ptr<int32_t>());
        output.signal_capacity=return_control[0].numel();
    } else DG_HOST_ASSERT(reserved_return_ctas==0);
    if(return_remote_mode)output.communication_ctas=uint32_t(return_local_rank+1);
    auto cfg = deep_gemm_fp8_common::get_best_configs(
        ceil_div(m, groups), n, k, groups, num_sms, GemmType::MGroupedNoPad);
    if (!config.empty()) {
        DG_HOST_ASSERT(config.size() == 5 && config[0] > 0 && config[1] == 128 &&
                       config[2] > 0 && config[3] > 0 && config[4] > 0);
        const int config_m=preprocessed_mode ? metadata_block_m : return_block_m ? return_block_m : 128;
        cfg = std::make_tuple(num_sms, config_m, config[0], config[1], config[2], config[3], config[4],
            deep_gemm_fp8_common::get_smem_config(config[4], k, config_m, config[0], config[1]));
    }
    // Small-K two-stage heuristics may select the native N-stage expansion.
    // This return wrapper currently preserves the regular mainloop only.
    if ((!preprocessed_mode || inline_kind) && k <= 512 && std::get<6>(cfg) == 2) {
        TORCH_CHECK(config.empty(), "FP8 peer return with K<=512 requires at least three stages");
        std::get<6>(cfg) = 3;
        std::get<7>(cfg) = deep_gemm_fp8_common::get_smem_config(
            3, k, std::get<1>(cfg), std::get<2>(cfg), std::get<3>(cfg));
    }
    const auto [sms, bm, bn, bk, wm, wn, stages, smem] = cfg;
    if(preprocessed_mode) {
        const auto& metadata=*expert_metadata;
        DG_HOST_ASSERT(metadata.device()==a.device() && metadata.is_contiguous() && metadata.dim()==1 && metadata.scalar_type()==torch::kInt32);
        DG_HOST_ASSERT(metadata_block_m==bm && metadata.numel()>=std::max<int64_t>(128,4*(1+ceil_div(m,bm)+groups)));
    }
    DG_HOST_ASSERT(bk == 128);
    if (tile_producer) DG_HOST_ASSERT(bm == 128 && ((bm/wm)*(bn/wn)*32 == 256 || (bm/wm)*(bn/wn)*32 == 512));
    if(output_mode)DG_HOST_ASSERT((bm==128 || bm==192 || bm==256) && bn==256 && (!return_block_m || bm==return_block_m));
    using Stride = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using Granularity = cute::Shape<cute::_1, cute::_128, cute::_128>;
    using ScaleConfig = decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<Granularity, false, true>(Granularity{}));
    const auto stride_a = cutlass::make_cute_packed_stride(Stride{}, cute::make_shape(m, k, 1));
    const auto stride_b = cutlass::make_cute_packed_stride(Stride{}, cute::make_shape(n, k, 1));
    const auto stride_d = cutlass::make_cute_packed_stride(Stride{}, cute::make_shape(m, n, 1));
    const auto layout_sa = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    const auto layout_sb = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
    cutlass::KernelHardwareInfo hw;
    hw.device_id = a.get_device();
    hw.cu_count = sms;
    const auto gemm = FP8GemmRuntime::GemmArguments{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape = {int(m), int(n), int(k), 1},
        .mainloopargs = {reinterpret_cast<cutlass::float_e4m3_t*>(a.data_ptr()), stride_a,
                         reinterpret_cast<cutlass::float_e4m3_t*>(b.data_ptr()), stride_b, 4,
                         aligned_sa.data_ptr<float>(), layout_sa, sb.data_ptr<float>(), layout_sb},
        .epilogueargs = {{1, 0}, nullptr, stride_d,
                         reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr()), stride_d},
        .hw_info = hw,
        .scheduler = {uint32_t(m), preprocessed_mode ? expert_metadata->data_ptr<int32_t>() : lengths.data_ptr<int32_t>()},
        .signal = nullptr};
    uint32_t* external_counter=nullptr;
    if(tile_work_counter) {
        const auto& counter=*tile_work_counter;
        TORCH_CHECK(inline_kind>=3 && preprocessed_mode && input_control.empty() && keep_local,
            "tile work counter requires a preprocessed external-ready local GEMM");
        TORCH_CHECK(counter.is_cuda() && counter.device()==a.device() && counter.is_contiguous() &&
            counter.scalar_type()==torch::kInt32 && counter.numel()==1,
            "tile work counter must be one contiguous device int32");
        for(const auto& tensor:inline_sources)TORCH_CHECK(!counter.is_alias_of(tensor),"tile counter must not alias source state");
        external_counter=reinterpret_cast<uint32_t*>(counter.data_ptr<int32_t>());
    }

    MegaMoEFp8InlineSource inline_source{};
    TORCH_CHECK(!inline_source_rotation || (inline_kind && return_local_rank>=0 && return_local_rank<destinations.numel()),"inline source rotation requires an inline consumer and valid local rank");
    inline_source.local_rank=return_local_rank;inline_source.num_ranks=destinations.numel();
    if(inline_kind==1 || inline_kind==2) {
        DG_HOST_ASSERT(k>=128 && k<=8192 && k%128==0 && m%128==0 && bm%4==0 && stages>=3 &&
            sa.stride(0)==1 && sa.stride(1)==m && inline_tokens>0 && inline_topk>0 && inline_ranks>0 && inline_ranks<=64 &&
            int64_t(inline_tokens)*inline_topk*inline_ranks<=INT32_MAX);
        for(const auto& t:inline_sources)DG_HOST_ASSERT(t.is_cuda() && t.device()==a.device() && t.is_contiguous());
        const auto check=[&](int index,int64_t size,torch::ScalarType dtype){DG_HOST_ASSERT(inline_sources[index].numel()==size && inline_sources[index].scalar_type()==dtype);};
        check(0,inline_ranks,torch::kInt64);check(1,inline_ranks,torch::kInt64);check(2,m,torch::kInt32);
        DG_HOST_ASSERT(inline_sources[2].data_ptr()==permutation.data_ptr());
        const int ready=inline_kind==1?3:6;
        DG_HOST_ASSERT(inline_sources[ready].scalar_type()==torch::kInt32 && inline_sources[ready].numel()>=ceil_div(m,bm)+groups);
        check(ready+1,1,torch::kInt32);
        DG_HOST_ASSERT(!inline_sources[ready].is_alias_of(inline_sources[ready+1]));
        inline_source.value_peers=reinterpret_cast<const uint64_t*>(inline_sources[0].data_ptr<int64_t>());
        inline_source.aux_peers=reinterpret_cast<const uint64_t*>(inline_sources[1].data_ptr<int64_t>());
        inline_source.permutation=permutation.data_ptr<int32_t>();
        inline_source.ready=reinterpret_cast<uint32_t*>(inline_sources[ready].data_ptr<int32_t>());
        inline_source.status=reinterpret_cast<uint32_t*>(inline_sources[ready+1].data_ptr<int32_t>());
        inline_source.tokens=inline_tokens;inline_source.topk=inline_topk;inline_source.num_ranks=inline_ranks;
        inline_source.signal_capacity=inline_sources[ready].numel();
        inline_source.local_rank=return_local_rank;
        if(inline_kind==2) {
            check(3,m*k,torch::kBFloat16);check(4,m*k,torch::kBFloat16);check(5,m*(k/128),torch::kFloat);
            inline_source.unweighted_y=inline_sources[3].data_ptr();inline_source.packed_dy=inline_sources[4].data_ptr();
            inline_source.score_partials=inline_sources[5].data_ptr<float>();
        }
    }
    if(inline_kind==3 || inline_kind==4) {
        DG_HOST_ASSERT(inline_tokens==0 && inline_topk==0 && inline_ranks==0);
        const auto& ready=inline_sources[0];const auto& status=inline_sources[1];
        for(const auto& t:inline_sources)DG_HOST_ASSERT(t.is_cuda() && t.device()==a.device() && t.is_contiguous());
        DG_HOST_ASSERT(ready.scalar_type()==torch::kInt32 && ready.numel()>=ceil_div(m,bm)+groups+1);
        DG_HOST_ASSERT(status.scalar_type()==torch::kInt32 && status.numel()==1 && !ready.is_alias_of(status));
        inline_source.ready=reinterpret_cast<uint32_t*>(ready.data_ptr<int32_t>());
        inline_source.status=reinterpret_cast<uint32_t*>(status.data_ptr<int32_t>());
        inline_source.signal_capacity=ready.numel()-1;
        if(inline_kind==4) {
            DG_HOST_ASSERT(inline_sources[2].scalar_type()==torch::kFloat && inline_sources[2].numel()==int64_t(bm)*(k/128));
            DG_HOST_ASSERT(inline_sources[3].scalar_type()==torch::kFloat && inline_sources[3].numel()==m);
            inline_source.broadcast_sf_tile=inline_sources[2].data_ptr<float>();
            inline_source.output_row_scores=inline_sources[3].data_ptr<float>();
        }
    }
    MegaMoEGemmParams peers{};
    peers.permutation = permutation.data_ptr<int32_t>();
    peers.destination_peers = reinterpret_cast<const uint64_t*>(destinations.data_ptr<int64_t>());
    peers.destination_rows_per_rank = destination_rows_per_rank;
    peers.num_ranks = destinations.numel();
    if(activation_backward)peers.activation_gate_up=activation_gate_up->data_ptr();
    if (tile_producer) {
        peers.source_peers = reinterpret_cast<const uint64_t*>(producer_sources[0].data_ptr<int64_t>());
        peers.source_rows_per_rank = producer_tokens;peers.dispatch_scratch = producer_sources[1].data_ptr();
        input.status = reinterpret_cast<uint32_t*>(producer_sources[2].data_ptr<int32_t>());
    }
    const std::string name = inline_kind ? "fp8_training_inline_source" : preprocessed_mode ? "fp8_grouped_deep_gemm_NoPad" : expert_end ? "fp8_training_expert_batch_native" : tile_producer ? "fp8_training_peer_tileproducer" : activation_backward ? "fp8_training_dhidden_activation" : return_store_bytes==16 ? "fp8_training_peer_return_wide16" : copy_only ? "fp8_training_peer_return_copy" : output_mode ? "fp8_training_peer_return_pipeline" : dynamic_input ? "fp8_training_peer_input_elastic" : co_resident_input ? "fp8_training_peer_input_coresident" : input_mode ? "fp8_training_peer_input" :
        keep_local ? "fp8_training_peer_return_keep" : "fp8_training_peer_return_discard";
    auto args = MegaMoEFp8ReturnRuntime::Args{
        .launch_info = {bm, bn, bk, wm, wn, preprocessed_mode ? 128 : int(groups), stages, "MGroupedNoPad", "Default", name, false},
        .launch_args = {get_grid_shape(sms), dim3((bm / wm) * (bn / wn) * 32), std::get<0>(smem)},
        .kernel_params = FP8GemmRuntime::to_underlying_arguments_rtc(gemm, nullptr),
        .peers = peers, .keep_local = keep_local, .input = input, .input_mode = input_mode,
        .output = output, .output_mode = output_mode, .copy_only = copy_only, .copy_threads = copy_threads,
        .return_store_bytes = return_store_bytes, .activation_backward = activation_backward,
        .input_work_counter = external_counter ? external_counter : (dynamic_input ? reinterpret_cast<uint32_t*>(input_control[2].data_ptr<int32_t>()) : nullptr),
        .input_ready_count = input_ready_count, .tile_producer = tile_producer, .producer_topk = producer_topk,
        .producer_gradient = producer_gradient, .gradient_producer = gradient_producer, .k_slice_input = k_slice_input, .input_sub_m=input_sub_m,.expert_begin=expert_begin,.expert_end=expert_end,.preprocessed_experts=preprocessed_mode,.inline_kind=inline_kind,.inline_bulk_ilp=inline_bulk_ilp,.inline_max_regs=inline_max_regs ? inline_max_regs : return_max_regs,.return_remote_mode=return_remote_mode,.inline_source_bulk=inline_source_bulk,.inline_source_rotation=inline_source_rotation,.inline_source=inline_source};
    if(inline_kind || (output_mode && !copy_only)){args.launch_args.smem_size=((args.launch_args.smem_size+127)/128)*128+128;DG_HOST_ASSERT(args.launch_args.smem_size<=256*1024);}
    if(return_store_bytes==16) {
        args.launch_args.smem_size=std::max<size_t>(args.launch_args.smem_size,size_t(bm)*(bn+16)*sizeof(cutlass::bfloat16_t));
        DG_HOST_ASSERT(args.launch_args.smem_size+size_t(bm)*sizeof(uint64_t)+8<=256*1024);
    }
    if(copy_only) { args.launch_args.block_dim.x=copy_threads; args.launch_args.smem_size=0; }
    const auto runtime = compiler->build(name, MegaMoEFp8ReturnRuntime::generate(args));
    int occupancy = 0;
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, runtime->kernel, args.launch_args.block_dim.x, args.launch_args.smem_size));
    DG_HOST_ASSERT(occupancy > 0);
    TORCH_CHECK(!co_resident_input || occupancy>=2,
                "co-resident input GEMM must permit at least two resident CTAs per CU");
    // Waiting consumers may never fill the CUs reserved for their producer.
    // In particular, multiplying this grid by occupancy can deadlock input.
    if(copy_only)args.launch_args.grid_dim.x=reserved_return_ctas;
    else if (input_mode) args.launch_args.grid_dim.x = co_resident_input ? num_sms : num_sms-reserved_input_ctas;
    else if(output_mode)args.launch_args.grid_dim.x=num_sms-reserved_return_ctas;
    else args.launch_args.grid_dim.x *= occupancy;
    // No polling consumer exists in metadata mode. Limiting work CTAs is
    // a resource experiment, not CU affinity or a joint-residency promise.
    if(preprocessed_mode && metadata_ctas)args.launch_args.grid_dim.x=std::min<uint32_t>(args.launch_args.grid_dim.x,metadata_ctas);
    if (!compile_only) MegaMoEFp8ReturnRuntime::launch(runtime, args);
    return args.launch_args.grid_dim.x;
}


// Read-only native heuristic query for preallocating a graph-stable schedule.
static std::vector<int> mega_moe_fp8_auto_config(int device,int m,int n,int k,int groups) {
    const c10::cuda::CUDAGuard guard(device);
    DG_HOST_ASSERT(m>0 && n>0 && k>0 && groups>0 && groups<128);
    const auto cfg=deep_gemm_fp8_common::get_best_configs(ceil_div(m,groups),n,k,groups,get_num_sms(),GemmType::MGroupedNoPad);
    return {std::get<1>(cfg),std::get<2>(cfg),std::get<3>(cfg),std::get<4>(cfg),std::get<5>(cfg),std::get<6>(cfg)};
}

static void register_mega_moe_fp8_apis(pybind11::module_& m) {
    m.def("_mega_moe_training_fp8_auto_config",&mega_moe_fp8_auto_config,py::arg("device"),py::arg("m"),py::arg("n"),py::arg("k"),py::arg("groups"));
    m.def("_mega_moe_training_fp8_gemm", &mega_moe_fp8_return,
          py::arg("a"), py::arg("weight"), py::arg("out"), py::arg("lengths"),
          py::arg("permutation"), py::arg("destination_peers"), py::arg("destination_rows_per_rank"),
          py::arg("keep_local") = true, py::arg("compile_only") = false,
          py::arg("config") = std::vector<int>{},
          py::arg("input_control") = std::vector<torch::Tensor>{},
          py::arg("reserved_input_ctas") = 0,
          py::arg("return_control") = std::vector<torch::Tensor>{},
          py::arg("reserved_return_ctas") = 0,
          py::arg("copy_only") = false, py::arg("copy_threads") = 512,
          py::arg("return_store_bytes") = 4,
          py::arg("activation_gate_up") = std::nullopt,
          py::arg("co_resident_input") = false, py::arg("input_ready_count") = 1,
          py::arg("producer_sources") = std::vector<torch::Tensor>{}, py::arg("producer_tokens") = 0, py::arg("producer_topk") = 0, py::arg("producer_gradient_sources") = std::vector<torch::Tensor>{}, py::arg("k_slice_input") = false, py::arg("input_sub_m")=128, py::arg("expert_begin")=0, py::arg("expert_end")=0, py::arg("expert_metadata")=std::nullopt, py::arg("metadata_block_m")=0, py::arg("metadata_ctas")=0,py::arg("inline_sources")=std::vector<torch::Tensor>{},py::arg("inline_tokens")=0,py::arg("inline_topk")=0,py::arg("inline_ranks")=0,py::arg("return_block_m")=0,py::arg("inline_max_regs")=0,py::arg("return_max_regs")=0,py::arg("return_remote_mode")=0,py::arg("return_local_rank")=0,py::arg("inline_source_bulk")=false,py::arg("inline_source_rotation")=false,py::arg("tile_work_counter")=std::nullopt,py::arg("inline_bulk_ilp")=4);
}

} // namespace deep_gemm::mega_moe_training
