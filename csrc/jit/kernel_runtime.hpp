#pragma once

#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/system.hpp"
#include "device_runtime.hpp"
#include "handle.hpp"

namespace deep_gemm {

struct LaunchArgs {
    dim3 grid_dim;
    dim3 block_dim;
    int smem_size;
};

class KernelRuntime final {
public:
    static std::filesystem::path sdk_home;

    LibraryHandle library;
    KernelHandle kernel;

    explicit KernelRuntime(const std::filesystem::path& dir_path) {
        // Check `prepare_init`
        DG_HOST_ASSERT(not sdk_home.empty());

        // NOLINT(*-pro-type-member-init)
        const auto& hgobjdump_path = sdk_home / "../bin" / "hgobjdump";
        const auto& hgbin_path = dir_path / "kernel.hgbin";
        if (get_env<int>("DG_JIT_DEBUG"))
            printf("Loading HGBIN: %s\n", hgbin_path.c_str());

        // Find the only symbol
        // TODO: use kernel enumeration for newer drivers
        const std::vector<std::string> illegal_names = {"vprintf", "__instantiate_kernel", "__internal",
                                                        "__assertfail"};
        const auto& [exit_code, symbols] = call_external_command(
            fmt::format("{} --dump-elf-symbols=0 {}", hgobjdump_path.c_str(), hgbin_path.c_str()));
        if (get_env<int>("DG_JIT_DEBUG"))
            printf("symbols: %s\n", symbols.c_str());
        std::string expected_name = dir_path.filename().string();
        if (expected_name.rfind("kernel.", 0) == 0) {
            expected_name = expected_name.substr(7); // remove "kernel."
        }
        size_t dot_pos = expected_name.find('.');
        if (dot_pos != std::string::npos) {
            expected_name = expected_name.substr(0, dot_pos);
        }

        kernel = load_kernel(hgbin_path, expected_name, &library);
    }

    static void prepare_init(const std::string& sdk_home_path) {
        sdk_home = sdk_home_path;
    }

    static bool check_validity(const std::filesystem::path& dir_path) {
        return std::filesystem::exists(dir_path / "kernel.cu") and std::filesystem::exists(dir_path / "kernel.hgbin");
    }

    ~KernelRuntime() noexcept(false) {
        unload_library(library);
    }
};

DG_DECLARE_STATIC_VAR_IN_CLASS(KernelRuntime, sdk_home);

template <typename Derived>
class LaunchRuntime {
public:
    template <typename Args>
    static std::string generate(const Args& args) {
        const auto& code = Derived::generate_impl(args);
        if (get_env<int>("DG_JIT_DEBUG", 0))
            printf("Generated kernel code: %s\n", code.c_str());
        return code;
    }

    template <typename Args>
    static void launch(const std::shared_ptr<KernelRuntime>& kernel_runtime, const Args& args) {
        const auto& kernel = kernel_runtime->kernel;
        const auto& stream = (hggcStream_t)0;  // default stream
        const LaunchArgs& launch_args = args.launch_args;
        auto config =
            construct_launch_config(kernel, stream, launch_args.smem_size, launch_args.grid_dim, launch_args.block_dim);
        // std::cout << " launch_args.grid_dim" << launch_args.grid_dim.x << launch_args.grid_dim.y <<
        // launch_args.grid_dim.z <<  std::endl;
        //  std::cout << " launch_args.block_dim" << launch_args.block_dim.x << launch_args.block_dim.y <<
        //  launch_args.block_dim.z <<  std::endl;

        // Launch in the derived class
        if (get_env<int>("DG_JIT_DEBUG")) {
            printf("Launch kernel with {%d, %d} x %d  %d, shared memory: %d bytes, stream: %ld\n",
                   launch_args.grid_dim.x, launch_args.grid_dim.y, launch_args.block_dim.x, launch_args.block_dim.y,
                   launch_args.smem_size, (uintptr_t)stream);
        }
        Derived::launch_impl(kernel, config, args);
    }
};

} // namespace deep_gemm
