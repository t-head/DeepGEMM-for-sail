#pragma once

#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/system.hpp"
#include "device_runtime.hpp"
#include "handle.hpp"
#include "include_parser.hpp"

namespace deep_gemm {

struct LaunchArgs {
    dim3 grid_dim; // the z dim is always 1
    dim3 block_dim; // the y,z dims are always 1
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
        const auto hgobjdump_path = sdk_home / "bin" / "hgobjdump";
        const auto hgbin_path = dir_path / "kernel.hgbin";
        if (get_env<int>("DG_JIT_DEBUG"))
            printf("Loading HGBIN: %s\n", hgbin_path.c_str());

        // Record start time
        std::chrono::high_resolution_clock::time_point start_time;
        if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_JIT_PRINT_LOAD_TIME"))
            start_time = std::chrono::high_resolution_clock::now();

#ifdef DG_JIT_USE_LIBRARY_ENUM_KERNELS
        // Load from the library
        kernel = load_kernel(hgbin_path, {}, &library);
#else
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
#endif

        // Print load time
        if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_JIT_PRINT_LOAD_TIME")) {
            std::chrono::duration<double, std::milli> load_time = std::chrono::high_resolution_clock::now() - start_time;
            printf("Load time (%s): %.2lf ms\n", dir_path.c_str(), load_time.count());
        }
    }

    static void prepare_init(const std::string& sdk_home_path) {
        sdk_home = sdk_home_path;
    }

    static bool check_validity(const std::filesystem::path& dir_path) {
        if (not std::filesystem::exists(dir_path))
            return false;

        // NOTES: if the directory exists, `kernel.cu` and `kernel.hgbin` must both exist,
        // because the directory is created atomically via rename
        if (not std::filesystem::exists(dir_path / "kernel.cu") or
            not std::filesystem::exists(dir_path / "kernel.hgbin")) {
            printf("Corrupted JIT cache directory (missing kernel.cu or kernel.hgbin): %s, "
                   "please run `rm -rf %s` and restart your task.\n",
                   dir_path.c_str(), dir_path.c_str());
            DG_HOST_ASSERT(false and "Corrupted JIT cache directory");
        }
        return true;
    }

    ~KernelRuntime() noexcept(false) {
        unload_library(library);
    }
};

DG_DECLARE_STATIC_VAR_IN_CLASS(KernelRuntime, sdk_home);

// Compile/launch mode switch, mirroring Python `jit/runtime.py` (CompileMode + set/get_compile_mode).
//   COMPILE_AND_RUN: normal -- Compiler::build compiles (+ caches) the kernel, then launch runs it.
//   ONLY_COMPILE:    still compile + cache the kernel, but skip the actual launch. Used for warm-up /
//                    ahead-of-time precompilation so that later serving hits the JIT cache.
enum class CompileMode : int {
    COMPILE_AND_RUN = 0,
    ONLY_COMPILE = 1,
};

// Single source of truth for the current mode. A function-local static behind inline accessors keeps
// it ODR-safe in a header (exactly one instance per process), matching Python's module-global semantics.
inline CompileMode& compile_mode_ref() {
    static CompileMode mode = CompileMode::COMPILE_AND_RUN;
    return mode;
}

// NOTE: `mode` is an int to match the Python calling convention `set_compile_mode(CompileMode.ONLY_COMPILE.value)`.
inline void set_compile_mode(int mode) {
    compile_mode_ref() = static_cast<CompileMode>(mode);
}

inline int get_compile_mode() {
    return static_cast<int>(compile_mode_ref());
}

template <typename Derived>
class LaunchRuntime {
public:
    template <typename Args>
    static std::string generate(const Args& args) {
        auto code = Derived::generate_impl(args);

        // NOTES: we require that `generate_impl`'s includes never change
        static std::string include_hash;
        if (include_hash.empty())
            include_hash = include_parser->get_hash_value(code);

        // TODO: optimize string concat performance
        code = fmt::format("// Includes' hash value: {}\n{}", include_hash, code);
        if (get_env<int>("DG_JIT_DEBUG"))
            printf("Generated kernel code:\n%s\n", code.c_str());
        return code;
    }

    template <typename Args>
    static void launch(const std::shared_ptr<KernelRuntime>& kernel_runtime, const Args& args) {
        // Compile-only / warm-up mode: Compiler::build already compiled + cached the kernel above, so
        // skip the actual launch. Mirrors Python `Runtime.__call__` ONLY_COMPILE / HGGC_WARM_UP check.
        if (get_compile_mode() == static_cast<int>(CompileMode::ONLY_COMPILE) or
            not get_env<std::string>("HGGC_WARM_UP").empty())
            return;

        const auto& kernel = kernel_runtime->kernel;
        const auto& stream = current_stream();
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
