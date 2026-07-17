#pragma once

#include <hggc_runtime_api.h>
#include <hggc.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <hgrtc.h>
#include <regex>
#include <string>

#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/hash.hpp"
#include "../utils/lazy_init.hpp"
#include "../utils/system.hpp"
#include "../utils/utils.hpp"
#include "cache.hpp"
#include "device_runtime.hpp"
#include "acarch.h"

namespace deep_gemm {

class Compiler {
public:
    static std::filesystem::path library_root_path;
    static std::filesystem::path library_include_path;
    static std::filesystem::path sdk_home;
    static std::string library_version;
    mutable int blocks_per_cu = 1;

    static std::string get_library_version() {
        std::vector<char> buffer;
        for (const auto& f: collect_files(library_include_path / "deep_gemm")) {
            std::ifstream in(f, std::ios::binary);
            DG_HOST_ASSERT(in.is_open());

            // Append into the buffer
            buffer.insert(buffer.end(),
                          std::istreambuf_iterator<char>(in),
                          std::istreambuf_iterator<char>());
        }
        return get_hex_digest(buffer);
    }

    static void prepare_init(const std::string& library_root_path,
                             const std::string& sdk_home_path) {
        Compiler::library_root_path = library_root_path;
        Compiler::library_include_path = Compiler::library_root_path / "include";
        Compiler::sdk_home = sdk_home_path;
        Compiler::library_version = get_library_version();
    }

    std::string signature, flags;
    std::filesystem::path cache_dir_path;

    Compiler() {
        // Check `prepare_init`
        DG_HOST_ASSERT(not library_root_path.empty());
        DG_HOST_ASSERT(not library_include_path.empty());
        DG_HOST_ASSERT(not sdk_home.empty());
        DG_HOST_ASSERT(not library_version.empty());

        // Cache settings
        cache_dir_path = std::filesystem::path(get_env<std::string>("HOME")) / ".deep_gemm";
        if (const auto& env_cache_dir_path = get_env<std::string>("DG_JIT_CACHE_DIR"); not env_cache_dir_path.empty())
            cache_dir_path = env_cache_dir_path;

        // The compiler flags applied to all derived compilers
        signature = "unknown-compiler";
        flags = fmt::format("-std=c++{} --diag-suppress=39,174,177,940 ",
                            //"--ptxas-options=--register-usage-level=10",
                            get_env<int>("DG_CPP_STANDARD", 17));
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0) or get_env("DG_JIT_PTXAS_CHECK", 0))
            flags += " --ptxas-options=--verbose,--warn-on-local-memory-usage";
        if (get_env("DG_JIT_WITH_LINEINFO", 0))
            flags += " -Xcompiler -rdynamic -lineinfo";
    }

    virtual ~Compiler() = default;

    std::filesystem::path make_tmp_dir() const {
        return make_dirs(cache_dir_path / "tmp");
    }

    std::filesystem::path get_tmp_file_path() const {
        return make_tmp_dir() / get_uuid();
    }

    int32_t get_max_block_per_cu() {
        return Compiler::blocks_per_cu;
    }

    void put(const std::filesystem::path& path, const std::string& data) const {
        const auto tmp_file_path = get_tmp_file_path();

        // Write into the temporary file
        std::ofstream out(tmp_file_path, std::ios::binary);
        DG_HOST_ASSERT(out.write(data.data(), data.size()));
        out.close();

        // Atomically replace
        std::filesystem::rename(tmp_file_path, path);
    }

    std::shared_ptr<KernelRuntime> build(const std::string& name, const std::string& code, int32_t thread_num = 0, int32_t smem_size = 0) const {
        const auto kernel_signature = fmt::format("{}$${}$${}$${}$${}", name, library_version, signature, flags, code);
        const auto dir_path = cache_dir_path / "cache" / fmt::format("kernel.{}.{}", name, get_hex_digest(kernel_signature));

        // Hit the runtime cache
        if (const auto& runtime = kernel_runtime_cache->get(dir_path); runtime != nullptr)
            return runtime;

        // Create the kernel directory
        make_dirs(dir_path);

        // Compile into a temporary HGBIN
        const auto tmp_hgbin_path = get_tmp_file_path();
        compile(code, dir_path, tmp_hgbin_path, name, thread_num, smem_size);

        // Replace into the cache directory
        make_dirs(dir_path);
        std::filesystem::rename(tmp_hgbin_path, dir_path / "kernel.hgbin");

        // Put into the runtime cache
        const auto& runtime = kernel_runtime_cache->get(dir_path);
        DG_HOST_ASSERT(runtime != nullptr);
        return runtime;
    }

    virtual void compile(const std::string &code, const std::filesystem::path& dir_path, const std::filesystem::path &hgbin_path, const std::string& name, int32_t thread_num, int32_t smem_size) const = 0;
};

DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_root_path);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_include_path);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, sdk_home);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_version);

class HGCCCompiler final: public Compiler {
    std::filesystem::path hgcc_path;

    // Query the hgcc driver version (best-effort; never fatal on format mismatch)
    std::string get_hgcc_version() const {
        DG_HOST_ASSERT(std::filesystem::exists(hgcc_path) and "hgcc compiler not found");

        // Call the version command
        const auto& command = std::string(hgcc_path) + " --version";
        const auto& [return_code, output] = call_external_command(command);
        DG_HOST_ASSERT(return_code == 0 and "Failed to query hgcc --version");

        std::smatch match;
        if (std::regex_search(output, match, std::regex(R"(version (\d+\.\d+(?:\.\d+)?))")))
            return match[1].str();
        return "unknown";
    }

public:
    HGCCCompiler() {
        // Locate the hgcc offline compiler shipped with the PPU SDK
        hgcc_path = "/usr/local/PPU_SDK/bin/hgcc";
        if (const auto& env_hgcc_path = get_env<std::string>("DG_JIT_HGCC_COMPILER"); not env_hgcc_path.empty())
            hgcc_path = env_hgcc_path;
        signature = fmt::format("HGCC{}", get_hgcc_version());

        // Build hgcc flags incrementally for clarity and maintainability.
        // All configurable paths come from environment variables or SDK detection.
        //
        // Environment variables:
        //   DG_CPP_STANDARD               - C++ standard version (default: 17)
        //   DG_CCBIN                       - host compiler path (e.g. /usr/bin/g++-13)
        //   DG_DELAYED_TEMPLATE_PARSING    - set to "false" to debug hgcc segfaults
        //
        // NOTE on `-fdelayed-template-parsing=false`:
        //   hgcc/hgrtc (clang 13 fork) crashes with NPE inside
        //   clang::Stmt::getBeginLoc() -> clang::InitializationSequence::Diagnose(...)
        //   under default delayed-template-parsing when a template instantiation fails.
        //   Setting DG_DELAYED_TEMPLATE_PARSING=false forces immediate parsing, avoiding
        //   the crash and yielding precise diagnostics. Trade-off: all non-dependent names
        //   must be visible at the template definition point.

        const int cpp_standard = get_env<int>("DG_CPP_STANDARD", 17);
        const std::string inc = library_include_path.string();

        // --- Language & defines ---
        flags = fmt::format("-std=c++{} -DUSE_HGGC -DUSE_CLANG -DUSE_ACWRAPPER ", cpp_standard);

        // --- Architecture ---
        if (is_ppu1v5_device()) {
            flags += "-arch=ppu_15 ";
        } else {
            flags += "-arch=ppu_10 ";
        }

        // --- Include paths (derived from library_include_path) ---
        if (is_ppu1v5_device()) {
            flags += fmt::format("-I{}/actlize_v1.0.0 -I{}/deep_gemm ", inc, inc);
        } else {
            flags += fmt::format("-I{} -I{}/actlize_v0.5.0 -I{}/deep_gemm ", inc, inc, inc);
        }

        // --- Output format & optimization ---
        flags += "-hgbin -ftemplate-depth=8192 -O3 -DNDEBUG ";

        // --- Host compiler flags (passed via -Xcompiler) ---
        flags += "-Xcompiler -fPIC ";
        flags += "-Xcompiler -Wno-deprecated-declarations -Xcompiler -Wno-abi ";

        // --- Optional: host compiler path (DG_CCBIN) ---
        if (const char* ccbin = std::getenv("DG_CCBIN"); ccbin && ccbin[0] != '\0') {
            flags += fmt::format("-ccbin {} ", ccbin);
        }

        // --- Optional: delayed-template-parsing control ---
        if (const auto& dtp = get_env<std::string>("DG_DELAYED_TEMPLATE_PARSING"); dtp == "false") {
            flags += "-fno-delayed-template-parsing ";
        }
        // NOTE: --ptxas-options=--register-usage-level=10 is not supported by hgcc
        std::string hgcc_extra_flags;
        if (is_ppu1v5_device()) {
            flags += " -Xllvm -ppu-patch-fence-ppu=false -Xllvm -wno-loop-miss-transform"
                     " -Xllvm -ppu-cg-to-kp1=true -Xllvm -ppu-fix-uninit=true";
        }

        // --- Source language ---
        flags += " -x hg ";
    }

    void compile(const std::string &code, const std::filesystem::path& dir_path, const std::filesystem::path &hgbin_path, const std::string& name, int32_t thread_num, int32_t smem_size) const override {
        // Write the code into the cache directory
        const auto& code_path = dir_path / "kernel.cu";
        put(code_path, code);

        // Per-kernel flags: warp-interleaving kernels (gemm_fp8, mqa_logits) use the full
        // -Xllvm tuning set; others only need -ppu-simt-branch=false (aligned with compiler.py logic)
        std::string per_kernel_flags;
        if (is_ppu1v5_device()) {
            const bool use_warp_interleaving = (name.find("fp8_grouped_deep_gemm") != std::string::npos) ||
                                               (name.find("fp8_deep_gemm") != std::string::npos) ||
                                               (name.find("mqa_logits") != std::string::npos &&
                                                name.find("paged") == std::string::npos);
            if (!use_warp_interleaving) {
                per_kernel_flags = " -Xllvm -ppu-simt-branch=false";
            } else {
                per_kernel_flags = " -Xllvm -ppu-blksync-nb-schedule-boundary=true"
                                   " -Xllvm -ppu-simt-branch=false"
                                   " -Xllvm -ppu-adjust-tsm-valu-war=13"
                                   " -Xllvm -ppu-reassign-subregs=true"
                                   " -Xllvm -ppu-pref-fma-reuse=true"
                                   " -Xllvm -ppu-pref-mma-reuse=true";
            }
            if (name.find("w4a16") != std::string::npos) {
                per_kernel_flags += " -Xllvm -sort-copy-before-coalesce";
            }
        }
        // Compile
        const auto& command = fmt::format("{} {} -o {} {}{}", hgcc_path.c_str(), code_path.c_str(), hgbin_path.c_str(), flags, per_kernel_flags);
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PRINT_COMPILER_COMMAND", 0))
            printf("Running HGCC command: %s\n", command.c_str());
        const auto& [return_code, output] = call_external_command(command);
        if (return_code != 0) {
            printf("HGCC compilation failed: %s\n", output.c_str());
            DG_HOST_ASSERT(false and "HGCC compilation failed");
        }

        // Print compiler log
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0))
            printf("%s", output.c_str());
    }
};

class HGRTCCompiler final: public Compiler {
public:
    class RtcOptions {
        public:
        RtcOptions() = delete;
        RtcOptions(acArch_t arch, bool use_actlize_v100, const std::string& name = "") {
            opts = {
            "--device-as-default-execution-space",
            "-DHGGC_COMPILER_WRAPPER_MODE",
            // "-U__linux__",
            "-DUSE_HGGC",
            };
            auto opts_insert = [this](const std::vector<std::string> & vb) {
                opts.insert(opts.end(), vb.begin(), vb.end());
            };
            auto includes_insert = [this](const std::vector<std::string> & paths) {
            for (auto & path: paths) {
                opts.emplace_back("--include-path=" + path);
            }
            };
            if (getenv("PPU_HOME") == nullptr) {
              printf("No PPU_HOME exist\n");
            }
            std::string sdk_include = std::string(getenv("PPU_HOME")) + "/include";

        #if defined(HGGC_VERSION) && HGGC_VERSION >= 13000
            std::string sdk_include_cccl = std::string(getenv("PPU_HOME")) + "/include/cccl";
            includes_insert({sdk_include, sdk_include_cccl});
        #else
            includes_insert({sdk_include});
        #endif
            std::string library_include_path_str = fmt::format("{}", library_include_path.c_str());
            if (arch == AC_PPU0010) {
                includes_insert({library_include_path_str+ "/actlize_v0.5.0", library_include_path_str + "/deep_gemm", library_include_path_str});
            } else {
                includes_insert({library_include_path_str+ "/actlize_v1.0.0", library_include_path_str + "/deep_gemm"});
            }
        // #else
            opts_insert({
                "-DNDEBUG",
                "-DUSE_CLANG",
                "-no-cache",
                "--diag-suppress=39,174,177,940",
                // "--ptxas-options=--register-usage-level=10", // not supported
            });

#ifdef DG_HGGC_SUPPORT_PCH
            if (!get_env<int>("DG_JIT_DISABLE_PCH", 0)) {
                opts_insert({"-pch"});
            }
#endif

            if (arch == AC_PPU0010) {
                opts_insert({
                    "--ppu-arch=ppu001",
                });
            } else if (arch == AC_PPU0015) {
                opts_insert({
                    "--ppu-arch=ppu0015",
                    "--ppu-tuning-options=-ppu-patch-fence-ppu=false",
                    "--ppu-tuning-options=-wno-loop-miss-transform",
                    "--ppu-tuning-options=-ppu-cg-to-kp1=true",
                    "--ppu-tuning-options=-ppu-fix-uninit=true",
                });

                const bool use_warp_interleaving =
                    (name.find("fp8_grouped_deep_gemm") != std::string::npos) ||
                    (name.find("fp8_deep_gemm")         != std::string::npos) ||
                    (name.find("mqa_logits")            != std::string::npos &&
                     name.find("paged")                 == std::string::npos);
                if (!use_warp_interleaving) {
                    opts_insert({
                        "--ppu-tuning-options=-ppu-simt-branch=false",
                    });
                } else {
                    opts_insert({
                        "--ppu-tuning-options=-ppu-blksync-nb-schedule-boundary=true",
                        "--ppu-tuning-options=-ppu-simt-branch=false",
                        "--ppu-tuning-options=-ppu-adjust-tsm-valu-war=13",
                        "--ppu-tuning-options=-ppu-reassign-subregs=true",
                        "--ppu-tuning-options=-ppu-pref-fma-reuse=true",
                        "--ppu-tuning-options=-ppu-pref-mma-reuse=true",
                    });
                }
                if (name.find("w4a16") != std::string::npos) {
                    opts_insert({
                        "--ppu-tuning-options=-sort-copy-before-coalesce",
                    });
                }
            }
        // #endif
            if (get_env<int>("DG_JIT_DEBUG", 0) || get_env<int>("DG_JIT_PTXAS_VERBOSE", 0) || get_env<int>("DG_JIT_PTXAS_CHECK", 0)) {
                opts_insert({"--ptxas-options=--verbose,--warn-on-local-memory-usage"});
            }
            if (get_env<int>("DG_JIT_WITH_LINEINFO", 0)) {
                opts_insert({"--generate-line-info"});
            }
            std::string standard = "--std=c++17";
            opts.emplace_back(standard);
            const char* rtc_ccbin_env = std::getenv("DG_CCBIN");
            if (rtc_ccbin_env && rtc_ccbin_env[0] != '\0') {
                opts.emplace_back("-ccbin");
                opts.emplace_back(rtc_ccbin_env);
            }
            // Optional: delayed-template-parsing control (DG_DELAYED_TEMPLATE_PARSING=false)
            if (const auto& dtp = get_env<std::string>("DG_DELAYED_TEMPLATE_PARSING"); dtp == "false") {
                opts.emplace_back("-fno-delayed-template-parsing");
            }

            opts_char.resize(opts.size());
            std::transform(opts.begin(), opts.end(), opts_char.begin(), [](const std::string& s) { return s.c_str(); });
        }

        size_t size() {
            return opts_char.size();
        }

        const char* const* data() {
            return opts_char.data();
        }
        private:
        std::vector<std::string> opts;
        std::vector<const char*> opts_char;
    };

    HGRTCCompiler() {
        // Override the compiler signature
        int major, minor;
        DG_HGRTC_CHECK(hgrtcVersion(&major, &minor));
        signature = fmt::format("HGRTC{}.{}", major, minor);

    }

    void compile(const std::string &code, const std::filesystem::path& dir_path, const std::filesystem::path &hgbin_path, const std::string& name, int32_t thread_num, int32_t smem_size) const override {
        // Write the code into the cache directory
        const auto& code_path = dir_path / "kernel.cu";
        put(code_path, code);
        acArch_t arch = AC_PPU0010;
        bool use_actlize_v100 = false;
        if (is_ppu1v5_device()) {
            arch = AC_PPU0015;
            use_actlize_v100 = true;
        }
        RtcOptions opts(arch, use_actlize_v100, name);
        // Print HGRTC compile options when DG_JIT_DEBUG is enabled
        if (get_env<int>("DG_JIT_DEBUG", 0)) {
            printf("HGRTC compile options (%zu):\n", opts.size());
            for (size_t i = 0; i < opts.size(); ++i) {
                printf("  [%zu] %s\n", i, opts.data()[i]);
            }
        }
        // Create HGRTC program and compile
        hgrtcProgram program;
        DG_HGRTC_CHECK(hgrtcCreateProgram(&program, code.c_str(), "kernel.cu", 0, nullptr, nullptr));
        const auto& compile_result = hgrtcCompileProgram(program, opts.size(), opts.data());

        // Get and print compiler log
        size_t log_size;
        DG_HGRTC_CHECK(hgrtcGetProgramLogSize(program, &log_size));
        if (get_env<int>("DG_JIT_DEBUG", 0) or compile_result != HGRTC_SUCCESS) {
            if (compile_result != HGRTC_SUCCESS)
                DG_HOST_ASSERT(log_size > 1);
            if (log_size > 1) {
                std::string compilation_log(log_size, '\0');
                DG_HGRTC_CHECK(hgrtcGetProgramLog(program, compilation_log.data()));
                printf("HGGCRTC log: %s\n", compilation_log.c_str());
            }
        }

        // Get HGBIN size and data
        size_t hgbin_size;
        DG_HGRTC_CHECK(hgrtcGetHGBINSize(program, &hgbin_size));
        std::string hgbin_data(hgbin_size, '\0');
        DG_HGRTC_CHECK(hgrtcGetHGBIN(program, hgbin_data.data()));

        // Write into the file system
        put(hgbin_path, hgbin_data);
        // Cleanup
        DG_HGRTC_CHECK(hgrtcDestroyProgram(&program));

        HGmodule module;
        hgModuleLoadData(&module, hgbin_data.data());

        HGfunction kernel_func;
        hgModuleGetFunction(&kernel_func, module, name.c_str());

        HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_cu,
        kernel_func,
        thread_num,
        smem_size
        );

        if (result != HGGC_SUCCESS) {
            printf("Get Max active blocks per SM failed!\n");
        }
    }
};

static auto compiler = LazyInit<Compiler>([]() -> std::shared_ptr<Compiler> {
    const int default_use_hgrtc = is_ppu1v5_device() ? 0 : 0;
    if (get_env<int>("DG_JIT_USE_HGRTC", default_use_hgrtc)) {
        return std::make_shared<HGRTCCompiler>();
    } else {
        return std::make_shared<HGCCCompiler>();
    }
});

} // namespace deep_gemm
