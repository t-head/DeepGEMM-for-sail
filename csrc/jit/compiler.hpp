#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <nvrtc.h>
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
    static std::filesystem::path cuda_home;
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
                             const std::string& cuda_home_path_by_python) {
        Compiler::library_root_path = library_root_path;
        Compiler::library_include_path = Compiler::library_root_path / "include";
        Compiler::cuda_home = cuda_home_path_by_python;
        Compiler::library_version = get_library_version();
    }

    std::string signature, flags;
    std::filesystem::path cache_dir_path;

    Compiler() {
        // Check `prepare_init`
        DG_HOST_ASSERT(not library_root_path.empty());
        DG_HOST_ASSERT(not library_include_path.empty());
        DG_HOST_ASSERT(not cuda_home.empty());
        DG_HOST_ASSERT(not library_version.empty());

        // Cache settings
        cache_dir_path = std::filesystem::path(get_env<std::string>("HOME")) / ".deep_gemm";
        if (const auto& env_cache_dir_path = get_env<std::string>("DG_JIT_CACHE_DIR"); not env_cache_dir_path.empty())
            cache_dir_path = env_cache_dir_path;

        // The compiler flags applied to all derived compilers
        signature = "unknown-compiler";
        flags = fmt::format("-std=c++{} --diag-suppress=39,174,177,940 ",
                            //"--ptxas-options=--register-usage-level=10",
                            get_env<int>("DG_NVCC_OVERRIDE_CPP_STANDARD", 17));
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

        // Compile into a temporary CUBIN
        const auto tmp_cubin_path = get_tmp_file_path();
        compile(code, dir_path, tmp_cubin_path, name, thread_num, smem_size);

        // Replace into the cache directory
        make_dirs(dir_path);
        std::filesystem::rename(tmp_cubin_path, dir_path / "kernel.cubin");

        // Put into the runtime cache
        const auto& runtime = kernel_runtime_cache->get(dir_path);
        DG_HOST_ASSERT(runtime != nullptr);
        return runtime;
    }

    virtual void compile(const std::string &code, const std::filesystem::path& dir_path, const std::filesystem::path &cubin_path, const std::string& name, int32_t thread_num, int32_t smem_size) const = 0;
};

DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_root_path);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_include_path);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, cuda_home);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_version);

class NVCCCompiler final: public Compiler {
    std::filesystem::path nvcc_path;

    std::pair<int, int> get_nvcc_version() const {
        DG_HOST_ASSERT(std::filesystem::exists(nvcc_path));

        // Call the version command
        const auto& command = std::string(nvcc_path) + " --version";
        const auto& [return_code, output] = call_external_command(command);
        DG_HOST_ASSERT(return_code == 0);

        // The version should be at least 12.3, for the best performance with 12.9
        int major, minor;
        std::smatch match;
        DG_HOST_ASSERT(std::regex_search(output, match, std::regex(R"(release (\d+\.\d+))")));
        std::sscanf(match[1].str().c_str(), "%d.%d", &major, &minor);
        DG_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVCC version should be >= 12.3");
        if (major == 12 and minor < 9)
            printf("Warning: please use at least NVCC 12.9 for the best DeepGEMM performance\n");
        return {major, minor};
    }

public:
    NVCCCompiler() {
        // Override the compiler signature
        nvcc_path = cuda_home / "bin" / "nvcc";
        if (const auto& env_nvcc_path = get_env<std::string>("DG_JIT_NVCC_COMPILER"); not env_nvcc_path.empty())
            nvcc_path = env_nvcc_path;
        const auto& [nvcc_major, nvcc_minor] = get_nvcc_version();
        signature = fmt::format("NVCC{}.{}", nvcc_major, nvcc_minor);

        const auto& arch = 89;//device_runtime->get_arch(false, nvcc_major > 12 or nvcc_minor >= 9);

        auto arch_flag = "";
        if (is_ppu1v5_device()) {
            arch_flag = "-gencode=arch=compute_89,code=sm_89";
            flags = fmt::format("{} -I{}/cutlass3 -I{}/deep_gemm {} "
                            " -O3 -fconcepts  -Wno-deprecated-declarations  -Wno-abi "
                            "-cubin --expt-relaxed-constexpr --expt-extended-lambda ",
                            flags, library_include_path.c_str(), library_include_path.c_str(), arch_flag);
        } else {
            arch_flag = "-gencode=arch=compute_80a,code=sm_80a";
            flags = fmt::format("{} -I{} -I{}/cutlass -I{}/deep_gemm {} "
                            " -O3  -fconcepts  -Wno-deprecated-declarations  -Wno-abi "
                            "-cubin --expt-relaxed-constexpr --expt-extended-lambda ",
                            flags, library_include_path.c_str(), library_include_path.c_str(), library_include_path.c_str(), arch_flag);
        }
        std::string nvcc_flags;
        if (is_ppu1v5_device()) {
            flags += " -ppu-patch-fence-ppu=false -wno-loop-miss-transform"
                     " -ppu-cg-to-kp1=true -ppu-fix-uninit=true"
                     " --ptxas-options=--register-usage-level=10";
        }
    }

    void compile(const std::string &code, const std::filesystem::path& dir_path, const std::filesystem::path &cubin_path, const std::string& name, int32_t thread_num, int32_t smem_size) const override {
        // Write the code into the cache directory
        const auto& code_path = dir_path / "kernel.cu";
        put(code_path, code);

        // Per-kernel flags: warp-interleaving kernels (gemm_fp8, mqa_logits) use -mllvm flags,
        // others only need -ppu-simt-branch=false (aligned with compiler.py logic)
        std::string per_kernel_flags;
        if (is_ppu1v5_device()) {
            const bool use_warp_interleaving = (name.find("fp8_grouped_deep_gemm") != std::string::npos) ||
                                               (name.find("fp8_deep_gemm") != std::string::npos) ||
                                               (name.find("mqa_logits") != std::string::npos);
            if (!use_warp_interleaving) {
                per_kernel_flags = " -ppu-simt-branch=false";
            } else {
                per_kernel_flags = " -mllvm -ppu-blksync-nb-schedule-boundary=true"
                                   " -mllvm -ppu-simt-branch=false"
                                   " -mllvm -ppu-adjust-tsm-valu-war=13"
                                   " -mllvm -ppu-reassign-subregs=true"
                                   " -mllvm -ppu-pref-fma-reuse=true"
                                   " -mllvm -ppu-pref-mma-reuse=true"
                                   " -mllvm -regalloc=pbqp";
            }
        }
        // Compile
        const auto& command = fmt::format("{} {} -o {} {}{}", nvcc_path.c_str(), code_path.c_str(), cubin_path.c_str(), flags, per_kernel_flags);
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PRINT_COMPILER_COMMAND", 0))
            printf("Running NVCC command: %s\n", command.c_str());
        const auto& [return_code, output] = call_external_command(command);
        // printf("return_code %s\n", return_code.c_str());
        if (return_code != 0) {
            printf("NVCC compilation failed: %s\n", output.c_str());
            DG_HOST_ASSERT(false and "NVCC compilation failed");
        }

        // Check local memory usage
        if (get_env("DG_JIT_PTXAS_CHECK", 0))
            DG_HOST_ASSERT(not std::regex_search(output, std::regex(R"(Local memory used)")));

        // Print PTXAS log
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0))
            printf("%s", output.c_str());
    }
};

class NVRTCCompiler final: public Compiler {
public:
    class RtcOptions {
        public:
        RtcOptions() = delete;
        RtcOptions(acArch_t arch, bool use_cutlass3) {
        #if defined(ACOMPUTE_VERSION) && ACOMPUTE_VERSION >= 10700
            std::string fp8_promotion = "-DUSELESS_MACRO";
            if (getenv("CLOSE_FP8_PROMOTION") != nullptr) {
            fp8_promotion = "-DCLOSE_FP8_PROMOTION";
            }
            std::string use_mma_k48 = "-DUSELESS_MACRO";
            if (getenv("USE_MMA_K48") != nullptr) {
            use_mma_k48 = "-DUSE_MMA_K48";
            }
        #endif
            opts = {
            "--device-as-default-execution-space",
            "-DHGGC_COMPILER_WRAPPER_MODE",
            // "-U__linux__",
            // "-D__CUDACC_RTC__",
            "-D__CUDACC__",
            };
            auto opts_insert = [this](const std::vector<std::string> & vb) {
                opts.insert(opts.end(), vb.begin(), vb.end());
            };
            auto includes_insert = [this](const std::vector<std::string> & paths) {
            for (auto & path: paths) {
                opts.emplace_back("--include-path=" + path);
            }
            };
        // #ifndef __HGGCCC__
        // #if defined(ACOMPUTE_VERSION) && ACOMPUTE_VERSION == 10700
        //     opts_insert({"--gpu-architecture=compute_90a"});
        // #elif defined(ACOMPUTE_VERSION) && ACOMPUTE_VERSION == 20000
        //     opts_insert({"--gpu-architecture=compute_100a"});
        // #else
        //     opts_insert({"--gpu-architecture=compute_80"});
        // #endif
            if (getenv("CUDA_HOME") == nullptr) {
              printf("No CUDA_HOME exist\n");
            }
            std::string cuda_home = std::string(getenv("CUDA_HOME")) + "/include";

        #if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
            std::string cuda_home1 = std::string(getenv("CUDA_HOME")) + "/include/cccl";
            includes_insert({cuda_home, cuda_home1, cuda_home1 + "/cuda/std"});
        #else
            includes_insert({cuda_home, cuda_home + "/cuda/std", cuda_home + "/../targets/x86_64-linux/include/thrust/system/cuda"});
        #endif
            std::string library_include_path_str = fmt::format("{}", library_include_path.c_str());
            if (arch == AC_PPU0010) {
                includes_insert({library_include_path_str+ "/cutlass", library_include_path_str + "/deep_gemm", library_include_path_str});
            } else {
                includes_insert({library_include_path_str+ "/cutlass3", library_include_path_str + "/deep_gemm"});
            }
        // #else
            opts_insert({"-DNDEBUG", "-DUSE_CLANG", "-no-cache"});
            if (arch == AC_PPU0010) {
                opts_insert({
                    "-DACOMPUTE_VERSION=10000",
                });
            } else if (arch == AC_PPU0015) {
                opts_insert({
                    "--ppu-arch=ppu0015",
                    "-DACOMPUTE_VERSION=10500",
                    "--ppu-tuning-options=-ppu-patch-fence-ppu=false",
                    "--ppu-tuning-options=-wno-loop-miss-transform",
                    "--ppu-tuning-options=-ppu-simt-branch=false",
                    "--ppu-tuning-options=-ppu-cg-to-kp1=true",
                    "--ppu-tuning-options=-ppu-fix-uninit=true",
                    "--ppu-tuning-options=-ppu-blksync-nb-schedule-boundary=true",
                    "--ppu-tuning-options=-ppu-adjust-tsm-valu-war=13",
                    "--ppu-tuning-options=-ppu-reassign-subregs=true",
                    "--ppu-tuning-options=-ppu-pref-fma-reuse=true",
                    "--ppu-tuning-options=-ppu-force-defer-sync=true",
                    "--ppu-tuning-options=-ppu-pref-mma-reuse=true",
                    "--ppu-tuning-options=-regalloc=pbqp",
                });
            } else if (arch == AC_PPU0017) {
                opts_insert({
                    #ifdef ClusterGroupSync
                    "-DClusterGroupSync",
                    #endif
                    #if defined(ACOMPUTE_VERSION) && ACOMPUTE_VERSION == 10700
                    fp8_promotion.c_str(),
                    use_mma_k48.c_str(),
                    #endif
                    "--gpu-architecture=compute_90a",
                    "-DACOMPUTE_VERSION=10700",
                    "--ppu-arch=ppu0017",
                    "--ppu-tuning-options=-ppu-patch-fence-ppu=false",
                    "--ppu-tuning-options=-wno-loop-miss-transform",
                    "--ppu-tuning-options=-ppu-simt-branch=false",
                    "--ppu-tuning-options=-ppu-cg-to-kp1=true",
                    "--ppu-tuning-options=-ppu-fix-uninit=true",
                    "-DHGGC_COMPILER_WRAPPER_MODE",
                    // "--ppu-tuning-options=-ppu-force-defer-sync=true",
                });
            }
        // #endif

            std::string standard = "--std=c++17";
            opts.emplace_back(standard);

            opts_char.resize(opts.size());
            std::transform(opts.begin(), opts.end(), opts_char.begin(), [](const std::string& s) { return s.c_str(); });

            // std::cout << "--- NVRTC Options Debug ---" << std::endl;
            // for (size_t i = 0; i < opts.size(); ++i) {
            //     std::cout << "  [" << i << "] '" << opts[i] << "'" << std::endl;
            // }
            // std::cout << "---------------------------" << std::endl;

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

    NVRTCCompiler() {
        // Override the compiler signature
        int major, minor;
        DG_NVRTC_CHECK(nvrtcVersion(&major, &minor));
        signature = fmt::format("NVRTC{}.{}", major, minor);
        DG_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVRTC version should be >= 12.3");

    }

    void compile(const std::string &code, const std::filesystem::path& dir_path, const std::filesystem::path &cubin_path, const std::string& name, int32_t thread_num, int32_t smem_size) const override {
        // Write the code into the cache directory
        const auto& code_path = dir_path / "kernel.cu";
        put(code_path, code);
        acArch_t arch = AC_PPU0010;
        bool use_cutlass3 = false;
        if (is_ppu1v5_device()) {
            arch = AC_PPU0015;
            use_cutlass3 = true;
        }
        RtcOptions opts(arch, use_cutlass3);
        // Create NVRTC program and compile
        nvrtcProgram program;
        DG_NVRTC_CHECK(nvrtcCreateProgram(&program, code.c_str(), "kernel.cu", 0, nullptr, nullptr));
        const auto& compile_result = nvrtcCompileProgram(program, opts.size(), opts.data());

        // Get and print compiler log
        size_t log_size;
        DG_NVRTC_CHECK(nvrtcGetProgramLogSize(program, &log_size));
        if (get_env<int>("DG_JIT_DEBUG", 0) or compile_result != NVRTC_SUCCESS) {
            if (compile_result != NVRTC_SUCCESS)
                DG_HOST_ASSERT(log_size > 1);
            if (log_size > 1) {
                std::string compilation_log(log_size, '\0');
                DG_NVRTC_CHECK(nvrtcGetProgramLog(program, compilation_log.data()));
                printf("NVRTC log: %s\n", compilation_log.c_str());
            }
        }

        // Get CUBIN size and data
        size_t cubin_size;
        DG_NVRTC_CHECK(nvrtcGetCUBINSize(program, &cubin_size));
        std::string cubin_data(cubin_size, '\0');
        DG_NVRTC_CHECK(nvrtcGetCUBIN(program, cubin_data.data()));

        // Write into the file system
        put(cubin_path, cubin_data);
        // Cleanup
        DG_NVRTC_CHECK(nvrtcDestroyProgram(&program));

        CUmodule module;
        cuModuleLoadData(&module, cubin_data.data());

        CUfunction kernel_func;
        cuModuleGetFunction(&kernel_func, module, name.c_str());

        CUresult result = cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_cu,
        kernel_func,
        thread_num,
        smem_size
        );

        if (result != CUDA_SUCCESS) {
            printf("Get Max active blocks per SM failed!\n");
        }
    }
};

static auto compiler = LazyInit<Compiler>([]() -> std::shared_ptr<Compiler> {
    if (get_env<int>("DG_JIT_USE_NVRTC", 0)) {
        return std::make_shared<NVRTCCompiler>();
    } else {
        return std::make_shared<NVCCCompiler>();
    }
});

} // namespace deep_gemm
