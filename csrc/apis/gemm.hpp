#pragma once

#include "../utils/compatibility.hpp"
#include "../utils/layout.hpp"
#include <torch/extension.h>
#include <ATen/MemoryOverlap.h>
#include "../jit_kernels/impls/fp8_gemm.hpp"
#include "../jit_kernels/impls/ppuxx_layout.hpp"

#include "../jit_kernels/impls/bf16_gemm.hpp"
#include "../jit_kernels/impls/fp32_gemm.hpp"
#include "../jit_kernels/impls/int8_gemm.hpp"
#include "../jit_kernels/impls/m_grouped_bf16_gemm.hpp"
#include "../jit_kernels/impls/m_grouped_fp8_gemm.hpp"
#include "../jit_kernels/impls/m_grouped_int8_gemm.hpp"
// #include "layout.hpp"
#include "../jit_kernels/impls/fp4_gemm.hpp"
#include "../jit_kernels/impls/m_grouped_fp4_gemm.hpp"
#include "../jit_kernels/impls/fused_moe_gemm.hpp"
#include "../jit_kernels/impls/moe_align.hpp"
#include "../jit_kernels/impls/m_grouped_w4a16_gemm.hpp"
#include "../jit_kernels/impls/m_grouped_w4a8_gemm.hpp"
#include "../jit_kernels/impls/ppuxx_acblaslt.hpp"

namespace deep_gemm::gemm {
using ConfigTuple = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;
// The 9-tuple returned by `w4a16_get_best_configs`:
// (num_sms, block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages, n_expand)
using W4A16ConfigTuple = deep_gemm_w4a16_common::W4A16ConfigTuple;
using W4A8ConfigTuple = deep_gemm::w4fa8::W4A8ConfigTuple;

static bool gemm_disjoint(const torch::Tensor& a, const torch::Tensor& b) {
    if (!a.numel() || !b.numel() || !a.is_alias_of(b) ||
        at::get_overlap_status(a, b) == at::MemOverlapStatus::No) return true;
    // TooHard is not proof of overlap. Bound positive-stride views in bytes
    // relative to their shared storage; holes inside intersecting bounds remain
    // conservatively rejected. Storage bounds also prevent arithmetic overflow.
    const auto range = [](const torch::Tensor& t) -> std::optional<std::pair<size_t, size_t>> {
        const size_t bytes = t.element_size(), limit = t.storage().nbytes() / bytes;
        if (t.storage_offset() < 0 || size_t(t.storage_offset()) >= limit) return std::nullopt;
        size_t last = t.storage_offset();
        for (int dim = 0; dim < t.dim(); ++dim) {
            if (t.stride(dim) < 0) return std::nullopt;
            const size_t stride = t.stride(dim), count = t.size(dim) - 1;
            if (stride && count > (limit - 1 - last) / stride) return std::nullopt;
            last += count * stride;
        }
        return std::make_pair(size_t(t.storage_offset()) * bytes, (last + 1) * bytes);
    };
    const auto x = range(a), y = range(b);
    return x && y && (x->second <= y->first || y->second <= x->first);
}

static void check_gemm_device_and_output(const torch::Tensor& a, const torch::Tensor& b,
                                         const torch::Tensor& d) {
    TORCH_CHECK(a.is_cuda() && b.device() == a.device() && d.device() == a.device(),
                "GEMM tensors must be on the same accelerator");
    int device;
    DG_HGGC_RUNTIME_CHECK(hggcGetDevice(&device));
    TORCH_CHECK(a.get_device() == device, "GEMM tensors must be on the current accelerator");
    TORCH_CHECK(gemm_disjoint(d, a) && gemm_disjoint(d, b),
                "GEMM output must not overlap an input");
}

// Validate user tiles before packing, accumulation or JIT. In particular, zero
// warp dimensions must not reach divisions in the lower-level launchers.
static void check_bf16_gemm_config(const std::optional<ConfigTuple>& configs, int64_t k,
                                   int output_bytes, bool batched = false) {
    if (!configs || is_ppu1v5_device()) return;
    const auto [sms, bm, bn, bk, wm, wn, stages, unused] = *configs;
    const auto pow2 = [](int x) { return x > 0 && (x & (x - 1)) == 0; };
    TORCH_CHECK(sms > 0 && sms <= get_num_sms(), "BF16 config CU count exceeds the configured budget");
    TORCH_CHECK(bm >= 16 && bm <= 512 && pow2(bm) && bn >= 16 && bn <= 512 && pow2(bn) &&
                bk >= 32 && bk <= 256 && pow2(bk) && wm >= 16 && pow2(wm) &&
                wn >= 16 && pow2(wn) && bm % wm == 0 && bn % wn == 0 &&
                stages >= 2 && stages <= 8, "invalid BF16 config tile or warp dimensions");
    TORCH_CHECK(k >= 64 && k % 32 == 0 && k / bk >= stages && (!batched || k % bk == 0),
                "BF16 config requires complete prologue K tiles (K>=64, K%32=0)");
    const int64_t threads = int64_t(bm / wm) * (bn / wn) * 32;
    const int64_t smem = std::max(int64_t(bm + bn) * bk * 2 * stages, int64_t(bm) * bn * output_bytes);
    const auto prop = device_runtime->get_prop();
    TORCH_CHECK(threads <= prop->maxThreadsPerBlock && smem <= prop->sharedMemPerBlock,
                "BF16 config exceeds device thread or shared-memory limits");
}

extern "C" {
static bool early_return(const int& m, const int &n, const int& k,
                         const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    // Do nothing if the problem is empty
    if (m == 0 or n == 0)
        return true;

    // Checks
    const bool is_cd_same = c.has_value() and c->data_ptr() == d.data_ptr();
    if (is_cd_same)
        DG_HOST_ASSERT(c->sizes() == d.sizes() and c->strides() == d.strides());
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    if (c.has_value()) {
        check_major_type_cd(c.value());
        DG_HOST_ASSERT(d.scalar_type() == c.value().scalar_type());
    }

    // No accumulation
    if (k == 0) {
        if (not is_cd_same)
            c.has_value() ? d.copy_(c.value()) : d.zero_();
        return true;
    }

    // With accumulation, do copy before GEMM (assuming the GEMM kernel does not support different C/D)
    if (c.has_value() and not is_cd_same)
        d.copy_(c.value());
    return false;
}

void bf16_gemm_nt(const torch::Tensor& a,
                  const torch::Tensor& b,
                  const torch::Tensor& d,
                  const std::optional<torch::Tensor>& c,
                  const std::string& compiled_dims,
                  std::optional<ConfigTuple> configs = std::nullopt) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && d.dim() == 2, "BF16 dense GEMM requires rank-2 tensors");
    check_gemm_device_and_output(a, b, d);
    // Host tile/grid calculations add up to 511 before division in int32.
    for (auto size : {a.size(0), a.size(1), b.size(0), b.size(1), d.size(0), d.size(1)})
        TORCH_CHECK(size <= std::numeric_limits<int>::max() - 511, "BF16 dense GEMM dimensions exceed int32 tile range");
    TORCH_CHECK(a.size(1) <= std::numeric_limits<int>::max() - 4096,
                "BF16 dense GEMM K exceeds the int32 K-loop range");
    const auto& [m, k] = get_shape<2>(a);
    const auto& [n, k_] = get_shape<2>(b);
    const auto& [m_, n_] = get_shape<2>(d);

    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "lhs must be BF16");
    TORCH_CHECK(b.scalar_type() == torch::kBFloat16, "rhs must be BF16");
    TORCH_CHECK(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat,
                "out must be BF16 or FP32");
    TORCH_CHECK(a.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    // Explicit tiles override the external LUT; both take precedence over
    // automatic GEMV, cute-free and generalized dispatch on 810E. Keep the
    // placeholder smem tuple until the tile passes the same checks as configs.
    if (!configs && m && n && k && model_810e_dispatch_device()) {
        if (auto tile = deep_gemm_lut::get_runtime_config(
                m, n, k, 1, GemmType::Normal, deep_gemm_lut::RuntimeLutKind::BF16)) {
            auto [bm, bn, bk, wm, wn, stages] = *tile;
            configs = std::make_tuple(get_num_sms(), bm, bn, bk, wm, wn, stages,
                                      std::make_tuple(0, 0, 0));
        }
    }
    const bool cute_free = m && n && k && model_810e_cute_free_requested();
    if (m && n && k && !cute_free) check_bf16_gemm_config(configs, k, d.element_size());
    if (configs && !cute_free && !is_ppu1v5_device() && m && n && k)
        TORCH_CHECK(n % (16 / d.element_size()) == 0 && (reinterpret_cast<uintptr_t>(d.data_ptr()) & 15) == 0,
                    "BF16 dense explicit/LUT configs require vector-aligned output");
    if (c.has_value()) {
        TORCH_CHECK(c->sizes() == d.sizes(), "C and D must have the same shape");
        TORCH_CHECK(c->scalar_type() == d.scalar_type(), "C and D must have the same dtype");
        TORCH_CHECK(c->device() == d.device(), "C and D must be on the same device");
        check_major_type_cd(c.value());
        TORCH_CHECK(c->data_ptr() != d.data_ptr() || c->strides() == d.strides(),
                    "C and D sharing the same address must have the same strides");
        TORCH_CHECK(c->data_ptr() == d.data_ptr() || gemm_disjoint(d, *c),
                    "BF16 GEMM C/D must be disjoint or exactly aliased");
        if (m != 0 && n != 0 && k != 0) {
            // BF16 product kernels (including GEMV and split-K) have beta=0.
            // Keep C alive until the FP32 product is complete, including C == D,
            // and add it before the only conversion to the requested output dtype.
            // Calls without C retain their existing dispatch and allocations.
            auto accum = torch::empty({m, n}, d.options().dtype(torch::kFloat));
            bf16_gemm_nt(a, b, accum, std::nullopt, compiled_dims, configs);
            // The product kernel is cached but leaves accum unwritten in these modes.
            if (is_compile_only()) return;
            accum.add_(c.value());
            d.copy_(accum);
            return;
        }
    }
    if (k == 0 && is_compile_only()) return;
    if (early_return(m, n, k, d, c)) return;
    if (model_810e_cute_free(a, b, d, configs)) return;
    if (n == 1 && !configs.has_value() && model_810e_bf16_n1(a, b, d)) return;
    if (!configs.has_value() && model_810e_dispatch_dense(a, b, d)) return;
    TORCH_CHECK(is_ppu1v5_device() || n % 2 == 0, "BF16 dense GEMM on PPU1.0 requires even N, got ", n);
    if (!is_ppu1v5_device()) {
        TORCH_CHECK(k >= 64 && k % 32 == 0 &&
                    (reinterpret_cast<uintptr_t>(a.data_ptr()) & 15) == 0 &&
                    (reinterpret_cast<uintptr_t>(b.data_ptr()) & 15) == 0,
                    "BF16 tiled GEMM requires K>=64, K%32=0 and 16-byte-aligned inputs");
        TORCH_CHECK(n % (16 / d.element_size()) == 0 && (reinterpret_cast<uintptr_t>(d.data_ptr()) & 15) == 0,
                    "BF16 tiled GEMM requires vector-aligned output");
        if (!configs) {
            configs = deep_gemm_bf16_common::get_best_configs(m, n, k, 1, get_num_sms());
            // PPU 1.0 MMA spans 16 N columns per warp. WN=8 oversteps the
            // shared tile even though it compiles. Short K also needs two
            // complete prologue tiles in the budget-aware fallback.
            std::get<5>(*configs) = std::max(16, std::get<5>(*configs));
            auto& bk = std::get<3>(*configs);
            if (k / bk < 2) bk = 32;
            std::get<6>(*configs) = std::min(std::get<6>(*configs), k / bk);
        }
        check_bf16_gemm_config(configs, k, d.element_size());
    }
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 8) {
        bf16_gemm(a, b, d, m, n, k, configs);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

void gemm_fp32_fp32_fp32_nt(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && d.dim() == 2, "FP32 GEMM requires rank-2 tensors");
    TORCH_CHECK(a.scalar_type() == torch::kFloat && b.scalar_type() == torch::kFloat &&
                d.scalar_type() == torch::kFloat, "FP32 GEMM requires FP32 inputs and output");
    check_gemm_device_and_output(a, b, d);
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && d.is_contiguous(),
                "FP32 GEMM requires contiguous tensors");
    TORCH_CHECK(a.size(1) == b.size(1) && d.size(0) == a.size(0) && d.size(1) == b.size(0),
                "FP32 GEMM expects A[M,K], B[N,K], D[M,N]");
    if (a.size(0) == 0 || b.size(0) == 0) return;
    if (a.size(1) == 0) { if (!is_compile_only()) d.zero_(); return; }
    fp32_gemm(a, b, d);
}

void bf16_gemm_strided_batched_nt(const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& d, std::optional<ConfigTuple> configs = std::nullopt) {
    if (model_810e_check_batch(a, b, d)) return;
    check_gemm_device_and_output(a, b, d);
    if (a.size(2) == 0) { if (!is_compile_only()) d.zero_(); return; }
    check_bf16_gemm_config(configs, a.size(2), d.element_size(), true);
    if (!model_810e_batch_native_input(a) || !model_810e_batch_native_input(b)) {
        if (!configs && model_810e_dispatch_strided_batch(a, b, d)) return;
        const int sms = configs ? std::get<0>(*configs) : get_num_sms();
        const auto packed_a = model_810e_pack_batch_input(a, sms);
        const auto packed_b = model_810e_pack_batch_input(b, sms);
        bf16_gemm_strided_batched_nt(packed_a, packed_b, d, configs);
        return;
    }
    if (!configs.has_value() && model_810e_dispatch_batch(a, b, d)) return;
    if (!configs.has_value() && a.size(2) == 128) {
        // Short-K batches are output-bound at large M. Use the coalesced TSM
        // epilogue; its existing alignment gate retains guarded direct stores.
        model_810e_bf16_strided_batched_tsm(a, b, d, configs);
        return;
    }
    model_810e_bf16_strided_batched(a, b, d, configs);
}

void bf16_gemm_nn(const torch::Tensor& a,
                  const torch::Tensor& b,
                  const torch::Tensor& d,
                  const std::optional<torch::Tensor>& c,
                  const std::string& compiled_dims,
                  std::optional<ConfigTuple> configs = std::nullopt) {
    DG_HOST_UNREACHABLE("BF16 NN layout is not supported: kernel does not support non-NT layouts due to performance concerns");
}

void bf16_gemm_tn(const torch::Tensor& a,
                  const torch::Tensor& b,
                  const torch::Tensor& d,
                  const std::optional<torch::Tensor>& c,
                  const std::string& compiled_dims,
                  std::optional<ConfigTuple> configs = std::nullopt) {
    DG_HOST_UNREACHABLE("BF16 TN layout is not supported: kernel does not support non-NT layouts due to performance concerns");
}

void bf16_gemm_tt(const torch::Tensor& a,
                  const torch::Tensor& b,
                  const torch::Tensor& d,
                  const std::optional<torch::Tensor>& c,
                  const std::string& compiled_dims,
                  std::optional<ConfigTuple> configs = std::nullopt) {
    DG_HOST_UNREACHABLE("BF16 TT layout is not supported: kernel does not support non-NT layouts due to performance concerns");
}

void int8_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                  const std::pair<torch::Tensor, torch::Tensor>& b,
                  const torch::Tensor& d,
                  const std::optional<torch::Tensor>& c,
                  const std::string& compiled_dims,
                  std::optional<ConfigTuple> configs = std::nullopt) {
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    check_major_type_cd(d);

    const auto& [m, k] = get_shape<2>(a.first);
    const auto& [n, k_] = get_shape<2>(b.first);
    const auto& [m_, n_] = get_shape<2>(d);

    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT((a.first.scalar_type() == torch::kInt8) || (a.first.scalar_type() == torch::kFloat8_e4m3fn));
    DG_HOST_ASSERT((b.first.scalar_type() == torch::kInt8) || (b.first.scalar_type() == torch::kFloat8_e4m3fn));
    DG_HOST_ASSERT(a.first.scalar_type() == b.first.scalar_type());
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT((a.second.size(0) == m) && (a.second.scalar_type() == torch::kFloat32));
    DG_HOST_ASSERT((b.second.size(0) == n) && (b.second.scalar_type() == torch::kFloat32));
    TORCH_CHECK(a.first.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(b.first.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    if (early_return(m, n, k, d, c)) return;
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 8) {
        int8_gemm(a.first, a.second, b.first, b.second, d, m, n, k, configs);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

void fp8_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                 const std::pair<torch::Tensor, torch::Tensor>& b,
                 const torch::Tensor& d,
                 const std::optional<torch::Tensor>& c,
                 const std::optional<std::tuple<int, int, int>>& recipe,
                 const std::optional<std::tuple<int, int>>& recipe_a,
                 const std::optional<std::tuple<int, int>>& recipe_b,
                 const std::string& compiled_dims,
                 const bool disable_ue8m0_cast,
                 std::optional<ConfigTuple> configs = std::nullopt) {
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    if (fp8_requires_k_major()) {
        DG_HOST_ASSERT(major_a == MajorType::K);
        DG_HOST_ASSERT(major_b == MajorType::K);
    }

    check_major_type_cd(d);

    const auto& lhs_scales = a.second;
    const auto& rhs_scales = b.second;
    // Type and shape checks
    const auto& [m, k] = get_shape<2>(a.first);
    const auto& [n, k_] = get_shape<2>(b.first);
    const auto& [m_, n_] = get_shape<2>(d);

    if ((lhs_scales.sizes() == std::vector<int64_t>{m, 1}) && (rhs_scales.sizes() == std::vector<int64_t>{n, 1})) {
        const auto arch_major = device_runtime->get_arch_major();
        if (arch_major == 8) {
            return int8_gemm_nt(a, b, d, c, compiled_dims, configs);
        } else {
            DG_HOST_UNREACHABLE("Unsupported architecture");
        }
    }
    // DG_HOST_ASSERT(k % 128 == 0);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT((lhs_scales.sizes() == std::vector<int64_t>{m, (k + 127) / 128}));
    DG_HOST_ASSERT((rhs_scales.sizes() == std::vector<int64_t>{(n + 127) / 128, (k + 127) / 128}));
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(a.first.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(b.first.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(rhs_scales.is_contiguous(), "rhs_scales must be contiguous");
    if (early_return(m, n, k, d, c)) return;
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 8) {
        fp8_gemm(a.first, a.second, b.first, b.second, d, m, n, k, configs);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

void fp8_gemm_nn(const std::pair<torch::Tensor, torch::Tensor>& a,
                 const std::pair<torch::Tensor, torch::Tensor>& b,
                 const torch::Tensor& d,
                 const std::optional<torch::Tensor>& c,
                 const std::optional<std::tuple<int, int, int>>& recipe,
                 const std::optional<std::tuple<int, int>>& recipe_a,
                 const std::optional<std::tuple<int, int>>& recipe_b,
                 const std::string& compiled_dims,
                 const bool disable_ue8m0_cast,
                 std::optional<ConfigTuple> configs = std::nullopt) {
    DG_HOST_UNREACHABLE("FP8 NN layout is not supported: kernel does not support non-NT layouts due to performance concerns");
}

void fp8_gemm_tn(const std::pair<torch::Tensor, torch::Tensor>& a,
                 const std::pair<torch::Tensor, torch::Tensor>& b,
                 const torch::Tensor& d,
                 const std::optional<torch::Tensor>& c,
                 const std::optional<std::tuple<int, int, int>>& recipe,
                 const std::optional<std::tuple<int, int>>& recipe_a,
                 const std::optional<std::tuple<int, int>>& recipe_b,
                 const std::string& compiled_dims,
                 const bool disable_ue8m0_cast,
                 std::optional<ConfigTuple> configs = std::nullopt) {
    DG_HOST_UNREACHABLE("FP8 TN layout is not supported: kernel does not support non-NT layouts due to performance concerns");
}

void fp8_gemm_tt(const std::pair<torch::Tensor, torch::Tensor>& a,
                 const std::pair<torch::Tensor, torch::Tensor>& b,
                 const torch::Tensor& d,
                 const std::optional<torch::Tensor>& c,
                 const std::optional<std::tuple<int, int, int>>& recipe,
                 const std::optional<std::tuple<int, int>>& recipe_a,
                 const std::optional<std::tuple<int, int>>& recipe_b,
                 const std::string& compiled_dims,
                 const bool disable_ue8m0_cast,
                 std::optional<ConfigTuple> configs = std::nullopt) {
    DG_HOST_UNREACHABLE("FP8 TT layout is not supported: kernel does not support non-NT layouts due to performance concerns");
}

void fp4_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                 const std::pair<torch::Tensor, torch::Tensor>& b,
                 const std::optional<torch::Tensor>& bias,
                 const torch::Tensor& d,
                 const std::optional<torch::Tensor>& c,
                 const std::optional<std::tuple<int, int, int>>& recipe,
                 const std::optional<std::tuple<int, int>>& recipe_a,
                 const std::optional<std::tuple<int, int>>& recipe_b,
                 const std::string& compiled_dims,
                 const bool disable_ue8m0_cast,
                 std::optional<ConfigTuple> configs = std::nullopt) {
    const auto& lhs = a.first;
    const auto& lhs_scales = a.second;
    const auto& rhs = b.first;
    const auto& rhs_scales = b.second;
    const auto major_a = get_major_type_ab(lhs);
    const auto major_b = get_major_type_ab(rhs);
    if (fp8_requires_k_major()) {
        DG_HOST_ASSERT(major_a == MajorType::K);
        DG_HOST_ASSERT(major_b == MajorType::K);
    }
    check_major_type_cd(d);
    const auto& [m, k] = get_shape<2>(lhs);
    const auto& [n, k_] = get_shape<2>(rhs);
    const auto& [m_, n_] = get_shape<2>(d);

    // Type and shape checks
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT(lhs.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    // Scales carry mxfp4 e8m0 scales: either raw uint8 (pre-preprocess) or uint16 (post-preprocess).
    DG_HOST_ASSERT(lhs_scales.scalar_type() == torch::kUInt16 || lhs_scales.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(rhs_scales.scalar_type() == torch::kUInt16 || rhs_scales.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(lhs_scales.dim() == 2);
    DG_HOST_ASSERT(rhs_scales.dim() == 2);
    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");

    // --- Scales layout check & preprocess ---
    torch::Tensor lhs_scales_t = lhs_scales;
    torch::Tensor rhs_scales_t = rhs_scales;
    if (!deep_gemm_fp4_common::check_mxfp4_scales_layout(lhs_scales_t)) {
        lhs_scales_t = deep_gemm_fp4_common::preprocess_mxfp4_scales(lhs_scales_t);
    }
    if (!deep_gemm_fp4_common::check_mxfp4_scales_layout(rhs_scales_t)) {
        rhs_scales_t = deep_gemm_fp4_common::post_preprocess_mxfp4_scales(rhs_scales_t);
        if (!deep_gemm_fp4_common::check_mxfp4_scales_layout(rhs_scales_t)) {
            rhs_scales_t = deep_gemm_fp4_common::preprocess_mxfp4_scales(rhs_scales_t);
        }
    }

    // Scale stride checks (validate post-preprocessing result)
    DG_HOST_ASSERT(deep_gemm_fp4_common::check_mxfp4_scales_layout(lhs_scales_t));
    DG_HOST_ASSERT(deep_gemm_fp4_common::check_mxfp4_scales_layout(rhs_scales_t));

    if (early_return(m, n, k, d, c)) return;

    // Handle bias - create empty tensor if not provided
    torch::Tensor bias_tensor;
    if (bias.has_value() && bias->defined() && bias->numel() > 0) {
        bias_tensor = *bias;
        DG_HOST_ASSERT(bias_tensor.scalar_type() == torch::kFloat32);
    } else {
        bias_tensor = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(lhs.device()));
    }

    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 8) {
        fp4_gemm(lhs, lhs_scales_t, rhs, rhs_scales_t, bias_tensor, d, m, n, k, configs);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

void m_grouped_int8_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                       const std::pair<torch::Tensor, torch::Tensor>& b,
                                       const torch::Tensor& d,
                                       const torch::Tensor& m_indices,
                                       const std::optional<std::tuple<int,int,int>>& recipe,
                                       const std::optional<std::tuple<int,int>>& recipe_a,
                                       const std::optional<std::tuple<int,int>>& recipe_b,
                                       const std::string& compiled_dims,
                                       const bool disable_ue8m0_cast,
                                       const bool use_psum_layout,
                                       const bool ensure_zero_padding,
                                       const std::optional<int> expected_m_for_psum_layout,
                                       std::optional<ConfigTuple> configs = std::nullopt) {
    const auto& lhs = a.first;
    const auto& lhs_scales = a.second;
    const auto& rhs = b.first;
    const auto& rhs_scales = b.second;

    const auto major_a = get_major_type_ab(lhs);
    const auto major_b = get_major_type_ab(rhs);
    DG_HOST_ASSERT(major_a == MajorType::K and major_b == MajorType::K);
    check_major_type_cd(d);

    const auto& [m, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_, n_] = get_shape<2>(d);
    int m__ = m_indices.numel();

    DG_HOST_ASSERT(m == m_ && m_ == m__ && k == k_ && n == n_);
    DG_HOST_ASSERT((lhs_scales.sizes() == std::vector<int64_t>{m, 1}));
    DG_HOST_ASSERT((rhs_scales.sizes() == std::vector<int64_t>{num_groups, n, 1}));

    DG_HOST_ASSERT(lhs.scalar_type() == torch::kInt8 || lhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kInt8 || rhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(lhs.scalar_type() == rhs.scalar_type());
    DG_HOST_ASSERT(lhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(rhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(m_indices.scalar_type() == torch::kInt32);

    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(m_indices.is_contiguous(), "m_indices must be contiguous");
    TORCH_CHECK(rhs_scales.is_contiguous(), "rhs_scales must be contiguous");

    if (m == 0) {
        return;
    }

    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 8) {
        m_grouped_gemm_int8_int8_bf16_nt_contiguous_impl(lhs, lhs_scales, rhs, rhs_scales, d, m_indices, m, n, k,
                                                         num_groups, configs);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

std::pair<int, int> m_grouped_int8_gemm_nt_masked(const std::pair<torch::Tensor, torch::Tensor>& a,
                                                  const std::pair<torch::Tensor, torch::Tensor>& b,
                                                  const torch::Tensor& d,
                                                  const torch::Tensor& masked_m,
                                                  int expected_m,
                                                  const std::optional<std::tuple<int,int,int>>& recipe,
                                                  const std::optional<std::tuple<int,int>>& recipe_a,
                                                  const std::optional<std::tuple<int,int>>& recipe_b,
                                                  const std::string& compiled_dims,
                                                  const bool disable_ue8m0_cast,
                                                  std::optional<int> max_block_n = 256,
                                                  std::optional<bool> enable_sbo_overlap = false,
                                                  std::optional<const torch::Tensor> signal = std::nullopt,
                                                  std::optional<ConfigTuple> configs = std::nullopt) {
    const auto& lhs = a.first;
    const auto& lhs_scales = a.second;
    const auto& rhs = b.first;
    const auto& rhs_scales = b.second;

    const auto major_a = get_major_type_ab(lhs);
    const auto major_b = get_major_type_ab(rhs);
    DG_HOST_ASSERT(major_a == MajorType::K and major_b == MajorType::K);
    check_major_type_cd(d);

    at::Tensor signal_tensor;
    if (signal.has_value() && signal->defined()) {
        signal_tensor = *signal;
    } else {
        signal_tensor = at::empty({0}, at::TensorOptions().dtype(at::kInt).device(d.device()));
    }

    const auto& [num_groups, m, k] = get_shape<3>(lhs);
    const auto& [num_groups_, n, k_] = get_shape<3>(rhs);
    const auto& [num_groups__, m_, n_] = get_shape<3>(d);
    int num_groups___ = masked_m.numel();

    // Type and shape checks
    DG_HOST_ASSERT(num_groups == num_groups_ && num_groups_ == num_groups__ && num_groups__ == num_groups___);
    DG_HOST_ASSERT(m == m_ && n == n_ && k == k_);
    DG_HOST_ASSERT(expected_m > 0 && m > 0 && n > 0 && k > 0 && num_groups > 0);
    DG_HOST_ASSERT((lhs_scales.sizes() == std::vector<int64_t>{num_groups, m, 1}));
    DG_HOST_ASSERT((rhs_scales.sizes() == std::vector<int64_t>{num_groups, n, 1}));

    DG_HOST_ASSERT(lhs.scalar_type() == torch::kInt8 || lhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kInt8 || rhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(lhs.scalar_type() == rhs.scalar_type());
    DG_HOST_ASSERT(lhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(rhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt32);

    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(masked_m.is_contiguous(), "masked_m must be contiguous");
    TORCH_CHECK(rhs_scales.is_contiguous(), "rhs_scales must be contiguous");

    if (enable_sbo_overlap.value_or(false)) {
        TORCH_CHECK(signal_tensor.defined(), "signal must be defined when enable_sbo_overlap is true");
        TORCH_CHECK(signal_tensor.is_contiguous(), "signal must be contiguous");
        TORCH_CHECK(signal_tensor.scalar_type() == torch::kInt32, "signal must be int32");
    }

    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 8) {
        return m_grouped_gemm_int8_int8_bf16_nt_masked_impl(lhs, lhs_scales, rhs, rhs_scales, d, masked_m, m, n, k,
                                                            num_groups, expected_m, configs, max_block_n.value_or(256),
                                                            enable_sbo_overlap.value_or(false), signal_tensor);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

void m_grouped_int8_gemm_nt_nopad(const std::pair<torch::Tensor, torch::Tensor>& a,
                                  const std::pair<torch::Tensor, torch::Tensor>& b,
                                  const torch::Tensor& d,
                                  const torch::Tensor& m_indices,
                                  std::optional<const torch::Tensor> m_rows = std::nullopt,
                                  const std::optional<std::tuple<int,int,int>>& recipe = std::nullopt,
                                  const std::optional<std::tuple<int,int>>& recipe_a = std::nullopt,
                                  const std::optional<std::tuple<int,int>>& recipe_b = std::nullopt,
                                  const std::string& compiled_dims = "nk",
                                  const bool disable_ue8m0_cast = false,
                                  const bool use_psum_layout = false,
                                  const bool ensure_zero_padding = true,
                                  const std::optional<int> expected_m_for_psum_layout = std::nullopt,
                                  std::optional<ConfigTuple> configs = std::nullopt) {
    const auto& lhs = a.first;
    const auto& lhs_scales = a.second;
    const auto& rhs = b.first;
    const auto& rhs_scales = b.second;

    const auto& [m, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_, n_] = get_shape<2>(d);
    int m__ = m_indices.numel();

    // Type and shape checks
    DG_HOST_ASSERT(m == m_ && m_ == m__ && k == k_ && n == n_);
    DG_HOST_ASSERT((lhs_scales.sizes() == std::vector<int64_t>{m, 1}));
    DG_HOST_ASSERT((rhs_scales.sizes() == std::vector<int64_t>{num_groups, n, 1}));

    DG_HOST_ASSERT(lhs.scalar_type() == torch::kInt8 || lhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kInt8 || rhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(lhs.scalar_type() == rhs.scalar_type());
    DG_HOST_ASSERT(lhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(rhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(m_indices.scalar_type() == torch::kInt32);

    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(m_indices.is_contiguous(), "m_indices must be contiguous");
    TORCH_CHECK(rhs_scales.is_contiguous(), "rhs_scales must be contiguous");

    if (m == 0) {
        return;
    }

    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 8) {
        m_grouped_gemm_int8_int8_bf16_nt_nopad_impl(lhs, lhs_scales, rhs, rhs_scales, d, m_indices, m, n, k, num_groups,
                                                    m_rows, configs);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void check_a8w4_int8_common(const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
                                   const torch::Tensor& rhs, const torch::Tensor& rhs_scales,
                                   const torch::Tensor& out, int num_groups, int n, int k) {
    TORCH_CHECK(lhs.dim() == 2 || lhs.dim() == 3, "A8W4 lhs must be 2D or 3D");
    TORCH_CHECK(rhs.dim() == 3, "A8W4 rhs must be 3D");
    TORCH_CHECK(out.dim() == lhs.dim(), "A8W4 output rank must match lhs rank");
    TORCH_CHECK(get_ppu_arch() == AC_PPU0010, "A8W4 INT8 is only supported on PPU0010");
    TORCH_CHECK(num_groups > 0 && n > 0 && k > 0, "A8W4 requires positive num_groups, N and K");
    // The weight permutation and the PPU0010 INT8 MMA operate on 32-element K groups. The
    // mainloop K tile is 64, but AIU padz safely zero-fills its second K32 atom on the tail tile.
    TORCH_CHECK(k % 32 == 0, "A8W4 INT8 requires K to be a multiple of 32");
    // The PPU0010 TSM epilogue stores eight BF16 values (16 bytes) per R2G vector and predicates
    // only the vector origin, so an N tail inside that vector would overwrite the next output row.
    TORCH_CHECK(n % 8 == 0, "A8W4 INT8 requires N to be a multiple of 8");
    TORCH_CHECK(lhs.scalar_type() == torch::kInt8, "A8W4 lhs must be int8");
    TORCH_CHECK(rhs.scalar_type() == torch::kUInt8,
                "A8W4 rhs must use portable packed signed-INT4 torch.uint8 storage");
    TORCH_CHECK(lhs_scales.scalar_type() == torch::kFloat32, "A8W4 lhs scales must be float32");
    TORCH_CHECK(rhs_scales.scalar_type() == torch::kFloat32, "A8W4 rhs scales must be float32");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "A8W4 output must be bfloat16");
    TORCH_CHECK(rhs.size(0) == num_groups && rhs.size(1) == n && rhs.size(2) * 2 == k,
                "A8W4 rhs must have shape [num_groups, n, k / 2]");
    TORCH_CHECK(rhs_scales.sizes() == std::vector<int64_t>({num_groups, n, 1}),
                "A8W4 rhs scales must have shape [num_groups, n, 1]");
    TORCH_CHECK(lhs.is_contiguous() && lhs_scales.is_contiguous() && rhs.is_contiguous() &&
                    rhs_scales.is_contiguous() && out.is_contiguous(),
                "A8W4 tensors must be contiguous");
}

// Returns (block_m, ceil_div(n, block_n)); the SBO-overlap signal check consumes both. With
// `enable_sbo_overlap` the caller owns `signal`: int32, zeroed before every call, and large enough
// for num_groups * ceil_div(m, block_m) entries, since the kernel strides it by the selected BLOCK_M.
std::pair<int, int> m_grouped_gemm_int8_int4_bf16_nt_masked(
    const std::pair<torch::Tensor, torch::Tensor>& a,
    const std::pair<torch::Tensor, torch::Tensor>& b,
    const torch::Tensor& out, const torch::Tensor& masked_m, int expected_m,
    std::optional<int> max_block_n = 256,
    std::optional<bool> enable_sbo_overlap = false, std::optional<const torch::Tensor> signal = std::nullopt,
    std::optional<ConfigTuple> configs = std::nullopt) {
    const auto& [num_groups, m, k] = get_shape<3>(a.first);
    const int n = b.first.size(1);
    check_a8w4_int8_common(a.first, a.second, b.first, b.second, out, num_groups, n, k);
    TORCH_CHECK(out.sizes() == std::vector<int64_t>({num_groups, m, n}),
                "A8W4 output must have shape [num_groups, m, n]");
    TORCH_CHECK(a.second.sizes() == std::vector<int64_t>({num_groups, m, 1}),
                "A8W4 lhs scales must have shape [num_groups, m, 1]");
    TORCH_CHECK(masked_m.scalar_type() == torch::kInt32 && masked_m.is_contiguous() &&
                    masked_m.numel() == num_groups,
                "A8W4 masked_m must be contiguous int32 with num_groups elements");
    TORCH_CHECK(expected_m > 0 && m > 0, "A8W4 expected_m and m must be positive");

    at::Tensor signal_tensor;
    if (signal.has_value() && signal->defined()) {
        signal_tensor = *signal;
    } else {
        signal_tensor = at::empty({0}, at::TensorOptions().dtype(at::kInt).device(out.device()));
    }

    if (enable_sbo_overlap.value_or(false)) {
        TORCH_CHECK(signal_tensor.defined(), "signal must be defined when enable_sbo_overlap is true");
        TORCH_CHECK(signal_tensor.is_contiguous(), "signal must be contiguous");
        TORCH_CHECK(signal_tensor.scalar_type() == torch::kInt32, "signal must be int32");
    }

    return m_grouped_gemm_int8_int4_bf16_nt_masked_impl(
        a.first, a.second, b.first, b.second, out, masked_m, m, n, k, num_groups,
        expected_m, configs, max_block_n.value_or(256), enable_sbo_overlap.value_or(false), signal_tensor);
}

void m_grouped_gemm_int8_int4_bf16_nt_nopad(
    const std::pair<torch::Tensor, torch::Tensor>& a,
    const std::pair<torch::Tensor, torch::Tensor>& b,
    const torch::Tensor& out, const torch::Tensor& m_indices,
    std::optional<const torch::Tensor> m_rows = std::nullopt,
    std::optional<ConfigTuple> configs = std::nullopt) {
    const auto& [m, k] = get_shape<2>(a.first);
    const int num_groups = b.first.size(0);
    const int n = b.first.size(1);
    check_a8w4_int8_common(a.first, a.second, b.first, b.second, out, num_groups, n, k);
    TORCH_CHECK(out.sizes() == std::vector<int64_t>({m, n}), "A8W4 output must have shape [m, n]");
    TORCH_CHECK(a.second.sizes() == std::vector<int64_t>({m, 1}), "A8W4 lhs scales must have shape [m, 1]");
    TORCH_CHECK(m_indices.scalar_type() == torch::kInt32 && m_indices.is_contiguous() && m_indices.numel() == m,
                "A8W4 m_indices must be contiguous int32 with m elements");
    if (m == 0) return;
    m_grouped_gemm_int8_int4_bf16_nt_nopad_impl(
        a.first, a.second, b.first, b.second, out, m_indices, m, n, k, num_groups, m_rows, configs);
}

void m_grouped_fp8_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                      const std::pair<torch::Tensor, torch::Tensor>& b,
                                      const torch::Tensor& d,
                                      const torch::Tensor& m_indices,
                                      std::optional<std::tuple<int, int, int>> recipe,
                                      std::optional<std::tuple<int, int>> recipe_a,
                                      std::optional<std::tuple<int, int>> recipe_b,
                                      const std::string& compiled_dims,
                                      const bool& disable_ue8m0_cast,
                                      const bool& use_psum_layout,
                                      const bool& ensure_zero_padding,
                                      const std::optional<int>& expected_m_for_psum_layout,
                                      std::optional<ConfigTuple> configs = std::nullopt) {
    const auto& lhs = a.first;
    const auto& lhs_scales = a.second;
    const auto& rhs = b.first;
    const auto& rhs_scales = b.second;

    const auto major_a = get_major_type_ab(lhs);
    const auto major_b = get_major_type_ab(rhs);
    DG_HOST_ASSERT(major_a == MajorType::K);
    if (fp8_requires_k_major())
        DG_HOST_ASSERT(major_b == MajorType::K);
    check_major_type_cd(d);

    const auto& [m, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_, n_] = get_shape<2>(d);
    int m__ = m_indices.numel();

    if ((lhs_scales.sizes() == std::vector<int64_t>{m, 1}) &&
        (rhs_scales.sizes() == std::vector<int64_t>{num_groups, n, 1})) {
        const auto arch_major = device_runtime->get_arch_major();
        if (arch_major == 8) {
            return m_grouped_int8_gemm_nt_contiguous(a, b, d, m_indices,
                                                     std::nullopt, std::nullopt, std::nullopt,
                                                     compiled_dims, false, false, true, std::nullopt,
                                                     configs);
        } else {
            DG_HOST_UNREACHABLE("Unsupported architecture");
        }
    }

    // Type and shape checks
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    if (use_psum_layout) {
        DG_HOST_ASSERT(m__ == num_groups);
    } else {
        DG_HOST_ASSERT(m__ == m_);
        DG_HOST_ASSERT(!expected_m_for_psum_layout.has_value());
    }
    DG_HOST_ASSERT((lhs_scales.sizes() == std::vector<int64_t>{m, (k + 127) / 128}));
    DG_HOST_ASSERT((rhs_scales.sizes() == std::vector<int64_t>{num_groups, (n + 127) / 128, (k + 127) / 128}));

    DG_HOST_ASSERT(lhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(lhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(rhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(m_indices.scalar_type() == torch::kInt32);

    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(m_indices.is_contiguous(), "m_indices must be contiguous");
    TORCH_CHECK(rhs_scales.is_contiguous(), "rhs_scales must be contiguous");

    if (m == 0) {
        return;
    }

    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 8) {
        m_grouped_gemm_fp8_fp8_bf16_nt_contiguous_impl(lhs, lhs_scales, rhs, rhs_scales, d, m_indices, m, n, k, num_groups,
                                                       configs);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

std::pair<int, int> m_grouped_fp8_gemm_nt_masked(const std::pair<torch::Tensor, torch::Tensor>& a,
                                                 const std::pair<torch::Tensor, torch::Tensor>& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& masked_m,
                                                 int expected_m,
                                                 std::optional<std::tuple<int, int, int>> recipe,
                                                 std::optional<std::tuple<int, int>> recipe_a,
                                                 std::optional<std::tuple<int, int>> recipe_b,
                                                 const std::string& compiled_dims,
                                                 const bool& disable_ue8m0_cast,
                                                 std::optional<int> max_block_n = 256,
                                                 std::optional<bool> enable_sbo_overlap = false,
                                                 std::optional<const torch::Tensor> signal = std::nullopt,
                                                 std::optional<ConfigTuple> configs = std::nullopt) {
    const auto& lhs = a.first;
    const auto& lhs_scales = a.second;
    const auto& rhs = b.first;
    const auto& rhs_scales = b.second;

    at::Tensor signal_tensor;
    if (signal.has_value() && signal->defined()) {
        signal_tensor = *signal;
    } else {
        signal_tensor = at::empty({0}, at::TensorOptions().dtype(at::kInt).device(d.device()));
    }

    const auto& [num_groups, m, k] = get_shape<3>(lhs);
    const auto& [num_groups_, n, k_] = get_shape<3>(rhs);
    const auto& [num_groups__, m_, n_] = get_shape<3>(d);
    int num_groups___ = masked_m.numel();

    const auto major_a = get_major_type_ab(lhs);
    const auto major_b = get_major_type_ab(rhs);
    DG_HOST_ASSERT(major_a == MajorType::K and major_b == MajorType::K);
    check_major_type_cd(d);

    if ((lhs_scales.sizes() == std::vector<int64_t>{num_groups, m, 1}) &&
        (rhs_scales.sizes() == std::vector<int64_t>{num_groups, n, 1})) {
        const auto arch_major = device_runtime->get_arch_major();
        if (arch_major == 8) {
            return m_grouped_int8_gemm_nt_masked(a, b, d, masked_m, expected_m,
                                                 std::nullopt, std::nullopt, std::nullopt,
                                                 compiled_dims, false,
                                                 max_block_n, enable_sbo_overlap, signal, configs);
        } else {
            DG_HOST_UNREACHABLE("Unsupported architecture");
        }
    }

    // Type and shape checks
    DG_HOST_ASSERT(num_groups == num_groups_ && num_groups_ == num_groups__ && num_groups__ == num_groups___);
    DG_HOST_ASSERT(m == m_ && n == n_ && k == k_);
    DG_HOST_ASSERT(expected_m > 0 && m > 0 && n > 0 && k > 0 && num_groups > 0);
    DG_HOST_ASSERT((lhs_scales.sizes() == std::vector<int64_t>{num_groups, m, (k + 127) / 128}));
    DG_HOST_ASSERT((rhs_scales.sizes() == std::vector<int64_t>{num_groups, (n + 127) / 128, (k + 127) / 128}));

    DG_HOST_ASSERT(lhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(lhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(rhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt32);

    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(masked_m.is_contiguous(), "masked_m must be contiguous");
    TORCH_CHECK(rhs_scales.is_contiguous(), "rhs_scales must be contiguous");

    if (enable_sbo_overlap.value_or(false)) {
        TORCH_CHECK(signal_tensor.defined(), "signal must be defined when enable_sbo_overlap is true");
        TORCH_CHECK(signal_tensor.is_contiguous(), "signal must be contiguous");
        TORCH_CHECK(signal_tensor.scalar_type() == torch::kInt32, "signal must be int32");
    }

    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 8) {
        return m_grouped_gemm_fp8_fp8_bf16_nt_masked_impl(lhs, lhs_scales, rhs, rhs_scales, d, masked_m, m, n, k,
                                                          num_groups, expected_m, configs, max_block_n.value_or(256),
                                                          enable_sbo_overlap.value_or(false), signal_tensor);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

void m_grouped_fp8_gemm_nt_nopad(const std::pair<torch::Tensor, torch::Tensor>& a,
                                 const std::pair<torch::Tensor, torch::Tensor>& b,
                                 const torch::Tensor& d,
                                 const torch::Tensor& m_indices,
                                 std::optional<const torch::Tensor> m_rows = std::nullopt,
                                 std::optional<std::tuple<int, int, int>> recipe = std::nullopt,
                                 std::optional<std::tuple<int, int>> recipe_a = std::nullopt,
                                 std::optional<std::tuple<int, int>> recipe_b = std::nullopt,
                                 const std::string& compiled_dims = "nk",
                                 const bool& disable_ue8m0_cast = false,
                                 const bool& use_psum_layout = false,
                                 const bool& ensure_zero_padding = true,
                                 const std::optional<int>& expected_m_for_psum_layout = std::nullopt,
                                 std::optional<ConfigTuple> configs = std::nullopt) {
    const auto& lhs = a.first;
    const auto& lhs_scales = a.second;
    const auto& rhs = b.first;
    const auto& rhs_scales = b.second;

    const auto& [m, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_, n_] = get_shape<2>(d);
    int m__ = m_indices.numel();

    if ((lhs_scales.sizes() == std::vector<int64_t>{m, 1}) &&
        (rhs_scales.sizes() == std::vector<int64_t>{num_groups, n, 1})) {
        const auto arch_major = device_runtime->get_arch_major();
        if (arch_major == 8) {
            return m_grouped_int8_gemm_nt_nopad(a, b, d, m_indices, m_rows,
                                                std::nullopt, std::nullopt, std::nullopt,
                                                compiled_dims, false, false, true, std::nullopt,
                                                configs);
        } else {
            DG_HOST_UNREACHABLE("Unsupported architecture");
        }
    }

    // Type and shape checks
    DG_HOST_ASSERT(m == m_ && m_ == m__ && k == k_ && n == n_);
    DG_HOST_ASSERT((lhs_scales.sizes() == std::vector<int64_t>{m, (k + 127) / 128}));
    DG_HOST_ASSERT((rhs_scales.sizes() == std::vector<int64_t>{num_groups, (n + 127) / 128, (k + 127) / 128}));

    DG_HOST_ASSERT(lhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(lhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(rhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(m_indices.scalar_type() == torch::kInt32);

    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(m_indices.is_contiguous(), "m_indices must be contiguous");
    TORCH_CHECK(rhs_scales.is_contiguous(), "rhs_scales must be contiguous");

    if (m == 0) {
        return;
    }

    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 8) {
        m_grouped_gemm_fp8_fp8_bf16_nt_nopad_impl(lhs, lhs_scales, rhs, rhs_scales, d, m_indices, m, n, k, num_groups,
                                                  m_rows, configs);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

void m_grouped_bf16_gemm_nt_contiguous(const torch::Tensor& lhs, const torch::Tensor& rhs,
                                       const torch::Tensor& out, const torch::Tensor& m_indices,
                                       const std::string& compiled_dims,
                                       const bool& use_psum_layout,
                                       const bool& ensure_zero_padding,
                                       const std::optional<int>& expected_m_for_psum_layout,
                                       std::optional<ConfigTuple> configs = std::nullopt) {
    const auto major_a = get_major_type_ab(lhs);
    const auto major_b = get_major_type_ab(rhs);
    DG_HOST_ASSERT(major_a == MajorType::K);

    const auto& [m, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_, n_] = get_shape<2>(out);
    int m__ = m_indices.numel();

    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(lhs.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(rhs.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(out.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(m_indices.dtype() == torch::kInt32);
    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(m_indices.is_contiguous(), "m_indices must be contiguous");

    if (use_psum_layout) {
        DG_HOST_ASSERT(m__ == num_groups);
    } else {
        DG_HOST_ASSERT(m == m_);
        DG_HOST_ASSERT(!expected_m_for_psum_layout.has_value());
    }

    check_major_type_cd(out);

    if (m == 0) {
        return;
    }

    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 8) {
        m_grouped_gemm_bf16_bf16_bf16_nt_contiguous_impl(lhs, rhs, out, m_indices, m, n, k, num_groups, configs);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

std::pair<int, int> m_grouped_bf16_gemm_nt_masked(const torch::Tensor& lhs, const torch::Tensor& rhs,
                                                  const torch::Tensor& out, const torch::Tensor& masked_m,
                                                  int expected_m, const std::string& compiled_dims,
                                                  std::optional<int> max_block_n = 256,
                                                  std::optional<bool> enable_sbo_overlap = false,
                                                  std::optional<const torch::Tensor> signal = std::nullopt,
                                                  std::optional<ConfigTuple> configs = std::nullopt) {
    at::Tensor signal_tensor;
    if (signal.has_value() && signal->defined()) {
        signal_tensor = *signal;
    } else {
        signal_tensor = at::empty({0}, at::TensorOptions().dtype(at::kInt).device(out.device()));
    }

    const auto major_a = get_major_type_ab(lhs);
    const auto major_b = get_major_type_ab(rhs);
    DG_HOST_ASSERT(major_a == MajorType::K and major_b == MajorType::K);

    const auto& [num_groups, m, k] = get_shape<3>(lhs);
    const auto& [num_groups_, n, k_] = get_shape<3>(rhs);
    const auto& [num_groups__, m_, n_] = get_shape<3>(out);
    int num_groups___ = masked_m.numel();

    DG_HOST_ASSERT(num_groups == num_groups_ && num_groups_ == num_groups__ && num_groups__ == num_groups___);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(lhs.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(rhs.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(out.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.dtype() == torch::kInt32);
    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(masked_m.is_contiguous(), "masked_m must be contiguous");

    if (enable_sbo_overlap.value_or(false)) {
        TORCH_CHECK(signal_tensor.defined(), "signal must be defined when enable_sbo_overlap is true");
        TORCH_CHECK(signal_tensor.is_contiguous(), "signal must be contiguous");
        TORCH_CHECK(signal_tensor.scalar_type() == torch::kInt32, "signal must be int32");
    }

    check_major_type_cd(out);

    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 8) {
        return m_grouped_gemm_bf16_bf16_bf16_nt_masked_impl(lhs, rhs, out, masked_m, m, n, k, num_groups, expected_m,
                                                            configs, max_block_n.value_or(256),
                                                            enable_sbo_overlap.value_or(false), signal_tensor);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

void m_grouped_bf16_gemm_nt_nopad(const torch::Tensor& lhs, const torch::Tensor& rhs,
                                  const torch::Tensor& out, const torch::Tensor& m_indices,
                                  std::optional<const torch::Tensor> m_rows = std::nullopt,
                                  const std::string& compiled_dims = "nk",
                                  const bool& use_psum_layout = false,
                                  const bool& ensure_zero_padding = true,
                                  const std::optional<int>& expected_m_for_psum_layout = std::nullopt,
                                  std::optional<ConfigTuple> configs = std::nullopt) {
    const auto& [m, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_, n_] = get_shape<2>(out);
    int m__ = m_indices.numel();

    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_ and m__ == m_);
    DG_HOST_ASSERT(lhs.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(rhs.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(out.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(m_indices.dtype() == torch::kInt32);
    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(m_indices.is_contiguous(), "m_indices must be contiguous");

    if (m == 0) {
        return;
    }

    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 8) {
        m_grouped_gemm_bf16_bf16_bf16_nt_nopad_impl(lhs, rhs, out, m_indices, m, n, k, num_groups, m_rows, configs);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void m_grouped_fp4_gemm_nt_nopad(const std::pair<torch::Tensor, torch::Tensor>& a,
                                        const std::pair<torch::Tensor, torch::Tensor>& b,
                                        const std::optional<torch::Tensor>& bias,
                                        const torch::Tensor& d,
                                        const torch::Tensor& m_indices,
                                        std::optional<const torch::Tensor> m_rows = std::nullopt,
                                        const std::string& compiled_dims = "nk",
                                        std::optional<torch::Tensor> out_scale = std::nullopt,
                                        std::optional<float> swiglu_limit = std::nullopt,
                                        std::optional<ConfigTuple> configs = std::nullopt,
                                        std::optional<std::string> activation_type = std::nullopt) {

    const auto& lhs = a.first;
    const auto& lhs_scales = a.second;
    const auto& rhs = b.first;
    const auto& rhs_scales = b.second;

    const auto& [m, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_, n_] = get_shape<2>(d);
    int m__ = m_indices.numel();

    DG_HOST_ASSERT(m == m_ && m_ == m__ && k == k_);
    DG_HOST_ASSERT(lhs.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(m_indices.scalar_type() == torch::kInt32);

    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(m_indices.is_contiguous(), "m_indices must be contiguous");

    // Undefined (rather than empty) means "no fused epilogue"; the impl keys off `defined()`.
    at::Tensor out_scale_tensor;
    if (out_scale.has_value() && out_scale->defined()) {
        out_scale_tensor = *out_scale;
    }

    // When `out_scale` is given, silu_and_mul + mxfp4 post-quant are fused into the epilogue:
    // `d` becomes uint8 of shape (m, n / 4) and `out_scale` uint16 of shape
    // (m, ceil_div(n / 4, 32)). Unlike the masked variant both are 2-D (no group dim).
    const bool enable_act_and_quant_fusing = out_scale_tensor.defined() && out_scale_tensor.numel() > 0;
    int sfm = 0, sfn = 0;
    if (enable_act_and_quant_fusing) {
        const int shape_n_out = n / 4;  // /2 for silu_and_mul, /2 for mxfp4 packing
        sfm = m;
        sfn = ceil_div(shape_n_out, 32);
        DG_HOST_ASSERT(n_ == shape_n_out);
        DG_HOST_ASSERT((out_scale_tensor.sizes() == std::vector<int64_t>{sfm, sfn}));
        DG_HOST_ASSERT(d.scalar_type() == torch::kUInt8);
        DG_HOST_ASSERT(out_scale_tensor.scalar_type() == torch::kUInt16);
        DG_HOST_ASSERT(out_scale_tensor.is_contiguous() ||
                       deep_gemm_fp4_common::check_mxfp4_scales_layout(out_scale_tensor));
        DG_HOST_ASSERT(!(bias.has_value() && bias->defined() && bias->numel() > 0));
    } else {
        DG_HOST_ASSERT(!swiglu_limit.has_value() || *swiglu_limit == 0.0f);  // only used with out_scale
        DG_HOST_ASSERT(!activation_type.has_value());                        // only used with out_scale
        DG_HOST_ASSERT(n_ == n);
        DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    }

    // --- Scales layout check & preprocess ---
    torch::Tensor lhs_scales_t = lhs_scales;
    torch::Tensor rhs_scales_t = rhs_scales;
    if (!deep_gemm_fp4_common::check_mxfp4_scales_layout(lhs_scales_t)) {
        lhs_scales_t = deep_gemm_fp4_common::preprocess_mxfp4_scales(lhs_scales_t);
    }
    if (!deep_gemm_fp4_common::check_mxfp4_scales_layout(rhs_scales_t)) {
        rhs_scales_t = deep_gemm_fp4_common::post_preprocess_mxfp4_scales(rhs_scales_t);
        if (!deep_gemm_fp4_common::check_mxfp4_scales_layout(rhs_scales_t)) {
            rhs_scales_t = deep_gemm_fp4_common::preprocess_mxfp4_scales(rhs_scales_t);
        }
    }

    // Scale stride checks (validate post-preprocessing result)
    DG_HOST_ASSERT(deep_gemm_fp4_common::check_mxfp4_scales_layout(lhs_scales_t));
    DG_HOST_ASSERT(deep_gemm_fp4_common::check_mxfp4_scales_layout(rhs_scales_t));

    if (m == 0) return;

    torch::Tensor bias_tensor;
    if (bias.has_value() && bias->defined() && bias->numel() > 0) {
        bias_tensor = *bias;
        DG_HOST_ASSERT(bias_tensor.scalar_type() == torch::kFloat32);
    } else {
        bias_tensor = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(lhs.device()));
    }

    torch::Tensor m_rows_tensor;
    if (m_rows.has_value() && m_rows->defined()) {
        m_rows_tensor = *m_rows;
    } else {
        m_rows_tensor = torch::Tensor();  // undefined
    }

    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 8) {
        m_grouped_gemm_fp4_fp4_bf16_nt_nopad_impl(lhs, lhs_scales_t, rhs, rhs_scales_t, bias_tensor, d,
                                                    m_indices, m_rows_tensor, m, n, k, num_groups, configs,
                                                    out_scale_tensor, swiglu_limit.value_or(0.0f), activation_type.value_or("silu_and_mul"));
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Re-stride out_scale in place to the N-major layout (1, sfm) that
    // the Gemm2 SFA reader expects. NOTE: this is 2-D here, unlike the masked variant's 3-D layout.
    // `out_scale_tensor` shares the TensorImpl with the caller's tensor, so this is visible in Python.
    if (enable_act_and_quant_fusing) {
        out_scale_tensor.as_strided_({sfm, sfn}, {1, (int64_t)sfm});
    }
}

void m_grouped_bf16_gemm_nt_fused(const torch::Tensor& lhs, const torch::Tensor& rhs,
                                  const torch::Tensor& out, const torch::Tensor& m_rows,
                                  const torch::Tensor& expert_ids_and_cumsum,
                                  const torch::Tensor& sorted_token_ids,
                                  const torch::Tensor& aligned_num_m_blocks,
                                  FusedCommonConfigTuple configs) {

    const auto& [num_token, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_sum, n_] = get_shape<2>(out);

    DG_HOST_ASSERT(k == k_ && n == n_);
    DG_HOST_ASSERT(lhs.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(rhs.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(out.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(m_rows.dtype() == torch::kInt32);
    DG_HOST_ASSERT(expert_ids_and_cumsum.dtype() == torch::kInt32);
    DG_HOST_ASSERT(sorted_token_ids.dtype() == torch::kInt32);
    DG_HOST_ASSERT(aligned_num_m_blocks.dtype() == torch::kInt32);
    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    DG_HOST_ASSERT(num_token > 0 && m_sum % num_token == 0);

    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 8) {
        m_grouped_gemm_bf16_bf16_bf16_nt_fused_impl(
            lhs, rhs, out, m_rows, expert_ids_and_cumsum,
            sorted_token_ids, aligned_num_m_blocks, configs);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

void m_grouped_fp8_gemm_nt_fused(const std::pair<torch::Tensor, torch::Tensor>& lhs_,
                                 const std::pair<torch::Tensor, torch::Tensor>& rhs_,
                                 const torch::Tensor& out,
                                 const torch::Tensor& m_rows,
                                 const torch::Tensor& expert_ids_and_cumsum,
                                 const torch::Tensor& sorted_token_ids,
                                 const torch::Tensor& aligned_num_m_blocks,
                                 FusedCommonConfigTuple configs) {

    const auto& lhs = lhs_.first;
    const auto& lhs_scales = lhs_.second;
    const auto& rhs = rhs_.first;
    const auto& rhs_scales = rhs_.second;

    const auto& [num_token, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_sum, n_] = get_shape<2>(out);

    DG_HOST_ASSERT(k == k_ && n == n_);
    DG_HOST_ASSERT(lhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(lhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(rhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(out.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(m_rows.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(expert_ids_and_cumsum.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(sorted_token_ids.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(aligned_num_m_blocks.scalar_type() == torch::kInt32);
    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    DG_HOST_ASSERT(k % 16 == 0);
    DG_HOST_ASSERT(num_token > 0 && m_sum % num_token == 0);

    const auto arch_major = device_runtime->get_arch_major();

    // Select per-channel quantization when each token and each expert output channel has one scale.
    if (lhs_scales.sizes() == std::vector<int64_t>{num_token, 1}
            && rhs_scales.sizes() == std::vector<int64_t>{num_groups, n, 1}) {
        if (arch_major == 8) {
            m_grouped_gemm_perchannel_nt_fused_impl(
                lhs, lhs_scales, rhs, rhs_scales, out, m_rows,
                expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks, configs);
        } else {
            DG_HOST_UNREACHABLE("Unsupported architecture");
        }
        return;
    }

    // blockwise quant
    if (m_sum == 0) return;
    int topk = m_sum / num_token;
    DG_HOST_ASSERT(n % 128 == 0);
    DG_HOST_ASSERT(k % 128 == 0);
    DG_HOST_ASSERT(std::get<3>(configs) == 128);  // block_k

    // Convert lhs scales to the column-major, TMA-aligned layout.
    torch::Tensor lhs_scales_col_major = get_mn_major_tma_aligned_tensor(lhs_scales);

    if (arch_major == 8) {
        m_grouped_gemm_blkwise_nt_fused_impl(
            lhs, lhs_scales_col_major, rhs, rhs_scales, out, m_rows,
            expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks, topk, configs);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

void m_grouped_int8_gemm_nt_fused(const std::pair<torch::Tensor, torch::Tensor>& lhs_,
                                  const std::pair<torch::Tensor, torch::Tensor>& rhs_,
                                  const torch::Tensor& out,
                                  const torch::Tensor& m_rows,
                                  const torch::Tensor& expert_ids_and_cumsum,
                                  const torch::Tensor& sorted_token_ids,
                                  const torch::Tensor& aligned_num_m_blocks,
                                  FusedCommonConfigTuple configs) {

    const auto& lhs = lhs_.first;
    const auto& lhs_scales = lhs_.second;
    const auto& rhs = rhs_.first;
    const auto& rhs_scales = rhs_.second;

    const auto& [num_token, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_sum, n_] = get_shape<2>(out);

    // Validate per-channel inputs before dispatch; the implementation does not re-check them.
    DG_HOST_ASSERT(k == k_ && n == n_);
    DG_HOST_ASSERT(lhs.scalar_type() == torch::kInt8);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kInt8);
    DG_HOST_ASSERT(lhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(rhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT((lhs_scales.sizes() == std::vector<int64_t>{num_token, 1}));
    DG_HOST_ASSERT((rhs_scales.sizes() == std::vector<int64_t>{num_groups, n, 1}));
    DG_HOST_ASSERT(out.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(m_rows.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(expert_ids_and_cumsum.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(sorted_token_ids.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(aligned_num_m_blocks.scalar_type() == torch::kInt32);
    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    DG_HOST_ASSERT(k % 16 == 0);
    DG_HOST_ASSERT(num_token > 0 && m_sum % num_token == 0);

    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 8) {
        m_grouped_gemm_perchannel_nt_fused_impl(
            lhs, lhs_scales, rhs, rhs_scales, out, m_rows,
            expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks, configs);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

// MoE GroupedFused FP4 GEMM. The lhs scale is K-major packed uint16, while rhs_scales must use the
// N-major layout produced by preprocess_mxfp4_scales. With out_scale, the epilogue writes packed
// uint8 activations and per-row E8M0 scales, then this API re-strides out_scale for Gemm2's SFA read.
void m_grouped_fp4_gemm_nt_fused(const std::pair<torch::Tensor, torch::Tensor>& lhs_,
                                 const std::pair<torch::Tensor, torch::Tensor>& rhs_,
                                 const torch::Tensor& out,
                                 const torch::Tensor& m_rows,
                                 const torch::Tensor& expert_ids_and_cumsum,
                                 const torch::Tensor& sorted_token_ids,
                                 const torch::Tensor& aligned_num_m_blocks,
                                 FusedCommonConfigTuple configs,
                                 std::optional<torch::Tensor> out_scale = std::nullopt,
                                 std::optional<float> swiglu_limit = std::nullopt,
                                 std::optional<std::string> activation_type = std::nullopt) {

    const auto& lhs = lhs_.first;
    const auto& lhs_scales = lhs_.second;
    const auto& rhs = rhs_.first;
    const auto& rhs_scales = rhs_.second;
    const auto& [num_token, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_sum, n_out] = get_shape<2>(out);

    torch::Tensor out_scale_tensor;
    const bool enable_act_and_quant_fusing =
        out_scale.has_value() && out_scale->defined();
    const float swiglu_limit_value = swiglu_limit.value_or(0.0f);

    DG_HOST_ASSERT(k == k_);
    DG_HOST_ASSERT(lhs.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kUInt8);
    TORCH_CHECK(
        lhs_scales.scalar_type() == torch::kUInt16,
        "fp4 fused MoE expects an uint16 SFA (uint8 E8M0 scales padded to an even count along K, "
        "then viewed as uint16)");
    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    const auto expected_lhs_scales_k = ceil_div(ceil_div(k, 16), 2);
    const auto expected_lhs_scales_shape = std::vector<int64_t>{num_token, expected_lhs_scales_k};
    TORCH_CHECK(
        lhs_scales.sizes() == expected_lhs_scales_shape,
        "fp4 fused MoE expects a K-major uint16 SFA of shape (", num_token, ", ",
        expected_lhs_scales_k, "), got ", lhs_scales.sizes());
    TORCH_CHECK(
        lhs_scales.is_contiguous(),
        "fp4 fused MoE expects an uint16 K-major (plain contiguous) SFA; "
        "do NOT call preprocess_mxfp4_scales on it");
    DG_HOST_ASSERT(deep_gemm_fp4_common::check_mxfp4_scales_layout(rhs_scales));
    TORCH_CHECK(
        k % 16 == 0,
        "K must be a multiple of 16, so that 16 8-bit elements can be loaded with 128b aligned "
        "vectorized memory access.");
    TORCH_CHECK(num_token > 0, "division by zero");

    int64_t sfm = 0;
    int64_t sfn = 0;
    if (enable_act_and_quant_fusing) {
        out_scale_tensor = *out_scale;
        const int64_t shape_n_out = n / 4;  // /2 silu_and_mul, /2 mxfp4 packing
        sfm = m_sum;
        sfn = ceil_div(shape_n_out, static_cast<int64_t>(32));
        TORCH_CHECK(n_out == shape_n_out, "n_=", n_out, ", expected ", shape_n_out);
        TORCH_CHECK(
            out_scale_tensor.sizes() == std::vector<int64_t>({sfm, sfn}),
            "out_scale shape ", out_scale_tensor.sizes(), ", expected (", sfm, ", ", sfn, ")");
        DG_HOST_ASSERT(out.scalar_type() == torch::kUInt8);
        DG_HOST_ASSERT(out_scale_tensor.scalar_type() == torch::kUInt16);
        DG_HOST_ASSERT(out_scale_tensor.is_contiguous() ||
                       deep_gemm_fp4_common::check_mxfp4_scales_layout(out_scale_tensor));
        TORCH_CHECK(
            n % 64 == 0,
            "n (", n, ") must be divisible by 64 in the SiluAndMulPostQuant epilogue: "
            "n // 4(act_func & quant) must stay a multiple of 16.");
    } else {
        TORCH_CHECK(
            !swiglu_limit.has_value() || swiglu_limit_value == 0.0f,
            "swiglu_limit is only used when out_scale is not None.");
        TORCH_CHECK(!activation_type.has_value(),
            "activation_type is only used when out_scale is not None.");
        TORCH_CHECK(n_out == n, "n_=", n_out, ", expected ", n);
        DG_HOST_ASSERT(out.scalar_type() == torch::kBFloat16);
        TORCH_CHECK(n % 2 == 0, "n (", n, ") must be divisible by 2.");
    }

    // Empty output returns before validating scheduler tensor dtypes.
    if (m_sum == 0) return;

    DG_HOST_ASSERT(m_rows.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(expert_ids_and_cumsum.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(sorted_token_ids.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(aligned_num_m_blocks.scalar_type() == torch::kInt32);

    m_grouped_gemm_fp4_fp4_bf16_nt_fused_impl(
        lhs, lhs_scales, rhs, rhs_scales, out, m_rows, expert_ids_and_cumsum,
        sorted_token_ids, aligned_num_m_blocks, configs, out_scale_tensor,
        swiglu_limit_value, activation_type.value_or("silu_and_mul"));

    if (enable_act_and_quant_fusing) {
        out_scale_tensor.as_strided_({sfm, sfn}, {1, sfm});
    }
}

static std::pair<int, int> m_grouped_fp4_gemm_nt_masked(const std::pair<torch::Tensor, torch::Tensor>& a,
                                                        const std::pair<torch::Tensor, torch::Tensor>& b,
                                                        const std::optional<torch::Tensor>& bias,
                                                        const torch::Tensor& d,
                                                        const torch::Tensor& masked_m,
                                                        int expected_m,
                                                        const std::optional<std::tuple<int,int,int>>& recipe,
                                                        const std::optional<std::tuple<int,int>>& recipe_a,
                                                        const std::optional<std::tuple<int,int>>& recipe_b,
                                                        const std::string& compiled_dims,
                                                        const bool disable_ue8m0_cast,
                                                        std::optional<int> max_block_n = 256,
                                                        std::optional<bool> enable_sbo_overlap = false,
                                                        std::optional<const torch::Tensor> signal = std::nullopt,
                                                        std::optional<torch::Tensor> out_scale = std::nullopt,
                                                        std::optional<float> swiglu_limit = std::nullopt,
                                                        std::optional<ConfigTuple> configs = std::nullopt,
                                                        std::optional<std::string> activation_type = std::nullopt) {

    const auto& lhs = a.first;
    const auto& lhs_scales = a.second;
    const auto& rhs = b.first;
    const auto& rhs_scales = b.second;

    const auto major_a = get_major_type_ab(lhs);
    const auto major_b = get_major_type_ab(rhs);
    DG_HOST_ASSERT(major_a == MajorType::K and major_b == MajorType::K);
    check_major_type_cd(d);

    const auto& [num_groups, m, k] = get_shape<3>(lhs);
    const auto& [num_groups_, n, k_] = get_shape<3>(rhs);
    const auto& [num_groups__, m_, n_] = get_shape<3>(d);
    int num_groups___ = masked_m.numel();

    DG_HOST_ASSERT(num_groups == num_groups_ && num_groups_ == num_groups__ && num_groups__ == num_groups___);
    DG_HOST_ASSERT(m == m_ && k == k_);
    DG_HOST_ASSERT(expected_m > 0 && m > 0 && n > 0 && k > 0 && num_groups > 0);
    DG_HOST_ASSERT(lhs.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt32);

    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(masked_m.is_contiguous(), "masked_m must be contiguous");

    at::Tensor signal_tensor;
    if (signal.has_value() && signal->defined()) {
        signal_tensor = *signal;
    } else {
        signal_tensor = at::empty({0}, at::TensorOptions().dtype(at::kInt).device(d.device()));
    }

    // Undefined (rather than empty) means "no fused epilogue"; the impl keys off `defined()`.
    at::Tensor out_scale_tensor;
    if (out_scale.has_value() && out_scale->defined()) {
        out_scale_tensor = *out_scale;
    }

    // When `out_scale` is given, silu_and_mul + mxfp4 post-quant are fused into the epilogue:
    // `d` becomes uint8 of shape (num_groups, m, n / 4) and `out_scale` uint16 of shape
    // (num_groups, m, ceil_div(n / 4, 32)). Bias is not supported in that mode.
    const bool enable_act_and_quant_fusing = out_scale_tensor.defined() && out_scale_tensor.numel() > 0;
    int sfm = 0, sfn = 0;
    if (enable_act_and_quant_fusing) {
        const int shape_n_out = n / 4;  // /2 for silu_and_mul, /2 for mxfp4 packing
        sfm = m;
        sfn = ceil_div(shape_n_out, 32);
        DG_HOST_ASSERT(n_ == shape_n_out);
        DG_HOST_ASSERT((out_scale_tensor.sizes() == std::vector<int64_t>{num_groups, sfm, sfn}));
        DG_HOST_ASSERT(d.scalar_type() == torch::kUInt8);
        DG_HOST_ASSERT(out_scale_tensor.scalar_type() == torch::kUInt16);
        DG_HOST_ASSERT(out_scale_tensor.is_contiguous() ||
                       deep_gemm_fp4_common::check_mxfp4_scales_layout(out_scale_tensor));
        DG_HOST_ASSERT(!(bias.has_value() && bias->defined() && bias->numel() > 0));
    } else {
        DG_HOST_ASSERT(!swiglu_limit.has_value() || *swiglu_limit == 0.0f);  // only used with out_scale
        DG_HOST_ASSERT(!activation_type.has_value());                        // only used with out_scale
        DG_HOST_ASSERT(n_ == n);
        DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    }

    // --- Scales layout check & preprocess ---
    torch::Tensor lhs_scales_t = lhs_scales;
    torch::Tensor rhs_scales_t = rhs_scales;
    if (!deep_gemm_fp4_common::check_mxfp4_scales_layout(lhs_scales_t)) {
        lhs_scales_t = deep_gemm_fp4_common::preprocess_mxfp4_scales(lhs_scales_t);
    }
    if (!deep_gemm_fp4_common::check_mxfp4_scales_layout(rhs_scales_t)) {
        rhs_scales_t = deep_gemm_fp4_common::post_preprocess_mxfp4_scales(rhs_scales_t);
        if (!deep_gemm_fp4_common::check_mxfp4_scales_layout(rhs_scales_t)) {
            rhs_scales_t = deep_gemm_fp4_common::preprocess_mxfp4_scales(rhs_scales_t);
        }
    }

    // Scale stride checks (validate post-preprocessing result)
    DG_HOST_ASSERT(deep_gemm_fp4_common::check_mxfp4_scales_layout(lhs_scales_t));
    DG_HOST_ASSERT(deep_gemm_fp4_common::check_mxfp4_scales_layout(rhs_scales_t));

    torch::Tensor bias_tensor;
    if (bias.has_value() && bias->defined() && bias->numel() > 0) {
        bias_tensor = *bias;
        DG_HOST_ASSERT(bias_tensor.scalar_type() == torch::kFloat32);
    } else {
        bias_tensor = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(lhs.device()));
    }

    if (enable_sbo_overlap.value_or(false)) {
        TORCH_CHECK(signal_tensor.defined(), "signal must be defined when enable_sbo_overlap is true");
        TORCH_CHECK(signal_tensor.is_contiguous(), "signal must be contiguous");
        TORCH_CHECK(signal_tensor.scalar_type() == torch::kInt32, "signal must be int32");
    }

    const auto arch_major = device_runtime->get_arch_major();
    std::pair<int, int> result;
    if (arch_major == 8) {
        result = m_grouped_gemm_fp4_fp4_bf16_nt_masked_impl(lhs, lhs_scales_t, rhs, rhs_scales_t, bias_tensor, d,
                                                     masked_m, m, n, k, num_groups, expected_m, configs,
                                                     max_block_n.value_or(256), enable_sbo_overlap.value_or(false),
                                                     signal_tensor,
                                                     out_scale_tensor,
                                                     swiglu_limit.value_or(0.0f),
                                                     activation_type.value_or("silu_and_mul"));
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Re-stride out_scale in place to the N-major layout
    // (sfm * sfn, 1, sfm) that the Gemm2 SFA reader expects. `out_scale_tensor` shares the
    // TensorImpl with the caller's tensor, so the re-stride is visible on the Python side.
    if (enable_act_and_quant_fusing) {
        out_scale_tensor.as_strided_({num_groups, sfm, sfn},
                                     {(int64_t)sfm * sfn, 1, (int64_t)sfm});
    }
    return result;
}

// moe_align preprocessing (C++ JIT) — apis-layer interface owns input
// validation; the impls layer (jit_kernels/impls/moe_align.hpp) only
// orchestrates config resolution + kernel dispatch.
//
// Configs preserve the dtype-specific seven- or nine-element tile format.
MoeAlignReturn moe_align_block_size(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const torch::Tensor& topk_ids,
    bool perchannel_quant = false,
    std::optional<FusedConfigTuple> config = std::nullopt,
    bool enable_act_and_quant_fusing = false) {
    DG_HOST_ASSERT(topk_ids.dtype() == torch::kInt32);
    DG_HOST_ASSERT(topk_ids.dim() == 2);

    return moe_align_block_size_impl(lhs, rhs, topk_ids, perchannel_quant, config,
                                     enable_act_and_quant_fusing);
}
// The problem sizes the W4A16 implementations need, derived from the operands here so the API layer
// owns the shape contract.
struct W4A16Operands {
    int m, n, k, num_groups, group_size;
};

// Shape, dtype and layout checks shared by the three W4A16 entries. The operand layout depends only on
// `w4a16_type`, which is why upstream kept these in `m_grouped_gemm_w4a16_common` instead of per entry;
// they live here so malformed operands are rejected at the API boundary like the FP8/INT8 entries do.
//
// NOTES: an empty problem short-circuits before the operand checks, exactly as upstream did -- nothing
// gets launched, so its shapes are never inspected.
static W4A16Operands check_w4a16_operands(deep_gemm_w4a16_common::W4A16Type w4a16_type,
                                          const torch::Tensor& lhs, const torch::Tensor& rhs,
                                          const torch::Tensor& rhs_scales, const torch::Tensor& out,
                                          GemmType gemm_type) {
    const bool is_mma = deep_gemm_w4a16_common::uses_mma_kernel(w4a16_type);

    int m, k, n_;
    if (gemm_type == GemmType::MGroupedMasked) {
        m = static_cast<int>(lhs.size(1)), k = static_cast<int>(lhs.size(2));
        n_ = static_cast<int>(out.size(2));
    } else {
        m = static_cast<int>(lhs.size(0)), k = static_cast<int>(lhs.size(1));
        n_ = static_cast<int>(out.size(1));
    }
    const int num_groups = static_cast<int>(rhs.size(0));
    // The MMA weight is already N-major, the packed one carries two N elements per int32
    const int n = is_mma ? n_ : static_cast<int>(rhs.size(2)) / 2;

    const int64_t scale_elements_per_group = rhs_scales.size(1) * rhs_scales.size(2);
    DG_HOST_ASSERT(static_cast<int64_t>(k) * n % scale_elements_per_group == 0);
    const int group_size = static_cast<int>(static_cast<int64_t>(k) * n / scale_elements_per_group);
    // W4A16 only supports group_size=32
    DG_HOST_ASSERT(group_size == 32);

    const W4A16Operands operands{m, n, k, num_groups, group_size};
    // Type and shape checks
    if (m == 0)
        return operands;
    // K must be a multiple of group_size
    DG_HOST_ASSERT(k % group_size == 0);
    DG_HOST_ASSERT(n == n_);
    if (is_mma) {
        // w4fa16_mma weight dtype must be uint8, shaped (num_groups, n, k / 2)
        DG_HOST_ASSERT(rhs.scalar_type() == torch::kUInt8);
        DG_HOST_ASSERT(rhs.dim() == 3 and rhs.size(0) == num_groups and rhs.size(1) == n and rhs.size(2) == k / 2);
        // w4fa16_mma scale dtype must be uint8, shaped (num_groups, n / 64, k * 2)
        DG_HOST_ASSERT(rhs_scales.scalar_type() == torch::kUInt8);
        DG_HOST_ASSERT(rhs_scales.dim() == 3 and rhs_scales.size(0) == num_groups and
                       rhs_scales.size(1) == n / 64 and rhs_scales.size(2) == k * 2);
    } else {
        DG_HOST_ASSERT(rhs.dim() == 3 and rhs.size(0) == num_groups and rhs.size(1) == k / 16 and
                       rhs.size(2) == n * 2);
        DG_HOST_ASSERT(rhs_scales.dim() == 3 and rhs_scales.size(0) == num_groups and
                       rhs_scales.size(1) == k / group_size and rhs_scales.size(2) == n);
        DG_HOST_ASSERT(rhs.scalar_type() == torch::kInt32);
    }
    DG_HOST_ASSERT(n > 0 and k > 0 and n % 64 == 0 and k % 16 == 0);
    DG_HOST_ASSERT(lhs.scalar_type() == torch::kBFloat16);
    if (w4a16_type == deep_gemm_w4a16_common::W4A16Type::mxfp4_e8m0) {
        // W4FA16 E8M0 scale dtype must be uint8
        DG_HOST_ASSERT(rhs_scales.scalar_type() == torch::kUInt8);
    } else if (not is_mma) {
        // W4A16 BF16 scale dtype must be bfloat16
        DG_HOST_ASSERT(rhs_scales.scalar_type() == torch::kBFloat16);
    }
    DG_HOST_ASSERT(out.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(lhs.is_contiguous() and rhs.is_contiguous() and out.is_contiguous());
    DG_HOST_ASSERT(rhs_scales.is_contiguous());
    return operands;
}

// NOTES: the weight and its scale arrive as one pair because they are always produced together by the
// quantizer; `w4a16_type` is resolved here for the operand checks and again inside the implementation,
// which needs it to pick `ElementB` / `ElementScale`
static void m_grouped_gemm_w4a16_nopad(const torch::Tensor& lhs,
                                       const std::pair<torch::Tensor, torch::Tensor>& rhs_,
                                       const torch::Tensor& out,
                                       const torch::Tensor& m_indices,
                                       const std::optional<torch::Tensor>& m_rows = std::nullopt,
                                       std::optional<W4A16ConfigTuple> configs = std::nullopt) {
    const auto& rhs = rhs_.first;
    const auto& rhs_scales = rhs_.second;
    const auto w4a16_type = deep_gemm_w4a16_common::get_w4a16_type(rhs.scalar_type(), rhs_scales.scalar_type());
    const auto& operands = check_w4a16_operands(w4a16_type, lhs, rhs, rhs_scales, out, GemmType::MGroupedNoPad);
    if (operands.m == 0)
        return;

    m_grouped_gemm_w4a16_nopad_impl(lhs, rhs, rhs_scales, out, m_indices,
                                    m_rows.value_or(torch::Tensor()),
                                    operands.m, operands.n, operands.k, operands.num_groups, operands.group_size,
                                    configs);
}

static void m_grouped_gemm_w4a16_masked(const torch::Tensor& lhs,
                                        const std::pair<torch::Tensor, torch::Tensor>& rhs_,
                                        const torch::Tensor& out,
                                        const torch::Tensor& masked_m,
                                        int expected_m,
                                        std::optional<W4A16ConfigTuple> configs = std::nullopt) {
    const auto& rhs = rhs_.first;
    const auto& rhs_scales = rhs_.second;
    const auto w4a16_type = deep_gemm_w4a16_common::get_w4a16_type(rhs.scalar_type(), rhs_scales.scalar_type());
    const auto& operands = check_w4a16_operands(w4a16_type, lhs, rhs, rhs_scales, out, GemmType::MGroupedMasked);
    if (operands.m == 0)
        return;

    m_grouped_gemm_w4a16_masked_impl(lhs, rhs, rhs_scales, out, masked_m,
                                     operands.m, operands.n, operands.k, operands.num_groups, operands.group_size,
                                     expected_m, configs);
}

static void m_grouped_gemm_w4a16_fused(const torch::Tensor& lhs,
                                       const std::pair<torch::Tensor, torch::Tensor>& rhs_,
                                       const torch::Tensor& out,
                                       const torch::Tensor& m_rows,
                                       const torch::Tensor& expert_ids_and_cumsum,
                                       const torch::Tensor& sorted_token_ids,
                                       const torch::Tensor& aligned_num_m_blocks,
                                       const W4A16ConfigTuple& configs) {
    const auto& rhs = rhs_.first;
    const auto& rhs_scales = rhs_.second;
    const auto w4a16_type = deep_gemm_w4a16_common::get_w4a16_type(rhs.scalar_type(), rhs_scales.scalar_type());
    const auto& operands = check_w4a16_operands(w4a16_type, lhs, rhs, rhs_scales, out, GemmType::MGroupedFused);
    if (operands.m == 0)
        return;

    m_grouped_gemm_w4a16_fused_impl(lhs, rhs, rhs_scales, out, m_rows, expert_ids_and_cumsum,
                                    sorted_token_ids, aligned_num_m_blocks,
                                    operands.m, operands.n, operands.k, operands.num_groups, operands.group_size,
                                    configs);
}

static void check_w4fa8_operands(const std::pair<torch::Tensor, torch::Tensor>& lhs,
                                        const std::pair<torch::Tensor, torch::Tensor>& rhs, const torch::Tensor& out,
                                        GemmType kind) {
    DG_HOST_ASSERT(is_ppu1v5_device());
    const auto& a = lhs.first;
    const auto& s = lhs.second;
    const auto& b = rhs.first;
    const auto& bs = rhs.second;
    const bool masked = kind == GemmType::MGroupedMasked;
    TORCH_CHECK(a.dim() == (masked ? 3 : 2) && b.dim() == 3 && out.dim() == a.dim(), "invalid W4FA8 tensor rank");
    const int m = a.size(-2), k = a.size(-1), e = b.size(0), n = b.size(1);
    TORCH_CHECK(e > 0 && n > 0 && n % 64 == 0 && k > 0 && k % 128 == 0, "W4FA8 requires N%64=0, K%128=0");
    TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn && s.scalar_type() == torch::kFloat32,
                "W4FA8 activation must be E4M3 with FP32 per-row scales");
    TORCH_CHECK(b.scalar_type() == torch::kUInt8 && bs.scalar_type() == torch::kUInt8 &&
                    out.scalar_type() == torch::kBFloat16,
                "W4FA8 requires uint8 weights/scales and BF16 output");
    TORCH_CHECK(b.size(2) == k / 2 && bs.sizes() == std::vector<int64_t>({e, n / 64, k * 2}),
                "invalid preprocessed weight layout");
    auto scale_shape = a.sizes().vec();
    scale_shape.back() = 1;
    TORCH_CHECK(s.sizes() == scale_shape, "activation scale shape must match rows with final dimension 1");
    TORCH_CHECK(out.size(-1) == n, "output N mismatch");
    if (masked) {
        TORCH_CHECK(a.size(0) == e && out.size(0) == e && out.size(1) == m, "masked shape mismatch");
    } else if (kind == GemmType::MGroupedNoPad) {
        TORCH_CHECK(out.size(0) == m, "nopad output M mismatch");
    } else {
        TORCH_CHECK(m == 0 ? out.size(0) == 0 : out.size(0) % m == 0 && out.size(0) / m > 0 && out.size(0) / m <= e,
                    "fused output must contain token * top-k rows");
    }
    for (const auto& t : {a, s, b, bs, out}) {
        TORCH_CHECK(t.is_cuda() && t.device() == a.device() && t.is_contiguous(),
                    "W4FA8 tensors must be contiguous on one PPU device");
    }
}

static void check_w4fa8_indices(const torch::Tensor& t, const torch::Tensor& a) {
    TORCH_CHECK(t.scalar_type() == torch::kInt32 && t.device() == a.device() && t.is_contiguous(),
                "W4FA8 scheduler tensors must be contiguous int32 on the input device");
}

static void fp8_fp4_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                           const std::pair<torch::Tensor, torch::Tensor>& b, const torch::Tensor& out,
                           std::optional<W4A8ConfigTuple> configs = std::nullopt, float inv_scale = 1.0f) {
    TORCH_CHECK(b.first.dim() == 2 && b.second.dim() == 2,
                "dense W4FA8 requires 2D weights and preprocessed scales");
    const auto grouped_b = std::make_pair(b.first.unsqueeze(0), b.second.unsqueeze(0));
    check_w4fa8_operands(a, grouped_b, out, GemmType::MGroupedNoPad);
    TORCH_CHECK(std::isfinite(inv_scale) && inv_scale > 0.f, "inv_scale must be finite and positive");
    const int m = a.first.size(0), n = b.first.size(0), k = a.first.size(1);
    if (!m)
        return;
    const auto tile = configs ? deep_gemm::w4fa8::unpack_config(*configs)
                              : deep_gemm::w4fa8::get_best_configs(m, n, k, 1, GemmType::MGroupedNoPad);
    const auto plan = deep_gemm::w4fa8::resolve_config(tile, m, n, k, 1, GemmType::MGroupedNoPad);
    const auto rows = at::full({1}, m, a.first.options().dtype(torch::kInt32));
    deep_gemm::w4fa8::execute_plan(plan, a.first, a.second, grouped_b.first, grouped_b.second, out, rows,
                                 GemmType::MGroupedNoPad, m, inv_scale);
}

static void m_grouped_gemm_fp8_fp4_bf16_nt_nopad(const std::pair<torch::Tensor, torch::Tensor>& a,
                                      const std::pair<torch::Tensor, torch::Tensor>& b, const torch::Tensor& out,
                                      const torch::Tensor& indices, std::optional<torch::Tensor> rows = std::nullopt,
                                      std::optional<W4A8ConfigTuple> configs = std::nullopt, float inv_scale = 1.0f) {
    check_w4fa8_operands(a, b, out, GemmType::MGroupedNoPad);
    TORCH_CHECK(std::isfinite(inv_scale) && inv_scale > 0.f, "inv_scale must be finite and positive");
    const int m = a.first.size(-2), n = b.first.size(1), k = a.first.size(-1), num_groups = b.first.size(0);
    TORCH_CHECK(num_groups <= 1024, "W4FA8 nopad supports at most 1024 experts");
    check_w4fa8_indices(indices, a.first);
    TORCH_CHECK(indices.dim() == 1 && indices.numel() == m, "m_indices shape mismatch");
    if (!m)
        return;
    torch::Tensor counts;
    if (rows.has_value())
        counts = *rows;
    else
        counts = at::zeros({num_groups}, indices.options()).index_add_(0, indices.to(at::kLong), at::ones_like(indices));
    check_w4fa8_indices(counts, a.first);
    TORCH_CHECK(counts.dim() == 1 && counts.numel() == num_groups, "m_rows shape mismatch");
    const int em = ceil_div(m, num_groups);
    const auto tile = configs ? deep_gemm::w4fa8::unpack_config(*configs)
                              : deep_gemm::w4fa8::get_best_configs(em, n, k, num_groups, GemmType::MGroupedNoPad);
    const auto plan = deep_gemm::w4fa8::resolve_config(tile, m, n, k, num_groups, GemmType::MGroupedNoPad);
    deep_gemm::w4fa8::execute_plan(plan, a.first, a.second, b.first, b.second, out, counts,
                                 GemmType::MGroupedNoPad, em, inv_scale);
}

static void m_grouped_gemm_fp8_fp4_bf16_nt_masked(const std::pair<torch::Tensor, torch::Tensor>& a,
                                       const std::pair<torch::Tensor, torch::Tensor>& b, const torch::Tensor& out,
                                       const torch::Tensor& rows, int expected_m,
                                       std::optional<W4A8ConfigTuple> configs = std::nullopt, float inv_scale = 1.0f) {
    check_w4fa8_operands(a, b, out, GemmType::MGroupedMasked);
    TORCH_CHECK(std::isfinite(inv_scale) && inv_scale > 0.f, "inv_scale must be finite and positive");
    const int m = a.first.size(-2), n = b.first.size(1), k = a.first.size(-1), num_groups = b.first.size(0);
    check_w4fa8_indices(rows, a.first);
    TORCH_CHECK(rows.dim() == 1 && rows.numel() == num_groups && expected_m >= 0, "masked_m shape or expected_m invalid");
    if (!m)
        return;
    const auto tile = configs ? deep_gemm::w4fa8::unpack_config(*configs)
                              : deep_gemm::w4fa8::get_best_configs(expected_m, n, k, num_groups, GemmType::MGroupedMasked);
    const auto plan = deep_gemm::w4fa8::resolve_config(tile, m, n, k, num_groups, GemmType::MGroupedMasked);
    deep_gemm::w4fa8::execute_plan(plan, a.first, a.second, b.first, b.second, out, rows,
                                 GemmType::MGroupedMasked, expected_m, inv_scale);
}

static void m_grouped_gemm_fp8_fp4_bf16_nt_fused(const std::pair<torch::Tensor, torch::Tensor>& a,
                                      const std::pair<torch::Tensor, torch::Tensor>& b, const torch::Tensor& out,
                                      const torch::Tensor& rows, const torch::Tensor& experts,
                                      const torch::Tensor& tokens, const torch::Tensor& aligned,
                                      const W4A8ConfigTuple& configs, float inv_scale = 1.0f) {
    check_w4fa8_operands(a, b, out, GemmType::MGroupedFused);
    TORCH_CHECK(std::isfinite(inv_scale) && inv_scale > 0.f, "inv_scale must be finite and positive");
    const int m = a.first.size(0), n = b.first.size(1), k = a.first.size(-1), num_groups = b.first.size(0);
    for (const auto& t : {rows, experts, tokens, aligned})
        check_w4fa8_indices(t, a.first);
    TORCH_CHECK(rows.numel() == num_groups && aligned.numel() >= 1, "fused metadata shape mismatch");
    if (!m)
        return;
    const auto plan = deep_gemm::w4fa8::resolve_config(deep_gemm::w4fa8::unpack_config(configs), out.size(0), n, k, num_groups, GemmType::MGroupedFused);
    TORCH_CHECK(tokens.dim() == 2 && tokens.size(1) == std::get<1>(configs),
                "sorted_token_ids routing BlockM must match W4FA8 configs");
    deep_gemm::w4fa8::execute_plan(plan, a.first, a.second, b.first, b.second, out, rows,
                                 GemmType::MGroupedFused, ceil_div(int(out.size(0)), num_groups), inv_scale, experts, tokens, aligned);
}

static void acblaslt_gemm_nt(const torch::Tensor& a, const torch::Tensor& b,
                             const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);

    // Type and shape checks
    const auto [m , k ] = get_shape<2>(a);
    const auto [n , k_] = get_shape<2>(b);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);

    // Early return for trivial cases
    if (early_return(m, n, k, d, c))
        return;

    acblaslt_gemm(a, b, d, m, n, k, major_a, major_b, c.has_value());
}

static void acblaslt_gemm_nn(const torch::Tensor& a, const torch::Tensor& b,
                             const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    acblaslt_gemm_nt(a, b.transpose(0, 1), d, c);
}

static void acblaslt_gemm_tn(const torch::Tensor& a, const torch::Tensor& b,
                             const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    acblaslt_gemm_nt(a.transpose(0, 1), b.transpose(0, 1), d, c);
}

static void acblaslt_gemm_tt(const torch::Tensor& a, const torch::Tensor& b,
                             const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    acblaslt_gemm_nt(a.transpose(0, 1), b, d, c);
}
}


static void register_apis(pybind11::module_& m) {
    // BF16 GEMMs
    m.def("gemm_fp32_fp32_fp32_nt", &gemm_fp32_fp32_fp32_nt,
          py::arg("a"), py::arg("b"), py::arg("d"));
    m.def("bf16_gemm_strided_batched_nt", &bf16_gemm_strided_batched_nt,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("configs") = std::nullopt);
    m.def("bf16_gemm_nt", &bf16_gemm_nt, py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("compiled_dims") = "nk", py::arg("configs") = std::nullopt);
    m.def("bf16_gemm_nn", &bf16_gemm_nn, py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("compiled_dims") = "nk", py::arg("configs") = std::nullopt);
    m.def("bf16_gemm_tn", &bf16_gemm_tn, py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("compiled_dims") = "nk", py::arg("configs") = std::nullopt);
    m.def("bf16_gemm_tt", &bf16_gemm_tt, py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("compiled_dims") = "nk", py::arg("configs") = std::nullopt);
    m.def("m_grouped_bf16_gemm_nt_contiguous", &m_grouped_bf16_gemm_nt_contiguous, py::arg("lhs"),
          py::arg("rhs"), py::arg("out"), py::arg("m_indices"), py::arg("compiled_dims") = "nk",
          py::arg("use_psum_layout") = false, py::arg("ensure_zero_padding") = true,
          py::arg("expected_m_for_psum_layout") = std::nullopt, py::arg("configs") = std::nullopt);
    m.def("m_grouped_bf16_gemm_nt_masked", &m_grouped_bf16_gemm_nt_masked, py::arg("lhs"),
          py::arg("rhs"), py::arg("out"), py::arg("masked_m"), py::arg("expected_m"), py::arg("compiled_dims") = "nk",
          py::arg("max_block_n") = 256, py::arg("enable_sbo_overlap") = false, py::arg("signal") = std::nullopt,
          py::arg("configs") = std::nullopt);
    m.def("m_grouped_bf16_gemm_nt_nopad", &m_grouped_bf16_gemm_nt_nopad, py::arg("lhs"),
          py::arg("rhs"), py::arg("out"), py::arg("m_indices"), py::arg("m_rows") = std::nullopt,
          py::arg("compiled_dims") = "nk", py::arg("use_psum_layout") = false,
          py::arg("ensure_zero_padding") = true,
          py::arg("expected_m_for_psum_layout") = std::nullopt, py::arg("configs") = std::nullopt);
    // INT8 GEMMs
    m.def("int8_gemm_nt", &int8_gemm_nt, py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("compiled_dims") = "nk", py::arg("configs") = std::nullopt);
    m.def("m_grouped_int8_gemm_nt_contiguous", &m_grouped_int8_gemm_nt_contiguous, py::arg("a"),
          py::arg("b"), py::arg("d"), py::arg("m_indices"),
          py::arg("recipe") = std::nullopt, py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk", py::arg("disable_ue8m0_cast") = false,
          py::arg("use_psum_layout") = false, py::arg("ensure_zero_padding") = true,
          py::arg("expected_m_for_psum_layout") = std::nullopt, py::arg("configs") = std::nullopt);
    m.def("m_grouped_int8_gemm_nt_masked", &m_grouped_int8_gemm_nt_masked, py::arg("a"),
          py::arg("b"), py::arg("d"), py::arg("masked_m"), py::arg("expected_m"),
          py::arg("recipe") = std::nullopt, py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk", py::arg("disable_ue8m0_cast") = false,
          py::arg("max_block_n") = 256, py::arg("enable_sbo_overlap") = false, py::arg("signal") = std::nullopt,
          py::arg("configs") = std::nullopt);
    m.def("m_grouped_int8_gemm_nt_nopad", &m_grouped_int8_gemm_nt_nopad, py::arg("a"), py::arg("b"),
          py::arg("d"), py::arg("m_indices"), py::arg("m_rows") = std::nullopt,
          py::arg("recipe") = std::nullopt, py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk", py::arg("disable_ue8m0_cast") = false,
          py::arg("use_psum_layout") = false, py::arg("ensure_zero_padding") = true,
          py::arg("expected_m_for_psum_layout") = std::nullopt, py::arg("configs") = std::nullopt);
    // A8W4 INT8 MoE GEMMs. The model stores portable packed signed-INT4 rhs; call
    // deep_gemm.preprocess_a8w4_weight after loading it to apply the device-specific permutation.
    m.def("m_grouped_gemm_int8_int4_bf16_nt_masked", &m_grouped_gemm_int8_int4_bf16_nt_masked,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("masked_m"), py::arg("expected_m"),
          py::arg("max_block_n") = 256, py::arg("enable_sbo_overlap") = false, py::arg("signal") = std::nullopt,
          py::arg("configs") = std::nullopt);
    m.def("m_grouped_gemm_int8_int4_bf16_nt_nopad", &m_grouped_gemm_int8_int4_bf16_nt_nopad,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("m_indices"),
          py::arg("m_rows") = std::nullopt, py::arg("configs") = std::nullopt);
    // FP8 GEMMs
    m.def("fp8_gemm_nt", &fp8_gemm_nt, py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("recipe") = std::nullopt, py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk", py::arg("disable_ue8m0_cast") = false, py::arg("configs") = std::nullopt);
    m.def("fp8_gemm_nn", &fp8_gemm_nn, py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("recipe") = std::nullopt, py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk", py::arg("disable_ue8m0_cast") = false, py::arg("configs") = std::nullopt);
    m.def("fp8_gemm_tn", &fp8_gemm_tn, py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("recipe") = std::nullopt, py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk", py::arg("disable_ue8m0_cast") = false, py::arg("configs") = std::nullopt);
    m.def("fp8_gemm_tt", &fp8_gemm_tt, py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("recipe") = std::nullopt, py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk", py::arg("disable_ue8m0_cast") = false, py::arg("configs") = std::nullopt);
    m.def("m_grouped_fp8_gemm_nt_contiguous", &m_grouped_fp8_gemm_nt_contiguous, py::arg("a"),
          py::arg("b"), py::arg("d"), py::arg("m_indices"),
          py::arg("recipe") = std::nullopt, py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk", py::arg("disable_ue8m0_cast") = false,
          py::arg("use_psum_layout") = false, py::arg("ensure_zero_padding") = true,
          py::arg("expected_m_for_psum_layout") = std::nullopt, py::arg("configs") = std::nullopt);
    m.def("m_grouped_fp8_gemm_nt_masked", &m_grouped_fp8_gemm_nt_masked, py::arg("a"), py::arg("b"),
          py::arg("d"), py::arg("masked_m"), py::arg("expected_m"),
          py::arg("recipe") = std::nullopt, py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk", py::arg("disable_ue8m0_cast") = false,
          py::arg("max_block_n") = 256, py::arg("enable_sbo_overlap") = false, py::arg("signal") = std::nullopt,
          py::arg("configs") = std::nullopt);
    m.def("m_grouped_fp8_gemm_nt_nopad", &m_grouped_fp8_gemm_nt_nopad, py::arg("a"), py::arg("b"),
          py::arg("d"), py::arg("m_indices"), py::arg("m_rows") = std::nullopt,
          py::arg("recipe") = std::nullopt, py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk", py::arg("disable_ue8m0_cast") = false,
          py::arg("use_psum_layout") = false, py::arg("ensure_zero_padding") = true,
          py::arg("expected_m_for_psum_layout") = std::nullopt, py::arg("configs") = std::nullopt);
    // FP4 GEMMs
    m.def("fp4_gemm_nt", &fp4_gemm_nt, py::arg("a"), py::arg("b"), py::arg("bias"),
          py::arg("d"), py::arg("c") = std::nullopt,
          py::arg("recipe") = std::nullopt, py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk", py::arg("disable_ue8m0_cast") = false, py::arg("configs") = std::nullopt);
    m.def("m_grouped_fp4_gemm_nt_nopad", &m_grouped_fp4_gemm_nt_nopad,
          R"(MoE GroupedNoPad FP4 GEMM.

When `out_scale` is not None, silu_and_mul + mxfp4 post-quant are fused into the epilogue:
`out` must be uint8 of shape (m, n // 4) and `out_scale` uint16 of shape
(m, ceil_div(n // 4, 32)). The gemm1 weight must have been interleaved with
`preprocess_weights_and_scales(mode=PreprocessMode.MXFP4, fuse_activation=True)` (instead of
`preprocess_mxfp4_scales`), so that the epilogue reads gate/up (W1/W3) pairs from adjacent N positions.

`swiglu_limit > 0` clamps around the activation; `0.0` or None disables the clamp. The exact
clamp order depends on `activation_type` (only used together with `out_scale`):
  - `silu_and_mul` (default): clamp BEFORE the activation. The gate is clamped from above only
    (`min(gate, limit)`) and the up projection on both sides (`clamp(up, -limit, limit)`).
  - `silu_and_mul_step` (STEP models): silu FIRST, then clamp `silu(gate)` from above
    (`min(silu(gate), limit)`); the up projection is still clamped to `[-limit, limit]`.

Shape `n` must be a multiple of 64, and a bias is not supported in the fused mode.

NOTE: when `out_scale` is not None, it is re-strided **in place** on return to the N-major
layout (1, sfm) expected by the Gemm2 SFA reader. Unlike the masked variant, both `out` and
`out_scale` are 2-D here (no group dimension) and the re-stride is (1, sfm) rather than
(sfm * sfn, 1, sfm). Passing an already re-strided `out_scale` back in is supported.)",
          py::arg("a"), py::arg("b"), py::arg("bias"), py::arg("d"), py::arg("m_indices"),
          py::arg("m_rows") = std::nullopt, py::arg("compiled_dims") = "nk",
          py::arg("out_scale") = std::nullopt, py::arg("swiglu_limit") = std::nullopt, py::arg("configs") = std::nullopt,
          py::arg("activation_type") = std::nullopt);
    m.def("m_grouped_fp4_gemm_nt_masked", &m_grouped_fp4_gemm_nt_masked,
          R"(MoE GroupedMasked FP4 GEMM.

When `out_scale` is not None, silu_and_mul + mxfp4 post-quant are fused into the epilogue:
`out` must be uint8 of shape (num_groups, m, n // 4) and `out_scale` uint16 of shape
(num_groups, m, ceil_div(n // 4, 32)). The gemm1 weight must have been interleaved with
`preprocess_weights_and_scales(mode=PreprocessMode.MXFP4, fuse_activation=True)` (instead of
`preprocess_mxfp4_scales`), so that the epilogue reads gate/up (W1/W3) pairs from adjacent N positions.

`swiglu_limit > 0` clamps around the activation; `0.0` or None disables the clamp. The exact
clamp order depends on `activation_type` (only used together with `out_scale`):
  - `silu_and_mul` (default): clamp BEFORE the activation. The gate is clamped from above only
    (`min(gate, limit)`) and the up projection on both sides (`clamp(up, -limit, limit)`).
  - `silu_and_mul_step` (STEP models): silu FIRST, then clamp `silu(gate)` from above
    (`min(silu(gate), limit)`); the up projection is still clamped to `[-limit, limit]`.

Shape `n` must be a multiple of 64, and a bias is not supported in the fused mode.

NOTE: when `out_scale` is not None, it is re-strided **in place** on return to the N-major
layout (sfm * sfn, 1, sfm) expected by the Gemm2 SFA reader.

Returns (block_m, ceil_div(n, block_n)); the SBO-overlap signal check consumes both.)",
          py::arg("a"), py::arg("b"), py::arg("bias"), py::arg("d"), py::arg("masked_m"),
          py::arg("expected_m"),
          py::arg("recipe") = std::nullopt, py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk", py::arg("disable_ue8m0_cast") = false,
          py::arg("max_block_n") = 256,
          py::arg("enable_sbo_overlap") = false, py::arg("signal") = std::nullopt,
          py::arg("out_scale") = std::nullopt, py::arg("swiglu_limit") = std::nullopt, py::arg("configs") = std::nullopt,
          py::arg("activation_type") = std::nullopt);
    m.def("fp8_fp4_gemm_nt", &fp8_fp4_gemm_nt, py::arg("lhs"), py::arg("rhs"), py::arg("out"),
          py::arg("configs") = std::nullopt, py::arg("inv_scale") = 1.0f);
    m.def("m_grouped_gemm_fp8_fp4_bf16_nt_nopad",&m_grouped_gemm_fp8_fp4_bf16_nt_nopad,py::arg("lhs"),py::arg("rhs"),py::arg("out"),py::arg("m_indices"),py::arg("m_rows")=std::nullopt,py::arg("configs")=std::nullopt,py::arg("inv_scale")=1.0f);
    m.def("m_grouped_gemm_fp8_fp4_bf16_nt_masked",&m_grouped_gemm_fp8_fp4_bf16_nt_masked,py::arg("lhs"),py::arg("rhs"),py::arg("out"),py::arg("masked_m"),py::arg("expected_m"),py::arg("configs")=std::nullopt,py::arg("inv_scale")=1.0f);
    m.def("m_grouped_gemm_fp8_fp4_bf16_nt_fused",&m_grouped_gemm_fp8_fp4_bf16_nt_fused,py::arg("lhs"),py::arg("rhs"),py::arg("out"),py::arg("m_rows"),py::arg("expert_ids_and_cumsum"),py::arg("sorted_token_ids"),py::arg("aligned_num_m_blocks"),py::arg("configs"),py::arg("inv_scale")=1.0f);
    // W4A16 / W4FA16 GEMMs
    m.def("m_grouped_gemm_w4a16_nopad", &m_grouped_gemm_w4a16_nopad, py::arg("lhs"), py::arg("rhs_"),
          py::arg("out"), py::arg("m_indices"), py::arg("m_rows") = std::nullopt,
          py::arg("configs") = std::nullopt);
    m.def("m_grouped_gemm_w4a16_masked", &m_grouped_gemm_w4a16_masked, py::arg("lhs"), py::arg("rhs_"),
          py::arg("out"), py::arg("masked_m"), py::arg("expected_m"), py::arg("configs") = std::nullopt);
    m.def("m_grouped_gemm_w4a16_fused", &m_grouped_gemm_w4a16_fused, py::arg("lhs"), py::arg("rhs_"),
          py::arg("out"), py::arg("m_rows"), py::arg("expert_ids_and_cumsum"), py::arg("sorted_token_ids"),
          py::arg("aligned_num_m_blocks"), py::arg("configs"));
    // BF16 Fused MoE GEMM
    m.def("m_grouped_bf16_gemm_nt_fused", &m_grouped_bf16_gemm_nt_fused,
          py::arg("lhs"), py::arg("rhs"), py::arg("out"), py::arg("m_rows"),
          py::arg("expert_ids_and_cumsum"), py::arg("sorted_token_ids"),
          py::arg("aligned_num_m_blocks"), py::arg("configs"));
    // FP8/INT8 Fused MoE GEMMs
    m.def("m_grouped_fp8_gemm_nt_fused", &m_grouped_fp8_gemm_nt_fused,
          py::arg("lhs"), py::arg("rhs"), py::arg("out"), py::arg("m_rows"),
          py::arg("expert_ids_and_cumsum"), py::arg("sorted_token_ids"),
          py::arg("aligned_num_m_blocks"), py::arg("configs"));
    m.def("m_grouped_int8_gemm_nt_fused", &m_grouped_int8_gemm_nt_fused,
          py::arg("lhs"), py::arg("rhs"), py::arg("out"), py::arg("m_rows"),
          py::arg("expert_ids_and_cumsum"), py::arg("sorted_token_ids"),
          py::arg("aligned_num_m_blocks"), py::arg("configs"));
    // FP4 Fused MoE GEMM
    m.def("m_grouped_fp4_gemm_nt_fused", &m_grouped_fp4_gemm_nt_fused,
          py::arg("lhs"), py::arg("rhs"), py::arg("out"), py::arg("m_rows"),
          py::arg("expert_ids_and_cumsum"), py::arg("sorted_token_ids"),
          py::arg("aligned_num_m_blocks"), py::arg("configs"),
          py::arg("out_scale") = std::nullopt, py::arg("swiglu_limit") = std::nullopt,
          py::arg("activation_type") = std::nullopt);
    // moe_align preprocessing (C++ JIT)
    m.def("moe_align_block_size", &moe_align_block_size,
          py::arg("lhs"), py::arg("rhs"), py::arg("topk_ids"),
          py::arg("perchannel_quant") = false, py::arg("config") = std::nullopt,
          py::arg("enable_act_and_quant_fusing") = false);
    // acBLASLt GEMMs
    m.def("acblaslt_gemm_nt", &acblaslt_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);
    m.def("acblaslt_gemm_nn", &acblaslt_gemm_nn,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);
    m.def("acblaslt_gemm_tn", &acblaslt_gemm_tn,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);
    m.def("acblaslt_gemm_tt", &acblaslt_gemm_tt,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);
}

} // namespace deep_gemm::gemm
