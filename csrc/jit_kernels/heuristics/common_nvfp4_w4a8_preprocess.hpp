#pragma once
#include <algorithm>
#include <torch/torch.h>

namespace deep_gemm::nvfp4::detail {

// Four checkpoint bytes contain eight FP4 codes. Keep their magnitudes as
// byte-permute selectors and move their signs to the decoded byte MSBs.
static torch::Tensor pack_nvfp4_bytes(const torch::Tensor& bytes) {
    auto shape = bytes.sizes().vec();
    shape.pop_back();
    auto packed = torch::zeros(shape, bytes.options().dtype(torch::kInt32));
    for (int i = 0; i < 4; ++i) {
        const auto byte = bytes.select(-1, i).to(torch::kInt32);
        packed.bitwise_or_(torch::bitwise_left_shift(torch::bitwise_and(byte, 0x77), 8 * i));
        const int low_sign_shift = i < 2 ? 4 + 16 * i : 16 * (i - 2);
        const int high_sign_shift = i < 2 ? 8 + 16 * i : 4 + 16 * (i - 2);
        packed.bitwise_or_(torch::bitwise_left_shift(torch::bitwise_and(byte, 8), low_sign_shift));
        packed.bitwise_or_(torch::bitwise_left_shift(torch::bitwise_and(byte, 128), high_sign_shift));
    }
    return packed;
}

// Checkpoint layout is low-nibble-first uint8 [E,N,K/2]. Both output layouts
// remain W4; temporary storage is bounded independently of the expert count.
static std::pair<torch::Tensor, torch::Tensor> preprocess_fp4_weights_and_scales(
        const torch::Tensor& weight, const torch::Tensor& scales, bool fp8_layout) {
    using namespace torch::indexing;
    TORCH_CHECK(weight.dim() == 3 && weight.scalar_type() == torch::kUInt8,
                "W4A8 packed weights must be uint8 [E,N,K/2]");
    const int64_t e = weight.size(0), n = weight.size(1), k = weight.size(2) * 2;
    TORCH_CHECK(e > 0 && n > 0 && n % 64 == 0 && k > 0 && k % 32 == 0,
                "W4A8 requires E>0, N divisible by 64 and K divisible by 32");
    TORCH_CHECK(scales.device() == weight.device() &&
                scales.sizes() == std::vector<int64_t>({e, n, k / 16}) &&
                scales.scalar_type() == torch::kFloat8_e4m3fn,
                "W4A8 requires matching [E,N,K/group_size] native scales");

    // The fused decoder uses independent K32 x N64 tiles. Gather their four
    // packed bytes directly instead of expanding all FP4 codes to integers.
    torch::Tensor n_order, k_order;
    if (!fp8_layout) {
        std::vector<int64_t> n_index(256), k_index(256 * 4);
        constexpr int byte_offsets[] = {0, 1, 8, 9};
        for (int col = 0; col < 256; ++col) {
            n_index[col] = col % 8 * 8 + col / 32;
            for (int byte = 0; byte < 4; ++byte)
                k_index[col * 4 + byte] = (col / 8 % 4) * 2 + byte_offsets[byte];
        }
        const auto options = torch::TensorOptions().dtype(torch::kInt64);
        n_order = at::tensor(n_index, options).to(weight.device()).view({256, 1});
        k_order = at::tensor(k_index, options).to(weight.device()).view({256, 4});
    }
    const auto weight_shape = fp8_layout ? std::vector<int64_t>{e, n, k / 8}
                                        : std::vector<int64_t>{e, k / 32, 4 * n};
    auto packed = torch::empty(weight_shape, weight.options().dtype(torch::kInt32));
    auto encoded = fp8_layout ? scales.contiguous().view(torch::kUInt8)
                             : torch::empty({e, k / 16, n}, scales.options().dtype(torch::kUInt8));
    auto valid_scales = torch::ones({}, scales.options().dtype(torch::kBool));

    // At most 8 MiB of packed input per chunk, including when a single expert
    // exceeds this size. Align chunk boundaries to the decoder's K32/N64 tiles.
    constexpr int64_t MaxElements = 2 * (8 << 20);
    const int64_t chunk_k = std::min(k, MaxElements / 64) / 32 * 32;
    const int64_t chunk_n = std::min(n, MaxElements / chunk_k) / 64 * 64;
    for (int64_t expert = 0; expert < e; ++expert) {
        for (int64_t ni = 0; ni < n; ni += chunk_n) {
            const int64_t ne = std::min(n, ni + chunk_n), rows = ne - ni;
            for (int64_t ki = 0; ki < k; ki += chunk_k) {
                const int64_t ke = std::min(k, ki + chunk_k), cols = ke - ki;
                const auto scale = scales.slice(0, expert, expert + 1).slice(1, ni, ne)
                                         .slice(2, ki / 16, ke / 16).contiguous();
                {
                    const auto values = scale.to(torch::kFloat);
                    valid_scales.logical_and_(torch::isfinite(values).logical_and(values.ge(0)).all());
                }
                const auto block = weight.slice(0, expert, expert + 1).slice(1, ni, ne)
                                         .slice(2, ki / 2, ke / 2).contiguous();
                if (fp8_layout) {
                    const auto bytes = block.view({1, rows, cols / 8, 4});
                    packed.slice(0, expert, expert + 1).slice(1, ni, ne).slice(2, ki / 8, ke / 8)
                          .copy_(pack_nvfp4_bytes(bytes));
                } else {
                    const auto tiles = block.view({1, rows / 64, 64, cols / 32, 16})
                                            .permute({0, 3, 1, 2, 4});
                    const auto ordered = tiles.index({Ellipsis, n_order, k_order});
                    packed.slice(0, expert, expert + 1).slice(1, ki / 32, ke / 32).slice(2, 4 * ni, 4 * ne)
                          .copy_(pack_nvfp4_bytes(ordered).view({1, cols / 32, 4 * rows}));
                    const auto bytes = scale.view(torch::kUInt8).view({1, rows / 64, 8, 8, cols / 16})
                                            .permute({0, 4, 1, 3, 2}).contiguous();
                    encoded.slice(0, expert, expert + 1).slice(1, ki / 16, ke / 16).slice(2, ni, ne)
                           .copy_(bytes.view({1, cols / 16, rows}));
                }
            }
        }
    }
    TORCH_CHECK(valid_scales.item<bool>(), "NVFP4 block scales must be finite and nonnegative");
    return {packed, encoded};
}

} // namespace deep_gemm::nvfp4::detail
