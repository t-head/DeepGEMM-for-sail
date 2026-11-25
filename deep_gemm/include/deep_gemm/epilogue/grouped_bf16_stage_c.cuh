#pragma once

#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/arch/memory_ppu.h>
#include <deep_gemm/cute_free/arch/ppu_tc02_copy.cuh>

namespace deep_gemm {

// The existing collective lays out [all A slots][all B slots]. Map a compact
// C prefix through retired slots, skipping the last live A and B. Excess C
// is read from global memory by the epilogue, so the A/B ring is never enlarged.
template <int BM, int BN, int BK, int Stages, int ShapeK>
struct GroupedBf16CStage {
    using Element = cutlass::bfloat16_t;
    static constexpr int a_slot = BM * BK;
    static constexpr int b_slot = BN * BK;
    static constexpr int retired_a = (Stages - 1) * a_slot;
    static constexpr int retired_b = (Stages - 1) * b_slot;
    static constexpr int staged = BM * BN < retired_a + retired_b ? BM * BN : retired_a + retired_b;
    static constexpr int last_slot = ((ShapeK - 1) / BK) % Stages;

    CUTLASS_HOST_DEVICE static constexpr int physical(int logical) {
        return logical + (logical >= last_slot * a_slot ? a_slot : 0) +
                         (logical >= retired_a + last_slot * b_slot ? b_slot : 0);
    }

    // AIU writes one 128-byte-wide swizzled strip at a time. Choose a strip
    // height that divides both operand slots so a transfer never spans a hole.
    static constexpr int gcd(int a, int b) { return b ? gcd(b, a % b) : a; }
    static constexpr int width = 64;
    static constexpr bool bulk = BN % width == 0;
    static constexpr int rows = gcd(BM, gcd(a_slot, b_slot) / width);

    // Bulk copies swizzle each strip separately; holes in either layout
    // depend only on the compact strip (or the linear position).
    CUTLASS_HOST_DEVICE static constexpr int logical(int row, int col) {
        if constexpr (bulk)
            return (col / width) * BM * width + (row / rows) * rows * width;
        else
            return row * BN + col;
    }

    template <int DR, int DC, int MaxLaneRow, int MaxLaneCol>
    CUTLASS_DEVICE static int pair_offset(int lane_row, int lane_col) {
        constexpr int lo = logical(DR, DC);
        constexpr int hi = logical(DR + MaxLaneRow, DC + MaxLaneCol);
        const int row = lane_row + DR, col = lane_col + DC;
        const int gap = [&]() {
            if constexpr (physical(lo) - lo == physical(hi) - hi)
                return physical(lo) - lo;
            else {
                const int index = logical(row, col);
                return physical(index) - index;
            }
        }();
        if constexpr (bulk) {
            const int strip_col = DC / width == (DC + MaxLaneCol) / width ? DC / width : col / width;
            return gap + strip_col * (BM * width) + row * width +
                   ((col % width) ^ (((row % rows) % 8) * 8));
        } else {
            return gap + row * BN + col;
        }
    }

    // Each copy in the staged prefix fits in one retired A or B slot.
    CUTLASS_DEVICE static int owner(int logical) {
        const int dst = physical(logical);
        if (dst < Stages * a_slot) return dst / a_slot;
        return (dst - Stages * a_slot) / b_slot;
    }

    CUTLASS_DEVICE static void prefetch(char* smem, Element const* c,
                                        int m0, int n0, int m, int n, int slot) {
        auto tile = reinterpret_cast<Element*>(smem);
        if constexpr (bulk) {
            if (threadIdx.x < 32) {
                CUTLASS_PRAGMA_UNROLL
                for (int col = 0; col < BN; col += width) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int row = 0; row < BM; row += rows) {
                        const int strip = col * BM + row * width;
                        if (strip >= staged || owner(strip) != slot) continue;
                        arch::aiu_bulk_tensor_load_bf16<rows, width>(
                            tile + physical(strip), c, m, n, m0 + row, n0 + col);
                    }
                }
            }
        } else {
            // The host requires 16-byte alignment on every row of C.
            for (int pos = threadIdx.x * 8; pos < staged; pos += blockDim.x * 8) {
                if (owner(pos) != slot) continue;
                const int row = m0 + pos / BN, col = n0 + pos % BN;
                const bool valid = row < m && col < n;
                auto src = valid ? c + int64_t(row) * n + col : c;
                auto dst = arch::smem_ptr_to_uint(tile + physical(pos));
                asm volatile("ppu.cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                             :: "r"(dst), "l"(src), "r"(valid ? 16 : 0) : "memory");
            }
        }
    }
};

// Preserve the normal epilogue store. For the BF16 16x16 MMA each consecutive
// pair of accumulator elements covers adjacent columns and needs one 32-bit load.
template <class Base, int BM, int BN, int BK, int Stages, int ShapeK>
struct Bf16StagedCEpilogueNoTsm : Base {
    using Params = typename Base::Params;
    using SharedStorage = typename Base::SharedStorage;
    Params const& epilogue_params;

    CUTLASS_HOST_DEVICE Bf16StagedCEpilogueNoTsm(Params const& p, SharedStorage const& s)
        : Base(p, s), epilogue_params(p) {}
    using CStage = GroupedBf16CStage<BM, BN, BK, Stages, ShapeK>;

    template <class Problem, class Tile, class Coord, class Fragment, class Mma, class Residue>
    CUTLASS_DEVICE void operator()(Problem problem, Tile tile, Coord coord, Fragment& accum,
                                   Mma mma, Residue residue, int tid, char* smem) {
        using namespace cute;
        auto identity = make_identity_tensor(take<0, 2>(tile));
        auto coords = mma.get_thread_slice(tid).partition_C(identity);
        // A partition of an identity tensor is an affine fragment layout:
        // lane origin + constant register coordinate. The last register gives
        // an upper bound for the origin since its translated position fits.
        constexpr auto ref = Mma{}.get_thread_slice(_0{}).partition_C(
            make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{})));
        constexpr int count = size(ref);
        static_assert(count % 2 == 0);
        constexpr auto last = ref(Int<count - 1>{});
        constexpr int max_lr = BM - 1 - int(get<0>(last));
        constexpr int max_lc = BN - 1 - int(get<1>(last));
        const auto lane = coords(_0{});
        const int lr = get<0>(lane), lc = get<1>(lane);
        auto cache = reinterpret_cast<cutlass::bfloat16_t const*>(smem);
        uint32_t dependency = 0;
        auto load_pair = [&](auto index) {
            constexpr int i = decltype(index)::value;
            constexpr auto xy = ref(Int<i>{});
            constexpr int dr = get<0>(xy), dc = get<1>(xy);
            // The compact logical index is monotone in each tile coordinate.
            // Resolve entire fragments at compile time whenever possible.
            constexpr int first = CStage::logical(dr, dc);
            constexpr int last = CStage::logical(dr + max_lr, dc + max_lc);
            auto staged_pair = [&]() {
                const int offset = CStage::template pair_offset<dr, dc, max_lr, max_lc>(lr, lc);
                return *reinterpret_cast<uint32_t const*>(cache + offset + dependency);
            };
            auto global_pair = [&]() -> uint32_t {
                const int row = lr + dr, col = lc + dc;
                if (row >= int(get<0>(residue)) || col >= int(get<1>(residue))) return 0;
                // StageC requires an 8-element-aligned pitch; a column pair is
                // in bounds as a unit, including the final partially valid tile.
                auto row0 = int64_t(get<0>(coord)) * BM + row;
                auto col0 = int64_t(get<1>(coord)) * BN + col;
                auto addr = epilogue_params.ptr_C + row0 * get<0>(epilogue_params.dC) + col0;
                return *reinterpret_cast<uint32_t const*>(addr + dependency);
            };
            if constexpr (last < CStage::staged) {
                return staged_pair();
            } else if constexpr (first >= CStage::staged) {
                return global_pair();
            } else {
                return CStage::logical(lr + dr, lc + dc) < CStage::staged ?
                       staged_pair() : global_pair();
            }
        };

        // Keep a small batch of BF16 pairs in flight before adding them.
        constexpr int batch = 4;
        for_each(make_int_sequence<(count / 2 + batch - 1) / batch>{}, [&](auto block) {
            constexpr int first = decltype(block)::value * batch;
            constexpr int n = count / 2 - first < batch ? count / 2 - first : batch;
            uint32_t pairs[n];
            for_each(make_int_sequence<n>{}, [&](auto j) {
                pairs[j] = load_pair(Int<2 * (first + decltype(j)::value)>{});
            });
            for_each(make_int_sequence<n>{}, [&](auto j) {
                constexpr int i = 2 * (first + decltype(j)::value);
                accum(i)     += __uint_as_float(pairs[j] << 16);
                accum(i + 1) += __uint_as_float(pairs[j] & 0xffff0000u);
            });
            if constexpr (first + n < count / 2) {
                // Grouped kernels launch whole warps. The low five bits of
                // blockDim.x are zero; consuming all results through this
                // runtime zero orders the next batch without a CTA fence.
                uint32_t done = 0;
                for_each(make_int_sequence<n>{}, [&](auto j) {
                    constexpr int i = 2 * (first + decltype(j)::value);
                    done |= __float_as_uint(accum(i)) | __float_as_uint(accum(i + 1));
                });
                dependency = (blockDim.x & 31u) & done;
            }
        });
        Base::operator()(problem, tile, coord, accum, mma, residue, tid, smem);
    }
};

} // namespace deep_gemm
