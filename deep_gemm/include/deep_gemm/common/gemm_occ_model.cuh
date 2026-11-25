#pragma once

// GemmOccModel: a constexpr occupancy (MinBlocksPerMultiprocessor) estimator for
// cutlass3-style GEMM kernels on PPU. C++ JIT kernel templates
// (bf16/int8/fp8 cutlass3 host wrappers) derive
// `__launch_bounds__(MaxThreadsPerBlock, MinBlocksPerMultiprocessor)` from
// tile shape, element sizes, and the epilogue TSM mode.
//
// Notes:
//   - This header does not depend on any `cutlass::` type; callers pass the
//     element sizes in bits (e.g. cute's `sizeof_bits_v<ElementX>`), so it can be
//     included independently of the numeric type headers.
//   - `kIsBlockwise` defaults to false (non-blockwise kernels); blockwise-quantized
//     kernels must pass true explicitly. It accounts for doubled ACC VREGs and
//     the current SFA fragment; additional costs come from kVregOverhead.
//   - `kSmemBytesPerCta` is the kernel's real per-CTA shared-memory footprint,
//     passed as `GemmKernel::SharedStorageSize` (== sizeof(GemmKernel::SharedStorage),
//     the mainloop/epilogue union). It is the authoritative value the launch
//     requests, so no scalar SMEM estimate (stages / bits / epilogue mode) is needed.
//   - Hardware constants are mandatory template parameters, injected by
//     the host-side JIT generator (bf16/int8/fp8_gemm.hpp generate_impl) from
//     hggcDeviceGetAttribute queries, so the model carries no hardcoded spec
//     values and new SKUs require no code change.
//   - GemmOccModel evaluates the shared estimator at compile time. Runtime
//     configuration selection calls estimate_gemm_occupancy with the same formulas.

#include <hggc_runtime_api.h>

#include <cute/config.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/numeric/math.hpp>
#include <deep_gemm/common/math.cuh>

namespace deep_gemm {

// ---------------------------------------------------------------------------
// PpuHwParams: process-wide singleton snapshot of device hardware constants,
// queried from the driver once (device 0) at JIT-codegen time (host side) and
// injected as constexpr literals into generated kernel source.
//
// PPU nodes are homogeneous (all devices same SKU) and these values are static
// per SKU, so one process-wide query suffices.
// ---------------------------------------------------------------------------
class PpuHwParams {
public:
    int tsm_per_cu          = 0;
    int max_threads_per_cta = 0;
    int max_warps_per_cu    = 0;
    int total_vreg_per_cu   = 0;

    static const PpuHwParams& instance() {
        static const PpuHwParams inst;
        return inst;
    }

    PpuHwParams(const PpuHwParams&)            = delete;
    PpuHwParams& operator=(const PpuHwParams&) = delete;

private:
    PpuHwParams() {
        int threads_cu = 0, regs = 0, warp_size = 0;
        hggcDeviceGetAttribute(&tsm_per_cu, hggcDevAttrMaxSharedMemoryPerMultiprocessor, 0);
        hggcDeviceGetAttribute(&max_threads_per_cta, hggcDevAttrMaxThreadsPerBlock, 0);
        hggcDeviceGetAttribute(&threads_cu, hggcDevAttrMaxThreadsPerMultiProcessor, 0);
        hggcDeviceGetAttribute(&regs, hggcDevAttrMaxRegistersPerMultiprocessor, 0);
        hggcDeviceGetAttribute(&warp_size, hggcDevAttrWarpSize, 0);
        max_warps_per_cu  = threads_cu / warp_size;
        total_vreg_per_cu = regs / 32;
    }
};

// Shared by runtime configuration selection and the constexpr launch-bounds wrapper.
// Resource-infeasible runtime candidates return occ == 0.
struct GemmOccEstimate {
    int acc_copies = 0;
    int warp_on_m = 0;
    int warp_on_n = 0;
    int warp_on_k = 0;
    int warps_per_cta = 0;
    int smem_occ = 0;
    int thread_occ = 0;
    int acc = 0;
    int mma_k_a = 0;
    int mma_k_b = 0;
    int vreg_a = 0;
    int vreg_b = 0;
    int mma_rows = 0;
    int mma_rows_per_thread = 0;
    int vreg_sfa = 0;
    int vreg_per_warp = 0;
    int we_per_cu = 0;
    int vreg_per_we = 0;
    int max_warps_per_we = 0;
    int warps_per_we = 0;
    int vreg_occ = 0;
    int occ = 0;
};

constexpr GemmOccEstimate estimate_gemm_occupancy(
    int bm, int bn, int bk, int wm, int wn, int wk, int smem_bytes_per_cta,
    int bits_a, int bits_b, int bits_acc, int tsm_per_cu, int max_threads_per_cta,
    int max_warps_per_cu, int total_vreg_per_cu,
    bool is_blockwise = false, int vreg_overhead = 0) {
  GemmOccEstimate result{};
  if (bm <= 0 || bn <= 0 || bk <= 0 || wm <= 0 || wn <= 0 || wk <= 0 ||
      bm % wm || bn % wn || bk % wk || bits_a <= 0 || bits_b <= 0 ||
      256 % bits_a || 256 % bits_b || bits_acc <= 0 || bits_acc % 8 ||
      smem_bytes_per_cta <= 0 || smem_bytes_per_cta > tsm_per_cu ||
      (bm / wm) * (bn / wn) * (bk / wk) * 32 > max_threads_per_cta ||
      max_warps_per_cu <= 0 || total_vreg_per_cu <= 0)
      return result;
  // Blockwise quantization keeps a running dequant accumulator alongside the raw MMA
  // accumulator, i.e. 2x the ACC VREG footprint (see fp8_occ_model_doc.md section 5).
  result.acc_copies = is_blockwise ? 2 : 1;

  // ---- CTA geometry ----
  result.warp_on_m = bm / wm;
  result.warp_on_n = bn / wn;
  result.warp_on_k = bk / wk;
  result.warps_per_cta = result.warp_on_m * result.warp_on_n * result.warp_on_k;

  // =========================================================================
  // 1) SMEM occupancy
  // =========================================================================
  result.smem_occ = tsm_per_cu / smem_bytes_per_cta;

  // =========================================================================
  // 2) Thread/warp occupancy
  // =========================================================================
  result.thread_occ = max_warps_per_cu / result.warps_per_cta;

  // =========================================================================
  // 3) VREG occupancy
  // =========================================================================
  result.acc = wm * wn * bits_acc / 8 / 128;
  result.mma_k_a = 256 / bits_a;
  result.mma_k_b = 256 / bits_b;
  // A/B operand VREG footprint carries a `* 2` factor that models double-buffered
  // (ping-pong) operand registers: the next MMA operand set is loaded while the
  // current one is being consumed, so up to two full copies may be live at once.
  //
  // This 2x is a conservative upper bound, NOT an exact figure. The compiler's real
  // allocation falls somewhere between 1x and 2x, because it can partially overlap
  // and reuse operand registers instead of always keeping two full live copies.
  result.vreg_a = wm * result.mma_k_a * 2 * bits_a / 8 / 128;
  result.vreg_b = wn * result.mma_k_b * 2 * bits_b / 8 / 128;
  // The current PPU 16x16 MMA accumulator layout gives each lane two M rows.
  // FP32 SFA broadcasts over N; only the current K scale fragment is resident.
  result.mma_rows = 16;
  result.mma_rows_per_thread = 2;
  result.vreg_sfa =
      is_blockwise ? math::ceil_div(wm, result.mma_rows) * result.mma_rows_per_thread : 0;
  result.vreg_per_warp =
      result.acc_copies * result.acc + result.vreg_a + result.vreg_b + result.vreg_sfa + vreg_overhead;

  result.we_per_cu = 8;
  result.vreg_per_we = total_vreg_per_cu / result.we_per_cu;
  result.max_warps_per_we = max_warps_per_cu / result.we_per_cu;
  // A single warp's VREG footprint must fit within one WE's VREG budget.
  if (result.vreg_per_warp <= 0 || result.vreg_per_warp > result.vreg_per_we)
      return result;
  result.warps_per_we =
      cute::min(result.max_warps_per_we, result.vreg_per_we / result.vreg_per_warp);

  result.vreg_occ = result.we_per_cu * result.warps_per_we >= result.warps_per_cta
            ? (result.we_per_cu * result.warps_per_we) / result.warps_per_cta : 1;

  // =========================================================================
  // Final occupancy = min of the three independent constraints.
  // =========================================================================
  result.occ =
      cute::min(cute::min(result.smem_occ, result.thread_occ), result.vreg_occ);
  return result;
}

// ---------------------------------------------------------------------------
// GemmOccModel
//
// Template params:
//   BM/BN/BK          : CTA tile shape
//   WM/WN/WK          : warp tile shape; WK defaults to BK (no K-split/reduction).
//                        BF16 kernels using WarpOnK reduction can pass a smaller WK.
//   kSmemBytesPerCta  : kernel per-CTA SMEM footprint; pass GemmKernel::SharedStorageSize
//   kBitsA/B/Acc      : element size in bits, e.g. cute::sizeof_bits_v<ElementA>
//                        (16 for bf16, 8 for fp8/int8, 4 for fp4); expressed in
//                        bits, not bytes, so packed formats like fp4 (0.5
//                        byte/element) are representable.
//   kIsBlockwise      : whether the kernel is blockwise-quantized (decides whether VREG
//                        carries doubled ACC and the current SFA fragment); defaults to false,
//                        blockwise kernels pass true.
//   kVregOverhead     : caller-provided additional VREG budget, including cp.async
//                        addresses (SFA data is counted separately); defaults to 0.
//   kTsmPerCu / kMaxThreadsPerCta / kMaxWarpsPerCu / kTotalVregPerCu:
//   REQUIRED hardware constants injected by the host-side JIT generator (see Notes
//   above); no defaults - the model must not hardcode any spec value.
//
// Usage (sketch):
//   // kIsBlockwise may be omitted (non-blockwise default).
//   using Occ = GemmOccModel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BLOCK_K,
//                            GemmKernel::SharedStorageSize, cute::sizeof_bits_v<ElementA>,
//                            cute::sizeof_bits_v<ElementB>,
//                            cute::sizeof_bits_v<ElementAcc>,
//                            kTsmPerCu, kMaxThreadsPerCta, kMaxWarpsPerCu,
//                            kTotalVregPerCu>;
//   // Blockwise-quantized kernel: pass kIsBlockwise = true explicitly.
//   using OccBlk = GemmOccModel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BLOCK_K,
//                            GemmKernel::SharedStorageSize, cute::sizeof_bits_v<ElementA>,
//                            cute::sizeof_bits_v<ElementB>,
//                            cute::sizeof_bits_v<ElementAcc>,
//                            kTsmPerCu, kMaxThreadsPerCta, kMaxWarpsPerCu,
//                            kTotalVregPerCu, true>;
//   static constexpr uint32_t MinBlocksPerMultiprocessor = Occ::kMinBlocksPerMultiprocessor;
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK,
          int WM, int WN, int WK,
          int kSmemBytesPerCta,
          int kBitsA, int kBitsB, int kBitsAcc,
          int kTsmPerCu,
          int kMaxThreadsPerCta,
          int kMaxWarpsPerCu,
          int kTotalVregPerCu,
          bool kIsBlockwise = false,
          int kVregOverhead = 0>
struct GemmOccModel {
  // ---- Template parameter validity (fail fast on illegal instantiations) ----
  static_assert(kSmemBytesPerCta > 0,
                "kSmemBytesPerCta must be positive (pass GemmKernel::SharedStorageSize)");
  static_assert(BM % WM == 0 && BN % WN == 0 && BK % WK == 0,
                "CTA tile must be divisible by the warp tile");
  static_assert(256 % kBitsA == 0 && 256 % kBitsB == 0,
                "kBitsA/kBitsB must divide 256 (mma_k = 256 / kBits must be integral)");
  static_assert(kBitsAcc % 8 == 0,
                "kBitsAcc must be byte-aligned (ACC VREG uses whole bytes)");

  static constexpr auto kEstimate = estimate_gemm_occupancy(
      BM, BN, BK, WM, WN, WK, kSmemBytesPerCta, kBitsA, kBitsB, kBitsAcc,
      kTsmPerCu, kMaxThreadsPerCta, kMaxWarpsPerCu, kTotalVregPerCu,
      kIsBlockwise, kVregOverhead);

  static constexpr int kAccCopies = kEstimate.acc_copies;
  static constexpr int kWarpOnM = kEstimate.warp_on_m;
  static constexpr int kWarpOnN = kEstimate.warp_on_n;
  static constexpr int kWarpOnK = kEstimate.warp_on_k;
  static constexpr int kWarpsPerCta = kEstimate.warps_per_cta;
  static constexpr int kSmemOcc = kEstimate.smem_occ;
  static constexpr int kThreadOcc = kEstimate.thread_occ;
  static constexpr int kAcc = kEstimate.acc;
  static constexpr int kMmaKA = kEstimate.mma_k_a;
  static constexpr int kMmaKB = kEstimate.mma_k_b;
  static constexpr int kVregA = kEstimate.vreg_a;
  static constexpr int kVregB = kEstimate.vreg_b;
  static constexpr int kMmaRows = kEstimate.mma_rows;
  static constexpr int kMmaRowsPerThread = kEstimate.mma_rows_per_thread;
  static constexpr int kVregSFA = kEstimate.vreg_sfa;
  static constexpr int kVregPerWarp = kEstimate.vreg_per_warp;
  static constexpr int kWePerCu = kEstimate.we_per_cu;
  static constexpr int kVregPerWe = kEstimate.vreg_per_we;
  static constexpr int kMaxWarpsPerWe = kEstimate.max_warps_per_we;
  static constexpr int kWarpsPerWe = kEstimate.warps_per_we;
  static constexpr int kVregOcc = kEstimate.vreg_occ;
  static constexpr int kOcc = kEstimate.occ;

  static_assert(kSmemBytesPerCta <= kTsmPerCu,
                "SMEM footprint must fit in the per-CU TSM limit");
  static_assert((BM / WM) * (BN / WN) * (BK / WK) * 32 <= kMaxThreadsPerCta,
                "CTA thread count exceeds the per-block thread limit");
  static_assert(kVregPerWarp <= kVregPerWe,
                "per-warp VREG exceeds the per-WE budget; occupancy would be 0");

  static constexpr uint32_t kMinBlocksPerMultiprocessor = uint32_t(kOcc);
};

} // namespace deep_gemm
