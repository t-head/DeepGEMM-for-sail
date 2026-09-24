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
//   - All quantities are computed purely from template parameters (constexpr), no
//     runtime cost.

#include "cute/config.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/numeric/math.hpp"
#include <deep_gemm/common/utils_rtc.cuh>

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

  // Blockwise quantization keeps a running dequant accumulator alongside the raw MMA
  // accumulator, i.e. 2x the ACC VREG footprint (see fp8_occ_model_doc.md section 5).
  static constexpr int kAccCopies = kIsBlockwise ? 2 : 1;

  // ---- CTA geometry ----
  static constexpr int kWarpOnM     = BM / WM;
  static constexpr int kWarpOnN     = BN / WN;
  static constexpr int kWarpOnK     = BK / WK;
  static constexpr int kWarpsPerCta = kWarpOnM * kWarpOnN * kWarpOnK;

  // =========================================================================
  // 1) SMEM occupancy
  // =========================================================================
  static_assert(kSmemBytesPerCta <= kTsmPerCu,
                "SMEM footprint must fit in the per-CU TSM limit");
  static constexpr int kSmemOcc = kTsmPerCu / kSmemBytesPerCta;

  // =========================================================================
  // 2) Thread/warp occupancy
  // =========================================================================
  static_assert(kWarpsPerCta * 32 <= kMaxThreadsPerCta,
                "CTA thread count exceeds the per-block thread limit");
  static constexpr int kThreadOcc = kMaxWarpsPerCu / kWarpsPerCta;

  // =========================================================================
  // 3) VREG occupancy
  // =========================================================================
  static constexpr int kAcc   = WM * WN * kBitsAcc / 8 / 128;
  static constexpr int kMmaKA = 256 / kBitsA;
  static constexpr int kMmaKB = 256 / kBitsB;
  // A/B operand VREG footprint carries a `* 2` factor that models double-buffered
  // (ping-pong) operand registers: the next MMA operand set is loaded while the
  // current one is being consumed, so up to two full copies may be live at once.
  //
  // This 2x is a conservative upper bound, NOT an exact figure. The compiler's real
  // allocation falls somewhere between 1x and 2x, because it can partially overlap
  // and reuse operand registers instead of always keeping two full live copies.
  static constexpr int kVregA = WM * kMmaKA * 2 * kBitsA / 8 / 128;
  static constexpr int kVregB = WN * kMmaKB * 2 * kBitsB / 8 / 128;
  // The current PPU 16x16 MMA accumulator layout gives each lane two M rows.
  // FP32 SFA broadcasts over N; only the current K scale fragment is resident.
  static constexpr int kMmaRows = 16;
  static constexpr int kMmaRowsPerThread = 2;
  static constexpr int kVregSFA =
      kIsBlockwise ? cute::ceil_div(WM, kMmaRows) * kMmaRowsPerThread : 0;
  static constexpr int kVregPerWarp =
      kAccCopies * kAcc + kVregA + kVregB + kVregSFA + kVregOverhead;

  static constexpr int kWePerCu       = 8;
  static constexpr int kVregPerWe     = kTotalVregPerCu / kWePerCu;
  static constexpr int kMaxWarpsPerWe = kMaxWarpsPerCu / kWePerCu;
  // A single warp's VREG footprint must fit within one WE's VREG budget.
  static_assert(kVregPerWarp <= kVregPerWe,
                "per-warp VREG exceeds the per-WE budget; occupancy would be 0");
  static constexpr int kWarpsPerWe =
      cute::min(kMaxWarpsPerWe, kVregPerWe / kVregPerWarp);

  static constexpr int kVregOcc = kWePerCu * kWarpsPerWe >= kWarpsPerCta
            ? (kWePerCu * kWarpsPerWe) / kWarpsPerCta : 1;

  // =========================================================================
  // Final occupancy = min of the three independent constraints.
  // =========================================================================
  static constexpr int kOcc =
      cute::min(cute::min(kSmemOcc, kThreadOcc), kVregOcc);

  static constexpr uint32_t kMinBlocksPerMultiprocessor = uint32_t(kOcc);
};

} // namespace deep_gemm
