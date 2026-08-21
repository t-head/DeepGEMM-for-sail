#pragma once

// GemmOccModel: a constexpr occupancy (MinBlocksPerMultiprocessor) estimator for
// cutlass3-style GEMM kernels on PPU. C++ JIT kernel templates
// (bf16/int8/fp8 cutlass3 host wrappers) derive
// `__launch_bounds__(MaxThreadsPerBlock, MinBlocksPerMultiprocessor)` from
// tile-shape / dtype-size / blockwise-ness.
//
// Notes:
//   - This header does not depend on any `cutlass::` type; callers pass the
//     element sizes in bits (e.g. cute's `sizeof_bits_v<ElementX>`), so it can be
//     included independently of the numeric type headers.
//   - `kIsBlockwise` defaults to false (non-blockwise kernels); blockwise-quantized
//     kernels must pass true explicitly. It only affects the SMEM scale buffers and
//     the doubled ACC VREG footprint; everything else is dtype/GemmType agnostic.
//   - Hardware constants are mandatory trailing template parameters, injected by
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
//   kNumStages        : pipeline stage count
//   kBitsA/B/Acc      : element size in bits, e.g. cute::sizeof_bits_v<ElementA>
//                        (16 for bf16, 8 for fp8/int8, 4 for fp4); expressed in
//                        bits, not bytes, so packed formats like fp4 (0.5
//                        byte/element) are representable.
//   kIsBlockwise      : whether the kernel is blockwise-quantized (decides whether the
//                        SMEM scale buffer exists and whether VREG carries a doubled
//                        ACC accumulator); defaults to false, blockwise kernels pass
//                        true.
//   kVregOverhead     : optional per-warp VREG slack for address/loop-variable
//                        registers that this model does not otherwise account for
//                        (see fp8_occ_model_doc.md known limitations); defaults to 0,
//                        matching the existing fp8_occ_model.py behavior. May need
//                        tuning later against measured `show_log=1` VREG counts.
//   kTsmPerCu / kMaxThreadsPerCta / kMaxWarpsPerCu / kTotalVregPerCu:
//   REQUIRED hardware constants injected by the host-side JIT generator (see Notes
//   above); no defaults - the model must not hardcode any spec value.
//
// Usage (sketch):
//   // kIsBlockwise may be omitted (non-blockwise default).
//   using Occ = GemmOccModel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BLOCK_K,
//                            kNumStages, cute::sizeof_bits_v<ElementA>,
//                            cute::sizeof_bits_v<ElementB>,
//                            cute::sizeof_bits_v<ElementAcc>,
//                            kTsmPerCu, kMaxThreadsPerCta, kMaxWarpsPerCu,
//                            kTotalVregPerCu>;
//   // Blockwise-quantized kernel: pass kIsBlockwise = true explicitly.
//   using OccBlk = GemmOccModel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BLOCK_K,
//                            kNumStages, cute::sizeof_bits_v<ElementA>,
//                            cute::sizeof_bits_v<ElementB>,
//                            cute::sizeof_bits_v<ElementAcc>,
//                            kTsmPerCu, kMaxThreadsPerCta, kMaxWarpsPerCu,
//                            kTotalVregPerCu, true>;
//   static constexpr uint32_t MinBlocksPerMultiprocessor = Occ::kMinBlocksPerMultiprocessor;
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK,
          int WM, int WN, int WK,
          int kNumStages,
          int kBitsA, int kBitsB, int kBitsAcc,
          int kTsmPerCu,
          int kMaxThreadsPerCta,
          int kMaxWarpsPerCu,
          int kTotalVregPerCu,
          bool kIsBlockwise = false,
          int kVregOverhead = 0>
struct GemmOccModel {
  // ---- Template parameter validity (fail fast on illegal instantiations) ----
  static_assert(kNumStages >= 1, "kNumStages must be at least 1");
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
  static constexpr int64_t kSmemA =
      cute::round_up(int64_t(kNumStages) * BM * BK * kBitsA / 8, int64_t{128});
  static constexpr int64_t kSmemB =
      cute::round_up(int64_t(kNumStages) * BN * BK * kBitsB / 8, int64_t{128});
  // fp32 dequant scale size in bytes, used by the blockwise SMEM scale buffers.
  static constexpr int kScaleBytes = 4;

  static constexpr int64_t kSmemScaleA =
      kIsBlockwise
          ? cute::round_up(int64_t(kNumStages) * BM * (BK / 128) * kScaleBytes, int64_t{128})
          : 0;
  static constexpr int64_t kSmemScaleB =
      kIsBlockwise
          ? cute::round_up(int64_t(kNumStages) * ceil_div(BN, 128) * (BK / 128) * kScaleBytes,
                     int64_t{256})
          : 0;
  static constexpr int64_t kSmemBytes = kSmemA + kSmemB + kSmemScaleA + kSmemScaleB;
  static_assert(kSmemBytes <= kTsmPerCu,
                "SMEM footprint must fit in the per-CU TSM limit");
  static constexpr int kSmemOcc = int(kTsmPerCu / kSmemBytes);

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
  static constexpr int kVregA = WM * kMmaKA * 2 * kBitsA / 8 / 128;
  static constexpr int kVregB = WN * kMmaKB * 2 * kBitsB / 8 / 128;
  static constexpr int kVregPerWarp = kAccCopies * kAcc + kVregA + kVregB + kVregOverhead;
  // Per-warp VREG cap: no driver attribute exists; hardcoded in the model.
  static_assert(kVregPerWarp <= 256,
                "per-warp VREG exceeds the hardware limit");

  static constexpr int kWePerCu       = 8;
  static constexpr int kVregPerWe     = kTotalVregPerCu / kWePerCu;
  static constexpr int kMaxWarpsPerWe = kMaxWarpsPerCu / kWePerCu;
  static constexpr int kWarpsPerWe =
      cute::min(kMaxWarpsPerWe, kVregPerWe / kVregPerWarp);
  static_assert(kWePerCu * kWarpsPerWe >= kWarpsPerCta,
                "VREG budget cannot host a single CTA");
  static constexpr int kVregOcc = (kWePerCu * kWarpsPerWe) / kWarpsPerCta;

  // =========================================================================
  // Final occupancy = min of the three independent constraints.
  // =========================================================================
  static constexpr int kOcc =
      cute::min(cute::min(kSmemOcc, kThreadOcc), kVregOcc);

  static constexpr uint32_t kMinBlocksPerMultiprocessor = uint32_t(kOcc);
};

} // namespace deep_gemm
