# NVFP4 W4A8 GEMMs (M890)

Accepts `lhs=(FP8 activation, FP32 scale)` and `rhs=(prepared weight, native K16 scale)`;
all entry points write BF16 output on M890.

| API | Activation | Output |
| :--- | :--- | :--- |
| `gemm_fp8_nvfp4_bf16_nt` | `[M,K]` | `[M,N]` |
| `m_grouped_gemm_fp8_nvfp4_bf16_nt_nopad` | `[M,K]`, grouped by expert | `[M,N]` |
| `m_grouped_gemm_fp8_nvfp4_bf16_nt_masked` | `[E,T,K]` | `[E,T,N]`, valid rows only |
| `m_grouped_gemm_fp8_nvfp4_bf16_nt_fused` | `[num_token,K]`, original token order | `[num_token*topk,N]`, grouped by expert |

Requires `N % 64 == 0`, `K % 32 == 0`, and contiguous operands/metadata on the
same device; decode-to-FP8 also accepts strided activation scales. Routing/count
metadata is int32. Dense calls may omit the E=1 dimension from prepared weights.

## Scales and preparation

- Activations are E4M3FN. `lhs_scale_group_size=0` (default) uses FP32 `[M,1]`
  per-token scales; G=16 or a positive multiple of 32 uses `[M,ceil(K/G)]`.
  Masked scales add E. `(activation, None)` selects unit per-token scales in fused execution.
- Checkpoint weights are uint8 `[E,N,K/2]` packed E2M1, even K in the low nibble,
  with E4M3FN K16 scales `[E,N,K/16]`. Preprocessing returns int32 weights and
  uint8 scale bytes. Blockwise execution uses the normalized scales below.
- `weight_global_scale` is the original NVFP4 FP32 global scale, scalar or `[E]`;
  `None` means one. All scales apply before BF16 output.

`rhs_scales` is a separate FP32 outer multiplier, distinct from `rhs[1]`:

| `rhs_scale_block_size` | `rhs_scales`, dense / grouped | Computation |
| :--- | :--- | :--- |
| `0` (default) | Optional `[N]` / `[E,N]`, also with a trailing singleton dimension | Per-channel; retains native K16 scaling without FP8 weight rounding |
| `128` | Required `[ceil(N/128),ceil(K/128)]` / `[E,ceil(N/128),ceil(K/128)]` | Round decoded weights to E4M3, then apply 128x128 outer scales |

For blockwise FP8 normalization, divide the checkpoint K16 scales offline **before
preprocessing**; decoding does not divide by `rhs_scales`:

```text
native_scale = E4M3(checkpoint_k16_scale / fp8_block_scale)
decoded_fp8 = E4M3(E2M1 * native_scale)
restored_weight = decoded_fp8 * fp8_block_scale * global_scale
```

Pass `fp8_block_scale` as `rhs_scales` and keep the original global scale.
Unit block scales allow unchanged checkpoint scales. FP8 conversion uses nearest-even
rounding and finite saturation; normalization can still lose small K16 scales.
See `_absorb_fp8_scale` in [the tests](../../tests/test_w4a8_nvfp4.py) for an example.

## Execution

All layouts support both modes. `execution` selects weight decoding; the `_fused`
API suffix selects routing gather independently. The caller chooses the mode explicitly.

| `execution` | `PreprocessMode` | Behavior |
| :--- | :--- | :--- |
| `'fused'` (default) | `W4A8_NVFP4` | Decode inside GEMM; no full FP8 workspace |
| `'dequantize'` | `W4A8_NVFP4_FP8` | Decode all weights on each call, then run FP8 GEMM |

The prepared layouts are not interchangeable. `dequantize` requires explicit per-token
or K128 activation scales, `rhs_scale_block_size=128`, and caller-owned contiguous
E4M3FN `fp8_workspace` `[E,N,K]` (`[N,K]` for dense). Allocate its `E*N*K` bytes once,
separate from operands and metadata; each call refreshes it on the current stream.

## Nopad example

Prepare weights once at model load time. Here `native_scale` is already normalized
using `block_scale`; `m_indices` holds each row's expert and `m_rows` its per-expert counts.
Activations use K128 FP32 scales and weights use 128x128 FP32 outer scales.

```python
import deep_gemm as dg

rhs = dg.preprocess_weights_and_scales(
    dg.PreprocessMode.W4A8_NVFP4, False, packed_weight, native_scale)
dg.m_grouped_gemm_fp8_nvfp4_bf16_nt_nopad(
    (a, a_scale), rhs, out, m_indices, m_rows=m_rows,
    lhs_scale_group_size=128, rhs_scales=block_scale,
    rhs_scale_block_size=128, weight_global_scale=global_scale)
```

Dense/nopad/masked select configs internally unless overridden. Fused routing requires
configs and metadata from `moe_align_block_size`: pass prepared `rhs[0]` with matching
scale granularities for fused decoding, or the FP8 workspace with default granularity
options for `dequantize`. Use the returned config for GEMM; dequantize routing requires
N/K divisible by 128 and `block_k=128`. The optional `config` overrides tile selection.

See [W4A8 tests](../../tests/README.MD#w4a8-correctness-and-performance-tests)
for complete runnable cases, accuracy references and performance reproduction.
