# PPU MegaGate API

`deep_gemm.bf16_mega_gate` fuses a BF16 gate GEMM, scoring and MoE top-k
routing in one kernel on PPU1.0 (810E) and PPU1.5 (M890).
It takes token features `x [T,H]` and router weights `weight [E,H]`,
accumulates `x @ weight.T` in FP32, and returns expert IDs and routing weights.

## Python signature

The parameter order and defaults match the
[pybind API](../../csrc/apis/mega_gate.hpp).

```python
def bf16_mega_gate(
    x, weight, num_topk,
    use_shared_as_routed, num_shared_experts,
    routed_scaling_factor, ep_rank,
    scoring_func="sqrtsoftplus",
    mask=None,
    bias=None,
    image_bias=None,
    image_token_mask=None,
    fix_routing_mask=None,
    to_physical_map=None,
    logical_count=None,
    unmapped_topk_idx=None,
    force_random=None,
    out=None,
) -> tuple[torch.Tensor, torch.Tensor]: ...
```

The first seven arguments are required. Define `K = num_topk` and
`S = num_shared_experts` when `use_shared_as_routed=True`, otherwise `S = 0`.

| Argument | Shape / type | Meaning |
| --- | --- | --- |
| `x` | BF16 `[T,H]` | Token features |
| `weight` | BF16 `[E,H]` | Weights for all routed experts |
| `num_topk` | int | Number of routed experts selected per token |
| `use_shared_as_routed` | bool | Append shared experts to each token's output |
| `num_shared_experts` | int | Number of shared experts when enabled |
| `routed_scaling_factor` | finite float | Multiply normalized routed weights by this value |
| `ep_rank` | nonnegative int | Rank-dependent selection among physical expert replicas |
| `scoring_func` | str | `"sqrtsoftplus"` (default) or `"sigmoid"` |
| `mask` | bool `[T]` | False entries disable the corresponding token |
| `bias` | FP32 `[E]` | Expert bias used for ranking |
| `image_bias` | FP32 `[E]` | Ranking bias for image tokens |
| `image_token_mask` | bool `[T]` | Select `image_bias` for True entries |
| `fix_routing_mask` | bool `[T]` | Reuse logical expert IDs from `unmapped_topk_idx` |
| `to_physical_map` | INT32 `[E+S,D]` | Physical expert IDs, with capacity for D replicas per logical expert |
| `logical_count` | INT32 `[E+S]` | Valid replica count per logical expert |
| `unmapped_topk_idx` | INT64 `[T,K]` | Optional logical-ID output; also input for fixed routing |
| `force_random` | bool `[T]` | Generate random physical routing for True entries |
| `out` | tuple of tensors | Reuse `(INT64 [T,K+S], FP32 [T,K+S])` output buffers |

## Scoring and outputs

For each token, normal routing computes:

```text
logits = x @ weight.T
scores = sqrt(log(1 + exp(logits)))   # sqrtsoftplus
scores = sigmoid(logits)             # sigmoid
ranking = scores + selected_bias
logical_ids = topk(ranking, K)
selected_scores = gather(scores, logical_ids)
```

`selected_bias` is `image_bias` for image tokens, otherwise `bias`, or zero
when the corresponding bias is absent. Bias affects expert selection.
Routing weights use the selected unbiased scores.

For `sqrtsoftplus`, routed weights are
`selected_scores / (sum(selected_scores) + 1e-20) * routed_scaling_factor`.
For `sigmoid`, the denominator is the selected-score sum when positive,
otherwise 1, so an all-zero selection produces zero weights.
Normal ranking breaks equal scores plus bias by the smaller logical expert ID.

The return value is `(topk_idx, topk_weights)`, with INT64 and FP32 tensors
of shape `[T,K+S]`. Without a physical map, IDs are logical expert IDs.
With a physical map, IDs come from `to_physical_map`; optional
`unmapped_topk_idx` receives the selected logical IDs before mapping.

Shared experts occupy the final S slots, use logical IDs `E .. E+S-1`
before mapping, and have weight 1.0. Normal routed weights use the first K slots.

Masked tokens produce IDs `-1` and weights `0`, including shared slots;
their `unmapped_topk_idx` entries are also `-1`.
For active tokens, `force_random` takes precedence over `fix_routing_mask`.
Random routing emits distinct physical IDs and random weights in `(0,1]`;
logical-ID output is `-1`.
Fixed routing reads the first K logical IDs from `unmapped_topk_idx`,
preserves their slot order, and computes their weights using the selected
scoring function before applying physical mapping.

## Layout and value requirements

- All tensors reside on the same PPU device as `x`. Inputs, metadata and
  `out` tensors are contiguous. `unmapped_topk_idx` also accepts padded
  rows with unit column stride.
- `x` and `weight` data pointers are 16-byte aligned.
- `0 <= T <= 2**20`; `H > 0` and `H % 256 == 0`.
- `0 < E <= 512` and `E % 4 == 0`; `1 <= K <= E` and `K+S <= 32`.
- When shared routing is enabled, `S` is 1 or 2, `K % S == 0`, and
  `E % (K // S) == 0`.
- Supply `image_bias` and `image_token_mask` together.
- Supply `to_physical_map` and `logical_count` together. Each count is in
  `[1,D]`; the first `logical_count[e]` map entries hold valid physical IDs
  for expert e. Shared expert entries are included when S is positive.
- Fixed routing requires a populated `unmapped_topk_idx`. For active fixed
  tokens, provide K distinct routed logical IDs in `[0,E)`.
- A zero-token call returns empty outputs of shape `[0,K+S]`.

## Example

```python
import torch
import deep_gemm

T, H, E, K = 128, 5120, 384, 6
x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(E, H, device="cuda", dtype=torch.bfloat16) * H**-0.5
bias = torch.zeros(E, device="cuda", dtype=torch.float32)

topk_ids, topk_weights = deep_gemm.bf16_mega_gate(
    x, weight, K,
    use_shared_as_routed=False,
    num_shared_experts=1,
    routed_scaling_factor=1.5,
    ep_rank=0,
    bias=bias,
)
assert topk_ids.shape == topk_weights.shape == (T, K)
assert topk_ids.dtype == torch.int64
assert topk_weights.dtype == torch.float32

# Select sigmoid scoring with the same inputs and output layout.
sigmoid_ids, sigmoid_weights = deep_gemm.bf16_mega_gate(
    x, weight, K,
    use_shared_as_routed=False,
    num_shared_experts=1,
    routed_scaling_factor=1.5,
    ep_rank=0,
    scoring_func="sigmoid",
    bias=bias,
)
```

For the DSpark draft model, use `E=128`, `H=5120`, and `K=3`.
See [MegaGate tests](../../tests/test_mega_gate.py) for routing controls and
[the check-in lists](../../tests/checkin/) for representative correctness cases.
