# Unsupported GEMM combinations

These checks reject unsupported inputs on the host before GEMM execution.
Bug numbers refer to the DeepGEMM correctness review in ppu-kernel-quirks.

| Bug | Constraint | Reason |
| --- | --- | --- |
| 07 | Grouped BF16 GEMV requires the selected vector tile's K/stride alignment. | Unsupported GEMV inputs previously returned without computing output. |
| 08 | INT8/FP8 dense and BF16/INT8/FP8 fused MoE require even N. | The packed BF16 epilogue cannot safely store an odd last column. BF16 dense retains its existing odd-N support. |
| 10 | BF16 fused MoE requires K divisible by the selected block_k. | The A copy does not mask a partial K tile. |
| 11 | A8W8 nopad requires contiguous A scales; blockwise FP8 einsum requires contiguous B scales. | These kernels use implicit contiguous scale layouts. |
| 12 | W4FA16 MMA with N expansion requires two pipeline stages. | Expanded short-K pipelines with other stage counts are unsupported. |
| 14 | moe_align_block_size requires contiguous topk_ids. | Both routing kernels index the route tensor as a flat contiguous array. |
