# Unsupported GEMM combinations

These checks reject unsupported inputs on the host before GEMM execution.
Bug numbers refer to the DeepGEMM correctness review in ppu-kernel-quirks.

| Bug | Constraint | Reason |
| --- | --- | --- |
| 07 | Grouped BF16 GEMV requires the selected vector tile's K/stride alignment. | Unsupported GEMV inputs previously returned without computing output. |
| 08 | INT8/FP8 dense and BF16/INT8/FP8 fused MoE require even N. | The packed BF16 epilogue cannot safely store an odd last column. BF16 dense retains its existing odd-N support. |
| 10 | BF16 fused MoE requires K divisible by the selected block_k. | The A copy does not mask a partial K tile. |
