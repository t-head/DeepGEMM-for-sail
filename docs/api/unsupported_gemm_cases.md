# Unsupported GEMM combinations

These checks reject unsupported inputs on the host before GEMM execution.
Bug numbers refer to the DeepGEMM correctness review in ppu-kernel-quirks.

| Bug | Constraint | Reason |
| --- | --- | --- |
| 07 | Grouped BF16 GEMV requires the selected vector tile's K/stride alignment. | Unsupported GEMV inputs previously returned without computing output. |
