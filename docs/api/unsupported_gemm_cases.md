# Unsupported GEMM combinations

These checks reject unsupported inputs on the host before GEMM execution.
Bug numbers refer to the DeepGEMM correctness review in ppu-kernel-quirks.

| Bug | Constraint | Reason |
| --- | --- | --- |
| 04 | W4A16_MXFP4 preprocessing accepts E8M0 codes in [1,254]. Prepared weights must come from this validated preprocessing path. | Its BF16 dequantizer does not implement special codes 0/255. Values are checked once during preprocessing, not scanned by GEMM. |
| 07 | Grouped BF16 GEMV requires the selected vector tile's K/stride alignment. | Unsupported GEMV inputs previously returned without computing output. |
| 08 | INT8/FP8 dense and BF16/INT8/FP8 fused MoE require even N. | The packed BF16 epilogue cannot safely store an odd last column. BF16 dense retains its existing odd-N support. |
| 10 | BF16 fused MoE requires K divisible by the selected block_k. | The A copy does not mask a partial K tile. |
| 11 | A8W8 nopad requires contiguous A scales; blockwise FP8 einsum requires contiguous B scales. | These kernels use implicit contiguous scale layouts. |
| 12 | W4FA16 MMA with N expansion requires two pipeline stages. | Expanded short-K pipelines with other stage counts are unsupported. |
| 14 | moe_align_block_size requires contiguous topk_ids. | Both routing kernels index the route tensor as a flat contiguous array. |
| 15 | INT8/per-channel FP8 fused MoE requires K divisible by block_k. | A partial K tile can race with scale staging. |
| 16 | W4FA16 MMA N expansion requires N divisible by block_n*n_expand. | Expanded full tiles outside N do not have column predicates. |
| 17 | ensure_zero_padding=True clears negative-ID rows after BF16/INT8/FP8 contiguous GEMM, including BF16 C accumulation, using device operations without host synchronization. Compact BF16 expert ends contain no padding IDs and skip the mask. Compile-only mode preserves output. | Callers preserving or discarding padding can set False to avoid the additional mask operations. |
| 18 | INT8/FP8 grouped contiguous requires use_psum_layout=False and per-row expert IDs. BF16 supports compact exclusive expert end offsets. | The quantized scheduler does not interpret prefix-sum endpoints. |
| 19 | FP4 masked requires enable_sbo_overlap=False. | Completion signaling is not implemented in either its standard or dynamic path. |
| 20 | Blockwise FP8 masked rejects SBO when its selected short-K kernel expands N fourfold. | Its completion count does not match the advertised threshold. |
| 21 | Sparse MQA metadata now allocates workspace in each capture's graph pool; eager calls retain their per-stream cache. | Independent graph pools can replay concurrently without sharing metadata counters. Intentionally shared pools retain PyTorch's serial-replay requirement. |
| 22 | Dynamic A8W8 masked requires N divisible by 256 when K>2048. | Its expanded scheduling drops a partial N tile. Disable dynamic tiling to use the static path for such shapes. |
| 23 | MXFP4 scales must have an even byte storage offset when reinterpreted as uint16. | Clone an odd-offset contiguous view. Padding an odd scale width already produces aligned storage. |
| 24 | Dense/nopad/masked FP4 bias must be contiguous. | The epilogue uses a fixed unit column stride. |
| 26 | Blockwise FP8 fused MoE requires at least two warps per CTA. | Warp 1 is the B-scale producer. |
| 27 | BF16/INT8/FP8/MXFP4 fused MoE requires a 16-byte-aligned A base address. | Logical contiguity alone does not guarantee aligned vector loads. |
| 28 | INT8/per-channel FP8/blockwise FP8 einsum requires BF16 output. | Its launcher and epilogue do not implement FP32 output. |
