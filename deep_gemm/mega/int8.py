"""Native INT8 MegaMoE API, using the shared C++ JIT and split-copy execution."""

from __future__ import annotations

import types
import warnings
import weakref
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist

from .. import _C
from ..utils.math import align
from . import _normalize_activation


def _get_buffer_staging(base, buffer_type, group, shape):
    if base is None:
        return shape
    if (not isinstance(base, buffer_type) or base.buffer is None or base.handle is None
            or base.group is not group
            or base.buffer.device != torch.device('cuda', torch.cuda.current_device())):
        raise ValueError("base must be a live MegaMoE buffer with the same precision, device and group")
    return base.staging_config


def _get_or_allocate_symm_buffer(group, num_bytes, base):
    if base is not None:
        if num_bytes > base.buffer.nbytes:
            raise ValueError(f"MegaMoE view needs {num_bytes} bytes, but base has {base.buffer.nbytes}")
        return base.buffer, base.handle
    if group.size() == 1:
        buffer = torch.empty(num_bytes, dtype=torch.uint8, device='cuda')
        handle = types.SimpleNamespace(buffer_ptrs=[buffer.data_ptr()])
    else:
        import torch.distributed._symmetric_memory as symm_mem

        buffer = symm_mem.empty(num_bytes, dtype=torch.uint8, device='cuda')
        handle = symm_mem.rendezvous(buffer, group=group)
    buffer.zero_()
    group.barrier()
    torch.cuda.synchronize()
    return buffer, handle


class Int8SymmBuffer:
    """Persistent workspace for native INT8 dispatch, GEMM1, activation, GEMM2 and combine.

    Select the current CUDA device before constructing. Views do not independently
    own storage. Retain the buffer, weights, and output until execution/replays
    complete. x stores one INT8 row per source token. base inherits the
    allocation's staging capacities; every view must fit that allocation.
    Construct views before execution/capture and use aliases serially in the
    same order on every rank.
    """

    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_shared_experts: int = 0,
                 mma_type: str = 'int8xint8',
                 activation: str = 'swiglu',
                 base: Optional['Int8SymmBuffer'] = None):
        if (not isinstance(num_max_tokens_per_rank, int)
                or isinstance(num_max_tokens_per_rank, bool) or num_max_tokens_per_rank <= 0):
            raise ValueError("num_max_tokens_per_rank must be a positive integer")
        self.group = group
        self.num_ranks = group.size()
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type = mma_type
        self.activation = activation
        self.staging_config = _get_buffer_staging(
            base, Int8SymmBuffer, group, (num_experts, self.num_max_tokens_per_rank, hidden))
        size, slicer = _C.get_symm_buffer_size_for_int8_mega_moe(
            group.size(), num_experts, self.num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden, mma_type, activation, num_shared_experts, self.staging_config)
        self.buffer, self.handle = _get_or_allocate_symm_buffer(group, size, base)
        (self.x, self.x_sf, self.topk_idx, self.topk_weights,
         self.l1_acts, self.l1_acts_sf, self.l2_acts, self.l2_acts_sf) = slicer(self.buffer)
        self.x._mega_moe_buffer = weakref.ref(self)

    def destroy(self) -> None:
        """Release this alias after synchronizing users and retiring its Graphs."""
        if self.buffer is None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Int8SymmBuffer cannot be destroyed during graph capture")
        self.buffer = self.handle = self.group = None
        self.x = self.x_sf = self.topk_idx = self.topk_weights = None
        self.l1_acts = self.l1_acts_sf = self.l2_acts = self.l2_acts_sf = None


def get_symm_buffer_for_int8_mega_moe(
    group: dist.ProcessGroup,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_shared_experts: int = 0,
    use_fp8_dispatch: Union[bool, None] = None,
    mma_type: str = "int8xint8",
    activation: str = "swiglu",
) -> Int8SymmBuffer:
    """Create the SM80 INT8 workspace with the same argument order as the FP4 factory.

    Only mma_type='int8xint8' and num_shared_experts=0 are supported.
    activation supports 'swiglu', 'swiglu_step', 'situ' and 'swiglu_oai', as for FP4.
    Capacity is rounded to 16 tokens and topk supports [1, 32].
    The backend requires H >= 2048 divisible by 512 and I divisible by 256.
    use_fp8_dispatch is a deprecated compatibility option; only None/False are valid.
    """
    if use_fp8_dispatch is not None:
        if use_fp8_dispatch is not False:
            raise ValueError("INT8 MegaMoE does not support FP8 dispatch")
        warnings.warn(
            '`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`',
            DeprecationWarning, stacklevel=3)
    return Int8SymmBuffer(
        group, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, num_shared_experts,
        mma_type=mma_type, activation=activation)


def transform_weights_for_int8_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    activation: str = "swiglu",
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Prepare raw INT8 weight pairs once at load time, following the FP4 API.

    Inputs are contiguous INT8 [E,2*I,H] / [E,H,I] with FP32 channel scales
    [E,2*I] / [E,H]. Raw L1 rows are [gate...][up...]. L1 weights and scales
    are interleaved in groups of four on SM80; L2 is unchanged.
    Reuse the returned pairs for compute; do not transform them twice.
    """
    _normalize_activation(activation)
    for name, pair in (('l1_weights', l1_weights), ('l2_weights', l2_weights)):
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2 or
                not all(isinstance(t, torch.Tensor) for t in pair)):
            raise TypeError(f'{name} must be an (INT8 weight, FP32 channel scale) pair')
        weight, scale = pair
        if (weight.ndim != 3 or weight.dtype != torch.int8 or not weight.is_cuda or
                not weight.is_contiguous() or scale.dtype != torch.float32 or
                scale.device != weight.device or not scale.is_contiguous() or
                tuple(scale.shape) != tuple(weight.shape[:2])):
            raise ValueError(f'{name} has an invalid dtype, shape, layout or device')
    w1, w2 = l1_weights[0], l2_weights[0]
    if (w1.device != w2.device or w1.shape[0] != w2.shape[0] or
            w1.shape[1] != 2 * w2.shape[2] or w1.shape[2] != w2.shape[1]):
        raise ValueError('L1 and L2 weight shapes and devices must agree')
    if (min(w1.shape) <= 0 or w1.shape[2] < 2048 or w1.shape[2] % 512 or
            w2.shape[2] % 256):
        raise ValueError('INT8 MegaMoE requires positive dimensions, H >= 2048 divisible by 512 and I divisible by 256')
    major, minor = torch.cuda.get_device_capability(w1.device)
    if major != 8 or minor != 0:
        raise ValueError('INT8 MegaMoE only support sm80 now')
    l1_prepared = _C.preprocess_weights_and_scales(
        _C.PreprocessMode.INT8, True, *l1_weights)
    l2_prepared = _C.preprocess_weights_and_scales(
        _C.PreprocessMode.INT8, False, *l2_weights)
    return l1_prepared, l2_prepared


def int8_mega_moe_pre_dispatch(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    buf_x: torch.Tensor,
    buf_x_sf: torch.Tensor,
    buf_topk_idx: torch.Tensor,
    buf_topk_weights: torch.Tensor,
    num_tokens: int,
    group_size: int = 0,
    mma_type: str = "int8xint8",
) -> None:
    """Publish BF16 inputs with one INT8 scale per token (group_size=0).

    Pass the four original views of one live Int8SymmBuffer. num_tokens must
    equal x.shape[0]. Routing IDs can be int32 or int64 and must be distinct within each token;
    negative and out-of-range IDs are masked. Submit pre-dispatch and compute
    on the same stream, in the same order on every rank. The workspace stores
    normalized routing IDs as int64.
    """
    _C.int8_mega_moe_pre_dispatch(
        x, topk_idx, topk_weights, buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights,
        num_tokens, group_size, mma_type)


def int8_int8_mega_moe(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: Int8SymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 0),
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    *,
    l1_weights_replicas: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    l2_weights_replicas: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    situ_beta: Optional[float] = 4.0,
    situ_linear_beta: Optional[float] = 25.0,
    activation_alpha: Optional[float] = None,
) -> None:
    """Run the complete native INT8 MoE through the C++ multi-stage implementation.

    Weights are contiguous int8 [local_experts, 2*I, H] / [local_experts, H, I],
    with FP32 per-output-channel scales [local_experts, 2*I] / [local_experts, H].
    Prepare raw [gate...][up...] L1 rows once with
    transform_weights_for_int8_mega_moe; compute consumes its interleaved pairs.
    The fused GEMM1 epilogue rounds GEMM1, SiLU and the product to BF16, then
    truncates per-row INT8 quantization (Legacy masked semantics). GEMM2 fuses
    BF16 scatter and weighted combine. Supports swiglu, swiglu_step, situ and
    swiglu_oai using FP4's activation formulas. fast_math=True uses PPU
    sigmoid/tanh instructions; False uses expf/division/tanhf for GEMM1 activation
    without changing MMA precision, BF16 rounding or INT8 quantization.
    STEP/SiTU/OAI evaluate their formula in FP32 and round the product to BF16
    before row quantization. STEP clamps after SiLU; SwiGLU/OAI clamp the gate.
    SiTU defaults to situ_beta=4.0 and situ_linear_beta=25.0; explicitly pass
    situ_linear_beta=None to disable the up softcap. Both betas must otherwise
    be positive and finite, with no clamp/alpha. OAI's positive finite activation_alpha
    defaults to 1.702. Public signatures stay aligned with FP4; the source
    backend's quantization-mode option is not exposed. Split-copy overlaps the
    independent copy grid with GEMM1; whole-row quantization follows GEMM1.
    Warm up on each capture stream before Graph capture.
    recipe must be (1, 1, 0): one scale per token/output channel, with 0
    denoting the full K dimension of each GEMM (H for L1, I for L2).
    y is contiguous BF16 [C,H], T <= C <= token capacity; unused rows produce
    zeros. Retain inputs, weights, output and buffer until asynchronous execution
    and captured Graphs retire. Separate replica weight banks are not supported;
    both replica arguments must be None. The expected-M estimate is internal.
    cumulative_local_expert_recv_stats is an optional
    contiguous CUDA int32 vector of local-expert counts, incremented once per
    valid route on every call, including routes with zero weight. Initialize
    or reset the cumulative counters in the caller.
    """
    if not isinstance(sym_buffer, Int8SymmBuffer) or sym_buffer.buffer is None:
        raise ValueError("a live Int8SymmBuffer is required")
    activation = _normalize_activation(activation)
    if l1_weights_replicas is not None or l2_weights_replicas is not None:
        raise ValueError("INT8 MegaMoE does not support separate replica weight banks")
    # C++ infers dimensions from weights; reject a differently shaped buffer before entering it.
    groups = sym_buffer.num_experts // sym_buffer.group.size()
    for pair, shape in ((l1_weights, (groups, 2 * sym_buffer.intermediate_hidden, sym_buffer.hidden)),
                        (l2_weights, (groups, sym_buffer.hidden, sym_buffer.intermediate_hidden))):
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or not isinstance(pair[0], torch.Tensor) or tuple(pair[0].shape) != shape):
            raise ValueError("weight shape must match the INT8 MegaMoE buffer")
    if activation != 'situ':
        situ_beta = None if situ_beta == 4.0 else situ_beta
        situ_linear_beta = None if situ_linear_beta == 25.0 else situ_linear_beta
    _C.int8_int8_mega_moe(
        y, l1_weights, l2_weights, cumulative_local_expert_recv_stats, sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank, sym_buffer.num_experts, sym_buffer.num_topk,
        tuple(recipe), activation, activation_clamp, fast_math,
        situ_beta, situ_linear_beta, activation_alpha,
        l1_weights_replicas, l2_weights_replicas,
        sym_buffer.staging_config)
