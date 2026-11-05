"""Public MegaMoE API and symmetric-buffer management."""

from __future__ import annotations

import warnings
import types
from typing import Optional, Tuple, Union

import torch
# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C
from ..utils.math import align, ceil_div

__all__ = [
    "SymmBuffer",
    "get_symm_buffer_for_mega_moe",
    "transform_weights_for_mega_moe",
    "fp4_fp4_mega_moe",
    "mega_moe_pre_dispatch",
    "FP8SymmBuffer",
    "W4A16SymmBuffer",
    "get_symm_buffer_for_fp8_mega_moe",
    "get_symm_buffer_for_w4a16_mega_moe",
    "transform_weights_for_fp8_mega_moe",
    "transform_weights_for_w4a16_mega_moe",
    "fp8_fp8_mega_moe",
    "bf16_fp4_mega_moe",
]

def _normalize_activation(activation: str) -> str:
    if activation not in ('swiglu', 'swiglu_step', 'situ', 'swiglu_oai'):
        raise ValueError(f'Unsupported MegaMoE activation: {activation}')
    return activation


class SymmBuffer:
    """Symmetric workspace for PPU MXFP4 MegaMoE.

    Select the HGGC device before construction. base shares storage and
    staging capacities; every view must fit the allocation.
    Construct views before execution or graph capture. Use aliases serially
    across all ranks, with pre-dispatch then compute on the same stream.

    x is the FP4 dispatch view in [token, hidden/2] order. x_sf is the pushed-SFA
    staging view in [parity, source_rank, source_token, scale_block]
    order. It is an internal output of pre-dispatch, not a source-scale input
    that callers may fill directly.
    """

    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_shared_experts: int = 0,
                 mma_type: str = 'mxf4xmxf4',
                 activation: str = 'swiglu',
                 base: Optional['SymmBuffer'] = None):
        # Align token count
        num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

        # Init
        activation = _normalize_activation(activation)
        self.group = group
        self.num_ranks = group.size()
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type = mma_type
        self.activation = activation

        # Allocate or reuse a symmetric buffer
        self.staging_config = (num_experts, num_max_tokens_per_rank, hidden)
        size_args = (
            self.num_ranks, num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            mma_type, activation,
            num_shared_experts,
        )
        if base is not None:
            self.staging_config = base.staging_config
            size_args += (self.staging_config,)
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(*size_args)
        if base is None:
            allocator = torch if self.num_ranks == 1 else symm_mem
            self.buffer = allocator.empty(num_bytes, dtype=torch.uint8, device='cuda')
            self.handle = (
                types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
                if self.num_ranks == 1
                else symm_mem.rendezvous(self.buffer, group=group)
            )
            self.buffer.zero_()
            self.group.barrier()
            torch.cuda.synchronize()
        else:
            assert base.buffer is not None and base.handle is not None and base.group is group, \
                'Cannot reuse an invalid symmetric buffer'
            assert num_bytes <= base.buffer.nbytes, \
                (f'The reused Mega MoE config requires {num_bytes} bytes, '
                 f'but the symmetric buffer only has {base.buffer.nbytes} bytes')
            self.buffer = base.buffer
            self.handle = base.handle

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


def get_symm_buffer_for_mega_moe(
    group: dist.ProcessGroup,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_shared_experts: int = 0,
    use_fp8_dispatch: Union[bool, None] = None,
    mma_type: str = "mxf4xmxf4",
    activation: str = "swiglu",
) -> SymmBuffer:
    """Create the PPU workspace with native MegaMoE dimensions.

    The workspace is filled by mega_moe_pre_dispatch and consumed by
    fp4_fp4_mega_moe. MXFP4 uses the MMA type 'mxf4xmxf4'.
    """
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split('x')[0] == 'fp8')
        warnings.warn(
            f'`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`',
            DeprecationWarning, stacklevel=3
        )
    return SymmBuffer(
        group, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, num_shared_experts,
        mma_type=mma_type, activation=activation)


def transform_weights_for_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    activation: str = "swiglu",
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Prepare raw MXFP4 weight pairs for the PPU MegaMoE kernels once at load time.

    Inputs are contiguous uint8 packed weights and raw UE8M0 scales, before any
    interleaving or scale preprocessing. L1 gate/up rows are interleaved one by
    one; L2 weights are unchanged. Both scales become packed uint16 N-major.
    Use the returned pairs; do not transform already-interleaved weights again.
    Transform independently allocated master and replica banks separately.
    """
    _normalize_activation(activation)

    def validate_pair(name, pair):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError(f'{name} must be a (MXFP4 weight, raw UE8M0 scale) pair')
        weight, sf = pair
        if not isinstance(weight, torch.Tensor) or not isinstance(sf, torch.Tensor):
            raise TypeError(f'{name} weight and scale must both be tensors')
        if weight.ndim != 3 or sf.ndim != 3:
            raise ValueError(f'{name} weight and scale must both be rank-3 tensors')
        if weight.dtype != torch.uint8 or sf.dtype != torch.uint8:
            raise TypeError(f'{name} requires uint8 packed weights and raw uint8 scales')
        if weight.device != sf.device:
            raise ValueError(f'{name} weight and scale must be on the same device')
        if not weight.is_contiguous() or not sf.is_contiguous():
            raise ValueError(f'{name} weight and scale must use contiguous raw layouts')
        experts, n, packed_k = weight.shape
        if min(experts, n, packed_k) <= 0:
            raise ValueError(f'{name} dimensions must be positive')
        expected_sf_shape = (experts, n, ceil_div(packed_k, 16))
        if tuple(sf.shape) != expected_sf_shape:
            raise ValueError(f'{name} scale shape must be {expected_sf_shape}, got {tuple(sf.shape)}')
        return weight, sf

    l1_fp4, l1_sf = validate_pair('l1_weights', l1_weights)
    l2_fp4, l2_sf = validate_pair('l2_weights', l2_weights)
    hidden, intermediate = l1_fp4.shape[2] * 2, l2_fp4.shape[2] * 2
    if (l1_fp4.shape[0] != l2_fp4.shape[0] or
            l1_fp4.shape[1] != 2 * intermediate or l2_fp4.shape[1] != hidden):
        raise ValueError('MXFP4 MegaMoE weights must have shapes (E, 2*I, H/2) and (E, H, I/2)')
    if hidden < 1024 or hidden % 512 or intermediate % 64:
        raise ValueError('MXFP4 MegaMoE requires H >= 1024, H divisible by 512 and I divisible by 64')
    if l1_fp4.device != l2_fp4.device:
        raise ValueError('L1 and L2 MXFP4 MegaMoE weights must be on the same device')
    l1_transformed = _C.preprocess_mxfp4_weight_for_act_and_quant_fusing(l1_fp4, l1_sf)
    l2_transformed = (l2_fp4, _C.preprocess_mxfp4_scales(l2_sf))
    return l1_transformed, l2_transformed


def mega_moe_pre_dispatch(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    sym_buffer: SymmBuffer,
    num_tokens: Optional[int] = None,
) -> None:
    """Prepare BF16 inputs and routes for the selected precision workspace.

    W4A4 expert IDs can be int32 or int64; the other precisions require int64.
    The workspace stores IDs as int64.
    Valid physical expert IDs must be distinct within each token.
    SFA is produced and pushed directly into internal owner staging; x_sf is not
    a public source-scale input, and callers must use this helper to prepare it.
    """
    if isinstance(sym_buffer, _RoutedSymmBuffer):
        sym_buffer._check_live()
        _C.routed_mega_moe_pre_dispatch(
            hidden_states, topk_idx, topk_weights,
            sym_buffer.x, sym_buffer.x_sf, sym_buffer.topk_idx, sym_buffer.topk_weights, num_tokens)
        return
    if not isinstance(sym_buffer, SymmBuffer) or sym_buffer.buffer is None:
        raise ValueError('a live MegaMoE SymmBuffer is required')
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError('hidden_states must be a 2D tensor')
    if num_tokens is None:
        num_tokens = hidden_states.shape[0]
    if (not isinstance(num_tokens, int) or isinstance(num_tokens, bool)
            or not 0 <= num_tokens <= hidden_states.shape[0]):
        raise ValueError('num_tokens is outside the input tensor')
    if not all(isinstance(t, torch.Tensor) and t.ndim == 2 for t in (topk_idx, topk_weights)):
        raise ValueError('topk_idx and topk_weights must be 2D tensors')
    if hidden_states.shape[1] != sym_buffer.hidden:
        raise ValueError(f'hidden_states must be contiguous BF16 with shape [tokens, {sym_buffer.hidden}]')
    if topk_idx.shape[1] != sym_buffer.num_topk:
        raise ValueError(f'topk_idx must be contiguous int32 or int64 with shape ({num_tokens}, {sym_buffer.num_topk})')
    _C.mega_moe_pre_dispatch(
        hidden_states[:num_tokens], topk_idx[:num_tokens], topk_weights[:num_tokens],
        sym_buffer.buffer, sym_buffer.handle.buffer_ptrs, sym_buffer.num_ranks, sym_buffer.num_experts,
        sym_buffer.num_max_tokens_per_rank, sym_buffer.group.rank(),
        sym_buffer.intermediate_hidden, sym_buffer.mma_type, sym_buffer.staging_config)


def fp4_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: SymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 32),
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
    """Consume published buffers through the complete C++ MegaMoE entry.

    y must be contiguous BF16 [num_tokens, hidden] on the buffer device.
    Unused output rows have invalid routes and produce zeros.
    activation supports 'swiglu', 'swiglu_step', 'situ' and 'swiglu_oai'.
    SwiGLU clamps gate before SiLU; SwiGLU-Step clamps SiLU(gate). Both clamp up
    symmetrically. None or +inf disables clamping; zero is a real threshold.
    SiTU requires positive finite situ_beta and no clamp; beta defaults to 4
    and linear_beta defaults to 25, matching main.
    situ_linear_beta=None disables up softcap; a provided value must be positive and finite.
    OAI uses gate * sigmoid(alpha * gate) * (up + 1), after gate/up clamping.
    activation_alpha is OAI-only, defaults to 1.702, and must be positive finite.
    All activations consume FP32 accumulators before MXFP4 quantization.
    fast_math selects approximate GEMM1 activation math; False uses
    expf/division/tanhf without changing MMA precision or MXFP4 formats.
    cumulative_local_expert_recv_stats is an optional HGGC int32 vector with
    one entry per local physical expert. Each call adds valid routed assignments
    from every rank, including zero-weight assignments. The caller initializes
    these cumulative counters; the library never resets them.
    UltraEP may pass separate L1/L2 replica weight pairs together. Local physical
    expert IDs index masters first, then replicas; num_experts includes both.
    Without replica pairs, l1_weights and l2_weights contain all physical experts.
    Transform each bank once; forward reads both allocations without concatenation.
    Callers must retain all weight banks and the optional statistics tensor until HGGC
    execution completes, including every replay of a captured Graph.
    """
    if not isinstance(sym_buffer, SymmBuffer):
        raise ValueError('Main MegaMoE requires a SymmBuffer')
    activation = _normalize_activation(activation)
    if not isinstance(y, torch.Tensor) or y.ndim != 2 or y.shape[1] != sym_buffer.hidden:
        raise ValueError(f'output must have shape [tokens, workspace hidden={sym_buffer.hidden}]')
    if (not isinstance(l1_weights, (tuple, list)) or len(l1_weights) != 2
            or not isinstance(l1_weights[0], torch.Tensor) or l1_weights[0].ndim != 3
            or l1_weights[0].shape[1] != 2 * sym_buffer.intermediate_hidden):
        raise ValueError(f'l1_weights must match workspace intermediate_hidden={sym_buffer.intermediate_hidden}')
    # Match main's SiTU defaults while retaining None as the up-softcap extension.
    if activation != 'situ':
        situ_beta = None if situ_beta == 4.0 else situ_beta
        situ_linear_beta = None if situ_linear_beta == 25.0 else situ_linear_beta
    _C.fp4_fp4_mega_moe(
        y, l1_weights, l2_weights, cumulative_local_expert_recv_stats, sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank, sym_buffer.num_experts, sym_buffer.num_topk,
        tuple(recipe), activation, activation_clamp, fast_math,
        situ_beta, situ_linear_beta, activation_alpha,
        l1_weights_replicas, l2_weights_replicas,
        sym_buffer.staging_config)


def _interleave_weights(tensor: torch.Tensor, gran: int = 8) -> torch.Tensor:
    experts, n, k = tensor.shape
    return (tensor.reshape(experts, 2, n // (2 * gran), gran, k)
            .transpose(1, 2).contiguous().reshape(experts, n, k))


class _RoutedSymmBuffer:
    """Library-owned storage shared by the FP8 and BF16 dispatch adapters."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        precision: str,
        activation: str,
        base: Optional["_RoutedSymmBuffer"] = None,
    ) -> None:
        if torch.cuda.get_device_capability() != (8, 9):
            raise RuntimeError("MegaMoE requires PPU1.5")
        allowed = ("swiglu", "swiglu_oai")
        if precision == "w4a16":
            allowed = ("situ",)
        if activation not in allowed:
            raise ValueError(f"unsupported activation for {precision}: {activation}")
        size_args = (group.size(), num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden)
        if precision == "w4a16":
            total, slices = _C.get_symm_buffer_size_for_w4a16_mega_moe(*size_args)
        else:
            total, slices = _C.get_symm_buffer_size_for_fp8_mega_moe(
                *size_args, precision.removeprefix("fp8_")
            )
        self.group, self.num_ranks = group, group.size()
        self.num_experts, self.num_topk = num_experts, num_topk
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden, self.intermediate_hidden = hidden, intermediate_hidden
        self.precision, self.activation = precision, activation
        self._layout_key = (
            self.num_ranks,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            precision,
        )
        if base is not None:
            if (
                type(base) is not type(self)
                or base.buffer is None
                or base.group is not group
                or self._layout_key != base._layout_key
                or base.buffer.device.index != torch.cuda.current_device()
                or base.buffer.nbytes < total
            ):
                raise ValueError("base must be a live compatible MegaMoE workspace on the same group and device")
            self.buffer, self.handle = base.buffer, base.handle
        else:
            if self.num_ranks == 1:
                self.buffer = torch.empty(total, dtype=torch.uint8, device="cuda")
                self.handle = types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            else:
                import torch.distributed._symmetric_memory as symm_mem

                self.buffer = symm_mem.empty(total, dtype=torch.uint8, device="cuda")
                self.handle = symm_mem.rendezvous(self.buffer, group=group)
            self.buffer.zero_()
            group.barrier()
            torch.cuda.synchronize()
        self.x, self.x_sf, self.topk_idx, self.topk_weights = slices(
            self.buffer, self.handle.buffer_ptrs, group.rank()
        )

    def _check_live(self) -> None:
        if self.buffer is None or self.handle is None or self.group is None:
            raise ValueError("MegaMoE workspace has been destroyed")

    def destroy(self) -> None:
        self.buffer = self.handle = self.group = None
        self.x = self.x_sf = self.topk_idx = self.topk_weights = None


class FP8SymmBuffer(_RoutedSymmBuffer):
    """FP8 workspace; ``intermediate_hidden`` is the activated width I.

    ``quantization='channelwise'`` uses one activation scale per row and one
    weight scale per output channel. ``'blockwise'`` uses activation groups of
    128 and weight blocks of 128x128. The buffer owns every intermediate and
    communication counter. Fill x/x_sf/topk_idx/topk_weights or call
    mega_moe_pre_dispatch, then execute fp8_fp8_mega_moe on every rank.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        quantization: str = "channelwise",
        activation: str = "swiglu",
        base: Optional["FP8SymmBuffer"] = None,
    ) -> None:
        if quantization not in ("channelwise", "blockwise"):
            raise ValueError("FP8 quantization must be 'channelwise' or 'blockwise'")
        self.quantization = quantization
        super().__init__(
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            "fp8_" + quantization,
            activation,
            base,
        )


class W4A16SymmBuffer(_RoutedSymmBuffer):
    """BF16 dispatch, group-32 MXFP4 weights, fused SiTU and BF16 combine.

    x is BF16 [capacity,H]. x_sf is an empty [capacity,0] view because input
    activations are not quantized. All ranks participate, including empty ranks.
    Workspace aliases execute sequentially on one stream, as for SymmBuffer.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        activation: str = "situ",
        base: Optional["W4A16SymmBuffer"] = None,
    ) -> None:
        super().__init__(
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            "w4a16",
            activation,
            base,
        )


def get_symm_buffer_for_fp8_mega_moe(
    group: dist.ProcessGroup,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    quantization: str = "channelwise",
    activation: str = "swiglu",
) -> FP8SymmBuffer:
    return FP8SymmBuffer(
        group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, quantization, activation
    )


def get_symm_buffer_for_w4a16_mega_moe(
    group: dist.ProcessGroup,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    activation: str = "situ",
) -> W4A16SymmBuffer:
    return W4A16SymmBuffer(
        group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, activation
    )


def _check_routed_weights(l1_weights, l2_weights, dtype, blockwise=False):
    for name, pair in (("l1_weights", l1_weights), ("l2_weights", l2_weights)):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError(f"{name} must be a (weight, scale) pair")
        if any(not isinstance(t, torch.Tensor) for t in pair):
            raise TypeError(f"{name} weights and scales must be tensors")
        if any(t.ndim != 3 for t in pair):
            raise ValueError(f"{name} must contain rank-3 weights and scales")
        w, s = pair
        if (
            w.dtype != dtype
            or s.dtype != (torch.uint8 if dtype == torch.uint8 else torch.float32)
            or not w.is_cuda
            or not s.is_cuda
            or not w.is_contiguous()
            or not s.is_contiguous()
            or w.device != s.device
        ):
            raise ValueError(f"invalid {name} dtype, device or layout")
        e, n, packed_k = w.shape
        k = packed_k * (2 if dtype == torch.uint8 else 1)
        expected = (e, n // 128, k // 128) if blockwise else (e, n, k // 32 if dtype == torch.uint8 else 1)
        if s.shape != expected:
            raise ValueError(f"invalid {name} scale shape: expected {expected}")
    e, n, packed_h = l1_weights[0].shape
    i, h = n // 2, l2_weights[0].shape[1]
    if (
        e <= 0
        or h <= 0
        or h % 256
        or i <= 0
        or i % 128
        or n != 2 * i
        or packed_h != h // (2 if dtype == torch.uint8 else 1)
        or l2_weights[0].shape != (e, h, i // (2 if dtype == torch.uint8 else 1))
        or l1_weights[0].device != l2_weights[0].device
    ):
        raise ValueError("weights must represent matching [E,2I,H] and [E,H,I] expert matrices")
    return e, h, i


def transform_weights_for_fp8_mega_moe(
        l1_weights: Tuple[torch.Tensor, torch.Tensor], l2_weights: Tuple[torch.Tensor, torch.Tensor],
        quantization: str = "channelwise",
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Reorder gate/up rows without requantizing E4M3 weights or FP32 scales."""
    if quantization not in ("channelwise", "blockwise"):
        raise ValueError("FP8 quantization must be 'channelwise' or 'blockwise'")
    blockwise = quantization == "blockwise"
    e, h, i = _check_routed_weights(l1_weights, l2_weights, torch.float8_e4m3fn, blockwise)
    w, s = l1_weights
    if blockwise:
        w = _interleave_weights(w.view(torch.uint8), 128).view(w.dtype)
        s = s.reshape(e, 2, i // 128, h // 128).transpose(1, 2).contiguous().reshape_as(s)
    else:
        w = _interleave_weights(w.view(torch.uint8)).view(w.dtype)
        s = _interleave_weights(s)
    return (w, s), l2_weights


def transform_weights_for_w4a16_mega_moe(
        l1_weights: Tuple[torch.Tensor, torch.Tensor], l2_weights: Tuple[torch.Tensor, torch.Tensor],
        activation: str = "situ",
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Pack canonical MXFP4 weights and UE8M0 scales for BF16 x FP4 SiTU."""
    if activation != "situ":
        raise ValueError("W4A16 MegaMoE supports activation='situ' only")
    _check_routed_weights(l1_weights, l2_weights, torch.uint8)

    def pack_scales(scales):
        e, n, groups = scales.shape
        return (
            scales.reshape(e, n // 64, 8, 8, groups // 2, 2)
            .permute(0, 1, 4, 3, 2, 5)
            .contiguous()
            .reshape(e, n // 64, groups // 2 * 128)
        )

    w1, s1 = l1_weights
    w2, s2 = l2_weights
    return (_interleave_weights(w1), pack_scales(_interleave_weights(s1))), (w2, pack_scales(s2))


def _check_routed_buffer(
    y: torch.Tensor,
    l1_weights: Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, float]],
    sym_buffer: _RoutedSymmBuffer,
    activation: str,
    weight_components: int = 2,
) -> None:
    sym_buffer._check_live()
    if activation != sym_buffer.activation:
        raise ValueError("activation must match the MegaMoE workspace")
    if (
        not isinstance(y, torch.Tensor)
        or y.ndim != 2
        or y.shape[1] != sym_buffer.hidden
    ):
        raise ValueError("output must have shape [tokens, workspace hidden]")
    if (not isinstance(l1_weights, (tuple, list)) or len(l1_weights) != weight_components
            or not isinstance(l1_weights[0], torch.Tensor)
            or l1_weights[0].ndim != 3 or l1_weights[0].shape[1] != 2 * sym_buffer.intermediate_hidden):
        raise ValueError("l1_weights must match the workspace intermediate_hidden")


def fp8_fp8_mega_moe(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: FP8SymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    *,
    l1_weights_replicas: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    l2_weights_replicas: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    swiglu_alpha: float = 1.702,
) -> None:
    """Complete FP8 MoE; scales follow sym_buffer.quantization, output is BF16.

    Original routing weights are applied exactly once in combine. The C++
    implementation runs dispatch, GEMM1/activation/quantization, GEMM2/scatter,
    and completion. Master and replica tensors may be separate allocations.
    """
    if not isinstance(sym_buffer, FP8SymmBuffer):
        raise TypeError("fp8_fp8_mega_moe requires an FP8SymmBuffer")
    _check_routed_buffer(y, l1_weights, sym_buffer, activation)
    _C.fp8_fp8_mega_moe(
        y, l1_weights, l2_weights, cumulative_local_expert_recv_stats,
        sym_buffer.buffer, sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank, sym_buffer.num_experts, sym_buffer.num_topk,
        sym_buffer.quantization, activation, activation_clamp, fast_math,
        l1_weights_replicas, l2_weights_replicas, swiglu_alpha,
    )


def bf16_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: W4A16SymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    activation: str = "situ",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    *,
    l1_weights_replicas: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    l2_weights_replicas: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
) -> None:
    """Complete BF16 x MXFP4 SiTU MoE with BF16 output in original token order."""
    if not isinstance(sym_buffer, W4A16SymmBuffer):
        raise TypeError("bf16_fp4_mega_moe requires a W4A16SymmBuffer")
    _check_routed_buffer(y, l1_weights, sym_buffer, activation)
    _C.bf16_fp4_mega_moe(
        y, l1_weights, l2_weights, cumulative_local_expert_recv_stats,
        sym_buffer.buffer, sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank, sym_buffer.num_experts, sym_buffer.num_topk,
        activation, activation_clamp, fast_math, l1_weights_replicas, l2_weights_replicas,
        situ_beta, situ_linear_beta,
    )
