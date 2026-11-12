"""Public MegaMoE API and symmetric-buffer management."""

from __future__ import annotations

import warnings
import types
import weakref
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
    "SymmBuffer", "get_symm_buffer_for_mega_moe", "transform_weights_for_mega_moe",
    "fp4_fp4_mega_moe", "mega_moe_pre_dispatch",
    "transform_weights_for_fp8_mega_moe", "transform_weights_for_w4a16_mega_moe",
    "fp8_fp8_mega_moe", "bf16_fp4_mega_moe",
    "transform_weights_for_w4a8_mega_moe", "fp8_fp4_mega_moe",
]

def _normalize_activation(activation: str) -> str:
    if activation not in ('swiglu', 'swiglu_step', 'situ', 'swiglu_oai'):
        raise ValueError(f'Unsupported MegaMoE activation: {activation}')
    return activation


def _normalize_mma_type(mma_type: str) -> str:
    return 'fp4xfp4' if mma_type == 'mxfp4xmxfp4' else mma_type


class SymmBuffer:
    """Symmetric workspace for the supported PPU MegaMoE precisions.

    mma_type='fp4xfp4' (also 'mxfp4xmxfp4') selects W4A4; 'fp8xfp4' selects W4A8.
    'bf16xfp4' selects W4A16; 'fp8xfp8' selects W8A8, with
    quantization='channelwise' (default) or 'blockwise'. intermediate_hidden is I.

    Select the HGGC device before construction. base shares storage and
    staging capacities; every view must fit the allocation.
    Construct views before execution or graph capture. Use aliases serially
    across all ranks, with pre-dispatch then compute on the same stream.
    Each compute call selects its activation within the precision's supported
    set; the allocation can be reused across those activations.

    x is the FP4 dispatch view in [token, hidden/2] order. x_sf is the pushed-SFA
    staging view in [parity, source_rank, source_token, scale_block]
    order. It is an internal output of pre-dispatch, not a source-scale input
    that callers may fill directly. W4A8 instead exposes E4M3 x [token, hidden]
    and FP32 x_sf [token, 1], as does FP8 channelwise. FP8 blockwise has one
    input scale per 128 elements. W4A16 exposes BF16 x and an empty x_sf view.
    """

    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_shared_experts: int = 0,
                 mma_type: str = 'fp4xfp4',
                 activation: str = 'swiglu',
                 base: Optional['SymmBuffer'] = None,
                 *, quantization: Optional[str] = None):
        mma_type = _normalize_mma_type(mma_type)
        if mma_type == 'fp8xfp8':
            quantization = 'channelwise' if quantization is None else quantization
            if quantization not in ('channelwise', 'blockwise'):
                raise ValueError("FP8 quantization must be 'channelwise' or 'blockwise'")
        elif quantization is not None:
            raise ValueError('quantization applies only to fp8xfp8 workspaces')
        # Keep the original positional constructor and W4A4 staging contract.
        num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())
        if base is not None:
            if (not isinstance(base, SymmBuffer) or base.mma_type != mma_type
                    or base.quantization != quantization):
                raise ValueError('base must use the same MegaMoE mma_type and quantization')
            if mma_type != 'fp4xfp4':
                base._check_live()
                if (base.group is not group or base.buffer.device.index != torch.cuda.current_device()
                        or (num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden) !=
                           (base.num_experts, base.num_max_tokens_per_rank, base.num_topk,
                            base.hidden, base.intermediate_hidden)):
                    raise ValueError('base must be a compatible MegaMoE workspace on the same group and device')

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
        self.quantization = quantization
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
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            *size_args, quantization=quantization)
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
        if mma_type == 'fp4xfp4':
            (self.x, self.x_sf,
             self.topk_idx, self.topk_weights,
             self.l1_acts, self.l1_acts_sf,
             self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)
        else:
            (self.x, self.x_sf,
             self.topk_idx, self.topk_weights) = slice_input_buffers(self.buffer, self.handle.buffer_ptrs, group.rank())
        self.x._mega_moe_buffer = weakref.ref(self)

    def _check_live(self) -> None:
        if self.buffer is None or self.handle is None or self.group is None:
            raise ValueError('MegaMoE workspace has been destroyed')

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
    mma_type: str = "fp4xfp4",
    activation: str = "swiglu",
    *,
    quantization: Optional[str] = None,
) -> SymmBuffer:
    """Create the PPU workspace with native MegaMoE dimensions.

    The workspace is filled by mega_moe_pre_dispatch and consumed by
    the corresponding complete compute entry. MMA modes are 'fp4xfp4' (alias 'mxfp4xmxfp4'),
    'fp8xfp4', 'bf16xfp4' and 'fp8xfp8'. Only fp8xfp8 accepts quantization,
    selecting channelwise (default) or blockwise scales.
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
        mma_type=mma_type, activation=activation, quantization=quantization)


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
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    buf_x: torch.Tensor,
    buf_x_sf: torch.Tensor,
    buf_topk_idx: torch.Tensor,
    buf_topk_weights: torch.Tensor,
    num_tokens: int,
    group_size: int = 32,
    mma_type: str = "fp4xfp4",
) -> None:
    """Prepare BF16 inputs and routes for the selected precision workspace.

    Pass the four original views of one live SymmBuffer, as in SGLang.
    num_tokens must match x.shape[0]. MXFP4 requires group_size=32;
    other precisions use the quantization granularity fixed by their workspace.
    Input expert IDs can be int32 or int64. The workspace stores them as int64.
    Valid physical expert IDs must be distinct within each token.
    W4A4 SFA is produced and pushed directly into internal owner staging; x_sf is not
    a public source-scale input, and callers must use this helper to prepare it.
    """
    _C.mega_moe_pre_dispatch(
        x, topk_idx, topk_weights,
        buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights,
        num_tokens, group_size, _normalize_mma_type(mma_type))


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
    global_num_tokens: Optional[int] = None,
) -> None:
    """Consume published buffers through the complete C++ MegaMoE entry.

    y must be contiguous BF16 [num_tokens, hidden] on the buffer device.
    Unused output rows have invalid routes and produce zeros.
    global_num_tokens is the total input token count across the EP group
    before top-k routing. It tunes GEMM tiles using estimated incoming work,
    including on ranks with zero local rows. None uses the local token estimate.
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
    if not isinstance(sym_buffer, SymmBuffer) or sym_buffer.mma_type != 'fp4xfp4':
        raise ValueError("fp4_fp4_mega_moe requires SymmBuffer(mma_type='fp4xfp4')")
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
        sym_buffer.staging_config, global_num_tokens)


def _interleave_weights(tensor: torch.Tensor, gran: int = 8) -> torch.Tensor:
    experts, n, k = tensor.shape
    return (tensor.reshape(experts, 2, n // (2 * gran), gran, k)
            .transpose(1, 2).contiguous().reshape(experts, n, k))


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
    """Pack canonical MXFP4 weights and UE8M0 scales for BF16 x MXFP4 SiTU."""
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
    sym_buffer: SymmBuffer,
    weight_components: int = 2,
) -> None:
    sym_buffer._check_live()
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
    sym_buffer: SymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    *,
    l1_weights_replicas: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    l2_weights_replicas: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    activation_alpha: Optional[float] = None,
    global_num_tokens: Optional[int] = None,
) -> None:
    """Complete FP8 MoE; scales follow sym_buffer.quantization, output is BF16.

    Original routing weights are applied exactly once in combine. The C++
    implementation runs dispatch, GEMM1/activation/quantization, GEMM2/scatter,
    and completion. Master and replica tensors may be separate allocations.
    activation_alpha follows fp4_fp4_mega_moe: OAI-only, with None using 1.702.
    global_num_tokens follows the same EP-wide tiling contract as fp4_fp4_mega_moe.
    """
    if not isinstance(sym_buffer, SymmBuffer) or sym_buffer.mma_type != 'fp8xfp8':
        raise TypeError("fp8_fp8_mega_moe requires SymmBuffer(mma_type='fp8xfp8')")
    _check_routed_buffer(y, l1_weights, sym_buffer)
    _C.fp8_fp8_mega_moe(
        y, l1_weights, l2_weights, cumulative_local_expert_recv_stats,
        sym_buffer.buffer, sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank, sym_buffer.num_experts, sym_buffer.num_topk,
        activation, activation_clamp, fast_math, activation_alpha,
        l1_weights_replicas, l2_weights_replicas, sym_buffer.quantization, global_num_tokens,
    )


def bf16_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: SymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    activation: str = "situ",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    *,
    l1_weights_replicas: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    l2_weights_replicas: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    situ_beta: Optional[float] = 4.0,
    situ_linear_beta: Optional[float] = 25.0,
    global_num_tokens: Optional[int] = None,
) -> None:
    """Complete BF16 x MXFP4 SiTU MoE with BF16 output in original token order.

    SiTU parameters follow fp4_fp4_mega_moe; situ_linear_beta=None disables
    the up-branch softcap. situ_beta must be positive and finite.
    global_num_tokens follows the same EP-wide tiling contract as fp4_fp4_mega_moe.
    """
    if not isinstance(sym_buffer, SymmBuffer) or sym_buffer.mma_type != 'bf16xfp4':
        raise TypeError("bf16_fp4_mega_moe requires SymmBuffer(mma_type='bf16xfp4')")
    _check_routed_buffer(y, l1_weights, sym_buffer)
    _C.bf16_fp4_mega_moe(
        y, l1_weights, l2_weights, cumulative_local_expert_recv_stats,
        sym_buffer.buffer, sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank, sym_buffer.num_experts, sym_buffer.num_topk,
        activation, activation_clamp, fast_math, situ_beta, situ_linear_beta,
        l1_weights_replicas, l2_weights_replicas, global_num_tokens,
    )


def transform_weights_for_w4a8_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor], l2_weights: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, float], Tuple[torch.Tensor, torch.Tensor, float]]:
    """Pack MXFP4 weights and UE8M0 scales, returning each bank's inverse scale.

    The inverse scale compensates weight exponent rebasing; it is independent
    of row activation scales and routing weights. Replica banks can be prepared
    separately and may have different inverse scales.
    """
    _check_routed_weights(l1_weights, l2_weights, torch.uint8)
    w1, s1 = l1_weights
    first = _C.preprocess_weights_and_scales(
        _C.PreprocessMode.W4A8_MXFP4_MMA, False,
        _interleave_weights(w1), _interleave_weights(s1))
    second = _C.preprocess_weights_and_scales(
        _C.PreprocessMode.W4A8_MXFP4_MMA, False, *l2_weights)
    return first, second


def fp8_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor, float],
    l2_weights: Tuple[torch.Tensor, torch.Tensor, float],
    sym_buffer: SymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    *,
    l1_weights_replicas: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None,
    l2_weights_replicas: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None,
    situ_beta: Optional[float] = 4.0,
    situ_linear_beta: Optional[float] = 25.0,
    activation_alpha: Optional[float] = None,
    global_num_tokens: Optional[int] = None,
) -> None:
    """Complete E4M3 x MXFP4 MoE, applying routing weights once in combine.

    SiTU and OAI parameters follow fp4_fp4_mega_moe. situ_linear_beta=None
    disables the up softcap; activation_alpha applies only to OAI.
    global_num_tokens follows the same EP-wide tiling contract as fp4_fp4_mega_moe.
    """
    if not isinstance(sym_buffer, SymmBuffer) or sym_buffer.mma_type != 'fp8xfp4':
        raise TypeError("fp8_fp4_mega_moe requires SymmBuffer(mma_type='fp8xfp4')")
    _check_routed_buffer(y, l1_weights, sym_buffer, weight_components=3)
    if activation != 'situ':
        situ_beta = None if situ_beta == 4.0 else situ_beta
        situ_linear_beta = None if situ_linear_beta == 25.0 else situ_linear_beta
    _C.fp8_fp4_mega_moe(
        y, l1_weights, l2_weights, cumulative_local_expert_recv_stats,
        sym_buffer.buffer, sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank, sym_buffer.num_experts, sym_buffer.num_topk,
        activation, activation_clamp, fast_math, situ_beta, situ_linear_beta, activation_alpha,
        l1_weights_replicas, l2_weights_replicas, global_num_tokens)
