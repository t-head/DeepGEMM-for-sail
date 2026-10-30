"""Public PPU MegaMoE workspace and complete-call wrappers.

Matches the export organization of upstream ``deep_gemm.mega``. Quantized
operands use the PPU MXFP4 format; C++ validates tensors and runs the pipeline.
"""
from typing import Optional, Tuple
from types import SimpleNamespace
import warnings

import torch
import torch.distributed as dist

from .. import deep_gemm_cpp
from ..jit_kernels.utils import is_ppu1v5_device

__all__ = [
    "SymmBuffer", "get_symm_buffer_for_mega_moe", "transform_weights_for_mega_moe",
    "fp4_fp4_mega_moe", "mega_moe_pre_dispatch",
]

def _check_activation(activation: str) -> None:
    if activation not in ("swiglu", "step", "situ"):
        raise ValueError(f"unsupported precision/activation combination for W4A4: {activation}")


class SymmBuffer:
    """One reusable allocation for the complete PPU W4A4 MoE workspace.

    ``intermediate_hidden`` is I; GEMM1 derives 2I internally. Public inputs
    are packed MXFP4 ``x`` [capacity,H/2], UE8M0 ``x_sf`` [capacity,H/32],
    int64 ``topk_idx`` and FP32 ``topk_weights`` [capacity,topk]. The complete
    call consumes the prefix selected by y.shape[0]. Internal storage and
    communication state are sliced and recycled by C++.
    Valid physical expert IDs must be distinct within each token's top-k;
    masked IDs such as -1 may repeat. External remapping must preserve this.

    ``base`` aliases an existing compatible allocation and communication handle;
    aliases must be used sequentially on one stream. Destroy after pending work
    completes. Shared experts and FP8 dispatch are not supported by this backend.
    """

    def __init__(self, group: dist.ProcessGroup, num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int, num_shared_experts: int = 0,
                 mma_type: str = "fp4xfp4", activation: str = "swiglu",
                 base: Optional["SymmBuffer"] = None):
        if not is_ppu1v5_device():
            raise RuntimeError("MegaMoE requires PPU1.5")
        if mma_type != "fp4xfp4":
            raise ValueError("PPU MegaMoE requires mma_type='fp4xfp4' (MXFP4)")
        _check_activation(activation)
        alignment = deep_gemm_cpp.get_token_alignment_for_mega_moe()
        num_max_tokens_per_rank = (num_max_tokens_per_rank + alignment - 1) // alignment * alignment
        total, slice_inputs = deep_gemm_cpp.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts, num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden, mma_type, activation, num_shared_experts)
        self.group, self.num_ranks = group, group.size()
        self.num_experts, self.num_topk = num_experts, num_topk
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden, self.intermediate_hidden = hidden, intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type, self.activation = mma_type, activation
        self._layout_key = (self.num_ranks, num_experts, num_max_tokens_per_rank,
                            num_topk, hidden, intermediate_hidden, mma_type)
        if base is not None:
            if (not isinstance(base, SymmBuffer) or base.buffer is None or base.handle is None
                    or base.group is not group):
                raise ValueError("Cannot reuse an invalid SymmBuffer or a different process group")
            if self._layout_key != base._layout_key:
                raise ValueError("SymmBuffer base reuse requires a compatible workspace layout")
            if (base.buffer.device.index != torch.cuda.current_device()
                    or total > base.buffer.nbytes):
                raise ValueError("SymmBuffer base has an incompatible device or insufficient capacity")
            self.buffer, self.handle = base.buffer, base.handle
        else:
            if self.num_ranks == 1:
                self.buffer = torch.empty(total, dtype=torch.uint8, device="cuda")
                self.handle = SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            else:
                import torch.distributed._symmetric_memory as symm_mem
                self.buffer = symm_mem.empty(total, dtype=torch.uint8, device="cuda")
                self.handle = symm_mem.rendezvous(self.buffer, group=group)
            self.buffer.zero_()
            group.barrier()
            torch.cuda.synchronize()
        self.x, self.x_sf, self.topk_idx, self.topk_weights = slice_inputs(self.buffer)

    def destroy(self) -> None:
        self.buffer = self.handle = self.group = None
        self.x = self.x_sf = self.topk_idx = self.topk_weights = None


def get_symm_buffer_for_mega_moe(
        group: dist.ProcessGroup, num_experts: int, num_max_tokens_per_rank: int,
        num_topk: int, hidden: int, intermediate_hidden: int, num_shared_experts: int = 0,
        use_fp8_dispatch: Optional[bool] = None, mma_type: str = "fp4xfp4",
        activation: str = "swiglu",
) -> SymmBuffer:
    """Compatibility factory; PPU supports MXFP4 dispatch and zero shared experts.

    Physical expert IDs cover masters followed by any replicas on each rank.
    Tile selection and intermediate metadata remain internal.
    """
    if use_fp8_dispatch is not None:
        if not isinstance(use_fp8_dispatch, bool):
            raise TypeError("use_fp8_dispatch must be a bool or None")
        if use_fp8_dispatch != (mma_type.split("x")[0] == "fp8"):
            raise ValueError("use_fp8_dispatch is incompatible with mma_type; PPU requires MXFP4 dispatch")
        warnings.warn("use_fp8_dispatch is deprecated; use mma_type instead",
                      DeprecationWarning, stacklevel=2)
    return SymmBuffer(group, num_experts, num_max_tokens_per_rank, num_topk,
                      hidden, intermediate_hidden, num_shared_experts, mma_type, activation)


def _interleave_weights(tensor: torch.Tensor, gran: int = 8) -> torch.Tensor:
    experts, n, k = tensor.shape
    return (tensor.reshape(experts, 2, n // (2 * gran), gran, k)
            .transpose(1, 2).contiguous().reshape(experts, n, k))


def transform_weights_for_mega_moe(
        l1_weights: Tuple[torch.Tensor, torch.Tensor],
        l2_weights: Tuple[torch.Tensor, torch.Tensor], activation: str = "swiglu",
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Pack logical [gate; up] MXFP4 weights once, without requantization.

    Packed uint8 weights are [E, N, K/2], with uint8 UE8M0 scales [E, N, K/32].
    Use the returned pairs only with this backend. NVFP4 E4M3 scales are unsupported.
    E covers the expert bank passed to this call, in the same order in both pairs.
    Transform separate master and replica banks in separate calls at model load time.
    """
    _check_activation(activation)
    if not is_ppu1v5_device():
        raise RuntimeError("MegaMoE requires PPU1.5")

    def validate_pair(name, pair):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError(f"{name} must be a (packed MXFP4 weight, raw UE8M0 scale) pair")
        if not all(isinstance(t, torch.Tensor) for t in pair):
            raise TypeError(f"{name} weights and scales must be tensors")
        weight, scale = pair
        if weight.ndim != 3 or scale.ndim != 3:
            raise ValueError(f"{name} weights and scales must be rank-3 tensors")
        if any(t.dtype != torch.uint8 or not t.is_cuda or not t.is_contiguous() for t in pair):
            raise ValueError(f"{name} requires contiguous uint8 weights and raw UE8M0 scales on a PPU")
        if weight.device != scale.device:
            raise ValueError(f"{name} weights and scales must share a device")
        e, n, k = weight.shape
        if scale.shape != (e, n, k // 16):
            raise ValueError(f"{name} scales must have shape [local_experts, N, K/32]")
        return weight, scale

    l1_weight, l1_scale = validate_pair("l1_weights", l1_weights)
    l2_weight, l2_scale = validate_pair("l2_weights", l2_weights)
    experts, n, _ = l1_weight.shape
    intermediate, hidden = n // 2, l2_weight.shape[1]
    if (experts <= 0 or intermediate <= 0 or intermediate % 128 != 0 or
            hidden <= 0 or hidden % 512 != 0):
        raise ValueError("unsupported MegaMoE weight dimensions")
    if (l1_weight.shape != (experts, 2 * intermediate, hidden // 2) or
            l2_weight.shape != (experts, hidden, intermediate // 2)):
        raise ValueError("MegaMoE packed weights must have shapes [E, 2*I, H/2] and [E, H, I/2]")
    if l1_weight.device != l2_weight.device:
        raise ValueError("all MegaMoE weights and scales must share a device")

    l1_transformed = _interleave_weights(l1_weight)
    # Each 256-column N tile stores its UE8M0 scale pairs in K-major order.
    l1_scale = (_interleave_weights(l1_scale).view(torch.uint16)
                .reshape(experts, n // 256, 256, hidden // 64)
                .transpose(2, 3).contiguous().reshape(experts, n * hidden // 64))
    # GEMM2 consumes uint16 scale pairs in N-major order within each expert.
    l2_scale = l2_scale.view(torch.uint16).transpose(1, 2).contiguous().transpose(1, 2)
    return (l1_transformed, l1_scale), (l2_weight, l2_scale)


def _check_buffer(sym_buffer: SymmBuffer) -> None:
    if not isinstance(sym_buffer, SymmBuffer):
        raise TypeError("sym_buffer must be a SymmBuffer")
    if sym_buffer.buffer is None or sym_buffer.handle is None or sym_buffer.group is None:
        raise ValueError("SymmBuffer has been destroyed")
    if sym_buffer.mma_type != "fp4xfp4":
        raise ValueError("PPU MegaMoE requires mma_type='fp4xfp4' (MXFP4)")


def mega_moe_pre_dispatch(
        hidden_states: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor,
        sym_buffer: SymmBuffer, num_tokens: Optional[int] = None,
) -> None:
    """Quantize BF16 inputs and stage routes into the four public input views.

    This optional PPU helper can be replaced by filling the views in a preceding
    operator. The complete call derives its actual token count from y.shape[0].
    """
    _check_buffer(sym_buffer)
    deep_gemm_cpp.mega_moe_pre_dispatch(
        hidden_states, topk_idx, topk_weights,
        sym_buffer.x, sym_buffer.x_sf, sym_buffer.topk_idx, sym_buffer.topk_weights, num_tokens)


def fp4_fp4_mega_moe(
        y: torch.Tensor, l1_weights: Tuple[torch.Tensor, torch.Tensor],
        l2_weights: Tuple[torch.Tensor, torch.Tensor], sym_buffer: SymmBuffer,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        recipe: Tuple[int, int, int] = (1, 1, 32),
        activation: str = "swiglu", activation_clamp: Optional[float] = None,
        fast_math: bool = True, *,
        l1_weights_replicas: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        l2_weights_replicas: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        situ_beta: float = 4.0, situ_linear_beta: float = 25.0,
) -> None:
    """Write complete W4A4 routed MoE results into BF16 y [local_tokens, H].

    Inputs must already occupy the matching prefix of x/x_sf/topk_idx/topk_weights.
    A token's valid physical expert IDs must be distinct, including after remapping.
    Routing weights are applied once in combine. Every rank calls, including empty
    ranks. The library owns scratch and recycling; graph replay retains allocations
    and shapes. fast_math selects approximate GEMM1 activation math; False uses
    expf/division/tanhf without changing GEMM accumulation or MXFP4 formats.
    Common arguments follow SM90 fp8_mega_moe. The MXFP4 recipe is (1, 1, 32).
    Optional INT32 receive statistics have one element per local physical expert;
    each call adds valid routed assignments from all ranks, including zero-weight
    assignments. Initialize them once and retain them for graph replay; the library
    never resets these caller-owned cumulative statistics. SiTU coefficients are
    keyword-only PPU extensions, as are separate replica weight pairs for UltraEP.
    Without replica arguments, both weight pairs contain all local physical experts.
    With replicas, each rank's physical slots contain masters followed by replicas;
    both sets use the same weight transform and need not share an allocation.
    """
    _check_buffer(sym_buffer)
    _check_activation(activation)
    if activation != sym_buffer.activation:
        raise ValueError("activation does not match SymmBuffer activation")
    if not isinstance(y, torch.Tensor) or y.ndim != 2 or y.shape[1] != sym_buffer.hidden:
        raise ValueError("output must have shape [tokens, workspace hidden]")
    if (not isinstance(l1_weights, (tuple, list)) or len(l1_weights) != 2
            or not isinstance(l1_weights[0], torch.Tensor)
            or l1_weights[0].ndim != 3 or l1_weights[0].shape[1] != 2 * sym_buffer.intermediate_hidden):
        raise ValueError("l1_weights must match the workspace intermediate_hidden")
    deep_gemm_cpp.fp4_fp4_mega_moe(
        y, l1_weights, l2_weights, cumulative_local_expert_recv_stats,
        sym_buffer.buffer, sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank, sym_buffer.num_experts, sym_buffer.num_topk,
        recipe, activation, activation_clamp, fast_math,
        l1_weights_replicas, l2_weights_replicas, situ_beta, situ_linear_beta)
