import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from .config import _DISPATCH_MAX_TOPK, _PREFILL_TOKEN_THRESHOLD, _WORKSPACE_BLOCK_M_CANDIDATES, _is_env_set, _uses_prefill_gemm

_TOKEN_ALIGNMENT = math.lcm(*_WORKSPACE_BLOCK_M_CANDIDATES)


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _get_workspace_block_m_candidates() -> Tuple[int, ...]:
    return _WORKSPACE_BLOCK_M_CANDIDATES


def _get_max_tokens_per_source(num_max_tokens_per_rank: int) -> int:
    return align_up(num_max_tokens_per_rank, _TOKEN_ALIGNMENT)


def _uses_dynamic_pool(declared_max_tokens_per_rank: int) -> bool:
    if _is_env_set("DG_STATIC_POOL"):
        return False
    return declared_max_tokens_per_rank > _PREFILL_TOKEN_THRESHOLD


def _should_size_pool_by_arrival(declared_max_tokens_per_rank: int) -> bool:
    if _is_env_set("DG_POOL_RESERVED"):
        return False
    return _uses_dynamic_pool(declared_max_tokens_per_rank)


def _get_reserved_pool_tokens_per_expert(num_ranks: int, max_tokens_per_source: int) -> int:
    return align_up(num_ranks * max_tokens_per_source, math.lcm(*_get_workspace_block_m_candidates()))


def _get_num_max_pool_tokens(num_ranks: int,
                             aligned_max_tokens_per_rank: int,
                             num_topk: int,
                             num_experts_per_rank: int,
                             block_m_candidates: Tuple[int, ...],
                             max_tokens_per_source: int,
                             uses_dynamic_pool: bool = False) -> int:
    lcm_block_m = math.lcm(*block_m_candidates)
    num_max_recv_tokens = num_ranks * aligned_max_tokens_per_rank
    num_max_experts_per_token = min(num_topk, num_experts_per_rank)
    if uses_dynamic_pool:
        return align_up(num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (lcm_block_m - 1), lcm_block_m)
    packed_pool_tokens = align_up(
        num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (max(block_m_candidates) - 1),
        lcm_block_m,
    )
    return max(packed_pool_tokens, num_experts_per_rank * _get_reserved_pool_tokens_per_expert(num_ranks, max_tokens_per_source))


@dataclass(frozen=True)
class MegamoeWorkspaceLayout:
    num_bytes: int
    aligned_max_tokens_per_rank: int
    num_max_pool_tokens: int
    num_sf_pool_tokens: int
    max_tokens_per_source: int
    regions: Dict[str, Tuple[int, int]]


def _check_layout_args(num_ranks: int, num_experts: int, num_topk: int, hidden: int, intermediate_hidden: int) -> None:
    if num_ranks <= 0:
        raise ValueError("num_ranks must be positive")
    if num_experts <= 0 or num_experts % num_ranks != 0:
        raise ValueError("num_experts must be positive and divisible by num_ranks")
    if num_topk <= 0 or num_topk > min(_DISPATCH_MAX_TOPK, num_experts // num_ranks):
        raise ValueError(f"num_topk must be in [1, min({_DISPATCH_MAX_TOPK}, num_experts_per_rank)]")
    if hidden <= 0 or intermediate_hidden <= 0:
        raise ValueError("hidden sizes must be positive")
    if hidden % 512 != 0:
        raise ValueError("hidden must be a multiple of 512 for the PPU scale-factor AIU layout")
    if intermediate_hidden % 256 != 0:
        raise ValueError("intermediate_hidden must be a multiple of 256 for the PPU "
                         "scale-factor AIU layout")


def _get_input_row_strides(hidden: int, num_topk: int) -> Tuple[int, int, int]:
    return align_up(hidden // 32, 128), align_up(num_topk * 8, 128), align_up(num_topk * 4, 128)


def _layout_kwargs(num_experts: int, num_max_tokens_per_rank: int, num_topk: int, hidden: int, intermediate_hidden: int) -> Dict[str, int]:
    return dict(num_experts=num_experts,
                num_max_tokens_per_rank=num_max_tokens_per_rank,
                num_topk=num_topk,
                hidden=hidden,
                intermediate_hidden=intermediate_hidden)


def _get_workspace_region_bytes(num_ranks: int, num_experts: int, num_experts_per_rank: int, num_topk: int, hidden: int,
                                max_tokens_per_source: int, declared_max_tokens_per_rank: int, num_max_pool_tokens: int) -> int:
    workspace_bytes = align_up(32 + num_experts * 8 * 2 + num_experts_per_rank * 8, 128)

    dispatch_reserved_header_bytes = 512 + 2048
    dispatch_slot_x_bytes_per_token = hidden // 2
    input_sf_row_stride_bytes, input_topk_idx_row_stride_bytes, input_topk_weight_row_stride_bytes = (_get_input_row_strides(
        hidden, num_topk))
    dispatch_reserved_bytes = num_ranks * (dispatch_reserved_header_bytes + max_tokens_per_source *
                                           (dispatch_slot_x_bytes_per_token + input_sf_row_stride_bytes + input_topk_idx_row_stride_bytes +
                                            input_topk_weight_row_stride_bytes))
    slot_bytes = 3 * 128 + hidden // 2
    expert_count_row_bytes = align_up(num_experts_per_rank * 4, 128)
    dispatch_region_bytes = (num_ranks * max_tokens_per_source * slot_bytes + expert_count_row_bytes + 128 + num_ranks * 128)
    if declared_max_tokens_per_rank > _PREFILL_TOKEN_THRESHOLD:
        dispatch_region_bytes += num_ranks * (align_up(num_experts_per_rank, 32) // 32) * 128
        dispatch_region_bytes += expert_count_row_bytes + 128
    workspace_bytes += max(dispatch_reserved_bytes, dispatch_region_bytes)
    workspace_bytes += num_max_pool_tokens * 12
    workspace_bytes = align_up(workspace_bytes, 128)
    workspace_bytes += num_experts_per_rank * 128
    return align_up(workspace_bytes, 16)


def get_megamoe_workspace_layout(num_ranks: int,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int,
                                 num_topk: int,
                                 hidden: int,
                                 intermediate_hidden: int,
                                 max_tokens_per_source: Optional[int] = None,
                                 declared_max_tokens_per_rank: Optional[int] = None) -> MegamoeWorkspaceLayout:
    _check_layout_args(num_ranks, num_experts, num_topk, hidden, intermediate_hidden)

    if declared_max_tokens_per_rank is None:
        declared_max_tokens_per_rank = num_max_tokens_per_rank
    aligned_max_tokens_per_rank = align_up(num_max_tokens_per_rank, _TOKEN_ALIGNMENT)
    if max_tokens_per_source is None:
        max_tokens_per_source = _get_max_tokens_per_source(aligned_max_tokens_per_rank)
    num_experts_per_rank = num_experts // num_ranks
    block_m_candidates = _get_workspace_block_m_candidates()
    num_max_pool_tokens = _get_num_max_pool_tokens(
        num_ranks,
        aligned_max_tokens_per_rank,
        num_topk,
        num_experts_per_rank,
        block_m_candidates,
        max_tokens_per_source,
        uses_dynamic_pool=_should_size_pool_by_arrival(declared_max_tokens_per_rank),
    )
    regions: Dict[str, Tuple[int, int]] = {}
    offset = 0

    workspace_bytes = _get_workspace_region_bytes(num_ranks, num_experts, num_experts_per_rank, num_topk, hidden, max_tokens_per_source,
                                                  declared_max_tokens_per_rank, num_max_pool_tokens)
    regions["workspace"] = (offset, workspace_bytes)
    offset += workspace_bytes

    def append_region(name: str, num_bytes: int) -> None:
        nonlocal offset
        regions[name] = (offset, num_bytes)
        offset += num_bytes

    append_region("x", aligned_max_tokens_per_rank * (hidden // 2))
    input_sf_stride, input_topk_idx_stride, input_topk_weight_stride = _get_input_row_strides(hidden, num_topk)
    append_region("input_sf", aligned_max_tokens_per_rank * input_sf_stride)
    append_region("input_topk_idx", aligned_max_tokens_per_rank * input_topk_idx_stride)
    append_region("input_topk_weight", aligned_max_tokens_per_rank * input_topk_weight_stride)
    append_region("l1_acts", num_max_pool_tokens * (hidden // 2))

    num_sf_pool_tokens = num_max_pool_tokens
    append_region("l1_acts_sf", num_sf_pool_tokens * (hidden // 32))
    append_region("l1_topk_weights", num_max_pool_tokens * 4)
    append_region("l2_acts", num_max_pool_tokens * (intermediate_hidden // 2))
    append_region(
        "l2_acts_sf",
        num_sf_pool_tokens * (intermediate_hidden // 32),
    )
    append_region(
        "combine",
        num_topk * aligned_max_tokens_per_rank * hidden * 2,
    )

    return MegamoeWorkspaceLayout(
        num_bytes=offset,
        aligned_max_tokens_per_rank=aligned_max_tokens_per_rank,
        num_max_pool_tokens=num_max_pool_tokens,
        num_sf_pool_tokens=num_sf_pool_tokens,
        max_tokens_per_source=max_tokens_per_source,
        regions=regions,
    )


def _normalize_peer_ptrs(peer_ptrs: Sequence[int] | torch.Tensor) -> torch.Tensor:
    if isinstance(peer_ptrs, torch.Tensor):
        if peer_ptrs.dtype != torch.int64:
            raise TypeError("peer_ptrs tensor must have int64 dtype")
        return peer_ptrs.detach().cpu().contiguous()
    return torch.tensor(list(peer_ptrs), dtype=torch.int64, device='cpu')


class MegamoeWorkspace:

    def __init__(self,
                 local_buffer: torch.Tensor,
                 peer_ptrs: Sequence[int] | torch.Tensor,
                 rank: int,
                 num_experts: int,
                 num_max_tokens_per_rank: int,
                 num_topk: int,
                 hidden: int,
                 intermediate_hidden: int,
                 should_zero: bool = False,
                 max_tokens_per_source: Optional[int] = None):
        if local_buffer.dtype not in (torch.uint8, torch.int8):
            raise TypeError("local_buffer must have uint8 or int8 dtype")
        if not local_buffer.is_cuda or not local_buffer.is_contiguous():
            raise ValueError("local_buffer must be a contiguous CUDA tensor")

        peer_ptrs = _normalize_peer_ptrs(peer_ptrs)
        self.num_ranks = int(peer_ptrs.numel())
        if not 0 <= rank < self.num_ranks:
            raise ValueError("rank is outside peer_ptrs")

        self.rank = rank
        self.num_experts = num_experts
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden

        self.declared_max_tokens_per_rank = num_max_tokens_per_rank
        self.layout = get_megamoe_workspace_layout(self.num_ranks,
                                                   max_tokens_per_source=max_tokens_per_source,
                                                   declared_max_tokens_per_rank=num_max_tokens_per_rank,
                                                   **_layout_kwargs(num_experts, num_max_tokens_per_rank, num_topk, hidden,
                                                                    intermediate_hidden))
        self.aligned_max_tokens_per_rank = self.layout.aligned_max_tokens_per_rank
        if local_buffer.numel() < self.layout.num_bytes:
            raise ValueError(f"symmetric buffer is too small: need {self.layout.num_bytes} "
                             f"bytes, got {local_buffer.numel()}")

        self.peer_ptrs = peer_ptrs
        self.buffer = self._slice_workspace_buffer(local_buffer, rank)

        self._bind_typed_views()
        self.symmetric_handle = None
        self._gemm_weight_cache = None

        if should_zero:
            self.zero_()

    def _slice_workspace_buffer(self, local_buffer: torch.Tensor, rank: int) -> torch.Tensor:
        buffer = local_buffer.view(torch.uint8).flatten()[:self.layout.num_bytes]
        if buffer.data_ptr() != int(self.peer_ptrs[rank]):
            raise ValueError("local_buffer.data_ptr() must equal peer_ptrs[rank]; apply the "
                             "same region offset to every peer base")
        if buffer.data_ptr() % 128 != 0:
            raise ValueError("Megamoe workspace base must be 128-byte aligned: the dispatch slot lines are "
                             "written and read as 128B lines, and only a 128B-aligned line store is one 128B "
                             "request on the fabric (see workspace._get_workspace_region_bytes)")
        return buffer

    def _bind_typed_views(self) -> None:
        num_tokens = self.aligned_max_tokens_per_rank
        _sf_stride, _topk_idx_stride, _topk_weight_stride = _get_input_row_strides(self.hidden, self.num_topk)

        self.x = self._make_typed_view("x", torch.uint8, (num_tokens, self.hidden // 2))
        self.x_sf = self._make_typed_view("input_sf", torch.float32, (num_tokens, self.hidden // 128), (_sf_stride // 4, 1))
        self.topk_idx = self._make_typed_view("input_topk_idx", torch.int64, (num_tokens, self.num_topk), (_topk_idx_stride // 8, 1))
        self.topk_weights = self._make_typed_view("input_topk_weight", torch.float32, (num_tokens, self.num_topk),
                                                  (_topk_weight_stride // 4, 1))
        self.l1_acts = self._make_typed_view("l1_acts", torch.uint8, (self.layout.num_max_pool_tokens, self.hidden // 2))
        self.l2_acts = self._make_typed_view("l2_acts", torch.uint8, (self.layout.num_max_pool_tokens, self.intermediate_hidden // 2))
        if _uses_prefill_gemm(self.declared_max_tokens_per_rank, self.intermediate_hidden):
            self.l1_acts_sf = self._make_typed_view("l1_acts_sf", torch.uint16, (self.layout.num_sf_pool_tokens, self.hidden // 64),
                                                    (1, self.layout.num_sf_pool_tokens))
            self.l2_acts_sf = self._make_typed_view("l2_acts_sf", torch.uint16,
                                                    (self.layout.num_sf_pool_tokens, self.intermediate_hidden // 64),
                                                    (1, self.layout.num_sf_pool_tokens))
        else:
            self.l1_acts_sf = None
            self.l2_acts_sf = self._make_typed_view("l2_acts_sf", torch.float32,
                                                    (self.layout.num_sf_pool_tokens, self.intermediate_hidden // 128),
                                                    (1, self.layout.num_sf_pool_tokens))
        self._tiled_scale_views = (self.l1_acts_sf, self.l2_acts_sf) if self.l1_acts_sf is not None else None
        self._decode_scale_view = self.l2_acts_sf if self.l1_acts_sf is None else None
        self.combine = self._make_typed_view("combine", torch.bfloat16, (self.num_topk, num_tokens, self.hidden))

    def _select_gemm_scale_views(self, uses_tiled_gemm: bool) -> None:
        if uses_tiled_gemm == (self.l1_acts_sf is not None):
            return
        if uses_tiled_gemm:
            if self._tiled_scale_views is None:
                self._tiled_scale_views = (self._make_typed_view("l1_acts_sf", torch.uint16,
                                                                 (self.layout.num_sf_pool_tokens, self.hidden // 64),
                                                                 (1, self.layout.num_sf_pool_tokens)),
                                           self._make_typed_view("l2_acts_sf", torch.uint16,
                                                                 (self.layout.num_sf_pool_tokens, self.intermediate_hidden // 64),
                                                                 (1, self.layout.num_sf_pool_tokens)))
            self.l1_acts_sf, self.l2_acts_sf = self._tiled_scale_views
        else:
            self.l1_acts_sf = None
            self.l2_acts_sf = self._decode_scale_view

    def _make_typed_view(self,
                         name: str,
                         dtype: torch.dtype,
                         shape: Tuple[int, ...],
                         stride: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        offset, num_bytes = self.layout.regions[name]
        raw = self.buffer.narrow(0, offset, num_bytes)
        typed = raw.view(dtype)
        if stride is not None:
            return torch.as_strided(typed, shape, stride)
        return typed.view(shape)

    def zero_(self) -> "MegamoeWorkspace":
        self.buffer.zero_()
        return self

    def destroy(self) -> None:
        self.x = None
        self.x_sf = None
        self.topk_idx = None
        self.topk_weights = None
        self.l1_acts = None
        self.l1_acts_sf = None
        self._tiled_scale_views = None
        self._decode_scale_view = None
        self.l2_acts = None
        self.l2_acts_sf = None
        self.combine = None
        self.peer_ptrs = None
        self.buffer = None
        self.symmetric_handle = None
        self._gemm_weight_cache = None

    @classmethod
    def allocate_symmetric(cls,
                           group: dist.ProcessGroup,
                           num_experts: int,
                           num_max_tokens_per_rank: int,
                           num_topk: int,
                           hidden: int,
                           intermediate_hidden: int,
                           device: Optional[torch.device] = None,
                           should_zero: bool = True) -> "MegamoeWorkspace":
        if device is None:
            device = torch.device('cuda', torch.cuda.current_device())
        world_size = group.size()
        rank = group.rank()
        max_tokens_per_source = _get_max_tokens_per_source(num_max_tokens_per_rank)
        layout = get_megamoe_workspace_layout(world_size,
                                              max_tokens_per_source=max_tokens_per_source,
                                              declared_max_tokens_per_rank=num_max_tokens_per_rank,
                                              **_layout_kwargs(num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden))
        if world_size == 1:
            local_buffer = torch.empty(layout.num_bytes, dtype=torch.int8, device=device)
            handle = None
            peer_ptrs = [local_buffer.data_ptr()]
        else:
            try:
                import torch.distributed._symmetric_memory as symm_mem
            except ImportError as ex:
                raise RuntimeError("torch symmetric memory is unavailable in this PyTorch build") from ex
            local_buffer = symm_mem.empty(layout.num_bytes, dtype=torch.int8, device=device)
            handle = symm_mem.rendezvous(local_buffer, group=group)
            peer_ptrs = handle.buffer_ptrs

        workspace = cls(local_buffer,
                        peer_ptrs,
                        rank,
                        should_zero=False,
                        max_tokens_per_source=max_tokens_per_source,
                        **_layout_kwargs(num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden))
        workspace.symmetric_handle = handle
        if should_zero:
            workspace.zero_()
            torch.cuda.synchronize(device)
            dist.barrier(group=group)
        return workspace


def get_symm_buffer_for_megamoe(group: dist.ProcessGroup,
                                num_experts: int,
                                num_max_tokens_per_rank: int,
                                num_topk: int,
                                hidden: int,
                                intermediate_hidden: int,
                                use_fp8_dispatch: bool = True,
                                activation: str = 'swiglu') -> MegamoeWorkspace:
    if not use_fp8_dispatch:
        raise ValueError("PPU890 fused Megamoe currently requires FP8 dispatch")
    if activation != 'swiglu':
        raise ValueError("PPU890 fused Megamoe currently supports only SwiGLU")
    return MegamoeWorkspace.allocate_symmetric(
        group, **_layout_kwargs(num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden))
