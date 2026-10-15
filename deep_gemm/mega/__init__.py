from typing import Tuple

import torch

try:
    import megamoe
except Exception:
    import os as _os
    import sys as _sys
    _megamoe_dir = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
    if _os.path.isfile(_os.path.join(_megamoe_dir, 'megamoe', '__init__.py')):
        _sys.path.insert(0, _megamoe_dir)
    try:
        import megamoe
    except Exception as exception:
        raise ImportError(
            'Failed to load mega kernels, please make sure the megamoe directory is present '
            f'(it ships with DeepGEMM): {exception}')


class SymmBuffer:
    def __init__(self, workspace, group, num_max_tokens_per_rank: int, hidden: int, intermediate_hidden: int):
        self._workspace = workspace
        self.group = group
        self.num_experts = workspace.num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = workspace.num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden

        self.buffer = workspace.buffer
        self.handle = workspace.symmetric_handle

        self.x = torch.empty((workspace.aligned_max_tokens_per_rank, hidden), dtype=torch.bfloat16,
                             device=workspace.buffer.device)
        self.x_sf = workspace.x_sf
        self.topk_idx = workspace.topk_idx
        self.topk_weights = workspace.topk_weights
        self.l1_acts = workspace.l1_acts
        self.l2_acts = workspace.l2_acts
        self.l2_acts_sf = workspace.l2_acts_sf
        self.l1_acts_sf = None

    def zero_(self) -> 'SymmBuffer':
        self._workspace.zero_()
        self.x.zero_()
        return self

    def destroy(self) -> None:
        self._workspace.destroy()
        self.x = None
        self.buffer = None
        self.handle = None
        self.x_sf = None
        self.topk_idx = None
        self.topk_weights = None
        self.l1_acts = None
        self.l2_acts = None
        self.l2_acts_sf = None


_DISPATCH_MAX_TOPK = 10


def _check_shape_constraints(num_ranks: int, num_experts: int, num_topk: int, hidden: int, intermediate_hidden: int) -> None:
    if hidden != 8192:
        raise NotImplementedError(f'PPU MegaMoE is laid out for hidden == 8192 (32 dispatch lines '
                                  f'+ 2 SF lines), got {hidden}')
    if hidden % 512 != 0:
        raise ValueError(f'hidden must be a multiple of 512 for the PPU scale-factor AIU layout, got {hidden}')
    if intermediate_hidden % 256 != 0:
        raise ValueError(f'intermediate_hidden must be a multiple of 256 for the PPU scale-factor AIU layout, '
                         f'got {intermediate_hidden}')
    if num_experts % num_ranks != 0:
        raise ValueError(f'num_experts ({num_experts}) must be divisible by num_ranks ({num_ranks})')
    if not 1 <= num_topk <= min(_DISPATCH_MAX_TOPK, num_experts // num_ranks):
        raise ValueError(f'num_topk must be in [1, min({_DISPATCH_MAX_TOPK}, num_experts // num_ranks)], '
                         f'got {num_topk}')


def get_symm_buffer_for_mega_moe(group,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 use_fp8_dispatch: bool = True,
                                 activation: str = 'swiglu') -> SymmBuffer:
    _check_shape_constraints(group.size(), num_experts, num_topk, hidden, intermediate_hidden)
    if not use_fp8_dispatch:
        raise ValueError('PPU MegaMoE currently requires FP8 dispatch')
    if activation != 'swiglu':
        raise ValueError('PPU MegaMoE currently supports only SwiGLU')
    workspace = megamoe.get_symm_buffer_for_megamoe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        use_fp8_dispatch=use_fp8_dispatch, activation=activation)
    return SymmBuffer(workspace, group, num_max_tokens_per_rank, hidden, intermediate_hidden)


def transform_weights_for_mega_moe(l1_weights: Tuple[torch.Tensor, torch.Tensor],
                                   l2_weights: Tuple[torch.Tensor, torch.Tensor]):
    return megamoe.transform_weights_for_megamoe(l1_weights, l2_weights)


def fp4_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: float = None,
                     fast_math: bool = True) -> None:
    if not isinstance(sym_buffer, SymmBuffer):
        raise TypeError('sym_buffer must be the object returned by get_symm_buffer_for_mega_moe')
    if recipe != (1, 1, 32):
        raise ValueError(f'PPU MegaMoE requires recipe == (1, 1, 32), got {recipe}')
    if activation != 'swiglu':
        raise ValueError(f'PPU MegaMoE supports only SwiGLU, got {activation!r}')
    num_tokens = y.size(0)
    x_bf16 = sym_buffer.x[:num_tokens]
    megamoe.launch_fp4_fp4_megamoe(
        y,
        l1_weights, l2_weights,
        sym_buffer._workspace,
        recipe=recipe,
        activation=activation,
        activation_clamp=activation_clamp,
        x_bf16=x_bf16)
