"""FP8 GEMM + ReduceScatter with explicit communication storage and output.

Obtain workspaces collectively after setting the current device. Calls must use
matching shapes and order on every rank and be serialized across streams. Warm
up before Graph capture and retain workspaces, inputs and outputs until pending work and captured graphs
using them have been released.
"""

import torch
import torch.distributed as dist

from . import _C as cpp

__all__ = [
    'GemmReduceScatterWorkspace',
    'get_workspace_for_gemm_reduce_scatter',
    'fp8_gemm_nt_reduce_scatter',
]


class GemmReduceScatterWorkspace:
    """Communication storage returned by ``get_workspace_for_gemm_reduce_scatter``.

    ``num_max_rows`` and ``num_max_cols`` bound the full output dimensions.
    M/N/K and quantization are inferred from each call's input tensors. Both
    quantizations and different shapes can reuse the allocation sequentially.
    The process group, symmetric allocation, handle, peer tables and epochs
    belong to this workspace; outputs always belong to the caller.

    ``preset='decode'`` selects validated fused profiles with native fallback;
    ``preset='native'`` uses native selection. Per-call ``configs`` override
    either preset. TP1 uses the native dense GEMM without communication.
    """

    def __init__(self, group, num_max_rows, num_max_cols, *, preset='decode'):
        if not isinstance(group, dist.ProcessGroup):
            raise ValueError('group must be a process group containing the current rank')
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        if self.world not in (1, 2, 4, 8) or self.rank < 0:
            raise ValueError('GEMM + RS supports TP1, TP2, TP4 and TP8')
        capacities = [(num_max_rows, num_max_cols)]
        if self.world > 1:
            capacities = [None] * self.world
            dist.all_gather_object(capacities, (num_max_rows, num_max_cols), group=group)
        if any(type(rows) is not int or type(cols) is not int or rows <= 0 or cols <= 0 or cols % 16
               for rows, cols in capacities):
            raise ValueError('workspace capacities must be positive integers, with columns divisible by 16')
        if any(capacity != capacities[0] for capacity in capacities):
            raise ValueError('GEMM + RS requires the same workspace capacities on all ranks')
        self.num_max_rows, self.num_max_cols = num_max_rows, num_max_cols
        self._phase_elements = (max(num_max_rows, 4 * self.world) * num_max_cols + 127) // 128 * 128
        if self._phase_elements > 2**31 - 1:
            raise ValueError('GEMM + RS workspace packet indices must fit in int32')
        if preset not in ('decode', 'native'):
            raise ValueError('preset must be decode or native')
        self.group, self.preset = group, preset
        self.device = torch.device('cuda', torch.cuda.current_device())
        self._configs = {}
        self._phase_bytes = self._phase_elements * 2
        self._allocate(2 * self._phase_bytes, self.world * 128)
        self._slots = self.buffer[: 2 * self._phase_bytes].view(torch.bfloat16).view(2, self._phase_elements)
        self._completed = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._reset_tensors = [self._flags, self._completed]
        # Immutable capacity/layout metadata; no per-call shapes or outputs.
        words = [self.peer_ptrs.data_ptr(), self._flag_peers.data_ptr(),
                 self._completed.data_ptr(), self._phase_elements, self.rank]
        self._state = torch.tensor(words, dtype=torch.int64, device=self.device)
        self._ready()

    def _allocate(self, payload_bytes, flag_bytes):
        self._flag_offset = (payload_bytes + 255) // 256 * 256
        total = self._flag_offset + flag_bytes
        if self.world == 1:
            self.buffer = torch.empty(total, dtype=torch.uint8, device=self.device)
            self.handle = None
            self._views = [self.buffer]
        else:
            try:
                import torch.distributed._symmetric_memory as symm
            except ImportError as error:
                raise RuntimeError('multi-rank GEMM + RS requires a PPU-enabled PyTorch build with '
                                   'torch.distributed._symmetric_memory support') from error

            self.buffer = symm.empty(total, dtype=torch.uint8, device=self.device)
            self.handle = symm.rendezvous(self.buffer, group=self.group)
            self._views = [self.handle.get_buffer(r, (total,), torch.uint8) for r in range(self.world)]
        self.peer_ptrs = torch.tensor([v.data_ptr() for v in self._views], dtype=torch.int64, device=self.device)
        self._flag_peers = self.peer_ptrs + self._flag_offset
        self._flags = self.buffer[self._flag_offset :].view(torch.int32)
        self._flags.zero_()

    def _ready(self):
        if self.handle is not None:
            self.handle.barrier(channel=0)
        torch.cuda.synchronize(self.device)

    def _get_config(self, m, n, k, channelwise, configs):
        if configs is not None:
            if not isinstance(configs, (tuple, list)) or len(configs) != 8:
                raise ValueError('configs must contain num_sms, BM, BN, BK, WM, WN, stages and SMEM')
            if not isinstance(configs[7], (tuple, list)) or len(configs[7]) != 3:
                raise ValueError('configs SMEM must contain three integers')
            configs = (*configs[:7], tuple(configs[7]))
            if any(type(value) is not int for value in (*configs[:7], *configs[7])):
                raise ValueError('configs entries must be integers')
        key = m, n, k, channelwise, configs, self.preset, cpp.get_num_sms()
        if key not in self._configs:
            quantization = 'channelwise' if channelwise else 'blockwise'
            query = getattr(cpp, f'_gemm_comm_fp8_{quantization}_gemm_rs_config')
            self._configs[key] = query(m, n, k, self.world, configs=configs, preset=self.preset)
        return self._configs[key]

    def reset_state(self):
        """Collectively reset epochs before 2**31 calls, outside Graph capture.

        Retire pending calls first. Buffer addresses remain stable, allowing
        existing captured graphs to be replayed after the reset.
        """
        if torch.cuda.current_device() != self.device.index:
            raise RuntimeError('reset_state requires the workspace device to be current')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('reset_state cannot run during Graph capture')
        torch.cuda.synchronize(self.device)
        self._ready()
        for tensor in self._reset_tensors:
            tensor.zero_()
        self._ready()


def get_workspace_for_gemm_reduce_scatter(group, num_max_rows, num_max_cols, *, preset='decode'):
    """Allocate reusable single-host PPU0015 TP1/2/4/8 communication storage.

    Call collectively with the selected process group after setting the current
    device, outside Graph capture. Capacities bound the full GEMM output M/N; K and quantization are
    determined by each computation call. Column capacity must be divisible by
    16. All ranks use identical capacities. No output tensor is allocated.
    """
    return GemmReduceScatterWorkspace(group, num_max_rows, num_max_cols, preset=preset)


def fp8_gemm_nt_reduce_scatter(lhs, rhs, out, *, workspace, configs=None):
    """Write the BF16 row shard of an NT FP8 GEMM into ``out``.

    Infer M/N/K from ``lhs[0]: [M,K]`` and ``rhs[0]: [N,K]``. FP32 scales
    select blockwise (A[M,K/128], W[ceil(N/128),K/128]) or channelwise
    (A[M,1], W[N,1]) quantization. Blockwise requires K>512 and K%128=0;
    channelwise supports K%32=0. ``out`` must be contiguous BF16 [M/TP,N].
    ``workspace`` owns capacity and communication state, independently of K and
    actual M/N. ``configs`` follows the public GEMM eight-element convention.
    TP1 uses the native GEMM and rejects explicit blockwise BN192 tiles,
    which require the fused launcher's weight-scale refinement.
    The operation returns None, like the existing GEMM interfaces.
    """
    if not isinstance(workspace, GemmReduceScatterWorkspace):
        raise ValueError('workspace must be a GemmReduceScatterWorkspace')
    if torch.cuda.current_device() != workspace.device.index:
        raise RuntimeError('GEMM + RS requires the workspace device to be current')
    for name, pair in (('A', lhs), ('W', rhs)):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f'{name} requires (FP8 data, FP32 scales)')
        data, scales = pair
        if not isinstance(data, torch.Tensor) or not isinstance(scales, torch.Tensor) or data.ndim != 2:
            raise ValueError(f'{name} requires matrix data and tensor scales')
        if data.dtype != torch.float8_e4m3fn or scales.dtype != torch.float32:
            raise ValueError(f'{name} requires FP8 E4M3 data and FP32 scales')
        if data.device != workspace.device or scales.device != workspace.device:
            raise ValueError(f'{name} operands must be on the workspace device')
        if not data.is_contiguous():
            raise ValueError(f'{name} data must be contiguous')
    m, k = lhs[0].shape
    n, rhs_k = rhs[0].shape
    if m <= 0 or m % workspace.world or n <= 0 or n % 16 or k <= 0 or k % 32 or k != rhs_k or k > 2**31 - 1:
        raise ValueError('requires positive shapes, matching K, M divisible by TP, N%16=0 and K%32=0')
    lm = m // workspace.world
    if lm not in (1, 2, 4, 8) and lm % 16:
        raise ValueError('GEMM + RS requires local M in {1,2,4,8} or a multiple of 16')
    if m > workspace.num_max_rows or n > workspace.num_max_cols:
        raise ValueError('operand shape exceeds communication workspace capacity')
    channelwise = tuple(lhs[1].shape) == (m, 1) and tuple(rhs[1].shape) == (n, 1)
    scale_shapes = ((m, 1), (n, 1)) if channelwise else ((m, k // 128), ((n + 127) // 128, k // 128))
    if not channelwise and (k <= 512 or k % 128):
        raise ValueError('blockwise requires K>512 and K%128=0')
    if tuple(lhs[1].shape) != scale_shapes[0] or tuple(rhs[1].shape) != scale_shapes[1]:
        raise ValueError(f'FP8 scale shapes must be {scale_shapes}')
    if not rhs[1].is_contiguous() or (channelwise and not lhs[1].is_contiguous()):
        raise ValueError('weight and channelwise activation scales must be contiguous')
    if not isinstance(out, torch.Tensor) or out.device != workspace.device or out.dtype != torch.bfloat16:
        raise ValueError('out must be a BF16 tensor on the workspace device')
    if tuple(out.shape) != (lm, n) or not out.is_contiguous():
        raise ValueError('out must be a contiguous [M/TP,N] tensor')
    if workspace.world == 1:
        if configs is not None:
            configs = workspace._get_config(m, n, k, channelwise, configs)[0]
        cpp.fp8_gemm_nt(lhs, rhs, out, configs=configs)
        return
    config = workspace._get_config(m, n, k, channelwise, configs)
    quantization = 'channelwise' if channelwise else 'blockwise'
    launch = getattr(cpp, f'_gemm_comm_fp8_{quantization}_gemm_rs')
    launch(lhs, rhs, out, workspace._state, m, workspace.rank, workspace.world, config)
