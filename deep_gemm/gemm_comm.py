"""FP8 GEMM + ReduceScatter with caller-owned BF16 output tensors.

Construct the context collectively after initializing the default process group.
Calls must use matching shapes and order on every rank and be serialized across
streams. Warm up before Graph capture and keep the context, inputs and outputs
alive until all pending work and graphs using them have been released.
"""

import torch
import torch.distributed as dist

from . import deep_gemm_cpp as cpp

__all__ = ['GemmReduceScatterContext', 'fp8_gemm_nt_reduce_scatter']


class GemmReduceScatterContext:
    """Reusable communication buffers for single-host PPU0015 TP2/TP4.

    ``m`` and ``n`` describe the full GEMM output; ``k`` is local to each rank.
    Both quantizations share this context. ``preset='decode'`` selects fused
    profiles for validated shapes, falling back to native configuration selection.
    ``preset='native'`` always uses native selection. Per-call ``configs`` override
    either preset, using the same eight-element tuple as ``fp8_gemm_nt``.

    No output tensor is retained or reused by the context. Graphs capture each
    call's explicit output pointer without mutating the communication pointer table.
    """

    def __init__(self, m, n, k, *, preset='decode'):
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        if self.world not in (2, 4):
            raise ValueError('GEMM + RS supports TP2 and TP4')
        if m <= 0 or m % self.world or n <= 0 or n % 16 or k <= 0 or k % 32:
            raise ValueError('GEMM + RS requires positive shapes, M divisible by TP, N%16=0 and K%32=0')
        if max(m, 4 * self.world) * n > 2**31 - 1 or k > 2**31 - 1:
            raise ValueError('GEMM + RS requires packet indices and K to fit in int32')
        lm = m // self.world
        if lm not in (1, 2, 4, 8) and lm % 16:
            raise ValueError('GEMM + RS requires local M in {1,2,4,8} or a multiple of 16')
        if preset not in ('decode', 'native'):
            raise ValueError('preset must be decode or native')
        self.m, self.n, self.k = m, n, k
        self.preset = preset
        self.device = torch.device('cuda', torch.cuda.current_device())
        self._configs = {}
        self._slot_m = self.world * max(4, lm)
        self._phase_bytes = self._slot_m * n * 2
        self._allocate(2 * self._phase_bytes, self.world * 128)
        self._slots = self._arena[: 2 * self._phase_bytes].view(torch.bfloat16).view(2, self._slot_m, n)
        self._completed = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._reset_tensors = [self._flags, self._completed]
        words = [self._peers.data_ptr(), self._flag_peers.data_ptr(),
                 self._completed.data_ptr(), m, n, self.rank]
        self._state = torch.tensor(words, dtype=torch.int64, device=self.device)
        self._ready()

    def _allocate(self, payload_bytes, flag_bytes):
        import torch.distributed._symmetric_memory as symm

        self._flag_offset = (payload_bytes + 255) // 256 * 256
        total = self._flag_offset + flag_bytes
        self._arena = symm.empty(total, dtype=torch.uint8, device=self.device)
        self._handle = symm.rendezvous(self._arena, group=dist.group.WORLD)
        self._views = [self._handle.get_buffer(r, (total,), torch.uint8) for r in range(self.world)]
        self._peers = torch.tensor([v.data_ptr() for v in self._views], dtype=torch.int64, device=self.device)
        self._flag_peers = self._peers + self._flag_offset
        self._flags = self._arena[self._flag_offset :].view(torch.int32)
        self._flags.zero_()

    def _ready(self):
        self._handle.barrier(channel=0)
        torch.cuda.synchronize(self.device)

    def _get_config(self, channelwise, configs):
        if configs is not None:
            if not isinstance(configs, (tuple, list)) or len(configs) != 8:
                raise ValueError('configs must contain num_sms, BM, BN, BK, WM, WN, stages and SMEM')
            if not isinstance(configs[7], (tuple, list)) or len(configs[7]) != 3:
                raise ValueError('configs SMEM must contain three integers')
            configs = (*configs[:7], tuple(configs[7]))
            if any(type(value) is not int for value in (*configs[:7], *configs[7])):
                raise ValueError('configs entries must be integers')
        key = channelwise, configs, cpp.get_num_sms()
        if key not in self._configs:
            quantization = 'channelwise' if channelwise else 'blockwise'
            query = getattr(cpp, f'_gemm_comm_fp8_{quantization}_gemm_rs_config')
            self._configs[key] = query(self.m, self.n, self.k, self.world, configs=configs, preset=self.preset)
        return self._configs[key]

    def reset_state(self):
        """Collectively reset epochs before 2**31 calls, outside Graph capture.

        Retire pending calls first. Buffer addresses remain stable, allowing
        existing captured graphs to be replayed after the reset.
        """
        if torch.cuda.current_device() != self.device.index:
            raise RuntimeError('reset_state requires the context device to be current')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('reset_state cannot run during Graph capture')
        torch.cuda.synchronize(self.device)
        self._ready()
        for tensor in self._reset_tensors:
            tensor.zero_()
        self._ready()


def fp8_gemm_nt_reduce_scatter(lhs, rhs, out, *, comm, configs=None):
    """Write the BF16 row shard of an NT FP8 GEMM into ``out``.

    Infer M/N/K from ``lhs[0]: [M,K]`` and ``rhs[0]: [N,K]``. FP32 scales
    select blockwise (A[M,K/128], W[ceil(N/128),K/128]) or channelwise
    (A[M,1], W[N,1]) quantization. Blockwise requires K>512 and K%128=0;
    channelwise supports K%32=0. ``out`` must be contiguous BF16 [M/TP,N].
    ``configs`` follows the public GEMM eight-element tuple convention; explicit
    configs take precedence over the context preset, with SMEM recomputed in C++.
    The operation returns None, like the existing GEMM interfaces.
    """
    if not isinstance(comm, GemmReduceScatterContext):
        raise ValueError('comm must be a GemmReduceScatterContext')
    if torch.cuda.current_device() != comm.device.index:
        raise RuntimeError('GEMM + RS requires the context device to be current')
    for name, pair in (('A', lhs), ('W', rhs)):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f'{name} requires (FP8 data, FP32 scales)')
        data, scales = pair
        if not isinstance(data, torch.Tensor) or not isinstance(scales, torch.Tensor) or data.ndim != 2:
            raise ValueError(f'{name} requires matrix data and tensor scales')
        if data.dtype != torch.float8_e4m3fn or scales.dtype != torch.float32:
            raise ValueError(f'{name} requires FP8 E4M3 data and FP32 scales')
        if data.device != comm.device or scales.device != comm.device:
            raise ValueError(f'{name} operands must be on the context device')
        if not data.is_contiguous():
            raise ValueError(f'{name} data must be contiguous')
    m, k = lhs[0].shape
    n, rhs_k = rhs[0].shape
    if (m, n, k) != (comm.m, comm.n, comm.k) or k != rhs_k:
        raise ValueError('operand shapes must match the communication context')
    channelwise = tuple(lhs[1].shape) == (m, 1) and tuple(rhs[1].shape) == (n, 1)
    scale_shapes = ((m, 1), (n, 1)) if channelwise else ((m, k // 128), ((n + 127) // 128, k // 128))
    if not channelwise and (k <= 512 or k % 128):
        raise ValueError('blockwise requires K>512 and K%128=0')
    if tuple(lhs[1].shape) != scale_shapes[0] or tuple(rhs[1].shape) != scale_shapes[1]:
        raise ValueError(f'FP8 scale shapes must be {scale_shapes}')
    if not rhs[1].is_contiguous() or (channelwise and not lhs[1].is_contiguous()):
        raise ValueError('weight and channelwise activation scales must be contiguous')
    if not isinstance(out, torch.Tensor) or out.device != comm.device or out.dtype != torch.bfloat16:
        raise ValueError('out must be a BF16 tensor on the context device')
    if tuple(out.shape) != (m // comm.world, n) or not out.is_contiguous():
        raise ValueError('out must be a contiguous [M/TP,N] tensor')
    config = comm._get_config(channelwise, configs)
    quantization = 'channelwise' if channelwise else 'blockwise'
    launch = getattr(cpp, f'_gemm_comm_fp8_{quantization}_gemm_rs')
    launch(lhs, rhs, out, comm._state, m, comm.rank, comm.world, config)
