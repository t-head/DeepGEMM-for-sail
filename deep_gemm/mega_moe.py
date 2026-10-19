"""MegaMoE symmetric-memory ownership and process-group lifecycle.

Compute APIs and launch configuration are provided by deep_gemm_cpp.
"""
from typing import Dict, Optional, NamedTuple, Union
import torch
from . import deep_gemm_cpp
from .jit_kernels.utils import is_ppu1v5_device

_DEFAULT_GEMM1_BLOCK_N = 256
_DEFAULT_GEMM1_BLOCK_K = 128


def _compute_fp4_layout(config):
    assert is_ppu1v5_device()
    return deep_gemm_cpp._ppu_mega_moe_layout(config, True)


def _compute_fp8_layout(config):
    return deep_gemm_cpp._ppu_mega_moe_layout(config, False)

class MegaMoEGemm2Metadata(NamedTuple):
    m_rows: torch.Tensor
    valid_m_rows: torch.Tensor
    m_offsets: torch.Tensor
    src_m_rows: torch.Tensor
    src_m_offsets: torch.Tensor
    packed_src_info: torch.Tensor


def _gemm2_metadata_views(buffer):
    experts = int(buffer._cfg['NUM_EXPERTS_PER_RANK'])
    rows = buffer._slice(buffer._layout['off_gemm2_meta'], (5, experts), torch.int32).unbind()
    packed = buffer._slice(buffer._layout['off_gemm2_packed'], (buffer.num_pool,), torch.int64)
    return MegaMoEGemm2Metadata(*rows, packed)


class MegaMoEDispatchGemm1SymmBuffer:
    """Symmetric buffer object for the fused dispatch+gemm1 kernel.

    Allocates ONE symmetric buffer big enough for the workspace, the cross-rank
    token quad and the dispatch landing pools. A BF16 gemm1 output is allocated
    lazily only for callers that do not request fused MXFP4 output.
    Users write only the token quad via the named views below; weights and the
    per-SM gemm1 SFA scratch are passed / allocated separately by the launcher.

    Write views (host->device input):
        input_token        uint8   [NMTPR, TOKEN_BYTES]
        input_sf           uint8   [NMTPR, SF_BYTES]
        input_topk_idx     int64   [NMTPR, NUM_TOPK]
        input_topk_weights float32 [NMTPR, NUM_TOPK]
    Read-back views (device->host output / verification):
        expert_recv_count  int64   [NUM_EXPERTS]
        token_src_metadata int32   [num_pool, 3]
        gemm1_output       bf16    [num_pool, GEMM1_N/2] for fused SwiGLU,
                                    [num_pool, GEMM1_N] otherwise; local tensor
    """

    def __init__(self, group, config: Dict[str, int],
                 gemm1_block_n: int, gemm1_block_k: int,
                 config_override: Optional[Dict[str, int]] = None, *,
                 bf16_hidden: Optional[int] = None):
        import torch.distributed._symmetric_memory as symm_mem

        self.group = group
        # Capacity config is derived once from NMTPR.  It owns only static
        # shapes and the initial/pre-fill launch variant; per-call launch
        # configs are cached separately below and never resize this buffer.
        self._cfg = dict(config)
        self._capacity_cfg = dict(self._cfg)
        self._gemm1_block_n = int(gemm1_block_n)
        self._sfb_layout_block_n = int(gemm1_block_n)
        self._gemm1_block_k = int(gemm1_block_k)
        self._config_override = dict(config_override or {})
        self._launch_cfg_cache = {
            int(self._cfg['NMTPR']): dict(self._cfg),
        }
        self._layout = _compute_fp4_layout(self._cfg)
        self.num_ranks = int(self._cfg['NUM_RANKS'])
        # Public semantic shape attributes. The raw
        # offset table (`_layout`) and compile-time constant dict (`_cfg`) stay
        # private -- users never touch offsets; runtime sizes come from the view
        # shapes below.
        self.num_experts = int(self._cfg['NUM_EXPERTS'])
        self.num_max_tokens_per_rank = int(self._cfg['NMTPR'])
        self.num_topk = int(self._cfg['NUM_TOPK'])
        self.hidden = int(self._cfg['TOKEN_BYTES']) * 2 if bf16_hidden is None else int(bf16_hidden)
        if bf16_hidden is not None:
            self.bf16_hidden = self.hidden
        self.intermediate_hidden = int(self._cfg['GEMM1_N'])
        self.num_pool = int(self._layout['num_pool'])

        total = int(self._layout['total'])
        self._buffer = symm_mem.empty(total, dtype=torch.uint8, device='cuda')
        self._buffer.zero_()
        self._handle = symm_mem.rendezvous(self._buffer, group=group)
        group.barrier()
        torch.cuda.synchronize()

        ptrs = [self._handle.get_buffer(r, (total, ), torch.uint8, storage_offset=0).data_ptr()
                for r in range(self.num_ranks)]
        self._buf_ptrs = torch.tensor(ptrs, dtype=torch.int64, device='cpu')
        # Allocate BF16 output only when a non-quantized caller asks for it.
        # Production fused-quant writes FP4 + scales directly and never needs
        # the multi-GiB [num_pool, GEMM1_N/2] intermediate.
        self._gemm1_output = None
        # SFA scratch holds the packed scale rows for all possible runtime M
        # blocks.  num_pool is divisible by every candidate BLOCK_M, so the
        # worst case is exactly num_pool rows regardless of the selected tile.
        self._sfa_scratch_numel = (
            self.num_pool * (int(self._cfg['SF_BYTES']) // 2)
        )
        self._sfa_scratch = torch.empty(
            self._sfa_scratch_numel, dtype=torch.uint16, device='cuda')

    # ---- internal: zero-copy typed slice of the symmetric buffer ----
    def _slice(self, off: int, shape, dtype) -> torch.Tensor:
        nelem = 1
        for s in shape:
            nelem *= int(s)
        nbytes = nelem * torch.empty((), dtype=dtype).element_size()
        chunk = self._buffer[int(off): int(off) + nbytes]
        return chunk.view(dtype).view(*shape)

    @property
    def buffer(self) -> torch.Tensor:
        return self._buffer

    @property
    def buf_ptrs(self) -> torch.Tensor:
        return self._buf_ptrs

    # ---- write views (token quad) ----
    @property
    def input_token(self) -> torch.Tensor:
        c = self._cfg
        return self._slice(self._layout['off_it'], (c['NMTPR'], c['TOKEN_BYTES']), torch.uint8)

    @property
    def input_token_bf16(self) -> torch.Tensor:
        assert hasattr(self, 'bf16_hidden'), \
            'input_token_bf16 is only valid for a BF16 dispatch buffer'
        return self.input_token.view(torch.bfloat16).view(
            self._cfg['NMTPR'], self.bf16_hidden)

    @property
    def input_sf(self) -> torch.Tensor:
        c = self._cfg
        return self._slice(self._layout['off_isf'], (c['NMTPR'], c['SF_BYTES']), torch.uint8)

    @property
    def input_topk_idx(self) -> torch.Tensor:
        c = self._cfg
        return self._slice(self._layout['off_iti'], (c['NMTPR'], c['NUM_TOPK']), torch.int64)

    @property
    def input_topk_weights(self) -> torch.Tensor:
        c = self._cfg
        return self._slice(self._layout['off_itw'], (c['NMTPR'], c['NUM_TOPK']), torch.float32)

    # ---- read-back views ----
    @property
    def expert_recv_count(self) -> torch.Tensor:
        off = self._layout['off_ws'] + self._layout['woff_rc']
        return self._slice(off, (self._cfg['NUM_EXPERTS'], ), torch.int64)

    @property
    def token_src_metadata(self) -> torch.Tensor:
        off = self._layout['off_ws'] + self._layout['woff_meta']
        return self._slice(off, (self._layout['num_pool'], 3), torch.int32)

    @property
    def recv_count_sum(self) -> torch.Tensor:
        off = self._layout['off_ws'] + self._layout['woff_rcs']
        return self._slice(
            off, (self._cfg['NUM_EXPERTS_PER_RANK'],), torch.int64)

    @property
    def dispatch_arrival_count(self) -> torch.Tensor:
        off = self._layout['off_ws'] + self._layout['woff_arr']
        return self._slice(
            off, (self._layout['num_pool_blocks'],), torch.uint32)

    @property
    def dispatched_token_bf16(self) -> torch.Tensor:
        assert hasattr(self, 'bf16_hidden'), \
            'dispatched_token_bf16 is only valid for a BF16 dispatch buffer'
        return self._slice(
            self._layout['off_pt'],
            (self.num_pool, self.bf16_hidden),
            torch.bfloat16,
        )

    @property
    def dispatched_topk_weights(self) -> torch.Tensor:
        return self._slice(
            self._layout['off_pw'], (self.num_pool,), torch.float32)

    @property
    def gemm1_output(self) -> torch.Tensor:
        # Backward-compatible lazy view for non-quantized/debug callers.
        return self.get_gemm1_output(fused_swiglu=True)

    def get_gemm1_output(self, fused_swiglu: bool) -> torch.Tensor:
        output_n = self._cfg['GEMM1_N'] // 2 if fused_swiglu else self._cfg['GEMM1_N']
        expected_shape = (self._layout['num_pool'], output_n)
        if self._gemm1_output is None or self._gemm1_output.shape != expected_shape:
            self._gemm1_output = torch.empty(
                expected_shape, dtype=torch.bfloat16, device='cuda')
        return self._gemm1_output

    def get_gemm2_metadata(self) -> MegaMoEGemm2Metadata:
        """Persistent views populated by dispatch/GEMM1 on every invocation.

        W4A4 selects GEMM2 padding internally, with ``gemm2_block_m`` available
        for explicit tile overrides.
        Other paths retain the dispatch layout. Views are ready on the same
        stream after GEMM1, including Graph replay and zero-token invocations.
        """
        return _gemm2_metadata_views(self)

    def get_sfa_scratch(self, numel: int) -> torch.Tensor:
        assert int(numel) <= self._sfa_scratch_numel, \
            f'SFA scratch request {numel} exceeds capacity {self._sfa_scratch_numel}'
        return self._sfa_scratch[:int(numel)]

    def destroy(self) -> None:
        self._handle = None
        self._buffer = None
        self._buf_ptrs = None
        self._gemm1_output = None
        self._sfa_scratch = None
        self._sfa_scratch_numel = 0
        self.group = None

class Fp8MegaMoEDispatchSymmBuffer:
    """One symmetric arena for FP8 routing and expert-major landing pools."""

    def __init__(self, group, config: Dict[str, int]):
        import torch.distributed._symmetric_memory as symm_mem

        self.group = group
        self._cfg = dict(config)
        self._layout = _compute_fp8_layout(self._cfg)
        self.num_ranks = int(config["NUM_RANKS"])
        self.num_experts = int(config["NUM_EXPERTS"])
        self.num_topk = int(config["NUM_TOPK"])
        self.hidden = int(config["TOKEN_BYTES"])
        self.num_scales = int(config["SF_BYTES"]) // 4
        self.num_max_tokens_per_rank = int(config["NMTPR"])
        self.block_m = int(config["BLOCK_M"])
        self.num_pool = int(self._layout["num_pool"])

        self._buffer = symm_mem.empty(
            int(self._layout["total"]), dtype=torch.uint8, device="cuda")
        self._buffer.zero_()
        self._handle = symm_mem.rendezvous(self._buffer, group=group)
        group.barrier()
        torch.cuda.synchronize()
        ptrs = [
            self._handle.get_buffer(
                rank, (int(self._layout["total"]), ), torch.uint8,
                storage_offset=0).data_ptr()
            for rank in range(self.num_ranks)
        ]
        self._buf_ptrs = torch.tensor(ptrs, dtype=torch.int64, device="cpu")

    def _slice(self, off: int, shape, dtype: torch.dtype) -> torch.Tensor:
        numel = 1
        for extent in shape:
            numel *= int(extent)
        nbytes = numel * torch.empty((), dtype=dtype).element_size()
        return self._buffer[int(off): int(off) + nbytes].view(dtype).view(*shape)

    @property
    def buffer(self) -> torch.Tensor:
        return self._buffer

    @property
    def buf_ptrs(self) -> torch.Tensor:
        return self._buf_ptrs

    @property
    def input_token(self) -> torch.Tensor:
        return self._slice(
            self._layout["off_it"],
            (self.num_max_tokens_per_rank, self.hidden), torch.float8_e4m3fn)

    @property
    def input_scale(self) -> torch.Tensor:
        return self._slice(
            self._layout["off_isf"],
            (self.num_max_tokens_per_rank, self.num_scales), torch.float32)

    @property
    def input_topk_idx(self) -> torch.Tensor:
        return self._slice(
            self._layout["off_iti"],
            (self.num_max_tokens_per_rank, self.num_topk), torch.int64)

    @property
    def input_topk_weights(self) -> torch.Tensor:
        return self._slice(
            self._layout["off_itw"],
            (self.num_max_tokens_per_rank, self.num_topk), torch.float32)

    @property
    def pool_token(self) -> torch.Tensor:
        return self._slice(
            self._layout["off_pt"], (self.num_pool, self.hidden),
            torch.float8_e4m3fn)

    @property
    def pool_scale(self) -> torch.Tensor:
        # Block scales land directly in the column-major layout used by the
        # blockwise grouped GEMM, avoiding a pool-sized transpose per layer.
        return self._slice(
            self._layout["off_psf"],
            (self.num_scales, self.num_pool), torch.float32).t()

    @property
    def pool_topk_weights(self) -> torch.Tensor:
        return self._slice(
            self._layout["off_pw"], (self.num_pool, ), torch.float32)

    @property
    def expert_recv_count(self) -> torch.Tensor:
        off = self._layout["off_ws"] + self._layout["woff_rc"]
        return self._slice(off, (self.num_experts, ), torch.int64)

    @property
    def recv_count_sum(self) -> torch.Tensor:
        off = self._layout["off_ws"] + self._layout["woff_rcs"]
        return self._slice(
            off, (self._cfg["NUM_EXPERTS_PER_RANK"], ), torch.int64)

    @property
    def token_src_metadata(self) -> torch.Tensor:
        off = self._layout["off_ws"] + self._layout["woff_meta"]
        return self._slice(off, (self.num_pool, 3), torch.int32)

    def get_gemm2_metadata(self) -> MegaMoEGemm2Metadata:
        """Persistent views populated by dispatch/GEMM1 on every invocation.

        FP8 GEMM2 retains the dispatch row layout. Views are ready on the same
        stream after GEMM1, including Graph replay and zero-token invocations.
        """
        return _gemm2_metadata_views(self)

    def destroy(self) -> None:
        self._handle = None
        self._buffer = None
        self._buf_ptrs = None
        self.group = None

def get_symm_buffer_for_mega_moe(
        group, num_experts: int, num_max_tokens_per_rank: int, num_topk: int, hidden: int, *,
        precision: str, intermediate_hidden: Optional[int] = None,
        block_m: Optional[int] = None,
        gemm1_block_n: Optional[int] = None, gemm1_block_k: Optional[int] = None,
        config_override: Optional[Dict[str, int]] = None,
) -> Union[MegaMoEDispatchGemm1SymmBuffer, Fp8MegaMoEDispatchSymmBuffer]:
    """Allocate one reusable MegaMoE symmetric workspace on every rank.

    ``precision`` selects w4a4, w4a16, fp8_channelwise or fp8_blockwise.
    W4 backends require ``intermediate_hidden`` including gate and up
    (``2 * intermediate``); FP8 transport buffers do not take this argument.
    ``hidden`` is always the logical input width.

    W4A4 accepts GEMM1 N/K tile limits and config_override; W4A16 and FP8
    accept block_m. Unsupported combinations fail before any collective
    allocation. All ranks must call this function with matching dimensions.
    """
    ranks = int(group.size())
    if not (0 < ranks <= 32 and num_experts > 0 and num_experts % ranks == 0
            and 0 < num_topk <= min(16, num_experts) and 0 < num_max_tokens_per_rank < (1 << 24)):
        raise ValueError("invalid MegaMoE ranks, expert count, top-k or token capacity")
    if precision == "fp8_blockwise" and ranks not in (1, 2, 4, 8):
        raise ValueError("blockwise MegaMoE currently targets intra-node EP1/2/4/8")
    config = deep_gemm_cpp._mega_moe_dispatch_config(
        num_ranks=ranks, num_experts=num_experts, num_topk=num_topk, hidden=hidden,
        num_max_tokens_per_rank=num_max_tokens_per_rank, precision=precision,
        intermediate_hidden=intermediate_hidden, block_m=block_m,
        gemm1_block_n=gemm1_block_n, gemm1_block_k=gemm1_block_k, overrides=config_override)
    if not is_ppu1v5_device():
        raise RuntimeError("MegaMoE requires PPU1.5")

    import torch.distributed._symmetric_memory as symm_mem
    try:
        symm_mem.enable_symm_mem_for_group(group.group_name)
    except Exception:
        pass

    if precision in ("fp8_channelwise", "fp8_blockwise"):
        return Fp8MegaMoEDispatchSymmBuffer(group, config)
    bf16 = precision == "w4a16"
    if bf16:
        config_override = {"BLOCK_M": block_m} if block_m is not None else None
    return MegaMoEDispatchGemm1SymmBuffer(
        group, config,
        gemm1_block_n=_DEFAULT_GEMM1_BLOCK_N if gemm1_block_n is None else gemm1_block_n,
        gemm1_block_k=_DEFAULT_GEMM1_BLOCK_K if gemm1_block_k is None else gemm1_block_k,
        config_override=config_override,
        bf16_hidden=hidden if bf16 else None)
