"""GPU-resident MoE schedules with a bounded activation ring and graph replay.

The caller owns the plan and parameters. A plan may be shared by sequential
layers/forwards on one stream: backward reconstructs every chunk instead of
retaining a layer's expanded expert activations in the plan.
"""

import math
import os
import threading
from contextlib import contextmanager

import torch
import torch.distributed as dist

import deep_gemm as dg


def _validate_accumulators(accumulators, w1, w2):
    if accumulators is None:
        return
    if len(accumulators) != 2:
        raise ValueError("weight_grad_accumulators must contain dW1 and dW2")
    for accumulator, weight in zip(accumulators, (w1, w2)):
        if (
            accumulator.shape != weight.shape
            or accumulator.device != weight.device
            or accumulator.dtype != torch.float32
            or not accumulator.is_contiguous()
            or accumulator.requires_grad
        ):
            raise ValueError(
                "weight-gradient accumulators must be contiguous FP32 buffers matching the weights"
            )

    def overlaps(a, b):
        if not a.numel() or not b.numel():
            return False

        # Accumulators are contiguous; the span also conservatively covers a
        # strided reference weight without reading any device values.
        def interval(tensor):
            span = 1 + sum(
                (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
            )
            start = tensor.data_ptr()
            return start, start + span * tensor.element_size()

        a_begin, a_end = interval(a)
        b_begin, b_end = interval(b)
        return a_begin < b_end and b_begin < a_end

    if overlaps(accumulators[0], accumulators[1]) or any(
        overlaps(accumulator, weight) for accumulator in accumulators for weight in (w1, w2)
    ):
        raise ValueError(
            "weight-gradient accumulators must not overlap each other or expert weights"
        )


def to_g8_weight(weight):
    """Convert [..., 2*I, H] split gate/up rows to [gate8, up8, ...]."""
    if weight.ndim < 2 or weight.shape[-2] % 16:
        raise ValueError("G8 weights require a 2*I row dimension divisible by16")
    shape = tuple(weight.shape)
    return (
        weight.reshape(*shape[:-2], 2, shape[-2] // 16, 8, shape[-1])
        .transpose(-4, -3)
        .contiguous()
        .view(shape)
    )


def from_g8_weight(weight):
    """Convert G8 weights or their matching gradients to split checkpoint rows."""
    if weight.ndim < 2 or weight.shape[-2] % 16:
        raise ValueError("G8 weights require a 2*I row dimension divisible by16")
    shape = tuple(weight.shape)
    return (
        weight.reshape(*shape[:-2], shape[-2] // 16, 2, 8, shape[-1])
        .transpose(-4, -3)
        .contiguous()
        .view(shape)
    )


def allocate_route_windows(plan):
    def allocate(name, shape):
        tensor = torch.empty(shape, device=plan.device, dtype=torch.int32)
        setattr(plan, name, tensor)
        plan.buffers[name] = tensor

    allocate("global_offsets", (plan.e + 1,))
    allocate("global_lengths", (plan.e,))
    allocate("global_permutation", (plan.full_row_capacity,))
    allocate("global_source_rows", (plan.full_row_capacity,))
    if not hasattr(plan, "source_rows"):
        allocate("source_rows", (plan.rows,))
    if not hasattr(plan, "group_ids"):
        allocate("group_ids", (plan.rows,))
    plan.route_window_count = (plan.full_row_capacity + plan.rows - 1) // plan.rows


def set_route_window(plan, start, compile_only=False):
    dg._C._mega_moe_training_launch(
        14,
        [
            plan.global_offsets,
            plan.global_permutation,
            plan.offsets,
            plan.lengths,
            plan.permutation,
            plan.source_rows,
            plan.status,
            plan.group_ids,
        ],
        [plan.e, plan.full_row_capacity, plan.rows, start, plan.k, plan.size * plan.chunk_tokens],
        compile_only,
    )


def _launch(operation, tensors, dimensions):
    dg._C._mega_moe_training_common_launch(operation, tensors, dimensions)


class _MoEPlanBase:
    """Internal shared expert math and workspace for Peer/DeepEP plans.

    All ranks agree on dimensions, chunk size and compute recipe. Input token
    counts may differ up to max_tokens. Call a plan from one CUDA/PPU stream;
    create separate plans for concurrent callers. Optional internal weight
    gradients use a joined auxiliary stream. The activation ring holds one
    chunk, independent of max_tokens and the number of pending layers.
    With optional forward-context reuse, give each outstanding captured
    layer its own plan so a later forward cannot overwrite its context.
    """

    transport = None

    def __init__(
        self,
        hidden,
        intermediate,
        local_experts,
        topk,
        max_tokens,
        *,
        chunk_tokens=128,
        compute="bf16",
        group=None,
        device=None,
        check_errors=True,
        m_schedule="nopad",
        fuse_activation_quant=True,
        active_pointwise=True,
        vectorized_routes=True,
        fast_weight_transpose=True,
        packed_rows=None,
        block_m=128,
        gate_up_layout="split",
        wgrad_overlap="none",
        overlap_expert_dy=False,
        fp8_transpose_quantized_weights=None,
        packed_capacity=None,
    ):
        if hidden <= 0 or hidden % 256 or intermediate <= 0 or intermediate % 128:
            raise ValueError("native training requires positive H%256=0 and I%128=0")
        if min(local_experts, topk, chunk_tokens) < 1 or not 0 <= max_tokens <= 2**31 - 1:
            raise ValueError("invalid expert, top-k or token capacity")
        if compute not in ("bf16", "fp8"):
            raise ValueError("compute must be 'bf16' or 'fp8'")
        if wgrad_overlap not in ("none", "dw1", "both"):
            raise ValueError("wgrad_overlap must be none, dw1 or both")
        if wgrad_overlap != "none" and (
            compute != "bf16" or packed_rows is None or self.transport not in ("peer", "deepep")
        ):
            raise ValueError("wgrad overlap requires a bounded BF16 peer or DeepEP plan")
        if fp8_transpose_quantized_weights is None:
            raw = os.environ.get("DG_MOE_FP8_TRANSPOSE_QUANTIZED_WEIGHTS", "0")
            if raw not in ("0", "1"):
                raise ValueError("DG_MOE_FP8_TRANSPOSE_QUANTIZED_WEIGHTS must be0or1")
            fp8_transpose_quantized_weights = raw == "1"
        if type(fp8_transpose_quantized_weights) is not bool:
            raise TypeError("fp8_transpose_quantized_weights must be bool")
        if fp8_transpose_quantized_weights and (
            compute != "fp8" or getattr(self, "fp8_async_weight_quantization", False)
        ):
            raise ValueError(
                "quantized weight transpose requires synchronous FP8 weight preparation"
            )
        self.fp8_transpose_quantized_weights = fp8_transpose_quantized_weights
        dw_raw = os.environ.get("DG_MOE_FP8_WGRAD", "0")
        dw_pipe = os.environ.get("DG_MOE_FP8_WGRAD_PREP_PIPELINE", "0")
        if dw_raw not in ("0", "1") or dw_pipe not in ("0", "1"):
            raise ValueError("FP8 DW flags must be0or1")
        self.fp8_wgrad = dw_raw == "1"
        self.fp8_wgrad_pipeline = dw_pipe == "1"
        fused_dw = os.environ.get("DG_MOE_FP8_WGRAD_FUSED_PRODUCER", "0")
        if fused_dw not in ("0", "1"):
            raise ValueError("FP8 DW fused producer flag must be0or1")
        self.fp8_wgrad_fused_producer = fused_dw == "1"
        if self.fp8_wgrad_fused_producer and (
            not self.fp8_wgrad
            or self.transport != "peer"
            or not getattr(self, "fp8_expert_gradient_batch", 0)
            or self.fp8_wgrad_pipeline
        ):
            raise ValueError(
                "Fused FP8 DW requires peer expert batches without the separate preparation stream"
            )
        self.fp8_wgrad_config = tuple(
            int(v)
            for v in os.environ.get("DG_MOE_FP8_WGRAD_CONFIG", "256,128,128,64,32,3").split(",")
        )
        self.fp8_wgrad_pack_ctas = int(os.environ.get("DG_MOE_FP8_WGRAD_PACK_CTAS", "4"))
        self.fp8_wgrad_block_k = int(os.environ.get("DG_MOE_FP8_WGRAD_BLOCK_K", "128"))
        if self.fp8_wgrad_block_k not in (128, 512):
            raise ValueError("FP8 DW scale K must be128or512")
        native_dw = os.environ.get("DG_MOE_FP8_WGRAD_NATIVE_BLOCKWISE", "0")
        if native_dw not in ("0", "1"):
            raise ValueError("Native FP8 DW flag must be0or1")
        self.fp8_wgrad_native_blockwise = native_dw == "1"
        self.fp8_dw_scale_m = int(os.environ.get("DG_MOE_FP8_DW_SCALE_M", "1"))
        if self.fp8_dw_scale_m not in (1, 64) or (
            self.fp8_dw_scale_m != 1 and not self.fp8_wgrad_native_blockwise
        ):
            raise ValueError(
                "DW compact scale granularity must1(default)or64 with native blockwise DW"
            )
        self.fp8_dw_metadata_mode = int(os.environ.get("DG_MOE_FP8_DW_METADATA_MODE", "0"))
        if self.fp8_dw_metadata_mode not in (0, 1, 2) or (
            not self.fp8_wgrad_native_blockwise and self.fp8_dw_metadata_mode
        ):
            raise ValueError("DW metadata mode requires native blockwise DW and0..2")
        self.fp8_dw_swizzle_m = int(os.environ.get("DG_MOE_FP8_DW_SWIZZLE_M", "2"))
        if self.fp8_dw_swizzle_m not in (1, 2, 4, 8, 16, 32, 64) or (
            not self.fp8_wgrad_native_blockwise and self.fp8_dw_swizzle_m != 2
        ):
            raise ValueError("Native DW traversal requires a power-of-two M tile group1..64")
        if self.fp8_wgrad_native_blockwise and (
            not self.fp8_wgrad
            or self.fp8_wgrad_block_k != 128
            or len(self.fp8_wgrad_config) != 6
            or self.fp8_wgrad_config[2] not in (128, 256)
        ):
            raise ValueError(
                "Native blockwise DW requires enabled FP8 DW with128K scales and128K/256K tiles"
            )
        if len(self.fp8_wgrad_config) != 6 or not 1 <= self.fp8_wgrad_pack_ctas <= 16:
            raise ValueError("FP8 DW requires six geometry values and1..16packing CTAs/CU")
        if self.fp8_wgrad and (compute != "fp8" or packed_rows is None or wgrad_overlap != "none"):
            raise ValueError(
                "FP8 DW requires bounded FP8 peer/DeepEP and serial weight-gradient GEMMs"
            )
        if self.fp8_wgrad_pipeline and (
            not self.fp8_wgrad
            or self.transport != "peer"
            or not getattr(self, "fp8_expert_gradient_batch", 0)
        ):
            raise ValueError("FP8 DW preparation pipeline requires FP8 peer expert batches")
        dw1_pipe = os.environ.get("DG_MOE_FP8_WGRAD_DW1_PIPELINE", "0")
        if dw1_pipe not in ("0", "1"):
            raise ValueError("FP8 DW1 preparation pipeline flag must be0or1")
        self.fp8_wgrad_dw1_pipeline = dw1_pipe == "1"
        if self.fp8_wgrad_dw1_pipeline and (
            not self.fp8_wgrad
            or self.transport != "peer"
            or not getattr(self, "fp8_expert_gradient_batch", 0)
        ):
            raise ValueError(
                "FP8 DW1 preparation pipeline requires bounded FP8 peer expert batches"
            )
        row_reuse = os.environ.get("DG_MOE_FP8_WGRAD_REUSE_ROW_SCALES", "0")
        if row_reuse not in ("0", "1"):
            raise ValueError("DW row-scale reuse flag must be0or1")
        self.fp8_wgrad_reuse_row_scales = row_reuse == "1"
        if self.fp8_wgrad_reuse_row_scales and (
            not self.fp8_wgrad
            or self.transport != "peer"
            or not getattr(self, "fp8_expert_gradient_batch", 0)
        ):
            raise ValueError("DW row-scale reuse requires bounded FP8 peer expert batches")

        vector_activation = os.environ.get("DG_MOE_FP8_VECTOR_ACTIVATION", "0")
        if vector_activation not in ("0", "1", "2", "4"):
            raise ValueError("FP8 vector activation must be0,1,2or4")
        self.fp8_vector_activation = int(vector_activation)
        self._fp8_dw_ready = None
        self._fp8_dw_ready_on_main = False
        fused_dhidden = os.environ.get("DG_MOE_FP8_FUSE_DHIDDEN_ACTIVATION", "0")
        if fused_dhidden not in ("0", "1"):
            raise ValueError("DG_MOE_FP8_FUSE_DHIDDEN_ACTIVATION must be0or1")
        self.fp8_fuse_dhidden_activation = fused_dhidden == "1"
        if self.fp8_fuse_dhidden_activation and (
            compute != "fp8" or not fuse_activation_quant or m_schedule != "nopad"
        ):
            raise ValueError(
                "fused dHidden/activation requires FP8 no-pad GEMM and activation quantization"
            )
        fused_prep = os.environ.get("DG_MOE_FP8_FUSED_WEIGHT_PREPARE", "0")
        if fused_prep not in ("0", "1"):
            raise ValueError("DG_MOE_FP8_FUSED_WEIGHT_PREPARE must be0or1")
        self.fp8_fused_weight_prepare = fused_prep == "1"
        self.fp8_weight_prep_ctas = int(os.environ.get("DG_MOE_FP8_WEIGHT_PREP_CTAS_PER_CU", "4"))
        if not 0 < self.fp8_weight_prep_ctas <= 16:
            raise ValueError("FP8 weight preparation CTA budget must be1..16")
        if self.fp8_fused_weight_prepare and not fp8_transpose_quantized_weights:
            raise ValueError("fused FP8 weight preparation requires quantized weight transpose")
        raw_swizzle = os.environ.get("DG_MOE_FP8_WEIGHT_TRANSPOSE_SWIZZLE", "0")
        if raw_swizzle not in ("0", "1"):
            raise ValueError("DG_MOE_FP8_WEIGHT_TRANSPOSE_SWIZZLE must be0or1")
        self.fp8_transpose_swizzle = raw_swizzle == "1"
        if self.fp8_transpose_swizzle and not fp8_transpose_quantized_weights:
            raise ValueError("FP8 transpose swizzle requires quantized weight transpose")
        row_quant = os.environ.get("DG_MOE_FP8_VECTOR_ROW_QUANT", "0")
        if row_quant not in ("0", "1"):
            raise ValueError("DG_MOE_FP8_VECTOR_ROW_QUANT must be0or1")
        self.fp8_vector_row_quant = row_quant == "1"
        if self.fp8_vector_row_quant and compute != "fp8":
            raise ValueError("vectorized FP8 row quantization requires FP8 compute")
        self.wgrad_overlap = wgrad_overlap
        if not isinstance(overlap_expert_dy, bool):
            raise TypeError("overlap_expert_dy must be bool")
        if overlap_expert_dy and (
            (compute != "bf16" and self.transport != "deepep")
            or packed_rows is None
            or self.transport not in ("peer", "deepep")
        ):
            raise ValueError("DY overlap requires a bounded peer/DeepEP plan; peer requires BF16")
        self.overlap_expert_dy = overlap_expert_dy
        if (
            type(block_m) is not int
            or block_m not in (128, 256)
            or (block_m != 128 and compute != "bf16")
        ):
            raise ValueError("block_m must be128, or256 for BF16")
        self.block_m = block_m
        if gate_up_layout not in ("split", "g8") or (gate_up_layout == "g8" and compute != "bf16"):
            raise ValueError("gate_up_layout must be split, or g8 with BF16 compute")
        self.gate_up_layout = gate_up_layout
        self.g8_interleaved = gate_up_layout == "g8"
        if m_schedule not in ("nopad", "contiguous"):
            raise ValueError("m_schedule must be 'nopad' or 'contiguous'")
        if not isinstance(fuse_activation_quant, bool):
            raise TypeError("fuse_activation_quant must be bool")
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if device is None
            else torch.device(device)
        )
        if self.device.type != "cuda":
            raise ValueError("MoE plan requires a PPU/CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if group is not None and not dist.is_initialized():
            raise RuntimeError("initialize distributed before creating an EP plan")
        self.group = group
        self.size = dist.get_world_size(group) if group is not None else 1
        self.rank = dist.get_rank(group) if group is not None else 0
        self.h, self.i, self.e, self.k = hidden, intermediate, local_experts, topk
        self.max_tokens, self.chunk_tokens = max_tokens, chunk_tokens
        self.compute, self.check_errors = compute, check_errors
        self.m_schedule = m_schedule
        self.fuse_activation_quant = fuse_activation_quant
        if not isinstance(active_pointwise, bool):
            raise TypeError("active_pointwise must be bool")
        self.active_pointwise = False
        self.vectorized_routes = False
        if not isinstance(fast_weight_transpose, bool):
            raise TypeError("fast_weight_transpose must be bool")
        self.fast_weight_transpose = fast_weight_transpose
        self.alignment = math.lcm(block_m, dg.get_m_alignment_for_contiguous_layout())
        self.routes = self.size * chunk_tokens * topk
        # Sum ceil(count_e/alignment)*alignment <= routes + E*(alignment-1).
        self.full_row_capacity = (
            math.ceil((self.routes + local_experts * (self.alignment - 1)) / self.alignment)
            * self.alignment
        )
        self.unbounded_row_capacity = self.full_row_capacity
        if packed_capacity is not None:
            if (
                self.transport != "deepep"
                or getattr(self, "deepep_layout", None) != "compact"
                or packed_rows is None
                or type(packed_capacity) is not int
                or packed_capacity <= 0
                or packed_capacity % self.alignment
                or packed_capacity > self.full_row_capacity
            ):
                raise ValueError("compact capacity requires an aligned bounded DeepEP workspace")
            self.full_row_capacity = packed_capacity
        if packed_rows is not None:
            if (
                type(packed_rows) is not int
                or packed_rows <= 0
                or packed_rows % self.alignment
                or self.transport not in ("peer", "deepep")
                or (
                    compute != "bf16"
                    and self.transport != "deepep"
                    and not getattr(self, "fp8_peer_compute", False)
                )
            ):
                raise ValueError(
                    "packed-row windows require an aligned supported peer/DeepEP workspace"
                )
        self.rows = (
            self.full_row_capacity
            if packed_rows is None
            else min(packed_rows, self.full_row_capacity)
        )
        if self.full_row_capacity > 2**31 - 1 or self.routes > 2**31 - 1:
            raise ValueError("native route capacity exceeds int32 metadata")
        # Reused Tensor Cell mainloops contain 32-bit element offsets. Bound
        # weights and ring matrices before allocation, including skew capacity.
        if (
            self.rows * max(hidden, 2 * intermediate) > 2**31 - 1
            or local_experts * 2 * intermediate * hidden > 2**31 - 1
        ):
            raise ValueError("native tensor capacity exceeds int32 element indexing")
        if group is not None:
            recipe = compute if fuse_activation_quant else compute + ":separate_activation_quant"
            if self.active_pointwise:
                recipe += ":active_pointwise"
            if self.vectorized_routes:
                recipe += ":vectorized_routes"
            description = (
                hidden,
                intermediate,
                local_experts,
                topk,
                max_tokens,
                chunk_tokens,
                recipe,
                self.transport,
                m_schedule,
                self.rows,
                block_m,
                gate_up_layout,
                wgrad_overlap,
                overlap_expert_dy,
                fp8_transpose_quantized_weights,
                self.fp8_vector_row_quant,
                self.fp8_transpose_swizzle,
                self.fp8_vector_activation,
                self.fp8_fused_weight_prepare,
                self.fp8_weight_prep_ctas,
                self.fp8_fuse_dhidden_activation,
                self.fp8_wgrad,
                self.fp8_wgrad_pipeline,
                self.fp8_wgrad_config,
                self.fp8_wgrad_pack_ctas,
                self.fp8_wgrad_block_k,
                self.fp8_wgrad_fused_producer,
                self.fp8_wgrad_native_blockwise,
                self.fp8_dw_scale_m,
                self.fp8_wgrad_dw1_pipeline,
                self.fp8_wgrad_reuse_row_scales,
            )
            descriptions = [None] * self.size
            dist.all_gather_object(descriptions, description, group=group)
            if any(other != description for other in descriptions):
                raise ValueError("EP ranks must create matching MoE plan configurations")
        self._lock = threading.RLock()
        self._stream = None
        self._prepared = False
        self.buffers = {}
        self._wgrad_stream = (
            torch.cuda.Stream(device=self.device) if wgrad_overlap != "none" else None
        )
        self._wgrad_pending = False
        self._gradient_stream = torch.cuda.Stream(device=self.device) if overlap_expert_dy else None
        self._gradient_prefetch_pending = False

        def allocate(name, shape, dtype=torch.bfloat16):
            unused = {"gathered_x", "gathered_ids", "route_values", "local_values", "local_dy"}
            if self.transport == "peer" and not getattr(self, "fp8_peer_compute", False):
                unused.add("group_ids")
            if name in unused:
                # These reference-transport matrices are replaced/deleted
                # by both transports before use. Avoid a full worst-skew
                # activation allocation while constructing a bounded ring.
                shape = (0,)
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            self.buffers[name] = tensor
            return tensor

        c, r, m, h, i, e, k = (
            chunk_tokens,
            self.routes,
            self.rows,
            hidden,
            intermediate,
            local_experts,
            topk,
        )
        self.local_x = allocate("local_x", (c, h))
        self.local_ids = allocate("local_ids", (c, k), torch.int64)
        self.gathered_x = allocate("gathered_x", (self.size * c, h))
        self.gathered_ids = allocate("gathered_ids", (self.size * c, k), torch.int64)
        self.counts = allocate("counts", (e,), torch.int32)
        self.offsets = allocate("offsets", (e + 1,), torch.int32)
        self.current_rows = self.offsets[e : e + 1]
        self.lengths = allocate("lengths", (e,), torch.int32)
        self.group_ids = allocate("group_ids", (m,), torch.int32)
        self.permutation = allocate("permutation", (m,), torch.int32)
        self.route_experts = allocate("route_experts", (r,), torch.int32)
        self.route_rows = allocate("route_rows", (r,), torch.int32)
        self.status = allocate("status", (1,), torch.int32).zero_()
        self._forward_context_generation_counter = 0
        self._forward_context_capture_key = 0
        self._forward_context_generation = None
        if getattr(self, "reuse_forward_context", False):
            self._forward_context_generation = allocate(
                "forward_context_generation", (1,), torch.int64
            ).zero_()
        if self.fp8_fuse_dhidden_activation:
            self._fp8_activation_destinations = allocate(
                "fp8_local_activation_dummy", (1,), torch.int64
            ).zero_()
        self.packed_x = allocate("packed_x", (m, h))
        self.gate_up = allocate("gate_up", (m, 2 * i))
        self.hidden = allocate("hidden", (m, i))
        self.packed_output = allocate("packed_output", (m, h))
        self.packed_dy = allocate("packed_dy", (m, h))
        self.route_values = allocate("route_values", (r, h))
        self.local_values = allocate("local_values", (c * k, h))
        self.local_dy = allocate("local_dy", (c * k, h))
        self.w1_t = allocate("w1_t", (e, h, 2 * i))
        self.w2_t = allocate("w2_t", (e, i, h))
        self.quantized = {}
        if compute == "fp8":
            for name, shape in (
                ("w1", (e, 2 * i, h)),
                ("w2", (e, h, i)),
                ("w1_t", (e, h, 2 * i)),
                ("w2_t", (e, i, h)),
            ):
                data = allocate(name + "_fp8", shape, torch.float8_e4m3fn)
                scales = allocate(
                    name + "_scale", (shape[0], shape[1] // 128, shape[2] // 128), torch.float32
                )
                self.quantized[name] = (data, scales)
            for width in {h, i, 2 * i}:
                data = allocate(f"activation_fp8_{width}", (m, width), torch.float8_e4m3fn)
                scales = allocate(f"activation_scale_{width}", (m, width // 128), torch.float32)
                self.quantized[width] = (data, scales)

        if self.fp8_wgrad:
            width = max(h, 2 * i)
            self._fp8_dw_left = allocate("fp8_dw_left", (m * width,), torch.float8_e4m3fn)
            self._fp8_dw_right = allocate("fp8_dw_right", (m * width,), torch.float8_e4m3fn)
            self._fp8_dw_sa = allocate("fp8_dw_sa", (m // 128 * width,), torch.float32)
            self._fp8_dw_sb = allocate("fp8_dw_sb", (width // 128 * (m // 128),), torch.float32)
            if self.fp8_wgrad_pipeline or self.fp8_wgrad_dw1_pipeline:
                self._fp8_dw_prep_stream = torch.cuda.Stream(device=self.device)
                self._fp8_dw_prep_done = torch.cuda.Event()

    @property
    def workspace_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.buffers.values())

    @property
    def activation_workspace_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for name, t in self.buffers.items()
            if not name.startswith(("w1", "w2"))
        )

    @contextmanager
    def _execution(self):
        with self._lock:
            stream = torch.cuda.current_stream(self.device).cuda_stream
            if self._stream is None:
                self._stream = stream
            elif self._stream != stream:
                raise RuntimeError(
                    "a MoE plan must remain on one stream; use a separate plan for graph capture"
                )
            yield

    def check(self):
        """Check device validation after eager execution or a graph replay."""
        status = int(self.status.item())
        if status:
            self._forward_context_available = False
            self._consume_forward_context_generation()
            errors = []
            if status & 1:
                errors.append("expert ID outside [-1, global_experts)")
            if status & 2:
                errors.append("route workspace capacity exceeded")
            if status & 4:
                errors.append("valid_tokens outside input capacity")
            if status & 8:
                errors.append("peer synchronization timed out")
            raise RuntimeError("native MoE: " + "; ".join(errors))

    def reset_errors(self):
        """Clear sticky validation flags once at a captured iteration boundary."""
        self.status.zero_()

    def _validate(self, x, ids, scores, w1, w2, valid_tokens):
        if x.ndim != 2 or x.shape[1] != self.h or x.shape[0] > self.max_tokens:
            raise ValueError("x must be [T,H] with T <= plan.max_tokens")
        if ids.shape != (x.shape[0], self.k) or scores.shape != ids.shape:
            raise ValueError("expected expert IDs and scores [T,topk]")
        if w1.shape != (self.e, 2 * self.i, self.h) or w2.shape != (self.e, self.h, self.i):
            raise ValueError("expert weights do not match the plan")
        for tensor, dtype in (
            (x, torch.bfloat16),
            (ids, torch.int64),
            (scores, torch.float32),
            (w1, torch.bfloat16),
            (w2, torch.bfloat16),
            (valid_tokens, torch.int32),
        ):
            if tensor.device != self.device or tensor.dtype != dtype or not tensor.is_contiguous():
                raise ValueError(
                    "native MoE tensors must have matching device, dtype and contiguous layout"
                )
        if valid_tokens.numel() != 1:
            raise ValueError("valid_tokens must be an int32 device scalar")

    def _quantize_weight(self, name, weight):
        data, scales = self.quantized[name]
        _launch(10, [weight, data, scales], list(weight.shape))

    def _prepare_weights(self, w1, w2, backward=False):
        cached = (
            backward
            and self.fp8_transpose_quantized_weights
            and getattr(self, "_using_forward_context", False)
        )
        if self.fp8_fused_weight_prepare:
            if not cached:
                _launch(
                    36,
                    [
                        w1,
                        w2,
                        *self.quantized["w1"],
                        *self.quantized["w2"],
                        *self.quantized["w1_t"],
                        *self.quantized["w2_t"],
                    ],
                    [self.e, self.h, self.i, self.fp8_weight_prep_ctas],
                )
            return
        if self.compute == "fp8" and not cached:
            self._quantize_weight("w1", w1)
            self._quantize_weight("w2", w2)
        if backward and self.fp8_transpose_quantized_weights:
            _launch(
                35 if self.fp8_transpose_swizzle else 32,
                [
                    *self.quantized["w1"],
                    *self.quantized["w2"],
                    *self.quantized["w1_t"],
                    *self.quantized["w2_t"],
                ],
                [self.e, self.h, self.i],
            )
            return
        if backward:
            if self.fast_weight_transpose and w1.data_ptr() % 16 == 0 and w2.data_ptr() % 16 == 0:
                _launch(22, [w1, w2, self.w1_t, self.w2_t], [self.e, self.h, self.i])
            else:
                self.w1_t.copy_(w1.transpose(1, 2))
                self.w2_t.copy_(w2.transpose(1, 2))
            if self.compute == "fp8":
                self._quantize_weight("w1_t", self.w1_t)
                self._quantize_weight("w2_t", self.w2_t)

    def _begin_forward_context(self):
        if self._forward_context_generation is not None:
            self._forward_context_available = False
            self._forward_context_generation.zero_()

    def _record_forward_context_generation(self):
        if self._forward_context_generation is not None:
            self._forward_context_generation_counter += 1
            self._forward_context_capture_key = dg._C._mega_moe_training_capture_key()
            self._forward_context_generation.fill_(self._forward_context_generation_counter)

    def _consume_forward_context_generation(self):
        if self._forward_context_generation is not None:
            self._forward_context_generation.zero_()

    def _has_live_forward_context(self):
        if not getattr(self, "_forward_context_available", False):
            return False
        # A backward-only capture must rematerialize: a prior eager/other-graph
        # forward cannot remain a valid prerequisite of every future replay.
        capture_key = dg._C._mega_moe_training_capture_key()
        if capture_key:
            return capture_key == self._forward_context_capture_key
        return (
            self._forward_context_generation is None
            or int(self._forward_context_generation.item())
            == self._forward_context_generation_counter
        )

    def _matching_forward_context(self, tensors):
        if not getattr(self, "reuse_forward_context", False):
            return False
        local = (
            self._has_live_forward_context()
            and self._forward_context_fingerprint == self._context_fingerprint(*tensors)
        )
        if torch.cuda.is_current_stream_capturing():
            if (
                not local
                and self._forward_context_available
                and self._forward_context_capture_key == dg._C._mega_moe_training_capture_key()
            ):
                raise RuntimeError("captured forward inputs changed before backward")
            return local
        if self.group is not None:
            # A single rank changing inputs invalidates cached routed values
            # on other ranks too. Agree using host metadata outside capture.
            decisions = [None] * self.size
            dist.all_gather_object(decisions, local, group=self.group)
            return all(decisions)
        return local

    @torch.no_grad()
    def prepare(self, w1, w2):
        """Warm local GEMMs before any EP payload collective is enqueued.

        Construct and warm on the intended execution stream before capture.
        No route data or collective payload is needed during cold compilation.
        """
        with self._execution():
            if self._prepared:
                return
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm up MoE plan before graph capture")
            error = None
            try:
                self._prepare_local(w1, w2)
            except Exception as exception:
                error = exception
            self._finish_preparation(error)

    def _finish_preparation(self, error):
        # Agree on host-side compilation success before publishing peer data.
        # Otherwise one failed compiler can strand other ranks in a GPU wait.
        if self.group is not None:
            errors = [None] * self.size
            dist.all_gather_object(
                errors, None if error is None else str(error)[:2048], group=self.group
            )
            failures = [
                (rank, message) for rank, message in enumerate(errors) if message is not None
            ]
            if failures:
                raise RuntimeError(
                    f"native MoE preparation failed on EP ranks: {failures}"
                ) from error
        elif error is not None:
            raise error
        self._prepared = True

    def _prepare_local(self, w1, w2):
        self._prepare_pointwise()
        w1, w2 = w1.to(torch.bfloat16), w2.to(torch.bfloat16)
        self._prepare_weights(w1, w2, backward=True)
        for buffer in (self.packed_x, self.gate_up, self.hidden, self.packed_dy):
            buffer.zero_()
        self.group_ids.zero_()
        self.lengths.zero_()
        # These zero-length warmups compile and execute in sequence without
        # publishing payloads; every operand is initialized to zero above.
        self._gemm(self.packed_x, w1, self.gate_up, "w1")
        self._gemm(self.hidden, w2, self.packed_output, "w2")
        self._gemm(self.packed_dy, self.w2_t, self.hidden, "w2_t")
        self._gemm(self.gate_up, self.w1_t, self.packed_output, "w1_t")
        if self.compute == "fp8" and self.fuse_activation_quant:
            self._activation()
            self._activation(backward=True)
        if self.fp8_wgrad:
            # The backend warms its FP8 DW family after this common setup.
            # Compiling the unused BF16 DW family adds two cold specializations.
            torch.cuda.synchronize(self.device)
            return
        warmup_dw = torch.zeros_like(w1, dtype=torch.float32)
        dg.k_grouped_bf16_gemm_tn_contiguous(
            self.gate_up,
            self.packed_x,
            warmup_dw,
            None,
            self.lengths,
            c=warmup_dw,
            max_group_k=self.rows,
        )
        del warmup_dw
        warmup_dw = torch.zeros_like(w2, dtype=torch.float32)
        dg.k_grouped_bf16_gemm_tn_contiguous(
            self.packed_dy,
            self.hidden,
            warmup_dw,
            None,
            self.lengths,
            c=warmup_dw,
            max_group_k=self.rows,
        )
        del warmup_dw
        torch.cuda.synchronize(self.device)

    def _prepare_pointwise(self):
        operations = {1, 6, 7, 8, 11, 12, 13}
        if self.compute == "fp8":
            operations.update((9, 10, 14, 15) if self.fuse_activation_quant else (7, 8, 9, 10))
        else:
            operations.update((7, 8))
        if self.active_pointwise:
            operations.update((16, 17, 18))
        if self.vectorized_routes:
            operations.update((19, 20, 21))
        if self.fast_weight_transpose:
            operations.add(22)
        dg._C._mega_moe_training_precompile(self.device.index, sorted(operations))

    def _gemm(self, a, weight, output, name, prequantized=False):
        # The physical ring covers the worst routing skew. GPU lengths select
        # only current expert rows; unused trailing capacity needs no GEMM work.
        # Keep the original contiguous schedule as an explicit benchmark control.
        if self.compute == "bf16":
            if self.m_schedule == "nopad":
                dg.m_grouped_bf16_gemm_nt_nopad(
                    a, weight, output, self.group_ids, m_rows=self.lengths
                )
            else:
                dg.m_grouped_bf16_gemm_nt_contiguous(a, weight, output, self.group_ids)
        else:
            quantized_a = self.quantized[a.shape[1]]
            if not prequantized:
                _launch(9, [a, *quantized_a], list(a.shape))
            if name == "w2_t" and self.fp8_fuse_dhidden_activation:
                return dg._C._mega_moe_training_fp8_gemm(
                    quantized_a,
                    self.quantized[name],
                    output,
                    self.lengths,
                    self.permutation,
                    self._fp8_activation_destinations,
                    self.rows,
                    keep_local=False,
                    activation_gate_up=self.gate_up,
                )
            if self.m_schedule == "nopad":
                dg.m_grouped_fp8_gemm_nt_nopad(
                    quantized_a,
                    self.quantized[name],
                    output,
                    self.group_ids,
                    m_rows=self.lengths,
                    disable_ue8m0_cast=True,
                )
            else:
                dg.m_grouped_fp8_gemm_nt_contiguous(
                    quantized_a,
                    self.quantized[name],
                    output,
                    self.group_ids,
                    disable_ue8m0_cast=True,
                )

    def _fp8_dw_views(self, left, right):
        m, n = left.shape[1], right.shape[1]
        r = self.rows
        return (
            (
                self._fp8_dw_left[: m * r].view(m, r),
                self._fp8_dw_sa[: m * (r // 128)].view(r // 128, m),
            ),
            (
                self._fp8_dw_right[: n * r].view(n, r),
                self._fp8_dw_sb[: (n // 128) * (r // 128)].view(n // 128, r // 128),
            ),
        )

    def _fp8_dw_quantize(self, left, right, first=0, last=None):
        last = self.e if last is None else last
        a, b = self._fp8_dw_views(left, right)
        if getattr(self, "fp8_dw_from_row_cache", False):
            metadata = [self.lengths[first:last], self.offsets[first : last + 1], self.status]
            dg._C._mega_moe_training_fp8_reuse(
                5,
                [left, *a, *metadata],
                [last - first, self.rows, left.shape[1], self.fp8_wgrad_pack_ctas],
            )
            if right.data_ptr() == self.packed_x.data_ptr():
                stream = torch.cuda.current_stream(self.device)
                self.fp8_source_x_q.record_stream(stream)
                self.fp8_source_x_sf.record_stream(stream)
                dg._C._mega_moe_training_fp8_reuse(
                    4,
                    [self.fp8_source_x_q, self.fp8_source_x_sf, *b, *metadata],
                    [last - first, self.rows, right.shape[1], 1, self.fp8_wgrad_pack_ctas],
                )
            else:
                dg._C._mega_moe_training_fp8_reuse(
                    6,
                    [right, *b, *metadata],
                    [last - first, self.rows, right.shape[1], self.fp8_wgrad_pack_ctas],
                )
            return a, b
        tensors = [
            left,
            right,
            *a,
            *b,
            self.lengths[first:last],
            self.offsets[first : last + 1],
            self.status,
        ]
        dims = [
            last - first,
            self.rows,
            left.shape[1],
            right.shape[1],
            7 if self.fp8_wgrad_block_k == 512 else 6,
            self.fp8_wgrad_pack_ctas,
        ]
        if self.fp8_wgrad_reuse_row_scales:
            # These two peer producers quantize the stored BF16 values before
            # their completion event. Unsupported roles keep the original scan.
            known = (
                left.data_ptr() == self.packed_dy.data_ptr()
                or left.data_ptr() == self.gate_up.data_ptr()
                and not getattr(self, "fp8_fuse_dhidden_activation", False)
                and not getattr(self, "fuse_activation_backward", False)
            )
            if known:
                row_sf = self.quantized_scale_storage[left.shape[1]]
                if getattr(self, "_fp8_dw_prep_stream", None) is not None:
                    row_sf.record_stream(torch.cuda.current_stream(self.device))
                tensors.append(row_sf)
                dims.append(int(self.fp8_mn_major_scales))
        _launch(33, tensors, dims)
        return a, b

    def _prepare_fp8_dw_async(self, left, right):
        if self._fp8_dw_ready is not None:
            raise RuntimeError("Previous FP8 DW operands were not consumed")
        main = torch.cuda.current_stream(self.device)
        self._fp8_dw_prep_stream.wait_stream(main)
        with torch.cuda.stream(self._fp8_dw_prep_stream):
            for tensor in (
                left,
                right,
                self.lengths,
                self.offsets,
                self.status,
                self._fp8_dw_left,
                self._fp8_dw_right,
                self._fp8_dw_sa,
                self._fp8_dw_sb,
            ):
                tensor.record_stream(self._fp8_dw_prep_stream)
            self._fp8_dw_quantize(left, right)
            self._fp8_dw_prep_done.record(self._fp8_dw_prep_stream)
        self._fp8_dw_ready = (left.data_ptr(), right.data_ptr())
        self._fp8_dw_ready_on_main = False

    def _fp8_wgrad(self, left, right, output, first_flags=None):
        if self._fp8_dw_ready == (left.data_ptr(), right.data_ptr()):
            if not self._fp8_dw_ready_on_main:
                torch.cuda.current_stream(self.device).wait_event(self._fp8_dw_prep_done)
            a, b = self._fp8_dw_views(left, right)
            self._fp8_dw_ready = None
            self._fp8_dw_ready_on_main = False
        else:
            a, b = self._fp8_dw_quantize(left, right)
        if self.fp8_dw_scale_m != 1:
            # This exact metadata copy is inside complete-layer timing/capture.
            a = (a[0], a[1][:, :: self.fp8_dw_scale_m].contiguous())
        dg._C._mega_moe_training_fp8_wgrad_nt(
            a,
            b,
            output,
            self.lengths,
            self.offsets,
            self.status,
            config=self.fp8_wgrad_config,
            block_scales=True,
            promotion_k=self.fp8_wgrad_block_k,
            first_flags=first_flags,
            native_blockwise=self.fp8_wgrad_native_blockwise,
            swizzle_m=self.fp8_dw_swizzle_m,
            metadata_mode=self.fp8_dw_metadata_mode,
            scale_m=self.fp8_dw_scale_m,
        )

    def _wgrad(self, left, right, output, *, asynchronous=False):
        if self.fp8_wgrad:
            if asynchronous:
                raise RuntimeError("FP8 DW must join before workspace reuse")
            self._fp8_wgrad(left, right, output)
            return
        if not asynchronous:
            dg.k_grouped_bf16_gemm_tn_contiguous(
                left,
                right,
                output,
                None,
                self.lengths,
                c=output,
                max_group_k=self.rows,
            )
            return
        main = torch.cuda.current_stream(self.device)
        self._wgrad_stream.wait_stream(main)
        with torch.cuda.stream(self._wgrad_stream):
            # Preserve allocator lifetime even if a later main-stream launch
            # raises before the normal join. Inputs otherwise remain read-only.
            for tensor in (left, right, output, self.lengths):
                tensor.record_stream(self._wgrad_stream)
            dg.k_grouped_bf16_gemm_tn_contiguous(
                left,
                right,
                output,
                None,
                self.lengths,
                c=output,
                max_group_k=self.rows,
            )
        self._wgrad_pending = True

    def _join_wgrad(self):
        if self._wgrad_pending:
            torch.cuda.current_stream(self.device).wait_stream(self._wgrad_stream)
            self._wgrad_pending = False

    def _join_gradient_prefetch(self):
        if self._gradient_prefetch_pending:
            torch.cuda.current_stream(self.device).wait_stream(self._gradient_stream)
            self._gradient_prefetch_pending = False

    def _activation(self, backward=False, hidden=None):
        hidden = self.hidden if hidden is None else hidden
        fused = self.compute == "fp8" and self.fuse_activation_quant
        if backward and self.fp8_fuse_dhidden_activation:
            _launch(9, [self.gate_up, *self.quantized[2 * self.i]], list(self.gate_up.shape))
            return True
        if fused:
            tensors = [self.gate_up, hidden] + ([self.gate_up] if backward else [])
            tensors += list(self.quantized[2 * self.i if backward else self.i])
            _launch(15 if backward else 14, tensors, [self.rows, self.i])
        elif self.active_pointwise:
            _launch(
                16,
                [self.gate_up, hidden, self.current_rows, self.status],
                [self.rows, self.i, int(backward)] + ([8] if self.g8_interleaved else []),
            )
        else:
            tensors = [self.gate_up, hidden] + ([self.gate_up] if backward else [])
            _launch(
                8 if backward else 7,
                tensors,
                [self.rows, self.i] + ([8] if self.g8_interleaved else []),
            )
        return fused

    def _combine_local_routes(self, values, scores, output, start, weighted):
        tensors = [values, scores, output]
        if getattr(self, "skip_masked_route_zero", False):
            tensors.append(self.local_ids)
        _launch(
            6, tensors, [self.chunk_tokens, self.h, self.k, scores.shape[0], start, int(weighted)]
        )

    @torch.no_grad()
    def _forward_local(self, x, ids, scores, w1, w2, valid_tokens, *, _chunk_output_callback=None):
        if _chunk_output_callback is not None:
            if not callable(_chunk_output_callback):
                raise TypeError("chunk callback must be callable")
            if self.transport != "peer" or getattr(self, "ring_rows", None) is None:
                raise ValueError("chunk callbacks require bounded peer source execution")
        self._validate(x, ids, scores, w1, w2, valid_tokens)
        with self._execution():
            self.prepare(w1, w2)
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.reset_errors()
            self._prepare_weights(w1, w2)
            output = torch.empty_like(x)
            for start in range(0, self.max_tokens, self.chunk_tokens):
                self._route_chunk(x, ids, valid_tokens, start)
                self._chunk_forward(w1, w2)
                self._combine_local_routes(self.local_values, scores, output, start, True)
                if _chunk_output_callback is not None:
                    _chunk_output_callback(
                        output, start, max(0, min(self.chunk_tokens, x.shape[0] - start))
                    )
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.check()
            return output


class _MegaMoEAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ids, scores, w1, w2, valid_tokens, plan, accumulators):

        _validate_accumulators(accumulators, w1, w2)
        if any(weight.dtype not in (torch.bfloat16, torch.float32) for weight in (w1, w2)):
            raise TypeError("expert weights must be BF16 or FP32 master parameters")
        forward_w1, forward_w2 = w1.to(torch.bfloat16), w2.to(torch.bfloat16)
        output = plan.forward(x, ids, scores, forward_w1, forward_w2, valid_tokens)
        ctx.reuse_forward_context = getattr(plan, "reuse_forward_context", False)
        if ctx.reuse_forward_context:
            # The forward context corresponds to these exact BF16 parameters;
            # retain them alongside FP32 masters through the matching backward.
            ctx.save_for_backward(x, ids, scores, w1, w2, valid_tokens, forward_w1, forward_w2)
        else:
            ctx.save_for_backward(x, ids, scores, w1, w2, valid_tokens)
        ctx.plan, ctx.accumulators = plan, accumulators
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        x, ids, scores, w1, w2, valid_tokens = ctx.saved_tensors[:6]
        backward_w1, backward_w2 = (
            ctx.saved_tensors[6:]
            if ctx.reuse_forward_context
            else (w1.to(torch.bfloat16), w2.to(torch.bfloat16))
        )
        dx, dw1, dw2, dscores = ctx.plan.backward(
            dy.contiguous(),
            x,
            ids,
            scores,
            backward_w1,
            backward_w2,
            valid_tokens,
            ctx.accumulators,
        )
        if ctx.accumulators is not None:
            dw1 = dw2 = None
        else:
            dw1, dw2 = dw1.to(w1.dtype), dw2.to(w2.dtype)
        return dx, None, dscores, dw1, dw2, None, None, None


def mega_moe(
    x, expert_ids, scores, w1, w2, *, plan, valid_tokens=None, weight_grad_accumulators=None
):
    """Routed SwiGLU training using GPU schedules and bounded replay storage.

    Input is BF16, parameters BF16 or FP32 masters, and scores FP32. FP32 masters
    are cast for compute and receive FP32 wgrad directly. IDs may be int32/int64; masked
    slots use -1. valid_tokens is an optional int32 device scalar for dynamic
    padding within a captured input shape. After graph replay, call plan.check()
    outside capture to inspect invalid-route/capacity errors.
    """
    if not isinstance(plan, _MoEPlanBase):
        raise TypeError("plan must be a MegaMoEPlan or DeepEPMoEPlan")
    if expert_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert IDs must be int32 or int64")
    if valid_tokens is None:
        # A device fill is capturable; torch.tensor(host_scalar, device=...)
        # performs a host-to-device transfer which the runtime rejects in capture.
        valid_tokens = torch.full((), x.shape[0], device=x.device, dtype=torch.int32)
    return _MegaMoEAutograd.apply(
        x, expert_ids.to(torch.int64), scores, w1, w2, valid_tokens, plan, weight_grad_accumulators
    )
