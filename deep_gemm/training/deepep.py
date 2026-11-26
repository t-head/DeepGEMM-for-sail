import math

"""Installed DeepEP BF16 dispatch/combine with BF16 or FP8 grouped GEMMs.

DeepEP transports one copy per destination rank. Local GPU metadata expands
only the expert rows needed by the grouped GEMMs. All receive capacities are
bounded and all routing counts remain on device during graph replay.
"""

import os

import torch
import torch.distributed as dist

import deep_gemm as dg
from ._common import (
    _MoEPlanBase,
    _launch,
    _validate_accumulators,
    allocate_route_windows,
    set_route_window,
)


class _CompletedEvent:
    def current_stream_wait(self):
        pass


class ElasticBufferAdapter:
    def __init__(self, group, chunk_tokens, hidden, topk, experts, comm_sms, hybrid):
        import deep_ep

        self.buffer = deep_ep.ElasticBuffer(
            group,
            num_max_tokens_per_rank=chunk_tokens,
            hidden=hidden,
            num_topk=topk,
            deterministic=False,
            allow_hybrid_mode=hybrid,
            explicitly_destroy=True,
        )
        self.experts, self.comm_sms = experts, comm_sms
        self.backend = "elastic_hybrid" if hybrid else "elastic_flat"
        self.runtime = self

    def get_num_rdma_ranks(self):
        return self.buffer.num_scaleout_ranks

    def get_dispatch_layout(self, ids, experts, async_finish=True):
        if experts != self.experts:
            raise ValueError("expert count must match the elastic plan")
        # ElasticBuffer constructs its routing metadata in dispatch on device.
        return None, None, None, None, _CompletedEvent()

    def dispatch(
        self,
        x,
        *,
        topk_idx=None,
        topk_weights=None,
        handle=None,
        async_finish=True,
        **legacy_layout,
    ):
        options = dict(
            num_sms=self.comm_sms,
            do_cpu_sync=False,
            do_expand=False,
            async_with_compute_stream=True,
        )
        if handle is None:
            options.update(
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_experts=self.experts,
                expert_alignment=1,
            )
        else:
            options["handle"] = handle
        rx, ri, rw, new_handle, event = self.buffer.dispatch(x, **options)
        if handle is None:
            # Without CPU synchronization Elastic allocates the worst-case
            # receive tensor but only initializes its GPU-counted prefix. The
            # legacy Buffer ABI instead guarantees -1 IDs in padding rows.
            event.current_stream_wait()
            data = rx[0] if isinstance(rx, tuple) else rx
            valid = (
                torch.arange(data.shape[0], device=data.device)
                < new_handle.psum_num_recv_tokens_per_scaleup_rank[-1]
            )
            ri.masked_fill_(~valid[:, None], -1)
            if rw is not None:
                rw.masked_fill_(~valid[:, None], 0)
            event = _CompletedEvent()
        return rx, ri, rw, None, new_handle, event

    def combine(self, x, handle, *, topk_weights=None, async_finish=True):
        return self.buffer.combine(
            x,
            handle,
            topk_weights=topk_weights,
            num_sms=self.comm_sms,
            async_with_compute_stream=True,
        )

    def destroy(self):
        self.buffer.destroy()


class DeepEPMoEPlan(_MoEPlanBase):
    transport = "deepep"

    def __init__(
        self,
        *args,
        comm_sms=8,
        gemm_configs=None,
        rdma_qps=None,
        ring_rows=None,
        elastic_hybrid=None,
        fp8_active_quantization=None,
        fp8_mn_major_scales=None,
        fp8_async_weight_quantization=None,
        fp8_weight_quantization=None,
        fp8_source_dispatch=None,
        reuse_forward_context=None,
        **kwargs,
    ):
        def quant_flag(value, key):
            if value is None:
                raw = os.environ.get(key, "0")
                if raw not in ("0", "1"):
                    raise ValueError(key + " must be0or1")
                value = raw == "1"
            if type(value) is not bool:
                raise TypeError(key + " must be bool")
            return value

        self.deepep_active_receive = quant_flag(None, "DG_MOE_DEEPEP_ACTIVE_RECEIVE")
        self.fp8_active_quantization = quant_flag(
            fp8_active_quantization, "DG_MOE_FP8_ACTIVE_QUANTIZATION"
        )
        self.fp8_mn_major_scales = quant_flag(fp8_mn_major_scales, "DG_MOE_FP8_MN_MAJOR_SCALES")
        self.fp8_async_weight_quantization = quant_flag(
            fp8_async_weight_quantization, "DG_MOE_FP8_ASYNC_WEIGHTS"
        )
        self.fp8_source_dispatch = quant_flag(fp8_source_dispatch, "DG_MOE_FP8_SOURCE_DISPATCH")
        self.reuse_forward_context = quant_flag(
            reuse_forward_context, "DG_MOE_REUSE_FORWARD_CONTEXT"
        )
        if self.reuse_forward_context and (
            ring_rows is None
            or kwargs.get("wgrad_overlap", "none") != "none"
            or (
                kwargs.get("compute", "bf16") == "fp8"
                and not (self.fp8_active_quantization and self.fp8_mn_major_scales)
            )
        ):
            raise ValueError(
                "DeepEP forward context requires routed windows and synchronous gradients, with active MN FP8 scales"
            )
        self._forward_context_available = self._using_forward_context = False
        self._forward_context_fingerprint = None
        if self.fp8_source_dispatch and (
            kwargs.get("compute", "bf16") != "fp8"
            or elastic_hybrid is None
            or not self.fp8_active_quantization
            or not self.fp8_mn_major_scales
        ):
            raise ValueError(
                "source FP8 dispatch requires Elastic FP8 compute with active MN scales"
            )
        self._fp8_dispatch_backward = False
        if fp8_weight_quantization is None:
            raw = os.environ.get("DG_MOE_FP8_WEIGHT_QUANTIZER", "0")
            if raw not in ("0", "1", "2"):
                raise ValueError("DG_MOE_FP8_WEIGHT_QUANTIZER must be0,1or2")
            fp8_weight_quantization = int(raw)
        if type(fp8_weight_quantization) is not int or fp8_weight_quantization not in (0, 1, 2):
            raise ValueError("fp8_weight_quantization must be0,1or2")
        self.fp8_weight_quantization = fp8_weight_quantization
        if self.fp8_mn_major_scales and not self.fp8_active_quantization:
            raise ValueError("MN-major FP8 scales require active-row quantization")
        if (
            self.fp8_active_quantization
            or self.fp8_async_weight_quantization
            or self.fp8_weight_quantization
        ) and kwargs.get("compute", "bf16") != "fp8":
            raise ValueError("FP8 quantization options require FP8 expert compute")
        if kwargs.get("compute", "bf16") == "fp8" and kwargs.get("m_schedule", "nopad") != "nopad":
            raise ValueError("DeepEP FP8 GEMMs require the nopad expert schedule")
        if kwargs.get("group") is None:
            raise ValueError("DeepEP requires an initialized EP process group")
        import deep_ep

        if "packed_rows" in kwargs:
            raise ValueError("use ring_rows to enable routed-row scheduling")
        self.ring_rows = ring_rows
        self.communication_dtype = "bf16"
        if self.fp8_source_dispatch:
            self.communication_dtype = "fp8_forward_input_bf16_backward_input_and_return"
        super().__init__(*args, packed_rows=ring_rows, **kwargs)
        stable_raw = os.environ.get(
            "DG_MOE_FP8_WGRAD_STABLE_ROUTING", "1" if self.fp8_wgrad else "0"
        )
        if stable_raw not in ("0", "1"):
            raise ValueError("Stable FP8 DW routing flag must be0or1")
        self.fp8_dw_stable_routing = stable_raw == "1"
        route_block_raw = os.environ.get("DG_MOE_FP8_DW_ROUTE_BLOCK", "2048")
        if route_block_raw not in ("256", "512", "1024", "2048"):
            raise ValueError("FP8 DW route block must be256,512,1024or2048")
        self.fp8_dw_route_block = int(route_block_raw)
        native_raw = os.environ.get("DG_MOE_FP8_DW_NATIVE_PREFIX", "1")
        if native_raw not in ("0", "1"):
            raise ValueError("Native route prefix flag must be0or1")
        self.fp8_dw_native_prefix = native_raw == "1"
        recipe = (
            self.deepep_active_receive,
            self.fp8_active_quantization,
            self.fp8_mn_major_scales,
            self.fp8_async_weight_quantization,
            self.fp8_weight_quantization,
            self.fp8_source_dispatch,
            self.reuse_forward_context,
            self.fp8_dw_stable_routing,
            self.fp8_dw_route_block,
            self.fp8_dw_native_prefix,
        )
        recipes = [None] * self.size
        dist.all_gather_object(recipes, recipe, group=self.group)
        if any(other != recipe for other in recipes):
            raise ValueError("DeepEP ranks disagree on FP8 quantization layout")
        self._weight_quant_stream = (
            torch.cuda.Stream(device=self.device) if self.fp8_async_weight_quantization else None
        )
        self._forward_weight_events = (
            {name: torch.cuda.Event() for name in ("w1", "w2")}
            if self.fp8_async_weight_quantization
            else {}
        )
        self._backward_weight_events = (
            {name: torch.cuda.Event() for name in ("w1", "w2", "w2_t", "w1_t")}
            if self.fp8_async_weight_quantization
            else {}
        )
        self._pending_weight_events = {}
        self.quantized_scale_storage = {}
        if self.compute == "fp8":
            for width in (self.h, self.i, 2 * self.i):
                if width in self.quantized_scale_storage:
                    continue
                if self.fp8_mn_major_scales:
                    storage = self._allocate(
                        "activation_scale_mn_" + str(width),
                        (width // 128, self.rows),
                        torch.float32,
                    )
                    data = self.quantized[width][0]
                    self.quantized[width] = (data, storage.t())
                    self.buffers.pop("activation_scale_" + str(width))
                else:
                    storage = self.quantized[width][1]
                self.quantized_scale_storage[width] = storage
        if ring_rows is not None and (
            self.k > 32 or self.size * self.chunk_tokens * self.h > 2**31 - 1
        ):
            raise ValueError(
                "DeepEP routed-row windows require topk<=32 and an int32-bounded receive matrix"
            )
        # Preserve forward Hidden for concurrent DW2; dHidden uses a distinct
        # bounded ring and the existing activation-backward kernel consumes it.
        self.backward_hidden = (
            self._allocate("deepep_backward_hidden", (self.rows, self.i), torch.bfloat16)
            if self.wgrad_overlap == "both"
            else None
        )
        self.active_pointwise = self.vectorized_routes = True
        self.gemm_configs = gemm_configs or {}
        if self.block_m == 256 and not self.gemm_configs:
            self.gemm_configs = {
                role: (256, 64, 64, 64, 2) for role in ("fc1", "fc2", "dhidden", "dx")
            }
        if type(comm_sms) is not int or comm_sms < 2 or comm_sms % 2:
            raise ValueError("DeepEP communication CU count must be an even integer >=2")
        self.comm_sms = comm_sms
        self.deep_ep_qps = max(12, (comm_sms + 1) // 2) if rdma_qps is None else rdma_qps
        if type(self.deep_ep_qps) is not int or self.deep_ep_qps < (comm_sms + 1) // 2:
            raise ValueError("DeepEP RDMA QPs must cover at least one pair of communication CUs")
        if elastic_hybrid is None:
            deep_ep.Buffer.set_num_sms(comm_sms)
            configs = (
                deep_ep.Buffer.get_dispatch_config(self.size),
                deep_ep.Buffer.get_combine_config(self.size),
            )
            nvl = max(c.get_nvl_buffer_size_hint(self.h * 2, self.size) for c in configs)
            rdma = max(c.get_rdma_buffer_size_hint(self.h * 2, self.size) for c in configs)
            self.deep_ep_buffer = deep_ep.Buffer(
                self.group,
                nvl,
                rdma,
                allow_mnnvl=True,
                explicitly_destroy=True,
                num_qps_per_rank=self.deep_ep_qps,
            )
        else:
            if not isinstance(elastic_hybrid, bool):
                raise TypeError("elastic_hybrid must be bool or None for legacy Buffer")

            self.deep_ep_buffer = ElasticBufferAdapter(
                self.group,
                self.chunk_tokens,
                self.h,
                self.k,
                self.e * self.size,
                comm_sms,
                elastic_hybrid,
            )
        self.deep_ep_backend = self.deep_ep_buffer.backend
        self.deep_ep_rdma_ranks = self.deep_ep_buffer.runtime.get_num_rdma_ranks()
        self.deep_ep_version = deep_ep.__version__
        # Avoid the all-gather reference's [ranks*C,topk,H] temporary. DeepEP
        # unpermutes directly into its deduplicated [receive_capacity,H] layout.
        for name in ("gathered_x", "gathered_ids", "route_values", "local_values", "local_dy"):
            self.buffers.pop(name)
            delattr(self, name)
        n = self.size * self.chunk_tokens
        self.recv_capacity = n
        self.local_scores = self._allocate(
            "deepep_local_scores", (self.chunk_tokens, self.k), torch.float32
        )
        self.local_dy_input = self._allocate(
            "deepep_local_dy", (self.chunk_tokens, self.h), torch.bfloat16
        )
        self.reduced = self._allocate("deepep_reduced", (n, self.h), torch.bfloat16)
        self.recv_dscores = self._allocate("deepep_dscores", (n, self.k), torch.float32)
        if self.fp8_source_dispatch:
            self.source_qx = self._allocate(
                "deepep_source_qx", (self.chunk_tokens, self.h), torch.float8_e4m3fn
            )
            self.source_qsf = self._allocate(
                "deepep_source_qsf", (self.chunk_tokens, self.h // 128), torch.float32
            )
            self.recv_qx = self._allocate("deepep_recv_qx", (n, self.h), torch.float8_e4m3fn)
            self.recv_qsf = self._allocate("deepep_recv_qsf", (n, self.h // 128), torch.float32)
        if self.fp8_dw_stable_routing:
            self._stable_recv_ids = self._allocate(
                "stable_recv_ids", (self.recv_capacity, self.k), torch.int64
            )
            self._stable_recv_pointer = torch.tensor(
                [self._stable_recv_ids.data_ptr()], device=self.device, dtype=torch.int64
            )
            self.buffers["stable_recv_pointer"] = self._stable_recv_pointer
            shape = ((self.routes + self.fp8_dw_route_block - 1) // self.fp8_dw_route_block, self.e)
            self._stable_recv_hist = self._allocate("stable_recv_hist", shape, torch.int32)
            self._stable_recv_prefix = self._allocate("stable_recv_prefix", shape, torch.int32)
        self.route_window_count = 1
        if ring_rows is not None:
            allocate_route_windows(self)
            self.recv_accumulator = self._allocate_receive_accumulator(n)
        self.forward_context_slots = []
        self.forward_context_receives = []
        if self.reuse_forward_context:
            original = {
                name: getattr(self, name)
                for name in (
                    "gate_up",
                    "hidden",
                    "packed_output",
                    "global_offsets",
                    "global_lengths",
                    "global_permutation",
                    "route_experts",
                    "route_rows",
                )
            }
            if self.compute == "fp8":
                original.update(
                    quantized_hidden=self.quantized[self.i][0],
                    quantized_hidden_scales=self.quantized_scale_storage[self.i],
                )
            for slot in range(
                max(1, (self.max_tokens + self.chunk_tokens - 1) // self.chunk_tokens)
            ):
                tensors = (
                    original
                    if slot == 0
                    else {name: torch.empty_like(value) for name, value in original.items()}
                )
                if slot:
                    self.buffers.update(
                        {f"forward_context_{slot}_{name}": value for name, value in tensors.items()}
                    )
                self.forward_context_slots.append(tensors)
                self.forward_context_receives.append(None)
            for name, shape, dtype in (
                ("replay_offsets", (self.e + 1,), torch.int32),
                ("replay_lengths", (self.e,), torch.int32),
                ("replay_blocks", (1,), torch.int32),
                ("cache_only_blocks", (1,), torch.int32),
                ("context_placeholder_blocks", (1,), torch.int32),
                ("forward_context_reused_windows", (1,), torch.int32),
                ("forward_context_reused_rows", (1,), torch.int64),
            ):
                value = self._allocate(name, shape, dtype)
                value.zero_()
                setattr(self, name, value)

    def _allocate_receive_accumulator(self, capacity):
        return self._allocate("deepep_recv_accumulator", (capacity, self.h), torch.float32)

    def _prefix_received_routes(self, offsets, lengths, capacity):
        _launch(1, [self.counts, offsets, lengths], [self.e, self.alignment])

    @staticmethod
    def _context_fingerprint(*tensors):
        return tuple((value.data_ptr(), value._version, tuple(value.shape)) for value in tensors)

    def _activate_forward_context(self, start):
        if not self.reuse_forward_context:
            return
        for name, value in self.forward_context_slots[start // self.chunk_tokens].items():
            if name == "quantized_hidden":
                self.quantized[self.i] = (value, self.quantized[self.i][1])
            elif name == "quantized_hidden_scales":
                self.quantized_scale_storage[self.i] = value
                self.quantized[self.i] = (self.quantized[self.i][0], value.t())
            else:
                setattr(self, name, value)

    def _allocate(self, name, shape, dtype):
        value = torch.empty(shape, device=self.device, dtype=dtype)
        self.buffers[name] = value
        return value

    def _quantize_weight(self, name, weight):
        data, scales = self.quantized[name]
        operation = (10, 29, 30)[self.fp8_weight_quantization]
        _launch(operation, [weight, data, scales], list(weight.shape))

    def _prepare_weights(self, w1, w2, backward=False):
        self._fp8_dispatch_backward = backward
        if not self.fp8_async_weight_quantization:
            return super()._prepare_weights(w1, w2, backward=backward)
        main = torch.cuda.current_stream(self.device)
        self._weight_quant_stream.wait_stream(main)
        events = self._backward_weight_events if backward else self._forward_weight_events
        with torch.cuda.stream(self._weight_quant_stream):
            # BF16 casts may be temporary tensors owned by the main stream.
            w1.record_stream(self._weight_quant_stream)
            w2.record_stream(self._weight_quant_stream)
            self._quantize_weight("w1", w1)
            events["w1"].record()
            self._quantize_weight("w2", w2)
            events["w2"].record()
            if backward:
                if (
                    self.fast_weight_transpose
                    and w1.data_ptr() % 16 == 0
                    and w2.data_ptr() % 16 == 0
                ):
                    _launch(22, [w1, w2, self.w1_t, self.w2_t], [self.e, self.h, self.i])
                else:
                    self.w1_t.copy_(w1.transpose(1, 2))
                    self.w2_t.copy_(w2.transpose(1, 2))
                # dHidden consumes W2_t before DX consumes W1_t.
                self._quantize_weight("w2_t", self.w2_t)
                events["w2_t"].record()
                self._quantize_weight("w1_t", self.w1_t)
                events["w1_t"].record()
        self._pending_weight_events = dict(events)

    def _wait_for_weight(self, name):
        event = self._pending_weight_events.pop(name, None)
        if event is not None:
            torch.cuda.current_stream(self.device).wait_event(event)

    def _join_weight_quantization(self):
        # Rejoin side-stream work even for an empty max-token capacity.
        for event in self._pending_weight_events.values():
            torch.cuda.current_stream(self.device).wait_event(event)
        self._pending_weight_events.clear()

    def _prepare_local(self, w1, w2):
        if self.fp8_active_quantization or self.fp8_wgrad:
            # Native warming zeros lengths but normally never reads offsets.
            # Active quantizers consume this GPU scalar even during warming.
            self.offsets.zero_()
        super()._prepare_local(w1, w2)
        if self.fp8_wgrad:
            # Zero lengths/offsets warm the actual DW pack and GEMM families
            # without route data or any outstanding dispatch payload.
            self._wgrad(self.gate_up, self.packed_x, torch.zeros_like(w1, dtype=torch.float32))
            self._wgrad(self.packed_dy, self.hidden, torch.zeros_like(w2, dtype=torch.float32))
            torch.cuda.synchronize(self.device)

    def _activation(self, backward=False, hidden=None):
        if not (self.fp8_active_quantization and self.fuse_activation_quant):
            return super()._activation(backward=backward, hidden=hidden)
        hidden = self.hidden if hidden is None else hidden
        width = 2 * self.i if backward else self.i
        if backward and self.fp8_fuse_dhidden_activation:
            _launch(
                34 if self.fp8_vector_row_quant else 26,
                [
                    self.gate_up,
                    self.quantized[width][0],
                    self.quantized_scale_storage[width],
                    self.current_rows,
                    self.status,
                ],
                [self.rows, width, int(self.fp8_mn_major_scales)],
            )
            return True
        tensors = [self.gate_up, hidden] + ([self.gate_up] if backward else [])
        tensors.extend(
            (
                self.quantized[width][0],
                self.quantized_scale_storage[width],
                self.current_rows,
                self.status,
            )
        )
        _launch(
            28 if backward else 27,
            tensors,
            [self.rows, self.i, int(self.fp8_mn_major_scales), int(self.fp8_vector_activation)],
        )
        return True

    def _prepare_pointwise(self):
        operations = [0, 1, 2, 7, 8, 13, 16, 19, 20, 23, 24]
        if self.compute == "fp8":
            operations.extend((9, 10, 14, 15))
            if self.fp8_weight_quantization:
                operations.append((10, 29, 30)[self.fp8_weight_quantization])
            if self.fp8_active_quantization:
                operations.extend((34 if self.fp8_vector_row_quant else 26, 27, 28))
            if self.fp8_source_dispatch:
                operations.append(31)
        if self.fast_weight_transpose:
            operations.append(22)
        if self.ring_rows is not None:
            operations.append(25)
            set_route_window(self, 0, compile_only=True)
            dg._C._mega_moe_training_launch(
                1,
                [self.global_permutation, self.global_source_rows],
                [self.full_row_capacity, self.recv_capacity],
                True,
            )
        if self.deepep_active_receive:
            operations.extend([37, 38])
        dg._C._mega_moe_training_precompile(self.device.index, operations)
        if self.fp8_dw_stable_routing:
            tensors = [
                self._stable_recv_pointer,
                self.counts,
                self.route_experts,
                self.route_rows,
                self.status,
                self._stable_recv_hist,
                self._stable_recv_prefix,
            ]
            dims = [self.routes, self.e, 0, 1, self.routes]
            for finish in (False, True):
                dg._C._mega_moe_training_stable_routes(
                    tensors,
                    dims,
                    finish=finish,
                    compile_only=True,
                    block_routes=self.fp8_dw_route_block,
                    scan_prefix=finish and self.fp8_dw_native_prefix,
                )
        if self.reuse_forward_context:
            self._prepare_context_replay(0, compile_only=True)

    def _prepare_context_replay(self, start, compile_only=False):
        dg._C._mega_moe_training_launch(
            20,
            [
                self.offsets,
                self.lengths,
                self.context_placeholder_blocks,
                self.global_offsets,
                self.replay_offsets,
                self.replay_lengths,
                self.replay_blocks,
                self.cache_only_blocks,
                self.status,
                self.forward_context_reused_windows,
                self.forward_context_reused_rows,
            ],
            [self.e, self.rows, start],
            compile_only,
        )

    def _gemm(self, a, weight, output, name, prequantized=False):
        role = {"w1": "fc1", "w2": "fc2", "w2_t": "dhidden", "w1_t": "dx"}[name]
        config = self.gemm_configs.get(role)
        options = (
            {}
            if config is None
            else {"configs": (dg.get_num_sms(), self.block_m, *config, (0, 0, 0))}
        )
        resolved = getattr(self, "resolved_gemm_configs", {}).get(role)
        if resolved is not None:
            options = {"configs": resolved}
        if self.compute == "fp8":
            quantized_a = self.quantized[a.shape[1]]
            if not prequantized:
                if self.fp8_active_quantization:
                    _launch(
                        34 if self.fp8_vector_row_quant else 26,
                        [
                            a,
                            quantized_a[0],
                            self.quantized_scale_storage[a.shape[1]],
                            self.current_rows,
                            self.status,
                        ],
                        [self.rows, a.shape[1], int(self.fp8_mn_major_scales)],
                    )
                else:
                    _launch(9, [a, *quantized_a], list(a.shape))
            self._wait_for_weight(name)
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
                    config=() if config is None else config,
                    activation_gate_up=self.gate_up,
                )
            dg.m_grouped_fp8_gemm_nt_nopad(
                quantized_a,
                self.quantized[name],
                output,
                self.group_ids,
                m_rows=self.lengths,
                disable_ue8m0_cast=True,
                **options,
            )
        else:
            dg.m_grouped_bf16_gemm_nt_nopad(
                a, weight, output, self.group_ids, m_rows=self.lengths, **options
            )

    def _dispatch_chunk(self, x, ids, scores, valid_tokens, start):
        self._activate_forward_context(start)
        t, c, h, k = x.shape[0], self.chunk_tokens, self.h, self.k
        _launch(
            13,
            [x, ids, self.local_x, self.local_ids, valid_tokens, self.status],
            [t, c, h, k, start],
        )
        available = max(0, min(c, t - start))
        self.local_scores.zero_()
        self.local_scores[:available].copy_(scores[start : start + available])
        buffer = self.deep_ep_buffer
        counts, rdma_counts, expert_counts, in_rank, event = buffer.get_dispatch_layout(
            self.local_ids, self.e * self.size, async_finish=True
        )
        event.current_stream_wait()
        payload = self.local_x
        if self.fp8_source_dispatch and not self._fp8_dispatch_backward:
            _launch(9, [self.local_x, self.source_qx, self.source_qsf], [c, h])
            payload = (self.source_qx, self.source_qsf)
        rx, ri, rw, _, handle, event = buffer.dispatch(
            payload,
            topk_idx=self.local_ids,
            topk_weights=self.local_scores,
            num_tokens_per_rank=counts,
            num_tokens_per_rdma_rank=rdma_counts,
            num_tokens_per_expert=expert_counts,
            is_token_in_rank=in_rank,
            num_worst_tokens=self.recv_capacity,
            async_finish=True,
        )
        event.current_stream_wait()
        if self.reuse_forward_context and not self._fp8_dispatch_backward:
            self.forward_context_receives[start // self.chunk_tokens] = (rx, rw, handle)
        data = rx[0] if isinstance(rx, tuple) else rx
        if data.shape != (self.recv_capacity, h) or ri.shape != (self.recv_capacity, k):
            raise RuntimeError("unexpected DeepEP fixed-capacity receive layout")
        if self.fp8_source_dispatch:
            if isinstance(rx, tuple):
                self.window_recv_q, self.window_recv_qsf = rx
                self.window_recv_x = None
            else:
                # Preserve original BF16 input for DW; quantize the unique
                # received token prefix once before expert/window expansion.
                self.window_recv_x = rx
                received = handle.psum_num_recv_tokens_per_scaleup_rank[-1:]
                _launch(
                    34 if self.fp8_vector_row_quant else 26,
                    [rx, self.recv_qx, self.recv_qsf, received, self.status],
                    [self.recv_capacity, h, 0],
                )
                self.window_recv_q, self.window_recv_qsf = self.recv_qx, self.recv_qsf
            if (
                self.window_recv_qsf.dtype != torch.float32
                or self.window_recv_qsf.shape != (self.recv_capacity, h // 128)
                or not self.window_recv_qsf.is_contiguous()
            ):
                raise RuntimeError("source FP8 dispatch requires row-major FP32 receive scales")
        # DeepEP converts global expert IDs into destination-local expert IDs.
        if self.fp8_dw_stable_routing:
            # Pin the graph-captured receive-ID pointer while retaining DeepEP's
            # receive row numbers for its existing combine/gradient handles.
            self._stable_recv_ids.copy_(ri)
            tensors = [
                self._stable_recv_pointer,
                self.counts,
                self.route_experts,
                self.route_rows,
                self.status,
                self._stable_recv_hist,
                self._stable_recv_prefix,
            ]
            dims = [self.routes, self.e, 0, 1, self.routes]
            dg._C._mega_moe_training_stable_routes(tensors, dims, block_routes=self.fp8_dw_route_block)
            if not self.fp8_dw_native_prefix:
                torch.cumsum(
                    self._stable_recv_hist, dim=0, dtype=torch.int32, out=self._stable_recv_prefix
                )
            dg._C._mega_moe_training_stable_routes(
                tensors,
                dims,
                True,
                block_routes=self.fp8_dw_route_block,
                scan_prefix=self.fp8_dw_native_prefix,
            )
        else:
            self.counts.zero_()
            _launch(
                0,
                [ri, self.counts, self.route_experts, self.route_rows, self.status],
                [self.routes, self.e, 0, 1],
            )
        offsets = self.offsets if self.ring_rows is None else self.global_offsets
        lengths = self.lengths if self.ring_rows is None else self.global_lengths
        permutation = self.permutation if self.ring_rows is None else self.global_permutation
        capacity = self.rows if self.ring_rows is None else self.full_row_capacity
        self._prefix_received_routes(offsets, lengths, capacity)
        if self.ring_rows is None:
            _launch(2, [self.offsets, self.group_ids, self.permutation], [self.rows, self.e])
        else:
            dg._C._mega_moe_training_launch(
                1, [permutation, self.global_source_rows], [capacity, self.recv_capacity], False
            )
        _launch(
            19,
            [self.route_experts, self.route_rows, offsets, permutation, self.status],
            [self.routes, capacity],
        )
        if self.ring_rows is not None:
            if not self.fp8_source_dispatch:
                self.window_recv_x = rx
            return rw, handle
        if not self.fp8_source_dispatch or self._fp8_dispatch_backward:
            _launch(
                20,
                [data, self.permutation, self.packed_x, self.current_rows, self.status],
                [self.rows, h, self.recv_capacity, k],
            )
        return rw, handle

    def _expert_forward(self, w1, w2, _context_masked=False):
        if self._using_forward_context and not _context_masked:
            self._prepare_context_replay(self._context_window_start)
            original = self.offsets, self.lengths, self.current_rows
            self.offsets, self.lengths = self.replay_offsets, self.replay_lengths
            self.current_rows = self.replay_offsets[self.e : self.e + 1]
            try:
                self._expert_forward(w1, w2, _context_masked=True)
            finally:
                self.offsets, self.lengths, self.current_rows = original
            return
        if self.fp8_source_dispatch:
            self._pack_received_fp8()
        self._gemm(self.packed_x, w1, self.gate_up, "w1", prequantized=self.fp8_source_dispatch)
        fused = self._activation()
        self._gemm(self.hidden, w2, self.packed_output, "w2", prequantized=fused)

    def _combine(self, rw, handle, weighted, dscores=None):
        if self.ring_rows is not None:
            # Accumulate all local expert/window contributions in FP32 and
            # round only once at DeepEP's existing BF16 payload boundary.
            if self.deepep_active_receive:
                _launch(
                    38,
                    [
                        self.recv_accumulator,
                        self.reduced,
                        handle.psum_num_recv_tokens_per_scaleup_rank[-1:],
                        self.status,
                    ],
                    [self.recv_capacity, self.h],
                )
            else:
                self.reduced.copy_(self.recv_accumulator)
        else:
            _launch(
                23,
                [
                    self.packed_output,
                    self.route_experts,
                    self.route_rows,
                    self.offsets,
                    rw,
                    self.reduced,
                    self.status,
                ],
                [self.rows, self.h, self.recv_capacity, self.k, self.e, int(weighted)],
            )
        output, ds, event = self.deep_ep_buffer.combine(
            self.reduced, handle, topk_weights=dscores, async_finish=True
        )
        event.current_stream_wait()
        return output, ds

    def _set_window(self, start):
        # This is the mutation boundary for packed X and device lengths in
        # both forward and backward. Auxiliary wgrads must finish reading them.
        self._join_wgrad()
        set_route_window(self, start)
        if not self.fp8_source_dispatch or self._fp8_dispatch_backward:
            _launch(
                20,
                [
                    self.window_recv_x,
                    self.permutation,
                    self.packed_x,
                    self.current_rows,
                    self.status,
                ],
                [self.rows, self.h, self.recv_capacity, self.k],
            )

    def _restore_forward_chunk(self, x, ids, valid_tokens, start):
        self._activate_forward_context(start)
        rx, rw, handle = self.forward_context_receives[start // self.chunk_tokens]
        if isinstance(rx, tuple):
            # Reuse the exact forward quantization. Only original BF16 X for
            # DW must be dispatched, using the same cached receive ordering.
            self.window_recv_q, self.window_recv_qsf = rx
            _launch(
                13,
                [x, ids, self.local_x, self.local_ids, valid_tokens, self.status],
                [x.shape[0], self.chunk_tokens, self.h, self.k, start],
            )
            self.window_recv_x, _, _, _, _, event = self.deep_ep_buffer.dispatch(
                self.local_x, handle=handle, async_finish=True
            )
            event.current_stream_wait()
        else:
            self.window_recv_x = rx
        return rw, handle

    def _pack_received_fp8(self):
        _launch(
            31,
            [
                self.window_recv_q,
                self.window_recv_qsf,
                self.permutation,
                self.quantized[self.h][0],
                self.quantized_scale_storage[self.h],
                self.current_rows,
                self.status,
            ],
            [self.rows, self.h, self.recv_capacity, self.k],
        )

    def _zero_recv_accumulator(self, handle):
        if self.deepep_active_receive:
            _launch(
                37,
                [
                    self.recv_accumulator,
                    handle.psum_num_recv_tokens_per_scaleup_rank[-1:],
                    self.status,
                ],
                [self.recv_capacity, self.h],
            )
        else:
            self.recv_accumulator.zero_()

    def _reduce_window(self, rw, start, weighted):
        _launch(
            25,
            [
                self.packed_output,
                self.route_experts,
                self.route_rows,
                self.global_offsets,
                rw,
                self.recv_accumulator,
                self.current_rows,
                self.status,
            ],
            [
                self.rows,
                self.h,
                self.recv_capacity,
                self.k,
                self.e,
                start,
                int(weighted),
                self.full_row_capacity,
            ],
        )

    @torch.no_grad()
    def forward(self, x, ids, scores, w1, w2, valid_tokens):
        self._validate(x, ids, scores, w1, w2, valid_tokens)
        with self._execution():
            self._fp8_dispatch_backward = self._using_forward_context = False
            self._begin_forward_context()
            self.prepare(w1, w2)
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.reset_errors()
            # Weights may change in-place between graph replays or eager calls.
            # Match Native's dynamic FP8 recipe; quantization is inside timing.
            self._prepare_weights(w1, w2)
            if self.reuse_forward_context:
                self.forward_context_reused_windows.zero_()
                self.forward_context_reused_rows.zero_()
            output = torch.empty_like(x)
            for start in range(0, self.max_tokens, self.chunk_tokens):
                rw, handle = self._dispatch_chunk(x, ids, scores, valid_tokens, start)
                if self.ring_rows is None:
                    self._expert_forward(w1, w2)
                else:
                    self._zero_recv_accumulator(handle)
                    for window in range(0, self.full_row_capacity, self.rows):
                        self._set_window(window)
                        self._expert_forward(w1, w2)
                        self._reduce_window(rw, window, weighted=True)
                local_y, _ = self._combine(rw, handle, weighted=True)
                count = max(0, min(self.chunk_tokens, x.shape[0] - start))
                output[start : start + count].copy_(local_y[:count])
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.check()
            self._join_weight_quantization()
            if self.reuse_forward_context:
                self._record_forward_context_generation()
                self._forward_context_fingerprint = self._context_fingerprint(
                    x, ids, scores, w1, w2, valid_tokens
                )
                self._forward_context_available = True
            return output

    @torch.no_grad()
    def backward(self, dy, x, ids, scores, w1, w2, valid_tokens, accumulators=None):

        self._validate(x, ids, scores, w1, w2, valid_tokens)
        _validate_accumulators(accumulators, w1, w2)
        if (
            dy.shape != x.shape
            or dy.dtype != x.dtype
            or dy.device != x.device
            or not dy.is_contiguous()
        ):
            raise ValueError("dy must be contiguous and match x")
        with self._execution():
            self._using_forward_context = self._matching_forward_context(
                (x, ids, scores, w1, w2, valid_tokens)
            )
            self._consume_forward_context_generation()
            self.prepare(w1, w2)
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.reset_errors()
            self._prepare_weights(w1, w2, backward=True)
            dx, dscores = torch.empty_like(x), torch.empty_like(scores)
            dw1, dw2 = (
                (
                    torch.zeros_like(w1, dtype=torch.float32),
                    torch.zeros_like(w2, dtype=torch.float32),
                )
                if accumulators is None
                else accumulators
            )
            for start in range(0, self.max_tokens, self.chunk_tokens):
                rw, handle = (
                    self._restore_forward_chunk(x, ids, valid_tokens, start)
                    if self._using_forward_context
                    else self._dispatch_chunk(x, ids, scores, valid_tokens, start)
                )
                if self.ring_rows is None:
                    self._expert_forward(w1, w2)
                count = max(0, min(self.chunk_tokens, x.shape[0] - start))
                self.local_dy_input.zero_()
                self.local_dy_input[:count].copy_(dy[start : start + count])
                if self.overlap_expert_dy:
                    main = torch.cuda.current_stream(self.device)
                    self._gradient_stream.wait_stream(main)
                    with torch.cuda.stream(self._gradient_stream):
                        self.local_dy_input.record_stream(self._gradient_stream)
                        rdy, _, _, _, _, event = self.deep_ep_buffer.dispatch(
                            self.local_dy_input, handle=handle, async_finish=True
                        )
                        event.current_stream_wait()
                    rdy.record_stream(main)
                    self._gradient_prefetch_pending = True
                else:
                    rdy, _, _, _, _, event = self.deep_ep_buffer.dispatch(
                        self.local_dy_input, handle=handle, async_finish=True
                    )
                    event.current_stream_wait()
                self.recv_dscores.zero_()
                if self.ring_rows is None:
                    self._expert_backward_rows(rdy, rw, dw1, dw2)
                else:
                    self._zero_recv_accumulator(handle)
                    windows = range(0, self.full_row_capacity, self.rows)
                    for window in reversed(windows) if self._using_forward_context else windows:
                        self._set_window(window)
                        self._context_window_start = window
                        self._expert_forward(w1, w2)
                        self._expert_backward_rows(rdy, rw, dw1, dw2)
                        self._reduce_window(rw, window, weighted=False)
                local_dx, local_ds = self._combine(
                    rw, handle, weighted=False, dscores=self.recv_dscores
                )
                dx[start : start + count].copy_(local_dx[:count])
                dscores[start : start + count].copy_(local_ds[:count])
                self._join_wgrad()
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.check()
            self._join_weight_quantization()
            self._forward_context_available = self._using_forward_context = False
            return dx, dw1, dw2, dscores

    def _expert_backward_rows(self, rdy, rw, dw1, dw2):
        # Cached DY dispatch may progress during the local forward replay.
        # Join at its first actual consumer, after that replay has completed.
        self._join_gradient_prefetch()
        _launch(
            24,
            [
                rdy,
                rw,
                self.permutation,
                self.packed_output,
                self.packed_dy,
                self.recv_dscores,
                self.current_rows,
                self.status,
            ],
            [self.rows, self.h, self.recv_capacity, self.k],
        )
        self._wgrad(self.packed_dy, self.hidden, dw2, asynchronous=self.wgrad_overlap == "both")
        dhidden = self.hidden if self.backward_hidden is None else self.backward_hidden
        self._gemm(self.packed_dy, self.w2_t, dhidden, "w2_t")
        fused = self._activation(backward=True, hidden=dhidden)
        self._wgrad(self.gate_up, self.packed_x, dw1, asynchronous=self.wgrad_overlap != "none")
        self._gemm(self.gate_up, self.w1_t, self.packed_output, "w1_t", prequantized=fused)

    def close(self):
        if self.deep_ep_buffer is not None:
            torch.cuda.synchronize(self.device)
            self.deep_ep_buffer.destroy()
            self.deep_ep_buffer = None


def plan_compact_route_capacity(
    ids, valid_tokens, local_experts, chunk_tokens, group=None, alignment=128, headroom=1.25
):
    """Reserve aligned capacity from representative routes outside capture.

    New routes may be replayed within the reservation. A different distribution
    that exceeds it must be reserved again before capture, or use the windowed
    plan. The execution path never truncates expert rows to this reservation.
    """
    if ids.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("plan compact capacity before graph capture")
    if ids.ndim != 2 or ids.shape[1] == 0 or ids.dtype != torch.int64:
        raise ValueError("route IDs must be a two-dimensional int64 tensor")
    if (
        type(local_experts) is not int
        or local_experts <= 0
        or type(chunk_tokens) is not int
        or chunk_tokens <= 0
        or type(alignment) is not int
        or alignment <= 0
    ):
        raise ValueError("expert count, source chunk and alignment must be positive integers")
    if not math.isfinite(headroom) or headroom < 1:
        raise ValueError("compact capacity headroom must be finite and at least one")
    valid = int(valid_tokens.item()) if isinstance(valid_tokens, torch.Tensor) else valid_tokens
    if type(valid) is not int or not 0 <= valid <= ids.shape[0]:
        raise ValueError("valid token count is outside the source tensor")
    size = dist.get_world_size(group) if group is not None else 1
    experts = size * local_experts
    histogram_device = (
        torch.device("cpu")
        if group is not None and dist.get_backend(group) == "gloo"
        else ids.device
    )
    maximum = torch.zeros((), device=histogram_device, dtype=torch.int64)
    per_chunk = []
    for start in range(0, ids.shape[0], chunk_tokens):
        flat = ids[start : min(start + chunk_tokens, valid)].reshape(-1)
        if bool(((flat < -1) | (flat >= experts)).any()):
            raise ValueError("route expert ID is outside the EP group")
        flat = flat[flat >= 0].to(histogram_device)
        histogram = torch.zeros(experts, device=histogram_device, dtype=torch.int64)
        histogram.scatter_add_(0, flat, torch.ones_like(flat))
        if group is not None:
            dist.all_reduce(histogram, group=group)
        padded = (
            (histogram.reshape(size, local_experts) + alignment - 1) // alignment * alignment
        ).sum(dim=1)
        per_chunk.append(padded)
        maximum = torch.maximum(maximum, padded.max())
    required = int(maximum.item())
    # Empty representative routing must still reserve the grouped execution
    # family floor enforced by CompactDeepEPMoEPlan, including E > alignment.
    minimum = math.ceil(local_experts / alignment) * alignment
    worst = max(
        minimum,
        math.ceil(
            (size * chunk_tokens * ids.shape[1] + local_experts * (alignment - 1)) / alignment
        )
        * alignment,
    )
    reserved = min(worst, max(minimum, math.ceil(required * headroom / alignment) * alignment))
    return dict(
        capacity=reserved,
        representative_max_padded_rows=required,
        representative_padded_rows_per_chunk=(
            torch.stack(per_chunk).cpu().tolist() if per_chunk else []
        ),
        alignment=alignment,
        headroom=headroom,
        worst_case_rows=worst,
    )


class CompactDeepEPMoEPlan(DeepEPMoEPlan):
    """Single compact permutation, expert computation, and active unpermutation.

    ``ring_rows`` still describes the reference windowed plan. Its final-window
    last-active-window reuse boundary is preserved exactly; compact storage must not
    accidentally eliminate the original backward recomputation.
    """

    deepep_layout = "compact"

    def __init__(
        self, hidden, intermediate, local_experts, topk, max_tokens, *, compact_rows, **kwargs
    ):
        if type(compact_rows) is not int or compact_rows <= 0:
            raise ValueError("compact_rows must be a positive integer")
        if compact_rows < local_experts:
            raise ValueError("compact capacity must retain the grouped GEMM execution family")
        if kwargs.get("wgrad_overlap", "none") != "none":
            raise ValueError("the matched compact baseline requires serial DW")
        if "packed_capacity" in kwargs:
            raise ValueError("compact_rows owns the compact workspace capacity")
        reference_rows = kwargs.pop("ring_rows", None)
        if reference_rows is not None and (type(reference_rows) is not int or reference_rows <= 0):
            raise ValueError("reference ring_rows must be a positive integer")
        if reference_rows is not None and reference_rows < local_experts:
            raise ValueError("the reference workspace must use grouped GEMM rather than GEMV")
        alignment = math.lcm(kwargs.get("block_m", 128), dg.get_m_alignment_for_contiguous_layout())
        if reference_rows is not None and reference_rows % alignment:
            raise ValueError("reference ring_rows must match the original alignment")
        kwargs.update(ring_rows=compact_rows, packed_capacity=compact_rows)
        super().__init__(hidden, intermediate, local_experts, topk, max_tokens, **kwargs)
        self.compact_rows = compact_rows
        self.reference_window_rows = reference_rows
        self.reference_recompute_policy = (
            "last_active_window" if reference_rows is not None else "full_replay"
        )
        self.deepep_layout = "compact"
        self.reference_gemm_rows = (
            min(reference_rows, self.unbounded_row_capacity)
            if reference_rows is not None
            else self.unbounded_row_capacity
        )
        shapes = dict(
            fc1=(2 * self.i, self.h),
            fc2=(self.h, self.i),
            dhidden=(self.i, self.h),
            dx=(self.h, 2 * self.i),
        )
        self.resolved_gemm_configs = {
            role: dg._C._mega_moe_training_nopad_config(
                self.device.index, self.compute, self.reference_gemm_rows, n, k, self.e
            )
            for role, (n, k) in shapes.items()
            if role not in self.gemm_configs
        }

    def _prepare_local(self, w1, w2):
        super()._prepare_local(w1, w2)
        if self.compute == "bf16":
            self._wgrad(self.gate_up, self.packed_x, torch.zeros_like(w1, dtype=torch.float32))
            self._wgrad(self.packed_dy, self.hidden, torch.zeros_like(w2, dtype=torch.float32))
            torch.cuda.synchronize(self.device)

    def _wgrad(self, left, right, output, *, asynchronous=False):
        if self.fp8_wgrad:
            return super()._wgrad(left, right, output, asynchronous=asynchronous)
        if asynchronous:
            raise RuntimeError("matched compact BF16 DW is serial")
        dg.k_grouped_bf16_gemm_tn_contiguous(
            left,
            right,
            output,
            None,
            self.lengths,
            c=output,
            max_group_k=self.rows,
            config_k=self.reference_gemm_rows,
        )

    def _allocate_receive_accumulator(self, capacity):
        # Direct unpermutation writes BF16, without an F32 receive accumulator.
        return None

    def _prefix_received_routes(self, offsets, lengths, capacity):
        _launch(
            40, [self.counts, offsets, lengths, self.status], [self.e, self.alignment, capacity]
        )

    def _prepare_pointwise(self):
        super()._prepare_pointwise()
        dg._C._mega_moe_training_precompile(self.device.index, [39, 40, 41])

    def _zero_recv_accumulator(self, handle):
        pass

    def _reduce_window(self, rw, start, weighted):
        if start != 0:
            raise RuntimeError("compact expert rows must occupy one window")

    def _combine(self, rw, handle, weighted, dscores=None):
        _launch(
            39,
            [
                self.packed_output,
                self.route_experts,
                self.route_rows,
                self.global_offsets,
                rw,
                self.reduced,
                handle.psum_num_recv_tokens_per_scaleup_rank[-1:],
                self.status,
            ],
            [self.rows, self.h, self.recv_capacity, self.k, self.e, int(weighted)],
        )
        output, ds, event = self.deep_ep_buffer.combine(
            self.reduced, handle, topk_weights=dscores, async_finish=True
        )
        event.current_stream_wait()
        return output, ds

    def _expert_forward(self, w1, w2, _context_masked=False):
        if not self._using_forward_context or _context_masked:
            return super()._expert_forward(w1, w2, _context_masked=True)
        _launch(
            41,
            [
                self.global_offsets,
                self.global_lengths,
                self.replay_offsets,
                self.replay_lengths,
                self.forward_context_reused_windows,
                self.forward_context_reused_rows,
                self.status,
            ],
            [self.e, self.rows, self.reference_window_rows or 0],
        )
        original = self.offsets, self.lengths, self.current_rows
        self.offsets, self.lengths = self.replay_offsets, self.replay_lengths
        self.current_rows = self.replay_offsets[self.e : self.e + 1]
        try:
            return super()._expert_forward(w1, w2, _context_masked=True)
        finally:
            self.offsets, self.lengths, self.current_rows = original
