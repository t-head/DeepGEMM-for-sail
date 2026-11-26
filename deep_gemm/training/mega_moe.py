"""Native peer-read/peer-write Tensor Cell GEMMs for PPU MoE training."""

import math
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


def _mega_moe(operation, tensors, dims, compile_only=False):
    dg._C._mega_moe_training_launch(operation, tensors, dims, compile_only)


class MegaMoEPlan(_MoEPlanBase):
    """BF16/FP8 MegaMoE training with fused dispatch, GEMMs and return stores.

    Requires a direct peer domain, including cross-host ICN within a supernode.
    """

    transport = "peer"

    def __init__(
        self,
        *args,
        gemm_configs=None,
        fuse_dispatch=True,
        fuse_return=True,
        active_rows=True,
        fuse_local_grad=True,
        fuse_activation_backward=False,
        vectorized_copy=True,
        reuse_packed_dy=False,
        dispatch_cache_ctas=2,
        dispatch_producer=False,
        dispatch_producer_tiles=1,
        producer_local_aiu=False,
        producer_input_aiu=False,
        pipeline_comm_ctas=2,
        expert_local_grad=False,
        ring_rows=None,
        route_capacity=None,
        return_vector_bytes=4,
        input_copy_ilp=1,
        fuse_fc1_activation=False,
        dgrad_compute_cus=0,
        fuse_dhidden_scores=False,
        transpose_free_dgrad=False,
        fp8_weight_quantization=None,
        fp8_input_ctas=None,
        fp8_input_threads=None,
        reuse_forward_context=None,
        **kwargs,
    ):
        self.training_fp8_mma_flags = int(os.environ.get("DG_JIT_FP8_TRAINING_MMA_FLAGS", "0"))
        self.training_bf16_mma_reuse = int(os.environ.get("DG_JIT_BF16_TRAINING_MMA_REUSE", "0"))
        if self.training_fp8_mma_flags not in (0, 1) or self.training_bf16_mma_reuse not in (0, 1):
            raise ValueError("training native MMA policies must be0or1")
        self._shared_fp8_weights_ready = False
        self._pending_last_dw1 = None
        self._defer_last_dw1_enabled = False
        # Optional CTA helper recipe; environment values allow the same large
        # reference and capacity harnesses to exercise the exact candidate.

        def helper_value(value, key, default):
            value = int(os.environ.get(key, str(default))) if value is None else value
            if type(value) is not int:
                raise TypeError(key + " must be an integer")
            return value

        target_device = kwargs.get("device", None)
        max_cus = torch.cuda.get_device_properties(target_device).multi_processor_count
        if not 0 <= 0 <= max_cus:
            raise ValueError("tail DW compute budget is outside the device CU range")
        if not isinstance(transpose_free_dgrad, bool):
            raise TypeError("transpose_free_dgrad must be bool")
        if transpose_free_dgrad and (
            not expert_local_grad
            or not reuse_packed_dy
            or (not fuse_dispatch)
            or (not fuse_return)
            or (not fuse_activation_backward)
            or (ring_rows is None)
        ):
            raise ValueError(
                "transpose-free dGrad requires routed bilateral expert-local gradients"
            )
        self.transpose_free_dgrad = transpose_free_dgrad
        if not isinstance(fuse_dhidden_scores, bool):
            raise TypeError("fuse_dhidden_scores must be bool")
        if fuse_dhidden_scores and (
            not expert_local_grad
            or not reuse_packed_dy
            or (not fuse_dispatch)
            or (not fuse_return)
            or (not fuse_activation_backward)
            or (not fuse_fc1_activation)
            or (kwargs.get("gate_up_layout", "split") != "g8")
            or (ring_rows is None)
            or kwargs.get("overlap_expert_dy", False)
        ):
            raise ValueError(
                "dHidden scores require G8 bilateral routed expert gradients with fused activations"
            )
        if fuse_dhidden_scores and (gemm_configs is None or not gemm_configs.get("dhidden")):
            raise ValueError("dHidden scores require an explicit dHidden GEMM configuration")
        self.fuse_dhidden_scores = fuse_dhidden_scores
        raw_grad_producer = os.environ.get("DG_MOE_BF16_GRAD_PRODUCER", "0")
        if raw_grad_producer not in ("0", "1"):
            raise ValueError("BF16 gradient producer flag must be0or1")
        self.bf16_grad_producer = raw_grad_producer == "1"
        raw_first = os.environ.get("DG_MOE_WGRAD_FIRST_NONEMPTY", "0")
        if raw_first not in ("0", "1"):
            raise ValueError("first nonempty DW must be0or1")
        self.wgrad_first_nonempty = raw_first == "1"
        self._fresh_first_dw_step = False
        raw_overwrite = os.environ.get("DG_MOE_WGRAD_FIRST_OVERWRITE", "0")
        if raw_overwrite not in ("0", "1"):
            raise ValueError("DW first overwrite must be0or1")
        if self.wgrad_first_nonempty and (
            ring_rows is None
            or not expert_local_grad
            or kwargs.get("wgrad_overlap", "none") != "none"
        ):
            raise ValueError(
                "first nonempty DW requires bounded serial gradients and no other initialization policy"
            )
        self._first_dw_outputs = set()
        if self.bf16_grad_producer and (
            kwargs.get("compute", "bf16") != "bf16"
            or not fuse_dhidden_scores
            or (not transpose_free_dgrad)
            or (ring_rows is None)
            or (not expert_local_grad)
            or (not reuse_packed_dy)
        ):
            raise ValueError(
                "BF16 gradient producer requires bounded BF16 score/activation fusion and native transposed dGrad"
            )
        if type(dgrad_compute_cus) is not int or dgrad_compute_cus < 0:
            raise ValueError("dgrad_compute_cus must be a nonnegative integer")
        self.dgrad_compute_cus = dgrad_compute_cus
        if kwargs.get("overlap_expert_dy", False) and (
            not dispatch_producer or not expert_local_grad or ring_rows is None
        ):
            raise ValueError(
                "DY prefetch requires the producer expert-gradient routed-window schedule"
            )
        if kwargs.get("wgrad_overlap") == "both" and (not fuse_activation_backward):
            raise ValueError(
                "DW2 overlap requires the fused dHidden epilogue to preserve forward Hidden"
            )
        if type(input_copy_ilp) is not int or input_copy_ilp not in (1, 2, 4, 8):
            raise ValueError("input_copy_ilp must be1,2,4or8")
        if input_copy_ilp != 1 and (not (dispatch_producer and producer_input_aiu)):
            raise ValueError("input copy ILP requires producer input staging")
        self.input_copy_ilp = input_copy_ilp
        if not isinstance(fuse_fc1_activation, bool):
            raise TypeError("fuse_fc1_activation must be bool")
        g8 = kwargs.get("gate_up_layout", "split") == "g8"
        if g8 and self.transport != "peer":
            raise ValueError("G8 requires direct peer expert execution")
        if fuse_fc1_activation and (
            not g8 or not expert_local_grad or (not fuse_dispatch) or (not fuse_return)
        ):
            raise ValueError(
                "FC1 activation fusion requires G8, bilateral fusion and expert-local gradients"
            )
        self.fuse_fc1_activation = fuse_fc1_activation
        if type(return_vector_bytes) is not int or return_vector_bytes not in (4, 8):
            raise ValueError("return_vector_bytes must be 4 or 8")
        if return_vector_bytes != 4 and (not fuse_return):
            raise ValueError("eight-byte return stores require fused return")
        self.return_vector_bytes = return_vector_bytes
        self.bf16_return_ncp_mode = int(os.environ.get("DG_MOE_BF16_RETURN_NCP_MODE", "0"))
        if self.bf16_return_ncp_mode not in (0, 2):
            raise ValueError("BF16 return NCP mode must0or2")
        if self.bf16_return_ncp_mode and (
            kwargs.get("compute", "bf16") != "bf16" or not fuse_return or return_vector_bytes != 4
        ):
            raise ValueError("BF16 NCP requires ordinary fused return with uncached destinations")
        block_m = kwargs.get("block_m", 128)
        if block_m != 128 and self.transport != "peer":
            raise ValueError("BM256 requires direct peer expert execution")
        if "packed_rows" in kwargs:
            raise ValueError("use ring_rows to enable routed-row scheduling")
        if ring_rows is not None:
            if not expert_local_grad:
                raise ValueError("routed-row windows require expert-local gradients")
        self.ring_rows = ring_rows
        self.fp8_peer_compute = kwargs.get("compute", "bf16") == "fp8"

        def fp8_flag(value, key):
            if value is None:
                raw = os.environ.get(key, "0")
                if raw not in ("0", "1"):
                    raise ValueError(key + " must be0or1")
                value = raw == "1"
            if type(value) is not bool:
                raise TypeError(key + " must be bool")
            return value

        self._fp8_weights_pending = False
        self.wgrad_aggregate_windows = fp8_flag(None, "DG_MOE_WGRAD_AGGREGATE_WINDOWS")
        self.wgrad_aggregate_tile = helper_value(None, "DG_MOE_WGRAD_AGGREGATE_TILE", 0)
        if self.wgrad_aggregate_tile not in (0,):
            raise ValueError("aggregate DW tile must be0or1")
        self.wgrad_aggregate_last_segments = 0
        self._aggregate_step_enabled = self._aggregate_window_active = False
        self._aggregate_dw_lists = {}
        batch_default = helper_value(None, "DG_MOE_FP8_EXPERT_BATCH", 0)
        self.fp8_expert_forward_batch = helper_value(
            None, "DG_MOE_FP8_EXPERT_FWD_BATCH", batch_default
        )
        self.fp8_expert_gradient_batch = helper_value(
            None, "DG_MOE_FP8_EXPERT_GRAD_BATCH", batch_default
        )
        self.fp8_expert_source_ctas = helper_value(
            None, "DG_MOE_FP8_EXPERT_SOURCE_CTAS", 16 * max_cus
        )
        self.fp8_expert_forward_ctas = helper_value(
            None, "DG_MOE_FP8_EXPERT_FWD_CTAS", self.fp8_expert_source_ctas
        )
        self.fp8_expert_gradient_ctas = helper_value(
            None, "DG_MOE_FP8_EXPERT_GRAD_CTAS", self.fp8_expert_source_ctas
        )
        if not all(
            (
                0 < v <= 16 * max_cus
                for v in (self.fp8_expert_forward_ctas, self.fp8_expert_gradient_ctas)
            )
        ):
            raise ValueError("expert source per-direction CTA count must fit hardware")
        self.fp8_expert_source_rows = helper_value(None, "DG_MOE_FP8_EXPERT_SOURCE_ROWS", 16)
        if any(
            (
                not 0 <= v <= 64
                for v in (self.fp8_expert_forward_batch, self.fp8_expert_gradient_batch)
            )
        ):
            raise ValueError("expert batch sizes must be0..64")
        self.fp8_expert_pipeline = bool(
            self.fp8_expert_forward_batch or self.fp8_expert_gradient_batch
        )
        if (
            not 0 < self.fp8_expert_source_ctas <= 16 * max_cus
            or self.fp8_expert_source_rows not in (16, 32, 64, 128)
        ):
            raise ValueError("expert source requires1..16CTA/CU and16/32/64/128rows")
        self.fp8_expert_source_fp8 = fp8_flag(None, "DG_MOE_FP8_EXPERT_SOURCE_FP8")
        self.fp8_source_shared_base = fp8_flag(None, "DG_MOE_FP8_SOURCE_SHARED_BASE")
        self.fp8_dw_from_row_cache = fp8_flag(None, "DG_MOE_FP8_DW_FROM_ROW_CACHE")
        if self.fp8_source_shared_base and (not self.fp8_expert_source_fp8):
            raise ValueError("shared source scales require source FP8 expert batches")
        if self.fp8_dw_from_row_cache and (not self.fp8_source_shared_base):
            raise ValueError("row-FP8 DW reuse requires shared current-input source scales")
        self.fp8_source_refetch_phase = os.environ.get("DG_MOE_FP8_SOURCE_REFETCH_PHASE", "early")
        if self.fp8_source_refetch_phase not in ("early", "dw2", "serial"):
            raise ValueError("source BF16 X refetch phase must be early, dw2 or serial")
        self.fp8_raw_gradient_quant_pipeline = fp8_flag(
            None, "DG_MOE_FP8_RAW_GRADIENT_QUANT_PIPELINE"
        )
        self.fp8_raw_gradient_source_prefetch = fp8_flag(
            None, "DG_MOE_FP8_RAW_GRADIENT_SOURCE_PREFETCH"
        )
        self.fp8_raw_gradient_source_grid = int(
            os.environ.get("DG_MOE_FP8_RAW_GRADIENT_SOURCE_GRID", "0")
        )
        if not 0 <= self.fp8_raw_gradient_source_grid <= dg.get_num_sms() * 16:
            raise ValueError("source grid must0(default)or1..16CTA/SM")
        self.fp8_raw_gradient_tile_ilp = int(
            os.environ.get("DG_MOE_FP8_RAW_GRADIENT_TILE_ILP", "2")
        )
        if self.fp8_raw_gradient_tile_ilp not in (1, 2, 4, 8, 16, 32):
            raise ValueError("source tile ILP must1,2,4,8,16or32")
        self.fp8_raw_gradient_score_ilp = int(
            os.environ.get("DG_MOE_FP8_RAW_GRADIENT_SCORE_ILP", "4")
        )
        if self.fp8_raw_gradient_score_ilp not in (1, 2, 4):
            raise ValueError("score ILP must1,2or4")
        self.fp8_raw_gradient_score_threads = int(
            os.environ.get("DG_MOE_FP8_RAW_GRADIENT_SCORE_THREADS", "256")
        )
        if not 0 <= 0 <= dg.get_num_sms() * 16 or self.fp8_raw_gradient_score_threads not in (
            128,
            256,
        ):
            raise ValueError("score grid0(default)or1..16CTA/SM and128/256threads required")
        self._fp8_raw_quant_pending = False
        self.fp8_raw_gradient_quant_ctas = int(
            os.environ.get("DG_MOE_FP8_RAW_GRADIENT_QUANT_CTAS", "0")
        )
        self.fp8_raw_gradient_quant_threads = int(
            os.environ.get("DG_MOE_FP8_RAW_GRADIENT_QUANT_THREADS", "256")
        )
        if (
            not 0 <= self.fp8_raw_gradient_quant_ctas <= dg.get_num_sms() * 16
            or self.fp8_raw_gradient_quant_threads not in (128, 256)
        ):
            raise ValueError("owner quant grid0(default)or1..16CTA/SM and128/256threads required")
        self.fp8_raw_gradient_score_overlap = fp8_flag(
            None, "DG_MOE_FP8_RAW_GRADIENT_SCORE_OVERLAP"
        )
        self._fp8_raw_score_pending = False
        self.fp8_raw_gradient_tile_pipeline = fp8_flag(
            None, "DG_MOE_FP8_RAW_GRADIENT_TILE_PIPELINE"
        )
        self.fp8_raw_gradient_source = int(os.environ.get("DG_MOE_FP8_RAW_GRADIENT_SOURCE", "0"))
        if self.fp8_raw_gradient_source not in (0, 1, 2, 3, 4, 7, 8):
            raise ValueError(
                "raw gradient source:0 original,1 paired,2 paired BF16 ref,3 common base,4 common BF16 ref,7 byte/tinySF,8 matching BF16/tinySF"
            )
        self.fp8_raw_gradient_source_threads = helper_value(
            None,
            "DG_MOE_FP8_RAW_GRADIENT_SOURCE_THREADS",
            256 if self.fp8_raw_gradient_source >= 3 else 128,
        )
        self.fp8_raw_gradient_source_ctas = helper_value(
            None, "DG_MOE_FP8_RAW_GRADIENT_SOURCE_CTAS", 4
        )
        if self.fp8_raw_gradient_source_threads not in (
            128,
            256,
        ) or self.fp8_raw_gradient_source_ctas not in (1, 2, 4, 8, 16):
            raise ValueError("raw source threads must128/256 and CTA multiplier1/2/4/8/16")
        self.fp8_raw_gradient_tile_ctas = helper_value(None, "DG_MOE_FP8_RAW_GRADIENT_TILE_CTAS", 1)
        self.fp8_raw_gradient_tile_threads = helper_value(
            None, "DG_MOE_FP8_RAW_GRADIENT_TILE_THREADS", 128
        )
        if self.fp8_raw_gradient_tile_ctas not in (
            1,
            2,
            4,
            8,
            16,
        ) or self.fp8_raw_gradient_tile_threads not in (64, 96, 128, 256, 512, 1024):
            raise ValueError("raw tile source CTA multiplier1/2/4/8/16 and threads128/256 required")
        self.fp8_raw_gradient_gemm_ctas = int(
            os.environ.get("DG_MOE_FP8_RAW_GRADIENT_GEMM_CTAS", "0")
        )
        if not 0 <= self.fp8_raw_gradient_gemm_ctas <= dg.get_num_sms() * 2:
            raise ValueError("raw GEMM metadata CTA cap must0(default)or1..2SM")
        raw_config = os.environ.get("DG_MOE_FP8_RAW_GRADIENT_GEMM_CONFIG", "")
        self.fp8_raw_gradient_gemm_config = (
            tuple(map(int, raw_config.split(","))) if raw_config else None
        )
        if self.fp8_raw_gradient_gemm_config is not None:
            cfg = self.fp8_raw_gradient_gemm_config
            if (
                not self.fp8_raw_gradient_source
                or len(cfg) != 6
                or cfg[0] not in (64, 128, 192, 256)
                or (cfg[1] not in (128, 256))
                or (cfg[2] != 128)
                or (cfg[3] <= 0)
                or (cfg[4] <= 0)
                or cfg[0] % cfg[3]
                or cfg[1] % cfg[4]
                or (cfg[5] not in (3, 4))
            ):
                raise ValueError("raw dHidden GEMM requires BM,BN,K128,WM,WN,stage3/4")
        self._fp8_raw_gradient_active = 0
        self.fp8_raw_gradient_last_active = 0
        self.fp8_source_prep_reuse = int(os.environ.get("DG_MOE_FP8_SOURCE_PREP_REUSE", "0"))
        if self.fp8_source_prep_reuse not in (0, 1, 2, 3):
            raise ValueError("source preparation reuse must be0,1,2or3")
        if self.fp8_expert_source_fp8 and (
            not self.fp8_peer_compute
            or not self.fp8_expert_forward_batch
            or ring_rows is None
            or (not expert_local_grad)
            or (not self.fp8_expert_gradient_batch)
        ):
            raise ValueError("sender FP8 requires bounded FP8 forward/backward expert batches")
        self.fp8_input_slices = helper_value(None, "DG_MOE_FP8_INPUT_SLICES", 1)
        if self.fp8_input_slices not in (1, 2, 4, 8, 16):
            raise ValueError("FP8 input slices must be1,2,4,8or16")
        if self.fp8_input_slices != 1:
            raise ValueError("sliced input requires the BF16 quantized producer")
        self.fp8_inline_bulk_ilp = int(os.environ.get("DG_MOE_FP8_INLINE_SOURCE_BULK_ILP", "4"))
        self.fp8_raw_bulk_ilp = int(os.environ.get("DG_MOE_FP8_RAW_GRADIENT_BULK_ILP", "4"))
        if self.fp8_inline_bulk_ilp not in (4, 8, 16) or self.fp8_raw_bulk_ilp not in (4, 8, 16):
            raise ValueError("bulk source ILP must be4,8,16")
        self.fp8_inline_source_bulk = int(os.environ.get("DG_MOE_FP8_INLINE_SOURCE_BULK", "0"))
        self.fp8_raw_gradient_source_bulk = int(
            os.environ.get("DG_MOE_FP8_RAW_GRADIENT_SOURCE_BULK", "0")
        )
        if self.fp8_inline_source_bulk not in (0, 1) or self.fp8_raw_gradient_source_bulk not in (
            0,
            1,
        ):
            raise ValueError("invalid FP8 bulk input mode")
        self.fp8_return_remote_mode = int(os.environ.get("DG_MOE_FP8_RETURN_REMOTE_MODE", "0"))
        if self.fp8_return_remote_mode not in (0, 1, 2, 3, 4):
            raise ValueError("invalid FP8 remote return mode")
        self.fp8_return_pipeline_mma_flags = int(
            os.environ.get("DG_JIT_FP8_RETURN_PIPELINE_MMA_FLAGS", "1")
        )
        if (
            0 not in (0, 224, 232, 240, 248)
            or 0 not in (0, 128, 192, 256)
            or self.fp8_return_pipeline_mma_flags not in (0, 1)
        ):
            raise ValueError("invalid native return resource configuration")
        self.fp8_direct_return_configs = {}
        for role in ("fc2", "dx"):
            value = os.environ.get("DG_MOE_FP8_DIRECT_RETURN_" + role.upper() + "_CONFIG", "")
            if value:
                cfg = tuple(map(int, value.split(",")))
                if (
                    len(cfg) != 6
                    or cfg[0] not in (64, 128, 192, 256)
                    or cfg[1] <= 0
                    or (cfg[2] != 128)
                    or (min(cfg[3:]) <= 0)
                ):
                    raise ValueError("direct return config requires BM,BN,128,WM,WN,stages")
                self.fp8_direct_return_configs[role] = cfg
        if self.fp8_direct_return_configs and (not self.fp8_peer_compute or not fuse_return):
            raise ValueError(
                "direct return geometry requires fused FP8 return without output pipeline"
            )
        self.fp8_return_ctas = helper_value(None, "DG_MOE_FP8_RETURN_CTAS", 4)
        self.fp8_return_threads = helper_value(None, "DG_MOE_FP8_RETURN_THREADS", 512)
        self.fp8_return_store_bytes = helper_value(None, "DG_MOE_FP8_RETURN_STORE_BYTES", 4)
        if self.fp8_return_store_bytes not in (4, 16):
            raise ValueError("FP8 return store width must be4or16bytes")
        if self.fp8_return_store_bytes != 4 and (not self.fp8_peer_compute or not fuse_return):
            raise ValueError("wide FP8 return stores require direct fused FP8 return")
        if self.fp8_return_threads not in (64, 128, 256, 512, 1024):
            raise ValueError("FP8 return copy threads must be256,512or1024")
        stage_quant = os.environ.get("DG_MOE_FP8_FUSED_STAGE_QUANT", "0")
        if stage_quant not in ("0", "1", "2", "3"):
            raise ValueError("DG_MOE_FP8_FUSED_STAGE_QUANT must be0,1,2or3")
        self.fp8_fused_stage_quant = int(stage_quant)
        tiled_mode = os.environ.get("DG_MOE_FP8_TILED_GRAD_QUANT", "0")
        if tiled_mode not in ("0", "1", "2"):
            raise ValueError("DG_MOE_FP8_TILED_GRAD_QUANT must be0,1or2")
        self.fp8_tiled_gradient_quant = tiled_mode != "0"
        self.fp8_tiled_gradient_cache = tiled_mode == "2"
        if self.fp8_tiled_gradient_quant and (not self.fp8_fused_stage_quant & 2):
            raise ValueError("tiled gradient quantization requires fused DY quantization")
        if self.fp8_fused_stage_quant and (
            not self.fp8_peer_compute or not expert_local_grad or kwargs.get("block_m", 128) != 128
        ):
            raise ValueError(
                "fused stage quantization requires staged FP8 M128 expert-local gradients"
            )
        self.reuse_forward_context = fp8_flag(reuse_forward_context, "DG_MOE_REUSE_FORWARD_CONTEXT")
        if self.fp8_source_prep_reuse and (
            not (self.fp8_peer_compute and self.reuse_forward_context)
        ):
            raise ValueError("source preparation reuse requires live FP8 forward contexts")
        if self.reuse_forward_context and (
            ring_rows is None
            or not expert_local_grad
            or (not self.fp8_peer_compute and (not fuse_dhidden_scores))
        ):
            raise ValueError(
                "forward context reuse requires routed expert-local FP8 or BF16 dHidden-score training"
            )
        self._forward_context_available = self._using_forward_context = False
        self._forward_context_fingerprint = None
        self.fp8_input_ctas = helper_value(fp8_input_ctas, "DG_MOE_FP8_INPUT_CTAS", 8)
        self.fp8_input_threads = helper_value(fp8_input_threads, "DG_MOE_FP8_INPUT_THREADS", 128)
        if self.fp8_input_threads not in (128, 256, 512):
            raise ValueError("FP8 producer threads must be128,256or512")
        self._fp8_backward = self._fp8_bf_input_pending = False
        if self.fp8_peer_compute and (
            self.transport != "peer"
            or fuse_dispatch
            or (not active_rows)
            or (not vectorized_copy)
            or fuse_activation_backward
            or dispatch_producer
            or producer_local_aiu
            or producer_input_aiu
            or fuse_fc1_activation
            or dgrad_compute_cus
            or fuse_dhidden_scores
            or transpose_free_dgrad
            or (input_copy_ilp != 1)
            or (return_vector_bytes != 4)
            or (kwargs.get("block_m", 128) != 128)
            or (kwargs.get("gate_up_layout", "split") != "split")
            or (kwargs.get("wgrad_overlap", "none") != "none")
            or kwargs.get("overlap_expert_dy", False)
            or (not kwargs.get("fuse_activation_quant", True))
        ):
            raise ValueError(
                "FP8 peer compute requires staged input, active M128/split quantization and separate backward kernels"
            )
        if fp8_weight_quantization is None:
            raw = (
                os.environ.get("DG_MOE_FP8_WEIGHT_QUANTIZER", "2") if self.fp8_peer_compute else "0"
            )
            if raw not in ("0", "1", "2"):
                raise ValueError("DG_MOE_FP8_WEIGHT_QUANTIZER must be0,1or2")
            fp8_weight_quantization = int(raw)
        if type(fp8_weight_quantization) is not int or fp8_weight_quantization not in (0, 1, 2):
            raise ValueError("fp8_weight_quantization must be0,1or2")
        if fp8_weight_quantization and (not self.fp8_peer_compute):
            raise ValueError("FP8 weight quantization requires FP8 peer compute")
        self.fp8_weight_quantization = fp8_weight_quantization
        self.fp8_active_quantization = self.fp8_mn_major_scales = self.fp8_peer_compute
        self.communication_dtype = "bf16"
        self.fp8_input_mode = "staged_bf16_then_row_quantization" if self.fp8_peer_compute else None
        if self.fp8_fused_stage_quant & 1:
            self.fp8_input_mode = "staged_bf16_with_fused_row_quantization"
        self._fp8_prequantized_input = None
        if not isinstance(fuse_dispatch, bool) or not isinstance(fuse_return, bool):
            raise TypeError("peer fusion switches must be bool")
        self.fuse_dispatch, self.fuse_return = (fuse_dispatch, fuse_return)
        if not isinstance(expert_local_grad, bool):
            raise TypeError("expert_local_grad must be bool")
        if expert_local_grad and (not reuse_packed_dy or not fuse_local_grad):
            raise ValueError(
                "expert-local gradients require packed DY reuse and local-gradient fusion without whole-FFN fusion"
            )
        self.expert_local_grad = expert_local_grad
        if not isinstance(dispatch_producer, bool):
            raise TypeError("dispatch_producer must be bool")
        if dispatch_producer and (not fuse_dispatch or self.transport != "peer"):
            raise ValueError(
                "producer dispatch requires fused peer dispatch without a private N cache"
            )
        if type(dispatch_producer_tiles) is not int or dispatch_producer_tiles not in (1, 2, 4):
            raise ValueError("dispatch_producer_tiles must be 1, 2 or 4")
        self.dispatch_producer, self.dispatch_producer_tiles = (
            dispatch_producer,
            dispatch_producer_tiles,
        )
        if not isinstance(producer_local_aiu, bool):
            raise TypeError("producer_local_aiu must be bool")
        if producer_local_aiu and (not dispatch_producer):
            raise ValueError("local AIU input requires producer dispatch")
        self.producer_local_aiu = producer_local_aiu
        if not isinstance(producer_input_aiu, bool):
            raise TypeError("producer_input_aiu must be bool")
        if producer_input_aiu and (not dispatch_producer or dispatch_producer_tiles != 1):
            raise ValueError("producer input AIU requires exactly one producer N tile")
        self.producer_input_aiu = producer_input_aiu
        self.bf16_source_owned_tiles = int(os.environ.get("DG_MOE_BF16_SOURCE_OWNED_TILES", "0"))
        if self.bf16_source_owned_tiles not in (0, 1, 2, 3):
            raise ValueError("BF16 owned source mode must be 0, 1, 2 or 3")
        if self.bf16_source_owned_tiles and (
            kwargs.get("compute", "bf16") != "bf16"
            or not dispatch_producer
            or not producer_input_aiu
            or (self.bf16_source_owned_tiles & 2 and not self.bf16_grad_producer)
        ):
            raise ValueError("owned source tiles require BF16 AIU producers")
        if type(pipeline_comm_ctas) is not int or pipeline_comm_ctas < 1:
            raise ValueError("pipeline_comm_ctas must be a positive integer")
        self.pipeline_comm_ctas = pipeline_comm_ctas
        if not isinstance(active_rows, bool) or not isinstance(fuse_local_grad, bool):
            raise TypeError("peer pointwise switches must be bool")
        self.active_rows, self.fuse_local_grad = (active_rows, fuse_local_grad)
        if not isinstance(fuse_activation_backward, bool):
            raise TypeError("fuse_activation_backward must be bool")
        self.fuse_activation_backward = fuse_activation_backward
        if not isinstance(vectorized_copy, bool):
            raise TypeError("vectorized_copy must be bool")
        self.vectorized_copy = vectorized_copy
        if type(dispatch_cache_ctas) is not int or dispatch_cache_ctas not in (1, 2, 4):
            raise ValueError("dispatch_cache_ctas must be 1, 2 or 4")
        self.dispatch_cache_ctas = dispatch_cache_ctas
        if not isinstance(reuse_packed_dy, bool):
            raise TypeError("reuse_packed_dy must be bool")
        if reuse_packed_dy and self.transport != "peer":
            raise ValueError("packed gradient reuse is a peer-plan option")
        self.reuse_packed_dy = reuse_packed_dy and (fuse_dispatch or self.fp8_peer_compute)
        self.skip_masked_route_zero = fp8_flag(None, "DG_MOE_SKIP_MASKED_ROUTE_ZERO")
        if self.skip_masked_route_zero and (
            not fuse_return or ring_rows is None or (not expert_local_grad)
        ):
            raise ValueError(
                "masked route zero elision requires direct fused return and expert-local routed windows"
            )
        super().__init__(*args, packed_rows=ring_rows, **kwargs)
        if route_capacity is None:
            route_capacity = helper_value(None, "DG_MOE_ROUTE_CAPACITY", 0) or None
        if route_capacity is not None:
            if (
                type(route_capacity) is not int
                or ring_rows is None
                or route_capacity < self.rows
                or route_capacity % self.alignment
            ):
                raise ValueError("route capacity must cover an aligned complete activation window")
            # Keep the original GEMM pitches and last-active-window replay
            # boundary. Only bound route storage and the number of windows.
            self.full_row_capacity = min(route_capacity, self.full_row_capacity)
        self.route_capacity = route_capacity
        if self.fp8_dw_from_row_cache and (not self.fp8_wgrad or self.fp8_wgrad_block_k != 128):
            raise ValueError("row-FP8 DW cache requires128K standalone precise DW packing")
        stable_raw = os.environ.get(
            "DG_MOE_FP8_WGRAD_STABLE_ROUTING", "1" if self.fp8_wgrad else "0"
        )
        if stable_raw not in ("0", "1"):
            raise ValueError("Stable FP8 DW routing flag must be0or1")
        self.fp8_dw_stable_routing = stable_raw == "1"
        self.fp8_dw_route_block = helper_value(None, "DG_MOE_FP8_DW_ROUTE_BLOCK", 2048)
        if self.fp8_dw_route_block not in (256, 512, 1024, 2048):
            raise ValueError("FP8 DW route block must be256,512,1024or2048")
        native_prefix_raw = os.environ.get("DG_MOE_FP8_DW_NATIVE_PREFIX", "1")
        if native_prefix_raw not in ("0", "1"):
            raise ValueError("Native route prefix flag must be0or1")
        self.fp8_dw_native_prefix = native_prefix_raw == "1"
        if self.fp8_dw_stable_routing:
            shape = ((self.routes + self.fp8_dw_route_block - 1) // self.fp8_dw_route_block, self.e)
            self._stable_route_hist = torch.empty(shape, device=self.device, dtype=torch.int32)
            self._stable_route_prefix = torch.empty_like(self._stable_route_hist)
            self.buffers.update(
                stable_route_hist=self._stable_route_hist,
                stable_route_prefix=self._stable_route_prefix,
            )
        if self.fp8_expert_pipeline and (
            not self.fp8_peer_compute
            or ring_rows is None
            or (not expert_local_grad)
            or (self.fp8_fused_stage_quant != 3)
            or (not self.fp8_tiled_gradient_quant)
            or kwargs.get("overlap_expert_dy", False)
            or (kwargs.get("wgrad_overlap", "none") != "none")
            or gemm_configs
        ):
            raise ValueError(
                "expert pipeline requires bounded serial native-auto FP8 both-stage quantization without other input/gradient pipelines"
            )
        if self.bf16_grad_producer and self.i <= gemm_configs["dhidden"][0]:
            raise ValueError("gradient producer needs at least two dHidden N tiles")
        if self.wgrad_aggregate_windows and (
            ring_rows is None
            or not expert_local_grad
            or (not self.reuse_forward_context)
            or (self.wgrad_overlap != "none")
            or self.overlap_expert_dy
            or (not self.fuse_return)
            or (self.compute == "fp8" and (not self.fp8_expert_gradient_batch))
            or (
                self.compute == "bf16"
                and (not (self.fuse_activation_backward and self.fuse_dhidden_scores))
            )
            or (max(1, math.ceil(self.max_tokens / self.chunk_tokens)) > 8)
        ):
            raise ValueError(
                "aggregate DW requires <=8 live-context chunks, retained hidden, bounded serial expert gradients and direct return"
            )
        self.fp8_inline_source = int(os.environ.get("DG_MOE_FP8_INLINE_SOURCE", "0"))
        self.fp8_forward_tile_source = int(
            os.environ.get("DG_MOE_FP8_FWD_TILE_SOURCE_PIPELINE", "0")
        )
        self.fp8_forward_tile_ctas = int(os.environ.get("DG_MOE_FP8_FWD_TILE_SOURCE_CTAS", "4"))
        self.fp8_forward_tile_threads = int(
            os.environ.get("DG_MOE_FP8_FWD_TILE_SOURCE_THREADS", "128")
        )
        self.fp8_forward_gemm_ctas = int(os.environ.get("DG_MOE_FP8_FWD_TILE_GEMM_CTAS", "35"))
        if (
            self.fp8_forward_tile_source not in (0, 1)
            or not 1 <= self.fp8_forward_tile_ctas <= dg.get_num_sms()
            or self.fp8_forward_tile_threads not in (128, 256, 512, 1024)
            or (not 1 <= self.fp8_forward_gemm_ctas <= dg.get_num_sms())
        ):
            raise ValueError("invalid forward tile pipeline resource allocation")
        if self.fp8_forward_tile_source and self.fp8_inline_bulk_ilp != 4:
            raise ValueError("forward tile pipeline requires bulk ILP 4")
        if self.fp8_forward_tile_source and (
            self.fp8_forward_gemm_ctas + self.fp8_forward_tile_ctas > dg.get_num_sms()
        ):
            raise ValueError("forward source and polling GEMM CTA budgets must fit the CU count")
        if self.fp8_forward_tile_source and (
            not self.fp8_inline_source or self.h % 512 or (not self.fp8_expert_forward_batch)
        ):
            raise ValueError("forward tile pipeline requires inline forward FP8 source")
        if self.fp8_inline_source not in (0, 1, 2):
            raise ValueError("inline source must be0(off),1(forward)or2(bidirectional)")
        if self.fp8_inline_source and (
            not self.fp8_expert_pipeline
            or not self.fp8_expert_source_fp8
            or (not self.fp8_source_shared_base)
            or (not self.fp8_dw_from_row_cache)
        ):
            raise ValueError(
                "inline source requires serial native FP8 DW/source cache with expert batches"
            )
        if self.fp8_raw_gradient_quant_pipeline and (
            self.fp8_raw_gradient_source != 7
            or self.fp8_source_prep_reuse != 3
            or (self.chunk_tokens + 127) // 128 > dg.get_num_sms() * 16
        ):
            raise ValueError("token quant pipeline requires mode7, prep3, bounded token CTAs")
        if self.fp8_raw_gradient_source_prefetch and (
            not (
                self.fp8_raw_gradient_source == 7
                and self.fp8_raw_gradient_quant_pipeline
                and self.fp8_raw_gradient_tile_pipeline
            )
        ):
            raise ValueError("raw source prefetch requires mode7 token quant and tile pipeline")
        if self.fp8_raw_gradient_score_overlap and self.fp8_raw_gradient_source < 3:
            raise ValueError("owner score overlap requires common-base gradient packets")
        if self.fp8_raw_gradient_tile_pipeline and (not self.fp8_raw_gradient_source):
            raise ValueError("raw gradient tile pipeline requires raw source mode")
        if self.fp8_raw_gradient_source and (
            not (
                self.fp8_peer_compute
                and self.fp8_wgrad
                and self.fp8_wgrad_native_blockwise
                and (self.fp8_wgrad_block_k == 128)
                and self.fp8_dw_from_row_cache
                and self.fp8_mn_major_scales
                and self.reuse_forward_context
                and self.fp8_expert_gradient_batch
                and self.fp8_expert_forward_batch
                and (self.fp8_inline_source != 2)
            )
        ):
            raise ValueError(
                "raw gradient packets require live-context native K128 FP8 DW and expert batches"
            )
        self.fp8_expert_configs = {}
        if self.fp8_expert_pipeline:
            if max(self.fp8_expert_forward_batch, self.fp8_expert_gradient_batch) > self.e:
                raise ValueError("expert batch cannot exceed local experts")
            for role, width, batch in [
                ("fc1", 2 * self.i, self.fp8_expert_forward_batch),
                ("dhidden", self.i, self.fp8_expert_gradient_batch),
            ]:
                if batch:
                    cfg = tuple(
                        dg._C._mega_moe_training_fp8_auto_config(
                            self.device.index, self.rows, width, self.h, self.e
                        )
                    )
                    if (
                        self.fp8_inline_source
                        and (role == "fc1" or self.fp8_inline_source == 2)
                        and (cfg[-1] == 2)
                    ):
                        cfg = cfg[:-1] + (3,)
                    if role == "dhidden" and self.fp8_raw_gradient_gemm_config is not None:
                        cfg = self.fp8_raw_gradient_gemm_config
                    if (
                        (self.fp8_raw_gradient_tile_pipeline or self.fp8_raw_gradient_source >= 7)
                        and role == "dhidden"
                        and (cfg[-1] == 2)
                    ):
                        cfg = cfg[:-1] + (3,)
                    self.fp8_expert_configs[role] = cfg
            self.fp8_input_mode = "native_metadata_expert_batch_pipeline"
            if self.fp8_expert_source_fp8:
                self.fp8_input_mode = "source_fp8_expert_batch_bf16_dw_refetch"
                self.communication_dtype = "fp8_input_bf16_dw_refetch_and_return"
                if self.fp8_source_shared_base:
                    self.fp8_input_mode = "shared_current_base_source_fp8_expert_batch"
                if self.fp8_dw_from_row_cache:
                    self.communication_dtype = "fp8_source_input_reused_for_fp8_dw_bf16_returns"
                self._fp8_bf_input_stream = torch.cuda.Stream(device=self.device)
        self.fp8_return_native_configs = {}
        if self.h % (128 * self.fp8_input_slices):
            raise ValueError("hidden width must be divisible by128times FP8 input slices")
        if self.transpose_free_dgrad:
            # dGrad reads the original expert parameters via native AIU Trans.
            # Release the now-unused transposed weight copies.
            for name in ("w1_t", "w2_t"):
                self.buffers.pop(name)
                setattr(self, name, None)
        if self.overlap_expert_dy:
            self.packed_raw_dy = torch.empty_like(self.packed_dy)
            self.buffers["packed_raw_dy"] = self.packed_raw_dy
        if self.group is not None:
            recipe = (
                self.bf16_return_ncp_mode,
                self.fp8_inline_bulk_ilp,
                self.fp8_raw_bulk_ilp,
                self.fp8_forward_tile_source,
                self.fp8_forward_tile_ctas,
                self.fp8_forward_tile_threads,
                self.fp8_forward_gemm_ctas,
                False,
                0,
                self.fp8_inline_source_bulk,
                self.fp8_raw_gradient_source_bulk,
                self.fp8_return_remote_mode,
                False,
                False,
                self.fp8_return_pipeline_mma_flags,
                0,
                0,
                False,
                tuple(sorted(self.fp8_return_native_configs.items())),
                tuple(sorted(self.fp8_direct_return_configs.items())),
                self.fp8_raw_gradient_source_grid,
                0,
                self.fp8_raw_gradient_tile_ilp,
                self.fp8_raw_gradient_score_ilp,
                0,
                self.fp8_raw_gradient_score_threads,
                self.fp8_raw_gradient_source_prefetch,
                self.fp8_raw_gradient_gemm_ctas,
                self.fp8_raw_gradient_quant_ctas,
                self.fp8_raw_gradient_quant_threads,
                self.fp8_raw_gradient_quant_pipeline,
                self.fp8_raw_gradient_score_overlap,
                self.fp8_raw_gradient_tile_ctas,
                self.fp8_raw_gradient_tile_threads,
                self.fp8_raw_gradient_source_threads,
                self.fp8_raw_gradient_source_ctas,
                self.fp8_raw_gradient_tile_pipeline,
                self.fp8_raw_gradient_source,
                self.fp8_source_prep_reuse,
                self.fp8_inline_source,
                self.fp8_vector_activation,
                self.fp8_dw_metadata_mode,
                self.fp8_dw_swizzle_m,
                self.fp8_source_shared_base,
                self.fp8_dw_from_row_cache,
                self.fp8_source_refetch_phase,
                False,
                self.fp8_expert_source_fp8,
                False,
                self.wgrad_first_nonempty,
                self.wgrad_aggregate_windows,
                self.wgrad_aggregate_tile,
                self.fp8_expert_forward_ctas,
                self.fp8_expert_gradient_ctas,
                False,
                False,
                self.training_fp8_mma_flags,
                self.training_bf16_mma_reuse,
                self.fp8_expert_forward_batch,
                self.fp8_expert_gradient_batch,
                self.fp8_expert_source_ctas,
                self.fp8_expert_source_rows,
                False,
                tuple(sorted(self.fp8_expert_configs.items())),
                False,
                self.bf16_grad_producer,
                self.bf16_source_owned_tiles,
                self.skip_masked_route_zero,
                self.fuse_dispatch,
                self.fuse_return,
                False,
                self.expert_local_grad,
                False,
                False,
                False,
                False,
                False,
                False,
                self.input_copy_ilp,
                False,
                0,
                False,
                False,
                self.fuse_dhidden_scores,
                self.transpose_free_dgrad,
                0,
                0,
                0,
                self.fp8_weight_quantization,
                False,
                self.fp8_input_ctas,
                False,
                self.fp8_input_threads,
                False,
                self.reuse_forward_context,
                0,
                False,
                self.fp8_fused_stage_quant,
                self.fp8_tiled_gradient_quant,
                self.fp8_tiled_gradient_cache,
                False,
                False,
                self.fp8_return_ctas,
                self.fp8_return_threads,
                self.fp8_return_store_bytes,
                False,
                False,
                0,
                self.fp8_input_slices,
                self.fp8_dw_stable_routing,
                self.fp8_dw_route_block,
                self.fp8_dw_native_prefix,
            )
            recipes = [None] * self.size
            dist.all_gather_object(recipes, recipe, group=self.group)
            if any((other != recipe for other in recipes)):
                raise ValueError("EP ranks must agree on the peer gradient and return protocol")
        if gemm_configs is None and self.block_m == 256:
            gemm_configs = {role: (256, 64, 64, 64, 2) for role in ("fc1", "fc2", "dhidden", "dx")}
        self.gemm_configs = (
            {}
            if gemm_configs is None
            else {name: tuple(value) for name, value in gemm_configs.items()}
        )
        if set(self.gemm_configs) - {"fc1", "fc2", "dhidden", "dx"}:
            raise ValueError("unknown peer GEMM configuration role")
        if self.size > 64:
            raise ValueError("peer training supports at most 64 ranks per domain")
        c, h, k = (self.chunk_tokens, self.h, self.k)
        descriptions = (
            ("local_x", (c, h), torch.bfloat16),
            ("local_ids", (c, k), torch.int64),
            ("local_values", (c * k, h), torch.bfloat16),
            ("local_dx", (c * k, h), torch.bfloat16),
            ("local_dy", (c * k, h), torch.bfloat16),
            ("barrier_state", (4,), torch.int32),
        )
        if self.expert_local_grad:
            # Expert-side gradients consume raw_dy and never the top-k DY matrix.
            descriptions = tuple(item for item in descriptions if item[0] != "local_dy")
            descriptions += (
                ("raw_dy", (c, h), torch.bfloat16),
                ("raw_scores", (c, k), torch.float32),
                ("local_dscores", (c, k), torch.float32),
            )
        if self.fp8_expert_source_fp8:
            descriptions += (
                ("local_qx", (c, h), torch.float8_e4m3fn),
                ("local_qx_scales", (c, h // 128), torch.float32),
            )
        if self.fp8_raw_gradient_source:
            descriptions += (
                ("local_qdy_hi", (c, h), torch.float8_e4m3fn),
                ("local_qdy_lo", (c, h), torch.float8_e4m3fn),
                ("local_qdy_scales", (c, h // 128), torch.float32),
            )
        if self.fp8_raw_gradient_source >= 3:
            descriptions += (("local_dy_feature_amax", (h // 128,), torch.float32),)
        if self.fp8_raw_gradient_source >= 7:
            descriptions += (("local_dy_token_ready", ((c + 127) // 128 + 1,), torch.int32),)
        if self.fp8_source_shared_base:
            descriptions += (("local_x_feature_amax", (h // 128,), torch.float32),)
        layout, size = ({}, 0)
        for name, shape, dtype in descriptions:
            size = (size + 15) // 16 * 16
            count = math.prod(shape)
            element = torch.empty((), dtype=dtype).element_size()
            layout[name] = (size, count * element, shape, dtype)
            size += count * element
        self.symm_handle = None
        if self.size == 1:
            self.payload = torch.empty(size, device=self.device, dtype=torch.uint8)
            self.peer_buffers = [self.payload]
        else:
            from torch.distributed import _symmetric_memory as sm

            self.payload = sm.empty((size,), dtype=torch.uint8, device=self.device)
            self.symm_handle = sm.rendezvous(self.payload, self.group)
            direct = getattr(self.symm_handle, "world_within_direct_access", None)
            direct = direct() if callable(direct) else direct
            # The native PPU Torch 2.10 CUDA backend imports/maps every peer
            # allocation through the driver, but omits the newer topology flag.
            # Accept that direct-buffer backend after successful rendezvous;
            # do not infer direct pointers for an unknown/RDMA memory backend.
            if direct is None and sm.get_backend(self.device) != "CUDA":
                raise RuntimeError("symmetric-memory backend cannot supply verified direct buffers")
            if direct is not None and (not direct):
                raise RuntimeError(
                    "peer training requires an EP group within one direct-access domain"
                )
            self.peer_buffers = [
                self.symm_handle.get_buffer(rank, (size,), torch.uint8) for rank in range(self.size)
            ]
        self.payload.zero_()
        self.pointer_tables = {}
        for name, (offset, length, shape, dtype) in layout.items():
            view = self.payload[offset : offset + length].view(dtype).reshape(shape)
            setattr(self, name, view)
            self.buffers[name] = view
            addresses = [buffer.data_ptr() + offset for buffer in self.peer_buffers]
            encoded = [value if value < 2**63 else value - 2**64 for value in addresses]
            table = torch.tensor(encoded, device=self.device, dtype=torch.int64)
            self.pointer_tables[name] = table
            self.buffers[name + "_peers"] = table
        for name in ("gathered_x", "gathered_ids", "route_values") + (
            () if self.fp8_peer_compute else ("group_ids",)
        ):
            self.buffers.pop(name)
            delattr(self, name)
        self.source_rows = torch.empty(self.rows, device=self.device, dtype=torch.int32)
        self.identity_rows = torch.arange(self.rows, device=self.device, dtype=torch.int32)
        self.work_info = torch.empty(
            (self.rows // self.block_m, 4), device=self.device, dtype=torch.int32
        )
        self.num_blocks = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.buffers.update(
            source_rows=self.source_rows,
            identity_rows=self.identity_rows,
            work_info=self.work_info,
            num_blocks=self.num_blocks,
        )
        self.route_window_count = 1
        if self.ring_rows is not None:
            allocate_route_windows(self)
        self.dispatch_scratch = None
        self.pipeline_ready = self.return_pipeline_ready = self.pipeline_statistics = None
        self.return_pipeline_statistics = None
        self.dispatch_pipeline_ctas = self.return_pipeline_ctas = 0
        if self.fp8_tiled_gradient_quant:
            self.fp8_dscore_partials = torch.empty(
                (self.rows, self.h // 128), device=self.device, dtype=torch.float32
            )
            self.buffers["fp8_dscore_partials"] = self.fp8_dscore_partials
        if self.fp8_raw_gradient_source:
            if self.fp8_raw_gradient_source >= 3:
                self.fp8_raw_gradient_row_amax = torch.empty(
                    (c, h // 128), device=self.device, dtype=torch.float32
                )
                self.fp8_raw_gradient_global_amax = torch.empty(
                    h // 128, device=self.device, dtype=torch.float32
                )
                self.buffers.update(
                    fp8_raw_gradient_row_amax=self.fp8_raw_gradient_row_amax,
                    fp8_raw_gradient_global_amax=self.fp8_raw_gradient_global_amax,
                )
                self.communication_dtype = "fp8_source_input_and_common_dy_bf16_returns"
            self.fp8_raw_score_partials = torch.empty(
                (c * k, h // 128), device=self.device, dtype=torch.float32
            )
            self.fp8_raw_gradient_scales = torch.empty(
                (h // 128, self.rows), device=self.device, dtype=torch.float32
            )
            self.fp8_raw_gradient_scores = torch.empty(
                self.rows, device=self.device, dtype=torch.float32
            )
            self.buffers.update(
                fp8_raw_score_partials=self.fp8_raw_score_partials,
                fp8_raw_gradient_scales=self.fp8_raw_gradient_scales,
                fp8_raw_gradient_scores=self.fp8_raw_gradient_scores,
            )
        if self.fuse_dhidden_scores:
            bn = self.gemm_configs["dhidden" if self.fuse_dhidden_scores else "fc2"][0]
            width = self.i if self.fuse_dhidden_scores else self.h
            self.score_partial_columns = (width + bn - 1) // bn
            self.score_partials = torch.empty(
                (self.rows, self.score_partial_columns), device=self.device, dtype=torch.float32
            )
            self.buffers["score_partials"] = self.score_partials
            if self.fuse_dhidden_scores:
                self.packed_row_scores = torch.empty(
                    self.rows, device=self.device, dtype=torch.float32
                )
                self.buffers["packed_row_scores"] = self.packed_row_scores
        self.gate_up.zero_()
        self.hidden.zero_()
        self.quantized_scale_storage = {}
        if self.fp8_peer_compute:
            for width in (self.h, self.i, 2 * self.i):
                if width in self.quantized_scale_storage:
                    continue
                storage = torch.empty(
                    (width // 128, self.rows), device=self.device, dtype=torch.float32
                )
                self.buffers["activation_scale_mn_" + str(width)] = storage
                self.quantized[width] = (self.quantized[width][0], storage.t())
                self.buffers.pop("activation_scale_" + str(width))
                self.quantized_scale_storage[width] = storage
            self.group_ids.zero_()
        if self.fp8_source_shared_base:
            self._source_row_amax = torch.empty(
                (c, h // 128), device=self.device, dtype=torch.float32
            )
            self._source_global_amax = torch.empty(
                (h // 128,), device=self.device, dtype=torch.float32
            )
            self.buffers.update(
                source_row_amax=self._source_row_amax, source_global_amax=self._source_global_amax
            )
        if self.fp8_dw_from_row_cache:
            self.fp8_source_x_q = torch.empty(
                (self.rows, h), device=self.device, dtype=torch.float8_e4m3fn
            )
            self.fp8_source_x_sf = torch.empty(
                (h // 128, self.rows), device=self.device, dtype=torch.float32
            )
            self.buffers.update(
                fp8_source_x_q=self.fp8_source_x_q, fp8_source_x_sf=self.fp8_source_x_sf
            )
        self.forward_context_slots = []
        if self.reuse_forward_context:
            # Each source chunk retains its last nonempty ring window. Earlier
            # windows can still be replayed without unbounded skew allocations.
            names = ["gate_up", "hidden", "global_offsets", "global_lengths", "global_permutation"]
            if self.fp8_dw_from_row_cache:
                names += ["fp8_source_x_q", "fp8_source_x_sf"]
            names += ["packed_output"] if self.fp8_peer_compute else ["packed_x"]
            if self.fp8_peer_compute:
                names.append("packed_x")
            original = {name: getattr(self, name) for name in names}
            if self.fp8_peer_compute:
                original.update(
                    quantized_hidden=self.quantized[self.i][0],
                    quantized_hidden_scales=self.quantized_scale_storage[self.i],
                )
            for slot in range(max(1, math.ceil(self.max_tokens / self.chunk_tokens))):
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
            for name, shape, dtype in (
                ("replay_offsets", (self.e + 1,), torch.int32),
                ("replay_lengths", (self.e,), torch.int32),
                ("replay_blocks", (1,), torch.int32),
                ("cache_only_blocks", (1,), torch.int32),
                ("forward_context_reused_windows", (1,), torch.int32),
                ("forward_context_reused_rows", (1,), torch.int64),
            ):
                value = torch.zeros(shape, device=self.device, dtype=dtype)
                setattr(self, name, value)
                self.buffers[name] = value
        self.fp8_inline_ready = {}
        self.fp8_expert_metadata = {}
        self.fp8_expert_events = {}
        if self.fp8_expert_pipeline:
            self._fp8_expert_stream = torch.cuda.Stream(device=self.device)
            for role, batch in [
                ("fc1", self.fp8_expert_forward_batch),
                ("dhidden", self.fp8_expert_gradient_batch),
            ]:
                if not batch:
                    continue
                bm = self.fp8_expert_configs[role][0]
                count = (self.e + batch - 1) // batch
                capacity = max(128, 4 * (1 + (self.rows + bm - 1) // bm + self.e))
                values = [
                    torch.empty(capacity, device=self.device, dtype=torch.int32)
                    for _ in range(count)
                ]
                self.fp8_expert_metadata[role] = values
                if self.fp8_inline_source:
                    ready = torch.zeros(
                        (self.rows + bm - 1) // bm
                        + self.e
                        + int(bool(self.fp8_forward_tile_source)),
                        device=self.device,
                        dtype=torch.int32,
                    )
                    self.fp8_inline_ready[role] = ready
                    self.buffers["fp8_inline_ready_" + role] = ready
                self.fp8_expert_events[role] = [torch.cuda.Event() for _ in values]
                self.buffers.update(
                    {f"fp8_expert_meta_{role}_{idx}": v for idx, v in enumerate(values)}
                )
            if self.fp8_expert_gradient_batch:
                self.fp8_expert_dhidden = torch.empty_like(self.hidden)
                self.buffers["fp8_expert_dhidden"] = self.fp8_expert_dhidden
        self._aggregate_cache = []
        if self.wgrad_aggregate_windows and len(self.forward_context_slots) > 1:
            for slot in range(max(1, math.ceil(self.max_tokens / self.chunk_tokens))):
                cache = dict(
                    dy=torch.empty_like(self.packed_dy),
                    offsets=torch.empty(self.e + 1, device=self.device, dtype=torch.int32),
                )
                self._aggregate_cache.append(cache)
                self.buffers.update(
                    {f"aggregate_dw_{slot}_{name}": value for name, value in cache.items()}
                )
        if self.wgrad_first_nonempty:
            self._wgrad_seen = torch.empty(self.e, device=self.device, dtype=torch.int32)
            self._wgrad_first_flags = torch.empty_like(self._wgrad_seen)
            self.buffers.update(
                wgrad_seen=self._wgrad_seen, wgrad_first_flags=self._wgrad_first_flags
            )
            if self.wgrad_aggregate_windows:
                pairs = (max(1, math.ceil(self.max_tokens / self.chunk_tokens)) + 1) // 2
                self._aggregate_first_flags = torch.empty(
                    (pairs, self.e), device=self.device, dtype=torch.int32
                )
                self.buffers["aggregate_first_flags"] = self._aggregate_first_flags
        self._dw_source_pending = False
        self._prepared = False
        torch.cuda.synchronize(self.device)
        if self.group is not None:
            dist.barrier(group=self.group)
        self._fp8_raw_source_pending = False
        if self.fp8_raw_gradient_source_prefetch:
            self.fp8_raw_gradient_prefetch_q = torch.empty_like(self.quantized[self.h][0])
            self.buffers["fp8_raw_gradient_prefetch_q"] = self.fp8_raw_gradient_prefetch_q
            self._fp8_raw_source_stream = torch.cuda.Stream(device=self.device)
        if self.fp8_raw_gradient_quant_pipeline:
            self._fp8_raw_quant_stream = torch.cuda.Stream(device=self.device)
            self._fp8_raw_quant_done = torch.cuda.Event()
        if self.fp8_raw_gradient_score_overlap:
            self._fp8_raw_score_stream = torch.cuda.Stream(device=self.device)
            self._fp8_raw_score_done = torch.cuda.Event()
        if self.fp8_raw_gradient_tile_pipeline or self.fp8_raw_gradient_source >= 7:
            bm = self.fp8_expert_configs["dhidden"][0]
            self.fp8_raw_gradient_ready = torch.zeros(
                (self.rows + bm - 1) // bm + self.e + 1, device=self.device, dtype=torch.int32
            )
            self.buffers["fp8_raw_gradient_ready"] = self.fp8_raw_gradient_ready
            if self.fp8_raw_gradient_source >= 7:
                self.fp8_raw_gradient_sf_tile = torch.empty(
                    (self.h // 128, bm), device=self.device, dtype=torch.float32
                )
                self.buffers["fp8_raw_gradient_sf_tile"] = self.fp8_raw_gradient_sf_tile

    def _join_fp8_weight_prepare(self):
        if self._fp8_weights_pending:
            torch.cuda.current_stream(self.device).wait_event(self._fp8_weight_event)
            self._fp8_weights_pending = False

    def _prepare_weights(self, w1, w2, backward=False):
        if self.fp8_peer_compute:
            self._fp8_prequantized_input = None
            self._fp8_backward = backward
            if self._shared_fp8_weights_ready:
                return
        if self.transpose_free_dgrad:
            self.w1_t, self.w2_t = (w1, w2)
            return
        self._join_fp8_weight_prepare()
        return super()._prepare_weights(w1, w2, backward=backward)

    @staticmethod
    def _context_fingerprint(*tensors):
        return tuple(((value.data_ptr(), value._version, tuple(value.shape)) for value in tensors))

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

    def _prepare_context_replay(self, start, compile_only=False):
        _mega_moe(
            20,
            [
                self.offsets,
                self.lengths,
                self.num_blocks,
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

    def _quantize_weight(self, name, weight):
        data, scales = self.quantized[name]
        _launch(
            (10, 29, 30)[self.fp8_weight_quantization], [weight, data, scales], list(weight.shape)
        )

    def _prepare_pointwise(self):
        super()._prepare_pointwise()
        if self.route_capacity is not None:
            dg._C._mega_moe_training_precompile(self.device.index, [40])
        if self.fp8_peer_compute:
            dg._C._mega_moe_training_precompile(
                self.device.index,
                [
                    9,
                    34 if self.fp8_vector_row_quant else 26,
                    27,
                    28,
                    (10, 29, 30)[self.fp8_weight_quantization],
                ],
                int(self.fp8_vector_activation),
            )

    def _barrier(self, tag, compile_only=False):
        _mega_moe(
            5,
            [self.pointer_tables["barrier_state"], self.barrier_state, self.status],
            [self.size, self.rank, 180 * 2000000000, tag],
            compile_only,
        )

    def _peer_gemm(
        self,
        a,
        weight,
        output,
        source_rows,
        *,
        read=False,
        write=False,
        source="local_x",
        destination="local_values",
        compile_only=False,
        pack_input=False,
        role=None,
        first_n_tile=0,
        num_n_tiles=0,
        local_input_aiu=False,
        producer_input_aiu=False,
        dispatch_pipeline=False,
        keep_gate_up=True,
        replay_gradients=False,
        fc1_helper_threads=0,
    ):
        if self.bf16_grad_producer and role == "dhidden" and (not read):
            if self.bf16_source_owned_tiles & 2:
                return None
            first_n_tile = 1
            num_n_tiles = 0
            local_input_aiu = True
        source_stride = (
            self.chunk_tokens if source in ("local_x", "raw_dy") else self.chunk_tokens * self.k
        )
        if role is None:
            role = (
                ("fc1" if source == "local_x" else "dhidden")
                if read
                else "dx" if destination == "local_dx" else "fc2"
            )
        if self.fp8_peer_compute:
            if read or dispatch_pipeline or first_n_tile or num_n_tiles or replay_gradients:
                raise ValueError("FP8 peer GEMM requires staged local input and a complete N range")
            name = {"fc1": "w1", "fc2": "w2", "dhidden": "w2_t", "dx": "w1_t"}[role]
            width = a.shape[1]
            quantized_a = self.quantized[width]
            if self._fp8_prequantized_input != a.data_ptr():
                _launch(
                    34 if self.fp8_vector_row_quant else 26,
                    [
                        a,
                        quantized_a[0],
                        self.quantized_scale_storage[width],
                        self.current_rows,
                        self.status,
                    ],
                    [self.rows, width, 1],
                )
            self._fp8_prequantized_input = None
            config = self.gemm_configs.get(role)
            if write:
                direct_cfg = self.fp8_direct_return_configs.get(role)
                return dg._C._mega_moe_training_fp8_gemm(
                    quantized_a,
                    self.quantized[name],
                    output,
                    self.lengths,
                    self.permutation,
                    self.pointer_tables[destination],
                    self.chunk_tokens * self.k,
                    keep_local=self.reuse_forward_context
                    and role == "fc2"
                    and (not self._fp8_backward)
                    and (
                        not (
                            False
                            and self.fp8_raw_gradient_source
                            and (len(self.forward_context_slots) == 1)
                        )
                    ),
                    compile_only=compile_only,
                    config=direct_cfg[1:] if direct_cfg else () if config is None else config,
                    return_store_bytes=self.fp8_return_store_bytes,
                    return_block_m=direct_cfg[0] if direct_cfg else 0,
                    return_max_regs=0,
                    return_remote_mode=self.fp8_return_remote_mode,
                    return_local_rank=self.rank,
                )
            options = (
                {}
                if config is None
                else {"configs": (dg.get_num_sms(), self.block_m, *config, (0, 0, 0))}
            )
            # Zero lengths make this a safe local warmup during preparation.
            dg.m_grouped_fp8_gemm_nt_nopad(
                quantized_a,
                self.quantized[name],
                output,
                self.group_ids,
                m_rows=self.lengths,
                disable_ue8m0_cast=True,
                **options,
            )
            return None
        return_pipeline = write and False
        fc1_activation = role == "fc1" and self.fuse_fc1_activation
        interleaved = self.g8_interleaved and (
            fc1_activation or (role == "dhidden" and self.fuse_activation_backward)
        )
        if return_pipeline and (not compile_only):
            self.return_pipeline_ready.zero_()
        extra_arguments = []
        if replay_gradients:
            extra_arguments.append(
                [
                    self.pointer_tables["raw_dy"],
                    self.pointer_tables["raw_scores"],
                    self.packed_dy,
                    self.score_partials,
                    self.status,
                ]
            )
        if self.fuse_dhidden_scores and role == "dhidden":
            if not extra_arguments:
                extra_arguments.append([])
            extra_arguments.append(
                [self.hidden, self.packed_row_scores, self.score_partials]
                + (
                    [self.pointer_tables["raw_scores"], self.packed_dy, self.status]
                    if self.bf16_grad_producer and read
                    else []
                )
            )
        if self.transpose_free_dgrad and role in ("dhidden", "dx"):
            while len(extra_arguments) < 2:
                extra_arguments.append([])
            extra_arguments.append(True)
        extra_options = {}
        if fc1_helper_threads:
            extra_options = dict(fc1_helper_threads=fc1_helper_threads, fc1_compute_block_m=0)
        grid = dg._C._mega_moe_training_gemm(
            a,
            weight,
            output,
            self.pointer_tables[source if read else "local_x"],
            self.pointer_tables[destination],
            source_rows,
            self.permutation,
            self.work_info,
            self.num_blocks,
            source_stride,
            self.chunk_tokens * self.k,
            read,
            write,
            compile_only,
            pack_input,
            self.gemm_configs.get(role, ()),
            (
                self.hidden
                if fc1_activation
                else self.gate_up if role == "dhidden" and self.fuse_activation_backward else None
            ),
            0,
            self.dispatch_scratch if read else None,
            False,
            False,
            self.dispatch_cache_ctas,
            first_n_tile,
            num_n_tiles,
            local_input_aiu,
            producer_input_aiu,
            (
                self.return_pipeline_ready
                if return_pipeline
                else self.pipeline_ready if dispatch_pipeline else None
            ),
            self.status if dispatch_pipeline or return_pipeline else None,
            (
                (
                    self.return_pipeline_statistics
                    if return_pipeline
                    else self.pipeline_statistics if dispatch_pipeline else None
                )
                if not compile_only
                else None
            ),
            self.pipeline_comm_ctas,
            False,
            False,
            self.block_m,
            self.return_vector_bytes if write else 4,
            interleaved,
            fc1_activation,
            keep_gate_up if fc1_activation else True,
            (self.input_copy_ilp if producer_input_aiu or fc1_helper_threads else 1),
            False,
            self.dgrad_compute_cus if role in ("dhidden", "dx") else 0,
            *extra_arguments,
            return_ncp_mode=self.bf16_return_ncp_mode if write and role in ("fc2", "dx") else 0,
            return_local_rank=self.rank,
            **extra_options,
        )
        if compile_only and (dispatch_pipeline or return_pipeline):
            grid_name = "return_pipeline_ctas" if return_pipeline else "dispatch_pipeline_ctas"
            setattr(self, grid_name, max(grid, getattr(self, grid_name)))

    def _raw_gradient_qpair(self):
        q = self.quantized[self.h]
        return (
            (self.fp8_raw_gradient_prefetch_q, q[1]) if self.fp8_raw_gradient_source_prefetch else q
        )

    def _start_fp8_raw_gradient_broadcast_source(self, *, compile_only=False):
        role = "dhidden"
        cfg = self.fp8_expert_configs[role]
        bm = cfg[0]
        mode = self._fp8_raw_gradient_active or self.fp8_raw_gradient_source
        meta = self.fp8_expert_metadata[role][0]
        ready = self.fp8_raw_gradient_ready
        q = self._raw_gradient_qpair()
        sf = self.quantized_scale_storage[self.h]
        if not compile_only and self._fp8_raw_source_pending:
            raise RuntimeError("raw source already pending")
        tile = self.fp8_raw_gradient_sf_tile
        quant_pipeline = self.fp8_raw_gradient_quant_pipeline and mode == 7
        operation = (
            (
                {1: 20, 2: 17, 4: 22, 8: 24, 16: 26, 32: 28}
                if quant_pipeline
                else {1: 21, 2: 11, 4: 23, 8: 25, 16: 27, 32: 29}
            )[self.fp8_raw_gradient_tile_ilp]
            if mode == 7
            else 13
        )
        if not quant_pipeline or compile_only:
            dg._C._mega_moe_training_fp8_gradient(
                15,
                [self.fp8_raw_gradient_global_amax, tile, self.status],
                [self.h, bm, 128],
                compile_only,
            )
        if quant_pipeline and compile_only:
            dg._C._mega_moe_training_fp8_gradient(
                16,
                [
                    self.raw_dy,
                    self.local_qdy_hi,
                    self.lengths[:1],
                    tile,
                    self.local_dy_token_ready,
                    self.status,
                ],
                [
                    self.chunk_tokens,
                    self.h,
                    bm,
                    self.fp8_raw_gradient_quant_threads,
                    min(
                        (self.chunk_tokens + 127) // 128,
                        self.fp8_raw_gradient_quant_ctas or (self.chunk_tokens + 127) // 128,
                    ),
                ],
                True,
            )
        _mega_moe(45, [self.lengths, meta, self.status], [self.e, self.rows, bm], compile_only)
        if not compile_only:
            ready.zero_()
        source = [
            self.pointer_tables["local_qdy_hi"] if mode == 7 else self.pointer_tables["raw_dy"],
            (
                (
                    self.pointer_tables["local_dy_token_ready"]
                    if quant_pipeline
                    else self.pointer_tables["local_qdy_scales"]
                )
                if mode == 7
                else tile
            ),
            self.pointer_tables["raw_scores"],
            self.permutation,
            self.lengths,
            q[0],
            sf,
            self.fp8_raw_gradient_scales,
            self.fp8_raw_gradient_scores,
            self.status,
            meta,
            ready,
        ]
        dims = [
            self.rows,
            self.h,
            self.chunk_tokens,
            self.k,
            self.size,
            0,
            self.e,
            self.fp8_raw_gradient_tile_ctas,
            self.fp8_raw_gradient_tile_threads,
            bm,
        ]
        if self.fp8_raw_gradient_source_bulk and mode == 7:
            operation = 30 if quant_pipeline else 31
            if self.fp8_raw_bulk_ilp != 4:
                if not quant_pipeline:
                    raise ValueError("bulk ILP override requires common-DY quant pipeline")
                operation = {8: 35, 16: 36}[self.fp8_raw_bulk_ilp]
            dims.extend(
                [
                    self.fp8_raw_gradient_source_grid
                    or dg.get_num_sms() * self.fp8_raw_gradient_tile_ctas,
                    self.rank,
                ]
            )
        elif self.fp8_raw_gradient_source_grid:
            dims.append(self.fp8_raw_gradient_source_grid)
        main = torch.cuda.current_stream(self.device)
        source_stream = (
            self._fp8_raw_source_stream
            if self.fp8_raw_gradient_source_prefetch
            else self._fp8_expert_stream
        )
        if compile_only:
            dg._C._mega_moe_training_fp8_gradient(operation, source, dims, True)
        else:
            source_stream.wait_stream(main)
            with torch.cuda.stream(source_stream):
                for tensor in source + [self.payload]:
                    tensor.record_stream(source_stream)
                dg._C._mega_moe_training_fp8_gradient(operation, source, dims)
        dg._C._mega_moe_training_fp8_gradient(
            6,
            [ready, self.status],
            [
                ready.numel(),
                self.fp8_raw_gradient_source_grid
                or dg.get_num_sms() * self.fp8_raw_gradient_tile_ctas,
                128,
            ],
            compile_only,
        )
        if not compile_only:
            self._fp8_raw_source_pending = True
        if not self.fp8_raw_gradient_tile_pipeline and (not compile_only):
            main.wait_stream(source_stream)

    def _fp8_raw_gradient_broadcast_input(self, *, compile_only=False):
        role = "dhidden"
        cfg = self.fp8_expert_configs[role]
        bm = cfg[0]
        q = self._raw_gradient_qpair()
        tile = self.fp8_raw_gradient_sf_tile
        meta = self.fp8_expert_metadata[role][0]
        ready = self.fp8_raw_gradient_ready
        main = torch.cuda.current_stream(self.device)
        counter = None
        if counter is not None and (not compile_only):
            counter.zero_()
        if compile_only or not self._fp8_raw_source_pending:
            self._start_fp8_raw_gradient_broadcast_source(compile_only=compile_only)
        if not compile_only:
            self._join_fp8_weight_prepare()
            self._join_dw_source_pair()
        dg._C._mega_moe_training_fp8_gemm(
            q,
            self.quantized["w2_t"],
            self.fp8_expert_dhidden,
            self.lengths,
            self.permutation,
            self.pointer_tables["local_values"],
            self.chunk_tokens * self.k,
            keep_local=True,
            compile_only=compile_only,
            config=cfg[1:],
            expert_metadata=meta,
            metadata_block_m=bm,
            metadata_ctas=self.fp8_raw_gradient_gemm_ctas,
            inline_max_regs=0,
            inline_sources=[ready, self.status, tile, self.fp8_raw_gradient_scores],
            inline_source_rotation=bool(0 and self.fp8_raw_gradient_source_bulk),
            return_local_rank=self.rank,
            tile_work_counter=counter,
        )
        if not compile_only:
            main.wait_stream(
                self._fp8_raw_source_stream
                if self.fp8_raw_gradient_source_prefetch
                else self._fp8_expert_stream
            )
            self._fp8_raw_source_pending = False
            self._fp8_prequantized_input = None

    def _fp8_raw_gradient_input_pipeline(self, *, compile_only=False):
        role = "dhidden"
        cfg = self.fp8_expert_configs[role]
        bm = cfg[0]
        meta = self.fp8_expert_metadata[role][0]
        ready = self.fp8_raw_gradient_ready
        q = self.quantized[self.h]
        sf = self.quantized_scale_storage[self.h]
        _mega_moe(45, [self.lengths, meta, self.status], [self.e, self.rows, bm], compile_only)
        if not compile_only:
            ready.zero_()
        source = [
            self.pointer_tables["local_qdy_hi"],
            self.pointer_tables["local_qdy_scales"],
            self.pointer_tables["raw_scores"],
            self.permutation,
            self.lengths,
            q[0],
            sf,
            self.fp8_raw_gradient_scales,
            self.fp8_raw_gradient_scores,
            self.status,
            meta,
            ready,
        ]
        main = torch.cuda.current_stream(self.device)
        if compile_only:
            dg._C._mega_moe_training_fp8_gradient(
                5,
                source,
                [
                    self.rows,
                    self.h,
                    self.chunk_tokens,
                    self.k,
                    self.size,
                    0,
                    self.e,
                    self.fp8_raw_gradient_tile_ctas,
                    self.fp8_raw_gradient_tile_threads,
                    bm,
                ],
                True,
            )
        else:
            self._fp8_expert_stream.wait_stream(main)
            with torch.cuda.stream(self._fp8_expert_stream):
                for tensor in source + [self.payload]:
                    tensor.record_stream(self._fp8_expert_stream)
                dg._C._mega_moe_training_fp8_gradient(
                    5,
                    source,
                    [
                        self.rows,
                        self.h,
                        self.chunk_tokens,
                        self.k,
                        self.size,
                        0,
                        self.e,
                        self.fp8_raw_gradient_tile_ctas,
                        self.fp8_raw_gradient_tile_threads,
                        bm,
                    ],
                )
            self._join_fp8_weight_prepare()
            self._join_dw_source_pair()
        dg._C._mega_moe_training_fp8_gradient(
            6,
            [ready, self.status],
            [ready.numel(), dg.get_num_sms() * self.fp8_raw_gradient_tile_ctas, 128],
            compile_only,
        )
        dg._C._mega_moe_training_fp8_gemm(
            q,
            self.quantized["w2_t"],
            self.fp8_expert_dhidden,
            self.lengths,
            self.permutation,
            self.pointer_tables["local_values"],
            self.chunk_tokens * self.k,
            keep_local=True,
            compile_only=compile_only,
            config=cfg[1:],
            expert_metadata=meta,
            metadata_block_m=bm,
            inline_sources=[ready, self.status],
        )
        if not compile_only:
            main.wait_stream(self._fp8_expert_stream)
            self._fp8_prequantized_input = None

    def _start_raw_gradient_quant(self, valid_tokens):
        if self._fp8_raw_quant_pending:
            raise RuntimeError("owner quantization not joined")
        tensors = [
            self.raw_dy,
            self.local_qdy_hi,
            valid_tokens,
            self.fp8_raw_gradient_sf_tile,
            self.local_dy_token_ready,
            self.status,
        ]
        self._fp8_raw_quant_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._fp8_raw_quant_stream):
            for tensor in tensors:
                tensor.record_stream(self._fp8_raw_quant_stream)
            dg._C._mega_moe_training_fp8_gradient(
                16,
                tensors,
                [
                    self.chunk_tokens,
                    self.h,
                    self.fp8_expert_configs["dhidden"][0],
                    self.fp8_raw_gradient_quant_threads,
                    min(
                        (self.chunk_tokens + 127) // 128,
                        self.fp8_raw_gradient_quant_ctas or (self.chunk_tokens + 127) // 128,
                    ),
                ],
            )
            self._fp8_raw_quant_done.record()
        self._fp8_raw_quant_pending = True
        # Every local owner CTA must have started before peer-ready waits can run.
        dg._C._mega_moe_training_fp8_gradient(
            6,
            [self.local_dy_token_ready, self.status],
            [
                self.local_dy_token_ready.numel(),
                min(
                    (self.chunk_tokens + 127) // 128,
                    self.fp8_raw_gradient_quant_ctas or (self.chunk_tokens + 127) // 128,
                ),
                128,
            ],
        )

    def _join_raw_gradient_quant(self):
        if self._fp8_raw_quant_pending:
            torch.cuda.current_stream(self.device).wait_event(self._fp8_raw_quant_done)
            self._fp8_raw_quant_pending = False

    def _start_raw_gradient_score(self, valid_tokens):
        if self._fp8_raw_score_pending:
            raise RuntimeError("owner scores were not joined")
        tensors = [
            self.raw_dy,
            self.raw_scores,
            self.local_values,
            self.local_ids,
            valid_tokens,
            self.local_qdy_hi,
            self.local_qdy_lo,
            self.local_qdy_scales,
            self.fp8_raw_score_partials,
            self.status,
        ]
        self._fp8_raw_score_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._fp8_raw_score_stream):
            for value in tensors:
                value.record_stream(self._fp8_raw_score_stream)
            self.local_dscores.record_stream(self._fp8_raw_score_stream)
            dg._C._mega_moe_training_fp8_gradient(
                {4: 10, 1: 18, 2: 19}[self.fp8_raw_gradient_score_ilp],
                tensors,
                [
                    self.chunk_tokens,
                    self.k,
                    self.h,
                    self.e * self.size,
                    4,
                    self.fp8_raw_gradient_score_threads,
                ]
                + [],
            )
            dg._C._mega_moe_training_fp8_gradient(
                1,
                [self.fp8_raw_score_partials, self.local_dscores, self.status],
                [self.chunk_tokens * self.k, self.h, 4, 128],
            )
            self._fp8_raw_score_done.record()
        self._fp8_raw_score_pending = True

    def _join_raw_gradient_score(self):
        if self._fp8_raw_score_pending:
            torch.cuda.current_stream(self.device).wait_event(self._fp8_raw_score_done)
            self._fp8_raw_score_pending = False

    def _fp8_raw_gradient_stats(self):
        dg._C._mega_moe_training_fp8_reuse(
            0, [self.raw_dy, self.fp8_raw_gradient_row_amax], [self.chunk_tokens, self.h]
        )
        dg._C._mega_moe_training_fp8_reuse(
            1,
            [self.fp8_raw_gradient_row_amax, self.local_dy_feature_amax],
            [self.chunk_tokens, self.h],
        )

    def _fp8_raw_gradient_publish(self, valid_tokens):
        tensors = [
            self.raw_dy,
            self.raw_scores,
            self.local_values,
            self.local_ids,
            valid_tokens,
            self.local_qdy_hi,
            self.local_qdy_lo,
            self.local_qdy_scales,
            self.fp8_raw_score_partials,
            self.status,
        ]
        if self._fp8_raw_gradient_active >= 3:
            dg._C._mega_moe_training_fp8_reuse(
                2,
                [self.pointer_tables["local_dy_feature_amax"], self.fp8_raw_gradient_global_amax],
                [self.size, self.h],
            )
            if self.fp8_raw_gradient_quant_pipeline and self._fp8_raw_gradient_active == 7:
                bm = self.fp8_expert_configs["dhidden"][0]
                dg._C._mega_moe_training_fp8_gradient(
                    15,
                    [self.fp8_raw_gradient_global_amax, self.fp8_raw_gradient_sf_tile, self.status],
                    [self.h, bm, 128],
                )
                self._start_raw_gradient_quant(valid_tokens)
            else:
                dg._C._mega_moe_training_fp8_gradient(
                    7,
                    tensors + [self.fp8_raw_gradient_global_amax],
                    [self.chunk_tokens, self.k, self.h, self.e * self.size, 4, 256],
                )
            if not self.fp8_raw_gradient_score_overlap:
                dg._C._mega_moe_training_fp8_gradient(
                    10, tensors, [self.chunk_tokens, self.k, self.h, self.e * self.size, 4, 256]
                )
        else:
            dg._C._mega_moe_training_fp8_gradient(
                0, tensors, [self.chunk_tokens, self.k, self.h, self.e * self.size, 4, 128]
            )
        if not (self._fp8_raw_gradient_active >= 3 and self.fp8_raw_gradient_score_overlap):
            dg._C._mega_moe_training_fp8_gradient(
                1,
                [self.fp8_raw_score_partials, self.local_dscores, self.status],
                [self.chunk_tokens * self.k, self.h, 4, 128],
            )

    def _fp8_raw_gradient_input(self, *, compile_only=False):
        # Numerical integration first. This preserves the existing per-expert
        # event ordering; later tile publication may refine the producer wait.
        mode = self._fp8_raw_gradient_active or self.fp8_raw_gradient_source
        if mode >= 7:
            return self._fp8_raw_gradient_broadcast_input(compile_only=compile_only)
        if mode in (1, 3) and self.fp8_raw_gradient_tile_pipeline:
            return self._fp8_raw_gradient_input_pipeline(compile_only=compile_only)
        role = "dhidden"
        batch = self.fp8_expert_gradient_batch
        bm = self.fp8_expert_configs[role][0]
        q = self.quantized[self.h]
        sf = self.quantized_scale_storage[self.h]
        main = torch.cuda.current_stream(self.device)
        serial = False or compile_only
        if not serial:
            self._fp8_expert_stream.wait_stream(main)
        try:
            for idx, begin in enumerate(range(0, self.e, batch)):
                end = min(self.e, begin + batch)
                meta = self.fp8_expert_metadata[role][idx]
                tensors = [
                    (
                        self.pointer_tables["local_qdy_hi"]
                        if mode in (1, 3)
                        else self.pointer_tables["raw_dy"]
                    ),
                    self.pointer_tables["local_qdy_scales"],
                    self.pointer_tables["raw_scores"],
                    self.permutation,
                    self.lengths,
                    q[0],
                    sf,
                    self.fp8_raw_gradient_scales,
                    self.fp8_raw_gradient_scores,
                    self.status,
                ]
                tensors += [meta] if mode in (1, 3) else [self.packed_dy, meta]

                def source():
                    dg._C._mega_moe_training_fp8_gradient(
                        2 if mode in (1, 3) else 9 if mode == 4 else 4,
                        tensors,
                        [
                            self.rows,
                            self.h,
                            self.chunk_tokens,
                            self.k,
                            self.size,
                            begin,
                            end,
                            self.fp8_raw_gradient_source_ctas,
                            self.fp8_raw_gradient_source_threads,
                            bm,
                        ],
                        compile_only,
                    )

                if serial:
                    source()
                else:
                    with torch.cuda.stream(self._fp8_expert_stream):
                        for tensor in tensors + [self.payload]:
                            tensor.record_stream(self._fp8_expert_stream)
                        source()
                        self.fp8_expert_events[role][idx].record()
                    main.wait_event(self.fp8_expert_events[role][idx])
                if not compile_only:
                    self._join_fp8_weight_prepare()
                    self._join_dw_source_pair()
                dg._C._mega_moe_training_fp8_gemm(
                    q,
                    self.quantized["w2_t"],
                    self.fp8_expert_dhidden,
                    self.lengths,
                    self.permutation,
                    self.pointer_tables["local_values"],
                    self.chunk_tokens * self.k,
                    keep_local=True,
                    compile_only=compile_only,
                    config=(
                        self.fp8_expert_configs[role][1:]
                        if self.fp8_raw_gradient_tile_pipeline
                        or self.fp8_raw_gradient_gemm_config is not None
                        else []
                    ),
                    expert_metadata=meta,
                    metadata_block_m=bm,
                )
        finally:
            if not serial:
                main.wait_stream(self._fp8_expert_stream)
        if not compile_only:
            self._fp8_prequantized_input = None

    def _fp8_dw_quantize(self, left, right, first=0, last=None):
        if (
            self._fp8_raw_gradient_active not in (1, 3, 4, 7, 8)
            or left.data_ptr() != self.packed_dy.data_ptr()
        ):
            return super()._fp8_dw_quantize(left, right, first, last)
        last = self.e if last is None else last
        a, b = self._fp8_dw_views(left, right)
        if self._fp8_raw_gradient_active >= 7:
            metadata = [self.lengths[first:last], self.offsets[first : last + 1], self.status]
            bm = self.fp8_expert_configs["dhidden"][0]
            dg._C._mega_moe_training_fp8_gradient(
                14,
                [
                    self._raw_gradient_qpair()[0],
                    self.fp8_raw_gradient_sf_tile,
                    *metadata[:2],
                    *a,
                    self.status,
                ],
                [last - first, self.rows, self.h, bm, self.fp8_wgrad_pack_ctas, 128],
            )
            dg._C._mega_moe_training_fp8_gradient(
                12,
                [right, self.fp8_raw_gradient_scores, *metadata[:2], *b, self.status],
                [last - first, self.rows, right.shape[1], self.fp8_wgrad_pack_ctas, 256],
            )
            return (a, b)
        if self._fp8_raw_gradient_active >= 3:
            metadata = [self.lengths[first:last], self.offsets[first : last + 1], self.status]
            dg._C._mega_moe_training_fp8_reuse(
                7,
                [self.quantized[self.h][0], self.fp8_raw_gradient_scales, *a, *metadata],
                [last - first, self.rows, self.h, 1, self.fp8_wgrad_pack_ctas],
            )
            dg._C._mega_moe_training_fp8_gradient(
                8,
                [right, self.fp8_raw_gradient_scores, *metadata[:2], *b, self.status],
                [last - first, self.rows, right.shape[1], self.fp8_wgrad_pack_ctas, 256],
            )
            return (a, b)
        dg._C._mega_moe_training_fp8_gradient(
            3,
            [
                self.quantized[self.h][0],
                self.fp8_raw_gradient_scales,
                self.fp8_raw_gradient_scores,
                self.permutation,
                self.pointer_tables["local_qdy_lo"],
                self.lengths[first:last],
                self.offsets[first : last + 1],
                *a,
                self.status,
            ],
            [
                last - first,
                self.rows,
                self.h,
                self.chunk_tokens,
                self.k,
                self.size,
                self.fp8_wgrad_pack_ctas,
                128,
            ],
        )
        dg._C._mega_moe_training_fp8_reuse(
            6,
            [right, *b, self.lengths[first:last], self.offsets[first : last + 1], self.status],
            [last - first, self.rows, right.shape[1], self.fp8_wgrad_pack_ctas],
        )
        return (a, b)

    def _fp8_forward_tile_pipeline(self, *, compile_only=False):
        cfg = self.fp8_expert_configs["fc1"]
        bm = cfg[0]
        meta = self.fp8_expert_metadata["fc1"][0]
        ready = self.fp8_inline_ready["fc1"]
        inputs = (self.fp8_source_x_q, self.fp8_source_x_sf.t())
        _mega_moe(45, [self.lengths, meta, self.status], [self.e, self.rows, bm], compile_only)
        source = [
            self.pointer_tables["local_qx"],
            self.pointer_tables["local_qx_scales"],
            self.permutation,
            meta,
            inputs[0],
            self.fp8_source_x_sf,
            ready,
            self.status,
        ]
        dims = [
            self.rows,
            self.h,
            self.chunk_tokens,
            self.k,
            self.size,
            bm,
            self.fp8_forward_tile_ctas,
            self.fp8_forward_tile_threads,
            self.rank,
            ready.numel(),
            self.e,
        ]
        main = torch.cuda.current_stream(self.device)
        if not compile_only:
            ready.zero_()
            self._join_fp8_weight_prepare()
            self._join_dw_source_pair()
            self._fp8_expert_stream.wait_stream(main)
            with torch.cuda.stream(self._fp8_expert_stream):
                for tensor in source:
                    tensor.record_stream(self._fp8_expert_stream)
                dg._C._mega_moe_training_fp8_gradient(34, source, dims, False)
        else:
            dg._C._mega_moe_training_fp8_gradient(34, source, dims, True)
        dg._C._mega_moe_training_fp8_gemm(
            inputs,
            self.quantized["w1"],
            self.gate_up,
            self.lengths,
            self.permutation,
            self.pointer_tables["local_values"],
            self.chunk_tokens * self.k,
            keep_local=True,
            compile_only=compile_only,
            config=cfg[1:],
            expert_metadata=meta,
            metadata_block_m=bm,
            metadata_ctas=self.fp8_forward_gemm_ctas,
            inline_sources=[ready, self.status],
            return_local_rank=self.rank,
        )
        if not compile_only:
            main.wait_stream(self._fp8_expert_stream)
        self._fp8_prequantized_input = None

    def _fp8_expert_input_inline(self, *, gradient=False, compile_only=False):
        if not gradient and self.fp8_forward_tile_source:
            return self._fp8_forward_tile_pipeline(compile_only=compile_only)
        role = "dhidden" if gradient else "fc1"
        cfg = self.fp8_expert_configs[role]
        bm = cfg[0]
        meta = self.fp8_expert_metadata[role][0]
        ready = self.fp8_inline_ready[role]
        inputs = (
            self.quantized[self.h] if gradient else (self.fp8_source_x_q, self.fp8_source_x_sf.t())
        )
        _mega_moe(45, [self.lengths, meta, self.status], [self.e, self.rows, bm], compile_only)
        if not compile_only:
            ready.zero_()
            self._join_fp8_weight_prepare()
            self._join_dw_source_pair()
        sources = (
            [
                self.pointer_tables["raw_dy"],
                self.pointer_tables["raw_scores"],
                self.permutation,
                self.packed_output,
                self.packed_dy,
                self.fp8_dscore_partials,
                ready,
                self.status,
            ]
            if gradient
            else [
                self.pointer_tables["local_qx"],
                self.pointer_tables["local_qx_scales"],
                self.permutation,
                ready,
                self.status,
            ]
        )
        dg._C._mega_moe_training_fp8_gemm(
            inputs,
            self.quantized["w2_t" if gradient else "w1"],
            self.fp8_expert_dhidden if gradient else self.gate_up,
            self.lengths,
            self.permutation,
            self.pointer_tables["local_values"],
            self.chunk_tokens * self.k,
            keep_local=True,
            compile_only=compile_only,
            config=cfg[1:],
            expert_metadata=meta,
            metadata_block_m=bm,
            inline_sources=sources,
            inline_tokens=self.chunk_tokens,
            inline_topk=self.k,
            inline_ranks=self.size,
            inline_bulk_ilp=self.fp8_inline_bulk_ilp if not gradient else 4,
            inline_source_bulk=bool(self.fp8_inline_source_bulk and (not gradient)),
            inline_source_rotation=bool(0),
            return_local_rank=self.rank,
        )
        self._fp8_prequantized_input = None

    def _fp8_expert_input(self, *, gradient=False, compile_only=False, prepare_fallback=False):
        if (
            gradient
            and self.fp8_raw_gradient_source
            and (self._fp8_raw_gradient_active or compile_only)
            and not prepare_fallback
        ):
            return self._fp8_raw_gradient_input(compile_only=compile_only)
        if self.fp8_inline_source and (not gradient or self.fp8_inline_source == 2):
            return self._fp8_expert_input_inline(gradient=gradient, compile_only=compile_only)
        role = "dhidden" if gradient else "fc1"
        input_quantized = self.quantized[self.h]
        input_scale_storage = self.quantized_scale_storage[self.h]
        if not gradient and self.fp8_dw_from_row_cache:
            input_quantized = (self.fp8_source_x_q, self.fp8_source_x_sf.t())
            input_scale_storage = self.fp8_source_x_sf
        batch = self.fp8_expert_gradient_batch if gradient else self.fp8_expert_forward_batch
        if not batch:
            raise RuntimeError("expert input role is disabled")
        bm = self.fp8_expert_configs[role][0]
        main = torch.cuda.current_stream(self.device)
        serial = False or compile_only
        if not serial:
            self._fp8_expert_stream.wait_stream(main)
        try:
            for idx, begin in enumerate(range(0, self.e, batch)):
                end = min(self.e, begin + batch)
                meta = self.fp8_expert_metadata[role][idx]
                tensors = (
                    [
                        self.pointer_tables["raw_dy"],
                        self.pointer_tables["raw_scores"],
                        self.pointer_tables["local_dscores"],
                        self.permutation,
                        self.packed_output,
                        self.packed_dy,
                        self.lengths,
                        self.status,
                        input_quantized[0],
                        input_scale_storage,
                        self.fp8_dscore_partials,
                    ]
                    if gradient
                    else [
                        self.pointer_tables["local_x"],
                        self.permutation,
                        self.packed_x,
                        self.status,
                        self.lengths,
                        input_quantized[0],
                        input_scale_storage,
                    ]
                )

                def source():
                    source_tensors = tensors + [meta]
                    source_dims = [
                        self.rows,
                        self.h,
                        self.chunk_tokens,
                        self.k,
                        self.size,
                        begin,
                        end,
                        self.fp8_expert_gradient_ctas if gradient else self.fp8_expert_forward_ctas,
                        self.fp8_expert_source_rows,
                        bm,
                    ]
                    if not gradient and self.fp8_expert_source_fp8:
                        source_tensors.extend(
                            [
                                self.pointer_tables["local_qx"],
                                self.pointer_tables["local_qx_scales"],
                            ]
                        )
                        source_dims.append(2 if self.fp8_source_shared_base else 1)
                    _mega_moe(29 if gradient else 28, source_tensors, source_dims, compile_only)

                if serial:
                    source()
                else:
                    with torch.cuda.stream(self._fp8_expert_stream):
                        extra = (
                            []
                            if gradient or not self.fp8_expert_source_fp8
                            else [
                                self.pointer_tables["local_qx"],
                                self.pointer_tables["local_qx_scales"],
                            ]
                        )
                        for tensor in tensors + [meta, self.payload] + extra:
                            tensor.record_stream(self._fp8_expert_stream)
                        source()
                        self.fp8_expert_events[role][idx].record()
                    main.wait_event(self.fp8_expert_events[role][idx])
                if not compile_only:
                    self._join_fp8_weight_prepare()
                    self._join_dw_source_pair()
                dg._C._mega_moe_training_fp8_gemm(
                    input_quantized,
                    self.quantized["w2_t" if gradient else "w1"],
                    self.fp8_expert_dhidden if gradient else self.gate_up,
                    self.lengths,
                    self.permutation,
                    self.pointer_tables["local_values"],
                    self.chunk_tokens * self.k,
                    keep_local=True,
                    compile_only=compile_only,
                    config=(
                        self.fp8_expert_configs[role][1:]
                        if gradient and self.fp8_raw_gradient_gemm_config is not None
                        else []
                    ),
                    expert_metadata=meta,
                    metadata_block_m=bm,
                )
        finally:
            if not serial:
                main.wait_stream(self._fp8_expert_stream)
        if not compile_only:
            self._fp8_prequantized_input = None

    def _join_dw_source_pair(self):
        if self._dw_source_pending:
            torch.cuda.current_stream(self.device).wait_event(self._dw_source_done)
            self._dw_source_pending = False

    def _fc1(
        self, w1, *, pack_input=False, compile_only=False, keep_gate_up=True, prefetch_dy=False
    ):
        if self.fp8_expert_forward_batch:
            self._fp8_expert_input(compile_only=compile_only)
            return
        if not compile_only:
            self._join_fp8_weight_prepare()
        if self.dispatch_producer:
            bn = self.gemm_configs.get("fc1", (128, 64, 64, 64, 3))[0]
            ntiles = (2 * self.i + bn - 1) // bn
            produced = min(ntiles, self.dispatch_producer_tiles)
            if self.bf16_source_owned_tiles & 1:
                produced = ntiles
            # The first GEMM slice dispatches X and computes real FC1 columns.
            # Stream ordering then publishes packed X to a local GEMM over
            # the remaining columns. No cross-CTA wait or dispatch-only kernel.
            self._peer_gemm(
                self.packed_x,
                w1,
                self.gate_up,
                self.source_rows,
                read=True,
                pack_input=True,
                role="fc1",
                compile_only=compile_only,
                first_n_tile=0,
                num_n_tiles=0 if self.bf16_source_owned_tiles & 1 else produced,
                producer_input_aiu=self.producer_input_aiu,
                keep_gate_up=keep_gate_up,
            )
            # X dispatch is complete. Move raw DY while the remaining FC1
            # columns and FC2 replay compute, avoiding competition with X.
            if prefetch_dy and (not compile_only):
                self._start_gradient_prefetch()
            if produced < ntiles:
                self._peer_gemm(
                    self.packed_x,
                    w1,
                    self.gate_up,
                    self.identity_rows,
                    role="fc1",
                    compile_only=compile_only,
                    first_n_tile=produced,
                    local_input_aiu=self.producer_local_aiu,
                    keep_gate_up=keep_gate_up,
                )
        else:
            self._peer_gemm(
                self.packed_x,
                w1,
                self.gate_up,
                self.source_rows if self.fuse_dispatch else self.identity_rows,
                read=self.fuse_dispatch,
                pack_input=pack_input and self.fuse_dispatch,
                role="fc1",
                compile_only=compile_only,
                keep_gate_up=keep_gate_up,
            )

    @torch.no_grad()
    def prepare(self, w1, w2):
        """Compile all peer GEMM variants before entering device peer barriers."""
        with self._execution():
            if not self._prepared:
                error = None
                try:
                    self._prepare(w1, w2)
                except Exception as exception:
                    error = exception
                self._finish_preparation(error)

    def _precompile_fp8_sources_and_dw(self, w1, w2):
        # All launches here are compile-only. Complete cold compilation before
        # _finish_preparation agrees across ranks and any device peer wait runs.
        if not self.fp8_peer_compute:
            return
        if self.fp8_source_shared_base:
            dg._C._mega_moe_training_fp8_reuse(
                0,
                [self.local_x, self._source_row_amax],
                [self.chunk_tokens, self.h],
                True,
            )
            dg._C._mega_moe_training_fp8_reuse(
                1,
                [self._source_row_amax, self.local_x_feature_amax],
                [self.chunk_tokens, self.h],
                True,
            )
            dg._C._mega_moe_training_fp8_reuse(
                2,
                [self.pointer_tables["local_x_feature_amax"], self._source_global_amax],
                [self.size, self.h],
                True,
            )
            dg._C._mega_moe_training_fp8_reuse(
                3,
                [
                    self.local_x,
                    self._source_row_amax,
                    self._source_global_amax,
                    self.local_qx,
                    self.local_qx_scales,
                ],
                [self.chunk_tokens, self.h],
                True,
            )
        if self.fp8_dw_stable_routing:
            tensors = [
                self.pointer_tables["local_ids"],
                self.counts,
                self.route_experts,
                self.route_rows,
                self.status,
                self._stable_route_hist,
                self._stable_route_prefix,
            ]
            dims = [self.routes, self.e, self.rank, self.size, self.chunk_tokens * self.k]
            dg._C._mega_moe_training_stable_routes(
                tensors,
                dims,
                False,
                True,
                block_routes=self.fp8_dw_route_block,
            )
            dg._C._mega_moe_training_stable_routes(
                tensors,
                dims,
                True,
                True,
                block_routes=self.fp8_dw_route_block,
                scan_prefix=self.fp8_dw_native_prefix,
            )
        if self.fp8_raw_gradient_source >= 3 and self.fp8_raw_gradient_score_overlap:
            dg._C._mega_moe_training_fp8_gradient(
                {4: 10, 1: 18, 2: 19}[self.fp8_raw_gradient_score_ilp],
                [
                    self.raw_dy,
                    self.raw_scores,
                    self.local_values,
                    self.local_ids,
                    self.lengths[:1],
                    self.local_qdy_hi,
                    self.local_qdy_lo,
                    self.local_qdy_scales,
                    self.fp8_raw_score_partials,
                    self.status,
                ],
                [
                    self.chunk_tokens,
                    self.k,
                    self.h,
                    self.e * self.size,
                    4,
                    self.fp8_raw_gradient_score_threads,
                ],
                True,
            )
            dg._C._mega_moe_training_fp8_gradient(
                1,
                [self.fp8_raw_score_partials, self.local_dscores, self.status],
                [self.chunk_tokens * self.k, self.h, 4, 128],
                True,
            )
        if not (self.fp8_wgrad and self.fp8_dw_from_row_cache):
            return
        # Compile-only DW calls can share one scratch allocation. Their views
        # retain its lifetime until the batch completes without retaining two
        # complete FP32 expert-gradient tensors.
        warmup_output = torch.empty(
            max(w1.numel(), w2.numel()), device=self.device, dtype=torch.float32
        )
        for left, right, weight in (
            (self.gate_up, self.packed_x, w1),
            (self.packed_dy, self.hidden, w2),
        ):
            a, b = self._fp8_dw_views(left, right)
            metadata = [self.lengths, self.offsets, self.status]
            if self.fp8_raw_gradient_source >= 7 and left.data_ptr() == self.packed_dy.data_ptr():
                dg._C._mega_moe_training_fp8_gradient(
                    14,
                    [
                        self._raw_gradient_qpair()[0],
                        self.fp8_raw_gradient_sf_tile,
                        *metadata[:2],
                        *a,
                        self.status,
                    ],
                    [
                        self.e,
                        self.rows,
                        self.h,
                        self.fp8_expert_configs["dhidden"][0],
                        self.fp8_wgrad_pack_ctas,
                        128,
                    ],
                    True,
                )
                dg._C._mega_moe_training_fp8_gradient(
                    12,
                    [right, self.fp8_raw_gradient_scores, *metadata[:2], *b, self.status],
                    [self.e, self.rows, right.shape[1], self.fp8_wgrad_pack_ctas, 256],
                    True,
                )
            dg._C._mega_moe_training_fp8_reuse(
                5,
                [left, *a, *metadata],
                [self.e, self.rows, left.shape[1], self.fp8_wgrad_pack_ctas],
                True,
            )
            if right.data_ptr() == self.packed_x.data_ptr():
                dg._C._mega_moe_training_fp8_reuse(
                    4,
                    [self.fp8_source_x_q, self.fp8_source_x_sf, *b, *metadata],
                    [self.e, self.rows, right.shape[1], 1, self.fp8_wgrad_pack_ctas],
                    True,
                )
            else:
                dg._C._mega_moe_training_fp8_reuse(
                    6,
                    [right, *b, *metadata],
                    [self.e, self.rows, right.shape[1], self.fp8_wgrad_pack_ctas],
                    True,
                )
            compact = (
                a[0],
                torch.empty(
                    (self.rows // 128, left.shape[1] // self.fp8_dw_scale_m),
                    device=self.device,
                    dtype=torch.float32,
                ),
            )
            output = warmup_output[: weight.numel()].view(weight.shape)
            flags = [None]
            if self.wgrad_first_nonempty:
                flags.append(self._wgrad_first_flags)
            for first in flags:
                dg._C._mega_moe_training_fp8_wgrad_nt(
                    compact,
                    b,
                    output,
                    self.lengths,
                    self.offsets,
                    self.status,
                    compile_only=True,
                    config=self.fp8_wgrad_config,
                    block_scales=True,
                    promotion_k=self.fp8_wgrad_block_k,
                    first_flags=first,
                    native_blockwise=self.fp8_wgrad_native_blockwise,
                    swizzle_m=self.fp8_dw_swizzle_m,
                    metadata_mode=self.fp8_dw_metadata_mode,
                    scale_m=self.fp8_dw_scale_m,
                )

    def _prepare(self, w1, w2):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm up MegaMoEPlan before graph capture")
        if self.fp8_peer_compute:
            self.lengths.zero_()
            self.offsets.zero_()
            self.num_blocks.zero_()
            self._fp8_prequantized_input = None
        self._prepare_pointwise()
        w1, w2 = (w1.to(torch.bfloat16), w2.to(torch.bfloat16))
        self._prepare_weights(w1, w2, backward=True)
        self._precompile_fp8_sources_and_dw(w1, w2)
        if self.expert_local_grad:
            self._expert_gradients(compile_only=True)
        if self.fp8_expert_gradient_batch and self.fp8_raw_gradient_source:
            # Multiple chunks/windows can disable the raw-source fast
            # path at runtime. Its ordinary gradient family must be warm
            # too, before any rank starts publishing peer payloads.
            self._fp8_expert_input(
                gradient=True,
                compile_only=True,
                prepare_fallback=True,
            )
        if self.overlap_expert_dy:
            self._prefetch_raw_dy(compile_only=True)
        if self.ring_rows is not None:
            set_route_window(self, 0, compile_only=True)
        if self.reuse_forward_context:
            self._prepare_context_replay(0, compile_only=True)
        self._barrier(0, compile_only=True)
        _mega_moe(
            0,
            [
                self.pointer_tables["local_ids"],
                self.counts,
                self.route_experts,
                self.route_rows,
                self.status,
            ],
            [self.routes, self.e, self.rank, self.size, self.chunk_tokens * self.k],
            True,
        )
        _mega_moe(
            1,
            [self.permutation, self.source_rows],
            [self.rows, self.size * self.chunk_tokens],
            True,
        )
        _mega_moe(
            2,
            [
                self.route_experts,
                self.route_rows,
                self.offsets,
                self.permutation,
                self.source_rows,
                self.status,
            ],
            [self.routes, self.rows, self.k],
            True,
        )
        _mega_moe(
            3,
            [self.offsets, self.lengths, self.work_info, self.num_blocks],
            [self.e, self.rows, self.block_m],
            True,
        )
        _mega_moe(
            4,
            [self.pointer_tables["local_x"], self.permutation, self.packed_x, self.status],
            [self.rows, self.h, self.chunk_tokens, self.k, self.size],
            True,
        )
        if not self.fuse_dispatch and (not self.fp8_expert_forward_batch):
            self._stage_input(compile_only=True)
        if self.active_rows:
            self._activation(compile_only=True)
            self._activation(backward=True, compile_only=True)
            if not self.expert_local_grad:
                self._pack_dy(compile_only=True)
        if self.fuse_local_grad and not self.expert_local_grad:
            empty_dy = torch.empty((0, self.h), device=self.device, dtype=torch.bfloat16)
            empty_scores = torch.empty((0, self.k), device=self.device, dtype=torch.float32)
            self._local_gradients(
                empty_dy,
                empty_scores,
                empty_scores,
                0,
                compile_only=True,
            )
        self._fc1(w1, compile_only=True)
        if self.fuse_fc1_activation:
            self._fc1(w1, compile_only=True, keep_gate_up=False)
        self._peer_gemm(
            self.hidden,
            w2,
            self.packed_output,
            self.identity_rows,
            write=self.fuse_return,
            compile_only=True,
        )
        if self.reuse_forward_context and self.fp8_peer_compute:
            # Backward recomputation also uses the local FC2 epilogue.
            self._peer_gemm(
                self.hidden,
                w2,
                self.packed_output,
                self.identity_rows,
                write=False,
                compile_only=True,
            )
            previous_backward = self._fp8_backward
            self._fp8_backward = False
            self._peer_gemm(
                self.hidden,
                w2,
                self.packed_output,
                self.identity_rows,
                write=self.fuse_return,
                compile_only=True,
            )
            self._fp8_backward = previous_backward
        if self.fuse_dhidden_scores:
            self._expert_gradients(compile_only=True)
            self._reduce_score_partials(compile_only=True)
        read_dy = self.fuse_dispatch and (not self.reuse_packed_dy)
        self._peer_gemm(
            self.packed_output if self.fuse_dhidden_scores else self.packed_dy,
            self.w2_t,
            self.hidden,
            self.permutation if read_dy else self.identity_rows,
            read=read_dy,
            source="local_dy",
            compile_only=True,
            role="dhidden",
        )
        self._peer_gemm(
            self.gate_up,
            self.w1_t,
            self.packed_output,
            self.identity_rows,
            write=self.fuse_return,
            destination="local_dx",
            compile_only=True,
        )
        if not self.fuse_return:
            self._scatter_peer("local_values", compile_only=True)
        self.lengths.zero_()
        self.offsets.zero_()
        flags = [None]
        if self.wgrad_first_nonempty:
            if not self.fp8_wgrad:
                flags.append(self._wgrad_first_flags)
            _mega_moe(
                3,
                [
                    self.offsets,
                    self.lengths,
                    self.work_info,
                    self.num_blocks,
                    self._wgrad_seen,
                    self._wgrad_first_flags,
                ],
                [self.e, self.rows, self.block_m],
                True,
            )
        for left, right, weight in (
            (self.gate_up, self.packed_x, w1),
            (self.packed_dy, self.hidden, w2),
        ):
            # Zero-length launches warm the same beta/first-nonempty variants
            # used with live routes. No payload or persistent DW is published.
            warmup_dw = torch.zeros_like(weight, dtype=torch.float32)
            if not self.fp8_wgrad:
                for first in flags:
                    dg.k_grouped_bf16_gemm_tn_contiguous(
                        left,
                        right,
                        warmup_dw,
                        None,
                        self.lengths,
                        c=warmup_dw,
                        max_group_k=self.rows,
                        first_flags=first,
                    )
                if self.wgrad_aggregate_windows and len(self.forward_context_slots) > 1:
                    segment_counts = {min(2, math.ceil(self.max_tokens / self.chunk_tokens))}
                    if math.ceil(self.max_tokens / self.chunk_tokens) % 2:
                        segment_counts.add(1)
                    for count in sorted(segment_counts):
                        for first in flags:
                            dg.k_grouped_bf16_gemm_tn_segmented(
                                [left] * count,
                                [right] * count,
                                warmup_dw,
                                [self.offsets] * count,
                                self.rows,
                                accumulate=True,
                                max_continuous_k=10240,
                                first_flags=first,
                            )
                    if self.wgrad_first_nonempty:
                        chunks = max(1, math.ceil(self.max_tokens / self.chunk_tokens))
                        _mega_moe(
                            44,
                            [
                                self._wgrad_seen,
                                self._aggregate_first_flags,
                                *[self.offsets] * chunks,
                            ],
                            [self.e, chunks],
                            True,
                        )
            if self.wgrad_first_nonempty:
                _mega_moe(
                    43,
                    [warmup_dw, self._wgrad_seen],
                    [self.e, warmup_dw[0].numel()],
                    True,
                )
            del warmup_dw
        torch.cuda.synchronize(self.device)

    def _route_chunk(self, x, ids, valid_tokens, start, zero_values=True):
        self._activate_forward_context(start)
        _launch(
            13,
            [x, ids, self.local_x, self.local_ids, valid_tokens, self.status],
            [x.shape[0], self.chunk_tokens, self.h, self.k, start],
        )
        if self.fp8_source_shared_base:
            self._source_shared_local_stats()
        elif self.fp8_expert_source_fp8:
            _launch(
                9, [self.local_x, self.local_qx, self.local_qx_scales], [self.chunk_tokens, self.h]
            )
        if zero_values and (not self.skip_masked_route_zero):
            self.local_values.zero_()
        if not self.skip_masked_route_zero:
            self.local_dx.zero_()
        self._barrier(start * 4)
        if self.fp8_source_shared_base:
            self._source_shared_quantize()
            self._barrier(start * 4 + 2)
        if self.fp8_dw_stable_routing:
            tensors = [
                self.pointer_tables["local_ids"],
                self.counts,
                self.route_experts,
                self.route_rows,
                self.status,
                self._stable_route_hist,
                self._stable_route_prefix,
            ]
            dims = [self.routes, self.e, self.rank, self.size, self.chunk_tokens * self.k]
            dg._C._mega_moe_training_stable_routes(
                tensors, dims, block_routes=self.fp8_dw_route_block
            )
            if not self.fp8_dw_native_prefix:
                torch.cumsum(
                    self._stable_route_hist, dim=0, dtype=torch.int32, out=self._stable_route_prefix
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
            _mega_moe(
                0,
                [
                    self.pointer_tables["local_ids"],
                    self.counts,
                    self.route_experts,
                    self.route_rows,
                    self.status,
                ],
                [self.routes, self.e, self.rank, self.size, self.chunk_tokens * self.k],
            )
        offsets = self.offsets if self.ring_rows is None else self.global_offsets
        lengths = self.lengths if self.ring_rows is None else self.global_lengths
        permutation = self.permutation if self.ring_rows is None else self.global_permutation
        source_rows = self.source_rows if self.ring_rows is None else self.global_source_rows
        capacity = self.rows if self.ring_rows is None else self.full_row_capacity
        if self.route_capacity is None:
            _launch(1, [self.counts, offsets, lengths], [self.e, self.alignment])
        else:
            _launch(
                40,
                [self.counts, offsets, lengths, self.status],
                [self.e, self.alignment, capacity],
            )
        _mega_moe(1, [permutation, source_rows], [capacity, self.size * self.chunk_tokens])
        _mega_moe(
            2,
            [self.route_experts, self.route_rows, offsets, permutation, source_rows, self.status],
            [self.routes, capacity, self.k],
        )
        if self.ring_rows is not None:
            return
        _mega_moe(
            3,
            [self.offsets, self.lengths, self.work_info, self.num_blocks],
            [self.e, self.rows, self.block_m],
        )

    def _chunk_forward(self, w1, w2, pack_input=False, return_output=True):
        if self.ring_rows is not None:
            if not return_output:
                raise RuntimeError("backward routed-row windows own their replay schedule")
            for start in range(0, self.full_row_capacity, self.rows):
                self._set_window(start)
                self._chunk_forward_single(w1, w2, pack_input=pack_input, synchronize_output=False)
            self._barrier(1)
            return
        self._chunk_forward_single(w1, w2, pack_input=pack_input, return_output=return_output)

    def _set_window(self, start):
        set_route_window(self, start)
        tensors = [self.offsets, self.lengths, self.work_info, self.num_blocks]
        if self._fresh_first_dw_step and (not self._aggregate_window_active):
            tensors.extend([self._wgrad_seen, self._wgrad_first_flags])
        _mega_moe(3, tensors, [self.e, self.rows, self.block_m])

    def _chunk_forward_single(
        self,
        w1,
        w2,
        pack_input=False,
        return_output=True,
        synchronize_output=True,
        prefetch_dy=False,
        _context_masked=False,
    ):
        if self._using_forward_context and (not return_output) and (not _context_masked):
            self._prepare_context_replay(self._context_window_start)
            original = (self.offsets, self.lengths, self.num_blocks, self.current_rows)
            self.offsets, self.lengths, self.num_blocks = (
                self.replay_offsets,
                self.replay_lengths,
                self.replay_blocks,
            )
            self.current_rows = self.replay_offsets[self.e : self.e + 1]
            try:
                self._chunk_forward_single(
                    w1,
                    w2,
                    pack_input,
                    return_output,
                    synchronize_output,
                    prefetch_dy,
                    _context_masked=True,
                )
            finally:
                self.offsets, self.lengths, self.num_blocks, self.current_rows = original
            return
        if not self.fuse_dispatch and (not self.fp8_expert_forward_batch):
            self._stage_input()
        # A reused BF16 forward must retain expert-packed X for DW1 even
        # when FC1 directly reads peers without a separate producer.
        self._fc1(
            w1,
            pack_input=pack_input or (return_output and self.reuse_forward_context),
            keep_gate_up=not return_output or self.reuse_forward_context,
            prefetch_dy=prefetch_dy,
        )
        if not self.fuse_fc1_activation:
            self._activation()
        if not return_output and self.fuse_dhidden_scores:
            return
        self._peer_gemm(
            self.hidden,
            w2,
            self.packed_output,
            self.identity_rows,
            write=self.fuse_return and return_output,
            replay_gradients=False and (not return_output),
        )
        if not return_output:
            return
        if not self.fuse_return:
            self._scatter_peer("local_values")
        if synchronize_output:
            self._barrier(1)

    def _scatter_peer(self, destination, compile_only=False):
        _mega_moe(
            12 if self.vectorized_copy else 6,
            [
                self.packed_output,
                self.permutation,
                self.pointer_tables[destination],
                self.status,
                self.num_blocks,
            ],
            [self.rows, self.h, self.chunk_tokens * self.k, self.size, self.block_m],
            compile_only,
        )

    def _fp8_stage_quantized_input(self, compile_only=False):
        _mega_moe(
            19,
            [
                self.pointer_tables["local_qx"],
                self.pointer_tables["local_qx_scales"],
                self.pointer_tables["local_x"],
                self.source_rows,
                self.quantized[self.h][0],
                self.quantized_scale_storage[self.h],
                self.packed_x,
                self.num_blocks,
                self.fp8_input_ready,
                self.status,
            ],
            [
                self.rows,
                self.h,
                self.chunk_tokens,
                self.size,
                self.fp8_input_ctas,
                0,
                self.fp8_input_threads,
                int(False),
            ],
            compile_only,
        )

    def _source_shared_local_stats(self):
        dg._C._mega_moe_training_fp8_reuse(
            0, [self.local_x, self._source_row_amax], [self.chunk_tokens, self.h]
        )
        dg._C._mega_moe_training_fp8_reuse(
            1, [self._source_row_amax, self.local_x_feature_amax], [self.chunk_tokens, self.h]
        )

    def _source_shared_quantize(self):
        dg._C._mega_moe_training_fp8_reuse(
            2,
            [self.pointer_tables["local_x_feature_amax"], self._source_global_amax],
            [self.size, self.h],
        )
        dg._C._mega_moe_training_fp8_reuse(
            3,
            [
                self.local_x,
                self._source_row_amax,
                self._source_global_amax,
                self.local_qx,
                self.local_qx_scales,
            ],
            [self.chunk_tokens, self.h],
        )

    def _start_source_bf16_refetch(self):
        # Window metadata and source X remain live through the existing join.
        main = torch.cuda.current_stream(self.device)
        self._fp8_bf_input_stream.wait_stream(main)
        with torch.cuda.stream(self._fp8_bf_input_stream):
            for tensor in (
                self.permutation,
                self.num_blocks,
                self.packed_x,
                self.status,
                self.pointer_tables["local_x"],
            ):
                tensor.record_stream(self._fp8_bf_input_stream)
            self._stage_input(force_bf16=True)
        self._fp8_bf_input_pending = True

    def _join_fp8_bf_input(self):
        if self._fp8_bf_input_pending:
            torch.cuda.current_stream(self.device).wait_stream(self._fp8_bf_input_stream)
            self._fp8_bf_input_pending = False

    def _stage_input(self, compile_only=False, force_bf16=False, blocks_override=None):
        if self.fp8_fused_stage_quant & 1 and (not force_bf16):
            _mega_moe(
                21,
                [
                    self.pointer_tables["local_x"],
                    self.permutation,
                    self.packed_x,
                    self.status,
                    self.num_blocks if blocks_override is None else blocks_override,
                    self.quantized[self.h][0],
                    self.quantized_scale_storage[self.h],
                ],
                [self.rows, self.h, self.chunk_tokens, self.k, self.size, self.block_m],
                compile_only,
            )
            if not compile_only:
                self._fp8_prequantized_input = self.packed_x.data_ptr()
            return
        _mega_moe(
            11 if self.vectorized_copy else 7,
            [
                self.pointer_tables["local_x"],
                self.permutation,
                self.packed_x,
                self.status,
                self.num_blocks if blocks_override is None else blocks_override,
            ],
            [self.rows, self.h, self.chunk_tokens, self.k, self.size, self.block_m],
            compile_only,
        )

    def _activation(self, backward=False, compile_only=False):
        if self.fp8_peer_compute:
            if not compile_only:
                width = 2 * self.i if backward else self.i
                hidden = (
                    self.fp8_expert_dhidden
                    if backward and self.fp8_expert_gradient_batch
                    else self.hidden
                )
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
                    [self.rows, self.i, 1, int(self.fp8_vector_activation)],
                )
                self._fp8_prequantized_input = (
                    self.gate_up.data_ptr() if backward else self.hidden.data_ptr()
                )
            return
        if self.active_rows:
            _mega_moe(
                9 if backward else 8,
                [self.gate_up, self.hidden, self.num_blocks, self.status],
                [self.rows, self.i, self.block_m] + ([8] if self.g8_interleaved else []),
                compile_only,
            )
        elif not compile_only:
            tensors = [self.gate_up, self.hidden] + ([self.gate_up] if backward else [])
            _launch(
                8 if backward else 7,
                tensors,
                [self.rows, self.i] + ([8] if self.g8_interleaved else []),
            )

    def _pack_dy(self, compile_only=False):
        tensors = [self.pointer_tables["local_dy"], self.permutation, self.packed_dy, self.status]
        if self.active_rows:
            tensors.append(self.num_blocks)
        operation = (11 if self.vectorized_copy else 7) if self.active_rows else 4
        _mega_moe(
            operation,
            tensors,
            [self.rows, self.h, self.chunk_tokens * self.k, 1, self.size]
            + ([self.block_m] if self.active_rows else []),
            compile_only,
        )

    def _local_gradients(self, dy, scores, dscores, start, compile_only=False):
        _mega_moe(
            10,
            [self.local_values, dy, scores, dscores, self.local_dy],
            [self.chunk_tokens, self.h, self.k, dy.shape[0], start],
            compile_only,
        )

    def _reduce_score_partials(self, compile_only=False):
        _mega_moe(
            17,
            [
                self.score_partials,
                self.permutation,
                self.num_blocks,
                self.pointer_tables["local_dscores"],
                self.status,
            ],
            [
                self.rows,
                self.score_partial_columns,
                self.chunk_tokens,
                self.k,
                self.size,
                self.block_m,
            ],
            compile_only,
        )

    def _expert_gradients(self, compile_only=False):
        if self.fp8_expert_gradient_batch:
            self._fp8_expert_input(gradient=True, compile_only=compile_only)
            if self._fp8_raw_gradient_active:
                return
            _mega_moe(
                17,
                [
                    self.fp8_dscore_partials,
                    self.permutation,
                    self.num_blocks,
                    self.pointer_tables["local_dscores"],
                    self.status,
                ],
                [self.rows, self.h // 128, self.chunk_tokens, self.k, self.size, self.block_m],
                compile_only,
            )
            return
        if self.bf16_grad_producer:
            self._peer_gemm(
                self.packed_output,
                self.w2_t,
                self.hidden,
                self.source_rows,
                read=True,
                pack_input=True,
                source="raw_dy",
                role="dhidden",
                compile_only=compile_only,
                first_n_tile=0,
                num_n_tiles=0 if self.bf16_source_owned_tiles & 2 else 1,
                producer_input_aiu=True,
            )
            return
        if self.fuse_dhidden_scores:
            _mega_moe(
                18,
                [
                    self.pointer_tables["raw_dy"],
                    self.pointer_tables["raw_scores"],
                    self.permutation,
                    self.packed_output,
                    self.packed_dy,
                    self.packed_row_scores,
                    self.num_blocks,
                    self.status,
                ],
                [self.rows, self.h, self.chunk_tokens, self.k, self.size, self.block_m],
                compile_only,
            )
            return
        tensors = [
            self.pointer_tables["raw_dy"],
            self.pointer_tables["raw_scores"],
            self.pointer_tables["local_dscores"],
            self.permutation,
            self.packed_output,
            self.packed_dy,
            self.num_blocks,
            self.status,
        ]
        if self.overlap_expert_dy:
            if not compile_only:
                self._join_gradient_prefetch()
            tensors.append(self.packed_raw_dy)
        if self.fp8_fused_stage_quant & 2:
            tensors.extend((self.quantized[self.h][0], self.quantized_scale_storage[self.h]))
            if self.fp8_tiled_gradient_quant:
                tensors.append(self.fp8_dscore_partials)
                _mega_moe(
                    24 if self.fp8_tiled_gradient_cache else 23,
                    tensors,
                    [self.rows, self.h, self.chunk_tokens, self.k, self.size, self.block_m],
                    compile_only,
                )
                _mega_moe(
                    17,
                    [
                        self.fp8_dscore_partials,
                        self.permutation,
                        self.num_blocks,
                        self.pointer_tables["local_dscores"],
                        self.status,
                    ],
                    [self.rows, self.h // 128, self.chunk_tokens, self.k, self.size, self.block_m],
                    compile_only,
                )
            else:
                _mega_moe(
                    22,
                    tensors,
                    [self.rows, self.h, self.chunk_tokens, self.k, self.size, self.block_m],
                    compile_only,
                )
            if not compile_only:
                self._fp8_prequantized_input = self.packed_dy.data_ptr()
        else:
            _mega_moe(
                13,
                tensors,
                [self.rows, self.h, self.chunk_tokens, self.k, self.size, self.block_m],
                compile_only,
            )

    def _prefetch_raw_dy(self, compile_only=False):
        _mega_moe(
            15,
            [
                self.pointer_tables["raw_dy"],
                self.permutation,
                self.packed_raw_dy,
                self.num_blocks,
                self.status,
            ],
            [self.rows, self.h, self.chunk_tokens, self.k, self.size, self.block_m],
            compile_only,
        )

    def _start_gradient_prefetch(self):
        self._gradient_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._gradient_stream):
            for tensor in (
                self.pointer_tables["raw_dy"],
                self.permutation,
                self.packed_raw_dy,
                self.num_blocks,
                self.status,
            ):
                tensor.record_stream(self._gradient_stream)
            self._prefetch_raw_dy()
        self._gradient_prefetch_pending = True

    def _publish_gradient_chunk(self, dy, scores, start):
        count = max(0, min(self.chunk_tokens, dy.shape[0] - start))
        # A complete copy overwrites every element of both payloads.
        # Short final chunks retain the existing zero-padding contract.
        if self.fp8_source_prep_reuse < 2 or count < self.chunk_tokens:
            self.raw_dy.zero_()
            self.raw_scores.zero_()
        self.local_dscores.zero_()
        self.raw_dy[:count].copy_(dy[start : start + count])
        self.raw_scores[:count].copy_(scores[start : start + count])

    def _backward_windows(self, dy, scores, w1, w2, dw1, dw2, dx, dscores, start):
        windows = range(0, self.full_row_capacity, self.rows)
        for window in reversed(windows) if self._using_forward_context else windows:
            previous_dy = self.packed_dy
            self._aggregate_window_active = self._aggregate_step_enabled and window == 0
            if self._aggregate_window_active:
                cache = self._aggregate_cache[start // self.chunk_tokens]
                self.packed_dy = cache["dy"]
                self._aggregate_current_offsets = cache["offsets"]
            try:
                self._set_window(window)
                if self._aggregate_window_active:
                    self._aggregate_current_offsets.copy_(self.offsets)
                self._context_window_start = window
                if self.fp8_raw_gradient_source_prefetch and self._fp8_raw_gradient_active == 7:
                    # Independent qDY packet storage avoids FC1 source aliasing.
                    # Length/permutation are the full window; replay uses its own mask.
                    self._start_fp8_raw_gradient_broadcast_source()
                self._chunk_forward_single(
                    w1, w2, pack_input=True, return_output=False, prefetch_dy=self.overlap_expert_dy
                )
                if (
                    self.fp8_expert_source_fp8
                    and (not self.fp8_dw_from_row_cache)
                    and (self.fp8_source_refetch_phase == "early")
                ):
                    self._start_source_bf16_refetch()
                self._expert_gradients()
                if (
                    self.fp8_expert_source_fp8
                    and (not self.fp8_dw_from_row_cache)
                    and (self.fp8_source_refetch_phase == "dw2")
                ):
                    self._start_source_bf16_refetch()
                self._wgrad(
                    self.packed_dy, self.hidden, dw2, asynchronous=self.wgrad_overlap == "both"
                )
                if not self.fp8_expert_gradient_batch:
                    self._peer_gemm(
                        self.packed_output if self.fuse_dhidden_scores else self.packed_dy,
                        self.w2_t,
                        self.hidden,
                        self.identity_rows,
                        role="dhidden",
                    )
                if self.fuse_dhidden_scores:
                    self._reduce_score_partials()
                if (
                    self.fp8_expert_source_fp8
                    and (not self.fp8_dw_from_row_cache)
                    and (self.fp8_source_refetch_phase == "serial")
                ):
                    self._stage_input(force_bf16=True)
                if not self.fuse_activation_backward:
                    self._activation(backward=True)
                self._join_fp8_bf_input()
                pipeline_dw1 = False and (not self._defer_last_dw1_enabled)
                if pipeline_dw1:
                    self._prepare_fp8_dw_async(self.gate_up, self.packed_x)
                elif (
                    self._defer_last_dw1_enabled
                    and self._using_forward_context
                    and (start + self.chunk_tokens >= self.max_tokens)
                    and (window == 0)
                ):
                    # dX only reads these operands. The final window/chunk has
                    # no following metadata overwrite before the caller joins.
                    self._pending_last_dw1 = (self.gate_up, self.packed_x, dw1, self.lengths)
                else:
                    self._wgrad(
                        self.gate_up, self.packed_x, dw1, asynchronous=self.wgrad_overlap != "none"
                    )
                self._peer_gemm(
                    self.gate_up,
                    self.w1_t,
                    self.packed_output,
                    self.identity_rows,
                    write=self.fuse_return,
                    destination="local_dx",
                )
                if not self.fuse_return:
                    self._scatter_peer("local_dx")
                if pipeline_dw1:
                    # dX reads the independent row-FP8 input, while the
                    # auxiliary stream prepares both column-FP8 DW inputs.
                    # Join before reusing any window or activation storage.
                    self._wgrad(self.gate_up, self.packed_x, dw1, asynchronous=False)
                # Both wgrads read window metadata/activations. Join before
                # any next-window overwrite and before entering rank barriers.
                self._join_wgrad()
            finally:
                self.packed_dy = previous_dy
                self._aggregate_window_active = False
        self._barrier(start * 4 + 3)
        self._combine_local_routes(self.local_dx, scores, dx, start, False)
        count = max(0, min(self.chunk_tokens, dy.shape[0] - start))
        self._join_raw_gradient_score()
        dscores[start : start + count].copy_(self.local_dscores[:count])

    def _wgrad(self, left, right, output, *, asynchronous=False):
        if self.fp8_wgrad:
            if asynchronous:
                raise RuntimeError("FP8 DW must finish before workspace reuse")
            self._fp8_wgrad(
                left, right, output, self._wgrad_first_flags if self._fresh_first_dw_step else None
            )
            return
        if self._fresh_first_dw_step and (not self._aggregate_window_active):
            if asynchronous:
                raise RuntimeError("first nonempty DW must execute serially")
            dg.k_grouped_bf16_gemm_tn_contiguous(
                left,
                right,
                output,
                None,
                self.lengths,
                c=output,
                max_group_k=self.rows,
                first_flags=self._wgrad_first_flags,
            )
            return
        if self._aggregate_window_active:
            if asynchronous:
                raise RuntimeError("aggregate DW cannot defer an asynchronous output")
            self._aggregate_dw_lists[output.data_ptr()].append(
                (left, right, self._aggregate_current_offsets)
            )
            return
        super()._wgrad(left, right, output, asynchronous=asynchronous)

    @torch.no_grad()
    def forward(self, x, ids, scores, w1, w2, valid_tokens, *, _chunk_output_callback=None):
        if self._pending_last_dw1 is not None:
            raise RuntimeError("complete deferred DW1 before reusing this workspace")
        with self._execution():
            self._validate(x, ids, scores, w1, w2, valid_tokens)
            self._fresh_first_dw_step = False
            self._begin_forward_context()
            self.prepare(w1, w2)
            if self.reuse_forward_context:
                self.forward_context_reused_windows.zero_()
                self.forward_context_reused_rows.zero_()
            output = self._forward_local(
                x, ids, scores, w1, w2, valid_tokens, _chunk_output_callback=_chunk_output_callback
            )
            self._join_fp8_weight_prepare()
            if self.reuse_forward_context:
                self._record_forward_context_generation()
                self._forward_context_fingerprint = self._context_fingerprint(
                    x, ids, scores, w1, w2, valid_tokens
                )
                self._forward_context_available = True
            return output

    @torch.no_grad()
    def backward(
        self,
        dy,
        x,
        ids,
        scores,
        w1,
        w2,
        valid_tokens,
        accumulators=None,
        *,
        _defer_last_dw1=False,
        _chunk_output_callback=None,
    ):
        if self._pending_last_dw1 is not None:
            raise RuntimeError("complete deferred DW1 before reusing this workspace")
        if type(_defer_last_dw1) is not bool:
            raise TypeError("_defer_last_dw1 must be bool")
        if self.wgrad_aggregate_windows and (_defer_last_dw1 or _chunk_output_callback is not None):
            raise ValueError("aggregate DW must finish before deferred-DW or chunk callbacks")
        if _defer_last_dw1 and self.wgrad_first_nonempty:
            raise ValueError("complete first DW overwrite before any deferred DW schedule")
        if _chunk_output_callback is not None:
            if not callable(_chunk_output_callback):
                raise TypeError("chunk callback must be callable")
            if self.ring_rows is None or not self.expert_local_grad or self.wgrad_overlap != "none":
                raise ValueError("chunk callbacks require bounded serial source gradients")
        if _defer_last_dw1 and (
            self.compute != "bf16"
            or self.ring_rows is None
            or (not self.expert_local_grad)
            or (not self.reuse_forward_context)
            or (self.wgrad_overlap != "none")
        ):
            raise ValueError("deferred DW1 requires serial routed BF16 live-context gradients")
        self._defer_last_dw1_enabled = _defer_last_dw1
        self._validate(x, ids, scores, w1, w2, valid_tokens)
        _validate_accumulators(accumulators, w1, w2)
        if (
            dy.shape != x.shape
            or dy.device != x.device
            or dy.dtype != x.dtype
            or (not dy.is_contiguous())
        ):
            raise ValueError("dy must be contiguous and match x")
        with self._execution():
            self._using_forward_context = self._matching_forward_context(
                (x, ids, scores, w1, w2, valid_tokens)
            )
            self._fp8_raw_gradient_active = (
                self.fp8_raw_gradient_source
                if self._using_forward_context and len(self.forward_context_slots) == 1
                else 0
            )
            self.fp8_raw_gradient_last_active = self._fp8_raw_gradient_active
            self._consume_forward_context_generation()
            self.prepare(w1, w2)
            if self.check_errors and (not torch.cuda.is_current_stream_capturing()):
                self.reset_errors()
            self._prepare_weights(w1, w2, backward=True)
            dx, dscores = (torch.empty_like(x), torch.empty_like(scores))
            # Fresh gradients may overwrite on their first routed window.
            # External accumulator buffers preserve their existing beta=1 contract.
            fresh_overwrite = False and accumulators is None and (self.max_tokens > 0)
            self._fresh_first_dw_step = self.wgrad_first_nonempty and accumulators is None
            if self._fresh_first_dw_step:
                self._wgrad_seen.zero_()
            factory = (
                torch.empty_like
                if fresh_overwrite or self._fresh_first_dw_step
                else torch.zeros_like
            )
            dw1, dw2 = (
                (factory(w1, dtype=torch.float32), factory(w2, dtype=torch.float32))
                if accumulators is None
                else accumulators
            )
            self._first_dw_outputs = {dw1.data_ptr(), dw2.data_ptr()} if fresh_overwrite else set()
            self._aggregate_step_enabled = (
                not self.fp8_wgrad
                and self.wgrad_aggregate_windows
                and self._using_forward_context
                and (len(self.forward_context_slots) > 1 or False)
            )
            self._aggregate_window_active = False
            self._aggregate_dw_lists = {dw1.data_ptr(): [], dw2.data_ptr(): []}
            self.wgrad_aggregate_last_segments = 0
            for start in range(0, self.max_tokens, self.chunk_tokens):
                if self.expert_local_grad:
                    self._publish_gradient_chunk(dy, scores, start)
                if self._using_forward_context:
                    self._activate_forward_context(start)
                    # The matching-context and workspace-generation checks
                    # already ran. With exactly one source chunk, these local
                    # buffers still hold that forward's input and route IDs.
                    # Every rank continues all current statistics/barriers.
                    if not (self.fp8_source_prep_reuse and len(self.forward_context_slots) == 1):
                        _launch(
                            13,
                            [x, ids, self.local_x, self.local_ids, valid_tokens, self.status],
                            [x.shape[0], self.chunk_tokens, self.h, self.k, start],
                        )
                    # With one live forward chunk, its symmetric FP8 payload
                    # has not been overwritten. The normal matching context
                    # generation/fingerprint checks above remain mandatory.
                    reuse_source_quant = (
                        self.fp8_expert_source_fp8
                        and False
                        and (len(self.forward_context_slots) == 1)
                    )
                    reuse_shared_source = (
                        self.fp8_source_prep_reuse == 3
                        and self.fp8_source_shared_base
                        and (len(self.forward_context_slots) == 1)
                    )
                    if self._fp8_raw_gradient_active >= 3 and self.fp8_raw_gradient_score_overlap:
                        self._start_raw_gradient_score(valid_tokens)
                    if self.fp8_source_shared_base:
                        if not reuse_shared_source:
                            self._source_shared_local_stats()
                    elif self.fp8_expert_source_fp8 and (not reuse_source_quant):
                        _launch(
                            9,
                            [self.local_x, self.local_qx, self.local_qx_scales],
                            [self.chunk_tokens, self.h],
                        )
                    if not self.skip_masked_route_zero:
                        self.local_dx.zero_()
                    if self._fp8_raw_gradient_active >= 3:
                        self._fp8_raw_gradient_stats()
                    elif self._fp8_raw_gradient_active:
                        self._fp8_raw_gradient_publish(valid_tokens)
                    token_quant_active = (
                        self.fp8_raw_gradient_quant_pipeline and self._fp8_raw_gradient_active == 7
                    )
                    if token_quant_active:
                        self.local_dy_token_ready.zero_()
                    self._barrier(start * 4)
                    if self.fp8_source_shared_base:
                        if not reuse_shared_source:
                            self._source_shared_quantize()
                        if self._fp8_raw_gradient_active >= 3:
                            self._fp8_raw_gradient_publish(valid_tokens)
                        # The current-token acquire/release protocol replaces only
                        # the second BWD barrier, with forward Xcodes already valid.
                        if not token_quant_active:
                            self._barrier(start * 4 + 2)
                else:
                    self._route_chunk(
                        x, ids, valid_tokens, start, zero_values=not self.expert_local_grad
                    )
                if self.ring_rows is not None:
                    self._backward_windows(dy, scores, w1, w2, dw1, dw2, dx, dscores, start)
                    if _chunk_output_callback is not None:
                        _chunk_output_callback(
                            dx, dscores, start, max(0, min(self.chunk_tokens, x.shape[0] - start))
                        )
                    continue
                self._chunk_forward(
                    w1, w2, pack_input=True, return_output=not self.expert_local_grad
                )
                dims = [self.chunk_tokens, self.h, self.k, x.shape[0], start]
                if self.expert_local_grad:
                    self._expert_gradients()
                elif self.fuse_local_grad:
                    self._local_gradients(dy, scores, dscores, start)
                else:
                    _launch(12, [self.local_values, dy, dscores], dims)
                    _launch(11, [dy, scores, self.local_dy], dims)
                if not self.expert_local_grad:
                    self._barrier(start * 4 + 2)
                    self._pack_dy()
                dg.k_grouped_bf16_gemm_tn_contiguous(
                    self.packed_dy,
                    self.hidden,
                    dw2,
                    None,
                    self.lengths,
                    c=dw2,
                    max_group_k=self.rows,
                )
                # DW2 requires packed_dy above. Reuse that mandatory staging
                # result instead of fetching the same peer gradients again.
                read_dy = self.fuse_dispatch and (not self.reuse_packed_dy)
                self._peer_gemm(
                    self.packed_dy,
                    self.w2_t,
                    self.hidden,
                    self.permutation if read_dy else self.identity_rows,
                    read=read_dy,
                    source="local_dy",
                    role="dhidden",
                )
                if not self.fuse_activation_backward:
                    self._activation(backward=True)
                self._join_fp8_bf_input()
                dg.k_grouped_bf16_gemm_tn_contiguous(
                    self.gate_up,
                    self.packed_x,
                    dw1,
                    None,
                    self.lengths,
                    c=dw1,
                    max_group_k=self.rows,
                )
                self._peer_gemm(
                    self.gate_up,
                    self.w1_t,
                    self.packed_output,
                    self.identity_rows,
                    write=self.fuse_return,
                    destination="local_dx",
                )
                if not self.fuse_return:
                    self._scatter_peer("local_dx")
                self._barrier(start * 4 + 3)
                self._combine_local_routes(self.local_dx, scores, dx, start, False)
                if self.expert_local_grad:
                    count = max(0, min(self.chunk_tokens, x.shape[0] - start))
                    dscores[start : start + count].copy_(self.local_dscores[:count])
            if self.check_errors and (not torch.cuda.is_current_stream_capturing()):
                self.check()
            if self._aggregate_step_enabled:
                first = self._aggregate_dw_lists[dw1.data_ptr()]
                second = self._aggregate_dw_lists[dw2.data_ptr()]
                if len(first) != len(second):
                    raise RuntimeError("DW1/DW2 aggregate segment counts disagree")
                self.wgrad_aggregate_last_segments = len(first)
                if self._fresh_first_dw_step and first:
                    _mega_moe(
                        44,
                        [
                            self._wgrad_seen,
                            self._aggregate_first_flags,
                            *[part[2] for part in first],
                        ],
                        [self.e, len(first)],
                    )
                for output, segments in [(dw2, second), (dw1, first)]:
                    if segments:
                        left, right, offsets = map(list, zip(*segments))
                        for start in range(0, len(left), 2):
                            dg.k_grouped_bf16_gemm_tn_segmented(
                                left[start : start + 2],
                                right[start : start + 2],
                                output,
                                offsets[start : start + 2],
                                self.rows,
                                accumulate=True,
                                max_continuous_k=10240,
                                first_flags=(
                                    self._aggregate_first_flags[start // 2]
                                    if self._fresh_first_dw_step
                                    else None
                                ),
                            )
            self._aggregate_step_enabled = False
            self._aggregate_dw_lists = {}
            if self._fresh_first_dw_step:
                for output in [dw1, dw2]:
                    _mega_moe(43, [output, self._wgrad_seen], [self.e, output[0].numel()])
            self._fresh_first_dw_step = False
            self._join_raw_gradient_score()
            self._join_raw_gradient_quant()
            self._fp8_raw_gradient_active = 0
            self._forward_context_available = self._using_forward_context = False
            if self._first_dw_outputs:
                raise RuntimeError("fresh DW was never initialized by its owning window")
            return (dx, dw1, dw2, dscores)
