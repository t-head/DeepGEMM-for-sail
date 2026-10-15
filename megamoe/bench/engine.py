"""Benchmark execution engine for PPU890 Megamoe fused kernel.

Holds the _Benchmark class -- all state management, kernel launch, baseline
runners, correctness checks, and performance reports for one benchmark
iteration.
"""

import math
import os
import random
import threading
import time
from typing import Dict, Tuple

import torch
import torch.distributed as dist

from .ref.dist import dist_print, uneven_all_gather
from .ref.generator import build_reference, build_reference_inputs, load_cached_reference
from .ref.quant import (
    normalize_mxfp4_dispatch_scale,
    swiglu_flat_to_mxfp4,
    swiglu_masked_to_mxfp4,
)
from .report import (
    HT_BASELINE_STAGE_NAMES,
    LL_BASELINE_STAGE_NAMES,
    dist_print_time_average,
    estimate_pipeline_work,
    estimate_routed_icn_bytes,
    estimate_masked_issued_m,
    fmt_gemm_model_header,
    fmt_gemm_model_line,
    fmt_pipeline_header,
    fmt_pipeline_line,
    fmt_stage_timing_header,
    fmt_stage_timing_line,
    kineto_range,
    make_pipeline_row_fields,
    make_stage_summary,
    safe_div,
)
from .ref.ht_layout import CONTIGUOUS_M_ALIGNMENT, ht_permute, ht_unpermute_reduce
import megamoe

try:
    import deep_ep as _deep_ep
except Exception:
    _deep_ep = None

try:
    import deep_gemm
except Exception:
    deep_gemm = None

FUSED_MXFP4_L2_ACTIVATION_SF_GRANULARITY = 32

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

# The torch reference, the inputs it reads and its disk cache live in ref/generator.py: that module
# hashes its own import closure, so an edit on the baseline side here cannot throw the cached
# reference away, and an edit to the reference itself cannot be missed.


def _iter_jit_builders(num_ranks: int, num_experts: int, aligned_max_tokens_per_rank: int, num_topk: int, hidden: int,
                       intermediate_hidden: int, num_tokens: int, activation_clamp, declared_max_tokens_per_rank: int):
    from megamoe.engine.sm_topology import _build_dead_sm_runtime
    from .bf16_to_mxfp4 import get_bf16_to_mxfp4_runtime

    def build_fused():
        megamoe.compile_fp4_fp4_megamoe(num_ranks,
                                          num_experts,
                                          aligned_max_tokens_per_rank,
                                          num_topk,
                                          hidden,
                                          intermediate_hidden,
                                          num_tokens,
                                          activation_clamp=activation_clamp,
                                          declared_max_tokens_per_rank=declared_max_tokens_per_rank)

    return {'fused': build_fused, 'dead_sm': _build_dead_sm_runtime, 'bf16_to_mxfp4': get_bf16_to_mxfp4_runtime}


def _start_jit_prewarm(builders: dict, outcomes: dict) -> list:

    def _run_one_build(name: str, build) -> None:
        started = time.time()
        try:
            build()
            outcomes[name] = f'ok in {time.time() - started:.1f}s'
        except Exception as ex:
            outcomes[name] = f'FAILED {type(ex).__name__}: {ex}'

    threads = []
    for name, build in builders.items():
        thread = threading.Thread(target=_run_one_build, args=(name, build), name=f'jit-prewarm-{name}', daemon=True)
        thread.start()
        threads.append(thread)
    return threads


# ---------------------------------------------------------------------------
# _Benchmark class
# ---------------------------------------------------------------------------


class _Benchmark:
    """Hold all state for one benchmark iteration (single token count)."""

    def __init__(self, group, rank_idx: int, num_ranks: int, args):
        self.group = group
        self.rank_idx = rank_idx
        self.num_ranks = num_ranks
        self.args = args

        self.declared_max_tokens_per_rank = args.num_max_tokens_per_rank
        self.hidden = args.hidden
        self.intermediate_hidden = args.intermediate_hidden
        self.num_experts = args.num_experts
        self.num_topk = args.num_topk
        self.num_experts_per_rank = args.num_experts // num_ranks
        self.activation_clamp = args.activation_clamp if math.isfinite(args.activation_clamp) else None

        self.num_tokens = (max(0, args.num_max_tokens_per_rank -
                               random.randint(0, args.num_max_removed_tokens)) if args.num_tokens == 0 else args.num_tokens)

        # Filled by build_inputs
        self.x_bf16 = None
        self.x_fp4 = None
        self.topk_idx = None
        self.topk_weights = None
        self.l1_weights = None
        self.l2_weights = None
        self.transformed_l1 = None
        self.transformed_l2 = None
        self.sym_buffer = None
        self.y_fused = None
        self.cumulative_recv_counts = None

        # Filled by prewarm_jit
        self.prewarm_threads = None
        self.prewarm_outcomes = {}

        # Baseline state; built lazily
        self._ht_buffer = None
        self._ll_buffer = None
        self._grouped_l1_weights = None
        self._grouped_l2_weights = None
        self._ll_l1_y = None
        self._ll_l2_y = None
        self._ll_combined = None
        self._ll_topk_idx = None

        # HT stats
        self._ht_useful_m = 0
        self._ht_issued_m = 0
        self._ht_active_experts = 0
        # LL stats
        self._ll_masked_m = None
        self._ll_l1_block_m = 0
        self._ll_l2_block_m = 0

        # Diff results (lives through report cycle)
        self.global_calc_diff = None
        # The memoized torch reference (see get_torch_reference)
        self._y_ref = None

    # --- input construction ---

    def _reference_kwargs(self) -> dict:
        """The reference's shape identity, shared by its input builder and its forward."""
        return dict(num_ranks=self.num_ranks,
                    num_experts=self.num_experts,
                    num_topk=self.num_topk,
                    hidden=self.hidden,
                    intermediate_hidden=self.intermediate_hidden)

    def build_inputs(self) -> torch.Tensor:
        assert self.num_tokens <= self.declared_max_tokens_per_rank

        self._ref_inputs = build_reference_inputs(rank_idx=self.rank_idx,
                                                  num_tokens=self.num_tokens,
                                                  masked_ratio=self.args.masked_ratio,
                                                  **self._reference_kwargs())
        self.x_bf16 = self._ref_inputs.x_bf16
        self.x_fp4 = self._ref_inputs.x_fp4
        self.topk_idx = self._ref_inputs.topk_idx
        self.topk_weights = self._ref_inputs.topk_weights
        self.l1_weights = self._ref_inputs.l1_weights
        self.l2_weights = self._ref_inputs.l2_weights

        self.transformed_l1, self.transformed_l2 = megamoe.transform_weights_for_megamoe(self.l1_weights, self.l2_weights)

        # The grouped quantizer's temporaries stay in the caching allocator's reserved pool (the
        # allocator keeps the freed segments); the symmetric workspace below is a driver-level
        # allocation that cannot reuse them, so hand the cached segments back before mapping the
        # workspace.
        torch.cuda.empty_cache()
        self.sym_buffer = megamoe.get_symm_buffer_for_megamoe(self.group, self.num_experts, self.declared_max_tokens_per_rank,
                                                                self.num_topk, self.hidden, self.intermediate_hidden)

        self.y_fused = torch.empty((self.num_tokens, self.hidden), dtype=torch.bfloat16, device="cuda")
        self.cumulative_recv_counts = torch.zeros((self.num_experts_per_rank, ), dtype=torch.int, device="cuda")

        # Grouped weight views for baselines (shared by HT and LL)
        self._grouped_l1_weights = (
            self.l1_weights[0].view(torch.uint8),
            self.l1_weights[1].permute(0, 2, 1),
        )
        self._grouped_l2_weights = (
            self.l2_weights[0].view(torch.uint8),
            self.l2_weights[1].permute(0, 2, 1),
        )
        return self.y_fused

    # --- JIT prewarm ---

    def prewarm_jit(self):
        self.prewarm_outcomes = {}
        self.prewarm_threads = _start_jit_prewarm(
            _iter_jit_builders(self.num_ranks, self.num_experts, self.sym_buffer.aligned_max_tokens_per_rank, self.num_topk, self.hidden,
                               self.intermediate_hidden, self.num_tokens, self.activation_clamp, self.declared_max_tokens_per_rank),
            self.prewarm_outcomes)

    def join_prewarm(self):
        if self.prewarm_threads:
            for thread in self.prewarm_threads:
                thread.join()
            dist_print('[jit-prewarm] ' + ', '.join(f'{name}={r}' for name, r in self.prewarm_outcomes.items()), once_in_node=True)

    # --- fused kernel ---

    def run_fused(self) -> torch.Tensor:
        self.sym_buffer.topk_idx[:self.num_tokens].copy_(self.topk_idx)
        self.sym_buffer.topk_weights[:self.num_tokens].copy_(self.topk_weights)
        megamoe.launch_fp4_fp4_megamoe(
            self.y_fused,
            self.transformed_l1,
            self.transformed_l2,
            self.sym_buffer,
            cumulative_local_expert_recv_counts=self.cumulative_recv_counts,
            recipe=(1, 1, 32),
            activation="swiglu",
            activation_clamp=self.activation_clamp,
            x_bf16=self.x_bf16,
        )
        return self.y_fused

    # --- high-throughput baseline ---

    def build_ht_baseline(self):
        # Both imports are optional and fail independently: the transport is deep_ep, the grouped GEMMs
        # are deep_gemm, so a baseline is only built when both are importable.
        if _deep_ep is None or deep_gemm is None:
            return
        # Read by the PPU DeepEP build at Buffer construction: fabric mode needs the value to
        # equal the world size (multi-node only). setdefault keeps an explicit override possible.
        os.environ.setdefault('DEEPEP_NUM_MAX_ICN_PEERS', str(self.num_ranks))
        self._ht_buffer = _deep_ep.Buffer(
            self.group,
            num_nvl_bytes=1024 * 1024 * 1024,
            num_rdma_bytes=0,
            low_latency_mode=False,
            num_qps_per_rank=1,
            explicitly_destroy=True,
            allow_mnnvl=True,
        )

    @property
    def ht_buffer(self):
        return self._ht_buffer

    def run_ht_baseline(self, profile_stages: bool = False) -> torch.Tensor:
        assert self._ht_buffer is not None
        with kineto_range(profile_stages, "ht_baseline.dispatch"):
            num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, \
                is_token_in_rank, _layout_event = self._ht_buffer.get_dispatch_layout(
                    self.topk_idx, self.num_experts, previous_event=None,
                    async_finish=False, allocate_on_comm_stream=False)
            recv_x, recv_topk_idx, recv_topk_weights, \
                num_recv_tokens_per_expert_list, handle, _dispatch_event = \
                self._ht_buffer.dispatch(
                    (self.x_fp4[0], self.x_fp4[1]),
                    num_tokens_per_rank=num_tokens_per_rank,
                    num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                    is_token_in_rank=is_token_in_rank,
                    num_tokens_per_expert=num_tokens_per_expert,
                    topk_idx=self.topk_idx,
                    topk_weights=self.topk_weights,
                    expert_alignment=1, previous_event=None,
                    async_finish=True, allocate_on_comm_stream=False)
            _dispatch_event.current_stream_wait()
        recv_x_fp4, recv_x_sf = recv_x
        num_recv_tokens = recv_x_fp4.size(0)
        combine_input = torch.empty((num_recv_tokens, self.hidden), dtype=torch.bfloat16, device="cuda")

        if num_recv_tokens > 0:
            with kineto_range(profile_stages, "ht_baseline.permute"):
                a1q, a1q_scale, expert_ids, inv_perm, m_rows = ht_permute(recv_x_fp4, recv_x_sf, recv_topk_idx,
                                                                          num_recv_tokens_per_expert_list, CONTIGUOUS_M_ALIGNMENT)
            num_padded_rows = a1q.size(0)
            self._ht_useful_m = int(sum(num_recv_tokens_per_expert_list))
            self._ht_issued_m = int(num_padded_rows)
            self._ht_active_experts = int(sum(1 for c in num_recv_tokens_per_expert_list if c > 0))

            l1_out = torch.empty((num_padded_rows, self.intermediate_hidden * 2), dtype=torch.bfloat16, device="cuda")
            l2_out = torch.empty((num_padded_rows, self.hidden), dtype=torch.bfloat16, device="cuda")
            with kineto_range(profile_stages, "ht_baseline.l1_gemm"):
                deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad((a1q.view(torch.uint8), normalize_mxfp4_dispatch_scale(a1q_scale)),
                                                               self._grouped_l1_weights, None, l1_out, expert_ids, m_rows)
            with kineto_range(profile_stages, "ht_baseline.swiglu_quant"):
                a2q, a2q_scale = swiglu_flat_to_mxfp4(l1_out, activation_clamp=self.activation_clamp)
            with kineto_range(profile_stages, "ht_baseline.l2_gemm"):
                deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad((a2q, a2q_scale), self._grouped_l2_weights, None, l2_out, expert_ids, m_rows)
            with kineto_range(profile_stages, "ht_baseline.unpermute"):
                ht_unpermute_reduce(l2_out, recv_topk_idx, recv_topk_weights, inv_perm, combine_input)
        with kineto_range(profile_stages, "ht_baseline.combine"):
            out = self._ht_buffer.combine(combine_input,
                                          handle=handle,
                                          topk_weights=None,
                                          previous_event=None,
                                          async_finish=False,
                                          allocate_on_comm_stream=False)[0]
        return out

    # --- low-latency baseline ---

    def build_ll_baseline(self):
        if _deep_ep is None or deep_gemm is None:
            return
        # Read by the PPU DeepEP build at Buffer construction: with no usable remote transport
        # in the container, the multi-node low-latency bring-up exits 255 without it. setdefault
        # keeps an explicit override possible.
        os.environ.setdefault('DEEPEP_DISABLE_NETWORK', '1')
        m_max = self.declared_max_tokens_per_rank * self.num_ranks
        self._ll_l1_y = torch.empty((self.num_experts_per_rank, m_max, self.intermediate_hidden * 2), dtype=torch.bfloat16, device="cuda")
        self._ll_l2_y = torch.empty((self.num_experts_per_rank, m_max, self.hidden), dtype=torch.bfloat16, device="cuda")
        self._ll_combined = torch.empty((self.num_tokens, self.hidden), dtype=torch.bfloat16, device="cuda")
        self._ll_topk_idx = self.topk_idx.to(torch.int64)
        num_rdma_bytes = _deep_ep.Buffer.get_low_latency_rdma_size_hint(self.declared_max_tokens_per_rank, self.hidden, self.num_ranks,
                                                                        self.num_experts)
        self._ll_buffer = _deep_ep.Buffer(self.group,
                                          num_nvl_bytes=0,
                                          num_rdma_bytes=num_rdma_bytes,
                                          low_latency_mode=True,
                                          num_qps_per_rank=self.num_experts // self.num_ranks,
                                          allow_nvlink_for_low_latency_mode=True,
                                          explicitly_destroy=True,
                                          allow_mnnvl=True)

    @property
    def ll_buffer(self):
        return self._ll_buffer

    def run_ll_baseline(self, profile_stages: bool = False) -> torch.Tensor:
        assert self._ll_buffer is not None
        with kineto_range(profile_stages, "ll_baseline.dispatch"):
            recv_x, masked_m, ll_handle, _event, _hook = self._ll_buffer.low_latency_dispatch(self.x_bf16,
                                                                                              self._ll_topk_idx,
                                                                                              self.declared_max_tokens_per_rank,
                                                                                              self.num_experts,
                                                                                              use_fp8=False,
                                                                                              round_scale=False,
                                                                                              use_ue8m0=False,
                                                                                              use_mxfp4=True,
                                                                                              quant_size=32,
                                                                                              mxfp4_scale_row_major=False,
                                                                                              async_finish=False,
                                                                                              return_recv_hook=False)
        self._ll_masked_m = masked_m
        block_m = max(1, (self.declared_max_tokens_per_rank * self.num_ranks * self.num_topk + self.num_experts - 1) // self.num_experts)
        with kineto_range(profile_stages, "ll_baseline.l1_gemm"):
            l1_gemm_cfg = deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked((recv_x[0], recv_x[1]), self._grouped_l1_weights, None,
                                                                          self._ll_l1_y, masked_m, block_m)
        if l1_gemm_cfg is not None:
            self._ll_l1_block_m = int(l1_gemm_cfg[0])
        with kineto_range(profile_stages, "ll_baseline.swiglu_quant"):
            l2_lhs = swiglu_masked_to_mxfp4(self._ll_l1_y, masked_m, activation_clamp=self.activation_clamp)
        with kineto_range(profile_stages, "ll_baseline.l2_gemm"):
            l2_gemm_cfg = deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(l2_lhs, self._grouped_l2_weights, None, self._ll_l2_y, masked_m,
                                                                          block_m)
        if l2_gemm_cfg is not None:
            self._ll_l2_block_m = int(l2_gemm_cfg[0])
        with kineto_range(profile_stages, "ll_baseline.combine"):
            combined, _event, _hook = self._ll_buffer.low_latency_combine(self._ll_l2_y,
                                                                          self._ll_topk_idx,
                                                                          self.topk_weights,
                                                                          ll_handle,
                                                                          zero_copy=False,
                                                                          async_finish=False,
                                                                          return_recv_hook=False,
                                                                          out=self._ll_combined)
        return combined

    # --- torch reference ---

    def _build_reference_output(self) -> torch.Tensor:
        return build_reference(self._ref_inputs,
                               rank_idx=self.rank_idx,
                               group=self.group,
                               activation_clamp=self.args.activation_clamp,
                               **self._reference_kwargs())

    def get_torch_reference(self) -> torch.Tensor:
        # Memoized: the reference is a pure function of the point, and both the pre-timing launch
        # check and the post-timing one ask for it, so the disk cache and the collective its load
        # performs are paid once.
        if self._y_ref is None:
            self._y_ref = load_cached_reference(
                lambda: self._build_reference_output(),
                config=(self.num_tokens, self.declared_max_tokens_per_rank, self.hidden, self.intermediate_hidden, self.num_experts,
                        self.num_topk, self.args.masked_ratio, self.args.activation_clamp, self.num_ranks),
                rank_idx=self.rank_idx,
                expected_shape=(self.num_tokens, self.hidden),
            )
        return self._y_ref

    # --- output diff ---

    def _get_output_diff_stats(self, y: torch.Tensor, y_ref: torch.Tensor) -> Tuple[torch.Tensor, float, float, float]:
        """The global calc_diff and abs/rel means, plus the per-element abs diff the report needs."""
        diff = (y.float() - y_ref.float()).abs()
        ref_abs = y_ref.float().abs()
        y_f64, y_ref_f64 = y.double(), y_ref.double()
        stats = torch.stack([
            (y_f64 * y_ref_f64).sum(),
            (y_f64 * y_f64 + y_ref_f64 * y_ref_f64).sum(),
            diff.double().sum(),
            ref_abs.double().sum(),
            torch.tensor(float(diff.numel()), dtype=torch.float64, device="cuda"),
        ])
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=self.group)
        calc_diff = (0.0 if stats[1].item() == 0 else 1.0 - 2.0 * stats[0].item() / stats[1].item())
        mean_abs = stats[2].item() / max(stats[4].item(), 1.0)
        mean_ref = stats[3].item() / max(stats[4].item(), 1.0)
        return diff, calc_diff, mean_abs, mean_abs / max(mean_ref, 1e-12)

    def compute_output_diff(self, y: torch.Tensor, y_ref: torch.Tensor, label: str = "fused vs torch reference, global") -> None:
        """Compute, print and enforce the output diff (fused vs torch reference).

        ``calc_diff`` is the point's acceptance statistic: it is exactly 0 when the fused output
        matches the reference bit for bit. A point that does not match cannot be compared against a
        baseline or against an earlier run, so a nonzero value fails the run here instead of being
        printed beside numbers nobody can trust. The raise sits after the barrier below, so every
        rank decides on the same all-reduced value and no peer is left waiting.
        """
        diff, calc_diff, global_mean_abs, global_mean_rel = self._get_output_diff_stats(y, y_ref)
        self.global_calc_diff = calc_diff

        if diff.numel() > 0:
            max_flat = int(diff.argmax().item())
            max_token = max_flat // self.hidden
            max_hidden = max_flat % self.hidden
            token_diff = diff[max_token]
            max_info = torch.cat([
                torch.tensor([
                    diff[max_token, max_hidden].item(),
                    float(self.rank_idx),
                    float(max_token),
                    float(max_hidden), y[max_token, max_hidden].float().item(), y_ref[max_token, max_hidden].float().item(),
                    token_diff.mean().item(),
                    token_diff.max().item()
                ],
                             dtype=torch.float64,
                             device="cuda"),
                self.topk_idx[max_token].to(torch.float64),
                self.topk_weights[max_token].to(torch.float64),
            ])
        else:
            max_info = torch.zeros((8 + self.num_topk * 2, ), dtype=torch.float64, device="cuda")
            max_info[0] = -1.0
            max_info[1] = float(self.rank_idx)
        gathered = [torch.empty_like(max_info) for _ in range(self.num_ranks)]
        dist.all_gather(gathered, max_info, group=self.group)
        if self.rank_idx == 0:
            infos = torch.stack(gathered)
            global_max_info = infos[int(infos[:, 0].argmax().item())].cpu()
            print(f"Output diff ({label}):", flush=True)
            print(
                f" > calc_diff={self.global_calc_diff:.6e}, "
                f"max_abs={global_max_info[0].item():.6e}, "
                f"mean_abs={global_mean_abs:.6e}, "
                f"mean_abs/mean_ref={global_mean_rel:.6e}",
                flush=True)
            for rank_info in infos.cpu():
                rank_idx = int(rank_info[1].item())
                token_idx = int(rank_info[2].item())
                hidden_idx = int(rank_info[3].item())
                topk_idx_row = [int(v) for v in rank_info[8:8 + self.num_topk].tolist()]
                topk_weights_row = [round(float(v), 6) for v in rank_info[8 + self.num_topk:8 + self.num_topk * 2].tolist()]
                print(
                    f" > [rank {rank_idx}] local_max_diff_at token={token_idx}, hidden={hidden_idx}, "
                    f"fused={rank_info[4].item():.6e}, ref={rank_info[5].item():.6e}, "
                    f"abs={rank_info[0].item():.6e}, token_mean_abs={rank_info[6].item():.6e}, "
                    f"token_max_abs={rank_info[7].item():.6e}, "
                    f"topk_idx={topk_idx_row}, topk_weights={topk_weights_row}",
                    flush=True)
        dist.barrier(group=self.group)
        if calc_diff != 0.0:
            raise RuntimeError(f"{label}: calc_diff={calc_diff:.6e} is not 0, so the fused output does not match "
                               "the torch reference and this point's numbers cannot be trusted")

    # --- baseline diff ---

    def compute_baseline_diff(self, label: str, target_y: torch.Tensor, baseline_y: torch.Tensor) -> None:
        """Report output difference between the fused kernel and a baseline."""
        diff = (target_y.float() - baseline_y.float()).abs()
        dist_print(f"Output diff (fused vs {label}):", once_in_node=True)
        if diff.numel() == 0:
            dist_print(" > (no local rows on this rank)", once_in_node=True)
            dist_print(once_in_node=True)
            return
        denom = target_y.float().abs().mean().clamp_min(1e-12)
        dist_print(
            f" > max_abs={diff.max().item():.6e}, "
            f"mean_abs={diff.mean().item():.6e}, "
            f"mean_abs/mean_ref={diff.mean().div(denom).item():.6e}",
            once_in_node=True)
        dist_print(once_in_node=True)

    # --- config display ---

    def print_config(self) -> None:
        dist_print("Config (PPU890 fused Megamoe):", once_in_node=True)
        dist_print(f" > Tokens: {self.num_tokens}/{self.declared_max_tokens_per_rank}", once_in_node=True)
        dist_print(f" > Hidden: {self.hidden}, Intermediate: {self.intermediate_hidden}", once_in_node=True)
        dist_print(f" > Experts: {self.num_topk}/{self.num_experts} "
                   f"(per-rank: {self.num_experts_per_rank})", once_in_node=True)
        dist_print(f" > Masked ratio: {self.args.masked_ratio}", once_in_node=True)
        dist_print(" > Quant: w4a4", once_in_node=True)
        dist_print(f" > Activation SF: L2 per-{FUSED_MXFP4_L2_ACTIVATION_SF_GRANULARITY} UE8M0 (fused and baseline)", once_in_node=True)
        dist_print(f" > High-throughput baseline: "
                   f"{'enabled' if self.args.run_baseline else 'disabled'}", once_in_node=True)
        dist_print(" > Torch reference: " + ("disabled" if not self.args.check_output_diff else "enabled"), once_in_node=True)
        dist_print(f" > Low-latency baseline: "
                   f"{'enabled' if self.args.run_low_latency_baseline else 'disabled'}", once_in_node=True)
        if self.args.run_low_latency_baseline:
            dist_print(" > Low-latency baseline quant path: MXFP4 dispatch + FP4 masked GEMM", once_in_node=True)
        dist_print(f" > Buffer: {self.sym_buffer.buffer.nbytes / 2**30:.3f} GiB", once_in_node=True)
        dist_print(once_in_node=True)

    # --- performance table ---

    def _compute_work_stats(self) -> tuple:
        """Return (recv_m, active_experts, tokens_per_expert_min, tokens_per_expert_max, tokens_per_expert_avg)."""
        gathered = uneven_all_gather(self.topk_idx, group=self.group)
        gathered[(gathered < self.rank_idx * self.num_experts_per_rank)
                 | (gathered >= (self.rank_idx + 1) * self.num_experts_per_rank)] = -1
        local_ids = gathered[gathered != -1]
        recv_m = int(local_ids.numel())
        active_experts = int(torch.unique(local_ids).numel())
        if recv_m:
            token_counts_per_expert = torch.bincount(local_ids - self.rank_idx * self.num_experts_per_rank,
                                                     minlength=self.num_experts_per_rank)
            active_token_counts = token_counts_per_expert[token_counts_per_expert > 0]
            tokens_per_expert_min = int(active_token_counts.min().item())
            tokens_per_expert_max = int(active_token_counts.max().item())
            tokens_per_expert_avg = float(active_token_counts.float().mean().item())
        else:
            tokens_per_expert_min = tokens_per_expert_max = tokens_per_expert_avg = 0
        return recv_m, active_experts, tokens_per_expert_min, tokens_per_expert_max, tokens_per_expert_avg

    def _make_gemm_work_kwargs(self) -> dict:
        """The GEMM work model's per-run keys, shared by the fused work and the stage reports."""
        return dict(hidden=self.hidden,
                    intermediate_hidden=self.intermediate_hidden,
                    l1_weights=self.l1_weights,
                    l2_weights=self.l2_weights)

    def _compute_work_flops(self, recv_m: int, active_experts: int) -> Tuple[float, float, float]:
        """Return (work_flops, work_hbm_bytes, icn_bytes) for the shared fused/baseline work model."""
        # The fused and baseline rows share one work model: the quantized input, the routed rows and
        # the weight bytes are identical, so it is computed once and reported for both.
        work_flops, work_hbm_bytes = estimate_pipeline_work(recv_m, active_experts, x_quantized=self.x_fp4, **self._make_gemm_work_kwargs())
        icn_bytes = estimate_routed_icn_bytes(recv_m, hidden=self.hidden, x_quantized=self.x_fp4)
        return work_flops, work_hbm_bytes, icn_bytes

    def _dist_print_by_rank(self, line: str) -> None:
        gathered = [None] * self.num_ranks
        dist.all_gather_object(gathered, line, group=self.group)
        if self.rank_idx == 0:
            for rank_line in gathered:
                print(rank_line, flush=True)
        dist.barrier(group=self.group)

    def print_perf_header(self) -> None:
        dist_print("Performance (useful/routed-token model):", once_in_node=True)
        dist_print(fmt_pipeline_header(), once_in_node=True)

    def print_fused_row(self, time_fused: float, recv_m: int, active_experts: int, fused_flops: float, fused_hbm_bytes: float,
                        icn_bytes: float, tokens_per_expert_min: int, tokens_per_expert_max: int, tokens_per_expert_avg: float) -> None:
        row_fields = make_pipeline_row_fields(rank_idx=self.rank_idx,
                                              num_ranks=self.num_ranks,
                                              recv_m=recv_m,
                                              active_experts=active_experts,
                                              tokens_per_expert_min=tokens_per_expert_min,
                                              tokens_per_expert_max=tokens_per_expert_max,
                                              tokens_per_expert_avg=tokens_per_expert_avg)
        self._dist_print_by_rank(
            fmt_pipeline_line(name="[fused]",
                              time_s=time_fused,
                              compute_tflops=safe_div(fused_flops / 1e12, time_fused),
                              hbm_gbs=safe_div(fused_hbm_bytes / 1e9, time_fused),
                              icn_gbs=safe_div(icn_bytes / 1e9, time_fused),
                              **row_fields))
        dist_print_time_average(time_fused, "fused", self.num_ranks)

    # --- stage timing reports ---

    def _dist_print_stage_report(self, summary: Dict, with_layout: bool, timing_caption: str, model_caption: str) -> None:
        """Print one baseline's stage-timing table and the GEMM model table under it."""
        dist_print(timing_caption, once_in_node=True)
        dist_print(fmt_stage_timing_header(with_layout=with_layout), once_in_node=True)
        self._dist_print_by_rank(fmt_stage_timing_line(summary, with_layout=with_layout))
        dist_print(model_caption, once_in_node=True)
        dist_print(fmt_gemm_model_header(), once_in_node=True)
        self._dist_print_by_rank(fmt_gemm_model_line(summary))

    def print_ht_stage_report(self, stage_times, time_ht: float) -> None:
        if stage_times is None:
            return
        gemm_work_kwargs = self._make_gemm_work_kwargs()
        summary = make_stage_summary(name="[ht_w4a4]",
                                     stage_times=stage_times,
                                     stage_names=HT_BASELINE_STAGE_NAMES,
                                     range_prefix="ht_baseline",
                                     event_time_s=time_ht,
                                     useful_m=self._ht_useful_m,
                                     issued_l1_m=self._ht_issued_m,
                                     issued_l2_m=self._ht_issued_m,
                                     active_experts=self._ht_active_experts,
                                     rank_idx=self.rank_idx,
                                     num_ranks=self.num_ranks,
                                     gemm_work_kwargs=gemm_work_kwargs)
        if summary is not None:
            self._dist_print_stage_report(summary, True, "High-throughput baseline stage timings (Kineto ranges; us):",
                                          "Baseline GEMM model (L1+L2 only; useful=routed rows, issued=expert-alignment padded rows):")

    def print_ll_stage_report(self, stage_times, time_ll: float) -> None:
        if stage_times is None or self._ll_masked_m is None:
            return
        gemm_work_kwargs = self._make_gemm_work_kwargs()
        summary = make_stage_summary(name="[ll_w4a4]",
                                     stage_times=stage_times,
                                     stage_names=LL_BASELINE_STAGE_NAMES,
                                     range_prefix="ll_baseline",
                                     event_time_s=time_ll,
                                     useful_m=int(self._ll_masked_m.sum().item()),
                                     issued_l1_m=estimate_masked_issued_m(self._ll_masked_m, self._ll_l1_block_m),
                                     issued_l2_m=estimate_masked_issued_m(self._ll_masked_m, self._ll_l2_block_m),
                                     active_experts=int((self._ll_masked_m > 0).sum().item()),
                                     rank_idx=self.rank_idx,
                                     num_ranks=self.num_ranks,
                                     gemm_work_kwargs=gemm_work_kwargs)
        if summary is not None:
            self._dist_print_stage_report(summary, False, "Low-latency baseline stage timings (Kineto ranges; us):",
                                          "Low-latency baseline GEMM model (L1+L2 only; "
                                          "useful=routed rows, issued=padding/tile rows):")

    # --- destroy ---

    def _destroy_baselines(self):
        if self._ht_buffer is not None:
            self._ht_buffer.destroy()
            self._ht_buffer = None
        if self._ll_buffer is not None:
            self._ll_buffer.destroy()
            self._ll_buffer = None

    def destroy(self):
        self.sym_buffer.destroy()
        self._destroy_baselines()
        dist.destroy_process_group()
