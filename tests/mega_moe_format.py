"""MegaMoE workload metadata shared by the UT replay entry and its host tests."""

from dataclasses import dataclass, fields
import math
import os
from types import SimpleNamespace


@dataclass(frozen=True)
class MegaMoEFormat:
    ep: int
    tokens: int
    capacity: int
    hidden: int
    intermediate_hidden: int
    experts: int
    topk: int
    rank: int = 0
    activation: str = "swiglu"
    activation_clamp: float | None = None
    fast_math: int = 1
    master_experts: int | None = None
    recv_stats: int = 0
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0
    input_format: str = "mxfp4"
    routing: str = "unknown"
    seed: int = 20261016
    reference_tokens: int = 8
    mode: str = "eager"
    warmup: int = 10
    iterations: int = 100
    repeats: int = 5

    @classmethod
    def from_params(cls, params):
        names = {field.name for field in fields(cls)}
        # The common parser inserts these GEMM defaults; they are not MoE inputs.
        common = {"gemm_type", "data_type", "quant_type", "distribution", "enable_sbo_overlap"}
        unknown = params.keys() - names - common
        if unknown:
            raise ValueError(f"Unknown MegaMoe fields: {', '.join(sorted(unknown))}")
        if params.get("quant_type", "group") != "group":
            raise ValueError("MegaMoe requires MXFP4 group quantization")
        values = {key: value for key, value in params.items() if key in names}
        if values.get("activation_clamp") == "none":
            values["activation_clamp"] = None
        required = ("ep", "tokens", "capacity", "hidden", "intermediate_hidden", "experts", "topk")
        missing = [key for key in required if key not in values]
        if missing:
            raise ValueError(f"Missing MegaMoe fields: {', '.join(missing)}")
        result = cls(**values)
        result.validate()
        return result

    def validate(self):
        for name in ("ep", "tokens", "capacity", "hidden", "intermediate_hidden", "experts",
                     "topk", "rank", "fast_math", "recv_stats", "seed", "reference_tokens",
                     "warmup", "iterations", "repeats"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"MegaMoe {name} must be an integer")
        if not 1 <= self.ep <= 32 or self.experts <= 0 or self.experts % self.ep:
            raise ValueError("MegaMoe requires 1..32 EP ranks and experts divisible by EP")
        if self.experts // self.ep > 1024:
            raise ValueError("MegaMoe supports at most 1024 local physical experts per rank")
        if not 0 <= self.rank < self.ep:
            raise ValueError("MegaMoe rank must be within EP")
        if not 1 <= self.capacity < (1 << 24) or not 0 <= self.tokens <= self.capacity:
            raise ValueError("MegaMoe requires 0 <= tokens <= positive capacity < 2^24")
        if not 0 < self.hidden < (1 << 31) or self.hidden % 512:
            raise ValueError("MegaMoe hidden must be a positive multiple of 512 within the signed 32-bit range")
        if not 0 < self.intermediate_hidden < (1 << 30) or self.intermediate_hidden % 128:
            raise ValueError("MegaMoe intermediate_hidden must be activated I, a positive multiple of 128, "
                             "with 2I within the signed 32-bit range")
        if not 1 <= self.topk <= min(16, self.experts):
            raise ValueError("MegaMoe topk must be within 1..min(16, experts)")
        if self.activation not in ("swiglu", "step", "situ"):
            raise ValueError("Unsupported MegaMoe activation")
        if self.activation_clamp is not None:
            if (not isinstance(self.activation_clamp, (int, float))
                    or not math.isfinite(self.activation_clamp) or self.activation_clamp < 0
                    or self.activation == "situ"):
                raise ValueError("MegaMoe activation_clamp requires a finite nonnegative SwiGLU/Step limit")
        for name in ("situ_beta", "situ_linear_beta"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"MegaMoe {name} must be finite and positive")
        if self.activation != "situ" and (self.situ_beta != 4 or self.situ_linear_beta != 25):
            raise ValueError("MegaMoe SiTU coefficients require activation:situ")
        if self.master_experts is not None and (type(self.master_experts) is not int
                or not 1 <= self.master_experts <= self.experts // self.ep):
            raise ValueError("MegaMoe master_experts must be within the local physical expert count")
        if self.fast_math not in (0, 1) or self.recv_stats not in (0, 1):
            raise ValueError("MegaMoe fast_math and recv_stats must be 0 or 1")
        if self.input_format != "mxfp4":
            raise ValueError("MegaMoe input_format describes the library API input and must be mxfp4")
        if self.routing not in ("unknown", "random", "round_robin", "skewed"):
            raise ValueError("Unsupported MegaMoe routing generator")
        if not 0 <= self.seed < (1 << 63) - self.ep or self.reference_tokens < 0:
            raise ValueError("Invalid MegaMoe seed or reference_tokens")
        if self.mode not in ("eager", "graph") or self.warmup < 0 or self.iterations < 1 or self.repeats < 1:
            raise ValueError("Invalid MegaMoe benchmark controls")


def run_mega_moe_format(params):
    # Import device dependencies only when this UT is selected.
    import torch
    import torch.distributed as dist
    from utils import get_acc_check, get_benchmark, get_ref_backend
    from test_mega_moe import MegaMoECase, _process_group, _run_mega_moe

    if params.get("data_type") != torch.uint8:
        raise ValueError("MegaMoe supports data_type:fp4 (W4A4 MXFP4)")
    spec = MegaMoEFormat.from_params(params)
    if get_acc_check() and get_ref_backend() != "device":
        raise ValueError("MegaMoe requires ref_backend=device")
    world = dist.get_world_size() if dist.is_initialized() else int(os.getenv("WORLD_SIZE", "1"))
    if world != spec.ep:
        raise ValueError(f"MegaMoe ep:{spec.ep} requires {spec.ep} launched ranks; got {world}")
    routing = "random" if spec.routing == "unknown" else spec.routing
    case = MegaMoECase("MegaMoe", spec.hidden, spec.intermediate_hidden, spec.experts,
                       spec.topk, spec.capacity, tokens=(spec.tokens,), activation=spec.activation,
                       clamp=spec.activation_clamp, worlds=(spec.ep,), routing=routing,
                       situ_beta=spec.situ_beta, situ_linear_beta=spec.situ_linear_beta)
    masters = spec.master_experts if spec.master_experts is not None else spec.experts // spec.ep
    replicas = masters < spec.experts // spec.ep
    with _process_group() as group:
        if dist.get_rank() == 0 and spec.routing == "unknown":
            print("MegaMoe shape replay: using seeded random routes; the HGTX tag does not dump original routes.", flush=True)
        if get_benchmark() and not os.getenv("HGGC_WARM_UP"):
            from bench_mega_moe import _benchmark_case
            args = SimpleNamespace(seed=spec.seed, replicas=replicas, master_experts=masters,
                recv_stats=bool(spec.recv_stats),
                edge_cases=False, fast_math=spec.fast_math, no_check=not get_acc_check(),
                reference_tokens=spec.reference_tokens, baseline="none", mode=spec.mode,
                warmup=spec.warmup, iterations=spec.iterations, repeats=spec.repeats)
            _benchmark_case(case, group, args, None, None)
        else:
            _run_mega_moe(case, group, check=bool(get_acc_check()),
                reference_tokens=spec.reference_tokens, fast_math=bool(spec.fast_math),
                replicas=replicas, master_experts=masters, recv_stats=bool(spec.recv_stats), seed=spec.seed)


@dataclass(frozen=True)
class MegaMoEPreDispatchFormat:
    tokens: int
    capacity: int
    hidden: int
    topk: int
    seed: int = 20261016
    reference_tokens: int = 8
    mode: str = "eager"
    warmup: int = 10
    iterations: int = 100
    repeats: int = 5

    @classmethod
    def from_params(cls, params):
        names = {field.name for field in fields(cls)}
        common = {"gemm_type", "data_type", "quant_type", "distribution", "enable_sbo_overlap"}
        unknown = params.keys() - names - common
        if unknown:
            raise ValueError(f"Unknown MegaMoePreDispatch fields: {', '.join(sorted(unknown))}")
        if params.get("quant_type", "non_quantized") != "non_quantized":
            raise ValueError("MegaMoePreDispatch requires BF16 input")
        required = ("tokens", "capacity", "hidden", "topk")
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(f"Missing MegaMoePreDispatch fields: {', '.join(missing)}")
        result = cls(**{key: value for key, value in params.items() if key in names})
        for name in names - {"mode"}:
            if type(getattr(result, name)) is not int:
                raise ValueError(f"MegaMoePreDispatch {name} must be an integer")
        if not 1 <= result.capacity < (1 << 24) or not 0 <= result.tokens <= result.capacity:
            raise ValueError("MegaMoePreDispatch requires 0 <= tokens <= positive capacity < 2^24")
        if not 0 < result.hidden < (1 << 31) or result.hidden % 512:
            raise ValueError("MegaMoePreDispatch hidden must be a positive multiple of 512 "
                             "within the signed 32-bit range")
        if not 1 <= result.topk <= 16:
            raise ValueError("MegaMoePreDispatch topk must be within 1..16")
        if not 0 <= result.seed < (1 << 63) - 32 or result.reference_tokens < 0:
            raise ValueError("Invalid MegaMoePreDispatch seed or reference_tokens")
        if (result.mode not in ("eager", "graph") or result.warmup < 0
                or result.iterations < 1 or result.repeats < 1):
            raise ValueError("Invalid MegaMoePreDispatch benchmark controls")
        return result


def run_mega_moe_pre_dispatch_format(params):
    import torch
    import torch.distributed as dist
    import deep_gemm
    from deep_gemm.testing import calc_diff
    from utils import get_acc_check, get_benchmark, get_ref_backend
    from test_mega_moe import MAX_DIFF, _process_group, _quantize_mxfp4, _dequantize_mxfp4

    if params.get("data_type") != torch.bfloat16:
        raise ValueError("MegaMoePreDispatch requires data_type:bf16")
    spec = MegaMoEPreDispatchFormat.from_params(params)
    check = bool(get_acc_check())
    if check and get_ref_backend() != "device":
        raise ValueError("MegaMoePreDispatch requires ref_backend=device")
    with _process_group():
        rng = torch.Generator(device="cuda").manual_seed(spec.seed + dist.get_rank())
        source = torch.randn(spec.tokens, spec.hidden, device="cuda", dtype=torch.bfloat16, generator=rng)
        ids = torch.randint(-1, 256, (spec.tokens, spec.topk), device="cuda", generator=rng)
        weights = torch.rand(spec.tokens, spec.topk, device="cuda", generator=rng)
        x = torch.empty(spec.capacity, spec.hidden // 2, device="cuda", dtype=torch.uint8)
        sf = torch.empty(spec.capacity, spec.hidden // 32, device="cuda", dtype=torch.uint8)
        staged_ids = torch.empty(spec.capacity, spec.topk, device="cuda", dtype=torch.int64)
        staged_weights = torch.empty(spec.capacity, spec.topk, device="cuda", dtype=torch.float32)
        if spec.tokens:
            source[0].zero_()
            ids[0, 0], weights[0, 0] = -1, 0

        def run():
            # The bound pre-dispatch entry accepts the four destination tensors
            # directly; no expert weights or communication workspace are needed.
            deep_gemm._C.mega_moe_pre_dispatch(
                source, ids, weights, x, sf, staged_ids, staged_weights, spec.tokens)
            return x[:spec.tokens]

        result = run()
        report = dict(operation="MegaMoePreDispatch", tokens=spec.tokens,
                      capacity=spec.capacity, hidden=spec.hidden, topk=spec.topk)
        if check:
            checked = min(spec.tokens, spec.reference_tokens) if spec.reference_tokens else spec.tokens
            ref_x, ref_sf = _quantize_mxfp4(source[:checked])
            diff = torch.tensor(float(calc_diff(_dequantize_mxfp4(x[:checked], sf[:checked]),
                                               _dequantize_mxfp4(ref_x, ref_sf))), device="cuda")
            dist.all_reduce(diff, op=dist.ReduceOp.MAX)
            assert diff.item() < MAX_DIFF, ("pre-dispatch quantization diff", diff.item())
            torch.testing.assert_close(sf[:checked], ref_sf, rtol=0, atol=0)
            torch.testing.assert_close(staged_ids[:spec.tokens], ids, rtol=0, atol=0)
            torch.testing.assert_close(staged_weights[:spec.tokens],
                                     torch.where(ids >= 0, weights, 0), rtol=0, atol=0)
            report["correctness"] = dict(status="passed", checked_tokens=checked, max_diff=diff.item())
        if get_benchmark() and not os.getenv("HGGC_WARM_UP"):
            from bench_mega_moe import _benchmark_pipeline
            report["benchmark"] = _benchmark_pipeline(run, warmup=spec.warmup,
                iterations=spec.iterations, repeats=spec.repeats, mode=spec.mode,
                expected=result.clone() if check else None)
            report["benchmark"]["scope"] = "bf16_input_to_staged_mxfp4"
        if dist.get_rank() == 0:
            import json
            print(json.dumps(report), flush=True)
