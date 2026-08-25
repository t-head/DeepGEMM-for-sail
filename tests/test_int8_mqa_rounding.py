import hashlib
import os

import pytest
import torch

import deep_gemm

# Filled from the established ZW810E implementation using the deterministic
# inputs below. These hashes deliberately cover FP32 logits and top-k indices:
# a tolerance-only check cannot detect rounding changes at routing boundaries.
EXPECTED_LOGITS_SHA256 = (
    "9b27241a742b586ecd5b7d009dd8c551fd93be7e3bf6ecc208b75144d7bc4543"
)
EXPECTED_TOPK_SHA256 = (
    "aa108e73f74553eb66ca70a23880e440879bdd9ee5fca561bd2328c6ca7c4b77"
)
USE_CPP_JIT = os.environ.get("USE_CPP_JIT_FOR_PYTHON", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().cpu().numpy().tobytes()).hexdigest()


def _is_zw810e() -> bool:
    return torch.cuda.is_available() and "zw810e" in torch.cuda.get_device_name(0).lower()


def _inputs():
    seq_len_q, seq_len_kv = 512, 2048
    num_heads, head_dim = 64, 128

    q_index = torch.arange(
        seq_len_q * num_heads * head_dim, device="cuda", dtype=torch.int64
    )
    q = ((q_index * 17 + 13) % 255 - 127).to(torch.int8)
    q = q.view(seq_len_q, num_heads, head_dim)

    k_index = torch.arange(seq_len_kv * head_dim, device="cuda", dtype=torch.int64)
    k = ((k_index * 29 + 7) % 251 - 125).to(torch.int8)
    k = k.view(seq_len_kv, head_dim)

    weight_index = torch.arange(
        seq_len_q * num_heads, device="cuda", dtype=torch.float32
    )
    weights = ((weight_index.remainder(257) - 128) / 64).view(seq_len_q, num_heads)
    scale_index = torch.arange(seq_len_kv, device="cuda", dtype=torch.float32)
    k_scale = (scale_index.remainder(29) + 1) / 37
    ks = torch.zeros(seq_len_q, device="cuda", dtype=torch.int32)
    ke = torch.full((seq_len_q,), seq_len_kv, device="cuda", dtype=torch.int32)
    return q, k, k_scale, weights, ks, ke


def _invoke(inputs):
    q, k, k_scale, weights, ks, ke = inputs
    return deep_gemm.int8_mqa_logits(
        q,
        (k, k_scale),
        weights,
        ks,
        ke,
        clean_logits=False,
    )


def test_int8_mqa_rounding_and_eager_repeatability():
    if not _is_zw810e():
        pytest.skip("ZW810E-specific FP32 rounding contract")

    inputs = _inputs()
    output = _invoke(inputs)
    logits_sha256 = _sha256(output)
    topk = torch.topk(output, k=512, dim=1, largest=True, sorted=False).indices
    topk = torch.sort(topk, dim=1).values.to(torch.int32)
    topk_sha256 = _sha256(topk)

    assert (logits_sha256, topk_sha256) == (
        EXPECTED_LOGITS_SHA256,
        EXPECTED_TOPK_SHA256,
    ), (logits_sha256, topk_sha256)

    # The public API allocates a fresh output. Repeated eager launches must be
    # bitwise exact over the complete tensor, not merely close on sampled rows.
    for _ in range(4):
        assert torch.equal(_invoke(inputs), output)


@pytest.mark.skipif(
    USE_CPP_JIT,
    reason="C++ JIT native launches are outside PyTorch graph capture",
)
def test_int8_mqa_graph_replay_exact():
    if not _is_zw810e():
        pytest.skip("ZW810E-specific FP32 rounding contract")

    # SGLang also calls this path from captured execution. Preserve the same
    # repeatability contract under graph replay.
    inputs = _inputs()
    output = _invoke(inputs)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        graph_output = _invoke(inputs)
    graph.replay()
    torch.cuda.synchronize()
    graph_reference = graph_output.clone()
    # Captured execution must preserve the same full-output contract as eager,
    # not only remain self-consistent across graph replays.
    assert torch.equal(graph_reference, output)
    for _ in range(4):
        # Poison the complete destination before replay.  Comparing an
        # untouched output can falsely pass when capture produced an empty
        # graph and replay did no work.
        graph_output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(graph_output, graph_reference)
