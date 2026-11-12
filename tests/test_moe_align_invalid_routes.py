"""Regression checks for invalid MoE routes in both alignment launch paths."""
import argparse

import torch
import deep_gemm


def check(ids, outputs, groups, block_m):
    topk = ids.shape[1]
    _, rows, metadata, tokens, blocks, inverse, experts = outputs
    ids = ids.cpu().flatten()
    rows, metadata, tokens, blocks, inverse, experts = [
        value.cpu().flatten() if name == 'tokens' else value.cpu()
        for name, value in zip(
            ['rows', 'metadata', 'tokens', 'blocks', 'inverse', 'experts'],
            [rows, metadata, tokens, blocks, inverse, experts],
        )
    ]
    padded = compact = 0
    for expert in range(groups):
        positions = torch.where(ids == expert)[0]
        count = positions.numel()
        assert rows[expert].item() == count
        offsets = inverse[positions].long() - compact
        assert torch.equal(offsets.sort().values, torch.arange(count))
        assert torch.equal(tokens[padded + offsets], positions // topk)
        assert (experts[compact:compact + count] == expert).all()
        extent = (count + block_m - 1) // block_m * block_m
        assert (tokens[padded + count:padded + extent] == ids.numel() // topk).all()
        for offset in range(padded, padded + extent, block_m):
            assert metadata[offset // block_m].tolist() == [expert, count, padded // block_m, compact]
        padded += extent
        compact += count
    assert (tokens[padded:] == ids.numel() // topk).all()
    assert blocks.item() == padded // block_m
    assert (experts[compact:] == -1).all()
    assert (inverse[(ids < 0) | (ids >= groups)] == -1).all()


def test_permuted_rows_require_consistent_inverse():
    ids = torch.tensor([[0], [0]], dtype=torch.int32)
    outputs = (None, torch.tensor([2]), torch.tensor([[0, 2, 0, 0]]),
               torch.tensor([[1, 0]]), torch.tensor([1]), torch.tensor([1, 0]),
               torch.tensor([0, 0]))
    check(ids, outputs, groups=1, block_m=2)
    try:
        check(ids, (*outputs[:5], torch.tensor([0, 1]), outputs[6]), groups=1, block_m=2)
    except AssertionError:
        return
    raise AssertionError("a swapped token without its inverse must be rejected")


def test_bf16_gemm_restores_valid_routes(check_accuracy=True):
    groups, topk, width = 128, 10, 128
    opts = dict(device="cuda", dtype=torch.bfloat16)
    rhs = torch.zeros((groups, width, width), **opts)
    rhs[:, :, 0] = 1
    rhs[:, :, 1] = torch.arange(groups, device="cuda").view(-1, 1)
    for rows in (8, 1639):
        lhs = torch.zeros((rows, width), **opts)
        token_value = torch.arange(rows, device="cuda") % 31
        lhs[:, 0] = token_value
        lhs[:, 1] = 1
        ids = torch.arange(rows * topk, device="cuda", dtype=torch.int32).reshape(rows, topk) % groups
        ids[:, :2] = -1
        ids[:, -1] = groups
        config, counts, metadata, sorted_ids, blocks, inverse, _ = deep_gemm.moe_align_block_size(lhs, rhs, ids)
        out = torch.empty((ids.numel(), width), **opts)
        deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_fused(
            lhs, rhs, out, counts, metadata, sorted_ids, blocks, config
        )
        if not check_accuracy:
            continue
        valid = (ids >= 0) & (ids < groups)
        expected = (token_value[:, None] + ids)[valid].to(torch.bfloat16)
        restored = out.index_select(0, inverse.reshape_as(ids)[valid].long())
        torch.testing.assert_close(restored, expected[:, None].expand_as(restored), atol=0, rtol=0)


def test_single_block_expert_limit():
    """Reject unsupported expert counts on the host before generating a kernel."""
    lhs = torch.empty((1, 128), device='cuda', dtype=torch.bfloat16)
    rhs = lhs.new_empty((1025, 128, 128))
    ids = torch.zeros((1, 1), device='cuda', dtype=torch.int32)
    config = (deep_gemm.get_num_sms(), 32, 128, 128, 2, 2, 3)
    expected = "single-block path supports at most 1024 experts, got 1025"
    try:
        deep_gemm.moe_align_block_size(lhs, rhs, ids, config=config)
    except RuntimeError as exc:
        assert expected in str(exc), str(exc)
    else:
        raise AssertionError("1025 experts must be rejected by the single-block host check")


def main(smoke=False, check_accuracy=True):
    align = deep_gemm.moe_align_block_size
    test_single_block_expert_limit()
    if check_accuracy:
        test_permuted_rows_require_consistent_inverse()
    if not smoke:
        test_bf16_gemm_restores_valid_routes(check_accuracy=check_accuracy)
    cases = [(128, n, 10) for n in (8, 1638, 1639, 8192)]
    cases += [(129, n, 1) for n in (16383, 16384, 16385)]
    cases += [(512, 513, 10), (512, 2048, 10), (16, 16385, 1)]
    distributions = ('valid', 'ep', 'invalid', 'out_of_range', 'hot', 'valid_again')
    if smoke:
        cases = [(129, n, 1) for n in (16384, 16385)]
        distributions = ('valid', 'out_of_range', 'invalid', 'valid_again')
    for groups, count, topk in cases:
        lhs = torch.empty((count, 128), device='cuda', dtype=torch.bfloat16)
        rhs = torch.empty((groups, 128, 128), device='cuda', dtype=torch.bfloat16)
        for block_m in ((32,) if smoke else (32, 64)):
            config = (deep_gemm.get_num_sms(), block_m, 128, 128, 2, 2, 3)
            ids = torch.arange(count * topk, device='cuda', dtype=torch.int32).reshape(count, topk) % groups
            # Reuse the same graph and buffers while changing the valid-route count.
            outputs = align(lhs, rhs, ids, config=config)
            if not check_accuracy:
                continue
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = align(lhs, rhs, ids, config=config)
            for distribution in distributions:
                ids.copy_(torch.arange(count * topk, device='cuda', dtype=torch.int32).reshape(count, topk) % groups)
                if distribution == 'ep':
                    ids.view(-1)[1::2] = -1
                    assert (ids >= 0).any() and (ids == -1).any()
                elif distribution == 'invalid':
                    ids.fill_(-1)
                elif distribution == 'out_of_range':
                    flat = ids.view(-1)
                    flat[1::3] = -7
                    flat[2::3] = groups + 1
                    assert ((ids >= 0) & (ids < groups)).any()
                    assert (ids < -1).any() and (ids >= groups).any()
                elif distribution == 'hot':
                    ids.zero_()
                eager = align(lhs, rhs, ids, config=config)
                check(ids, eager, groups, block_m)
                for value in outputs[1:]:
                    value.fill_(1234567)
                graph.replay()
                check(ids, outputs, groups, block_m)
                print('PASS', 'cpp', count, block_m, distribution, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='Only check the single/four-kernel boundary')
    args = parser.parse_args()
    main(args.smoke)
