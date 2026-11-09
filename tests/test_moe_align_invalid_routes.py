"""Regression checks for invalid MoE routes in both alignment launch paths."""
import torch
import deep_gemm


def check(ids, outputs, groups, block_m):
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
        assert torch.equal(tokens[padded:padded + count], positions // 10)
        assert torch.equal(inverse[positions], torch.arange(compact, compact + count, dtype=torch.int32))
        assert (experts[compact:compact + count] == expert).all()
        extent = (count + block_m - 1) // block_m * block_m
        assert (tokens[padded + count:padded + extent] == ids.numel() // 10).all()
        for offset in range(padded, padded + extent, block_m):
            assert metadata[offset // block_m].tolist() == [expert, count, padded // block_m, compact]
        padded += extent
        compact += count
    assert blocks.item() == padded // block_m
    assert (experts[compact:] == -1).all()
    assert (inverse[(ids < 0) | (ids >= groups)] == -1).all()


def main():
    align = deep_gemm.moe_align_block_size
    groups = 128
    for count in (8, 1638, 1639, 8192):
        lhs = torch.empty((count, 128), device='cuda', dtype=torch.bfloat16)
        rhs = torch.empty((groups, 128, 128), device='cuda', dtype=torch.bfloat16)
        for block_m in (32, 64):
            config = (deep_gemm.get_num_sms(), block_m, 128, 128, 2, 2, 3)
            ids = torch.arange(count * 10, device='cuda', dtype=torch.int32).reshape(count, 10) % groups
            # Reuse the same graph and buffers while changing the valid-route count.
            outputs = align(lhs, rhs, ids, config=config)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = align(lhs, rhs, ids, config=config)
            for distribution in ('valid', 'ep', 'invalid', 'out_of_range', 'valid_again'):
                ids.copy_(torch.arange(count * 10, device='cuda', dtype=torch.int32).reshape(count, 10) % groups)
                if distribution == 'ep':
                    ids[:, 2:] = -1
                elif distribution == 'invalid':
                    ids.fill_(-1)
                elif distribution == 'out_of_range':
                    ids[:, 2:5] = -7
                    ids[:, 5:] = groups + 1
                eager = align(lhs, rhs, ids, config=config)
                check(ids, eager, groups, block_m)
                for value in outputs[1:]:
                    value.fill_(1234567)
                graph.replay()
                check(ids, outputs, groups, block_m)
                print('PASS', 'cpp', count, block_m, distribution, flush=True)


if __name__ == '__main__':
    main()
