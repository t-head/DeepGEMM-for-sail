"""Different graphs captured on one stream need independent metadata scratch."""
import pytest
import torch
import deep_gemm as dg


@pytest.mark.parametrize('paged', [False, True])
def test_sparse_metadata_graph_workspace(paged):
    capture_stream = torch.cuda.Stream()
    replay_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs, outputs, references, inputs, tables = [], [], [], [], []
    for q in (8192, 4096):
        starts = torch.zeros(q, dtype=torch.int32, device='cuda')
        ends = torch.full((q,), 16, dtype=torch.int32, device='cuda')
        indices = torch.zeros((q, 256), dtype=torch.int32, device='cuda')
        inputs.append((starts, ends, indices))
    torch.cuda.synchronize()
    for starts, ends, indices in inputs:
        if paged:
            table = torch.zeros((starts.numel(), 1), dtype=torch.int32, device='cuda')
            tables.append(table)
            call = lambda: dg.get_paged_sparse_mqa_logits_metadata(
                ends, table, starts, 16, indices, torch.int8, 16)
        else:
            call = lambda: dg.get_sparse_mqa_logits_metadata(starts, ends, 16, indices, torch.int8, 16)
        torch.cuda.synchronize()
        with torch.cuda.stream(capture_stream):
            ref = call()
        capture_stream.synchronize()
        references.append(ref.view(torch.int32)[:3].clone())
        graph = torch.cuda.CUDAGraph()
        # Each graph has its own pool; both are captured using the same stream.
        with torch.cuda.graph(graph, stream=capture_stream):
            out = call()
        graphs.append(graph)
        outputs.append(out)
    torch.cuda.synchronize()
    for _ in range(40):
        for out in outputs:
            out.fill_(0x55)
        torch.cuda.synchronize()
        for graph, stream in zip(graphs, replay_streams):
            with torch.cuda.stream(stream):
                graph.replay()
        torch.cuda.synchronize()
        for out, expected in zip(outputs, references):
            torch.testing.assert_close(out.view(torch.int32)[:3], expected, rtol=0, atol=0)
