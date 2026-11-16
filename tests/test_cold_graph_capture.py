"""First-use graph capture in fresh processes with empty and populated JIT caches."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def run_worker(kind, phase, cache):
    compiler_keys = ('DG_JIT_HGCC_COMPILER', 'DG_HGCC_COMPILER')
    env = {key: value for key, value in os.environ.items()
           if (not key.startswith('DG_') or key in compiler_keys) and key != 'HGGC_WARM_UP'}
    env.update(DG_JIT_CACHE_DIR=str(cache), DG_CACHE_DIR=str(cache / 'python'),
               DG_JIT_USE_HGRTC='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    # Let the check-in runner time out and terminate the whole process group.
    subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                    '--worker', kind, phase], env=env,
                   stderr=subprocess.STDOUT, check=True)


@pytest.mark.parametrize('kind', ['bf16', 'fp8-channel', 'fp8-block'])
@pytest.mark.parametrize('cache_state', ['empty', 'disk'])
def test_cold_graph_capture(kind, cache_state, tmp_path):
    cache = tmp_path / 'jit'
    if cache_state == 'disk':
        # This process exits before capture, retaining only the disk cache.
        run_worker(kind, 'seed', cache)
    run_worker(kind, cache_state, cache)


def worker(kind, phase):
    import torch
    import deep_gemm as dg
    from deep_gemm.utils.math import (per_block_cast_to_fp8, per_custom_dims_cast_to_fp8,
                                     per_token_cast_to_fp8)

    torch.set_num_threads(1)
    assert dg.get_compile_mode() == 0
    cache = Path(os.environ['DG_JIT_CACHE_DIR'])
    manifest = cache / 'target.json'

    def binaries():
        return {str(path.relative_to(cache)) for path in cache.rglob('kernel.hgbin')}

    def inputs(shape, seed):
        m, n, k = shape
        generator = torch.Generator().manual_seed(seed)
        a = torch.randn((m, k), generator=generator).bfloat16()
        b = torch.randn((n, k), generator=generator).bfloat16()
        if kind == 'bf16':
            tensors, reference = (a, b), a.float() @ b.float().T
        elif kind == 'fp8-channel':
            aq, sa = per_custom_dims_cast_to_fp8(a, (0,), keep_scale_dim=True)
            bq, sb = per_custom_dims_cast_to_fp8(b, (0,), keep_scale_dim=True)
            tensors, reference = (aq, bq, sa, sb), (aq.float() * sa) @ (bq.float() * sb).T
        else:
            aq, sa = per_token_cast_to_fp8(a)
            bq, sb = per_block_cast_to_fp8(b)
            ad = aq.float() * sa.repeat_interleave(128, dim=1)
            bd = bq.float() * sb.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
            tensors, reference = (aq, bq, sa, sb), ad @ bd.T
        return tensors, reference

    def launch(tensors, out):
        if kind == 'bf16':
            dg.bf16_gemm_nt(*tensors, out)
        else:
            dg.fp8_gemm_nt((tensors[0], tensors[2]), (tensors[1], tensors[3]), out)

    def check(actual, reference):
        assert torch.isfinite(actual).all()
        actual, reference = actual.double(), reference.double()
        diff = ((actual - reference).square().sum() /
                (actual.square() + reference.square()).sum()).item()
        assert diff < .001, diff
        return diff

    target_shape = (128, 256, 256)
    target_cpu, reference = inputs(target_shape, 185)
    target = [tensor.to('cuda') for tensor in target_cpu]
    out = torch.empty(target_shape[:2], dtype=torch.bfloat16, device='cuda')
    if phase == 'seed':
        assert not binaries()
        launch(target, out)
        torch.cuda.synchronize()
        check(out.cpu(), reference)
        assert binaries()
        manifest.write_text(json.dumps(sorted(binaries())))
        return

    seeded = set(json.loads(manifest.read_text())) if phase == 'disk' else set()
    assert binaries() == seeded
    # A shape change alone can reuse an FP8 specialization. Warm a distinct BF16
    # kernel, and verify that the target remains unseen rather than assuming it.
    warm_cpu = (torch.ones((64, 128), dtype=torch.bfloat16),
                torch.ones((128, 128), dtype=torch.bfloat16))
    warm = [tensor.to('cuda') for tensor in warm_cpu]
    warm_out = torch.empty((64, 128), dtype=torch.bfloat16, device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        dg.bf16_gemm_nt(*warm, warm_out)
    torch.cuda.synchronize()
    before = binaries()
    assert before - seeded, 'Warm-up must compile a different specialization'
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch(target, out)
    after = binaries()
    if phase == 'empty':
        assert after - before, 'Target must first compile inside capture'
    else:
        assert after == before, 'Target must load the pre-existing disk cache'

    for seed in (185, 186, 901):
        cpu, reference = inputs(target_shape, seed)
        for destination, source in zip(target, cpu):
            destination.copy_(source)
        out.fill_(float('nan'))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        actual = out.cpu()
        diff = check(actual, reference)
        out.fill_(float('nan'))
        launch(target, out)
        torch.cuda.synchronize()
        assert torch.equal(actual, out.cpu()), 'Graph/eager mismatch'
        print(json.dumps(dict(kind=kind, cache_state=phase, seed=seed,
                              diff=diff, eager_graph_exact=True)), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        worker(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
