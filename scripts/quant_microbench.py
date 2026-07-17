"""Single-GPU quant microbench — isolates the mxfp4 quantize kernel (no peers,
generation=0, no arrival) for acu profiling and Phase1/Phase2 clock split.

Run:
  python scripts/quant_microbench.py            # timing only
  acu --devices 0 --kernel-name-base mangled \
      --kernel-name regex:'mxfp4_quantize' --launch-count 1 \
      python scripts/quant_microbench.py
"""
import os, sys, torch
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo)
from deep_gemm import get_sym_buffer_size
from deep_gemm.jit_kernels.dispatch_fused_gemm import _mxfp4_quantize_to_sym_buffer

dev = 'cuda:0'
num_local = 12
num_total = 12
hidden = 7168
num_tokens = 256
topk = 6
max_tokens = 256

torch.manual_seed(0)
x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=dev)
x[:, ::32] *= 50.0
scores = torch.randn(num_tokens, num_total, device=dev)
topk_ids = torch.topk(scores, topk, dim=-1).indices.to(torch.int32)

buf_size = get_sym_buffer_size(num_local, num_total, max_tokens, hidden)
sym_buf = torch.zeros(buf_size, dtype=torch.uint8, device=dev)

def one():
    _mxfp4_quantize_to_sym_buffer(
        x, topk_ids, sym_buf,
        num_local_experts=num_total, num_total_experts=num_total,
        max_tokens_per_expert=max_tokens, generation=0)

# warmup (also triggers JIT compile)
for _ in range(5):
    one()
torch.cuda.synchronize()

n = 100
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
ev[0].record()
for _ in range(n):
    one()
ev[1].record()
torch.cuda.synchronize()
print(f"quant kernel (gen=0, 1 rank): {ev[0].elapsed_time(ev[1])/n*1e3:.2f} us/call  "
      f"(num_tokens={num_tokens}, hidden={hidden}, topk={topk})")
