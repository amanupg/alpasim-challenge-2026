"""Stochastic-rounding AdamW (a15_ft.train.SRAdamW): small updates to bf16 weights must survive in expectation."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from a15_ft.train import SRAdamW  # noqa: E402

bad = 0


def check(name, ok):
    global bad
    bad += not ok
    print(("ok  " if ok else "BAD ") + name)


torch.manual_seed(0)
x = torch.full((200000,), 0.02) - 1e-5
check("round-to-nearest loses a 1e-5 step on 0.02", float(x.to(torch.bfloat16).float().mean()) == float(torch.tensor(0.02).to(torch.bfloat16)))
check("stochastic rounding keeps it in expectation", abs(float(SRAdamW._sr_bf16(x.clone()).float().mean()) - float(x.mean())) < 2e-6)
p = torch.nn.Parameter(torch.full((20000,), 0.02, dtype=torch.bfloat16))
q = torch.nn.Parameter(torch.full((20000,), 0.02, dtype=torch.bfloat16))
o, o2 = SRAdamW([p], lr=1e-5), torch.optim.AdamW([q], lr=1e-5, weight_decay=0.0)
for _ in range(50):
    p.grad, q.grad = torch.ones_like(p), torch.ones_like(q)
    o.step(); o2.step()
want = 0.02 - 50 * 1e-5
print(f"after 50 steps: want {want:.6f}, SRAdamW {float(p.float().mean()):.6f}, torch AdamW on bf16 {float(q.float().mean()):.6f}")
check("SRAdamW tracks the fp32 trajectory", abs(float(p.float().mean()) - want) < 2e-5)
print("PASS" if bad == 0 else f"FAIL ({bad})")
sys.exit(bad > 0)
