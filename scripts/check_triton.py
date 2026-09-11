"""Verify Triton can compile a kernel on this host: ``python scripts/check_triton.py``."""

import torch

x = torch.randn(4, 8, 16, device="cuda", dtype=torch.bfloat16)
y = torch.compile(lambda a: a.softmax(-1))(x)
print("triton ok", tuple(y.shape))
