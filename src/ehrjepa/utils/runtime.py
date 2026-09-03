"""Seeding, device and precision resolution, and peak-memory accounting.

Kept dependency-light and free of any import from :mod:`ehrjepa.models` or
:mod:`ehrjepa.train`, per the subpackage contract.
"""

from __future__ import annotations

import os
import random
from contextlib import AbstractContextManager, nullcontext

import numpy as np
import torch

__all__ = [
    "autocast_for",
    "peak_memory_bytes",
    "reset_peak_memory",
    "resolve_device",
    "seed_everything",
    "supports_bf16_autocast",
]


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch, and ask torch for deterministic kernels.

    ``deterministic`` is best-effort: it sets the cuDNN flags and the
    ``CUBLAS_WORKSPACE_CONFIG`` env var, but does not call
    ``use_deterministic_algorithms`` -- several ops used here (scatter-style
    indexing, ``scaled_dot_product_attention`` backward) have no deterministic
    MPS kernel and would raise rather than run.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(name: str = "auto") -> torch.device:
    """``"auto"`` picks CUDA, then MPS, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def supports_bf16_autocast(device: torch.device) -> bool:
    """Probe whether bf16 autocast actually runs a linear + attention on ``device``.

    torch 2.14 does accept ``torch.autocast("mps", bfloat16)``, but the support is
    version- and op-dependent enough that guessing is worse than a 2 ms probe.
    """
    if device.type == "cuda":
        return bool(torch.cuda.is_bf16_supported())
    if device.type == "cpu":
        return True
    try:
        x = torch.randn(2, 4, 8, device=device)
        layer = torch.nn.Linear(8, 8).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            y = layer(x)
            torch.nn.functional.scaled_dot_product_attention(
                y.unsqueeze(1), y.unsqueeze(1), y.unsqueeze(1)
            )
        y.float().sum().backward()
    except Exception:  # pragma: no cover - depends on the torch build
        return False
    return True


def autocast_for(device: torch.device, precision: str) -> AbstractContextManager:
    """Autocast context for ``precision`` in ``{"fp32", "bf16", "auto"}``.

    ``"auto"`` means bf16 on CUDA and float32 everywhere else -- the MPS backend
    is float32 by default here because bf16 buys little on unified memory and the
    kernel coverage varies; ``"bf16"`` forces it on and probes first.
    """
    if precision == "fp32":
        return nullcontext()
    if precision == "auto":
        return (
            torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        )
    if precision != "bf16":
        raise ValueError(f"precision must be 'fp32', 'bf16' or 'auto', got {precision!r}")
    if not supports_bf16_autocast(device):
        return nullcontext()
    return torch.autocast(device.type, dtype=torch.bfloat16)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_bytes(device: torch.device) -> int:
    """Peak allocator bytes so far. On MPS this is the *current* driver
    allocation, which the caller maxes over steps; there is no peak counter."""
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    return 0
