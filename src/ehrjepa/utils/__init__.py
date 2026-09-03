"""Shared utilities used across the package.

This subpackage will hold the small cross-cutting helpers that do not belong to
any one stage of the pipeline: seeding for Python, NumPy, and torch (including
per-worker dataloader seeding) for reproducible runs; device resolution across
CPU, CUDA, and Apple MPS; structured logging setup and a JSONL metric writer;
timing and throughput counters; and filesystem helpers for run directories and
artifact paths. Everything here is owner-authored and dependency-light; nothing
in it may import from :mod:`ehrjepa.models` or :mod:`ehrjepa.train`, so that the
data and evaluation layers can use it without pulling in torch modules they do
not need.
"""

from ehrjepa.utils.runtime import (
    autocast_for,
    peak_memory_bytes,
    resolve_device,
    seed_everything,
    supports_bf16_autocast,
)

__all__ = [
    "autocast_for",
    "peak_memory_bytes",
    "resolve_device",
    "seed_everything",
    "supports_bf16_autocast",
]
