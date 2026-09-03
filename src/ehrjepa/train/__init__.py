"""Pretraining: configuration, the training loop, and checkpointing.

This subpackage will own self-supervised pretraining. Planned contents: typed
configuration dataclasses loaded from plain YAML (no hydra) covering data paths,
tokenizer and masking parameters, model width/depth, optimizer and schedule, and
the SIGReg coefficient; the training loop itself, with AdamW, cosine learning
rate and weight decay schedules, linear warmup, gradient clipping, EMA momentum
scheduling for the target encoder, and mixed precision; device selection across
CPU, CUDA, and Apple MPS; resumable checkpointing that stores model, EMA, and
optimizer state together with the config that produced them; and lightweight
JSONL metric logging so runs stay inspectable without a tracking service. The
loop is written single-process first; distributed support, if added, will be
plain ``torchrun`` + DDP.
"""

from ehrjepa.train.config import (
    DataConfig,
    MaskingConfig,
    OptimConfig,
    PretrainConfig,
    RunConfig,
    load_config,
)

__all__ = [
    "DataConfig",
    "MaskingConfig",
    "OptimConfig",
    "PretrainConfig",
    "RunConfig",
    "load_config",
]
