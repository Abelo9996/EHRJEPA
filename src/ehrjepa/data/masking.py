"""Context/target masks over a padded batch of event windows.

A JEPA step needs, per sequence, two disjoint sets of valid positions: the
**context** the encoder is allowed to attend to, and the **targets** the
predictor has to reconstruct in latent space. Two strategies are mixed per batch.

``future_span``
    The clinically interesting one. A cut index is drawn uniformly in
    ``[0.3L, 0.9L]``; the targets are the next ``U(8, 64)`` events after the cut,
    and the context is everything strictly before it. Everything after the target
    span is **dropped** from both, so the task is genuinely "given this history,
    what does the next stretch of this patient's record look like" rather than an
    interpolation with both endpoints pinned.

``multi_block``
    The I-JEPA one, which forces the encoder to represent local structure rather
    than only a prefix summary. Two to four target blocks, each ``5-15%`` of the
    sequence, are carved out; the context is their complement, thinned further by
    dropping a random ``0-30%`` of what is left.

Both return boolean ``(B, L)`` masks that are disjoint, never touch padding, and
always leave at least one target -- short sequences degrade gracefully rather
than producing an empty batch. Sampling is done on the CPU with an explicit
:class:`torch.Generator` so a run is reproducible from its seed.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "DEFAULT_P_FUTURE",
    "MASK_STRATEGIES",
    "multi_block_mask",
    "sample_masks",
    "future_span_mask",
]

MASK_STRATEGIES = ("future_span", "multi_block")

#: Fraction of a batch that gets ``future_span``; the rest get ``multi_block``.
DEFAULT_P_FUTURE = 0.6


def _randint(low: int, high: int, generator: torch.Generator | None) -> int:
    """Uniform integer in ``[low, high]`` inclusive, tolerating ``low >= high``."""
    if high <= low:
        return low
    return int(torch.randint(low, high + 1, (1,), generator=generator).item())


def _rand(generator: torch.Generator | None) -> float:
    return float(torch.rand((1,), generator=generator).item())


def future_span_mask(
    length: int,
    min_span: int = 8,
    max_span: int = 64,
    cut_low: float = 0.3,
    cut_high: float = 0.9,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Context = events before a cut, targets = the span just after it.

    Returns two boolean vectors of length ``length``. Events after the target span
    are in neither. ``length`` may be as small as 2; the span and the cut are
    clamped so the context and the targets each keep at least one event.
    """
    context = torch.zeros(length, dtype=torch.bool)
    target = torch.zeros(length, dtype=torch.bool)
    if length < 2:
        return context, target
    low = max(1, int(cut_low * length))
    high = max(low, min(length - 1, int(cut_high * length)))
    cut = _randint(low, high, generator)
    span = _randint(min(min_span, length - cut), min(max_span, length - cut), generator)
    span = max(1, min(span, length - cut))
    context[:cut] = True
    target[cut : cut + span] = True
    return context, target


def multi_block_mask(
    length: int,
    min_blocks: int = 2,
    max_blocks: int = 4,
    block_low: float = 0.05,
    block_high: float = 0.15,
    context_drop: float = 0.3,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Two to four target blocks; context is the thinned complement.

    Blocks are placed one at a time and may overlap -- the union is what counts,
    so the realised target fraction is at most ``max_blocks * block_high``. The
    context then drops a random ``U(0, context_drop)`` share of the remaining
    valid positions, which is I-JEPA's way of keeping the context from being a
    near-complete view of the sequence.
    """
    target = torch.zeros(length, dtype=torch.bool)
    if length < 2:
        return torch.zeros(length, dtype=torch.bool), target
    n_blocks = _randint(min_blocks, max_blocks, generator)
    for _ in range(n_blocks):
        size = max(
            1, int(round(length * (block_low + (block_high - block_low) * _rand(generator))))
        )
        size = min(size, max(1, length - 1))
        start = _randint(0, length - size, generator)
        target[start : start + size] = True
    if not bool(target.any()):
        target[_randint(0, length - 1, generator)] = True
    if bool(target.all()):
        target[_randint(0, length - 1, generator)] = False
    context = ~target
    keep = torch.rand(length, generator=generator) >= context_drop * _rand(generator)
    thinned = context & keep
    if bool(thinned.any()):
        context = thinned
    return context, target


def sample_masks(
    attention_mask: Tensor,
    p_future: float = DEFAULT_P_FUTURE,
    generator: torch.Generator | None = None,
    strategy: str | None = None,
    **kwargs: object,
) -> tuple[Tensor, Tensor]:
    """Draw ``(context_mask, target_mask)`` for a collated batch.

    ``attention_mask`` is the ``(B, L)`` mask from
    :func:`~ehrjepa.data.dataset.collate_events`; only its valid prefix of each
    row is ever masked into context or targets, so padding is structurally
    excluded. Each row independently gets ``future_span`` with probability
    ``p_future`` and ``multi_block`` otherwise, unless ``strategy`` forces one.

    Keyword arguments prefixed ``future_`` or ``block_`` are forwarded to the
    corresponding strategy (e.g. ``future_max_span=32``).
    """
    if strategy is not None and strategy not in MASK_STRATEGIES:
        raise ValueError(f"strategy must be one of {MASK_STRATEGIES}, got {strategy!r}")
    future_kwargs = {k[len("future_") :]: v for k, v in kwargs.items() if k.startswith("future_")}
    block_kwargs = {k[len("block_") :]: v for k, v in kwargs.items() if k.startswith("block_")}
    unknown = (
        set(kwargs) - {f"future_{k}" for k in future_kwargs} - {f"block_{k}" for k in block_kwargs}
    )
    if unknown:
        raise ValueError(f"unknown masking options: {sorted(unknown)}")

    valid = attention_mask.bool().cpu()
    batch, width = valid.shape
    context = torch.zeros(batch, width, dtype=torch.bool)
    target = torch.zeros(batch, width, dtype=torch.bool)
    for row in range(batch):
        length = int(valid[row].sum())
        if length < 2:
            continue
        use_future = strategy == "future_span" or (strategy is None and _rand(generator) < p_future)
        if use_future:
            ctx, tgt = future_span_mask(length, generator=generator, **future_kwargs)  # type: ignore[arg-type]
        else:
            ctx, tgt = multi_block_mask(length, generator=generator, **block_kwargs)  # type: ignore[arg-type]
        context[row, :length] = ctx
        target[row, :length] = tgt

    device = attention_mask.device
    return context.to(device), target.to(device)
