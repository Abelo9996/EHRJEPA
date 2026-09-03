"""The next-code autoregressive objective -- the compute-matched control for JEPA.

The question phase 5a exists to answer is whether predicting *latents* buys
anything over predicting *tokens* at the same token budget. That needs a baseline
that differs from the JEPA runs in exactly one place: the loss. So this module
reuses :class:`~ehrjepa.models.embedding.EventEmbedding` and
:class:`~ehrjepa.models.encoder.Encoder` unchanged (the encoder switched to
``causal=True``) and adds the smallest possible head on top.

**The head.** ``logits[b, i] = W @ h[b, i] + bias`` over the code vocabulary,
where ``h[b, i]`` is the encoder output at event ``i``, which under causal
attention has seen events ``0..i`` and the CLS key. The target is
``code_id[b, i + 1]``. With ``tie_embeddings=True`` (the default) ``W`` *is* the
code embedding table, LeCun/Press-Wolf style: 7.7M of the small model's 13.5M
parameters live in that table, and giving the head its own copy would make the
"same encoder, different loss" comparison a comparison of parameter counts.

**What is ignored.** A position contributes to the loss only when both it and its
successor are real events: ``PAD`` targets are dropped via ``ignore_index``, and
so is any position whose own row is padding. The last valid position of every
window has no successor and is dropped too. Only ``code_id`` is predicted --
``value_bin``, ``value_z``, ``age`` and ``log_delta`` remain inputs, never
targets, which keeps the objective one softmax rather than a multi-task blend
whose weighting would be another knob to defend.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ehrjepa.data.tokenize import PAD_ID

__all__ = ["ARObjective", "ARStats", "NextCodeHead", "ar_loss", "next_code_targets"]


class NextCodeHead(nn.Module):
    """``dim -> vocab_size`` logits, optionally tied to the code embedding table.

    Parameters
    ----------
    dim:
        Encoder width.
    vocab_size:
        Rows of the code vocabulary.
    tied_weight:
        The ``(vocab_size, dim)`` embedding matrix to reuse as the output
        projection. ``None`` allocates a fresh untied ``nn.Linear``.
    """

    def __init__(self, dim: int, vocab_size: int, tied_weight: Tensor | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.tied = tied_weight is not None
        self.norm = nn.LayerNorm(dim)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        if tied_weight is None:
            self.proj: nn.Linear | None = nn.Linear(dim, vocab_size, bias=False)
            nn.init.trunc_normal_(self.proj.weight, std=0.02, a=-0.04, b=0.04)
            self._tied: list[Tensor] = []
        else:
            self.proj = None
            # Held inside a plain list on purpose: ``nn.Module.__setattr__``
            # intercepts a bare ``Parameter`` and would register a second entry
            # for a tensor the embedding already owns, putting it twice in
            # ``state_dict`` and once too often in the weight-decay bookkeeping.
            self._tied = [tied_weight]

    @property
    def weight(self) -> Tensor:
        if self._tied:
            return self._tied[0]
        assert self.proj is not None
        return self.proj.weight

    def forward(self, hidden: Tensor) -> Tensor:
        """``(B, L, dim) -> (B, L, vocab_size)``."""
        return F.linear(self.norm(hidden), self.weight, self.bias)


def next_code_targets(code_id: Tensor, valid_mask: Tensor) -> Tensor:
    """Targets for position ``i``: ``code_id[i + 1]``, or ``PAD_ID`` where undefined.

    ``PAD_ID`` doubles as the ignore index, which is safe because a real event can
    never carry it -- ``PAD_ID`` is reserved by the tokenizer.
    """
    valid = valid_mask.bool()
    shifted = torch.full_like(code_id, PAD_ID)
    shifted[:, :-1] = code_id[:, 1:]
    keep = torch.zeros_like(valid)
    keep[:, :-1] = valid[:, :-1] & valid[:, 1:]
    return torch.where(keep, shifted, torch.full_like(shifted, PAD_ID))


class ARStats(dict):
    """``loss``/``ce``/``top1``/``top10``/``n_targets``, all scalar tensors."""


def ar_loss(logits: Tensor, targets: Tensor, top_k: tuple[int, ...] = (1, 10)) -> ARStats:
    """Softmax cross-entropy over ``code_id``, ignoring ``PAD`` targets.

    ``logits`` is ``(B, L, V)`` with ``targets`` ``(B, L)``, or already-gathered
    ``(N, V)`` with ``(N,)`` -- :class:`~ehrjepa.models.ar.EHRAR` gathers first, so
    the ``V``-wide tensor is never materialised at padding positions. Either way
    ``PAD_ID`` marks positions to skip. Top-``k`` accuracies are computed on the
    same non-ignored positions and returned detached; only ``ce`` carries gradient.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    keep = flat_targets != PAD_ID
    n = int(keep.sum())
    if n == int(flat_targets.numel()):
        keep = None  # already gathered; skip a full-size copy of the logits
    stats = ARStats(n_targets=torch.as_tensor(float(n)))
    if n == 0:
        zero = flat_logits.new_zeros(())
        stats.update(loss=zero, ce=zero.detach(), **{f"top{k}": zero.detach() for k in top_k})
        return stats
    picked = (flat_logits if keep is None else flat_logits[keep]).float()
    gold = flat_targets if keep is None else flat_targets[keep]
    ce = F.cross_entropy(picked, gold)
    stats["loss"] = ce
    stats["ce"] = ce.detach()
    with torch.no_grad():
        largest = max(top_k)
        ranked = picked.topk(min(largest, picked.shape[-1]), dim=-1).indices
        hit = ranked == gold[:, None]
        for k in top_k:
            stats[f"top{k}"] = hit[:, :k].any(dim=-1).float().mean().detach()
    return stats


class ARObjective(nn.Module):
    """Module wrapper so the trainer can hold the AR loss the way it holds JEPA's."""

    def __init__(self, top_k: tuple[int, ...] = (1, 10)) -> None:
        super().__init__()
        self.top_k = top_k

    def forward(self, logits: Tensor, targets: Tensor) -> ARStats:
        return ar_loss(logits, targets, top_k=self.top_k)
