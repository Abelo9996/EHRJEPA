"""Invariants of the context/target mask sampler.

The masks are the task definition: if they overlap, the predictor can copy; if
they touch padding, the loss is computed on nothing; if the target set is empty,
the step is a no-op that still consumes budget.
"""

from __future__ import annotations

import pytest
import torch

from ehrjepa.data.masking import future_span_mask, multi_block_mask, sample_masks


def _attention_mask(lengths: list[int]) -> torch.Tensor:
    width = max(lengths)
    return (torch.arange(width)[None, :] < torch.tensor(lengths)[:, None]).long()


@pytest.mark.parametrize("strategy", ["future_span", "multi_block"])
def test_masks_are_disjoint_nonempty_and_inside_the_valid_prefix(strategy: str) -> None:
    lengths = [512, 300, 128, 64, 33, 17]
    attention = _attention_mask(lengths)
    generator = torch.Generator().manual_seed(7)
    for _ in range(20):
        context, target = sample_masks(attention, generator=generator, strategy=strategy)
        assert not bool((context & target).any()), "context and target overlap"
        padding = attention == 0
        assert not bool((context & padding).any()), "context includes padding"
        assert not bool((target & padding).any()), "target includes padding"
        for row, length in enumerate(lengths):
            assert bool(target[row, :length].any()), f"row {row} has no targets"
            assert bool(context[row, :length].any()), f"row {row} has no context"


def test_future_span_targets_never_precede_the_context() -> None:
    generator = torch.Generator().manual_seed(11)
    for length in (20, 64, 200, 512):
        for _ in range(50):
            context, target = future_span_mask(length, generator=generator)
            ctx_idx = context.nonzero().flatten()
            tgt_idx = target.nonzero().flatten()
            assert len(ctx_idx) and len(tgt_idx)
            assert int(ctx_idx.max()) < int(tgt_idx.min()), "a target lands before the cut"
            # The target span is contiguous and starts exactly at the cut.
            assert int(tgt_idx.min()) == int(ctx_idx.max()) + 1
            assert torch.equal(tgt_idx, torch.arange(int(tgt_idx.min()), int(tgt_idx.max()) + 1))


def test_future_span_respects_the_cut_and_span_bounds() -> None:
    generator = torch.Generator().manual_seed(3)
    length = 400
    for _ in range(200):
        context, target = future_span_mask(length, generator=generator)
        cut = int(context.sum())
        span = int(target.sum())
        assert 0.3 * length - 1 <= cut <= 0.9 * length
        assert 1 <= span <= 64
        # Everything after the span is dropped from both masks.
        assert not bool(context[cut:].any())
        assert not bool(target[cut + span :].any())


def test_multi_block_context_is_a_subset_of_the_target_complement() -> None:
    generator = torch.Generator().manual_seed(5)
    for _ in range(100):
        context, target = multi_block_mask(200, generator=generator)
        assert not bool((context & target).any())
        assert bool(target.any()) and bool(context.any())
        # Context is the complement, minus the extra random drop.
        assert bool((context | target).sum() <= 200)


def test_mixture_uses_both_strategies() -> None:
    attention = _attention_mask([256] * 64)
    generator = torch.Generator().manual_seed(0)
    context, target = sample_masks(attention, p_future=0.5, generator=generator)
    # future_span rows have a contiguous target block that starts right after the
    # context; multi_block rows generally do not. Both must appear.
    contiguous = [
        bool(
            torch.equal(
                target[row].nonzero().flatten(),
                torch.arange(
                    int(target[row].nonzero().min()), int(target[row].nonzero().max()) + 1
                ),
            )
        )
        for row in range(attention.shape[0])
    ]
    assert any(contiguous) and not all(contiguous)


def test_short_sequences_are_skipped_rather_than_crashing() -> None:
    attention = _attention_mask([1, 1, 40])
    context, target = sample_masks(attention, generator=torch.Generator().manual_seed(1))
    assert not bool(context[0].any()) and not bool(target[0].any())
    assert bool(target[2].any())


def test_sampling_is_reproducible_from_the_generator_seed() -> None:
    attention = _attention_mask([128, 96, 64])
    a = sample_masks(attention, generator=torch.Generator().manual_seed(42))
    b = sample_masks(attention, generator=torch.Generator().manual_seed(42))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_unknown_masking_option_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown masking options"):
        sample_masks(_attention_mask([32]), nonsense=1)
