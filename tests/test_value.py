"""The continuous ``value_z`` auxiliary: what it is masked to, and that it learns.

Three properties carry the design.

1. **The mask is the whole safety property.** ``value_z`` is stored as ``0.0`` for
   events that carry no measurement, and ``0.0`` is also an ordinary z-score, so a
   term that regressed on the unmasked tensor would spend most of its gradient
   teaching the head to emit "average" for admissions and discharges. The mask is
   ``value_bin != 0``, and these tests pin that changing a masked-out target's
   ``value_z`` cannot move the loss.
2. **It reaches the right rows.** For ``nextlatent`` the head reads the predicted
   latent at position ``i``, horizon ``k``, and the target is the *number carried
   by the event at* ``i + k`` -- concatenated across horizons in the same row
   order as the predictions, which is easy to get one horizon out of step. For
   ``ar`` the head reads ``h_i`` and the target is event ``i + 1``'s number, the
   same event whose code the softmax predicts.
3. **It trains.** A single batch, memorised, with every other term switched off.

``lambda_value: 0`` builds no head, so the shipped defaults are untouched --
``tests/test_train.py::test_the_default_objective_reproduces_its_recorded_loss``
is the test that says so, and it is unchanged.
"""

from __future__ import annotations

import pytest
import torch

from ehrjepa.models.ar import EHRAR
from ehrjepa.models.jepa import EHRJEPA, EHRJEPAConfig
from ehrjepa.models.latent import EHRNextLatent
from ehrjepa.objectives.ar import ARObjective
from ehrjepa.objectives.latent import LatentObjective
from ehrjepa.objectives.loss import (
    JEPAObjective,
    ObjectiveConfig,
    value_regression_loss,
)

VOCAB = 48
DIM = 32


def _config(**overrides) -> EHRJEPAConfig:
    values = dict(
        vocab_size=VOCAB,
        dim=DIM,
        depth=2,
        heads=4,
        mlp="gelu",
        mlp_ratio=2.0,
        dropout=0.0,
        n_freq=8,
        pred_dim=16,
        pred_depth=1,
        pred_heads=2,
        value_head=True,
    )
    values.update(overrides)
    return EHRJEPAConfig(**values)


def _batch(batch: int = 3, length: int = 12, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    code_id = torch.randint(4, VOCAB, (batch, length), generator=g)
    # Every third event carries no number, so the mask has something to do.
    value_bin = torch.randint(1, 11, (batch, length), generator=g)
    value_bin[:, ::3] = 0
    return {
        "code_id": code_id,
        "value_bin": value_bin,
        "value_z": torch.randn(batch, length, generator=g),
        "age": torch.rand(batch, length, generator=g) * 80,
        "log_delta": torch.rand(batch, length, generator=g),
        "attention_mask": torch.ones(batch, length, dtype=torch.long),
        "time_min": torch.arange(length).expand(batch, length) * 60,
    }


# --------------------------------------------------------------------------- #
# The loss itself


def test_the_value_term_scores_only_the_targets_that_carry_a_number() -> None:
    torch.manual_seed(0)
    head = torch.nn.Linear(DIM, 1)
    hidden = torch.randn(20, DIM)
    value_z = torch.randn(20)
    value_bin = torch.zeros(20, dtype=torch.long)
    value_bin[:7] = torch.randint(1, 11, (7,))

    full = value_regression_loss(head, hidden, value_z, value_bin)
    subset = value_regression_loss(
        head,
        hidden[:7],
        value_z[:7],
        torch.ones(7, dtype=torch.long),
    )
    assert torch.allclose(full, subset, atol=0.0)


def test_a_value_less_target_cannot_move_the_value_term() -> None:
    torch.manual_seed(0)
    head = torch.nn.Linear(DIM, 1)
    hidden = torch.randn(16, DIM)
    value_bin = torch.zeros(16, dtype=torch.long)
    value_bin[::2] = 4
    value_z = torch.randn(16)
    before = value_regression_loss(head, hidden, value_z, value_bin)
    poisoned = value_z.clone()
    poisoned[1::2] += 100.0  # every masked-out row
    after = value_regression_loss(head, hidden, poisoned, value_bin)
    assert torch.allclose(before, after, atol=0.0)


def test_a_batch_with_no_numeric_target_scores_zero_rather_than_nan() -> None:
    head = torch.nn.Linear(DIM, 1)
    hidden = torch.randn(8, DIM)
    loss = value_regression_loss(head, hidden, torch.randn(8), torch.zeros(8, dtype=torch.long))
    assert float(loss) == 0.0


def test_the_huber_transition_bounds_a_wild_outlier() -> None:
    """Delta 1.0: an error of 100 costs ~100, not ~10,000."""
    head = torch.nn.Identity()
    hidden = torch.zeros(1, 1)
    loss = value_regression_loss(head, hidden, torch.tensor([100.0]), torch.tensor([5]))
    assert float(loss) == pytest.approx(99.5, abs=1e-3)


# --------------------------------------------------------------------------- #
# Where the targets come from, per objective


def test_nextlatent_lines_its_value_targets_up_with_its_predictions() -> None:
    """Horizon ``k``'s targets are the numbers of the events ``k`` steps ahead."""
    torch.manual_seed(0)
    horizons = [1, 3]
    model = EHRNextLatent(_config(causal=True, build_predictor=False, horizons=horizons)).eval()
    batch = _batch()
    with torch.no_grad():
        out = model(batch)
    sizes = [int(n) for n in out.extras["horizon_sizes"].tolist()]
    assert sizes == [3 * 11, 3 * 9]
    start = 0
    for step, size in zip(horizons, sizes, strict=True):
        stop = start + size
        expected_z = batch["value_z"][:, step:].reshape(-1)
        expected_bin = batch["value_bin"][:, step:].reshape(-1)
        assert torch.allclose(out.extras["value_target_z"][start:stop], expected_z, atol=0.0)
        assert torch.equal(out.extras["value_target_bin"][start:stop], expected_bin)
        start = stop


def test_ar_takes_the_next_events_number_at_the_positions_it_scores() -> None:
    torch.manual_seed(0)
    model = EHRAR(_config(causal=True)).eval()
    batch = _batch()
    with torch.no_grad():
        out = model(batch)
    # Every position but the last of each row is scored, in row-major order.
    assert out.value_z.shape == out.targets.shape
    assert torch.allclose(out.value_z, batch["value_z"][:, 1:].reshape(-1), atol=0.0)
    assert torch.equal(out.value_bin, batch["value_bin"][:, 1:].reshape(-1))


def test_masked_span_jepa_takes_the_target_events_own_number() -> None:
    torch.manual_seed(0)
    model = EHRJEPA(_config()).eval()
    batch = _batch()
    target_mask = torch.zeros_like(batch["attention_mask"], dtype=torch.bool)
    target_mask[:, -4:] = True
    context_mask = ~target_mask
    with torch.no_grad():
        out = model(batch, context_mask, target_mask)
    index = target_mask.nonzero(as_tuple=True)
    assert torch.allclose(out.extras["value_target_z"], batch["value_z"][index], atol=0.0)
    assert torch.equal(out.extras["value_target_bin"], batch["value_bin"][index])


# --------------------------------------------------------------------------- #
# That it learns


def _memorise(model, objective, batch, steps: int, forward) -> tuple[float, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = last = 0.0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = forward(model, objective, batch)
        losses["loss"].backward()
        optimizer.step()
        value = float(losses["value_loss"])
        if step == 0:
            first = value
        last = value
    return first, last


def test_the_value_term_falls_on_a_memorizable_batch_under_nextlatent() -> None:
    torch.manual_seed(0)
    config = ObjectiveConfig(
        kind="nextlatent",
        horizons=[1],
        lambda_pred=0.0,
        lambda_sigreg=0.0,
        lambda_recon=0.0,
        lambda_value=1.0,
    )
    model = EHRNextLatent(_config(causal=True, build_predictor=False, horizons=[1]))
    objective = LatentObjective(config, value_head=model.value_head)
    first, last = _memorise(
        model,
        objective,
        _batch(seed=1),
        60,
        lambda m, o, b: o(m(b, compute_targets=False)),
    )
    assert last < 0.5 * first, f"value_loss went {first:.4f} -> {last:.4f}"


def test_the_value_term_falls_on_a_memorizable_batch_under_ar() -> None:
    torch.manual_seed(0)
    model = EHRAR(_config(causal=True))
    objective = ARObjective(lambda_value=1.0, value_head=model.value_head)

    def forward(m, o, b):
        out = m(b)
        return o(m.head, out.hidden, out.targets, value_z=out.value_z, value_bin=out.value_bin)

    first, last = _memorise(model, objective, _batch(seed=2), 60, forward)
    assert last < 0.5 * first, f"value_loss went {first:.4f} -> {last:.4f}"


# --------------------------------------------------------------------------- #
# The decile head beside the code term, per objective


def test_recon_value_builds_a_bin_head_for_ar_and_for_nextlatent() -> None:
    """The grid's `ar_bins` and `hybrid_bins` cells need it on both objectives.

    Before this, ``objective.recon_value`` was read only by ``kind: jepa``: the
    head was built for a hybrid cell and never called, and for an ``ar`` cell it
    was not built at all. Both now predict the *next* event's decile off the same
    hidden row whose code the softmax predicts.
    """
    from ehrjepa.train.config import from_mapping

    def resolve(**objective) -> EHRJEPAConfig:
        raw = {"model": {"dim": DIM, "causal": True}, "objective": objective}
        return from_mapping(raw).model_config(VOCAB)

    assert resolve(kind="ar", recon_value=True).recon_value_head
    assert resolve(kind="nextlatent", lambda_recon=0.1, recon_value=True).recon_value_head
    # No code term to sit beside means no bin head for the latent objectives.
    assert not resolve(kind="nextlatent", lambda_recon=0.0, recon_value=True).recon_value_head
    assert not resolve(kind="ar").recon_value_head


def test_the_ar_bin_term_scores_the_next_events_decile() -> None:
    torch.manual_seed(0)
    model = EHRAR(_config(causal=True, value_head=False, recon_value_head=True))
    objective = ARObjective(recon_value_head=model.recon_value_head)
    batch = _batch(seed=3)
    out = model(batch)
    stats = objective(model.head, out.hidden, out.targets, value_bin=out.value_bin)
    expected = torch.nn.functional.cross_entropy(
        model.recon_value_head(out.hidden).float(), out.value_bin
    )
    assert torch.allclose(stats["recon_value_loss"], expected.detach(), atol=0.0)
    assert torch.allclose(stats["loss"], stats["ce"] + expected, atol=1e-6)


def test_the_nextlatent_bin_term_rides_inside_the_recon_weight() -> None:
    torch.manual_seed(0)
    config = ObjectiveConfig(kind="nextlatent", horizons=[1], lambda_recon=0.1, lambda_sigreg=0.0)
    model = EHRNextLatent(
        _config(
            causal=True,
            build_predictor=False,
            horizons=[1],
            value_head=False,
            recon_head=True,
            recon_value_head=True,
        )
    )
    objective = LatentObjective(
        config,
        recon_head=model.recon_head,
        recon_value_head=model.recon_value_head,
    )
    batch = _batch(seed=4)
    out = model(batch, compute_targets=False)
    # The bins are the next event's, at the positions the code term scores.
    shifted = torch.zeros_like(batch["value_bin"])
    shifted[:, :-1] = batch["value_bin"][:, 1:]
    scored = out.extras["recon_value_bin"]
    assert scored.numel() == out.extras["recon_code_id"].numel()
    assert set(scored.tolist()) <= set(shifted.reshape(-1).tolist())
    losses = objective(out)
    assert float(losses["recon_value_loss"]) > 0.0
    # ``recon_loss`` is the code term *plus* the bin term, and the total weights
    # the pair by ``lambda_recon`` -- the bin head does not get a weight of its
    # own, which is the whole point of putting it inside.
    assert float(losses["recon_loss"]) > float(losses["recon_value_loss"])
    assert float(losses["loss"]) == pytest.approx(
        float(losses["pred_loss"]) + 0.1 * float(losses["recon_loss"]), abs=1e-6
    )


def test_no_head_is_built_and_no_term_is_added_at_the_default_weight() -> None:
    """The shipped default is off, and off means "absent", not "weighted zero"."""
    assert ObjectiveConfig().lambda_value == 0.0
    assert not EHRJEPAConfig(vocab_size=VOCAB).value_head
    model = EHRJEPA(_config(value_head=False))
    assert model.value_head is None
    assert not any("value_head" in name for name in model.state_dict())
    objective = JEPAObjective(ObjectiveConfig(), recon_head=None, value_head=None)
    batch = _batch()
    target_mask = torch.zeros_like(batch["attention_mask"], dtype=torch.bool)
    target_mask[:, -4:] = True
    losses = objective(model(batch, ~target_mask, target_mask))
    assert "value_loss" not in losses
