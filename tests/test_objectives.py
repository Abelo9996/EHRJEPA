"""SIGReg behaviour on known distributions, the prediction loss, and diagnostics.

The point of SIGReg is that it is near zero on a genuine isotropic Gaussian sample
and large on anything degenerate. These tests pin both ends of that, since a
regularizer that fires on everything or on nothing would still train.
"""

from __future__ import annotations

import math

import pytest
import torch

from ehrjepa.data.masking import sample_masks
from ehrjepa.models import EHRJEPA, EHRJEPAConfig
from ehrjepa.objectives.loss import (
    JEPAObjective,
    ObjectiveConfig,
    collapse_diagnostics,
    jepa_loss,
)
from ehrjepa.objectives.sigreg import epps_pulley, random_directions, sigreg


def _sigreg(x: torch.Tensor, seed: int = 0, **kwargs) -> float:
    return float(sigreg(x, generator=torch.Generator().manual_seed(seed), **kwargs))


def test_random_directions_are_unit_norm_and_fresh() -> None:
    a = random_directions(16, 128, generator=torch.Generator().manual_seed(0))
    assert a.shape == (16, 128)
    assert torch.allclose(a.norm(dim=0), torch.ones(128), atol=1e-5)
    b = random_directions(16, 128, generator=torch.Generator().manual_seed(1))
    assert not torch.allclose(a, b)


def test_sigreg_is_near_zero_on_a_true_standard_normal_sample() -> None:
    x = torch.randn(8192, 64, generator=torch.Generator().manual_seed(0))
    value = _sigreg(x, n_directions=256)
    assert value < 5e-3, f"SIGReg on N(0, I) should be ~0, got {value}"


def test_sigreg_is_large_on_collapsed_constant_embeddings() -> None:
    normal = torch.randn(4096, 64, generator=torch.Generator().manual_seed(0))
    constant = torch.full((4096, 64), 0.7)
    assert _sigreg(constant, n_directions=256) > 100 * _sigreg(normal, n_directions=256)
    # A point mass has |phi(t)| = 1 everywhere, so the statistic converges to the
    # weighted integral of |exp(i t mu) - exp(-t^2/2)|^2, an O(0.1) positive
    # constant independent of the sample size -- the floor collapse cannot go below.
    assert _sigreg(constant, n_directions=256) > 0.2


def test_sigreg_is_large_on_a_rank_one_embedding() -> None:
    g = torch.Generator().manual_seed(0)
    direction = torch.randn(64, generator=g)
    direction = direction / direction.norm()
    rank_one = torch.randn(4096, 1, generator=g) * direction  # every row on one line
    normal = torch.randn(4096, 64, generator=g)
    assert _sigreg(rank_one, n_directions=256) > 50 * _sigreg(normal, n_directions=256)


def test_sigreg_detects_wrong_scale_and_wrong_mean() -> None:
    g = torch.Generator().manual_seed(0)
    base = torch.randn(4096, 32, generator=g)
    reference = _sigreg(base, n_directions=256)
    assert _sigreg(base * 4.0, n_directions=256) > 20 * reference
    assert _sigreg(base + 3.0, n_directions=256) > 20 * reference


def test_epps_pulley_matches_a_direct_quadrature() -> None:
    """The vectorised implementation against a literal transcription of the formula."""
    x = torch.randn(512, 1, generator=torch.Generator().manual_seed(0))
    grid = torch.linspace(-5.0, 5.0, 17)
    values = []
    for t in grid.tolist():
        real = float(torch.cos(t * x).mean()) - math.exp(-0.5 * t * t)
        imag = float(torch.sin(t * x).mean())
        values.append((real**2 + imag**2) * math.exp(-t * t))
    expected = float(torch.trapezoid(torch.tensor(values), grid))
    assert float(epps_pulley(x)[0]) == pytest.approx(expected, rel=1e-4)


def test_epps_pulley_n_prefactor_is_exactly_the_row_count() -> None:
    x = torch.randn(300, 4, generator=torch.Generator().manual_seed(0))
    plain = epps_pulley(x)
    scaled = epps_pulley(x, scale_by_n=True)
    assert torch.allclose(scaled, plain * 300, rtol=1e-5)


def test_sigreg_chunking_does_not_change_the_value() -> None:
    x = torch.randn(1024, 16, generator=torch.Generator().manual_seed(0))
    a = epps_pulley(x, t_chunk=1)
    b = epps_pulley(x, t_chunk=17)
    assert torch.allclose(a, b, atol=1e-6)


def test_sigreg_carries_gradient() -> None:
    x = torch.randn(256, 8, requires_grad=True)
    value = sigreg(x, n_directions=32, generator=torch.Generator().manual_seed(0))
    value.backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0


def test_sigreg_does_not_standardize_its_input() -> None:
    """Shifting the input must move the statistic; a centering step would hide it."""
    x = torch.randn(2048, 16, generator=torch.Generator().manual_seed(0))
    assert _sigreg(x + 2.0, n_directions=128) != pytest.approx(_sigreg(x, n_directions=128))


# --------------------------------------------------------------------------- #


def test_jepa_loss_is_zero_when_predictions_equal_normed_targets() -> None:
    targets = torch.randn(64, 32)
    normed = torch.nn.functional.layer_norm(targets, (32,))
    assert float(jepa_loss(normed, targets)) == pytest.approx(0.0, abs=1e-6)
    assert float(jepa_loss(torch.zeros(0, 32), torch.zeros(0, 32))) == 0.0


def test_jepa_loss_beta_changes_the_quadratic_region() -> None:
    predictions = torch.zeros(8, 4)
    targets = torch.nn.functional.layer_norm(torch.randn(8, 4), (4,))
    small = float(jepa_loss(predictions, targets, beta=0.1))
    large = float(jepa_loss(predictions, targets, beta=2.0))
    assert small > large  # smaller beta leaves the quadratic region sooner


def test_collapse_diagnostics_separate_collapse_from_structure() -> None:
    mask = torch.ones(8, 16, dtype=torch.bool)
    healthy = torch.randn(8, 16, 32)
    collapsed = torch.zeros(8, 16, 32) + 0.3
    predictions = torch.randn(64, 32)

    a = collapse_diagnostics(healthy, mask, predictions, predictions)
    b = collapse_diagnostics(collapsed, mask, predictions, torch.randn(64, 32))
    assert a["effective_rank"] > 20 and b["effective_rank"] < 2
    assert a["mean_std"] > 0.8 and b["mean_std"] == pytest.approx(0.0, abs=1e-6)
    # Predicting your own target is the maximum cosine gap available.
    assert a["cos_gap"] > 0.5
    assert abs(b["cos_gap"]) < 0.5


# --------------------------------------------------------------------------- #
# The auxiliary reconstruction term
# --------------------------------------------------------------------------- #

_RECON_VOCAB = 48


def _recon_model(**overrides) -> EHRJEPA:
    values = dict(
        vocab_size=_RECON_VOCAB,
        dim=32,
        depth=2,
        heads=4,
        pred_dim=16,
        pred_depth=2,
        pred_heads=2,
        dropout=0.0,
        n_freq=8,
        target_mode="ema",
    )
    values.update(overrides)
    return EHRJEPA(EHRJEPAConfig(**values))


def _recon_batch(batch: int = 4, length: int = 32) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(11)
    return {
        "code_id": torch.randint(1, _RECON_VOCAB, (batch, length), generator=g),
        "value_bin": torch.randint(0, 11, (batch, length), generator=g),
        "value_z": torch.randn(batch, length, generator=g),
        "age": torch.rand(batch, length, generator=g) * 90,
        "log_delta": torch.rand(batch, length, generator=g) * 8,
        "attention_mask": torch.ones(batch, length, dtype=torch.long),
    }


def test_reconstruction_is_off_and_costs_nothing_by_default() -> None:
    """No head, no term, no key missing from the logged row."""
    model = _recon_model()
    assert model.recon_head is None and model.recon_value_head is None
    assert not any("recon" in name for name in model.state_dict())

    batch = _recon_batch()
    context, target = sample_masks(
        batch["attention_mask"], generator=torch.Generator().manual_seed(3)
    )
    losses = JEPAObjective(ObjectiveConfig())(model(batch, context, target))
    assert float(losses["recon_loss"]) == 0.0
    assert float(losses["recon_value_loss"]) == 0.0


def test_reconstruction_loss_falls_on_a_memorizable_batch() -> None:
    """One batch, over and over: the code CE must come down off ``log(vocab)``.

    This is the cheapest evidence that the term is wired to something that can
    learn -- the head reads the predictor's output, which carries gradient, and
    the targets are that batch's own codes.
    """
    torch.manual_seed(0)
    model = _recon_model(recon_head=True, recon_value_head=True).train()
    objective = JEPAObjective(
        ObjectiveConfig(lambda_sigreg=0.0, lambda_recon=1.0, recon_value=True),
        recon_head=model.recon_head,
        recon_value_head=model.recon_value_head,
    )
    batch = _recon_batch()
    context, target = sample_masks(
        batch["attention_mask"], generator=torch.Generator().manual_seed(3)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    history = []
    for _ in range(60):
        losses = objective(model(batch, context, target))
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
        history.append((float(losses["recon_loss"]), float(losses["recon_value_loss"])))

    first_code, first_value = history[0]
    last_code, last_value = history[-1]
    assert first_code == pytest.approx(math.log(_RECON_VOCAB), abs=0.6)
    assert last_code < first_code - 1.0, history[:: len(history) // 4]
    assert last_value < first_value - 0.1


def test_lambda_pred_zero_drops_the_prediction_term_and_logs_nan() -> None:
    """``lambda_pred: 0`` -> loss is exactly ``lambda_recon * recon`` plus SIGReg.

    ``output.targets`` here stands in for the zero placeholder
    :meth:`EHRJEPA.forward <ehrjepa.models.jepa.EHRJEPA.forward>` returns for
    ``compute_targets=False`` -- the objective must not fold a term computed
    against it into the total, and must report ``pred_loss`` as ``NaN`` rather
    than a real-looking number.
    """
    torch.manual_seed(0)
    model = _recon_model(recon_head=True)
    batch = _recon_batch()
    context, target = sample_masks(
        batch["attention_mask"], generator=torch.Generator().manual_seed(3)
    )
    out = model(batch, context, target, compute_targets=False)
    assert torch.equal(out.targets, torch.zeros_like(out.predictions))

    objective = JEPAObjective(
        ObjectiveConfig(lambda_pred=0.0, lambda_recon=0.1, lambda_sigreg=0.05),
        recon_head=model.recon_head,
    )
    losses = objective(out)
    assert math.isnan(float(losses["pred_loss"]))

    expected = 0.1 * float(losses["recon_loss"]) + 0.05 * (
        float(losses["sigreg_tokens"]) + float(losses["sigreg_cls"])
    )
    assert float(losses["loss"].detach()) == pytest.approx(expected, rel=1e-5)


def test_lambda_pred_default_reproduces_the_full_loss() -> None:
    """The default (``lambda_pred: 1``) must be a pure pass-through of ``pred_loss``."""
    torch.manual_seed(0)
    model = _recon_model()
    batch = _recon_batch()
    context, target = sample_masks(
        batch["attention_mask"], generator=torch.Generator().manual_seed(3)
    )
    out = model(batch, context, target)
    objective = JEPAObjective(ObjectiveConfig(lambda_sigreg=0.05))
    losses = objective(out)
    expected = float(losses["pred_loss"]) + 0.05 * (
        float(losses["sigreg_tokens"]) + float(losses["sigreg_cls"])
    )
    assert float(losses["loss"].detach()) == pytest.approx(expected, rel=1e-6)


def test_reconstruction_head_is_tied_and_not_double_counted() -> None:
    """The head reuses the code table; the objective must not register it twice."""
    model = _recon_model(recon_head=True)
    assert model.recon_head is not None
    assert model.recon_head.weight is model.embed.code_emb.weight
    objective = JEPAObjective(ObjectiveConfig(lambda_recon=0.1), recon_head=model.recon_head)
    assert list(objective.parameters()) == []
    keys = [k for k in model.state_dict() if "recon_head" in k]
    assert keys == ["recon_head.bias", "recon_head.norm.weight", "recon_head.norm.bias"]
