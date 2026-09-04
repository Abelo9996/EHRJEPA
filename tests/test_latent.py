"""The two causal latent objectives: leakage, horizon arithmetic, and the skip rule.

Four properties carry the design and are pinned numerically here rather than
argued for in a docstring:

1. ``nextlatent`` cannot see the future. Perturbing any event after position ``i``
   leaves the prediction at ``i`` bit-identical.
2. Its loss over several horizons is the *mean* of the per-horizon losses, not a
   row-count-weighted blend of them.
3. With ``lambda_pred: 0``, ``lambda_sigreg: 0`` and ``lambda_recon: 1`` it is
   exactly the next-code AR objective -- same number, to floating point.
4. ``window`` pools only events inside ``(t_a, t_a + H]``, skips anchors whose
   horizon runs past the window's last event or contains no event at all, and
   builds its multi-hot label from that same set.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import pytest
import torch

from ehrjepa.data.masking import sample_anchors
from ehrjepa.data.tokenize import PAD_ID
from ehrjepa.eval import probe
from ehrjepa.models.ar import EHRAR
from ehrjepa.models.jepa import EHRJEPAConfig
from ehrjepa.models.latent import EHRNextLatent, EHRWindowLatent
from ehrjepa.objectives.ar import ar_loss_chunked, next_code_targets
from ehrjepa.objectives.latent import LatentObjective, multilabel_bce_chunked
from ehrjepa.objectives.loss import ObjectiveConfig, jepa_loss
from ehrjepa.train.config import load_config
from ehrjepa.train.pretrain import Trainer

REPO = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = REPO / "configs" / "pretrain_debug.yaml"
DEMO_CACHE = REPO / "data" / "cache" / "mimic-demo"

requires_cache = pytest.mark.skipif(
    not (DEMO_CACHE / "meta.json").exists(), reason="mimic-demo cache is not built"
)

VOCAB = 64
DAY = 1440  # minutes


def _config(**overrides) -> EHRJEPAConfig:
    values = dict(
        vocab_size=VOCAB,
        dim=32,
        depth=2,
        heads=4,
        mlp="gelu",
        mlp_ratio=2.0,
        dropout=0.0,
        n_freq=8,
        causal=True,
        build_predictor=False,
        target_mode="shared",
    )
    values.update(overrides)
    return EHRJEPAConfig(**values)


def _batch(batch: int = 3, length: int = 24, day_step: int = 1) -> dict[str, torch.Tensor]:
    """A batch whose events are ``day_step`` days apart, so horizons are countable."""
    g = torch.Generator().manual_seed(7)
    times = torch.arange(length).repeat(batch, 1) * day_step * DAY
    return {
        "code_id": torch.randint(1, VOCAB, (batch, length), generator=g),
        "value_bin": torch.randint(0, 11, (batch, length), generator=g),
        "value_z": torch.randn(batch, length, generator=g),
        "age": torch.rand(batch, length, generator=g) * 90,
        "log_delta": torch.rand(batch, length, generator=g) * 8,
        "attention_mask": torch.ones(batch, length, dtype=torch.long),
        "time_min": times,
    }


# --------------------------------------------------------------------------- #
# Anchor sampling
# --------------------------------------------------------------------------- #


def test_anchors_fall_in_the_band_are_distinct_and_are_seeded() -> None:
    mask = torch.ones(4, 40, dtype=torch.long)
    mask[3, 20:] = 0  # a short row
    anchors, valid = sample_anchors(mask, n_anchors=8, generator=torch.Generator().manual_seed(0))
    assert anchors.shape == valid.shape == (4, 8)
    for row in range(4):
        length = int(mask[row].sum())
        picks = anchors[row][valid[row]].tolist()
        assert len(picks) == len(set(picks)), "anchors are drawn without replacement"
        assert picks == sorted(picks)
        assert all(math.ceil(0.3 * length) <= p < 0.9 * length for p in picks)
        assert all(p >= 1 for p in picks), "position 0 has no history to summarise"

    again, _ = sample_anchors(mask, n_anchors=8, generator=torch.Generator().manual_seed(0))
    other, _ = sample_anchors(mask, n_anchors=8, generator=torch.Generator().manual_seed(1))
    assert torch.equal(anchors, again)
    assert not torch.equal(anchors, other)


def test_a_band_narrower_than_k_takes_what_it_can() -> None:
    mask = torch.ones(1, 10, dtype=torch.long)  # band is [3, 9): six positions
    anchors, valid = sample_anchors(mask, n_anchors=8, generator=torch.Generator().manual_seed(0))
    assert int(valid.sum()) == 6
    assert not bool(valid[0, 6:].any())


# --------------------------------------------------------------------------- #
# Design A -- dense next-latent
# --------------------------------------------------------------------------- #


def test_nextlatent_predictions_do_not_see_the_future() -> None:
    """Everything after position ``i`` is perturbed; the prediction at ``i`` must not move."""
    torch.manual_seed(0)
    model = EHRNextLatent(_config(horizons=[1, 4])).eval()
    batch = _batch()
    cut = 10
    with torch.no_grad():
        base = model(batch)
        tampered = {k: v.clone() for k, v in batch.items()}
        g = torch.Generator().manual_seed(99)
        tampered["code_id"][:, cut + 1 :] = torch.randint(
            1, VOCAB, tampered["code_id"][:, cut + 1 :].shape, generator=g
        )
        tampered["value_bin"][:, cut + 1 :] = 0
        tampered["value_z"][:, cut + 1 :] = 0.0
        after = model(tampered)

    rows, cols = base.target_index
    keep = cols <= cut
    assert int(keep.sum()) > 0
    assert torch.equal(base.predictions[keep], after.predictions[keep])
    # And the control: something downstream of the cut *did* move.
    assert not torch.equal(base.predictions, after.predictions)


def test_nextlatent_pairs_every_valid_position_with_its_k_step_target() -> None:
    model = EHRNextLatent(_config(horizons=[1, 4])).eval()
    batch = _batch(batch=2, length=16)
    with torch.no_grad():
        out = model(batch)
    assert out.extras["horizon_sizes"].tolist() == [2 * 15, 2 * 12]
    assert out.predictions.shape == out.targets.shape == (2 * 27, 32)

    with torch.no_grad():
        tokens = model.embed_batch(batch)
        latents = model.window_targets(batch, tokens)
    assert torch.allclose(out.targets[: 2 * 15], latents[:, 1:].reshape(-1, 32))


def test_nextlatent_loss_is_the_mean_over_horizons_not_over_rows() -> None:
    torch.manual_seed(0)
    model = EHRNextLatent(_config(horizons=[1, 4])).eval()
    batch = _batch()
    with torch.no_grad():
        out = model(batch)
    objective = LatentObjective(
        ObjectiveConfig(kind="nextlatent", horizons=[1, 4], lambda_sigreg=0.0)
    )
    first = int(out.extras["horizon_sizes"][0])
    sizes = out.extras["horizon_sizes"].tolist()
    per_horizon = [
        float(jepa_loss(out.predictions[:first], out.targets[:first])),
        float(jepa_loss(out.predictions[first:], out.targets[first:])),
    ]
    assert sizes[0] != sizes[1], "the horizons must score different numbers of rows"
    losses = objective(out)
    assert float(losses["pred_loss"]) == pytest.approx(sum(per_horizon) / 2, rel=1e-6)
    pooled = float(jepa_loss(out.predictions, out.targets))
    assert float(losses["pred_loss"]) != pytest.approx(pooled, rel=1e-6)


def test_nextlatent_with_lambda_pred_zero_is_exactly_the_ar_objective() -> None:
    """Same weights, same batch: the loss must equal ``EHRAR`` + ``ar_loss_chunked``."""
    torch.manual_seed(0)
    config = _config(recon_head=True)
    model = EHRNextLatent(config).eval()
    ar = EHRAR(_config(recon_head=False, build_predictor=True)).eval()
    ar.embed.load_state_dict(model.embed.state_dict())
    ar.encoder.load_state_dict(model.encoder.state_dict())
    ar.head.load_state_dict(model.recon_head.state_dict())

    batch = _batch()
    objective = LatentObjective(
        ObjectiveConfig(kind="nextlatent", lambda_pred=0.0, lambda_sigreg=0.0, lambda_recon=1.0),
        recon_head=model.recon_head,
    )
    with torch.no_grad():
        losses = objective(model(batch, compute_targets=False))
        ar_out = ar(batch)
        reference = ar_loss_chunked(ar.head, ar_out.hidden, ar_out.targets)

    assert math.isnan(float(losses["pred_loss"]))
    assert float(losses["loss"]) == pytest.approx(float(reference["loss"]), rel=1e-6)
    assert float(losses["ce"]) == pytest.approx(float(reference["ce"]), rel=1e-6)
    assert float(losses["top10"]) == pytest.approx(float(reference["top10"]), rel=1e-6)


def test_nextlatent_recon_reads_the_encoder_not_the_predicted_latent() -> None:
    """The AR term's rows are the encoder outputs at every position with a successor."""
    model = EHRNextLatent(_config(recon_head=True)).eval()
    batch = _batch()
    with torch.no_grad():
        out = model(batch)
    scored = next_code_targets(batch["code_id"], batch["attention_mask"].bool()) != PAD_ID
    assert torch.equal(out.extras["recon_hidden"], out.context_tokens[scored])
    assert out.extras["recon_code_id"].shape[0] == int(scored.sum())


# --------------------------------------------------------------------------- #
# Design B -- pooled future window
# --------------------------------------------------------------------------- #


def _window_model(**overrides) -> EHRWindowLatent:
    torch.manual_seed(0)
    return EHRWindowLatent(_config(**overrides)).eval()


def test_window_target_pools_exactly_the_events_inside_the_horizon() -> None:
    """One anchor, one horizon, events one day apart: the pool is countable by hand."""
    model = _window_model(window_horizons=[3.0])
    batch = _batch(batch=1, length=20, day_step=1)
    anchors = torch.tensor([[6]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    with torch.no_grad():
        out = model(batch, anchors, mask)
        tokens = model.embed_batch(batch)
        latents = model.window_targets(batch, tokens)
    # Events at day 7, 8, 9 are in (day 6, day 9]; day 10 is not, day 6 is not.
    expected = latents[0, 7:10].mean(dim=0)
    assert out.targets.shape == (1, 32)
    assert torch.allclose(out.targets[0], expected, atol=1e-6)


def test_window_summary_is_the_encoder_output_just_before_the_anchor() -> None:
    model = _window_model(window_horizons=[3.0])
    batch = _batch(batch=1, length=20)
    anchors = torch.tensor([[6]])
    with torch.no_grad():
        out = model(batch, anchors, torch.ones(1, 1, dtype=torch.bool))
        direct = model.window_head(
            out.context_tokens[:, 5] + model.horizon_emb.weight[0]
        )
    assert torch.allclose(out.predictions, direct, atol=1e-6)


def test_window_skips_horizons_that_run_past_the_window_and_pools_that_are_empty() -> None:
    """Two horizons, one observable and one not, plus an anchor with a gap after it."""
    model = _window_model(window_horizons=[2.0, 100.0])
    batch = _batch(batch=1, length=12, day_step=1)
    # Push everything after position 8 far into the future, so an anchor at 8
    # has no event within two days even though the window continues.
    batch["time_min"][0, 9:] = torch.tensor([50, 51, 52]) * DAY
    anchors = torch.tensor([[4, 8]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    with torch.no_grad():
        out = model(batch, anchors, mask)

    # Horizon 2 days: anchor 4 (events at day 5, 6) kept; anchor 8 empty.
    # Horizon 100 days: both ends run past the last event (day 52), both skipped.
    assert out.extras["horizon_sizes"].tolist() == [1, 0]
    assert float(out.extras["anchors_offered"]) == 4.0
    assert float(out.extras["anchors_unobserved"]) == 2.0
    assert float(out.extras["anchors_empty"]) == 1.0
    losses = LatentObjective(ObjectiveConfig(kind="window", window_horizons=[2.0, 100.0]))(out)
    assert float(losses["skipped_frac"]) == pytest.approx(0.75)


def test_window_multihot_labels_are_the_codes_inside_the_horizon() -> None:
    model = _window_model(window_horizons=[3.0], recon_head=True)
    batch = _batch(batch=1, length=20, day_step=1)
    batch["code_id"][0, 7:10] = torch.tensor([5, 5, 11])  # a repeat and a distinct code
    anchors = torch.tensor([[6]])
    with torch.no_grad():
        out = model(batch, anchors, torch.ones(1, 1, dtype=torch.bool))
    codes = out.extras["window_codes"]
    assert codes.shape == (1, 20)
    assert sorted(set(codes[0].tolist()) - {PAD_ID}) == [5, 11]
    assert float(out.extras["positives_per_anchor"]) == pytest.approx(2.0)

    # And the BCE the objective computes is the one those labels define.
    with torch.no_grad():
        logits = model.recon_head(out.predictions)
        target = torch.zeros_like(logits)
        target[0, 5] = 1.0
        target[0, 11] = 1.0
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        actual = multilabel_bce_chunked(model.recon_head, out.predictions, codes)
    assert float(actual) == pytest.approx(float(expected), rel=1e-6)


def test_window_bce_chunking_does_not_change_the_value() -> None:
    model = _window_model(window_horizons=[3.0, 10.0], recon_head=True)
    batch = _batch(batch=3, length=24)
    anchors, mask = sample_anchors(
        batch["attention_mask"], n_anchors=4, generator=torch.Generator().manual_seed(2)
    )
    with torch.no_grad():
        out = model(batch, anchors, mask)
        whole = multilabel_bce_chunked(
            model.recon_head, out.predictions, out.extras["window_codes"], chunk=10_000
        )
        sliced = multilabel_bce_chunked(
            model.recon_head, out.predictions, out.extras["window_codes"], chunk=3
        )
    assert float(whole) == pytest.approx(float(sliced), rel=1e-6)


# --------------------------------------------------------------------------- #
# Construction and the trainer
# --------------------------------------------------------------------------- #


def test_the_latent_models_refuse_a_bidirectional_encoder_or_a_target_span() -> None:
    with pytest.raises(ValueError, match="causal"):
        EHRNextLatent(_config(causal=False))
    with pytest.raises(ValueError, match="span_only"):
        EHRWindowLatent(_config(target_span_only=True))
    with pytest.raises(ValueError, match="build_predictor"):
        EHRNextLatent(_config(build_predictor=True))
    with pytest.raises(ValueError, match="horizons"):
        EHRNextLatent(_config(horizons=[]))


def test_the_config_folds_the_horizons_into_the_model_section() -> None:
    config = load_config(
        DEBUG_CONFIG,
        [
            "objective.kind=nextlatent",
            "objective.horizons=[1,4,16]",
            "model.causal=true",
        ],
    )
    model_config = config.model_config(vocab_size=101)
    assert model_config.build_predictor is False
    assert model_config.horizons == [1, 4, 16]
    assert len(EHRNextLatent(model_config).pred_heads) == 3

    jepa = load_config(DEBUG_CONFIG).model_config(vocab_size=101)
    assert jepa.build_predictor is True, "the masked-span model keeps its predictor"


@requires_cache
@pytest.mark.parametrize(
    ("kind", "overrides"),
    [
        ("nextlatent", ["objective.horizons=[1,4]", "objective.lambda_recon=0.1"]),
        # Minutes, not months: a mimic-demo window is one ICU stay, so a 30-day
        # horizon would be unobserved at every anchor and the cell would train on
        # nothing. The DE-SynPUF grid's horizons are in the grid file.
        ("window", ["objective.window_horizons=[0.005,0.01]", "objective.window_anchors=4"]),
    ],
)
def test_a_short_run_trains_logs_and_reloads_through_the_probe(
    tmp_path: Path, kind: str, overrides: list[str]
) -> None:
    started = time.perf_counter()
    config = load_config(
        DEBUG_CONFIG,
        [
            f"objective.kind={kind}",
            "run.steps=3",
            "run.log_every=1",
            "run.tensorboard=false",
            f"run.out_dir={tmp_path}",
            *overrides,
        ],
    )
    trainer = Trainer(config)
    assert trainer.model_config.causal is True, "the trainer enables causal attention"
    final = trainer.train()
    assert time.perf_counter() - started < 60.0
    assert final["pred_loss"] > 0
    if kind == "window":
        assert 0.0 <= final["skipped_frac"] < 1.0, "the run must score some anchors"

    model, _ = probe.load_encoder(tmp_path / "final.pt")
    assert isinstance(model, EHRNextLatent if kind == "nextlatent" else EHRWindowLatent)
    assert probe.checkpoint_is_causal(tmp_path / "final.pt") is True
    assert probe.default_features(causal=True) == "last"
