"""Shapes, masking behaviour, the information-leak guarantee, and the EMA update."""

from __future__ import annotations

import copy

import pytest
import torch

from ehrjepa.data.masking import sample_masks
from ehrjepa.models import EHRJEPA, EHRJEPAConfig, EventEmbedding, ema_momentum
from ehrjepa.models.layers import FourierFeatures, RotaryEmbedding, apply_rope

VOCAB = 64


def _config(**overrides) -> EHRJEPAConfig:
    values = dict(
        vocab_size=VOCAB,
        dim=32,
        depth=2,
        heads=4,
        pred_dim=16,
        pred_depth=2,
        pred_heads=2,
        dropout=0.0,
        n_freq=8,
    )
    values.update(overrides)
    return EHRJEPAConfig(**values)


def _batch(batch: int = 3, length: int = 40, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    attention = torch.ones(batch, length, dtype=torch.long)
    attention[-1, length - 7 :] = 0
    return {
        "code_id": torch.randint(0, VOCAB, (batch, length), generator=g),
        "value_bin": torch.randint(0, 11, (batch, length), generator=g),
        "value_z": torch.randn(batch, length, generator=g),
        "age": torch.rand(batch, length, generator=g) * 90,
        "log_delta": torch.rand(batch, length, generator=g) * 8,
        "attention_mask": attention,
    }


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


def test_fourier_features_shape_and_range() -> None:
    fourier = FourierFeatures(n_freq=16)
    out = fourier(torch.linspace(-5, 120, 37))
    assert out.shape == (37, 32)
    assert out.abs().max() <= 1.0 + 1e-6
    # Log-spaced: the frequency ratios are constant.
    ratios = fourier.freqs[1:] / fourier.freqs[:-1]
    assert torch.allclose(ratios, ratios[0].expand_as(ratios), rtol=1e-5)


def test_rope_preserves_norm_and_is_relative() -> None:
    rope = RotaryEmbedding(16)
    cos, sin = rope(8, torch.device("cpu"), torch.float32)
    x = torch.randn(2, 3, 8, 16)
    rotated = apply_rope(x, cos, sin)
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    # Rotary is relative: <R_i u, R_j u> depends only on (i - j).
    u = torch.randn(16)
    rotated = apply_rope(u.expand(1, 1, 8, 16), cos, sin)[0, 0]
    for offset in (1, 2, 3):
        products = [float(rotated[i] @ rotated[i + offset]) for i in range(8 - offset)]
        assert max(products) - min(products) < 1e-4, f"offset {offset} is not translation-invariant"


def test_embedding_gates_value_z_off_when_the_bin_is_zero() -> None:
    torch.manual_seed(0)
    embed = EventEmbedding(VOCAB, 32, n_freq=8).eval()
    code = torch.zeros(1, 4, dtype=torch.long) + 5
    bins = torch.tensor([[0, 0, 3, 3]])
    age = torch.zeros(1, 4)
    delta = torch.zeros(1, 4)
    left = embed(code, bins, torch.zeros(1, 4), age, delta)
    right = embed(code, bins, torch.tensor([[2.5, -1.0, 2.5, -1.0]]), age, delta)
    # value_bin == 0 positions must be untouched by value_z ...
    assert torch.allclose(left[0, :2], right[0, :2], atol=1e-6)
    # ... and value_bin != 0 positions must not be.
    assert not torch.allclose(left[0, 2:], right[0, 2:], atol=1e-4)


# --------------------------------------------------------------------------- #
# Encoder / predictor / EHRJEPA
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mlp", ["swiglu", "gelu"])
def test_forward_shapes(mlp: str) -> None:
    torch.manual_seed(0)
    model = EHRJEPA(_config(mlp=mlp)).eval()
    batch = _batch()
    context, target = sample_masks(
        batch["attention_mask"], generator=torch.Generator().manual_seed(1)
    )
    out = model(batch, context, target)
    n_targets = int(target.sum())
    assert out.predictions.shape == (n_targets, 32)
    assert out.targets.shape == (n_targets, 32)
    assert out.cls.shape == (3, 32)
    assert out.context_tokens.shape == (3, 40, 32)
    assert not out.targets.requires_grad, "targets must be stop-gradient"


def test_encoder_output_is_invariant_to_masked_out_positions() -> None:
    """A position outside the context mask cannot influence any output."""
    torch.manual_seed(0)
    model = EHRJEPA(_config()).eval()
    batch = _batch()
    context = batch["attention_mask"].bool().clone()
    context[:, 20:] = False
    a = model.encoder(model.embed_batch(batch), context)
    perturbed = dict(batch)
    perturbed["code_id"] = batch["code_id"].clone()
    perturbed["code_id"][:, 20:] = (perturbed["code_id"][:, 20:] + 17) % VOCAB
    b = model.encoder(model.embed_batch(perturbed), context)
    assert torch.allclose(a.cls, b.cls, atol=1e-6)
    assert torch.allclose(a.tokens[:, :20], b.tokens[:, :20], atol=1e-6)


def test_predictor_never_sees_target_codes_or_values() -> None:
    """The information-leak assertion.

    Perturb every code, value bin and z-score at the *target* positions and check
    that the predictions are unchanged. If any of it reached the predictor -- via
    the mask tokens or via a context representation that had attended to a target
    -- the prediction task would be partly a copy.
    """
    torch.manual_seed(0)
    model = EHRJEPA(_config()).eval()
    batch = _batch()
    context, target = sample_masks(
        batch["attention_mask"], generator=torch.Generator().manual_seed(2)
    )
    assert int(target.sum()) > 0

    baseline = model(batch, context, target).predictions

    perturbed = {k: v.clone() for k, v in batch.items()}
    perturbed["code_id"][target] = (perturbed["code_id"][target] + 31) % VOCAB
    perturbed["value_bin"][target] = (perturbed["value_bin"][target] + 5) % 11
    perturbed["value_z"][target] = perturbed["value_z"][target] + 3.0
    after = model(perturbed, context, target).predictions

    assert torch.equal(baseline, after), "target code/value information reached the predictor"
    # The time features at the targets *are* allowed through, so changing them
    # must change the prediction -- otherwise the test above is vacuous.
    timed = {k: v.clone() for k, v in batch.items()}
    timed["age"][target] = timed["age"][target] + 11.0
    assert not torch.allclose(baseline, model(timed, context, target).predictions)


def test_padding_does_not_change_the_subject_embedding() -> None:
    torch.manual_seed(0)
    model = EHRJEPA(_config()).eval()
    batch = _batch(batch=1, length=24)
    short = {k: v[:, :16] for k, v in batch.items()}
    padded = {k: v.clone() for k, v in batch.items()}
    padded["attention_mask"][:, 16:] = 0
    assert torch.allclose(model.encode(short), model.encode(padded), atol=1e-5)


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #


def test_ema_momentum_schedule_is_linear_from_0996_to_1() -> None:
    assert ema_momentum(0, 100) == pytest.approx(0.996)
    assert ema_momentum(50, 100) == pytest.approx(0.998)
    assert ema_momentum(100, 100) == pytest.approx(1.0)
    assert ema_momentum(500, 100) == pytest.approx(1.0)  # clamped
    assert ema_momentum(0, 0) == pytest.approx(1.0)


def test_ema_update_math() -> None:
    torch.manual_seed(0)
    model = EHRJEPA(_config(target_mode="ema"))
    assert model.uses_ema
    for p in model.target_encoder.parameters():
        assert not p.requires_grad

    before = copy.deepcopy([p.detach().clone() for p in model.target_encoder.parameters()])
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(torch.randn_like(p))
    online = [p.detach().clone() for p in model.encoder.parameters()]

    m = 0.99
    model.update_ema(m)
    for old, new, live in zip(before, model.target_encoder.parameters(), online, strict=True):
        assert torch.allclose(new, m * old + (1 - m) * live, atol=1e-6)

    # momentum 1.0 freezes the target network entirely.
    frozen = [p.detach().clone() for p in model.target_encoder.parameters()]
    model.update_ema(1.0)
    for old, new in zip(frozen, model.target_encoder.parameters(), strict=True):
        assert torch.equal(old, new)


def test_shared_target_mode_has_no_second_encoder() -> None:
    model = EHRJEPA(_config(target_mode="shared"))
    assert model.target_encoder is None and model.target_embed is None
    assert not model.uses_ema
    model.update_ema(0.99)  # a no-op, not an error


def test_unknown_target_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="target_mode"):
        _config(target_mode="teacher")
