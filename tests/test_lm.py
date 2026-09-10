"""``model.encoder: lm``: the serialisation, the span offsets, causality, and a run.

The whole arm rests on one property that no loss curve would show as wrong: the
representation of event ``i`` must be a function of events ``0..i`` and nothing
later. With a from-scratch encoder that is an attention mask; here it is the
*arithmetic of span offsets* -- get the gather one token late and event ``i``'s
row reads the first token of event ``i + 1``, which is a leak that would make
this arm win for the wrong reason. So:

``test_the_event_representation_does_not_depend_on_later_events``
    Perturb every event after ``i`` -- its code and its value, so its text and its
    token count both change -- and require the rows at and before ``i`` to be
    bit-identical.
``test_span_ends_are_the_last_token_of_each_event``
    Decode ``input_ids[: span_end[i] + 1]`` and require it to be exactly the
    first ``i + 1`` events' spans, separator included.

The rest pins the serialisation (a number only where there is one, the raw value
rather than the z-score, the gap as text), the left truncation (oldest events go
first, and what is dropped is masked out rather than fed a garbage row), and that
the trainer and :mod:`ehrjepa.eval.probe` both drive the stack end to end.

**No download.** The base model is a two-layer GPT-2 built from a config in the
test, and it is injected by replacing :func:`ehrjepa.models.lm.load_base_model`.
The *tokenizer* is the real GPT-2 one, because a tokenizer is what makes "one
event is a handful of subword tokens" true and a fabricated one would test
arithmetic against itself; the whole module skips if it cannot be loaded from a
local cache.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ehrjepa.data.tokenize import PAD_ID
from ehrjepa.eval import probe
from ehrjepa.models import lm
from ehrjepa.models.jepa import EHRJEPAConfig
from ehrjepa.models.latent import EHRNextLatent
from ehrjepa.train.config import load_config
from ehrjepa.train.pretrain import Trainer

REPO = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = REPO / "configs" / "pretrain_debug.yaml"
ICU_CACHE = REPO / "data" / "cache" / "physionet2019"

requires_cache = pytest.mark.skipif(
    not (ICU_CACHE / "meta.json").exists(), reason="physionet2019 cache is not built"
)

pytest.importorskip("transformers", reason="the `lm` extra is not installed")
pytest.importorskip("peft", reason="the `lm` extra is not installed")

VOCAB = 47  # physionet2019
DIM = 32
LM_HIDDEN = 48


def _tokenizer_available() -> bool:
    from transformers import AutoTokenizer

    try:
        AutoTokenizer.from_pretrained("gpt2")
    except Exception:  # pragma: no cover - offline machine with no HF cache
        return False
    return True


requires_tokenizer = pytest.mark.skipif(
    not _tokenizer_available(), reason="the gpt2 tokenizer is not available locally"
)


@pytest.fixture(autouse=True)
def tiny_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """A two-layer randomly-initialised GPT-2 in place of the 0.5B download.

    Every dropout is zeroed: the real configuration has none either (the trunk is
    frozen and ``lora_dropout`` is 0), and the shared-target path depends on the
    forward pass being a deterministic function of the token ids.
    """
    from transformers.models.gpt2 import GPT2Config, GPT2Model

    def build(config: EHRJEPAConfig) -> GPT2Model:
        return GPT2Model(
            GPT2Config(
                vocab_size=50257,
                n_positions=1024,
                n_embd=LM_HIDDEN,
                n_layer=2,
                n_head=2,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
        )

    monkeypatch.setattr(lm, "load_base_model", build)


def _config(**overrides) -> EHRJEPAConfig:
    values = dict(
        vocab_size=VOCAB,
        dim=DIM,
        causal=True,
        encoder="lm",
        lm_cache_dir=str(ICU_CACHE),
        lm_tokenizer="gpt2",
        lm_lora_targets="c_attn",
        lm_lora_r=4,
        lm_max_tokens=1024,
        lm_grad_checkpointing=False,
        build_predictor=False,
        horizons=[1],
    )
    values.update(overrides)
    return EHRJEPAConfig(**values)


def _batch(length: int = 8, lengths: tuple[int, ...] = (8, 5)) -> dict[str, torch.Tensor]:
    """Two windows of ICU events, the second short so padding is exercised."""
    g = torch.Generator().manual_seed(0)
    batch = len(lengths)
    code_id = torch.randint(4, VOCAB, (batch, length), generator=g)
    value_bin = torch.randint(1, 11, (batch, length), generator=g)
    value_bin[:, 2] = 0  # one value-less event per row
    attention = torch.zeros(batch, length, dtype=torch.long)
    for row, n in enumerate(lengths):
        attention[row, :n] = 1
        code_id[row, n:] = PAD_ID
        value_bin[row, n:] = 0
    return {
        "code_id": code_id,
        "value_bin": value_bin,
        "value_z": torch.randn(batch, length, generator=g),
        "age": torch.rand(batch, length, generator=g) * 80,
        "log_delta": torch.log1p(torch.rand(batch, length, generator=g) * 3),
        "attention_mask": attention,
    }


def _tokenize(config: EHRJEPAConfig, batch: dict[str, torch.Tensor]) -> lm.LMTokens:
    stage = lm.LMEventTokenizer(config)
    return stage(
        batch["code_id"],
        batch["value_bin"],
        batch["value_z"],
        batch["age"],
        batch["log_delta"],
    )


# --------------------------------------------------------------------------- #
# The serialisation


@requires_cache
@requires_tokenizer
def test_an_event_serialises_to_its_words_its_number_and_its_gap() -> None:
    phrases = lm.code_phrases(ICU_CACHE)
    assert phrases[5] == "heart rate"  # VAR//HR
    assert phrases[46] == "onset of sepsis"  # SEPSIS_ONSET
    spans = lm.event_texts(
        phrases,
        code_id=_np([[5, 46]]),
        value_bin=_np([[7, 0]]),
        raw_value=_np([[92.4321, 0.0]], float),
        delta_hours=_np([[1.0, 0.5]], float),
        valid=_np([[True, True]], bool),
    )
    assert spans[0][0] == "heart rate 92.4 (+1h); "
    # value_bin 0 means "no measurement": the stored 0.0 must not be printed.
    assert spans[0][1] == "onset of sepsis (+0.5h); "


@requires_cache
@requires_tokenizer
def test_the_number_printed_is_the_raw_value_not_the_z_score() -> None:
    """``value_z`` is inverted through the cache's own per-code mean and std."""
    config = _config()
    stage = lm.LMEventTokenizer(config)
    hr = 5  # VAR//HR
    mean, std = float(stage.value_mean[hr]), float(stage.value_std[hr])
    assert not bool(stage.value_log1p[hr])
    raw = stage.raw_values(torch.tensor([[hr]]), torch.tensor([[1.5]]))
    assert float(raw) == pytest.approx(mean + 1.5 * std, rel=1e-5)
    # ICU_HOUR is one of the log1p-ed codes, so the inverse is expm1.
    assert bool(stage.value_log1p[4])
    raw = stage.raw_values(torch.tensor([[4]]), torch.tensor([[0.0]]))
    assert float(raw) == pytest.approx(float(torch.expm1(stage.value_mean[4])), rel=1e-5)


@requires_cache
@requires_tokenizer
def test_dropping_the_time_features_drops_the_gap_from_the_text() -> None:
    config = _config()
    stage = lm.LMEventTokenizer(config)
    batch = _batch()
    with_time = stage(
        batch["code_id"],
        batch["value_bin"],
        batch["value_z"],
        batch["age"],
        batch["log_delta"],
        use_time=True,
    )
    without = stage(
        batch["code_id"],
        batch["value_bin"],
        batch["value_z"],
        batch["age"],
        batch["log_delta"],
        use_time=False,
    )
    assert int(without.attn.sum()) < int(with_time.attn.sum())
    text = stage.tokenizer.decode(without.input_ids[0][: int(without.attn[0].sum())])
    assert "h)" not in text


# --------------------------------------------------------------------------- #
# Span offsets and gathering


@requires_cache
@requires_tokenizer
def test_span_ends_are_the_last_token_of_each_event() -> None:
    config = _config()
    stage = lm.LMEventTokenizer(config)
    batch = _batch()
    tokens = stage(
        batch["code_id"],
        batch["value_bin"],
        batch["value_z"],
        batch["age"],
        batch["log_delta"],
    )
    raw = stage.raw_values(batch["code_id"], batch["value_z"]).float().numpy()
    spans = lm.event_texts(
        stage.phrases,
        batch["code_id"].numpy(),
        batch["value_bin"].numpy(),
        raw,
        torch.expm1(batch["log_delta"]).numpy(),
        (batch["code_id"] != PAD_ID).numpy(),
    )
    for row in range(batch["code_id"].shape[0]):
        for col in range(int(batch["attention_mask"][row].sum())):
            end = int(tokens.span_end[row, col])
            decoded = stage.tokenizer.decode(tokens.input_ids[row][: end + 1])
            assert decoded == "".join(spans[row][: col + 1])
            # The last token of a span is the separator, by construction.
            assert decoded.endswith(lm.EVENT_SEPARATOR)


@requires_cache
@requires_tokenizer
def test_padded_events_are_masked_and_their_rows_are_zero() -> None:
    config = _config()
    tokens = _tokenize(config, _batch())
    encoder = lm.LMEncoder(config).eval()
    with torch.no_grad():
        rows = encoder(tokens).tokens
    assert rows.shape == (2, 8, DIM)
    assert not tokens.kept[1, 5:].any()
    assert float(rows[1, 5:].abs().sum()) == 0.0
    assert float(rows[1, :5].abs().sum()) > 0.0


@requires_cache
@requires_tokenizer
def test_the_event_representation_does_not_depend_on_later_events() -> None:
    """The load-bearing test: no leak from the future through the span gather."""
    config = _config()
    encoder = lm.LMEncoder(config).eval()
    batch = _batch(lengths=(8,))
    cut = 4
    with torch.no_grad():
        before = encoder(_tokenize(config, batch)).tokens
        poisoned = {k: v.clone() for k, v in batch.items()}
        poisoned["code_id"][0, cut + 1 :] = 20
        poisoned["value_z"][0, cut + 1 :] += 4.0
        poisoned["value_bin"][0, cut + 1 :] = 9
        after = encoder(_tokenize(config, poisoned)).tokens
    assert torch.equal(before[0, : cut + 1], after[0, : cut + 1])
    assert not torch.allclose(before[0, cut + 1 :], after[0, cut + 1 :])


# --------------------------------------------------------------------------- #
# The token cap


@requires_cache
@requires_tokenizer
def test_the_token_cap_drops_the_oldest_events_and_keeps_the_newest() -> None:
    batch = _batch(lengths=(8,))
    stage = lm.LMEventTokenizer(_config(lm_max_tokens=30))
    tokens = stage(
        batch["code_id"],
        batch["value_bin"],
        batch["value_z"],
        batch["age"],
        batch["log_delta"],
    )
    kept = tokens.kept[0]
    assert not bool(kept.all()), "the cap did not bite; the test proves nothing"
    assert bool(kept[-1]), "the newest event must never be the one dropped"
    # A contiguous suffix: once an event is kept, every later one is too.
    first_kept = int(kept.float().argmax())
    assert bool(kept[first_kept:].all())
    assert int(tokens.attn.sum()) <= 30
    assert stage.last_kept_frac == pytest.approx(float(kept.float().mean()))


# --------------------------------------------------------------------------- #
# Parameters, training and probing


@requires_cache
@requires_tokenizer
def test_only_the_adapters_the_projection_and_the_code_table_are_trainable() -> None:
    model = EHRNextLatent(_config(recon_head=True, value_head=True))
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert not any("wte" in name or "wpe" in name for name in trainable)
    assert any("lora_" in name for name in trainable)
    assert "encoder.proj.weight" in trainable
    assert "embed.code_emb.weight" in trainable
    total = sum(p.numel() for p in model.parameters())
    assert model.encoder.n_trainable() < 0.05 * total


@requires_cache
@requires_tokenizer
def test_three_steps_of_hybrid_training_with_an_lm_encoder(tmp_path: Path) -> None:
    overrides = {
        "data.cache_dir": str(ICU_CACHE),
        "data.max_len": 24,
        "data.min_len": 8,
        "model.encoder": "lm",
        "model.causal": "true",
        "model.lm_tokenizer": "gpt2",
        "model.lm_lora_targets": "c_attn",
        "model.lm_lora_r": 4,
        "model.lm_max_tokens": 512,
        "model.lm_grad_checkpointing": "true",
        "objective.kind": "nextlatent",
        "objective.horizons": "[1]",
        "objective.lambda_recon": 0.1,
        "objective.recon_value": "true",
        "objective.lambda_value": 1.0,
        "objective.lambda_sigreg": 0.0,
        "run.steps": 3,
        "run.batch_size": 2,
        "run.out_dir": str(tmp_path),
        "run.tensorboard": "false",
    }
    trainer = Trainer(load_config(DEBUG_CONFIG, [f"{k}={v}" for k, v in overrides.items()]))
    final = trainer.train()
    assert trainer.step == 3
    assert final["loss"] > 0
    assert final["value_loss"] > 0
    assert final["lm_events_kept"] == pytest.approx(1.0)
    header = (tmp_path / "metrics.csv").read_text().splitlines()[0]
    assert header.endswith("value_loss,lm_events_kept")
    assert (tmp_path / "final.pt").exists()


@requires_cache
@requires_tokenizer
def test_the_probe_rebuilds_the_lm_encoder_from_its_checkpoint(tmp_path: Path) -> None:
    """``load_encoder`` must reconstruct the stack from the config alone.

    The tokenizer id and the cache the descriptions come from are recorded in
    the checkpoint's ``model_config``, which is the only reason a probe run on
    another machine can embed anything at all.
    """
    overrides = {
        "data.cache_dir": str(ICU_CACHE),
        "data.max_len": 24,
        "data.min_len": 8,
        "model.encoder": "lm",
        "model.causal": "true",
        "model.lm_tokenizer": "gpt2",
        "model.lm_lora_targets": "c_attn",
        "model.lm_lora_r": 4,
        "model.lm_grad_checkpointing": "false",
        "objective.kind": "nextlatent",
        "objective.horizons": "[1]",
        "objective.lambda_sigreg": 0.0,
        "run.steps": 1,
        "run.batch_size": 2,
        "run.out_dir": str(tmp_path),
        "run.tensorboard": "false",
    }
    trainer = Trainer(load_config(DEBUG_CONFIG, [f"{k}={v}" for k, v in overrides.items()]))
    trainer.train()

    assert probe.checkpoint_is_causal(tmp_path / "final.pt")
    assert probe.default_features(True) == "last"
    model, max_len = probe.load_encoder(tmp_path / "final.pt")
    assert max_len == 24
    assert isinstance(model.encoder, lm.LMEncoder)
    assert model.embed.tokenizer.name_or_path == "gpt2"

    # Exactly what ehrjepa.eval.probe.embed does with the loaded model.
    batch = _batch(lengths=(8, 5))
    with torch.no_grad():
        tokens = model.embed_batch(batch)
        encoded = model.encoder(tokens, batch["attention_mask"])
        rows = torch.cat(
            probe._pool(encoded.tokens, encoded.cls, batch["attention_mask"], "last"), dim=-1
        )
    assert rows.shape == (2, DIM)
    assert torch.isfinite(rows).all()


# --------------------------------------------------------------------------- #
# What the LM stack refuses


@requires_cache
def test_the_lm_encoder_rejects_the_knobs_it_cannot_honour() -> None:
    with pytest.raises(ValueError, match="lm_cache_dir"):
        EHRJEPAConfig(vocab_size=VOCAB, encoder="lm")
    with pytest.raises(ValueError, match="target_mode"):
        _config(target_mode="ema")
    with pytest.raises(ValueError, match="span_only"):
        _config(target_span_only=True)
    with pytest.raises(ValueError, match="time encoders"):
        _config(share_time_encoders=True)
    with pytest.raises(ValueError, match="encoder must be one of"):
        EHRJEPAConfig(vocab_size=VOCAB, encoder="qwen")


@requires_cache
def test_masked_span_jepa_is_refused_at_config_load() -> None:
    from ehrjepa.train.config import from_mapping

    config = from_mapping(
        {
            "data": {"cache_dir": str(ICU_CACHE)},
            "model": {"encoder": "lm", "lm_tokenizer": "gpt2"},
            "objective": {"kind": "jepa"},
        }
    )
    with pytest.raises(ValueError, match="cannot run objective.kind: jepa"):
        config.model_config(VOCAB)


def _np(values, dtype=int):
    import numpy as np

    return np.asarray(values, dtype=dtype)
